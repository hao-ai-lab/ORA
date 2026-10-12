"""Fresh Servo views shared by Native and managed policy input paths.

Camera roles come from the public descriptor. Missing wrists are disclosed,
never guessed from private sensors or replaced with historical pixels.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from .observation_projection import (
    AUXILIARY_CONTACT_SHEET_CAMERAS,
    AUXILIARY_CONTACT_SHEET_VIEW,
    ObservationProjectionError,
    materialize_auxiliary_contact_sheet,
)


POST_ACTION_OBSERVATION_SCHEMA = "ora.servo-post-action-observation.v1"
CURRENT_OBSERVATION_BUNDLE_SCHEMA = "ora.current-observation-bundle.v1"
SERVO_ACTION_SCHEMA = "ora-servo-action.v1"


class PostActionObservationError(RuntimeError):
    """Mandatory fresh action-end evidence cannot be delivered safely."""


def is_servo_action(action: Mapping[str, Any] | None) -> bool:
    return isinstance(action, Mapping) and action.get("schema") == SERVO_ACTION_SCHEMA


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_file(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    path = root / candidate
    if candidate.is_absolute() or ".." in candidate.parts:
        raise PostActionObservationError("image path escapes its evidence root")
    if any((root / Path(*candidate.parts[:index])).is_symlink()
           for index in range(1, len(candidate.parts) + 1)):
        raise PostActionObservationError("image path is symlinked")
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise PostActionObservationError("current image is unavailable")
    return path


def materialize_post_action_observation(
    *, run_root: Path, workspace: Path, observation: Mapping[str, Any],
    observation_root: Path, policy_state: Mapping[str, Any],
    previous_action: Mapping[str, Any] | None, view_config: Mapping[str, Any],
    camera_roles: Mapping[str, str],
) -> dict[str, Any]:
    """Freeze defaults plus selected public wrists from one new observation."""

    all_current = view_config.get("current_observation_delivery") == "all_public_v1"
    if not all_current and not is_servo_action(previous_action):
        raise PostActionObservationError("post-action projection requires a Servo action")
    previous_action = previous_action or {}
    run_root, workspace = run_root.resolve(), workspace.resolve()
    source = observation_root.resolve()
    if not workspace.is_relative_to(run_root) or not source.is_relative_to(run_root):
        raise PostActionObservationError("post-action evidence must remain in the run")
    manifest = _safe_file(source, "observation.json")
    if json.loads(manifest.read_text(encoding="utf-8")) != dict(observation):
        raise PostActionObservationError("current observation manifest differs from measurements")
    observation_id = observation.get("observation_id")
    if not observation_id or observation_id == previous_action.get("observation_id"):
        raise PostActionObservationError("action requires a fresh observation identity")
    if policy_state.get("observation_id", observation_id) != observation_id:
        raise PostActionObservationError("policy-state observation identity differs")
    arguments = previous_action.get("arguments", {})
    if not isinstance(arguments, Mapping):
        raise PostActionObservationError("Servo action lacks arguments")
    selected_arms = (arguments.get("arms") if arguments.get("mode") == "hold"
                     else arguments.get("targets"))
    if all_current and not is_servo_action(previous_action):
        selected_arms = []
    elif not isinstance(selected_arms, (list, tuple, Mapping)) or not selected_arms or set(selected_arms) - {"left", "right"}:
        raise PostActionObservationError("Servo action lacks selected arms")
    arms = [arm for arm in ("left", "right") if arm in selected_arms]
    defaults = list(view_config.get("automatic_actual_views", ()))
    if not defaults or not all(isinstance(name, str) and name for name in defaults):
        raise PostActionObservationError("Servo requires declared default public views")
    cameras = observation.get("cameras")
    if not isinstance(cameras, Mapping):
        raise PostActionObservationError("current observation lacks public cameras")
    selected = list(dict.fromkeys(defaults))
    missing = []
    wrist_arms = {}
    for arm in (() if all_current else arms):
        matches = sorted(name for name, role in camera_roles.items() if role == f"{arm}_wrist")
        if len(matches) > 1:
            raise PostActionObservationError(f"ambiguous public wrist role: {arm}")
        if not matches:
            missing.append({"arm": arm, "camera": None, "reason": "public_wrist_not_declared"})
            continue
        name = matches[0]
        wrist_arms[name] = arm
        if name not in selected:
            selected.append(name)
    manifest_hash = _hash(manifest)
    identity = hashlib.sha256(f"{observation_id}:{manifest_hash}".encode()).hexdigest()
    destination = workspace / "inputs/action-observations" / identity
    destination.mkdir(parents=True, exist_ok=True)

    def camera_source(name: str) -> tuple[Path, Mapping[str, Any]]:
        record = cameras.get(name)
        if not isinstance(record, Mapping):
            raise PostActionObservationError("current_camera_unavailable")
        files = record.get("files", {})
        rgb = files.get("rgb_lossless_png") or files.get("rgb")
        if not isinstance(rgb, Mapping) or not isinstance(rgb.get("path"), str):
            raise PostActionObservationError("current_rgb_unavailable")
        path = _safe_file(source, rgb["path"])
        if _hash(path) != rgb.get("sha256"):
            raise PostActionObservationError("current_rgb_hash_mismatch")
        return path, record

    images = []
    for name in selected:
        try:
            derivation = None
            calibration = None
            if name == AUXILIARY_CONTACT_SHEET_VIEW:
                ordered = tuple(view_config.get("auxiliary_contact_sheet_cameras") or AUXILIARY_CONTACT_SHEET_CAMERAS)
                paths = {camera: camera_source(camera)[0] for camera in ordered}
                derived = materialize_auxiliary_contact_sheet(
                    run_root=run_root, observation_step=int(policy_state["step"]),
                    camera_paths=paths, cameras=ordered,
                )
                path = derived["path"]
                derivation = {"operation": "auxiliary_contact_sheet", "cameras": list(ordered),
                    "manifest_path": derived["manifest_path"].relative_to(run_root).as_posix(),
                    "manifest_sha256": _hash(derived["manifest_path"])}
            else:
                path, camera = camera_source(name)
                calibration_ref = camera.get("files", {}).get("calibration")
                if isinstance(calibration_ref, Mapping):
                    calibration_path = _safe_file(source, calibration_ref["path"])
                    if _hash(calibration_path) != calibration_ref.get("sha256"):
                        raise PostActionObservationError("current_calibration_hash_mismatch")
                    calibration = {"path": calibration_path.relative_to(run_root).as_posix(),
                        "sha256": _hash(calibration_path)}
            with Image.open(path) as image:
                image.load()
                width, height = image.size
            resolution = view_config.get("rgb_resolution", "native")
            if name != AUXILIARY_CONTACT_SHEET_VIEW and resolution != "native" and [width, height] != list(resolution):
                raise PostActionObservationError("current_rgb_resolution_mismatch")
        except (OSError, ValueError, KeyError, PostActionObservationError, ObservationProjectionError) as exc:
            reason = str(exc) if isinstance(exc, PostActionObservationError) else "current_rgb_decode_or_read_failed"
            if name in defaults:
                raise PostActionObservationError(f"mandatory default fresh view unavailable: {name}: {reason}") from exc
            missing.append({"arm": wrist_arms[name], "camera": name, "reason": reason})
            continue
        frozen = destination / f"{len(images):02d}-{hashlib.sha256(name.encode()).hexdigest()[:12]}.png"
        data = path.read_bytes()
        if frozen.exists():
            if frozen.read_bytes() != data:
                raise PostActionObservationError("immutable post-action image drift")
        else:
            frozen.write_bytes(data)
            frozen.chmod(0o444)
        images.append({"camera": name, "role": camera_roles.get(name),
            "path": frozen.relative_to(workspace).as_posix(),
            "run_path": frozen.relative_to(run_root).as_posix(),
            "sha256": _hash(frozen), "width": width, "height": height,
            "calibration": calibration, "derivation": derivation})
    return {"schema": CURRENT_OBSERVATION_BUNDLE_SCHEMA if all_current else POST_ACTION_OBSERVATION_SCHEMA,
        "action_id": previous_action.get("action_id"), "mode": arguments.get("mode"),
        "selected_arms": arms, "observation_id": observation_id,
        "observation_sha256": manifest_hash, "observation_step": policy_state["step"],
        "state_sha256": policy_state.get("state_sha256"),
        "captured_at_unix_s": observation.get("captured_at_unix_s"),
        "observation_sequence": observation.get("sequence"),
        "status": "degraded" if missing else "complete",
        "missing_wrist_views": missing, "images": images}


def resolve_post_action_images(
    context: Mapping[str, Any], *, root: Path, path_key: str = "path",
) -> list[dict[str, Any]]:
    """Verify the set immediately before actual runtime delivery."""

    record = context.get("current_visual_observation") or context.get("post_action_observation")
    if record is None:
        return []
    from .current_visual import FULL_VISUAL_SCHEMA
    if not isinstance(record, Mapping) or record.get("schema") not in {
        POST_ACTION_OBSERVATION_SCHEMA, CURRENT_OBSERVATION_BUNDLE_SCHEMA, FULL_VISUAL_SCHEMA,
    }:
        raise PostActionObservationError("unsupported post-action observation schema")
    state = context.get("policy_state")
    if isinstance(state, Mapping) and (
        state.get("step") != record.get("observation_step")
        or state.get("state_sha256") != record.get("state_sha256")
    ):
        raise PostActionObservationError("post-action images differ from the current policy state")
    images = record.get("images")
    if not isinstance(images, (list, tuple)) or not images:
        raise PostActionObservationError("post-action observation has no fresh images")
    result = []
    seen = set()
    for image in images:
        if not isinstance(image, Mapping) or image.get("camera") in seen:
            raise PostActionObservationError("invalid or duplicate post-action image")
        seen.add(image.get("camera"))
        path = _safe_file(root, str(image.get(path_key, "")))
        if _hash(path) != image.get("sha256"):
            raise PostActionObservationError("post-action image hash mismatch")
        result.append({**dict(image), "path": path})
    return result
