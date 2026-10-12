"""Complete current public camera delivery, shared by all policy runtimes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from .action_views import PostActionObservationError, _safe_file

FULL_VISUAL_SCHEMA = "ora.current-visual-observation.v1"


def full_visual_enabled(view_config: Mapping[str, Any] | None) -> bool:
    return bool(view_config and view_config.get("current_visual_delivery") == "full_rgb_v1")


def _read(root: Path, ref: Mapping[str, Any]) -> Path:
    path = _safe_file(root, ref["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != ref.get("sha256"):
        raise PostActionObservationError("current visual artifact hash mismatch")
    return path


def camera_geometry(root: Path, camera: Mapping[str, Any]) -> dict[str, Any]:
    """Publish only calibrated sensor geometry; never arbitrary artifact fields."""
    raw = json.loads(_read(root, camera["files"]["calibration"]).read_text())
    intrinsic = np.asarray(raw.get("intrinsic_cv"), dtype=float)
    extrinsic = np.asarray(raw.get("extrinsic_cv"), dtype=float)
    if extrinsic.shape == (3, 4):
        extrinsic = np.vstack((extrinsic, [0.0, 0.0, 0.0, 1.0]))
    if (intrinsic.shape != (3, 3) or extrinsic.shape != (4, 4)
            or not np.isfinite(intrinsic).all() or not np.isfinite(extrinsic).all()
            or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0
            or not np.allclose(extrinsic[3], [0, 0, 0, 1], atol=1e-8)):
        raise PostActionObservationError("invalid current camera calibration")
    try:
        camera_to_world = np.linalg.inv(extrinsic)
    except np.linalg.LinAlgError as exc:
        raise PostActionObservationError("singular current camera calibration") from exc
    return {"intrinsic_cv": intrinsic.tolist(), "extrinsic_cv": extrinsic.tolist(),
        "cam2world_cv": camera_to_world.tolist(),
        "camera_origin_world_m": camera_to_world[:3, 3].tolist(),
        "camera_axes_world": {name: camera_to_world[:3, index].tolist()
            for index, name in enumerate(("right", "down", "forward"))},
        "depth_unit": "metres",
        "pixel_coordinates": "origin top-left; u right, v down",
        "camera_coordinates": "OpenCV: x right, y down, z forward",
        "extrinsic_convention": "world_to_camera; cam2world_cv is its inverse",
        "depth_semantics": "optical-axis z distance in metres"}


def materialize_current_visual(
    *, run_root: Path, workspace: Path, source_root: Path,
    cameras: Mapping[str, Any], policy_state: Mapping[str, Any],
    observation_id: str, observation_sha256: str | None,
    required_views: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Deliver every declared public RGB camera and its current calibration."""
    run_root, workspace, source_root = run_root.resolve(), workspace.resolve(), source_root.resolve()
    if not workspace.is_relative_to(run_root) or not source_root.is_relative_to(run_root):
        raise PostActionObservationError("current visual evidence escapes run")
    if not cameras:
        raise PostActionObservationError("current visual observation lacks cameras")
    if set(required_views) - set(cameras):
        raise PostActionObservationError("declared current camera is unavailable")
    if required_views and set(cameras) - set(required_views):
        raise PostActionObservationError("current observation contains an undeclared camera")
    identity = hashlib.sha256(json.dumps({"cameras": cameras, "state": dict(policy_state)},
        sort_keys=True).encode()).hexdigest()
    destination = workspace / "inputs/current-visual" / identity
    destination.mkdir(parents=True, exist_ok=True)
    images, geometry = [], {}
    for index, (name, camera) in enumerate(sorted(cameras.items())):
        files = camera["files"]
        rgb = _read(source_root, files.get("rgb_lossless_png") or files["rgb"])
        geometry[name] = camera_geometry(source_root, camera)
        with Image.open(rgb) as source_image:
            width, height = source_image.size
        target = destination / f"{index:02d}-rgb.png"
        data = rgb.read_bytes()
        if target.exists() and target.read_bytes() != data:
            raise PostActionObservationError("immutable current visual artifact drift")
        if not target.exists():
            target.write_bytes(data)
            target.chmod(0o444)
        images.append({"camera": name, "modality": "rgb",
            "path": target.relative_to(workspace).as_posix(),
            "run_path": target.relative_to(run_root).as_posix(),
            "sha256": hashlib.sha256(data).hexdigest(), "width": width, "height": height,
            "derivation": None})
    return {"schema": FULL_VISUAL_SCHEMA, "observation_id": observation_id,
        "observation_sha256": observation_sha256, "observation_step": policy_state["step"],
        "state_sha256": policy_state.get("state_sha256"), "images": images,
        "camera_geometry": geometry, "depth_delivery": "on_demand"}


def visual_feedback(record: Mapping[str, Any]) -> dict[str, Any]:
    return {"camera_geometry": record["camera_geometry"],
        "views": [image["camera"] for image in record["images"]],
        "delivery": "complete_current_rgb_and_calibration", "depth_delivery": "on_demand", "observation_step": record["observation_step"]}
