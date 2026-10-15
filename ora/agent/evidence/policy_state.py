"""Immutable policy-state snapshots and context-call evidence."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


POLICY_STATE_SCHEMA = "ora.policy-state.v1"
STATE_MANIFEST_SCHEMA = "ora.policy-state-manifest.v1"
CONTEXT_BUNDLE_SCHEMA = "ora.context-bundle.v1"
REQUIRED_ABSENT = {
    "actor_state",
    "object_gt_pose",
    "segmentation",
    "contact",
    "success_predicate",
    "expert_program",
    "task_source",
}
SNAPSHOT_FIELDS = {
    "schema", "run_id", "step", "cause", "task_instruction", "observation",
    "cameras", "main_camera", "proprioception", "previous_public_action",
    "previous_execution_receipt", "remaining_budget",
    "runtime_notice",
    "forbidden_modalities_absent", "previous_state_sha256", "state_sha256",
}
FORBIDDEN_POLICY_KEY_FRAGMENTS = {
    "actor_state", "object_gt", "segmentation", "contact", "success_predicate",
    "official_success", "expert_program", "task_source",
}
# Match forbidden terms on word boundaries with plural forms, mirroring the
# trace scanner: the controller phase noun "precontact" embedded inside a
# key is public motion vocabulary, not contact-sensing evidence.
_FORBIDDEN_POLICY_KEY_PATTERNS = tuple(
    re.compile(rf"(?<![a-z0-9]){re.escape(fragment)}(?:s|es)?(?![a-z0-9])")
    for fragment in sorted(FORBIDDEN_POLICY_KEY_FRAGMENTS)
)
# Exact policy-authored motion-parameter names that legitimately contain a
# forbidden term on a word boundary.  `contact_approach` is the public
# staged-approach strategy argument the policy itself writes
# (`staged_contact_approach_v1`); it cannot leak state the policy never saw.
# Values under these keys are still scanned recursively.
PUBLIC_MOTION_VOCABULARY_KEYS = frozenset({"contact_approach"})


class PolicyStateError(RuntimeError):
    """Policy-state evidence is unavailable, inconsistent or tampered."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode()


def json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PolicyStateError(f"cannot read policy-state object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PolicyStateError(f"policy-state object must be a JSON object: {path}")
    return value


def _relative_file(run_root: Path, path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or not resolved.is_relative_to(run_root):
        raise PolicyStateError(f"policy-state artifact escapes the run: {path}")
    return {
        "path": resolved.relative_to(run_root).as_posix(),
        "sha256": file_sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def _with_identity(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    plain = dict(body)
    return {**plain, field: json_sha256(plain)}


def _check_identity(value: Mapping[str, Any], field: str) -> bool:
    basis = {key: child for key, child in value.items() if key != field}
    return isinstance(value.get(field), str) and json_sha256(basis) == value[field]


def _sanitize_embedded_receipt(value: Any, *, parent_key: str | None = None) -> Any:
    """Drop forbidden-fragment phase keys from an embedded execution receipt.

    The staged-approach RoboTwin receipt names its controller-tracking phase
    ledger entries ``global_precontact``/``local_contact``. Those key names
    trip the substring privilege scan even though the phase telemetry itself
    is public Supervisor output, which made every snapshot embedding such a
    receipt unreadable (EXP06A closing repair A1). Sanitize exactly the
    ``controller_tracking.phases`` ledger of the embedded copy: flagged phase
    keys are removed and listed under ``phases_redacted``. Every other key
    remains subject to the read-side privilege scan.
    """

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if (
                parent_key == "controller_tracking"
                and key == "phases"
                and isinstance(child, Mapping)
            ):
                kept: dict[str, Any] = {}
                redacted: list[str] = []
                for phase, telemetry in child.items():
                    lowered = str(phase).lower()
                    if any(
                        fragment in lowered
                        for fragment in FORBIDDEN_POLICY_KEY_FRAGMENTS
                    ):
                        redacted.append(str(phase))
                    else:
                        kept[str(phase)] = _sanitize_embedded_receipt(
                            telemetry, parent_key=str(key),
                        )
                result[str(key)] = kept
                if redacted:
                    result["phases_redacted"] = sorted(redacted)
            else:
                result[str(key)] = _sanitize_embedded_receipt(
                    child, parent_key=str(key),
                )
        return result
    if isinstance(value, list):
        return [
            _sanitize_embedded_receipt(item, parent_key=parent_key)
            for item in value
        ]
    return value


def _forbidden_policy_key(value: Any, *, path: str = "$") -> str | None:
    from .motion_feedback import is_public_contact_feedback_field

    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            aggregate_contact = (
                lowered in {"contact", "contact_estimated"}
                and (path.endswith(".feedback") or path.endswith(".motion_feedback"))
                and (isinstance(child, bool) or child is None)
            )
            staged_phase_label = (
                lowered == "local_contact" and path.endswith(".phases")
            )
            if (
                not is_public_contact_feedback_field(path, lowered, child)
                and not aggregate_contact
                and not staged_phase_label
                and lowered not in PUBLIC_MOTION_VOCABULARY_KEYS
                and any(
                    pattern.search(lowered)
                    for pattern in _FORBIDDEN_POLICY_KEY_PATTERNS
                )
            ):
                return f"{path}.{key}"
            if key != "forbidden_modalities_absent":
                found = _forbidden_policy_key(child, path=f"{path}.{key}")
                if found:
                    return found
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found = _forbidden_policy_key(child, path=f"{path}[{index}]")
            if found:
                return found
    return None


@dataclass
class PolicyStateStore:
    """The supervisor-owned writer for one run's public state timeline."""

    run_root: Path
    run_id: str
    plan_sha256: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    context_calls: list[dict[str, Any]] = field(default_factory=list)
    finalized: bool = False
    finalization: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.run_root = self.run_root.resolve()
        self.root = self.run_root / "policy-state"
        self.steps_root = self.root / "steps"
        self.context_root = self.run_root / "context-bundles"
        if (self.root / "manifest.json").exists():
            raise PolicyStateError("refusing to reuse an existing policy-state manifest")
        self.steps_root.mkdir(parents=True, exist_ok=True)
        self.context_root.mkdir(parents=True, exist_ok=True)
        self._write_manifest()

    def _write_manifest(self) -> dict[str, Any]:
        body = {
            "schema": STATE_MANIFEST_SCHEMA,
            "run_id": self.run_id,
            "plan_sha256": self.plan_sha256,
            "steps": list(self.steps),
            "context_calls": list(self.context_calls),
            "finalized": self.finalized,
            "finalization": self.finalization,
        }
        manifest = _with_identity(body, "manifest_sha256")
        atomic_json(self.root / "manifest.json", manifest)
        return manifest

    def publish(
        self,
        *,
        observation: Mapping[str, Any],
        observation_root: Path,
        cause: str,
        task_instruction: str,
        previous_public_action: Mapping[str, Any] | None,
        remaining_budget: Mapping[str, Any],
        runtime_notice: Mapping[str, Any] | None = None,
        previous_execution_receipt: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.finalized:
            raise PolicyStateError("policy-state timeline is finalized")
        if observation.get("schema") != "ora-observation.v1":
            raise PolicyStateError("policy-state source observation schema mismatch")
        absent = set(observation.get("forbidden_modalities_absent", []))
        if not REQUIRED_ABSENT.issubset(absent):
            raise PolicyStateError("policy-state source lacks privilege-boundary declaration")
        source = observation_root.resolve()
        if not source.is_dir() or not source.is_relative_to(self.run_root):
            raise PolicyStateError("policy-state observation root escapes the run")
        observation_manifest = _relative_file(self.run_root, source / "observation.json")
        persisted_observation = _load_object(source / "observation.json")
        if canonical_bytes(persisted_observation) != canonical_bytes(dict(observation)):
            raise PolicyStateError("published observation differs from its persisted manifest")
        cameras: dict[str, Any] = {}
        for name, record in sorted(dict(observation.get("cameras", {})).items()):
            if not isinstance(record, Mapping):
                raise PolicyStateError(f"invalid public camera record: {name}")
            files: dict[str, Any] = {}
            for role, reference in sorted(dict(record.get("files", {})).items()):
                if not isinstance(reference, Mapping) or not isinstance(reference.get("path"), str):
                    raise PolicyStateError(f"invalid public camera artifact: {name}/{role}")
                artifact = _relative_file(self.run_root, source / reference["path"])
                if artifact["sha256"] != reference.get("sha256"):
                    raise PolicyStateError(f"public camera artifact hash drift: {name}/{role}")
                files[role] = artifact
            cameras[str(name)] = {
                "width": int(record["width"]),
                "height": int(record["height"]),
                "mount": record.get("mount"),
                "calibration_dynamic": bool(record.get("calibration_dynamic", False)),
                "modalities": list(record.get("modalities", [])),
                "files": files,
            }
        step = len(self.steps)
        previous_hash = self.steps[-1]["state_sha256"] if self.steps else None
        body = {
            "schema": POLICY_STATE_SCHEMA,
            "run_id": self.run_id,
            "step": step,
            "cause": cause,
            "task_instruction": task_instruction.strip(),
            "observation": {
                "id": str(observation.get("observation_id", "")),
                "sequence": observation.get("sequence"),
                "manifest": observation_manifest,
            },
            "cameras": cameras,
            "main_camera": str(observation.get("main_camera", "head_camera")),
            "proprioception": observation.get("proprioception"),
            "previous_public_action": (
                None if previous_public_action is None else dict(previous_public_action)
            ),
            "previous_execution_receipt": _sanitize_embedded_receipt(
                dict(previous_execution_receipt)
                if previous_execution_receipt is not None
                else observation.get("previous_execution_receipt")
            ),
            "remaining_budget": dict(remaining_budget),
            "runtime_notice": None if runtime_notice is None else dict(runtime_notice),
            "forbidden_modalities_absent": sorted(absent),
            "previous_state_sha256": previous_hash,
        }
        snapshot = _with_identity(body, "state_sha256")
        relative = f"policy-state/steps/step-{step:04d}.json"
        target = self.run_root / relative
        if target.exists():
            raise PolicyStateError(f"refusing to overwrite policy-state step {step}")
        atomic_json(target, snapshot)
        target.chmod(0o444)
        record = {
            "step": step,
            "path": relative,
            "sha256": file_sha256(target),
            "state_sha256": snapshot["state_sha256"],
            "observation_id": snapshot["observation"]["id"],
        }
        self.steps.append(record)
        self._write_manifest()
        return snapshot

    def register_context_bundle(self, record: Mapping[str, Any]) -> None:
        if self.finalized:
            raise PolicyStateError("cannot register context after finalization")
        expected_call = len(self.context_calls)
        if record.get("call") != expected_call:
            raise PolicyStateError("context calls must be registered in order")
        self.context_calls.append(dict(record))
        self._write_manifest()

    def finalize(self, *, state: str, termination: str) -> None:
        if self.finalized:
            return
        self.finalized = True
        self.finalization = {"state": state, "termination": termination}
        self._write_manifest()


def context_bundle_images(bundle: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Read legacy singular or versioned action-bound plural attachments."""
    multiple = bundle.get("image_attachments")
    if multiple is None:
        image = bundle.get("image_attachment")
        return [image] if isinstance(image, Mapping) else []
    record = bundle.get("current_visual_observation") or bundle.get("post_action_observation")
    if (
        not isinstance(multiple, list) or not multiple
        or not all(isinstance(image, Mapping) for image in multiple)
        or bundle.get("image_attachment") is not None
        or not isinstance(record, Mapping)
        or record.get("schema") != "ora.current-visual-observation.v1"
    ):
        raise PolicyStateError("invalid versioned context image attachment inventory")
    return list(multiple)


def write_context_bundle(
    *,
    run_root: Path,
    call: int,
    source_state: Mapping[str, Any],
    message_kind: str,
    exact_user_message: str,
    image_path: Path | None,
    workspace_files: Sequence[Path],
    public_history: Sequence[str],
    tool_definitions: Sequence[Mapping[str, Any]],
    agent_session_id: str,
    agent_session_backend: str,
    thread_id_before: str | None,
    rollover_generation: int,
    rollover_policy: str,
    image_paths: Sequence[Path] = (),
    post_action_observation: Mapping[str, Any] | None = None,
    current_visual_observation: Mapping[str, Any] | None = None,
    skill_observations: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    run_root = run_root.resolve()
    state_path = Path(str(source_state["path"]))
    if state_path.is_absolute() or ".." in state_path.parts:
        raise PolicyStateError("context source state path must be run-relative")
    snapshot_path = (run_root / state_path).resolve()
    if not snapshot_path.is_file() or not snapshot_path.is_relative_to(run_root):
        raise PolicyStateError("context source state is unavailable")
    snapshot = _load_object(snapshot_path)
    image_attachment = None
    if image_path is not None and image_paths:
        raise PolicyStateError("context image input cannot mix legacy and multi-image delivery")
    image_attachments = []
    if image_paths:
        from ora.agent.context.action_views import resolve_post_action_images

        verified = resolve_post_action_images({
            "policy_state": source_state,
            "post_action_observation": post_action_observation,
            "current_visual_observation": current_visual_observation,
        }, root=run_root, path_key="run_path")
        visual_record = current_visual_observation or post_action_observation
        from ora.agent.context.skill_observations import resolve_skill_images

        historical = resolve_skill_images(
            {"skill_observations": list(skill_observations or ())}, root=run_root, path_key="run_path",
        )
        expected_paths = [item["path"].resolve() for item in verified] + list(historical)
        if expected_paths != [Path(path).resolve() for path in image_paths]:
            raise PolicyStateError("context image vector differs from action-bound fresh views")
        for item in verified:
            image_attachments.append({
                **_relative_file(run_root, item["path"]), "camera": item["camera"],
                "width": item["width"], "height": item["height"],
                "observation_id": visual_record["observation_id"],
                "observation_sha256": visual_record["observation_sha256"],
                "derivation": item.get("derivation"), "calibration": item.get("calibration"),
            })
        for item, path in zip(skill_observations or (), historical):
            image_attachments.append({
                **_relative_file(run_root, path), "camera": item["camera"],
                "width": item["width"], "height": item["height"],
                "observation_id": item["observation_id"],
                "observation_sha256": item["observation_sha256"],
                "role": "post_close_before_lift",
            })
    if image_path is not None:
        main_camera = snapshot.get("main_camera", "head_camera")
        source_image = (
            snapshot.get("cameras", {}).get(main_camera, {})
            .get("files", {}).get("rgb_lossless_png")
        )
        if not isinstance(source_image, Mapping):
            raise PolicyStateError("context source state lacks a main image")
        delivered = _relative_file(run_root, image_path)
        suffix = image_path.suffix.lower() or ".bin"
        immutable_image = run_root / f"context-bundles/image-call-{call:04d}{suffix}"
        if immutable_image.exists():
            raise PolicyStateError(f"refusing to overwrite context image {call}")
        immutable_image.parent.mkdir(parents=True, exist_ok=True)
        immutable_image.write_bytes(image_path.read_bytes())
        immutable_image.chmod(0o444)
        image_attachment = {
            **_relative_file(run_root, immutable_image),
            "delivered_path": delivered["path"],
            "source": dict(source_image),
            "derivation": {
                "operation": "fit_thumbnail",
                "maximum_size_px": [512, 512],
                "format": "PNG",
            },
        }
    workspace_evidence: list[dict[str, Any]] = []
    workspace_snapshot_root = run_root / f"context-bundles/workspace-call-{call:04d}"
    for index, path in enumerate(workspace_files):
        if not path.is_file():
            continue
        delivered = _relative_file(run_root, path)
        target = workspace_snapshot_root / f"{index:02d}-{path.name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        target.chmod(0o444)
        workspace_evidence.append({
            **_relative_file(run_root, target),
            "delivered_path": delivered["path"],
        })
    body = {
        "schema": CONTEXT_BUNDLE_SCHEMA,
        "call": call,
        "source_state": dict(source_state),
        "message": {
            "kind": message_kind,
            "exact_text": exact_user_message,
            "sha256": hashlib.sha256(exact_user_message.encode()).hexdigest(),
            "characters": len(exact_user_message),
            "utf8_bytes": len(exact_user_message.encode("utf-8")),
        },
        "image_attachment": image_attachment,
        "workspace_files": workspace_evidence,
        "public_history": list(public_history),
        "tool_surface": {
            "advertised_in_order": [dict(item) for item in tool_definitions],
            "schema_sha256": json_sha256([dict(item) for item in tool_definitions]),
        },
        "session": {
            "agent_session_id": agent_session_id,
            "backend": agent_session_backend,
            "thread_id_before": thread_id_before,
            "fresh_thread": thread_id_before is None,
            "rollover_generation": rollover_generation,
            "rollover_policy": rollover_policy,
        },
    }
    if image_attachments:
        body["image_attachments"] = image_attachments
        if skill_observations:
            body["skill_observations"] = [dict(item) for item in skill_observations]
        if post_action_observation is not None:
            body["post_action_observation"] = dict(post_action_observation)
        if current_visual_observation is not None:
            body["current_visual_observation"] = dict(current_visual_observation)
    bundle = _with_identity(body, "bundle_sha256")
    relative = f"context-bundles/call-{call:04d}.json"
    target = run_root / relative
    if target.exists():
        raise PolicyStateError(f"refusing to overwrite context bundle {call}")
    atomic_json(target, bundle)
    target.chmod(0o444)
    record = {
        "call": call,
        "path": relative,
        "sha256": file_sha256(target),
        "bundle_sha256": bundle["bundle_sha256"],
        "source_step": source_state["step"],
        "source_state_sha256": source_state["state_sha256"],
    }
    return bundle, record


def verify_policy_state(
    run_root: Path, *, verify_artifacts: bool = True,
) -> dict[str, Any]:
    """Validate the manifest, step chain, observation refs and context bundles."""

    run_root = run_root.resolve()
    errors: list[str] = []
    manifest_path = run_root / "policy-state/manifest.json"
    try:
        manifest = _load_object(manifest_path)
    except PolicyStateError as exc:
        return {"status": "invalid", "errors": [str(exc)], "manifest": {}}
    if manifest.get("schema") != STATE_MANIFEST_SCHEMA:
        errors.append("state_manifest_schema_mismatch")
    if not _check_identity(manifest, "manifest_sha256"):
        errors.append("state_manifest_identity_mismatch")
    previous: str | None = None
    loaded_steps: list[dict[str, Any]] = []
    seen_observations: set[str] = set()
    records = manifest.get("steps")
    if not isinstance(records, list):
        records = []
        errors.append("state_steps_not_a_list")
    indexed_step_paths = {
        str(item.get("path")) for item in records if isinstance(item, Mapping)
    }
    actual_step_paths = {
        path.relative_to(run_root).as_posix()
        for path in (run_root / "policy-state/steps").glob("step-*.json")
        if path.is_file()
    }
    if indexed_step_paths != actual_step_paths:
        errors.append("state_step_inventory_mismatch")
    for expected_step, record in enumerate(records):
        if not isinstance(record, Mapping) or record.get("step") != expected_step:
            errors.append("state_step_order_mismatch")
            continue
        relative = Path(str(record.get("path", "")))
        path = (run_root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(run_root) or not path.is_file():
            errors.append(f"missing_state_step:{expected_step}")
            continue
        if file_sha256(path) != record.get("sha256"):
            errors.append(f"state_step_file_hash_mismatch:{expected_step}")
        try:
            snapshot = _load_object(path)
        except PolicyStateError:
            errors.append(f"invalid_state_step:{expected_step}")
            continue
        loaded_steps.append(snapshot)
        if snapshot.get("schema") != POLICY_STATE_SCHEMA or snapshot.get("step") != expected_step:
            errors.append(f"state_step_schema_or_index_mismatch:{expected_step}")
        if frozenset(snapshot) != frozenset(SNAPSHOT_FIELDS):
            errors.append(f"state_step_fields_mismatch:{expected_step}")
        if snapshot.get("cause") not in {"reset", "after_action"}:
            errors.append(f"state_step_cause_mismatch:{expected_step}")
        if not REQUIRED_ABSENT.issubset(set(snapshot.get("forbidden_modalities_absent", []))):
            errors.append(f"state_privilege_boundary_mismatch:{expected_step}")
        forbidden_path = _forbidden_policy_key(snapshot)
        if forbidden_path:
            errors.append(f"state_forbidden_policy_key:{expected_step}:{forbidden_path}")
        if not _check_identity(snapshot, "state_sha256"):
            errors.append(f"state_step_identity_mismatch:{expected_step}")
        if snapshot.get("state_sha256") != record.get("state_sha256"):
            errors.append(f"state_step_record_identity_mismatch:{expected_step}")
        if snapshot.get("previous_state_sha256") != previous:
            errors.append(f"state_step_chain_mismatch:{expected_step}")
        previous = snapshot.get("state_sha256")
        observation = snapshot.get("observation", {})
        observation_id = observation.get("id") if isinstance(observation, Mapping) else None
        if not isinstance(observation_id, str) or not observation_id or observation_id in seen_observations:
            errors.append(f"state_observation_identity_mismatch:{expected_step}")
        else:
            seen_observations.add(observation_id)
        for reference in ([
            observation.get("manifest") if isinstance(observation, Mapping) else None,
            *[
                ref
                for camera in dict(snapshot.get("cameras", {})).values()
                if isinstance(camera, Mapping)
                for ref in dict(camera.get("files", {})).values()
            ],
        ] if verify_artifacts else []):
            if not isinstance(reference, Mapping):
                errors.append(f"state_artifact_reference_invalid:{expected_step}")
                continue
            relative_artifact = Path(str(reference.get("path", "")))
            artifact = (run_root / relative_artifact).resolve()
            if (
                relative_artifact.is_absolute()
                or not artifact.is_relative_to(run_root)
                or not artifact.is_file()
                or file_sha256(artifact) != reference.get("sha256")
            ):
                errors.append(f"state_artifact_hash_mismatch:{expected_step}")
    calls = manifest.get("context_calls")
    if not isinstance(calls, list):
        calls = []
        errors.append("context_calls_not_a_list")
    indexed_bundle_paths = {
        str(item.get("path")) for item in calls if isinstance(item, Mapping)
    }
    actual_bundle_paths = {
        path.relative_to(run_root).as_posix()
        for path in (run_root / "context-bundles").glob("call-*.json")
        if path.is_file()
    }
    if indexed_bundle_paths != actual_bundle_paths:
        errors.append("context_bundle_inventory_mismatch")
    state_by_hash = {item.get("state_sha256"): item for item in loaded_steps}
    for expected_call, record in enumerate(calls):
        if not isinstance(record, Mapping) or record.get("call") != expected_call:
            errors.append("context_call_order_mismatch")
            continue
        relative = Path(str(record.get("path", "")))
        path = (run_root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(run_root) or not path.is_file():
            errors.append(f"missing_context_bundle:{expected_call}")
            continue
        if file_sha256(path) != record.get("sha256"):
            errors.append(f"context_bundle_file_hash_mismatch:{expected_call}")
        try:
            bundle = _load_object(path)
        except PolicyStateError:
            errors.append(f"invalid_context_bundle:{expected_call}")
            continue
        if bundle.get("schema") != CONTEXT_BUNDLE_SCHEMA or bundle.get("call") != expected_call:
            errors.append(f"context_bundle_schema_or_index_mismatch:{expected_call}")
        if not _check_identity(bundle, "bundle_sha256"):
            errors.append(f"context_bundle_identity_mismatch:{expected_call}")
        if bundle.get("bundle_sha256") != record.get("bundle_sha256"):
            errors.append(f"context_bundle_record_identity_mismatch:{expected_call}")
        if record.get("source_state_sha256") not in state_by_hash:
            errors.append(f"context_bundle_unknown_source_state:{expected_call}")
        source_state = bundle.get("source_state")
        source = state_by_hash.get(record.get("source_state_sha256"))
        if (
            not isinstance(source_state, Mapping)
            or source is None
            or source_state.get("step") != source.get("step")
            or source_state.get("state_sha256") != source.get("state_sha256")
            or record.get("source_step") != source.get("step")
        ):
            errors.append(f"context_bundle_source_mismatch:{expected_call}")
        message = bundle.get("message", {})
        if (
            not isinstance(message, Mapping)
            or not isinstance(message.get("exact_text"), str)
            or hashlib.sha256(str(message.get("exact_text", "")).encode()).hexdigest()
            != message.get("sha256")
        ):
            errors.append(f"context_bundle_message_hash_mismatch:{expected_call}")
        tool_surface = bundle.get("tool_surface", {})
        if (
            not isinstance(tool_surface, Mapping)
            or json_sha256(tool_surface.get("advertised_in_order", []))
            != tool_surface.get("schema_sha256")
        ):
            errors.append(f"context_bundle_tool_hash_mismatch:{expected_call}")
        references = (
            [bundle.get("image_attachment"), *bundle.get("image_attachments", []),
             *bundle.get("workspace_files", [])]
            if verify_artifacts else []
        )
        for reference in references:
            if reference is None:
                continue
            if not isinstance(reference, Mapping):
                errors.append(f"context_bundle_artifact_invalid:{expected_call}")
                continue
            relative_artifact = Path(str(reference.get("path", "")))
            artifact = (run_root / relative_artifact).resolve()
            if (
                relative_artifact.is_absolute()
                or not artifact.is_relative_to(run_root)
                or not artifact.is_file()
                or file_sha256(artifact) != reference.get("sha256")
            ):
                errors.append(f"context_bundle_artifact_hash_mismatch:{expected_call}")
    if manifest.get("finalized") is True:
        finalization = manifest.get("finalization")
        if not isinstance(finalization, Mapping) or finalization.get("state") not in {
            "completed", "incomplete"
        }:
            errors.append("state_manifest_finalization_mismatch")
        if isinstance(finalization, Mapping) and finalization.get("state") == "completed" and not loaded_steps:
            errors.append("completed_state_timeline_is_empty")
    elif manifest.get("finalization") is not None:
        errors.append("unfinalized_state_has_finalization")
    return {
        "status": "valid" if not errors else "invalid",
        "errors": sorted(set(errors)),
        "manifest": manifest,
        "steps": loaded_steps,
    }


def load_policy_state(run_root: Path, step: int | None = None) -> dict[str, Any]:
    verification = verify_policy_state(run_root, verify_artifacts=False)
    if verification["status"] != "valid":
        raise PolicyStateError("invalid policy-state evidence: " + ", ".join(verification["errors"]))
    steps = verification["steps"]
    if not steps:
        raise PolicyStateError("policy-state timeline is empty")
    if step is None:
        return steps[-1]
    if isinstance(step, bool) or not isinstance(step, int) or step < 0 or step >= len(steps):
        raise PolicyStateError(f"unknown policy-state step: {step}")
    return steps[step]


__all__ = [
    "CONTEXT_BUNDLE_SCHEMA",
    "POLICY_STATE_SCHEMA",
    "STATE_MANIFEST_SCHEMA",
    "PolicyStateError",
    "PolicyStateStore",
    "file_sha256",
    "json_sha256",
    "load_policy_state",
    "verify_policy_state",
    "write_context_bundle",
]
