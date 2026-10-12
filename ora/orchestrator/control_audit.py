"""Read-only control-validity classification for historical LIBERO runs."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


CONTROL_AUDIT_SCHEMA = "ora.libero-control-validity-audit.v2"
CONTROL_VALIDITY_VALUES = {
    "control-valid", "contaminated", "indeterminate", "not-applicable",
}
_V2_RECEIPT = "ora.libero-pro-motion-receipt.v2"
_V2_HEALTH = "ora.libero-control-health.v2"
_TRACKING_SCHEMA = "ora.libero-pose-tracking-evidence.v1"
_HYBRID_RECEIPT = "ora.libero-pro-hybrid-motion-receipt.v2"
_SPARSE_RECEIPT = "ora.libero-sparse-motion-receipt.v1"
_STANDARD_LIBERO_BOUNDS = {
    "x": [-0.60, 0.60], "y": [-0.50, 0.50], "z": [-0.10, 1.30],
}


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _official_success(root: Path) -> bool | None:
    for name in ("result.json", "status.json", "evaluation.json"):
        value = _load(root / name).get("official_success")
        if isinstance(value, bool):
            return value
    return None


def _receipt_records(root: Path) -> Iterable[tuple[Path, Mapping[str, Any]]]:
    for path in sorted(root.glob("**/receipts/action-*.json")):
        if path.is_file():
            yield path, _load(path) if _contained_regular_file(path, root) else {}


def _contained_regular_file(path: Path, root: Path) -> bool:
    try:
        if not path.is_relative_to(root):
            return False
        if any(part.is_symlink() for part in (path, *path.parents) if part.is_relative_to(root)):
            return False
        return path.resolve().is_relative_to(root) and path.is_file()
    except (OSError, RuntimeError, ValueError):
        return False


def _bound_artifact_path(
    root: Path, bases: Iterable[Path], reference: Any, *, label: str,
) -> tuple[Path | None, list[str]]:
    """Resolve exactly one immutable relative reference within this run."""
    if not isinstance(reference, Mapping):
        return None, [f"{label}_evidence_missing_or_invalid"]
    relative, digest = reference.get("path"), reference.get("sha256")
    if (
        not isinstance(relative, str) or not relative
        or Path(relative).is_absolute() or ".." in Path(relative).parts
        or not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
    ):
        return None, [f"{label}_evidence_path_or_hash_invalid"]
    candidates = {base / relative for base in bases if _contained_regular_file(base / relative, root)}
    if len(candidates) != 1:
        return None, [f"{label}_evidence_ambiguous" if candidates else f"{label}_evidence_unavailable"]
    path = candidates.pop()
    try:
        if _sha256(path) != digest:
            return None, [f"{label}_evidence_hash_mismatch"]
    except OSError:
        return None, [f"{label}_evidence_unavailable"]
    return path, []


def _artifact_record(
    root: Path, receipt_path: Path, reference: Any, *, label: str,
) -> tuple[Path | None, dict[str, Any], list[str]]:
    path, issues = _bound_artifact_path(
        root, (root, receipt_path.parent.parent), reference, label=label,
    )
    record = _load(path) if path is not None else {}
    if path is not None and not record:
        issues.append(f"{label}_evidence_invalid")
    return path, record, issues


def _quaternion_angle(left: Any, right: Any) -> float | None:
    if not (
        isinstance(left, list) and len(left) == 4
        and isinstance(right, list) and len(right) == 4
    ):
        return None
    try:
        left_f = [float(item) for item in left]
        right_f = [float(item) for item in right]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in [*left_f, *right_f]):
        return None
    left_norm = math.sqrt(sum(item * item for item in left_f))
    right_norm = math.sqrt(sum(item * item for item in right_f))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return None
    dot = abs(sum(a * b for a, b in zip(left_f, right_f))) / (
        left_norm * right_norm
    )
    return 2.0 * math.acos(min(1.0, max(0.0, dot)))


def _legacy_motion_anomalies(receipt: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    before = receipt.get("before_pose_xyz_xyzw")
    target = receipt.get("commanded_pose_xyz_xyzw")
    reached = receipt.get("reached_pose_xyz_xyzw")
    if all(isinstance(value, list) and len(value) == 7 for value in (
        before, target, reached,
    )):
        try:
            requested = [float(target[index]) - float(before[index]) for index in range(3)]
            realized = [float(reached[index]) - float(before[index]) for index in range(3)]
        except (TypeError, ValueError):
            issues.append("legacy_pose_evidence_malformed")
        else:
            request_distance = math.sqrt(sum(item * item for item in requested))
            realized_distance = math.sqrt(sum(item * item for item in realized))
            if request_distance > 1e-9:
                direction = [item / request_distance for item in requested]
                along = sum(a * b for a, b in zip(realized, direction))
                cross = math.sqrt(sum(
                    (realized[index] - direction[index] * along) ** 2
                    for index in range(3)
                ))
            else:
                cross = realized_distance
            if cross > 0.02 + 1e-6:
                issues.append("legacy_cross_axis_drift_over_0_02m")
            if realized_distance - request_distance > 0.02 + 1e-6:
                issues.append("legacy_travel_overshoot_over_0_02m")
            requested_rotation = _quaternion_angle(before[3:], target[3:])
            realized_rotation = _quaternion_angle(before[3:], reached[3:])
            if (
                requested_rotation is not None and realized_rotation is not None
                and realized_rotation - requested_rotation > 0.18 + 1e-6
            ):
                issues.append("legacy_orientation_overshoot_over_0_18rad")
    health = receipt.get("control_health")
    if isinstance(health, Mapping) and health.get("status") == "fault":
        issues.append("recorded_control_health_fault")
    gripper = receipt.get("gripper_evidence")
    if isinstance(gripper, Mapping):
        before_gripper = gripper.get("measured_before")
        after_gripper = gripper.get("measured_after")
        if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in (
            before_gripper, after_gripper,
        )) and abs(float(after_gripper) - float(before_gripper)) > 0.03 + 1e-6:
            invariant = receipt.get("command_invariant")
            if not (
                isinstance(invariant, Mapping)
                and invariant.get("held_dimension") == "gripper_actuator_target"
            ):
                issues.append("legacy_measured_aperture_preservation_claim")
    return issues


def _validate_v2_receipt(
    root: Path, receipt_path: Path, receipt: Mapping[str, Any],
) -> list[str]:
    issues: list[str] = []
    health = receipt.get("control_health")
    if not isinstance(health, Mapping) or health.get("schema") != _V2_HEALTH or health.get("status") not in {"healthy", "degraded", "fault"}:
        issues.append("v2_control_health_missing_or_invalid")
    elif health.get("status") == "fault":
        issues.append("recorded_control_health_fault")
    for field in ("pose_contract_sha256", "compiled_pose_sha256"):
        if receipt.get("action") == "move_pose" and not (
            isinstance(receipt.get(field), str) and len(receipt[field]) == 64
        ):
            issues.append(f"{field}_missing_or_invalid")
    tracking = receipt.get("tracking_evidence")
    if not isinstance(tracking, Mapping):
        issues.append("tracking_evidence_missing")
        return issues
    _path, record, binding_issues = _artifact_record(root, receipt_path, tracking, label="tracking")
    issues.extend(binding_issues)
    if binding_issues:
        return issues
    if (
        record.get("schema") != _TRACKING_SCHEMA
        or record.get("policy_visibility") != "control_only"
        or not isinstance(receipt.get("action_id"), str)
        or record.get("action_id") != receipt["action_id"]
        or record.get("action") != receipt.get("action")
        or record.get("compiled_pose_sha256")
        != receipt.get("compiled_pose_sha256")
    ):
        issues.append("tracking_evidence_contract_mismatch")
    return issues


def _command_invariant_issues(receipt: Mapping[str, Any]) -> list[str]:
    """Audit commanded dimensions, never infer a fault from contact motion."""

    invariant = receipt.get("command_invariant")
    if not isinstance(invariant, Mapping) or invariant.get("schema") != (
        "ora.command-invariant.v2"
    ):
        return ["modern_command_invariant_missing_or_invalid"]
    action = receipt.get("action")
    if action == "set_gripper":
        expected = ("gripper", "eef_pose_target")
    elif action in {"move_pose", "move_free_space", "servo_delta", "recover_motion"}:
        expected = ("pose", "gripper_actuator_target")
    else:
        return ["modern_action_dimension_invalid"]
    if not all(isinstance(invariant.get(key), str) for key in ("controlled_dimension", "held_dimension")):
        return ["modern_command_invariant_evidence_invalid"]
    if (invariant["controlled_dimension"], invariant["held_dimension"]) != expected:
        return ["recorded_command_invariant_violation"]
    if expected[0] == "pose":
        before, after = invariant.get("directive_before"), invariant.get("directive_after")
        command = invariant.get("action_dimension_command")
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) for value in (before, after, command)
        ):
            return ["modern_command_invariant_evidence_invalid"]
        if before != after or command != 0.0:
            return ["recorded_command_invariant_violation"]
    else:
        before = receipt.get("before_pose_xyz_xyzw")
        commanded = receipt.get("commanded_pose_xyz_xyzw")
        recorded_target = invariant.get("target_from_measured_before")
        if not all(
            isinstance(pose, list) and len(pose) == 7
            and all(isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value) for value in pose)
            for pose in (before, commanded, recorded_target)
        ):
            return ["modern_command_invariant_evidence_invalid"]
        for pose in (commanded, recorded_target):
            angle = _quaternion_angle(before[3:], pose[3:])
            if angle is None:
                return ["modern_command_invariant_evidence_invalid"]
            if max(abs(a - b) for a, b in zip(before[:3], pose[:3])) > 1e-9 or angle > 1e-7:
                return ["recorded_command_invariant_violation"]
    return [] if invariant.get("verified") is True else ["modern_command_invariant_evidence_invalid"]


def _modern_execution_count_issues(receipt: Mapping[str, Any]) -> list[str]:
    steps = receipt.get("steps_used")
    return [] if (
        type(steps) is int and steps >= 0
        and receipt.get("state_changed") is (steps > 0)
        and receipt.get("consumed") is True
    ) else ["modern_execution_count_invalid"]


def _global_planner_issues(
    root: Path, receipt_path: Path, planner: Mapping[str, Any],
) -> list[str]:
    path, record, issues = _artifact_record(
        root, receipt_path, planner.get("canonical_artifact", planner.get("artifact")), label="planner",
    )
    if issues:
        return issues
    if planner.get("source") == "motion_probe_plan_ref":
        binding = record.get("binding")
        if (
            record.get("schema") != "ora.motion-plan-artifact.v1"
            or record.get("backend_id") != "libero_pro"
            or record.get("embodiment_id") != "franka_panda"
            or not isinstance(binding, Mapping)
            or binding.get("schema") not in {
                "ora.libero-plan-reference.v1", "ora.libero-plan-reference.v2",
            }
        ):
            return ["planner_evidence_contract_mismatch"]
        expected_ref = "plan-" + hashlib.sha256(json.dumps(
            binding, sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest()[:32]
        if planner.get("plan_ref") != expected_ref:
            issues.append("planner_evidence_contract_mismatch")
        _array_path, array_issues = _bound_artifact_path(
            root, (path.parent,), record.get("arrays"), label="planner_arrays",
        )
        issues.extend(array_issues)
    else:
        if path.name != receipt_path.name:
            issues.append("planner_evidence_action_mismatch")
        if record.get("status") not in {"planned", "not_reachable", "planner_error"} or any(
            planner.get(field) != record.get(field)
            for field in ("status", "reachable", "planner_status")
        ):
            issues.append("planner_evidence_contract_mismatch")
        if record.get("status") == "planner_error":
            issues.append("recorded_planner_error")
    return issues


def _global_execution_issues(
    root: Path, receipt_path: Path, receipt: Mapping[str, Any],
) -> list[str]:
    execution = receipt.get("execution")
    path, record, issues = _artifact_record(
        root, receipt_path,
        execution.get("canonical_artifact") if isinstance(execution, Mapping) else None,
        label="execution",
    )
    if issues:
        return issues
    if path.name != "joint-" + receipt_path.name:
        issues.append("execution_evidence_action_mismatch")
    execution_id = record.get("execution_id")
    if execution_id is not None:
        identity = str(execution_id).split(":", 2)
        sequence = receipt_path.stem.removeprefix("action-")
        if len(identity) != 3 or not identity[1].isdigit() or not sequence.isdigit() or int(identity[1]) != int(sequence) or identity[2] != receipt.get("action_id"):
            issues.append("execution_evidence_action_mismatch")
    if record.get("controller") != "JOINT_POSITION" or record.get("steps_used") != receipt.get("steps_used") or any(
        key != "canonical_artifact" and value != record.get(key)
        for key, value in execution.items()
    ):
        issues.append("execution_evidence_contract_mismatch")
    commands = record.get("commands")
    if (
        not isinstance(commands, list) or len(commands) != receipt.get("steps_used")
        or any(not isinstance(row, list) or len(row) != 8 or any(
            type(value) not in {int, float} or not math.isfinite(value) or abs(value) > 1.0
            for value in row
        ) for row in commands)
    ):
        issues.append("modern_command_invariant_evidence_invalid")
    elif any(row[7] != 0.0 for row in commands):
        issues.append("recorded_command_invariant_violation")
    # The production global writer proves the held gripper dimension with
    # these actual JOINT_POSITION commands, not an OSC-only invariant object.
    if receipt.get("action") not in {"move_pose", "move_free_space", "recover_motion"}:
        issues.append("modern_action_dimension_invalid")
    return issues


def _validate_hybrid_receipt(
    root: Path, receipt_path: Path, receipt: Mapping[str, Any],
) -> list[str]:
    # OSC and gripper segments retain the existing hash-bound tracking
    # contract. Their measured aperture may change during either action;
    # only a contradictory command invariant establishes dimension misuse.
    issues = _modern_execution_count_issues(receipt)
    if isinstance(receipt.get("tracking_evidence"), Mapping):
        return issues + _validate_v2_receipt(root, receipt_path, receipt) + _command_invariant_issues(receipt)

    health = receipt.get("control_health")
    if not isinstance(health, Mapping) or health.get("schema") != _V2_HEALTH or health.get("status") not in {"healthy", "degraded", "fault"}:
        issues.append("modern_control_health_missing_or_invalid")
    elif health.get("status") == "fault":
        issues.append("recorded_control_health_fault")
    steps = receipt.get("steps_used")
    planner = receipt.get("planner")
    if isinstance(planner, Mapping):
        issues.extend(_global_planner_issues(root, receipt_path, planner))
    elif steps != 0 or receipt.get("motion_status") != "refused":
        issues.append("planner_evidence_missing_or_invalid")
    if steps == 0 and receipt.get("state_changed") is False:
        # A zero-write refusal cannot violate a command dimension. It must
        # still retain the available planner/error evidence above.
        return issues
    issues.extend(_global_execution_issues(root, receipt_path, receipt))
    if "command_invariant" in receipt:
        issues.extend(_command_invariant_issues(receipt))
    return issues


def _validate_wait_receipt(receipt: Mapping[str, Any]) -> list[str]:
    """The shared hold: its own invariant and the recorded health, no segment."""
    from ora.environment.controllers.libero_wait import libero_wait_receipt_issues

    issues = libero_wait_receipt_issues(receipt)
    health = receipt.get("control_health")
    if not isinstance(health, Mapping) or health.get("schema") != _V2_HEALTH or health.get("status") not in {"healthy", "degraded", "fault"}:
        issues.append("modern_control_health_missing_or_invalid")
    elif health.get("status") == "fault":
        issues.append("recorded_control_health_fault")
    return issues


def _validate_sparse_receipt(
    root: Path, receipt_path: Path, receipt: Mapping[str, Any],
) -> list[str]:
    if receipt.get("action") == "wait":
        return _validate_wait_receipt(receipt)
    segment = receipt.get("segment")
    if not isinstance(segment, Mapping) or segment.get("schema") != _HYBRID_RECEIPT:
        return ["sparse_segment_missing_or_invalid"]
    issues = []
    for field in ("action_id", "consumed", "state_changed", "steps_used", "sim_steps_total"):
        if field not in receipt or receipt[field] != segment.get(field):
            issues.append("sparse_segment_contract_mismatch")
            break
    if (
        receipt.get("action") != "move_pose"
        or segment.get("action") not in {"move_pose", "move_free_space", "servo_delta"}
        or receipt.get("execution_status") not in {
            "reached": {"reached"}, "not_reached": {"not_reached"},
            "stopped": {"not_reached"},
            "refused": {"plan_rejected", "planner_failed", "execution_fault"},
        }.get(segment.get("motion_status"), set())
    ):
        issues.append("sparse_segment_contract_mismatch")
    for field in ("control_health", "before", "after"):
        if receipt.get(field) != segment.get(field):
            issues.append("sparse_segment_contract_mismatch")
    issues.extend(_validate_hybrid_receipt(root, receipt_path, segment))
    return issues


def audit_libero_run(root: Path) -> dict[str, Any]:
    """Classify one run without modifying its evidence or official verdict."""

    root = root.expanduser().resolve()
    plan = _load(root / "plan.json")
    runtime = plan.get("runtime")
    runtime = runtime if isinstance(runtime, Mapping) else {}
    backend = runtime.get("kind")
    official = _official_success(root)
    if backend != "libero_pro":
        return {
            "schema": CONTROL_AUDIT_SCHEMA,
            "run_id": root.name,
            "backend": backend,
            "official_success": official,
            "control_validity": "not-applicable",
            "issues": [],
            "receipt_count": 0,
        }
    issues: list[str] = []
    receipt_count = 0
    v2_count = 0
    hybrid_count = 0
    sparse_count = 0
    legacy_count = 0
    for path, receipt in _receipt_records(root):
        receipt_count += 1
        if not receipt:
            issues.append("motion_receipt_unreadable")
            continue
        if receipt.get("schema") == _V2_RECEIPT:
            v2_count += 1
            issues.extend(_validate_v2_receipt(root, path, receipt))
        elif receipt.get("schema") == _HYBRID_RECEIPT:
            hybrid_count += 1
            issues.extend(_validate_hybrid_receipt(root, path, receipt))
        elif receipt.get("schema") == _SPARSE_RECEIPT:
            sparse_count += 1
            issues.extend(_validate_sparse_receipt(root, path, receipt))
        else:
            legacy_count += 1
            issues.extend(_legacy_motion_anomalies(receipt))
    workspace = runtime.get("workspace_bounds_m")
    if isinstance(workspace, Mapping) and dict(workspace) != _STANDARD_LIBERO_BOUNDS:
        issues.append("nonstandard_or_legacy_action_workspace")
    issues = sorted(set(issues))
    explicit_contamination = any(
        item.startswith("legacy_")
        or item in {
            "recorded_control_health_fault",
            "recorded_command_invariant_violation",
            "nonstandard_or_legacy_action_workspace",
        }
        for item in issues
    )
    integrity_problem = any(
        item.endswith(("invalid", "unavailable", "mismatch"))
        or item in {"tracking_evidence_missing", "motion_receipt_unreadable"}
        for item in issues
    )
    if explicit_contamination:
        validity = "contaminated"
    elif (
        receipt_count > 0 and v2_count + hybrid_count + sparse_count == receipt_count
        and not issues and not integrity_problem
    ):
        validity = "control-valid"
    else:
        validity = "indeterminate"
        if receipt_count == 0:
            issues.append("no_motion_receipts")
        if legacy_count:
            issues.extend([
                "legacy_frame_contract_not_provable",
                "legacy_rotation_tracking_not_provable",
                "legacy_gripper_hold_semantics_not_provable",
            ])
        issues = sorted(set(issues))
    assert validity in CONTROL_VALIDITY_VALUES
    return {
        "schema": CONTROL_AUDIT_SCHEMA,
        "run_id": root.name,
        "backend": "libero_pro",
        "source_commit": (
            plan.get("source", {}).get("commit")
            if isinstance(plan.get("source"), Mapping) else None
        ),
        "official_success": official,
        "control_validity": validity,
        "issues": issues,
        "receipt_count": receipt_count,
        "receipt_schema_counts": {
            "v2": v2_count,
            "hybrid": hybrid_count,
            "sparse": sparse_count,
            "legacy": legacy_count,
        },
        "verdict_semantics": (
            "official_success_is_preserved; control_validity independently "
            "describes whether the motion evidence supports behavioral use"
        ),
    }


__all__ = [
    "CONTROL_AUDIT_SCHEMA", "CONTROL_VALIDITY_VALUES", "audit_libero_run",
]
