from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ora.orchestrator.control_audit import audit_libero_run
from viewer.registry import register_run
from viewer.trajectory import compile_trajectory


BOUNDS = {"x": [-0.6, 0.6], "y": [-0.5, 0.5], "z": [-0.1, 1.3]}


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _run(tmp_path: Path, *, bounds: dict | None = None) -> Path:
    root = tmp_path / "runs" / "audit-run"
    _write(root / "plan.json", {
        "runtime": {"kind": "libero_pro", "workspace_bounds_m": bounds or BOUNDS},
        "source": {"commit": "1" * 40},
    })
    _write(root / "result.json", {"official_success": True})
    return root


def _v2_receipt(root: Path, *, status: str = "healthy") -> None:
    tracking = {
        "schema": "ora.libero-pose-tracking-evidence.v1",
        "action_id": "action-0", "action": "move_pose",
        "policy_visibility": "control_only",
        "compiled_pose_sha256": "2" * 64,
        "samples": [],
        "guard": {"outcome": "not_triggered"},
    }
    tracking_path = root / "native/tracking/action-0000.json"
    _write(tracking_path, tracking)
    digest = hashlib.sha256(tracking_path.read_bytes()).hexdigest()
    _write(root / "native/receipts/action-0000.json", {
        "schema": "ora.libero-pro-motion-receipt.v2",
        "action_id": "action-0",
        "action": "move_pose",
        "pose_contract_sha256": "3" * 64,
        "compiled_pose_sha256": "2" * 64,
        "control_health": {
            "schema": "ora.libero-control-health.v2",
            "status": status,
            "codes": [] if status == "healthy" else ["tracking_terminal_fault"],
        },
        "tracking_evidence": {
            "path": "tracking/action-0000.json",
            "sha256": digest,
        },
    })


def test_control_audit_accepts_only_complete_v2_evidence(tmp_path: Path) -> None:
    root = _run(tmp_path)
    _v2_receipt(root)
    record = audit_libero_run(root)
    assert record["official_success"] is True
    assert record["control_validity"] == "control-valid"
    assert record["issues"] == []
    registered = register_run(root)
    assert registered.official_success is True
    assert registered.control_validity == "control-valid"
    assert registered.control_issues == ()
    trajectory = compile_trajectory(root)
    assert trajectory["summary"]["official_success"] is True
    assert trajectory["summary"]["control_validity"] == "control-valid"


def test_control_audit_preserves_verdict_but_marks_fault_contaminated(
    tmp_path: Path,
) -> None:
    root = _run(tmp_path)
    _v2_receipt(root, status="fault")
    record = audit_libero_run(root)
    assert record["official_success"] is True
    assert record["control_validity"] == "contaminated"
    assert "recorded_control_health_fault" in record["issues"]


def test_control_audit_marks_unprovable_legacy_receipt_indeterminate(
    tmp_path: Path,
) -> None:
    root = _run(tmp_path)
    _write(root / "native/receipts/action-0000.json", {
        "schema": "ora.libero-pro-motion-receipt.v1",
        "before_pose_xyz_xyzw": [0, 0, 0.5, 0, 0, 0, 1],
        "commanded_pose_xyz_xyzw": [0.02, 0, 0.5, 0, 0, 0, 1],
        "reached_pose_xyz_xyzw": [0.02, 0, 0.5, 0, 0, 0, 1],
    })
    record = audit_libero_run(root)
    assert record["control_validity"] == "indeterminate"
    assert "legacy_frame_contract_not_provable" in record["issues"]


def test_control_audit_detects_legacy_cross_axis_contamination(
    tmp_path: Path,
) -> None:
    root = _run(tmp_path)
    _write(root / "native/receipts/action-0000.json", {
        "schema": "ora.libero-pro-motion-receipt.v1",
        "before_pose_xyz_xyzw": [0, 0, 0.5, 0, 0, 0, 1],
        "commanded_pose_xyz_xyzw": [0.02, 0, 0.5, 0, 0, 0, 1],
        "reached_pose_xyz_xyzw": [0.02, 0.08, 0.5, 0, 0, 0, 1],
    })
    record = audit_libero_run(root)
    assert record["control_validity"] == "contaminated"
    assert "legacy_cross_axis_drift_over_0_02m" in record["issues"]


def test_control_audit_flags_legacy_narrow_workspace(tmp_path: Path) -> None:
    root = _run(tmp_path, bounds={
        "x": [-0.3, 0.3], "y": [-0.3, 0.3], "z": [-0.1, 1.3],
    })
    record = audit_libero_run(root)
    assert record["official_success"] is True
    assert record["control_validity"] == "contaminated"
    assert "nonstandard_or_legacy_action_workspace" in record["issues"]


def _hybrid_receipt(root: Path, *, gripper_action: bool = True) -> tuple[Path, dict]:
    """Aperture change and commanded dimensions reproduce the G receipt case."""
    _v2_receipt(root)
    path = root / "native/receipts/action-0000.json"
    record = json.loads(path.read_text())
    pose = [0.05, -0.13, 1.09, 0.0, 0.0, 0.0, 1.0]
    record.update({
        "schema": "ora.libero-pro-hybrid-motion-receipt.v2",
        "action_id": "action-gripper-or-pose",
        "action": "set_gripper" if gripper_action else "servo_delta",
        "consumed": True, "state_changed": True,
        "motion_status": "not_reached",
        "steps_used": 80, "sim_steps_total": 265,
        "before_pose_xyz_xyzw": pose,
        "commanded_pose_xyz_xyzw": pose,
        "gripper_evidence": {"measured_before": 0.988, "measured_after": 0.227},
        "command_invariant": ({
            "schema": "ora.command-invariant.v2",
            "controlled_dimension": "gripper", "held_dimension": "eef_pose_target",
            "target_from_measured_before": pose, "verified": True,
        } if gripper_action else {
            "schema": "ora.command-invariant.v2",
            "controlled_dimension": "pose", "held_dimension": "gripper_actuator_target",
            "directive_before": -1.0, "directive_after": -1.0,
            "action_dimension_command": 0.0, "verified": True,
        }),
    })
    tracking_path = root / "native/tracking/action-0000.json"
    tracking = json.loads(tracking_path.read_text())
    tracking.update(action_id=record["action_id"], action=record["action"])
    _write(tracking_path, tracking)
    record["tracking_evidence"]["sha256"] = hashlib.sha256(tracking_path.read_bytes()).hexdigest()
    _write(path, record)
    return path, record


@pytest.mark.parametrize("gripper_action", [True, False])
def test_modern_audit_does_not_treat_aperture_change_as_command_pollution(
    tmp_path: Path, gripper_action: bool,
) -> None:
    root = _run(tmp_path)
    path, _ = _hybrid_receipt(root, gripper_action=gripper_action)
    original = path.read_bytes()
    result = audit_libero_run(root)
    assert result["schema"] == "ora.libero-control-validity-audit.v2"
    assert result["control_validity"] == "control-valid"
    assert result["issues"] == []
    assert result["receipt_schema_counts"]["hybrid"] == 1
    assert result["receipt_schema_counts"]["legacy"] == 0
    assert result["official_success"] is True
    assert path.read_bytes() == original


@pytest.mark.parametrize("violation", ["dimension", "pose_target", "gripper_directive"])
def test_modern_audit_still_detects_real_command_dimension_violations(
    tmp_path: Path, violation: str,
) -> None:
    root = _run(tmp_path)
    path, record = _hybrid_receipt(root, gripper_action=violation != "gripper_directive")
    if violation == "dimension":
        record["command_invariant"]["controlled_dimension"] = "pose"
    elif violation == "pose_target":
        record["commanded_pose_xyz_xyzw"] = [0.15, -0.13, 1.09, 0, 0, 0, 1]
    else:
        record["command_invariant"]["directive_after"] = 1.0
    _write(path, record)
    result = audit_libero_run(root)
    assert result["control_validity"] == "contaminated"
    assert result["issues"] == ["recorded_command_invariant_violation"]
    assert result["official_success"] is True


def test_sparse_audit_validates_the_embedded_segment_once(tmp_path: Path) -> None:
    root = _run(tmp_path)
    path, segment = _hybrid_receipt(root, gripper_action=False)
    record = {key: segment[key] for key in (
        "action_id", "consumed", "state_changed", "steps_used", "sim_steps_total",
        "control_health",
    )}
    record.update({
        "schema": "ora.libero-sparse-motion-receipt.v1",
        "action": "move_pose", "segment": segment, "execution_status": "not_reached",
    })
    _write(path, record)
    result = audit_libero_run(root)
    assert result["control_validity"] == "control-valid"
    assert result["receipt_count"] == 1
    assert result["receipt_schema_counts"]["sparse"] == 1
    record["steps_used"] += 1
    _write(path, record)
    result = audit_libero_run(root)
    assert result["control_validity"] == "indeterminate"
    assert "sparse_segment_contract_mismatch" in result["issues"]


def test_modern_missing_invariant_or_broken_artifact_is_unproven_not_legacy(
    tmp_path: Path,
) -> None:
    root = _run(tmp_path)
    path, record = _hybrid_receipt(root)
    record.pop("command_invariant")
    record["tracking_evidence"]["sha256"] = "0" * 64
    _write(path, record)
    result = audit_libero_run(root)
    assert result["control_validity"] == "indeterminate"
    assert "tracking_evidence_hash_mismatch" in result["issues"]
    assert "modern_command_invariant_missing_or_invalid" in result["issues"]
    assert not any(issue.startswith("legacy_") for issue in result["issues"])


@pytest.mark.parametrize("route", ["direct", "reference"])
def test_audit_accepts_production_global_receipts_using_actual_joint_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: str,
) -> None:
    from test_libero_correctness_repair import _action, _probe, _session

    root = _run(tmp_path)
    session, plant = _session(root / "native", monkeypatch)
    if route == "reference":
        session._execute_plan_reference(_action(plan_ref=_probe(session)))
    else:
        session._execute_planned_motion(_action())
    assert plant.commands
    result = audit_libero_run(root)
    assert result["control_validity"] == "control-valid", result["issues"]
    assert result["receipt_count"] == 1


@pytest.mark.parametrize("corruption", [
    "command_dimension", "wrong_action", "summary", "planner_summary",
])
def test_global_audit_checks_hash_bound_content_and_action_joins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str,
) -> None:
    from test_libero_correctness_repair import _action, _session

    root = _run(tmp_path)
    session, _plant = _session(root / "native", monkeypatch)
    session._execute_planned_motion(_action())
    receipt_path = root / "native/receipts/action-0000.json"
    receipt = json.loads(receipt_path.read_text())
    execution_path = root / "native" / receipt["execution"]["canonical_artifact"]["path"]
    execution = json.loads(execution_path.read_text())
    if corruption == "command_dimension":
        execution["commands"][0][7] = 1.0
    elif corruption == "wrong_action":
        execution["execution_id"] = "0:0:other-action"
    elif corruption == "summary":
        receipt["execution"]["joint_positions"][0] += 0.1
    else:
        receipt["planner"]["status"] = "not_reachable"
    _write(execution_path, execution)
    receipt["execution"]["canonical_artifact"]["sha256"] = hashlib.sha256(execution_path.read_bytes()).hexdigest()
    _write(receipt_path, receipt)
    result = audit_libero_run(root)
    assert result["control_validity"] == ("contaminated" if corruption == "command_dimension" else "indeterminate")
    assert result["official_success"] is True


@pytest.mark.parametrize("corruption", [
    "absolute", "traversal", "external_symlink", "ambiguous", "action_join", "unreadable",
])
def test_audit_rejects_unsafe_ambiguous_or_wrong_action_tracking_evidence(
    tmp_path: Path, corruption: str,
) -> None:
    root = _run(tmp_path)
    receipt_path, receipt = _hybrid_receipt(root)
    path = root / "native/tracking/action-0000.json"
    tracking = receipt["tracking_evidence"]
    if corruption == "absolute":
        tracking["path"] = str(path)
    elif corruption == "traversal":
        tracking["path"] = "native/../native/tracking/action-0000.json"
    elif corruption == "external_symlink":
        outside = tmp_path / "outside-tracking.json"
        outside.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(outside)
    elif corruption == "ambiguous":
        duplicate = root / "tracking/action-0000.json"
        duplicate.parent.mkdir()
        duplicate.write_bytes(path.read_bytes())
    elif corruption == "action_join":
        record = json.loads(path.read_text())
        record["action_id"] = "another-action"
        _write(path, record)
        tracking["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    else:
        path.write_bytes(b"\xff")
        tracking["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    _write(receipt_path, receipt)
    result = audit_libero_run(root)
    assert result["control_validity"] == "indeterminate", result["issues"]
    assert result["official_success"] is True
