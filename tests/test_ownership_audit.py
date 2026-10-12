"""Deterministic behavioral-ownership audit over synthetic run evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ora.orchestrator.ownership_audit import (
    OWNERSHIP_AUDIT_SCHEMA,
    OwnershipAuditError,
    build_ownership_audit,
    write_ownership_audit,
)


def _jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _synthetic_run(tmp_path: Path) -> Path:
    run_root = tmp_path / "run"
    (run_root / "native").mkdir(parents=True)
    (run_root / "plan.json").write_text(json.dumps({
        "policy": {
            "tool_surface": {
                "profile_id": "robotwin_pose_primitives_observer_dual_arm_v13",
                "mutation_tools": ["move_pose", "set_gripper"],
                "lifecycle_tools": ["finish_episode"],
            },
        },
    }), encoding="utf-8")
    (run_root / "native/agent-session.json").write_text(json.dumps({
        "calls": [{"turn_id": f"turn-{index}"} for index in range(7)],
    }), encoding="utf-8")
    _jsonl(run_root / "agent-events.jsonl", [
        {"type": "action_proposed", "model_call_id": "turn-1"},
        {"type": "action_proposed", "model_call_id": "turn-3"},
        {"type": "action_proposed", "model_call_id": "turn-5"},
    ])
    sparse_receipt = {
        "schema": "ora.robotwin-sparse-motion-receipt.v7",
        "execution_status": "reached",
        "arms": {
            "left": {"plan_validation": {"status": "accepted"}},
            "right": {"planner": {"called": False}},
        },
        "controller_tracking": {"sampled_steps": 42},
    }
    failed_receipt = {
        "schema": "ora.robotwin-sparse-motion-receipt.v7",
        "execution_status": "planner_failed",
        "arms": {"left": {"planner": {"called": True}}},
        "controller_tracking": {"sampled_steps": 0},
    }
    _jsonl(run_root / "events.jsonl", [
        {"type": "run_started"},
        {"type": "observation"},
        {"type": "tool_result", "tool": "observe"},
        {"type": "tool_result", "tool": "vision.ground"},
        {"type": "tool_result", "tool": "grasp.plan"},
        {"type": "tool_result", "tool": "motion.probe"},
        {"type": "tool_error", "tool": "move_pose"},
        {"type": "tool_result", "tool": "move_pose"},
        {"type": "action_submitted", "action": {
            "schema": "ora-action.v3", "action": "move_pose",
        }},
        {"type": "motion_receipt", "record": sparse_receipt},
        {"type": "observation"},
        {"type": "tool_result", "tool": "move_pose"},
        {"type": "action_submitted", "action": {
            "schema": "ora-action.v3", "action": "move_pose",
        }},
        {"type": "motion_receipt", "record": failed_receipt},
        {"type": "observation"},
        {"type": "observation"},
        {"type": "tool_result", "tool": "finish_episode"},
    ])
    return run_root


def test_ownership_audit_counts_selection_and_delegation(tmp_path: Path) -> None:
    run_root = _synthetic_run(tmp_path)
    report = write_ownership_audit(run_root)
    assert report["schema"] == OWNERSHIP_AUDIT_SCHEMA
    assert report["surface_profile_id"] == (
        "robotwin_pose_primitives_observer_dual_arm_v13"
    )
    assert report["model_calls"] == 7
    assert report["agent_action_proposals"] == 3

    actions = report["explicit_robot_actions"]
    assert actions["total"] == 2
    assert actions["by_action"] == {"move_pose": 2}

    hidden = report["hidden_planner_controller_work"]
    assert hidden["planner_backed_receipts"] == 2
    assert hidden["controller_sampled_steps"] == 42
    assert hidden["motion_probe_calls"] == 1
    assert hidden["grasp_plan_calls"] == 1

    boundaries = report["reobservation_boundaries"]
    assert boundaries["policy_visible_observations"] == 4

    classes = report["tool_calls"]["result_classes"]
    assert classes["mutation"] == 2
    assert classes["lifecycle"] == 1
    assert classes["read_only"] == 4
    assert report["tool_calls"]["errors_by_tool"] == {"move_pose": 1}
    assert report["execution_status"] == {"planner_failed": 1, "reached": 1}
    assert report["motion_receipts"] == 2
    assert "one-call macro" in report["claim_boundary"]

    persisted = json.loads(
        (run_root / "ownership-audit.json").read_text(encoding="utf-8")
    )
    assert persisted == report
    assert build_ownership_audit(run_root) == report


def test_ownership_audit_requires_the_plan(tmp_path: Path) -> None:
    with pytest.raises(OwnershipAuditError, match="required evidence"):
        build_ownership_audit(tmp_path)


def test_ownership_audit_marks_missing_surface_classification(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "plan.json").write_text(json.dumps({"policy": {}}), encoding="utf-8")
    report = build_ownership_audit(run_root)
    assert report["surface_profile_id"] is None
    assert report["tool_calls"]["classification_source"] == (
        "unclassified_missing_surface_snapshot"
    )
    assert report["model_calls"] is None
