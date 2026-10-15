"""Decision-relevant planner failures survive the private/public boundary."""

from __future__ import annotations

import json

import pytest

from ora.agent.native.receipt_projection import (
    CONTACT_MOTION_FEEDBACK_CAPABILITY, PUBLIC_MOTION_EFFECT_CAPABILITY,
    ReceiptProjectionContext, public_motion_failure_code,
    resolve_receipt_effect_projector, public_contact_motion_cause,
)


@pytest.mark.parametrize("raw,expected", [
    ("planner_error", "planner_error"),
    ("planner_internal_error", "planner_error"),
    ("planner_timeout", "planner_error"),
    ("MotionGenStatus.INVALID_START_STATE_JOINT_LIMITS", "invalid_start_state"),
    ("MotionGenStatus.INVALID_START_STATE_SELF_COLLISION", "invalid_start_state"),
    ("MotionGenStatus.IK_FAIL", "planner_failed"),
    ("MotionGenStatus.TRAJOPT_FAIL", "planner_failed"),
])
def test_public_planner_codes_distinguish_errors_start_and_search(
    raw: str, expected: str,
) -> None:
    assert public_motion_failure_code("planner", raw) == expected


def test_actual_sparse_projection_keeps_implementation_failure_without_private_error() -> None:
    record = {
        "schema": "ora.libero-sparse-motion-receipt.v1",
        "action": "move_pose", "action_id": "action-0124",
        "execution_status": "execution_fault", "failure_stage": "planner",
        "failure_code": "planner_error", "motion_strategy": "global",
        "executor": ["libero_curobo_joint_position"],
        "terminal_after_action": False,
        "segment": {"planner": {"error": {
            "type": "AttributeError", "message": "primitive_collision_constraint",
        }}},
    }
    projector = resolve_receipt_effect_projector(record["schema"])
    assert projector is not None
    public = projector(record, ReceiptProjectionContext(
        last_action_id="action-0124", last_action_public_name="move_pose",
    ))
    assert public["status"] == "execution_fault"
    assert public["failure_code"] == "planner_error"
    assert public["failure_stage"] == "planner"
    assert public["physical_cause"] == "unknown"
    assert "primitive_collision_constraint" not in json.dumps(public)
    assert "AttributeError" not in json.dumps(public)


def test_controller_timeout_remains_distinct_from_planning_failure() -> None:
    assert public_motion_failure_code("execution", "waypoint_tracking_timeout") == "controller_timeout"
    assert public_motion_failure_code("endpoint_verification", "pose_error") == "endpoint_residual_exceeded"


def _framed_contact(**changes):
    return {
        "contact_estimated": True, "contact_force_threshold_n": 10.,
        "external_force_magnitude_n": 30.9,
        "external_force_direction_world": [0., 0., 1.],
        "force_frame": "world",
        "force_direction_convention": "environment_on_eef_reaction_estimate",
        "realized_translation_world_m": [0., 0., .02],
        "requested_translation_world_m": [0., 0., 0.],
        **changes,
    }


@pytest.mark.parametrize("changes,cause", [
    ({}, "pushed_by_contact"),
    ({"force_frame": "sensor"}, "unknown"),
    ({"force_direction_convention": "unknown"}, "unknown"),
    ({"external_force_direction_world": None}, "unknown"),
    ({"external_force_magnitude_n": float("nan")}, "unknown"),
    ({"external_force_magnitude_n": 10.}, "unknown"),
    ({"contact_estimated": False}, "unknown"),
    ({"realized_translation_world_m": [0., 0., 0.]}, "blocked_by_contact"),
    ({"realized_translation_world_m": [0., 0., -.02],
      "requested_translation_world_m": [0., 0., .01]}, "blocked_by_contact"),
    ({"realized_translation_world_m": [0., 0., -.02]}, "unknown"),
])
def test_physical_cause_requires_framed_direction_or_measured_block(changes, cause):
    assert public_contact_motion_cause(_framed_contact(**changes)) == cause


def test_contact_projection_uses_measured_endpoints_and_v7_capability():
    record = {
        "schema": "ora.libero-sparse-motion-receipt.v1",
        "action": "move_pose", "execution_status": "not_reached",
        "failure_stage": "execution", "failure_code": "contact_stop",
        "motion_strategy": "local", "executor": ["libero_robosuite_osc_pose"],
        "segment": {
            "feedback": {**_framed_contact(), "private_sensor_id": "hidden"},
            "before_pose_xyz_xyzw": [0., 0., 0., 0., 0., 0., 1.],
            "reached_pose_xyz_xyzw": [0., 0., .02, 0., 0., 0., 1.],
            "commanded_pose_xyz_xyzw": [0., 0., 0., 0., 0., 0., 1.],
            "motion_realization": {"realized_translation_world_m": [9., 9., 9.]},
        },
        "command_drift": {"right": {"drift_from_last_commanded_m": [0., 0., .02],
                                      "drift_accumulating": True,
                                      "private_target": [1., 2., 3.]}},
    }
    projector = resolve_receipt_effect_projector(record["schema"])
    old = projector(record, ReceiptProjectionContext(None, None))
    new = projector(record, ReceiptProjectionContext(None, None,
        lambda capability: capability == CONTACT_MOTION_FEEDBACK_CAPABILITY))
    assert old["physical_cause"] == "unknown"
    assert "contact_motion" not in old["detail"]
    assert new["physical_cause"] == "pushed_by_contact"
    assert new["failure_code"] == "contact_stop"
    assert new["detail"]["contact_motion"]["realized_translation_world_m"] == [0., 0., .02]
    assert new["detail"]["command_drift"]["right"]["drift_accumulating"] is True
    assert "private_sensor_id" not in json.dumps(new)
    assert "private_target" not in json.dumps(new)


def test_robotwin_blocked_progress_reaches_public_receipt_without_solver_data():
    record = {
        "schema": "ora.robotwin-sparse-motion-receipt.v7",
        "action": "move_pose", "arms": {"left": {}, "right": {}},
        "synchronization": {"mode": "single_arm"}, "motion_strategy": "global",
        "execution_status": "not_reached", "failure_stage": "execution",
        "failure_code": "blocked", "controller_tracking": {"abort": {
            "code": "blocked", "blocked_at_stage": 65, "frozen_steps": 16,
            "progress": .65, "stall_shape": "abrupt",
            "cause_evidence": "measured_joint_stall_at_unchanged_target",
            "private_controller_target": [1, 2, 3],
        }},
    }
    projector = resolve_receipt_effect_projector(record["schema"])
    result = projector(record, ReceiptProjectionContext(None, None,
        lambda capability: capability in {CONTACT_MOTION_FEEDBACK_CAPABILITY,
                                          PUBLIC_MOTION_EFFECT_CAPABILITY}))
    assert result["physical_cause"] == "blocked_by_contact"
    assert result["failure_code"] == "blocked"
    assert result["detail"]["controller_tracking"]["progress"] == .65
    assert result["detail"]["controller_tracking"]["frozen_steps"] == 16
    assert "private_controller_target" not in json.dumps(result)


def test_hybrid_recover_refusal_retains_planner_start_failure():
    record = {
        "schema": "ora.libero-pro-hybrid-motion-receipt.v2",
        "action": "recover_motion", "motion_status": "refused",
        "feedback": {"planning_failure_reason": "MotionGenStatus.INVALID_START_STATE_SELF_COLLISION"},
    }
    projector = resolve_receipt_effect_projector(record["schema"])
    result = projector(record, ReceiptProjectionContext(None, None))
    assert (result["action"], result["status"], result["failure_stage"],
            result["failure_code"]) == ("recover_pose", "plan_rejected", "planner", "invalid_start_state")


def test_drift_tracker_excludes_refusals_repeats_and_intentional_translation():
    from ora.environment.controllers.command_drift import CommandDriftTracker

    tracker = CommandDriftTracker()
    def record(action, start, final, target, steps=1):
        return tracker.record(action_id=action, steps_used=steps, arms={"right": {
            "start_position_m": [0., 0., start],
            "final_position_m": [0., 0., final],
            "commanded_position_m": [0., 0., target],
        }})
    assert record("initial", 0., 0., 0.) == {}
    assert record("refusal", 0., 9., 9., steps=0) == {}
    first = record("rotation1", 0., .02, 0.)
    assert first["right"]["drift_from_last_commanded_m"] == [0., 0., .02]
    assert first["right"]["drift_accumulating"] is False
    assert record("rotation1", 0., 9., 9.) == first
    assert record("rotation2", .02, .04, .02)["right"]["drift_accumulating"] is True
    assert record("intentional", .04, .08, .08)["right"]["drift_accumulating"] is False
    assert record("rotation3", .08, .10, .08)["right"]["drift_accumulating"] is False


@pytest.mark.parametrize("checker_name", ["state", "trace"])
def test_public_contact_feedback_validation_is_path_and_type_bounded(checker_name):
    from ora.agent.evidence.policy_state import _forbidden_policy_key
    from ora.agent.evidence.trajectory import _forbidden_path
    check = _forbidden_policy_key if checker_name == "state" else _forbidden_path
    public = {"last_action_effect": {"detail": {"contact_motion": {
        "contact_estimated": True, "contact_force_threshold_n": 2.,
        "external_force_world_n": [0., 0., 3.],
    }}}, "previous_execution_receipt": {"motion_feedback": {"right": {
        "contact_estimated": None, "contact_force_threshold_n": 2.,
    }}, "segment": {"feedback": {"contact_force_threshold_n": 2.}}}}
    assert check(public) is None
    for value in ({"contact_estimated": True},
        {"feedback": {"contact_force_threshold_n": [1., 2.]}},
        {"feedback": {"contact_force_threshold_n": float("nan")}},
        {"feedback": {"contact_force_threshold_n": -1}},
        {"motion_feedback": {"right": {"contact_estimated": {"body": "object"}}}},
        {"last_action_effect": {"detail": {"contact_motion": {"object_gt": [1., 2., 3.]}}}},
        {"last_action_effect": {"detail": {"contact_motion": {"external_force_world_n": {"raw_contacts": []}}}}}):
        assert check(value) is not None
