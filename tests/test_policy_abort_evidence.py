"""abort is admitted only on an infrastructure failure the Harness recorded this round.

EXP11A g012 (E11G12-002): with the policy failure exit withheld, six rounds
ended in finish_episode(abort) with task judgements after 0-48 of 80 decisions;
the runtime accepted any non-empty reason. The abort admission
(policy_abort_infrastructure_round_v2) reads the run's own event log since the
round's reset -- tool errors the runtime wrote and receipts the Supervisor
wrote -- and refuses abort when it holds no infrastructure-class failure. The
behavioural rows below are every distinct tool-error and receipt failure shape
of the seven sealed g012 runs.

Offline: the classification of the Harness record.
"""
from __future__ import annotations

import json

import pytest

from ora.agent.composition.abort_evidence_schema import (
    EXECUTION_FAULT_CODES, PLANNER_FAULT_CODES, SKILL_FAULT_STOP_REASONS, TOOL_ERROR_CODES,
)
from ora.agent.tools import abort_evidence

def _tool_error(code, tool="grasp_plan"):
    return {"type": "tool_error", "tool": tool, "arguments": {}, "time": 1.0,
            "error": {"code": code, "message": "x", "retryable": True}}


def _receipt(**fields):
    return {"type": "motion_receipt", "time": 1.0, "consumed_decision": True,
            "record": {"path": "receipts/action-0001.json", "sha256": "0" * 64,
                       "receipt": {"action": "move_pose", "action_id": "action-1", **fields}}}


def _skill(stop_reason, stages):
    return {"type": "motion_receipt", "time": 1.0, "consumed_decision": True,
            "record": {"skill": "grasp_execute", "action_id": "action-2", "status": "aborted",
                       "stop_reason": stop_reason, "stage_receipts": stages}}


def _stage(stage, status, reason, receipt=None):
    return {"stage": stage, "status": status, "reason": reason, "action_id": f"action-2:{stage}",
            "receipt": None if receipt is None else {"path": "receipts/a.json", "sha256": "0" * 64,
                                                     "receipt": receipt}}


RESET = {"type": "reset", "time": 1.0, "record": {"round_index": 1}}
#: Every distinct failure shape the seven sealed g012 runs recorded.
G012_RECORD = (
    _tool_error("stale_observation", "grasp_plan"), _tool_error("stale_observation", "align_plan"),
    _tool_error("invalid_arguments", "compute"), _tool_error("invalid_arguments", "move_pose"),
    _tool_error("stale_candidate_observation", "grasp_execute"),
    _receipt(failure_stage="planner", failure_code="ik_failed:move", execution_status="planner_failed"),
    _receipt(failure_stage="planner", failure_code="ik_failed:local", execution_status="planner_failed"),
    _receipt(failure_stage="planner", failure_code="ik_failed:precontact", execution_status="planner_failed"),
    _receipt(failure_stage="plan_validation", failure_code="ik_branch_jump:move", execution_status="plan_rejected"),
    _receipt(failure_stage="execution", failure_code="controller_stalled", execution_status="not_reached"),
    _receipt(action="set_gripper", failure_stage="execution", failure_code="gripper_stalled_before_target",
             execution_status="not_reached"),
    _receipt(failure_stage="execution", failure_code="native_action_budget_exhausted",
             execution_status="not_reached"),
    _receipt(action="wait", failure_stage="execution", failure_code="episode_terminal_during_execution",
             execution_status="not_reached"),
    _receipt(failure_stage=None, failure_code=None, execution_status="reached"),
    _skill("pregrasp_not_reached", [
        _stage("open", "executed", "executed", {"execution_status": "reached"}),
        _stage("pregrasp", "not_reached", "controller_stalled",
               {"failure_stage": "execution", "failure_code": "controller_stalled",
                "execution_status": "not_reached"})]),
    _skill("sequence_completed", [
        _stage("close", "executed", "gripper_stalled_before_target",
               {"failure_stage": "execution", "failure_code": "gripper_stalled_before_target",
                "execution_status": "not_reached"}),
        _stage("verify", "executed", "verified_against_observation")]),
)
#: Receipts the Harness itself marks as an infrastructure fault.
INFRASTRUCTURE_RECEIPTS = {
    "execution_fault": _receipt(execution_status="execution_fault"),
    "execution_fault_record": _receipt(execution_status="not_reached", execution_fault={"code": "x"}),
    "execution_uncertain": _receipt(execution_status="not_reached", execution_uncertain=True),
    "planner_error": _receipt(failure_stage="planner", failure_code="planner_internal_error",
                              execution_status="planner_failed"),
    "invalid_measured_state": _receipt(failure_stage="execution", failure_code="invalid_measured_state",
                                       execution_status="not_reached"),
    "skill_fault": _skill("trusted_execution_fault", [_stage("approach", "unavailable", "trusted_execution_fault")]),
    "skill_stage_fault": _skill("approach_not_reached", [
        _stage("approach", "not_reached", "execution_fault", {"execution_status": "execution_fault"})]),
}
_INTER_ARM_DIAGNOSTICS = {"outcome": "inter_arm_collision_path", "phase": "joint_space",
                          "rejection_counts": {"inter_arm_collision_path": 1}}
#: The EXP11A g013 kernel unit's other-arm refusals (E11G12-004a), in the shapes
#: sparse_motion, inter_arm and recovery write them: ordinary planner and
#: recovery refusals of a target or route, not an infrastructure failure.
INTER_ARM_REFUSALS = {
    "single_arm_move": _receipt(failure_stage="planner", failure_code="inter_arm_collision:move",
                                execution_status="planner_failed", planning_diagnostics=_INTER_ARM_DIAGNOSTICS),
    "lockstep_pair_local": _receipt(failure_stage="planner", failure_code="inter_arm_collision:local",
                                    execution_status="planner_failed", planning_diagnostics=_INTER_ARM_DIAGNOSTICS),
    "after_motion_started": _receipt(failure_stage="planner", failure_code="inter_arm_collision:lift",
                                     execution_status="not_reached", planning_diagnostics=_INTER_ARM_DIAGNOSTICS),
    "recovery_before_motion": _receipt(action="recover_pose", failure_stage="plan_validation",
                                       failure_code="recovery_inter_arm_collision_path",
                                       execution_status="plan_rejected"),
    "recovery_mid_route": _receipt(action="recover_pose", failure_stage="execution",
                                   failure_code="recovery_inter_arm_collision_path", execution_status="not_reached"),
    "grasp_execute_stage": _skill("pregrasp_not_reached", [
        _stage("open", "executed", "executed", {"execution_status": "reached"}),
        _stage("pregrasp", "planner_failed", "inter_arm_collision:move",
               {"failure_stage": "planner", "failure_code": "inter_arm_collision:move",
                "execution_status": "planner_failed", "planning_diagnostics": _INTER_ARM_DIAGNOSTICS})]),
}


# --------------------------------------------------------------------------
# The classification of the Harness record
# --------------------------------------------------------------------------

def _signals(events):
    return abort_evidence.round_signals(json.dumps(event, sort_keys=True) for event in events)


def test_nothing_g012_recorded_is_an_infrastructure_failure():
    assert _signals(G012_RECORD) == []


@pytest.mark.parametrize("name", sorted(INFRASTRUCTURE_RECEIPTS))
def test_a_receipt_the_harness_marks_as_a_fault_qualifies(name):
    signals = _signals([*G012_RECORD, INFRASTRUCTURE_RECEIPTS[name]])
    assert len(signals) == 1 and signals[0]["source"] == "environment_receipt"


@pytest.mark.parametrize("code", sorted(TOOL_ERROR_CODES))
def test_an_infrastructure_tool_error_qualifies(code):
    assert _signals([_tool_error(code, "vision_ground")]) == [
        {"source": "tool_error", "tool": "vision_ground", "code": code}]


@pytest.mark.parametrize("code", ["invalid_arguments", "stale_observation", "target_ref_unavailable",
                                  "failure_exit_withheld", "abort_without_infrastructure_evidence",
                                  "task_success_not_confirmed",
                                  "grounding_failed", "segmentation_failed", "tool_unavailable"])
def test_a_refused_or_behavioural_tool_error_does_not(code):
    assert _signals([_tool_error(code)]) == []


@pytest.mark.parametrize("name", sorted(INTER_ARM_REFUSALS))
def test_an_inter_arm_refusal_is_not_an_infrastructure_failure(name):
    """E11G12-004a: the other arm in the way is a refused target or route, not a Harness failure."""
    assert _signals([RESET, *G012_RECORD, INTER_ARM_REFUSALS[name]]) == []
    for code in ("inter_arm_collision", "inter_arm_collision_path", "recovery_inter_arm_collision_path"):
        assert code not in TOOL_ERROR_CODES and code not in PLANNER_FAULT_CODES
        assert code not in EXECUTION_FAULT_CODES and code not in SKILL_FAULT_STOP_REASONS


def test_only_the_current_round_counts():
    fault = INFRASTRUCTURE_RECEIPTS["execution_fault"]
    assert _signals([_tool_error("internal_tool_error"), fault, RESET]) == []
    assert len(_signals([RESET, fault, _tool_error("internal_tool_error")])) == 2


def test_an_unreadable_line_is_not_evidence():
    lines = ["not json", json.dumps({"type": "tool_error"}), json.dumps([1, 2])]
    assert abort_evidence.round_signals(lines) == []
