"""A refused proposal that wrote nothing does not spend a robot decision.

The policy is told its decision allowance counts accepted world mutations. The
Coordinator used to charge every dispatched proposal, so the EXP11A g003
RoboDojo batch spent 118 decisions on planner and route refusals that executed
zero native actions. These contracts pin the certificate that identifies a
refusal, the separate counter and the per-round budget it no longer consumes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from ora.agent.evidence.policy_state import verify_policy_state
from ora.environment import SyntheticSession, validate_runtime_action
from ora.orchestrator.coordinator import run_episode
from ora.orchestrator.decision_accounting import is_zero_write_refusal
from ora.orchestrator.evaluation import evaluate
from ora.orchestrator.interaction import EpisodeInteractionState
from test_runtime_v1 import synthetic_plan


# The g003 shape: the RoboDojo worker wraps the receipt it persisted.
DOJO_PLANNER_REFUSAL = {
    "path": "receipts/action-0000.json", "sha256": "0" * 64,
    "receipt": {
        "schema": "ora.robodojo-sparse-motion-receipt.v1",
        "execution_status": "planner_failed", "consumed": True,
        "state_changed": False, "native_actions": 0, "physics_steps": 0,
    },
}


@pytest.mark.parametrize("receipt", [
    DOJO_PLANNER_REFUSAL,
    {"execution_status": "plan_rejected", "state_changed": False, "steps_used": 0},
    {"motion_status": "refused", "state_changed": False, "steps_used": 0,
     "sim_steps_total": 412},
    {
        "schema": "ora.composite-motion-receipt.v1",
        "phase_receipts": [
            {"phase": "pose", "status": "not_reached", "receipt": DOJO_PLANNER_REFUSAL},
            {"phase": "gripper", "status": "skipped_motion_not_reached", "receipt": None},
        ],
    },
], ids=["dojo-planner-failed", "sparse-plan-rejected", "libero-refused", "composite"])
def test_a_certified_zero_write_receipt_is_a_refusal(receipt):
    assert is_zero_write_refusal(receipt) is True


@pytest.mark.parametrize("receipt", [
    # RoboTwin omits its zero certificate exactly when execution was attempted
    # or faulted: an unchanged flag alone is not evidence of zero execution.
    {"execution_status": "not_reached", "state_changed": False},
    {"state_changed": False, "native_actions": 3},
    {"state_changed": False, "native_actions": 0, "physics_steps": 10},
    {"state_changed": False, "native_actions": False},
    {"state_changed": True, "native_actions": 0},
    {"state_changed": None, "steps_used": 0},
    {
        "schema": "ora.composite-motion-receipt.v1",
        "phase_receipts": [
            {"phase": "pose", "status": "skipped_environment_terminal", "receipt": None},
        ],
    },
    {
        "schema": "ora.composite-motion-receipt.v1",
        "phase_receipts": [
            {"phase": "pose", "status": "reached",
             "receipt": {"state_changed": True, "native_actions": 4}},
            {"phase": "gripper", "status": "executed", "receipt": DOJO_PLANNER_REFUSAL},
        ],
    },
    {"schema": "ora.manipulation-skill-sequence-receipt.v1",
     "status": "aborted", "robot_decisions": 1},
    None,
], ids=[
    "robotwin-uncertified", "nonzero-native", "nonzero-physics", "bool-count",
    "changed", "unknown-flag", "composite-nothing-dispatched",
    "composite-partly-executed", "skill-sequence", "missing",
])
def test_anything_without_that_certificate_spends_a_decision(receipt):
    assert is_zero_write_refusal(receipt) is False


def test_interaction_state_counts_a_refusal_separately(tmp_path: Path) -> None:
    state = EpisodeInteractionState(run_id="refusal", evidence_path=tmp_path / "i.json")
    state.start()
    for index, consumed in enumerate((False, True)):
        state.publish_observation(observation_id=f"observation-{index}", step=index)
        state.begin_policy()
        state.record_policy_outcome(kind="action", model_calls=1, action_id=f"a-{index}")
        state.begin_execution()
        state.record_execution(receipt_id=None, consumed_decision=consumed)
        assert state.fresh_observation_required is True
        assert state.world_state_certainty == ("changed" if consumed else "unchanged")
    snapshot = json.loads((tmp_path / "i.json").read_text())
    assert (snapshot["decisions"], snapshot["refused_proposals"]) == (1, 1)


REFUSED_Z = 1.25


class RefusingSyntheticSession(SyntheticSession):
    """Refuses any pose at ``REFUSED_Z`` before dispatch, writing nothing."""

    def execute(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        action = validate_runtime_action(payload)
        if action["arguments"].get("position_m", [None] * 3)[2] != REFUSED_Z:
            return super().execute(payload)
        receipt = {
            "schema": "ora.synthetic-motion-receipt.v1",
            "action": action, "consumed": True, "state_changed": False,
            "execution_status": "planner_failed", "native_actions": 0,
            "before": self.end_effectors, "after": self.end_effectors,
            "semantic_action_count": self.round_action_sequence,
        }
        return {"path": None, "sha256": None, "receipt": receipt}


def _pose(z: float) -> dict[str, Any]:
    return {"action": "move_pose", "arm": "left", "position_m": [0, 0, z],
            "orientation_xyzw": [0, 0, 0, 1]}


def test_refusals_leave_the_decision_budget_and_are_recorded(tmp_path: Path) -> None:
    run = tmp_path / "refusal-run"
    run.mkdir()
    plan = synthetic_plan()
    # Two refusals then two accepted moves under a two-decision allowance. The
    # old accounting ended the round on robot_decision_budget after the two
    # refusals, before either accepted move ran.
    plan["policy"]["actions"] = [_pose(REFUSED_Z), _pose(REFUSED_Z), _pose(1.01), _pose(1.02)]
    plan["runtime"]["terminal_actions"] = 2
    plan["budgets"].update(max_robot_decisions=2, max_model_calls=8)
    session = RefusingSyntheticSession(
        run / "native", "synthetic_tool_boundary", 0, max_actions=2,
    )
    (run / "plan.json").write_text(json.dumps(plan))
    result = run_episode(
        run_root=run, plan=plan, run_id="refusal-run",
        environment_session_factory=lambda **_kwargs: session,
    )

    assert result["termination"] == "trusted_runtime_boundary"
    assert (result["decisions"], result["refused_proposals"]) == (2, 2)
    status = json.loads((run / "status.json").read_text())
    assert (status["decisions"], status["refused_proposals"]) == (2, 2)

    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    receipts = [event for event in events if event["type"] == "motion_receipt"]
    assert [event["consumed_decision"] for event in receipts] == [False, False, True, True]

    # What the policy saw: a refusal leaves its remaining allowance unchanged,
    # an accepted move spends one.
    steps = verify_policy_state(run)["steps"]
    remaining = [step["remaining_budget"]["robot_decisions"] for step in steps]
    assert remaining == [2, 2, 2, 1, 0]

    interaction = json.loads((run / "native/episode-interaction.json").read_text())
    assert (interaction["decisions"], interaction["refused_proposals"]) == (2, 2)
    assert evaluate(run)["status"] == "valid"
