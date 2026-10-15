"""The binding native-action allowance is reported beside the other allowances.

RoboDojo ends an episode when its native-action limit is spent, and a Harness
physics cap below that limit ends it sooner, yet the policy saw only robot
decisions, model calls and wall time. These contracts pin the effective limit,
the usage sum over executed receipts and its delivery in the published
remaining budget.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from ora.agent.evidence.policy_state import verify_policy_state
from ora.environment import SyntheticSession, validate_runtime_action
from ora.orchestrator.coordinator import run_episode
from ora.orchestrator.native_allowance import (
    NativeActionAllowance, effective_native_action_limit, receipt_native_actions,
)
from tests.orchestrator.test_runtime_v1 import synthetic_plan


def _plan(kind: str, limit: Any, max_sim_steps: int) -> dict[str, Any]:
    return {"runtime": {"kind": kind, "native_action_limit": limit},
            "budgets": {"max_sim_steps": max_sim_steps}}


@pytest.mark.parametrize("plan, expected", [
    # g003 imitate_sorting_sequence: official 1600, physics cap 10000/10.
    (_plan("robodojo", 1600, 10000), 1000),
    (_plan("robodojo", 550, 5500), 550),
    # A kind without a registered physics ratio keeps its official limit.
    (_plan("synthetic", 40, 5), 40),
    # No official limit: nothing is reported.
    ({"runtime": {"kind": "robotwin"}, "budgets": {"max_sim_steps": 400}}, None),
    (_plan("robodojo", True, 10000), None),
    (_plan("robodojo", 0, 10000), None),
])
def test_the_effective_limit_is_the_smaller_of_official_and_physics_cap(plan, expected):
    assert effective_native_action_limit(plan) == expected


def _worker(count: int | None) -> dict[str, Any]:
    receipt = {"schema": "ora.robodojo-sparse-motion-receipt.v1",
               "state_changed": bool(count)}
    if count is not None:
        receipt["native_actions"] = count
    return {"path": "receipts/x.json", "sha256": "0" * 64, "receipt": receipt}


@pytest.mark.parametrize("receipt, expected", [
    (_worker(12), 12),
    (_worker(0), 0),
    (_worker(None), None),
    ({"schema": "ora.composite-motion-receipt.v1", "phase_receipts": [
        {"phase": "pose", "status": "reached", "receipt": _worker(8)},
        {"phase": "gripper", "status": "executed", "receipt": _worker(3)},
    ]}, 11),
    ({"schema": "ora.composite-motion-receipt.v1", "phase_receipts": [
        {"phase": "pose", "status": "skipped_environment_terminal", "receipt": None},
    ]}, 0),
    ({"schema": "ora.manipulation-skill-sequence-receipt.v1", "stage_receipts": [
        {"stage": "approach", "receipt": _worker(9)},
        {"stage": "verify", "action_id": None, "measurements": []},
        {"stage": "close", "receipt": _worker(4)},
    ]}, 13),
    ({"schema": "ora.manipulation-skill-sequence-receipt.v1", "stage_receipts": [
        {"stage": "approach", "receipt": _worker(None)},
    ]}, None),
    ({"schema": "ora.synthetic-motion-receipt.v1"}, None),
    (None, None),
])
def test_usage_sums_the_executed_worker_counts(receipt, expected):
    assert receipt_native_actions(receipt) == expected


def test_an_uncounted_receipt_makes_the_remainder_unknown_not_guessed():
    allowance = NativeActionAllowance(_plan("robodojo", 550, 5500))
    assert allowance.fields() == {"native_actions_limit": 550, "native_actions_remaining": 550}
    allowance.record(_worker(20))
    assert allowance.fields()["native_actions_remaining"] == 530
    allowance.record(_worker(None))
    allowance.record(_worker(20))
    assert allowance.fields() == {"native_actions_limit": 550, "native_actions_remaining": None}
    allowance.reset()
    assert allowance.fields()["native_actions_remaining"] == 550
    assert NativeActionAllowance({"runtime": {"kind": "robotwin"}}).fields() == {}


REFUSED_Z = 1.25


class CountingSyntheticSession(SyntheticSession):
    """Reports seven native actions per accepted move; refuses at ``REFUSED_Z``."""

    def execute(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        action = validate_runtime_action(payload)
        if action["arguments"].get("position_m", [None] * 3)[2] == REFUSED_Z:
            receipt = {"schema": "ora.synthetic-motion-receipt.v1",
                       "action": action, "consumed": True, "state_changed": False,
                       "native_actions": 0,
                       "semantic_action_count": self.round_action_sequence}
            return {"path": None, "sha256": None, "receipt": receipt}
        record = super().execute(payload)
        return {**record, "receipt": {**record["receipt"], "native_actions": 7}}


def _pose(z: float) -> dict[str, Any]:
    return {"action": "move_pose", "arm": "left", "position_m": [0, 0, z],
            "orientation_xyzw": [0, 0, 0, 1]}


def _episode(tmp_path: Path, runtime_extra: dict[str, Any]) -> list[dict[str, Any]]:
    run = tmp_path / "allowance-run"
    run.mkdir()
    plan = synthetic_plan()
    plan["policy"]["actions"] = [_pose(1.01), _pose(REFUSED_Z), _pose(1.02)]
    plan["runtime"].update(terminal_actions=2, **runtime_extra)
    plan["budgets"].update(max_robot_decisions=3, max_model_calls=8)
    session = CountingSyntheticSession(
        run / "native", "synthetic_tool_boundary", 0, max_actions=2,
    )
    (run / "plan.json").write_text(json.dumps(plan))
    run_episode(run_root=run, plan=plan, run_id="allowance-run",
                environment_session_factory=lambda **_kwargs: session)
    return [step["remaining_budget"] for step in verify_policy_state(run)["steps"]]


def test_the_published_remaining_budget_reports_the_native_allowance(tmp_path: Path) -> None:
    budgets = _episode(tmp_path, {"native_action_limit": 40})
    assert [(b["native_actions_limit"], b["native_actions_remaining"]) for b in budgets] == [
        (40, 40), (40, 33), (40, 33), (40, 26),
    ]
    # The refusal spent neither a decision nor a native action.
    assert [b["robot_decisions"] for b in budgets] == [3, 2, 2, 1]


def test_a_backend_without_a_limit_keeps_its_budget_shape(tmp_path: Path) -> None:
    budgets = _episode(tmp_path, {})
    assert all("native_actions_limit" not in b and "native_actions_remaining" not in b
               for b in budgets)


@pytest.mark.parametrize("retention", ["append_only_epoch_v1", "rewrite_compact_v1"])
def test_native_delivers_the_allowance_beside_the_other_allowances(tmp_path, retention):
    """Native filters remaining-budget keys; the allowance must pass that filter."""

    from types import SimpleNamespace

    from ora.agent.native.session import NativePlanner
    from tests.agent.context.test_public_robot_facts import published_fixture

    surface, context, _, _ = published_fixture(tmp_path, "robodojo")
    planner = NativePlanner.__new__(NativePlanner)
    planner.engine = SimpleNamespace()
    planner.run_root, planner.supports_images = tmp_path, True
    planner.tools = SimpleNamespace(surface=surface, controlled_end_effector="right")
    planner.last_actual_snapshot, planner.pending_receipt_body, planner.pending_receipt = None, None, None
    planner.actual_sequence, planner.context_retention = 0, retention
    planner._current_observation_id = lambda: context["public_robot_facts"]["observation_id"]

    def delivered(budget):
        value = {**context, "remaining_budget": budget}
        message = planner._causal_observation_message(context=value, terminal_lifecycle=False)
        return json.loads(message["content"][0]["text"])["remaining_budget"]

    base = dict(context["remaining_budget"])
    assert delivered({**base, "native_actions_limit": 1000, "native_actions_remaining": 988}) == {
        **base, "native_actions_limit": 1000, "native_actions_remaining": 988,
    }
    assert delivered(base) == base
