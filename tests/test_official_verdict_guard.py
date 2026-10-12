"""The round-boundary guard between the official verdict and the checker.

The recorded verdict, the policy-visible ``check_task`` record and the trusted
termination reason are three projections of one environment predicate. On
2026-09-08 a RoboTwin ``hanging_mug`` round told the policy ``check_task`` was
true and then recorded ``official_success: false``. These tests pin the guard
that turns that contradiction into an infrastructure fault for the round,
leaves an agreeing round a behavioral sample, and treats a late or shadowed
latch -- record true, snapshot false -- as a stale snapshot to republish, not
as a contradiction.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any, Mapping

from ora.environment import SyntheticSession
from ora.environment.base import (
    ENVIRONMENT_STATUS_SCHEMA, validate_environment_status,
)
from ora.orchestrator.coordinator import run_episode
from ora.orchestrator.spec import resolve_run_spec
from test_multi_round_session import live_exp01b_spec
from ora.orchestrator.verdict_guard import (
    OFFICIAL_VERDICT_CHECKER_DIVERGENCE,
    official_verdict_divergence,
    verdict_divergence_fault,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = "synthetic-native-atomic-causal-cache-v1"


def test_official_verdict_divergence_compares_only_comparable_records() -> None:
    verdict = {"official_success": False, "source": "RoboTwin.eval_success"}
    checker = {"task_success": True, "source": "robotwin_session"}

    divergence = official_verdict_divergence(
        verdict, task_check=checker, round_index=0,
    )
    assert divergence is not None
    assert divergence["reason"] == OFFICIAL_VERDICT_CHECKER_DIVERGENCE
    assert divergence["policy_visible_task_success"] is True
    assert divergence["recorded_official_success"] is False
    assert divergence["round_index"] == 0

    fault = verdict_divergence_fault(divergence)
    assert fault.code == OFFICIAL_VERDICT_CHECKER_DIVERGENCE
    assert fault.layer == "environment"
    assert fault.retryable is False
    assert fault.as_dict()["code"] == OFFICIAL_VERDICT_CHECKER_DIVERGENCE

    assert official_verdict_divergence(verdict, task_check=None) is None
    assert official_verdict_divergence(None, task_check=checker) is None
    # Synthetic M0 has no official predicate at all.
    assert official_verdict_divergence(
        {"official_success": None}, task_check=checker,
    ) is None
    assert official_verdict_divergence(
        {"official_success": True}, task_check=checker,
    ) is None


class _AgreeingSyntheticSession(SyntheticSession):
    """Terminal on the official predicate, with an agreeing verdict."""

    official_success = True

    def status(self) -> dict[str, Any]:
        reason = self.abort_reason or (
            "official_success" if bool(self.terminal) else None
        )
        return validate_environment_status({
            "schema": ENVIRONMENT_STATUS_SCHEMA,
            "phase": "terminal" if reason is not None else "running",
            "terminal": reason is not None,
            "reason": reason,
            "source": "synthetic_verdict_guard_session",
            "policy_visibility": "control_only",
            "signals": [],
        })

    def persist_official_verdict(self) -> dict[str, Any]:
        verdict = {
            "schema": "ora.synthetic-verdict-guard.v1",
            "official_success": bool(self.official_success),
            "source": "synthetic_verdict_guard_session",
        }
        (self.output_root / "official-verdict.json").write_text(
            json.dumps(verdict, sort_keys=True) + "\n", encoding="utf-8",
        )
        return verdict


class _ContradictingSyntheticSession(_AgreeingSyntheticSession):
    """Terminal on the official predicate, then records the opposite."""

    official_success = False


class _ActThenFinishProvider:
    protocol = "anthropic_messages"

    def __init__(self) -> None:
        from ora.agent.providers.native import translate_anthropic_request

        self.prepare_request = translate_anthropic_request
        self.requests: list[dict[str, Any]] = []

    def complete(
        self, *, request: Mapping[str, Any], model_call_id: str,
    ) -> dict[str, Any]:
        self.requests.append(copy.deepcopy(dict(request)))
        call = len(self.requests) - 1
        name = "close_gripper" if call == 0 else "finish_episode"
        arguments: dict[str, Any] = {} if call == 0 else {"outcome": "success"}
        now = time.time()
        return {
            "message": {
                "role": "assistant", "content": "",
                "tool_calls": [{
                    "id": f"{name}-{call}", "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments, sort_keys=True),
                    },
                }],
            },
            "finish_reason": "tool_calls",
            "usage": {
                "input_tokens": 1000, "cached_input_tokens": 0,
                "output_tokens": 20, "reasoning_output_tokens": 0,
                "total_tokens": 1020,
            },
            "raw_events": [{"verdict_guard": True, "call": call}],
            "model_call_id": model_call_id,
            "started_unix_s": now, "finished_unix_s": now, "attempts": 1,
        }

    def close(self) -> None:
        return None


def _guard_plan() -> dict[str, Any]:
    spec = live_exp01b_spec(BASE_CONFIG)
    spec["experiment"] = "verdict-guard-test"
    spec["output_workspace"] = "outputs/verdict-guard-test"
    spec["milestone"] = "VERDICT-GUARD"
    spec["runtime"]["terminal_actions"] = 1
    spec["budgets"] = {
        "max_robot_decisions": 1, "max_model_calls": 3,
        "max_sim_steps": 1, "wall_time_s": 30,
    }
    return resolve_run_spec(spec, root=ROOT)


def _run_guarded_episode(
    tmp_path: Path, session_class: type[_AgreeingSyntheticSession],
) -> tuple[dict[str, Any], Path]:
    plan = _guard_plan()
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "plan.json").write_text(json.dumps(plan), encoding="utf-8")

    def environment_factory(**kwargs: Any) -> _AgreeingSyntheticSession:
        return session_class(
            kwargs["output_root"], kwargs["task"], kwargs["seed"],
            max_actions=1, main_camera="agentview",
            camera_names=("agentview", "wrist"),
        )

    result = run_episode(
        run_root=run_root, plan=plan, run_id="verdict-guard",
        provider_override=_ActThenFinishProvider(),
        environment_session_factory=environment_factory,
    )
    return result, run_root


def test_round_boundary_guard_reports_a_verdict_contradiction_as_infrastructure(
    tmp_path: Path,
) -> None:
    """A checker/verdict contradiction ends the round as a Harness fault."""

    result, run_root = _run_guarded_episode(
        tmp_path, _ContradictingSyntheticSession,
    )

    assert result["state"] == "incomplete"
    assert result["termination_authority"] == "infrastructure"
    assert result["behavioral_sample_valid"] is False
    # Never a behavioral verdict: the round yields no official success value.
    assert result["official_success"] is None
    divergence = result["official_verdict_checker_divergence"]
    assert divergence["reason"] == OFFICIAL_VERDICT_CHECKER_DIVERGENCE
    assert divergence["policy_visible_task_success"] is True
    assert divergence["recorded_official_success"] is False

    recorded = [
        json.loads(line) for line in
        (run_root / "control/official-verdict-divergence.jsonl")
        .read_text(encoding="utf-8").splitlines()
    ]
    assert [item["reason"] for item in recorded] == [
        OFFICIAL_VERDICT_CHECKER_DIVERGENCE,
    ]
    events = [
        json.loads(line) for line in
        (run_root / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    faults = [
        event for event in events
        if event.get("type") == "runtime_fault"
        and event["fault"]["code"] == OFFICIAL_VERDICT_CHECKER_DIVERGENCE
    ]
    assert len(faults) == 1
    assert faults[0]["fault"]["layer"] == "environment"
    decisions = [
        event for event in events
        if event.get("type") == "termination_authority_decision"
        and event["source"] == "verdict_checker_guard"
    ]
    assert [item["decision"]["authority"] for item in decisions] == [
        "infrastructure",
    ]
    assert [item["decision"]["reason"] for item in decisions] == [
        OFFICIAL_VERDICT_CHECKER_DIVERGENCE,
    ]


def test_agreeing_verdict_and_checker_stay_a_behavioral_sample(
    tmp_path: Path,
) -> None:
    """The guard is silent when the two projections agree."""

    result, run_root = _run_guarded_episode(tmp_path, _AgreeingSyntheticSession)

    assert result["state"] == "completed"
    assert result["termination_authority"] == "environment"
    assert result["behavioral_sample_valid"] is True
    assert result["official_success"] is True
    assert result["official_verdict_checker_divergence"] is None
    assert not (run_root / "control/official-verdict-divergence.jsonl").exists()
    events = [
        json.loads(line) for line in
        (run_root / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert not [
        event for event in events
        if event.get("type") == "runtime_fault"
    ]


# ---------------------------------------------------------------------------
# Every eval-path physics step polls, and the motion stops where it latches


def test_a_late_latch_is_not_a_divergence_but_a_stale_checker_snapshot(
) -> None:
    """The verdict may lead the checker snapshot; it may never trail it.

    ``publish`` writes the checker from ``status()`` at a policy boundary and
    ``status()`` gives a trusted abort reason precedence over
    ``official_success``, so the snapshot can say false for an episode that
    did latch. The latch never clears, so the reverse cannot happen honestly:
    a checker that says true against a recorded false is the contradiction
    that compelled a success claim, and only that is raised.
    """

    late_latch = official_verdict_divergence(
        {
            "official_success": True, "eval_success_seen": True,
            "source": "RoboTwin.eval_success (latched check_success)",
        },
        task_check={"task_success": False, "source": "robotwin_session"},
        round_index=1,
    )
    assert late_latch is None

    contradiction = official_verdict_divergence(
        {"official_success": False, "eval_success_seen": False},
        task_check={"task_success": True, "source": "robotwin_session"},
        round_index=1,
    )
    assert contradiction is not None
    assert contradiction["policy_visible_task_success"] is True
    assert contradiction["recorded_official_success"] is False


class _LateLatchSyntheticSession(_AgreeingSyntheticSession):
    """Not terminal at the last publish, then persists a latched success.

    This is the shape a genuine late latch takes: the checker snapshot the
    policy read says false, a trusted boundary then aborts the environment --
    which makes ``status()`` report the abort reason instead of
    ``official_success`` -- and the persisted record says true because the
    latch never clears. Nulling that round would throw away a success the
    published protocol counts.
    """

    official_success = True


def test_a_late_latch_stays_a_behavioral_success_and_refreshes_the_checker(
    tmp_path: Path,
) -> None:
    plan = _guard_plan()
    plan["budgets"]["max_robot_decisions"] = 4
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "plan.json").write_text(json.dumps(plan), encoding="utf-8")

    def environment_factory(**kwargs: Any) -> _LateLatchSyntheticSession:
        return _LateLatchSyntheticSession(
            kwargs["output_root"], kwargs["task"], kwargs["seed"],
            max_actions=4, main_camera="agentview",
            camera_names=("agentview", "wrist"),
        )

    result = run_episode(
        run_root=run_root, plan=plan, run_id="late-latch",
        provider_override=_ActThenFinishProvider(),
        environment_session_factory=environment_factory,
    )

    # A budget boundary aborts the environment, so the last checker snapshot
    # says false while the record says true: the exact stale-snapshot shape.
    assert result["termination_authority"] == "experiment_budget"
    # The round the published protocol counts is not nulled by the guard.
    assert result["official_success"] is True
    assert result["behavioral_sample_valid"] is True
    assert result["official_verdict_checker_divergence"] is None
    assert not (run_root / "control/official-verdict-divergence.jsonl").exists()

    # And the policy-visible checker now agrees with the record, so a
    # terminal turn and the finish rule read the scored predicate.
    checker = json.loads(
        (run_root / "control/policy-task-check.json").read_text(encoding="utf-8")
    )
    assert checker["task_success"] is True
    assert checker["terminal"] is True
    assert checker["projected_from"] == "persisted_official_verdict"


# ---------------------------------------------------------------------------
