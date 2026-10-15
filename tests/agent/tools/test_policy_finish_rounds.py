"""The finish outcome reaches the round record, result.json and evaluation (F3).

User decision 2026-10-04 (integration 0059): on the v2 finish contract a
``give_up`` ends the round as a valid behavioural failure sample with the
official verdict kept; an abort the Harness admitted on recorded
infrastructure evidence ends only the current round, records no official
verdict for it and marks the round's sample invalid. Before this unit every
finish -- abort included -- ended as a valid sample and the outcome itself was
dropped before the round record.

The outcome, category and reason are read from the Harness-written
``policy-finish.json`` and carried on the round (or single-run) result and the
``termination_authority_decision`` event; evaluation checks that an admitted
abort round is the only kind of incomplete round a completed session may hold.
A surface without the capability keeps its records exactly.

Offline: Synthetic sessions on a test-local v2 copy of the synthetic surface
and a scripted provider.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import time
from pathlib import Path
from typing import Any, Mapping

import tests.support.finish_v2_surfaces as v2
from ora.agent.contracts import LifecycleRequest
from ora.agent.tools.runtime import ManipulationToolRuntime
from ora.orchestrator.coordinator import run_episode
from ora.orchestrator.episode_bindings import robodojo_official_complete
from ora.orchestrator.evaluation import _robodojo_evidence_errors, evaluate, verify_multi_round_session
from ora.orchestrator.policy_factory import build_policy_session
from ora.orchestrator.policy_finish import policy_finish_reasons, summarize
from ora.orchestrator.spec import resolve_run_spec
from ora.orchestrator.termination import TerminationAuthority, policy_outcome_decision
from tests.agent.native.test_multi_round_session import ROOT, _RoundProvider, _spec

SYNTHETIC = "native_atomic_causal_v1"
GIVE_UP = {"outcome": "give_up", "category": "object_lost",
           "reason": "the target block left the table and no camera shows it any more"}
ABORT = {"outcome": "abort", "reason": "report_issue crashed with internal_tool_error in this round"}


class _ScriptProvider(_RoundProvider):
    """One scripted tool call per provider request."""

    def __init__(self, script: list[tuple[str, Mapping[str, Any]]]) -> None:
        super().__init__()
        self.script = list(script)

    def complete(self, *, request: Mapping[str, Any], model_call_id: str) -> dict[str, Any]:
        self.requests.append(copy.deepcopy(dict(request)))
        call = len(self.requests) - 1
        name, arguments = self.script[call]
        now = time.time()
        return {
            "message": {"role": "assistant", "content": "", "tool_calls": [{
                "id": f"{name}-{call}", "type": "function",
                "function": {"name": name, "arguments": json.dumps(dict(arguments), sort_keys=True)}}]},
            "finish_reason": "tool_calls",
            "usage": {"input_tokens": 1000, "cached_input_tokens": 0, "output_tokens": 20,
                      "reasoning_output_tokens": 0, "total_tokens": 1020},
            "raw_events": [{"qualification": True, "call": call}],
            "model_call_id": model_call_id,
            "started_unix_s": now, "finished_unix_s": now, "attempts": 1,
        }


def _crashing_report_issue(monkeypatch) -> None:
    def crashed(self, arguments):
        raise RuntimeError("issue service connection refused")

    monkeypatch.setattr(ManipulationToolRuntime, "_report_issue", crashed)


class _OverrunPolicy:
    """Reports every lifecycle turn as having spent past the model-call budget."""

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.capabilities = delegate.capabilities

    def act(self, turn: Any) -> Any:
        outcome = self.delegate.act(turn)
        if outcome.kind == "lifecycle":
            return dataclasses.replace(outcome, model_calls=outcome.model_calls + 5)
        return outcome

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)


def _run(monkeypatch, tmp_path: Path, script, *, rounds: int | None, declared: bool = True,
         overrun: bool = False):
    if declared:
        v2.install(monkeypatch, tmp_path, SYNTHETIC, profile_id=SYNTHETIC)
    spec = _spec()
    if rounds is not None:
        spec["session"] = {"cadence": "persistent_agent_k_rounds_v1", "rounds": rounds}
    plan = resolve_run_spec(spec, root=ROOT)
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    extra = ({"policy_session_factory": lambda **kwargs: _OverrunPolicy(build_policy_session(**kwargs))}
             if overrun else {})
    result = run_episode(run_root=run_root, plan=plan, run_id="policy-finish-v2",
                         provider_override=_ScriptProvider(script), **extra)
    events = [json.loads(line) for line in (run_root / "events.jsonl").read_text().splitlines()]
    return plan, run_root, result, events


# --------------------------------------------------------------------------
# The authority decision
# --------------------------------------------------------------------------

def test_an_admitted_infrastructure_abort_is_a_round_scoped_infrastructure_boundary():
    request = LifecycleRequest(reason="policy_finished_episode", model_calls=1)
    released = policy_outcome_decision(request)
    assert released.authority is TerminationAuthority.POLICY_LIFECYCLE and released.behavioral_boundary
    decision = policy_outcome_decision(request, infrastructure_abort=True)
    assert decision.reason == "policy_finished_episode"
    assert decision.authority is TerminationAuthority.INFRASTRUCTURE
    assert decision.scope == "round" and decision.abort_environment is True
    assert decision.persist_official_verdict is False and decision.behavioral_boundary is False


# --------------------------------------------------------------------------
# A persistent session on a v2 surface
# --------------------------------------------------------------------------

def test_give_up_is_valid_and_an_admitted_abort_invalidates_only_its_round(monkeypatch, tmp_path):
    _crashing_report_issue(monkeypatch)
    plan, run_root, result, events = _run(monkeypatch, tmp_path, [
        ("finish_episode", GIVE_UP),                                       # round 0
        ("report_issue", {"message": "issue"}), ("finish_episode", ABORT),  # round 1
        ("finish_episode", ABORT), ("finish_episode", GIVE_UP),            # round 2
    ], rounds=3)

    assert result["state"] == "completed" and result["termination"] == "session_round_limit"
    rounds = result["rounds"]
    assert [item["termination"] for item in rounds] == ["policy_finished_episode"] * 3
    assert [item["termination_authority"] for item in rounds] == [
        "policy_lifecycle", "infrastructure", "policy_lifecycle"]
    assert [item["behavioral_sample_valid"] for item in rounds] == [True, False, True]
    assert [item["state"] for item in rounds] == ["completed", "incomplete", "completed"]
    # give_up keeps the official verdict (Synthetic M0's is null by design);
    # the aborted round persists none.
    assert rounds[0]["verdict"] is not None and rounds[2]["verdict"] is not None
    assert rounds[1]["official_success"] is None and rounds[1]["verdict"] is None
    finishes = [item["policy_finish"] for item in rounds]
    assert [item["outcome"] for item in finishes] == ["give_up", "abort", "give_up"]
    assert [item["category"] for item in finishes] == ["object_lost", None, "object_lost"]
    assert finishes[0]["reason"] == GIVE_UP["reason"] and all(item["ended_round"] for item in finishes)
    assert finishes[1]["abort_admission"] == "infrastructure_evidence"
    assert finishes[1]["abort_infrastructure_signal_count"] == 1
    assert finishes[0]["untried_candidates"]["reason_names_untried"] is False
    for index in range(3):
        persisted = json.loads((run_root / "rounds" / f"round-{index:04d}" / "result.json").read_text())
        assert persisted["policy_finish"] == finishes[index]
    decisions = [event for event in events if event.get("type") == "termination_authority_decision"
                 and event.get("source") == "policy_outcome"]
    assert [event["policy_finish"]["outcome"] for event in decisions] == ["give_up", "abort", "give_up"]
    assert [event["decision"]["authority"] for event in decisions] == [
        "policy_lifecycle", "infrastructure", "policy_lifecycle"]
    # Round 2's abort found no infrastructure failure since its own reset.
    refused = [event for event in events if event.get("type") == "tool_error"
               and event.get("tool") == "finish_episode"]
    assert [event["error"]["code"] for event in refused] == ["abort_without_infrastructure_evidence"]

    assert verify_multi_round_session(run_root, plan, result, events) == []
    assert policy_finish_reasons(result) == []
    evaluation = evaluate(run_root)
    assert evaluation["status"] == "valid", evaluation["reasons"]
    assert [(item["round_index"], item["outcome"], item["behavioral_sample_valid"])
            for item in evaluation["policy_finish"]] == [
        (0, "give_up", True), (1, "abort", False), (2, "give_up", True)]


def test_evaluation_refuses_an_abort_round_the_record_does_not_support(monkeypatch, tmp_path):
    _crashing_report_issue(monkeypatch)
    plan, run_root, result, events = _run(monkeypatch, tmp_path, [
        ("finish_episode", GIVE_UP),
        ("report_issue", {"message": "issue"}), ("finish_episode", ABORT),
    ], rounds=2)
    counted_valid = copy.deepcopy(result)
    counted_valid["rounds"][1]["behavioral_sample_valid"] = True
    assert "policy_finish_abort_record_invalid:1" in policy_finish_reasons(counted_valid)
    unsupported = copy.deepcopy(result)
    unsupported["rounds"][1]["policy_finish"]["abort_infrastructure_signal_count"] = 0
    assert "policy_finish_abort_record_invalid:1" in policy_finish_reasons(unsupported)
    # An incomplete round that is not an admitted abort still breaks a completed session.
    plain = copy.deepcopy(result)
    del plain["rounds"][1]["policy_finish"]
    assert "multi_round_state_invalid:1" in verify_multi_round_session(run_root, plan, plain, events)
    uncategorised = copy.deepcopy(result)
    uncategorised["rounds"][0]["policy_finish"]["category"] = None
    assert "policy_finish_record_invalid:0" in policy_finish_reasons(uncategorised)
    # Each summary is bound to its archived finish record and decision event.
    assert policy_finish_reasons(result, run_root=run_root, events=events) == []
    recategorised = copy.deepcopy(result)
    recategorised["rounds"][0]["policy_finish"]["category"] = "other"
    assert "policy_finish_record_unbound:0" in policy_finish_reasons(
        recategorised, run_root=run_root, events=events)
    relabelled = copy.deepcopy(result)
    relabelled["rounds"][0].update(termination_authority="infrastructure", behavioral_sample_valid=False,
                                   official_success=None, state="incomplete",
                                   policy_finish=copy.deepcopy(result["rounds"][1]["policy_finish"]))
    reasons = policy_finish_reasons(relabelled, run_root=run_root, events=events)
    assert "policy_finish_record_unbound:0" in reasons
    assert "policy_finish_record_unbound:0" in evaluate_reasons(run_root, relabelled)


def evaluate_reasons(run_root: Path, result: Mapping[str, Any]) -> list[str]:
    """Evaluate a tampered result.json in place of the sealed one."""
    (run_root / "result.json").write_text(json.dumps(result), encoding="utf-8")
    return evaluate(run_root)["reasons"]


def test_a_finish_on_a_budget_overrun_turn_is_still_recorded(monkeypatch, tmp_path):
    """The turn that spent past the model-call budget may carry the round's finish."""
    _crashing_report_issue(monkeypatch)
    _plan, _run_root, result, _events = _run(monkeypatch, tmp_path, [
        ("finish_episode", GIVE_UP),
        ("report_issue", {"message": "issue"}), ("finish_episode", ABORT),
    ], rounds=2, overrun=True)
    give_up, abort = result["rounds"]
    # The budget ends the give_up round, and the give_up is still on the record.
    assert give_up["termination"] == "model_call_budget"
    assert give_up["termination_authority"] == "experiment_budget"
    assert give_up["behavioral_sample_valid"] is True
    assert give_up["policy_finish"]["outcome"] == "give_up"
    assert give_up["policy_finish"]["category"] == "object_lost"
    assert give_up["policy_finish"]["ended_round"] is False
    # An admitted abort still invalidates its round.
    assert abort["termination"] == "policy_finished_episode"
    assert abort["termination_authority"] == "infrastructure"
    assert abort["behavioral_sample_valid"] is False and abort["official_success"] is None
    assert abort["policy_finish"]["outcome"] == "abort" and abort["policy_finish"]["ended_round"] is True
    assert policy_finish_reasons(result) == []


# --------------------------------------------------------------------------
# RoboDojo: a session with an aborted round still completes
# --------------------------------------------------------------------------

def _abort_summary() -> dict[str, Any]:
    record = {"schema": "ora.openeta-policy-finish.v1", "finish_contract": "policy_finish_outcomes_v2",
              **ABORT, "category": None, "abort_admission": "infrastructure_evidence",
              "abort_infrastructure_evidence": [
                  {"source": "tool_error", "tool": "report_issue", "code": "internal_tool_error"}]}
    return summarize(json.dumps(record).encode(), ended_round=True)


def test_a_robodojo_session_with_an_admitted_abort_round_completes(tmp_path):
    import tests.environments.robodojo.test_robodojo_runtime as dojo
    run_root, plan, result, events = dojo._multi_round_evidence(tmp_path)
    (run_root / "native/rounds/round-0001/official-verdict.json").unlink()
    record = result["rounds"][1]
    record.update(state="incomplete", termination="policy_finished_episode",
                  termination_authority="infrastructure", behavioral_sample_valid=False,
                  official_success=None, verdict=None, policy_finish=_abort_summary())
    assert robodojo_official_complete(None, result["rounds"], multi_round=True, round_limit=3) is True
    assert _robodojo_evidence_errors(run_root, plan, result, events) == []
    # Without the admitted-abort summary the same round is still a missing verdict.
    del record["policy_finish"]
    assert robodojo_official_complete(None, result["rounds"], multi_round=True, round_limit=3) is False
    assert "robodojo_official_verdict_missing:1" in _robodojo_evidence_errors(run_root, plan, result, events)


# --------------------------------------------------------------------------
# A single-episode run
# --------------------------------------------------------------------------

def test_a_single_run_carries_its_give_up_on_result_json(monkeypatch, tmp_path):
    _plan, run_root, result, _events = _run(monkeypatch, tmp_path, [("finish_episode", GIVE_UP)], rounds=None)
    assert result["state"] == "completed" and result["termination"] == "policy_finished_episode"
    assert result["behavioral_sample_valid"] is True
    assert result["policy_finish"]["outcome"] == "give_up"
    assert json.loads((run_root / "result.json").read_text())["policy_finish"] == result["policy_finish"]
    assert evaluate(run_root)["status"] == "valid"


def test_a_single_run_ended_by_an_admitted_abort_is_incomplete(monkeypatch, tmp_path):
    _crashing_report_issue(monkeypatch)
    _plan, run_root, result, _events = _run(monkeypatch, tmp_path, [
        ("report_issue", {"message": "issue"}), ("finish_episode", ABORT)], rounds=None)
    assert result["state"] == "incomplete"
    assert result["termination"] == "policy_finished_episode"
    assert result["termination_authority"] == "infrastructure"
    assert result["behavioral_sample_valid"] is False and result["official_success"] is None
    assert result["policy_finish"]["outcome"] == "abort"
    evaluation = evaluate(run_root)
    assert evaluation["status"] == "incomplete", evaluation["reasons"]
    assert evaluation["learning_eligibility"]["eligible"] is False


# --------------------------------------------------------------------------
# A released surface
# --------------------------------------------------------------------------

def test_a_surface_without_the_capability_keeps_its_records(monkeypatch, tmp_path):
    _plan, run_root, result, events = _run(monkeypatch, tmp_path, [
        ("finish_episode", {"outcome": "abort", "reason": "camera returns no frames"}),
    ], rounds=1, declared=False)
    record = result["rounds"][0]
    assert record["termination_authority"] == "policy_lifecycle"
    assert record["behavioral_sample_valid"] is True
    assert "policy_finish" not in record and "policy_finish" not in result
    assert all("policy_finish" not in event for event in events)
    evaluation = evaluate(run_root)
    assert evaluation["status"] == "valid" and "policy_finish" not in evaluation
