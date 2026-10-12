"""Termination authority is explicit and ordinary NoAction owns none."""

from __future__ import annotations

from ora.agent.contracts import ActionProposal, LifecycleRequest, NoAction
from ora.orchestrator.termination import (
    TerminationAuthority,
    budget_boundary,
    environment_boundary,
    infrastructure_boundary,
    operator_boundary,
    policy_outcome_decision,
)


def test_no_action_never_owns_round_termination_or_environment_abort() -> None:
    for reason in (
        "policy_no_action",
        "model_call_limit",
        "context_compaction_rejected",
        "stagnation_guard",
    ):
        decision = policy_outcome_decision(NoAction(reason=reason, model_calls=1))
        assert decision.authority is TerminationAuthority.NONE
        assert decision.scope == "none"
        assert decision.recovery == "continue_same_observation"
        assert decision.abort_environment is False
        assert decision.persist_official_verdict is False


def test_only_declared_authorities_receive_an_explicit_boundary() -> None:
    action = policy_outcome_decision(ActionProposal(action={}, model_calls=1))
    lifecycle = policy_outcome_decision(LifecycleRequest(
        reason="policy_finished_episode", model_calls=1,
    ))
    decisions = {
        "action": action,
        "environment": environment_boundary("official_success"),
        "operator": operator_boundary(),
        "budget": budget_boundary("robot_decision_budget"),
        "lifecycle": lifecycle,
        "infrastructure": infrastructure_boundary("provider_unavailable"),
    }

    assert decisions["action"].terminates is False
    assert decisions["environment"].behavioral_boundary is True
    assert decisions["budget"].abort_environment is True
    assert decisions["lifecycle"].authority is TerminationAuthority.POLICY_LIFECYCLE
    assert decisions["lifecycle"].scope == "round"
    assert decisions["operator"].scope == "session"
    assert decisions["infrastructure"].persist_official_verdict is False
    assert decisions["infrastructure"].behavioral_boundary is False
