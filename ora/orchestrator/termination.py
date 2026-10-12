"""Explicit authority and blast radius for ending an environment round."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from ora.agent.contracts import ActionProposal, LifecycleRequest, PolicyOutcome


class TerminationAuthority(str, Enum):
    """The only principals allowed to end a round or session."""

    NONE = "none"
    ENVIRONMENT = "environment"
    OPERATOR = "operator"
    EXPERIMENT_BUDGET = "experiment_budget"
    POLICY_LIFECYCLE = "policy_lifecycle"
    INFRASTRUCTURE = "infrastructure"


TerminationScope = Literal["none", "round", "session"]
RecoveryAction = Literal["continue_same_observation", "none"]


@dataclass(frozen=True)
class TerminationDecision:
    """One auditable decision about whether a policy outcome owns termination."""

    reason: str
    authority: TerminationAuthority
    scope: TerminationScope
    recovery: RecoveryAction = "none"
    abort_environment: bool = False
    persist_official_verdict: bool = False
    behavioral_boundary: bool = False

    @property
    def terminates(self) -> bool:
        return self.scope != "none"

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "ora.termination-decision.v1",
            "reason": self.reason,
            "authority": self.authority.value,
            "scope": self.scope,
            "recovery": self.recovery,
            "abort_environment": self.abort_environment,
            "persist_official_verdict": self.persist_official_verdict,
            "behavioral_boundary": self.behavioral_boundary,
        }


def environment_boundary(reason: str) -> TerminationDecision:
    return TerminationDecision(
        reason=reason,
        authority=TerminationAuthority.ENVIRONMENT,
        scope="round",
        persist_official_verdict=True,
        behavioral_boundary=True,
    )


def budget_boundary(reason: str) -> TerminationDecision:
    return TerminationDecision(
        reason=reason,
        authority=TerminationAuthority.EXPERIMENT_BUDGET,
        scope="round",
        abort_environment=True,
        persist_official_verdict=True,
        behavioral_boundary=True,
    )


def operator_boundary(reason: str = "operator_interrupt") -> TerminationDecision:
    return TerminationDecision(
        reason=reason,
        authority=TerminationAuthority.OPERATOR,
        scope="session",
    )


def infrastructure_boundary(
    reason: str, *, scope: TerminationScope = "session",
) -> TerminationDecision:
    return TerminationDecision(
        reason=reason,
        authority=TerminationAuthority.INFRASTRUCTURE,
        scope=scope,
    )


def policy_outcome_decision(
    outcome: PolicyOutcome, *, infrastructure_abort: bool = False,
) -> TerminationDecision:
    """Keep ordinary no-action outcomes below the environment lifecycle boundary.

    ``infrastructure_abort`` marks a lifecycle request that is an abort the
    Harness admitted on this round's recorded infrastructure evidence under
    the v2 finish contract (``policy_finish.infrastructure_abort``): it ends
    only this round, records no official verdict and is no behavioural
    sample. Every other lifecycle request keeps the policy-lifecycle boundary.
    """

    if isinstance(outcome, ActionProposal):
        return TerminationDecision(
            reason="action_proposed",
            authority=TerminationAuthority.NONE,
            scope="none",
        )
    reason = str(getattr(outcome, "reason", "policy_no_action"))
    if isinstance(outcome, LifecycleRequest) and infrastructure_abort:
        return TerminationDecision(
            reason=reason,
            authority=TerminationAuthority.INFRASTRUCTURE,
            scope="round",
            abort_environment=True,
        )
    if isinstance(outcome, LifecycleRequest):
        return TerminationDecision(
            reason=reason,
            authority=TerminationAuthority.POLICY_LIFECYCLE,
            scope="round",
            abort_environment=True,
            persist_official_verdict=True,
            behavioral_boundary=True,
        )
    if reason == "operator_interrupt":
        return operator_boundary(reason)
    if reason == "context_exhausted":
        # A genuinely exhausted PolicySession cannot safely manufacture an
        # Environment verdict. It stops infrastructure, not the robot task.
        return infrastructure_boundary(reason)
    return TerminationDecision(
        reason=reason,
        authority=TerminationAuthority.NONE,
        scope="none",
        recovery="continue_same_observation",
    )


__all__ = [
    "TerminationAuthority",
    "TerminationDecision",
    "budget_boundary",
    "environment_boundary",
    "infrastructure_boundary",
    "operator_boundary",
    "policy_outcome_decision",
]
