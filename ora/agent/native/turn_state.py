"""What one Native turn carries between its steps, and how a turn ends.

A turn handles one fresh observation. :class:`Turn` holds what stays fixed for
it -- the observation, its trace event, the request deadline and call budget.
:class:`ProviderRequest` and
:class:`ProviderReply` carry one provider exchange. Every turn returns one of
the outcome records below (or ``interruption.operator_interrupted``).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Mapping

from ora.agent.contracts import PolicySessionError


@dataclass
class Turn:
    """One observation's turn."""

    step: int
    observation: dict[str, Any]
    observation_event: Mapping[str, Any]
    current_observation_id: Any
    request_deadline: float | None
    call_limit: int
    allow_reminder: bool
    terminal_lifecycle: bool


@dataclass(frozen=True)
class ProviderRequest:
    """One persisted provider request and its ``provider_request`` event."""

    canonical: dict[str, Any]
    execution: dict[str, Any]
    wire: dict[str, Any]
    event: Mapping[str, Any]


@dataclass(frozen=True)
class ProviderReply:
    """One recorded provider reply; ``assistant`` is the message in history."""

    model_call_id: str
    assistant: dict[str, Any]
    response_event: Mapping[str, Any]
    usage: Any
    finish_reason: Any
    output_rejection: Mapping[str, Any] | None
    calls: list[Any]
    """Execution copies of the proposed tool calls."""


def request_deadline(context: Mapping[str, Any]) -> float | None:
    """Monotonic deadline for every provider request in this turn."""

    wall_time = context.get("remaining_budget", {}).get("wall_time_s")
    if wall_time is not None and (type(wall_time) not in (int, float) or not math.isfinite(wall_time) or wall_time < 0):
        raise PolicySessionError("remaining provider wall-time budget is invalid")
    return None if wall_time is None else time.monotonic() + wall_time


def call_limit(max_calls_per_observation: int, context: Mapping[str, Any]) -> int:
    """Per-observation call cap, lowered to the run's remaining model calls."""

    remaining_budget = context.get("remaining_budget")
    remaining_calls = (
        remaining_budget.get("model_calls")
        if isinstance(remaining_budget, Mapping) else None
    )
    return min(
        max_calls_per_observation,
        remaining_calls
        if isinstance(remaining_calls, int) and not isinstance(remaining_calls, bool)
        else max_calls_per_observation,
    )


def _ended(termination: str, attempts: int) -> dict[str, Any]:
    return {"submitted": False, "termination": termination, "attempts": attempts}


def context_exhausted(attempts: int) -> dict[str, Any]:
    return _ended("context_exhausted", attempts)


def no_action(attempts: int) -> dict[str, Any]:
    return _ended("policy_no_action", attempts)


def finished_episode(attempts: int) -> dict[str, Any]:
    return _ended("policy_finished_episode", attempts)


def call_limit_reached(attempts: int) -> dict[str, Any]:
    return _ended("model_call_limit", attempts)


def action_submitted(action: Any, attempts: int) -> dict[str, Any]:
    return {"submitted": True, "action": action, "attempts": attempts}
