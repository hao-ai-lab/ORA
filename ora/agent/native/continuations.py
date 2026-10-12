"""Harness messages that keep the policy on the same observation.

When a provider reply cannot act -- its tool batch was rejected or cut off at
the output limit, or it proposed no tool at all -- the turn continues with one
Harness user message that says why. A continuation is the derived trace event
that explains the decision plus the text the policy reads next.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class Continuation:
    event_type: str
    payload: dict[str, Any]
    content: str


def after_rejected_output(output_rejection: Mapping[str, Any]) -> Continuation:
    """The provider rejected the batch (truncated or invalid tool-input JSON)."""

    return Continuation(
        event_type="provider_output_rejected",
        payload={
            "code": output_rejection.get(
                "code", "invalid_provider_output"
            ),
            "provider_stop_reason": output_rejection.get(
                "provider_stop_reason"
            ),
            "policy_outcome": "continue_same_observation",
            "world_state_changed": False,
        },
        content=(
            "Your previous response reached the Provider output limit "
            "while proposing tools. The entire tool batch was discarded. "
            "No tool ran and the world did not change. Retry one concise "
            "valid tool call for this same observation."
            if output_rejection.get("code") == "truncated_tool_call" else
            "Your previous response ended with incomplete or invalid "
            "tool-input JSON. No tool ran and the world did not change. "
            "Retry one concise valid tool call for this same observation."
        ),
    )


def after_truncated_output() -> Continuation:
    """The reply hit the output limit before proposing any tool."""

    return Continuation(
        event_type="provider_output_truncated",
        payload={
            "tool_call_count": 0,
            "policy_outcome": "continue_same_observation",
        },
        content=(
            "Your previous response reached the Provider output limit "
            "before proposing an action. Continue working on this same "
            "observation and use the available tools."
        ),
    )


def after_no_action(mutation_tools: Iterable[str]) -> Continuation:
    """The reply ended without a tool call while the turn may continue."""

    mutations = ", ".join(mutation_tools)
    return Continuation(
        event_type="provider_no_action_continuation",
        payload={
            "tool_call_count": 0,
            "policy_outcome": "continue_same_observation",
        },
        content=(
            "No action was proposed for this observation. Continue; use "
            "read-only tools if needed, then "
            f"submit one action with {mutations}, or use an advertised "
            "lifecycle tool when its contract applies."
        ),
    )
