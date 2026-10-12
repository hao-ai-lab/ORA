"""What one Native provider reply must satisfy, and how its call is accounted.

A reply's completion reason must agree with the tool calls it carries, so a
truncated or rejected batch can never reach dispatch. Every accepted or
rejected reply then adds one call record and, when the provider reported
usage, one cumulative usage event to the session manifest.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

from ora.agent.contracts import PolicySessionError
from ora.agent.evidence.trajectory import canonical_bytes


def check_wire_echo(response: Mapping[str, Any], wire_request: Mapping[str, Any]) -> None:
    """A provider that echoes its wire request must have sent the traced one."""

    returned_wire = response.get("wire_request")
    if returned_wire is not None and canonical_bytes(returned_wire) != canonical_bytes(wire_request):
        raise PolicySessionError(
            "native provider sent a wire request different from its traced projection"
        )


def reply_message(
    response: Mapping[str, Any],
) -> tuple[dict[str, Any], Mapping[str, Any] | None]:
    """A copy of the assistant message and the validated output rejection."""

    message = response.get("message")
    if not isinstance(message, Mapping) or message.get("role") != "assistant":
        raise PolicySessionError("native provider response lacks an assistant message")
    assistant = copy.deepcopy(dict(message))
    output_rejection = response.get("output_rejection")
    if output_rejection is not None and not isinstance(
        output_rejection, Mapping,
    ):
        raise PolicySessionError(
            "native provider output rejection must be an object"
        )
    return assistant, output_rejection


def response_payload(
    response: Mapping[str, Any], *, assistant: dict[str, Any], usage: Any,
    output_rejection: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """The ``provider_response`` trace payload."""

    return {
        "message": assistant,
        "usage": usage,
        "finish_reason": response.get("finish_reason"),
        "attempts": response.get("attempts", 1),
        "accepted": output_rejection is None,
        **({
            "output_rejection": copy.deepcopy(dict(output_rejection)),
        } if isinstance(output_rejection, Mapping) else {}),
    }


def error_payload(exc: BaseException) -> dict[str, Any]:
    """The ``provider_error`` trace payload of a failed provider request."""

    failure = getattr(exc, "provider_failure", {})
    if not isinstance(failure, Mapping):
        failure = {}
    return {
        "exception_type": type(exc).__name__,
        "cause_type": type(exc.__cause__).__name__ if exc.__cause__ else None,
        "attempts": failure.get("attempts"),
        "response_started": failure.get("response_started"),
        "accepted": False,
    }


def execution_tool_calls(assistant: Mapping[str, Any]) -> list[Any]:
    """Copies of the proposed calls; provider history remains exact."""

    raw_calls = assistant.get("tool_calls")
    return copy.deepcopy(raw_calls) if isinstance(raw_calls, list) else []


def check_reply_completion(
    finish_reason: Any, calls: list[Any], output_rejection: Mapping[str, Any] | None,
) -> None:
    """A completion reason must agree with the tool calls the reply carries."""

    if output_rejection is not None and calls:
        raise PolicySessionError(
            "rejected native provider output cannot expose tool calls"
        )
    if (
        finish_reason == "invalid_tool_input"
        and output_rejection is None
    ):
        raise PolicySessionError(
            "invalid_tool_input completion lacks rejection evidence"
        )
    if finish_reason == "length" and calls:
        raise PolicySessionError(
            "native provider response contains a truncated tool call"
        )
    if calls and finish_reason != "tool_calls":
        raise PolicySessionError(
            "native provider returned tool calls without a tool_calls completion"
        )
    if finish_reason == "tool_calls" and not calls:
        raise PolicySessionError(
            "native provider declared tool_calls without a complete tool call"
        )


def call_record(
    *,
    model_call_id: str,
    request: Mapping[str, Any],
    response: Mapping[str, Any],
    usage: Any,
    context_projection: Mapping[str, Any],
    output_rejection: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """The session-manifest record of one provider call."""

    encoded_request = canonical_bytes(request)
    started = response.get("started_unix_s")
    finished = response.get("finished_unix_s")
    return {
        "turn_id": model_call_id,
        "message_kind": "native_projected_request",
        "project_message_chars": len(encoded_request.decode("utf-8")),
        "project_message_bytes": len(encoded_request),
        "started_unix_s": started,
        "finished_unix_s": finished,
        "latency_s": (
            round(float(finished) - float(started), 6)
            if isinstance(started, (int, float)) and isinstance(finished, (int, float))
            else None
        ),
        "usage": usage,
        "context_projection": context_projection,
        "provider_output_status": (
            "accepted" if output_rejection is None else "rejected"
        ),
    }


def add_usage(
    usage: Any,
    *,
    model_call_id: str,
    reported_totals: dict[str, int],
    usage_events: list[dict[str, Any]],
    context_window_tokens: int,
) -> None:
    """Add reported integer usage to the running totals and log one event."""

    if not isinstance(usage, Mapping):
        return
    for key in reported_totals:
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            reported_totals[key] += value
    usage_events.append({
        "sequence": len(usage_events),
        "turn_id": model_call_id,
        "last": dict(usage),
        "total": dict(reported_totals),
        "model_context_window": context_window_tokens,
    })
