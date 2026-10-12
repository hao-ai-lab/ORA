"""Accumulate one Anthropic Messages response from its raw events.

``AnthropicMessagesProvider._parse_events`` reads a completed response in two
steps. This module checks the event order of a streamed response (or the shape
of one non-streamed message) and accumulates its content blocks, tool-input
fragments, usage, stop reason and response ID; ``native.py`` then decodes the
tool inputs and normalizes the assistant message. Each event kind has one
handler, applied in stream order; any violation raises before a block can
reach the canonical message.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ora.agent.contracts import PolicySessionError

from .openai_wire_options import nonnegative_count as _number


@dataclass
class AnthropicResponseEvents:
    """Everything one response's events established, in stream order."""

    blocks_by_index: dict[int, dict[str, Any]] = field(default_factory=dict)
    tool_fragments: dict[int, list[str]] = field(default_factory=dict)
    usage_values: dict[str, Any] = field(default_factory=dict)
    stop_reason: str | None = None
    response_id: str | None = None
    streaming: bool = False
    stopped_blocks: set[int] = field(default_factory=set)
    message_start_seen: bool = False
    message_stop_seen: bool = False
    non_stream_message_seen: bool = False

    def merge_usage(self, value: Any) -> None:
        if isinstance(value, Mapping):
            for key, amount in value.items():
                if _number(amount) is not None:
                    self.usage_values[key] = amount


def accumulate_anthropic_events(
    raw_events: Sequence[Mapping[str, Any]],
) -> AnthropicResponseEvents:
    """Apply every event in order, then require a streamed response to be complete."""

    state = AnthropicResponseEvents()
    for event in raw_events:
        _apply_event(state, event)
    _require_completed_stream(state)
    return state


def _apply_event(state: AnthropicResponseEvents, event: Mapping[str, Any]) -> None:
    event_type = event.get("type")
    if state.message_stop_seen:
        raise PolicySessionError(
            "Anthropic provider returned content after message_stop"
        )
    if event_type == "ping":
        return
    if event_type == "error":
        raise _stream_error(event)
    if event_type == "message_start":
        _apply_message_start(state, event)
        return
    if event_type == "content_block_start":
        _apply_content_block_start(state, event)
        return
    if event_type == "content_block_delta":
        _apply_content_block_delta(state, event)
        return
    if event_type == "content_block_stop":
        _apply_content_block_stop(state, event)
        return
    if event_type == "message_delta":
        _apply_message_delta(state, event)
        return
    if event_type == "message_stop":
        _apply_message_stop(state)
        return
    _apply_non_streaming_message(state, event, event_type)


def _stream_error(event: Mapping[str, Any]) -> PolicySessionError:
    error = event.get("error")
    error_type = (
        str(error.get("type"))
        if isinstance(error, Mapping) else "provider_error"
    )
    failure = PolicySessionError(
        f"Anthropic provider stream returned {error_type}"
    )
    setattr(failure, "retryable", error_type == "overloaded_error")
    return failure


def _apply_message_start(
    state: AnthropicResponseEvents, event: Mapping[str, Any],
) -> None:
    if state.message_start_seen or state.streaming:
        raise PolicySessionError("Anthropic message_start was duplicated")
    state.streaming = True
    state.message_start_seen = True
    message = event.get("message")
    if isinstance(message, Mapping):
        if isinstance(message.get("id"), str) and message["id"]:
            state.response_id = message["id"]
        state.merge_usage(message.get("usage"))
        if isinstance(message.get("stop_reason"), str):
            state.stop_reason = message["stop_reason"]


def _apply_content_block_start(
    state: AnthropicResponseEvents, event: Mapping[str, Any],
) -> None:
    if not state.message_start_seen:
        raise PolicySessionError(
            "Anthropic content block precedes message_start"
        )
    state.streaming = True
    index = event.get("index")
    block = event.get("content_block")
    if not isinstance(index, int) or not isinstance(block, Mapping):
        raise PolicySessionError(
            "Anthropic content_block_start is incomplete"
        )
    block_type = block.get("type")
    if block_type not in {
        "text", "thinking", "redacted_thinking", "tool_use",
    }:
        raise PolicySessionError(
            f"unsupported Anthropic assistant content block: {block_type!r}"
        )
    if index in state.blocks_by_index:
        raise PolicySessionError(
            "Anthropic assistant content block index was reused"
        )
    state.blocks_by_index[index] = copy.deepcopy(dict(block))
    if block_type == "tool_use":
        state.tool_fragments[index] = []


def _apply_content_block_delta(
    state: AnthropicResponseEvents, event: Mapping[str, Any],
) -> None:
    state.streaming = True
    index = event.get("index")
    delta = event.get("delta")
    if (
        not isinstance(index, int)
        or not isinstance(delta, Mapping)
        or index not in state.blocks_by_index
        or index in state.stopped_blocks
    ):
        raise PolicySessionError(
            "Anthropic content block delta lacks its block start"
        )
    block = state.blocks_by_index[index]
    delta_type = delta.get("type")
    if delta_type == "text_delta" and isinstance(delta.get("text"), str):
        if block.get("type") != "text":
            raise PolicySessionError("Anthropic text delta targets a non-text block")
        block["text"] = str(block.get("text", "")) + delta["text"]
    elif delta_type == "thinking_delta" and isinstance(
        delta.get("thinking"), str
    ):
        if block.get("type") != "thinking":
            raise PolicySessionError(
                "Anthropic thinking delta targets a non-thinking block"
            )
        block["thinking"] = (
            str(block.get("thinking", "")) + delta["thinking"]
        )
    elif delta_type == "signature_delta" and isinstance(
        delta.get("signature"), str
    ):
        if block.get("type") != "thinking":
            raise PolicySessionError(
                "Anthropic signature delta targets a non-thinking block"
            )
        block["signature"] = (
            str(block.get("signature", "")) + delta["signature"]
        )
    elif delta_type == "input_json_delta" and isinstance(
        delta.get("partial_json"), str
    ):
        if block.get("type") != "tool_use":
            raise PolicySessionError(
                "Anthropic tool-input delta targets a non-tool block"
            )
        state.tool_fragments.setdefault(index, []).append(
            delta["partial_json"]
        )
    else:
        raise PolicySessionError(
            f"unsupported Anthropic content block delta: {delta_type!r}"
        )


def _apply_content_block_stop(
    state: AnthropicResponseEvents, event: Mapping[str, Any],
) -> None:
    if not state.message_start_seen:
        raise PolicySessionError(
            "Anthropic content block stop precedes message_start"
        )
    state.streaming = True
    index = event.get("index")
    if (
        not isinstance(index, int)
        or index not in state.blocks_by_index
        or index in state.stopped_blocks
    ):
        raise PolicySessionError(
            "Anthropic content_block_stop lacks one active block"
        )
    state.stopped_blocks.add(index)


def _apply_message_delta(
    state: AnthropicResponseEvents, event: Mapping[str, Any],
) -> None:
    if not state.message_start_seen or state.stopped_blocks != set(state.blocks_by_index):
        raise PolicySessionError(
            "Anthropic message_delta precedes completed content blocks"
        )
    state.streaming = True
    delta = event.get("delta")
    if isinstance(delta, Mapping) and isinstance(delta.get("stop_reason"), str):
        state.stop_reason = delta["stop_reason"]
    state.merge_usage(event.get("usage"))


def _apply_message_stop(state: AnthropicResponseEvents) -> None:
    state.streaming = True
    if not state.message_start_seen or state.stopped_blocks != set(state.blocks_by_index):
        raise PolicySessionError(
            "Anthropic message_stop precedes completed content blocks"
        )
    state.message_stop_seen = True


def _apply_non_streaming_message(
    state: AnthropicResponseEvents, event: Mapping[str, Any], event_type: Any,
) -> None:
    # Non-streaming Anthropic response.
    if isinstance(event.get("content"), list):
        if state.streaming or state.non_stream_message_seen:
            raise PolicySessionError(
                "Anthropic response mixed streaming and non-streaming content"
            )
        state.non_stream_message_seen = True
        if isinstance(event.get("id"), str) and event["id"]:
            state.response_id = event["id"]
        stop = event.get("stop_reason")
        if isinstance(stop, str):
            state.stop_reason = stop
        state.merge_usage(event.get("usage"))
        for index, block in enumerate(event["content"]):
            if not isinstance(block, Mapping):
                raise PolicySessionError(
                    "Anthropic assistant content block must be an object"
                )
            state.blocks_by_index[index] = copy.deepcopy(dict(block))
        return
    raise PolicySessionError(
        f"unsupported Anthropic response event: {event_type!r}"
    )


def _require_completed_stream(state: AnthropicResponseEvents) -> None:
    if state.streaming and (
        not state.message_start_seen
        or not state.message_stop_seen
        or state.stopped_blocks != set(state.blocks_by_index)
    ):
        raise PolicySessionError(
            "Anthropic streaming response ended before all blocks completed"
        )
