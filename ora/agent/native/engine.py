"""Mutable state owned by one Native policy session."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _zero_usage() -> dict[str, int]:
    return {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_output_tokens": 0,
        "total_tokens": 0,
    }


@dataclass
class NativeSessionEngine:
    """Conversation, usage and causal lifecycle state; no environment writer."""

    messages: list[dict[str, Any]] = field(default_factory=list)
    context_sources: list[dict[str, Any]] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    usage_events: list[dict[str, Any]] = field(default_factory=list)
    reported_totals: dict[str, int] = field(default_factory=_zero_usage)
    call_sequence: int = 0
    closed: bool = False
    interrupted: bool = False
    pending_receipt: dict[str, Any] | None = None
    pending_receipt_body: dict[str, Any] | None = None
    last_actual_snapshot: dict[str, Any] | None = None
    actual_sequence: int = -1
    #: ``native_slots_v4``: the current round's slot fields, and a slot message
    #: built with an observation but not yet appended after it.
    round_static_facts: dict[str, Any] | None = None
    pending_round_static_facts: dict[str, Any] | None = None
    last_event_id: str | None = None
    last_action_event_id: str | None = None
    last_action_step: int | None = None
    last_action_id: str | None = None
    last_action_public_name: str | None = None
    last_fresh_observation_event_id: str | None = None
    system_text: str | None = None
    bound_task_instruction: str | None = None
    task_sha256: str | None = None
    started: bool = False
