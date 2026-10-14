"""Context-source records a typed-slot projection carries into its manifest.

The Native session registers its static sources (system, project and task)
once. Each provider request then appends the per-request
sources -- an optional rollout checkpoint, the projected history, the current
observation envelope and the tool schemas -- with the byte count and SHA-256
the request actually carries, so the trace states which slot every request
byte came from.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

from ora.agent.evidence.trajectory import canonical_bytes

from .constants import COMPOSITION_CONTEXT_CAPABILITIES, NATIVE_CAUSAL_SLOTS
from .operator_messages import OPERATOR_MESSAGE_PREFIX

_CHECKPOINT_PREFIX = "<rollout_checkpoint>\n"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def annotate_context_sources(
    projected: dict[str, Any],
    *,
    composition: str | None,
    observation: Mapping[str, Any],
    tool_schemas: Sequence[Mapping[str, Any]],
    context_sources: Sequence[Mapping[str, Any]],
) -> None:
    """Add the per-request source records to ``projected["manifest"]`` in place.

    The manifest object is the one the request assembler and the call record
    read afterwards, so it is edited rather than replaced. Compositions other
    than the typed causal slots carry no source records.
    """

    if composition in NATIVE_CAUSAL_SLOTS:
        _annotate_v3(
            projected, observation=observation, tool_schemas=tool_schemas,
            context_sources=context_sources,
        )
        if composition in COMPOSITION_CONTEXT_CAPABILITIES:
            _annotate_round_static_facts(projected)
    if composition in NATIVE_CAUSAL_SLOTS:
        _annotate_operator_messages(projected)


def _annotate_operator_messages(projected: dict[str, Any]) -> None:
    records = projected["manifest"]["context_sources"]
    order = max(int(item["order"]) for item in records) + 1
    for message in projected["messages"]:
        content = message.get("content")
        if message.get("role") == "user" and isinstance(content, str) and content.startswith(OPERATOR_MESSAGE_PREFIX):
            data = canonical_bytes(message)
            records.append({
                "kind": "operator_message", "role": "user", "order": order,
                "status": "protected_human_guidance", "bytes": len(data),
                "sha256": _sha256(data),
            })
            order += 1


def _round_static_facts_slot(message: Mapping[str, Any]) -> bool:
    content = message.get("content")
    if message.get("role") != "user" or not isinstance(content, list) or len(content) != 1:
        return False
    text = content[0].get("text") if isinstance(content[0], Mapping) else None
    try:
        body = json.loads(text) if isinstance(text, str) else None
    except json.JSONDecodeError:
        return False
    slot = body.get("round_static_facts") if isinstance(body, Mapping) else None
    return isinstance(slot, Mapping) and "current_actual" in slot


def _annotate_round_static_facts(projected: dict[str, Any]) -> None:
    """``native_slots_v4``: the protected round slots the history hash leaves out."""

    records = projected["manifest"]["context_sources"]
    order = max(int(item["order"]) for item in records) + 1
    for message in projected["messages"]:
        if _round_static_facts_slot(message):
            data = canonical_bytes(message)
            records.append({
                "kind": "round_static_facts", "role": "user", "order": order,
                "status": "protected_round_slot", "bytes": len(data),
                "sha256": _sha256(data),
            })
            order += 1


def _rollout_checkpoint_message(
    messages: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    """The protected rollout checkpoint user message, when one is projected."""

    return next((
        message for message in messages
        if (
            message.get("role") == "user"
            and isinstance(message.get("content"), str)
            and message["content"].startswith(_CHECKPOINT_PREFIX)
        )
    ), None)


def _annotate_v3(
    projected: dict[str, Any], *, observation: Mapping[str, Any],
    tool_schemas: Sequence[Mapping[str, Any]],
    context_sources: Sequence[Mapping[str, Any]],
) -> None:
    causal_envelope = observation["_causal_observation"]
    causal_bytes = canonical_bytes(causal_envelope)
    tool_bytes = canonical_bytes(tool_schemas)
    source_records = sorted(context_sources, key=lambda item: int(item["order"]))
    # The optional checkpoint shifts every later order by one.
    checkpoint_message = _rollout_checkpoint_message(projected["messages"])
    checkpoint_source = []
    if checkpoint_message is not None:
        checkpoint_bytes = canonical_bytes(checkpoint_message)
        checkpoint_source = [{
            "kind": "rollout_context_checkpoint", "role": "user",
            "order": 3,
            "status": "protected_current",
            "bytes": len(checkpoint_bytes),
            "sha256": _sha256(checkpoint_bytes),
        }]
    dynamic_order = 3 + int(checkpoint_message is not None)
    source_records.extend(checkpoint_source)
    source_records.extend([
        {
            "kind": "history_projection", "role": "mixed",
            "order": dynamic_order,
            "status": "projected",
            "bytes": projected["manifest"]["history_utf8_bytes"],
            "sha256": projected["manifest"]["history_sha256"],
        },
        {
            "kind": "causal_observation", "role": "user",
            "order": dynamic_order + 1,
            "status": "current_allowlisted", "bytes": len(causal_bytes),
            "sha256": _sha256(causal_bytes),
        },
        {
            "kind": "tool_schemas", "role": "provider_field",
            "order": dynamic_order + 2,
            "status": "canonical", "bytes": len(tool_bytes),
            "sha256": _sha256(tool_bytes),
        },
    ])
    projected["manifest"]["context_sources"] = source_records
    projected["manifest"]["episode_task_protected"] = True
