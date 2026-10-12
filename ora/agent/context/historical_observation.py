"""Recover old public observation bodies without rewriting causal outcomes."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import TYPE_CHECKING, Any, Mapping

from ora.agent.contracts import PolicySessionError
from ora.agent.memory.artifacts import (
    ContextArtifactCatalog, recoverable_artifact_pointer,
)

if TYPE_CHECKING:
    from .inline_media import InlineMediaCache


def materialize(
    message: dict[str, Any], *, catalog: ContextArtifactCatalog | InlineMediaCache,
    round_index: int | None, action_step: int | None,
) -> dict[str, Any] | None:
    """Store only the already public JSON envelope, never observation metadata.

    ``catalog`` may be the projection's verified-admission cache; both expose
    the same ``put_bytes`` contract and return the same public reference.
    """
    content = message.get("content")
    if not (
        message.get("role") == "user"
        and isinstance(message.get("_observation"), Mapping)
        and isinstance(content, list) and content
        and isinstance(content[0], Mapping) and content[0].get("type") == "text"
    ):
        return None
    original = content[0].get("text")
    if not isinstance(original, str):
        return None
    try:
        body = json.loads(original)
    except ValueError:
        return None
    if not isinstance(body, dict) or not isinstance(body.get("current_actual"), dict):
        return None
    ref = catalog.put_bytes(
        original.encode("utf-8"), kind="context_observation", suffix=".json",
        media_type="application/json", round_index=round_index, action_step=action_step,
    )
    message["_context_observation_ref"] = ref
    message["_context_observation_original"] = original
    return ref


def _marker(original: str, ref: Mapping[str, Any]) -> str:
    body = json.loads(original)
    # Keep comparison, last_action_effect, terminal verdict and lifecycle notices
    # verbatim. Only the large measured robot/camera body is replaced; its exact
    # original (including all receipts) is available through the same pointer.
    actual = body["current_actual"]
    body["current_actual"] = {
        key: copy.deepcopy(actual[key])
        for key in ("step", "actual_sequence", "observation_id") if key in actual
    }
    body["recoverable_observation"] = recoverable_artifact_pointer(ref)
    body["observation_externalized"] = True
    return json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def externalize(
    messages: list[dict[str, Any]], *, eligible_indexes: set[int],
) -> tuple[list[dict[str, Any]], list[int], list[dict[str, Any]]]:
    # Copy-on-write: only a replaced message (and its content list) is copied.
    output = list(messages)
    changed: list[int] = []
    refs: list[dict[str, Any]] = []
    for index in sorted(eligible_indexes):
        message = output[index]
        ref = message.get("_context_observation_ref")
        original = message.get("_context_observation_original")
        if not isinstance(ref, Mapping) or not isinstance(original, str):
            continue
        replacement = _marker(original, ref)
        if len(replacement.encode("utf-8")) >= len(original.encode("utf-8")):
            continue
        message = dict(message)
        message["content"] = list(message["content"])
        output[index] = message
        message["content"][0] = {"type": "text", "text": replacement}
        message["_lossless_observation_externalized"] = True
        changed.append(index)
        refs.append(copy.deepcopy(dict(ref)))
    return output, changed, refs


def invariant_block(message: Mapping[str, Any]) -> dict[str, Any] | None:
    """Verify the original bytes or exact replacement before normalizing."""
    ref = message.get("_context_observation_ref")
    if not isinstance(ref, Mapping):
        return None
    original = message.get("_context_observation_original")
    if not isinstance(original, str) or hashlib.sha256(
        original.encode("utf-8")
    ).hexdigest() != ref.get("sha256") or len(original.encode("utf-8")) != ref.get("bytes"):
        raise PolicySessionError("observation Artifact identity drift")
    expected = _marker(original, ref) if message.get(
        "_lossless_observation_externalized"
    ) is True else original
    if message["content"][0] != {"type": "text", "text": expected}:
        raise PolicySessionError("observation body differs from recoverable source")
    return {"type": "text", "text": original}
