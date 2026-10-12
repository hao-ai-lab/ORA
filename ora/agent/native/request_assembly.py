"""Canonical Native request assembly and provider wire projection."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping

from ora.agent.contracts import PolicySessionError
from ora.agent.evidence.trajectory import canonical_bytes
from ora.agent.providers.continuation import (
    PROVIDER_CONTINUATION_MESSAGE_KEY,
    strip_provider_continuation,
)


def _prefix_relation(previous: list[Any], current: list[Any]) -> dict[str, Any]:
    common = 0
    for old, new in zip(previous, current):
        if old != new:
            break
        common += 1
    status = (
        "equal" if previous == current
        else "exact_prefix" if common == len(previous)
        else "rewritten"
    )
    return {
        "status": status,
        "common_prefix_units": common,
        "previous_units": len(previous),
        "current_units": len(current),
    }


def _rewrite_diagnosis(
    relation: Mapping[str, Any], previous: list[Any], current: list[Any],
) -> str:
    """Name the message that changed, because the request itself is discarded.

    The assembly raises before the request is emitted, so no provider_request
    or context_projected artifact records the candidate: the run dies and the
    only evidence of what was rewritten dies with it. Diagnosing one occurrence
    then means reconstructing the tenth request from the ninth by hand, and the
    thing that would have answered it in one line -- which message, and how it
    differs -- was already computed here and thrown away.
    """

    index = int(relation.get("common_prefix_units") or 0)
    detail = (
        f" at message {index} of {relation.get('previous_units')}"
        f" (now {relation.get('current_units')})"
    )
    def digest(units: list[Any]) -> str:
        if index >= len(units):
            return "absent"
        return hashlib.sha256(canonical_bytes(units[index])).hexdigest()[:16]
    return f"{detail}; was {digest(previous)}, now {digest(current)}"


def _wire_message_units(request: Mapping[str, Any]) -> list[Any]:
    """Flatten Anthropic blocks so adjacent canonical user messages compare."""

    messages = request.get("messages")
    if not isinstance(messages, list):
        raise PolicySessionError("native provider wire request lacks messages")
    anthropic_blocks = all(
        isinstance(message, Mapping)
        and set(message).issubset({"role", "content"})
        and isinstance(message.get("content"), list)
        for message in messages
    )
    if not anthropic_blocks:
        return copy.deepcopy(messages)
    units: list[Any] = []
    for message in messages:
        assert isinstance(message, Mapping)
        for block in message["content"]:
            units.append({
                "role": message.get("role"),
                "content": copy.deepcopy(block),
            })
    return units


def _cache_stable_non_message_fields(request: Mapping[str, Any]) -> dict[str, Any]:
    """Return request fields that must remain fixed inside one cache epoch.

    ``tool_choice`` is an invocation constraint, not cached prompt content: a
    request may tighten it from ``auto`` to a specific tool without changing
    the system, tools, or message prefix that the provider can cache.
    """

    return {
        str(key): copy.deepcopy(value)
        for key, value in request.items()
        if key not in {"messages", "tool_choice"}
    }


@dataclass(frozen=True)
class PreparedNativeRequest:
    canonical: Mapping[str, Any]
    wire: Mapping[str, Any]
    execution: Mapping[str, Any]
    tool_choice_evidence: Mapping[str, Any]


@dataclass
class NativeRequestAssembler:
    model: str
    reasoning_effort: str
    max_output_tokens: int
    stream: bool
    provider: Any
    _previous_canonical: dict[str, Any] | None = field(
        default=None, init=False, repr=False,
    )
    _previous_wire: dict[str, Any] | None = field(
        default=None, init=False, repr=False,
    )
    _previous_epoch: int | None = field(default=None, init=False, repr=False)
    _previous_protected_prefix_sha256: str | None = field(
        default=None, init=False, repr=False,
    )
    _previous_provider_image_window_epoch: int | None = field(
        default=None, init=False, repr=False,
    )

    def prepare(
        self, projected: Mapping[str, Any], *, tool_choice: Any = "auto",
    ) -> PreparedNativeRequest:
        messages = projected.get("messages")
        tools = projected.get("tools")
        if not isinstance(messages, list) or not isinstance(tools, list):
            raise PolicySessionError("native context projection is incomplete")
        execution_request: dict[str, Any] = {
            "model": self.model,
            "messages": copy.deepcopy(messages),
            "tools": copy.deepcopy(tools),
            "tool_choice": copy.deepcopy(tool_choice),
            "max_tokens": self.max_output_tokens,
            "stream": self.stream,
        }
        if self.reasoning_effort != "provider_default":
            execution_request["reasoning_effort"] = self.reasoning_effort
        if self.stream:
            execution_request["stream_options"] = {"include_usage": True}
        request = copy.deepcopy(execution_request)
        request["messages"] = [
            strip_provider_continuation(message)
            if isinstance(message, Mapping) else copy.deepcopy(message)
            for message in messages
        ]
        prepare_request = getattr(self.provider, "prepare_request", None)
        has_continuation = any(
            isinstance(message, Mapping)
            and PROVIDER_CONTINUATION_MESSAGE_KEY in message
            for message in messages
        )
        if has_continuation and not callable(prepare_request):
            raise PolicySessionError(
                "provider continuation state requires an explicit wire adapter"
            )
        wire = (
            prepare_request(execution_request)
            if callable(prepare_request) else copy.deepcopy(execution_request)
        )
        if not isinstance(wire, Mapping):
            raise PolicySessionError("native provider wire request must be an object")
        wire_request = copy.deepcopy(dict(wire))
        tool_choice_evidence: dict[str, Any] = {
            "requested": copy.deepcopy(tool_choice),
            "effective": copy.deepcopy(request["tool_choice"]),
            "mode": "provider_named" if isinstance(tool_choice, Mapping) else "automatic",
        }
        wire_thinking = wire_request.get("thinking")
        wire_tool_choice = wire_request.get("tool_choice")
        if (
            isinstance(tool_choice, Mapping)
            and isinstance(wire_thinking, Mapping)
            and wire_thinking.get("type") in {"adaptive", "enabled"}
            and isinstance(wire_tool_choice, Mapping)
            and wire_tool_choice.get("type") in {"tool", "any"}
        ):
            # Anthropic Messages rejects named/any tool choice while adaptive
            # thinking is enabled. Preserve reasoning and make the exact tool a
            # Runtime-validated requirement instead of emitting an invalid wire
            # request. The bounded caller still fails closed unless that exact
            # tool is returned.
            execution_request["tool_choice"] = "auto"
            request["tool_choice"] = "auto"
            wire = (
                prepare_request(execution_request)
                if callable(prepare_request) else copy.deepcopy(execution_request)
            )
            if not isinstance(wire, Mapping):
                raise PolicySessionError(
                    "native provider wire request must be an object"
                )
            wire_request = copy.deepcopy(dict(wire))
            tool_choice_evidence.update({
                "effective": "auto",
                "mode": "runtime_validated_auto",
                "reason": "provider_thinking_incompatible_with_named_choice",
            })
        manifest = projected.get("manifest")
        if (
            isinstance(manifest, dict)
            and manifest.get("context_retention") == "action_step_lossless_v4"
        ):
            self._bind_cache_stability_evidence(
                manifest=manifest,
                canonical=request,
                wire=wire_request,
            )
        return PreparedNativeRequest(
            canonical=request,
            wire=wire_request,
            execution=copy.deepcopy(execution_request),
            tool_choice_evidence=tool_choice_evidence,
        )

    def _bind_cache_stability_evidence(
        self,
        *,
        manifest: dict[str, Any],
        canonical: dict[str, Any],
        wire: dict[str, Any],
    ) -> None:
        epoch = manifest.get("cache_epoch")
        image_window = manifest.get("provider_image_window")
        image_window_epoch = (
            image_window.get("epoch")
            if isinstance(image_window, Mapping) else 0
        )
        protected_hash = manifest.get("protected_prefix_sha256")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise PolicySessionError("cache-stable projection lacks a valid cache epoch")
        if not isinstance(protected_hash, str) or len(protected_hash) != 64:
            raise PolicySessionError(
                "cache-stable projection lacks protected prefix identity"
            )
        if (
            isinstance(image_window_epoch, bool)
            or not isinstance(image_window_epoch, int)
            or image_window_epoch < 0
        ):
            raise PolicySessionError(
                "cache-stable projection lacks a valid provider image-window epoch"
            )

        canonical_hash = hashlib.sha256(canonical_bytes(canonical)).hexdigest()
        wire_hash = hashlib.sha256(canonical_bytes(wire)).hexdigest()
        previous_hash = (
            hashlib.sha256(canonical_bytes(self._previous_canonical)).hexdigest()
            if self._previous_canonical is not None else None
        )
        if self._previous_canonical is None or self._previous_wire is None:
            request_relation = "initial"
            canonical_relation = {
                "status": "initial",
                "common_prefix_units": 0,
                "previous_units": 0,
                "current_units": len(canonical.get("messages", [])),
            }
            wire_relation = {
                "status": "initial",
                "common_prefix_units": 0,
                "previous_units": 0,
                "current_units": len(_wire_message_units(wire)),
            }
        else:
            if self._previous_provider_image_window_epoch is None:
                raise PolicySessionError(
                    "cache-stable projection lost provider image-window state"
                )
            if image_window_epoch not in {
                self._previous_provider_image_window_epoch,
                self._previous_provider_image_window_epoch + 1,
            }:
                raise PolicySessionError(
                    "cache_prefix_rewritten: provider image-window epoch is not contiguous"
                )
            image_window_declares_transition = (
                isinstance(image_window, Mapping)
                and image_window.get("epoch_transition") is True
            )
            if (
                image_window_epoch == self._previous_provider_image_window_epoch + 1
            ) != image_window_declares_transition:
                raise PolicySessionError(
                    "cache_prefix_rewritten: provider image-window transition evidence differs"
                )
            if protected_hash != self._previous_protected_prefix_sha256:
                raise PolicySessionError(
                    "cache_prefix_rewritten: protected Native context changed inside one session"
                )
            if _cache_stable_non_message_fields(
                self._previous_canonical
            ) != _cache_stable_non_message_fields(canonical):
                raise PolicySessionError(
                    "cache_prefix_rewritten: canonical request fields changed inside one session"
                )
            if _cache_stable_non_message_fields(
                self._previous_wire
            ) != _cache_stable_non_message_fields(wire):
                raise PolicySessionError(
                    "cache_prefix_rewritten: provider wire fields changed inside one session"
                )
            previous_messages = self._previous_canonical.get("messages")
            current_messages = canonical.get("messages")
            if not isinstance(previous_messages, list) or not isinstance(
                current_messages, list
            ):
                raise PolicySessionError("canonical Native request lacks messages")
            canonical_relation = _prefix_relation(previous_messages, current_messages)
            wire_relation = _prefix_relation(
                _wire_message_units(self._previous_wire),
                _wire_message_units(wire),
            )
            if epoch == self._previous_epoch:
                image_window_advanced = (
                    self._previous_provider_image_window_epoch is not None
                    and image_window_epoch
                    == self._previous_provider_image_window_epoch + 1
                    and isinstance(image_window, Mapping)
                    and image_window.get("epoch_transition") is True
                )
                if canonical_relation["status"] not in {"equal", "exact_prefix"} and not image_window_advanced:
                    raise PolicySessionError(
                        "cache_prefix_rewritten: canonical history changed "
                        "without an epoch"
                        + _rewrite_diagnosis(
                            canonical_relation, previous_messages, current_messages,
                        )
                    )
                if wire_relation["status"] not in {"equal", "exact_prefix"} and not image_window_advanced:
                    raise PolicySessionError(
                        "cache_prefix_rewritten: provider wire history changed "
                        "without an epoch"
                        + _rewrite_diagnosis(
                            wire_relation,
                            _wire_message_units(self._previous_wire),
                            _wire_message_units(wire),
                        )
                    )
                request_relation = (
                    "provider_image_window_epoch"
                    if image_window_advanced else "append_only"
                )
            elif epoch == self._previous_epoch + 1:
                transition = manifest.get("epoch_transition")
                if not isinstance(transition, Mapping) or transition.get(
                    "to_epoch"
                ) != epoch:
                    raise PolicySessionError(
                        "cache_prefix_rewritten: cache epoch changed without transition evidence"
                    )
                request_relation = "projection_epoch"
            else:
                raise PolicySessionError(
                    "cache_prefix_rewritten: cache epoch sequence is not contiguous"
                )

        manifest.update({
            "previous_request_sha256": previous_hash,
            "canonical_request_sha256": canonical_hash,
            "provider_wire_request_sha256": wire_hash,
            "canonical_request_utf8_bytes": len(canonical_bytes(canonical)),
            "provider_wire_request_utf8_bytes": len(canonical_bytes(wire)),
            "request_relation": request_relation,
            "canonical_prefix_relation": canonical_relation,
            "provider_wire_prefix_relation": wire_relation,
        })
        self._previous_canonical = copy.deepcopy(canonical)
        self._previous_wire = copy.deepcopy(wire)
        self._previous_epoch = epoch
        self._previous_provider_image_window_epoch = image_window_epoch
        self._previous_protected_prefix_sha256 = protected_hash
