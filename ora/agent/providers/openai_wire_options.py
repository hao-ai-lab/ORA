"""Profile-owned options of the OpenAI-compatible Chat Completions wire.

``tool_strictness: explicit_false`` sends ``"strict": false`` on every function
tool of the final wire body. A Responses-backed gateway decides strictness by
itself when the field is absent. A probe through the CLIProxyAPI Codex gateway
on 2026-10-03 (one ``geometry_query``-shaped tool, two repetitions each,
deterministic) showed:

* with no ``strict`` field, GPT-6 Astra filled every key and invented a
  placeholder ``region``;
* with ``"strict": false``, it sent only ``camera`` and ``points``;
* with nullable optional properties and no ``strict``, it sent ``null`` for
  the unused one.

So that backend decodes a strict-compatible schema as strict. EXP11A g016
(EXP11B g003) is the cost: 535 of 535 Astra ``geometry_query`` calls carried
every optional key and were refused.

The option is gated per provider profile. A DeepSeek probe of the field
(2026-10-03) could not run: the credential file named for it returned HTTP 401.
DeepSeek and every other profile without the option keep their wire bytes.

``prepare_request`` is the one place the wire body is built. As a
``profile_bound`` attribute it is the plain static projection when read from
the class, and the projection with the instance's own options when read from a
provider. The Native request assembler records exactly what ``complete`` sends,
including ``max_completion_tokens`` and ``strict``.
"""

from __future__ import annotations

import copy
import functools
from typing import Any, Callable, Mapping

#: Closed values of a profile's ``tool_strictness``.
TOOL_STRICTNESS_MODES = frozenset({"explicit_false"})
#: Wire names of streamed and replayed reasoning. The DeepSeek API uses
#: ``reasoning_content``; vLLM streams ``reasoning`` and reads only ``reasoning``
#: back from assistant history, so a self-hosted model would otherwise never
#: record its thinking and every replayed tool turn reached the chat template
#: with empty thinking. Canonical messages and traces keep ``reasoning_content``.
REASONING_FIELDS = frozenset({"reasoning_content", "reasoning"})


def project_tool_strictness(
    tools: Any, tool_strictness: str | None,
) -> Any:
    """Return the wire tools with ``strict`` stated on every function tool."""

    if tool_strictness is None:
        return tools
    if tool_strictness not in TOOL_STRICTNESS_MODES:
        raise ValueError(f"unsupported tool_strictness: {tool_strictness}")
    if not isinstance(tools, list):
        return tools
    projected = []
    for tool in tools:
        if isinstance(tool, Mapping) and isinstance(tool.get("function"), Mapping):
            tool = {**copy.deepcopy(dict(tool)), "function": {**tool["function"], "strict": False}}
        projected.append(tool)
    return projected


def project_wire_body(
    body: dict[str, Any], *, tool_strictness: str | None, reasoning_field: str = "reasoning_content",
) -> dict[str, Any]:
    """Apply the profile-owned wire options to an assembled Chat Completions body.

    "No tools" is an absent field on this wire: vLLM rejects ``"tools": []``
    with HTTP 400, which the preflight continuity-resume probe sends.
    """

    if reasoning_field not in REASONING_FIELDS:
        raise ValueError(f"unsupported reasoning_field: {reasoning_field}")
    if body.get("tools") == []:
        body.pop("tools")
    if "tools" in body:
        body["tools"] = project_tool_strictness(body["tools"], tool_strictness)
    if reasoning_field != "reasoning_content":
        for message in body.get("messages", []):
            if isinstance(message, dict) and "reasoning_content" in message:
                message[reasoning_field] = message.pop("reasoning_content")
    return body


class profile_bound:  # noqa: N801 - used as a decorator, like staticmethod
    """A wire projection read from the class is static; read from a provider it is bound to its options."""

    def __init__(self, function: Callable[..., dict[str, Any]]) -> None:
        self.function = function
        functools.update_wrapper(self, function)

    def __get__(self, instance: Any, owner: type | None = None) -> Callable[..., dict[str, Any]]:
        if instance is None:
            return self.function
        return functools.partial(
            self.function,
            output_token_parameter=instance.output_token_parameter,
            tool_strictness=instance.tool_strictness,
            reasoning_field=instance.reasoning_field,
        )


def checkpoint_request_options(*, model: str) -> dict[str, Any]:
    """Provider-owned settings for short, tool-free checkpoint calls.

    JSON mode and low reasoning are supported by these DeepSeek bindings.
    Do not infer support from the OpenAI-compatible transport alone.
    Ordinary policy requests never consult this hook.
    """
    if model in {
        "deepseek-flash", "deepseek-v4-flash", "deepseek-v4-pro",
        "deepseek-v4-flash-vision-exp",
    }:
        return {
            "reasoning_effort": "low",
            "response_format": {"type": "json_object"},
        }
    return {}


def nonnegative_count(value: Any) -> int | None:
    """A reported token count, or None when absent, boolean or negative."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value) if value >= 0 else None


def normalize_usage(value: Any) -> dict[str, Any] | None:
    """Chat Completions usage in the Harness's provider-neutral usage shape."""
    if not isinstance(value, Mapping):
        return None
    details = value.get("prompt_tokens_details")
    details = details if isinstance(details, Mapping) else {}
    completion_details = value.get("completion_tokens_details")
    completion_details = completion_details if isinstance(completion_details, Mapping) else {}
    cached_value = (
        nonnegative_count(details.get("cached_tokens"))
        if "cached_tokens" in details
        else nonnegative_count(value.get("prompt_cache_hit_tokens"))
    )
    cache_read_reported = (
        "cached_tokens" in details or "prompt_cache_hit_tokens" in value
    )
    output: dict[str, Any] = {
        "input_tokens": nonnegative_count(value.get("prompt_tokens")),
        "cached_input_tokens": cached_value,
        "output_tokens": nonnegative_count(value.get("completion_tokens")),
        "reasoning_output_tokens": nonnegative_count(completion_details.get("reasoning_tokens")),
        "total_tokens": nonnegative_count(value.get("total_tokens")),
    }
    if all(item is None for item in output.values()):
        return None
    normalized = {key: (0 if amount is None else amount) for key, amount in output.items()}
    normalized.update({
        "cache_read_input_tokens": cached_value if cache_read_reported else None,
        "cache_creation_input_tokens": None,
        "cache_read_status": "reported" if cache_read_reported else "missing",
        "cache_creation_status": "missing",
    })
    return normalized


__all__ = [
    "TOOL_STRICTNESS_MODES", "checkpoint_request_options", "nonnegative_count",
    "normalize_usage", "profile_bound", "project_tool_strictness",
]
