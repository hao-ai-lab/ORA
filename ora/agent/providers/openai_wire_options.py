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

So that backend decodes a strict-compatible schema as strict. One recorded batch
shows the cost: 535 of 535 ``geometry_query`` calls carried
every optional key and were refused.

The option is gated per provider profile. A DeepSeek probe of the field
(2026-10-03) could not run: the credential file named for it returned HTTP 401.
DeepSeek and every other profile without the option keep their wire bytes.

``request_options`` is a closed set of profile-owned request fields merged into
the final wire body: sampling (``temperature``, ``top_p``, ``top_k``), vLLM's
``chat_template_kwargs`` and OpenRouter's ``reasoning`` and ``provider``
routing objects. NVIDIA Nemotron 3.5 Super VL switches thinking on with
``chat_template_kwargs: {"enable_thinking": true}`` and its hosted endpoints do
not take ``reasoning_effort``; each model card names its own sampling. A field
the request already carries is refused rather than overwritten, and a profile
without the option keeps its wire bytes.

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
#: Closed keys of a profile's ``request_options`` (see the module docstring).
REQUEST_OPTION_FIELDS = frozenset({
    "temperature", "top_p", "top_k", "chat_template_kwargs", "reasoning", "provider",
})


def _json_value(value: Any) -> bool:
    if value is None or isinstance(value, (bool, str)):
        return True
    if isinstance(value, (int, float)):
        return value == value and value not in {float("inf"), float("-inf")}
    if isinstance(value, list):
        return all(_json_value(item) for item in value)
    if isinstance(value, Mapping):
        return all(isinstance(key, str) and _json_value(item) for key, item in value.items())
    return False


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_request_options(value: Any) -> dict[str, Any] | None:
    """Return a copy of valid profile ``request_options``; raise ``ValueError`` otherwise."""

    if value is None:
        return None
    if not isinstance(value, Mapping) or not value:
        raise ValueError("request_options must be a non-empty mapping")
    unknown = sorted(set(value) - REQUEST_OPTION_FIELDS)
    if unknown:
        raise ValueError(f"unsupported request_options fields: {unknown}")
    if not _json_value(value):
        raise ValueError("request_options must hold finite JSON values")
    if "temperature" in value and not (_number(value["temperature"]) and 0 <= value["temperature"] <= 2):
        raise ValueError("request_options.temperature must be a number in 0..2")
    if "top_p" in value and not (_number(value["top_p"]) and 0 < value["top_p"] <= 1):
        raise ValueError("request_options.top_p must be a number in (0, 1]")
    top_k = value.get("top_k")
    if "top_k" in value and not (isinstance(top_k, int) and not isinstance(top_k, bool)
                                 and (top_k == -1 or top_k >= 1)):
        raise ValueError("request_options.top_k must be -1 or a positive integer")
    for field in ("chat_template_kwargs", "reasoning", "provider"):
        if field in value and (not isinstance(value[field], Mapping) or not value[field]):
            raise ValueError(f"request_options.{field} must be a non-empty mapping")
    return copy.deepcopy(dict(value))


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
    request_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the profile-owned wire options to an assembled Chat Completions body.

    "No tools" is an absent field on this wire: vLLM rejects ``"tools": []``
    with HTTP 400, which the preflight continuity-resume probe sends.
    """

    if reasoning_field not in REASONING_FIELDS:
        raise ValueError(f"unsupported reasoning_field: {reasoning_field}")
    options = validate_request_options(request_options) or {}
    collisions = sorted(set(options) & set(body))
    if collisions:
        raise ValueError(f"request_options collide with request fields: {collisions}")
    body.update(options)
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
            request_options=instance.request_options,
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
    "REQUEST_OPTION_FIELDS", "TOOL_STRICTNESS_MODES", "validate_request_options", "checkpoint_request_options", "nonnegative_count",
    "normalize_usage", "profile_bound", "project_tool_strictness",
]
