"""Direct model providers and deterministic native-loop fake.

The native loop owns one OpenAI/ChatML-shaped canonical message IR.  Provider
adapters translate only at the wire boundary, preserving one context, tool and
trace implementation for every protocol.
"""

from __future__ import annotations

from ora.agent.providers.anthropic_history import (
    _has_complete_anthropic_thinking_signature,
)

import base64
import copy
import hashlib
import json
import re
import ssl
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit

from ora.agent.contracts import PolicySessionError

from .openai_wire_options import (
    REASONING_FIELDS, TOOL_STRICTNESS_MODES, checkpoint_request_options, nonnegative_count as _number,
    normalize_usage, profile_bound, project_tool_strictness, project_wire_body,
)
from .anthropic_events import AnthropicResponseEvents, accumulate_anthropic_events
from .stream_failures import http_status_failure, stream_failure_after_content
from ora.agent.evidence.trajectory import canonical_bytes
from ora.agent.providers.continuation import (
    PROVIDER_CONTINUATION_MESSAGE_KEY,
    ProviderContinuationState,
    canonical_assistant_sha256,
)
from ora.agent.providers.attempts import (
    _StreamInterrupted, credential_echo_failure, partial_credential_echo, record_attempt_failure,
)
from ora.agent.providers.tool_names import (
    tool_alias, openai_tool_names, project_openai_tool_names,
)
from ora.agent.providers.recovery import (
    RECOVERY_POLICY, RecoveryWindow, ProviderRecoveryDeadline, bounded_request,
    transient_transport,
)


_DATA_URL = re.compile(r"^data:([^;,]+);base64,(.*)$", re.DOTALL)
_OPENAI_CONTINUATION_KIND = "openai_chat_reasoning_content_v1"
_KIMI_CODING_BASE_URLS = {
    "https://api.kimi.com/coding",
    "https://api.kimi.com/coding/v1",
    "https://api.kimi.com/coding/v1/messages",
}
_KIMI_TRANSIENT_SIGNATURE_REJECTION = (
    "malformed encrypted reasoning content: invalid base64url encoding"
)
# visible_only_v1 replay composes only at the commits that composed it.
_ANTHROPIC_CONTINUATION_REPLAY_MODES = {"exact_v1"}


def _retryable_transport_error(exc: Exception) -> bool:
    return transient_transport(exc)


def _reject_truncated_tools(
    message: Mapping[str, Any], *, provider_stop_reason: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reject the entire capped tool turn, preserving its wire as evidence.

    Even syntactically complete arguments cannot turn an output-limit stop into
    a completed proposal. Do not replay reasoning signed against discarded
    tools or synthesize a signature for the edited assistant message.
    """
    calls = message["tool_calls"]
    rejected = []
    for index, call in enumerate(calls):
        function = call.get("function", {})
        fragment = function.get("arguments", "")
        rejected.append({"index": index, "tool_call_id": call.get("id"),
            "tool_name": function.get("name"), "fragment_chars": len(fragment),
            "fragment_sha256": hashlib.sha256(fragment.encode("utf-8")).hexdigest(),
            "reason": "provider_output_limit"})
    return {"role": "assistant", "content": message.get("content", "")}, {
        "code": "truncated_tool_call", "provider_stop_reason": provider_stop_reason,
        "rejected_tool_blocks": rejected, "discarded_tool_block_count": len(calls),
        "policy_outcome": "continue_same_observation", "world_state_changed": False,
    }


def _record_provider_failure(exc: Exception, attempts: int) -> None:
    # Bounded metadata only: exception messages can contain endpoint/credential
    # details. The loop records this alongside the already sealed request ID.
    setattr(exc, "provider_failure", {
        "exception_type": type(exc).__name__,
        "cause_type": type(exc.__cause__).__name__ if exc.__cause__ else None,
        "attempts": attempts,
        "response_started": bool(getattr(exc, "response_started", False)),
    })


def _echo_failure(cause: Exception, attempts: int) -> PolicySessionError:
    failure = credential_echo_failure(cause)
    _record_provider_failure(failure, attempts)
    return failure


class ProviderClosed(PolicySessionError):
    """A closed adapter refuses to start or retry a request.

    ``close()`` is how an operator interrupt reaches an in-flight turn: the
    Supervisor's stop handler closes the PolicySession, which closes its
    provider from another thread. A transport error raised by that close is
    the stop itself, not a transient fault, so no retry may follow it and no
    new request may start after it.
    """

    retryable = False

    def __init__(self) -> None:
        super().__init__("native provider is closed; no request was sent")


def _anthropic_tool_alias(name: str) -> str:
    """Encode canonical dotted names for strict Anthropic-compatible gateways."""

    return tool_alias(name)


def _anthropic_tool_names(request: Mapping[str, Any]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    raw_tools = request.get("tools", [])
    if not isinstance(raw_tools, list):
        return aliases
    for item in raw_tools:
        function = item.get("function") if isinstance(item, Mapping) else None
        name = function.get("name") if isinstance(function, Mapping) else None
        if not isinstance(name, str):
            continue
        alias = _anthropic_tool_alias(name)
        previous = aliases.setdefault(alias, name)
        if previous != name:
            raise PolicySessionError("Anthropic tool-name alias collision")
    return aliases


@dataclass
class FakeProvider:
    """Return scripted assistant messages without network access."""

    responses: Sequence[Mapping[str, Any]]
    requests: list[dict[str, Any]] = field(default_factory=list)
    index: int = 0
    closed: bool = False
    protocol: str = "fake"

    def complete(
        self, *, request: Mapping[str, Any], model_call_id: str
    ) -> dict[str, Any]:
        if self.closed:
            raise PolicySessionError("FakeProvider is closed")
        if self.index >= len(self.responses):
            raise PolicySessionError("FakeProvider response script is exhausted")
        self.requests.append(copy.deepcopy(dict(request)))
        scripted = copy.deepcopy(dict(self.responses[self.index]))
        self.index += 1
        message = scripted.get("message", scripted)
        if not isinstance(message, Mapping):
            raise PolicySessionError("FakeProvider response message must be an object")
        usage = scripted.get("usage") if "message" in scripted else None
        finish_reason = scripted.get("finish_reason")
        if finish_reason is None:
            finish_reason = (
                "tool_calls"
                if isinstance(message.get("tool_calls"), list)
                and message["tool_calls"]
                else "stop"
            )
        return {
            "message": dict(message),
            "usage": usage,
            "finish_reason": finish_reason,
            "raw_events": [{"fake": True, "index": self.index - 1}],
            **({
                "output_rejection": copy.deepcopy(scripted["output_rejection"]),
            } if isinstance(scripted.get("output_rejection"), Mapping) else {}),
            "model_call_id": model_call_id,
            "started_unix_s": time.time(),
            "finished_unix_s": time.time(),
            "ttft_s": 0.0,
            "attempts": 1,
        }

    def close(self) -> None:
        self.closed = True


@dataclass
class OpenAICompatibleProvider:
    """Small direct Chat Completions adapter with bounded safe retries."""

    base_url: str
    api_key: str
    timeout_seconds: float = 900.0
    max_retries: int = 2
    preflight_probe: bool = field(default=False, repr=False, kw_only=True)
    recovery_policy: str = field(default="legacy_v1", kw_only=True)
    recovery_window_s: float | None = field(default=None, kw_only=True)
    request_deadline_monotonic: float | None = field(default=None, repr=False, kw_only=True)
    attempt_observer: Callable[[Mapping[str, Any]], None] | None = field(default=None, repr=False, kw_only=True)
    client: Any = None
    _owns_client: bool = field(default=False, init=False)
    closed: bool = field(default=False, init=False)
    protocol: str = field(default="openai_chat_completions", init=False)
    output_token_parameter: str = field(default="max_tokens", kw_only=True)
    # The closed x_session_id_run_v1 mode: a non-secret run identity sent as
    # X-Session-ID so a gateway or prefix-caching router keeps one upstream
    # session (and its prompt cache) per run. Admitted for HTTPS endpoints and
    # HTTP loopback gateways; refused for plain HTTP to any other host.
    session_affinity_id: str | None = field(default=None, kw_only=True)
    # Profile-owned "strict" statement on function tools (openai_wire_options).
    tool_strictness: str | None = field(default=None, kw_only=True)
    reasoning_field: str = field(default="reasoning_content", kw_only=True)  # vLLM: "reasoning"

    checkpoint_request_options = staticmethod(checkpoint_request_options)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        if not self.base_url or not self.api_key:
            raise PolicySessionError("native provider base URL and API key are required")
        if self.output_token_parameter not in {"max_tokens", "max_completion_tokens"}:
            raise PolicySessionError("unsupported OpenAI output token parameter")
        if self.tool_strictness not in {None, *TOOL_STRICTNESS_MODES} or self.reasoning_field not in REASONING_FIELDS:
            raise PolicySessionError("unsupported OpenAI tool_strictness or reasoning_field")
        if self.session_affinity_id is not None:
            if not re.fullmatch(r"[A-Za-z0-9:._-]{1,200}", self.session_affinity_id):
                raise PolicySessionError("invalid provider session-affinity identity")
            endpoint = urlsplit(self.base_url)
            if endpoint.scheme != "https" and (
                endpoint.scheme != "http" or endpoint.hostname not in {"127.0.0.1", "localhost", "::1"}
            ):
                raise PolicySessionError(
                    "provider session affinity is admitted only for a loopback gateway or an HTTPS endpoint"
                )
        if self.client is None:
            try:
                import httpx
            except ImportError as exc:  # pragma: no cover - dependency gate
                raise PolicySessionError("httpx is required for native_api policies") from exc
            self.client = httpx.Client(timeout=httpx.Timeout(
                self.timeout_seconds, connect=min(10.0, self.timeout_seconds),
            ))
            self._owns_client = True

    def _request_once(self, body: Mapping[str, Any]) -> tuple[list[dict[str, Any]], float | None]:
        raw_events: list[dict[str, Any]] = []
        first_event_at: float | None = None
        headers = {
            "authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }
        if self.session_affinity_id is not None:
            headers["x-session-id"] = self.session_affinity_id
        response = self.client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers=headers,
            content=canonical_bytes(body),
        )
        with response as stream:
            if stream.status_code >= 400:
                detail = stream.read().decode("utf-8", errors="replace")[-4000:]
                detail = detail.replace(self.api_key, "[REDACTED]")
                raise http_status_failure(stream.status_code, detail, base_url=self.base_url)
            if body.get("stream") is False:
                value = json.loads(stream.read())
                if not isinstance(value, Mapping):
                    raise PolicySessionError("native provider returned non-object JSON")
                return [dict(value)], 0.0
            try:
                announced_event: str | None = None
                finish_received = False
                for line in stream.iter_lines():
                    if not line:
                        continue
                    text = line.decode("utf-8", errors="replace") if isinstance(line, bytes) else str(line)
                    if text.startswith(":"):
                        continue
                    if text.startswith("event:"):
                        if announced_event is not None:
                            raise PolicySessionError(
                                "native provider SSE event lacks its data payload"
                            )
                        announced_event = text[6:].strip()
                        if not announced_event:
                            raise PolicySessionError(
                                "native provider SSE event name is empty"
                            )
                        continue
                    if not text.startswith("data:"):
                        raise PolicySessionError(
                            "native provider returned malformed SSE framing"
                        )
                    data = text[5:].strip()
                    if data == "[DONE]":
                        announced_event = None
                        break
                    try:
                        parsed = json.loads(data)
                    except json.JSONDecodeError as exc:
                        raise PolicySessionError("native provider returned malformed SSE JSON") from exc
                    if not isinstance(parsed, Mapping):
                        raise PolicySessionError("native provider SSE item must be an object")
                    if (
                        announced_event is not None
                        and parsed.get("type") != announced_event
                    ):
                        raise PolicySessionError(
                            "native provider SSE event name does not match its payload"
                        )
                    announced_event = None
                    if first_event_at is None:
                        first_event_at = time.monotonic()
                    raw_events.append(dict(parsed))
                    choices = parsed.get("choices")
                    if isinstance(choices, list):
                        finish_received |= any(
                            isinstance(choice, Mapping)
                            and isinstance(choice.get("finish_reason"), str)
                            for choice in choices
                        )
                if announced_event is not None:
                    raise PolicySessionError(
                        "native provider SSE ended before an announced event payload"
                    )
                if not finish_received:
                    raise EOFError("native provider stream ended before completion status")
            except Exception as exc:
                if raw_events:
                    failure = stream_failure_after_content(exc)
                    failure.partial_events = raw_events
                    if self.api_key in json.dumps(raw_events, ensure_ascii=False):
                        failure.retryable = False
                    raise failure from exc
                raise
        return raw_events, first_event_at

    @profile_bound
    def prepare_request(
        request: Mapping[str, Any], *, output_token_parameter: str = "max_tokens",
        tool_strictness: str | None = None, reasoning_field: str = "reasoning_content",
    ) -> dict[str, Any]:
        """Return the exact Chat Completions wire body (a provider's own options when bound)."""

        forbidden = sorted(set(request) & {
            "input", "instructions", "system", "thinking", "output_config",
            "input_schema",
        })
        if forbidden:
            raise PolicySessionError(
                "OpenAI canonical request contains Anthropic/Responses fields: "
                + ", ".join(forbidden)
            )
        messages = request.get("messages")
        model = request.get("model")
        if not isinstance(messages, list) or not isinstance(model, str) or not model:
            raise PolicySessionError("OpenAI request lacks messages or model")
        body = copy.deepcopy(dict(request))
        replayed: list[dict[str, Any]] = []
        for raw_message in messages:
            if not isinstance(raw_message, Mapping):
                raise PolicySessionError("OpenAI request message must be an object")
            message = copy.deepcopy(dict(raw_message))
            raw_state = message.pop(PROVIDER_CONTINUATION_MESSAGE_KEY, None)
            if raw_state is None:
                if {"reasoning_content", "reasoning"} & set(message):
                    raise PolicySessionError(
                        "unsigned OpenAI reasoning_content cannot enter replay history"
                    )
                replayed.append(message)
                continue
            if message.get("role") != "assistant" or not isinstance(raw_state, Mapping):
                raise PolicySessionError("OpenAI continuation is not bound to an assistant")
            state = ProviderContinuationState.from_mapping(raw_state)
            state.validate_binding(
                message=raw_message,
                provider_protocol="openai_chat_completions",
                model=model,
            )
            if state.native_kind != _OPENAI_CONTINUATION_KIND:
                raise PolicySessionError("unsupported OpenAI continuation native kind")
            if set(state.native_payload) != {"reasoning_content"}:
                raise PolicySessionError("OpenAI continuation payload is not canonical")
            reasoning = state.native_payload.get("reasoning_content")
            # An explicitly returned empty string is a valid signed payload:
            # some compatible providers use it for zero-reasoning tool turns.
            # Preserve it exactly; missing/non-string payloads remain invalid.
            if not isinstance(reasoning, str):
                raise PolicySessionError("OpenAI continuation lacks reasoning_content")
            message["reasoning_content"] = reasoning
            replayed.append(message)
        body["messages"] = replayed
        project_openai_tool_names(body, openai_tool_names(request))
        project_wire_body(body, tool_strictness=tool_strictness, reasoning_field=reasoning_field)
        if output_token_parameter not in {"max_tokens", "max_completion_tokens"}:
            raise PolicySessionError("unsupported OpenAI output token parameter")
        if output_token_parameter == "max_completion_tokens" and "max_tokens" in body:
            body["max_completion_tokens"] = body.pop("max_tokens")
        return body

    def complete(
        self, *, request: Mapping[str, Any], model_call_id: str
    ) -> dict[str, Any]:
        started_unix = time.time()
        started_mono = time.monotonic()
        window = RecoveryWindow.for_provider(self)
        body = self.prepare_request(request)  # the recorded wire request, exactly
        tool_names_by_alias = openai_tool_names(request)
        saw_content = False
        raw_events: list[dict[str, Any]] = []
        first_event_at: float | None = None
        attempts = 0
        while True:
            if self.closed:
                closed = ProviderClosed()
                _record_provider_failure(closed, attempts)
                raise closed
            attempts += 1
            attempt_started, attempt_mono = time.time(), time.monotonic()
            try:
                raw_events, first_event_at = (bounded_request(self, body, window)
                    if window is not None else self._request_once(body))
                break
            except Exception as exc:
                _record_provider_failure(exc, attempts)
                echoed = partial_credential_echo(exc, (self.api_key,))
                if window is not None:
                    delay = window.transport_delay(exc, attempts)
                    partial = getattr(exc, "partial_events", [])
                    if self.closed or echoed:
                        delay = None
                    record_attempt_failure(self, exc, body=body, model_call_id=model_call_id,
                        attempt=attempts, credential_index=0, next_index=0 if delay is not None else None,
                        started=attempt_started, started_monotonic=attempt_mono,
                        retry_delay=delay or 0, events=partial, phase="transport")
                    if echoed:
                        raise _echo_failure(exc, attempts) from exc
                    if delay is None:
                        raise
                    try:
                        window.pause(delay)
                    except ProviderRecoveryDeadline as deadline:
                        _record_provider_failure(deadline, attempts)
                        raise deadline from exc
                    continue
                retryable = bool(getattr(exc, "retryable", False))
                if not retryable:
                    retryable = _retryable_transport_error(exc)
                partial = getattr(exc, "partial_events", [])
                will_retry = (
                    not self.closed
                    and not self.preflight_probe and not saw_content and retryable
                    and attempts < self.max_retries + 1 and not echoed
                )
                delay = min(2 ** (attempts - 1), 4) if will_retry else 0
                record_attempt_failure(self, exc, body=body, model_call_id=model_call_id,
                    attempt=attempts, credential_index=0,
                    next_index=0 if will_retry else None,
                    started=attempt_started, started_monotonic=attempt_mono,
                    retry_delay=delay, events=partial, phase="transport")
                if echoed:
                    raise _echo_failure(exc, attempts) from exc
                if not will_retry:
                    raise
                time.sleep(delay)

        try:
            content: list[str] = []
            reasoning: list[str] = []
            calls: dict[int, dict[str, Any]] = {}
            usage: dict[str, Any] | None = None
            finish_reason: str | None = None
            finish_seen = False
            for event in raw_events:
                normalized = normalize_usage(event.get("usage"))
                if normalized is not None:
                    usage = normalized
                choices = event.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                if len(choices) != 1 or not isinstance(choices[0], Mapping):
                    raise PolicySessionError(
                        "native provider must return exactly one response choice"
                    )
                if finish_seen:
                    raise PolicySessionError(
                        "native provider returned response content after completion"
                    )
                choice = choices[0]
                finish = choice.get("finish_reason")
                if isinstance(finish, str):
                    finish_reason = finish
                    finish_seen = True
                delta = choice.get("delta") if body.get("stream") is not False else choice.get("message")
                if not isinstance(delta, Mapping):
                    continue
                if isinstance(delta.get("content"), str):
                    content.append(delta["content"])
                    saw_content = True
                if isinstance(delta.get(self.reasoning_field), str):
                    reasoning.append(delta[self.reasoning_field])
                    saw_content = True
                tool_calls = delta.get("tool_calls")
                if isinstance(tool_calls, list):
                    for raw in tool_calls:
                        if not isinstance(raw, Mapping):
                            continue
                        index = raw.get("index", len(calls))
                        index = index if isinstance(index, int) else len(calls)
                        current = calls.setdefault(index, {
                            "id": None,
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        })
                        if isinstance(raw.get("id"), str):
                            current["id"] = raw["id"]
                        function = raw.get("function")
                        if isinstance(function, Mapping):
                            if isinstance(function.get("name"), str):
                                current["function"]["name"] += function["name"]
                            if isinstance(function.get("arguments"), str):
                                current["function"]["arguments"] += function["arguments"]
                        saw_content = True
            if self.api_key and self.api_key in json.dumps(raw_events, ensure_ascii=False):
                raise PolicySessionError("native provider echoed credential material; response was discarded")
            normalized_calls = []
            for index, call in sorted(calls.items()):
                if not call.get("id"):
                    call["id"] = f"{model_call_id}-tool-{index:02d}"
                name = call["function"]["name"]
                call["function"]["name"] = tool_names_by_alias.get(name, name)
                normalized_calls.append(call)
            if finish_reason is None:
                incomplete = PolicySessionError(
                    "native provider response has no completion status; request was not retried"
                )
                setattr(incomplete, "retryable", False)
                raise incomplete
            message: dict[str, Any] = {
                "role": "assistant",
                "content": "".join(content),
            }
            if reasoning:
                message["reasoning_content"] = "".join(reasoning)
            if normalized_calls:
                message["tool_calls"] = normalized_calls
            output_rejection = None
            if finish_reason == "length" and normalized_calls:
                message, output_rejection = _reject_truncated_tools(
                    message, provider_stop_reason="length")
                finish_reason = "invalid_tool_input"
            elif reasoning:
                continuation = ProviderContinuationState.create(
                    provider_protocol="openai_chat_completions",
                    model=str(body.get("model", "")),
                    native_kind=_OPENAI_CONTINUATION_KIND,
                    native_payload={"reasoning_content": "".join(reasoning)},
                    canonical_message=message,
                )
                message[PROVIDER_CONTINUATION_MESSAGE_KEY] = continuation.to_dict()
        except PolicySessionError as exc:
            setattr(exc, "response_started", bool(raw_events))
            _record_provider_failure(exc, attempts)
            if window is not None:
                record_attempt_failure(self, exc, body=body, model_call_id=model_call_id,
                    attempt=attempts, credential_index=0, next_index=None,
                    started=attempt_started, started_monotonic=attempt_mono,
                    retry_delay=0, events=raw_events, phase="parse")
            raise
        if window is not None and window.remaining() <= 0:
            error = ProviderRecoveryDeadline(raw_events)
            _record_provider_failure(error, attempts)
            raise error
        finished = time.time()
        return {
            "message": message,
            "usage": usage,
            "finish_reason": finish_reason,
            **({"output_rejection": output_rejection} if output_rejection is not None else {}),
            "raw_events": raw_events,
            "model_call_id": model_call_id,
            "started_unix_s": started_unix,
            "finished_unix_s": finished,
            "ttft_s": None if first_event_at is None else round(first_event_at - started_mono, 6),
            "attempts": attempts,
        }

    def close(self) -> None:
        self.closed = True
        if self._owns_client:
            self.client.close()


def _anthropic_content(content: Any) -> list[dict[str, Any]]:
    """Translate canonical text/image content into Anthropic blocks."""

    if isinstance(content, str):
        return [] if not content else [{"type": "text", "text": content}]
    if not isinstance(content, list):
        raise PolicySessionError("canonical message content must be text or a block list")
    blocks: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, Mapping):
            raise PolicySessionError("canonical message content block must be an object")
        block_type = item.get("type")
        if block_type == "text" and isinstance(item.get("text"), str):
            blocks.append({"type": "text", "text": item["text"]})
            continue
        if block_type == "image_url":
            image_url = item.get("image_url")
            url = image_url.get("url") if isinstance(image_url, Mapping) else image_url
            if not isinstance(url, str):
                raise PolicySessionError("canonical image_url block lacks a URL")
            match = _DATA_URL.fullmatch(url)
            if match is None:
                raise PolicySessionError(
                    "Anthropic adapter accepts only inline base64 image data URLs"
                )
            media_type, data = match.groups()
            try:
                base64.b64decode(data, validate=True)
            except Exception as exc:
                raise PolicySessionError("canonical image data URL is invalid base64") from exc
            blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": data,
                },
            })
            continue
        raise PolicySessionError(f"unsupported canonical content block: {block_type!r}")
    return blocks


def _declared_property_types(
    tools: Any,
) -> dict[str, dict[str, str]]:
    """Map each tool's top-level property names to their declared JSON type.

    Used to decide which stringified arguments are safe to decode. Only the
    tool's own schema can answer that; guessing from the value's shape cannot.
    """

    declared: dict[str, dict[str, str]] = {}
    if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes)):
        return declared
    for tool in tools:
        if not isinstance(tool, Mapping):
            continue
        name = tool.get("name")
        schema = tool.get("input_schema")
        if not isinstance(name, str) or not isinstance(schema, Mapping):
            continue
        types = _property_types(schema)
        if types:
            declared[name] = types
    return declared


def _property_types(schema: Mapping[str, Any]) -> dict[str, str]:
    """Collect property types, including through a top-level oneOf.

    move_pose declares its flat and enhanced forms as oneOf branches, so its
    properties live one level down and a reader that only looks at
    schema["properties"] sees an empty tool. Where branches disagree about a
    property's type the entry is dropped rather than guessed: an ambiguous
    declaration is not a basis for rewriting a value.
    """

    collected: dict[str, str] = {}
    ambiguous: set[str] = set()

    def absorb(node: Any) -> None:
        if not isinstance(node, Mapping):
            return
        properties = node.get("properties")
        if isinstance(properties, Mapping):
            for key, value in properties.items():
                if not (isinstance(key, str) and isinstance(value, Mapping)):
                    continue
                kind = value.get("type")
                if not isinstance(kind, str):
                    continue
                if key in collected and collected[key] != kind:
                    ambiguous.add(key)
                collected[key] = kind
        for branch_key in ("oneOf", "anyOf", "allOf"):
            branches = node.get(branch_key)
            if isinstance(branches, Sequence) and not isinstance(
                branches, (str, bytes)
            ):
                for branch in branches:
                    absorb(branch)

    absorb(schema)
    for key in ambiguous:
        collected.pop(key, None)
    return collected


def _decode_stringified_object_arguments(
    arguments: dict[str, Any],
    declared_types: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Recover object-valued tool arguments that arrived as JSON text.

    Qwen3's tool calls are XML, where every parameter value is text, and a
    server-side parser has to restore the declared types. vLLM's qwen3_xml
    parser does not always restore nested objects, so a call whose schema
    declares `targets: {...}` can arrive as the *string*
    '{"right": {"position": ...}}'. The ToolSurface then rejects it as
    invalid_motion_arguments, and the policy burns calls guessing at a format
    it had already produced correctly.

    What is safe to decode comes from the tool's own schema, not from the shape
    of the value. Passing declared_types decodes a string exactly when the
    property is declared "object" or "array", which leaves a genuine string
    parameter alone even when its text happens to look like JSON. Without it,
    only objects are decoded, since guessing is far more likely to corrupt a
    real string than to help.

    The array case is not hypothetical. Measured over three recorded pilot cells,
    the surviving invalid_motion_arguments rejections were dominated by
      {"position_m": "[-0.119, -0.003, 0.975]", ...}
    where the vector, not the enclosing object, is the value left as text.

    This is a transport repair, not schema coercion: it cannot fix a wrong key
    name, and a genuinely malformed argument still reaches the ToolSurface and
    is still reported.
    """

    decodable = {"object", "array"}
    repaired: dict[str, Any] = {}
    for name, value in arguments.items():
        if declared_types is not None:
            wanted = declared_types.get(name)
            opener = {"object": "{", "array": "["}.get(wanted or "")
            eligible = (
                isinstance(value, str)
                and wanted in decodable
                and value.lstrip()[:1] == opener
            )
        else:
            eligible = isinstance(value, str) and value.lstrip()[:1] == "{"
        if eligible:
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                repaired[name] = value
                continue
            # Accept the decoded value only if it matches what the schema
            # declared. A string that parses to the wrong kind is more likely a
            # genuine text argument than a transport artefact.
            expected = (
                dict if (declared_types or {}).get(name) == "object" else list
            ) if declared_types is not None else dict
            repaired[name] = decoded if isinstance(decoded, expected) else value
        else:
            repaired[name] = value
    return repaired


def _anthropic_tool_input(call: Mapping[str, Any]) -> tuple[str, str, dict[str, Any]]:
    function = call.get("function")
    if not isinstance(function, Mapping) or not isinstance(function.get("name"), str):
        raise PolicySessionError("canonical tool call lacks a function name")
    raw = function.get("arguments", "{}")
    if isinstance(raw, Mapping):
        arguments = dict(raw)
    elif isinstance(raw, str):
        try:
            arguments = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PolicySessionError("canonical tool arguments are invalid JSON") from exc
    else:
        raise PolicySessionError("canonical tool arguments must be JSON text or an object")
    if not isinstance(arguments, dict):
        raise PolicySessionError("canonical tool arguments must decode to an object")
    # History is already canonical; preserve string values exactly on replay.
    call_id = call.get("id")
    if not isinstance(call_id, str) or not call_id:
        raise PolicySessionError("canonical tool call lacks an ID")
    return call_id, function["name"], arguments


_ANTHROPIC_CONTINUATION_KIND = "anthropic_assistant_content_blocks.v1"


def _normalize_anthropic_blocks(
    blocks: Sequence[Mapping[str, Any]],
    *,
    model_call_id: str,
    tool_names_by_alias: Mapping[str, str] | None,
) -> dict[str, Any]:
    """Normalize finalized Anthropic assistant blocks into canonical IR."""

    text_parts: list[str] = []
    thinking_parts: list[str] = []
    normalized_calls: list[dict[str, Any]] = []
    for index, block in enumerate(blocks):
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if not isinstance(text, str):
                raise PolicySessionError("Anthropic text block lacks text")
            text_parts.append(text)
            continue
        if block_type == "thinking":
            thinking = block.get("thinking")
            signature = block.get("signature")
            if not isinstance(thinking, str):
                raise PolicySessionError("Anthropic thinking block lacks thinking text")
            if not isinstance(signature, str) or not signature:
                raise PolicySessionError(
                    "Anthropic thinking block lacks a replayable signature"
                )
            thinking_parts.append(thinking)
            continue
        if block_type == "redacted_thinking":
            data = block.get("data")
            if not isinstance(data, str) or not data:
                raise PolicySessionError(
                    "Anthropic redacted_thinking block lacks opaque data"
                )
            continue
        if block_type == "tool_use":
            arguments = block.get("input")
            if not isinstance(arguments, Mapping):
                raise PolicySessionError("Anthropic tool input must be an object")
            name = block.get("name")
            if not isinstance(name, str) or not name:
                raise PolicySessionError("Anthropic tool_use lacks a name")
            if tool_names_by_alias is not None:
                name = tool_names_by_alias.get(name, name)
            call_id = block.get("id")
            if not isinstance(call_id, str) or not call_id:
                call_id = f"{model_call_id}-tool-{index:02d}"
            normalized_calls.append({
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(
                        dict(arguments), ensure_ascii=False, sort_keys=True,
                        separators=(",", ":"),
                    ),
                },
            })
            continue
        raise PolicySessionError(
            f"unsupported Anthropic assistant content block: {block_type!r}"
        )

    message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(text_parts),
    }
    if thinking_parts:
        message["reasoning_content"] = "".join(thinking_parts)
    if normalized_calls:
        message["tool_calls"] = normalized_calls
    return message


def _anthropic_continuation_blocks(
    message: Mapping[str, Any],
    *,
    model: str,
    tool_names_by_alias: Mapping[str, str],
) -> list[dict[str, Any]] | None:
    raw_state = message.get(PROVIDER_CONTINUATION_MESSAGE_KEY)
    if raw_state is None:
        return None
    if not isinstance(raw_state, Mapping):
        raise PolicySessionError("provider continuation state must be an object")
    state = ProviderContinuationState.from_mapping(raw_state)
    state.validate_binding(
        message=message,
        provider_protocol="anthropic_messages",
        model=model,
    )
    if state.native_kind != _ANTHROPIC_CONTINUATION_KIND:
        raise PolicySessionError("unsupported Anthropic continuation native kind")
    raw_blocks = state.native_payload.get("content")
    if not isinstance(raw_blocks, list) or not raw_blocks:
        raise PolicySessionError("Anthropic continuation lacks assistant content blocks")
    if not all(isinstance(block, Mapping) for block in raw_blocks):
        raise PolicySessionError("Anthropic continuation content block is invalid")
    blocks = [copy.deepcopy(dict(block)) for block in raw_blocks]
    # A continuation is bound to the assistant message produced by the
    # request that created it.  The next request may intentionally expose a
    # different tool set.  Recover historical wire aliases from
    # the bound canonical tool calls themselves instead of requiring every
    # historical tool to remain in the next request's tool definitions.
    replay_tool_names = dict(tool_names_by_alias)
    canonical_calls = message.get("tool_calls")
    canonical_calls = canonical_calls if isinstance(canonical_calls, list) else []
    canonical_names_by_id: dict[str, str] = {}
    for raw_call in canonical_calls:
        if not isinstance(raw_call, Mapping):
            continue
        call_id = raw_call.get("id")
        function = raw_call.get("function")
        name = function.get("name") if isinstance(function, Mapping) else None
        if isinstance(call_id, str) and call_id and isinstance(name, str) and name:
            canonical_names_by_id[call_id] = name
    for block in blocks:
        if block.get("type") != "tool_use":
            continue
        call_id = block.get("id")
        wire_name = block.get("name")
        canonical_name = canonical_names_by_id.get(call_id)
        if not (
            isinstance(wire_name, str)
            and wire_name
            and isinstance(canonical_name, str)
            and canonical_name
        ):
            continue
        previous = replay_tool_names.setdefault(wire_name, canonical_name)
        if previous != canonical_name:
            raise PolicySessionError(
                "Anthropic continuation tool-name binding conflicts with the request"
            )
    normalized = _normalize_anthropic_blocks(
        blocks,
        model_call_id=state.response_id or "provider-continuation",
        tool_names_by_alias=replay_tool_names,
    )
    if canonical_assistant_sha256(normalized) != state.canonical_message_sha256:
        raise PolicySessionError(
            "Anthropic continuation payload does not match its canonical assistant"
        )
    return blocks


def translate_anthropic_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Translate the canonical native request into an Anthropic Messages body."""

    forbidden = sorted(set(request) & {
        "input", "instructions", "system", "thinking", "output_config",
        "input_schema",
    })
    if forbidden:
        raise PolicySessionError(
            "Anthropic canonical request contains wire/Responses fields: "
            + ", ".join(forbidden)
        )
    model = request.get("model")
    max_tokens = request.get("max_tokens")
    messages = request.get("messages")
    if not isinstance(model, str) or not model:
        raise PolicySessionError("canonical provider request lacks a model")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise PolicySessionError("canonical provider request has invalid max_tokens")
    if not isinstance(messages, list):
        raise PolicySessionError("canonical provider request messages must be a list")

    system_blocks: list[dict[str, Any]] = []
    translated: list[dict[str, Any]] = []
    tool_names_by_alias = _anthropic_tool_names(request)

    def append_message(role: str, blocks: list[dict[str, Any]]) -> None:
        if not blocks:
            return
        if translated and translated[-1]["role"] == role:
            existing = translated[-1]["content"]
            if (
                blocks[0].get("type") == "tool_result"
                and any(item.get("type") != "tool_result" for item in existing)
            ):
                raise PolicySessionError(
                    "Anthropic tool results must immediately follow tool_use blocks"
                )
            existing.extend(blocks)
        else:
            translated.append({"role": role, "content": blocks})

    for raw_message in messages:
        if not isinstance(raw_message, Mapping):
            raise PolicySessionError("canonical provider message must be an object")
        role = raw_message.get("role")
        if role == "system":
            system_blocks.extend(_anthropic_content(raw_message.get("content", "")))
            continue
        if role == "user":
            append_message("user", _anthropic_content(raw_message.get("content", "")))
            continue
        if role == "assistant":
            blocks = _anthropic_continuation_blocks(
                raw_message,
                model=model,
                tool_names_by_alias=tool_names_by_alias,
            )
            if blocks is None:
                blocks = _anthropic_content(raw_message.get("content", ""))
                calls = raw_message.get("tool_calls")
                if isinstance(calls, list):
                    for raw_call in calls:
                        if not isinstance(raw_call, Mapping):
                            raise PolicySessionError("canonical tool call must be an object")
                        call_id, name, arguments = _anthropic_tool_input(raw_call)
                        blocks.append({
                            "type": "tool_use", "id": call_id,
                            "name": _anthropic_tool_alias(name), "input": arguments,
                        })
                # Normalized reasoning without a provider continuation is trace
                # evidence only.  It must never be forged into an unsigned
                # Anthropic thinking block.
            append_message("assistant", blocks)
            continue
        if role == "tool":
            call_id = raw_message.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                raise PolicySessionError("canonical tool result lacks tool_call_id")
            content = raw_message.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False, sort_keys=True)
            is_error = False
            try:
                decoded = json.loads(content)
                is_error = isinstance(decoded, Mapping) and decoded.get("ok") is False
            except json.JSONDecodeError:
                pass
            block: dict[str, Any] = {
                "type": "tool_result", "tool_use_id": call_id, "content": content,
            }
            if is_error:
                block["is_error"] = True
            append_message("user", [block])
            continue
        raise PolicySessionError(f"unsupported canonical message role: {role!r}")

    renamed_tools = {
        canonical: alias for alias, canonical in tool_names_by_alias.items()
        if alias != canonical
    }
    if renamed_tools:
        # Anthropic-compatible gateways reject dots in tool names. Keep the
        # canonical request and task text untouched, but disclose the exact
        # provider-wire aliases so runtime facts that name capabilities cannot
        # disagree with the callable tools the model actually sees.
        system_blocks.append({
            "type": "text",
            "text": (
                "<wire_tool_aliases>\n"
                + json.dumps(renamed_tools, ensure_ascii=False, sort_keys=True)
                + "\n</wire_tool_aliases>"
            ),
        })

    tools: list[dict[str, Any]] = []
    raw_tools = request.get("tools", [])
    if not isinstance(raw_tools, list):
        raise PolicySessionError("canonical provider tools must be a list")
    for item in raw_tools:
        function = item.get("function") if isinstance(item, Mapping) else None
        if not isinstance(function, Mapping):
            raise PolicySessionError("canonical tool definition lacks function")
        name = function.get("name")
        schema = function.get("parameters")
        if not isinstance(name, str) or not isinstance(schema, Mapping):
            raise PolicySessionError("canonical tool definition is incomplete")
        wire_schema = copy.deepcopy(dict(schema))
        if "type" not in wire_schema and "oneOf" in wire_schema:
            # Anthropic-compatible gateways require
            # input_schema.type == "object"; a top-level oneOf whose
            # branches are all objects is semantically identical with the
            # explicit type, so the wire projection states it. The
            # composition-layer schema bytes stay untouched.
            wire_schema = {"type": "object", **wire_schema}
        tool = {
            "name": _anthropic_tool_alias(name),
            "description": function.get("description", ""),
            "input_schema": wire_schema,
        }
        tools.append(tool)

    body: dict[str, Any] = {
        "model": model,
        "messages": translated,
        "max_tokens": max_tokens,
        "stream": bool(request.get("stream", False)),
    }
    if system_blocks:
        body["system"] = system_blocks
    if tools:
        body["tools"] = tools
    tool_choice = request.get("tool_choice")
    if tool_choice == "auto":
        body["tool_choice"] = {"type": "auto"}
    elif tool_choice == "none":
        body["tool_choice"] = {"type": "none"}
    elif isinstance(tool_choice, Mapping):
        function = tool_choice.get("function")
        name = function.get("name") if isinstance(function, Mapping) else None
        if isinstance(name, str):
            body["tool_choice"] = {
                "type": "tool", "name": _anthropic_tool_alias(name),
            }
    for field in ("temperature", "top_p"):
        if field in request:
            body[field] = request[field]
    reasoning_effort = request.get("reasoning_effort")
    if reasoning_effort is not None and reasoning_effort != "provider_default":
        if reasoning_effort not in {"low", "medium", "high", "xhigh", "max"}:
            raise PolicySessionError(
                "Anthropic reasoning_effort cannot be projected to output_config"
            )
        body["thinking"] = {"type": "adaptive"}
        body["output_config"] = {"effort": reasoning_effort}
    stop = request.get("stop")
    if isinstance(stop, str):
        body["stop_sequences"] = [stop]
    elif isinstance(stop, list):
        body["stop_sequences"] = stop
    return body


def normalize_anthropic_usage(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    uncached = _number(value.get("input_tokens"))
    cache_read = _number(value.get("cache_read_input_tokens"))
    cache_write = _number(value.get("cache_creation_input_tokens"))
    output = _number(value.get("output_tokens"))
    if all(item is None for item in (uncached, cache_read, cache_write, output)):
        return None
    total_input = sum(item or 0 for item in (uncached, cache_read, cache_write))
    output_tokens = output or 0
    return {
        "input_tokens": total_input,
        "cached_input_tokens": cache_read or 0,
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": cache_write,
        "cache_read_status": (
            "reported" if "cache_read_input_tokens" in value else "missing"
        ),
        "cache_creation_status": (
            "reported" if "cache_creation_input_tokens" in value else "missing"
        ),
        "output_tokens": output_tokens,
        "reasoning_output_tokens": 0,
        "total_tokens": total_input + output_tokens,
    }


def _anthropic_blocks_with_tool_inputs(
    response: AnthropicResponseEvents,
    declared_types_by_tool: Mapping[str, Mapping[str, str]] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Blocks in index order with decoded tool inputs, and the inputs rejected."""

    blocks: list[dict[str, Any]] = []
    rejected_tool_inputs: list[dict[str, Any]] = []
    for index, block in sorted(response.blocks_by_index.items()):
        fragments = "".join(response.tool_fragments.get(index, []))
        if fragments:
            try:
                # Same transport repair as the non-streaming path: a parser
                # that leaves an object-valued argument as JSON text would
                # otherwise reach the ToolSurface as a string. The Harness
                # streams, so this is the path that actually runs.
                arguments = json.loads(fragments)
                if isinstance(arguments, dict):
                    arguments = _decode_stringified_object_arguments(
                        arguments,
                        (declared_types_by_tool or {}).get(
                            str(block.get("name") or "")
                        ),
                    )
            except json.JSONDecodeError:
                rejected_tool_inputs.append({
                    "index": index,
                    "tool_call_id": block.get("id"),
                    "tool_name": block.get("name"),
                    "fragment_chars": len(fragments),
                    "fragment_sha256": hashlib.sha256(
                        fragments.encode("utf-8")
                    ).hexdigest(),
                    "reason": "invalid_json",
                })
            else:
                if not isinstance(arguments, Mapping):
                    rejected_tool_inputs.append({
                        "index": index,
                        "tool_call_id": block.get("id"),
                        "tool_name": block.get("name"),
                        "fragment_chars": len(fragments),
                        "fragment_sha256": hashlib.sha256(
                            fragments.encode("utf-8")
                        ).hexdigest(),
                        "reason": "not_an_object",
                    })
                else:
                    block["input"] = copy.deepcopy(dict(arguments))
        if block.get("type") == "tool_use":
            if (
                not isinstance(block.get("input"), Mapping)
                and not any(
                    item["index"] == index for item in rejected_tool_inputs
                )
            ):
                raise PolicySessionError("Anthropic tool use lacks complete input")
            if not isinstance(block.get("id"), str) or not block.get("id"):
                raise PolicySessionError("Anthropic tool use lacks a provider call id")
        blocks.append(block)
    return blocks, rejected_tool_inputs


def _rejected_tool_input_reply(
    response: AnthropicResponseEvents,
    blocks: list[dict[str, Any]],
    rejected_tool_inputs: list[dict[str, Any]],
    *,
    model_call_id: str,
    tool_names_by_alias: Mapping[str, str] | None,
) -> tuple[dict[str, Any], dict[str, Any] | None, str, dict[str, Any]]:
    # Never guess or repair robot tool arguments.  A complete Provider
    # stream with malformed tool JSON is a recoverable no-world-change
    # response: retain its exact raw events, discard every tool block
    # from the canonical conversation, and let the policy retry on the
    # same observation through an explicitly counted model call.
    safe_blocks = [
        block for block in blocks if block.get("type") != "tool_use"
    ]
    message = _normalize_anthropic_blocks(
        safe_blocks,
        model_call_id=model_call_id,
        tool_names_by_alias=tool_names_by_alias,
    )
    rejection = {
        "code": "invalid_tool_input_json",
        "provider_stop_reason": response.stop_reason,
        "rejected_tool_blocks": rejected_tool_inputs,
        "discarded_tool_block_count": sum(
            block.get("type") == "tool_use" for block in blocks
        ),
        "policy_outcome": "continue_same_observation",
        "world_state_changed": False,
    }
    return (
        message,
        normalize_anthropic_usage(response.usage_values),
        "invalid_tool_input",
        rejection,
    )


def _anthropic_reply(
    response: AnthropicResponseEvents,
    blocks: list[dict[str, Any]],
    *,
    model_call_id: str,
    model: str,
    tool_names_by_alias: Mapping[str, str] | None,
) -> tuple[dict[str, Any], dict[str, Any] | None, str | None, dict[str, Any] | None]:
    """The finish reason and canonical message of a reply whose tool inputs decoded.

    An output-limit stop that proposes tools is rejected whole; a reply with
    thinking blocks binds them as its provider continuation.
    """

    stop_reason = response.stop_reason
    finish_reason = {
        "tool_use": "tool_calls",
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
    }.get(stop_reason, stop_reason)
    message = _normalize_anthropic_blocks(
        blocks,
        model_call_id=model_call_id,
        tool_names_by_alias=tool_names_by_alias,
    )
    if finish_reason == "length" and message.get("tool_calls"):
        message, rejection = _reject_truncated_tools(message, provider_stop_reason=stop_reason)
        return message, normalize_anthropic_usage(response.usage_values), "invalid_tool_input", rejection
    if any(
        block.get("type") in {"thinking", "redacted_thinking"}
        for block in blocks
    ):
        continuation = ProviderContinuationState.create(
            provider_protocol="anthropic_messages",
            model=model,
            native_kind=_ANTHROPIC_CONTINUATION_KIND,
            native_payload={"content": blocks},
            canonical_message=message,
            response_id=response.response_id,
        )
        message[PROVIDER_CONTINUATION_MESSAGE_KEY] = continuation.to_dict()
    return message, normalize_anthropic_usage(response.usage_values), finish_reason, None


@dataclass
class AnthropicMessagesProvider:
    """Direct Anthropic Messages adapter with canonical response normalization."""

    base_url: str
    api_key: str
    timeout_seconds: float = 900.0
    max_retries: int = 2
    preflight_probe: bool = field(default=False, repr=False, kw_only=True)
    recovery_policy: str = field(default="legacy_v1", kw_only=True)
    recovery_window_s: float | None = field(default=None, kw_only=True)
    request_deadline_monotonic: float | None = field(default=None, repr=False, kw_only=True)
    attempt_observer: Callable[[Mapping[str, Any]], None] | None = field(
        default=None, repr=False, kw_only=True,
    )
    anthropic_version: str = "2023-06-01"
    auth_scheme: str = "x_api_key"
    continuation_replay: str = "exact_v1"
    session_affinity_id: str | None = None
    # Host-local credential rotation: one session stays on one key for cache
    # locality and moves to the next when the current key is exhausted,
    # rejected, or an incomplete stream can safely retry. Keys are operator
    # environment values; only their ordinal
    # index is ever reported.
    standby_api_keys: tuple[str, ...] = ()
    credential_start_index: int = 0
    client: Any = None
    _owns_client: bool = field(default=False, init=False)
    _credential_index: int = field(default=0, init=False)
    _credential_ring: tuple[str, ...] = field(default=(), init=False)
    _rotations: list[dict[str, Any]] = field(default_factory=list, init=False)
    closed: bool = field(default=False, init=False)
    protocol: str = field(default="anthropic_messages", init=False)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        if not self.base_url or not self.api_key:
            raise PolicySessionError("native provider base URL and API key are required")
        if any(not key for key in self.standby_api_keys):
            raise PolicySessionError("standby provider credentials must be non-empty")
        # The rotation ring starts at this session's assigned key and returns
        # to it after one lap; provider usage limits reset on their own clock.
        self._credential_ring = (self.api_key, *self.standby_api_keys)
        if self.auth_scheme not in {"bearer", "x_api_key"}:
            raise PolicySessionError("unsupported Anthropic provider auth scheme")
        if self.continuation_replay not in _ANTHROPIC_CONTINUATION_REPLAY_MODES:
            raise PolicySessionError("unsupported Anthropic continuation replay mode")
        if self.session_affinity_id is not None and not re.fullmatch(
            r"[A-Za-z0-9:._-]{1,200}", self.session_affinity_id,
        ):
            raise PolicySessionError("invalid provider session-affinity identity")
        if self.client is None:
            try:
                import httpx
            except ImportError as exc:  # pragma: no cover - dependency gate
                raise PolicySessionError("httpx is required for native_api policies") from exc
            self.client = httpx.Client(timeout=httpx.Timeout(
                self.timeout_seconds, connect=min(10.0, self.timeout_seconds),
            ))
            self._owns_client = True

    def _rotate_credential(self, *, status_code: int, detail: str) -> bool:
        """Move to the next host-local credential after an exhausted key.

        The ring is cyclic: a provider usage limit resets on the provider's
        own clock, so a key that is exhausted now is a key to come back to,
        not a key to discard. One full lap without success means every
        credential is currently exhausted, which the caller turns into a
        patient wait instead of a fatal fault.
        """

        ring = self._credential_ring
        if len(ring) < 2:
            return False
        self._credential_index += 1
        self.api_key = ring[self._credential_index % len(ring)]
        self._rotations.append({
            "schema": "ora.provider-credential-rotation.v1",
            "credential_index": self._credential_index,
            "credential_start_index": self.credential_start_index,
            "completed_laps": self._credential_index // len(ring),
            "status_code": status_code,
            "unix_s": time.time(),
        })
        return True

    def credential_rotations(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._rotations]

    def _redact(self, text: str) -> str:
        for key in sorted(self._credential_ring, key=len, reverse=True):
            if key:
                text = text.replace(key, "[REDACTED]")
        return text

    def prepare_request(self, request: Mapping[str, Any]) -> dict[str, Any]:
        return translate_anthropic_request(request)

    def _messages_url(self) -> str:
        if self.base_url.endswith("/v1/messages"):
            return self.base_url
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/messages"
        return f"{self.base_url}/v1/messages"

    def _record_attempt_failure(
        self, exc: Exception, *, body: Mapping[str, Any], model_call_id: str,
        attempt: int, credential_index: int, next_index: int | None,
        started: float, started_monotonic: float, retry_delay: float,
        events: Sequence[Mapping[str, Any]], phase: str,
    ) -> None:
        record_attempt_failure(self, exc, body=body, model_call_id=model_call_id,
            attempt=attempt, credential_index=credential_index, next_index=next_index,
            started=started, started_monotonic=started_monotonic,
            retry_delay=retry_delay, events=events, phase=phase)

    def _request_once(self, body: Mapping[str, Any]) -> tuple[list[dict[str, Any]], float | None]:
        raw_events: list[dict[str, Any]] = []
        first_event_at: float | None = None
        headers = {
            "anthropic-version": self.anthropic_version,
            "content-type": "application/json",
        }
        if self.auth_scheme == "bearer":
            headers["authorization"] = f"Bearer {self.api_key}"
        else:
            headers["x-api-key"] = self.api_key
        if self.session_affinity_id is not None:
            headers["x-session-id"] = self.session_affinity_id
        response = self.client.stream(
            "POST",
            self._messages_url(),
            headers=headers,
            content=canonical_bytes(body),
        )
        with response as stream:
            if stream.status_code >= 400:
                detail = stream.read().decode("utf-8", errors="replace")[-4000:]
                detail = self._redact(detail)
                rotated = (
                    self._rotate_credential(
                        status_code=stream.status_code, detail=detail,
                    )
                    if stream.status_code in {401, 402, 403, 429}
                    and not self.preflight_probe and self.max_retries > 0 else False
                )
                error = http_status_failure(stream.status_code, detail, base_url=self.base_url)
                kimi_signature_flake = (
                    stream.status_code == 400
                    and self.base_url in _KIMI_CODING_BASE_URLS
                    and _KIMI_TRANSIENT_SIGNATURE_REJECTION in detail
                    and _has_complete_anthropic_thinking_signature(body)
                )
                setattr(error, "retryable", error.retryable or kimi_signature_flake or rotated)
                if rotated:
                    setattr(error, "credential_rotated", True)
                if kimi_signature_flake:
                    setattr(error, "retry_delay_s", 30.0)
                if stream.status_code == 429:
                    # A provider-side throttle asks for patience, not three
                    # sub-second attempts; a fatal fault here kills the whole
                    # persistent session and its sunk rounds.
                    setattr(error, "retry_delay_s", 20.0)
                    setattr(error, "retry_budget", max(self.max_retries, 6))
                if rotated and self._credential_index >= len(
                    self._credential_ring
                ):
                    # One full lap: every configured credential is currently
                    # rejecting. Provider usage limits reset on their own
                    # clock, so wait for that rather than killing a persistent
                    # session and its completed rounds. This is the most
                    # patient case and therefore wins over the throttle wait.
                    setattr(error, "retry_delay_s", 90.0)
                    setattr(error, "retry_budget", max(self.max_retries, 8))
                raise error
            if body.get("stream") is False:
                value = json.loads(stream.read())
                if not isinstance(value, Mapping):
                    raise PolicySessionError("native provider returned non-object JSON")
                return [dict(value)], 0.0
            try:
                announced_event: str | None = None
                for line in stream.iter_lines():
                    if not line:
                        continue
                    text = line.decode("utf-8", errors="replace") if isinstance(line, bytes) else str(line)
                    if text.startswith(":"):
                        continue
                    if text.startswith("event:"):
                        if announced_event is not None:
                            raise PolicySessionError(
                                "Anthropic SSE event lacks its data payload"
                            )
                        announced_event = text[6:].strip()
                        if not announced_event:
                            raise PolicySessionError(
                                "Anthropic SSE event name is empty"
                            )
                        continue
                    if not text.startswith("data:"):
                        raise PolicySessionError(
                            "Anthropic provider returned malformed SSE framing"
                        )
                    data = text[5:].strip()
                    if data == "[DONE]":
                        announced_event = None
                        break
                    try:
                        parsed = json.loads(data)
                    except json.JSONDecodeError as exc:
                        raise PolicySessionError("native provider returned malformed SSE JSON") from exc
                    if not isinstance(parsed, Mapping):
                        raise PolicySessionError("native provider SSE item must be an object")
                    if (
                        announced_event is not None
                        and parsed.get("type") != announced_event
                    ):
                        raise PolicySessionError(
                            "Anthropic SSE event name does not match its payload"
                        )
                    announced_event = None
                    if first_event_at is None:
                        first_event_at = time.monotonic()
                    raw_event = dict(parsed)
                    raw_events.append(raw_event)
                    # Anthropic defines message_stop as the protocol terminal
                    # event. Do not require the HTTP/SSE transport to close
                    # cleanly after a complete response: some loopback routers
                    # reset the connection once the terminal event has been
                    # forwarded. A failure before message_stop remains a
                    # partial stream, which may be discarded and retried before
                    # returning any tool calls to the Native dispatcher.
                    if raw_event.get("type") == "message_stop":
                        break
                if announced_event is not None:
                    raise PolicySessionError(
                        "Anthropic SSE ended before an announced event payload"
                    )
                if not any(event.get("type") in {"message_stop", "error"} for event in raw_events):
                    raise EOFError("Anthropic stream ended before message_stop")
            except Exception as exc:
                # Framing/JSON/SSL certificate failures remain terminal. Only
                # transport interruption (including H's raw TLS EOF case) or
                # clean premature EOF can retry before any tool is dispatched.
                partial = _StreamInterrupted(
                    raw_events, transient=_retryable_transport_error(exc) or isinstance(exc, EOFError),
                )
                raise partial from exc
        return raw_events, first_event_at

    @staticmethod
    def _parse_events(
        raw_events: Sequence[Mapping[str, Any]], model_call_id: str, model: str,
        tool_names_by_alias: Mapping[str, str] | None = None,
        declared_types_by_tool: Mapping[str, Mapping[str, str]] | None = None,
    ) -> tuple[
        dict[str, Any], dict[str, Any] | None, str | None,
        dict[str, Any] | None,
    ]:
        """Accumulate the events, decode each tool input, then build the reply."""

        response = accumulate_anthropic_events(raw_events)
        blocks, rejected_tool_inputs = _anthropic_blocks_with_tool_inputs(
            response, declared_types_by_tool,
        )
        if rejected_tool_inputs:
            return _rejected_tool_input_reply(
                response, blocks, rejected_tool_inputs,
                model_call_id=model_call_id,
                tool_names_by_alias=tool_names_by_alias,
            )
        return _anthropic_reply(
            response, blocks, model_call_id=model_call_id, model=model,
            tool_names_by_alias=tool_names_by_alias,
        )

    def complete(
        self, *, request: Mapping[str, Any], model_call_id: str
    ) -> dict[str, Any]:
        started_unix = time.time()
        started_mono = time.monotonic()
        window = RecoveryWindow.for_provider(self)
        tool_names_by_alias = _anthropic_tool_names(request)
        body = self.prepare_request(request)
        raw_events: list[dict[str, Any]] = []
        first_event_at: float | None = None
        attempts = 0
        incomplete_block_retries = 0
        while True:
            if self.closed:
                closed = ProviderClosed()
                _record_provider_failure(closed, attempts)
                raise closed
            attempts += 1
            attempt_started = time.time()
            attempt_monotonic = time.monotonic()
            credential_index = (
                self.credential_start_index + self._credential_index
            ) % len(self._credential_ring)
            try:
                raw_events, first_event_at = (bounded_request(self, body, window)
                    if window is not None else self._request_once(body))
            except Exception as exc:
                _record_provider_failure(exc, attempts)
                retryable = bool(getattr(exc, "retryable", False))
                if not retryable:
                    retryable = _retryable_transport_error(exc)
                retry_budget = getattr(exc, "retry_budget", None)
                if not isinstance(retry_budget, int):
                    retry_budget = self.max_retries
                will_retry = (
                    not self.preflight_probe and self.max_retries > 0
                    and retryable and attempts < retry_budget + 1
                )
                partial_events = getattr(exc, "partial_events", [])
                echoed = partial_credential_echo(exc, self._credential_ring)
                will_retry = will_retry and not echoed
                retry_delay = getattr(exc, "retry_delay_s", None)
                if isinstance(retry_delay, (int, float)) and retry_delay > 0:
                    delay = min(retry_delay * 2 ** (attempts - 1), 60.0)
                else:
                    delay = min(2 ** (attempts - 1), 4)
                if window is not None:
                    bounded_delay = window.transport_delay(exc, attempts)
                    will_retry = bounded_delay is not None and not echoed
                    delay = bounded_delay or 0.0
                if self.closed:
                    will_retry = False
                switch = window is None and will_retry and isinstance(exc, _StreamInterrupted) and len(self._credential_ring) > 1
                next_index = (self.credential_start_index + self._credential_index + int(switch)) % len(self._credential_ring)
                self._record_attempt_failure(
                    exc, body=body, model_call_id=model_call_id, attempt=attempts,
                    credential_index=credential_index,
                    next_index=next_index if will_retry else None,
                    started=attempt_started, started_monotonic=attempt_monotonic,
                    retry_delay=delay if will_retry else 0.0,
                    events=partial_events, phase="transport",
                )
                if echoed:
                    raise _echo_failure(exc, attempts) from exc
                if not will_retry:
                    raise
                if switch:
                    self._credential_index += 1
                    self.api_key = self._credential_ring[self._credential_index % len(self._credential_ring)]
                if window is not None:
                    try:
                        window.pause(delay)
                    except ProviderRecoveryDeadline as deadline:
                        _record_provider_failure(deadline, attempts)
                        raise deadline from exc
                else:
                    time.sleep(delay)
                continue
            try:
                serialized_events = json.dumps(raw_events, ensure_ascii=False)
                if any(key in serialized_events for key in self._credential_ring):
                    raise PolicySessionError(
                        "native provider echoed credential material; response was discarded"
                    )
                message, usage, finish_reason, output_rejection = (
                    self._parse_events(
                        raw_events, model_call_id,
                        str(request.get("model", "")), tool_names_by_alias,
                        _declared_property_types(request.get("tools")),
                    )
                )
                if finish_reason is None:
                    raise PolicySessionError(
                        "native provider response has no completion status; request was not retried"
                    )
            except PolicySessionError as exc:
                setattr(exc, "response_started", bool(raw_events))
                _record_provider_failure(exc, attempts)
                # The complete response is parsed before any policy tool can run.
                # One exact replay is therefore side-effect safe when a gateway
                # emits message_stop while omitting only content_block_stop.  Do
                # not generalize this to arbitrary partial streams or malformed
                # tool arguments.
                safe_incomplete_blocks = (
                    str(exc)
                    == "Anthropic message_stop precedes completed content blocks"
                )
                will_retry = (
                    not self.preflight_probe and safe_incomplete_blocks
                    and incomplete_block_retries < 1 and attempts < self.max_retries + 1
                )
                delay = min(2 ** (attempts - 1), 4) if will_retry else 0.0
                if window is not None:
                    if safe_incomplete_blocks and incomplete_block_retries < 1:
                        setattr(exc, "retryable", True)
                    bounded_delay = window.delay(exc, attempts)
                    will_retry = bounded_delay is not None and not any(
                        key in json.dumps(raw_events, ensure_ascii=False) for key in self._credential_ring)
                    delay = bounded_delay or 0.0
                if self.closed:
                    will_retry = False
                self._record_attempt_failure(
                    exc, body=body, model_call_id=model_call_id, attempt=attempts,
                    credential_index=credential_index,
                    next_index=credential_index if will_retry else None,
                    started=attempt_started, started_monotonic=attempt_monotonic,
                    retry_delay=delay, events=raw_events, phase="parse",
                )
                if not will_retry:
                    raise
                if safe_incomplete_blocks:
                    incomplete_block_retries += 1
                if window is not None:
                    try:
                        window.pause(delay)
                    except ProviderRecoveryDeadline as deadline:
                        _record_provider_failure(deadline, attempts)
                        raise deadline from exc
                else:
                    time.sleep(delay)
                continue
            break
        if window is not None and window.remaining() <= 0:
            error = ProviderRecoveryDeadline(raw_events)
            _record_provider_failure(error, attempts)
            raise error
        response_id = next((
            str(message["id"])
            for event in raw_events
            for message in (
                [event.get("message")]
                if event.get("type") == "message_start"
                else [event]
            )
            if isinstance(message, Mapping)
            and isinstance(message.get("id"), str)
            and message["id"]
        ), None)
        finished = time.time()
        return {
            "message": message,
            "usage": usage,
            "finish_reason": finish_reason,
            **({
                "output_rejection": output_rejection,
            } if output_rejection is not None else {}),
            "response_id": response_id,
            "raw_events": raw_events,
            "wire_request": body,
            "model_call_id": model_call_id,
            "started_unix_s": started_unix,
            "finished_unix_s": finished,
            "ttft_s": None if first_event_at is None else round(first_event_at - started_mono, 6),
            "attempts": attempts,
        }

    def close(self) -> None:
        self.closed = True
        if self._owns_client:
            self.client.close()


def build_native_provider(
    provider: str, *, base_url: str, api_key: str, timeout_seconds: float = 900.0,
    auth_scheme: str | None = None, continuation_replay: str = "exact_v1",
    session_affinity_id: str | None = None,
    standby_api_keys: Sequence[str] = (),
    credential_start_index: int = 0,
    recovery_policy: str = "legacy_v1",
    recovery_window_s: float | None = None,
    output_token_parameter: str | None = None,
    tool_strictness: str | None = None,
    reasoning_field: str | None = None,
) -> OpenAICompatibleProvider | AnthropicMessagesProvider:
    """Construct the explicit RunSpec-selected native provider."""

    if recovery_policy not in {"legacy_v1", RECOVERY_POLICY}:
        raise PolicySessionError("unsupported native provider recovery policy")
    if recovery_window_s is not None and recovery_policy == "legacy_v1":
        raise PolicySessionError("recovery_window_s requires the same-credential deadline recovery policy")
    if provider == "openai_compatible":
        if auth_scheme not in {None, "bearer"}:
            raise PolicySessionError("OpenAI-compatible providers require bearer auth")
        arguments: dict[str, Any] = {
            "base_url": base_url,
            "api_key": api_key,
            "timeout_seconds": timeout_seconds,
        }
        if recovery_policy != "legacy_v1":
            arguments["recovery_policy"] = recovery_policy
        if recovery_window_s is not None:
            arguments["recovery_window_s"] = float(recovery_window_s)
        if output_token_parameter is not None:
            arguments["output_token_parameter"] = output_token_parameter
        if tool_strictness is not None:
            arguments["tool_strictness"] = tool_strictness
        if reasoning_field is not None:
            arguments["reasoning_field"] = reasoning_field
        if session_affinity_id is not None:
            arguments["session_affinity_id"] = session_affinity_id
        return OpenAICompatibleProvider(**arguments)
    if provider == "anthropic_messages":
        if output_token_parameter is not None or tool_strictness is not None or reasoning_field is not None:
            raise PolicySessionError(
                "output_token_parameter, tool_strictness and reasoning_field apply only to OpenAI-compatible providers"
            )
        arguments: dict[str, Any] = {
            "base_url": base_url,
            "api_key": api_key,
            "timeout_seconds": timeout_seconds,
        }
        if recovery_policy != "legacy_v1":
            arguments["recovery_policy"] = recovery_policy
        if recovery_window_s is not None:
            arguments["recovery_window_s"] = float(recovery_window_s)
        if auth_scheme is not None:
            arguments["auth_scheme"] = auth_scheme
        if continuation_replay not in _ANTHROPIC_CONTINUATION_REPLAY_MODES:
            raise PolicySessionError("unsupported Anthropic continuation replay mode")
        if session_affinity_id is not None:
            arguments["session_affinity_id"] = session_affinity_id
        if standby_api_keys:
            arguments["standby_api_keys"] = tuple(standby_api_keys)
        if credential_start_index:
            arguments["credential_start_index"] = int(credential_start_index)
        return AnthropicMessagesProvider(**arguments)
    raise PolicySessionError(f"unsupported native provider protocol: {provider}")
