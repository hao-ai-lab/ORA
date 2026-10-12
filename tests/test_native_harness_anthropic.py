"""Native harness anthropic contract regressions."""

from __future__ import annotations
import hashlib
import json
import re
import ssl
from typing import Any
import httpx
import pytest
from ora.orchestrator.coordinator import run_episode
from ora.agent.contracts import PolicySessionError
from ora.agent.native.session import NativePlanner
from ora.agent.providers.native import AnthropicMessagesProvider, FakeProvider, OpenAICompatibleProvider, normalize_anthropic_usage, translate_anthropic_request
from ora.agent.providers.continuation import PROVIDER_CONTINUATION_MESSAGE_KEY, ProviderContinuationState
from ora.agent.evidence.trajectory import canonical_bytes, verify_agent_trace
from test_native_harness import (
    AGENT_RULES,
    PROMPT,
    _AnthropicAttemptClient,
    native_plan,
    publish_native_boundary,
    usage,
)


def test_anthropic_request_translation_preserves_images_tools_and_pairing() -> None:
    request = {
        "model": "glm-5.2",
        "messages": [
            {"role": "system", "content": "stable policy"},
            {"role": "user", "content": [
                {"type": "text", "text": "observe"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,aGVsbG8="},
                },
            ]},
            {
                "role": "assistant", "content": "", "reasoning_content": "private",
                "tool_calls": [
                    {
                        "id": "read-1", "type": "function",
                        "function": {"name": "vision.inspect", "arguments": "{}"},
                    },
                    {
                        "id": "read-2", "type": "function",
                        "function": {
                            "name": "robot.read_manual",
                            "arguments": '{"section":"safety"}',
                        },
                    },
                ],
            },
            {
                "role": "tool", "tool_call_id": "read-1", "name": "vision.inspect",
                "content": '{"ok":true}',
            },
            {
                "role": "tool", "tool_call_id": "read-2", "name": "robot.read_manual",
                "content": '{"ok":false,"error":{"code":"missing"}}',
            },
            {"role": "user", "content": "continue"},
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "vision.inspect", "description": "Inspect image",
                "parameters": {"type": "object", "properties": {}},
            },
        }],
        "tool_choice": "auto", "max_tokens": 1024, "stream": True,
        "reasoning_effort": "high",
        "stream_options": {"include_usage": True},
    }
    wire = translate_anthropic_request(request)
    assert wire["system"][0] == {"type": "text", "text": "stable policy"}
    assert wire["messages"][0]["content"][1] == {
        "type": "image",
        "source": {
            "type": "base64", "media_type": "image/png", "data": "aGVsbG8=",
        },
    }
    assert [block["type"] for block in wire["messages"][1]["content"]] == [
        "tool_use", "tool_use",
    ]
    paired = wire["messages"][2]["content"]
    assert [block["type"] for block in paired] == ["tool_result", "tool_result", "text"]
    assert paired[1]["is_error"] is True
    assert "private" not in json.dumps(wire)
    alias = wire["tools"][0]["name"]
    assert alias != "vision.inspect"
    assert re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", alias)
    assert wire["messages"][1]["content"][0]["name"] == alias
    assert wire["messages"][1]["content"][1]["name"] != "robot.read_manual"
    assert wire["system"][1] == {
        "type": "text",
        "text": (
            "<wire_tool_aliases>\n"
            + json.dumps({"vision.inspect": alias}, sort_keys=True)
            + "\n</wire_tool_aliases>"
        ),
    }
    assert wire["tools"][0]["input_schema"] == {
        "type": "object", "properties": {},
    }
    assert wire["tool_choice"] == {"type": "auto"}
    assert "reasoning_effort" not in wire
    assert wire["thinking"] == {"type": "adaptive"}
    assert wire["output_config"] == {"effort": "high"}
    assert "stream_options" not in wire


def test_anthropic_provider_restores_canonical_dotted_tool_name() -> None:
    class Response:
        status_code = 200

        def __init__(self, alias: str) -> None:
            self.alias = alias

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def iter_lines(self):
            events = [
                    {
                        "type": "message_start",
                        "message": {"id": "msg-alias", "usage": {}},
                    },
                    {
                        "type": "content_block_start", "index": 0,
                    "content_block": {
                        "type": "tool_use", "id": "tool-1",
                        "name": self.alias, "input": {"section": "Safety"},
                        },
                    },
                    {"type": "content_block_stop", "index": 0},
                    {
                    "type": "message_delta",
                    "delta": {"stop_reason": "tool_use"},
                },
                {"type": "message_stop"},
            ]
            for event in events:
                yield "data: " + json.dumps(event)

    class Client:
        def __init__(self) -> None:
            self.alias = ""

        def stream(self, *_args, **kwargs):
            body = json.loads(kwargs["content"])
            self.alias = body["tools"][0]["name"]
            return Response(self.alias)

        def close(self):
            return None

    client = Client()
    provider = AnthropicMessagesProvider(
        "https://provider.invalid", "secret", client=client,
    )
    response = provider.complete(
        request={
            "model": "m",
            "messages": [{"role": "user", "content": "read safety"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "robot.read_manual", "description": "Read manual",
                    "parameters": {"type": "object", "properties": {}},
                },
            }],
            "max_tokens": 64, "stream": True,
        },
        model_call_id="alias-call",
    )
    assert client.alias != "robot.read_manual"
    assert response["wire_request"]["tools"][0]["name"] == client.alias
    tool_event = next(
        event for event in response["raw_events"]
        if event.get("type") == "content_block_start"
    )
    assert tool_event["content_block"]["name"] == client.alias
    assert response["message"]["tool_calls"][0]["function"]["name"] == (
        "robot.read_manual"
    )


def test_anthropic_provider_reassembles_fragmented_sse_and_usage() -> None:
    events = [
        {
            "type": "message_start",
            "message": {
                "id": "msg-signed-1",
                "usage": {
                    "input_tokens": 10,
                    "cache_read_input_tokens": 3,
                    "cache_creation_input_tokens": 2,
                },
            },
        },
        {
            "type": "content_block_start", "index": 0,
            "content_block": {
                "type": "thinking", "thinking": "look ", "signature": "",
            },
        },
        {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "carefully"},
        },
        {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "signature_delta", "signature": "signed-k3-state"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start", "index": 1,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta", "index": 1,
            "delta": {"type": "text_delta", "text": "I will inspect."},
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "content_block_start", "index": 2,
            "content_block": {
                "type": "tool_use", "id": "tool-1",
                "name": "robot.submit_action", "input": {},
            },
        },
        {
            "type": "content_block_delta", "index": 2,
            "delta": {"type": "input_json_delta", "partial_json": '{"action":'},
        },
        {
            "type": "content_block_delta", "index": 2,
            "delta": {"type": "input_json_delta", "partial_json": '"set_gripper"}'},
        },
        {"type": "content_block_stop", "index": 2},
        {
            "type": "message_delta", "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 5},
        },
        {"type": "message_stop"},
    ]

    class Response:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def iter_lines(self):
            for event in events:
                yield "data: " + json.dumps(event)

    class Client:
        def __init__(self):
            self.calls = []

        def stream(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return Response()

        def close(self):
            return None

    client = Client()
    provider = AnthropicMessagesProvider(
        "https://open.bigmodel.cn/api/anthropic", "secret", client=client,
    )
    request = {
        "model": "glm-5.2", "messages": [{"role": "user", "content": "act"}],
        "tools": [], "max_tokens": 1024, "stream": True,
    }
    response = provider.complete(request=request, model_call_id="anthropic-call")
    assert response["message"]["reasoning_content"] == "look carefully"
    assert response["message"]["content"] == "I will inspect."
    continuation = response["message"]["_provider_continuation_state"]
    assert continuation["provider_protocol"] == "anthropic_messages"
    assert continuation["model"] == "glm-5.2"
    assert continuation["response_id"] == "msg-signed-1"
    assert continuation["native_payload"]["content"][0] == {
        "type": "thinking",
        "thinking": "look carefully",
        "signature": "signed-k3-state",
    }
    call = response["message"]["tool_calls"][0]
    assert call["function"]["arguments"] == '{"action":"set_gripper"}'
    continued = translate_anthropic_request({
        "model": "glm-5.2",
        "messages": [
            {"role": "user", "content": "act"},
            response["message"],
            {
                "role": "tool", "tool_call_id": "tool-1",
                "name": "robot.submit_action", "content": '{"ok":true}',
            },
        ],
        "tools": [], "max_tokens": 1024, "stream": True,
    })
    assert continued["messages"][1]["content"] == [
        {
            "type": "thinking", "thinking": "look carefully",
            "signature": "signed-k3-state",
        },
        {"type": "text", "text": "I will inspect."},
        {
            "type": "tool_use", "id": "tool-1",
            "name": "robot.submit_action",
            "input": {"action": "set_gripper"},
        },
    ]
    assert "reasoning_content" not in json.dumps(continued)
    assert "_provider_continuation_state" not in json.dumps(continued)
    assert response["finish_reason"] == "tool_calls"
    assert response["usage"] == {
        "input_tokens": 15, "cached_input_tokens": 3, "output_tokens": 5,
        "reasoning_output_tokens": 0, "total_tokens": 20,
        "cache_read_input_tokens": 3, "cache_creation_input_tokens": 2,
        "cache_read_status": "reported", "cache_creation_status": "reported",
    }
    missing_cache = normalize_anthropic_usage({"input_tokens": 7, "output_tokens": 2})
    assert missing_cache["cached_input_tokens"] == 0
    assert missing_cache["cache_read_input_tokens"] is None
    assert missing_cache["cache_read_status"] == "missing"
    explicit_zero = normalize_anthropic_usage({
        "input_tokens": 7, "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0, "output_tokens": 2,
    })
    assert explicit_zero["cache_read_input_tokens"] == 0
    assert explicit_zero["cache_creation_input_tokens"] == 0
    assert explicit_zero["cache_read_status"] == "reported"
    assert explicit_zero["cache_creation_status"] == "reported"
    args, kwargs = client.calls[0]
    assert args[1] == "https://open.bigmodel.cn/api/anthropic/v1/messages"
    sent = json.loads(kwargs["content"])
    assert sent == response["wire_request"]
    assert "secret" not in json.dumps(sent)
    with pytest.raises(PolicySessionError, match="overloaded_error"):
        provider._parse_events(
            [{"type": "error", "error": {"type": "overloaded_error"}}],
            "error-call", "glm-5.2",
        )


def test_kimi_retries_one_exact_provider_issued_signature_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assistant: dict[str, Any] = {
        "role": "assistant",
        "content": "inspect",
        "tool_calls": [{
            "id": "tool-1",
            "type": "function",
            "function": {"name": "observe", "arguments": "{}"},
        }],
    }
    continuation = ProviderContinuationState.create(
        provider_protocol="anthropic_messages",
        model="k3",
        native_kind="anthropic_assistant_content_blocks.v1",
        native_payload={"content": [
            {"type": "thinking", "thinking": "inspect", "signature": "+/8="},
            {"type": "text", "text": "inspect"},
            {
                "type": "tool_use", "id": "tool-1", "name": "observe",
                "input": {},
            },
        ]},
        canonical_message=assistant,
        response_id="msg-kimi-signed",
    )
    assistant[PROVIDER_CONTINUATION_MESSAGE_KEY] = continuation.to_dict()
    request = {
        "model": "k3",
        "messages": [
            {"role": "user", "content": "inspect"},
            assistant,
            {
                "role": "tool", "tool_call_id": "tool-1",
                "name": "observe", "content": '{"ok":true}',
            },
        ],
        "tools": [],
        "max_tokens": 64,
        "stream": True,
    }

    class RejectedResponse:
        status_code = 400

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps({
                "error": {
                    "type": "invalid_request_error",
                    "message": (
                        "messages.1.content.0.signature: malformed encrypted "
                        "reasoning content: invalid base64url encoding"
                    ),
                },
                "type": "error",
            }).encode()

    class AcceptedResponse:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def iter_lines(self):
            events = [
                {
                    "type": "message_start",
                    "message": {"id": "msg-retry", "usage": {"input_tokens": 4}},
                },
                {
                    "type": "content_block_start", "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
                {
                    "type": "content_block_delta", "index": 0,
                    "delta": {"type": "text_delta", "text": "done"},
                },
                {"type": "content_block_stop", "index": 0},
                {
                    "type": "message_delta", "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 1},
                },
                {"type": "message_stop"},
            ]
            for event in events:
                yield "data: " + json.dumps(event)

    class Client:
        def __init__(self):
            self.bodies: list[bytes] = []

        def stream(self, *_args, **kwargs):
            self.bodies.append(kwargs["content"])
            return RejectedResponse() if len(self.bodies) <= 2 else AcceptedResponse()

        def close(self):
            return None

    client = Client()
    provider = AnthropicMessagesProvider(
        "https://api.kimi.com/coding", "secret", client=client,
    )
    delays: list[float] = []
    monkeypatch.setattr(
        "ora.agent.providers.native.time.sleep",
        lambda delay: delays.append(delay),
    )
    response = provider.complete(request=request, model_call_id="kimi-retry")
    assert response["attempts"] == 3
    assert response["message"]["content"] == "done"
    assert delays == [30.0, 60.0]
    assert len(client.bodies) == 3
    assert client.bodies[0] == client.bodies[1] == client.bodies[2]
    replay = json.loads(client.bodies[0])
    assert replay["messages"][1]["content"][0]["signature"] == "+/8="


def test_non_kimi_signature_rejection_is_not_retried() -> None:
    class Response:
        status_code = 400

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return (
                b'{"error":{"message":"malformed encrypted reasoning content: '
                b'invalid base64url encoding"}}'
            )

    class Client:
        def __init__(self):
            self.calls = 0

        def stream(self, *_args, **_kwargs):
            self.calls += 1
            return Response()

        def close(self):
            return None

    client = Client()
    provider = AnthropicMessagesProvider(
        "https://provider.invalid", "secret", client=client,
    )
    with pytest.raises(PolicySessionError, match="HTTP 400"):
        provider.complete(
            request={
                "model": "m", "messages": [{"role": "user", "content": "x"}],
                "tools": [], "max_tokens": 64, "stream": True,
            },
            model_call_id="not-kimi",
        )
    assert client.calls == 1

    kimi_client = Client()
    kimi_provider = AnthropicMessagesProvider(
        "https://api.kimi.com/coding", "secret", client=kimi_client,
    )
    with pytest.raises(PolicySessionError, match="HTTP 400"):
        kimi_provider.complete(
            request={
                "model": "k3", "messages": [{"role": "user", "content": "x"}],
                "tools": [], "max_tokens": 64, "stream": True,
            },
            model_call_id="kimi-without-signed-continuation",
        )
    assert kimi_client.calls == 1


def test_anthropic_retries_one_message_stop_with_missing_block_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ora.agent.providers.native.time.sleep", lambda _seconds: None,
    )

    class Response:
        status_code = 200

        def __init__(self, *, complete: bool) -> None:
            self.complete = complete

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def iter_lines(self):
            events = [
                {
                    "type": "message_start",
                    "message": {"id": "msg-safe-retry", "usage": {
                        "input_tokens": 3,
                    }},
                },
                {
                    "type": "content_block_start", "index": 0,
                    "content_block": {"type": "text", "text": "done"},
                },
            ]
            if self.complete:
                events.extend([
                    {"type": "content_block_stop", "index": 0},
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn"},
                        "usage": {"output_tokens": 1},
                    },
                ])
            events.append({"type": "message_stop"})
            for event in events:
                yield "data: " + json.dumps(event)

    class Client:
        calls = 0

        def stream(self, *_args, **_kwargs):
            self.calls += 1
            return Response(complete=self.calls == 2)

        def close(self):
            return None

    client = Client()
    provider = AnthropicMessagesProvider(
        "https://provider.invalid", "secret", client=client,
    )
    response = provider.complete(
        request={
            "model": "m", "messages": [{"role": "user", "content": "x"}],
            "tools": [], "max_tokens": 64, "stream": True,
        },
        model_call_id="safe-incomplete-block-retry",
    )
    assert client.calls == 2
    assert response["attempts"] == 2
    assert response["message"]["content"] == "done"


def test_anthropic_stream_refuses_a_late_delta_and_binds_its_timeouts() -> None:
    provider = AnthropicMessagesProvider(
        "https://api.kimi.com/coding", "secret", client=object(),
    )
    with pytest.raises(PolicySessionError, match="lacks its block start"):
        provider._parse_events([
            {"type": "message_start", "message": {"id": "msg-late-delta"}},
            {
                "type": "content_block_start", "index": 0,
                "content_block": {"type": "text", "text": "done"},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "content_block_delta", "index": 0,
                "delta": {"type": "text_delta", "text": "late"},
            },
        ], "late-delta-call", "glm-5.2")
    configured = AnthropicMessagesProvider(
        "https://provider.invalid", "secret", timeout_seconds=900,
    )
    assert configured.client.timeout.connect == 10.0
    assert configured.client.timeout.read == 900
    configured.close()


def test_anthropic_provider_preserves_and_rejects_invalid_tool_json() -> None:
    events = [
        {
            "type": "message_start",
            "message": {
                "id": "msg-invalid-tool-json",
                "usage": {"input_tokens": 12},
            },
        },
        {
            "type": "content_block_start", "index": 0,
            "content_block": {
                "type": "tool_use", "id": "tool-invalid",
                "name": "robot.submit_action", "input": {},
            },
        },
        {
            "type": "content_block_delta", "index": 0,
            "delta": {
                "type": "input_json_delta",
                "partial_json": '{"target_xyz_world_m":[0.1,',
            },
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "max_tokens"},
            "usage": {"output_tokens": 8192},
        },
        {"type": "message_stop"},
    ]

    class Response:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def iter_lines(self):
            for event in events:
                yield "data: " + json.dumps(event)

    class Client:
        calls = 0

        def stream(self, *_args, **_kwargs):
            self.calls += 1
            return Response()

        def close(self):
            return None

    client = Client()
    provider = AnthropicMessagesProvider(
        "https://provider.invalid", "secret", client=client,
    )
    response = provider.complete(
        request={
            "model": "k3", "messages": [{"role": "user", "content": "act"}],
            "tools": [], "max_tokens": 8192, "stream": True,
        },
        model_call_id="invalid-tool-json-call",
    )
    assert client.calls == 1
    assert response["raw_events"] == events
    assert response["finish_reason"] == "invalid_tool_input"
    assert response["message"] == {"role": "assistant", "content": ""}
    rejection = response["output_rejection"]
    assert rejection["code"] == "invalid_tool_input_json"
    assert rejection["provider_stop_reason"] == "max_tokens"
    assert rejection["discarded_tool_block_count"] == 1
    assert rejection["world_state_changed"] is False
    assert rejection["rejected_tool_blocks"][0]["reason"] == "invalid_json"
    assert "partial_json" not in rejection["rejected_tool_blocks"][0]
    assert response["usage"]["output_tokens"] == 8192
    provider.close()


def test_anthropic_continuation_fails_closed_on_tamper_or_model_drift() -> None:
    message = {
        "role": "assistant",
        "content": "",
        "reasoning_content": "inspect",
        "tool_calls": [{
            "id": "tool-1", "type": "function",
            "function": {"name": "observe", "arguments": "{}"},
        }],
    }
    from ora.agent.providers.continuation import ProviderContinuationState

    state = ProviderContinuationState.create(
        provider_protocol="anthropic_messages",
        model="k3",
        native_kind="anthropic_assistant_content_blocks.v1",
        native_payload={"content": [
            {"type": "thinking", "thinking": "inspect", "signature": "sig"},
            {
                "type": "tool_use", "id": "tool-1", "name": "observe",
                "input": {},
            },
        ]},
        canonical_message=message,
        response_id="msg-1",
    )
    message["_provider_continuation_state"] = state.to_dict()
    request = {
        "model": "k3", "messages": [message], "tools": [],
        "max_tokens": 32, "stream": True,
    }
    assert translate_anthropic_request(request)["messages"][0]["content"][0][
        "signature"
    ] == "sig"

    tampered = json.loads(json.dumps(request))
    tampered["messages"][0]["_provider_continuation_state"]["native_payload"][
        "content"
    ][0]["signature"] = "forged"
    with pytest.raises(PolicySessionError, match="state hash mismatch"):
        translate_anthropic_request(tampered)

    rewritten = json.loads(json.dumps(request))
    rewritten["messages"][0]["tool_calls"][0]["function"]["arguments"] = (
        '{"view":"wrist"}'
    )
    with pytest.raises(PolicySessionError, match="not bound"):
        translate_anthropic_request(rewritten)

    wrong_model = dict(request)
    wrong_model["model"] = "another-model"
    with pytest.raises(PolicySessionError, match="model does not match"):
        translate_anthropic_request(wrong_model)


def test_anthropic_continuation_survives_dynamic_tool_projection() -> None:
    """Historical signed tool use must not depend on the next tool set."""

    from ora.agent.providers.continuation import ProviderContinuationState

    message = {
        "role": "assistant",
        "content": "",
        "reasoning_content": "save the durable lesson",
        "tool_calls": [{
            "id": "tool-write", "type": "function",
            "function": {
                "name": "memory.write",
                "arguments": '{"files":[],"remove_paths":[]}',
            },
        }],
    }
    state = ProviderContinuationState.create(
        provider_protocol="anthropic_messages",
        model="k3",
        native_kind="anthropic_assistant_content_blocks.v1",
        native_payload={"content": [
            {
                "type": "thinking", "thinking": "save the durable lesson",
                "signature": "signed-memory-write",
            },
            {
                "type": "tool_use", "id": "tool-write",
                "name": "memory_write_historical_alias",
                "input": {"files": [], "remove_paths": []},
            },
        ]},
        canonical_message=message,
        response_id="msg-write",
    )
    message["_provider_continuation_state"] = state.to_dict()
    wire = translate_anthropic_request({
        "model": "k3",
        "messages": [message, {
            "role": "tool", "tool_call_id": "tool-write",
            "name": "memory.write", "content": '{"status":"accepted"}',
        }],
        # Finalization intentionally hides the historical memory.write tool.
        "tools": [{
            "type": "function",
            "function": {
                "name": "memory.finish", "description": "finish",
                "parameters": {"type": "object", "properties": {}},
            },
        }],
        "max_tokens": 8192,
        "stream": True,
    })
    assert wire["messages"][0]["content"][1] == {
        "type": "tool_use", "id": "tool-write",
        "name": "memory_write_historical_alias",
        "input": {"files": [], "remove_paths": []},
    }


def test_anthropic_continuation_preserves_redacted_thinking_and_rejects_unsigned() -> None:
    message, _usage, finish, rejection = AnthropicMessagesProvider._parse_events(
        [{
            "id": "msg-redacted",
            "content": [
                {"type": "redacted_thinking", "data": "opaque-ciphertext"},
                {"type": "text", "text": "continue"},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 2, "output_tokens": 1},
        }],
        "call-redacted",
        "k3",
        {},
    )
    assert rejection is None
    assert finish == "stop"
    assert message["content"] == "continue"
    wire = translate_anthropic_request({
        "model": "k3", "messages": [message], "max_tokens": 32,
    })
    assert wire["messages"][0]["content"][0] == {
        "type": "redacted_thinking", "data": "opaque-ciphertext",
    }

    with pytest.raises(PolicySessionError, match="replayable signature"):
        AnthropicMessagesProvider._parse_events(
            [{
                "content": [{"type": "thinking", "thinking": "unsigned"}],
                "stop_reason": "end_turn",
            }],
            "call-unsigned",
            "k3",
            {},
        )


def test_anthropic_provider_supports_explicit_bearer_auth_gateway() -> None:
    class Response:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps({
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 3, "output_tokens": 1},
            }).encode()

    class Client:
        def __init__(self):
            self.calls = []

        def stream(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return Response()

        def close(self):
            return None

    client = Client()
    provider = AnthropicMessagesProvider(
        "https://ark.cn-beijing.volces.com/api/plan",
        "secret",
        auth_scheme="bearer",
        session_affinity_id="ora:run-1:native",
        client=client,
    )
    response = provider.complete(
        request={
            "model": "glm-5.2",
            "messages": [{"role": "user", "content": "reply ok"}],
            "max_tokens": 16,
            "stream": False,
        },
        model_call_id="volcengine-auth",
    )
    args, kwargs = client.calls[0]
    assert args[1] == "https://ark.cn-beijing.volces.com/api/plan/v1/messages"
    assert kwargs["headers"]["authorization"] == "Bearer secret"
    assert "x-api-key" not in kwargs["headers"]
    assert kwargs["headers"]["x-session-id"] == "ora:run-1:native"
    assert response["message"]["content"] == "ok"


def test_anthropic_provider_sends_stable_pool_session_header() -> None:
    class Response:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps({
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 3, "output_tokens": 1},
            }).encode()

    class Client:
        def __init__(self):
            self.calls = []

        def stream(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return Response()

        def close(self):
            return None

    client = Client()
    provider = AnthropicMessagesProvider(
        "http://127.0.0.1:18317", "secret", client=client,
        session_affinity_id="ora:run-1:native-policy-1",
    )
    provider.complete(request={
        "model": "k3", "messages": [{"role": "user", "content": "ok"}],
        "max_tokens": 16, "stream": False,
    }, model_call_id="pool-session")
    _args, kwargs = client.calls[0]
    assert kwargs["headers"]["x-session-id"] == (
        "ora:run-1:native-policy-1"
    )


@pytest.mark.parametrize("error_type", [httpx.ReadError, ssl.SSLEOFError])
def test_anthropic_provider_zero_budget_does_not_retry_a_partial_stream(
    monkeypatch: pytest.MonkeyPatch,
    error_type,
) -> None:
    monkeypatch.setattr("ora.agent.providers.native.time.sleep", lambda _seconds: None)

    class PartialResponse:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def iter_lines(self):
            yield (
                'data: {"type":"content_block_delta","index":0,"delta":'
                '{"type":"text_delta","text":"partial"}}'
            )
            raise error_type("connection lost")

    class Client:
        calls = 0

        def stream(self, *_args, **_kwargs):
            self.calls += 1
            return PartialResponse()

        def close(self):
            return None

    client = Client()
    provider = AnthropicMessagesProvider(
        "https://provider.invalid", "secret", client=client, max_retries=0,
    )
    with pytest.raises(Exception, match="before message_stop"):
        provider.complete(
            request={
                "model": "m", "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 64, "stream": True,
            },
            model_call_id="partial-anthropic",
        )
    assert client.calls == 1


@pytest.mark.parametrize("provider_type", [AnthropicMessagesProvider, OpenAICompatibleProvider])
@pytest.mark.parametrize("preflight", [False, True])
@pytest.mark.parametrize("error_type", [ssl.SSLEOFError, ssl.SSLCertVerificationError])
def test_native_tls_failure_retry_scope_and_attempt_evidence(monkeypatch, provider_type, preflight, error_type):
    monkeypatch.setattr("ora.agent.providers.native.time.sleep", lambda _seconds: None)

    class Client:
        calls = []

        def stream(self, *_args, **kwargs):
            self.calls.append(kwargs["content"])
            raise error_type("transport failure")

    client = Client()
    provider = provider_type("https://provider.invalid", "secret", client=client,
                             max_retries=2, preflight_probe=preflight)
    with pytest.raises(error_type) as caught:
        provider.complete(request={"model": "m", "messages": [{"role": "user", "content": "hi"}],
                                   "max_tokens": 64, "stream": True}, model_call_id="tls-retry")
    expected = 3 if error_type is ssl.SSLEOFError and not preflight else 1
    assert len(client.calls) == expected
    assert len(set(client.calls)) == 1  # Retry the same request; no history mutation.
    assert caught.value.provider_failure == {
        "exception_type": error_type.__name__, "cause_type": None,
        "attempts": expected, "response_started": False,
    }


def test_native_failed_request_records_attempts_without_accepting_tools(tmp_path):
    pointer, workspace, tools = publish_native_boundary(tmp_path)

    class FailedProvider(FakeProvider):
        def complete(self, **_kwargs):
            error = ssl.SSLEOFError("secret must not enter failure event")
            error.provider_failure = {"attempts": 3, "response_started": False}
            raise error

    planner = NativePlanner(tmp_path, "failed-provider", tools, FailedProvider([]),
        AGENT_RULES, PROMPT, "fake", "native_minimal_v1", 32768, max_output_tokens=2048)
    with pytest.raises(ssl.SSLEOFError):
        planner.run_observation(task_instruction="Move once.", context=pointer,
                                image_path=workspace / pointer["main_image"])
    planner.close()
    events = [json.loads(line) for line in (tmp_path / "agent-events.jsonl").read_text().splitlines()]
    failure = next(e for e in events if e["type"] == "provider_error")
    request = next(e for e in events if e["type"] == "provider_request")
    assert failure["causal_parent_id"] == request["event_id"]
    assert failure["payload"]["attempts"] == 3
    assert failure["payload"]["accepted"] is False
    assert "secret" not in json.dumps(failure)
    assert not any(e["type"] in {"provider_response", "action_proposed"} for e in events)
    assert not (tmp_path / "control/submitted-action.json").exists()


@pytest.mark.parametrize("mode", ["transport", "timeout", "eof", "tls_eof"])
@pytest.mark.parametrize("standby", [(), ("fake-backup-key",)])
def test_anthropic_partial_stream_recovers_exact_request(mode, standby, monkeypatch):
    monkeypatch.setattr("ora.agent.providers.native.time.sleep", lambda _: None)
    client = _AnthropicAttemptClient([mode, "success"])
    failures = []
    provider = AnthropicMessagesProvider(
        "https://offline.invalid", "fake-original-key", standby_api_keys=standby,
        client=client, attempt_observer=failures.append,
    )
    response = provider.complete(request={"model": "k3", "messages": [{"role": "user", "content": "test"}], "stream": True, "max_tokens": 16}, model_call_id="recovery")
    assert response["attempts"] == 2
    assert client.calls[0]["content"] == client.calls[1]["content"]
    assert client.calls[1]["headers"]["x-api-key"] == (standby[0] if standby else "fake-original-key")
    assert len(failures) == 1
    assert failures[0]["partial_event_count"] == 2
    assert failures[0]["retry_scheduled"] is True
    assert failures[0]["tool_execution_started"] is False
    assert failures[0]["error_chain"][-1]["type"] in {"ReadError", "ReadTimeout", "EOFError", "SSLEOFError"}


@pytest.mark.parametrize("mode,expected_attempts", [
    ("transport", 3), ("eof", 3), ("malformed_json", 1),
    ("parse_error", 1), ("credential_echo", 1),
    ("tls_eof", 3), ("tls_certificate", 1), ("tls_other", 1),
])
def test_anthropic_recovery_is_bounded_and_records_fatal_cause(mode, expected_attempts, monkeypatch):
    monkeypatch.setattr("ora.agent.providers.native.time.sleep", lambda _: None)
    client = _AnthropicAttemptClient([mode])
    failures = []
    provider = AnthropicMessagesProvider("https://offline.invalid", "fake-original-key", standby_api_keys=("fake-backup-key",), client=client, attempt_observer=failures.append)
    with pytest.raises(PolicySessionError) as caught:
        provider.complete(request={"model": "k3", "messages": [{"role": "user", "content": "test"}], "stream": True, "max_tokens": 16}, model_call_id="failure")
    assert len(client.calls) == len(failures) == expected_attempts
    assert failures[-1]["retry_scheduled"] is False
    assert [f["attempt"] for f in failures] == list(range(1, expected_attempts+1))
    serialized = json.dumps(failures)
    assert "fake-original-key" not in serialized and "fake-backup-key" not in serialized
    assert all(f["partial_events"] and f["error_chain"] for f in failures)
    assert caught.value.provider_failure["attempts"] == expected_attempts
    assert caught.value.provider_failure["response_started"] is True


@pytest.mark.parametrize("settings", [{"preflight_probe": True}, {"max_retries": 0}])
@pytest.mark.parametrize("mode", ["transport", "tls_eof", "eof", "http401", "http402", "http403", "http429"])
def test_anthropic_stream_recovery_preserves_one_shot_calls(settings, mode, monkeypatch):
    monkeypatch.setattr("ora.agent.providers.native.time.sleep", lambda _: pytest.fail("one-shot call slept"))
    client = _AnthropicAttemptClient([mode, "success"])
    provider = AnthropicMessagesProvider("https://offline.invalid", "fake-original-key", standby_api_keys=("fake-backup-key",), client=client, **settings)
    with pytest.raises(PolicySessionError):
        provider.complete(request={"model": "k3", "messages": [{"role": "user", "content": "test"}], "stream": True, "max_tokens": 16}, model_call_id="one-shot")
    assert len(client.calls) == 1
    assert provider.api_key == "fake-original-key"
    assert provider.credential_rotations() == []


@pytest.mark.parametrize("start_index", [0, 1])
def test_anthropic_auth_rotation_then_stream_recovery_share_the_pool(monkeypatch, start_index):
    monkeypatch.setattr("ora.agent.providers.native.time.sleep", lambda _: None)
    client = _AnthropicAttemptClient(["http401", "transport", "success"])
    failures = []
    provider = AnthropicMessagesProvider("https://offline.invalid", "fake-original-key", standby_api_keys=("fake-backup-key",), credential_start_index=start_index, client=client, attempt_observer=failures.append)
    result = provider.complete(request={"model": "k3", "messages": [{"role": "user", "content": "test"}], "stream": True, "max_tokens": 16}, model_call_id="mixed")
    assert result["attempts"] == 3
    assert [call["headers"]["x-api-key"] for call in client.calls] == ["fake-original-key", "fake-backup-key", "fake-original-key"]
    assert [f["credential_index"] for f in failures] == [start_index, (start_index + 1) % 2]


@pytest.mark.parametrize("recover", [True, False])
def test_native_records_failed_stream_without_executing_discarded_tool(tmp_path, monkeypatch, recover):
    monkeypatch.setattr("ora.agent.providers.native.time.sleep", lambda _: None)
    run = tmp_path / "run"
    run.mkdir()
    plan = native_plan()
    plan["policy"]["provider"] = "anthropic_messages"
    (run / "plan.json").write_text(json.dumps(plan))
    client = _AnthropicAttemptClient(["transport", "success"] if recover else ["transport"], tool=True)
    provider = AnthropicMessagesProvider("https://offline.invalid", "fake-original-key", standby_api_keys=("fake-backup-key",), client=client)
    result = run_episode(run_root=run, plan=plan, run_id="stream-m0", provider_override=provider)
    assert result["decisions"] == int(recover)
    assert result["state"] == ("completed" if recover else "incomplete")
    events = [json.loads(line) for line in (run / "agent-events.jsonl").read_text().splitlines()]
    failed = [e for e in events if e["type"] == "provider_attempt_failed"]
    assert len(failed) == (1 if recover else 3)
    assert all(e.get("tool_call_id") != "discarded" for e in events)
    assert len([e for e in events if e["type"] == "action_proposed"]) == int(recover)
    for e in failed:
        artifact = run / e["artifact_ref"]["path"]
        evidence = json.loads(artifact.read_text())
        assert evidence["error_chain"][-1]["type"] == "ReadError"
        assert evidence["partial_events"][1]["content_block"]["id"] == "discarded"
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == e["artifact_ref"]["sha256"]
    assert provider.attempt_observer is None
    if not recover:
        terminal = [event for event in events if event["type"] == "provider_error"]
        assert len(terminal) == 1
        assert terminal[0]["payload"] == {
            "exception_type": "_StreamInterrupted", "cause_type": "ReadError",
            "attempts": 3, "response_started": True, "accepted": False,
        }
        assert any("request_without_response" in error for error in verify_agent_trace(run)["errors"])
    if recover:
        assert verify_agent_trace(run)["status"] == "valid"
        failed[0]["payload"]["tool_execution_started"] = True
        previous = None
        for event in events:
            event["previous_event_sha256"] = previous
            event["event_sha256"] = hashlib.sha256(canonical_bytes({
                key: value for key, value in event.items() if key != "event_sha256"
            })).hexdigest()
            previous = event["event_sha256"]
        (run / "agent-events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
        assert any("provider_attempt_contract" in error for error in verify_agent_trace(run)["errors"])


def test_provider_failure_redaction_covers_escaped_keys_and_json_property_names():
    from ora.agent.providers.attempts import _failure_evidence

    key = 'fake-key-with-"quote\\slash'
    result = _failure_evidence(RuntimeError(key), keys=[key], events=[{key: key}])
    assert result["credentials_redacted"] is True
    assert result["partial_events"] == [{"[REDACTED]": "[REDACTED]"}]
    assert result["error_chain"][0]["message"] == "[REDACTED]"


def test_anthropic_provider_accepts_message_stop_before_transport_reset() -> None:
    class TerminalThenResetResponse:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def iter_lines(self):
            yield (
                'data: {"type":"message_start","message":'
                '{"id":"msg-terminal","usage":{"input_tokens":3}}}'
            )
            yield (
                'data: {"type":"content_block_start","index":0,'
                '"content_block":{"type":"text","text":""}}'
            )
            yield (
                'data: {"type":"content_block_delta","index":0,'
                '"delta":{"type":"text_delta","text":"ok"}}'
            )
            yield 'data: {"type":"content_block_stop","index":0}'
            yield (
                'data: {"type":"message_delta","delta":'
                '{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}'
            )
            yield 'data: {"type":"message_stop"}'
            raise httpx.ReadError("connection reset after terminal event")

    class Client:
        calls = 0

        def stream(self, *_args, **_kwargs):
            self.calls += 1
            return TerminalThenResetResponse()

        def close(self):
            return None

    client = Client()
    provider = AnthropicMessagesProvider(
        "https://provider.invalid", "secret", client=client,
    )
    response = provider.complete(
        request={
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 64,
            "stream": True,
        },
        model_call_id="terminal-reset-anthropic",
    )
    assert client.calls == 1
    assert response["finish_reason"] == "stop"
    assert response["message"]["content"] == "ok"


def test_anthropic_output_limit_rejects_even_complete_tool_json():
    events = [
        {"type": "message_start", "message": {"id": "capped-tools"}},
        {"type": "content_block_start", "index": 0, "content_block": {
            "type": "thinking", "thinking": "private reasoning", "signature": "retained-in-raw-only"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {
            "type": "tool_use", "id": "never-dispatch", "name": "move_pose", "input": {"targets": {}}}},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {"stop_reason": "max_tokens"}, "usage": {"output_tokens": 8192}},
        {"type": "message_stop"},
    ]
    provider = AnthropicMessagesProvider("https://provider.invalid", "secret", client=object())
    message, usage_value, finish, rejection = provider._parse_events(events, "cap", "model")
    assert message == {"role": "assistant", "content": ""}
    assert usage_value["output_tokens"] == 8192 and finish == "invalid_tool_input"
    assert rejection["code"] == "truncated_tool_call" and rejection["provider_stop_reason"] == "max_tokens"


@pytest.mark.parametrize('value', ['[]', 'null', '1', '"text"'])
def test_anthropic_non_object_tool_json_is_recoverably_rejected(value):
    events = [
        {'type': 'message_start', 'message': {'id': 'non-object'}},
        {'type': 'content_block_start', 'index': 0, 'content_block': {
            'type': 'tool_use', 'id': 'discard', 'name': 'move_pose', 'input': {}}},
        {'type': 'content_block_delta', 'index': 0, 'delta': {
            'type': 'input_json_delta', 'partial_json': value}},
        {'type': 'content_block_stop', 'index': 0},
        {'type': 'message_delta', 'delta': {'stop_reason': 'tool_use'},
         'usage': {'input_tokens': 12, 'output_tokens': 8}},
        {'type': 'message_stop'},
    ]
    message, usage, finish, rejection = AnthropicMessagesProvider._parse_events(
        events, 'call', 'k3')
    assert not message.get('tool_calls')
    assert finish == 'invalid_tool_input'
    assert rejection['rejected_tool_blocks'][0]['reason'] == 'not_an_object'
    assert rejection['world_state_changed'] is False
    assert usage['output_tokens'] == 8


def test_anthropic_unsigned_history_preserves_json_looking_string():
    wire = translate_anthropic_request({
        'model': 'k3', 'max_tokens': 100, 'messages': [
            {'role': 'assistant', 'content': '', 'tool_calls': [{
                'id': 'call', 'type': 'function', 'function': {
                    'name': 'note', 'arguments': json.dumps({'note': '{"x":1}'})}}]},
            {'role': 'tool', 'tool_call_id': 'call', 'content': 'ok'}],
        'tools': [{'type': 'function', 'function': {'name': 'note',
            'parameters': {'type': 'object', 'properties': {'note': {'type': 'string'}}}}}],
    })
    assert wire['messages'][0]['content'][0]['input'] == {'note': '{"x":1}'}


def test_anthropic_factory_refuses_openai_output_field():
    from ora.agent.providers.native import build_native_provider
    with pytest.raises(PolicySessionError,match="only to OpenAI"):
        build_native_provider("anthropic_messages",base_url="https://example.invalid",api_key="test",output_token_parameter="max_completion_tokens")
