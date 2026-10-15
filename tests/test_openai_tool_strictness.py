"""A profile can state function-tool strictness on the OpenAI-compatible wire.

The Codex gateway's Responses backend decodes a function tool without
``strict`` as strict when its schema allows it (probe 2026-10-03): GPT-6 Astra
then fills every optional key. ``tool_strictness: explicit_false`` sends
``"strict": false`` on every function tool of the final wire body, and the
recorded wire is exactly the body ``complete`` sends.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from ora.agent.native.request_assembly import NativeRequestAssembler
from ora.agent.providers.native import OpenAICompatibleProvider, build_native_provider
from ora.agent.providers.profiles import NativeProviderProfileError, load_provider_profiles

ROOT = Path(__file__).resolve().parents[1]
TOOLS = [
    {"type": "function", "function": {"name": "geometry_query", "description": "d", "parameters": {
        "type": "object", "required": ["camera"], "additionalProperties": False,
        "properties": {"camera": {"type": "string"}, "region": {"type": "array"}}}}},
    {"type": "function", "function": {"name": "artifact_read", "description": "d", "parameters": {
        "type": "object", "properties": {"ref": {"type": "string"}}}}},
]


def _request():
    return {"model": "m", "messages": [{"role": "user", "content": "x"}], "tools": json.loads(json.dumps(TOOLS)),
            "tool_choice": "auto", "max_tokens": 64, "stream": False}


def test_the_class_projection_and_unset_profiles_keep_their_bytes():
    assert all("strict" not in tool["function"] for tool in OpenAICompatibleProvider.prepare_request(_request())["tools"])
    provider = build_native_provider("openai_compatible", base_url="https://example.invalid", api_key="test-secret-key")
    try:
        assert provider.prepare_request(_request()) == OpenAICompatibleProvider.prepare_request(_request())
    finally:
        provider.close()


def test_explicit_false_states_strict_on_every_function_tool_and_nothing_else():
    provider = build_native_provider(
        "openai_compatible", base_url="https://example.invalid", api_key="test-secret-key",
        output_token_parameter="max_completion_tokens", tool_strictness="explicit_false",
    )
    try:
        body = provider.prepare_request(_request())
        assert [tool["function"]["strict"] for tool in body["tools"]] == [False, False]
        assert body["max_completion_tokens"] == 64 and "max_tokens" not in body
        plain = OpenAICompatibleProvider.prepare_request(_request(), output_token_parameter="max_completion_tokens")
        for tool in body["tools"]:
            tool["function"].pop("strict")
        assert body == plain
    finally:
        provider.close()


def test_the_recorded_wire_is_the_body_complete_sends():
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"index": 0, "finish_reason": "stop",
                                                       "message": {"role": "assistant", "content": "ok"}}]})

    provider = OpenAICompatibleProvider(
        base_url="https://example.invalid", api_key="test-secret-key", output_token_parameter="max_completion_tokens",
        tool_strictness="explicit_false", client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assembler = NativeRequestAssembler(model="m", reasoning_effort="provider_default", max_output_tokens=64,
                                       stream=False, provider=provider)
    prepared = assembler.prepare({"messages": [{"role": "user", "content": "x"}], "tools": json.loads(json.dumps(TOOLS))})
    provider.complete(request=prepared.execution, model_call_id="call-0")
    assert sent == [dict(prepared.wire)]
    assert all(tool["function"]["strict"] is False for tool in sent[0]["tools"])


def test_anthropic_providers_refuse_the_option():
    with pytest.raises(Exception, match="tool_strictness"):
        build_native_provider("anthropic_messages", base_url="https://example.invalid", api_key="test-secret-key",
                              tool_strictness="explicit_false")


def test_the_profile_field_is_closed_and_openai_only(tmp_path):
    def catalog(protocol, value):
        profile = {
            "protocol": protocol, "endpoint": "https://example.invalid", "model": "m", "model_env": "M",
            "base_url_env": "B", "api_key_env": "K", "proxy_env": [], "reasoning_effort": "low",
            "context_window_tokens": 200000, "max_output_tokens": 16384, "stream": True, "supports_images": True,
            "qualification_schema": "ora.native-provider-probe.v1", "qualification_version": 1,
            "tool_strictness": value,
        }
        path = tmp_path / f"{protocol}-{value}.yaml"
        path.write_text(json.dumps({"schema": "ora.native-provider-catalog.v1", "profiles": {"p": profile}}))
        return path

    assert load_provider_profiles(catalog("openai_compatible", "explicit_false"))["p"]["tool_strictness"] == "explicit_false"
    for protocol, value in (("openai_compatible", "true"), ("anthropic_messages", "explicit_false")):
        with pytest.raises(NativeProviderProfileError, match="tool_strictness"):
            load_provider_profiles(catalog(protocol, value))


