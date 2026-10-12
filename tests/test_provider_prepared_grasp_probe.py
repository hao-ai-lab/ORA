"""Bounded provider diagnostic uses real adapter normalization with mock HTTP only."""
from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import httpx
import pytest

from experiments.runtime_v1 import provider_prepared_grasp_probe as probe
from ora.agent.composition import resolve_tool_surface
from ora.agent.providers.native import AnthropicMessagesProvider, OpenAICompatibleProvider
from ora.agent.context.skill_observations import materialize_skill_observations
from test_public_robot_facts import fixture
from test_skill_observation_delivery import _baseline

KEY = "fixture_secret_that_must_not_appear_in_artifacts"
PROFILE_ID = probe.KIMI_PROFILE


@pytest.fixture(autouse=True, params=[probe.KIMI_PROFILE, probe.DEEPSEEK_PROFILE])
def selected_profile(request, monkeypatch):
    monkeypatch.setitem(globals(), "PROFILE_ID", request.param)


def _openai_stream(blocks, stop):
    delta = {"role": "assistant", "content": "", "reasoning_content": "Check public evidence"}
    calls = []
    for block in blocks:
        if block["type"] == "text":
            delta["content"] += block["text"]
        if block["type"] == "tool_use":
            calls.append({"index": len(calls), "id": block["id"], "type": "function",
                "function": {"name": block["name"], "arguments": json.dumps(block["input"])}})
    if calls:
        delta["tool_calls"] = calls
    event = {"id": "resp-id", "choices": [{"delta": delta,
        "finish_reason": "tool_calls" if stop == "tool_use" else "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 9, "total_tokens": 21}}
    return "data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n"



def _stream(blocks, stop):
    events = [{"type": "message_start", "message": {"id": "resp-id", "type": "message",
        "role": "assistant", "content": [], "model": "k3", "usage": {"input_tokens": 12}}}]
    for index, block in enumerate(blocks):
        events += [{"type": "content_block_start", "index": index, "content_block": block},
                   {"type": "content_block_stop", "index": index}]
    events += [{"type": "message_delta", "delta": {"stop_reason": stop}, "usage": {"output_tokens": 9}},
               {"type": "message_stop"}]
    return "\n\n".join("data: " + json.dumps(item) for item in events) + "\n\n"


def _tool(name="grasp_execute", candidate="grasp-fixture"):
    return {"type": "tool_use", "id": "tool-id", "name": name,
            "input": {"candidate_id": candidate, "rationale": "Inspect the selected public candidate."}}


def _setup(tmp_path, monkeypatch, *, propose=None, review=None, fail=False):
    profile = probe.load_probe_binding(PROFILE_ID)["profile"]
    for name, value in ((profile["model_env"], profile["model"]),
                        (profile["base_url_env"], profile["endpoint"]), (profile["api_key_env"], KEY)):
        monkeypatch.setenv(name, value)
    wire_requests = []
    proposal = [_tool()] if propose is None else propose
    review = [{"type": "text", "text": "Approach stopped; lift did not run. Holding remains unknown."}] if review is None else review
    responses = [([{"type": "thinking", "thinking": "Check public evidence", "signature": "c2lnbmF0dXJl"}, *proposal],
                  "tool_use" if any(b["type"] == "tool_use" for b in proposal) else "end_turn"),
                 (review, "tool_use" if any(b["type"] == "tool_use" for b in review) else "end_turn")]
    def transport(request):
        wire_requests.append(json.loads(request.content))
        if fail:
            return httpx.Response(429, text="secret echo " + KEY)
        blocks, stop = responses[len(wire_requests)-1]
        return httpx.Response(200, text=(_stream if profile["protocol"] == "anthropic_messages" else _openai_stream)(blocks, stop), headers={"content-type": "text/event-stream"})
    client = httpx.Client(transport=httpx.MockTransport(transport))
    def build(protocol, **kwargs):
        assert protocol == profile["protocol"] and kwargs["timeout_seconds"] == 180
        provider_class = AnthropicMessagesProvider if protocol == "anthropic_messages" else OpenAICompatibleProvider
        return provider_class(client=client, **kwargs)
    monkeypatch.setattr(probe, "build_native_provider", build)
    surface, observation, observation_root = fixture(tmp_path, "robotwin", sequence=1)
    current = resolve_tool_surface("robotwin_toolchain_b2_closed_noservo_v1")
    host = SimpleNamespace(surface=SimpleNamespace(view_config=surface.view_config, supports=current.supports),
        manual_path=probe.SOURCE / "ora/policy/embodiments/aloha-agilex/manual-public-geometry-v3.md",
        definitions=lambda: current.definitions)
    case = {"arm": "right", "mode": "closed_start", "candidate": {"candidate_id": "grasp-fixture",
        "pregrasp_pose": {"position_m": [0, 0, 1], "orientation_xyzw": [0, 0, 0, 1]},
        "contact_pose": {"position_m": [0, 0, .97], "orientation_xyzw": [0, 0, 0, 1]},
        "private_source_path": "must_not_escape"}}
    instance = probe.ProviderGraspProbe(probe.load_probe_binding(PROFILE_ID), tmp_path, "right-closed_start")
    return instance, host, case, observation, observation_root, wire_requests, client


def _receipt():
    return {"schema": "ora.manipulation-skill-sequence-receipt.v1", "skill": "grasp_execute",
        "candidate_id": "grasp-fixture", "action_id": "action-1", "status": "aborted",
        "committed_sequence": ["open", "pregrasp", "approach", "close", "lift", "verify"],
        "stop_reason": "approach_not_reached", "stage_receipts": [
            {"stage": "open", "status": "executed"}, {"stage": "pregrasp", "status": "executed"},
            {"stage": "approach", "status": "not_reached", "reason": "approach_not_reached",
             "receipt": {"private_backend_detail": "must_not_escape"}}]}


def test_two_real_adapter_calls_preserve_continuation_images_and_public_receipt(tmp_path, monkeypatch):
    instance, host, case, observation, source, requests, client = _setup(tmp_path, monkeypatch)
    try:
        assert instance.propose(host, case, observation, source)["candidate_id"] == "grasp-fixture"
        baseline, _ = _baseline(tmp_path, ("head_camera", "front_camera", "left_camera", "right_camera"))
        images = materialize_skill_observations(receipt=baseline, run_root=tmp_path,
            workspace=tmp_path / "workspace", camera_names=["head_camera", "front_camera", "left_camera", "right_camera"])
        report = instance.review(host, _receipt(), observation, source, images)
        assert report["logical_calls"] == report["transport_attempts"] == len(requests) == 2
        assert report["calls"][0]["usage"]["output_tokens"] == 9
        assert report["projected_skill_effect"]["detail"]["stages_not_reached"] == ["close", "lift", "verify"]
        assistant = next(m for m in requests[1]["messages"] if m["role"] == "assistant")
        if PROFILE_ID == probe.KIMI_PROFILE:
            assert requests[1]["tool_choice"] == {"type": "none"}
            assert assistant["content"][0]["signature"] == "c2lnbmF0dXJl"
            assert all(r["model"] == "k3" and r["max_tokens"] == 8192 and r["output_config"]["effort"] == "high" for r in requests)
        else:
            assert requests[1]["tool_choice"] == "none"
            assert assistant["reasoning_content"] == "Check public evidence"
            assert all(r["model"] == "deepseek-flash" and r["max_tokens"] == 16384 and r["reasoning_effort"] == "low" for r in requests)
        assert report["profile_id"] == PROFILE_ID
        assert len(probe._image_payloads(requests[0])) == 2
        assert len(probe._image_payloads(requests[1])) == 6  # prior 2 + current 2 + baseline 2
        for index in (1, 2):
            response = json.loads((tmp_path / f"provider-probe/call-{index}-response.json").read_text())
            assert response["raw_events"]
            prepared = json.loads((tmp_path / f"provider-probe/call-{index}-wire-request.json").read_text())
            assert prepared == requests[index-1]
            if PROFILE_ID == probe.KIMI_PROFILE:
                assert response["wire_request"] == requests[index-1]
            assert json.loads((tmp_path / f"provider-probe/call-{index}-images.json").read_text())["original_bytes_verified"]
        assert "must_not_escape" not in json.dumps(requests)
        for path in (tmp_path / "provider-probe").iterdir():
            assert KEY not in path.read_text()
        with pytest.raises(ValueError, match="cannot be repeated"):
            instance.propose(host, case, observation, source)
        with pytest.raises(ValueError, match="preceding proposal"):
            instance.review(host, _receipt(), observation, source, [])
        assert len(requests) == 2
    finally:
        instance.close(); client.close()


@pytest.mark.parametrize("blocks", [[_tool(candidate="wrong")], [_tool(), _tool()],
    [_tool(name="set_gripper")], [{"type": "text", "text": "No call."}]])
def test_unpermitted_proposal_is_rejected_without_retry_or_mutation(tmp_path, monkeypatch, blocks):
    instance, host, case, observation, source, requests, client = _setup(tmp_path, monkeypatch, propose=blocks)
    try:
        with pytest.raises(ValueError):
            instance.propose(host, case, observation, source)
        assert len(requests) == 1
        assert not (tmp_path / "control/submitted-action.json").exists()
        with pytest.raises(ValueError):
            instance.review(host, _receipt(), observation, source, [])
    finally:
        instance.close(); client.close()


def test_review_cannot_propose_an_extra_action(tmp_path, monkeypatch):
    instance, host, case, observation, source, requests, client = _setup(tmp_path, monkeypatch, review=[_tool()])
    try:
        instance.propose(host, case, observation, source)
        with pytest.raises(ValueError, match="zero tool calls"):
            instance.review(host, _receipt(), observation, source, [])
        assert len(requests) == 2
    finally:
        instance.close(); client.close()


def test_http_failure_is_single_attempt_and_credential_safe(tmp_path, monkeypatch):
    instance, host, case, observation, source, requests, client = _setup(tmp_path, monkeypatch, fail=True)
    try:
        with pytest.raises(RuntimeError, match="bounded failure evidence") as error:
            instance.propose(host, case, observation, source)
        assert KEY not in str(error.value) and len(requests) == 1
        failure = json.loads((tmp_path / "provider-probe/call-1-failure.json").read_text())
        assert failure["attempts"] == 1 and failure["usage"] is None
        assert all(KEY not in path.read_text() for path in (tmp_path / "provider-probe").iterdir())
    finally:
        instance.close(); client.close()


def test_rgb_hash_drift_stops_before_request(tmp_path, monkeypatch):
    instance, host, case, observation, source, requests, client = _setup(tmp_path, monkeypatch)
    try:
        ref = observation["cameras"]["head_camera"]["files"]["rgb_lossless_png"]
        (source / ref["path"]).write_bytes(b"changed")
        with pytest.raises(ValueError, match="RGB source identity"):
            instance.propose(host, case, observation, source)
        assert not requests
    finally:
        instance.close(); client.close()


@pytest.mark.parametrize("field,value", [("model_env", "different"),
    ("base_url_env", "https://example.invalid"), ("api_key_env", "")])
def test_binding_refuses_substitution_before_client_creation(tmp_path, monkeypatch, field, value):
    binding = probe.load_probe_binding(PROFILE_ID)
    profile = binding["profile"]
    for name, expected in ((profile["model_env"], profile["model"]),
                           (profile["base_url_env"], profile["endpoint"]), (profile["api_key_env"], KEY)):
        monkeypatch.setenv(name, expected)
    monkeypatch.setenv(profile[field], value)
    monkeypatch.setattr(probe, "build_native_provider", lambda *a, **k: pytest.fail("must not construct client"))
    with pytest.raises(ValueError):
        probe.ProviderGraspProbe(binding, tmp_path, "case")
