"""Native provider loop, context projection, trace and training-export contracts."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import ssl
import subprocess
from pathlib import Path
from typing import Any

import httpx
import pytest

from ora.orchestrator.evaluation import evaluate, seal
from ora.orchestrator.metrics import build_runtime_metrics
from ora.orchestrator.spec import (
    RunSpecError, file_sha256, resolve_run_spec, validate_run_plan,
)
from ora.orchestrator.coordinator import run_episode
from ora.agent.context.native import (
    ContextCompactionRejected,
    NativeContextPipeline,
)
from ora.agent.contracts import PolicySessionError
from ora.agent.native.gateway import openai_tool_definitions
from ora.agent.native.session import NativePlanner
from ora.agent.providers.native import (
    AnthropicMessagesProvider, FakeProvider, OpenAICompatibleProvider,
    normalize_anthropic_usage, normalize_usage, translate_anthropic_request,
)
from ora.agent.providers.profiles import (
    BUILTIN_PROVIDER_CATALOG,
    load_builtin_provider_profiles,
    select_provider_profile,
)
from ora.agent.providers.continuation import (
    PROVIDER_CONTINUATION_MESSAGE_KEY,
    ProviderContinuationState,
)
from ora.agent.tools.robot import RobotToolRuntime, ToolError, manual_sections, tool_definitions
from ora.agent.evidence.trajectory import (
    ArtifactStore,
    TrajectoryError,
    TrajectoryRecorder,
    canonical_bytes,
    export_sft,
    verify_agent_trace,
)
from ora.environment import SyntheticSession
from ora.agent.context import ContextProjector
from ora.agent.evidence.policy_state import PolicyStateStore


ROOT = Path(__file__).resolve().parents[1]
PROMPT = ROOT / "ora/policy/core/thin-react-v1.md"
MANUAL = ROOT / "ora/policy/embodiments/aloha-agilex/manual-v1.md"
AGENT_RULES = ROOT / "ora/policy/planners/codex/workspace-v1.md"
BOUNDS = {"x": [-0.6, 0.6], "y": [-0.4, 0.4], "z": [0.7, 1.45]}


def test_builtin_volcengine_agent_plan_registration_is_text_tool_qualified() -> None:
    profile_id = "volcengine_agent_plan_glm_5_2_anthropic_v1"
    profiles = load_builtin_provider_profiles()
    profile = profiles[profile_id]
    assert profile == {
        "protocol": "anthropic_messages",
        "endpoint": "https://ark.cn-beijing.volces.com/api/plan",
        "model": "glm-5.2",
        "model_env": "VOLCENGINE_AGENT_PLAN_MODEL_ID",
        "base_url_env": "VOLCENGINE_AGENT_PLAN_BASE_URL",
        "api_key_env": "VOLCENGINE_AGENT_PLAN_API_KEY",
        "proxy_env": ["HTTPS_PROXY", "https_proxy"],
        "reasoning_effort": "provider_default",
        "context_window_tokens": 1048576,
        "max_output_tokens": 131072,
        "stream": True,
        "supports_images": False,
        "qualification_schema": "ora.native-provider-probe.v1",
        "qualification_version": 1,
        "auth_scheme": "bearer",
        "admission": "text_tool_qualified",
        "usage_scope": "agent_plan_api",
        "source_url": "https://www.volcengine.com/docs/82379/2373746?lang=zh",
    }
    assert select_provider_profile(
        BUILTIN_PROVIDER_CATALOG,
        profile_id=profile_id,
        default_profile_id=None,
    ) == (profile_id, profile)


def native_plan(*, terminal_actions: int = 1) -> dict:
    return {
        "schema": "ora.run-plan.v1",
        "plan_sha256": "native-test-plan",
        "plan_id": "native-m0-test",
        "experiment": "native-harness",
        "config": "native-minimal",
        "milestone": "M0",
        "output_workspace": "outputs/native-harness",
        "task": "synthetic_tool_boundary",
        "task_instruction": "Move safely once.",
        "seed": 0,
        "evidence_tier": "diagnostic",
        "policy": {
            "kind": "native_api",
            "provider": "openai_compatible",
            "endpoint": "https://provider.invalid/v1",
            "model": "fake-vision-model",
            "reasoning_effort": "provider_default",
            "base_url_env": "NATIVE_MODEL_BASE_URL",
            "api_key_env": "NATIVE_MODEL_API_KEY",
            "context_window_tokens": 32768,
            "max_output_tokens": 2048,
            "stream": True,
            "supports_images": True,
            "context_profile": "native_minimal_v1",
            "summary_policy": "disabled",
            "memory_policy": "disabled",
            "tool_allowlist": [item for item in (
                "observation.dump_state", "vision.inspect", "geometry.query",
                "robot.read_manual", "robot.submit_action",
            )],
        },
        "runtime": {
            "kind": "synthetic",
            "workspace_bounds_m": BOUNDS,
            "terminal_actions": terminal_actions,
        },
        "budgets": {
            "max_robot_decisions": terminal_actions,
            "max_model_calls": 6 * terminal_actions,
            "max_sim_steps": terminal_actions,
            "wall_time_s": 30,
        },
        "policy_context": {
            "codex_project": [{
                "path": "ora/policy/planners/codex/workspace-v1.md",
                "sha256": file_sha256(AGENT_RULES),
            }],
            "episode_prefix": [{
                "path": "ora/policy/core/thin-react-v1.md",
                "sha256": file_sha256(PROMPT),
            }],
            "tool_manual": [{
                "path": "ora/policy/embodiments/aloha-agilex/manual-v1.md",
                "sha256": file_sha256(MANUAL),
            }],
            "composition_sha256": "native-test-composition",
        },
        "source": {"commit": "test", "dirty": False},
        "qualification": {"required": False},
    }


def publish_native_boundary(tmp_path: Path):
    source = tmp_path / "policy-observations"
    environment = SyntheticSession(tmp_path / "native/environment", "native-boundary", 0)
    environment.start()
    observation, _ = environment.observe(source)
    store = PolicyStateStore(tmp_path, "native-boundary", "native-plan")
    state = store.publish(
        observation=observation,
        observation_root=source / observation["observation_id"],
        cause="reset",
        task_instruction="Move once.",
        previous_public_action=None,
        remaining_budget={"robot_decisions": 1, "model_calls": 8},
    )
    record = store.steps[-1]
    workspace = tmp_path / "sandbox/workspace"
    control = tmp_path / "control"
    projector = ContextProjector(workspace, control, PROMPT, MANUAL, AGENT_RULES)
    pointer = projector.publish(
        observation=observation,
        observation_root=source / observation["observation_id"],
        task_instruction="Move once.",
        remaining_budget={"robot_decisions": 1, "model_calls": 8},
        policy_state={
            "step": state["step"],
            "path": record["path"],
            "sha256": record["sha256"],
            "state_sha256": state["state_sha256"],
        },
    )
    tools = RobotToolRuntime(
        control, MANUAL, tmp_path / "events.jsonl", BOUNDS, state_root=store.root
    )
    return pointer, workspace, tools


def assistant_tool(call_id: str, name: str, arguments: dict, *, reasoning: str | None = None):
    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }],
    }
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return message


def usage(input_tokens: int, output_tokens: int = 10) -> dict:
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": 0,
        "output_tokens": output_tokens,
        "reasoning_output_tokens": 0,
        "total_tokens": input_tokens + output_tokens,
    }


def native_run_spec(tmp_path: Path) -> dict[str, Any]:
    """A minimal OpenAI-compatible native RunSpec over a committed scratch repository."""
    subprocess.run(["git", "init", "-q", tmp_path], check=True)
    subprocess.run(["git", "-C", tmp_path, "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", tmp_path, "config", "user.name", "Test"], check=True)
    for name in ("prompt.md", "manual.md", "agents.md"):
        (tmp_path / name).write_text(
            "## overview\n\nmanual\n" if name == "manual.md" else name,
            encoding="utf-8",
        )
    subprocess.run(["git", "-C", tmp_path, "add", "."], check=True)
    subprocess.run(["git", "-C", tmp_path, "commit", "-qm", "initial"], check=True)
    spec = {
        "schema": "ora.run-spec.v1",
        "experiment": "native", "config": "minimal", "milestone": "M0",
        "output_workspace": "outputs/native", "task": "synthetic",
        "task_instruction": "Move once.", "seed": 0, "evidence_tier": "diagnostic",
        "policy": {
            "kind": "native_api", "provider": "openai_compatible", "model": "model-a",
            "endpoint": "https://provider.invalid/v1/",
            "proxy_env": ["HTTPS_PROXY", "https_proxy"],
            "context_window_tokens": 32768,
        },
        "runtime": {"kind": "synthetic", "workspace_bounds_m": BOUNDS},
        "budgets": {"max_robot_decisions": 1, "max_model_calls": 2, "max_sim_steps": 1, "wall_time_s": 10},
        "policy_context": {
            "codex_project": ["agents.md"], "episode_prefix": ["prompt.md"],
            "tool_manual": ["manual.md"],
        },
        "qualification": {"required": False},
    }
    return spec


def test_native_runspec_binds_provider_capabilities_and_rejects_l5(
    tmp_path: Path,
) -> None:
    spec = native_run_spec(tmp_path)
    resolved = resolve_run_spec(spec, root=tmp_path)
    assert validate_run_plan(resolved) == resolved
    tampered = copy.deepcopy(resolved)
    tampered["budgets"]["max_model_calls"] += 1
    with pytest.raises(RunSpecError, match="hash does not match"):
        validate_run_plan(tampered)
    assert resolved["policy"]["context_profile"] == "native_minimal_v1"
    assert resolved["policy"]["base_url_env"] == "NATIVE_MODEL_BASE_URL"
    assert resolved["policy"]["endpoint"] == "https://provider.invalid/v1"
    assert resolved["policy"]["proxy_env"] == ["HTTPS_PROXY", "https_proxy"]
    assert resolved["policy"]["stream"] is True
    changed = json.loads(json.dumps(spec))
    changed["policy"]["base_url_env"] = "OTHER_NATIVE_BASE_URL"
    assert resolve_run_spec(changed, root=tmp_path)["plan_sha256"] != resolved["plan_sha256"]
    changed_profile = json.loads(json.dumps(spec))
    changed_profile["policy"]["provider_profile"] = "different-profile"
    assert (
        resolve_run_spec(changed_profile, root=tmp_path)["plan_sha256"]
        != resolved["plan_sha256"]
    )
    anthropic = json.loads(json.dumps(spec))
    anthropic["policy"]["provider"] = "anthropic_messages"
    anthropic_resolved = resolve_run_spec(anthropic, root=tmp_path)
    assert anthropic_resolved["policy"]["provider"] == "anthropic_messages"
    assert anthropic_resolved["plan_sha256"] != resolved["plan_sha256"]
    endpoint_changed = json.loads(json.dumps(spec))
    endpoint_changed["policy"]["endpoint"] = "https://other.invalid/v1"
    assert (
        resolve_run_spec(endpoint_changed, root=tmp_path)["plan_sha256"]
        != resolved["plan_sha256"]
    )
    invalid_endpoint = json.loads(json.dumps(spec))
    invalid_endpoint["policy"]["endpoint"] = "http://secret@example.test/v1?token=x"
    with pytest.raises(RunSpecError, match="must be HTTPS"):
        resolve_run_spec(invalid_endpoint, root=tmp_path)
    loopback = json.loads(json.dumps(anthropic))
    loopback["policy"].update({
        "endpoint": "http://127.0.0.1:18317",
        "endpoint_transport": "loopback_http_v1",
        "provider_session_affinity": "x_session_id_run_v1",
    })
    loopback_resolved = resolve_run_spec(loopback, root=tmp_path)
    assert loopback_resolved["policy"]["endpoint_transport"] == (
        "loopback_http_v1"
    )
    remote_http = json.loads(json.dumps(loopback))
    remote_http["policy"]["endpoint"] = "http://provider.invalid/v1"
    with pytest.raises(RunSpecError, match="must be HTTP loopback"):
        resolve_run_spec(remote_http, root=tmp_path)
    spec["policy"]["summary_policy"] = "provider_summary"
    with pytest.raises(RunSpecError, match="summary_policy=disabled"):
        resolve_run_spec(spec, root=tmp_path)


def test_manual_sections_drive_exact_runtime_specific_tool_enums(tmp_path: Path) -> None:
    franka_manual = ROOT / "ora/policy/embodiments/franka-panda/manual-public-geometry-v3.md"
    assert manual_sections(MANUAL) == (
        "Embodiment", "Cameras", "Saved state", "Frames", "Actions", "Safety",
    )
    assert manual_sections(franka_manual) == (
        "Identity and topology", "Controlled points", "Frames and units",
        "End-effector axes", "Motion capabilities", "Gripper semantics",
        "Sensors and cameras", "Preservation and evidence limits",
        "Dynamic facts and provenance",
    )
    definition = next(
        item for item in tool_definitions(
            ["robot.read_manual", "robot.submit_action"], manual_path=franka_manual,
        )
        if item["name"] == "robot.read_manual"
    )
    assert definition["inputSchema"]["properties"]["section"]["enum"] == [
        "Identity and topology", "Controlled points", "Frames and units",
        "End-effector axes", "Motion capabilities", "Gripper semantics",
        "Sensors and cameras", "Preservation and evidence limits",
        "Dynamic facts and provenance",
    ]
    duplicate = tmp_path / "duplicate.md"
    duplicate.write_text("## Same\nA\n## Same\nB\n", encoding="utf-8")
    with pytest.raises(ToolError, match="unique"):
        manual_sections(duplicate)
    empty = tmp_path / "empty.md"
    empty.write_text("# No sections\n", encoding="utf-8")
    with pytest.raises(ToolError, match="non-empty"):
        manual_sections(empty)


def test_native_provider_adapters_reject_cross_protocol_request_fields() -> None:
    canonical = {
        "model": "m", "messages": [{"role": "user", "content": "hello"}],
        "tools": [], "max_tokens": 64, "stream": True,
    }
    openai = OpenAICompatibleProvider.prepare_request(canonical)
    assert "messages" in openai and "input" not in openai
    anthropic = translate_anthropic_request(canonical)
    assert "messages" in anthropic and "input" not in anthropic
    for forbidden in (
        {"input": "hello"},
        {"system": "already wire-shaped"},
        {"thinking": {"type": "adaptive"}},
        {"output_config": {"effort": "medium"}},
    ):
        with pytest.raises(PolicySessionError, match="fields"):
            OpenAICompatibleProvider.prepare_request({**canonical, **forbidden})
        with pytest.raises(PolicySessionError, match="fields"):
            translate_anthropic_request({**canonical, **forbidden})


@pytest.mark.parametrize("provider_type", [AnthropicMessagesProvider, OpenAICompatibleProvider])
def test_native_bad_record_mac_uses_bounded_transport_retry(monkeypatch, provider_type):
    monkeypatch.setattr("ora.agent.providers.native.time.sleep", lambda _seconds: None)

    class Client:
        calls = []

        def stream(self, *_args, **kwargs):
            self.calls.append(kwargs["content"])
            raise ssl.SSLError(
                1,
                "[SSL: SSLV3_ALERT_BAD_RECORD_MAC] ssl/tls alert bad record mac",
            )

    client = Client()
    provider = provider_type(
        "https://provider.invalid", "secret", client=client, max_retries=2,
    )
    with pytest.raises(ssl.SSLError) as caught:
        provider.complete(
            request={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 64,
                "stream": True,
            },
            model_call_id="bad-record-mac-retry",
        )
    assert len(client.calls) == 3
    assert len(set(client.calls)) == 1
    assert caught.value.provider_failure == {
        "exception_type": "SSLError",
        "cause_type": None,
        "attempts": 3,
        "response_started": False,
    }


class _AnthropicAttemptClient:
    """Deterministic HTTP fixtures; never connects to a provider."""

    def __init__(self, modes, *, tool=False):
        self.modes = modes
        self.calls = []
        self.tool = tool

    def stream(self, *_args, **kwargs):
        index = len(self.calls)
        self.calls.append(kwargs)
        mode = self.modes[min(index, len(self.modes)-1)]
        body = json.loads(kwargs["content"])
        tool = self.tool

        class Response:
            status_code = int(mode[4:]) if mode.startswith("http") else 200

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self):
                return b"unauthorized"

            def iter_lines(self):
                def line(value):
                    return "data: " + json.dumps(value)
                yield line({"type": "message_start", "message": {"id": "fixture", "usage": {"input_tokens": 3}}})
                if tool:
                    name = next(t["name"] for t in body["tools"] if t["name"].startswith("robot_submit_action"))
                    block = {"type": "tool_use", "id": "accepted" if mode == "success" else "discarded", "name": name,
                             "input": {"action": "move_pose", "arm": "left", "position_m": [0, 0, 1.02], "orientation_xyzw": [0, 0, 0, 1]}}
                else:
                    block = {"type": "text", "text": "valid reply"}
                yield line({"type": "content_block_start", "index": 0, "content_block": block})
                if mode == "transport":
                    raise httpx.ReadError("fixture interrupted response")
                if mode == "timeout":
                    raise httpx.ReadTimeout("fixture read deadline")
                if mode == "tls_eof":
                    raise ssl.SSLEOFError("fixture TLS EOF")
                if mode == "tls_certificate":
                    raise ssl.SSLCertVerificationError("fixture invalid certificate")
                if mode == "tls_other":
                    raise ssl.SSLError("fixture non-EOF TLS error")
                if mode == "eof":
                    return
                if mode == "malformed_json":
                    yield "data: {broken"
                    return
                if mode == "parse_error":
                    yield line({"type": "content_block_delta", "index": 12, "delta": {"type": "text_delta", "text": "bad block"}})
                if mode == "credential_echo":
                    yield line({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "fake-original-key"}})
                    raise httpx.ReadError("echo fixture fake-backup-key")
                yield line({"type": "content_block_stop", "index": 0})
                yield line({"type": "message_delta", "delta": {"stop_reason": "tool_use" if tool else "end_turn"}, "usage": {"output_tokens": 2}})
                yield line({"type": "message_stop"})
                if mode == "terminal_reset":
                    raise httpx.ReadError("ignored after terminal")

        return Response()


def test_native_metrics_count_canonical_images_across_wire_protocols(
    tmp_path: Path,
) -> None:
    run = tmp_path / "anthropic-metrics-run"
    store = ArtifactStore(run)
    recorder = TrajectoryRecorder(run, "run", "session")
    canonical = {
        "model": "k3",
        "messages": [{
            "role": "user",
            "content": [{
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,YWJj"},
            }],
        }],
    }
    canonical_ref = store.put_json(canonical, kind="native_canonical_provider_request")
    wire_ref = store.put_json(
        translate_anthropic_request({**canonical, "max_tokens": 64, "stream": True}),
        kind="native_provider_wire_request",
    )
    recorder.append(
        "provider_request", model_call_id="call", artifact_ref=wire_ref,
        payload={"model": "k3", "canonical_request_ref": canonical_ref},
    )
    (run / "plan.json").write_text(json.dumps({
        "plan_sha256": "metrics-plan",
        "policy": {"kind": "native_api"},
    }))
    (run / "result.json").write_text(json.dumps({
        "run_id": "run", "plan_sha256": "metrics-plan", "wall_duration_s": 1,
    }))
    (run / "events.jsonl").write_text("")
    (run / "native/agent-session.json").write_text(json.dumps({
        "backend": "native_anthropic_messages_v1",
        "usage_events": [],
        "calls": [],
    }))

    metrics = build_runtime_metrics(run)
    assert metrics["communication"]["inline_images_per_request"]["total"] == 1.0
    assert metrics["communication"]["inline_image_bytes"]["total"] == 3.0
    assert metrics["harness_input"]["images"]["count"] == 1


def test_declared_property_types_reads_tool_schemas() -> None:
    from ora.agent.providers.native import _declared_property_types

    declared = _declared_property_types([
        {
            "name": "move_pose",
            "input_schema": {
                "type": "object",
                "properties": {
                    "targets": {"type": "object"},
                    "position_m": {"type": "array"},
                    "interaction": {"type": "string"},
                },
            },
        },
        {"name": "no_schema"},
    ])
    assert declared == {
        "move_pose": {
            "targets": "object",
            "position_m": "array",
            "interaction": "string",
        }
    }


def test_stringified_arguments_decode_by_declared_type() -> None:
    """A vector left as text is repaired; a genuine string is not.

    Both halves matter. The measured EXP09A rejections were dominated by
    "position_m": "[-0.119, -0.003, 0.975]", so arrays have to be recovered.
    But deciding from the value's shape would rewrite a real string argument
    that happens to look like JSON, so the tool's own schema decides.
    """

    from ora.agent.providers.native import (
        _decode_stringified_object_arguments,
    )

    declared = {
        "targets": "object",
        "position_m": "array",
        "note": "string",
        "interaction": "string",
    }
    repaired = _decode_stringified_object_arguments(
        {
            "targets": '{"right": {"mode": "absolute"}}',
            "position_m": "[-0.119, -0.003, 0.975]",
            "note": "[not, really, json]",
            "interaction": "free_space",
        },
        declared,
    )
    assert repaired["targets"] == {"right": {"mode": "absolute"}}
    assert repaired["position_m"] == [-0.119, -0.003, 0.975]
    assert repaired["note"] == "[not, really, json]"
    assert repaired["interaction"] == "free_space"


def test_stringified_arguments_leave_mismatched_and_malformed_alone() -> None:
    from ora.agent.providers.native import (
        _decode_stringified_object_arguments,
    )

    declared = {"targets": "object", "position_m": "array"}
    repaired = _decode_stringified_object_arguments(
        {
            # Declared an object, parses to an array: not a transport artefact.
            "targets": "[1, 2]",
            # Truncated JSON, observed in the pilot: must survive untouched and
            # still reach the ToolSurface as the rejection it is.
            "position_m": '[0.04, 0.02',
        },
        declared,
    )
    assert repaired["targets"] == "[1, 2]"
    assert repaired["position_m"] == '[0.04, 0.02'


def test_stringified_arguments_stay_conservative_without_schema() -> None:
    from ora.agent.providers.native import (
        _decode_stringified_object_arguments,
    )

    repaired = _decode_stringified_object_arguments(
        {"targets": '{"right": 1}', "position_m": "[1, 2, 3]"}
    )
    assert repaired["targets"] == {"right": 1}
    assert repaired["position_m"] == "[1, 2, 3]"


@pytest.mark.parametrize("parameter", [None, "max_tokens", "max_completion_tokens"])
def test_native_provider_factory_accepts_output_parameter_and_projects_wire(parameter):
    from ora.agent.providers.native import build_native_provider
    provider=build_native_provider("openai_compatible",base_url="https://example.invalid",api_key="test",output_token_parameter=parameter)
    request={"model":"test","messages":[],"max_tokens":16384}
    try:
        body=provider.prepare_request(request,output_token_parameter=provider.output_token_parameter)
        field=parameter or "max_tokens"
        assert body[field]==16384
        assert ("max_tokens" in body)!=("max_completion_tokens" in body)
        assert request["max_tokens"]==16384
    finally:
        provider.close()
