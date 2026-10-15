"""Shared Native causal/atomic Runtime contracts and frozen EXP01B bindings."""

from __future__ import annotations

import base64
import copy
import dataclasses
import hashlib
import io
import json
import time
from pathlib import Path
from typing import Any, Mapping

import pytest
from PIL import Image

from ora.agent.composition import resolve_tool_surface
from ora.agent.context import ContextProjector
from ora.agent.contracts import PolicySessionError
from ora.agent.evidence.trajectory import canonical_bytes
from ora.agent.native import NativePlanner
from ora.agent.native.gateway import NativeToolGateway
from ora.agent.native.request_assembly import NativeRequestAssembler
from ora.agent.providers.native import translate_anthropic_request
from ora.agent.providers.continuation import ProviderContinuationState
from ora.agent.tools.runtime import ManipulationToolRuntime
from ora.environment import SyntheticSession
from ora.orchestrator.coordinator import run_episode
from ora.orchestrator.spec import RunSpecError, resolve_run_spec
from tests.agent.context.test_lossless_context_runtime import _pipeline as _lossless_pipeline
from tests.agent.native.test_multi_round_session import live_exp01b_spec


ROOT = Path(__file__).resolve().parents[3]
MANUAL = ROOT / "tests/data/prompts/tool-schema-delivery-v1.md"
PROMPT = ROOT / "tests/data/prompts/episode-task-v1.md"
PROJECT = ROOT / "ora/policy/environments/synthetic/boundary-v1.md"
BOUNDS = {"x": [-0.6, 0.6], "y": [-0.4, 0.4], "z": [0.7, 1.45]}


def _assistant_tool(
    call_id: str, name: str, arguments: Mapping[str, Any], *, reasoning: str | None = None,
) -> dict[str, Any]:
    message = {
        "role": "assistant", "content": "",
        "tool_calls": [{
            "id": call_id, "type": "function", "function": {
                "name": name,
                "arguments": json.dumps(dict(arguments), sort_keys=True),
            },
        }],
    }
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return message


def _latest_actual_message(request: Mapping[str, Any]) -> Mapping[str, Any]:
    for message in reversed(request["messages"]):
        content = message.get("content")
        if (
            message.get("role") == "user" and isinstance(content, list)
            and any(
                isinstance(block, Mapping)
                and "ACTUAL" in str(block.get("text", ""))
                for block in content
            )
        ):
            return message
    raise AssertionError("provider request lacks an actual observation message")


class AdaptiveCausalProvider:
    protocol = "anthropic_messages"
    prepare_request = staticmethod(translate_anthropic_request)

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.index = 0
        self.closed = False

    @staticmethod
    def _preview_id(request: Mapping[str, Any]) -> str:
        for message in reversed(request["messages"]):
            if message.get("role") != "tool" or message.get("name") != "preview_move":
                continue
            value = json.loads(message["content"])
            if isinstance(value.get("preview_id"), str):
                return value["preview_id"]
        raise AssertionError("preview result is absent from provider context")

    def complete(
        self, *, request: Mapping[str, Any], model_call_id: str,
    ) -> dict[str, Any]:
        self.requests.append(copy.deepcopy(dict(request)))
        script = (
            lambda: _assistant_tool("mark-0", "mark_point", {
                "point_id": "P0", "view": "agentview", "x": 2, "y": 1,
            }),
            lambda: _assistant_tool("preview-0", "preview_move", {
                "delta_mm": [1.0, 0.0, 0.0],
            }),
            lambda: _assistant_tool("move-0", "move_pose", {
                "preview_id": self._preview_id(request),
            }),
            lambda: _assistant_tool(
                "close-0", "close_gripper", {}, reasoning="private close reasoning",
            ),
            lambda: _assistant_tool("lift-0", "move_pose", {
                "delta_mm": [0.0, 0.0, 10.0],
            }),
            lambda: _assistant_tool("open-0", "open_gripper", {}),
            lambda: _assistant_tool("finish-0", "finish_episode", {
                "outcome": "failure", "reason": "synthetic qualification complete",
                "attempts": ["atomic pose, close, lift, and open completed"],
            }),
        )
        message = script[self.index]()
        self.index += 1
        return {
            "message": message,
            "finish_reason": "tool_calls",
            "usage": {
                "input_tokens": 1000 + self.index * 100,
                "cached_input_tokens": 800,
                "cache_read_input_tokens": 800,
                "output_tokens": 20,
                "reasoning_output_tokens": 5,
                "total_tokens": 1025 + self.index * 100,
                "cache_read_status": "reported",
                "cache_creation_status": "missing",
            },
            "raw_events": [{"fake": True, "index": self.index - 1}],
            "model_call_id": model_call_id,
            "started_unix_s": time.time(), "finished_unix_s": time.time(),
            "attempts": 1,
        }

    def close(self) -> None:
        self.closed = True


def _cache_prefix() -> list[dict[str, Any]]:
    return [
        {
            "role": "system", "content": "minimal harness",
            "_protected_context": True, "_context_source": "harness_system",
        },
        {
            "role": "user", "content": "environment and embodiment",
            "_protected_context": True,
            "_context_source": "project_instructions",
        },
        {
            "role": "user", "content": "exact episode task",
            "_protected_context": True, "_context_source": "episode_task",
        },
    ]


def _cache_actual(sequence: int) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": json.dumps({
                    "current_actual": {"actual_sequence": sequence},
                    "comparison": {"available": sequence > 0},
                }, sort_keys=True),
            },
            {
                "type": "text",
                "text": f"ACTION RESULT ACTUAL · sequence {sequence} · agentview",
            },
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
            },
        ],
        "_observation": {
            "step": sequence,
            "observation_id": f"internal-observation-{sequence}",
            "keyframe": True,
        },
        "_causal_observation": {
            "step": sequence, "result_actual_sequence": sequence,
        },
        "_session_round": 1,
    }


class _AnthropicProjectionProvider:
    prepare_request = staticmethod(translate_anthropic_request)


def _cache_assembler() -> NativeRequestAssembler:
    return NativeRequestAssembler(
        model="k3", reasoning_effort="medium", max_output_tokens=4096,
        stream=True, provider=_AnthropicProjectionProvider(),
    )


def test_cache_prefix_allows_terminal_tool_choice_escalation(tmp_path: Path) -> None:
    messages = [*_cache_prefix(), _cache_actual(0)]
    tools = [{
        "type": "function",
        "function": {
            "name": "memory.finish",
            "description": "Finalize the Memory transaction.",
            "parameters": {
                "type": "object", "properties": {},
                "additionalProperties": False,
            },
        },
    }]
    pipeline = _lossless_pipeline(tmp_path)
    assembler = NativeRequestAssembler(
        model="k3", reasoning_effort="provider_default",
        max_output_tokens=4096, stream=True,
        provider=_AnthropicProjectionProvider(),
    )
    initial = pipeline.project(messages=messages, tools=tools, observation_step=0)
    assembler.prepare(initial)
    terminal = pipeline.project(messages=messages, tools=tools, observation_step=0)
    prepared = assembler.prepare(
        terminal,
        tool_choice={
            "type": "function", "function": {"name": "memory.finish"},
        },
    )

    assert terminal["manifest"]["request_relation"] == "append_only"
    assert terminal["manifest"]["canonical_prefix_relation"]["status"] == "equal"
    assert prepared.canonical["tool_choice"]["function"]["name"] == "memory.finish"
    assert prepared.wire["tool_choice"]["type"] == "tool"
    assert prepared.wire["tool_choice"]["name"].startswith("memory_finish_")
    assert prepared.tool_choice_evidence["mode"] == "provider_named"


def test_anthropic_thinking_uses_runtime_validated_tool_requirement(
    tmp_path: Path,
) -> None:
    messages = [*_cache_prefix(), _cache_actual(0)]
    tools = [{
        "type": "function",
        "function": {
            "name": "memory.finish",
            "description": "Finalize the Memory transaction.",
            "parameters": {
                "type": "object", "properties": {},
                "additionalProperties": False,
            },
        },
    }]
    pipeline = _lossless_pipeline(tmp_path)
    assembler = _cache_assembler()
    projected = pipeline.project(messages=messages, tools=tools, observation_step=0)
    prepared = assembler.prepare(
        projected,
        tool_choice={
            "type": "function", "function": {"name": "memory.finish"},
        },
    )

    assert prepared.canonical["tool_choice"] == "auto"
    assert prepared.execution["tool_choice"] == "auto"
    assert prepared.wire["tool_choice"] == {"type": "auto"}
    assert prepared.wire["thinking"] == {"type": "adaptive"}
    assert prepared.tool_choice_evidence == {
        "requested": {
            "type": "function", "function": {"name": "memory.finish"},
        },
        "effective": "auto",
        "mode": "runtime_validated_auto",
        "reason": "provider_thinking_incompatible_with_named_choice",
    }


def test_cache_stable_request_rejects_silent_history_rewrite(tmp_path: Path) -> None:
    messages = [*_cache_prefix(), _cache_actual(0)]
    pipeline = _lossless_pipeline(tmp_path)
    assembler = _cache_assembler()
    first = pipeline.project(messages=messages, tools=[], observation_step=0)
    prepared = assembler.prepare(first)
    assert first["manifest"]["canonical_request_sha256"] == hashlib.sha256(
        canonical_bytes(prepared.canonical)
    ).hexdigest()

    rewritten = copy.deepcopy(messages)
    rewritten[-1]["content"][1]["text"] = "rewritten historical actual"
    rewritten.append(_cache_actual(1))
    # The refusal names the message, because the candidate request is
    # discarded with the run and nothing else records what changed. One live
    # occurrence cost an hour of reconstructing the rejected request from its
    # predecessor by hand, to learn an index this check had already computed.
    with pytest.raises(PolicySessionError, match="cache_prefix_rewritten") as raised:
        assembler.prepare(
            pipeline.project(messages=rewritten, tools=[], observation_step=1)
        )
    reported = str(raised.value)
    assert "at message " in reported, reported
    assert "was " in reported and ", now " in reported, reported


def test_signed_provider_continuation_survives_context_and_cache_projection(
    tmp_path: Path,
) -> None:
    assistant = {
        "role": "assistant", "content": "ready",
        "reasoning_content": "private thought",
    }
    assistant["_provider_continuation_state"] = ProviderContinuationState.create(
        provider_protocol="anthropic_messages",
        model="k3",
        native_kind="anthropic_assistant_content_blocks.v1",
        native_payload={"content": [
            {
                "type": "thinking", "thinking": "private thought",
                "signature": "signed-state",
            },
            {"type": "text", "text": "ready"},
        ]},
        canonical_message=assistant,
        response_id="msg-signed",
    ).to_dict()
    messages = [*_cache_prefix(), _cache_actual(0), assistant]
    pipeline = _lossless_pipeline(tmp_path)
    assembler = _cache_assembler()
    projected = pipeline.project(messages=messages, tools=[], observation_step=0)
    assert "reasoning_content" not in projected["messages"][-1]
    assert "_provider_continuation_state" in projected["messages"][-1]
    prepared = assembler.prepare(projected)
    assert "_provider_continuation_state" not in prepared.canonical["messages"][-1]
    assert "_provider_continuation_state" in prepared.execution["messages"][-1]
    assert prepared.wire["messages"][-1]["content"] == [
        {
            "type": "thinking", "thinking": "private thought",
            "signature": "signed-state",
        },
        {"type": "text", "text": "ready"},
    ]

    messages.append({"role": "user", "content": "continue"})
    appended = pipeline.project(messages=messages, tools=[], observation_step=0)
    appended_prepared = assembler.prepare(appended)
    assert appended["manifest"]["canonical_prefix_relation"]["status"] == (
        "exact_prefix"
    )
    assert appended["manifest"]["provider_wire_prefix_relation"]["status"] == (
        "exact_prefix"
    )
    assert appended_prepared.wire["messages"][-2]["content"][0][
        "signature"
    ] == "signed-state"


def _runtime(
    tmp_path: Path, *, tool_surface_profile: str = "native_atomic_causal_v1",
) -> ManipulationToolRuntime:
    source = tmp_path / "source"
    session = SyntheticSession(
        tmp_path / "environment", "atomic", 0, max_actions=8,
        main_camera="agentview", camera_names=("agentview", "wrist"),
    )
    session.start()
    observation, _images = session.observe(source)
    control = tmp_path / "control"
    workspace = tmp_path / "workspace"
    projector = ContextProjector(
        workspace, control, PROMPT, MANUAL, PROJECT,
        expose_manual=False, main_camera="agentview",
    )
    projector.publish(
        observation=observation,
        observation_root=source / observation["observation_id"],
        task_instruction="atomic qualification",
        remaining_budget={"robot_decisions": 8, "model_calls": 16},
        policy_state={
            "step": 0, "path": "unused", "sha256": "unused",
            "state_sha256": "unused",
        },
    )
    return ManipulationToolRuntime(
        control_root=control, manual_path=MANUAL,
        event_path=tmp_path / "events.jsonl", workspace_bounds_m=BOUNDS,
        state_root=None, tool_surface_profile=tool_surface_profile,
        controlled_end_effector="right",
    )


def _reopen_same_observation(runtime: ManipulationToolRuntime) -> None:
    """Test-only reset; production requires a newly published actual observation."""

    path = runtime.control_root / "current-observation.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["open"] = True
    path.write_text(json.dumps(value), encoding="utf-8")


def test_atomic_surface_cannot_construct_pose_gripper_composites(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    surface = resolve_tool_surface("native_atomic_causal_v1")
    assert runtime.definitions() == list(surface.definitions)
    assert NativeToolGateway(runtime).delivered() == runtime.definitions()
    assert surface.mutation_tools == ("move_pose", "close_gripper", "open_gripper")
    assert surface.call_effect_profile == "native_atomic_causal_effects_v1"
    assert runtime.classify_call_effect("preview_move", {}) == "read_only"
    assert runtime.classify_call_effect("move_pose", {}) == "mutation"
    assert runtime.classify_call_effect("close_gripper", {}) == "mutation"
    assert runtime.classify_call_effect("open_gripper", {}) == "mutation"

    preview = runtime.invoke("preview_move", {"delta_mm": [1.0, 0.0, 0.0]})
    assert preview["status"] == "candidate"
    assert not (runtime.control_root / "submitted-action.json").exists()
    moved = runtime.invoke("move_pose", {"preview_id": preview["preview_id"]})
    action = json.loads((runtime.control_root / "submitted-action.json").read_text())
    assert moved == {
        "status": "submitted", "action": "move_pose",
        "_runtime": {"directive": "action_submitted", "action_id": action["action_id"]},
    }
    assert action["schema"] == "ora-action.v3"
    assert action["action"] == "move_pose"
    assert "gripper" not in action["arguments"]

    (runtime.control_root / "submitted-action.json").unlink()
    _reopen_same_observation(runtime)
    runtime.invoke("close_gripper", {})
    close = json.loads((runtime.control_root / "submitted-action.json").read_text())
    assert close["action"] == "set_gripper"
    assert close["arguments"] == {"arm": "right", "gripper": 0.0}
    assert "position_m" not in close["arguments"]

    (runtime.control_root / "submitted-action.json").unlink()
    _reopen_same_observation(runtime)
    runtime.invoke("open_gripper", {})
    opened = json.loads((runtime.control_root / "submitted-action.json").read_text())
    assert opened["arguments"] == {"arm": "right", "gripper": 1.0}


#: The shared policy-view contract the robot defaults declare
#: (``shared_policy_view_config_v1``), lifted onto the Synthetic atomic runtime,
#: whose kept surface does not declare one.
ATOMIC_VIEW_CONFIG = {
    "schema": "ora.policy-view-config.v1",
    "rgb_resolution": "native",
    "derived_resolution": [256, 256],
    "automatic_actual_views": ["agentview"],
    "default_observe_views": ["agentview"],
    "coordinate_convention": "opencv_top_left_xy",
}


def test_atomic_policy_views_share_native_coordinates_and_one_camera_default(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    runtime.surface = dataclasses.replace(
        runtime.surface, view_config=ATOMIC_VIEW_CONFIG,
        capabilities=(*runtime.surface.capabilities, "shared_policy_view_config_v1"),
    )
    runtime.view_config = dict(ATOMIC_VIEW_CONFIG)
    preview = runtime.invoke("preview_move", {"delta_mm": [1.0, 0.0, 0.0]})
    assert preview["view_dimensions"] == {
        "candidate_agentview": {"width": 4, "height": 3},
    }
    observed = runtime.invoke("observe", {})
    assert observed["views"] == ["agentview"]
    assert observed["view_dimensions"] == {
        "agentview": {"width": 4, "height": 3},
    }
    image = Image.open(io.BytesIO(base64.b64decode(observed["images_base64"][0])))
    assert image.size == (4, 3)

    wrist = runtime.invoke("observe", {"views": ["wrist"]})
    assert wrist["views"] == ["wrist"]
    assert wrist["view_dimensions"]["wrist"] == {"width": 4, "height": 3}

    marked = runtime.invoke("mark_point", {
        "point_id": "native-center", "view": "agentview", "x": 2, "y": 1,
    })
    assert marked["status"] == "solved"
    point_store = json.loads(runtime.point_path.read_text(encoding="utf-8"))
    source = point_store["native-center"]["sources"][0]
    assert {key: source[key] for key in (
        "view", "x", "y", "source_x", "source_y",
    )} == {
        "view": "agentview", "x": 2, "y": 1,
        "source_x": 2, "source_y": 1,
    }
    assert source["observation_id"] == observed["observation_id"]
    assert source["measurement_quality"]["method"] == "local_supported_depth_v1"
    feedback = Image.open(io.BytesIO(base64.b64decode(marked["images_base64"][0])))
    assert feedback.size == (4, 3)

    pointcloud = runtime.invoke("observe", {"views": ["pointcloud_top"]})
    assert pointcloud["view_dimensions"] == {
        "pointcloud_top": {"width": 256, "height": 256},
    }
    derived = Image.open(io.BytesIO(base64.b64decode(pointcloud["images_base64"][0])))
    assert derived.size == (256, 256)


def test_atomic_preview_is_bound_to_one_actual_observation(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    preview = runtime.invoke("preview_move", {"delta_mm": [1.0, 0.0, 0.0]})
    pointer = runtime.control_root / "current-observation.json"
    current = json.loads(pointer.read_text(encoding="utf-8"))
    current["observation_id"] = "a-different-actual-observation"
    pointer.write_text(json.dumps(current), encoding="utf-8")
    from ora.agent.tools.robot import ToolError

    try:
        runtime.invoke("move_pose", {"preview_id": preview["preview_id"]})
    except ToolError as exc:
        assert exc.code == "stale_preview"
    else:
        raise AssertionError("stale preview was accepted")


def test_post_mutation_capture_failure_is_world_state_unknown(tmp_path: Path) -> None:
    class FailAfterMutation(SyntheticSession):
        def observe(self, output_root: Path):  # type: ignore[override]
            if self.action_sequence:
                raise RuntimeError("post-action capture failed")
            return super().observe(output_root)

    spec = live_exp01b_spec("synthetic-native-atomic-causal-cache-v1")
    plan = resolve_run_spec(spec, root=ROOT)
    run_root = tmp_path / "failed-run"
    run_root.mkdir()
    (run_root / "plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    provider = AdaptiveCausalProvider()
    # Start at the direct move_pose: the first call mutates the world.
    provider.index = 4
    session = FailAfterMutation(
        run_root / "external", plan["task"], 0, max_actions=6,
        main_camera="agentview", camera_names=("agentview", "wrist"),
    )
    result = run_episode(
        run_root=run_root, plan=plan, run_id="native-exp01b-capture-fault",
        provider_override=provider,
        environment_session_factory=lambda **_kwargs: session,
    )
    assert result["state"] == "incomplete"
    assert result["termination"] == "trusted_fault"
    interaction = json.loads(
        (run_root / "native/episode-interaction.json").read_text(encoding="utf-8")
    )
    assert interaction["world_state_certainty"] == "unknown"
    assert any(
        transition["to"] == "world_state_unknown"
        for transition in interaction["transitions"]
    )
    assert len(provider.requests) == 1


def test_causal_pose_effect_distinguishes_miss_from_unknown() -> None:
    planner = object.__new__(NativePlanner)
    missed = planner._minimal_action_effect({
        "action": {
            "action": "move_pose",
            "arguments": {"position_m": [0.1, 0.0, 1.0]},
        },
        "commanded_pose_xyz_xyzw": [0.1, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "reached_pose_xyz_xyzw": [0.2, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    })
    assert missed == {
        "action": "move_pose",
        "status": "not_reached",
        "target_pose_xyz_xyzw": [0.1, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "actual_pose_xyz_xyzw": [0.2, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "position_error_m": 0.1,
        "orientation_error_rad": 0.0,
    }
    assert planner._minimal_action_effect({
        "action": {"action": "move_pose", "arguments": {}},
    }) == {"action": "move_pose", "status": "world_state_unknown"}


def test_a_semantic_manifest_is_refused() -> None:
    """No capability manifest is registered, so policy.semantic_manifest is refused."""

    spec = live_exp01b_spec("synthetic-native-atomic-causal-cache-v1")
    spec["policy"]["semantic_manifest"] = {"id": "manipulation_core_draft_v0", "instance": "x"}
    with pytest.raises(RunSpecError, match="semantic_manifest is not supported"):
        resolve_run_spec(spec, root=ROOT)


def test_anthropic_wire_projection_states_object_type_for_oneof_tools() -> None:
    """Pool gateways require input_schema.type == "object" (EXP08A launch)."""

    body = translate_anthropic_request({
        "model": "k3",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "move_pose",
                "description": "d",
                "parameters": {
                    "$defs": {},
                    "oneOf": [
                        {"type": "object", "properties": {},
                         "additionalProperties": False},
                    ],
                },
            },
        }],
    })
    schema = body["tools"][0]["input_schema"]
    assert schema["type"] == "object"
    assert "oneOf" in schema


def test_provider_rotates_host_local_credentials_on_exhaustion() -> None:
    """A quota rejection moves to the next key and retries (host-local pool)."""

    from ora.agent.providers.native import AnthropicMessagesProvider

    class Stream:
        def __init__(self, status: int, body: bytes) -> None:
            self.status_code = status
            self._body = body

        def __enter__(self):
            return self

        def __exit__(self, *exc) -> bool:
            return False

        def read(self) -> bytes:
            return self._body

    class Client:
        def __init__(self) -> None:
            self.keys: list[str] = []

        def stream(self, method, url, headers, content):
            self.keys.append(headers.get("x-api-key"))
            return Stream(429, b'{"error":"rate_limit"}')

    client = Client()
    provider = AnthropicMessagesProvider(
        base_url="https://api.kimi.com/coding",
        api_key="key-one",
        standby_api_keys=("key-two", "key-three"),
        client=client,
    )
    for _ in range(3):
        try:
            provider._request_once({"model": "k3", "messages": [], "stream": False})
        except Exception as error:  # noqa: BLE001 - contract under test
            assert getattr(error, "retryable", False) is True
    assert client.keys == ["key-one", "key-two", "key-three"]
    rotations = provider.credential_rotations()
    assert [item["credential_index"] for item in rotations] == [1, 2, 3]
    assert all("api_key" not in json.dumps(item) for item in rotations)
    # The ring is cyclic: an exhausted key is one to come back to once the
    # provider's usage window resets, not one to discard.
    try:
        provider._request_once({"model": "k3", "messages": [], "stream": False})
    except Exception as error:  # noqa: BLE001 - contract under test
        assert getattr(error, "retryable", False) is True
        assert getattr(error, "retry_delay_s", 0) >= 60.0
        assert getattr(error, "retry_budget", 0) >= 8
    assert client.keys[-1] == "key-one"
    assert provider.credential_rotations()[-1]["completed_laps"] == 1


def test_credential_ring_spreads_a_batch_by_slot_ordinal() -> None:
    """Each concurrent slot starts on a different key; rotation order holds."""

    from ora.orchestrator.policy_factory import _credential_ring

    keys = ["key-one", "key-two", "key-three"]
    assert _credential_ring(keys[0], keys[1:], "gpu1") == (
        ["key-two", "key-three", "key-one"], 1,
    )
    assert _credential_ring(keys[0], keys[1:], "gpu2") == (
        ["key-three", "key-one", "key-two"], 2,
    )
    # gpu3 wraps back onto the first key; gpu4 lands on the second.
    assert _credential_ring(keys[0], keys[1:], "gpu3")[0][0] == "key-one"
    assert _credential_ring(keys[0], keys[1:], "gpu4")[0][0] == "key-two"
    # No slot identity and a single key both leave the ring untouched.
    assert _credential_ring(keys[0], keys[1:], None) == (keys, 0)
    assert _credential_ring("only", [], "gpu5") == (["only"], 0)


def test_provider_throttle_gets_patient_retry_budget(monkeypatch) -> None:
    """A 429 storm is retried with long backoff instead of a fast fatal fault."""

    from ora.agent.providers.native import AnthropicMessagesProvider

    class Stream:
        def __init__(self, status: int, body: bytes) -> None:
            self.status_code = status
            self._body = body

        def __enter__(self):
            return self

        def __exit__(self, *exc) -> bool:
            return False

        def read(self) -> bytes:
            return self._body

    class Client:
        def __init__(self) -> None:
            self.calls = 0

        def stream(self, method, url, headers, content):
            self.calls += 1
            return Stream(429, b'{"error":{"type":"rate_limit_error"}}')

    sleeps: list[float] = []
    monkeypatch.setattr(
        "ora.agent.providers.native.time.sleep", sleeps.append
    )
    client = Client()
    provider = AnthropicMessagesProvider(
        base_url="https://api.kimi.com/coding",
        api_key="key-one",
        client=client,
    )
    with pytest.raises(Exception) as excinfo:
        provider.complete(
            request={
                "model": "k3", "messages": [], "tools": [],
                "max_tokens": 512,
            },
            model_call_id="call-throttle",
        )
    assert "429" in str(excinfo.value)
    # max_retries stays 2 for other failures, but a throttle earns at least
    # seven total attempts with 20s-and-up waits between them.
    assert client.calls == 7
    assert len(sleeps) == 6
    assert sleeps[0] == pytest.approx(20.0)
    assert all(delay >= 20.0 for delay in sleeps)
    assert max(sleeps) <= 60.0
