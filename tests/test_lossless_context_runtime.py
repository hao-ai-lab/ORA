from __future__ import annotations

from ora.agent.tools.robot import ToolError

import base64
import copy
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.usefixtures("ready_model_services")

from ora.agent.context import ContextProjector
from ora.agent.context.native import NativeContextPipeline
from ora.agent.contracts import PolicySessionError, SummaryResult
from ora.agent.evidence.trajectory import ArtifactStore, canonical_bytes
from ora.agent.native.request_assembly import NativeRequestAssembler
from ora.agent.native.session import NativePlanner
from ora.agent.native.tool_loop import _provider_estimated_input_tokens
from ora.agent.providers.native import AnthropicMessagesProvider, FakeProvider
from ora.agent.memory.artifacts import (
    ContextArtifactCatalog, RecoverableArtifactToolRuntime,
    decode_recoverable_artifact_marker, encode_recoverable_artifact_marker,
    recoverable_artifact_envelope, recoverable_artifact_pointer,
    validate_recoverable_artifact_pointer,
)
from ora.agent.tools.grasp import DepthPromptSegmenter
from ora.agent.tools.runtime import ManipulationToolRuntime
from ora.environment import SyntheticSession


ROOT = Path(__file__).resolve().parents[1]
HARNESS_SYSTEM = ROOT / "tests/data/prompts/harness-system-v1.md"
EPISODE_TASK = ROOT / "tests/data/prompts/episode-task-v1.md"
PROJECT_POLICY = ROOT / "ora/policy/environments/libero_pro/boundary-v2.md"
TOOL_MANUAL = ROOT / "tests/data/prompts/tool-schema-delivery-v1.md"


MANAGEMENT = {
    "schema": "ora.native-context-management.v4",
    "causal_exact_tail_action_steps": 3,
    "inline_media_tail_action_steps": 2,
    "eager_large_result_bytes": 50_000,
    "eager_preview_bytes": 2_048,
    "eager_exact_deduplication": True,
    "trigger_inline_images": 2,
    "trigger_estimated_tokens": 1_000_000,
    "trigger_provider_input_ratio": 0.70,
    "episode_checkpoint_trigger_tokens": 10_000,
    "episode_checkpoint_trigger_ratio": 0.20,
    "emergency_fallback_ratio": 0.80,
    "emergency_target_ratio": 0.65,
    "checkpoint_target_estimated_tokens": 1200,
    "checkpoint_max_estimated_tokens": 1600,
}
TOOLS = [{
    "type": "function",
    "function": {
        "name": "move_pose", "description": "move",
        "parameters": {"type": "object"},
    },
}]
CHECKPOINT = {
    "task_progress": ["The object has not been confirmed as transported."],
    "confirmed_facts": ["The retained actual view is authoritative."],
    "invalidated_beliefs": ["An older alignment belief may be stale."],
    "failed_attempts": [{
        "failure": "The earlier attempt did not establish transport.",
        "evidence": "The retained evidence did not confirm attachment.",
        "correction": "Re-ground before the next grasp attempt.",
    }],
    "current_constraints": ["Use qualitative state only."],
    "do_not_repeat": ["Do not infer attachment from closure."],
    "re_ground_next": ["Inspect the current actual view."],
}


def _image(label: str) -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64," + base64.b64encode(
            label.encode("utf-8")
        ).decode("ascii")},
    }


def _png_block(data: bytes) -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(data).decode("ascii"),
        },
    }


def _replace_agentview_with_object(
    observation: dict[str, Any], observation_root: Path,
) -> None:
    camera_root = observation_root / "agentview"
    rgb_path = camera_root / "rgb.png"
    depth_path = camera_root / "depth-m.npy"
    calibration_path = camera_root / "calibration.json"
    rgb = np.full((64, 64, 3), [48, 52, 60], dtype=np.uint8)
    rgb[20:44, 24:40] = [210, 48, 32]
    Image.fromarray(rgb).save(rgb_path, format="PNG")
    depth = np.full((64, 64), 1.2, dtype=np.float32)
    depth[20:44, 24:40] = 0.8
    np.save(depth_path, depth, allow_pickle=False)
    calibration_path.write_text(json.dumps({
        "intrinsic_cv": [
            [400.0, 0.0, 32.0], [0.0, 400.0, 32.0], [0.0, 0.0, 1.0],
        ],
        "extrinsic_cv": [
            [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0],
        ],
        "cam2world_gl": [
            [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0],
        ],
        "depth_unit": "metres",
    }, sort_keys=True), encoding="utf-8")
    record = observation["cameras"]["agentview"]
    record["width"] = 64
    record["height"] = 64
    for role, path in {
        "rgb_lossless_png": rgb_path,
        "depth_numpy_m_f32": depth_path,
        "calibration": calibration_path,
    }.items():
        record["files"][role] = {
            "path": path.relative_to(observation_root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
    (observation_root / "observation.json").write_text(
        json.dumps(observation, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class _IndependentPointSegmenter(DepthPromptSegmenter):
    """Answers every point of a batch independently with the depth baseline."""

    def segment_points(
        self, *, rgb_u8: Any, depth_m: Any, points_uv: list[list[int]],
    ) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
        raw, queries = [], []
        for index, point in enumerate(points_uv):
            mask, _report = self.segment(
                rgb_u8=rgb_u8, depth_m=depth_m, point_uv=tuple(point),
            )
            raw.append({
                "query_index": index, "source_index": 0, "point_uv": point,
                "mask": np.asarray(mask, dtype=bool), "backend_score": 1.0,
            })
            queries.append({
                "query_index": index, "point_uv": point,
                "raw_candidate_count": 1, "error_code": None,
            })
        return raw, {"query_results": queries, "image_encodings": 1}


def _media_runtime(
    tmp_path: Path,
) -> tuple[ManipulationToolRuntime, str, bytes, dict[str, Any]]:
    source = tmp_path / "source"
    environment = SyntheticSession(
        tmp_path / "environment", "lossless-context-media", 0, max_actions=4,
        main_camera="agentview", camera_names=("agentview", "wrist"),
    )
    environment.start()
    observation, _images = environment.observe(source)
    observation_root = source / observation["observation_id"]
    _replace_agentview_with_object(observation, observation_root)
    control = tmp_path / "control"
    projector = ContextProjector(
        tmp_path / "workspace", control, EPISODE_TASK, TOOL_MANUAL,
        PROJECT_POLICY, expose_manual=False, main_camera="agentview",
    )
    pointer = projector.publish(
        observation=observation, observation_root=observation_root,
        task_instruction="segment and inspect the red object",
        remaining_budget={
            "robot_decisions": 4, "model_calls": 12, "session_round": 1,
        },
        policy_state={
            "step": 0, "path": "unused", "sha256": "unused",
            "state_sha256": "unused",
        },
    )
    runtime = ManipulationToolRuntime(
        control_root=control, manual_path=TOOL_MANUAL,
        event_path=tmp_path / "tool-events.jsonl", workspace_bounds_m={
            "x": [-0.6, 0.6], "y": [-0.4, 0.4], "z": [0.7, 1.45],
        }, state_root=None,
        tool_surface_profile="libero_placement_direct_overhead_v1",
        controlled_end_effector="right",
        segmentation_backend=_IndependentPointSegmenter(),
    )
    return runtime, str(observation["observation_id"]), (
        observation_root / "agentview/rgb.png"
    ).read_bytes(), pointer


def _protected() -> list[dict[str, Any]]:
    return [
        {
            "role": "system", "content": "system",
            "_protected_context": True, "_context_source": "harness_system",
        },
        {
            "role": "user", "content": "embodiment",
            "_protected_context": True,
            "_context_source": "project_instructions",
        },
        {
            "role": "user", "content": "pick object",
            "_protected_context": True, "_context_source": "episode_task",
        },
    ]


def _messages(
    completed: int, *, large_decisions: set[int] | None = None,
) -> list[dict[str, Any]]:
    messages = _protected()
    large = large_decisions or set()
    for step in range(completed + 1):
        if step == 2:
            messages.append({
                "role": "user",
                "content": (
                    '<official_round_result>\n{"instruction":"reflect",'
                    '"passed":false,"round":1}\n</official_round_result>'
                ),
                "_protected_context": True,
                "_context_source": "official_round_result_1",
                "_source_event_ids": ["agent-event-900000"],
            })
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": json.dumps({
                    "current_actual": {"step": step},
                    "observation_text": f"round observation {step}",
                })},
                _image(f"image-{step}"),
            ],
            "_observation": {
                "step": step,
                "receipt": (
                    {"event_id": f"receipt-{step}", "action_id": f"action-{step - 1}"}
                    if step else None
                ),
            },
            "_session_round": 1 if step < 2 else 2,
            "_source_event_ids": [f"agent-event-{step * 5:06d}"],
        })
        if step == completed:
            continue
        read_id = f"observe-{step}"
        mutation_id = f"mutation-{step}"
        messages.extend([
            {
                "role": "assistant",
                "content": (
                    f"decision and reflection {step}" + " durable" * 20_000
                    if step in large else f"decision and reflection {step}"
                ),
                "tool_calls": [{
                    "id": read_id, "type": "function",
                    "function": {"name": "observe", "arguments": "{}"},
                }],
                "_source_event_ids": [f"agent-event-{step * 5 + 1:06d}"],
            },
            {
                "role": "tool", "tool_call_id": read_id, "name": "observe",
                "content": json.dumps({
                    "status": "actual", "views": ["agentview", "wrist"],
                    "diagnostic": "same read-only payload",
                }),
                "_source_event_ids": [f"agent-event-{step * 5 + 2:06d}"],
            },
            {
                "role": "assistant", "content": f"action intent {step}",
                "tool_calls": [{
                    "id": mutation_id, "type": "function",
                    "function": {
                        "name": "move_pose",
                        "arguments": json.dumps({"step": step}),
                    },
                }],
                "_source_event_ids": [f"agent-event-{step * 5 + 3:06d}"],
            },
            {
                "role": "tool", "tool_call_id": mutation_id,
                "name": "move_pose",
                "content": json.dumps({"status": "submitted", "step": step}),
                "_source_event_ids": [f"agent-event-{step * 5 + 4:06d}"],
            },
        ])
    return messages


def _lossless_messages(completed: int) -> list[dict[str, Any]]:
    messages = _messages(completed)
    messages[5]["content"] = json.dumps({
        "status": "actual", "diagnostic": "L" * 60_000,
    })
    round_two_start = next(
        index for index, message in enumerate(messages)
        if message.get("_session_round") == 2
        and isinstance(message.get("_observation"), Mapping)
    )
    messages.insert(round_two_start, {
        "role": "user",
        "content": (
            '<official_round_result>{"round":0,"passed":false}'
            "</official_round_result>"
        ),
        "_protected_context": True,
        "_context_source": "official_round_result_0",
        "_source_event_ids": ["agent-event-verdict-000000"],
    })
    return messages


def _single_round_messages(
    completed: int, *, large_decisions: set[int] | None = None,
) -> list[dict[str, Any]]:
    messages = [
        message for message in _messages(
            completed, large_decisions=large_decisions,
        )
        if message.get("_context_source") != "official_round_result_1"
    ]
    for message in messages:
        if isinstance(message.get("_observation"), Mapping):
            message["_session_round"] = 1
    return messages


def _two_round_messages(
    *, first_round_action_steps: int, second_round_completed: int,
    large_decisions: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Build an append-shaped transcript with one closed and one open Episode."""

    completed = first_round_action_steps + second_round_completed
    messages = _single_round_messages(
        completed, large_decisions=large_decisions,
    )
    round_two_observation_index = next(
        index for index, message in enumerate(messages)
        if isinstance(message.get("_observation"), Mapping)
        and message["_observation"].get("step") == first_round_action_steps
    )
    messages.insert(round_two_observation_index, {
        "role": "user",
        "content": (
            '<official_round_result>\n{"instruction":"reflect",'
            '"passed":false,"round":1}\n</official_round_result>'
        ),
        "_protected_context": True,
        "_context_source": "official_round_result_1",
        "_source_event_ids": ["agent-event-900000"],
    })
    for message in messages:
        observation = message.get("_observation")
        if isinstance(observation, Mapping):
            message["_session_round"] = (
                1
                if observation.get("step", 0) < first_round_action_steps
                else 2
            )
    return messages


def _exchange_for_test(
    name: str, call_id: str, content: str, *, decision: str,
) -> list[dict[str, Any]]:
    return [{
        "role": "assistant", "content": decision,
        "tool_calls": [{
            "id": call_id, "type": "function",
            "function": {"name": name, "arguments": "{}"},
        }],
        "_source_event_ids": [f"agent-event-call-{call_id}"],
    }, {
        "role": "tool", "name": name, "tool_call_id": call_id,
        "content": content,
        "_source_event_ids": [f"agent-event-result-{call_id}"],
    }]


def _tool_content(messages: Sequence[Mapping[str, Any]], call_id: str) -> str:
    return str(next(
        message["content"] for message in messages
        if message.get("role") == "tool"
        and message.get("tool_call_id") == call_id
    ))


@dataclass
class Summary:
    output: Mapping[str, Any]
    calls: list[dict[str, Any]] = field(default_factory=list)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def summarize(
        self, *, messages: Sequence[Mapping[str, Any]],
        source_refs: Sequence[Mapping[str, Any]],
    ) -> SummaryResult:
        self.calls.append({
            "messages": copy.deepcopy(list(messages)),
            "source_refs": copy.deepcopy(list(source_refs)),
        })
        refs = copy.deepcopy(list(source_refs))
        return SummaryResult(
            content=json.dumps(self.output), source_refs=refs,
            source_set_sha256=hashlib.sha256(canonical_bytes(refs)).hexdigest(),
            metadata={
                "call_id": f"summary-{len(self.calls)}", "attempts": 1,
                **copy.deepcopy(dict(self.metadata)),
            },
        )


def _pipeline(
    root: Path, *, management: Mapping[str, Any] = MANAGEMENT,
    context_window: int = 200_000, summary: Summary | None = None,
) -> NativeContextPipeline:
    artifacts = ArtifactStore(root)
    catalog = ContextArtifactCatalog(root, "policy-session-1", artifacts)
    return NativeContextPipeline(
        profile="native_causal_context_v1", artifacts=artifacts,
        context_window_tokens=context_window, max_output_tokens=2048,
        retention="action_step_lossless_v4",
        summary_policy="rollout_checkpoint_v1",
        summary_provider=summary or Summary(CHECKPOINT),
        context_management=management,
        context_artifacts=catalog,
        mutation_tools=("move_pose", "close_gripper", "open_gripper"),
    )


@dataclass
class _Surface:
    result_schemas: Mapping[str, Any] = field(default_factory=dict)
    mutation_tools: tuple[str, ...] = (
        "move_pose", "close_gripper", "open_gripper",
    )

    @staticmethod
    def supports(_capability: str) -> bool:
        return False


@dataclass
class _BaseRuntime:
    surface: _Surface = field(default_factory=_Surface)
    control_root: Path = Path("control")
    allowed_tools: tuple[str, ...] = ("observe",)
    reset_rounds: list[int] = field(default_factory=list)
    canonical_reader: bool = False

    def definitions(self) -> list[dict[str, Any]]:
        extra = [{"name": "vision_segment", "description": "fixture", "inputSchema": {"type": "object", "properties": {}}}] if self.canonical_reader else []
        return extra + [{
            "name": "observe", "description": "observe",
            "inputSchema": {
                "type": "object", "properties": {},
                "additionalProperties": False,
            },
        }]

    def classify_call_effect(self, name: str, arguments: Mapping[str, Any]) -> str:
        return "read_only"

    def invoke(self, name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        return {"status": "actual"}

    def reset_round(self, round_index: int) -> None:
        self.reset_rounds.append(round_index)


def test_artifacts_recover_exact_bytes_and_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "run"
    artifacts = ArtifactStore(root)
    catalog = ContextArtifactCatalog(root, "policy-session-1", artifacts)
    image = catalog.put_bytes(
        b"exact-image", kind="context_image", suffix=".png",
        media_type="image/png", round_index=1, action_step=2,
    )
    document = catalog.put_bytes(
        b'{"outer":{"value":"exact"}}', kind="context_tool_result",
        suffix=".json", media_type="application/json", round_index=1,
        action_step=2,
    )
    long_text = catalog.put_bytes(
        b"0123456789", kind="context_tool_result", suffix=".txt",
        media_type="text/plain", round_index=2, action_step=8,
    )
    legacy_envelope = recoverable_artifact_envelope(
        image, source={
            "origin": "tool_image", "message_role": "user",
            "tool": "vision.segment", "tool_call_id": "segment-0001",
            "image_index": 0,
        },
    )
    pointer = recoverable_artifact_pointer(image)
    marker = encode_recoverable_artifact_marker(pointer)
    legacy_marker = (
        "<recoverable_artifact>"
        + json.dumps(
            legacy_envelope, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        )
        + "</recoverable_artifact>"
    )
    assert decode_recoverable_artifact_marker(marker) == pointer
    assert decode_recoverable_artifact_marker(legacy_marker) == pointer
    assert len(marker.encode("utf-8")) < len(legacy_marker.encode("utf-8")) * 0.55
    with pytest.raises(ValueError, match="pointer fields"):
        validate_recoverable_artifact_pointer({**pointer, "path": "/host/private"})
    with pytest.raises(ValueError, match="pointer is malformed"):
        validate_recoverable_artifact_pointer({
            **pointer, "open": "artifact.read",
        })
    unicode_text = catalog.put_bytes(
        b"a" * 4095 + "中".encode("utf-8"), kind="context_tool_result",
        suffix=".txt", media_type="text/plain", round_index=2, action_step=9,
    )
    events: list[dict[str, Any]] = []
    runtime = RecoverableArtifactToolRuntime(
        base=_BaseRuntime(), run_root=root, event_path=root / "unused.jsonl",
    )
    runtime.bind_context_artifacts(
        catalog, event_sink=lambda kind, **fields: events.append({
            "type": kind, **fields,
        }),
    )
    observed = runtime.invoke_with_context(
        "observe", {"ref": image["ref"]}, call_context={},
    )
    assert base64.b64decode(observed["images_base64"][0]) == b"exact-image"
    selected = runtime.invoke_with_context(
        "artifact.read", {"ref": document["ref"], "json_pointer": "/outer/value"},
        call_context={},
    )
    assert selected["json_value"] == "exact"
    chunk = runtime.invoke_with_context(
        "artifact.read", {"ref": long_text["ref"], "offset": 2, "limit": 4},
        call_context={},
    )
    assert chunk["content"] == "2345" and chunk["next_offset"] == 6
    unicode_head = runtime.invoke_with_context(
        "artifact.read", {"ref": unicode_text["ref"]}, call_context={},
    )
    assert unicode_head["content"] == "a" * 4095
    unicode_tail = runtime.invoke_with_context(
        "artifact.read", {
            "ref": unicode_text["ref"], "offset": unicode_head["next_offset"],
        }, call_context={},
    )
    assert unicode_tail["content"] == "中" and unicode_tail["eof"] is True
    exact_eof = runtime.invoke_with_context(
        "artifact.read", {
            "ref": long_text["ref"], "offset": len(b"0123456789"), "limit": 4,
        }, call_context={},
    )
    assert exact_eof["content"] == ""
    assert exact_eof["next_offset"] == len(b"0123456789")
    assert exact_eof["eof"] is True
    # Argument errors are invalid_arguments naming the argument, store faults
    # keep infrastructure codes (EXP11B census 6, 8): never internal_tool_error.
    with pytest.raises(ToolError, match="past the artifact") as past:
        runtime.invoke_with_context(
            "artifact.read", {
                "ref": long_text["ref"], "offset": len(b"0123456789") + 1,
            }, call_context={},
        )
    assert (past.value.code, past.value.field) == ("invalid_arguments", "offset")
    runtime.reset_round(3)
    observed_after_reset = runtime.invoke_with_context(
        "observe", {"ref": image["ref"]}, call_context={},
    )
    assert base64.b64decode(
        observed_after_reset["images_base64"][0]
    ) == b"exact-image"
    assert runtime.base.reset_rounds == [3]
    with pytest.raises(ToolError, match="another PolicySession") as foreign:
        runtime.invoke_with_context(
            "artifact.read", {"ref": document["ref"].replace(
                "policy-session-1", "policy-session-2"
            )}, call_context={},
        )
    assert (foreign.value.code, foreign.value.field) == ("invalid_arguments", "ref")
    with pytest.raises(ToolError, match="image artifact") as image_read:
        runtime.invoke_with_context(
            "artifact.read", {"ref": image["ref"]}, call_context={},
        )
    assert (image_read.value.code, image_read.value.field) == ("invalid_arguments", "ref")
    with pytest.raises(ToolError, match="image artifact"):
        runtime.invoke_with_context(
            "observe", {"ref": long_text["ref"]}, call_context={},
        )
    with pytest.raises(ToolError, match="takes ref alone") as extra:
        runtime.invoke_with_context(
            "observe", {"ref": image["ref"], "views": ["agentview"]},
            call_context={},
        )
    assert (extra.value.code, extra.value.field) == ("invalid_arguments", "views")
    missing = "artifact://policy-session-1/" + "f" * 64
    with pytest.raises(ToolError, match="no artifact") as unknown:
        runtime.invoke_with_context(
            "artifact.read", {"ref": missing}, call_context={},
        )
    assert (unknown.value.code, unknown.value.field) == ("invalid_arguments", "ref")
    _entry, data = catalog.resolve(document["ref"])
    assert data == b'{"outer":{"value":"exact"}}'
    index = json.loads(catalog.index_path.read_text(encoding="utf-8"))
    stored = root / index["artifacts"][document["ref"]]["storage_ref"]["path"]
    stored.write_bytes(b"tampered")
    with pytest.raises(ToolError, match="hash") as drifted:
        runtime.invoke_with_context(
            "artifact.read", {"ref": document["ref"]}, call_context={},
        )
    assert drifted.value.code == "artifact_hash_mismatch"
    assert [event["payload"]["status"] for event in events] == [
        "ok", "ok", "ok", "ok", "ok", "ok", "rejected", "ok",
        "rejected", "rejected", "rejected", "rejected", "rejected", "rejected",
    ]
    assert all(
        isinstance(event["payload"]["duration_ms"], float)
        and event["payload"]["duration_ms"] >= 0
        for event in events
    )


def test_policy_media_share_one_first_seen_ref_projection_and_observe_decoder(
    tmp_path: Path,
) -> None:
    base, _observation_id, observation_png, _pointer = _media_runtime(
        tmp_path / "world"
    )
    run_root = tmp_path / "run"
    run_root.mkdir()
    tools = RecoverableArtifactToolRuntime(
        base=base, run_root=run_root, event_path=run_root / "unused.jsonl",
    )
    planner = NativePlanner(
        run_root=run_root, run_id="lossless-media-roundtrip", tools=tools,
        provider=FakeProvider([]), project_instructions_path=PROJECT_POLICY,
        episode_prefix_path=EPISODE_TASK, harness_system_path=HARNESS_SYSTEM,
        task_id="task-0", model="k3", reasoning_effort="medium",
        context_profile="native_causal_context_v1",
        context_window_tokens=200_000, max_output_tokens=2048,
        supports_images=True, initial_observation="automatic",
        context_composition="native_slots_v3",
        context_retention="action_step_lossless_v4",
        summary_policy="rollout_checkpoint_v1",
        context_management=MANAGEMENT,
        policy_session_id="policy-session-media",
    )
    planner.start()

    def execute_and_record(
        name: str, arguments: Mapping[str, Any], call_id: str,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        proposal = planner._append(
            "tool_call_proposed", observation_step=0,
            model_call_id="fixture-model", tool_call_id=call_id,
            payload={"name": name, "arguments": dict(arguments), "effect": "read_only"},
        )
        result = dict(tools.invoke_with_context(
            name, arguments, call_context={
                "policy_session_id": planner.policy_session_id,
                "model_call_id": "fixture-model", "tool_call_id": call_id,
                "observation_step": 0,
            },
        ))
        before = len(planner.messages)
        image_message = planner._record_tool_result(
            call_id=call_id, model_call_id="fixture-model", step=0,
            proposal_event=proposal, name=name, result=result,
        )
        assert len(planner.messages) == before + 1
        assert image_message is not None
        return result, copy.deepcopy(planner.messages[-1]), image_message

    segment, segment_tool, segment_image = execute_and_record(
        "vision_segment", {
            "camera": "agentview", "points_uv": [[32, 32]],
        }, "segment-0",
    )
    marked, mark_tool, mark_image = execute_and_record(
        "mark_point", {
            "view": "agentview", "point_ids": ["red-center"],
            "points_uv": [[32, 32]],
        }, "mark-0",
    )
    expected_media = {
        "vision_segment": base64.b64decode(segment["images_base64"][0]),
        "mark_point": base64.b64decode(marked["images_base64"][0]),
    }
    recorded = {
        "vision_segment": (segment_tool, segment_image, "segment-0"),
        "mark_point": (mark_tool, mark_image, "mark-0"),
    }
    messages = _protected()
    for step in range(5):
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": f"actual observation {step}"},
                _png_block(observation_png),
            ],
            "_observation": {"step": step, "receipt": None},
                "_session_round": 1 if step < 3 else 2,
            "_source_event_ids": [f"agent-event-{100 + step:06d}"],
        })
        if step == 0:
            for name, (tool_message, image_message, call_id) in recorded.items():
                messages.extend([
                    {
                        "role": "assistant", "content": f"inspect with {name}",
                        "tool_calls": [{
                            "id": call_id, "type": "function", "function": {
                                "name": name, "arguments": "{}",
                            },
                        }],
                        "_source_event_ids": [f"agent-event-call-{call_id}"],
                    },
                    copy.deepcopy(tool_message), copy.deepcopy(image_message),
                ])
    # Repeat segmentation in the exact open tail. Its inline bytes and the old
    # externalized occurrence must advertise the same content identity.
    messages.extend([
        {
            "role": "assistant", "content": "recheck segmentation",
            "tool_calls": [{
                "id": "segment-0", "type": "function", "function": {
                    "name": "vision_segment", "arguments": "{}",
                },
            }],
            "_source_event_ids": ["agent-event-call-segment-current"],
        },
            {
                **copy.deepcopy(segment_tool), "tool_call_id": "segment-0",
                "_source_event_ids": ["agent-event-result-segment-current"],
            },
            {
                **copy.deepcopy(segment_image),
                "_tool_image_call_id": "segment-0",
                "_source_event_ids": ["agent-event-image-segment-current"],
            },
    ])
    first_later_observation = next(
        index for index, message in enumerate(messages)
        if isinstance(message.get("_observation"), Mapping)
        and message["_observation"].get("step") == 1
    )
    first_delivery = planner.pipeline.project(
        messages=messages[:first_later_observation],
        tools=planner.tool_schemas, observation_step=0,
    )
    def marker_inventory(
        projection: Mapping[str, Any],
    ) -> dict[tuple[Any, ...], str]:
        inventory: dict[tuple[Any, ...], str] = {}
        for message in projection["messages"]:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not (
                    isinstance(block, Mapping) and block.get("type") == "text"
                    and str(block.get("text", "")).startswith(
                        "<recoverable_artifact>"
                    )
                ):
                    continue
                marker = str(block["text"])
                pointer = decode_recoverable_artifact_marker(marker)
                key = (
                    pointer["ref"], pointer["at"][0], pointer["at"][1],
                )
                inventory[key] = marker
        return inventory

    first_delivery_markers = marker_inventory(first_delivery)
    assert first_delivery_markers
    projected = planner.pipeline.project(
        messages=messages, tools=planner.tool_schemas, observation_step=4,
    )
    assert projected["manifest"][
        "lossless_externalized_through_action_step"
    ] == 2
    assert projected["manifest"][
        "lifecycle_media_externalized_through_action_step"
    ] == 2
    assert any(
        item["kind"] == "lifecycle_media_externalization"
        for item in projected["manifest"]["projection_transitions"]
    )

    pointers: list[dict[str, Any]] = []
    inline_refs: set[str] = set()
    for message in projected["messages"]:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for index, block in enumerate(content):
            if not (
                isinstance(block, Mapping) and block.get("type") == "text"
                and str(block.get("text", "")).startswith("<recoverable_artifact>")
            ):
                continue
            pointer = decode_recoverable_artifact_marker(str(block["text"]))
            pointers.append(pointer)
            if index + 1 < len(content) and isinstance(content[index + 1], Mapping):
                if content[index + 1].get("type") == "image_url":
                    inline_refs.add(str(pointer["ref"]))
    by_tool = {
        name: next(
            pointer for pointer in pointers
            if pointer["ref"].endswith(
                "/" + hashlib.sha256(expected).hexdigest()
            )
        )
        for name, expected in expected_media.items()
    }
    assert set(expected_media) <= set(by_tool)
    segment_refs = {
        pointer["ref"] for pointer in pointers
        if pointer["ref"].endswith(
            "/" + hashlib.sha256(expected_media["vision_segment"]).hexdigest()
        )
    }
    assert len(segment_refs) == 1
    assert next(iter(segment_refs)) in inline_refs
    projected_markers = marker_inventory(projected)
    assert all(
        projected_markers[ref] == marker
        for ref, marker in first_delivery_markers.items()
    )
    observation_pointer = next(
        pointer for pointer in pointers
        if pointer["ref"].endswith(
            "/" + hashlib.sha256(observation_png).hexdigest()
        )
        and pointer["at"][1] == 0
    )
    recovered_observation = tools.invoke_with_context(
        observation_pointer["open"], {"ref": observation_pointer["ref"]},
        call_context={},
    )
    assert base64.b64decode(
        recovered_observation["images_base64"][0]
    ) == observation_png
    for name, expected in expected_media.items():
        pointer = by_tool[name]
        assert pointer["open"] == "observe"
        reopened = tools.invoke_with_context(
            pointer["open"], {"ref": pointer["ref"]}, call_context={},
        )
        assert base64.b64decode(reopened["images_base64"][0]) == expected

    tools.reset_round(2)
    segment_pointer = by_tool["vision_segment"]
    reopened = tools.invoke_with_context(
        segment_pointer["open"], {"ref": segment_pointer["ref"]},
        call_context={},
    )
    proposal = planner._append(
        "tool_call_proposed", observation_step=4,
        model_call_id="fixture-reopen", tool_call_id="observe-history",
        payload={"name": "observe", "effect": "read_only"},
    )
    before = len(planner.messages)
    reopened_image = planner._record_tool_result(
        call_id="observe-history", model_call_id="fixture-reopen", step=4,
        proposal_event=proposal, name="observe", result=reopened,
    )
    reopened_tool = copy.deepcopy(planner.messages[-1])
    del planner.messages[before:]
    assert reopened_image is not None
    reopened_visible = json.loads(reopened_tool["content"])
    assert reopened_visible["ref"] == segment_pointer["ref"]
    assert reopened_visible["sha256"] == segment_pointer["ref"].rsplit("/", 1)[1]
    assert reopened_visible["status"] == "actual_historical_artifact"
    messages.extend([
        {
            "role": "assistant", "content": "reopen exact segmentation overlay",
            "tool_calls": [{
                "id": "observe-history", "type": "function", "function": {
                    "name": "observe", "arguments": json.dumps(
                        {"ref": segment_pointer["ref"]}, sort_keys=True,
                    ),
                },
            }],
            "_source_event_ids": ["agent-event-call-observe-history"],
        },
        reopened_tool, reopened_image,
    ])
    reinjected = planner.pipeline.project(
        messages=messages, tools=planner.tool_schemas, observation_step=4,
    )
    historical_images = [
        base64.b64decode(block["image_url"]["url"].split(",", 1)[1])
        for message in reinjected["messages"]
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if isinstance(block, Mapping) and block.get("type") == "image_url"
    ]
    assert expected_media["vision_segment"] in historical_images

    private_mask = next(base.artifact_root.rglob("mask.npy"))
    private_ref = (
        f"artifact://{planner.policy_session_id}/"
        + hashlib.sha256(private_mask.read_bytes()).hexdigest()
    )
    with pytest.raises(ToolError, match="no artifact") as unregistered:
        tools.invoke_with_context("observe", {"ref": private_ref}, call_context={})
    assert (unregistered.value.code, unregistered.value.field) == ("invalid_arguments", "ref")
    trace = (run_root / "agent-events.jsonl").read_text(encoding="utf-8")
    assert '"type":"context_artifact_retrieval"' in trace.replace(" ", "")


def test_large_probe_preview_keeps_all_query_summaries_and_exact_arrays(tmp_path: Path) -> None:
    from ora.agent.tools.motion_probe_feedback import summarize_motion_probe

    pipeline = _pipeline(tmp_path / "run")
    # This reopens a historical v7 result. Keep its original wire shape instead
    # of asking the current planner to generate a legacy receipt. The v7
    # surface that validated this shape left the open-source tree.
    planner = {
        "called": True, "status": "Fail", "stage": "IK_FAIL",
        "valid_query": None, "attempts": None, "trajopt_attempts": None,
        "used_graph": None, "goalset_index": None, "path_steps": 0,
        "interpolation_dt_s": None, "error": None,
        "errors": {"position": [float(seed) for seed in range(512)],
                   "rotation": None, "cspace": None},
        "timings_s": {key: None for key in ("solve", "ik", "graph", "trajopt", "finetune", "total")},
        "constraint_rejection_counts": {
            "non_finite_solution": 0, "solver_unsuccessful": 254, "joint_limit": 2,
        },
    }
    result = {
        "schema": "ora.motion-probe-result.v1", "observation_id": "obs-0",
        "world_mutation_sequence": 0, "variant_policy": "parallel_jaw_equivalents",
        "robot_decisions_consumed": 0, "world_state_changed": False,
        "results": [{
        "query_id": f"q{index}", "arm": "left", "reachable": False, "plan_ref": None,
        "selected_variant_id": None, "selected_target_pose_xyzw": None,
        "variants": [{"variant_id": f"v{variant}", "reachable": False,
            "failure": {"code": "IK_FAIL", "message": "No admitted IK solution."},
            "planner": copy.deepcopy(planner)} for variant in range(6)],
    } for index in range(4)]}
    result["summary"] = summarize_motion_probe(result)
    from ora.agent.context.projection import causal_provider_value
    result = causal_provider_value(result)
    content = json.dumps(result)
    assert len(content.encode()) > pipeline.spill_threshold_bytes
    messages = _messages(0)
    messages.extend(_exchange_for_test(
        "motion.probe", "probe-current", content,
        decision="Compare the admitted variants for all four target poses.",
    ))
    messages.extend(_messages(4)[6:])
    projected = pipeline.project(messages=messages, tools=TOOLS, observation_step=4)
    marker = json.loads(_tool_content(projected["messages"], "probe-current"))
    assert marker["externalized"] is True
    l1 = next(stage for stage in projected["manifest"]["passes"]
              if stage["name"] == "l1_large_result_reference")
    assert len(l1["changed_messages"]) == 1
    preview = json.loads(marker["preview"])
    assert [query["query_id"] for query in preview] == ["q0", "q1", "q2", "q3"]
    assert all(query["reachable"] is False and query["admitted_count"] == 0 for query in preview)
    assert preview[0]["variant_failure_counts"] == {"IK_FAIL": 6}
    assert preview[0]["constraint_rejection_counts"] == {
        "non_finite_solution": 0, "solver_unsuccessful": 1524, "joint_limit": 12,
    }
    assert "errors" not in marker["preview"]
    _entry, raw = pipeline.context_artifacts.resolve(marker["recoverable_artifact"]["ref"])
    assert json.loads(raw) == result
    reader = RecoverableArtifactToolRuntime(
        base=_BaseRuntime(), run_root=tmp_path / "run", event_path=tmp_path / "reads.jsonl",
    )
    reads = []
    reader.bind_context_artifacts(
        pipeline.context_artifacts,
        event_sink=lambda kind, **fields: reads.append({"type": kind, **fields}),
    )
    # The whole 86 kB root is larger than one read (EXP11B census A5): it is
    # refused with its size and keys, and each key reopens exactly.
    with pytest.raises(ToolError) as whole:
        reader.invoke_with_context(
            "artifact.read", {"ref": marker["recoverable_artifact"]["ref"], "json_pointer": ""},
            call_context={},
        )
    assert whole.value.field == "json_pointer"
    assert whole.value.constraints["keys"] == sorted(result)
    assert reads[-1]["payload"]["status"] == "rejected"
    for key in sorted(result):
        if len(json.dumps(result[key])) > 8192:
            continue
        reopened = reader.invoke_with_context(
            "artifact.read", {"ref": marker["recoverable_artifact"]["ref"], "json_pointer": f"/{key}"},
            call_context={},
        )
        assert reopened["json_value"] == result[key]
        assert reads[-1]["payload"]["status"] == "ok"


def test_lossless_projection_preserves_invariants_tail_and_is_idempotent(
    tmp_path: Path,
) -> None:
    pipeline = _pipeline(tmp_path / "run")
    messages = _lossless_messages(4)
    first = pipeline.project(messages=messages, tools=TOOLS, observation_step=4)
    assembler = NativeRequestAssembler(
        model="k3", reasoning_effort="medium", max_output_tokens=2048,
        stream=True, provider=AnthropicMessagesProvider(
            base_url="https://provider.invalid", api_key="test", client=object(),
        ),
    )
    assembler.prepare(first)
    manifest = first["manifest"]
    assert manifest["request_relation"] == "initial"
    assert manifest["canonical_request_utf8_bytes"] > 0
    assert manifest["provider_wire_request_utf8_bytes"] > 0
    assert len(manifest["provider_wire_request_sha256"]) == 64
    assert manifest["lossless_externalized_through_action_step"] == 1
    assert manifest[
        "lifecycle_media_externalized_through_action_step"
    ] == 1
    assert manifest["pressure_externalized_through_action_step"] == 0
    assert manifest["episode_checkpointed_through_action_step"] is None
    assert manifest["episode_checkpoint_count"] == 0
    assert manifest["emergency_compaction_count"] == 0
    assert manifest["raw_estimated_input_tokens"] > 0
    assert _provider_estimated_input_tokens({
        "pressure_estimated_input_tokens": 8513,
    }) == 8513
    assert _provider_estimated_input_tokens({
        "raw_estimated_input_tokens": 40395,
    }) == 40395
    with pytest.raises(PolicySessionError, match="input-token estimate"):
        _provider_estimated_input_tokens({})
    assert manifest["active_inline_images"] == 3
    assert manifest["retained_exact_causal_action_steps"] == 3
    assert manifest["lossless_invariants_before_sha256"] == manifest[
        "lossless_invariants_after_sha256"
    ]
    assert any("artifact://policy-session-1/" in str(message.get("content"))
               for message in first["messages"])
    l1 = next(item for item in manifest["passes"] if item["name"] == (
        "l1_large_result_reference"
    ))
    assert len(l1["changed_messages"]) == 1
    assert any(
        "official_round_result" in str(message.get("content"))
        for message in first["messages"]
    )
    index_before = json.loads(
        pipeline.context_artifacts.index_path.read_text(encoding="utf-8")
    )
    second = pipeline.project(messages=messages, tools=TOOLS, observation_step=4)
    assembler.prepare(second)
    index_after = json.loads(
        pipeline.context_artifacts.index_path.read_text(encoding="utf-8")
    )
    assert second["manifest"]["projection_transitions"] == []
    assert second["manifest"]["request_relation"] == "append_only"
    assert second["manifest"]["request_sha256"] == manifest["request_sha256"]
    assert index_after == index_before
    old_ref = next(iter(index_after["artifacts"]))
    _entry, recovered = pipeline.context_artifacts.resolve(old_ref)
    assert recovered
    extended = pipeline.project(
        messages=_lossless_messages(5), tools=TOOLS, observation_step=5,
    )
    assembler.prepare(extended)
    assert extended["manifest"][
        "lossless_externalized_through_action_step"
    ] == 2
    assert any(
        item["kind"] == "lifecycle_media_externalization"
        for item in extended["manifest"]["projection_transitions"]
    )
    assert extended["manifest"]["request_relation"] == "projection_epoch"


def _identified(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for message in messages:
        observation = message.get("_observation")
        if isinstance(observation, dict):
            observation["observation_id"] = f"observation-{observation['step']:04d}"
    return messages


def _no_action_turn(
    messages: list[dict[str, Any]], *, turn: int,
) -> list[dict[str, Any]]:
    """Spend one turn on refused read-only calls, then present it again.

    This is the Supervisor's ``continue_same_observation`` recovery after
    ``model_call_limit``: the Session receives the same observation step and
    identity a second time, with no action and no receipt in between.
    """

    current = next(
        message for message in reversed(messages)
        if isinstance(message.get("_observation"), Mapping)
    )
    refused = json.dumps({"ok": False, "error": {"code": "invalid_arguments"}})
    for index in range(3):
        messages.extend(_exchange_for_test(
            "observe", f"refused-{turn}-{index}", refused,
            decision=f"query again {turn}-{index}",
        ))
    again = copy.deepcopy(current)
    again["_observation"]["receipt"] = None
    again["_source_event_ids"] = [f"agent-event-{900100 + turn:06d}"]
    messages.append(again)
    return messages


def test_lossless_projection_accepts_an_observation_presented_again_after_no_action(
    tmp_path: Path,
) -> None:
    """EXP11A g016 classify_objects died with a fatal orchestrator fault.

    120 read-only calls at observation step 50 reached
    ``max_model_calls_per_observation``; the Supervisor presented step 50 again
    and the next projection raised "ActionStep 50 post-action observation is not
    the next step", because the inventory required every closed ActionStep to be
    followed by the next step even when it had proposed no action.
    """

    pipeline = _pipeline(tmp_path / "run")
    messages = _identified(_single_round_messages(3))
    pipeline.project(messages=messages, tools=TOOLS, observation_step=3)
    for turn in range(2):
        messages = _no_action_turn(messages, turn=turn)
        projected = pipeline.project(
            messages=messages, tools=TOOLS, observation_step=3,
        )
    inventory = projected["manifest"]["action_step_inventory"]
    assert [item["observation_step"] for item in inventory] == [0, 1, 2, 3, 3, 3]
    assert [item["post_observation_step"] for item in inventory] == [
        1, 2, 3, 3, 3, None,
    ]
    assert [item["mutation"] for item in inventory[3:]] == [None, None, None]
    assert [item["complete"] for item in inventory] == [True] * 5 + [False]


@pytest.mark.parametrize("change", ["mutation", "other_identity", "skipped_step"])
def test_only_a_no_action_step_may_be_followed_by_its_own_observation(
    change: str,
) -> None:
    from ora.agent.context.projection import _action_step_inventory

    messages = _no_action_turn(
        _identified(_single_round_messages(3)), turn=0,
    )
    again = messages[-1]["_observation"]
    if change == "mutation":
        # An accepted mutation must be followed by its post-action observation.
        messages[-2]["content"] = json.dumps({"status": "submitted"})
        messages[-3]["tool_calls"][0]["function"]["name"] = "move_pose"
        messages[-2]["name"] = "move_pose"
    elif change == "other_identity":
        again["observation_id"] = "observation-9999"
    else:
        again["step"] = 5
    with pytest.raises(
        PolicySessionError,
        match="ActionStep 3 post-action observation is not the next step",
    ):
        _action_step_inventory(
            messages, mutation_tools=("move_pose",), require_session_round=True,
        )


def _robotwin_messages(completed: int) -> list[dict[str, Any]]:
    """RoboTwin-shaped transcript: two images per actual, sparse receipts.

    Every observation carries the observer camera plus the derived auxiliary
    contact sheet, dual-arm sparse public state and (after an action) one
    sparse-motion receipt v7 whose action_id joins the previous mutation.
    """

    messages = _protected()
    for step in range(completed + 1):
        receipt = None
        if step:
            receipt = {
                "event_id": f"receipt-{step}",
                "action_id": f"action-{step - 1}",
            }
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": json.dumps({
                    "current_actual": {"step": step},
                    "end_effectors": {
                        "left": {"pose_xyz_xyzw": [0.1] * 7,
                                 "gripper_command_target": "open"},
                        "right": {"pose_xyz_xyzw": [0.2] * 7,
                                  "gripper_command_target": "close"},
                    },
                })},
                _image(f"observer-{step}"),
                _image(f"contact-sheet-{step}"),
            ],
            "_observation": {"step": step, "receipt": receipt},
            "_session_round": 1,
            "_source_event_ids": [f"agent-event-{step * 5:06d}"],
        })
        if step == completed:
            continue
        probe_id = f"probe-{step}"
        mutation_id = f"mutation-{step}"
        probe_payload = json.dumps({
            "results": [{
                "query_id": "q0", "arm": "left",
                "plan_ref": "plan-" + "a" * 32,
                "variants": [{"telemetry": "V" * (60_000 if step == 0 else 40)}],
            }],
        })
        messages.extend([
            {
                "role": "assistant", "content": f"probe reachability {step}",
                "tool_calls": [{
                    "id": probe_id, "type": "function",
                    "function": {"name": "motion.probe", "arguments": "{}"},
                }],
                "_source_event_ids": [f"agent-event-{step * 5 + 1:06d}"],
            },
            {
                "role": "tool", "tool_call_id": probe_id,
                "name": "motion.probe", "content": probe_payload,
                "_source_event_ids": [f"agent-event-{step * 5 + 2:06d}"],
            },
            {
                "role": "assistant", "content": f"move left arm {step}",
                "tool_calls": [{
                    "id": mutation_id, "type": "function",
                    "function": {
                        "name": "move_pose",
                        "arguments": json.dumps({"targets": {"left": {
                            "interaction": "free_space",
                            "position": {"mode": "hold_current"},
                            "orientation": {"mode": "hold_current"},
                        }}}),
                    },
                }],
                "_source_event_ids": [f"agent-event-{step * 5 + 3:06d}"],
            },
            {
                "role": "tool", "tool_call_id": mutation_id,
                "name": "move_pose",
                "content": json.dumps({
                    "status": "submitted", "action": "move_pose",
                }),
                "_source_event_ids": [f"agent-event-{step * 5 + 4:06d}"],
            },
        ])
    return messages


ROBOTWIN_MUTATION_TOOLS = (
    "move_to", "move_pose", "rotate_wrist", "rotate_pitch",
    "set_gripper", "recover_pose",
)


def test_lossless_projection_handles_robotwin_dual_arm_action_steps(
    tmp_path: Path,
) -> None:
    """EXP08A D2 prerequisite: lossless v4 over RoboTwin-shaped evidence.

    Dual-arm sparse ActionSteps keep the one-observation/one-mutation/one-
    receipt structure, two inline images per actual (observer + contact
    sheet) obey the media tail, the large read-only motion.probe result is
    externalized and byte-recoverable, and the mutation result plus every
    receipt-bearing causal field stays inline.
    """

    root = tmp_path / "run"
    artifacts = ArtifactStore(root)
    catalog = ContextArtifactCatalog(root, "policy-session-1", artifacts)
    pipeline = NativeContextPipeline(
        profile="native_causal_context_v1", artifacts=artifacts,
        context_window_tokens=200_000, max_output_tokens=2048,
        retention="action_step_lossless_v4",
        summary_policy="rollout_checkpoint_v1",
        summary_provider=Summary(CHECKPOINT),
        context_management=MANAGEMENT,
        context_artifacts=catalog,
        mutation_tools=ROBOTWIN_MUTATION_TOOLS,
    )
    messages = _robotwin_messages(4)
    projected = pipeline.project(
        messages=messages, tools=TOOLS, observation_step=4,
    )
    manifest = projected["manifest"]

    # The oversized read-only probe result is externalized and recoverable.
    l1 = next(
        item for item in manifest["passes"]
        if item["name"] == "l1_large_result_reference"
    )
    assert len(l1["changed_messages"]) == 1
    probe_visible = json.loads(_tool_content(projected["messages"], "probe-0"))
    pointer = probe_visible["recoverable_artifact"]
    _entry, recovered = pipeline.context_artifacts.resolve(pointer["ref"])
    assert json.loads(recovered.decode("utf-8"))["results"][0][
        "plan_ref"
    ] == "plan-" + "a" * 32

    # Mutation results are causal evidence and are never externalized.
    for step in range(4):
        mutation_visible = _tool_content(
            projected["messages"], f"mutation-{step}",
        )
        assert "recoverable_artifact" not in mutation_visible
        assert json.loads(mutation_visible)["status"] == "submitted"

    # Every step's causal observation text stays inline; only media move.
    inline_observation_texts = [
        block["text"]
        for message in projected["messages"]
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if isinstance(block, dict) and block.get("type") == "text"
        and "current_actual" in str(block.get("text", ""))
    ]
    assert len(inline_observation_texts) == 5
    assert all(
        "end_effectors" in text for text in inline_observation_texts
    )
    # The canonical transcript keeps its receipt joins untouched.
    receipts = [
        message["_observation"]["receipt"] for message in messages
        if isinstance(message.get("_observation"), Mapping)
        and message["_observation"].get("receipt")
    ]
    assert [entry["action_id"] for entry in receipts] == [
        f"action-{step}" for step in range(4)
    ]

    # Two inline images per actual: historical media beyond the configured
    # tail are externalized while the exact tail keeps both views inline.
    inline_tail = int(MANAGEMENT["inline_media_tail_action_steps"])
    expected_inline = 2 * (inline_tail + 1)
    assert manifest["active_inline_images"] == expected_inline
    l3 = next(
        item for item in manifest["passes"]
        if item["name"] == "l3_lifecycle_media_reference"
    )
    assert l3["changed_messages"]

    # A projected historical image is byte-recoverable through the catalog.
    historical = next(
        block for message in projected["messages"]
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if isinstance(block, dict) and block.get("type") == "text"
        and "recoverable_artifact" in str(block.get("text", ""))
    )
    marker = json.loads(
        str(historical["text"]).split("<recoverable_artifact>")[1].split(
            "</recoverable_artifact>"
        )[0]
    )
    _entry, image_bytes = pipeline.context_artifacts.resolve(marker["ref"])
    assert image_bytes.startswith(b"observer-") or image_bytes.startswith(
        b"contact-sheet-"
    )

    # Idempotence: re-projecting the same transcript is append-only.
    second = pipeline.project(
        messages=messages, tools=TOOLS, observation_step=4,
    )
    assert second["manifest"]["projection_transitions"] == []
    assert second["manifest"]["request_sha256"] == manifest["request_sha256"]

    # Isolation holds under RoboTwin-shaped ids too: the context catalog
    # resolves only artifact:// references from its own session scope.
    with pytest.raises(Exception):
        pipeline.context_artifacts.resolve(
            "artifact://another-policy-session/" + "a" * 64
        )


@pytest.mark.parametrize("reader_name", ["artifact.read", "artifact_read"])
def test_current_artifact_read_stays_visible_through_large_result_pass(tmp_path: Path, reader_name: str) -> None:
    from ora.agent.context.projection import _materialize_recoverable_payloads

    pipeline = _pipeline(tmp_path / "run")
    messages = _messages(0)
    original = pipeline.context_artifacts.put_bytes(
        json.dumps({"needed": "R" * 60_000}).encode(),
        kind="context_tool_result", suffix=".json", media_type="application/json",
        round_index=0, action_step=0,
    )
    # The reader now refuses a value over ARTIFACT_READ_LIMIT_BYTES (EXP11B
    # census A5), so no read it serves reaches this pipeline's spill threshold.
    # The projection rule under test holds for any artifact_read result, so
    # the result a reader would have returned is built here directly.
    arguments = {"ref": original["ref"], "json_pointer": ""}
    entry, data = pipeline.context_artifacts.resolve(original["ref"])
    result = {
        "status": "ok", "ref": entry["ref"], "sha256": entry["sha256"],
        "media_type": entry["media_type"], "bytes": entry["bytes"], "offset": 0,
        "next_offset": len(data), "eof": True, "content": None, "json_value": json.loads(data),
    }
    payload = json.dumps(result)
    assert len(payload.encode()) > pipeline.spill_threshold_bytes
    for call_id in ("read-current", "read-current-again"):
        exchange = _exchange_for_test(
            reader_name, call_id, payload,
            decision="Read the exact evidence returned by the reference.",
        )
        exchange[0]["tool_calls"][0]["function"]["arguments"] = json.dumps(arguments)
        messages.extend(exchange)

    # Registration still preserves a private backing identity for historical
    # pressure collection, without advertising a new wrapper to the policy.
    materialized, _refs, advertised = _materialize_recoverable_payloads(
        messages, catalog=pipeline.context_artifacts, mutation_tools=pipeline.mutation_tools,
        advertise_tool_result_over_bytes=pipeline.spill_threshold_bytes,
    )
    read_index = next(i for i, message in enumerate(materialized)
                      if message.get("tool_call_id") == "read-current")
    private_ref = materialized[read_index]["_context_artifact_ref"]["ref"]
    assert private_ref != original["ref"]
    assert read_index not in advertised
    _entry, backing = pipeline.context_artifacts.resolve(private_ref)
    assert backing.decode() == payload
    rematerialized, _refs, advertised_again = _materialize_recoverable_payloads(
        materialized, catalog=pipeline.context_artifacts, mutation_tools=pipeline.mutation_tools,
        advertise_tool_result_over_bytes=pipeline.spill_threshold_bytes,
    )
    assert rematerialized[read_index] == materialized[read_index]
    assert read_index not in advertised_again

    projected = pipeline.project(messages=messages, tools=TOOLS, observation_step=0)
    for call_id in ("read-current", "read-current-again"):
        visible = json.loads(_tool_content(projected["messages"], call_id))
        assert visible == result
        assert visible["ref"] == original["ref"]
        assert visible["json_value"] == {"needed": "R" * 60_000}
        assert "recoverable_artifact" not in visible and "externalized" not in visible
    for stage in projected["manifest"]["passes"]:
        if stage["name"] in {"l1_large_result_reference", "l2_exact_duplicate_reference"}:
            assert stage["changed_messages"] == []
    second = pipeline.project(messages=messages, tools=TOOLS, observation_step=0)
    assert _tool_content(second["messages"], "read-current") == payload
    assert second["manifest"]["projection_transitions"] == []
    assert second["manifest"]["request_sha256"] == projected["manifest"]["request_sha256"]
    from ora.agent.context.lossless_render import render_lossless
    # Force L1/L2 eligibility independently of the near window. L4 still owns
    # historical pressure collection; do not disable that existing contract.
    rendered = render_lossless(materialized, mutation_tools=pipeline.mutation_tools,
        spill_threshold_bytes=pipeline.spill_threshold_bytes, preview_bytes=0,
        active_eager_through=100, pressure_through=0, lifecycle_media_through=0,
        superseded_tool_image_event_ids=(), episode_checkpointed_action_steps=0,
        episode_capsules=[], emergency_summarized_action_steps=0, emergency_checkpoint=None,
        emergency_transcript_ref=None)
    for stage in rendered[2]["passes"]:
        if stage["name"] in {"l1_large_result_reference", "l2_exact_duplicate_reference"}:
            assert stage["changed_messages"] == []


def test_spilled_tool_results_hide_artifact_identity_from_provider() -> None:
    """Provider-visible spill stubs must not leak artifact path or hash."""

    from ora.agent.context.native import _spill_large_tool_results

    class _Store:
        def put_bytes(self, raw, *, kind, suffix, media_type):
            assert kind == "native_tool_result"
            return {
                "artifact_id": "native-tool-result-0001",
                "path": "artifact-store/native-tool-result-0001.txt",
                "sha256": "f" * 64,
            }

    projected, refs = _spill_large_tool_results(
        [{"role": "tool", "content": "x" * 4096}],
        artifacts=_Store(), threshold_bytes=64, preview_bytes=16,
    )
    visible = projected[0]["content"]
    payload = json.loads(visible)
    assert payload["complete_result_recorded"] is True
    assert payload["truncated"] is True
    assert "artifact_ref" not in payload
    for leaked in ("native-tool-result-0001", "artifact-store", "f" * 64):
        assert leaked not in visible
    assert projected[0]["_artifact_ref"]["artifact_id"] == "native-tool-result-0001"
    assert refs and refs[0]["artifact_id"] == "native-tool-result-0001"


def _project_with_input_usage(pipeline, messages, completed, input_tokens):
    prepared = pipeline._project_action_step_lossless_v4(
        messages=copy.deepcopy(messages), tools=TOOLS, observation_step=completed,
        allow_transition=False,
    )
    pipeline.observe_provider_usage(
        estimated_tokens=_provider_estimated_input_tokens(prepared["manifest"]),
        input_tokens=input_tokens, projection_epoch=prepared["manifest"]["projection_epoch"],
        provider_wire_request_sha256="c" * 64,
    )
    return pipeline.project(messages=messages, tools=TOOLS, observation_step=completed)
