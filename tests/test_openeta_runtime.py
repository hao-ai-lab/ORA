"""Shared OpenETA Runtime contracts."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from ora.orchestrator.action_executor import execute_runtime_action
from ora.orchestrator.evaluation import seal
from ora.orchestrator.spec import resolve_run_spec
from ora.environment import (
    SyntheticSession, validate_runtime_action,
)
from ora.environment.backends.libero_pro.session import (
    libero_environment_init_params,
    libero_public_environment_context,
)
from ora.agent.context import ContextProjector
from ora.agent.tools.runtime.core import (
    ManipulationToolRuntime, _opencv_camera_point_to_world,
    _robust_depth_measurement,
)
from ora.agent.native.gateway import openai_tool_definitions
from ora.agent.providers.profiles import probe_native_provider
from ora.agent.composition.tool_surface import (
    ToolSurfaceError, compose_tool_definitions, resolve_tool_surface,
)
from ora.agent.tools.robot import ToolError
from native_synthetic_spec import synthetic_spec


LIBERO_PUBLIC_ENVIRONMENT_CONTEXT = libero_public_environment_context(
    {"problem": "Libero", "problem_folder": "libero_object_swap"},
    init_params=libero_environment_init_params(max_sim_steps=1),
)
ROOT = Path(__file__).resolve().parents[1]
MANUAL = ROOT / "ora/policy/embodiments/aloha-agilex/manual-v1.md"
PROMPT = ROOT / "tests/data/prompts/startup-prompt-v1.md"
AGENTS = ROOT / "ora/policy/planners/native/project-v1.md"
BOUNDS = {"x": [-0.6, 0.6], "y": [-0.4, 0.4], "z": [0.7, 1.45]}


def _assistant_tool(
    call_id: str, name: str, arguments: dict, *, content: str = "",
) -> dict:
    return {
        "role": "assistant", "content": content, "tool_calls": [{
            "id": call_id, "type": "function", "function": {
                "name": name,
                "arguments": json.dumps(arguments, sort_keys=True),
            },
        }],
    }


def _openeta_runtime(tmp_path: Path) -> tuple[ManipulationToolRuntime, Path]:
    source = tmp_path / "source"
    session = SyntheticSession(
        tmp_path / "native", "openeta_synthetic", 0,
        main_camera="agentview", camera_names=("agentview", "wrist"),
    )
    session.start()
    observation, _ = session.observe(source)
    control = tmp_path / "control"
    workspace = tmp_path / "workspace"
    projector = ContextProjector(
        workspace, control, PROMPT, MANUAL, AGENTS,
        expose_manual=False, main_camera="agentview",
    )
    projector.publish(
        observation=observation,
        observation_root=source / observation["observation_id"],
        task_instruction="test",
        remaining_budget={"robot_decisions": 2, "model_calls": 8},
        policy_state={"step": 0, "path": "unused", "sha256": "unused", "state_sha256": "unused"},
    )
    runtime = ManipulationToolRuntime(
        control_root=control, manual_path=MANUAL, event_path=tmp_path / "events.jsonl",
        workspace_bounds_m=BOUNDS, state_root=None,
        tool_surface_profile="native_atomic_causal_v1",
    )
    return runtime, source


def test_description_layers_are_ordered_and_fail_closed() -> None:
    base = [{"name": "t", "description": "base", "inputSchema": {"type": "object"}}]
    resolved = compose_tool_definitions(base, [
        {"prepend": {"t": "before"}},
        {"append": {"t": "after"}},
        {"replace": {"t": "final"}},
        {"append": {"t": "tail"}},
    ])
    assert resolved[0]["description"] == "final tail"
    with pytest.raises(ToolSurfaceError, match="unknown tool"):
        compose_tool_definitions(base, [{"append": {"missing": "x"}}])
    with pytest.raises(ToolSurfaceError, match="invalid type"):
        compose_tool_definitions(base, [], schema_replacements=[{"t": {"type": "tuple"}}])


def test_native_prompt_v2_run_spec_rejects_environment_profile_drift() -> None:
    spec = synthetic_spec()
    spec["policy"]["context_retention"] = "rewrite_compact_v1"
    resolve_run_spec(copy.deepcopy(spec), root=ROOT)
    spec["runtime"]["public_environment_context"]["robot_models"] = ["Sawyer"]
    with pytest.raises(
        ValueError,
        match="public environment context must match exactly one prompt profile",
    ):
        resolve_run_spec(spec, root=ROOT)


def test_shared_provider_probe_uses_native_tool_and_image_contract() -> None:
    class ProbeProvider:
        def __init__(self) -> None:
            self.requests: list[dict] = []

        def complete(self, *, request: dict, model_call_id: str) -> dict:
            self.requests.append(request)
            if model_call_id == "probe-text":
                marker = request["messages"][0]["content"].rsplit(": ", 1)[-1]
                message = {"role": "assistant", "content": marker}
                finish = "stop"
            elif model_call_id == "probe-tool":
                message = _assistant_tool(
                    "move-probe", "preview_move", {"delta_mm": [0, 0, 10]},
                )
                finish = "tool_calls"
            elif model_call_id == "probe-continuity":
                assert request["messages"][1]["tool_calls"][0]["id"] == "move-probe"
                assert request["messages"][2]["tool_call_id"] == "move-probe"
                assert json.loads(request["messages"][2]["content"])["ok"] is False
                marker = request["messages"][3]["content"].rsplit(": ", 1)[-1]
                message = {"role": "assistant", "content": marker}
                finish = "stop"
            else:
                message = {"role": "assistant", "content": "red"}
                finish = "stop"
            return {
                "message": message, "finish_reason": finish,
                "usage": {"input_tokens": 10, "output_tokens": 2},
            }

    surface = resolve_tool_surface("native_atomic_causal_v1")
    definitions = openai_tool_definitions(
        [item["name"] for item in surface.definitions],
        definitions=surface.definitions,
    )
    provider = ProbeProvider()
    result = probe_native_provider(
        provider,
        model="k3",
        tool_definitions=definitions,
        tool_name="preview_move",
        tool_arguments={"delta_mm": [0, 0, 10]},
        include_images=True,
        reasoning_effort="medium",
    )
    assert result["passed"] is True
    assert [check["check"] for check in result["checks"]] == [
        "streaming_text", "tool_schema", "tool_result_continuity", "image_content",
    ]
    assert all(request["reasoning_effort"] == "medium" for request in provider.requests)
    assert all(request["stream"] is True for request in provider.requests)
    assert all("stream_options" in request for request in provider.requests)
    assert provider.requests[1]["tools"][0]["function"]["name"] == "preview_move"
    assert "data:image/png;base64," in json.dumps(provider.requests[3])


def test_shared_provider_probe_honors_non_streaming_profile_transport() -> None:
    class ProbeProvider:
        def __init__(self) -> None:
            self.requests: list[dict] = []

        def complete(self, *, request: dict, model_call_id: str) -> dict:
            self.requests.append(request)
            if model_call_id == "probe-text":
                marker = request["messages"][0]["content"].rsplit(": ", 1)[-1]
                message = {"role": "assistant", "content": marker}
                finish = "stop"
            elif model_call_id == "probe-tool":
                message = _assistant_tool(
                    "move-probe", "preview_move", {"delta_mm": [0, 0, 10]},
                )
                finish = "tool_calls"
            elif model_call_id == "probe-continuity":
                marker = request["messages"][3]["content"].rsplit(": ", 1)[-1]
                message = {"role": "assistant", "content": marker}
                finish = "stop"
            else:
                message = {"role": "assistant", "content": "red"}
                finish = "stop"
            return {
                "message": message,
                "finish_reason": finish,
                "usage": {"input_tokens": 10, "output_tokens": 2},
            }

    surface = resolve_tool_surface("native_atomic_causal_v1")
    definitions = openai_tool_definitions(
        [item["name"] for item in surface.definitions],
        definitions=surface.definitions,
    )
    provider = ProbeProvider()
    result = probe_native_provider(
        provider,
        model="inkling-small",
        tool_definitions=definitions,
        tool_name="preview_move",
        tool_arguments={"delta_mm": [0, 0, 10]},
        include_images=True,
        reasoning_effort="medium",
        stream=False,
    )
    assert result["passed"] is True
    assert result["stream"] is False
    assert all(request["stream"] is False for request in provider.requests)
    assert all("stream_options" not in request for request in provider.requests)


def test_target_state_v4_expands_pose_then_gripper() -> None:
    action = {
        "schema": "ora-action.v4", "observation_id": "o", "action_id": "a",
        "action": "target_state", "bound_observation_sha256": "h",
        "arguments": {
            "arm": "right",
            "pose": {"position_m": [0.1, 0.0, 1.0], "orientation_xyzw": [0, 0, 0, 1]},
            "gripper": 0.0,
        },
    }
    assert validate_runtime_action(action)["action"] == "target_state"
    assert validate_runtime_action({
        **action, "arguments": {"arm": "right", "pose": action["arguments"]["pose"]},
    })["arguments"].get("gripper") is None
    assert validate_runtime_action({
        **action, "arguments": {"arm": "right", "gripper": 0.0},
    })["arguments"].get("pose") is None
    with pytest.raises(ValueError, match="pose or gripper"):
        validate_runtime_action({**action, "arguments": {"arm": "right"}})


def test_mark_point_remeasures_before_claiming_independent_verification(
    tmp_path: Path,
) -> None:
    runtime, _source = _openeta_runtime(tmp_path)
    first = runtime.invoke("mark_point", {
        "point_id": "P0", "view": "agentview", "x": 256, "y": 256,
    })
    duplicate = runtime.invoke("mark_point", {
        "point_id": "P0", "view": "agentview", "x": 256, "y": 256,
    })
    assert duplicate["status"] == "solved"
    points_after_duplicate = json.loads(
        (tmp_path / "control/openeta-points.json").read_text()
    )
    assert len(points_after_duplicate["P0"]["sources"]) == 1

    verified = runtime.invoke("mark_point", {
        "point_id": "P0", "view": "wrist", "x": 256, "y": 256,
    })
    assert verified["status"] == "verified"
    points = json.loads((tmp_path / "control/openeta-points.json").read_text())
    assert points["P0"]["last_verification"]["observation_id"]
    assert {source["view"] for source in points["P0"]["sources"]} == {
        "agentview", "wrist",
    }
    assert first["xyz_m"] == pytest.approx(verified["xyz_m"])


def test_mark_point_depth_estimator_rejects_edges_and_far_plane() -> None:
    calibration = {"valid_depth_range_m": {"near": 0.1, "far": 10.0}}
    measured, quality = _robust_depth_measurement(
        np.ones((5, 5), dtype=np.float32), source_x=2, source_y=2,
        calibration=calibration,
    )
    assert measured == pytest.approx(1.0)
    assert quality["supporting_samples"] == 25

    edge = np.full((5, 5), 3.0, dtype=np.float32)
    edge[1:4, 1:4] = 1.0
    with pytest.raises(ToolError) as discontinuity:
        _robust_depth_measurement(
            edge, source_x=2, source_y=2, calibration=calibration,
        )
    assert discontinuity.value.code == "depth_discontinuity"

    with pytest.raises(ToolError) as far_plane:
        _robust_depth_measurement(
            np.full((5, 5), 9.9, dtype=np.float32),
            source_x=2, source_y=2, calibration=calibration,
        )
    assert far_plane.value.code == "invalid_depth"

def test_native_mark_point_reports_measurement_not_action_reachability(
    tmp_path: Path,
) -> None:
    runtime, _source = _openeta_runtime(tmp_path)
    # Pixels are in native_atomic_causal_v1's 512x512 policy view.
    marked = runtime.invoke("mark_point", {
        "point_id": "background", "view": "agentview", "x": 256, "y": 256,
    })
    assert marked["status"] == "solved"
    assert marked["action_workspace"]["inside"] is False
    assert marked["action_workspace"]["semantics"] == (
        "measurement_not_reachability"
    )
    assert marked["action_workspace"]["motion_use"] == (
        "measurement_only_outside_action_envelope"
    )

def test_action_v4_composite_receipt(tmp_path: Path) -> None:
    session = SyntheticSession(tmp_path / "native", "action", 0, max_actions=2)
    session.start()
    action = {
        "schema": "ora-action.v4", "observation_id": "o", "action_id": "a",
        "action": "target_state", "bound_observation_sha256": "h",
        "arguments": {
            "arm": "right",
            "pose": {"position_m": [0.1, 0.0, 1.0], "orientation_xyzw": [0, 0, 0, 1]},
            "gripper": 0.0,
        },
    }
    receipt = execute_runtime_action(session, action)
    assert receipt["schema"] == "ora.composite-motion-receipt.v1"
    assert [phase["phase"] for phase in receipt["phase_receipts"]] == ["pose", "gripper"]
    assert all(phase["status"] in {"reached", "executed"} for phase in receipt["phase_receipts"])

    class MissedRuntime:
        def status(self):
            return {"terminal": False, "reason": None}

        def execute(self, phase):
            return {
                "commanded_pose_xyz_xyzw": [0.1, 0, 1, 0, 0, 0, 1],
                "reached_pose_xyz_xyzw": [0.2, 0, 1, 0, 0, 0, 1],
            }

    missed = execute_runtime_action(MissedRuntime(), action)
    assert [phase["status"] for phase in missed["phase_receipts"]] == [
        "not_reached", "skipped_motion_not_reached",
    ]


def test_mark_point_accepts_standard_opencv_3x4_world_extrinsic() -> None:
    calibration = {
        "intrinsic_cv": [
            [400.0, 0.0, 32.0],
            [0.0, 400.0, 32.0],
            [0.0, 0.0, 1.0],
        ],
        "extrinsic_cv": [
            [1.0, 0.0, 0.0, 0.1],
            [0.0, 1.0, 0.0, -0.2],
            [0.0, 0.0, 1.0, 0.3],
        ],
    }
    assert _opencv_camera_point_to_world(
        [1.0, 2.0, 3.0, 1.0], calibration=calibration,
    ) == pytest.approx([0.9, 2.2, 2.7])

    calibration["extrinsic_cv"] = [
        *calibration["extrinsic_cv"], [0.0, 0.0, 0.0, 1.0],
    ]
    assert _opencv_camera_point_to_world(
        [1.0, 2.0, 3.0, 1.0], calibration=calibration,
    ) == pytest.approx([0.9, 2.2, 2.7])

    calibration["extrinsic_cv"] = [[1.0, 0.0], [0.0, 1.0]]
    with pytest.raises(ToolError) as invalid:
        _opencv_camera_point_to_world(
            [1.0, 2.0, 3.0, 1.0], calibration=calibration,
        )
    assert invalid.value.code == "invalid_camera_calibration"
