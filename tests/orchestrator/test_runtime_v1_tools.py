"""Runtime v1 tools contract regressions."""

from __future__ import annotations
import base64
import json
from pathlib import Path
import pytest
from PIL import Image
from ora.orchestrator.stagnation import StagnationGuard
from ora.orchestrator.action_executor import _motion_reached
from ora.agent.context import ContextProjector
from ora.environment import SyntheticSession
from ora.agent.native.gateway import NativeToolGateway
from ora.agent.tools.robot import TOOL_DEFINITIONS, RobotToolRuntime, ToolError
from ora.agent.evidence.policy_state import PolicyStateStore
from ora.contracts.action import ActionValidationError, validate_runtime_action
from tests.orchestrator.test_runtime_v1 import (
    AGENT_RULES,
    BOUNDS,
    MANUAL,
    PROMPT,
    publish_synthetic,
)


def test_five_tools_dump_state_and_asymmetric_image_orientation(tmp_path: Path) -> None:
    observation, _pointer, _workspace, _control, tools, _state_store = publish_synthetic(tmp_path)
    assert [item["name"] for item in TOOL_DEFINITIONS] == [
        "observation.dump_state", "vision.inspect", "geometry.query",
        "robot.read_manual", "robot.submit_action",
    ]
    dumped = tools.invoke("observation.dump_state", {})
    assert dumped["state"]["step"] == 0
    assert dumped["state"]["observation"]["id"] == observation["observation_id"]
    assert dumped["image_camera"] == "head_camera"
    assert "success_predicate" not in dumped["state"]
    inspected = tools.invoke("vision.inspect", {"camera": "head_camera", "resolution": "full"})
    import base64
    import io

    image = Image.open(io.BytesIO(base64.b64decode(inspected["image_base64"]))).convert("RGB")
    assert image.size == (4, 3)
    assert image.getpixel((0, 0)) == (255, 0, 0)
    assert image.getpixel((3, 0)) == (0, 255, 0)
    assert image.getpixel((0, 2)) == (0, 0, 255)
    assert image.getpixel((3, 2)) == (255, 255, 0)
    assert inspected["coordinate_convention"] == "opencv_top_left_uv"
    assert observation["cameras"]["head_camera"]["width"] == 4


def test_geometry_crop_manual_and_terminal_submit(tmp_path: Path) -> None:
    _observation, pointer, _workspace, control, tools, _state_store = publish_synthetic(tmp_path)
    crop = tools.invoke("vision.inspect", {"camera": "head_camera", "crop": [1, 1, 4, 3], "resolution": "full"})
    assert crop["source_roi_xyxy"] == [1, 1, 4, 3]
    assert crop["returned_size"] == [3, 2]
    geometry = tools.invoke("geometry.query", {"camera": "head_camera", "points": [[2, 1]], "window_px": 0})
    assert geometry["points"][0]["world_xyz_m"] == [2.0, 1.0, 1.0]
    manual = tools.invoke("robot.read_manual", {"section": "Frames"})
    assert "OpenCV" in manual["content"]
    manual_definition = next(
        item for item in tools.definitions() if item["name"] == "robot.read_manual"
    )
    assert manual_definition["inputSchema"]["properties"]["section"]["enum"] == [
        "Embodiment", "Cameras", "Saved state", "Frames", "Actions", "Safety",
    ]
    for invalid in ("frames", "Frame", "Frames extra"):
        with pytest.raises(ToolError) as captured:
            tools.invoke("robot.read_manual", {"section": invalid})
        assert captured.value.code == "unknown_section"
        assert captured.value.constraints["sections"] == [
            "Embodiment", "Cameras", "Saved state", "Frames", "Actions", "Safety",
        ]
    result = tools.invoke("robot.submit_action", {
        "action": "move_pose", "arm": "left", "position_m": [0.0, 0.0, 1.02],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    })
    assert result["accepted"] is True
    action = json.loads((control / "submitted-action.json").read_text())
    assert action["observation_id"].startswith("synthetic:synthetic_tool_boundary")
    assert action["action"] == "move_pose"
    assert json.loads((control / "current-observation.json").read_text())["open"] is False
    with pytest.raises(ToolError, match="closed"):
        tools.invoke("vision.inspect", {"camera": "head_camera"})


def test_gateway_lists_exact_tools_and_returns_png_image(tmp_path: Path) -> None:
    _observation, _pointer, _workspace, _control, tools, _state_store = publish_synthetic(tmp_path)
    gateway = NativeToolGateway(tools)
    listed = gateway.delivered()
    assert [tool["name"] for tool in listed] == [
        "observation.dump_state", "vision.inspect", "geometry.query",
        "robot.read_manual", "robot.submit_action",
    ]
    delivered_manual = next(item for item in listed if item["name"] == "robot.read_manual")
    runtime_manual = next(
        item for item in tools.definitions() if item["name"] == "robot.read_manual"
    )
    assert delivered_manual == runtime_manual
    response = gateway.execute("vision.inspect", {"camera": "head_camera"})
    assert base64.b64decode(response["image_base64"]).startswith(b"\x89PNG")
    dumped = gateway.execute("observation.dump_state", {"step": 0})
    assert dumped["state"]["step"] == 0
    assert isinstance(dumped["image_base64"], str)


def test_tool_ablation_is_enforced_and_manual_is_not_materialized(tmp_path: Path) -> None:
    source = tmp_path / "source"
    session = SyntheticSession(tmp_path / "native", "synthetic_tool_boundary", 0)
    session.start()
    observation, _ = session.observe(source)
    workspace = tmp_path / "workspace"
    control = tmp_path / "control"
    projector = ContextProjector(
        workspace, control, PROMPT, MANUAL, AGENT_RULES, expose_manual=False
    )
    state_store = PolicyStateStore(tmp_path, "ablation-run", "test-plan")
    snapshot = state_store.publish(
        observation=observation, observation_root=source / observation["observation_id"],
        cause="reset", task_instruction="Move once.", previous_public_action=None,
        remaining_budget={"robot_decisions": 1},
    )
    state_record = state_store.steps[-1]
    pointer = projector.publish(
        observation=observation, observation_root=source / observation["observation_id"],
        task_instruction="Move once.", remaining_budget={"robot_decisions": 1},
        policy_state={
            "step": 0, "path": state_record["path"], "sha256": state_record["sha256"],
            "state_sha256": snapshot["state_sha256"],
        },
    )
    assert pointer["manual"] is None
    assert not (workspace / "contract/robot-manual.md").exists()
    assert (workspace / "AGENTS.md").stat().st_mode & 0o222 == 0
    tools = RobotToolRuntime(
        control, MANUAL, tmp_path / "events.jsonl", BOUNDS,
        allowed_tools=["vision.inspect", "robot.submit_action"],
        state_root=state_store.root,
    )
    with pytest.raises(ToolError, match="disabled"):
        tools.invoke("geometry.query", {"camera": "head_camera", "points": [[0, 0]]})
    with pytest.raises(ToolError, match="disabled"):
        tools.invoke("observation.dump_state", {})


def test_invalid_action_and_hidden_fields_fail_closed() -> None:
    action = {
        "schema": "ora-action.v3", "observation_id": "o", "action_id": "a",
        "action": "move_pose", "bound_observation_sha256": "h",
        "arguments": {"arm": "left", "position_m": [0, 0, 1], "orientation_xyzw": [0, 0, 0, 1]},
    }
    assert validate_runtime_action(action)["action"] == "move_pose"
    scaled_quaternion = json.loads(json.dumps(action))
    scaled_quaternion["arguments"]["orientation_xyzw"] = [0, 0, 0, 1.02]
    assert validate_runtime_action(scaled_quaternion)["arguments"][
        "orientation_xyzw"
    ] == pytest.approx([0, 0, 0, 1])
    target_state = {
        **scaled_quaternion,
        "schema": "ora-action.v4",
        "action": "target_state",
        "arguments": {
            "arm": "left",
            "pose": {
                "position_m": [0, 0, 1],
                "orientation_xyzw": [0, 0, 0, 0.98],
            },
        },
    }
    assert validate_runtime_action(target_state)["arguments"]["pose"][
        "orientation_xyzw"
    ] == pytest.approx([0, 0, 0, 1])
    with pytest.raises(ActionValidationError):
        validate_runtime_action({**action, "action": "stop_episode"})
    poisoned = dict(action)
    poisoned["arguments"] = {**action["arguments"], "object_gt_pose": [0, 0, 0]}
    with pytest.raises(ActionValidationError, match="forbidden"):
        validate_runtime_action(poisoned)
    gripper = {
        **action, "action": "set_gripper", "arguments": {"arm": "left", "gripper": 0.0},
    }
    assert validate_runtime_action(gripper)["action"] == "set_gripper"
    combined = {
        **action,
        "arguments": {**action["arguments"], "gripper": 0.0},
    }
    with pytest.raises(ActionValidationError, match="move_pose arguments"):
        validate_runtime_action(combined)
    reached, residual = _motion_reached({
        "commanded_pose_xyz_xyzw": [0, 0, 1, 0, 0, 0, 1.02],
        "reached_pose_xyz_xyzw": [0, 0, 1, 0, 0, 0, 0.98],
    })
    assert reached is True
    assert residual["orientation_residual_rad"] == pytest.approx(0.0)


def test_move_and_gripper_receipts_verify_unspecified_dimension(tmp_path: Path) -> None:
    session = SyntheticSession(
        tmp_path / "native", "synthetic_tool_boundary", 0, max_actions=2,
    )
    session.start()
    observation, _ = session.observe(tmp_path / "observations")
    move = {
        "schema": "ora-action.v3", "observation_id": observation["observation_id"],
        "action_id": "move", "action": "move_pose", "bound_observation_sha256": "h",
        "arguments": {"arm": "left", "position_m": [0.01, 0, 1.02],
                      "orientation_xyzw": [0, 0, 0, 1]},
    }
    move_receipt = session.execute(move)["receipt"]
    assert move_receipt["preserved_dimension"] == {
        "field": "gripper", "before": 1.0, "after": 1.0,
        "residual": 0.0, "verified": True,
    }
    observation, _ = session.observe(tmp_path / "observations")
    close = {
        "schema": "ora-action.v3", "observation_id": observation["observation_id"],
        "action_id": "close", "action": "set_gripper", "bound_observation_sha256": "h2",
        "arguments": {"arm": "left", "gripper": 0.0},
    }
    close_receipt = session.execute(close)["receipt"]
    assert close_receipt["preserved_dimension"]["field"] == "eef_pose"
    assert close_receipt["preserved_dimension"]["verified"] is True
    assert close_receipt["gripper_evidence"] == {
        "measured_before": 1.0, "measured_after": 0.0,
        "closure_delta": 1.0, "closure_residual": 0.0, "grasp_success": None,
    }


def test_tool_errors_are_structured_and_policy_visible(tmp_path: Path) -> None:
    _observation, _pointer, _workspace, control, tools, _state_store = publish_synthetic(tmp_path)
    with pytest.raises(ToolError) as raised:
        tools.invoke("robot.submit_action", {
            "action": "move_pose", "arm": "left", "position_m": [99, 0, 1],
            "orientation_xyzw": [0, 0, 0, 1],
        })
    assert raised.value.as_dict() == {
        "code": "out_of_bounds",
        "message": "position x is outside workspace bounds",
        "retryable": True,
        # Refused on its own argument values: the unchanged request is refused again.
        "retry_requires_changed_arguments": True,
        "field": "position_m[0]",
        "received": 99.0,
        "constraints": {"minimum": -0.6, "maximum": 0.6},
        "infrastructure": False,
    }
    event = json.loads((tmp_path / "events.jsonl").read_text().splitlines()[-1])
    assert event["type"] == "tool_error"
    assert event["error"] == raised.value.as_dict()
    with pytest.raises(ToolError) as section:
        tools.invoke("robot.read_manual", {"section": "not-a-section"})
    assert section.value.code == "unknown_section"
    assert section.value.constraints["sections"]
    with pytest.raises(ToolError) as artifact:
        tools.invoke("vision.inspect", {"camera": "head_camera", "step": 99})
    # A step past the saved chain is the argument, with the saved range (census C2).
    assert artifact.value.code == "invalid_arguments"
    assert artifact.value.field == "step"
    assert artifact.value.constraints == {"minimum": 0, "maximum": 0, "scope": "run"}
    tools.invoke("robot.submit_action", {
        "action": "set_gripper", "arm": "left", "gripper": 0.5,
    })
    with pytest.raises(ToolError) as duplicate:
        tools.invoke("robot.submit_action", {
            "action": "set_gripper", "arm": "left", "gripper": 0.4,
        })
    assert duplicate.value.code == "duplicate_action"
    with pytest.raises(ToolError) as stale:
        tools.invoke("vision.inspect", {"camera": "head_camera"})
    # A closed observation has its own code; stale_observation refuses a value
    # bound to an earlier observation and is retryable (EXP11B census E5).
    assert stale.value.code == "action_pending_fresh_observation"
    assert stale.value.retryable is False
    unavailable = RobotToolRuntime(
        tmp_path / "missing-control", MANUAL, tmp_path / "missing-events.jsonl",
        BOUNDS, state_root=tmp_path / "policy-state",
    )
    with pytest.raises(ToolError) as missing:
        unavailable.invoke("robot.submit_action", {
            "action": "set_gripper", "arm": "left", "gripper": 0.5,
        })
    assert missing.value.code == "environment_unavailable"


def test_the_legacy_codec_no_longer_produces_a_stagnation_judgement(
    tmp_path: Path,
) -> None:
    """The deletion is deliberate, not a claim that nothing was running.

    On `runtime_action_v3` the guard worked: `arguments` carried `arm` and the
    receipt carried a numeric `gripper`, so four identical no-progress actions
    raised `warning` on the third and armed `is_high_confidence` on the
    fourth. That profile is still resolvable, so this test pins the loss
    rather than an unreachable branch. It was removed because the Harness
    should not form and deliver that judgement to the policy, and because it
    was already constant-false on every profile a maintained backend resolves
    to. See `ora/orchestrator/stagnation.py`.
    """

    image = tmp_path / "same.png"
    Image.new("RGB", (32, 32), (12, 34, 56)).save(image)
    guard = StagnationGuard(tmp_path / "control/stagnation.json")
    records = [
        guard.observe(before_image=image, after_image=image) for _ in range(4)
    ]

    assert records == [{"head_phash_hamming": 0}] * 4
    persisted = json.loads((tmp_path / "control/stagnation.json").read_text())
    assert persisted == {
        "schema": "ora.observation-delta.v3",
        "records": records,
    }
