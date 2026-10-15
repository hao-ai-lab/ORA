"""Robotwin motion v2 surface contract regressions."""

from __future__ import annotations
from pathlib import Path
from typing import Any
import pytest
from ora.agent.composition import resolve_tool_surface
from ora.agent.tools.runtime import ManipulationToolRuntime
from ora.agent.tools.robot import ToolError
from ora.contracts import MOTION_ACTION_V3_SCHEMA, compile_enhanced_move_pose
from tests.environments.robotwin.test_robotwin_motion_v2 import (
    pose,
    primitive_runtime,
)

#: The kept RoboTwin default surface (primitive_runtime's default).
DEFAULT = "robotwin_toolchain_b2_closed_noservo_v2"


def test_runtime_camera_allowlist_tracks_surface_observer_capability() -> None:
    runtime = object.__new__(ManipulationToolRuntime)
    runtime.surface = resolve_tool_surface(DEFAULT)
    assert not runtime.surface.supports("policy_observer_camera_v1")
    assert runtime._robotwin_policy_cameras() == {
        "head_camera", "front_camera", "left_camera", "right_camera",
    }


def test_move_pose_argument_errors_carry_contract_examples(
    tmp_path: Path,
) -> None:
    """EXP06A closing repair A5: rejected formats teach the accepted contract."""

    from ora.agent.tools.motion_arguments import arm_target
    from ora.contracts.motion import compile_enhanced_move_pose

    runtime, observation, _persisted = primitive_runtime(tmp_path)
    with pytest.raises(ToolError) as excinfo:
        runtime._motion_move_pose_v2({"targets": {"left": {"pose": "up"}}})
    error = excinfo.value
    assert error.code == "invalid_motion_arguments"
    examples = error.constraints["examples"]
    # This surface's move_pose admits no plan_ref, so none is taught.
    move_pose = next(item["inputSchema"] for item in runtime.surface.definitions
                     if item["name"] == "move_pose")
    assert "plan_ref" not in move_pose["properties"]
    assert all("plan_ref" not in example for example in examples)

    # Every advertised example must satisfy the actual compile contract, so
    # the taught format can never drift from the accepted one.
    current = observation["proprioception"]["end_effectors"]["left"][
        "pose_xyz_xyzw"
    ]
    compiled_examples = 0
    for example in examples:
        for request in example.get("targets", {}).values():
            # The flat spelling this surface publishes reads as the nested one.
            pose_request = arm_target({
                key: value for key, value in request.items()
                if key != "interaction"
            })
            compiled = compile_enhanced_move_pose(
                pose_request,
                current_pose_xyz_xyzw=current,
                controlled_point="end_effector",
            )
            assert len(compiled["target_pose_xyz_xyzw"]) == 7
            compiled_examples += 1
    # The bare canonical TCP pose is not in this surface's published schema,
    # so it is no longer taught here (EXP11B census 1): every example must
    # also pass the delivered validator (test_motion_refusal_envelopes.py).
    assert compiled_examples >= 3


def test_pose_primitives_preserve_restricted_dimensions_and_public_intent(
    tmp_path: Path,
) -> None:
    runtime, _observation, persisted = primitive_runtime(tmp_path)
    moved = runtime._move_to({
        "targets": {"right": {"position_m": [0.3, 0.1, 1.1]}},
    })
    assert moved["action"] == "move_to"
    assert moved["compiled_pose"]["canonical_action"] == "move_pose"
    assert moved["compiled_pose"]["targets"]["right"][
        "target_pose_xyz_xyzw"
    ] == pytest.approx([0.3, 0.1, 1.1, 0.0, 0.0, 0.0, 1.0])
    assert persisted[-1]["action"] == "move_pose"
    assert persisted[-1]["arguments"]["targets"]["right"] == pose(0.3) | {
        "position_m": [0.3, 0.1, 1.1],
    }

    wrist = runtime._motion_rotate_wrist_v2({"targets": {"left": 90.0}})
    root_half = 2.0 ** -0.5
    assert wrist["action"] == "rotate_wrist"
    assert wrist["compiled_pose"]["targets"]["left"][
        "target_pose_xyz_xyzw"
    ] == pytest.approx([
        -0.2, 0.0, 1.0, root_half, 0.0, 0.0, root_half,
    ])

    pitch = runtime._motion_rotate_pitch_v2({"targets": {"right": 90.0}})
    assert pitch["action"] == "rotate_pitch"
    assert pitch["compiled_pose"]["targets"]["right"][
        "target_pose_xyz_xyzw"
    ] == pytest.approx([
        0.2, 0.0, 1.0, 0.0, root_half, 0.0, root_half,
    ])


def test_exp06a_v10_compiles_intent_without_policy_controller_route(
    tmp_path: Path,
) -> None:
    runtime, _observation, persisted = primitive_runtime(tmp_path)
    runtime._motion_move_pose_v2({"targets": {"right": {
        "position": {
            "mode": "delta", "translation_m": [0.0, 0.0, -0.02],
            "frame": "world",
        },
        "orientation": {"mode": "hold_current"},
        "interaction": "contact_approach",
    }}})
    submitted = persisted[-1]
    assert submitted["schema"] == MOTION_ACTION_V3_SCHEMA
    assert submitted["arguments"]["interactions"] == {
        "right": "contact_approach",
    }
    assert "execution_mode" not in submitted["arguments"]
    runtime._motion_rotate_wrist_v2({"targets": {"left": 90.0}})
    assert persisted[-1]["arguments"]["interactions"] == {
        "left": "precision",
    }
    assert "execution_mode" not in persisted[-1]["arguments"]
    runtime._move_to({
        "targets": {"right": {"position_m": [0.3, 0.1, 1.1]}},
    })
    assert persisted[-1]["arguments"]["interactions"] == {
        "right": "free_space",
    }


def test_recover_pose_maps_public_previous_and_home_without_episode_reset(
    tmp_path: Path,
) -> None:
    runtime, _observation, persisted = primitive_runtime(tmp_path)
    result = runtime._motion_recover_pose_v2({
        "targets": {"left": "previous", "right": "home"},
    })
    assert result["action"] == "recover_pose"
    assert result["compiled_pose"]["canonical_action"] == "recover_motion"
    assert persisted[-1]["action"] == "recover_motion"
    assert persisted[-1]["arguments"]["targets"] == {
        "left": "backtrack_last_safe", "right": "home",
    }


def test_enhanced_move_pose_compiles_pure_translation_without_copying_quaternion() -> None:
    compiled = compile_enhanced_move_pose({
        "position": {
            "mode": "delta",
            "translation_m": [0.0, 0.0, 0.04],
            "frame": "world",
        },
        "orientation": {"mode": "hold_current"},
    }, current_pose_xyz_xyzw=[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0])
    assert compiled["target_pose_xyz_xyzw"] == pytest.approx([
        0.1, 0.2, 0.34, 0.0, 0.0, 0.0, 1.0,
    ])
    assert compiled["position_mode"] == "delta"
    assert compiled["orientation_mode"] == "hold_current"


def test_enhanced_move_pose_compiles_standard_absolute_proposal_pose() -> None:
    compiled = compile_enhanced_move_pose({
        "position_m": [0.1, 0.2, 0.4],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }, current_pose_xyz_xyzw=[
        0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0,
    ])
    assert compiled["target_pose_xyz_xyzw"] == pytest.approx([
        0.1, 0.2, 0.4, 0.0, 0.0, 0.0, 1.0,
    ])
    assert compiled["position_mode"] == "absolute"
    assert compiled["orientation_mode"] == "absolute_quaternion"

    with pytest.raises(ValueError, match="either a standard absolute TCP pose"):
        compile_enhanced_move_pose({
            "position_m": [0.1, 0.2, 0.4],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "position": {"mode": "hold_current"},
        }, current_pose_xyz_xyzw=[
            0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0,
        ])


def test_enhanced_move_pose_compiles_rotation_only_about_tool_axis() -> None:
    compiled = compile_enhanced_move_pose({
        "position": {"mode": "hold_current"},
        "orientation": {
            "mode": "axis_angle_delta",
            "axis": "z",
            "angle_deg": 90.0,
            "frame": "end_effector",
        },
    }, current_pose_xyz_xyzw=[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0])
    root_half = 2.0 ** -0.5
    assert compiled["target_pose_xyz_xyzw"] == pytest.approx([
        0.1, 0.2, 0.3, 0.0, 0.0, root_half, root_half,
    ])
    assert compiled["translation_distance_m"] == 0.0
    assert compiled["rotation_angle_deg"] == pytest.approx(90.0)


def test_enhanced_move_pose_resolves_tool_frame_translation() -> None:
    root_half = 2.0 ** -0.5
    compiled = compile_enhanced_move_pose({
        "position": {
            "mode": "delta",
            "translation_m": [0.1, 0.0, 0.0],
            "frame": "end_effector",
        },
        "orientation": {"mode": "hold_current"},
    }, current_pose_xyz_xyzw=[
        0.0, 0.0, 1.0, 0.0, 0.0, root_half, root_half,
    ])
    assert compiled["target_pose_xyz_xyzw"][:3] == pytest.approx([
        0.0, 0.1, 1.0,
    ])


@pytest.mark.parametrize("field,value", [
    ("no_op", {
        "position": {"mode": "hold_current"},
        "orientation": {"mode": "hold_current"},
    }),
    ("angle", {
        "position": {"mode": "hold_current"},
        "orientation": {
            "mode": "axis_angle_delta", "axis": "z",
            "angle_deg": 181.0, "frame": "end_effector",
        },
    }),
    ("zero_delta", {
        "position": {
            "mode": "delta", "translation_m": [0.0, 0.0, 0.0],
            "frame": "world",
        },
        "orientation": {"mode": "hold_current"},
    }),
])
def test_enhanced_move_pose_rejects_noop_and_invalid_angle(
    field: str, value: dict[str, Any],
) -> None:
    del field
    with pytest.raises(ValueError):
        compile_enhanced_move_pose(
            value,
            current_pose_xyz_xyzw=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        )
