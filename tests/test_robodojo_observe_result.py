"""The live RoboDojo ``observe`` result admits what every Dojo observation carries.

EXP11A g003 run ``b50ff4c37763`` (``imitate_sorting_sequence``) called
``observe`` twice and received ``internal_tool_error`` both times:
``$result.public_robot_facts has unknown result fields ['controller_feedback']``.
The bound result (``robodojo-live-observe-results-v2.json``) also declared the
RoboTwin camera names and no ARX jaw branch, so an observe call on that surface
could not validate at all. No g003 run ever received an ``observe`` result.

These tests drive the real dispatcher with an observation in the recorded
shape -- controller contract, per-arm gripper feedback, three Dojo cameras --
and assert that the result validates and still carries every recorded field.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ora.agent.composition.tool_surface import resolve_tool_surface
from ora.agent.tools.runtime import ManipulationToolRuntime
from test_robodojo_perception import (
    BOUNDS, FakeHeadGrounding, FakeX5Proposer, _synthetic_robodojo_observation,
)

ROOT = Path(__file__).resolve().parents[1]
SURFACE = "robodojo_placement_closed_skills_v2"

# Recorded verbatim from run b50ff4c37763 observation-0002 (imitate_sorting_sequence).
RECORDED_CONTROLLER_FEEDBACK = {
    "arrival_comparison": "<=",
    "gripper_drive_limit": {
        "applies_to": "explicit_gripper_commands",
        "coupling": "both_drive_targets_from_limited_primary_target",
        "kind": "measured_primary_relative", "maximum_normalized_offset": 0.2,
        "pose_hold": "preserve_last_applied_drive_target",
        "reference": "current_primary_finger_encoder",
    },
    "gripper_tolerance_normalized": 0.03, "native_actions_per_command": 20,
    "orientation_tolerance_rad": 0.1, "position_tolerance_m": 0.01,
    "schema": "ora.controller-feedback.v1",
}


def _gripper_feedback(measured: float, requested: float, drive: float) -> dict:
    command = {0.0: "close", 1.0: "open"}[requested]
    effective = 0.1851851851851852 if requested == 0.0 else 1.0
    residual = abs(measured - effective)
    return {
        "schema": "ora.gripper-feedback.v1",
        "requested_opening_normalized": requested, "drive_target_opening_normalized": drive,
        "measured_opening_normalized": measured, "effective_target_opening_normalized": effective,
        "requested_command": command,
        "arrival_check": {"comparison": "<=", "residual_normalized": residual,
                          "tolerance_normalized": 0.03, "within_tolerance": residual <= 0.03},
        "drive_target_source": "resolved_sdk_joint_position_target",
        "measurement_source": "primary_finger_encoder",
        "pose_hold_semantics": "preserve_last_applied_drive_target",
    }


def _runtime(tmp_path: Path) -> ManipulationToolRuntime:
    source, observation = _synthetic_robodojo_observation(tmp_path / "observations")
    observation["controller_feedback"] = json.loads(json.dumps(RECORDED_CONTROLLER_FEEDBACK))
    from ora.contracts.robot_geometry import GEOMETRY_VERIFICATION_SCHEMA
    observation["robot_geometry_verification"] = {
        "schema": GEOMETRY_VERIFICATION_SCHEMA, "scope": "loaded_robot_geometry_only", "passed": True,
        "descriptor_sha256": resolve_tool_surface(SURFACE).view_config["public_geometry_descriptor"]["sha256"],
    }
    effectors = observation["proprioception"]["end_effectors"]
    # One open arm and one arm holding a stalled close, as in the recorded scene.
    effectors["left"]["gripper_feedback"] = _gripper_feedback(1.0, 1.0, 1.0)
    effectors["right"].update(gripper=0.5, gripper_command=0.0, gripper_command_target="close",
                              gripper_finger_qpos=[0.02, 0.02])
    effectors["right"]["gripper_feedback"] = _gripper_feedback(0.5, 0.0, 0.3)
    (source / "observation.json").write_text(json.dumps(observation, sort_keys=True, indent=2) + "\n")
    control = tmp_path / "control"
    control.mkdir()
    runtime = ManipulationToolRuntime(
        control_root=control,
        manual_path=ROOT / "ora/policy/prompt_bundles/templates/tool-schema-delivery-v1.md",
        event_path=tmp_path / "events.jsonl", workspace_bounds_m=BOUNDS, state_root=None,
        tool_surface_profile=SURFACE, controlled_end_effector="right",
        segmentation_backend=FakeHeadGrounding(), grounding_backend=FakeHeadGrounding(),
        grasp_backend=FakeX5Proposer())
    pointer = {"open": True, "observation_id": observation["observation_id"],
               "observation_path": str(source / "observation.json")}
    runtime._current = lambda: (pointer, source, observation)
    return runtime


def test_observe_validates_and_keeps_the_controller_contract_and_jaw_feedback(tmp_path):
    runtime = _runtime(tmp_path)
    result = runtime.invoke("observe", {})
    facts = result["public_robot_facts"]
    assert facts["controller_feedback"] == RECORDED_CONTROLLER_FEEDBACK
    assert set(facts["cameras"]) == {"head", "left_wrist", "right_wrist"}
    right = facts["end_effectors"]["right"]["gripper"]
    assert right["measured_aperture_mm"] is None
    assert right["feedback"]["drive_target_opening_normalized"] == 0.3
    assert right["command_target"] == "close"


@pytest.mark.parametrize("views", [["head"], ["left_wrist", "right_wrist"]])
def test_observe_with_selected_views_validates(tmp_path, views):
    result = _runtime(tmp_path).invoke("observe", {"views": views})
    assert set(result["public_robot_facts"]["cameras"]) == set(views)

