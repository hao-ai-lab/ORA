"""move_pose takes an orientation as world approach and jaw-line directions; every pose target echoes them.

EXP11A g012 (E11G12-007): organize_table, stack_bowls and arrange_largest_number
each meant a downward approach and hand-computed ``[0, -0.7071, 0, 0.7071]``,
whose link6 local +X (the approach axis) points world +Z. The planner refused
and the policy blamed reach or self-collision. Under
``pose_orientation_from_world_axes_v1`` the ``world_axes`` orientation mode
compiles ``approach_world`` and ``jaw_line_world`` deterministically into the
absolute quaternion, and every ``move_pose``/``move_to``/``rotate_*`` target
reports ``target_axes_world`` from the quaternion it submits. An upward
absolute quaternion stays legal and unchanged. Offline: pure compiler checks,
the resolved surfaces and the fake Dojo plant runtime.
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import replace

import numpy as np
import pytest

from ora.agent.composition import resolve_tool_surface
from ora.agent.composition.tool_surface import ToolSurfaceError
from ora.agent.composition.world_axes_schema import (
    CAPABILITY, MODE, TARGET_AXES_PROPERTY, TOOLS, bind_world_axes_definitions, bind_world_axes_orientation,
)
from ora.agent.tools import world_axes
from ora.agent.tools.robot import ToolError
from ora.contracts.pose import quaternion_matrix

DOJO = "robodojo_placement_closed_skills_v2"
# The released RoboTwin surface (the v2 default declares the mode with M5).
ROBOTWIN_LIVE = ("robotwin_toolchain_b2_closed_noservo_v1",)
X5 = {"approach": [1.0, 0.0, 0.0], "closing": [0.0, 1.0, 0.0]}
PANDA = {"approach": [0.0, 0.0, 1.0], "closing": [1.0, 0.0, 0.0]}
DOWN = [0.0, 0.0, -1.0]
#: The three g012 requests that meant "pointing down" (organize_table seq 481,
#: stack_bowls seq 1623, arrange_largest_number seq 1992).
G012_UPWARD = ([0.0, -0.7071068, 0.0, 0.7071068], [0.0, -0.7071, 0.0, 0.7071], [0.0, -0.7071, 0.0, 0.7071])


def _columns(quaternion):
    return np.asarray(quaternion_matrix(quaternion), dtype=float)


def _unit(vector):
    vector = np.asarray(vector, dtype=float)
    return vector / np.linalg.norm(vector)


@pytest.mark.parametrize("jaw_line", [[0, 1, 0], [1, 0, 0], [1, 1, 0], [0, -2, 0], [0.3, -0.8, 0.2]])
def test_a_downward_approach_compiles_to_local_x_down_and_local_y_along_the_jaw_line(jaw_line):
    quaternion = world_axes.quaternion_from_world_axes(DOWN, jaw_line, local_axes=X5, arm="left")
    rotation = _columns(quaternion)
    assert np.linalg.norm(quaternion) == pytest.approx(1.0)
    assert rotation[:, 0] == pytest.approx(DOWN, abs=1e-9)
    # The jaw line keeps only its part perpendicular to the approach.
    expected_jaw = _unit(np.asarray(jaw_line, float) - np.dot(jaw_line, DOWN) * np.asarray(DOWN))
    assert rotation[:, 1] == pytest.approx(expected_jaw, abs=1e-9)


def test_the_downward_jaw_along_world_y_is_the_quaternion_g012_meant():
    quaternion = world_axes.quaternion_from_world_axes(DOWN, [0, 1, 0], local_axes=X5, arm="right")
    assert quaternion == pytest.approx([0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)])


@pytest.mark.parametrize("approach,jaw_line", [([1, 0, 0], [0, 0, 1]), ([0.6, 0, -0.8], [0, 1, 0]),
                                               ([0, 0, 3], [0, 1, 0]), ([-1, 1, 0], [0, 0, -1])])
def test_any_approach_round_trips_whatever_its_length(approach, jaw_line):
    rotation = _columns(world_axes.quaternion_from_world_axes(approach, jaw_line, local_axes=X5, arm="left"))
    assert rotation[:, 0] == pytest.approx(_unit(approach), abs=1e-9)
    assert rotation[:, 1] == pytest.approx(_unit(jaw_line), abs=1e-9)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_the_axes_are_placed_where_the_embodiment_declares_them():
    rotation = _columns(world_axes.quaternion_from_world_axes(DOWN, [0, 1, 0], local_axes=PANDA, arm="right"))
    assert rotation[:, 2] == pytest.approx(DOWN, abs=1e-9) and rotation[:, 0] == pytest.approx([0, 1, 0], abs=1e-9)


@pytest.mark.parametrize("field,approach,jaw_line", [
    ("approach_world", [0, 0, 0], [0, 1, 0]),
    ("approach_world", [1e-9, 0, 0], [0, 1, 0]),
    ("jaw_line_world", DOWN, [0, 0, 0]),
])
def test_a_near_zero_direction_is_refused_naming_the_field_and_the_tolerance(field, approach, jaw_line):
    with pytest.raises(ToolError) as caught:
        world_axes.quaternion_from_world_axes(approach, jaw_line, local_axes=X5, arm="left")
    error = caught.value
    assert error.code == "invalid_arguments" and error.field == f"targets.left.{field}"
    assert error.constraints["min_length"] == world_axes.MIN_DIRECTION_LENGTH
    assert str(world_axes.MIN_DIRECTION_LENGTH) in str(error)


@pytest.mark.parametrize("jaw_line", [[0, 0, -1], [0, 0, 5], [math.sin(math.radians(2)), 0, -math.cos(math.radians(2))]])
def test_a_jaw_line_near_parallel_to_the_approach_is_refused_with_the_angle_bound(jaw_line):
    with pytest.raises(ToolError) as caught:
        world_axes.quaternion_from_world_axes(DOWN, jaw_line, local_axes=X5, arm="right")
    error = caught.value
    assert error.code == "invalid_arguments" and error.field == "targets.right.jaw_line_world"
    assert error.constraints["min_angle_from_approach_deg"] == world_axes.MIN_SEPARATION_DEG == 5.0
    assert error.constraints["received_angle_deg"] < 5.0
    assert "5.0 degrees" in str(error)


def test_a_jaw_line_just_outside_the_tolerance_compiles():
    tilt = math.radians(6)
    rotation = _columns(world_axes.quaternion_from_world_axes(
        DOWN, [math.sin(tilt), 0, -math.cos(tilt)], local_axes=X5, arm="left"))
    assert rotation[:, 1] == pytest.approx([1, 0, 0], abs=1e-9)


@pytest.mark.parametrize("vector", [[0, 1], [0, 1, 0, 0], ["0", 1, 0], [math.nan, 0, 1], [True, 0, 0]])
def test_a_malformed_direction_is_refused_by_name(vector):
    with pytest.raises(ToolError) as caught:
        world_axes.quaternion_from_world_axes(vector, [1, 0, 0], local_axes=X5, arm="left")
    assert caught.value.field == "targets.left.approach_world"


@pytest.mark.parametrize("quaternion", G012_UPWARD)
def test_the_g012_quaternions_echo_an_upward_approach(quaternion):
    axes = world_axes.target_axes_world(quaternion, local_axes=X5)
    assert axes["approach"] == pytest.approx([0.0, 0.0, 1.0], abs=1e-4)
    assert axes["jaw_line"] == pytest.approx([0.0, 1.0, 0.0], abs=1e-4)
    meant = world_axes.target_axes_world([0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)], local_axes=X5)
    assert meant == {"approach": [0.0, 0.0, -1.0], "jaw_line": [0.0, 1.0, 0.0]}


# --------------------------------------------------------------------------
# Composition: the Dojo surface declares the mode and the echo.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dojo():
    return resolve_tool_surface(DOJO)


def _arm_schemas(surface):
    move_pose = next(item for item in surface.definitions if item["name"] == "move_pose")
    return move_pose, [move_pose["inputSchema"]["properties"]["targets"]["properties"][arm]["properties"]
                       for arm in ("left", "right")]


def test_the_dojo_move_pose_advertises_world_axes_with_its_companion_fields(dojo):
    assert dojo.supports(CAPABILITY)
    move_pose, arms = _arm_schemas(dojo)
    for properties in arms:
        assert properties["orientation_mode"]["enum"][-1] == MODE
        assert "world_axes REQUIRES approach_world AND jaw_line_world" in properties["orientation_mode"]["description"]
        for name, companion in (("approach_world", "jaw_line_world"), ("jaw_line_world", "approach_world")):
            field = properties[name]
            # optional_argument_null_as_omitted_v1 (bound after this binder) adds null.
            assert (field["type"], field["minItems"], field["maxItems"]) == (["array", "null"], 3, 3)
            assert field["description"].startswith(f"REQUIRED with orientation_mode=world_axes, together with {companion}.")
    assert "orientation_mode (hold_current, absolute_quaternion, axis_angle_delta, world_axes)" in move_pose["description"]
    assert "world_axes takes approach_world and jaw_line_world;" in move_pose["description"]
    assert "target_axes_world" in move_pose["description"]
    for name in TOOLS:
        target = dojo.result_schemas[name]["properties"]["compiled_pose"]["properties"]["targets"][
            "additionalProperties"]["properties"]
        assert target["target_axes_world"] == TARGET_AXES_PROPERTY


def test_the_definitions_binder_is_idempotent_over_the_released_schema(dojo):
    # Resolution binds optional_argument_null_as_omitted_v1 after this binder,
    # so idempotence is judged on the schema this binder released.
    from tests.environments.robodojo.test_robodojo_closed_skill_surface_contract import _without_null_rule

    released = _without_null_rule(dojo.profile_id)
    definitions = copy.deepcopy(list(released.definitions))
    bind_world_axes_definitions(definitions, released.capabilities, released.view_config)
    assert definitions == list(released.definitions)


@pytest.mark.parametrize("profile", ROBOTWIN_LIVE)
def test_robotwin_surfaces_do_not_declare_it(profile):
    surface = resolve_tool_surface(profile)
    assert not surface.supports(CAPABILITY)
    assert MODE not in json.dumps([item["inputSchema"] for item in surface.definitions])


def test_the_binder_refuses_a_surface_it_cannot_serve(dojo):
    definitions, results = copy.deepcopy(list(dojo.definitions)), copy.deepcopy(dict(dojo.result_schemas))
    with pytest.raises(ToolSurfaceError, match="controlled_frame_convention"):
        bind_world_axes_orientation(copy.deepcopy(definitions), copy.deepcopy(results), dojo.capabilities, {})
    without = [item for item in dojo.capabilities if item != "flat_motion_arguments_v1"]
    with pytest.raises(ToolSurfaceError, match="flat_motion_arguments_v1"):
        bind_world_axes_orientation(copy.deepcopy(definitions), copy.deepcopy(results), without, dojo.view_config)


# --------------------------------------------------------------------------
# Runtime on the fake Dojo plant.
# --------------------------------------------------------------------------


def _runtime(tmp_path, *, declared=True):
    from tests.agent.motion.test_move_pose_reach import _dojo_runtime

    runtime = _dojo_runtime(tmp_path)
    if not declared:
        runtime.surface = replace(runtime.surface, capabilities=tuple(
            item for item in runtime.surface.capabilities if item != CAPABILITY))
    return runtime


def _submitted(runtime):
    return json.loads((runtime.control_root / "submitted-action.json").read_text())


def _move(arm="left", **orientation):
    return {"targets": {arm: {"interaction": "free_space", "position_mode": "absolute",
                              "position_m": [0.05, -0.1, 0.99], **orientation}}}


def test_world_axes_submits_the_compiled_absolute_quaternion_and_echoes_its_axes(tmp_path):
    runtime = _runtime(tmp_path)
    result = runtime.invoke("move_pose", _move(orientation_mode="world_axes", approach_world=[0, 0, -2],
                                               jaw_line_world=[1, 1, 0.5]))
    submitted = _submitted(runtime)["arguments"]["targets"]["left"]["orientation_xyzw"]
    rotation = _columns(submitted)
    assert rotation[:, 0] == pytest.approx(DOWN, abs=1e-9)
    assert rotation[:, 1] == pytest.approx(_unit([1, 1, 0]), abs=1e-9)
    target = result["compiled_pose"]["targets"]["left"]
    assert target["orientation_mode"] == "absolute_quaternion"
    assert target["target_axes_world"]["approach"] == pytest.approx(DOWN, abs=1e-6)
    assert target["target_axes_world"]["jaw_line"] == pytest.approx(_unit([1, 1, 0]), abs=1e-6)


def test_a_parallel_world_axes_request_is_refused_before_anything_is_submitted(tmp_path):
    runtime = _runtime(tmp_path)
    with pytest.raises(ToolError) as caught:
        runtime.invoke("move_pose", _move(arm="right", orientation_mode="world_axes", approach_world=DOWN,
                                          jaw_line_world=[0, 0, 1]))
    assert caught.value.field == "targets.right.jaw_line_world"
    assert not (runtime.control_root / "submitted-action.json").exists()


@pytest.mark.parametrize("quaternion", G012_UPWARD)
def test_an_upward_absolute_quaternion_is_submitted_unchanged_and_echoed_upward(tmp_path, quaternion):
    runtime = _runtime(tmp_path)
    result = runtime.invoke("move_pose", _move(orientation_mode="absolute_quaternion", orientation_xyzw=quaternion))
    submitted = _submitted(runtime)["arguments"]["targets"]["left"]["orientation_xyzw"]
    assert abs(float(np.dot(_unit(submitted), _unit(quaternion)))) == pytest.approx(1.0)
    axes = result["compiled_pose"]["targets"]["left"]["target_axes_world"]
    assert axes["approach"] == pytest.approx([0.0, 0.0, 1.0], abs=1e-4)


def test_a_jaw_turn_is_echoed_as_the_jaw_line_it_submits(tmp_path):
    """Under the jaw-symmetry headroom the submitted turn reverses the jaw line, never the approach."""
    from tests.agent.motion.test_move_pose_reach import RIGHT_FROM_HOME, _bridge, _turned

    runtime = _runtime(tmp_path)
    quaternion = world_axes.quaternion_from_world_axes(DOWN, [0, 1, 0], local_axes=X5, arm="left")
    runtime.align_plan_callback = _bridge([], requested=quaternion, exact=(RIGHT_FROM_HOME, None),
                                          turned=(_turned(RIGHT_FROM_HOME), None))
    result = runtime.invoke("move_pose", _move(orientation_mode="world_axes", approach_world=DOWN,
                                               jaw_line_world=[0, 1, 0]))
    target = result["compiled_pose"]["targets"]["left"]
    assert target["jaw_symmetry"]["variant"] == "jaw_flip"
    assert target["target_axes_world"]["approach"] == pytest.approx(DOWN, abs=1e-6)
    assert target["target_axes_world"]["jaw_line"] == pytest.approx([0, -1, 0], abs=1e-6)


@pytest.mark.parametrize("tool,arguments", [("move_to", {"targets": {"left": {"position_m": [0.3, -0.1, 0.9]}}}),
                                            ("rotate_wrist", {"targets": {"right": 30.0}}),
                                            ("rotate_pitch", {"targets": {"right": -20.0}})])
def test_every_pose_primitive_echoes_its_compiled_target_axes(tmp_path, tool, arguments):
    runtime = _runtime(tmp_path)
    result = runtime.invoke(tool, arguments)
    arm, target = next(iter(result["compiled_pose"]["targets"].items()))
    orientation = _submitted(runtime)["arguments"]["targets"][arm]["orientation_xyzw"]
    assert target["target_axes_world"] == world_axes.target_axes_world(orientation, local_axes=X5)


def test_a_surface_without_the_capability_neither_accepts_the_mode_nor_echoes(tmp_path):
    runtime = _runtime(tmp_path, declared=False)
    with pytest.raises(ToolError) as caught:
        runtime.invoke("move_pose", _move(orientation_mode="world_axes", approach_world=DOWN, jaw_line_world=[0, 1, 0]))
    assert caught.value.code == "invalid_arguments"
    result = runtime.invoke("move_pose", _move(orientation_mode="hold_current"))
    assert "target_axes_world" not in result["compiled_pose"]["targets"]["left"]
