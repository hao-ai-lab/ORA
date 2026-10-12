"""Motion refusals teach shapes the surface admits and name public fields and bounds.

EXP11B refusal census 1, 2, 3 and 5 (0059 open-bugs triage rows 5, 6, 17, 24):

* every ``move_pose`` refusal example is written in the spelling the bound
  schema admits and compiles; the nested examples were refused by every flat
  surface;
* a rotation angle or quaternion outside the compiler's bounds is refused under
  its public field with the bound, and a compiler message names the primitive
  that was called, not ``move_pose``;
* the workspace refusal names the argument the target came from, not ``xyz_m``;
* an undeclared tool is not retryable and lists the declared ones.
"""

from __future__ import annotations

import pytest

from ora.agent.composition import resolve_tool_surface, validate_schema_instance
from ora.agent.composition.tool_surface import PROFILE_CATALOG
from ora.agent.tools import motion_arguments, motion_refusals
from ora.agent.tools.runtime.move_pose import ENHANCED_MOVE_POSE_EXAMPLES
from ora.agent.tools.robot import ToolError
from ora.contracts import compile_enhanced_move_pose

HOME = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
ENHANCED = sorted(
    profile for profile in PROFILE_CATALOG
    if resolve_tool_surface(profile).supports("enhanced_move_pose_v1")
)


def _refused(call, *args, **kwargs) -> dict:
    with pytest.raises(ToolError) as caught:
        call(*args, **kwargs)
    return caught.value.as_dict()


@pytest.mark.parametrize("profile", ENHANCED)
def test_every_move_pose_example_passes_the_delivered_schema_and_the_compiler(profile):
    surface = resolve_tool_surface(profile)
    schema = motion_arguments.surface_move_pose_schema(surface)
    examples = motion_refusals.move_pose_examples(surface, ENHANCED_MOVE_POSE_EXAMPLES)
    targets_only = [example for example in ENHANCED_MOVE_POSE_EXAMPLES if "plan_ref" not in example]
    written = [item for item in examples if "plan_ref" not in item]
    # A flat surface gets every example back in its own spelling; a nested one
    # that publishes no bare TCP pose loses only that one.
    if motion_arguments.declares_flat_targets(schema):
        assert len(written) == len(targets_only)
    else:
        assert len(written) >= len(targets_only) - 1
    for example in examples:
        validate_schema_instance(example, schema, root_path="$input")
        if "plan_ref" in example:
            continue
        arms = example.get("targets", {"right": example})
        for target in motion_arguments.arm_targets(arms).values():
            request = {key: value for key, value in target.items() if key != "interaction"}
            compile_enhanced_move_pose(request, current_pose_xyz_xyzw=HOME, controlled_point="end_effector")


def _dojo(tmp_path):
    from test_move_pose_reach import _dojo_runtime

    return _dojo_runtime(tmp_path)


def test_a_refused_flat_move_pose_is_taught_the_flat_shape(tmp_path):
    runtime = _dojo(tmp_path)
    error = _refused(runtime._motion_move_pose_v2, {"targets": {"left": {"position": {"mode": "absolute"}}}})
    schema = motion_arguments.surface_move_pose_schema(runtime.surface)
    assert error["constraints"]["examples"]
    for example in error["constraints"]["examples"]:
        assert "position_mode" in example["targets"][next(iter(example["targets"]))]
        validate_schema_instance(example, schema, root_path="$input")


@pytest.mark.parametrize("target,field,bound", [
    ({"orientation_mode": "axis_angle_delta", "rotation_axis": "x", "rotation_angle_deg": 200.0,
      "orientation_frame": "end_effector"}, "targets.left.rotation_angle_deg", {"maximum": 180.0}),
    ({"orientation_mode": "absolute_quaternion", "orientation_xyzw": [0.0, 0.0, 0.0, 2.0]},
     "targets.left.orientation_xyzw", {"norm_maximum": 1.02, "received_norm": 2.0}),
])
def test_compiler_bounds_are_refused_under_the_public_field(tmp_path, target, field, bound):
    runtime = _dojo(tmp_path)
    request = {"position_mode": "hold_current", "interaction": "free_space", **target}
    error = _refused(runtime._motion_move_pose_v2, {"targets": {"left": request}})
    assert (error["field"], error["received"]) == (field, target.get("rotation_angle_deg",
                                                                       target.get("orientation_xyzw")))
    assert bound.items() <= error["constraints"].items()


def test_a_rotate_primitive_is_refused_as_itself_with_its_bound(tmp_path):
    runtime = _dojo(tmp_path)
    beyond = _refused(runtime._motion_rotate_wrist_v2, {"targets": {"left": 200.0}})
    assert beyond["field"] == "targets.left"
    assert beyond["constraints"] == {"minimum": -180.0, "maximum": 180.0, "unit": "degrees"}
    unchanged = _refused(runtime._motion_rotate_pitch_v2, {"targets": {"left": 0.0}})
    assert "move_pose" not in unchanged["message"] and "rotate_pitch" in unchanged["message"]
    assert unchanged["field"] == "targets.left"


@pytest.mark.parametrize("tool,arguments", [
    ("move_pose", {"targets": {"left": {"interaction": "free_space", "position_mode": "absolute",
                                        "position_m": [0.0, 0.0, 5.0], "orientation_mode": "hold_current"}}}),
    ("move_to", {"targets": {"left": {"position_m": [0.0, 0.0, 5.0]}}}),
])
def test_the_workspace_refusal_names_the_argument_the_target_came_from(tmp_path, tool, arguments):
    runtime = _dojo(tmp_path)
    error = _refused(runtime.invoke, tool, arguments)
    assert (error["code"], error["field"]) == ("workspace_bounds", "targets.left.position_m")
    assert error["received"] == [0.0, 0.0, 5.0]


def test_an_undeclared_tool_is_not_retryable_and_lists_the_declared_ones(tmp_path):
    runtime = _dojo(tmp_path)
    error = _refused(runtime.invoke, "bindings", {})
    assert (error["code"], error["field"], error["received"]) == ("tool_unavailable", "tool", "bindings")
    assert error["retryable"] is False
    assert error["constraints"]["allowed"] == list(runtime.allowed_tools)
