"""move_pose checks its target with read-only IK: the jaw turn and reach along the request.

EXP11A g008 stack_bowls (run ``515591023f48``) round 1: every move_pose was
free_space with one absolute quaternion. From home the right arm solved it at
joint6 -2.465 (+2.8 min) and the left at -3.007 (+3.2 min), 0.68 and 0.13 rad
from the +/-3.14 stop; the next translations at the same orientation needed
joint6 +3.088 (right, +5.3 min) and +2.526 (left, +6.4 min), 5.5 rad branch
jumps, and were refused. The joint vectors below are those receipts' IK
selections. On the X5 the parallel-jaw turn shares joints 1-5 and moves joint6
by pi (EXP11A g007 probe records: within 3e-6 rad), so each turned member is the
recorded vector with joint6 shifted by pi. Rows are what the RoboDojo reach
check computes from such vectors (``controllers.reach_check``).

Everything is offline: a stub IK bridge and, for the dispatcher test, the fake
Dojo plant observation with result-schema admission.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from ora.agent.composition import resolve_tool_surface
from ora.agent.composition.move_pose_reach_schema import (
    JAW_CAPABILITY, LINE_FRACTIONS, REACH_CAPABILITY, bind_move_pose_reach,
)
from ora.agent.composition.tool_surface import ToolSurfaceError
from ora.agent.tools import move_pose_reach, reach_reference
from ora.contracts.reach_check import RESULT_SCHEMA
from ora.environment.controllers.reach_check import joint_delta_rad, joint_limit_margin_rad

DOJO = "robodojo_placement_closed_skills_v2"
ROBOTWIN_LIVE = ("robotwin_toolchain_b2_closed_noservo_v2", "robotwin_toolchain_b2_closed_noservo_v1")
X5_LIMITS = [[-9.999999046325684, 9.999999046325684]] * 5 + [[-3.1399998664855957, 3.1399998664855957]]
HOME = [0.0] * 6
BOUNDS = {"x": [-1.0, 1.0], "y": [-1.0, 1.0], "z": [0.0, 2.0]}
#: The round-1 absolute quaternion every move used.
REQUESTED = [-0.428955228, -0.473037167, 0.570079933, -0.51695466]
RIGHT_FROM_HOME = [0.676, 2.179, 1.97, -1.445, -0.192, -2.465]
LEFT_FROM_HOME = [0.136, 1.996, 1.493, -1.238, -0.122, -3.007]
RIGHT_JUMP = [-0.054, 2.263, 1.882, -1.38, -0.088, 3.088]


def _local_y(orientation):
    x, y, z, w = orientation
    return np.array([2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)])


def _turned(joints):
    shifted = joints[5] + math.pi if joints[5] < 0 else joints[5] - math.pi
    return [*joints[:5], shifted]


def _row(joints, reference, *, reason=None):
    if joints is None:
        return {"reachable": False, "reason": reason or "ik_failed", "route": None, "position_error_m": None,
                "orientation_error_rad": None, "joint_limit_margin_rad": None, "joint_delta_rad": None}
    return {"reachable": True, "reason": None, "route": "global", "position_error_m": 0.0001,
            "orientation_error_rad": 0.001,
            "joint_limit_margin_rad": round(joint_limit_margin_rad(joints, X5_LIMITS), 5),
            "joint_delta_rad": round(joint_delta_rad(joints, reference), 5)}


def _bridge(requests, *, requested=REQUESTED, exact=None, turned=None, measured=HOME, line=None):
    """Answer by member: the requested jaw direction is ``exact``, its reverse ``turned``.

    ``line`` answers the reach-along samples (a fraction -> reachable map);
    a sample is recognised by its position between start and target.
    """
    reference_y = _local_y(requested)

    def bridge(request):
        requests.append(copy.deepcopy(request))
        rows = []
        for query in request["queries"]:
            if line is not None and len(request["queries"]) == len(LINE_FRACTIONS):
                index = len(rows)
                rows.append(_row(HOME if line(LINE_FRACTIONS[index]) else None, measured))
                continue
            is_exact = float(np.dot(_local_y(query["pose_xyzw"][3:]), reference_y)) > 0
            joints, reason = (exact if is_exact else turned) or (None, None)
            rows.append(_row(joints, measured, reason=reason))
        return {"schema": RESULT_SCHEMA, "arm": request["arm"],
                "ik_model": "robodojo_sdk_ik_endpoint_selection_v2", "matches_move_pose_admission": True,
                "motion_planner_called": False, "claim_boundary": "stub IK", "results": rows}
    return bridge


def _observation(arm, *, joints=HOME, gripper="open"):
    return {"proprioception": {"end_effectors": {arm: {
        "arm_joint_qpos": list(joints), "gripper_command_target": gripper,
        "pose_xyz_xyzw": [0.0, -0.2, 1.0, 0.0, 0.0, 0.0, 1.0]}}}}


def _host(bridge, profile=DOJO):
    return SimpleNamespace(surface=resolve_tool_surface(profile), align_plan_callback=bridge,
                           workspace_bounds_m=BOUNDS)


def _plan(bridge, arm="right", *, observation=None, interaction="free_space", mode="absolute_quaternion",
          target=(0.32, -0.03, 0.93), current=(0.0, -0.2, 1.0, 0.0, 0.0, 0.0, 1.0)):
    return move_pose_reach.plan_target(
        _host(bridge), arm, current_pose=list(current), target_pose=[*target, *REQUESTED],
        interaction=interaction, orientation_mode=mode, observation=observation or _observation(arm))


# --------------------------------------------------------------------------
# The jaw turn, from the recorded g008 joints
# --------------------------------------------------------------------------

def test_the_g008_right_arm_from_home_submits_the_turned_jaw():
    requests = []
    pose, notes = _plan(_bridge(requests, exact=(RIGHT_FROM_HOME, None), turned=(_turned(RIGHT_FROM_HOME), None)))
    assert notes["jaw_symmetry"]["variant"] == "jaw_flip"
    assert notes["jaw_symmetry"]["reason"] == "larger_joint6_margin"
    margins = notes["jaw_symmetry"]["joint6_margin_rad"]
    assert margins["exact"] == pytest.approx(3.14 - 2.465, abs=1e-4)
    assert margins["jaw_flip"] == pytest.approx(3.14 - (math.pi - 2.465), abs=1e-4)
    # Same position and approach; the jaw axis reverses.
    assert pose[:3] == [0.32, -0.03, 0.93]
    assert float(np.dot(_local_y(pose[3:]), _local_y(REQUESTED))) == pytest.approx(-1)
    assert len(requests) == 1 and len(requests[0]["queries"]) == 2
    assert notes["reach_along_request"] == {"target_reachable": True}


def test_the_g008_left_arm_at_joint6_minus_3_007_is_turned_too():
    _pose, notes = _plan(_bridge([], exact=(LEFT_FROM_HOME, None), turned=(_turned(LEFT_FROM_HOME), None)),
                         arm="left")
    assert (notes["jaw_symmetry"]["variant"], notes["jaw_symmetry"]["reason"]) == ("jaw_flip", "larger_joint6_margin")
    assert notes["jaw_symmetry"]["joint6_margin_rad"]["exact"] == pytest.approx(0.133, abs=1e-3)


def test_a_refused_branch_jump_is_submitted_as_the_turn_that_solves():
    """+5.3 min: the requested turn needed joint6 +3.088 from -2.465; the turned one needs -0.054."""
    _pose, notes = _plan(_bridge([], exact=(None, "ik_branch_jump"), turned=(_turned(RIGHT_JUMP), None),
                                 measured=RIGHT_FROM_HOME),
                         observation=_observation("right", joints=RIGHT_FROM_HOME))
    assert (notes["jaw_symmetry"]["variant"], notes["jaw_symmetry"]["reason"]) == ("jaw_flip", "only_reachable_variant")
    assert notes["jaw_symmetry"]["joint6_margin_rad"]["exact"] is None
    assert notes["jaw_symmetry"]["joint6_margin_rad"]["jaw_flip"] == pytest.approx(3.14 - 0.0536, abs=1e-3)


def test_a_requested_turn_with_enough_margin_is_kept():
    keep = [0.5, 2.0, 1.8, -1.4, -0.1, 1.0]
    pose, notes = _plan(_bridge([], exact=(keep, None), turned=(_turned(keep), None)))
    assert notes["jaw_symmetry"] == {"variant": "exact", "reason": "requested_turn_margin_sufficient",
                                     "joint6_margin_rad": {"exact": pytest.approx(2.14, abs=1e-4),
                                                           "jaw_flip": pytest.approx(3.14 - (math.pi - 1.0),
                                                                                     abs=1e-4)}}
    assert pose[3:] == REQUESTED


def test_a_low_margin_turn_is_kept_when_the_other_is_no_better_or_fails():
    low = [0.5, 2.0, 1.8, -1.4, -0.1, -2.6]
    _pose, notes = _plan(_bridge([], exact=(low, None), turned=None))
    assert (notes["jaw_symmetry"]["variant"], notes["jaw_symmetry"]["reason"]) == ("exact", "requested_turn_kept")
    neither = _plan(_bridge([], exact=None, turned=None))[1]
    assert (neither["jaw_symmetry"]["variant"], neither["jaw_symmetry"]["reason"]) == ("exact", "no_reachable_variant")


@pytest.mark.parametrize("change", [
    {"observation": _observation("right", gripper="close")},
    {"observation": _observation("right", gripper="unknown")},
    {"mode": "hold_current"}, {"mode": "axis_angle_delta"},
    {"interaction": "precision"}, {"interaction": "contact_approach"},
])
def test_a_held_object_or_a_relative_or_local_move_is_never_turned(change):
    requests = []
    pose, notes = _plan(_bridge(requests, exact=(RIGHT_FROM_HOME, None), turned=(_turned(RIGHT_FROM_HOME), None)),
                        **change)
    assert "jaw_symmetry" not in notes
    assert pose[3:] == REQUESTED
    assert len(requests[0]["queries"]) == 1
    assert requests[0]["queries"][0]["interaction"] == change.get("interaction", "free_space")


def test_asking_for_the_half_turn_of_the_current_pose_is_not_turned_back():
    turned = move_pose_reach._member(REQUESTED, "jaw_flip", [1.0, 0.0, 0.0])
    requests = []
    pose, notes = _plan(_bridge(requests, exact=(RIGHT_FROM_HOME, None), turned=(_turned(RIGHT_FROM_HOME), None)),
                        current=(0.32, -0.03, 0.93, *turned))
    assert "jaw_symmetry" not in notes and pose[3:] == REQUESTED


def test_joint6_is_not_claimed_when_the_measured_joints_do_not_isolate_it():
    wound = [7.5, *HOME[1:]]
    _pose, notes = _plan(_bridge([], exact=(RIGHT_FROM_HOME, None), turned=(_turned(RIGHT_FROM_HOME), None),
                                 measured=wound),
                         observation=_observation("right", joints=wound))
    assert notes["jaw_symmetry"]["joint6_margin_rad"] == {"exact": None, "jaw_flip": None}
    assert (notes["jaw_symmetry"]["variant"], notes["jaw_symmetry"]["reason"]) == ("exact", "requested_turn_kept")


# --------------------------------------------------------------------------
# Reach along the request
# --------------------------------------------------------------------------

def test_an_unsolved_target_reports_the_farthest_gap_free_fraction():
    requests = []
    # The samples solve up to 3/8, fail at 1/2 and solve again at 5/8: the
    # reported reach stops at the first gap.
    bridge = _bridge(requests, exact=None, turned=None, line=lambda fraction: fraction != 0.5 and fraction < 0.7)
    _pose, notes = _plan(bridge, target=(0.32, -0.2, 1.0), current=(0.0, -0.2, 1.0, *REQUESTED),
                         observation=_observation("right", gripper="close"))
    reach = notes["reach_along_request"]
    assert reach["target_reachable"] is False and reach["reason"] == "ik_failed"
    assert reach["requested_translation_m"] == pytest.approx(0.32)
    assert reach["requested_rotation_rad"] == pytest.approx(0.0, abs=1e-6)
    assert reach["farthest_reachable_fraction"] == 0.375
    assert reach["farthest_reachable_translation_m"] == pytest.approx(0.12)
    assert reach["farthest_reachable_pose"]["position_m"] == pytest.approx([0.12, -0.2, 1.0])
    assert [row["reachable"] for row in reach["samples"]] == [True, True, True, False, True, False, False]
    assert all(query["interaction"] == "free_space" and query["from_index"] is None
               for query in requests[1]["queries"])
    assert reach["claim_boundary"] == move_pose_reach.REACH_CLAIM


def test_reach_along_interpolates_orientation_with_the_same_interaction():
    requests = []
    bridge = _bridge(requests, exact=None, turned=None, line=lambda fraction: True)
    _pose, notes = _plan(bridge, interaction="precision", target=(0.02, -0.2, 1.0),
                         current=(0.0, -0.2, 1.0, 0.0, 0.0, 0.0, 1.0))
    assert notes["reach_along_request"]["farthest_reachable_fraction"] == 0.875
    halfway = requests[1]["queries"][3]
    assert halfway["interaction"] == "precision"
    angle = notes["reach_along_request"]["requested_rotation_rad"]
    reached = 2 * math.acos(min(1.0, abs(float(np.dot(halfway["pose_xyzw"][3:], [0.0, 0.0, 0.0, 1.0])))))
    assert reached == pytest.approx(angle / 2, abs=1e-4)  # the reported angle is rounded to 1e-5


def test_no_sample_solving_reports_fraction_zero():
    _pose, notes = _plan(_bridge([], exact=None, turned=None, line=lambda fraction: False),
                         observation=_observation("right", gripper="close"))
    assert notes["reach_along_request"]["farthest_reachable_fraction"] == 0.0
    assert notes["reach_along_request"]["farthest_reachable_pose"] is None


def test_an_unavailable_ik_reference_never_blocks_the_move():
    def broken(request):
        raise RuntimeError("worker died")

    pose, notes = _plan(broken)
    assert pose[3:] == REQUESTED
    assert notes == {"jaw_symmetry": {"variant": "exact", "reason": "ik_reference_unavailable",
                                      "joint6_margin_rad": {"exact": None, "jaw_flip": None}},
                     "reach_along_request": {"status": "unavailable", "failure": "RuntimeError"}}


# --------------------------------------------------------------------------
# Surfaces, schema and the real dispatcher
# --------------------------------------------------------------------------

def test_the_live_dojo_surface_declares_both_and_robotwin_neither():
    dojo = resolve_tool_surface(DOJO)
    assert dojo.supports(JAW_CAPABILITY) and dojo.supports(REACH_CAPABILITY)
    target = dojo.result_schemas["move_pose"]["properties"]["compiled_pose"]["properties"]["targets"]
    # observed_clearance is observed_clearance_reference_v1's (test_observed_clearance_runtime.py) and
    # target_axes_world is pose_orientation_from_world_axes_v1's (test_pose_world_axes.py).
    assert set(target["additionalProperties"]["properties"]) == {
        "jaw_symmetry", "reach_along_request", "observed_clearance", "target_axes_world"}
    for profile in ROBOTWIN_LIVE:
        surface = resolve_tool_surface(profile)
        assert not surface.supports(JAW_CAPABILITY) and not surface.supports(REACH_CAPABILITY)
        # RoboTwin reaches move_pose_reach only through its own reach reference
        # (move_pose_reach_along_request_v2, test_reach_reference_v2.py).
        assert move_pose_reach.declared(surface) is reach_reference.move_pose_declared(surface)


def test_the_binder_refuses_a_surface_it_cannot_describe():
    results = {"move_pose": {"type": "object", "properties": {"compiled_pose": {"type": "object"}}}}
    with pytest.raises(ToolSurfaceError, match="enhanced_move_pose_v1"):
        bind_move_pose_reach(copy.deepcopy(results), [REACH_CAPABILITY], {})
    with pytest.raises(ToolSurfaceError, match="ARX X5"):
        bind_move_pose_reach(copy.deepcopy(results), [JAW_CAPABILITY, "enhanced_move_pose_v1"],
                             {"controlled_frame_convention": "aloha_grip_center"})
    with pytest.raises(ToolSurfaceError, match="compiled_pose"):
        bind_move_pose_reach({"move_pose": {"type": "object"}}, [REACH_CAPABILITY, "enhanced_move_pose_v1"], {})


def _dojo_runtime(tmp_path):
    import test_robodojo_b2_parity as parity

    runtime = parity.tool_runtime(tmp_path)
    runtime.surface = resolve_tool_surface(DOJO)
    runtime.allowed_tools = tuple(item["name"] for item in runtime.surface.definitions)
    pointer, source, observation = runtime._current()
    # A submission binds the observation generation and digest the runtime publishes.
    pointer = {**pointer, "generation": 3, "observation_sha256": hashlib.sha256(
        json.dumps(observation, sort_keys=True).encode()).hexdigest()}
    runtime._current = lambda: (pointer, source, observation)
    return runtime


def test_the_dispatcher_submits_the_turn_it_reports(tmp_path):
    runtime = _dojo_runtime(tmp_path)
    requests = []
    runtime.align_plan_callback = _bridge(requests, exact=(RIGHT_FROM_HOME, None),
                                          turned=(_turned(RIGHT_FROM_HOME), None))
    result = runtime.invoke("move_pose", {"targets": {"left": {
        "interaction": "free_space", "position_mode": "absolute", "position_m": [0.05, -0.1, 0.99],
        "orientation_mode": "absolute_quaternion", "orientation_xyzw": REQUESTED}}})
    target = result["compiled_pose"]["targets"]["left"]
    assert target["jaw_symmetry"]["variant"] == "jaw_flip"
    assert target["reach_along_request"] == {"target_reachable": True}
    submitted = json.loads((runtime.control_root / "submitted-action.json").read_text())
    orientation = submitted["arguments"]["targets"]["left"]["orientation_xyzw"]
    assert abs(float(np.dot(orientation, target["target_pose_xyz_xyzw"][3:]))) == pytest.approx(1)
    assert float(np.dot(_local_y(orientation), _local_y(REQUESTED))) == pytest.approx(-1)
    assert submitted["arguments"]["targets"]["left"]["position_m"] == pytest.approx([0.05, -0.1, 0.99])
    assert target["position_mode"] == "absolute" and target["orientation_mode"] == "absolute_quaternion"


def test_a_surface_without_the_capabilities_submits_as_released(tmp_path):
    runtime = _dojo_runtime(tmp_path)
    runtime.surface = replace(runtime.surface, capabilities=tuple(
        item for item in runtime.surface.capabilities if item not in {JAW_CAPABILITY, REACH_CAPABILITY}))
    requests = []
    runtime.align_plan_callback = _bridge(requests, exact=(RIGHT_FROM_HOME, None))
    result = runtime.invoke("move_pose", {"targets": {"left": {
        "interaction": "free_space", "position_mode": "absolute", "position_m": [0.05, -0.1, 0.99],
        "orientation_mode": "absolute_quaternion", "orientation_xyzw": REQUESTED}}})
    assert requests == []
    assert "jaw_symmetry" not in result["compiled_pose"]["targets"]["left"]
