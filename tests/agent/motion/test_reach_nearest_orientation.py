"""An unsolved move target carries the nearest reachable turn of its orientation.

EXP11A g012 (E11G12-003): an offline replica of the runtime IK (99.8-100%
agreement with 1,066 recorded solves) found every placement target reachable by
some arm, many only with a 15-75 degree tilt, while the policy tried
straight-down poses and every feedback channel answered only the requested
orientation. Under reach_nearest_orientation_v1 the requested orientation is
turned about axes perpendicular to its approach, holding the nominal jaw
centre, one read-only reach-check request per rotation, smallest first.

Offline: stub IK bridges, the recorded g012 drawer-top target and the fake Dojo
plant observation with result-schema admission.
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from ora.agent.composition import resolve_tool_surface
from ora.agent.composition.reach_orientation_schema import (
    CAPABILITY, CLAIM, MAXIMUM_QUERIES, REQUIRES, ROTATIONS_DEG, TOOLS, TURN_DIRECTIONS,
    bind_reach_nearest_orientation,
)
from ora.agent.composition.tool_surface import ToolSurfaceError
from ora.agent.tools import move_pose_reach, reach_orientation
from ora.contracts.pose import quaternion_matrix
from ora.contracts.reach_check import MAXIMUM_POSES, RESULT_SCHEMA
from tests.agent.motion.test_reach_check import dojo  # noqa: F401  (the RoboDojo reach-check fixture)

DOJO = "robodojo_placement_closed_skills_v2"
ROBOTWIN_LIVE = ("robotwin_toolchain_b2_closed_noservo_v2",)
BOUNDS = {"x": [-1.0, 1.0], "y": [-1.0, 1.0], "z": [0.0, 2.0]}
APPROACH = np.array([1.0, 0.0, 0.0])
OFFSET = np.array([0.1501, 0.0, 0.0])
#: Straight down (link6 +x to world -z), jaw line along world y.
DOWN = [0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)]
#: EXP11A g012 organize_table round 0 (run 1e6b0c549a13, action-4a809f7b8b1f61be):
#: the left arm's straight-down link6 target over the drawer-unit top, refused
#: no_valid_endpoint. The offline replica's smallest reachable tilt at the same
#: jaw centre (-0.42, 0.10, 1.02) is 45 degrees.
DRAWER_TOP = [-0.42, 0.1, 1.17, 0.70446, 0.061911, -0.704252, 0.06271]


def _approach(pose):
    return np.asarray(quaternion_matrix(pose[3:])) @ APPROACH


def _tilt(pose):
    return math.degrees(math.acos(max(-1.0, min(1.0, -_approach(pose)[2]))))


def _jaw_center(pose):
    return np.asarray(pose[:3]) + np.asarray(quaternion_matrix(pose[3:])) @ OFFSET


#: The Dojo reach check's model since the EXP11A g013 kernel unit (E11G12-004a):
#: its path screen also holds the other arm at its measured joints.
IK_MODEL = "robodojo_sdk_ik_endpoint_selection_v2"


def _row(ok, margin=1.0, reason="ik_failed"):
    return {"reachable": ok, "reason": None if ok else reason, "route": "global" if ok else None,
            "position_error_m": 0.0001 if ok else None, "orientation_error_rad": 0.001 if ok else None,
            "joint_limit_margin_rad": margin if ok else None, "joint_delta_rad": 0.3 if ok else None}


def _bridge(requests, reachable, margin=lambda pose: 1.0, refused_reason=lambda pose: "ik_failed"):
    def bridge(request):
        requests.append(copy.deepcopy(request))
        poses = [query["pose_xyzw"] for query in request["queries"]]
        rows = [_row(bool(reachable(pose)), margin(pose), refused_reason(pose)) for pose in poses]
        return {"schema": RESULT_SCHEMA, "arm": request["arm"], "ik_model": IK_MODEL,
                "matches_move_pose_admission": True, "motion_planner_called": False, "claim_boundary": "stub IK",
                "results": rows}
    return bridge


def _nearest(bridge, target=(0.3, -0.1, 1.0, *DOWN), interaction="free_space"):
    return reach_orientation.nearest_orientation(
        "left", target_pose=list(target), interaction=interaction, approach_local=APPROACH, offset_local=OFFSET,
        bridge=bridge, workspace_bounds_m=BOUNDS)


# --------------------------------------------------------------------------
# The candidates
# --------------------------------------------------------------------------

@pytest.mark.parametrize("angle", ROTATIONS_DEG)
def test_each_turn_tilts_the_approach_by_its_angle_and_holds_the_jaw_center(angle):
    target = [0.3, -0.1, 1.0, *DOWN]
    candidates, center = reach_orientation.turned_candidates(
        target, approach_local=APPROACH, offset_local=OFFSET, rotation_deg=angle)
    assert len(candidates) == TURN_DIRECTIONS <= MAXIMUM_POSES
    assert center == pytest.approx(_jaw_center(target))
    requested = np.asarray(quaternion_matrix(DOWN))
    for candidate in candidates:
        pose = candidate["pose_xyzw"]
        assert _jaw_center(pose) == pytest.approx(center, abs=1e-9)
        assert _tilt(pose) == pytest.approx(angle, abs=1e-6)
        turn = np.asarray(quaternion_matrix(pose[3:])) @ requested.T
        assert math.degrees(math.acos((np.trace(turn) - 1) / 2)) == pytest.approx(angle, abs=1e-6)
        # The turn axis is horizontal for a straight-down request: the jaw keeps its roll.
        axis = np.array([turn[2, 1] - turn[1, 2], turn[0, 2] - turn[2, 0], turn[1, 0] - turn[0, 1]])
        assert abs(axis[2]) < 1e-9


def test_the_turn_axes_are_even_and_perpendicular_to_any_approach():
    approach = np.array([0.6, 0.0, -0.8])
    axes = reach_orientation.turn_axes(approach)
    assert len(axes) == TURN_DIRECTIONS and abs(axes[0][2]) < 1e-12
    for left, right in zip(axes, axes[1:] + axes[:1]):
        assert float(np.dot(left, approach)) == pytest.approx(0.0, abs=1e-12)
        assert math.degrees(math.acos(float(np.dot(left, right)))) == pytest.approx(360 / TURN_DIRECTIONS)


# --------------------------------------------------------------------------
# The sweep
# --------------------------------------------------------------------------

def test_the_g012_drawer_top_target_is_answered_at_45_degrees():
    """A stub that solves link6 up to z 1.13 m: the straight-down target needs z 1.17."""
    requests = []
    nearest = _nearest(_bridge(requests, lambda pose: pose[2] <= 1.13), target=DRAWER_TOP)
    assert nearest["status"] == "found" and nearest["rotation_from_request_deg"] == 45
    assert nearest["tilt_from_straight_down_deg"] == pytest.approx(45.0, abs=0.2)
    assert nearest["jaw_center_m"] == pytest.approx([-0.42, 0.1, 1.02], abs=2e-3)
    assert nearest["pose"]["position_m"][2] == pytest.approx(1.02 + 0.1501 * math.cos(math.radians(45)), abs=3e-3)
    assert _jaw_center([*nearest["pose"]["position_m"], *nearest["pose"]["orientation_xyzw"]]) == pytest.approx(
        nearest["jaw_center_m"], abs=1e-4)
    assert nearest["queries"] == 3 * TURN_DIRECTIONS and nearest["claim_boundary"] == CLAIM
    assert nearest["reachable_turns_at_rotation"] == TURN_DIRECTIONS
    # One request per rotation, every query on the target's route from the measured joints.
    assert [len(request["queries"]) for request in requests] == [TURN_DIRECTIONS] * 3
    assert all(query["interaction"] == "free_space" and query["from_index"] is None
               for request in requests for query in request["queries"])


def test_the_smallest_rotation_wins_and_then_the_larger_joint_margin():
    requests = []
    tilted_toward_plus_y = lambda pose: _approach(pose)[1] > 0.2  # noqa: E731
    nearest = _nearest(_bridge(requests, tilted_toward_plus_y, margin=lambda pose: float(_approach(pose)[0] + 2)))
    assert nearest["rotation_from_request_deg"] == 15 and len(requests) == 1
    best = max((c for c in reach_orientation.turned_candidates(
        [0.3, -0.1, 1.0, *DOWN], approach_local=APPROACH, offset_local=OFFSET, rotation_deg=15)[0]
        if tilted_toward_plus_y(c["pose_xyzw"])), key=lambda c: c["approach_world"][0])
    assert nearest["approach_world"] == pytest.approx(best["approach_world"], abs=1e-4)
    assert 1 <= nearest["reachable_turns_at_rotation"] < TURN_DIRECTIONS


def test_nothing_within_the_sweep_is_said_so_after_every_rotation():
    requests = []
    nearest = _nearest(_bridge(requests, lambda pose: False), interaction="precision")
    assert nearest == {"status": "none_within_sweep", "largest_rotation_deg": ROTATIONS_DEG[-1],
                       "jaw_center_m": pytest.approx(list(_jaw_center([0.3, -0.1, 1.0, *DOWN])), abs=1e-5),
                       "queries": MAXIMUM_QUERIES, "claim_boundary": CLAIM}
    assert len(requests) == len(ROTATIONS_DEG)
    assert {query["interaction"] for request in requests for query in request["queries"]} == {"precision"}


def test_a_turn_refused_only_by_the_other_arm_is_not_reported_reachable():
    """v2 reach check: IK solves every turn, the other arm blocks those tilted toward -y."""
    requests = []
    toward_minus_y = lambda pose: _approach(pose)[1] < -0.05  # noqa: E731
    nearest = _nearest(_bridge(requests, lambda pose: not toward_minus_y(pose),
                               refused_reason=lambda pose: "inter_arm_collision_path"))
    assert nearest["status"] == "found" and nearest["rotation_from_request_deg"] == 15
    assert nearest["approach_world"][1] >= -0.05
    candidates = reach_orientation.turned_candidates(
        [0.3, -0.1, 1.0, *DOWN], approach_local=APPROACH, offset_local=OFFSET, rotation_deg=15)[0]
    blocked = sum(toward_minus_y(item["pose_xyzw"]) for item in candidates)
    assert 0 < blocked < TURN_DIRECTIONS
    assert nearest["reachable_turns_at_rotation"] == TURN_DIRECTIONS - blocked
    # Every turn the other arm blocks, at every rotation: nothing is reachable.
    requests.clear()
    nearest = _nearest(_bridge(requests, lambda pose: False, refused_reason=lambda pose: "inter_arm_collision_path"))
    assert nearest["status"] == "none_within_sweep" and nearest["queries"] == MAXIMUM_QUERIES


def test_the_v2_dojo_reach_check_drives_the_sweep_and_its_inter_arm_refusals_end_it_empty(dojo, monkeypatch):
    """Through the RoboDojo backend's own reach check (ik_model v2), nothing written."""
    from ora.contracts.reach_check import ROBODOJO_REASONS
    from ora.environment.backends.robodojo import ik

    assert "inter_arm_collision_path" in ROBODOJO_REASONS
    answers = []

    def bridge(request):
        answers.append(dojo.align_plan(request))
        return answers[-1]

    target = [0.02, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0]
    clear = _nearest(bridge, target=target)
    assert clear["status"] == "found" and clear["rotation_from_request_deg"] == ROTATIONS_DEG[0]
    assert {answer["ik_model"] for answer in answers} == {IK_MODEL}
    answers.clear()
    monkeypatch.setattr(ik, "inter_arm_screen", lambda session: SimpleNamespace(
        straight_paths=lambda starts, ends: {"status": "inter_arm_collision", "min_clearance_m": -0.01}))
    blocked = _nearest(bridge, target=target)
    assert blocked["status"] == "none_within_sweep" and blocked["queries"] == MAXIMUM_QUERIES
    assert {row["reason"] for answer in answers for row in answer["results"]} == {"inter_arm_collision_path"}
    assert dojo.environment.commands == []


# --------------------------------------------------------------------------
# Through move_pose and the primitives
# --------------------------------------------------------------------------

def _host(bridge, profile=DOJO, *, declared=True):
    surface = resolve_tool_surface(profile)
    if not declared:
        surface = replace(surface, capabilities=tuple(c for c in surface.capabilities if c != CAPABILITY))
    return SimpleNamespace(surface=surface, align_plan_callback=bridge, workspace_bounds_m=BOUNDS)


def _plan(host, target):
    observation = {"proprioception": {"end_effectors": {"left": {
        "arm_joint_qpos": [0.0] * 6, "gripper_command_target": "close",
        "pose_xyz_xyzw": [-0.3, -0.35, 0.92, *DOWN]}}}}
    return move_pose_reach.plan_target(
        host, "left", current_pose=[-0.3, -0.35, 0.92, *DOWN], target_pose=target, interaction="free_space",
        orientation_mode="absolute_quaternion", observation=observation)


def test_an_unsolved_move_pose_target_carries_it_and_a_solved_one_does_not():
    requests = []
    pose, notes = _plan(_host(_bridge(requests, lambda pose: pose[2] <= 1.13)), DRAWER_TOP)
    assert pose == DRAWER_TOP  # the submitted action never changes
    reach = notes["reach_along_request"]
    assert reach["target_reachable"] is False and reach["farthest_reachable_fraction"] == 0.75
    assert reach["nearest_reachable_orientation"]["rotation_from_request_deg"] == 45
    requests.clear()
    _pose, solved = _plan(_host(_bridge(requests, lambda pose: True)), DRAWER_TOP)
    assert solved["reach_along_request"] == {"target_reachable": True} and len(requests) == 1


def test_an_unavailable_sweep_keeps_the_rest_of_the_reference():
    calls = []

    def bridge(request):
        calls.append(request)
        if len(calls) > 2:
            raise RuntimeError("worker died")
        return _bridge([], lambda pose: False)(request)

    _pose, notes = _plan(_host(bridge), DRAWER_TOP)
    reach = notes["reach_along_request"]
    assert reach["farthest_reachable_fraction"] == 0.0
    assert reach["nearest_reachable_orientation"] == {"status": "unavailable", "failure": "RuntimeError"}


def test_a_surface_without_the_capability_reports_as_released():
    requests = []
    _pose, notes = _plan(_host(_bridge(requests, lambda pose: False), declared=False), DRAWER_TOP)
    assert "nearest_reachable_orientation" not in notes["reach_along_request"] and len(requests) == 2


def test_the_dispatcher_admits_it_for_move_pose_and_move_to(tmp_path):
    from tests.agent.motion.test_move_pose_reach import _dojo_runtime

    runtime = _dojo_runtime(tmp_path)
    runtime.align_plan_callback = _bridge([], lambda pose: _tilt(pose) >= 29.0)
    result = runtime.invoke("move_pose", {"targets": {"left": {
        "interaction": "free_space", "position_mode": "absolute", "position_m": [0.05, -0.1, 0.99],
        "orientation_mode": "absolute_quaternion", "orientation_xyzw": DOWN}}})
    nearest = result["compiled_pose"]["targets"]["left"]["reach_along_request"]["nearest_reachable_orientation"]
    assert nearest["status"] == "found" and nearest["rotation_from_request_deg"] == 30
    submitted = json.loads((runtime.control_root / "submitted-action.json").read_text())
    assert submitted["arguments"]["targets"]["left"]["orientation_xyzw"] == pytest.approx(DOWN)
    (tmp_path / "primitive").mkdir()
    runtime = _dojo_runtime(tmp_path / "primitive")
    runtime.align_plan_callback = _bridge([], lambda pose: False)
    moved = runtime.invoke("move_to", {"targets": {"left": {"position_m": [0.3, -0.1, 0.9]}}})
    assert moved["compiled_pose"]["targets"]["left"]["reach_along_request"][
        "nearest_reachable_orientation"]["status"] == "none_within_sweep"


# --------------------------------------------------------------------------
# Surfaces and schema
# --------------------------------------------------------------------------

def test_the_live_dojo_surface_declares_it_on_every_reach_reference_and_robotwin_does_not():
    dojo = resolve_tool_surface(DOJO)
    assert dojo.supports(CAPABILITY)
    for name in TOOLS:
        reach = dojo.result_schemas[name]["properties"]["compiled_pose"]["properties"]["targets"][
            "additionalProperties"]["properties"]["reach_along_request"]
        unsolved = next(item for item in reach["oneOf"]
                        if item["properties"].get("target_reachable") == {"const": False})
        assert "nearest_reachable_orientation" in unsolved["required"], name
    description = next(item for item in dojo.definitions if item["name"] == "move_pose")["description"]
    assert "nearest_reachable_orientation" in description
    for profile in ROBOTWIN_LIVE:
        assert not resolve_tool_surface(profile).supports(CAPABILITY)


def test_the_binder_refuses_a_surface_it_cannot_describe():
    dojo = resolve_tool_surface(DOJO)
    view = dojo.view_config
    with pytest.raises(ToolSurfaceError, match=REQUIRES):
        bind_reach_nearest_orientation({}, [CAPABILITY], view)
    with pytest.raises(ToolSurfaceError, match="geometry descriptor"):
        bind_reach_nearest_orientation({}, [CAPABILITY, REQUIRES], {})
    with pytest.raises(ToolSurfaceError, match="move_pose reach_along_request"):
        bind_reach_nearest_orientation({"move_pose": {"type": "object"}}, [CAPABILITY, REQUIRES], view)
    bound = copy.deepcopy(dict(dojo.result_schemas))
    with pytest.raises(ToolSurfaceError, match="overwrite"):
        bind_reach_nearest_orientation(bound, [CAPABILITY, REQUIRES], view)
