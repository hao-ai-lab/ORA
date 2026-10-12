"""Candidate-level RoboDojo IK continuity without SDK, GPU or simulator startup."""
from __future__ import annotations

import math
import re
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from ora.environment.backends.robodojo import ik

TARGET = [0.2, 0.1, 0.4, 0.0, 0.0, 0.0, 1.0]


def candidates(values, *, success=None, feasible=None):
    count, joints = len(values), len(values[0])
    return {"solutions": values, "joint_names": [f"joint_{i}" for i in range(joints)],
            "success": [True] * count if success is None else success,
            "feasible": [True] * count if feasible is None else feasible,
            "position_tolerance_m": 0.01, "orientation_tolerance_rad": 0.02}


def select(value, *, current=None, limits=None, fk=None, bound=1.0, path_clearance=None, canonical=None):
    joints = len(value["joint_names"])
    extra = {} if canonical is None else {"canonical_jump_limit_rad": canonical}
    return ik.select_candidate_set(candidates=value, current=[0.0] * joints if current is None else current,
        limits=[[-10.0, 10.0]] * joints if limits is None else limits,
        target=TARGET, fk=(lambda _: TARGET) if fk is None else fk, jump_limit_rad=bound,
        path_clearance=path_clearance, **extra)


def test_each_seed_must_be_successful_and_feasible_before_fk():
    seen = []
    result = select(candidates([[0.01], [0.02], [0.3]],
                              success=[False, True, True], feasible=[True, False, True]),
                    fk=lambda q: seen.append(q) or TARGET)
    assert result["joint_target"] == [0.3]
    assert seen == [[0.3]]
    assert [row["rejection"] for row in result["candidates"]] == [
        "sdk_not_feasible_or_converged", "sdk_not_feasible_or_converged", None]


def test_invalid_joint_endpoints_do_not_reach_fk():
    seen = []
    result = select(candidates([[float("nan")], [float("inf")], [1.1], [-1.1], [0.5]]),
                    limits=[[-1, 1]], fk=lambda q: seen.append(q) or TARGET)
    assert seen == [[0.5]]
    assert result["ik_selection"]["valid_endpoint_count"] == 1
    assert [r["rejection"] for r in result["candidates"]] == [
        "nonfinite_joint_target", "nonfinite_joint_target", "joint_limits", "joint_limits", None]
    assert result["candidates"][0]["joint_target"] is None


@pytest.mark.parametrize("measured", [
    [0.22, 0.1, 0.4, 0, 0, 0, 1],
    [0.2, 0.1, 0.4, 0, 0, math.sin(0.1), math.cos(0.1)],
    [float("nan"), 0.1, 0.4, 0, 0, 0, 1],
])
def test_sdk_success_requires_finite_fk_residual_inside_both_tolerances(measured):
    result = select(candidates([[0.1], [0.2]]), fk=lambda q: measured if q[0] == 0.1 else TARGET)
    assert result["joint_target"] == [0.2]
    assert result["candidates"][0]["rejection"] == "fk_residual_exceeded"


def test_fk_exception_cannot_produce_an_admitted_target():
    def broken_fk(_):
        raise ik.RobodojoError("kinematics unavailable")
    with pytest.raises(ik.RobodojoError, match="kinematics unavailable"):
        select(candidates([[0.1]]), fk=broken_fk)


def test_jump_filter_precedes_l2_ranking():
    # The first row has smaller L2 distance, but violates one joint's bound.
    result = select(candidates([[1.1, 0.0], [0.8, 0.8]]))
    assert result["joint_target"] == [0.8, 0.8]
    assert result["ik_selection"]["selected_candidate_index"] == 1
    assert result["ik_selection"]["within_jump_limit_count"] == 1
    assert result["candidates"][0]["rejection"] == "joint_jump"


def test_joint_interpolation_distance_does_not_wrap_at_two_pi():
    result = select(candidates([[-3.13], [2.8]]), current=[3.13])
    assert result["joint_target"] == [2.8]
    assert result["candidates"][0]["joint_delta_rad"] == pytest.approx(6.26)
    assert result["candidates"][0]["rejection"] == "joint_jump"


def test_all_discontinuous_keeps_nearest_valid_endpoint_for_refusal_evidence():
    result = select(candidates([[2.0, 0.0], [0.0, -1.5]]))
    assert result["joint_target"] == [0.0, -1.5]
    summary = result["ik_selection"]
    assert summary["outcome"] == "joint_jump"
    assert summary["valid_endpoint_count"] == 2 and summary["within_jump_limit_count"] == 0
    assert summary["selected_candidate_index"] == 1
    assert summary["limiting_joint"] == "joint_1"
    assert summary["limiting_joint_delta_rad"] == -1.5
    assert "not guaranteed" in summary["recovery"]
    assert result["candidates"][1]["rejection"] == "joint_jump"


def test_no_valid_endpoint_does_not_invent_a_joint_target():
    result = select(candidates([[0.1]], success=[False]))
    assert result["ik_status"] == "Fail" and result["joint_target"] is None
    assert result["ik_selection"]["outcome"] == "no_valid_endpoint"
    assert result["ik_selection"]["selected_candidate_index"] is None
    assert "not proof" in result["ik_selection"]["recovery"]


@pytest.mark.parametrize("seeds, returned", [(1, 1), (3, 3), (64, 32)])
def test_sdk_extracts_active_named_order_and_bounds_returned_seeds_without_reconfiguration(monkeypatch, seeds, returned):
    calls = []
    transformer = ModuleType("utils.transformer")
    transformer.calculate_target_pose = lambda base, origin, pose: calls.append((base, origin, pose)) or pose
    monkeypatch.setitem(sys.modules, "utils.transformer", transformer)
    config = SimpleNamespace(num_seeds=seeds, position_tolerance=0.005, orientation_tolerance=0.04)
    original_config = vars(config).copy()
    values = np.tile([[[8.0, 0.2, 0.1]]], (1, returned, 1))
    result = SimpleNamespace(js_solution=SimpleNamespace(joint_names=["finger", "joint_b", "joint_a"], position=values),
                             success=np.ones((1, returned), dtype=bool), feasible=np.ones((1, returned), dtype=bool))
    result.success[0, -1] = False
    kwargs_seen = []
    def solve_pose(**kwargs):
        kwargs_seen.append(kwargs)
        return result
    planner = SimpleNamespace(robot_origin_pose=[0, 0, 0, 1, 0, 0, 0], frame_bias=[0.1, 0.2, 0.3],
        active_joints_name=["joint_a", "joint_b"],
        ik_solver=SimpleNamespace(config=config, solve_pose=solve_pose),
        _trans_from_world_to_base=lambda origin, target: (target[:3], target[3:]),
        _build_goal_pose=lambda position, quaternion: (position.tolist(), quaternion.tolist()),
        _build_joint_state=lambda current: tuple(current))
    selected = ik.sdk_candidates(planner, [0.0, 0.0], TARGET, [1, 2, 3, 1, 0, 0, 0])
    np.testing.assert_allclose(selected["solutions"], np.tile([[0.1, 0.2]], (returned, 1)))
    assert selected["joint_names"] == ["joint_a", "joint_b"]
    assert selected["revolute"] == [False, False]
    assert selected["success"].tolist() == [True] * (returned - 1) + [False]
    assert vars(config) == original_config
    assert len(kwargs_seen) == 1 and kwargs_seen[0]["return_seeds"] == returned
    assert kwargs_seen[0]["current_state"] == (0.0, 0.0)
    assert calls[0][2] == [0.2, 0.1, 0.4, 1.0, 0.0, 0.0, 0.0]
    np.testing.assert_allclose(kwargs_seen[0]["goal_tool_poses"][0], [0.3, 0.3, 0.7])


def test_solve_ik_uses_explicit_local_bound_or_session_global_bound(monkeypatch):
    from ora.environment.backends.robodojo import reachability
    robot = SimpleNamespace(robot_name="left", arm_joints_name=["joint_0"], entity_origin_pose=TARGET)
    manager = SimpleNamespace(get_joint=lambda *a, **k: [[0.0]], ik_solver={"left": object()})
    session = SimpleNamespace(environment=SimpleNamespace(robot_manager=manager), robots={"left": robot},
                              global_joint_jump_limit_rad=1.0)
    monkeypatch.setattr(ik, "sdk_candidates", lambda *a: candidates([[0.6]]))
    monkeypatch.setattr(ik, "articulation_limits", lambda *a: [[-2.0, 2.0]])
    monkeypatch.setattr(reachability, "fk_link6", lambda *a: TARGET)
    monkeypatch.setattr(ik, "self_collision_model", lambda *a: SimpleNamespace(
        straight_path=lambda start, end: {"status": "clear"}))
    assert ik.solve_ik(session, "left", TARGET)["ik_selection"]["outcome"] == "admitted"
    local = ik.solve_ik(session, "left", TARGET, jump_limit_rad=0.25)
    assert local["ik_selection"]["outcome"] == "joint_jump"
    assert local["ik_selection"]["joint_jump_limit_rad"] == 0.25


def test_wide_revolute_alias_commands_rechecked_equivalent_target_and_keeps_raw_evidence():
    raw = 2 * math.pi + 0.2
    value = candidates([[raw]])
    value["revolute"] = [True]
    seen = []
    result = select(value, limits=[[-10, 10]], fk=lambda q: seen.append(q) or TARGET)
    assert result["ik_selection"]["outcome"] == "admitted"
    assert result["joint_target"] == pytest.approx([0.2])
    assert seen and seen[-1] == pytest.approx([0.2])
    row = result["candidates"][0]
    assert row["solver_joint_target"] == [raw]
    assert row["equivalent_turns"] == [-1]
    assert row["joint_target"] == pytest.approx([0.2])
    assert result["joint_delta_rad"] == pytest.approx(0.2)


def test_sdk_marks_only_source_bound_arx_arm_joints_as_revolute(monkeypatch):
    from ora.environment.backends.robodojo.ik import ARX_X5_REVOLUTE_JOINT_NAMES
    names = list(ARX_X5_REVOLUTE_JOINT_NAMES)
    assert len(names) == 6
    active = [*names, "unknown_slide"]
    transformer = ModuleType("utils.transformer")
    transformer.calculate_target_pose = lambda base, origin, pose: pose
    monkeypatch.setitem(sys.modules, "utils.transformer", transformer)
    result = SimpleNamespace(
        js_solution=SimpleNamespace(joint_names=active, position=np.zeros((1, 1, 7))),
        success=np.ones((1, 1), dtype=bool), feasible=np.ones((1, 1), dtype=bool))
    planner = SimpleNamespace(robot_origin_pose=[0, 0, 0, 1, 0, 0, 0], frame_bias=[0, 0, 0],
        active_joints_name=active,
        ik_solver=SimpleNamespace(config=SimpleNamespace(num_seeds=1, position_tolerance=0.005,
            orientation_tolerance=0.04), solve_pose=lambda **kw: result),
        _trans_from_world_to_base=lambda origin, target: (target[:3], target[3:]),
        _build_goal_pose=lambda position, quaternion: (position, quaternion),
        _build_joint_state=lambda current: current)
    selected = ik.sdk_candidates(planner, [0] * 7, TARGET, planner.robot_origin_pose)
    assert selected["revolute"] == [True] * 6 + [False]
    # Review U1 decision: joint2-joint5 may not pass +/-pi; joint1 and joint6 may.
    assert selected["pitch"] == [False, True, True, True, True, False, False]


def test_finite_joint6_cannot_cross_its_actual_limit_via_alias():
    value = candidates([[0, 0, 0, 0, 0, 3.13]])
    value["revolute"] = [True] * 6
    result = select(value, current=[0, 0, 0, 0, 0, -3.13],
                    limits=[[-10, 10]] * 5 + [[-3.14, 3.14]])
    assert result["ik_selection"]["outcome"] == "joint_jump"
    assert result["joint_target"][-1] == 3.13
    assert result["joint_delta_rad"] == pytest.approx(6.26)
    assert result["candidates"][0]["equivalent_turns"][-1] == 0


def test_equivalent_configuration_must_pass_fk_again():
    value = candidates([[2 * math.pi + 0.2]])
    value["revolute"] = [True]
    seen = []
    def fk(q):
        seen.append(q)
        # Raw solver coordinates match, but the proposed alias does not.
        return TARGET if q[0] > 6 else [0.4, *TARGET[1:]]
    result = select(value, limits=[[-10, 10]], fk=fk)
    assert any(abs(q[0] - 0.2) < 1e-10 for q in seen)
    assert result["ik_selection"]["outcome"] == "no_valid_endpoint"
    assert result["joint_target"] is None
    assert result["candidates"][0]["rejection"] == "fk_residual_exceeded"


def test_mixed_joint_types_leave_prismatic_coordinates_unchanged():
    raw = [2 * math.pi + 0.1, 2 * math.pi + 0.1, -2 * math.pi + 0.2]
    value = candidates([raw])
    value["revolute"] = [True, False, True]
    result = select(value, limits=[[-10, 10]] * 3)
    assert result["ik_selection"]["outcome"] == "joint_jump"
    assert result["joint_target"] == pytest.approx([0.1, raw[1], 0.2])
    assert result["candidates"][0]["equivalent_turns"] == [-1, 0, 1]
    assert result["candidates"][0]["solver_joint_target"] == raw


@pytest.mark.parametrize("mask", [None, [False]])
def test_untyped_or_nonrevolute_joint_does_not_gain_modulo_admission(mask):
    raw = 2 * math.pi + 0.1
    value = candidates([[raw]])
    if mask is not None:
        value["revolute"] = mask
    result = select(value)
    assert result["joint_target"] == [raw]
    assert result["ik_selection"]["outcome"] == "joint_jump"
    assert result["candidates"][0]["equivalent_turns"] == [0]


def test_original_solver_outside_actual_bounds_is_rejected_before_alias_or_fk():
    value = candidates([[2 * math.pi + 0.1]])
    value["revolute"] = [True]
    seen = []
    result = select(value, limits=[[-1, 1]], fk=lambda q: seen.append(q) or TARGET)
    assert seen == []
    assert result["joint_target"] is None
    assert result["candidates"][0]["rejection"] == "joint_limits"


def test_alias_mapping_and_per_joint_admission_precede_candidate_distance_selection():
    value = candidates([[2 * math.pi + 1.1, 0], [2 * math.pi + 0.8, 0.8]])
    value["revolute"] = [True, True]
    result = select(value, limits=[[-10, 10]] * 2)
    assert result["ik_selection"]["selected_candidate_index"] == 1
    assert result["joint_target"] == pytest.approx([0.8, 0.8])
    assert result["candidates"][0]["joint_delta_rad"] == pytest.approx(1.1)
    assert result["candidates"][0]["rejection"] == "joint_jump"
    assert result["candidates"][1]["equivalent_turns"] == [-1, 0]


# --- Canonical-interval preference among continuous candidates (EXP11A g005 audit D3) ---
# Keeping each revolute joint's equivalent nearest the measured joints lets an
# arm drift past +/-pi one continuous step at a time. Among continuous
# candidates, one whose revolute joints all lie in [-pi, pi] is tried first, but
# only while its jump stays inside the local joint-jump bound (1 rad by default);
# otherwise the nearest candidate is kept. Admission bounds are unchanged.

PAST_PI = 3.3 - math.tau  # a raw solver coordinate whose equivalent nearest 3.0 is 3.3


def periodic(values):
    value = candidates(values)
    value["revolute"] = [True] * len(values[0])
    return value


def test_a_continuous_canonical_candidate_beats_a_nearer_one_past_pi():
    result = select(periodic([[PAST_PI], [2.6]]), current=[3.0])
    assert result["joint_target"] == pytest.approx([2.6])
    wound, canonical = result["candidates"]
    assert wound["joint_target"] == pytest.approx([3.3]) and wound["canonical"] is False
    assert canonical["canonical"] is True
    summary = result["ik_selection"]
    assert summary["selection"] == "canonical_first_nearest_admissible_raw_joint_distance"
    assert summary["selected_canonical"] is True and summary["canonical_admitted_count"] == 1
    assert summary["canonical_interval_rad"] == pytest.approx([-math.pi, math.pi])
    assert summary["canonical_jump_limit_rad"] == 1.0


def test_the_preference_never_trades_the_nearest_candidate_for_a_jump_beyond_the_local_bound():
    # The global pi bound admits 1.6 (a 1.4 rad jump); the preference stops at 1 rad.
    result = select(periodic([[PAST_PI], [1.6]]), current=[3.0], bound=math.pi, canonical=1.0)
    assert result["joint_target"] == pytest.approx([3.3])
    assert result["ik_selection"]["selected_canonical"] is False
    assert result["ik_selection"]["canonical_admitted_count"] == 0


def test_the_branch_jump_bound_still_refuses_a_canonical_candidate():
    # A local move: the canonical 1.6 is a 1.4 rad jump beyond the 1 rad bound.
    result = select(periodic([[PAST_PI], [1.6]]), current=[3.0], bound=1.0)
    assert result["joint_target"] == pytest.approx([3.3])
    assert result["candidates"][1]["canonical"] is True
    assert result["candidates"][1]["rejection"] == "joint_jump"
    assert result["ik_selection"]["within_jump_limit_count"] == 1
    # The same solution's canonical coordinate (-2.98) is a 6 rad sweep; the command stays at 3.3.
    only = select(periodic([[PAST_PI]]), current=[3.0], bound=math.pi, canonical=1.0)
    assert only["joint_target"] == pytest.approx([3.3]) and only["ik_selection"]["outcome"] == "admitted"


def test_a_blocked_canonical_path_falls_back_to_the_nearest_remaining_candidate():
    seen = []

    def clearance(q):
        seen.append(q)
        return {"status": "self_collision" if abs(q[0] - 2.6) < 1e-9 else "clear"}
    result = select(periodic([[PAST_PI], [2.6]]), current=[3.0], path_clearance=clearance)
    assert seen[0] == pytest.approx([2.6])
    assert result["joint_target"] == pytest.approx([3.3])
    assert result["ik_selection"]["selected_canonical"] is False


@pytest.mark.parametrize("local, expected", [(1.0, 2.3), (0.5, 3.3)])
def test_solve_ik_limits_the_canonical_preference_to_the_session_local_bound(monkeypatch, local, expected):
    from ora.environment.backends.robodojo import reachability
    robot = SimpleNamespace(robot_name="left", arm_joints_name=["joint_0"], entity_origin_pose=TARGET)
    manager = SimpleNamespace(get_joint=lambda *a, **k: [[3.0]], ik_solver={"left": object()})
    session = SimpleNamespace(environment=SimpleNamespace(robot_manager=manager), robots={"left": robot},
                              global_joint_jump_limit_rad=math.pi, local_joint_jump_limit_rad=local)
    monkeypatch.setattr(ik, "sdk_candidates", lambda *a: periodic([[PAST_PI], [2.3]]))
    monkeypatch.setattr(ik, "articulation_limits", lambda *a: [[-10.0, 10.0]])
    monkeypatch.setattr(reachability, "fk_link6", lambda *a: TARGET)
    monkeypatch.setattr(ik, "self_collision_model", lambda *a: SimpleNamespace(
        straight_path=lambda start, end: {"status": "clear"}))
    result = ik.solve_ik(session, "left", TARGET)
    assert result["joint_target"] == pytest.approx([expected])
    assert result["ik_selection"]["canonical_jump_limit_rad"] == local


def test_canonical_candidates_keep_their_nearest_order_among_themselves():
    # Both are canonical, so nothing is reordered: B is nearer (L2 1.05 vs 2.12)
    # although A alone lies within the 1 rad L-infinity preference bound.
    a, b = [0.95] * 5 + [0.0], [1.05] + [0.0] * 5
    value = candidates([a, b])
    value["revolute"] = [True] * 6
    result = select(value, current=[0.0] * 6, bound=math.pi, canonical=1.0)
    assert result["joint_target"] == pytest.approx(b)
    assert result["ik_selection"]["selected_canonical"] is True


# --- Pitch joints may not wind past +/-pi (EXP11A g005 audit D3, review U1) ---
# A candidate that carries joint2-joint5 out of [-pi, pi] -- or farther out
# than it already is -- is refused as pitch_joint_wrap. joint1 may still wind.

X5_NAMES = [f"joint{i}" for i in range(1, 7)]
X5_PITCH = [False, True, True, True, True, False]


def x5(values):
    value = candidates(values)
    value.update(joint_names=list(X5_NAMES), revolute=[True] * 6, pitch=list(X5_PITCH))
    return value


def arm(**joints):
    values = [0.0, 0.5, 1.0, 0.0, 0.0, 0.0]
    for name, angle in joints.items():
        values[X5_NAMES.index(name)] = angle
    return values


def test_a_pitch_joint_leaving_the_canonical_interval_is_refused():
    result = select(x5([arm(joint3=3.3)]), current=arm(joint3=2.9), bound=math.pi, canonical=1.0)
    assert result["joint_target"] is None and result["ik_status"] == "Fail"
    assert result["candidates"][0]["rejection"] == "pitch_joint_wrap"
    summary = result["ik_selection"]
    assert summary["outcome"] == "pitch_joint_wrap" and summary["pitch_joint_wrap_count"] == 1
    assert summary["pitch_joints"] == ["joint2", "joint3", "joint4", "joint5"]
    assert summary["pitch_joint_wrap"] == {"joint": "joint3", "current_rad": 2.9,
                                           "target_rad": pytest.approx(3.3),
                                           "canonical_interval_rad": pytest.approx([-math.pi, math.pi])}
    # The policy text names the physical limit, not joint indices.
    assert "beyond full extension" in summary["recovery"]
    assert not re.search(r"joint\d", summary["recovery"])
    assert result["planning_diagnostics"]["rejection_counts"] == {"pitch_joint_wrap": 1}


def test_a_pitch_joint_already_outside_may_only_return():
    back = select(x5([arm(joint3=2.9)]), current=arm(joint3=3.3), bound=math.pi, canonical=1.0)
    assert back["ik_selection"]["outcome"] == "admitted"
    assert back["joint_target"] == pytest.approx(arm(joint3=2.9))
    further = select(x5([arm(joint3=3.4)]), current=arm(joint3=3.3), bound=math.pi, canonical=1.0)
    assert further["candidates"][0]["rejection"] == "pitch_joint_wrap"
    # Holding a wound pitch joint where it is does not wind it farther.
    held = select(x5([arm(joint3=3.3, joint1=0.2)]), current=arm(joint3=3.3), bound=math.pi, canonical=1.0)
    assert held["ik_selection"]["outcome"] == "admitted"


@pytest.mark.parametrize("current, target, refused", [
    (-2.9, -3.3, True),     # the mirror of 2.9 -> 3.3
    (-3.3, -2.9, False),    # returning from below -pi
    (3.3, 3.3005, False),   # a wound joint held within 1e-3 rad of where it is
    (3.3, 3.302, True),     # 2 mrad farther out is winding, not holding
])
def test_the_pitch_rule_is_symmetric_and_tolerates_a_hold(current, target, refused):
    result = select(x5([arm(joint3=target)]), current=arm(joint3=current), bound=math.pi, canonical=1.0)
    assert (result["candidates"][0]["rejection"] == "pitch_joint_wrap") is refused


def test_a_missing_pitch_mask_is_derived_from_the_revolute_joint_names():
    derived = x5([arm(joint3=3.3)])
    del derived["pitch"]
    assert select(derived, current=arm(joint3=2.9), bound=math.pi)["ik_selection"]["outcome"] == "pitch_joint_wrap"
    # A coordinate not marked revolute cannot wrap, so it is never a pitch joint.
    del derived["revolute"]
    assert select(derived, current=arm(joint3=2.9), bound=math.pi)["ik_selection"]["outcome"] == "admitted"


def test_a_yaw_joint_may_still_wind_past_pi():
    result = select(x5([arm(joint1=3.3)]), current=arm(joint1=2.9), bound=math.pi, canonical=1.0)
    assert result["ik_selection"]["outcome"] == "admitted"
    assert result["joint_target"] == pytest.approx(arm(joint1=3.3))
    assert result["ik_selection"]["selected_canonical"] is False
    assert result["ik_selection"]["pitch_joint_wrap_count"] == 0
    assert result["ik_selection"]["pitch_joint_wrap"] is None


def test_a_pitch_wrap_leaves_a_continuous_alternative_selectable():
    # The nearer candidate winds joint3; the farther one keeps it inside.
    result = select(x5([arm(joint3=3.3), arm(joint3=2.4)]), current=arm(joint3=2.9),
                    bound=math.pi, canonical=1.0)
    assert result["ik_selection"]["outcome"] == "admitted"
    assert result["joint_target"] == pytest.approx(arm(joint3=2.4))
    assert result["ik_selection"]["pitch_joint_wrap_count"] == 1


# --- Straight joint-path self-collision screening (EXP11A g001 action 0121) ---

from pathlib import Path

from ora.environment.controllers.joint_path_clearance import (
    JointPathClearanceError, SphereSelfCollisionModel,
)

# A planar two-link arm: link1 rotates about z at the base; link2 rotates at
# the elbow 0.3 m out. Folding the elbow past about 150 degrees drives the
# forearm sphere back into the base sphere.
PLANAR_URDF = """<robot name="planar">
  <link name="base"/><link name="link1"/><link name="link2"/>
  <joint name="j1" type="revolute"><parent link="base"/><child link="link1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/></joint>
  <joint name="j2" type="revolute"><parent link="link1"/><child link="link2"/>
    <origin xyz="0.3 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/></joint>
</robot>"""
PLANAR_KINEMATICS = {
    "base_link": "base",
    "collision_spheres": {
        "base": [{"center": [0.0, 0.0, 0.0], "radius": 0.05}],
        "link1": [{"center": [0.15, 0.0, 0.0], "radius": 0.03}],
        "link2": [{"center": [0.25, 0.0, 0.0], "radius": 0.04}],
    },
    "self_collision_ignore": {"base": ["link1"], "link1": ["link2"]},
    "self_collision_buffer": {"base": 0.0, "link2": 0.0},
}


def planar_model():
    return SphereSelfCollisionModel.from_solver_kinematics(
        PLANAR_KINEMATICS, urdf_text=PLANAR_URDF, active_joint_names=["j1", "j2"])


def test_sphere_model_checks_only_unignored_pairs_and_reports_the_overlap():
    model = planar_model()
    assert model.pairs == (("base", "link2"),)
    open_gap, pair = model.clearance([0.0, 0.0])
    assert pair == ("base", "link2") and open_gap == pytest.approx(0.55 - 0.09)
    folded_gap, _ = model.clearance([0.0, math.pi])
    assert folded_gap == pytest.approx(0.05 - 0.09)


def test_straight_path_through_the_fold_is_blocked_but_a_detour_endpoint_is_clear():
    model = planar_model()
    through = model.straight_path([0.0, 0.0], [0.0, 3.0])
    assert through["status"] == "self_collision" and through["min_clearance_m"] < 0
    assert through["limiting_pair"] == ["base", "link2"]
    assert through["samples"] == 60 and 0 < through["limiting_fraction"] <= 1
    assert model.straight_path([0.0, 0.0], [1.2, 1.0])["status"] == "clear"


def test_start_inside_the_sphere_model_does_not_forbid_a_path_that_leaves_it():
    model = planar_model()
    assert model.straight_path([0.0, math.pi], [0.0, 0.0])["status"] == "clear"
    assert model.straight_path([0.0, 2.9], [0.0, math.pi])["status"] == "self_collision"


def test_sphere_model_fails_closed_on_missing_links_or_geometry():
    with pytest.raises(JointPathClearanceError):
        SphereSelfCollisionModel.from_solver_kinematics(
            {"base_link": "base"}, urdf_text=PLANAR_URDF, active_joint_names=["j1", "j2"])
    with pytest.raises(JointPathClearanceError):
        SphereSelfCollisionModel.from_solver_kinematics(
            {**PLANAR_KINEMATICS, "collision_spheres": {"ghost": [{"center": [0, 0, 0], "radius": 0.1}]}},
            urdf_text=PLANAR_URDF, active_joint_names=["j1", "j2"])
    with pytest.raises(JointPathClearanceError):
        planar_model().clearance([0.0, float("nan")])


def test_selector_skips_a_nearer_self_colliding_path_and_records_its_evidence():
    seen = []

    def clearance(q):
        seen.append(q)
        return {"status": "self_collision" if q == [0.2] else "clear", "min_clearance_m": -0.05}
    result = select(candidates([[0.2], [0.6], [0.9]]), path_clearance=clearance)
    assert seen == [[0.2], [0.6]]
    assert result["joint_target"] == [0.6]
    summary = result["ik_selection"]
    assert summary["outcome"] == "admitted" and summary["self_collision_path_count"] == 1
    assert result["candidates"][0]["rejection"] == "self_collision_path"
    assert result["candidates"][0]["path_clearance"]["min_clearance_m"] == -0.05
    assert "path_clearance" not in result["candidates"][2]
    assert "self-collision" in result["planning_diagnostics"]["evidence_scope"]


def test_every_admitted_path_blocked_leaves_no_executable_joint_target():
    result = select(candidates([[0.2], [0.6], [1.5]]),
                    path_clearance=lambda q: {"status": "self_collision"})
    assert result["ik_status"] == "Fail" and result["joint_target"] is None
    summary = result["ik_selection"]
    assert summary["outcome"] == "self_collision_path"
    assert summary["selected_candidate_index"] is None and summary["self_collision_path_count"] == 2
    assert "nothing moved" in summary["recovery"]
    diagnostics = result["planning_diagnostics"]
    assert diagnostics["outcome"] == "self_collision_path"
    assert diagnostics["rejection_counts"] == {"self_collision_path": 2, "joint_jump": 1}
    assert diagnostics["recovery"] == [summary["recovery"]]


def test_discontinuous_candidates_keep_their_refusal_without_path_screening():
    calls = []
    result = select(candidates([[2.0]]), path_clearance=lambda q: calls.append(q) or {"status": "clear"})
    assert calls == [] and result["ik_selection"]["outcome"] == "joint_jump"


def test_solve_ik_binds_the_planner_sphere_model_once_and_fails_closed_without_it(monkeypatch, tmp_path):
    from ora.environment.backends.robodojo import reachability
    urdf = tmp_path / "planar.urdf"
    urdf.write_text(PLANAR_URDF)
    planner = SimpleNamespace(robot_cfg={"robot_cfg": {"kinematics": {**PLANAR_KINEMATICS, "urdf_path": str(urdf)}}})
    robot = SimpleNamespace(robot_name="left", arm_joints_name=["j1", "j2"], entity_origin_pose=TARGET)
    manager = SimpleNamespace(get_joint=lambda *a, **k: [[0.0, 0.0]], ik_solver={"left": planner})
    session = SimpleNamespace(environment=SimpleNamespace(robot_manager=manager), robots={"left": robot},
                              global_joint_jump_limit_rad=math.pi)
    monkeypatch.setattr(ik, "sdk_candidates", lambda *a: {
        **candidates([[0.0, 2.95], [2.0, 2.2]]), "joint_names": ["j1", "j2"]})
    monkeypatch.setattr(ik, "articulation_limits", lambda *a: [[-10.0, 10.0]] * 2)
    monkeypatch.setattr(reachability, "fk_link6", lambda *a: TARGET)
    result = ik.solve_ik(session, "left", TARGET)
    assert result["joint_target"] == [2.0, 2.2]
    assert result["candidates"][0]["rejection"] == "self_collision_path"
    cached = session._self_collision_models["left"]
    ik.solve_ik(session, "left", TARGET)
    assert session._self_collision_models["left"] is cached
    manager.ik_solver["right"] = object()
    session.robots["right"] = SimpleNamespace(robot_name="right", arm_joints_name=["j1", "j2"],
                                              entity_origin_pose=TARGET)
    with pytest.raises(ik.RobodojoError, match="sphere model is unavailable"):
        ik.solve_ik(session, "right", TARGET)


ROBODOJO_X5 = Path.home() / "robodojo-runtime/source/RoboDojo/Assets/Robots/x5"


@pytest.mark.skipif(not (ROBODOJO_X5 / "curobo.yml").is_file(), reason="pinned RoboDojo X5 assets absent")
def test_recorded_g001_near_base_sweep_is_blocked_by_the_sdk_spheres():
    import yaml

    kinematics = yaml.safe_load((ROBODOJO_X5 / "curobo.yml").read_text())["robot_cfg"]["kinematics"]
    model = SphereSelfCollisionModel.from_solver_kinematics(
        kinematics, urdf_text=(ROBODOJO_X5 / "X5A.urdf").read_text(),
        active_joint_names=[f"joint{i}" for i in range(1, 7)])
    # EXP11A g001 receipts/action-0121: measured start and the three distinct
    # endpoint families the SDK returned for (-0.088, -0.297, 0.96) top-down.
    start = [5.1e-06, 0.0742, 0.0205, -0.00056, -4.2e-06, 0.00214]
    executed = [-1.9476, 0.7902, -2.5906, 1.8100, 4.9e-06, -0.3768]
    blocked = model.straight_path(start, executed)
    assert blocked["status"] == "self_collision" and blocked["min_clearance_m"] < -0.05
    assert model.straight_path(start, [1.2, 2.16, 1.83, -1.23, 0.0, 2.77])["status"] == "clear"
    # A routine g001 transport (receipts/action-0061) stays clear.
    assert model.straight_path([0.2470, 1.4826, 0.8637, -0.9538, 0.0, 1.8176],
                               [0.2473, 1.4122, 1.0699, -1.2282, 0.0, 1.8178])["status"] == "clear"
