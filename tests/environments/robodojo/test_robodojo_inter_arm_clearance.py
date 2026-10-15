"""Inter-arm screening of RoboDojo joint paths with the solver's own spheres (E11G12-004a).

EXP11A g012 recorded 80 ``controller_stalled`` moves; an offline geometric replay
put 15 of them on one ARX arm driving into the other (link6 13-18 cm apart before
the move). RoboDojo admission screened each arm only against itself. These tests
pin the shared host-safe screen (planar arms with hand-checkable numbers, then
recorded g012 joint lines with the pinned X5 spheres) and its use in IK
candidate selection.
"""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ora.environment.backends.robodojo import ik
from ora.environment.controllers.joint_path_clearance import (
    INTER_ARM_MINIMUM_CLEARANCE_M, InterArmSphereScreen, JointPathClearanceError,
    SphereSelfCollisionModel,
)

# Two planar two-link arms (the IK-continuity fixture): link1 turns about z at
# the base and link2 about z at the elbow 0.3 m out. The left base sits at
# x = -0.6 facing +x; the right base at x = +0.55 facing -x (a half turn about
# z). With both arms at [0, 0] the link2 spheres (r = 0.04) are 0.05 m apart:
# a 0.03 m overlap.
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
    # A self-collision buffer must not shrink the spheres seen by the other arm.
    "self_collision_buffer": {"base": -0.01, "link1": -0.01, "link2": -0.01},
}
HALF_TURN_WXYZ = [0.0, 0.0, 0.0, 1.0]


def planar_model():
    return SphereSelfCollisionModel.from_solver_kinematics(
        PLANAR_KINEMATICS, urdf_text=PLANAR_URDF, active_joint_names=["j1", "j2"])


def planar_screen(right_x=0.55, **kwargs):
    bases = {"left": InterArmSphereScreen.base_transform([-0.6, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
             "right": InterArmSphereScreen.base_transform([right_x, 0.0, 0.0], HALF_TURN_WXYZ)}
    return InterArmSphereScreen({"left": planar_model(), "right": planar_model()}, bases, **kwargs)


def test_clearance_uses_the_raw_spheres_of_both_arms_at_their_world_bases():
    screen = planar_screen()
    gap, links = screen.clearance({"left": [0.0, 0.0], "right": [0.0, 0.0]})
    assert gap == pytest.approx(0.05 - 0.08)
    assert links == {"left": "link2", "right": "link2"}
    # Both arms turned to +y stand 1.15 m apart along x.
    apart, _ = screen.clearance({"left": [math.pi / 2, 0.0], "right": [-math.pi / 2, 0.0]})
    assert apart == pytest.approx(1.15 - 0.10)


def test_the_solver_world_buffer_inflates_every_sphere():
    screen = planar_screen(sphere_buffers_m={"left": 0.002, "right": 0.002})
    gap, _ = screen.clearance({"left": [0.0, 0.0], "right": [0.0, 0.0]})
    assert gap == pytest.approx(0.05 - 0.084)


def test_a_path_into_the_held_other_arm_is_refused_with_its_links_and_clearance():
    screen = planar_screen()
    held = [0.0, 0.0]
    result = screen.straight_paths({"left": [math.pi / 2, 0.0], "right": held}, {"left": [0.0, 0.0], "right": held})
    assert result["status"] == "inter_arm_collision"
    assert result["min_clearance_m"] == pytest.approx(-0.03)
    assert result["required_clearance_m"] == INTER_ARM_MINIMUM_CLEARANCE_M == 0.005
    assert result["limiting_links"] == {"left": "link2", "right": "link2"}
    assert result["limiting_fraction"] == pytest.approx(1.0)
    # Sampled every 0.02 rad (INTER_ARM_MAX_STEP_RAD): pi/2 in 79 steps.
    assert result["moving_arms"] == ["left"] and result["samples"] == 79 and result["max_step_rad"] == 0.02
    assert "finger" in result["claim_boundary"]


def test_a_path_that_keeps_the_margin_or_leaves_an_overlap_is_clear():
    screen = planar_screen(right_x=0.70)  # link2 spheres 0.20 m apart at [0, 0]: 0.12 m clearance
    held = [0.0, 0.0]
    assert screen.straight_paths({"left": [math.pi / 2, 0.0], "right": held},
                                 {"left": [0.0, 0.0], "right": held})["status"] == "clear"
    overlapping = planar_screen()
    leaving = overlapping.straight_paths({"left": [0.0, 0.0], "right": held}, {"left": [math.pi / 2, 0.0], "right": held})
    assert leaving["status"] == "clear" and leaving["start_clearance_m"] == pytest.approx(-0.03)


def test_arms_moving_together_are_screened_in_lockstep():
    screen = planar_screen()
    starts = {"left": [math.pi / 2, 0.0], "right": [-math.pi / 2, 0.0]}
    ends = {"left": [0.0, 0.0], "right": [0.0, 0.0]}
    # Each arm alone against the other's start keeps well clear ...
    for arm, other in (("left", "right"), ("right", "left")):
        alone = screen.straight_paths({arm: starts[arm], other: starts[other]},
                                      {arm: ends[arm], other: starts[other]})
        assert alone["status"] == "clear"
    # ... but together they meet.
    together = screen.straight_paths(starts, ends)
    assert together["status"] == "inter_arm_collision" and together["moving_arms"] == ["left", "right"]


def test_the_screen_fails_closed_on_missing_arms_or_invalid_bases():
    model = planar_model()
    base = InterArmSphereScreen.base_transform([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    with pytest.raises(JointPathClearanceError, match="two"):
        InterArmSphereScreen({"left": model}, {"left": base})
    with pytest.raises(JointPathClearanceError, match="base"):
        InterArmSphereScreen({"left": model, "right": model}, {"left": base, "right": np.full((4, 4), np.nan)})
    with pytest.raises(JointPathClearanceError, match="quaternion"):
        InterArmSphereScreen.base_transform([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0])
    with pytest.raises(JointPathClearanceError):
        planar_screen().straight_paths({"left": [0.0, 0.0]}, {"left": [1.0, 0.0]})


def test_a_frame_bias_is_removed_before_the_base_pose_is_applied():
    biased = InterArmSphereScreen.base_transform([1.0, 2.0, 3.0], [1.0, 0.0, 0.0, 0.0], frame_bias=[0.1, 0.0, 0.0])
    assert biased[:3, 3] == pytest.approx([0.9, 2.0, 3.0])


# --- Recorded EXP11A g012 joint lines with the pinned ARX X5 spheres ---------

ROBODOJO_X5 = Path.home() / "robodojo-runtime/source/RoboDojo/Assets/Robots/x5"
#: env_cfg/robot/dual_x5.yml default_root_pos / default_root_rot (wxyz).
X5_ROOTS = {"left": ([-0.3, -0.45, 0.765], [0.707, 0.0, 0.0, 0.707]),
            "right": ([0.3, -0.45, 0.765], [0.707, 0.0, 0.0, 0.707])}


def x5_screen():
    import yaml

    kinematics = yaml.safe_load((ROBODOJO_X5 / "curobo.yml").read_text())["robot_cfg"]["kinematics"]
    model = SphereSelfCollisionModel.from_solver_kinematics(
        kinematics, urdf_text=(ROBODOJO_X5 / "X5A.urdf").read_text(),
        active_joint_names=[f"joint{i}" for i in range(1, 7)])
    bases = {arm: InterArmSphereScreen.base_transform(*X5_ROOTS[arm]) for arm in X5_ROOTS}
    return InterArmSphereScreen({"left": model, "right": model}, bases)


#: (task, receipt seq, moving arm, measured start, IK joint target, other arm's measured joints)
RECORDED_STALLS = [
    ("arrange_largest_number", 388, "right", [0.487, 1.4818, 0.9373, -1.0239, 0.0, 1.2678],
     [0.5367, 2.3089, 2.0393, -1.2987, 0.0001, 1.3171], [-0.602, 2.6959, 2.6732, -1.6055, -0.0001, -0.4437]),
    ("build_tower", 538, "left", [-1.0152, 2.3711, 2.1461, -1.3548, 0.0011, 0.5743],
     [-0.7075, 2.3744, 2.2053, -1.4016, -0.0, 0.8632], [0.4511, 2.034, 1.8739, -1.4044, 0.0012, -0.4788]),
    ("put_bottles_into_dustbin", 788, "left", [0.0] * 6,
     [-0.733, 2.482, 2.367, -1.5036, -0.106, 3.0273], [0.6153, 2.4599, 2.3784, -1.4908, -0.0013, 0.7294]),
]


@pytest.mark.skipif(not (ROBODOJO_X5 / "curobo.yml").is_file(), reason="pinned RoboDojo X5 assets absent")
@pytest.mark.parametrize("task,seq,arm,start,end,other_joints", RECORDED_STALLS)
def test_recorded_g012_moves_that_stalled_against_the_other_arm_are_refused(task, seq, arm, start, end, other_joints):
    other = "left" if arm == "right" else "right"
    result = x5_screen().straight_paths({arm: start, other: other_joints}, {arm: end, other: other_joints})
    assert result["status"] == "inter_arm_collision", (task, seq)
    assert result["min_clearance_m"] < -0.03


@pytest.mark.skipif(not (ROBODOJO_X5 / "curobo.yml").is_file(), reason="pinned RoboDojo X5 assets absent")
def test_a_recorded_g012_move_that_arrived_beside_the_other_arm_stays_admitted():
    # arrange_largest_number seq 2553 (right arm) arrived; its executed hull
    # clearance was 12.8 mm and the spheres keep 16 mm along the joint line.
    start, end = [0.549, 2.0324, 1.5959, -1.1325, 0.0008, 0.5325], [0.6374, 2.2107, 1.8733, -1.2317, 0.001, 0.6206]
    held = [-0.4279, 2.2036, 1.9722, -1.3381, 0.0001, 0.206]
    result = x5_screen().straight_paths({"right": start, "left": held}, {"right": end, "left": held})
    assert result["status"] == "clear" and 0.01 < result["min_clearance_m"] < 0.03


# --- IK candidate selection against the other arm ---------------------------

from tests.environments.robodojo.test_robodojo_ik_continuity import TARGET, candidates, select  # noqa: E402


def _blocked(evidence):
    return {"status": "inter_arm_collision", "min_clearance_m": 0.0,
            "inter_arm": {"status": "inter_arm_collision", "min_clearance_m": -0.012, "required_clearance_m": 0.005,
                          "limiting_links": {"left": "link4", "right": "link6"}, "limiting_fraction": 0.5,
                          "start_clearance_m": 0.1}}


def test_selector_skips_a_candidate_whose_path_meets_the_other_arm():
    result = select(candidates([[0.2], [0.6]]),
                    path_clearance=lambda q: _blocked(q) if q == [0.2] else {"status": "clear"})
    assert result["joint_target"] == [0.6]
    assert result["candidates"][0]["rejection"] == "inter_arm_collision_path"
    summary = result["ik_selection"]
    assert summary["outcome"] == "admitted"
    assert summary["inter_arm_collision_path_count"] == 1 and summary["self_collision_path_count"] == 0


def test_every_path_meeting_the_other_arm_is_refused_with_links_and_clearance():
    result = select(candidates([[0.2], [0.6]]), path_clearance=_blocked)
    assert result["joint_target"] is None
    summary = result["ik_selection"]
    assert summary["outcome"] == "inter_arm_collision_path"
    assert summary["inter_arm_clearance"] == {
        "min_clearance_m": -0.012, "required_clearance_m": 0.005,
        "limiting_links": {"left": "link4", "right": "link6"}, "limiting_fraction": 0.5,
        "start_clearance_m": 0.1}
    assert "other arm" in summary["recovery"] and "not proof" in summary["recovery"]
    diagnostics = result["planning_diagnostics"]
    assert diagnostics["outcome"] == "inter_arm_collision_path"
    assert diagnostics["rejection_counts"] == {"inter_arm_collision_path": 2}
    assert diagnostics["note"] == (
        "Nearest blocked path: left link4 to right link6 sphere clearance -0.0120 m "
        "(required 0.0050 m) at 50% of the joint path.")


def test_a_self_collision_still_names_its_own_outcome_first():
    order = []

    def clearance(q):
        order.append(q)
        return {"status": "self_collision"} if q == [0.2] else _blocked(q)
    result = select(candidates([[0.2], [0.6]]), path_clearance=clearance)
    assert result["ik_selection"]["outcome"] == "self_collision_path"
    assert result["planning_diagnostics"]["rejection_counts"] == {
        "self_collision_path": 1, "inter_arm_collision_path": 1}


def _planar_session(tmp_path, *, arms=("left", "right"), joints=None, retained=None):
    urdf = tmp_path / "planar.urdf"
    urdf.write_text(PLANAR_URDF)
    origins = {"left": [-0.6, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], "right": [0.55, 0.0, 0.0, *HALF_TURN_WXYZ]}
    current = {"left": [math.pi / 2, 0.0], "right": [0.0, 0.0], **(joints or {})}
    robots = {arm: SimpleNamespace(robot_name=arm, arm_name=f"{arm}_arm", arm_joints_name=["j1", "j2"],
                                   entity_origin_pose=origins[arm]) for arm in arms}
    planners = {arm: SimpleNamespace(
        robot_cfg={"robot_cfg": {"kinematics": {**PLANAR_KINEMATICS, "urdf_path": str(urdf)}}},
        frame_bias=[0.0, 0.0, 0.0]) for arm in arms}
    # The joint target the plant keeps applying to each arm (SDK prev_control).
    targets = {**current, **(retained or {})}
    control = SimpleNamespace(prev_control=[{f"{arm}_arm_joint_state": {"position": list(targets[arm])}
                                             for arm in arms}])
    manager = SimpleNamespace(ik_solver=planners, control_manager=control,
                              process_name=lambda name: name + "_joint_state",
                              get_joint=lambda robot, env_idx_list=None: [current[robot.robot_name]])
    return SimpleNamespace(environment=SimpleNamespace(robot_manager=manager), robots=robots,
                           global_joint_jump_limit_rad=math.pi)


@pytest.fixture
def planar_ik(monkeypatch):
    from ora.environment.backends.robodojo import reachability

    proposals = {"values": [[0.05, 0.0], [3.1, 0.0]]}
    monkeypatch.setattr(ik, "sdk_candidates", lambda *a: {
        **candidates(proposals["values"]), "joint_names": ["j1", "j2"]})
    monkeypatch.setattr(ik, "articulation_limits", lambda *a: [[-10.0, 10.0]] * 2)
    monkeypatch.setattr(reachability, "fk_link6", lambda *a: TARGET)
    return proposals


def test_solve_ik_screens_the_moving_arm_against_the_other_arm_at_its_measured_joints(tmp_path, planar_ik):
    session = _planar_session(tmp_path)
    result = ik.solve_ik(session, "left", TARGET)
    # [0.05, 0] is nearer but swings link2 into the right arm's link2.
    assert result["joint_target"] == [3.1, 0.0]
    blocked = result["candidates"][0]
    assert blocked["rejection"] == "inter_arm_collision_path"
    assert blocked["path_clearance"]["inter_arm"]["limiting_links"] == {"left": "link2", "right": "link2"}
    assert "other arm" in result["planning_diagnostics"]["evidence_scope"]
    assert session._inter_arm_screen is ik.inter_arm_screen(session)


def test_solve_ik_refuses_when_every_path_meets_the_other_arm(tmp_path, planar_ik):
    planar_ik["values"] = [[0.05, 0.0], [0.0, 0.1]]
    result = ik.solve_ik(_planar_session(tmp_path), "left", TARGET)
    assert result["joint_target"] is None
    assert result["ik_selection"]["outcome"] == "inter_arm_collision_path"
    assert result["ik_selection"]["inter_arm_clearance"]["min_clearance_m"] < 0


def test_an_arm_moving_with_the_other_one_is_left_to_the_lockstep_screen(tmp_path, planar_ik):
    session = _planar_session(tmp_path)
    alone = ik.solve_ik(session, "left", TARGET, static_other_arm=False)
    assert alone["joint_target"] == [0.05, 0.0]
    assert "other arm" not in alone["planning_diagnostics"]["evidence_scope"]
    together = ik.lockstep_inter_arm(session, {"left": [math.pi / 2, 0.0], "right": [-math.pi / 2, 0.0]},
                                     {"left": [0.0, 0.0], "right": [0.0, 0.0]})
    assert together["status"] == "inter_arm_collision"
    # An arm not named holds its measured joints.
    held = ik.lockstep_inter_arm(session, {"left": [math.pi / 2, 0.0]}, {"left": [0.05, 0.0]})
    assert held["status"] == "inter_arm_collision" and held["moving_arms"] == ["left"]


def test_a_one_armed_session_has_no_other_arm_to_screen(tmp_path, planar_ik):
    session = _planar_session(tmp_path, arms=("left",))
    assert ik.solve_ik(session, "left", TARGET)["joint_target"] == [0.05, 0.0]
    assert ik.lockstep_inter_arm(session, {"left": [0.0, 0.0]}, {"left": [1.0, 0.0]}) is None


def test_the_inter_arm_screen_fails_closed_without_the_solver_spheres(tmp_path, planar_ik):
    session = _planar_session(tmp_path)
    session.environment.robot_manager.ik_solver["right"] = SimpleNamespace(frame_bias=[0, 0, 0])
    with pytest.raises(ik.RobodojoError, match="sphere model is unavailable"):
        ik.solve_ik(session, "left", TARGET)


def test_the_other_arm_is_screened_along_the_target_it_is_still_driven_to(tmp_path, planar_ik):
    """An arm a command does not name keeps its previous joint target (other_arms_hold).

    After a move that ran out of native actions it is still travelling there, so
    the screen moves it from its measured joints to that retained target in
    lockstep rather than freezing it where it was measured.
    """
    still = _planar_session(tmp_path, joints={"right": [-math.pi / 2, 0.0]})
    assert ik.solve_ik(still, "left", TARGET)["joint_target"] == [0.05, 0.0]
    moving = _planar_session(tmp_path, joints={"right": [-math.pi / 2, 0.0]}, retained={"right": [0.0, 0.0]})
    result = ik.solve_ik(moving, "left", TARGET)
    assert result["joint_target"] == [3.1, 0.0]
    assert result["candidates"][0]["path_clearance"]["inter_arm"]["moving_arms"] == ["left", "right"]
    assert ik.retained_arm_target(moving, "right") == [0.0, 0.0]
    held = ik.lockstep_inter_arm(moving, {"left": [math.pi / 2, 0.0]}, {"left": [0.05, 0.0]})
    assert held["status"] == "inter_arm_collision" and held["moving_arms"] == ["left", "right"]


def test_a_missing_retained_target_holds_the_measured_joints_and_a_corrupt_one_fails_closed(tmp_path, planar_ik):
    session = _planar_session(tmp_path)
    session.environment.robot_manager.control_manager.prev_control[0].pop("right_arm_joint_state")
    assert ik.retained_arm_target(session, "right") == [0.0, 0.0]
    session.environment.robot_manager.control_manager.prev_control[0]["right_arm_joint_state"] = {
        "position": [float("nan"), 0.0]}
    with pytest.raises(ik.RobodojoError, match="retained joint target"):
        ik.retained_arm_target(session, "right")
