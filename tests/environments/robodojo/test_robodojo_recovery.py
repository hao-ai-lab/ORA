"""RoboDojo recovery from wound joint states (EXP11A g003, unit B1).

Recorded evidence: ``classify_objects`` run ``f3a0e6751511`` receipts
``action-0043/0044`` (round 0) and ``action-0079/0080`` (round 1). The arms
were wound -- joint1 near +/-3.5 rad, joint3 near 3.7-4.9 rad -- with the jaws
lowered into or between the task baskets. The released straight joint line
made no lateral progress for 20 native actions twice in a row and both calls
were delivered as ``recovery_not_reached`` (public ``execution_fault``). Where
the arm was unobstructed (``fce4e426dc1e`` action-0055, ``1352c70393c3``
action-0025) the joints moved at about 0.2 rad per native action and one
command ended short while still moving.

The fake plant has toy kinematics (joints 0-2 are the link6 position), so the
session tests reproduce the recorded *joint* states and the two recorded plant
behaviours -- a rate limit and a lateral block that holds only while the jaws
are still low -- rather than the ARX geometry. The sphere-model tests use the
pinned ARX X5 assets when present.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from ora.agent.native.receipt_projection import (
    ReceiptProjectionContext, public_motion_failure_code, resolve_receipt_effect_projector,
)
from ora.environment.backends.robodojo import motion_stall, recovery
from ora.environment.backends.robodojo.bridge import RobodojoError
from tests.environments.robodojo.test_robodojo_backend import fake_sdk_candidate_boundary as _fake_boundary
from tests.environments.robodojo.test_robodojo_backend import sparse_session, v3

fake_sdk_candidate_boundary = pytest.fixture(autouse=True)(_fake_boundary.__wrapped__)

ROBODOJO_X5 = Path.home() / "robodojo-runtime/source/RoboDojo/Assets/Robots/x5"
# Measured arm_joint_qpos before each recorded not_reached recovery.
WOUND = {
    "f3a0-0043-right": [3.519, 1.164, 4.506, 0.733, -0.011, 0.017],
    "f3a0-0044-right": [3.509, 1.154, 4.871, 0.004, 0.008, -0.003],
    "f3a0-0079-left": [-3.809, 0.694, 3.693, 0.946, 0.456, -0.566],
    "f3a0-0079-right": [3.932, 1.265, 4.516, 1.406, -0.007, -0.2],
    "f3a0-0080-right": [3.518, 0.928, 4.072, 1.415, 0.008, 0.001],
}
# Right-arm joint samples of f3a0 action-0044: 20 native actions, no progress.
BLOCKED_TRACE = [[3.509, 1.155, 4.871, 0.005, 0.008, -0.003]] * 5
# Right-arm samples of fce4 action-0055: unobstructed, ~0.2 rad per native action.
MOVING_TRACE = [[3.235, 0.655, 3.595, 1.615, 0.007, -0.626], [3.045, 0.527, 3.397, 1.47, 0.004, -0.426],
                [2.851, 0.366, 3.195, 1.274, 0.003, -0.228], [2.656, 0.186, 2.995, 1.073, 0.002, -0.045]]
RATE_RAD = 0.2


def _stalled(trace):
    """The shared per-native-action stall monitor recovery now drives, fed one recorded trace."""
    monitor = motion_stall.NativeActionStallMonitor({"right": trace[0]}, {"right": [0.0] * 6}, budget=20)
    for joints in trace[1:]:
        monitor.sample({"right": joints})
    return monitor.frozen("right")


def test_the_stall_rule_separates_the_two_recorded_outcomes():
    assert _stalled(BLOCKED_TRACE)
    assert not _stalled(MOVING_TRACE)
    assert not _stalled(BLOCKED_TRACE[:3])  # needs three steps of evidence


def test_unwind_prefers_the_straight_line_and_falls_back_in_a_fixed_order():
    start, home = WOUND["f3a0-0043-right"], [0.0] * 6
    clear = lambda a, b: {"status": "clear", "min_clearance_m": 0.02}
    assert recovery.select_unwind(start, home, clear)["order"] == "straight_joint_line"

    def straight_blocked(a, b):
        straight = all(abs(x) < 1e-12 for x in b) and abs(a[0] - start[0]) < 1e-12 and a[1] == start[1]
        return {"status": "self_collision" if straight else "clear", "min_clearance_m": -0.01}

    chosen = recovery.select_unwind(start, home, straight_blocked)
    assert chosen["order"] == "fold_then_yaw"
    assert [name for name, _ in chosen["segments"]] == ["recover_fold", "recover"]
    assert chosen["segments"][0][1] == [start[0], 0.0, 0.0, 0.0, 0.0, 0.0]
    refused = recovery.select_unwind(start, home, lambda a, b: {"status": "self_collision"})
    assert refused["order"] is None and len(refused["screens"]) == 3


@pytest.mark.skipif(not (ROBODOJO_X5 / "curobo.yml").is_file(), reason="pinned RoboDojo X5 assets absent")
@pytest.mark.parametrize("name", sorted(WOUND))
def test_the_recorded_wound_states_unwind_on_a_self_clear_straight_line(name):
    """Self-collision was not the block: every recorded straight path is sphere-clear."""
    import yaml
    from ora.environment.controllers.joint_path_clearance import SphereSelfCollisionModel

    kinematics = yaml.safe_load((ROBODOJO_X5 / "curobo.yml").read_text())["robot_cfg"]["kinematics"]
    model = SphereSelfCollisionModel.from_solver_kinematics(
        kinematics, urdf_text=(ROBODOJO_X5 / "X5A.urdf").read_text(),
        active_joint_names=[f"joint{i}" for i in range(1, 7)])
    assert recovery.select_unwind(WOUND[name], [0.0] * 6, model.straight_path)["order"] == "straight_joint_line"


def _plant_with(session, arm: str, joints: list[float], *, basket: bool):
    """Put ``arm`` at recorded joints on a rate-limited plant; optionally block lateral motion while low."""
    plant = session.environment
    manager = plant.robot_manager
    for articulation in manager.robot_key:
        articulation.data.joint_pos_limits = np.array([[[-10.0, 10.0]] * 5 + [[-3.14, 3.14]] * 3])
    key = f"{arm}_arm"
    manager.joints[key] = list(joints)
    original = plant.native_tick
    start = list(joints)

    def limited(control):
        before = {name: list(values) for name, values in manager.joints.items()}
        result = original(control)
        for name, values in manager.joints.items():
            moved = [b + float(np.clip(a - b, -RATE_RAD, RATE_RAD)) for a, b in zip(values, before[name])]
            low = before[name][2] < start[2] + 0.05
            over_basket = abs(before[name][0] - start[0]) < 0.15 and abs(before[name][1] - start[1]) < 0.15
            if basket and name == key and low and over_basket:
                # The walls stop lateral motion and the floor stops descent; lifting is free.
                moved[0], moved[1] = before[name][0], before[name][1]
                moved[2] = max(moved[2], before[name][2])
            manager.joints[name] = moved
        return result

    plant.native_tick = limited
    return plant


WIDE = {"workspace_bounds_m": {"x": [-10, 10], "y": [-10, 10], "z": [0, 10]}}


def _recover(session, tmp_path, arm="right"):
    return session.execute(v3(session, tmp_path, kind="recover_motion", targets={arm: "home"}))["receipt"]


def test_released_route_reports_a_blocked_wound_arm_as_stalled_and_stops_early(tmp_path):
    session = sparse_session(tmp_path, **WIDE)
    try:
        _plant_with(session, "right", WOUND["f3a0-0043-right"], basket=True)
        receipt = _recover(session, tmp_path)
        progress = receipt["arms"]["right"]["recovery_progress"]
        assert receipt["execution_status"] == "not_reached"
        assert receipt["failure_code"] == "controller_stalled"
        assert progress["outcome"] == "stalled" and progress["route"] == "straight_joint_line"
        assert receipt["native_actions"] < 20  # no longer spends the whole window blocked
        assert receipt["executor"] == ["robodojo_sdk_joint_interpolation"]
        assert "retract" not in progress
        # The recovery reports the one shared stall rule and its stop evidence.
        assert progress["stall_rule"] == motion_stall.stall_rule()
        assert receipt["arms"]["right"]["motion_progress"]["outcome"] == "stalled"
        assert receipt["arms"]["right"]["motion_progress"]["still_native_actions"] == 3
        # The blocked arm holds where it stopped instead of pushing on toward home.
        held = receipt["arms"]["right"]["motion_progress"]["measured_joints_rad"]
        assert receipt["command_targets_after"]["right_arm_joint_state"]["position"] == pytest.approx(held)
        assert session.pose_targets["right"] == pytest.approx(receipt["after"]["right"]["pose_xyz_xyzw"])
    finally:
        session.close()


def test_retract_then_unwind_frees_the_recorded_state_and_finishes_on_a_second_call(tmp_path):
    session = sparse_session(tmp_path, recovery_route=recovery.RETRACT_THEN_UNWIND, **WIDE)
    try:
        _plant_with(session, "right", WOUND["f3a0-0044-right"], basket=True)
        first = _recover(session, tmp_path)
        progress = first["arms"]["right"]["recovery_progress"]
        segments = [item["segment"] for item in first["arms"]["right"]["segments"]]
        assert segments == ["retract", "recover"]
        assert progress["retract"]["status"] == "planned" and progress["retract"]["outcome"] == "arrived"
        assert progress["retract"]["lift_m"] == pytest.approx(0.10)
        assert progress["unwind_order"] == "straight_joint_line"
        assert first["executor"] == ["robodojo_sdk_ik_joint_interpolation", "robodojo_sdk_joint_interpolation"]
        # 0.2 rad per native action cannot unwind 4.9 rad inside one 20-action window.
        assert first["native_actions"] == 20
        assert first["failure_code"] == "native_action_budget_exhausted"
        assert progress["outcome"] == "still_moving_at_window_end"
        assert progress["joint_distance_end_rad"] < progress["joint_distance_start_rad"] - 3.0
        second = _recover(session, tmp_path)
        assert second["execution_status"] == "reached" and second["failure_code"] is None
        assert second["arms"]["right"]["recovery_progress"]["outcome"] == "arrived"
    finally:
        session.close()


def test_every_unwind_order_blocked_is_refused_before_any_write(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from ora.environment.backends.robodojo import ik

    session = sparse_session(tmp_path, recovery_route=recovery.RETRACT_THEN_UNWIND, **WIDE)
    try:
        _plant_with(session, "right", WOUND["f3a0-0079-right"], basket=False)
        monkeypatch.setattr(ik, "self_collision_model", lambda *a: SimpleNamespace(
            straight_path=lambda start, end: {"status": "self_collision", "min_clearance_m": -0.03,
                                              "limiting_pair": ["base_link", "link4"]}))
        receipt = _recover(session, tmp_path)
        assert receipt["execution_status"] == "plan_rejected" and receipt["native_actions"] == 0
        assert receipt["failure_code"] == recovery.SELF_COLLISION_REFUSAL
        assert receipt["arms"]["right"]["recovery_progress"]["outcome"] == "refused_before_motion"
    finally:
        session.close()


def test_an_unknown_recovery_route_is_refused_at_session_construction(tmp_path):
    with pytest.raises(RobodojoError, match="recovery_route"):
        sparse_session(tmp_path, recovery_route="teleport")


@pytest.mark.parametrize("raw,public", [("controller_stalled", "controller_stalled"),
                                         ("native_action_budget_exhausted", "controller_timeout"),
                                         ("recovery_not_reached", "execution_fault")])
def test_public_codes_no_longer_call_a_blocked_recovery_an_execution_fault(raw, public):
    # The last row is the released code: it read as an infrastructure fault.
    assert public_motion_failure_code("execution", raw) == public


def test_the_policy_sees_the_recovery_progress(tmp_path):
    session = sparse_session(tmp_path, recovery_route=recovery.RETRACT_THEN_UNWIND, **WIDE)
    try:
        _plant_with(session, "right", WOUND["f3a0-0044-right"], basket=True)
        receipt = _recover(session, tmp_path)
    finally:
        session.close()
    projector = resolve_receipt_effect_projector(receipt["schema"])
    effect = projector(receipt, ReceiptProjectionContext(
        last_action_id=receipt["action_id"], last_action_public_name="recover_pose",
        surface_supports=lambda capability: True))
    text = str(effect)
    assert "recovery_progress" in text and "controller_timeout" in text


def test_the_runspec_derives_the_route_only_from_the_declaring_surface():
    from ora.orchestrator.spec import RunSpecError, resolve_run_spec
    from tests.environments.robodojo.test_robodojo_closed_skills_default import ROOT, spec_for

    # The Dojo default declares the route; a runtime value the surface does not
    # declare is refused.
    spec = spec_for("stack_blocks")
    assert resolve_run_spec(spec, root=ROOT)["runtime"]["recovery_route"] == recovery.RETRACT_THEN_UNWIND
    spec["runtime"]["recovery_route"] = "some_other_route"
    with pytest.raises(RunSpecError, match="recovery_route"):
        resolve_run_spec(spec, root=ROOT)


# --- Screened path equals executed path (EXP11A g004 audit E11G4-001) ----------
# Recorded: classify_objects 157b7cd13b33 seq849 retract stalled 67.8 mm short and
# the unwind screened from the *planned* retract target still ran 10 native
# actions from the measured state; dd9223354f34 seq279 and 6602ff5357b7 seq337
# likewise continued after a partial retract.


def _slow_plant(session, arm, joints, *, rate, frozen=lambda before: False):
    """Rate-limited plant; ``frozen(before)`` blocks every joint of ``arm`` (a contact stall)."""
    plant = session.environment
    manager = plant.robot_manager
    for articulation in manager.robot_key:
        articulation.data.joint_pos_limits = np.array([[[-10.0, 10.0]] * 5 + [[-3.14, 3.14]] * 3])
    key = f"{arm}_arm"
    manager.joints[key] = list(joints)
    original = plant.native_tick

    def limited(control):
        before = {name: list(values) for name, values in manager.joints.items()}
        result = original(control)
        for name, values in manager.joints.items():
            if name == key and frozen(before[name]):
                manager.joints[name] = before[name]
                continue
            manager.joints[name] = [b + float(np.clip(a - b, -rate, rate)) for a, b in zip(values, before[name])]
        return result

    plant.native_tick = limited
    return plant


def _recording_screen(monkeypatch, verdict=lambda start, end: "clear"):
    from types import SimpleNamespace
    from ora.environment.backends.robodojo import ik

    starts = []

    def straight_path(start, end):
        starts.append([float(v) for v in start])
        return {"status": verdict(start, end), "min_clearance_m": 0.02}

    monkeypatch.setattr(ik, "self_collision_model", lambda *a: SimpleNamespace(straight_path=straight_path))
    return starts


def test_a_stalled_retract_stops_the_arm_before_any_unwind(tmp_path, monkeypatch):
    _recording_screen(monkeypatch)
    session = sparse_session(tmp_path, recovery_route=recovery.RETRACT_THEN_UNWIND, **WIDE)
    try:
        _slow_plant(session, "right", WOUND["f3a0-0044-right"], rate=RATE_RAD, frozen=lambda before: True)
        receipt = _recover(session, tmp_path)
        arm = receipt["arms"]["right"]
        progress = arm["recovery_progress"]
        assert [item["segment"] for item in arm["segments"]] == ["retract"]
        assert progress["retract"]["outcome"] == "stalled"
        assert progress["stopped"] == {"stage": "retract", "reason": "retract_stalled"}
        assert progress["outcome"] == "stalled" and receipt["failure_code"] == "controller_stalled"
        assert receipt["native_actions"] <= recovery.RETRACT_NATIVE_ACTIONS
    finally:
        session.close()


def test_an_unwind_is_screened_from_the_measured_state_after_a_partial_retract(tmp_path, monkeypatch):
    starts = _recording_screen(monkeypatch)
    session = sparse_session(tmp_path, recovery_route=recovery.RETRACT_THEN_UNWIND, **WIDE)
    try:
        # 0.01 per native action cannot finish a 0.10 m lift in the six-action window.
        _slow_plant(session, "right", WOUND["f3a0-0080-right"], rate=0.01)
        receipt = _recover(session, tmp_path)
        arm = receipt["arms"]["right"]
        progress = arm["recovery_progress"]
        assert progress["retract"]["outcome"] == "still_moving_at_window_end"
        executed = arm["segments"][1]
        assert executed["segment"] == "recover"
        retract_end = arm["segments"][0]
        # The screen that admitted the executed unwind started where the retract really ended.
        measured_after_retract = starts[-1]
        assert recovery.joint_distance(measured_after_retract, arm["segments"][0]["joint_target"]) > 0.02
        assert retract_end["reached"] is False
        assert progress["unwind_screened_from"] == "measured_joint_state"
    finally:
        session.close()


def test_a_stalled_intermediate_segment_does_not_start_the_next(tmp_path, monkeypatch):
    start = WOUND["f3a0-0043-right"]

    def straight_blocked(a, b):
        # A line that yaws joint1 and folds the others at once is blocked; either alone is clear.
        yaw = abs(a[0] - b[0]) > 0.5
        fold = max(abs(x - y) for x, y in zip(a[1:], b[1:])) > 0.1
        return "self_collision" if yaw and fold else "clear"

    _recording_screen(monkeypatch, straight_blocked)
    session = sparse_session(tmp_path, recovery_route=recovery.RETRACT_THEN_UNWIND, **WIDE)
    try:
        lifted = {"done": False}

        def frozen(before):
            # Lifting is free; once lifted, the fold is blocked by contact.
            if before[2] >= start[2] + 0.099:
                lifted["done"] = True
            return lifted["done"]

        _slow_plant(session, "right", start, rate=RATE_RAD, frozen=frozen)
        receipt = _recover(session, tmp_path)
        arm = receipt["arms"]["right"]
        progress = arm["recovery_progress"]
        assert [item["segment"] for item in arm["segments"]] == ["retract", "recover_fold"]
        assert progress["stopped"] == {"stage": "recover_fold", "reason": "segment_stalled"}
        assert receipt["failure_code"] == "controller_stalled"
    finally:
        session.close()


def test_no_clear_order_from_the_measured_state_refuses_after_the_retract(tmp_path, monkeypatch):
    planned = {}

    def blocked_unless_planned(a, b):
        # Clear from the planned retract target, blocked from anywhere else.
        return "clear" if planned.get("joints") is None or recovery.joint_distance(a, planned["joints"]) < 1e-9 else "self_collision"

    starts = _recording_screen(monkeypatch, blocked_unless_planned)
    original = recovery._retract_plan

    def remember(execution, arm, strategy):
        plan = original(execution, arm, strategy)
        planned["joints"] = plan.get("joints")
        return plan

    monkeypatch.setattr(recovery, "_retract_plan", remember)
    session = sparse_session(tmp_path, recovery_route=recovery.RETRACT_THEN_UNWIND, **WIDE)
    try:
        _slow_plant(session, "right", WOUND["f3a0-0080-right"], rate=0.01)
        receipt = _recover(session, tmp_path)
        arm = receipt["arms"]["right"]
        progress = arm["recovery_progress"]
        assert [item["segment"] for item in arm["segments"]] == ["retract"]
        assert progress["outcome"] == "refused_after_retract"
        assert progress["stopped"]["reason"] == "no_self_clear_order_from_measured_state"
        assert receipt["failure_code"] == recovery.SELF_COLLISION_REFUSAL
        assert public_motion_failure_code("execution", receipt["failure_code"]) == recovery.SELF_COLLISION_REFUSAL
        assert receipt["native_actions"] > 0
    finally:
        session.close()


# --- Fallback across calls (EXP11A g005 stuck-state replays) -------------------
# Replaying the recorded control streams to three stuck arms: a straight line home
# freed none; backing off the blocked joints by 0.2 rad then going straight freed
# build_tower g004 in 26 native actions; yawing first freed imitate g005 in 39.
# One-sided block below: joints cannot move toward home while still near the
# stuck pose; backing away releases them.


def _one_sided_plant(session, arm, joints, *, home, release_rad=0.1, rate=RATE_RAD):
    stuck = list(joints)

    def frozen_toward_home(before):
        return False

    plant = _slow_plant(session, arm, joints, rate=rate, frozen=frozen_toward_home)
    manager = plant.robot_manager
    key = f"{arm}_arm"
    inner = plant.native_tick

    freed = {"latched": False}

    def blocked(control):
        before = list(manager.joints[key])
        result = inner(control)
        # Backing away beyond the release distance frees the snag for good.
        if any(abs(b - s) > release_rad for b, s in zip(before[:3], stuck[:3])):
            freed["latched"] = True
        if not freed["latched"]:
            after = manager.joints[key]
            manager.joints[key] = [b if abs(a - h) < abs(b - h) and index < 3 else a
                                   for index, (a, b, h) in enumerate(zip(after, before, home))]
        return result

    plant.native_tick = blocked
    return plant


def _no_retract(monkeypatch):
    original = recovery._retract_plan
    monkeypatch.setattr(recovery, "_retract_plan",
                        lambda execution, arm, strategy: {"status": "skipped", "reason": "test_no_retract"})
    return original


def test_a_stalled_recovery_escalates_to_backing_off_the_blocked_joints(tmp_path, monkeypatch):
    _recording_screen(monkeypatch)
    _no_retract(monkeypatch)
    session = sparse_session(tmp_path, recovery_route=recovery.RETRACT_THEN_UNWIND, **WIDE)
    try:
        home = list(session.home_joints["right"])
        _one_sided_plant(session, "right", WOUND["f3a0-0043-right"], home=home)
        first = _recover(session, tmp_path)["arms"]["right"]["recovery_progress"]
        assert first["fallback_level"] == 0 and first["fallback_strategy"] == "retract_then_unwind"
        assert first["outcome"] == "stalled"
        second = _recover(session, tmp_path)
        progress = second["arms"]["right"]["recovery_progress"]
        segments = [item["segment"] for item in second["arms"]["right"]["segments"]]
        assert progress["fallback_level"] == 1 and progress["fallback_strategy"] == "backoff_then_unwind"
        assert progress["blocked_joints"] == [1, 2, 3]
        assert segments[:2] == ["recover_probe", "recover_backoff"]
        assert progress["outcome"] in {"arrived", "still_moving_at_window_end"}
    finally:
        session.close()


def test_a_second_stall_escalates_to_yaw_first(tmp_path, monkeypatch):
    _recording_screen(monkeypatch)
    _no_retract(monkeypatch)
    session = sparse_session(tmp_path, recovery_route=recovery.RETRACT_THEN_UNWIND, **WIDE)
    try:
        _slow_plant(session, "right", WOUND["f3a0-0044-right"], rate=RATE_RAD, frozen=lambda before: True)
        levels = [_recover(session, tmp_path)["arms"]["right"]["recovery_progress"] for _ in range(4)]
        assert [item["fallback_level"] for item in levels] == [0, 1, 2, 0]
        assert levels[2]["fallback_strategy"] == "yaw_first_unwind"
        assert levels[2]["unwind_orders_preferred"][0] == "yaw_then_fold"
    finally:
        session.close()


def test_an_arrived_recovery_resets_the_fallback_level(tmp_path, monkeypatch):
    _recording_screen(monkeypatch)
    _no_retract(monkeypatch)
    session = sparse_session(tmp_path, recovery_route=recovery.RETRACT_THEN_UNWIND, **WIDE)
    try:
        home = list(session.home_joints["right"])
        _one_sided_plant(session, "right", WOUND["f3a0-0043-right"], home=home, rate=1.0)
        assert _recover(session, tmp_path)["arms"]["right"]["recovery_progress"]["outcome"] == "stalled"
        second = _recover(session, tmp_path)["arms"]["right"]["recovery_progress"]
        assert second["fallback_level"] == 1 and second["outcome"] == "arrived"
        assert session._recovery_fallback["levels"].get("right", 0) == 0
    finally:
        session.close()


def test_the_fallback_level_is_kept_per_round(tmp_path, monkeypatch):
    _recording_screen(monkeypatch)
    _no_retract(monkeypatch)
    session = sparse_session(tmp_path, recovery_route=recovery.RETRACT_THEN_UNWIND, **WIDE)
    try:
        _slow_plant(session, "right", WOUND["f3a0-0044-right"], rate=RATE_RAD, frozen=lambda before: True)
        _recover(session, tmp_path)
        assert session._recovery_fallback["levels"]["right"] == 1
        session.round_index += 1
        assert _recover(session, tmp_path)["arms"]["right"]["recovery_progress"]["fallback_level"] == 0
    finally:
        session.close()


# --- Whole-turn unwind target (EXP11A g005 audit D3) ---------------------------
# X5A.urdf limits joint1-joint5 to +/-10 rad and IK kept the equivalent nearest
# the measured joints, so arms drifted past +/-pi: 100 of g005's 1397 receipts
# ended with a joint beyond +/-pi -- joint1 in 11, joint2 in 13, joint3 in 98
# (|q3| up to 5.83 rad). A home recovery
# aimed at the literal stored joints is then a whole turn back: of 125 per-arm
# home recoveries, 14 of the 17 that started more than pi (largest joint) from
# literal home did not arrive, against 17 of the other 108. The target is now
# the stored joints' whole-turn equivalent nearest the measured joints.

TAU = math.tau
HOME = [0.0, 0.0, 0.8, 0.0, 0.0, 0.0]  # the fake plant's home joints
ONE_TURN = [HOME[0] + TAU, HOME[1], HOME[2] + TAU, HOME[3] - TAU, HOME[4], HOME[5]]


def test_the_unwind_target_is_the_home_equivalent_nearest_the_measured_joints():
    from ora.environment.backends.robodojo.common import nearest_equivalent_joints

    limits = [[-10.0, 10.0]] * 5 + [[-3.14, 3.14]]
    target, turns = nearest_equivalent_joints(HOME, ONE_TURN, limits, [True] * 6)
    assert turns == [1, 0, 1, -1, 0, 0]
    assert target == pytest.approx(ONE_TURN)
    # joint6 is bounded: no whole-turn equivalent fits its actual limits.
    assert nearest_equivalent_joints([0.0] * 6, [0, 0, 0, 0, 0, 3.0 + math.pi], limits, [True] * 6)[1] == [0] * 6
    # A coordinate not marked revolute never gains an equivalent.
    assert nearest_equivalent_joints([0.0], [TAU], [[-10.0, 10.0]], [False]) == ([0.0], [0])


def _periodic_pose(joints):
    """The fake plant's toy kinematics made 2*pi-periodic in every joint, as the X5 revolute joints are."""
    q = [math.remainder(float(value), TAU) for value in joints]
    vector = q[3:6]
    return q[:3] + vector + [math.sqrt(max(0.0, 1.0 - sum(v * v for v in vector)))]


def _periodic_kinematics(monkeypatch, session):
    from ora.environment.backends.robodojo import reachability

    manager = session.environment.robot_manager

    def endpose(robot, env_idx_list=None):
        xyzw = _periodic_pose(manager.joints[robot.arm_name])
        return {0: xyzw[:3] + [xyzw[6]] + xyzw[3:6]}

    monkeypatch.setattr(manager, "get_real_endpose", endpose)
    monkeypatch.setattr(reachability, "fk_link6", lambda session, arm, joints: _periodic_pose(joints))


@pytest.mark.parametrize("route", [None, recovery.RETRACT_THEN_UNWIND])
def test_a_whole_turn_from_home_recovers_in_place_instead_of_sweeping_back(tmp_path, monkeypatch, route):
    session = sparse_session(tmp_path, recovery_route=route, **WIDE)
    try:
        _periodic_kinematics(monkeypatch, session)
        _slow_plant(session, "right", ONE_TURN, rate=RATE_RAD)
        assert session.home_joints["right"] == HOME
        receipt = _recover(session, tmp_path)
        arm = receipt["arms"]["right"]
        # Literal home is a 2*pi sweep: about 32 native actions at 0.2 rad each,
        # more than one 20-action window. The equivalent is where the arm is.
        assert arm["segments"][-1]["joint_target"] == pytest.approx(ONE_TURN)
        assert receipt["execution_status"] == "reached" and receipt["failure_code"] is None
        assert receipt["native_actions"] <= 2
        unwind = arm["recovery_progress"]["unwind_target"]
        assert unwind["status"] == "unwound" and unwind["source"] == "home"
        assert unwind["stored_joints"] == HOME
        assert unwind["equivalent_turns"] == [1, 0, 1, -1, 0, 0]
        assert unwind["correction_rad"] == pytest.approx([TAU, 0, TAU, -TAU, 0, 0])
        assert unwind["joint_target"] == pytest.approx(ONE_TURN)
    finally:
        session.close()


def test_an_equivalent_that_fk_does_not_confirm_keeps_the_stored_joints(tmp_path):
    """The fake plant's toy kinematics are not periodic, so a wound state keeps literal home."""
    session = sparse_session(tmp_path, **WIDE)
    try:
        _plant_with(session, "right", WOUND["f3a0-0044-right"], basket=False)
        unwind = _recover(session, tmp_path)["arms"]["right"]["recovery_progress"]["unwind_target"]
        assert unwind["status"] == "fk_mismatch"
        # joint3 (a pitch joint) at 4.871 may not move out to 0.8 + 2*pi = 7.08.
        assert unwind["pitch_vetoed_turns"] == [0, 0, 1, 0, 0, 0]
        assert unwind["rejected_equivalent_turns"] == [1, 0, 0, 0, 0, 0]
        assert unwind["equivalent_turns"] == [0] * 6 and unwind["correction_rad"] == [0.0] * 6
        assert unwind["joint_target"] == unwind["stored_joints"] == session.home_joints["right"]
    finally:
        session.close()


def test_an_arm_on_its_stored_joints_needs_no_unwind(tmp_path):
    session = sparse_session(tmp_path, **WIDE)
    try:
        unwind = _recover(session, tmp_path)["arms"]["right"]["recovery_progress"]["unwind_target"]
        assert unwind["status"] == "not_needed" and unwind["equivalent_turns"] == [0] * 6
        assert unwind["joint_target"] == unwind["stored_joints"] == HOME
    finally:
        session.close()


# --- Review U1 follow-ups: limits, pitch joints, FK tolerance, finite state ---
X5_LIMITS = [[-10.0, 10.0]] * 5 + [[-3.14, 3.14]]


@pytest.mark.parametrize("stored", [3.1432, 3.2])
def test_a_stored_value_outside_its_limits_gains_no_equivalent(stored):
    """joint6 stored and measured at 3.1432 used to map to -3.14, a whole turn from the arm."""
    from ora.environment.backends.robodojo.common import equivalent_turns, nearest_equivalent_joints

    joints = [0.0] * 5 + [stored]
    target, turns = nearest_equivalent_joints(joints, joints, X5_LIMITS, [True] * 6)
    assert turns == [0] * 6 and target == joints
    assert equivalent_turns(-12.0, 0.0, -10.0, 10.0) == 0
    assert equivalent_turns(0.0, float("nan"), -10.0, 10.0) == 0


def _stub_session(monkeypatch, fk=_periodic_pose):
    from types import SimpleNamespace
    from ora.environment.backends.robodojo import ik, reachability

    monkeypatch.setattr(ik, "articulation_limits", lambda session, arm: [list(bound) for bound in X5_LIMITS])
    monkeypatch.setattr(reachability, "fk_link6", lambda session, arm, joints: fk(joints))
    return SimpleNamespace(robots={"right": SimpleNamespace(arm_joints_name=[f"joint{i}" for i in range(1, 7)])})


def test_the_unwind_never_carries_a_pitch_joint_out_of_the_canonical_interval(monkeypatch):
    session = _stub_session(monkeypatch)
    # joint3 measured at -2.5 lies inside [-pi, pi]; its nearest home equivalent
    # 0.8 - 2*pi = -5.48 does not, so joint3 keeps literal home. joint1 (yaw) winds.
    record = recovery.unwind_target(session, "right", HOME, [-4.0, 0.0, -2.5, 0.0, 0.0, 0.0], "home")
    assert record["status"] == "unwound"
    assert record["equivalent_turns"] == [-1, 0, 0, 0, 0, 0]
    assert record["pitch_vetoed_turns"] == [0, 0, -1, 0, 0, 0]
    assert record["joint_target"] == pytest.approx([-TAU, 0.0, 0.8, 0.0, 0.0, 0.0])


def test_a_stored_target_already_farther_out_is_flagged(monkeypatch):
    # backtrack_last_safe may hold a pitch joint recorded past +pi before the IK
    # rule; no turn brings it nearer, so it is driven as stored and flagged.
    session = _stub_session(monkeypatch)
    record = recovery.unwind_target(session, "right", [0.0, 0.0, 3.5, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 2.9, 0.0, 0.0, 0.0], "backtrack_last_safe")
    assert record["status"] == "not_needed" and record["joint_target"][2] == 3.5
    assert record["stored_pitch_wrap"] == ["joint3"]
    assert "stored_pitch_wrap" not in recovery.unwind_target(session, "right", HOME, HOME, "home")


def test_the_fk_check_catches_an_equivalent_a_few_micrometres_off(monkeypatch):
    # The 0.01 m arrival tolerance accepted 5 um; a genuine whole turn is exact
    # to float rounding, so the check now allows 1e-6 m and 1e-5 rad.
    def fk(joints):
        pose = _periodic_pose(joints)
        return [pose[0] + (5e-6 if joints[0] > math.pi else 0.0), *pose[1:]]

    session = _stub_session(monkeypatch, fk)
    record = recovery.unwind_target(session, "right", HOME, [TAU, 0.0, 0.8, 0.0, 0.0, 0.0], "home")
    assert record["status"] == "fk_mismatch" and record["joint_target"] == HOME
    assert record["rejected_equivalent_turns"] == [1, 0, 0, 0, 0, 0]


def test_a_non_finite_stored_joint_refuses_the_recovery_before_any_write(tmp_path):
    session = sparse_session(tmp_path, **WIDE)
    try:
        session.home_joints["right"] = [float("nan"), *HOME[1:]]
        receipt = _recover(session, tmp_path)
        assert receipt["execution_status"] == "plan_rejected" and receipt["native_actions"] == 0
        assert receipt["failure_code"] == "recovery_state_unavailable"
    finally:
        session.close()
