"""RoboDojo arm motion stops on a measured joint stall and holds there (EXP11A D2).

Released behaviour: a move ended only at arrival or when its 20 native actions
ran out. An arm pressed into the table or a basket held its drive at the effort
limit for the rest of the window and the receipt said
``native_action_budget_exhausted``, which the public projection renamed
``controller_timeout`` -- "still moving, call again". Replaying the production
rule over the recorded EXP11A g001-g006 move_pose/move_free_space commands (the
EXP11A stall replay), 354 of the 403 segments that ended unreached had been
still for at least their last three native actions ("trailing still"; g005: 106
of 119), and the rule fires in 362 of them ("fired": it also fires in a
segment that stalled, moved again and ended unreached; g005: 107).

Stopping early is not enough: the SDK re-applies the retained command on every
later native action, so an arm left with its unreached target keeps pushing
through the next jaw command or wait (review replay: 83 of 84 such moves). A
stalled arm therefore holds the joints measured at its stall.

The fake plant has toy kinematics (joints 0-2 are the link6 position), so a
"table" here is a floor on joint 2.
"""
from __future__ import annotations

import json

import pytest

from ora.agent.composition import resolve_tool_surface
from ora.agent.native.receipt_projection import (
    ReceiptProjectionContext, public_motion_failure_code, resolve_receipt_effect_projector,
)
from ora.environment.backends.robodojo import motion_stall, recovery
from ora.orchestrator.motion_stage import dojo_motion_stage_evidence
from test_robodojo_backend import fake_sdk_candidate_boundary as _fake_boundary
from test_robodojo_backend import pose, probe_request, sparse_session, v3, v3_for

fake_sdk_candidate_boundary = pytest.fixture(autouse=True)(_fake_boundary.__wrapped__)

HOME = [0.0, 0.0, 0.8, 0.0, 0.0, 0.0]
LEFT_KEY = "left_arm_joint_state"
# Right-arm samples of fce4 action-0055 (test_robodojo_recovery.MOVING_TRACE):
# unobstructed, about 0.2 rad per native action.
MOVING_TRACE = [[3.235, 0.655, 3.595, 1.615, 0.007, -0.626], [3.045, 0.527, 3.397, 1.47, 0.004, -0.426],
                [2.851, 0.366, 3.195, 1.274, 0.003, -0.228], [2.656, 0.186, 2.995, 1.073, 0.002, -0.045]]


def _monitor(trace, target, *, budget=20):
    monitor = motion_stall.NativeActionStallMonitor({"left": trace[0]}, {"left": target}, budget=budget)
    verdicts = []
    for joints in trace[1:]:
        monitor.sample({"left": joints})
        verdicts.append(monitor.frozen("left"))
    return monitor, verdicts


# --- the rule on synthetic joint trajectories --------------------------------


def test_the_rule_is_the_released_recovery_rule_on_the_shared_monitor():
    assert motion_stall.STALL_WINDOW_NATIVE_ACTIONS == 3
    assert motion_stall.STALL_JOINT_STEP_RAD == 0.005
    assert motion_stall.STALL_CODE == "controller_stalled"
    from ora.environment.controllers.motion_progress import JointBlockedMonitor

    monitor = motion_stall.NativeActionStallMonitor({"left": HOME}, {"left": HOME}, budget=20)
    assert isinstance(monitor.monitor, JointBlockedMonitor)


def test_an_arm_still_approaching_its_target_never_stalls():
    target = [0.3, 0.0, 0.8, 0.0, 0.0, 0.0]
    # Each native action closes 5% of the remaining joint distance: after 20
    # actions it still moves about 0.0057 rad per action and has not arrived.
    trace, joints = [list(HOME)], list(HOME)
    for _ in range(20):
        joints = [q + 0.05 * (t - q) for q, t in zip(joints, target)]
        trace.append(list(joints))
    _, verdicts = _monitor(trace, target)
    assert not any(verdicts)
    _, recorded = _monitor(MOVING_TRACE, [0.0] * 6)
    assert not any(recorded)


def test_three_still_native_actions_stall_and_one_moving_action_resets_the_count():
    target = [0.0, 0.0, 0.7, 0.0, 0.0, 0.0]
    blocked = [0.0, 0.0, 0.78, 0.0, 0.0, 0.0]
    creeping = [0.0, 0.0, 0.7799, 0.0, 0.0, 0.0]  # 0.0001 rad: still
    _, verdicts = _monitor([HOME, blocked, creeping, blocked, creeping], target)
    assert verdicts == [False, False, False, True]
    freed = [0.0, 0.0, 0.77, 0.0, 0.0, 0.0]  # 0.01 rad: moving again
    _, verdicts = _monitor([HOME, blocked, blocked, freed, freed, freed, freed], target)
    assert verdicts == [False, False, False, False, False, True]


def test_the_stall_evidence_is_compact_and_names_joints_residual_and_hold():
    target = [0.0, 0.0, 0.7, 0.0, 0.0, 0.0]
    blocked = [0.0, 0.0, 0.78, 0.0, 0.0, 0.0]
    monitor, _ = _monitor([HOME, blocked, blocked, blocked, blocked], target, budget=17)
    state = {"arm_joint_qpos": blocked, "pose_xyz_xyzw": [0.0, 0.0, 0.78, 0.0, 0.0, 0.0, 1.0]}
    assert monitor.latch("left", state) == blocked and monitor.stalled("left")
    progress = monitor.progress("left", state, segment="move", arrived=False, terminal=False)
    assert progress["outcome"] == "stalled" and progress["segment"] == "move"
    assert progress["still_native_actions"] == 3 and progress["native_actions"] == 4
    assert progress["measured_joints_rad"] == blocked
    assert progress["joint_residual_max_abs_rad"] == pytest.approx(0.08)
    assert progress["last_joint_step_rad"] == pytest.approx(0.0)
    assert progress["hold"] == "measured_joints_at_stall"
    # The rule is static help, the target and pose residual live in segments[-1].
    assert not {"stall_rule", "joint_target_rad", "pose_residual", "native_action_budget"} & set(progress)
    arrived = monitor.progress("left", state, segment="move", arrived=True, terminal=False)
    assert {key: arrived[key] for key in ("segment", "outcome")} == {"segment": "move", "outcome": "arrived"}
    assert set(arrived) == {"schema", "segment", "outcome"}


# --- sparse motion on the fake plant -----------------------------------------


def _table(session, arm, floor_z, *, give=0.0):
    """Joint 2 of ``arm`` (link6 height in the toy kinematics) cannot go below ``floor_z``.

    The wrapper also applies the retained command to an arm a native action does
    not name, as the SDK does, so a target left below the surface keeps pushing.
    Pushed into it, the surface yields ``give`` per native action.
    """
    plant = session.environment
    manager = plant.robot_manager
    original = plant.native_tick
    key, name = f"{arm}_arm_joint_state", f"{arm}_arm"
    surface = {"floor": floor_z}

    def blocked(control):
        commanded = (control.get(key) or manager.control_manager.prev_control[0][key])["position"]
        result = original(control)
        fraction = getattr(plant, "step_fraction", 1.0)
        joints = [q + fraction * (float(t) - q) for q, t in zip(manager.joints[name], commanded)] \
            if key not in control else list(manager.joints[name])
        if joints[2] < surface["floor"]:
            surface["floor"] -= give
            joints[2] = surface["floor"]
        manager.joints[name] = joints
        return result

    plant.native_tick = blocked
    return surface


def _move(session, tmp_path, targets, **kwargs):
    return session.execute(v3(session, tmp_path, targets=targets, **kwargs))["receipt"]


def _wait(session, tmp_path, ticks):
    return session.execute(v3(session, tmp_path, kind="wait", targets=None) | {"arguments": {"ticks": ticks}})[
        "receipt"]


def _journal(session, receipt):
    trace = json.loads((session.output_root / receipt["control_trace"]["path"]).read_text())
    lines = (session.output_root / trace["physics_journal"]["path"]).read_text().splitlines()
    return trace["samples"], [json.loads(line) for line in lines]


def test_a_move_pressed_into_the_table_ends_early_as_controller_stalled(tmp_path):
    session = sparse_session(tmp_path)
    try:
        _table(session, "left", 0.78)
        receipt = _move(session, tmp_path, {"left": pose([0.0, 0.0, 0.70])})
    finally:
        session.close()
    assert receipt["execution_status"] == "not_reached" and receipt["failure_stage"] == "execution"
    assert receipt["failure_code"] == "controller_stalled"
    # One moving native action, three still ones and one journaled hold -- not the 20-action window.
    assert receipt["native_actions"] == 5
    progress = receipt["arms"]["left"]["motion_progress"]
    assert progress["outcome"] == "stalled" and progress["still_native_actions"] == 3
    assert progress["measured_joints_rad"][2] == pytest.approx(0.78)
    assert progress["joint_residual_max_abs_rad"] == pytest.approx(0.08)
    assert receipt["arms"]["left"]["segments"][-1]["position_error_m"] == pytest.approx(0.08)
    assert receipt["arms"]["left"]["segments"][-1]["reached"] is False


def test_a_stalled_arm_holds_its_measured_joints_through_the_next_commands(tmp_path):
    session = sparse_session(tmp_path)
    try:
        # A surface that yields 1 mm per pushed native action: the arm creeps below
        # the 0.005 rad threshold, stalls, and would keep sinking if still pushed.
        _table(session, "left", 0.78, give=0.001)
        move = _move(session, tmp_path, {"left": pose([0.0, 0.0, 0.70])})
        held = move["arms"]["left"]["motion_progress"]["measured_joints_rad"]
        samples, _ = _journal(session, move)
        wait = _wait(session, tmp_path, 5)
        gripper = session.execute(v3(session, tmp_path, kind="set_gripper", targets={"left": "close"}))["receipt"]
        wait_samples, wait_journal = _journal(session, wait)
        _, gripper_journal = _journal(session, gripper)
        pose_target = list(session.pose_targets["left"])
    finally:
        session.close()
    assert move["failure_code"] == "controller_stalled"
    assert move["arms"]["left"]["motion_progress"]["hold"] == "measured_joints_at_stall"
    assert samples[-1]["segment"] == "stall_hold"
    assert samples[-1]["control"][LEFT_KEY]["position"] == pytest.approx(held)
    assert move["command_targets_after"][LEFT_KEY]["position"] == pytest.approx(held)
    assert pose_target[:3] == pytest.approx(held[:3])  # toy FK: joints 0-2 are the position
    for line in wait_journal + gripper_journal:
        assert line["applied_position_targets"][LEFT_KEY] == pytest.approx(held)
    measured = [sample["measured"]["left"]["arm_joint_qpos"] for sample in wait_samples]
    assert all(joints == pytest.approx(held) for joints in measured)  # no further creep
    assert gripper["after"]["left"]["arm_joint_qpos"] == pytest.approx(held)


def test_an_arm_that_stalls_on_the_last_budgeted_action_holds_through_its_retained_command(tmp_path):
    session = sparse_session(tmp_path)
    try:
        plant = session.environment
        plant.step_fraction = 0.05  # still moving about 0.006 rad per action at action 17
        original, count = plant.native_tick, {"ticks": 0}

        def frozen_after_17(control):
            before = list(plant.robot_manager.joints["left_arm"])
            result = original(control)
            count["ticks"] += 1
            if count["ticks"] > 17:
                plant.robot_manager.joints["left_arm"] = before
            return result

        plant.native_tick = frozen_after_17
        receipt = _move(session, tmp_path, {"left": pose([0.3, 0.0, 0.8])})
        held = receipt["arms"]["left"]["motion_progress"]["measured_joints_rad"]
        _, journal = _journal(session, _wait(session, tmp_path, 2))
    finally:
        session.close()
    assert receipt["failure_code"] == "controller_stalled" and receipt["native_actions"] == 20
    assert receipt["command_targets_after"][LEFT_KEY]["position"] == pytest.approx(held)
    assert all(line["applied_position_targets"][LEFT_KEY] == pytest.approx(held) for line in journal)


def test_a_stalled_arm_holds_while_the_other_arm_keeps_moving(tmp_path):
    session = sparse_session(tmp_path)
    try:
        session.environment.step_fraction = 0.3
        _table(session, "left", 0.78)
        receipt = _move(session, tmp_path, {"left": pose([0.0, 0.0, 0.70]), "right": pose([0.3, 0.0, 0.8])})
        samples, _ = _journal(session, receipt)
    finally:
        session.close()
    held = receipt["arms"]["left"]["motion_progress"]["measured_joints_rad"]
    assert receipt["arms"]["left"]["motion_progress"]["outcome"] == "stalled"
    assert receipt["arms"]["right"]["motion_progress"]["outcome"] == "arrived"
    assert receipt["failure_code"] == "controller_stalled"
    # The left arm stalled on native action 4; from action 5 it is commanded where it stopped.
    assert [sample["control"][LEFT_KEY]["position"] == pytest.approx(held) for sample in samples] == (
        [False] * 4 + [True] * (len(samples) - 4))
    assert len(samples) > 5 and all(sample["segment"] == "move" for sample in samples)


def test_an_arm_still_moving_when_the_window_ends_keeps_the_budget_code(tmp_path):
    session = sparse_session(tmp_path)
    try:
        session.environment.step_fraction = 0.05
        receipt = _move(session, tmp_path, {"left": pose([0.3, 0.0, 0.8])})
    finally:
        session.close()
    assert receipt["failure_code"] == "native_action_budget_exhausted" and receipt["native_actions"] == 20
    progress = receipt["arms"]["left"]["motion_progress"]
    assert progress["outcome"] == "still_moving_at_window_end" and "hold" not in progress
    assert receipt["command_targets_after"][LEFT_KEY]["position"] == pytest.approx([0.3, 0.0, 0.8, 0, 0, 0])


def test_an_arriving_move_reports_only_its_arrival(tmp_path):
    session = sparse_session(tmp_path)
    try:
        receipt = _move(session, tmp_path, {"left": pose([0.02, 0.05, 0.85])})
    finally:
        session.close()
    assert receipt["execution_status"] == "reached" and receipt["failure_code"] is None
    assert receipt["arms"]["left"]["motion_progress"] == {
        "schema": motion_stall.PROGRESS_SCHEMA, "segment": "move", "outcome": "arrived"}
    assert "motion_progress" not in receipt["arms"]["right"]


def test_a_local_step_into_the_table_stalls(tmp_path):
    session = sparse_session(tmp_path)
    try:
        _table(session, "left", 0.78)
        receipt = _move(session, tmp_path, {"left": pose([0.0, 0.0, 0.76])}, interaction="precision")
    finally:
        session.close()
    assert receipt["motion_strategy"] == "local"
    assert receipt["failure_code"] == "controller_stalled" and receipt["native_actions"] == 5


def test_a_free_space_descent_stalls_in_its_own_segment(tmp_path):
    session = sparse_session(tmp_path)
    try:
        _table(session, "left", 0.78)
        receipt = session.execute(v3(session, tmp_path, kind="move_free_space",
                                     targets={"left": pose([0.10, 0.05, 0.70])}))["receipt"]
    finally:
        session.close()
    segments = receipt["arms"]["left"]["segments"]
    assert [item["segment"] for item in segments] == ["lift", "translate", "descend"]
    assert [item["reached"] for item in segments] == [True, True, False]
    assert receipt["failure_code"] == "controller_stalled" and receipt["native_actions"] == 7
    assert receipt["arms"]["left"]["motion_progress"]["segment"] == "descend"


def test_a_dual_arm_move_stops_once_one_arm_arrived_and_the_other_stalled(tmp_path):
    session = sparse_session(tmp_path)
    try:
        _table(session, "left", 0.78)
        receipt = _move(session, tmp_path, {"left": pose([0.0, 0.0, 0.70]), "right": pose([0.0, 0.0, 0.85])})
    finally:
        session.close()
    assert receipt["failure_code"] == "controller_stalled" and receipt["native_actions"] == 5
    assert receipt["arms"]["left"]["motion_progress"]["outcome"] == "stalled"
    assert receipt["arms"]["right"]["motion_progress"]["outcome"] == "arrived"


def test_a_probed_plan_reference_into_the_table_stalls(tmp_path):
    session = sparse_session(tmp_path)
    try:
        observation, request = probe_request(session, tmp_path, [0.0, 0.0, 0.70])
        plan_ref = session.probe_motion(request)["results"][0]["plan_ref"]
        _table(session, "left", 0.78)
        receipt = session.execute(v3_for(observation, session, plan_ref=plan_ref))["receipt"]
    finally:
        session.close()
    assert receipt["failure_code"] == "controller_stalled" and receipt["native_actions"] == 5


def test_recovery_uses_the_shared_rule_and_keeps_no_copy_of_its_own():
    assert not hasattr(recovery, "stalled")
    assert not hasattr(recovery, "STALL_WINDOW") and not hasattr(recovery, "STALL_JOINT_STEP_RAD")


# --- what the policy sees ----------------------------------------------------


def _project(receipt):
    projector = resolve_receipt_effect_projector(receipt["schema"])
    return projector(receipt, ReceiptProjectionContext(
        last_action_id=receipt["action_id"], last_action_public_name="move_pose",
        surface_supports=lambda capability: True))


def _stalled_move(tmp_path):
    session = sparse_session(tmp_path)
    try:
        _table(session, "left", 0.78)
        return _move(session, tmp_path, {"left": pose([0.0, 0.0, 0.70])})
    finally:
        session.close()


def test_a_stalled_move_reaches_the_policy_as_a_stall_with_its_evidence(tmp_path):
    effect = _project(_stalled_move(tmp_path))
    assert effect["failure_code"] == "controller_stalled"
    progress = effect["detail"]["arms"]["left"]["motion_progress"]
    assert progress["outcome"] == "stalled" and progress["still_native_actions"] == 3
    assert progress["hold"] == "measured_joints_at_stall" and "schema" not in progress
    assert "controller_timeout" not in str(effect)


def test_a_grasp_execute_stage_carries_the_stop_and_the_public_code(tmp_path):
    receipt = _stalled_move(tmp_path)
    residual = {"reason": receipt["failure_code"], "errors_by_arm": {"left": {"translation_m": 0.08}}}
    public, extra = dojo_motion_stage_evidence(receipt, residual)
    assert public["reason"] == "controller_stalled" and public["errors_by_arm"] == residual["errors_by_arm"]
    assert extra == [{"quantity": "motion_progress",
                      "value": {"left": receipt["arms"]["left"]["motion_progress"]}}]
    budget = {**receipt, "failure_code": "native_action_budget_exhausted",
              "arms": {"left": {**receipt["arms"]["left"],
                                "motion_progress": {"outcome": "still_moving_at_window_end"}}}}
    public, _ = dojo_motion_stage_evidence(budget, {"reason": "native_action_budget_exhausted"})
    assert public["reason"] == "controller_timeout"  # the private name never reaches the policy
    other = {"schema": "ora.robotwin-sparse-motion-receipt.v7", "failure_code": "x"}
    assert dojo_motion_stage_evidence(other, {"reason": "x"}) == ({"reason": "x"}, [])


def _budget_receipt(outcomes):
    return {"schema": "ora.robodojo-sparse-motion-receipt.v1", "action_id": "action-3",
            "action": "move_pose", "execution_status": "not_reached", "failure_stage": "execution",
            "failure_code": "native_action_budget_exhausted", "target_arms": list(outcomes),
            "terminal_after_action": False,
            "arms": {arm: {"targeted": True, **({"motion_progress": {"outcome": outcome}} if outcome else {})}
                     for arm, outcome in outcomes.items()}}


@pytest.mark.parametrize("outcomes,expected", [
    ({"left": "stalled"}, "controller_stalled"),
    ({"left": "stalled", "right": "arrived"}, "controller_stalled"),
    ({"left": "stalled", "right": "still_moving_at_window_end"}, "controller_timeout"),
    ({"left": "still_moving_at_window_end"}, "controller_timeout"),
    ({"left": None}, "controller_timeout"),  # released receipts carry no progress block
])
def test_the_projection_reads_the_receipt_s_own_progress_before_calling_a_stop_a_timeout(outcomes, expected):
    assert _project(_budget_receipt(outcomes))["failure_code"] == expected


def test_a_stalled_recovery_is_never_projected_as_a_timeout():
    record = _budget_receipt({"right": None})
    record["arms"]["right"]["recovery_progress"] = {"outcome": "stalled"}
    assert _project(record)["failure_code"] == "controller_stalled"
    assert public_motion_failure_code("execution", "native_action_budget_exhausted") == "controller_timeout"


@pytest.mark.parametrize("surface", ["robodojo_placement_closed_skills_v2"])
def test_the_dojo_help_states_the_stall_and_hold_the_executor_applies(surface):
    from ora.agent.context import policy_units

    resolved = resolve_tool_surface(surface)
    help_text = {item["name"]: item["description"] for item in resolved.definitions}
    rule = (f"no joint moves {motion_stall.STALL_JOINT_STEP_RAD:g} rad per native action for "
            f"{motion_stall.STALL_WINDOW_NATIVE_ACTIONS} native actions short of target, then holds its "
            "measured joints: controller_stalled with motion_progress")
    if resolved.supports(policy_units.CAPABILITY):  # the same rule, restated in degrees
        rule = policy_units.degrees_text(rule)
        assert " rad " not in rule
    assert rule in help_text["move_pose"]


def test_the_live_recovery_help_points_at_the_move_pose_rule():
    help_text = {item["name"]: item["description"]
                 for item in resolve_tool_surface("robodojo_placement_closed_skills_v2").definitions}
    assert "no joint moved for 3 native actions" not in help_text["recover_pose"]
    assert "move_pose stall rule" in help_text["recover_pose"]
