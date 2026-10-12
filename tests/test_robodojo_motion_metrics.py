"""U1 v2 L0: pinned RoboDojo motion metrics over synthetic native journals.

Every metric is checked for its numerator, its denominator and the edge of its
definition (predecessor physics step, round starts, segment classes, cohorts,
the still rule). The journals are synthetic; nothing here starts Isaac.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from ora.orchestrator import robodojo_motion_measures as measures
from ora.orchestrator import robodojo_motion_metrics as metrics
from ora.orchestrator.robodojo_motion_journal import (
    JournalError, load_run_journal, segment_class, summarize_values,
)
from tests.robodojo_journal_fixture import (
    ZERO, action, control, joint_ramp, moving, pose_target, receipt, segment_entry, still, tick, write_native,
)

RIGHT_EE = {"right_ee_joint_state": {"position": [0.0, 0.0]}}
HELD = {"gripper": 0.4, "target": "close"}


def _metrics(run: Path, **kwargs):
    return metrics.build_report([run], **kwargs)["runs"][0]["metrics"]


def _vec(*head: float) -> list[float]:
    return [*head, *([0.0] * (6 - len(head)))]


def _gripper_tick(**right_kwargs) -> dict:
    return tick("gripper", RIGHT_EE, right=still("right", **right_kwargs))


def _stalled_close(arm: str = "right", floor: float = 0.05, state: str = "stalled_before_target") -> dict:
    return receipt("set_gripper", status="not_reached", code="gripper_stalled_before_target",
                   arguments={"targets": {arm: "close"}}, gripper_floors={arm: floor},
                   arms={arm: {"gripper_outcome": {"state": state}}})


def _open(arm: str = "right") -> dict:
    return receipt("set_gripper", arguments={"targets": {arm: "open"}},
                   arms={arm: {"gripper_outcome": {"state": "reached_target"}}})


def test_physics_step_saturation_counts_commanded_arm_ticks_with_the_predecessor_step(tmp_path):
    t0 = joint_ramp(ZERO, _vec(0.02))  # 5 rad/s inside the first tick of the round
    t1 = joint_ramp(t0[-1], _vec(0.01))  # 2.5 rad/s
    t2 = [_vec(t1[-1][0] + 0.02)] * 10  # still inside, 5 rad/s across the tick boundary
    z0 = [1.0 - 0.001 * (k + 1) for k in range(10)]
    z1 = [z0[-1] - 0.004 * (k + 1) for k in range(10)]
    round0 = [
        action(receipt("move_pose"), [
            tick("move", control(right=_vec(1.0)), right=moving("right", t0, z0)),
            tick("move", control(right=_vec(1.0)), right=moving("right", t1, z1)),
            tick("move", control(right=_vec(1.0)), right=moving("right", t2, [z1[-1]] * 10, **HELD)),
        ]),
        action(receipt("set_gripper"), [_gripper_tick()]),
    ]
    round1 = [action(receipt("move_pose"), [
        tick("move", control(right=_vec(1.0)), right=moving("right", [_vec(0.02)] * 10))])]
    result = _metrics(write_native(tmp_path / "run", [round0, round1]))
    baseline = result["v1_baseline_diagnostics"]["saturation_physics_step"]
    assert (baseline["numerator"], baseline["denominator"], baseline["share"]) == (2, 4, 0.5)
    assert (baseline["closed_command"]["numerator"], baseline["closed_command"]["denominator"]) == (1, 1)
    assert baseline["holding_mode"]["denominator"] == 0
    speed = result["link6_peak_speed_m_s"]
    assert speed["all"]["count"] == 4
    assert speed["all"]["p50"] == pytest.approx(0.25)
    assert speed["all"]["p90"] == pytest.approx(1.0)
    assert speed["closed_command"]["max"] == pytest.approx(0.0)


def test_commanded_speed_violation_counts_only_the_steps_inside_each_tick(tmp_path):
    def applied(per_step: float, start: float = 0.0) -> list[dict]:
        return [{"right_arm_joint_state": _vec(start + per_step * (k + 1)), "left_arm_joint_state": list(ZERO)}
                for k in range(10)]

    fast = tick("move", control(right=_vec(1.0)), applied=applied(0.02))  # 5 rad/s > 4.0
    slow = tick("move", control(right=_vec(1.0)), applied=applied(0.01, 0.2))  # 2.5 rad/s, ends at 0.3
    snap = tick("move", control(right=_vec(1.0)), applied=applied(0.0, 0.5))  # 0.2 rad back-step at the boundary
    unrecorded = tick("move", control(right=_vec(1.0)), applied=[{}] * 10)
    closed = tick("move", control(right=_vec(1.0)), right=still("right", **HELD), applied=applied(0.0, 0.3))
    run = write_native(tmp_path / "run", [[action(receipt("move_pose"), [fast, slow, snap, unrecorded, closed])]])
    violation = _metrics(run)["commanded_speed_violation"]
    assert (violation["numerator"], violation["denominator"]) == (1, 4)
    assert violation["unrecorded_arm_ticks"] == 1
    assert (violation["v_cmd_rad_s"], violation["v_cmd_status"]) == (4.0, "candidate")
    assert violation["closed_command"]["denominator"] == 1
    assert violation["commanded_peak_rad_s"]["max"] == pytest.approx(5.0)
    assert _metrics(run, v_cmd=6.0)["commanded_speed_violation"]["numerator"] == 0


def test_tick_displacement_saturation_counts_moving_arm_ticks_against_the_0_2_rad_tick_cap(tmp_path):
    t0 = joint_ramp(ZERO, _vec(0.019))  # no predecessor: excluded
    t1 = joint_ramp(t0[-1], _vec(0.0195))  # 0.195 rad: saturated
    t2 = joint_ramp(t1[-1], _vec(0.01))  # 0.1 rad: moving
    t3 = [_vec(t2[-1][0] + 0.001)] * 10  # 0.001 rad: still, excluded
    ticks = [tick("move", control(right=_vec(3.0)), right=moving("right", path)) for path in (t0, t1, t2, t3)]
    result = _metrics(write_native(tmp_path / "run", [[action(receipt("move_pose"), ticks)]]))
    baseline = result["v1_baseline_diagnostics"]["saturation_tick_displacement"]
    assert (baseline["numerator"], baseline["denominator"]) == (1, 2)


def test_out_of_step_uses_constant_target_segments_and_joints_moving_more_than_0_05_rad(tmp_path):
    a0 = joint_ramp(ZERO, _vec(0.1, 0.04, 0.004))  # spread 0.6; joint 3 moves 0.04 and is ignored
    a1 = joint_ramp(a0[-1], _vec(0.1, 0.06))  # spread 0.4
    a2a = joint_ramp(a1[-1], _vec(0.05, 0.04))
    a2b = joint_ramp(a2a[-1], _vec(0.05))  # same segment name, another target: not a v1 segment
    round0 = [
        action(receipt("set_gripper"), [_gripper_tick()]),
        action(receipt("move_pose"), [tick("move", control(right=_vec(1.0, 1.0, 0.04)),
                                           right=moving("right", a0))]),
        action(receipt("move_pose"), [tick("move", control(right=_vec(2.0, 1.4, 0.04)),
                                           right=moving("right", a1))]),
        action(receipt("move_pose"), [
            tick("local", control(right=_vec(2.5, 1.5, 0.04)), right=moving("right", a2a)),
            tick("local", control(right=_vec(3.0, 1.4, 0.04)), right=moving("right", a2b)),
        ]),
    ]
    out = _metrics(write_native(tmp_path / "run", [round0]))["v1_baseline_diagnostics"]["out_of_step"]
    assert (out["spread_above_0.3"]["numerator"], out["spread_above_0.3"]["denominator"]) == (2, 2)
    assert (out["spread_above_0.5"]["numerator"], out["spread_above_0.5"]["denominator"]) == (1, 2)
    assert out["changing_target_segments"] == 1
    assert "not applicable to planned trajectories" in out["applies_to"]


def test_sag_uses_the_receipt_goal_height_on_pose_segments_by_holding_state(tmp_path):
    dip = [0.99, 0.98, 0.97, 0.97, 0.98, 0.99, 1.0, 1.0, 1.0, 1.0]
    shallow = [0.995, 0.99, 0.985, 0.985, 0.99, 0.995, 1.0, 1.0, 1.0, 1.0]
    short = [0.99, 0.98, 0.975, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97]  # ends 30 mm under its 1.0 goal
    deep = [0.99, 0.97, 0.95, 0.95, 0.97, 0.99, 1.0, 1.0, 1.0, 1.0]

    def move(segment, z, goal_z, **kwargs):
        entries = [] if goal_z is None else [segment_entry(segment, 1, goal_z)]
        return action(receipt("move_pose", arms={"right": {"segments": entries}}),
                      [tick(segment, control(right=_vec(0.5)), right=moving("right", [_vec(0.1)] * 10, z, **kwargs))])

    round0 = [
        action(receipt("set_gripper"), [_gripper_tick()]),
        move("move", dip, 1.0), move("move", shallow, 1.0), move("move", short, 1.0),
        move("local", [0.9] * 5 + [1.0] * 5, 1.0), move("move", dip, None),
        action(_stalled_close(), [tick("gripper", RIGHT_EE, right=still("right", **HELD))]),
        move("precontact", deep, 1.0, **HELD), move("retract", deep, 1.0, **HELD),
    ]
    sag = _metrics(write_native(tmp_path / "run", [round0]))["sag"]
    assert (sag["free_space"]["numerator"], sag["free_space"]["denominator"]) == (2, 3)
    assert (sag["holding"]["numerator"], sag["holding"]["denominator"]) == (1, 1)
    assert sag["unmatched"] == 1
    assert sag["segments"] == ["move", "plan_ref", "precontact"]


def _held_rounds() -> list[list[dict]]:
    def carry(arm, gripper):
        body = still(arm, gripper=gripper, target="close")
        return action(receipt("move_pose"), [tick("move", control(**{arm: _vec(0.1)}), **{arm: body})])

    def close_tick(arm, gripper=0.4):
        key = {f"{arm}_ee_joint_state": {"position": [0.0, 0.0]}}
        return tick("gripper", key, **{arm: still(arm, gripper=gripper, target="close")})

    round0 = [
        action(_stalled_close(), [close_tick("right")]),
        carry("right", 0.07),  # the jaw closes to its floor: lost
        action(_open(), [tick("gripper", RIGHT_EE)]),
        action(_stalled_close(), [close_tick("right")]),
        carry("right", 0.3),
    ]
    reached_close = receipt("set_gripper", arguments={"targets": {"right": "close"}}, gripper_floors={"right": 0.05},
                            arms={"right": {"gripper_outcome": {"state": "reached_target"}}})
    round1 = [
        action(reached_close, [close_tick("right", gripper=0.05)]),  # at its floor: holds nothing
        action(_stalled_close("left"), [close_tick("left")]),
        carry("left", 0.2),
        action(_stalled_close(state="not_reached"), [close_tick("right", gripper=0.3)]),
        carry("right", 0.06),  # closed-command cohort only, lost
    ]
    return [round0, round1]


def test_held_then_lost_reports_the_holding_mode_and_closed_command_cohorts(tmp_path):
    held = _metrics(write_native(tmp_path / "run", _held_rounds()))["held_then_lost"]
    assert (held["holding_mode"]["numerator"], held["holding_mode"]["denominator"]) == (1, 3)
    assert (held["closed_command"]["numerator"], held["closed_command"]["denominator"]) == (2, 4)
    assert held["holding_mode"]["jaw_floor"] == 1
    assert held["holding_mode"]["object_displacement"] is None and held["objects"] is None


def test_held_then_lost_selects_the_object_at_the_jaw_contact_centre_in_a_replay(tmp_path):
    run = write_native(tmp_path / "run", _held_rounds())
    centre = [0.3 + measures.JAW_CONTACT_CENTRE_LINK6_X_M, -0.35, 1.0]
    at_link6 = [0.3, -0.35, 1.0]  # nearer link6 than the block, but not between the pads
    rows = [
        {"action_sequence": 3, "tick_index": 0, "objects": {"block": {"position_m": centre},
                                                           "decoy": {"position_m": at_link6}}},
        {"action_sequence": 4, "tick_index": 0, "objects": {"block": {"position_m": [*centre[:2], 0.97]},
                                                           "decoy": {"position_m": at_link6}}},
    ]
    poses = tmp_path / "object-poses.jsonl"
    poses.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    held = _metrics(run, object_poses=metrics.load_object_poses(poses))["held_then_lost"]
    assert (held["holding_mode"]["numerator"], held["holding_mode"]["denominator"]) == (2, 3)
    assert (held["holding_mode"]["jaw_floor"], held["holding_mode"]["object_displacement"]) == (1, 1)
    [record] = held["objects"]
    assert (record["object"], record["action_sequence"], record["lost"]) == ("block", 3, True)
    assert record["distance_to_jaw_centre_m"] == pytest.approx(0.0, abs=1e-12)
    assert record["max_link6_drift_m"] == pytest.approx(0.03)
    assert held["object_frame_check"]["status"] == "consistent"


def test_object_drift_is_measured_in_the_link6_frame():
    quarter_turn_z = [0.3, 0.0, 1.0, 0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)]
    assert metrics.link6_frame_offset(quarter_turn_z, [0.3, 0.15, 1.0]) == pytest.approx([0.15, 0.0, 0.0], abs=1e-12)
    assert measures.jaw_contact_centre(quarter_turn_z) == pytest.approx([0.3, 0.1501, 1.0], abs=1e-12)
    assert measures.object_frame_check([{"distance_to_jaw_centre_m": 1.2}])["status"] == "inconsistent"


def test_object_drift_in_an_inconsistent_frame_is_withheld(tmp_path):
    run = write_native(tmp_path / "run", _held_rounds())
    far = [1.5, -0.35, 1.0]  # 1.05 m from the jaw centre: not link6's frame
    rows = [{"action_sequence": 3, "tick_index": 0, "objects": {"block": {"position_m": far}}},
            {"action_sequence": 4, "tick_index": 0, "objects": {"block": {"position_m": [*far[:2], 0.9]}}}]
    poses = tmp_path / "object-poses.jsonl"
    poses.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    held = _metrics(run, object_poses=metrics.load_object_poses(poses))["held_then_lost"]
    assert held["object_frame_check"]["status"] == "inconsistent"
    assert held["object_evidence"].startswith("withheld")
    assert held["holding_mode"]["object_displacement"] is None
    assert (held["holding_mode"]["numerator"], held["holding_mode"]["denominator"]) == (1, 3)


def test_a_stalled_close_already_at_the_floor_opens_no_holding_mode(tmp_path):
    floor_close = [tick("gripper", RIGHT_EE, right=still("right", gripper=0.06, target="close"))]
    carry = tick("move", control(right=_vec(0.5)),
                 right=moving("right", joint_ramp(ZERO, _vec(0.02)), gripper=0.06, target="close"))
    run = write_native(tmp_path / "run", [[action(_stalled_close(), floor_close),
                                           action(receipt("move_pose"), [carry])]])
    result = _metrics(run)
    assert result["v1_baseline_diagnostics"]["saturation_physics_step"]["holding_mode"]["denominator"] == 0
    assert result["v1_baseline_diagnostics"]["saturation_physics_step"]["closed_command"]["denominator"] == 1
    assert result["held_then_lost"]["holding_mode"]["denominator"] == 0


def test_arrival_counts_every_pose_command_with_the_executed_subset_beside_it(tmp_path):
    def reach(z_end, arguments, **kwargs):
        z = [1.0 + (z_end - 1.0) * (k + 1) / 10 for k in range(10)]
        return action(receipt("move_pose", arguments=arguments, **kwargs),
                      [tick("move", control(right=_vec(0.5)), right=moving("right", [_vec(0.5)] * 10, z))])

    plan_ref = receipt("move_pose", arguments={"plan_ref": "plan-x"},
                       arms={"right": {"segments": [segment_entry("plan_ref", 1, 0.9)]}})
    round0 = [
        reach(0.955, {"targets": {"right": pose_target([0.3, -0.35, 0.95])}}),
        reach(0.9, {"targets": {"right": pose_target([0.3, -0.35, 0.8])}},
              status="not_reached", code="native_action_budget_exhausted"),
        action(receipt("move_pose", status="planner_failed", code="ik_failed:move",
                       arguments={"targets": {"right": pose_target([0.3, -0.35, 0.7])}})),
        action(plan_ref, [tick("plan_ref", control(right=_vec(0.5)),
                               right=moving("right", [_vec(0.5)] * 10, [0.9] * 10))]),
        action(receipt("recover_motion", arguments={"targets": {"right": "home"}}),
               [tick("recover", control(right=ZERO))]),
    ]
    arrival = _metrics(write_native(tmp_path / "run", [round0]))["arrival"]
    assert (arrival["numerator"], arrival["denominator"]) == (2, 4)
    assert (arrival["executed"]["numerator"], arrival["executed"]["denominator"]) == (2, 3)
    assert arrival["receipt_status"] == {"not_reached": 1, "planner_failed": 1, "reached": 2}


def test_native_action_cost_splits_global_and_local_segments_and_reports_the_round_share(tmp_path):
    both = control(left=_vec(0.3), right=_vec(0.3))

    def ticks(segment, target, count):
        return [tick(segment, target) for _ in range(count)]

    round0 = [
        action(receipt("move_pose"), ticks("move", control(right=_vec(0.1)), 3)),
        action(receipt("move_pose"), ticks("precontact", control(right=_vec(0.2)), 1)
               + ticks("local", control(right=_vec(0.25)), 2)),
        action(receipt("recover_motion"),
               ticks("retract", both, 2) + ticks("recover", control(left=ZERO, right=ZERO), 4)),
        action(receipt("set_gripper"), [_gripper_tick(), _gripper_tick()]),
        action(receipt("wait"), [tick("wait", {})]),
        action(receipt("move_pose", status="planner_failed", code="ik_failed:move")),
    ]
    round1 = [action(receipt("move_pose"), ticks("move", control(right=_vec(0.4)), 5))]
    cost = _metrics(write_native(tmp_path / "run", [round0, round1]))["native_actions"]
    per_segment = cost["per_segment"]
    assert per_segment["global"] == {"count": 5, "mean": 3.0, "p50": 3, "p90": 5, "max": 5}
    assert per_segment["local"]["count"] == 1 and per_segment["local"]["mean"] == 2.0
    assert per_segment["by_segment"]["move"]["count"] == 2
    assert per_segment["by_segment"]["recover"]["max"] == 4
    assert "all" not in per_segment
    per_command = cost["per_command"]
    assert per_command["move_pose"]["count"] == 3 and per_command["move_pose"]["max"] == 5
    assert per_command["recover_motion"]["mean"] == 6.0
    assert (cost["per_round"]["numerator"], cost["per_round"]["denominator"]) == (20, 1100)


def test_timeout_stillness_uses_the_controller_stalled_rule(tmp_path):
    def budget(paths):
        return action(receipt("move_pose", status="not_reached", code="native_action_budget_exhausted"),
                      [tick("move", control(right=_vec(2.0)), right=moving("right", path)) for path in paths])

    s0 = joint_ramp(ZERO, _vec(0.01))
    s1, s2, s3 = ([_vec(s0[-1][0] + d)] * 10 for d in (0.003, 0.005, 0.006))
    m0 = joint_ramp(s3[-1], _vec(0.005))
    m1 = joint_ramp(m0[-1], _vec(0.005))
    m2 = joint_ramp(m1[-1], _vec(0.005))
    round0 = [
        action(receipt("set_gripper"), [_gripper_tick()]),
        budget([s0, s1, s2, s3]),  # last three tick-to-tick steps: 0.003, 0.002, 0.001
        budget([m0, m1, m2]),
        budget([[m2[-1]] * 10, [m2[-1]] * 10]),  # two ticks cannot show three still ones
        action(receipt("recover_motion", status="not_reached", code="native_action_budget_exhausted"),
               [tick("recover", control(right=ZERO), right=still("right", m2[-1]))] * 3),
    ]
    result = _metrics(write_native(tmp_path / "run", [round0]))["timeout_stillness"]
    assert (result["numerator"], result["denominator"]) == (1, 3)


def test_tracking_error_uses_the_recorded_endpoint_for_v1_and_waypoints_when_planned(tmp_path):
    ends = [_vec(0.11), _vec(0.25)]
    ticks = [tick("move", control(right=_vec(1.0)), right=still("right", q)) for q in ends]
    run = write_native(tmp_path / "run", [[action(receipt("move_pose"), ticks)]])
    v1 = _metrics(run)
    assert v1["tracking_error"]["waypoint_source"] == "recorded_control_target"
    assert (v1["tracking_error"]["numerator"], v1["tracking_error"]["denominator"]) == (2, 2)
    assert v1["tracking_error"]["error_rad"]["max"] == pytest.approx(0.89)
    assert v1["peak_ratio"]["status"] == v1["impulse_saturation"]["status"] == "not_applicable"
    planned = tmp_path / "planned.json"
    planned.write_text(json.dumps({
        "schema": metrics.PLANNED_WAYPOINTS_SCHEMA,
        "actions": [{"action_sequence": 0, "arm": "right", "positions": [_vec(0.1), _vec(0.2)],
                     "predicted_in_tick_speed_rad_s": [3.0, 3.0]}],
    }), encoding="utf-8")
    v2 = _metrics(run, planned=metrics.load_planned_waypoints(planned))
    assert v2["tracking_error"]["waypoint_source"] == "planned_waypoints"
    assert (v2["tracking_error"]["numerator"], v2["tracking_error"]["denominator"]) == (1, 2)
    assert v2["tracking_error"]["error_rad"]["max"] == pytest.approx(0.05)
    assert (v2["peak_ratio"]["numerator"], v2["peak_ratio"]["denominator"]) == (1, 1)
    assert v2["peak_ratio"]["ratio"]["max"] == pytest.approx(14.0)
    assert (v2["impulse_saturation"]["numerator"], v2["impulse_saturation"]["denominator"]) == (1, 2)


@pytest.mark.parametrize("damage, message", [
    ("truncate", "10 physics lines"), ("half_line", "cannot read"), ("action_id", "action_id"),
    ("native_action", "native actions"),
])
def test_journal_loader_refuses_inconsistent_controller_files(tmp_path, damage, message):
    run = write_native(tmp_path / "run", [[action(receipt("move_pose"), [tick("move", control(right=_vec(0.1)))])]])
    controller = run / "native" / "controller"
    lines = (controller / "action-0000.jsonl").read_text().splitlines(keepends=True)
    trace = json.loads((controller / "action-0000.json").read_text())
    if damage == "truncate":
        (controller / "action-0000.jsonl").write_text("".join(lines[:9]))
    elif damage == "half_line":
        (controller / "action-0000.jsonl").write_text("".join(lines[:9]) + lines[9][:40])
    elif damage == "action_id":
        (controller / "action-0000.json").write_text(json.dumps({**trace, "action_id": "other"}))
    else:
        trace["samples"][0]["native_action"] = 7
        (controller / "action-0000.json").write_text(json.dumps(trace))
    with pytest.raises(JournalError, match=message):
        load_run_journal(run)


def test_journal_root_override_reads_replayed_traces_with_the_original_receipts(tmp_path):
    run = write_native(tmp_path / "run", [[
        action(receipt("move_pose"), [tick("move", control(right=_vec(0.1)))]),
        action(receipt("move_pose"), [tick("move", control(right=_vec(0.2)))]),
    ]])
    replay = tmp_path / "replay"
    (replay / "controller").mkdir(parents=True)
    for suffix in (".json", ".jsonl"):
        source = run / "native" / "controller" / f"action-0000{suffix}"
        (replay / "controller" / source.name).write_bytes(source.read_bytes())
    journal = load_run_journal(run, journal_root=replay)
    assert [item.sequence for item in journal.actions] == [0]
    assert load_run_journal(run, upto=1).actions[-1].sequence == 0


def test_segment_classes_and_the_pinned_percentile_rule():
    assert [segment_class(name) for name in ("move", "plan_ref", "precontact", "retract", "recover_fold",
                                              "local", "gripper", "wait", "stall_hold")] == [
        "global", "global", "global", "global", "global", "local", None, None, None]
    assert summarize_values([4, 1, 3, 2]) == {"count": 4, "mean": 2.5, "p50": 3, "p90": 4, "max": 4}
    assert summarize_values([]) == {"count": 0, "mean": None, "p50": None, "p90": None, "max": None}


def test_definitions_are_pinned_to_their_sources():
    from ora.agent.tools.arx_x5_gripper import ARX_X5_GRIPPER_GEOMETRY
    from ora.contracts.robodojo_arx import ARX_X5_CONTRACT
    from ora.environment.backends.robodojo import motion_stall
    from ora.environment.benchmarks.robodojo import PHYSICS_DT_S

    assert measures.PHYSICS_DT_S == PHYSICS_DT_S == ARX_X5_CONTRACT["physics_dt_s"]
    assert measures.NATIVE_ACTION_S == pytest.approx(1.0 / ARX_X5_CONTRACT["control_hz"])
    assert measures.SATURATED_JOINT_SPEED_RAD_S == pytest.approx(4.75)
    assert measures.SATURATED_TICK_DISPLACEMENT_RAD == pytest.approx(0.19)
    assert measures.STILL_JOINT_STEP_RAD == motion_stall.STALL_JOINT_STEP_RAD == 0.005
    assert measures.STILL_WINDOW_NATIVE_ACTIONS == motion_stall.STALL_WINDOW_NATIVE_ACTIONS == 3
    assert measures.POSITION_TOLERANCE_M == ARX_X5_CONTRACT["position_tolerance_m"]
    assert measures.ORIENTATION_TOLERANCE_RAD == ARX_X5_CONTRACT["orientation_tolerance_rad"]
    assert measures.GRIPPER_TOLERANCE == ARX_X5_CONTRACT["gripper_tolerance_normalized"]
    assert measures.JAW_CONTACT_CENTRE_LINK6_X_M == ARX_X5_GRIPPER_GEOMETRY["contact_baseline_link6_x_m"]
    assert (measures.HELD_OBJECT_DISPLACEMENT_M, measures.OUT_OF_STEP_MINIMUM_EXCURSION_RAD) == (0.02, 0.05)
    assert (measures.SAG_MARGIN_M, measures.V_CMD_CANDIDATE_RAD_S, measures.TRACKING_TOLERANCE_RAD) == (0.02, 4.0, 0.03)
    assert metrics.DEFINITIONS["version"] == metrics.METRICS_VERSION == measures.METRICS_VERSION


def test_cli_pools_runs_records_a_malformed_run_and_refuses_to_write_inside_a_run(tmp_path, capsys):
    ticks = [tick("move", control(right=_vec(1.0)), right=moving("right", joint_ramp(ZERO, _vec(0.02))))]
    first = write_native(tmp_path / "a", [[action(receipt("move_pose"), ticks)]])
    second = write_native(tmp_path / "b", [[action(receipt("move_pose"), ticks)]])
    output = tmp_path / "report.json"
    assert metrics.main([str(first), str(second / "native"), "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["schema"] == metrics.REPORT_SCHEMA
    assert report["metric_definition_version"] == metrics.METRICS_VERSION
    assert [run["run_id"] for run in report["runs"]] == ["a", "b"]
    pooled = report["pooled"]["metrics"]["v1_baseline_diagnostics"]["saturation_physics_step"]
    assert (pooled["numerator"], pooled["denominator"]) == (2, 2)
    journal = second / "native" / "controller" / "action-0000.jsonl"
    journal.write_text(journal.read_text()[:-20], encoding="utf-8")
    assert metrics.main([str(first), str(second), "--output", str(output)]) == 1
    report = json.loads(output.read_text())
    assert [entry["run_id"] for entry in report["errors"]] == ["b"] and report["pooled"]["runs"] == 1
    with pytest.raises(SystemExit):
        metrics.main([str(first), "--output", str(first / "native" / "metrics.json")])
    capsys.readouterr()
