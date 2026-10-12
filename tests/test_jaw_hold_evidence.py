"""The RoboDojo verify stage reports a hold signal from jaw-stall evidence (EXP11A g003, B2).

All ten completed g003 ``grasp_execute`` verify stages reported
``gripper_opening_m: null`` and ``closure_strength: unavailable``: the ARX pad
aperture is uncalibrated, so no metric jaw separation exists. Their close
stages had measured the question anyway. The recorded close outcomes below are
from run ``f3a0e6751511`` (stalled at 0.823, held drive target 0.623, 0.822
after the lift) and ``7005abe3a69b`` (stalled at 0.594, 0.490 after the lift).

EXP11A g007 stack_bowls (run ``849f55dccacd``) adds the false positive: at
+31.7 min the left close stalled after 3 native actions at 0.99999999 -- the
jaws never left the open position, fingers at their 44 mm stop -- and 0.9972
after the lift was read as "still held apart". At +38.8 min the right close
stalled mid-travel at 0.225 and read 0.234 after the lift, a genuine stall
that stays ``likely``.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from ora.agent.tools import jaw_hold, skill_driver

TOLERANCE = 0.03
F3A0 = {"state": "stalled_before_target", "commanded_opening_normalized": 0.0,
        "effective_target_opening_normalized": 0.1851851851851852,
        "measured_opening_normalized": 0.8227210188353503,
        "sdk_limited_drive_target_normalized": 0.622751924947456, "ticks_used": 5, "ticks_budget": 20}
SEVEN = {**F3A0, "measured_opening_normalized": 0.594, "sdk_limited_drive_target_normalized": 0.394}
EMPTY = {**F3A0, "state": "reached_target", "measured_opening_normalized": 0.1851851851851852,
         "sdk_limited_drive_target_normalized": 0.0}
#: g007 +31.7 min, left arm: the recorded open and close outcomes.
G007_OPEN = {"state": "reached_target", "commanded_opening_normalized": 1.0,
             "effective_target_opening_normalized": 1.0, "measured_opening_normalized": 0.9771581021723924,
             "sdk_limited_drive_target_normalized": 1.0, "ticks_used": 8, "ticks_budget": 20}
G007_NEVER_CLOSED = {"state": "stalled_before_target", "commanded_opening_normalized": 0.0,
                     "effective_target_opening_normalized": 0.1851851851851852,
                     "measured_opening_normalized": 0.9999999955848412,
                     "sdk_limited_drive_target_normalized": 0.7999999955848411, "ticks_used": 3,
                     "ticks_budget": 20}
G007_NEVER_CLOSED_AFTER_LIFT = 0.9971917475815173
#: g007 +38.8 min, right arm: a genuine mid-travel stall.
G007_HELD = {**G007_NEVER_CLOSED, "measured_opening_normalized": 0.2250370817879836,
             "sdk_limited_drive_target_normalized": 0.025483017145759523, "ticks_used": 9}


@pytest.mark.parametrize("outcome,after,held,reason", [
    (F3A0, 0.8216556547968477, "likely", "jaws_still_held_apart_at_the_close_opening"),
    (SEVEN, 0.49, "unknown", "jaws_closed_further_during_lift"),
    (F3A0, 0.62, "unlikely", "jaws_reached_the_held_drive_target_during_lift"),
    (F3A0, 0.2, "unlikely", "jaws_closed_to_the_floor_during_lift"),
    (EMPTY, 0.1851851851851852, "unlikely", "close_reached_the_closed_floor"),
    ({**F3A0, "state": "moving_at_budget"}, 0.8, "unknown", "close_ended_moving_at_budget"),
    (None, 0.8, "unknown", "close_outcome_unavailable"),
    (F3A0, None, "unknown", "after_lift_opening_unavailable"),
    (G007_NEVER_CLOSED, G007_NEVER_CLOSED_AFTER_LIFT, "unlikely", "jaws_never_closed"),
    (G007_NEVER_CLOSED, None, "unlikely", "jaws_never_closed"),
    (G007_HELD, 0.23371460226674876, "likely", "jaws_still_held_apart_at_the_close_opening"),
    ({**F3A0, "measured_opening_normalized": 0.969}, 0.969, "likely",
     "jaws_still_held_apart_at_the_close_opening"),
])
def test_the_recorded_and_boundary_cases_classify_honestly(outcome, after, held, reason):
    result = jaw_hold.hold_evidence(outcome=outcome, after_lift=after, tolerance=TOLERANCE)
    assert (result["held"], result["reason"]) == (held, reason)
    assert result["evidence"]["tolerance_normalized"] == TOLERANCE
    assert "aperture" in result["claim_boundary"] and "retention" in result["claim_boundary"]


def test_a_missing_arrival_tolerance_is_unknown_not_guessed():
    result = jaw_hold.hold_evidence(outcome=F3A0, after_lift=0.82, tolerance=None)
    assert (result["held"], result["reason"]) == ("unknown", "arrival_tolerance_unavailable")


def _dojo_record(opening: float) -> dict:
    return {"pose_frame": "robodojo_episode_local_ee_link", "gripper": opening,
            "gripper_finger_qpos": [0.03, 0.03],
            "gripper_feedback": {"measured_opening_normalized": opening,
                                 "arrival_check": {"comparison": "<=", "tolerance_normalized": TOLERANCE}}}


def _verify(supported: bool, *, completed):
    host = SimpleNamespace(surface=SimpleNamespace(supports=lambda name: supported and name == jaw_hold.CAPABILITY))
    observation = {"observation_id": "robodojo:x:observation-0002",
                   "proprioception": {"end_effectors": {"left": _dojo_record(0.8216556547968477)}}}
    return skill_driver._measure(host, "verify", observation, {"arm": "left", "localization": None}, completed)


CLOSE_STAGE = [{"stage": "close", "status": "executed", "reason": "gripper_stalled_before_target",
                "measurements": [{"quantity": "gripper_outcome", "value": {"left": F3A0}}]}]


def test_the_declaring_surface_adds_the_hold_signal_beside_the_unavailable_strength():
    measured = _verify(True, completed=CLOSE_STAGE)
    quantities = {item["quantity"]: item["value"] for item in measured["measurements"]}
    assert quantities["closure_strength"] == "unavailable"
    assert quantities["gripper_opening_m"] is None  # still no metric aperture claimed
    assert quantities["jaw_hold"] == "likely"
    assert measured["verification"]["jaw_hold"]["evidence"]["close_opening_normalized"] == pytest.approx(0.8227, abs=1e-4)


def test_other_surfaces_keep_the_released_verify_stage():
    measured = _verify(False, completed=CLOSE_STAGE)
    assert "jaw_hold" not in {item["quantity"] for item in measured["measurements"]}
    assert "jaw_hold" not in measured["verification"]


def test_a_close_that_never_left_the_open_position_is_not_a_hold():
    """The g007 +31.7 false positive, through the verify stage with its open stage."""
    completed = [
        {"stage": "open", "status": "executed", "reason": "executed",
         "measurements": [{"quantity": "gripper_outcome", "value": {"left": G007_OPEN}}]},
        {"stage": "close", "status": "executed", "reason": "gripper_stalled_before_target",
         "measurements": [{"quantity": "gripper_outcome", "value": {"left": G007_NEVER_CLOSED}}]},
    ]
    measured = _verify(True, completed=completed)
    quantities = {item["quantity"]: item["value"] for item in measured["measurements"]}
    assert quantities["jaw_hold"] == "unlikely"
    hold = measured["verification"]["jaw_hold"]
    assert hold["reason"] == "jaws_never_closed"
    assert hold["evidence"]["open_position_normalized"] == 1.0
    assert hold["evidence"]["close_opening_normalized"] == pytest.approx(1.0, abs=1e-6)


def test_the_open_position_is_the_open_stage_s_command_else_the_open_directive():
    partial = {**G007_OPEN, "commanded_opening_normalized": 0.6}
    completed = [{"stage": "open", "measurements": [{"quantity": "gripper_outcome",
                                                     "value": {"left": partial}}]}]
    assert jaw_hold.open_position(completed, "left") == 0.6
    assert jaw_hold.open_position([], "left") == jaw_hold.OPEN_POSITION_NORMALIZED == 1.0
    result = jaw_hold.hold_evidence(outcome={**G007_NEVER_CLOSED, "measured_opening_normalized": 0.59},
                                    after_lift=0.59, tolerance=TOLERANCE, opened=0.6)
    assert (result["held"], result["reason"]) == ("unlikely", "jaws_never_closed")
