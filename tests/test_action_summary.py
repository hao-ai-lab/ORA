"""D1: one plain-language line heads the last-action result (``last_action_summary_v1``).

The line restates facts the effect already carries -- distance from target,
the outcome and its code, whether the arm moved -- and never adds a hint.
The full detail follows it unchanged.
"""

from __future__ import annotations

import json
import math
import re

import pytest

from delivery_lessons_fixtures import (
    DEGREES, DEDUP, SUMMARY, dojo_gripper_feedback, libero_hybrid_receipt, libero_receipt, project,
    robodojo_receipt, robotwin_receipt, skill_receipt,
)
from ora.agent.native.action_summary import CAPABILITY, action_summary

FACTS = ("public_motion_effect_v1", "current_public_robot_facts_v1", "contact_motion_feedback_v1")


def _stalled_gripper():
    feedback = {"left": dojo_gripper_feedback(measured=0.41, requested=0.0),
                "right": dojo_gripper_feedback()}
    return robodojo_receipt(action="set_gripper", status="not_reached", stage="execution",
                            code="gripper_stalled_before_target", pose_error=None,
                            native_actions=20, feedback=feedback, gripper_check=0.41)


CASES = [
    ("robotwin refusal", robotwin_receipt(), "move_pose",
     "left arm 104 mm / 120° from target, plan rejected: inter arm collision, not moved"),
    ("robodojo arrival", robodojo_receipt(), None,
     "left arm 8.7 mm / 1.6° from target, reached, ran 9 of 20 native actions"),
    ("robodojo planner failure",
     robodojo_receipt(status="planner_failed", stage="planner", code="planner_failed",
                      pose_error=(0.3558, 0.5), native_actions=0), None,
     "left arm 356 mm / 29° from target, planner failed, not moved"),
    ("robodojo stalled gripper", _stalled_gripper(), None,
     "left gripper opening 0.41 / target 0.00, not reached: gripper stalled before target, "
     "ran 20 of 20 native actions"),
    ("libero arrival", libero_receipt(), None,
     "right arm 3.3 mm / 0.4° from target, reached, TCP moved 50 mm"),
    ("libero hybrid gripper", libero_hybrid_receipt(), None,
     "right gripper 0.03 from target opening, reached, TCP moved 0.2 mm"),
    ("closed grasp skill", skill_receipt(), None,
     "left arm 188 mm / 150° from target, aborted at pregrasp: IK FAIL, 2 of 6 stages attempted"),
]


def test_capability_name_is_the_one_the_fixtures_declare():
    assert CAPABILITY == SUMMARY


@pytest.mark.parametrize(("label", "receipt", "public_name", "expected"), CASES,
                         ids=[case[0] for case in CASES])
def test_summary_restates_distance_outcome_and_motion(label, receipt, public_name, expected):
    effect = project(receipt, (SUMMARY, *FACTS), public_name=public_name)
    assert effect["action_summary"] == expected


def test_a_left_arm_move_never_reports_the_right_arm_translation():
    """The shared contact detail reads the right arm; a left move must not cite it."""
    receipt = robodojo_receipt()
    receipt["after"]["right"]["pose_xyz_xyzw"] = [0.5, -0.35, 0.92, 0.0, 0.0, 0.7071, 0.7071]
    summary = project(receipt, (SUMMARY, *FACTS))["action_summary"]
    assert "TCP moved" not in summary


def test_a_terminated_round_is_named_last():
    receipt = robodojo_receipt()
    receipt["terminal_after_action"] = True
    assert project(receipt, (SUMMARY, *FACTS))["action_summary"].endswith(", round ended")


def _numbers(value, found):
    if isinstance(value, bool) or value is None:
        return found
    if isinstance(value, (int, float)):
        for rendered in (f"{value * 1000:.0f}", f"{value * 1000:.1f}", f"{value:.2f}",
                         f"{math.degrees(value):.0f}", f"{math.degrees(value):.1f}"):
            found.add(rendered)
        if isinstance(value, int):
            found.add(str(value))
    elif isinstance(value, dict):
        for child in value.values():
            _numbers(child, found)
    elif isinstance(value, list):
        # A count of listed items, and the length of a listed 3-vector, restate it.
        found.add(str(len(value)))
        if len(value) == 3 and all(isinstance(item, (int, float)) for item in value):
            _numbers(math.dist(value, (0.0, 0.0, 0.0)), found)
        for child in value:
            _numbers(child, found)
    return found


@pytest.mark.parametrize(("label", "receipt", "public_name", "expected"), CASES,
                         ids=[case[0] for case in CASES])
def test_summary_adds_no_number_and_no_advice(label, receipt, public_name, expected):
    """Every figure in the line is a rounding of a value the effect already carries."""
    legacy = project(receipt, FACTS, public_name=public_name)
    summary = action_summary(legacy)
    assert summary == expected
    derivable = _numbers(legacy, set())
    for number in re.findall(r"\d+(?:\.\d+)?", summary):
        assert number in derivable, (number, summary)
    for word in ("try", "should", "retry", "consider", "instead", "recommend", "please",
                 "must", "next", "advice"):
        assert not re.search(rf"\b{word}\b", summary, re.IGNORECASE), word


def test_summary_heads_the_effect_and_leaves_the_detail_alone():
    legacy = project(robodojo_receipt(), FACTS)
    delivered = project(robodojo_receipt(), (SUMMARY, *FACTS))
    text = json.dumps(delivered, ensure_ascii=False, sort_keys=True)
    assert text.startswith('{"action": "move_pose", "action_summary": "left arm 8.7 mm')
    assert {key: value for key, value in delivered.items() if key != "action_summary"} == legacy


def test_summary_is_born_in_degrees_whatever_else_is_declared():
    alone = project(robodojo_receipt(), (SUMMARY, *FACTS))["action_summary"]
    combined = project(robodojo_receipt(), (SUMMARY, DEGREES, DEDUP, *FACTS))["action_summary"]
    assert alone == combined


def test_no_summary_without_the_capability():
    assert "action_summary" not in project(robodojo_receipt(), FACTS)


def test_unknown_effect_shapes_get_no_line():
    assert action_summary({"action": "unknown", "status": "world_state_unknown"}) == (
        "world state unknown, motion unknown")
    assert action_summary({"kind": "servo"}) is None
    assert action_summary(None) is None
