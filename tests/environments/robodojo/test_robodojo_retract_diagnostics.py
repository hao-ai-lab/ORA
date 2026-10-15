"""A skipped optional retract is not the arm's planning outcome (E11G7-001).

A home ``recover_pose`` first tries an optional retract lift. When that IK
solve fails the kernel skips the lift, still copies the solve's diagnostic to
the arm (``arms.<arm>.planning_diagnostics`` is the arm's last IK attempt,
whatever its segment), and the unified sparse projector delivers it. A recovery
that reached home therefore told the policy ``no_valid_endpoint`` and "try an
alternative", although the policy chose no target and the skip is already
stated by ``recovery_progress.retract``.

``per_arm_realized_translation_v1`` (declared only by the next surface
generation) omits an arm's diagnostic when its last IK attempt belongs to a
skipped retract; surfaces without it keep the recorded bytes.

The receipts are trimmed copies of recorded EXP11B receipts
(``tests/data/robodojo/retract-diagnostic-receipts.json``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.support.delivery_lessons_fixtures import project
from ora.agent.native.receipt_projection import (
    CONTACT_MOTION_FEEDBACK_CAPABILITY, PER_ARM_TRANSLATION_CAPABILITY,
)

DATA = Path(__file__).resolve().parents[2] / "data/robodojo/retract-diagnostic-receipts.json"
RECEIPTS = json.loads(DATA.read_text(encoding="utf-8"))["receipts"]
SKIPPED = RECEIPTS["action-597e9a02045cd4d2"]  # right arm, reached, retract ik_failed
PLANNED = RECEIPTS["action-0375d9a8ef21b00b"]  # left arm, reached, retract planned
MIXED = RECEIPTS["action-98471ac0e2820cc4"]  # left planned, right skipped
NOT_REACHED = RECEIPTS["action-ee3f26dfdb8da8c4"]  # right arm, not_reached, retract ik_failed
JOINT_JUMP = RECEIPTS["action-720050a0e440a199"]  # left arm, retract ik_branch_jump
MOVE_REFUSED = RECEIPTS["action-ea1927b02c860e75"]  # move_pose, required segment
LEGACY = (CONTACT_MOTION_FEEDBACK_CAPABILITY,)
FIXED = (CONTACT_MOTION_FEEDBACK_CAPABILITY, PER_ARM_TRANSLATION_CAPABILITY)


def arms(record, capabilities):
    return project(record, capabilities, public_name="recover_pose")["detail"]["arms"]


def test_without_the_capability_a_reached_recovery_still_carries_the_retract_diagnostic():
    effect = project(SKIPPED, LEGACY, public_name="recover_pose")
    assert effect["status"] == "reached"
    right = effect["detail"]["arms"]["right"]
    # The recorded EXP11B effect: home reached, yet "try an alternative".
    assert right["planning_diagnostics"]["outcome"] == "no_valid_endpoint"
    assert "try an alternative" in right["planning_diagnostics"]["recovery"][0]
    assert right["recovery_progress"]["retract"]["status"] == "skipped"


@pytest.mark.parametrize("record,arm,reason", [
    (SKIPPED, "right", "ik_failed"),
    (NOT_REACHED, "right", "ik_failed"),
    (JOINT_JUMP, "left", "ik_branch_jump"),
])
def test_a_skipped_retract_diagnostic_leaves_the_arm(record, arm, reason):
    assert [a["segment"] for a in record["ik_attempts"] if a["arm"] == arm][-1] == "retract"
    legacy, fixed = arms(record, LEGACY), arms(record, FIXED)
    assert "planning_diagnostics" in legacy[arm]
    assert "planning_diagnostics" not in fixed[arm]
    # Nothing else changes: the skip is still stated where it belongs.
    bare, alone = arms(record, ()), arms(record, (PER_ARM_TRANSLATION_CAPABILITY,))
    assert alone[arm] == {key: value for key, value in bare[arm].items()
                          if key != "planning_diagnostics"}
    assert fixed[arm]["recovery_progress"]["retract"]["status"] == "skipped"
    assert fixed[arm]["recovery_progress"]["retract"]["reason"] == reason


def test_a_planned_retract_keeps_its_admitted_diagnostic():
    assert PLANNED["arms"]["left"]["recovery_progress"]["retract"]["status"] == "planned"
    fixed = arms(PLANNED, FIXED)["left"]
    assert fixed["planning_diagnostics"] == arms(PLANNED, LEGACY)["left"]["planning_diagnostics"]
    assert fixed["planning_diagnostics"]["outcome"] == "admitted"


def test_each_arm_of_a_both_arm_recovery_is_judged_by_its_own_retract():
    legacy, fixed = arms(MIXED, LEGACY), arms(MIXED, FIXED)
    assert fixed["left"]["planning_diagnostics"] == legacy["left"]["planning_diagnostics"]
    assert legacy["right"]["planning_diagnostics"]["outcome"] == "no_valid_endpoint"
    assert "planning_diagnostics" not in fixed["right"]


def test_a_refused_move_keeps_its_diagnostic():
    effect = project(MOVE_REFUSED, FIXED)
    assert effect["status"] == "planner_failed"
    left = effect["detail"]["arms"]["left"]
    assert left["planning_diagnostics"] == project(MOVE_REFUSED, LEGACY)["detail"]["arms"]["left"][
        "planning_diagnostics"]
    assert left["planning_diagnostics"]["outcome"] == "no_valid_endpoint"


def test_the_retract_rule_does_not_need_contact_feedback():
    # Planning diagnostics are delivered without contact feedback, so their
    # attribution is too; the translations still need it.
    alone = arms(SKIPPED, (PER_ARM_TRANSLATION_CAPABILITY,))["right"]
    assert "planning_diagnostics" not in alone
    assert "realized_translation_world_m" not in alone
    assert "planning_diagnostics" in arms(SKIPPED, ())["right"]
