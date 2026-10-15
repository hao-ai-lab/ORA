"""Each arm reports its own realized translation (E11BG6-004).

The unified sparse projector computed ``contact_motion.realized_translation_world_m``
from the right arm whatever the receipt targeted, a rule written for LIBERO's
single Panda arm. Every RoboDojo left-arm action therefore reported zero
translation. ``per_arm_realized_translation_v1`` (declared only by the next
surface generation) reports each targeted arm's own translation beside its
``pose_error``; surfaces without it keep the recorded bytes.

The receipts are trimmed copies of recorded EXP11B receipts
(``tests/data/robodojo/exp11b-arm-translation-receipts.json``).
"""

from __future__ import annotations

import json
import math
from pathlib import Path


from delivery_lessons_fixtures import project
from ora.agent.composition.profile_catalog import PROFILE_CATALOG
from ora.agent.composition.tool_surface import resolve_tool_surface
from ora.agent.native.action_summary import _millimetres, action_summary
from ora.agent.native.receipt_projection import (
    CONTACT_MOTION_FEEDBACK_CAPABILITY, PER_ARM_TRANSLATION_CAPABILITY,
)

DATA = Path(__file__).resolve().parent / "data/robodojo/exp11b-arm-translation-receipts.json"
RECEIPTS = json.loads(DATA.read_text(encoding="utf-8"))["receipts"]
LEFT_MOVE = RECEIPTS["action-2f0458f8a19b86cb"]
RIGHT_MOVE = RECEIPTS["action-477fa0b7a2f99666"]
BOTH_MOVE = RECEIPTS["action-373156b08ce03c24"]
LEGACY = (CONTACT_MOTION_FEEDBACK_CAPABILITY,)
FIXED = (CONTACT_MOTION_FEEDBACK_CAPABILITY, PER_ARM_TRANSLATION_CAPABILITY)


def displacement(record, arm):
    before = record["before"][arm]["pose_xyz_xyzw"][:3]
    after = record["after"][arm]["pose_xyz_xyzw"][:3]
    return [b - a for a, b in zip(before, after)]


def requested(record, arm):
    before = record["before"][arm]["pose_xyz_xyzw"][:3]
    target = record["arguments"]["targets"][arm]["position_m"]
    return [b - a for a, b in zip(before, target)]


def test_without_the_capability_a_left_move_still_reports_the_recorded_zero():
    effect = project(LEFT_MOVE, LEGACY)
    # The recorded EXP11B effect: the right arm stood still while the left
    # moved 230 mm.
    assert effect["detail"]["contact_motion"] == {"realized_translation_world_m": [0.0, 0.0, 0.0]}
    assert math.dist(displacement(LEFT_MOVE, "left"), (0, 0, 0)) > 0.2
    assert all("realized_translation_world_m" not in arm for arm in effect["detail"]["arms"].values())


def test_a_left_move_reports_the_left_arm_translation():
    effect = project(LEFT_MOVE, FIXED)
    left = effect["detail"]["arms"]["left"]
    assert left["realized_translation_world_m"] == displacement(LEFT_MOVE, "left")
    assert left["requested_translation_world_m"] == requested(LEFT_MOVE, "left")
    # Not targeted: its drift is inactive_arm_drift's business, not a translation.
    assert "realized_translation_world_m" not in effect["detail"]["arms"]["right"]
    # Nothing is left to say under contact_motion without force evidence.
    assert "contact_motion" not in effect["detail"]
    assert effect["physical_cause"] == "unknown"


def test_a_right_move_reports_the_same_numbers_it_always_did():
    legacy = project(RIGHT_MOVE, LEGACY)["detail"]["contact_motion"]
    fixed = project(RIGHT_MOVE, FIXED)["detail"]["arms"]["right"]
    assert fixed["realized_translation_world_m"] == legacy["realized_translation_world_m"]
    assert fixed["requested_translation_world_m"] == legacy["requested_translation_world_m"]


def test_a_both_arm_move_reports_each_arm():
    detail = project(BOTH_MOVE, FIXED)["detail"]
    for arm in ("left", "right"):
        assert detail["arms"][arm]["realized_translation_world_m"] == displacement(BOTH_MOVE, arm)
        assert detail["arms"][arm]["requested_translation_world_m"] == requested(BOTH_MOVE, arm)


def test_the_summary_line_cites_the_arm_that_moved():
    from delivery_lessons_fixtures import SUMMARY

    legacy = project(LEFT_MOVE, (*LEGACY, SUMMARY))["action_summary"]
    fixed = project(LEFT_MOVE, (*FIXED, SUMMARY))["action_summary"]
    assert "TCP moved" not in legacy
    assert fixed.endswith("TCP moved 230 mm")
    both = action_summary(project(BOTH_MOVE, FIXED))
    moved = [_millimetres(math.dist(displacement(BOTH_MOVE, arm), (0, 0, 0)))
             for arm in ("left", "right")]
    assert f"left TCP moved {moved[0]}, right TCP moved {moved[1]}" in both


def test_the_capability_needs_contact_feedback():
    effect = project(LEFT_MOVE, (PER_ARM_TRANSLATION_CAPABILITY,))
    assert "contact_motion" not in effect["detail"]
    assert all("realized_translation_world_m" not in arm for arm in effect["detail"]["arms"].values())


def test_only_the_v2_defaults_declare_it():
    """The v2 defaults (LIBERO's is its overhead successor), nothing frozen."""
    declaring = {profile for profile in PROFILE_CATALOG
                 if PER_ARM_TRANSLATION_CAPABILITY in resolve_tool_surface(profile).capabilities}

    assert declaring == {"robotwin_toolchain_b2_closed_noservo_v2", "libero_placement_direct_overhead_v1",
                         "robodojo_placement_closed_skills_v2"}
