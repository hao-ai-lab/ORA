"""A held/not-held signal from the jaw-stall evidence a backend already reports.

``closure_evidence`` compares a metric jaw separation with the target's visible
extents. RoboDojo publishes no metric separation: the ARX pad aperture is
uncalibrated (``localization.measured_jaw_separation_m`` returns ``None`` on
purpose), so every verify stage of EXP11A g003's ten completed
``grasp_execute`` calls reported ``closure_strength: unavailable`` while the
close stage beside it had measured exactly the fact in question.

Under the ARX drive limiter a jaw that closes on something stops short of the
closed floor (``stalled_before_target``) and holds a drive target 0.2 below
that encoder reading; a jaw that closes on nothing reaches the floor; a jaw
whose object leaves during the lift travels down to the held drive target.
This module turns those normalized encoder readings into ``likely``,
``unlikely`` or ``unknown``, and returns the readings next to the thresholds.
It never converts them to a width, and none of its answers is contact,
attachment or retention.

A close that stalls without leaving the open position is not the first case:
jaws that never moved were stopped by something outside them (a fingertip on
the rim, the table) or never driven, and nothing between them pushes the pads
past fully open. EXP11A g007 stack_bowls (+31.7 min) stalled at 0.99999999
after 3 native actions -- the fingers at their 44 mm stop -- and the released
rule read the unchanged reading after the lift as "still held apart". Such a
close is ``unlikely`` with reason ``jaws_never_closed``: its reading lies
within the arrival tolerance of the opening the open stage commanded, the band
that stage itself counted as open. On the X5 no admitted candidate needs that
band: the 80 mm admission ceiling is a normalized opening of 0.918.

The readings are the two-finger mean the backend publishes (EXP11A g012
E11G12-001). The rules above assume the fingers move as a symmetric pair; a pair
shifted to one side says nothing about holding either way (g012: one finger at
its 44 mm stop, the other 30 mm in, read as "fully open"). When the close or
after-lift reading has a lateral finger offset beyond ``SYMMETRIC_OFFSET_BOUND_M``
the verdict is ``unknown`` with reason ``finger_offset_exceeds_symmetric_bound``.
Only the readings a verdict compares count: a close that reached the floor or
never left the open position is judged on the close reading alone.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

CAPABILITY = "gripper_stall_hold_evidence_v1"
SCHEMA = "ora.jaw-stall-hold-evidence.v1"
CLAIM_BOUNDARY = (
    "normalized jaw-encoder readings under the backend drive limiter; not a calibrated aperture, "
    "contact, attachment, retention or task-success verdict"
)


#: The open directive of a committed grasp (``skill_driver.GRIPPER_OPEN``),
#: used when the sequence carries no open stage of its own.
OPEN_POSITION_NORMALIZED = 1.0
#: Largest |finger_lateral_offset_m| (half the finger difference) read as a
#: symmetric pair. Recorded ARX closes: build_tower's 252 control samples keep
#: the fingers within 2 mm in free air and 6 mm on an object (3 mm offset); a
#: close the SDK primary-relative limit holds with one finger pinned ends up to
#: 0.2 x 54 mm = 10.8 mm apart (5.4 mm); the 14 such g012 close stalls stood
#: 9.3-10.0 mm apart, and 96 of all 124 g012 close stalls within 6 mm. 4 mm
#: separates the two.
SYMMETRIC_OFFSET_BOUND_M = 0.004
#: Verdicts that compare the close reading alone (not the after-lift one).
_CLOSE_ONLY_VERDICTS = {"close_reached_the_closed_floor", "jaws_never_closed"}


def _number(value: Any) -> float | None:
    if type(value) in (int, float) and math.isfinite(float(value)):
        return float(value)
    return None


def _stage_outcome(completed: Sequence[Mapping[str, Any]], arm: str, name: str) -> Mapping[str, Any] | None:
    for stage in reversed(list(completed)):
        if stage.get("stage") != name:
            continue
        for item in stage.get("measurements", ()):
            value = item.get("value") if isinstance(item, Mapping) else None
            if item.get("quantity") == "gripper_outcome" and isinstance(value, Mapping):
                outcome = value.get(arm)
                return outcome if isinstance(outcome, Mapping) else None
    return None


def close_outcome(completed: Sequence[Mapping[str, Any]], arm: str) -> Mapping[str, Any] | None:
    """The committed close stage's per-arm ``gripper_outcome`` measurement, if any."""
    return _stage_outcome(completed, arm, "close")


def open_position(completed: Sequence[Mapping[str, Any]], arm: str) -> float:
    """The opening the sequence's open stage commanded, else the grasp open directive."""
    commanded = _number((_stage_outcome(completed, arm, "open") or {}).get("commanded_opening_normalized"))
    return OPEN_POSITION_NORMALIZED if commanded is None else commanded


def after_lift_opening(record: Any) -> float | None:
    if not isinstance(record, Mapping):
        return None
    feedback = record.get("gripper_feedback")
    if isinstance(feedback, Mapping):
        return _number(feedback.get("measured_opening_normalized"))
    return _number(record.get("gripper"))


def after_lift_offset(record: Any) -> float | None:
    """The post-lift finger pair's lateral offset, when the typed feedback carries it."""
    feedback = record.get("gripper_feedback") if isinstance(record, Mapping) else None
    return _number(feedback.get("finger_lateral_offset_m")) if isinstance(feedback, Mapping) else None


def hold_evidence(*, outcome: Mapping[str, Any] | None, after_lift: float | None,
                  tolerance: float | None, opened: float | None = OPEN_POSITION_NORMALIZED,
                  after_lift_offset: float | None = None) -> dict[str, Any]:
    """Classify one arm's close-then-lift jaw readings; unknown whenever evidence is missing."""
    state = outcome.get("state") if isinstance(outcome, Mapping) else None
    stall = _number((outcome or {}).get("measured_opening_normalized"))
    floor = _number((outcome or {}).get("effective_target_opening_normalized"))
    drive = _number((outcome or {}).get("sdk_limited_drive_target_normalized"))
    close_offset = _number((outcome or {}).get("finger_lateral_offset_m"))
    opened = _number(opened)
    evidence = {
        "close_state": state, "close_opening_normalized": stall, "closed_floor_normalized": floor,
        "held_drive_target_normalized": drive, "after_lift_opening_normalized": after_lift,
        "change_during_lift_normalized": (None if stall is None or after_lift is None else after_lift - stall),
        "tolerance_normalized": tolerance, "open_position_normalized": opened,
        "close_finger_lateral_offset_m": close_offset, "after_lift_finger_lateral_offset_m": after_lift_offset,
        "symmetric_offset_bound_m": SYMMETRIC_OFFSET_BOUND_M,
    }
    held, reason = "unknown", "close_outcome_unavailable"
    if _number(tolerance) is None:
        reason = "arrival_tolerance_unavailable"
    elif state == "reached_target":
        held, reason = "unlikely", "close_reached_the_closed_floor"
    elif state == "stalled_before_target" and stall is not None and floor is not None:
        if opened is not None and stall >= opened - tolerance:
            held, reason = "unlikely", "jaws_never_closed"
        elif after_lift is None:
            reason = "after_lift_opening_unavailable"
        elif after_lift <= floor + tolerance:
            held, reason = "unlikely", "jaws_closed_to_the_floor_during_lift"
        elif drive is not None and after_lift <= drive + tolerance:
            held, reason = "unlikely", "jaws_reached_the_held_drive_target_during_lift"
        elif after_lift >= stall - tolerance:
            held, reason = "likely", "jaws_still_held_apart_at_the_close_opening"
        else:
            reason = "jaws_closed_further_during_lift"
    elif state is not None:
        reason = f"close_ended_{state}"
    compared = (close_offset,) if reason in _CLOSE_ONLY_VERDICTS else (close_offset, after_lift_offset)
    if held != "unknown" and any(value is not None and abs(value) > SYMMETRIC_OFFSET_BOUND_M for value in compared):
        held, reason = "unknown", "finger_offset_exceeds_symmetric_bound"
    return {"schema": SCHEMA, "held": held, "reason": reason, "evidence": evidence,
            "claim_boundary": CLAIM_BOUNDARY}


def verify_stage_evidence(completed: Sequence[Mapping[str, Any]], arm: str, record: Any) -> dict[str, Any]:
    """The verify stage's hold evidence for ``arm`` from its close stage and the post-lift record."""
    feedback = record.get("gripper_feedback") if isinstance(record, Mapping) else None
    check = feedback.get("arrival_check") if isinstance(feedback, Mapping) else None
    return hold_evidence(outcome=close_outcome(completed, arm), after_lift=after_lift_opening(record),
                         tolerance=check.get("tolerance_normalized") if isinstance(check, Mapping) else None,
                         opened=open_position(completed, arm), after_lift_offset=after_lift_offset(record))


__all__ = ["CAPABILITY", "CLAIM_BOUNDARY", "OPEN_POSITION_NORMALIZED", "SCHEMA", "SYMMETRIC_OFFSET_BOUND_M",
           "after_lift_offset", "after_lift_opening", "close_outcome", "hold_evidence", "open_position",
           "verify_stage_evidence"]
