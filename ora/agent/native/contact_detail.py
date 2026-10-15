"""The contact-motion detail of an action effect, and each arm's own translation.

:func:`contact_detail` is the shared detail behind ``contact_motion_feedback_v1``:
the measured force fields and the TCP translation. It was written for LIBERO's
single Panda arm, which the receipts name ``right``, and reads that arm whatever
was targeted. The unified sparse projector also serves RoboDojo, whose receipts
carry both arms, so ``contact_motion.realized_translation_world_m`` reported the
right arm for every action: a left-arm move read as zero translation.
Surfaces without the capability below keep those bytes.

With ``per_arm_realized_translation_v1``, each targeted arm's realized and requested translation
sit in ``detail.arms.<arm>`` beside its ``pose_error``, and ``contact_motion``
keeps only the measured force fields. Each value is the computation the shared
detail already made, applied to the arm that was targeted:

* realized -- the arm's measured TCP position after minus before;
* requested -- the commanded position minus before, from the segment's compiled
  pose for a single-arm receipt that has one, else the arm's ``position_m``
  argument.

Physical-cause estimation keeps its single-arm input: the targeted arm's
translation when exactly one arm was targeted, otherwise none.

The same capability attributes each arm's planning diagnostic to the segment
that decides its outcome. A RoboDojo receipt copies the arm's last IK attempt
to ``arms.<arm>.planning_diagnostics`` whatever its segment, so a home
``recover_pose`` whose optional retract lift was skipped after a failed solve
reported ``no_valid_endpoint`` and "try an alternative" beside a reached home.
:func:`without_skipped_retract_diagnostics` omits that diagnostic;
``recovery_progress.retract`` still states the skip and its reason. Planning
diagnostics are delivered without contact feedback, so this rule does not need
``contact_motion_feedback_v1``.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

from ora.agent.evidence.motion_feedback import _vector3

CAPABILITY = "per_arm_realized_translation_v1"
TRANSLATION_KEYS = ("realized_translation_world_m", "requested_translation_world_m")
#: The optional first segment of a home ``recover_pose`` (RoboDojo ``recovery.py``).
RETRACT_SEGMENT = "retract"
_FORCE_FIELDS = (
    "contact_estimated", "contact_force_threshold_n",
    "external_force_magnitude_n", "external_force_world_n",
    "external_force_direction_world", "force_frame",
    "force_direction_convention", "force_direction_status", "force_threshold_exceeded",
)


def contact_detail(record: Mapping[str, Any], *, translations: bool = True) -> dict[str, Any]:
    """Measured force fields, and the right arm's translation unless excluded."""
    segment = record.get("segment", record)
    if not isinstance(segment, Mapping):
        return {}
    feedback = segment.get("feedback")
    feedback = feedback if isinstance(feedback, Mapping) else {}
    result = {key: copy.deepcopy(feedback[key]) for key in _FORCE_FIELDS if key in feedback}
    if not translations:
        return result
    before = segment.get("before_pose_xyz_xyzw")
    after = segment.get("reached_pose_xyz_xyzw")
    for key, current in (("before", before), ("after", after)):
        nested = segment.get(key)
        arm = nested.get("right") if isinstance(nested, Mapping) else None
        if current is None and isinstance(arm, Mapping):
            if key == "before":
                before = arm.get("pose_xyz_xyzw")
            else:
                after = arm.get("pose_xyz_xyzw")
    start = _vector3(before[:3]) if isinstance(before, list) else None
    end = _vector3(after[:3]) if isinstance(after, list) else None
    target = segment.get("commanded_pose_xyz_xyzw")
    if target is None:
        arguments = record.get("arguments")
        targets = arguments.get("targets") if isinstance(arguments, Mapping) else None
        arm = targets.get("right") if isinstance(targets, Mapping) else None
        if isinstance(arm, Mapping):
            target = arm.get("position_m")
    target = _vector3(target[:3]) if isinstance(target, list) else None
    if start is not None and end is not None:
        result["realized_translation_world_m"] = [b - a for a, b in zip(start, end)]
    if start is not None and target is not None:
        result["requested_translation_world_m"] = [b - a for a, b in zip(start, target)]
    return result


def _position(side: Any, arm: str, flat: Any, *, sole: bool) -> list[float] | None:
    state = side.get(arm) if isinstance(side, Mapping) else None
    pose = state.get("pose_xyz_xyzw") if isinstance(state, Mapping) else None
    if pose is None and sole:
        pose = flat
    return _vector3(pose[:3]) if isinstance(pose, list) else None


def _target(source: Mapping[str, Any], record: Mapping[str, Any], arm: str, *,
            sole: bool) -> list[float] | None:
    target = source.get("commanded_pose_xyz_xyzw") if sole else None
    if target is None:
        arguments = record.get("arguments")
        targets = arguments.get("targets") if isinstance(arguments, Mapping) else None
        state = targets.get(arm) if isinstance(targets, Mapping) else None
        target = state.get("position_m") if isinstance(state, Mapping) else None
    return _vector3(target[:3]) if isinstance(target, list) else None


def arm_translations(record: Mapping[str, Any]) -> dict[str, dict[str, list[float]]]:
    """Realized and requested translation per targeted arm that has measurements."""
    arms = record.get("target_arms")
    if not isinstance(arms, list):
        return {}
    segment = record.get("segment", record)
    source = segment if isinstance(segment, Mapping) else record
    sole = len(arms) == 1
    result: dict[str, dict[str, list[float]]] = {}
    for arm in arms:
        if not isinstance(arm, str):
            continue
        start = _position(source.get("before"), arm, source.get("before_pose_xyz_xyzw"), sole=sole)
        if start is None:
            continue
        values: dict[str, list[float]] = {}
        end = _position(source.get("after"), arm, source.get("reached_pose_xyz_xyzw"), sole=sole)
        if end is not None:
            values["realized_translation_world_m"] = [b - a for a, b in zip(start, end)]
        target = _target(source, record, arm, sole=sole)
        if target is not None:
            values["requested_translation_world_m"] = [b - a for a, b in zip(start, target)]
        if values:
            result[arm] = values
    return result


def with_arm_translations(arms: Any, translations: Mapping[str, Mapping[str, Any]]) -> Any:
    """The public per-arm detail with each targeted arm's translation added."""
    if not isinstance(arms, Mapping):
        return arms
    return {arm: ({**state, **copy.deepcopy(dict(translations[arm]))}
                  if arm in translations and isinstance(state, Mapping) else state)
            for arm, state in arms.items()}


def cause_evidence(contact: Mapping[str, Any], record: Mapping[str, Any],
                   translations: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Force fields plus the sole targeted arm's translation, for the cause estimate."""
    evidence = dict(contact)
    arms = record.get("target_arms")
    if isinstance(arms, list) and len(arms) == 1 and arms[0] in translations:
        evidence.update(translations[arms[0]])
    return evidence


def _retract_status(effect: Any) -> Any:
    progress = effect.get("recovery_progress") if isinstance(effect, Mapping) else None
    retract = progress.get("retract") if isinstance(progress, Mapping) else None
    return retract.get("status") if isinstance(retract, Mapping) else None


def skipped_retract_arms(record: Mapping[str, Any]) -> frozenset[str]:
    """Arms whose last IK attempt belongs to a retract that was then skipped."""
    attempts, arms = record.get("ik_attempts"), record.get("arms")
    if not isinstance(attempts, list) or not isinstance(arms, Mapping):
        return frozenset()
    last = {attempt["arm"]: attempt.get("segment") for attempt in attempts
            if isinstance(attempt, Mapping) and isinstance(attempt.get("arm"), str)}
    return frozenset(arm for arm, segment in last.items()
                     if segment == RETRACT_SEGMENT and _retract_status(arms.get(arm)) == "skipped")


def without_skipped_retract_diagnostics(arms: Any, record: Mapping[str, Any]) -> Any:
    """The public per-arm detail without a skipped optional retract's IK diagnostic."""
    skipped = skipped_retract_arms(record)
    if not skipped or not isinstance(arms, Mapping):
        return arms
    return {arm: ({key: value for key, value in state.items() if key != "planning_diagnostics"}
                  if arm in skipped and isinstance(state, Mapping) else state)
            for arm, state in arms.items()}


__all__ = ["CAPABILITY", "RETRACT_SEGMENT", "TRANSLATION_KEYS", "arm_translations",
           "cause_evidence", "contact_detail", "skipped_retract_arms", "with_arm_translations",
           "without_skipped_retract_diagnostics"]
