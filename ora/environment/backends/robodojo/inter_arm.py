"""Inter-arm admission of RoboDojo motions that drive both arms at once.

EXP11A g012 (E11G12-004a) recorded 15 of 80 ``controller_stalled`` moves as one
ARX arm driving into the other; admission had screened each arm only against
itself. ``ik.solve_ik`` now screens a single moving arm against the other arm
held at its measured joints while it ranks IK candidates. When one command
drives both arms -- a dual-arm ``move_pose`` segment or a two-arm recovery
step -- each arm's IK is screened only against itself and the selected pair is
screened here, in lockstep, before anything is written. A refusal names the
limiting links and the sphere clearance found and required.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from ora.contracts.planning_diagnostics import SCHEMA, validate_planning_diagnostics
from ora.environment.controllers.joint_path_clearance import INTER_ARM_MINIMUM_CLEARANCE_M

from . import ik

#: ``inter_arm_collision:<segment>``: the segment's joint path meets the other arm.
SEGMENT_CODE = "inter_arm_collision"
LOCKSTEP_EVIDENCE_SCOPE = (
    "Both arms' SDK collision spheres sampled along their straight joint interpolations in lockstep "
    "(finger links are not in the sphere model); no scene, table, payload or task-success proof."
)
LOCKSTEP_RECOVERY = (
    "Driven together, the two arms' straight joint paths bring their SDK collision spheres within "
    f"{INTER_ARM_MINIMUM_CLEARANCE_M:g} m of each other, so nothing moved. Move one arm at a time, or choose "
    "targets that keep the arms apart. This is not proof that either target is unreachable."
)
EVIDENCE_KEYS = (*ik.INTER_ARM_EVIDENCE_KEYS, "moving_arms")


def code(segment: str) -> str:
    return f"{SEGMENT_CODE}:{segment}"


def refusal_code(solved: Mapping[str, Any], segment: str) -> str:
    """The failure code of an IK refusal: the other arm's, else the solver's."""
    outcome = (solved.get("ik_selection") or {}).get("outcome")
    return code(segment) if outcome == ik.INTER_ARM_OUTCOME else f"ik_failed:{segment}"


def compact(evidence: Mapping[str, Any]) -> dict[str, Any]:
    return {key: evidence.get(key) for key in EVIDENCE_KEYS}


def diagnostic(evidence: Mapping[str, Any]) -> dict[str, Any]:
    return validate_planning_diagnostics({
        "schema": SCHEMA, "phase": "joint_space", "outcome": ik.INTER_ARM_OUTCOME, "candidate_counts": {},
        "rejection_counts": {ik.INTER_ARM_OUTCOME: 1}, "details_available": True,
        "recovery": [LOCKSTEP_RECOVERY], "evidence_scope": LOCKSTEP_EVIDENCE_SCOPE,
        "note": ik.inter_arm_note(evidence),
    })


def blocked(evidence: Mapping[str, Any] | None) -> bool:
    return evidence is not None and evidence.get("status") != "clear"


def screen_from_measured(session: Any, targets: Mapping[str, Sequence[float]]) -> dict[str, Any] | None:
    """Lockstep screen from the arms' joints measured now; an unnamed arm holds still."""
    return ik.lockstep_inter_arm(session, {arm: ik.measured_arm_joints(session, arm) for arm in targets}, targets)


def lockstep_refusal(execution: Any, segment: str, poses: Mapping[str, Sequence[float]],
                     joints: Mapping[str, Sequence[float]],
                     planned: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]] | None:
    """Screen a segment that drives both arms; when blocked, record it and return per-arm receipt extras."""
    if len(joints) < 2:
        return None
    evidence = screen_from_measured(execution.session, joints)
    if not blocked(evidence):
        return None
    for arm in joints:
        execution.rejected_ik(segment, arm, poses[arm], {"joint_target": list(joints[arm])},
                              {**planned[arm], "inter_arm_clearance": compact(evidence)})
    return {arm: {"planning_diagnostics": diagnostic(evidence)} for arm in joints}


__all__ = [
    "EVIDENCE_KEYS", "LOCKSTEP_EVIDENCE_SCOPE", "LOCKSTEP_RECOVERY", "SEGMENT_CODE", "blocked", "code",
    "compact", "diagnostic", "lockstep_refusal", "refusal_code", "screen_from_measured",
]
