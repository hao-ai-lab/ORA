"""Publish the parallel-jaw member of each X5 grasp candidate with the most wrist-roll headroom.

A parallel jaw closes on the same two faces whichever way round it is rolled
about its approach axis, so a compiled candidate is one member of a two-member
class: its orientation, and that orientation turned pi about link6 +x. On the
ARX X5 that axis is joint6 (``X5A.urdf``, hashed as ``urdf_sha256`` in
``arx_x5_gripper.ARX_X5_SOURCE_BINDING``: joint6 axis ``1 0 0``, limit
+/-3.14 rad; the finger joints slide along +/-y), so the two members share
joints 1-5 and their joint6 values differ by pi. The pooled IK selection
(``reach_check``) tried the flip only when the compiled orientation did not
solve, and the precommit keeps the compiled orientation whenever it is
admitted. EXP11A g007 stack_bowls (+38.8 min): the right-arm 9 mm wall pinch
solved at joint6 2.977 rad (pregrasp) and 3.004 rad (contact), 0.14 rad from
the stop, while its flip solved at -0.164 and -0.137 rad with joints 1-5 within
3e-6 rad; every later carry move and wrist rotation was refused at the joint6
limit. Across that run's eleven precommit probes both members shared joints
1-5 whenever both solved, and the compiled one had the smaller joint6 margin
in four of those nine pairs (and in a tenth probe the compiled pregrasp did not
solve at all).

Under ``grasp_plan_jaw_symmetry_headroom_v1`` (schema in
``composition/grasp_jaw_symmetry_schema.py``) ``reach_check`` checks both
members of every pool candidate and this module picks the one to publish:
reachable first, then the larger joint6 margin over the pregrasp and contact
solutions, then the smaller joint move from the measured joints, then the
compiled orientation. The published ``pregrasp_pose`` and ``contact_pose``
carry the chosen orientation, so ``grasp_execute`` and a copied ``move_pose``
target execute what was chosen; candidate count, ranking and order are
untouched (a roll about the approach axis changes neither reachability nor the
approach tilt).

Why the backend's minimum margin is the joint6 margin. The reach-check row
reports ``joint_limit_margin_rad``, the smallest margin of any arm joint, and
``joint_delta_rad``, the largest single-joint move from its start. The X5 table
gives joints 1-5 +/-10 rad (the live articulation reads 9.999999 in float32)
and joint6 +/-3.14 rad, so joint6's margin never exceeds 3.14 rad, and a joint
of 1-5 can only undercut it beyond |q| = 9.999999 - 3.14. Its value at a
solution is at most its measured value plus the moves along the chain
(pregrasp from the measured joints, contact from the pregrasp solution). When
that bound stays inside ``JOINT6_ISOLATION_BOUND_RAD`` the minimum margin is the
joint6 margin; otherwise ``joint6_margin_rad`` is null and the choice compares
the minimum margins instead, under its own reason. No kernel file changes: the
row contract and the worker are untouched.

An elbow-extension margin is not published. The shoulder-to-wrist distance
(joint2 axis to joint4 axis, 0.264 + 0.2513 = 0.5153 m at full extension)
needs the joint4 axis position, 0.06775 m along link4 x and 0.02895 m along the
approach behind the link6 origin; link4's x direction depends on the joint5
branch and on joint1, which only the backend's IK solution fixes, and the row
carries no joint3.
Deriving it here would be a second, branch-guessing IK; the honest source is a
reach-check row field, which is a worker-kernel change.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from ora.agent.composition.grasp_jaw_symmetry_schema import (
    CAPABILITY, CLAIM_BOUNDARY, DELTA_TIE_RAD, MARGIN_TIE_RAD, RULE, STAGES, VARIANTS,
)

#: Joints 1-5 of the X5: the smaller of the URDF's 10 rad and the float32
#: articulation readback; joint6, the wrist roll about the approach axis.
ARX_X5_ARM_JOINT_LIMIT_RAD = 9.999999
ARX_X5_WRIST_ROLL_LIMIT_RAD = 3.14
#: A bound on |q| of joints 1-5 under which joint6 is the binding joint; the
#: 1 mrad covers the 0.01 mrad rounding of the published row numbers.
JOINT6_ISOLATION_BOUND_RAD = round(ARX_X5_ARM_JOINT_LIMIT_RAD - ARX_X5_WRIST_ROLL_LIMIT_RAD - 1e-3, 6)
#: Controlled frames whose wrist-roll table this module knows.
ISOLATION_BOUNDS = {"arx_x5_link6": JOINT6_ISOLATION_BOUND_RAD}
_EMPTY_HEADROOM = {stage: None for stage in STAGES}


def measured_arm_joints(observation: Any, arm: str) -> list[float] | None:
    """The six measured arm joints the observation publishes for ``arm``, or None."""
    proprioception = observation.get("proprioception") if isinstance(observation, Mapping) else None
    effectors = proprioception.get("end_effectors") if isinstance(proprioception, Mapping) else None
    record = effectors.get(arm) if isinstance(effectors, Mapping) else None
    joints = record.get("arm_joint_qpos") if isinstance(record, Mapping) else None
    if (not isinstance(joints, (list, tuple)) or len(joints) != 6
            or any(isinstance(item, bool) or not isinstance(item, (int, float))
                   or not math.isfinite(item) for item in joints)):
        return None
    return [float(item) for item in joints]


def rule_for(surface: Any, arm: str, observation: Any) -> dict[str, Any] | None:
    """The member-choice context a declaring surface passes to the pool selection, else None."""
    if not surface.supports(CAPABILITY):
        return None
    convention = str((surface.view_config or {}).get("controlled_frame_convention"))
    bound = ISOLATION_BOUNDS.get(convention)
    return {
        "identity": {"capability": CAPABILITY, "rule": RULE, "margin_tie_rad": MARGIN_TIE_RAD,
                     "joint_delta_tie_rad": DELTA_TIE_RAD, "joint6_isolation_bound_rad": bound},
        "isolation_bound_rad": bound,
        "measured_joints": measured_arm_joints(observation, arm) if bound is not None else None,
    }


def _stage_headroom(row: Mapping[str, Any], reach: float | None, bound: float | None) -> dict[str, Any]:
    margin = row.get("joint_limit_margin_rad")
    isolated = margin is not None and reach is not None and bound is not None and reach <= bound
    return {"min_joint_limit_margin_rad": margin, "joint6_margin_rad": margin if isolated else None}


def member_headroom(pregrasp: Mapping[str, Any], approach: Mapping[str, Any],
                    rule: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Both stages' margins for one member whose pregrasp and approach both solved.

    ``reach`` bounds |q| of joints 1-5 at each solution: the largest measured
    one plus every move along the chain that led there.
    """
    measured, bound = rule.get("measured_joints"), rule.get("isolation_bound_rad")
    reach = None if measured is None else max(abs(value) for value in measured[:5])
    chained = []
    for row in (pregrasp, approach):
        delta = row.get("joint_delta_rad")
        reach = None if reach is None or delta is None else reach + float(delta)
        chained.append(reach)
    return {stage: _stage_headroom(row, value, bound)
            for stage, row, value in zip(STAGES, (pregrasp, approach), chained)}


def _smallest(values: Sequence[Any]) -> float | None:
    return None if any(value is None for value in values) else min(float(value) for value in values)


def _compare(rows: Mapping[str, Sequence[Mapping[str, Any]]], joint6: Mapping[str, Any],
             headroom: Mapping[str, Mapping[str, Any]]) -> tuple[str, str]:
    """Both members solved: the larger margin, then the smaller move, then the compiled one."""
    isolated = all(joint6[name] is not None for name in VARIANTS)
    margin = joint6 if isolated else {
        name: _smallest([headroom[name][stage]["min_joint_limit_margin_rad"] for stage in STAGES])
        for name in VARIANTS}
    exact_margin, flip_margin = margin["exact"], margin["jaw_flip"]
    if (exact_margin is not None and flip_margin is not None
            and abs(exact_margin - flip_margin) > MARGIN_TIE_RAD):
        return ("exact" if exact_margin > flip_margin else "jaw_flip"), (
            "larger_joint6_margin" if isolated else "larger_joint_limit_margin")
    exact_move, flip_move = (rows[name][0].get("joint_delta_rad") for name in VARIANTS)
    if exact_move is not None and flip_move is not None and abs(exact_move - flip_move) > DELTA_TIE_RAD:
        return ("exact" if exact_move < flip_move else "jaw_flip"), "margin_tie_smaller_joint_delta"
    return "exact", "margin_and_joint_delta_tie"


def choose(rows: Mapping[str, Sequence[Mapping[str, Any]]], rule: Mapping[str, Any]) -> dict[str, Any]:
    """The member to publish from each member's (pregrasp, approach) reach-check rows."""
    solved = [name for name in VARIANTS if rows[name][0]["reachable"] and rows[name][1]["reachable"]]
    headroom = {name: member_headroom(*rows[name], rule) for name in solved}
    joint6 = {name: (_smallest([headroom[name][stage]["joint6_margin_rad"] for stage in STAGES])
                     if name in headroom else None) for name in VARIANTS}
    if not solved:
        variant, reason = "exact", "no_reachable_variant"
    elif len(solved) == 1:
        variant, reason = solved[0], "only_reachable_variant"
    else:
        variant, reason = _compare(rows, joint6, headroom)
    return {
        "member": variant if solved else None,
        "jaw_symmetry": {"variant": variant, "reason": reason, "joint6_margin_rad": joint6},
        "joint_headroom": dict(headroom.get(variant, _EMPTY_HEADROOM)),
    }


def unavailable() -> dict[str, Any]:
    """The fields a candidate carries when the IK reference could not be taken."""
    return {
        "jaw_symmetry": {"variant": "exact", "reason": "ik_reference_unavailable",
                         "joint6_margin_rad": {name: None for name in VARIANTS}},
        "joint_headroom": dict(_EMPTY_HEADROOM),
    }


def summary() -> dict[str, str]:
    return {"rule": RULE, "claim_boundary": CLAIM_BOUNDARY}


__all__ = [
    "ARX_X5_ARM_JOINT_LIMIT_RAD", "ARX_X5_WRIST_ROLL_LIMIT_RAD", "CAPABILITY", "ISOLATION_BOUNDS",
    "JOINT6_ISOLATION_BOUND_RAD", "choose", "measured_arm_joints", "member_headroom", "rule_for",
    "summary", "unavailable",
]
