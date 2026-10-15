"""Planning-time IK reference for a move_pose target: jaw turn and reach along the request.

Two surface capabilities read the backend's read-only reach check
(``reach_check.check_queries`` over the ``align_plan_callback`` bridge, the
route ``move_pose`` itself runs) before the action is submitted. Neither
changes what the backend executes or how it refuses; both only choose or
report from the same IK.

``move_pose_jaw_symmetry_headroom_v1`` -- a parallel jaw turned 180 degrees
about its approach axis is the same jaw, and on the ARX X5 that turn is joint6
+/- pi with joints 1-5 unchanged (``grasp_jaw_symmetry``). An empty-jaw move
(the arm's last gripper command was open) to an absolute orientation along the
free-space route is checked in both turns; the requested turn is kept unless
its joint6 margin is below ``JOINT6_MARGIN_FLOOR_RAD`` or it does not solve,
and then the reachable turn with more joint6 margin is submitted. A recorded
stack_bowls round 1: from home the requested orientation solved at joint6
-2.465 (right) and -3.007 (left), 0.68 and 0.13 rad from the +/-3.14 stop, and
the next translations at the same orientation needed joint6 +3.09 / +2.53, a
5.5 rad branch jump that was refused twice; the turned orientation sits at
joint6 about 0.68 and 0.13. A held object would turn with the jaw, so a
closed or unknown jaw is never turned.

``move_pose_reach_along_request_v1`` -- when the submitted target does not
solve, the straight interpolation from the measured pose to the target
(position lerp, orientation slerp) is sampled at fixed fractions with the same
interaction, and the result reports the farthest fraction reachable without a
gap from the start. In recorded runs a refusal said only ``no_valid_endpoint``
and "not proof that the target is unreachable"; the policy shortened steps that
the arm, at full extension, could not take at all.

Under ``reach_nearest_orientation_v1`` an unsolved target's
``reach_along_request`` also carries ``nearest_reachable_orientation``
(``reach_orientation.py``): the smallest reachable turn of the requested
orientation with the nominal jaw centre held.

Under the v2 generations (``move_pose_reach_along_request_v2``,
``pose_primitive_reach_along_request_v2``, ``reach_nearest_orientation_v2``;
``reach_reference.py``) the same check is asked the same questions, and every
annotation also names the IK that answered, whether it is ``move_pose``'s own
admission and its measured disagreement when it is not -- on RoboTwin it is a
single-seed damped-least-squares descent, not the route ``move_pose`` takes.

Any failure of the reference leaves the target as compiled and marks the
annotation unavailable: an IK reference never blocks a move.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

from ora.agent.composition.move_pose_reach_schema import (
    JAW_CAPABILITY, JAW_REASONS, LINE_FRACTIONS, REACH_CAPABILITY,
)
from ora.agent.composition.pose_primitive_reach_schema import CAPABILITY as PRIMITIVE_CAPABILITY
from ora.contracts.motion import compile_enhanced_move_pose

from . import reach_orientation, reach_reference
from .grasp_jaw_symmetry import ISOLATION_BOUNDS, measured_arm_joints
from .reach_check import _member, check_queries
from .robot import ToolError

#: A requested turn with at least this much joint6 (wrist-roll) margin is
#: kept: a quarter of the half-turn each side, enough for a later wrist
#: rotation of 45 degrees in either direction.
JOINT6_MARGIN_FLOOR_RAD = round(math.pi / 4, 6)
#: Margins this close are one IK answer, not a preference.
MARGIN_TIE_RAD = 0.001
REACH_CLAIM = (
    "Read-only IK on the route move_pose takes, from the measured joints, at sampled points of the "
    "straight interpolation to the target; reachability is not proven monotone between samples, and "
    "no path, collision, payload or arrival claim."
)
Bridge = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 5)


def _joint6_margin(row: Mapping[str, Any], measured: Sequence[float] | None, bound: float | None) -> float | None:
    """The row's minimum joint-limit margin when the measured joints prove joint6 binds."""
    margin, delta = row.get("joint_limit_margin_rad"), row.get("joint_delta_rad")
    if not row.get("reachable") or margin is None or delta is None or measured is None or bound is None:
        return None
    return float(margin) if max(abs(value) for value in measured[:5]) + float(delta) <= bound else None


def _jaw_choice(rows: Mapping[str, Mapping[str, Any]], margins: Mapping[str, float | None]) -> tuple[str, str]:
    exact, flip = rows["exact"], rows["jaw_flip"]
    if exact["reachable"]:
        if margins["exact"] is not None and margins["exact"] >= JOINT6_MARGIN_FLOOR_RAD:
            return "exact", "requested_turn_margin_sufficient"
        if (flip["reachable"] and margins["exact"] is not None and margins["jaw_flip"] is not None
                and margins["jaw_flip"] > margins["exact"] + MARGIN_TIE_RAD):
            return "jaw_flip", "larger_joint6_margin"
        return "exact", "requested_turn_kept"
    if flip["reachable"]:
        return "jaw_flip", "only_reachable_variant"
    return "exact", "no_reachable_variant"


def _slerp(start: Sequence[float], end: Sequence[float], fraction: float) -> list[float]:
    start, end = _unit(start), _unit(end)
    dot = sum(a * b for a, b in zip(start, end))
    end = [-item for item in end] if dot < 0 else list(end)
    dot = min(1.0, abs(dot))
    if dot > 1.0 - 1e-9:
        mixed = [a + fraction * (b - a) for a, b in zip(start, end)]
    else:
        angle = math.acos(dot)
        weight_start = math.sin((1.0 - fraction) * angle) / math.sin(angle)
        weight_end = math.sin(fraction * angle) / math.sin(angle)
        mixed = [weight_start * a + weight_end * b for a, b in zip(start, end)]
    norm = math.sqrt(sum(item * item for item in mixed))
    return [item / norm for item in mixed]


def _unit(quaternion: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(float(item) ** 2 for item in quaternion))
    return [float(item) / norm for item in quaternion]


def _rotation_rad(start: Sequence[float], end: Sequence[float]) -> float:
    return 2.0 * math.acos(min(1.0, abs(sum(a * b for a, b in zip(_unit(start), _unit(end))))))


def reach_along_request(
    arm: str, *, current_pose: Sequence[float], target_pose: Sequence[float], interaction: str,
    target_row: Mapping[str, Any], bridge: Bridge, workspace_bounds_m: Mapping[str, Any],
) -> dict[str, Any]:
    """Reachability along the straight interpolation to an unreachable target."""
    current, target = [float(item) for item in current_pose], [float(item) for item in target_pose]
    queries = [{"pose_xyzw": [*(current[i] + f * (target[i] - current[i]) for i in range(3)),
                              *_slerp(current[3:], target[3:], f)],
                "interaction": interaction, "from_index": None} for f in LINE_FRACTIONS]
    rows = check_queries(arm, queries, bridge=bridge, workspace_bounds_m=workspace_bounds_m)["results"]
    farthest = 0.0
    for fraction, row in zip(LINE_FRACTIONS, rows):
        if not row["reachable"]:
            break
        farthest = fraction
    translation = math.dist(current[:3], target[:3])
    best = queries[LINE_FRACTIONS.index(farthest)]["pose_xyzw"] if farthest else None
    return {
        "target_reachable": False, "reason": target_row.get("reason"),
        "requested_translation_m": _round(translation),
        "requested_rotation_rad": _round(_rotation_rad(current[3:], target[3:])),
        "farthest_reachable_fraction": farthest,
        "farthest_reachable_translation_m": _round(farthest * translation),
        "farthest_reachable_pose": None if best is None else {
            "position_m": [_round(item) for item in best[:3]],
            "orientation_xyzw": [_round(item) for item in best[3:]]},
        "samples": [{"fraction": fraction, "reachable": bool(row["reachable"]), "reason": row["reason"]}
                    for fraction, row in zip(LINE_FRACTIONS, rows)],
        "claim_boundary": REACH_CLAIM,
    }


def _unsolved_reference(
    host: Any, arm: str, *, current_pose: Sequence[float], target_pose: Sequence[float], interaction: str,
    target_row: Mapping[str, Any], bridge: Bridge, v2_answer: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """``reach_along_request`` for a target that does not solve, with the nearest orientation when declared.

    ``v2_answer`` is the reach check's answer for the target on a v2 surface,
    whose annotation names that check (``reach_reference.unsolved``).
    """
    reach = reach_along_request(arm, current_pose=current_pose, target_pose=target_pose, interaction=interaction,
                                target_row=target_row, bridge=bridge, workspace_bounds_m=host.workspace_bounds_m)
    if v2_answer is not None:
        return reach_reference.unsolved(host, arm, reach, answer=v2_answer, target_pose=target_pose,
                                        interaction=interaction)
    return {**reach, **reach_orientation.reference(host, arm, target_pose=target_pose, interaction=interaction)}


def primitive_reference(
    host: Any, arm: str, compiled: Mapping[str, Any], *, current_pose: Sequence[float], interaction: str,
) -> dict[str, Any]:
    """``reach_along_request`` for one semantic-primitive target on a declaring surface, else nothing.

    The primitive is submitted unchanged; only the read-only reference is added.
    """
    v2 = reach_reference.primitive_declared(host.surface)
    if not (v2 or host.surface.supports(PRIMITIVE_CAPABILITY)):
        return {}
    target = [float(item) for item in compiled["target_pose_xyz_xyzw"]]
    bridge = getattr(host, "align_plan_callback", None)
    try:
        answer = check_queries(arm, [{"pose_xyzw": target, "interaction": interaction, "from_index": None}],
                               bridge=bridge, workspace_bounds_m=host.workspace_bounds_m)
        row = answer["results"][0]
        if row["reachable"]:
            return {"reach_along_request": reach_reference.solved(answer) if v2 else {"target_reachable": True}}
        return {"reach_along_request": _unsolved_reference(
            host, arm, current_pose=current_pose, target_pose=target, interaction=interaction, target_row=row,
            bridge=bridge, v2_answer=answer if v2 else None)}
    except Exception as error:  # an IK reference never blocks a move
        failure = error.code if isinstance(error, ToolError) else type(error).__name__
        return {"reach_along_request": {"status": "unavailable", "failure": failure}}


def declared(surface: Any) -> bool:
    return (surface.supports(JAW_CAPABILITY) or surface.supports(REACH_CAPABILITY)
            or reach_reference.move_pose_declared(surface))


def plan_target(
    host: Any, arm: str, *, current_pose: Sequence[float], target_pose: Sequence[float], interaction: str,
    orientation_mode: str, observation: Mapping[str, Any],
) -> tuple[list[float], dict[str, Any]]:
    """The pose to submit for one arm and the annotations its compiled target carries."""
    from .alignment import frame_local_axes

    surface = host.surface
    jaw_declared, reach_declared = surface.supports(JAW_CAPABILITY), surface.supports(REACH_CAPABILITY)
    reach_v2 = reach_reference.move_pose_declared(surface)
    reach_declared = reach_declared or reach_v2
    target = [float(item) for item in target_pose]
    if not (jaw_declared or reach_declared):
        return target, {}
    convention = str((surface.view_config or {}).get("controlled_frame_convention"))
    effector = ((observation.get("proprioception") or {}).get("end_effectors") or {}).get(arm) or {}
    jaw_eligible = (jaw_declared and interaction == "free_space" and orientation_mode == "absolute_quaternion"
                    and effector.get("gripper_command_target") == "open" and convention in ISOLATION_BOUNDS)
    annotations: dict[str, Any] = {}
    flip = ([*target[:3], *_member(target[3:], "jaw_flip", frame_local_axes(convention)["approach"])]
            if jaw_eligible else None)
    # Asking for the half-turn of the current pose is a request to turn the
    # wrist; its turned member would be no move at all.
    if flip is not None and (math.dist(flip[:3], current_pose[:3]) < 1e-9
                             and _rotation_rad(flip[3:], current_pose[3:]) < 1e-9):
        jaw_eligible, flip = False, None
    members = {"exact": target, **({"jaw_flip": flip} if jaw_eligible else {})}
    bridge = getattr(host, "align_plan_callback", None)
    try:
        answer = check_queries(arm, [{"pose_xyzw": pose, "interaction": interaction, "from_index": None}
                                     for pose in members.values()],
                               bridge=bridge, workspace_bounds_m=host.workspace_bounds_m)
        rows = dict(zip(members, answer["results"]))
    except Exception as error:  # an IK reference never blocks a move
        failure = error.code if isinstance(error, ToolError) else type(error).__name__
        if jaw_eligible:
            annotations["jaw_symmetry"] = {"variant": "exact", "reason": "ik_reference_unavailable",
                                           "joint6_margin_rad": {"exact": None, "jaw_flip": None}}
        if reach_declared:
            annotations["reach_along_request"] = {"status": "unavailable", "failure": failure}
        return target, annotations
    chosen = "exact"
    if jaw_eligible:
        measured, bound = measured_arm_joints(observation, arm), ISOLATION_BOUNDS[convention]
        margins = {name: _joint6_margin(row, measured, bound) for name, row in rows.items()}
        chosen, reason = _jaw_choice(rows, margins)
        annotations["jaw_symmetry"] = {"variant": chosen, "reason": reason,
                                       "joint6_margin_rad": {name: _round(value) for name, value in margins.items()}}
    if reach_declared:
        row = rows[chosen]
        if row["reachable"] or bridge is None:
            annotations["reach_along_request"] = (
                reach_reference.solved(answer) if reach_v2 and row["reachable"]
                else {"target_reachable": bool(row["reachable"])})
        else:
            try:
                annotations["reach_along_request"] = _unsolved_reference(
                    host, arm, current_pose=current_pose, target_pose=members[chosen], interaction=interaction,
                    target_row=row, bridge=bridge, v2_answer=answer if reach_v2 else None)
            except Exception as error:  # an IK reference never blocks a move
                failure = error.code if isinstance(error, ToolError) else type(error).__name__
                annotations["reach_along_request"] = {"status": "unavailable", "failure": failure}
    return members[chosen], annotations


def apply_reference(
    host: Any, arm: str, compiled: Mapping[str, Any], *, current_pose: Sequence[float], interaction: str,
    observation: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """The compiled target to submit, its action pose and its public annotations.

    A jaw turn keeps the position and replaces the absolute orientation; the
    compilation is redone so every published number describes the move sent.
    """
    submitted, reference = plan_target(
        host, arm, current_pose=current_pose, target_pose=compiled["target_pose_xyz_xyzw"],
        interaction=interaction, orientation_mode=str(compiled["orientation_mode"]), observation=observation)
    if submitted != [float(item) for item in compiled["target_pose_xyz_xyzw"]]:
        compiled = {**compile_enhanced_move_pose({"position_m": submitted[:3], "orientation_xyzw": submitted[3:]},
                                                 current_pose_xyz_xyzw=current_pose, controlled_point="end_effector"),
                    "position_mode": compiled["position_mode"],
                    "position_delta_frame": compiled["position_delta_frame"]}
    pose = [float(item) for item in compiled["target_pose_xyz_xyzw"]]
    return dict(compiled), {"position_m": pose[:3], "orientation_xyzw": pose[3:]}, reference


__all__ = [
    "JAW_CAPABILITY", "JAW_REASONS", "JOINT6_MARGIN_FLOOR_RAD", "LINE_FRACTIONS", "REACH_CAPABILITY",
    "REACH_CLAIM", "apply_reference", "declared", "plan_target", "primitive_reference", "reach_along_request",
]
