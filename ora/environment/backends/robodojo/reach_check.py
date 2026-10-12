"""Read-only reach check that models the RoboDojo move_pose route, per query.

``sparse_motion._execute_pose_targets`` picks a route from the interaction and
the start pose: ``free_space`` is global (SDK IK, joint jump limit pi);
``precision`` is local (one bounded step, jump limit 1 rad) within the
50 mm / 15 degree envelope and global beyond it; ``contact_approach`` is local
within the envelope and, beyond it, staged (global to a pre-contact pose, then
the local step) when the session enables staging and refused otherwise. This
module replays that choice and the same ``ik.solve_ik`` calls without writing
the plant, from the measured joints or -- for a chained query -- from the
solution an earlier query in the batch reached, with that query's target as
the start pose. The executor starts from the *measured* pose after arrival,
which lies within the arrival tolerance of that target; the claim boundary
says so.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ora.contracts.motion import compile_servo_delta
from ora.contracts.reach_check import START_UNREACHABLE
from ora.environment.controllers.motion_transaction import _precontact_pose
from ora.environment.controllers.reach_check import joint_limit_margin_rad
from ora.environment.controllers.reachability import cartesian_pose_error_twist

from .common import pose_error
from .reachability import fk_calibrated, fk_link6

#: v2 adds the other arm, held at its measured joints, to the path screen (E11G12-004a).
IK_MODEL = "robodojo_sdk_ik_endpoint_selection_v2"
CLAIM_BOUNDARY = (
    "The SDK IK solve, joint limits, route jump bound (pi global, 1 rad local) and SDK "
    "collision spheres along the straight joint path that the modelled move_pose route runs, "
    "checked against this arm itself and against the other arm held at its measured joints, "
    "from the measured joints or, for a chained query, from the earlier query's solution at its "
    "target pose (execution starts within the arrival tolerance of it): no scene, table, "
    "finger-link or payload collision, no physics and no arrival, grasp or task-success claim."
)
_OUTCOME_REASONS = {"joint_jump": "ik_branch_jump", "self_collision_path": "self_collision_path",
                    "inter_arm_collision_path": "inter_arm_collision_path"}


def _unreached(solved: Mapping[str, Any], *, local: bool) -> str:
    outcome = solved.get("ik_selection", {}).get("outcome")
    if outcome == "joint_jump":
        return "local_ik_branch_jump" if local else "ik_branch_jump"
    if outcome in _OUTCOME_REASONS:
        return _OUTCOME_REASONS[outcome]
    rejections = {row.get("rejection") for row in solved.get("candidates", []) if isinstance(row, Mapping)}
    # SDK-feasible solutions exist but lie outside the articulation limits, or
    # land off the pose: name that rather than the generic solver failure.
    if "joint_limits" in rejections:
        return "joint_limit"
    if "fk_residual_exceeded" in rejections:
        return "fk_residual_exceeded"
    return "ik_failed"


def _solve(session: Any, arm: str, target: Sequence[float], start_joints: Sequence[float], *,
           local: bool) -> tuple[Mapping[str, Any], str | None]:
    """One executor IK call; the reason it would refuse, or ``None``."""
    from .ik import solve_ik

    limit = session.local_joint_jump_limit_rad if local else session.global_joint_jump_limit_rad
    solved = solve_ik(session, arm, target, current=start_joints,
                      **({"jump_limit_rad": limit} if local else {}))
    if solved["joint_target"] is None or solved["ik_selection"]["outcome"] != "admitted":
        return solved, _unreached(solved, local=local)
    if solved["joint_delta_rad"] > limit:
        return solved, "local_ik_branch_jump" if local else "ik_branch_jump"
    return solved, None


def _local_target(start_pose: Sequence[float], target: Sequence[float]) -> list[float]:
    """The pose ``sparse_motion._local_step`` actually commands for this step."""
    from .sparse_motion import rotation_vector_between

    compiled = compile_servo_delta({
        "translation_m": [target[index] - start_pose[index] for index in range(3)],
        "rotation_rotvec_rad": rotation_vector_between(target[3:], start_pose[3:]),
        "frame": "world",
    }, current_pose_xyz_xyzw=list(start_pose), controlled_point="link6")
    return [float(item) for item in compiled["target_pose_xyz_xyzw"]]


def _route(session: Any, interaction: str, start_pose: Sequence[float], target: Sequence[float]) -> str | None:
    from .sparse_motion import ROUTE_REFUSAL_CODE, STAGED_CONTACT_STRATEGY, within_local_envelope

    local = within_local_envelope(*pose_error(target, start_pose))
    if interaction == "free_space":
        return "global"
    if interaction == "precision":
        return "local" if local else "global"
    if local:
        return "local"
    return STAGED_CONTACT_STRATEGY if session.contact_approach_route == STAGED_CONTACT_STRATEGY else ROUTE_REFUSAL_CODE


def _row(reachable: bool, reason: str | None, route: str | None, *, error=None, joints=None,
         solved: Mapping[str, Any] | None = None, reference: Sequence[float] | None = None) -> dict[str, Any]:
    position = None if error is None else float(np.linalg.norm(error[:3]))
    rotation = None if error is None else float(np.linalg.norm(error[3:]))
    limits = None if solved is None else solved.get("joint_limits")
    delta = (None if joints is None or reference is None
             else float(np.max(np.abs(np.asarray(joints, dtype=float) - np.asarray(reference, dtype=float)))))
    return {"reachable": reachable, "reason": reason, "route": route,
            "position_error_m": position, "orientation_error_rad": rotation,
            "joint_limit_margin_rad": None if joints is None or limits is None
            else joint_limit_margin_rad(joints, limits),
            "joint_delta_rad": delta}


def _answer(session: Any, arm: str, query: Mapping[str, Any], start_joints: Sequence[float],
            start_pose: Sequence[float], *, tolerances: tuple[float, float]) -> tuple[dict[str, Any], list[float] | None]:
    """One query on its modelled route: the row and the joints it ends on."""
    from .sparse_motion import ROUTE_REFUSAL_CODE, STAGED_CONTACT_STRATEGY
    from .staged_contact import PRECONTACT_LOCAL_ROTATION_RAD, PRECONTACT_LOCAL_TRANSLATION_M

    target = [float(item) for item in query["pose_xyzw"]]
    route = _route(session, query["interaction"], start_pose, target)
    if route == ROUTE_REFUSAL_CODE:
        return _row(False, ROUTE_REFUSAL_CODE, None), None
    joints_from, pose_from = list(start_joints), list(start_pose)
    if route == STAGED_CONTACT_STRATEGY:
        precontact = _precontact_pose(pose_from, target, local_translation_m=PRECONTACT_LOCAL_TRANSLATION_M,
                                      local_rotation_rad=PRECONTACT_LOCAL_ROTATION_RAD)
        solved, reason = _solve(session, arm, precontact, joints_from, local=False)
        if reason is not None:
            return _row(False, reason, route, solved=solved), None
        joints_from, pose_from = list(solved["joint_target"]), precontact
    local = route != "global"
    commanded = _local_target(pose_from, target) if local else target
    solved, reason = _solve(session, arm, commanded, joints_from, local=local)
    joints = solved.get("joint_target")
    if reason is not None:
        # A branch jump still names the nearest valid solution; report its
        # residual and move so the refusal can be weighed.
        kept = joints if reason in {"ik_branch_jump", "local_ik_branch_jump"} else None
        error = None if kept is None else cartesian_pose_error_twist(target, fk_link6(session, arm, kept))
        return _row(False, reason, route, error=error, joints=kept, solved=solved, reference=start_joints), None
    error = cartesian_pose_error_twist(target, fk_link6(session, arm, joints))
    within = float(np.linalg.norm(error[:3])) <= tolerances[0] and float(np.linalg.norm(error[3:])) <= tolerances[1]
    return (_row(within, None if within else "fk_residual_exceeded", route, error=error, joints=joints,
                 solved=solved, reference=start_joints),
            list(joints) if within else None)


def reach_check_batch(session: Any, arm: str, queries: Sequence[Mapping[str, Any]], *,
                      position_tolerance_m: float, orientation_tolerance_rad: float) -> dict[str, Any]:
    """One row per query in order; nothing is written and no physics advances."""
    measured = session._proprioception()["end_effectors"][arm]
    clock = session._clock()
    rows: list[dict[str, Any]] = []
    ends: list[tuple[list[float], list[float]] | None] = []
    calibrated = fk_calibrated(session, arm, measured)
    for query in queries:
        start = ((list(measured["arm_joint_qpos"]), list(measured["pose_xyz_xyzw"]))
                 if query["from_index"] is None else ends[query["from_index"]])
        if not calibrated:
            # The measured link6 frame disagrees with FK of the measured joints:
            # the arm is still settling or the model is off. A typed answer the
            # policy can retry after the next observation, not an internal fault.
            rows.append(_row(False, "fk_calibration_mismatch", None))
            ends.append(None)
            continue
        if start is None:
            rows.append(_row(False, START_UNREACHABLE, None))
            ends.append(None)
            continue
        row, joints = _answer(session, arm, query, start[0], start[1],
                              tolerances=(position_tolerance_m, orientation_tolerance_rad))
        rows.append(row)
        ends.append(None if joints is None else (joints, [float(item) for item in query["pose_xyzw"]]))
    if session._clock() != clock:
        session._fault("reach_check_advanced_physics")
        raise RuntimeError("read-only reach check advanced physics")
    return {"ik_model": IK_MODEL, "matches_move_pose_admission": True,
            "claim_boundary": CLAIM_BOUNDARY, "results": rows}
