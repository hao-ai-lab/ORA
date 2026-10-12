"""Bounded, continuous selection from the pinned SDK's existing IK solve.

Only the isolated worker imports the SDK. One solve keeps its configured seed
count, exact target, collision checks and joint limits. Selection uses actual
joint coordinates: no modulo shortcut is valid for the joint interpolator.

Keeping each revolute joint at the equivalent nearest the measured joints lets
an arm drift past +/-pi one continuous step at a time (EXP11A g005 audit D3).
A candidate that carries a pitch joint (``ARX_X5_PITCH_JOINT_NAMES``) out of
``common.CANONICAL_JOINT_INTERVAL_RAD``, or farther outside it than it already
is, is refused as ``pitch_joint_wrap``. joint1 (base yaw) may still wind, but
when admitted candidates differ in canonicity, those whose revolute joints all
lie in the interval are tried first while their jump stays inside the session's
local joint-jump bound. That ordering never admits a candidate.

Each admitted candidate's straight joint path is screened with the SDK's own
collision spheres: against this arm itself, and -- when the other arm holds
still -- against the other arm at its measured joints (EXP11A g012 E11G12-004a:
15 of 80 stalls were one arm driving into the other). An arm that moves with
the other one in the same command is left to ``lockstep_inter_arm``.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence
from collections import Counter
from pathlib import Path

import numpy as np

from ora.environment.controllers.motion_transaction import select_nearest_joint_solution
from ora.contracts.planning_diagnostics import SCHEMA, validate_planning_diagnostics
from ora.environment.controllers.joint_path_clearance import (
    INTER_ARM_MINIMUM_CLEARANCE_M, InterArmSphereScreen, JointPathClearanceError, SphereSelfCollisionModel,
)

from .bridge import RobodojoError
from .common import (
    CANONICAL_JOINT_INTERVAL_RAD, array, canonical_joints, native_pose, nearest_equivalent_joints, pitch_joint_wraps,
    pose_error,
)

# Assets a14409d7, Assets/Robots/x5/ARX.usd, SHA256
# 5723a835bf5627a935a8dabb3a44447725bd1465243ce791e221b3e990121d63
# is the composed articulation source; the six controlled arm
# joints are PhysicsRevoluteJoint. Limits are read from the actual articulation,
# never inferred here from URDF defaults. Finger joints are not periodic.
ARX_X5_REVOLUTE_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 7))
# Pitch joints may not pass +/-pi (review U1 decision). By X5A.urdf forward
# kinematics (review U1): joint2 at +/-pi holds the upper arm level and
# forward, past it below level; joint3 at +/-pi is full elbow extension
# (joint2-joint4 at its 0.512 m maximum) while its fold is at 0 / 2*pi, so for
# joint3 the rule caps hyperextension -- a policy choice, not a self-collision
# guard, and one-sided: the fold side [-pi, 0) stays allowed and the SDK
# spheres ignore link2 against link3/link4; joint4 and joint5 at +/-pi fold
# the jaws onto the forearm (2.2 cm and 1 mm), which the straight-path sphere
# screen already covers (link3-link5/link6, link4-link6). EXP11A g005 wound
# joint3 past +/-pi in 98 of 1397 receipts and joint2 in 13; joint4 and joint5
# never. joint1 (base yaw) may wind; joint6 is limited to +/-3.14 rad.
ARX_X5_PITCH_JOINT_NAMES = ("joint2", "joint3", "joint4", "joint5")

MAX_CANDIDATES = 32
EVIDENCE_SCOPE = "One bounded SDK IK solve and endpoint FK checks; no path or task-success proof."
PATH_EVIDENCE_SCOPE = (
    "One bounded SDK IK solve, endpoint FK checks and the SDK collision spheres sampled along the straight "
    "joint interpolation for self-collision; no scene, table, payload or task-success proof."
)
INTER_ARM_PATH_EVIDENCE_SCOPE = (
    "One bounded SDK IK solve, endpoint FK checks and the SDK collision spheres sampled along the straight "
    "joint interpolation for self-collision and against the other arm at its measured joints (finger links "
    "are not in the sphere model); no scene, table, payload or task-success proof."
)
INTER_ARM_OUTCOME = "inter_arm_collision_path"
INTER_ARM_PATH_RECOVERY = (
    "Every continuous endpoint in this bounded solve reaches the target only through a straight joint path "
    f"that brings this arm's SDK collision spheres within {INTER_ARM_MINIMUM_CLEARANCE_M:g} m of the other arm's, "
    "so nothing moved. Move the other arm clear first, or choose a target or an intermediate pose farther "
    "from it. This is not proof that the target is unreachable."
)
OTHER_ARM = {"left": "right", "right": "left"}
SELF_COLLISION_PATH_RECOVERY = (
    "Every continuous endpoint in this bounded solve reaches the target only through a straight joint path "
    "whose SDK collision spheres self-intersect, so nothing moved. Try an intermediate pose that changes "
    "the arm configuration gradually, or a different target. This is not proof that the target is unreachable."
)
SELECTION_RULE = "canonical_first_nearest_admissible_raw_joint_distance"
SELECTION_RECOVERY = {
    "admitted": None,
    "self_collision_path": SELF_COLLISION_PATH_RECOVERY,
    INTER_ARM_OUTCOME: INTER_ARM_PATH_RECOVERY,
    "joint_jump": "No continuous solution in this bounded solve; consider a different exact pose or "
                  "intermediate pose. A smaller step is not guaranteed to succeed.",
    "no_valid_endpoint": "No valid endpoint in this bounded solve; inspect candidate pose and try an "
                         "alternative. This is not proof that the target is unreachable.",
    "pitch_joint_wrap": "Every valid endpoint in this bounded solve needs the shoulder, elbow or wrist to bend "
                        "past its half-turn limit -- for the elbow, beyond full extension -- so nothing moved. "
                        "Choose a different approach direction or an intermediate pose. This is not proof that "
                        "the target is unreachable.",
}


def articulation_limits(session: Any, arm: str) -> list[list[float]]:
    manager, robot = session.environment.robot_manager, session.robots[arm]
    data = manager.robot_key[manager.robot_list.index(robot)].data
    for name in ("joint_pos_limits", "joint_limits"):
        if hasattr(data, name):
            limits = array(getattr(data, name))[0][list(robot.arm_joint_indices)]
            if (limits.shape != (len(robot.arm_joint_indices), 2)
                    or not np.isfinite(limits).all() or np.any(limits[:, 0] > limits[:, 1])):
                raise RobodojoError("RoboDojo articulation joint limits are invalid")
            return limits.tolist()
    raise RobodojoError("RoboDojo articulation exposes no joint limits")


def sdk_candidates(planner: Any, current: Sequence[float], pose: Sequence[float],
                   robot_base_pose: Sequence[float]) -> dict[str, Any]:
    """Use the SDK's own transform/builders, retaining every returned seed.

    ``pose`` is public world link6 xyzw. The SDK owns world/base conversion;
    result joint order is name-bound even when locked finger joints appear.
    """
    from utils.transformer import calculate_target_pose

    target = calculate_target_pose(robot_base_pose, planner.robot_origin_pose, native_pose(pose))
    position, quaternion = planner._trans_from_world_to_base(
        np.asarray(planner.robot_origin_pose, dtype=np.float32), np.asarray(target, dtype=np.float32),
    )
    position = np.asarray(position).copy() + np.asarray(planner.frame_bias)
    count = min(MAX_CANDIDATES, int(planner.ik_solver.config.num_seeds))
    if count < 1:
        raise RobodojoError("SDK IK has no configured seeds")
    result = planner.ik_solver.solve_pose(
        goal_tool_poses=planner._build_goal_pose(position, quaternion),
        current_state=planner._build_joint_state(current), return_seeds=count,
    )
    names = list(result.js_solution.joint_names)
    active = list(planner.active_joints_name)
    if len(set(names)) != len(names) or len(set(active)) != len(active) or not set(active) <= set(names):
        raise RobodojoError("SDK IK candidate joint names do not match the active articulation")
    values = array(result.js_solution.position)
    if values.shape != (1, count, len(names)):
        raise RobodojoError("SDK IK candidate dimensions drifted")
    values = values[0][:, [names.index(name) for name in active]]
    success, feasible = array(result.success), array(result.feasible)
    if success.shape != (1, count) or feasible.shape != (1, count):
        raise RobodojoError("SDK IK candidate feasibility dimensions drifted")
    return {"joint_names": active, "solutions": values, "success": success[0].astype(bool),
            "feasible": feasible[0].astype(bool),
            "revolute": [name in ARX_X5_REVOLUTE_JOINT_NAMES for name in active],
            "pitch": [name in ARX_X5_PITCH_JOINT_NAMES for name in active],
            "position_tolerance_m": float(planner.ik_solver.config.position_tolerance),
            "orientation_tolerance_rad": float(planner.ik_solver.config.orientation_tolerance)}


def _joint_mask(candidates: Mapping[str, Any], key: str, default: Sequence[bool]) -> list[bool]:
    mask = list(candidates.get(key, default))
    if len(mask) != len(default) or any(type(x) is not bool for x in mask):
        raise RobodojoError(f"IK joint {key} mask is invalid")
    return mask


def _validated_inputs(candidates: Mapping[str, Any], current: Sequence[float],
                      limits: Sequence[Sequence[float]], jump_limit_rad: float) -> dict[str, Any]:
    """The candidate arrays, joint masks and SDK tolerances, or a fail-closed error."""
    values = np.asarray(candidates["solutions"], dtype=float)
    reference, bounds = np.asarray(current, dtype=float), np.asarray(limits, dtype=float)
    names = list(candidates["joint_names"])
    successes, feasible = np.asarray(candidates["success"], dtype=bool), np.asarray(candidates["feasible"], dtype=bool)
    if (values.ndim != 2 or not 1 <= len(values) <= MAX_CANDIDATES
            or reference.shape != (len(names),) or values.shape[1:] != reference.shape
            or len(set(names)) != len(names) or bounds.shape != (len(names), 2)
            or successes.shape != (len(values),) or feasible.shape != successes.shape
            or not np.isfinite(reference).all() or not np.isfinite(bounds).all()
            or np.any(bounds[:, 0] > bounds[:, 1]) or not np.isfinite(jump_limit_rad) or jump_limit_rad <= 0):
        raise RobodojoError("IK candidate set, limits or current state is invalid")
    revolute = _joint_mask(candidates, "revolute", [False] * len(names))
    # Without an explicit mask a pitch joint is a source-bound revolute joint
    # by name; a coordinate that cannot wrap is never one.
    pitch = _joint_mask(candidates, "pitch", [periodic and name in ARX_X5_PITCH_JOINT_NAMES
                                             for name, periodic in zip(names, revolute)])
    tolerances = (float(candidates["position_tolerance_m"]), float(candidates["orientation_tolerance_rad"]))
    if not all(np.isfinite(x) and x > 0 for x in tolerances):
        raise RobodojoError("SDK IK tolerances are invalid")
    return {"values": values, "reference": reference, "bounds": bounds, "names": names, "success": successes,
            "feasible": feasible, "revolute": revolute, "pitch": pitch, "tolerances": tolerances}


def _evaluate_candidate(index: int, inputs: Mapping[str, Any], *, target: Sequence[float], fk: Any,
                        jump_limit_rad: float) -> tuple[dict[str, Any], np.ndarray]:
    """One candidate's evidence row and its command coordinates.

    ``rejection`` stays ``None`` for an admitted candidate and is
    ``joint_jump`` for a valid but discontinuous one.
    """
    reference, bounds = inputs["reference"], inputs["bounds"]
    q = inputs["values"][index].copy()
    finite = bool(np.isfinite(q).all())
    row = {"candidate_index": index, "sdk_success": bool(inputs["success"][index]),
           "sdk_feasible": bool(inputs["feasible"][index]), "joint_target": q.tolist() if finite else None,
           "solver_joint_target": q.tolist() if finite else None, "equivalent_turns": [0] * len(q),
           "joint_delta_rad": float(np.max(np.abs(q - reference))) if finite else None, "canonical": None,
           "position_error_m": None, "orientation_error_rad": None, "rejection": None}
    if not finite:
        row["rejection"] = "nonfinite_joint_target"
    elif not row["sdk_success"] or not row["sdk_feasible"]:
        row["rejection"] = "sdk_not_feasible_or_converged"
    elif np.any(q < bounds[:, 0]) or np.any(q > bounds[:, 1]):
        row["rejection"] = "joint_limits"
    else:
        # Actual command coordinates, not a wrapped distance; each integer-turn
        # equivalent stays inside the real finite joint limits, so bounded
        # joint6 cannot cross its stop by taking a shortcut.
        command, turns = nearest_equivalent_joints(q, reference, bounds, inputs["revolute"])
        q = np.asarray(command, dtype=float)
        position, rotation = pose_error(target, fk(q.tolist()))
        wrapped = pitch_joint_wraps(q, reference, inputs["pitch"])
        row.update(joint_target=q.tolist(), equivalent_turns=turns,
                   joint_delta_rad=float(np.max(np.abs(q - reference))),
                   canonical=canonical_joints(q, inputs["revolute"]),
                   position_error_m=position, orientation_error_rad=rotation)
        position_tolerance, rotation_tolerance = inputs["tolerances"]
        if not np.isfinite([position, rotation]).all() or position > position_tolerance or rotation > rotation_tolerance:
            row["rejection"] = "fk_residual_exceeded"
        elif wrapped:
            row["rejection"] = "pitch_joint_wrap"
            row["pitch_joint_wrap"] = [inputs["names"][joint] for joint in wrapped]
        elif row["joint_delta_rad"] > jump_limit_rad:
            row["rejection"] = "joint_jump"
    return row, q


def _admission_sets(rows: Sequence[Mapping[str, Any]], canonical_limit: float) -> dict[str, list[int]]:
    """Valid, admitted, canonical-preferred and pitch-wrapped candidate indices.

    The canonical preference applies only when admitted candidates differ in
    canonicity; among candidates of one kind the nearest keeps its place.
    """
    valid = [row["candidate_index"] for row in rows if row["rejection"] in (None, "joint_jump")]
    admitted = [index for index in valid if rows[index]["rejection"] is None]
    canonical = [index for index in admitted if rows[index]["canonical"]
                 and rows[index]["joint_delta_rad"] <= canonical_limit]
    kinds = {bool(rows[index]["canonical"]) for index in admitted}
    mixed = kinds == {True, False}
    return {"valid": valid, "admitted": admitted, "canonical": canonical, "preferred": canonical if mixed else [],
            "wrapped": [row["candidate_index"] for row in rows if row["rejection"] == "pitch_joint_wrap"]}


def _screen(rows: list[dict[str, Any]], commands: np.ndarray, order: Sequence[int],
            path_clearance: Callable[[Sequence[float]], Mapping[str, Any]] | None) -> tuple[int | None, list[int]]:
    """The first candidate in ``order`` whose straight joint path is clear, and every blocked one."""
    blocked: list[int] = []
    for candidate in order:
        if path_clearance is None:
            return candidate, blocked
        evidence = dict(path_clearance(commands[candidate].tolist()))
        rows[candidate]["path_clearance"] = evidence
        if evidence.get("status") == "clear":
            return candidate, blocked
        if evidence.get("status") == "inter_arm_collision":
            rows[candidate]["rejection"] = "inter_arm_collision_path"
        else:
            rows[candidate]["rejection"] = "self_collision_path"
        blocked.append(candidate)
    return None, blocked


def _pitch_wrap_evidence(rows: Sequence[Mapping[str, Any]], commands: np.ndarray, reference: np.ndarray,
                         names: Sequence[str], wrapped: Sequence[int]) -> dict[str, Any] | None:
    """The nearest pitch-wrapped candidate's first wrapped joint, as the policy needs it."""
    if not wrapped:
        return None
    nearest = min(wrapped, key=lambda index: float(np.linalg.norm(commands[index] - reference)))
    joint = list(names).index(rows[nearest]["pitch_joint_wrap"][0])
    return {"joint": names[joint], "current_rad": float(reference[joint]),
            "target_rad": float(commands[nearest][joint]),
            "canonical_interval_rad": list(CANONICAL_JOINT_INTERVAL_RAD)}


def _outcome(selected: int | None, sets: Mapping[str, list[int]], blocked: Sequence[int],
             rows: Sequence[Mapping[str, Any]]) -> str:
    if selected is not None and sets["admitted"]:
        return "admitted"
    if blocked:
        # The first screened candidate is the one the executor would have run.
        return str(rows[blocked[0]]["rejection"])
    if sets["valid"]:
        return "joint_jump"
    return "pitch_joint_wrap" if sets["wrapped"] else "no_valid_endpoint"


INTER_ARM_EVIDENCE_KEYS = ("min_clearance_m", "required_clearance_m", "limiting_links", "limiting_fraction",
                           "start_clearance_m")


def _inter_arm_clearance(rows: Sequence[Mapping[str, Any]], blocked: Sequence[int]) -> dict[str, Any] | None:
    """The nearest path the other arm blocked: its links and sphere clearance."""
    for index in blocked:
        if rows[index]["rejection"] == INTER_ARM_OUTCOME:
            evidence = rows[index].get("path_clearance", {}).get("inter_arm") or {}
            return {key: evidence.get(key) for key in INTER_ARM_EVIDENCE_KEYS}
    return None


def _selection_summary(rows: Sequence[Mapping[str, Any]], commands: np.ndarray, inputs: Mapping[str, Any], *,
                       selected: int | None, outcome: str, sets: Mapping[str, list[int]], blocked: Sequence[int],
                       jump_limit_rad: float, canonical_limit: float, reordered: bool) -> dict[str, Any]:
    reference, names = inputs["reference"], inputs["names"]
    delta = None if selected is None else commands[selected] - reference
    limiting = None if delta is None else int(np.argmax(np.abs(delta)))
    return {"selection": SELECTION_RULE, "candidate_count": len(rows),
            "valid_endpoint_count": len(sets["valid"]), "within_jump_limit_count": len(sets["admitted"]),
            "selected_candidate_index": selected, "joint_jump_limit_rad": float(jump_limit_rad),
            "equivalent_turns": None if selected is None else rows[selected]["equivalent_turns"],
            "canonical_interval_rad": list(CANONICAL_JOINT_INTERVAL_RAD),
            "canonical_jump_limit_rad": canonical_limit, "canonical_admitted_count": len(sets["canonical"]),
            "canonical_preference_reordered": reordered,
            "selected_canonical": None if selected is None else bool(rows[selected]["canonical"]),
            "pitch_joints": [name for name, bending in zip(names, inputs["pitch"]) if bending],
            "pitch_joint_wrap_count": len(sets["wrapped"]),
            "pitch_joint_wrap": _pitch_wrap_evidence(rows, commands, reference, names, sets["wrapped"]),
            "limiting_joint": None if limiting is None else names[limiting],
            "limiting_joint_delta_rad": None if limiting is None else float(delta[limiting]),
            "self_collision_path_count": sum(rows[i]["rejection"] == "self_collision_path" for i in blocked),
            "inter_arm_collision_path_count": sum(rows[i]["rejection"] == INTER_ARM_OUTCOME for i in blocked),
            "inter_arm_clearance": _inter_arm_clearance(rows, blocked),
            "outcome": outcome, "recovery": SELECTION_RECOVERY[outcome]}


def inter_arm_note(evidence: Mapping[str, Any]) -> str:
    """One public sentence naming the limiting links and the sphere clearance."""
    links = evidence.get("limiting_links") or {}
    pair = " to ".join(f"{arm} {link}" for arm, link in sorted(links.items())) or "the two arms"
    required, fraction = evidence.get("required_clearance_m"), evidence.get("limiting_fraction")
    return (f"Nearest blocked path: {pair} sphere clearance {float(evidence['min_clearance_m']):.4f} m"
            + (f" (required {float(required):.4f} m)" if required is not None else "")
            + (f" at {float(fraction):.0%} of the joint path." if fraction is not None else "."))


def _diagnostic(rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any], *, scope: str) -> dict[str, Any]:
    clearance = summary.get("inter_arm_clearance")
    note = ({"note": inter_arm_note(clearance)}
            if summary["outcome"] == INTER_ARM_OUTCOME and clearance and clearance.get("min_clearance_m") is not None
            else {})
    return validate_planning_diagnostics({
        "schema": SCHEMA, "phase": "ik_candidate_validation", "outcome": summary["outcome"],
        "candidate_counts": {"returned": len(rows), "valid_endpoint": summary["valid_endpoint_count"],
                             "continuous": summary["within_jump_limit_count"]},
        "rejection_counts": dict(Counter(row["rejection"] for row in rows if row["rejection"])),
        "details_available": True,
        "recovery": [] if summary["outcome"] == "admitted" else [summary["recovery"]],
        "evidence_scope": scope, **note,
    })


def _canonical_limit(jump_limit_rad: float, canonical_jump_limit_rad: float | None) -> float:
    limit = float(jump_limit_rad if canonical_jump_limit_rad is None else canonical_jump_limit_rad)
    if not np.isfinite(limit) or limit <= 0:
        raise RobodojoError("IK canonical preference bound is invalid")
    return min(limit, float(jump_limit_rad))


def select_candidate_set(*, candidates: Mapping[str, Any], current: Sequence[float],
                         limits: Sequence[Sequence[float]], target: Sequence[float],
                         fk: Any, jump_limit_rad: float,
                         path_clearance: Callable[[Sequence[float]], Mapping[str, Any]] | None = None,
                         canonical_jump_limit_rad: float | None = None,
                         screens_other_arm: bool = False,
                         ) -> dict[str, Any]:
    """Validate endpoints, then rank admissible candidates by raw joint distance.

    A candidate that carries a pitch joint out of the canonical interval (or
    farther outside it) is refused. When admitted candidates differ in
    canonicity, the canonical ones whose jump is at most
    ``canonical_jump_limit_rad`` (default: the jump bound) are tried first;
    otherwise the nearest comes first. The nearest valid but discontinuous
    candidate is retained when every valid candidate exceeds the jump bound,
    so the caller can preserve its non-mutating branch refusal and disclose the
    limiting joint. ``path_clearance`` screens admitted candidates in that
    order; when every one is blocked there is no joint target at all.
    """
    inputs = _validated_inputs(candidates, current, limits, jump_limit_rad)
    canonical_limit = _canonical_limit(jump_limit_rad, canonical_jump_limit_rad)
    reference, names = inputs["reference"], inputs["names"]
    evaluated = [_evaluate_candidate(index, inputs, target=target, fk=fk, jump_limit_rad=jump_limit_rad)
                 for index in range(len(inputs["values"]))]
    rows = [row for row, _ in evaluated]
    commands = np.asarray([q for _, q in evaluated], dtype=float)
    sets = _admission_sets(rows, canonical_limit)

    def distance(index: int) -> float:
        return float(np.linalg.norm(commands[index] - reference))

    nearest_first = sorted(sets["admitted"], key=distance)
    order = sorted(sets["preferred"], key=distance) + [i for i in nearest_first if i not in sets["preferred"]]
    selected, blocked = _screen(rows, commands, order, path_clearance)
    if selected is None and not sets["admitted"] and sets["valid"]:
        _, index, _ = select_nearest_joint_solution(commands[sets["valid"]], reference,
                                                    continuous=[False] * len(names))
        selected = sets["valid"][index]
    summary = _selection_summary(rows, commands, inputs, selected=selected,
                                 outcome=_outcome(selected, sets, blocked, rows),
                                 sets=sets, blocked=blocked, jump_limit_rad=jump_limit_rad,
                                 canonical_limit=canonical_limit, reordered=order != nearest_first)
    q = None if selected is None else commands[selected]
    delta = None if q is None else q - reference
    return {"ik_status": "Success" if q is not None else "Fail", "joint_target": None if q is None else q.tolist(),
            "joint_delta_rad": None if delta is None else float(np.max(np.abs(delta))),
            "ik_selection": summary, "joint_names": names, "current_joints": reference.tolist(),
            "joint_limits": inputs["bounds"].tolist(), "candidates": rows,
            "planning_diagnostics": _diagnostic(rows, summary, scope=(
                EVIDENCE_SCOPE if path_clearance is None
                else INTER_ARM_PATH_EVIDENCE_SCOPE if screens_other_arm else PATH_EVIDENCE_SCOPE))}


def solve_ik(session: Any, arm: str, pose: Sequence[float], *,
             jump_limit_rad: float | None = None,
             current: Sequence[float] | None = None,
             static_other_arm: bool = True) -> dict[str, Any]:
    """One bounded SDK IK selection from ``current`` (default: the measured joints).

    A read-only reach check passes an earlier solution as ``current`` to answer
    a chained pose the way the executor would reach it; every write path keeps
    the measured default. The canonical-interval preference is bounded by the
    session's local joint-jump bound on every consumer, global moves included.
    ``static_other_arm`` screens each path against the other arm held at its
    measured joints; a caller moving both arms passes ``False`` and screens
    the pair with ``lockstep_inter_arm``.
    """
    from .reachability import fk_link6

    manager, robot = session.environment.robot_manager, session.robots[arm]
    current = (array(manager.get_joint(robot, env_idx_list=[0])[0]) if current is None
               else np.asarray(current, dtype=float))
    planner = manager.ik_solver[robot.robot_name]
    candidates = sdk_candidates(planner, current, pose, robot.entity_origin_pose)
    if candidates["joint_names"] != list(robot.arm_joints_name):
        raise RobodojoError("SDK IK active order differs from the controlled articulation")
    screen, other = path_screen(session, arm, static_other_arm=static_other_arm)
    return select_candidate_set(
        candidates=candidates, current=current, limits=articulation_limits(session, arm), target=pose,
        fk=lambda q: fk_link6(session, arm, q),
        jump_limit_rad=session.global_joint_jump_limit_rad if jump_limit_rad is None else jump_limit_rad,
        path_clearance=lambda q: screen(current, q),
        canonical_jump_limit_rad=getattr(session, "local_joint_jump_limit_rad", None),
        screens_other_arm=other is not None,
    )


def measured_arm_joints(session: Any, arm: str) -> list[float]:
    manager = session.environment.robot_manager
    return array(manager.get_joint(session.robots[arm], env_idx_list=[0])[0]).tolist()


def retained_arm_target(session: Any, arm: str) -> list[float]:
    """The joint target the plant keeps applying to an arm no command names.

    The ARX contract holds other arms at their previous joint target
    (``other_arms_hold``), so an arm whose last move ran out of native actions
    is still travelling there during the next command. The measured joints
    stand in when the SDK control buffer holds no target for the arm; a
    malformed retained target fails closed.
    """
    measured = measured_arm_joints(session, arm)
    manager = session.environment.robot_manager
    control = getattr(manager, "control_manager", None)
    entry = (control.prev_control[0].get(manager.process_name(session.robots[arm].arm_name))
             if control is not None else None)
    if not isinstance(entry, Mapping) or "position" not in entry:
        return measured
    target = array(entry["position"]).reshape(-1)
    if target.shape != (len(measured),) or not np.isfinite(target).all():
        raise RobodojoError(f"the retained joint target of the {arm} arm is malformed")
    return target.tolist()


def path_screen(session: Any, arm: str, *, static_other_arm: bool = True,
                ) -> tuple[Callable[[Sequence[float], Sequence[float]], dict[str, Any]], str | None]:
    """A straight-joint-path screen for ``arm`` and the other arm it holds still, if any.

    The self-collision screen runs first; a self-clear path is then screened
    against the other arm, moved in lockstep from its joints measured now to
    the target the plant still applies to it (``retained_arm_target``; usually
    the same joints). Returns the screen and the name of the held arm
    (``None`` when only self-collision is screened).
    """
    manager = session.environment.robot_manager
    model = self_collision_model(session, arm, manager.ik_solver[session.robots[arm].robot_name])
    other = OTHER_ARM.get(arm)
    if not static_other_arm or other not in session.robots:
        return (lambda first, last: dict(model.straight_path(first, last))), None
    screen = inter_arm_screen(session)
    held, driven = measured_arm_joints(session, other), retained_arm_target(session, other)

    def clearance(first: Sequence[float], last: Sequence[float]) -> dict[str, Any]:
        evidence = dict(model.straight_path(first, last))
        if evidence.get("status") != "clear":
            return evidence
        between = screen.straight_paths({arm: list(first), other: held}, {arm: list(last), other: driven})
        return {**evidence, "status": between["status"], "inter_arm": between}

    return clearance, other


def lockstep_inter_arm(session: Any, starts: Mapping[str, Sequence[float]],
                       targets: Mapping[str, Sequence[float]]) -> dict[str, Any] | None:
    """Screen the named arms' straight joint lines in lockstep with every other arm.

    An arm not named moves from its measured joints to the target the plant
    still applies to it. ``None`` when the embodiment has no second arm.
    """
    if set(session.robots) != set(OTHER_ARM):
        return None
    first = {arm: list(starts[arm]) if arm in starts else measured_arm_joints(session, arm) for arm in OTHER_ARM}
    last = {arm: list(targets[arm]) if arm in targets else retained_arm_target(session, arm) for arm in OTHER_ARM}
    return inter_arm_screen(session).straight_paths(first, last)


def inter_arm_screen(session: Any) -> InterArmSphereScreen:
    """Both arms' SDK sphere models at their world bases, bound once per session.

    The base transform is the one ``reachability.fk_link6`` applies to the
    solver's link6 pose (``entity_origin_pose`` after the planner frame bias),
    and the radii carry the solver's own world-collision buffer.
    """
    cached = getattr(session, "_inter_arm_screen", None)
    if cached is not None:
        return cached
    manager = session.environment.robot_manager
    models, bases, buffers = {}, {}, {}
    for arm, robot in session.robots.items():
        planner = manager.ik_solver[robot.robot_name]
        models[arm] = self_collision_model(session, arm, planner)
        try:
            origin = [float(value) for value in robot.entity_origin_pose]
            bases[arm] = InterArmSphereScreen.base_transform(origin[:3], origin[3:], planner.frame_bias)
            kinematics = planner.robot_cfg["robot_cfg"]["kinematics"]
            buffers[arm] = float(kinematics.get("collision_sphere_buffer", 0.0) or 0.0)
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise RobodojoError(f"SDK inter-arm sphere placement is unavailable: {error}") from error
    try:
        screen = InterArmSphereScreen(models, bases, sphere_buffers_m=buffers)
    except JointPathClearanceError as error:
        raise RobodojoError(f"SDK inter-arm sphere model is unavailable: {error}") from error
    session._inter_arm_screen = screen
    return screen


def self_collision_model(session: Any, arm: str, planner: Any) -> SphereSelfCollisionModel:
    """Bind the SDK planner's own URDF and collision spheres once per arm.

    ``CuroboPlanner.robot_cfg`` is the exact kinematics mapping its IK solver
    was built from; the sphere model is therefore the solver's, not a copy.
    Missing or malformed evidence fails closed instead of skipping the check.
    """
    cache = getattr(session, "_self_collision_models", None)
    if cache is None:
        cache = {}
        setattr(session, "_self_collision_models", cache)
    if arm not in cache:
        robot = session.robots[arm]
        try:
            kinematics = planner.robot_cfg["robot_cfg"]["kinematics"]
            urdf_text = Path(kinematics["urdf_path"]).read_text(encoding="utf-8")
            cache[arm] = SphereSelfCollisionModel.from_solver_kinematics(
                kinematics, urdf_text=urdf_text, active_joint_names=list(robot.arm_joints_name))
        except (AttributeError, KeyError, TypeError, OSError, JointPathClearanceError) as error:
            raise RobodojoError(f"SDK self-collision sphere model is unavailable: {error}") from error
    return cache[arm]
