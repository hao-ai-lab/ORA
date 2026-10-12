"""Home and last-safe recovery on the RoboDojo sparse v3 route.

The released route drives each selected arm along one straight joint line to
the stored joints for the whole 20-native-action command. EXP11A g003 recorded
where that fails. In seven ``not_reached`` recoveries across three runs the
arm was wound (joint1 near +/-3.5 rad, joint3 near 3.7-4.9 rad) with its jaws
lowered into or between the task baskets. The first metre of the joint line is
a lateral yaw sweep; the basket wall stopped it, joints 1-3 made no progress
for 20 native actions, and a repeat call did the same (``classify_objects``
round 1 ended 0.42 m / 3.0 rad from home). Unobstructed, the same states still
need about 25 native actions at the measured joint rate of about 0.2 rad per
native action, so a single command ended short while still moving. Both
outcomes were delivered as ``recovery_not_reached``, which the public
projection reports as ``execution_fault``.

With ``runtime.recovery_route = "retract_then_unwind"`` -- derived by the
RunSpec from a surface declaring ``robodojo_retract_then_unwind_recovery_v1``
-- a ``home`` recovery first lifts link6 straight up by up to 0.10 m (one SDK
IK solve inside the local joint-jump bound, orientation held, at most six
native actions), then unwinds in joint space. The unwinding order is the
straight joint line when the SDK's own collision spheres keep it clear, else
fold-then-yaw, else yaw-then-fold; no clear order refuses before any write.
Every segment stops early once each arm has arrived or stalled. The sphere
model covers self-collision only: no scene, table or payload is modelled, and
the retract is a heuristic exit, not a clearance certificate.

On every route the receipt names how the recovery ended per arm
(``recovery_progress``): arrived, stalled (the one RoboDojo motion stall rule,
``motion_stall``: three consecutive native actions each moving no joint
0.005 rad), or still moving when the command window ran out; the failure code
follows (``controller_stalled`` or ``native_action_budget_exhausted``) so a
policy can tell "call again" from "blocked". Every segment is driven by
``motion_stall.drive_until_settled``, the same driver as every sparse move, so
a stalled arm holds the joints measured at its stall.

On every route the joint target is the stored joints' whole-turn equivalent
nearest the measured joints (``unwind_target``), not the literal stored values.
X5A.urdf limits joint1-joint5 to +/-10 rad and IK kept the equivalent nearest
the measured joints, so arms drifted past +/-pi: in EXP11A g005 (audit D3) 100
of 1397 receipts ended with a joint beyond +/-pi -- joint1 in 11, joint2 in 13
and joint3 in 98 (|q3| up to 5.83 rad; joint4 reached 2.07 and joint5 1.58) --
and of 125 per-arm home recoveries the 17 whose ``joint_distance_start_rad``
to literal home exceeded pi ended 14 not arrived, against 17 of the other 108,
since a literal target is a whole turn back. An equivalent must lie inside the
joint's limits and be strictly nearer the measured joint; it may not carry a
pitch joint farther outside [-pi, pi] than it is (the IK rule); and FK must
place it on the stored joints' own link6 pose within float rounding, so the
pose target is unchanged. Non-finite stored or measured joints refuse the
recovery as ``recovery_state_unavailable`` before any write.

In practice the unwind now acts only on the yaw joint joint1: replayed on
those 17 recoveries, joint3 dominated 15 and its turn is vetoed wherever one
arises, 14 keep the literal home, 3 unwind joint1, and only 7abc-0046 then
starts within pi of its target. joint3 winding is prevented at the IK source
instead (not yet measured live). The veto is physically right: carrying joint3
on to 2*pi passes through its fold, and the SDK spheres ignore link2 against
link4.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

from . import ik, inter_arm, motion_stall
from .common import nearest_equivalent_joints, pitch_joint_wraps, pose_error, pose_reached, workspace_violation

RETRACT_THEN_UNWIND = "retract_then_unwind"
RECOVERY_ROUTE_CAPABILITY = "robodojo_retract_then_unwind_recovery_v1"
PROGRESS_SCHEMA = "ora.robodojo-recovery-progress.v1"
UNWIND_TARGET_SCHEMA = "ora.robodojo-recovery-unwind-target.v1"
#: A whole turn of a revolute joint leaves every link pose unchanged up to float
#: rounding: a float32 emulation of the X5A.urdf chain on recorded wound states
#: moved link6 at most 1.1e-7 m and 1.8e-6 rad (review U1 LOW-b), so FK outside
#: these bounds means the proposed joints are not a whole-turn equivalent.
UNWIND_FK_POSITION_TOLERANCE_M = 1e-6
UNWIND_FK_ORIENTATION_TOLERANCE_RAD = 1e-5
RECOVERY_EXECUTOR = "robodojo_sdk_joint_interpolation"
RETRACT_EXECUTOR = "robodojo_sdk_ik_joint_interpolation"
RETRACT_LIFT_M = 0.10
MINIMUM_RETRACT_M = 0.01
RETRACT_NATIVE_ACTIONS = 6
JOINT_ARRIVAL_RAD = 0.02
SELF_COLLISION_REFUSAL = "recovery_self_collision_path"
INTER_ARM_REFUSAL = "recovery_inter_arm_collision_path"
NO_ORDER_FROM_MEASURED = "no_self_clear_order_from_measured_state"
INTER_ARM_FROM_MEASURED = "inter_arm_collision_from_measured"
#: Escalation across calls for one arm within one round: a call that stalls
#: moves the arm to the next strategy, one still moving keeps it, arrival resets
#: it. Replays of recorded stuck arms freed build_tower (g004) only by backing
#: off and imitate_sorting_sequence (g005) only by yawing first.
FALLBACK_STRATEGIES = ("retract_then_unwind", "backoff_then_unwind", "yaw_first_unwind")
BACKOFF_PROBE_NATIVE_ACTIONS = 3
BACKOFF_RAD = 0.2
BACKOFF_NATIVE_ACTIONS = 8
PROBE_PROGRESS_RAD = 0.02
PROBE_REQUIRED_TRAVEL_RAD = 0.05
YAW_FIRST_ORDERS = ("yaw_then_fold", "straight_joint_line", "fold_then_yaw")
#: Straight, or two segments of either fallback order; a re-screen may repeat a nearly done fold.
MAX_UNWIND_SEGMENTS = 3


def configured_route(config: Mapping[str, Any]) -> str | None:
    """Absent keeps the released straight joint-line recovery."""
    route = config.get("recovery_route")
    if route not in (None, RETRACT_THEN_UNWIND):
        from .bridge import RobodojoError
        raise RobodojoError("recovery_route names no supported route")
    return route


def joint_distance(first: Sequence[float], second: Sequence[float]) -> float:
    return max(abs(float(a) - float(b)) for a, b in zip(first, second, strict=True))


def _same_link6_pose(session: Any, arm: str, first: Sequence[float], second: Sequence[float]) -> bool:
    from .reachability import fk_link6

    position, rotation = pose_error(fk_link6(session, arm, list(first)), fk_link6(session, arm, list(second)))
    return position <= UNWIND_FK_POSITION_TOLERANCE_M and rotation <= UNWIND_FK_ORIENTATION_TOLERANCE_RAD


def unwind_target(session: Any, arm: str, stored: Sequence[float], measured: Sequence[float],
                  source: str) -> dict[str, Any]:
    """The stored joints' whole-turn equivalent nearest the measured joints.

    ``status`` is ``not_needed`` (already the nearest), ``unwound`` or
    ``fk_mismatch`` (FK does not confirm the equivalent, so the stored joints
    are kept and the proposed turns are recorded). A turn that would carry a
    pitch joint farther outside [-pi, pi] than it is measured is dropped and
    recorded as ``pitch_vetoed_turns``. No turn is ever added that winds a pitch
    joint; a stored target that already lies farther out (a
    ``backtrack_last_safe`` state recorded before the IK rule) is driven as
    stored and its joints are named in ``stored_pitch_wrap``.
    """
    stored_joints = [float(value) for value in stored]
    names = list(session.robots[arm].arm_joints_name)
    revolute = [name in ik.ARX_X5_REVOLUTE_JOINT_NAMES for name in names]
    _, turns = nearest_equivalent_joints(stored_joints, measured, ik.articulation_limits(session, arm), revolute)
    wound = [value + count * math.tau for value, count in zip(stored_joints, turns)]
    pitch = [name in ik.ARX_X5_PITCH_JOINT_NAMES for name in names]
    vetoed = [index for index in pitch_joint_wraps(wound, measured, pitch) if turns[index]]
    kept = [0 if index in vetoed else count for index, count in enumerate(turns)]
    proposed = [value + count * math.tau for value, count in zip(stored_joints, kept)]
    unwound = any(kept) and _same_link6_pose(session, arm, stored_joints, proposed)
    applied = kept if unwound else [0] * len(kept)
    status = "unwound" if unwound else "fk_mismatch" if any(kept) else "not_needed"
    target = list(proposed if unwound else stored_joints)
    stored_wrap = [names[index] for index in pitch_joint_wraps(target, measured, pitch)]
    return {"schema": UNWIND_TARGET_SCHEMA, "source": source, "status": status, "stored_joints": stored_joints,
            "joint_target": target, "equivalent_turns": applied,
            "correction_rad": [count * math.tau for count in applied],
            **({"rejected_equivalent_turns": kept} if status == "fk_mismatch" else {}),
            **({"pitch_vetoed_turns": [turns[i] if i in vetoed else 0 for i in range(len(turns))]}
               if vetoed else {}),
            **({"stored_pitch_wrap": stored_wrap} if stored_wrap else {})}


def unwind_orders(start: Sequence[float], target: Sequence[float]) -> list[tuple[str, list[tuple[str, list[float]]]]]:
    """Candidate joint-space orders to ``target``, straight line first."""
    goal = [float(value) for value in target]
    fold = [float(start[0]), *goal[1:]]
    yaw = [goal[0], *(float(value) for value in start[1:])]
    return [("straight_joint_line", [("recover", goal)]),
            ("fold_then_yaw", [("recover_fold", fold), ("recover", goal)]),
            ("yaw_then_fold", [("recover_yaw", yaw), ("recover", goal)])]


def select_unwind(
    start: Sequence[float], target: Sequence[float],
    straight_path: Callable[[Sequence[float], Sequence[float]], Mapping[str, Any]],
    preferred: Sequence[str] | None = None,
) -> dict[str, Any]:
    """The first order whose every non-degenerate segment the sphere model keeps clear."""
    screens: list[dict[str, Any]] = []
    orders = unwind_orders(start, target)
    if preferred is not None:
        rank = {name: index for index, name in enumerate(preferred)}
        orders = sorted(orders, key=lambda item: rank.get(item[0], len(rank)))
    for order, segments in orders:
        cursor, kept, clear = [float(value) for value in start], [], True
        for name, joints in segments:
            if joint_distance(cursor, joints) <= 1e-9:
                continue
            screen = dict(straight_path(cursor, joints))
            screens.append({"order": order, "segment": name, "status": screen.get("status"),
                            "min_clearance_m": screen.get("min_clearance_m"),
                            "limiting_pair": screen.get("limiting_pair"),
                            **({"inter_arm_clearance": inter_arm.compact(screen["inter_arm"])}
                               if inter_arm.blocked(screen.get("inter_arm")) else {})})
            if screen.get("status") != "clear":
                clear = False
                break
            kept.append((name, joints))
            cursor = joints
        if clear:
            return {"order": order, "segments": kept or [("recover", [float(v) for v in target])],
                    "screens": screens}
    return {"order": None, "segments": [], "screens": screens}


def _progress(start: Sequence[float], end: Sequence[float], target: Sequence[float], *,
              arrived: bool, halted: bool, route: str, **extra: Any) -> dict[str, Any]:
    outcome = "arrived" if arrived else "stalled" if halted else "still_moving_at_window_end"
    return {"schema": PROGRESS_SCHEMA, "route": route, "outcome": outcome,
            "joint_distance_start_rad": joint_distance(start, target),
            "joint_distance_end_rad": joint_distance(end, target),
            "stall_rule": motion_stall.stall_rule(), **extra}


def failure_code(progress: Mapping[str, Mapping[str, Any]], *, terminal: bool) -> str | None:
    outcomes = [item["outcome"] for item in progress.values()]
    if all(outcome == "arrived" for outcome in outcomes):
        return None
    if terminal:
        return motion_stall.TERMINAL_CODE
    moving = any(outcome == "still_moving_at_window_end" for outcome in outcomes)
    if moving:
        return motion_stall.BUDGET_CODE
    if any(outcome.startswith("refused_") for outcome in outcomes):
        reasons = {(item.get("stopped") or {}).get("reason") for item in progress.values()}
        return INTER_ARM_REFUSAL if INTER_ARM_FROM_MEASURED in reasons else SELF_COLLISION_REFUSAL
    return motion_stall.STALL_CODE


def _run_segment(execution: Any, name: str, joints: Mapping[str, Sequence[float]],
                 arrived: Mapping[str, Callable[[Mapping[str, Any]], bool]], budget: int,
                 poses: Mapping[str, Sequence[float] | None], recovery: Mapping[str, str]) -> dict[str, dict]:
    """Drive one synchronized joint segment until every arm arrived or stalled.

    The segment is first screened in lockstep from the measured joints against
    the other arm; a blocked segment is not driven and its arms report ``refused``.
    """
    evidence = inter_arm.screen_from_measured(execution.session, joints)
    if inter_arm.blocked(evidence):
        return _refuse_segment(execution, name, joints, poses, recovery, evidence)
    used, stall = motion_stall.drive_until_settled(
        execution, segment=name, joint_targets=joints, arrived=arrived,
        budget=max(0, min(budget, execution.remaining())))
    after = execution.session._proprioception()["end_effectors"]
    motion_stall.record_progress(execution, stall, after, segment=name, arrived=arrived)
    outcome = {}
    for arm, target in joints.items():
        reached, halted = arrived[arm](after[arm]), stall.stalled(arm)
        pose = poses.get(arm)
        entry = {"segment": name, "target_pose_xyz_xyzw": None if pose is None else [float(v) for v in pose],
                 "joint_target": [float(v) for v in target], "recovery": recovery[arm],
                 "native_actions": used, "reached": reached, "stalled": halted and not reached,
                 "joint_distance_end_rad": joint_distance(after[arm]["arm_joint_qpos"], target)}
        if pose is not None:
            entry["position_error_m"], entry["orientation_error_rad"] = pose_error(pose, after[arm]["pose_xyz_xyzw"])
        execution.segments[arm].append(entry)
        outcome[arm] = {"reached": reached, "stalled": halted and not reached, "joints": after[arm]["arm_joint_qpos"]}
    return outcome


def _refuse_segment(execution: Any, name: str, joints: Mapping[str, Sequence[float]],
                    poses: Mapping[str, Sequence[float] | None], recovery: Mapping[str, str],
                    evidence: Mapping[str, Any]) -> dict[str, dict]:
    measured = execution.session._proprioception()["end_effectors"]
    outcome = {}
    for arm, target in joints.items():
        pose = poses.get(arm)
        execution.segments[arm].append({
            "segment": name, "target_pose_xyz_xyzw": None if pose is None else [float(v) for v in pose],
            "joint_target": [float(v) for v in target], "recovery": recovery[arm], "native_actions": 0,
            "reached": False, "stalled": False, "refused": ik.INTER_ARM_OUTCOME,
            "inter_arm_clearance": inter_arm.compact(evidence)})
        outcome[arm] = {"reached": False, "stalled": False, "refused": True, "joints": measured[arm]["arm_joint_qpos"]}
    return outcome


def _retract_plan(execution: Any, arm: str, strategy: str) -> dict[str, Any]:
    session = execution.session
    if strategy != "home":
        return {"status": "skipped", "reason": "not_a_home_recovery"}
    pose = [float(v) for v in execution.before[arm]["pose_xyz_xyzw"]]
    ceiling = float(session.config["workspace_bounds_m"]["z"][1])
    lifted = [pose[0], pose[1], min(pose[2] + RETRACT_LIFT_M, ceiling), *pose[3:]]
    if lifted[2] - pose[2] < MINIMUM_RETRACT_M or workspace_violation(lifted, session.config["workspace_bounds_m"]):
        return {"status": "skipped", "reason": "workspace_ceiling", "lift_m": lifted[2] - pose[2]}
    # Two recovering arms retract together: the pair is screened in lockstep later.
    solved = ik.solve_ik(session, arm, lifted, jump_limit_rad=session.local_joint_jump_limit_rad,
                         static_other_arm=len(execution.arms) == 1)
    execution.record_ik("retract", arm, lifted, solved)
    if solved["joint_target"] is None:
        refused = solved["ik_selection"]["outcome"] == ik.INTER_ARM_OUTCOME
        return {"status": "skipped", "reason": ik.INTER_ARM_OUTCOME if refused else "ik_failed",
                "lift_m": lifted[2] - pose[2]}
    if solved["joint_delta_rad"] > session.local_joint_jump_limit_rad:
        return {"status": "skipped", "reason": "ik_branch_jump", "lift_m": lifted[2] - pose[2],
                "joint_delta_rad": solved["joint_delta_rad"]}
    return {"status": "planned", "lift_m": lifted[2] - pose[2], "pose": lifted, "joints": solved["joint_target"]}


def _joint_arrival(target: Sequence[float]) -> Callable[[Mapping[str, Any]], bool]:
    return lambda state: joint_distance(state["arm_joint_qpos"], target) <= JOINT_ARRIVAL_RAD


def _pose_arrival(target: Sequence[float]) -> Callable[[Mapping[str, Any]], bool]:
    return lambda state: pose_reached(target, state["pose_xyz_xyzw"])



def _fallback_state(session: Any) -> dict[str, Any]:
    """Per-round escalation levels; a new round starts every arm at level 0."""
    state = getattr(session, "_recovery_fallback", None)
    if not isinstance(state, dict) or state.get("round") != session.round_index:
        state = {"round": session.round_index, "levels": {}}
        session._recovery_fallback = state
    return state


def _backoff(execution: Any, arms: Sequence[str], joints: Mapping[str, Sequence[float]],
             poses: Mapping[str, Sequence[float]], screens: Mapping[str, Any], labels: Mapping[str, str],
             unwind: Mapping[str, dict[str, Any]], blocked: dict[str, list[int]]) -> None:
    """Probe toward home, then move the joints that made no progress away from it."""
    session = execution.session
    if not arms or session.is_terminal or execution.remaining() <= 0:
        return
    before = session._proprioception()["end_effectors"]
    start = {arm: [float(v) for v in before[arm]["arm_joint_qpos"]] for arm in arms}
    probe = _run_segment(execution, "recover_probe", {arm: joints[arm] for arm in arms},
                         {arm: _pose_arrival(poses[arm]) for arm in arms}, BACKOFF_PROBE_NATIVE_ACTIONS,
                         {arm: poses[arm] for arm in arms}, labels)
    after = session._proprioception()["end_effectors"]
    targets: dict[str, list[float]] = {}
    for arm in arms:
        if probe[arm].get("refused"):
            # The probe never ran (the other arm is in its way): no joint showed
            # it cannot move, so nothing is backed off on that evidence.
            blocked[arm] = []
            continue
        end = [float(v) for v in after[arm]["arm_joint_qpos"]]
        stuck = [index for index in range(len(end))
                 if abs(float(joints[arm][index]) - start[arm][index]) > PROBE_REQUIRED_TRAVEL_RAD
                 and abs(end[index] - start[arm][index]) < PROBE_PROGRESS_RAD]
        blocked[arm] = [index + 1 for index in stuck]
        if not stuck:
            continue
        target = list(end)
        for index in stuck:
            target[index] += BACKOFF_RAD * (1.0 if end[index] > float(joints[arm][index]) else -1.0)
        screen = dict(screens[arm](end, target))
        unwind[arm]["screens"].append({"order": "backoff", "segment": "recover_backoff",
                                       "status": screen.get("status"),
                                       "min_clearance_m": screen.get("min_clearance_m"),
                                       "limiting_pair": screen.get("limiting_pair"),
                                       **({"inter_arm_clearance": inter_arm.compact(screen["inter_arm"])}
                                          if inter_arm.blocked(screen.get("inter_arm")) else {})})
        if screen.get("status") == "clear":
            targets[arm] = target
    if targets and not session.is_terminal and execution.remaining() > 0:
        _run_segment(execution, "recover_backoff", targets,
                     {arm: _joint_arrival(target) for arm, target in targets.items()},
                     min(BACKOFF_NATIVE_ACTIONS, execution.remaining()), {arm: None for arm in targets}, labels)


def _unwind_from_measured(execution: Any, unwind: Mapping[str, dict[str, Any]], screens: Mapping[str, Any],
                          joints: Mapping[str, Sequence[float]], poses: Mapping[str, Sequence[float]],
                          labels: Mapping[str, str], halted: dict[str, bool],
                          stopped: dict[str, dict[str, str]],
                          preferred: Mapping[str, Sequence[str] | None] | None = None) -> None:
    """Execute unwind segments, each screened from the measured joints it starts from.

    An arm stops when a segment stalls or no self-clear order exists from where
    it actually is; the other arm continues. Only the first segment of each
    re-screened order runs before the next measurement.
    """
    session = execution.session
    active = [arm for arm in execution.arms if arm not in stopped]
    last_stage = {arm: "retract" for arm in execution.arms}
    for _ in range(MAX_UNWIND_SEGMENTS):
        if not active or session.is_terminal or execution.remaining() <= 0:
            return
        measured = session._proprioception()["end_effectors"]
        step: dict[str, tuple[str, list[float]]] = {}
        for arm in active:
            plan = select_unwind(measured[arm]["arm_joint_qpos"], joints[arm], screens[arm],
                                 (preferred or {}).get(arm))
            unwind[arm]["screens"].extend(plan["screens"])
            if plan["order"] is None:
                stopped[arm] = {"stage": last_stage[arm], "reason": NO_ORDER_FROM_MEASURED}
                continue
            name, target = plan["segments"][0]
            step[arm] = (name, [float(value) for value in target])
        active = [arm for arm in active if arm in step]
        if not step:
            return
        final = {arm: name == "recover" for arm, (name, _) in step.items()}
        name = next((segment for segment, _ in step.values() if segment != "recover"), "recover")
        outcome = _run_segment(
            execution, name, {arm: target for arm, (_, target) in step.items()},
            {arm: (_pose_arrival(poses[arm]) if final[arm] else _joint_arrival(target))
             for arm, (_, target) in step.items()}, execution.remaining(),
            {arm: (poses[arm] if final[arm] else None) for arm in step}, labels)
        for arm, (segment, _) in step.items():
            last_stage[arm] = segment
            if outcome[arm].get("refused"):
                stopped[arm] = {"stage": segment, "reason": INTER_ARM_FROM_MEASURED}
                active.remove(arm)
            elif outcome[arm]["reached"] and final[arm]:
                active.remove(arm)
            elif outcome[arm]["stalled"]:
                halted[arm] = True
                stopped[arm] = {"stage": segment, "reason": "segment_stalled"}
                active.remove(arm)


def _recovery_targets(execution: Any, targets: Mapping[str, str]) -> tuple[dict, dict, dict] | None:
    """Per-arm unwound joint target, stored pose and unwind record; ``None`` when state is missing."""
    session = execution.session
    joints: dict[str, list[float]] = {}
    poses: dict[str, list[float]] = {}
    unwound: dict[str, dict[str, Any]] = {}
    for arm in execution.arms:
        home = targets[arm] == "home"
        source_joints = session.home_joints if home else session.last_safe_joints
        source_poses = session.home_poses if home else session.last_safe_poses
        if arm not in source_joints or arm not in source_poses:
            return None
        stored, measured = source_joints[arm], execution.before[arm]["arm_joint_qpos"]
        if not all(math.isfinite(float(value)) for value in [*stored, *measured]):
            return None
        unwound[arm] = unwind_target(session, arm, stored, measured, targets[arm])
        joints[arm], poses[arm] = list(unwound[arm]["joint_target"]), list(source_poses[arm])
    return joints, poses, unwound


def _screen_retract(session: Any, start: Mapping[str, Sequence[float]], retract: dict[str, dict[str, Any]]) -> None:
    """Skip planned retracts whose lockstep lift meets the other arm; the retract is a heuristic exit."""
    planned = {arm: plan["joints"] for arm, plan in retract.items() if plan["status"] == "planned"}
    evidence = ik.lockstep_inter_arm(session, {arm: start[arm] for arm in planned}, planned) if planned else None
    if inter_arm.blocked(evidence):
        for arm in planned:
            retract[arm] = {"status": "skipped", "reason": ik.INTER_ARM_OUTCOME, "lift_m": retract[arm]["lift_m"],
                            "inter_arm_clearance": inter_arm.compact(evidence)}


def _first_step_evidence(session: Any, start: Mapping[str, Sequence[float]],
                         retract: Mapping[str, Mapping[str, Any]],
                         unwind: Mapping[str, Mapping[str, Any]]) -> dict[str, Any] | None:
    """Lockstep screen of each arm's first unwind segment, from where its retract is planned to end."""
    cursors = {arm: (retract[arm]["joints"] if retract.get(arm, {}).get("status") == "planned" else start[arm])
               for arm in unwind}
    return ik.lockstep_inter_arm(session, cursors, {arm: plan["segments"][0][1] for arm, plan in unwind.items()})


def _order_refusal(unwind: Mapping[str, Mapping[str, Any]]) -> str:
    """The other arm's code when it blocked any order of a refused arm: moving it could free that order."""
    other_arm = any(screen.get("status") == "inter_arm_collision" for plan in unwind.values()
                    if plan["order"] is None for screen in plan["screens"])
    return INTER_ARM_REFUSAL if other_arm else SELF_COLLISION_REFUSAL


def _refuse_before_motion(execution: Any, *, start: Mapping[str, Sequence[float]], joints: Mapping[str, Any],
                          poses: Mapping[str, Any], unwound: Mapping[str, Any], retract: Mapping[str, Any],
                          unwind: Mapping[str, Mapping[str, Any]], route: str, code: str,
                          evidence: Mapping[str, Any] | None = None) -> dict[str, Any]:
    clearance = {} if evidence is None else {"inter_arm_clearance": inter_arm.compact(evidence)}
    progress = {arm: {**_progress(start[arm], start[arm], joints[arm], arrived=False, halted=False,
                                  route=route, unwind_target=unwound[arm], retract=retract.get(arm),
                                  unwind_order=unwind[arm]["order"], path_screens=unwind[arm]["screens"]),
                      "outcome": "refused_before_motion", **clearance}
                for arm in execution.arms}
    for plan in retract.values():
        plan.pop("joints", None)  # the planned joint vector stays Control-side, as on the driven route
    return execution.finish(status="plan_rejected", strategy=None, stage="plan_validation",
                            code=code, executor=None, target_poses=poses,
                            arm_extras={arm: {"recovery_progress": value} for arm, value in progress.items()})


def _stop_label(stop: Mapping[str, str]) -> str:
    """The refused outcome of an arm stopped by a screen from the measured state."""
    if stop["stage"] == "retract":
        # An inter-arm refusal of the retract drove nothing.
        return "refused_before_motion" if stop["reason"] == INTER_ARM_FROM_MEASURED else "refused_after_retract"
    return "refused_mid_unwind"


def execute_recovery(execution: Any, targets: Mapping[str, str]) -> dict[str, Any]:
    """Run one ``recover_motion`` and finish its receipt with per-arm progress."""
    session = execution.session
    resolved = _recovery_targets(execution, targets)
    if resolved is None:
        return execution.finish(status="plan_rejected", strategy=None, stage="plan_validation",
                                code="recovery_state_unavailable", executor=None)
    joints, poses, unwound = resolved
    start = {arm: list(execution.before[arm]["arm_joint_qpos"]) for arm in execution.arms}
    segmented = getattr(session, "recovery_route", None) == RETRACT_THEN_UNWIND
    fallback = _fallback_state(session) if segmented else {"levels": {}}
    levels = {arm: int(fallback["levels"].get(arm, 0)) % len(FALLBACK_STRATEGIES) for arm in execution.arms}
    retract = {arm: (_retract_plan(execution, arm, targets[arm]) if levels[arm] == 0 else
                     {"status": "skipped", "reason": f"fallback_{FALLBACK_STRATEGIES[levels[arm]]}"})
               for arm in execution.arms} if segmented else {}
    _screen_retract(session, start, retract)
    # Admission ranks orders as execution will (yaw-first at level 2).
    preferred = {arm: (YAW_FIRST_ORDERS if levels[arm] == 2 else None) for arm in execution.arms}
    unwind: dict[str, dict[str, Any]] = {}
    screens: dict[str, Any] = {}
    for arm in execution.arms:
        if not segmented:
            unwind[arm] = {"order": "straight_joint_line", "segments": [("recover", joints[arm])], "screens": []}
            continue
        # Admission before any write screens from where the retract is planned
        # to end; execution re-screens every segment from the measured state.
        # One recovering arm ranks its orders with the other arm held in the
        # screen; two recovering arms are screened together step by step.
        cursor = retract[arm]["joints"] if retract[arm]["status"] == "planned" else start[arm]
        screens[arm], _ = ik.path_screen(session, arm, static_other_arm=len(execution.arms) == 1)
        unwind[arm] = select_unwind(cursor, joints[arm], screens[arm], preferred[arm])
    route = RETRACT_THEN_UNWIND if segmented else "straight_joint_line"
    refusal = {"start": start, "joints": joints, "poses": poses, "unwound": unwound, "retract": retract,
               "unwind": unwind, "route": route}
    if any(plan["order"] is None for plan in unwind.values()):
        return _refuse_before_motion(execution, code=_order_refusal(unwind), **refusal)
    first = _first_step_evidence(session, start, retract, unwind)
    if inter_arm.blocked(first):
        return _refuse_before_motion(execution, code=INTER_ARM_REFUSAL, evidence=first, **refusal)
    execution.begin([execution.arm_key(arm) for arm in execution.arms])
    labels = {arm: targets[arm] for arm in execution.arms}
    executors = [RECOVERY_EXECUTOR]
    halted = {arm: False for arm in execution.arms}
    stopped: dict[str, dict[str, str]] = {}
    planned = {arm: plan for arm, plan in retract.items() if plan["status"] == "planned"}
    if planned:
        executors = [RETRACT_EXECUTOR, RECOVERY_EXECUTOR]
        outcome = _run_segment(
            execution, "retract", {arm: plan["joints"] for arm, plan in planned.items()},
            {arm: _pose_arrival(plan["pose"]) for arm, plan in planned.items()}, RETRACT_NATIVE_ACTIONS,
            {arm: plan["pose"] for arm, plan in planned.items()}, labels)
        for arm, plan in planned.items():
            plan["outcome"] = ("refused" if outcome[arm].get("refused") else "arrived" if outcome[arm]["reached"]
                               else "stalled" if outcome[arm]["stalled"] else "still_moving_at_window_end")
            if plan["outcome"] == "refused":
                stopped[arm] = {"stage": "retract", "reason": INTER_ARM_FROM_MEASURED}
            if plan["outcome"] == "stalled":
                # A stalled lift is a contact the sphere model does not cover;
                # unwinding from there is exactly the unscreened motion.
                halted[arm] = True
                stopped[arm] = {"stage": "retract", "reason": "retract_stalled"}
    blocked: dict[str, list[int]] = {}
    if segmented:
        _backoff(execution, [arm for arm in execution.arms if levels[arm] == 1 and arm not in stopped],
                 joints, poses, screens, labels, unwind, blocked)
        _unwind_from_measured(execution, unwind, screens, joints, poses, labels, halted, stopped, preferred)
    elif not session.is_terminal and execution.remaining() > 0:
        outcome = _run_segment(
            execution, "recover", {arm: joints[arm] for arm in execution.arms},
            {arm: _pose_arrival(poses[arm]) for arm in execution.arms}, execution.remaining(),
            {arm: poses[arm] for arm in execution.arms}, labels)
        halted = {arm: outcome[arm]["stalled"] for arm in execution.arms}
        stopped.update({arm: {"stage": "recover", "reason": INTER_ARM_FROM_MEASURED}
                        for arm in execution.arms if outcome[arm].get("refused")})
    for arm in execution.arms:
        # A stalled arm holds the joints measured at its stall, and that pose.
        session.pose_targets[arm] = motion_stall.pose_target(execution, arm, poses[arm])
    after = session._proprioception()["end_effectors"]
    progress = {arm: _progress(
        start[arm], after[arm]["arm_joint_qpos"], joints[arm],
        arrived=pose_reached(poses[arm], after[arm]["pose_xyz_xyzw"]), halted=halted[arm], route=route,
        unwind_target=unwound[arm],
        **({"retract": retract[arm], "unwind_order": unwind[arm]["order"],
            "unwind_screened_from": "measured_joint_state", "stopped": stopped.get(arm),
            "fallback_level": levels[arm], "fallback_strategy": FALLBACK_STRATEGIES[levels[arm]],
            **({"blocked_joints": blocked.get(arm, [])} if levels[arm] == 1 else {}),
            **({"unwind_orders_preferred": list(YAW_FIRST_ORDERS)} if levels[arm] == 2 else {}),
            "path_screens": unwind[arm]["screens"]} if segmented else
           {"stopped": stopped[arm]} if arm in stopped else {}),
    ) for arm in execution.arms}
    for arm, stop in stopped.items():
        if (stop["reason"] in (NO_ORDER_FROM_MEASURED, INTER_ARM_FROM_MEASURED)
                and progress[arm]["outcome"] != "arrived"):
            progress[arm]["outcome"] = _stop_label(stop)
    if segmented:
        for arm in execution.arms:
            outcome = progress[arm]["outcome"]
            if outcome == "arrived":
                fallback["levels"][arm] = 0
            elif outcome != "still_moving_at_window_end":
                fallback["levels"][arm] = (levels[arm] + 1) % len(FALLBACK_STRATEGIES)
    for plan in retract.values():
        plan.pop("joints", None)
    code = failure_code(progress, terminal=session.is_terminal)
    return execution.finish(
        status="reached" if code is None else "not_reached", strategy="global",
        stage=None if code is None else "execution", code=code, executor=executors, target_poses=poses,
        arm_extras={arm: {"recovery_progress": value} for arm, value in progress.items()},
    )


__all__ = [
    "INTER_ARM_FROM_MEASURED", "INTER_ARM_REFUSAL", "PROGRESS_SCHEMA", "RECOVERY_EXECUTOR",
    "RECOVERY_ROUTE_CAPABILITY", "RETRACT_EXECUTOR",
    "RETRACT_LIFT_M", "RETRACT_THEN_UNWIND", "SELF_COLLISION_REFUSAL", "UNWIND_FK_ORIENTATION_TOLERANCE_RAD",
    "UNWIND_FK_POSITION_TOLERANCE_M", "UNWIND_TARGET_SCHEMA",
    "configured_route", "execute_recovery", "failure_code", "joint_distance", "select_unwind",
    "unwind_orders", "unwind_target",
]
