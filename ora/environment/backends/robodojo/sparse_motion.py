"""Sparse motion v3 over the qualified RoboDojo SDK IK and joint interpolation.

v1 global moves are one SDK CuRobo IK solve followed by the original
joint-target interpolation, not CuRobo trajectory planning: they certify a
kinematic target, never a collision-free path, and every receipt discloses
that as ``non_parity_motion`` plan validation.  Local moves compile one
bounded Cartesian step through the shared servo contract and solve it with
the same IK.  All segments of one command share the fixed 20-native-action
budget so the controller contract and the evidence joins stay unchanged; a
segment ends early once every arm arrived or stalled (``motion_stall``).
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import time
from typing import Any, Mapping, Sequence

import numpy as np

from ora.contracts.action import SERVO_MAX_ROTATION_RAD, SERVO_MAX_TRANSLATION_M
from ora.contracts.motion import compile_servo_delta
from ora.contracts.planning_diagnostics import (
    IK_CONSTRAINT_KEYS, PROBE_FEEDBACK_REQUEST_SCHEMA, PROBE_IK_CANDIDATE_FEEDBACK,
    public_planning_diagnostics,
)
from ora.contracts.robodojo_arx import ARX_X5_CONTRACT
from ora.environment.controllers.motion_transaction import write_motion_plan_artifact

from . import inter_arm, motion_stall
from .bridge import RobodojoError
from .ik import solve_ik
from ora.environment.controllers.gripper_stall import (
    GRIPPER_STALL_REASON, gripper_outcome, gripper_outcome_state, gripper_stalled,
)

from .common import (
    ARX_GRIPPER_STALL, GRIPPER_TOLERANCE, array, free_space_segments, gripper_opening, native_pose,
    physical_gripper, pose_error, pose_reached, primary_limit_holds_close, workspace_violation, write_json,
)


SPARSE_RECEIPT_SCHEMA = "ora.robodojo-sparse-motion-receipt.v1"
PLAN_REFERENCE_SCHEMA = "ora.robodojo-plan-reference.v1"
PROBE_REQUEST_SCHEMA = "ora.motion-probe-request.v1"
PROBE_RESULT_SCHEMA = "ora.motion-probe-result.v1"
GLOBAL_EXECUTOR = "robodojo_sdk_ik_joint_interpolation"
LOCAL_EXECUTOR = "robodojo_sdk_ik_local_cartesian_step"
RECOVERY_EXECUTOR = "robodojo_sdk_joint_interpolation"
GRIPPER_EXECUTOR = "robodojo_sdk_gripper_actuator"
GLOBAL_PLANNING_NON_PARITY = "sdk_ik_joint_interpolation_not_curobo_planning"
PLAN_VALIDATION = {
    "schema": "ora.robodojo-plan-validation.v1", "status": "kinematics_only",
    "global_planner": "sdk_ik_joint_interpolation", "parity": "non_parity_motion",
    "collision_model": None,
}
FREE_SPACE_CLEARANCE_M = 0.05
LOCAL_JOINT_JUMP_LIMIT_RAD = 1.0
# IK endpoints are not paths. Preserve the half-turn raw-coordinate bound
# after legal equivalent-coordinate selection; a live 179-degree probe once
# returned a 6.5 rad jump and drove the arm 0.5 m off.
GLOBAL_JOINT_JUMP_LIMIT_RAD = math.pi
PLAN_REFERENCE_JOINT_TOLERANCE_RAD = 0.02
ROUTE_REFUSAL_CODE = "route_contact_approach_exceeds_local_bound"
#: Opt-in staged route for a large contact approach (see ``staged_contact``).
STAGED_CONTACT_STRATEGY = "global_precontact_then_local"
ARMS = ("left", "right")
# Parallel-jaw symmetry axis: joint7/joint8 close along link6 +/-Y from an
# origin 0.087 m along link6 +X (X5A.urdf), so the approach axis is link6 +X.
JAW_SYMMETRY_AXIS = (1.0, 0.0, 0.0)


def within_local_envelope(translation_m: float, rotation_rad: float) -> bool:
    """Route locally only when the servo contract itself will accept the step."""
    return (translation_m <= SERVO_MAX_TRANSLATION_M + 1e-12
            and rotation_rad <= SERVO_MAX_ROTATION_RAD + 1e-12)


def _quaternion_multiply(left: Sequence[float], right: Sequence[float]) -> list[float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    value = np.asarray([
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ], dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm < 1e-12:
        raise RobodojoError("rotation composition produced an invalid orientation")
    return (value / norm).tolist()


def rotate_about_local_axis(orientation_xyzw: Sequence[float], axis: Sequence[float], angle_rad: float) -> list[float]:
    unit = np.asarray(axis, dtype=np.float64)
    unit /= np.linalg.norm(unit)
    half = angle_rad / 2.0
    delta = [*(unit * math.sin(half)).tolist(), math.cos(half)]
    return _quaternion_multiply(orientation_xyzw, delta)


def rotation_vector_between(target_xyzw: Sequence[float], current_xyzw: Sequence[float]) -> list[float]:
    """Return the world-frame rotation vector taking ``current`` to ``target``."""
    cx, cy, cz, cw = (float(item) for item in current_xyzw)
    error = np.asarray(_quaternion_multiply(target_xyzw, [-cx, -cy, -cz, cw]))
    if error[3] < 0:
        error = -error
    vector_norm = float(np.linalg.norm(error[:3]))
    if vector_norm < 1e-12:
        return [0.0, 0.0, 0.0]
    angle = 2.0 * math.atan2(vector_norm, float(error[3]))
    return (error[:3] * (angle / vector_norm)).tolist()


def _pose_list(target: Mapping[str, Any]) -> list[float]:
    return [float(item) for item in target["position_m"]] + [float(item) for item in target["orientation_xyzw"]]


def gripper_closed_floor(session: Any, arm: str) -> float:
    """Normalized opening at the primary finger joint's lower limit.

    The command scale starts below the X5 finger joint's lower limit, so a
    ``close`` directive of 0.0 is never measurable; the plant stops at the
    articulation's own limit and that is the honest closed target.
    """
    manager, robot = session.environment.robot_manager, session.robots[arm]
    data = manager.robot_key[manager.robot_list.index(robot)].data
    for name in ("joint_pos_limits", "joint_limits"):
        if hasattr(data, name):
            limits = array(getattr(data, name))[0]
            return gripper_opening(robot, float(limits[int(robot.gripper_joint_indices[0])][0]))
    return 0.0


class _SparseAction:
    """Evidence and budget of one sparse v3 action across its segments."""

    def __init__(self, session: Any, action: Mapping[str, Any], arms: Sequence[str]) -> None:
        self.session = session
        self.action = action
        self.arms = tuple(arms)
        self.manager = session.environment.robot_manager
        self.before = session._proprioception()["end_effectors"]
        self.command_before = deepcopy(self.manager.control_manager.prev_control[0])
        if getattr(session.environment, "interact", False):
            from .interaction_control import policy_targets
            self.command_before = policy_targets(session, self.command_before)
        self.held_targets: dict[str, Any] = deepcopy(self.command_before)
        self.pose_targets_before = deepcopy(session.pose_targets)
        self.gripper_directives_before = deepcopy(session.gripper_directives)
        self.native_before = int(session.environment.take_action_cnt[0])
        self.steps_before = session.sim_steps
        self.started = time.time()
        self.budget = int(session.native_actions_per_command)
        self.used = 0
        self.tracking: list[dict[str, Any]] = []
        self.segments: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
        self.ik_attempts: list[dict[str, Any]] = []
        self.stall = motion_stall.ActionStallState()
        self.journal = session._open_journal()

    def record_ik(self, segment: str, arm: str, pose: Sequence[float], solved: Mapping[str, Any]) -> dict[str, Any]:
        """Keep all candidate evidence Control-side and a compact public choice."""
        self.ik_attempts.append({"arm": arm, "segment": segment,
                                 "target_pose_xyz_xyzw": list(pose), **deepcopy(dict(solved))})
        return {"ik_status": solved["ik_status"], "joint_delta_rad": solved["joint_delta_rad"],
                "ik_selection": deepcopy(solved.get("ik_selection"))}

    def rejected_ik(self, segment: str, arm: str, pose: Sequence[float],
                    solved: Mapping[str, Any], summary: Mapping[str, Any]) -> None:
        self.segments[arm].append({"segment": segment, "target_pose_xyz_xyzw": list(pose),
                                  "joint_target": deepcopy(solved["joint_target"]), **dict(summary),
                                  "native_actions": 0, "reached": False})

    def begin(self, controlled_keys: Sequence[str]) -> None:
        self.held_targets = {key: value for key, value in self.command_before.items() if key not in set(controlled_keys)}

    def arm_key(self, arm: str) -> str:
        return self.manager.process_name(self.session.robots[arm].arm_name)

    def gripper_key(self, arm: str) -> str:
        return self.manager.process_name(self.session.robots[arm].gripper_name)

    def remaining(self) -> int:
        return self.budget - self.used

    def run_joint_segment(
        self, segment: str, joint_targets: Mapping[str, Sequence[float]],
        pose_targets: Mapping[str, Sequence[float]], planned: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> bool:
        """Drive until every arm arrived or stalled, or the budget ends; ``True`` when all arrived."""
        arrival = {arm: motion_stall.pose_arrival(pose) for arm, pose in pose_targets.items()}
        before_used = self.used
        _, stall = motion_stall.drive_until_settled(
            self, segment=segment, joint_targets=joint_targets, arrived=arrival, budget=self.remaining())
        after = self.session._proprioception()["end_effectors"]
        for arm, pose in pose_targets.items():
            self.session.pose_targets[arm] = [float(item) for item in pose]
            distance, angle = pose_error(pose, after[arm]["pose_xyz_xyzw"])
            self.segments[arm].append({
                "segment": segment, "target_pose_xyz_xyzw": [float(item) for item in pose],
                "joint_target": [float(item) for item in joint_targets[arm]],
                **dict((planned or {}).get(arm, {})),
                "native_actions": self.used - before_used,
                "reached": pose_reached(pose, after[arm]["pose_xyz_xyzw"]),
                "position_error_m": distance, "orientation_error_rad": angle,
            })
        # After the targets above: a stalled arm's pose target becomes where it holds.
        motion_stall.record_progress(self, stall, after, segment=segment, arrived=arrival)
        return all(arrival[arm](after[arm]) for arm in pose_targets)

    def run_gripper(self, directives: Mapping[str, float], effective: Mapping[str, float]) -> dict[str, str]:
        """Drive the jaws until each arrives or stalls; name how each one ended.

        A jaw stopped on an object holds the SDK-limited drive target 0.2 below
        its primary finger's encoder and never arrives, so the command ends once
        every targeted jaw has arrived or stalled, both read on the published
        two-finger mean (E11G12-001), instead of spending the budget.
        """
        control = {self.gripper_key(arm): {"position": physical_gripper(self.session.robots[arm], value)}
                   for arm, value in directives.items()}
        traces = {arm: [float(self.before[arm]["gripper"])] for arm in effective}

        def arrived(current: Mapping[str, Any], arm: str) -> bool:
            return abs(current[arm]["gripper"] - effective[arm]) <= GRIPPER_TOLERANCE

        def settled(current: Mapping[str, Any]) -> bool:
            # Called exactly once after every native action.
            for arm in effective:
                traces[arm].append(float(current[arm]["gripper"]))
            return all(arrived(current, arm) or gripper_stalled(
                traces[arm], target=effective[arm], rule=ARX_GRIPPER_STALL) for arm in effective)

        budget = self.remaining()
        used = self.session._run_native_control(
            action_id=str(self.action["action_id"]), control=control, held_targets=self.held_targets,
            journal=self.journal, tracking=self.tracking, segment="gripper", reached=settled,
            budget=budget,
        )
        self.used += used
        after = self.session._proprioception()["end_effectors"]
        return {arm: gripper_outcome_state(
            reached=arrived(after, arm),
            stalled=gripper_stalled(traces[arm], target=effective[arm], rule=ARX_GRIPPER_STALL),
            ticks_used=used, ticks_budget=budget,
        ) for arm in effective}

    def finish(
        self, *, status: str, strategy: str | None, stage: str | None, code: str | None,
        executor: list[str] | None, interactions: Mapping[str, str] | None = None,
        target_poses: Mapping[str, Sequence[float]] | None = None,
        plan_refs: Mapping[str, str] | None = None, gripper_targets: Mapping[str, float] | None = None,
        gripper_states: Mapping[str, str] | None = None,
        precontact: Mapping[str, Mapping[str, Any]] | None = None,
        arm_extras: Mapping[str, Mapping[str, Any]] | None = None, extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        session = self.session
        after = session._proprioception()["end_effectors"]
        command_after = deepcopy(self.manager.control_manager.prev_control[0])
        if getattr(session.environment, "interact", False):
            from .interaction_control import policy_targets
            command_after = policy_targets(session, command_after)
        held_verified = session._held_targets_unchanged(self.held_targets)
        if not held_verified:
            session._fault("robodojo_held_command_changed")
            raise RobodojoError("action changed a held SDK command")
        pose_action = gripper_targets is None
        hold = not self.arms  # ``wait``: no dimension is controlled, every one is held
        gripper_floors = {arm: gripper_closed_floor(session, arm) for arm in (gripper_targets or {})}
        arms_effect: dict[str, Any] = {}
        for arm in ARMS:
            targeted = arm in self.arms
            effect: dict[str, Any] = {
                "targeted": targeted, "interaction": (interactions or {}).get(arm),
                "pose_error": None, "gripper_error": None, "plan_validation": None,
                "plan_ref": (plan_refs or {}).get(arm), "segments": self.segments[arm],
                "inactive_arm_drift": None,
            }
            attempts = [attempt for attempt in self.ik_attempts if attempt["arm"] == arm]
            if attempts:
                effect["planning_diagnostics"] = deepcopy(attempts[-1].get("planning_diagnostics"))
            if targeted and target_poses is not None and arm in target_poses:
                distance, angle = pose_error(target_poses[arm], after[arm]["pose_xyz_xyzw"])
                effect["pose_error"] = {"translation_m": distance, "orientation_rad": angle}
                staged = strategy == STAGED_CONTACT_STRATEGY and GLOBAL_EXECUTOR in (executor or ())
                effect["plan_validation"] = (deepcopy(PLAN_VALIDATION)
                                             if (strategy == "global" or staged) and executor else None)
            if targeted and precontact is not None and arm in precontact:
                effect["precontact"] = deepcopy(dict(precontact[arm]))
            if targeted and arm in self.stall.progress:
                effect["motion_progress"] = deepcopy(self.stall.progress[arm])
            if targeted and gripper_targets is not None and arm in gripper_targets:
                effect["gripper_error"] = abs(after[arm]["gripper"] - gripper_targets[arm])
            if targeted and gripper_states and arm in gripper_states:
                effect["gripper_outcome"] = _gripper_outcome(state=gripper_states[arm], after=after[arm],
                                                             budget=self.budget, used=self.used,
                                                             robot=session.robots[arm])
            if not targeted:
                distance, angle = pose_error(self.before[arm]["pose_xyz_xyzw"], after[arm]["pose_xyz_xyzw"])
                effect["inactive_arm_drift"] = {"translation_m": distance, "orientation_rad": angle}
            arms_effect[arm] = {**effect, **deepcopy(dict((arm_extras or {}).get(arm, {})))}
        native_used = int(session.environment.take_action_cnt[0]) - self.native_before
        receipt = {
            "schema": SPARSE_RECEIPT_SCHEMA, "action_id": self.action["action_id"], "action": self.action["action"],
            "arguments": deepcopy(dict(self.action["arguments"])), "consumed": True,
            "state_changed": native_used > 0,
            "execution_status": status, "motion_strategy": strategy,
            "failure_stage": stage, "failure_code": code, "executor": executor,
            "target_arms": list(self.arms),
            "synchronization": {
                "mode": "hold" if hold else "single_arm" if len(self.arms) == 1 else "dual_arm_common_native_tick",
                "requested": len(self.arms) == 2, "executed": native_used > 0,
            },
            "arms": arms_effect, "before": self.before, "after": after,
            "ik_attempts": self.ik_attempts,
            "native_actions": native_used, "physics_steps": session.sim_steps - self.steps_before,
            "native_action_budget": self.budget, "steps_used": native_used, "sim_steps_total": session.sim_steps,
            "gripper_directives": deepcopy(session.gripper_directives),
            "gripper_floors": gripper_floors, "gripper_targets_effective": dict(gripper_targets or {}),
            "command_invariant": {
                "schema": "ora.command-invariant.v2",
                "controlled_dimension": "none" if hold else "eef_pose_target" if pose_action else "gripper",
                "held_dimension": ("eef_pose_target_and_gripper_directive" if hold else "gripper_directive"
                                   if pose_action else "eef_pose_target"),
                "verified": held_verified and (
                    session.gripper_directives == self.gripper_directives_before or not pose_action) and (
                    session.pose_targets == self.pose_targets_before or pose_action and not hold),
                "applied_targets": "per_physics_step_sdk_joint_position_buffer",
            },
            "command_targets_before": self.command_before, "command_targets_after": command_after,
            "pose_targets_before": self.pose_targets_before, "pose_targets_after": deepcopy(session.pose_targets),
            "round_index": session.round_index, "world_mutation_sequence": session.world_mutation_sequence,
            "official_evaluator_queried": True, "terminal_after_action": bool(session.is_terminal),
            "started_unix_s": self.started, "finished_unix_s": time.time(), **dict(extra or {}),
        }
        return session._persist_receipt(receipt, tracking=self.tracking, journal=self.journal)


def _remember_backtrack(session: Any, before: Mapping[str, Any], arms: Sequence[str]) -> None:
    for arm in arms:
        session.last_safe_joints[arm] = list(before[arm]["arm_joint_qpos"])
        session.last_safe_poses[arm] = list(before[arm]["pose_xyz_xyzw"])


def _run_global(
    execution: _SparseAction, segments: Sequence[tuple[str, Mapping[str, Sequence[float]]]],
    common: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Run global segments in order; a finished receipt when one stops, else ``None``."""
    for segment_id, poses in segments:
        joints: dict[str, list[float]] = {}
        planned: dict[str, dict[str, Any]] = {}
        for arm, pose in poses.items():
            # Both arms moving: each IK is screened against itself, the pair in lockstep below.
            solved = solve_ik(execution.session, arm, pose, static_other_arm=len(poses) == 1)
            planned[arm] = execution.record_ik(segment_id, arm, pose, solved)
            if solved["joint_target"] is None:
                execution.rejected_ik(segment_id, arm, pose, solved, planned[arm])
                return execution.finish(
                    status="planner_failed" if execution.used == 0 else "not_reached",
                    stage="planner", code=inter_arm.refusal_code(solved, segment_id),
                    executor=[GLOBAL_EXECUTOR] if execution.used else None, **common,
                )
            if solved["joint_delta_rad"] > execution.session.global_joint_jump_limit_rad:
                execution.rejected_ik(segment_id, arm, pose, solved, planned[arm])
                return execution.finish(
                    status="plan_rejected" if execution.used == 0 else "not_reached",
                    stage="plan_validation", code=f"ik_branch_jump:{segment_id}",
                    executor=[GLOBAL_EXECUTOR] if execution.used else None, **common,
                )
            joints[arm] = solved["joint_target"]
        extras = inter_arm.lockstep_refusal(execution, segment_id, poses, joints, planned)
        if extras is not None:
            return execution.finish(status="planner_failed" if execution.used == 0 else "not_reached",
                                    stage="planner", code=inter_arm.code(segment_id), arm_extras=extras,
                                    executor=[GLOBAL_EXECUTOR] if execution.used else None, **common)
        if execution.remaining() <= 0:
            return execution.finish(status="not_reached", stage="execution",
                                    code="native_action_budget_exhausted", executor=[GLOBAL_EXECUTOR], **common)
        reached = execution.run_joint_segment(segment_id, joints, poses, planned)
        if execution.session.is_terminal:
            return None
        if not reached:
            return execution.finish(
                status="not_reached", stage="execution", code=motion_stall.not_reached_code(
                    execution, motion_stall.BUDGET_CODE if execution.remaining() <= 0
                    else f"segment_not_reached:{segment_id}"),
                executor=[GLOBAL_EXECUTOR], **common,
            )
    return None


def _global_segments(
    execution: _SparseAction, segments: Sequence[tuple[str, Mapping[str, Sequence[float]]]],
    final_targets: Mapping[str, Sequence[float]], interactions: Mapping[str, str],
) -> dict[str, Any]:
    common = {"strategy": "global", "interactions": interactions, "target_poses": final_targets}
    stopped = _run_global(execution, segments, common)
    if stopped is not None:
        return stopped
    after = execution.session._proprioception()["end_effectors"]
    reached = all(pose_reached(pose, after[arm]["pose_xyz_xyzw"]) for arm, pose in final_targets.items())
    return execution.finish(
        status="reached" if reached else "not_reached",
        stage=None if reached else "endpoint_verification",
        code=None if reached else ("episode_terminal_during_execution" if execution.session.is_terminal
                                   else "endpoint_residual_exceeded"),
        executor=[GLOBAL_EXECUTOR], **common,
    )


def _local_step(
    execution: _SparseAction, targets: Mapping[str, Sequence[float]], interactions: Mapping[str, str],
    *, start: Mapping[str, Any] | None = None, strategy: str = "local",
    executor: Sequence[str] = (LOCAL_EXECUTOR,), precontact: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    # After a staged pre-contact write a refusal is not_reached (the others promise no write).
    common = {"strategy": strategy, "interactions": interactions, "target_poses": targets,
              **({"precontact": precontact} if precontact is not None else {})}
    written = list(executor[:-1]) or None
    measured = execution.before if start is None else start
    joints: dict[str, list[float]] = {}
    poses: dict[str, list[float]] = {}
    planned: dict[str, dict[str, Any]] = {}
    for arm, target in targets.items():
        current = [float(item) for item in measured[arm]["pose_xyz_xyzw"]]
        try:
            compiled = compile_servo_delta({
                "translation_m": [target[index] - current[index] for index in range(3)],
                "rotation_rotvec_rad": rotation_vector_between(target[3:], current[3:]),
                "frame": "world",
            }, current_pose_xyz_xyzw=current, controlled_point="link6")
        except ValueError:
            return execution.finish(status="plan_rejected" if execution.used == 0 else "not_reached",
                                    stage="route", code="servo_envelope_exceeded", executor=written, **common)
        poses[arm] = [float(item) for item in compiled["target_pose_xyz_xyzw"]]
        solved = solve_ik(execution.session, arm, poses[arm], static_other_arm=len(targets) == 1,
                          jump_limit_rad=execution.session.local_joint_jump_limit_rad)
        planned[arm] = {**execution.record_ik("local", arm, poses[arm], solved),
                        "servo_frame": compiled["servo_frame"]}
        if solved["joint_target"] is None:
            execution.rejected_ik("local", arm, poses[arm], solved, planned[arm])
            return execution.finish(status="planner_failed" if execution.used == 0 else "not_reached", stage="planner",
                                    code=inter_arm.refusal_code(solved, "local"), executor=written, **common)
        if solved["joint_delta_rad"] > execution.session.local_joint_jump_limit_rad:
            execution.rejected_ik("local", arm, poses[arm], solved, planned[arm])
            # One bounded Cartesian step that needs a large joint excursion is an
            # aliased IK branch, not a local move; it must not write the plant.
            return execution.finish(status="plan_rejected" if execution.used == 0 else "not_reached",
                                    stage="plan_validation", code="local_ik_branch_jump", executor=written,
                                    **common)
        joints[arm] = solved["joint_target"]
    extras = inter_arm.lockstep_refusal(execution, "local", poses, joints, planned)
    if extras is not None:
        return execution.finish(status="planner_failed" if execution.used == 0 else "not_reached", stage="planner",
                                code=inter_arm.code("local"), executor=written, arm_extras=extras, **common)
    reached = execution.run_joint_segment("local", joints, poses, planned)
    return execution.finish(
        status="reached" if reached else "not_reached",
        stage=None if reached else "execution",
        code=None if reached else motion_stall.not_reached_code(execution, motion_stall.BUDGET_CODE),
        executor=list(executor), **common,
    )


def _execute_pose_targets(execution: _SparseAction, targets: Mapping[str, Any], interactions: Mapping[str, str]) -> dict[str, Any]:
    session = execution.session
    kind = execution.action["action"]
    unique = set(interactions.values())
    poses = {arm: _pose_list(targets[arm]) for arm in execution.arms}
    if len(unique) != 1:
        return execution.finish(status="plan_rejected", strategy=None, stage="route", code="mixed_interactions",
                                executor=None, interactions=interactions, target_poses=poses)
    for arm, pose in poses.items():
        axis = workspace_violation(pose, session.config.get("workspace_bounds_m"))
        if axis is not None:
            return execution.finish(status="plan_rejected", strategy=None, stage="plan_validation",
                                    code=f"workspace_target_{axis}", executor=None,
                                    interactions=interactions, target_poses=poses)
    interaction = unique.pop()
    local_eligible = all(
        within_local_envelope(*pose_error(pose, execution.before[arm]["pose_xyz_xyzw"]))
        for arm, pose in poses.items()
    )
    if kind == "move_free_space" or interaction == "free_space":
        strategy: str | None = "global"
    elif interaction == "precision":
        strategy = "local" if local_eligible else "global"
    elif local_eligible:
        strategy = "local"
    else:
        strategy = STAGED_CONTACT_STRATEGY if session.contact_approach_route == STAGED_CONTACT_STRATEGY else None
    if strategy is None:
        return execution.finish(status="plan_rejected", strategy=None, stage="route", code=ROUTE_REFUSAL_CODE,
                                executor=None, interactions=interactions, target_poses=poses)
    _remember_backtrack(session, execution.before, execution.arms)
    execution.begin([execution.arm_key(arm) for arm in execution.arms])
    if strategy == "local":
        return _local_step(execution, poses, interactions)
    if strategy == STAGED_CONTACT_STRATEGY:
        from .staged_contact import staged_contact_approach
        return staged_contact_approach(execution, poses, interactions)
    segments = (free_space_segments(session, execution.before, poses) if kind == "move_free_space"
                else [("move", poses)])
    return _global_segments(execution, segments, poses, interactions)


def _gripper_outcome(*, state: str, after: Mapping[str, Any], budget: int, used: int, robot: Any) -> dict[str, Any]:
    """One arm's jaw outcome: request, target, two-finger reading, SDK-limited drive and its hold flag."""
    feedback = after["gripper_feedback"]
    fingers = [float(value) for value in after["gripper_finger_qpos"]]
    return gripper_outcome(
        rule=ARX_GRIPPER_STALL, state=state,
        target=feedback["requested_opening_normalized"],
        measured=feedback["measured_opening_normalized"],
        ticks_used=used, ticks_budget=budget,
        extra={
            "effective_target_opening_normalized": feedback["effective_target_opening_normalized"],
            "sdk_limited_drive_target_normalized": feedback["drive_target_opening_normalized"],
            "sdk_drive_limit_normalized": float(
                ARX_X5_CONTRACT["gripper_drive_limit"]["maximum_normalized_offset"]),
            "measured_finger_displacements_m": fingers,
            "finger_lateral_offset_m": feedback["finger_lateral_offset_m"],
            "primary_finger_limit_holds_close": primary_limit_holds_close(
                robot, fingers, requested=feedback["requested_opening_normalized"],
                drive=feedback["drive_target_opening_normalized"],
                effective=feedback["effective_target_opening_normalized"]),
            "tick_unit": ARX_GRIPPER_STALL.tick_unit,
            "tick_period_s": ARX_GRIPPER_STALL.tick_period_s,
        },
    )


def gripper_failure_code(states: Mapping[str, str], *, terminal: bool) -> str | None:
    """One public code for a jaw command; ``None`` only when every jaw arrived."""
    if all(state == "reached_target" for state in states.values()):
        return None
    if terminal:
        return "episode_terminal_during_execution"
    if all(state in {"reached_target", "stalled_before_target"} for state in states.values()):
        return GRIPPER_STALL_REASON
    return "gripper_not_reached"


def _execute_gripper(execution: _SparseAction, targets: Mapping[str, str]) -> dict[str, Any]:
    session = execution.session
    directives = {arm: 1.0 if targets[arm] == "open" else 0.0 for arm in execution.arms}
    execution.begin([execution.gripper_key(arm) for arm in execution.arms])
    for arm, value in directives.items():
        session.gripper_directives[arm] = value
    effective = {arm: max(value, gripper_closed_floor(session, arm)) for arm, value in directives.items()}
    states = execution.run_gripper(directives, effective)
    code = gripper_failure_code(states, terminal=session.is_terminal)
    return execution.finish(
        status="reached" if code is None else "not_reached", strategy=None,
        stage=None if code is None else "execution", code=code,
        executor=[GRIPPER_EXECUTOR], gripper_targets=effective, gripper_states=states,
    )


def _execute_plan_reference(session: Any, action: Mapping[str, Any]) -> dict[str, Any]:
    plan_ref = str(action["arguments"]["plan_ref"])
    stored = session.plan_references.get(plan_ref)
    if not isinstance(stored, Mapping):
        raise RobodojoError("plan_ref is unknown or tampered")
    recorded = int(stored["world_mutation_sequence"])
    same_snapshot = (
        recorded == session.world_mutation_sequence
        and stored["observation_id"] == action["observation_id"]
        and stored["observation_sha256"] == action["bound_observation_sha256"]
    )
    intervening = session.world_mutation_actions[recorded:]
    gripper_only = bool(intervening) and all(kind == "set_gripper" for kind in intervening)
    if not (same_snapshot or gripper_only):
        raise RobodojoError("plan_ref is stale for the current world state")
    arm = str(stored["arm"])
    current = session._proprioception()["end_effectors"][arm]["arm_joint_qpos"]
    if max(abs(float(a) - float(b)) for a, b in zip(current, stored["measured_start_joints"])) > PLAN_REFERENCE_JOINT_TOLERANCE_RAD:
        raise RobodojoError("plan_ref start joints no longer match")
    execution = _SparseAction(session, action, (arm,))
    _remember_backtrack(session, execution.before, (arm,))
    execution.begin([execution.arm_key(arm)])
    target = [float(item) for item in stored["target_pose_xyzw"]]
    reached = execution.run_joint_segment(
        "plan_ref", {arm: stored["joint_target"]}, {arm: target},
        {arm: {"ik_status": "Success", "joint_delta_rad": stored["joint_delta_rad"], "source": "motion_probe_plan_ref"}},
    )
    return execution.finish(
        status="reached" if reached else "not_reached", strategy="global",
        stage=None if reached else "execution",
        code=None if reached else motion_stall.not_reached_code(execution, motion_stall.BUDGET_CODE),
        executor=[GLOBAL_EXECUTOR], interactions={arm: "free_space"}, target_poses={arm: target},
        plan_refs={arm: plan_ref},
    )


def execute_sparse_motion_v3(session: Any, action: Mapping[str, Any]) -> dict[str, Any]:
    """Dispatch one validated sparse v3 action to its RoboDojo executor."""
    kind = action["action"]
    arguments = action["arguments"]
    if kind == "move_pose" and set(arguments) == {"plan_ref"}:
        return _execute_plan_reference(session, action)
    if kind == "wait":
        from .wait import execute_wait
        return execute_wait(_SparseAction(session, action, ()))
    targets = arguments["targets"]
    arms = tuple(arm for arm in ARMS if arm in targets)
    execution = _SparseAction(session, action, arms)
    if kind == "set_gripper":
        return _execute_gripper(execution, targets)
    if kind == "recover_motion":
        from .recovery import execute_recovery
        return execute_recovery(execution, targets)
    return _execute_pose_targets(execution, targets, arguments["interactions"])


#: SDK candidate rejections (``ik.select_candidate_set``) in the public
#: constraint names the probe summary counts.
_IK_REJECTION_NAMES = {
    "nonfinite_joint_target": "non_finite_solution",
    "sdk_not_feasible_or_converged": "solver_unsuccessful",
    "joint_limits": "joint_limit",
    "fk_residual_exceeded": "fk_residual_exceeded",
    "joint_jump": "joint_jump",
    "pitch_joint_wrap": "pitch_joint_wrap",
}


def _planner_summary(solved: Mapping[str, Any], *, ik_feedback: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The bounded planner record of one probe variant.

    The v1 shape is what the frozen ``robodojo_unified_manip_v1`` probe result
    schema admits, so it stays exactly as released. ``ik_feedback`` is supplied
    only for a v2 feedback request and keeps the SDK solve's own candidate
    evidence -- returned/valid/continuous counts, per-filter rejections and the
    retained diagnostic file -- that the v1 record reduces to ``Fail``.
    """
    reachable = solved["joint_target"] is not None
    record = {
        "called": True, "status": solved["ik_status"], "stage": "SUCCESS" if reachable else "FAILED",
        "valid_query": True, "attempts": None, "trajopt_attempts": None, "used_graph": None,
        "goalset_index": None, "path_steps": 1 if reachable else 0, "interpolation_dt_s": None,
        "errors": {"position": None, "rotation": None, "cspace": None},
        "timings_s": {"solve": None, "ik": None, "graph": None, "trajopt": None, "finetune": None, "total": None},
        "error": None, "planner": "sdk_ik_joint_interpolation", "parity": "non_parity_motion",
    }
    if ik_feedback is None:
        return record
    rejections = Counter(
        _IK_REJECTION_NAMES.get(str(row.get("rejection")), str(row.get("rejection")))
        for row in solved.get("candidates", []) if isinstance(row, Mapping) and row.get("rejection")
    )
    record["constraint_rejection_counts"] = {key: int(rejections.get(key, 0)) for key in IK_CONSTRAINT_KEYS}
    diagnostics = public_planning_diagnostics(solved.get("planning_diagnostics"))
    if diagnostics:
        record["planning_diagnostics"] = diagnostics
    record["ik_validation_artifact"] = dict(ik_feedback)
    return record


def probe_motion(session: Any, request: Mapping[str, Any]) -> dict[str, Any]:
    """Answer reachability by IK from the current snapshot without writing the world."""
    fields = {"schema", "observation_id", "observation_sha256", "queries", "variant_policy"}
    # v1 is the released request; v2 differs only by naming the planner
    # feedback it asks for, so frozen surfaces keep receiving the v1 record.
    if not isinstance(request, Mapping) or not (
        (request.get("schema") == PROBE_REQUEST_SCHEMA and set(request) == fields)
        or (request.get("schema") == PROBE_FEEDBACK_REQUEST_SCHEMA
            and set(request) == fields | {"planner_feedback"}
            and request.get("planner_feedback") == PROBE_IK_CANDIDATE_FEEDBACK)
    ):
        raise RobodojoError("motion probe request fields drifted")
    ik_feedback = request.get("schema") == PROBE_FEEDBACK_REQUEST_SCHEMA
    observation_id, observation_sha256 = request.get("observation_id"), request.get("observation_sha256")
    queries, variant_policy = request.get("queries"), request.get("variant_policy")
    if (not isinstance(observation_id, str) or not observation_id
            or not isinstance(observation_sha256, str) or not observation_sha256
            or not isinstance(queries, list) or not 1 <= len(queries) <= 12
            or variant_policy not in {"exact", "parallel_jaw_equivalents"}):
        raise RobodojoError("motion probe request is invalid")
    clock_before = session._clock()
    measured = session._proprioception()["end_effectors"]
    results: list[dict[str, Any]] = []
    query_ids: set[str] = set()
    for query in queries:
        if not isinstance(query, Mapping) or set(query) != {"query_id", "arm", "pose"}:
            raise RobodojoError("motion probe query fields drifted")
        query_id, arm, pose = query.get("query_id"), query.get("arm"), query.get("pose")
        if (not isinstance(query_id, str) or not query_id or query_id in query_ids or arm not in ARMS
                or not isinstance(pose, Mapping) or set(pose) != {"position_m", "orientation_xyzw"}):
            raise RobodojoError("motion probe query is invalid")
        query_ids.add(query_id)
        position = [float(value) for value in pose["position_m"]]
        orientation = [float(value) for value in pose["orientation_xyzw"]]
        if (len(position) != 3 or len(orientation) != 4
                or not all(math.isfinite(value) for value in [*position, *orientation])
                or abs(float(np.linalg.norm(orientation)) - 1.0) > 0.02):
            raise RobodojoError("motion probe pose is invalid")
        if workspace_violation(position, session.config.get("workspace_bounds_m")) is not None:
            raise RobodojoError("motion probe target is outside the action workspace")
        variants = [("variant-00-exact-+0deg", orientation)]
        if variant_policy == "parallel_jaw_equivalents":
            variants.append(("variant-01-jaw-flip-180deg", rotate_about_local_axis(orientation, JAW_SYMMETRY_AXIS, math.pi)))
        rows: list[dict[str, Any]] = []
        admitted: list[dict[str, Any]] = []
        for index, (variant_id, variant_orientation) in enumerate(variants):
            variant_pose = position + list(variant_orientation)
            solved = solve_ik(session, arm, variant_pose)
            # Keep the existing closed public probe schema. Persist full
            # failed and successful IK evidence in the canonical workspace.
            diagnostic = {"schema": "ora.robodojo-probe-ik.v1",
                          "observation_id": observation_id, "observation_sha256": observation_sha256,
                          "world_mutation_sequence": session.world_mutation_sequence,
                          "round_index": session.round_index, "query_id": query_id,
                          "variant_id": variant_id, "arm": arm,
                          "target_pose_xyzw": variant_pose, "ik": solved}
            diagnostic_hash = hashlib.sha256(json.dumps(
                diagnostic, sort_keys=True, separators=(",", ":"), allow_nan=False,
            ).encode()).hexdigest()
            diagnostic_path = session.output_root / "motion-probes" / "ik-diagnostics" / f"{diagnostic_hash}.json"
            write_json(diagnostic_path, diagnostic)
            summary = _planner_summary(solved, ik_feedback={
                "path": f"motion-probes/ik-diagnostics/{diagnostic_hash}.json",
                "schema": diagnostic["schema"],
                "sha256": hashlib.sha256(diagnostic_path.read_bytes()).hexdigest(),
            } if ik_feedback else None)
            if solved["joint_target"] is None:
                rows.append({"variant_id": variant_id, "reachable": False, "planner": summary,
                             "failure": {"code": solved["ik_status"], "message": "SDK IK admitted no joint target"}})
                continue
            if solved["joint_delta_rad"] > session.global_joint_jump_limit_rad:
                rows.append({"variant_id": variant_id, "reachable": False, "planner": summary,
                             "failure": {"code": "ik_branch_jump",
                                         "message": f"IK branch is {solved['joint_delta_rad']:.3f} rad away"}})
                continue
            reference_body = {
                "schema": PLAN_REFERENCE_SCHEMA, "observation_id": observation_id,
                "observation_sha256": observation_sha256,
                "world_mutation_sequence": session.world_mutation_sequence, "round_index": session.round_index,
                "arm": arm, "query_id": query_id, "variant_id": variant_id,
                "target_pose_xyzw": variant_pose, "measured_start_joints": list(measured[arm]["arm_joint_qpos"]),
                "joint_target": solved["joint_target"], "joint_delta_rad": solved["joint_delta_rad"],
            }
            plan_ref = "plan-" + hashlib.sha256(json.dumps(
                reference_body, sort_keys=True, separators=(",", ":"),
            ).encode()).hexdigest()[:32]
            artifact = write_motion_plan_artifact(
                session.output_root / "motion-probes",
                transaction_id=f"probe-{session.motion_probe_sequence:04d}-{arm}-{index:03d}",
                backend_id="robodojo", embodiment_id="arx_x5", binding=reference_body,
                validation=PLAN_VALIDATION,
                arrays={f"planned_{arm}_joint__position": [solved["joint_target"]]},
            )
            # The artifact writer records paths relative to its own root;
            # evaluation resolves against the run native root.
            artifact = {**artifact, "path": f"motion-probes/{artifact['path']}"}
            session.plan_references[plan_ref] = {**reference_body, "plan_ref": plan_ref, "artifact": artifact}
            entry = {
                "variant_id": variant_id, "reachable": True, "target_pose_xyzw": variant_pose,
                "planner": summary, "endpoint_error": {"translation_m": None, "orientation_rad": None},
                "trajectory_steps": 1, "trajectory_duration_s": 0.0,
                "joint_delta_rad": solved["joint_delta_rad"], "plan_ref": plan_ref,
            }
            rows.append(entry)
            admitted.append(entry)
            session.motion_probe_sequence += 1
        admitted.sort(key=lambda row: row["joint_delta_rad"])
        selected = admitted[0] if admitted else None
        results.append({
            "query_id": query_id, "arm": arm, "reachable": bool(admitted),
            "selected_variant_id": selected["variant_id"] if selected else None,
            "plan_ref": selected["plan_ref"] if selected else None,
            "selected_target_pose_xyzw": selected["target_pose_xyzw"] if selected else None,
            "variants": rows,
        })
    if session._clock() != clock_before:
        session._fault("motion_probe_advanced_physics")
        raise RobodojoError("read-only motion probe advanced physics")
    return {
        "schema": PROBE_RESULT_SCHEMA, "observation_id": observation_id,
        "world_mutation_sequence": session.world_mutation_sequence, "variant_policy": variant_policy,
        "robot_decisions_consumed": 0, "world_state_changed": False, "results": results,
    }


__all__ = [
    "GLOBAL_EXECUTOR", "GLOBAL_JOINT_JUMP_LIMIT_RAD", "GLOBAL_PLANNING_NON_PARITY", "GRIPPER_EXECUTOR",
    "LOCAL_EXECUTOR",
    "PLAN_VALIDATION", "RECOVERY_EXECUTOR", "ROUTE_REFUSAL_CODE", "SPARSE_RECEIPT_SCHEMA",
    "STAGED_CONTACT_STRATEGY",
    "execute_sparse_motion_v3", "free_space_segments", "gripper_closed_floor", "gripper_failure_code",
    "probe_motion",
    "rotate_about_local_axis",
    "rotation_vector_between", "solve_ik", "within_local_envelope",
]
