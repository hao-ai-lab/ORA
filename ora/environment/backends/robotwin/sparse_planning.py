"""Sparse-motion planning, validation and blocked-motion recovery.

A mixin of ``session.RoboTwinLiveSession``, moved verbatim from
``session.py`` on 2026-10-01; it holds methods only, no dataclass fields.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping

from ora.environment.backends.robotwin.planning import (
    RoboTwinPlanRejected,
    ValidatedRoboTwinPlan,
    select_topk_validated_plan_set,
    validate_inter_arm_clearance,
    validate_single_arm_plan,
)
from ora.environment.controllers.profile_capabilities import (
    CONTACT_RECOVERY_SINCE,
    profile_has,
)

from .common import (
    RoboTwinLiveError,
    _SparseMotionState,
    _validate_workspace_target,
    _wxyz_to_xyzw,
    _xyzw_to_wxyz,
)


class RoboTwinSparsePlanningMixin:
    """Sparse-motion planning, validation and blocked-motion recovery."""

    def _plan_and_validate(self, state: _SparseMotionState) -> bool:
        """Compile targets, plan and validate; return planner admission."""

        for arm in state.target_arms:
            if state.action["action"] == "recover_motion":
                strategy = state.targets[arm]
                state.recovery_joint_targets[arm] = list(
                    self.last_safe_joints.get(
                        arm, state.measured_joint_snapshot[arm],
                    )
                    if strategy == "backtrack_last_safe"
                    else self.home_joints[arm]
                )
                target_xyzw = self._pinocchio_public_pose_xyzw(
                    state.environment, arm, state.recovery_joint_targets[arm],
                )
                public_wxyz = _xyzw_to_wxyz(
                    target_xyzw[:3], target_xyzw[3:],
                )
                planner_wxyz = (
                    {"kind": "joint", "joints": state.recovery_joint_targets[arm]}
                    if state.motion_adapter is not None
                    else self._public_to_planner_wxyz(arm, public_wxyz)
                )
            else:
                target_pose = state.targets[arm]
                _validate_workspace_target(
                    list(target_pose["position_m"]), self.workspace_bounds_m,
                )
                public_wxyz = _xyzw_to_wxyz(
                    target_pose["position_m"], target_pose["orientation_xyzw"],
                )
                state.public_pose_targets[arm] = public_wxyz
                planning_public_wxyz = public_wxyz
                if state.global_precontact_then_local_requested:
                    raw_precontact = (
                        (state.motion_strategy or {}).get(
                            "precontact_public_tcp_poses", {}
                        ).get(arm)
                    )
                    if (
                        not isinstance(raw_precontact, list)
                        or len(raw_precontact) != 7
                    ):
                        raise RoboTwinLiveError(
                            "staged contact route omitted its pre-contact pose"
                        )
                    _validate_workspace_target(
                        [float(value) for value in raw_precontact[:3]],
                        self.workspace_bounds_m,
                    )
                    planning_public_wxyz = _xyzw_to_wxyz(
                        raw_precontact[:3], raw_precontact[3:],
                    )
                planner_wxyz = (
                    state.motion_adapter.compile_public_target(
                        arm, [
                            *planning_public_wxyz[:3],
                            *planning_public_wxyz[4:],
                            planning_public_wxyz[3],
                        ],
                    )
                    if state.motion_adapter is not None
                    else self._public_to_planner_wxyz(
                        arm, planning_public_wxyz,
                    )
                )
            if state.action["action"] == "recover_motion":
                state.public_pose_targets[arm] = public_wxyz
                planning_public_wxyz = public_wxyz
            state.planner_public_pose_targets[arm] = planning_public_wxyz
            state.planner_pose_targets[arm] = planner_wxyz

        if not self._retreat_before_blocked_recovery(state):
            return False
        # Both planners read the same unchanged phase-start state. A completed
        # blocked-recovery retreat refreshes this snapshot before either plan.
        for arm in state.target_arms:
            if state.motion_adapter is not None:
                state.planner_results[arm] = state.motion_adapter.plan_global(
                    arm,
                    state.planner_pose_targets[arm],
                    state.motion_action_start,
                )
            elif arm in state.recovery_joint_targets:
                route = str(self._arm_profile(arm)["planner_joint_route"])
                owner, method = route.split(".", 1)
                planner = getattr(getattr(state.environment, owner), method)
                state.planner_results[arm] = planner(state.recovery_joint_targets[arm])
            else:
                planner = getattr(state.environment.robot, f"{arm}_plan_path")
                state.planner_results[arm] = planner(state.planner_pose_targets[arm])
            self._record_ik_candidate_evidence(state.planner_results[arm], arm=arm,
                identity={"kind": "action", "action_id": state.action["action_id"],
                          "action_sequence": self.action_sequence})
        planners_succeeded = all(
            isinstance(state.planner_results[arm], Mapping)
            and state.planner_results[arm].get("status") == "Success"
            for arm in state.target_arms
        )
        if planners_succeeded:
            validation_profile = self.control_profile["profile"]["plan_validation"]
            validators = {}
            for arm in state.target_arms:
                target_xyzw = _wxyz_to_xyzw(state.planner_public_pose_targets[arm])
                def validate_candidate(
                    candidate: Mapping[str, Any], *, selected_arm: str = arm,
                    selected_target: list[float] = target_xyzw,
                ) -> ValidatedRoboTwinPlan:
                    return validate_single_arm_plan(
                        candidate,
                        expected_joint_names=self._arm_profile(selected_arm)[
                            "planner_joint_names"
                        ],
                        measured_start_joints=state.measured_joint_snapshot[selected_arm],
                        measured_start_pose_xyzw=state.planning_start_public_poses.get(
                            selected_arm, state.before[selected_arm]["pose_xyz_xyzw"],
                        ),
                        target_pose_xyzw=selected_target,
                        joint_limits=(
                            state.motion_adapter.joint_limits(selected_arm)
                            if state.motion_adapter is not None
                            else self._arm_joint_limits(state.environment, selected_arm)
                        ),
                        fk_public_pose=(
                            (
                                lambda joints: state.motion_adapter.fk_public_tcp_pose(
                                    selected_arm, joints,
                                )
                            )
                            if state.motion_adapter is not None
                            else (
                                lambda joints: self._pinocchio_public_pose_xyzw(
                                    state.environment, selected_arm, joints,
                                )
                            )
                        ),
                        workspace_bounds_m=(
                            state.motion_adapter.workspace_bounds_m()
                            if state.motion_adapter is not None
                            else self.workspace_bounds_m
                        ),
                        profile=validation_profile,
                        fk_link6_pose=(
                            (
                                lambda joints: state.motion_adapter.fk_control_pose(
                                    selected_arm, joints,
                                )
                            )
                            if state.motion_adapter is not None
                            else (
                                lambda joints: self._pinocchio_link6_pose_xyzw(
                                    state.environment, selected_arm, joints,
                                )
                            )
                        ),
                    )

                validators[arm] = validate_candidate
            try:
                selected, validated, admission = select_topk_validated_plan_set(
                    {arm: state.planner_results[arm] for arm in state.target_arms},
                    validators=validators,
                    set_validator=lambda plans: self._validate_planned_motion_set(state, plans),
                )
                state.planner_results.update(selected)
                state.validated_plans.update(validated)
                state.collision_validation = admission["collision_validation"]
                state.payload_validation = admission["payload_validation"]
            except RoboTwinPlanRejected as exc:
                for arm in state.target_arms:
                    state.plan_rejections[arm] = {
                        "code": exc.code, "message": str(exc), "details": exc.details,
                    }
            planners_succeeded = not state.plan_rejections and all(
                arm in state.validated_plans for arm in state.target_arms
            )
        return planners_succeeded

    @staticmethod
    def _blocked_recovery_source_receipt(record: Mapping[str, Any] | None) -> Mapping[str, Any]:
        """The canonical Move receipt one recovery path starts from."""
        previous = (record or {}).get("receipt", record or {})
        if not isinstance(previous, Mapping):
            return {}
        return previous

    def _retreat_before_blocked_recovery(self, state: _SparseMotionState) -> bool:
        """Use a recorded approach segment, then plan from the measured retreat.

        The opposite commanded approach is a bounded retry direction, not a
        collision normal or a guarantee of release. Missing direction evidence
        cannot silently fall back to an invented world axis.
        """
        if (state.action["action"] != "recover_motion"
                or not profile_has(self.control_profile, since_version=CONTACT_RECOVERY_SINCE)):
            return True
        previous = self._blocked_recovery_source_receipt(self.last_receipt)
        if previous.get("failure_code") != "blocked":
            return True
        blocked = previous.get("blocked_motion") or {}
        prior_abort = (previous.get("controller_tracking") or {}).get("abort") or {}
        blocked_arms = set(prior_abort.get("arms") or previous.get("target_arms") or state.target_arms)
        selected = blocked_arms & set(state.target_arms)
        if not selected:
            return True
        distance = float(self.control_profile["profile"]["blocked_motion_recovery"]["retreat_distance_m"])
        current = {arm: _wxyz_to_xyzw(self._public_pose_wxyz(state.environment, arm))
                   for arm in state.target_arms}
        retreat_targets = copy.deepcopy(current)
        directions = {}
        for arm in sorted(selected):
            record = blocked.get("arms", {}).get(arm, {})
            direction = record.get("approach_direction_world")
            valid = (isinstance(direction, list) and len(direction) == 3
                     and all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
                             for x in direction))
            norm = math.sqrt(sum(float(x) ** 2 for x in direction)) if valid else 0.0
            if not valid or abs(norm - 1.0) > 1e-5:
                for target_arm in state.target_arms:
                    state.plan_rejections[target_arm] = {
                        "code": "blocked_recovery_direction_unavailable",
                        "message": "automatic retreat needs a nonzero recorded approach; use a fresh observation and an explicit bounded local move",
                        "details": {"arm": arm, "controller_writes": 0},
                    }
                return False
            retreat_targets[arm][:3] = [current[arm][i] - distance * float(direction[i]) for i in range(3)]
            try:
                _validate_workspace_target(retreat_targets[arm][:3], self.workspace_bounds_m)
            except RoboTwinLiveError:
                for target_arm in state.target_arms:
                    state.plan_rejections[target_arm] = {
                        "code": "blocked_recovery_retreat_outside_workspace",
                        "message": "the recorded opposite approach leaves the workspace; choose a fresh observation and an explicit bounded local move",
                        "details": {"arm": arm, "controller_writes": 0},
                    }
                return False
            directions[arm] = copy.deepcopy(record)
        state.blocked_recovery = {
            "schema": "ora.robotwin-blocked-recovery.v1",
            "source_action_id": previous.get("action_id"),
            "retreat_distance_m": distance, "retreated_arms": sorted(selected),
            "held_selected_arms": sorted(set(state.target_arms) - selected),
            "approach_evidence": directions, "retreat_targets_xyzw": retreat_targets,
            "direction_claim": "opposite_previous_commanded_approach_not_contact_normal",
            "global_planning_after_retreat": False,
        }
        state.execution_attempted = True
        tracking, arrays, sequence = self._execute_local_cartesian_action(
            environment=state.environment, target_pose_xyzw=retreat_targets,
            action=state.action,
        )
        state.executed = bool(tracking.get("sampled_steps", 0))
        state.blocked_recovery["retreat_completed"] = bool(tracking["completed"])
        state.blocked_recovery["retreat_tracking"] = copy.deepcopy(tracking)
        state.controller_tracking, state.controller_tracking_arrays, state.control_sequence_record = tracking, arrays, sequence
        if not tracking["completed"] or tracking.get("termination"):
            if not tracking.get("termination"):
                state.tracking_not_reached_reason = str((tracking.get("abort") or {}).get("code", "blocked_recovery_retreat_not_reached"))
                if not state.executed:
                    for arm in state.target_arms:
                        state.plan_rejections[arm] = {
                            "code": state.tracking_not_reached_reason,
                            "message": "blocked-recovery retreat was refused; global recovery was not attempted",
                            "details": {"phase": "blocked_recovery_retreat", "controller_writes": 0},
                        }
            return False
        state.recovery_prefix = (tracking, arrays, sequence)
        state.controller_tracking, state.controller_tracking_arrays, state.control_sequence_record = None, {}, None
        state.measured_joint_snapshot = {arm: self._measured_arm_joints(state.environment, arm)
                                         for arm in ("left", "right")}
        state.planning_start_public_poses = {arm: _wxyz_to_xyzw(self._public_pose_wxyz(state.environment, arm))
                                              for arm in ("left", "right")}
        state.motion_action_start = state.motion_adapter.capture_action_start()
        state.blocked_recovery["global_planning_after_retreat"] = True
        return True

    @staticmethod
    def _combine_blocked_recovery_prefix(state: _SparseMotionState) -> None:
        """Join both executed phases without erasing the retreat on later refusal."""
        if state.recovery_prefix is None:
            return
        import numpy as np
        prefix, prefix_arrays, prefix_sequence = state.recovery_prefix
        suffix = state.controller_tracking or {
            "completed": False, "sampled_steps": 0,
            "abort": {"code": "recovery_global_planning_failed"},
        }
        if state.controller_tracking is None and state.tracking_not_reached_reason is None:
            state.tracking_not_reached_reason = "recovery_global_planning_failed"
        suffix_arrays = state.controller_tracking_arrays

        def concatenate(first: Any, second: Any) -> np.ndarray:
            a, b = np.asarray(first), np.asarray(second)
            return np.concatenate([a, b]) if a.size and b.size else (a.copy() if a.size else b.copy())

        arrays = {key: concatenate(prefix_arrays.get(key, []), suffix_arrays.get(key, []))
                  for key in prefix_arrays.keys() | suffix_arrays.keys()}
        arrays.update({"recovery_retreat__" + key: value for key, value in prefix_arrays.items()})
        arrays.update({"recovery_global__" + key: value for key, value in suffix_arrays.items()})
        report = copy.deepcopy(suffix)
        for key in ("sampled_steps", "controller_write_attempts", "physics_step_attempts"):
            report[key] = int(prefix.get(key, 0)) + int(suffix.get(key, 0))
        report["execution_started"] = bool(prefix.get("execution_started") or suffix.get("execution_started"))
        report["execution_phases"] = [
            {"phase": "blocked_recovery_retreat", "tracking": prefix},
            {"phase": "global_recovery", "tracking": suffix},
        ]
        report["arms"] = copy.deepcopy(suffix.get("arms", prefix.get("arms", {})))
        for arm, values in report["arms"].items():
            for key in ("maximum_joint_tracking_error_rad", "maximum_tcp_step_m"):
                values[key] = max(float(prefix.get("arms", {}).get(arm, {}).get(key, 0)), float(values.get(key, 0)))
        sequence = {}
        for arm in ("left", "right"):
            key = f"{arm}_arm"
            a, b = prefix_sequence.get(key), (state.control_sequence_record or {}).get(key)
            sequence[key] = {
                "status": "IssuedBlockedRecovery",
                "position": concatenate((a or {}).get("position", []), (b or {}).get("position", [])),
                "velocity": concatenate((a or {}).get("velocity", []), (b or {}).get("velocity", [])),
            } if a is not None or b is not None else None
            sequence[f"{arm}_gripper"] = None
        state.executed = bool(state.executed or prefix.get("sampled_steps", 0))
        state.controller_tracking, state.controller_tracking_arrays, state.control_sequence_record = report, arrays, sequence
        state.recovery_prefix = None

    @staticmethod
    def _blocked_motion_evidence(state: _SparseMotionState) -> dict[str, Any] | None:
        tracking = state.controller_tracking or {}
        abort = tracking.get("abort") or {}
        if abort.get("code") != "blocked":
            return None
        stage = int(abort.get("stage_index", 0))
        stages = int(tracking.get("planned_stages", 1))
        arms = {}
        for arm in abort.get("arms", state.target_arms):
            plan = state.validated_plans.get(arm)
            vector = None
            source = None
            if plan is not None:
                points = plan.planned_tcp_pose_xyzw
                end = min(len(points) - 1, int(round(stage * (len(points) - 1) / max(1, stages - 1))))
                for index in range(end, 0, -1):
                    delta = [float(points[index][i] - points[index - 1][i]) for i in range(3)]
                    if math.sqrt(sum(x * x for x in delta)) > 1e-8:
                        vector, source = delta, {"kind": "validated_tcp_segment", "start_index": index - 1, "end_index": index}
                        break
            elif arm in state.public_pose_targets and arm in state.before:
                target = (state.blocked_recovery or {}).get("retreat_targets_xyzw", {}).get(
                    arm, state.public_pose_targets[arm],
                ) if not (state.blocked_recovery or {}).get("global_planning_after_retreat") else state.public_pose_targets[arm]
                delta = [float(target[i] - state.before[arm]["pose_xyz_xyzw"][i]) for i in range(3)]
                if math.sqrt(sum(x * x for x in delta)) > 1e-8:
                    vector, source = delta, {"kind": "commanded_local_tcp_delta"}
            norm = math.sqrt(sum(x * x for x in vector)) if vector is not None else 0.0
            arms[arm] = {
                "approach_direction_world": [x / norm for x in vector] if norm else None,
                "source": source, "direction_available": bool(norm),
            }
        return {"schema": "ora.robotwin-blocked-motion.v1", "arms": arms,
                "blocked_at_stage": stage, "planned_stages": stages,
                "frozen_steps": abort.get("frozen_steps"),
                "progress": abort.get("progress"), "stall_shape": abort.get("stall_shape")}

    def _validate_planned_motion_set(
        self, state: _SparseMotionState,
        plans: Mapping[str, ValidatedRoboTwinPlan],
    ) -> dict[str, Any]:
        """Complete collision and optional payload admission for one plan set."""

        collision = (
            state.motion_adapter.validate_collision(plans, state.motion_action_start)
            if state.motion_adapter is not None
            else validate_inter_arm_clearance(
                plans, measured_joints=state.measured_joint_snapshot,
                fk_chain_points=lambda arm, joints: self._pinocchio_chain_points(
                    state.environment, arm, joints,
                ),
                link_names={arm: spec["pinocchio_chain_links"] for arm, spec in self.control_profile["profile"]["arms"].items()},
                profile=self.control_profile["profile"]["dual_arm_collision_check"],
            )
        )
        return {"collision_validation": collision, "payload_validation": None}
