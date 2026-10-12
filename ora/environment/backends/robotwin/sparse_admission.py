"""Sparse-motion entry: admission, plan references and strategy routing.

A mixin of ``session.RoboTwinLiveSession``, moved verbatim from
``session.py`` on 2026-10-01; it holds methods only, no dataclass fields.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ora.environment.controllers.motion_transaction import (
    MotionTransactionError,
    select_motion_strategy,
)
from ora.environment.controllers.profile_capabilities import (
    OLDEST_COMPOSED_CONTROL_PROFILE,
    profile_has,
)

from .common import (
    RoboTwinLiveError,
    _SparseMotionState,
    _validate_workspace_target,
    _wxyz_to_xyzw,
    _xyzw_to_wxyz,
)


class RoboTwinSparseAdmissionMixin:
    """Sparse-motion entry: admission, plan references and strategy routing."""

    def _execute_sparse_motion(
        self, *, action: Mapping[str, Any], receipt_path: Path, started: float,
    ) -> dict[str, Any]:
        """Execute one versioned sparse update through the active motion transaction.

        The session binds a control profile of version
        ``OLDEST_COMPOSED_CONTROL_PROFILE`` or newer, so every sparse update runs
        the motion transaction. Semantic v3 never exposes controller choice to
        policy.
        """

        if not profile_has(self.control_profile, since_version=OLDEST_COMPOSED_CONTROL_PROFILE):
            raise RoboTwinLiveError(
                "sparse motion requires a control profile of version >= "
                f"{OLDEST_COMPOSED_CONTROL_PROFILE}"
            )
        state = self._admit_sparse_motion(
            action=action, receipt_path=receipt_path, started=started,
        )
        self._route_motion_strategy(state)
        self._begin_sparse_decision(state.environment)
        if self.operator_execution_video:
            self._capture_operator_execution_frame(
                action=action, phase="action_start", control_step_index=None,
            )
        if state.motion_strategy_error is not None:
            for arm in state.target_arms:
                state.public_pose_targets[arm] = _xyzw_to_wxyz(
                    state.targets[arm]["position_m"], state.targets[arm]["orientation_xyzw"],
                )
        try:
            if state.motion_strategy_error is not None:
                for arm in state.target_arms:
                    state.plan_rejections[arm] = {
                        "code": "motion_intent_rejected",
                        "message": state.motion_strategy_error,
                        "details": {"controller_writes": 0},
                    }
            elif state.local_cartesian_requested:
                self._run_local_servo(state)
            elif state.pose_action:
                if self._plan_and_validate(state):
                    self._execute_validated_plans(state)
            else:
                self._apply_gripper_targets(state)
        except Exception as exc:
            self._classify_sparse_exception(state, exc)
        self._combine_blocked_recovery_prefix(state)
        self._persist_motion_artifacts(state)
        self._settle_and_classify(state)
        return self._compose_sparse_receipt(state)

    def _admit_sparse_motion(
        self, *, action: Mapping[str, Any], receipt_path: Path, started: float,
    ) -> _SparseMotionState:
        """Admit one sparse action."""

        environment = self._require_environment()
        grip_center_profile = self._uses_grip_center_profile()
        arguments = action["arguments"]
        targets = dict(arguments["targets"])
        target_arms = tuple(arm for arm in ("left", "right") if arm in targets)
        return _SparseMotionState(
            action=action,
            receipt_path=receipt_path,
            started=started,
            environment=environment,
            grip_center_profile=grip_center_profile,
            arguments=arguments,
            targets=targets,
            target_arms=target_arms,
        )

    def _route_motion_strategy(self, state: _SparseMotionState) -> None:
        """Snapshot action-start state and select the motion strategy."""

        # The route is selected below from canonical physical intent.
        state.local_cartesian_requested = False
        state.global_then_local_settle_requested = False
        state.global_precontact_then_local_requested = False
        state.pose_action = state.action["action"] in {
            "move_pose", "move_free_space", "servo_delta", "recover_motion",
        }
        state.before = {
            arm: {
                "pose_xyz_xyzw": _wxyz_to_xyzw(
                    self._public_pose_wxyz(state.environment, arm)
                ),
                "gripper_command_target": self.gripper_command_targets[arm],
                "arm_drive_target_sha256": self._drive_target_digest(state.environment, arm),
                "gripper_drive_target_sha256": self._gripper_drive_target_digest(
                    state.environment, arm,
                ),
                **({
                    "controlled_point_semantics": "parallel_jaw_grip_center_tcp",
                } if state.grip_center_profile else {}),
            }
            for arm in ("left", "right")
        }
        state.measured_joint_snapshot = (
            {
                arm: self._measured_arm_joints(state.environment, arm)
                for arm in ("left", "right")
            }
            if state.grip_center_profile else {}
        )
        state.motion_adapter = (
            self._motion_backend_adapter(session=self, environment=state.environment)
            if self._uses_motion_transaction_profile() else None
        )
        state.motion_action_start = (
            state.motion_adapter.capture_action_start()
            if state.motion_adapter is not None else None
        )
        if self._uses_motion_intent_profile() and state.pose_action:
            if state.action["action"] == "recover_motion":
                state.motion_strategy = {
                    "schema": "ora.motion-strategy-decision.v1",
                    "strategy": "global",
                    "interactions": {
                        arm: "precision" for arm in state.target_arms
                    },
                    "physical_delta": {},
                    "local_eligible": False,
                    "endpoint_settle_enabled": False,
                    "reason": "joint_configuration_recovery",
                }
            else:
                interactions = dict(state.arguments["interactions"])
                try:
                    decision = select_motion_strategy(
                        action=str(state.action["action"]),
                        start_public_tcp_poses={
                            arm: state.before[arm]["pose_xyz_xyzw"] for arm in state.target_arms
                        },
                        target_public_tcp_poses={
                            arm: [
                                *state.targets[arm]["position_m"],
                                *state.targets[arm]["orientation_xyzw"],
                            ]
                            for arm in state.target_arms
                        },
                        interactions=interactions,
                        capabilities=state.motion_adapter.capabilities,
                        profile=self.control_profile["profile"]["strategy_router"],  # type: ignore[index]
                        plan_reference=False,
                    )
                except MotionTransactionError as exc:
                    state.motion_strategy_error = str(exc)
                    state.motion_strategy = {
                        "schema": "ora.motion-strategy-decision.v1",
                        "strategy": "rejected",
                        "interactions": interactions,
                        "physical_delta": {},
                        "local_eligible": False,
                        "endpoint_settle_enabled": False,
                        "reason": "intent_admission_failed",
                    }
                else:
                    state.motion_strategy = decision.as_dict()
                    state.local_cartesian_requested = decision.strategy == "local"
                    state.global_then_local_settle_requested = (
                        decision.strategy == "global_then_local_settle"
                        and decision.endpoint_settle_enabled
                    )
                    state.global_precontact_then_local_requested = (
                        decision.strategy == "global_precontact_then_local"
                    )

    def _run_local_servo(self, state: _SparseMotionState) -> None:
        """Run the local Cartesian DLS servo route."""

        for arm in state.target_arms:
            target_pose = state.targets[arm]
            _validate_workspace_target(
                list(target_pose["position_m"]), self.workspace_bounds_m,
            )
            state.public_pose_targets[arm] = _xyzw_to_wxyz(
                target_pose["position_m"], target_pose["orientation_xyzw"],
            )
        (
            state.controller_tracking,
            state.controller_tracking_arrays,
            state.control_sequence_record,
        ) = self._execute_local_cartesian_action(
            environment=state.environment,
            target_pose_xyzw={
                arm: _wxyz_to_xyzw(state.public_pose_targets[arm])
                for arm in state.target_arms
            },
            action=state.action,
        )
        state.execution_attempted = bool(state.controller_tracking.get("execution_started"))
        state.executed = bool(state.controller_tracking.get("sampled_steps", 0))
        state.payload_validation = state.controller_tracking.get(
            "payload_validation"
        )
        if not state.controller_tracking["completed"]:
            state.tracking_not_reached_reason = str(
                state.controller_tracking["abort"]["code"]
            )
        state.collision_validation = {
            "schema": (
                "ora.robotwin-local-cartesian-clearance.v1"
            ),
            "status": (
                "accepted"
                if state.controller_tracking["completed"] else "stopped"
            ),
            "checked_each_iteration": True,
            "minimum_center_distance_m": state.controller_tracking.get(
                "minimum_inter_arm_center_distance_m"
            ),
            **({"inter_arm_clearance": state.controller_tracking["inter_arm_clearance"]}
               if "inter_arm_clearance" in state.controller_tracking else {}),
        }
