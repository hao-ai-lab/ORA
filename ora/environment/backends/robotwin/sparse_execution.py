"""Execution of validated sparse plans and gripper targets, and their artifacts.

A mixin of ``session.RoboTwinLiveSession``; it holds methods only, no
dataclass fields. ``_execute_validated_plans`` builds the dense control
sequence, runs it through the measured motion transaction and, for a staged route,
selects the post-global phase and runs the local Cartesian settle, whose
control sequence ``_merged_settle_control_sequence`` appends to the global one.
"""

from __future__ import annotations

from typing import Any, Mapping

from ora.environment.controllers.motion_transaction import (
    select_post_global_phase,
    write_motion_plan_artifact,
)

from .common import (
    RoboTwinControllerTrackingError,
    RoboTwinLiveError,
    _SparseMotionState,
    _json_hash,
    _pose_error,
    _wxyz_to_xyzw,
)


class RoboTwinSparseExecutionMixin:
    """Execution of validated sparse plans and gripper targets, and their artifacts."""

    def _execute_validated_plans(self, state: _SparseMotionState) -> None:
        """Execute admitted plans, including staged post-global phases."""

        control_seq = self._validated_plan_control_sequence(state)
        state.execution_attempted = True
        state.control_sequence_record = control_seq
        self._dispatch_validated_plan_sequence(state, control_seq)
        state.executed = True
        if (
            (
                state.global_then_local_settle_requested
                or state.global_precontact_then_local_requested
            )
            and state.motion_strategy is not None
            and not bool(getattr(state.environment, "eval_success", False))
        ):
            self._run_post_global_phase(state)

    def _validated_plan_control_sequence(
        self, state: _SparseMotionState,
    ) -> dict[str, Any]:
        """Build the dense control sequence of the targeted arms."""

        control_seq = {
            "left_arm": None, "left_gripper": None,
            "right_arm": None, "right_gripper": None,
        }
        for arm in state.target_arms:
            result = (
                {
                    "status": "Validated",
                    "position": state.validated_plans[arm].position,
                    "velocity": state.validated_plans[arm].velocity,
                }
                if arm in state.validated_plans else state.planner_results[arm]
            )
            control_seq[f"{arm}_arm"] = result
        return control_seq

    def _dispatch_validated_plan_sequence(
        self, state: _SparseMotionState, control_seq: dict[str, Any],
    ) -> None:
        """Run the control sequence through the measured motion transaction."""

        precontact_position_tolerance = None
        precontact_orientation_tolerance = None
        if (
            state.global_precontact_then_local_requested
            and self._uses_rgbd_grasp_staging_profile()
        ):
            router_profile = self.control_profile["profile"][  # type: ignore[index]
                "strategy_router"
            ]
            precontact_position_tolerance = float(
                router_profile["precontact_arrival_reserve_m"]
            )
            precontact_orientation_tolerance = float(
                router_profile["precontact_arrival_reserve_rad"]
            )
        (
            state.controller_tracking,
            state.controller_tracking_arrays,
            state.control_sequence_record,
        ) = self._execute_motion_transaction_dense_action(
            environment=state.environment,
            control_sequence=control_seq,
            action=state.action,
            endpoint_public_tcp_targets={
                arm: _wxyz_to_xyzw(
                    state.planner_public_pose_targets[arm]
                )
                for arm in state.target_arms
            },
            endpoint_position_tolerance_m=(
                precontact_position_tolerance
            ),
            endpoint_orientation_tolerance_rad=(
                precontact_orientation_tolerance
            ),
        )
        if not state.controller_tracking["completed"]:
            state.tracking_not_reached_reason = str(
                state.controller_tracking["abort"]["code"]
            )

    def _run_post_global_phase(self, state: _SparseMotionState) -> None:
        """After a staged global stage, settle locally or record why not."""

        post_global_phase = self._select_sparse_post_global_phase(state)
        if post_global_phase != "run_local":
            state.motion_strategy["settle_result"] = post_global_phase
            if (
                post_global_phase != "not_needed"
                and state.tracking_not_reached_reason is None
            ):
                state.tracking_not_reached_reason = post_global_phase
        if post_global_phase == "not_needed":
            state.motion_strategy["settle_result"] = "not_needed"
        elif post_global_phase == "run_local":
            self._run_local_settle_phase(state)

    def _select_sparse_post_global_phase(self, state: _SparseMotionState) -> str:
        """Measure the global-stage endpoint errors and select the next phase."""

        arrival = self.control_profile["profile"]["arrival_check"]  # type: ignore[index]
        endpoint_profile = self.control_profile["profile"][  # type: ignore[index]
            "strategy_router"
        ]["endpoint_settle"]
        current_after_global = {
            arm: _wxyz_to_xyzw(
                self._public_pose_wxyz(state.environment, arm)
            )
            for arm in state.target_arms
        }
        global_stage_errors = {
            arm: _pose_error(
                _wxyz_to_xyzw(state.planner_public_pose_targets[arm]),
                current_after_global[arm],
            )
            for arm in state.target_arms
        }
        global_tracking_completed = bool(
            isinstance(state.controller_tracking, Mapping)
            and state.controller_tracking.get("completed") is True
            and state.tracking_not_reached_reason is None
        )
        global_final_errors = {
            arm: _pose_error(
                _wxyz_to_xyzw(state.public_pose_targets[arm]),
                current_after_global[arm],
            )
            for arm in state.target_arms
        }
        post_global_phase = select_post_global_phase(
            acceptance_evidence=state.acceptance_checks,
            staged_contact=state.global_precontact_then_local_requested,
            tracking_completed=global_tracking_completed,
            stage_endpoint_errors=global_stage_errors,
            final_endpoint_errors=global_final_errors,
            arrival_position_tolerance_m=float(
                arrival["position_tolerance_m"]
            ),
            arrival_orientation_tolerance_rad=float(
                arrival["orientation_tolerance_rad"]
            ),
            local_translation_envelope_m=float(
                endpoint_profile["maximum_translation_m"]
            ),
            local_rotation_envelope_rad=float(
                endpoint_profile["maximum_rotation_rad"]
            ),
            stage_arrival_position_tolerance_m=(
                float(
                    self.control_profile["profile"][  # type: ignore[index]
                        "strategy_router"
                    ]["precontact_arrival_reserve_m"]
                )
                if (
                    state.global_precontact_then_local_requested
                    and self._uses_rgbd_grasp_staging_profile()
                )
                else None
            ),
            stage_arrival_orientation_tolerance_rad=(
                float(
                    self.control_profile["profile"][  # type: ignore[index]
                        "strategy_router"
                    ]["precontact_arrival_reserve_rad"]
                )
                if (
                    state.global_precontact_then_local_requested
                    and self._uses_rgbd_grasp_staging_profile()
                )
                else None
            ),
            final_arrival_position_tolerance_m=(
                float(
                    self.control_profile["profile"][  # type: ignore[index]
                        "local_cartesian"
                    ]["position_tolerance_m"]
                )
                if (
                    state.global_precontact_then_local_requested
                    and self._uses_rgbd_grasp_staging_profile()
                )
                else None
            ),
            final_arrival_orientation_tolerance_rad=(
                float(
                    self.control_profile["profile"][  # type: ignore[index]
                        "local_cartesian"
                    ]["orientation_tolerance_rad"]
                )
                if (
                    state.global_precontact_then_local_requested
                    and self._uses_rgbd_grasp_staging_profile()
                )
                else None
            ),
        )
        state.motion_strategy["global_stage_endpoint_error"] = (
            global_stage_errors
        )
        state.motion_strategy["global_final_endpoint_error"] = (
            global_final_errors
        )
        state.motion_strategy["settle_attempted"] = False
        return post_global_phase

    def _run_local_settle_phase(self, state: _SparseMotionState) -> None:
        """Run the local Cartesian phase and merge it with the global stage."""

        global_tracking = dict(state.controller_tracking)
        global_arrays = dict(state.controller_tracking_arrays)
        global_sequence = state.control_sequence_record
        (
            settle_tracking,
            settle_arrays,
            settle_sequence,
        ) = self._execute_local_cartesian_action(
            environment=state.environment,
            target_pose_xyzw={
                arm: _wxyz_to_xyzw(state.public_pose_targets[arm])
                for arm in state.target_arms
            },
            action=state.action,
        )
        state.motion_strategy["settle_attempted"] = True
        state.motion_strategy["settle_result"] = (
            "reached"
            if settle_tracking.get("completed") is True
            else str(
                (settle_tracking.get("abort") or {}).get(
                    "code", "not_reached"
                )
            )
        )
        state.controller_tracking = {
            "schema": "ora.motion-execution-pipeline.v1",
            "transaction_schema": "ora.motion-transaction.v1",
            "backend": "robotwin",
            "mode": (
                "global_precontact_then_local"
                if state.global_precontact_then_local_requested
                else "global_then_local_settle"
            ),
            "completed": bool(settle_tracking.get("completed")),
            "abort": settle_tracking.get("abort"),
            "sampled_steps": int(global_tracking.get("sampled_steps", 0))
            + int(settle_tracking.get("sampled_steps", 0)),
            "phases": ({
                "global_precontact": global_tracking,
                "local_contact": settle_tracking,
            } if state.global_precontact_then_local_requested else {
                "global": global_tracking,
                "local_settle": settle_tracking,
            }),
            "arms": settle_tracking.get("arms", {}),
        }
        state.settle_steps = int(
            settle_tracking.get("sampled_steps", 0)
        )
        state.controller_tracking_arrays = {
            **{
                f"global__{key}": value
                for key, value in global_arrays.items()
            },
            **dict(settle_arrays),
        }
        if (
            isinstance(global_sequence, Mapping)
            and isinstance(settle_sequence, Mapping)
        ):
            state.control_sequence_record = _merged_settle_control_sequence(
                global_sequence, settle_sequence,
                staged_contact=state.global_precontact_then_local_requested,
            )
        if not state.controller_tracking["completed"]:
            state.tracking_not_reached_reason = str(
                (state.controller_tracking.get("abort") or {}).get(
                    "code", "endpoint_settle_not_reached"
                )
            )

    def _apply_gripper_targets(self, state: _SparseMotionState) -> None:
        """Apply set_gripper targets while holding arm drive targets."""

        gripper_paths: dict[str, Any] = {"left": None, "right": None}
        if len(state.target_arms) == 2:
            gripper_paths["left"], gripper_paths["right"] = state.environment.set_gripper(
                set_tag="together",
                left_pos=1.0 if state.targets["left"] == "open" else 0.0,
                right_pos=1.0 if state.targets["right"] == "open" else 0.0,
            )
        else:
            arm = state.target_arms[0]
            gripper_paths[arm] = state.environment.set_gripper(
                set_tag=arm,
                **{f"{arm}_pos": 1.0 if state.targets[arm] == "open" else 0.0},
            )
        if all(gripper_paths[arm] is not None for arm in state.target_arms):
            state.execution_attempted = True
            control_sequence = {
                "left_arm": None,
                "left_gripper": gripper_paths["left"],
                "right_arm": None,
                "right_gripper": gripper_paths["right"],
            }
            state.control_sequence_record = control_sequence
            (
                state.controller_tracking,
                state.controller_tracking_arrays,
                state.control_sequence_record,
            ) = self._execute_motion_transaction_dense_action(
                environment=state.environment,
                control_sequence=control_sequence,
                action=state.action,
            )
            if not state.controller_tracking["completed"]:
                state.tracking_not_reached_reason = str(
                    state.controller_tracking["abort"]["code"]
                )
            state.executed = True
            for arm in state.target_arms:
                self.gripper_command_targets[arm] = str(state.targets[arm])

    def _classify_sparse_exception(
        self, state: _SparseMotionState, exc: Exception,
    ) -> None:
        """Classify one execution exception into fault or rejection."""

        if isinstance(exc, RoboTwinControllerTrackingError):
            state.execution_fault = f"{type(exc).__name__}: {exc}"
            state.controller_tracking = dict(exc.summary)
            state.controller_tracking_arrays = dict(exc.arrays)
            if exc.control_sequence is not None:
                state.control_sequence_record = dict(exc.control_sequence)
            state.executed = bool(
                state.controller_tracking.get("execution_started")
                or state.controller_tracking.get("sampled_steps", 0)
            )
        elif isinstance(exc, RoboTwinLiveError) and not state.execution_attempted:
            # Admission or validation rejected the request before any
            # controller write. That is a plan rejection the policy can
            # act on, not an infrastructure execution fault.
            for arm in state.target_arms:
                state.plan_rejections.setdefault(arm, {
                    "code": "admission_rejected",
                    "message": str(exc),
                    "details": {
                        "exception": type(exc).__name__,
                        "stage": "pre_execution_admission",
                    },
                })
        else:
            # Anything else is an infrastructure fault; keep the honest
            # phase so a pre-write program error is never presented as a
            # simulator execution failure.
            stage = "execution" if state.execution_attempted else "pre_execution"
            state.execution_fault = f"{stage} {type(exc).__name__}: {exc}"

    def _persist_motion_artifacts(self, state: _SparseMotionState) -> None:
        """Write plan artifacts and the controller replay record."""

        if state.validated_plans and not state.plan_rejections:
            artifact_arrays: dict[str, Any] = dict(state.controller_tracking_arrays)
            for arm, validated in state.validated_plans.items():
                artifact_arrays.update({
                    f"planned_{arm}_joint__position": validated.position,
                    f"planned_{arm}_joint__velocity": validated.velocity,
                    f"planned_{arm}_control__pose_xyzw": validated.planned_link6_pose_xyzw,
                    f"planned_{arm}_public_tcp__pose_xyzw": validated.planned_tcp_pose_xyzw,
                })
            artifact = write_motion_plan_artifact(
                self.output_root,
                transaction_id=(
                    f"action-{self.action_sequence:04d}-"
                    f"{_json_hash({'action_id': state.action['action_id']})[:12]}"
                ),
                backend_id="robotwin",
                embodiment_id="aloha_agilex",
                binding={
                    "action_id": state.action["action_id"],
                    "source_plan_ref": None,
                    "source_revision": self.source_manifest["robotwin_revision"],
                    "control_profile_sha256": self.control_profile["sha256"],
                    "world_mutation_sequence_at_start": self.world_mutation_sequence,
                    "measured_start_joints": {
                        arm: state.measured_joint_snapshot[arm] for arm in state.target_arms
                    },
                    "target_public_tcp_pose_xyzw": {
                        arm: _wxyz_to_xyzw(state.planner_public_pose_targets[arm])
                        for arm in state.target_arms
                    },
                    "requested_final_public_tcp_pose_xyzw": {
                        arm: _wxyz_to_xyzw(state.public_pose_targets[arm])
                        for arm in state.target_arms
                    },
                    "collision_validation": state.collision_validation,
                },
                validation={
                    "arms": {
                        arm: state.validated_plans[arm].summary for arm in state.target_arms
                    },
                    "execution": state.controller_tracking,
                },
                arrays=artifact_arrays,
            )
            for arm in state.target_arms:
                state.plan_artifacts[arm] = artifact
        if state.local_cartesian_requested and state.controller_tracking is not None:
            artifact = write_motion_plan_artifact(
                self.output_root,
                transaction_id=(
                    f"action-{self.action_sequence:04d}-local-"
                    f"{_json_hash({'action_id': state.action['action_id']})[:12]}"
                ),
                backend_id="robotwin",
                embodiment_id="aloha_agilex",
                binding={
                    "action_id": state.action["action_id"],
                    "source_revision": self.source_manifest["robotwin_revision"],
                    "control_profile_sha256": self.control_profile["sha256"],
                    "world_mutation_sequence_at_start": self.world_mutation_sequence,
                    "measured_start_joints": {
                        arm: state.measured_joint_snapshot[arm] for arm in state.target_arms
                    },
                    "target_public_tcp_pose_xyzw": {
                        arm: _wxyz_to_xyzw(state.public_pose_targets[arm])
                        for arm in state.target_arms
                    },
                },
                validation={
                    "mode": "local_cartesian_dls",
                    "execution": state.controller_tracking,
                },
                arrays=state.controller_tracking_arrays,
            )
            for arm in state.target_arms:
                state.plan_artifacts[arm] = artifact
        if state.control_sequence_record is not None:
            try:
                self._record_control_sequence(
                    state,
                    state.control_sequence_record,
                    replayable=state.execution_fault is None,
                )
            except Exception as exc:
                state.execution_fault = f"{type(exc).__name__}: {exc}"
        if state.executed:
            self.world_mutation_sequence += 1
            self.world_mutation_actions.append(str(state.action["action"]))
        if not state.replay_recorded:
            self.replay_recorder.record(
                action_sequence=self.action_sequence,
                action=state.action,
                controller="robotwin_dense_joint_targets",
                arrays={},
                physics_steps=0,
                replayable=False,
                metadata={"reason": "no_controller_sequence_executed"},
            )


def _merged_settle_control_sequence(
    global_sequence: Mapping[str, Any],
    settle_sequence: Mapping[str, Any],
    *,
    staged_contact: bool,
) -> dict[str, Any]:
    """Concatenate the global and local control sequences channel by channel."""

    import numpy as np

    merged_sequence: dict[str, Any] = {}
    for channel in (
        "left_arm", "left_gripper",
        "right_arm", "right_gripper",
    ):
        first = global_sequence.get(channel)
        second = settle_sequence.get(channel)
        if isinstance(first, Mapping) and isinstance(second, Mapping):
            merged = dict(second)
            for field in ("position", "velocity"):
                if field in first and field in second:
                    merged[field] = np.concatenate(
                        [
                            np.asarray(first[field]),
                            np.asarray(second[field]),
                        ],
                        axis=0,
                    )
            merged["status"] = (
                "GlobalPrecontactThenLocal"
                if staged_contact
                else "GlobalThenLocalSettle"
            )
            merged_sequence[channel] = merged
        else:
            merged_sequence[channel] = (
                second if second is not None else first
            )
    return merged_sequence
