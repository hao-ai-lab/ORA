"""Settling, the official terminal latch and the sparse receipt.

A mixin of ``session.RoboTwinLiveSession``, moved verbatim from
``session.py`` on 2026-10-01; it holds methods only, no dataclass fields.
"""

from __future__ import annotations

import time
from typing import Any, Mapping

from ora.contracts.acceptance import motion_acceptance_check
from ora.environment.controllers.motion_transaction import (
    classify_motion_status,
)
from ora.environment.controllers.profile_capabilities import (
    CONTACT_RECOVERY_SINCE,
    profile_has,
)

from .common import (
    SPARSE_RECEIPT_SCHEMA,
    SPARSE_RECEIPT_V3_SCHEMA,
    SPARSE_RECEIPT_V4_SCHEMA,
    SPARSE_RECEIPT_V5_SCHEMA,
    SPARSE_RECEIPT_V6_SCHEMA,
    SPARSE_RECEIPT_V7_SCHEMA,
    _SparseMotionState,
    _atomic_json,
    _file_hash,
    _jsonable,
    _pose_error,
    _wxyz_to_xyzw,
)


class RoboTwinSparseSettlementMixin:
    """Settling, the official terminal latch and the sparse receipt."""

    def _settle_and_classify(self, state: _SparseMotionState) -> None:
        """Refresh reached poses and classify execution status."""

        updater = getattr(state.environment, "_update_render", None)
        if callable(updater):
            updater()
        state.reached = {
            arm: _wxyz_to_xyzw(self._public_pose_wxyz(state.environment, arm))
            for arm in ("left", "right")
        }
        if bool(getattr(state.environment, "eval_success", False)) and state.execution_fault is None:
            state.execution_status = "executed"
        elif state.pose_action:
            arrival = (
                self.control_profile["profile"]["local_cartesian"]  # type: ignore[index]
                if (
                    state.local_cartesian_requested
                    or state.global_precontact_then_local_requested
                )
                else self.control_profile["profile"]["arrival_check"]  # type: ignore[index]
            )
            state.execution_status, _transaction_endpoint_errors = classify_motion_status(
                planner_failed=(
                    not state.local_cartesian_requested
                    and not state.executed
                    and not state.plan_rejections
                ),
                plan_rejected=bool(state.plan_rejections),
                execution_started=state.executed,
                execution_fault=state.execution_fault,
                convergence_completed=bool(
                    isinstance(state.controller_tracking, Mapping)
                    and state.controller_tracking.get("completed") is True
                    and state.tracking_not_reached_reason is None
                ),
                target_poses={
                    arm: _wxyz_to_xyzw(state.public_pose_targets[arm])
                    for arm in state.target_arms
                },
                measured_poses={arm: state.reached[arm] for arm in state.target_arms},
                position_tolerance_m=float(arrival["position_tolerance_m"]),
                orientation_tolerance_rad=float(
                    arrival["orientation_tolerance_rad"]
                ),
            )
            if _transaction_endpoint_errors:
                state.acceptance_checks.append(motion_acceptance_check(
                    stage="final_endpoint", errors=_transaction_endpoint_errors,
                    position_tolerance_m=float(arrival["position_tolerance_m"]),
                    orientation_tolerance_rad=float(arrival["orientation_tolerance_rad"])))
        elif state.execution_fault is not None:
            state.execution_status = "execution_fault"
        elif state.plan_rejections:
            state.execution_status = "plan_rejected"
        elif state.tracking_not_reached_reason is not None:
            state.execution_status = "not_reached"
        else:
            state.execution_status = "executed" if state.executed else "not_executed"
        if (
            state.execution_status == "reached"
            and state.action["action"] != "recover_motion"
        ):
            for arm in state.target_arms:
                self.last_safe_pose[arm] = list(state.reached[arm])
                self.last_safe_joints[arm] = self._measured_arm_joints(state.environment, arm)

    @staticmethod
    def _latch_official_success(environment: Any) -> bool:
        """Read the official per-episode latch, polling its predicate if needed."""

        if bool(getattr(environment, "eval_success", False)):
            return True
        checker = getattr(environment, "check_success", None)
        if not callable(checker):
            return False
        try:
            held = bool(checker())
        except Exception:
            # A missing mid-step predicate value never invents success.
            return False
        if held:
            environment.eval_success = True
        return held

    def _after_physics_step(
        self, environment: Any, *, action: Mapping[str, Any], phase: str,
    ) -> bool:
        """Stop at the first official success and freeze its public observation."""

        self.physics_step_sequence += 1
        if not self._latch_official_success(environment):
            return False
        self._freeze_official_terminal(environment, action=action, phase=phase)
        return True

    def _freeze_official_terminal(
        self, environment: Any, *, action: Mapping[str, Any] | None = None,
        phase: str = "upstream_action_boundary",
    ) -> None:
        """Persist one immutable control-only snapshot at the official boundary."""

        import numpy as np

        if self.official_terminal is not None:
            return
        raw = self._read_observation_data(environment)
        public_poses = {
            arm: list(self._public_pose_wxyz(environment, arm))
            for arm in ("left", "right")
        }
        directives = dict(self.gripper_command_targets)
        from .gripper_state import public_gripper_state

        # The frozen terminal observation is replayed by ``observe``; it must
        # carry the same measured jaw evidence as any live observation.
        raw["public_gripper_state"] = public_gripper_state(
            getattr(environment, "robot", None),
        )
        if action is not None and action.get("action") == "set_gripper":
            targets = action.get("arguments", {}).get("targets", {})
            directives.update({arm: str(value) for arm, value in targets.items()})
        root = self.output_root / "official-terminal" / f"round-{self.round_index:04d}"
        root.mkdir(parents=True, exist_ok=False)
        arrays = {
            f"{name}__{key}": np.asarray(camera[key])
            for name, camera in raw["observation"].items()
            for key in ("rgb", "depth", "intrinsic_cv", "extrinsic_cv", "cam2world_gl")
        }
        arrays_path = root / "rgbd.npz"
        np.savez_compressed(arrays_path, **arrays)
        captured = time.time()
        snapshot = {
            "schema": "ora.robotwin-terminal-snapshot.v1",
            "captured_unix_s": captured,
            "endpose": _jsonable(raw.get("endpose")),
            "joint_action": _jsonable(raw.get("joint_action")),
            "public_pose_xyz_wxyz": public_poses,
            "gripper_command_targets": directives,
            **({"public_gripper_state": raw["public_gripper_state"]}
               if "public_gripper_state" in raw else {}),
            "cameras": sorted(raw["observation"]),
            "rgbd": {"path": "rgbd.npz", "sha256": _file_hash(arrays_path)},
        }
        path = root / "snapshot.json"
        _atomic_json(path, snapshot)
        self.terminal_observation = raw
        self.terminal_public_poses = public_poses
        self.terminal_gripper_targets = directives
        self.official_terminal = {
            "schema": "ora.robotwin-official-terminal.v1",
            "source": "RoboTwin.eval_success",
            "official_success": True,
            "round_index": self.round_index,
            "semantic_action_count": int(environment.take_action_cnt),
            "physics_step_index": (
                self.physics_step_sequence
                if phase != "upstream_action_boundary" else None
            ),
            "action_sequence": self.action_sequence,
            "action_id": action.get("action_id") if action is not None else None,
            "phase": phase,
            "captured_unix_s": captured,
            "snapshot": {
                "path": path.relative_to(self.output_root).as_posix(),
                "sha256": _file_hash(path),
            },
        }
        _atomic_json(root / "terminal.json", self.official_terminal)
        for artifact in root.iterdir():
            artifact.chmod(0o444)

    def _compose_sparse_receipt(
        self, state: _SparseMotionState,
    ) -> dict[str, Any]:
        """Compose, persist and register the sparse action receipt."""

        official_success = bool(getattr(state.environment, "eval_success", False))
        if official_success:
            self._freeze_official_terminal(state.environment, action=state.action)
        self._capture_operator_frame(label=f"action-{self.action_sequence:04d}")
        after = {
            arm: {
                "pose_xyz_xyzw": state.reached[arm],
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
        arm_effects: dict[str, Any] = {}
        for arm in ("left", "right"):
            target_xyzw = (
                _wxyz_to_xyzw(state.public_pose_targets[arm])
                if arm in state.public_pose_targets else None
            )
            arm_effect = {
                "targeted": arm in state.targets,
                "planner": self._planner_record(
                    state.planner_results[arm],
                    called=(
                        state.pose_action and arm in state.targets
                        and not state.local_cartesian_requested
                        and not (state.action["action"] == "recover_motion"
                                 and state.planner_results[arm] is None)
                    ),
                ),
                "pose_error": (
                    _pose_error(target_xyzw, after[arm]["pose_xyz_xyzw"])
                    if target_xyzw is not None else None
                ),
                "inactive_arm_drift": (
                    _pose_error(
                        state.before[arm]["pose_xyz_xyzw"], after[arm]["pose_xyz_xyzw"],
                    )
                    if arm not in state.targets or not state.pose_action else None
                ),
                "command_preservation": {
                    "arm_drive_target_unchanged": (
                        state.before[arm]["arm_drive_target_sha256"] is not None
                        and state.before[arm]["arm_drive_target_sha256"]
                        == after[arm]["arm_drive_target_sha256"]
                    ) if state.action["action"] == "set_gripper" or arm not in state.targets else None,
                    "gripper_drive_target_unchanged": (
                        state.before[arm]["gripper_drive_target_sha256"] is not None
                        and state.before[arm]["gripper_drive_target_sha256"]
                        == after[arm]["gripper_drive_target_sha256"]
                    ) if state.action["action"] != "set_gripper" or arm not in state.targets else None,
                    "gripper_command_target_before": state.before[arm]["gripper_command_target"],
                    "gripper_command_target_after": after[arm]["gripper_command_target"],
                },
            }
            if state.grip_center_profile:
                arm_effect["motion_backend"] = (
                    "intent_rejected"
                    if state.motion_strategy_error is not None and arm in state.targets
                    else "local_cartesian_dls"
                    if state.local_cartesian_requested and arm in state.targets
                    else "global_curobo_precontact_then_local"
                    if state.global_precontact_then_local_requested and arm in state.targets
                    else "global_curobo_then_local_settle"
                    if state.global_then_local_settle_requested and arm in state.targets
                    else "global_curobo"
                    if state.pose_action and arm in state.targets
                    else None
                )
                arm_effect["hold_controller"] = {
                    "engaged": state.hold_engaged[arm],
                    "source": (
                        "action_start_measured_joint_state"
                        if state.hold_engaged[arm] else None
                    ),
                    "planner_called": (
                        state.pose_action and arm in state.targets
                        and not state.local_cartesian_requested
                        and not (state.action["action"] == "recover_motion"
                                 and state.planner_results[arm] is None)
                    ),
                }
            if self._uses_validated_planner_profile():
                arm_effect["plan_validation"] = (
                    state.validated_plans[arm].summary
                    if arm in state.validated_plans
                    else {
                        "schema": (
                            "ora.motion-plan-validation.v1"
                            if self._uses_motion_transaction_profile()
                            else "ora.robotwin-plan-validation.v1"
                        ),
                        "status": "rejected",
                        **state.plan_rejections[arm],
                    }
                    if arm in state.plan_rejections else None
                )
                arm_effect["plan_artifact"] = state.plan_artifacts.get(arm)
            arm_effects[arm] = arm_effect
        failure_stage: str | None = None
        failure_code: str | None = None
        if state.execution_status == "planner_failed":
            failure_stage = "planner"
            failed = next(
                (
                    state.planner_results[arm] for arm in state.target_arms
                    if isinstance(state.planner_results.get(arm), Mapping)
                    and state.planner_results[arm].get("status") != "Success"
                ),
                None,
            )
            failure_code = str(
                failed.get("stage")
                or failed.get("planner_status")
                or failed.get("status")
                or "planner_failed"
            ) if isinstance(failed, Mapping) else "planner_failed"
        elif state.execution_status == "plan_rejected":
            failure_stage = "plan_validation"
            rejected = next(iter(state.plan_rejections.values()), {})
            failure_code = str(rejected.get("code", "plan_rejected"))
        elif state.execution_status == "execution_fault":
            failure_stage = "execution"
            failure_code = str(
                (state.controller_tracking or {}).get("abort", {}).get(
                    "code", "execution_fault"
                )
                if isinstance((state.controller_tracking or {}).get("abort"), Mapping)
                else "execution_fault"
            )
        elif state.execution_status == "not_reached":
            failure_stage = (
                "execution" if state.tracking_not_reached_reason else "endpoint_verification"
            )
            failure_code = (
                state.tracking_not_reached_reason or "endpoint_residual_exceeded"
            )
        receipt = {
            "schema": (
                SPARSE_RECEIPT_V7_SCHEMA
                if self._uses_motion_intent_profile()
                else SPARSE_RECEIPT_V6_SCHEMA
                if self._uses_motion_transaction_profile()
                else SPARSE_RECEIPT_V5_SCHEMA
                if self._uses_validated_planner_profile()
                else SPARSE_RECEIPT_V4_SCHEMA
                if self._uses_controller_tracking_profile()
                else SPARSE_RECEIPT_V3_SCHEMA
                if state.grip_center_profile else SPARSE_RECEIPT_SCHEMA
            ),
            "action_id": state.action["action_id"],
            **({"acceptance_checks": state.acceptance_checks} if state.acceptance_checks else {}),
            "action": state.action["action"],
            "arguments": state.action["arguments"],
            "target_arms": list(state.target_arms),
            "consumed": True,
            "state_changed": state.executed,
            # A pre-execution refusal must carry an explicit zero certificate.
            # Missing tracking is not itself evidence of zero execution.
            **({"steps_used": 0} if not state.execution_attempted and not state.executed
               and state.execution_fault is None else {}),
            **({
                "execution_status": state.execution_status,
                "controlled_point_semantics": "parallel_jaw_grip_center_tcp",
                "planner_input_point": "link6",
                "motion_backend": (
                    "intent_rejected"
                    if state.motion_strategy_error is not None
                    else "local_cartesian_dls"
                    if state.local_cartesian_requested
                    else "global_curobo_precontact_then_local"
                    if state.global_precontact_then_local_requested
                    else "global_curobo_then_local_settle"
                    if state.global_then_local_settle_requested
                    else "global_curobo" if state.pose_action else "gripper_actuator"
                ),
                **({"motion_strategy": state.motion_strategy}
                   if self._uses_motion_intent_profile() and state.pose_action else {}),
                **({
                    "failure_stage": failure_stage,
                    "failure_code": failure_code,
                } if self._uses_motion_intent_profile() else {}),
                "settle_steps": state.settle_steps,
                **({
                    "not_reached_reason": state.tracking_not_reached_reason,
                    "plan_rejections": state.plan_rejections,
                } if self._uses_validated_planner_profile() else {}),
                **({
                    "controller_tracking": state.controller_tracking,
                } if self._uses_controller_tracking_profile() else {}),
                **({
                "collision_validation": state.collision_validation,
                    "plan_ref": None,
                    "payload_ref": None,
                    "payload_validation": state.payload_validation,
                    "invalidated_payload_refs": state.invalidated_payload_refs,
                } if self._uses_validated_planner_profile() else {}),
            } if state.grip_center_profile else {}),
            "synchronization": {
                "requested": len(state.target_arms) == 2,
                "planned_from_common_snapshot": (
                    len(state.target_arms) == 2 and state.pose_action
                    and state.motion_strategy_error is None
                ),
                "all_planners_succeeded": all(
                    arm_effects[arm]["planner"]["status"] == "Success"
                    for arm in state.target_arms
                ) if state.pose_action and not state.local_cartesian_requested else None,
                **({
                    "all_plans_admitted": (
                        not state.plan_rejections
                        and all(arm in state.validated_plans for arm in state.target_arms)
                    ) if state.pose_action and not state.local_cartesian_requested else None,
                } if self._uses_validated_planner_profile() else {}),
                "executed": state.executed,
            },
            "arms": arm_effects,
            "before": state.before,
            "after": after,
            "execution_fault": state.execution_fault,
            **({
                "blocked_motion": self._blocked_motion_evidence(state),
                "blocked_recovery": state.blocked_recovery,
                "inter_arm_clearance": (state.controller_tracking or {}).get(
                    "inter_arm_clearance", state.collision_validation,
                ),
            } if profile_has(self.control_profile, since_version=CONTACT_RECOVERY_SINCE) else {}),
            "official_evaluator_queried": True,
            "terminal_after_action": official_success,
            "semantic_action_count": int(state.environment.take_action_cnt),
            "started_unix_s": state.started,
            "finished_unix_s": time.time(),
        }
        _atomic_json(state.receipt_path, receipt)
        self.replay_recorder.update(
            self.action_sequence,
            execution_status=state.execution_status,
            settle_steps=state.settle_steps,
            execution_fault=state.execution_fault,
        )
        record = {
            "path": state.receipt_path.relative_to(self.output_root).as_posix(),
            "sha256": _file_hash(state.receipt_path),
            "receipt": receipt,
        }
        self.last_receipt = record
        self.action_sequence += 1
        return record
