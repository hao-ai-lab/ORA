"""The global CuRobo / JOINT_POSITION route and its world-to-planner frames.

A mixin of ``session.LiberoProSession``; it holds methods only, no dataclass
fields. Shared constants and helpers come from ``common``.
"""

from __future__ import annotations

import math
import time
from typing import Any, Mapping

from ora.environment.controllers.curobo_motion import _curobo_failure_reason
from ora.environment.controllers.hybrid_motion import (
    LIBERO_ENDPOINT_ROUTE_GLOBAL,
    libero_endpoint_contract,
    libero_global_endpoint_tolerances,
    precision_force_stop_enabled,
)
from ora.contracts.workspace import workspace_state

from .common import (
    HYBRID_RECEIPT_SCHEMA,
    CONTROL_HEALTH_SCHEMA,
    LiberoProError,
    _quaternion_error_rotvec,
)


class LiberoPlannedMotionMixin:
    """The global CuRobo / JOINT_POSITION route and its world-to-planner frames."""

    def _remember_backtrack_state(self) -> None:
        """Freeze the state before a non-recovery motion mutation."""

        state = self._motion_state()
        joints = state["joint_positions"]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in joints
        ):
            raise LiberoProError("backtrack state has invalid Panda joints")
        self.last_backtrack_joint_state = [float(value) for value in joints]
        pose = state.get("grip_site_pose_world_xyzw")
        self.last_backtrack_tcp_pose = list(pose) if pose is not None else None

    def _refresh_joint_observation(self, steps: int) -> None:
        """Update the driver's observation cache from the last dispatched step, without stepping."""

        if steps == 0:
            return
        raw = self._worker_attr("harness_joint_execution_observation")
        if not isinstance(raw, Mapping):
            raise LiberoProError("joint execution observation is unavailable")
        environment = self._require_environment()
        environment.current_raw_obs = [dict(raw)]
        # Direct joint dispatch bypasses the driver's step, including its counter. The
        # observation transfer itself consumes zero simulation steps.
        if hasattr(environment, "_elapsed_steps"):
            environment._elapsed_steps += steps
        self.last_raw_observation = dict(raw)

    def _validate_planned_start(
        self, planned: Mapping[str, Any], motion_state: Mapping[str, Any],
    ) -> None:
        """Apply identical measured-start admission to direct/probe/reference."""

        import numpy as np
        from scipy.spatial.transform import Rotation

        trajectory = planned.get("trajectory")
        if not isinstance(trajectory, Mapping):
            raise LiberoProError("CuRobo omitted its trajectory")
        names = [f"panda_joint{index}" for index in range(1, 8)]
        if trajectory.get("joint_names") != names or motion_state.get("joint_names") != names:
            raise LiberoProError("CuRobo Panda joint identity drifted")
        positions = np.asarray(trajectory.get("positions"), dtype=np.float64)
        measured_joints = np.asarray(motion_state["joint_positions"], dtype=np.float64)
        if (
            positions.ndim != 2 or positions.shape[1] != 7 or not len(positions)
            or measured_joints.shape != (7,)
            or not np.isfinite(positions).all()
            or not np.isfinite(measured_joints).all()
        ):
            raise LiberoProError("CuRobo trajectory start is malformed")
        if float(np.max(np.abs(positions[0] - measured_joints))) > 0.02:
            raise LiberoProError("CuRobo trajectory start joints no longer match")
        planner_start = planned.get("start_ee_pose_base_xyzw")
        if not isinstance(planner_start, list) or len(planner_start) != 7:
            raise LiberoProError("CuRobo omitted start FK cross-validation")
        fk = np.asarray(planner_start, dtype=np.float64)
        if not np.isfinite(fk).all() or not np.isclose(
            float(np.linalg.norm(fk[3:])), 1.0, atol=1e-5, rtol=0,
        ):
            raise LiberoProError("CuRobo start FK is malformed")
        measured_start = self._matrix_pose(
            np.linalg.inv(self._pose_matrix(motion_state["base_pose_world_xyzw"]))
            @ self._pose_matrix(motion_state["hand_pose_world_xyzw"])
        )
        position_error = float(np.linalg.norm(np.asarray(measured_start[:3]) - fk[:3]))
        orientation_error = float((
            Rotation.from_quat(measured_start[3:]).inv()
            * Rotation.from_quat(fk[3:])
        ).magnitude())
        if position_error > 0.005 or orientation_error > 0.05:
            raise LiberoProError("CuRobo Panda FK does not align with MuJoCo panda_hand")

    @staticmethod
    def _pose_matrix(pose_xyzw: Any) -> Any:
        import numpy as np
        from scipy.spatial.transform import Rotation

        pose = np.asarray(pose_xyzw, dtype=np.float64)
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise LiberoProError("rigid pose is malformed")
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
        transform[:3, 3] = pose[:3]
        return transform

    @staticmethod
    def _matrix_pose(transform: Any) -> list[float]:
        import numpy as np
        from scipy.spatial.transform import Rotation

        matrix = np.asarray(transform, dtype=np.float64)
        return matrix[:3, 3].tolist() + Rotation.from_matrix(
            matrix[:3, :3]
        ).as_quat().tolist()

    def _grip_target_planner_pose(
        self, motion_state: Mapping[str, Any], target_pose: list[float],
    ) -> tuple[list[float], list[float]]:
        """Compile a world grip-site target into the CuRobo hand-frame request."""

        import numpy as np

        world_from_base = self._pose_matrix(motion_state["base_pose_world_xyzw"])
        world_from_hand = self._pose_matrix(motion_state["hand_pose_world_xyzw"])
        world_from_grip = self._pose_matrix(
            motion_state["grip_site_pose_world_xyzw"]
        )
        hand_from_grip = np.linalg.inv(world_from_hand) @ world_from_grip
        world_from_target_grip = self._pose_matrix(target_pose)
        base_from_target_hand = (
            np.linalg.inv(world_from_base)
            @ world_from_target_grip
            @ np.linalg.inv(hand_from_grip)
        )
        return (
            self._matrix_pose(base_from_target_hand),
            self._matrix_pose(np.linalg.inv(world_from_base) @ world_from_hand),
        )

    def _execute_planned_motion(
        self, action: Mapping[str, Any], *,
        preplanned: tuple[Mapping[str, Any], Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        import numpy as np

        if self.motion_planner is None and preplanned is None:
            raise LiberoProError("CuRobo planner is not active")
        environment = self._require_environment()
        arguments = action["arguments"]
        before_position, before_orientation, before_gripper = self._current_state()
        motion_state = self._motion_state()
        request: dict[str, Any] = {
            "joint_names": motion_state["joint_names"],
            "joint_positions": motion_state["joint_positions"],
        }
        target_pose = None
        strategy = None
        if preplanned is not None:
            target_pose = preplanned[0].get("target_pose_world_xyzw")
        elif action["action"] == "move_free_space":
            target_pose = (
                arguments["target_pose"]["position_m"]
                + arguments["target_pose"]["orientation_xyzw"]
            )
            if not workspace_state(target_pose[:3], self.workspace_bounds_m)["inside"]:
                raise LiberoProError("move_free_space target is outside the workspace")
            target_base, _measured_start = self._grip_target_planner_pose(
                motion_state, target_pose,
            )
            request["target_pose_base_xyzw"] = target_base
        else:
            strategy = str(arguments["strategy"])
            goal = (
                self.home_joint_state
                if strategy == "home" else self.last_backtrack_joint_state
            )
            if goal is None:
                return self._planned_refusal_receipt(
                    action, before_position, before_orientation, before_gripper,
                    "recovery_state_unavailable",
                )
            request["goal_joint_positions"] = list(goal)
            if precision_force_stop_enabled(self.motion_profile):
                target_pose = (self.home_tcp_pose if strategy == "home"
                               else self.last_backtrack_tcp_pose)
                if target_pose is None:
                    return self._planned_refusal_receipt(
                        action, before_position, before_orientation, before_gripper,
                        "recovery_state_unavailable",
                    )
        if preplanned is not None:
            planned, planner_evidence = (
                dict(preplanned[0]), dict(preplanned[1]),
            )
        else:
            planned = self.motion_planner.plan(request)
            planner_evidence = self._persist_planner_evidence(planned)
        if not planned.get("reachable") or not isinstance(
            planned.get("trajectory"), Mapping
        ):
            return self._planned_refusal_receipt(
                action, before_position, before_orientation, before_gripper,
                _curobo_failure_reason(planned),
                planner=planner_evidence,
            )
        self._validate_planned_start(planned, motion_state)
        vector_environment = getattr(environment, "env", None)
        setter = getattr(vector_environment, "set_env_attr", None)
        if not callable(setter):
            raise LiberoProError("joint trajectory command transport is unavailable")
        trajectory = planned["trajectory"]
        execution_id = f"{self.round_index}:{self.action_sequence}:{action['action_id']}"
        try:
            setter("harness_joint_trajectory_command", {
                "execution_id": execution_id,
                "positions": trajectory["positions"],
                "joint_names": trajectory["joint_names"],
                "dt_s": trajectory["dt_s"],
                **({"target_pose_world_xyzw": target_pose}
                   if precision_force_stop_enabled(self.motion_profile) else {}),
                "stop_request_path": str(
                    (self.output_root / "control/stop.request").resolve()
                ),
            })
            execution = self._worker_attr("harness_joint_execution_result")
            if not isinstance(execution, Mapping) or execution.get("execution_id") != execution_id:
                raise LiberoProError("joint trajectory execution identity is unavailable")
        except Exception:
            # Controller restoration or observation formatting may fail after a
            # successful official step. Read the separate early-latched outcome
            # before allowing that infrastructure failure to propagate.
            try:
                outcome = self._worker_attr("harness_joint_execution_status")
            except Exception:
                outcome = None
            # A failed command transport can leave the preceding command's
            # outcome behind. Never count it again or attach it to this action.
            if isinstance(outcome, Mapping) and outcome.get("execution_id") == execution_id:
                if outcome["steps_used"] and action["action"] != "recover_motion":
                    self.last_backtrack_joint_state = list(motion_state["joint_positions"])
                    self.last_backtrack_tcp_pose = list(motion_state["grip_site_pose_world_xyzw"])
                self._latch_environment_outcome(
                    terminated=outcome["terminated"], truncated=outcome["truncated"],
                    steps=int(outcome["steps_used"]), action=action,
                    controller="JOINT_POSITION",
                )
            raise
        steps_used = int(execution["steps_used"])
        if steps_used and action["action"] != "recover_motion":
            self.last_backtrack_joint_state = list(motion_state["joint_positions"])
            self.last_backtrack_tcp_pose = list(motion_state["grip_site_pose_world_xyzw"])
        self._latch_environment_outcome(
            terminated=execution.get("terminated"), truncated=execution.get("truncated"),
            steps=steps_used, action=action, controller="JOINT_POSITION",
        )
        execution_evidence = self._persist_joint_execution_evidence(execution)
        issued_commands = execution.get("commands")
        if not isinstance(issued_commands, list) or len(issued_commands) != steps_used:
            raise LiberoProError("joint trajectory omitted issued controller commands")
        self.replay_recorder.record(
            action_sequence=self.action_sequence,
            action=action,
            controller="libero_joint_position",
            arrays={"commands": np.asarray(issued_commands).reshape((-1, 8))},
            physics_steps=steps_used,
            replayable=steps_used > 0,
            metadata={
                "controller_switch": "JOINT_POSITION_then_OSC_POSE",
                "recorded_plan_artifact": planner_evidence.get(
                    "canonical_artifact"
                ),
            },
        )
        self._refresh_joint_observation(steps_used)
        after_position, after_orientation, after_gripper = self._current_state()
        after_motion_state = self._motion_state()
        after_wrench = after_motion_state.get("external_wrench_clipped")
        force_magnitude_n, force_threshold_exceeded = self._force_telemetry(
            after_motion_state
        )
        contact_threshold = float(
            self.motion_profile["sensing"]["contact_force_threshold_n"]
        )
        after_contact = (
            None if after_wrench is None
            else float(np.linalg.norm(
                np.asarray(after_wrench[:3], dtype=np.float64)
            )) > contact_threshold
        )
        tracking_error = float(execution["tracking_error_l2_rad"])
        tracking_limit = float(
            self.motion_profile["global_motion"]["joint_tracking_error_limit_rad"]
        )
        stopped = execution.get("stop_reason") is not None
        endpoint_error = None
        endpoint_reached = True
        position_tolerance, orientation_tolerance = libero_global_endpoint_tolerances()
        if target_pose is not None:
            endpoint_error = {
                "position_m": float(np.linalg.norm(
                    np.asarray(target_pose[:3]) - after_position
                )),
                "orientation_rad": float(np.linalg.norm(_quaternion_error_rotvec(
                    np.asarray(target_pose[3:]), after_orientation,
                ))),
            }
            endpoint_reached = (
                endpoint_error["position_m"] <= position_tolerance
                and endpoint_error["orientation_rad"] <= orientation_tolerance
            )
        healthy = tracking_error <= tracking_limit and endpoint_reached and not stopped
        endpoint_failure = (
            "endpoint_residual_exceeded" if not endpoint_reached
            else "joint_tracking_error"
        )
        settle_record = execution.get("terminal_settle")
        settle_record = settle_record if isinstance(settle_record, Mapping) else None
        # The global route is judged by the CuRobo transport bound and its last
        # millimetres are owned by the terminal settle. Compile that verdict
        # once, here, so no projection layer has to guess which bound applied.
        endpoint_contract = libero_endpoint_contract(
            route=LIBERO_ENDPOINT_ROUTE_GLOBAL,
            position_tolerance_m=position_tolerance,
            orientation_tolerance_rad=orientation_tolerance,
            position_residual_m=(
                None if endpoint_error is None else endpoint_error["position_m"]
            ),
            orientation_residual_rad=(
                None if endpoint_error is None
                else endpoint_error["orientation_rad"]
            ),
            within_tolerance=(None if endpoint_error is None else endpoint_reached),
            convergence_budget="terminal_settle",
            execution_phase=execution.get("execution_phase", "unknown"),
            stop_reason=execution.get("stop_reason"),
            waypoints_reached=execution.get("waypoints_reached"),
            waypoints_total=execution.get("waypoints_total"),
            ticks_used=(None if settle_record is None else settle_record.get("ticks")),
            ticks_budget=(
                None if settle_record is None else settle_record.get("max_ticks")
            ),
            budget_exhausted=(
                False if settle_record is None
                else bool(settle_record.get("exhausted"))
            ),
        )
        receipt = {
            "schema": HYBRID_RECEIPT_SCHEMA,
            "action_id": action["action_id"],
            "action": action["action"],
            "arguments": action["arguments"],
            "consumed": True,
            "state_changed": steps_used > 0,
            "motion_status": (
                "reached" if healthy else "stopped" if stopped else "not_reached"
            ),
            **({"commanded_pose_xyz_xyzw": list(target_pose)}
               if precision_force_stop_enabled(self.motion_profile) and target_pose is not None else {}),
            "planner": planner_evidence,
            "execution": execution_evidence,
            "endpoint_contract": endpoint_contract,
            "feedback": {
                "endpoint_contract": endpoint_contract,
                "measured_pose_xyz_xyzw": after_position.tolist() + after_orientation.tolist(),
                "joint_state": {
                    "names": after_motion_state["joint_names"],
                    "positions": execution["joint_positions"],
                },
                "tracking_error_l2_rad": tracking_error,
                "endpoint_error": endpoint_error,
                "endpoint_tolerance": {
                    "position_m": position_tolerance,
                    "orientation_rad": orientation_tolerance,
                } if target_pose is not None else None,
                "endpoint_comparison": "<=",
                "contact_estimated": after_contact,
                "collision": None,
                "reflex": False,
                "external_wrench_clipped": after_wrench,
                "external_force_magnitude_n": force_magnitude_n,
                "force_threshold_exceeded": force_threshold_exceeded,
                **self._force_direction_feedback(after_motion_state),
                "gripper_aperture": after_gripper,
                "motion_state": "stopped" if stopped else "idle",
                "reachability": True,
                "recovery_reason": strategy,
                "stop_reason": execution.get("stop_reason"),
            },
            "before": {"right": {"pose_xyz_xyzw": before_position.tolist() + before_orientation.tolist(), "gripper": before_gripper}},
            "after": {"right": {"pose_xyz_xyzw": after_position.tolist() + after_orientation.tolist(), "gripper": after_gripper}},
            "steps_used": steps_used,
            "sim_steps_total": self.sim_steps,
            "control_health": {
                "schema": CONTROL_HEALTH_SCHEMA,
                "status": "healthy" if healthy else "degraded",
                "codes": (
                    [] if healthy else [
                        str(execution.get("stop_reason") or endpoint_failure)
                    ]
                ),
            },
            "robustness_event": self._persist_new_robustness_event(),
            "finished_unix_s": time.time(),
        }
        self.replay_recorder.update(
            self.action_sequence,
            execution_status=str(receipt["motion_status"]),
            execution_fault=(
                str(execution.get("stop_reason"))
                if execution.get("stop_reason") is not None else None
            ),
        )
        return self._persist_receipt(receipt)

    def _planned_refusal_receipt(
        self, action: Mapping[str, Any], position: Any, orientation: Any,
        gripper: float, reason: str, *,
        planner: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        recorder = getattr(self, "replay_recorder", None)
        if recorder is not None:
            recorder.record(
                action_sequence=self.action_sequence,
                action=action,
                controller="libero_joint_position",
                arrays={},
                physics_steps=0,
                replayable=False,
                metadata={"reason": reason},
            )
        receipt = {
            "schema": HYBRID_RECEIPT_SCHEMA,
            "action_id": action["action_id"], "action": action["action"],
            "arguments": action["arguments"], "consumed": True,
            "state_changed": False, "motion_status": "refused",
            "planner": None if planner is None else dict(planner),
            "feedback": {
                "measured_pose_xyz_xyzw": position.tolist() + orientation.tolist(),
                "joint_state": None, "tracking_error_l2_rad": None,
                "contact_estimated": None, "collision": None, "reflex": False,
                "external_wrench_clipped": None, "gripper_aperture": gripper,
                "external_force_magnitude_n": None,
                "force_threshold_exceeded": False,
                "motion_state": "refused",
                "reachability": False, "planning_failure_reason": reason,
            },
            "before": {"right": {"pose_xyz_xyzw": position.tolist() + orientation.tolist(), "gripper": gripper}},
            "after": {"right": {"pose_xyz_xyzw": position.tolist() + orientation.tolist(), "gripper": gripper}},
            "steps_used": 0, "sim_steps_total": self.sim_steps,
            "control_health": {"schema": CONTROL_HEALTH_SCHEMA, "status": "degraded", "codes": [reason]},
            "robustness_event": None, "finished_unix_s": time.time(),
        }
        if recorder is not None:
            recorder.update(
                self.action_sequence, execution_status="refused",
                execution_fault=reason,
            )
        return self._persist_receipt(receipt)
