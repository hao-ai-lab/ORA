"""Control health, evidence and the receipt of one OSC local action.

A mixin of ``session.LiberoProSession``; it holds methods only, no dataclass
fields. ``local_motion`` runs the bounded loop and hands its ``_LocalMotion``
state here once the reached state is measured.
"""

from __future__ import annotations

import time
from typing import Any, Mapping

from ora.contracts import (
    MOTION_ACTION_SCHEMA,
    POSE_ORIENTATION_TOLERANCE_RAD,
    POSE_POSITION_TOLERANCE_M,
)
from ora.environment.backends.libero_pro.motion_settle import gripper_outcome
from ora.environment.controllers.hybrid_motion import (
    LIBERO_ENDPOINT_ROUTE_LOCAL,
    aggregate_motion_contact,
    libero_endpoint_contract,
    libero_local_endpoint_tolerances,
)
from ora.contracts.workspace import is_monotonic_workspace_recovery
from ora.environment.controllers.libero_osc import (
    GRIPPER_TOLERANCE_NORMALIZED,
    CONTROL_HEALTH_NUMERIC_TOLERANCE,
    GROSS_ORIENTATION_GROWTH_RAD,
)

from .common import (
    RECEIPT_SCHEMA,
    HYBRID_RECEIPT_SCHEMA,
    CONTROL_HEALTH_SCHEMA,
    TRACKING_EVIDENCE_SCHEMA,
    LiberoProError,
    _atomic_json,
    _file_hash,
)
from .local_motion import _LocalMotion


def _local_motion_codes(run: _LocalMotion) -> list[str]:
    """Every control-health code the measured local action earned, in order."""

    codes: list[str] = []
    if run.force_stop_reason is not None:
        codes.append(run.force_stop_reason)
    if run.steps_used >= run.loop_limit and (
        run.target_error_after_m >= POSE_POSITION_TOLERANCE_M
        or run.target_orientation_error_after_rad
        >= POSE_ORIENTATION_TOLERANCE_RAD
    ):
        codes.append("servo_target_not_reached")
    if run.terminal_tracking_reason is not None:
        codes.extend([
            "tracking_terminal_fault",
            f"tracking_{run.terminal_tracking_reason}",
        ])
    if (
        run.workspace_after["max_violation_m"]
        > CONTROL_HEALTH_NUMERIC_TOLERANCE
    ):
        codes.append("eef_outside_action_workspace")
    if (
        run.target_error_after_m
        > run.target_error_before_m + 0.05 + CONTROL_HEALTH_NUMERIC_TOLERANCE
    ):
        codes.append("gross_target_divergence")
    if (
        run.target_orientation_error_after_rad
        > run.target_orientation_error_before_rad
        + GROSS_ORIENTATION_GROWTH_RAD
        + CONTROL_HEALTH_NUMERIC_TOLERANCE
    ):
        codes.append("gross_orientation_divergence")
    if (
        run.workspace_after["max_violation_m"]
        > 0.05 + CONTROL_HEALTH_NUMERIC_TOLERANCE
    ):
        codes.append("gross_workspace_overrun")
    if run.action["action"] == "set_gripper" and (
        run.pose_translation > 0.05 + CONTROL_HEALTH_NUMERIC_TOLERANCE
        or run.pose_rotation > 0.25 + CONTROL_HEALTH_NUMERIC_TOLERANCE
    ):
        codes.append("gripper_action_moved_eef")
    if run.pose_action and (
        run.gripper_hold_directive_after != run.gripper_hold_directive
    ):
        codes.append("gripper_directive_changed_during_pose_motion")
    return codes


def _local_motion_realization(run: _LocalMotion) -> dict[str, Any]:
    compiled_pose = run.compiled_pose
    return {
        "requested_translation_world_m": (
            compiled_pose["translation_world_m"] if compiled_pose else [0.0, 0.0, 0.0]
        ),
        "requested_translation_distance_m": (
            compiled_pose["translation_distance_m"] if compiled_pose else 0.0
        ),
        "realized_translation_world_m": (
            run.reached_position - run.before_position
        ).tolist(),
        "realized_translation_distance_m": run.pose_translation,
        "remaining_translation_error_m": run.target_error_after_m,
        "requested_rotation_angle_rad": (
            compiled_pose["rotation_angle_rad"] if compiled_pose else 0.0
        ),
        "requested_rotation_angle_deg": (
            compiled_pose["rotation_angle_deg"] if compiled_pose else 0.0
        ),
        "realized_rotation_angle_rad": run.pose_rotation,
        "remaining_orientation_error_rad": (
            run.target_orientation_error_after_rad
        ),
        "measured_aperture_delta": run.reached_gripper - run.before_gripper,
    }


def _local_gripper_evidence(run: _LocalMotion) -> dict[str, Any]:
    return {
        "measured_before": run.before_gripper,
        "measured_after": run.reached_gripper,
        "measured_delta": run.reached_gripper - run.before_gripper,
        "actuator_target_before": run.gripper_hold_directive,
        "actuator_target_after": run.gripper_hold_directive_after,
        "pose_hold_semantics": (
            "actuator_target" if run.pose_action else None
        ),
        "closure_delta": run.before_gripper - run.reached_gripper,
        "closure_residual": (
            abs(run.target_gripper - run.reached_gripper)
            if run.action["action"] == "set_gripper" else None
        ),
        "grasp_success": None,
    }


def _local_command_invariant(run: _LocalMotion) -> dict[str, Any]:
    return (
        {
            "schema": "ora.command-invariant.v2",
            "controlled_dimension": "pose",
            "held_dimension": "gripper_actuator_target",
            "directive_before": run.gripper_hold_directive,
            "directive_after": run.gripper_hold_directive_after,
            "action_dimension_command": run.gripper_command,
            "verified": (
                run.gripper_hold_directive is not None
                and run.gripper_hold_directive_after
                == run.gripper_hold_directive
                and run.gripper_command == 0.0
            ),
        }
        if run.pose_action
        else {
            "schema": "ora.command-invariant.v2",
            "controlled_dimension": "gripper",
            "held_dimension": "eef_pose_target",
            "target_from_measured_before": (
                run.before_position.tolist() + run.before_orientation.tolist()
            ),
            "verified": True,
        }
    )


def _local_preserved_dimension(run: _LocalMotion) -> dict[str, Any]:
    return (
        {
            "field": "gripper_actuator_target",
            "before": run.gripper_hold_directive,
            "after": run.gripper_hold_directive_after,
            "verified": (
                run.gripper_hold_directive is not None
                and run.gripper_hold_directive_after == run.gripper_hold_directive
            ),
            "measured_aperture_before": run.before_gripper,
            "measured_aperture_after": run.reached_gripper,
            "measured_aperture_delta": run.reached_gripper - run.before_gripper,
        }
        if run.pose_action
        else {
            "field": "eef_pose",
            "translation_residual_m": run.pose_translation,
            "orientation_residual_rad": run.pose_rotation,
            "verified": (
                run.pose_translation
                <= 0.005 + CONTROL_HEALTH_NUMERIC_TOLERANCE
                and run.pose_rotation
                <= 0.05 + CONTROL_HEALTH_NUMERIC_TOLERANCE
            ),
        }
    )


class LiberoLocalMotionReportMixin:
    """Control health, evidence and the receipt of one OSC local action."""

    def _report_local_motion(self, run: _LocalMotion) -> dict[str, Any]:
        """Judge the measured action, persist its evidence and its receipt."""

        run.codes = _local_motion_codes(run)
        self._judge_local_motion(run)
        control_health = self._local_control_health(run)
        tracking_path = self._persist_local_tracking(run, control_health)
        robustness_event = self._persist_new_robustness_event()
        sensor_feedback = self._local_sensor_feedback(run, control_health)
        receipt = self._local_motion_receipt(
            run, control_health=control_health, tracking_path=tracking_path,
            sensor_feedback=sensor_feedback, robustness_event=robustness_event,
        )
        if run.health_status == "fault":
            self.control_fault_reason = "libero_control_health_fault"
        self.replay_recorder.record(
            action_sequence=self.action_sequence,
            action=run.action,
            controller="libero_osc_pose",
            arrays={"commands": run.issued_commands},
            physics_steps=len(run.issued_commands),
            replayable=bool(run.issued_commands),
            metadata={
                "controller_action_contract_sha256": (
                    self.controller_action_contract["contract_sha256"]
                ),
            },
        )
        self.replay_recorder.update(
            self.action_sequence,
            execution_status=run.motion_status,
            execution_fault=run.terminal_tracking_reason,
        )
        return self._persist_receipt(receipt)

    def _judge_local_motion(self, run: _LocalMotion) -> None:
        """Health status, the local endpoint verdict and the motion status."""

        fault_codes = {
            "tracking_terminal_fault", "gross_workspace_overrun",
            "gripper_directive_changed_during_pose_motion",
        }
        run.health_status = (
            "fault" if fault_codes.intersection(run.codes)
            else "degraded" if run.codes else "healthy"
        )
        local_position_tolerance, local_orientation_tolerance = (
            libero_local_endpoint_tolerances()
        )
        pose_target_reached = (
            run.target_error_after_m < local_position_tolerance
            and run.target_orientation_error_after_rad
            < local_orientation_tolerance
        )
        # The local route is judged by the tighter shared action contract, and
        # its convergence budget is the OSC iteration limit. Compile the same
        # verdict record the global route emits so the two routes are readable
        # against each other.
        run.local_endpoint_contract = libero_endpoint_contract(
            route=LIBERO_ENDPOINT_ROUTE_LOCAL,
            position_tolerance_m=local_position_tolerance,
            orientation_tolerance_rad=local_orientation_tolerance,
            position_residual_m=run.target_error_after_m,
            orientation_residual_rad=run.target_orientation_error_after_rad,
            within_tolerance=pose_target_reached,
            convergence_budget="osc_servo_iterations",
            execution_phase="local_control",
            stop_reason=(
                run.terminal_tracking_reason or run.force_stop_reason or run.gripper_stall_reason
                or ("environment_truncated" if self.truncated else None)
                or ("environment_terminated" if self.terminated else None)
                or ("servo_iteration_budget_exhausted" if run.steps_used >= run.loop_limit else None)
            ),
            ticks_used=int(run.steps_used),
            ticks_budget=int(run.loop_limit),
            budget_exhausted=run.steps_used >= run.loop_limit,
        )
        run.gripper_target_reached = abs(run.target_gripper - run.reached_gripper) < GRIPPER_TOLERANCE_NORMALIZED
        run.motion_status = (
            "fault"
            if run.health_status == "fault"
            else "stopped" if run.force_stop_reason is not None
            or run.gripper_stall_reason is not None
            else "reached"
            if (
                pose_target_reached
                and (
                    run.pose_action
                    or run.gripper_target_reached
                )
            )
            else "not_reached"
        )

    def _local_control_health(self, run: _LocalMotion) -> dict[str, Any]:
        return {
            "schema": CONTROL_HEALTH_SCHEMA,
            "status": run.health_status,
            "codes": run.codes,
            "controller_action_contract_sha256": self.controller_action_contract[
                "contract_sha256"
            ],
            "target_error_before_m": run.target_error_before_m,
            "target_error_after_m": run.target_error_after_m,
            "target_orientation_error_before_rad": (
                run.target_orientation_error_before_rad
            ),
            "target_orientation_error_after_rad": (
                run.target_orientation_error_after_rad
            ),
            "tracking_guard": {
                "outcome": (
                    "terminal_fault" if run.terminal_tracking_reason is not None
                    else "not_triggered"
                ),
                "reason": run.terminal_tracking_reason,
                "first_threshold_tick": run.tracking_stopped_tick,
                "stopped_tick": run.tracking_stopped_tick,
                "telemetry_thresholds_crossed": sorted(run.tracking_advisories),
                "telemetry_first_ticks": {
                    reason: run.first_threshold_ticks[reason]
                    for reason in sorted(run.tracking_advisories)
                },
                "thresholds": dict(
                    self.controller_action_contract["tracking_envelope"]
                ),
            },
            "force_telemetry": {
                "threshold_n": float(
                    self.motion_profile["sensing"][
                        "force_telemetry_threshold_n"
                    ]
                ) if self.action_codec == "motion_action_v3" else None,
                "exceeded": run.force_threshold_exceeded,
                "first_threshold_tick": run.force_threshold_first_tick,
                "stop_authority": run.force_stop_enabled and run.active_interaction == "precision",
                **({"stop_reason": run.force_stop_reason, "stopped_tick": run.force_stopped_tick}
                   if run.force_stop_enabled else {}),
            },
            "workspace_before": run.workspace_before,
            "workspace_after": run.workspace_after,
            "monotonic_workspace_recovery": (
                not run.workspace_before["inside"]
                and is_monotonic_workspace_recovery(
                    run.before_position.tolist(), run.target.tolist(),
                    self.workspace_bounds_m,
                )
            ),
        }

    def _persist_local_tracking(
        self, run: _LocalMotion, control_health: Mapping[str, Any],
    ) -> Any:
        """Write the control-only per-tick tracking evidence; return its path."""

        tracking_record = {
            "schema": TRACKING_EVIDENCE_SCHEMA,
            "policy_visibility": "control_only",
            "action_id": run.action["action_id"],
            "action": run.action["action"],
            "controller_action_contract_sha256": self.controller_action_contract[
                "contract_sha256"
            ],
            "compiled_pose_sha256": (
                run.compiled_pose["compiled_pose_sha256"] if run.compiled_pose else None
            ),
            "samples": run.tracking_samples,
            "guard": control_health["tracking_guard"],
        }
        tracking_path = (
            self.output_root / "tracking"
            / f"action-{self.action_sequence:04d}.json"
        )
        if tracking_path.exists():
            raise LiberoProError(
                f"refusing to overwrite tracking evidence: {tracking_path}"
            )
        _atomic_json(tracking_path, tracking_record)
        return tracking_path

    def _local_sensor_feedback(
        self, run: _LocalMotion, control_health: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Sensor-equivalent feedback for a motion-v1 action; ``None`` otherwise."""

        if run.action["schema"] != MOTION_ACTION_SCHEMA:
            return None
        measured_motion = self._motion_state()
        wrench = measured_motion.get("external_wrench_clipped")
        force_magnitude_n, final_force_threshold_exceeded = (
            self._force_telemetry(measured_motion)
        )
        run.force_threshold_exceeded = (
            run.force_threshold_exceeded or final_force_threshold_exceeded
        )
        if (
            final_force_threshold_exceeded
            and run.force_threshold_first_tick is None
        ):
            run.force_threshold_first_tick = run.steps_used
        control_health["force_telemetry"]["exceeded"] = (
            run.force_threshold_exceeded
        )
        control_health["force_telemetry"]["first_threshold_tick"] = (
            run.force_threshold_first_tick
        )
        contact = aggregate_motion_contact(
            external_force_xyz=(None if wrench is None else wrench[:3]),
            force_threshold_n=float(
                self.motion_profile["sensing"]["contact_force_threshold_n"]
            ),
            closing=(
                run.action["action"] == "set_gripper" and run.target_gripper == 0.0
            ),
            measured_aperture_normalized=run.reached_gripper,
            closure_contact_min_normalized=float(
                self.motion_profile["gripper"][
                    "closure_contact_min_normalized"
                ]
            ),
        )
        return {
            "measured_pose_xyz_xyzw": run.reached_position.tolist()
            + run.reached_orientation.tolist(),
            "joint_state": {
                "names": measured_motion["joint_names"],
                "positions": measured_motion["joint_positions"],
                "velocities": measured_motion["joint_velocities"],
            },
            "tracking_error_m": run.target_error_after_m,
            "tracking_error_rad": run.target_orientation_error_after_rad,
            "endpoint_error": {"position_m": run.target_error_after_m,
                               "orientation_rad": run.target_orientation_error_after_rad},
            "endpoint_tolerance": {"position_m": POSE_POSITION_TOLERANCE_M,
                                   "orientation_rad": POSE_ORIENTATION_TOLERANCE_RAD},
            "endpoint_comparison": "<",
            "contact_estimated": contact,
            "collision": None,
            "reflex": False,
            "external_wrench_clipped": wrench,
            "external_force_magnitude_n": force_magnitude_n,
            "force_threshold_exceeded": run.force_threshold_exceeded,
            **self._force_direction_feedback(measured_motion),
            **({"stop_reason": run.force_stop_reason or run.gripper_stall_reason}
               if run.force_stop_enabled or run.gripper_stall_reason else {}),
            "gripper_aperture": run.reached_gripper,
            "motion_state": (
                "stopped" if run.terminal_tracking_reason is not None or run.force_stop_reason is not None
                or run.gripper_stall_reason is not None else "idle"
            ),
            "reachability": None,
        }

    def _local_tracking_evidence(self, run: _LocalMotion, tracking_path: Any) -> dict[str, Any]:
        return {
            "path": tracking_path.relative_to(self.output_root).as_posix(),
            "sha256": _file_hash(tracking_path),
            "sample_count": len(run.tracking_samples),
            "first_divergence_tick": (
                min(
                    run.first_threshold_ticks[reason]
                    for reason in run.tracking_advisories
                )
                if run.tracking_advisories else run.tracking_stopped_tick
            ),
            "stopped_tick": run.tracking_stopped_tick,
            "stop_reason": run.terminal_tracking_reason,
        }

    def _local_motion_receipt(
        self, run: _LocalMotion, *, control_health: dict[str, Any], tracking_path: Any,
        sensor_feedback: dict[str, Any] | None, robustness_event: Any,
    ) -> dict[str, Any]:
        action = run.action
        compiled_pose = run.compiled_pose
        return {
            "schema": (
                HYBRID_RECEIPT_SCHEMA
                if action["schema"] == MOTION_ACTION_SCHEMA else RECEIPT_SCHEMA
            ),
            "action_id": action["action_id"],
            "action": action["action"],
            "arguments": action["arguments"],
            "consumed": True,
            "state_changed": run.steps_used > 0,
            "motion_status": run.motion_status,
            **({"gripper_outcome": gripper_outcome(
                state=("reached_target" if run.gripper_target_reached
                       else "stalled_before_target" if run.gripper_stall_reason
                       else "moving_at_budget" if run.steps_used >= run.loop_limit
                       else "interrupted"),
                target=run.target_gripper, measured=run.reached_gripper,
                ticks_used=run.steps_used, ticks_budget=run.loop_limit,
                maximum_aperture_m=self.motion_profile["gripper"]["maximum_aperture_m"],
            )} if action["action"] == "set_gripper" and self.motion_profile else {}),
            "endpoint_contract": run.local_endpoint_contract,
            "commanded_pose_xyz_xyzw": run.target.tolist()
            + run.target_orientation.tolist(),
            "reached_pose_xyz_xyzw": run.reached_position.tolist()
            + run.reached_orientation.tolist(),
            "reached_gripper": run.reached_gripper,
            "before_pose_xyz_xyzw": run.before_position.tolist()
            + run.before_orientation.tolist(),
            "pose_contract_sha256": (
                compiled_pose["pose_contract_sha256"] if compiled_pose else None
            ),
            "compiled_pose_sha256": (
                compiled_pose["compiled_pose_sha256"] if compiled_pose else None
            ),
            "motion_realization": _local_motion_realization(run),
            "tracking_evidence": self._local_tracking_evidence(run, tracking_path),
            "gripper_evidence": _local_gripper_evidence(run),
            "command_invariant": _local_command_invariant(run),
            "preserved_dimension": _local_preserved_dimension(run),
            "before": {
                "right": {
                    "pose_xyz_xyzw": run.before_position.tolist()
                    + run.before_orientation.tolist(),
                    "gripper": run.before_gripper,
                }
            },
            "after": {
                "right": {
                    "pose_xyz_xyzw": run.reached_position.tolist()
                    + run.reached_orientation.tolist(),
                    "gripper": run.reached_gripper,
                }
            },
            "steps_used": run.steps_used,
            "sim_steps_total": self.sim_steps,
            "control_health": control_health,
            "feedback": sensor_feedback,
            "robustness_event": robustness_event,
            "started_unix_s": run.started,
            "finished_unix_s": time.time(),
        }
