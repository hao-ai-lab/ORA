"""Action dispatch and the bounded OSC local executor on the right arm.

A mixin of ``session.LiberoProSession``; it holds methods only, no dataclass
fields. ``_execute_without_capture`` routes every action to its executor; a
``move_pose``, ``servo_delta`` or ``set_gripper`` runs here as one bounded OSC
loop over a ``_LocalMotion`` working state, and ``local_motion_report`` turns
the loop's measurements into control health, evidence and the receipt.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

from ora.contracts import (
    MOTION_ACTION_V3_SCHEMA,
    POSE_ORIENTATION_TOLERANCE_RAD,
    POSE_POSITION_TOLERANCE_M,
    compile_absolute_pose,
)
from ora.environment.backends.libero_pro.motion_settle import (
    GRIPPER_STALL_REASON, gripper_stalled,
)
from ora.environment.controllers.hybrid_motion import (
    precision_force_stop_enabled,
)
from ora.environment.controllers.force_feedback import (
    precision_force_stop_reason, public_force_direction,
)
from ora.contracts.action import validate_runtime_action
from ora.contracts.workspace import (
    is_monotonic_workspace_recovery,
    workspace_state,
)
from ora.environment.controllers.libero_osc import (
    GRIPPER_TOLERANCE_NORMALIZED,
    CONTROL_HEALTH_NUMERIC_TOLERANCE,
    GROSS_SAFETY_ENVELOPE_OVERRUN_M,
    GROSS_SINGLE_TICK_TRANSLATION_M,
    TRACKING_CONSECUTIVE_TICKS,
    TRACKING_ORIENTATION_GROWTH_TELEMETRY_RAD,
    TRACKING_POSITION_GROWTH_TELEMETRY_M,
    TRACKING_TRAVEL_OVERSHOOT_TELEMETRY_M,
    inverse_controller_scale as _inverse_controller_scale,
)

from .common import LiberoProError, _pose_tracking_sample, _quaternion_error_rotvec


@dataclass
class _LocalMotion:
    """The working state of one OSC local action, from its target to its receipt."""

    action: Mapping[str, Any]
    before_position: Any
    before_orientation: Any
    before_gripper: float
    pose_action: bool
    loop_limit: int
    workspace_recovery_action: bool
    target: Any
    target_orientation: Any
    target_gripper: float
    gripper_hold_directive: float | None
    gripper_command: float
    started: float
    target_error_before_m: float
    target_orientation_error_before_rad: float
    compiled_pose: dict[str, Any] | None
    force_stop_enabled: bool
    active_interaction: Any
    previous_position: Any
    last_position: Any
    last_orientation: Any
    last_gripper: float
    steps_used: int = 0
    tracking_samples: list[dict[str, Any]] = field(default_factory=list)
    issued_commands: list[list[float]] = field(default_factory=list)
    tracking_streaks: dict[str, int] = field(default_factory=lambda: {
        "position_error_growth": 0,
        "travel_overshoot": 0,
        "orientation_error_growth": 0,
    })
    first_threshold_ticks: dict[str, int] = field(default_factory=dict)
    tracking_advisories: set[str] = field(default_factory=set)
    force_threshold_exceeded: bool = False
    force_threshold_first_tick: int | None = None
    force_stop_reason: str | None = None
    force_stopped_tick: int | None = None
    terminal_tracking_reason: str | None = None
    tracking_stopped_tick: int | None = None
    gripper_trace: list[float] = field(default_factory=list)
    gripper_stall_reason: str | None = None
    # Measured once the loop ends.
    reached_position: Any = None
    reached_orientation: Any = None
    reached_gripper: Any = None
    gripper_hold_directive_after: float | None = None
    pose_translation: float | None = None
    pose_rotation: float | None = None
    gripper_residual: Any = None  # computed as before; not reported
    target_error_after_m: float | None = None
    target_orientation_error_after_rad: float | None = None
    workspace_before: dict[str, Any] | None = None
    workspace_after: dict[str, Any] | None = None
    # Judged from those measurements (``local_motion_report``).
    codes: list[str] = field(default_factory=list)
    health_status: str | None = None
    local_endpoint_contract: dict[str, Any] | None = None
    gripper_target_reached: bool | None = None
    motion_status: str | None = None


@dataclass(frozen=True)
class _LocalTick:
    """One control tick's measurement before any command is issued."""

    position: Any
    orientation: Any
    gripper: float
    actuator_target: float | None
    delta: Any
    rotvec: Any
    sample: dict[str, Any]
    workspace: dict[str, Any]


def _record_tracking_telemetry(run: _LocalMotion, sample: Mapping[str, Any]) -> None:
    conditions = {
        "position_error_growth": (
            sample["target_position_error_growth_m"]
            > TRACKING_POSITION_GROWTH_TELEMETRY_M
            + CONTROL_HEALTH_NUMERIC_TOLERANCE
        ),
        "travel_overshoot": (
            sample["travel_overshoot_m"]
            > TRACKING_TRAVEL_OVERSHOOT_TELEMETRY_M
            + CONTROL_HEALTH_NUMERIC_TOLERANCE
        ),
        "orientation_error_growth": (
            sample["target_orientation_error_growth_rad"]
            > TRACKING_ORIENTATION_GROWTH_TELEMETRY_RAD
            + CONTROL_HEALTH_NUMERIC_TOLERANCE
        ),
    }
    for reason, active in conditions.items():
        run.tracking_streaks[reason] = (
            run.tracking_streaks[reason] + 1 if active else 0
        )
        if active and reason not in run.first_threshold_ticks:
            run.first_threshold_ticks[reason] = int(sample["tick"])
        if run.tracking_streaks[reason] >= TRACKING_CONSECUTIVE_TICKS:
            run.tracking_advisories.add(reason)


class LiberoLocalMotionMixin:
    """Action dispatch and the bounded OSC local executor on the right arm."""

    def _execute_without_capture(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        environment = self._require_environment()
        if self.is_terminal:
            raise LiberoProError("episode is already terminal")
        action = validate_runtime_action(payload)
        if action["schema"] == MOTION_ACTION_V3_SCHEMA:
            return self._execute_sparse_motion_v3(action)
        arguments = action["arguments"]
        if arguments["arm"] != "right":
            raise LiberoProError("LIBERO-Pro exposes only the right single arm")
        if action["action"] in {"move_free_space", "recover_motion"}:
            return self._execute_planned_motion(action)
        return self._execute_local_motion(action, environment)

    def _execute_local_motion(
        self, action: Mapping[str, Any], environment: Any,
    ) -> dict[str, Any]:
        """Run one move_pose, servo_delta or set_gripper through the OSC loop."""

        run = self._prepare_local_motion(action)
        for _ in range(run.loop_limit):
            tick = self._sample_local_tick(run)
            if tick is None or self._local_tick_stops(run, tick):
                break
            if run.pose_action:
                _record_tracking_telemetry(run, tick.sample)
            if self._local_tick_settled(run, tick):
                break
            if not self._step_local_command(run, environment, tick):
                break
        self._measure_local_motion_end(run)
        self._measure_local_motion_residuals(run)
        return self._report_local_motion(run)

    def _prepare_local_motion(self, action: Mapping[str, Any]) -> _LocalMotion:
        """Bind the target, the held dimension and the loop budget."""

        import numpy as np

        arguments = action["arguments"]
        before_position, before_orientation, before_gripper = self._current_state()
        pose_action = action["action"] in {"move_pose", "servo_delta"}
        loop_limit = (
            int(arguments["max_iterations"])
            if action["action"] == "servo_delta" else self.servo_max_steps
        )
        workspace_recovery_action = False
        if pose_action:
            pose_arguments = (
                arguments["target_pose"]
                if action["action"] == "servo_delta" else arguments
            )
            target = np.asarray(pose_arguments["position_m"], dtype=np.float64)
            target_workspace = workspace_state(target.tolist(), self.workspace_bounds_m)
            if (
                not target_workspace["inside"]
                and not is_monotonic_workspace_recovery(
                    before_position.tolist(), target.tolist(), self.workspace_bounds_m,
                )
            ):
                raise LiberoProError(
                    "move_pose target is outside the action workspace and is not "
                    "a monotonic recovery"
                )
            workspace_recovery_action = is_monotonic_workspace_recovery(
                before_position.tolist(), target.tolist(), self.workspace_bounds_m,
            )
            target_orientation = np.asarray(
                pose_arguments["orientation_xyzw"], dtype=np.float64
            )
            target_gripper = before_gripper
            gripper_hold_directive = self._gripper_hold_directive()
        else:
            target = before_position.copy()
            target_orientation = before_orientation.copy()
            target_gripper = float(arguments["gripper"])
            gripper_hold_directive = None
        gripper_command = (
            0.0 if pose_action
            else 1.0 - 2.0 * target_gripper
        )
        started = time.time()
        target_error_before_m = float(np.linalg.norm(target - before_position))
        target_orientation_error_before_rad = float(np.linalg.norm(
            _quaternion_error_rotvec(target_orientation, before_orientation)
        ))
        compiled_pose = (
            compile_absolute_pose(
                current_pose_xyz_xyzw=(
                    before_position.tolist() + before_orientation.tolist()
                ),
                target_pose_xyz_xyzw=(
                    target.tolist() + target_orientation.tolist()
                ),
                controlled_point="robosuite_grip_site",
            )
            if pose_action else None
        )
        return _LocalMotion(
            action=action,
            before_position=before_position,
            before_orientation=before_orientation,
            before_gripper=before_gripper,
            pose_action=pose_action,
            loop_limit=loop_limit,
            workspace_recovery_action=workspace_recovery_action,
            target=target,
            target_orientation=target_orientation,
            target_gripper=target_gripper,
            gripper_hold_directive=gripper_hold_directive,
            gripper_command=gripper_command,
            started=started,
            target_error_before_m=target_error_before_m,
            target_orientation_error_before_rad=target_orientation_error_before_rad,
            compiled_pose=compiled_pose,
            force_stop_enabled=precision_force_stop_enabled(self.motion_profile),
            active_interaction=getattr(self, "_active_motion_interaction", None),
            previous_position=before_position.copy(),
            last_position=before_position.copy(),
            last_orientation=before_orientation.copy(),
            last_gripper=before_gripper,
        )

    def _sample_local_tick(self, run: _LocalMotion) -> _LocalTick | None:
        """Measure the plant before a tick; ``None`` when its state is unreadable."""

        try:
            position, orientation, gripper = self._current_state()
            actuator_target = (
                self._gripper_hold_directive()
                if run.pose_action else None
            )
        except (LiberoProError, ValueError, FloatingPointError):
            run.terminal_tracking_reason = "non_finite_or_unreadable_state"
            run.tracking_stopped_tick = run.steps_used
            return None
        run.last_position = position.copy()
        run.last_orientation = orientation.copy()
        run.last_gripper = gripper
        run.gripper_trace.append(float(gripper))
        delta = run.target - position
        rotvec = _quaternion_error_rotvec(run.target_orientation, orientation)
        sample = _pose_tracking_sample(
            tick=run.steps_used,
            before_position=run.before_position,
            before_orientation=run.before_orientation,
            target_position=run.target,
            target_orientation=run.target_orientation,
            position=position,
            orientation=orientation,
            previous_position=run.previous_position,
            measured_aperture=gripper,
            actuator_target=actuator_target,
            initial_position_error_m=run.target_error_before_m,
            initial_orientation_error_rad=run.target_orientation_error_before_rad,
        )
        run.tracking_samples.append(sample)
        current_workspace = workspace_state(
            position.tolist(), self.workspace_bounds_m,
        )
        if self.action_codec == "motion_action_v3":
            measured_force_state = self._motion_state()
            _force_magnitude, force_over_threshold = self._force_telemetry(measured_force_state)
            if run.force_stop_enabled:
                sample["force_feedback"] = {
                    "external_force_magnitude_n": _force_magnitude,
                    **public_force_direction(measured_force_state),
                    "source_binding": measured_force_state.get("force_source_binding"),
                }
            if force_over_threshold:
                run.force_threshold_exceeded = True
                if run.force_threshold_first_tick is None:
                    run.force_threshold_first_tick = run.steps_used
            run.force_stop_reason = precision_force_stop_reason(
                interaction=run.active_interaction, magnitude_n=_force_magnitude,
                threshold_n=float(self.motion_profile["sensing"]["force_telemetry_threshold_n"]),
                enabled=run.force_stop_enabled,
            )
        return _LocalTick(
            position=position, orientation=orientation, gripper=gripper,
            actuator_target=actuator_target, delta=delta, rotvec=rotvec,
            sample=sample, workspace=current_workspace,
        )

    @staticmethod
    def _local_tick_stops(run: _LocalMotion, tick: _LocalTick) -> bool:
        """Apply the terminal tracking guard and the precision force stop."""

        if (
            not all(math.isfinite(float(value)) for value in (
                *tick.position, *tick.orientation, tick.gripper,
            ))
        ):
            run.terminal_tracking_reason = "non_finite_state"
        elif (
            run.pose_action
            and tick.actuator_target != run.gripper_hold_directive
        ):
            run.terminal_tracking_reason = "gripper_actuator_target_changed"
        elif (
            tick.workspace["max_violation_m"]
            > GROSS_SAFETY_ENVELOPE_OVERRUN_M
            + CONTROL_HEALTH_NUMERIC_TOLERANCE
            and not run.workspace_recovery_action
        ):
            run.terminal_tracking_reason = "gross_safety_envelope_overrun"
        elif (
            tick.sample["single_tick_translation_m"]
            > GROSS_SINGLE_TICK_TRANSLATION_M
            + CONTROL_HEALTH_NUMERIC_TOLERANCE
        ):
            run.terminal_tracking_reason = "gross_single_tick_translation"
        if run.terminal_tracking_reason is not None:
            run.tracking_stopped_tick = run.steps_used
            return True
        if run.force_stop_reason is not None:
            run.force_stopped_tick = run.steps_used
            return True
        return False

    @staticmethod
    def _local_tick_settled(run: _LocalMotion, tick: _LocalTick) -> bool:
        """Stop on arrival, or on a set_gripper whose jaws have stalled."""

        import numpy as np

        pose_reached = (
            float(np.linalg.norm(tick.delta)) < POSE_POSITION_TOLERANCE_M
            and float(np.linalg.norm(tick.rotvec))
            < POSE_ORIENTATION_TOLERANCE_RAD
        )
        gripper_reached = abs(tick.gripper - run.target_gripper) < GRIPPER_TOLERANCE_NORMALIZED
        if run.action["action"] == "set_gripper" and gripper_stalled(
            run.gripper_trace, target=run.target_gripper,
        ):
            run.gripper_stall_reason = GRIPPER_STALL_REASON
            return True
        return bool((
            run.pose_action and pose_reached
        ) or (
            run.action["action"] == "set_gripper" and pose_reached and gripper_reached
        ))

    def _step_local_command(
        self, run: _LocalMotion, environment: Any, tick: _LocalTick,
    ) -> bool:
        """Issue one bounded OSC command; ``False`` once the episode ends."""

        import numpy as np

        command = np.zeros(7, dtype=np.float32)
        physical_delta = np.concatenate([
            np.clip(tick.delta, -0.025, 0.025),
            np.clip(tick.rotvec, -0.10, 0.10),
        ])
        command[:6] = _inverse_controller_scale(
            physical_delta, self.controller_action_contract,
        )
        command[6] = run.gripper_command
        run.issued_commands.append(command.astype(np.float64).tolist())
        run.previous_position = tick.position.copy()
        if run.steps_used == 0 and run.pose_action and self.action_codec == "motion_action_v3":
            self._remember_backtrack_state()
        _obs, _reward, term, trunc, _info = environment.step(command[None])
        run.steps_used += 1
        self._latch_environment_outcome(
            terminated=term, truncated=trunc, steps=1,
            action=run.action, controller="OSC_POSE",
        )
        if self.terminated or self.truncated or self.sim_steps >= self.max_sim_steps:
            self.truncated = self.truncated or self.sim_steps >= self.max_sim_steps
            return False
        return True

    def _measure_local_motion_end(self, run: _LocalMotion) -> None:
        """Read the reached state and close the tracking record at the last tick."""

        try:
            run.reached_position, run.reached_orientation, run.reached_gripper = self._current_state()
        except (LiberoProError, ValueError, FloatingPointError):
            run.reached_position = run.last_position
            run.reached_orientation = run.last_orientation
            run.reached_gripper = run.last_gripper
            run.terminal_tracking_reason = (
                run.terminal_tracking_reason or "non_finite_or_unreadable_state"
            )
            run.tracking_stopped_tick = (
                run.steps_used if run.tracking_stopped_tick is None else run.tracking_stopped_tick
            )
        try:
            run.gripper_hold_directive_after = (
                self._gripper_hold_directive()
                if run.pose_action else None
            )
        except LiberoProError:
            run.gripper_hold_directive_after = None
            run.terminal_tracking_reason = (
                run.terminal_tracking_reason or "gripper_actuator_target_unreadable"
            )
            run.tracking_stopped_tick = (
                run.steps_used if run.tracking_stopped_tick is None else run.tracking_stopped_tick
            )
        if not run.tracking_samples or run.tracking_samples[-1]["tick"] != run.steps_used:
            self._append_final_tracking_sample(run)

    def _append_final_tracking_sample(self, run: _LocalMotion) -> None:
        final_sample = _pose_tracking_sample(
            tick=run.steps_used,
            before_position=run.before_position,
            before_orientation=run.before_orientation,
            target_position=run.target,
            target_orientation=run.target_orientation,
            position=run.reached_position,
            orientation=run.reached_orientation,
            previous_position=run.previous_position,
            measured_aperture=run.reached_gripper,
            actuator_target=run.gripper_hold_directive_after,
            initial_position_error_m=run.target_error_before_m,
            initial_orientation_error_rad=run.target_orientation_error_before_rad,
        )
        run.tracking_samples.append(final_sample)
        if run.pose_action:
            _record_tracking_telemetry(run, final_sample)
        final_workspace = workspace_state(
            run.reached_position.tolist(), self.workspace_bounds_m,
        )
        if (
            run.terminal_tracking_reason is None
            and final_workspace["max_violation_m"]
            > GROSS_SAFETY_ENVELOPE_OVERRUN_M
            + CONTROL_HEALTH_NUMERIC_TOLERANCE
            and not run.workspace_recovery_action
        ):
            run.terminal_tracking_reason = "gross_safety_envelope_overrun"
        if (
            run.terminal_tracking_reason is None
            and final_sample["single_tick_translation_m"]
            > GROSS_SINGLE_TICK_TRANSLATION_M
            + CONTROL_HEALTH_NUMERIC_TOLERANCE
        ):
            run.terminal_tracking_reason = "gross_single_tick_translation"
        if run.terminal_tracking_reason is not None and run.tracking_stopped_tick is None:
            run.tracking_stopped_tick = run.steps_used

    def _measure_local_motion_residuals(self, run: _LocalMotion) -> None:
        """Residuals against the target and the start, and the last force reading."""

        import numpy as np

        run.pose_translation = float(np.linalg.norm(run.reached_position - run.before_position))
        run.pose_rotation = float(np.linalg.norm(
            _quaternion_error_rotvec(run.reached_orientation, run.before_orientation)
        ))
        run.gripper_residual = abs(run.reached_gripper - run.before_gripper)
        run.target_error_after_m = float(np.linalg.norm(run.target - run.reached_position))
        run.target_orientation_error_after_rad = float(np.linalg.norm(
            _quaternion_error_rotvec(run.target_orientation, run.reached_orientation)
        ))
        run.workspace_before = workspace_state(
            run.before_position.tolist(), self.workspace_bounds_m,
        )
        run.workspace_after = workspace_state(
            run.reached_position.tolist(), self.workspace_bounds_m,
        )
        if (run.force_stop_enabled and run.active_interaction == "precision"
                and run.terminal_tracking_reason is None and not self.terminated and not self.truncated):
            self._measure_final_local_force(run)

    def _measure_final_local_force(self, run: _LocalMotion) -> None:
        # The last iteration may consume the entire action budget, so it
        # has no next-loop measurement. Still report a stop at that tick.
        final_force_state = self._motion_state()
        final_magnitude, final_exceeded = self._force_telemetry(final_force_state)
        if run.tracking_samples and run.tracking_samples[-1]["tick"] == run.steps_used:
            run.tracking_samples[-1]["force_feedback"] = {
                "external_force_magnitude_n": final_magnitude,
                **public_force_direction(final_force_state),
                "source_binding": final_force_state.get("force_source_binding"),
            }
        run.force_threshold_exceeded = run.force_threshold_exceeded or final_exceeded
        if final_exceeded and run.force_threshold_first_tick is None:
            run.force_threshold_first_tick = run.steps_used
        run.force_stop_reason = run.force_stop_reason or precision_force_stop_reason(
            interaction=run.active_interaction, magnitude_n=final_magnitude,
            threshold_n=float(self.motion_profile["sensing"]["force_telemetry_threshold_n"]),
            enabled=run.force_stop_enabled,
        )
        if run.force_stop_reason is not None and run.force_stopped_tick is None:
            run.force_stopped_tick = run.steps_used
