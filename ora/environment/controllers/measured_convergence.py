"""The stage loop behind ``motion_transaction.execute_measured_convergence``.

``execute_measured_convergence`` admits its request in ``convergence_request``,
reads the starting public TCP, builds the optional joint-blocked monitor and
then runs one :class:`MeasuredConvergence`. That object holds the
transaction's issued and measured evidence and owns the stage loop: write the
stage's targets, advance one physics step, sample every arm, apply the gross
guards, and repeat until the stage converges, its wait budget expires, a guard
aborts or the environment reports a terminal.

This module imports the shared contract primitives from ``motion_transaction``;
that module imports this one only when the executor runs.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Callable, Mapping, NoReturn, Sequence

import numpy as np

from ora.environment.controllers.convergence_request import ConvergenceRequest
from ora.environment.controllers.motion_transaction import (
    MOTION_EXECUTION_SCHEMA,
    MOTION_TRANSACTION_SCHEMA,
    MotionBackendAdapter,
    MotionExecutionFault,
    MotionTransactionError,
    _finite_vector,
    _pose_error,
    _stage_index,
    _validated_pose,
)


class MeasuredConvergence:
    """Issued and measured evidence of one admitted transaction, and its stage loop.

    Every stage writes its targets, advances one physics step, samples every
    arm and repeats until the stage converges, its wait budget expires, a
    gross guard aborts or the environment reports a terminal. Guards raise
    :class:`MotionExecutionFault` carrying the report built so far.
    """

    def __init__(
        self, adapter: MotionBackendAdapter, request: ConvergenceRequest, *,
        previous_tcp: dict[str, np.ndarray], blocked_monitor: Any,
        before_step: Callable[[int, int], None] | None,
        after_step: Callable[[int, int], None] | None,
        terminal_check: Callable[[int, int], bool] | None,
        measured_state_guard: Callable[
            [Mapping[str, Sequence[float]]], Mapping[str, Any] | None
        ] | None,
    ) -> None:
        arms = request.arms
        self.adapter = adapter
        self.request = request
        self.before_step = before_step
        self.after_step = after_step
        self.terminal_check = terminal_check
        self.measured_state_guard = measured_state_guard
        self.issued_position: dict[str, list] = {arm: [] for arm in arms}
        self.issued_velocity: dict[str, list] = {arm: [] for arm in arms}
        # Native Cartesian command evidence: no admitted request writes it
        # now, but the report keeps its (empty) arrays.
        self.issued_tcp: dict[str, list] = {arm: [] for arm in arms}
        self.issued_controller: dict[str, list] = {arm: [] for arm in arms}
        self.measured_joints: dict[str, list] = {arm: [] for arm in arms}
        self.measured_tcp: dict[str, list] = {arm: [] for arm in arms}
        self.previous_tcp = previous_tcp
        self.maximum_error = {arm: 0.0 for arm in arms}
        self.maximum_tcp_step_by_arm = {arm: 0.0 for arm in arms}
        self.waits: list[int] = []
        self.abort: dict[str, Any] | None = None
        self.sampled_steps = 0
        self.controller_write_attempts = 0
        self.physics_step_attempts = 0
        self.callback_attempts = 0
        self.terminal_endpoint_errors: dict[str, dict[str, float]] = {}
        self.termination: dict[str, Any] | None = None
        self.blocked_monitor = blocked_monitor

    def run(self) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        """Converge every stage in order; stop at the first abort or terminal."""

        for stage in range(self.request.total_stages):
            finished = self._converge_stage(stage, self._stage_targets(stage))
            if finished is not None:
                return finished
            if self.abort is not None:
                break
        return self.materialize()

    def _stage_targets(self, stage: int) -> dict[str, tuple[np.ndarray, np.ndarray, float]]:
        targets: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
        for arm, (position, velocity) in self.request.normalized.items():
            index = _stage_index(len(position), self.request.total_stages, stage)
            target = position[index]
            planned_velocity = velocity[index]
            targets[arm] = (target, planned_velocity, self.request.fixed_tolerance)
        return targets

    def _converge_stage(
        self, stage: int, targets: Mapping[str, tuple[np.ndarray, np.ndarray, float]],
    ) -> tuple[dict[str, Any], dict[str, np.ndarray]] | None:
        """Wait out one stage. Return the evidence only when the transaction ends early."""

        request = self.request
        converged = False
        joint_targets_converged = False
        for wait_index in range(request.maximum_wait):
            if self.sampled_steps >= request.maximum_total:
                self.abort = {
                    "code": "execution_step_budget_exhausted",
                    "stage_index": stage,
                    "maximum_total_physics_steps": request.maximum_total,
                }
                break
            self._dispatch(stage, wait_index, targets)
            self._advance_physics(stage, wait_index)
            all_converged, joint_targets_converged = self._measure(stage, targets)
            self._apply_measured_state_guard(stage)
            self._run_after_step(stage, wait_index)
            if self.termination is not None:
                self.waits.append(wait_index + 1)
                return self.materialize()
            if self.blocked_monitor is not None:
                blocked = self.blocked_monitor.sample(
                    {arm: self.measured_joints[arm][-1] for arm in request.arms},
                    {arm: target.tolist() for arm, (target, _velocity, _tolerance) in targets.items()},
                    tolerance_rad=request.fixed_tolerance, stage_index=stage,
                    planned_stages=request.total_stages, waypoint_wait_steps=self.waits,
                )
                if blocked is not None:
                    self.abort = blocked
                    self.waits.append(wait_index + 1)
                    return self.materialize()
            if all_converged:
                self.waits.append(wait_index + 1)
                converged = True
                break
        if self.abort is None and not converged:
            self.waits.append(request.maximum_wait)
            self.abort = {
                "code": (
                    "terminal_endpoint_tracking_timeout"
                    if request.endpoint_enabled
                    and stage == request.total_stages - 1
                    and joint_targets_converged
                    else "waypoint_tracking_timeout"
                ),
                "stage_index": stage,
                "maximum_physics_steps": request.maximum_wait,
            }
        return None

    def _dispatch(
        self, stage: int, wait_index: int,
        targets: Mapping[str, tuple[np.ndarray, np.ndarray, float]],
    ) -> None:
        """Write this sample's controller targets; only the first wait carries velocity."""

        for arm, (target, planned_velocity, _tolerance) in targets.items():
            velocity = planned_velocity if wait_index == 0 else np.zeros_like(planned_velocity)
            self.controller_write_attempts += 1
            try:
                self.adapter.write_joint_target(
                    arm, target.tolist(), velocity.tolist(),
                )
            except Exception as exc:
                self._fail(
                    "backend_write_fault", arm=arm, stage=stage,
                    detail=type(exc).__name__,
                )
            self.issued_position[arm].append(target.tolist())
            self.issued_velocity[arm].append(velocity.tolist())

    def _advance_physics(self, stage: int, wait_index: int) -> None:
        """Run ``before_step``, one physics step, then poll the environment terminal."""

        if self.before_step is not None:
            self.callback_attempts += 1
            try:
                self.before_step(stage, wait_index)
            except Exception as exc:
                self._fail(
                    "before_step_fault", stage=stage,
                    detail=type(exc).__name__,
                )
        self.physics_step_attempts += 1
        try:
            self.adapter.step_physics()
        except Exception as exc:
            self._fail(
                "backend_step_fault", stage=stage,
                detail=type(exc).__name__,
            )
        self.sampled_steps += 1
        # The environment owns its terminal boundary. Poll immediately
        # after physics, before a waypoint wait, settle or next dispatch.
        # A terminal stop is normal completion, not a tracking timeout.
        if self.terminal_check is not None:
            try:
                terminal = self.terminal_check(stage, wait_index)
                if not isinstance(terminal, bool):
                    raise MotionTransactionError("terminal check must return a boolean")
            except Exception as exc:
                self._fail("terminal_check_fault", stage=stage, detail=type(exc).__name__)
            if terminal:
                self.termination = {
                    "reason": "environment_terminal",
                    "stage_index": stage,
                    "wait_index": wait_index,
                    "physics_step_index": self.sampled_steps,
                }

    def _measure(
        self, stage: int, targets: Mapping[str, tuple[np.ndarray, np.ndarray, float]],
    ) -> tuple[bool, bool]:
        """Sample every arm once; return (all targets converged, joint targets converged)."""

        request = self.request
        all_converged = True
        joint_targets_converged = True
        for arm in request.arms:
            joints, tcp = self._read_measured(arm, stage)
            self.measured_joints[arm].append(joints.tolist())
            self.measured_tcp[arm].append(tcp.tolist())
            if self.termination is not None:
                continue
            self._check_gross_motion(arm, stage, tcp)
            if arm in targets:
                target, _velocity, tolerance = targets[arm]
                error = float(np.max(np.abs(joints - target)))
                self.maximum_error[arm] = max(self.maximum_error[arm], error)
                if error > tolerance:
                    all_converged = False
                    joint_targets_converged = False
            if not self._terminal_endpoint_converged(arm, stage, tcp):
                all_converged = False
        return all_converged, joint_targets_converged

    def _read_measured(self, arm: str, stage: int) -> tuple[np.ndarray, np.ndarray]:
        try:
            joints = _finite_vector(
                self.adapter.read_joint_positions(arm), label=f"{arm} joints",
            )
            tcp = _validated_pose(
                self.adapter.read_public_tcp_pose(arm), label=f"{arm} TCP",
            )
            if joints.shape != self.request.initial_joint_shapes[arm]:
                raise MotionTransactionError(
                    f"{arm} measured joint shape changed during execution"
                )
        except Exception:
            self._fail(
                "invalid_measured_state", arm=arm,
                stage=stage, value=math.inf,
            )
        return joints, tcp

    def _check_gross_motion(self, arm: str, stage: int, tcp: np.ndarray) -> None:
        """Abort on a gross per-sample TCP step or a TCP outside the workspace."""

        request = self.request
        tcp_step = float(np.linalg.norm(tcp[:3] - self.previous_tcp[arm][:3]))
        self.maximum_tcp_step_by_arm[arm] = max(
            self.maximum_tcp_step_by_arm[arm], tcp_step,
        )
        if tcp_step > request.maximum_tcp_step_limit:
            self._fail("gross_tcp_step", arm=arm, stage=stage, value=tcp_step)
        self.previous_tcp[arm] = tcp
        for axis_index, axis in enumerate(("x", "y", "z")):
            lower, upper = request.workspace[axis]
            if not lower - request.workspace_margin <= tcp[axis_index] <= upper + request.workspace_margin:
                self._fail(
                    f"workspace_{axis}", arm=arm, stage=stage,
                    value=float(tcp[axis_index]),
                )

    def _terminal_endpoint_converged(self, arm: str, stage: int, tcp: np.ndarray) -> bool:
        """The last waypoint also holds until the strict public-TCP endpoint is met."""

        request = self.request
        if not (
            request.endpoint_enabled
            and stage == request.total_stages - 1
            and arm in request.normalized_endpoints
        ):
            return True
        endpoint_error = _pose_error(
            request.normalized_endpoints[arm], tcp,
        )
        self.terminal_endpoint_errors[arm] = endpoint_error
        return not (
            endpoint_error["translation_m"]
            > request.endpoint_position_tolerance
            or endpoint_error["orientation_rad"]
            > request.endpoint_orientation_tolerance
        )

    def _apply_measured_state_guard(self, stage: int) -> None:
        if self.measured_state_guard is None or self.termination is not None:
            return
        try:
            guard = self.measured_state_guard({
                arm: self.measured_joints[arm][-1] for arm in self.request.arms
            })
        except Exception as exc:
            self._fail(
                "measured_state_guard_fault", stage=stage,
                detail=type(exc).__name__,
            )
        if guard is not None:
            if not isinstance(guard, Mapping) or not isinstance(
                guard.get("code"), str,
            ):
                self._fail(
                    "measured_state_guard_fault", stage=stage,
                    detail="invalid_guard_result",
                )
            self.abort = copy.deepcopy(dict(guard))
            self.abort.setdefault("stage_index", stage)
            report, arrays = self.materialize()
            raise MotionExecutionFault(
                f"motion execution guard {self.abort['code']}",
                report=report, arrays=arrays,
            )

    def _run_after_step(self, stage: int, wait_index: int) -> None:
        if self.after_step is not None:
            self.callback_attempts += 1
            try:
                self.after_step(stage, wait_index)
            except Exception as exc:
                self._fail(
                    "after_step_fault", stage=stage,
                    detail=type(exc).__name__,
                )

    def _fail(
        self, code: str, *, stage: int, arm: str | None = None,
        value: float | None = None, detail: str | None = None,
    ) -> NoReturn:
        self.abort = {
            "code": code,
            "stage_index": stage,
            **({"arm": arm} if arm is not None else {}),
            **({
                "value": round(value, 8) if math.isfinite(value) else "non_finite",
            } if value is not None else {}),
            **({"detail": detail} if detail is not None else {}),
        }
        report, arrays = self.materialize()
        raise MotionExecutionFault(
            f"motion execution guard {code} on {arm}",
            report=report,
            arrays=arrays,
        )

    def materialize(self) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        """The execution report and issued/measured arrays as they stand now."""

        request = self.request
        arms = request.arms
        abort = self.abort
        termination = self.termination
        endpoint_enabled = request.endpoint_enabled
        report = {
            "schema": MOTION_EXECUTION_SCHEMA,
            "transaction_schema": MOTION_TRANSACTION_SCHEMA,
            "backend": self.adapter.capabilities.backend_id,
            "mode": request.mode,
            "target_arms": list(request.raw_targets),
            "planned_stages": request.total_stages,
            "sampled_steps": self.sampled_steps,
            "controller_write_attempts": self.controller_write_attempts,
            "physics_step_attempts": self.physics_step_attempts,
            "execution_started": bool(
                self.controller_write_attempts or self.physics_step_attempts
                or self.callback_attempts
            ),
            "completed": abort is None,
            "abort": abort,
            **({"termination": copy.deepcopy(termination)} if termination else {}),
            "waypoint_wait_steps": self.waits,
            **({key: copy.deepcopy(abort[key]) for key in (
                "blocked_at_stage", "frozen_steps", "progress", "stall_shape",
            ) if key in abort} if self.blocked_monitor is not None and abort else {}),
            "terminal_endpoint_check": {
                "enabled": endpoint_enabled,
                "target_arms": list(request.normalized_endpoints),
                "position_tolerance_m": (
                    request.endpoint_position_tolerance if endpoint_enabled else None
                ),
                "orientation_tolerance_rad": (
                    request.endpoint_orientation_tolerance if endpoint_enabled else None
                ),
                "final_errors": copy.deepcopy(self.terminal_endpoint_errors),
            },
            "arms": {
                arm: {
                    "targeted": arm in request.raw_targets,
                    "maximum_joint_tracking_error_rad": round(self.maximum_error[arm], 8),
                    "maximum_tcp_step_m": round(self.maximum_tcp_step_by_arm[arm], 8),
                }
                for arm in arms
            },
        }
        return report, self._arrays()

    def _arrays(self) -> dict[str, np.ndarray]:
        arms = self.request.arms
        return {
            **{
                f"issued_{arm}_arm__position": np.asarray(
                    self.issued_position[arm], dtype=np.float32,
                )
                for arm in arms
            },
            **{
                f"issued_{arm}_arm__velocity": np.asarray(
                    self.issued_velocity[arm], dtype=np.float32,
                )
                for arm in arms
            },
            **{
                f"issued_{arm}_tcp__pose_xyzw": np.asarray(
                    self.issued_tcp[arm], dtype=np.float32,
                )
                for arm in arms
            },
            **{
                f"issued_{arm}_controller__command": np.asarray(
                    self.issued_controller[arm], dtype=np.float32,
                )
                for arm in arms
            },
            **{
                f"measured_{arm}_arm__position": np.asarray(
                    self.measured_joints[arm], dtype=np.float32,
                )
                for arm in arms
            },
            **{
                f"measured_{arm}_tcp__pose_xyzw": np.asarray(
                    self.measured_tcp[arm], dtype=np.float32,
                )
                for arm in arms
            },
        }


__all__ = ["MeasuredConvergence"]
