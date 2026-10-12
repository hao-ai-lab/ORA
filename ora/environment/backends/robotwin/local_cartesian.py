"""Bounded local Cartesian servo execution.

A mixin of ``session.RoboTwinLiveSession``; it holds methods only, no
dataclass fields. ``_execute_local_cartesian_action`` runs one servo iteration
at a time through the module-level steps below (solve each arm's joint target,
check clearance, then converge through the motion transaction or
hold the target over physics steps);
``local_cartesian_tracking.LocalCartesianRun`` holds the per-call state and
builds the tracking report, arrays and control sequence.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from ora.environment.backends.robotwin.planning import (
    RoboTwinPlanRejected,
    bounded_dls_joint_delta,
    cartesian_pose_error_twist,
    numerical_cartesian_jacobian,
)
from ora.environment.controllers.inter_arm_clearance import (
    InterArmClearancePolicy,
    minimum_inter_arm_distance as minimum_inter_arm_distance_value,
)
from ora.environment.controllers.motion_transaction import (
    MotionExecutionFault,
    MotionTransactionError,
    execute_measured_convergence,
    require_motion_capabilities,
)
from ora.environment.controllers.profile_capabilities import (
    CONTACT_RECOVERY_SINCE,
    LOCAL_SERVO_DIAGNOSTICS_SINCE,
    profile_has,
)

from .common import (
    RoboTwinControllerTrackingError,
    RoboTwinLiveError,
    _pose_error,
    _wxyz_to_xyzw,
)
from .local_cartesian_tracking import (
    LocalCartesianRun,
    open_local_cartesian_run,
)


class RoboTwinLocalCartesianMixin:
    """Bounded local Cartesian servo execution."""

    def _execute_local_cartesian_action(
        self,
        *,
        environment: Any,
        target_pose_xyzw: Mapping[str, list[float]],
        action: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Execute one bounded Cartesian delta without invoking CuRobo."""

        run = open_local_cartesian_run(
            self,
            environment=environment,
            target_pose_xyzw=target_pose_xyzw,
            action=action,
        )
        for iteration in range(int(run.local["maximum_iterations"]) + 1):
            outcome = _local_cartesian_iteration(self, run, iteration)
            if outcome is not None:
                return run.result(completed=outcome)
        return run.result(completed=False)


def _local_cartesian_iteration(
    session: Any, run: LocalCartesianRun, iteration: int,
) -> bool | None:
    """One servo iteration: ``True`` completes, ``False`` stops, ``None`` goes on."""

    import numpy as np

    environment = run.environment
    local = run.local
    if bool(getattr(environment, "eval_success", False)):
        return True
    current_joints = {
        arm: session._measured_arm_joints(environment, arm)
        for arm in ("left", "right")
    }
    current_pose = {
        arm: _wxyz_to_xyzw(session._public_pose_wxyz(environment, arm))
        for arm in ("left", "right")
    }
    if not run.check_pair_clearance(current_joints):
        return False
    errors = {
        arm: cartesian_pose_error_twist(
            run.target_pose_xyzw[arm], current_pose[arm],
        )
        for arm in run.arms
    }
    if all(
        float(np.linalg.norm(errors[arm][:3]))
        <= float(local["position_tolerance_m"])
        and float(np.linalg.norm(errors[arm][3:]))
        <= float(local["orientation_tolerance_rad"])
        for arm in run.arms
    ):
        if not run.check_pair_clearance(current_joints, endpoint=True):
            return False
        return True
    if iteration == int(local["maximum_iterations"]):
        run.abort = {
            "code": "local_servo_max_iterations",
            "maximum_iterations": int(local["maximum_iterations"]),
        }
        return False

    score = max(
        float(np.linalg.norm(errors[arm][:3]))
        + 0.1 * float(np.linalg.norm(errors[arm][3:]))
        for arm in run.arms
    )
    if run.best_score - score > 1e-5:
        run.best_score = score
        run.stall_count = 0
    else:
        run.stall_count += 1
    if run.stall_count >= int(local["stall_window"]):
        run.abort = {
            "code": "local_servo_stalled",
            "iterations_without_progress": run.stall_count,
        }
        return False

    next_joints: dict[str, list[float]] = {}
    next_velocities: dict[str, list[float]] = {}
    _solve_local_joint_targets(
        session, run, iteration, current_joints, current_pose, errors,
        next_joints, next_velocities,
    )
    collision_joints = {
        arm: list(next_joints.get(arm, current_joints[arm]))
        for arm in ("left", "right")
    }
    if run.abort is not None or not run.check_pair_clearance(collision_joints, planning=True):
        return False

    run.iterations = iteration + 1
    if session._uses_motion_transaction_profile():
        return _execute_local_transaction(
            session, run, iteration, next_joints, next_velocities,
        )
    return _execute_local_physics_steps(
        session, run, iteration, next_joints, next_velocities,
    )


def _solve_local_joint_targets(
    session: Any,
    run: LocalCartesianRun,
    iteration: int,
    current_joints: Mapping[str, list[float]],
    current_pose: Mapping[str, list[float]],
    errors: Mapping[str, Any],
    next_joints: dict[str, list[float]],
    next_velocities: dict[str, list[float]],
) -> None:
    """Fill each target arm's next joint target; a rejection sets ``run.abort``."""

    import numpy as np

    local = run.local
    for arm in run.arms:
        if (
            float(np.linalg.norm(errors[arm][:3]))
            <= float(local["position_tolerance_m"])
            and float(np.linalg.norm(errors[arm][3:]))
            <= float(local["orientation_tolerance_rad"])
        ):
            # An arrived arm holds while its peer converges. Sending
            # zero Cartesian error through the progress-only line
            # search would reject the entire synchronized action.
            next_joints[arm] = list(current_joints[arm])
            next_velocities[arm] = [0.0] * len(current_joints[arm])
            continue
        if run.local_motion_adapter is not None:
            solved = _adapter_local_joint_target(
                session, run, iteration, arm, current_joints, current_pose,
                next_joints, next_velocities,
            )
        else:
            solved = _numeric_local_joint_target(
                session, run, iteration, arm, current_joints, errors,
                next_joints, next_velocities,
            )
        if not solved:
            break


def _adapter_local_joint_target(
    session: Any,
    run: LocalCartesianRun,
    iteration: int,
    arm: str,
    current_joints: Mapping[str, list[float]],
    current_pose: Mapping[str, list[float]],
    next_joints: dict[str, list[float]],
    next_velocities: dict[str, list[float]],
) -> bool:
    """The motion adapter's analytic-Jacobian joint target for one arm."""

    try:
        command = run.local_motion_adapter.compute_local_joint_target(
            arm,
            current_joints[arm],
            current_pose[arm],
            run.target_pose_xyzw[arm],
            run.local,
        )
    except (RuntimeError, RoboTwinPlanRejected) as exc:
        code = (
            exc.code
            if isinstance(exc, RoboTwinPlanRejected)
            else "local_servo_joint_target_rejected"
        )
        run.abort = {
            "code": code,
            "arm": arm,
            "iteration": iteration,
            "message": str(exc),
            **({"details": exc.details}
               if isinstance(exc, RoboTwinPlanRejected) else {}),
        }
        return False
    next_joints[arm] = [
        float(value) for value in command["position"]
    ]
    next_velocities[arm] = [
        float(value) for value in command["velocity"]
    ]
    if profile_has(
        session.control_profile,
        since_version=LOCAL_SERVO_DIAGNOSTICS_SINCE,
    ):
        required_diagnostics = {
            "predicted_score_before", "predicted_score_after",
            "line_search_scale", "solver",
        }
        if not required_diagnostics.issubset(command):
            run.abort = {
                "code": "local_servo_diagnostics_missing",
                "arm": arm,
                "iteration": iteration,
            }
            return False
        run.solver_iterations.append({
            "iteration": iteration,
            "arm": arm,
            "predicted_score_before": round(
                float(command["predicted_score_before"]), 8,
            ),
            "predicted_score_after": round(
                float(command["predicted_score_after"]), 8,
            ),
            "line_search_scale": float(
                command["line_search_scale"]
            ),
            "solver": dict(command["solver"]),
        })
    return True


def _numeric_local_joint_target(
    session: Any,
    run: LocalCartesianRun,
    iteration: int,
    arm: str,
    current_joints: Mapping[str, list[float]],
    errors: Mapping[str, Any],
    next_joints: dict[str, list[float]],
    next_velocities: dict[str, list[float]],
) -> bool:
    """The Pinocchio numeric-Jacobian DLS joint target for one arm."""

    import numpy as np

    environment = run.environment
    local = run.local
    jacobian = numerical_cartesian_jacobian(
        current_joints[arm],
        fk_public_pose=lambda joints, selected=arm: (
            session._pinocchio_public_pose_xyzw(
                environment, selected, joints,
            )
        ),
        epsilon_rad=float(local["finite_difference_epsilon_rad"]),
    )
    delta = bounded_dls_joint_delta(
        errors[arm], jacobian,
        damping=float(local["damping"]),
        maximum_translation_step_m=float(
            local["maximum_translation_step_m"]
        ),
        maximum_rotation_step_rad=float(
            local["maximum_rotation_step_rad"]
        ),
        maximum_joint_step_rad=min(
            float(local["maximum_joint_step_rad"]),
            float(local["maximum_command_to_convergence_ratio"])
            * float(
                session.control_profile["profile"][  # type: ignore[index]
                    "waypoint_execution"
                ]["joint_convergence_tolerance_rad"]
            ),
        ),
    )
    target = np.asarray(current_joints[arm], dtype=np.float64) + delta
    limits = np.asarray(
        session._arm_joint_limits(environment, arm), dtype=np.float64,
    )
    if np.any(target < limits[:, 0]) or np.any(target > limits[:, 1]):
        run.abort = {
            "code": "local_servo_joint_limit",
            "arm": arm,
            "iteration": iteration,
        }
        return False
    next_joints[arm] = target.tolist()
    velocity = delta / float(local["velocity_target_horizon_s"])
    velocity = np.clip(
        velocity,
        -float(local["maximum_velocity_target_rad_s"]),
        float(local["maximum_velocity_target_rad_s"]),
    )
    next_velocities[arm] = velocity.tolist()
    return True


def _execute_local_transaction(
    session: Any,
    run: LocalCartesianRun,
    iteration: int,
    next_joints: Mapping[str, list[float]],
    next_velocities: Mapping[str, list[float]],
) -> bool | None:
    """Converge one iteration's joint target through the motion transaction."""

    environment = run.environment
    action = run.action
    arms = run.arms
    profile = run.profile
    adapter = session._motion_backend_adapter(
        session=session, environment=environment,
    )
    require_motion_capabilities(
        adapter.capabilities,
        requested_arms=arms,
        require_local=True,
        require_synchronized=len(arms) > 1,
    )

    def capture_local_step(_stage: int, _wait: int) -> None:
        if not session.operator_execution_video:
            return
        updater = getattr(environment, "_update_render", None)
        if callable(updater):
            updater()
        session._capture_operator_execution_frame(
            action=action,
            phase="motion_transaction_local",
            control_step_index=len(run.measured_tcp["left"]),
        )

    def guard_local_state(
        measured: Mapping[str, Sequence[float]],
    ) -> Mapping[str, Any] | None:
        return (
            None
            if run.check_pair_clearance({
                arm: [float(value) for value in measured[arm]]
                for arm in ("left", "right")
            })
            else dict(run.abort or {"code": "inter_arm_collision"})
        )

    try:
        transaction_report, transaction_arrays = (
            execute_measured_convergence(
                adapter=adapter,
                paths={
                    arm: {
                        "position": [next_joints[arm]],
                        "velocity": [next_velocities[arm]],
                    }
                    for arm in arms
                },
                all_arms=("left", "right"),
                profile=profile["motion_transaction"][
                    "local_convergence"
                ],
                workspace_bounds_m=session.workspace_bounds_m,
                mode="local",
                after_step=capture_local_step,
                terminal_check=lambda _stage, _wait: session._after_physics_step(
                    environment, action=action, phase="motion_transaction_local",
                ),
                measured_state_guard=guard_local_state,
                **({"blocked_policy": profile["blocked_motion_recovery"]}
                   if profile_has(session.control_profile, since_version=CONTACT_RECOVERY_SINCE) else {}),
            )
        )
    except MotionExecutionFault as exc:
        run.merge_transaction(exc.report, exc.arrays)
        run.abort = dict(
            exc.report.get("abort")
            or {"code": "local_motion_execution_fault"}
        )
        run.abort.setdefault("iteration", iteration)
        raise RoboTwinControllerTrackingError(
            str(exc), summary=run.report(completed=False), arrays=run.arrays(),
            control_sequence=run.replay(),
        ) from exc
    except MotionTransactionError as exc:
        if run.execution_started or run.measured_tcp["left"]:
            run.abort = {
                "code": "local_motion_transaction_fault",
                "iteration": iteration,
                "detail": type(exc).__name__,
            }
            raise RoboTwinControllerTrackingError(
                f"local motion transaction failed closed: {exc}",
                summary=run.report(completed=False), arrays=run.arrays(),
                control_sequence=run.replay(),
            ) from exc
        raise RoboTwinLiveError(
            f"local motion transaction failed closed: {exc}"
        ) from exc
    run.merge_transaction(transaction_report, transaction_arrays)
    if transaction_report.get("termination") is not None:
        return True
    if not transaction_report["completed"]:
        run.abort = dict(transaction_report["abort"])
        run.abort["iteration"] = iteration
        return False
    return None


def _execute_local_physics_steps(
    session: Any,
    run: LocalCartesianRun,
    iteration: int,
    next_joints: Mapping[str, list[float]],
    next_velocities: Mapping[str, list[float]],
) -> bool | None:
    """Hold one iteration's joint target for up to the per-iteration step budget."""

    environment = run.environment
    action = run.action
    local = run.local
    for wait_index in range(
        int(local["maximum_physics_steps_per_iteration"])
    ):
        for arm in run.arms:
            run.robot.set_arm_joints(
                next_joints[arm], next_velocities[arm], arm,
            )
            run.issued_position[arm].append(list(next_joints[arm]))
            run.issued_velocity[arm].append(list(next_velocities[arm]))
        run.scene_step()
        terminal = session._after_physics_step(
            environment, action=action, phase="local_cartesian",
        )
        all_converged = _sample_local_physics_step(
            session, run, next_joints, terminal=terminal,
        )
        if terminal:
            run.iteration_wait_steps.append(wait_index + 1)
            return True
        if not run.check_pair_clearance({
            arm: run.measured_joints[arm][-1]
            for arm in ("left", "right")
        }):
            raise RoboTwinControllerTrackingError(
                "local Cartesian tracking guard inter_arm_collision",
                summary=run.report(completed=False), arrays=run.arrays(),
            )
        if session.operator_execution_video:
            updater = getattr(environment, "_update_render", None)
            if callable(updater):
                updater()
            session._capture_operator_execution_frame(
                action=action, phase="local_cartesian",
                control_step_index=len(run.measured_tcp["left"]) - 1,
            )
        if (
            all_converged
            and wait_index + 1
            >= int(local["minimum_physics_steps_per_iteration"])
        ):
            run.iteration_wait_steps.append(wait_index + 1)
            return None
    run.iteration_wait_steps.append(
        int(local["maximum_physics_steps_per_iteration"])
    )
    run.abort = {
        "code": "local_servo_joint_timeout",
        "iteration": iteration,
        "maximum_physics_steps": int(
            local["maximum_physics_steps_per_iteration"]
        ),
    }
    return False


def _sample_local_physics_step(
    session: Any,
    run: LocalCartesianRun,
    next_joints: Mapping[str, list[float]],
    *,
    terminal: bool,
) -> bool:
    """Record both arms after one physics step and apply the gross guards."""

    environment = run.environment
    gross = run.gross
    all_converged = True
    for arm in ("left", "right"):
        joints = session._measured_arm_joints(environment, arm)
        tcp = _wxyz_to_xyzw(session._public_pose_wxyz(environment, arm))
        run.measured_joints[arm].append(joints)
        run.measured_tcp[arm].append(tcp)
        if terminal:
            continue
        if not all(math.isfinite(value) for value in [*joints, *tcp]):
            run.fail_gross("non_finite_measured_state", arm, math.inf)
        tcp_step = math.sqrt(sum(
            (tcp[index] - run.previous_tcp[arm][index]) ** 2
            for index in range(3)
        ))
        run.maximum_tcp_step[arm] = max(run.maximum_tcp_step[arm], tcp_step)
        if tcp_step > float(gross["maximum_tcp_step_m"]):
            run.fail_gross("gross_tcp_step", arm, tcp_step)
        run.previous_tcp[arm] = tcp
        drift = math.sqrt(sum(
            (tcp[index] - run.initial_pose[arm][index]) ** 2
            for index in range(3)
        ))
        run.maximum_position_drift[arm] = max(
            run.maximum_position_drift[arm], drift,
        )
        margin = float(gross["workspace_margin_m"])
        for index, axis in enumerate(("x", "y", "z")):
            lower, upper = session.workspace_bounds_m[axis]
            if not float(lower) - margin <= tcp[index] <= float(upper) + margin:
                run.fail_gross(f"workspace_{axis}", arm, tcp[index])
        if arm in run.arms:
            joint_error = max(
                abs(actual - expected)
                for actual, expected in zip(joints, next_joints[arm])
            )
            run.maximum_joint_error[arm] = max(
                run.maximum_joint_error[arm], joint_error,
            )
            if joint_error > float(
                session.control_profile["profile"]["waypoint_execution"][  # type: ignore[index]
                    "joint_convergence_tolerance_rad"
                ]
            ):
                all_converged = False
    return all_converged
