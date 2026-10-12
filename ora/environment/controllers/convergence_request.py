"""Request admission for ``motion_transaction.execute_measured_convergence``.

Admission runs before any controller write, in a fixed order -- route,
terminal endpoint, measured initial joints, targets and stage count, profile,
workspace -- and that order decides which error a caller sees. Its result,
:class:`ConvergenceRequest`, is what ``measured_convergence.MeasuredConvergence``
executes. The starting public TCP and the optional joint-blocked monitor are
read and built after admission, as their own steps.

This module imports the shared contract primitives from ``motion_transaction``;
that module imports this one only when the executor runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ora.environment.controllers.motion_transaction import (
    MotionBackendAdapter,
    MotionTransactionError,
    _finite_vector,
    _strict_positive_int,
    _strict_real,
    _validated_pose,
    _validated_workspace_bounds,
)


@dataclass(frozen=True)
class ConvergenceRequest:
    """One admitted joint-path target set with its measured start, stage count and limits."""

    mode: str
    arms: tuple[str, ...]
    raw_targets: Mapping[str, Any]
    normalized: dict[str, tuple[np.ndarray, np.ndarray]]
    endpoint_enabled: bool
    normalized_endpoints: dict[str, np.ndarray]
    endpoint_position_tolerance: float
    endpoint_orientation_tolerance: float
    initial_joints: dict[str, np.ndarray]
    initial_joint_shapes: dict[str, tuple[int, ...]]
    total_stages: int
    fixed_tolerance: float
    maximum_wait: int
    maximum_total: int
    maximum_tcp_step_limit: float
    workspace_margin: float
    workspace: dict[str, tuple[float, float]]


def admit_convergence_request(
    *, adapter: MotionBackendAdapter,
    paths: Mapping[str, Mapping[str, Any]] | None,
    all_arms: Sequence[str],
    profile: Mapping[str, Any],
    workspace_bounds_m: Mapping[str, Sequence[float]],
    mode: str,
    endpoint_public_tcp_targets: Mapping[str, Sequence[float]] | None,
    endpoint_position_tolerance_m: float | None,
    endpoint_orientation_tolerance_rad: float | None,
    stage_count: int | None,
) -> ConvergenceRequest:
    """Validate one request before any write, reading only measured joints."""

    arms, raw_targets = _admit_route(
        adapter, paths=paths, all_arms=all_arms, mode=mode,
    )
    (
        endpoint_enabled, normalized_endpoints,
        endpoint_position_tolerance, endpoint_orientation_tolerance,
    ) = _admit_terminal_endpoint(
        endpoint_public_tcp_targets, endpoint_position_tolerance_m,
        endpoint_orientation_tolerance_rad,
        raw_targets=raw_targets,
    )
    initial_joints = _read_initial_joints(adapter, arms)
    initial_joint_shapes = {
        arm: joints.shape for arm, joints in initial_joints.items()
    }
    normalized, total_stages = _normalize_targets(
        raw_targets, initial_joints=initial_joints, stage_count=stage_count,
    )
    (
        fixed_tolerance, maximum_wait, maximum_total, maximum_tcp_step_limit,
        workspace_margin,
    ) = _admit_profile(profile)
    workspace = _validated_workspace_bounds(
        workspace_bounds_m, margin_m=workspace_margin,
    )
    return ConvergenceRequest(
        mode=mode, arms=arms, raw_targets=raw_targets, normalized=normalized,
        endpoint_enabled=endpoint_enabled,
        normalized_endpoints=normalized_endpoints,
        endpoint_position_tolerance=endpoint_position_tolerance,
        endpoint_orientation_tolerance=endpoint_orientation_tolerance,
        initial_joints=initial_joints, initial_joint_shapes=initial_joint_shapes,
        total_stages=total_stages, fixed_tolerance=fixed_tolerance,
        maximum_wait=maximum_wait, maximum_total=maximum_total,
        maximum_tcp_step_limit=maximum_tcp_step_limit,
        workspace_margin=workspace_margin, workspace=workspace,
    )


def _admit_route(
    adapter: MotionBackendAdapter, *,
    paths: Mapping[str, Mapping[str, Any]] | None,
    all_arms: Sequence[str], mode: str,
) -> tuple[tuple[str, ...], Mapping[str, Any]]:
    """Mode, adapter topology and the joint-path target set."""

    if mode not in {"global", "local"}:
        raise MotionTransactionError("motion convergence mode must be global or local")
    arms = tuple(all_arms)
    if tuple(adapter.capabilities.arms) != arms:
        raise MotionTransactionError("adapter topology differs from requested all_arms")
    if paths is None:
        raise MotionTransactionError("motion convergence requires a joint target set")
    raw_targets = paths
    if set(raw_targets) - set(arms):
        raise MotionTransactionError("joint path targets an unknown arm")
    return arms, raw_targets


def _admit_terminal_endpoint(
    endpoint_public_tcp_targets: Mapping[str, Sequence[float]] | None,
    endpoint_position_tolerance_m: float | None,
    endpoint_orientation_tolerance_rad: float | None,
    *, raw_targets: Mapping[str, Any],
) -> tuple[bool, dict[str, np.ndarray], float, float]:
    """The optional strict public-TCP endpoint of a joint path's last waypoint."""

    endpoint_arguments = (
        endpoint_public_tcp_targets,
        endpoint_position_tolerance_m,
        endpoint_orientation_tolerance_rad,
    )
    endpoint_enabled = all(item is not None for item in endpoint_arguments)
    if any(item is not None for item in endpoint_arguments) and not endpoint_enabled:
        raise MotionTransactionError(
            "terminal endpoint target and both tolerances must be supplied together"
        )
    normalized_endpoints: dict[str, np.ndarray] = {}
    endpoint_position_tolerance = 0.0
    endpoint_orientation_tolerance = 0.0
    if endpoint_enabled:
        assert endpoint_public_tcp_targets is not None
        if set(endpoint_public_tcp_targets) != set(raw_targets):
            raise MotionTransactionError(
                "terminal endpoint arms differ from joint-path target arms"
            )
        normalized_endpoints = {
            arm: _validated_pose(
                pose, label=f"{arm} terminal public TCP target",
            )
            for arm, pose in endpoint_public_tcp_targets.items()
        }
        endpoint_position_tolerance = _strict_real(
            endpoint_position_tolerance_m,
            label="terminal endpoint position tolerance",
        )
        endpoint_orientation_tolerance = _strict_real(
            endpoint_orientation_tolerance_rad,
            label="terminal endpoint orientation tolerance",
        )
        if not (
            0.0 < endpoint_position_tolerance <= 0.2
            and 0.0 < endpoint_orientation_tolerance <= math.pi
        ):
            raise MotionTransactionError(
                "terminal endpoint convergence tolerances are invalid"
            )
    return (
        endpoint_enabled, normalized_endpoints,
        endpoint_position_tolerance, endpoint_orientation_tolerance,
    )


def _read_initial_joints(
    adapter: MotionBackendAdapter, arms: tuple[str, ...],
) -> dict[str, np.ndarray]:
    try:
        initial_joints = {
            arm: _finite_vector(
                adapter.read_joint_positions(arm), label=f"{arm} initial joints",
            )
            for arm in arms
        }
    except Exception as exc:
        if isinstance(exc, MotionTransactionError):
            raise
        raise MotionTransactionError(
            "motion backend failed while reading initial joints"
        ) from exc
    return initial_joints


def _normalize_targets(
    raw_targets: Mapping[str, Any], *,
    initial_joints: Mapping[str, np.ndarray], stage_count: int | None,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], int]:
    """Validated joint paths and the planned stage count."""

    normalized: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for arm, path in raw_targets.items():
        if not isinstance(path, Mapping) or set(path) != {"position", "velocity"}:
            raise MotionTransactionError("joint path fields drifted")
        position = np.asarray(path["position"], dtype=np.float64)
        velocity = np.asarray(path["velocity"], dtype=np.float64)
        measured = initial_joints[arm]
        if (
            position.ndim != 2 or position.shape[1] != measured.shape[0]
            or velocity.shape != position.shape or not len(position)
            or not np.isfinite(position).all() or not np.isfinite(velocity).all()
        ):
            raise MotionTransactionError(f"{arm} joint path is malformed")
        normalized[arm] = (position, velocity)
    inferred_stages = max((len(item[0]) for item in normalized.values()), default=0)
    if stage_count is not None and (
        isinstance(stage_count, bool) or not isinstance(stage_count, int)
    ):
        raise MotionTransactionError("motion execution stage count must be an integer")
    total_stages = inferred_stages if stage_count is None else stage_count
    if total_stages <= 0 or inferred_stages > total_stages:
        raise MotionTransactionError("motion execution stage count is invalid")
    return normalized, total_stages


def _admit_profile(
    profile: Mapping[str, Any],
) -> tuple[float, int, int, float, float]:
    """The closed convergence profile's validated limits.

    Returns the joint tolerance, per-waypoint and total physics-step budgets,
    gross TCP step limit and workspace margin.
    """

    required = {
        "joint_convergence_tolerance_rad", "maximum_physics_steps_per_waypoint",
        "maximum_total_physics_steps", "maximum_tcp_step_m", "workspace_margin_m",
    }
    if not isinstance(profile, Mapping) or set(profile) != required:
        raise MotionTransactionError("motion convergence profile fields drifted")
    fixed_tolerance = _strict_real(
        profile["joint_convergence_tolerance_rad"],
        label="joint convergence tolerance",
    )
    maximum_wait = _strict_positive_int(
        profile["maximum_physics_steps_per_waypoint"],
        label="maximum physics steps per waypoint", maximum=120,
    )
    maximum_total = _strict_positive_int(
        profile["maximum_total_physics_steps"],
        label="maximum total physics steps", maximum=20000,
    )
    maximum_tcp_step_limit = _strict_real(
        profile["maximum_tcp_step_m"], label="maximum TCP step",
    )
    workspace_margin = _strict_real(
        profile["workspace_margin_m"], label="workspace margin",
    )
    if not (
        math.isfinite(fixed_tolerance) and 0.0 < fixed_tolerance <= 0.2
        and 1 <= maximum_wait <= 120
        and 1 <= maximum_total <= 20000
        and math.isfinite(maximum_tcp_step_limit) and maximum_tcp_step_limit > 0.0
        and math.isfinite(workspace_margin) and workspace_margin >= 0.0
    ):
        raise MotionTransactionError("motion convergence profile values are invalid")
    return (
        fixed_tolerance, maximum_wait, maximum_total, maximum_tcp_step_limit,
        workspace_margin,
    )


def read_initial_tcp(
    adapter: MotionBackendAdapter, request: ConvergenceRequest,
) -> dict[str, np.ndarray]:
    """Measured starting public TCP per arm, admitted inside the workspace."""

    workspace, workspace_margin = request.workspace, request.workspace_margin
    try:
        previous_tcp = {
            arm: _validated_pose(
                adapter.read_public_tcp_pose(arm), label=f"{arm} initial TCP",
            )
            for arm in request.arms
        }
    except Exception as exc:
        if isinstance(exc, MotionTransactionError):
            raise
        raise MotionTransactionError(
            "motion backend failed while reading initial TCP state"
        ) from exc
    for arm, tcp in previous_tcp.items():
        for axis_index, axis in enumerate(("x", "y", "z")):
            lower, upper = workspace[axis]
            if not lower - workspace_margin <= tcp[axis_index] <= upper + workspace_margin:
                raise MotionTransactionError(
                    f"{arm} initial TCP is outside workspace {axis}"
                )
    return previous_tcp


def joint_blocked_monitor(
    blocked_policy: Mapping[str, Any] | None, request: ConvergenceRequest,
) -> Any:
    """The optional per-sample joint-blocked monitor; ``None`` without a policy."""

    if blocked_policy is None:
        return None
    from .motion_progress import JointBlockedMonitor
    count = blocked_policy.get("consecutive_frozen_physics_steps")
    limit = blocked_policy.get("maximum_frozen_joint_step_rad")
    if (isinstance(count, bool) or not isinstance(count, int)
            or count <= 0 or count > request.maximum_wait
            or isinstance(limit, bool) or not isinstance(limit, (int, float))
            or not math.isfinite(limit) or limit <= 0):
        raise MotionTransactionError("blocked progress policy is invalid for joint convergence")
    return JointBlockedMonitor(request.initial_joints,
        consecutive_physics_steps=count, maximum_joint_step_rad=float(limit))


__all__ = [
    "ConvergenceRequest", "admit_convergence_request", "joint_blocked_monitor",
    "read_initial_tcp",
]
