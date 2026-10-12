"""Backend-neutral Cartesian reachability: forward kinematics in, verdict out.

These primitives answer "can this arm hold this pose" from a forward-kinematics
callable and joint limits alone. They take no adapter and no session, so they
cannot reach a motion planner even by accident -- which is the point. Probing
reachability through the planner is what made ``motion.probe`` 91% of tool
wall-clock in the v7 hanging-mug run.

They lived under ``backends/robotwin/`` only because that is where they were
first written. Nothing in them is RoboTwin-specific: a backend that can report
its end-effector pose at a chosen joint vector, and its joint limits, can use
them unchanged.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import numpy as np


class ReachabilityInputRejected(ValueError):
    """One reachability input is structurally invalid."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.details = dict(details)


def _reject(code: str, message: str, **details: Any) -> None:
    raise ReachabilityInputRejected(code, message, **details)


def _quaternion_matrix_xyzw(value: Sequence[float]) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        _reject("quaternion", "Cartesian pose quaternion is invalid")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        _reject("quaternion", "Cartesian pose quaternion has zero norm")
    x, y, z, w = quaternion / norm
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _rotation_vector(matrix: np.ndarray) -> np.ndarray:
    cosine = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    skew = np.asarray([
        matrix[2, 1] - matrix[1, 2],
        matrix[0, 2] - matrix[2, 0],
        matrix[1, 0] - matrix[0, 1],
    ], dtype=np.float64)
    if angle < 1e-8:
        return 0.5 * skew
    sine = math.sin(angle)
    if abs(sine) > 1e-6:
        return angle * skew / (2.0 * sine)
    # Stable near-pi axis extraction for large diagnostic rotations.
    diagonal = np.maximum(0.0, (np.diag(matrix) + 1.0) * 0.5)
    axis = np.sqrt(diagonal)
    if np.linalg.norm(axis) <= 1e-8:
        _reject("orientation_error", "Cartesian orientation error is singular")
    largest = int(np.argmax(axis))
    if largest == 0:
        axis[1] = math.copysign(axis[1], matrix[0, 1] + matrix[1, 0])
        axis[2] = math.copysign(axis[2], matrix[0, 2] + matrix[2, 0])
    elif largest == 1:
        axis[0] = math.copysign(axis[0], matrix[0, 1] + matrix[1, 0])
        axis[2] = math.copysign(axis[2], matrix[1, 2] + matrix[2, 1])
    else:
        axis[0] = math.copysign(axis[0], matrix[0, 2] + matrix[2, 0])
        axis[1] = math.copysign(axis[1], matrix[1, 2] + matrix[2, 1])
    axis /= np.linalg.norm(axis)
    return angle * axis


def cartesian_pose_error_twist(
    target_pose_xyzw: Sequence[float], current_pose_xyzw: Sequence[float],
) -> np.ndarray:
    """Return world-frame translation and rotation-vector error."""

    target = np.asarray(target_pose_xyzw, dtype=np.float64)
    current = np.asarray(current_pose_xyzw, dtype=np.float64)
    if (
        target.shape != (7,) or current.shape != (7,)
        or not np.isfinite(target).all() or not np.isfinite(current).all()
    ):
        _reject("pose", "Cartesian target/current pose is invalid")
    rotation = _quaternion_matrix_xyzw(target[3:]) @ (
        _quaternion_matrix_xyzw(current[3:]).T
    )
    return np.concatenate([target[:3] - current[:3], _rotation_vector(rotation)])


def numerical_cartesian_jacobian(
    joints: Sequence[float], *, fk_public_pose: Callable[[Sequence[float]], Sequence[float]],
    epsilon_rad: float,
) -> np.ndarray:
    """Numerically differentiate the public TCP pose in the world frame."""

    q = np.asarray(joints, dtype=np.float64)
    if q.ndim != 1 or not np.isfinite(q).all() or epsilon_rad <= 0.0:
        _reject("jacobian_input", "Cartesian Jacobian input is invalid")
    current = np.asarray(fk_public_pose(q), dtype=np.float64)
    columns: list[np.ndarray] = []
    for index in range(q.shape[0]):
        perturbed = q.copy()
        perturbed[index] += epsilon_rad
        pose = np.asarray(fk_public_pose(perturbed), dtype=np.float64)
        columns.append(cartesian_pose_error_twist(pose, current) / epsilon_rad)
    jacobian = np.stack(columns, axis=1)
    if jacobian.shape != (6, q.shape[0]) or not np.isfinite(jacobian).all():
        _reject("jacobian", "Cartesian numerical Jacobian is invalid")
    return jacobian


def bounded_dls_joint_delta(
    error_twist: Sequence[float], jacobian: Any, *, damping: float,
    maximum_translation_step_m: float, maximum_rotation_step_rad: float,
    maximum_joint_step_rad: float,
) -> np.ndarray:
    """Solve one bounded damped-least-squares Cartesian update."""

    error = np.asarray(error_twist, dtype=np.float64).copy()
    matrix = np.asarray(jacobian, dtype=np.float64)
    if error.shape != (6,) or matrix.ndim != 2 or matrix.shape[0] != 6:
        _reject("dls_shape", "Cartesian DLS input shape is invalid")
    if not np.isfinite(error).all() or not np.isfinite(matrix).all():
        _reject("dls_finite", "Cartesian DLS input is non-finite")
    for values, limit in ((error[:3], maximum_translation_step_m), (error[3:], maximum_rotation_step_rad)):
        norm = float(np.linalg.norm(values))
        if norm > limit:
            values *= float(limit) / norm
    regularized = matrix @ matrix.T + (float(damping) ** 2) * np.eye(6)
    try:
        delta = matrix.T @ np.linalg.solve(regularized, error)
    except np.linalg.LinAlgError as exc:
        _reject("dls_singular", "Cartesian DLS solve failed", error=str(exc))
    maximum = float(np.max(np.abs(delta))) if delta.size else 0.0
    if maximum > maximum_joint_step_rad:
        delta *= float(maximum_joint_step_rad) / maximum
    if not np.isfinite(delta).all():
        _reject("dls_non_finite", "Cartesian DLS produced non-finite joints")
    return delta


def solve_reachability(
    *, target_pose_xyzw: Sequence[float], seed_joints: Sequence[float],
    fk_public_pose: Callable[[Sequence[float]], Sequence[float]],
    joint_limits: Sequence[Sequence[float]],
    position_tolerance_m: float, orientation_tolerance_rad: float,
    maximum_iterations: int = 60, damping: float = 0.01,
    maximum_joint_step_rad: float = 0.15,
) -> dict[str, Any]:
    """Answer reachability with inverse kinematics and nothing else.

    The pieces were all present -- forward kinematics, a numerical Jacobian,
    joint limits, one damped-least-squares step -- and nothing composed them,
    so the only available answer to "can the arm reach this pose" came from the
    motion planner. Probing reachability that way is what made ``motion.probe``
    91% of tool wall-clock in the v7 hanging-mug run.

    This function takes forward kinematics and limits, never an adapter or a
    session, so it cannot reach ``plan_global`` or ``plan_batch_global`` even by
    accident. A solved pose is reachable by this model of the arm; it is not a
    claim that a collision-free path to it exists, and callers must say so.

    Refusal is reported, never raised and never silently clamped: a caller that
    reads ``reachable`` alone still gets a correct answer, and one that reads
    ``reason`` learns whether the limit was a joint stop, a singular
    configuration or an iteration budget.
    """

    limits = np.asarray(joint_limits, dtype=np.float64)
    joints = np.asarray(seed_joints, dtype=np.float64).copy()
    if (limits.ndim != 2 or limits.shape[1] != 2
            or limits.shape[0] != joints.shape[0]
            or not np.isfinite(limits).all() or not np.isfinite(joints).all()):
        _reject("reachability_input", "reachability solver input is invalid")
    if position_tolerance_m <= 0.0 or orientation_tolerance_rad <= 0.0:
        _reject("reachability_tolerance", "reachability tolerances must be positive")

    reason: str | None = None
    error = np.zeros(6, dtype=np.float64)
    iterations = 0
    for iterations in range(1, int(maximum_iterations) + 1):
        error = cartesian_pose_error_twist(
            target_pose_xyzw, fk_public_pose(joints),
        )
        position_error = float(np.linalg.norm(error[:3]))
        orientation_error = float(np.linalg.norm(error[3:]))
        if (position_error <= position_tolerance_m
                and orientation_error <= orientation_tolerance_rad):
            reason = None
            break
        jacobian = numerical_cartesian_jacobian(
            joints, fk_public_pose=fk_public_pose, epsilon_rad=1e-5,
        )
        if float(np.linalg.matrix_rank(jacobian)) < 1.0:
            reason = "singular"
            break
        delta = bounded_dls_joint_delta(
            error, jacobian, damping=damping,
            maximum_translation_step_m=max(position_tolerance_m * 10.0, 0.02),
            maximum_rotation_step_rad=max(orientation_tolerance_rad * 10.0, 0.1),
            maximum_joint_step_rad=maximum_joint_step_rad,
        )
        proposed = joints + delta
        clamped = np.clip(proposed, limits[:, 0], limits[:, 1])
        if np.allclose(clamped, joints, atol=1e-12):
            # Every remaining step is absorbed by a joint stop: the arm cannot
            # move further toward this pose, and reporting the clamped
            # configuration as a solution would be a false positive.
            joints = clamped
            reason = "joint_limit"
            break
        joints = clamped
    else:
        reason = "not_converged"

    error = cartesian_pose_error_twist(target_pose_xyzw, fk_public_pose(joints))
    position_error = float(np.linalg.norm(error[:3]))
    orientation_error = float(np.linalg.norm(error[3:]))
    solved = (
        reason is None
        and position_error <= position_tolerance_m
        and orientation_error <= orientation_tolerance_rad
    )
    result: dict[str, Any] = {
        "reachable": bool(solved),
        "joints": [float(value) for value in joints],
        "iterations": int(iterations),
        "position_error_m": position_error,
        "orientation_error_rad": orientation_error,
        "claim_boundary": (
            "Inverse kinematics for this arm model only: one damped-least-squares "
            "descent from the seed configuration, with no collision model of any "
            "kind. A reachable pose is not a collision-free path and not a "
            "successful motion, and this verdict is not the backend planner's."
        ),
    }
    if not solved:
        result["reason"] = reason or "not_converged"
    return result
