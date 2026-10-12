"""Backend-neutral joint-plan normalization and admission."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np


NORMALIZED_PLAN_RESULT_SCHEMA = "ora.normalized-plan-result.v1"
MOTION_PLAN_VALIDATION_SCHEMA = "ora.motion-plan-validation.v1"


class MotionPlanRejected(ValueError):
    """A plan failed before any controller target was written."""

    def __init__(self, code: str, message: str, *, details: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


@dataclass(frozen=True)
class ValidatedMotionPlan:
    """One admitted joint path and independent frame projections."""

    position: np.ndarray
    velocity: np.ndarray
    planned_control_pose_xyzw: np.ndarray
    planned_public_tcp_pose_xyzw: np.ndarray
    summary: dict[str, Any]

    @property
    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "planner_joint_position": self.position,
            "planner_joint_velocity": self.velocity,
            "planned_control_pose_xyzw": self.planned_control_pose_xyzw,
            "planned_public_tcp_pose_xyzw": self.planned_public_tcp_pose_xyzw,
        }


def _reject(code: str, message: str, **details: Any) -> None:
    raise MotionPlanRejected(code, message, details=details)


def pose_error(expected: Sequence[float], actual: Sequence[float]) -> dict[str, float]:
    target = np.asarray(expected, dtype=np.float64)
    measured = np.asarray(actual, dtype=np.float64)
    if target.shape != (7,) or measured.shape != (7,) or not (
        np.isfinite(target).all() and np.isfinite(measured).all()
    ):
        _reject("pose", "public TCP pose is invalid")
    left_norm = float(np.linalg.norm(target[3:]))
    right_norm = float(np.linalg.norm(measured[3:]))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        _reject("pose", "public TCP quaternion is invalid")
    dot = min(1.0, abs(float(np.dot(
        target[3:] / left_norm, measured[3:] / right_norm,
    ))))
    return {
        "translation_m": float(np.linalg.norm(target[:3] - measured[:3])),
        "orientation_rad": float(2.0 * math.acos(dot)),
    }


def normalize_joint_plan_result(
    result: Any, *, backend_id: str, source_schema: str,
    success_values: Sequence[str] = ("Success",),
) -> dict[str, Any]:
    """Normalize one backend planner result without discarding diagnostics."""

    if not isinstance(result, Mapping):
        _reject("malformed_result", "planner result is not an object")
    if result.get("schema") != source_schema:
        _reject("planner_schema", "planner result schema is unavailable or stale")
    status = str(result.get("status") or "invalid_result")
    raw_errors = result.get("errors")
    raw_timings = result.get("timings_s")
    raw_error = result.get("error")
    normalized = {
        "schema": NORMALIZED_PLAN_RESULT_SCHEMA,
        "backend_id": backend_id,
        "source_schema": source_schema,
        "success": status in set(success_values),
        "status": status,
        "stage": str(result.get("planner_status") or "unknown"),
        "valid_query": (
            bool(result["valid_query"])
            if result.get("valid_query") is not None else None
        ),
        "attempts": (
            int(result["attempts"]) if result.get("attempts") is not None else None
        ),
        "trajopt_attempts": (
            int(result["trajopt_attempts"])
            if result.get("trajopt_attempts") is not None else None
        ),
        "used_graph": (
            bool(result["used_graph"])
            if result.get("used_graph") is not None else None
        ),
        "goalset_index": (
            int(result["goalset_index"])
            if result.get("goalset_index") is not None else None
        ),
        "interpolation_dt_s": (
            float(result["interpolation_dt_s"])
            if result.get("interpolation_dt_s") is not None else None
        ),
        "joint_names": list(result.get("joint_names") or []),
        "position": result.get("position"),
        "velocity": result.get("velocity"),
        "errors": dict(raw_errors) if isinstance(raw_errors, Mapping) else {},
        "timings_s": dict(raw_timings) if isinstance(raw_timings, Mapping) else {},
        "error": dict(raw_error) if isinstance(raw_error, Mapping) else None,
        "ik": (
            dict(result["ik"])
            if isinstance(result.get("ik"), Mapping) else None
        ),
    }
    return normalized


def planner_public_summary(result: Any) -> dict[str, Any]:
    """Return the bounded diagnostics exposed in public receipts."""

    if not isinstance(result, Mapping) or result.get("schema") != NORMALIZED_PLAN_RESULT_SCHEMA:
        return {
            "called": True, "status": "invalid_result", "stage": "planner_contract",
            "valid_query": None, "attempts": None, "trajopt_attempts": None,
            "used_graph": None, "goalset_index": None, "path_steps": 0,
            "interpolation_dt_s": None, "errors": {}, "timings_s": {},
            "ik": None, "error": None,
        }
    position = result.get("position")
    shape = getattr(position, "shape", None)
    if shape is None:
        try:
            shape = np.asarray(position).shape
        except Exception:
            shape = ()
    steps = int(shape[0]) if shape and len(shape) == 2 else 0

    def finite_json(value: Any) -> Any:
        if value is None:
            return None
        try:
            array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(array).all():
            return None
        return float(array) if array.ndim == 0 else array.tolist()

    timings = {
        str(key): finite_json(value)
        for key, value in dict(result.get("timings_s") or {}).items()
    }
    errors = {
        str(key): finite_json(value)
        for key, value in dict(result.get("errors") or {}).items()
    }
    ik = result.get("ik")
    public_ik = None
    if isinstance(ik, Mapping):
        public_ik = {
            key: ik.get(key)
            for key in (
                "status", "seed_count", "valid_solution_count",
                "selected_solution_index", "selected_distance_rad",
            )
            if key in ik
        }
    raw_error = result.get("error")
    public_error = None
    if isinstance(raw_error, Mapping):
        public_error = {
            "stage": str(raw_error.get("stage") or result.get("stage") or "unknown"),
            "type": str(raw_error.get("type") or "PlannerError"),
            "message": str(raw_error.get("message") or "planner error"),
        }
    return {
        "called": True,
        "status": str(result.get("status") or "invalid_result"),
        "stage": str(result.get("stage") or "unknown"),
        "valid_query": result.get("valid_query"),
        "attempts": result.get("attempts"),
        "trajopt_attempts": result.get("trajopt_attempts"),
        "used_graph": result.get("used_graph"),
        "goalset_index": result.get("goalset_index"),
        "path_steps": steps,
        "interpolation_dt_s": finite_json(result.get("interpolation_dt_s")),
        "errors": errors,
        "timings_s": timings,
        "ik": public_ik,
        "error": public_error,
    }


def validate_joint_plan(
    result: Any, *, expected_joint_names: Sequence[str],
    measured_start_joints: Sequence[float],
    measured_start_pose_xyzw: Sequence[float],
    target_pose_xyzw: Sequence[float],
    joint_limits: Sequence[Sequence[float]],
    fk_public_pose: Callable[[Sequence[float]], Sequence[float]],
    fk_control_pose: Callable[[Sequence[float]], Sequence[float]],
    workspace_bounds_m: Mapping[str, Sequence[float]],
    profile: Mapping[str, Any],
) -> ValidatedMotionPlan:
    """Admit a normalized joint plan before the first plant write."""

    if not isinstance(result, Mapping) or result.get("schema") != NORMALIZED_PLAN_RESULT_SCHEMA:
        _reject("planner_schema", "normalized planner result is unavailable")
    if result.get("success") is not True:
        _reject(
            "planner_failed", "planner did not produce a successful trajectory",
            planner=planner_public_summary(result),
        )
    names = result.get("joint_names")
    if names != list(expected_joint_names):
        _reject(
            "joint_order", "planner joint order differs from the embodiment profile",
            expected=list(expected_joint_names), actual=names,
        )
    position = np.asarray(result.get("position"), dtype=np.float64)
    velocity = np.asarray(result.get("velocity"), dtype=np.float64)
    dof = len(expected_joint_names)
    if position.ndim != 2 or position.shape[1] != dof:
        _reject("position_shape", "planner position trajectory shape is invalid")
    if not 1 <= len(position) <= int(profile["maximum_waypoints"]):
        _reject("waypoint_count", "planner waypoint count is outside the profile")
    if velocity.shape != position.shape:
        _reject("velocity_shape", "planner velocity trajectory shape is invalid")
    if not np.isfinite(position).all() or not np.isfinite(velocity).all():
        _reject("non_finite_plan", "planner trajectory contains non-finite values")
    start = np.asarray(measured_start_joints, dtype=np.float64)
    if start.shape != (dof,) or not np.isfinite(start).all():
        _reject("measured_start", "measured action-start joints are invalid")
    start_error = float(np.max(np.abs(position[0] - start)))
    if start_error > float(profile["maximum_start_joint_error_rad"]):
        _reject(
            "start_joint_error", "planner path does not begin at action-start joints",
            maximum_error_rad=start_error,
        )
    limits = np.asarray(joint_limits, dtype=np.float64)
    if limits.shape != (dof, 2) or not np.isfinite(limits).all():
        _reject("joint_limits", "embodiment joint limits are invalid")
    epsilon = float(profile["joint_limit_epsilon_rad"])
    if np.any(position < limits[:, 0] - epsilon) or np.any(position > limits[:, 1] + epsilon):
        _reject("joint_limit", "planner path leaves an embodiment joint limit")

    planned_public = np.asarray([fk_public_pose(row) for row in position], dtype=np.float64)
    planned_control = np.asarray([fk_control_pose(row) for row in position], dtype=np.float64)
    for label, array in (("public", planned_public), ("control", planned_control)):
        if array.shape != (len(position), 7) or not np.isfinite(array).all():
            _reject("planned_fk", f"planned {label} FK is invalid")
        norms = np.linalg.norm(array[:, 3:], axis=1)
        if np.any(norms <= 1e-12):
            _reject("planned_fk_orientation", f"planned {label} FK quaternion is invalid")
        array[:, 3:] /= norms[:, None]

    start_pose_error = pose_error(measured_start_pose_xyzw, planned_public[0])
    if (
        start_pose_error["translation_m"] > float(profile["maximum_public_tcp_start_error_m"])
        or start_pose_error["orientation_rad"]
        > float(profile["maximum_public_tcp_start_orientation_error_rad"])
    ):
        _reject("public_tcp_error", "planned FK differs at action start", **start_pose_error)
    margin = float(profile["workspace_margin_m"])
    for index, axis in enumerate(("x", "y", "z")):
        interval = workspace_bounds_m.get(axis)
        if not isinstance(interval, Sequence) or len(interval) != 2:
            _reject("workspace_profile", f"workspace bound {axis} is invalid")
        values = planned_public[:, index]
        if np.any(values < float(interval[0]) - margin) or np.any(
            values > float(interval[1]) + margin
        ):
            _reject("workspace_path", f"planned TCP leaves the {axis} workspace")
    maximum_step = (
        float(np.max(np.linalg.norm(np.diff(planned_public[:, :3], axis=0), axis=1)))
        if len(planned_public) > 1 else 0.0
    )
    if maximum_step > float(profile["maximum_tcp_step_m"]):
        _reject("planned_tcp_step", "planned TCP path has a gross step", maximum_step_m=maximum_step)
    endpoint_error = pose_error(target_pose_xyzw, planned_public[-1])
    if (
        endpoint_error["translation_m"] > float(profile["maximum_endpoint_error_m"])
        or endpoint_error["orientation_rad"]
        > float(profile["maximum_endpoint_orientation_error_rad"])
    ):
        _reject(
            "planning_endpoint_error", "planned FK misses requested public TCP",
            **endpoint_error,
        )
    summary = {
        "schema": MOTION_PLAN_VALIDATION_SCHEMA,
        "status": "accepted",
        "backend_id": result["backend_id"],
        "joint_order": list(expected_joint_names),
        "waypoints": int(len(position)),
        "maximum_start_joint_error_rad": round(start_error, 8),
        "maximum_planned_tcp_step_m": round(maximum_step, 8),
        "public_tcp_start_error": {
            key: round(value, 8) for key, value in start_pose_error.items()
        },
        "planning_endpoint_error": {
            key: round(value, 8) for key, value in endpoint_error.items()
        },
        "planner": planner_public_summary(result),
    }
    return ValidatedMotionPlan(
        position=position, velocity=velocity,
        planned_control_pose_xyzw=planned_control,
        planned_public_tcp_pose_xyzw=planned_public,
        summary=summary,
    )


__all__ = [
    "MOTION_PLAN_VALIDATION_SCHEMA", "NORMALIZED_PLAN_RESULT_SCHEMA",
    "MotionPlanRejected", "ValidatedMotionPlan", "normalize_joint_plan_result",
    "planner_public_summary", "pose_error", "validate_joint_plan",
]
