"""Pure validation and evidence helpers for source-bound RoboTwin plans."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ora.environment.controllers.plan_validation import (
    MotionPlanRejected,
    normalize_joint_plan_result,
    validate_joint_plan,
)
from ora.environment.controllers.inter_arm_clearance import (
    InterArmClearancePolicy, minimum_inter_arm_distance,
)


PLAN_RESULT_SCHEMA = "robotwin.curobo-plan-result.v1"
TOPK_PLAN_RESULT_SCHEMA = "robotwin.curobo-plan-result.v2"
PLAN_ARTIFACT_SCHEMA = "ora.robotwin-plan-artifact.v1"
PLAN_REFERENCE_SCHEMA = "ora.robotwin-plan-reference.v1"


# These moved to the shared controller layer once a second backend needed them.
# Re-exported so every existing importer keeps working unchanged.
from ora.environment.controllers.reachability import (  # noqa: F401
    _quaternion_matrix_xyzw,
    _rotation_vector,
    bounded_dls_joint_delta,
    cartesian_pose_error_twist,
    numerical_cartesian_jacobian,
    solve_reachability,
)


class RoboTwinPlanRejected(ValueError):
    """A planner result failed before any controller target was written."""

    def __init__(self, code: str, message: str, *, details: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


@dataclass(frozen=True)
class ValidatedRoboTwinPlan:
    """One validated single-arm plan and its independent FK projection."""

    position: np.ndarray
    velocity: np.ndarray
    planned_link6_pose_xyzw: np.ndarray
    planned_tcp_pose_xyzw: np.ndarray
    summary: dict[str, Any]

    @property
    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "planner_joint_position": self.position,
            "planner_joint_velocity": self.velocity,
            "planned_link6_pose_xyzw": self.planned_link6_pose_xyzw,
            "planned_public_tcp_pose_xyzw": self.planned_tcp_pose_xyzw,
        }


def public_tcp_point_jacobian(
    spatial_jacobian_world: Any, point_offset_world_m: Sequence[float],
) -> np.ndarray:
    """Shift a world-frame spatial Jacobian to the public TCP point.

    The pinned SAPIEN Pinocchio binding exposes a spatial velocity about the
    articulation-root origin.  For a point displaced by ``r`` from that
    spatial origin, its linear velocity is ``v + omega x r`` while angular
    velocity is unchanged. Rows use ``[linear, angular]`` order.
    """

    matrix = np.asarray(spatial_jacobian_world, dtype=np.float64)
    offset = np.asarray(point_offset_world_m, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != 6 or offset.shape != (3,):
        _reject("analytic_jacobian_shape", "analytic TCP Jacobian input is invalid")
    if not np.isfinite(matrix).all() or not np.isfinite(offset).all():
        _reject("analytic_jacobian_finite", "analytic TCP Jacobian input is non-finite")
    skew = np.asarray([
        [0.0, -offset[2], offset[1]],
        [offset[2], 0.0, -offset[0]],
        [-offset[1], offset[0], 0.0],
    ], dtype=np.float64)
    linear = matrix[:3] - skew @ matrix[3:]
    shifted = np.vstack((linear, matrix[3:]))
    if not np.isfinite(shifted).all():
        _reject("analytic_jacobian_finite", "analytic TCP Jacobian is non-finite")
    return shifted


def weighted_svd_dls_joint_delta(
    error_twist: Sequence[float], jacobian: Any, *,
    current_joints: Sequence[float], joint_limits: Sequence[Sequence[float]],
    base_damping: float, singular_value_threshold: float,
    maximum_damping: float, joint_limit_margin_rad: float,
    minimum_joint_mobility: float, maximum_translation_step_m: float,
    maximum_rotation_step_rad: float, maximum_joint_step_rad: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Solve one unit-balanced, limit-weighted adaptive SVD DLS update."""

    error = np.asarray(error_twist, dtype=np.float64).copy()
    matrix = np.asarray(jacobian, dtype=np.float64)
    joints = np.asarray(current_joints, dtype=np.float64)
    limits = np.asarray(joint_limits, dtype=np.float64)
    if (
        error.shape != (6,) or matrix.shape != (6, joints.size)
        or limits.shape != (joints.size, 2)
        or not np.isfinite(error).all() or not np.isfinite(matrix).all()
        or not np.isfinite(joints).all() or not np.isfinite(limits).all()
    ):
        _reject("weighted_dls_input", "weighted Cartesian DLS input is invalid")
    positive = (
        base_damping, singular_value_threshold, maximum_damping,
        joint_limit_margin_rad, minimum_joint_mobility,
        maximum_translation_step_m, maximum_rotation_step_rad,
        maximum_joint_step_rad,
    )
    if any(float(value) <= 0.0 for value in positive):
        _reject("weighted_dls_profile", "weighted Cartesian DLS bounds are invalid")
    if base_damping > maximum_damping or minimum_joint_mobility > 1.0:
        _reject("weighted_dls_profile", "weighted Cartesian DLS damping is invalid")

    for values, bound in (
        (error[:3], maximum_translation_step_m),
        (error[3:], maximum_rotation_step_rad),
    ):
        norm = float(np.linalg.norm(values))
        if norm > float(bound):
            values *= float(bound) / norm

    row_scale = np.asarray([
        *([1.0 / float(maximum_translation_step_m)] * 3),
        *([1.0 / float(maximum_rotation_step_rad)] * 3),
    ], dtype=np.float64)
    lower_distance = joints - limits[:, 0]
    upper_distance = limits[:, 1] - joints
    nearest_limit = np.minimum(lower_distance, upper_distance)
    mobility = np.clip(
        nearest_limit / float(joint_limit_margin_rad),
        float(minimum_joint_mobility), 1.0,
    )
    scaled_matrix = (row_scale[:, None] * matrix) * mobility[None, :]
    scaled_error = row_scale * error
    try:
        left, singular, right_t = np.linalg.svd(scaled_matrix, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        _reject("weighted_dls_svd", "weighted Cartesian SVD failed", error=str(exc))
    minimum_singular = float(np.min(singular)) if singular.size else 0.0
    damping = min(
        float(maximum_damping),
        float(base_damping) * max(
            1.0,
            (float(singular_value_threshold) / max(minimum_singular, 1e-9)) ** 2,
        ),
    )
    gains = singular / (singular * singular + damping * damping)
    weighted_delta = right_t.T @ (gains * (left.T @ scaled_error))
    delta = mobility * weighted_delta
    maximum = float(np.max(np.abs(delta))) if delta.size else 0.0
    if maximum > float(maximum_joint_step_rad):
        delta *= float(maximum_joint_step_rad) / maximum
    if not np.isfinite(delta).all():
        _reject("weighted_dls_non_finite", "weighted Cartesian DLS is non-finite")
    return delta, {
        "minimum_singular_value": minimum_singular,
        "adaptive_damping": damping,
        "minimum_joint_mobility": float(np.min(mobility)),
        "maximum_joint_delta_rad": (
            float(np.max(np.abs(delta))) if delta.size else 0.0
        ),
    }


def select_topk_validated_plan(
    result: Any, *,
    validator: Callable[[Mapping[str, Any]], ValidatedRoboTwinPlan],
) -> tuple[dict[str, Any], ValidatedRoboTwinPlan, dict[str, Any]]:
    """Validate all successful top-K trajectories and select deterministically."""

    if not isinstance(result, Mapping) or result.get("schema") != TOPK_PLAN_RESULT_SCHEMA:
        validated = validator(result)
        return dict(result), validated, {
            "schema": "ora.robotwin-topk-plan-selection.v1",
            "candidate_count": 1,
            "admitted_count": 1,
            "selected_candidate_index": None,
            "candidates": [],
        }
    candidates = result.get("candidates")
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 16:
        _reject("topk_candidates", "top-K planner candidates are invalid")
    admitted: list[
        tuple[tuple[float, float, float, int], dict[str, Any], ValidatedRoboTwinPlan]
    ] = []
    diagnostics: list[dict[str, Any]] = []
    for ordinal, raw in enumerate(candidates):
        if not isinstance(raw, Mapping):
            _reject("topk_candidate", "top-K planner candidate is malformed", index=ordinal)
        candidate_index = raw.get("ik_candidate_index", ordinal)
        if isinstance(candidate_index, bool) or not isinstance(candidate_index, int):
            _reject("topk_candidate", "top-K candidate index is invalid", index=ordinal)
        record = {
            **dict(raw),
            "schema": PLAN_RESULT_SCHEMA,
            "joint_names": list(result.get("joint_names") or raw.get("joint_names") or []),
        }
        diagnostic: dict[str, Any] = {
            "candidate_index": candidate_index,
            "ik_distance_rad": raw.get("ik_distance_rad"),
            "planning_attempt": raw.get("planning_attempt"),
            "planner_status": raw.get("planner_status"),
            "status": raw.get("status"),
            "failure_stages": [
                {
                    "planning_attempt": item.get("planning_attempt"),
                    "planner_status": item.get("planner_status"),
                }
                for item in raw.get("failed_attempts", [])
                if isinstance(item, Mapping)
            ][:4],
        }
        if raw.get("status") != "Success":
            diagnostic["admission"] = "planner_failed"
            diagnostics.append(diagnostic)
            continue
        try:
            validated = validator(record)
        except RoboTwinPlanRejected as exc:
            diagnostic.update({
                "admission": "plan_rejected",
                "rejection": {"code": exc.code, "details": exc.details},
            })
            diagnostics.append(diagnostic)
            continue
        endpoint = validated.summary["planning_endpoint_error"]
        interpolation_dt = raw.get("interpolation_dt_s")
        timed = (
            isinstance(interpolation_dt, (int, float))
            and not isinstance(interpolation_dt, bool)
            and math.isfinite(float(interpolation_dt))
            and float(interpolation_dt) > 0.0
        )
        # Untimed plans rank strictly behind every honestly timed plan
        # instead of comparing a waypoint count against seconds.
        duration = (
            float(interpolation_dt) * int(validated.position.shape[0])
            if timed else float(validated.position.shape[0])
        )
        diagnostic.update({
            "admission": "accepted",
            "execution_duration_s": duration,
            "planning_endpoint_error": dict(endpoint),
            "waypoints": int(validated.position.shape[0]),
        })
        diagnostics.append(diagnostic)
        admitted.append((
            (
                0 if timed else 1,
                duration,
                float(endpoint["translation_m"]),
                float(endpoint["orientation_rad"]),
                candidate_index,
            ),
            record,
            validated,
        ))
    if not admitted:
        _reject(
            "topk_no_admitted_plan",
            "no successful top-K trajectory passed plan admission",
            candidates=diagnostics,
        )
    admitted.sort(key=lambda item: item[0])
    _key, selected_record, selected_validated = admitted[0]
    selected_index = int(selected_record.get("ik_candidate_index", 0))
    selected_record["topk_selection"] = {
        "candidate_count": len(candidates),
        "admitted_count": len(admitted),
        "selected_candidate_index": selected_index,
        "selection_order": [
            "validator_pass", "shortest_execution_time",
            "minimum_endpoint_fk_residual", "ik_candidate_index",
        ],
        "candidates": diagnostics,
        "ik": result.get("ik"),
    }
    if isinstance(result.get("ik_validation_artifact"), Mapping):
        selected_record["ik_validation_artifact"] = dict(result["ik_validation_artifact"])
    return selected_record, selected_validated, {
        "schema": "ora.robotwin-topk-plan-selection.v1",
        **selected_record["topk_selection"],
    }


def _reject(code: str, message: str, **details: Any) -> None:
    raise RoboTwinPlanRejected(code, message, details=details)


def select_topk_validated_plan_set(
    results: Mapping[str, Any], *,
    validators: Mapping[str, Callable[[Mapping[str, Any]], ValidatedRoboTwinPlan]],
    set_validator: Callable[[Mapping[str, ValidatedRoboTwinPlan]], Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, ValidatedRoboTwinPlan], dict[str, Any]]:
    """Admit complete synchronized candidate sets before selecting a trajectory.

    Per-arm FK admission cannot establish inter-arm or payload clearance.
    In particular, selecting each arm's fastest path before testing the pair
    may reject a request whose alternate IK branch has a safe paired path.
    The source-bounded top-K sizes cap this Cartesian product at 256 pairs.
    """

    arms = tuple(results)
    if not 1 <= len(arms) <= 2 or set(validators) != set(arms):
        _reject("topk_plan_set", "plan set must bind one or two arm validators")
    candidates: dict[str, list[tuple[dict[str, Any], ValidatedRoboTwinPlan]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for arm in arms:
        collected: list[tuple[dict[str, Any], ValidatedRoboTwinPlan]] = []

        def collect(raw: Mapping[str, Any]) -> ValidatedRoboTwinPlan:
            validated = validators[arm](raw)
            collected.append((dict(raw), validated))
            return validated

        _raw, _plan, summary = select_topk_validated_plan(results[arm], validator=collect)
        candidates[arm] = collected
        summaries[arm] = summary

    admitted = []
    rejections: list[dict[str, Any]] = []
    last_rejection: RoboTwinPlanRejected | None = None
    for combination in product(*(candidates[arm] for arm in arms)):
        records = {arm: item[0] for arm, item in zip(arms, combination)}
        plans = {arm: item[1] for arm, item in zip(arms, combination)}
        indices = {arm: int(records[arm].get("ik_candidate_index", 0)) for arm in arms}
        try:
            admission = dict(set_validator(plans))
        except RoboTwinPlanRejected as exc:
            last_rejection = exc
            rejections.append({"candidate_indices": indices, "code": exc.code, "details": exc.details})
            continue
        durations = []
        untimed = False
        for arm in arms:
            dt = records[arm].get("interpolation_dt_s")
            timed = isinstance(dt, (int, float)) and not isinstance(dt, bool) and math.isfinite(dt) and dt > 0
            untimed |= not timed
            durations.append(len(plans[arm].position) * (float(dt) if timed else 1.0))
        rank = (
            int(untimed), max(durations),
            sum(float(plans[a].summary["planning_endpoint_error"]["translation_m"]) for a in arms),
            sum(float(plans[a].summary["planning_endpoint_error"]["orientation_rad"]) for a in arms),
            tuple(indices[a] for a in arms),
        )
        admitted.append((rank, records, plans, admission))
    if not admitted:
        if len(rejections) == 1 and last_rejection is not None:
            raise last_rejection
        _reject("topk_no_admitted_plan", "no candidate combination passed full admission", candidates=rejections)
    _rank, selected, plans, admission = min(admitted, key=lambda item: item[0])
    for arm in arms:
        if results[arm].get("schema") != TOPK_PLAN_RESULT_SCHEMA:
            continue
        selected[arm] = {
            **selected[arm],
            "topk_selection": {
                **{key: value for key, value in summaries[arm].items() if key != "schema"},
                "selected_candidate_index": int(selected[arm].get("ik_candidate_index", 0)),
                "admitted_count": len({
                    int(item[1][arm].get("ik_candidate_index", 0)) for item in admitted
                }),
                "selection_order": ["full_pair_admission", "shortest_execution_time", "minimum_endpoint_fk_residual", "ik_candidate_index"],
                "admitted_combination_count": len(admitted),
                "combination_rejections": rejections,
            },
        }
    return selected, plans, admission


def ik_planning_diagnostics(result: Mapping[str, Any]) -> dict[str, Any] | None:
    """Compact measured IK filter evidence; absence is never zero or collision."""
    topk = result.get("topk_selection")
    source = result.get("ik")
    if not isinstance(source, Mapping) and isinstance(topk, Mapping):
        source = topk.get("ik")
    if not isinstance(source, Mapping):
        if result.get("planner_status") != "IK_FAIL":
            return None
        source = {}

    def count(value: Any) -> int | None:
        return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None

    counts = {name: count(source.get(key)) for name, key in (
        ("requested", "seed_count"), ("returned", "returned_solution_count"),
        ("retained", "valid_solution_count"), ("deduplicated", "deduplicated_solution_count"),
        ("planned", "planned_candidate_count"),
    )}
    raw_rejections = source.get("constraint_rejection_counts")
    raw_rejections = raw_rejections if isinstance(raw_rejections, Mapping) else {}
    rejections = {key: count(raw_rejections.get(key)) for key in (
        "non_finite_solution", "solver_unsuccessful", "joint_limit",
    )}
    rows = source.get("candidate_validation")
    detail = isinstance(rows, list) and len(rows) == counts["returned"]
    outcome = ("no_valid_endpoint" if counts["retained"] == 0 else
               "candidates_retained" if counts["retained"] is not None else "unknown")
    return {
        "phase": "ik_candidate_validation", "outcome": outcome,
        "candidate_counts": counts, "rejection_counts": rejections,
        "details_available": detail, "unknown_reasons": ["solver_failure_detail"],
        "recovery": [("No candidate survived this bounded exact-pose IK solve. "
            "Solver failure does not identify collision or prove the target unreachable. "
            "Inspect the reported filters and consider a different exact pose or intermediate pose."
            if outcome == "no_valid_endpoint" else
            "IK candidates remain subject to trajectory and execution admission."
            if outcome == "candidates_retained" else
            "Candidate filter details were not recorded; the cause is unknown. "
            "Consider a different exact pose or intermediate pose; failure does not prove unreachability.")],
    }


def planner_public_summary(
    result: Any, *, include_topk_selection: bool = True,
    include_constraint_rejection_counts: bool = False,
) -> dict[str, Any]:
    """Return the bounded diagnostic subset allowed in a policy receipt."""

    if not isinstance(result, Mapping):
        result = {
            "status": "invalid_result",
            "planner_status": "planner_contract",
        }
    positions = result.get("position")
    shape = getattr(positions, "shape", None)
    steps = int(shape[0]) if shape is not None and len(shape) else 0
    status = result.get("status")
    raw_errors = result.get("errors")
    errors = dict(raw_errors) if isinstance(raw_errors, Mapping) else {}
    raw_timings = result.get("timings_s")
    timings = dict(raw_timings) if isinstance(raw_timings, Mapping) else {}
    raw_error = result.get("error")
    error = (
        {
            "stage": str(raw_error.get("stage") or result.get("planner_status") or "unknown"),
            "type": str(raw_error.get("type") or "PlannerError"),
            "message": str(raw_error.get("message") or "planner error"),
        }
        if isinstance(raw_error, Mapping)
        else None
    )

    def optional_scalar(value: Any, expected: type) -> Any:
        if value is None:
            return None
        if expected is bool:
            return bool(value)
        if expected is int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
        return value

    def numeric_or_list(value: Any) -> float | list[float] | None:
        if value is None:
            return None
        array = np.asarray(value, dtype=np.float64)
        if not np.isfinite(array).all():
            return None
        if array.ndim == 0:
            return float(array)
        return [float(item) for item in array.reshape(-1)]

    def optional_seconds(name: str) -> float | None:
        value = timings.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = float(value)
            return value if math.isfinite(value) and value >= 0.0 else None
        return None

    record: dict[str, Any] = {
        "called": True,
        "status": str(status) if status is not None else "invalid_result",
        "stage": str(result.get("planner_status") or "unknown"),
        "valid_query": optional_scalar(result.get("valid_query"), bool),
        "attempts": optional_scalar(result.get("attempts"), int),
        "trajopt_attempts": optional_scalar(result.get("trajopt_attempts"), int),
        "used_graph": optional_scalar(result.get("used_graph"), bool),
        "goalset_index": optional_scalar(result.get("goalset_index"), int),
        "path_steps": steps,
        "interpolation_dt_s": (
            float(result["interpolation_dt_s"])
            if isinstance(result.get("interpolation_dt_s"), (int, float))
            and not isinstance(result.get("interpolation_dt_s"), bool)
            and math.isfinite(float(result["interpolation_dt_s"]))
            and float(result["interpolation_dt_s"]) >= 0.0 else None
        ),
        "errors": {
            "position": numeric_or_list(errors.get("position")),
            "rotation": numeric_or_list(errors.get("rotation")),
            "cspace": numeric_or_list(errors.get("cspace")),
        },
        "timings_s": {
            name: optional_seconds(name)
            for name in ("solve", "ik", "graph", "trajopt", "finetune", "total")
        },
        "error": error,
    }
    topk = result.get("topk_selection")
    if include_constraint_rejection_counts:
        ik = result.get("ik")
        if not isinstance(ik, Mapping) and isinstance(topk, Mapping):
            ik = topk.get("ik")
        raw_counts = ik.get("constraint_rejection_counts") if isinstance(ik, Mapping) else None
        if isinstance(raw_counts, Mapping):
            counts = {
                key: int(raw_counts[key])
                for key in ("non_finite_solution", "solver_unsuccessful", "joint_limit")
                if isinstance(raw_counts.get(key), int)
                and not isinstance(raw_counts[key], bool) and raw_counts[key] >= 0
            }
            if counts:
                record["constraint_rejection_counts"] = counts
    if (
        not isinstance(topk, Mapping)
        and result.get("schema") == TOPK_PLAN_RESULT_SCHEMA
        and isinstance(result.get("candidates"), list)
    ):
        # An all-candidate planner failure never reaches plan admission, so it
        # has no selected-plan ``topk_selection`` record.  Preserve the same
        # bounded candidate outcomes in the public receipt instead of
        # collapsing a rich source-bound failure to the unhelpful string
        # ``Fail``.  Successful candidates remain explicitly unvalidated here;
        # only ``select_topk_validated_plan`` may call one accepted.
        raw_candidates = [
            item for item in result["candidates"] if isinstance(item, Mapping)
        ]
        topk = {
            "candidate_count": len(raw_candidates),
            "admitted_count": 0,
            "selected_candidate_index": None,
            "candidates": [
                {
                    "candidate_index": item.get("ik_candidate_index", ordinal),
                    "status": item.get("status"),
                    "admission": (
                        "planner_failed"
                        if item.get("status") != "Success"
                        else "not_admitted"
                    ),
                    "planning_attempt": item.get("planning_attempt"),
                    "planner_status": item.get("planner_status"),
                    "ik_distance_rad": item.get("ik_distance_rad"),
                    "failure_stages": [
                        {
                            "planning_attempt": stage.get("planning_attempt"),
                            "planner_status": stage.get("planner_status"),
                        }
                        for stage in item.get("failed_attempts", [])
                        if isinstance(stage, Mapping)
                    ][:4],
                }
                for ordinal, item in enumerate(raw_candidates)
            ],
        }
    if include_topk_selection and isinstance(topk, Mapping):
        record["topk_selection"] = {
            "candidate_count": int(topk.get("candidate_count", 0)),
            "admitted_count": int(topk.get("admitted_count", 0)),
            "selected_candidate_index": optional_scalar(
                topk.get("selected_candidate_index"), int,
            ),
            "candidate_outcomes": [
                {
                    "candidate_index": optional_scalar(
                        item.get("candidate_index"), int,
                    ),
                    "status": str(item.get("status") or "invalid"),
                    "admission": str(item.get("admission") or "unknown"),
                    "planning_attempt": str(
                        item.get("planning_attempt") or "unknown"
                    ),
                    "planner_status": str(
                        item.get("planner_status") or "unknown"
                    ),
                    "ik_distance_rad": optional_scalar(
                        item.get("ik_distance_rad"), float,
                    ),
                    "failure_stages": [
                        {
                            "planning_attempt": str(
                                stage.get("planning_attempt") or "unknown"
                            ),
                            "planner_status": str(
                                stage.get("planner_status") or "unknown"
                            ),
                        }
                        for stage in item.get("failure_stages", [])
                        if isinstance(stage, Mapping)
                    ][:4],
                    **({
                        "rejection_code": str(item["rejection"].get("code")),
                    } if isinstance(item.get("rejection"), Mapping) else {}),
                }
                for item in topk.get("candidates", [])
                if isinstance(item, Mapping)
            ],
        }
    diagnostics = ik_planning_diagnostics(result)
    if diagnostics is not None:
        record["planning_diagnostics"] = diagnostics
    reference = result.get("ik_validation_artifact")
    if isinstance(reference, Mapping):
        record["ik_validation_artifact"] = {
            key: reference[key] for key in ("path", "sha256", "schema", "bytes") if key in reference
        }
    return record


def split_batch_plan_result(result: Any, index: int) -> dict[str, Any]:
    """Project one CuRobo batch row into the scalar plan-result contract."""

    if not isinstance(result, Mapping) or result.get("schema") != PLAN_RESULT_SCHEMA:
        _reject("planner_schema", "batch planner result schema is unavailable")
    statuses = result.get("status")
    if not isinstance(statuses, list) or not statuses:
        _reject("batch_status", "batch planner status must be a non-empty list")
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(statuses):
        _reject("batch_index", "batch planner index is outside the result")
    raw_position = result.get("position")
    raw_velocity = result.get("velocity")
    if (
        isinstance(raw_position, list)
        and isinstance(raw_velocity, list)
        and len(raw_position) == len(statuses)
        and len(raw_velocity) == len(statuses)
    ):
        selected_position = raw_position[index]
        selected_velocity = raw_velocity[index]
        if selected_position is None or selected_velocity is None:
            position_row = None
            velocity_row = None
        else:
            position_row = np.asarray(selected_position, dtype=np.float64)
            velocity_row = np.asarray(selected_velocity, dtype=np.float64)
            if position_row.ndim != 2 or velocity_row.shape != position_row.shape:
                _reject(
                    "batch_trajectory_shape",
                    "selected batch trajectory shape is invalid",
                    position_shape=list(position_row.shape),
                    velocity_shape=list(velocity_row.shape),
                )
        position = velocity = None
    else:
        position = np.asarray(raw_position, dtype=np.float64)
        velocity = np.asarray(raw_velocity, dtype=np.float64)
        position_row = velocity_row = None
    if position is not None and (
        len(statuses) == 1
        and position.ndim == 2
        and velocity.shape == position.shape
    ):
        position = position[None, ...]
        velocity = velocity[None, ...]
    if position is not None and (
        position.ndim != 3
        or velocity.shape != position.shape
        or position.shape[0] != len(statuses)
    ):
        if statuses[index] == "Success":
            _reject(
                "batch_trajectory_shape",
                "successful batch trajectory shape is invalid",
                status_count=len(statuses),
                position_shape=list(position.shape),
                velocity_shape=list(velocity.shape),
            )
    elif position is not None:
        position_row = position[index]
        velocity_row = velocity[index]

    def select(value: Any) -> Any:
        if isinstance(value, list) and len(value) == len(statuses):
            return value[index]
        return value

    projected = {
        **dict(result),
        "status": statuses[index],
        "planner_status": select(result.get("planner_status")),
        "valid_query": select(result.get("valid_query")),
        "attempts": select(result.get("attempts")),
        "trajopt_attempts": select(result.get("trajopt_attempts")),
        "used_graph": select(result.get("used_graph")),
        "goalset_index": index,
        "path_buffer_last_tstep": select(result.get("path_buffer_last_tstep")),
        "interpolation_dt_s": select(result.get("interpolation_dt_s")),
        "timings_s": {
            key: select(value)
            for key, value in dict(result.get("timings_s") or {}).items()
        },
        "errors": {
            key: select(value)
            for key, value in dict(result.get("errors") or {}).items()
        },
        "ik": select(result.get("ik")),
        "error": select(result.get("error")),
    }
    if position_row is None:
        projected.pop("position", None)
        projected.pop("velocity", None)
    else:
        projected["position"] = position_row
        projected["velocity"] = velocity_row
    return projected


def validate_single_arm_plan(
    result: Any,
    *,
    expected_joint_names: Sequence[str],
    measured_start_joints: Sequence[float],
    measured_start_pose_xyzw: Sequence[float],
    target_pose_xyzw: Sequence[float],
    joint_limits: Sequence[Sequence[float]],
    fk_public_pose: Callable[[Sequence[float]], Sequence[float]],
    workspace_bounds_m: Mapping[str, Sequence[float]],
    profile: Mapping[str, Any],
    fk_link6_pose: Callable[[Sequence[float]], Sequence[float]] | None = None,
) -> ValidatedRoboTwinPlan:
    """Validate a CuRobo plan before the first drive-target write."""

    # The successor path normalizes backend output and delegates all generic
    # joint/FK/workspace admission to the shared motion transaction.  The
    # wrapper retains the frozen RoboTwin result type and artifact keys used by
    # EXP06A v5-v8 evidence.
    try:
        normalized = normalize_joint_plan_result(
            result,
            backend_id="robotwin",
            source_schema=PLAN_RESULT_SCHEMA,
        )
        admitted = validate_joint_plan(
            normalized,
            expected_joint_names=expected_joint_names,
            measured_start_joints=measured_start_joints,
            measured_start_pose_xyzw=measured_start_pose_xyzw,
            target_pose_xyzw=target_pose_xyzw,
            joint_limits=joint_limits,
            fk_public_pose=fk_public_pose,
            fk_control_pose=(fk_link6_pose or fk_public_pose),
            workspace_bounds_m=workspace_bounds_m,
            profile=profile,
        )
    except MotionPlanRejected as exc:
        raise RoboTwinPlanRejected(
            exc.code, str(exc), details=exc.details,
        ) from exc
    return ValidatedRoboTwinPlan(
        position=admitted.position,
        velocity=admitted.velocity,
        planned_link6_pose_xyzw=admitted.planned_control_pose_xyzw,
        planned_tcp_pose_xyzw=admitted.planned_public_tcp_pose_xyzw,
        summary=admitted.summary,
    )


def validate_inter_arm_clearance(
    plans: Mapping[str, ValidatedRoboTwinPlan],
    *,
    measured_joints: Mapping[str, Sequence[float]],
    fk_chain_points: Callable[[str, Sequence[float]], Sequence[Sequence[float]]],
    profile: Mapping[str, Any],
    link_names: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Reject a single- or dual-arm plan that intersects the other arm.

    Both arms are projected on one normalized timeline. An unplanned arm is
    evaluated at the immutable action-start joint snapshot.
    """

    if not plans or set(plans) - {"left", "right"}:
        _reject("collision_input", "collision validation requires left/right plans")
    lengths = {arm: int(plan.position.shape[0]) for arm, plan in plans.items()}
    stages = max(lengths.values())
    radius = float(profile["link_radius_m"])
    clearance = float(profile["minimum_surface_clearance_m"])
    try:
        initial_distance, _ = minimum_inter_arm_distance(
            fk_chain_points("left", measured_joints["left"]),
            fk_chain_points("right", measured_joints["right"]),
        )
        policy = InterArmClearancePolicy.from_profile(profile, initial_distance)
    except ValueError as exc:
        _reject("collision_fk", str(exc))
    minimum_distance = math.inf
    minimum_stage = 0
    minimum_pair = [0, 0]

    def joints_at(arm: str, stage: int) -> Sequence[float]:
        plan = plans.get(arm)
        if plan is None:
            return measured_joints[arm]
        count = lengths[arm]
        index = 0 if count <= 1 or stages <= 1 else min(
            count - 1, int(round(stage * (count - 1) / (stages - 1))),
        )
        return plan.position[index]

    for stage in range(stages):
        try:
            left_points = fk_chain_points("left", joints_at("left", stage))
            right_points = fk_chain_points("right", joints_at("right", stage))
            value, pair = minimum_inter_arm_distance(left_points, right_points)
        except ValueError as exc:
            _reject("collision_fk", str(exc))
        if value < minimum_distance:
            minimum_distance = value
            minimum_stage = stage
            minimum_pair = [int(pair[0]), int(pair[1])]
        decision = policy.assess(value, planning=True, endpoint=stage == stages - 1)
        if not decision["allowed"]:
            _reject(
                "inter_arm_collision",
                "planned arm link spheres collide with the other arm",
                stage_index=stage,
                link_pair=[int(pair[0]), int(pair[1])],
                link_identity={
                    "frame": "world", "geometry": "sphere_at_named_link_origin",
                    "left": {"index": int(pair[0]),
                             "name": link_names["left"][int(pair[0])] if link_names else None,
                             "center_m": [float(x) for x in left_points[int(pair[0])]],
                             "radius_m": radius},
                    "right": {"index": int(pair[1]),
                              "name": link_names["right"][int(pair[1])] if link_names else None,
                              "center_m": [float(x) for x in right_points[int(pair[1])]],
                              "radius_m": radius},
                },
                **{key: item for key, item in decision.items() if key != "allowed"},
            )
    return {
        "schema": "ora.robotwin-inter-arm-clearance.v1",
        "status": "accepted",
        "stages": stages,
        "minimum_center_distance_m": round(minimum_distance, 8),
        "minimum_stage_index": minimum_stage,
        "minimum_link_pair": minimum_pair,
        "link_radius_m": radius,
        "minimum_surface_clearance_m": clearance,
        **({key: item for key, item in policy.assess(value, planning=True, endpoint=True).items()
            if key not in {"allowed", "center_distance_m"}}
           if "escape_maximum_inward_deviation_m" in profile else {}),
    }


def write_plan_artifact(
    root: Path,
    *,
    action_sequence: int,
    arm: str,
    validated: ValidatedRoboTwinPlan,
    binding: Mapping[str, Any],
    execution_arrays: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist compact numeric plan evidence and a hash-bound manifest."""

    target = root / "plans" / f"action-{action_sequence:04d}-{arm}"
    target.mkdir(parents=True, exist_ok=False)
    arrays_path = target / "arrays.npz"
    temporary = target / "arrays.npz.tmp"
    arrays = dict(validated.arrays)
    for key, value in (execution_arrays or {}).items():
        array = np.asarray(value)
        if array.dtype.kind not in {"b", "i", "u", "f"} or not np.isfinite(array).all():
            raise ValueError(f"plan execution array {key} is not finite numeric data")
        arrays[str(key)] = array
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(arrays_path)
    arrays_sha256 = hashlib.sha256(arrays_path.read_bytes()).hexdigest()
    body = {
        "schema": PLAN_ARTIFACT_SCHEMA,
        "action_sequence": action_sequence,
        "arm": arm,
        "binding": json.loads(json.dumps(binding, sort_keys=True)),
        "validation": validated.summary,
        "arrays": {
            "path": "arrays.npz",
            "sha256": arrays_sha256,
            "keys": sorted(arrays),
        },
    }
    body_sha256 = hashlib.sha256(json.dumps(
        body, allow_nan=False, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    manifest = {**body, "artifact_sha256": body_sha256}
    manifest_path = target / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "schema": PLAN_ARTIFACT_SCHEMA,
        "path": manifest_path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "artifact_sha256": body_sha256,
    }


__all__ = [
    "PLAN_ARTIFACT_SCHEMA",
    "PLAN_RESULT_SCHEMA",
    "PLAN_REFERENCE_SCHEMA",
    "RoboTwinPlanRejected",
    "ValidatedRoboTwinPlan",
    "bounded_dls_joint_delta",
    "cartesian_pose_error_twist",
    "numerical_cartesian_jacobian",
    "public_tcp_point_jacobian",
    "planner_public_summary",
    "select_topk_validated_plan",
    "select_topk_validated_plan_set",
    "split_batch_plan_result",
    "TOPK_PLAN_RESULT_SCHEMA",
    "validate_single_arm_plan",
    "validate_inter_arm_clearance",
    "write_plan_artifact",
    "weighted_svd_dls_joint_delta",
]
