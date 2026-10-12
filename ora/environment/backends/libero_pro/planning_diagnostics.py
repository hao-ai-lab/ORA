"""Observe existing CuRobo IK metrics without rerunning or changing a solve.

The pinned MotionGen keeps only its top trajectory seeds. This observer copies
the already evaluated optimizer population before that evidence disappears.
Constraint feasibility is aggregate: no per-term collision cause is invented.
"""
from __future__ import annotations

import math
from types import MethodType
from typing import Any, Mapping

MAX_ATTEMPTS = 8
MAX_CANDIDATES = 512


def _array(value: Any) -> Any:
    import numpy as np

    result = np.asarray(value.detach().cpu().tolist())
    if not result.size or not np.isfinite(result).all():
        raise ValueError("missing_or_nonfinite_ik_metrics")
    return result


def capture_attempt(solver: Any, num_seeds: int, raw: Any, returned: Any) -> dict[str, Any]:
    """Copy one original optimizer result; never call FK, rollout or solve."""
    import numpy as np

    q = _array(raw.action.position)
    # POSITION control returns [seed, dof]; full action buffers carry the
    # singleton IK horizon explicitly as [seed, 1, dof]. Neither is a batch of
    # trajectories, and a larger horizon is not silently flattened.
    if q.ndim == 3 and q.shape[1:] == (1, 7):
        q = q[:, 0, :]
    if q.ndim != 2 or q.shape[1:] != (7,) or not 1 <= q.shape[0] <= MAX_CANDIDATES:
        raise ValueError("unexpected_ik_population_shape")
    count = len(q)
    if count != num_seeds:
        raise ValueError("single_query_ik_population_mismatch")
    names = list(solver.rollout_fn.kinematics.joint_names)
    if names != [f"panda_joint{i}" for i in range(1, 8)]:
        raise ValueError("unexpected_ik_joint_order")
    metrics = raw.metrics
    position = _array(metrics.position_error).reshape(-1)
    rotation = _array(metrics.rotation_error).reshape(-1)
    feasible = _array(metrics.feasible).reshape(-1)
    if any(len(values) != count for values in (position, rotation, feasible)):
        raise ValueError("ik_metric_population_mismatch")
    if not np.isin(feasible, [False, True]).all() or (position < 0).any() or (rotation < 0).any():
        raise ValueError("invalid_ik_metric_values")
    thresholds = [float(solver.position_threshold), float(solver.rotation_threshold)]
    if not all(math.isfinite(value) and value > 0 for value in thresholds):
        raise ValueError("invalid_ik_convergence_thresholds")
    converged = (position <= thresholds[0]) & (rotation <= thresholds[1])
    feasible = feasible.astype(bool)
    accepted = feasible & converged
    returned_success = _array(returned.success).reshape(-1)
    if len(returned_success) > count or not np.isin(returned_success, [False, True]).all():
        raise ValueError("invalid_returned_ik_success_mask")
    # This names a directly observed position-limit violation. It does not
    # equate a raw bound comparison with all bound-constraint terms/activation.
    outside = None
    limits = None
    try:
        limits = _array(solver.solver.safety_rollout.bound_constraint.joint_limits.position)
        if limits.shape != (2, 7) or not (limits[0] < limits[1]).all():
            raise ValueError("invalid_position_limits")
        outside = ((q < limits[0]) | (q > limits[1])).any(axis=1)
    except (AttributeError, TypeError, ValueError):
        limits = None
    return {
        "status": "captured", "scope": "all_optimized_seeds_for_one_original_ik_attempt",
        "joint_names": names, "joint_positions": q.tolist(),
        "position_errors_m": position.tolist(),
        "rotation_errors": rotation.tolist(),
        "rotation_error_units": "curobo_v0.7.8_rotation_metric_not_radians",
        "position_threshold_m": thresholds[0], "rotation_threshold": thresholds[1],
        "feasible": feasible.tolist(), "converged": converged.tolist(),
        "accepted": accepted.tolist(), "returned_success": returned_success.astype(bool).tolist(),
        "position_limits": None if limits is None else limits.tolist(),
        "outside_position_limits": None if outside is None else outside.tolist(),
        "candidate_counts": {
            "optimized": count, "returned": len(returned_success),
            "returned_successful": int(returned_success.sum()),
            "converged": int(converged.sum()), "feasible": int(feasible.sum()),
            "accepted": int(accepted.sum()),
        },
        "rejection_counts": {
            "position_tolerance": int((position > thresholds[0]).sum()),
            "rotation_tolerance": int((rotation > thresholds[1]).sum()),
            "solver_infeasible": int((~feasible).sum()),
            "outside_position_limits": None if outside is None else int(outside.sum()),
            "self_collision": None,
        },
        "unknown_reasons": ["per_term_constraint_costs_not_retained_by_sdk"],
        "claim_boundary": "finite_optimizer_samples;counts_overlap;not_global_unreachability_or_scene_clearance",
    }


def install_ik_observer(motion_gen: Any) -> None:
    """Wrap only this worker's result extraction, leaving SDK code untouched."""
    solver = motion_gen.ik_solver
    original = solver._get_result

    def observed(_solver: Any, num_seeds: int, result: Any, goal_pose: Any, return_seeds: int) -> Any:
        # Preserve the exact call, returned object and exception behavior.
        answer = original(num_seeds, result, goal_pose, return_seeds)
        records = motion_gen.ik_diagnostic_attempts
        if len(records) >= MAX_ATTEMPTS:
            motion_gen.ik_diagnostic_attempts_dropped += 1
            return answer
        try:
            record = capture_attempt(solver, num_seeds, result, answer)
        except Exception as exc:
            # Diagnosis cannot turn an original result into a planner fault.
            record = {"status": "unavailable", "reason": "ik_metric_capture_failed",
                      "error_type": type(exc).__name__, "error_message": str(exc)[:500]}
        records.append(record)
        return answer

    reset_ik_observer(motion_gen)
    solver._get_result = MethodType(observed, solver)


def reset_ik_observer(motion_gen: Any) -> None:
    motion_gen.ik_diagnostic_attempts = []
    motion_gen.ik_diagnostic_attempts_dropped = 0


def summarize(motion_gen: Any, response: Mapping[str, Any], request: Mapping[str, Any] | None) -> tuple[dict, dict]:
    attempts = list(getattr(motion_gen, "ik_diagnostic_attempts", []))
    dropped = int(getattr(motion_gen, "ik_diagnostic_attempts_dropped", 0))
    status = str(response.get("planner_status") or response.get("status") or "unknown")
    last = attempts[-1] if attempts else {}
    captured = last.get("status") == "captured"
    if "INVALID_START_STATE" in status:
        phase = "start_state"
    elif request is not None and "goal_joint_positions" in request:
        phase = "joint_space"
    elif "IK_FAIL" in status:
        phase = "ik"
    elif response.get("status") in {"planned", "not_reachable"}:
        phase = "trajectory_planning"
    else:
        phase = "unknown"
    unknown = list(last.get("unknown_reasons", [])) if captured else [
        last.get("reason", "no_target_ik_metrics_retained_for_this_request")]
    if dropped:
        unknown.append("attempt_capture_limit_exceeded")
    recovery = []
    if phase == "ik" and response.get("reachable") is False:
        recovery = [
            "Try a different target position or orientation, or an intermediate pose.",
            "This finite IK search does not prove that no valid solution exists.",
        ]
    summary = {
        "phase": phase, "outcome": status,
        "candidate_counts": dict(last.get("candidate_counts", {})),
        "rejection_counts": dict(last.get("rejection_counts", {})),
        "details_available": captured, "recovery": recovery,
        "unknown_reasons": unknown,
        "evidence_scope": "last_recorded_original_ik_attempt" if captured else "no_target_ik_details",
        "note": "Counts overlap; raw joint bounds do not isolate self-collision or other constraints. World collision model is absent.",
    }
    evidence = {"schema": "ora.libero-ik-diagnostics.v1",
                "attempts": attempts, "attempts_dropped": dropped,
                "world_model": "none", "extra_solver_calls": 0}
    return summary, evidence


#: The order an unaccepted CuRobo IK seed is attributed to one filter, in the
#: shared probe vocabulary (``contracts.planning_diagnostics.IK_CONSTRAINT_KEYS``):
#: a seed outside the raw joint position limits; else one the solver's own
#: constraint check marks infeasible (self-collision or the bound activation band,
#: which the SDK does not separate); else one that did not converge to the
#: pose within the solver's thresholds. ``joint_jump`` and ``pitch_joint_wrap``
#: are not LIBERO filters and stay zero.
TALLY_ORDER = ("joint_limit", "solver_unsuccessful", "fk_residual_exceeded")


def ik_candidate_tally(attempt: Any) -> dict[str, Any] | None:
    """Exclusive counts of one captured IK seed population, or None if none was captured.

    ``returned`` is every seed the solver optimised and ``valid_endpoint`` those
    both feasible and converged (what the solver accepts); every other seed is
    counted once, under the first ``TALLY_ORDER`` filter it fails, so the
    rejections sum to ``returned - valid_endpoint``. Seed arrays are captured
    finite (``capture_attempt``), so ``non_finite_solution`` is zero.
    """
    from ora.contracts.planning_diagnostics import IK_CONSTRAINT_KEYS

    if not isinstance(attempt, Mapping) or attempt.get("status") != "captured":
        return None
    accepted, feasible, converged = (list(attempt[key]) for key in ("accepted", "feasible", "converged"))
    outside = attempt.get("outside_position_limits")
    outside = list(outside) if isinstance(outside, list) else [False] * len(accepted)
    if not len(accepted) == len(feasible) == len(converged) == len(outside):
        raise ValueError("captured IK seed arrays differ in length")
    rejections = {key: 0 for key in IK_CONSTRAINT_KEYS}
    for ok, fits, close, beyond in zip(accepted, feasible, converged, outside):
        if ok:
            continue
        rejections["joint_limit" if beyond else "solver_unsuccessful" if not fits
                   else "fk_residual_exceeded"] += 1
    return {"returned": len(accepted), "valid_endpoint": sum(bool(item) for item in accepted),
            "rejections": rejections, "limits_known": isinstance(attempt.get("outside_position_limits"), list)}


def public_summary(planned: Mapping[str, Any]) -> dict[str, Any] | None:
    from ora.contracts.planning_diagnostics import public_planning_diagnostics

    value = planned.get("planning_diagnostics")
    if not isinstance(value, Mapping):
        return None
    # The worker/client contract validates this object. Never project the raw
    # per-seed positions or the canonical artifact's host path to the policy.
    return public_planning_diagnostics(value)
