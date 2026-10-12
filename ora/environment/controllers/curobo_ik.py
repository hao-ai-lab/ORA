"""Host-safe wire contract of the isolated CuRobo worker's read-only ``ik`` command.

The LIBERO planner worker (``backends/libero_pro/curobo_worker.py``) answers a
batch of exact base-frame ``panda_hand`` poses with one IK solve each, through
the IK solver of the same MotionGen instance the global ``move_pose`` segment
plans with. No trajectory is planned and nothing is executed. A query may name
an earlier query (``from_index``) whose solution seeds it; a query chained onto
one that did not solve is ``skipped``. The worker reports raw solver output;
deciding reachability at the controlled point belongs to the backend session.
"""
from __future__ import annotations

import json
import math
from typing import Any, Mapping

from ora.contracts.reach_check import MAXIMUM_POSES

from .curobo_motion import _PANDA_JOINT_NAMES, CuroboPlannerError

CUROBO_IK_RESPONSE_SCHEMA = "ora.curobo-ik-response.v1"
_REQUEST_FIELDS = frozenset({"joint_names", "joint_positions", "queries"})
_QUERY_FIELDS = frozenset({"target_pose_base_xyzw", "from_index"})
_RESULT_FIELDS = frozenset({
    "solved", "skipped", "joint_positions", "ee_pose_base_xyzw", "position_error_m", "rotation_error",
})
_SOLVED_FIELDS = frozenset({
    "schema", "status", "world_model", "solver", "joint_names", "joint_limits",
    "start_ee_pose_base_xyzw", "results", "solve_time_s",
})
_ERROR_FIELDS = frozenset({"schema", "status", "world_model", "solve_time_s", "error"})
_SOLVER_FIELDS = frozenset({"num_seeds", "position_threshold_m", "rotation_threshold", "attempts_per_query"})


def _finite_vector(value: Any, length: int) -> bool:
    return (isinstance(value, list) and len(value) == length
            and all(not isinstance(item, bool) and isinstance(item, (int, float))
                    and math.isfinite(float(item)) for item in value))


def _unit_pose(value: Any) -> bool:
    return _finite_vector(value, 7) and math.isclose(
        sum(float(item) ** 2 for item in value[3:]), 1.0, abs_tol=1e-5)


def _non_negative(value: Any) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(float(value)) and float(value) >= 0)


def validate_curobo_ik_request(value: Any) -> dict[str, Any]:
    """The closed IK request: measured Panda joints and 1..MAXIMUM_POSES chained poses."""
    if not isinstance(value, Mapping) or set(value) != _REQUEST_FIELDS:
        raise CuroboPlannerError("CuRobo IK request fields drifted")
    if value["joint_names"] != _PANDA_JOINT_NAMES or not _finite_vector(value["joint_positions"], 7):
        raise CuroboPlannerError("CuRobo IK request has invalid current joints")
    queries = value["queries"]
    if not isinstance(queries, list) or not 1 <= len(queries) <= MAXIMUM_POSES:
        raise CuroboPlannerError(f"CuRobo IK request must hold 1 to {MAXIMUM_POSES} queries")
    for index, query in enumerate(queries):
        if not isinstance(query, Mapping) or set(query) != _QUERY_FIELDS:
            raise CuroboPlannerError("CuRobo IK query fields drifted")
        if not _unit_pose(query["target_pose_base_xyzw"]):
            raise CuroboPlannerError("CuRobo IK target must be a finite pose with a unit quaternion")
        start = query["from_index"]
        if start is not None and (isinstance(start, bool) or not isinstance(start, int)
                                  or not 0 <= start < index):
            raise CuroboPlannerError("CuRobo IK from_index must name an earlier query")
    return json.loads(json.dumps(value))


def _validate_result(result: Any, *, seed_available: bool) -> None:
    if not isinstance(result, Mapping) or set(result) != _RESULT_FIELDS:
        raise CuroboPlannerError("CuRobo IK result fields drifted")
    solved, skipped = result["solved"], result["skipped"]
    if type(solved) is not bool or type(skipped) is not bool or (solved and skipped):
        raise CuroboPlannerError("CuRobo IK result flags are invalid")
    if skipped is seed_available:
        raise CuroboPlannerError("CuRobo IK skipped a query whose seed exists, or solved an orphan")
    if solved:
        if not _finite_vector(result["joint_positions"], 7) or not _unit_pose(result["ee_pose_base_xyzw"]):
            raise CuroboPlannerError("CuRobo IK solution is malformed")
    elif result["joint_positions"] is not None or result["ee_pose_base_xyzw"] is not None:
        raise CuroboPlannerError("an unsolved CuRobo IK query carries a solution")
    for key in ("position_error_m", "rotation_error"):
        if result[key] is not None and not _non_negative(result[key]):
            raise CuroboPlannerError(f"CuRobo IK {key} is malformed")
        if skipped and result[key] is not None:
            raise CuroboPlannerError("a skipped CuRobo IK query carries solver errors")


def validate_curobo_ik_response(value: Any, request: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate the worker's answer against the request it answers.

    ``request`` is ``None`` only for a ``planner_error`` raised before the
    request validated; a solved answer is always checked query by query.
    """
    if not isinstance(value, Mapping) or value.get("schema") != CUROBO_IK_RESPONSE_SCHEMA:
        raise CuroboPlannerError("CuRobo IK response schema drift")
    if value.get("world_model") != "none":
        raise CuroboPlannerError("CuRobo IK response must declare world_model=none")
    if not _non_negative(value.get("solve_time_s")):
        raise CuroboPlannerError("CuRobo IK solve time is malformed")
    if value.get("status") == "planner_error":
        error = value.get("error")
        if (set(value) != _ERROR_FIELDS or not isinstance(error, Mapping) or set(error) != {"type", "message"}
                or any(not isinstance(error[key], str) or not error[key] for key in error)):
            raise CuroboPlannerError("CuRobo IK planner_error omitted its error identity")
        return json.loads(json.dumps(value))
    if value.get("status") != "solved" or set(value) != _SOLVED_FIELDS or request is None:
        raise CuroboPlannerError("CuRobo IK response fields drifted")
    solver = value["solver"]
    if (not isinstance(solver, Mapping) or set(solver) != _SOLVER_FIELDS
            or any(type(solver[key]) is not int or solver[key] < 1 for key in ("num_seeds", "attempts_per_query"))
            or not all(_non_negative(solver[key]) and solver[key] > 0
                       for key in ("position_threshold_m", "rotation_threshold"))):
        raise CuroboPlannerError("CuRobo IK solver identity is malformed")
    limits = value["joint_limits"]
    if (value["joint_names"] != _PANDA_JOINT_NAMES or not isinstance(limits, list) or len(limits) != 2
            or not all(_finite_vector(row, 7) for row in limits)
            or any(low >= high for low, high in zip(limits[0], limits[1]))):
        raise CuroboPlannerError("CuRobo IK joint limits are malformed")
    if not _unit_pose(value["start_ee_pose_base_xyzw"]):
        raise CuroboPlannerError("CuRobo IK start forward kinematics is malformed")
    results, queries = value["results"], request["queries"]
    if not isinstance(results, list) or len(results) != len(queries):
        raise CuroboPlannerError("CuRobo IK answered a different number of queries")
    for query, result in zip(queries, results):
        start = query["from_index"]
        _validate_result(result, seed_available=start is None or results[start]["solved"] is True)
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise CuroboPlannerError("CuRobo IK response contains invalid JSON evidence") from exc


__all__ = ["CUROBO_IK_RESPONSE_SCHEMA", "validate_curobo_ik_request", "validate_curobo_ik_response"]
