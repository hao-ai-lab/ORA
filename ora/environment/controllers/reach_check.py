"""Answer exact-pose reach checks with the backend's read-only inverse kinematics.

Backend-neutral like ``alignment.py`` beside it: this module takes an adapter
exposing ``reach_check_batch`` and a control profile, and has no import path to
a planner. Each backend's adapter answers with its own IK -- RoboTwin the
damped-least-squares solve ``align_plan`` already uses, RoboDojo the SDK IK
selection on the move_pose route the query's interaction selects, LIBERO one
CuRobo IK solve in its isolated planner worker -- names its model in
``ik_model`` and says in ``matches_move_pose_admission`` whether that is the
admission move_pose itself runs.

The RoboTwin and RoboDojo ``align_plan`` session methods call
``alignment.plan_alignment``, which routes a request carrying the reach-check
schema here; LIBERO's ``align_plan`` (``backends/libero_pro/reach_check.py``)
serves only that schema and calls ``check_reach`` directly. The read-only IK
bridge therefore carries the reach-check schema while the worker, RPC and
coordinator stay unchanged.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from ora.contracts.reach_check import (
    BACKEND_REASONS, BATCH_FIELDS, INTERACTIONS, MAXIMUM_POSES, QUERY_FIELDS,
    REQUEST_FIELDS, REQUEST_SCHEMA, RESULT_SCHEMA, ROBODOJO_ROUTES, ROW_FIELDS,
    ReachCheckRequestDrift,
)
from ora.contracts.refusal import EnvironmentRefusal

_NUMBER_FIELDS = (
    "position_error_m", "orientation_error_rad", "joint_limit_margin_rad",
    "joint_delta_rad",
)


def _queries(value: Any) -> list[dict[str, Any]]:
    """Closed query list: pose, modelled interaction, optional earlier start."""

    if not isinstance(value, list) or not 1 <= len(value) <= MAXIMUM_POSES:
        raise EnvironmentRefusal(
            f"queries must hold one to {MAXIMUM_POSES} poses", field="queries",
            received=len(value) if isinstance(value, list) else value,
            allowed=[f"1..{MAXIMUM_POSES} queries"],
        )
    queries = []
    for index, query in enumerate(value):
        if not isinstance(query, Mapping) or set(query) != QUERY_FIELDS:
            raise ReachCheckRequestDrift("reach-check query fields drifted")
        pose = query["pose_xyzw"]
        if (not isinstance(pose, list) or len(pose) != 7
                or any(isinstance(item, bool) or not isinstance(item, (int, float))
                       or not math.isfinite(item) for item in pose)):
            raise EnvironmentRefusal(
                "a query pose must be seven finite numbers", field=f"queries[{index}].pose_xyzw",
                received=pose, allowed=["[x, y, z, qx, qy, qz, qw]"],
            )
        if query["interaction"] not in INTERACTIONS:
            raise EnvironmentRefusal(
                "interaction is not a move_pose interaction", field=f"queries[{index}].interaction",
                received=query["interaction"], allowed=list(INTERACTIONS),
            )
        start = query["from_index"]
        if start is not None and (isinstance(start, bool) or not isinstance(start, int)
                                  or not 0 <= start < index):
            raise EnvironmentRefusal(
                "from_index must name an earlier query", field=f"queries[{index}].from_index",
                received=start, allowed=[f"null or 0..{index - 1}"] if index else ["null"],
            )
        queries.append({"pose_xyzw": [float(item) for item in pose],
                        "interaction": query["interaction"], "from_index": start})
    return queries


def _row(value: Any) -> dict[str, Any]:
    """One backend answer, checked against the closed shared row contract."""

    if not isinstance(value, Mapping) or set(value) != ROW_FIELDS:
        raise ReachCheckRequestDrift("reach-check backend row fields drifted")
    reachable, reason = value["reachable"], value["reason"]
    if type(reachable) is not bool or (reason is None) != reachable:
        raise ReachCheckRequestDrift("reach-check verdict and reason disagree")
    if reason is not None and reason not in BACKEND_REASONS:
        raise ReachCheckRequestDrift(f"reach-check reason is undeclared: {reason!r}")
    if value["route"] is not None and value["route"] not in ROBODOJO_ROUTES:
        raise ReachCheckRequestDrift(f"reach-check route is undeclared: {value['route']!r}")
    row: dict[str, Any] = {"reachable": reachable, "reason": reason, "route": value["route"]}
    for key in _NUMBER_FIELDS:
        number = value[key]
        if number is not None and (
            isinstance(number, bool) or not isinstance(number, (int, float))
            or not math.isfinite(number) or number < 0
        ):
            raise ReachCheckRequestDrift(f"reach-check {key} is not a finite non-negative number")
        row[key] = None if number is None else float(number)
    return row


def check_reach(
    request: Mapping[str, Any], *, adapter: Any,
    control_profile: Mapping[str, Any], available_arms: Sequence[str],
) -> dict[str, Any]:
    """Answer one arm's batch of exact-pose queries, one row per query, in order."""

    if set(request) != REQUEST_FIELDS or request.get("schema") != REQUEST_SCHEMA:
        raise ReachCheckRequestDrift("reach-check request fields drifted")
    arm = request.get("arm")
    if arm not in available_arms:
        raise EnvironmentRefusal(
            "arm is not an arm of this embodiment", field="arm", received=arm,
            allowed=list(available_arms),
        )
    queries = _queries(request.get("queries"))
    tolerances = (control_profile.get("profile") or {}).get("arrival_check")
    if not isinstance(tolerances, Mapping):
        raise ReachCheckRequestDrift(
            "control profile declares no arrival_check tolerances for reachability"
        )
    batch = adapter.reach_check_batch(
        str(arm), queries,
        position_tolerance_m=float(tolerances["position_tolerance_m"]),
        orientation_tolerance_rad=float(tolerances["orientation_tolerance_rad"]),
    )
    if (not isinstance(batch, Mapping) or set(batch) != BATCH_FIELDS
            or not isinstance(batch["ik_model"], str) or not batch["ik_model"]
            or type(batch["matches_move_pose_admission"]) is not bool
            or not isinstance(batch["claim_boundary"], str) or not batch["claim_boundary"]
            or not isinstance(batch["results"], list) or len(batch["results"]) != len(queries)):
        raise ReachCheckRequestDrift("reach-check backend batch drifted")
    return {
        "schema": RESULT_SCHEMA,
        "arm": str(arm),
        "ik_model": batch["ik_model"],
        "matches_move_pose_admission": batch["matches_move_pose_admission"],
        "motion_planner_called": False,
        "claim_boundary": batch["claim_boundary"],
        "results": [_row(item) for item in batch["results"]],
    }


def joint_limit_margin_rad(
    joints: Sequence[float] | None, limits: Sequence[Sequence[float]],
) -> float | None:
    """Smallest distance of any joint to its nearer limit, in radians."""

    if joints is None:
        return None
    margins = [
        min(float(value) - float(low), float(high) - float(value))
        for value, (low, high) in zip(joints, limits)
    ]
    if len(margins) != len(limits) or not margins:
        raise ReachCheckRequestDrift("joint vector and limit table differ in length")
    return max(0.0, min(margins))


def joint_delta_rad(
    joints: Sequence[float] | None, reference: Sequence[float],
) -> float | None:
    """Largest single-joint move from the measured configuration, in radians."""

    if joints is None:
        return None
    if len(joints) != len(reference):
        raise ReachCheckRequestDrift("joint vector and reference differ in length")
    return max(abs(float(a) - float(b)) for a, b in zip(joints, reference))
