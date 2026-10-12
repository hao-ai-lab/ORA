"""Shared action-workspace validation and recovery semantics.

An action workspace is a hard command envelope, not an inverse-kinematics or
collision-free reachability claim.  Agent tools and environment backends use
these helpers so a target cannot be accepted on one side and rejected on the
other.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


AXES = ("x", "y", "z")


class WorkspaceError(ValueError):
    """A workspace envelope or Cartesian point is malformed."""


def normalize_workspace_bounds(value: Mapping[str, Any]) -> dict[str, list[float]]:
    if set(value) != set(AXES):
        raise WorkspaceError("workspace bounds must define x/y/z")
    result: dict[str, list[float]] = {}
    for axis in AXES:
        interval = value[axis]
        if (
            not isinstance(interval, (list, tuple))
            or len(interval) != 2
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                for item in interval
            )
        ):
            raise WorkspaceError(f"invalid workspace bound: {axis}")
        low, high = (float(interval[0]), float(interval[1]))
        if low >= high:
            raise WorkspaceError(f"workspace bound must be increasing: {axis}")
        result[axis] = [low, high]
    return result


def workspace_violations(
    position_m: Sequence[float], bounds_m: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    if (
        not isinstance(position_m, (list, tuple))
        or len(position_m) != 3
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in position_m
        )
    ):
        raise WorkspaceError("Cartesian position must contain three finite numbers")
    bounds = normalize_workspace_bounds(bounds_m)
    result: dict[str, dict[str, Any]] = {}
    for axis, coordinate in zip(AXES, position_m):
        received = float(coordinate)
        low, high = bounds[axis]
        distance = low - received if received < low else received - high if received > high else 0.0
        if distance > 0:
            result[axis] = {
                "received": received,
                "allowed": [low, high],
                "distance_m": distance,
            }
    return result


def workspace_state(
    position_m: Sequence[float], bounds_m: Mapping[str, Any],
) -> dict[str, Any]:
    violations = workspace_violations(position_m, bounds_m)
    return {
        "inside": not violations,
        "violations": violations,
        "max_violation_m": max(
            (float(item["distance_m"]) for item in violations.values()),
            default=0.0,
        ),
        "total_violation_m": sum(
            float(item["distance_m"]) for item in violations.values()
        ),
    }


def is_monotonic_workspace_recovery(
    current_position_m: Sequence[float],
    target_position_m: Sequence[float],
    bounds_m: Mapping[str, Any],
    *,
    tolerance_m: float = 1e-9,
) -> bool:
    """Allow an out-of-envelope state to move strictly back toward safety.

    The target may remain outside for a multi-step recovery, but it cannot add a
    violated axis or make any existing violation worse.
    """

    current = workspace_state(current_position_m, bounds_m)
    target = workspace_state(target_position_m, bounds_m)
    if current["inside"] or target["inside"]:
        return bool(not current["inside"] and target["inside"])
    current_violations = current["violations"]
    target_violations = target["violations"]
    if set(target_violations) - set(current_violations):
        return False
    if any(
        float(target_violations.get(axis, {}).get("distance_m", 0.0))
        > float(details["distance_m"]) + tolerance_m
        for axis, details in current_violations.items()
    ):
        return False
    return (
        float(target["total_violation_m"])
        < float(current["total_violation_m"]) - tolerance_m
    )


__all__ = [
    "AXES",
    "WorkspaceError",
    "is_monotonic_workspace_recovery",
    "normalize_workspace_bounds",
    "workspace_state",
    "workspace_violations",
]
