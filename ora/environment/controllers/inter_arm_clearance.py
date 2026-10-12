"""One immutable action-start clearance policy for planning and execution.

Distances refer only to source-bound robot link spheres. They do not describe
object contact or a scene collision world. Historical profiles omit the
execution/escape fields and retain their original single-threshold rule.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


def minimum_inter_arm_distance(
    left_points: Sequence[Sequence[float]],
    right_points: Sequence[Sequence[float]],
) -> tuple[float, list[int]]:
    points = [np.asarray(value, dtype=np.float64) for value in (left_points, right_points)]
    if any(value.ndim != 2 or value.shape[1:] != (3,) or not len(value)
           or not np.isfinite(value).all() for value in points):
        raise ValueError("inter-arm collision FK is invalid")
    distances = np.linalg.norm(points[0][:, None, :] - points[1][None, :, :], axis=2)
    pair = np.unravel_index(int(np.argmin(distances)), distances.shape)
    return float(distances[pair]), [int(pair[0]), int(pair[1])]


@dataclass(frozen=True)
class InterArmClearancePolicy:
    """Keep the escape floor bound to the original measurement, never a tick."""

    initial_center_distance_m: float
    planning_center_distance_m: float
    execution_center_distance_m: float
    escape_maximum_inward_deviation_m: float | None

    @classmethod
    def from_profile(cls, profile: Mapping[str, Any], initial: float) -> "InterArmClearancePolicy":
        radius = float(profile["link_radius_m"])
        planning = 2 * radius + float(profile["minimum_surface_clearance_m"])
        execution = 2 * radius + float(profile.get(
            "execution_minimum_surface_clearance_m", profile["minimum_surface_clearance_m"],
        ))
        escape = profile.get("escape_maximum_inward_deviation_m")
        if (not all(math.isfinite(value) and value >= 0 for value in (initial, planning, execution))
                or execution > planning
                or escape is not None and (not math.isfinite(float(escape)) or float(escape) < 0)):
            raise ValueError("inter-arm clearance policy is invalid")
        return cls(float(initial), planning, execution, None if escape is None else float(escape))

    @property
    def escape_allowed(self) -> bool:
        return (self.escape_maximum_inward_deviation_m is not None
                and self.initial_center_distance_m < self.execution_center_distance_m)

    def assess(self, distance: float, *, planning: bool = False,
               endpoint: bool = False) -> dict[str, Any]:
        threshold = self.planning_center_distance_m if planning else self.execution_center_distance_m
        if self.escape_allowed:
            threshold = self.initial_center_distance_m - float(self.escape_maximum_inward_deviation_m)
        allowed = math.isfinite(distance) and distance >= threshold - 1e-12
        if self.escape_allowed and endpoint:
            allowed = allowed and distance > self.initial_center_distance_m + 1e-12
        return {
            "allowed": bool(allowed), "escape_allowed": self.escape_allowed,
            "current_center_distance_m": round(self.initial_center_distance_m, 8),
            "center_distance_m": round(float(distance), 8),
            "required_center_distance_m": round(threshold, 8),
            "planning_center_distance_m": round(self.planning_center_distance_m, 8),
            "execution_center_distance_m": round(self.execution_center_distance_m, 8),
            "escape_endpoint_must_increase": bool(self.escape_allowed),
        }
