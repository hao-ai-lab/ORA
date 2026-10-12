"""Sensor-frame force projection and bounded force-stop policy.

No simulator imports or contact truth are used here. A wrist reaction includes
gravity and inertia; reversing it estimates environment-on-EEF loading rather
than identifying a collision or a grasped object.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


FORCE_DIRECTION_CONVENTION = "environment_on_eef_reaction_estimate"


def sensor_reaction_feedback(
    force_sensor_n: Sequence[float] | None,
    world_from_sensor_rotation: Sequence[Sequence[float]] | None,
) -> dict[str, Any]:
    """Rotate and reverse MuJoCo's child-to-parent site-frame reaction.

The rotation must be the force sensor's actual site rotation at the same
measurement. A missing or malformed calibration never produces a direction.
"""
    result: dict[str, Any] = {
        "external_force_world_n": None,
        "external_force_direction_world": None,
        "force_frame": "world",
        "force_direction_convention": FORCE_DIRECTION_CONVENTION,
        "force_direction_status": "unavailable",
    }
    try:
        force = [float(value) for value in force_sensor_n]  # type: ignore[union-attr]
        rotation = [[float(value) for value in row]
                    for row in world_from_sensor_rotation]  # type: ignore[union-attr]
        if len(force) != 3 or len(rotation) != 3 or any(len(row) != 3 for row in rotation):
            return result
        if not all(math.isfinite(value) for value in force + sum(rotation, [])):
            return result
        for i in range(3):
            for j in range(3):
                if abs(sum(rotation[k][i] * rotation[k][j] for k in range(3))
                       - float(i == j)) > 1e-6:
                    return result
        determinant = sum(rotation[0][i] * (
            rotation[1][(i + 1) % 3] * rotation[2][(i + 2) % 3]
            - rotation[1][(i + 2) % 3] * rotation[2][(i + 1) % 3]
        ) for i in range(3))
        if abs(determinant - 1.0) > 1e-6:
            return result
    except (TypeError, ValueError, OverflowError):
        return result
    world = [-sum(row[i] * force[i] for i in range(3)) for row in rotation]
    magnitude = math.sqrt(sum(value * value for value in world))
    result.update(
        external_force_world_n=world,
        external_force_direction_world=(
            [value / magnitude for value in world] if magnitude > 1e-12 else None
        ),
        force_direction_status="available" if magnitude > 1e-12 else "zero_force",
    )
    return result


def public_force_direction(state: Mapping[str, Any]) -> dict[str, Any]:
    """Select calibrated scalar/vector fields without leaking simulator names."""
    return {key: state.get(key) for key in (
        "external_force_world_n", "external_force_direction_world", "force_frame",
        "force_direction_convention", "force_direction_status",
    )}


def precision_force_stop_reason(
    *, interaction: str | None, magnitude_n: float | None,
    threshold_n: float, enabled: bool,
) -> str | None:
    """One stop decision for the legacy precision adapter and bounded Servo."""
    if not enabled or interaction != "precision" or magnitude_n is None:
        return None
    if (math.isfinite(magnitude_n) and math.isfinite(threshold_n)
            and magnitude_n > threshold_n > 0.0):
        return "contact_stop"
    return None
