"""Runtime half of ``pose_orientation_from_world_axes_v1`` (see ``composition/world_axes_schema.py``).

The ``world_axes`` orientation mode of ``move_pose`` names the target
orientation by two world directions: ``approach_world`` for the controlled
frame's declared approach axis and ``jaw_line_world`` for its declared closing
(jaw-line) axis -- link6 local +X and +Y on the ARX X5. The jaw line keeps only
its part perpendicular to the approach, and the pair compiles deterministically
into the existing absolute quaternion before ``compile_enhanced_move_pose`` and
the action envelope see it, so the backend and every kernel file receive the
released ``absolute_quaternion`` request.

Every compiled target of ``move_pose`` and the primitives that compile through
it (``move_to``, ``rotate_wrist``, ``rotate_pitch``) reports
``target_axes_world``: the world directions of the approach axis and jaw line
of the quaternion actually submitted. An absolute quaternion is never rewritten
or refused for where it points; an upward approach is legal.

In a recorded RoboDojo batch three cells meant "pointing down" and hand-computed
``[0, -0.7071, 0, 0.7071]``, whose approach axis points world +Z; the planner
refused and the policy attributed the refusal to reach or self-collision.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from ora.agent.composition.world_axes_schema import CAPABILITY, MODE
from ora.contracts.pose import matrix_quaternion, quaternion_matrix

from .robot import ToolError

#: A direction shorter than this has no direction.
MIN_DIRECTION_LENGTH = 1e-6
#: The jaw line must be at least this far from parallel to the approach (either
#: sign): closer, its perpendicular part -- the roll it fixes -- is a small
#: remainder of the input.
MIN_SEPARATION_DEG = 5.0
_DIGITS = 6


def declared(surface: Any) -> bool:
    return surface.supports(CAPABILITY)


def _local_axes(surface: Any) -> dict[str, list[float]]:
    from .alignment import frame_local_axes

    return frame_local_axes(str((surface.view_config or {}).get("controlled_frame_convention")))


def _reject(message: str, field: str, **constraints: Any) -> ToolError:
    return ToolError(message, code="invalid_arguments", retryable=True, field=field, constraints=constraints)


def _direction(value: Any, field: str) -> list[float]:
    if (not isinstance(value, (list, tuple)) or len(value) != 3
            or not all(type(item) in (int, float) and math.isfinite(item) for item in value)):
        raise _reject(f"{field.rsplit('.', 1)[-1]} must be three finite numbers [x,y,z]", field, length=3)
    length = math.sqrt(sum(float(item) ** 2 for item in value))
    if length < MIN_DIRECTION_LENGTH:
        raise _reject(f"{field.rsplit('.', 1)[-1]} has length {length:.3g}, below {MIN_DIRECTION_LENGTH}; "
                      "it names no direction", field, min_length=MIN_DIRECTION_LENGTH, received_length=length)
    return [float(item) / length for item in value]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _cross(left: Sequence[float], right: Sequence[float]) -> list[float]:
    return [left[1] * right[2] - left[2] * right[1], left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0]]


def _perpendicular_jaw(approach: list[float], jaw: list[float], field: str) -> list[float]:
    angle = math.degrees(math.acos(min(1.0, abs(_dot(approach, jaw)))))
    if angle < MIN_SEPARATION_DEG:
        raise _reject(f"jaw_line_world is {angle:.2f} degrees from parallel to approach_world; it must be at "
                      f"least {MIN_SEPARATION_DEG} degrees from either direction of the approach", field,
                      min_angle_from_approach_deg=MIN_SEPARATION_DEG, received_angle_deg=round(angle, 4))
    along = _dot(approach, jaw)
    perpendicular = [j - along * a for j, a in zip(jaw, approach)]
    norm = math.sqrt(_dot(perpendicular, perpendicular))
    return [item / norm for item in perpendicular]


def quaternion_from_world_axes(
    approach_world: Any, jaw_line_world: Any, *, local_axes: Mapping[str, Sequence[float]], arm: str,
) -> list[float]:
    """The xyzw quaternion that turns the declared local axes onto the two world directions."""
    prefix = f"targets.{arm}"
    approach = _direction(approach_world, f"{prefix}.approach_world")
    jaw = _perpendicular_jaw(approach, _direction(jaw_line_world, f"{prefix}.jaw_line_world"),
                             f"{prefix}.jaw_line_world")
    local_approach = [float(item) for item in local_axes["approach"]]
    local_jaw = [float(item) for item in local_axes["closing"]]
    world = (approach, jaw, _cross(approach, jaw))
    local = (local_approach, local_jaw, _cross(local_approach, local_jaw))
    # R = W L^T maps each local column onto its world column.
    matrix = [[sum(world[k][row] * local[k][column] for k in range(3)) for column in range(3)]
              for row in range(3)]
    return matrix_quaternion(matrix)


def resolve_request(surface: Any, request: Mapping[str, Any], *, arm: str) -> Mapping[str, Any]:
    """Compile a ``world_axes`` orientation into the released absolute quaternion; pass anything else."""
    orientation = request.get("orientation") if isinstance(request, Mapping) else None
    if not isinstance(orientation, Mapping) or orientation.get("mode") != MODE:
        return request
    if not declared(surface):
        raise _reject(f"orientation_mode {MODE} is not declared on this surface", f"targets.{arm}.orientation_mode")
    quaternion = quaternion_from_world_axes(orientation.get("approach_world"), orientation.get("jaw_line_world"),
                                            local_axes=_local_axes(surface), arm=arm)
    return {**request, "orientation": {"mode": "absolute_quaternion", "quaternion_xyzw": quaternion}}


def resolve_targets(surface: Any, targets: Mapping[str, Any]) -> dict[str, Any]:
    """Every arm's request with any ``world_axes`` orientation compiled."""
    return {arm: resolve_request(surface, request, arm=arm) for arm, request in targets.items()}


def _clean(vector: Sequence[float]) -> list[float]:
    return [0.0 if abs(value) < 10 ** -_DIGITS else round(value, _DIGITS) for value in vector]


def target_axes_world(orientation_xyzw: Sequence[float], *, local_axes: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    """World directions of the approach axis and jaw line of one orientation."""
    matrix = quaternion_matrix(list(orientation_xyzw))

    def rotate(vector: Sequence[float]) -> list[float]:
        return _clean([sum(matrix[row][k] * float(vector[k]) for k in range(3)) for row in range(3)])

    return {"approach": rotate(local_axes["approach"]), "jaw_line": rotate(local_axes["closing"])}


def echo(surface: Any, target_pose: Mapping[str, Any]) -> dict[str, Any]:
    """``target_axes_world`` for one submitted target on a declaring surface, else nothing."""
    if not declared(surface):
        return {}
    return {"target_axes_world": target_axes_world(target_pose["orientation_xyzw"], local_axes=_local_axes(surface))}


__all__ = [
    "CAPABILITY", "MIN_DIRECTION_LENGTH", "MIN_SEPARATION_DEG", "MODE", "declared", "echo",
    "quaternion_from_world_axes", "resolve_request", "resolve_targets", "target_axes_world",
]
