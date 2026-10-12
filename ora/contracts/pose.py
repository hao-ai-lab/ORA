"""Canonical, backend-neutral compilation of policy pose requests."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence


POSE_FRAME_CONTRACT_SCHEMA = "ora.pose-frame-contract.v2"
COMPILED_POSE_SCHEMA = "ora.compiled-pose.v1"
_AXIS_ORTHOGONAL_DOT_MAX = 0.05
_POSE_FIELDS = {
    "target_xyz_world_m",
    "translation_delta_mm",
    "translation_delta_frame",
    "target_orientation_world_xyzw",
    "grip_site_z_axis_world",
    "grip_site_x_axis_world",
}
_CANONICAL_DECIMALS = 12


class PoseCompilationError(ValueError):
    """Raised when a policy pose request has no unique physical meaning."""


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


POSE_FRAME_CONTRACT = {
    "schema": POSE_FRAME_CONTRACT_SCHEMA,
    "controlled_point": "robosuite_grip_site",
    "world_frame": {
        "handedness": "right_handed",
        "positive_z": "up",
        "position_unit": "metre",
    },
    "translation": {
        "absolute_field": "target_xyz_world_m",
        "delta_field": "translation_delta_mm",
        "delta_unit": "millimetre",
        "delta_frames": ["world", "grip_site"],
        "grip_site_component_axes": ["+X", "+Y", "+Z"],
    },
    "orientation": {
        "absolute_quaternion_field": "target_orientation_world_xyzw",
        "quaternion_order": "xyzw",
        "axis_fields": {
            "grip_site_z_axis_world": "grip_site_local_+Z_in_world",
            "grip_site_x_axis_world": "grip_site_local_+X_in_world",
        },
        "top_down_convention": "grip_site_local_+Z_toward_world_-Z",
        "maximum_abs_axis_dot": _AXIS_ORTHOGONAL_DOT_MAX,
    },
    "omitted_pose_dimension": "execution_start_measured_grip_site_pose",
    "pose_only_gripper_hold": "actuator_target",
    "canonical_numeric_decimals": _CANONICAL_DECIMALS,
}
POSE_FRAME_CONTRACT_SHA256 = _canonical_sha256(POSE_FRAME_CONTRACT)


def _finite_vector(value: Any, length: int, field: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise PoseCompilationError(f"{field} must contain {length} numbers")
    output: list[float] = []
    for item in value:
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
        ):
            raise PoseCompilationError(f"{field} must contain finite numbers")
        output.append(float(item))
    return output


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _cross(left: Sequence[float], right: Sequence[float]) -> list[float]:
    return [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]


def _norm(value: Sequence[float]) -> float:
    return math.sqrt(_dot(value, value))


def _stable_float(value: float) -> float:
    rounded = round(float(value), _CANONICAL_DECIMALS)
    return 0.0 if rounded == 0 else rounded


def _stable_vector(value: Sequence[float]) -> list[float]:
    return [_stable_float(item) for item in value]


def _unit(value: Sequence[float], field: str) -> list[float]:
    magnitude = _norm(value)
    if magnitude < 1e-9:
        raise PoseCompilationError(f"{field} must be a non-zero direction")
    return [component / magnitude for component in value]


def canonicalize_quaternion(value: Any, field: str) -> list[float]:
    quaternion = _finite_vector(value, 4, field)
    magnitude = _norm(quaternion)
    if not 0.98 <= magnitude <= 1.02:
        raise PoseCompilationError(f"{field} must be normalized")
    quaternion = [component / magnitude for component in quaternion]
    sign_reference = next(
        (component for component in reversed(quaternion) if abs(component) > 1e-12),
        1.0,
    )
    if sign_reference < 0:
        quaternion = [-component for component in quaternion]
    return [0.0 if abs(component) < 1e-15 else component for component in quaternion]


def quaternion_matrix(xyzw: Sequence[float]) -> list[list[float]]:
    x, y, z, w = canonicalize_quaternion(list(xyzw), "orientation_xyzw")
    return [
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ]


def matrix_quaternion(matrix: Sequence[Sequence[float]]) -> list[float]:
    """Return the canonical normalized xyzw quaternion for a 3x3 matrix."""
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2
        quaternion = [
            (matrix[2][1] - matrix[1][2]) / scale,
            (matrix[0][2] - matrix[2][0]) / scale,
            (matrix[1][0] - matrix[0][1]) / scale,
            0.25 * scale,
        ]
    else:
        index = max(range(3), key=lambda item: matrix[item][item])
        if index == 0:
            scale = math.sqrt(1 + matrix[0][0] - matrix[1][1] - matrix[2][2]) * 2
            quaternion = [
                0.25 * scale,
                (matrix[0][1] + matrix[1][0]) / scale,
                (matrix[0][2] + matrix[2][0]) / scale,
                (matrix[2][1] - matrix[1][2]) / scale,
            ]
        elif index == 1:
            scale = math.sqrt(1 + matrix[1][1] - matrix[0][0] - matrix[2][2]) * 2
            quaternion = [
                (matrix[0][1] + matrix[1][0]) / scale,
                0.25 * scale,
                (matrix[1][2] + matrix[2][1]) / scale,
                (matrix[0][2] - matrix[2][0]) / scale,
            ]
        else:
            scale = math.sqrt(1 + matrix[2][2] - matrix[0][0] - matrix[1][1]) * 2
            quaternion = [
                (matrix[0][2] + matrix[2][0]) / scale,
                (matrix[1][2] + matrix[2][1]) / scale,
                0.25 * scale,
                (matrix[1][0] - matrix[0][1]) / scale,
            ]
    return canonicalize_quaternion(quaternion, "resolved_orientation_xyzw")


def _columns(matrix: Sequence[Sequence[float]]) -> tuple[list[float], ...]:
    return tuple([[matrix[row][column] for row in range(3)] for column in range(3)])


def _fallback_perpendicular(axis: Sequence[float]) -> list[float]:
    candidate = min(
        ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
        key=lambda item: abs(_dot(axis, item)),
    )
    projected = [
        candidate[index] - axis[index] * _dot(candidate, axis)
        for index in range(3)
    ]
    return _unit(projected, "resolved_perpendicular_axis")


def _resolve_axes_orientation(
    *, current_orientation: Sequence[float], z_raw: Any, x_raw: Any,
) -> list[float]:
    current_x, _current_y, current_z = _columns(quaternion_matrix(current_orientation))
    supplied_z = z_raw is not None
    supplied_x = x_raw is not None
    z_axis = _unit(
        _finite_vector(z_raw, 3, "grip_site_z_axis_world")
        if supplied_z else current_z,
        "grip_site_z_axis_world",
    )
    x_axis = _unit(
        _finite_vector(x_raw, 3, "grip_site_x_axis_world")
        if supplied_x else current_x,
        "grip_site_x_axis_world",
    )
    if supplied_z and supplied_x:
        absolute_dot = abs(_dot(x_axis, z_axis))
        if absolute_dot > _AXIS_ORTHOGONAL_DOT_MAX:
            raise PoseCompilationError(
                "grip_site_x_axis_world and grip_site_z_axis_world must be "
                "approximately orthogonal"
            )
        x_axis = _unit(
            [x_axis[index] - z_axis[index] * _dot(x_axis, z_axis) for index in range(3)],
            "grip_site_x_axis_world",
        )
    elif supplied_z:
        projected = [
            current_x[index] - z_axis[index] * _dot(current_x, z_axis)
            for index in range(3)
        ]
        x_axis = (
            _unit(projected, "resolved_grip_site_x_axis_world")
            if _norm(projected) >= 1e-9 else _fallback_perpendicular(z_axis)
        )
    elif supplied_x:
        projected = [
            current_z[index] - x_axis[index] * _dot(current_z, x_axis)
            for index in range(3)
        ]
        z_axis = (
            _unit(projected, "resolved_grip_site_z_axis_world")
            if _norm(projected) >= 1e-9 else _fallback_perpendicular(x_axis)
        )
    y_axis = _unit(_cross(z_axis, x_axis), "resolved_grip_site_y_axis_world")
    x_axis = _unit(_cross(y_axis, z_axis), "resolved_grip_site_x_axis_world")
    matrix = [[x_axis[row], y_axis[row], z_axis[row]] for row in range(3)]
    return matrix_quaternion(matrix)


def _rotate(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    return [_dot(row, vector) for row in matrix]


def _semantic_compilation(
    *, current_position: Sequence[float], current_orientation: Sequence[float],
    target_position: Sequence[float], target_orientation: Sequence[float],
    controlled_point: str,
) -> dict[str, Any]:
    current_p = _stable_vector(current_position)
    target_p = _stable_vector(target_position)
    current_q = _stable_vector(canonicalize_quaternion(
        list(current_orientation), "current_orientation_xyzw",
    ))
    target_q = _stable_vector(canonicalize_quaternion(
        list(target_orientation), "target_orientation_xyzw",
    ))
    translation = _stable_vector([
        target_p[index] - current_p[index]
        for index in range(3)
    ])
    quaternion_dot = min(1.0, abs(_dot(current_q, target_q)))
    rotation_rad = 2.0 * math.acos(quaternion_dot)
    target_x, target_y, target_z = _columns(quaternion_matrix(target_q))
    semantic = {
        "schema": COMPILED_POSE_SCHEMA,
        "pose_contract_sha256": POSE_FRAME_CONTRACT_SHA256,
        "controlled_point": controlled_point,
        "current_pose_xyz_xyzw": [*current_p, *current_q],
        "target_pose_xyz_xyzw": [*target_p, *target_q],
        "translation_world_m": translation,
        "translation_distance_m": _stable_float(_norm(translation)),
        "rotation_angle_rad": _stable_float(rotation_rad),
        "rotation_angle_deg": _stable_float(math.degrees(rotation_rad)),
        "resolved_grip_site_axes_world": {
            "x": _stable_vector(target_x),
            "y": _stable_vector(target_y),
            "z": _stable_vector(target_z),
        },
        "pose_only_gripper_hold": "actuator_target",
    }
    return {**semantic, "compiled_pose_sha256": _canonical_sha256(semantic)}


def compile_pose_request(
    request: Mapping[str, Any], *, current_pose_xyz_xyzw: Sequence[float],
    controlled_point: str = "robosuite_grip_site",
) -> dict[str, Any]:
    """Resolve one v2 policy request into one immutable absolute pose."""

    if not isinstance(request, Mapping):
        raise PoseCompilationError("pose request must be an object")
    unknown = set(request) - _POSE_FIELDS
    if unknown:
        raise PoseCompilationError(f"pose request contains unknown fields: {sorted(unknown)}")
    current = _finite_vector(list(current_pose_xyz_xyzw), 7, "current_pose_xyz_xyzw")
    current_position = current[:3]
    current_orientation = canonicalize_quaternion(current[3:], "current_orientation_xyzw")
    absolute_position = request.get("target_xyz_world_m")
    delta = request.get("translation_delta_mm")
    if absolute_position is not None and delta is not None:
        raise PoseCompilationError(
            "provide target_xyz_world_m or translation_delta_mm, not both"
        )
    if "translation_delta_frame" in request and delta is None:
        raise PoseCompilationError(
            "translation_delta_frame requires translation_delta_mm"
        )
    target_position = list(current_position)
    if absolute_position is not None:
        target_position = _finite_vector(
            absolute_position, 3, "target_xyz_world_m",
        )
    elif delta is not None:
        displacement = [
            component / 1000.0
            for component in _finite_vector(delta, 3, "translation_delta_mm")
        ]
        frame = request.get("translation_delta_frame", "world")
        if frame not in {"world", "grip_site"}:
            raise PoseCompilationError(
                "translation_delta_frame must be world or grip_site"
            )
        if frame == "grip_site":
            displacement = _rotate(quaternion_matrix(current_orientation), displacement)
        target_position = [
            current_position[index] + displacement[index] for index in range(3)
        ]
    quaternion = request.get("target_orientation_world_xyzw")
    z_axis = request.get("grip_site_z_axis_world")
    x_axis = request.get("grip_site_x_axis_world")
    if quaternion is not None and (z_axis is not None or x_axis is not None):
        raise PoseCompilationError(
            "provide target_orientation_world_xyzw or grip-site axes, not both"
        )
    target_orientation = list(current_orientation)
    if quaternion is not None:
        target_orientation = canonicalize_quaternion(
            quaternion, "target_orientation_world_xyzw",
        )
    elif z_axis is not None or x_axis is not None:
        target_orientation = _resolve_axes_orientation(
            current_orientation=current_orientation, z_raw=z_axis, x_raw=x_axis,
        )
    if (
        absolute_position is None and delta is None and quaternion is None
        and z_axis is None and x_axis is None
    ):
        raise PoseCompilationError(
            "pose request requires a position, translation delta, quaternion, or axis"
        )
    compiled = _semantic_compilation(
        current_position=current_position,
        current_orientation=current_orientation,
        target_position=target_position,
        target_orientation=target_orientation,
        controlled_point=controlled_point,
    )
    return {
        **compiled,
        "normalized_request": {
            key: request[key] for key in sorted(request) if request[key] is not None
        },
        "position_was_omitted": absolute_position is None and delta is None,
        "orientation_was_omitted": quaternion is None and z_axis is None and x_axis is None,
    }


def compile_absolute_pose(
    *, current_pose_xyz_xyzw: Sequence[float], target_pose_xyz_xyzw: Sequence[float],
    controlled_point: str = "robosuite_grip_site",
) -> dict[str, Any]:
    """Rebuild the semantic compiled-pose identity at the execution boundary."""

    current = _finite_vector(list(current_pose_xyz_xyzw), 7, "current_pose_xyz_xyzw")
    target = _finite_vector(list(target_pose_xyz_xyzw), 7, "target_pose_xyz_xyzw")
    return _semantic_compilation(
        current_position=current[:3], current_orientation=current[3:],
        target_position=target[:3], target_orientation=target[3:],
        controlled_point=controlled_point,
    )


__all__ = [
    "COMPILED_POSE_SCHEMA",
    "POSE_FRAME_CONTRACT",
    "POSE_FRAME_CONTRACT_SCHEMA",
    "POSE_FRAME_CONTRACT_SHA256",
    "PoseCompilationError",
    "canonicalize_quaternion",
    "compile_absolute_pose",
    "compile_pose_request",
    "quaternion_matrix",
]
