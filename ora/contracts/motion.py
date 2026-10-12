"""Shared two-layer motion compilation and controller thresholds."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .action import SERVO_MAX_ROTATION_RAD, SERVO_MAX_TRANSLATION_M
from .pose import compile_absolute_pose, quaternion_matrix

POSE_POSITION_TOLERANCE_M = 0.005
POSE_ORIENTATION_TOLERANCE_RAD = 0.05


def _vector(value: Any, *, length: int, name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain {length} numbers")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain finite numbers")
    return result


def _quaternion_multiply(left: Sequence[float], right: Sequence[float]) -> list[float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    value = [
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ]
    norm = math.sqrt(sum(item * item for item in value))
    if norm < 1e-12:
        raise ValueError("relative rotation produced an invalid orientation")
    return [item / norm for item in value]


def _rotvec_quaternion(rotvec: Sequence[float]) -> list[float]:
    angle = math.sqrt(sum(item * item for item in rotvec))
    if angle < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    scale = math.sin(angle / 2.0) / angle
    return [
        rotvec[0] * scale,
        rotvec[1] * scale,
        rotvec[2] * scale,
        math.cos(angle / 2.0),
    ]


def compile_enhanced_move_pose(
    request: Mapping[str, Any], *, current_pose_xyz_xyzw: Sequence[float],
    controlled_point: str = "robosuite_grip_site",
) -> dict[str, Any]:
    """Compile one planner-backed absolute, relative, or hold pose request.

    This is intentionally not a Cartesian-servo contract.  Relative inputs are
    ergonomic policy syntax that resolve once against the measured action-start
    pose; the backend still receives one complete absolute planner target.
    """

    if not isinstance(request, Mapping):
        raise ValueError("move_pose target must be an object")
    if set(request) == {"position_m", "orientation_xyzw"}:
        # Pose-producing tools use one canonical public TCP shape. Compile it
        # through the same absolute path as the ergonomic form so proposal and
        # execution tools share one lossless handoff contract.
        request = {
            "position": {
                "mode": "absolute", "position_m": request["position_m"],
            },
            "orientation": {
                "mode": "absolute_quaternion",
                "quaternion_xyzw": request["orientation_xyzw"],
            },
        }
    elif set(request) != {"position", "orientation"}:
        raise ValueError(
            "move_pose target must be either a standard absolute TCP pose "
            "or contain enhanced position and orientation objects"
        )

    current = _vector(current_pose_xyz_xyzw, length=7, name="current_pose_xyz_xyzw")
    current_position = current[:3]
    current_orientation = current[3:]

    position = request["position"]
    if not isinstance(position, Mapping):
        raise ValueError("move_pose position must be an object")
    position_mode = position.get("mode")
    if position_mode == "hold_current":
        if set(position) != {"mode"}:
            raise ValueError("hold_current position accepts only mode")
        target_position = list(current_position)
        position_frame = None
    elif position_mode == "absolute":
        if set(position) != {"mode", "position_m"}:
            raise ValueError("absolute position requires position_m")
        target_position = _vector(
            position["position_m"], length=3, name="position_m",
        )
        position_frame = "world"
    elif position_mode == "delta":
        if set(position) != {"mode", "translation_m", "frame"}:
            raise ValueError(
                "delta position requires translation_m and frame"
            )
        displacement = _vector(
            position["translation_m"], length=3, name="translation_m",
        )
        position_frame = position["frame"]
        if position_frame not in {"world", "end_effector"}:
            raise ValueError("position delta frame must be world or end_effector")
        if position_frame == "end_effector":
            matrix = quaternion_matrix(current_orientation)
            displacement = [
                sum(matrix[row][column] * displacement[column] for column in range(3))
                for row in range(3)
            ]
        target_position = [
            current_position[index] + displacement[index] for index in range(3)
        ]
    else:
        raise ValueError(
            "move_pose position mode must be hold_current, absolute, or delta"
        )

    orientation = request["orientation"]
    if not isinstance(orientation, Mapping):
        raise ValueError("move_pose orientation must be an object")
    orientation_mode = orientation.get("mode")
    if orientation_mode == "hold_current":
        if set(orientation) != {"mode"}:
            raise ValueError("hold_current orientation accepts only mode")
        target_orientation = list(current_orientation)
        orientation_frame = None
    elif orientation_mode == "absolute_quaternion":
        if set(orientation) != {"mode", "quaternion_xyzw"}:
            raise ValueError(
                "absolute_quaternion orientation requires quaternion_xyzw"
            )
        target_orientation = _vector(
            orientation["quaternion_xyzw"], length=4, name="quaternion_xyzw",
        )
        orientation_frame = "world"
    elif orientation_mode == "axis_angle_delta":
        if set(orientation) != {"mode", "axis", "angle_deg", "frame"}:
            raise ValueError(
                "axis_angle_delta orientation requires axis, angle_deg and frame"
            )
        axis = orientation["axis"]
        if axis not in {"x", "y", "z"}:
            raise ValueError("axis must be x, y, or z")
        raw_angle = orientation["angle_deg"]
        if (
            isinstance(raw_angle, bool)
            or not isinstance(raw_angle, (int, float))
            or not math.isfinite(float(raw_angle))
            or abs(float(raw_angle)) > 180.0
        ):
            raise ValueError("angle_deg must be finite and within [-180, 180]")
        orientation_frame = orientation["frame"]
        if orientation_frame not in {"world", "end_effector"}:
            raise ValueError(
                "orientation delta frame must be world or end_effector"
            )
        components = {
            "x": [1.0, 0.0, 0.0],
            "y": [0.0, 1.0, 0.0],
            "z": [0.0, 0.0, 1.0],
        }[axis]
        angle_rad = math.radians(float(raw_angle))
        delta_quaternion = _rotvec_quaternion([
            component * angle_rad for component in components
        ])
        target_orientation = (
            _quaternion_multiply(current_orientation, delta_quaternion)
            if orientation_frame == "end_effector"
            else _quaternion_multiply(delta_quaternion, current_orientation)
        )
    else:
        raise ValueError(
            "move_pose orientation mode must be hold_current, "
            "absolute_quaternion, or axis_angle_delta"
        )

    if position_mode == orientation_mode == "hold_current":
        raise ValueError("move_pose cannot hold both position and orientation")
    compiled = compile_absolute_pose(
        current_pose_xyz_xyzw=current,
        target_pose_xyz_xyzw=[*target_position, *target_orientation],
        controlled_point=controlled_point,
    )
    if (
        float(compiled["translation_distance_m"]) < 1e-12
        and float(compiled["rotation_angle_rad"]) < 1e-12
    ):
        raise ValueError("move_pose target does not change the measured pose")
    return {
        **compiled,
        "position_mode": position_mode,
        "orientation_mode": orientation_mode,
        "position_delta_frame": position_frame if position_mode == "delta" else None,
        "orientation_delta_frame": (
            orientation_frame if orientation_mode == "axis_angle_delta" else None
        ),
    }


def compile_servo_delta(
    request: Mapping[str, Any], *, current_pose_xyz_xyzw: Sequence[float],
    controlled_point: str = "robosuite_grip_site",
) -> dict[str, Any]:
    """Compile one bounded world/tool-frame Cartesian delta into an absolute pose."""

    if not isinstance(request, Mapping) or set(request) != {
        "translation_m", "rotation_rotvec_rad", "frame",
    }:
        raise ValueError(
            "servo request must contain translation_m, rotation_rotvec_rad and frame"
        )
    current = _vector(current_pose_xyz_xyzw, length=7, name="current_pose_xyz_xyzw")
    translation = _vector(request["translation_m"], length=3, name="translation_m")
    rotation = _vector(
        request["rotation_rotvec_rad"], length=3, name="rotation_rotvec_rad"
    )
    if math.sqrt(sum(item * item for item in translation)) > SERVO_MAX_TRANSLATION_M + 1e-12:
        raise ValueError("servo translation exceeds 0.05 m")
    if math.sqrt(sum(item * item for item in rotation)) > SERVO_MAX_ROTATION_RAD + 1e-12:
        raise ValueError("servo rotation exceeds 15 degrees")
    frame = request["frame"]
    if frame not in {"world", "grip_site", "end_effector"}:
        raise ValueError("servo frame must be world or end_effector")
    local_frame = frame in {"grip_site", "end_effector"}
    current_position = current[:3]
    current_orientation = current[3:]
    delta_quaternion = _rotvec_quaternion(rotation)
    if local_frame:
        matrix = quaternion_matrix(current_orientation)
        translation_world = [
            sum(matrix[row][column] * translation[column] for column in range(3))
            for row in range(3)
        ]
        target_orientation = _quaternion_multiply(
            current_orientation, delta_quaternion
        )
    else:
        translation_world = translation
        target_orientation = _quaternion_multiply(
            delta_quaternion, current_orientation
        )
    target_position = [
        current_position[index] + translation_world[index] for index in range(3)
    ]
    compiled = compile_absolute_pose(
        current_pose_xyz_xyzw=current,
        target_pose_xyz_xyzw=[*target_position, *target_orientation],
        controlled_point=controlled_point,
    )
    return {
        **compiled,
        "servo_frame": frame,
        "requested_translation_m": translation,
        "requested_rotation_rotvec_rad": rotation,
        "resolved_translation_world_m": translation_world,
    }


__all__ = [
    "POSE_ORIENTATION_TOLERANCE_RAD",
    "POSE_POSITION_TOLERANCE_M",
    "compile_enhanced_move_pose",
    "compile_servo_delta",
]
