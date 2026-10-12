"""Closed, observation-bound public measurements shared by policy runtimes.

This module reads only public observation artifacts. It never imports a
simulator or a task, exposes filesystem identities, or infers object state.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from ora.contracts.gripper_feedback import validate_gripper_feedback, validate_controller_feedback
from ora.agent.composition.embodiment_geometry import (
    PUBLIC_GEOMETRY_DESCRIPTORS, public_local_axes,
)
from .action_views import PostActionObservationError, _safe_file
from .observation_projection import (
    AUXILIARY_CONTACT_SHEET_CAMERAS, AUXILIARY_CONTACT_SHEET_VIEW,
    CONTACT_SHEET_HEADER_HEIGHT,
)

PUBLIC_FACTS_SCHEMA = "ora.current-public-robot-facts.v1"
PUBLIC_FACTS_DELIVERY = "all_public_v1"


class PublicFactsError(ValueError):
    """A declared public measurement is malformed or cannot be verified."""


def _measured_pose(value: Any, name: str) -> list[float]:
    """A finite [x, y, z, qx, qy, qz, qw] pose with its quaternion normalized."""

    message = f"{name} must contain 7 finite numbers"
    if not isinstance(value, (list, tuple)) or len(value) != 7 or any(
        isinstance(item, bool) or not isinstance(item, (int, float)) for item in value
    ):
        raise ValueError(message)
    try:
        pose = [float(item) for item in value]
    except (OverflowError, ValueError) as exc:
        raise ValueError(message) from exc
    if not all(math.isfinite(item) for item in pose):
        raise ValueError(message)
    norm = math.hypot(*pose[3:])
    if abs(norm - 1.) > 1e-3:
        raise ValueError(f"{name} quaternion must be normalized")
    return [*pose[:3], *(component / norm for component in pose[3:])]


def full_public_delivery(view_config: Mapping[str, Any] | None) -> bool:
    return isinstance(view_config, Mapping) and view_config.get("current_observation_delivery") == PUBLIC_FACTS_DELIVERY


GRIPPER_COMMAND_DELIVERY = "commanded_beside_measured_v1"
CONTROLLED_POINT_PROJECTION = "per_camera_pixel_v1"


def controlled_point_projection_delivery(view_config: Mapping[str, Any] | None) -> bool:
    """Whether the surface publishes each arm's TCP pixel in every camera."""
    return (full_public_delivery(view_config)
            and view_config.get("controlled_point_projection") == CONTROLLED_POINT_PROJECTION)


def project_controlled_point(intrinsic: list[list[float]], world_to_camera: list[list[float]],
                             point_xyz: Any, *, width: int, height: int) -> dict[str, Any]:
    """Project one public world point with the same camera's public calibration.

    Pixels are native OpenCV coordinates of that camera. A point behind the
    camera has no pixel; a point in front may still fall outside the image,
    which is itself the information a wrist-camera policy needs.
    """
    x, y, z = (_finite(value, "controlled point coordinate") for value in point_xyz)
    camera = [sum(world_to_camera[row][col] * value for col, value in enumerate((x, y, z, 1.0)))
              for row in range(3)]
    if camera[2] <= 0:
        return {"pixel_xy": None, "inside_image": False, "in_front_of_camera": False,
                "depth_m": camera[2]}
    u = intrinsic[0][0] * camera[0] / camera[2] + intrinsic[0][2]
    v = intrinsic[1][1] * camera[1] / camera[2] + intrinsic[1][2]
    return {"pixel_xy": [u, v], "inside_image": 0 <= u < width and 0 <= v < height,
            "in_front_of_camera": True, "depth_m": camera[2]}


def gripper_command_delivery(view_config: Mapping[str, Any] | None) -> bool:
    """Whether the surface publishes the commanded jaw target beside the encoder."""
    return (full_public_delivery(view_config)
            and view_config.get("gripper_command_delivery") == GRIPPER_COMMAND_DELIVERY)


def _finite(value: Any, label: str) -> float:
    if type(value) not in (float, int) or not math.isfinite(value):
        raise PublicFactsError(f"{label} must be finite numeric evidence")
    return float(value)


def _matrix(value: Any, rows: int, columns: int, label: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != rows or any(
        not isinstance(row, list) or len(row) != columns for row in value
    ):
        raise PublicFactsError(f"{label} has invalid dimensions")
    return [[_finite(x, label) for x in row] for row in value]


def _read_bound_json(root: Path, record: Mapping[str, Any]) -> Any:
    if not isinstance(record.get("path"), str):
        raise PublicFactsError("public artifact lacks a relative file identity")
    try:
        path = _safe_file(root, record["path"])
        raw = path.read_bytes()
    except (OSError, ValueError, PostActionObservationError) as exc:
        raise PublicFactsError("public artifact is unavailable") from exc
    if hashlib.sha256(raw).hexdigest() != record.get("sha256"):
        raise PublicFactsError("public artifact hash mismatch")
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise PublicFactsError("public artifact is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise PublicFactsError("public artifact must be an object")
    return value


def world_axes(pose: list[float], convention: str) -> dict[str, list[float]]:
    pose = _measured_pose(pose, "public measured pose")
    if convention not in PUBLIC_GEOMETRY_DESCRIPTORS:
        raise PublicFactsError("unsupported controlled-frame convention")
    x, y, z, w = pose[3:]
    norm = math.sqrt(x*x + y*y + z*z + w*w)
    x, y, z, w = (value / norm for value in (x, y, z, w))
    rotation = [
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ]
    return {name + "_world": [sum(row[i]*axis[i] for i in range(3)) for row in rotation]
            for name, axis in public_local_axes(convention).items()}


def public_end_effectors(proprioception: Mapping[str, Any], convention: str,
                         *, gripper_command_delivered: bool = False) -> dict[str, Any]:
    effectors = proprioception.get("end_effectors")
    expected = {"right"} if convention == "panda_grip_site" else {"left", "right"}
    if not isinstance(effectors, Mapping) or set(effectors) != expected:
        raise PublicFactsError("public end-effector topology differs from the declared convention")
    result = {}
    for arm in sorted(effectors):
        eef = effectors[arm]
        if not isinstance(eef, Mapping):
            raise PublicFactsError("public end-effector state is invalid")
        if convention == "panda_grip_site":
            valid_frame = eef.get("pose_frame") == "robosuite_grip_site"
        elif convention == "arx_x5_link6":
            valid_frame = eef.get("pose_frame") == "robodojo_episode_local_ee_link"
        else:
            valid_frame = eef.get("controlled_point_semantics") == "parallel_jaw_grip_center_tcp"
        if not valid_frame:
            raise PublicFactsError("measured controlled point differs from the declared convention")
        pose = _measured_pose(eef.get("pose_xyz_xyzw"), "public measured pose")
        value = {"pose_xyz_xyzw": pose, "position_frame": "world", "position_unit": "metre",
                 "orientation_order": "xyzw", "controlled_frame": convention,
                 "axes": world_axes(pose, convention)}
        if convention == "panda_grip_site":
            qpos = eef.get("gripper_finger_qpos")
            if not isinstance(qpos, list) or len(qpos) != 2:
                raise PublicFactsError("Panda requires both measured finger displacements")
            fingers = [_finite(q, "finger displacement") for q in qpos]
            command = eef.get("gripper_command")
            value["gripper"] = {
                "measured_finger_displacements_m": fingers,
                "measured_aperture_mm": sum(abs(q) for q in fingers) * 1000.,
                "measured_opening_normalized": _finite(eef.get("gripper"), "normalized opening"),
                "measurement_semantics": "two_prismatic_finger_displacements_not_command",
                "grasp_evidence": "not_established_by_joint_measurement",
            }
            if gripper_command_delivered:
                # The command sits beside the measurement, never instead of
                # it: "did my close take?" needs both numbers, and a backend
                # publishing only the encoder leaves the policy holding one of
                # the two. Absent stays absent rather than inferred.
                value["gripper"].update({
                    "commanded_opening_normalized": (
                        _finite(command, "commanded opening")
                        if command is not None else None
                    ),
                    "command_target": eef.get("gripper_command_target"),
                })
        elif convention == "arx_x5_link6":
            qpos = eef.get("gripper_finger_qpos")
            if not isinstance(qpos, list) or len(qpos) != 2:
                raise PublicFactsError("ARX requires both measured finger displacements")
            value["gripper"] = {
                "measured_finger_displacements_m": [_finite(q, "finger displacement") for q in qpos],
                "measured_opening_normalized": _finite(eef.get("gripper"), "normalized opening"),
                "commanded_opening_normalized": _finite(eef.get("gripper_command"), "commanded opening"),
                "command_target": eef.get("gripper_command_target"),
                "measured_aperture_mm": None,
                "measurement_semantics": "prismatic_encoders_nominal_geometry_not_calibrated_pad_aperture",
                "grasp_evidence": "not_established_by_joint_measurement",
            }
        else:
            source = eef.get("gripper_state")
            state = {"status": "unavailable", "opening_semantics": "normalized_closed_0_open_1_not_aperture_m",
                     "grasp_evidence": "not_established_by_joint_measurement"}
            if isinstance(source, Mapping):
                for key in ("commanded_opening_normalized", "drive_target_opening_normalized",
                            "measured_opening_normalized", "measured_primary_joint_position"):
                    raw = source.get(key)
                    state[key] = None if raw is None else _finite(raw, key)
                state["status"] = "reported" if state["measured_opening_normalized"] is not None else "unavailable"
            command = eef.get("gripper_command_target", "unknown")
            if command not in {"open", "close", "unknown"}:
                raise PublicFactsError("invalid public gripper command")
            state["command_target"] = command
            value["gripper"] = state
        if "gripper_feedback" in eef:
            try:
                feedback = validate_gripper_feedback(eef["gripper_feedback"])
            except ValueError as error:
                raise PublicFactsError(str(error)) from error
            if feedback["measured_opening_normalized"] != value["gripper"]["measured_opening_normalized"]:
                raise PublicFactsError("gripper feedback differs from the current encoder")
            # Shared projection: preserve the source contract for any backend
            # supplying it, without borrowing another embodiment's bounds.
            value["gripper"].update(
                commanded_opening_normalized=feedback["requested_opening_normalized"],
                command_target=feedback["requested_command"], feedback=feedback,
            )
        result[arm] = value
    return result


def project_current_public_facts(*, observation: Mapping[str, Any], observation_root: Path,
                                 view_config: Mapping[str, Any],
                                 selected_views: list[str] | None = None,
                                 snapshot_role: str = "current",
                                 gripper_command_delivered: bool = False,
                                 controlled_point_projection: bool = False) -> dict[str, Any]:
    """Project all declared current cameras and their exact public calibration.

    Each matrix refers to its physical camera's native RGB coordinates. Sheet
    coordinates are translated by an explicit offset; there is no resampling.
    Missing calibration is disclosed, while changed/malformed artifacts fail.

    ``gripper_command_delivered`` publishes the Panda commanded target beside
    the encoder reading. It defaults to off because the result schema for this
    block is a closed ``oneOf`` shared with frozen surfaces: a surface that has
    not declared the field would reject the observation at runtime rather than
    ignore it. Callers turn it on from ``gripper_command_delivery(view_config)``,
    never by default.
    """
    if snapshot_role not in {"current", "historical"}:
        raise PublicFactsError("unsupported snapshot role")
    observation_id = observation.get("observation_id")
    if not isinstance(observation_id, str) or not observation_id:
        raise PublicFactsError("public facts require an observation identity")
    from ora.contracts.robot_geometry import GEOMETRY_VERIFICATION_SCHEMA
    verification = observation.get("robot_geometry_verification")
    descriptor = view_config.get("public_geometry_descriptor")
    if (not isinstance(verification, Mapping) or not isinstance(descriptor, Mapping)
            or verification.get("schema") != GEOMETRY_VERIFICATION_SCHEMA
            or verification.get("descriptor_sha256") != descriptor.get("sha256")
            or verification.get("scope") != "loaded_robot_geometry_only"
            or verification.get("passed") is not True):
        raise PublicFactsError("current robot geometry is not verified against the selected descriptor")
    root = observation_root.resolve()
    try:
        manifest = _safe_file(root, "observation.json")
        if json.loads(manifest.read_text()) != dict(observation):
            raise PublicFactsError("observation manifest differs from current evidence")
    except (OSError, UnicodeError, PostActionObservationError) as exc:
        raise PublicFactsError("public observation manifest is unavailable") from exc
    cameras = observation.get("cameras")
    if not isinstance(cameras, Mapping) or not cameras:
        raise PublicFactsError("current observation has no public cameras")
    views = view_config["automatic_actual_views"] if selected_views is None else selected_views
    sheet = list(view_config.get("auxiliary_contact_sheet_cameras") or AUXILIARY_CONTACT_SHEET_CAMERAS)
    covered = []
    for view in views:
        covered.extend(sheet if view == AUXILIARY_CONTACT_SHEET_VIEW else [view])
    if (len(covered) != len(set(covered)) or not set(covered) <= set(cameras)
            or selected_views is None and set(covered) != set(cameras)):
        raise PublicFactsError("all-public delivery must cover each physical camera exactly once")
    sizes = {}
    for name, camera in cameras.items():
        size = [camera.get("width"), camera.get("height")]
        if any(type(n) is not int or n <= 0 for n in size):
            raise PublicFactsError("public camera dimensions are invalid")
        sizes[name] = size
    cell_width = max(sizes[name][0] for name in sheet) if AUXILIARY_CONTACT_SHEET_VIEW in views else 0
    cell_height = max(sizes[name][1] for name in sheet) + CONTACT_SHEET_HEADER_HEIGHT if cell_width else 0
    projected = {}
    for name in covered:
        width, height = sizes[name]
        entry = {"width": width, "height": height, "pixel_coordinates": "opencv_top_left_xy",
                 "image_view": name, "native_pixel_to_image_offset_xy": [0, 0], "resampling": "none"}
        if AUXILIARY_CONTACT_SHEET_VIEW in views and name in sheet:
            index = sheet.index(name)
            entry.update(image_view=AUXILIARY_CONTACT_SHEET_VIEW,
                native_pixel_to_image_offset_xy=[(index % 2)*cell_width+(cell_width-width)//2,
                                                (index // 2)*cell_height+CONTACT_SHEET_HEADER_HEIGHT])
        calibration_ref = cameras[name].get("files", {}).get("calibration")
        entry["calibration"] = {"status": "unavailable"}
        if isinstance(calibration_ref, Mapping):
            calibration = _read_bound_json(root, calibration_ref)
            intrinsic = _matrix(calibration.get("intrinsic_cv"), 3, 3, "intrinsic_cv")
            raw_extrinsic = calibration.get("extrinsic_cv")
            if isinstance(raw_extrinsic, list) and len(raw_extrinsic) == 3:
                raw_extrinsic = [*raw_extrinsic, [0, 0, 0, 1]]
            extrinsic = _matrix(raw_extrinsic, 4, 4, "extrinsic_cv")
            if (intrinsic[0][0] <= 0 or intrinsic[1][1] <= 0
                    or intrinsic[2] != [0, 0, 1] or extrinsic[3] != [0, 0, 0, 1]):
                raise PublicFactsError("public camera calibration is invalid")
            entry["calibration"] = {"status": "reported", "intrinsic_cv": intrinsic,
                "world_to_camera_cv": extrinsic, "camera_axes": "x_right_y_down_z_forward",
                "world_position_unit": "metre", "pixel_basis": "native_rgb_before_sheet_translation"}
        projected[name] = entry
    if controlled_point_projection:
        effectors = observation.get("proprioception", {}).get("end_effectors", {})
        for name, entry in projected.items():
            calibration = entry["calibration"]
            if calibration.get("status") != "reported":
                continue
            entry["controlled_point_pixels"] = {
                arm: project_controlled_point(
                    calibration["intrinsic_cv"], calibration["world_to_camera_cv"],
                    effector["pose_xyz_xyzw"][:3], width=entry["width"], height=entry["height"])
                for arm, effector in sorted(effectors.items())
                if isinstance(effector, Mapping) and isinstance(effector.get("pose_xyz_xyzw"), list)
            }
    controller = None
    if "controller_feedback" in observation:
        try:
            controller = validate_controller_feedback(observation["controller_feedback"])
        except ValueError as error:
            raise PublicFactsError(str(error)) from error
        effectors = observation.get("proprioception", {}).get("end_effectors", {})
        if any(not isinstance(effectors.get(arm), Mapping) or "gripper_feedback" not in effectors[arm]
               for arm in ("left", "right")):
            raise PublicFactsError("controller feedback requires both arm gripper snapshots")
    return {"schema": PUBLIC_FACTS_SCHEMA, "observation_id": observation_id,
            "snapshot_role": snapshot_role,
            **({"controller_feedback": controller} if controller is not None else {}),
            "cameras": projected,
            "end_effectors": public_end_effectors(
                observation.get("proprioception", {}),
                view_config["controlled_frame_convention"],
                gripper_command_delivered=gripper_command_delivered)}
