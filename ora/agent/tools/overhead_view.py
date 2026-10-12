"""Footprint centre of a segmented target, and the wrist pose that views it from overhead.

``center_xyz_m`` is the median of the target's visible depth points. From an
oblique camera those points are the camera-facing side and top, so the median
sits horizontally toward the camera by up to half the object's depth: on the
LIBERO mugs it was 2.75 cm off the body origin (probe M, 2026-10-09), while a
stacking check such as LIBERO's ``On`` passes only within about 3 cm, so placing
by the median fails most of the time. The visible-point AABB midpoint is less
biased but is cut short by an occluding neighbour.

This module adds two things:

- ``footprint_center_xyz_m``: the centre of the occupied-cell box, in its own
  principal frame, of the target's points deprojected from its mask eroded by
  one pixel (boundary pixels carry background depth). From an overhead view the
  visible points are the whole top outline, and this centre was within 0.2 cm
  of the body origin on the LIBERO mugs (handles included) and bowls; from an
  oblique view it is still a near-side estimate. ``view_tilt_deg`` (the angle
  between the camera-to-target ray and straight down) and ``footprint_view``
  (``overhead`` at or below :data:`OVERHEAD_MAX_TILT_DEG`, else ``oblique``)
  say which case a measurement is.
- ``overhead_view_pose``: for an oblique measurement, the grip-site pose that
  puts the wrist camera :data:`VIEW_HEIGHT_M` above the footprint centre,
  looking straight down with the current jaw yaw. The camera-in-grip transform
  is read from the current observation's wrist calibration and grip-site pose,
  so no embodiment constant is assumed. It is a copyable ``move_pose`` target,
  not a reachability claim.

A scripted grasp-carry-place probe (probe R, 2026-10-09) placed 35/35 LIBERO
mugs and bowls within ``On`` with the overhead measurement and 9/35 with the
oblique median. Delivery is gated on :data:`CAPABILITY`: the producers measure
the fields, and :func:`ora.agent.tools.grasp.project_target_frame`
removes them for surfaces that do not declare it.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

CAPABILITY = "overhead_footprint_localization_v1"
#: Localization keys this module owns, in publication order.
KEYS = ("footprint_center_xyz_m", "footprint_extent_m", "view_tilt_deg", "footprint_view")
#: Key added beside them for an oblique measurement when a wrist camera is current.
POSE_KEY = "overhead_view_pose"

#: A camera-to-target ray within this angle of straight down sees the top outline.
OVERHEAD_MAX_TILT_DEG = 20.0
#: Wrist-camera height above the footprint centre for the overhead view.
VIEW_HEIGHT_M = 0.30
#: Occupancy grid cell for the footprint box.
_CELL_M = 0.002
_MINIMUM_POINTS = 16


def empty() -> dict[str, Any]:
    """The published shape when no estimate is available."""
    return {"footprint_center_xyz_m": None, "footprint_extent_m": None,
            "view_tilt_deg": None, "footprint_view": None}


def footprint(world_points: Any, camera_position: Any) -> dict[str, Any]:
    """Footprint centre and extent of ``world_points`` (N x 3) seen from ``camera_position``."""
    import numpy as np

    points = np.asarray(world_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < _MINIMUM_POINTS:
        return empty()
    cells = np.unique(np.floor(points[:, :2] / _CELL_M).astype(np.int64), axis=0)
    centres = (cells + 0.5) * _CELL_M
    mean = centres.mean(axis=0)
    if len(centres) >= 3:
        _values, vectors = np.linalg.eigh(np.cov((centres - mean).T))
        axes = vectors[:, ::-1]
    else:
        axes = np.eye(2)
    projected = (centres - mean) @ axes
    low = projected.min(axis=0) - _CELL_M / 2
    high = projected.max(axis=0) + _CELL_M / 2
    centre_xy = mean + axes @ ((low + high) / 2)
    centre = np.array([centre_xy[0], centre_xy[1], float(np.median(points[:, 2]))])
    ray = centre - np.asarray(camera_position, dtype=np.float64)
    norm = float(np.linalg.norm(ray))
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, -ray[2] / norm)))) if norm > 0 else None
    extent = sorted((float(value) for value in high - low), reverse=True)
    return {
        "footprint_center_xyz_m": [float(value) for value in centre],
        "footprint_extent_m": extent,
        "view_tilt_deg": None if tilt is None else round(tilt, 1),
        "footprint_view": None if tilt is None else ("overhead" if tilt <= OVERHEAD_MAX_TILT_DEG else "oblique"),
    }


def _matrix(pose_xyz_xyzw: Any) -> Any:
    import numpy as np
    from scipy.spatial.transform import Rotation

    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat(pose_xyz_xyzw[3:]).as_matrix()
    transform[:3, 3] = pose_xyz_xyzw[:3]
    return transform


def view_pose(target_xyz: Any, grip_pose_xyz_xyzw: Any, wrist_camera_to_world: Any,
              *, height_m: float = VIEW_HEIGHT_M) -> dict[str, Any] | None:
    """Grip-site pose putting the wrist camera ``height_m`` above ``target_xyz``, looking straight down.

    The wrist camera's transform in the grip frame comes from the current grip-site
    pose and the same observation's wrist calibration. The grip approach axis
    (local +Z) points down and the jaw keeps its current yaw.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation

    grip = _matrix(grip_pose_xyz_xyzw)
    camera = np.asarray(wrist_camera_to_world, dtype=np.float64)
    if camera.shape == (3, 4):
        camera = np.vstack([camera, [0.0, 0.0, 0.0, 1.0]])
    if camera.shape != (4, 4) or not np.isfinite(camera).all():
        return None
    camera_in_grip = np.linalg.inv(grip) @ camera
    jaw = grip[:3, 0].copy()
    jaw[2] = 0.0
    if np.linalg.norm(jaw) < 1e-6:
        jaw = grip[:3, 1].copy()
        jaw[2] = 0.0
    jaw /= np.linalg.norm(jaw)
    approach = np.array([0.0, 0.0, -1.0])
    rotation = np.column_stack([jaw, np.cross(approach, jaw), approach])
    # Rotate the down-looking grip so the camera's own optical axis is the one pointing down.
    optical = rotation @ camera_in_grip[:3, 2]
    correction, _ = Rotation.align_vectors([[0.0, 0.0, -1.0]], [optical])
    rotation = correction.as_matrix() @ rotation
    camera_target = np.asarray(target_xyz, dtype=np.float64) + np.array([0.0, 0.0, float(height_m)])
    position = camera_target - rotation @ camera_in_grip[:3, 3]
    quaternion = Rotation.from_matrix(rotation).as_quat()
    if quaternion[int(np.argmax(np.abs(quaternion)))] < 0:
        quaternion = -quaternion
    return {
        "position_m": [round(float(value), 4) for value in position],
        "orientation_xyzw": [round(float(value), 6) for value in quaternion],
        "camera": "wrist",
        "camera_height_m": float(height_m),
        "claim_boundary": "copyable move_pose target computed from the current grip pose and wrist calibration; "
                          "not a reachability or collision check",
    }


def annotate(result: Any, *, grip_pose_xyz_xyzw: Any, wrist_camera_to_world: Any) -> Any:
    """Add ``overhead_view_pose`` beside every oblique footprint measurement in ``result``."""
    if not isinstance(result, Mapping):
        return result
    holders = [result, *(item for item in result.get("candidates") or () if isinstance(item, Mapping))]
    for holder in holders:
        localization = holder.get("localization")
        if not isinstance(localization, dict) or localization.get("footprint_view") != "oblique":
            continue
        centre = localization.get("footprint_center_xyz_m")
        if centre is not None:
            localization[POSE_KEY] = view_pose(centre, grip_pose_xyz_xyzw, wrist_camera_to_world)
    return result


__all__ = ["CAPABILITY", "KEYS", "OVERHEAD_MAX_TILT_DEG", "POSE_KEY", "VIEW_HEIGHT_M",
           "annotate", "empty", "footprint", "view_pose"]
