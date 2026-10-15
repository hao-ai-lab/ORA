"""Observed clearance below a commanded jaw footprint, from public depth only.

``observed_clearance_reference_v2`` annotates each ``move_pose`` target (and
the ``move_to`` / ``rotate_wrist`` / ``rotate_pitch`` targets that compile to
it) with one measurement taken before submission: how far the lowest point of
the jaw footprint at the commanded pose lies above the highest surface the
current depth images show beneath it or within ``RADIUS_M`` of it.

Inputs are only what the policy already receives: the observation's public
depth images with their per-observation calibration, the measured robot state,
the public embodiment geometry (finger collision bounds in the controlled-point
frame, opened along their published directions by the measured finger
positions) and, for a held object, the policy's own earlier localization of it
(``observed_clearance_runtime``). No scene geometry, layout or contact data.

Method. The footprint is the two finger boxes (plus the held-object box) posed
at the target; its lowest corner is the reported lowest point. Depth pixels
whose rays can reach the footprint column are back-projected to world points.
Points of the robot are excluded with the public geometry at the measured
state, as vertical prisms over the XY hull of each arm's measured finger boxes,
held box, controlled point and wrist camera: dilated ``ROBOT_CORE_MARGIN_M``
from the lowest finger or held corner upward, and ``ROBOT_MARGIN_M`` from the
lowest of the finger roots, controlled point and wrist camera upward, because
the descriptor bounds only the fingers and the housing, wrist and links rise
above them (offline over a recorded batch: with the jaw prism alone, 138 of 236 negative
readings on arrived moves were the moving arm's own housing, wrist camera or
links, all within 9 cm of link6). Of the remaining points within ``RADIUS_M``
horizontally of any footprint solid and no higher than the footprint's top, the
surface height is the ``MIN_SUPPORT_POINTS``-th highest. The result is
``unknown`` when no depth is valid, or fewer than ``MIN_SUPPORT_POINTS`` points
remain or less than ``MIN_OBSERVED_FRACTION`` of the area's 1 cm cells holds
one. Nothing here blocks or alters a move.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

#: Horizontal reach beside the footprint within which a surface counts.
RADIUS_M = 0.02
#: Horizontal dilation of the measured jaw prism (fingers and held object).
ROBOT_CORE_MARGIN_M = 0.01
#: Horizontal dilation of the measured body prism: the gripper housing, the
#: wrist and the links above the finger roots, which the descriptor does not bound.
ROBOT_MARGIN_M = 0.05
#: Each prism starts this far below its floor.
ROBOT_FLOOR_MARGIN_M = 0.01
#: A surface height is the one at least this many depth points reach.
MIN_SUPPORT_POINTS = 5
#: Points this close below the surface height are reported as its support.
SURFACE_BAND_M = 0.005
#: Below this observed share of the area's 1 cm cells the result is unknown.
MIN_OBSERVED_FRACTION = 0.25
CELL_M = 0.01
#: The released depth cap of the observed-extent review.
MAX_DEPTH_M = 2.5
#: How far beneath the footprint a surface is looked for.
BELOW_M = 1.5
#: A pixel window larger than this (a camera inside the footprint column, in
#: practice a wrist camera near the target) is read at a pixel stride.
MAX_WINDOW_PIXELS = 80_000
REASONS = ("depth_unavailable", "footprint_unobserved", "geometry_unavailable", "reference_failed")
CLAIM_BOUNDARY = (
    "From the current depth images and their calibration only; points inside the measured jaws, the held "
    "object and the space above them are excluded as robot, other robot links are not; occluded surfaces "
    "and object undersides are unknown; points up to the footprint's top within radius_m count; not a "
    "collision check or a contact prediction, and the move is submitted unchanged."
)


@dataclass(frozen=True)
class FingerBounds:
    """One finger's public collision bounds in the controlled-point frame at reference opening."""

    minimum: Sequence[float]
    maximum: Sequence[float]
    opening_direction: Sequence[float]


@dataclass(frozen=True)
class DepthView:
    camera: str
    depth_m: Any
    intrinsic: Any
    camera_to_world: Any


@dataclass(frozen=True)
class Footprint:
    """World corner sets, each labelled ``fingers`` or ``held_object``."""

    solids: tuple[tuple[str, Any], ...]

    @property
    def points(self) -> Any:
        return np.vstack([corners for _part, corners in self.solids])

    def lowest(self) -> tuple[list[float], str]:
        index, offset = int(np.argmin(self.points[:, 2])), 0
        for part, corners in self.solids:
            if index < offset + len(corners):
                return [float(value) for value in corners[index - offset]], part
            offset += len(corners)
        raise AssertionError("unreachable")


@dataclass(frozen=True)
class RobotVolume:
    """Vertical prisms over one measured arm's jaws: robot, or hidden by it.

    The jaw prism (``ROBOT_CORE_MARGIN_M``) rises from the lowest finger or
    held-object corner; the wider body prism (``margin_m``) rises from the
    lowest of the finger roots, the controlled point and the wrist camera.
    """

    hull_xy: Any
    floor_z: float
    body_floor_z: float
    margin_m: float


@dataclass
class _Candidates:
    points: list[Any] = field(default_factory=list)
    cameras: list[Any] = field(default_factory=list)
    valid_views: int = 0
    excluded: int = 0


#: ``box_corners`` indices of the low-x face: a finger's root face when the
#: approach axis is local +x (ARX X5, ALOHA). ``root_face(approach)`` gives it
#: for any axis-aligned approach.
ROOT_FACE = (0, 1, 2, 3)


def root_face(approach_local: Sequence[float]) -> tuple[int, ...]:
    """The four ``box_corners`` indices on the face farthest back along an axis-aligned approach."""
    axis = int(np.argmax(np.abs(np.asarray(approach_local, float))))
    back = 0 if float(approach_local[axis]) > 0 else 1
    shift = 2 - axis  # index = 4 * ix + 2 * iy + iz
    return tuple(index for index in range(8) if (index >> shift) & 1 == back)


def box_corners(minimum: Sequence[float], maximum: Sequence[float]) -> Any:
    """The eight corners, low-x face first (a finger's root face in its controlled-point frame)."""
    low, high = np.asarray(minimum, float), np.asarray(maximum, float)
    return np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])


def rotation(quaternion_xyzw: Sequence[float]) -> Any:
    x, y, z, w = (float(value) for value in quaternion_xyzw)
    norm = (x * x + y * y + z * z + w * w) ** 0.5
    if not norm:
        raise ValueError("orientation is not a rotation")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def footprint_at(
    pose_xyz_xyzw: Sequence[float], *, finger_qpos: Sequence[float], fingers: Sequence[FingerBounds],
    held_local_corners: Any = None,
) -> Footprint:
    """The finger boxes, opened by the measured finger positions, and the held box at one pose."""
    turn, origin = rotation(pose_xyz_xyzw[3:]), np.asarray(pose_xyz_xyzw[:3], float)
    solids = []
    for finger, opening in zip(fingers, finger_qpos, strict=True):
        shift = float(opening) * np.asarray(finger.opening_direction, float)
        local = box_corners(np.asarray(finger.minimum, float) + shift, np.asarray(finger.maximum, float) + shift)
        solids.append(("fingers", local @ turn.T + origin))
    if held_local_corners is not None:
        solids.append(("held_object", np.asarray(held_local_corners, float) @ turn.T + origin))
    return Footprint(solids=tuple(solids))


def robot_volume(
    solids: Sequence[tuple[str, Any]], *, controlled_point: Sequence[float], extra_points: Sequence[Any] = (),
    margin_m: float | None = None, root_indices: Sequence[int] = ROOT_FACE,
) -> RobotVolume:
    """``root_indices`` pick each finger box's root face (``box_corners`` order); the default is low-x."""
    corners = np.vstack([np.asarray(item, float) for _part, item in solids])
    roots = list(root_indices)
    body = np.vstack([np.asarray(controlled_point, float)[None], *[np.asarray(point, float)[None]
                                                                   for point in extra_points],
                      *[np.asarray(item, float)[roots] for part, item in solids if part == "fingers"]])
    return RobotVolume(hull_xy=_hull(np.vstack([corners[:, :2], body[:, :2]])), floor_z=float(corners[:, 2].min()),
                       body_floor_z=float(body[:, 2].min()),
                       margin_m=ROBOT_MARGIN_M if margin_m is None else float(margin_m))


def _hull(points: Any) -> Any:
    """Counter-clockwise convex hull (monotone chain) of 2-D points."""
    unique = sorted({(round(float(x), 9), round(float(y), 9)) for x, y in np.asarray(points, float)})
    if len(unique) <= 2:
        return np.array(unique, float)

    def cross(origin, a, b):
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

    def half(sequence):
        chain: list[tuple[float, float]] = []
        for point in sequence:
            while len(chain) >= 2 and cross(chain[-2], chain[-1], point) <= 0:
                chain.pop()
            chain.append(point)
        return chain

    lower, upper = half(unique), half(reversed(unique))
    return np.array(lower[:-1] + upper[:-1], float)


#: Directions of the polygonal disk a hull is dilated by: the dilated outline
#: lies within ``reach * (1 - cos(pi / 16))`` (1 mm at 5 cm) of the exact one.
_DILATION = np.stack([np.cos(np.arange(16) * np.pi / 8), np.sin(np.arange(16) * np.pi / 8)], axis=1)


def _within(points: Any, hull: Any, reach: float) -> Any:
    """Whether each point lies within ``reach`` horizontally of a convex polygon (polygonal dilation)."""
    outline = _hull((np.asarray(hull, float)[:, None, :] + reach * _DILATION[None]).reshape(-1, 2))
    edge = np.roll(outline, -1, axis=0) - outline
    normal = np.stack([edge[:, 1], -edge[:, 0]], axis=1)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-18)
    result = np.zeros(len(points), dtype=bool)
    boxed = np.flatnonzero(np.all((points >= outline.min(axis=0)) & (points <= outline.max(axis=0)), axis=1))
    if len(boxed):
        result[boxed] = ((points[boxed] @ normal.T) <= (outline * normal).sum(axis=1) + 1e-12).all(axis=1)
    return result


def _near(points_xy: Any, hulls: Sequence[Any], reach: float) -> Any:
    near = np.zeros(len(points_xy), dtype=bool)
    for hull in hulls:
        near |= _within(points_xy, hull, reach)
    return near


def _pixel_window(view: DepthView, box_low: Any, box_high: Any) -> tuple[slice, slice]:
    height, width = np.asarray(view.depth_m).shape
    to_camera = np.linalg.inv(np.asarray(view.camera_to_world, float))
    corners = box_corners(box_low, box_high) @ to_camera[:3, :3].T + to_camera[:3, 3]
    if (corners[:, 2] <= 0.01).any():
        return slice(0, height), slice(0, width)
    intrinsic = np.asarray(view.intrinsic, float)
    u = intrinsic[0, 0] * corners[:, 0] / corners[:, 2] + intrinsic[0, 2]
    v = intrinsic[1, 1] * corners[:, 1] / corners[:, 2] + intrinsic[1, 2]
    columns = slice(max(0, int(np.floor(u.min()))), min(width, int(np.ceil(u.max())) + 1))
    rows = slice(max(0, int(np.floor(v.min()))), min(height, int(np.ceil(v.max())) + 1))
    return rows, columns


def _world_points(view: DepthView, window: tuple[slice, slice]) -> Any:
    pixels = max(0, window[0].stop - window[0].start) * max(0, window[1].stop - window[1].start)
    if not pixels:
        return np.empty((0, 3))
    stride = max(1, int(np.ceil(np.sqrt(pixels / MAX_WINDOW_PIXELS))))
    depth = np.asarray(view.depth_m, dtype=np.float64)[window][::stride, ::stride]
    rows, columns = np.nonzero(np.isfinite(depth) & (depth > 1e-6) & (depth <= MAX_DEPTH_M))
    if not len(rows):
        return np.empty((0, 3))
    z = depth[rows, columns]
    rows, columns = rows * stride + window[0].start, columns * stride + window[1].start
    intrinsic, camera_to_world = np.asarray(view.intrinsic, float), np.asarray(view.camera_to_world, float)
    camera = np.stack([(columns - intrinsic[0, 2]) * z / intrinsic[0, 0],
                       (rows - intrinsic[1, 2]) * z / intrinsic[1, 1], z], axis=1)
    return camera @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]


def _in_robot(points: Any, robot: Sequence[RobotVolume]) -> Any:
    excluded = np.zeros(len(points), dtype=bool)
    for volume in robot:
        above = points[:, 2] >= volume.floor_z - ROBOT_FLOOR_MARGIN_M
        if not above.any():
            continue
        planar, body = points[above, :2], points[above, 2] >= volume.body_floor_z - ROBOT_FLOOR_MARGIN_M
        inside = _within(planar, volume.hull_xy, ROBOT_CORE_MARGIN_M)
        if body.any():
            inside[body] |= _within(planar[body], volume.hull_xy, volume.margin_m)
        excluded[above] |= inside
    return excluded


def _collect(views: Sequence[DepthView], footprint: Footprint, robot: Sequence[RobotVolume],
             hulls: Sequence[Any], radius_m: float) -> _Candidates:
    corners = footprint.points
    low = np.array([*(corners[:, :2].min(axis=0) - radius_m), corners[:, 2].min() - BELOW_M])
    high = np.array([*(corners[:, :2].max(axis=0) + radius_m), corners[:, 2].max()])
    found = _Candidates()
    for view in views:
        depth = np.asarray(view.depth_m)
        if depth.ndim != 2 or not (np.isfinite(depth) & (depth > 1e-6)).any():
            continue
        found.valid_views += 1
        points = _world_points(view, _pixel_window(view, low, high))
        if not len(points):
            continue
        points = points[np.all((points >= low - 1e-9) & (points <= high + 1e-9), axis=1)]
        if len(points):
            points = points[_near(points[:, :2], hulls, radius_m)]
        robot_mask = _in_robot(points, robot) if robot and len(points) else np.zeros(len(points), bool)
        points = points[~robot_mask]
        found.excluded += int(robot_mask.sum())
        found.points.append(points)
        found.cameras.append(np.full(len(points), view.camera, dtype=object))
    return found


def _observed_fraction(points: Any, hulls: Sequence[Any], radius_m: float, low_xy: Any, high_xy: Any) -> float:
    counts = np.maximum(np.ceil((high_xy - low_xy) / CELL_M).astype(int), 1)
    grid = np.stack(np.meshgrid(*(low_xy[axis] + (np.arange(counts[axis]) + 0.5) * CELL_M for axis in (0, 1)),
                                indexing="ij"), axis=-1).reshape(-1, 2)
    region = _near(grid, hulls, radius_m)
    if not region.any() or not len(points):
        return 0.0
    cells = np.clip(((points[:, :2] - low_xy) // CELL_M).astype(int), 0, counts - 1)
    observed = np.zeros(int(counts.prod()), dtype=bool)
    observed[cells[:, 0] * counts[1] + cells[:, 1]] = True
    return float((observed & region).sum() / region.sum())


def measure(
    views: Sequence[DepthView], *, observation_id: str, footprint: Footprint, robot: Sequence[RobotVolume] = (),
    radius_m: float | None = None,
) -> dict[str, Any]:
    """The observed clearance of one footprint (see the module docstring)."""
    radius_m = RADIUS_M if radius_m is None else float(radius_m)
    lowest, part = footprint.lowest()
    hulls = [_hull(corners[:, :2]) for _part, corners in footprint.solids]
    found = _collect(views, footprint, robot, hulls, radius_m)
    points = np.vstack(found.points) if found.points else np.empty((0, 3))
    cameras = np.concatenate(found.cameras) if found.cameras else np.empty(0, dtype=object)
    corners = footprint.points
    fraction = _observed_fraction(points, hulls, radius_m, corners[:, :2].min(axis=0) - radius_m,
                                  corners[:, :2].max(axis=0) + radius_m)
    base = {
        "footprint_lowest_m": [round(value, 4) for value in lowest], "footprint_lowest_part": part,
        "radius_m": radius_m, "footprint_observed_fraction": round(fraction, 2),
        "robot_points_excluded": found.excluded,
    }
    if not found.valid_views:
        return {"status": "unknown", "reason": "depth_unavailable", **base, "claim_boundary": CLAIM_BOUNDARY}
    if len(points) < MIN_SUPPORT_POINTS or fraction < MIN_OBSERVED_FRACTION:
        return {"status": "unknown", "reason": "footprint_unobserved", **base, "claim_boundary": CLAIM_BOUNDARY}
    order = np.argsort(-points[:, 2], kind="stable")
    chosen = int(order[MIN_SUPPORT_POINTS - 1])
    height, camera = float(points[chosen, 2]), str(cameras[chosen])
    support = int(((cameras == camera) & (points[:, 2] >= height - SURFACE_BAND_M)).sum())
    return {
        "status": "measured", "clearance_m": round(lowest[2] - height, 4), **base,
        "highest_surface": {"position_m": [round(float(value), 4) for value in points[chosen]], "camera": camera,
                            "observation_id": observation_id, "depth_points": support},
        "claim_boundary": CLAIM_BOUNDARY,
    }


def view_from_files(camera: str, depth_path: Any, calibration: Mapping[str, Any]) -> DepthView:
    """A depth view from a public observation's depth array and calibration record."""
    intrinsic = np.asarray(calibration["intrinsic_cv"], dtype=np.float64)
    extrinsic = np.asarray(calibration["extrinsic_cv"], dtype=np.float64)
    if extrinsic.shape == (3, 4):
        extrinsic = np.vstack([extrinsic, [0.0, 0.0, 0.0, 1.0]])
    if intrinsic.shape != (3, 3) or extrinsic.shape != (4, 4):
        raise ValueError(f"{camera} calibration shape mismatch")
    return DepthView(camera=camera, depth_m=np.load(depth_path, allow_pickle=False), intrinsic=intrinsic,
                     camera_to_world=np.linalg.inv(extrinsic))


__all__ = [
    "BELOW_M", "CELL_M", "CLAIM_BOUNDARY", "DepthView", "FingerBounds", "Footprint", "MAX_DEPTH_M",
    "MAX_WINDOW_PIXELS", "MIN_OBSERVED_FRACTION", "MIN_SUPPORT_POINTS", "RADIUS_M", "REASONS", "ROBOT_CORE_MARGIN_M",
    "ROBOT_FLOOR_MARGIN_M", "ROBOT_MARGIN_M", "ROOT_FACE", "RobotVolume", "SURFACE_BAND_M", "box_corners",
    "footprint_at", "measure", "robot_volume", "root_face", "rotation", "view_from_files",
]
