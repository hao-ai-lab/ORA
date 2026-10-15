"""Trusted-side reach footprint of a saved RoboDojo layout's task objects.

Host-side and read-only, like ``robodojo_layout_catalog``: it reads one pinned
layout file and the pinned asset metadata that sizes each object in scope,
imports no simulator and never reaches a policy. A grasp lands on an object's
extent, not on its centre (a recorded stack_bowls run on layout 51: every grasp candidate
pinched the far rim of a bowl whose centre was reachable), so the
scheduled-layout scene smoke aims its read-only top-down ``align_plan`` at the
centre and at ``FOOTPRINT_POINTS`` points around the object's horizontal
footprint, and a pool review measures the same points against the fixed arm
bases (``footprint_base_distances``): schedule selection and smoke judge one
footprint.

Scope: the free rigid bodies a task's policy moves, i.e. the layout's
``Rigid`` instances except ``cluttered`` distractors. Upstream registers static
fixtures and containers (the floor-standing dustbin, baskets, toolbox, camera
stand) as kinematic ``Geometry`` and keeps articulated, conveyor, garment and
fluid instances in their own sections; none of them is a reach object.

Footprint: the instance's metadata ``geometry.aligned_bbox`` (asset frame),
scaled by the layout ``scale`` and turned by ``default_ori`` (w, x, y, z, as
upstream's ``LayoutManager.to_transformation_matrix`` reads it). Of the box's
three axes the one nearest world z is its height; the other two span the
horizontal footprint, and the points lie on the ellipse inscribed in that
rectangle: a circle for a round object (a bowl's rim), both ends and both
sides for a lying bottle.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from ora.environment.benchmarks.robodojo import RoboDojoBenchmarkError

#: The layout section whose instances are free rigid bodies a policy moves.
REACH_SECTION = "Rigid"
#: Rigid instances of this ``type`` are scattered distractors no task needs moved.
DISTRACTOR_TYPE = "cluttered"
#: Footprint points per object, evenly spaced around its horizontal footprint.
FOOTPRINT_POINTS = 8
#: The fixed ARX X5 base mounts (x, y) of ``env_cfg/robot/dual_x5.yml`` in the
#: episode-local frame: a copy of the worker's
#: ``backends/robodojo/motion_checks.X5_BASE_MOUNTS_M`` that a test holds equal.
X5_BASE_MOUNTS_XY_M = {"left": (-0.3, -0.45), "right": (0.3, -0.45)}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RoboDojoBenchmarkError(message)


def object_metadata_path(category: str, category_idx: int) -> str:
    """Assets-relative metadata of one ``Rigid`` model, as upstream's layout manager resolves it."""

    _require(isinstance(category, str) and category not in {"", ".", ".."} and "/" not in category
             and type(category_idx) is int and category_idx >= 0,
             f"layout object {category!r}/{category_idx!r} names no asset model")
    return f"Object/RoboDojo/{REACH_SECTION}/{category}/{category_idx:05d}/metadata.json"


def _finite(values: Any, size: int) -> list[float]:
    _require(isinstance(values, Sequence) and len(values) == size
             and all(isinstance(value, (int, float)) and math.isfinite(value) for value in values),
             "layout object geometry is malformed")
    return [float(value) for value in values]


def _rotation(wxyz: Sequence[float]) -> list[list[float]]:
    w, x, y, z = wxyz
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    _require(norm > 0.0, "layout object orientation is not a quaternion")
    w, x, y, z = (value / norm for value in (w, x, y, z))
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def footprint(instance: Mapping[str, Any], metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Horizontal footprint points of one placed instance, from its asset bounding box."""

    vertices = ((metadata.get("geometry") or {}).get("aligned_bbox") or {}).get("vertices")
    _require(isinstance(vertices, list) and len(vertices) == 8, "asset metadata has no aligned bounding box")
    corners = [_finite(vertex, 3) for vertex in vertices]
    scale = _finite(instance.get("scale", [1.0, 1.0, 1.0]), 3)
    _require(all(value > 0.0 for value in scale), "layout object scale must be positive")
    low = [min(corner[axis] for corner in corners) * scale[axis] for axis in range(3)]
    high = [max(corner[axis] for corner in corners) * scale[axis] for axis in range(3)]
    offset = [(low[axis] + high[axis]) / 2 for axis in range(3)]
    half = [(high[axis] - low[axis]) / 2 for axis in range(3)]
    rotation = _rotation(_finite(instance.get("default_ori", [1.0, 0.0, 0.0, 0.0]), 4))
    position = _finite(instance.get("default_pos"), 3)
    columns = [[rotation[row][axis] for row in range(3)] for axis in range(3)]
    height = max(range(3), key=lambda axis: abs(columns[axis][2]))
    first, second = (axis for axis in range(3) if axis != height)
    centre = [position[row] + sum(rotation[row][axis] * offset[axis] for axis in range(3)) for row in range(2)]
    spans = [[columns[axis][row] * half[axis] for row in range(2)] for axis in (first, second)]
    points = []
    for index in range(FOOTPRINT_POINTS):
        angle = 2 * math.pi * index / FOOTPRINT_POINTS
        dx = math.cos(angle) * spans[0][0] + math.sin(angle) * spans[1][0]
        dy = math.cos(angle) * spans[0][1] + math.sin(angle) * spans[1][1]
        points.append({"point": f"edge{index}", "xy_m": [round(centre[0] + dx, 6), round(centre[1] + dy, 6)],
                       "bearing_deg": round(math.degrees(math.atan2(dy, dx))) % 360})
    return {"half_extents_m": [round(half[first], 6), round(half[second], 6)], "points": points}


def layout_reach_objects(asset_root: Path, layout_file: str) -> list[dict[str, Any]]:
    """Every reach object of a saved layout: spawn centre, sizing metadata and footprint points."""

    root = Path(asset_root)
    layout = json.loads((root / layout_file).read_text())
    objects = []
    for category, instances in sorted((layout.get(REACH_SECTION) or {}).items()):
        _require(isinstance(instances, list), f"layout section {REACH_SECTION} is malformed")
        for instance in instances:
            if instance.get("type") == DISTRACTOR_TYPE:
                continue
            relative = object_metadata_path(category, instance.get("category_idx"))
            path = root / relative
            _require(path.is_file(), f"asset metadata {relative} is missing")
            data = path.read_bytes()
            objects.append({
                "label": instance.get("label") or category, "category": category,
                "category_idx": instance["category_idx"],
                "position_m": _finite(instance.get("default_pos"), 3),
                "metadata": {"path": relative, "sha256": hashlib.sha256(data).hexdigest()},
                "footprint": footprint(instance, json.loads(data)),
            })
    labels = [item["label"] for item in objects]
    _require(len(set(labels)) == len(labels), f"{layout_file} repeats a layout object label")
    return objects


def footprint_base_distances(item: Mapping[str, Any]) -> dict[str, Any]:
    """Horizontal distance from the nearer arm base, to the centre and to the farthest footprint point.

    The footprint distance is per arm the farthest of the centre and every
    footprint point, minimised over the arms: one arm must cover the whole
    object, as the smoke requires. Compare it with a measured reach envelope;
    the scene smoke's IK, not this distance, decides.
    """

    centre = item["position_m"][:2]
    points = [centre, *(point["xy_m"] for point in item["footprint"]["points"])]
    farthest = {arm: max(math.dist(point, base) for point in points) for arm, base in X5_BASE_MOUNTS_XY_M.items()}
    arm = min(farthest, key=farthest.get)
    return {"centre_nearer_base_m": min(math.dist(centre, base) for base in X5_BASE_MOUNTS_XY_M.values()),
            "footprint_arm": arm, "footprint_nearer_base_m": farthest[arm]}


__all__ = [
    "DISTRACTOR_TYPE", "FOOTPRINT_POINTS", "REACH_SECTION", "X5_BASE_MOUNTS_XY_M", "footprint",
    "footprint_base_distances", "layout_reach_objects", "object_metadata_path",
]
