"""Observed clearance below a commanded jaw footprint, from public depth only.

EXP11A g012 (E11G12-004): 31 of 80 motion stalls drove the fingers or the held
object into a basket rim or wall, and in 30 of them the commanded end pose
itself put the fingertips or the object below the rim top. Admission checks
only self-collision and nothing in the result says the target lies below a
surface the policy can itself observe there.

These tests ray-cast small synthetic scenes into a pinhole depth image with the
public calibration conventions (OpenCV camera, ``extrinsic_cv`` world->camera,
depth = distance to the image plane) and check the pure measurement in
``agent/tools/observed_clearance.py``: footprint at the target, robot points
excluded at the measured state, the highest observed surface beneath or within
the stated radius, and ``unknown`` when depth or the footprint area is missing.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from ora.agent.tools import observed_clearance as oc

#: The public ARX X5 finger bounds (descriptor-public-geometry-v4), link6 frame.
FINGERS = (
    oc.FingerBounds(minimum=(0.07157, 0.0004016, -0.03064), maximum=(0.15757, 0.0377516, 0.03036),
                    opening_direction=(0.0, 1.0, 0.0)),
    oc.FingerBounds(minimum=(0.07157, -0.0377556, -0.03064), maximum=(0.15757, -0.0004056, 0.03036),
                    opening_direction=(0.0, -1.0, 0.0)),
)
#: link6 approach (+x) pointing straight down, jaw-closing line along world y.
DOWN = (0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5))
TABLE_Z = 0.75
RIM_TOP = 0.842


def _box_hits(origin, directions, low, high):
    """Slab-method ray/AABB distances along each direction (inf when missed)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inverse = 1.0 / directions
        t1 = (np.asarray(low) - origin) * inverse
        t2 = (np.asarray(high) - origin) * inverse
    near = np.nanmax(np.minimum(t1, t2), axis=1)
    far = np.nanmin(np.maximum(t1, t2), axis=1)
    hit = (far >= near) & (far > 0)
    return np.where(hit, np.where(near > 0, near, far), np.inf)


def _camera(position, *, look=(0.0, 0.0, -1.0), up=(0.0, 1.0, 0.0), size=(160, 120), focal=110.0):
    forward = np.asarray(look, float) / np.linalg.norm(look)
    right = np.cross(forward, np.asarray(up, float))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    camera_to_world = np.eye(4)
    camera_to_world[:3, 0], camera_to_world[:3, 1], camera_to_world[:3, 2] = right, down, forward
    camera_to_world[:3, 3] = position
    width, height = size
    intrinsic = np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]])
    return camera_to_world, intrinsic


def _render(camera_to_world, intrinsic, size, boxes, *, plane_z=TABLE_Z):
    """Depth (to the image plane) of a table plane plus axis-aligned boxes."""
    width, height = size
    rows, columns = np.mgrid[0:height, 0:width]
    rays_camera = np.stack([(columns - intrinsic[0, 2]) / intrinsic[0, 0],
                            (rows - intrinsic[1, 2]) / intrinsic[1, 1], np.ones_like(rows, float)], axis=-1)
    rays = rays_camera.reshape(-1, 3) @ camera_to_world[:3, :3].T
    origin = camera_to_world[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        plane = (plane_z - origin[2]) / rays[:, 2]
    distance = np.where(plane > 0, plane, np.inf)
    for low, high in boxes:
        distance = np.minimum(distance, _box_hits(origin, rays, low, high))
    depth = np.where(np.isfinite(distance), distance, 0.0)  # z of rays_camera is 1: t is image-plane depth
    return depth.reshape(height, width).astype(np.float32)


def _view(name, position, boxes, **kwargs):
    size = kwargs.pop("size", (160, 120))
    camera_to_world, intrinsic = _camera(position, size=size, **kwargs)
    return oc.DepthView(camera=name, depth_m=_render(camera_to_world, intrinsic, size, boxes),
                        intrinsic=intrinsic, camera_to_world=camera_to_world)


def _footprint(position, *, opening=0.044, held=None):
    pose = [*position, *DOWN]
    return oc.footprint_at(pose, finger_qpos=(opening, opening), fingers=FINGERS, held_local_corners=held)


def _measure(views, footprint, robot=(), **kwargs):
    return oc.measure(views, observation_id="obs-7", footprint=footprint, robot=list(robot), **kwargs)


def test_fingers_hovering_over_an_open_table_report_positive_clearance():
    # link6 0.30 m above the table: fingertips 0.1576 m below link6.
    view = _view("head", (0.0, 0.0, 1.6), [])
    result = _measure([view], _footprint((0.0, 0.0, TABLE_Z + 0.30)))
    assert result["status"] == "measured"
    assert result["clearance_m"] == pytest.approx(0.30 - 0.15757, abs=0.003)
    assert result["footprint_lowest_part"] == "fingers"
    assert result["footprint_lowest_m"][2] == pytest.approx(TABLE_Z + 0.30 - 0.15757, abs=1e-4)
    surface = result["highest_surface"]
    assert surface["camera"] == "head" and surface["observation_id"] == "obs-7"
    assert surface["position_m"][2] == pytest.approx(TABLE_Z, abs=0.002)
    assert surface["depth_points"] >= oc.MIN_SUPPORT_POINTS
    assert result["radius_m"] == oc.RADIUS_M and result["footprint_observed_fraction"] > 0.9


def test_fingertips_commanded_below_an_observed_rim_report_the_rim():
    # A basket wall 5 mm thick whose top is 0.842, 14 mm beside the closed jaws,
    # seen at about 3 mm per pixel like the 640x480 head camera.
    wall = ((0.045, -0.20, TABLE_Z), (0.05, 0.20, RIM_TOP))
    view = _view("head", (0.0, -0.3, 1.6), [wall], look=(0.0, 0.25, -1.0), size=(400, 300), focal=300.0)
    footprint = _footprint((0.0, 0.0, 0.80 + 0.15757), opening=0.0)
    result = _measure([view], footprint)
    assert result["status"] == "measured"
    assert result["clearance_m"] == pytest.approx(0.80 - RIM_TOP, abs=0.004)
    top = result["highest_surface"]["position_m"]
    assert top[2] == pytest.approx(RIM_TOP, abs=0.004) and 0.044 <= top[0] <= 0.051


def test_a_rim_beyond_the_radius_is_not_beside_the_footprint():
    wall = ((0.12, -0.20, TABLE_Z), (0.125, 0.20, RIM_TOP))
    view = _view("head", (0.0, -0.3, 1.6), [wall], look=(0.0, 0.25, -1.0))
    result = _measure([view], _footprint((0.0, 0.0, 0.80 + 0.15757), opening=0.0))
    assert result["clearance_m"] == pytest.approx(0.80 - TABLE_Z, abs=0.004)


def test_a_held_object_hanging_below_the_fingers_sets_the_lowest_point():
    # An 8 cm object held between the pads hangs 5.2 cm below the fingertips.
    local = oc.box_corners((0.13, -0.02, -0.02), (0.13 + 0.08, 0.02, 0.02))
    footprint = _footprint((0.0, 0.0, TABLE_Z + 0.40), held=local)
    view = _view("head", (0.0, 0.0, 1.6), [])
    result = _measure([view], footprint)
    assert result["footprint_lowest_part"] == "held_object"
    assert result["clearance_m"] == pytest.approx(0.40 - 0.21, abs=0.003)


def test_points_on_the_measured_jaws_are_excluded_as_robot():
    # A rotation in place 5 cm over the table: the camera sees the jaws themselves.
    position = (0.0, 0.0, TABLE_Z + 0.05 + 0.15757)
    footprint = _footprint(position, opening=0.02)
    jaw_boxes = [(corners.min(axis=0), corners.max(axis=0)) for _part, corners in footprint.solids]
    view = _view("head", (0.0, -0.25, 1.6), jaw_boxes, look=(0.0, 0.2, -1.0))
    robot = oc.robot_volume(footprint.solids, controlled_point=position)
    excluded = _measure([view], footprint, robot=[robot])
    assert excluded["clearance_m"] == pytest.approx(0.05, abs=0.004)
    assert excluded["robot_points_excluded"] > 0
    included = _measure([view], footprint)
    assert included["clearance_m"] < 0


def test_the_space_above_the_measured_jaws_is_robot_or_hidden_by_it():
    # A lift: the wrist and arm rise above the current jaws and occupy the
    # target column (the descriptor bounds only the fingers).
    position = (0.0, 0.0, TABLE_Z + 0.16)
    current = _footprint(position, opening=0.0)
    wrist = ((-0.03, -0.04, position[2] - 0.08), (0.03, 0.04, position[2] + 0.20))
    view = _view("head", (0.0, -0.25, 1.6), [wrist], look=(0.0, 0.2, -1.0))
    robot = oc.robot_volume(current.solids, controlled_point=position)
    target = _footprint((0.0, 0.0, position[2] + 0.10), opening=0.0)
    result = _measure([view], target, robot=[robot])
    assert result["status"] == "measured"
    assert result["highest_surface"]["position_m"][2] == pytest.approx(TABLE_Z, abs=0.003)


def test_a_second_camera_names_itself_when_it_sees_the_highest_surface():
    block = ((-0.02, 0.06, TABLE_Z), (0.02, 0.10, 0.80))
    head = _view("head", (0.0, -0.6, 1.4), [], look=(0.0, 0.8, -1.0))
    wrist = _view("left_wrist", (0.0, 0.0, 1.3), [block])
    result = _measure([head, wrist], _footprint((0.0, 0.0, 0.79 + 0.15757), opening=0.044))
    assert result["highest_surface"]["camera"] == "left_wrist"
    assert result["clearance_m"] == pytest.approx(0.79 - 0.80, abs=0.004)


def test_missing_depth_is_unknown_with_its_reason():
    view = _view("head", (0.0, 0.0, 1.6), [])
    blank = oc.DepthView(camera="head", depth_m=np.zeros_like(view.depth_m), intrinsic=view.intrinsic,
                         camera_to_world=view.camera_to_world)
    result = _measure([blank], _footprint((0.0, 0.0, TABLE_Z + 0.3)))
    assert result["status"] == "unknown" and result["reason"] == "depth_unavailable"
    assert "clearance_m" not in result and result["footprint_lowest_m"][2] == pytest.approx(
        TABLE_Z + 0.3 - 0.15757, abs=1e-4)
    assert _measure([], _footprint((0.0, 0.0, 1.0)))["reason"] == "depth_unavailable"


def test_a_footprint_outside_every_view_is_unknown():
    view = _view("head", (0.0, 0.0, 1.6), [])
    result = _measure([view], _footprint((3.0, 3.0, TABLE_Z + 0.3)))
    assert result["status"] == "unknown" and result["reason"] == "footprint_unobserved"
    assert result["footprint_observed_fraction"] == 0.0


def test_the_result_states_measurements_and_its_claim_boundary_only():
    view = _view("head", (0.0, 0.0, 1.6), [])
    result = _measure([view], _footprint((0.0, 0.0, TABLE_Z + 0.3)))
    boundary = result["claim_boundary"].lower()
    for phrase in ("current depth", "occluded", "not a collision check"):
        assert phrase in boundary
    text = repr(result).lower()
    for advice in ("should", "try ", "raise the", "recommend", "safe"):
        assert advice not in text


def test_the_measurement_is_bounded_on_a_full_resolution_view():
    import time

    size = (640, 480)
    views = [_view(name, (0.3 * index - 0.3, -0.4, 1.3), [], look=(0.0, 0.5, -0.87), size=size, focal=288.0)
             for index, name in enumerate(("head", "left_wrist", "right_wrist"))]
    footprint = _footprint((0.0, 0.0, TABLE_Z + 0.2))
    started = time.perf_counter()
    for _ in range(5):
        _measure(views, footprint)
    assert (time.perf_counter() - started) / 5 < 0.25
