"""Footprint centre, view classification and the overhead wrist pose (overhead_footprint_localization_v1)."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from ora.agent.tools import grasp, overhead_view


def _mug_top_outline(centre_xy=(0.10, -0.05), radius=0.04, handle=0.03, z=0.95, step=0.001):
    """Points a camera straight above a mug sees: its rim disc and a handle on +x; body origin at the box centre."""
    xs, ys = np.meshgrid(np.arange(-radius - handle, radius + step, step), np.arange(-radius, radius + step, step))
    disc = xs ** 2 + ys ** 2 <= radius ** 2
    handle_band = (xs >= radius - 0.002) & (xs <= radius + handle) & (np.abs(ys) <= 0.006)
    keep = disc | handle_band
    points = np.stack([xs[keep], ys[keep], np.full(keep.sum(), z)], axis=1)
    points[:, 0] += centre_xy[0]
    points[:, 1] += centre_xy[1]
    return points


def test_overhead_footprint_centre_is_the_outline_box_centre_and_classified_overhead():
    points = _mug_top_outline()
    box_centre = (points[:, :2].min(0) + points[:, :2].max(0)) / 2
    result = overhead_view.footprint(points, camera_position=[box_centre[0], box_centre[1], 1.25])
    assert result["footprint_view"] == "overhead" and result["view_tilt_deg"] < 1.0
    assert np.linalg.norm(np.asarray(result["footprint_center_xyz_m"][:2]) - box_centre) < 0.003
    # The visible median is pulled off-centre by the uneven point mass the handle adds.
    assert result["footprint_extent_m"][0] > result["footprint_extent_m"][1]


def test_a_camera_far_to_the_side_is_oblique():
    points = _mug_top_outline()
    result = overhead_view.footprint(points, camera_position=[0.9, -0.05, 1.3])
    assert result["footprint_view"] == "oblique" and result["view_tilt_deg"] > overhead_view.OVERHEAD_MAX_TILT_DEG


def test_too_few_points_give_the_empty_shape():
    assert overhead_view.footprint(np.zeros((3, 3)), camera_position=[0, 0, 1]) == overhead_view.empty()


def _transform(position, quaternion):
    out = np.eye(4)
    out[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    out[:3, 3] = position
    return out


def test_the_overhead_pose_puts_the_wrist_camera_above_the_target_looking_down():
    # A tilted current grip pose and a wrist camera mounted as on the LIBERO Panda (grip frame offset, +Z optical axis).
    grip_now = _transform([0.02, 0.05, 1.00], Rotation.from_euler("xyz", [170, 12, 35], degrees=True).as_quat())
    camera_in_grip = np.eye(4)
    camera_in_grip[:3, 3] = [0.0, 0.05, -0.097]
    camera_now = grip_now @ camera_in_grip
    pose = [*grip_now[:3, 3], *Rotation.from_matrix(grip_now[:3, :3]).as_quat()]
    target = [0.11, -0.04, 0.93]
    view = overhead_view.view_pose(target, pose, camera_now)
    grip_view = _transform(view["position_m"], view["orientation_xyzw"])
    camera_view = grip_view @ camera_in_grip
    assert np.allclose(camera_view[:3, 3], np.asarray(target) + [0, 0, overhead_view.VIEW_HEIGHT_M], atol=2e-3)
    assert np.allclose(camera_view[:3, 2], [0, 0, -1], atol=1e-3)


def test_annotate_adds_a_pose_only_to_oblique_measurements():
    camera = np.eye(4)
    camera[:3, 3] = [0.0, 0.05, -0.097]
    pose = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    result = {"candidates": [
        {"localization": {"footprint_view": "oblique", "footprint_center_xyz_m": [0.1, 0.0, 0.9]}},
        {"localization": {"footprint_view": "overhead", "footprint_center_xyz_m": [0.1, 0.0, 0.9]}},
    ]}
    overhead_view.annotate(result, grip_pose_xyz_xyzw=pose, wrist_camera_to_world=_transform(pose[:3], pose[3:]) @ camera)
    assert overhead_view.POSE_KEY in result["candidates"][0]["localization"]
    assert overhead_view.POSE_KEY not in result["candidates"][1]["localization"]


def test_undeclared_surfaces_receive_none_of_the_footprint_fields():
    localization = {key: 1 for key in overhead_view.KEYS} | {overhead_view.POSE_KEY: {}, "center_xyz_m": [0, 0, 0]}
    kept = grasp.project_target_frame({"localization": dict(localization)}, declared=True, support_declared=True,
                                      footprint_declared=True)
    dropped = grasp.project_target_frame({"localization": dict(localization)}, declared=True, support_declared=True)
    assert set(overhead_view.KEYS) <= set(kept["localization"])
    assert not (set(overhead_view.KEYS) | {overhead_view.POSE_KEY}) & set(dropped["localization"])
