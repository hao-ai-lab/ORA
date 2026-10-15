"""Support-aware target height from one synthetic RGB-D view.

``center_xyz_m`` is the median of the target's visible depth points, so from an
oblique camera it sits on the top face and upper side of an object rather than
at its middle. ``target_support_height_v1`` adds a single-view estimate of the
support plane under the target (from depth pixels in a ring around its mask),
the visible top's height above it and the midpoint between them. These tests
ray-cast a box standing on a table so the true support, top and middle are
known exactly, and check the estimate, its refusals and its capability gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ora.agent.tools import grasp, target_support

TABLE_Z = 0.76
BOX_HALF = np.array([0.03, 0.03, 0.04])  # a 6 x 6 x 8 cm box standing on the table
WIDTH, HEIGHT = 320, 240


def _look_at(eye, target):
    """World-to-camera 4x4 in the OpenCV convention (x right, y down, z forward)."""
    forward = np.asarray(target, float) - np.asarray(eye, float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rotation
    extrinsic[:3, 3] = -rotation @ np.asarray(eye, float)
    return extrinsic


def _render(eye, box_centre_xy=(0.0, 0.0), *, support_z=TABLE_Z, fx=260.0):
    """Analytic depth of a box on an infinite horizontal support, and its mask."""
    intrinsic = np.array([[fx, 0.0, WIDTH / 2], [0.0, fx, HEIGHT / 2], [0.0, 0.0, 1.0]])
    extrinsic = _look_at(eye, [box_centre_xy[0], box_centre_xy[1], support_z + BOX_HALF[2]])
    camera_to_world = np.linalg.inv(extrinsic)
    columns, rows = np.meshgrid(np.arange(WIDTH), np.arange(HEIGHT))
    rays_camera = np.stack([(columns - intrinsic[0, 2]) / fx, (rows - intrinsic[1, 2]) / fx,
                            np.ones_like(columns, dtype=float)], axis=-1)
    rays_world = rays_camera @ camera_to_world[:3, :3].T
    origin = camera_to_world[:3, 3]
    centre = np.array([box_centre_xy[0], box_centre_xy[1], support_z + BOX_HALF[2]])
    low, high = centre - BOX_HALF, centre + BOX_HALF
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (low - origin) / rays_world
        t2 = (high - origin) / rays_world
    near = np.nanmax(np.minimum(t1, t2), axis=-1)
    far = np.nanmin(np.maximum(t1, t2), axis=-1)
    hits_box = (far >= near) & (near > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_plane = (support_z - origin[2]) / rays_world[..., 2]
    t_plane = np.where(t_plane > 0, t_plane, np.inf)
    t = np.where(hits_box, np.minimum(near, t_plane), t_plane)
    depth = t * rays_camera[..., 2]  # OpenCV depth is the camera-frame z
    depth = np.where(np.isfinite(depth), depth, 0.0)
    return depth, hits_box & (near <= t_plane), intrinsic, extrinsic


def _calibration(tmp_path: Path, intrinsic, extrinsic) -> Path:
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"intrinsic_cv": intrinsic.tolist(), "extrinsic_cv": extrinsic.tolist()}))
    return path


@pytest.mark.parametrize("eye", [(0.0, -0.45, 1.20), (0.25, -0.40, 1.05), (0.0, -0.05, 1.30)],
                         ids=["oblique-front", "oblique-side", "near-top-down"])
def test_the_support_mid_height_recovers_the_box_middle_that_the_visible_centre_misses(tmp_path, eye):
    depth, mask, intrinsic, extrinsic = _render(eye)
    result = grasp._mask_depth_localization(
        mask=mask, depth_m=depth, calibration_path=_calibration(tmp_path, intrinsic, extrinsic),
        extended_frame=True)
    assert result["status"] == "localized"
    top = TABLE_Z + 2 * BOX_HALF[2]
    middle = TABLE_Z + BOX_HALF[2]
    assert result["support_height_basis"] == target_support.BASIS
    assert result["support_point_count"] >= 12
    assert result["support_z_m"] == pytest.approx(TABLE_Z, abs=0.003)
    assert result["height_above_support_m"] == pytest.approx(2 * BOX_HALF[2], abs=0.005)
    assert result["mid_height_z_m"] == pytest.approx(middle, abs=0.004)
    # The visible-surface median is biased toward the top the camera sees.
    assert result["center_xyz_m"][2] > result["mid_height_z_m"] + 0.005
    assert result["aabb_max_xyz_m"][2] == pytest.approx(top, abs=0.003)


def test_a_raised_support_is_measured_where_the_target_stands(tmp_path):
    """The plane is local: a box on a 10 cm shelf reports the shelf, not the floor."""
    depth, mask, intrinsic, extrinsic = _render((0.0, -0.45, 1.25), support_z=0.86)
    result = grasp._mask_depth_localization(
        mask=mask, depth_m=depth, calibration_path=_calibration(tmp_path, intrinsic, extrinsic),
        extended_frame=True)
    assert result["support_z_m"] == pytest.approx(0.86, abs=0.003)
    assert result["mid_height_z_m"] == pytest.approx(0.86 + BOX_HALF[2], abs=0.004)


def test_no_support_evidence_is_reported_as_unavailable_rather_than_guessed(tmp_path):
    depth, mask, intrinsic, extrinsic = _render((0.0, -0.45, 1.20))
    depth = np.where(mask, depth, 0.0)  # nothing valid outside the target
    result = grasp._mask_depth_localization(
        mask=mask, depth_m=depth, calibration_path=_calibration(tmp_path, intrinsic, extrinsic),
        extended_frame=True)
    assert result["status"] == "localized"
    assert {key: result[key] for key in target_support.KEYS} == target_support.empty()


def test_the_legacy_producer_and_insufficient_depth_keep_their_shapes(tmp_path):
    depth, mask, intrinsic, extrinsic = _render((0.0, -0.45, 1.20))
    calibration = _calibration(tmp_path, intrinsic, extrinsic)
    legacy = grasp._mask_depth_localization(mask=mask, depth_m=depth, calibration_path=calibration)
    assert not set(target_support.KEYS) & set(legacy)
    empty = grasp._mask_depth_localization(mask=mask, depth_m=np.zeros_like(depth),
                                           calibration_path=calibration, extended_frame=True)
    assert empty["status"] == "insufficient_depth"
    assert {key: empty[key] for key in target_support.KEYS} == target_support.empty()


def test_delivery_follows_the_declared_capabilities():
    measured = {"principal_axes_world": [[1, 0, 0]], "center_xyz_m": [0, 0, 0.8],
                **dict.fromkeys(target_support.KEYS, 1.0)}

    def result():
        return {"localization": dict(measured), "candidates": [{"localization": dict(measured)}]}

    neither = grasp.project_target_frame(result(), declared=False)
    both = grasp.project_target_frame(result(), declared=True, support_declared=True)
    support_only = grasp.project_target_frame(result(), declared=False, support_declared=True)
    for holder in (neither["localization"], neither["candidates"][0]["localization"]):
        assert "principal_axes_world" not in holder and not set(target_support.KEYS) & set(holder)
    assert set(target_support.KEYS) <= set(both["localization"]) and "principal_axes_world" in both["localization"]
    assert set(target_support.KEYS) <= set(support_only["localization"])
    assert "principal_axes_world" not in support_only["localization"]
    # A surface written before the capability existed keeps its released shape.
    frame_only = grasp.project_target_frame(result(), declared=True)
    assert "principal_axes_world" in frame_only["localization"]
    assert not set(target_support.KEYS) & set(frame_only["localization"])


LIVE = ("libero_placement_direct_overhead_v1", "robotwin_toolchain_b2_closed_noservo_v2",
        "robodojo_placement_closed_skills_v2")


@pytest.mark.parametrize("profile", LIVE)
def test_the_live_surfaces_declare_publish_and_explain_the_estimate(profile, tmp_path):
    from ora.agent.composition import resolve_tool_surface
    from ora.agent.composition.tool_surface import validate_schema_instance

    surface = resolve_tool_surface(profile)
    assert surface.supports(target_support.CAPABILITY)
    assert surface.supports(grasp.TARGET_PRINCIPAL_FRAME_CAPABILITY)
    depth, mask, intrinsic, extrinsic = _render((0.0, -0.45, 1.20))
    localization = grasp._mask_depth_localization(
        mask=mask, depth_m=depth, calibration_path=_calibration(tmp_path, intrinsic, extrinsic),
        extended_frame=True)
    projected = grasp.project_target_frame(
        {"localization": localization},
        **target_support.projection(surface, grasp.TARGET_PRINCIPAL_FRAME_CAPABILITY))["localization"]
    assert set(target_support.KEYS) <= set(projected)
    for tool in ("vision_segment", "vision_ground"):
        schema = surface.result_schemas[tool]
        definitions = schema["$defs"]["localization"]
        validate_schema_instance(projected, {**definitions, "$defs": schema["$defs"]})
        help_text = next(d["description"] for d in surface.definitions if d["name"] == tool)
        assert "mid_height_z_m" in help_text and "visible-surface median" in help_text
        assert "single-view" in help_text

