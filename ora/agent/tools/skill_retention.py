"""Bounded lift evidence from measured TCP, jaws and public RGB-D surfaces.

The reference is the Agent's selected target mask before grasp execution.
We require that surface to remain visible in the post-close baseline, then
compare its stationary and measured-TCP rigid-motion hypotheses against the
post-lift depth cloud. This is a scene-correspondence diagnostic, not a fresh
segmentation, semantic re-identification, contact sensor or attachment oracle.
Occlusion and indistinguishable hypotheses produce unknown.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ora.contracts.pose import quaternion_matrix

from . import geometry, localization
from .robot import ToolError

CAPABILITY = "prepared_grasp_execution_v2"
LIFT_M = 0.03
MINIMUM_POINTS = 16
MAXIMUM_POINTS = 512
NOISE_M = 0.006
MINIMUM_MOTION_M = 0.018
MINIMUM_SUPPORT = 0.9
MINIMUM_SUPPORT_GAP = 0.25


def reference(host: Any, **kwargs: Any) -> dict[str, Any] | None:
    """Freeze only the selected mask's public visible surface, with a work cap."""
    try:
        source = host._current_target_geometry(**kwargs)
        points = np.asarray(source["voxel_points_world_m"], dtype=float)
        if (points.ndim != 2 or points.shape[1:] != (3,)
                or len(points) < MINIMUM_POINTS or not np.isfinite(points).all()):
            return None
        if len(points) > MAXIMUM_POINTS:
            points = points[np.linspace(0, len(points) - 1, MAXIMUM_POINTS, dtype=int)]
        return {"camera": source["source_camera"], "points": points.tolist()}
    except (ToolError, KeyError, ValueError, OSError):
        return None


def capture_baseline(observation: Mapping[str, Any], root: Path) -> dict[str, Any]:
    manifest = root / "observation.json"
    return {
        "role": "post_close_before_lift",
        "observation": dict(observation),
        "observation_root": str(root.resolve()),
        "observation_id": observation["observation_id"],
        "observation_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }


def _effector(observation: Mapping[str, Any], arm: str) -> Mapping[str, Any]:
    return observation.get("proprioception", {}).get("end_effectors", {}).get(arm, {})


def _pose(observation: Mapping[str, Any], arm: str) -> tuple[np.ndarray, np.ndarray]:
    pose = np.asarray(_effector(observation, arm).get("pose_xyz_xyzw"), dtype=float)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("measured_tcp_unavailable")
    norm = np.linalg.norm(pose[3:])
    if not 0.99 <= norm <= 1.01:
        raise ValueError("measured_tcp_orientation_unavailable")
    return pose[:3], np.asarray(quaternion_matrix((pose[3:] / norm).tolist()))


def compare_surfaces(
    *, points: np.ndarray, before_scene: np.ndarray, after_scene: np.ndarray,
    before_position: np.ndarray, before_rotation: np.ndarray,
    after_position: np.ndarray, after_rotation: np.ndarray,
) -> dict[str, Any]:
    """Report separated public scene hypotheses, with no holding verdict."""
    from scipy.spatial import cKDTree

    result: dict[str, Any] = {
        "status": "unknown", "reason": "visible_surface_evidence_unavailable",
        "reference_points": int(len(points)),
        "baseline_support_fraction": None,
        "rigid_support_fraction": None, "stationary_support_fraction": None,
        "expected_surface_motion_m": None,
        "scope": "selected_pregrasp_surface_vs_public_scene;semantic_identity_unverified",
    }
    if any(value.ndim != 2 or value.shape[1:] != (3,)
           or len(value) < MINIMUM_POINTS or not np.isfinite(value).all()
           for value in (points, before_scene, after_scene)):
        return result
    baseline = cKDTree(before_scene).query(points, workers=1)[0]
    baseline_support = float(np.mean(baseline <= NOISE_M))
    result["baseline_support_fraction"] = baseline_support
    if baseline_support < MINIMUM_SUPPORT:
        result["reason"] = "selected_surface_not_confirmed_visible_after_close"
        return result
    predicted = (points - before_position) @ before_rotation @ after_rotation.T + after_position
    excitation = float(np.median(np.linalg.norm(predicted - points, axis=1)))
    result["expected_surface_motion_m"] = excitation
    if excitation < MINIMUM_MOTION_M:
        result["reason"] = "measured_motion_below_hypothesis_separation"
        return result
    current = cKDTree(after_scene)
    rigid = float(np.mean(current.query(predicted, workers=1)[0] <= NOISE_M))
    stationary = float(np.mean(current.query(points, workers=1)[0] <= NOISE_M))
    result.update(rigid_support_fraction=rigid, stationary_support_fraction=stationary)
    if rigid >= MINIMUM_SUPPORT and rigid - stationary >= MINIMUM_SUPPORT_GAP:
        result.update(status="rigid_motion_supported", reason="rigid_surface_hypothesis_has_more_depth_support")
    elif stationary >= MINIMUM_SUPPORT and stationary - rigid >= MINIMUM_SUPPORT_GAP:
        result.update(status="stationary_surface_supported", reason="stationary_surface_hypothesis_has_more_depth_support")
    else:
        result["reason"] = "occluded_or_ambiguous_scene_correspondence"
    return result


def measure(
    host: Any, *, commitment: Mapping[str, Any],
    observation: Mapping[str, Any], observation_root: Path,
) -> dict[str, Any]:
    baseline = commitment.get("lift_baseline")
    arm = commitment["arm"]
    result: dict[str, Any] = {
        "requested_world_lift_m": LIFT_M,
        "measured_tcp_translation_m": None, "measured_tcp_lift_m": None,
        "jaw_before_lift_m": None,
        "jaw_after_lift_m": localization.measured_jaw_separation_m(_effector(observation, arm)),
        "surface_motion": {"status": "unknown", "reason": "post_close_baseline_unavailable"},
        "claim_boundary": "short_lift_evidence_only;not_attachment_or_stable_transport_or_task_success",
    }
    if not isinstance(baseline, Mapping):
        return result
    before = baseline["observation"]
    result["jaw_before_lift_m"] = localization.measured_jaw_separation_m(_effector(before, arm))
    try:
        p0, r0 = _pose(before, arm)
        p1, r1 = _pose(observation, arm)
        result.update(measured_tcp_translation_m=(p1 - p0).tolist(), measured_tcp_lift_m=float(p1[2] - p0[2]))
        selected = commitment.get("retention_reference")
        if not isinstance(selected, Mapping):
            result["surface_motion"]["reason"] = "selected_public_target_surface_unavailable"
            return result
        scenes = [geometry.fuse_rgbd_pointcloud_sources(
            observation_root=root, observation=frame,
            workspace_bounds_m=host.workspace_bounds_m,
            camera_names=(selected["camera"],), voxel_size_m=0.003,
        )["xyz_world_m"] for frame, root in (
            (before, Path(baseline["observation_root"])), (observation, observation_root),
        )]
        result["surface_motion"] = compare_surfaces(
            points=np.asarray(selected["points"]), before_scene=scenes[0], after_scene=scenes[1],
            before_position=p0, before_rotation=r0, after_position=p1, after_rotation=r1,
        )
    except (ToolError, KeyError, ValueError, OSError, np.linalg.LinAlgError):
        result["surface_motion"] = {"status": "unknown", "reason": "public_rgbd_or_tcp_evidence_unavailable"}
    return result
