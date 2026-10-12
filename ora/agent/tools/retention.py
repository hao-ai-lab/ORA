"""Source-bound, post-close RGB-D motion evidence; never an attachment oracle.

The post-close interval and SE(3) transform follow the reviewed EXP08C v2
implementation. V3 compares visible surfaces against both rigid and stationary
hypotheses. A visible centroid is reported as a diagnostic, not a material
point whose displacement alone can authorize a held-object motion.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping

import numpy as np

from ora.contracts.pose import quaternion_matrix
from .robot import ToolError


SCHEMA = "ora.grasp-retention-verification.v3"
FULL_REFERENCE_CAPABILITY = "public_rgbd_grasp_retention_verification_v4"
FULL_REFERENCE_SCHEMA = "ora.grasp-retention-verification.v4"
ADVISORY_CAPABILITY = "public_rgbd_retention_advisory_v1"
ADVISORY_SCHEMA = "ora.grasp-retention-advisory.v1"
MINIMUM_MOTION_M = 0.015
NOISE_FLOOR_M = 0.004
MAXIMUM_POINTS = 512
MAXIMUM_REFERENCE_POINTS = 65536
MINIMUM_POINTS = 16
MINIMUM_OVERLAP = 0.9
MAXIMUM_VIEW_CHANGE_RAD = math.radians(20)


def enabled(surface: Any) -> bool:
    return surface.supports(FULL_REFERENCE_CAPABILITY)


def _vector(value: Any, size: int) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ToolError("public geometry is malformed", code="target_geometry_unavailable") from exc
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ToolError("public geometry is unavailable", code="target_geometry_unavailable")
    return result


def _effector(state: Mapping[str, Any], arm: str) -> Mapping[str, Any]:
    value = state.get("proprioception", {}).get("end_effectors", {}).get(arm)
    if not isinstance(value, Mapping):
        raise ToolError("public TCP state is unavailable", code="target_geometry_unavailable")
    return value


def _pose(state: Mapping[str, Any], arm: str) -> tuple[np.ndarray, np.ndarray]:
    value = _vector(_effector(state, arm).get("pose_xyz_xyzw"), 7)
    norm = float(np.linalg.norm(value[3:]))
    if not 0.99 <= norm <= 1.01:
        raise ToolError("public TCP quaternion is invalid", code="target_geometry_unavailable")
    return value[:3], np.asarray(quaternion_matrix((value[3:] / norm).tolist()))


def _command(
    state: Mapping[str, Any], arm: str, previous: Mapping[str, Any],
) -> tuple[str | None, str | None]:
    """Read consumed public commands, including LIBERO without target telemetry."""
    action = state.get("previous_public_action")
    receipt = state.get("previous_execution_receipt")
    if not isinstance(action, Mapping):
        return None, "public_action_history_unavailable"
    observation = previous.get("observation", {})
    if (not previous.get("state_sha256")
            or state.get("previous_state_sha256") != previous["state_sha256"]
            or action.get("observation_id") != observation.get("id")
            or action.get("bound_observation_sha256") != observation.get("manifest", {}).get("sha256")):
        return None, "public_action_source_history_unjoined"
    if (not isinstance(receipt, Mapping) or receipt.get("consumed") is not True
            or receipt.get("action_id") != action.get("action_id")):
        return None, "public_action_receipt_history_unjoined"
    if action.get("action") != "set_gripper":
        return None, None
    targets = action.get("arguments", {}).get("targets")
    if not isinstance(targets, Mapping):
        return None, "public_gripper_command_unavailable"
    value = targets.get(arm)
    if value is not None and value not in {"open", "close"}:
        return None, "public_gripper_command_unavailable"
    return value, None


def _closed_interval(
    before: Mapping[str, Any], current: Mapping[str, Any], arm: str,
    load_step: Callable[[int], Mapping[str, Any]],
) -> tuple[str | None, str | None]:
    start, end = before.get("step"), current.get("step")
    if type(start) is not int or type(end) is not int or start > end:
        return "baseline_must_precede_current_observation", None
    explicit_before = _effector(before, arm).get("gripper_command_target")
    explicit_current = _effector(current, arm).get("gripper_command_target")
    if explicit_before is not None and explicit_before != "close":
        return "baseline_must_be_post_close", None
    if explicit_current is not None and explicit_current != "close":
        return "current_gripper_is_not_closed", None

    # Find the close that actually preceded the baseline, without crossing a
    # reset or trusting a measured aperture as a command/attachment indicator.
    state, close_action_id = before, None
    for step in range(start, -1, -1):
        if step == 0 or state.get("step") != step or state.get("cause") != "after_action":
            return "baseline_must_be_post_close", None
        previous = load_step(step - 1)
        directive, error = _command(state, arm, previous)
        if error:
            return error, None
        if directive == "open":
            return "baseline_must_be_post_close", None
        if directive == "close":
            close_action_id = state["previous_public_action"].get("action_id")
            if not isinstance(close_action_id, str) or not close_action_id:
                return "public_close_action_unavailable", None
            break
        state = previous
    if close_action_id is None:
        return "baseline_must_be_post_close", None

    previous = before
    for step in range(start + 1, end + 1):
        state = current if step == end else load_step(step)
        if (state.get("step") != step or not previous.get("state_sha256")
                or state.get("previous_state_sha256") != previous["state_sha256"]):
            return "public_state_interval_is_not_contiguous", close_action_id
        if state.get("cause") != "after_action":
            return "baseline_crosses_episode_reset", close_action_id
        directive, error = _command(state, arm, previous)
        if error:
            return error, close_action_id
        if directive == "open":
            return "gripper_opened_since_baseline", close_action_id
        # A redundant close on an already-closed gripper leaves the commanded
        # state unchanged and cannot release the target, so it does not end the
        # interval. The backward scan above and this loop's exit condition keep
        # the invariant that the gripper was commanded closed at `previous`;
        # an `open` is the only directive that can break it. Re-closing may
        # still shift the target slightly, which the surface comparison
        # measures as residual rather than assumes away.
        explicit = _effector(state, arm).get("gripper_command_target")
        if explicit is not None and explicit != "close":
            return "gripper_opened_or_became_unknown_since_baseline", close_action_id
        previous = state
    if start == end:
        return "baseline_must_precede_current_observation", close_action_id
    return None, close_action_id


def _points(geometry: Mapping[str, Any], *, full_reference: bool = False) -> np.ndarray | None:
    raw = geometry.get("voxel_points_world_m")
    if raw is None:
        return None
    try:
        points = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ToolError("visible point cloud is malformed", code="target_geometry_unavailable") from exc
    if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
        raise ToolError("visible point cloud is malformed", code="target_geometry_unavailable")
    if len(points) < MINIMUM_POINTS:
        return None
    if full_reference and len(points) > MAXIMUM_REFERENCE_POINTS:
        raise ToolError("visible reference point budget exceeded", code="target_geometry_unavailable")
    if not full_reference and len(points) > MAXIMUM_POINTS:
        points = points[np.linspace(0, len(points) - 1, MAXIMUM_POINTS, dtype=int)]
    return points


def _surface_distance(
    a: np.ndarray, b: np.ndarray, radius: float, *, full_reference: bool = False,
) -> tuple[float, float, float]:
    if full_reference:
        from scipy.spatial import cKDTree

        def queries(points: np.ndarray) -> np.ndarray:
            if len(points) <= MAXIMUM_POINTS:
                return points
            return points[np.linspace(0, len(points) - 1, MAXIMUM_POINTS, dtype=int)]

        # Bound query work without discarding the measured surface against
        # which a distance is computed. Independently sparse reference sets
        # introduce artificial holes that can exceed the sensor noise bound.
        forward = cKDTree(b).query(queries(a), workers=1)[0]
        backward = cKDTree(a).query(queries(b), workers=1)[0]
    else:
        squared = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
        forward = np.sqrt(np.min(squared, axis=1))
        backward = np.sqrt(np.min(squared, axis=0))
    error = max(float(np.percentile(forward, 90)), float(np.percentile(backward, 90)))
    return error, float(np.mean(forward <= radius)), float(np.mean(backward <= radius))


def _view_change(
    before: Mapping[str, Any], current: Mapping[str, Any],
    c0: np.ndarray, c1: np.ndarray, r0: np.ndarray, r1: np.ndarray,
) -> float:
    if "camera_origin_world_m" in before and "camera_origin_world_m" in current:
        a = r0.T @ (_vector(before["camera_origin_world_m"], 3) - c0)
        b = r1.T @ (_vector(current["camera_origin_world_m"], 3) - c1)
        scale = float(np.linalg.norm(a) * np.linalg.norm(b))
        if scale > 1e-12:
            return math.acos(float(np.clip(a @ b / scale, -1, 1)))
    # Missing camera geometry never makes a large wrist rotation comparable.
    return math.acos(float(np.clip((np.trace(r1 @ r0.T) - 1) / 2, -1, 1)))


def _changed_view_remedy(
    before_geometry: Mapping[str, Any], before_state: Mapping[str, Any],
) -> str:
    """Name the step that ends the suppression, not only its cause.

    A large view change still forbids a negative claim: different visible
    surfaces are not evidence that the target stopped following the gripper.
    The policy can act on that only if the result names the pair that would be
    comparable. Both identities are already source-bound and eligibility has
    forced the current pair onto the baseline's camera, so the remedy is to
    re-observe that camera from the baseline viewpoint and verify again. The
    measured angle and its threshold stay in `surface_comparison` and
    `thresholds`; this clause only says what to do with them.
    """
    return ("changed_view_of_visible_surfaces_prevents_negative_retention_claim;"
            f"remedy=re_observe_camera_{before_geometry.get('source_camera')}"
            f"_from_the_step_{before_state.get('step')}_baseline_viewpoint_then_verify_again")


def verify_retention(
    *, arm: str, before_ref: str, current_ref: str,
    before_geometry: Mapping[str, Any], current_geometry: Mapping[str, Any],
    before_state: Mapping[str, Any], current_state: Mapping[str, Any],
    load_step: Callable[[int], Mapping[str, Any]],
    full_reference: bool = False,
    allow_border_evidence: bool = False,
) -> dict[str, Any]:
    """Compare fixed rigid motion with stationary geometry, without ICP fitting.

    Surface correspondence is not semantic identity. In particular, symmetric
    objects, changed visible surfaces and motions below the noise separation
    remain unknown instead of granting a transport permit from a centroid.
    """
    p0, r0 = _pose(before_state, arm)
    p1, r1 = _pose(current_state, arm)
    c0 = _vector(before_geometry.get("centroid_world_m"), 3)
    c1 = _vector(current_geometry.get("centroid_world_m"), 3)
    local0, local1 = r0.T @ (c0 - p0), r1.T @ (c1 - p1)
    predicted = p1 + r1 @ local0
    expected_delta, observed_delta = predicted - c0, c1 - c0
    expected = float(np.linalg.norm(expected_delta))
    observed = float(np.linalg.norm(observed_delta))
    residual = float(np.linalg.norm(c1 - predicted))
    cosine = (float(np.clip(expected_delta @ observed_delta / (expected * observed), -1, 1))
              if expected > 1e-12 and observed > 1e-12 else None)
    interval_reason, close_id = _closed_interval(before_state, current_state, arm, load_step)
    reason = interval_reason
    if reason is None and before_geometry.get("source_camera") != current_geometry.get("source_camera"):
        reason = "visible_surfaces_from_different_cameras_are_not_comparable"
    points0 = _points(before_geometry, full_reference=full_reference)
    points1 = _points(current_geometry, full_reference=full_reference)
    if reason is None and (points0 is None or points1 is None):
        reason = "visible_surface_evidence_is_insufficient"
    for geometry in (before_geometry, current_geometry):
        quality = geometry.get("source_quality")
        valid_quality = isinstance(quality, Mapping) and all(
            type(quality.get(key)) in (int, float)
            and math.isfinite(quality[key]) and minimum <= quality[key] <= 1.0
            for key, minimum in (("depth_valid_fraction", 0.7),
                                 ("depth_inlier_fraction", 0.7),
                                 ("primary_component_fraction", 0.6))
        ) and (quality.get("touches_image_boundary") is False
               or allow_border_evidence and quality.get("touches_image_boundary") is True)
        if reason is None and not valid_quality:
            reason = "visible_mask_depth_or_boundary_comparability_is_insufficient"
    voxel = max(float(g.get("voxel_size_m", 0.003)) for g in (before_geometry, current_geometry))
    if not math.isfinite(voxel) or voxel <= 0:
        raise ToolError("visible geometry voxel size is invalid", code="target_geometry_unavailable")
    noise = max(NOISE_FLOOR_M, 1.5 * voxel)
    excitation = max(MINIMUM_MOTION_M, 3 * noise)
    view_change = _view_change(before_geometry, current_geometry, c0, c1, r0, r1)
    surface = {
        "method": ("fixed_se3_vs_stationary_full_reference_v2" if full_reference
                   else "fixed_se3_vs_stationary_visible_surface_v1"),
        "before_point_count": 0 if points0 is None else len(points0),
        "current_point_count": 0 if points1 is None else len(points1),
        "rigid_p90_error_m": None, "stationary_p90_error_m": None,
        "centered_shape_p90_error_m": None,
        "rigid_overlap_before": None, "rigid_overlap_current": None,
        "model_separation_m": None, "view_direction_change_rad": view_change,
        "centroid_is_material_point": False,
    }
    if full_reference:
        surface.update(before_query_count=min(surface["before_point_count"], MAXIMUM_POINTS),
                       current_query_count=min(surface["current_point_count"], MAXIMUM_POINTS),
                       maximum_reference_points=MAXIMUM_REFERENCE_POINTS)
    eligible = reason is None
    status = "unknown"
    if eligible:
        assert points0 is not None and points1 is not None
        transformed = (points0 - p0) @ r0 @ r1.T + p1
        rigid, overlap0, overlap1 = _surface_distance(
            transformed, points1, noise, full_reference=full_reference)
        stationary, _, _ = _surface_distance(
            points0, points1, noise, full_reference=full_reference)
        centered, _, _ = _surface_distance(
            transformed - np.median(transformed, axis=0),
            points1 - np.median(points1, axis=0), noise, full_reference=full_reference,
        )
        separation = stationary - rigid
        surface.update(rigid_p90_error_m=rigid, stationary_p90_error_m=stationary,
                       centered_shape_p90_error_m=centered, rigid_overlap_before=overlap0,
                       rigid_overlap_current=overlap1, model_separation_m=separation)
        if expected < excitation:
            reason = "predicted_target_motion_is_below_noise_separation"
        elif (rigid <= noise and min(overlap0, overlap1) >= MINIMUM_OVERLAP
              and separation >= noise):
            status, reason = "consistent", "visible_surfaces_match_rigid_motion_over_stationary_hypothesis"
        elif view_change > MAXIMUM_VIEW_CHANGE_RAD:
            reason = _changed_view_remedy(before_geometry, before_state)
        elif stationary <= noise and rigid - stationary >= noise:
            status, reason = "inconsistent", "visible_surfaces_support_stationary_not_rigid_motion"
        elif (centered <= noise and rigid > 2 * noise and residual > 2 * noise
              and residual > 0.75 * expected):
            status, reason = "inconsistent", "comparable_visible_shape_did_not_follow_rigid_motion"
        else:
            fit_failed = rigid > noise
            overlap_failed = min(overlap0, overlap1) < MINIMUM_OVERLAP
            if fit_failed and overlap_failed:
                reason = "rigid_surface_fit_and_overlap_fail_acceptance"
            elif fit_failed:
                reason = "rigid_surface_fit_error_exceeds_noise"
            elif overlap_failed:
                reason = "rigid_surface_overlap_below_minimum"
            else:
                reason = "visible_surface_motion_hypotheses_are_not_separated"
    return {
        "schema": FULL_REFERENCE_SCHEMA if full_reference else SCHEMA,
        "status": status, "reason": reason, "arm": arm,
        "before_target_ref": before_ref, "current_target_ref": current_ref,
        "before_observation_id": before_geometry["observation_id"],
        "current_observation_id": current_geometry["observation_id"],
        "source_cameras": {"before": before_geometry["source_camera"],
                           "current": current_geometry["source_camera"]},
        "comparison_eligible": eligible,
        "baseline": {"before_state_step": before_state.get("step"),
                     "current_state_step": current_state.get("step"),
                     "close_action_id": close_id,
                     "closed_interval_verified": interval_reason is None},
        "geometry": {
            "target_to_tcp_before_local_m": local0.tolist(),
            "target_to_tcp_current_local_m": local1.tolist(),
            "predicted_target_world_m": predicted.tolist(),
            "expected_target_delta_world_m": expected_delta.tolist(),
            "observed_target_delta_world_m": observed_delta.tolist(),
            "expected_target_motion_m": expected, "observed_target_motion_m": observed,
            "rigid_centroid_residual_m": residual, "centroid_motion_direction_cosine": cosine,
        },
        "surface_comparison": surface,
        "thresholds": {
            "minimum_expected_target_motion_m": excitation, "surface_noise_radius_m": noise,
            "minimum_model_separation_m": noise, "minimum_rigid_overlap_fraction": MINIMUM_OVERLAP,
            "maximum_negative_view_direction_change_rad": MAXIMUM_VIEW_CHANGE_RAD,
            "minimum_visible_points": MINIMUM_POINTS,
            "inconsistent_centroid_residual_to_motion_ratio": 0.75,
        },
        "claim_boundary": (
            "public_visible_surface_motion_consistency_only;semantic_identity_unverified;"
            "centroids_not_material_points;not_attachment_contact_support_or_official_success"
        ),
    }
