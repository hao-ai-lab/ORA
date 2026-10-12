"""CPU-only placement proposals and public, partial-geometry checks.

Inputs are immutable public RGB-D derivatives, never simulator object state.
No function mutates a robot, creates an executable motion plan, fills unseen
support, certifies semantic identity, or establishes contact/stability.  In
particular, a convex hull below is an *outer observed envelope for rejection*,
not a completed support surface.  Candidate footprints must instead match
actually observed support cells.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


DESTINATIONS = frozenset({"support_surface", "open_container", "front_cavity"})
PHASES = frozenset({"preplace", "lower", "exit"})
CELL_M = 0.01
PLANE_RESIDUAL_M = 0.006
REGION_MARGIN_M = 0.015
CLAIM_BOUNDARY = (
    "observed RGB-D geometry only; no semantic identity, hidden shape, contact, "
    "stability, complete containment, collision-free path, reachability, "
    "attachment or official-success claim; no world mutation"
)


def _array_points(value: Any, name: str, *, minimum: int = 1) -> np.ndarray:
    try:
        points = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite [x,y,z] points") from exc
    if (
        points.ndim != 2 or points.shape[1] != 3
        or not minimum <= len(points) <= 100_000
        or not np.isfinite(points).all()
    ):
        raise ValueError(f"{name} must contain {minimum}..100000 finite [x,y,z] points")
    return points


def _points(geometry: Mapping[str, Any], name: str) -> np.ndarray:
    if not isinstance(geometry, Mapping):
        raise ValueError(f"{name} must be a geometry mapping")
    return _array_points(geometry.get("voxel_points_world_m"), name)


def _pose(value: Any) -> np.ndarray:
    try:
        pose = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError("tcp_pose must be finite [x,y,z,qx,qy,qz,qw]") from exc
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("tcp_pose must be finite [x,y,z,qx,qy,qz,qw]")
    if abs(float(np.linalg.norm(pose[3:])) - 1.0) > 1e-3:
        raise ValueError("tcp_pose quaternion must be unit length")
    return pose


def _check(status: str, reason: str, metrics: Mapping[str, Any] | None = None,
           thresholds: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {"status": status, "reason": reason, "metrics": dict(metrics or {}),
            "thresholds": dict(thresholds or {})}


def _sample(points: np.ndarray, maximum: int) -> np.ndarray:
    if len(points) <= maximum:
        return points
    return points[np.linspace(0, len(points) - 1, maximum, dtype=int)]


def _basis(normal: np.ndarray) -> np.ndarray:
    reference = np.asarray([1.0, 0.0, 0.0])
    if abs(float(reference @ normal)) > 0.9:
        reference = np.asarray([0.0, 1.0, 0.0])
    first = reference - float(reference @ normal) * normal
    first /= np.linalg.norm(first)
    return np.stack([first, np.cross(normal, first)])


def _plane(points: np.ndarray, *, vertical: bool, refine: bool = False) -> dict[str, Any]:
    thresholds = {"minimum_points": 16, "maximum_residual_m": PLANE_RESIDUAL_M,
                  "minimum_inlier_fraction": 0.55, "maximum_tilt_deg": 15.0,
                  "minimum_two_axis_span_m": 0.025,
                  "ambiguous_secondary_plane_fraction": 0.75,
                  "separated_plane_distance_m": 0.025}
    if len(points) < 16:
        return _check("unknown", "insufficient_plane_points", thresholds=thresholds)
    sampled = _sample(points, 2048)
    candidates: list[tuple[int, float, np.ndarray, np.ndarray]] = []
    rng = np.random.default_rng(0)
    for _ in range(96):
        a, b, c = sampled[rng.choice(len(sampled), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = float(np.linalg.norm(normal))
        if length < 1e-10:
            continue
        normal /= length
        vertical_component = abs(float(normal[2]))
        incompatible = (
            vertical_component > math.sin(math.radians(15)) if vertical
            else vertical_component < math.cos(math.radians(15))
        )
        if incompatible:
            continue
        residual = np.abs((sampled - a) @ normal)
        inliers = residual <= PLANE_RESIDUAL_M
        candidates.append((int(inliers.sum()), -float(np.median(residual[inliers])),
                           a, normal))
    if not candidates:
        return _check("unknown", "no_compatible_plane", thresholds=thresholds)
    sampled_inlier_count, _, origin, normal = max(candidates, key=lambda row: row[:2])
    inliers = np.abs((points - origin) @ normal) <= PLANE_RESIDUAL_M
    selected = points[inliers]
    if len(selected) < 16 or float(inliers.mean()) < 0.55:
        return _check("unknown", "plane_coverage_insufficient", {
            "point_count": len(points), "inlier_count": len(selected),
            "inlier_fraction": float(inliers.mean()),
        }, thresholds)
    refinement_iterations = 0
    if refine:
        # The sampled RANSAC plane and its least-squares refit are different
        # models. Re-evaluate inliers against the latter, using the SAME 6 mm
        # limit and the original complete retained-point denominator. This is
        # bounded model refinement, not a changed fit criterion or filled plane.
        for _ in range(4):
            refit_origin = np.mean(selected, axis=0)
            _, refit_vectors = np.linalg.eigh(np.cov((selected - refit_origin).T))
            refined = np.abs((points - refit_origin) @ refit_vectors[:, 0]) <= PLANE_RESIDUAL_M
            refinement_iterations += 1
            if int(refined.sum()) < 16 or float(refined.mean()) < .55:
                return _check("unknown", "plane_coverage_insufficient_after_refit", {
                    "point_count": len(points), "inlier_count": int(refined.sum()),
                    "inlier_fraction": float(refined.mean()),
                    "refinement_iterations": refinement_iterations}, thresholds)
            unchanged = np.array_equal(refined, inliers)
            inliers, selected = refined, points[refined]
            if unchanged:
                break
    origin = np.mean(selected, axis=0)
    values, vectors = np.linalg.eigh(np.cov((selected - origin).T))
    normal = vectors[:, 0]
    if normal[int(np.argmax(np.abs(normal)))] < 0:
        normal = -normal
    if not vertical and normal[2] < 0:
        normal = -normal
    basis = _basis(normal)
    uv = (selected - origin) @ basis.T
    spans = np.ptp(uv, axis=0)
    residual = np.abs((selected - origin) @ normal)
    tilt = math.degrees(math.asin(min(1.0, abs(float(normal[2]))))) if vertical else (
        math.degrees(math.acos(min(1.0, abs(float(normal[2])))))
    )
    metrics = {"point_count": len(points), "inlier_count": len(selected),
               "inlier_fraction": float(inliers.mean()),
               "plane_origin_world_m": origin.tolist(),
               "plane_normal_world": normal.tolist(),
               "plane_axes_world": basis.tolist(), "span_m": spans.tolist(),
               "plane_residual_p95_m": float(np.percentile(residual, 95)),
               "tilt_deg": tilt}
    if refine:
        metrics["refinement_iterations"] = refinement_iterations
    if (tilt > 15 or float(np.min(spans)) < 0.025 or values[1] <= 1e-10
            or metrics["plane_residual_p95_m"] > PLANE_RESIDUAL_M):
        return _check("unknown", "plane_degenerate_or_misaligned", metrics, thresholds)
    # Two similarly supported, separated planes are not one identified opening.
    alternatives = [row for row in candidates if row[0] >= 0.75 * sampled_inlier_count]
    if any(abs(float((a - origin) @ normal)) > 0.025
           and abs(float(n @ normal)) > 0.95 for _, _, a, n in alternatives):
        return _check("unknown", "multiple_parallel_planes", metrics, thresholds)
    cells = np.floor(uv / CELL_M + 1e-9).astype(np.int64)
    unique_cells = np.unique(cells, axis=0)
    metrics.update({"occupied_cell_count": len(unique_cells), "cell_size_m": CELL_M,
                    "occupied_area_m2": float(len(unique_cells) * CELL_M**2)})
    result = _check("consistent", "observed_plane_measured", metrics, thresholds)
    result["observed_points_world_m"] = selected.tolist()
    result["occupied_cells"] = unique_cells.tolist()
    return result


def fit_support_geometry(support_geometry: Mapping[str, Any]) -> dict[str, Any]:
    """Fit only a measured near-horizontal patch, without completing holes."""
    return _plane(_points(support_geometry, "support_geometry"), vertical=False,
        refine=support_geometry.get("depth_boundary_evidence", {}).get("schema") ==
        "ora.placement-depth-boundary-evidence.v1")


def fit_rim_geometry(rim_points_world: Any | None) -> dict[str, Any]:
    """Fit an Agent-selected visible ring; a PCA object axis is never used."""
    thresholds = {"minimum_points": 6, "maximum_points": 12,
                  "minimum_radius_m": 0.018, "maximum_radius_m": 0.10,
                  "maximum_plane_residual_m": 0.003,
                  "maximum_circle_residual_m": 0.003,
                  "minimum_arc_coverage_deg": 240.0,
                  "upright_maximum_tilt_deg": 20.0,
                  "sideways_minimum_tilt_deg": 60.0}
    if rim_points_world is None:
        return _check("unknown", "rim_points_not_provided", thresholds=thresholds)
    points = _array_points(rim_points_world, "rim_points_world", minimum=6)
    if len(points) > 12:
        raise ValueError("rim_points_world must contain 6..12 points")
    origin = points.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov((points - origin).T))
    if values[1] <= 1e-8 or values[0] / values[1] > 0.03:
        return _check("unknown", "rim_not_a_resolved_plane", thresholds=thresholds)
    normal = vectors[:, 0]
    if normal[2] < 0:
        normal = -normal
    axes = _basis(normal)
    uv = (points - origin) @ axes.T
    design = np.column_stack([2 * uv, np.ones(len(points))])
    solution, _, rank, _ = np.linalg.lstsq(design, np.sum(uv**2, axis=1), rcond=None)
    if rank < 3:
        return _check("unknown", "rim_circle_degenerate", thresholds=thresholds)
    center = solution[:2]
    radius_squared = float(solution[2] + center @ center)
    if radius_squared <= 0:
        return _check("unknown", "rim_circle_degenerate", thresholds=thresholds)
    radius = math.sqrt(radius_squared)
    radial = np.linalg.norm(uv - center, axis=1)
    circle_residual = float(np.max(np.abs(radial - radius)))
    plane_residual = float(np.max(np.abs((points - origin) @ normal)))
    angles = np.sort(np.mod(np.arctan2(uv[:, 1] - center[1], uv[:, 0] - center[0]),
                            2 * math.pi))
    largest_gap = float(np.max(np.diff(np.r_[angles, angles[0] + 2 * math.pi])))
    coverage = math.degrees(2 * math.pi - largest_gap)
    tilt = math.degrees(math.acos(min(1.0, abs(float(normal[2])))))
    metrics = {"point_count": len(points), "radius_m": radius,
               "center_world_m": (origin + center @ axes).tolist(),
               "normal_world": normal.tolist(), "tilt_deg": tilt,
               "plane_residual_max_m": plane_residual,
               "circle_residual_max_m": circle_residual, "arc_coverage_deg": coverage}
    if not 0.018 <= radius <= 0.10 or plane_residual > 0.003 or circle_residual > 0.003:
        return _check("unknown", "rim_circle_fit_not_supported", metrics, thresholds)
    if coverage < 240 - 1e-6:
        return _check("unknown", "rim_arc_coverage_insufficient", metrics, thresholds)
    if tilt <= 20:
        return _check("consistent", "selected_rim_axis_near_vertical", metrics, thresholds)
    if tilt >= 60:
        return _check("contradicted", "selected_rim_axis_sideways", metrics, thresholds)
    return _check("unknown", "selected_rim_axis_tilted", metrics, thresholds)


def _opening(boundary_geometry: Mapping[str, Any] | None, support: dict[str, Any],
             kind: str) -> dict[str, Any]:
    if kind == "support_surface":
        return _check("unknown", "opening_not_required")
    if boundary_geometry is None:
        return _check("unknown", "opening_geometry_not_provided")
    boundary = _points(boundary_geometry, "boundary_geometry")
    model = _plane(boundary, vertical=kind == "front_cavity",
        refine=boundary_geometry.get("depth_boundary_evidence", {}).get("schema") ==
        "ora.placement-depth-boundary-evidence.v1")
    if model["status"] != "consistent" or support["status"] != "consistent":
        return _check("unknown", "opening_or_interior_support_unresolved", {
            "boundary_plane_status": model["status"],
            "boundary_plane_reason": model["reason"],
        })
    origin = np.asarray(model["metrics"]["plane_origin_world_m"])
    normal = np.asarray(model["metrics"]["plane_normal_world"])
    interior = np.asarray(support["observed_points_world_m"])
    signed = (interior - origin) @ normal
    positive = float(np.mean(signed > 0.015))
    negative = float(np.mean(signed < -0.015))
    if max(positive, negative) < 0.75:
        return _check("unknown", "opening_inward_direction_ambiguous", {
            "support_positive_fraction": positive, "support_negative_fraction": negative,
        }, {"minimum_separated_support_fraction": 0.75, "separation_m": 0.015})
    inward = normal if positive > negative else -normal
    if kind == "open_container" and inward[2] > -0.95:
        return _check("unknown", "container_rim_not_above_support")
    # An opening is more than an infinite plane.  Its observed outer envelope
    # supplies necessary rejection bounds, never an inferred clear aperture.
    # In particular, a hull across frame/rim points does not fill an unseen hole.
    boundary_points = np.asarray(model["observed_points_world_m"])
    axes = np.asarray(model["metrics"]["plane_axes_world"])
    hull = _hull((boundary_points - origin) @ axes.T)
    area = 0.0 if len(hull) < 3 else abs(float(np.sum(
        hull[:, 0] * np.roll(hull[:, 1], -1)
        - hull[:, 1] * np.roll(hull[:, 0], -1),
    ))) / 2
    model["metrics"]["observed_opening_outer_envelope_area_m2"] = area
    if area <= CELL_M**2:
        return _check("unknown", "opening_outer_envelope_degenerate", model["metrics"],
                      {"minimum_outer_envelope_area_m2": CELL_M**2})
    if kind == "open_container":
        support_margin = _signed_hull_margin((interior - origin) @ axes.T, hull)
        overlaps = bool(np.any(support_margin >= -REGION_MARGIN_M))
    else:
        # A front aperture and its interior floor need a common observed
        # lateral range; the floor may legitimately sit below the frame.
        support_normal = np.asarray(support["metrics"]["plane_normal_world"])
        lateral = np.cross(support_normal, inward)
        length = float(np.linalg.norm(lateral))
        if length <= 1e-8:
            return _check("unknown", "support_opening_lateral_axis_unresolved")
        lateral /= length
        support_lateral = interior @ lateral
        boundary_lateral = boundary_points @ lateral
        overlap = min(float(support_lateral.max()), float(boundary_lateral.max())) - max(
            float(support_lateral.min()), float(boundary_lateral.min()),
        )
        model["metrics"]["observed_support_opening_lateral_overlap_m"] = overlap
        overlaps = overlap > 0
    if not overlaps:
        return _check("unknown", "selected_support_does_not_overlap_observed_opening",
                      model["metrics"])
    model["reason"] = "observed_opening_plane_with_support_side"
    model["metrics"]["inward_normal_world"] = inward.tolist()
    model["metrics"]["support_side_fraction"] = max(positive, negative)
    model["metrics"]["outer_envelope_is_passage_certificate"] = False
    return model


def _hull(points: np.ndarray) -> np.ndarray:
    ordered = sorted(set(map(tuple, points.tolist())))
    if len(ordered) < 3:
        return np.asarray(ordered, dtype=float)
    def cross(a: tuple, b: tuple, c: tuple) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    lower: list[tuple] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def _signed_hull_margin(points: np.ndarray, hull: np.ndarray) -> np.ndarray:
    if len(hull) < 3:
        return np.full(len(points), -math.inf)
    edges = np.roll(hull, -1, axis=0) - hull
    relative = points[:, None, :] - hull[None, :, :]
    distances = (edges[None, :, 0] * relative[:, :, 1]
                 - edges[None, :, 1] * relative[:, :, 0]) / np.linalg.norm(edges, axis=1)
    return np.min(distances, axis=1)


def _support_coordinates(points: np.ndarray, support: dict[str, Any]) -> np.ndarray:
    metrics = support["metrics"]
    return (points - np.asarray(metrics["plane_origin_world_m"])) @ np.asarray(
        metrics["plane_axes_world"]).T


def _opening_outer_margin(points: np.ndarray, opening: dict[str, Any]) -> np.ndarray:
    """Project observed points to the opening's rejection-only outer envelope."""
    metrics = opening["metrics"]
    origin = np.asarray(metrics["plane_origin_world_m"])
    axes = np.asarray(metrics["plane_axes_world"])
    observed = np.asarray(opening["observed_points_world_m"])
    return _signed_hull_margin((points - origin) @ axes.T, _hull((observed - origin) @ axes.T))


def _covered(uv: np.ndarray, support: dict[str, Any]) -> np.ndarray:
    # Only the cell directly hit by an observed point is supported.  No hull,
    # morphological closing, neighbouring-cell dilation or hole filling.
    occupied = set(map(tuple, support["occupied_cells"]))
    cells = np.floor(uv / CELL_M + 1e-9).astype(np.int64)
    return np.asarray([tuple(cell) in occupied for cell in cells], dtype=bool)


def _region_check(points: np.ndarray, support: dict[str, Any], opening: dict[str, Any],
                  kind: str) -> dict[str, Any]:
    thresholds = {"outside_margin_m": REGION_MARGIN_M, "outside_fraction": 0.15,
                  "minimum_observed_footprint_coverage": 1.0}
    if support["status"] != "consistent":
        return _check("unknown", "support_plane_unresolved", thresholds=thresholds)
    if kind != "support_surface" and opening["status"] != "consistent":
        return _check("unknown", "opening_plane_unresolved", thresholds=thresholds)
    uv = _support_coordinates(points, support)
    support_uv = _support_coordinates(np.asarray(support["observed_points_world_m"]), support)
    margin = _signed_hull_margin(uv, _hull(support_uv))
    covered = _covered(uv, support)
    metrics = {"visible_point_count": len(points),
               "outside_observed_envelope_fraction": float(np.mean(margin < -REGION_MARGIN_M)),
               "observed_cell_coverage_fraction": float(covered.mean()),
               "minimum_observed_envelope_margin_m": float(np.min(margin))}
    opening_aligned = True
    if kind != "support_surface":
        opening_margin = _opening_outer_margin(points, opening)
        metrics.update({
            "outside_observed_opening_envelope_fraction": float(np.mean(
                opening_margin < -REGION_MARGIN_M)),
            "minimum_observed_opening_envelope_margin_m": float(np.min(opening_margin)),
            "opening_outer_envelope_is_passage_certificate": False,
        })
        if metrics["outside_observed_opening_envelope_fraction"] > 0.15:
            return _check("contradicted", "visible_object_exceeds_observed_opening_envelope",
                          metrics, thresholds)
        opening_aligned = bool((opening_margin >= -1e-9).all())
    if kind == "front_cavity":
        om = opening["metrics"]
        signed = (points - np.asarray(om["plane_origin_world_m"])) @ np.asarray(
            om["inward_normal_world"])
        metrics.update({"outside_opening_fraction": float(np.mean(signed < -REGION_MARGIN_M)),
                        "minimum_inward_distance_m": float(np.min(signed))})
        if metrics["outside_opening_fraction"] > 0.15:
            return _check("contradicted", "visible_object_outside_cavity_opening", metrics, thresholds)
    if metrics["outside_observed_envelope_fraction"] > 0.15:
        return _check("contradicted", "visible_footprint_exceeds_observed_support_envelope", metrics, thresholds)
    if not opening_aligned:
        return _check("unknown", "observed_opening_envelope_alignment_unresolved", metrics, thresholds)
    if bool(covered.all()):
        return _check("consistent", "visible_footprint_over_observed_support_only", metrics, thresholds)
    return _check("unknown", "support_region_occluded_or_unobserved", metrics, thresholds)


def _protected_check(protected_geometries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not isinstance(protected_geometries, (tuple, list)) or len(protected_geometries) > 8:
        raise ValueError("protected_geometries must contain at most eight correspondence pairs")
    items = []
    # The caller's pairs arrive as two index-aligned reference arrays, so each
    # comparison must name the index it answers.  Position in `items` already
    # carries it, but only by convention; `source_index` states it, which is
    # what lets the flattened input keep per-item attribution.
    for index, pair in enumerate(protected_geometries):
        if not isinstance(pair, Mapping) or set(pair) != {"before_geometry", "current_geometry"}:
            raise ValueError("protected geometry requires before_geometry and current_geometry")
        before = _points(pair["before_geometry"], "protected before_geometry")
        current = _points(pair["current_geometry"], "protected current_geometry")
        if pair["before_geometry"].get("source_camera") != pair["current_geometry"].get("source_camera"):
            items.append({"source_index": index, **_check("unknown", "protected_camera_changed")})
            continue
        displacement = float(np.linalg.norm(np.median(current, axis=0) - np.median(before, axis=0)))
        extent_change = float(np.linalg.norm(np.ptp(current, axis=0) - np.ptp(before, axis=0)))
        metrics = {"visible_centroid_shift_m": displacement, "visible_extent_change_m": extent_change}
        if extent_change > 0.025:
            item = _check("unknown", "protected_visibility_or_shape_changed", metrics)
        elif displacement > 0.025:
            item = _check("contradicted", "protected_visible_geometry_moved", metrics)
        else:
            item = _check("consistent", "protected_visible_geometry_compatible", metrics)
        items.append({"source_index": index, **item})
    if not items:
        return _check("unknown", "no_protected_correspondences")
    status = _aggregate(items)
    return _check(status, "protected_public_geometry_comparison", {"items": items},
                  {"maximum_centroid_shift_m": 0.025, "maximum_extent_change_m": 0.025})


def _aggregate(checks: Sequence[Mapping[str, Any]]) -> str:
    if any(check["status"] == "contradicted" for check in checks):
        return "contradicted"
    if any(check["status"] == "unknown" for check in checks):
        return "unknown"
    return "consistent"


def assess_geometry(
    object_geometry: Mapping[str, Any], support_geometry: Mapping[str, Any],
    boundary_geometry: Mapping[str, Any] | None = None, *, destination_kind: str,
    tcp_pose: Any, rim_points_world: Any | None = None,
    protected_geometries: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Return decomposed evidence, never a placement/stability Boolean."""
    if destination_kind not in DESTINATIONS:
        raise ValueError("destination_kind must be support_surface, open_container or front_cavity")
    pose = _pose(tcp_pose)
    points = _points(object_geometry, "object_geometry")
    support = fit_support_geometry(support_geometry)
    opening = _opening(boundary_geometry, support, destination_kind)
    rim = fit_rim_geometry(rim_points_world)
    support_check = _check("unknown", "support_plane_unresolved")
    if support["status"] == "consistent":
        metrics = support["metrics"]
        distances = (points - np.asarray(metrics["plane_origin_world_m"])) @ np.asarray(
            metrics["plane_normal_world"])
        below = float(np.mean(distances < -0.015))
        support_check = _check(
            "contradicted" if below > 0.15 else "unknown",
            "visible_geometry_below_selected_support" if below > 0.15 else "base_support_not_observable",
            {"minimum_visible_plane_distance_m": float(np.min(distances)),
             "visible_plane_distance_p05_m": float(np.percentile(distances, 5)),
             "below_plane_fraction": below},
            {"below_plane_margin_m": 0.015, "below_plane_fraction": 0.15},
        )
    checks = {"region": _region_check(points, support, opening, destination_kind),
              "support": support_check, "orientation": rim,
              "protected_objects": _protected_check(protected_geometries)}
    geometry = {"support": support, "opening": opening, "rim": rim}
    from .placement_depth import SCHEMA as depth_schema
    source_geometries = {"object": object_geometry, "support": support_geometry}
    if boundary_geometry is not None:
        source_geometries["boundary"] = boundary_geometry
    depth_sources = {role: value["depth_boundary_evidence"] for role, value in source_geometries.items()
                     if value.get("depth_boundary_evidence", {}).get("schema") == depth_schema}
    if depth_sources:
        count = sum(value["measurements"]["uncertain_pixel_count"] for value in depth_sources.values())
        usable = all(value["status"] == "usable" for value in depth_sources.values())
        raw_points = points
        excluded = depth_sources.get("object", {}).get("uncertain_points_world_m", [])
        if excluded:
            raw_points = np.vstack((points, _array_points(excluded, "uncertain object points")))
        raw_region = _region_check(raw_points, support, opening, destination_kind)
        geometry["depth_boundary"] = {
            "schema": depth_schema, "status": "usable" if usable else "unknown",
            "uncertain_pixel_count": count, "full_object_fit": "unknown",
            "raw_object_region_status": raw_region["status"],
            "raw_object_region_reason": raw_region["reason"],
            "retained_object_region_status": checks["region"]["status"],
            "comparison_scope": "raw_and_retained_object_points_against_retained_destination_observations",
            "sources": {role: {key: value[key] for key in ("status", "reason", "measurements", "thresholds")}
                        for role, value in depth_sources.items()},
            "uncertain_pixels_are_certified_background": False,
        }
        if not usable:
            checks["region"] = _check("unknown", "public_depth_boundary_evidence_unusable")
        elif count and checks["region"]["status"] == "consistent":
            checks["region"] = _check("unknown", "uncertain_depth_boundary_full_object_fit_unresolved",
                                      checks["region"]["metrics"], checks["region"]["thresholds"])
    return {"status": _aggregate(list(checks.values())), "checks": checks,
            "geometry": geometry,
            "object_visible_offset_from_tcp_world_m": (np.median(points, axis=0) - pose[:3]).tolist(),
            "claim_boundary": CLAIM_BOUNDARY}


def _patch_centers(points: np.ndarray, support: dict[str, Any],
                   protected_geometries: Sequence[Mapping[str, Any]],
                   ledger: dict[str, Any] | None = None) -> list[np.ndarray]:
    axes = np.asarray(support["metrics"]["plane_axes_world"])
    origin = np.asarray(support["metrics"]["plane_origin_world_m"])
    cells = np.asarray(support["occupied_cells"], dtype=float)
    centers = (cells + 0.5) * CELL_M
    object_uv = _support_coordinates(points, support)
    # Every observed object point matters; subsampling could omit a handle or
    # corner that is the only visible contradiction to a candidate footprint.
    offsets = object_uv - np.median(object_uv, axis=0)
    observed_uv = _support_coordinates(np.asarray(support["observed_points_world_m"]), support)
    outer = _hull(observed_uv)
    margins = _signed_hull_margin(centers, outer)
    ordered = np.argsort(-margins, kind="stable")[:256]
    protected = []
    for pair in protected_geometries:
        protected.extend(_support_coordinates(_points(pair["current_geometry"], "protected geometry"), support))
    protected_tree = cKDTree(np.asarray(protected)) if protected else None
    selected: list[np.ndarray] = []
    counts = {"uncovered_footprint": 0, "protected_proximity": 0, "too_close_to_selected": 0}
    for index in ordered:
        uv = centers[index]
        footprint = uv + offsets
        if not _covered(footprint, support).all():
            counts["uncovered_footprint"] += 1
            continue
        if protected_tree is not None:
            distances, _ = protected_tree.query(footprint)
            if float(np.min(distances)) < 0.02:
                counts["protected_proximity"] += 1
                continue
        world = origin + uv @ axes
        if any(np.linalg.norm(world - previous) < 0.025 for previous in selected):
            counts["too_close_to_selected"] += 1
            continue
        selected.append(world)
        if len(selected) == 3:
            break
    if ledger is not None:
        # A policy cannot reason about a rejection it cannot see. These were
        # computed to pick a reason string and then discarded; the hull margin
        # in particular is the only number that says how much room the best
        # observed patch actually had.
        ledger.update(
            considered_patch_count=int(len(ordered)),
            rejected_patch_counts=dict(counts),
            best_observed_hull_margin_m=float(margins[ordered[0]]) if len(ordered) else None,
        )
    return selected


def _protected_tcp_segment_conflict(
    start: np.ndarray, end: np.ndarray,
    protected_geometries: Sequence[Mapping[str, Any]],
) -> bool:
    """Reject observed points near the TCP segment, not certify arm clearance."""
    delta = end - start
    squared_length = float(delta @ delta)
    for pair in protected_geometries:
        points = _points(pair["current_geometry"], "protected current_geometry")
        fraction = np.clip((points - start) @ delta / squared_length, 0, 1) if (
            squared_length > 1e-12) else np.zeros(len(points))
        closest = start + fraction[:, None] * delta
        if float(np.min(np.linalg.norm(points - closest, axis=1))) < 0.02:
            return True
    return False


def propose_placement(
    object_geometry: Mapping[str, Any], support_geometry: Mapping[str, Any],
    boundary_geometry: Mapping[str, Any] | None = None, *, destination_kind: str,
    tcp_pose: Any, phase: str, gripper_open: bool = False,
    rim_points_world: Any | None = None,
    protected_geometries: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Propose at most three current-pose, hold-orientation numerical goals.

    These are not executable plan references.  Every world change invalidates
    them.  The caller owns action-history checks, regular motion admission and
    new observation/replanning after each single step.
    """
    if phase not in PHASES:
        raise ValueError("phase must be preplace, lower or exit")
    if not isinstance(gripper_open, bool):
        raise ValueError("gripper_open must be a Boolean established by the caller")
    pose = _pose(tcp_pose)
    points = _points(object_geometry, "object_geometry")
    assessment = assess_geometry(
        object_geometry, support_geometry, boundary_geometry,
        destination_kind=destination_kind, tcp_pose=pose,
        rim_points_world=rim_points_world, protected_geometries=protected_geometries,
    )
    # Deliver what was measured even when no candidate survives. Every return
    # below shares this ledger, so an empty candidate list still carries the
    # counts and margins that explain why it is empty, and the observed lowest
    # point that decides whether a descent is even available.
    ledger: dict[str, Any] = {
        "considered_patch_count": None,
        "rejected_patch_counts": {},
        "best_observed_hull_margin_m": None,
        "rejected_for_opening_envelope": 0,
        "rejected_for_protected_proximity": 0,
        "visible_lowest_point_above_support_m": None,
        "available_descent_m": None,
    }
    assessment["geometry"]["proposal_rejections"] = ledger
    result: dict[str, Any] = {"status": "unknown", "reason": "support_plane_unresolved",
                              "candidates": [], "assessment": assessment,
                              "claim_boundary": CLAIM_BOUNDARY}
    depth_boundary = assessment["geometry"].get("depth_boundary")
    if depth_boundary and depth_boundary["status"] != "usable":
        result["reason"] = "public_depth_boundary_evidence_unusable"
        return result
    support = assessment["geometry"]["support"]
    opening = assessment["geometry"]["opening"]
    if support["status"] != "consistent":
        return result
    if destination_kind != "support_surface" and opening["status"] != "consistent":
        result["reason"] = "opening_plane_unresolved"
        return result
    if phase in {"preplace", "lower"} and (
        assessment["checks"]["orientation"]["status"] == "contradicted"
    ):
        result["reason"] = "requested_upright_rim_contradicted_no_implicit_rotation"
        return result
    normal = np.asarray(support["metrics"]["plane_normal_world"])
    origin = np.asarray(support["metrics"]["plane_origin_world_m"])
    goals: list[tuple[np.ndarray, dict[str, Any]]] = []
    rejected_opening = 0
    if phase == "preplace":
        if gripper_open:
            result["reason"] = "preplace_requires_closed_gripper_not_retention_proof"
            return result
        for center in _patch_centers(points, support, protected_geometries, ledger):
            object_center = np.median(points, axis=0)
            planar_delta = center - object_center
            planar_delta -= float(planar_delta @ normal) * normal
            clearance = 0.08
            if destination_kind == "open_container":
                rim_origin = np.asarray(opening["metrics"]["plane_origin_world_m"])
                clearance += max(0.0, float((rim_origin - origin) @ normal))
            # Only observed lower surface is used; hidden base extent is unknown.
            lowest = float(np.min((points + planar_delta - origin) @ normal))
            if destination_kind == "front_cavity":
                # A frontal opening has finite headroom: the free-space 80 mm
                # standoff may contradict its observed upper bound. Select the
                # highest bounded height that retains EVERY visible point in
                # the unchanged rejection-only envelope. Never descend below
                # 20 mm or treat this as a release/free-passage certificate.
                for clearance in (0.08, 0.06, 0.04, 0.02):
                    translated = points + planar_delta + (clearance - lowest) * normal
                    if bool((_opening_outer_margin(translated, opening) >= -1e-9).all()):
                        break
                else:
                    rejected_opening += 1
                    continue
            target = pose[:3] + planar_delta + (clearance - lowest) * normal
            goals.append((target, {"visible_lower_surface_standoff_m": clearance,
                                   "visible_geometry_rigid_offset_assumption": True,
                                   "hidden_base_extent_known": False}))
    elif phase == "lower":
        if gripper_open:
            result["reason"] = "lower_requires_closed_gripper_not_retention_proof"
            return result
        region = assessment["checks"]["region"]
        if region["status"] != "consistent":
            result["reason"] = "visible_footprint_alignment_not_established"
            return result
        lowest = float(np.min((points - origin) @ normal))
        distance = min(0.03, lowest - 0.015)
        ledger["visible_lowest_point_above_support_m"] = lowest
        ledger["available_descent_m"] = distance
        if distance < 0.005:
            result["reason"] = "near_support_no_further_descent_without_base_evidence"
            return result
        goals.append((pose[:3] - distance * normal,
                      {"translation_m": distance, "release_height_established": False}))
    else:
        if not gripper_open:
            result["reason"] = "exit_requires_observed_open_gripper"
            return result
        direction = -np.asarray(opening["metrics"]["inward_normal_world"]) if (
            destination_kind == "front_cavity") else normal
        for distance in (0.02, 0.03, 0.04):
            goals.append((pose[:3] + distance * direction,
                          {"translation_m": distance,
                           "disengagement_established": False,
                           "finger_and_arm_clearance_established": False}))
    rejected_protected = 0
    for target, metrics in goals:
        if destination_kind != "support_surface" and phase in {"preplace", "lower"}:
            translated_points = points + (target - pose[:3])
            opening_margin = _opening_outer_margin(translated_points, opening)
            # Check every observed object point, including thin handles/corners.
            # Passing this necessary outer bound does not establish free passage.
            if not bool((opening_margin >= -1e-9).all()):
                rejected_opening += 1
                continue
            metrics["minimum_observed_opening_envelope_margin_m"] = float(np.min(opening_margin))
            metrics["opening_outer_envelope_is_passage_certificate"] = False
        if _protected_tcp_segment_conflict(pose[:3], target, protected_geometries):
            rejected_protected += 1
            continue
        metrics["protected_tcp_segment_radius_m"] = 0.02
        metrics["protected_check_scope"] = "visible_points_near_tcp_segment_only_not_fingers_or_arm"
        if depth_boundary:
            metrics["conditional_on_unresolved_silhouette"] = depth_boundary["uncertain_pixel_count"] > 0
            metrics["full_object_fit"] = "unknown"
            metrics["uncertain_depth_boundary_pixels"] = depth_boundary["uncertain_pixel_count"]
        result["candidates"].append({
            "candidate_id": f"P{len(result['candidates'])}", "phase": phase,
            "target_pose": {"position_m": target.tolist(), "orientation_xyzw": pose[3:].tolist()},
            "metrics": metrics, "claim_boundary": CLAIM_BOUNDARY,
        })
    ledger["rejected_for_opening_envelope"] = rejected_opening
    ledger["rejected_for_protected_proximity"] = rejected_protected
    result["status"] = "candidates" if result["candidates"] else "no_candidate"
    result["reason"] = (
        "single_step_observed_geometry_proposals" if result["candidates"] else
        "visible_object_exceeds_observed_opening_outer_envelope" if rejected_opening else
        "protected_visible_geometry_near_tcp_segment" if rejected_protected else
        "no_observed_patch_fits_visible_footprint"
    )
    return result
