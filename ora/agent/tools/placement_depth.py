"""Opt-in public depth-edge uncertainty for placement, never mask correction.

The original semantic mask is immutable. A narrowly defined far-side boundary
discontinuity is uncertain, not proven background. Numerical proposals using
the remaining observations are conditional on that missing silhouette.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import binary_erosion


CAPABILITY = "public_rgbd_placement_depth_edges_v1"
SCHEMA = "ora.placement-depth-boundary-evidence.v1"
THRESHOLDS = {
    "boundary_width_pixels": 1,
    "minimum_closer_mask_neighbors": 2,
    "minimum_depth_jump_m": 0.020,
    "maximum_outside_depth_difference_m": 0.006,
    "maximum_uncertain_fraction": 0.10,
    "minimum_retained_points": 16,
    "maximum_depth_m": 2.5,
}


def classify_boundary_depth(mask: Any, depth_m: Any) -> dict[str, Any]:
    """Classify only a local far-side jump continuous with observed exterior.

    No object size, global quantile, support destination or candidate outcome
    enters this rule. Interior points and the near side of edges are retained.
    """
    raw = np.asarray(mask)
    depth = np.asarray(depth_m, dtype=np.float64)
    if (raw.ndim != 2 or depth.shape != raw.shape or not raw.size
        or raw.size > 1_000_000 or raw.dtype.kind not in "bui"
        or not np.isin(raw, (0, 1)).all()):
        raise ValueError("placement depth requires an aligned binary mask and metric depth")
    selected = raw.astype(bool)
    valid = np.isfinite(depth) & (depth > 1e-6) & (depth <= THRESHOLDS["maximum_depth_m"])
    observed = selected & valid
    boundary = selected & ~binary_erosion(selected, structure=np.ones((3, 3)), border_value=0)
    closer = np.zeros(depth.shape, dtype=np.uint8)
    exterior_continuity = np.zeros(depth.shape, dtype=bool)
    height, width = depth.shape
    # Slice overlaps rather than rolling: image boundaries cannot wrap around.
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if not (dy or dx):
                continue
            here = (slice(max(0, -dy), min(height, height - dy)),
                    slice(max(0, -dx), min(width, width - dx)))
            there = (slice(max(0, dy), min(height, height + dy)),
                     slice(max(0, dx), min(width, width + dx)))
            with np.errstate(invalid="ignore"):
                difference = depth[here] - depth[there]
                closer[here] += (observed[there] &
                    (difference >= THRESHOLDS["minimum_depth_jump_m"]))
                exterior_continuity[here] |= (valid[there] & ~selected[there] &
                    (np.abs(difference) <= THRESHOLDS["maximum_outside_depth_difference_m"]))
    uncertain = observed & boundary & exterior_continuity & (
        closer >= THRESHOLDS["minimum_closer_mask_neighbors"])
    retained = observed & ~uncertain
    total, count, remaining = int(observed.sum()), int(uncertain.sum()), int(retained.sum())
    fraction = count / total if total else 0.0
    reason = ("too_few_retained_depth_points" if remaining < THRESHOLDS["minimum_retained_points"] else
        "too_many_uncertain_boundary_pixels" if fraction > THRESHOLDS["maximum_uncertain_fraction"] else
        "local_far_side_boundary_uncertainty" if count else "no_classified_boundary_discontinuity")
    return {"status": "unknown" if remaining < 16 or fraction > .10 else "usable", "reason": reason,
        "retained_mask": retained, "uncertain_mask": uncertain,
        "measurements": {"mask_pixel_count": int(selected.sum()), "raw_valid_depth_point_count": total,
            "invalid_depth_pixel_count": int((selected & ~valid).sum()),
            "retained_point_count": remaining, "uncertain_pixel_count": count,
            "uncertain_fraction": fraction}, "thresholds": dict(THRESHOLDS)}


def project_geometry(mask: Any, depth_m: Any, intrinsic: Any, extrinsic: Any, *,
                     observation_id: str, target_ref: str, camera: str) -> dict[str, Any]:
    """Project every retained pixel; keep excluded pixel and world evidence.

    Unlike the earlier voxel representative projection, this placement-only
    projection does not subsample a retained thin handle or isolated corner.
    """
    classified = classify_boundary_depth(mask, depth_m)
    depth = np.asarray(depth_m, dtype=np.float64)
    k, e = np.asarray(intrinsic, dtype=np.float64), np.asarray(extrinsic, dtype=np.float64)
    if e.shape == (3, 4):
        e = np.vstack((e, [0., 0., 0., 1.]))
    if (k.shape != (3, 3) or e.shape != (4, 4) or not np.isfinite(k).all()
        or not np.isfinite(e).all() or k[0, 0] <= 0 or k[1, 1] <= 0
        or not np.allclose(k[2], [0, 0, 1], atol=1e-9)
        or not np.allclose(e[3], [0, 0, 0, 1], atol=1e-9)
        or abs(float(np.linalg.det(e))) < 1e-9):
        raise ValueError("placement depth calibration is invalid")
    transform = np.linalg.inv(e)

    def points(selected):
        y, x = np.nonzero(selected)
        z = depth[y, x]
        camera_points = np.column_stack(((x - k[0, 2]) * z / k[0, 0],
            (y - k[1, 2]) * z / k[1, 1], z, np.ones(len(z))))
        return (camera_points @ transform.T)[:, :3], np.column_stack((x, y))

    retained, _ = points(classified["retained_mask"])
    uncertain, pixels = points(classified["uncertain_mask"])
    if len(retained) < 16 or len(retained) > 100_000:
        from .placement_refusal import RetainedPointBoundsError
        raise RetainedPointBoundsError(len(retained), minimum=16, maximum=100_000,
                                       measurements=classified["measurements"])
    centroid = np.mean(retained, axis=0)
    values, vectors = np.linalg.eigh(np.cov((retained - centroid).T))
    order = np.argsort(values)[::-1]
    axes = vectors[:, order].T
    for axis in axes:
        if axis[int(np.argmax(np.abs(axis)))] < 0:
            axis *= -1
    evidence = {"schema": SCHEMA, **{key: classified[key] for key in
        ("status", "reason", "measurements", "thresholds")},
        "uncertain_pixels_uv": pixels.tolist(), "uncertain_points_world_m": uncertain.tolist(),
        "original_mask_modified": False, "depth_values_reconstructed": False,
        "uncertain_pixels_are_certified_background": False, "full_silhouette_established": False}
    return {"schema": "ora.observation-bound-target-geometry.v2",
        "observation_id": observation_id, "target_ref": target_ref, "source_camera": camera,
        "centroid_world_m": centroid.tolist(), "principal_axes_world": axes.tolist(),
        "principal_variances_m2": values[order].tolist(),
        "principal_extents_m": np.ptp((retained - centroid) @ axes.T, axis=0).tolist(),
        "visible_point_count": len(retained), "voxel_size_m": 0.0,
        "voxel_points_world_m": retained.tolist(), "depth_boundary_evidence": evidence,
        "claim_boundary": "all retained public depth observations; flagged far-side silhouette pixels remain "
            "uncertain, not certified background; no complete object, hidden surface or safe-fit certificate"}
