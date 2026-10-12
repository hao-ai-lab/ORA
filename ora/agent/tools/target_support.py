"""Support-aware height of a segmented target, from one RGB-D view.

``center_xyz_m`` is the median of the target's visible depth points. From an
oblique or top-down camera those points are mostly the object's top and
camera-facing side, so the centre sits well above the object's middle, and a
parallel-jaw side grasp aimed at it closes near the top edge. Nothing the policy
receives says where the object stands.

This module estimates the local support plane beneath the target from the depth
pixels in a ring just outside its mask, evaluates that plane under the target,
and reports the support height, the height of the visible top above it and the
midpoint between the two. All three are single-view estimates: the ring can see
a neighbouring object, the far side of a container or nothing but background,
and the visible top is the highest *visible* surface, not the object's true top.
They are labelled as such and never presented as ground truth.

Delivery is gated on :data:`CAPABILITY`. The live producers measure the fields
beside the principal frame, and :func:`ora.agent.tools.grasp.
project_target_frame` removes them for surfaces that do not declare it, whose
closed localization schemas would otherwise reject the extra properties.
"""

from __future__ import annotations

import math
from typing import Any

CAPABILITY = "target_support_height_v1"
BASIS = "single_view_depth_ring_estimate_not_ground_truth"
#: Localization keys this module owns, in publication order.
KEYS = ("support_z_m", "height_above_support_m", "mid_height_z_m",
        "support_point_count", "support_height_basis")

#: Depth the Runtime treats as valid, matching the localizer's own window.
_DEPTH_RANGE_M = (1e-6, 2.5)
#: A support more than this far below the visible top is not this target's.
_MAXIMUM_HEIGHT_M = 0.35
#: A support plane tilted more than this is not a surface the target stands on.
_MAXIMUM_TILT_RAD = 0.35
_MINIMUM_POINTS = 12
_MINIMUM_RESIDUAL_M = 0.004
_MAXIMUM_RING_PX = 60


def empty() -> dict[str, Any]:
    """The published shape when no estimate is available."""
    return {"support_z_m": None, "height_above_support_m": None, "mid_height_z_m": None,
            "support_point_count": 0, "support_height_basis": BASIS}


def _ring(mask: Any) -> Any:
    """Pixels a short distance outside the mask, skipping its mixed-depth edge."""
    import cv2
    import numpy as np

    selected = np.asarray(mask, dtype=np.uint8)
    radius = math.sqrt(max(1, int(selected.sum())) / math.pi)
    inner = max(2, round(0.15 * radius))
    outer = min(_MAXIMUM_RING_PX, inner + max(4, round(0.6 * radius)))

    def grown(pixels: int) -> Any:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pixels + 1, 2 * pixels + 1))
        return cv2.dilate(selected, kernel).astype(bool)

    return grown(outer) & ~grown(inner)


def _fit_plane(points: Any) -> tuple[Any, Any] | None:
    """Robust ``z = a x + b y + c`` over the ring points, or ``None``."""
    import numpy as np

    inliers = np.ones(len(points), dtype=bool)
    coefficients = None
    for _ in range(4):
        if int(inliers.sum()) < _MINIMUM_POINTS:
            return None
        chosen = points[inliers]
        design = np.column_stack([chosen[:, 0], chosen[:, 1], np.ones(len(chosen))])
        coefficients, *_ = np.linalg.lstsq(design, chosen[:, 2], rcond=None)
        residual = points[:, 2] - (points[:, 0] * coefficients[0] + points[:, 1] * coefficients[1]
                                   + coefficients[2])
        spread = float(np.median(np.abs(residual[inliers] - np.median(residual[inliers]))))
        inliers = np.abs(residual) <= max(_MINIMUM_RESIDUAL_M, 3.0 * 1.4826 * spread)
    if coefficients is None or int(inliers.sum()) < _MINIMUM_POINTS:
        return None
    return coefficients, inliers


def estimate(*, mask: Any, depth_m: Any, intrinsic: Any, camera_to_world: Any,
             object_points: Any) -> dict[str, Any]:
    """Support height under one localized target, from the same view's depth.

    ``object_points`` are the target's own world inlier points, the ones its
    centre and bounds were measured from. The visible top is their 97.5th
    percentile height, the same statistic ``aabb_max_xyz_m`` reports.
    """

    import numpy as np

    result = empty()
    depth = np.asarray(depth_m, dtype=np.float64)
    points = np.asarray(object_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < _MINIMUM_POINTS:
        return result
    ring = _ring(mask) & np.isfinite(depth) & (depth > _DEPTH_RANGE_M[0]) & (depth < _DEPTH_RANGE_M[1])
    rows, columns = np.nonzero(ring)
    if len(rows) < _MINIMUM_POINTS:
        return result
    z = depth[rows, columns]
    camera = np.stack([(columns - intrinsic[0, 2]) * z / intrinsic[0, 0],
                       (rows - intrinsic[1, 2]) * z / intrinsic[1, 1], z, np.ones_like(z)], axis=1)
    world = (np.asarray(camera_to_world, dtype=np.float64) @ camera.T).T[:, :3]
    top = float(np.percentile(points[:, 2], 97.5))
    middle = float(np.median(points[:, 2]))
    # The support lies below the target's visible surface and within a
    # tabletop object's height of its top; anything else is not beneath it.
    beneath = world[(world[:, 2] < middle) & (world[:, 2] > top - _MAXIMUM_HEIGHT_M)]
    fitted = _fit_plane(beneath) if len(beneath) >= _MINIMUM_POINTS else None
    if fitted is None:
        return result
    (a, b, c), inliers = fitted
    if math.atan(math.hypot(float(a), float(b))) > _MAXIMUM_TILT_RAD:
        result["support_point_count"] = int(inliers.sum())
        return result
    x, y = (float(value) for value in np.median(points[:, :2], axis=0))
    support = float(a * x + b * y + c)
    if not math.isfinite(support) or support >= top:
        return result
    result.update(support_z_m=support, height_above_support_m=top - support,
                  mid_height_z_m=0.5 * (support + top), support_point_count=int(inliers.sum()))
    return result


def projection(surface: Any, principal_frame_capability: str) -> dict[str, bool]:
    """Which measured target fields a surface receives, for ``project_target_frame``."""
    return {"declared": bool(surface.supports(principal_frame_capability)),
            "support_declared": bool(surface.supports(CAPABILITY))}


__all__ = ["BASIS", "CAPABILITY", "KEYS", "empty", "estimate", "projection"]
