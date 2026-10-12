"""``mark_point``: turning one pixel of one view into one named world mark.

A pixel measurement is cheap -- 39 ms median on the host CPU -- but a model turn
is not. Across 222 sealed runs, 379 turns (36.4% of every turn that called
``mark_point``) issued several ``mark_point`` calls against one unchanged
observation, and the median provider request behind such a turn carried 258,193
input tokens against 1,476 output. Each of those extra turns bought a re-read of
the whole context for one pixel. The batched form in ``mark_point_batch``
measures up to eight pixels of one view in a single call, so a policy that needs
the two rim points and the handle of a mug pays for one turn instead of three.

This module owns the measurement itself, unchanged from the single-pixel form it
was extracted from: the depth estimator that fails closed at silhouettes and far
planes, the OpenCV camera-to-world projection, the reuse and cross-view
verification rules, and the per-call overlay. ``execute`` is the only entry
point a tool runtime needs; it routes the historical four-field call straight to
``measure`` and anything naming ``point_ids``/``points_uv`` to the batch.
"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping

from .geometry import orthographic_click_constraint
from .mark_point_overlay import begin_feedback_batch, finish_feedback_batch
from .mark_point_schema import (
    MARK_POINT_AGGREGATION_CAPABILITY,
    MARK_POINT_BATCH_CAPABILITY,
    MARK_POINT_BATCH_COLOURS,
    MARK_POINT_BATCH_RESULT_SCHEMA,
    MARK_POINT_DEPTH_WINDOW_RADIUS,
    MARK_POINT_MAX_BATCH,
    MARK_POINT_VERIFY_TOLERANCE_M,
    POINTCLOUD_VIEWS,
    UNMEASURED_METHOD,
    batch_input_schema,
)
from .robot import ToolError

__all__ = [
    "MARK_POINT_AGGREGATION_CAPABILITY",
    "MARK_POINT_BATCH_CAPABILITY",
    "MARK_POINT_BATCH_COLOURS",
    "MARK_POINT_BATCH_RESULT_SCHEMA",
    "MARK_POINT_DEPTH_WINDOW_RADIUS",
    "MARK_POINT_MAX_BATCH",
    "MARK_POINT_VERIFY_TOLERANCE_M",
    "batch_input_schema",
    "begin_feedback_batch",
    "execute",
    "finish_feedback_batch",
    "measure",
    "opencv_camera_point_to_world",
    "robust_depth_measurement",
]


def _quality(
    sink: dict[str, Any] | None, *, status: str, method: str, **fields: Any,
) -> None:
    """Record what the estimator learned, including on the paths that fail.

    The single-pixel form discards this on the error path, so a rejected pixel
    tells the policy only its error code.  For a batch that is not enough: the
    remedy for ``depth_discontinuity`` (aim further inside the surface) is not
    the remedy for ``invalid_depth`` (the pixel sees the far plane), and a
    model correcting seven pixels at once needs to tell them apart.  The sink
    is filled in place so a raising estimator can still report what it saw.
    """

    if sink is None:
        return
    sink.clear()
    sink.update({"quality_status": status, "quality_method": method})
    sink.update(fields)


def robust_depth_measurement(
    depth: Any, *, source_x: int, source_y: int,
    calibration: Mapping[str, Any],
    quality_sink: dict[str, Any] | None = None,
) -> tuple[float, dict[str, Any]]:
    """Measure one local surface or fail closed at depth boundaries.

    A clicked integer pixel is not sufficient evidence near silhouettes.  This
    estimator requires a locally supported cluster around the clicked depth,
    rejects mixed foreground/background windows, and rejects camera far-plane
    values.  It intentionally prefers asking the policy to click a safer pixel
    over returning a precise-looking but arbitrary world point.

    ``quality_sink``, when given, receives the diagnostics gathered so far even
    when this raises.  Passing nothing keeps the historical behavior exactly.
    """

    import numpy as np

    _quality(quality_sink, status="unmeasured", method=UNMEASURED_METHOD)
    values = np.asarray(depth, dtype=np.float64)
    if values.ndim != 2 or not (
        0 <= source_y < values.shape[0] and 0 <= source_x < values.shape[1]
    ):
        raise ToolError("selected RGB-D pixel is outside the depth array")
    bounds = calibration.get("valid_depth_range_m", {})
    near = float(bounds.get("near", 0.0)) if isinstance(bounds, Mapping) else 0.0
    far_raw = bounds.get("far") if isinstance(bounds, Mapping) else None
    far = float(far_raw) if isinstance(far_raw, (int, float)) else math.inf
    center = float(values[source_y, source_x])
    far_limit = far * 0.98 if math.isfinite(far) else math.inf
    radius = MARK_POINT_DEPTH_WINDOW_RADIUS
    if not math.isfinite(center) or center <= max(near, 1e-6) or center >= far_limit:
        _quality(
            quality_sink, status="partial", method="local_supported_depth_v1",
            quality_window_radius_px=radius,
            quality_depth_m=center if math.isfinite(center) else None,
        )
        raise ToolError(
            "selected RGB-D pixel is invalid or at the camera far plane",
            code="invalid_depth",
        )
    y0, y1 = max(0, source_y-radius), min(values.shape[0], source_y+radius+1)
    x0, x1 = max(0, source_x-radius), min(values.shape[1], source_x+radius+1)
    window = values[y0:y1, x0:x1].reshape(-1)
    valid = window[
        np.isfinite(window)
        & (window > max(near, 1e-6))
        & (window < far_limit)
    ]
    partial = {
        "quality_window_radius_px": radius,
        "quality_depth_m": center,
        "quality_valid_samples": int(valid.size),
    }
    if valid.size < 3:
        _quality(
            quality_sink, status="partial", method="local_supported_depth_v1",
            **partial,
        )
        raise ToolError(
            "selected RGB-D neighborhood has insufficient valid depth support",
            code="insufficient_depth_support",
        )
    tolerance = max(0.015, 0.03 * center)
    compatible = valid[np.abs(valid - center) <= tolerance]
    incompatible = valid[np.abs(valid - center) > tolerance]
    partial["quality_supporting_samples"] = int(compatible.size)
    partial["quality_rejected_samples"] = int(incompatible.size)
    if compatible.size < 3:
        _quality(
            quality_sink, status="partial", method="local_supported_depth_v1",
            **partial,
        )
        raise ToolError(
            "selected RGB-D pixel is not supported by a local surface",
            code="insufficient_depth_support",
        )
    if incompatible.size >= max(3, int(math.ceil(valid.size * 0.20))):
        separation = abs(float(np.median(incompatible)) - float(np.median(compatible)))
        if separation > max(0.05, 0.08 * center):
            _quality(
                quality_sink, status="partial",
                method="local_supported_depth_v1", **partial,
            )
            raise ToolError(
                "selected RGB-D neighborhood crosses a depth discontinuity; "
                "click farther inside one visible surface",
                code="depth_discontinuity",
            )
    measured = float(np.median(compatible))
    deviations = np.abs(compatible - measured)
    quality = {
        "method": "local_supported_depth_v1",
        "window_radius_px": radius,
        "valid_samples": int(valid.size),
        "supporting_samples": int(compatible.size),
        "rejected_samples": int(incompatible.size),
        "depth_m": measured,
        "max_support_deviation_m": float(np.max(deviations)),
    }
    _quality(
        quality_sink, status="measured", method="local_supported_depth_v1",
        quality_window_radius_px=radius,
        quality_depth_m=measured,
        quality_valid_samples=int(valid.size),
        quality_supporting_samples=int(compatible.size),
        quality_rejected_samples=int(incompatible.size),
        quality_max_support_deviation_m=float(np.max(deviations)),
    )
    return measured, quality


def opencv_camera_point_to_world(
    camera_point_xyzw: Any, *, calibration: Mapping[str, Any],
) -> list[float]:
    """Transform one OpenCV camera point through a validated world extrinsic.

    OpenCV commonly serializes a world-to-camera extrinsic as ``[R|t]`` with
    shape 3x4.  A homogeneous 4x4 matrix is equally valid.  Keeping this
    normalization at the RGB-D measurement boundary prevents callers from
    treating the standard 3x4 representation as directly invertible.
    """

    import numpy as np

    intrinsic = np.asarray(calibration.get("intrinsic_cv"), dtype=np.float64)
    extrinsic = np.asarray(calibration.get("extrinsic_cv"), dtype=np.float64)
    point = np.asarray(camera_point_xyzw, dtype=np.float64)
    if (
        intrinsic.shape != (3, 3)
        or not np.all(np.isfinite(intrinsic))
        or intrinsic[0, 0] <= 0.0
        or intrinsic[1, 1] <= 0.0
    ):
        raise ToolError(
            "camera intrinsic_cv must be a finite 3x3 OpenCV matrix with "
            "positive focal lengths",
            code="invalid_camera_calibration",
        )
    if extrinsic.shape == (3, 4):
        extrinsic = np.vstack((extrinsic, [0.0, 0.0, 0.0, 1.0]))
    if extrinsic.shape != (4, 4) or not np.all(np.isfinite(extrinsic)):
        raise ToolError(
            "camera extrinsic_cv must be a finite OpenCV 3x4 [R|t] or "
            "homogeneous 4x4 matrix",
            code="invalid_camera_calibration",
        )
    if point.shape != (4,) or not np.all(np.isfinite(point)):
        raise ToolError(
            "camera point must be one finite homogeneous XYZ1 vector",
            code="invalid_camera_calibration",
        )
    try:
        world = np.linalg.solve(extrinsic, point)
    except np.linalg.LinAlgError as exc:
        raise ToolError(
            "camera extrinsic_cv is singular",
            code="invalid_camera_calibration",
        ) from exc
    if not np.all(np.isfinite(world[:3])) or abs(float(world[3])) < 1e-12:
        raise ToolError(
            "camera calibration produced an invalid homogeneous world point",
            code="invalid_camera_calibration",
        )
    return (world[:3] / world[3]).tolist()


def measure(
    runtime: Any, *, point_id: str, view: str, x: int, y: int,
    suppress_images: bool = False,
    quality_sink: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure one pixel of one view into one named world mark.

    This is the whole of the historical single-pixel behavior, unchanged: the
    same reuse short-circuit, the same cross-view verification tolerance, the
    same point-cloud axis accumulation and the same per-call overlay.
    ``suppress_images`` is what a batch uses to take the one composite overlay
    instead of one image per pixel, and ``quality_sink`` is how it recovers the
    depth diagnostics when this raises.
    """

    from .runtime.core import _atomic_json, _png_base64

    if not isinstance(point_id, str) or not point_id or not isinstance(view, str):
        raise ToolError("point_id and view must be non-empty strings")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (x, y)):
        raise ToolError("x and y must be integers")
    control, root, observation, _pose, _gripper = runtime._current_grip_state()
    points = runtime._load_points()
    previous = points.get(point_id)
    solved_previous = (
        isinstance(previous, Mapping) and previous.get("status") == "solved"
    )
    measurement_identity = {
        "observation_id": control["observation_id"],
        "view": view, "x": x, "y": y,
    }
    if solved_previous and any(
        all(source.get(key) == value for key, value in measurement_identity.items())
        for source in previous.get("sources", [])
        if isinstance(source, Mapping)
    ):
        result = {
            "success": True, "status": "solved", "point_id": point_id,
            "xyz_m": previous["xyz_m"],
            "views": [source["view"] for source in previous.get("sources", [])],
        }
        if runtime.surface.runtime_id == "native_atomic_causal":
            result["action_workspace"] = dict(previous.get(
                "action_workspace", runtime._marked_point_workspace(previous["xyz_m"]),
            ))
        return result
    if (
        isinstance(previous, Mapping) and not solved_previous
        and previous.get("observation_id") != control["observation_id"]
    ):
        raise ToolError("pending point is stale after world change", code="stale_point")
    source_view = "agentview" if view == "candidate_agentview" else view
    if source_view in observation.get("cameras", {}):
        import numpy as np

        _record, rgb_path, depth_path, calibration_path = runtime._camera(
            root, observation, source_view
        )
        depth = np.load(depth_path, allow_pickle=False)
        output_width, output_height = runtime._rgb_output_resolution(rgb_path)
        if not (0 <= x < output_width and 0 <= y < output_height):
            raise ToolError(
                f"click [{x}, {y}] is outside the {output_width}x{output_height} {view} view",
                field="x" if not 0 <= x < output_width else "y", received=[x, y],
                constraints={"width": output_width, "height": output_height,
                             "coordinate_convention": "opencv_top_left_xy"},
            )
        source_x = int(round(
            x * (depth.shape[1] - 1) / max(output_width - 1, 1)
        ))
        source_y = int(round(
            y * (depth.shape[0] - 1) / max(output_height - 1, 1)
        ))
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        z, measurement_quality = robust_depth_measurement(
            depth, source_x=source_x, source_y=source_y,
            calibration=calibration, quality_sink=quality_sink,
        )
        intrinsic = np.asarray(calibration.get("intrinsic_cv"), dtype=float)
        if (
            intrinsic.shape != (3, 3)
            or not np.all(np.isfinite(intrinsic))
            or intrinsic[0, 0] <= 0.0
            or intrinsic[1, 1] <= 0.0
        ):
            raise ToolError(
                "camera intrinsic_cv must be a finite 3x3 OpenCV matrix "
                "with positive focal lengths",
                code="invalid_camera_calibration",
            )
        camera_point = np.asarray([
            (source_x-intrinsic[0,2])*z/intrinsic[0,0],
            (source_y-intrinsic[1,2])*z/intrinsic[1,1], z, 1.0,
        ])
        xyz = opencv_camera_point_to_world(
            camera_point, calibration=calibration,
        )
        source = {
            **measurement_identity,
            "source_x": source_x, "source_y": source_y,
            "measurement_quality": measurement_quality,
        }
        if solved_previous:
            disagreement = float(np.linalg.norm(
                np.asarray(previous["xyz_m"], dtype=float)
                - np.asarray(xyz, dtype=float)
            ))
            if disagreement > MARK_POINT_VERIFY_TOLERANCE_M:
                raise ToolError(
                    f"independent RGB-D measurements disagree by {disagreement:.4f} m, "
                    f"more than the {MARK_POINT_VERIFY_TOLERANCE_M} m maximum",
                    code="inconsistent_mark",
                    constraints={
                        "distance_m": disagreement,
                        "maximum_m": MARK_POINT_VERIFY_TOLERANCE_M,
                    },
                )
            record = dict(previous)
            record["sources"] = [*previous.get("sources", []), source]
            record["last_verification"] = {
                "observation_id": control["observation_id"],
                "distance_m": disagreement,
                "measurement_quality": measurement_quality,
            }
        else:
            record = {
                "status": "solved", "point_id": point_id, "xyz_m": xyz,
                "observation_id": control["observation_id"],
                "sources": [source],
            }
    elif view in POINTCLOUD_VIEWS:
        pointcloud = runtime._pointcloud(root, observation)
        constraint = orthographic_click_constraint(pointcloud["views"][view], x, y)
        if solved_previous:
            xyz_by_axis = dict(zip(("x", "y", "z"), previous["xyz_m"]))
            disagreement = max(
                abs(float(xyz_by_axis[axis]) - float(value))
                for axis, value in constraint.items()
            )
            if disagreement > MARK_POINT_VERIFY_TOLERANCE_M:
                raise ToolError(
                    f"independent pointcloud measurements disagree by {disagreement:.4f} m, "
                    f"more than the {MARK_POINT_VERIFY_TOLERANCE_M} m maximum",
                    code="inconsistent_mark",
                    constraints={
                        "distance_m": disagreement,
                        "maximum_m": MARK_POINT_VERIFY_TOLERANCE_M,
                    },
                )
            record = dict(previous)
            record["sources"] = [
                *previous.get("sources", []), measurement_identity,
            ]
            record["last_verification"] = {
                "observation_id": control["observation_id"],
                "axis_max_error_m": disagreement,
            }
        else:
            constraints = dict(previous.get("constraints", {})) if isinstance(previous, Mapping) else {}
            for axis, value in constraint.items():
                if axis in constraints and abs(float(constraints[axis]) - value) > 0.02:
                    raise ToolError(
                        f"complementary pointcloud clicks disagree on {axis} by "
                        f"{abs(float(constraints[axis]) - value):.4f} m, more than the 0.02 m maximum",
                        code="inconsistent_mark", constraints={
                            "axis": axis, "distance_m": abs(float(constraints[axis]) - value),
                            "maximum_m": 0.02})
                constraints[axis] = value
            sources = list(previous.get("sources", [])) if isinstance(previous, Mapping) else []
            sources.append(measurement_identity)
            if set(constraints) == {"x", "y", "z"}:
                record = {"status": "solved", "point_id": point_id, "xyz_m": [constraints[a] for a in ("x", "y", "z")], "observation_id": control["observation_id"], "sources": sources}
            else:
                record = {"status": "pending", "point_id": point_id, "constraints": constraints, "observation_id": control["observation_id"], "sources": sources}
    else:
        raise ToolError("view is not markable")
    points[point_id] = record
    if record["status"] == "solved":
        record["action_workspace"] = runtime._marked_point_workspace(
            record["xyz_m"]
        )
    _atomic_json(runtime.point_path, points)
    result_status = "verified" if solved_previous else record["status"]
    result = {"success": True, "status": result_status, "point_id": point_id}
    if record["status"] == "solved":
        result["xyz_m"] = record["xyz_m"]
        result["views"] = [source["view"] for source in record["sources"]]
        if runtime.surface.runtime_id == "native_atomic_causal":
            result["action_workspace"] = record["action_workspace"]
        if not suppress_images and runtime._active_mark_point_batch is None:
            encoded = []
            pointcloud = None
            labelled = [(point_id, list(record["xyz_m"]), (255, 0, 255))]
            for source in record["sources"]:
                source_view = str(source["view"])
                if (
                    source_view in observation.get("cameras", {})
                    or source_view == "candidate_agentview"
                ):
                    encoded.append(_png_base64(runtime._annotated_rgb(
                        root=root, observation=observation,
                        camera="agentview" if source_view == "candidate_agentview" else source_view,
                        labelled_world_points=labelled,
                        identity_kind="mark_point_feedback",
                    )))
                elif source_view in POINTCLOUD_VIEWS:
                    pointcloud = pointcloud or runtime._pointcloud(root, observation)
                    encoded.append(_png_base64(runtime._annotated_pointcloud(
                        observation=observation, pointcloud=pointcloud,
                        view=source_view, labelled_world_points=labelled,
                        identity_kind="mark_point_feedback",
                    )))
            result["images_base64"] = encoded
    else:
        result["next_views"] = sorted(
            set(POINTCLOUD_VIEWS)
            - {source["view"] for source in record["sources"]}
        )
    return result


def execute(runtime: Any, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Dispatch one ``mark_point`` call to the form its surface publishes.

    On a surface without the batch capability the four-field call is matched
    exactly and reaches the historical path untouched, down to the error text
    for a malformed call.

    On a surface *with* it the four-field call is refused rather than served.
    A declaring surface publishes only the batched result schema -- admitting
    both from one contract would need a top-level ``oneOf`` -- so measuring a
    single pixel through the old path would produce a result the surface's own
    schema rejects, and the runtime would raise that at the policy as a
    non-retryable internal fault instead of a correctable call error. Routing it
    into the batch validator turns it into ``invalid_arguments`` naming the
    three fields that surface accepts.
    """

    keys = set(arguments)
    batched = runtime.surface.supports(MARK_POINT_BATCH_CAPABILITY)
    if keys == {"point_id", "view", "x", "y"} and not batched:
        return measure(
            runtime, point_id=arguments["point_id"], view=arguments["view"],
            x=arguments["x"], y=arguments["y"],
        )
    if batched or keys & {"point_ids", "points_uv"}:
        from .mark_point_batch import execute_batch

        return execute_batch(runtime, arguments)
    raise ToolError("mark_point requires point_id, view, x and y")
