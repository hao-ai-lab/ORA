"""Deterministic two-view RGB-D fusion for the reduced OpenETA-Light surface."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


VIEW_AXES = {
    "pointcloud_top": ("x", "y"),
    "pointcloud_front": ("x", "z"),
    "pointcloud_side": ("y", "z"),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha(path),
        "bytes": path.stat().st_size,
    }


def fuse_rgbd_pointcloud_sources(
    *, observation_root: Path, observation: Mapping[str, Any],
    workspace_bounds_m: Mapping[str, list[float]],
    camera_names: Sequence[str] = ("agentview", "wrist"),
    voxel_size_m: float = 0.01, max_depth_m: float | None = None,
) -> dict[str, Any]:
    """Fuse public RGB-D sources in world coordinates with stable attribution."""

    import numpy as np
    from PIL import Image

    if not camera_names or len(set(camera_names)) != len(camera_names):
        raise ValueError("pointcloud camera_names must be unique and non-empty")
    if not isinstance(voxel_size_m, (int, float)) or float(voxel_size_m) <= 0:
        raise ValueError("pointcloud voxel_size_m must be positive")
    if max_depth_m is not None and (
        not isinstance(max_depth_m, (int, float)) or float(max_depth_m) <= 0
    ):
        raise ValueError("pointcloud max_depth_m must be positive when provided")

    resolved_root = observation_root.resolve()
    axes = {"x": 0, "y": 1, "z": 2}
    origin = np.asarray([
        float(workspace_bounds_m[axis][0]) for axis in ("x", "y", "z")
    ])
    for axis in axes:
        bounds = workspace_bounds_m.get(axis)
        if (
            not isinstance(bounds, list) or len(bounds) != 2
            or not all(isinstance(value, (int, float)) for value in bounds)
            or not float(bounds[0]) < float(bounds[1])
        ):
            raise ValueError("pointcloud workspace bounds must contain finite xyz ranges")

    points_parts = []
    colour_parts = []
    source_parts = []
    sources: list[dict[str, Any]] = []
    for source_index, camera_name in enumerate(camera_names):
        record = observation.get("cameras", {}).get(camera_name)
        if not isinstance(record, Mapping):
            raise ValueError(f"pointcloud fusion requires camera {camera_name}")
        files = record.get("files", {})
        resolved: dict[str, Path] = {}
        source_files: dict[str, Any] = {}
        for role in ("rgb_lossless_png", "depth_numpy_m_f32", "calibration"):
            reference = files.get(role) if isinstance(files, Mapping) else None
            relative = reference.get("path") if isinstance(reference, Mapping) else None
            if not isinstance(relative, str):
                raise ValueError(f"{camera_name} lacks {role}")
            path = (observation_root / relative).resolve()
            if not path.is_file() or not path.is_relative_to(resolved_root):
                raise ValueError(f"{camera_name} artifact is unavailable")
            digest = _sha(path)
            if digest != reference.get("sha256"):
                raise ValueError(f"{camera_name} {role} hash drift")
            resolved[role] = path
            source_files[role] = {
                "path": relative,
                "sha256": digest,
                "bytes": path.stat().st_size,
            }

        rgb = np.asarray(Image.open(resolved["rgb_lossless_png"]).convert("RGB"))
        depth = np.load(
            resolved["depth_numpy_m_f32"], allow_pickle=False,
        ).astype(np.float64)
        calibration = json.loads(
            resolved["calibration"].read_text(encoding="utf-8")
        )
        intrinsic = np.asarray(calibration["intrinsic_cv"], dtype=np.float64)
        extrinsic = np.asarray(calibration["extrinsic_cv"], dtype=np.float64)
        if extrinsic.shape == (3, 4):
            extrinsic = np.vstack([extrinsic, [0.0, 0.0, 0.0, 1.0]])
        if (
            rgb.shape[:2] != depth.shape or intrinsic.shape != (3, 3)
            or extrinsic.shape != (4, 4)
        ):
            raise ValueError(f"{camera_name} RGB-D calibration shape mismatch")
        valid = np.isfinite(depth) & (depth > 1e-6)
        if max_depth_m is not None:
            valid &= depth <= float(max_depth_m)
        rows, columns = np.nonzero(valid)
        z = depth[rows, columns]
        camera_points = np.stack([
            (columns - intrinsic[0, 2]) * z / intrinsic[0, 0],
            (rows - intrinsic[1, 2]) * z / intrinsic[1, 1],
            z,
            np.ones_like(z),
        ], axis=1)
        world = (np.linalg.inv(extrinsic) @ camera_points.T).T[:, :3]
        workspace_mask = np.ones(world.shape[0], dtype=bool)
        for axis, axis_index in axes.items():
            low, high = workspace_bounds_m[axis]
            workspace_mask &= (
                (world[:, axis_index] >= float(low))
                & (world[:, axis_index] <= float(high))
            )
        world = world[workspace_mask]
        colours = rgb[rows, columns][workspace_mask]
        points_parts.append(world)
        colour_parts.append(colours)
        source_parts.append(np.full(world.shape[0], source_index, dtype=np.int16))
        sources.append({
            "camera": camera_name,
            "source_index": source_index,
            "source_resolution": [int(rgb.shape[1]), int(rgb.shape[0])],
            "files": source_files,
            "valid_depth_points": int(len(rows)),
            "workspace_points": int(world.shape[0]),
        })

    points = np.concatenate(points_parts, axis=0)
    colours = np.concatenate(colour_parts, axis=0)
    source_indices = np.concatenate(source_parts, axis=0)
    if not len(points):
        raise ValueError("fused pointcloud has no points inside workspace bounds")
    voxels = np.floor((points - origin) / float(voxel_size_m)).astype(np.int64)
    original_order = np.arange(len(points), dtype=np.int64)
    order = np.lexsort((original_order, voxels[:, 2], voxels[:, 1], voxels[:, 0]))
    sorted_voxels = voxels[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = np.any(sorted_voxels[1:] != sorted_voxels[:-1], axis=1)
    retained = order[first]
    points = points[retained]
    colours = colours[retained]
    source_indices = source_indices[retained]
    for source in sources:
        source["retained_voxel_points"] = int(np.count_nonzero(
            source_indices == source["source_index"]
        ))
    return {
        "xyz_world_m": points,
        "rgb_u8": colours,
        "source_index": source_indices,
        "sources": sources,
        "camera_order": list(camera_names),
        "voxel_size_m": float(voxel_size_m),
        "max_depth_m": float(max_depth_m) if max_depth_m is not None else None,
        "workspace_bounds_m": {
            axis: [float(value) for value in workspace_bounds_m[axis]]
            for axis in ("x", "y", "z")
        },
        "input_point_count": int(sum(
            source["valid_depth_points"] for source in sources
        )),
        "workspace_point_count": int(sum(
            source["workspace_points"] for source in sources
        )),
        "filtered_voxel_count": int(points.shape[0]),
        "deduplication": "stable_first_by_camera_order",
    }


def materialize_two_view_pointcloud(
    *, observation_root: Path, observation: Mapping[str, Any], output_root: Path,
    workspace_bounds_m: Mapping[str, list[float]], voxel_size_m: float = 0.01,
    resolution: int | Sequence[int] = 512,
) -> dict[str, Any]:
    """Back-project agentview+wrist, fuse deterministically, and render three views."""

    import numpy as np
    from PIL import Image

    started_ns = time.monotonic_ns()
    if isinstance(resolution, int):
        width = height = resolution
    elif (
        isinstance(resolution, Sequence)
        and len(resolution) == 2
        and all(isinstance(value, int) and value > 0 for value in resolution)
    ):
        width, height = int(resolution[0]), int(resolution[1])
    else:
        raise ValueError("pointcloud resolution must be a positive integer or [width, height]")
    output_root.mkdir(parents=True, exist_ok=True)
    fused = fuse_rgbd_pointcloud_sources(
        observation_root=observation_root,
        observation=observation,
        workspace_bounds_m=workspace_bounds_m,
        camera_names=("agentview", "wrist"),
        voxel_size_m=voxel_size_m,
    )
    points = fused["xyz_world_m"]
    colours = fused["rgb_u8"]
    axes = {"x": 0, "y": 1, "z": 2}
    np.savez_compressed(output_root / "points.npz", xyz_m=points, rgb_u8=colours)
    views: dict[str, Any] = {}
    for view_name, (horizontal, vertical) in VIEW_AXES.items():
        h_index, v_index = axes[horizontal], axes[vertical]
        h_low, h_high = [float(value) for value in workspace_bounds_m[horizontal]]
        v_low, v_high = [float(value) for value in workspace_bounds_m[vertical]]
        x = np.rint((points[:, h_index] - h_low) / (h_high - h_low) * (width - 1)).astype(int)
        y = np.rint((v_high - points[:, v_index]) / (v_high - v_low) * (height - 1)).astype(int)
        in_frame = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        canvas = np.full((height, width, 3), 245, dtype=np.uint8)
        # Stable far-to-near overwrite follows retained voxel order.
        canvas[y[in_frame], x[in_frame]] = colours[in_frame]
        path = output_root / f"{view_name}.png"
        Image.fromarray(canvas, mode="RGB").save(path)
        views[view_name] = {
            "artifact": _artifact(path, output_root),
            "resolution": [width, height],
            "horizontal_axis": horizontal,
            "vertical_axis": vertical,
            "horizontal_bounds_m": [h_low, h_high],
            "vertical_bounds_m": [v_low, v_high],
            "coordinate_convention": "opencv_top_left_xy",
        }
    manifest = {
        "schema": "ora.two-view-pointcloud.v1",
        "observation_id": observation.get("observation_id"),
        "camera_count": 2,
        "source": "two_view_rgbd",
        "sources": [
            {
                "camera": source["camera"],
                "source_resolution": source["source_resolution"],
                "rgb_sha256": source["files"]["rgb_lossless_png"]["sha256"],
                "depth_sha256": source["files"]["depth_numpy_m_f32"]["sha256"],
                "calibration_sha256": source["files"]["calibration"]["sha256"],
                "valid_depth_points": source["valid_depth_points"],
            }
            for source in fused["sources"]
        ],
        "voxel_size_m": voxel_size_m,
        "workspace_bounds_m": dict(workspace_bounds_m),
        "input_point_count": fused["input_point_count"],
        "filtered_voxel_count": fused["filtered_voxel_count"],
        "points": _artifact(output_root / "points.npz", output_root),
        "views": views,
        "robot_visuals_hidden": False,
        "materialization_duration_s": round((time.monotonic_ns() - started_ns) / 1e9, 6),
    }
    _atomic_json(output_root / "manifest.json", manifest)
    manifest["manifest"] = _artifact(output_root / "manifest.json", output_root)
    for artifact_path in sorted(path for path in output_root.iterdir() if path.is_file()):
        artifact_path.chmod(0o444)
    return manifest


def orthographic_click_constraint(view: Mapping[str, Any], x: int, y: int) -> dict[str, float]:
    """Map one orthographic pixel to its two displayed world-axis coordinates."""

    width, height = view["resolution"]
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError("pointcloud click is outside the selected view")
    h_low, h_high = view["horizontal_bounds_m"]
    v_low, v_high = view["vertical_bounds_m"]
    return {
        str(view["horizontal_axis"]): float(h_low + x / (width - 1) * (h_high - h_low)),
        str(view["vertical_axis"]): float(v_high - y / (height - 1) * (v_high - v_low)),
    }
