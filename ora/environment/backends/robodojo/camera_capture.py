"""Exact single-environment RGB-D extraction from original tiled annotators."""

from __future__ import annotations

import json
from pathlib import Path
from types import MethodType
import time
from typing import Any

import numpy as np


SCHEMA = "ora.robodojo-camera-buffer-readback.v1"
FORMATS = {
    "rgb": ("rgba", 4, np.dtype("uint8")),
    "rgba": ("rgba", 4, np.dtype("uint8")),
    "depth": ("distance_to_image_plane", 1, np.dtype("float32")),
    "distance_to_image_plane": ("distance_to_image_plane", 1, np.dtype("float32")),
}


def renderer_intrinsics(parameters: dict[str, Any], resolution: tuple[int, int]) -> np.ndarray:
    """Convert the actual centered pinhole render projection to pixel units."""
    width, height = resolution
    observed = tuple(int(value) for value in parameters["renderProductResolution"])
    projection = np.asarray(parameters["cameraProjection"], dtype=np.float64).reshape(4, 4)
    if (parameters.get("cameraModel") != "pinhole" or observed != resolution
            or not np.isfinite(projection).all() or min(width, height) <= 0
            or projection[0, 0] <= 0 or projection[1, 1] <= 0
            or not np.allclose(projection[2, :2], 0, atol=1e-8)
            or not np.allclose(parameters["cameraApertureOffset"], 0, atol=1e-8)):
        raise ValueError("original camera render projection differs from the centered pinhole binding")
    return np.array([[projection[0, 0] * width / 2, 0, width / 2],
                     [0, projection[1, 1] * height / 2, height / 2], [0, 0, 1]])


def attach_camera_parameters(manager: Any) -> None:
    """Observe calibration on each original single-camera render product."""
    import omni.replicator.core as rep

    if manager.num_envs != 1 or len(manager.tiled_cameras) != 3:
        raise ValueError("camera calibration requires three original single-camera products")
    for view in manager.tiled_cameras:
        annotator = rep.AnnotatorRegistry.get_annotator("CameraParams")
        annotator.attach([view._render_product_path])
        view._harness_camera_params = annotator
        for name, attribute in (("IsaacReadSimulationTime", "_harness_simulation_time"),
                                ("ReferenceTime", "_harness_reference_time")):
            clock = rep.AnnotatorRegistry.get_annotator(name)
            clock.attach([view._render_product_path])
            setattr(view, attribute, clock)


def synchronize_render_products(*, manager: Any, render: Any, physics_clock: Any,
                                simulation_time: Any, physical_poses: Any, output_root: Path,
                                max_renders: int = 8,
                                simulation_time_offset_s: float | None = None) -> dict[str, Any]:
    """Drain delayed render frames without taking a physics step.

    ``simulation_time_offset_s`` is the simulated time of every earlier
    simulation context in this worker (a scheduled RoboDojo session relaunches
    one per round). The ``IsaacReadSimulationTime`` annotator keeps counting
    across a relaunch (its ``resetOnStop`` input defaults to False) while the
    new SimulationContext clock restarts at 0, so a camera is current when its
    annotator time minus that offset equals this context's clock. ``None``
    (every unscheduled session) compares the raw annotator time and writes the
    record unchanged.
    """
    initial_step = physics_clock()
    expected_time = float(simulation_time())
    offset = 0.0 if simulation_time_offset_s is None else float(simulation_time_offset_s)
    if (manager.num_envs != 1 or len(manager.tiled_cameras) != 3 or not np.isfinite(expected_time)
            or not np.isfinite(offset) or offset < 0):
        raise ValueError("render synchronization requires three cameras and a finite simulation clock")
    record = {"schema": "ora.robodojo-render-synchronization.v1",
              "physics_step": initial_step, "simulation_time_s": expected_time,
              "matched": False, "samples": []}
    if simulation_time_offset_s is not None:
        record["simulation_time_offset_s"] = offset
    try:
        for index in range(max_renders):
            render()
            if physics_clock() != initial_step or float(simulation_time()) != expected_time:
                raise ValueError("camera synchronization advanced physics")
            cameras = []
            poses = physical_poses()
            for view, physical in zip(manager.tiled_cameras, poses, strict=True):
                timestamp = view._harness_simulation_time.get_data()
                reference = view._harness_reference_time.get_data()
                rendered = renderer_camera_pose(view._harness_camera_params.get_data(), np.zeros(3))
                cameras.append({"render_product": str(view._render_product_path),
                                "simulation_time_s": float(timestamp["simulationTime"]),
                                "optical_pose_max_abs_error": float(np.max(np.abs(rendered - physical))),
                                "reference_time": [int(reference["referenceTimeNumerator"]),
                                                   int(reference["referenceTimeDenominator"])]})
            record["samples"].append({"render_call": index + 1, "cameras": cameras})
            current = all(np.isfinite(row["simulation_time_s"])
                          and abs(row["simulation_time_s"] - offset - expected_time) <= 1e-6 for row in cameras)
            same_frame = len({tuple(row["reference_time"]) for row in cameras}) == 1
            aligned = all(row["optical_pose_max_abs_error"] <= 1e-4 for row in cameras)
            # The pinned headless SDK can stamp the current time while its
            # scene transforms still belong to the previous render. Flush that
            # first frame even when all camera mounts happened to stay still.
            if index >= 1 and current and same_frame and aligned:
                record["matched"] = True
                return record
        raise TimeoutError("camera render products did not reach the current physics frame")
    finally:
        with (output_root / "render-synchronization.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")


def renderer_camera_pose(parameters: dict[str, Any], origin: Any) -> np.ndarray:
    """Read the rendered CV optical pose in episode-local coordinates."""
    view = np.asarray(parameters["cameraViewTransform"], dtype=np.float64).reshape(4, 4).T
    pose = np.linalg.inv(np.diag([1., -1., -1., 1.]) @ view)
    pose[:3, 3] -= np.asarray(origin, dtype=np.float64)
    if not np.isfinite(pose).all():
        raise ValueError("rendered camera pose is not finite")
    return pose


def physical_camera_pose(camera: Any, *, use_fabric: bool) -> np.ndarray:
    """Read the actual optical prim from the active physical transform store."""
    def array(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float64)

    positions, orientations = camera._prim_view.get_world_poses(usd=not use_fabric)
    w, x, y, z = array(orientations)[0]
    pose = np.eye(4)
    pose[:3, 3] = array(positions)[0]
    pose[:3, :3] = [
        [1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)],
        [2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
        [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)],
    ]
    return pose @ np.diag([1., -1., -1., 1.])


def wait_for_render_assets(*, manager: Any, render: Any, assets_loading: Any,
                           stage_status: Any, physics_clock: Any, output_root: Path,
                           timeout_s: float = 180, record_name: str = "render-readiness.jsonl") -> None:
    """Apply the SDK texture wait once original camera products can render."""
    if manager.num_envs != 1 or len(manager.tiled_cameras) != 3:
        raise ValueError("texture readiness requires all three original render products")
    initial_step = physics_clock()
    started = time.monotonic()
    last_recorded = -5.0
    frames = 0
    with (output_root / record_name).open("x") as stream:
        while True:
            elapsed = time.monotonic() - started
            loading = bool(assets_loading())
            if frames == 0 or not loading or elapsed - last_recorded >= 5 or elapsed >= timeout_s:
                stream.write(json.dumps({"elapsed_s": elapsed, "render_frames": frames,
                                         "assets_loading": loading, "stage": stage_status(),
                                         "physics_step": physics_clock(),
                                         "render_products": [str(view._render_product_path)
                                                             for view in manager.tiled_cameras]}) + "\n")
                stream.flush()
                last_recorded = elapsed
            if physics_clock() != initial_step:
                raise ValueError("texture readiness rendering advanced physics")
            # Render once to submit the newly created products before accepting
            # a cached ready flag. Never replace or override the SDK predicate.
            if elapsed >= timeout_s:
                raise TimeoutError("original camera assets did not become ready")
            if frames and not loading:
                return
            render()
            frames += 1


def install_single_environment_readback(manager: Any, output_root: Path) -> None:
    """Replace only the upstream buffer reshape, preserving capture and timing."""
    if manager.num_envs != 1 or len(manager.tiled_cameras) != 3:
        raise ValueError("RGB-D readback requires one environment and three original cameras")
    receipts: dict[str, Any] = {}
    destination = output_root / "camera-buffer-readback.json"

    def get_data(view: Any, annotator_type: str, *, tiled: bool = False, out: Any = None):
        if tiled or annotator_type not in FORMATS or len(view.prims) != 1:
            raise ValueError("only single-environment batched RGB-D readback is supported")
        name, channels, dtype = FORMATS[annotator_type]
        width, height = map(int, view.camera_resolution)
        if min(width, height) <= 0 or tuple(view.tiled_resolution) != (width, height):
            raise ValueError("camera tile dimensions differ from the single original camera")
        annotator = view._annotators[name]
        raw = annotator.get_data(device="cpu")
        buffer, info = (raw["data"], raw.get("info", {})) if isinstance(raw, dict) else (raw, {})
        source_device = str(getattr(buffer, "device", "cpu"))
        key = f"{view._render_product_path}:{name}"
        first_read = key not in receipts
        raw_metadata = {
            "returned_type": type(buffer).__name__, "returned_device": source_device,
            "returned_shape": list(getattr(buffer, "shape", ())),
            "returned_dtype": str(getattr(buffer, "dtype", "unknown")),
            "supplied_output_device": str(getattr(out, "device", "none")),
        }
        if first_read:
            node = annotator.get_node() if hasattr(annotator, "get_node") else None
            if node is not None:
                raw_metadata["annotator_outputs"] = {
                    field: str(node.get_attribute(f"outputs:{field}").get())
                    for field in ("width", "height", "bufferSize", "cudaDeviceIndex")
                    if node.get_attribute(f"outputs:{field}").is_valid()
                }
            texture = getattr(view._render_product, "hydra_texture", None) if hasattr(view, "_render_product") else None
            if texture is not None and hasattr(texture, "get_updates_enabled"):
                raw_metadata["render_product_updates_enabled"] = bool(texture.get_updates_enabled())
            receipts[key] = {"render_product": str(view._render_product_path),
                             "annotator": name, "conversion_pending": True, **raw_metadata}
            destination.write_text(json.dumps({"schema": SCHEMA, "buffers": list(receipts.values())}, indent=2) + "\n")
        values = np.asarray(buffer.numpy() if hasattr(buffer, "numpy") else buffer)
        allowed = {(height, width, channels), (height * width * channels,)}
        if channels == 1:
            allowed.add((height, width))
        receipt = {
            **raw_metadata, "conversion_pending": False,
            "render_product": str(view._render_product_path), "annotator": name,
            "requested_device": "cpu", "returned_device": source_device,
            "shape": list(values.shape), "strides_bytes": list(values.strides),
            "dtype": str(values.dtype), "elements": int(values.size),
            "expected_batch_shape": [1, height, width, channels],
            "extraction": "numpy_exact_copy_and_batch_axis", "pixels_resampled": False,
        }
        if first_read:
            receipts[key] = receipt
            destination.write_text(json.dumps({"schema": SCHEMA, "buffers": list(receipts.values())}, indent=2) + "\n")
        if values.dtype != dtype or values.shape not in allowed:
            raise ValueError(f"original {name} camera buffer has unexpected shape/dtype: {values.shape}/{values.dtype}")
        # A single tile is already in raster order. Copy before returning: the
        # original annotator storage can be reused by the next renderer update.
        return values.reshape(1, height, width, channels).copy(), info

    for view in manager.tiled_cameras:
        view.get_data = MethodType(get_data, view)
