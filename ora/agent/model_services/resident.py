"""Shared resident SAM3 and Contact-GraspNet loopback model services.

Each process loads exactly one model, runs one warmup request, and only then
opens its HTTP listener.  The policy never talks to these endpoints directly;
the trusted ToolRuntime validates their model identities and keeps raw masks
outside policy context.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import logging
import os
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping


CAPX_SOURCE_COMMIT = "53e9966d7a8e2fa7494676772bccc35280f5c0ed"
SAM3_SOURCE_COMMIT = "6fe87d64a5beb9084923d7a9e002741178635b09"
SAM3_CHECKPOINT_SHA256 = (
    "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
)
CONTACT_SOURCE_COMMIT = "da3dcfb2f53e43b186083ee4a9d1e232f73efc98"
CONTACT_CHECKPOINT_SHA256 = (
    "39fc3439d5814043ba64e0715c127c2e8aca6ea376c22614e11af1bc89317762"
)
CONTACT_CONFIG_SHA256 = (
    "7ae6cc15c726e8ec2c46b0c799d58ca95caeb9ce9f261aaaf3e467712df76ac5"
)
# Upstream crops the scene around the target and resamples the crop to
# ``DATA.raw_num_points`` (20000). A denser crop is subsampled without
# replacement, which also randomizes point order. A sparser crop -- every
# surveyed LIBERO/RoboTwin/RoboDojo crop, 975-14204 distinct points -- keeps
# the scene's own scan/voxel order and gets random duplicates appended. The
# PointNet++ ball query keeps the first ``nsample`` neighbours by index, so an
# ordered cloud gives every centroid a one-sided neighbourhood, and the width
# head then collapses to its narrowest bin (8.97 mm). Measured on saved
# requests, order is the driver; duplicates mostly move scores. This adapter
# feeds each distinct crop point once, in a request-seeded random order, and
# only ever downsamples (with upstream's own sampler), never pads.
CONTACT_INPUT_POLICY = "distinct_shuffled_crop_points_v1"
CONTACT_INPUT_POLICY_SCHEMA = "ora.contact-graspnet-input-policy.v1"
CONTACT_INPUT_REGIME_SCHEMA = "ora.contact-graspnet-input-regime.v1"
CONTACT_MIN_REGION_UNIQUE_POINTS = 16


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_file(path: Path, expected_sha256: str) -> Path:
    value = path.resolve()
    if not value.is_file():
        raise RuntimeError(f"required model artifact is unavailable: {value}")
    actual = _sha256(value)
    if actual != expected_sha256:
        raise RuntimeError(
            f"model artifact hash drift for {value}: {actual} != {expected_sha256}"
        )
    return value


def _require_git_source(path: Path, expected_commit: str) -> Path:
    """Fail closed unless one model source is the exact clean pinned checkout."""

    value = path.resolve()
    try:
        head = subprocess.run(
            ["git", "-C", str(value), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            [
                "git", "-C", str(value), "status", "--porcelain",
                "--untracked-files=no",
            ],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"model source is not a readable Git checkout: {value}") from exc
    if head != expected_commit:
        raise RuntimeError(
            f"model source revision drift for {value}: {head} != {expected_commit}"
        )
    if dirty:
        raise RuntimeError(f"model source has tracked modifications: {value}")
    return value


def _encode_npy(value: Any) -> str:
    import numpy as np

    stream = io.BytesIO()
    np.save(stream, np.asarray(value), allow_pickle=False)
    return base64.b64encode(stream.getvalue()).decode("ascii")


def _decode_npy(value: Any) -> Any:
    import numpy as np

    if not isinstance(value, str):
        raise ValueError("array payload must be a base64 string")
    raw = base64.b64decode(value, validate=True)
    return np.load(io.BytesIO(raw), allow_pickle=False)


def _decode_png(value: Any) -> Any:
    from PIL import Image

    if not isinstance(value, str):
        raise ValueError("image payload must be a base64 string")
    raw = base64.b64decode(value, validate=True)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _request_id() -> str:
    return "perception-" + uuid.uuid4().hex


def _response_service_identity(service: Any) -> dict[str, Any]:
    health = service.health()
    return {"schema": "ora.model-service-response-identity.v1",
            **{key: health[key] for key in ("instance_id", "pid", "started_unix_s", "adapter_sha256")}}


class SegmentService:
    schema = "ora.model-service.segment-point.v1"

    def __init__(self, *, source_root: Path, checkpoint: Path) -> None:
        started = time.perf_counter()
        checkpoint = _require_file(checkpoint, SAM3_CHECKPOINT_SHA256)
        source_root = _require_git_source(source_root, SAM3_SOURCE_COMMIT)
        if not (source_root / "sam3/model_builder.py").is_file():
            raise RuntimeError(f"SAM3 source is unavailable: {source_root}")
        sys.path.insert(0, str(source_root))
        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        if not torch.cuda.is_available():
            raise RuntimeError("SAM3 service requires CUDA")
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.torch = torch
        self.model = build_sam3_image_model(
            device="cuda",
            checkpoint_path=str(checkpoint),
            load_from_HF=False,
            enable_inst_interactivity=True,
        )
        self.model.eval()
        self.processor = Sam3Processor(
            self.model, device="cuda", confidence_threshold=0.0,
        )
        self.model_identity = {
            "family": "sam3",
            "source_commit": SAM3_SOURCE_COMMIT,
            "checkpoint_sha256": SAM3_CHECKPOINT_SHA256,
            "capx_service_source_commit": CAPX_SOURCE_COMMIT,
            "device": torch.cuda.get_device_name(0),
        }
        self.adapter_sha256 = _sha256(Path(__file__).resolve())
        self.instance_id = "model-instance-" + uuid.uuid4().hex
        self.load_count = 1
        self.load_ms = (time.perf_counter() - started) * 1000
        self.warmup_ms = self._warmup()
        self.started_unix_s = time.time()
        self.requests = 0

    def _predict(self, image: Any, point_uv: tuple[float, float]) -> tuple[Any, float]:
        import numpy as np

        with self.torch.inference_mode(), self.torch.autocast(
            "cuda", dtype=self.torch.bfloat16,
        ):
            state = self.processor.set_image(image)
            points = np.asarray([[point_uv[0], point_uv[1]]], dtype=np.float32)
            labels = np.asarray([1], dtype=np.int64)
            masks, scores, _logits = self.model.predict_inst(
                state,
                point_coords=points,
                point_labels=labels,
                multimask_output=True,
            )
        masks = np.asarray(masks)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        if masks.ndim == 2:
            masks = masks[None]
        if masks.ndim != 3 or len(masks) != len(scores) or not len(scores):
            raise RuntimeError("SAM3 returned no aligned masks and scores")
        u = min(max(int(round(point_uv[0])), 0), masks.shape[2] - 1)
        v = min(max(int(round(point_uv[1])), 0), masks.shape[1] - 1)
        order = sorted(
            range(len(scores)),
            key=lambda index: (
                not bool(np.asarray(masks[index])[v, u] > 0),
                -float(scores[index]),
            ),
        )
        selected = order[0]
        mask = np.asarray(masks[selected]) > 0
        if not bool(mask[v, u]):
            raise RuntimeError("SAM3 masks do not contain the positive point prompt")
        return mask, float(scores[selected])

    def _warmup(self) -> float:
        import numpy as np
        from PIL import Image

        image = np.full((128, 128, 3), 32, dtype=np.uint8)
        image[40:92, 48:84] = [220, 48, 32]
        started = time.perf_counter()
        self._predict(Image.fromarray(image), (64.0, 64.0))
        self.torch.cuda.synchronize()
        return (time.perf_counter() - started) * 1000

    def infer(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if payload.get("schema") != (
            "ora.model-service.segment-point-request.v1"
        ):
            raise ValueError("segmentation request schema mismatch")
        point = payload.get("point_uv")
        if not (
            isinstance(point, list) and len(point) == 2
            and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in point)
        ):
            raise ValueError("point_uv must contain two numbers")
        image = _decode_png(payload.get("image_png_base64"))
        if not (0 <= point[0] < image.width and 0 <= point[1] < image.height):
            raise ValueError("point_uv is outside the image")
        mask, score = self._predict(image, (float(point[0]), float(point[1])))
        self.torch.cuda.synchronize()
        self.requests += 1
        return {
            "schema": self.schema,
            "request_id": _request_id(),
            "model": dict(self.model_identity),
            "score": score,
            "mask_npy_base64": _encode_npy(mask.astype("uint8")),
        }

    def health(self) -> dict[str, Any]:
        return {
            "schema": "ora.model-service.health.v1",
            "ready": True,
            "service": "segmentation",
            "model": dict(self.model_identity),
            "load_ms": self.load_ms,
            "warmup_ms": self.warmup_ms,
            "warmup_complete": True,
            "adapter_sha256": self.adapter_sha256,
            "instance_id": self.instance_id,
            "load_count": self.load_count,
            "started_unix_s": self.started_unix_s,
            "requests": self.requests,
            "uptime_s": time.time() - self.started_unix_s,
            "pid": os.getpid(),
            "capabilities": ["segment_point_v1", "segment_points_v1"],
        }

    def infer_batch(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """One image encoding, independent point prompts, all decoder alternatives.

        The pinned SAM3 predict_inst resets its interactive predictor for each
        call and consumes the image backbone from state. No mask_input or past
        point prompts are fed into a later query.
        """
        import numpy as np

        if payload.get("schema") != "ora.model-service.segment-points-request.v1":
            raise ValueError("batch segmentation request schema mismatch")
        points = payload.get("points_uv")
        if (not isinstance(points, list) or not 1 <= len(points) <= 8
                or any(not isinstance(p, list) or len(p) != 2
                       or any(type(v) is not int for v in p) for p in points)):
            raise ValueError("points_uv must contain 1..8 independent integer [u,v] points")
        image = _decode_png(payload.get("image_png_base64"))
        if any(not (0 <= p[0] < image.width and 0 <= p[1] < image.height) for p in points):
            raise ValueError("batch point is outside the source image")
        queries = []
        with self.torch.inference_mode(), self.torch.autocast("cuda", dtype=self.torch.bfloat16):
            state = self.processor.set_image(image)
            for index, point in enumerate(points):
                query = {"query_index": index, "point_uv": point, "candidates": [], "error": None}
                try:
                    masks, scores, _ = self.model.predict_inst(
                        state, point_coords=np.asarray([point], dtype=np.float32),
                        point_labels=np.asarray([1], dtype=np.int64), multimask_output=True,
                    )
                    masks = np.asarray(masks)
                    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
                    if masks.ndim == 4 and masks.shape[1] == 1:
                        masks = masks[:, 0]
                    if masks.ndim == 2:
                        masks = masks[None]
                    if (masks.ndim != 3 or masks.shape[1:] != (image.height, image.width)
                            or len(masks) != len(scores) or not len(scores) or len(scores) > 3
                            or not np.isfinite(scores).all() or not np.isfinite(masks).all()):
                        raise ValueError("SAM3 returned malformed independent mask alternatives")
                    query["candidates"] = [
                        {"source_index": i, "score": float(score),
                         "mask_npy_base64": _encode_npy((mask > 0).astype("uint8"))}
                        for i, (mask, score) in enumerate(zip(masks, scores))
                    ]
                except (ValueError, RuntimeError) as exc:
                    query["error"] = {"code": "point_inference_failed", "exception_type": type(exc).__name__}
                queries.append(query)
        self.torch.cuda.synchronize()
        self.requests += 1
        return {"schema": "ora.model-service.segment-points.v1",
                "request_id": _request_id(), "model": dict(self.model_identity),
                "queries": queries, "image_encodings": 1}


def contact_input_regime(
    *, status: str, crop_points: int, unique_points: int,
    configured_input_points: int, input_points: int,
) -> dict[str, Any]:
    """One request's Contact-GraspNet input regime, as evidence records it.

    ``padding_factor`` is rows fed per distinct fed point under this adapter
    (1.0, or None without a region); ``upstream_padding_factor`` is what the
    unpatched upstream crop would have fed: ``configured / unique`` below the
    configured size, else 1.0. ``upstream_scene_order`` says whether upstream
    would have kept the scene's point order (any crop not above the
    configured size); this adapter always feeds a seeded permutation.
    """

    fed = min(unique_points, configured_input_points)
    return {
        "schema": CONTACT_INPUT_REGIME_SCHEMA,
        "policy": CONTACT_INPUT_POLICY,
        "status": status,
        "crop_points": int(crop_points),
        "unique_points": int(unique_points),
        "configured_input_points": int(configured_input_points),
        "input_points": int(input_points),
        "padding_factor": (
            round(input_points / fed, 4) if input_points and fed else None
        ),
        "upstream_padding_factor": (
            round(configured_input_points / unique_points, 4)
            if 0 < unique_points < configured_input_points else
            (1.0 if unique_points else None)
        ),
        "point_order": "request_seeded_permutation" if input_points else None,
        "upstream_scene_order": (
            crop_points <= configured_input_points if crop_points else None
        ),
    }


class GraspService:
    schema = "ora.model-service.grasp-proposals.v1"

    def __init__(
        self, *, source_root: Path, checkpoint: Path, config: Path,
    ) -> None:
        started = time.perf_counter()
        checkpoint = _require_file(checkpoint, CONTACT_CHECKPOINT_SHA256)
        config = _require_file(config, CONTACT_CONFIG_SHA256)
        source_root = _require_git_source(source_root, CONTACT_SOURCE_COMMIT)
        package = source_root / "contact_graspnet_pytorch"
        pointnet = source_root / "Pointnet_Pointnet2_pytorch"
        if not package.is_dir() or not pointnet.is_dir():
            raise RuntimeError(f"Contact-GraspNet source is unavailable: {source_root}")
        sys.path[:0] = [str(source_root), str(pointnet)]
        import numpy as np
        import torch
        import yaml
        from contact_graspnet_pytorch.checkpoints import CheckpointIO
        from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator
        from contact_graspnet_pytorch.data import (
            regularize_pc_point_count,
            reject_median_outliers,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("Contact-GraspNet service requires CUDA")
        torch.cuda.set_device(0)
        self.np = np
        self.torch = torch
        model_config = yaml.safe_load(config.read_text(encoding="utf-8"))
        model_config["DATA"]["classes"] = None
        self.estimator = GraspEstimator(model_config)
        CheckpointIO(
            checkpoint_dir=str(checkpoint.parent), model=self.estimator.model,
        ).load(checkpoint.name)
        self.estimator.model.eval()
        self._install_input_policy(
            regularize=regularize_pc_point_count,
            reject_outliers=reject_median_outliers,
        )
        self.model_identity = {
            "family": "contact_graspnet_pytorch",
            "source_commit": CONTACT_SOURCE_COMMIT,
            "checkpoint_sha256": CONTACT_CHECKPOINT_SHA256,
            "config_sha256": CONTACT_CONFIG_SHA256,
            "capx_service_source_commit": CAPX_SOURCE_COMMIT,
            "device": torch.cuda.get_device_name(0),
        }
        self.adapter_sha256 = _sha256(Path(__file__).resolve())
        self.instance_id = "model-instance-" + uuid.uuid4().hex
        self.load_count = 1
        self.load_ms = (time.perf_counter() - started) * 1000
        self.warmup_ms = self._warmup()
        self.started_unix_s = time.time()
        self.requests = 0

    def _install_input_policy(
        self, *, regularize: Any, reject_outliers: Any,
    ) -> None:
        """Shadow the estimator's crop and per-region inference on this instance.

        Upstream ``extract_3d_cam_boxes`` pads a sparse crop to the configured
        input size with appended duplicates, in scene order, and
        ``predict_grasps`` resamples to that same fixed size. The PointNet++
        backbone accepts any point count, so the per-request input size is the
        distinct crop count. The model configuration dict is shared with the
        network and is never mutated.
        """

        estimator = self.estimator
        self.configured_input_points = int(estimator._num_input_points)
        self._regularize = regularize
        self._reject_outliers = reject_outliers
        self._regimes: list[dict[str, Any]] = []
        estimator.extract_3d_cam_boxes = self._extract_unique_regions
        estimator.predict_grasps = self._predict_region

    def _extract_unique_regions(
        self, full_pc: Any, pc_segments: dict[Any, Any],
        min_size: float = 0.3, max_size: float = 0.6,
    ) -> tuple[dict[Any, Any], dict[Any, Any]]:
        """Upstream's target-centred cube crop, without padding or scene order."""

        np = self.np
        data = self.estimator._contact_grasp_cfg["DATA"]
        configured = self.configured_input_points
        regions: dict[Any, Any] = {}
        centers: dict[Any, Any] = {}
        for key in pc_segments:
            # Upstream filters the segment in place; grasp filtering reuses it.
            pc_segments[key] = self._reject_outliers(
                pc_segments[key], m=0.4, z_only=False,
            )
            if not np.any(pc_segments[key]):
                continue
            upper = np.max(pc_segments[key][:, :3], axis=0)
            lower = np.min(pc_segments[key][:, :3], axis=0)
            center = lower + (upper - lower) / 2
            size = np.minimum(
                np.maximum(np.max(upper - lower) * 2, min_size), max_size,
            )
            crop = full_pc[
                np.all(full_pc > (center - size / 2), axis=1)
                & np.all(full_pc < (center + size / 2), axis=1)
            ]
            if not np.any(crop):
                continue
            _, first = np.unique(crop, axis=0, return_index=True)
            unique = crop[np.sort(first)]
            if len(unique) > configured:
                # Exactly upstream's downsample (random subset, random order).
                region = self._regularize(
                    unique, configured,
                    use_farthest_point=data["use_farthest_point"],
                )
            else:
                # Upstream's sampler at the distinct count: a permutation.
                region = unique[np.random.choice(
                    len(unique), size=len(unique), replace=False,
                )]
            usable = len(unique) >= CONTACT_MIN_REGION_UNIQUE_POINTS
            self._regimes.append(contact_input_regime(
                status="region" if usable else "too_few_unique_points",
                crop_points=len(crop), unique_points=len(unique),
                configured_input_points=configured,
                input_points=len(region) if usable else 0,
            ))
            if usable:
                regions[key] = region
                centers[key] = center
        return regions, centers

    def _predict_region(self, pc: Any, *args: Any, **kwargs: Any) -> Any:
        """Run upstream ``predict_grasps`` at this region's own point count."""

        estimator = self.estimator
        estimator._num_input_points = int(len(pc))
        try:
            return type(estimator).predict_grasps(estimator, pc, *args, **kwargs)
        finally:
            estimator._num_input_points = self.configured_input_points

    def _predict(
        self, scene: Any, target: Any, *, forward_passes: int, seed: int,
    ) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        np = self.np
        self._regimes = []
        random.seed(seed)
        np.random.seed(seed)
        self.torch.manual_seed(seed)
        self.torch.cuda.manual_seed_all(seed)
        with self.torch.inference_mode():
            grasps, scores, contacts, openings = self.estimator.predict_scene_grasps(
                scene,
                pc_segments={1: target},
                local_regions=True,
                filter_grasps=True,
                forward_passes=forward_passes,
            )
        regime = self._regimes[0] if self._regimes else contact_input_regime(
            status="no_region", crop_points=0, unique_points=0,
            configured_input_points=self.configured_input_points,
            input_points=0,
        )
        pose_values = np.asarray(grasps.get(1, []), dtype=np.float32)
        score_values = np.asarray(scores.get(1, []), dtype=np.float32).reshape(-1)
        contact_values = np.asarray(contacts.get(1, []), dtype=np.float32)
        opening_values = np.asarray(openings.get(1, []), dtype=np.float32).reshape(-1)
        if not len(pose_values):
            return (
                np.empty((0, 4, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                regime,
            )
        pose_values = pose_values.reshape(-1, 4, 4)
        contact_values = contact_values.reshape(-1, 3)
        if not (
            len(pose_values) == len(score_values)
            == len(contact_values) == len(opening_values)
        ):
            raise RuntimeError("Contact-GraspNet returned unaligned candidate arrays")
        order = np.argsort(-score_values, kind="stable")
        return (
            pose_values[order], score_values[order],
            contact_values[order], opening_values[order], regime,
        )

    def _warmup(self) -> float:
        np = self.np
        rng = np.random.default_rng(7)
        target = np.column_stack([
            rng.uniform(-0.035, 0.035, 4096),
            rng.uniform(-0.025, 0.025, 4096),
            rng.uniform(0.68, 0.75, 4096),
        ]).astype(np.float32)
        plane = np.column_stack([
            rng.uniform(-0.25, 0.25, 12000),
            rng.uniform(-0.25, 0.25, 12000),
            np.full(12000, 0.78),
        ]).astype(np.float32)
        started = time.perf_counter()
        self._predict(
            np.concatenate([target, plane]), target,
            forward_passes=1, seed=7,
        )
        self.torch.cuda.synchronize()
        return (time.perf_counter() - started) * 1000

    def infer(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        np = self.np
        if payload.get("schema") != (
            "ora.model-service.grasp-proposals-request.v1"
        ):
            raise ValueError("grasp request schema mismatch")
        scene = np.asarray(
            _decode_npy(payload.get("scene_points_npy_base64")), dtype=np.float32,
        )
        target = np.asarray(
            _decode_npy(payload.get("target_points_npy_base64")), dtype=np.float32,
        )
        if scene.ndim != 2 or scene.shape[1] != 3 or len(scene) < 16:
            raise ValueError("scene point cloud must be Nx3 with at least 16 points")
        if target.ndim != 2 or target.shape[1] != 3 or len(target) < 16:
            raise ValueError("target point cloud must be Nx3 with at least 16 points")
        maximum = payload.get("max_candidates")
        forward_passes = payload.get("forward_passes")
        seed = payload.get("seed")
        if isinstance(maximum, bool) or not isinstance(maximum, int) or not 1 <= maximum <= 32:
            raise ValueError("max_candidates must be an integer in [1,32]")
        if forward_passes != 1:
            raise ValueError("this service profile fixes forward_passes=1")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("seed must be one uint32 integer")
        poses, scores, contacts, openings, regime = self._predict(
            scene, target, forward_passes=forward_passes, seed=seed,
        )
        poses = poses[:maximum]
        scores = scores[:maximum]
        contacts = contacts[:maximum]
        openings = openings[:maximum]
        self.torch.cuda.synchronize()
        self.requests += 1
        return {
            "schema": self.schema,
            "request_id": _request_id(),
            "model": dict(self.model_identity),
            "poses_camera_npy_base64": _encode_npy(poses),
            "scores_npy_base64": _encode_npy(scores),
            "contact_points_camera_npy_base64": _encode_npy(contacts),
            "openings_m_npy_base64": _encode_npy(openings),
            "input_regime": regime,
        }

    def health(self) -> dict[str, Any]:
        return {
            "schema": "ora.model-service.health.v1",
            "ready": True,
            "service": "grasp_proposals",
            "model": dict(self.model_identity),
            "capabilities": ["grasp_proposals_v1", "distinct_shuffled_crop_input_v1"],
            "input_policy": {
                "schema": CONTACT_INPUT_POLICY_SCHEMA,
                "policy": CONTACT_INPUT_POLICY,
                "configured_input_points": self.configured_input_points,
                "duplicate_padding": False,
                "point_order": "request_seeded_permutation",
                "minimum_region_unique_points": CONTACT_MIN_REGION_UNIQUE_POINTS,
            },
            "load_ms": self.load_ms,
            "warmup_ms": self.warmup_ms,
            "warmup_complete": True,
            "adapter_sha256": self.adapter_sha256,
            "instance_id": self.instance_id,
            "load_count": self.load_count,
            "started_unix_s": self.started_unix_s,
            "requests": self.requests,
            "uptime_s": time.time() - self.started_unix_s,
            "pid": os.getpid(),
        }


def _serve(service: Any, *, host: str, port: int) -> None:
    import uvicorn
    from fastapi import FastAPI, HTTPException

    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError("model services must bind to loopback")
    app = FastAPI(title=f"Robot Harness {service.health()['service']} service")
    lock = asyncio.Lock()

    @app.get("/healthz")
    async def health() -> dict[str, Any]:
        return service.health()

    async def execute(method: str, payload: dict[str, Any]) -> dict[str, Any]:
        received = time.perf_counter()
        try:
            async with lock:
                acquired = time.perf_counter()
                result = await asyncio.to_thread(getattr(service, method), payload)
                finished = time.perf_counter()
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        result["metrics"] = {
            "queue_ms": (acquired - received) * 1000,
            "inference_ms": (finished - acquired) * 1000,
            "total_ms": (finished - received) * 1000,
        }
        if method == "infer_batch":
            result["service_identity"] = _response_service_identity(service)
        return result

    async def run_inference(payload: dict[str, Any]) -> dict[str, Any]:
        return await execute("infer", payload)

    async def run_batch(payload: dict[str, Any]) -> dict[str, Any]:
        return await execute("infer_batch", payload)

    if isinstance(service, SegmentService):
        app.add_api_route(
            "/v1/segment-point", run_inference, methods=["POST"],
        )
        app.add_api_route("/v1/segment-points", run_batch, methods=["POST"])
    else:
        app.add_api_route(
            "/v1/grasp-proposals", run_inference, methods=["POST"],
        )
    logging.info(
        "service ready after load %.1f ms and warmup %.1f ms: %s",
        service.load_ms, service.warmup_ms,
        json.dumps(service.model_identity, sort_keys=True),
    )
    uvicorn.run(app, host=host, port=port, workers=1, access_log=False)


def main(
    argv: list[str] | None = None, *, adapter_path: Path | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="service", required=True)
    segment = subparsers.add_parser("segmentation")
    segment.add_argument("--source-root", type=Path, required=True)
    segment.add_argument("--checkpoint", type=Path, required=True)
    segment.add_argument("--host", default="127.0.0.1")
    segment.add_argument("--port", type=int, default=8114)
    grasp = subparsers.add_parser("grasp")
    grasp.add_argument("--source-root", type=Path, required=True)
    grasp.add_argument("--checkpoint", type=Path, required=True)
    grasp.add_argument("--config", type=Path, required=True)
    grasp.add_argument("--host", default="127.0.0.1")
    grasp.add_argument("--port", type=int, default=8115)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    if args.service == "segmentation":
        service = SegmentService(
            source_root=args.source_root, checkpoint=args.checkpoint,
        )
    else:
        service = GraspService(
            source_root=args.source_root,
            checkpoint=args.checkpoint,
            config=args.config,
        )
    if adapter_path is not None:
        service.adapter_sha256 = _sha256(adapter_path.resolve())
    _serve(service, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
