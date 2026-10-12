"""Placement fixtures shared by the placement and geometry tests.

A deterministic point-mask inference (``SelectedMask``), the hashing and JSON
helpers, the prompt/manual fixture paths and ``materialize_segmented_target``,
which writes one canonical single-point segmented-target record (the format
``grasp.load_segmented_target`` reads) for tests that need a ``target_ref``
without the batch service. The LIBERO ``place.plan`` and
``place.verify`` adapter tests that used to live here ran only on the retired
``libero_placement_support`` surfaces and were removed with them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANUAL = ROOT / "tests/data/prompts/tool-schema-delivery-v1.md"
PROMPT = ROOT / "tests/data/prompts/episode-task-v1.md"
PROJECT = ROOT / "ora/policy/environments/synthetic/boundary-v1.md"
POSE = [0.0, 0.0, 0.50, 0.0, 0.0, 0.0, 1.0]
RIM = [[54, 48], [52, 52], [48, 54], [44, 52], [42, 48], [44, 44], [48, 42], [52, 44]]


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


class SelectedMask:
    """A selected public-image region, never an environment segmentation."""

    backend_id = "placement_public_pixel_fixture"
    backend_version = "v1"

    def __init__(self, mask: np.ndarray | None = None):
        self.mask = mask

    def configuration(self):
        return {"mask_sha256": None if self.mask is None else
                hashlib.sha256(self.mask.tobytes()).hexdigest()}

    def segment(self, *, rgb_u8, depth_m, point_uv):
        assert rgb_u8.shape[:2] == depth_m.shape == self.mask.shape
        assert self.mask[point_uv[1], point_uv[0]]
        return self.mask.copy(), {"source": "selected_public_pixels"}


from ora.agent.tools.grasp import (  # noqa: E402 - fixture writer over the canonical record helpers
    LOCALIZATION_METHOD, TARGET_SCHEMA, PointSegmentationBackend, _artifact, _atomic_json, _atomic_npy,
    _atomic_png, _canonical_hash, _mask_depth_localization, _mask_summary, _materialize_target_index,
    _render_mask_overlay, _segmentation_source_identity, _segmented_target_root, _verify_artifact,
)


def materialize_segmented_target(
    *, artifact_root: Path, observation_id: str, camera: str,
    rgb_path: Path, depth_path: Path, calibration_path: Path,
    point_uv: tuple[int, int], label: str | None,
    backend: PointSegmentationBackend,
) -> tuple[dict[str, Any], Path]:
    import numpy as np
    from PIL import Image

    source = _segmentation_source_identity(
        observation_id=observation_id, camera=camera,
        rgb_path=rgb_path, depth_path=depth_path,
        calibration_path=calibration_path,
    )
    backend_record = {
        "id": backend.backend_id,
        "version": backend.backend_version,
        "config_sha256": _canonical_hash(dict(backend.configuration())),
    }
    identity = {
        "observation_id": observation_id,
        "camera": camera,
        "point_uv": list(point_uv),
        "label": label,
        "source": source,
        "backend": backend_record,
        "localization_method": LOCALIZATION_METHOD,
    }
    target_ref = "target-" + _canonical_hash(identity)[:24]
    target_root = _segmented_target_root(
        artifact_root=artifact_root, observation_id=observation_id,
        camera=camera, target_ref=target_ref,
    )
    manifest_path = target_root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != TARGET_SCHEMA or manifest.get("identity") != identity:
            raise ValueError("segmented target identity drift")
        _verify_artifact(target_root, manifest["private_mask"])
        overlay = _verify_artifact(target_root, manifest["overlay"])
        _materialize_target_index(
            artifact_root=artifact_root, target_root=target_root,
            manifest_path=manifest_path, observation_id=observation_id,
            camera=camera, target_ref=target_ref,
        )
        return manifest, overlay

    rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
    depth = np.load(depth_path, allow_pickle=False).astype(np.float64)
    mask, raw_diagnostics = backend.segment(
        rgb_u8=rgb, depth_m=depth, point_uv=point_uv,
    )
    diagnostics = dict(raw_diagnostics)
    response_service_identity = diagnostics.pop("_service_identity", None)
    summary = _mask_summary(mask, depth)
    localization = _mask_depth_localization(
        mask=mask, depth_m=depth, calibration_path=calibration_path,
        extended_frame=True,
    )
    target_root.mkdir(parents=True, exist_ok=True)
    mask_path = target_root / "mask.npy"
    overlay_path = target_root / "overlay.png"
    _atomic_npy(mask_path, np.asarray(mask, dtype=np.uint8))
    overlay = _render_mask_overlay(
        rgb_u8=rgb, mask=mask, point_uv=point_uv, summary=summary,
    )
    _atomic_png(overlay_path, overlay)
    mask_path.chmod(0o444)
    overlay_path.chmod(0o444)
    manifest = {
        "schema": TARGET_SCHEMA,
        "target_ref": target_ref,
        "identity": identity,
        "prompt": {"point_uv": list(point_uv), "label": label},
        "backend": backend_record,
        "summary": summary,
        "localization": localization,
        "diagnostics": dict(diagnostics),
        "response_service_identity": response_service_identity,
        "private_mask": _artifact(mask_path, target_root),
        "overlay": _artifact(overlay_path, target_root),
    }
    _atomic_json(manifest_path, manifest)
    manifest_path.chmod(0o444)
    _materialize_target_index(
        artifact_root=artifact_root, target_root=target_root,
        manifest_path=manifest_path, observation_id=observation_id,
        camera=camera, target_ref=target_ref,
    )
    return manifest, overlay_path
