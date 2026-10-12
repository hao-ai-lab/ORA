"""Compact decision evidence for opt-in perception surfaces; no action authority."""
from __future__ import annotations

from . import tool_names

import base64
import hashlib
import io
from pathlib import Path
from typing import Any, Mapping

CAPABILITY = "public_rgbd_perception_feedback_v2"


def enabled(surface: Any) -> bool:
    return surface is not None and surface.supports(CAPABILITY)


def image_label(index: int, role: str, *, camera: str = "", candidate_id: str = "",
                target_ref: str = "", step: int | None = None) -> dict[str, Any]:
    return dict(image_index=index, role=role, camera=camera, candidate_id=candidate_id,
                target_ref=target_ref, observation_step=step)


def grasp_review(result: dict[str, Any], manifest: Mapping[str, Any], overlay: Path) -> None:
    """Deliver the already source-bound public scene review, when available."""
    result["image_mapping"] = [image_label(0, "local_contact_candidates")]
    result["scene_review"] = {"status": "unavailable", "reason": "public_scene_review_unavailable",
        "scope": "observed_public_rgbd_only;unobserved_space_unknown;axes_and_ids_match_local_candidates"}
    review = manifest.get("review_visualization", {})
    if review.get("status") != "available":
        return
    artifact = review["artifact"]
    path = (overlay.parent / artifact["path"]).resolve()
    try:
        if not path.is_relative_to(overlay.parent.resolve()):
            raise ValueError("review path escaped query")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != artifact["sha256"]:
            raise ValueError("review image content drift")
    except (OSError, ValueError):
        result["scene_review"]["reason"] = "public_scene_review_integrity_unavailable"
        return
    result["images_base64"].append(base64.b64encode(data).decode("ascii"))
    result["image_mapping"].append(image_label(1, "global_public_scene_candidates"))
    result["scene_review"].update(status="available", reason="public_source_bound_review")


def threshold_gaps(comparison: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Signed acceptance margins; positive always satisfies this one threshold."""
    geometry, surface, thresholds = (comparison[key] for key in
                                   ("geometry", "surface_comparison", "thresholds"))
    specs = [
        ("motion_excitation", geometry.get("expected_target_motion_m"), thresholds["minimum_expected_target_motion_m"], "minimum", "m"),
        ("rigid_surface_error", surface.get("rigid_p90_error_m"), thresholds["surface_noise_radius_m"], "maximum", "m"),
        ("model_separation", surface.get("model_separation_m"), thresholds["minimum_model_separation_m"], "minimum", "m"),
        ("overlap_before", surface.get("rigid_overlap_before"), thresholds["minimum_rigid_overlap_fraction"], "minimum", "fraction"),
        ("overlap_current", surface.get("rigid_overlap_current"), thresholds["minimum_rigid_overlap_fraction"], "minimum", "fraction"),
        ("negative_view_change", surface.get("view_direction_change_rad"), thresholds["maximum_negative_view_direction_change_rad"], "maximum", "rad"),
    ]
    return [dict(name=name, measured=value, threshold=threshold, criterion=criterion, unit=unit,
                 acceptance_margin=None if value is None else
                 (value - threshold if criterion == "minimum" else threshold - value))
            for name, value, threshold, criterion, unit in specs]


def retention_pairs(host: Any, result: dict[str, Any]) -> None:
    import numpy as np
    from PIL import Image, ImageDraw, ImageOps
    from .grasp import load_segmented_target

    result["pair_feedback"], result["image_mapping"], result["images_base64"] = [], [], []
    for pair_index, pair in enumerate(result.get("comparisons", [result])):
        entry = {"pair_id": f"V{pair_index}", "status": pair["status"], "reason": pair["reason"],
            "before_target_ref": pair["before_target_ref"], "current_target_ref": pair["current_target_ref"],
            "before_camera": pair["source_cameras"]["before"], "current_camera": pair["source_cameras"]["current"],
            "threshold_gaps": threshold_gaps(pair), "image_status": "available"}
        panels = []
        try:
            for label in ("before", "current"):
                ref = pair[f"{label}_target_ref"]
                _target, mask_path, overlay = load_segmented_target(artifact_root=host.artifact_root, target_ref=ref)
                with Image.open(overlay) as source:
                    panel = source.convert("RGB")
                mask = np.load(mask_path, allow_pickle=False).astype(bool)
                ys, xs = np.nonzero(mask)
                if not len(xs) or mask.shape != (panel.height, panel.width):
                    raise ValueError("paired overlay and mask differ")
                crop = panel.crop((max(0, int(xs.min()) - 12), max(0, int(ys.min()) - 12),
                                   min(panel.width, int(xs.max()) + 13), min(panel.height, int(ys.max()) + 13)))
                canvas = Image.new("RGB", (480, 306), "white")
                for column, view in enumerate((panel, crop)):
                    # The crop enlarges saved pixels, never sensor resolution.
                    tile = ImageOps.contain(view, (236, 236), Image.Resampling.NEAREST)
                    canvas.paste(tile, (column * 240 + 2, 66))
                draw = ImageDraw.Draw(canvas)
                step = pair["baseline"][f"{label}_state_step"]
                draw.text((8, 5), f"V{pair_index} {label.upper()} / step {step} / {pair['source_cameras'][label]}", fill="black")
                draw.text((8, 22), ref, fill="black")
                draw.text((8, 46), "SOURCE + MASK", fill="black")
                draw.text((244, 46), "MASK CROP / same saved pixels", fill="black")
                panels.append(canvas)
            sheet = Image.new("RGB", (960, 334), "white")
            for index, panel in enumerate(panels):
                sheet.paste(panel, (index * 480, 28))
            ImageDraw.Draw(sheet).text((8, 8), "PUBLIC PAIRED MASKS / semantic correspondence chosen by Agent / no attachment verdict", fill="black")
            buffer = io.BytesIO()
            sheet.save(buffer, format="PNG")
            data = buffer.getvalue()
            path = host.artifact_root / "perception-feedback" / (hashlib.sha256(data).hexdigest() + ".png")
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                with path.open("xb") as stream:
                    stream.write(data)
            elif path.is_symlink() or path.read_bytes() != data:
                raise ValueError("paired image integrity mismatch")
            image_index = len(result["images_base64"])
            result["images_base64"].append(base64.b64encode(data).decode("ascii"))
            result["image_mapping"].append(image_label(image_index, f"{entry['pair_id']}_before_current_masks"))
        except (OSError, ValueError, KeyError):
            # Numeric evidence remains available if optional rendering fails.
            entry["image_status"] = "unavailable"
        result["pair_feedback"].append(entry)


#: The tools that carry a measurement contract, named canonically.
#:
#: The membership gate below is this set and the lookup is this set, because
#: they were two literals for one fact and they drifted the moment the runtime
#: started resolving spellings: the gate was canonicalised, the table's
#: ``motion.probe`` key was not, and a `motion.probe` call passed the gate and
#: raised `KeyError` on the line after it -- reaching a policy as a
#: non-retryable `internal_tool_error`.
MEASUREMENT_CONTRACTS = frozenset({"observe", "mark_point"})


def enrich(host: Any, name: str, arguments: Mapping[str, Any], result: dict[str, Any]) -> None:
    # Every branch below names a tool. A surface may advertise either
    # spelling, and an unmatched branch skips enrichment silently.
    name = tool_names.canonical(name)
    if not enabled(host.surface):
        return
    if name == "vision_ground":
        for candidate in result["candidates"]:
            candidate["candidate_id"] = f"T{candidate['rank']}"
    if name == "grasp_verify_retention":
        retention_pairs(host, result)
    if name in MEASUREMENT_CONTRACTS:
        contracts = {
            "observe": "RGB pixels: (u,v), top-left origin; world geometry: metres; each image is the named saved view, not a new capture. Missing/occluded pixels provide no evidence.",
            "mark_point": "xyz_m is visible world geometry in metres; residuals measure projection agreement, not semantic identity. Pending marks need complementary views; moved-camera pixels are observation-specific.",
        }
        assert contracts.keys() == MEASUREMENT_CONTRACTS
        result["measurement_contract"] = contracts[name]
    if "images_base64" in result and "image_mapping" not in result:
        views = result.get("views", [])
        candidates = result.get("candidates", [])
        mapping = []
        for index, _ in enumerate(result["images_base64"]):
            if name in {"vision_segment", "vision_ground"}:
                candidate = candidates[index - 1] if 0 < index <= len(candidates) else {}
                item = image_label(index, "source_overview" if index == 0 else "candidate_source_crop",
                    camera=arguments.get("camera", ""), candidate_id=candidate.get("candidate_id", ""),
                    target_ref=candidate.get("target_ref", ""), step=arguments.get("step"))
            else:
                item = image_label(index, "saved_public_view" if name == "observe" else "public_geometry_review",
                    camera=views[index] if index < len(views) else "", step=arguments.get("step"))
            mapping.append(item)
        result["image_mapping"] = mapping
