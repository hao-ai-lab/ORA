"""Independent public-image point queries with a per-point, inspectable mask ledger.

Every queried pixel owns its own bounded candidate list, so asking about one
point can never empty another point's answer. Raw masks remain private
artifacts, scores order proposals inside one point only, IoU still removes near
duplicates across points, and the Agent retains semantic selection authority.

This is the *second* published contract for one shared runtime path; the first
pooled `max_candidates` across every point of a call and is declared by no kept
surface. A surface declaring this module's `CAPABILITY` gets the per-point
ledger below; `select_contract` is the only place that choice is made.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

from . import grasp, grounding_preview


CAPABILITY = "public_rgbd_batch_point_segmentation_v2"
SCHEMA = "ora.point-segmentation-batch.v2"
RENDERER = "point-batch-per-point-inspection.v2"
SELECTION_METHOD = "per_point_reserved_then_rank_major_v1"
MAX_POINTS = 8
MAX_CANDIDATES = 3
# One returned mask per queried point is reserved before any surplus is shared,
# so this ceiling must never fall below MAX_POINTS. The remaining slots bound
# the delivered result against the eager large-result threshold.
MAX_RETURNED_CANDIDATES = 12
# Enlarged crops are the expensive half of the host cost and the only part that
# adds inline images, so they are capped separately from the mask ledger.
MAX_PREVIEW_IMAGES = 5
MASK_IOU_THRESHOLD = .9
POINT_COLOURS = ((48, 220, 96), (64, 180, 255), (255, 190, 48), (255, 96, 160),
                 (170, 120, 255), (64, 232, 232), (255, 132, 64), (200, 200, 80))
UNANSWERED_COLOUR = (168, 168, 168)


def select_contract(surface: Any) -> Any:
    """Return the batch implementation the resolved surface declares, or None.

    Capability, never profile id: a new profile inherits exactly the contract
    its hash-bound manifest names.
    """

    return sys.modules[__name__] if surface.supports(CAPABILITY) else None


def validate_points(points: Any, *, width: int, height: int) -> None:
    if (not isinstance(points, list) or not 1 <= len(points) <= MAX_POINTS
            or any(not isinstance(p, list) or len(p) != 2
                   or any(type(v) is not int for v in p) for p in points)):
        raise ValueError("points_uv must contain 1..8 independent integer [u,v] points")
    if any(not (0 <= p[0] < width and 0 <= p[1] < height) for p in points):
        raise ValueError("points_uv must lie inside the bound source image")


def _normalize(raw: list[dict[str, Any]], *, points_uv: list[list[int]],
               shape: tuple[int, int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep the prompt-connected component of every mask that contains its point."""
    import cv2
    import numpy as np

    normalized, rejected, seen = [], [], set()
    per_query = [0] * len(points_uv)
    for item in raw:
        q, c, score = item.get("query_index"), item.get("source_index"), item.get("backend_score")
        if (type(q) is not int or not 0 <= q < len(points_uv) or type(c) is not int or c < 0
                or (q, c) in seen or item.get("point_uv") != points_uv[q]
                or type(score) not in (int, float) or not math.isfinite(score)):
            raise ValueError("batch candidate lost its query identity or finite score")
        seen.add((q, c))
        per_query[q] += 1
        if per_query[q] > MAX_CANDIDATES:
            raise ValueError("batch backend exceeded its per-query mask bound")
        mask = np.asarray(item["mask"])
        if mask.shape != shape or not np.isfinite(mask).all():
            raise ValueError("batch candidate mask is not aligned with the source image")
        mask = mask.astype(bool)
        u, v = points_uv[q]
        if not mask[v, u]:
            rejected.append({"query_index": q, "source_index": c, "reason": "mask_excludes_positive_point"})
            continue
        count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        component = labels == int(labels[v, u])
        raw_count, kept_count = int(mask.sum()), int(component.sum())
        normalized.append({**item, "mask": component, "backend_score": float(score),
            "normalization": {"method": "positive_prompt_8_connected_component_v1",
                "raw_component_count": count - 1, "raw_pixel_count": raw_count,
                "prompt_component_pixel_count": kept_count, "retained_fraction": kept_count / raw_count},
            "sources": [{"query_index": q, "source_index": c}]})
    return normalized, rejected


def _deduplicate(normalized: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge near-identical masks so one object costs one artifact, not one per point."""
    import numpy as np

    normalized.sort(key=lambda item: (-item["backend_score"], item["query_index"], item["source_index"]))
    distinct: list[dict[str, Any]] = []
    for item in normalized:
        for representative in distinct:
            intersection = int(np.count_nonzero(item["mask"] & representative["mask"]))
            union = int(np.count_nonzero(item["mask"] | representative["mask"]))
            if intersection / union >= MASK_IOU_THRESHOLD:
                representative["sources"].extend(item["sources"])
                break
        else:
            distinct.append(item)
    return distinct


def select_masks(raw: list[dict[str, Any]], *, points_uv: list[list[int]],
                 shape: tuple[int, int], max_candidates: int,
                 max_returned: int = MAX_RETURNED_CANDIDATES,
                 max_previews: int = MAX_PREVIEW_IMAGES,
                 ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Give every point its own bounded mask list under one shared delivery cap.

    Rank 0 of every point is admitted before any point's rank 1, so a point that
    produced a usable mask always keeps one. Only the surplus competes, and each
    point reports exactly how many of its own masks it lost and to which limit.
    """
    validate_points(points_uv, width=shape[1], height=shape[0])
    if type(max_candidates) is not int or not 1 <= max_candidates <= MAX_CANDIDATES:
        raise ValueError("max_candidates must be an integer in 1..3")
    if (type(max_returned) is not int
            or not len(points_uv) <= max_returned <= MAX_RETURNED_CANDIDATES):
        raise ValueError("max_returned must reserve one mask for every queried point")
    if type(max_previews) is not int or not 1 <= max_previews <= MAX_PREVIEW_IMAGES:
        raise ValueError("max_previews must be an integer in 1..5")
    if not isinstance(raw, list) or len(raw) > MAX_CANDIDATES * len(points_uv):
        raise ValueError("batch backend exceeded its per-query mask bound")

    normalized, rejected = _normalize(raw, points_uv=points_uv, shape=shape)
    distinct = _deduplicate(normalized)
    owned = [[item for item in distinct
              if any(source["query_index"] == q for source in item["sources"])]
             for q in range(len(points_uv))]
    beyond = [max(0, len(masks) - max_candidates) for masks in owned]
    owned = [masks[:max_candidates] for masks in owned]

    admitted: list[dict[str, Any]] = []
    keys: set[int] = set()
    for rank in range(max_candidates):
        tier = sorted(((masks[rank], q) for q, masks in enumerate(owned) if len(masks) > rank),
                      key=lambda pair: (-pair[0]["backend_score"], pair[1]))
        for item, _query in tier:
            if id(item) in keys or len(admitted) >= max_returned:
                continue
            keys.add(id(item))
            admitted.append(item)

    kept = [[item for item in masks if id(item) in keys] for masks in owned]
    for masks, survivors in zip(owned, kept):
        if masks and not survivors:
            raise ValueError("per-point mask reserve failed under the delivery cap")
    previews: list[dict[str, Any]] = []
    preview_keys: set[int] = set()
    for survivors in kept:
        if not survivors or id(survivors[0]) in preview_keys or len(previews) >= max_previews:
            continue
        preview_keys.add(id(survivors[0]))
        previews.append(survivors[0])
    ordered = previews + [item for item in admitted if id(item) not in preview_keys]
    position = {id(item): index for index, item in enumerate(ordered)}
    preview_index = {id(item): index + 1 for index, item in enumerate(previews)}

    queries = []
    for query, (masks, survivors) in enumerate(zip(owned, kept)):
        queries.append({"query_index": query, "point_uv": points_uv[query],
            "candidate_ids": [f"T{position[id(item)]}" for item in survivors],
            "omitted_over_point_limit": beyond[query],
            "omitted_over_call_cap": len(masks) - len(survivors),
            "preview_image_index": preview_index.get(id(survivors[0])) if survivors else None,
            "previews_omitted": int(bool(survivors) and id(survivors[0]) not in preview_index)})
    return ordered, {
        "method": SELECTION_METHOD, "mask_iou_threshold": MASK_IOU_THRESHOLD,
        "max_candidates_per_point": max_candidates, "max_returned_candidates": max_returned,
        "raw_candidate_count": len(raw), "usable_candidate_count": len(normalized),
        "distinct_candidate_count": len(distinct), "duplicate_count": len(normalized) - len(distinct),
        "returned_count": len(ordered),
        "omitted_over_point_limit_total": sum(beyond),
        "omitted_over_call_cap_total": sum(item["omitted_over_call_cap"] for item in queries),
        "rejected_candidates": rejected, "semantic_selection": "agent",
        "queries": queries, "preview_count": len(previews),
    }


def _write(path: Path, value: Any) -> None:
    data = (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode()
    if path.exists():
        if path.is_symlink() or path.read_bytes() != data:
            raise ValueError("batch evidence identity drift")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
    path.chmod(0o444)


def _render_overview(image: Any, *, points_uv: list[list[int]],
                     best: list[Mapping[str, Any] | None]) -> Any:
    """Draw every prompt cross in its own colour with that point's leading mask.

    Attribution is per point rather than per pool rank, so a point that received
    no mask is still visible as an answered-but-empty prompt.
    """
    import cv2
    import numpy as np
    from PIL import Image

    canvas = np.asarray(image, dtype=np.uint8).astype(np.float64)
    for index, item in enumerate(best):
        if item is None:
            continue
        mask = np.asarray(item["mask"], dtype=bool)
        colour = np.asarray(POINT_COLOURS[index % len(POINT_COLOURS)], dtype=np.float64)
        canvas[mask] = np.clip(.60 * canvas[mask] + .40 * colour, 0, 255)
    rendered = canvas.astype(np.uint8)
    for index, item in enumerate(best):
        if item is None:
            continue
        colour = tuple(int(value) for value in POINT_COLOURS[index % len(POINT_COLOURS)])
        contours, _hierarchy = cv2.findContours(
            np.asarray(item["mask"], dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rendered, contours, -1, colour, 1)
        x0, y0, x1, y1 = item["bbox_xyxy"]
        cv2.rectangle(rendered, (x0, y0), (x1 - 1, y1 - 1), colour, 1)
    for index, (u, v) in enumerate(points_uv):
        item = best[index]
        colour = (tuple(int(value) for value in POINT_COLOURS[index % len(POINT_COLOURS)])
                  if item is not None else UNANSWERED_COLOUR)
        label = f"Q{index}:{item['candidate_id']}" if item is not None else f"Q{index}:none"
        cv2.drawMarker(rendered, (u, v), colour, cv2.MARKER_CROSS, 9, 1)
        cv2.putText(rendered, label, (u + 4, max(10, v - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, .35, colour, 1, cv2.LINE_AA)
    return Image.fromarray(rendered, mode="RGB")


def _materialize_target(*, artifact_root: Path, source: Mapping[str, Any], backend: Mapping[str, Any],
                        camera: str, observation_id: str, rgb: Any, depth: Any, calibration_path: Path,
                        item: Mapping[str, Any], position: int) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np
    mask = item["mask"]
    identity = {"observation_id": observation_id, "camera": camera,
        "prompt_kind": "independent_point_candidate", "point_uv": item["point_uv"],
        "mask_sha256": hashlib.sha256(np.asarray(mask, dtype=np.uint8).tobytes(order="C")).hexdigest(),
        "source": dict(source), "backend": dict(backend), "localization_method": grasp.LOCALIZATION_METHOD}
    ref = "target-" + grasp._canonical_hash(identity)[:24]
    root = grasp._segmented_target_root(artifact_root=artifact_root, observation_id=observation_id,
                                       camera=camera, target_ref=ref)
    summary = grasp._mask_summary(mask, depth)
    localization = grasp._mask_depth_localization(
        mask=mask, depth_m=depth, calibration_path=calibration_path,
        extended_frame=True,
    )
    path = root / "manifest.json"
    if path.exists():
        manifest, _, _ = grasp.load_segmented_target(artifact_root=artifact_root, target_ref=ref)
        if manifest["identity"] != identity:
            raise ValueError("batch target source identity drift")
    else:
        root.mkdir(parents=True, exist_ok=True)
        grasp._atomic_npy(root / "mask.npy", np.asarray(mask, dtype=np.uint8))
        # Target artifacts are position-independent so reuse in another query pool
        # cannot silently display an obsolete candidate number.
        overlay = grasp._render_mask_overlay(rgb_u8=rgb, mask=mask,
                    point_uv=tuple(item["point_uv"]), summary=summary)
        grasp._atomic_png(root / "overlay.png", overlay)
        for file in (root / "mask.npy", root / "overlay.png"):
            file.chmod(0o444)
        manifest = {"schema": grasp.TARGET_SCHEMA, "target_ref": ref, "identity": identity,
            "prompt": {"kind": "independent_point_candidate", "point_uv": item["point_uv"]},
            "backend": dict(backend), "summary": summary, "localization": localization,
            "diagnostics": {"candidate_metadata_scope": "point_batch_query_manifest"},
            "private_mask": grasp._artifact(root / "mask.npy", root),
            "overlay": grasp._artifact(root / "overlay.png", root)}
        _write(path, manifest)
        grasp._materialize_target_index(artifact_root=artifact_root, target_root=root, manifest_path=path,
            observation_id=observation_id, camera=camera, target_ref=ref)
    candidate = {"candidate_id": f"T{position}", "position": position, "target_ref": ref,
        "backend_score": item["backend_score"], "source_queries": item["sources"],
        "shared_with_source_indexes": sorted({s["query_index"] for s in item["sources"]}),
        "representative_point_uv": item["point_uv"], "normalization": item["normalization"],
        "bbox_xyxy": summary["bbox_xyxy"], "mask_summary": summary, "localization": localization,
        "mask_touches_image_boundary": bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()),
        "overlay_sha256": manifest["overlay"]["sha256"]}
    rendered = {**candidate, "mask": mask, "rank": position}
    return candidate, rendered


def materialize_batch(*, artifact_root: Path, observation_id: str, camera: str,
                      rgb_path: Path, depth_path: Path, calibration_path: Path,
                      points_uv: list[list[int]], max_candidates: int,
                      backend: grasp.BatchPointSegmentationBackend,
                      depth_profile: Mapping[str, Any] | None = None,
                      ) -> tuple[dict[str, Any], list[Path], bool]:
    import numpy as np
    from PIL import Image

    rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
    depth = np.load(depth_path, allow_pickle=False)
    if depth.shape != rgb.shape[:2]:
        raise ValueError("batch RGB and depth shapes differ")
    validate_points(points_uv, width=rgb.shape[1], height=rgb.shape[0])
    if type(max_candidates) is not int or not 1 <= max_candidates <= MAX_CANDIDATES:
        raise ValueError("max_candidates must be an integer in 1..3")
    source = grasp._segmentation_source_identity(observation_id=observation_id, camera=camera,
        rgb_path=rgb_path, depth_path=depth_path, calibration_path=calibration_path)
    backend_record = {"id": backend.backend_id, "version": backend.backend_version,
                      "config_sha256": grasp._canonical_hash(dict(backend.configuration()))}
    identity = {"observation_id": observation_id, "camera": camera, "points_uv": points_uv,
        "max_candidates_per_point": max_candidates, "max_returned_candidates": MAX_RETURNED_CANDIDATES,
        "max_preview_images": MAX_PREVIEW_IMAGES, "source": source, "backend": backend_record,
        "selection": SELECTION_METHOD, "mask_iou_threshold": MASK_IOU_THRESHOLD,
        "renderer": RENDERER, **grounding_preview.identity(depth_profile)}
    ref = "segment-query-" + grasp._canonical_hash(identity)[:24]
    root = artifact_root / "segmentation-batches" / ref
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("identity") != identity or manifest.get("schema") != SCHEMA:
            raise ValueError("batch query identity drift")
        grasp._verify_artifact(root, manifest["raw_masks"])
        paths = [grasp._verify_artifact(root, record) for record in manifest["images"]]
        for candidate in manifest["candidates"]:
            grasp.load_segmented_target(artifact_root=artifact_root, target_ref=candidate["target_ref"])
        return manifest, paths, True
    raw, diagnostics = backend.segment_points(rgb_u8=rgb, depth_m=depth, points_uv=points_uv)
    selected, selection = select_masks(raw, points_uv=points_uv, shape=depth.shape,
                                       max_candidates=max_candidates)
    ledger, preview_count = selection.pop("queries"), selection.pop("preview_count")
    if source != grasp._segmentation_source_identity(observation_id=observation_id, camera=camera,
            rgb_path=rgb_path, depth_path=depth_path, calibration_path=calibration_path):
        raise ValueError("batch observation changed during inference")
    root.mkdir(parents=True, exist_ok=True)
    raw_stream = io.BytesIO()
    np.savez_compressed(raw_stream, **{f"q{x['query_index']}_c{x['source_index']}":
        np.asarray(x["mask"], dtype=np.uint8) for x in raw})
    raw_path = root / "raw-masks.npz"
    raw_path.write_bytes(raw_stream.getvalue())
    raw_path.chmod(0o444)
    candidates, rendered = [], []
    for position, item in enumerate(selected):
        candidate, drawing = _materialize_target(artifact_root=artifact_root, source=source,
            backend=backend_record, camera=camera, observation_id=observation_id, rgb=rgb, depth=depth,
            calibration_path=calibration_path, item=item, position=position)
        candidates.append(candidate)
        rendered.append(drawing)
    leading: list[Mapping[str, Any] | None] = []
    for entry in ledger:
        ids = entry["candidate_ids"]
        leading.append(rendered[int(ids[0][1:])] if ids else None)
    overview = _render_overview(rgb, points_uv=points_uv, best=leading)
    paths = [root / "overview.png"]
    grasp._atomic_png(paths[0], overview)
    previews = []
    for item in rendered[:preview_count]:
        preview, record = grasp._render_grounding_candidate_preview(
            rgb_u8=rgb, candidate=item, text="independent point proposals", points_uv=points_uv,
            depth_profile=grounding_preview.profile_input(
                depth_profile, depth_m=depth, calibration_path=calibration_path,
                localization=item["localization"]))
        paths.append(root / f"candidate-{item['candidate_id']}.png")
        grasp._atomic_png(paths[-1], preview)
        previews.append(record)
    for path in paths:
        path.chmod(0o444)
    queries = []
    backend_queries = diagnostics.get("query_results", [])
    if not isinstance(backend_queries, list) or len(backend_queries) != len(points_uv):
        raise ValueError("batch backend omitted per-query completion evidence")
    for entry, point, query in zip(ledger, points_uv, backend_queries):
        index = entry["query_index"]
        if query.get("query_index") != index or query.get("point_uv") != point:
            raise ValueError("batch backend query evidence identity drift")
        raw_count = sum(x["query_index"] == index for x in raw)
        backend_error = query.get("error_code")
        if query.get("raw_candidate_count") != raw_count or backend_error and raw_count:
            raise ValueError("batch backend query outcome contradicts its candidates")
        if backend_error:
            status, error_code = "failed", backend_error
        elif entry["candidate_ids"]:
            status, error_code = "candidates", None
        else:
            status = "no_candidate"
            error_code = "backend_returned_no_mask" if not raw_count else "mask_excludes_positive_point"
        queries.append({**entry, "raw_candidate_count": raw_count,
                        "candidate_count": len(entry["candidate_ids"]),
                        "status": status, "error_code": error_code})
    answered = sum(entry["status"] == "candidates" for entry in queries)
    manifest = {"schema": SCHEMA, "query_ref": ref, "identity": identity, "backend": backend_record,
        "status": "candidates" if answered == len(queries) else (
            "partial" if answered else "no_candidate"),
        "candidates": candidates, "queries": queries, "selection": selection,
        "diagnostics": {k: v for k, v in diagnostics.items() if k != "_service_identity"},
        "response_service_identity": diagnostics.get("_service_identity"),
        "raw_masks": grasp._artifact(raw_path, root),
        "raw_candidates": [{k: v for k, v in item.items() if k != "mask"} for item in raw],
        "images": [grasp._artifact(path, root) for path in paths], "candidate_previews": previews}
    _write(manifest_path, manifest)
    return manifest, paths, False


def public_result(manifest: Mapping[str, Any], *, cache_reused: bool) -> dict[str, Any]:
    identity = manifest["identity"]
    previews = manifest["candidate_previews"]
    return {"schema": SCHEMA, "status": manifest["status"], "query_ref": manifest["query_ref"],
        "observation_id": identity["observation_id"], "camera": identity["camera"],
        "points_uv": identity["points_uv"], "query_count": len(manifest["queries"]),
        "candidates": manifest["candidates"], "queries": manifest["queries"],
        "selection": manifest["selection"],
        "candidate_previews": [{"image_index": i, "position": record["rank"],
            **{k: record[k] for k in ("target_ref", "source_dimensions", "source_bbox_xyxy",
                "crop_xyxy", "render_dimensions", "resampling")}}
            for i, record in enumerate(previews, 1)],
        "image_order": ["overview", *[c["candidate_id"] for c in manifest["candidates"][:len(previews)]]],
        "cache_reused": cache_reused, "semantic_identity_verified": False,
        "claim_boundary": "independent point-prompt proposals with a per-point mask budget; "
            "one point never consumes another point's masks, an empty point means no mask at that "
            "pixel rather than an exhausted budget, and omitted_over_* counts every mask withheld; "
            "inspect the proposals and choose semantic identity; crops add no sensor detail"}
