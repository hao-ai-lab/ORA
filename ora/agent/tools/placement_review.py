"""Derived placement review image rendering from admitted public measurements."""
from __future__ import annotations
import io
import json
from .robot import ToolError
from pathlib import Path
from typing import Any, Mapping

def _review_image(path: Path, *, targets: list[dict[str, Any]], computation: Mapping[str, Any],
                  rim: Mapping[str, Any] | None, phase: str, stage: Mapping[str, Any]) -> None:
    import numpy as np
    from PIL import Image, ImageDraw
    canvas = Image.new("RGB", (1080, 850), (248, 248, 245))
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 15), f"PUBLIC PLACEMENT GEOMETRY / {phase} / command stage: {stage['status']}", fill=(30, 38, 44))
    palette = [(16, 137, 162), (202, 103, 40), (144, 74, 168), (77, 144, 64)]
    for index, target in enumerate(targets[:2]):
        rgb = Image.open(target["rgb"]).convert("RGB")
        annotator = ImageDraw.Draw(rgb)
        box = target["manifest"]["summary"]["bbox_xyxy"]
        annotator.rectangle(box, outline=palette[index], width=2)
        depth_evidence = target["geometry"].get("depth_boundary_evidence", {})
        for pixel in depth_evidence.get("uncertain_pixels_uv", []):
            annotator.point(tuple(pixel), fill=(255, 20, 65))
        if index == 0 and rim:
            for count, (x, y) in enumerate(rim["points_uv"]):
                annotator.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 30, 30))
                annotator.text((x + 3, y), str(count), fill=(255, 30, 30))
        rgb.thumbnail((480, 285))
        x = 20 + index * 540
        canvas.paste(rgb, (x, 66))
        suffix = (f" / red: {depth_evidence['measurements']['uncertain_pixel_count']} uncertain depth pixels"
                  if depth_evidence else "")
        draw.text((x, 45), target["source"]["role"] + " / " + target["source"]["camera"] + suffix,
                  fill=palette[index])
    groups = [np.asarray(target["geometry"]["voxel_points_world_m"], dtype=float) for target in targets]
    all_points = np.concatenate(groups)
    uncertain_groups = [np.asarray(target["geometry"].get("depth_boundary_evidence", {}).get(
        "uncertain_points_world_m", []), dtype=float).reshape(-1, 3) for target in targets]
    if any(len(points) for points in uncertain_groups):
        all_points = np.concatenate([all_points, *uncertain_groups])
    candidates = computation.get("candidates", [])
    for panel, axes in enumerate(((0, 1), (0, 2))):
        left, top, width, height = 28 + panel * 540, 406, 478, 230
        low, high = np.min(all_points[:, axes], axis=0), np.max(all_points[:, axes], axis=0)
        span = np.maximum(high - low, 0.10)
        low = (high + low) / 2 - span * 0.6
        span *= 1.2
        draw.rectangle((left, top, left + width, top + height), outline=(110, 110, 110))
        draw.text((left, top - 22), f"Observed world {'XY' if panel == 0 else 'XZ'} projection (metres); not extra views", fill=(30, 38, 44))
        def project(points):
            uv = (points[:, axes] - low) / span
            return [(left + float(p[0]) * width, top + (1 - float(p[1])) * height) for p in uv]
        for index, points in enumerate(groups):
            for x, y in project(points[::max(1, len(points) // 1200)]):
                draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=palette[index % len(palette)])
        for points in uncertain_groups:
            for x, y in project(points):
                draw.line((x - 2, y - 2, x + 2, y + 2), fill=(240, 20, 65), width=1)
                draw.line((x - 2, y + 2, x + 2, y - 2), fill=(240, 20, 65), width=1)
        for candidate in candidates:
            point = np.asarray([candidate["target_pose"]["position_m"]])
            x, y = project(point)[0]
            if left <= x <= left + width and top <= y <= top + height:
                draw.line((x - 5, y, x + 5, y), fill=(200, 20, 35), width=2)
                draw.line((x, y - 5, x, y + 5), fill=(200, 20, 35), width=2)
                draw.text((x + 6, y), candidate["candidate_id"] + " TCP", fill=(200, 20, 35))
        draw.text((left, top + height + 5), f"axis ranges: {low[0]:.3f}..{low[0]+span[0]:.3f} / {low[1]:.3f}..{low[1]+span[1]:.3f}", fill=(60, 60, 60))
    assessment = computation.get("assessment", computation)
    for index, (name, check) in enumerate(assessment["checks"].items()):
        draw.text((20, 682 + index * 22), f"{name}: {check['status']} / {check['reason']}", fill=(30, 38, 44))
    depth_active = any("depth_boundary_evidence" in target["geometry"] for target in targets)
    notice = ("RED = UNCERTAIN, NOT CERTIFIED BACKGROUND. Candidates are conditional; full-object fit UNKNOWN."
              if depth_active else "MISSING SURFACES REMAIN MISSING. RGB-D points are not complete object geometry.")
    draw.text((20, 791), notice, fill=(145, 62, 30))
    draw.text((20, 813), "Not identity, support contact, stability, full containment, collision-free path or official success.", fill=(145, 62, 30))
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    rendered = buffer.getvalue()
    if path.exists():
        if path.is_symlink() or path.read_bytes() != rendered:
            raise ToolError("placement overlay identity drift", code="artifact_hash_mismatch", retryable=False)
        return
    with path.open("xb") as stream:
        stream.write(rendered)
