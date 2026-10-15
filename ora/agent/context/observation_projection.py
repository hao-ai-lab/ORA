"""Shared deterministic projections of public current-observation RGB."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping


AUXILIARY_CONTACT_SHEET_VIEW = "auxiliary_contact_sheet"
AUXILIARY_CONTACT_SHEET_SCHEMA = (
    "ora.current-auxiliary-camera-contact-sheet.v1"
)
AUXILIARY_CONTACT_SHEET_CAMERAS = (
    "head_camera", "front_camera", "left_camera", "right_camera",
)
AUXILIARY_CONTACT_SHEET_LAYOUTS = {
    4: ("two_by_two_native_rgb_with_label_header", 2, 2),
    3: ("one_by_three_native_rgb_with_label_header", 3, 1),
}
CONTACT_SHEET_HEADER_HEIGHT = 24


class ObservationProjectionError(RuntimeError):
    """A derived current-observation projection cannot be verified."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


def materialize_auxiliary_contact_sheet(
    *, run_root: Path, observation_step: int,
    camera_paths: Mapping[str, Path],
    cameras: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Compose three or four current native RGB views without resampling pixels.

    Four cameras form a two-by-two sheet; three form one row. Their
    names come from the ToolSurface view_config
    (``auxiliary_contact_sheet_cameras``); the module default preserves the
    frozen RoboTwin surfaces that predate that field.
    """

    from PIL import Image, ImageDraw, ImageFont

    ordered_cameras = (
        tuple(cameras) if cameras is not None
        else AUXILIARY_CONTACT_SHEET_CAMERAS
    )
    layout = AUXILIARY_CONTACT_SHEET_LAYOUTS.get(len(ordered_cameras))
    if layout is None or len(set(ordered_cameras)) != len(ordered_cameras):
        raise ObservationProjectionError(
            "auxiliary contact sheet requires three or four unique cameras"
        )
    layout_name, columns, rows = layout
    run_root = run_root.resolve()
    if set(camera_paths) != set(ordered_cameras):
        raise ObservationProjectionError(
            "auxiliary contact sheet sources must match its configured cameras"
        )
    sources: list[dict[str, object]] = []
    opened: dict[str, Image.Image] = {}
    try:
        for camera in ordered_cameras:
            path = camera_paths[camera].resolve()
            if (
                not path.is_file() or path.is_symlink()
                or not path.is_relative_to(run_root)
            ):
                raise ObservationProjectionError(
                    f"auxiliary source image is unavailable: {camera}"
                )
            image = Image.open(path).convert("RGB")
            opened[camera] = image
            sources.append({
                "camera": camera,
                "path": path.relative_to(run_root).as_posix(),
                "sha256": _file_sha256(path),
                "width": image.width,
                "height": image.height,
            })
        identity = {
            "schema": AUXILIARY_CONTACT_SHEET_SCHEMA,
            "observation_step": int(observation_step),
            "camera_order": list(ordered_cameras),
            "layout": layout_name,
            "resampling": "none",
            "sources": sources,
        }
        identity_sha256 = _canonical_sha256(identity)
        root = run_root / "native" / "derived-observations" / (
            "auxiliary-contact-sheets"
        )
        root.mkdir(parents=True, exist_ok=True)
        stem = f"step-{int(observation_step):04d}-{identity_sha256[:16]}"
        image_path = root / f"{stem}.png"
        manifest_path = root / f"{stem}.json"
        if image_path.exists() or manifest_path.exists():
            if not image_path.is_file() or not manifest_path.is_file():
                raise ObservationProjectionError(
                    "auxiliary contact sheet evidence is incomplete"
                )
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ObservationProjectionError(
                    "auxiliary contact sheet manifest is invalid"
                ) from exc
            if (
                existing.get("identity") != identity
                or existing.get("image_sha256") != _file_sha256(image_path)
            ):
                raise ObservationProjectionError(
                    "auxiliary contact sheet evidence drifted"
                )
            return {
                "path": image_path,
                "width": int(existing["width"]),
                "height": int(existing["height"]),
                "manifest_path": manifest_path,
                "sha256": existing["image_sha256"],
            }

        cell_width = max(image.width for image in opened.values())
        image_height = max(image.height for image in opened.values())
        header_height = CONTACT_SHEET_HEADER_HEIGHT
        cell_height = header_height + image_height
        canvas = Image.new("RGB", (cell_width * columns, cell_height * rows), (12, 15, 19))
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
        except OSError:
            font = ImageFont.load_default()
        for index, camera in enumerate(ordered_cameras):
            column, row = index % columns, index // columns
            left, top = column * cell_width, row * cell_height
            image = opened[camera]
            x = left + (cell_width - image.width) // 2
            y = top + header_height
            canvas.paste(image, (x, y))
            draw.text((left + 8, top + 4), camera.upper(), fill=(238, 242, 247), font=font)
        temporary_image = image_path.with_suffix(".png.tmp")
        canvas.save(temporary_image, format="PNG", optimize=False)
        os.replace(temporary_image, image_path)
        manifest = {
            "identity": identity,
            "identity_sha256": identity_sha256,
            "image_path": image_path.relative_to(run_root).as_posix(),
            "image_sha256": _file_sha256(image_path),
            "width": canvas.width,
            "height": canvas.height,
        }
        temporary_manifest = manifest_path.with_suffix(".json.tmp")
        temporary_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_manifest, manifest_path)
        return {
            "path": image_path,
            "width": canvas.width,
            "height": canvas.height,
            "manifest_path": manifest_path,
            "sha256": manifest["image_sha256"],
        }
    finally:
        for image in opened.values():
            image.close()


__all__ = [
    "AUXILIARY_CONTACT_SHEET_CAMERAS", "AUXILIARY_CONTACT_SHEET_SCHEMA",
    "AUXILIARY_CONTACT_SHEET_VIEW", "ObservationProjectionError",
    "materialize_auxiliary_contact_sheet",
]
