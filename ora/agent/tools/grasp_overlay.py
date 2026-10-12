"""Drawing steps of the policy-visible ``grasp.plan`` target-local contact sheet.

``grasp._render_grasp_overlay`` reads the target RGB-D, compiles each candidate
and calls these steps in order: tint the target mask, draw the target crop,
the legend and one card per candidate. A card has a camera-projected
parallel-jaw wireframe above and a target-point isometric view below. The
steps only draw and project; candidate compilation stays in ``grasp.py``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, NamedTuple

# Candidate card layout on the 1440x1080 contact sheet.
_CARD_TOP = 536
_CARD_WIDTH, _CARD_HEIGHT, _CARD_GAP = 456, 518, 12
_CAMERA_VIEW_SIZE = 304
_LOCAL_VIEW_WIDTH, _LOCAL_VIEW_HEIGHT = 420, 150


class _OverlayTarget(NamedTuple):
    """The source RGB size, its mask-tinted copy and the mask pixel indices."""

    width: int
    height: int
    array: Any
    mask_rows: Any
    mask_columns: Any


class _OverlayFonts(NamedTuple):
    title: Any
    body: Any
    small: Any


def _project_world(
    point: Any, *, intrinsic: Any, extrinsic: Any,
) -> tuple[int, int] | None:
    import numpy as np

    world = np.asarray([*point, 1.0], dtype=np.float64)
    camera = extrinsic @ world
    if float(camera[2]) <= 1e-9:
        return None
    pixel = intrinsic @ camera[:3]
    return int(round(float(pixel[0] / pixel[2]))), int(round(float(pixel[1] / pixel[2])))


def _tinted_target_source(rgb_path: Path, mask_path: Path) -> tuple[Any, Any, Any]:
    """Return the RGB image, the boolean mask and the mask-tinted RGB array."""

    import numpy as np
    from PIL import Image

    source = Image.open(rgb_path).convert("RGB")
    mask = np.load(mask_path, allow_pickle=False)
    if mask.shape != (source.height, source.width):
        raise ValueError("grasp overlay target mask shape does not match RGB")
    mask = np.asarray(mask, dtype=bool)
    source_array = np.asarray(source, dtype=np.uint8).copy()
    source_array[mask] = np.clip(
        source_array[mask].astype(np.float64) * 0.55
        + np.asarray([32.0, 210.0, 190.0]) * 0.45,
        0, 255,
    ).astype(np.uint8)
    eroded = mask.copy()
    for row_shift, column_shift in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        shifted = np.zeros_like(mask)
        source_rows = slice(max(0, -row_shift), min(mask.shape[0], mask.shape[0] - row_shift))
        source_columns = slice(max(0, -column_shift), min(mask.shape[1], mask.shape[1] - column_shift))
        target_rows = slice(max(0, row_shift), min(mask.shape[0], mask.shape[0] + row_shift))
        target_columns = slice(max(0, column_shift), min(mask.shape[1], mask.shape[1] + column_shift))
        shifted[target_rows, target_columns] = mask[source_rows, source_columns]
        eroded &= shifted
    source_array[mask & ~eroded] = np.asarray([255, 230, 64], dtype=np.uint8)
    return source, mask, source_array


def _overlay_camera_matrices(calibration_path: Path) -> tuple[Any, Any]:
    """Return the OpenCV intrinsic and the 4x4 world-to-camera extrinsic."""

    import numpy as np

    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    intrinsic = np.asarray(calibration["intrinsic_cv"], dtype=np.float64)
    extrinsic = np.asarray(calibration["extrinsic_cv"], dtype=np.float64)
    if extrinsic.shape == (3, 4):
        extrinsic = np.vstack([extrinsic, [0.0, 0.0, 0.0, 1.0]])
    return intrinsic, extrinsic


def _overlay_fonts() -> _OverlayFonts:
    from PIL import ImageFont

    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
        body_font = ImageFont.truetype("DejaVuSans.ttf", 16)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        title_font = body_font = small_font = ImageFont.load_default()
    return _OverlayFonts(title_font, body_font, small_font)


def _overlay_target(source: Any, mask: Any, source_array: Any) -> _OverlayTarget:
    import numpy as np

    mask_rows, mask_columns = np.nonzero(mask)
    if not len(mask_rows):
        raise ValueError("grasp overlay target mask is empty")
    return _OverlayTarget(source.width, source.height, source_array, mask_rows, mask_columns)


def _square_crop(
    target: _OverlayTarget, pixels: list[tuple[int, int] | None], *,
    minimum_side: int, target_scale: float,
) -> tuple[int, int, int, int]:
    """Square source crop around the target mask and every in-frame pixel."""

    mask_rows, mask_columns = target.mask_rows, target.mask_columns
    valid = [
        value for value in pixels
        if value is not None
        and 0 <= value[0] < target.width
        and 0 <= value[1] < target.height
    ]
    xs = [int(mask_columns.min()), int(mask_columns.max()), *(v[0] for v in valid)]
    ys = [int(mask_rows.min()), int(mask_rows.max()), *(v[1] for v in valid)]
    target_extent = max(
        int(mask_columns.max() - mask_columns.min() + 1),
        int(mask_rows.max() - mask_rows.min() + 1),
    )
    side = max(
        minimum_side,
        int(math.ceil(max(max(xs) - min(xs) + 1, max(ys) - min(ys) + 1) * 1.35)),
        int(math.ceil(target_extent * target_scale)),
    )
    side = min(side, target.width, target.height)
    center_x = (min(xs) + max(xs)) / 2.0
    center_y = (min(ys) + max(ys)) / 2.0
    left = int(round(center_x - side / 2))
    top = int(round(center_y - side / 2))
    left = min(max(0, left), target.width - side)
    top = min(max(0, top), target.height - side)
    return left, top, left + side, top + side


def _gripper_wireframe(geometry: Mapping[str, Any], opening: float) -> dict[str, Any]:
    """Palm, finger and approach segments plus the eight marker points."""

    import numpy as np

    origin = np.asarray(geometry["source_gripper_origin"], dtype=np.float64)
    baseline = np.asarray(geometry["contact_baseline"], dtype=np.float64)
    jaw = np.asarray(geometry["jaw_axis"], dtype=np.float64)
    outer_half = max(opening / 2.0 + 0.012, 0.02)
    palm_negative = origin - jaw * outer_half
    palm_positive = origin + jaw * outer_half
    contact_negative = np.asarray(
        geometry["contact_negative_jaw"], dtype=np.float64,
    )
    contact_positive = np.asarray(
        geometry["contact_positive_jaw"], dtype=np.float64,
    )
    return {
        "segments": (
            (palm_negative, palm_positive),
            (palm_negative, contact_negative),
            (palm_positive, contact_positive),
            (origin, baseline),
        ),
        "points": (
            palm_negative, palm_positive, contact_negative, contact_positive,
            origin, baseline, geometry["grip_site"],
            geometry["pregrasp_grip_site"],
        ),
    }


def _new_contact_sheet(
    *, observation_id: str, target_ref: str, fonts: _OverlayFonts,
) -> tuple[Any, Any]:
    from PIL import Image, ImageDraw

    canvas_width, canvas_height = 1440, 1080
    image = Image.new("RGB", (canvas_width, canvas_height), (18, 21, 27))
    draw = ImageDraw.Draw(image)
    draw.text((24, 18), "GRASP.PLAN TARGET-LOCAL CONTACT SHEET", fill=(242, 244, 248), font=fonts.title)
    draw.text(
        (24, 50),
        f"{observation_id[-30:]}  ·  {target_ref}",
        fill=(165, 174, 190), font=fonts.small,
    )
    return image, draw


def _draw_target_panel(
    image: Any, draw: Any, target: _OverlayTarget, fonts: _OverlayFonts,
) -> tuple[int, int, int, int]:
    """Draw the mask-only target crop and return its source crop box."""

    from PIL import Image

    target_crop = _square_crop(target, [], minimum_side=96, target_scale=4.0)
    target_panel = Image.fromarray(target.array, mode="RGB").crop(target_crop).resize(
        (420, 420), Image.Resampling.NEAREST,
    )
    image.paste(target_panel, (24, 86))
    draw.rectangle((24, 86, 444, 506), outline=(90, 100, 116), width=2)
    draw.text((38, 98), "TARGET CROP · MASK ONLY", fill=(242, 244, 248), font=fonts.body)
    draw.text(
        (38, 474),
        f"native crop {target_crop[2]-target_crop[0]}×{target_crop[3]-target_crop[1]} px → 420×420",
        fill=(242, 244, 248), font=fonts.small,
    )
    return target_crop


def _draw_contact_sheet_legend(
    draw: Any, *, compiler_identity: Mapping[str, Any], execution_tool: str,
    fonts: _OverlayFonts,
) -> None:
    legend_x = 480
    draw.text((legend_x, 94), "HOW TO READ THE CANDIDATE CARDS", fill=(242, 244, 248), font=fonts.title)
    if execution_tool not in {"move_free_space", "move_pose"}:
        raise ValueError("grasp overlay execution tool is unsupported")
    embodiment_label = (
        "ALOHA"
        if compiler_identity.get("embodiment_descriptor_id") == "aloha_agilex"
        else "X5"
        if compiler_identity.get("embodiment_descriptor_id") == "arx_x5_robodojo_v1"
        else "Panda"
    )
    controlled_point = str(compiler_identity.get("controlled_point", "grip_site"))
    legend_lines = (
        ("Each candidate has its own panel; labels and grippers never overlap.", (210, 215, 224)),
        ("Upper: target-local RGB projection with a full parallel-jaw wireframe.", (210, 215, 224)),
        ("Lower: target-point 3D isometric view using the same public RGB-D.", (210, 215, 224)),
        ("gray ×  Contact-GraspNet source origin", (176, 184, 198)),
        (f"colored ○  compiled {embodiment_label} {controlled_point}", (80, 210, 255)),
        (f"colored □  compiled pregrasp {controlled_point}", (255, 205, 64)),
        (f"U wireframe  {embodiment_label} palm, fingers and predicted jaw opening", (242, 244, 248)),
    )
    line_y = 136
    for label, colour in legend_lines:
        draw.text((legend_x, line_y), label, fill=colour, font=fonts.body)
        line_y += 35
    draw.text((legend_x, 400), "GEOMETRY PREVIEW ONLY", fill=(255, 176, 64), font=fonts.title)
    draw.text((legend_x, 438), f"IK and execution remain delegated to {execution_tool}.", fill=(198, 204, 214), font=fonts.body)
    draw.text((legend_x, 470), "Contact, retention and task success are not established.", fill=(198, 204, 214), font=fonts.body)


def _map_crop(pixel: tuple[int, int] | None, crop: tuple[int, int, int, int], origin: tuple[int, int]) -> tuple[int, int] | None:
    if pixel is None:
        return None
    left, top, right, bottom = crop
    x = origin[0] + int(round((pixel[0] - left) / (right - left) * _CAMERA_VIEW_SIZE))
    y = origin[1] + int(round((pixel[1] - top) / (bottom - top) * _CAMERA_VIEW_SIZE))
    return x, y


def _isometric_pixels(points: Any, bounds_points: Any, origin: tuple[int, int]) -> Any:
    import numpy as np

    values = np.asarray(points, dtype=np.float64)
    bounds = np.asarray(bounds_points, dtype=np.float64)
    basis_u = np.asarray([1.0, -1.0, 0.0]) / math.sqrt(2.0)
    basis_v = np.asarray([-1.0, -1.0, 2.0]) / math.sqrt(6.0)
    projected = np.column_stack([values @ basis_u, values @ basis_v])
    projected_bounds = np.column_stack([bounds @ basis_u, bounds @ basis_v])
    lower = projected_bounds.min(axis=0)
    upper = projected_bounds.max(axis=0)
    span = np.maximum(upper - lower, 1e-6)
    scale = min((_LOCAL_VIEW_WIDTH - 24) / span[0], (_LOCAL_VIEW_HEIGHT - 24) / span[1])
    center = (lower + upper) / 2.0
    pixels = np.empty_like(projected)
    pixels[:, 0] = origin[0] + _LOCAL_VIEW_WIDTH / 2 + (projected[:, 0] - center[0]) * scale
    pixels[:, 1] = origin[1] + _LOCAL_VIEW_HEIGHT / 2 - (projected[:, 1] - center[1]) * scale
    return np.rint(pixels).astype(int)


def _draw_candidate_card(
    image: Any, draw: Any, *, index: int, compiled_records: list[dict[str, Any]],
    target: _OverlayTarget, intrinsic: Any, extrinsic: Any, target_world: Any,
    fonts: _OverlayFonts,
) -> None:
    """Draw card ``index``: the camera view above, the isometric view below."""

    card_left = 24 + index * (_CARD_WIDTH + _CARD_GAP)
    draw.rectangle(
        (card_left, _CARD_TOP, card_left + _CARD_WIDTH, _CARD_TOP + _CARD_HEIGHT),
        fill=(28, 32, 40), outline=(82, 90, 104), width=2,
    )
    if index >= len(compiled_records):
        draw.text((card_left + 22, _CARD_TOP + 24), f"G{index} · NO CANDIDATE", fill=(125, 134, 148), font=fonts.title)
        return
    record = compiled_records[index]
    _draw_candidate_camera_view(
        image, draw, record=record, card_left=card_left, target=target,
        intrinsic=intrinsic, extrinsic=extrinsic, body_font=fonts.body,
    )
    _draw_candidate_isometric_view(
        draw, record=record, card_left=card_left, target_world=target_world,
        small_font=fonts.small,
    )


def _draw_candidate_camera_view(
    image: Any, draw: Any, *, record: Mapping[str, Any], card_left: int,
    target: _OverlayTarget, intrinsic: Any, extrinsic: Any, body_font: Any,
) -> None:
    """Target-local RGB crop with the projected wireframe and markers."""

    from PIL import Image

    candidate = record["candidate"]
    colour = record["colour"]
    crop = _square_crop(
        target, record["source_pixels"], minimum_side=72, target_scale=2.4,
    )
    crop_image = Image.fromarray(target.array, mode="RGB").crop(crop).resize(
        (_CAMERA_VIEW_SIZE, _CAMERA_VIEW_SIZE), Image.Resampling.NEAREST,
    )
    camera_origin = (card_left + (_CARD_WIDTH - _CAMERA_VIEW_SIZE) // 2, _CARD_TOP + 62)
    image.paste(crop_image, camera_origin)
    draw.rectangle(
        (*camera_origin, camera_origin[0] + _CAMERA_VIEW_SIZE, camera_origin[1] + _CAMERA_VIEW_SIZE),
        outline=(100, 108, 122), width=2,
    )
    shape = record["wireframe"]
    for start, end in shape["segments"]:
        start_pixel = _map_crop(
            _project_world(start, intrinsic=intrinsic, extrinsic=extrinsic),
            crop, camera_origin,
        )
        end_pixel = _map_crop(
            _project_world(end, intrinsic=intrinsic, extrinsic=extrinsic),
            crop, camera_origin,
        )
        if start_pixel is not None and end_pixel is not None:
            draw.line((start_pixel, end_pixel), fill=colour, width=6)
    geometry = record["geometry"]
    raw_pixel = _map_crop(
        _project_world(geometry["source_gripper_origin"], intrinsic=intrinsic, extrinsic=extrinsic),
        crop, camera_origin,
    )
    grip_pixel = _map_crop(
        _project_world(geometry["grip_site"], intrinsic=intrinsic, extrinsic=extrinsic),
        crop, camera_origin,
    )
    pregrasp_pixel = _map_crop(
        _project_world(geometry["pregrasp_grip_site"], intrinsic=intrinsic, extrinsic=extrinsic),
        crop, camera_origin,
    )
    if raw_pixel is not None:
        x, y = raw_pixel
        draw.line((x - 7, y - 7, x + 7, y + 7), fill=(190, 196, 208), width=3)
        draw.line((x - 7, y + 7, x + 7, y - 7), fill=(190, 196, 208), width=3)
    if grip_pixel is not None:
        x, y = grip_pixel
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), outline=colour, width=4)
    if pregrasp_pixel is not None:
        x, y = pregrasp_pixel
        draw.rectangle((x - 7, y - 7, x + 7, y + 7), outline=colour, width=3)
    draw.text(
        (card_left + 16, _CARD_TOP + 16),
        f"G{candidate['rank']}  score {float(candidate['backend_score']):.3f}  opening {float(candidate['gripper_opening_m'])*1000:.1f} mm",
        fill=colour, font=body_font,
    )


def _draw_candidate_isometric_view(
    draw: Any, *, record: Mapping[str, Any], card_left: int, target_world: Any,
    small_font: Any,
) -> None:
    """Target points and the gripper wireframe in one isometric 3D view."""

    import numpy as np

    colour = record["colour"]
    shape = record["wireframe"]
    geometry = record["geometry"]
    local_origin = (card_left + 18, _CARD_TOP + 374)
    draw.rectangle(
        (*local_origin, local_origin[0] + _LOCAL_VIEW_WIDTH, local_origin[1] + _LOCAL_VIEW_HEIGHT),
        outline=(68, 75, 89), width=1,
    )
    bounds_points = np.vstack([target_world, *shape["points"], geometry["pregrasp_grip_site"]])
    target_pixels = _isometric_pixels(target_world, bounds_points, local_origin)
    for x, y in target_pixels[::max(1, len(target_pixels) // 1800)]:
        draw.point((int(x), int(y)), fill=(32, 210, 190))
    for start, end in shape["segments"]:
        segment_pixels = _isometric_pixels(np.vstack([start, end]), bounds_points, local_origin)
        draw.line(tuple(map(tuple, segment_pixels)), fill=colour, width=5)
    marker_pixels = _isometric_pixels(
        np.vstack([
            geometry["source_gripper_origin"], geometry["grip_site"],
            geometry["pregrasp_grip_site"],
        ]),
        bounds_points, local_origin,
    )
    raw_3d, grip_3d, pregrasp_3d = marker_pixels
    draw.line((raw_3d[0]-5, raw_3d[1]-5, raw_3d[0]+5, raw_3d[1]+5), fill=(190, 196, 208), width=2)
    draw.ellipse((grip_3d[0]-6, grip_3d[1]-6, grip_3d[0]+6, grip_3d[1]+6), outline=colour, width=3)
    draw.rectangle((pregrasp_3d[0]-5, pregrasp_3d[1]-5, pregrasp_3d[0]+5, pregrasp_3d[1]+5), outline=colour, width=2)
    draw.text((local_origin[0] + 8, local_origin[1] + 7), "TARGET-LOCAL 3D", fill=(230, 234, 240), font=small_font)
