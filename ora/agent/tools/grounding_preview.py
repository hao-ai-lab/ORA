"""One enlargement card per grounding or point-segmentation candidate.

The card was two 480 px panels -- the exact source crop and the same crop with
the candidate mask tinted -- and so it showed only what the camera's 2-D pixels
show. A candidate's shape in depth was nowhere in an image: a recorded
put_bottles_into_dustbin run ranked a floor-standing dustbin first in
``vision_ground`` (centre (-0.81, 0.07, 0.59), visible z 0.39-0.73) and the
policy still did not recognise it, because a tall open container seen from
above is a dark ring in a crop.

Under :data:`CAPABILITY` the card grows from 1024x576 to 1024x800 and a strip
below the crops draws the candidate's own depth points, the inlier set the
localizer measures (``grasp._mask_depth_inlier_selection``), in the
candidate's colour, over grey context points from every depth-valid pixel of
the same crop, in two metric views at one scale:

* elevation -- horizontal is the camera-right axis projected onto the floor
  plane, as an offset from the candidate centre; vertical is world z;
* top -- world x horizontal, world y vertical (up is +y, as in the grasp
  review's TOP panel).

Both windows are centred on the candidate's localized centre. The vertical
span is ``max(0.10 m, 2.5 x largest extent)``; the horizontal span follows
from the plot's aspect at the same metres-per-pixel. A dashed line marks the
localizer's ``support_z_m`` in the elevation view when the surface also
declares ``target_support_height_v1`` (so the line never shows a number the
surface's own results withhold) and the estimate exists. Every value drawn is
the public RGB-D measurement already behind the candidate's localization;
nothing is task-specific or simulator-derived.

The crops keep their size, position and resampling, so the published
``resampling`` string and the policy-visible preview record keep their keys;
only ``render_dimensions`` changes value. Without the capability the card is
byte-identical to the released renderer.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .grounding_palette import grounding_candidate_colour

GROUNDING_VISUALIZATION_ID = "grounding-candidate-inspection.v2"
CAPABILITY = "grounding_candidate_depth_profile_v1"
PROFILE_RENDERER = "grounding-candidate-depth-profile.v1"
SUPPORT_CAPABILITY = "target_support_height_v1"

CARD_WIDTH = 1024
LEGACY_CARD_HEIGHT = 576
PROFILE_CARD_HEIGHT = 800
MINIMUM_SPAN_M = 0.10
EXTENT_SPAN_FACTOR = 2.5
#: The localizer's valid depth window; context points use the same one.
VALID_DEPTH_M = (1e-6, 2.5)

# Strip geometry: two panels under the crops, each with a data area whose
# metres-per-pixel is shared by both axes and both views.
_STRIP_TOP = 580
_PANEL_LEFTS = (16, 528)
_PANEL_WIDTH = 480
_PANEL_HEIGHT = 212
_DATA_LEFT_PAD = 46
_DATA_RIGHT_PAD = 8
_DATA_TOP_PAD = 22
_DATA_BOTTOM_PAD = 18
_DATA_WIDTH = _PANEL_WIDTH - _DATA_LEFT_PAD - _DATA_RIGHT_PAD
_DATA_HEIGHT = _PANEL_HEIGHT - _DATA_TOP_PAD - _DATA_BOTTOM_PAD
_TICK_STEPS_M = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
_CONTEXT_COLOUR = (112, 118, 130)
_FRAME_COLOUR = (82, 90, 104)
_TEXT_COLOUR = (225, 229, 236)
_LABEL_COLOUR = (178, 187, 200)
_SUPPORT_COLOUR = (255, 176, 64)


def options(surface: Any) -> dict[str, Any] | None:
    """What a resolved surface asks the card to draw, or ``None`` for the released card."""

    if surface is None or not surface.supports(CAPABILITY):
        return None
    return {"renderer": PROFILE_RENDERER,
            "support_line": bool(surface.supports(SUPPORT_CAPABILITY))}


def identity(profile_options: Mapping[str, Any] | None) -> dict[str, Any]:
    """The query-identity extension; empty without the capability so released refs keep their hashes."""

    return {} if profile_options is None else {"depth_profile": dict(profile_options)}


def profile_input(
    profile_options: Mapping[str, Any] | None, *, depth_m: Any,
    calibration_path: Path, localization: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Bundle what one card's strip reads; ``None`` keeps the released card."""

    if profile_options is None:
        return None
    return {"options": dict(profile_options), "depth_m": depth_m,
            "calibration_path": Path(calibration_path),
            "localization": dict(localization or {})}


def crop_box(*, bbox_xyxy: list[int], image_width: int, image_height: int) -> list[int]:
    """Return one deterministic square context crop around a grounded mask."""

    x0, y0, x1, y1 = bbox_xyxy
    box_width = max(1, x1 - x0)
    box_height = max(1, y1 - y0)
    side = min(
        min(image_width, image_height),
        max(64, int(math.ceil(max(box_width, box_height) * 3.0))),
    )
    center_x = (x0 + x1) / 2.0
    center_y = (y0 + y1) / 2.0
    left = int(round(center_x - side / 2.0))
    top = int(round(center_y - side / 2.0))
    left = min(max(0, left), max(0, image_width - side))
    top = min(max(0, top), max(0, image_height - side))
    right = min(image_width, left + side)
    bottom = min(image_height, top + side)
    return [left, top, right, bottom]


def render_candidate_preview(
    *, rgb_u8: Any, candidate: Mapping[str, Any], text: str,
    points_uv: Sequence[Sequence[int]] = (),
    depth_profile: Mapping[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Render raw and mask-highlighted enlargements for one semantic candidate."""

    import cv2
    import numpy as np
    from PIL import Image

    rgb = np.asarray(rgb_u8, dtype=np.uint8)
    height, width = rgb.shape[:2]
    crop_xyxy = crop_box(
        bbox_xyxy=[int(item) for item in candidate["bbox_xyxy"]],
        image_width=width,
        image_height=height,
    )
    x0, y0, x1, y1 = crop_xyxy
    raw_crop = rgb[y0:y1, x0:x1]
    mask_crop = np.asarray(candidate["mask"], dtype=bool)[y0:y1, x0:x1]
    if raw_crop.size == 0 or mask_crop.shape != raw_crop.shape[:2]:
        raise ValueError("grounding candidate crop is empty or unaligned")

    rank = int(candidate["rank"])
    colour = np.asarray(grounding_candidate_colour(rank), dtype=np.float64)
    selected = raw_crop.copy()
    selected[mask_crop] = np.clip(
        0.62 * selected[mask_crop].astype(np.float64) + 0.38 * colour,
        0, 255,
    ).astype(np.uint8)

    panel_size = 480
    raw_panel = cv2.resize(
        raw_crop, (panel_size, panel_size), interpolation=cv2.INTER_CUBIC,
    )
    selected_panel = cv2.resize(
        selected, (panel_size, panel_size), interpolation=cv2.INTER_CUBIC,
    )
    card_height = LEGACY_CARD_HEIGHT if depth_profile is None else PROFILE_CARD_HEIGHT
    canvas = np.full((card_height, CARD_WIDTH, 3), 20, dtype=np.uint8)
    canvas[80:560, 16:496] = raw_panel
    canvas[80:560, 528:1008] = selected_panel
    for index, (u, v) in enumerate(points_uv):
        if x0 <= u < x1 and y0 <= v < y1:
            px = min(panel_size - 1, int((u - x0) * panel_size / (x1 - x0)))
            py = min(panel_size - 1, int((v - y0) * panel_size / (y1 - y0)))
            for left in (16, 528):
                location = (left + px, 80 + py)
                cv2.drawMarker(canvas, location, (255, 64, 128), cv2.MARKER_CROSS, 13, 2)
                cv2.putText(canvas, f"Q{index}", (location[0] + 5, location[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 64, 128), 1, cv2.LINE_AA)
    display_colour = tuple(int(item) for item in colour)
    title = (
        f"T{rank}  score={float(candidate['backend_score']):.4f}  "
        f"query={text[:72]}"
    )
    cv2.putText(
        canvas, title, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
        display_colour, 2, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, "SOURCE CROP + QUERY POINTS" if points_uv else "RAW SOURCE CROP", (26, 108), cv2.FONT_HERSHEY_SIMPLEX,
        0.55, (255, 255, 255), 1, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, "MASK HIGHLIGHT", (538, 108), cv2.FONT_HERSHEY_SIMPLEX,
        0.55, (255, 255, 255), 1, cv2.LINE_AA,
    )
    cv2.rectangle(canvas, (16, 80), (495, 559), display_colour, 2)
    cv2.rectangle(canvas, (528, 80), (1007, 559), display_colour, 2)
    metadata = {
        "renderer": GROUNDING_VISUALIZATION_ID,
        "rank": rank,
        "target_ref": candidate["target_ref"],
        "backend_score": float(candidate["backend_score"]),
        "source_dimensions": [width, height],
        "source_bbox_xyxy": [int(item) for item in candidate["bbox_xyxy"]],
        "crop_xyxy": crop_xyxy,
        "render_dimensions": [CARD_WIDTH, card_height],
        "resampling": "bicubic_visual_enlargement_no_new_source_detail",
        "claim_boundary": (
            "policy-visible enlargement of exact public RGB pixels and mask; "
            "not higher-resolution sensing or semantic ground truth"
        ),
    }
    if depth_profile is not None:
        metadata["depth_profile"] = _draw_depth_profile(
            canvas, mask=np.asarray(candidate["mask"], dtype=bool),
            crop_xyxy=crop_xyxy, colour=display_colour, profile=depth_profile,
        )
    return Image.fromarray(canvas, mode="RGB"), metadata


def _world_points(depth: Any, rows: Any, columns: Any, intrinsic: Any, camera_to_world: Any) -> Any:
    import numpy as np

    z = depth[rows, columns]
    camera = np.stack([
        (columns - intrinsic[0, 2]) * z / intrinsic[0, 0],
        (rows - intrinsic[1, 2]) * z / intrinsic[1, 1],
        z,
    ], axis=1)
    return camera @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]


def _calibration(path: Path) -> tuple[Any, Any]:
    import numpy as np

    calibration = json.loads(Path(path).read_text(encoding="utf-8"))
    intrinsic = np.asarray(calibration["intrinsic_cv"], dtype=np.float64)
    extrinsic = np.asarray(calibration["extrinsic_cv"], dtype=np.float64)
    if extrinsic.shape == (3, 4):
        extrinsic = np.vstack([extrinsic, [0.0, 0.0, 0.0, 1.0]])
    if intrinsic.shape != (3, 3) or extrinsic.shape != (4, 4):
        raise ValueError("depth profile calibration shape mismatch")
    return intrinsic, np.linalg.inv(extrinsic)


def _tick_step(span_m: float, most: int) -> float:
    """The smallest listed step that leaves at most ``most`` ticks across ``span_m``."""

    for step in _TICK_STEPS_M:
        if span_m / step <= most:
            return step
    return _TICK_STEPS_M[-1]


def _ticks(low: float, high: float, step: float) -> list[float]:
    """Multiples of ``step`` inside ``[low, high)``, rounded so labels never read -0.00."""

    first = math.ceil(low / step - 1e-9)
    values = []
    index = first
    while index * step < high - 1e-12:
        value = round(index * step, 6)
        values.append(0.0 if abs(value) < 1e-9 else value)
        index += 1
    return values


def _strip_unavailable(canvas: Any, record: dict[str, Any], reason: str) -> dict[str, Any]:
    import cv2

    canvas[LEGACY_CARD_HEIGHT:] = 20
    for left in _PANEL_LEFTS:
        cv2.rectangle(canvas, (left, _STRIP_TOP), (left + _PANEL_WIDTH - 1,
                      _STRIP_TOP + _PANEL_HEIGHT - 1), _FRAME_COLOUR, 1)
    cv2.putText(canvas, "DEPTH PROFILE unavailable: " + reason.replace("_", " "),
                (26, _STRIP_TOP + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _TEXT_COLOUR, 1, cv2.LINE_AA)
    return {**record, "status": "unavailable", "reason": reason}


def _draw_depth_profile(
    canvas: Any, *, mask: Any, crop_xyxy: list[int], colour: tuple[int, int, int],
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Draw the strip under the crops; return the private record of what it shows.

    The strip is evidence about the card, never a reason to lose the card: an
    unusable input leaves a one-line ``unavailable`` strip and its reason.
    """

    import numpy as np

    record: dict[str, Any] = {"renderer": PROFILE_RENDERER, "options": dict(profile["options"])}
    localization = profile["localization"]
    depth = np.asarray(profile["depth_m"], dtype=np.float64)
    if (localization.get("status") != "localized" or localization.get("center_xyz_m") is None
            or depth.shape != mask.shape):
        return _strip_unavailable(canvas, record, "candidate_not_localized")
    try:
        geometry = _profile_geometry(mask=mask, depth=depth, crop_xyxy=crop_xyxy,
                                     profile=profile)
    except (ValueError, KeyError, OSError, np.linalg.LinAlgError):
        return _strip_unavailable(canvas, record, "calibration_unusable")
    views = (
        ("ELEVATION", "camera-right offset / world z (m)",
         lambda points: (points - geometry["center"]) @ geometry["right"],
         lambda points: points[:, 2], 0.0, float(geometry["center"][2]), True),
        ("TOP", "world x / world y (m)", lambda points: points[:, 0],
         lambda points: points[:, 1], float(geometry["center"][0]),
         float(geometry["center"][1]), False),
    )
    view_records = [
        _draw_view(canvas, left=left, spec=spec, geometry=geometry, colour=colour)
        for left, spec in zip(_PANEL_LEFTS, views, strict=True)
    ]
    return {**record, "status": "rendered",
            "window_center_xyz_m": [float(value) for value in geometry["center"]],
            "vertical_span_m": geometry["vertical_span"],
            "horizontal_span_m": geometry["horizontal_span"],
            "metres_per_pixel": geometry["metres_per_pixel"],
            "elevation_horizontal_axis_world": [float(value) for value in geometry["right"]],
            "candidate_depth_points": int(len(geometry["target"])),
            "context_depth_points": int(len(geometry["context"])),
            "support_z_m": geometry["support"],
            "views": view_records}


def _profile_geometry(*, mask: Any, depth: Any, crop_xyxy: list[int],
                      profile: Mapping[str, Any]) -> dict[str, Any]:
    """World points, window and axes the two views share."""

    import numpy as np
    from . import grasp  # grasp imports this module; the localizer's selection lives there

    localization = profile["localization"]
    intrinsic, camera_to_world = _calibration(profile["calibration_path"])
    inliers, _selection = grasp._mask_depth_inlier_selection(mask=mask, depth_m=depth)
    rows, columns = np.nonzero(inliers)
    x0, y0, x1, y1 = crop_xyxy
    window = depth[y0:y1, x0:x1]
    valid = np.isfinite(window) & (window > VALID_DEPTH_M[0]) & (window <= VALID_DEPTH_M[1])
    context_rows, context_columns = np.nonzero(valid)
    extent = localization.get("extent_xyz_m") or [0.0, 0.0, 0.0]
    largest = max([float(localization.get("largest_extent_m") or 0.0),
                   *[float(value) for value in extent]])
    vertical_span = max(MINIMUM_SPAN_M, EXTENT_SPAN_FACTOR * largest)
    right = camera_to_world[:3, 0].copy()
    right[2] = 0.0
    if float(np.linalg.norm(right)) < 1e-3:
        right = np.asarray([1.0, 0.0, 0.0])
    support = localization.get("support_z_m") if profile["options"].get("support_line") else None
    return {
        "target": _world_points(depth, rows, columns, intrinsic, camera_to_world),
        "context": _world_points(depth, context_rows + y0, context_columns + x0, intrinsic,
                                 camera_to_world),
        "center": np.asarray(localization["center_xyz_m"], dtype=np.float64),
        "right": right / np.linalg.norm(right),
        "vertical_span": float(vertical_span),
        "horizontal_span": float(vertical_span / _DATA_HEIGHT * _DATA_WIDTH),
        "metres_per_pixel": float(vertical_span / _DATA_HEIGHT),
        "support": None if support is None else float(support),
    }


def _draw_view(canvas: Any, *, left: int, spec: tuple, geometry: Mapping[str, Any],
               colour: tuple[int, int, int]) -> dict[str, Any]:
    """One metric scatter panel: grey context, the candidate in colour, ticks, support."""

    import cv2
    import numpy as np

    name, axes_label, h_of, v_of, h_center, v_center, relative = spec
    metres_per_pixel = geometry["metres_per_pixel"]
    vertical_span, horizontal_span = geometry["vertical_span"], geometry["horizontal_span"]
    data_left, data_top = left + _DATA_LEFT_PAD, _STRIP_TOP + _DATA_TOP_PAD
    h_low, v_high = h_center - horizontal_span / 2.0, v_center + vertical_span / 2.0

    def pixels(points: Any) -> tuple[Any, Any, Any]:
        column = np.floor((h_of(points) - h_low) / metres_per_pixel).astype(np.int64)
        row = np.floor((v_high - v_of(points)) / metres_per_pixel).astype(np.int64)
        inside = (column >= 0) & (column < _DATA_WIDTH) & (row >= 0) & (row < _DATA_HEIGHT)
        return column[inside], row[inside], inside

    cv2.rectangle(canvas, (left, _STRIP_TOP), (left + _PANEL_WIDTH - 1,
                  _STRIP_TOP + _PANEL_HEIGHT - 1), colour, 1)
    plot = canvas[data_top:data_top + _DATA_HEIGHT, data_left:data_left + _DATA_WIDTH]
    plot[:] = (28, 32, 40)
    context_x, context_y, context_inside = pixels(geometry["context"])
    plot[context_y, context_x] = _CONTEXT_COLOUR
    target_x, target_y, target_inside = pixels(geometry["target"])
    for delta_x, delta_y in ((0, 0), (1, 0), (0, 1), (1, 1)):
        plot[np.clip(target_y + delta_y, 0, _DATA_HEIGHT - 1),
             np.clip(target_x + delta_x, 0, _DATA_WIDTH - 1)] = colour
    for value in _ticks(v_high - vertical_span, v_high, _tick_step(vertical_span, 5)):
        y = data_top + int((v_high - value) / metres_per_pixel)
        cv2.line(canvas, (data_left - 4, y), (data_left - 1, y), _LABEL_COLOUR, 1)
        cv2.putText(canvas, f"{value:.2f}", (left + 4, y + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, _LABEL_COLOUR, 1, cv2.LINE_AA)
    for value in _ticks(h_low, h_low + horizontal_span, _tick_step(horizontal_span, 7)):
        x, y = data_left + int((value - h_low) / metres_per_pixel), data_top + _DATA_HEIGHT
        if not 12 <= x - data_left <= _DATA_WIDTH - 12:
            continue
        cv2.line(canvas, (x, y), (x, y + 3), _LABEL_COLOUR, 1)
        label = f"{value:+.2f}" if relative and value else f"{value:.2f}"
        cv2.putText(canvas, label, (x - 14, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    _LABEL_COLOUR, 1, cv2.LINE_AA)
    support = geometry["support"]
    support_drawn = bool(relative and support is not None
                         and v_high - vertical_span <= support <= v_high)
    if support_drawn:
        y = data_top + int((v_high - support) / metres_per_pixel)
        for x in range(data_left, data_left + _DATA_WIDTH, 10):
            cv2.line(canvas, (x, y), (min(x + 5, data_left + _DATA_WIDTH - 1), y),
                     _SUPPORT_COLOUR, 1)
        cv2.putText(canvas, f"support_z_m {support:.3f}", (data_left + _DATA_WIDTH - 136, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, _SUPPORT_COLOUR, 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (data_left - 1, data_top - 1),
                  (data_left + _DATA_WIDTH, data_top + _DATA_HEIGHT), _FRAME_COLOUR, 1)
    cv2.putText(canvas, f"{name}  {axes_label}  1 px = {metres_per_pixel * 1000:.1f} mm",
                (left + 6, _STRIP_TOP + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, _TEXT_COLOUR,
                1, cv2.LINE_AA)
    return {"view": name.lower(), "horizontal_bounds_m": [h_low, h_low + horizontal_span],
            "vertical_bounds_m": [v_high - vertical_span, v_high],
            "candidate_points_in_window": int(np.count_nonzero(target_inside)),
            "context_points_in_window": int(np.count_nonzero(context_inside)),
            "support_line_drawn": support_drawn}
