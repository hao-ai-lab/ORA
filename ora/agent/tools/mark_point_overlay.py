"""Drawing the marks of one call into one image, and who gets that image.

Two callers converge here. The batched input form measures up to eight pixels
in one call and wants one overlay for all of them. The older aggregation path
catches several *parallel single-pixel* calls issued in one assistant response
and gives them one overlay between them. Both end at the same contact sheet, so
the renderer and the colour assignment live in one place.

The aggregation path is kept deliberately. A proposal to batch the input argued
that it becomes dead code worth roughly 200 lines; it does not. Six registered
profiles declare ``batched_mark_overlay_v1`` -- the three
``native_atomic_causal`` revisions and three ``openeta_light`` ones -- and
``native/session.py`` still routes parallel read-only calls through it, which
``tests/agent/perception/test_mark_point_batch.py`` demonstrates by driving the whole path
through a Native session. Deleting it would have silently changed those
surfaces from one shared overlay to one image per call.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .mark_point_schema import (
    MARK_POINT_AGGREGATION_CAPABILITY,
    MARK_POINT_BATCH_COLOURS,
    POINTCLOUD_VIEWS,
)
from .robot import ToolError, _append


def render_panels(
    runtime: Any, *, root: Path, observation: Mapping[str, Any],
    views: Sequence[str],
    selected: Sequence[tuple[str, list[float], tuple[int, int, int]]],
) -> list[tuple[str, Path]]:
    """Draw every selected mark, in its own colour, on each named view."""

    rendered: list[tuple[str, Path]] = []
    pointcloud = None
    for view in views:
        source_view = "agentview" if view == "candidate_agentview" else view
        if source_view in observation.get("cameras", {}):
            panel = runtime._annotated_rgb(
                root=root, observation=observation, camera=source_view,
                labelled_world_points=list(selected),
                identity_kind="mark_point_batch_panel",
            )
        elif source_view in POINTCLOUD_VIEWS:
            pointcloud = pointcloud or runtime._pointcloud(root, observation)
            panel = runtime._annotated_pointcloud(
                observation=observation, pointcloud=pointcloud,
                view=source_view, labelled_world_points=list(selected),
                identity_kind="mark_point_batch_panel",
            )
        else:
            continue
        rendered.append((source_view, panel))
    return rendered


def compose_batch_feedback(
    runtime: Any, *, observation_id: str, panels: Sequence[tuple[str, Path]],
    labelled_points: Sequence[tuple[str, list[float], tuple[int, int, int]]],
) -> Path:
    """Render one bounded contact sheet for one assistant mark batch."""

    from PIL import Image, ImageDraw

    from .runtime.core import _artifact_record

    identity = {
        "kind": "mark_point_batch_feedback",
        "observation_id": observation_id,
        "panels": [
            {
                "view": view,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for view, path in panels
        ],
        "points": [
            {"point_id": point_id, "xyz_m": xyz, "colour": list(colour)}
            for point_id, xyz, colour in labelled_points
        ],
        "output_resolution": list(runtime._derived_resolution()),
    }
    encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True).encode()
    target = runtime.artifact_root / hashlib.sha256(
        observation_id.encode()
    ).hexdigest()[:20] / "overlays" / (
        hashlib.sha256(encoded).hexdigest()[:24] + ".png"
    )

    width, height = runtime._derived_resolution()
    count = len(panels)
    columns = 1 if count == 1 else math.ceil(math.sqrt(count))
    rows = math.ceil(count / columns)
    cell_width = width // columns
    cell_height = height // rows
    canvas = Image.new("RGB", (width, height), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    for index, (view, path) in enumerate(panels):
        panel = Image.open(path).convert("RGB")
        panel.thumbnail(
            (max(1, cell_width), max(1, cell_height)),
            Image.Resampling.LANCZOS,
        )
        column = index % columns
        row = index // columns
        left = column * cell_width + (cell_width - panel.width) // 2
        top = row * cell_height + (cell_height - panel.height) // 2
        canvas.paste(panel, (left, top))
        draw.rectangle(
            (column * cell_width, row * cell_height,
             column * cell_width + min(cell_width, 8 + 7 * len(view)),
             row * cell_height + 14),
            fill=(0, 0, 0),
        )
        draw.text(
            (column * cell_width + 3, row * cell_height + 2),
            view,
            fill=(255, 255, 255),
        )

    legend_columns = min(4, len(labelled_points))
    legend_rows = math.ceil(len(labelled_points) / legend_columns)
    legend_height = 16 * legend_rows
    legend_top = height - legend_height
    legend_width = width // legend_columns
    draw.rectangle((0, legend_top, width, height), fill=(0, 0, 0))
    for index, (point_id, _xyz, colour) in enumerate(labelled_points):
        column = index % legend_columns
        row = index // legend_columns
        left = column * legend_width
        top = legend_top + row * 16
        draw.rectangle((left + 3, top + 4, left + 10, top + 11), fill=colour)
        draw.text((left + 14, top + 2), point_id, fill=colour)

    stream = io.BytesIO()
    canvas.save(stream, format="PNG")
    expected = stream.getvalue()
    if target.is_file():
        if target.read_bytes() != expected:
            raise ToolError(
                "batched mark feedback artifact hash drift",
                code="artifact_hash_mismatch", retryable=False,
            )
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp.png")
        temporary.write_bytes(expected)
        os.replace(temporary, target)
        target.chmod(0o444)
    _append(runtime.event_path, {
        "type": "openeta_view_derived",
        "observation_id": observation_id,
        "identity": identity,
        "artifact": _artifact_record(target, runtime.control_root.parent),
        "time": time.time(),
    })
    return target


def begin_feedback_batch(
    runtime: Any, name: str, calls: Sequence[Mapping[str, Any]],
) -> bool:
    """Suppress per-call images while parallel single-pixel calls execute.

    This is the older answer to the same cost: several ``mark_point`` calls
    issued in one assistant response still each pay a tool round trip, but they
    share one overlay instead of returning one image each.  The batched input
    form above is strictly better, but this path stays: six registered
    manifests declare ``batched_mark_overlay_v1`` and the Native session still
    routes through it, so deleting it would silently change those surfaces.
    """

    if name != "mark_point":
        return False
    if not runtime.surface.supports(MARK_POINT_AGGREGATION_CAPABILITY):
        return False
    if runtime._active_mark_point_batch is not None:
        raise ToolError("mark_point feedback batch is already active")
    normalized = tuple(dict(call) for call in calls)
    if len(normalized) < 2 or any(
        set(call) != {"point_id", "view", "x", "y"} for call in normalized
    ):
        return False
    runtime._active_mark_point_batch = normalized
    return True


def finish_feedback_batch(
    runtime: Any, name: str, results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return at most one composite image for the active mark batch."""

    from .runtime.core import _png_base64

    if name != "mark_point":
        raise ToolError("active read-only feedback batch changed tool identity")
    calls = runtime._active_mark_point_batch
    runtime._active_mark_point_batch = None
    if calls is None:
        return {}
    if len(calls) != len(results):
        raise ToolError("mark_point feedback batch result count drift")
    control, root, observation, _pose, _gripper = runtime._current_grip_state()
    points = runtime._load_points()
    selected: list[tuple[str, list[float], tuple[int, int, int]]] = []
    views: list[str] = []
    for call, result in zip(calls, results):
        point_id = str(call["point_id"])
        record = points.get(point_id)
        if not (
            result.get("success") is True
            and result.get("status") in {"solved", "verified"}
            and isinstance(record, Mapping)
            and record.get("status") == "solved"
            and isinstance(record.get("xyz_m"), list)
        ):
            continue
        colour = MARK_POINT_BATCH_COLOURS[
            len(selected) % len(MARK_POINT_BATCH_COLOURS)
        ]
        selected.append((point_id, list(record["xyz_m"]), colour))
        view = str(call["view"])
        if view not in views:
            views.append(view)
    if not selected:
        return {}
    rendered = render_panels(
        runtime, root=root, observation=observation, views=views,
        selected=selected,
    )
    if not rendered:
        return {}
    composite = compose_batch_feedback(
        runtime, observation_id=str(control["observation_id"]),
        panels=rendered, labelled_points=selected,
    )
    return {"images_base64": [_png_base64(composite)]}
