"""Measuring up to eight pixels of one view in one call.

Two properties decide the shape of the result, and both are consequences of how
the context layer delivers it rather than of taste.

**The 50,000-byte cliff.** Above ``eager_large_result_bytes`` the projection
layer replaces a tool result with a 2,048-byte prefix and an instruction to
repeat the call. A batch that trips it is worse than no batch: the model pays
for the call and gets back the first two marks and a suggestion. Every field
below is bounded so that eight maximal entries cannot approach the threshold,
and ``tests/agent/perception/test_mark_point_batch.py`` measures the worst case rather than
trusting this note.

**Images are evicted one block at a time.** The inline-image window drops
individual image blocks by position, never a tool result's images as a group, so
"the i-th image belongs to the i-th input" is a convention that silently
misattributes survivors. This call therefore returns one composite overlay and
puts the attribution in the *body*: each entry carries the ``colour_rgb`` it was
drawn in, which no eviction can reach, and ``image_mapping`` names the
``source_index`` a panel depicts or null when it depicts several.

A surface that declares the batch capability publishes *only* this result
shape. Admitting the four-field single-pixel result and this one from a single
contract would need a top-level ``oneOf``, which the closed-surface rule
forbids, so an opted-in surface expresses one pixel as a batch of one and every
surface that has not opted in keeps the four-field form exactly as it was. The
runtime accepts both shapes; only a published contract has to choose.

The third property is per-item attributability. ``marks`` has the same length
and order as ``point_ids``, every entry repeats its own ``source_index``, and a
pixel that cannot be measured fails only its own entry: one bad pixel never
costs the other seven their measurement. Every entry carries depth-support
diagnostics on success *and* on failure, which the single-pixel form discards on
its error path -- the remedy for ``depth_discontinuity`` (aim further inside the
surface) is not the remedy for ``invalid_depth`` (the pixel sees the far plane),
and a model correcting several pixels at once has to tell them apart.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .mark_point import measure
from .mark_point_overlay import compose_batch_feedback, render_panels
from .mark_point_schema import (
    MARK_POINT_BATCH_CAPABILITY,
    MARK_POINT_BATCH_COLOURS,
    MARK_POINT_BATCH_RESULT_SCHEMA,
    MARK_POINT_MAX_BATCH,
    MAX_REPORTED_VIEWS,
    POINTCLOUD_VIEWS,
    UNMEASURED_METHOD,
    _MAX_ERROR_MESSAGE_CHARS,
    _MAX_POINT_ID_CHARS,
)
from .robot import ToolError

# Conditions that belong to one pixel. Anything else -- the observation
# disappearing mid-call, for instance -- is a property of the whole call and
# must fail it rather than be recorded as one entry's problem, which would also
# put a code outside the published enum into the result.
_ENTRY_ERROR_CODES = frozenset({
    "invalid_arguments",
    "invalid_depth",
    "insufficient_depth_support",
    "depth_discontinuity",
    "inconsistent_mark",
    "stale_point",
    # invalid_camera_calibration is not one pixel's problem: the published
    # calibration is infrastructure and fails the whole call.
})


def _reject(
    message: str, *, field: str | None = None, received: Any = None,
    **constraints: Any,
) -> ToolError:
    """A model-correctable call error, in the shared structured shape.

    ``field`` and ``received`` go to their own slots rather than into
    ``constraints``: the projection layer carries all three to the provider,
    and a caller reading ``error.field`` should not have to know that this tool
    hid it one level down.
    """

    return ToolError(
        message, code="invalid_arguments", retryable=True,
        field=field, received=received, constraints=constraints or None,
    )


def _batch_arguments(
    runtime: Any, arguments: Mapping[str, Any],
) -> tuple[str, list[str], list[tuple[int, int]]]:
    """Validate the whole batch before measuring any of it.

    A malformed *call* is rejected as a call; a pixel that cannot be measured
    is not.  The difference matters: the model can only fix a length mismatch
    or a duplicate name by rewriting the call, whereas it fixes a bad pixel by
    re-aiming that one entry.
    """

    if not runtime.surface.supports(MARK_POINT_BATCH_CAPABILITY):
        raise _reject(
            "this tool surface measures one pixel per mark_point call; call "
            "it once per point with point_id, view, x and y",
            accepted_fields=["point_id", "view", "x", "y"],
        )
    if set(arguments) != {"view", "point_ids", "points_uv"}:
        raise _reject(
            "batched mark_point takes exactly view, point_ids and points_uv",
            accepted_fields=["view", "point_ids", "points_uv"],
            received=sorted(arguments),
        )
    view = arguments["view"]
    if not isinstance(view, str) or not view:
        raise _reject("view must be a non-empty view name", field="view")
    if view in POINTCLOUD_VIEWS:
        raise _reject(
            "a point-cloud view accumulates axis constraints across views, so "
            "it stays on the single-pixel form; batch a camera view instead",
            field="view", received=view,
        )
    raw_ids, raw_uv = arguments["point_ids"], arguments["points_uv"]
    for field, value in (("point_ids", raw_ids), ("points_uv", raw_uv)):
        if not isinstance(value, list) or not value:
            raise _reject(f"{field} must be a non-empty array", field=field)
        if len(value) > MARK_POINT_MAX_BATCH:
            raise _reject(
                f"{field} accepts at most {MARK_POINT_MAX_BATCH} entries, one "
                "per distinguishable overlay colour",
                field=field, received=len(value), maximum=MARK_POINT_MAX_BATCH,
            )
    if len(raw_ids) != len(raw_uv):
        raise _reject(
            "point_ids and points_uv must have the same length: entry i names "
            "the point measured at pixel i",
            point_ids=len(raw_ids), points_uv=len(raw_uv),
        )
    point_ids: list[str] = []
    for index, value in enumerate(raw_ids):
        if (
            not isinstance(value, str) or not value
            or len(value) > _MAX_POINT_ID_CHARS
        ):
            raise _reject(
                "each point_id must be a name of 1 to "
                f"{_MAX_POINT_ID_CHARS} characters",
                field=f"point_ids[{index}]", received=value,
                minLength=1, maxLength=_MAX_POINT_ID_CHARS,
            )
        point_ids.append(value)
    if len(set(point_ids)) != len(point_ids):
        repeat = next(index for index, value in enumerate(point_ids) if value in point_ids[:index])
        raise _reject(
            f"point_ids must be distinct within one call: {point_ids[repeat]!r} "
            f"repeats at indices {point_ids.index(point_ids[repeat])} and {repeat}",
            field=f"point_ids[{repeat}]", received=point_ids[repeat],
            first_index=point_ids.index(point_ids[repeat]),
        )
    points_uv: list[tuple[int, int]] = []
    for index, value in enumerate(raw_uv):
        if not isinstance(value, (list, tuple)) or len(value) != 2 or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in value
        ):
            raise _reject(
                "each pixel must be [u,v] with non-negative integers in the "
                "selected view's own top-left OpenCV frame",
                field=f"points_uv[{index}]",
            )
        points_uv.append((int(value[0]), int(value[1])))
    return view, point_ids, points_uv


def _entry(
    *, source_index: int, point_id: str, point_uv: tuple[int, int],
    quality: Mapping[str, Any],
) -> dict[str, Any]:
    """One uniformly shaped result entry, so every key is always readable."""

    entry = {
        "source_index": source_index,
        "point_id": point_id,
        "point_uv": [point_uv[0], point_uv[1]],
        "status": "failed",
        "error_code": None,
        "error_message": None,
        "xyz_m": None,
        "views": [],
        "next_views": [],
        "colour_rgb": None,
        "verification_distance_m": None,
        "quality_method": UNMEASURED_METHOD,
        "quality_status": "unmeasured",
        "quality_depth_m": None,
        "quality_window_radius_px": None,
        "quality_valid_samples": None,
        "quality_supporting_samples": None,
        "quality_rejected_samples": None,
        "quality_max_support_deviation_m": None,
        "action_workspace_inside": None,
        "action_workspace_use": None,
    }
    entry.update({
        key: value for key, value in quality.items() if key in entry
    })
    return entry


def execute_batch(runtime: Any, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Measure up to eight pixels of one view and return one overlay."""

    from ora.agent.composition import validate_schema_instance

    from .runtime.core import _png_base64

    view, point_ids, points_uv = _batch_arguments(runtime, arguments)
    control, root, observation, _pose, _gripper = runtime._current_grip_state()
    reports_workspace = runtime.surface.runtime_id == "native_atomic_causal"

    marks: list[dict[str, Any]] = []
    for index, (point_id, pixel) in enumerate(zip(point_ids, points_uv)):
        sink: dict[str, Any] = {}
        try:
            outcome = measure(
                runtime, point_id=point_id, view=view,
                x=pixel[0], y=pixel[1],
                suppress_images=True, quality_sink=sink,
            )
        except ToolError as error:
            if error.code not in _ENTRY_ERROR_CODES:
                raise
            entry = _entry(
                source_index=index, point_id=point_id, point_uv=pixel,
                quality=sink,
            )
            entry["error_code"] = error.code
            entry["error_message"] = str(error)[:_MAX_ERROR_MESSAGE_CHARS]
            marks.append(entry)
            continue
        entry = _entry(
            source_index=index, point_id=point_id, point_uv=pixel,
            quality=sink,
        )
        entry["status"] = str(outcome["status"])
        entry["xyz_m"] = list(outcome["xyz_m"]) if "xyz_m" in outcome else None
        entry["views"] = _distinct_views(outcome.get("views", []))
        entry["next_views"] = [
            str(name) for name in outcome.get("next_views", [])
        ]
        workspace = outcome.get("action_workspace")
        if reports_workspace and isinstance(workspace, Mapping):
            entry["action_workspace_inside"] = bool(workspace["inside"])
            entry["action_workspace_use"] = str(workspace["motion_use"])
        marks.append(entry)

    _attach_stored_evidence(runtime, marks)

    selected: list[tuple[str, list[float], tuple[int, int, int]]] = []
    for entry in marks:
        if entry["status"] not in {"solved", "verified"} or not entry["xyz_m"]:
            continue
        colour = MARK_POINT_BATCH_COLOURS[
            len(selected) % len(MARK_POINT_BATCH_COLOURS)
        ]
        entry["colour_rgb"] = list(colour)
        selected.append((entry["point_id"], list(entry["xyz_m"]), colour))

    solved = sum(entry["status"] != "failed" for entry in marks)
    result: dict[str, Any] = {
        "success": True,
        "status": (
            "solved" if solved == len(marks)
            else "failed" if solved == 0 else "partial"
        ),
        "view": view,
        "observation_id": str(control["observation_id"]),
        "mark_count": len(marks),
        "solved_count": solved,
        "failed_count": len(marks) - solved,
        "marks": marks,
        "image_mapping": [],
    }

    if selected:
        rendered = render_panels(
            runtime, root=root, observation=observation, views=[view],
            selected=selected,
        )
        if rendered:
            composite = compose_batch_feedback(
                runtime, observation_id=str(control["observation_id"]),
                panels=rendered, labelled_points=selected,
            )
            # One image for the whole batch, so eight marks cost one slot of
            # the inline-image window instead of eight. source_index is only
            # honest when the panel draws a single mark; otherwise the colours
            # in each entry carry the attribution.
            drawn = [
                mark["source_index"] for mark in marks
                if mark["colour_rgb"] is not None
            ]
            only = drawn[0] if len(drawn) == 1 else None
            result["images_base64"] = [_png_base64(composite)]
            result["image_mapping"] = [{
                "image_index": 0,
                "role": "mark_batch_panel",
                "camera": rendered[0][0],
                "source_index": only,
                "candidate_id": None,
                "target_ref": None,
                "observation_step": _observation_step(control),
            }]

    validate_schema_instance(
        {
            key: value for key, value in result.items()
            if key not in {"image_base64", "images_base64", "_runtime"}
        },
        MARK_POINT_BATCH_RESULT_SCHEMA,
    )
    return result


def _distinct_views(values: Sequence[Any]) -> list[str]:
    """Report which views measured this name, not how many times each did.

    ``sources`` grows by one entry per measurement, so a name re-verified
    through a long episode would otherwise return an unbounded array of
    repeats. The distinct names are the decision-relevant part, and the tail is
    capped so the entry cannot drift over its declared bound.
    """

    seen: list[str] = []
    for value in values:
        name = str(value)
        if name not in seen:
            seen.append(name)
    return seen[-MAX_REPORTED_VIEWS:]


def _observation_step(control: Mapping[str, Any]) -> int | None:
    """The step this panel was rendered at, or null when it is not carried."""

    state = control.get("policy_state")
    step = state.get("step") if isinstance(state, Mapping) else None
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        return None
    return step


def _attach_stored_evidence(
    runtime: Any, marks: Sequence[dict[str, Any]],
) -> None:
    """Fill in what the stored record knows, reading the point file once.

    Two entries need it. A verified entry knows it agreed with a stored
    measurement but not by how much. And re-marking the exact pixel of a point
    already solved in this observation short-circuits before the estimator
    runs, so the entry is genuinely solved while the estimator reported
    nothing: reporting "solved" beside "unmeasured" would be a contradiction,
    when the honest answer is the quality of the measurement being reused.
    """

    reusing = [
        mark for mark in marks
        if mark["status"] in {"solved", "verified"}
        and mark["quality_status"] == "unmeasured"
    ]
    verified = [mark for mark in marks if mark["status"] == "verified"]
    if not reusing and not verified:
        return
    points = runtime._load_points()

    for mark in verified:
        record = points.get(mark["point_id"])
        verification = record.get("last_verification") if isinstance(
            record, Mapping,
        ) else None
        if not isinstance(verification, Mapping):
            continue
        for key in ("distance_m", "axis_max_error_m"):
            value = verification.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                mark["verification_distance_m"] = float(value)
                break

    for mark in reusing:
        record = points.get(mark["point_id"])
        sources = record.get("sources") if isinstance(record, Mapping) else None
        quality = next(
            (
                source["measurement_quality"]
                for source in reversed(sources or [])
                if isinstance(source, Mapping)
                and isinstance(source.get("measurement_quality"), Mapping)
            ),
            None,
        )
        if quality is None:
            continue
        mark.update({
            "quality_status": "measured",
            "quality_method": str(quality.get("method", UNMEASURED_METHOD)),
            "quality_depth_m": quality.get("depth_m"),
            "quality_window_radius_px": quality.get("window_radius_px"),
            "quality_valid_samples": quality.get("valid_samples"),
            "quality_supporting_samples": quality.get("supporting_samples"),
            "quality_rejected_samples": quality.get("rejected_samples"),
            "quality_max_support_deviation_m": quality.get(
                "max_support_deviation_m",
            ),
        })
