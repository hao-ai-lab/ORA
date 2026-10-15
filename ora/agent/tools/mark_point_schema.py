"""What a ``mark_point`` call may ask for and what it is promised back.

The bounds here are not style choices, so they live with the schemas that
publish them rather than being repeated at each use.

``MARK_POINT_MAX_BATCH`` is the one that matters. It is *derived* from
``MARK_POINT_BATCH_COLOURS`` rather than written down, because the colour table
is the actual constraint: one batched call returns one composite overlay, each
mark is drawn in its own colour, and the renderer cycles the table modulo its
length. A ninth mark would be drawn in a colour already spent and would stop
being attributable in the only image the call returns. Extending the table is
therefore what raises the bound; nothing else does. Eight covers 98.4% of the
731 single-observation mark groups observed across 222 sealed runs.

The schema shape follows the closed-surface rule, which exists because of
concrete failures. Nesting stops one level below the root: the model reads flat
keys off each entry rather than walking into a nested record. There is no
top-level ``oneOf``, and no ``$defs``/``$ref`` -- a recorded run lost a tool call to
a dangling reference, where the delivery layer dropped the definitions and the
model said outright that it could not see them. Every array declares
``maxItems``, every property carries units and frame in a non-empty
description, and every object closes with ``additionalProperties: false``.
"""

from __future__ import annotations

from typing import Any, Sequence

from .robot import ToolError

# Eight distinguishable overlay colours. The batch bound is derived from this
# table: a ninth mark would reuse the first colour and become unattributable.
MARK_POINT_BATCH_COLOURS = (
    (255, 64, 64),
    (64, 255, 255),
    (255, 220, 64),
    (96, 255, 96),
    (255, 128, 255),
    (96, 160, 255),
    (255, 160, 64),
    (255, 255, 255),
)

#: One call may measure at most as many pixels as the overlay has colours.
MARK_POINT_MAX_BATCH = len(MARK_POINT_BATCH_COLOURS)

#: Two independent measurements of one name may disagree by this much, in
#: metres, before the second is refused as ``inconsistent_mark``.
MARK_POINT_VERIFY_TOLERANCE_M = 0.025

#: Radius in pixels of the square depth-support window around a clicked pixel.
MARK_POINT_DEPTH_WINDOW_RADIUS = 2

#: Hash-bound manifests opt into the batched input form with this capability.
#: Until a surface declares it the batched shape is refused and only the
#: single-pixel form is reachable, so no frozen profile changes behavior.
MARK_POINT_BATCH_CAPABILITY = "mark_point_batch_v1"

#: Aggregation of several *parallel single-pixel* calls into one overlay. This
#: predates the batched input form and stays live: six registered manifests
#: declare it and the Native session still routes through it.
MARK_POINT_AGGREGATION_CAPABILITY = "batched_mark_overlay_v1"

POINTCLOUD_VIEWS = ("pointcloud_top", "pointcloud_front", "pointcloud_side")

# Every bound below is what keeps eight maximal entries clear of the
# 50,000-byte delivery threshold; tests/agent/perception/test_mark_point_batch.py measures the
# worst case they permit. The closed vocabularies matter as much as the
# lengths: an enum costs its longest member, a free string costs its cap.
_MAX_POINT_ID_CHARS = 64
_MAX_VIEW_NAME_CHARS = 32
_MAX_ERROR_MESSAGE_CHARS = 200
_MAX_OBSERVATION_ID_CHARS = 64
#: A point accumulates one source per measurement; the result reports the
#: distinct views, most recent first, capped so the array cannot grow without
#: bound over a long episode.
MAX_REPORTED_VIEWS = 8
DEPTH_METHOD = "local_supported_depth_v1"
UNMEASURED_METHOD = "not_measured"
WORKSPACE_USES = (
    "candidate_requires_motion_preview",
    "measurement_only_outside_action_envelope",
)


def batch_input_schema(view_names: Sequence[str]) -> dict[str, Any]:
    """Return the batched input schema with this surface's closed view set.

    The accepted views differ per embodiment, so the enum is filled in by the
    surface that binds this tool rather than frozen here.  Point-cloud views
    are deliberately absent: their axis accumulation is order dependent across
    views and belongs to the single-pixel form.

    Schema shape follows the closed-surface rule.  Nesting stops one level
    below the root, there is no top-level ``oneOf`` and no ``$defs``/``$ref``
    -- a dangling reference is what cost a recorded run, where the delivery layer
    dropped the definitions and the model said it could not see them -- every
    array declares ``maxItems`` and every property, nested ones included,
    carries units and frame in its description.  The two arrays carry a string
    and a fixed-length ``[u,v]`` pair, never an object, because a parallel
    array of records is the shape weak models get wrong.
    """

    views = [str(name) for name in view_names]
    if not views or len(views) != len(set(views)):
        raise ToolError("batched mark_point needs a closed unique view set")
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["view", "point_ids", "points_uv"],
        "properties": {
            "view": {
                "type": "string",
                "enum": views,
                "description": (
                    "Exact returned current RGB view containing every pixel in "
                    "this call. All [u,v] pairs are in this view's top-left "
                    "OpenCV pixel frame at the resolution that view was "
                    "returned at, and every resulting coordinate is in metres "
                    "in the world frame. A wrist view is valid calibration "
                    "only at its current post-motion pose: pixels read from a "
                    "wrist view before it moved are not current geometry."
                ),
            },
            "point_ids": {
                "type": "array",
                "minItems": 1,
                "maxItems": MARK_POINT_MAX_BATCH,
                "uniqueItems": True,
                "description": (
                    "One to eight stable names, index-aligned with points_uv: "
                    "point_ids[i] names the geometric point measured at "
                    "points_uv[i]. Must have the same length as points_uv or "
                    "the whole call is rejected. Names must be distinct within "
                    "one call: reusing a name already solved in this round "
                    "from another view verifies it against the stored "
                    "measurement, which is meaningless against a second pixel "
                    "of the same image."
                ),
                "items": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": _MAX_POINT_ID_CHARS,
                    "description": (
                        "One short stable name for this geometric point, at "
                        "most 64 characters, for example \"mug_rim_left\". "
                        "Dimensionless identifier, not a coordinate."
                    ),
                },
            },
            "points_uv": {
                "type": "array",
                "minItems": 1,
                "maxItems": MARK_POINT_MAX_BATCH,
                "description": (
                    "One to eight pixels to measure, index-aligned with "
                    "point_ids and measured in array order. Must have the same "
                    "length as point_ids or the whole call is rejected. Each "
                    "pixel is measured independently: one pixel outside the "
                    "image or without usable depth fails only its own entry "
                    "and leaves the rest solved."
                ),
                "items": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "description": (
                        "One pixel as [u,v] in integer pixels of the exact "
                        "returned view, top-left OpenCV origin: u from the "
                        "left edge, v from the top edge."
                    ),
                    "items": {
                        "type": "integer",
                        "minimum": 0,
                        "description": (
                            "Non-negative pixel coordinate in the exact "
                            "returned view's own resolution."
                        ),
                    },
                },
            },
        },
    }


_MARK_ENTRY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "source_index", "point_id", "point_uv", "status", "error_code",
        "error_message", "xyz_m", "views", "next_views", "colour_rgb",
        "verification_distance_m", "quality_method", "quality_status",
        "quality_depth_m", "quality_window_radius_px", "quality_valid_samples",
        "quality_supporting_samples", "quality_rejected_samples",
        "quality_max_support_deviation_m", "action_workspace_inside",
        "action_workspace_use",
    ],
    "properties": {
        "source_index": {
            "type": "integer", "minimum": 0,
            "maximum": MARK_POINT_MAX_BATCH - 1,
            "description": (
                "Zero-based index of the request entry this result measures: "
                "point_ids[source_index] and points_uv[source_index]. Present "
                "on every entry so attribution never depends on array "
                "position surviving delivery."
            ),
        },
        "point_id": {
            "type": "string", "minLength": 1, "maxLength": _MAX_POINT_ID_CHARS,
            "description": "The requested name for this geometric point.",
        },
        "point_uv": {
            "type": "array", "minItems": 2, "maxItems": 2,
            "description": (
                "The requested pixel [u,v] in integer pixels of the requested "
                "view, top-left OpenCV origin."
            ),
            "items": {
                "type": "integer", "minimum": 0,
                "description": "Pixel coordinate in the requested view.",
            },
        },
        "status": {
            "type": "string",
            "enum": ["solved", "verified", "pending", "failed"],
            "description": (
                "solved: this call measured the point and stored it. "
                "verified: the name was already solved from another view and "
                "this measurement agreed with it. pending: a point-cloud "
                "click constrained some axes but not all three. failed: this "
                "pixel produced no measurement; the other entries are "
                "unaffected."
            ),
        },
        "error_code": {
            "type": ["string", "null"],
            "enum": [
                "invalid_arguments", "invalid_depth",
                "insufficient_depth_support", "depth_discontinuity",
                "inconsistent_mark", "stale_point",
                "invalid_camera_calibration", None,
            ],
            "description": (
                "Null unless this entry failed. Names why this one pixel "
                "produced no measurement, using exactly the codes the "
                "single-pixel form raises for the same conditions: "
                "invalid_arguments for a pixel outside the view or a view that "
                "cannot be marked, invalid_depth at the camera far plane, "
                "insufficient_depth_support and depth_discontinuity near an "
                "edge, inconsistent_mark when two measurements of one name "
                "disagree by more than 25 mm, stale_point when the world moved "
                "under a pending name. error_message carries the distinction "
                "the shared code does not."
            ),
        },
        "error_message": {
            "type": ["string", "null"], "maxLength": _MAX_ERROR_MESSAGE_CHARS,
            "description": (
                "Null unless this entry failed. One sentence, at most 200 "
                "characters, naming the correction for this pixel."
            ),
        },
        "xyz_m": {
            "type": ["array", "null"], "minItems": 3, "maxItems": 3,
            "description": (
                "World position [x,y,z] in metres, or null when this entry did "
                "not solve. A single-view depth reading measures a visible "
                "surface, not an object centre."
            ),
            "items": {
                "type": "number",
                "description": "World-frame coordinate in metres.",
            },
        },
        "views": {
            "type": "array", "maxItems": MAX_REPORTED_VIEWS,
            "description": (
                "The distinct views this name has now been measured from, "
                "oldest first, at most the eight most recent. Dimensionless "
                "camera names."
            ),
            "items": {
                "type": "string", "maxLength": _MAX_VIEW_NAME_CHARS,
                "description": "One view name this measurement came from.",
            },
        },
        "next_views": {
            "type": "array", "maxItems": len(POINTCLOUD_VIEWS),
            "description": (
                "For a pending point-cloud mark, the complementary views that "
                "would still constrain the missing axes. Empty otherwise."
            ),
            "items": {
                "type": "string", "maxLength": _MAX_VIEW_NAME_CHARS,
                "description": "One complementary view name.",
            },
        },
        "colour_rgb": {
            "type": ["array", "null"], "minItems": 3, "maxItems": 3,
            "description": (
                "The [r,g,b] colour, 0-255 per channel, that draws this mark "
                "in the composite overlay. This is the attribution carrier: it "
                "lives in the result body, so it survives even if the image is "
                "dropped from a later request. Null when nothing was drawn."
            ),
            "items": {
                "type": "integer", "minimum": 0, "maximum": 255,
                "description": "One 0-255 colour channel.",
            },
        },
        "verification_distance_m": {
            "type": ["number", "null"], "minimum": 0,
            "description": (
                "For a verified entry, the distance in metres between the "
                "stored measurement and this one; for a point-cloud "
                "verification, the largest per-axis disagreement in metres. "
                "Null otherwise."
            ),
        },
        "quality_method": {
            "type": "string", "enum": [DEPTH_METHOD, UNMEASURED_METHOD],
            "description": (
                "Which depth estimator produced this entry, or "
                "\"not_measured\" when the entry failed before any depth was "
                "read. Dimensionless identifier."
            ),
        },
        "quality_status": {
            "type": "string",
            "enum": ["measured", "partial", "unmeasured"],
            "description": (
                "measured: a supported local surface was found. partial: depth "
                "was read but rejected, and the counters below say why. "
                "unmeasured: the entry failed before depth was read."
            ),
        },
        "quality_depth_m": {
            "type": ["number", "null"],
            "description": (
                "Measured distance in metres along the camera's optical axis "
                "at this pixel, or null when no depth was accepted."
            ),
        },
        "quality_window_radius_px": {
            "type": ["integer", "null"], "minimum": 0,
            "description": (
                "Radius in pixels of the square support window sampled around "
                "the pixel. Null when no depth was read."
            ),
        },
        "quality_valid_samples": {
            "type": ["integer", "null"], "minimum": 0,
            "description": (
                "Count of finite in-range depth samples in that window. Null "
                "when no depth was read."
            ),
        },
        "quality_supporting_samples": {
            "type": ["integer", "null"], "minimum": 0,
            "description": (
                "Count of those samples compatible with the clicked depth. A "
                "small count on a solved mark means a thin or grazing surface."
            ),
        },
        "quality_rejected_samples": {
            "type": ["integer", "null"], "minimum": 0,
            "description": (
                "Count of window samples at an incompatible depth. A large "
                "count means the pixel sits near a silhouette."
            ),
        },
        "quality_max_support_deviation_m": {
            "type": ["number", "null"], "minimum": 0,
            "description": (
                "Largest deviation in metres between a supporting sample and "
                "the reported depth. Null when no depth was accepted."
            ),
        },
        "action_workspace_inside": {
            "type": ["boolean", "null"],
            "description": (
                "Whether the measured point lies inside the command envelope, "
                "or null when this surface does not report it. This is a "
                "command-envelope test, never a reachability, collision or "
                "grasp claim."
            ),
        },
        "action_workspace_use": {
            "type": ["string", "null"], "enum": [*WORKSPACE_USES, None],
            "description": (
                "What the envelope test permits: a candidate still needing a "
                "motion preview, or a measurement outside the envelope. Null "
                "when this surface does not report it."
            ),
        },
    },
}

_IMAGE_BINDING_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "image_index", "role", "camera", "source_index", "candidate_id",
        "target_ref", "observation_step",
    ],
    "properties": {
        "image_index": {
            "type": "integer", "minimum": 0,
            "maximum": MARK_POINT_MAX_BATCH - 1,
            "description": (
                "Position of this image among the images returned by this "
                "call."
            ),
        },
        "role": {
            "type": "string",
            "enum": ["mark_batch_panel", "mark_point_feedback"],
            "description": (
                "mark_batch_panel: one view with every solved mark of this "
                "call drawn in its own colour_rgb and a colour legend."
            ),
        },
        "camera": {
            "type": "string", "maxLength": _MAX_VIEW_NAME_CHARS,
            "description": "The view this panel renders. Dimensionless name.",
        },
        "source_index": {
            "type": ["integer", "null"], "minimum": 0,
            "maximum": MARK_POINT_MAX_BATCH - 1,
            "description": (
                "The request entry this image depicts, or null when the panel "
                "depicts several marks at once. Never infer it from position: "
                "inline images are dropped one block at a time, so a surviving "
                "image may no longer sit where it was returned."
            ),
        },
        "candidate_id": {
            "type": "null",
            "description": (
                "Always null: mark_point measures a pixel, it does not select "
                "a candidate. Present so every perception tool's image binding "
                "has one shape."
            ),
        },
        "target_ref": {
            "type": "null",
            "description": (
                "Always null: a mark is a coordinate, not a target reference. "
                "Present so every image binding has one shape."
            ),
        },
        "observation_step": {
            "type": ["integer", "null"], "minimum": 0,
            "description": (
                "Observation step this panel was rendered from, or null when "
                "the step is not carried."
            ),
        },
    },
}

#: Public result contract for the batched form.  A surface that declares
#: ``MARK_POINT_BATCH_CAPABILITY`` must bind exactly this schema; the runtime
#: validates against it regardless, so the two cannot drift apart.
MARK_POINT_BATCH_RESULT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "success", "status", "view", "observation_id", "mark_count",
        "solved_count", "failed_count", "marks", "image_mapping",
    ],
    "properties": {
        "success": {
            "type": "boolean", "const": True,
            "description": (
                "The call itself ran. It says nothing about whether any pixel "
                "measured: read status and each entry's own status."
            ),
        },
        "status": {
            "type": "string", "enum": ["solved", "partial", "failed"],
            "description": (
                "solved: every entry produced a measurement or a pending "
                "constraint. partial: some entries failed and the rest stand. "
                "failed: no entry produced a measurement."
            ),
        },
        "view": {
            "type": "string", "maxLength": _MAX_VIEW_NAME_CHARS,
            "description": "The single view every pixel of this call was read from.",
        },
        "observation_id": {
            "type": "string", "maxLength": _MAX_OBSERVATION_ID_CHARS,
            "description": (
                "The observation these pixels and coordinates are bound to. "
                "Pixels from a different observation are not current geometry."
            ),
        },
        "mark_count": {
            "type": "integer", "minimum": 1, "maximum": MARK_POINT_MAX_BATCH,
            "description": "Number of requested pixels, equal to len(marks).",
        },
        "solved_count": {
            "type": "integer", "minimum": 0, "maximum": MARK_POINT_MAX_BATCH,
            "description": "How many entries reached solved, verified or pending.",
        },
        "failed_count": {
            "type": "integer", "minimum": 0, "maximum": MARK_POINT_MAX_BATCH,
            "description": "How many entries failed and would need a new pixel.",
        },
        "marks": {
            "type": "array", "minItems": 1, "maxItems": MARK_POINT_MAX_BATCH,
            "description": (
                "One entry per requested pixel, in request order and of the "
                "same length, each repeating its own source_index."
            ),
            "items": {
                **_MARK_ENTRY_SCHEMA,
                "description": (
                    "One measurement attempt: its own source_index, the name "
                    "and pixel it was asked for, its outcome, and the depth "
                    "evidence behind it."
                ),
            },
        },
        "image_mapping": {
            "type": "array", "minItems": 0, "maxItems": MARK_POINT_MAX_BATCH,
            "description": (
                "What each returned image shows, bound by source_index rather "
                "than by position."
            ),
            "items": {
                **_IMAGE_BINDING_SCHEMA,
                "description": (
                    "One returned image and what it depicts, bound by "
                    "source_index rather than by its position."
                ),
            },
        },
        "measurement_contract": {
            "type": "string", "maxLength": 512,
            "description": (
                "Units and evidence boundary for these measurements, added by "
                "surfaces that deliver it."
            ),
        },
    },
}
