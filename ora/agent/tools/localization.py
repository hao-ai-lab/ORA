"""Canonical observation-bound target localization shared by every consumer.

``grasp.segmented_target_geometry`` already derives a centroid, PCA principal
axes and per-axis extents from one bound RGB-D mask, and three separate callers
memoise it independently.  This module names that result a *localization*, so
that ``grasp.plan``, ``place.plan``, the gripper primitives and the closed
grasp/place skills quote one contract rather than three ad-hoc projections of
the same arrays.

Two measurement functions live here because they are subtractions of canonical
localizations rather than capabilities of their own:

``offset_between``
    The world offset between two same-frame targets.  This is the entire
    computation the separate ``vision.offset`` tool performed; keeping it as
    geometry rather than as a tool removes a call without removing a number.

``closure_evidence``
    The measured jaw separation placed beside the target's measured extents.
    Both quantities already existed and neither knew the other did.  It returns
    labelled evidence and deliberately no Boolean: see ``closure_evidence``.

Nothing here infers object identity, contact, attachment, retention or task
success.  Extents are of the *visible* surface only and under-measure a partly
occluded object.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .robot import ToolError


SCHEMA = "ora.target-localization.v1"
OFFSET_SCHEMA = "ora.target-offset.v1"
CLOSURE_SCHEMA = "ora.closure-extent-evidence.v1"

#: Jaw separation at or below which the jaws are treated as having met with
#: nothing between them.  3 mm is the ``near_closed`` bound the Panda aperture
#: projection in the manipulation tool runtime already publishes; a surface whose embodiment
#: differs must pass its own value rather than inherit this one silently.
NEAR_CLOSED_M = 0.003

CLAIM_BOUNDARY = (
    "current visible RGB-D mask geometry only; extents are of the visible "
    "surface and under-measure hidden extent; not object identity, pose, "
    "contact, attachment, retention or official success"
)


def _finite_vector(value: Any, size: int, name: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ToolError(f"{name} is unavailable", code="target_geometry_unavailable")
    values = list(value)
    if len(values) != size:
        raise ToolError(f"{name} must hold {size} values", code="target_geometry_unavailable")
    result = []
    for item in values:
        if type(item) not in (int, float) or not math.isfinite(float(item)):
            raise ToolError(f"{name} is not finite", code="target_geometry_unavailable")
        result.append(float(item))
    return result


def localize(geometry: Mapping[str, Any]) -> dict[str, Any]:
    """Project one ``segmented_target_geometry`` result into the shared contract.

    The input is the mask-bound geometry; the output is the closed, source-free
    shape every consumer quotes.  ``principal_extents_m`` keeps its principal
    variance order, because that is what pairs each extent with its axis in
    ``principal_axes_world``.  Variance order is not size order: a real target
    measured [0.1034, 0.0704, 0.0751], where index 2 is not the smallest
    dimension.  The named largest/middle/smallest are therefore taken from the
    sizes themselves, so a consumer quoting them never has to know which
    convention produced the array.
    """

    if not isinstance(geometry, Mapping):
        raise ToolError("target geometry is unavailable", code="target_geometry_unavailable")
    centroid = _finite_vector(geometry.get("centroid_world_m"), 3, "centroid_world_m")
    extents = _finite_vector(geometry.get("principal_extents_m"), 3, "principal_extents_m")
    axes_raw = geometry.get("principal_axes_world")
    if not isinstance(axes_raw, Sequence) or len(list(axes_raw)) != 3:
        raise ToolError("principal_axes_world is unavailable", code="target_geometry_unavailable")
    axes = [_finite_vector(axis, 3, "principal_axes_world") for axis in axes_raw]
    for name in ("observation_id", "target_ref", "source_camera"):
        if not isinstance(geometry.get(name), str) or not geometry[name]:
            raise ToolError(f"localization {name} is unavailable", code="target_geometry_unavailable")
    result = {
        "schema": SCHEMA,
        "observation_id": geometry["observation_id"],
        "target_ref": geometry["target_ref"],
        "source_camera": geometry["source_camera"],
        "centroid_world_m": centroid,
        "principal_axes_world": axes,
        "principal_extents_m": extents,
        # Named so a consumer never has to remember the ordering convention.
        "largest_extent_m": max(extents),
        "middle_extent_m": sorted(extents)[1],
        "smallest_extent_m": min(extents),
        "visible_point_count": int(geometry.get("visible_point_count") or 0),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    # v2 geometry carries measured mask/depth quality; v1 does not. Read
    # defensively so an older record does not fail an otherwise valid call.
    if "status" in geometry:
        result["status"] = geometry["status"]
        result["quality_failures"] = list(geometry.get("quality_failures") or ())
    return result


def offset_between(
    a: Mapping[str, Any], b: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the world offset from localization ``a`` to localization ``b``.

    Both localizations must bind the same observation and the same source
    camera.  Two centroids measured through different cameras see different
    halves of their objects, so their difference is not a physical offset; that
    refusal is the whole reason the retired ``vision.offset`` tool enforced a
    shared frame, and it is preserved here rather than re-derived by callers.
    """

    for name, value in (("a", a), ("b", b)):
        if not isinstance(value, Mapping) or value.get("schema") != SCHEMA:
            raise ToolError(f"{name} must be a canonical localization", code="invalid_arguments", field=name)
    if a["target_ref"] == b["target_ref"]:
        raise ToolError("an offset needs two different targets", code="invalid_arguments", field="b")
    if a["observation_id"] != b["observation_id"]:
        raise ToolError(
            "both targets must bind the current observation",
            code="stale_observation", field="b",
        )
    if a["source_camera"] != b["source_camera"]:
        raise ToolError(
            "both targets must come from the same camera frame and depth source",
            code="cross_camera_targets", field="b",
        )
    translation = [b["centroid_world_m"][i] - a["centroid_world_m"][i] for i in range(3)]
    return {
        "schema": OFFSET_SCHEMA,
        "observation_id": a["observation_id"],
        "source_camera": a["source_camera"],
        "from_target_ref": a["target_ref"],
        "to_target_ref": b["target_ref"],
        "translation_world_m": translation,
        "distance_m": math.sqrt(sum(v * v for v in translation)),
        "basis": "difference_of_two_visible_mask_centroids_in_one_camera_frame",
        "claim_boundary": (
            "visible-surface centroids are not material points; this offset "
            "moves with the visible face, not with the object's centre of mass"
        ),
    }


def measured_jaw_separation_m(record: Any) -> float | None:
    """The measured jaw separation of one end effector, or ``None``.

    Two backends publish the same physical quantity in different shapes.
    RoboTwin publishes ``gripper_opening_m`` directly. LIBERO publishes
    ``gripper_finger_qpos``: two prismatic finger displacements whose summed
    magnitude is the separation, which is the conversion
    ``agent/context/public_facts.py`` already applies when it reports
    ``measured_aperture_mm`` to the policy. Reading only the first key left the
    absorbed verify stage reporting ``unavailable`` on every LIBERO episode
    while the measurement sat one field away in the same record.

    The normalized ``gripper`` value in that record is a command, not a length,
    and is never a substitute for either.
    """

    if not isinstance(record, Mapping):
        return None
    published = record.get("gripper_opening_m")
    if type(published) in (int, float) and math.isfinite(float(published)):
        return float(published)
    if record.get("pose_frame") == "robodojo_episode_local_ee_link":
        return None  # ARX q=0 has a source mesh gap; encoder sum is not aperture.
    fingers = record.get("gripper_finger_qpos")
    if not isinstance(fingers, list) or len(fingers) != 2:
        return None
    if not all(
        type(q) in (int, float) and math.isfinite(float(q)) for q in fingers
    ):
        return None
    return sum(abs(float(q)) for q in fingers)


def closure_evidence(
    *, gripper_opening_m: Any, localization: Mapping[str, Any] | None,
    near_closed_m: float = NEAR_CLOSED_M,
) -> dict[str, Any]:
    """Put the measured jaw separation beside the target's measured extents.

    These are the two quantities that discriminate the two failure modes the
    episode self-reports are dominated by — closing on nothing, and closing on
    something that then slips.  Both were already computed and neither consumer
    knew the other existed.

    The result is labelled by *strength* and carries no Boolean:

    ``strong_negative``
        The jaws met at or below ``near_closed_m``.  Nothing of the target's
        measured thickness is between them.  This is the one direction in which
        the pair is close to conclusive.

    ``weak_negative``
        The separation exceeds the target's largest *visible* extent.  Visible
        extents under-measure a partly occluded object, so this is suggestive
        and no more.

    ``not_contradictory``
        The separation falls inside the target's measured extent range.  This
        proves nothing: a jaw can rest at an object's width while holding it,
        beside it, or against a neighbouring object.  It is reported so the
        absence of a strong negative is visible, not so it can be read as a
        positive.

    ``unavailable``
        One of the two quantities was not measured.  Both backends do publish a
        jaw separation -- RoboTwin as ``gripper_opening_m``, LIBERO as two
        finger displacements -- so read it through
        :func:`measured_jaw_separation_m` rather than one backend's key.  What
        is genuinely absent is the target's extents when no localization was
        bound.  Saying so is the point: a silent omission reads as a passed
        check.
    """

    if not isinstance(near_closed_m, (int, float)) or not math.isfinite(float(near_closed_m)) or near_closed_m < 0:
        raise ToolError("near_closed_m must be a finite non-negative length", code="invalid_arguments")
    opening = None
    if type(gripper_opening_m) in (int, float) and math.isfinite(float(gripper_opening_m)):
        opening = float(gripper_opening_m)
    extents = None
    if isinstance(localization, Mapping) and localization.get("schema") == SCHEMA:
        extents = list(localization["principal_extents_m"])
    result: dict[str, Any] = {
        "schema": CLOSURE_SCHEMA,
        "gripper_opening_m": opening,
        "target_extents_m": extents,
        "target_largest_extent_m": None if extents is None else max(extents),
        "target_smallest_extent_m": None if extents is None else min(extents),
        "near_closed_m": float(near_closed_m),
        # The jaw-closing direction in the target's principal frame is not
        # measured, so the comparison is against the whole extent range rather
        # than a single axis chosen for it.
        "compared_against": "full_visible_extent_range_jaw_axis_unknown",
        "strength": "unavailable",
        "reason": "measured_jaw_separation_and_target_extents_are_not_both_available",
        "claim_boundary": (
            "jaw separation beside visible extents is evidence, not a holding "
            "verdict; no value of this pair establishes contact or retention"
        ),
    }
    if opening is None or extents is None:
        result["missing"] = [
            name for name, value in
            (("gripper_opening_m", opening), ("target_extents_m", extents))
            if value is None
        ]
        return result
    result.pop("missing", None)
    if opening <= float(near_closed_m):
        result.update(
            strength="strong_negative",
            reason="jaws_met_with_less_than_the_near_closed_bound_between_them",
        )
    elif opening > max(extents):
        result.update(
            strength="weak_negative",
            reason="jaw_separation_exceeds_the_largest_visible_extent_which_under_measures_hidden_extent",
        )
    else:
        result.update(
            strength="not_contradictory",
            reason="jaw_separation_lies_inside_the_measured_visible_extent_range_and_proves_nothing",
        )
    return result
