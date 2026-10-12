"""State each repeated item once per payload (``within_payload_deduplication_v1``).

A RoboDojo action-result observation carried the controller rule twice (463
characters, in the current facts and in the action effect), each arm's gripper
feedback twice (about 1,160 characters), the same convention strings once per
camera and once per arm, and a per-segment IK journal in the effect that the
projector's own docstring calls Control-only.

On a declaring surface:

* the current public robot facts own ``controller_feedback`` and each arm's
  ``gripper.feedback``; the action effect names that owner instead of a copy;
* a convention string repeated by every camera or every arm is stated once,
  under ``conventions["cameras.<camera>"]`` or ``conventions["end_effectors.<arm>"]``;
* the five convention strings Native ``current_actual.camera_geometry`` repeats
  per camera are stated once, under
  ``current_actual.conventions["camera_geometry.<camera>"]``;
* RoboDojo ``arms.<arm>.segments`` leaves the effect.

Every reference stays inside one payload: the effect and the facts it names
are siblings in the same Native observation envelope, and nothing points at
an earlier observation. Context tail aging replaces an aged ``current_actual``
by a marker and a recoverable pointer while the effect stays inline, so for an
aged step the named feedback is read from that recoverable observation; the
exact near window keeps both inline. Only literally equal values are stated
once; state that merely coincides (two open grippers) stays with its arm.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

CAPABILITY = "within_payload_deduplication_v1"

#: Where the effect's owner lives in the Native observation envelope. Effects
#: are delivered only by Native, beside ``current_actual``.
FACTS_FIELD = "current_actual.public_robot_facts"
REPORTED_IN_FACTS = "reported_in_current_public_robot_facts"
CONVENTIONS_KEY = "conventions"

#: Convention fields per repeated entry, by path inside the entry. Explicit:
#: measured or commanded state never qualifies, even when two arms agree.
CONVENTION_FIELDS: Mapping[str, tuple[str, tuple[tuple[str, ...], ...]]] = {
    "cameras": ("<camera>", (
        ("pixel_coordinates",), ("resampling",),
        ("calibration", "status"), ("calibration", "pixel_basis"),
        ("calibration", "camera_axes"), ("calibration", "world_position_unit"),
    )),
    "end_effectors": ("<arm>", (
        ("position_frame",), ("position_unit",), ("orientation_order",), ("controlled_frame",),
        ("controlled_point_semantics",),
        ("gripper", "grasp_evidence"), ("gripper", "measurement_semantics"),
        ("gripper", "opening_semantics"),
        ("gripper", "feedback", "drive_target_source"), ("gripper", "feedback", "measurement_source"),
        ("gripper", "feedback", "pose_hold_semantics"),
        ("gripper", "feedback", "arrival_check", "comparison"),
    )),
}

#: Native ``current_actual.camera_geometry`` (``current_visual.camera_geometry``):
#: the per-camera strings, never the matrices or the derived origin and axes.
CAMERA_GEOMETRY_CONVENTION_FIELDS: Mapping[str, tuple[str, tuple[tuple[str, ...], ...]]] = {
    "camera_geometry": ("<camera>", (
        ("pixel_coordinates",), ("camera_coordinates",), ("extrinsic_convention",),
        ("depth_semantics",), ("depth_unit",),
    )),
}

_MISSING = object()


def _get(entry: Any, path: tuple[str, ...]) -> Any:
    for step in path:
        if not isinstance(entry, Mapping) or step not in entry:
            return _MISSING
        entry = entry[step]
    return entry


def _pop(entry: dict[str, Any], path: tuple[str, ...]) -> None:
    parents = [entry]
    for step in path[:-1]:
        parents.append(parents[-1][step])
    del parents[-1][path[-1]]
    # A parent emptied by the move holds nothing the conventions do not state.
    for parent, step in zip(reversed(parents[:-1]), reversed(path[:-1])):
        if parent[step]:
            break
        del parent[step]


def _put(target: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    for step in path[:-1]:
        target = target.setdefault(step, {})
    target[path[-1]] = value


def hoist_fact_conventions(facts: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of the facts with each convention shared by every entry stated once."""
    return _hoist(facts, CONVENTION_FIELDS)


def hoist_camera_geometry_conventions(current_actual: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of a Native ``current_actual`` with camera conventions stated once."""
    return _hoist(current_actual, CAMERA_GEOMETRY_CONVENTION_FIELDS)


def _hoist(value: Mapping[str, Any],
           table: Mapping[str, tuple[str, tuple[tuple[str, ...], ...]]]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    conventions: dict[str, Any] = {}
    for collection, (placeholder, fields) in table.items():
        entries = result.get(collection)
        if not isinstance(entries, Mapping) or len(entries) < 2:
            continue
        shared: dict[str, Any] = {}
        for path in fields:
            values = [_get(entry, path) for entry in entries.values()]
            first = values[0]
            if isinstance(first, str) and all(value == first for value in values):
                _put(shared, path, first)
                for entry in entries.values():
                    _pop(entry, path)
        if shared:
            conventions[f"{collection}.{placeholder}"] = shared
    if conventions:
        result[CONVENTIONS_KEY] = conventions
    return result


def point_effect_to_facts(effect: Mapping[str, Any]) -> dict[str, Any]:
    """The effect with feedback the current facts own replaced by its owner's field."""
    detail = effect.get("detail")
    if not isinstance(detail, Mapping):
        return dict(effect)
    delivered = dict(detail)
    if isinstance(detail.get("controller_feedback"), Mapping):
        delivered["controller_feedback"] = {
            "status": REPORTED_IN_FACTS, "field": f"{FACTS_FIELD}.controller_feedback"}
    grippers = detail.get("gripper_feedback")
    if isinstance(grippers, Mapping) and grippers:
        delivered["gripper_feedback"] = {
            "status": REPORTED_IN_FACTS,
            "field": f"{FACTS_FIELD}.end_effectors.<arm>.gripper.feedback",
            "arms": sorted(grippers)}
    arms = detail.get("arms")
    if isinstance(arms, Mapping):
        delivered["arms"] = {
            arm: ({key: value for key, value in state.items() if key != "segments"}
                  if isinstance(state, Mapping) else state)
            for arm, state in arms.items()}
    return {**effect, "detail": delivered}


__all__ = [
    "CAMERA_GEOMETRY_CONVENTION_FIELDS", "CAPABILITY", "CONVENTIONS_KEY", "CONVENTION_FIELDS",
    "FACTS_FIELD", "REPORTED_IN_FACTS", "hoist_camera_geometry_conventions",
    "hoist_fact_conventions", "point_effect_to_facts",
]
