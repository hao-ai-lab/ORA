"""The nearest reachable orientation an unsolved move target carries.

Under ``reach_nearest_orientation_v1`` (which requires
``move_pose_reach_along_request_v1``) every ``reach_along_request`` that says
the target does not solve -- ``move_pose`` and, where
``pose_primitive_reach_along_request_v1`` is declared, ``move_to``,
``rotate_wrist`` and ``rotate_pitch`` -- also carries
``nearest_reachable_orientation``: the reachable turn of the requested
orientation with the smallest angle, holding the nominal jaw centre (the
controlled point plus the descriptor's jaw offset) where the request put it.
Candidates turn the requested orientation about axes perpendicular to its
approach axis -- for a straight-down request, horizontal axes -- in
``TURN_DIRECTIONS`` evenly spaced directions at each of ``ROTATIONS_DEG``,
so the approach tilts by exactly that angle and the jaw keeps its roll about
the approach. Each rotation is one read-only reach-check request on the
target's own route from the measured joints; the sweep stops at the first
rotation with a reachable candidate. The submitted action never changes.
Runtime half: ``agent/tools/reach_orientation.py``.

In one recorded RoboDojo batch, an offline replica of the runtime IK (99.8-100%
agreement over 1,066 recorded solves) found every placement target reachable
by some arm, many only with a 15-75 degree tilt; the policy tried
straight-down poses and concluded "unreachable", and every feedback channel
answered only the requested orientation.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

CAPABILITY = "reach_nearest_orientation_v1"
REQUIRES = "move_pose_reach_along_request_v1"
PRIMITIVE_CAPABILITY = "pose_primitive_reach_along_request_v1"
TOOLS = ("move_pose", "move_to", "rotate_wrist", "rotate_pitch")
#: Turn angles from the requested orientation, smallest first.
ROTATIONS_DEG = (15, 30, 45, 60, 75)
#: Evenly spaced turn axes about the requested approach axis at each angle.
TURN_DIRECTIONS = 12
#: The most IK queries one unsolved target can add.
MAXIMUM_QUERIES = len(ROTATIONS_DEG) * TURN_DIRECTIONS
CLAIM = (
    "Read-only IK at each candidate endpoint on the route this move takes, from the measured joints: the "
    "requested orientation turned about axes perpendicular to its approach axis, holding the nominal jaw "
    "centre; no path, collision, payload or arrival claim, and an orientation outside the sweep may solve."
)

_VECTOR3 = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
_QUERIES = {"type": "integer", "minimum": 0, "maximum": MAXIMUM_QUERIES}
NEAREST_PROPERTY = {
    "type": "object",
    "description": "The reachable turn of the requested orientation with the smallest angle, holding the nominal "
                   "jaw centre, from read-only IK at the endpoint.",
    "oneOf": [
        {"additionalProperties": False,
         "required": ["status", "rotation_from_request_deg", "tilt_from_straight_down_deg", "approach_world",
                      "pose", "jaw_center_m", "reachable_turns_at_rotation", "queries", "claim_boundary"],
         "properties": {
             "status": {"const": "found"},
             "rotation_from_request_deg": {"type": "number", "enum": list(ROTATIONS_DEG)},
             "tilt_from_straight_down_deg": {"type": "number", "minimum": 0, "maximum": 180},
             "approach_world": copy.deepcopy(_VECTOR3),
             "pose": {"type": "object", "additionalProperties": False, "required": ["position_m", "orientation_xyzw"],
                      "properties": {"position_m": copy.deepcopy(_VECTOR3),
                                     "orientation_xyzw": {"type": "array", "items": {"type": "number"},
                                                          "minItems": 4, "maxItems": 4}}},
             "jaw_center_m": copy.deepcopy(_VECTOR3),
             "reachable_turns_at_rotation": {"type": "integer", "minimum": 1, "maximum": TURN_DIRECTIONS},
             "queries": copy.deepcopy(_QUERIES),
             "claim_boundary": {"type": "string", "minLength": 1},
         }},
        {"additionalProperties": False,
         "required": ["status", "largest_rotation_deg", "jaw_center_m", "queries", "claim_boundary"],
         "properties": {
             "status": {"const": "none_within_sweep"},
             "largest_rotation_deg": {"const": ROTATIONS_DEG[-1]},
             "jaw_center_m": copy.deepcopy(_VECTOR3),
             "queries": copy.deepcopy(_QUERIES),
             "claim_boundary": {"type": "string", "minLength": 1},
         }},
        {"additionalProperties": False, "required": ["status", "failure"],
         "properties": {"status": {"const": "unavailable"}, "failure": {"type": "string"}}},
    ],
}


def _unreachable_variant(result: Any, name: str) -> dict[str, Any]:
    from .tool_surface import ToolSurfaceError

    try:
        reach = (result["properties"]["compiled_pose"]["properties"]["targets"]["additionalProperties"]
                 ["properties"]["reach_along_request"])
        variant = next(item for item in reach["oneOf"]
                       if item.get("properties", {}).get("target_reachable") == {"const": False})
    except (KeyError, TypeError, StopIteration) as error:
        raise ToolSurfaceError(f"{CAPABILITY} requires the {name} reach_along_request result") from error
    if "nearest_reachable_orientation" in variant["properties"]:
        raise ToolSurfaceError(f"{CAPABILITY} would overwrite {name} nearest_reachable_orientation")
    return variant


def bind_reach_nearest_orientation(results: dict[str, Any], capabilities: Sequence[str],
                                   view_config: Mapping[str, Any] | None) -> dict[str, str]:
    """Add nearest_reachable_orientation to every declared unsolved reach_along_request."""
    from .tool_surface import ToolSurfaceError

    if REQUIRES not in capabilities:
        raise ToolSurfaceError(f"{CAPABILITY} requires {REQUIRES}")
    if not isinstance((view_config or {}).get("public_geometry_descriptor"), Mapping):
        raise ToolSurfaceError(f"{CAPABILITY} requires the public geometry descriptor for the jaw centre")
    names = ["move_pose", *([name for name in TOOLS[1:] if name in results]
                            if PRIMITIVE_CAPABILITY in capabilities else [])]
    for name in names:
        variant = _unreachable_variant(results.get(name), name)
        variant["properties"]["nearest_reachable_orientation"] = copy.deepcopy(NEAREST_PROPERTY)
        variant["required"] = [*variant["required"], "nearest_reachable_orientation"]
    path = Path(__file__)
    return {
        "slot": f"derived:{CAPABILITY}",
        "ref": "ora/agent/composition/reach_orientation_schema.py",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


__all__ = [
    "CAPABILITY", "CLAIM", "MAXIMUM_QUERIES", "NEAREST_PROPERTY", "PRIMITIVE_CAPABILITY", "REQUIRES",
    "ROTATIONS_DEG", "TOOLS", "TURN_DIRECTIONS", "bind_reach_nearest_orientation",
]
