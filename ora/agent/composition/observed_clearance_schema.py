"""The ``observed_clearance`` result field ``observed_clearance_reference_v2`` adds to motion targets.

``compiled_pose.targets.<arm>`` of ``move_pose`` and of the semantic primitives
that compile through it (``move_to``, ``rotate_wrist``, ``rotate_pitch``) gains
``observed_clearance``: the jaw footprint's lowest point at the commanded pose
minus the highest surface the current depth images show beneath it or within
``radius_m``, with where that surface was observed, or ``unknown`` with a
reason. The binder is ``observed_clearance_v2_schema``. Runtime half:
``agent/tools/observed_clearance.py`` (measurement) and
``agent/tools/observed_clearance_runtime.py`` (inputs).

EXP11A g012 (E11G12-004): 31 of 80 stalls drove the fingers or the held object
into a basket rim or wall; in 30 the commanded end pose itself put them below
the rim top, and admission checks only self-collision.
"""
from __future__ import annotations

import copy

TOOLS = ("move_pose", "move_to", "rotate_wrist", "rotate_pitch")
_XYZ = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
_HELD = {
    "type": ["object", "null"], "additionalProperties": False,
    "required": ["target_ref", "extent_included", "recentered_under_jaws"],
    "description": "null unless the jaws measure as holding: the target the policy named when they closed and "
                   "whether its localized extent is part of the footprint.",
    "properties": {"target_ref": {"type": ["string", "null"]}, "extent_included": {"type": "boolean"},
                   "recentered_under_jaws": {"type": "boolean"}},
}
_FOOTPRINT = {
    "footprint_lowest_m": dict(_XYZ), "footprint_lowest_part": {"type": "string", "enum": ["fingers", "held_object"]},
    "radius_m": {"type": "number", "exclusiveMinimum": 0},
    "footprint_observed_fraction": {"type": "number", "minimum": 0, "maximum": 1},
    "robot_points_excluded": {"type": "integer", "minimum": 0},
    "held_object": copy.deepcopy(_HELD), "claim_boundary": {"type": "string", "minLength": 1},
}
_SURFACE = {
    "type": "object", "additionalProperties": False,
    "required": ["position_m", "camera", "observation_id", "depth_points"],
    "properties": {"position_m": dict(_XYZ), "camera": {"type": "string"}, "observation_id": {"type": "string"},
                   "depth_points": {"type": "integer", "minimum": 1}},
}
PROPERTY = {
    "type": "object",
    "description": "Read-only, from the current depth images: the jaw footprint's lowest point at this target "
                   "minus the highest observed surface beneath it or within radius_m; negative means that surface "
                   "is higher. Not a collision check.",
    "oneOf": [
        {"additionalProperties": False,
         "required": ["status", "clearance_m", "highest_surface", *_FOOTPRINT],
         "properties": {"status": {"const": "measured"}, "clearance_m": {"type": "number"},
                        "highest_surface": copy.deepcopy(_SURFACE), **copy.deepcopy(_FOOTPRINT)}},
        {"additionalProperties": False, "required": ["status", "reason", *_FOOTPRINT],
         "properties": {"status": {"const": "unknown"},
                        "reason": {"type": "string", "enum": ["depth_unavailable", "footprint_unobserved"]},
                        **copy.deepcopy(_FOOTPRINT)}},
        {"additionalProperties": False, "required": ["status", "reason", "radius_m", "claim_boundary"],
         "properties": {"status": {"const": "unknown"},
                        "reason": {"type": "string", "enum": ["geometry_unavailable", "reference_failed"]},
                        "radius_m": {"type": "number", "exclusiveMinimum": 0},
                        "claim_boundary": {"type": "string", "minLength": 1}}},
    ],
}


__all__ = ["PROPERTY", "TOOLS"]
