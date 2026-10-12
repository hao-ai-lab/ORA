"""The move_pose result fields the planning-time IK reference adds.

``compiled_pose.targets.<arm>`` gains ``jaw_symmetry`` (which turn of the
parallel jaw was submitted, why, and both turns' joint6 margins) under
``move_pose_jaw_symmetry_headroom_v1`` and ``reach_along_request`` (whether
the target solved and, if not, how far along the straight interpolation the
same route solves) under ``move_pose_reach_along_request_v1``. The runtime
half is ``agent/tools/move_pose_reach.py``. Both bind only on a surface whose
move_pose compiles through ``enhanced_move_pose_v1``; the joint6 margin is an
ARX X5 fact, so the jaw turn binds only on the ``arx_x5_link6`` frame.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

JAW_CAPABILITY = "move_pose_jaw_symmetry_headroom_v1"
REACH_CAPABILITY = "move_pose_reach_along_request_v1"
JAW_REASONS = (
    "requested_turn_margin_sufficient", "larger_joint6_margin", "requested_turn_kept",
    "only_reachable_variant", "no_reachable_variant", "ik_reference_unavailable",
)
#: Fractions of the requested motion sampled when its target does not solve.
LINE_FRACTIONS = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875)

_NUMBER = {"type": ["number", "null"]}
_MARGIN = {"type": ["number", "null"], "minimum": 0}
JAW_PROPERTY = {
    "type": "object", "additionalProperties": False, "required": ["variant", "reason", "joint6_margin_rad"],
    "description": "Which turn of the parallel jaw this empty-jaw absolute move submitted: exact is the requested "
                   "orientation, jaw_flip the same jaw turned 180 degrees about the approach axis.",
    "properties": {
        "variant": {"type": "string", "enum": ["exact", "jaw_flip"]},
        "reason": {"type": "string", "enum": list(JAW_REASONS)},
        "joint6_margin_rad": {
            "type": "object", "additionalProperties": False, "required": ["exact"],
            "description": "Each checked turn's joint6 margin at its IK solution; null when unreachable or "
                           "joint6 is not proven the binding joint.",
            "properties": {"exact": dict(_MARGIN), "jaw_flip": dict(_MARGIN)},
        },
    },
}
_POSE = {
    "type": ["object", "null"], "additionalProperties": False, "required": ["position_m", "orientation_xyzw"],
    "properties": {"position_m": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                   "orientation_xyzw": {"type": "array", "items": {"type": "number"}, "minItems": 4,
                                        "maxItems": 4}},
}
REACH_PROPERTY = {
    "type": "object",
    "description": "Read-only IK of the submitted target on the route move_pose takes; when it does not solve, "
                   "the straight interpolation from the measured pose is sampled.",
    "oneOf": [
        {"additionalProperties": False, "required": ["target_reachable"],
         "properties": {"target_reachable": {"const": True}}},
        {"additionalProperties": False,
         "required": ["target_reachable", "reason", "requested_translation_m", "requested_rotation_rad",
                      "farthest_reachable_fraction", "farthest_reachable_translation_m",
                      "farthest_reachable_pose", "samples", "claim_boundary"],
         "properties": {
             "target_reachable": {"const": False},
             "reason": {"type": ["string", "null"]},
             "requested_translation_m": dict(_NUMBER), "requested_rotation_rad": dict(_NUMBER),
             "farthest_reachable_fraction": {"type": "number", "enum": [0.0, *LINE_FRACTIONS]},
             "farthest_reachable_translation_m": dict(_NUMBER),
             "farthest_reachable_pose": copy.deepcopy(_POSE),
             "samples": {"type": "array", "minItems": len(LINE_FRACTIONS), "maxItems": len(LINE_FRACTIONS),
                         "items": {"type": "object", "additionalProperties": False,
                                   "required": ["fraction", "reachable", "reason"],
                                   "properties": {"fraction": {"type": "number"}, "reachable": {"type": "boolean"},
                                                  "reason": {"type": ["string", "null"]}}}},
             "claim_boundary": {"type": "string", "minLength": 1},
         }},
        {"additionalProperties": False, "required": ["status", "failure"],
         "properties": {"status": {"const": "unavailable"}, "failure": {"type": "string"}}},
    ],
}


def bind_move_pose_reach(
    results: dict[str, Any], capabilities: Sequence[str], view_config: Mapping[str, Any] | None,
) -> dict[str, str]:
    """Add the declared annotations to move_pose's compiled_pose targets; identify the derivation."""
    from .tool_surface import ToolSurfaceError

    declared = [name for name in (JAW_CAPABILITY, REACH_CAPABILITY) if name in capabilities]
    if "enhanced_move_pose_v1" not in capabilities:
        raise ToolSurfaceError(f"{', '.join(declared)} requires enhanced_move_pose_v1")
    if JAW_CAPABILITY in declared and (view_config or {}).get("controlled_frame_convention") != "arx_x5_link6":
        raise ToolSurfaceError(f"{JAW_CAPABILITY} requires the ARX X5 link6 controlled frame")
    move_pose = results.get("move_pose")
    compiled = move_pose.get("properties", {}).get("compiled_pose") if isinstance(move_pose, dict) else None
    if not isinstance(compiled, dict) or compiled.get("type") != "object":
        raise ToolSurfaceError(f"{', '.join(declared)} requires the move_pose compiled_pose result")
    targets = compiled.setdefault("properties", {}).setdefault("targets", {"type": "object"})
    target = targets.setdefault("additionalProperties", {"type": "object"})
    properties = target.setdefault("properties", {})
    for name, key, value in ((JAW_CAPABILITY, "jaw_symmetry", JAW_PROPERTY),
                             (REACH_CAPABILITY, "reach_along_request", REACH_PROPERTY)):
        if name not in declared:
            continue
        if key in properties:
            raise ToolSurfaceError(f"{name} would overwrite compiled_pose.targets.<arm>.{key}")
        properties[key] = copy.deepcopy(value)
    path = Path(__file__)
    return {
        "slot": "derived:move_pose_reach",
        "ref": "ora/agent/composition/move_pose_reach_schema.py",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


__all__ = [
    "JAW_CAPABILITY", "JAW_PROPERTY", "JAW_REASONS", "LINE_FRACTIONS", "REACH_CAPABILITY", "REACH_PROPERTY",
    "bind_move_pose_reach",
]
