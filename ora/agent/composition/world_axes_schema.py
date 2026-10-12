"""The move_pose orientation mode and pose-target echo of ``pose_orientation_from_world_axes_v1``.

The binder adds ``world_axes`` to every arm target's ``orientation_mode`` enum
in ``move_pose`` with its two companion fields ``approach_world`` and
``jaw_line_world`` (descriptions compiled from the motion reader's own mode
table by ``motion_arguments.annotate_mode_companions``), names the mode where
the description lists the modes, and adds ``target_axes_world`` to the
``compiled_pose.targets.<arm>`` result of ``move_pose``, ``move_to``,
``rotate_wrist`` and ``rotate_pitch``. The runtime half is
``agent/tools/world_axes.py``; the policy-facing sentence is the surface's own
help layer.

The axes are the surface's declared controlled-frame convention
(``view_config.controlled_frame_convention``, resolved by
``embodiment_geometry.public_local_axes``): link6 local +X and +Y on the ARX X5.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

CAPABILITY = "pose_orientation_from_world_axes_v1"
MODE = "world_axes"
COMPANIONS = ("approach_world", "jaw_line_world")
TOOLS = ("move_pose", "move_to", "rotate_wrist", "rotate_pitch")
REQUIRED_CAPABILITIES = ("enhanced_move_pose_v1", "flat_motion_arguments_v1", "motion_mode_companion_help_v1")
_VECTOR = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
TARGET_AXES_PROPERTY = {
    "type": "object", "additionalProperties": False, "required": ["approach", "jaw_line"],
    "description": "World directions of the approach axis and jaw line of the submitted target orientation, "
                   "computed from its compiled quaternion.",
    "properties": {"approach": copy.deepcopy(_VECTOR), "jaw_line": copy.deepcopy(_VECTOR)},
}
#: Exact released phrases the description states the modes in, and their bound form.
DESCRIPTION_EDITS = (
    ("orientation_mode (hold_current, absolute_quaternion, axis_angle_delta)",
     "orientation_mode (hold_current, absolute_quaternion, axis_angle_delta, world_axes)"),
    ("absolute_quaternion takes orientation_xyzw;",
     "absolute_quaternion takes orientation_xyzw; world_axes takes approach_world and jaw_line_world;"),
)


def _error(message: str) -> Exception:
    from .tool_surface import ToolSurfaceError

    return ToolSurfaceError(f"{CAPABILITY} {message}")


def _check_surface(capabilities: Sequence[str], view_config: Mapping[str, Any] | None) -> None:
    from .embodiment_geometry import PUBLIC_GEOMETRY_DESCRIPTORS

    for required in REQUIRED_CAPABILITIES:
        if required not in capabilities:
            raise _error(f"requires {required}")
    if (view_config or {}).get("controlled_frame_convention") not in PUBLIC_GEOMETRY_DESCRIPTORS:
        raise _error("requires a declared controlled_frame_convention with public local axes")


def _with_companions(properties: dict[str, Any]) -> dict[str, Any]:
    """The arm target's properties with the two vectors placed after the released orientation fields."""
    if all(name in properties for name in COMPANIONS):
        return properties
    anchor = "orientation_frame" if "orientation_frame" in properties else "orientation_mode"
    placed: dict[str, Any] = {}
    for key, value in properties.items():
        placed[key] = value
        if key == anchor:
            placed.update({name: copy.deepcopy(_VECTOR) for name in COMPANIONS})
    return placed


def _bind_arm_targets(schema: Any) -> int:
    """Add the mode to every flat arm target under ``schema``; return how many were bound."""
    if not isinstance(schema, dict):
        return 0
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return 0
    bound = 0
    if "orientation_mode" in properties and "position_mode" in properties:
        enum = properties["orientation_mode"].setdefault("enum", [])
        if MODE not in enum:
            enum.append(MODE)
        schema["properties"] = properties = _with_companions(properties)
        bound += 1
    return bound + sum(_bind_arm_targets(value) for value in properties.values())


def _edit_description(text: str) -> str:
    for released, bound in DESCRIPTION_EDITS:
        if bound in text:
            continue
        if text.count(released) != 1:
            raise _error(f"cannot name the mode: {released!r} occurs {text.count(released)} times")
        text = text.replace(released, bound)
    return text


def bind_world_axes_definitions(
    definitions: list[dict[str, Any]], capabilities: Sequence[str], view_config: Mapping[str, Any] | None,
) -> None:
    """Advertise ``world_axes`` in move_pose's input schema and description (idempotent)."""
    from ..tools.motion_arguments import annotate_mode_companions

    _check_surface(capabilities, view_config)
    entries = [item for item in definitions if item.get("name") == "move_pose"]
    if len(entries) != 1:
        raise _error("requires one declared move_pose")
    definition = entries[0]
    if not _bind_arm_targets(definition.get("inputSchema")):
        raise _error("requires the flat move_pose arm targets")
    annotate_mode_companions([definition])
    definition["description"] = _edit_description(str(definition.get("description", "")))


def bind_world_axes_results(results: dict[str, Any]) -> None:
    """Add ``target_axes_world`` to every declared pose tool's compiled_pose targets."""
    for name in TOOLS:
        result = results.get(name)
        if result is None and name != "move_pose":
            continue
        compiled = result.get("properties", {}).get("compiled_pose") if isinstance(result, dict) else None
        if not isinstance(compiled, dict) or compiled.get("type") != "object":
            raise _error(f"requires the {name} compiled_pose result")
        targets = compiled.setdefault("properties", {}).setdefault("targets", {"type": "object"})
        properties = targets.setdefault("additionalProperties", {"type": "object"}).setdefault("properties", {})
        if "target_axes_world" in properties:
            raise _error(f"would overwrite {name} target_axes_world")
        properties["target_axes_world"] = copy.deepcopy(TARGET_AXES_PROPERTY)


def bind_world_axes_orientation(
    definitions: list[dict[str, Any]], results: dict[str, Any], capabilities: Sequence[str],
    view_config: Mapping[str, Any] | None,
) -> dict[str, str]:
    """Bind the mode and the echo; identify the derivation."""
    bind_world_axes_definitions(definitions, capabilities, view_config)
    bind_world_axes_results(results)
    path = Path(__file__)
    return {
        "slot": f"derived:{CAPABILITY}",
        "ref": "ora/agent/composition/world_axes_schema.py",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


__all__ = [
    "CAPABILITY", "COMPANIONS", "DESCRIPTION_EDITS", "MODE", "REQUIRED_CAPABILITIES", "TARGET_AXES_PROPERTY",
    "TOOLS", "bind_world_axes_definitions", "bind_world_axes_orientation", "bind_world_axes_results",
]
