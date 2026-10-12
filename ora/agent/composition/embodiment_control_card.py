"""Control-only nine-node embodiment cards and shared public text vocabulary.

The caller validates descriptor structure and provenance. Rendering preserves
source-bound wording without asserting unavailable physical contact geometry.
"""
from __future__ import annotations

from typing import Any, Mapping


_EVIDENCE_LIMIT_TEXT = {
    "physical_gripper_geometry_unavailable": "Physical jaw aperture and flange-to-contact/TCP transform are unavailable; do not infer them from the encoder or apply a Panda grip-site offset.",
    "motion_completion_not_contact": (
        "Motion completion does not establish contact."
    ),
    "gripper_closure_not_contact": (
        "A commanded gripper closure does not identify contact."
    ),
    "gripper_closure_not_retention": (
        "Gripper closure or aperture alone does not prove object retention."
    ),
    "geometry_not_collision_guarantee": (
        "A geometric target is not an executed state or a collision guarantee."
    ),
}

_CONTROLLED_POINT_TEXT = {
    "grip_center": "the model-defined point centered between the gripper fingers",
    "tool_center": "the model-defined tool center point",
    "wrist_flange": "the wrist flange reference frame",
}

_STATE_FIELD_TEXT = {
    "controlled_point_pose": "the measured controlled-point pose",
    "jaw_separation": "the measured full jaw separation",
    "rotary_joint_positions": "the measured rotary joint positions",
    "robotiq_2f85_rotary_encoder_opening": "the measured rotary encoder opening (clipped 1-q/(pi/4), where q is finger_joint in radians; 1=open, 0=closed), not physical jaw width",
}

_STATE_UNIT_TEXT = {
    "world_metre_xyzw": "world-frame metres and an `xyzw` quaternion",
    "millimetre": "millimetres",
    "robot_root_metre_xyzw": "robot-root-frame metres and an `xyzw` quaternion",
    "radian": "radians",
    "dimensionless": "dimensionless normalized units",
    "tool_surface_declared": "the units declared by the current ToolSurface",
}


def render_control_only_card(descriptor: Mapping[str, Any]) -> str:
    """Nine-node card for measured control without asserted contact geometry."""
    from .embodiment import EmbodimentDescriptorError, REVIEW_TOKEN_LIMIT

    if descriptor["geometry"]["grippers"]:
        raise EmbodimentDescriptorError("control-only card cannot assert gripper geometry")
    control = descriptor["control"]
    sensing = descriptor["sensing"]
    arms = ", ".join(f"{a['side']} {a['joint_dof']}-DoF {a['kind'].replace('_', ' ')}" for a in descriptor["topology"]["manipulators"])
    lines = ["# Embodiment Card v1", "", "## Identity and topology", "",
             f"{descriptor['display_name']}. {descriptor['identity']['base_mobility'].capitalize()} base; {arms}.",
             "", "## Controlled points", ""]
    for point in control["controlled_points"]:
        lines.append(f"`{point['id']}` is {_CONTROLLED_POINT_TEXT[point['description']]}; commands target this reference directly.")
    lines += ["", "## Frames and units", "",
              f"Positions use {control['position_frame']} {control['position_unit']}; pose quaternions use `{control['orientation']['order']}`. Rotary joints use radians.",
              "", "## End-effector axes", "",
              "Contact approach/closing axes and flange-to-contact/TCP transform are unavailable. Do not infer a contact center from the controlled frame.",
              "", "## Motion capabilities", "",
              f"Action abstraction: {control['action_abstraction']}. The ToolSurface owns exact arguments, windows and limits. Holding a command does not freeze measured state.",
              "", "## Gripper semantics", ""]
    for field in sensing["state_fields"]:
        if field["meaning"] != "controlled_point_pose":
            names = (" Joint order: `" + ", ".join(field["joint_names"]) + "`." if "joint_names" in field else "")
            lines.append(f"`{field['name']}` reports {_STATE_FIELD_TEXT[field['meaning']]} in {_STATE_UNIT_TEXT[field['unit']]}.{names}")
    lines += ["Physical jaw aperture is unavailable.", "", "## Sensors and cameras", ""]
    for camera in sensing["cameras"]:
        moving = camera["mounting"] in {"manipulator", "end_effector"}
        lines.append(f"`{camera['public_name']}`: {'moving' if moving else 'fixed'} RGB-D; use calibration from the same observation.")
    lines += ["", "## Preservation and evidence limits", ""]
    lines.extend(_EVIDENCE_LIMIT_TEXT[name] for name in descriptor["evidence_limits"])
    lines += ["", "## Dynamic facts and provenance", "",
              "Current pose, encoder measurements, calibration and budgets belong to current observation/Runtime. Source hashes remain manifest-only; this card does not qualify physical contact geometry."]
    rendered = "\n".join(lines) + "\n"
    if (len(rendered.encode()) + 3) // 4 > REVIEW_TOKEN_LIMIT:
        raise EmbodimentDescriptorError("control-only card exceeds review token limit")
    return rendered
