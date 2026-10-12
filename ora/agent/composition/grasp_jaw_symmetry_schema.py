"""The grasp_plan result fields the parallel-jaw member choice adds.

Layered on the pooled IK annotation (``grasp_ik_reachability_schema.py``),
which must already be bound: each candidate's ``ik_reachability`` gains
``jaw_symmetry`` (which member of the parallel-jaw class the published poses
carry, why, and both members' joint6 margins) and ``joint_headroom`` (the
chosen member's joint-limit margins at its pregrasp and contact solutions),
and the result-level ``ik_reachability`` gains the rule and its claim
boundary. The runtime half is ``agent/tools/grasp_jaw_symmetry.py``.

The joint6 margin is an ARX X5 fact (wrist roll about the link6 approach
axis), so the capability binds only on a RoboDojo surface whose controlled
frame is ``arx_x5_link6``. A separate module keeps the released annotation's
derivation hash -- which three surfaces, RoboTwin included, publish -- intact.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from .grasp_ik_reachability_schema import CAPABILITY as IK_ANNOTATION_CAPABILITY


CAPABILITY = "grasp_plan_jaw_symmetry_headroom_v1"
RULE = "reachable_then_larger_joint6_margin_then_smaller_joint_delta_then_compiled"
VARIANTS = ["exact", "jaw_flip"]
STAGES = ("pregrasp", "approach")
REASONS = [
    "larger_joint6_margin", "larger_joint_limit_margin", "margin_tie_smaller_joint_delta",
    "margin_and_joint_delta_tie", "only_reachable_variant", "no_reachable_variant",
    "ik_reference_unavailable",
]
#: Margins or joint moves this close are one IK answer, not a preference.
MARGIN_TIE_RAD = 0.001
DELTA_TIE_RAD = 0.001
CLAIM_BOUNDARY = (
    "Joint-limit margins of the backend's read-only IK solutions for the published pregrasp "
    "and contact poses at planning time; joint6 is the X5 wrist roll about the approach axis "
    "and is reported only when the measured joints prove it is the binding joint. An IK "
    "reference, not a carry guarantee: execution solves again, and lifting, carrying or "
    "rotating the wrist after the grasp moves every margin."
)
_MARGIN = {"type": ["number", "null"], "minimum": 0}


def _headroom_stage(stage: str) -> dict[str, Any]:
    return {
        "type": ["object", "null"], "additionalProperties": False,
        "required": ["min_joint_limit_margin_rad", "joint6_margin_rad"],
        "description": f"Margins at the {stage} IK solution of the published member; null when it was not reachable.",
        "properties": {
            "min_joint_limit_margin_rad": {
                **_MARGIN, "description": "Smallest distance of any arm joint to its limit, in radians.",
            },
            "joint6_margin_rad": {
                **_MARGIN, "description": "Wrist-roll (joint6) distance to its +/-3.14 rad limit; null when "
                                          "the measured joints cannot prove joint6 is the binding joint.",
            },
        },
    }


CANDIDATE_PROPERTIES = {
    "jaw_symmetry": {
        "type": "object", "additionalProperties": False,
        "required": ["variant", "reason", "joint6_margin_rad"],
        "description": "Which member of the parallel-jaw class the published pregrasp_pose and contact_pose "
                       "carry: exact is the compiled orientation, jaw_flip the same grasp turned 180 degrees "
                       "about the approach axis.",
        "properties": {
            "variant": {"type": "string", "enum": list(VARIANTS)},
            "reason": {"type": "string", "enum": list(REASONS)},
            "joint6_margin_rad": {
                "type": "object", "additionalProperties": False, "required": list(VARIANTS),
                "description": "Each member's smaller joint6 margin over its pregrasp and contact solutions; "
                               "null when unreachable or not isolated.",
                "properties": {name: dict(_MARGIN) for name in VARIANTS},
            },
        },
    },
    "joint_headroom": {
        "type": "object", "additionalProperties": False, "required": list(STAGES),
        "description": "Planning-time IK joint-limit margins of the published member; not a carry guarantee.",
        "properties": {name: _headroom_stage(name) for name in STAGES},
    },
}
SUMMARY_PROPERTY = {
    "type": "object", "additionalProperties": False, "required": ["rule", "claim_boundary"],
    "properties": {
        "rule": {"type": "string", "const": RULE},
        "claim_boundary": {"type": "string", "minLength": 1},
    },
}


def bind_grasp_jaw_symmetry(
    results: dict[str, Any], capabilities: Sequence[str], view_config: Mapping[str, Any] | None,
) -> dict[str, str]:
    """Add the member choice to an annotated X5 grasp_plan result; identify the derivation."""
    from .tool_surface import ToolSurfaceError

    if IK_ANNOTATION_CAPABILITY not in capabilities or "robodojo_rgbd_v1" not in capabilities:
        raise ToolSurfaceError(f"{CAPABILITY} requires the RoboDojo pooled grasp IK annotation")
    if (view_config or {}).get("controlled_frame_convention") != "arx_x5_link6":
        raise ToolSurfaceError(f"{CAPABILITY} requires the ARX X5 link6 controlled frame")
    grasp = results.get("grasp_plan")
    candidate = grasp.get("$defs", {}).get("candidate") if isinstance(grasp, dict) else None
    annotation = (candidate.get("properties", {}).get("ik_reachability")
                  if isinstance(candidate, dict) else None)
    summary = grasp.get("properties", {}).get("ik_reachability") if isinstance(grasp, dict) else None
    for node in (annotation, summary):
        if (not isinstance(node, dict) or node.get("additionalProperties") is not False
                or not isinstance(node.get("properties"), dict)):
            raise ToolSurfaceError(f"{CAPABILITY} requires the bound grasp_plan ik_reachability contract")
    for node, additions in ((annotation, CANDIDATE_PROPERTIES), (summary, {"jaw_symmetry": SUMMARY_PROPERTY})):
        for key, value in additions.items():
            if key in node["properties"]:
                raise ToolSurfaceError(f"{CAPABILITY} would overwrite ik_reachability.{key}")
            node["properties"][key] = copy.deepcopy(value)
            node["required"] = [*node.get("required", []), key]
    path = Path(__file__)
    return {
        "slot": f"derived:{CAPABILITY}",
        "ref": "ora/agent/composition/grasp_jaw_symmetry_schema.py",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
