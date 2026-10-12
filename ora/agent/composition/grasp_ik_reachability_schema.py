"""The grasp_plan result fields the pooled IK reachability selection adds.

Derived rather than copied: the Dojo and RoboTwin grasp_plan result schemas are
different files (the Dojo one is backend-owned), and a copy of either to add two
properties is a copy that can drift from its source. The stage reason
vocabulary is the closed per-backend set in ``contracts/reach_check.py`` -- the
same one the controller admits from that backend's internal read-only IK check
-- selected by the surface's backend identity capability.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from ora.contracts.reach_check import (
    ROBODOJO_REASONS, ROBOTWIN_REASONS, SHARED_REASONS,
)


CAPABILITY = "grasp_plan_ik_reachability_annotation_v1"
ORDER = "ik_reachable_then_approach_within_30_deg_of_vertical_then_backend_order"
STATUSES = ["annotated", "unavailable", "no_candidate"]
MEMBERS = ["exact", "jaw_flip", None]
POOL_LIMIT = 32
#: Backend identity capability -> the refusal reasons that backend's IK check returns.
BACKEND_REASONS = {"robodojo_rgbd_v1": ROBODOJO_REASONS, "robotwin_rgbd_v1": ROBOTWIN_REASONS}


def _stage(reasons: list[Any]) -> dict[str, Any]:
    return {
        "type": ["object", "null"], "additionalProperties": False,
        "required": ["reachable", "reason"],
        "description": "Read-only IK answer for this stage of the reported member (the exact compiled orientation when none was admitted); null when the annotation was unavailable.",
        "properties": {
            "reachable": {"type": "boolean"},
            "reason": {"type": ["string", "null"], "enum": reasons},
        },
    }


def bind_grasp_ik_reachability(
    results: dict[str, Any], capabilities: Sequence[str],
) -> dict[str, str]:
    """Add the annotation to the surface's grasp_plan result; identify the derivation."""
    from .tool_surface import ToolSurfaceError

    grasp = results.get("grasp_plan")
    if not isinstance(grasp, dict):
        raise ToolSurfaceError(f"{CAPABILITY} requires a grasp_plan result schema")
    backends = [name for name in BACKEND_REASONS if name in capabilities]
    if len(backends) != 1:
        raise ToolSurfaceError(f"{CAPABILITY} requires exactly one backend with a read-only IK check")
    reasons = [*SHARED_REASONS, *BACKEND_REASONS[backends[0]], None]
    candidate = grasp.get("$defs", {}).get("candidate")
    if (not isinstance(candidate, dict) or candidate.get("additionalProperties") is not False
            or not isinstance(candidate.get("properties"), dict)
            or grasp.get("additionalProperties") is not False):
        raise ToolSurfaceError(f"{CAPABILITY} requires the closed grasp_plan candidate contract")
    per_candidate = {
        "type": "object", "additionalProperties": False,
        "required": ["reachable", "member", "pregrasp", "approach", "approach_tilt_from_vertical_deg"],
        "description": "IK reference at planning time; null reachable means the annotation was unavailable.",
        "properties": {
            "reachable": {"type": ["boolean", "null"]},
            "member": {"type": ["string", "null"], "enum": list(MEMBERS)},
            "pregrasp": _stage(reasons),
            "approach": _stage(reasons),
            "approach_tilt_from_vertical_deg": {"type": "number", "minimum": 0, "maximum": 180},
        },
    }
    summary = {
        "type": "object", "additionalProperties": False,
        "required": ["status", "ik_model", "matches_move_pose_admission", "order",
                     "checked_candidate_count", "reachable_candidate_count", "claim_boundary"],
        "properties": {
            "status": {"type": "string", "enum": list(STATUSES)},
            "ik_model": {"type": ["string", "null"]},
            "matches_move_pose_admission": {"type": ["boolean", "null"]},
            "order": {"type": "string", "const": ORDER},
            "checked_candidate_count": {"type": "integer", "minimum": 0, "maximum": POOL_LIMIT},
            "reachable_candidate_count": {"type": "integer", "minimum": 0, "maximum": POOL_LIMIT},
            "claim_boundary": {"type": "string", "minLength": 1},
        },
    }
    for node, value in ((candidate, per_candidate), (grasp, summary)):
        if "ik_reachability" in node["properties"]:
            raise ToolSurfaceError(f"{CAPABILITY} would overwrite an existing ik_reachability")
        node["properties"]["ik_reachability"] = copy.deepcopy(value)
        node["required"] = [*node.get("required", []), "ik_reachability"]
    path = Path(__file__)
    return {
        "slot": f"derived:{CAPABILITY}",
        "ref": "ora/agent/composition/grasp_ik_reachability_schema.py",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
