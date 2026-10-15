"""LIBERO's grasp_plan IK reachability fields.

``grasp_ik_reachability_schema.py`` derives the pooled-selection fields and
selects the closed stage reason set from the RoboTwin or RoboDojo backend
identity capability. Its file hash is a component of three released surfaces,
so LIBERO is not added there: a surface declaring
``grasp_plan_ik_reachability_annotation_v1`` with ``libero_pro_rgbd_v1`` (the
LIBERO backend identity; it gates nothing else) is bound here instead, with the
same field shapes -- built from the released module's own constants and stage
helper -- and LIBERO's closed reason set from ``contracts/reach_check.py``.
Both files are named in the composition, so a change to either changes it.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Sequence

from ora.contracts.reach_check import LIBERO_REASONS, SHARED_REASONS

from . import grasp_ik_reachability_schema as released

IDENTITY_CAPABILITY = "libero_pro_rgbd_v1"
_REF = "ora/agent/composition/grasp_ik_reachability_libero_schema.py"


def declared(capabilities: Sequence[str]) -> bool:
    """A LIBERO surface asking for the annotation."""
    return released.CAPABILITY in capabilities and IDENTITY_CAPABILITY in capabilities


def bind_libero_grasp_ik_reachability(
    results: dict[str, Any], capabilities: Sequence[str],
) -> list[dict[str, str]]:
    """Add the annotation to LIBERO's grasp_plan result; identify both derivations."""
    from .tool_surface import ToolSurfaceError

    others = [name for name in released.BACKEND_REASONS if name in capabilities]
    if others:
        raise ToolSurfaceError(f"{released.CAPABILITY} requires exactly one backend with a read-only IK check")
    grasp = results.get("grasp_plan")
    if not isinstance(grasp, dict):
        raise ToolSurfaceError(f"{released.CAPABILITY} requires a grasp_plan result schema")
    candidate = grasp.get("$defs", {}).get("candidate")
    if (not isinstance(candidate, dict) or candidate.get("additionalProperties") is not False
            or not isinstance(candidate.get("properties"), dict)
            or grasp.get("additionalProperties") is not False):
        raise ToolSurfaceError(f"{released.CAPABILITY} requires the closed grasp_plan candidate contract")
    reasons = [*SHARED_REASONS, *LIBERO_REASONS, None]
    per_candidate = {
        "type": "object", "additionalProperties": False,
        "required": ["reachable", "member", "pregrasp", "approach", "approach_tilt_from_vertical_deg"],
        "description": "IK reference at planning time; null reachable means the annotation was unavailable.",
        "properties": {
            "reachable": {"type": ["boolean", "null"]},
            "member": {"type": ["string", "null"], "enum": list(released.MEMBERS)},
            "pregrasp": released._stage(reasons),
            "approach": released._stage(reasons),
            "approach_tilt_from_vertical_deg": {"type": "number", "minimum": 0, "maximum": 180},
        },
    }
    summary = {
        "type": "object", "additionalProperties": False,
        "required": ["status", "ik_model", "matches_move_pose_admission", "order",
                     "checked_candidate_count", "reachable_candidate_count", "claim_boundary"],
        "properties": {
            "status": {"type": "string", "enum": list(released.STATUSES)},
            "ik_model": {"type": ["string", "null"]},
            "matches_move_pose_admission": {"type": ["boolean", "null"]},
            "order": {"type": "string", "const": released.ORDER},
            "checked_candidate_count": {"type": "integer", "minimum": 0, "maximum": released.POOL_LIMIT},
            "reachable_candidate_count": {"type": "integer", "minimum": 0, "maximum": released.POOL_LIMIT},
            "claim_boundary": {"type": "string", "minLength": 1},
        },
    }
    for node, value in ((candidate, per_candidate), (grasp, summary)):
        if "ik_reachability" in node["properties"]:
            raise ToolSurfaceError(f"{released.CAPABILITY} would overwrite an existing ik_reachability")
        node["properties"]["ik_reachability"] = copy.deepcopy(value)
        node["required"] = [*node.get("required", []), "ik_reachability"]

    def descriptor(slot: str, ref: str) -> dict[str, str]:
        path = Path(__file__).resolve().parents[3] / ref
        return {"slot": slot, "ref": ref, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    return [descriptor(f"derived:{released.CAPABILITY}",
                       "ora/agent/composition/grasp_ik_reachability_schema.py"),
            descriptor(f"derived:{released.CAPABILITY}:{IDENTITY_CAPABILITY}", _REF)]


__all__ = ["IDENTITY_CAPABILITY", "bind_libero_grasp_ik_reachability", "declared"]
