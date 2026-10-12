"""The pose-primitive result field the planning-time reach reference adds.

Under ``pose_primitive_reach_along_request_v1`` each target of ``move_to``,
``rotate_wrist`` and ``rotate_pitch`` -- the semantic primitives that compile
through canonical ``move_pose`` -- carries ``reach_along_request``, the same
annotation ``move_pose_reach_along_request_v1`` gives ``move_pose`` (shape:
``move_pose_reach_schema.REACH_PROPERTY``). The primitives hold position or
orientation and never change the jaw's turn, so no jaw-symmetry field is
added. Runtime half: ``agent/tools/move_pose_reach.primitive_reference``.

EXP11A g009 stack_bowls issued 37 ``move_to`` against 6 ``move_pose``; two
refused lifts in its last round reported only ``planner_failed``.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Sequence

from .move_pose_reach_schema import REACH_PROPERTY

CAPABILITY = "pose_primitive_reach_along_request_v1"
PRIMITIVES = ("move_to", "rotate_wrist", "rotate_pitch")


def bind_pose_primitive_reach(results: dict[str, Any], capabilities: Sequence[str]) -> dict[str, str]:
    """Add reach_along_request to every declared primitive's compiled_pose targets."""
    from .tool_surface import ToolSurfaceError

    for required in ("semantic_pose_primitives_v1", "motion_intent_router_v1"):
        if required not in capabilities:
            raise ToolSurfaceError(f"{CAPABILITY} requires {required}")
    bound = 0
    for name in PRIMITIVES:
        result = results.get(name)
        if result is None:
            continue
        compiled = result.get("properties", {}).get("compiled_pose") if isinstance(result, dict) else None
        if not isinstance(compiled, dict) or compiled.get("type") != "object":
            raise ToolSurfaceError(f"{CAPABILITY} requires the {name} compiled_pose result")
        targets = compiled.setdefault("properties", {}).setdefault("targets", {"type": "object"})
        properties = targets.setdefault("additionalProperties", {"type": "object"}).setdefault("properties", {})
        if "reach_along_request" in properties:
            raise ToolSurfaceError(f"{CAPABILITY} would overwrite {name} reach_along_request")
        properties["reach_along_request"] = copy.deepcopy(REACH_PROPERTY)
        bound += 1
    if not bound:
        raise ToolSurfaceError(f"{CAPABILITY} requires at least one of {', '.join(PRIMITIVES)}")
    path = Path(__file__)
    return {
        "slot": f"derived:{CAPABILITY}",
        "ref": "ora/agent/composition/pose_primitive_reach_schema.py",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


__all__ = ["CAPABILITY", "PRIMITIVES", "bind_pose_primitive_reach"]
