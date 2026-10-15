"""``observed_clearance_reference_v2``: the observed clearance on every jaw model.

The result field is ``observed_clearance_schema.PROPERTY``:
``compiled_pose.targets.<arm>`` of ``move_pose``, ``move_to``, ``rotate_wrist``
and ``rotate_pitch`` gains ``observed_clearance``. The released v1 read the jaw
from fields only the ARX X5 descriptor publishes (``contact_center_distance_m``,
an empty-close encoder reference) and from two-finger encoder positions; the
ALOHA and Panda observations and descriptors carry other, equivalent public
facts. v2 binds only to a controlled frame whose jaw model the runtime knows
(``agent/tools/observed_clearance_geometry.JAW_MODELS``): the jaw centre, finger
opening and holding test per embodiment, each from a public source.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from .observed_clearance_schema import PROPERTY, TOOLS

CAPABILITY = "observed_clearance_reference_v2"


def _error(message: str) -> Exception:
    from .tool_surface import ToolSurfaceError

    return ToolSurfaceError(f"{CAPABILITY} {message}")


def bind_observed_clearance_v2(
    results: dict[str, Any], capabilities: Sequence[str], view_config: Mapping[str, Any] | None,
) -> dict[str, str]:
    """Add ``observed_clearance`` to every declared motion target; identify the derivation."""
    from ..tools.observed_clearance_geometry import JAW_MODELS

    if "enhanced_move_pose_v1" not in capabilities:
        raise _error("requires enhanced_move_pose_v1")
    view = view_config or {}
    if not isinstance(view.get("public_geometry_descriptor"), Mapping):
        raise _error("requires a public geometry descriptor")
    if view.get("controlled_frame_convention") not in JAW_MODELS:
        raise _error("requires a controlled frame with a known jaw model")
    if "move_pose" not in results:
        raise _error("requires the move_pose compiled_pose result")
    for name in TOOLS:
        result = results.get(name)
        if result is None:
            continue
        compiled = result.get("properties", {}).get("compiled_pose") if isinstance(result, dict) else None
        if not isinstance(compiled, dict) or compiled.get("type") != "object":
            raise _error(f"requires the {name} compiled_pose result")
        targets = compiled.setdefault("properties", {}).setdefault("targets", {"type": "object"})
        properties = targets.setdefault("additionalProperties", {"type": "object"}).setdefault("properties", {})
        if "observed_clearance" in properties:
            raise _error(f"would overwrite {name} observed_clearance")
        properties["observed_clearance"] = copy.deepcopy(PROPERTY)
    return {
        "slot": f"derived:{CAPABILITY}",
        "ref": "ora/agent/composition/observed_clearance_v2_schema.py",
        "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


__all__ = ["CAPABILITY", "bind_observed_clearance_v2"]
