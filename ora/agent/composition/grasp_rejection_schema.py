"""Optional compiler-opening diagnostics shared by maintained grasp surfaces."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Mapping


CAPABILITY = "grasp_opening_rejection_feedback_v1"
LIMIT_SEMANTICS = "compiler_admission_not_measured_physical_aperture"
OPENING_PROPERTIES = {
    "predicted_opening_m": {
        "type": ["number", "null"],
        "description": "Raw predicted opening in metres; a finite invalid prediction may be negative, and unavailable is null.",
    },
    "required_opening_m": {
        "type": ["number", "null"], "minimum": 0,
        "description": "PCA-derived required opening in metres, or null when unavailable; not measured jaw aperture.",
    },
    "maximum_opening_m": {
        "type": "number", "exclusiveMinimum": 0,
        "description": "Positive compiler admission ceiling in metres; not measured physical aperture.",
    },
    "limit_semantics": {"type": "string", "enum": [LIMIT_SEMANTICS]},
}


def bind_grasp_rejection_diagnostics(results: Mapping[str, Any]) -> dict[str, str]:
    """Extend only the shared rejection definition and identify this derivation."""
    from .tool_surface import ToolSurfaceError

    name = "grasp_plan" if "grasp_plan" in results else "grasp.plan"
    schema = results.get(name, {})
    rejection = schema.get("$defs", {}).get("rejected_backend_candidate")
    if (not isinstance(rejection, dict) or rejection.get("type") != "object"
            or rejection.get("additionalProperties") is not False
            or set(rejection.get("required", [])) != {"candidate_id", "reasons"}
            or not isinstance(rejection.get("properties"), dict)):
        raise ToolSurfaceError(f"{CAPABILITY} requires the closed backend-rejection contract")
    for key, value in OPENING_PROPERTIES.items():
        existing = rejection["properties"].get(key)
        if existing is not None and existing != value:
            raise ToolSurfaceError(f"conflicting grasp rejection schema for {key}")
        rejection["properties"][key] = copy.deepcopy(value)
    path = Path(__file__)
    return {
        "slot": f"derived:{CAPABILITY}",
        "ref": "ora/agent/composition/grasp_rejection_schema.py",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
