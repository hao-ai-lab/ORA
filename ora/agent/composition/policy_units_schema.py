"""Restate help text in degrees on a ``policy_angles_degrees_v1`` surface.

At runtime the surface delivers every angle in degrees
(:mod:`ora.agent.context.policy_units`). Its help text must name the
same fields and units: ``orientation_error_rad in radians`` becomes
``orientation_error_deg in degrees``, ``orientation error <= 0.1 rad`` becomes
``<= 5.73 deg``, and a generated controller contract restates its JSON pair.
The binder runs after every binder that writes prose, so generated rules
(controller feedback, reach, jaw symmetry) are restated too.

Result schemas stay as they are: they validate the canonical radian result
before delivery converts it. Native projects only the top-level properties a
result schema declares, so a radian top-level property would vanish once
renamed; a declaring surface with one is refused here rather than at runtime.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..context import policy_units as units
from ..context.policy_units import RADIAN_TO_DEGREE_KEYS, degrees_text

CAPABILITY = units.CAPABILITY


def _descriptions(node: Any) -> Any:
    """A copy of one schema with every ``description`` string restated."""
    if isinstance(node, Mapping):
        return {key: (degrees_text(value) if key == "description" and isinstance(value, str)
                      else _descriptions(value)) for key, value in node.items()}
    if isinstance(node, list):
        return [_descriptions(item) for item in node]
    return node


def _top_level_properties(schema: Mapping[str, Any]) -> set[str]:
    names = set(schema.get("properties") or ())
    for keyword in ("oneOf", "anyOf"):
        for branch in schema.get(keyword) or ():
            if isinstance(branch, Mapping):
                names.update(branch.get("properties") or ())
    return names


def bind_policy_angles_degrees(
    definitions: list[dict[str, Any]], results: Mapping[str, Mapping[str, Any]],
    capabilities: Sequence[str],
) -> list[dict[str, str]]:
    """Restate every tool's help text in degrees, in place."""
    from .tool_surface import ToolSurfaceError

    if CAPABILITY not in capabilities:
        raise ToolSurfaceError(f"{CAPABILITY} binder called for a surface that does not declare it")
    for name, schema in sorted(results.items()):
        radian = sorted(_top_level_properties(schema) & set(RADIAN_TO_DEGREE_KEYS))
        if radian:
            raise ToolSurfaceError(
                f"{CAPABILITY}: result {name} declares top-level radian properties {radian}; "
                "Native delivers only declared top-level keys, so the degree name would be dropped")
    for definition in definitions:
        definition["description"] = degrees_text(definition["description"])
        definition["inputSchema"] = _descriptions(definition["inputSchema"])
    return [
        {"slot": f"derived:{CAPABILITY}:{path.stem}",
         "ref": "ora/" + str(path).split("/ora/", 1)[1],
         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in (Path(__file__).resolve(), Path(units.__file__).resolve())
    ]


__all__ = ["CAPABILITY", "bind_policy_angles_degrees"]
