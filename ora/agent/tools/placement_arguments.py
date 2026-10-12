"""One owner for the `place.plan` / `place.verify` argument shapes.

Both tools were authored with a nested `oneOf` destination and an array of
`{before_target_ref, current_target_ref}` objects.  Neither shape survives
delivery to a weak model.  `agent/native/gateway.py` flattens a union only at
the schema root, so a union nested one level down reaches the provider with all
three branches and their `const` discriminators intact -- the run H6-013 shape
(`$result must match exactly one result branch`, `constraints={}`) one level
down.  An array of objects is the run H6-012 shape: the model could not see
what to put inside the item and said so in its own reasoning.

The flat form replaces the union with one enum plus two named scalars, and the
array of objects with two index-aligned string arrays.  What the schema can no
longer state -- "open_container also needs a boundary reference", "the two
arrays are the same length" -- becomes a structured error that names the field.
It is deliberately not an `anyOf`: a union anywhere in this schema reproduces
H6-013, which is the defect being repaired.

Both forms are read here.  The frozen surfaces (`placement_support_v1`/`v2`,
`robotwin_placement_support_v1`/`v2`, `placement_unified_v3`..`v7`) keep the
nested shape and their sealed evidence, and sealed records keep it forever, so
only the bound `inputSchema` decides which form a policy may actually send.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .robot import ToolError


BOUNDARY_KINDS = frozenset({"open_container", "front_cavity"})
FLAT_DESTINATION_KEYS = (
    "destination_kind", "destination_support_ref", "destination_boundary_ref",
)
FLAT_PROTECTED_KEYS = ("protected_before_refs", "protected_current_refs")


def declares_flat_destination(schema: Mapping[str, Any]) -> bool:
    """Report which destination form one bound `inputSchema` advertises."""
    return "destination_kind" in schema.get("properties", {})


def declares_flat_protected(schema: Mapping[str, Any]) -> bool:
    """Report which protected-region form one bound `inputSchema` advertises."""
    return "protected_before_refs" in schema.get("properties", {})


def destination(arguments: Mapping[str, Any]) -> dict[str, str]:
    """Normalize either destination form into `{kind, support_ref, ...}`."""
    if "destination" in arguments:
        nested = arguments["destination"]
        value = {"kind": nested["kind"], "support_ref": nested["support_ref"]}
        if nested.get("boundary_ref") is not None:
            value["boundary_ref"] = nested["boundary_ref"]
        return value
    kind = arguments["destination_kind"]
    boundary = arguments.get("destination_boundary_ref")
    if kind in BOUNDARY_KINDS and boundary is None:
        raise ToolError(
            f"{kind} also needs its visible opening: pass destination_boundary_ref, a "
            "current target reference selecting the rim or front aperture region",
            code="invalid_arguments", field="destination_boundary_ref",
            received={"destination_kind": kind, "destination_boundary_ref": None})
    if kind not in BOUNDARY_KINDS and boundary is not None:
        raise ToolError(
            "support_surface has no opening to clear: omit destination_boundary_ref, or "
            "select open_container or front_cavity if the destination has one",
            code="invalid_arguments", field="destination_boundary_ref",
            received={"destination_kind": kind, "destination_boundary_ref": boundary})
    value = {"kind": kind, "support_ref": arguments["destination_support_ref"]}
    if boundary is not None:
        value["boundary_ref"] = boundary
    return value


def protected_pairs(arguments: Mapping[str, Any]) -> list[dict[str, str]]:
    """Normalize either protected-region form into ordered before/current pairs."""
    if "protected_objects" in arguments:
        return [{"before_target_ref": pair["before_target_ref"],
                 "current_target_ref": pair["current_target_ref"]}
                for pair in arguments["protected_objects"]]
    before = list(arguments.get("protected_before_refs") or ())
    current = list(arguments.get("protected_current_refs") or ())
    if len(before) != len(current):
        short = ("protected_current_refs" if len(current) < len(before)
                 else "protected_before_refs")
        raise ToolError(
            "protected_before_refs and protected_current_refs are compared entry by "
            f"entry and must be the same length; {short} is short",
            code="invalid_arguments", field=short,
            received={"protected_before_refs": len(before),
                      "protected_current_refs": len(current)})
    return [{"before_target_ref": first, "current_target_ref": second}
            for first, second in zip(before, current)]


def destination_fields(
    schema: Mapping[str, Any], *, kind: str, support_ref: str,
    boundary_ref: str | None = None,
) -> dict[str, Any]:
    """Emit destination arguments in the form one bound schema declares."""
    if not declares_flat_destination(schema):
        value = {"kind": kind, "support_ref": support_ref}
        if boundary_ref is not None:
            value["boundary_ref"] = boundary_ref
        return {"destination": value}
    value = {"destination_kind": kind, "destination_support_ref": support_ref}
    if boundary_ref is not None:
        value["destination_boundary_ref"] = boundary_ref
    return value


def protected_fields(
    schema: Mapping[str, Any], pairs: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    """Emit protected-region arguments in the form one bound schema declares."""
    if not declares_flat_protected(schema):
        return {"protected_objects": [
            {"before_target_ref": pair["before_target_ref"],
             "current_target_ref": pair["current_target_ref"]} for pair in pairs]}
    return {"protected_before_refs": [pair["before_target_ref"] for pair in pairs],
            "protected_current_refs": [pair["current_target_ref"] for pair in pairs]}
