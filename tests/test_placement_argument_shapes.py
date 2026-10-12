"""The placement argument shapes a weak model can actually fill in.

Two defects are under test, both with a run behind them.

Run H6-012: `move_pose` used `$defs` + `$ref`, the delivery layer dropped the
`$defs`, and the model said in its own reasoning that it could not see the
definition of the parameter it was being asked for. An array of objects fails
the same way -- the item shape is one indirection away from the field name --
and `place.plan.protected_objects` is one.

Run H6-013: a top-level `oneOf` turned a missing `orientation_xyzw` into
`$result must match exactly one result branch` with `constraints={}`. The union
caught every branch's specific failure and discarded it. `place.plan`'s
`destination` is that union one level down, where `agent/native/gateway.py`
does not reach it, so it is delivered to the provider intact.

The repair is structural, not documentary: two index-aligned string arrays in
place of the array of objects, and one enum plus two named scalars in place of
the union. What the schema then cannot state becomes a structured error that
names the field -- never an `anyOf`, which would be H6-013 again.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ora.agent.tools import tool_names

from ora.agent.composition.tool_surface import (
    ToolSurfaceError, validate_schema_instance,
)
from ora.agent.native.gateway import openai_tool_definitions
from ora.agent.tools import placement_arguments
from ora.agent.tools.placement import assess_geometry
from ora.agent.tools.robot import ToolError


ROOT = Path(__file__).resolve().parents[1]
SURFACES = ROOT / "ora/policy/tool_surfaces"
#: Named canonically; the live surfaces declare `canonical_tool_names_v1` and a
#: frozen one would advertise the dotted spelling, so every lookup below goes
#: through `tool_names` rather than comparing a literal.
PLACEMENT_TOOLS = ("place_plan", "place_verify")
SUPPORT = "target-" + "1" * 24
BOUNDARY = "target-" + "2" * 24
BEFORE = "target-" + "3" * 24
CURRENT = "target-" + "4" * 24

NESTED_DESTINATION = {"kind": "open_container", "support_ref": SUPPORT,
                      "boundary_ref": BOUNDARY}
FLAT_DESTINATION = {"destination_kind": "open_container",
                    "destination_support_ref": SUPPORT,
                    "destination_boundary_ref": BOUNDARY}
FLAT_SCHEMA = {"type": "object", "properties": dict.fromkeys(
    (*placement_arguments.FLAT_DESTINATION_KEYS,
     *placement_arguments.FLAT_PROTECTED_KEYS), {"type": "string"})}
NESTED_SCHEMA = {"type": "object", "properties": {
    "destination": {"type": "object"}, "protected_objects": {"type": "array"}}}


def _points(center: tuple[float, float, float]) -> dict[str, Any]:
    return {"voxel_points_world_m": [[center[0] + dx, center[1] + dy, center[2] + dz]
                                     for dx in (-.012, 0, .012) for dy in (-.012, 0, .012)
                                     for dz in (-.02, .02)],
            "source_camera": "agentview"}


def _support() -> dict[str, Any]:
    return {"voxel_points_world_m": [[x / 100, y / 100, 0.0]
                                     for x in range(-15, 16) for y in range(-15, 16)],
            "source_camera": "agentview"}


@pytest.mark.parametrize("kind", ["support_surface", "open_container", "front_cavity"])
def test_both_destination_forms_select_the_same_regions(kind: str) -> None:
    """The flat form is a spelling change, not a change of what is selected."""
    boundary = None if kind == "support_surface" else BOUNDARY
    nested: dict[str, Any] = {"kind": kind, "support_ref": SUPPORT}
    flat = {"destination_kind": kind, "destination_support_ref": SUPPORT}
    if boundary is not None:
        nested["boundary_ref"] = boundary
        flat["destination_boundary_ref"] = boundary
    expected = {"kind": kind, "support_ref": SUPPORT}
    if boundary is not None:
        expected["boundary_ref"] = boundary
    assert placement_arguments.destination({"destination": nested}) == expected
    assert placement_arguments.destination(flat) == expected


@pytest.mark.parametrize("kind", ["open_container", "front_cavity"])
def test_a_missing_opening_reference_is_an_error_that_names_the_field(kind: str) -> None:
    """H6-013's failure was a union that named nothing. This names the field."""
    with pytest.raises(ToolError) as error:
        placement_arguments.destination(
            {"destination_kind": kind, "destination_support_ref": SUPPORT})
    reported = error.value.as_dict()
    assert reported["code"] == "invalid_arguments"
    assert reported["field"] == "destination_boundary_ref"
    assert reported["received"] == {"destination_kind": kind,
                                    "destination_boundary_ref": None}
    assert "destination_boundary_ref" in reported["message"]


def test_an_opening_reference_on_a_bare_surface_names_the_same_field() -> None:
    """The union rejected this by branch closure; the error must say which field."""
    with pytest.raises(ToolError) as error:
        placement_arguments.destination({
            "destination_kind": "support_surface", "destination_support_ref": SUPPORT,
            "destination_boundary_ref": BOUNDARY})
    reported = error.value.as_dict()
    assert reported["field"] == "destination_boundary_ref"
    assert reported["received"]["destination_boundary_ref"] == BOUNDARY


def test_both_protected_forms_produce_the_same_ordered_pairs() -> None:
    pairs = [{"before_target_ref": BEFORE, "current_target_ref": CURRENT},
             {"before_target_ref": CURRENT, "current_target_ref": BEFORE}]
    assert placement_arguments.protected_pairs({"protected_objects": pairs}) == pairs
    assert placement_arguments.protected_pairs({
        "protected_before_refs": [BEFORE, CURRENT],
        "protected_current_refs": [CURRENT, BEFORE]}) == pairs


@pytest.mark.parametrize("before,current,short", [
    ([BEFORE, CURRENT], [CURRENT], "protected_current_refs"),
    ([BEFORE], [CURRENT, BEFORE], "protected_before_refs"),
    ([BEFORE], [], "protected_current_refs"),
])
def test_unequal_protected_arrays_name_the_array_that_is_short(
    before: list[str], current: list[str], short: str,
) -> None:
    """Parallel arrays only work if the length check says which one disagreed."""
    with pytest.raises(ToolError) as error:
        placement_arguments.protected_pairs({"protected_before_refs": before,
                                             "protected_current_refs": current})
    reported = error.value.as_dict()
    assert reported["code"] == "invalid_arguments"
    assert reported["field"] == short
    assert reported["received"] == {"protected_before_refs": len(before),
                                    "protected_current_refs": len(current)}


def test_the_named_field_survives_the_placement_argument_error_handler() -> None:
    """`invoke_placement` rewraps its parse failures into a bare message.

    It catches `(ToolSurfaceError, ValueError, TypeError, KeyError)` and
    re-raises `ToolError(str(error))` with no field and no `received`. A
    conditional-requirement error raised as one of those would arrive at the
    model exactly as H6-013 did: a rejection that does not say what to fix.
    """
    assert not issubclass(ToolError, (ToolSurfaceError, ValueError, TypeError, KeyError))


def test_omitting_both_protected_arrays_is_no_comparison_and_not_an_error() -> None:
    assert placement_arguments.protected_pairs({}) == []
    assert placement_arguments.protected_pairs(
        {"protected_before_refs": None, "protected_current_refs": None}) == []


@pytest.mark.parametrize("schema,expected", [
    (FLAT_SCHEMA, FLAT_DESTINATION), (NESTED_SCHEMA, {"destination": NESTED_DESTINATION}),
])
def test_the_emitter_follows_the_schema_and_the_normalizer_reads_it_back(
    schema: dict[str, Any], expected: dict[str, Any],
) -> None:
    """One owner for the shape, so a surface flip cannot silently break a caller."""
    emitted = placement_arguments.destination_fields(
        schema, kind="open_container", support_ref=SUPPORT, boundary_ref=BOUNDARY)
    assert emitted == expected
    assert placement_arguments.destination(emitted) == NESTED_DESTINATION
    pairs = [{"before_target_ref": BEFORE, "current_target_ref": CURRENT}]
    fields = placement_arguments.protected_fields(schema, pairs)
    assert placement_arguments.protected_pairs(fields) == pairs


def _placement_schemas() -> list[tuple[str, str, dict[str, Any]]]:
    found = []
    for path in sorted(SURFACES.rglob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        for item in document.get("tools", []) if isinstance(document, dict) else []:
            # These are the component files as written. They keep the dotted
            # spelling -- the rename happens during composition so the frozen
            # catalogs a live profile hash-references stay byte-identical -- so
            # the comparison canonicalises rather than matching a literal.
            if (tool_names.canonical(str(item.get("name"))) in PLACEMENT_TOOLS
                    and "inputSchema" in item):
                found.append((str(path.relative_to(SURFACES)), item["name"],
                              item["inputSchema"]))
    assert found, "no registered surface advertises a placement tool"
    return found


@pytest.mark.parametrize("surface,name,schema", _placement_schemas(),
                         ids=lambda value: value if isinstance(value, str) else "")
def test_every_registered_surface_is_read_in_the_form_it_declares(
    surface: str, name: str, schema: dict[str, Any],
) -> None:
    """The runtime accepts both forms; the bound schema decides which is legal.

    Frozen surfaces keep the nested shape with their sealed evidence, so this is
    the guard against the runtime and the schema drifting apart in either
    direction -- the defect `docs/guides/tool-surface-authoring.md` names.
    """
    arguments = {"object_ref": "target-" + "0" * 24, "orientation_requirement": "none",
                 ("phase" if tool_names.canonical(name) == "place_plan" else "stage"):
                     ("preplace" if tool_names.canonical(name) == "place_plan" else "pre_release"),
                 **placement_arguments.destination_fields(
                     schema, kind="open_container", support_ref=SUPPORT,
                     boundary_ref=BOUNDARY),
                 **placement_arguments.protected_fields(
                     schema, [{"before_target_ref": BEFORE, "current_target_ref": CURRENT}])}
    if "arm" in schema.get("properties", {}):
        arguments["arm"] = "right"
    validate_schema_instance(arguments, schema)
    assert placement_arguments.destination(arguments) == NESTED_DESTINATION
    assert placement_arguments.protected_pairs(arguments) == [
        {"before_target_ref": BEFORE, "current_target_ref": CURRENT}]
    other = (FLAT_DESTINATION if "destination" in arguments
             else {"destination": NESTED_DESTINATION})
    with pytest.raises(ToolSurfaceError):
        validate_schema_instance(
            {key: value for key, value in arguments.items()
             if key not in placement_arguments.FLAT_DESTINATION_KEYS
             and key != "destination"} | other, schema)


def _unions(node: Any) -> int:
    if isinstance(node, dict):
        return (sum(key in node for key in ("oneOf", "anyOf", "$ref", "$defs"))
                + sum(_unions(value) for value in node.values()))
    if isinstance(node, list):
        return sum(_unions(item) for item in node)
    return 0


def test_a_union_one_level_down_reaches_the_provider_but_the_flat_form_has_none() -> None:
    """Why the repair is structural: the projection only unwraps a root union.

    `_openai_parameters` flattens `oneOf` at the schema root. A union inside a
    property is handed over verbatim, discriminator branches and all, which is
    the H6-013 shape one level down.
    """
    nested = {"name": "place.plan", "description": "x", "inputSchema": {
        "type": "object", "additionalProperties": False,
        "required": ["destination"],
        "properties": {"destination": {"oneOf": [
            {"type": "object", "required": ["kind", "support_ref"], "properties": {
                "kind": {"const": "support_surface"}, "support_ref": {"type": "string"}}},
            {"type": "object", "required": ["kind", "support_ref", "boundary_ref"],
             "properties": {"kind": {"const": "open_container"},
                            "support_ref": {"type": "string"},
                            "boundary_ref": {"type": "string"}}}]}}}}
    flat = {"name": "place.plan", "description": "x", "inputSchema": {
        "type": "object", "additionalProperties": False,
        "required": ["destination_kind", "destination_support_ref"],
        "properties": {
            "destination_kind": {"type": "string", "enum": [
                "support_surface", "open_container", "front_cavity"]},
            "destination_support_ref": {"type": "string"},
            "destination_boundary_ref": {"type": ["string", "null"]}}}}
    delivered = {item["function"]["name"]: item["function"]["parameters"]
                 for item in openai_tool_definitions(
                     ["place.plan"], definitions=[nested])}
    assert _unions(delivered["place.plan"]) == 1
    delivered = {item["function"]["name"]: item["function"]["parameters"]
                 for item in openai_tool_definitions(["place.plan"], definitions=[flat])}
    assert _unions(delivered["place.plan"]) == 0


def test_each_protected_comparison_names_the_input_index_it_answers() -> None:
    """Parallel arrays are only legible if the result maps back to the index."""
    moved = _points((.06, 0, .04))
    still = _points((0, 0, .04))
    other_camera = dict(still, source_camera="wrist")
    pairs = [{"before_geometry": still, "current_geometry": still},
             {"before_geometry": still, "current_geometry": other_camera},
             {"before_geometry": still, "current_geometry": moved}]
    check = assess_geometry(_points((0, 0, .04)), _support(),
                            destination_kind="support_surface",
                            tcp_pose=[0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0],
                            protected_geometries=pairs)["checks"]["protected_objects"]
    items = check["metrics"]["items"]
    assert [item["source_index"] for item in items] == [0, 1, 2]
    assert [item["status"] for item in items] == ["consistent", "unknown", "contradicted"]


# --- The resolved surface, not the catalog file the resolved surface starts from.
def _flat_profiles() -> set[str]:
    """Every profile that declares the flat argument capability, after resolution.

    Derived rather than listed. A name list is a second copy of the manifests'
    own answer, and a sibling profile that hash-reuses these components inherits
    the flat form without appearing in it -- which is exactly how the closed
    manipulation arm arrived.
    """
    from ora.agent.composition.tool_surface import (
        PROFILE_CATALOG, resolve_tool_surface,
    )

    return {
        profile for profile in PROFILE_CATALOG
        if resolve_tool_surface(profile).supports("flat_placement_arguments_v1")
    }


FLAT_PROFILES = _flat_profiles()

def _flat_placement_pairs() -> list[tuple[str, str]]:
    """Every (profile, placement tool) pair that actually exists after resolution.

    Derived rather than a cross product. The closed manipulation arm declares
    `place_plan` and not `place_verify`, because the refactor plan lists the
    verifier as the verify stage of `place.execute`. Asserting the cross product
    would demand a tool from an arm that deliberately absorbed it.
    """
    from ora.agent.composition.tool_surface import resolve_tool_surface

    pairs = []
    for profile in sorted(FLAT_PROFILES):
        declared = {item["name"] for item in resolve_tool_surface(profile).definitions}
        pairs += [(profile, name) for name in PLACEMENT_TOOLS if name in declared]
    assert pairs
    return pairs


FLAT_PLACEMENT_PAIRS = _flat_placement_pairs()



def _resolved_placement_schemas() -> list[tuple[str, str, dict[str, Any]]]:
    """Every profile that advertises a placement tool, after composition.

    `_placement_schemas` reads the `*-definitions.json` catalogs directly, which
    is where the nested shape lives and where it must stay: those files are
    frozen. A live profile changes a tool by overriding it in a later
    `schema_order` component, and that override only exists after resolution --
    so a scan of the catalog files cannot see the flat form at all.
    """
    from ora.agent.composition.tool_surface import (
        PROFILE_CATALOG, resolve_tool_surface,
    )

    found = []
    for profile in sorted(PROFILE_CATALOG):
        surface = resolve_tool_surface(profile)
        for item in surface.definitions:
            if item["name"] in PLACEMENT_TOOLS:
                found.append((profile, item["name"], item["inputSchema"]))
    assert found
    return found


@pytest.mark.parametrize("profile,name,schema", _resolved_placement_schemas(),
                         ids=lambda value: value if isinstance(value, str) else "")
def test_every_resolved_surface_is_read_in_the_form_it_binds(
    profile: str, name: str, schema: dict[str, Any],
) -> None:
    """What a policy is handed is the composition, not the file it came from."""
    expected = profile in FLAT_PROFILES
    assert placement_arguments.declares_flat_destination(schema) is expected
    assert placement_arguments.declares_flat_protected(schema) is expected
    arguments = {"object_ref": "target-" + "0" * 24, "orientation_requirement": "none",
                 ("phase" if tool_names.canonical(name) == "place_plan" else "stage"):
                     ("preplace" if tool_names.canonical(name) == "place_plan" else "pre_release"),
                 **placement_arguments.destination_fields(
                     schema, kind="open_container", support_ref=SUPPORT,
                     boundary_ref=BOUNDARY),
                 **placement_arguments.protected_fields(
                     schema, [{"before_target_ref": BEFORE, "current_target_ref": CURRENT}])}
    if "arm" in schema.get("properties", {}):
        arguments["arm"] = "right"
    validate_schema_instance(arguments, schema)
    assert placement_arguments.destination(arguments) == NESTED_DESTINATION
    assert placement_arguments.protected_pairs(arguments) == [
        {"before_target_ref": BEFORE, "current_target_ref": CURRENT}]


@pytest.mark.parametrize("profile,name", FLAT_PLACEMENT_PAIRS,
                         ids=lambda v: v.rsplit("_", 1)[-1] if isinstance(v, str) else "")
def test_a_flat_surface_carries_no_union_and_no_array_of_objects(
    profile: str, name: str,
) -> None:
    """Both repaired defects, checked on the delivered shape rather than asserted.

    `anyOf` is listed alongside `oneOf` deliberately: expressing "open_container
    also needs a boundary reference" as a union would reproduce H6-013 while
    looking like a schema improvement.
    """
    from ora.agent.composition.tool_surface import resolve_tool_surface

    surface = resolve_tool_surface(profile)
    assert surface.supports("flat_placement_arguments_v1")
    definition = next(item for item in surface.definitions if item["name"] == name)
    schema = definition["inputSchema"]
    encoded = json.dumps(schema)
    for forbidden in ("oneOf", "anyOf", "allOf", "$ref", "$defs"):
        assert forbidden not in encoded, f"{name} still delivers {forbidden}"
    for field, node in schema["properties"].items():
        if node.get("type") == "array":
            assert "maxItems" in node, field
            assert node["items"].get("type") != "object", field
    # The help names the fields a policy now fills in, not the ones it replaced.
    assert "protected_objects" not in definition["description"]
    for field in placement_arguments.FLAT_DESTINATION_KEYS:
        assert field in definition["description"], field


@pytest.mark.parametrize("profile,name", FLAT_PLACEMENT_PAIRS,
                         ids=lambda v: v.rsplit("_", 1)[-1] if isinstance(v, str) else "")
def test_a_flat_surface_still_delivers_its_boundary_field_to_a_provider(
    profile: str, name: str,
) -> None:
    """H6-012 was a field the delivery layer dropped, so check after projection."""
    from ora.agent.composition.tool_surface import resolve_tool_surface

    surface = resolve_tool_surface(profile)
    projected = next(item for item in openai_tool_definitions(
        [name], definitions=list(surface.definitions))
                     if item["function"]["name"].replace("__", ".") == name)
    properties = projected["function"]["parameters"]["properties"]
    assert set(placement_arguments.FLAT_DESTINATION_KEYS) <= set(properties)
    assert set(placement_arguments.FLAT_PROTECTED_KEYS) <= set(properties)
    assert properties["destination_boundary_ref"]["type"] == ["string", "null"]
    assert all(properties[field]["description"].strip()
               for field in (*placement_arguments.FLAT_DESTINATION_KEYS,
                             *placement_arguments.FLAT_PROTECTED_KEYS))
