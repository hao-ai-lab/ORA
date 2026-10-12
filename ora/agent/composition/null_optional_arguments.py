"""Optional tool arguments accept an explicit null, and null means omitted.

EXP11A g016 (EXP11B g003), GPT-6 Astra: all 160 ``geometry_query`` calls
filled every optional property -- ``points``, ``region``, ``step``,
``window_px`` and ``world_points_m`` -- inventing placeholders for the unused
ones (``region`` [0,0,1,1], ``world_points_m`` [[0,0,0]]); all 153 that
reached the handler were refused by its exactly-one-of rule. DeepSeek and Kimi
sent one operation and succeeded. The keys were forced: the Codex gateway's
Responses backend decodes a function tool without ``strict`` as strict when its
schema allows it (probe 2026-10-03), so every key must be present and a
non-nullable optional key has no legal "unused" value -- the decoder emits the
smallest schema-valid placeholder. (dde39f75 read Astra's sparse arguments on
other tools as proof the gateway forced nothing; those tools carry keywords
strict mode rejects.) The one optional property Astra left unused was ``step``,
the only one typed ``["integer", "null"]``. ``tool_strictness: explicit_false``
(agent/providers/openai_wire_options.py) now states ``"strict": false`` on that
profile; this rule keeps a strict decoder's calls legal as well.

Under ``optional_argument_null_as_omitted_v1`` every delivered tool takes one
flat argument object in which every optional member, at every depth, accepts
``null`` meaning omitted:

* composition adds ``"null"`` to the type (and ``null`` to the ``enum``) of
  every optional property of every object schema in every tool's input schema
  -- top level, nested objects, array items, ``$defs`` -- and appends
  ``Null means omitted.`` to its description. A required property never accepts
  null. A property that already accepts null keeps its bytes; on a declaring
  surface its null must already mean omitted. In a union, a property is
  optional only when neither the union's owner nor any branch requires it, so
  every branch gains the same nullable property. An optional property with no
  plain ``type`` or typeless ``enum`` is refused at composition.
* ``deliver_tool_definitions`` applies the same rule to the FINAL tool list a
  gateway delivers, so tools appended after composition (``artifact_read``,
  ``observe``'s ``ref`` mode) are covered, and flattens every root
  union into one object: the root's own ``properties`` when it declares them,
  otherwise the merged branch properties; ``required`` is what the root and
  every branch require. A root union that cannot be flattened fails delivery.
  The composed surface keeps its branches, so the Runtime still validates the
  exact mode rules.
* both gateways -- Native ``NativeToolGateway`` and the managed MCP façade --
  drop every null-valued optional member, recursively and schema-directed,
  before classification, validation and dispatch, so a handler sees exactly
  the call without those keys. A per-arm map (``targets.left``) whose arm is
  null therefore commands that arm exactly as omission does: not at all; a map
  whose arms are all null is the empty map. Nothing else is normalised: a
  placeholder value is still a value.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

CAPABILITY = "optional_argument_null_as_omitted_v1"
NULL_SENTENCE = "Null means omitted."
_UNIONS = ("oneOf", "anyOf")


def _resolve(schema: Any, defs: Mapping[str, Any]) -> Any:
    reference = schema.get("$ref") if isinstance(schema, Mapping) else None
    if isinstance(reference, str) and reference.startswith("#/$defs/"):
        target = defs.get(reference.split("/", 2)[2])
        return _resolve(target, defs) if isinstance(target, Mapping) else schema
    return schema


def _object_branches(schema: Mapping[str, Any], defs: Mapping[str, Any] | None = None) -> list[Mapping[str, Any]]:
    defs = defs or {}
    branches = [schema] if isinstance(schema.get("properties"), Mapping) else []
    for keyword in _UNIONS:
        for raw in schema.get(keyword) or ():
            branch = _resolve(raw, defs)
            if isinstance(branch, Mapping) and isinstance(branch.get("properties"), Mapping):
                branches.append(branch)
    return branches


def optional_argument_names(
    schema: Mapping[str, Any], defs: Mapping[str, Any] | None = None,
) -> frozenset[str]:
    """Member names of one object schema that neither it nor any branch requires."""

    branches = _object_branches(schema, defs)
    required = set(schema.get("required") or ())
    names: set[str] = set()
    for branch in branches:
        required.update(branch.get("required") or ())
        names.update(branch["properties"])
    return frozenset(names - required)


def accepts_null(schema: Mapping[str, Any]) -> bool:
    """Whether a JSON null validates against this property schema."""

    kind = schema.get("type")
    enum = schema.get("enum")
    if "const" in schema:
        return schema["const"] is None
    if kind is None and isinstance(enum, list):
        return None in enum
    typed = kind == "null" or (isinstance(kind, list) and "null" in kind)
    if typed and (not isinstance(enum, list) or None in enum):
        return True
    return any(
        isinstance(branch, Mapping) and accepts_null(branch)
        for keyword in _UNIONS for branch in schema.get(keyword) or ()
    )


def _nullable(path: str, schema: Mapping[str, Any]) -> dict[str, Any]:
    from .tool_surface import ToolSurfaceError

    value = copy.deepcopy(dict(schema))
    kind = schema.get("type")
    enum = schema.get("enum")
    if "const" in schema or not (isinstance(kind, (str, list)) or (kind is None and isinstance(enum, list))):
        raise ToolSurfaceError(f"{CAPABILITY} cannot make {path} nullable: it declares no plain type")
    if kind is not None:
        kinds = [kind] if isinstance(kind, str) else list(kind)
        value["type"] = kinds if "null" in kinds else [*kinds, "null"]
    if isinstance(enum, list) and None not in enum:
        value["enum"] = [*enum, None]
    description = str(value.get("description", "")).rstrip()
    if not description.endswith(NULL_SENTENCE):
        value["description"] = f"{description} {NULL_SENTENCE}".strip()
    return value


def _bind(node: Any, *, path: str, defs: Mapping[str, Any], own: bool = True) -> Any:
    """A copy of ``node`` with every optional member at every depth nullable."""

    if not isinstance(node, Mapping) or "$ref" in node:
        return copy.deepcopy(node)
    value = copy.deepcopy(dict(node))
    if isinstance(value.get("properties"), Mapping):
        value["properties"] = {
            name: _bind(child, path=f"{path}.{name}", defs=defs)
            for name, child in value["properties"].items()
        }
    if isinstance(value.get("items"), Mapping):
        value["items"] = _bind(value["items"], path=f"{path}[]", defs=defs)
    for keyword in (*_UNIONS, "allOf"):
        if isinstance(value.get(keyword), list):
            value[keyword] = [
                _bind(branch, path=f"{path}.{keyword}[{index}]", defs=defs, own=False)
                for index, branch in enumerate(value[keyword])
            ]
    if isinstance(value.get("$defs"), Mapping):
        value["$defs"] = {
            name: _bind(child, path=f"$defs.{name}", defs=defs)
            for name, child in value["$defs"].items()
        }
    if own:
        optional = optional_argument_names(value, defs)
        for branch in _object_branches(value):
            for name in sorted(optional & set(branch["properties"])):
                if not accepts_null(_resolve(branch["properties"][name], defs)):
                    branch["properties"][name] = _nullable(f"{path}.{name}", branch["properties"][name])
    return value


def bind_null_optional_arguments(definitions: list[dict[str, Any]]) -> dict[str, str]:
    """Make every optional member of every tool's input schema nullable, in place."""

    for definition in definitions:
        schema = definition.get("inputSchema")
        if isinstance(schema, Mapping):
            defs = schema.get("$defs") if isinstance(schema.get("$defs"), Mapping) else {}
            definition["inputSchema"] = _bind(schema, path=str(definition.get("name")), defs=defs)
    return {
        "slot": f"derived:{CAPABILITY}",
        "ref": "ora/agent/composition/null_optional_arguments.py",
        "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def flatten_root_union(tool: str, schema: Mapping[str, Any]) -> dict[str, Any]:
    """One flat argument object for a root ``oneOf``/``anyOf``, or ToolSurfaceError."""

    from .tool_surface import ToolSurfaceError

    keyword = next((key for key in _UNIONS if isinstance(schema.get(key), list)), None)
    if keyword is None:
        return copy.deepcopy(dict(schema))
    defs = schema.get("$defs") if isinstance(schema.get("$defs"), Mapping) else {}
    branches = [_resolve(branch, defs) for branch in schema[keyword]]
    if not branches or not all(
        isinstance(branch, Mapping) and isinstance(branch.get("properties"), Mapping) for branch in branches
    ):
        raise ToolSurfaceError(f"{tool}: a root union whose branches are not objects cannot be delivered flat")
    if isinstance(schema.get("properties"), Mapping):
        properties = copy.deepcopy(dict(schema["properties"]))
        missing = {name for branch in branches for name in branch["properties"]} - set(properties)
        if missing:
            raise ToolSurfaceError(f"{tool}: the root property table omits branch members {sorted(missing)}")
    else:
        properties = {}
        for branch in branches:
            for name, child in branch["properties"].items():
                if name in properties and properties[name] != child:
                    raise ToolSurfaceError(
                        f"{tool}: branch definitions of {name} differ; declare a root property table"
                    )
                properties[name] = copy.deepcopy(child)
    required = set(schema.get("required") or ())
    required |= set.intersection(*(set(branch.get("required") or ()) for branch in branches))
    flat: dict[str, Any] = {
        key: copy.deepcopy(value) for key, value in schema.items()
        if key not in {*_UNIONS, "properties", "required", "type", "additionalProperties"}
    }
    flat.update({"type": "object", "properties": properties, "required": sorted(required)})
    if schema.get("additionalProperties") is False or all(
        branch.get("additionalProperties") is False for branch in branches
    ):
        flat["additionalProperties"] = False
    return flat


def deliver_tool_definitions(
    definitions: Sequence[Mapping[str, Any]], capabilities: Sequence[str],
) -> list[dict[str, Any]]:
    """The final tool list as a declaring surface delivers it; unchanged otherwise."""

    from .schema_validation import lint_input_schema

    values = [copy.deepcopy(dict(item)) for item in definitions]
    if CAPABILITY not in tuple(capabilities or ()):
        return values
    for item in values:
        lint_input_schema(item.get("inputSchema"), source=f"delivered.{item.get('name')}")
    for item in values:
        if isinstance(item.get("inputSchema"), Mapping):
            item["inputSchema"] = flatten_root_union(str(item.get("name")), item["inputSchema"])
    bind_null_optional_arguments(values)
    return values


def strip_null_optional(value: Any, schema: Any, defs: Mapping[str, Any]) -> Any:
    """``value`` without null optional members, following ``schema``; same object if unchanged."""

    schema = _resolve(schema, defs)
    if not isinstance(schema, Mapping):
        return value
    if isinstance(value, Mapping):
        branches = _object_branches(schema, defs)
        if not branches:
            return value
        optional = optional_argument_names(schema, defs)
        output: dict[str, Any] = {}
        changed = False
        for key, child in value.items():
            if child is None and key in optional:
                changed = True
                continue
            shapes = [branch["properties"][key] for branch in branches if key in branch["properties"]]
            if shapes and all(shape == shapes[0] for shape in shapes):
                stripped = strip_null_optional(child, shapes[0], defs)
                changed |= stripped is not child
                child = stripped
            output[key] = child
        return output if changed else value
    if isinstance(value, list) and isinstance(schema.get("items"), Mapping):
        items = [strip_null_optional(item, schema["items"], defs) for item in value]
        return items if any(new is not old for new, old in zip(items, value)) else value
    return value


def omit_null_optional_arguments(
    surface: Any, name: str, arguments: Mapping[str, Any],
    definitions: Sequence[Mapping[str, Any]] | None = None,
) -> Mapping[str, Any]:
    """Drop null optional members at every depth on a declaring surface.

    ``definitions`` is the delivered list when the caller has it (it covers
    tools appended after composition); otherwise the surface's own. Returns
    ``arguments`` itself when nothing is dropped.
    """

    if CAPABILITY not in tuple(getattr(surface, "capabilities", ()) or ()):
        return arguments
    if not isinstance(arguments, Mapping):
        return arguments
    from ora.agent.tools import tool_names

    listed: Sequence[Mapping[str, Any]] = (
        definitions if definitions is not None else getattr(surface, "definitions", ())
    )
    definition = next((
        item for item in listed
        if tool_names.canonical(str(item.get("name"))) == tool_names.canonical(str(name))
    ), None)
    schema = definition.get("inputSchema") if isinstance(definition, Mapping) else None
    if not isinstance(schema, Mapping):
        return arguments
    defs = schema.get("$defs") if isinstance(schema.get("$defs"), Mapping) else {}
    return strip_null_optional(arguments, schema, defs)


__all__ = [
    "CAPABILITY", "NULL_SENTENCE", "accepts_null", "bind_null_optional_arguments",
    "deliver_tool_definitions", "flatten_root_union", "omit_null_optional_arguments",
    "optional_argument_names", "strip_null_optional",
]
