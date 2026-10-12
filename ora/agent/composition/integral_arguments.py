"""An integral float at an integer position is that integer.

JSON has one number type, and JSON Schema (draft 6 onward) counts ``316.0`` as
an integer. The Harness validator and the handlers did not: an integral float
at an ``integer`` position was refused, and ``compute`` returns its scalars as
floats, so a pixel copied from a ``compute`` result into ``mark_point`` or
``vision_segment`` was refused (EXP11B census A2).

``coerce_integral_arguments`` rewrites, before dispatch, every finite float
with no fractional part whose position the delivered schema types as
``integer`` and never as ``number``. It reads only the schema: no tool, field or
value is named here, and a non-integral or non-finite float is left for the
validator to refuse. Both gateways (Native and managed MCP) call it after
null-as-omitted, on the definitions they delivered.

Kept out of ``null_optional_arguments.py`` on purpose: that module is hashed
into the composed surfaces of the profiles that declare it, and this rule is a
runtime normalisation that changes no composed byte.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence


def _types(node: Mapping[str, Any]) -> set[str]:
    declared = node.get("type")
    if isinstance(declared, str):
        return {declared}
    return {item for item in declared if isinstance(item, str)} if isinstance(declared, list) else set()


def _expand(nodes: Iterable[Any], root: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The schema nodes that apply at one position: ``$ref`` resolved, unions flattened."""

    expanded: list[Mapping[str, Any]] = []
    pending = [node for node in nodes if isinstance(node, Mapping)]
    seen: set[int] = set()
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        expanded.append(node)
        reference = node.get("$ref")
        if isinstance(reference, str) and reference.startswith("#/$defs/"):
            target = root.get("$defs", {}).get(reference[len("#/$defs/"):])
            if isinstance(target, Mapping):
                pending.append(target)
        for keyword in ("oneOf", "anyOf", "allOf"):
            branches = node.get(keyword)
            if isinstance(branches, list):
                pending.extend(branch for branch in branches if isinstance(branch, Mapping))
    return expanded


def _coerce(value: Any, nodes: Sequence[Any], root: Mapping[str, Any]) -> Any:
    applying = _expand(nodes, root)
    if isinstance(value, float):
        types = set().union(*(_types(node) for node in applying)) if applying else set()
        if ("integer" in types and "number" not in types
                and math.isfinite(value) and value.is_integer()):
            return int(value)
        return value
    if isinstance(value, Mapping):
        changed = {}
        for key, item in value.items():
            children = [node["properties"][key] for node in applying
                        if isinstance(node.get("properties"), Mapping) and key in node["properties"]]
            children += [node["additionalProperties"] for node in applying
                         if isinstance(node.get("additionalProperties"), Mapping)]
            coerced = _coerce(item, children, root) if children else item
            if coerced is not item:
                changed[key] = coerced
        return {**value, **changed} if changed else value
    if isinstance(value, list):
        children = [node["items"] for node in applying if isinstance(node.get("items"), Mapping)]
        if not children:
            return value
        coerced = [_coerce(item, children, root) for item in value]
        return coerced if any(new is not old for new, old in zip(coerced, value)) else value
    return value


def coerce_integral_arguments(
    definitions: Iterable[Mapping[str, Any]], name: str, arguments: Mapping[str, Any],
) -> Mapping[str, Any]:
    """``arguments`` with integral floats at integer positions as ints; unchanged otherwise."""

    if not isinstance(arguments, Mapping):
        return arguments
    from ora.agent.tools import tool_names  # the dotted and canonical spellings

    wanted = tool_names.canonical(str(name))
    for definition in definitions:
        if (tool_names.canonical(str(definition.get("name"))) == wanted
                and isinstance(definition.get("inputSchema"), Mapping)):
            schema = definition["inputSchema"]
            return _coerce(arguments, [schema], schema)
    return arguments


__all__ = ["coerce_integral_arguments"]
