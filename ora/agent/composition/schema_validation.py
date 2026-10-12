"""The JSON-Schema subset Harness tool inputs and results are validated against.

A mismatch raises ``SchemaViolation``: the instance path, the violated keyword,
the constraint with its bound and the received shape (type, length, member
names -- never values). ``agent.tools.tool_errors.schema_refusal`` turns it into
the policy-visible ``invalid_arguments`` envelope.

Inputs are checked against every keyword in ``VALIDATED_KEYWORDS``, and
``lint_input_schema`` refuses at composition any other keyword in an input
schema, so a constraint the model is shown is a constraint the Runtime checks.
Results keep the keyword set they were checked against before EXP11B.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Mapping


class ToolSurfaceError(RuntimeError):
    """A tool surface is unknown, malformed, or changed after qualification."""


#: Keywords validate_schema_instance enforces, and annotations it ignores. An
#: input schema may use nothing else (``lint_input_schema``); a keyword the
#: model is shown must be a keyword the Runtime checks.
VALIDATED_KEYWORDS = frozenset({
    "$ref", "$defs", "type", "const", "enum", "minimum", "maximum",
    "exclusiveMinimum", "exclusiveMaximum", "oneOf", "anyOf", "allOf",
    "properties", "required", "additionalProperties", "minProperties",
    "maxProperties", "items", "minItems", "maxItems", "uniqueItems",
    "minLength", "maxLength", "pattern",
})
ANNOTATION_KEYWORDS = frozenset({
    "description", "default", "title", "examples", "$comment", "format", "$schema",
})
#: Keywords enforced on Runtime-produced results since before EXP11B. Results
#: keep that set: enforcing a newly checked keyword on Harness output could turn
#: a correct live result into an internal fault.
_RESULT_KEYWORDS = frozenset(VALIDATED_KEYWORDS - {
    "anyOf", "allOf", "minProperties", "maxProperties", "uniqueItems", "maxLength", "pattern",
})


class SchemaViolation(ToolSurfaceError):
    """One instance/schema mismatch, with the path, the violated keyword and its bound."""

    def __init__(
        self, message: str, *, path: str, keyword: str,
        constraints: Mapping[str, Any], received: Mapping[str, Any],
    ) -> None:
        super().__init__(message)
        self.path = path
        self.keyword = keyword
        self.constraints = dict(constraints)
        self.received = dict(received)


def _json_kind(instance: Any) -> str:
    if instance is None:
        return "null"
    if isinstance(instance, bool):
        return "boolean"
    if isinstance(instance, int):
        return "integer"
    if isinstance(instance, float):
        return "number"
    if isinstance(instance, str):
        return "string"
    if isinstance(instance, list):
        return "array"
    return "object" if isinstance(instance, Mapping) else type(instance).__name__


def _shape(instance: Any) -> dict[str, Any]:
    """What was received, described without its values."""

    shape: dict[str, Any] = {"type": _json_kind(instance)}
    if isinstance(instance, (list, str)):
        shape["length"] = len(instance)
    if isinstance(instance, Mapping):
        shape["names"] = sorted(str(name) for name in instance)
    return shape


def validate_schema_instance(
    value: Any, schema: Mapping[str, Any], *, root_path: str = "$result",
) -> None:
    """Validate a public tool input or result; a mismatch raises SchemaViolation."""

    value_kind = "input" if root_path == "$input" else "result"
    enforced = VALIDATED_KEYWORDS if value_kind == "input" else _RESULT_KEYWORDS

    def check(instance: Any, node: Mapping[str, Any], root: Mapping[str, Any], path: str) -> None:
        def fail(keyword: str, message: str, constraints: Mapping[str, Any]) -> None:
            raise SchemaViolation(
                f"{path} {message}", path=path, keyword=keyword,
                constraints=constraints, received=_shape(instance),
            )

        reference = node.get("$ref")
        if isinstance(reference, str):
            prefix = "#/$defs/"
            if not reference.startswith(prefix):
                raise ToolSurfaceError(f"unsupported schema reference at {path}")
            target = root.get("$defs", {}).get(reference[len(prefix):])
            if not isinstance(target, Mapping):
                raise ToolSurfaceError(f"unresolved schema reference at {path}")
            check(instance, target, root, path)
            return
        declared = node.get("type")
        if declared is not None:
            types = declared if isinstance(declared, list) else [declared]
            matches_type = any({
                "null": instance is None,
                "object": isinstance(instance, Mapping),
                "array": isinstance(instance, list),
                "string": isinstance(instance, str),
                "boolean": isinstance(instance, bool),
                "integer": isinstance(instance, int) and not isinstance(instance, bool),
                "number": isinstance(instance, (int, float)) and not isinstance(instance, bool),
            }.get(kind, False) for kind in types)
            if not matches_type:
                fail("type", f"has the wrong {value_kind} type; expected {' or '.join(types)}",
                     {"type": list(types)})
        if "const" in node and instance != node["const"]:
            fail("const", f"differs from {value_kind} const", {"const": node["const"]})
        if "enum" in node and instance not in node["enum"]:
            fail("enum", f"is outside the {value_kind} enum {node['enum']}", {"enum": list(node["enum"])})
        if isinstance(instance, (int, float)) and not isinstance(instance, bool):
            if isinstance(instance, float) and not math.isfinite(instance):
                fail("type", f"must be a finite {value_kind} number", {"finite": True})
            for keyword, violates in (
                ("minimum", lambda actual, bound: actual < bound),
                ("maximum", lambda actual, bound: actual > bound),
                ("exclusiveMinimum", lambda actual, bound: actual <= bound),
                ("exclusiveMaximum", lambda actual, bound: actual >= bound),
            ):
                if keyword not in node:
                    continue
                bound = node[keyword]
                if (
                    not isinstance(bound, (int, float))
                    or isinstance(bound, bool)
                    or (isinstance(bound, float) and not math.isfinite(bound))
                ):
                    raise ToolSurfaceError(f"{path} has an invalid {keyword} bound")
                if violates(instance, bound):
                    fail(keyword, f"violates {value_kind} {keyword} {bound}", {keyword: bound})
        branches = node.get("oneOf")
        if isinstance(branches, list):
            matched_branches: list[int] = []
            failures: list[str] = []
            for index, branch in enumerate(branches):
                try:
                    check(instance, branch, root, path)
                except ToolSurfaceError as error:
                    failures.append(f"oneOf[{index}]: {error}")
                    continue
                matched_branches.append(index)
            if len(matched_branches) != 1:
                detail = (
                    "; ".join(failures) if not matched_branches
                    else f"matched oneOf branches {matched_branches}"
                )
                fail("oneOf", f"must match exactly one {value_kind} branch; {detail}",
                     {"oneOf": len(branches), "matched": matched_branches})
            if "properties" not in node and "items" not in node:
                return
        if "anyOf" in enforced and isinstance(node.get("anyOf"), list):
            failures = []
            for index, branch in enumerate(node["anyOf"]):
                try:
                    check(instance, branch, root, path)
                    break
                except ToolSurfaceError as error:
                    failures.append(f"anyOf[{index}]: {error}")
            else:
                fail("anyOf", f"must match at least one {value_kind} branch; {'; '.join(failures)}",
                     {"anyOf": len(node["anyOf"])})
        if "allOf" in enforced and isinstance(node.get("allOf"), list):
            for branch in node["allOf"]:
                check(instance, branch, root, path)
        if isinstance(instance, Mapping):
            properties = node.get("properties", {})
            required = node.get("required", [])
            missing = set(required) - set(instance)
            if missing:
                fail("required", f"lacks required {value_kind} fields {sorted(missing)}",
                     {"required": list(required), "missing": sorted(missing)})
            if node.get("additionalProperties") is False:
                unknown = set(instance) - set(properties)
                if unknown:
                    fail("additionalProperties", f"has unknown {value_kind} fields {sorted(unknown)}",
                         {"allowed": sorted(properties), "unknown": sorted(unknown)})
            for keyword, violates in (
                ("minProperties", lambda size, bound: size < bound),
                ("maxProperties", lambda size, bound: size > bound),
            ):
                if keyword in enforced and isinstance(node.get(keyword), int) and violates(len(instance), node[keyword]):
                    fail(keyword, f"violates {value_kind} {keyword} {node[keyword]}", {keyword: node[keyword]})
            extra = node.get("additionalProperties")
            for name, child in instance.items():
                child_schema = properties.get(name) if isinstance(properties, Mapping) else None
                if not isinstance(child_schema, Mapping) and isinstance(extra, Mapping):
                    child_schema = extra
                if isinstance(child_schema, Mapping):
                    check(child, child_schema, root, f"{path}.{name}")
        if isinstance(instance, list):
            if isinstance(node.get("minItems"), int) and len(instance) < node["minItems"]:
                fail("minItems", f"is shorter than minItems {node['minItems']}", {"minItems": node["minItems"]})
            if isinstance(node.get("maxItems"), int) and len(instance) > node["maxItems"]:
                fail("maxItems", f"is longer than maxItems {node['maxItems']}", {"maxItems": node["maxItems"]})
            if "uniqueItems" in enforced and node.get("uniqueItems") is True:
                encoded = [json.dumps(item, sort_keys=True) for item in instance]
                if len(set(encoded)) != len(encoded):
                    fail("uniqueItems", "repeats an item; items must be unique", {"uniqueItems": True})
            item_schema = node.get("items")
            if isinstance(item_schema, Mapping):
                for index, child in enumerate(instance):
                    check(child, item_schema, root, f"{path}[{index}]")
        if isinstance(instance, str):
            if isinstance(node.get("minLength"), int) and len(instance) < node["minLength"]:
                fail("minLength", f"is shorter than minLength {node['minLength']}", {"minLength": node["minLength"]})
            if "maxLength" in enforced and isinstance(node.get("maxLength"), int) and len(instance) > node["maxLength"]:
                fail("maxLength", f"is longer than maxLength {node['maxLength']}", {"maxLength": node["maxLength"]})
            if "pattern" in enforced and isinstance(node.get("pattern"), str) and re.search(node["pattern"], instance) is None:
                fail("pattern", f"does not match pattern {node['pattern']}", {"pattern": node["pattern"]})

    check(value, schema, schema, root_path)


def lint_input_schema(node: Any, *, source: str) -> None:
    """Refuse an input-schema keyword the validator neither enforces nor ignores."""

    if isinstance(node, Mapping):
        unknown = sorted(set(node) - VALIDATED_KEYWORDS - ANNOTATION_KEYWORDS)
        if unknown:
            raise ToolSurfaceError(f"{source} uses schema keywords the Runtime does not validate: {unknown}")
        for keyword in ("properties", "$defs"):
            if isinstance(node.get(keyword), Mapping):
                for name, child in node[keyword].items():
                    lint_input_schema(child, source=f"{source}.{keyword}.{name}")
        for keyword in ("items", "additionalProperties"):
            if isinstance(node.get(keyword), Mapping):
                lint_input_schema(node[keyword], source=f"{source}.{keyword}")
        for keyword in ("oneOf", "anyOf", "allOf"):
            for index, branch in enumerate(node.get(keyword) or ()):
                lint_input_schema(branch, source=f"{source}.{keyword}[{index}]")
