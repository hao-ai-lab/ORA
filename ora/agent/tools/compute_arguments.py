"""The tool-call edge of the restricted geometry evaluator.

:mod:`compute` is the kernel: the AST allowlist, the namespace, the forked
worker and the serialiser. This module is everything between a tool call and
that kernel -- reading either published spelling of ``bindings``, and joining
the result to the surface that declared the tool.

A surface publishes ``bindings`` as a list of ``{name, value, from?}`` entries,
because ``policy_tool_help_v1`` requires every object node to close
``additionalProperties`` and a mapping keyed by the policy's own identifiers
cannot. The entry form also states the optional origin as a field rather than
as a second accepted shape for the value, removing the one union this tool's
arguments carried. The mapping form stays readable: it is what the kernel was
written against and what an unpublished caller still passes.
"""

from __future__ import annotations

import ast
from typing import Any, Mapping

from . import compute as kernel
from . import tool_names
from .compute import MAX_BINDINGS, _reject, shared_runtime


#: Recorded in ``_runtime.provenance`` when a block was read with its escaped
#: line breaks as statement separators. Never policy-visible.
LITERAL_NEWLINE_NORMALIZATION = "literal_backslash_n_as_newline"


def normalize_block(source: Any) -> tuple[Any, str | None]:
    """Read an escaped line break as the statement separator it stands for.

    A policy that escapes its JSON string twice sends ``a = 1\\nb = 2``: a
    backslash followed by ``n`` where a line break was meant. Outside a string
    literal that pair is never valid Python, so the block was refused as
    "unexpected character after line continuation character" although every
    statement in it was admissible (four such refusals in the EXP11A g003
    batch). This reads the pair as a line break only when the block as sent does
    not parse and the rewritten block does. A block that already parses is never
    rewritten, so an escape inside a short string label keeps its meaning; a
    rewrite that would put a line break inside a quoted label does not parse and
    the original refusal stands. The rewritten text still passes the same AST
    allowlist, so the evaluator's admitted node set is unchanged.
    """

    if not isinstance(source, str) or "\\n" not in source:
        return source, None
    try:
        ast.parse(source, filename="<compute>", mode="exec")
        return source, None
    except SyntaxError:
        pass
    candidate = source.replace("\\r\\n", "\n").replace("\\n", "\n")
    try:
        ast.parse(candidate, filename="<compute>", mode="exec")
    except SyntaxError:
        return source, None
    return candidate, LITERAL_NEWLINE_NORMALIZATION


def read_bindings(value: Any) -> dict[str, Any]:
    """Read either published binding spelling into the evaluator's mapping.

    A surface publishes ``bindings`` as a list of ``{name, value, from?}``
    entries, because ``policy_tool_help_v1`` requires every object node to close
    ``additionalProperties`` and a mapping keyed by the policy's own identifiers
    cannot. The entry form also states the optional origin as a field rather
    than as a second accepted shape for the value, removing the one union this
    tool's arguments carried.

    The mapping form stays readable: it is what the kernel was written against
    and what an unpublished caller still passes. This function is the only place
    either spelling is read.
    """

    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, list):
        raise _reject(
            "bindings must be a list of {name, value} entries",
            received=type(value).__name__,
        )
    if len(value) > MAX_BINDINGS:
        raise _reject(
            "the call binds more names than one block may carry",
            received=len(value), maximum_bindings=MAX_BINDINGS,
        )
    bindings: dict[str, Any] = {}
    for index, entry in enumerate(value):
        if not isinstance(entry, Mapping):
            raise _reject(
                "each binding is an object with name and value",
                received=type(entry).__name__, field=f"bindings[{index}]",
            )
        unknown = sorted(set(entry) - {"name", "value", "from"})
        if unknown:
            raise _reject(
                "unknown binding field",
                received=unknown, allowed=["name", "value", "from"],
                field=f"bindings[{index}]",
            )
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise _reject(
                "a binding needs a non-empty name",
                received=name, field=f"bindings[{index}].name",
            )
        _check_binding_name(name, field=f"bindings[{index}].name")
        if name in bindings:
            raise _reject(
                "a name is bound twice", received=name,
                field=f"bindings[{index}].name",
            )
        if "value" not in entry:
            raise _reject(
                "a binding needs a value", received=sorted(entry),
                field=f"bindings[{index}].value",
            )
        _check_binding_value(entry["value"], field=f"bindings[{index}].value")
        origin = entry.get("from")
        _check_binding_origin(origin, field=f"bindings[{index}].from")
        bindings[name] = (
            {"value": entry["value"], "from": origin} if origin is not None
            else entry["value"]
        )
    return bindings


#: The published ``from`` bound; longer was silently truncated before EXP11B.
MAX_ORIGIN_CHARACTERS = 120


def _check_binding_name(name: Any, *, field: str) -> None:
    """The source-side assignment rule, applied to one binding name."""

    if not isinstance(name, str) or not name.isidentifier():
        raise _reject("a binding name must be a plain identifier", received=repr(name)[:60], field=field)
    if name.startswith("_"):
        raise _reject("binding names beginning with an underscore are not accepted", received=name, field=field)
    if name in kernel.RESERVED_NAMES:
        raise _reject(
            "this name is bound by the evaluator and cannot be shadowed",
            received=name, field=field, reserved=sorted(kernel.RESERVED_NAMES),
        )


def _count_elements(value: Any) -> int:
    if isinstance(value, (list, tuple)):
        return sum(_count_elements(item) for item in value)
    return 1


def _check_binding_value(value: Any, *, field: str) -> None:
    count = _count_elements(value)
    if count > kernel.MAX_VALUE_ELEMENTS:
        raise _reject(
            "a binding value has more elements than one value may carry",
            received=count, field=field, maximum_elements=kernel.MAX_VALUE_ELEMENTS,
        )


def _check_binding_origin(origin: Any, *, field: str) -> None:
    if origin is None:
        return
    if not isinstance(origin, str) or not origin.strip():
        raise _reject("a binding's 'from' must name the result it was read from",
                      received=type(origin).__name__, field=field)
    if len(origin) > MAX_ORIGIN_CHARACTERS:
        raise _reject("a binding's 'from' is longer than accepted", received=len(origin),
                      field=field, maximum_characters=MAX_ORIGIN_CHARACTERS)


def _split_bindings(
    bindings: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Separate the values the child evaluates from the record of where they came from.

    A binding is either a bare value the policy brought itself, or
    ``{"value": ..., "from": "<origin>"}`` naming the tool result it was read
    from. Both are accepted and neither is required: a reference form that were
    mandatory would block the policy from bringing a constant, which is a
    legitimate and common need. The annotation only buys a better receipt.

    The record uses fixed keys with the policy's identifier as a *value*, for
    the same reason the outputs do: ``_forbidden_path`` scans key paths.
    """

    values: dict[str, Any] = {}
    provenance: list[dict[str, Any]] = []
    for name in sorted(bindings):
        # The same rule the source side enforces on assignment targets. Without
        # it a binding silently replaces what the evaluator guarantees: `np` and
        # `Rotation` become whatever was passed, and `__builtins__` stops being
        # the empty mapping. JSON cannot carry a callable, so this is not a
        # route to execution -- but it voids the namespace contract and reports
        # a dependency's `KeyError: '__import__'` instead of a reason.
        _check_binding_name(name, field=f"bindings.{name}" if isinstance(name, str) else "bindings")
        entry = bindings[name]
        origin = "literal"
        value = entry
        if isinstance(entry, Mapping) and "value" in entry:
            value = entry["value"]
            declared = entry.get("from")
            if declared is not None:
                _check_binding_origin(declared, field=f"bindings.{name}.from")
                origin = declared.strip()
        values[name] = value
        provenance.append({"name": name, "origin": origin, "sha256": kernel._digest(value)})
    return values, provenance



def handle(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Tool-call entry point, so binding this tool costs one dispatch entry.

    Takes ``{"expressions": <block>, "bindings": {<name>: <value or annotated>}}``
    and returns the evaluator's result. Kept here rather than on a runtime class
    because the evaluator needs no control root, manual, event log or
    environment bridge -- it is a pure function over its arguments.
    """

    if not isinstance(arguments, Mapping):
        raise _reject(
            "compute takes an object with expressions and optional bindings",
            received=type(arguments).__name__,
        )
    unknown = sorted(set(arguments) - {"expressions", "bindings", "outputs"})
    if unknown:
        raise _reject(
            "unknown field",
            received=unknown, allowed=["expressions", "bindings", "outputs"],
        )
    submitted = arguments.get("expressions", "")
    source, normalization = normalize_block(submitted)
    result = shared_runtime().evaluate(
        source, read_bindings(arguments.get("bindings")),
        outputs=arguments.get("outputs"),
    )
    if normalization is not None:
        provenance = result.get("_runtime", {}).get("provenance")
        if isinstance(provenance, dict):
            provenance["source_normalization"] = {
                "kind": normalization,
                "submitted_source_sha256": kernel._digest(submitted),
            }
    return result


def invoke(host: Any, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate one block and check the result against what the surface declared.

    Validates the visible half -- ``_runtime`` popped, the way
    ``native/session.py`` assembles the tool message -- because a published
    schema that nothing checks against the delivered shape is a guard that
    cannot fail, and auditing the raw payload tests a string never put on the
    wire.
    """

    from ora.agent.composition import validate_schema_instance

    result = handle(arguments)
    schemas = host.surface.result_schemas
    published = tool_names.as_advertised(schemas, "compute")
    visible = {key: value for key, value in result.items() if key != "_runtime"}
    validate_schema_instance(visible, schemas[published])
    return result
