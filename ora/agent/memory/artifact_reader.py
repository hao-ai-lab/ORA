"""``artifact_read`` and ``observe(ref)``: argument refusals and the bounded read.

The two Context readers are added to a surface at runtime and reach no input
schema validation. Their argument errors were raised as ``ValueError``,
``PermissionError`` or ``FileNotFoundError`` and arrived as
``internal_tool_error`` -- infrastructure, not retryable -- although a changed
ref, offset, limit or pointer fixes every one of them (EXP11B census 6, 8),
and an infrastructure code a policy argument can raise can unlock abort. A
``json_pointer`` of ``""`` returned the whole JSON root with no size bound
(census A5), where a byte range is bounded by ``ARTIFACT_READ_LIMIT_BYTES``.

Here each argument is checked by name before the store is read, then against
the published schema, a ref the store cannot resolve is refused as the
argument or -- when the store itself failed -- with an infrastructure code,
and a selected JSON value is bounded like a byte range.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Mapping

from ora.agent.composition.schema_validation import (
    ToolSurfaceError, validate_schema_instance,
)
from ora.agent.tools.tool_errors import ToolError, schema_refusal

#: The published ``artifact://`` reference shape (``artifacts._REF_PATTERN``).
REF_PATTERN = r"^artifact://[A-Za-z0-9][A-Za-z0-9._-]{0,159}/[0-9a-f]{64}$"
READ_ARGUMENTS = ("ref", "offset", "limit", "json_pointer")


class RefSyntaxError(ValueError):
    """Not an ``artifact://`` reference."""


class RefScopeError(PermissionError):
    """A reference of another PolicySession."""


class RefUnknownError(FileNotFoundError):
    """A well-formed reference this catalog never registered."""


class RefStorageError(FileNotFoundError):
    """A registered reference whose stored file is missing or escapes the run."""


class RefHashError(ValueError):
    """A stored file that no longer matches its registered hash."""


#: The catalog's own exception classes (kept for its internal callers), by refusal kind.
_KINDS = ((RefSyntaxError, "syntax"), (RefScopeError, "scope"), (RefUnknownError, "unknown"),
          (RefStorageError, "storage"), (RefHashError, "hash"))


def policy_resolve(catalog: Any, ref: str) -> tuple[dict[str, Any], bytes]:
    """``catalog.resolve`` for a policy call: each failure as its refusal."""

    try:
        return catalog.resolve(ref)
    except (RefSyntaxError, RefScopeError, RefUnknownError, RefStorageError, RefHashError) as exc:
        kind = next(kind for klass, kind in _KINDS if isinstance(exc, klass))
        raise ref_refusal(kind, ref, scope=getattr(catalog, "policy_session_id", None)
                          if kind == "scope" else None) from exc


def ref_refusal(kind: str, ref: Any, *, scope: str | None = None) -> ToolError:
    """A reference the catalog refused: the argument, or the store behind it."""

    if kind == "storage":
        return ToolError("the artifact store has no readable file for this reference",
                         code="artifact_integrity_fault", retryable=False, field="ref", received=ref)
    if kind == "hash":
        return ToolError("artifact content no longer matches its reference hash",
                         code="artifact_hash_mismatch", retryable=False, field="ref", received=ref)
    message = {
        "syntax": "ref is not an artifact:// reference",
        "scope": "ref belongs to another PolicySession",
        "unknown": "no artifact in this PolicySession carries that ref",
    }[kind]
    constraints: dict[str, Any] = {"pattern": REF_PATTERN}
    if scope is not None:
        constraints["policy_session_id"] = scope
    return ToolError(message, code="invalid_arguments", field="ref", received=ref,
                     constraints=constraints)


def _refuse(field: str, message: str, received: Any, **constraints: Any) -> ToolError:
    return ToolError(message, code="invalid_arguments", field=field, received=received,
                     constraints=constraints)


def _validate(arguments: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    try:
        validate_schema_instance(dict(arguments), schema, root_path="$input")
    except ToolSurfaceError as error:
        raise schema_refusal(error) from error


def check_read_arguments(arguments: Mapping[str, Any], schema: Mapping[str, Any], limit: int) -> None:
    """Each ``artifact_read`` argument by name, then the published schema."""

    unknown = sorted(set(arguments) - set(READ_ARGUMENTS))
    if unknown:
        raise _refuse(unknown[0], f"artifact_read does not take {', '.join(unknown)}", unknown,
                      allowed=list(READ_ARGUMENTS))
    ref = arguments.get("ref")
    if not isinstance(ref, str) or not ref:
        raise _refuse("ref", "ref must be a non-empty artifact:// reference", ref, pattern=REF_PATTERN)
    pointer = arguments.get("json_pointer")
    if pointer is not None and not isinstance(pointer, str):
        raise _refuse("json_pointer", "json_pointer must be a string or null", pointer,
                      type=["string", "null"])
    offset, size = arguments.get("offset"), arguments.get("limit")
    if offset is not None and (type(offset) is not int or offset < 0):
        raise _refuse("offset", "offset must be a non-negative integer byte offset or null", offset,
                      minimum=0)
    if size is not None and (type(size) is not int or not 1 <= size <= limit):
        raise _refuse("limit", f"limit must be an integer byte count in [1, {limit}] or null", size,
                      minimum=1, maximum=limit)
    if pointer is not None and (offset not in {None, 0} or size not in {None, 4096}):
        raise _refuse(
            "json_pointer", "json_pointer cannot be combined with an explicit range; "
            "omit offset and limit to select a JSON value",
            {"json_pointer": pointer, "offset": offset, "limit": size},
            allowed=["ref + json_pointer", "ref + optional offset/limit"])
    _validate(arguments, schema)


#: ``observe``'s historical branch, as ``recoverable_artifact_definitions`` publishes it.
HISTORICAL_OBSERVE_INPUT = {
    "type": "object", "required": ["ref"], "additionalProperties": False,
    "properties": {"ref": {"type": "string", "pattern": REF_PATTERN}},
}


def check_observe_ref_arguments(arguments: Mapping[str, Any]) -> None:
    """``observe(ref)`` takes ``ref`` alone."""

    if set(arguments) != {"ref"}:
        others = sorted(set(arguments) - {"ref"})
        raise _refuse(others[0] if others else "ref",
                      "observe with ref reopens one historical image and takes ref alone",
                      sorted(arguments), allowed=["ref"])
    _validate(arguments, HISTORICAL_OBSERVE_INPUT)


def json_pointer(value: Any, pointer: str) -> Any:
    """RFC 6901 selection; ``""`` is the root."""

    if pointer == "":
        return copy.deepcopy(value)
    if not pointer.startswith("/"):
        raise ValueError("json_pointer must be empty or start with /")
    current = value
    for raw in pointer[1:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Mapping):
            if token not in current:
                raise KeyError("json_pointer does not exist")
            current = current[token]
        elif isinstance(current, list):
            if not token.isdigit() or int(token) >= len(current):
                raise KeyError("json_pointer does not exist")
            current = current[int(token)]
        else:
            raise KeyError("json_pointer crosses a scalar")
    return copy.deepcopy(current)


def _shape(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {"value_type": "object", "keys": sorted(str(key) for key in value)}
    if isinstance(value, list):
        return {"value_type": "array", "length": len(value)}
    return {"value_type": "scalar"}


def bounded_pointer_value(selected: Any, pointer: str, limit: int) -> Any:
    """Refuse a selected value larger than one byte-range read."""

    size = len(json.dumps(selected, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    if size <= limit:
        return selected
    raise _refuse("json_pointer",
                  f"the value at json_pointer {pointer!r} is {size} bytes, more than the "
                  f"{limit}-byte read limit", pointer,
                  value_bytes=size, maximum_bytes=limit, **_shape(selected))


def read_range(data: bytes, offset: int, limit: int) -> tuple[str, int]:
    """One UTF-8 text range starting at ``offset``; refuses ranges that cannot decode."""

    if offset > len(data):
        raise _refuse("offset", f"offset {offset} is past the artifact's {len(data)} bytes", offset,
                      minimum=0, maximum=len(data))
    end = min(len(data), offset + limit)
    while True:
        chunk = data[offset:end]
        try:
            text = chunk.decode("utf-8")
        except UnicodeDecodeError as exc:
            if exc.reason == "unexpected end of data" and end > offset:
                end -= 1
                continue
            if exc.start == 0:
                raise _refuse("offset", "offset does not start a UTF-8 character", offset,
                              rule="offset at a UTF-8 character boundary") from exc
            raise ToolError("artifact range is not UTF-8 text", code="artifact_integrity_fault",
                            retryable=False, field="ref") from exc
        if not text and offset < len(data):
            raise _refuse("limit", f"limit {limit} is shorter than the UTF-8 character at offset "
                          f"{offset}", limit, minimum=4)
        return text, end


__all__ = [
    "RefHashError", "RefScopeError", "RefStorageError", "RefSyntaxError", "RefUnknownError",
    "policy_resolve", "HISTORICAL_OBSERVE_INPUT", "READ_ARGUMENTS", "REF_PATTERN", "bounded_pointer_value",
    "check_observe_ref_arguments", "check_read_arguments", "json_pointer", "read_range",
    "ref_refusal",
]
