"""The shared, policy-visible tool error envelope and its argument rules.

Every policy tool refuses through :class:`ToolError`. Its ``as_dict()`` is the
one structured envelope the Native and managed gateways deliver:

``code``
    Stable refusal identity.
``message``
    Concise text.
``retryable``
    ``True`` when the policy can still recover through a later call of this
    tool -- with corrected arguments, or after the public state the refusal
    depends on changes. ``False`` when no later policy call of this tool can
    (an internal fault, a trusted terminal boundary, a withheld exit, an
    unavailable tool). It is not an instruction to resend the same request,
    and every consumer -- the environment refusal RPC contract, the Native
    nameless-call inventory, frozen tool manuals -- reads it this way.
``retry_requires_changed_arguments``
    Present only when known. ``True`` for an argument refusal: the Runtime
    refused the request's own argument values before anything executed, so the
    unchanged request is refused again against the same public state and only
    a changed request can succeed. Derived from ``ARGUMENT_REFUSAL_CODES``
    unless the raising site states it.
``field``, ``received``, ``constraints``
    The public argument, what was received and the violated rule, when known.
``infrastructure``
    Whether the code says the Harness, a model service, the bridge or a sensor
    failed rather than the policy's request: membership of
    ``agent.error_classes.INFRASTRUCTURE_CODES``, the table the abort evidence
    gate counts.

The ``require_*`` helpers below are the shared form of mutually exclusive
argument rules. They report argument *names* only, never values.
"""

from __future__ import annotations

from typing import Any, Collection, Mapping, Sequence


#: Codes that mean "these argument values were refused before execution".
ARGUMENT_REFUSAL_CODES = frozenset({
    "invalid_arguments",
    "invalid_motion_arguments",
    "invalid_servo_delta",
    "invalid_tool_call",
    "out_of_bounds",
    # A malformed or unknown target ref; a store fault behind a ref carries an
    # artifact code instead (``target_refusals.unavailable_target``).
    "target_ref_unavailable",
    "unknown_section",
    "workspace_bounds",
})


from ora.agent.error_classes import INFRASTRUCTURE_CODES  # noqa: E402


class ToolError(RuntimeError):
    """A structured, policy-visible tool failure."""

    def __init__(
        self, message: str, *, code: str = "invalid_arguments",
        retryable: bool = True, field: str | None = None,
        received: Any = None, constraints: Mapping[str, Any] | None = None,
        retry_requires_changed_arguments: bool | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.field = field
        self.received = received
        self.constraints = dict(constraints or {})
        if retry_requires_changed_arguments is None and code in ARGUMENT_REFUSAL_CODES:
            retry_requires_changed_arguments = True
        self.retry_requires_changed_arguments = retry_requires_changed_arguments

    def as_dict(self) -> dict[str, Any]:
        reported = {
            "code": self.code,
            "message": str(self),
            "retryable": self.retryable,
            "field": self.field,
            "received": self.received,
            "constraints": self.constraints,
            "infrastructure": self.code in INFRASTRUCTURE_CODES,
        }
        if self.retry_requires_changed_arguments is not None:
            reported["retry_requires_changed_arguments"] = bool(
                self.retry_requires_changed_arguments
            )
        return reported


def schema_refusal(error: Any) -> ToolError:
    """The policy-visible refusal for one input SchemaViolation.

    ``field`` is the argument path (``targets.left.position_m``), ``received``
    the shape that arrived (type, length, member names -- never values) and
    ``constraints`` the violated keyword with its bound.
    """

    path = str(getattr(error, "path", "") or "")
    field = path[len("$input."):] if path.startswith("$input.") else None
    keyword = getattr(error, "keyword", None)
    constraints = dict(getattr(error, "constraints", {}) or {})
    return ToolError(
        str(error), code="invalid_arguments", field=field,
        received=getattr(error, "received", None),
        constraints={"violated": keyword, **constraints} if keyword else constraints,
    )


#: The rule a free-text argument meets beyond its declared ``minLength``: at
#: least one non-whitespace character.
NON_WHITESPACE_TEXT = {"minLength": 1, "rule": "non_whitespace"}


def blank_text_refusal(name: str, value: Any, *, message: str) -> ToolError:
    """An empty or whitespace-only text argument, refused with its field and rule.

    ``received`` is the shape (``{"type": "string", "length": n}``), never the
    text; a missing argument has none.
    """

    received = {"type": "string", "length": len(value)} if isinstance(value, str) else None
    return ToolError(
        message, code="invalid_arguments", field=name, received=received,
        constraints=dict(NON_WHITESPACE_TEXT),
    )


def argument_names(arguments: Mapping[str, Any]) -> list[str]:
    """The received argument names, sorted; values are never reported."""

    return sorted(str(name) for name in arguments)


def _refuse(
    arguments: Mapping[str, Any], message: str, constraints: Mapping[str, Any],
) -> ToolError:
    return ToolError(
        message, code="invalid_arguments", received=argument_names(arguments),
        constraints=constraints,
    )


def require_exactly_one_of(
    arguments: Mapping[str, Any], names: Sequence[str], *, message: str,
) -> str:
    """Return the one present name; a present key counts even when null."""

    present = [name for name in names if name in arguments]
    if len(present) != 1:
        raise _refuse(arguments, message, {
            "exactly_one_of": list(names), "received": present,
        })
    return present[0]


def require_at_most_one_of(
    arguments: Mapping[str, Any], names: Sequence[str], *, message: str,
) -> str | None:
    """Return the provided name, if any; an explicit null counts as omitted."""

    present = [name for name in names if arguments.get(name) is not None]
    if len(present) > 1:
        raise _refuse(arguments, message, {
            "at_most_one_of": list(names), "received": present,
        })
    return present[0] if present else None


def require_known_arguments(
    arguments: Mapping[str, Any], *, allowed: Collection[str],
    required: Collection[str], message: str,
) -> None:
    """Refuse unknown or missing argument names, naming both sets."""

    unknown = sorted(set(arguments) - set(allowed))
    missing = sorted(set(required) - set(arguments))
    if unknown or missing:
        raise _refuse(arguments, message, {
            "allowed": sorted(allowed), "required": sorted(required),
            "unknown": unknown, "missing": missing,
        })
