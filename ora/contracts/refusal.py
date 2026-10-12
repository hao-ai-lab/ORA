"""Public pre-execution input refusals shared by Agent and Environment."""

from __future__ import annotations

import copy
from typing import Any, Mapping


class EnvironmentRefusal(RuntimeError):
    """A declared public input refusal before the requested operation executes.

    Backends raise this only at known argument boundaries. Ordinary validation,
    solver and implementation exceptions are not implicitly converted to it.
    The fields must describe public input and admissible values, never private
    simulator state or worker implementation details.
    """

    code = "invalid_arguments"
    retryable = True

    def __init__(self, message: str, *, field: str, received: Any, allowed: Any) -> None:
        if not isinstance(message, str) or not message or not isinstance(field, str) or not field:
            raise ValueError("environment refusal requires a message and public field")
        if allowed is None:
            raise ValueError("environment refusal requires allowed input values")
        super().__init__(message)
        self.field = field
        self.received = copy.deepcopy(received)
        self.allowed = copy.deepcopy(allowed)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "refusal", "type": "EnvironmentRefusal",
            "code": self.code, "message": str(self), "retryable": self.retryable,
            "field": self.field, "received": copy.deepcopy(self.received),
            "constraints": {"allowed": copy.deepcopy(self.allowed)},
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EnvironmentRefusal:
        constraints = value.get("constraints")
        if (
            set(value) != {"kind", "type", "code", "message", "retryable", "field", "received", "constraints"}
            or value.get("kind") != "refusal" or value.get("type") != "EnvironmentRefusal"
            or value.get("code") != cls.code or value.get("retryable") is not True
            or not isinstance(constraints, Mapping) or set(constraints) != {"allowed"}
        ):
            raise ValueError("environment refusal envelope is malformed")
        return cls(
            value["message"], field=value["field"], received=value["received"],
            allowed=constraints["allowed"],
        )
