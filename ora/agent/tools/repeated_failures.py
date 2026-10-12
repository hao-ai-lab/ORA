"""Count byte-identical consecutive tool failures within one policy session.

A policy that resends a refused request unchanged receives the same refusal
every time, and nothing in a single result says that this has already
happened. The gateway that delivers results -- Native ``NativeToolGateway``
and the managed MCP façade -- owns one :class:`RepeatedFailureTracker` per
policy session and passes every completed call through :meth:`record`.

When a call repeats the immediately preceding call's tool name and canonical
arguments and fails with the same error code, the new result's ``error``
gains a ``repetition`` record: the count and one neutral sentence stating it.
The sentence adds that an unchanged resend fails again only when the refusal
itself says so (``retry_requires_changed_arguments``). It describes what the
tool did, never what the policy should do next.

Only the new result is annotated, and it is returned as a new object. Results
already delivered are never edited, so every request stays a strict extension
of the previous one, and identical successful results keep identical bytes for
exact deduplication. Any other call -- a different tool, different arguments,
a success, a different code or a call whose arguments did not decode -- resets
the count.
"""

from __future__ import annotations

import json
from typing import Any, Mapping


def canonical_arguments(arguments: Mapping[str, Any]) -> str | None:
    """Return the canonical JSON identity of one argument object, if any."""

    try:
        return json.dumps(
            arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError):
        return None


def repetition_note(count: int, code: str, *, unchanged_fails: bool) -> str:
    note = f"this identical request has failed {count} times in a row with {code}"
    return f"{note}; sending it unchanged will fail again" if unchanged_fails else note


class RepeatedFailureTracker:
    """The current streak of identical failing calls in one policy session."""

    def __init__(self) -> None:
        self._identity: tuple[str, str, str] | None = None
        self._count = 0

    def reset(self) -> None:
        self._identity = None
        self._count = 0

    def record(
        self, name: str, arguments: Mapping[str, Any] | None,
        result: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Record one completed call and return its result, annotated on repeat."""

        raw_error = result.get("error")
        error: Mapping[str, Any] = raw_error if isinstance(raw_error, Mapping) else {}
        code = error.get("code")
        canonical = (
            canonical_arguments(arguments) if isinstance(arguments, Mapping) else None
        )
        if not isinstance(code, str) or canonical is None:
            self.reset()
            return result
        identity = (str(name), canonical, code)
        self._count = self._count + 1 if identity == self._identity else 1
        self._identity = identity
        if self._count < 2:
            return result
        repetition = {
            "identical_failures_in_a_row": self._count,
            "note": repetition_note(
                self._count, code,
                unchanged_fails=error.get("retry_requires_changed_arguments") is True,
            ),
        }
        return {**result, "error": {**error, "repetition": repetition}}
