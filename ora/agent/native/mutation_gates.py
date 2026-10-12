"""Typed-slot rules a proposed world mutation must pass before it executes.

``native_slots_v3`` (and ``_v4``) close world mutation after the trusted
terminal boundary.
"""

from __future__ import annotations

from ora.agent.tools.robot import ToolError

from .constants import NATIVE_CAUSAL_SLOTS


def _terminal_boundary_error() -> ToolError:
    return ToolError(
        "world mutation is unavailable after the trusted terminal boundary",
        code="episode_terminal", retryable=False,
    )


def mutation_protocol_error(
    *, composition: str | None, effect: str | None, terminal_lifecycle: bool,
) -> Exception | None:
    """The error that refuses this call, or None when it may execute."""

    if composition in NATIVE_CAUSAL_SLOTS and effect == "mutation" and terminal_lifecycle:
        return _terminal_boundary_error()
    return None
