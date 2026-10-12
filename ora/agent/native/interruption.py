"""Native turn outcome after an operator interrupt.

The Supervisor's stop handler interrupts the session from another thread and
closes its provider. Whatever the in-flight turn yields afterwards -- a
projection that finished, a request failure caused by the close, or a reply
that completed anyway -- ends the turn with this outcome. No further request
starts and no returned tool call runs.
"""

from __future__ import annotations

from typing import Any


def operator_interrupted(attempts: int) -> dict[str, Any]:
    """``attempts`` counts provider requests already started in this turn."""

    return {
        "submitted": False,
        "termination": "operator_interrupt",
        "attempts": attempts,
    }
