"""Operator stop versus an in-flight policy turn.

`ora stop` sends SIGTERM to the Supervisor. Its handler marks the
episode interrupted and interrupts the PolicySession from another thread,
which closes the provider under any in-flight request. The turn can therefore
end in a closed-client or broken-stream error, or return a reply that was
already on the wire. Either result is the stop itself: it is converted into an
operator-owned `NoAction` so the Supervisor's ordinary outcome path ends the
session with operator authority, executes nothing and records no
infrastructure fault (RoboDojo D3R-006). Without a stop, errors propagate
unchanged and remain infrastructure faults.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable

from ora.agent import ActionProposal, NoAction, PolicyOutcome

OPERATOR_INTERRUPT = "operator_interrupt"


def act_unless_stopped(
    planner: Any, turn: Any, stopped: Callable[[], bool], events: Path,
) -> PolicyOutcome:
    """Run one policy turn; after a stop, return the operator outcome."""

    try:
        outcome: PolicyOutcome | None = planner.act(turn)
        error: Exception | None = None
    except Exception as exc:
        if not stopped():
            raise
        outcome, error = None, exc
    if not stopped():
        return outcome
    if isinstance(outcome, NoAction) and outcome.reason == OPERATOR_INTERRUPT:
        return outcome
    model_calls = 0 if outcome is None else outcome.model_calls
    record = {
        "type": "policy_turn_cancelled",
        "reason": OPERATOR_INTERRUPT,
        "policy_outcome": None if outcome is None else outcome.kind,
        "discarded_action": isinstance(outcome, ActionProposal),
        "error_type": None if error is None else type(error).__name__,
        "error": None if error is None else str(error),
        "model_calls": model_calls,
        "time": time.time(),
    }
    _append(events, record)
    return NoAction(
        reason=OPERATOR_INTERRUPT,
        model_calls=model_calls,
        details={key: value for key, value in record.items() if key != "type"},
    )


def _append(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


__all__ = ["act_unless_stopped"]
