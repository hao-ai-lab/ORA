"""The infrastructure signals an evidence-gated abort rests on.

``finish_outcomes`` admits ``abort`` (``policy_abort_infrastructure_round_v2``)
only when these signals exist. The reader reads the run's own event log -- the file the tool runtime
appends its ``tool_error`` events to and the Supervisor its ``motion_receipt``
and ``reset`` events -- from the current round's ``reset`` on, and keeps the
entries ``composition/abort_evidence_schema.py`` classifies as infrastructure.
Nothing the policy says enters the decision. The classes and the recorded
evidence are documented there.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from ora.agent.composition.abort_evidence_schema import (
    EXECUTION_FAULT_CODES, PLANNER_FAULT_CODES, SKILL_FAULT_STOP_REASONS, TOOL_ERROR_CODES,
)

_KINDS = ("tool_error", "motion_receipt", "reset")
_MARKERS = tuple(f'"type": "{kind}"' for kind in _KINDS)


def _event(line: str) -> Mapping[str, Any] | None:
    """One relevant event of the log, or ``None`` for any other or unreadable line."""
    if not any(marker in line for marker in _MARKERS):
        return None
    try:
        value = json.loads(line)
    except ValueError:
        return None
    return value if isinstance(value, Mapping) and value.get("type") in _KINDS else None


def _tool_error_signal(event: Mapping[str, Any]) -> list[dict[str, Any]]:
    error = event.get("error")
    code = error.get("code") if isinstance(error, Mapping) else None
    if code not in TOOL_ERROR_CODES:
        return []
    return [{"source": "tool_error", "tool": str(event.get("tool")), "code": str(code)}]


def _fault_class(receipt: Mapping[str, Any]) -> str | None:
    """Why one backend receipt is an infrastructure fault, or ``None``."""
    stage, code = receipt.get("failure_stage"), receipt.get("failure_code")
    if receipt.get("execution_status") == "execution_fault" or isinstance(receipt.get("execution_fault"), Mapping):
        return "execution_fault"
    if receipt.get("execution_uncertain") is True:
        return "execution_uncertain"
    if stage == "planner" and code in PLANNER_FAULT_CODES:
        return "planner_error"
    if stage == "execution" and code in EXECUTION_FAULT_CODES:
        return str(code)
    return None


def _receipt_signals(record: Any) -> list[dict[str, Any]]:
    """Every fault in one receipt; a skill sequence is read stage by stage."""
    if not isinstance(record, Mapping):
        return []
    if isinstance(record.get("receipt"), Mapping) and "path" in record:
        return _receipt_signals(record["receipt"])
    if isinstance(record.get("stage_receipts"), list):
        found = ([{"source": "environment_receipt", "action_id": record.get("action_id"),
                   "class": str(record["stop_reason"])}]
                 if record.get("stop_reason") in SKILL_FAULT_STOP_REASONS else [])
        for stage in record["stage_receipts"]:
            found.extend(_receipt_signals(stage.get("receipt") if isinstance(stage, Mapping) else None))
        return found
    fault = _fault_class(record)
    if fault is None:
        return []
    return [{"source": "environment_receipt", "action_id": record.get("action_id"),
             "action": record.get("action"), "class": fault,
             "failure_stage": record.get("failure_stage"), "failure_code": record.get("failure_code")}]


def round_signals(lines: Iterable[str]) -> list[dict[str, Any]]:
    """Infrastructure signals recorded since the last round reset in these log lines."""
    signals: list[dict[str, Any]] = []
    for line in lines:
        event = _event(line)
        if event is None:
            continue
        if event["type"] == "reset":
            signals = []
        elif event["type"] == "tool_error":
            signals.extend(_tool_error_signal(event))
        else:
            signals.extend(_receipt_signals(event.get("record")))
    return signals


def recorded_signals(event_path: Path) -> list[dict[str, Any]]:
    """The current round's infrastructure signals in the run's event log."""
    if not event_path.is_file():
        return []
    with event_path.open(encoding="utf-8") as stream:
        return round_signals(stream)


__all__ = ["recorded_signals", "round_signals"]
