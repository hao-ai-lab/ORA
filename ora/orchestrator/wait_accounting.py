"""Per-round wait accounting: how much of a round's native-action use was waiting.

EXP11A g012 (E11G12-005): with the policy failure exit withheld, 300
``wait_ticks`` calls consumed 5,663 of the 13,071 native actions its seven
cells used (43.3%), and 280 of them came after their round's last non-wait
proposal, several explicitly to spend the remaining budget. Each round result
named only the boundary that ended it (``robot_decision_budget``,
``trusted_runtime_boundary``).

Measurement only. Nothing here restricts, refuses or reinterprets a wait, and
the termination, its authority and the official verdict are untouched.

Over the round's dispatched proposals in order -- each ``action_submitted``
with its ``motion_receipt``, the pair the Supervisor executes:

* a *wait* is the shared motion-v3 ``wait`` action (the ``wait_ticks`` tool);
* the *last non-wait action* is the last dispatched proposal that is not a
  wait, whether it executed or was refused: a refused move is still an attempt
  to change the world. ``index`` is its 0-based position among the round's
  ``dispatched_actions``;
* a *terminal wait* is a wait after it, or every wait when there is none.
  Read-only tool calls and tool-less answers dispatch nothing, so they never
  end the terminal segment;
* native actions are each receipt's own count
  (``native_allowance.receipt_native_actions``): what executed, not the ticks
  requested. One receipt without a count makes every sum it belongs to
  ``None`` rather than an estimate; the call counts stay exact.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from ora.contracts.wait import WAIT_ACTION

from .native_allowance import NativeActionAllowance, receipt_native_actions

ROUND_SCHEMA = "ora.round-wait-accounting.v1"
SESSION_SCHEMA = "ora.session-wait-accounting.v1"
#: Session totals: every round field that sums.
SUMMED_FIELDS = (
    "dispatched_actions", "native_actions", "wait_calls", "wait_native_actions",
    "terminal_wait_calls", "terminal_wait_native_actions",
)


@dataclass(frozen=True)
class _Dispatched:
    wait: bool
    native_actions: int | None
    action_id: str | None
    action: str | None


def _dispatched(action: Any, receipt: Any) -> _Dispatched:
    record = action if isinstance(action, Mapping) else {}
    name = record.get("action")
    action_id = record.get("action_id")
    return _Dispatched(
        wait=name == WAIT_ACTION,
        native_actions=receipt_native_actions(receipt),
        action_id=str(action_id) if action_id is not None else None,
        action=str(name) if name is not None else None,
    )


def _sum(items: Iterable[_Dispatched]) -> int | None:
    total = 0
    for item in items:
        if item.native_actions is None:
            return None
        total += item.native_actions
    return total


def _summary(entries: Sequence[_Dispatched]) -> dict[str, Any]:
    last = max((index for index, item in enumerate(entries) if not item.wait), default=None)
    waits = [item for item in entries if item.wait]
    terminal = [item for index, item in enumerate(entries)
                if item.wait and (last is None or index > last)]
    return {
        "schema": ROUND_SCHEMA,
        "dispatched_actions": len(entries),
        "native_actions": _sum(entries),
        "wait_calls": len(waits),
        "wait_native_actions": _sum(waits),
        "terminal_wait_calls": len(terminal),
        "terminal_wait_native_actions": _sum(terminal),
        "last_non_wait_action": None if last is None else {
            "index": last, "action_id": entries[last].action_id, "action": entries[last].action,
        },
    }


def round_wait_accounting(entries: Iterable[tuple[Any, Any]]) -> dict[str, Any]:
    """The round block for ``(action, receipt)`` pairs in dispatch order."""
    return _summary([_dispatched(action, receipt) for action, receipt in entries])


class RoundWaitAccounting:
    """The Supervisor's live accumulator; ``snapshot`` is the persisted block."""

    def __init__(self) -> None:
        self._entries: list[_Dispatched] = []

    def reset(self) -> None:
        self._entries = []

    def record(self, action: Any, receipt: Any) -> None:
        self._entries = [*self._entries, _dispatched(action, receipt)]

    def snapshot(self) -> dict[str, Any]:
        return _summary(self._entries)


class RoundActionLedger(NativeActionAllowance):
    """The round's native-action allowance and its wait accounting, reset together.

    The Supervisor records every executed receipt once; the allowance keeps
    publishing its remaining-budget fields unchanged.
    """

    def __init__(self, plan: Mapping[str, Any]) -> None:
        super().__init__(plan)
        self.waits = RoundWaitAccounting()

    def reset(self) -> None:
        super().reset()
        self.waits.reset()

    def record(self, receipt: Any, action: Any = None) -> None:
        super().record(receipt)
        self.waits.record(action, receipt)

    def wait_accounting(self) -> dict[str, Any]:
        return self.waits.snapshot()


def rounds_from_events(events: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Replay every finished round's block from ``events.jsonl`` rows.

    A ``motion_receipt`` pairs with the latest ``action_submitted``; a round
    closes at its ``round_finished``. Proposals after the last one belong to no
    finished round and are left out.
    """
    rounds: list[dict[str, Any]] = []
    current: list[_Dispatched] = []
    pending: Any = None
    for event in events:
        kind = event.get("type")
        if kind == "action_submitted":
            pending = event.get("action")
        elif kind == "motion_receipt":
            current = [*current, _dispatched(pending, event.get("record"))]
            pending = None
        elif kind == "round_finished":
            rounds = [*rounds, _summary(current)]
            current, pending = [], None
    return rounds


def _total(values: Sequence[Any]) -> int | None:
    if any(type(value) is not int for value in values):
        return None
    return sum(values)


def session_wait_accounting(round_records: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Session totals over the round blocks, or ``None`` when any round has none."""
    blocks = [record.get("wait_accounting") for record in round_records]
    if not blocks or not all(isinstance(block, Mapping) for block in blocks):
        return None
    return {
        "schema": SESSION_SCHEMA, "rounds": len(blocks),
        **{field: _total([block.get(field) for block in blocks]) for field in SUMMED_FIELDS},
    }


__all__ = [
    "ROUND_SCHEMA", "SESSION_SCHEMA", "SUMMED_FIELDS", "RoundActionLedger", "RoundWaitAccounting",
    "round_wait_accounting",
    "rounds_from_events", "session_wait_accounting",
]
