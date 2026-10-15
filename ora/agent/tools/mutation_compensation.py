"""Undo the action a failed mutation call persisted, so the observation stays open.

A mutation handler persists ``submitted-action.json`` before its result is
validated. A failure after that point -- the result schema, a post-submit
annotation -- used to leave the action persisted, which closes the current
observation for the rest of the turn, and was not logged. The runtime takes a ``snapshot`` before the handler and, on any failure
of a mutation call, ``restore`` removes only the action this call created and
its entry in the submitted-mutation ledger. An action already pending before
the call is kept, so a refused second mutation cannot delete the first.
Holding-gate transitions are left as recorded: a refused close deliberately
records what it consumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class Snapshot:
    action_path: Path | None
    action_existed: bool
    submitted_count: int


def snapshot(host: Any, name: str, arguments: Mapping[str, Any]) -> Snapshot:
    control_root = getattr(host, "control_root", None)
    try:
        mutation = host.classify_call_effect(name, arguments) == "mutation"
    except Exception:  # noqa: BLE001 - an unclassifiable call submits nothing
        mutation = False
    action_path = control_root / "submitted-action.json" if mutation and isinstance(control_root, Path) else None
    return Snapshot(
        action_path=action_path,
        action_existed=action_path is not None and action_path.exists(),
        submitted_count=len(getattr(host, "_submitted_mutations", ()) or ()),
    )


def restore(host: Any, taken: Snapshot) -> None:
    """Remove the action and ledger entry the failed call created; keep anything older."""

    if taken.action_path is None:
        return
    if not taken.action_existed:
        taken.action_path.unlink(missing_ok=True)
    submitted = getattr(host, "_submitted_mutations", None)
    if isinstance(submitted, list):
        del submitted[taken.submitted_count:]
