"""The run result's fields derived from its round records.

A multi-round session result reports how many rounds ran, how many the
official verdict called successful and, when every round carries one, the sum
of the rounds' ``wait_accounting`` blocks. A single-episode run result has no
persisted round record, so it carries its one round's block. Decisions,
refusals and model calls are counted live by the Supervisor; evaluation
reconciles them and the wait totals with the rounds
(``evaluation.verify_multi_round_session``).
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from .wait_accounting import session_wait_accounting


def run_round_aggregates(
    round_results: Sequence[Mapping[str, Any]], *, multi_round: bool,
) -> dict[str, Any]:
    """The fields a run or session result adds from its finished rounds."""
    if not multi_round:
        block = round_results[0].get("wait_accounting") if round_results else None
        return {} if block is None else {"wait_accounting": block}
    fields: dict[str, Any] = {
        "round_count": len(round_results),
        "round_success_count": sum(item.get("official_success") is True for item in round_results),
    }
    totals = session_wait_accounting(round_results)
    return fields if totals is None else {**fields, "wait_accounting": totals}


__all__ = ["run_round_aggregates"]
