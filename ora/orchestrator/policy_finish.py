"""The policy's finish outcome on the round record.

Before this module nothing downstream read the outcome of ``finish_episode``:
every lifecycle request -- abort included -- became a policy-lifecycle round
boundary with the official verdict persisted and a valid behavioural sample.
On a surface declaring the v2 finish contract (``policy_finish_outcomes_v2``)
the Supervisor now reads the Harness-written ``policy-finish.json`` of the
round and

- carries a summary (outcome, category, reason, abort evidence count and the
  give_up candidate ledger's verdict) on the ``termination_authority_decision``
  event, the round record and a single run's ``result.json``;
- ends the round at a round-scoped infrastructure boundary when the finish is
  an abort the tool runtime admitted on recorded infrastructure evidence
  (``policy_abort_infrastructure_round_v2``): the environment round is
  aborted, no official verdict is persisted, the sample is invalid and the
  session continues with its next round. A give_up keeps the policy-lifecycle
  boundary: a valid behavioural failure sample with the official verdict kept.

The termination literal stays ``policy_finished_episode`` in every case, so
backend comparability rules keyed on it (RoboDojo ``round_end_basis``) are
unchanged. A finish in the terminal turn after an environment boundary is
recorded with ``ended_round: false`` and changes no authority. A surface
without the capability gets no summary and keeps its records exactly.

Evaluation reads the same summary (``policy_finish_reasons``,
``admitted_abort_round``, ``policy_finish_report``) and binds each one to the
round's archived finish record and decision event, so a result edited after
the run cannot relabel a round.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ora.agent.composition.finish_outcomes_schema import (
    ABORT, ABORT_ADMISSION, CAPABILITY, CATEGORIES, OUTCOMES,
)

FINISH_RECORD = "policy-finish.json"
SUMMARY_SCHEMA = "ora.policy-finish-summary.v1"
POLICY_FINISHED = "policy_finished_episode"
_LEDGER_KEYS = ("methods", "untried_candidate_ids", "named_untried_candidate_ids", "reason_names_untried")


def summarize(raw: bytes, *, ended_round: bool) -> dict[str, Any] | None:
    """The round summary of one finish record's bytes, or ``None`` for a released record."""

    record = json.loads(raw)
    if not isinstance(record, Mapping) or record.get("finish_contract") != CAPABILITY:
        return None
    evidence = record.get("abort_infrastructure_evidence")
    ledger = record.get("untried_candidates")
    return {
        "schema": SUMMARY_SCHEMA,
        "contract": CAPABILITY,
        "outcome": record.get("outcome"),
        "category": record.get("category"),
        "reason": record.get("reason"),
        "ended_round": ended_round,
        "abort_admission": record.get("abort_admission"),
        "abort_infrastructure_signal_count": len(evidence) if isinstance(evidence, list) else None,
        "untried_candidates": (
            {key: ledger.get(key) for key in _LEDGER_KEYS} if isinstance(ledger, Mapping) else None
        ),
        "record_sha256": hashlib.sha256(raw).hexdigest(),
    }


def read_policy_finish(control: Path, surface: Any, *, ended_round: bool) -> dict[str, Any] | None:
    """The summary of this round's v2 finish record, or ``None``."""

    if not surface.supports(CAPABILITY):
        return None
    path = control / FINISH_RECORD
    return summarize(path.read_bytes(), ended_round=ended_round) if path.is_file() else None


def infrastructure_abort(summary: Any) -> bool:
    """Whether a summary is an abort admitted on recorded infrastructure that ended its round."""

    count = summary.get("abort_infrastructure_signal_count") if isinstance(summary, Mapping) else None
    return (
        isinstance(summary, Mapping)
        and summary.get("outcome") == ABORT
        and summary.get("abort_admission") == ABORT_ADMISSION
        and summary.get("ended_round") is True
        and isinstance(count, int) and not isinstance(count, bool) and count > 0
    )


def admitted_abort_round(record: Mapping[str, Any]) -> bool:
    """Whether a round (or single-run) record is a round an admitted abort ended as invalid."""

    return (
        infrastructure_abort(record.get("policy_finish"))
        and record.get("termination") == POLICY_FINISHED
        and record.get("termination_authority") == "infrastructure"
        and record.get("behavioral_sample_valid") is False
        and record.get("official_success") is None
    )


def _records(result: Mapping[str, Any]) -> list[tuple[int, Mapping[str, Any]]]:
    rounds = result.get("rounds")
    if isinstance(rounds, list):
        return [(index, record) for index, record in enumerate(rounds) if isinstance(record, Mapping)]
    return [(0, result)]


def _record_reason(index: int, record: Mapping[str, Any]) -> str | None:
    summary = record["policy_finish"]
    if (
        not isinstance(summary, Mapping) or summary.get("schema") != SUMMARY_SCHEMA
        or summary.get("outcome") not in (*OUTCOMES, ABORT)
        or (summary.get("outcome") == "give_up") != (summary.get("category") in CATEGORIES)
    ):
        return f"policy_finish_record_invalid:{index}"
    if summary.get("ended_round") is not True:
        return None  # a terminal-turn finish claims no authority
    if summary["outcome"] == ABORT:
        return None if admitted_abort_round(record) else f"policy_finish_abort_record_invalid:{index}"
    if record.get("official_verdict_checker_divergence") is not None:
        return None  # the verdict guard owns that round's authority
    if (record.get("termination") != POLICY_FINISHED
            or record.get("termination_authority") != "policy_lifecycle"
            or record.get("behavioral_sample_valid") is not True):
        return f"policy_finish_record_invalid:{index}"
    return None


def _finish_record_path(run_root: Path, index: int) -> Path:
    """Where round ``index``'s finish record is: archived at the next reset, else live."""

    archived = run_root / "control" / "rounds" / f"round-{index:04d}" / FINISH_RECORD
    return archived if archived.is_file() else run_root / "control" / FINISH_RECORD


def _bound(run_root: Path, events: Sequence[Mapping[str, Any]], index: int,
           summary: Mapping[str, Any]) -> bool:
    """Whether a round's summary is the one its own finish record and event produce.

    The summary is recomputed from the archived ``policy-finish.json`` and must
    equal the record's; it must also be the one the Supervisor wrote on that
    round's ``termination_authority_decision`` (or the terminal-turn
    ``terminal_policy_lifecycle``) event. A record edited after the run cannot
    relabel a round's outcome, category or abort evidence.
    """

    path = _finish_record_path(run_root, index)
    try:
        expected = (summarize(path.read_bytes(), ended_round=summary.get("ended_round") is True)
                    if path.is_file() else None)
    except ValueError:
        return False
    if expected != dict(summary):
        return False
    return any(
        isinstance(event, Mapping) and event.get("policy_finish") == expected
        and (event.get("type") == "terminal_policy_lifecycle"
             or (event.get("type") == "termination_authority_decision"
                 and event.get("round_index") == index))
        for event in events
    )


def policy_finish_reasons(result: Mapping[str, Any], *, run_root: Path | None = None,
                          events: Sequence[Mapping[str, Any]] | None = None) -> list[str]:
    """Every round whose finish summary disagrees with its authority, validity or evidence.

    With ``run_root`` and ``events`` each summary is also bound to its own
    finish record and decision event (``policy_finish_record_unbound``).
    """

    reasons = []
    for index, record in _records(result):
        if "policy_finish" not in record:
            continue
        reason = _record_reason(index, record)
        if reason is not None:
            reasons.append(reason)
        if run_root is not None and events is not None and not (
                isinstance(record["policy_finish"], Mapping)
                and _bound(run_root, events, index, record["policy_finish"])):
            reasons.append(f"policy_finish_record_unbound:{index}")
    return reasons


def policy_finish_report(result: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    """One line per finished round for evaluation.json, or ``None`` when no round has one."""

    report = []
    for index, record in _records(result):
        summary = record.get("policy_finish")
        if not isinstance(summary, Mapping):
            continue
        ledger = summary.get("untried_candidates")
        report.append({
            "round_index": record.get("round_index", index),
            "outcome": summary.get("outcome"),
            "category": summary.get("category"),
            "ended_round": summary.get("ended_round"),
            "termination_authority": record.get("termination_authority"),
            "behavioral_sample_valid": record.get("behavioral_sample_valid"),
            "reason_names_untried": ledger.get("reason_names_untried") if isinstance(ledger, Mapping) else None,
        })
    return report or None


__all__ = [
    "FINISH_RECORD", "SUMMARY_SCHEMA", "admitted_abort_round", "infrastructure_abort",
    "policy_finish_reasons", "policy_finish_report", "read_policy_finish", "summarize",
]
