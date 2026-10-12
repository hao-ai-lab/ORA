"""One run's record for the run report: identity, attempts, validity, effort, tokens and cost.

`read_run` reads a sealed or in-progress run directory and never writes. Its
attempts are the run's rounds: from ``result.json`` when the run has one,
otherwise from ``rounds/round-*/result.json``; a session that stopped before
its planned rounds keeps the round in flight as one more attempt, built from
the run-level termination. `classify_ending` names each attempt's ending in
one vocabulary across RoboDojo, RoboTwin and LIBERO. `load_corrections` reads
the post-seal correction records an operator writes under
``<namespace>/diagnostics/corrections/``; a correction overrides validity and
learning eligibility, while the run's sealed evaluation is reported unchanged.
The seal and evaluation hashes an entry cites are checked against the run's
files: a mismatch still applies the correction and is flagged
(``binding: mismatch``), so a re-sealed run is never silently re-admitted.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .metrics_values import _integer, _round, summarize
from .run_report_events import api_equivalent_cost, metrics_tokens, scan_agent_events

#: Every ending class, in report order.
ENDINGS = (
    "success_latch", "failure_latch", "native_action_limit", "episode_end", "environment_end",
    "policy_finish", "decision_budget", "model_call_budget", "wall_time_budget", "simulation_step_cap",
    "memory_horizon", "operator_interrupt", "infrastructure", "in_progress", "other",
)
_UPSTREAM = {
    "official_success_latch": "success_latch",
    "official_failure_latch": "failure_latch",
    "native_action_limit": "native_action_limit",
    "official_episode_end": "episode_end",
}
_ENVIRONMENT_REASONS = {
    "official_failure_latch": "failure_latch",
    "official_native_action_limit": "native_action_limit",
    "official_native_action_limit_or_failure": "episode_end",
}
_BUDGETS = {
    "robot_decision_budget": "decision_budget",
    "model_call_budget": "model_call_budget",
    "wall_time_budget": "wall_time_budget",
    "simulation_step_cap": "simulation_step_cap",
    "memory_teacher_soft_horizon": "memory_horizon",
}
_INFRASTRUCTURE = {"trusted_fault", "context_exhausted", "zero_progress_policy_no_action",
                   "libero_control_health_fault"}
_SETTLED_STATES = {"completed", "incomplete", "failed"}
#: Sealed files a correction entry may cite by hash; a cited hash that differs is flagged, not ignored.
_CITED_FILES = (("evaluation.json", "evaluation_file_sha256"), ("seal.json", "seal_file_sha256"))
_REASON_LIMIT = 500


class RunReadError(ValueError):
    """A directory is not a readable run (no readable ``plan.json``)."""


def classify_ending(record: Mapping[str, Any]) -> str:
    """One ending class for a round record (see ``ENDINGS``)."""

    upstream = record.get("upstream_ending_condition")
    if upstream in _UPSTREAM:
        return _UPSTREAM[upstream]
    termination, authority = record.get("termination"), record.get("termination_authority")
    if authority == "infrastructure" or termination in _INFRASTRUCTURE:
        return "infrastructure"
    if authority == "operator" or termination == "operator_interrupt":
        return "operator_interrupt"
    if termination in _BUDGETS:
        return _BUDGETS[termination]
    if termination == "policy_finished_episode":
        return "policy_finish"
    if authority == "environment" or termination == "trusted_runtime_boundary":
        status = record.get("environment_status")
        reason = status.get("reason") if isinstance(status, Mapping) else None
        if record.get("official_success") is True:
            return "success_latch"
        return _ENVIRONMENT_REASONS.get(reason, "environment_end")
    if termination is None and record.get("state") not in _SETTLED_STATES:
        return "in_progress"
    return "other"


def _json(path: Path, *, required: bool = False) -> Any:
    if not path.is_file():
        if required:
            raise RunReadError(f"{path.name}: missing")
        return None
    try:
        return json.loads(path.read_bytes())
    except ValueError as exc:
        if required:
            raise RunReadError(f"{path.name}: unreadable") from exc
        return None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


# -- post-seal corrections ----------------------------------------------------

def correction_paths(run_dirs: Iterable[Path]) -> list[Path]:
    """Every correction record under the namespaces of ``run_dirs`` (``<namespace>/runs/<run>``)."""

    folders = sorted({run.parent.parent / "diagnostics" / "corrections"
                      for run in run_dirs if run.parent.name == "runs"})
    return [path for folder in folders if folder.is_dir() for path in sorted(folder.glob("*.json"))]


def load_corrections(paths: Sequence[Path]) -> tuple[dict[str, dict[str, Any]], list[dict], list[dict]]:
    """Correction entries by run id, the records read, and the files that are not correction records."""

    by_run: dict[str, dict[str, Any]] = {}
    listing: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for path in paths:
        record = _json(path)
        runs = record.get("runs") if isinstance(record, Mapping) else None
        if not isinstance(runs, list):
            errors.append({"path": str(path), "error": "not a post-seal correction record (no runs list)"})
            continue
        identifier = str(record.get("correction_id") or path.stem)
        names = []
        for entry in runs:
            if isinstance(entry, Mapping) and isinstance(entry.get("run_id"), str) \
                    and isinstance(entry.get("corrected"), Mapping):
                names.append(entry["run_id"])
                by_run.setdefault(entry["run_id"], {"correction_id": identifier, "path": str(path),
                                                    "entry": dict(entry)})
        listing.append({"path": str(path), "correction_id": identifier, "runs": names})
    return by_run, listing, errors


def _validity(run_dir: Path, state: Any, evaluation: Any, correction: Mapping | None) -> tuple[Any, ...]:
    """Validity, learning eligibility, the applied correction and warnings."""

    warnings: list[str] = []
    if correction is not None:
        entry = correction["entry"]
        corrected = entry["corrected"]
        cited = {name: entry.get(field) for name, field in _CITED_FILES if isinstance(entry.get(field), str)}
        mismatched = sorted(name for name, digest in cited.items() if _sha256(run_dir / name) != digest)
        warnings.extend(f"correction_sha256_mismatch:{name}" for name in mismatched)
        applied = {"correction_id": correction["correction_id"], "path": correction["path"],
                   "binding": "unbound" if not cited else "mismatch" if mismatched else "matched",
                   "verified_files": sorted(cited), "mismatched_files": mismatched, "corrected": dict(corrected)}
        validity = str(corrected.get("validity") or "corrected_invalid")
        if corrected.get("counts_as_model_attempt") is False and validity == "valid":
            validity = "corrected_invalid"
        return validity, corrected.get("learning_eligible"), applied, warnings
    if isinstance(evaluation, Mapping) and isinstance(evaluation.get("status"), str):
        eligible = _mapping(evaluation.get("learning_eligibility")).get("eligible")
        return evaluation["status"], eligible if isinstance(eligible, bool) else None, None, warnings
    return ("unevaluated" if state in _SETTLED_STATES else "in_progress"), None, None, warnings


# -- attempts -------------------------------------------------------------------

def _recorded_rounds(run_dir: Path, result: Any) -> list[dict[str, Any]]:
    if isinstance(result, Mapping):
        rounds = result.get("rounds")
        if "rounds" not in result:  # a single-run record is its own round
            return [{**dict(result), "round_index": 0}]
        return [dict(item) for item in rounds if isinstance(item, Mapping)] if isinstance(rounds, list) else []
    records = (_json(path) for path in sorted(run_dir.glob("rounds/round-*/result.json")))
    return [dict(item) for item in records if isinstance(item, Mapping)]


def _run_value(field: str, result: Mapping, status: Mapping, rounds: Sequence[Mapping]) -> int | None:
    for source in (result, status):
        value = _integer(source.get(field))
        if value is not None:
            return value
    values = [_integer(item.get(field)) for item in rounds]
    return sum(value for value in values if value is not None) if rounds else None


def _remainder(total: int | None, rounds: Sequence[Mapping], field: str) -> int | None:
    if total is None:
        return None
    return max(total - sum(_integer(item.get(field)) or 0 for item in rounds), 0)


def _unfinished_round(rounds: list[dict], plan: Mapping, result: Mapping, status: Mapping,
                      state: Any, run_wall_s: float | None) -> dict[str, Any] | None:
    """The round in flight when a session stopped (or is still running) before its planned rounds."""

    planned = _integer(_mapping(plan.get("session")).get("rounds")) or 1
    if state == "completed" or len(rounds) >= planned or (rounds and rounds[-1].get("state") != "completed"):
        return None
    walls = [float(item["wall_duration_s"]) for item in rounds
             if isinstance(item.get("wall_duration_s"), (int, float))]
    return {
        "round_index": len(rounds), "state": "running" if state == "running" else state,
        "termination": result.get("termination") or status.get("termination"),
        "termination_authority": result.get("termination_authority") or status.get("termination_authority"),
        "official_success": None, "behavioral_sample_valid": None,
        "decisions": _remainder(_run_value("decisions", result, status, ()), rounds, "decisions"),
        "model_calls": _remainder(_run_value("model_calls", result, status, ()), rounds, "model_calls"),
        "refused_proposals": _remainder(_run_value("refused_proposals", result, status, ()), rounds,
                                        "refused_proposals"),
        "wall_duration_s": max(run_wall_s - sum(walls), 0.0) if run_wall_s is not None else None,
        "synthesized": True,
    }


def _finish_summary(value: Mapping[str, Any], source: str) -> dict[str, Any]:
    reason = value.get("reason")
    return {"outcome": value.get("outcome"), "category": value.get("category"),
            "reason": reason[:_REASON_LIMIT] if isinstance(reason, str) else None, "source": source}


def _policy_finish(record: Mapping, evaluation: Any, calls: Sequence[Mapping], single: bool) -> dict | None:
    """The round's finish outcome: v2 round summary, else evaluation report, else an accepted call."""

    if isinstance(record.get("policy_finish"), Mapping):
        return _finish_summary(record["policy_finish"], "round_record")
    reported = _mapping(evaluation).get("policy_finish")
    for item in reported if isinstance(reported, list) else []:
        if isinstance(item, Mapping) and item.get("round_index") == record.get("round_index"):
            return _finish_summary(item, "evaluation")
    start, end = record.get("started_unix_s"), record.get("finished_unix_s")
    bounded = isinstance(start, (int, float)) and isinstance(end, (int, float))
    accepted = [call for call in calls if call.get("ok") is True and (
        (bounded and call.get("unix_s") is not None and start <= call["unix_s"] <= end) or (not bounded and single))]
    return _finish_summary(accepted[-1], "tool_call") if accepted else None


def _attempt(record: Mapping, validity: str, evaluation: Any, calls: Sequence[Mapping], single: bool) -> dict:
    success = record.get("official_success")
    synthesized = record.get("synthesized") is True
    wall = record.get("wall_duration_s")
    return {
        "round_index": record.get("round_index"),
        "official_success": success if isinstance(success, bool) else None,
        "state": record.get("state"),
        "ending": classify_ending(record),
        "termination": record.get("termination"),
        "termination_authority": record.get("termination_authority"),
        "upstream_ending_condition": record.get("upstream_ending_condition"),
        "behavioral_sample_valid": record.get("behavioral_sample_valid"),
        "valid_attempt": (validity == "valid" and not synthesized and isinstance(success, bool)
                          and record.get("behavioral_sample_valid") is not False),
        "decisions": _integer(record.get("decisions")),
        "model_calls": _integer(record.get("model_calls")),
        "refused_proposals": _integer(record.get("refused_proposals")),
        "wall_minutes": _round(float(wall) / 60.0) if isinstance(wall, (int, float)) else None,
        "policy_finish": _policy_finish(record, evaluation, calls, single),
        "synthesized": synthesized,
    }


# -- run record -----------------------------------------------------------------

def _run_wall_s(result: Mapping, status: Mapping, launch: Mapping) -> float | None:
    if isinstance(result.get("wall_duration_s"), (int, float)):
        return float(result["wall_duration_s"])
    start, end = launch.get("created_unix_s"), status.get("updated_unix_s")
    if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end >= start:
        return float(end - start)
    return None


def _tokens(scan: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    usage = scan["usage"]
    counted = usage["responses"] - usage["responses_without_usage"]
    source, totals = "agent_events_provider_response", usage
    if not counted:
        fallback = metrics_tokens(run_dir)
        source, totals = ("runtime_metrics_workload", fallback) if fallback else ("missing", None)
    fields = ("input", "cached_input", "output", "reasoning_output")
    values = {field: (totals or {}).get(field) for field in fields}
    known = values["input"] is not None and values["cached_input"] is not None
    return {
        "source": source, **values,
        "uncached_input": values["input"] - values["cached_input"] if known else None,
        "cache_share": _round(values["cached_input"] / values["input"]) if known and values["input"] else None,
        "responses": usage["responses"], "responses_without_usage": usage["responses_without_usage"],
    }


def _effort(result: Mapping, status: Mapping, rounds: Sequence[Mapping], wall_s: float | None,
            scan: Mapping[str, Any]) -> dict[str, Any]:
    refused = {name: {"count": sum(codes.values()), "codes": dict(sorted(codes.items()))}
               for name, codes in sorted(scan["refused"].items())}
    return {
        "decisions": _run_value("decisions", result, status, rounds),
        "model_calls": _run_value("model_calls", result, status, rounds),
        "provider_responses": scan["usage"]["responses"],
        "refused_proposals": _run_value("refused_proposals", result, status, rounds),
        "wall_minutes": _round(wall_s / 60.0) if wall_s is not None else None,
        "tool_calls": dict(sorted(scan["tool_calls"].items())),
        "refused_tool_calls": refused,
        "refused_tool_call_total": sum(item["count"] for item in refused.values()),
    }


def _context(scan: Mapping[str, Any]) -> dict[str, Any]:
    samples = {name: list(scan[name]) for name in
               ("request_input_tokens", "observation_text_bytes", "inline_images_per_request")}
    return {**{name: summarize(values) for name, values in samples.items()}, "_samples": samples}


def _identity(run_dir: Path, plan: Mapping, model: Any) -> dict[str, Any]:
    admission, policy = _mapping(plan.get("launch_admission")), _mapping(plan.get("policy"))
    identity = _mapping(plan.get("experiment_identity"))
    parts = [identity.get(key) for key in ("family", "branch", "generation")]
    return {
        "run_id": run_dir.name, "run_dir": str(run_dir),
        "namespace": str(run_dir.parent.parent) if run_dir.parent.name == "runs" else None,
        "cohort_id": admission.get("cohort_id"), "cell_id": admission.get("cell_id"),
        "task": plan.get("task"), "backend": _mapping(plan.get("runtime")).get("kind"),
        "model": model, "provider_profile": policy.get("provider_profile"),
        "reasoning_effort": policy.get("reasoning_effort"),
        "experiment": "/".join(str(item) for item in parts) if all(parts) else None,
    }


def read_run(run_dir: Path, pricing: Mapping[str, Any], corrections: Mapping[str, Mapping]) -> dict[str, Any]:
    """The report record of one run directory; raises ``RunReadError`` without a readable plan."""

    plan = _json(run_dir / "plan.json", required=True)
    if not isinstance(plan, Mapping):
        raise RunReadError("plan.json: not an object")
    result_raw = _json(run_dir / "result.json")
    result, status = _mapping(result_raw), _mapping(_json(run_dir / "status.json"))
    evaluation = _json(run_dir / "evaluation.json")
    state = result.get("state") or status.get("state")
    scan = scan_agent_events(run_dir)
    model = scan["model"] or _mapping(plan.get("policy")).get("model")
    validity, eligible, correction, warnings = _validity(run_dir, state, evaluation, corrections.get(run_dir.name))
    rounds = _recorded_rounds(run_dir, result_raw)
    wall_s = _run_wall_s(result, status, _mapping(_json(run_dir / "launch.json")))
    unfinished = _unfinished_round(rounds, plan, result, status, state, wall_s)
    records = rounds + ([unfinished] if unfinished else [])
    if scan["unreadable_lines"]:
        warnings.append(f"agent_events_unreadable_lines:{scan['unreadable_lines']}")
    tokens = _tokens(scan, run_dir)
    reasons = _mapping(evaluation).get("reasons")
    return {
        **_identity(run_dir, plan, model),
        "state": state, "sealed": (run_dir / "seal.json").is_file(),
        "termination": result.get("termination") or status.get("termination"),
        "termination_authority": result.get("termination_authority") or status.get("termination_authority"),
        "planned_rounds": _integer(_mapping(plan.get("session")).get("rounds")) or 1,
        "evaluation": None if not isinstance(evaluation, Mapping) else {
            "status": evaluation.get("status"),
            "learning_eligible": _mapping(evaluation.get("learning_eligibility")).get("eligible"),
            "reason_count": len(reasons) if isinstance(reasons, list) else 0,
            "reasons": list(reasons[:5]) if isinstance(reasons, list) else [],
        },
        "correction": correction, "validity": validity, "learning_eligible": eligible,
        "attempts": [_attempt(item, validity, evaluation, scan["finish_calls"], len(records) == 1)
                     for item in records],
        "effort": _effort(result, status, rounds, wall_s, scan),
        "tokens": tokens, "cost": api_equivalent_cost(tokens, model, pricing),
        "context": _context(scan), "warnings": warnings,
    }


__all__ = ["ENDINGS", "RunReadError", "classify_ending", "correction_paths", "load_corrections", "read_run"]
