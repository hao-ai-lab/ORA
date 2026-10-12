"""Cohort, task and selection totals of run report records.

`aggregate` sums one group of run records: attempts and the valid ones,
official successes among valid attempts, ending classes, policy finish
outcomes, effort, tool calls and refusals, provider tokens, API-equivalent
cost and pooled context sizes. `group_reports` builds the per-cohort and
per-(model, task) groups and the selection total.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Mapping, Sequence

from .metrics_values import _round, summarize
from .run_report_events import COST_LABEL

_TOKEN_FIELDS = ("input", "cached_input", "uncached_input", "output", "reasoning_output")
_EFFORT_FIELDS = ("decisions", "model_calls", "provider_responses", "refused_proposals")
_CONTEXT_FIELDS = ("request_input_tokens", "observation_text_bytes", "inline_images_per_request")


def _sorted_counts(counter: Counter) -> dict[str, int]:
    return dict(sorted(counter.items()))


def _attempt_counts(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    endings, valid_endings, excluded = Counter(), Counter(), Counter()
    finishes, categories = Counter(), Counter()
    attempts = valid = successes = 0
    for run in runs:
        for attempt in run["attempts"]:
            attempts += 1
            endings[attempt["ending"]] += 1
            finish = attempt.get("policy_finish")
            if isinstance(finish, Mapping) and finish.get("outcome"):
                finishes[str(finish["outcome"])] += 1
                if finish["outcome"] == "give_up" and finish.get("category"):
                    categories[str(finish["category"])] += 1
            if attempt["valid_attempt"]:
                valid += 1
                valid_endings[attempt["ending"]] += 1
                successes += attempt["official_success"] is True
            else:
                excluded["invalid_sample" if run["validity"] == "valid" else str(run["validity"])] += 1
    return {
        "attempts": attempts, "valid_attempts": valid, "successes": successes,
        "success_rate": _round(successes / valid) if valid else None,
        "excluded_attempts": _sorted_counts(excluded), "endings": _sorted_counts(endings),
        "valid_endings": _sorted_counts(valid_endings), "policy_finish": _sorted_counts(finishes),
        "give_up_categories": _sorted_counts(categories),
    }


def _effort(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    effort: dict[str, Any] = {field: sum(run["effort"][field] or 0 for run in runs) for field in _EFFORT_FIELDS}
    minutes = [run["effort"]["wall_minutes"] for run in runs if run["effort"]["wall_minutes"] is not None]
    tools, refused, codes = Counter(), Counter(), {}
    for run in runs:
        tools.update(run["effort"]["tool_calls"])
        for name, item in run["effort"]["refused_tool_calls"].items():
            refused[name] += item["count"]
            codes.setdefault(name, Counter()).update(item["codes"])
    effort.update(
        wall_minutes=_round(sum(minutes)) if minutes else None,
        tool_calls=dict(sorted(tools.items())),
        refused_tool_calls={name: {"count": refused[name], "codes": _sorted_counts(codes[name])}
                            for name in sorted(refused)},
        refused_tool_call_total=sum(refused.values()),
    )
    return effort


def _tokens(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counted = [run["tokens"] for run in runs if run["tokens"]["input"] is not None]
    totals: dict[str, Any] = {field: sum(item[field] or 0 for item in counted) if counted else None
                              for field in _TOKEN_FIELDS}
    totals["cache_share"] = (_round(totals["cached_input"] / totals["input"])
                             if counted and totals["input"] else None)
    totals["runs_without_tokens"] = len(runs) - len(counted)
    return totals


def _cost(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    priced = [run["cost"]["usd"] for run in runs if run["cost"]["usd"] is not None]
    unpriced = sorted({str(run["cost"]["model"]) for run in runs if run["cost"]["usd"] is None})
    return {"label": COST_LABEL, "usd": _round(sum(priced)) if priced else None, "priced_runs": len(priced),
            "unpriced_runs": len(runs) - len(priced), "unpriced_models": unpriced}


def aggregate(runs: Sequence[Mapping[str, Any]], key: Mapping[str, Any]) -> dict[str, Any]:
    """One group's totals; ``key`` names the group."""

    pooled = {name: [value for run in runs for value in run["context"]["_samples"][name]]
              for name in _CONTEXT_FIELDS}
    return {
        "key": dict(key), "runs": len(runs),
        "models": sorted({str(run["model"]) for run in runs}),
        "cohorts": sorted({str(run["cohort_id"]) for run in runs}),
        "validity": _sorted_counts(Counter(str(run["validity"]) for run in runs)),
        "learning_eligible_runs": sum(run["learning_eligible"] is True for run in runs),
        **_attempt_counts(runs),
        "effort": _effort(runs), "tokens": _tokens(runs), "cost": _cost(runs),
        "context": {name: summarize(values) for name, values in pooled.items()},
    }


def _groups(runs: Sequence[Mapping[str, Any]], key: Callable[[Mapping], tuple]) -> list[tuple[tuple, list]]:
    grouped: dict[tuple, list] = {}
    for run in runs:
        grouped.setdefault(key(run), []).append(run)
    return sorted(grouped.items(), key=lambda item: tuple(str(part) for part in item[0]))


def group_reports(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-cohort and per-(model, task) aggregates and the selection total."""

    return {
        "cohorts": [aggregate(members, {"cohort_id": cohort})
                    for (cohort,), members in _groups(runs, lambda run: (run["cohort_id"],))],
        "tasks": [aggregate(members, {"model": model, "task": task})
                  for (model, task), members in _groups(runs, lambda run: (run["model"], run["task"]))],
        "total": aggregate(runs, {"scope": "selection"}),
    }


__all__ = ["aggregate", "group_reports"]
