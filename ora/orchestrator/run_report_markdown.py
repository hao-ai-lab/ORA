"""Compact Markdown tables of a run report.

`render_markdown` turns the JSON report into cohort, task, run, context and
tool tables with one notes section stating how validity and the
API-equivalent cost are defined. It reads only the report.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

_TOOL_ROWS_PER_COHORT = 12


def _cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines.extend("| " + " | ".join(_cell(item) for item in row) + " |" for row in rows)
    return lines + [""]


def _percent(value: float | None) -> str | None:
    return None if value is None else f"{100.0 * value:.1f}%"


def _millions(value: int | None) -> str | None:
    return None if value is None else f"{value / 1e6:.2f}"


def _thousands(value: int | None) -> str | None:
    return None if value is None else f"{value / 1e3:.1f}"


def _usd(value: float | None) -> str | None:
    return None if value is None else f"{value:,.2f}"


def _number(value: float | None, digits: int = 0) -> str | None:
    return None if value is None else f"{value:,.{digits}f}"


def _counts(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{name} {count}" for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _short(run_id: str) -> str:
    return run_id.split("--")[0][-12:] if "--" in run_id else run_id


def _group_row(group: Mapping[str, Any], label: Sequence[Any]) -> list[Any]:
    tokens, effort = group["tokens"], group["effort"]
    hours = effort["wall_minutes"] / 60.0 if effort["wall_minutes"] is not None else None
    return [
        *label, group["runs"], group["attempts"], group["valid_attempts"], group["successes"],
        _percent(group["success_rate"]), _counts(group["endings"]), effort["decisions"], effort["model_calls"],
        _number(hours, 1), effort["refused_tool_call_total"], _millions(tokens["input"]),
        _percent(tokens["cache_share"]), _thousands(tokens["output"]), _usd(group["cost"]["usd"]),
    ]


_GROUP_HEADER = ["runs", "attempts", "valid", "success", "success rate", "endings (all attempts)", "decisions",
                 "model calls", "wall h", "refused tools", "input Mtok", "cached", "output ktok",
                 "API-equivalent USD"]


def _validity(run: Mapping[str, Any]) -> str:
    correction = run.get("correction")
    if not correction:
        return str(run["validity"])
    flag = ", hash mismatch" if correction.get("binding") == "mismatch" else ""
    return f"{run['validity']} ({correction['correction_id']}{flag})"


def _run_rows(runs: Sequence[Mapping[str, Any]]) -> list[list[Any]]:
    rows = []
    for run in runs:
        for attempt in run["attempts"]:
            finish = attempt.get("policy_finish") or {}
            outcome = "/".join(str(part) for part in (finish.get("outcome"), finish.get("category")) if part)
            context = run["context"]
            rows.append([
                _short(run["run_id"]) + (f" r{attempt['round_index']}" if len(run["attempts"]) > 1 else ""),
                run["cohort_id"], run["task"], _validity(run), attempt["official_success"], attempt["ending"],
                outcome or None, attempt["decisions"], attempt["model_calls"], _number(attempt["wall_minutes"], 1),
                run["effort"]["refused_tool_call_total"], _millions(run["tokens"]["input"]),
                _percent(run["tokens"]["cache_share"]), _usd(run["cost"]["usd"]),
                _number(context["request_input_tokens"]["p50"]), _number(context["request_input_tokens"]["max"]),
                _number(context["observation_text_bytes"]["p50"]),
            ])
    return rows


def _tool_rows(cohorts: Sequence[Mapping[str, Any]]) -> list[list[Any]]:
    rows = []
    for group in cohorts:
        effort = group["effort"]
        refused = effort["refused_tool_calls"]
        ranked = sorted(effort["tool_calls"].items(), key=lambda item: (-item[1], item[0]))
        for name, count in ranked[:_TOOL_ROWS_PER_COHORT]:
            item = refused.get(name, {"count": 0, "codes": {}})
            rows.append([group["key"]["cohort_id"], name, count, item["count"], _counts(item["codes"]) or None])
    return rows


def _notes(report: Mapping[str, Any]) -> list[str]:
    pricing = report["pricing"]
    unpriced = report["total"]["cost"]["unpriced_models"]
    lines = [
        "Notes:",
        "",
        "- An attempt is one round. It is valid only when its run's evaluation status is `valid` after any "
        "post-seal correction and the round has an official Boolean; success counts valid attempts only.",
        "- API-equivalent USD: provider-reported tokens priced at the per-model API list prices in "
        f"`{pricing.get('path') or 'the supplied table'}` (sha256 {str(pricing.get('sha256'))[:12]}). "
        "It is not what the runs paid. Unpriced models are shown as `-`"
        + (f": {', '.join(unpriced)}." if unpriced else "."),
    ]
    for item in report["corrections"]:
        lines.append(f"- Correction `{item['correction_id']}` applied to {len(item['runs'])} run(s), shown as "
                     f"excluded attempts and not learning-eligible: {', '.join(_short(name) for name in item['runs'])}"
                     f" (`{item['path']}`).")
    for item in report["errors"]:
        lines.append(f"- Not read: `{item['run_dir']}`: {item['error']}.")
    for item in report["warnings"]:
        run = f" for {_short(item['run_id'])}" if item.get("run_id") else ""
        lines.append(f"- Warning `{item['path']}`{run}: {item['error']}.")
    return lines + [""]


def render_markdown(report: Mapping[str, Any]) -> str:
    """The report as Markdown tables."""

    cohorts, tasks, total = report["cohorts"], report["tasks"], report["total"]
    lines = [f"# Run report: {total['runs']} runs, {total['valid_attempts']} valid of {total['attempts']} "
             f"attempts, {total['successes']} successes", ""]
    lines += ["## Cohorts", ""]
    lines += _table(["cohort", "model", *_GROUP_HEADER], [
        *(_group_row(group, [group["key"]["cohort_id"], ", ".join(group["models"])]) for group in cohorts),
        _group_row(total, ["**total**", ", ".join(total["models"])]),
    ])
    lines += ["## Tasks", ""]
    lines += _table(["model", "task", *_GROUP_HEADER],
                    (_group_row(group, [group["key"]["model"], group["key"]["task"]]) for group in tasks))
    lines += ["## Attempts", ""]
    lines += _table(["run", "cohort", "task", "validity", "success", "ending", "finish", "decisions", "model calls",
                     "min", "refused tools", "input Mtok", "cached", "API-equivalent USD", "request tok p50",
                     "request tok max", "observation text B p50"], _run_rows(report["runs"]))
    lines += ["## Context", ""]
    lines += _table(["cohort", "requests", "request tok p50", "request tok max", "observation text B p50",
                     "observation text B max", "inline images max"], (
        [group["key"]["cohort_id"], group["context"]["request_input_tokens"]["count"],
         _number(group["context"]["request_input_tokens"]["p50"]),
         _number(group["context"]["request_input_tokens"]["max"]),
         _number(group["context"]["observation_text_bytes"]["p50"]),
         _number(group["context"]["observation_text_bytes"]["max"]),
         _number(group["context"]["inline_images_per_request"]["max"])] for group in cohorts))
    lines += ["## Tool calls", ""]
    lines += _table(["cohort", "tool", "calls", "refused", "refusal codes"], _tool_rows(cohorts))
    return "\n".join(lines + _notes(report))


__all__ = ["render_markdown"]
