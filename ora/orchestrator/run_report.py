"""Standard stop-reason and cost report for Robot Harness runs; read-only.

For sealed or in-progress run directories it reports, per attempt (round)
and aggregated per cohort, per (model, task) and for the whole selection:

- ending: official success, validity (evaluation status, learning
  eligibility and any post-seal correction under
  ``<namespace>/diagnostics/corrections/``), termination and its authority,
  the ending class (``run_report_run.ENDINGS``) and the policy finish outcome
  and give_up category when the run records one;
- effort: decisions, model calls, wall minutes, the tool-call mix and refused
  tool calls by tool and error code;
- tokens: input, cached input, output and reasoning tokens from the
  provider responses in ``agent-events.jsonl``, the cache share and an
  API-equivalent cost from ``run_report_pricing.yaml`` (unpriced models null);
- context: request input tokens and current-observation text bytes.

It never writes inside a run, never starts a simulator or a provider and
refuses an output path inside a source run.

    python -m ora.orchestrator.run_report RUN [RUN ...] [--json FILE] [--markdown FILE]
    python -m ora.orchestrator.run_report --namespace NS --cohort ID [--cohort ID ...]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

from .run_report_aggregate import group_reports
from .run_report_events import load_pricing
from .run_report_markdown import render_markdown
from .run_report_run import ENDINGS, RunReadError, correction_paths, load_corrections, read_run

REPORT_SCHEMA = "ora.run-report.v1"
DEFINITIONS = {
    "attempt": "one round of a run; a session that stopped before its planned rounds keeps the round in flight",
    "valid_attempt": "run validity is valid (evaluation status after any post-seal correction), the round has an "
                     "official Boolean and its behavioral sample is not marked invalid",
    "success": "official_success true on a valid attempt",
    "ending_classes": list(ENDINGS),
    "cost": "API-equivalent USD: (input - cached_input) x input price + cached_input x cached price + output x "
            "output price, per million tokens; output includes reasoning; not what the run paid",
    "cache_share": "cached_input / input over provider-reported usage",
}


class RunReportError(ValueError):
    """The requested selection cannot be reported."""


def _plan_cohort(run: Path) -> str | None:
    try:
        plan = json.loads((run / "plan.json").read_bytes())
    except (OSError, ValueError):
        return None
    admission = plan.get("launch_admission") if isinstance(plan, Mapping) else None
    cohort = admission.get("cohort_id") if isinstance(admission, Mapping) else None
    return cohort if isinstance(cohort, str) else None


def discover_runs(namespace: Path, cohorts: Sequence[str] = ()) -> list[Path]:
    """Run directories under ``<namespace>/runs`` with a plan, optionally limited to ``cohorts``."""

    root = Path(namespace) / "runs"
    if not root.is_dir():
        raise RunReportError(f"{namespace}: no runs directory")
    found: list[tuple[str, str, Path]] = []
    for run in sorted(path for path in root.iterdir() if path.is_dir() and (path / "plan.json").is_file()):
        cohort = _plan_cohort(run)
        if not cohorts or cohort in cohorts:
            found.append((str(cohort), run.name, run))
    missing = sorted(set(cohorts) - {cohort for cohort, _name, _run in found})
    if missing:
        raise RunReportError(f"{namespace}: no runs for cohort(s) {', '.join(missing)}")
    return [run for _cohort, _name, run in sorted(found)]


def _read(run: Path, pricing: Mapping[str, Any], corrections: Mapping[str, Mapping]) -> tuple[str, Any]:
    try:
        return "run", read_run(run, pricing, corrections)
    except (RunReadError, OSError) as exc:
        return "error", {"run_dir": str(run), "error": str(exc)}


def _public(value: Any) -> Any:
    """The report without private sample lists (keys starting with ``_``)."""

    if isinstance(value, Mapping):
        return {key: _public(item) for key, item in value.items() if not str(key).startswith("_")}
    if isinstance(value, list):
        return [_public(item) for item in value]
    return value


def _binding_warnings(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One warning per sealed file whose bytes differ from the hash its correction cites."""

    return [{"path": record["correction"]["path"], "run_id": record["run_id"],
             "error": f"cited sha256 differs from {name}"}
            for record in records if record["correction"] is not None
            for name in record["correction"]["mismatched_files"]]


def _pricing_summary(pricing: Mapping[str, Any]) -> dict[str, Any]:
    return {"label": pricing.get("label", "api_equivalent"), "currency": pricing.get("currency", "USD"),
            "unit": pricing.get("unit", "per_million_tokens"), "path": pricing.get("path"),
            "sha256": pricing.get("sha256"), "priced_models": sorted(pricing.get("models") or {})}


def build_report(
    run_dirs: Sequence[Path], *, pricing: Mapping[str, Any] | None = None,
    corrections: Sequence[Path] | None = None, jobs: int = 1,
) -> dict[str, Any]:
    """The report of ``run_dirs``.

    ``corrections`` None discovers correction records from each run's
    namespace; a list names record files or folders. Unreadable runs are
    ``errors``; a correction file that is not a correction record, and a
    sealed file whose bytes differ from the hash a correction cites, are
    ``warnings`` entries.
    """

    runs = [Path(path).absolute() for path in run_dirs]
    table = load_pricing() if pricing is None else dict(pricing)
    paths = correction_paths(runs) if corrections is None else [
        item for path in map(Path, corrections)
        for item in (sorted(path.glob("*.json")) if path.is_dir() else [path])]
    by_run, listing, correction_warnings = load_corrections(paths)
    reader = partial(_read, pricing=table, corrections=by_run)
    if jobs > 1 and len(runs) > 1:
        with ProcessPoolExecutor(max_workers=min(jobs, len(runs))) as pool:
            outcomes = list(pool.map(reader, runs))
    else:
        outcomes = [reader(run) for run in runs]
    records = [value for kind, value in outcomes if kind == "run"]
    applied = {record["run_id"] for record in records if record["correction"] is not None}
    return _public({
        "schema": REPORT_SCHEMA, "definitions": DEFINITIONS,
        "inputs": {"run_dirs": [str(run) for run in runs]},
        "pricing": _pricing_summary(table),
        "corrections": [{**item, "runs": [name for name in item["runs"] if name in applied]}
                        for item in listing if any(name in applied for name in item["runs"])],
        "runs": records, **group_reports(records),
        "errors": [value for kind, value in outcomes if kind == "error"],
        "warnings": correction_warnings + _binding_warnings(records),
    })


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ora.orchestrator.run_report",
        description="Read-only stop-reason, validity, effort, token and API-equivalent cost report.")
    parser.add_argument("runs", nargs="*", type=Path, help="run directories (or use --namespace)")
    parser.add_argument("--namespace", type=Path, help="experiment output namespace holding runs/")
    parser.add_argument("--cohort", action="append", default=[], help="launch-admission cohort id (repeatable)")
    parser.add_argument("--json", type=Path, help="write the JSON report here (outside every source run)")
    parser.add_argument("--markdown", type=Path, help="write the Markdown tables here (outside every source run)")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown",
                        help="stdout format when no output file is given (default markdown)")
    parser.add_argument("--pricing", type=Path, help="price table (default run_report_pricing.yaml)")
    parser.add_argument("--corrections", type=Path, action="append",
                        help="correction record or folder of them (repeatable); "
                             "default: <namespace>/diagnostics/corrections/*.json")
    parser.add_argument("--no-corrections", action="store_true", help="apply no post-seal correction")
    parser.add_argument("--jobs", type=int, default=0, help="parallel readers (default min(8, CPUs, runs))")
    return parser


def _selection(parser: argparse.ArgumentParser, args: argparse.Namespace) -> list[Path]:
    if bool(args.runs) == (args.namespace is not None):
        parser.error("give run directories or --namespace, not both")
    if args.cohort and args.namespace is None:
        parser.error("--cohort needs --namespace")
    if args.no_corrections and args.corrections:
        parser.error("--corrections and --no-corrections exclude each other")
    if args.namespace is None:
        return list(args.runs)
    try:
        return discover_runs(args.namespace, args.cohort)
    except RunReportError as exc:
        parser.error(str(exc))


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    runs = _selection(parser, args)
    outputs = [path.absolute() for path in (args.json, args.markdown) if path is not None]
    sources = [run.absolute() for run in runs]
    if any(output.is_relative_to(source) for output in outputs for source in sources):
        parser.error("--json and --markdown must be outside every source run")
    jobs = args.jobs if args.jobs > 0 else min(8, os.cpu_count() or 1, max(len(runs), 1))
    corrections = [] if args.no_corrections else args.corrections
    report = build_report(runs, pricing=load_pricing(args.pricing), corrections=corrections, jobs=jobs)
    text = {"json": json.dumps(report, indent=2, sort_keys=True) + "\n", "markdown": render_markdown(report)}
    for path, kind in ((args.json, "json"), (args.markdown, "markdown")):
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text[kind], encoding="utf-8")
    if outputs:
        print(json.dumps({"json": str(args.json) if args.json else None,
                          "markdown": str(args.markdown) if args.markdown else None,
                          "runs": len(report["runs"]), "errors": len(report["errors"])}))
    else:
        sys.stdout.write(text[args.format])
    return 1 if report["errors"] else 0


__all__ = ["DEFINITIONS", "REPORT_SCHEMA", "RunReportError", "build_report", "discover_runs", "main",
           "render_markdown"]


if __name__ == "__main__":
    sys.exit(main())
