"""Versioned, experiment-declared Viewer projections over immutable run evidence."""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml


SCHEMA = "ora.viewer-experiment.v1"
LAYOUTS = frozenset({
    "single_episode", "matrix", "persistent_rounds", "diagnostic_set",
})
LAYERS = frozenset({
    "infra_control", "tooling", "harness_feedback", "model_capability", "other",
})
EPISTEMIC_STATUSES = frozenset({
    "observation", "inference", "hypothesis", "negative_finding",
})
PRIORITIES = frozenset({"p0", "p1", "p2", "diagnostic"})
_ID = re.compile(r"[a-z][a-z0-9_]*\Z")
_TOP_LEVEL = frozenset({
    "schema", "experiment", "layout", "title", "subtitle", "claim_boundary",
    "cells", "findings", "exclusions",
})
_CELL_FIELDS = frozenset({
    "id", "accepted_evidence", "label", "role", "summary", "row", "column",
})
_FINDING_FIELDS = frozenset({
    "id", "layer", "epistemic_status", "priority", "title", "summary",
    "evidence_cells",
})
_EXCLUSION_FIELDS = frozenset({"evidence", "reason", "summary"})





@dataclass(frozen=True)
class ExperimentViewAudit:
    experiment: str
    path: Path
    layout: str
    structural_issues: tuple[str, ...]
    evidence_issues: tuple[str, ...]
    unavailable_run_ids: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.structural_issues and not self.evidence_issues


@dataclass(frozen=True)
class ExperimentViewProjection:
    present: bool
    html: str
    audit: ExperimentViewAudit | None


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return value if isinstance(value, dict) else {}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def load_experiment_view(definition_path: Path | None) -> tuple[Path | None, dict[str, Any]]:
    if definition_path is None:
        return None, {}
    path = definition_path.with_name("viewer.yaml")
    if not path.is_file():
        return None, {}
    return path, _load_yaml(path)


def _plain_text(value: object, field: str, issues: list[str], *, required: bool = True) -> str:
    if not isinstance(value, str) or (required and not value.strip()):
        issues.append(f"{field}: expected non-empty text")
        return ""
    text = value.strip()
    if any(ord(character) < 32 and character not in "\n\t" for character in text):
        issues.append(f"{field}: control characters are forbidden")
    if "<" in text or ">" in text:
        issues.append(f"{field}: HTML/CSS markup is forbidden")
    return text


def _stable_id(value: object, field: str, issues: list[str]) -> str:
    text = _plain_text(value, field, issues)
    if text and _ID.fullmatch(text) is None:
        issues.append(f"{field}: expected [a-z][a-z0-9_]*")
    return text


def validate_experiment_view(
    manifest: Mapping[str, Any], definition: Mapping[str, Any],
) -> tuple[str, ...]:
    issues: list[str] = []
    unknown = sorted(set(manifest) - _TOP_LEVEL)
    if unknown:
        issues.append(f"unknown top-level fields: {', '.join(unknown)}")
    if manifest.get("schema") != SCHEMA:
        issues.append(f"schema: expected {SCHEMA}")
    experiment = _plain_text(manifest.get("experiment"), "experiment", issues)
    if experiment and experiment != str(definition.get("experiment") or ""):
        issues.append("experiment: does not match experiment.yaml")
    layout = str(manifest.get("layout") or "")
    if layout not in LAYOUTS:
        issues.append(f"layout: expected one of {', '.join(sorted(LAYOUTS))}")
    _plain_text(manifest.get("title"), "title", issues)
    _plain_text(manifest.get("subtitle", ""), "subtitle", issues, required=False)
    _plain_text(manifest.get("claim_boundary"), "claim_boundary", issues)

    accepted = definition.get("accepted_evidence")
    if not isinstance(accepted, Mapping):
        issues.append("experiment.yaml: accepted_evidence must be an object")
        accepted = {}
    cells = manifest.get("cells")
    if not isinstance(cells, list) or not cells:
        issues.append("cells: expected a non-empty list")
        cells = []
    cell_ids: set[str] = set()
    evidence_keys: set[str] = set()
    matrix_pairs: set[tuple[str, str]] = set()
    for index, raw in enumerate(cells):
        prefix = f"cells[{index}]"
        if not isinstance(raw, Mapping):
            issues.append(f"{prefix}: expected object")
            continue
        unknown = sorted(set(raw) - _CELL_FIELDS)
        if unknown:
            issues.append(f"{prefix}: unknown fields: {', '.join(unknown)}")
        cell_id = _stable_id(raw.get("id"), f"{prefix}.id", issues)
        if cell_id in cell_ids:
            issues.append(f"{prefix}.id: duplicate {cell_id}")
        cell_ids.add(cell_id)
        evidence_key = _stable_id(
            raw.get("accepted_evidence"), f"{prefix}.accepted_evidence", issues,
        )
        if evidence_key in evidence_keys:
            issues.append(f"{prefix}.accepted_evidence: duplicate {evidence_key}")
        evidence_keys.add(evidence_key)
        evidence = accepted.get(evidence_key) if isinstance(accepted, Mapping) else None
        if not isinstance(evidence, Mapping):
            issues.append(f"{prefix}.accepted_evidence: unknown key {evidence_key}")
        else:
            if not isinstance(evidence.get("run_id"), str) or not evidence.get("run_id"):
                issues.append(f"accepted_evidence.{evidence_key}: missing run_id")
            if not isinstance(evidence.get("source_commit"), str) or not evidence.get("source_commit"):
                issues.append(f"accepted_evidence.{evidence_key}: missing source_commit")
        for field in ("label", "role"):
            _plain_text(raw.get(field), f"{prefix}.{field}", issues)
        _plain_text(raw.get("summary", ""), f"{prefix}.summary", issues, required=False)
        row, column = raw.get("row"), raw.get("column")
        if layout == "matrix":
            row_text = _stable_id(row, f"{prefix}.row", issues)
            column_text = _stable_id(column, f"{prefix}.column", issues)
            pair = (row_text, column_text)
            if pair in matrix_pairs:
                issues.append(f"{prefix}: duplicate matrix position {row_text}/{column_text}")
            matrix_pairs.add(pair)
        elif row is not None or column is not None:
            issues.append(f"{prefix}: row/column are allowed only for matrix layout")
    if layout in {"single_episode", "persistent_rounds"} and len(cells) != 1:
        issues.append(f"layout {layout}: expected exactly one cell")
    if layout == "diagnostic_set" and len(cells) < 2:
        issues.append("layout diagnostic_set: expected at least two cells")

    findings = manifest.get("findings", [])
    if not isinstance(findings, list):
        issues.append("findings: expected list")
        findings = []
    finding_ids: set[str] = set()
    for index, raw in enumerate(findings):
        prefix = f"findings[{index}]"
        if not isinstance(raw, Mapping):
            issues.append(f"{prefix}: expected object")
            continue
        unknown = sorted(set(raw) - _FINDING_FIELDS)
        if unknown:
            issues.append(f"{prefix}: unknown fields: {', '.join(unknown)}")
        finding_id = _stable_id(raw.get("id"), f"{prefix}.id", issues)
        if finding_id in finding_ids:
            issues.append(f"{prefix}.id: duplicate {finding_id}")
        finding_ids.add(finding_id)
        if raw.get("layer") not in LAYERS:
            issues.append(f"{prefix}.layer: invalid value")
        if raw.get("epistemic_status") not in EPISTEMIC_STATUSES:
            issues.append(f"{prefix}.epistemic_status: invalid value")
        if raw.get("priority") not in PRIORITIES:
            issues.append(f"{prefix}.priority: invalid value")
        _plain_text(raw.get("title"), f"{prefix}.title", issues)
        _plain_text(raw.get("summary"), f"{prefix}.summary", issues)
        refs = raw.get("evidence_cells")
        if not isinstance(refs, list) or not refs:
            issues.append(f"{prefix}.evidence_cells: expected non-empty list")
        else:
            for ref in refs:
                if ref not in cell_ids:
                    issues.append(f"{prefix}.evidence_cells: unknown cell {ref}")

    exclusions = manifest.get("exclusions", [])
    if not isinstance(exclusions, list):
        issues.append("exclusions: expected list")
        exclusions = []
    excluded_keys: set[str] = set()
    for index, raw in enumerate(exclusions):
        prefix = f"exclusions[{index}]"
        if not isinstance(raw, Mapping):
            issues.append(f"{prefix}: expected object")
            continue
        unknown = sorted(set(raw) - _EXCLUSION_FIELDS)
        if unknown:
            issues.append(f"{prefix}: unknown fields: {', '.join(unknown)}")
        evidence_key = _stable_id(raw.get("evidence"), f"{prefix}.evidence", issues)
        if evidence_key in excluded_keys:
            issues.append(f"{prefix}.evidence: duplicate {evidence_key}")
        excluded_keys.add(evidence_key)
        if evidence_key in evidence_keys:
            issues.append(f"{prefix}.evidence: accepted/excluded conflict {evidence_key}")
        if not isinstance(accepted.get(evidence_key), Mapping):
            issues.append(f"{prefix}.evidence: unknown key {evidence_key}")
        _stable_id(raw.get("reason"), f"{prefix}.reason", issues)
        _plain_text(raw.get("summary", ""), f"{prefix}.summary", issues, required=False)
    return tuple(issues)


def audit_experiment_view(
    path: Path, manifest: Mapping[str, Any], definition: Mapping[str, Any],
    runs: Sequence[Any],
) -> ExperimentViewAudit:
    structural = validate_experiment_view(manifest, definition)
    evidence_issues: list[str] = []
    unavailable: list[str] = []
    runs_by_id = {str(getattr(run, "run_id", "")): run for run in runs}
    accepted = definition.get("accepted_evidence")
    accepted = accepted if isinstance(accepted, Mapping) else {}
    if not structural:
        for cell in manifest.get("cells", []):
            evidence = accepted[str(cell["accepted_evidence"])]
            run_id = str(evidence["run_id"])
            run = runs_by_id.get(run_id)
            if run is None:
                unavailable.append(run_id)
                continue
            expected_source = str(evidence["source_commit"])
            actual_source = str(getattr(run, "source_commit", ""))
            if actual_source != expected_source:
                evidence_issues.append(
                    f"{cell['id']}: source mismatch {actual_source or 'missing'} != {expected_source}"
                )
            if manifest.get("layout") == "persistent_rounds":
                result = _load_json(Path(run.root) / "result.json")
                if result.get("official_success") is not None:
                    evidence_issues.append(
                        f"{cell['id']}: persistent session aggregate official_success must be null"
                    )
                if not isinstance(result.get("rounds"), list) or not result.get("rounds"):
                    evidence_issues.append(f"{cell['id']}: persistent session rounds missing")
    return ExperimentViewAudit(
        experiment=str(manifest.get("experiment") or definition.get("experiment") or "unknown"),
        path=path,
        layout=str(manifest.get("layout") or "unknown"),
        structural_issues=structural,
        evidence_issues=tuple(evidence_issues),
        unavailable_run_ids=tuple(unavailable),
    )


def _number(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _compact(value: object) -> str:
    number = _number(value)
    if number is None:
        return "—"
    absolute = abs(number)
    if absolute >= 1_000_000:
        return f"{number / 1_000_000:.2f}M"
    if absolute >= 1_000:
        return f"{number / 1_000:.1f}k"
    return f"{int(number):,}" if number.is_integer() else f"{number:.1f}"


def _duration(value: object) -> str:
    number = _number(value)
    if number is None:
        return "—"
    hours, remainder = divmod(max(0, int(round(number))), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}"


def _verdict(value: object) -> tuple[str, str]:
    if value is True:
        return "success", "success"
    if value is False:
        return "failure", "failure"
    return "unscored", "unscored"


def _render_rounds(result: Mapping[str, Any], run_id: str) -> str:
    rounds = result.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        return "<p class='xview-invalid'>Persistent-round evidence is unavailable.</p>"
    rows = []
    for index, item in enumerate(rounds):
        value = item if isinstance(item, Mapping) else {}
        label, css = _verdict(value.get("official_success"))
        step_range = value.get("policy_state_steps")
        step_range = step_range if isinstance(step_range, Mapping) else {}
        start = step_range.get("start")
        round_link = (
            f"<a href='/run/{html.escape(run_id)}#step-{int(start)}'>Round {index + 1} chat</a>"
            if isinstance(start, int) and not isinstance(start, bool)
            else str(index + 1)
        )
        rows.append(
            "<tr>"
            f"<td>{round_link}</td><td><span class='xview-verdict {css}'>{label}</span></td>"
            f"<td>{_compact(value.get('decisions'))}</td>"
            f"<td>{_compact(value.get('model_calls'))}</td>"
            f"<td>{_duration(value.get('wall_duration_s'))}</td>"
            f"<td>{html.escape(str(value.get('termination') or '—'))}</td></tr>"
        )
    successes = sum(
        isinstance(item, Mapping) and item.get("official_success") is True
        for item in rounds
    )
    return (
        "<p class='xview-warning'><strong>Persistent session:</strong> "
        f"{successes}/{len(rounds)} rounds succeeded. Aggregate official verdict is intentionally absent.</p>"
        "<div class='ui-table-scroll'><table class='xview-rounds'><thead><tr><th>round</th><th>official verdict</th>"
        "<th>decisions</th><th>calls</th><th>wall</th><th>termination</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _render_card(
    cell: Mapping[str, Any], evidence: Mapping[str, Any], run: Any | None,
    metric_resolver: Callable[[Path], Mapping[str, Any]], *, persistent: bool,
) -> str:
    label = html.escape(str(cell.get("label") or cell.get("id") or "cell"))
    role = html.escape(str(cell.get("role") or "accepted evidence"))
    summary = html.escape(str(cell.get("summary") or ""))
    expected_id = str(evidence.get("run_id") or "")
    if run is None:
        return (
            "<article class='xview-card'><div class='xview-card-body'>"
            f"<span class='xview-role'>{role}</span><h3>{label}</h3>"
            "<p class='xview-warning'>Accepted evidence is unavailable in the configured output roots.</p>"
            f"<code>{html.escape(expected_id)}</code></div></article>"
        )
    result = _load_json(Path(run.root) / "result.json")
    metrics = metric_resolver(Path(run.root))
    verdict, verdict_css = _verdict(getattr(run, "official_success", None))
    control = str(getattr(run, "control_validity", "indeterminate"))
    source_matches = str(evidence.get("source_commit") or "") == str(
        getattr(run, "source_commit", "")
    )
    cache = _number(metrics.get("cache_token_ratio"))
    cache_text = "—" if cache is None else f"{cache * 100:.1f}%"
    cost = metrics.get("cost") if isinstance(metrics.get("cost"), Mapping) else {}
    total_cost = _number(cost.get("total_cost"))
    currency = str(cost.get("currency") or "")
    cost_text = "—" if total_cost is None else f"{currency} {total_cost:.2f}".strip()
    card = (
        "<article class='xview-card'>"
        f"<video controls preload='metadata' src='/run/{html.escape(expected_id)}/trajectory.mp4'></video>"
        "<div class='xview-card-body'>"
        f"<span class='xview-role'>{role}</span><h3>{label}</h3>"
        f"<span class='xview-verdict {verdict_css}'>{verdict}</span> "
        f"<span class='xview-control {html.escape(control)}'>{html.escape(control)}</span>"
        f"<p>{summary}</p><div class='xview-metrics'>"
        f"<div><span>decisions</span><strong>{_compact(result.get('decisions'))}</strong></div>"
        f"<div><span>calls</span><strong>{_compact(result.get('model_calls'))}</strong></div>"
        f"<div><span>wall</span><strong>{_duration(result.get('wall_duration_s'))}</strong></div>"
        f"<div><span>peak context</span><strong>{_compact(metrics.get('context_peak'))}</strong></div>"
        f"<div><span>avg images</span><strong>{_compact(metrics.get('images_mean'))}</strong></div>"
        f"<div><span>cache / cost</span><strong>{cache_text} · {html.escape(cost_text)}</strong></div></div>"
        f"<div class='xview-links'><a href='/run/{html.escape(expected_id)}'>Chat + action notebook</a>"
        f"<a href='/run/{html.escape(expected_id)}/trajectory/raw'>Exact chat trace</a>"
        f"<a href='/run/{html.escape(expected_id)}/composition'>Composition</a>"
        f"<a href='/run/{html.escape(expected_id)}/trajectory.mp4'>Video</a></div>"
        f"<code>{'source-bound' if source_matches else 'SOURCE MISMATCH'} · "
        f"{html.escape(str(getattr(run, 'source_commit', ''))[:12])} · {html.escape(expected_id)}</code>"
    )
    has_rounds = isinstance(result.get("rounds"), list) and bool(result.get("rounds"))
    if persistent or has_rounds:
        card += _render_rounds(result, expected_id)
    return card + "</div></article>"


def _render_cells(
    manifest: Mapping[str, Any], definition: Mapping[str, Any], runs: Sequence[Any],
    metric_resolver: Callable[[Path], Mapping[str, Any]],
) -> str:
    accepted = definition["accepted_evidence"]
    runs_by_id = {str(getattr(run, "run_id", "")): run for run in runs}
    cells = manifest["cells"]
    cards = {
        str(cell["id"]): _render_card(
            cell, accepted[cell["accepted_evidence"]],
            runs_by_id.get(str(accepted[cell["accepted_evidence"]]["run_id"])),
            metric_resolver, persistent=manifest["layout"] == "persistent_rounds",
        )
        for cell in cells
    }
    if manifest["layout"] != "matrix":
        return f"<div class='xview-grid'>{''.join(cards[str(cell['id'])] for cell in cells)}</div>"
    rows = list(dict.fromkeys(str(cell["row"]) for cell in cells))
    columns = list(dict.fromkeys(str(cell["column"]) for cell in cells))
    by_position = {
        (str(cell["row"]), str(cell["column"])): cards[str(cell["id"])]
        for cell in cells
    }
    body = []
    for row in rows:
        body.append(
            f"<tr><th>{html.escape(row)}</th>" + "".join(
                f"<td>{by_position.get((row, column), '—')}</td>" for column in columns
            ) + "</tr>"
        )
    return (
        "<div class='xview-matrix-wrap'><table class='xview-matrix'><thead><tr><th>row / column</th>"
        + "".join(f"<th>{html.escape(column)}</th>" for column in columns)
        + f"</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
    )


def render_experiment_view(
    definition_path: Path | None, definition: Mapping[str, Any], runs: Sequence[Any],
    metric_resolver: Callable[[Path], Mapping[str, Any]],
) -> ExperimentViewProjection:
    path, manifest = load_experiment_view(definition_path)
    if path is None:
        return ExperimentViewProjection(present=False, html="", audit=None)
    audit = audit_experiment_view(path, manifest, definition, runs)
    if audit.structural_issues:
        issues = "".join(f"<li>{html.escape(item)}</li>" for item in audit.structural_issues)
        content = (
            f"<section class='xview'><div class='xview-invalid'>"
            "<strong>Invalid experiment Viewer manifest.</strong>"
            f"<ul>{issues}</ul></div></section>"
        )
        return ExperimentViewProjection(present=True, html=content, audit=audit)

    available = len(manifest["cells"]) - len(audit.unavailable_run_ids)
    outcomes = {"success": 0, "failure": 0, "unscored": 0}
    accepted = definition["accepted_evidence"]
    runs_by_id = {str(getattr(run, "run_id", "")): run for run in runs}
    for cell in manifest["cells"]:
        run_id = str(accepted[cell["accepted_evidence"]]["run_id"])
        run = runs_by_id.get(run_id)
        label, _css = _verdict(getattr(run, "official_success", None) if run else None)
        outcomes[label] += 1
    source_warning = ""
    if audit.evidence_issues:
        items = "".join(f"<li>{html.escape(item)}</li>" for item in audit.evidence_issues)
        source_warning = f"<div class='xview-invalid'><strong>Evidence identity audit failed.</strong><ul>{items}</ul></div>"
    cells_html = _render_cells(manifest, definition, runs, metric_resolver)
    finding_cards = []
    for finding in manifest.get("findings", []):
        priority = str(finding["priority"])
        refs = ", ".join(str(item) for item in finding["evidence_cells"])
        finding_cards.append(
            f"<article class='xview-finding {html.escape(priority)}'><header>"
            f"<span>{html.escape(str(finding['layer']).replace('_', ' / '))}</span>"
            f"<span>{html.escape(priority)}</span></header>"
            f"<span class='xview-epistemic'>{html.escape(str(finding['epistemic_status']))}</span>"
            f"<h3>{html.escape(str(finding['title']))}</h3>"
            f"<p>{html.escape(str(finding['summary']))}</p>"
            f"<div class='xview-evidence-refs'>evidence cells · {html.escape(refs)}</div></article>"
        )
    exclusions = []
    for exclusion in manifest.get("exclusions", []):
        evidence = accepted[exclusion["evidence"]]
        explanation = exclusion.get("summary") or evidence.get("reason") or exclusion["reason"]
        exclusions.append(
            f"<li><code>{html.escape(str(evidence.get('run_id') or exclusion['evidence']))}</code> · "
            f"{html.escape(str(exclusion['reason']))} · {html.escape(str(explanation))}</li>"
        )
    exclusion_html = (
        "<section class='xview-exclusions'><strong>Excluded evidence</strong><ul>"
        + "".join(exclusions) + "</ul></section>"
        if exclusions else ""
    )
    content = (
        f"<section class='xview'>"
        "<header class='xview-hero'><span class='eyebrow'>Versioned experiment view · "
        f"{html.escape(str(manifest['layout']))}</span><h2>{html.escape(str(manifest['title']))}</h2>"
        f"<p>{html.escape(str(manifest.get('subtitle') or ''))}</p>"
        "<div class='xview-roster'>"
        f"<div><span>accepted cells</span><strong>{len(manifest['cells'])}</strong></div>"
        f"<div><span>locally available</span><strong>{available}</strong></div>"
        f"<div><span>exclusions</span><strong>{len(manifest.get('exclusions', []))}</strong></div>"
        f"<div><span>layout</span><strong>{html.escape(str(manifest['layout']))}</strong></div></div></header>"
        f"{source_warning}<h2 class='section-title'>Outcome summary</h2><div class='xview-outcomes'>"
        f"<div><span>success</span><strong>{outcomes['success']}</strong></div>"
        f"<div><span>failure</span><strong>{outcomes['failure']}</strong></div>"
        f"<div><span>unscored / unavailable</span><strong>{outcomes['unscored']}</strong></div>"
        f"<div><span>pooled claim</span><strong>forbidden</strong></div></div>"
        "<h2 class='section-title'>Trajectory gallery</h2>"
        f"{cells_html}<p class='xview-warning'><strong>Claim boundary:</strong> "
        f"{html.escape(str(manifest['claim_boundary']))}</p>"
        "<h2 class='section-title'>Diagnostics and interpretation</h2>"
        f"<div class='xview-findings'>{''.join(finding_cards)}</div>{exclusion_html}</section>"
    )
    return ExperimentViewProjection(present=True, html=content, audit=audit)


def audit_experiment_view_files(
    root: Path, runs: Iterable[Any],
) -> tuple[ExperimentViewAudit, ...]:
    registrations = tuple(runs)
    reports = []
    for path in sorted((root / "experiments").glob("*/viewer.yaml")):
        definition_path = path.with_name("experiment.yaml")
        definition = _load_yaml(definition_path)
        manifest = _load_yaml(path)
        experiment = str(manifest.get("experiment") or definition.get("experiment") or "")
        experiment_runs = [
            run for run in registrations
            if str(getattr(run, "experiment", "")) == experiment
        ]
        reports.append(audit_experiment_view(path, manifest, definition, experiment_runs))
    return tuple(reports)
