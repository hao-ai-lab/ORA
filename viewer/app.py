"""Plain, auditable viewer for plans, transcripts, images and motions."""

from __future__ import annotations

import argparse
import html
import json
import time
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from threading import Timer
from typing import Any, Mapping
from urllib.parse import urlencode

import yaml
from flask import Flask, Response, abort, jsonify, request, send_file

from ora.orchestrator.paths import (
    OUTPUTS_ROOT,
    iter_run_roots,
    output_roots as default_output_roots,
    resolve_run_root,
)
from viewer.libero_pro_gallery import create_libero_pro_blueprint
from viewer.libero_previews import load_preview_manifest
from viewer.robodojo_catalog import (
    catalog_payload as robodojo_catalog_payload,
    task as robodojo_task,
)
from viewer.robotwin_catalog import (
    EMBODIMENTS,
    FAMILY_LABELS,
    FAMILY_ORDER,
    ROBOTWIN_REVISION,
    TASKS,
    TASK_BY_ID,
    RobotTwinTask,
    catalog_payload,
)
from viewer.registry import (
    discover_run_roots,
    registry as build_registry,
    resolve_discovered_run,
    visible_runs,
)
from viewer.experiment_view import (
    audit_experiment_view_files,
    render_experiment_view,
)
from viewer.pricing import estimate_run_cost, load_price_catalog
from viewer.ui import render_page, render_style_guide
from viewer.experiment_identity import (
    create_identity_blueprint, filter_identity_request, identity_controls, identity_label,
)
from viewer.player import render_process_player
from viewer.process_timeline import compile_process_timeline
from viewer.process_video import PresentationPendingError, VideoOptions, presentation_video
from viewer.rounds import render_round_matrix
from viewer.review_media import find_execution_media
from viewer.replay import REPLAY_MODES
from ora.agent.composition import inspect_run_configuration
from viewer.trajectory import (
    TrajectoryVideoPendingError,
    compile_raw_trajectory, compile_trajectory, compile_trajectory_video,
    pending_replay_source, raw_trajectory_event, trajectory_video_cached,
)


ROOT = Path(__file__).resolve().parents[1]
app = Flask(__name__)
app.register_blueprint(create_identity_blueprint(lambda: viewer_run_roots()))
VIEWER_START_PAGES = {
    "runs": "/",
    "libero": "/gallery/libero",
    "libero-pro": "/gallery/libero-pro",
    "robotwin": "/gallery/robotwin",
    "robodojo": "/gallery/robodojo",
}

def load(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def load_yaml(path: Path) -> dict:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, yaml.YAMLError):
        return {}


def experiment_definition(experiment_id: str) -> tuple[Path | None, dict]:
    """Return the tracked declaration for one registered experiment."""

    for candidate in sorted((ROOT / "experiments").glob("*/experiment.yaml")):
        value = load_yaml(candidate)
        if str(value.get("experiment")) == experiment_id:
            return candidate, value
    return None, {}


def run_root(run_id: str) -> Path:
    try:
        configured = app.config.get("VIEWER_OUTPUT_ROOTS")
        root = (
            resolve_discovered_run(run_id, configured)
            if configured else resolve_run_root(run_id)
        )
    except RuntimeError:
        abort(404)
    return root


# Recursive run discovery walks every configured evidence root (including
# legacy worktree snapshots) and costs seconds per request; a short TTL keeps
# interactive filtering instant while a newly landed run still appears within
# one refresh interval.
_RUN_ROOT_TTL_S = 5.0
_RUN_ROOT_CACHE: dict[tuple[Path, ...], tuple[float, list[Path]]] = {}


def viewer_run_roots() -> list[Path]:
    configured = app.config.get("VIEWER_OUTPUT_ROOTS")
    key = tuple(configured) if configured else tuple(default_output_roots())
    now = time.monotonic()
    cached = _RUN_ROOT_CACHE.get(key)
    if cached is not None and now - cached[0] < _RUN_ROOT_TTL_S:
        return list(cached[1])
    roots = list(discover_run_roots(configured) if configured else iter_run_roots())
    _RUN_ROOT_CACHE[key] = (time.monotonic(), roots)
    return list(roots)


def viewer_output_roots() -> tuple[Path, ...]:
    configured = app.config.get("VIEWER_OUTPUT_ROOTS")
    return tuple(configured) if configured else default_output_roots()


def scoped_run_roots(*, experiment: str = "", suite: str = "", task_id: int | None = None) -> list[Path]:
    """Select cheap RunPlan identities before loading unrelated trace evidence."""
    selected = []
    for root in viewer_run_roots():
        plan = load(root / "plan.json")
        if experiment and str(plan.get("experiment") or root.parent.parent.name) != experiment:
            continue
        if suite:
            runtime = plan.get("runtime") or {}
            if runtime.get("kind") != "libero_pro":
                continue
            if runtime.get("suite") != suite or runtime.get("task_id") != task_id:
                if str(plan.get("task")) != f"{suite}-t{task_id}":
                    continue
        selected.append(root)
    return selected


def _utc_parts(value: float | None) -> tuple[str, str, str]:
    if not isinstance(value, (int, float)):
        return "Date unknown", "—", ""
    instant = datetime.fromtimestamp(value, tz=timezone.utc)
    return (
        instant.strftime("%Y-%m-%d"),
        instant.strftime("%H:%M:%S UTC"),
        instant.isoformat().replace("+00:00", "Z"),
    )


def _libero_task_runs_url(
    suite: str, task_id: int, **query: str,
) -> str:
    values = {key: value for key, value in query.items() if value}
    path = f"/task/libero_pro/{suite}/{task_id}/runs"
    return path + (f"?{urlencode(values)}" if values else "")


def _registered_outcome(run: object) -> tuple[str, str, str]:
    """Keep multi-round success evidence distinct from one-episode verdicts."""
    count = getattr(run, "round_count", None)
    successes = getattr(run, "round_success_count", None)
    if isinstance(count, int) and count > 0 and isinstance(successes, int):
        css = "ok" if successes == count else "bad" if successes == 0 else ""
        return f"{successes}/{count} rounds", css, "persistent K-session"
    verdict = getattr(run, "official_success", None)
    if verdict is True:
        return "success", "ok", "single episode verdict"
    if verdict is False:
        return "failure", "bad", "single episode verdict"
    return "—", "", "unscored"


def page(title: str, body: str, *, wide: bool = False) -> str:
    return render_page(title, body, wide=wide)


@app.get("/style-guide")
def style_guide() -> str:
    return render_style_guide()


@app.get("/")
def index() -> str:
    view = request.args.get("view", "formal")
    if view not in {"formal", "attempts", "qualifications", "diagnostics", "all"}:
        view = "formal"
    backend = request.args.get("backend", "")
    experiment = request.args.get("experiment", "")
    task_key = request.args.get("task", "")
    seed = request.args.get("seed", "")
    state = request.args.get("state", "")
    outcome = request.args.get("outcome", "")
    registrations = build_registry(viewer_run_roots(), inspect_video=False)
    registrations = filter_identity_request(registrations, request.args)
    backends = sorted({run.backend for run in registrations if run.backend != "unknown"})
    experiments = sorted({
        run.experiment for run in registrations
        if not backend or run.backend == backend
    })
    if experiment not in experiments:
        experiment = ""
    task_registrations = [
        run for run in registrations
        if run.task_key is not None
        and (not backend or run.backend == backend)
        and (not experiment or run.experiment == experiment)
    ]
    task_keys = sorted({str(run.task_key) for run in task_registrations})
    if task_key not in task_keys:
        task_key = ""
    task_seeds = sorted(
        {str(run.seed) for run in task_registrations if run.task_key == task_key},
        key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value),
    ) if task_key else []
    if seed not in task_seeds:
        seed = ""
    states = sorted({
        run.state for run in registrations
        if run.state
        and (not backend or run.backend == backend)
        and (not experiment or run.experiment == experiment)
    })
    if state not in states:
        state = ""
    if outcome not in {"success", "failure", "unscored"}:
        outcome = ""
    selected = visible_runs(
        registrations, view=view, backend=backend, experiment=experiment,
        task_key=task_key, seed=seed, state=state, outcome=outcome,
    )
    counts = {category: sum(run.category == category for run in registrations) for category in ("formal", "attempt", "qualification", "diagnostic")}
    date_counts: dict[str, int] = {}
    for run in selected:
        date, _clock, _iso = _utc_parts(run.started_unix_s)
        date_counts[date] = date_counts.get(date, 0) + 1
    rows = []
    current_date = None
    for run in selected:
        label, outcome_class, outcome_kind = _registered_outcome(run)
        video = (
            "check in run" if run.video_cached is None
            else "ready" if run.video_cached else "not compiled"
        )
        date, clock, iso = _utc_parts(run.started_unix_s)
        if date != current_date:
            rows.append(
                f"<tr class='run-date-row'><th colspan='8'>{html.escape(date)}"
                f"<span>{date_counts[date]} run{'s' if date_counts[date] != 1 else ''}</span></th></tr>"
            )
            current_date = date
        category_label = {
            "formal": "formal behavioral",
            "attempt": "behavioral attempt",
            "qualification": "qualification",
            "diagnostic": "synthetic diagnostic",
        }.get(run.category, run.category)
        time_attr = f" datetime='{html.escape(iso)}'" if iso else ""
        task_label = html.escape(run.task)
        if run.task_key is not None:
            task_url = _libero_task_runs_url(str(run.suite), int(run.task_id))
            task_label = f"<a href='{html.escape(task_url)}'>{task_label}</a>"
        rows.append(
            f"<tr><td class='run-time'><time{time_attr}><strong>{html.escape(clock.removesuffix(' UTC'))}</strong>"
            "<small>UTC</small></time></td>"
            f"<td class='run-experiment'><a href='/experiment/{html.escape(run.experiment)}/runs'>{html.escape(run.experiment)}</a>"
            f"{identity_label(run)}<small>{html.escape(run.config)}</small></td>"
            f"<td class='run-id'><a href='/run/{html.escape(run.run_id)}'><code>{html.escape(run.run_id)}</code></a>"
            f"<small>source {html.escape(run.source_commit[:12])}</small></td>"
            f"<td><span class='run-kind {html.escape(run.category)}'>{html.escape(category_label)}</span>"
            f"<small>{html.escape(run.milestone)} · {html.escape(run.evidence_tier)}</small></td>"
            f"<td>{task_label}"
            f"<small>seed {html.escape(str(run.seed))}</small></td>"
            f"<td>{html.escape(run.backend)}<small>{html.escape(run.policy_kind)} · {html.escape(run.model)}</small></td>"
            f"<td class='{outcome_class}'>{html.escape(run.state)} · {html.escape(label)}"
            f"<small>{html.escape(outcome_kind)} · evaluation {html.escape(run.evaluation)}</small>"
            f"<small>control {html.escape(run.control_validity)}</small></td>"
            f"<td><div class='review-links'><a href='/run/{html.escape(run.run_id)}'>Summary</a>"
            f"<a href='/run/{html.escape(run.run_id)}/composition'>Composition</a>"
            f"<a href='/run/{html.escape(run.run_id)}/trajectory/raw'>Raw trace</a>"
            f"<a href='/run/{html.escape(run.run_id)}/raw'>Evidence</a></div>"
            f"<small>video {video}</small></td></tr>"
        )
    tabs = "".join(
        f"<a class='{'active' if value == view else ''}' href='/?{urlencode({'family': request.args.get('family', ''), 'generation': request.args.get('generation', ''), 'view': value, 'backend': backend, 'experiment': experiment, 'task': task_key, 'seed': seed, 'state': state, 'outcome': outcome})}'>{label}</a>"
        for value, label in (
            ("formal", f"Formal · {counts['formal']}"),
            ("attempts", f"Behavioral · {counts['formal'] + counts['attempt']}"),
            ("qualifications", f"Qualifications · {counts['qualification']}"),
            ("diagnostics", f"Diagnostics · {counts['diagnostic']}"),
            ("all", f"All · {len(registrations)}"),
        )
    )
    backend_options = ["<option value=''>All backends</option>"] + [
        f"<option value='{html.escape(value)}'{' selected' if value == backend else ''}>{html.escape({'robotwin': 'RoboTwin', 'libero_pro': 'LIBERO / LIBERO-Pro'}.get(value, value))}</option>"
        for value in backends
    ]
    experiment_options = ["<option value=''>All experiments</option>"] + [
        f"<option value='{html.escape(value)}'{' selected' if value == experiment else ''}>{html.escape(value)}</option>"
        for value in experiments
    ]
    task_labels = {
        str(run.task_key): f"{run.suite} / task {run.task_id}"
        for run in task_registrations
    }
    task_options = ["<option value=''>All tasks</option>"] + [
        f"<option value='{html.escape(value)}'{' selected' if value == task_key else ''}>{html.escape(task_labels[value])}</option>"
        for value in task_keys
    ]
    seed_options = ["<option value=''>All seeds</option>"] + [
        f"<option value='{html.escape(value)}'{' selected' if value == seed else ''}>{html.escape(value)}</option>"
        for value in task_seeds
    ]
    state_options = ["<option value=''>All lifecycle states</option>"] + [
        f"<option value='{html.escape(value)}'{' selected' if value == state else ''}>{html.escape(value)}</option>"
        for value in states
    ]
    outcome_options = "".join(
        f"<option value='{value}'{' selected' if value == outcome else ''}>{label}</option>"
        for value, label in (
            ("", "All official outcomes"),
            ("success", "Official success"),
            ("failure", "Official failure"),
            ("unscored", "Unscored"),
        )
    )
    evidence_roots = viewer_output_roots()
    root_items = "".join(
        f"<li><code>{html.escape(str(path))}</code></li>" for path in evidence_roots
    )
    body = "<span class='eyebrow'>All experiments · ORA run registry</span><h1>ORA experiments</h1><p class='muted'>Filter every discovered RoboTwin and LIBERO run by experiment, lifecycle and official outcome. Formal runs are completed, evaluator-valid, sealed behavioral evidence; synthetic and diagnostic runs remain available under All.</p>"
    body += (
        "<details class='provenance'><summary>Evidence workspace · "
        f"{html.escape(str(OUTPUTS_ROOT))}</summary>"
        "<p>New runs from every linked worktree use the shared evidence root. "
        "Legacy worktree and imported roots remain read-only discovery inputs.</p>"
        f"<ul>{root_items}</ul></details>"
    )
    body += "<nav class='nav'><a href='/experiments'>Experiment families</a><a href='/costs'>Cost ledger</a><a href='/gallery/libero'>LIBERO 130 atlas</a><a href='/gallery/robotwin'>RoboTwin tasks</a><a href='/gallery/robodojo'>RoboDojo tasks</a></nav>"
    body += (
        f"<div class='registry-controls'><nav class='tabs'>{tabs}</nav><form>"
        f"<input type='hidden' name='view' value='{html.escape(view)}'>"
        f"{identity_controls(request.args)}"
        f"<label>Experiment<select name='experiment' onchange='this.form.submit()'>{''.join(experiment_options)}</select></label>"
        f"<label>Backend<select name='backend' onchange='this.form.submit()'>{''.join(backend_options)}</select></label>"
        f"<label>Task<select name='task' onchange='this.form.submit()'>{''.join(task_options)}</select></label>"
        f"<label>Seed<select name='seed' onchange='this.form.submit()'{' disabled' if not task_key else ''}>{''.join(seed_options)}</select></label>"
        f"<label>Lifecycle<select name='state' onchange='this.form.submit()'>{''.join(state_options)}</select></label>"
        f"<label>Outcome<select name='outcome' onchange='this.form.submit()'>{outcome_options}</select></label>"
        f"<a href='/?view={html.escape(view)}'>Clear filters</a>"
        "</form></div>"
    )
    body += f"<p class='muted'><strong>{len(selected)}</strong> matching runs across <strong>{len({run.experiment for run in selected})}</strong> experiments.</p>"
    body += (
        "<table class='run-table'><thead><tr><th>time</th><th>experiment / config</th>"
        "<th>exact run ID</th><th>type</th><th>task</th><th>runtime / policy</th>"
        "<th>outcome</th><th>review</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>"
    )
    return page("ORA Runtime", body)


@app.get("/costs")
def cost_dashboard() -> str:
    """Aggregate provider-reported token spend across behavioral runs."""

    view = request.args.get("view", "formal")
    if view not in {"formal", "behavioral"}:
        view = "formal"
    registrations = [
        run for run in build_registry(viewer_run_roots(), inspect_video=False)
        if run.backend == "libero_pro"
        and run.category in {"formal", "attempt"}
    ]
    experiments = sorted({run.experiment for run in registrations})
    experiment = request.args.get("experiment", "")
    if experiment not in experiments:
        experiment = ""
    selected = [
        run for run in registrations
        if (view != "formal" or run.category == "formal")
        and (not experiment or run.experiment == experiment)
    ]
    estimates = {run.run_id: _cost_projection(run.root) for run in selected}
    priced = [
        value for value in estimates.values()
        if value.get("total_cost") is not None
    ]
    total_cost = sum(float(value["total_cost"]) for value in priced)
    total_input = sum(float(value.get("input_tokens") or 0) for value in priced)
    total_cached = sum(float(value.get("cached_tokens") or 0) for value in priced)
    total_output = sum(float(value.get("output_tokens") or 0) for value in priced)
    cache_ratio = total_cached / total_input if total_input else None

    task_groups: dict[str, list] = {}
    for run in selected:
        task_groups.setdefault(str(run.task_key or run.task), []).append(run)
    task_cards = []
    for key, runs in sorted(task_groups.items()):
        values = [
            estimates[run.run_id] for run in runs
            if estimates[run.run_id].get("total_cost") is not None
        ]
        task_cost = sum(float(value["total_cost"]) for value in values)
        successes = sum(run.official_success is True for run in runs)
        task_link = html.escape(runs[0].task)
        if runs[0].task_key is not None:
            task_link = (
                f"<a href='{html.escape(_libero_task_runs_url(str(runs[0].suite), int(runs[0].task_id)))}'>"
                f"{task_link}</a>"
            )
        task_cards.append(
            "<article class='task-cost-card'>"
            f"<span class='eyebrow'>{html.escape(key)}</span><h3>{task_link}</h3>"
            f"<strong>{_money(task_cost) if values else '—'}</strong>"
            f"<p>{len(values)}/{len(runs)} priced runs · {successes} success · "
            f"{_money(task_cost / len(values)) if values else '—'} mean/run</p></article>"
        )

    rows = []
    for run in selected:
        value = estimates[run.run_id]
        date, clock, _iso = _utc_parts(run.started_unix_s)
        verdict = run.official_success
        verdict_text = "unscored" if verdict is None else (
            "success" if verdict else "failure"
        )
        input_tokens = value.get("input_tokens")
        cached_tokens = value.get("cached_tokens")
        run_cache_ratio = (
            float(cached_tokens) / float(input_tokens)
            if _metric_number(cached_tokens) is not None
            and _metric_number(input_tokens) not in {None, 0}
            else None
        )
        task_link = html.escape(run.task)
        if run.task_key is not None:
            task_link = (
                f"<a href='{html.escape(_libero_task_runs_url(str(run.suite), int(run.task_id)))}'>"
                f"{task_link}</a>"
            )
        rows.append(
            "<tr>"
            f"<td>{html.escape(date)}<small>{html.escape(clock)}</small></td>"
            f"<td>{task_link}<small>seed {html.escape(str(run.seed))} · {html.escape(str(run.task_key or ''))}</small></td>"
            f"<td><a href='/experiment/{html.escape(run.experiment)}/runs'>{html.escape(run.experiment)}</a>"
            f"<small>{html.escape(run.config)} · {html.escape(run.model)}</small></td>"
            f"<td><span class='outcome-mark {html.escape(verdict_text)}'>{html.escape(verdict_text)}</span>"
            f"<small>{html.escape(run.category)} · {html.escape(run.state)}</small></td>"
            "<td class='token-breakdown'>"
            f"<b>{_compact_metric(value.get('input_tokens'))}</b> input · "
            f"{_compact_metric(value.get('cached_tokens'))} hit · "
            f"{_compact_metric(value.get('uncached_tokens'))} miss"
            f"<small>{_compact_metric(value.get('output_tokens'))} output · "
            f"{_compact_metric(value.get('auxiliary_calls'))} auxiliary calls</small></td>"
            f"<td>{'—' if run_cache_ratio is None else f'{run_cache_ratio * 100:.1f}%'}"
            f"<small>token-weighted cache read</small></td>"
            f"<td>{_cost_cell(value)}</td>"
            f"<td><div class='review-links'><a href='/run/{html.escape(run.run_id)}'>Trajectory</a>"
            f"<a href='/run/{html.escape(run.run_id)}/composition'>Composition</a>"
            f"<a href='/run/{html.escape(run.run_id)}/raw'>Evidence</a></div>"
            f"<small><code>{html.escape(run.run_id)}</code></small></td></tr>"
        )

    price_catalog = load_price_catalog()
    prices = price_catalog.get("prices") if isinstance(price_catalog, dict) else []
    price = prices[0] if isinstance(prices, list) and prices else {}
    rates = price.get("rates_per_million_tokens", {}) if isinstance(price, dict) else {}
    experiment_options = ["<option value=''>All experiments</option>"] + [
        f"<option value='{html.escape(value)}'{' selected' if value == experiment else ''}>{html.escape(value)}</option>"
        for value in experiments
    ]
    query_common = {"experiment": experiment}
    tabs = "".join(
        f"<a class='{'active' if value == view else ''}' href='/costs?{urlencode({**query_common, 'view': value})}'>{label}</a>"
        for value, label in (
            ("formal", "Formal completed"),
            ("behavioral", "All behavioral / live"),
        )
    )
    body = (
        "<nav class='nav'><a href='/'>← Run registry</a></nav>"
        "<section class='cost-hero'><div><span>Provider spend observatory</span>"
        "<h1>What did one robot task cost?</h1>"
        "<p>Direct provider counters, split into cache-hit input, cache-miss input and output. "
        "Context-compactor calls are included rather than hidden.</p></div>"
        f"<div><dt>selected spend</dt><dd>{_money(total_cost) if priced else '—'}</dd></div>"
        f"<div><dt>priced runs</dt><dd>{len(priced)} / {len(selected)}</dd></div>"
        f"<div><dt>cache reuse</dt><dd>{'—' if cache_ratio is None else f'{cache_ratio * 100:.1f}%'}</dd></div></section>"
        "<p class='price-note'><strong>Official Kimi K3 API list-price equivalent.</strong> "
        f"Cache hit ¥{html.escape(str(rates.get('cache_hit_input', '—')))} / MTok · "
        f"cache miss ¥{html.escape(str(rates.get('cache_miss_input', '—')))} / MTok · "
        f"output ¥{html.escape(str(rates.get('output', '—')))} / MTok. "
        "This is not an invoice: subscriptions, negotiated discounts, taxes and non-model infrastructure are excluded. "
        f"<a href='{html.escape(str(price.get('source_url') or '#'))}'>Official source</a> · "
        f"snapshot <code>{html.escape(str(price.get('price_id') or 'missing'))}</code>.</p>"
        f"<div class='registry-controls'><nav class='tabs'>{tabs}</nav>"
        f"<form><input type='hidden' name='view' value='{html.escape(view)}'>"
        f"<label>Experiment<select name='experiment' onchange='this.form.submit()'>{''.join(experiment_options)}</select></label>"
        "</form></div>"
        "<span class='eyebrow'>Task roll-up</span>"
        f"<div class='task-cost-grid'>{''.join(task_cards) or '<p>No matching tasks.</p>'}</div>"
        "<div class='cohort-scroll'><table class='cost-ledger'><thead><tr>"
        "<th>date</th><th>task</th><th>experiment / model</th><th>result</th>"
        "<th>provider tokens</th><th>cache read</th><th>list-price cost</th><th>review</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
        f"<p class='measurement-footnote'>Selected totals: {_compact_metric(total_input)} input tokens, "
        f"{_compact_metric(total_cached)} cache-read tokens and {_compact_metric(total_output)} output tokens. "
        "Sealed runs use deterministic runtime-metrics.json; live rows are explicitly provisional and update "
        "from the current Native AgentSession without modifying run evidence.</p>"
    )
    return page("Provider cost ledger", body, wide=True)


def _gallery_url(task_id: str | None = None, **query: str) -> str:
    path = "/gallery/robotwin" + (f"/{task_id}" if task_id else "")
    values = {key: value for key, value in query.items() if value}
    return path + (f"?{urlencode(values)}" if values else "")


def _task_matches(task: RobotTwinTask, query: str, family: str) -> bool:
    if family and task.family != family:
        return False
    needle = query.casefold().strip()
    haystack = " ".join((task.id, task.title, task.requirement, *task.objects)).casefold()
    return not needle or needle in haystack


def _robotwin_gallery(selected_id: str | None = None) -> str:
    query = request.args.get("q", "").strip()
    family = request.args.get("family", "")
    if family not in FAMILY_LABELS:
        family = ""
    visible = [task for task in TASKS if _task_matches(task, query, family)]
    selected = TASK_BY_ID.get(selected_id or "click_bell")
    if selected is None:
        abort(404)
    if visible and selected not in visible:
        selected = visible[0]

    embodiment = request.args.get("embodiment", selected.video_embodiments[0])
    if embodiment not in selected.video_embodiments:
        embodiment = selected.video_embodiments[0]
    viewpoint = request.args.get("viewpoint", "world")
    if viewpoint not in {"world", "head"}:
        viewpoint = "world"
    retained = {"q": query, "family": family}

    family_options = [f"<option value=''>All families · {len(TASKS)}</option>"]
    for family_id in FAMILY_ORDER:
        count = sum(task.family == family_id for task in TASKS)
        label = FAMILY_LABELS[family_id][0]
        selected_option = " selected" if family_id == family else ""
        family_options.append(
            f"<option value='{family_id}'{selected_option}>{html.escape(label)} · {count}</option>"
        )

    embodiment_tabs = "".join(
        f"<a class='{'active' if identifier == embodiment else ''}' href='"
        f"{html.escape(_gallery_url(selected.id, **retained, embodiment=identifier, viewpoint=viewpoint))}'>"
        f"{html.escape(label)}</a>"
        for identifier, label in EMBODIMENTS
        if identifier in selected.video_embodiments
    )
    viewpoint_tabs = "".join(
        f"<a class='{'active' if item == viewpoint else ''}' href='"
        f"{html.escape(_gallery_url(selected.id, **retained, embodiment=embodiment, viewpoint=item))}'>"
        f"{'Full world' if item == 'world' else 'Head camera'}</a>"
        for item in ("world", "head")
    )
    rates = "".join(
        f"<div><span>{html.escape(label)}</span><strong>{selected.generation_success_percent[identifier]}%</strong>"
        f"<small>{'official video' if identifier in selected.video_embodiments else 'no video'}</small></div>"
        for identifier, label in EMBODIMENTS
    )
    objects = "".join(f"<code>{html.escape(item)}</code>" for item in selected.objects)
    cards = "".join(
        f"<a class='task-card {'active' if task.id == selected.id else ''}' href='"
        f"{html.escape(_gallery_url(task.id, **retained))}'>"
        f"<small>{TASKS.index(task) + 1:02d} · {html.escape(FAMILY_LABELS[task.family][0])}</small>"
        f"<strong>{html.escape(task.title)}</strong><code>{html.escape(task.id)}</code>"
        f"<span>{html.escape(task.requirement)}</span></a>"
        for task in visible
    ) or "<p>No tasks match this filter.</p>"
    selected_number = TASKS.index(selected) + 1
    video_url = selected.video_url(embodiment, viewpoint)

    body = "<nav class='nav'><a href='/'>← Runs</a><a href='/api/robotwin/tasks'>JSON</a></nav>"
    body += (
        "<section class='gallery-head'><div><span class='eyebrow'>Task catalog</span>"
        "<h1>RoboTwin Tasks</h1><p>Browse tasks, videos, objects, and generation stats.</p></div>"
        f"<dl><div><dt>Total</dt><dd>{len(TASKS)}</dd></div><div><dt>Shown</dt><dd>{len(visible)}</dd></div>"
        "<div><dt>Views</dt><dd>2</dd></div></dl></section>"
    )
    body += (
        "<form class='filters' method='get' action='/gallery/robotwin'>"
        f"<label>Search<input name='q' value='{html.escape(query)}' placeholder='task, object, or requirement'></label>"
        f"<label>Family<select name='family'>{''.join(family_options)}</select></label>"
        "<button type='submit'>Apply filter</button></form>"
    )
    body += (
        "<details class='provenance'><summary>About these videos</summary>"
        "<p>RoboTwin's task-specific <code>play_once()</code> reads privileged simulator state, plans a path, and records an episode. "
        "Only planner-valid episodes satisfying the task predicate enter generated data.</p>"
        "<dl class='provenance-grid'><div><dt>Action teacher</dt><dd>Task-specific expert program and CuRobo path</dd></div>"
        "<div><dt>Task ground truth</dt><dd>SAPIEN actor poses, functional annotations, task config and seed</dd></div>"
        "<div><dt>Verifier</dt><dd><code>plan_success &amp;&amp; check_success()</code> during generation</dd></div>"
        "<div><dt>Policy visibility</dt><dd>These privileged inputs remain outside the ORA policy boundary</dd></div></dl>"
        "<p class='muted'>These are expert examples, not policy rollouts or Harness results.</p></details>"
    )
    body += (
        f"<section class='task-detail'><header><div><span class='eyebrow'>Task {selected_number:02d} / 50 · "
        f"{html.escape(FAMILY_LABELS[selected.family][0])}</span><h2>{html.escape(selected.title)}</h2>"
        f"<code>{html.escape(selected.id)}</code></div><a href='{html.escape(selected.official_page)}' target='_blank' rel='noreferrer'>official task page ↗</a></header>"
        f"<p>{html.escape(selected.requirement)}</p><div class='grid'><section><div class='tabs'>{embodiment_tabs}</div>"
        f"<div class='tabs'>{viewpoint_tabs}</div><video controls muted loop playsinline preload='metadata' src='{html.escape(video_url)}'></video>"
        f"<p class='muted'>{html.escape(embodiment)} · {html.escape(viewpoint)} · official 320×240 MP4</p></section>"
        "<section><dl class='facts'>"
        f"<div><dt>Avg. steps</dt><dd>{selected.average_steps}</dd></div>"
        f"<div><dt>Embodiments</dt><dd>{len(selected.video_embodiments)}</dd></div>"
        "<div><dt>Views</dt><dd>world / head</dd></div></dl>"
        f"<h3>Objects</h3><div class='chips'>{objects}</div><h3>Official data-generation acceptance</h3>"
        "<p class='muted'>Generator acceptance rates, not policy success rates.</p>"
        f"<div class='rates'>{rates}</div><p><a href='{html.escape(selected.source_code)}' target='_blank' rel='noreferrer'>Environment source ↗</a></p>"
        "</section></div></section>"
    )
    body += f"<section><h2>All tasks</h2><div class='task-grid'>{cards}</div></section>"
    body += (
        "<p class='catalog-source'>Catalog migrated from Agent-as-Policy commit "
        "<code>a762defefdecff6286ba3079cb02a77cd5327141</code>. Runtime source links are pinned to RoboTwin "
        f"<code>{ROBOTWIN_REVISION}</code>; official documentation and MP4 hosting remain external.</p>"
    )
    return page("RoboTwin Tasks", body)


@app.get("/gallery/robotwin")
def robotwin_gallery() -> str:
    return _robotwin_gallery()


@app.get("/gallery/robotwin/<task_id>")
def robotwin_task(task_id: str) -> str:
    return _robotwin_gallery(task_id)


def _robodojo_gallery(selected_id: str | None = None) -> str:
    """Render the RoboDojo capability-dimension atlas.

    The task list comes from the checkout's own inventory rather than a copied
    table, so it stays correct as upstream changes. Where local demonstration
    data has been fetched, each episode's head and wrist previews are offered
    alongside its recorded instruction.
    """
    payload = robodojo_catalog_payload()
    body = (
        "<nav class='nav'><a href='/'>← Runs</a>"
        "<a href='/gallery/robotwin'>RoboTwin tasks</a>"
        "<a href='/api/robodojo/tasks'>JSON</a></nav>"
    )
    if not payload["available"]:
        body += (
            "<p class='muted'>RoboDojo is not bound on this host. Set "
            f"<code>{html.escape(payload['root_env'])}</code> to a checkout, and "
            f"<code>{html.escape(payload['data_root_env'])}</code> to the "
            "demonstration dataset to see recorded episodes.</p>"
        )
        return page("RoboDojo tasks", body)

    counts = payload["counts"]
    body += (
        f"<p class='muted'>{counts['tasks']} tasks · {counts['runnable']} runnable · "
        f"{counts['with_demonstrations']} with local demonstrations</p>"
    )

    selected = robodojo_task(selected_id) if selected_id else None
    if selected is not None:
        body += f"<h2>{html.escape(selected.title)}</h2>"
        body += (
            "<p class='muted'>"
            f"{html.escape(selected.dimension)} · {html.escape(selected.variant)} · "
            f"{'runnable' if selected.runnable else 'not runnable'}</p>"
        )
        if selected.instruction:
            body += f"<p><em>{html.escape(selected.instruction)}</em></p>"
        body += (
            "<pre>" + html.escape(selected.config) + "\n"
            + html.escape(selected.module) + "</pre>"
        )
        if selected.demonstrations:
            body += (
                f"<p class='muted'>{len(selected.demonstrations)} recorded episodes · "
                "expert demonstrations, not policy output</p>"
            )
            for demo in selected.demonstrations[:3]:
                label = html.escape(demo["episode"])
                available = demo.get("episodes_available")
                if demo.get("source") == "simulation":
                    label += (
                        f" <span class='muted'>· simulation "
                        f"({html.escape(str(demo.get('dataset', '')))})"
                        + (f", {available} episodes in dataset" if available else "")
                        + "</span>"
                    )
                body += f"<h3>{label}</h3><p>"
                for view in sorted(demo["views"]):
                    src = (
                        f"/gallery/robodojo/{html.escape(selected.id)}"
                        f"/video/{html.escape(demo['episode'])}/{html.escape(view)}"
                    )
                    body += (
                        f"<span style='display:inline-block;margin:0 12px 12px 0'>"
                        f"<span class='muted'>{html.escape(view)}</span><br>"
                        f"<video src='{src}' controls muted loop playsinline "
                        f"width='320' preload='metadata'></video></span>"
                    )
                body += "</p>"
        else:
            body += "<p class='muted'>No local demonstration data for this task.</p>"
        body += "<p><a href='/gallery/robodojo'>← all tasks</a></p>"
        return page(f"RoboDojo · {selected.title}", body)

    playable = {
        entry["id"] for entry in payload["tasks"] if entry.get("demonstrations")
    }
    body += "<p class='muted'>● has playable demonstration video</p>"
    for dimension in payload["dimensions"]:
        body += f"<h2>{html.escape(dimension['label'])}</h2><p>"
        for task_id in dimension["tasks"]:
            mark = "● " if task_id in playable else "○ "
            body += (
                f"<a href='/gallery/robodojo/{html.escape(task_id)}'>"
                f"{mark}{html.escape(task_id)}</a> &nbsp; "
            )
        body += "</p>"
    return page("RoboDojo tasks", body)


@app.get("/api/robotwin/tasks")
def robotwin_catalog_api():
    return jsonify(catalog_payload())


@app.get("/gallery/robodojo/<task_id>/video/<episode>/<view>")
def robodojo_demo_video(task_id: str, episode: str, view: str) -> Response:
    """Serve one recorded demonstration view.

    Paths are resolved from the catalog's own record rather than built from the
    URL, so a crafted episode or view name cannot reach outside the dataset.
    """
    entry = robodojo_task(task_id)
    if entry is None:
        return Response("unknown RoboDojo task", status=404, mimetype="text/plain")
    for demo in entry.demonstrations:
        if demo["episode"] != episode:
            continue
        name = demo["views"].get(view)
        if name is None:
            break
        path = Path(demo["preview_root"]) / name
        if not path.is_file():
            break
        return send_file(path, mimetype="video/mp4", conditional=True)
    return Response("unknown demonstration view", status=404, mimetype="text/plain")


@app.get("/api/robodojo/tasks")
def robodojo_catalog_api():
    return jsonify(robodojo_catalog_payload())


@app.get("/gallery/robodojo")
@app.get("/gallery/robodojo/<task_id>")
def robodojo_gallery(task_id: str | None = None):
    return _robodojo_gallery(task_id)


app.register_blueprint(create_libero_pro_blueprint(
    page_renderer=page,
    preview_loader=lambda: load_preview_manifest(),
    task_runs_url=_libero_task_runs_url,
))


def raw(path: Path) -> str:
    return html.escape(path.read_text(encoding="utf-8", errors="replace")) if path.is_file() else "not available"


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _timeline(root: Path) -> str:
    labels = {
        "agent_session_step_started": "Codex turn",
        "tool_result": "tool call/result",
        "tool_error": "tool error",
        "action_submitted": "submitted action",
        "motion_receipt": "environment receipt",
        "observation": "fresh observation",
        "environment_status": "Control-only environment status",
        "agent_session_native_compaction": "native compaction",
        "stagnation_observation": "public image delta across the action",
        "action_rejected_stagnation": "legacy stagnation termination",
        "action_stagnation_advisory": "legacy stagnation advisory (action executed)",
    }
    selected = []
    for sequence, event in enumerate(load_jsonl(root / "events.jsonl")):
        kind = str(event.get("type", ""))
        if kind not in labels:
            continue
        timestamp = event.get("time", event.get("started", event.get("finished", 0)))
        selected.append((float(timestamp or 0), sequence, kind, event))
    selected.sort()
    rows = []
    for _timestamp, sequence, kind, event in selected:
        detail = {
            key: event[key] for key in (
                "call", "turn_id", "tool", "arguments", "action", "record",
                "observation_id", "error", "streak",
            ) if key in event
        }
        rows.append(
            f"<article><strong>{sequence:03d} · {html.escape(labels[kind])}</strong>"
            f"<pre>{html.escape(json.dumps(detail, ensure_ascii=False, indent=2, sort_keys=True))}</pre></article>"
        )
    return "<div class='timeline'>" + ("".join(rows) or "<p>no causal timeline</p>") + "</div>"


def _session_tables(session: dict) -> str:
    calls = session.get("calls", []) if isinstance(session.get("calls"), list) else []
    call_rows = []
    for call in calls:
        usage = call.get("usage") if isinstance(call.get("usage"), dict) else {}
        call_rows.append(
            "<tr>"
            f"<td>{html.escape(str(call.get('call', '—')))}</td>"
            f"<td>{html.escape(str(call.get('turn_id', '—')))}</td>"
            f"<td class='metric'>{html.escape(str(usage.get('input_tokens', 'missing')))}</td>"
            f"<td class='metric'>{html.escape(str(usage.get('cached_input_tokens', 'missing')))}</td>"
            f"<td class='metric'>{html.escape(str(usage.get('output_tokens', 'missing')))}</td>"
            f"<td class='metric'>{html.escape(str(call.get('project_message_bytes', '—')))}</td>"
            f"<td class='metric'>{html.escape(str(call.get('latency_s', '—')))}</td></tr>"
        )
    usage_rows = []
    for event in session.get("usage_events", []):
        total = event.get("total") if isinstance(event.get("total"), dict) else {}
        usage_rows.append(
            f"<tr><td>{html.escape(str(event.get('sequence', '—')))}</td>"
            f"<td>{html.escape(str(event.get('turn_id', '—')))}</td>"
            f"<td>{html.escape(str(total.get('input_tokens', 'missing')))}</td>"
            f"<td>{html.escape(str(total.get('cached_input_tokens', 'missing')))}</td>"
            f"<td>{html.escape(str(total.get('output_tokens', 'missing')))}</td></tr>"
        )
    result = (
        "<table><thead><tr><th>call</th><th>turn</th><th>input</th><th>cached</th>"
        "<th>output</th><th>Harness bytes</th><th>latency s</th></tr></thead><tbody>"
        + "".join(call_rows) + "</tbody></table>"
    )
    if usage_rows:
        result += (
            "<h3>Provider cumulative workload notifications</h3><table><thead><tr>"
            "<th>event</th><th>turn</th><th>total input</th><th>total cached</th>"
            "<th>total output</th></tr></thead><tbody>" + "".join(usage_rows) + "</tbody></table>"
        )
    return result


def _metric(value: object, suffix: str = "") -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        rendered = f"{value:,.3f}".rstrip("0").rstrip(".")
    elif isinstance(value, int):
        rendered = f"{value:,}"
    else:
        rendered = str(value)
    return html.escape(rendered + suffix)


def _metric_number(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _compact_metric(value: object) -> str:
    number = _metric_number(value)
    if number is None:
        return "—"
    absolute = abs(number)
    if absolute >= 1_000_000:
        return f"{number / 1_000_000:.2f}M"
    if absolute >= 1_000:
        return f"{number / 1_000:.1f}k"
    if number.is_integer():
        return f"{int(number):,}"
    return f"{number:.1f}"


def _cost_projection(root: Path) -> dict[str, object]:
    estimate = estimate_run_cost(root)
    cost = estimate.get("cost") if isinstance(estimate.get("cost"), dict) else {}
    tokens = (
        estimate.get("tokens") if isinstance(estimate.get("tokens"), dict) else {}
    )
    total_tokens = (
        tokens.get("total") if isinstance(tokens.get("total"), dict) else {}
    )
    auxiliary = (
        tokens.get("auxiliary")
        if isinstance(tokens.get("auxiliary"), dict) else {}
    )
    price = (
        estimate.get("price") if isinstance(estimate.get("price"), dict) else {}
    )
    return {
        "status": str(estimate.get("status") or "unavailable"),
        "measurement": str(estimate.get("measurement") or "missing"),
        "currency": str(cost.get("currency") or price.get("currency") or ""),
        "total_cost": _metric_number(cost.get("total")),
        "cached_cost": _metric_number(cost.get("cache_hit_input")),
        "uncached_cost": _metric_number(cost.get("cache_miss_input")),
        "output_cost": _metric_number(cost.get("output")),
        "input_tokens": _metric_number(total_tokens.get("input_tokens")),
        "cached_tokens": _metric_number(
            total_tokens.get("cache_read_input_tokens")
        ),
        "uncached_tokens": _metric_number(total_tokens.get("uncached_input_tokens")),
        "output_tokens": _metric_number(total_tokens.get("output_tokens")),
        "auxiliary_calls": _metric_number(auxiliary.get("calls")),
        "price_id": str(price.get("price_id") or ""),
        "source_url": str(price.get("source_url") or ""),
    }


def _money(value: object, currency: object = "CNY") -> str:
    number = _metric_number(value)
    if number is None:
        return "—"
    prefix = "¥" if str(currency) == "CNY" else f"{currency} "
    return f"{prefix}{number:,.2f}"


def _cost_cell(cost: dict[str, object]) -> str:
    if cost.get("total_cost") is None:
        return (
            "<div class='metric-missing'>cost unavailable</div>"
            f"<small>{html.escape(str(cost.get('status') or 'missing'))}</small>"
        )
    provisional = cost.get("status") == "provisional"
    status = "live provisional" if provisional else "sealed estimate"
    return (
        f"<div class='metric-primary{' provisional' if provisional else ''}>"
        f"{_money(cost.get('total_cost'), cost.get('currency'))}<span>list price</span></div>"
        f"<small>hit {_money(cost.get('cached_cost'), cost.get('currency'))} · "
        f"miss {_money(cost.get('uncached_cost'), cost.get('currency'))} · "
        f"out {_money(cost.get('output_cost'), cost.get('currency'))}</small>"
        f"<small>{html.escape(status)} · {int(cost.get('auxiliary_calls') or 0)} auxiliary calls</small>"
    )


def _comparison_metrics(root: Path) -> dict[str, object]:
    """Project one run's deterministic Runtime report for cohort comparison."""

    cost = _cost_projection(root)
    metrics = load(root / "runtime-metrics.json")
    if not metrics:
        return {"status": "missing", "cost": cost}
    recorded_run_id = metrics.get("run_id")
    if recorded_run_id is not None and recorded_run_id != root.name:
        return {"status": "identity_mismatch", "cost": cost}
    context = metrics.get("context") if isinstance(metrics.get("context"), dict) else {}
    request_input = (
        context.get("request_input_tokens")
        if isinstance(context.get("request_input_tokens"), dict) else {}
    )
    projection = (
        context.get("projection")
        if isinstance(context.get("projection"), dict) else {}
    )
    harness = (
        metrics.get("harness_input")
        if isinstance(metrics.get("harness_input"), dict) else {}
    )
    images = harness.get("images") if isinstance(harness.get("images"), dict) else {}
    image_per_request = (
        images.get("per_request")
        if isinstance(images.get("per_request"), dict) else {}
    )
    cache = metrics.get("cache") if isinstance(metrics.get("cache"), dict) else {}
    quality = (
        metrics.get("measurement_quality")
        if isinstance(metrics.get("measurement_quality"), dict) else {}
    )
    return {
        "status": str(metrics.get("measurement_status") or "reported"),
        "calls": _metric_number(context.get("exchange_count")),
        "context_mean": _metric_number(request_input.get("mean")),
        "context_p95": _metric_number(request_input.get("p95")),
        "context_peak": _metric_number(context.get("peak_request_input_tokens")),
        "context_window": _metric_number(context.get("model_context_window_tokens")),
        "context_utilization": _metric_number(
            context.get("peak_request_input_utilization")
        ),
        "context_measurement": str(context.get("measurement") or "missing"),
        "context_quality": str(quality.get("request_context") or "missing"),
        "context_profile": str(
            projection.get("context_retention")
            or projection.get("profile")
            or "not recorded"
        ),
        "epoch_transitions": _metric_number(projection.get("epoch_transition_count")),
        "images_mean": _metric_number(image_per_request.get("mean")),
        "images_p95": _metric_number(image_per_request.get("p95")),
        "images_peak": _metric_number(image_per_request.get("max")),
        "images_total": _metric_number(images.get("count")),
        "images_measurement": str(images.get("measurement") or "missing"),
        "images_quality": str(quality.get("image_inventory") or "missing"),
        "cache_token_ratio": _metric_number(cache.get("token_cache_read_ratio")),
        "cache_call_ratio": _metric_number(cache.get("call_cache_read_hit_rate")),
        "cache_read_tokens": _metric_number(cache.get("cache_read_input_tokens")),
        "cache_input_tokens": _metric_number(
            cache.get("input_tokens_with_cache_read_telemetry")
        ),
        "cache_measurement": str(cache.get("measurement") or "missing"),
        "cache_quality": str(quality.get("provider_cache_telemetry") or "missing"),
        "cost": cost,
    }


def _runtime_metric_cards(metrics: dict) -> str:
    if not metrics:
        return "<p>runtime-metrics.json has not been generated.</p>"
    context = metrics.get("context", {})
    workload = metrics.get("workload", {})
    harness = metrics.get("harness_input", {})
    text = harness.get("text", {})
    images = harness.get("images", {})
    latency = metrics.get("latency", {})
    turn_latency = latency.get("agent_turn_s", {})
    later_bytes = text.get("later_turn_utf8_bytes", {})
    utilization = context.get("peak_request_input_utilization")
    utilization_text = (
        _metric(utilization * 100, "%") if isinstance(utilization, (int, float))
        else "missing"
    )
    return (
        "<div class='cards'>"
        "<article class='card'><h3>Context occupancy</h3>"
        f"<p>peak input: <strong>{_metric(context.get('peak_request_input_tokens'))}</strong> / "
        f"{_metric(context.get('model_context_window_tokens'))} tokens</p>"
        f"<p>utilization: {utilization_text} · observed headroom: "
        f"{_metric(context.get('observed_input_headroom_tokens'))}</p></article>"
        "<article class='card'><h3>Provider workload</h3>"
        f"<p>input: <strong>{_metric(workload.get('input_tokens'))}</strong> · cached: "
        f"{_metric(workload.get('cached_input_tokens'))}</p>"
        f"<p>uncached: {_metric(workload.get('uncached_input_tokens'))} · output: "
        f"{_metric(workload.get('output_tokens'))}</p></article>"
        "<article class='card'><h3>Harness additions</h3>"
        f"<p>later-turn bytes p50 / p95: {_metric(later_bytes.get('p50'))} / "
        f"{_metric(later_bytes.get('p95'))}</p>"
        f"<p>image attachments: {_metric(images.get('count'))} · text/image tokens: opaque</p>"
        "</article>"
        "<article class='card'><h3>Latency</h3>"
        f"<p>episode: <strong>{_metric(latency.get('episode_wall_s'), ' s')}</strong></p>"
        f"<p>agent turn p50 / p95: {_metric(turn_latency.get('p50'), ' s')} / "
        f"{_metric(turn_latency.get('p95'), ' s')}</p></article>"
        "</div>"
    )


def _composition_page(run_id: str, inspection: dict) -> str:
    identity = inspection.get("identity", {})
    composition = inspection.get("composition", {})
    prompt = inspection.get("prompt", {})
    tool_surface = inspection.get("tool_surface", {})
    planner = inspection.get("planner", {})
    run_date, run_clock, _run_iso = _utc_parts(identity.get("started_unix_s"))
    safe_payload = json.dumps(inspection, ensure_ascii=False).replace("</", "<\\/")
    body = (
        "<nav class='nav'><a href='/'>← All runs</a>"
        f"<a href='/experiment/{html.escape(str(identity.get('experiment') or ''))}/configs'>Experiment configs</a>"
        f"<a href='/run/{html.escape(run_id)}'>Trajectory summary</a>"
        f"<a href='/run/{html.escape(run_id)}/trajectory/raw'>Raw trajectory</a>"
        f"<a href='/run/{html.escape(run_id)}/raw'>Evidence inventory</a>"
        f"<a class='raw-link' href='/api/run/{html.escape(run_id)}/composition'>Composition JSON</a></nav>"
        "<section class='composition-shell'>"
        "<header class='composition-head'><div><span class='eyebrow'>Exact recorded configuration</span>"
        f"<h1>{html.escape(str(identity.get('config') or 'Policy composition'))}</h1>"
        f"<p>{html.escape(str(identity.get('task_instruction') or identity.get('task') or ''))}</p></div>"
        f"<div class='composition-hash'>composition<br>{html.escape(str(composition.get('composition_sha256') or 'missing'))}</div></header>"
        "<div class='composition-facts'>"
        f"<div><span>Experiment ID</span><strong>{html.escape(str(identity.get('experiment') or '—'))}</strong></div>"
        f"<div><span>Exact run ID</span><strong>{html.escape(run_id)}</strong></div>"
        f"<div><span>Started</span><strong>{html.escape(run_date)} · {html.escape(run_clock)}</strong></div>"
        f"<div><span>Planner</span><strong>{html.escape(str(planner.get('kind') or '—'))} · {html.escape(str(planner.get('model') or '—'))}</strong></div>"
        f"<div><span>Prompt slots</span><strong>{len(prompt.get('slots', []))} · {len(prompt.get('sources', []))} source files</strong></div>"
        f"<div><span>Tool surface</span><strong>{html.escape(str(tool_surface.get('profile') or '—'))} · {len(tool_surface.get('definitions', []))} tools</strong></div>"
        "</div>"
        "<div class='composition-grid'><aside class='composition-index'>"
        "<section class='composition-group'><h2>Provider request order</h2><div id='prompt-index'></div></section>"
        "<section class='composition-group'><h2>Advertised tools</h2><div id='tool-index'></div></section>"
        "</aside><section class='composition-detail'>"
        "<header class='composition-detail-head'><div><h2 id='composition-title'>Select one entry</h2><p id='composition-subtitle'></p></div><div id='composition-badges' class='composition-badges'></div></header>"
        "<div id='composition-sources' class='composition-sources'></div>"
        "<pre id='composition-payload'>Select a prompt slot or tool from the left.</pre>"
        "</section></div></section>"
        f"<script id='composition-data' type='application/json'>{safe_payload}</script>"
        """<script>
const inspection=JSON.parse(document.getElementById('composition-data').textContent);
const promptIndex=document.getElementById('prompt-index'), toolIndex=document.getElementById('tool-index');
const title=document.getElementById('composition-title'), subtitle=document.getElementById('composition-subtitle'), badges=document.getElementById('composition-badges'), sources=document.getElementById('composition-sources'), payload=document.getElementById('composition-payload');
const node=(tag,cls,text)=>{const el=document.createElement(tag);if(cls)el.className=cls;if(text!==undefined)el.textContent=text;return el;};
const pretty=value=>JSON.stringify(value,null,2);
let active=null;
function itemButton(entry,index,type){const button=node('button','composition-item');button.type='button';button.dataset.key=type+':'+index;button.append(node('code','',type==='prompt'?String(entry.order).padStart(2,'0'):'T'+String(index).padStart(2,'0')));const body=node('span');body.append(node('strong','',type==='prompt'?entry.kind:entry.name),node('small','',type==='prompt'?([entry.role,entry.status].filter(Boolean).join(' · ')):entry.state_effect));button.append(body);button.onclick=()=>render(type,index);return button;}
function addBadges(values){badges.replaceChildren();values.filter(Boolean).forEach(value=>badges.append(node('span','composition-badge',String(value))));}
function addSources(values){sources.replaceChildren();(values||[]).forEach(source=>{const row=node('div','composition-source');row.append(node('span','',source.path||source.channel||'source'),node('span','',(source.bytes??'—')+' B · '+String(source.sha256||'').slice(0,12)));sources.append(row);});}
function render(type,index){document.querySelectorAll('.composition-item').forEach(button=>button.classList.toggle('active',button.dataset.key===type+':'+index));if(type==='prompt'){const entry=inspection.prompt.slots[index];title.textContent=entry.kind;subtitle.textContent=entry.note||entry.delivery||'Provider-visible context slot';addBadges([entry.role,entry.status,entry.sha256&&('sha '+entry.sha256.slice(0,12))]);addSources(entry.sources);if(entry.text)payload.textContent=entry.text;else if(entry.example)payload.textContent=pretty(entry.example);else payload.textContent=entry.status==='opaque'?'Owned by the managed runtime; exact text is not recorded.':'This slot is assembled dynamically at runtime. See its status, delivery and dynamic-call evidence above.';}else{const tool=inspection.tool_surface.definitions[index];title.textContent=tool.name;subtitle.textContent=tool.description||'No description recorded.';addBadges([tool.state_effect,inspection.tool_surface.profile]);addSources([]);payload.textContent=pretty({input_schema:tool.input_schema,result_schema:tool.result_schema});}}
(inspection.prompt.slots||[]).forEach((entry,index)=>promptIndex.append(itemButton(entry,index,'prompt')));(inspection.tool_surface.definitions||[]).forEach((entry,index)=>toolIndex.append(itemButton(entry,index,'tool')));if((inspection.prompt.slots||[]).length)render('prompt',0);
</script>"""
    )
    return page(f"Composition · {run_id}", body, wide=True)


def _trajectory_page(
    run_id: str, trajectory: dict, *, video_pending: bool = False,
) -> str:
    summary = trajectory.get("summary", {})
    session_rounds = (
        trajectory.get("session_rounds")
        if isinstance(trajectory.get("session_rounds"), dict) else None
    )
    verdict = summary.get("official_success")
    if session_rounds is not None:
        successes = session_rounds.get("round_success_count", 0)
        round_count = session_rounds.get("round_count", 0)
        verdict_label = f"{successes}/{round_count} rounds"
        verdict_class = (
            "success" if session_rounds.get("all_rounds_succeeded") is True
            else "failure" if successes == 0 else ""
        )
    else:
        verdict_label = "unscored" if verdict is None else ("success" if verdict else "failure")
        verdict_class = "" if verdict is None else verdict_label
    control_validity = str(summary.get("control_validity") or "indeterminate")
    control_issues = summary.get("control_issues")
    control_issues = control_issues if isinstance(control_issues, list) else []
    control_note = (
        "No control-contract issue was found by the v2 receipt audit."
        if control_validity == "control-valid"
        else "Control audit: " + ", ".join(str(item) for item in control_issues)
        if control_issues
        else "Control validity could not be established from this evidence."
    )
    run_date, run_clock, _run_iso = _utc_parts(summary.get("run_started_unix_s"))
    safe_payload = json.dumps(trajectory, ensure_ascii=False).replace("</", "<\\/")
    title = trajectory.get("instruction") or trajectory.get("task") or run_id
    task_identity = trajectory.get("task") or trajectory.get("config") or "task unknown"
    notebook_kind = (
        "Persistent K-session notebook" if session_rounds is not None
        else "Episode notebook"
    )
    video_source = (
        trajectory.get("video_source")
        if isinstance(trajectory.get("video_source"), dict) else {}
    )
    continuous_execution = video_source.get("continuous_execution") is True
    source_kind = video_source.get("kind")
    reel_caption = (
        "derived controller replay · same source and seed · synchronized cameras"
        if source_kind == "viewer_derived_controller_replay"
        else "Original execution capture · simulation clock · idle policy time omitted"
        if continuous_execution
        else "Original observations · 1.4 s / snapshot · intervening motion not recorded"
    )
    if video_source.get("capture_status") == "incomplete":
        reel_caption += " · incomplete recording; frame gaps are marked"
    reel_media = render_process_player(run_id)
    session_ledger = ""
    if session_rounds is not None:
        round_cards = []
        steps = trajectory.get("steps") if isinstance(trajectory.get("steps"), list) else []
        state_to_index = {
            step.get("state_step"): index for index, step in enumerate(steps)
            if isinstance(step, dict) and isinstance(step.get("state_step"), int)
        }
        for item in session_rounds.get("rounds", []):
            if not isinstance(item, dict):
                continue
            ordinal = item.get("round_ordinal", "—")
            official = item.get("official_success")
            outcome = "SUCCESS" if official is True else "FAILURE" if official is False else "UNSCORED"
            outcome_class = "success" if official is True else "failure" if official is False else ""
            bounds = item.get("policy_state_steps") if isinstance(item.get("policy_state_steps"), dict) else {}
            start = bounds.get("start")
            end = bounds.get("end_exclusive")
            anchor = state_to_index.get(start, 0)
            duration = item.get("wall_duration_s")
            duration_text = f"{float(duration):.1f}s" if isinstance(duration, (int, float)) else "—"
            range_text = (
                f"state steps {start}–{end - 1}" if isinstance(start, int) and isinstance(end, int)
                else "step range unavailable"
            )
            round_cards.append(
                f"<a class='round-card' href='#step-{anchor}' data-round='{html.escape(str(ordinal))}'>"
                f"<span>Round {html.escape(str(ordinal))}</span><strong class='{outcome_class}'>{outcome}</strong>"
                "<dl class='round-metrics'>"
                f"<div><dt>decisions</dt><dd>{html.escape(str(item.get('decisions', '—')))}</dd></div>"
                f"<div><dt>calls</dt><dd>{html.escape(str(item.get('model_calls', '—')))}</dd></div>"
                f"<div><dt>wall</dt><dd>{html.escape(duration_text)}</dd></div></dl>"
                f"<div class='round-step-range'>{html.escape(range_text)}</div></a>"
            )
        integrity = session_rounds.get("integrity") if isinstance(session_rounds.get("integrity"), dict) else {}
        session_ledger = (
            "<section class='session-ledger'><header class='session-contract'><div>"
            "<span class='episode-kicker'>Session topology · round verdicts remain separate</span>"
            f"<h2>One AgentSession, {html.escape(str(session_rounds.get('round_count', '—')))} reset EnvironmentSessions.</h2>"
            "<p>The transcript, reasoning history and compacted context continue across rounds. "
            "The Supervisor resets the environment after success or abandonment. This is "
            "<strong>success-within-K</strong>, not standard Pass@K; aggregate <code>official_success</code> intentionally remains null.</p>"
            "</div><div class='session-identity'>"
            f"{html.escape(str(session_rounds.get('cadence') or 'cadence unknown'))}<br>"
            f"budget · {html.escape(str(session_rounds.get('budget_scope') or 'unknown'))}<br>"
            f"projection · {html.escape(str(integrity.get('status') or 'unknown'))}</div></header>"
            f"<div class='round-grid'>{''.join(round_cards)}</div></section>"
        )
    body = (
        "<nav class='nav'><a href='/'>← All runs</a>"
        f"<a href='/experiment/{html.escape(str(trajectory.get('experiment')))}/runs'>Experiment runs</a>"
        f"<a class='raw-link' href='/run/{html.escape(run_id)}/composition'>Composition</a>"
        f"<a class='raw-link' href='/run/{html.escape(run_id)}/trajectory/raw'>Raw trajectory</a>"
        f"<a class='raw-link' href='/run/{html.escape(run_id)}/raw'>Raw evidence</a>"
        f"<a class='raw-link' href='/api/run/{html.escape(run_id)}/trajectory'>Trajectory JSON</a></nav>"
        "<section class='episode-notebook'>"
        f"<header class='episode-mast'><div><span class='episode-kicker'>{html.escape(notebook_kind)} · read-only projection</span>"
        f"<h1>{html.escape(str(title))}</h1>"
        f"<p>{html.escape(str(task_identity))} · {html.escape(run_id)}</p></div>"
        f"<div class='verdict-stack'><div class='verdict-stamp {verdict_class}'>behavior · {html.escape(verdict_label)}</div>"
        f"<div class='control-validity {html.escape(control_validity)}'>control · {html.escape(control_validity)}</div></div></header>"
        "<div class='episode-meta'>"
        f"<div><span>Experiment ID</span><strong>{html.escape(str(trajectory.get('experiment') or '—'))}</strong></div>"
        f"<div><span>Config</span><strong>{html.escape(str(trajectory.get('config') or '—'))}</strong></div>"
        f"<div><span>Started</span><strong>{html.escape(run_date)} · {html.escape(run_clock)}</strong></div>"
        f"<div><span>Exact run ID</span><strong>{html.escape(run_id)}</strong></div>"
        f"<div><span>Source</span><strong>{html.escape(str(trajectory.get('source_commit') or 'unknown')[:12])}</strong></div>"
        "</div>"
        "<div class='episode-statbar'>"
        f"<div class='episode-stat'><span>robot decisions</span><strong>{html.escape(str(summary.get('decisions', '—')))}</strong></div>"
        f"<div class='episode-stat'><span>model calls</span><strong>{html.escape(str(summary.get('model_calls', '—')))}</strong></div>"
        f"<div class='episode-stat'><span>causal steps</span><strong>{len(trajectory.get('steps', []))}</strong></div>"
        f"<div class='episode-stat'><span>wall duration</span><strong>{html.escape(str(round(summary.get('wall_duration_s', 0), 1) if isinstance(summary.get('wall_duration_s'), (int, float)) else '—'))}s</strong></div>"
        f"<div class='episode-stat'><span>termination</span><strong>{html.escape(str(summary.get('termination', '—')))}</strong></div>"
        "</div>"
        f"{session_ledger}"
        "<section class='episode-overview'>"
        f"{reel_media}"
        "</section>"
        "<section id='round-board' class='round-board' hidden></section>"
        "<section id='investigation-board' class='investigation-board' hidden></section>"
        "<div class='review-controls'><button id='expand-all' type='button'>Expand all</button><button id='collapse-all' type='button'>Collapse all</button>"
        "<span><kbd>J</kbd>/<kbd>K</kbd> move between steps · click an image for the original artifact</span></div>"
        "<div class='episode-workspace'><aside class='episode-outline'><h2>Episode outline</h2><nav id='outline-list' class='outline-list'></nav></aside>"
        "<section id='step-feed' class='step-feed' aria-label='Causal ActionStep timeline'></section></div></section>"
        f"<script id='trajectory-data' type='application/json'>{safe_payload}</script>"
        """<script>
	const trajectory = JSON.parse(document.getElementById('trajectory-data').textContent);
	const analysis = trajectory.analysis || null;
	const incidents = analysis?.incidents || [];
const node = (tag, cls, text) => { const el=document.createElement(tag); if(cls) el.className=cls; if(text!==undefined) el.textContent=text; return el; };
const pretty = value => JSON.stringify(value, null, 2);
const artifactUrl = path => '/artifact/' + encodeURIComponent(trajectory.run_id) + '/' + path.split('/').map(encodeURIComponent).join('/');
const truncate = (value, width=260) => { const text=String(value ?? '').replace(/\\s+/g,' ').trim(); return text.length>width ? text.slice(0,width-1)+'…' : text; };
	const outline=document.getElementById('outline-list'), feed=document.getElementById('step-feed');
	const incidentsForStep=step=>incidents.filter(item=>(item.steps||[]).includes(step.state_step));
	const toneForStep=step=>{const tones=incidentsForStep(step).flatMap(item=>(item.stages||[]).filter(stage=>stage.step===step.state_step).map(stage=>stage.tone||item.tone));return tones.includes('critical')?'critical':tones.includes('warning')?'warning':tones.includes('recovery')?'recovery':null;};
	const jumpStep=stateStep=>{const index=trajectory.steps.findIndex(step=>step.state_step===stateStep);const card=document.getElementById('step-'+index);if(card){card.open=true;card.scrollIntoView({behavior:'smooth',block:'start'});window.dispatchEvent(new CustomEvent('viewer-step-selected',{detail:{stateStep}}));}};
const actionLabel=step=>step.action?.kind || step.tools?.[0]?.tool || 'observe / reason';
const messagesFor=step=>step.model_calls.flatMap(call=>call.messages || []).filter(Boolean);
const previewFor=step=>truncate(messagesFor(step)[0] || step.model_calls.find(call=>call.reasoning)?.reasoning || (step.action ? 'World mutation proposed and executed through the Runtime.' : 'No public textual response recorded for this step.'),160);
function sectionHeading(title,note){ const head=node('header','section-heading'); head.append(node('h3','',title),node('span','',note)); return head; }
function frame(frame,phase){
  const link=node('a','evidence-frame'); link.href=artifactUrl(frame.path); link.target='_blank'; link.rel='noopener';
  const fig=node('figure'); const img=node('img'); img.src=link.href; img.loading='lazy'; img.alt=[phase,frame.label,frame.visibility].filter(Boolean).join(' · ');
  const caption=node('figcaption'); caption.append(node('span','frame-chip',phase+' · '+(frame.label||'artifact')));
  if(frame.kind && frame.kind!=='rgb') caption.append(node('span','frame-chip',frame.kind));
  const visibility=frame.visibility||'visibility_unrecorded'; caption.append(node('span','frame-chip'+(visibility==='control_only'?' control':''),visibility.replaceAll('_',' ')));
  fig.append(img,caption); link.append(fig); return link;
}
function imageGrid(frames,phase){ const grid=node('div','evidence-grid'); if(!frames.length){grid.append(node('div','empty-evidence','No renderable visual artifact recorded for this phase.'));return grid;} frames.forEach(item=>grid.append(frame(item,phase))); return grid; }
function payloadDrawer(label,value){ const details=node('details','payload-drawer'); details.append(node('summary','',label),node('pre','',pretty(value))); return details; }
function compact(value){ if(value===null||value===undefined)return 'not recorded'; const text=JSON.stringify(value); return truncate(text,220); }
function executionCard(number,title,status,summary,payload,bad=false){
  const card=node('article','execution-card'); const head=node('header','execution-head'); head.append(node('span','execution-seq',String(number).padStart(2,'0')),node('strong','',title));
  head.append(node('span','execution-status'+(bad?' bad':''),status)); card.append(head,node('div','execution-summary',summary),payloadDrawer('Exact projected payload',payload)); return card;
}
function modelResponse(call,index){
  const card=node('article','model-response'); card.append(node('div','response-mark','M'+(index+1))); const body=node('div');
  const meta=node('div','response-meta'); meta.append(node('span','',String(call.message_kind||'model response').replaceAll('_',' ')),node('span','',call.latency_s==null?'latency not recorded':Number(call.latency_s).toFixed(2)+' s'));
  if(call.call!==null&&call.call!==undefined) meta.append(node('span','','call '+call.call)); body.append(meta);
  const messages=(call.messages||[]).filter(Boolean); messages.forEach(message=>body.append(node('p','assistant-message',message)));
  if(!messages.length && !call.reasoning) body.append(node('p','assistant-message','No textual model response was recorded.'));
  if(call.reasoning){ const reasoning=node('details','reasoning-block'); reasoning.append(node('summary','','API-visible reasoning (recorded, collapsed)'),node('pre','',call.reasoning)); body.append(reasoning); }
  if(call.usage) body.append(payloadDrawer('Recorded usage',call.usage)); card.append(body); return card;
}
	function renderStep(step,index){
	  const warnings=[...(step.warnings||[])]; if(step.receipt?.reached===false) warnings.push('receipt_not_reached');
	  const incidentTone=toneForStep(step); const stepIncidents=incidentsForStep(step);
	  const details=node('details','episode-step'+(warnings.length?' has-warning':'')+(incidentTone?' incident-'+incidentTone:'')); details.id='step-'+index; details.open=true;
  const displayIndex=step.round ? ('R'+step.round.round_ordinal+'·'+String(step.round.local_step_index).padStart(2,'0')) : String(index).padStart(2,'0');
  const summary=node('summary','step-summary'); summary.append(node('span','step-number',displayIndex));
  const heading=node('span'); heading.append(node('strong','step-title',actionLabel(step)),node('span','step-preview',previewFor(step))); summary.append(heading);
  const badges=node('span','step-badges'); badges.append(node('span','step-badge','models '+step.model_calls.length),node('span','step-badge','tools '+step.tools.length));
	  if(step.round) badges.append(node('span','step-badge round','round '+step.round.round_ordinal+'/'+step.round.round_count)); else if(step.round_index!==null&&step.round_index!==undefined) badges.append(node('span','step-badge round','round '+(step.round_index+1))); if(step.action) badges.append(node('span','step-badge mutation','world mutation')); if(warnings.length) badges.append(node('span','step-badge warning','warning '+warnings.length)); stepIncidents.forEach(item=>badges.append(node('span','step-badge incident',item.id))); summary.append(badges); details.append(summary);
  const body=node('div','step-body');
  const visual=node('section','step-section'); visual.append(sectionHeading('Observation & visual artifacts',(step.observation_id||'observation id missing')+' · visibility is explicit')); visual.append(imageGrid(step.images||[],'current')); body.append(visual);
  const response=node('section','step-section'); response.append(sectionHeading('Model response',step.model_calls.length+' recorded response'+(step.model_calls.length===1?'':'s'))); const responses=node('div','response-stack');
  if(step.model_calls.length) step.model_calls.forEach((call,i)=>responses.append(modelResponse(call,i))); else responses.append(node('div','empty-evidence','No model response was associated with this causal step.')); response.append(responses); body.append(response);
  const execution=node('section','step-section'); execution.append(sectionHeading('Execution details',step.tools.length+' policy-visible tool call'+(step.tools.length===1?'':'s'))); const stack=node('div','execution-stack'); let seq=1;
  step.tools.forEach(tool=>stack.append(executionCard(seq++,tool.tool||'unknown tool',tool.ok?'returned':'error',compact(tool.arguments),{arguments:tool.arguments,result:tool.result},!tool.ok)));
  if(step.action||step.receipt){ const pair=node('div','execution-pair');
    pair.append(executionCard(seq++,'Runtime action',step.action?'submitted':'missing',step.action?compact({kind:step.action.kind,position_m:step.action.position_m,gripper:step.action.gripper}):'No world mutation was recorded.',step.action));
    const receiptStatus=step.receipt?.reached===false?'not reached':step.receipt?.reached===true?'reached':(step.receipt?.status||'recorded');
	    pair.append(executionCard(seq++,'Environment receipt',receiptStatus,step.receipt?compact({reached:step.receipt.reached,pose_residual:step.receipt.pose_residual,aperture_mm:step.receipt.aperture_mm,control_health:step.receipt.control_health,phases:step.receipt.phases}):'No receipt was recorded.',step.receipt,step.receipt?.reached===false)); stack.append(pair);
  }
  if(!step.tools.length&&!step.action&&!step.receipt) stack.append(node('div','empty-evidence','No tool call, world mutation or environment receipt was recorded in this step.'));
  if((step.after_images||[]).length){ const post=node('details','post-observation'); post.append(node('summary','','Post-action camera projection · '+step.after_images.length+' frames (collapsed to avoid duplicate imagery)'),imageGrid(step.after_images,'after')); stack.append(post); }
  execution.append(stack); body.append(execution);
  if(warnings.length||step.stagnation){ const warning=node('section','step-section'); warning.append(sectionHeading('Recorded warnings','no semantic inference')); warning.append(node('div','warning-card',pretty({warnings,stagnation:step.stagnation}))); body.append(warning); }
  const technical=node('footer','step-section technical-line'); const remaining=step.remaining_budget||{}; technical.append(node('span','','state step '+(step.state_step??'—')),node('span','','cause '+(step.cause||'—')),node('span','','robot decisions left '+(remaining.robot_decisions??'—'))); if(step.round) technical.append(node('span','','persistent AgentSession · environment round '+step.round.round_ordinal)); body.append(technical);
	  details.append(body); return details;
	}
	function renderRoundBoard(){
	  const root=document.getElementById('round-board'); const rounds=trajectory.rounds||[]; if(!rounds.length)return; root.hidden=false;
	  const head=node('div','round-board-head'); const left=node('div'); left.append(node('span','episode-kicker','Persistent session · environment resets'),node('h2','',rounds.length+' rounds in one AgentSession')); head.append(left,node('p','', 'Each card uses the recorded round boundary and official per-round verdict. Click a round to jump to its first actual observation.')); root.append(head);
	  const strip=node('div','round-strip'); rounds.forEach(round=>{const note=(analysis?.round_notes||[]).find(item=>item.round_index===round.round_index)||{};const card=node('button','round-card '+(note.tone||''));card.type='button';const header=node('header');header.append(node('span','', 'Round '+(round.round_index+1)),node('span','',round.official_success===true?'official success':round.official_success===false?'official failure':'unscored'));card.append(header,node('strong','',note.label||round.termination||'Round result'),node('p','',note.note||'No curated round note.'));const bounds=round.policy_state_steps||{};card.append(node('footer','',(round.decisions??'—')+' decisions · '+(round.model_calls??'—')+' calls · steps '+(bounds.start??'—')+'–'+((bounds.end_exclusive??1)-1)));card.onclick=()=>jumpStep(bounds.start);strip.append(card);});root.append(strip);
	}
	function renderInvestigation(){
	  const root=document.getElementById('investigation-board'); if(!analysis)return; root.hidden=false;
	  const head=node('div','investigation-head');const left=node('div');left.append(node('span','episode-kicker','Curated incident review · '+String(analysis.status||'curated').replaceAll('_',' ')),node('h2','',analysis.headline),node('p','',(analysis.title||'Run analysis')+' · '+(analysis.source_path||'tracked source')));head.append(left,node('p','analysis-boundary',analysis.boundary));root.append(head);
	  const metrics=node('div','analysis-metrics');(analysis.metrics||[]).forEach(item=>{const card=node('div','analysis-metric '+(item.tone||''));card.append(node('span','',item.label),node('strong','',item.value));metrics.append(card);});root.append(metrics);
	  const grid=node('div','finding-grid');(analysis.findings||[]).forEach(item=>{const card=node('article','finding-card '+(item.tone||''));const header=node('header');header.append(node('span','finding-id',item.id),node('span','finding-severity',item.severity));card.append(header,node('h3','',item.title),node('p','',item.summary),node('small','',item.evidence));const links=node('div','finding-links');(item.step_refs||[]).forEach(step=>{const button=node('button','', 'S'+String(step).padStart(2,'0'));button.type='button';button.onclick=()=>jumpStep(step);links.append(button);});card.append(links);grid.append(card);});root.append(grid);
	}
	renderRoundBoard(); renderInvestigation();
function roundDivider(round){
  const box=node('header','round-divider'); box.id='round-'+round.round_ordinal;
  const verdict=round.official_success===true?'official success':round.official_success===false?'official failure':'official verdict missing';
  box.append(node('span','','Round '+round.round_ordinal+' / '+trajectory.session_rounds.round_count),node('strong','',verdict),node('small','',round.decisions+' decisions · '+round.model_calls+' model calls · Supervisor reset follows; AgentSession continues'));
  return box;
}
const roundByOrdinal=new Map((trajectory.session_rounds?.rounds||[]).map(round=>[round.round_ordinal,round])); let lastRound=null;
trajectory.steps.forEach((step,index)=>{
  if(step.round && step.round.round_ordinal!==lastRound){ const round=roundByOrdinal.get(step.round.round_ordinal); outline.append(node('div','outline-round','Round '+step.round.round_ordinal)); if(round) feed.append(roundDivider(round)); lastRound=step.round.round_ordinal; }
  const link=node('a','outline-item'); link.href='#step-'+index; link.dataset.index=index; const shortIndex=step.round?('R'+step.round.round_ordinal+'.'+String(step.round.local_step_index).padStart(2,'0')):String(index).padStart(2,'0'); link.append(node('b','',shortIndex),node('span','',actionLabel(step))); outline.append(link); feed.append(renderStep(step,index));
});
const cards=[...document.querySelectorAll('.episode-step')], links=[...document.querySelectorAll('.outline-item')];
document.getElementById('expand-all').onclick=()=>cards.forEach(card=>card.open=true); document.getElementById('collapse-all').onclick=()=>cards.forEach(card=>card.open=false);
let active=0; const activate=index=>{active=Math.max(0,Math.min(index,cards.length-1));links.forEach((link,i)=>link.classList.toggle('active',i===active));};
if(cards.length){activate(0); const observer=new IntersectionObserver(entries=>{ const visible=entries.filter(entry=>entry.isIntersecting).sort((a,b)=>a.boundingClientRect.top-b.boundingClientRect.top); if(visible.length)activate(cards.indexOf(visible[0].target));},{rootMargin:'-12% 0px -72% 0px'}); cards.forEach(card=>observer.observe(card));}
document.addEventListener('keydown',event=>{if(event.target.matches('input,textarea,select'))return;if(event.key.toLowerCase()==='j'||event.key.toLowerCase()==='k'){event.preventDefault();const delta=event.key.toLowerCase()==='j'?1:-1;const target=Math.max(0,Math.min(active+delta,cards.length-1));cards[target]?.scrollIntoView({behavior:'smooth',block:'start'});activate(target);}});
</script>"""
    )
    return page(f"Trajectory · {run_id}", body, wide=True)


def _raw_trajectory_page(run_id: str, trace: dict) -> str:
    summary = trace.get("summary", {})
    sources = trace.get("sources", [])
    stream_counts = {
        item.get("stream"): item.get("line_count", 0)
        for item in sources if isinstance(item, dict)
    }
    start = summary.get("start_unix_s")
    end = summary.get("end_unix_s")
    duration = end - start if isinstance(start, (int, float)) and isinstance(end, (int, float)) else None
    safe_payload = json.dumps(trace, ensure_ascii=False).replace("</", "<\\/")
    verdict = summary.get("official_success")
    verdict_text = "unscored" if verdict is None else ("success" if verdict else "failure")
    run_date, run_clock, _run_iso = _utc_parts(
        summary.get("run_started_unix_s") or start
    )
    body = (
        "<nav class='nav'><a href='/'>← All runs</a>"
        f"<a href='/run/{html.escape(run_id)}'>Trajectory summary</a>"
        f"<a href='/run/{html.escape(run_id)}/composition'>Composition</a>"
        f"<a href='/run/{html.escape(run_id)}/raw'>Evidence inventory</a>"
        f"<a class='raw-link' href='/api/run/{html.escape(run_id)}/trajectory/raw'>Raw trajectory API</a></nav>"
        "<section class='trace-shell'>"
        "<header class='trace-mast'><div><span class='eyebrow'>Raw trajectory</span>"
        f"<h1>{html.escape(str(trace.get('task') or trace.get('config') or run_id))}</h1>"
        f"<p>{html.escape(str(trace.get('instruction') or run_id))}</p></div>"
        "<div class='trace-boundary'>Read-only timestamp merge. The exact source path, "
        "line number, line hash and stream-local order remain authoritative.</div></header>"
        "<div class='trace-meta'>"
        f"<div><span>Experiment ID</span><strong>{html.escape(str(trace.get('experiment') or '—'))}</strong></div>"
        f"<div><span>Config</span><strong>{html.escape(str(trace.get('config') or '—'))}</strong></div>"
        f"<div><span>Started</span><strong>{html.escape(run_date)} · {html.escape(run_clock)}</strong></div>"
        f"<div><span>Exact run ID</span><strong>{html.escape(run_id)}</strong></div>"
        f"<div><span>Source</span><strong>{html.escape(str(trace.get('source_commit') or 'unknown')[:12])}</strong></div>"
        "</div>"
        "<div class='trace-stats'>"
        f"<div class='trace-stat'><span>events</span><strong>{html.escape(str(summary.get('event_count', 0)))}</strong></div>"
        f"<div class='trace-stat'><span>actual cycles</span><strong>{html.escape(str(summary.get('observation_cycles', 0)))}</strong></div>"
        f"<div class='trace-stat'><span>runtime / agent</span><strong>{stream_counts.get('runtime', 0)} / {stream_counts.get('agent', 0)}</strong></div>"
        f"<div class='trace-stat'><span>recorded span</span><strong>{'—' if duration is None else f'{duration:.1f}s'}</strong></div>"
        f"<div class='trace-stat'><span>official</span><strong>{html.escape(verdict_text)}</strong></div>"
        "</div>"
        "<div id='incident-strip' class='incident-strip' hidden></div>"
        "<div class='trace-toolbar'>"
        "<input id='trace-search' type='search' autocomplete='off' placeholder='Filter type, IDs, tool args, visible text  /'>"
        "<div id='stream-filters' class='stream-filters'>"
        "<button class='stream-filter active' data-stream='all'>All</button>"
        "<button class='stream-filter' data-stream='runtime'>Runtime</button>"
        "<button class='stream-filter' data-stream='agent'>Agent</button>"
        "<button class='stream-filter' data-stream='transcript'>Transcript</button></div>"
        "<select id='round-filter' aria-label='Session round'><option value='all'>All rounds</option><option value='prelude'>Prelude</option></select>"
        "<select id='step-filter' aria-label='Observation cycle'><option value='all'>All cycles</option><option value='prelude'>Prelude</option></select>"
        "<select id='kind-filter' aria-label='Event family'><option value='all'>All event families</option><option value='observation'>Observations</option><option value='model'>Model I/O</option><option value='tool'>Tools</option><option value='action'>Actions / receipts</option><option value='error'>Errors / rejects</option></select>"
        "<span id='trace-count' class='trace-count'></span></div>"
        "<div class='review-lenses'><span>Review lens</span><button class='review-lens active' data-lens='all'>All evidence</button><button class='review-lens' data-lens='query'>Point / Mask</button><button class='review-lens' data-lens='control'>Control deviations</button><button class='review-lens' data-lens='context'>Context checkpoints</button><button class='review-lens' data-lens='task'>Task checks / verdicts</button></div>"
        "<div class='trace-grid'>"
        "<aside class='event-browser'><header class='trace-pane-head'><h2>Original event lines</h2><span>merged by recorded Unix time</span></header><div id='event-list' class='event-list'></div><button id='load-events' class='load-events'>Load more</button></aside>"
        "<section class='event-inspector'><header class='trace-pane-head'><h2>Exact source event</h2><span id='event-source'>select an event</span></header>"
        "<div class='event-identity'><h2 id='event-title'>No event selected</h2><div id='event-chips' class='event-chips'></div></div>"
        "<div class='event-tabs'><button class='event-tab active' data-tab='exact'>Exact JSONL line</button><button class='event-tab' data-tab='structured'>Parsed structure</button><button id='toggle-wrap' class='event-tab'>Wrap</button><button id='copy-event' class='trace-action'>Copy raw</button></div>"
        "<div class='event-payload'><pre id='event-payload' class='exact'>Select a source event from the left rail.</pre><div id='artifact-strip' class='artifact-strip'></div></div></section>"
        "<aside class='cycle-context'><header class='trace-pane-head'><h2>Observation context</h2><span id='cycle-label'>prelude</span></header><div id='cycle-scroll' class='cycle-scroll'><div class='cycle-empty'>This event is not bound to an observation cycle.</div></div></aside>"
        "</div>"
        "<footer class='trace-legend'><span>Center payload is the original JSONL source line. Right-side cycle context is a labeled navigation projection.</span>"
        "<span class='keycaps'><span><kbd>J</kbd>/<kbd>K</kbd> event</span><span><kbd>/</kbd> search</span><span><kbd>C</kbd> copy</span><span><kbd>Esc</kbd> clear</span></span></footer>"
        "</section>"
        f"<script id='raw-trajectory-data' type='application/json'>{safe_payload}</script>"
        """<script>
	const trace = JSON.parse(document.getElementById('raw-trajectory-data').textContent);
	const events = trace.events || [];
	const analysis = trace.analysis || null;
	const incidents = analysis?.incidents || [];
const eventByKey = new Map(events.map(event => [event.key, event]));
const list = document.getElementById('event-list');
	const search = document.getElementById('trace-search');
	const roundFilter = document.getElementById('round-filter');
	const stepFilter = document.getElementById('step-filter');
const kindFilter = document.getElementById('kind-filter');
const count = document.getElementById('trace-count');
const load = document.getElementById('load-events');
	let stream = 'all', lens = 'all', visibleLimit = 220, selectedKey = null, selectedDetail = null, selectedIncident = null, tab = 'exact', wrap = false;
const node = (tag, cls, text) => { const el=document.createElement(tag); if(cls) el.className=cls; if(text!==undefined) el.textContent=text; return el; };
const pretty = value => JSON.stringify(value, null, 2);
const artifactUrl = path => '/artifact/' + encodeURIComponent(trace.run_id) + '/' + path.split('/').map(encodeURIComponent).join('/');
const start = trace.summary.start_unix_s;
	const relativeTime = value => typeof value === 'number' && typeof start === 'number' ? '+'+(value-start).toFixed(3)+'s' : 'time —';
	const incidentsForStep = step => incidents.filter(item=>(item.steps||[]).includes(step));
	const stageTone = (incident,step) => (incident.stages||[]).find(stage=>stage.step===step)?.tone || incident.tone || 'warning';
	const incidentTone = step => {const tones=incidentsForStep(step).map(item=>stageTone(item,step));return tones.includes('critical')?'critical':tones.includes('warning')?'warning':tones.includes('recovery')?'recovery':null;};
const eventFamily = event => {
  const type=(event.type || '').toLowerCase();
  if(event.error) return 'error';
  if(type.includes('observation') || type.includes('reset')) return 'observation';
  if(type.includes('provider') || type.includes('model') || type.includes('context') || type.includes('session_step')) return 'model';
  if(type.includes('tool') || type.includes('openeta_view')) return 'tool';
  if(type.includes('action') || type.includes('receipt') || type.includes('motion')) return 'action';
  return 'other';
};
	const filtered = () => {
	  const query=search.value.trim().toLowerCase(); const round=roundFilter.value; const step=stepFilter.value; const family=kindFilter.value;
	  return events.filter(event => {
	    if(stream !== 'all' && event.stream !== stream) return false;
	    if(round === 'prelude' && event.round_index !== null) return false;
	    if(round !== 'all' && round !== 'prelude' && event.round_index !== Number(round)) return false;
	    if(step === 'prelude' && event.observation_step !== null) return false;
	    if(step !== 'all' && step !== 'prelude' && event.observation_step !== Number(step)) return false;
	    if(family !== 'all' && eventFamily(event) !== family) return false;
	    const haystack=(event.type+' '+event.summary+' '+event.search_text+' '+JSON.stringify(event.linkage)).toLowerCase();
	    if(lens==='query' && !/(vision\\.segment|mark_point|mask|segmentation|point_uv)/.test(haystack)) return false;
	    if(lens==='control' && !/(motion_receipt|servo_target_not_reached|control_health|action_submitted|not_reached)/.test(haystack)) return false;
	    if(lens==='context' && !/(context_projected|checkpoint|compaction|semantic_transition)/.test(haystack)) return false;
	    if(lens==='task' && !/(check_task|official_success|verdict|finish_episode|round_|environment_status)/.test(haystack)) return false;
	    if(query && !haystack.includes(query)) return false;
	    return true;
  });
};
function renderList(reset=false) {
  if(reset) visibleLimit=220;
  const matches=filtered(); list.replaceChildren();
  matches.slice(0,visibleLimit).forEach(event => {
	    const tone=incidentTone(event.observation_step); const button=node('button','event-row'+(event.error?' error':'')+(tone?' incident-'+tone:'')+(event.key===selectedKey?' active':'')); button.type='button'; button.dataset.key=event.key;
    button.append(node('span','event-seq',String(event.index).padStart(4,'0')));
    const main=node('span','event-main'); const title=node('span','event-type');
    const type=node('span',''); type.append(node('i','stream-mark '+event.stream),document.createTextNode(event.type));
	    title.append(type,node('span','step-stamp',(event.round_index===null?'PRE':'R'+(event.round_index+1))+' · '+(event.observation_step===null?'PRE':'S'+String(event.observation_step).padStart(2,'0'))+' · '+relativeTime(event.timestamp_unix_s)));
    main.append(title,node('span','event-summary',event.summary || event.source_path+':'+event.source_line)); button.append(main);
    button.onclick=()=>selectEvent(event.key); list.append(button);
  });
  count.textContent=matches.length+' matching / '+events.length+' total';
  load.hidden=matches.length<=visibleLimit; load.textContent='Load '+Math.min(220,matches.length-visibleLimit)+' more';
}
function chips(values) {
  const root=document.getElementById('event-chips'); root.replaceChildren();
  values.filter(([,value])=>value!==null && value!==undefined && value!=='').forEach(([label,value])=>root.append(node('span','event-chip',label+' '+value)));
}
function renderPayload() {
  const payload=document.getElementById('event-payload');
  if(!selectedDetail) return;
  payload.textContent=tab==='exact' ? selectedDetail.raw_json : pretty(selectedDetail.record);
  payload.className=tab==='exact' ? 'exact'+(wrap?' wrap':'') : '';
  document.querySelectorAll('.event-tab[data-tab]').forEach(button=>button.classList.toggle('active',button.dataset.tab===tab));
}
function renderArtifacts(detail) {
  const strip=document.getElementById('artifact-strip'); strip.replaceChildren();
  (detail.artifacts || []).forEach(artifact=>{ const link=node('a','artifact-link'); link.href=artifactUrl(artifact.path); link.target='_blank'; link.rel='noreferrer'; link.append(node('span','',artifact.path),node('span','',artifact.kind || artifact.media_type || artifact.bytes+' B')); strip.append(link); });
}
	function renderCycle(step) {
	  const root=document.getElementById('cycle-scroll'); root.replaceChildren();
	  const cycle=trace.cycles.find(item=>item.step===step); document.getElementById('cycle-label').textContent=cycle ? 'cycle '+String(step).padStart(2,'0') : 'prelude';
	  if(!cycle){ root.append(node('div','cycle-empty','This source event is not bound to an observation cycle.')); return; }
	  incidentsForStep(cycle.state_step).forEach(item=>{const card=node('section','cycle-analysis '+stageTone(item,cycle.state_step));card.append(node('span','',item.id+' · curated review'),node('h3','',item.title),node('p','',item.summary));root.append(card);});
  const frames=[...(cycle.images || []).map(frame=>({...frame,phase:'current / actual group'})),...(cycle.after_images || []).map(frame=>({...frame,phase:'next actual'}))];
  const grid=node('div','cycle-images');
  frames.slice(0,8).forEach(frame=>{ const figure=node('figure','cycle-image'); const link=node('a'); link.href=artifactUrl(frame.path); link.target='_blank'; const image=node('img'); image.src=artifactUrl(frame.path); image.loading='lazy'; image.alt=frame.phase+' '+frame.label; link.append(image); figure.append(link,node('figcaption','',frame.phase+' · '+frame.label+(frame.kind?' · '+frame.kind:''))); grid.append(figure); });
  if(frames.length) root.append(grid); else root.append(node('div','cycle-empty','No renderable image is indexed for this observation cycle.'));
	  [['Cycle identity',{round:cycle.round_index===null?null:cycle.round_index+1,state_step:cycle.state_step,observation_id:cycle.observation_id,remaining_budget:cycle.remaining_budget,model_calls:cycle.model_call_count,tools:cycle.tool_count,warnings:cycle.warnings}],['Action navigation summary',cycle.action || {status:'no mutation'}],['Receipt navigation summary',cycle.receipt || {status:'no receipt'}]].forEach(([title,value])=>{ const card=node('section','cycle-card'); card.append(node('h3','',title),node('pre','',pretty(value))); root.append(card); });
	}
	function renderIncidents(){
	  const root=document.getElementById('incident-strip'); if(!analysis||!incidents.length)return;root.hidden=false;root.replaceChildren();
	  incidents.forEach(item=>{const button=node('button','incident-card '+(item.tone||'')+(selectedIncident===item.id?' active':''));button.type='button';const header=node('header');header.append(node('span','',item.id+' · Round '+(item.round_index+1)),node('span','', 'S'+String(item.focus_step).padStart(2,'0')));button.append(header,node('strong','',item.title),node('p','',item.summary));const frames=node('div','incident-keyframes');(item.keyframes||[item.focus_step]).slice(0,3).forEach(step=>{const cycle=trace.cycles.find(value=>value.state_step===step);const frame=(cycle?.images||[]).find(value=>value.main&&value.visibility==='policy_visible')||(cycle?.images||[]).find(value=>value.visibility==='policy_visible');if(!frame)return;const figure=node('span','incident-keyframe');const image=node('img');image.src=artifactUrl(frame.path);image.loading='lazy';image.alt=item.id+' state step '+step;figure.append(image,node('span','', 'S'+String(step).padStart(2,'0')));frames.append(figure);});if(frames.childElementCount)button.append(frames);const chain=node('div','incident-chain');(item.stages||[]).forEach((stage,index)=>{if(index)chain.append(node('span','incident-arrow','→'));chain.append(node('span','incident-stage '+(stage.tone||''),'S'+String(stage.step).padStart(2,'0')+' '+stage.label));});button.append(chain);button.onclick=()=>{selectedIncident=item.id;roundFilter.value=String(item.round_index);stepFilter.value='all';lens='all';document.querySelectorAll('.review-lens').forEach(value=>value.classList.toggle('active',value.dataset.lens==='all'));renderIncidents();renderList(true);const candidates=events.filter(event=>event.observation_step===item.focus_step);const preferred=candidates.find(event=>/(tool_result|motion_receipt|provider_response)/.test(event.type))||candidates[0];if(preferred)selectEvent(preferred.key,true);};root.append(button);});
	}
async function selectEvent(key, scroll=false) {
  const event=eventByKey.get(key); if(!event) return; selectedKey=key; renderList();
  const selected=list.querySelector('[data-key="'+CSS.escape(key)+'"]'); if(scroll && selected) selected.scrollIntoView({block:'nearest'});
  document.getElementById('event-title').textContent=event.type;
  document.getElementById('event-source').textContent=event.source_path+':'+event.source_line+' · '+relativeTime(event.timestamp_unix_s);
  chips([['stream',event.stream],['step',event.observation_step],['line sha',event.line_sha256.slice(0,12)],...Object.entries(event.linkage || {})]);
  renderCycle(event.observation_step);
  const requested=key;
  selectedDetail=null; const payload=document.getElementById('event-payload'); payload.textContent='Loading exact source line…'; document.getElementById('artifact-strip').replaceChildren();
  try {
    const response=await fetch('/api/run/'+encodeURIComponent(trace.run_id)+'/trajectory/raw/'+encodeURIComponent(event.stream)+'/'+event.source_line);
    if(!response.ok) throw new Error('HTTP '+response.status);
    const detail=await response.json(); if(selectedKey!==requested) return; selectedDetail=detail; renderPayload(); renderArtifacts(detail);
  } catch(error) {
    if(selectedKey!==requested) return; payload.textContent='Could not load this source line: '+error.message;
  }
  history.replaceState(null,'','#event='+encodeURIComponent(key));
}
	(trace.rounds||[]).forEach(round=>{const option=node('option','', 'Round '+(round.round_index+1)+' · '+(round.official_success===true?'success':round.official_success===false?'failure':'unscored'));option.value=String(round.round_index);roundFilter.append(option);});
	trace.cycles.forEach(cycle=>{ const option=node('option','', 'Cycle '+String(cycle.step).padStart(2,'0')+' · '+(cycle.observation_id || 'observation')); option.value=String(cycle.step); stepFilter.append(option); });
	document.querySelectorAll('.stream-filter').forEach(button=>button.onclick=()=>{ stream=button.dataset.stream; document.querySelectorAll('.stream-filter').forEach(item=>item.classList.toggle('active',item===button)); renderList(true); });
	document.querySelectorAll('.review-lens').forEach(button=>button.onclick=()=>{lens=button.dataset.lens;document.querySelectorAll('.review-lens').forEach(item=>item.classList.toggle('active',item===button));renderList(true);});
	search.oninput=()=>renderList(true); roundFilter.onchange=()=>renderList(true); stepFilter.onchange=()=>renderList(true); kindFilter.onchange=()=>renderList(true); load.onclick=()=>{visibleLimit+=220; renderList();};
document.querySelectorAll('.event-tab[data-tab]').forEach(button=>button.onclick=()=>{tab=button.dataset.tab; renderPayload();});
document.getElementById('toggle-wrap').onclick=()=>{wrap=!wrap; document.getElementById('toggle-wrap').classList.toggle('active',wrap); renderPayload();};
document.getElementById('copy-event').onclick=async()=>{if(selectedDetail) await navigator.clipboard.writeText(selectedDetail.raw_json);};
document.addEventListener('keydown',event=>{ const typing=['INPUT','SELECT','TEXTAREA'].includes(document.activeElement.tagName); if(event.key==='/' && !typing){event.preventDefault(); search.focus();} if(event.key==='Escape'){search.value=''; search.blur(); renderList(true);} if(typing) return; const matches=filtered(); const index=matches.findIndex(item=>item.key===selectedKey); if(event.key.toLowerCase()==='j' && matches.length) selectEvent(matches[Math.min(matches.length-1,index+1)].key,true); if(event.key.toLowerCase()==='k' && matches.length) selectEvent(matches[Math.max(0,index-1)].key,true); if(event.key.toLowerCase()==='c' && selectedDetail) navigator.clipboard.writeText(selectedDetail.raw_json); });
	renderIncidents(); renderList(); const initial=new URLSearchParams(location.hash.slice(1)).get('event'); selectEvent(eventByKey.has(initial)?initial:(events[0]||{}).key);
</script>"""
    )
    return page(f"Raw trajectory · {run_id}", body, wide=True)


@app.get("/api/run/<run_id>/trajectory")
def trajectory_api(run_id: str):
    return jsonify(compile_trajectory(run_root(run_id)))


@app.get("/api/run/<run_id>/process")
def process_api(run_id: str):
    return jsonify(compile_process_timeline(run_root(run_id)))


def _presentation_options() -> VideoOptions:
    selected_round = request.args.get("round")
    reasoning, context = request.args.get("reasoning", "0"), request.args.get("context", "0")
    if reasoning not in {"0", "1"} or context not in {"0", "1"}:
        raise ValueError("reasoning/context must be 0 or 1")
    return VideoOptions(
        mode=request.args.get("video_mode", "execution"),
        round_index=int(selected_round) - 1 if selected_round is not None else None,
        reasoning=reasoning == "1", context=context == "1",
        card_seconds=float(request.args.get("card_seconds", "2.5")),
        replay_mode=request.args.get("replay_mode", "review"),
    )


@app.get("/api/run/<run_id>/presentation-video")
def presentation_video_api(run_id: str):
    try:
        _video, manifest = presentation_video(run_root(run_id), _presentation_options())
    except PresentationPendingError as error:
        return jsonify({"status": "pending_compile", "compile_hint": str(error)}), 503
    except ValueError as error:
        return jsonify({"status": "invalid_request", "detail": str(error)}), 400
    return jsonify(manifest)


@app.get("/run/<run_id>/presentation.mp4")
def presentation_video_file(run_id: str):
    try:
        video, _manifest = presentation_video(run_root(run_id), _presentation_options())
    except PresentationPendingError as error:
        return Response(str(error), status=503, mimetype="text/plain")
    except ValueError as error:
        return Response(str(error), status=400, mimetype="text/plain")
    return send_file(video, mimetype="video/mp4", conditional=True)


@app.get("/api/run/<run_id>/cost")
def run_cost_api(run_id: str):
    return jsonify(estimate_run_cost(run_root(run_id)))


@app.get("/api/run/<run_id>/trajectory/raw")
def raw_trajectory_api(run_id: str):
    return jsonify(compile_raw_trajectory(run_root(run_id)))


@app.get("/api/run/<run_id>/trajectory/raw/<stream>/<int:source_line>")
def raw_trajectory_event_api(run_id: str, stream: str, source_line: int):
    try:
        return jsonify(raw_trajectory_event(run_root(run_id), stream, source_line))
    except ValueError:
        abort(404)


@app.get("/run/<run_id>/trajectory/raw")
def raw_trajectory_detail(run_id: str) -> str:
    return _raw_trajectory_page(run_id, compile_raw_trajectory(run_root(run_id)))


@app.get("/run/<run_id>/trajectory.mp4")
def trajectory_video(run_id: str) -> Response:
    replay_mode = request.args.get("replay_mode", "review")
    if replay_mode not in REPLAY_MODES:
        abort(400)
    root = run_root(run_id)
    existing = find_execution_media(root, compile_trajectory(root), replay_mode=replay_mode)
    if existing is not None:
        return send_file(existing["path"], mimetype="video/mp4", conditional=True)
    try:
        video, _manifest = compile_trajectory_video(
            root, replay_mode=replay_mode, allow_replay_compile=False,
        )
    except TrajectoryVideoPendingError as error:
        return Response(str(error), status=503, mimetype="text/plain")
    return send_file(video, mimetype="video/mp4", conditional=True)


@app.get("/api/run/<run_id>/trajectory-video")
def trajectory_video_api(run_id: str):
    replay_mode = request.args.get("replay_mode", "review")
    if replay_mode not in REPLAY_MODES:
        abort(400)
    root = run_root(run_id)
    existing = find_execution_media(root, compile_trajectory(root), replay_mode=replay_mode)
    if existing is not None:
        return jsonify(existing["manifest"])
    try:
        _video, manifest = compile_trajectory_video(
            root, replay_mode=replay_mode, allow_replay_compile=False,
        )
    except TrajectoryVideoPendingError as error:
        return jsonify({"status": "pending_compile", "detail": str(error)}), 503
    return jsonify(manifest)


@app.get("/api/run/<run_id>/composition")
def composition_api(run_id: str):
    try:
        return jsonify(inspect_run_configuration(run_root(run_id)))
    except ValueError:
        abort(404)


@app.get("/run/<run_id>/composition")
def composition_detail(run_id: str) -> str:
    try:
        inspection = inspect_run_configuration(run_root(run_id))
    except ValueError:
        abort(404)
    return _composition_page(run_id, inspection)


@app.get("/run/<run_id>")
def trajectory_detail(run_id: str) -> str:
    root = run_root(run_id)
    trajectory = compile_trajectory(root)
    return _trajectory_page(run_id, trajectory)


@app.get("/task/libero_pro/<suite>/<int:task_id>/runs")
def libero_task_runs(suite: str, task_id: int) -> str:
    """Compare one exact LIBERO task across experiment/source conditions."""

    task_runs = [
        run for run in build_registry(scoped_run_roots(suite=suite, task_id=task_id), inspect_video=False)
        if run.backend == "libero_pro"
        and run.suite == suite
        and run.task_id == task_id
        and run.category in {"formal", "attempt"}
    ]
    if not task_runs:
        abort(404)
    view = request.args.get("view", "completed")
    if view not in {"completed", "formal", "all"}:
        view = "completed"
    outcome = request.args.get("outcome", "")
    if outcome not in {"", "success", "failure", "unscored"}:
        outcome = ""
    seeds = sorted(
        {str(run.seed) for run in task_runs},
        key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value),
    )
    seed = request.args.get("seed", "")
    if seed not in seeds:
        seed = ""

    def included(run) -> bool:
        if view == "completed" and run.state != "completed":
            return False
        if view == "formal" and run.category != "formal":
            return False
        if seed and str(run.seed) != seed:
            return False
        if outcome == "success" and run.official_success is not True:
            return False
        if outcome == "failure" and run.official_success is not False:
            return False
        if outcome == "unscored" and run.official_success is not None:
            return False
        return True

    selected = [run for run in task_runs if included(run)]
    completed = [run for run in task_runs if run.state == "completed"]
    formal = [run for run in task_runs if run.category == "formal"]
    successes = [run for run in completed if run.official_success is True]
    failures = [run for run in completed if run.official_success is False]
    task_label = task_runs[0].task
    task_key = str(task_runs[0].task_key)

    metrics_by_run = {
        run.run_id: _comparison_metrics(run.root) for run in selected
    }
    available_metrics = [
        value for value in metrics_by_run.values()
        if value.get("status") not in {"missing", "identity_mismatch"}
    ]
    available_costs = [
        value["cost"] for value in metrics_by_run.values()
        if isinstance(value.get("cost"), dict)
        and value["cost"].get("total_cost") is not None
    ]
    image_scale = max(
        (
            float(value["images_mean"])
            for value in available_metrics
            if value.get("images_mean") is not None
        ),
        default=1.0,
    ) or 1.0

    def percentage(value: object) -> str:
        number = _metric_number(value)
        return "—" if number is None else f"{number * 100:.1f}%"

    def bar_width(value: object, *, scale: float = 1.0) -> str:
        number = _metric_number(value)
        if number is None:
            return "0"
        return f"{max(0.0, min(100.0, number / scale * 100)):.2f}"

    rows = []
    for run in selected:
        date, clock, iso = _utc_parts(run.started_unix_s)
        verdict = run.official_success
        verdict_text, outcome_class, outcome_kind = _registered_outcome(run)
        verdict_css = (
            "success" if outcome_class == "ok" else
            "failure" if outcome_class == "bad" else "unscored"
        )
        category_label = (
            "formal" if run.category == "formal" else "non-formal attempt"
        )
        comparison = metrics_by_run[run.run_id]
        cost = (
            comparison.get("cost")
            if isinstance(comparison.get("cost"), dict) else {}
        )
        cost_cell = _cost_cell(cost)
        unavailable = comparison.get("status") in {"missing", "identity_mismatch"}
        if unavailable:
            context_cell = images_cell = cache_cell = (
                "<div class='metric-missing'>runtime-metrics.json "
                f"{html.escape(str(comparison.get('status'))).replace('_', ' ')}</div>"
            )
            policy_metric = "metrics unavailable"
        else:
            context_utilization = comparison.get("context_utilization")
            context_window = comparison.get("context_window")
            context_cell = (
                f"<div class='metric-primary'>{_compact_metric(comparison.get('context_mean'))}<span>avg tokens</span></div>"
                f"<div class='metric-track'><i style='width:{bar_width(context_utilization)}%'></i></div>"
                f"<small>p95 {_compact_metric(comparison.get('context_p95'))} · peak {_compact_metric(comparison.get('context_peak'))}</small>"
                f"<small>{percentage(context_utilization)} of {_compact_metric(context_window)} window · {html.escape(str(comparison.get('context_quality')))}</small>"
            )
            image_mean = comparison.get("images_mean")
            images_cell = (
                f"<div class='metric-primary'>{_compact_metric(image_mean)}<span>avg images</span></div>"
                f"<div class='metric-track images'><i style='width:{bar_width(image_mean, scale=image_scale)}%'></i></div>"
                f"<small>p95 {_compact_metric(comparison.get('images_p95'))} · peak {_compact_metric(comparison.get('images_peak'))} · total {_compact_metric(comparison.get('images_total'))}</small>"
                f"<small>{html.escape(str(comparison.get('images_quality')))} · {html.escape(str(comparison.get('images_measurement')))}</small>"
            )
            cache_ratio = comparison.get("cache_token_ratio")
            cache_cell = (
                f"<div class='metric-primary'>{percentage(cache_ratio)}<span>token read</span></div>"
                f"<div class='metric-track cache'><i style='width:{bar_width(cache_ratio)}%'></i></div>"
                f"<small>{percentage(comparison.get('cache_call_ratio'))} calls hit · {_compact_metric(comparison.get('cache_read_tokens'))} cached tokens</small>"
                f"<small>{html.escape(str(comparison.get('cache_quality')))} · provider telemetry</small>"
            )
            transitions = comparison.get("epoch_transitions")
            transition_text = (
                "epochs not recorded" if transitions is None
                else f"{int(transitions)} projection transition{'s' if transitions != 1 else ''}"
            )
            policy_metric = (
                f"{_compact_metric(comparison.get('calls'))} calls · {transition_text}"
            )
        rows.append(
            "<tr>"
            f"<td class='treatment-cell'><a href='/experiment/{html.escape(run.experiment)}/runs'>{html.escape(run.experiment)}</a>"
            f"<strong>{html.escape(run.config)}</strong><div class='treatment-meta'><span>seed {html.escape(str(run.seed))}</span>"
            f"<span>{html.escape(run.model)}</span><span>source {html.escape(run.source_commit[:12])}</span>"
            f"<span>{html.escape(date)} · {html.escape(clock)}</span></div></td>"
            f"<td class='result-cell'><span class='outcome-mark {html.escape(verdict_css)}'>{html.escape(verdict_text)}</span>"
            f"<small>{html.escape(category_label)} · {html.escape(outcome_kind)} · evaluation {html.escape(run.evaluation)}</small></td>"
            f"<td class='policy-cell'><strong>{html.escape(policy_metric)}</strong>"
            f"<code>{html.escape(run.policy_kind)} · {html.escape(str(comparison.get('context_profile') or 'not recorded'))}</code></td>"
            f"<td class='metric-cell'>{context_cell}</td>"
            f"<td class='metric-cell'>{images_cell}</td>"
            f"<td class='metric-cell'>{cache_cell}</td>"
            f"<td class='metric-cell'>{cost_cell}</td>"
            f"<td class='review-cell'><div class='review-links'><a href='/run/{html.escape(run.run_id)}'>Trajectory</a>"
            f"<a href='/run/{html.escape(run.run_id)}/composition'>Composition</a>"
            f"<a href='/run/{html.escape(run.run_id)}/trajectory/raw'>Raw trace</a>"
            f"<a href='/run/{html.escape(run.run_id)}/trajectory.mp4'>Video</a></div>"
            f"<small><code>{html.escape(run.run_id)}</code></small></td></tr>"
        )

    tabs = "".join(
        f"<a class='{'active' if value == view else ''}' href='{html.escape(_libero_task_runs_url(suite, task_id, view=value, seed=seed, outcome=outcome))}'>{label}</a>"
        for value, label in (
            ("completed", f"Completed · {len(completed)}"),
            ("formal", f"Formal · {len(formal)}"),
            ("all", f"All behavioral · {len(task_runs)}"),
        )
    )
    seed_options = ["<option value=''>All seeds</option>"] + [
        f"<option value='{html.escape(value)}'{' selected' if value == seed else ''}>{html.escape(value)}</option>"
        for value in seeds
    ]
    outcome_options = [
        ("", "All outcomes"), ("success", "Success"),
        ("failure", "Failure"), ("unscored", "Unscored"),
    ]
    outcome_select = "".join(
        f"<option value='{value}'{' selected' if value == outcome else ''}>{label}</option>"
        for value, label in outcome_options
    )
    root_query = urlencode({
        "view": "attempts", "backend": "libero_pro", "task": task_key,
        "seed": seed,
    })

    def visible_range(field: str, *, percent: bool = False) -> str:
        values = sorted(
            float(value[field]) for value in available_metrics
            if value.get(field) is not None
        )
        if not values:
            return "—"
        formatter = (
            (lambda number: f"{number * 100:.1f}%")
            if percent else _compact_metric
        )
        if len(values) == 1 or values[0] == values[-1]:
            return formatter(values[0])
        return f"{formatter(values[0])}–{formatter(values[-1])}"

    selection_label = f"seed {seed}" if seed else "all seeds"
    selected_cost = sum(
        float(value["total_cost"]) for value in available_costs
        if value.get("total_cost") is not None
    )
    body = (
        f"<nav class='nav'><a href='/?{html.escape(root_query)}'>← Run registry</a></nav>"
        "<span class='eyebrow'>Cross-experiment task cohort</span>"
        f"<h1>{html.escape(task_label)}</h1>"
        f"<p class='muted'><code>libero_pro / {html.escape(suite)} / task {task_id}</code> · "
        f"{len({run.experiment for run in task_runs})} experiments · newest first</p>"
        "<div class='cards'>"
        f"<article class='card'><h3>Completed</h3><p><strong>{len(completed)}</strong> behavioral runs</p></article>"
        f"<article class='card'><h3>Evidence</h3><p><strong>{len(formal)}</strong> formal · {len(task_runs) - len(formal)} attempts</p></article>"
        f"<article class='card'><h3>Official result</h3><p><strong>{len(successes)}</strong> success · {len(failures)} failure</p></article>"
        f"<article class='card'><h3>Selected</h3><p><strong>{len(selected)}</strong> visible runs</p></article>"
        "</div>"
        "<p class='warning'>This page groups one exact environment task, not one matched treatment. "
        "Prompt, tools, context, model, budgets, source revision and infrastructure validity may differ; "
        "inspect Composition before making a causal or pooled success-rate claim.</p>"
        f"<div class='registry-controls'><nav class='tabs'>{tabs}</nav>"
        f"<form><input type='hidden' name='view' value='{html.escape(view)}'>"
        f"<label>Seed<select name='seed' onchange='this.form.submit()'>{''.join(seed_options)}</select></label>"
        f"<label>Outcome<select name='outcome' onchange='this.form.submit()'>{outcome_select}</select></label>"
        "</form></div>"
        f"{render_round_matrix(selected)}"
        "<section class='comparison-note'><div><span class='eyebrow'>Context workload comparison</span>"
        f"<h2>{html.escape(selection_label)} · descriptive telemetry ledger</h2>"
        "<p>Request context is provider-reported input per exchange, not cumulative workload. "
        "Image averages count submitted image instances per request, so retained images are counted again; "
        "a managed channel may not expose the same inventory. Cache read is token-weighted provider telemetry, "
        "not evidence of useful attention or task competence.</p></div><dl class='comparison-key'>"
        f"<div><dt>reports</dt><dd>{len(available_metrics)}/{len(selected)}</dd></div>"
        f"<div><dt>avg context range</dt><dd>{visible_range('context_mean')}</dd></div>"
        f"<div><dt>avg images range</dt><dd>{visible_range('images_mean')}</dd></div>"
        f"<div><dt>cache read range</dt><dd>{visible_range('cache_token_ratio', percent=True)}</dd></div>"
        f"<div><dt>selected list price</dt><dd>{_money(selected_cost) if available_costs else '—'}</dd></div>"
        "</dl></section>"
        "<div class='cohort-scroll'><table class='cohort-ledger'><thead><tr>"
        "<th>experiment / exact treatment</th><th>result</th><th>calls / context policy</th>"
        "<th>request context</th><th>images / request</th><th>cache reuse</th><th>cost</th><th>review</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
        "<p class='measurement-footnote'>All values come from each run's deterministic "
        "<code>runtime-metrics.json</code>. Missing telemetry remains missing; the Viewer does not estimate "
        "image tokens, hidden provider context or KV-cache contents. Cost is the versioned official API "
        "list-price equivalent, not an account invoice; taxes, subscriptions and discounts are excluded.</p>"
    )
    return page(f"Task cohort · {task_label}", body, wide=True)


@app.get("/experiment/<experiment_id>/runs")
def experiment_runs(experiment_id: str) -> str:
    runs = [
        run for run in build_registry(scoped_run_roots(experiment=experiment_id), inspect_video=False)
        if run.experiment == experiment_id
    ]
    if not runs:
        abort(404)
    definition_path, definition = experiment_definition(experiment_id)
    experiment_view = render_experiment_view(
        definition_path, definition, runs, _comparison_metrics,
    )
    rows = []
    for run in runs:
        date, clock, iso = _utc_parts(run.started_unix_s)
        verdict_text, _outcome_class, outcome_kind = _registered_outcome(run)
        rows.append(
            f"<tr><td class='run-time'><time datetime='{html.escape(iso)}'><strong>{html.escape(date)}</strong>"
            f"<small>{html.escape(clock)}</small></time></td>"
            f"<td>{html.escape(run.config)}<small>{html.escape(run.experiment)}</small><small>{html.escape(run.category)} · {html.escape(run.milestone)}</small></td>"
            f"<td>{html.escape(run.task)}<small>seed {html.escape(str(run.seed))}</small></td>"
            f"<td>{html.escape(run.state)} · {html.escape(verdict_text)}<small>{html.escape(outcome_kind)} · evaluation {html.escape(run.evaluation)}</small></td>"
            f"<td><div class='review-links'><a href='/run/{html.escape(run.run_id)}'>Summary</a>"
            f"<a href='/run/{html.escape(run.run_id)}/composition'>Composition</a>"
            f"<a href='/run/{html.escape(run.run_id)}/trajectory/raw'>Raw trace</a>"
            f"<a href='/run/{html.escape(run.run_id)}/raw'>Evidence</a></div></td>"
            f"<td class='run-id'><code>{html.escape(run.run_id)}</code></td></tr>"
        )
    config_link = f"<a href='/experiment/{html.escape(experiment_id)}/configs'>Configuration catalog</a>"
    body = f"<nav class='nav'><a href='/?view=all'>← All experiments / runs</a>{config_link}</nav>"
    title = str(definition.get('title') or experiment_id)
    body += f"<span class='eyebrow'>Registered evidence collection</span><h1>{html.escape(title)}</h1>"
    body += f"<p class='muted'><code>{html.escape(experiment_id)}</code> · {len(rows)} locally indexed runs · newest first</p>"
    body += render_round_matrix(runs)
    body += experiment_view.html
    body += (
        "<div class='ui-table-scroll'><table><thead><tr><th>date</th><th>config / type</th><th>task</th>"
        "<th>outcome</th><th>review</th><th>exact run ID</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table></div>"
    )
    return page(f"Experiment · {experiment_id}", body, wide=True)


@app.get("/experiment/<experiment_id>/configs")
def experiment_configs(experiment_id: str) -> str:
    definition_path, definition = experiment_definition(experiment_id)
    if definition_path is None:
        abort(404)
    configs = definition.get("configs")
    if not isinstance(configs, dict):
        abort(404)
    indexed_runs = [
        run for run in build_registry(scoped_run_roots(experiment=experiment_id), inspect_video=False)
        if run.experiment == experiment_id
    ]
    latest_by_config = {}
    for run in indexed_runs:
        latest_by_config.setdefault(run.config, run)
    rows = []
    for config_id, raw_config in configs.items():
        config = raw_config if isinstance(raw_config, dict) else {}
        task = (
            str(config.get("task")) if config.get("task") is not None
            else f"{config.get('suite', 'task')} / task {config.get('task_id', '—')}"
        )
        prompt = " · ".join(filter(None, (
            str(config.get("policy") or ""),
            f"prompt {config['prompt_version']}" if config.get("prompt_version") else None,
            f"context {config['context_profile']}" if config.get("context_profile") else None,
        )))
        controls = " · ".join(filter(None, (
            f"surface {config['surface']}" if config.get("surface") else None,
            str(config.get("evidence_tier") or ""),
        )))
        budgets = json.dumps(
            config.get("budgets", {}), ensure_ascii=False, sort_keys=True,
        )
        run = latest_by_config.get(str(config_id))
        if run is None:
            evidence = "<span class='config-status'>not indexed</span><small>No local run evidence for this config.</small>"
        else:
            date, clock, _iso = _utc_parts(run.started_unix_s)
            evidence = (
                f"<a href='/run/{html.escape(run.run_id)}/composition'>Exact latest composition</a>"
                f"<small>{html.escape(date)} {html.escape(clock)} · seed {html.escape(str(run.seed))}</small>"
                f"<small><code>{html.escape(run.run_id)}</code></small>"
            )
        instruction = config.get("instruction")
        instruction_html = (
            f"<small>{html.escape(str(instruction))}</small>" if instruction else ""
        )
        rows.append(
            f"<tr><td><code>{html.escape(str(config_id))}</code>"
            f"<small>{html.escape(prompt or 'policy unspecified')}</small></td>"
            f"<td>{html.escape(task)}{instruction_html}</td>"
            f"<td>{html.escape(controls or '—')}</td>"
            f"<td><code>{html.escape(budgets)}</code></td><td>{evidence}</td></tr>"
        )
    experiment_dir = definition_path.parent.relative_to(ROOT).as_posix()
    run_script = definition_path.parent / "run.py"
    supports_inspection = (
        run_script.is_file()
        and "inspect-config" in run_script.read_text(encoding="utf-8", errors="replace")
    )
    command = (
        f"uv run --locked python {experiment_dir}/run.py inspect-config "
        "CONFIG --seed 0"
    )
    run_nav = (
        f"<a href='/experiment/{html.escape(experiment_id)}/runs'>Indexed runs</a>"
        if indexed_runs else ""
    )
    inspection_html = (
        f"<p>Resolve one config without launching a provider or environment job:</p><pre>{html.escape(command)}</pre>"
        if supports_inspection else
        "<p>This archived package does not register the shared prospective inspector. Use a recorded run's Composition page for exact resolved evidence.</p>"
    )
    body = (
        f"<nav class='nav'><a href='/'>← All runs</a>{run_nav}</nav>"
        "<span class='eyebrow'>Tracked experiment declaration</span>"
        f"<h1>{html.escape(str(definition.get('title') or experiment_id))}</h1>"
        "<p class='muted'>This table is the static experiment.yaml catalog. It tells you which cells may be composed; "
        "the Composition link on a recorded run is authoritative for the exact prompt text, request order, tools and hashes actually resolved for that run.</p>"
        f"<details class='provenance'><summary>Pre-run exact inspection</summary>{inspection_html}"
        f"<p>Declaration: <code>{html.escape(definition_path.relative_to(ROOT).as_posix())}</code></p></details>"
        "<table class='config-table'><thead><tr><th>config / policy</th><th>task declaration</th>"
        "<th>surface / gates</th><th>budgets</th><th>exact run evidence</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>"
    )
    return page(f"Configurations · {experiment_id}", body)


@app.get("/run/<run_id>/raw")
def detail(run_id: str) -> str:
    root = run_root(run_id)
    plan = load(root / "plan.json")
    context_manifest = load(root / "context-manifest.json")
    result = load(root / "result.json")
    session = load(root / "native/agent-session.json")
    runtime_metrics = load(root / "runtime-metrics.json")
    context_audit = load(root / "context-audit.json")
    main_camera = str(plan.get("runtime", {}).get("main_camera", "head_camera"))
    observations = sorted((root / "policy-observations").glob(f"*/{main_camera}/rgb.png"))
    state_manifest = load(root / "policy-state/manifest.json")
    state_records = state_manifest.get("steps", [])
    try:
        selected_step = int(request.args.get("step", len(state_records) - 1))
    except ValueError:
        selected_step = len(state_records) - 1
    if not 0 <= selected_step < len(state_records):
        selected_step = len(state_records) - 1
    selected_state = (
        load(root / state_records[selected_step]["path"])
        if state_records and selected_step >= 0 else {}
    )
    state_tabs = "".join(
        f"<a class='{'active' if item.get('step') == selected_step else ''}' "
        f"href='?step={item.get('step')}'>{item.get('step')}</a>"
        for item in state_records
    )
    state_image_ref = (
        selected_state.get("cameras", {})
        .get(selected_state.get("main_camera", "head_camera"), {})
        .get("files", {}).get("rgb_lossless_png", {}).get("path")
    )
    state_image = (
        f"<img src='/artifact/{html.escape(run_id)}/{html.escape(str(state_image_ref))}'>"
        if state_image_ref else "<p>no state image</p>"
    )
    images = "".join(
        f"<figure><img src='/artifact/{html.escape(run_id)}/{path.relative_to(root).as_posix()}'><figcaption>{html.escape(path.parent.parent.name)}</figcaption></figure>"
        for path in observations
    ) or "<p>no images</p>"
    video = root / "native/operator-world.mp4"
    video_html = (
        f"<video controls src='/artifact/{html.escape(run_id)}/{video.relative_to(root).as_posix()}'></video>"
        if video.is_file() else "<p>no world video</p>"
    )
    body = (
        "<nav class='nav'><a href='/'>← All runs</a>"
        f"<a href='/run/{html.escape(run_id)}'>Trajectory summary</a>"
        f"<a href='/run/{html.escape(run_id)}/composition'>Composition</a>"
        f"<a href='/run/{html.escape(run_id)}/trajectory/raw'>Raw trajectory</a></nav>"
        f"<h1>{html.escape(run_id)}</h1>"
    )
    body += f"<p>{html.escape(str(plan.get('milestone')))} · {html.escape(str(plan.get('task')))} · seed {html.escape(str(plan.get('seed')))}</p>"
    body += "<div class='grid'><section><h2>Plan / task / policy context</h2>"
    body += (
        f"<pre>{raw(root/'plan.json')}</pre>"
        f"<h3>Compiled episode prefix</h3><pre>{raw(root/'compiled-policy/episode-prefix.md')}</pre>"
        f"<h3>Context manifest</h3><pre>{raw(root/'context-manifest.json')}</pre></section>"
    )
    body += f"<section><h2>World video</h2>{video_html}<h2>Policy images</h2>{images}</section></div>"
    memory = context_manifest.get("memory_system", {})
    environment_status = result.get("environment_status", {})
    body += (
        "<h2>Runtime lanes</h2><div class='cards'>"
        f"<article class='card'><h3>AgentSession</h3><p>backend: <code>{html.escape(str(session.get('backend', 'not recorded')))}</code></p>"
        f"<p>process: {html.escape(str(session.get('process_pid', '—')))} · invocations: {html.escape(str(session.get('process_invocations', '—')))} · restarts: {html.escape(str(session.get('process_restarts', 0)))}</p>"
        f"<p>thread: <code>{html.escape(str(session.get('thread_id', '—')))}</code></p></article>"
        f"<article class='card'><h3>Memory</h3><p>{'active' if memory.get('active') else 'disabled'}</p>"
        f"<p>provider: {html.escape(str(memory.get('provider', '—')))} · generation: {html.escape(str(memory.get('generation', '—')))}</p>"
        f"<p>block hash: <code>{html.escape(str(memory.get('block_sha256', '—')))}</code></p></article>"
        f"<article class='card'><h3>Native context</h3><p>compactions: {len(session.get('compaction_events', []))}</p>"
        f"<p>Harness rollovers: {len(session.get('rollovers', []))}</p><p>Codex history/KV remains opaque.</p></article>"
        f"<article class='card'><h3>Environment status</h3><p>{html.escape(str(environment_status.get('phase', 'not recorded')))}</p>"
        f"<p>reason: <code>{html.escape(str(environment_status.get('reason', '—')))}</code> · visibility: "
        f"{html.escape(str(environment_status.get('policy_visibility', '—')))}</p></article>"
        "</div>"
    )
    body += "<h2>Runtime evaluation</h2>" + _runtime_metric_cards(runtime_metrics)
    body += "<details><summary>Complete runtime-metrics.json</summary>"
    body += f"<pre>{raw(root/'runtime-metrics.json')}</pre></details>"
    disclosure = context_audit.get("camera_disclosure", {})
    coding = context_audit.get("coding", {})
    repetition = context_audit.get("verifiable_redundancy", {})
    body += (
        "<h2>Audit lanes</h2>"
        "<div class='lane'><h3>Context</h3>"
        f"<p>turns: {_metric(len(context_audit.get('turns', [])))} · later bytes median/max: "
        f"{_metric(context_audit.get('thresholds', {}).get('observed_later_median_bytes'))} / "
        f"{_metric(context_audit.get('thresholds', {}).get('observed_later_max_bytes'))}</p>"
        f"<p>exact repeated Harness messages after first: {_metric(repetition.get('harness_repeated_exact_messages', {}).get('repetitions_after_first'))}</p></div>"
        "<div class='lane'><h3>Coding</h3>"
        f"<p>shell commands: {_metric(coding.get('command_count'))} · scratch generations: {_metric(coding.get('scratch_artifact_generations'))}</p>"
        f"<p>immutable guard: {html.escape(str(coding.get('integrity_passed', 'not recorded')))}</p></div>"
        "<div class='lane'><h3>Camera disclosure</h3>"
        f"<p>automatic: <code>{html.escape(str(disclosure.get('automatic_policy_camera', main_camera)))}</code> · images: {_metric(disclosure.get('automatic_images'))}</p>"
        f"<p>on-demand tool images: {_metric(len(disclosure.get('tool_images', [])))} · auxiliary: {_metric(len(disclosure.get('auxiliary_camera_tool_images', [])))}</p></div>"
    )
    body += "<details><summary>Complete context-audit.json</summary>"
    body += f"<pre>{raw(root/'context-audit.json')}</pre></details>"
    body += "<h2>Usage and appended context</h2>" + _session_tables(session)
    body += "<h2>Unified causal timeline</h2>" + _timeline(root)
    body += f"<h2>Public agent transcript</h2><pre>{raw(root/'agent-transcript.jsonl')}</pre>"
    body += (
        f"<h2>Policy-state timeline</h2><div class='tabs'>{state_tabs}</div>"
        f"<div class='grid'><section>{state_image}</section>"
        f"<section><pre>{html.escape(json.dumps(selected_state, ensure_ascii=False, indent=2, sort_keys=True))}</pre></section></div>"
        f"<h3>State manifest</h3><pre>{raw(root/'policy-state/manifest.json')}</pre>"
    )
    bundles = sorted((root / "context-bundles").glob("call-*.json"))
    body += "<h2>Context bundles</h2>" + (
        "".join(f"<h3>{html.escape(path.name)}</h3><pre>{raw(path)}</pre>" for path in bundles)
        if bundles else "<p>no Codex context calls</p>"
    )
    body += f"<h2>Causal events</h2><pre>{raw(root/'events.jsonl')}</pre>"
    body += f"<h2>Evaluation / seal</h2><pre>{raw(root/'evaluation.json')}</pre><pre>{raw(root/'seal.json')}</pre>"
    return page(run_id, body)


@app.get("/artifact/<run_id>/<path:relative>")
def artifact(run_id: str, relative: str) -> Response:
    root = run_root(run_id)
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        abort(404)
    return send_file(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ora-viewer", description=__doc__)
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="bind address; keep the default for local-only evidence viewing",
    )
    parser.add_argument("--port", type=int, default=5012, help="HTTP listen port")
    parser.add_argument(
        "--open-browser", action="store_true",
        help="open the Viewer in the default browser after starting",
    )
    parser.add_argument(
        "--page", choices=tuple(VIEWER_START_PAGES), default="runs",
        help=(
            "browser start page when --open-browser is enabled "
            "(default: runs)"
        ),
    )
    parser.add_argument(
        "--outputs-root", type=Path, action="append", default=[],
        help=(
            "replace automatic shared/worktree discovery with this outputs root "
            "or experiment workspace; repeatable"
        ),
    )
    parser.add_argument(
        "--compile-videos", action="store_true",
        help="compile missing/current trajectory videos and exit",
    )
    parser.add_argument(
        "--audit-experiment-views", action="store_true",
        help="validate every tracked viewer.yaml and available source binding, then exit",
    )
    parser.add_argument(
        "--video-scope", choices=("formal", "libero", "robotwin"), default="formal",
        help="batch video scope (default: every formal run)",
    )
    parser.add_argument(
        "--replay-mode", choices=tuple(sorted(REPLAY_MODES)), default="review",
        help=(
            "review: original recorded frames only (default); forensic: explicit "
            "simulator reconstruction for diagnosis, not original visual evidence"
        ),
    )
    parser.add_argument("--run-id", action="append", default=[], help="compile an exact indexed run; repeatable, overrides batch scope")
    parser.add_argument("--video-mode", choices=("execution", "process"), default="execution", help="execution or optional full recorded process")
    parser.add_argument("--round", dest="video_round", type=int, help="export one recorded round (one-based)")
    parser.add_argument("--include-reasoning", action="store_true", help="include recorded API-visible reasoning excerpts in process videos")
    parser.add_argument("--include-context", action="store_true", help="include recorded Context/Memory lifecycle events in process videos")
    parser.add_argument("--card-seconds", type=float, default=2.5, help="reading-card duration, 0.5–15 seconds (default: 2.5)")
    return parser


def browser_start_url(*, host: str, port: int, page: str) -> str:
    """Resolve one allowlisted local Viewer start page."""

    browser_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    if ":" in browser_host and not browser_host.startswith("["):
        browser_host = f"[{browser_host}]"
    return f"http://{browser_host}:{port}{VIEWER_START_PAGES[page]}"


def compile_registered_videos(
    output_roots: list[Path], *, scope: str, replay_mode: str = "review",
    run_ids: list[str] | None = None, video_options: VideoOptions | None = None,
) -> tuple[int, int]:
    roots = list(discover_run_roots(output_roots))
    if run_ids:
        for run_id in set(run_ids):
            matches = [root for root in roots if root.name == run_id]
            if len(matches) != 1:
                raise ValueError(f"requested run ID is missing or ambiguous: {run_id}")
        roots = [root for root in roots if root.name in run_ids]
    registrations = build_registry(roots, inspect_video=False)
    selected = [
        run for run in registrations
        if (run.run_id in run_ids if run_ids else (
            (scope == "formal" and run.category == "formal")
            or (
                scope == "libero"
                and run.category == "formal"
                and run.backend == "libero_pro"
            )
            or (
                scope == "robotwin"
                and run.backend == "robotwin"
                and run.category in {"formal", "attempt", "diagnostic"}
            )
        ))
    ]
    if run_ids and set(run_ids) - {run.run_id for run in selected}:
        raise ValueError("requested run ID is not present in the configured output roots")
    completed = 0
    for run in selected:
        try:
            if video_options is None:
                video, manifest = compile_trajectory_video(run.root, replay_mode=replay_mode)
            else:
                try:
                    video, manifest = presentation_video(run.root, video_options, allow_compile=True, progress=lambda message: print(f"BUILD\t{run.run_id}\t{message}", flush=True))
                except PresentationPendingError:
                    # The exact execution compile is explicitly requested by
                    # this offline CLI. Process mode never enters a simulator.
                    if video_options.mode != "execution":
                        raise
                    compile_trajectory_video(run.root, replay_mode=replay_mode)
                    video, manifest = presentation_video(run.root, video_options, allow_compile=True)
        except Exception as exc:  # operator command reports all incompatible historical runs
            print(f"FAIL\t{run.run_id}\t{type(exc).__name__}: {exc}")
            continue
        completed += 1
        print(f"OK\t{run.run_id}\t{manifest.get('frame_count')} frames\t{video}")
    return completed, len(selected)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output_roots = [path.resolve() for path in args.outputs_root]
    if args.compile_videos and args.audit_experiment_views:
        raise SystemExit("--compile-videos and --audit-experiment-views are mutually exclusive")
    if args.compile_videos:
        video_options = None
        if args.run_id or args.video_mode != "execution" or args.video_round is not None or args.include_reasoning or args.include_context:
            video_options = VideoOptions(
                mode=args.video_mode,
                round_index=args.video_round - 1 if args.video_round is not None else None,
                reasoning=args.include_reasoning, context=args.include_context,
                card_seconds=args.card_seconds, replay_mode=args.replay_mode,
            )
        completed, selected = compile_registered_videos(
            output_roots or list(default_output_roots()),
            scope=args.video_scope,
            replay_mode=args.replay_mode,
            run_ids=args.run_id, video_options=video_options,
        )
        if completed != selected:
            raise SystemExit(1)
        return
    if args.audit_experiment_views:
        roots = output_roots or list(default_output_roots())
        reports = audit_experiment_view_files(
            ROOT, build_registry(discover_run_roots(roots)),
        )
        for report in reports:
            status = "OK" if report.passed else "FAIL"
            unavailable = (
                f" · {len(report.unavailable_run_ids)} unavailable"
                if report.unavailable_run_ids else ""
            )
            print(
                f"{status}\t{report.experiment}\t{report.layout}\t"
                f"{report.path.relative_to(ROOT)}{unavailable}"
            )
            for issue in (*report.structural_issues, *report.evidence_issues):
                print(f"  {issue}")
        if any(not report.passed for report in reports):
            raise SystemExit(1)
        return
    if output_roots:
        app.config["VIEWER_OUTPUT_ROOTS"] = output_roots
    if args.open_browser:
        timer = Timer(
            0.6,
            webbrowser.open_new_tab,
            args=(browser_start_url(
                host=args.host, port=args.port, page=args.page,
            ),),
        )
        timer.daemon = True
        timer.start()
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
