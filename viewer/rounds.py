"""Read-only round ledgers and exact-run comparison matrices."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import quote


ROUND_SCHEMA = "ora.viewer-round-ledger.v1"
PERSISTENT_CADENCE = "persistent_agent_k_rounds_v1"


def load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def iter_records(path: Path):
    if not path.is_file():
        return
    with path.open(encoding="utf-8", errors="replace") as source:
        for line_number, line in enumerate(source, 1):
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if isinstance(value, dict):
                yield line_number, value


def integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def scene_label(scene: Any) -> str | None:
    """`layout 37 · d57e3333` for a scheduled round's recorded scene, else None."""
    if not isinstance(scene, Mapping) or integer(scene.get("layout_id")) is None:
        return None
    digest = str(scene.get("layout_sha256") or "")[:8]
    return f"layout {scene['layout_id']}" + (f" · {digest}" if digest else "")


def round_ledger(root: Path) -> dict[str, Any]:
    """Keep planned, observed, completed and unscored rounds distinct.

    Persisted result rows have priority. Before the result is closed, only
    round_finished and published observation notices establish provisional rows.
    No repeated run is inferred to be a continuation of another Session.
    """
    plan, result, status = (load_object(root / name) for name in (
        "plan.json", "result.json", "status.json",
    ))
    session = result.get("session") or plan.get("session") or {}
    session = session if isinstance(session, Mapping) else {}
    raw_rounds = result.get("rounds") or []
    raw_rounds = raw_rounds if isinstance(raw_rounds, list) else []
    planned = integer(session.get("rounds"))
    is_session = session.get("cadence") == PERSISTENT_CADENCE or bool(raw_rounds)
    rows: dict[int, dict[str, Any]] = {}
    issues: list[str] = []
    for ordinal, item in enumerate(raw_rounds):
        if not isinstance(item, Mapping):
            issues.append(f"invalid_round:{ordinal}")
            continue
        index = integer(item.get("round_index"))
        index = ordinal if index is None else index
        if index < 0 or index in rows:
            issues.append(f"duplicate_or_invalid_round:{index}")
            continue
        rows[index] = {**item, "round_index": index, "source": "result.json", "provisional": False}

    run_state = str(status.get("state") or result.get("state") or "unknown")
    if is_session and run_state != "completed":
        for line, event in iter_records(root / "events.jsonl"):
            if event.get("type") == "round_finished":
                item = event.get("result")
                index = integer(event.get("round_index"))
                if (isinstance(item, Mapping) and index is not None and index >= 0
                        and (index not in rows or rows[index].get("observation_only"))):
                    rows[index] = {**item, "round_index": index, "source": f"runtime:{line}", "provisional": True}
            if event.get("type") != "observation":
                continue
            context = event.get("context") or {}
            budget = context.get("remaining_budget") or {}
            ordinal = integer(budget.get("session_round"))
            state = event.get("policy_state") or context.get("policy_state") or {}
            step = integer(state.get("step"))
            if ordinal is None or ordinal <= 0 or ordinal - 1 in rows and not rows[ordinal - 1].get("observation_only"):
                continue
            index = ordinal - 1
            prior = rows.get(index, {})
            start = (prior.get("policy_state_steps") or {}).get("start", step)
            rows[index] = {
                "round_index": index, "state": "running" if run_state == "running" else "incomplete",
                "official_success": None, "source": f"runtime:{line}", "provisional": True,
                "observation_only": True, "policy_state_steps": {"start": start, "end_exclusive": step + 1 if step is not None else None},
            }

    if not is_session:
        rows[0] = {**result, "round_index": 0, "state": run_state, "source": "result.json", "provisional": not bool(result)}
        planned = 1
    extent = max(rows, default=-1) + 1
    planned = planned if planned is not None and planned > 0 else extent
    if planned < extent:
        issues.append("recorded_round_exceeds_plan")
    # Bound a malformed plan without silently claiming a different K.
    if max(planned, extent) > 500:
        raise ValueError("Viewer round ledger exceeds 500 rounds")
    projected = []
    for index in range(max(planned, extent)):
        item = rows.get(index)
        verdict = item.get("official_success") if item else None
        verdict = verdict if isinstance(verdict, bool) else None
        state = str(item.get("state") or "unknown") if item else "not_run"
        outcome = (
            "success" if verdict is True else "failure" if verdict is False else
            "not_run" if item is None else "running" if state == "running" else
            "incomplete" if state in {"incomplete", "faulted", "failed", "stopped"} else "unscored"
        )
        projected.append({
            "round_index": index, "round_ordinal": index + 1, "state": state,
            "outcome": outcome, "official_success": verdict,
            "decisions": item.get("decisions") if item else None,
            "refused_proposals": item.get("refused_proposals") if item else None,
            "model_calls": item.get("model_calls") if item else None,
            "wall_duration_s": item.get("wall_duration_s") if item else None,
            "termination": item.get("termination") if item else None,
            "policy_state_steps": item.get("policy_state_steps") or {} if item else {},
            "scene": item.get("scene") if item and isinstance(item.get("scene"), Mapping) else None,
            "source": item.get("source") if item else "plan.json",
            "provisional": bool(item and item.get("provisional")),
        })
    successes = [row for row in projected if row["official_success"] is True]
    observed = [row for row in projected if row["outcome"] != "not_run"]
    return {
        "schema": ROUND_SCHEMA, "run_id": root.name, "is_session": is_session,
        "cadence": session.get("cadence"), "planned_rounds": planned,
        "recorded_rounds": len(observed),
        "completed_rounds": sum(row["state"] == "completed" for row in observed),
        "scored_rounds": sum(row["official_success"] is not None for row in observed),
        "success_count": len(successes),
        "first_success_round": successes[0]["round_ordinal"] if successes else None,
        "aggregate_official_success": None if is_session else result.get("official_success"),
        "rounds": projected, "issues": issues,
    }


def render_round_matrix(registrations: Iterable[Any]) -> str:
    """One row per exact run; no implicit treatment pooling or best-run choice."""
    ledgers = [(run, round_ledger(run.root)) for run in registrations]
    if not any(ledger["is_session"] for _run, ledger in ledgers):
        return ""
    width = max(len(ledger["rounds"]) for _run, ledger in ledgers)
    headers = "".join(f"<th scope='col'>R{i + 1}</th>" for i in range(width))
    body = []
    for run, ledger in ledgers:
        encoded = quote(run.run_id, safe="")
        cells = []
        for index in range(width):
            row = next((item for item in ledger["rounds"] if item["round_index"] == index), None)
            if row is None:
                cells.append("<td class='round-na' aria-label='Outside this run’s planned rounds'>—</td>")
                continue
            label = row["outcome"].replace("_", " ")
            refused = f", {row['refused_proposals']} refused" if row.get("refused_proposals") else ""
            action_count = html.escape(f"{row['decisions']} actions{refused}") if row["decisions"] is not None else "—"
            detail = f"{label}; {row['model_calls']} calls; {row['termination'] or row['state']}"
            provisional = " · provisional" if row["provisional"] else ""
            scene = scene_label(row.get("scene"))
            scene = f"<small>{html.escape(scene)}</small>" if scene else ""
            content = f"<strong>{label}</strong><small>{action_count}{provisional}</small>{scene}"
            if row["outcome"] != "not_run":
                content = f"<a href='/run/{encoded}?round={index + 1}#review-player' title='{html.escape(detail, quote=True)}'>{content}</a>"
            cells.append(f"<td class='round-cell {row['outcome']}'>{content}</td>")
        first = f"first success R{ledger['first_success_round']}" if ledger["first_success_round"] else "no recorded success"
        kind = "persistent Session" if ledger["cadence"] == PERSISTENT_CADENCE else "recorded rounds" if ledger["is_session"] else "single episode"
        body.append(
            "<tr><th scope='row' class='round-run'>"
            f"<a href='/run/{encoded}'>{html.escape(run.task)}</a>"
            f"<small>seed {html.escape(str(run.seed))} · {html.escape(run.config)}</small>"
            f"<details><summary>{html.escape(run.category)} · {kind}</summary><code>{html.escape(run.run_id)}</code>"
            f"<small>source {html.escape(str(run.source_commit or 'unknown'))}</small></details></th>"
            + "".join(cells)
            + f"<td class='round-summary'><strong>{ledger['success_count']}/{ledger['scored_rounds']} scored</strong>"
            f"<small>{ledger['completed_rounds']}/{ledger['planned_rounds']} completed · {first}</small></td></tr>"
        )
    return (
        "<section class='round-comparison' aria-label='Results by round'><header><div>"
        "<h2>Results by round</h2></div>"
        "<p>Choose a result to open that round’s video and recorded process. "
        "Each row is one run.</p></header>"
        "<div class='round-table-scroll'><table><thead><tr><th scope='col'>Task / exact run</th>"
        + headers + "<th scope='col'>Session progress</th></tr></thead><tbody>" + "".join(body)
        + "</tbody></table></div></section>"

    )
