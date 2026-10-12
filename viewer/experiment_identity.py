"""Read-only family and generation navigation over the existing run registry."""
from __future__ import annotations

import html
from collections import Counter
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import urlencode

from flask import Blueprint, abort, jsonify

from ora.orchestrator.experiment_identity import load_catalog
from viewer.registry import RunRegistration, registry
from viewer.ui import render_page


def filter_identity_request(runs: Iterable[RunRegistration], args: Mapping[str, str]) -> list[RunRegistration]:
    family, generation = args.get("family", ""), args.get("generation", "")
    return [r for r in runs if (not family or r.experiment_identity.get("family") == family)
            and (not generation or r.experiment_identity.get("generation_id") == generation)]


def identity_controls(args: Mapping[str, str]) -> str:
    catalog = load_catalog()
    controls = []
    for name, label, values in (
        ("family", "Family", catalog["families"]),
        ("generation", "Generation", {k: v for k, v in catalog["generations"].items()
                                     if not args.get("family") or v["identity"]["family"] == args["family"]}),
    ):
        selected = args.get(name, "")
        options = [f"<option value=''>All {label.lower()} values</option>"]
        for value in values:
            escaped = html.escape(value)
            options.append(f"<option value='{escaped}'{' selected' if value == selected else ''}>{escaped}</option>")
        if selected and selected not in values:
            escaped = html.escape(selected)
            options.append(f"<option value='{escaped}' selected>Unknown: {escaped}</option>")
        controls.append(f"<label>{label}<select name='{name}' onchange='this.form.submit()'>{''.join(options)}</select></label>")
    return "".join(controls)


def identity_label(run: RunRegistration) -> str:
    projection = run.experiment_identity
    family = projection.get("family")
    if not family:
        return ""
    key = projection.get("generation_id") or f"{family} / generation unmapped"
    binding = projection["binding"]
    recorded = projection.get("recorded", {}).get("generation_id")
    note = f" · recorded {html.escape(recorded)}" if recorded else ""
    return (f"<small><a href='/experiments/{html.escape(family)}'>{html.escape(key)}</a>"
            f" · {html.escape(binding)}{note}</small>")


def create_identity_blueprint(run_roots: Callable[[], Iterable[Any]]) -> Blueprint:
    blueprint = Blueprint("experiment_identity", __name__)

    def payload(family: str) -> dict[str, Any]:
        catalog = load_catalog()
        if family not in catalog["families"]:
            abort(404, description="Unknown experiment family")
        discovered = registry(run_roots(), inspect_video=False)
        runs = [r for r in discovered if r.experiment_identity.get("family") == family]
        # Runs this family recorded that a reviewed migration shows under another family.
        moved = Counter(r.experiment_identity["family"] for r in discovered
                        if r.experiment_identity.get("recorded", {}).get("family") == family
                        and r.experiment_identity.get("family") != family)
        return {"family": family, **catalog["families"][family],
                "generations": {k: v for k, v in catalog["generations"].items()
                                if v["identity"]["family"] == family},
                "runs": [r.public() for r in runs], "migrated_runs": dict(moved)}

    @blueprint.get("/api/experiments/<family>")
    def api_family(family: str):
        return jsonify(payload(family))

    @blueprint.get("/experiments")
    def families() -> str:
        catalog = load_catalog()
        rows = "".join(f"<li><a href='/experiments/{html.escape(k)}'>{html.escape(k)}</a>"
                       f" — {html.escape(v['title'])}</li>" for k, v in catalog["families"].items())
        return render_page("Experiment families", "<nav><a href='/'>Runs</a></nav><h1>Experiment families</h1>"
                           f"<ul>{rows}</ul><p>Generations and forks share a family. Exact runs retain their original evidence identities.</p>")

    @blueprint.get("/experiments/<family>")
    def family_page(family: str) -> str:
        value = payload(family)
        counts = Counter(r["experiment_identity"].get("generation_id") for r in value["runs"])
        rows = []
        for key, node in value["generations"].items():
            url = "/?" + urlencode({"view": "all", "family": family, "generation": key})
            parents = ", ".join(node["identity"]["parents"]) or "root"
            rows.append(f"<tr><td><a href='{html.escape(url)}'>{html.escape(key)}</a></td>"
                        f"<td>{html.escape(parents)}</td><td>{html.escape(node['status'])}</td>"
                        f"<td>{html.escape(node['label'])}</td><td>{counts[key]}</td></tr>")
        body = ("<nav class='nav'><a href='/experiments'>Families</a><a href='/'>Runs</a>"
                f"<a href='/api/experiments/{html.escape(family)}'>JSON</a></nav>"
                f"<h1>{html.escape(family)} · {html.escape(value['title'])}</h1>"
                f"<p>{html.escape(value['description'])}</p>"
                "<p>Counts include discovered attempts and diagnostics. Compare official outcomes within exact treatments; "
                "historical generations can contain multiple recorded source revisions.</p>"
                "<div class='table-wrap'><table><thead><tr><th>Generation</th><th>Parents</th><th>Status</th>"
                "<th>Description</th><th>Runs</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
                f"<p>{counts[None]} runs have family membership with unresolved generation.</p>"
                + "".join(f"<p>{n} runs recorded under {html.escape(family)} are shown under "
                          f"<a href='/experiments/{html.escape(other)}'>{html.escape(other)}</a> "
                          "by reviewed migration; their plans are unchanged.</p>"
                          for other, n in sorted(value["migrated_runs"].items())))
        return render_page(family, body, wide=True)

    return blueprint
