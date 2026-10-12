"""LIBERO / LIBERO-Pro atlas routes and presentation projection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import quote, urlencode

from flask import Blueprint, abort, jsonify, render_template, request, send_file

from viewer.libero_pro_catalog import (
    CODE_ROOT,
    COLLECTION_ORDER,
    COLLECTIONS,
    DATA_ROOT,
    FAMILY_LABELS,
    FAMILY_ORDER,
    LEGACY_ENVIRONMENT,
    LIBERO_PRO_CODE_REVISION,
    LIBERO_PRO_DATA_REVISION,
    MIGRATED_FROM_COMMIT,
    PERTURBATIONS,
    PERTURBATION_ORDER,
    TASK_BY_KEY,
    TASKS,
    LiberoProTask,
    catalog_payload,
    robustness_assets,
)
from viewer.libero_pro_results import published_results_payload


PageRenderer = Callable[..., str]
PreviewLoader = Callable[[], dict[str, Any]]
TaskRunsUrl = Callable[..., str]


def _gallery_url(
    path_family: str | None = None, task_id: int | None = None, **query: str,
) -> str:
    path = "/gallery/libero"
    if path_family is not None and task_id is not None:
        path += f"/{path_family}/{task_id}"
    values = {key: value for key, value in query.items() if value != ""}
    return path + (f"?{urlencode(values)}" if values else "")


def _matches(task: LiberoProTask, query: str, family: str) -> bool:
    if family and task.family != family:
        return False
    needle = query.casefold().strip()
    haystack = " ".join((
        task.family,
        task.base_suite,
        task.collection,
        task.scene or "",
        task.name,
        task.canonical_instruction,
        *(variant.instruction for variant in task.core_variants),
    )).casefold()
    return not needle or needle in haystack


def _preview_url(record: Mapping[str, Any] | None) -> str | None:
    if record is None:
        return None
    return "/gallery/libero-pro/preview/" + quote(
        str(record["image"]), safe="/",
    )


def _context(
    *,
    selected_family: str | None,
    selected_task_id: int | None,
    preview_loader: PreviewLoader,
    task_runs_url: TaskRunsUrl,
) -> dict[str, Any]:
    query = request.args.get("q", "").strip()
    family = request.args.get("family", "")
    if family not in FAMILY_LABELS:
        family = ""
    perturbation = request.args.get("perturbation", "position")
    if perturbation not in PERTURBATIONS:
        perturbation = "position"
    if selected_family is not None:
        selected = TASK_BY_KEY.get((selected_family, selected_task_id))
        if selected_family not in FAMILY_LABELS or selected is None:
            abort(404)
    else:
        selected = TASK_BY_KEY[(family or "object", 0)]
    visible = [task for task in TASKS if _matches(task, query, family)]
    if selected_family is None and visible and selected not in visible:
        selected = visible[0]

    preview_status = preview_loader()
    preview_index = preview_status["index"]

    def record(layer: str, suite: str, task_id: int) -> dict[str, Any] | None:
        value = preview_index.get(f"{layer}:{suite}:task-{task_id:02d}")
        return value if isinstance(value, dict) else None

    retained = {"q": query, "family": family, "perturbation": perturbation}
    task_cards = []
    for task in visible:
        task_cards.append({
            "task": task,
            "family": FAMILY_LABELS[task.family],
            "href": _gallery_url(task.family, task.task_id, **retained),
            "preview_url": _preview_url(record(
                "base", task.base_suite, task.task_id,
            )),
        })
    variants = []
    if selected.pro_anchor:
        for variant_id in PERTURBATION_ORDER:
            variant = selected.variant(variant_id)
            variants.append({
                "id": variant_id,
                "variant": variant,
                "presentation": PERTURBATIONS[variant_id],
                "href": _gallery_url(
                    selected.family, selected.task_id,
                    q=query, family=family, perturbation=variant_id,
                ),
                "runs_url": task_runs_url(variant.suite, selected.task_id),
                "preview_url": _preview_url(
                    record("core", variant.suite, selected.task_id)
                ),
            })
    robustness = []
    for asset in robustness_assets(selected):
        robustness.append({
            "asset": asset,
            "runs_url": task_runs_url(
                asset["experiment_config"], selected.task_id,
            ),
            "preview_url": _preview_url(record(
                "robustness", asset["experiment_config"], selected.task_id,
            )),
        })
    selected_variant = selected.variant(perturbation) if selected.pro_anchor else None
    selected_preview_url = _preview_url(record(
        "core" if selected_variant is not None else "base",
        selected_variant.suite if selected_variant is not None else selected.base_suite,
        selected.task_id,
    ))
    task_groups = []
    for identifier in FAMILY_ORDER:
        cards = [item for item in task_cards if item["task"].family == identifier]
        if cards:
            task_groups.append({
                "id": identifier,
                "family": FAMILY_LABELS[identifier],
                "cards": cards,
            })
    published_results = published_results_payload()
    return {
        "query": query,
        "family_filter": family,
        "perturbation": perturbation,
        "selected": selected,
        "selected_family": FAMILY_LABELS[selected.family],
        "selected_variant": selected_variant,
        "selected_perturbation": (
            PERTURBATIONS[perturbation] if selected_variant is not None else None
        ),
        "selected_preview_url": selected_preview_url,
        "task_cards": task_cards,
        "task_groups": task_groups,
        "variants": variants,
        "robustness": robustness,
        "families": [
            {
                "id": identifier,
                "count": sum(task.family == identifier for task in TASKS),
                **FAMILY_LABELS[identifier],
            }
            for identifier in FAMILY_ORDER
        ],
        "collections": [
            {
                "id": collection_id,
                "families": [
                    {
                        "id": identifier,
                        "count": sum(task.family == identifier for task in TASKS),
                        **FAMILY_LABELS[identifier],
                    }
                    for identifier in FAMILY_ORDER
                    if FAMILY_LABELS[identifier]["collection"] == collection_id
                ],
                **COLLECTIONS[collection_id],
            }
            for collection_id in COLLECTION_ORDER
        ],
        "preview_status": preview_status,
        "preview_label": (
            f"{preview_status['available']} / {preview_status['expected']} "
            "seed-0 previews loaded"
        ),
        "code_root": CODE_ROOT,
        "data_root": DATA_ROOT,
        "code_revision": LIBERO_PRO_CODE_REVISION,
        "data_revision": LIBERO_PRO_DATA_REVISION,
        "migrated_from_commit": MIGRATED_FROM_COMMIT,
        "legacy_environment": LEGACY_ENVIRONMENT,
        "published_results": published_results,
        "result_cohorts": published_results["cohorts"],
        "result_by_key": {
            row["key"]: row
            for cohort in published_results["cohorts"]
            for row in cohort["methods"]
        },
        "openeta_boundary": published_results["openeta"],
    }


def create_libero_pro_blueprint(
    *, page_renderer: PageRenderer, preview_loader: PreviewLoader,
    task_runs_url: TaskRunsUrl,
) -> Blueprint:
    blueprint = Blueprint("libero_pro_gallery", __name__)

    def gallery(family: str | None = None, task_id: int | None = None) -> str:
        body = render_template(
            "libero_pro_gallery.html",
            **_context(
                selected_family=family,
                selected_task_id=task_id,
                preview_loader=preview_loader,
                task_runs_url=task_runs_url,
            ),
        )
        return page_renderer("LIBERO / LIBERO-Pro Atlas", body, wide=True)

    blueprint.add_url_rule(
        "/gallery/libero-pro", endpoint="gallery", view_func=gallery,
    )
    blueprint.add_url_rule(
        "/gallery/libero", endpoint="libero_gallery", view_func=gallery,
    )
    blueprint.add_url_rule(
        "/gallery/libero-pro/<family>/<int:task_id>",
        endpoint="task",
        view_func=gallery,
    )
    blueprint.add_url_rule(
        "/gallery/libero/<family>/<int:task_id>",
        endpoint="libero_task",
        view_func=gallery,
    )

    @blueprint.get("/gallery/libero-pro/preview/<path:relative>")
    def preview(relative: str):
        status = preview_loader()
        allowed = {
            str(record.get("image"))
            for record in status["index"].values()
            if isinstance(record, Mapping)
        }
        if relative not in allowed:
            abort(404)
        root = Path(status["root"]).resolve()
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            abort(404)
        return send_file(
            path, mimetype="image/png", conditional=True, max_age=24 * 60 * 60,
        )

    @blueprint.get("/api/libero-pro/tasks")
    def tasks_api():
        return jsonify(catalog_payload())

    blueprint.add_url_rule(
        "/api/libero/tasks", endpoint="libero_tasks_api", view_func=tasks_api,
    )

    @blueprint.get("/api/libero-pro/previews")
    def previews_api():
        status = preview_loader()
        return jsonify({
            "ready": status["ready"],
            "available": status["available"],
            "expected": status["expected"],
            "missing": status["missing"],
            "manifest": status["manifest"],
        })

    @blueprint.get("/api/libero-pro/results")
    def results_api():
        return jsonify(published_results_payload())

    return blueprint


__all__ = ["create_libero_pro_blueprint"]
