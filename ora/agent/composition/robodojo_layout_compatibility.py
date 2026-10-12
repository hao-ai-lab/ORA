"""Prove a reviewed RoboDojo contract holds on every scheduled layout, or on one other layout.

A task-contract record reviews one layout. A RunPlan whose persistent session
schedules other layouts (``runtime.layout_schedule``) may deliver that
contract only when the human-reviewed layout catalog names each scheduled
layout unchanged in the task's reviewed pool, and the catalog's hash-bound
contract appendix extends this exact record to each of them. A row may bind
the asset bytes the official evaluator reads in that layout (for example each
bottle model's metadata); the disclosure carries them so fresh preflight
re-hashes them like the record's own dependencies. Reads repository JSON only; Agent composition imports no Environment module, so the catalog
path and schemas are repeated here and a test holds them equal.

A one-round, unscheduled episode on an official layout other than the
record's own (EXP14A g002) is the second case: the reviewed layout-portability
table (``LAYOUT_PORTABILITY_PATH``) must list the task with this exact record
and contract-layer hash, and the episode's layout file and bytes among its
audited layouts. It also says where the episode's instruction comes from:
``fixed`` (the record's own) or ``layout_readback`` (the lane gate's scene-smoke
readback of that layout, for tasks whose instruction is generated from it).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

LAYOUT_CATALOG_PATH = "ora/environment/benchmarks/robodojo-layout-pools-v1.json"
CATALOG_SCHEMA = "ora.robodojo-layout-pools.v1"
APPENDIX_SCHEMA = "ora.robodojo-task-contract-layout-appendix.v1"
REVIEWED = "reviewed"
LAYOUT_PORTABILITY_PATH = "ora/policy/environments/robodojo/task-contracts/layout-portability-v2.json"
PORTABILITY_SCHEMA = "ora.robodojo-task-contract-layout-portability.v1"
#: Where a single-layout episode's instruction comes from (the layout-portability table).
INSTRUCTION_FIXED, INSTRUCTION_LAYOUT_READBACK = "fixed", "layout_readback"
_SHA256_HEX = frozenset("0123456789abcdef")


def _asset_dependencies(row: Mapping[str, Any], task: str) -> list[dict[str, str]]:
    dependencies = row.get("evaluator_asset_dependencies", [])
    if not isinstance(dependencies, list) or any(
            not isinstance(item, Mapping) or set(item) != {"root", "path", "sha256"} or item["root"] != "assets"
            or not str(item["path"]).startswith("Object/") or ".." in Path(str(item["path"])).parts
            or len(str(item["sha256"])) != 64 or not set(str(item["sha256"])) <= _SHA256_HEX
            for item in dependencies):
        raise RuntimeError(f"a {task} contract layout appendix row binds malformed evaluator assets")
    return [dict(item) for item in dependencies]


def _read(root: Path, relative: str) -> bytes:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise RuntimeError(f"RoboDojo layout review file is missing or escapes the repository: {relative}")
    return path.read_bytes()


def verify_scheduled_contract_layouts(
    runtime: Mapping[str, Any], *, record_path: str, project_root: Path,
) -> dict[str, Any]:
    """Return the disclosure, or refuse a schedule the reviewed contract does not cover."""

    task = str(runtime.get("task_variant"))
    schedule = runtime["layout_schedule"]
    data = _read(project_root, LAYOUT_CATALOG_PATH)
    catalog = json.loads(data)
    if (catalog.get("schema") != CATALOG_SCHEMA
            or schedule["selection"]["catalog"] != LAYOUT_CATALOG_PATH
            or schedule["selection"]["catalog_sha256"] != hashlib.sha256(data).hexdigest()):
        raise RuntimeError("runtime.layout_schedule was resolved from another layout catalog revision")
    entry = (catalog.get("tasks") or {}).get(task)
    if not isinstance(entry, Mapping) or entry.get("status") != REVIEWED:
        raise RuntimeError(f"{task} has no reviewed layout pool; its scheduled layouts cannot carry its contract")
    if schedule["eval_seed"] != catalog.get("eval_seed") or schedule["eval_seed"] != runtime.get("eval_seed"):
        raise RuntimeError("runtime.layout_schedule evaluation seed differs from the reviewed catalog")
    pool = {item.get("layout_id"): item for item in entry.get("pool", [])}
    for scheduled in schedule["rounds"]:
        if pool.get(scheduled["layout_id"]) != dict(scheduled):
            raise RuntimeError(
                f"scheduled layout {scheduled['layout_id']} of {task} is not an unchanged reviewed pool member")
    reference = entry.get("contract_appendix")
    if not isinstance(reference, str):
        raise RuntimeError(f"the {task} layout pool names no contract appendix path")
    # Catalog and appendix name each other and the record by path; policy.lock
    # carries their integrity and the disclosure records the hashes read here.
    from .locked_assets import locked_asset
    appendix_asset = locked_asset(project_root, reference, label=f"the {task} contract layout appendix",
                                  error=RuntimeError)
    appendix = json.loads(appendix_asset.data)
    if (appendix.get("schema") != APPENDIX_SCHEMA or appendix.get("task") != task
            or appendix.get("contract_record") != record_path):
        raise RuntimeError(f"the {task} contract layout appendix does not extend this reviewed contract record")
    audited = {row.get("ordinal"): row for row in appendix.get("layouts", [])}
    assets: dict[str, dict[str, str]] = {}
    for scheduled in schedule["rounds"]:
        row = audited.get(scheduled["layout_id"])
        if (not isinstance(row, Mapping) or row.get("compatible") is not True
                or row.get("layout_file") != scheduled["layout_file"]
                or row.get("layout_sha256") != scheduled["layout_sha256"]):
            raise RuntimeError(
                f"scheduled layout {scheduled['layout_id']} of {task} is not audited compatible with its contract")
        for item in _asset_dependencies(row, task):
            if assets.setdefault(item["path"], item) != item:
                raise RuntimeError(f"the {task} contract layout appendix binds {item['path']} twice differently")
    return {
        "catalog": LAYOUT_CATALOG_PATH, "catalog_sha256": schedule["selection"]["catalog_sha256"],
        "appendix": appendix_asset.relative, "appendix_sha256": appendix_asset.sha256,
        "scheduled_layouts": [
            {"layout_id": item["layout_id"], "layout_sha256": item["layout_sha256"]}
            for item in schedule["rounds"]
        ],
        "evaluator_dependencies": [assets[path] for path in sorted(assets)],
    }


def portable_layout(task: str, layout_id: int, *, project_root: Path,
                    registry: str | None = None) -> dict[str, Any]:
    """The audited portability row of ``task`` on official layout ``layout_id``, or a refusal.

    Returns the table's entry fields for that layout: ``record``,
    ``contract_layer_sha256``, ``instruction`` (``fixed`` or
    ``layout_readback``), ``layout`` (``{layout_id, layout_file,
    layout_sha256}``) and the table's own path and hash.
    """

    from .locked_assets import locked_asset
    asset = locked_asset(project_root, LAYOUT_PORTABILITY_PATH, label="the RoboDojo layout-portability table",
                         error=RuntimeError)
    table = json.loads(asset.data)
    if table.get("schema") != PORTABILITY_SCHEMA or (registry is not None and table.get("registry") != registry):
        raise RuntimeError("the RoboDojo layout-portability table is malformed or audits another registry")
    entry = (table.get("tasks") or {}).get(task)
    if not isinstance(entry, Mapping):
        raise RuntimeError(f"{task} is not audited for layout portability; its reviewed contract holds on its "
                           "reviewed layout only")
    if entry.get("instruction") not in (INSTRUCTION_FIXED, INSTRUCTION_LAYOUT_READBACK):
        raise RuntimeError(f"the {task} layout-portability entry names no instruction source")
    rows = [row for row in entry.get("layouts", []) if isinstance(row, Mapping) and row.get("layout_id") == layout_id]
    if type(layout_id) is not int or len(rows) != 1:
        raise RuntimeError(f"layout {layout_id!r} of {task} is not an audited portable layout of its reviewed contract")
    return {"record": entry.get("record"), "contract_layer_sha256": entry.get("contract_layer_sha256"),
            "instruction": entry["instruction"], "layout": dict(rows[0]), "eval_seed": table.get("eval_seed"),
            "table": asset.relative, "table_sha256": asset.sha256}


def verify_single_layout_contract(
    runtime: Mapping[str, Any], *, record_path: str, record: Mapping[str, Any], project_root: Path,
    registry: str,
) -> dict[str, Any]:
    """Return the disclosure, or refuse a one-round episode whose layout the reviewed contract does not cover."""

    task, layout_id = str(runtime.get("task_variant")), runtime.get("layout_id")
    row = portable_layout(task, layout_id, project_root=project_root, registry=registry)
    if row["record"] != record_path or row["contract_layer_sha256"] != record["contract_layer"]["sha256"]:
        raise RuntimeError(f"the {task} layout-portability entry audited another contract record or layer")
    if row["eval_seed"] != runtime.get("eval_seed") or row["layout"] != {
            "layout_id": layout_id, "layout_file": runtime.get("layout_file"),
            "layout_sha256": runtime.get("layout_sha256")}:
        raise RuntimeError(f"layout {layout_id} of {task} differs from its audited file or bytes")
    return {"table": row["table"], "table_sha256": row["table_sha256"], "layout_id": layout_id,
            "layout_sha256": row["layout"]["layout_sha256"], "instruction_source": row["instruction"]}


__all__ = [
    "APPENDIX_SCHEMA", "CATALOG_SCHEMA", "INSTRUCTION_FIXED", "INSTRUCTION_LAYOUT_READBACK", "LAYOUT_CATALOG_PATH",
    "LAYOUT_PORTABILITY_PATH", "PORTABILITY_SCHEMA", "portable_layout", "verify_scheduled_contract_layouts",
    "verify_single_layout_contract",
]
