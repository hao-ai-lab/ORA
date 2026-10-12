"""The human-reviewed RoboDojo layout pools and the default schedule resolver.

Host-side only: the isolated worker never reads the catalog. It answers which
official layouts a persistent session may schedule for a task
(``robodojo-layout-pools-v1.json``: one pool per task in default order, with
its review status and the hash-bound contract appendix that
``agent/composition/robodojo_layout_compatibility.py`` checks at RunSpec
resolution). A task without a pool, or whose pool is still pending review, cannot
be scheduled; nothing here falls back to layout 0.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ora.environment.benchmarks.robodojo import (
    REVISIONS, RoboDojoBenchmarkError, selected_layout,
)
from ora.environment.benchmarks.robodojo_layouts import (
    ENV_SEED_POLICY, LAYOUT_CATALOG_PATH, LAYOUT_SCHEDULE_SCHEMA, validate_layout_schedule,
)

CATALOG_SCHEMA = "ora.robodojo-layout-pools.v1"
#: A pool the resolver may use, and one a survey is still selecting.
REVIEWED, PENDING_REVIEW = "reviewed", "pending_review"
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RoboDojoBenchmarkError(message)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _repository_file(root: Path, relative: str) -> bytes:
    path = (root / relative).resolve()
    _require(path.is_relative_to(root.resolve()) and path.is_file(),
             f"reviewed layout file is missing or escapes the repository: {relative}")
    return path.read_bytes()


def load_layout_catalog(project_root: Path = PROJECT_ROOT) -> tuple[dict[str, Any], str]:
    """The catalog and the sha256 of its exact bytes."""

    data = _repository_file(project_root, LAYOUT_CATALOG_PATH)
    catalog = json.loads(data)
    _require(isinstance(catalog, Mapping) and catalog.get("schema") == CATALOG_SCHEMA
             and catalog.get("robodojo_revision") == REVISIONS["robodojo"]
             and catalog.get("assets_revision") == REVISIONS["assets"]
             and type(catalog.get("eval_seed")) is int
             and isinstance(catalog.get("tasks"), Mapping),
             "the RoboDojo layout catalog is malformed or bound to other sources")
    return catalog, _sha256(data)


def task_pool(catalog: Mapping[str, Any], task: str) -> list[dict[str, Any]]:
    """A task's reviewed pool in default order, or a refusal."""

    entry = catalog["tasks"].get(task)
    _require(isinstance(entry, Mapping),
             f"RoboDojo task {task!r} has no reviewed layout pool; a persistent session of it cannot "
             "start each round on a new layout (select repeat_round0_layout explicitly to replay round 0)")
    _require(entry.get("status") != PENDING_REVIEW,
             f"the layout pool of RoboDojo task {task!r} is pending review; its sessions cannot be "
             "scheduled until the reviewed pool and contract appendix land, and never fall back to layout 0")
    _require(entry.get("status") == REVIEWED, f"the layout pool of {task} has no reviewed status")
    pool = entry.get("pool")
    _require(isinstance(pool, list) and len(pool) >= 2
             and len({item.get("layout_id") for item in pool}) == len(pool),
             f"the reviewed layout pool of {task} is malformed")
    return [dict(item) for item in pool]


def build_layout_schedule(
    task: str, rounds: int, *, asset_root: Path, project_root: Path = PROJECT_ROOT,
    layout_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Resolve one reviewed layout per round, hashed from the pinned Assets.

    The default is the first ``rounds`` pool entries; an explicit selection
    must name distinct pool members. Every file is re-resolved through the
    upstream ordinal and must still hash to the catalog's reviewed bytes.
    """

    _require(type(rounds) is int and rounds >= 2, "a layout schedule needs at least two rounds")
    catalog, catalog_sha256 = load_layout_catalog(project_root)
    pool = {item["layout_id"]: item for item in task_pool(catalog, task)}
    ordered = list(pool)
    selected = list(ordered[:rounds] if layout_ids is None else layout_ids)
    _require(len(selected) == rounds, f"the reviewed {task} pool cannot schedule {rounds} rounds")
    _require(all(type(layout_id) is int and layout_id in pool for layout_id in selected),
             f"an explicit {task} layout selection must stay inside the reviewed pool {ordered}")
    assets = Path(asset_root).resolve()
    entries = []
    for layout_id in selected:
        path = selected_layout(assets, task=task, eval_seed=catalog["eval_seed"], layout_id=layout_id)
        item = pool[layout_id]
        _require(path.relative_to(assets).as_posix() == item["layout_file"]
                 and _sha256(path.read_bytes()) == item["layout_sha256"],
                 f"pinned layout {layout_id} of {task} differs from its reviewed catalog entry")
        entries.append(dict(item))
    schedule = {
        "schema": LAYOUT_SCHEDULE_SCHEMA, "eval_seed": catalog["eval_seed"],
        "env_seed_policy": ENV_SEED_POLICY, "rounds": entries,
        "selection": {"catalog": LAYOUT_CATALOG_PATH, "catalog_sha256": catalog_sha256},
    }
    first = entries[0]
    return validate_layout_schedule(
        schedule, task=task, eval_seed=catalog["eval_seed"], layout_id=first["layout_id"],
        layout_file=first["layout_file"], layout_sha256=first["layout_sha256"], rounds=rounds,
    )


__all__ = [
    "CATALOG_SCHEMA", "PENDING_REVIEW", "REVIEWED", "build_layout_schedule", "load_layout_catalog",
    "task_pool",
]
