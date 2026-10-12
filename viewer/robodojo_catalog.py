"""Read-only RoboDojo task catalog for the Viewer.

Unlike the RoboTwin catalog, which hard-codes upstream metadata and links to
videos hosted on the benchmark's own site, RoboDojo's inventory is derived from
the checkout: `scripts/internal/task_inventory.py` is the benchmark's own
authority on which tasks exist, which capability dimension each belongs to, and
whether a task is runnable. Reading it keeps this module correct across upstream
revisions instead of drifting from a copied list.

Demonstration media is local. The `RoboDojo-Benchmark/RoboDojo` dataset ships
per-episode HDF5 files carrying a natural-language `instruction` plus head and
wrist preview videos; when that data is present the catalog exposes it, so a
reader can see what a task actually looks like rather than only its name.

The larger `RoboDojo_lerobot_v30_video` release is the *simulation* dataset
(35 tasks x 100 episodes) repacked in LeRobot v3.0 layout — the real-robot
recordings live in a sibling `RoboDojo_real` directory and are not fetched
here. LeRobot v3.0 packs many episodes into one container mp4 addressed by
timestamp, so nothing in it is directly playable. A preview pass slices one
episode per task out of its container; this module binds those clips to tasks
by exact instruction text, read from each task's own `gen_instruction`, rather
than by guessing at filename similarity.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

CATALOG_SCHEMA = "ora.robodojo-task-catalog.v1"

# Display labels only. Which dimensions exist is read from the inventory — a
# dimension absent from this table still renders, under its raw name. Upstream
# spells the key `long-horizon`; an earlier `long_horizon` entry here silently
# dropped all eight of those tasks from the gallery, because the render loop
# walked this table instead of the data.
DIMENSION_LABELS = {
    "generalization": "Generalization",
    "memory": "Memory",
    "precision": "Precision",
    "long-horizon": "Long-Horizon",
    "open": "Open",
}
DIMENSION_ORDER = tuple(DIMENSION_LABELS)

_ROOT_ENV = "ROBODOJO_ROOT"
_DATA_ENV = "ROBODOJO_DATA_ROOT"
_PREVIEW_ENV = "ROBODOJO_PREVIEW_ROOT"


def _title(task_id: str) -> str:
    return " ".join(word.capitalize() for word in task_id.split("_"))


@dataclass(frozen=True)
class RoboDojoTask:
    id: str
    title: str
    dimension: str
    variant: str
    runnable: bool
    config: str
    module: str
    instruction: str
    demonstrations: tuple[dict[str, Any], ...]

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "dimension": self.dimension,
            "dimension_label": DIMENSION_LABELS.get(self.dimension, self.dimension),
            "variant": self.variant,
            "runnable": self.runnable,
            "config": self.config,
            "module": self.module,
            "instruction": self.instruction,
            "demonstrations": list(self.demonstrations),
        }


def _root() -> Path | None:
    value = os.environ.get(_ROOT_ENV)
    if not value:
        return None
    root = Path(value)
    return root if (root / "scripts/internal/task_inventory.py").is_file() else None


def _inventory(root: Path) -> list[dict[str, Any]]:
    """Ask RoboDojo itself what tasks exist. Never hard-code the list."""
    completed = subprocess.run(
        ["python3", "scripts/internal/task_inventory.py", "--format", "json"],
        cwd=root, capture_output=True, text=True, timeout=120, check=False,
    )
    if completed.returncode != 0:
        return []
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return []
    tasks = payload.get("tasks")
    return tasks if isinstance(tasks, list) else []


def _demonstrations(task_id: str) -> tuple[dict[str, Any], ...]:
    """Locate local demonstration episodes, if the dataset has been fetched.

    Episodes carry their own natural-language instruction, so the task a
    recording belongs to is read from the file rather than inferred from the
    directory name.
    """
    value = os.environ.get(_DATA_ENV)
    if not value:
        return ()
    root = Path(value)
    if not root.is_dir():
        return ()
    found: list[dict[str, Any]] = []
    for episode in sorted(root.glob("**/data/episode_*.hdf5")):
        preview = episode.parent.parent / "preview_video"
        stem = episode.stem
        views = {
            path.name.rsplit("_cam_", 1)[-1].removesuffix(".mp4"): path.name
            for path in sorted(preview.glob(f"{stem}_cam_*.mp4"))
        }
        if not views:
            continue
        found.append({
            "episode": stem,
            "hdf5": str(episode),
            "preview_root": str(preview),
            "views": views,
        })
    return tuple(found)


def _instruction(root: Path, module: str) -> str:
    """Read a task's natural-language instruction from its own source.

    `gen_instruction` returns a list of templates. Parsing it statically avoids
    importing the task, which would drag in the simulator. Tasks that build the
    string at runtime (the language-conditioned variants) yield nothing, and are
    simply left without an instruction rather than guessed at.
    """
    path = root / module
    if not path.is_file():
        return ""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError):
        return ""
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "gen_instruction"):
            continue
        for assign in ast.walk(node):
            if not (isinstance(assign, ast.Assign) and isinstance(assign.value, ast.List)):
                continue
            texts = [
                element.value for element in assign.value.elts
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            ]
            if texts:
                return texts[0]
    return ""


@lru_cache(maxsize=1)
def _lerobot_previews() -> dict[str, list[dict[str, Any]]]:
    """Sliced preview clips, keyed by the instruction they were recorded for."""
    value = os.environ.get(_PREVIEW_ENV)
    if not value:
        return {}
    index = Path(value) / "index.json"
    if not index.is_file():
        return {}
    try:
        entries = json.loads(index.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    by_instruction: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        instruction = str(entry.get("instruction") or "")
        views = entry.get("views") or {}
        if not instruction or not views:
            continue
        by_instruction.setdefault(instruction, []).append({
            "episode": f"sim_{entry['task_id']}",
            "source": "simulation",
            "dataset": "RoboDojo_lerobot_v30_video",
            "episodes_available": entry.get("episodes_available"),
            "preview_root": str(Path(value) / entry["task_id"]),
            "views": views,
        })
    return by_instruction


@lru_cache(maxsize=1)
def tasks() -> tuple[RoboDojoTask, ...]:
    root = _root()
    if root is None:
        return ()
    demos_by_nothing = _demonstrations("")
    previews = _lerobot_previews()
    collected: list[RoboDojoTask] = []
    for entry in _inventory(root):
        name = str(entry.get("name") or "")
        if not name:
            continue
        module = str(entry.get("module") or "")
        instruction = _instruction(root, module)
        demonstrations = list(previews.get(instruction, ()))
        # The bundled demo package covers one task only; it is matched by name
        # because that package carries no instruction of its own.
        if name == "stack_bowls":
            demonstrations.extend(demos_by_nothing)
        collected.append(RoboDojoTask(
            id=name,
            title=_title(name),
            dimension=str(entry.get("dimension") or "open"),
            variant=str(entry.get("variant") or "standard"),
            runnable=bool(entry.get("runnable")),
            config=str(entry.get("config") or ""),
            module=module,
            instruction=instruction,
            # Clips are bound by instruction text, not by task name, so a
            # recording is attached only where it genuinely belongs.
            demonstrations=tuple(demonstrations),
        ))
    return tuple(collected)


def catalog_payload() -> dict[str, Any]:
    """Stable read-only JSON served by the Viewer."""
    collected = tasks()
    by_dimension: dict[str, list[str]] = {}
    for task in collected:
        by_dimension.setdefault(task.dimension, []).append(task.id)
    # Known dimensions first, in the documented order; anything upstream adds
    # follows. Nothing is dropped for being unrecognised.
    ordered = [key for key in DIMENSION_ORDER if key in by_dimension]
    ordered += sorted(key for key in by_dimension if key not in DIMENSION_ORDER)
    return {
        "schema": CATALOG_SCHEMA,
        "available": bool(collected),
        "root_env": _ROOT_ENV,
        "data_root_env": _DATA_ENV,
        "dimensions": [
            {
                "id": key,
                "label": DIMENSION_LABELS.get(key, key),
                "tasks": sorted(by_dimension[key]),
            }
            for key in ordered
        ],
        "counts": {
            "tasks": len(collected),
            "runnable": sum(1 for task in collected if task.runnable),
            "with_demonstrations": sum(1 for task in collected if task.demonstrations),
        },
        "tasks": [task.public() for task in collected],
    }


def task(task_id: str) -> RoboDojoTask | None:
    for item in tasks():
        if item.id == task_id:
            return item
    return None
