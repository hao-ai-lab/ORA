"""The public RoboTwin task instruction a RunSpec delivers, one per task.

``robotwin-instructions-v1.json`` holds the instruction the EXP10A RoboTwin
arms delivered for each of the fifty tasks (moved here from the official suite
package on 2026-10-05, entries unchanged). Two tasks, ``place_a2b_left`` and
``place_a2b_right``, name the objects their seed spawns, so their instruction
is per (task, seed) and comes from a public episode-names mapping
(``{"episodes": {"<task>:s<seed>": {"instruction": ...}}}``).

``robotwin-episode-instructions-unseen-v1.json`` holds the benchmark's own
per-episode instruction for the 250 standard ``demo_randomized`` episodes
(fifty tasks, five seeds each, the seeds RPent evaluates). RoboTwin's
evaluation (``script/eval_policy.py``, ``instruction_type: unseen`` in every
official policy config) obtains the episode's objects and arms from the
expert's ``play_once()`` and draws one sentence from the unseen template and
object-description pools. Each entry pins that draw (RoboTwin's generator under
an RNG seeded from ``<task>:<seed>``) with the episode values it used; the
values were checked against RPent's sentence for the same seed, which names the
same objects from the seen pool. For those episodes the official sentence is
the default instruction; an explicit episode name still overrides it.

This is separate from ``robotwin.py`` because the seed scan
(``orchestrator/robotwin_evaluation.py::source_dependencies``) binds that file
as an execution dependency; the instruction text is not one.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from .robotwin import TASKS

INSTRUCTIONS_PATH = Path(__file__).resolve().with_name("robotwin-instructions-v1.json")
INSTRUCTIONS_SCHEMA = "ora.robotwin-public-instructions.v1"
OFFICIAL_EPISODES_PATH = Path(__file__).resolve().with_name("robotwin-episode-instructions-unseen-v1.json")
OFFICIAL_EPISODES_SCHEMA = "ora.robotwin-episode-instructions.v1"


class RoboTwinInstructionError(ValueError):
    """A RoboTwin task has no public instruction for the requested cell."""


@lru_cache(maxsize=1)
def _table() -> Mapping[str, Mapping[str, Any]]:
    value = json.loads(INSTRUCTIONS_PATH.read_text(encoding="utf-8"))
    if value.get("schema") != INSTRUCTIONS_SCHEMA or set(value.get("tasks", {})) != set(TASKS):
        raise RoboTwinInstructionError("the RoboTwin instruction table does not cover the fifty-task roster")
    return value["tasks"]


def public_instructions() -> dict[str, dict[str, Any]]:
    """Every task's entry: ``instruction`` (None when ``seed_named``) and ``seed_named``."""
    return {task: dict(entry) for task, entry in _table().items()}


def seed_named(task: str) -> bool:
    """True when the task's instruction names the objects its seed spawns."""
    return bool(_entry(task).get("seed_named"))


def _entry(task: str) -> Mapping[str, Any]:
    entry = _table().get(task)
    if entry is None:
        raise RoboTwinInstructionError(f"unknown RoboTwin task: {task!r}")
    return entry


@lru_cache(maxsize=1)
def _official_episodes() -> Mapping[str, Any]:
    value = json.loads(OFFICIAL_EPISODES_PATH.read_text(encoding="utf-8"))
    if value.get("schema") != OFFICIAL_EPISODES_SCHEMA:
        raise RoboTwinInstructionError("the official RoboTwin episode-instruction table has an unknown schema")
    return value


def official_episode_instruction(task: str, seed: int | None, task_config: str | None) -> str | None:
    """The benchmark's pinned per-episode (unseen) instruction for this cell, or None outside the table."""
    table = _official_episodes()
    if seed is None or task_config != table["task_config"]:
        return None
    entry = table["episodes"].get(f"{task}:s{seed}")
    return str(entry["instruction"]) if entry else None


def instruction_source(task: str, *, seed: int | None = None, episode_names: Mapping[str, Any] | None = None,
                       task_config: str | None = None) -> str:
    """Which of the three sources ``public_instruction`` uses for this cell."""
    key = f"{task}:s{seed}"
    named = ((episode_names or {}).get("episodes") or {}).get(key) if seed is not None else None
    if isinstance(named, Mapping) and str(named.get("instruction") or "").strip():
        return "episode_name"
    if official_episode_instruction(task, seed, task_config) is not None:
        return "official_episode_unseen_v1"
    return "task_level"


def public_instruction(task: str, *, seed: int | None = None,
                       episode_names: Mapping[str, Any] | None = None, task_config: str | None = None) -> str:
    """The instruction delivered for ``task`` and ``seed``.

    In order: an explicit episode name for ``<task>:s<seed>`` (any task); the
    benchmark's pinned per-episode instruction for a standard ``demo_randomized``
    episode; the task-level instruction. A seed-named task has no task-level
    instruction, so outside the table it needs its episode name.
    """
    entry = _entry(task)
    source = instruction_source(task, seed=seed, episode_names=episode_names, task_config=task_config)
    if source == "episode_name":
        return str(episode_names["episodes"][f"{task}:s{seed}"]["instruction"])
    if source == "official_episode_unseen_v1":
        return str(official_episode_instruction(task, seed, task_config))
    if not entry.get("seed_named"):
        return str(entry["instruction"])
    raise RoboTwinInstructionError(
        f"{task} names the objects its seed spawns; the episode names lack {task}:s{seed}")


__all__ = [
    "INSTRUCTIONS_PATH", "INSTRUCTIONS_SCHEMA", "OFFICIAL_EPISODES_PATH", "OFFICIAL_EPISODES_SCHEMA",
    "RoboTwinInstructionError", "instruction_source", "official_episode_instruction",
    "public_instruction", "public_instructions", "seed_named",
]
