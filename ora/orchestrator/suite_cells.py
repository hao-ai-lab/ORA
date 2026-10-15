"""A cell list for ``suite_queue``: RoboTwin, LIBERO-Pro and LIBERO cells composed by ``benchmark_spec``.

A cell list (schema ``ora.suite-cells.v1``) names cells only: the
benchmark, task or suite, seed or initial state, and for RoboTwin the scene
configuration and optionally its own seed manifest. Everything a RunSpec owns
beyond the cell (namespace, cohort, Provider, rounds, slot, readiness report)
comes from the queue, so one list can run under several treatments::

    {"schema": "ora.suite-cells.v1",
     "cells": [
       {"environment": "robotwin", "task": "adjust_bottle", "seed": 100002, "task_config": "demo_clean"},
       {"environment": "libero_pro", "suite": "libero_10_swap", "task_id": 6, "seed": 0},
       {"environment": "libero", "suite": "libero_object", "task_id": 0, "seed": 0}]}

A RoboTwin cell without ``seed_manifest`` reads ``<seed manifests>/<task>.json``
(the layout ``robotwin_evaluation qualify`` writes). A cell whose seed our
expert never solved carries ``expert_unsolved: {reason, evidence: [scan
manifest paths]}`` instead and runs under the labelled qualification exception
(``qualification_exception``). A standard ``demo_randomized`` episode
receives the benchmark's pinned per-episode instruction by default
(``robotwin_instructions.official_episode_instruction``); any RoboTwin cell may
instead carry an explicit ``episode_instruction``, which overrides it (a
seed-named task outside that table needs one). ``config`` overrides the
builder's cell id; ids must be distinct because admission is per cell. Any
cell may name a registered ``baseline`` (``policy.baseline``); without it the default
baseline composes, so one list can hold both arms of a paired comparison
that differs only in the baseline.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ora.orchestrator import benchmark_spec

CELLS_SCHEMA = "ora.suite-cells.v1"
#: The Runtime kind of each cell environment: a lane must register this kind.
RUNTIME_KINDS = {"robotwin": "robotwin", "libero_pro": "libero_pro", "libero": "libero_pro"}
_FIELDS = {
    "robotwin": ({"environment", "task", "seed"}, {"task_config", "seed_manifest", "episode_instruction",
                                                    "expert_unsolved", "config", "baseline"}),
    "libero_pro": ({"environment", "suite", "task_id", "seed"}, {"config", "baseline"}),
    "libero": ({"environment", "suite", "task_id", "seed"}, {"config", "baseline"}),
}


@dataclass(frozen=True)
class Cell:
    """One listed cell; ``config`` is its admission cell id."""

    environment: str
    config: str
    fields: Mapping[str, Any] = field(default_factory=dict)

    @property
    def runtime_kind(self) -> str:
        return RUNTIME_KINDS[self.environment]

    def summary(self) -> dict[str, Any]:
        return {"environment": self.environment,
                **{key: (True if key == "expert_unsolved" else value) for key, value in self.fields.items()
                   if key != "seed_manifest"}}


def default_config(entry: Mapping[str, Any]) -> str:
    """The builder's default cell id for a list entry."""
    environment = entry["environment"]
    if environment == "robotwin":
        tag = benchmark_spec.TASK_CONFIG_TAGS[entry.get("task_config", "demo_randomized")]
        return f"rt-{tag}-{entry['task']}-s{entry['seed']}"
    prefix = "lp" if environment == "libero_pro" else "lb"
    return f"{prefix}-{entry['suite']}-t{int(entry['task_id']):02d}-s{entry['seed']}"


def parse_cells(value: Mapping[str, Any]) -> list[Cell]:
    """The cells of a decoded cell list; refused on an unknown field, environment or a repeated id."""
    if not isinstance(value, Mapping) or value.get("schema") != CELLS_SCHEMA:
        raise ValueError(f"a cell list must carry schema {CELLS_SCHEMA}")
    entries = value.get("cells")
    if not isinstance(entries, list) or not entries:
        raise ValueError("a cell list needs a non-empty cells array")
    cells = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping) or entry.get("environment") not in _FIELDS:
            raise ValueError(f"cell {index}: environment must be one of {sorted(_FIELDS)}")
        required, optional = _FIELDS[entry["environment"]]
        missing, unknown = required - set(entry), set(entry) - required - optional
        if missing or unknown:
            raise ValueError(f"cell {index}: missing {sorted(missing)}, unknown {sorted(unknown)}")
        if entry["environment"] == "robotwin" and entry.get("task_config", "demo_randomized") not in (
                benchmark_spec.TASK_CONFIG_TAGS):
            raise ValueError(f"cell {index}: task_config must be one of {sorted(benchmark_spec.TASK_CONFIG_TAGS)}")
        if "baseline" in entry and entry["baseline"] not in benchmark_spec.BASELINE_CATALOGS:
            raise ValueError(f"cell {index}: baseline must be one of {sorted(benchmark_spec.BASELINE_CATALOGS)}")
        if "episode_instruction" in entry and not (
                entry["environment"] == "robotwin" and isinstance(entry["episode_instruction"], str)
                and entry["episode_instruction"].strip()):
            raise ValueError(f"cell {index}: episode_instruction is the non-empty per-episode instruction "
                             "of a RoboTwin cell")
        unsolved = entry.get("expert_unsolved")
        if unsolved is not None and not (
                isinstance(unsolved, Mapping) and set(unsolved) == {"reason", "evidence"}
                and isinstance(unsolved["reason"], str) and unsolved["reason"].strip()
                and isinstance(unsolved["evidence"], list) and unsolved["evidence"]
                and all(isinstance(item, str) and item for item in unsolved["evidence"])
                and "seed_manifest" not in entry):
            raise ValueError(f"cell {index}: expert_unsolved is {{reason, evidence: [manifest paths]}} "
                             "and replaces seed_manifest")
        config = entry.get("config") or default_config(entry)
        cells.append(Cell(entry["environment"], config,
                          {key: value for key, value in entry.items() if key not in {"environment", "config"}}))
    configs = [cell.config for cell in cells]
    if len(set(configs)) != len(configs):
        raise ValueError("cell ids must be distinct: admission is one attempt per cell")
    return cells


def load_cells(path: Path) -> tuple[list[Cell], str]:
    """The cells of the list at ``path`` and the list's sha256."""
    data = Path(path).read_bytes()
    return parse_cells(json.loads(data)), hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class Treatment:
    """What the queue, not the list, decides for every cell."""

    experiment: str
    cohort_id: str
    service_report: Path
    host: str | None = None
    provider_profile: str | None = None
    provider_catalog: Path | None = None
    rounds: int | None = None
    seed_manifests: Path | None = None


def seed_manifest(cell: Cell, treatment: Treatment) -> Path | None:
    """The manifest admitting a RoboTwin cell's seed: its own, else ``<seed manifests>/<task>.json``."""
    if cell.environment != "robotwin" or cell.fields.get("expert_unsolved"):
        return None
    own = cell.fields.get("seed_manifest")
    if own:
        return Path(own)
    if treatment.seed_manifests is None:
        raise ValueError(f"{cell.config}: a RoboTwin cell needs seed_manifest or the queue's seed manifests")
    return treatment.seed_manifests / f"{cell.fields['task']}.json"


def build_spec(cell: Cell, lane: str, treatment: Treatment) -> dict[str, Any]:
    """The RunSpec of ``cell`` on ``lane`` under ``treatment`` (``benchmark_spec`` validates everything)."""
    common = dict(seed=cell.fields["seed"], experiment=treatment.experiment, resource_profile=lane,
                  host=treatment.host, config=cell.config, provider_profile=treatment.provider_profile,
                  provider_catalog=treatment.provider_catalog, service_report=treatment.service_report,
                  cohort_id=treatment.cohort_id, rounds=treatment.rounds, baseline=cell.fields.get("baseline"))
    if cell.environment == "robotwin":
        task, named = cell.fields["task"], cell.fields.get("episode_instruction")
        episode_names = {"episodes": {f"{task}:s{cell.fields['seed']}": {"instruction": named}}} if named else None
        unsolved = cell.fields.get("expert_unsolved") or {}
        return benchmark_spec.build_robotwin_spec(
            task, task_config=cell.fields.get("task_config", "demo_randomized"),
            seed_manifest=seed_manifest(cell, treatment), episode_names=episode_names,
            expert_unsolved=[Path(item) for item in unsolved.get("evidence", [])] or None,
            expert_unsolved_reason=unsolved.get("reason"), **common)
    builder = benchmark_spec.build_libero_pro_spec if cell.environment == "libero_pro" else (
        benchmark_spec.build_libero_spec)
    return builder(cell.fields["suite"], int(cell.fields["task_id"]), **common)


def compose(cell: Cell, spec: Mapping[str, Any], treatment: Treatment) -> Path:
    """Write the spec under ``outputs/<experiment>/specs`` and import its seed manifest."""
    unsolved = cell.fields.get("expert_unsolved") or {}
    return benchmark_spec.compose(spec, seed_manifest=seed_manifest(cell, treatment),
                                  expert_unsolved=[Path(item) for item in unsolved.get("evidence", [])] or None)


__all__ = ["CELLS_SCHEMA", "Cell", "RUNTIME_KINDS", "Treatment", "build_spec", "compose", "default_config",
           "load_cells", "parse_cells", "seed_manifest"]
