"""Prepared benchmark tasks for ORA; task language and admission stay source-bound."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .paths import PROJECT_ROOT
from .spec import LAUNCH_ADMISSION_SCHEMA, _launch_admission, load_run_spec

NAMESPACE = "ora-interactive"
BENCHMARKS = frozenset({"robotwin", "libero_pro", "robodojo"})


@dataclass(frozen=True)
class PreparedTask:
    value: str
    path: Path
    digest: str
    spec: Mapping[str, Any]

    def item(self) -> dict[str, str]:
        return {"value": self.value,
                "label": f"{self.spec['runtime']['kind']} / {self.spec['task']}",
                "description": f"{self.path.name} · seed {self.spec.get('seed')} · {self.spec['task_instruction']}"}

    def interactive_spec(self, attempt: str, initial_messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if hashlib.sha256(self.path.read_bytes()).hexdigest() != self.digest:
            raise RuntimeError(f"prepared RunSpec changed; reopen the task picker: {self.path}")
        spec = copy.deepcopy(dict(self.spec))
        admission = spec.get("launch_admission")
        if admission is not None and not isinstance(admission, Mapping):
            raise RuntimeError("prepared launch_admission must be an object")
        if admission is not None and admission.get("mode") == "disabled":
            raise RuntimeError("a context-only RunSpec cannot become an interactive launch")
        if admission is not None and admission.get("gate") is not None:
            raise RuntimeError("a source-bound launch gate cannot be relabelled; compose an ora-interactive RunSpec")
        # Namespace changes must not remove a source experiment's admission rules.
        _launch_admission(admission, experiment=str(spec.get("experiment", "")),
                          config=str(spec.get("config", "")), root=PROJECT_ROOT)
        spec.update(experiment=NAMESPACE, output_workspace=f"outputs/{NAMESPACE}")
        spec.pop("experiment_identity", None)
        spec["operator_control"] = {"mode": "file_inbox_v1"}
        if initial_messages:
            spec["operator_control"]["initial_messages"] = copy.deepcopy(list(initial_messages))
        # This is a new human-interactive treatment, never a claim on an old study's cell.
        spec["launch_admission"] = {
            "schema": LAUNCH_ADMISSION_SCHEMA, "mode": "single_attempt_cell",
            "cohort_id": f"ora-interactive-{attempt}", "cell_id": self.value, "retry_allowed": False,
        }
        return spec


def load_tasks(paths: Sequence[Path]) -> list[PreparedTask]:
    tasks = []
    for index, raw in enumerate(paths):
        path = raw.expanduser().resolve()
        before = path.read_bytes()
        spec = load_run_spec(path)
        if before != path.read_bytes():
            raise RuntimeError(f"RunSpec changed while reading: {path}")
        if not isinstance(spec.get("policy"), Mapping) or spec["policy"].get("kind") != "native_api":
            raise RuntimeError(f"ORA interactive requires a Native RunSpec: {path}")
        runtime = spec.get("runtime")
        if not isinstance(runtime, Mapping) or runtime.get("kind") not in BENCHMARKS:
            raise RuntimeError(f"choose a prepared RoboTwin, LIBERO or RoboDojo benchmark RunSpec: {path}")
        if any(not isinstance(spec.get(key), str) or not spec[key].strip() for key in ("task", "task_instruction")):
            raise RuntimeError(f"RunSpec has no benchmark task/instruction: {path}")
        tasks.append(PreparedTask(f"task-{index + 1}", path, hashlib.sha256(before).hexdigest(), spec))
    if not tasks:
        raise RuntimeError("at least one prepared benchmark RunSpec is required")
    return tasks
