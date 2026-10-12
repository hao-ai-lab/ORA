"""Register LIBERO-Pro perturbation suites missing from upstream benchmark maps."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ora.environment.benchmarks.libero_pro import CORE_SUITES



def _language(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"\(:language\s+([^\r\n)]*?)\s*\)", text, re.IGNORECASE)
    if match is None or not match.group(1).strip():
        raise RuntimeError(f"LIBERO-Pro BDDL has no :language field: {path}")
    return match.group(1).strip()


def ensure_perturbation_suites(benchmark: Any) -> None:
    """Install exact task maps/classes into one imported benchmark module.

    The pinned LIBERO-Pro dataset contains the perturbation
    data directories but does not register these 16 suites. This adapter keeps
    the external checkout clean and makes the missing registration explicit.
    """

    existing = benchmark.get_benchmark_dict()
    for suite in CORE_SUITES:
        if suite in existing:
            continue
        family, _separator, _perturbation = suite.rpartition("_")
        base_tasks = benchmark.task_maps.get(family)
        if not base_tasks:
            raise RuntimeError(f"LIBERO base task map is unavailable: {family}")
        tasks = {}
        for name, base_task in base_tasks.items():
            bddl = (
                Path(benchmark.get_libero_path("bddl_files"))
                / suite / f"{name}.bddl"
            )
            init_states = (
                Path(benchmark.get_libero_path("init_states"))
                / suite / f"{name}.pruned_init"
            )
            if not bddl.is_file() or not init_states.is_file() or init_states.stat().st_size == 0:
                raise RuntimeError(
                    f"LIBERO-Pro authoritative data is unavailable for {suite}/{name}"
                )
            tasks[name] = benchmark.Task(
                name=name,
                language=_language(bddl),
                problem=base_task.problem,
                problem_folder=suite,
                bddl_file=f"{name}.bddl",
                init_states_file=f"{name}.pruned_init",
            )
        benchmark.task_maps[suite] = tasks

        def initialize(self: Any, task_order_index: int = 0, *, _suite: str = suite) -> None:
            benchmark.Benchmark.__init__(self, task_order_index=task_order_index)
            self.name = _suite
            self._make_benchmark()

        registered = type(suite.upper(), (benchmark.Benchmark,), {"__init__": initialize})
        benchmark.register_benchmark(registered)
