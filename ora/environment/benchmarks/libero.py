"""Host-safe identities for the five original LIBERO base suites."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


LIBERO_BASE_SUITE_TASK_COUNTS = {
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_90": 90,
    "libero_10": 10,
}
LIBERO_BASE_SUITES = tuple(LIBERO_BASE_SUITE_TASK_COUNTS)


class LiberoBenchmarkError(ValueError):
    """An original LIBERO suite or family-local task identity is invalid."""


def validate_base_suite_task(suite: str, task_id: int) -> None:
    """Validate one benchmark-ordered task slot in an original base suite."""

    try:
        task_count = LIBERO_BASE_SUITE_TASK_COUNTS[suite]
    except KeyError as exc:
        raise LiberoBenchmarkError(
            f"unsupported original LIBERO base suite: {suite!r}"
        ) from exc
    if isinstance(task_id, bool) or not isinstance(task_id, int):
        raise LiberoBenchmarkError("LIBERO task_id must be an integer")
    if not 0 <= task_id < task_count:
        raise LiberoBenchmarkError(
            f"{suite} task_id must be in [0,{task_count - 1}]"
        )


def validate_base_runtime(runtime: Mapping[str, Any]) -> None:
    """Validate a Runtime mapping that selects an original LIBERO suite."""

    suite = runtime.get("suite")
    if not isinstance(suite, str):
        raise LiberoBenchmarkError("runtime.suite must identify a LIBERO suite")
    validate_base_suite_task(suite, runtime.get("task_id"))


__all__ = [
    "LIBERO_BASE_SUITES",
    "LIBERO_BASE_SUITE_TASK_COUNTS",
    "LiberoBenchmarkError",
    "validate_base_runtime",
    "validate_base_suite_task",
]
