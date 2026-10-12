"""Host-safe LIBERO-Pro suite and robustness identity contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .libero import (
    LIBERO_BASE_SUITES as ORIGINAL_BASE_SUITES,
    LiberoBenchmarkError,
    validate_base_suite_task,
)


LIBERO_PRO_CODE_REVISION = "2b910b5b5f53016bef9907632f6f840f1ce2229c"
LIBERO_PRO_DATA_REVISION = "c86fc3b8293185a6f373677018ff3e37f8391602"
ROBUSTNESS_SCHEMA = "ora.libero-pro-robustness.v3"

FAMILY_ORDER = ("object", "goal", "spatial", "libero_10")
BASE_SUITES = tuple(
    family if family.startswith("libero_") else f"libero_{family}"
    for family in FAMILY_ORDER
)
CORE_VARIANT_ORDER = ("position", "language", "object", "task")
CORE_VARIANT_SUFFIXES = {
    "position": "swap",
    "language": "lan",
    "object": "object",
    "task": "task",
}
CORE_SUITES = tuple(
    f"{base_suite}_{suffix}"
    for base_suite in BASE_SUITES
    for suffix in ("task", "swap", "lan", "object")
)
RUNTIME_SUITES = (*ORIGINAL_BASE_SUITES, *CORE_SUITES)
ROBUSTNESS_CASE_FOLDERS = {
    "visual_noise_glare": "01_visual_noise_glare",
    "camera_view_angle": "02_camera_view_angle",
    "runtime_object_move": "03_runtime_object_move",
    "object_texture": "04_object_texture",
    "view_occlusion": "05_view_occlusion",
    "object_shape": "06_object_shape",
    "initial_pose_position_angle": "07_initial_pose_position_angle",
}
ROBUSTNESS_CASES = tuple(ROBUSTNESS_CASE_FOLDERS)
#: LIBERO evaluates every task on its fixed initial states; the pinned files
#: carry 50 per task, and the Runtime selects state ``seed % 50``.
OFFICIAL_CORE_INIT_STATES = 50
#: The LIBERO-Pro leaderboard's column for each core perturbation.
OFFICIAL_PERTURBATION_LABELS = {
    "object": "Obj", "position": "Pos", "language": "Sem", "task": "Task",
}
OFFICIAL_EVALUATION_SCHEMA = "ora.libero-pro-official-evaluation.v1"


class LiberoProBenchmarkError(ValueError):
    """A host-side LIBERO-Pro benchmark identity is invalid."""


def base_suite(family: str) -> str:
    """Return the registered base suite for one public family identifier."""

    if family not in FAMILY_ORDER:
        raise LiberoProBenchmarkError(f"unsupported LIBERO-Pro family: {family!r}")
    return family if family.startswith("libero_") else f"libero_{family}"


def core_suite(family: str, perturbation: str) -> str:
    try:
        suffix = CORE_VARIANT_SUFFIXES[perturbation]
    except KeyError as exc:
        raise LiberoProBenchmarkError(
            f"unsupported LIBERO-Pro core perturbation: {perturbation!r}"
        ) from exc
    return f"{base_suite(family)}_{suffix}"


def robustness_suite(family: str, case: str) -> str:
    if case not in ROBUSTNESS_CASE_FOLDERS:
        raise LiberoProBenchmarkError(
            f"unsupported LIBERO-Pro robustness case: {case!r}"
        )
    return f"{base_suite(family)}_robust_{case}"


def expand_robustness_suites(
    suites: Mapping[str, Any], selection: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Expand one explicit family × case selection into Runtime suite handles.

    The experiment file owns which cells are selected.  This function owns the
    deterministic naming and rejects aliases, duplicates, and partial drift.
    """

    expanded = {str(key): dict(value) for key, value in suites.items()}
    if selection is None:
        return expanded
    families = selection.get("families")
    cases = selection.get("cases")
    if (
        not isinstance(families, Sequence)
        or isinstance(families, (str, bytes))
        or not isinstance(cases, Sequence)
        or isinstance(cases, (str, bytes))
    ):
        raise LiberoProBenchmarkError(
            "robustness selection requires family and case lists"
        )
    selected_families = tuple(str(item) for item in families)
    selected_cases = tuple(str(item) for item in cases)
    if len(set(selected_families)) != len(selected_families):
        raise LiberoProBenchmarkError("robustness families contain duplicates")
    if len(set(selected_cases)) != len(selected_cases):
        raise LiberoProBenchmarkError("robustness cases contain duplicates")
    for family in selected_families:
        runtime_suite = base_suite(family)
        for case in selected_cases:
            identifier = robustness_suite(family, case)
            if identifier in expanded:
                raise LiberoProBenchmarkError(
                    f"expanded robustness suite collides: {identifier}"
                )
            expanded[identifier] = {
                "family": family,
                "perturbation": "robustness",
                "runtime_suite": runtime_suite,
                "robustness_case": case,
            }
    return expanded


def official_core_evaluation(
    *,
    families: Sequence[str] = FAMILY_ORDER,
    perturbations: Sequence[str] = CORE_VARIANT_ORDER,
    init_states: int = OFFICIAL_CORE_INIT_STATES,
) -> dict[str, Any]:
    """Describe the official LIBERO-Pro core evaluation as Runtime cells.

    One cell per (family, perturbation, task, initial state): the published
    ``_task``/``_swap``/``_lan``/``_object`` BDDL and initial-state files at
    the pinned revisions, 50 initial states per task as the LIBERO protocol
    evaluates. Env perturbation is not included: its initial states are
    generated at run time and are not reproducible from the published data.
    """

    if (
        not isinstance(families, Sequence) or isinstance(families, (str, bytes))
        or not families or len(set(families)) != len(families)
        or any(family not in FAMILY_ORDER for family in families)
    ):
        raise LiberoProBenchmarkError("families must be distinct members of FAMILY_ORDER")
    if (
        not isinstance(perturbations, Sequence) or isinstance(perturbations, (str, bytes))
        or not perturbations or len(set(perturbations)) != len(perturbations)
        or any(item not in CORE_VARIANT_ORDER for item in perturbations)
    ):
        raise LiberoProBenchmarkError("perturbations must be distinct core perturbations")
    if (
        isinstance(init_states, bool) or not isinstance(init_states, int)
        or not 1 <= init_states <= OFFICIAL_CORE_INIT_STATES
    ):
        raise LiberoProBenchmarkError(
            f"init_states must be in [1, {OFFICIAL_CORE_INIT_STATES}]"
        )
    cells = [
        {"suite": core_suite(family, perturbation), "family": family,
         "perturbation": perturbation,
         "label": OFFICIAL_PERTURBATION_LABELS[perturbation],
         "task_id": task_id, "seed": seed}
        for family in families
        for perturbation in perturbations
        for task_id in range(10)
        for seed in range(init_states)
    ]
    return {
        "schema": OFFICIAL_EVALUATION_SCHEMA,
        "libero_pro_code_revision": LIBERO_PRO_CODE_REVISION,
        "libero_pro_data_revision": LIBERO_PRO_DATA_REVISION,
        "families": list(families),
        "perturbations": list(perturbations),
        "tasks_per_suite": 10,
        "init_states_per_task": init_states,
        "init_state_selection": "seed selects pinned initial state seed % 50; seeds 0..49 cover each state once",
        "episodes_per_seed": 1,
        "episode_count": len(cells),
        "excluded": {
            "env": "initial states generated at run time; not reproducible from the published data",
        },
        "step_accounting": (
            "Harness budgets count native LIBERO environment steps (runtime.max_sim_steps) "
            "and Agent decisions per episode; the per-suite maximum-step caps of learned-policy "
            "evaluators bound a policy acting every control step and are not adopted. A "
            "comparison row states both."
        ),
        "cells": cells,
    }


def validate_runtime_extension(runtime: Mapping[str, Any]) -> None:
    """Validate the optional robustness fields without importing a backend."""

    suite = runtime.get("suite")
    task_id = runtime.get("task_id")
    if not isinstance(suite, str):
        raise LiberoProBenchmarkError("runtime.suite must identify a LIBERO suite")
    if suite in ORIGINAL_BASE_SUITES:
        try:
            validate_base_suite_task(suite, task_id)
        except LiberoBenchmarkError as exc:
            raise LiberoProBenchmarkError(str(exc)) from exc
    elif suite in CORE_SUITES:
        if (
            isinstance(task_id, bool)
            or not isinstance(task_id, int)
            or not 0 <= task_id < 10
        ):
            raise LiberoProBenchmarkError(
                f"{suite} task_id must be in [0,9]"
            )
    else:
        raise LiberoProBenchmarkError(
            f"unsupported LIBERO Runtime suite: {suite!r}"
        )
    case = runtime.get("robustness_case")
    if case is None:
        return
    if case not in ROBUSTNESS_CASES:
        raise LiberoProBenchmarkError("runtime.robustness_case is not supported")
    if runtime.get("suite") not in BASE_SUITES:
        raise LiberoProBenchmarkError(
            "runtime.robustness_case requires a base LIBERO suite"
        )
    if runtime.get("required_libero_pro_revision") != LIBERO_PRO_CODE_REVISION:
        raise LiberoProBenchmarkError(
            "runtime.robustness_case requires its paired LIBERO-Pro revision"
        )


__all__ = [
    "BASE_SUITES",
    "CORE_SUITES",
    "CORE_VARIANT_ORDER",
    "CORE_VARIANT_SUFFIXES",
    "FAMILY_ORDER",
    "LIBERO_PRO_CODE_REVISION",
    "LIBERO_PRO_DATA_REVISION",
    "LiberoProBenchmarkError",
    "OFFICIAL_CORE_INIT_STATES",
    "OFFICIAL_EVALUATION_SCHEMA",
    "OFFICIAL_PERTURBATION_LABELS",
    "ROBUSTNESS_CASE_FOLDERS",
    "ROBUSTNESS_CASES",
    "ROBUSTNESS_SCHEMA",
    "RUNTIME_SUITES",
    "base_suite",
    "core_suite",
    "expand_robustness_suites",
    "official_core_evaluation",
    "robustness_suite",
    "validate_runtime_extension",
]
