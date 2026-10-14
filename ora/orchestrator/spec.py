"""RunSpec validation, resource binding and deterministic identity.

``resolve_run_spec`` turns one RunSpec into a content-addressed RunPlan as a
fixed sequence of section resolvers:

- ``spec_policy.py``: the fields every policy kind shares, the ToolSurface and
  its action codec; ``POLICY_KIND_RESOLVERS`` then selects the scripted or
  native_api (``spec_native_policy.py``, ``spec_native_provider.py``) resolver;
- ``spec_runtime.py``: the runtime by kind, budgets and session;
- ``spec_composition.py``: prompt layers and the hashed policy composition.

This module keeps the episode fields, qualification, launch admission and the
RunPlan identity, and verifies persisted plans. Several names imported below
are used only by the section modules; they remain importable from here.
"""

from __future__ import annotations

from ora.orchestrator.experiment_identity import bind_identity, validate_identity
from ora.orchestrator.qualification_exception import QualificationExceptionError, resolve_exception
from ora.orchestrator.spec_values import (
    RunSpecError, canonical_bytes, json_sha256, file_sha256, _mapping, _positive_int, _stable_id,
    _resource, _resource_list,
)

import copy
import hashlib
import importlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, NamedTuple
from urllib.parse import urlsplit

import yaml

from .paths import PROJECT_ROOT, resolve_workspace_root
from .environment_spec import resolve_typed_environment_prompt_profile
from ora.agent.tools import tool_names
from ora.agent.memory.artifacts import (
    LOSSLESS_CONTEXT_RETENTION,
    recoverable_artifact_definitions,
    recoverable_artifact_result_schemas,
)
from ora.agent.composition import (
    POLICY_CONTEXT_V3_SCHEMA, V2_CHANNELS,
    ToolSurfaceError,
    resolve_tool_surface,
)
from ora.agent.composition.environment_context import (
    EnvironmentContextError,
    resolve_environment_prompt_profile,
)
from ora.agent.context.management import (
    ContextManagementConfigError,
    validate_context_management,
)
from ora.agent.tools.robot import manual_sections
from ora.contracts.workspace import WorkspaceError, normalize_workspace_bounds
from ora.environment.definitions import (
    SPARSE_MOTION_QUALIFIED_ENVIRONMENTS,
)
from ora.environment.workspace_profiles import resolve_workspace_profile
from .resources import ExecutionResourceError, resolve_execution_resource
from .spec_native_provider import (
    resolve_native_provider_endpoint, resolve_native_provider_protocol,
)
from .spec_composition import bind_policy_context, compose_policy
from .spec_native_policy import resolve_native_policy
from .spec_operator_control import resolve_operator_control, validate_persisted_operator_control
from .spec_policy import (
    POLICY_KINDS, resolve_policy_surface, resolve_scripted_policy,
)
from .spec_runtime import (
    ENVIRONMENT_SIGNAL_POLICIES, ENVIRONMENT_TRANSPORTS, RUNTIME_KINDS,
    SESSION_CADENCES, resolve_budgets, resolve_runtime, resolve_session,
)
from .spec_source import bind_policy_identity, valid_plan_identity


SCHEMA = "ora.run-spec.v1"
PLAN_SCHEMA = "ora.run-plan.v1"
EVIDENCE_TIERS = {"diagnostic", "infrastructure", "behavioral"}
LAUNCH_ADMISSION_SCHEMA = "ora.run-launch-admission.v1"
EXPERIMENT_LAUNCH_POLICY_SCHEMA = (
    "ora.experiment-launch-admission.v1"
)

# One resolver per policy.kind for the fields only that kind has. Each runs
# after ``resolve_policy_surface`` and normalizes the policy copy in place.
POLICY_KIND_RESOLVERS = {
    "scripted": resolve_scripted_policy,
    "native_api": resolve_native_policy,
}


def validate_run_plan(
    plan: Mapping[str, Any], *, run_root: Path | None = None,
    admission: str = "live",
) -> dict[str, Any]:
    """Verify one persisted RunPlan's complete content-addressed identity."""

    if not isinstance(plan, Mapping) or plan.get("schema") != PLAN_SCHEMA:
        raise RunSpecError("persisted RunPlan schema is invalid")
    validate_persisted_operator_control(plan)
    if "experiment_identity" in plan:
        try:
            validate_identity(plan["experiment_identity"])
        except ValueError as exc:
            raise RunSpecError(str(exc)) from exc
    source = plan.get("source")
    if not valid_plan_identity(plan):
        raise RunSpecError("persisted RunPlan source identity is invalid")
    if source["dirty"]:
        raise RunSpecError("persisted RunPlan was resolved from dirty source")
    if "launch_admission" in plan:
        normalized_admission = _launch_admission(
            plan.get("launch_admission"),
            experiment=str(plan.get("experiment", "")),
            config=str(plan.get("config", "")),
            root=PROJECT_ROOT,
        )
        if normalized_admission != plan.get("launch_admission"):
            raise RunSpecError("persisted RunPlan launch admission is invalid")
    else:
        experiment = str(plan.get("experiment", ""))
        config = str(plan.get("config", ""))
        source_policy = _source_launch_policy(experiment, root=PROJECT_ROOT)
        if source_policy is not None and (
            _launch_requirement(
                experiment, config, root=PROJECT_ROOT,
            ) is not None
            or config not in source_policy["declared_configs"]
        ):
            raise RunSpecError("persisted RunPlan lost required launch admission")
    portable = {
        str(key): copy.deepcopy(value)
        for key, value in plan.items()
        if key not in {"plan_sha256", "plan_id", "confirmation_code"}
    }
    portable["source"] = {"commit": source["commit"]}
    expected_hash = json_sha256(portable)
    expected_id = (
        f"{plan.get('experiment')}-{str(plan.get('milestone', '')).lower()}-"
        f"{expected_hash[:12]}"
    )
    if plan.get("plan_sha256") != expected_hash:
        raise RunSpecError("persisted RunPlan hash does not match its content")
    if plan.get("plan_id") != expected_id:
        raise RunSpecError("persisted RunPlan id does not match its content")
    if plan.get("confirmation_code") != expected_hash[-6:]:
        raise RunSpecError("persisted RunPlan confirmation code is invalid")
    verify_launch_admission(plan, run_root=run_root, admission=admission)
    return copy.deepcopy(dict(plan))


def _git(*args: str, root: Path = PROJECT_ROOT) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True,
        timeout=60, check=False,
    )
    if result.returncode:
        raise RunSpecError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def _source_launch_policy(
    experiment: str, *, root: Path,
) -> dict[str, Any] | None:
    package = root / "experiments" / experiment.replace("-", "_")
    definition_path = package / "experiment.yaml"
    if not definition_path.is_file():
        return None
    try:
        definition = yaml.safe_load(definition_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RunSpecError("experiment launch policy is unreadable") from exc
    if not isinstance(definition, Mapping) or definition.get("experiment") != experiment:
        return None
    policy = definition.get("launch_admission_policy")
    if policy is None:
        return None
    if (
        not isinstance(policy, Mapping)
        or set(policy) != {
            "schema", "default_mode", "validator", "cells",
        }
        or policy.get("schema") != EXPERIMENT_LAUNCH_POLICY_SCHEMA
        or policy.get("default_mode") != "inspection_only"
        or policy.get("validator") != "experiment_package_v1"
        or not isinstance(policy.get("cells"), Mapping)
        or not policy["cells"]
    ):
        raise RunSpecError("experiment launch policy is malformed")
    cells: dict[str, dict[str, str]] = {}
    for raw_config, raw_cell in policy["cells"].items():
        cell_config = _stable_id(raw_config, "launch policy config")
        if (
            not isinstance(raw_cell, Mapping)
            or set(raw_cell) != {
                "mode", "cohort_id", "cell_id", "resource_profile", "slot_id",
                "gate",
            }
            or raw_cell.get("mode") != "single_attempt_cell"
        ):
            raise RunSpecError("experiment launch policy cell is malformed")
        gate = raw_cell.get("gate")
        gate_path = Path(str(gate))
        if (
            not isinstance(gate, str)
            or gate_path.is_absolute()
            or ".." in gate_path.parts
            or gate_path.parts[:2] != ("outputs", experiment)
        ):
            raise RunSpecError("experiment launch policy gate is invalid")
        cells[cell_config] = {
            "cohort_id": _stable_id(raw_cell.get("cohort_id"), "launch cohort"),
            "cell_id": _stable_id(raw_cell.get("cell_id"), "launch cell"),
            "resource_profile": _stable_id(
                raw_cell.get("resource_profile"), "launch resource profile",
            ),
            "slot_id": _stable_id(raw_cell.get("slot_id"), "launch slot"),
            "gate": gate_path.as_posix(),
            "validator": str(policy["validator"]),
        }
    declared_configs = definition.get("configs")
    if not isinstance(declared_configs, Mapping):
        raise RunSpecError("experiment launch policy requires declared configs")
    return {
        "schema": str(policy["schema"]),
        "default_mode": "inspection_only",
        "validator": str(policy["validator"]),
        "cells": cells,
        "declared_configs": tuple(str(item) for item in declared_configs),
        "package": experiment.replace("-", "_"),
    }


def _launch_requirement(
    experiment: str, config: str, *, root: Path,
) -> dict[str, str] | None:
    policy = _source_launch_policy(experiment, root=root)
    if policy is None:
        return None
    value = policy["cells"].get(config)
    return copy.deepcopy(value) if value is not None else None


def _launch_admission(
    value: Any, *, experiment: str, config: str, root: Path,
) -> dict[str, Any] | None:
    """Normalize launch/retry admission and enforce source-required cells."""

    policy = _source_launch_policy(experiment, root=root)
    requirement = _launch_requirement(experiment, config, root=root)
    if value is None:
        if policy is not None:
            raise RunSpecError(
                "this experiment cell requires source-bound launch admission"
            )
        return None
    if not isinstance(value, Mapping):
        raise RunSpecError("launch_admission must be an object")
    mode = value.get("mode")
    if mode == "disabled":
        if set(value) != {"schema", "mode"}:
            raise RunSpecError("disabled launch_admission fields are not closed")
        if value.get("schema") != LAUNCH_ADMISSION_SCHEMA:
            raise RunSpecError("launch_admission schema is invalid")
        return {"schema": LAUNCH_ADMISSION_SCHEMA, "mode": "disabled"}
    if mode == "not_required":
        # A policy-covered experiment's gate-free config declares itself
        # explicitly and stays launchable; a source-required cell may not
        # opt out, and without a source policy the declaration is
        # meaningless.
        if set(value) != {"schema", "mode"}:
            raise RunSpecError(
                "not_required launch_admission fields are not closed"
            )
        if value.get("schema") != LAUNCH_ADMISSION_SCHEMA:
            raise RunSpecError("launch_admission schema is invalid")
        if requirement is not None:
            raise RunSpecError(
                "a source-required launch cell cannot declare not_required"
            )
        if policy is None:
            raise RunSpecError(
                "not_required launch_admission needs a source launch policy"
            )
        return {"schema": LAUNCH_ADMISSION_SCHEMA, "mode": "not_required"}
    allowed_fields = {
        "schema", "mode", "cohort_id", "cell_id", "retry_allowed",
    }
    if "gate" in value:
        allowed_fields.add("gate")
    if mode != "single_attempt_cell" or set(value) != allowed_fields:
        raise RunSpecError("launch_admission fields are invalid")
    if value.get("schema") != LAUNCH_ADMISSION_SCHEMA:
        raise RunSpecError("launch_admission schema is invalid")
    if value.get("retry_allowed") is not False:
        raise RunSpecError(
            "single_attempt_cell launch_admission must disable retry"
        )
    normalized = {
        "schema": LAUNCH_ADMISSION_SCHEMA,
        "mode": "single_attempt_cell",
        "cohort_id": _stable_id(value.get("cohort_id"), "launch cohort"),
        "cell_id": _stable_id(value.get("cell_id"), "launch cell"),
        "retry_allowed": False,
    }
    if "gate" in value:
        normalized["gate"] = copy.deepcopy(value["gate"])
    if requirement is not None and (
        normalized["cohort_id"] != requirement["cohort_id"]
        or normalized["cell_id"] != requirement["cell_id"]
        or "gate" not in normalized
    ):
        raise RunSpecError("source-bound launch admission identity is invalid")
    if requirement is None and policy is not None:
        raise RunSpecError("this experiment config is inspection-only")
    return normalized


def _generated_resource(
    root: Path, experiment: str, value: Any, label: str,
) -> tuple[dict[str, Any], Path]:
    """Bind an ignored resource in the Git-common experiment workspace."""

    if not isinstance(value, str) or not value:
        raise RunSpecError(f"{label} must be a project-relative path")
    relative = Path(value)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.parts[:2] != ("outputs", experiment)
    ):
        raise RunSpecError(f"{label} must remain under outputs/{experiment}")
    workspace_root = resolve_workspace_root(root)
    path = (workspace_root / relative).resolve()
    experiment_root = (workspace_root / "outputs" / experiment).resolve()
    if (
        not path.is_file()
        or path.is_symlink()
        or not path.is_relative_to(experiment_root)
    ):
        raise RunSpecError(f"{label} is unavailable: {value}")
    return ({
        "path": relative.as_posix(),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }, path)


def _validate_launch_gate_record(
    *, experiment: str, config: str, gate_path: Path,
    requirement: Mapping[str, str], source_commit: str,
    runtime: Mapping[str, Any], root: Path,
) -> None:
    binding = runtime.get("resource_binding")
    if not isinstance(binding, Mapping):
        raise RunSpecError("launch qualification gate lacks runtime bindings")
    if (
        binding.get("profile_id") != requirement["resource_profile"]
        or binding.get("slot_id") != requirement["slot_id"]
    ):
        raise RunSpecError("launch admission resource slot differs from source policy")
    module_name = f"experiments.{experiment.replace('-', '_')}.run"
    try:
        module = importlib.import_module(module_name)
        validator = getattr(module, "validate_launch_admission_gate")
    except (ImportError, AttributeError) as exc:
        raise RunSpecError("experiment launch validator is unavailable") from exc
    try:
        validator(
            config=config, gate_path=gate_path,
            source_commit=source_commit, runtime=runtime,
        )
    except (RuntimeError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise RunSpecError(f"launch qualification gate is invalid: {exc}") from exc


def _validate_launch_admission_plan(plan: Mapping[str, Any]) -> None:
    """Run an opt-in experiment check against the complete, resolved plan.

    Runtime-only gate validation cannot bind top-level seed, session, budgets
    or the resolved policy/context composition. Experiments that own a fixed
    full composition may add this read-only hook; existing packages without
    it retain their gate contract. Disabled inspection plans deliberately skip
    the hook, allowing an implementation to resolve its expected preview.
    """
    admission = plan.get("launch_admission")
    if not isinstance(admission, Mapping) or (
        admission.get("mode") != "single_attempt_cell" or "gate" not in admission
    ):
        return
    module_name = f"experiments.{str(plan['experiment']).replace('-', '_')}.run"
    try:
        module = importlib.import_module(module_name)
        validator = getattr(module, "validate_launch_admission_plan", None)
        if validator is None:
            return
        if not callable(validator):
            raise TypeError("experiment full-plan validator is not callable")
        result = validator(
            config=str(plan["config"]),
            source_commit=str(plan["source"]["commit"]),
            plan=copy.deepcopy(dict(plan)),
        )
        if result is not None:
            raise TypeError("experiment full-plan validator must return None on success")
    except Exception as exc:
        raise RunSpecError(f"launch qualification full-plan binding is invalid: {exc}") from exc


def verify_launch_admission(
    plan: Mapping[str, Any], *, root: Path = PROJECT_ROOT,
    run_root: Path | None = None, admission: str = "live",
) -> str:
    """Recheck a required ignored admission artifact immediately before spawn.

    Returns how the admission was verified: ``live`` (the workspace gate file
    matches the plan-recorded hash and passes the experiment validator),
    ``snapshot`` (the live file is gone or changed, but the launch-time gate
    snapshot inside the run directory matches the plan-recorded hash),
    ``recorded_only`` (backfill mode only: neither byte source survives and
    the plan/launch records alone attest the launch-time verification), or
    ``absent`` (no admission requirement applies).
    """

    if admission not in {"live", "backfill"}:
        raise RunSpecError("unknown launch admission verification mode")
    experiment = str(plan.get("experiment", ""))
    config = str(plan.get("config", ""))
    requirement = _launch_requirement(experiment, config, root=root)
    plan_admission = plan.get("launch_admission")
    if requirement is None or (
        isinstance(plan_admission, Mapping)
        and plan_admission.get("mode") == "disabled"
    ):
        return "absent"
    if not isinstance(plan_admission, Mapping):
        raise RunSpecError("required launch admission is unavailable")
    gate_resource = plan_admission.get("gate")
    if not isinstance(gate_resource, Mapping) or set(gate_resource) != {
        "path", "sha256", "bytes",
    }:
        raise RunSpecError("launch qualification gate resource is invalid")
    expected_path = requirement["gate"]
    if gate_resource.get("path") != expected_path:
        raise RunSpecError("launch qualification gate path is invalid")
    if admission == "live":
        _validate_launch_admission_plan(plan)
    workspace = resolve_workspace_root(root)
    experiment_root = (workspace / "outputs" / experiment).resolve()
    gate_path = (workspace / expected_path).resolve()
    if (
        gate_path.is_file()
        and not gate_path.is_symlink()
        and gate_path.is_relative_to(experiment_root)
        and gate_path.stat().st_size == gate_resource.get("bytes")
        and file_sha256(gate_path) == gate_resource.get("sha256")
    ):
        if admission == "live":
            _validate_launch_gate_record(
                experiment=experiment, config=config, gate_path=gate_path,
                requirement=requirement,
                source_commit=str(plan["source"]["commit"]),
                runtime=plan["runtime"], root=root,
            )
        # In backfill mode byte equality with the plan-recorded hash is the
        # attestation: the experiment validator binds a gate to the current
        # checkout's HEAD and cannot re-validate a historical admission.
        return "live"
    if run_root is not None:
        snapshot = (run_root / "qualification" / "launch-gate.json").resolve()
        if (
            snapshot.is_file()
            and not snapshot.is_symlink()
            and snapshot.stat().st_size == gate_resource.get("bytes")
            and file_sha256(snapshot) == gate_resource.get("sha256")
        ):
            # These exact bytes passed the experiment validator at launch;
            # re-running that validator would wrongly bind the historical
            # gate to the current checkout's HEAD.
            return "snapshot"
    if admission == "backfill":
        return "recorded_only"
    raise RunSpecError("launch qualification gate is unavailable or changed")


def require_launch_admission_for_spawn(
    plan: Mapping[str, Any], *, root: Path = PROJECT_ROOT,
) -> None:
    """Reject source-declared inspection-only plans before preflight or spawn."""

    experiment = str(plan.get("experiment", ""))
    if (
        _source_launch_policy(experiment, root=root) is not None
        and not isinstance(plan.get("launch_admission"), Mapping)
    ):
        raise RunSpecError(
            "this experiment config is inspection-only and cannot be launched"
        )


def load_run_spec(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise RunSpecError(f"RunSpec is unavailable: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RunSpecError(f"cannot load RunSpec: {exc}") from exc
    if not isinstance(value, dict):
        raise RunSpecError("RunSpec root must be an object")
    return value


class _Episode(NamedTuple):
    """The validated episode fields of a RunSpec."""

    experiment: str
    config: str
    milestone: str
    task: str
    task_instruction: str
    seed: int
    evidence_tier: str
    output_workspace: str


def resolve_run_spec(spec: Mapping[str, Any], *, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Validate one RunSpec and bind it into a content-addressed RunPlan.

    Sections resolve in this fixed order, each raising the first error it
    finds, so a RunSpec with several invalid fields always reports the same
    one. The policy and runtime sections default and normalize shallow copies
    of ``spec["policy"]`` and ``spec["runtime"]``.
    """

    root = root.resolve()
    if spec.get("schema") != SCHEMA:
        raise RunSpecError(f"RunSpec schema must be {SCHEMA}")
    spec = _expand_defaults(spec, root=root)
    episode = _resolve_episode(spec)

    policy, surface, profile_id = resolve_policy_surface(spec)
    POLICY_KIND_RESOLVERS[policy["kind"]](policy, surface, spec.get("runtime"))
    _refuse_memory(policy)
    operator_control = resolve_operator_control({**spec, "policy": policy}) if "operator_control" in spec else None

    runtime, surface = resolve_runtime(
        spec, policy, surface, task=episode.task, seed=episode.seed,
        task_instruction=episode.task_instruction, root=root,
    )
    budgets = resolve_budgets(spec, policy, runtime)
    session = resolve_session(spec, policy, runtime, surface, seed=episode.seed)

    policy_context = bind_policy_context(
        spec, policy, runtime, surface, profile_id=profile_id, root=root,
    )
    composition = compose_policy(policy, surface, policy_context, root=root)
    qualification = _resolve_qualification(spec, episode.experiment, root=root)

    source_commit = _git("rev-parse", "HEAD", root=root)
    launch_admission = _bind_launch_admission(
        spec, episode, source_commit=source_commit, runtime=runtime, root=root,
    )
    source_dirty = bool(_git("status", "--porcelain", "--untracked-files=normal", root=root))
    portable = {
        "schema": PLAN_SCHEMA,
        "experiment": episode.experiment,
        "config": episode.config,
        "milestone": episode.milestone,
        "task": episode.task,
        "task_instruction": episode.task_instruction.strip(),
        "seed": episode.seed,
        "evidence_tier": episode.evidence_tier,
        "output_workspace": episode.output_workspace,
        "policy": policy,
        "runtime": runtime,
        "budgets": budgets,
        "policy_context": policy_context,
        "policy_composition": composition,
        "qualification": qualification,
        "prerequisite": spec.get("prerequisite"),
        "source": {"commit": source_commit},
    }
    if launch_admission is not None:
        portable["launch_admission"] = launch_admission
    if session is not None:
        portable["session"] = session
    if operator_control is not None:
        portable["operator_control"] = operator_control
    portable["policy_identity"] = bind_policy_identity(portable, root=root)
    plan = _seal_plan(
        spec, portable, episode=episode, source_commit=source_commit,
        source_dirty=source_dirty, root=root,
    )
    _validate_launch_admission_plan(plan)
    return plan


def _expand_defaults(spec: Mapping[str, Any], *, root: Path) -> Mapping[str, Any]:
    """Apply the maintained Native defaults to ``spec``."""

    from .defaults import expand_native_defaults

    try:
        spec = expand_native_defaults(spec, root=root)
    except (ValueError, OSError, EnvironmentContextError) as exc:
        raise RunSpecError(str(exc)) from exc
    return spec


def _resolve_episode(spec: Mapping[str, Any]) -> _Episode:
    experiment = _stable_id(spec.get("experiment"), "experiment")
    config = _stable_id(spec.get("config"), "config")
    milestone = _stable_id(spec.get("milestone"), "milestone")
    task = _stable_id(spec.get("task"), "task")
    task_instruction = spec.get("task_instruction")
    if not isinstance(task_instruction, str) or not task_instruction.strip():
        raise RunSpecError("task_instruction must be non-empty")
    seed = spec.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise RunSpecError("seed must be a non-negative integer")
    evidence_tier = spec.get("evidence_tier")
    if evidence_tier not in EVIDENCE_TIERS:
        raise RunSpecError(f"unsupported evidence_tier: {evidence_tier}")
    output_workspace = spec.get("output_workspace")
    expected_workspace = (Path("outputs") / experiment).as_posix()
    if output_workspace != expected_workspace:
        raise RunSpecError(f"output_workspace must be {expected_workspace}")
    return _Episode(
        experiment, config, milestone, task, task_instruction, seed,
        evidence_tier, output_workspace,
    )


def _refuse_memory(policy: dict[str, Any]) -> None:
    """Resolve the only Memory selection: ``memory_policy: disabled``.

    The field stays in every plan (baselines pin it); Task Memory is not part
    of this tree, so a binding or management block is refused.
    """

    policy.setdefault("memory_policy", "disabled")
    if policy["memory_policy"] != "disabled":
        raise RunSpecError("policy.memory_policy must be disabled")
    for field in ("memory_binding", "memory_management"):
        if field in policy:
            raise RunSpecError(f"policy.{field} is not supported")


def _resolve_qualification(
    spec: Mapping[str, Any], experiment: str, *, root: Path,
) -> dict[str, Any]:
    """Bind a required qualification manifest, tracked or generated."""

    qualification = dict(_mapping(spec.get("qualification", {}), "qualification"))
    qualification.setdefault("required", False)
    if not isinstance(qualification["required"], bool):
        raise RunSpecError("qualification.required must be boolean")
    if qualification["required"]:
        raw_manifest = qualification.get("manifest")
        relative_manifest = Path(raw_manifest) if isinstance(raw_manifest, str) else None
        if (
            relative_manifest is not None
            and relative_manifest.parts[:2] == ("outputs", experiment)
        ):
            qualification["manifest"], _manifest_path = _generated_resource(
                root, experiment, raw_manifest, "qualification.manifest"
            )
        else:
            qualification["manifest"] = _resource(
                root, raw_manifest, "qualification.manifest"
            )
    try:  # an expert-unsolved RoboTwin seed (qualification_exception.py)
        exception = resolve_exception(qualification, lambda value, label: _generated_resource(root, experiment, value, label)[0])
    except QualificationExceptionError as exc:
        raise RunSpecError(str(exc)) from exc
    if exception is not None:
        qualification["exception"] = exception
    return qualification


def _bind_launch_admission(
    spec: Mapping[str, Any], episode: _Episode, *, source_commit: str,
    runtime: Mapping[str, Any], root: Path,
) -> dict[str, Any] | None:
    """Normalize launch admission and validate its source-required gate."""

    experiment, config = episode.experiment, episode.config
    launch_admission = _launch_admission(
        spec.get("launch_admission"), experiment=experiment, config=config,
        root=root,
    )
    requirement = _launch_requirement(experiment, config, root=root)
    if (
        launch_admission is not None
        and launch_admission.get("mode") == "single_attempt_cell"
        and "gate" in launch_admission
    ):
        raw_gate = launch_admission["gate"]
        gate_resource, gate_path = _generated_resource(
            root, experiment, raw_gate, "launch_admission.gate",
        )
        if requirement is None:
            raise RunSpecError(
                "launch qualification gates require a source-registered cell"
            )
        _validate_launch_gate_record(
            experiment=experiment, config=config, gate_path=gate_path,
            requirement=requirement, source_commit=source_commit,
            runtime=runtime, root=root,
        )
        launch_admission["gate"] = gate_resource
    return launch_admission


def _seal_plan(
    spec: Mapping[str, Any], portable: dict[str, Any], *, episode: _Episode,
    source_commit: str, source_dirty: bool, root: Path,
) -> dict[str, Any]:
    """Add the experiment identity, then the plan hash, id and confirmation."""

    try:
        identity = bind_identity(spec, root=root)
    except ValueError as exc:
        raise RunSpecError(str(exc)) from exc
    if identity is not None:
        portable["experiment_identity"] = identity
    plan_hash = json_sha256(portable)
    return {
        **portable,
        "plan_sha256": plan_hash,
        "plan_id": f"{episode.experiment}-{episode.milestone.lower()}-{plan_hash[:12]}",
        "source": {"commit": source_commit, "dirty": source_dirty},
        "confirmation_code": plan_hash[-6:],
    }
