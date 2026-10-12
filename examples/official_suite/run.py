"""Compose the official-protocol LIBERO-Pro and RoboTwin suites on the default baseline.

Every cell is one episode on one seed or initial state. The tool surface,
prompt bundle, embodiment card and Context come from the default baseline: the
RunSpec names none of them, so the default resolver composes the default
catalog's instance for each benchmark. This example selects only the benchmark
cell, the Provider profile, budgets, the resident-service binding and the
qualification evidence.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from ora.agent.composition import resolve_tool_surface
from ora.agent.composition.baseline import load_native_baseline
from ora.agent.model_services import operator as services
from ora.agent.providers.profiles import load_builtin_provider_profiles, load_provider_profiles
from ora.environment.benchmarks.libero_pro import CORE_VARIANT_ORDER, FAMILY_ORDER, official_core_evaluation
from ora.environment.benchmarks.robotwin import TASKS as ROBOTWIN_TASKS, TASK_CONFIGS
from ora.orchestrator import paths
from ora.orchestrator.resources import HOST_MARKER_ENV

EXPERIMENT = "official-suite"
MILESTONE = "OFFICIAL-SUITE"
SEED_MANIFEST_SCHEMA = "ora.robotwin-evaluation-seed-manifest.v1"
PROVIDER_FIELDS = (
    "endpoint", "model", "model_env", "base_url_env", "api_key_env", "proxy_env",
    "reasoning_effort", "context_window_tokens", "max_output_tokens", "stream",
    "supports_images", "endpoint_transport",
)


def definition() -> dict[str, Any]:
    value = yaml.safe_load((PACKAGE / "experiment.yaml").read_text())
    if value.get("experiment") != EXPERIMENT or set(value.get("arms", {})) != {"libero_pro", "robotwin"}:
        raise RuntimeError("official suite definition drifted")
    return value


def workspace() -> Path:
    return paths.experiment_workspace(EXPERIMENT)


def declared_host(host: str | None) -> str:
    value = host or os.environ.get(HOST_MARKER_ENV)
    if not value:
        raise RuntimeError(f"an execution host is required: --host or {HOST_MARKER_ENV}")
    return value


@dataclass(frozen=True)
class Cell:
    config: str
    arm: str
    task: str
    seed: int
    instruction: str
    runtime: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)


def libero_cells(*, families=None, perturbations=None, init_states: int | None = None) -> list[Cell]:
    """Official LIBERO-Pro core cells; instructions from the pinned task catalog."""
    from viewer.libero_pro_catalog import TASK_BY_KEY

    kwargs: dict[str, Any] = {}
    if families is not None:
        kwargs["families"] = list(families)
    if perturbations is not None:
        kwargs["perturbations"] = list(perturbations)
    if init_states is not None:
        kwargs["init_states"] = init_states
    protocol = official_core_evaluation(**kwargs)
    cells = []
    for item in protocol["cells"]:
        variant = TASK_BY_KEY[(item["family"], item["task_id"])].variant(item["perturbation"])
        if variant.suite != item["suite"]:
            raise RuntimeError(f"catalog suite {variant.suite} differs from {item['suite']}")
        task = f"{item['suite']}-t{item['task_id']:02d}"
        cells.append(Cell(
            config=f"lp-{item['suite']}-t{item['task_id']:02d}-s{item['seed']}", arm="libero_pro",
            task=task, seed=item["seed"], instruction=variant.instruction,
            runtime={"suite": item["suite"], "task_id": item["task_id"]},
            evidence={"protocol": protocol["schema"], "label": item["label"],
                      "libero_pro_code_revision": protocol["libero_pro_code_revision"],
                      "libero_pro_data_revision": protocol["libero_pro_data_revision"]},
        ))
    return cells


def load_instructions() -> dict[str, Any]:
    """The maintained public RoboTwin instructions (moved out of this package 2026-10-05)."""
    from ora.environment.benchmarks.robotwin_instructions import public_instructions

    return public_instructions()


def load_seed_manifest(manifests: Path, task: str, task_config: str) -> dict[str, Any]:
    path = manifests / f"{task}.json"
    manifest = json.loads(path.read_text())
    if (manifest.get("schema") != SEED_MANIFEST_SCHEMA or manifest.get("task") != task
            or manifest.get("task_config") != task_config):
        raise RuntimeError(f"{path} is not a {task_config} seed manifest for {task}")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != manifest.get("manifest_sha256"):
        raise RuntimeError(f"{path} content digest differs from its manifest_sha256")
    return {**manifest, "path": str(path)}


def robotwin_cells(*, task_config: str, manifests: Path, tasks=None,
                   episode_names: Path | None = None, accepted_limit: int | None = None) -> list[Cell]:
    """One cell per expert-accepted seed a sealed manifest lists, per task."""
    if task_config not in TASK_CONFIGS:
        raise RuntimeError("task_config must be demo_clean or demo_randomized")
    selected = list(tasks) if tasks is not None else list(ROBOTWIN_TASKS)
    if len(set(selected)) != len(selected) or not set(selected) <= set(ROBOTWIN_TASKS):
        raise RuntimeError("tasks must be distinct members of the fifty-task roster")
    instructions = load_instructions()
    names = json.loads(episode_names.read_text())["episodes"] if episode_names is not None else {}
    short = "clean" if task_config == "demo_clean" else "random"
    cells = []
    for task in selected:
        manifest = load_seed_manifest(manifests, task, task_config)
        seeds = list(manifest["accepted_seeds"])
        if accepted_limit is not None:
            seeds = seeds[:accepted_limit]
        entry = instructions[task]
        for seed in seeds:
            if entry.get("seed_named"):
                public = names.get(f"{task}:s{seed}")
                if not public or not public.get("instruction"):
                    raise RuntimeError(
                        f"{task} names the objects its seed spawns; --episode-names lacks {task}:s{seed}")
                instruction = str(public["instruction"])
            else:
                instruction = str(entry["instruction"])
            cells.append(Cell(
                config=f"rt-{short}-{task}-s{seed}", arm="robotwin", task=task, seed=seed,
                instruction=instruction, runtime={"task_config": task_config},
                evidence={"seed_manifest": manifest["path"], "seed_manifest_sha256": manifest["manifest_sha256"],
                          "robotwin_revision": manifest["robotwin_revision"], "complete": manifest["complete"]},
            ))
    return cells


def provider_policy(arm: str, profile_id: str | None = None) -> dict[str, Any]:
    value = definition()
    selected = profile_id or value["arms"][arm]["provider_profile"]
    catalog = load_builtin_provider_profiles()
    extra = value.get("provider_catalog")
    if extra:
        catalog = {**catalog, **load_provider_profiles(ROOT / extra)}
    if selected not in catalog:
        raise RuntimeError(f"provider profile is not registered: {selected}")
    profile = catalog[selected]
    policy = {"kind": "native_api", "provider": profile["protocol"], "provider_profile": selected,
              **{key: copy.deepcopy(profile[key]) for key in PROVIDER_FIELDS if key in profile},
              "provider_qualification_schema": profile["qualification_schema"],
              "provider_qualification_version": profile["qualification_version"]}
    if profile.get("session_affinity") is not None:
        policy["provider_session_affinity"] = profile["session_affinity"]
    return policy


def _import_seed_manifest(cell: Cell) -> str:
    source = Path(cell.evidence["seed_manifest"])
    target = workspace() / "qualification" / f"{cell.task}-{cell.runtime['task_config']}-seed-manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    data = source.read_bytes()
    if target.exists() and target.read_bytes() != data:
        raise RuntimeError(f"imported seed manifest changed: {target}")
    if not target.exists():
        target.write_bytes(data)
    return f"outputs/{EXPERIMENT}/qualification/{target.name}"


def build_spec(cell: Cell, lane: str, *, host: str | None = None, service_report: Path | None = None,
               provider_profile: str | None = None) -> dict[str, Any]:
    value = definition()
    arm = value["arms"][cell.arm]
    runtime: dict[str, Any] = {"kind": cell.arm, "required_host": declared_host(host),
                               "resource_profile": lane, **copy.deepcopy(cell.runtime)}
    qualification: dict[str, Any] = {"required": False}
    if cell.arm == "libero_pro":
        runtime["expected_task_language"] = cell.instruction
    else:
        # runtime.task binds the reviewed completion contract; without it the
        # plan records task_contract.delivered false.
        runtime["task"] = cell.task
        qualification = {"required": True, "manifest": _import_seed_manifest(cell)}
    spec = {
        "schema": "ora.run-spec.v1", "experiment": EXPERIMENT,
        "config": cell.config, "output_workspace": f"outputs/{EXPERIMENT}", "milestone": MILESTONE,
        "task": cell.task, "task_instruction": cell.instruction, "seed": cell.seed,
        "evidence_tier": "behavioral",
        "policy": provider_policy(cell.arm, provider_profile),
        "runtime": runtime, "budgets": dict(arm["budgets"]), "qualification": qualification,
        "launch_admission": {"schema": "ora.run-launch-admission.v1", "mode": "single_attempt_cell",
                             "cohort_id": value["cohort_id"], "cell_id": cell.config, "retry_allowed": False},
    }
    if service_report is not None:
        catalog = load_native_baseline(ROOT)
        surface_id = catalog["instances"][cell.arm]["tool_surface"]
        binding = services.binding(service_report, surface_id, catalog["required_service_capabilities"])
        if set(binding["capability_scope"]) - set(resolve_tool_surface(surface_id).capabilities):
            raise RuntimeError("default surface lacks the qualified resident service capabilities")
        spec["prerequisite"] = binding
    return spec


def compose(cell: Cell, lane: str, **kwargs: Any) -> Path:
    spec = build_spec(cell, lane, **kwargs)
    target = workspace() / "specs" / f"{cell.config}.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(spec, sort_keys=False))
    return target


def select_cells(args: argparse.Namespace) -> list[Cell]:
    if args.arm == "libero_pro":
        return libero_cells(families=args.families, perturbations=args.perturbations, init_states=args.init_states)
    if args.seed_manifests is None:
        raise RuntimeError("robotwin cells require --seed-manifests DIR from robotwin_evaluation qualify")
    return robotwin_cells(task_config=args.task_config, manifests=args.seed_manifests, tasks=args.tasks,
                          episode_names=args.episode_names, accepted_limit=args.accepted_limit)


def add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--arm", choices=("libero_pro", "robotwin"), required=True)
    parser.add_argument("--families", nargs="+", choices=FAMILY_ORDER)
    parser.add_argument("--perturbations", nargs="+", choices=CORE_VARIANT_ORDER)
    parser.add_argument("--init-states", type=int, help="initial states per LIBERO task (official: 50)")
    parser.add_argument("--task-config", choices=sorted(TASK_CONFIGS), default="demo_randomized")
    parser.add_argument("--seed-manifests", type=Path, help="seed-manifests directory written by robotwin_evaluation qualify")
    parser.add_argument("--tasks", nargs="+", choices=ROBOTWIN_TASKS)
    parser.add_argument("--accepted-limit", type=int, help="use at most this many accepted seeds per task (pilot)")
    parser.add_argument("--episode-names", type=Path, help="public episode names for seed-named tasks")


def add_binding_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", help=f"registered execution host; default {HOST_MARKER_ENV}")
    parser.add_argument("--service-report", type=Path, help="perception readiness report from `services qualify`")
    parser.add_argument("--provider-profile", help="override the arm's Provider profile id")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    cells = sub.add_parser("cells", help="list the selected cells without composing")
    add_selection_arguments(cells)
    compose_parser = sub.add_parser("compose", help="write one cell's RunSpec")
    add_selection_arguments(compose_parser)
    add_binding_arguments(compose_parser)
    compose_parser.add_argument("--config", required=True)
    compose_parser.add_argument("--lane", required=True, help="resource profile registered on the host")
    qualify = sub.add_parser("services", help="qualify the resident SAM3 and grasp services")
    qualify.add_argument("action", choices=("qualify",))
    queue = sub.add_parser("queue", help="launch the selected cells across lanes, one attempt each")
    add_selection_arguments(queue)
    add_binding_arguments(queue)
    queue.add_argument("--lanes", nargs="+", required=True, help="resource profiles, one per concurrent lane")
    queue.add_argument("--group-id", required=True)
    queue.add_argument("--resume", action="store_true", help="skip cells sealed by earlier groups of this cohort")
    args = parser.parse_args(argv)
    if args.command == "services":
        print(services.qualify(workspace() / "qualification" / "services"))
        return 0
    selected = select_cells(args)
    if args.command == "cells":
        print(json.dumps({"experiment": EXPERIMENT, "arm": args.arm, "count": len(selected),
                          "configs": [cell.config for cell in selected]}, indent=2))
        return 0
    if args.command == "compose":
        cell = next((item for item in selected if item.config == args.config), None)
        if cell is None:
            raise SystemExit(f"{args.config} is not in the selection")
        print(compose(cell, args.lane, host=args.host, service_report=args.service_report,
                      provider_profile=args.provider_profile))
        return 0
    from examples.official_suite.scheduler import run_queue
    return run_queue(selected, lanes=args.lanes, group_id=args.group_id, host=args.host,
                     service_report=args.service_report, provider_profile=args.provider_profile,
                     resume=args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
