"""Fresh RoboDojo RunSpecs from the default baseline plus task, provider and resource inputs.

New RoboDojo experiments start here, the way new RoboTwin and LIBERO-Pro
experiments start from the default resolver: the RunSpec this module writes
names no ToolSurface, prompt bundle, Context or semantic manifest, so
``defaults.expand_native_defaults`` composes the default catalog's RoboDojo
instance (``native_full_explicit_motion_direct_v4``: the
``robodojo_placement_closed_skills_v2`` surface, the ``current_v4`` bundle and
``native_slots_v4`` Context) and its reviewed task contract. What a RunSpec
owns comes from here: the task's reviewed runtime
(``robodojo_runtime_config.runtime_configuration``), its instruction from the
default task-contract registry, the GPU and execution resource, an optional
K-round persistent session with its per-round layout schedule, the Provider
profile, budgets, the infrastructure-qualification binding and the
resident-service binding. A one-round episode may instead run on another
official layout of evaluation seed 0 (``layout_id``, EXP14A g002): only where
the reviewed layout-portability table extends the task's contract to that
layout, and only with the scene-smoke lane gate of that layout, which also
supplies the instruction of a task whose instruction the layout generates.

It replaces the earlier B2 RoboDojo builder (the ``toolchain_b2_robodojo`` launcher),
whose composition came from the TOOLCHAIN-B2 archive and the EXP08A Context.
Composing a RunSpec is not host qualification, service
startup or authority to launch: ``scripts/ora plan|preflight|launch``
still resolve and check the written spec.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ora.agent.composition.robodojo_layout_compatibility import (
    INSTRUCTION_LAYOUT_READBACK, portable_layout,
)
from ora.agent.composition.robodojo_task_contract import REGISTRY_PATH, load_task_contract
from ora.environment.benchmarks.robodojo import NATIVE_ACTION_LIMITS, physics_step_ceiling
from ora.environment.benchmarks.robodojo_layouts import REPEAT_ROUND0_LAYOUT, layout_cell_tag
from ora.orchestrator.paths import PROJECT_ROOT, WORKSPACE_ROOT, experiment_workspace
from ora.orchestrator.robodojo_runtime_config import (
    robodojo_layout_schedule, runtime_configuration, with_layout_schedule,
)
# The Provider, service and admission inputs every maintained builder shares;
# their names stay importable from here.
from ora.orchestrator.run_spec_inputs import (  # noqa: F401
    PROVIDER_FIELDS, SERVICE_VARIABLES, SPEC_SCHEMA, bind_service_environment, provider_policy,
    single_attempt_admission,
)
from ora.orchestrator import run_spec_inputs

ENVIRONMENT = "robodojo"
#: The cadence of a K-round persistent session (``rounds`` given).
SESSION_CADENCE = "persistent_agent_k_rounds_v1"
#: Budgets of the B2 RoboDojo builder, the ones EXP11A and EXP11B ran; per
#: round in a persistent session. ``max_sim_steps`` defaults to the task's
#: official ceiling (``physics_step_ceiling``), as EXP11A g004 onward did.
DEFAULT_BUDGETS = {"max_robot_decisions": 80, "max_model_calls": 200, "wall_time_s": 21600}


def default_instance(root: Path = PROJECT_ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    """The default catalog and its RoboDojo instance; refused when it has none."""
    return run_spec_inputs.default_instance(ENVIRONMENT, root)


def contract_registry(root: Path = PROJECT_ROOT) -> dict[str, dict[str, Any]]:
    """The default task-contract registry's entries, in registry order."""
    return json.loads((root / REGISTRY_PATH).read_text(encoding="utf-8"))["tasks"]


def task_roster(root: Path = PROJECT_ROOT) -> tuple[str, ...]:
    """Every task with a reviewed contract in the default registry and a Runtime limit."""
    return tuple(task for task in contract_registry(root) if task in NATIVE_ACTION_LIMITS)


def task_contract(task: str, *, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """The task's record and completion layer as the default registry binds them."""
    entry = contract_registry(root).get(task)
    if entry is None:
        raise ValueError(f"{task} has no reviewed contract in {REGISTRY_PATH}")
    return load_task_contract(task, bindings={task: entry}, project_root=root)


def single_layout(task: str, layout_id: int | None, *, root: Path = PROJECT_ROOT) -> dict[str, Any] | None:
    """None for the reviewed layout 0; the audited portability row of another single-episode layout.

    Refuses an ordinal that is not a non-negative integer, and a layout the
    reviewed layout-portability table does not extend the task's contract to.
    """
    if layout_id is None or (type(layout_id) is int and layout_id == 0):
        return None
    if type(layout_id) is not int or layout_id < 0:
        raise ValueError(f"a RoboDojo layout id is a non-negative official ordinal, not {layout_id!r}")
    try:
        return portable_layout(task, layout_id, project_root=root, registry=REGISTRY_PATH)
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc


def runtime_for(task: str, *, gpu: str, resource_profile: str | None = None,
                rounds: int = 1, layout_schedule: Sequence[int] | str | None = None,
                layout_id: int | None = None, task_instruction: str | None = None,
                root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """The task's reviewed runtime on physical GPU ``gpu`` and its execution slot.

    The default instance owns the workspace profile and the resolver derives
    the public environment context, so neither is written here. ``rounds`` of
    two or more start each round on a new reviewed layout: ``layout_schedule``
    None takes the task's default schedule, a sequence of layout ids selects
    inside its reviewed pool, and ``repeat_round0_layout`` is the explicit
    historical treatment that replays layout 0 (no ``layout_schedule``).

    ``layout_id`` k > 0 runs a one-round episode on official layout k
    (``single_layout``). Its instruction is the record's when the table calls
    it ``fixed``; a ``layout_readback`` instruction is ``task_instruction``
    (the lane gate's readback), or is left unbound so the scene smoke reads it
    back from the official reset.
    """
    if task not in task_roster(root):
        raise ValueError(f"unknown or uncontracted RoboDojo task: {task}")
    portable = single_layout(task, layout_id, root=root)
    if portable is not None and (rounds >= 2 or layout_schedule is not None):
        raise ValueError("a single-layout RoboDojo episode is one round; a persistent session selects its "
                         "layouts with a layout schedule")
    runtime = (runtime_configuration(str(gpu), task) if portable is None
               else runtime_configuration(str(gpu), task, layout_id))
    for owned_elsewhere in ("workspace_bounds_m", "public_environment_context"):
        runtime.pop(owned_elsewhere)
    runtime["resource_profile"] = resource_profile or f"robodojo-gpu{gpu}"
    from ora.orchestrator.resources import resolve_execution_resource
    binding = resolve_execution_resource(
        runtime["resource_profile"], runtime_kind=ENVIRONMENT, required_host=runtime["required_host"])
    if binding["worker_environment"].get("CUDA_VISIBLE_DEVICES") != runtime["cuda_visible_devices"]:
        raise ValueError("the resource profile and the runtime GPU differ")
    instruction = task_contract(task, root=root)["record"]["instruction"]["text"]
    if portable is not None and portable["instruction"] == INSTRUCTION_LAYOUT_READBACK:
        instruction = task_instruction
    elif task_instruction not in (None, instruction):
        raise ValueError(f"{task} renders its reviewed instruction on every layout, not {task_instruction!r}")
    if instruction is not None:
        runtime["expected_task_language"] = instruction
    if layout_schedule == REPEAT_ROUND0_LAYOUT or rounds < 2:
        if layout_schedule not in (None, REPEAT_ROUND0_LAYOUT):
            raise ValueError("a layout schedule needs two or more rounds")
        return runtime
    ids = None if layout_schedule is None else [int(layout_id) for layout_id in layout_schedule]
    return with_layout_schedule(runtime, robodojo_layout_schedule(task, rounds, layout_ids=ids))


def qualification_runtime(task: str, *, gpu: str, resource_profile: str | None = None, rounds: int = 1,
                          layout_schedule: Sequence[int] | str | None = None, layout_id: int | None = None,
                          root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """The runtime a RunSpec built here resolves to, in the shape the infrastructure probes run.

    The card and task records and the scene smoke must measure the
    configuration the RunPlan will carry: ``robodojo_qualification`` compares
    their configuration binding (``workspace_bounds_m`` included) and
    ``robodojo_revalidation`` their public geometry descriptor with the plan's.
    So this applies the default instance exactly as the resolver does
    (``expand_native_defaults``), then binds the workspace profile's bounds,
    the default surface's geometry descriptor and the slot's resource binding.
    It carries no qualification binding: the probes produce one. On a
    single layout whose instruction the layout generates it carries no
    ``expected_task_language`` either: the scene smoke reads it back.
    """
    from ora.agent.composition import resolve_tool_surface
    from ora.environment.workspace_profiles import resolve_workspace_profile
    from ora.orchestrator.defaults import expand_native_defaults
    from ora.orchestrator.resources import resolve_execution_resource

    runtime = runtime_for(task, gpu=gpu, resource_profile=resource_profile, rounds=rounds,
                          layout_schedule=layout_schedule, layout_id=layout_id, root=root)
    expanded = expand_native_defaults(
        {"policy": {"kind": "native_api", "supports_images": True}, "runtime": runtime}, root=root)
    runtime = expanded["runtime"]
    runtime["workspace_bounds_m"] = resolve_workspace_profile(runtime.pop("workspace_profile"))["bounds_m"]
    surface = resolve_tool_surface(expanded["policy"]["tool_surface"]["profile"])
    runtime["public_geometry_descriptor"] = copy.deepcopy(surface.view_config["public_geometry_descriptor"])
    runtime["resource_binding"] = resolve_execution_resource(
        runtime["resource_profile"], runtime_kind=ENVIRONMENT, required_host=runtime["required_host"])
    return runtime


def service_prerequisite(report: Path, *, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Bind a readiness report (``scripts/model-services qualify``) to the default RoboDojo surface."""
    return run_spec_inputs.service_prerequisite(report, ENVIRONMENT, root=root)


def _reference(path: Path) -> dict[str, str]:
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(WORKSPACE_ROOT.resolve()):
        raise ValueError(f"a qualification record must live in the shared workspace: {resolved}")
    return {"path": resolved.relative_to(WORKSPACE_ROOT.resolve()).as_posix(),
            "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest()}


def layered_qualification(task_record: Path) -> dict[str, Any]:
    """The card/task pair a task record names (``robodojo_lane_qualification task``).

    Records from ``robodojo_card_qualification`` without a supplied runtime
    measure the wider compat workspace bounds, so a default RunPlan refuses
    their card configuration at preflight; produce them with
    ``robodojo_lane_qualification``, which measures the plan's own runtime.
    """
    record = json.loads(Path(task_record).read_text(encoding="utf-8"))
    if record.get("passed") is not True or not isinstance(record.get("card_qualification"), Mapping):
        raise ValueError(f"{task_record} is not a passing RoboDojo task qualification record")
    return {"card": dict(record["card_qualification"]), "task": _reference(Path(task_record))}


def qualification_reference(path: Path) -> dict[str, Any]:
    """A binding stored in a file: a ``{path, sha256}`` record or a ``{card, task}`` pair (a lane gate)."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or set(value) not in ({"path", "sha256"}, {"card", "task"}):
        raise ValueError(f"{path} holds no RoboDojo qualification binding")
    return dict(value)


def gate_instruction(binding: Mapping[str, Any], *, task: str, layout_id: int) -> str:
    """The official instruction a lane gate's scene smoke read back on ``task``'s layout ``layout_id``.

    ``binding`` is the gate's ``{path, sha256}`` revalidation certificate;
    its configuration must be this task and layout. Launch admission
    re-evaluates the same certificate against the RunPlan's instruction.
    """
    from ora.orchestrator.robodojo_revalidation import SCHEMA, read_reference

    if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
        raise ValueError(f"layout {layout_id} of {task} needs the scene-smoke lane gate of that layout "
                         "(robodojo_lane_qualification smoke + revalidate --layout), not a card/task pair")
    try:
        record, _ = read_reference(dict(binding), WORKSPACE_ROOT)
    except Exception as exc:
        raise ValueError(f"the lane gate of layout {layout_id} of {task} is unreadable: {exc}") from exc
    configuration = record.get("configuration") or {}
    instruction = record.get("task_instruction")
    if (record.get("schema") != SCHEMA or configuration.get("task_variant") != task
            or configuration.get("layout_id") != layout_id or not isinstance(instruction, str)
            or not instruction.strip()):
        raise ValueError(f"the lane gate binds no scene-smoke instruction of layout {layout_id} of {task}")
    return instruction


def build_spec(task: str, *, experiment: str, gpu: str, infrastructure_qualification: Mapping[str, Any],
               config: str | None = None, resource_profile: str | None = None,
               provider_profile: str | None = None, provider_catalog: Path | None = None,
               provider_recovery_window_s: float | None = None, rounds: int | None = None,
               layout_schedule: Sequence[int] | str | None = None, layout_id: int | None = None,
               budgets: Mapping[str, int] | None = None, service_report: Path | None = None,
               cohort_id: str | None = None, milestone: str = "ROBODOJO",
               evidence_tier: str = "behavioral", root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """One RoboDojo cell on the default baseline; only the RunSpec-owned choices are inputs.

    ``rounds`` None is the standard single episode; an integer is a K-round
    persistent session whose budgets are per round. ``layout_id`` k > 0 runs
    the single episode on official layout k and needs the lane gate of that
    layout. ``cohort_id`` adds a single-attempt launch admission for the cell.
    """
    default_instance(root)
    portable = single_layout(task, layout_id, root=root)
    instruction = None
    if portable is not None:
        if set(infrastructure_qualification) != {"path", "sha256"}:
            # A task record qualifies layout 0 only.
            raise ValueError(f"layout {layout_id} of {task} needs the scene-smoke lane gate of that layout "
                             "(robodojo_lane_qualification smoke + revalidate --layout), not a card/task pair")
        if portable["instruction"] == INSTRUCTION_LAYOUT_READBACK:
            instruction = gate_instruction(infrastructure_qualification, task=task, layout_id=layout_id)
    runtime = runtime_for(task, gpu=gpu, resource_profile=resource_profile, rounds=rounds or 1,
                          layout_schedule=layout_schedule, layout_id=layout_id, task_instruction=instruction,
                          root=root)
    if "layout_schedule" in runtime and set(infrastructure_qualification) == {"card", "task"}:
        # A task record qualifies the round-0 layout only; a scheduled session
        # needs the lane gate of a scene smoke that relaunched every layout.
        raise ValueError("a layout-scheduled RoboDojo session needs a scene-smoke lane gate "
                         "(robodojo_lane_qualification smoke + revalidate), not a card/task pair")
    runtime["infrastructure_qualification"] = copy.deepcopy(dict(infrastructure_qualification))
    policy = provider_policy(provider_profile, catalog=provider_catalog)
    if provider_recovery_window_s is not None:
        policy["provider_recovery_window_s"] = provider_recovery_window_s
    cell = config or f"{task}-{layout_cell_tag(runtime.get('layout_schedule'), runtime['layout_id'])}-s0"
    spec: dict[str, Any] = {
        "schema": SPEC_SCHEMA, "experiment": experiment, "config": cell,
        "output_workspace": f"outputs/{experiment}", "milestone": milestone,
        "task": task, "task_instruction": runtime["expected_task_language"], "seed": 0,
        "evidence_tier": evidence_tier, "policy": policy, "runtime": runtime,
        "budgets": {**DEFAULT_BUDGETS, "max_sim_steps": physics_step_ceiling(task), **dict(budgets or {})},
        "qualification": {"required": False},
    }
    if rounds is not None:
        spec["session"] = {"cadence": SESSION_CADENCE, "rounds": int(rounds)}
    if service_report is not None:
        spec["prerequisite"] = service_prerequisite(service_report, root=root)
    if cohort_id is not None:
        spec["launch_admission"] = single_attempt_admission(cohort_id, cell)
    return spec


def compose(spec: Mapping[str, Any]) -> Path:
    """Write ``spec`` under its experiment workspace, where plan/preflight/launch read it."""
    bind_service_environment(spec)
    path = experiment_workspace(spec["experiment"]) / "specs" / f"{spec['config']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _qualification_argument(args: argparse.Namespace) -> dict[str, Any]:
    if args.task_qualification is not None:
        return layered_qualification(args.task_qualification)
    return qualification_reference(args.qualification_reference)


def main(argv: list[str] | None = None) -> int:
    """Operator entry point: ``python -m ora.orchestrator.robodojo_spec``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("tasks", help="list the contracted tasks with their physics device and limits")
    for name, text in (("spec", "print the RunSpec"), ("compose", "write the RunSpec under outputs/<experiment>/specs")):
        command = commands.add_parser(name, help=text)
        command.add_argument("task")
        command.add_argument("--experiment", required=True, help="the new experiment's execution namespace")
        command.add_argument("--gpu", required=True, choices=list("01234567"))
        command.add_argument("--resource-profile", help="a registered RoboDojo slot (default robodojo-gpu<N>)")
        command.add_argument("--provider-profile", help="default: the shared catalog's default profile")
        command.add_argument("--provider-catalog", type=Path, help="an experiment-owned Provider catalog")
        command.add_argument("--rounds", type=int, help="a K-round persistent session (default: one episode)")
        command.add_argument("--layouts", type=int, nargs="+", help="per-round layout ids inside the reviewed pool")
        command.add_argument("--layout", type=int, dest="layout_id",
                             help="one episode on this official layout of seed 0 (default 0); needs its lane gate")
        command.add_argument("--service-report", type=Path, help="readiness report from `scripts/model-services qualify`")
        command.add_argument("--cohort", help="single-attempt launch admission cohort id")
        command.add_argument("--config", help="cell id (default <task>-l<layouts>-s0)")
        source = command.add_mutually_exclusive_group(required=True)
        source.add_argument("--task-qualification", type=Path,
                            help="task-qualification.json from robodojo_lane_qualification task (one round)")
        source.add_argument("--qualification-reference", type=Path,
                            help="a lane gate from robodojo_lane_qualification revalidate, or another file "
                                 "holding a {path, sha256} or {card, task} binding")
    args = parser.parse_args(argv)
    if args.command == "tasks":
        from ora.environment.benchmarks.robodojo_physics import physics_device_for
        print(json.dumps({task: {"physics_device": physics_device_for(task),
                                 "native_action_limit": NATIVE_ACTION_LIMITS[task],
                                 "max_sim_steps": physics_step_ceiling(task)} for task in task_roster()},
                         indent=2))
        return 0
    spec = build_spec(
        args.task, experiment=args.experiment, gpu=args.gpu,
        infrastructure_qualification=_qualification_argument(args), config=args.config,
        resource_profile=args.resource_profile, provider_profile=args.provider_profile,
        provider_catalog=args.provider_catalog, rounds=args.rounds, layout_schedule=args.layouts,
        layout_id=args.layout_id, service_report=args.service_report, cohort_id=args.cohort,
    )
    if args.command == "spec":
        print(json.dumps(spec, ensure_ascii=False, indent=2))
    else:
        print(compose(spec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
