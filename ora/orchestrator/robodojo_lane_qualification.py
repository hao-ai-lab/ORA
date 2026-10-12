"""Qualify one RoboDojo lane for the RunSpecs ``robodojo_spec`` composes.

Every producer here measures ``robodojo_spec.qualification_runtime``, the
runtime such a RunPlan resolves to (its tabletop workspace bounds, physics
device, layout schedule and geometry descriptor), so preflight admits what it
writes for that plan:

- ``card``: the per-card, per-installation record
  (``robodojo_card_qualification.qualify_card``: two full-length episodes that
  witness the official terminal boundary).
- ``task``: the per-task, round-0 record on that card (``qualify_task``); its
  ``{card, task}`` pair qualifies a one-round session (``robodojo_spec
  --task-qualification``).
- ``smoke``: the bounded scene smoke (``robodojo_scene_smoke``), with one
  relaunch per later layout of a scheduled session; ``--layout k`` smokes the
  one-round episode on official layout k and reads back its instruction.
- ``revalidate``: no simulator; joins a card record with the newest passing
  smoke of exactly this lane into a ``ora.robodojo-revalidation.v1``
  certificate and writes the lane gate ``outputs/<experiment>/gates/
  <task>-l<layouts>-<device>.json`` (``robodojo_spec
  --qualification-reference``). This is how a K>=2 scheduled session is
  qualified on every layout, and a single episode on layout k by its own
  ``l<k>`` gate.

Maintained producers (integration 0062 U12) for what the earlier B2
RoboDojo scene smoke, layout smoke and revalidation produced; artifact schemas
are unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ora.environment.benchmarks.robodojo import (
    CARD_QUALIFICATION_SCHEMA, configuration_binding,
)
from ora.environment.benchmarks.robodojo_layouts import layout_cell_tag
from ora.orchestrator.paths import PROJECT_ROOT, WORKSPACE_ROOT, experiment_workspace
from ora.orchestrator.robodojo_revalidation import evaluate, reference
from ora.orchestrator.robodojo_runtime_config import qualification_device_label
from ora.orchestrator.robodojo_scene_smoke import OWNER_PREFIX as SMOKE_PREFIX, REPORT_SCHEMA

CHECKER = "ora/orchestrator/robodojo_revalidation.py"


def _runtime(task: str, *, gpu: str, resource_profile: str | None, rounds: int,
             layout_schedule: Sequence[int] | str | None, layout_id: int | None = None) -> dict[str, Any]:
    from ora.orchestrator.robodojo_spec import qualification_runtime
    if layout_id is None:
        return qualification_runtime(task, gpu=gpu, resource_profile=resource_profile, rounds=rounds,
                                     layout_schedule=layout_schedule)
    return qualification_runtime(task, gpu=gpu, resource_profile=resource_profile, rounds=rounds,
                                 layout_schedule=layout_schedule, layout_id=layout_id)


def card(task: str, *, gpu: str, resource_profiles: Sequence[str] = ()) -> Path:
    """The card record, measured on the lane runtime; two profiles run the episodes in parallel."""
    from ora.orchestrator.robodojo_card_qualification import qualify_card

    profiles = tuple(resource_profiles)
    if len(profiles) > 2:
        raise ValueError("a card runs on one resource profile, or two for parallel episodes")
    runtime = _runtime(task, gpu=gpu, resource_profile=profiles[0] if profiles else None, rounds=1,
                       layout_schedule=None)
    parallel = profiles if len(profiles) == 2 else None
    if parallel is not None:
        # Each parallel episode binds its own slot (qualify_card).
        runtime.pop("resource_binding")
    prefix = "robodojo-lane-card" + ("" if runtime["physics_device"] == "cuda:0" else "-cpu")
    return qualify_card(gpu, task, runtime=runtime, owner_prefix=prefix, parallel_resource_profiles=parallel)


def task_record(task: str, *, gpu: str, card_record: Path, resource_profile: str | None = None) -> Path:
    """The round-0 task record on ``card_record``, measured on the lane runtime."""
    from ora.orchestrator.robodojo_card_qualification import qualify_task

    runtime = _runtime(task, gpu=gpu, resource_profile=resource_profile, rounds=1, layout_schedule=None)
    return qualify_task(gpu, task, card_qualification=Path(card_record), runtime=runtime,
                        owner_prefix="robodojo-lane-task")


def card_reference(path: Path) -> dict[str, str]:
    """A card record's ``{path, sha256}``, from the record itself or a card gate that names it."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(value, Mapping) and isinstance(value.get("qualification"), Mapping):
        return dict(value["qualification"])
    if isinstance(value, Mapping) and value.get("schema") == CARD_QUALIFICATION_SCHEMA:
        if value.get("passed") is not True:
            raise ValueError(f"{path} is not a passing RoboDojo card qualification record")
        return reference(Path(path), WORKSPACE_ROOT)
    raise ValueError(f"{path} is neither a card qualification record nor a card gate")


def gate_path(experiment: str, runtime: Mapping[str, Any]) -> Path:
    """``outputs/<experiment>/gates/<task>-l<layouts>-<device>.json`` for one lane runtime.

    ``l<layouts>`` is the schedule's layouts, or the one layout of an
    unscheduled runtime (``l0`` by default).
    """
    tag = layout_cell_tag(runtime.get("layout_schedule"), runtime.get("layout_id", 0))
    device = qualification_device_label(runtime["cuda_visible_devices"], runtime["physics_device"])
    return experiment_workspace(experiment) / "gates" / f"{runtime['task_variant']}-{tag}-{device}.json"


def smoke_candidates(experiment: str, runtime: Mapping[str, Any]) -> list[Path]:
    """Passing smoke reports of exactly this lane: task, GPU, schedule and configuration (layout included)."""
    schedule = runtime.get("layout_schedule")
    layout_ids = None if schedule is None else [entry["layout_id"] for entry in schedule["rounds"]]
    found = []
    for path in (experiment_workspace(experiment) / "qualification").glob(f"{SMOKE_PREFIX}-*/report.json"):
        report = json.loads(path.read_text())
        plan_path = path.parent / "plan.json"
        if (report.get("schema") != REPORT_SCHEMA or report.get("passed") is not True
                or report.get("task") != runtime["task_variant"]
                or report.get("gpu") != runtime["cuda_visible_devices"]
                or report.get("layout_schedule") != layout_ids or not plan_path.is_file()):
            continue
        smoke_runtime = json.loads(plan_path.read_text()).get("runtime", {})
        try:
            if configuration_binding(smoke_runtime) != configuration_binding(runtime):
                continue
        except KeyError:
            continue
        found.append(path)
    return sorted(found, key=lambda item: item.stat().st_mtime)


def _source_commit() -> str:
    from ora.orchestrator.robodojo_card_qualification import _source_commit as clean_commit
    return clean_commit()


def revalidate(task: str, *, gpu: str, experiment: str, card: Path, resource_profile: str | None = None,
               rounds: int = 1, layout_schedule: Sequence[int] | str | None = None,
               layout_id: int | None = None) -> Path:
    """Certify this lane from a card record and its newest passing smoke; return the lane gate."""
    runtime = _runtime(task, gpu=gpu, resource_profile=resource_profile, rounds=rounds,
                       layout_schedule=layout_schedule, layout_id=layout_id)
    candidates = smoke_candidates(experiment, runtime)
    if not candidates:
        raise ValueError(f"no passing {task}/GPU{gpu}/{runtime['physics_device']} scene smoke of this lane "
                         f"and layout schedule under {experiment_workspace(experiment) / 'qualification'}")
    commit = _source_commit()
    result = evaluate(runtime, workspace_root=WORKSPACE_ROOT, harness_commit=commit,
                      card_reference=card_reference(card),
                      smoke_reference=reference(candidates[-1], WORKSPACE_ROOT))
    result["checker_source_commit"] = commit
    result["checker_sha256"] = hashlib.sha256((PROJECT_ROOT / CHECKER).read_bytes()).hexdigest()
    root = experiment_workspace(experiment) / "qualification" / "revalidation"
    root.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    record = root / (hashlib.sha256(data).hexdigest() + ".json")
    if not record.exists():
        record.write_bytes(data)
    gate = gate_path(experiment, runtime)
    gate.parent.mkdir(parents=True, exist_ok=True)
    gate.write_text(json.dumps(reference(record, WORKSPACE_ROOT), indent=2) + "\n")
    return gate


def _lane_arguments(command: argparse.ArgumentParser, *, schedule: bool) -> None:
    command.add_argument("task")
    command.add_argument("--gpu", required=True, choices=list("01234567"))
    if schedule:
        command.add_argument("--experiment", required=True, help="the experiment namespace that owns the lane")
        command.add_argument("--resource-profile", help="a registered RoboDojo slot (default robodojo-gpu<N>)")
        command.add_argument("--rounds", type=int, default=1, help="rounds of the persistent session (default 1)")
        command.add_argument("--layouts", type=int, nargs="+", help="per-round layout ids inside the reviewed pool")
        command.add_argument("--layout", type=int, dest="layout_id",
                             help="one-round episode on this official layout of seed 0 (default 0)")


def main(argv: list[str] | None = None) -> int:
    """Operator entry point: ``python -m ora.orchestrator.robodojo_lane_qualification``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    card_command = commands.add_parser("card", help="two full-length episodes on the lane runtime (35-56 min)")
    _lane_arguments(card_command, schedule=False)
    card_command.add_argument("--resource-profile", nargs="+", default=[], dest="resource_profiles",
                              help="one slot, or two for parallel episodes on the same GPU")
    task_command = commands.add_parser("task", help="the round-0 task record on a card record")
    _lane_arguments(task_command, schedule=False)
    task_command.add_argument("--resource-profile")
    task_command.add_argument("--card", type=Path, required=True)
    smoke_command = commands.add_parser("smoke", help="the bounded scene smoke of one lane")
    _lane_arguments(smoke_command, schedule=True)
    revalidate_command = commands.add_parser("revalidate", help="the lane gate, offline")
    _lane_arguments(revalidate_command, schedule=True)
    revalidate_command.add_argument("--card", type=Path, required=True,
                                    help="a card-qualification.json, or a card gate naming one")
    for command in (card_command, task_command, smoke_command):
        command.add_argument("--yes", action="store_true", help="confirm the bounded simulator episodes")
    args = parser.parse_args(argv)
    if args.command != "revalidate" and not args.yes:
        parser.error(f"{args.command} requires --yes to confirm its simulator episodes")
    if args.command == "card":
        print(card(args.task, gpu=args.gpu, resource_profiles=args.resource_profiles))
    elif args.command == "task":
        print(task_record(args.task, gpu=args.gpu, card_record=args.card, resource_profile=args.resource_profile))
    elif args.command == "smoke":
        from ora.orchestrator.robodojo_scene_smoke import smoke
        layout = {} if args.layout_id is None else {"layout_id": args.layout_id}
        report = smoke(args.task, gpu=args.gpu, experiment=args.experiment, resource_profile=args.resource_profile,
                       rounds=args.rounds, layout_schedule=args.layouts, **layout)
        return 0 if report["passed"] else 1
    else:
        layout = {} if args.layout_id is None else {"layout_id": args.layout_id}
        print(revalidate(args.task, gpu=args.gpu, experiment=args.experiment, card=args.card,
                         resource_profile=args.resource_profile, rounds=args.rounds,
                         layout_schedule=args.layouts, **layout))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
