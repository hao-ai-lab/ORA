"""Seed-scan segments: plan a task as parallel seed ranges, seal a running scan, merge in seed order.

A task whose 100th expert-accepted seed lies far from the official start
(``put_object_cabinet`` needs about 650 candidates under ``demo_clean``) is
scanned as several (task, offset[, length]) cases, each its own lane. ``seal``
writes interim manifests from ledgers that are still growing; ``merge`` joins
each task's segments into one first-N manifest of the official schema, which
``robotwin_evaluation.seed_admission`` reads like any ``qualify`` manifest.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from ora.environment.benchmarks.robotwin import TASKS, evaluation_protocol
from ora.orchestrator.robotwin_evaluation import ROOT, SEED_MANIFEST_SCHEMA, digest, save, seed_manifests

INTERIM_FOLDER = "seed-manifests-interim"


def _offset(value) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def plan_segments(tasks, candidate_count: int, *, start_offsets=(0,), segments=None):
    """The (task, offset) cases of a scan and each case's length.

    ``start_offsets`` scans every task from the official start plus each
    offset; ``segments`` lists explicit (task, offset[, count]) entries
    instead, for tasks that need different numbers of segments. ``count``
    defaults to ``candidate_count`` and may not exceed it."""
    offsets = list(start_offsets)
    if not offsets or len(set(offsets)) != len(offsets) or not all(_offset(o) for o in offsets):
        raise ValueError("start_offsets must be distinct non-negative integers")
    triples = ([(task, offset, candidate_count) for offset in offsets for task in tasks] if segments is None
               else [(str(item[0]), item[1], item[2] if len(item) > 2 else candidate_count) for item in segments])
    pairs = [(task, offset) for task, offset, _ in triples]
    lengths = {(task, offset): count for task, offset, count in triples}
    if (len(set(pairs)) != len(pairs) or {task for task, _ in pairs} != set(tasks)
            or not all(_offset(o) for _, o in pairs)
            or any(isinstance(c, bool) or not isinstance(c, int) or not 1 <= c <= candidate_count
                   for c in lengths.values())):
        raise ValueError("segments must be distinct (task, non-negative offset[, 1..candidate_count]) "
                         "entries covering tasks")
    return pairs, lengths


def case_id(task: str, offset: int, start: int) -> str:
    """The task name for offset 0, else ``task@<first seed>``; it names the case's output and manifest."""
    return task if offset == 0 else f"{task}@{start + offset}"


def segment_cases(pairs, lengths, *, start: int, task_config: str, accepted_per_task: int, dependency_hash: str):
    """One qualification case per (task, offset); the accepted target never exceeds the length."""
    cases = []
    for task, offset in pairs:
        length = lengths[(task, offset)]
        target = min(accepted_per_task, length)
        cases.append({
            "task": task, "case_id": case_id(task, offset, start),
            "start_requested_seed": start + offset, "exclusive_seed_limit": start + offset + length,
            "accepted_target": target, "repeats": 1,
            "case_key": digest({"task": task, "task_config": task_config, "seed_start": start + offset,
                                "candidate_count": length, "accepted_target": target,
                                "producer_dependency_hash": dependency_hash}),
            "decision": "execute_missing_evaluation_evidence"})
    return cases


def parse_segments(items):
    """``TASK:OFFSET[:COUNT]`` command-line entries as tuples."""
    return None if items is None else [(item.split(":")[0], *map(int, item.split(":")[1:])) for item in items]


def seal_interim(destination: Path):
    """Seal manifests from a running scan's ledgers, without touching the final ones.

    Each case's manifest covers the seeds its worker has recorded so far; a
    case still scanning is ``complete: false``. ``merge`` with a small
    ``--accepted-per-task`` then yields a pilot list that is a prefix of the
    final one, because accepted seeds are kept in seed order."""
    plan = json.loads((destination / "prepared-plan.json").read_text())
    checksum = plan.pop("plan_sha256")
    # Interim manifests are derived and re-sealed each time; final ones are not.
    shutil.rmtree(destination / INTERIM_FOLDER, ignore_errors=True)
    return seed_manifests(destination, plan, checksum, plan.get("task_config", "demo_randomized"),
                          folder=INTERIM_FOLDER)


def task_dependency_hash(plan, task: str) -> str:
    """A plan's dependency closure restricted to one task.

    The plan-wide producer hash covers every scheduled task's envs/<task>.py,
    so two plans with different task sets differ even when every file is the
    same. This keeps the shared inputs, the checker and this task's own file.
    """
    root = Path(plan["robotwin_root"])
    others = {f"envs/{name}.py" for name in TASKS if name != task}
    items = []
    for item in plan["qualification"]["dependency_closure"]:
        path = Path(item["path"])
        if path.is_relative_to(root):
            if path.relative_to(root).as_posix() in others:
                continue
            relative = "robotwin/" + path.relative_to(root).as_posix()
        elif path.is_relative_to(ROOT):
            relative = "harness/" + path.relative_to(ROOT).as_posix()
        else:
            relative = path.as_posix()
        items.append({"input": relative, "sha256": item["sha256"]})
    return digest({"checker": plan["qualification"].get("checker_hash"),
                   "inputs": sorted(items, key=lambda entry: entry["input"])})


def _load_segment(path: Path, task_config: str):
    manifest = json.loads(path.read_text())
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if manifest.get("schema") != SEED_MANIFEST_SCHEMA or digest(body) != manifest.get("manifest_sha256"):
        raise ValueError(f"{path} is not a sealed seed manifest")
    if manifest.get("task_config") != task_config:
        raise ValueError(f"{path} is not a {task_config} manifest")
    plan_path = path.parent.parent / "prepared-plan.json"
    plan = json.loads(plan_path.read_text()) if plan_path.is_file() else {}
    if plan.get("robotwin_root") and "dependency_closure" in plan.get("qualification", {}):
        if plan.get("plan_sha256") != manifest.get("plan_sha256"):
            raise ValueError(f"{path} does not belong to {plan_path}")
        manifest = {**manifest, "_task_dependency_hash": task_dependency_hash(plan, manifest["task"])}
    return manifest


def _outcomes(task, ordered, conflicting_outcomes: str):
    """Each scanned seed's (kind, record) in segment order, and the seeds whose segments disagree."""
    outcome, conflicting = {}, set()
    for _, m in ordered:
        entries = [(seed, ("accepted", m["seeds"][str(seed)])) for seed in m["accepted_seeds"]]
        entries += [(rejected["requested_seed"], ("rejected", rejected)) for rejected in m["rejected"]]
        for seed, entry in entries:
            held = outcome.setdefault(seed, entry)
            if held == entry:
                continue
            if conflicting_outcomes == "refuse":
                raise ValueError(f"{task} seed {seed} has conflicting outcomes")
            if held[0] != entry[0] or seed in conflicting:
                conflicting.add(seed)
                outcome[seed] = ("rejected", {
                    "classification": "conflicting_outcomes", "requested_seed": seed,
                    "error": "segments disagree on the expert outcome of this seed"})
    return outcome, conflicting


def merge_seed_manifests(paths, *, task_config: str, accepted_target: int = 100, outer_seed: int = 0,
                         conflicting_outcomes: str = "refuse"):
    """Join a task's scanned segments into one first-N accepted-seed manifest.

    Segments come from ``qualify`` runs over disjoint seed ranges of one
    revision and producer. Walking seeds upward from the official start, the
    merged list keeps the first ``accepted_target`` accepted seeds of the
    contiguous scanned prefix, which is the set a single sequential scan would
    have collected. A gap before the target leaves the manifest incomplete;
    one seed recorded with two different outcomes is refused.

    ``conflicting_outcomes="reject"`` instead treats a seed whose segments
    disagree (accepted in one, rejected in another) as rejected, with
    classification ``conflicting_outcomes``, and keeps the earliest segment's
    record when segments agree on the outcome but differ in detail. The
    oracle's scene stability is not reproducible run to run on every host, so a
    seed the expert solved only sometimes is not an evaluation seed. The merged manifest lists those seeds.
    """
    if conflicting_outcomes not in {"refuse", "reject"}:
        raise ValueError("conflicting_outcomes must be refuse or reject")
    start = evaluation_protocol(outer_seed=outer_seed, task_config=task_config)["start_requested_seed"]
    by_task = {}
    for path in map(Path, paths):
        manifest = _load_segment(path, task_config)
        by_task.setdefault(manifest["task"], []).append((path, manifest))
    merged = {}
    for task, segments in sorted(by_task.items()):
        # Compare each segment's task-level closure when its plan is beside it;
        # a bare manifest falls back to the plan-wide producer hash.
        dependency = ("_task_dependency_hash" if all("_task_dependency_hash" in m for _, m in segments)
                      else "producer_dependency_hash")
        for field in ("robotwin_revision", dependency):
            if len({m.get(field) for _, m in segments}) != 1:
                raise ValueError(f"{task} segments differ in {field.lstrip('_')}")
        ordered = sorted(segments, key=lambda item: (item[1]["start_requested_seed"], item[1]["manifest_sha256"]))
        outcome, conflicting = _outcomes(task, ordered, conflicting_outcomes)
        accepted, rejected, evaluated, seeds = [], [], [], {}
        seed = start
        while seed in outcome and len(accepted) < accepted_target:
            kind, record = outcome[seed]
            evaluated.append(seed)
            if kind == "accepted":
                accepted.append(seed)
                seeds[str(seed)] = record
            else:
                rejected.append(record)
            seed += 1
        sources = sorted(({"path": str(path), "manifest_sha256": m["manifest_sha256"], "plan_sha256": m["plan_sha256"],
                           "source_commit": m["source_commit"], "start_requested_seed": m["start_requested_seed"],
                           "exclusive_seed_limit": m["exclusive_seed_limit"]} for path, m in segments),
                         key=lambda item: item["start_requested_seed"])
        commits = sorted({item["source_commit"] for item in sources})
        manifest = {
            "schema": SEED_MANIFEST_SCHEMA, "task": task, "task_config": task_config,
            "robotwin_revision": segments[0][1]["robotwin_revision"],
            "source_commit": commits[0] if len(commits) == 1 else ",".join(commits),
            "plan_sha256": digest([item["plan_sha256"] for item in sources]),
            "producer_dependency_hash": segments[0][1][dependency],
            "dependency_scope": "task" if dependency == "_task_dependency_hash" else "plan",
            "start_requested_seed": start, "exclusive_seed_limit": seed,
            "accepted_target": accepted_target, "accepted_seeds": accepted, "evaluated_seeds": evaluated,
            "rejected": rejected, "complete": len(accepted) >= accepted_target, "seeds": seeds,
            "oracle_success": bool(seeds) and all(s["oracle_success"] for s in seeds.values()),
            "reset_predicate_false": bool(seeds) and all(s["reset_predicate_false"] for s in seeds.values()),
            "post_success_physics_hold": bool(seeds) and all(s["post_success_physics_hold"] for s in seeds.values()),
            "unstable_accepted_seeds": [int(k) for k, s in seeds.items() if not s["post_success_physics_hold"]],
            **({"conflicting_outcomes": "reject", "conflicting_seeds": sorted(conflicting)}
               if conflicting_outcomes == "reject" else {}),
            "required_stability_steps": max(m["required_stability_steps"] for _, m in segments),
            "terminal_stability_steps": min((s["terminal_stability_steps"] for s in seeds.values()), default=0),
            "ledger_sha256": digest([m["ledger_sha256"] for _, m in segments]),
            "merged_from": sources,
            "next_unscanned_seed": None if len(accepted) >= accepted_target else seed,
        }
        manifest["manifest_sha256"] = digest(manifest)
        merged[task] = manifest
    return merged


def add_arguments(parser) -> None:
    """The ``prepare`` segment options and the ``merge``/``seal`` inputs."""
    parser.add_argument("--start-offsets", type=int, nargs="+", default=[0],
                        help="scan segments starting this many seeds after the official start (prepare)")
    parser.add_argument("--segments", nargs="+", metavar="TASK:OFFSET[:COUNT]",
                        help="explicit (task, offset[, length]) cases instead of tasks x start-offsets (prepare)")
    parser.add_argument("--process-attempt-limit", type=int, default=8,
                        help="seeds one oracle process scans before re-executing itself (prepare)")
    parser.add_argument("--inputs", type=Path, nargs="+",
                        help="qualification directories or seed manifests (merge); qualification directories (seal)")
    parser.add_argument("--output", type=Path, help="directory for merged per-task manifests (merge)")
    parser.add_argument("--conflicting-outcomes", choices=("refuse", "reject"), default="refuse",
                        help="merge: refuse a seed whose segments disagree, or count it as rejected")


def _manifest_paths(inputs):
    """Final manifests of a qualification directory, else its interim ones; files as given."""
    paths = []
    for item in inputs:
        if item.is_dir():
            folder = item / "seed-manifests"
            if not folder.is_dir() and (item / INTERIM_FOLDER).is_dir():
                folder = item / INTERIM_FOLDER
            paths.extend(sorted(folder.glob("*.json")))
        else:
            paths.append(item)
    return paths


def seal_command(args, parser) -> int:
    if not args.inputs:
        parser.error("seal requires --inputs <qualification directories>")
    for item in args.inputs:
        written = seal_interim(item)
        print(json.dumps({"qualification": str(item), "interim_manifests": len(written)}))
    return 0


def merge_command(args, parser) -> int:
    if not args.inputs or args.output is None:
        parser.error("merge requires --inputs and --output")
    paths = _manifest_paths(args.inputs)
    merged = merge_seed_manifests(paths, task_config=args.task_config,
                                  accepted_target=args.accepted_per_task, outer_seed=args.outer_seed,
                                  conflicting_outcomes=args.conflicting_outcomes)
    for task, manifest in merged.items():
        save(args.output / f"{task}.json", manifest)
    summary = {task: {"accepted": len(m["accepted_seeds"]), "complete": m["complete"],
                      "next_unscanned_seed": m["next_unscanned_seed"]} for task, m in merged.items()}
    save(args.output / "merge-report.json", {"task_config": args.task_config, "inputs": [str(p) for p in paths],
                                             "accepted_target": args.accepted_per_task, "tasks": summary})
    print(json.dumps({"output": str(args.output), "tasks": len(merged),
                      "complete": sum(m["complete"] for m in merged.values()),
                      "incomplete": {t: v for t, v in summary.items() if not v["complete"]}}, indent=2))
    return 0 if all(m["complete"] for m in merged.values()) else 2
