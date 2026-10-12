"""Prepare the 50-task evaluation scene contract; qualify only explicit cases."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ora.environment.benchmarks.robotwin import TASK_CONFIGS, TASKS, evaluation_protocol, validate_task_config
from ora.orchestrator.paths import resolve_outputs_root, resolve_workspace_root
from ora.orchestrator.resources import declared_execution_host, resolve_execution_resource

# Under the shared outputs root (ORA_OUTPUTS_ROOT), not the node-local workspace.
WORKSPACE = resolve_outputs_root(resolve_workspace_root(ROOT), project_root=ROOT) / "runtime-v1" / "robotwin-evaluation"
#: The vendored install the Runtime loads (vendored_install.INSTALL_REVISION);
#: accepted seeds are bound to it, as the scenes evaluation loads are.
REQUIRED_REVISION = "cf4872cc90538fd4bca761c8db912d8f65e49f2b"
SEED_MANIFEST_SCHEMA = "ora.robotwin-evaluation-seed-manifest.v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


#: An expert-unsolved seed is run only on scan evidence that the expert failed or raised on it.
EXPERT_UNSOLVED_CLASSIFICATIONS = frozenset({"oracle_failure", "oracle_exception"})


def expert_unsolved(manifest, *, task, seed, task_config, revision):
    """Whether an official manifest shows the expert failed ``seed`` of ``task``, and the evidence.

    It is the opposite verdict of ``seed_admission`` on the same manifest:
    the seed was evaluated and not accepted, it is not a conflicting seed, and
    every rejection recorded for it is an expert failure or exception. An
    unstable scene, a reset that already satisfies the task or an
    infrastructure failure is not an expert-unsolved seed and is refused.
    """
    rejections = [item for item in manifest.get("rejected") or []
                  if isinstance(item, dict) and item.get("requested_seed") == seed]
    digest_valid = digest({k: v for k, v in manifest.items() if k != "manifest_sha256"}) == manifest.get("manifest_sha256")
    passed = (
        manifest.get("schema") == SEED_MANIFEST_SCHEMA
        and manifest.get("task") == task
        and manifest.get("task_config") == task_config
        and manifest.get("robotwin_revision") == revision
        and digest_valid
        and seed in (manifest.get("evaluated_seeds") or [])
        and seed not in (manifest.get("accepted_seeds") or [])
        and seed not in (manifest.get("conflicting_seeds") or [])
        and bool(rejections)
        and all(item.get("classification") in EXPERT_UNSOLVED_CLASSIFICATIONS for item in rejections)
    )
    return passed, {"task": manifest.get("task"), "task_config": manifest.get("task_config"),
                    "robotwin_revision": manifest.get("robotwin_revision"), "manifest_sha256_valid": digest_valid,
                    "rejections": [item.get("classification") for item in rejections]}


def seed_admission(manifest, *, task, seed, task_config, revision):
    """Whether a seed manifest admits ``seed`` of ``task``, and the evidence for the verdict.

    Launch preflight (``seed_qualification``) and the maintained RunSpec
    builder (``benchmark_spec``) apply this one rule. An official-scale
    manifest (``SEED_MANIFEST_SCHEMA``, written by ``qualify``) binds a scene
    configuration and revision and carries per-seed evidence; the seed is
    admitted on its own reset/oracle record. The post-success hold is evidence,
    never admission: upstream collects such seeds and the benchmark's verdict
    latches on success. Any other manifest is the older per-task shape, which
    names no scene configuration or revision: it admits a listed seed when its
    reset/oracle flags hold and the terminal hold lasted 250 steps.
    """
    accepted = manifest.get("accepted_seeds", [])
    if manifest.get("schema") == SEED_MANIFEST_SCHEMA:
        record = (manifest.get("seeds") or {}).get(str(seed)) or {}
        digest_valid = digest({k: v for k, v in manifest.items() if k != "manifest_sha256"}) == manifest.get("manifest_sha256")
        passed = (
            manifest.get("task") == task
            and manifest.get("task_config") == task_config
            and manifest.get("robotwin_revision") == revision
            and digest_valid
            and seed in accepted
            and record.get("reset_predicate_false") is True
            and record.get("oracle_success") is True
        )
        return passed, {
            "schema": manifest.get("schema"), "task": manifest.get("task"),
            "task_config": manifest.get("task_config"), "expected_task_config": task_config,
            "robotwin_revision": manifest.get("robotwin_revision"), "expected_revision": revision,
            "manifest_sha256_valid": digest_valid, "accepted": seed in accepted, "seed": record,
        }
    passed = (
        manifest.get("task") == task
        and seed in accepted
        and manifest.get("reset_predicate_false") is True
        and manifest.get("oracle_success") is True
        and int(manifest.get("terminal_stability_steps", 0)) >= 250
    )
    return passed, {
        "task": manifest.get("task"), "accepted_seeds": accepted,
        "reset_predicate_false": manifest.get("reset_predicate_false"),
        "oracle_success": manifest.get("oracle_success"),
        "terminal_stability_steps": manifest.get("terminal_stability_steps"),
    }


def file_binding(path: Path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, timeout=30).strip()


def save(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def source_dependencies(robotwin_root: Path, tasks, task_config: str = "demo_randomized"):
    # Execution dependencies only. HEAD is separately recorded as provenance.
    files = [ROOT / p for p in (
        "ora/environment/backends/robotwin/qualification.py",
        "ora/environment/backends/robotwin/session.py",
        # The evaluator-state helper lives in session.py; the error it raises
        # moved to common.py with the 2026-10-01 session split.
        "ora/environment/backends/robotwin/common.py",
        "ora/environment/benchmarks/robotwin.py",
        "uv.lock", "pyproject.toml",
    )]
    files += [robotwin_root / "envs" / f"{task}.py" for task in tasks]
    files += [robotwin_root / p for p in (
        "envs/_base_task.py", f"task_config/{task_config}.yml",
        "task_config/_embodiment_config.yml", "task_config/_camera_config.yml",
        "task_config/_eval_step_limit.yml", "script/eval_policy.py",
        "assets/objects/objaverse/list.json", "assets/objects/same.json",
    )]
    files += sorted((robotwin_root / "envs" / "utils").glob("*.py"))
    files += sorted((robotwin_root / "envs" / "robot").glob("*.py"))
    files += sorted((robotwin_root / "envs" / "camera").glob("*.py"))
    return [file_binding(path) for path in files]


def capacity():
    # The lease probe's bound: a busy scan node answers nvidia-smi in ~13 s.
    from ora.orchestrator.resource_inventory import GPU_MEMORY_PROBE_TIMEOUT_S

    result = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu", "--format=csv,noheader,nounits",
    ], text=True, timeout=GPU_MEMORY_PROBE_TIMEOUT_S)
    gpus = {}
    for line in result.splitlines():
        device, free, utilization = (part.strip() for part in line.split(","))
        if not (free.isdigit() and utilization.isdigit()):
            continue  # a GPU that needs a reset reports [N/A]; the lease probe leaves it out too
        gpus[device] = {"free_memory_mib": int(free), "utilization_percent": int(utilization)}
    available_kib = next(int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
    return {"cpu_count": os.cpu_count(), "available_memory_gib": available_kib / 1024**2, "gpus": gpus}


def prepare(*, robotwin_root: Path, python: Path, tasks, slots, candidate_count: int,
            outer_seed: int = 0, purpose: str = "targeted_regression", required_host: str | None = None,
            accepted_per_task: int = 1, task_config: str = "demo_randomized",
            start_offsets=(0,), segments=None, process_attempt_limit: int = 8):
    """Plan an expert seed scan: the first ``accepted_per_task`` seeds from the
    official start seed that the expert solves, per task, under one scene
    configuration, trying at most ``candidate_count`` seeds per task.

    ``start_offsets`` or explicit ``segments`` split a long task into seed
    ranges scanned in parallel (``robotwin_seed_segments.plan_segments``).
    ``process_attempt_limit`` is how many seeds one oracle process scans
    before re-executing itself: one seed per process re-created the CUDA
    context, renderer and CuRobo warm-up every seed, and ~100 such workers per
    node serialized on the NVIDIA driver's locks (H200 node, 1,026 s per seed)."""
    from ora.orchestrator.robotwin_seed_segments import plan_segments, segment_cases

    host = declared_execution_host(required_host)
    if task_config not in TASK_CONFIGS:
        raise ValueError("task_config must be demo_clean or demo_randomized")
    if isinstance(accepted_per_task, bool) or not isinstance(accepted_per_task, int) or accepted_per_task < 1:
        raise ValueError("accepted_per_task must be positive")
    runtime = {"kind": "robotwin", "transport": "local_rpc",
               "root_env": "ROBOTWIN_ROOT", "python_env": "ROBOTWIN_PYTHON",
               "required_revision": REQUIRED_REVISION, "task_config": task_config,
               "required_host": host}
    # absolute(), not resolve(): a venv's bin/python is a symlink to the base
    # interpreter, and Python finds the venv from the pyvenv.cfg beside the path
    # it was invoked as, so a resolved path drops the venv's site-packages.
    robotwin_root, python = robotwin_root.resolve(), python.expanduser().absolute()
    if git(robotwin_root, "rev-parse", "HEAD") != REQUIRED_REVISION:
        raise ValueError("RoboTwin revision differs from the source-tracked preparation")
    if git(robotwin_root, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("RoboTwin tracked source is dirty")
    if not python.is_file():
        raise ValueError("isolated RoboTwin Python does not exist")
    if not tasks or not set(tasks).issubset(TASKS) or len(set(tasks)) != len(tasks):
        raise ValueError("qualification tasks must be distinct catalog members")
    if isinstance(candidate_count, bool) or not isinstance(candidate_count, int) or candidate_count < 1:
        raise ValueError("candidate_count must be positive")
    if candidate_count < accepted_per_task:
        raise ValueError("candidate_count must be at least accepted_per_task")
    if isinstance(process_attempt_limit, bool) or not isinstance(process_attempt_limit, int) or process_attempt_limit < 1:
        raise ValueError("process_attempt_limit must be a positive integer")
    pairs, lengths = plan_segments(tasks, candidate_count, start_offsets=start_offsets, segments=segments)
    if purpose not in {"targeted_regression", "development_diagnostic"}:
        raise ValueError("purpose must be targeted_regression or development_diagnostic")
    config_path = robotwin_root / f"task_config/{task_config}.yml"
    official = yaml.safe_load(config_path.read_text())
    validate_task_config(task_config, official)
    roster = {path.stem for path in (robotwin_root / "description/task_instruction").glob("*.json")}
    if roster != set(TASKS):
        raise ValueError("pinned upstream task roster differs from the 50-task registry")
    textures = robotwin_root / "assets/background_texture/unseen"
    texture_names = {path.name for path in textures.iterdir() if path.is_file()}
    if not texture_names or texture_names != {f"{i}.png" for i in range(len(texture_names))}:
        raise ValueError("unseen evaluation texture pool is missing or non-contiguous")
    protocol = evaluation_protocol(outer_seed=outer_seed, task_config=task_config)
    resources = capacity()
    bindings = [resolve_execution_resource(slot, runtime_kind="robotwin", required_host=host) for slot in slots]
    if not bindings or len({binding["slot_id"] for binding in bindings}) != len(bindings):
        raise ValueError("distinct qualified simulator slots are required")
    bindings = bindings[:len(pairs)]
    for binding in bindings:
        device = binding["worker_environment"]["CUDA_VISIBLE_DEVICES"]
        if resources["gpus"][device]["free_memory_mib"] < binding["minimum_free_memory_mib"]:
            raise ValueError(f"insufficient GPU memory for {binding['profile_id']}")
    deps = source_dependencies(robotwin_root, tasks, task_config)
    dependency_hash = digest([
        {"input": ("harness/" + str(Path(item["path"]).relative_to(ROOT))
                   if Path(item["path"]).is_relative_to(ROOT)
                   else "robotwin/" + str(Path(item["path"]).relative_to(robotwin_root))),
         "sha256": item["sha256"]} for item in deps
    ])
    cases = segment_cases(pairs, lengths, start=protocol["start_requested_seed"], task_config=task_config,
                          accepted_per_task=accepted_per_task, dependency_hash=dependency_hash)
    # A lane's "tasks" are case ids: the task name for offset 0, else task@<first seed>.
    case_ids = [case["case_id"] for case in cases]
    lanes = [{"resource_binding": binding, "tasks": case_ids[index::len(bindings)]} for index, binding in enumerate(bindings)]
    seconds = 90 + max(len(lane["tasks"]) for lane in lanes) * candidate_count * 540
    minutes = seconds / 60
    result = {
        "schema": "ora.robotwin-evaluation-preparation.v1",
        "source_commit": git(ROOT, "rev-parse", "HEAD"),
        "source_dirty": bool(git(ROOT, "status", "--porcelain")),
        "robotwin_root": str(robotwin_root), "robotwin_python": str(python),
        "robotwin_revision": REQUIRED_REVISION,
        "required_host": host,
        "task_config": task_config, "accepted_per_task": accepted_per_task,
        "protocol": protocol, "official_config": official,
        "official_config_source": file_binding(config_path),
        "unseen_texture_count": len(texture_names),
        "all_tasks": [{"task": task, "runtime": {**runtime, "eval_mode": True, "is_test": True},
                       "seed_selection": dict(protocol), "qualification_status": "not_bound",
                       "formal_launch_authorized": False} for task in TASKS],
        "qualification": {"purpose": "development_diagnostic" if minutes > 15 else purpose,
                          "coverage_policy": "representative-v1", "cases": cases,
                          "reuse": [], "dependency_closure": deps,
                          "producer_dependency_hash": dependency_hash,
                          "dependency_coverage": "declared source/config inputs; selected assets recorded by producer; no blanket cross-installation reuse claim",
                          "checker_hash": file_binding(ROOT / "ora/environment/backends/robotwin/qualification.py")["sha256"],
                          "parallel_execution": {"capacity": resources, "lanes": lanes, "leases": "atomically acquired and rechecked at execution"},
                          "process_attempt_limit": process_attempt_limit,
                          "estimated_wall_minutes": minutes,
                          "estimate_basis": "540s per candidate (historical bread oracle 462–486s plus margin); early stop at the accepted target; longest lane plus 90s setup/finalization",
                          "nondeterminism_evidence": None},
        "claim_boundary": "All 50 scene configurations prepared; only selected oracle cases scheduled. Accepted seeds are bound to this scene configuration and revision. Not 50 qualified tasks, no policy run and no leaderboard-equivalence claim.",
        "policy_binding": None, "formal_launch_authorized": False,
    }
    result["plan_sha256"] = digest(result)
    return result


def seed_manifests(destination: Path, plan, checksum: str, task_config: str, *, folder: str = "seed-manifests"):
    """Seal one accepted-seed manifest per scanned task from the worker's ledger.

    The worker accepts a seed when the reset predicate is false, the expert
    plans and its post-play official predicate holds, and the terminal
    observation carries paired RGB-D; that is upstream's collection rule. The
    post-success hold is measured and recorded, never a reason to reject a
    seed. Every other candidate is recorded with its classification. The
    manifest lists both in requested-seed order and carries per-seed evidence
    (``seeds[<seed>]``: ``reset_predicate_false``, ``oracle_success``,
    ``post_success_physics_hold``, ``terminal_stability_steps`` as measured)
    plus aggregates derived from those records; launch preflight admits a seed
    on its own reset/oracle evidence and treats the hold as evidence.
    A segment case (``task@<first seed>``) is sealed under its case id; a torn
    final ledger line from a running worker is skipped (``seal``). Records that
    say nothing about their seed (``infra_failure``, ``invalid_artifact``, a
    watchdog timeout under a shorter bound than the one in force) are left out,
    so those seeds count as unscanned (``robotwin_seed_lanes.valid_ledger_records``).
    """
    from ora.orchestrator.robotwin_seed_lanes import read_ledger_records, valid_ledger_records

    written = {}
    for case in plan["qualification"]["cases"]:
        task = case["task"]
        case_id = case.get("case_id", task)
        ledger = destination / case_id / task / "seed-ledger.jsonl"
        if not ledger.exists():
            continue
        records = valid_ledger_records(read_ledger_records(ledger))
        records.sort(key=lambda record: record["requested_seed"])
        accepted = [r["requested_seed"] for r in records if r.get("accepted") is True]
        rejected = [{"requested_seed": r["requested_seed"], "classification": r.get("classification"),
                     "error": r.get("error")} for r in records if r.get("accepted") is not True]
        run_manifest_path = destination / case_id / "run-manifest.json"
        run_manifest = json.loads(run_manifest_path.read_text()) if run_manifest_path.exists() else {}
        target = case.get("accepted_target", 1)
        seeds = {}
        for r in records:
            if r.get("accepted") is not True:
                continue
            expert, stability = r.get("expert") or {}, r.get("stability") or {}
            seeds[str(r["requested_seed"])] = {
                "reset_predicate_false": r.get("reset_official_predicate") is False,
                "oracle_success": bool(expert.get("plan_success")) and bool(expert.get("post_play_official_predicate")),
                "post_success_physics_hold": stability.get("passed") is True,
                "terminal_stability_steps": int(stability.get("checks") or 0),
            }
        manifest = {
            "schema": SEED_MANIFEST_SCHEMA, "task": task, "task_config": task_config,
            "robotwin_revision": plan["robotwin_revision"], "source_commit": plan["source_commit"],
            "plan_sha256": checksum,
            "producer_dependency_hash": plan["qualification"].get("producer_dependency_hash"),
            "start_requested_seed": case["start_requested_seed"],
            "exclusive_seed_limit": case["exclusive_seed_limit"],
            "accepted_target": target, "accepted_seeds": accepted,
            "evaluated_seeds": [r["requested_seed"] for r in records],
            "rejected": rejected, "complete": len(accepted) >= target, "seeds": seeds,
            "oracle_success": bool(seeds) and all(s["oracle_success"] for s in seeds.values()),
            "reset_predicate_false": bool(seeds) and all(s["reset_predicate_false"] for s in seeds.values()),
            "post_success_physics_hold": bool(seeds) and all(s["post_success_physics_hold"] for s in seeds.values()),
            "unstable_accepted_seeds": [int(seed) for seed, s in seeds.items() if not s["post_success_physics_hold"]],
            "required_stability_steps": int(run_manifest.get("terminal_stability_physics_steps", 250)),
            "terminal_stability_steps": min((s["terminal_stability_steps"] for s in seeds.values()), default=0),
            "ledger_sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
        }
        manifest["manifest_sha256"] = digest(manifest)
        path = destination / folder / f"{case_id}.json"
        save(path, manifest)
        written[case_id] = path
    return written


def qualify(plan_path: Path, confirm: str):
    plan = json.loads(plan_path.read_text())
    checksum = plan.pop("plan_sha256")
    if (plan.get("schema") != "ora.robotwin-evaluation-preparation.v1"
            or digest(plan) != checksum or confirm != f"QUALIFY-ROBOTWIN:{checksum}"):
        raise ValueError("qualification confirmation or plan hash mismatch")
    declared_execution_host(plan.get("required_host"))
    if git(ROOT, "status", "--porcelain") or git(ROOT, "rev-parse", "HEAD") != plan["source_commit"] or plan["source_dirty"]:
        raise ValueError("qualification requires the exact clean committed preparation source")
    root = Path(plan["robotwin_root"])
    if git(root, "status", "--porcelain", "--untracked-files=no") or git(root, "rev-parse", "HEAD") != plan["robotwin_revision"]:
        raise ValueError("RoboTwin source changed after prepare")
    for item in plan["qualification"]["dependency_closure"]:
        if file_binding(Path(item["path"])) != item:
            raise ValueError(f"qualification dependency changed: {item['path']}")
    current_capacity = capacity()
    destination = WORKSPACE / "qualification" / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{time.time_ns()}"
    save(destination / "prepared-plan.json", {**plan, "plan_sha256": checksum})
    save(destination / "fresh-capacity.json", current_capacity)
    from ora.orchestrator.cli import claim_resource_lease, release_resource_lease
    from ora.orchestrator import robotwin_seed_lanes as lanes_module
    by_task = {case.get("case_id", case["task"]): case for case in plan["qualification"]["cases"]}
    # Plans prepared before these fields existed scanned demo_randomized for one seed.
    task_config = plan.get("task_config", "demo_randomized")

    def lane_worker(index, lane):
        time.sleep(index * lanes_module.LANE_START_STAGGER_S)
        binding = lane["resource_binding"]
        device = binding["worker_environment"]["CUDA_VISIBLE_DEVICES"]
        if current_capacity["gpus"][device]["free_memory_mib"] < binding["minimum_free_memory_mib"]:
            return [{"tasks": lane["tasks"], "state": "blocked_resource", "passed": False}]
        owner = f"robotwin-evaluation-oracle-{binding['slot_id']}-{os.getpid()}-{time.time_ns()}"
        lease = lanes_module.claim_with_probe_retry(
            claim_resource_lease, {"plan_sha256": checksum, "runtime": {"kind": "robotwin", "resource_binding": binding}},
            owner, os.getpid())
        results = []
        try:
            environment = {**os.environ, **binding["worker_environment"], "PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
            for task in lane["tasks"]:  # case ids; equal to the task for offset 0
                case = by_task[task]
                output = destination / task
                output.mkdir()
                command = [plan["robotwin_python"], "-m", "ora.environment.backends.robotwin.qualification",
                           "--robotwin-root", str(root), "--output-root", str(output), "--tasks", case["task"],
                           "--task-config", task_config, "--scene-mode", "evaluation",
                           "--required-revision", plan["robotwin_revision"],
                           "--accepted-per-task", str(case.get("accepted_target", 1)),
                           "--start-requested-seed", str(case["start_requested_seed"]),
                           "--max-requested-seeds", str(case["exclusive_seed_limit"]),
                           "--stability-steps", "250", "--process-attempt-limit",
                           # Plans prepared before the field existed rotated after every seed.
                           str(plan["qualification"].get("process_attempt_limit", 1))]
                save(output / "execution-binding.json", {"source_commit": plan["source_commit"], "command": command,
                     "resource_binding": binding, "thread_environment": {key: environment[key] for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")}})
                with (output / "worker.log").open("xb") as log:
                    # A hung seed is rejected and the worker restarted at the next seed.
                    code, timed_out = lanes_module.run_with_seed_watchdog(
                        command, cwd=root, env=environment, log=log, task_root=output / case["task"],
                        task=case["task"], task_config=task_config,
                        max_restarts=case["exclusive_seed_limit"] - case["start_requested_seed"])
                summary_path = output / "summary.json"
                summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
                results.append({"task": task, "returncode": code, "passed": code == 0 and summary is not None and summary["passed"],
                                "summary": file_binding(summary_path) if summary else None,
                                "oracle_timeouts": timed_out, "evidence_root": str(output)})
        finally:
            release_resource_lease(owner, claim_token=lease["claim_token"])
        return results

    lanes = plan["qualification"]["parallel_execution"]["lanes"]
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(lanes)) as pool:
        futures = {pool.submit(lane_worker, index, lane): lane for index, lane in enumerate(lanes)}
        for future in concurrent.futures.as_completed(futures):
            try:
                results.extend(future.result())
            except Exception as exc:
                results.append({"tasks": futures[future]["tasks"], "passed": False, "state": "infrastructure_error", "error": f"{type(exc).__name__}: {exc}"})
    manifests = seed_manifests(destination, plan, checksum, task_config)
    report = {"source_commit": plan["source_commit"], "plan_sha256": checksum, "results": results,
              "passed": len(results) == len(by_task) and all(item["passed"] for item in results),
              "seed_manifests": {task: str(path) for task, path in manifests.items()},
              "formal_runs_launched": 0}
    save(destination / "qualification-report.json", report)
    print(json.dumps({"report": str(destination / "qualification-report.json"), **report}, indent=2))
    return 0 if report["passed"] else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("configs", "prepare", "qualify", "merge", "seal"))
    parser.add_argument("--robotwin-root", type=Path)
    parser.add_argument("--robotwin-python", type=Path)
    parser.add_argument("--tasks", nargs="+", choices=TASKS)
    parser.add_argument("--slots", nargs="+")
    parser.add_argument("--candidate-count", type=int,
                        help="seeds to try per task, from the official start seed")
    parser.add_argument("--accepted-per-task", type=int, default=1,
                        help="expert-accepted seeds to collect per task (official protocol: 100)")
    parser.add_argument("--task-config", choices=sorted(TASK_CONFIGS), default="demo_randomized")
    parser.add_argument("--outer-seed", type=int, default=0)
    parser.add_argument("--purpose", choices=("targeted_regression", "development_diagnostic"),
                        default="targeted_regression")
    parser.add_argument("--required-host", help="registered execution host; defaults to the host marker")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--confirm")
    from ora.orchestrator import robotwin_seed_segments as segments
    segments.add_arguments(parser)
    args = parser.parse_args(argv)
    if args.command == "configs":
        print(json.dumps({"protocol": evaluation_protocol(outer_seed=args.outer_seed, task_config=args.task_config), "tasks": TASKS, "formal_launch_authorized": False}, indent=2))
        return 0
    if args.command == "qualify":
        if args.plan is None or args.confirm is None:
            parser.error("qualify requires --plan and --confirm")
        return qualify(args.plan, args.confirm)
    if args.command in ("merge", "seal"):
        return (segments.merge_command if args.command == "merge" else segments.seal_command)(args, parser)
    if any(value is None for value in (args.robotwin_root, args.robotwin_python, args.tasks, args.slots, args.candidate_count)):
        parser.error("prepare requires explicit --robotwin-root, --robotwin-python, --tasks, --slots and --candidate-count")
    plan = prepare(robotwin_root=args.robotwin_root, python=args.robotwin_python,
                   tasks=args.tasks, slots=args.slots, candidate_count=args.candidate_count,
                   outer_seed=args.outer_seed, purpose=args.purpose, required_host=args.required_host,
                   accepted_per_task=args.accepted_per_task, task_config=args.task_config,
                   start_offsets=args.start_offsets, segments=segments.parse_segments(args.segments),
                   process_attempt_limit=args.process_attempt_limit)
    path = WORKSPACE / "preparations" / f"{time.time_ns()}" / "plan.json"
    save(path, plan)
    print(json.dumps({"plan": str(path), "tasks_prepared": len(plan["all_tasks"]),
                      "qualification_cases": plan["qualification"]["cases"],
                      "purpose": plan["qualification"]["purpose"],
                      "estimated_wall_minutes": plan["qualification"]["estimated_wall_minutes"],
                      "confirmation": f"QUALIFY-ROBOTWIN:{plan['plan_sha256']}"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
