"""Host launcher and judge of the RoboDojo physics scene probe (CPU-physics lane, qualification only).

Each cell is one task on one qualified RoboDojo slot. The launcher resolves the
task's default runtime (``robodojo_runtime_config.runtime_configuration``, so
the CPU scenes run ``cpu`` physics) on the slot's GPU, claims that slot's
lease, writes the worker configuration the factory would build (execution
capture off, no infrastructure-qualification binding), stages per-cell scratch
and runs ``ora.environment.backends.robodojo.physics_scene_probe``
with the RoboDojo interpreter in its own process group. Cells given together
run in parallel, one thread, lease and process group each; ``--report-only``
rejudges existing cells without a simulator.

Pass criteria per cell (``judge_cell``):

* all scenes: the worker booted, reset and held every requested tick (or the
  official terminal ended the hold), the physics mode is the declared one
  (``isaaclab_sim_device``; on CPU no Fabric and no suppressed readback), and
  no object fell more than ``SUPPORT_DROP_TOLERANCE_M`` below its reset height,
  went below the floor, became non-finite or vanished;
* conveyor: a belt prim with ``PhysxSurfaceVelocityAPI`` exists, at least one
  rigid object rests on it at reset, and every such object stays at belt height
  and advances at least ``CONVEYOR_MIN_SPEED_FRACTION`` of the belt speed per
  native action (0.1 m/s x 0.04 s = 4 mm);
* articulation: at least one articulation is registered, its joint positions
  stay finite and within ``ARTICULATION_HOLD_TOLERANCE`` of reset;
* fluid: at least one fluid object exists, with a positive, finite and
  constant particle count.

Nothing here decides admission or a card: a passing probe is development
evidence for the CPU-lane design (conveyor contact modification, articulation
initialization, fluid creation). It is never policy-visible.

Operator command (one cell per task x slot, all in parallel)::

    ORA_EXECUTION_HOST=<host-id> uv run --locked python -m \\
      ora.orchestrator.robodojo_physics_probe --output <dir> \\
      --cell press_by_number:robodojo-gpu4-d1 --cell pick_from_conveyor_by_image:robodojo-gpu5-d1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import signal
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from ora.environment.benchmarks.robodojo import NATIVE_ACTION_LIMITS
from ora.environment.benchmarks.robodojo_physics import (
    WORKER_REPORT_NAME, WORKER_REPORT_SCHEMA, probe_only_admission,
    CPU_PHYSICS_DEVICE, PHYSICS_DEVICES, physics_device_for, scene_kind_for,
)

from .paths import PROJECT_ROOT

REPORT_SCHEMA = "ora.robodojo-physics-scene-probe.v1"
REPORT_NAME = "robodojo-physics-scene-probe-report.json"
WORKER_MODULE = "ora.environment.backends.robodojo.physics_scene_probe"
WORKER_SOURCE = "ora/environment/backends/robodojo/physics_scene_probe.py"
#: Upstream ``self.step_lim`` of a task not yet admitted to the Runtime
#: (RoboDojo ee67a146, ``task/RoboDojo/tasks/<task>.py``); only this probe's own
#: processes see it. Empty since 2026-10-04: the two fluid tasks it held,
#: ``pour_liquid_into_cup`` (400) and ``pour_by_language`` (800), passed their
#: fluid probe here (``robodojo-physics-scene-probe-20261004T012747Z-pour-r2``)
#: and are admitted in ``NATIVE_ACTION_LIMITS`` with reviewed contracts.
PROBE_ONLY_NATIVE_ACTION_LIMITS: Mapping[str, int] = MappingProxyType({})
#: Native actions held per scene kind: 100 for a conveyor (0.4 m of belt travel), else 50.
HOLD_TICKS: Mapping[str, int] = MappingProxyType({"conveyor": 100})
DEFAULT_HOLD_TICKS = 50
#: One native action is ten 0.004 s PhysX steps.
NATIVE_ACTION_S = 0.04
#: The upstream belts' surface speed, used when the prim's velocity is unreadable.
BELT_SPEED_FALLBACK_M_S = 0.1
CONVEYOR_MIN_SPEED_FRACTION = 0.5
#: Below the belt top by more than this, an object has left the belt (or sunk through it).
BELT_HEIGHT_TOLERANCE_M = 0.03
#: An object whose reset height is this far above the belt top is not resting on it.
ON_BELT_MAX_HEIGHT_M = 0.30
SUPPORT_DROP_TOLERANCE_M = 0.03
FLOOR_Z_M = -0.01
ARTICULATION_HOLD_TOLERANCE = 0.05
#: Layout kinds that are scenery, not objects with a support.
UNSUPPORTED_KINDS = ("room", "ground")
DEFAULT_TIMEOUT_S = 1800.0


class ProbeLaunchError(RuntimeError):
    """A probe cell cannot be launched or reported as requested."""


# --- arguments -------------------------------------------------------------------------------------

def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def parse_cell(text: str) -> tuple[str, str]:
    task, separator, profile = str(text).partition(":")
    if not separator or not task or not profile:
        raise argparse.ArgumentTypeError("a cell is TASK:RESOURCE_PROFILE, e.g. make_toast:robodojo-gpu4-d1")
    if task not in NATIVE_ACTION_LIMITS and task not in PROBE_ONLY_NATIVE_ACTION_LIMITS:
        raise argparse.ArgumentTypeError(f"{task} is neither an admitted RoboDojo task nor a probe-only one")
    return task, profile


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m ora.orchestrator.robodojo_physics_probe",
                                     description="RoboDojo physics scene probe (qualification only).")
    parser.add_argument("--cell", type=parse_cell, action="append", default=[],
                        help="TASK:RESOURCE_PROFILE; repeatable; the cells run in parallel")
    parser.add_argument("--output", type=Path, required=True, help="probe directory; one subdirectory per cell")
    parser.add_argument("--ticks", type=_positive, default=None,
                        help=f"native actions to hold (default {HOLD_TICKS['conveyor']} conveyor, "
                             f"{DEFAULT_HOLD_TICKS} otherwise)")
    parser.add_argument("--sample-every", type=_positive, default=5, help="sample interval in native actions")
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S, help="per-cell worker timeout")
    parser.add_argument("--physics-device", choices=PHYSICS_DEVICES, default=None,
                        help="override every cell's default device (a comparison; recorded in the report)")
    parser.add_argument("--in-place-restart", action="store_true",
                        help="also restart round 1 in place and read rigid-object physics flags (report only)")
    parser.add_argument("--report-only", action="store_true", help="rejudge existing cells without a simulator")
    arguments = parser.parse_args(argv)
    if not arguments.report_only and not arguments.cell:
        parser.error("at least one --cell is required unless --report-only")
    directories = [cell_directory(*cell) for cell in arguments.cell]
    if len(set(directories)) != len(directories):
        parser.error("a task:profile cell is named twice")
    return arguments


def hold_ticks(task: str, override: int | None) -> int:
    return int(override) if override is not None else HOLD_TICKS.get(scene_kind_for(task), DEFAULT_HOLD_TICKS)


def cell_directory(task: str, profile: str) -> str:
    return f"{task}--{profile}"


# --- runtime and worker ----------------------------------------------------------------------------

def _probe_only_limit(task: str) -> int | None:
    return None if task in NATIVE_ACTION_LIMITS else PROBE_ONLY_NATIVE_ACTION_LIMITS[task]


def probe_runtime(task: str, profile: str, *, physics_device: str | None = None) -> dict[str, Any]:
    """The task's default runtime on the slot's GPU, capture off and without a qualification binding."""
    from .resources import resolve_execution_resource
    from .robodojo_runtime_config import execution_host, runtime_configuration

    binding = resolve_execution_resource(profile, runtime_kind="robodojo", required_host=execution_host())
    gpu = str(binding["worker_environment"]["CUDA_VISIBLE_DEVICES"])
    with probe_only_admission(task, _probe_only_limit(task), NATIVE_ACTION_LIMITS):
        runtime = runtime_configuration(gpu, task)
    runtime.pop("infrastructure_qualification", None)
    return {**runtime, "physics_device": physics_device or runtime["physics_device"],
            "resource_profile": profile, "resource_binding": dict(binding),
            "execution_capture": {"enabled": False}}


def worker_command(python: Path, *, config: Path, pass_root: Path, ticks: int, sample_every: int,
                   probe_only_limit: int | None, in_place_restart: bool) -> list[str]:
    command = [str(python), "-m", WORKER_MODULE, "--config", str(config), "--output", str(pass_root),
               "--ticks", str(ticks), "--sample-every", str(sample_every)]
    if probe_only_limit is not None:
        command += ["--probe-only-native-action-limit", str(probe_only_limit)]
    return command + (["--in-place-restart"] if in_place_restart else [])


# --- judgement -------------------------------------------------------------------------------------

def _check(failures: list[str], **detail: Any) -> dict[str, Any]:
    return {"passed": not failures, "failures": failures, **detail}


def _finite(values: Any) -> bool:
    return isinstance(values, list) and bool(values) and all(
        isinstance(item, (int, float)) and math.isfinite(item) for item in values)


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) and "error" not in value else None


def check_worker(worker: Mapping[str, Any]) -> dict[str, Any]:
    failures = []
    if worker.get("error"):
        failures.append(f"worker error: {worker['error']}")
    elif worker.get("completed") is not True:
        failures.append("worker did not complete")
    requested, held = worker.get("ticks_requested"), worker.get("ticks_held")
    if not worker.get("terminal_before_hold_end") and held != requested:
        failures.append(f"held {held} of {requested} native actions")
    if len(worker.get("samples") or []) < 2:
        failures.append("fewer than two samples")
    return _check(failures, ticks_requested=requested, ticks_held=held,
                  terminal_before_hold_end=worker.get("terminal_before_hold_end"))


def check_physics_mode(declared: str, mode: Any) -> dict[str, Any]:
    failures = []
    selected = mode if isinstance(mode, Mapping) else {}
    if selected.get("isaaclab_sim_device") != declared:
        failures.append(f"IsaacLab simulation device {selected.get('isaaclab_sim_device')!r}, declared {declared!r}")
    if declared == CPU_PHYSICS_DEVICE:
        if selected.get("use_fabric") is True:
            failures.append("CPU physics ran with Fabric")
        if selected.get("suppress_readback") is True:
            failures.append("CPU physics suppressed the USD readback")
    return _check(failures, declared=declared, readback=mode)


def check_supported(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """No object below its support: no drop beyond tolerance, nothing below the floor, vanished or non-finite."""
    first = _mapping(samples[0].get("objects")) if samples else None
    if first is None:
        return _check(["no object positions at reset"], objects={})
    failures: list[str] = []
    detail: dict[str, Any] = {}
    for name, entry in sorted(first.items()):
        if not isinstance(entry, Mapping) or entry.get("kind") in UNSUPPORTED_KINDS or "position_m" not in entry:
            continue
        if not _finite(entry["position_m"]):
            failures.append(f"{name} has a non-finite position at reset")
            continue
        z0 = z_min = float(entry["position_m"][2])
        for sample in samples[1:]:
            objects = _mapping(sample.get("objects"))
            current = objects.get(name) if objects is not None else None
            if objects is None:
                failures.append(f"object positions unreadable at tick {sample.get('tick')}")
                break
            if not isinstance(current, Mapping) or "position_m" not in current:
                failures.append(f"{name} vanished by tick {sample.get('tick')}")
                break
            if not _finite(current["position_m"]):
                failures.append(f"{name} became non-finite at tick {sample.get('tick')}")
                break
            z_min = min(z_min, float(current["position_m"][2]))
        if z0 - z_min > SUPPORT_DROP_TOLERANCE_M:
            failures.append(f"{name} fell {z0 - z_min:.3f} m below its reset height ({z0:.3f} -> {z_min:.3f})")
        if z_min < FLOOR_Z_M:
            failures.append(f"{name} went below the floor (z {z_min:.3f})")
        detail[name] = {"kind": entry.get("kind"), "z0_m": z0, "z_min_m": z_min}
    return _check(failures, objects=detail, drop_tolerance_m=SUPPORT_DROP_TOLERANCE_M)


def _on_belt(position: Sequence[float], belt: Mapping[str, Any]) -> bool:
    low, high = belt["world_aabb_min"], belt["world_aabb_max"]
    top = float(high[2])
    return (low[0] <= position[0] <= high[0] and low[1] <= position[1] <= high[1]
            and top - BELT_HEIGHT_TOLERANCE_M <= position[2] <= top + ON_BELT_MAX_HEIGHT_M)


def _belt_speed(belt: Mapping[str, Any]) -> float:
    velocity = belt.get("surface_velocity")
    speed = math.hypot(*velocity[:3]) if _finite(velocity) else BELT_SPEED_FALLBACK_M_S
    return speed * NATIVE_ACTION_S


def check_conveyor(samples: Sequence[Mapping[str, Any]], belts: Any) -> dict[str, Any]:
    """Objects resting on a belt stay at belt height and advance with it."""
    usable = [belt for belt in belts if isinstance(belt, Mapping) and _finite(belt.get("world_aabb_min"))
              and _finite(belt.get("world_aabb_max"))] if isinstance(belts, list) else []
    if not usable:
        return _check(["no surface-velocity belt prim was found"], belts=belts)
    first = _mapping(samples[0].get("objects")) if samples else None
    resting = {name: (entry, belt) for name, entry in sorted((first or {}).items())
               if isinstance(entry, Mapping) and entry.get("kind") == "rigid" and _finite(entry.get("position_m"))
               for belt in usable[:] if _on_belt(entry["position_m"], belt)}
    if not resting:
        return _check(["no rigid object rests on a belt at reset"], belts=usable)
    failures: list[str] = []
    detail: dict[str, Any] = {}
    for name, (entry, belt) in resting.items():
        top = float(belt["world_aabb_max"][2])
        trail = [(sample["native_action"] if isinstance(sample.get("native_action"), int) else sample.get("tick"),
                  (_mapping(sample.get("objects")) or {}).get(name)) for sample in samples]
        trail = [(action, item["position_m"]) for action, item in trail
                 if isinstance(item, Mapping) and _finite(item.get("position_m")) and isinstance(action, int)]
        actions = trail[-1][0] - trail[0][0] if len(trail) > 1 else 0
        if actions <= 0:
            failures.append(f"{name} has no positions across a hold")
            continue
        start, end = trail[0][1], trail[-1][1]
        speed = math.hypot(end[0] - start[0], end[1] - start[1]) / actions
        expected = _belt_speed(belt)
        z_min = min(position[2] for _, position in trail)
        if z_min < top - BELT_HEIGHT_TOLERANCE_M:
            failures.append(f"{name} left the belt height (z {z_min:.3f}, belt top {top:.3f})")
        if speed < CONVEYOR_MIN_SPEED_FRACTION * expected:
            failures.append(f"{name} advanced {speed * 1000:.2f} mm per native action, "
                            f"belt {expected * 1000:.2f} mm")
        detail[name] = {"belt": belt.get("path"), "belt_top_m": top, "z_min_m": z_min, "native_actions": actions,
                        "speed_m_per_action": speed, "expected_m_per_action": expected}
    return _check(failures, objects=detail, min_speed_fraction=CONVEYOR_MIN_SPEED_FRACTION)


def check_articulation(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    first = _mapping(samples[0].get("articulations")) if samples else None
    if not first:
        return _check(["no articulation registered at reset"], articulations={})
    failures: list[str] = []
    detail: dict[str, Any] = {}
    for name, entry in sorted(first.items()):
        q0 = entry.get("joint_positions") if isinstance(entry, Mapping) else None
        if not _finite(q0):
            failures.append(f"{name} joint positions are not finite at reset")
            continue
        drift = 0.0
        for sample in samples[1:]:
            current = (_mapping(sample.get("articulations")) or {}).get(name)
            q = current.get("joint_positions") if isinstance(current, Mapping) else None
            if not isinstance(q, list) or len(q) != len(q0):
                failures.append(f"{name} joints unreadable at tick {sample.get('tick')}")
                break
            if not _finite(q):
                failures.append(f"{name} joints became non-finite at tick {sample.get('tick')}")
                break
            drift = max(drift, max(abs(a - b) for a, b in zip(q, q0)))
        if drift > ARTICULATION_HOLD_TOLERANCE:
            failures.append(f"{name} joints drifted {drift:.3f} from reset while held")
        detail[name] = {"dof_names": entry.get("dof_names"), "max_drift": drift}
    return _check(failures, articulations=detail, hold_tolerance=ARTICULATION_HOLD_TOLERANCE)


def check_fluid(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    first = _mapping(samples[0].get("fluids")) if samples else None
    if not first:
        return _check(["no fluid object at reset"], fluids={})
    failures: list[str] = []
    detail: dict[str, Any] = {}
    for name, entry in sorted(first.items()):
        count0 = entry.get("count") if isinstance(entry, Mapping) else None
        counts = []
        for sample in samples:
            current = (_mapping(sample.get("fluids")) or {}).get(name)
            count = current.get("count") if isinstance(current, Mapping) else None
            if type(count) is not int or count <= 0:
                failures.append(f"{name} has no particles at tick {sample.get('tick')}")
                break
            if current.get("finite") is not True:
                failures.append(f"{name} particles are not finite at tick {sample.get('tick')}")
                break
            if count != count0:
                failures.append(f"{name} particle count changed {count0} -> {count} by tick {sample.get('tick')}")
                break
            counts.append(count)
        detail[name] = {"counts": counts, "last": (_mapping(samples[-1].get("fluids")) or {}).get(name)}
    return _check(failures, fluids=detail)


def judge_cell(task: str, worker: Any, *, declared_device: str) -> dict[str, Any]:
    """Pass/fail checks of one cell for its scene kind; see the module docstring."""
    kind = scene_kind_for(task)
    base = {"task": task, "scene_kind": kind, "physics_device": declared_device}
    if not isinstance(worker, Mapping) or worker.get("schema") != WORKER_REPORT_SCHEMA:
        checks = {"worker_completed": _check(["no worker report"])}
    else:
        samples = worker.get("samples") or []
        checks = {"worker_completed": check_worker(worker),
                  "physics_mode": check_physics_mode(declared_device, worker.get("physics_mode")),
                  "supported": check_supported(samples)}
        if kind == "conveyor":
            checks["conveyor"] = check_conveyor(samples, worker.get("belts"))
        elif kind == "articulation":
            checks["articulation"] = check_articulation(samples)
        elif kind == "fluid":
            checks["fluid"] = check_fluid(samples)
    failures = [f"{name}: {reason}" for name, check in checks.items() for reason in check["failures"]]
    return {**base, "checks": checks, "passed": not failures, "failures": failures}


PASS_CRITERIA = (
    "the worker booted, reset and held every requested native action, or the official terminal ended the hold",
    "the IsaacLab simulation device is the declared one; on CPU without Fabric or suppressed USD readback",
    f"no object fell more than {SUPPORT_DROP_TOLERANCE_M} m below its reset height, below the floor, "
    "became non-finite or vanished",
    f"conveyor: objects resting on a surface-velocity belt stay within {BELT_HEIGHT_TOLERANCE_M} m of the belt "
    f"top and advance at least {CONVEYOR_MIN_SPEED_FRACTION} of the belt speed per native action",
    f"articulation: joints stay finite and within {ARTICULATION_HOLD_TOLERANCE} of reset",
    "fluid: every fluid object has a positive, finite and constant particle count",
)


def _load(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).is_file() else None
    except (OSError, ValueError):
        return None


def build_report(output: Path, *, identity: Mapping[str, Any]) -> dict[str, Any]:
    cells = []
    for root in sorted(path for path in Path(output).iterdir() if path.is_dir() and "--" in path.name):
        launch = _load(root / "launch.json") or {}
        task, _, profile = root.name.partition("--")
        task = launch.get("task", task)
        device = launch.get("physics_device") or physics_device_for(task)
        worker = _load(root / WORKER_REPORT_NAME)
        judged = judge_cell(task, worker, declared_device=device)
        cells.append({**judged, "resource_profile": launch.get("resource_profile", profile),
                      "physics_override": launch.get("physics_override", False),
                      "returncode": launch.get("returncode"), "seconds": launch.get("seconds"),
                      "worker_report": str(root / WORKER_REPORT_NAME),
                      "physics_mode": (worker or {}).get("physics_mode") if isinstance(worker, Mapping) else None,
                      "in_place_restart": (worker or {}).get("in_place_restart") if isinstance(worker, Mapping) else None})
    return {"schema": REPORT_SCHEMA, "harness": dict(identity), "cells": cells,
            "passed": bool(cells) and all(cell["passed"] for cell in cells),
            "pass_criteria": list(PASS_CRITERIA),
            "claim_boundary": ("development qualification of scene physics only: no card, gate, admission, "
                               "motion or behavioural claim; never policy-visible")}


# --- launcher --------------------------------------------------------------------------------------

def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def harness_identity() -> dict[str, Any]:
    status = _git("status", "--porcelain", "--untracked-files=no")
    worker = PROJECT_ROOT / WORKER_SOURCE
    return {"commit": _git("rev-parse", "HEAD"), "tracked_worktree_dirty": None if status is None else bool(status),
            "worker_sha256": hashlib.sha256(worker.read_bytes()).hexdigest() if worker.is_file() else None}


def launch_identity() -> dict[str, Any]:
    from ora.orchestrator.resources import ExecutionResourceError, declared_execution_host
    try:
        declared_execution_host()
    except ExecutionResourceError as exc:
        raise ProbeLaunchError(f"a registered RoboDojo host marker is required: {exc}") from exc
    identity = harness_identity()
    if identity["commit"] is None or identity["tracked_worktree_dirty"] is not False:
        raise ProbeLaunchError("the probe launches only from a clean committed Git checkout")
    return identity


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _worker_config(runtime: Mapping[str, Any], task: str, cell_root: Path) -> tuple[Any, Path]:
    from ora.environment.factory import build_robodojo_worker_session
    from ora.environment.rpc import RPC_SCHEMA

    session = build_robodojo_worker_session(
        runtime=runtime, output_root=cell_root / "environment", task=task, seed=0,
        max_sim_steps=int(runtime["native_action_limit"]) * 10, action_codec="motion_action_v3")
    config = {**session.worker_config, "schema": RPC_SCHEMA, "session_id": f"env-{uuid.uuid4().hex}",
              "shared_root": str(cell_root)}
    _write_json(cell_root / "worker-config.json", config)
    return session, cell_root / "worker-config.json"


def launch_cell(task: str, profile: str, cell_root: Path, *, ticks: int | None, sample_every: int,
                timeout_s: float, identity: Mapping[str, Any], physics_device: str | None,
                in_place_restart: bool) -> int:
    """Claim the slot lease, run one probe worker, release; returns its exit code."""
    from . import host_storage
    from .robodojo_motion_replay import _leased_spawn

    runtime = probe_runtime(task, profile, physics_device=physics_device)
    cell_root.mkdir(parents=True, exist_ok=False)
    held = hold_ticks(task, ticks)
    limit = _probe_only_limit(task)
    session, config = _worker_config(runtime, task, cell_root)
    probe_id = f"robodojo-physics-probe-{cell_directory(task, profile)}-{uuid.uuid4().hex[:8]}"
    staged = host_storage.stage_launch({"runtime": runtime}, probe_id)
    command = worker_command(session.python_executable, config=config, pass_root=cell_root, ticks=held,
                             sample_every=sample_every, probe_only_limit=limit, in_place_restart=in_place_restart)
    launch = {"run_id": probe_id, "task": task, "resource_profile": profile, "gpu": runtime["cuda_visible_devices"],
              "physics_device": runtime["physics_device"], "physics_override": physics_device is not None,
              "ticks": held, "probe_only_native_action_limit": limit, "identity": dict(identity),
              "command": command, **staged}
    _write_json(cell_root / "launch.json", launch)
    lease_plan = {"schema": "ora.robodojo-infrastructure-plan.v1", "runtime": runtime,
                  "harness_commit": identity["commit"], "episodes": 1, "task_solving": False,
                  "diagnostic": f"robodojo physics scene probe of {task}"}
    lease_plan["plan_sha256"] = hashlib.sha256(json.dumps(lease_plan, sort_keys=True).encode()).hexdigest()

    def started(pid: int) -> None:
        _write_json(cell_root / "launch.json", {**launch, "worker_pid": pid, "worker_pgid": pid})

    try:
        code, seconds = _leased_spawn(lease_plan, session, command, cell_root, timeout_s,
                                      host_storage.scratch_environment(staged), started)
    finally:
        host_storage.release_run_scratch(cell_root)
    _write_json(cell_root / "launch.json", {**launch, "returncode": code, "seconds": seconds})
    return code


def _launch_cells(output: Path, arguments: argparse.Namespace, identity: Mapping[str, Any]) -> None:
    from .robodojo_motion_replay import kill_active_groups

    for task, profile in arguments.cell:
        if (output / cell_directory(task, profile)).exists():
            raise ProbeLaunchError(f"cell {cell_directory(task, profile)} already exists under {output}")
    kwargs = {"ticks": arguments.ticks, "sample_every": arguments.sample_every, "timeout_s": arguments.timeout_s,
              "identity": identity, "physics_device": arguments.physics_device,
              "in_place_restart": arguments.in_place_restart}
    # A pool thread blocked in wait() never sees SIGINT; the main thread's handler kills the groups.
    previous = {sig: signal.signal(sig, kill_active_groups) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        with ThreadPoolExecutor(max_workers=len(arguments.cell)) as pool:
            list(pool.map(lambda cell: launch_cell(cell[0], cell[1], output / cell_directory(*cell), **kwargs),
                          arguments.cell))
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    output = arguments.output.resolve()
    if not arguments.report_only:
        identity = launch_identity()
        output.mkdir(parents=True, exist_ok=True)
        started = time.time()
        _launch_cells(output, arguments, identity)
        print(json.dumps({"cells": len(arguments.cell), "seconds": round(time.time() - started, 1)}), flush=True)
    report = build_report(output, identity=harness_identity())
    _write_json(output / REPORT_NAME, report)
    print(json.dumps({"report": str(output / REPORT_NAME), "passed": report["passed"],
                      "cells": {cell["task"] + "@" + str(cell["resource_profile"]): cell["passed"]
                                for cell in report["cells"]}}))
    return 0 if report["passed"] else 1


__all__ = [
    "ARTICULATION_HOLD_TOLERANCE", "BELT_HEIGHT_TOLERANCE_M", "CONVEYOR_MIN_SPEED_FRACTION", "DEFAULT_HOLD_TICKS",
    "HOLD_TICKS", "PROBE_ONLY_NATIVE_ACTION_LIMITS", "REPORT_NAME", "REPORT_SCHEMA", "SUPPORT_DROP_TOLERANCE_M",
    "WORKER_MODULE", "WORKER_SOURCE", "build_report", "cell_directory", "check_articulation", "check_conveyor",
    "check_fluid", "check_physics_mode", "check_supported", "check_worker", "hold_ticks", "judge_cell",
    "launch_cell", "main", "parse_arguments", "parse_cell", "probe_runtime", "worker_command",
]


if __name__ == "__main__":
    sys.exit(main())
