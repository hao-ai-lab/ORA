"""Host launcher and report for RoboDojo control replay (qualification layer L0 only).

The launcher reads a sealed run's RunPlan, stages per-pass scratch
(``host_storage.stage_launch``), claims one qualified RoboDojo slot lease,
writes the worker configuration the factory would build (execution capture
off, no infrastructure-qualification binding, the plan's ``max_sim_steps``)
and runs ``ora.environment.backends.robodojo.control_replay`` with
the run's RoboDojo interpreter (``runtime.python_env``, normally
``ROBODOJO_PYTHON``) in its own process group. It refuses to launch from a
dirty checkout, or when this checkout's RoboDojo worker kernel differs from the
run's in a way ``robodojo_replay_kernel.kernel_admission`` does not admit.
Everything it writes stays under ``--output``, outside the source run.

Modes: ``replay`` (one pass; every physics step compared with the original),
``metrics-after-replay`` (also the metrics of the original and the replayed
journals, which must be equal) and ``determinism`` (two passes, in parallel on
two slots with ``--second-resource-profile``; joints must agree exactly,
private object poses are reported against a 1 mm bound). ``--report-only``
rebuilds the report from existing passes without a simulator. The operator
command is in ``docs/reference/robodojo-motion-qualification.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from .paths import PROJECT_ROOT
from .robodojo_motion_journal import ARMS, JournalError, journal_lines, native_root_of
from .robodojo_motion_measures import METRICS_VERSION
from .robodojo_motion_metrics import build_report as build_metrics_report, load_object_poses
from .robodojo_replay_kernel import ADMITTED_KERNELS, admission_note, kernel_admission

REPORT_SCHEMA = "ora.robodojo-motion-replay-report.v1"
REPORT_NAME = "robodojo-motion-replay-report.json"
WORKER_MODULE = "ora.environment.backends.robodojo.control_replay"
WORKER_SOURCE = "ora/environment/backends/robodojo/control_replay.py"
MODES = ("replay", "metrics-after-replay", "determinism")
PASSES = {"replay": ("pass-a",), "metrics-after-replay": ("pass-a",), "determinism": ("pass-a", "pass-b")}
#: Acceptance: the two replays' private object poses agree within 1 mm (to confirm).
OBJECT_DETERMINISM_BOUND_M = 0.001
#: The largest recorded run replays in about ten minutes (0.28 s per native action after a 90 s start).
DEFAULT_TIMEOUT_S = 7200.0
DIVERGENCE_FIELDS = ("joint_rad", "finger_m", "gripper", "link6_m", "applied_target", "applied_keys_differ")


class ReplayLaunchError(RuntimeError):
    """A replay cannot be launched or reported as requested."""


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m ora.orchestrator.robodojo_motion_replay",
                                     description="RoboDojo control replay of a sealed run (qualification only).")
    parser.add_argument("--run", type=Path, required=True, help="sealed run directory (read-only)")
    parser.add_argument("--output", type=Path, required=True, help="new directory outside the run")
    parser.add_argument("--mode", choices=MODES, default="metrics-after-replay")
    parser.add_argument("--resource-profile", help="qualified RoboDojo slot, e.g. robodojo-gpu4-d1")
    parser.add_argument("--second-resource-profile", help="second slot for a parallel determinism pass")
    parser.add_argument("--upto", type=int, help="replay only the first N recorded actions")
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S, help="per-pass worker timeout")
    parser.add_argument("--report-only", action="store_true", help="rebuild the report from existing passes")
    arguments = parser.parse_args(argv)
    if arguments.upto is not None and arguments.upto < 1:
        parser.error("--upto must be at least 1")
    if arguments.second_resource_profile and arguments.mode != "determinism":
        parser.error("--second-resource-profile applies only to --mode determinism")
    if not arguments.report_only and not arguments.resource_profile:
        parser.error("--resource-profile is required unless --report-only")
    return arguments


def _run_root(path: Path) -> Path:
    native = native_root_of(path)
    if native.name != "native":
        raise ReplayLaunchError("--run must be a run directory containing native/")
    return native.parent


def validate_output(run_root: Path, output: Path, *, report_only: bool) -> Path:
    run_root, output = Path(run_root).resolve(), Path(output).resolve()
    if output.is_relative_to(run_root):
        raise ReplayLaunchError(f"replay output {output} is inside the source run {run_root}")
    if not report_only and output.exists() and any(output.iterdir()):
        raise ReplayLaunchError(f"replay output {output} is not empty")
    return output


def worker_command(python: Path, *, config: Path, native_root: Path, pass_root: Path,
                   upto: int | None) -> list[str]:
    command = [str(python), "-m", WORKER_MODULE, "--config", str(config), "--native", str(native_root),
               "--output", str(pass_root)]
    return command + (["--upto", str(upto)] if upto is not None else [])


def replay_runtime(runtime: Mapping[str, Any], binding: Mapping[str, Any]) -> dict[str, Any]:
    """The run's runtime moved to the leased slot, capture off, no qualification binding."""
    device = str(binding["worker_environment"]["CUDA_VISIBLE_DEVICES"])
    kept = {key: deepcopy(value) for key, value in runtime.items() if key != "infrastructure_qualification"}
    return {**kept, "cuda_visible_devices": device, "renderer_gpu": int(device),
            "resource_profile": binding["profile_id"], "resource_binding": deepcopy(dict(binding)),
            "execution_capture": {**dict(runtime.get("execution_capture") or {}), "enabled": False}}


def _load_optional(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _largest(pairs: Any) -> float:
    return max((abs(float(a) - float(b)) for a, b in pairs), default=0.0)


def _applied_divergence(first: Mapping[str, Any], second: Mapping[str, Any]) -> tuple[float, float]:
    """Largest applied-target difference over shared keys, and 1.0 when the key sets differ."""
    ours, theirs = first.get("applied_position_targets", {}), second.get("applied_position_targets", {})
    shared = sorted(set(ours) & set(theirs))
    value = _largest((a, b) for key in shared for a, b in zip(ours[key], theirs[key], strict=True))
    return value, float(set(ours) != set(theirs))


def _jaw_source(state: Mapping[str, Any]) -> Any:
    return (state.get("gripper_feedback") or {}).get("measurement_source")


def line_divergence(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, float]:
    """Arm joints, finger joints, normalized jaw, link6 position and every applied target.

    The normalized jaw is compared only between lines that derive it from the
    same measurement source (the two-finger mean in current runs); finger joints
    are compared either way.
    """
    ours, theirs = first["measured"], second["measured"]
    return {
        "joint_rad": _largest((a, b) for arm in ARMS
                              for a, b in zip(ours[arm]["arm_joint_qpos"], theirs[arm]["arm_joint_qpos"], strict=True)),
        "finger_m": _largest((a, b) for arm in ARMS for a, b in zip(
            ours[arm].get("gripper_finger_qpos", []), theirs[arm].get("gripper_finger_qpos", []), strict=True)),
        "gripper": _largest((ours[arm]["gripper"], theirs[arm]["gripper"]) for arm in ARMS
                            if _jaw_source(ours[arm]) == _jaw_source(theirs[arm])),
        "link6_m": max(math.dist(ours[arm]["pose_xyz_xyzw"][:3], theirs[arm]["pose_xyz_xyzw"][:3]) for arm in ARMS),
        **dict(zip(("applied_target", "applied_keys_differ"), _applied_divergence(first, second))),
    }


def divergence_cause(control: Mapping[str, Any], line: int, ours: Mapping[str, Any],
                     theirs: Mapping[str, Any]) -> str:
    """Where to look first for one diverging physics-step line."""
    first, second = ours.get("applied_position_targets", {}), theirs.get("applied_position_targets", {})
    keys = sorted(key for key in set(first) | set(second) if first.get(key) != second.get(key))
    if not keys:
        return "measured_state: equal applied targets, the plant state differs"
    if any(key.startswith("support_") for key in keys):
        return ("support_arm_target: the support-arm queue differs; policy observations, which the replay "
                "does not repeat, may have consumed it (check first)")
    if all(key not in control for key in keys):
        return ("held_target_at_tick_start: a held dimension differs; an executor rewrite of prev_control "
                "outside a tick (motion_stall hold without budget) was not reproduced" if line % 10 == 0
                else "held_target_within_tick: a held dimension changed inside a tick")
    return "commanded_target: an interpolated target differs"


def _first_cause(reference_root: Path, sequence: int, line: int, ours: Mapping[str, Any],
                 theirs: Mapping[str, Any]) -> str:
    trace = _load_optional(Path(reference_root) / "controller" / f"action-{sequence:04d}.json") or {}
    samples = trace.get("samples", [])
    control = samples[line // 10].get("control", {}) if line // 10 < len(samples) else {}
    return divergence_cause(control, line, ours, theirs)


def physics_step_divergence(reference_root: Path, candidate_root: Path, *, upto: int | None = None) -> dict[str, Any]:
    """Compare every physics-step line the candidate journaled with the reference line."""
    actions = lines = mismatched = 0
    maxima = {name: 0.0 for name in DIVERGENCE_FIELDS}
    first = None
    for path in sorted((Path(candidate_root) / "controller").glob("action-*.jsonl")):
        sequence = int(path.stem.split("-")[1])
        if upto is not None and sequence >= upto:
            continue
        reference, candidate = journal_lines(Path(reference_root) / "controller" / path.name), journal_lines(path)
        actions, mismatched = actions + 1, mismatched + int(len(reference) != len(candidate))
        for index, (ours, theirs) in enumerate(zip(reference, candidate)):
            value = line_divergence(ours, theirs)
            lines += 1
            maxima = {name: max(maxima[name], value[name]) for name in DIVERGENCE_FIELDS}
            if first is None and any(value[name] > 0.0 for name in DIVERGENCE_FIELDS):
                first = {"action_sequence": sequence, "line": index, **value,
                         "cause_hint": _first_cause(reference_root, sequence, index, ours, theirs)}
    return {"actions_compared": actions, "lines_compared": lines, "line_count_mismatches": mismatched,
            "max_divergence": maxima if lines else None,
            "max_joint_divergence_rad": maxima["joint_rad"] if lines else None, "first_divergence": first}


def _object_rows(path: Path) -> dict[tuple[int, int], Mapping[str, Any]]:
    try:
        rows = journal_lines(path) if Path(path).is_file() else []
    except JournalError:
        rows = []
    return {(int(row["action_sequence"]), int(row["tick_index"])): row["objects"]
            for row in rows if isinstance(row.get("objects"), Mapping)}


def object_pose_divergence(first_path: Path, second_path: Path) -> dict[str, Any]:
    """Tick-by-tick private object-position spread between two replay passes, against 1 mm."""
    first, second = _object_rows(first_path), _object_rows(second_path)
    if not first and not second:
        return {"status": "unavailable", "rows_compared": 0, "rows_unmatched": 0,
                "max_position_divergence_m": None, "first_divergence": None, "bound_m": OBJECT_DETERMINISM_BOUND_M}
    compared = unmatched = 0
    maximum, earliest = 0.0, None
    for key in sorted(set(first) | set(second)):
        ours, theirs = first.get(key), second.get(key)
        if ours is None or theirs is None or set(ours) != set(theirs):
            unmatched += 1
            continue
        compared += 1
        value = max((math.dist(ours[name]["position_m"], theirs[name]["position_m"]) for name in ours), default=0.0)
        maximum = max(maximum, value)
        if value > 0.0 and earliest is None:
            earliest = {"action_sequence": key[0], "tick_index": key[1], "position_m": value}
    status = ("unmatched" if unmatched else "within_bound" if maximum <= OBJECT_DETERMINISM_BOUND_M
              else "exceeds_bound")
    return {"status": status, "rows_compared": compared, "rows_unmatched": unmatched,
            "max_position_divergence_m": maximum, "first_divergence": earliest, "bound_m": OBJECT_DETERMINISM_BOUND_M}


def metric_differences(first: Any, second: Any, prefix: str = "") -> list[str]:
    """Dotted paths whose metric values differ exactly."""
    if isinstance(first, Mapping) and isinstance(second, Mapping):
        return [path for key in sorted(set(first) | set(second))
                for path in metric_differences(first.get(key), second.get(key), f"{prefix}{key}.")]
    return [] if first == second else [prefix.rstrip(".")]


def _pass_entry(native: Path, pass_root: Path, upto: int | None) -> dict[str, Any]:
    worker = _load_optional(pass_root / "replay-worker.json")
    environment = pass_root / "environment"
    entry: dict[str, Any] = {"worker": worker if worker is not None else {"status": "missing"},
                             "launch": _load_optional(pass_root / "launch.json"), "divergence_vs_original": None,
                             "journal_error": None}
    if (environment / "controller").is_dir():
        try:
            entry["divergence_vs_original"] = physics_step_divergence(native, environment, upto=upto)
        except JournalError as exc:
            entry["journal_error"] = str(exc)
    return entry


def _metrics_of(run_root: Path, upto: int | None, **kwargs: Any) -> dict[str, Any] | None:
    return build_metrics_report([run_root], upto=upto, **kwargs)["runs"][0]["metrics"]


def _metrics_comparison(run_root: Path, output: Path, passes: Sequence[str], upto: int | None) -> dict[str, Any]:
    original = _metrics_of(run_root, upto)
    replayed, differences, with_objects = {}, {}, {}
    for name in passes:
        environment = output / name / "environment"
        readable = (environment / "controller").is_dir()
        metrics = _metrics_of(run_root, upto, journal_root=environment) if readable else None
        if metrics is None:
            differences[name] = ["replay journal missing or unreadable"]
            continue
        replayed[name], differences[name] = metrics, metric_differences(original, metrics)
        poses = output / name / "private" / "object-poses.jsonl"
        if poses.is_file():
            with_objects[name] = (_metrics_of(run_root, upto, journal_root=environment,
                                              object_poses=load_object_poses(poses)) or {}).get("held_then_lost")
    return {"original": original, "replay": replayed, "differences": differences,
            "held_then_lost_with_private_objects": with_objects}


def _divergence_failures(name: str, divergence: Mapping[str, Any]) -> list[str]:
    maxima = divergence.get("max_divergence") or {}
    failures = [f"{name} {field} diverges from the original" for field in DIVERGENCE_FIELDS
                if maxima.get(field, 0.0) > 0.0]
    return failures + ([f"{name} journal line counts differ from the original"]
                       if divergence["line_count_mismatches"] else [])


def _pass_failures(name: str, entry: Mapping[str, Any], identity: Mapping[str, Any]) -> list[str]:
    worker, divergence = entry["worker"], entry["divergence_vs_original"]
    if worker.get("status") == "missing":
        return [f"{name} has no worker report"]
    failures = [f"{name} worker failed: {worker['error']}"] if worker.get("error") else []
    if not worker.get("error") and not worker.get("completed"):
        failures.append(f"{name} did not replay every recorded tick")
    if (worker.get("replay") or {}).get("applied_target_mismatches"):
        failures.append(f"{name} applied targets differed from the resolved command")
    launch = (entry.get("launch") or {}).get("identity") or {}
    if launch and identity.get("commit") is not None and launch.get("commit") != identity.get("commit"):
        failures.append(f"{name} was launched at another commit ({launch.get('commit')})")
    if launch and (launch.get("kernel_admission") or {}).get("status") not in ADMITTED_KERNELS:
        failures.append(f"{name} was launched with an unadmitted worker kernel")
    if entry.get("journal_error"):
        return [*failures, f"{name} replay journal is malformed: {entry['journal_error']}"]
    if divergence is None:
        return [*failures, f"{name} has no replay journal"]
    return failures + _divergence_failures(name, divergence)


def _failures(report: Mapping[str, Any]) -> list[str]:
    identity = report["harness"]
    failures = [text for name, entry in report["passes"].items() for text in _pass_failures(name, entry, identity)]
    if (identity.get("kernel_admission") or {}).get("status") == "control_path_changed":
        failures.append("this checkout's RoboDojo control-replay path differs from the run's kernel")
    for name, paths in report.get("metrics", {}).get("differences", {}).items():
        if paths:
            failures.append(f"{name} metrics differ from the original: {', '.join(paths[:5])}")
    determinism = report.get("determinism")
    if determinism is not None:
        failures.extend(_divergence_failures("pass-b against pass-a", determinism["joints"]))
    return failures


def _file_sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _run_identity(run_root: Path) -> dict[str, Any]:
    plan = _load_optional(run_root / "plan.json") or {}
    manifest = run_root / "native" / "source-manifest.json"
    return {"run_id": run_root.name, "run_root": str(run_root),
            "task": (plan.get("runtime") or {}).get("task_variant"),
            "harness_commit": (plan.get("source") or {}).get("commit"),
            "source_manifest_sha256": _file_sha256(manifest),
            "source_revisions": (_load_optional(manifest) or {}).get("source_revisions")}


def build_report(run_root: Path, output: Path, *, mode: str, upto: int | None,
                 identity: Mapping[str, Any]) -> dict[str, Any]:
    passes = PASSES[mode]
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA, "replay_contract": "control", "mode": mode, "run": _run_identity(run_root),
        "harness": dict(identity), "metric_definition_version": METRICS_VERSION, "upto": upto,
        "passes": {name: _pass_entry(run_root / "native", output / name, upto) for name in passes},
    }
    if mode != "replay":
        report["metrics"] = _metrics_comparison(run_root, output, passes, upto)
    if mode == "determinism":
        first, second = output / passes[0], output / passes[1]
        report["determinism"] = {
            "joints": physics_step_divergence(first / "environment", second / "environment", upto=upto),
            "objects": object_pose_divergence(first / "private" / "object-poses.jsonl",
                                              second / "private" / "object-poses.jsonl")}
    failures = _failures(report)
    return {**report, "kernel_notice": admission_note(identity.get("kernel_admission")),
            "rebuilt_outside_git": identity.get("commit") is None, "failures": failures, "passed": not failures,
            "pass_criteria": ["every pass replays every selected recorded tick without a worker error or an "
                              "applied-target mismatch",
                              "arm joints, finger joints, jaw, link6 and every applied target equal the original "
                              "on every physics step",
                              "metrics of the replayed journals equal the original's (metrics modes)",
                              "determinism: the two passes agree on every physics step; private object poses are "
                              "reported against 1 mm and do not decide the result",
                              "the worker kernel equals the run's or its drift is admitted (reviewed "
                              "executor-only modules, add-only changes elsewhere), and every pass ran at this "
                              "commit"]}


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _kernel_at(commit: str | None) -> str | None:
    from ora.environment.benchmarks.robodojo import RoboDojoBenchmarkError

    from .robodojo_kernel import worker_kernel_sha256

    try:
        return worker_kernel_sha256(commit, project_root=PROJECT_ROOT) if commit else None
    except RoboDojoBenchmarkError:
        return None


def harness_identity(run_commit: str | None = None) -> dict[str, Any]:
    """This checkout, its RoboDojo worker kernel and its admission against the run's kernel.

    Outside a Git checkout (a copied tree rebuilding a report) every Git-derived field is ``None``.
    """
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    current, recorded = _kernel_at(commit), _kernel_at(run_commit)
    return {"commit": commit, "tracked_worktree_dirty": None if status is None else bool(status),
            "worker_kernel_sha256": current, "run_worker_kernel_sha256": recorded,
            "worker_kernel_matches_run": None if recorded is None or current is None else recorded == current,
            "kernel_admission": kernel_admission(run_commit, commit) if commit else None,
            "replay_worker_sha256": _file_sha256(PROJECT_ROOT / WORKER_SOURCE)}


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _worker_config(runtime: Mapping[str, Any], plan: Mapping[str, Any], pass_root: Path,
                   native: Path) -> tuple[Any, Path]:
    from ora.environment.benchmarks.robodojo import NATIVE_ACTION_LIMITS
    from ora.environment.factory import build_robodojo_worker_session
    from ora.environment.rpc import RPC_SCHEMA

    task = str(runtime["task_variant"])
    codec = (_load_optional(native / "source-manifest.json") or {}).get("action_codec") or "motion_action_v3"
    budget = (plan.get("budgets") or {}).get("max_sim_steps") or NATIVE_ACTION_LIMITS[task] * 10
    session = build_robodojo_worker_session(runtime=runtime, output_root=pass_root / "environment", task=task,
                                            seed=0, max_sim_steps=int(budget), action_codec=codec)
    config = {**session.worker_config, "schema": RPC_SCHEMA, "session_id": f"env-{uuid.uuid4().hex}",
              "shared_root": str(pass_root)}
    _write_json(pass_root / "worker-config.json", config)
    return session, pass_root / "worker-config.json"


#: Worker process groups currently running, killed by the parallel launcher's signal handler.
_ACTIVE_GROUPS: set[int] = set()
_GROUPS_LOCK = threading.Lock()


def _spawn(session: Any, command: Sequence[str], pass_root: Path, timeout_s: float,
           scratch: Mapping[str, str], on_start: Any = None) -> int:
    """Run the worker in its own process group; a timeout or an interrupt kills the whole group.

    The group is registered (and ``on_start(pid)`` records it) before the first wait, so a signal
    handler on the main thread can kill it while a pool thread is blocked in ``wait``.
    """
    from ora.environment.rpc import _worker_environment

    environment = _worker_environment(PROJECT_ROOT, {**session.worker_environment, **scratch})
    with (pass_root / "worker.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(list(command), cwd=PROJECT_ROOT, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        with _GROUPS_LOCK:
            _ACTIVE_GROUPS.add(process.pid)
        try:
            if on_start is not None:
                on_start(process.pid)
            return process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _kill_group(process)
            return -int(signal.SIGKILL)
        except BaseException:
            _kill_group(process)
            raise
        finally:
            with _GROUPS_LOCK:
                _ACTIVE_GROUPS.discard(process.pid)


def kill_active_groups(signum: int, frame: Any = None) -> None:
    """Signal handler for parallel passes: kill every registered worker group, then stop."""
    with _GROUPS_LOCK:
        groups = sorted(_ACTIVE_GROUPS)
    for group in groups:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if signum == signal.SIGINT:
        raise KeyboardInterrupt
    raise SystemExit(128 + signum)


def _kill_group(process: subprocess.Popen) -> None:
    """Kit leaves child processes; the worker's session id is its own process group."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def _leased_spawn(lease_plan: Mapping[str, Any], session: Any, command: Sequence[str], pass_root: Path,
                  timeout_s: float, scratch: Mapping[str, str], on_start: Any = None) -> tuple[int, float]:
    from .cli import assign_resource_lease, claim_resource_lease, release_resource_lease

    owner = f"robodojo-motion-replay-{uuid.uuid4().hex[:12]}"
    lease = claim_resource_lease(lease_plan, owner, os.getpid())
    assigned, started = False, time.time()
    try:
        assign_resource_lease(lease, os.getpid())
        assigned = True
        code = _spawn(session, command, pass_root, timeout_s, scratch, on_start)
    finally:
        # An unassigned lease has no supervisor pid to match; release it by its claim token alone.
        release_resource_lease(owner, owner_pid=os.getpid() if assigned else None, claim_token=lease["claim_token"])
    return code, round(time.time() - started, 1)


def launch_pass(run_root: Path, pass_root: Path, profile: str, *, upto: int | None, timeout_s: float,
                identity: Mapping[str, Any]) -> int:
    """Stage scratch, claim the slot lease, run one worker pass, release both; returns the exit code."""
    from . import host_storage
    from .resources import resolve_execution_resource

    plan = json.loads((run_root / "plan.json").read_text(encoding="utf-8"))
    binding = resolve_execution_resource(profile, runtime_kind="robodojo",
                                         required_host=plan["runtime"].get("required_host"))
    runtime = replay_runtime(plan["runtime"], binding)
    pass_root.mkdir(parents=True, exist_ok=False)
    session, config = _worker_config(runtime, plan, pass_root, run_root / "native")
    replay_id = f"robodojo-motion-replay-{run_root.name}-{pass_root.name}-{uuid.uuid4().hex[:8]}"
    staged = host_storage.stage_launch({"runtime": runtime}, replay_id)
    command = worker_command(session.python_executable, config=config, native_root=run_root / "native",
                             pass_root=pass_root, upto=upto)
    launch = {"run_id": replay_id, "profile": profile, "identity": dict(identity), "command": command, **staged}
    _write_json(pass_root / "launch.json", launch)
    lease_plan = {"schema": "ora.robodojo-infrastructure-plan.v1", "runtime": runtime,
                  "harness_commit": identity["commit"], "episodes": 1, "task_solving": False,
                  "diagnostic": f"robodojo control replay of {run_root.name}"}
    lease_plan["plan_sha256"] = hashlib.sha256(json.dumps(lease_plan, sort_keys=True).encode()).hexdigest()
    def started(pid: int) -> None:
        _write_json(pass_root / "launch.json", {**launch, "worker_pid": pid, "worker_pgid": pid})

    try:
        code, seconds = _leased_spawn(lease_plan, session, command, pass_root, timeout_s,
                                      host_storage.scratch_environment(staged), started)
    finally:
        host_storage.release_run_scratch(pass_root)
    _write_json(pass_root / "launch.json", {**launch, "returncode": code, "seconds": seconds})
    return code


def launch_identity(run_root: Path) -> dict[str, Any]:
    """Refuse a launch from a dirty or non-Git checkout, or with a worker kernel other than the run's."""
    plan = json.loads((run_root / "plan.json").read_text(encoding="utf-8"))
    host = plan["runtime"].get("required_host")
    if host is not None and os.environ.get("ORA_EXECUTION_HOST") != host:
        raise ReplayLaunchError(f"the {host} host marker is required")
    identity = harness_identity(_run_identity(run_root)["harness_commit"])
    if identity["commit"] is None or identity["tracked_worktree_dirty"] is not False:
        raise ReplayLaunchError("replay launches only from a clean committed Git checkout")
    admission = identity["kernel_admission"] or {}
    if admission.get("status") not in ADMITTED_KERNELS:
        raise ReplayLaunchError(f"this checkout's RoboDojo worker kernel is not admitted for the run: {admission}")
    return identity


def _launch_passes(run_root: Path, output: Path, arguments: argparse.Namespace, identity: Mapping[str, Any]) -> None:
    names = PASSES[arguments.mode]
    profiles = [arguments.resource_profile, arguments.second_resource_profile or arguments.resource_profile]
    kwargs = {"upto": arguments.upto, "timeout_s": arguments.timeout_s, "identity": identity}
    if len(names) == 2 and arguments.second_resource_profile:
        # A pool thread blocked in wait() never sees SIGINT, and the workers run in their own
        # sessions; the main thread's handler kills the registered groups so the pool can join.
        previous = {sig: signal.signal(sig, kill_active_groups) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(lambda pair: launch_pass(run_root, output / pair[0], pair[1], **kwargs),
                              zip(names, profiles)))
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        return
    for name, profile in zip(names, profiles):
        launch_pass(run_root, output / name, profile, **kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    run_root = _run_root(arguments.run)
    output = validate_output(run_root, arguments.output, report_only=arguments.report_only)
    if not arguments.report_only:
        identity = launch_identity(run_root)
        output.mkdir(parents=True, exist_ok=True)
        _launch_passes(run_root, output, arguments, identity)
    identity = harness_identity(_run_identity(run_root)["harness_commit"])
    report = build_report(run_root, output, mode=arguments.mode, upto=arguments.upto, identity=identity)
    _write_json(output / REPORT_NAME, report)
    print(json.dumps({"report": str(output / REPORT_NAME), "passed": report["passed"],
                      "failures": report["failures"][:5]}))
    return 0 if report["passed"] else 1


__all__ = [
    "ADMITTED_KERNELS", "DEFAULT_TIMEOUT_S", "MODES", "OBJECT_DETERMINISM_BOUND_M",
    "REPORT_NAME", "REPORT_SCHEMA", "ReplayLaunchError", "WORKER_MODULE", "build_report", "divergence_cause",
    "harness_identity", "kernel_admission", "kill_active_groups", "launch_identity", "launch_pass",
    "line_divergence", "main", "metric_differences",
    "object_pose_divergence", "parse_arguments", "physics_step_divergence", "replay_runtime", "validate_output",
    "worker_command",
]


if __name__ == "__main__":
    sys.exit(main())
