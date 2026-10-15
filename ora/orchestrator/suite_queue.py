"""One single-attempt launch per listed cell across fixed lanes; never repair or reseal.

The maintained queue for RoboTwin, LIBERO-Pro and LIBERO cells
(``suite_cells``), returned from the official-suite queues. A group is one immutable intent: clean source, cell list,
treatment, lanes and readiness report, under
``outputs/<experiment>/groups/<group>/``. Each lane runs one cell at a time;
a worker takes the next pending cell whose Runtime kind its lane registers.

What settles a cell:

- ``sealed_terminal``: the run landed with its seal and a valid or incomplete
  evaluation bound to the queue's source;
- ``trusted_fault_recorded``: a trusted fault before any robot decision (an
  unstable scene, a worker that never started); the attempt is consumed and
  recorded and the queue continues;
- ``unsealed_recorded``: a run that landed without a seal after its
  supervisor exited (an evaluator false positive, a session-level fault); the
  group pauses at the ``unsealed_pause_limit``-th such landing.

The queue waits for a seal while the run's supervisor is alive: a released
lease with a result is not yet a landing, because the supervisor seals after
both. It also never exits while a supervisor it launched is alive,
since ending the queue ends its process group or scheduler step. A preflight
refusal that consumed no attempt goes back to the end of the order, at most
``PREFLIGHT_RETRIES`` times. Launches are spaced ``stagger_s`` apart, and
``max_live`` caps live cells below the lane count, so lanes of two Runtime
kinds can share one cap. Creating ``<group>/drain`` stops new launches;
running cells finish and seal, and ``--resume`` in a new group skips the
cohort's sealed and claimed cells.

Run the queue on the node that owns its lanes: leases and supervisor PIDs are
checked locally.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping

from ora.orchestrator import benchmark_spec, paths, suite_cells
from ora.orchestrator.cli import resource_lease_for_owner
from ora.orchestrator.paths import PROJECT_ROOT
from ora.orchestrator.resources import load_host_profile
from ora.orchestrator.spec import load_run_spec, resolve_run_spec

QUEUE_SCHEMA = "ora.suite-queue.v1"
POLL_SECONDS = 5.0
LAUNCH_TIMEOUT_SECONDS = 1800
TERMINAL_STATES = {"completed", "incomplete", "invalid", "blocked"}
#: Cell states that settle a cell; a group whose cells all reach one completes.
SETTLED_STATES = frozenset({"sealed_terminal", "trusted_fault_recorded", "unsealed_recorded"})
#: Unsealed landings a group tolerates before it pauses (the limit-th pauses).
UNSEALED_PAUSE_LIMIT = 3
#: A preflight refusal that consumed no attempt is retried this many times.
PREFLIGHT_RETRIES = 3
PREFLIGHT_RETRY_DELAY_S = 120.0
#: Seconds between two launches of one group: simulator and worker start-up are the heavy part.
DEFAULT_STAGGER_S = 30.0
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


class UnsealedLanding(RuntimeError):
    """A run landed with a result but no seal, and its supervisor has exited."""

    def __init__(self, run_id: str, record: dict[str, Any]) -> None:
        super().__init__(f"Run landed without a sealed terminal result: {run_id}")
        self.record = record


def _atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def source_commit(root: Path = PROJECT_ROOT) -> str:
    """The clean committed source the queue binds; refused when the checkout is dirty."""
    if subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True).stdout.strip():
        raise RuntimeError("the queue requires a clean committed source")
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
                          check=True).stdout.strip()


def groups_root(experiment: str) -> Path:
    return benchmark_spec.experiment_root(experiment) / "groups"


def sealed_configs(experiment: str, cohort_id: str) -> set[str]:
    """Cells earlier groups of this cohort already drove to sealed terminal evidence."""
    sealed: set[str] = set()
    root = groups_root(experiment)
    for status_path in sorted(root.glob("*/status.json")) if root.is_dir() else []:
        status = _load(status_path)
        if status.get("cohort_id") == cohort_id:
            sealed |= {name for name, cell in status.get("cells", {}).items()
                       if cell.get("state") == "sealed_terminal"}
    return sealed


def claimed_configs(experiment: str, cohort_id: str) -> set[str]:
    """Cells whose single attempt this cohort already consumed, landed or not."""
    root = benchmark_spec.experiment_root(experiment) / "launch-admissions" / cohort_id
    return {path.stem for path in root.glob("*.json")} if root.is_dir() else set()


def supervisor_alive(root: Path, run_id: str) -> bool:
    """True while this node runs the supervisor recorded for ``run_id``."""
    try:
        process = _load(root / "process.json")
        if process.get("run_id") != run_id:
            return False
        cmdline = Path(f"/proc/{int(process['pid'])}/cmdline").read_bytes()
    except (OSError, KeyError, TypeError, ValueError):
        return False
    return run_id.encode() in cmdline


def _action_free_trusted_fault(root: Path, run_id: str, result: Mapping[str, Any]) -> dict[str, Any] | None:
    """The record of a run that failed before any robot decision, else None."""
    evaluation_path = root / "evaluation.json"
    if (result.get("state") != "incomplete" or result.get("termination") != "trusted_fault"
            or result.get("decisions") != 0 or (root / "seal.json").exists()
            or not evaluation_path.is_file()):
        return None
    evaluation = _load(evaluation_path)
    if (evaluation.get("run_id") != run_id or evaluation.get("status") != "invalid"
            or "trusted_fault" not in (evaluation.get("reasons") or [])):
        return None
    return {"state": "trusted_fault_recorded", "run_id": run_id, "run_root": str(root),
            "result": result["state"], "evaluation": "invalid", "official_success": None,
            "termination": "trusted_fault", "decisions": 0, "evaluation_reasons": list(evaluation["reasons"])}


def _landing(root: Path, run_id: str, plan: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    """The settled record of a run whose supervisor has exited (or sealed)."""
    seal_path, evaluation_path = root / "seal.json", root / "evaluation.json"
    fault = _action_free_trusted_fault(root, run_id, result)
    if fault is not None:
        return fault
    if result.get("state") not in TERMINAL_STATES or not seal_path.is_file():
        evaluation = _load(evaluation_path) if evaluation_path.is_file() else {}
        raise UnsealedLanding(run_id, {
            "state": "unsealed_recorded", "run_id": run_id, "run_root": str(root),
            "result": result.get("state"), "termination": result.get("termination"),
            "decisions": result.get("decisions"), "evaluation": evaluation.get("status"),
            "evaluation_reasons": list(evaluation.get("reasons") or [])})
    seal, evaluation = _load(seal_path), _load(evaluation_path)
    if (seal.get("run_id") != run_id or evaluation.get("status") not in {"valid", "incomplete"}
            or seal.get("evaluation_status") != evaluation.get("status")
            or seal.get("sealed_source_commit") != plan["source"]["commit"]
            or seal.get("sealed_source_dirty") is not False or evaluation.get("run_id") != run_id):
        raise RuntimeError(f"Run seal/evaluation identity is not valid: {run_id}")
    return {"state": "sealed_terminal", "run_id": run_id, "run_root": str(root), "result": result["state"],
            "evaluation": evaluation["status"], "official_success": result.get("official_success"),
            "termination": result.get("termination")}


def wait_for_terminal(run_id: str, plan: Mapping[str, Any], *, poll: float = POLL_SECONDS) -> dict[str, Any]:
    """Block until the run settles; a released lease with a result waits for the live supervisor's seal."""
    root = paths.resolve_run_root(run_id)
    recorded = _load(root / "plan.json")
    if recorded.get("source") != plan["source"] or recorded.get("plan_sha256") != plan["plan_sha256"]:
        raise RuntimeError("Launched run identity differs from the pre-launch plan")
    while True:
        lease = resource_lease_for_owner(run_id)
        result_path = root / "result.json"
        alive = supervisor_alive(root, run_id)
        if lease is None and result_path.is_file():
            if (root / "seal.json").is_file() or not alive:
                return _landing(root, run_id, plan, _load(result_path))
        elif lease is None and not alive:
            # A missing lease alone is not an ending (an off-node lease scan
            # used to delete live leases); a dead supervisor without a result is.
            raise RuntimeError(f"Supervisor ended without a result: {run_id}")
        elif lease is not None and not alive and (root / "process.json").is_file():
            raise RuntimeError(f"Supervisor exited with an unreleased lease: {run_id}")
        time.sleep(poll)


def _preflight_failures(plan: Mapping[str, Any], returncode: int | None, log_text: str) -> list[str]:
    """The failed check names of a preflight that refused before an attempt existed."""
    if returncode != 2 or "preflight: BLOCKED" not in log_text.splitlines():
        return []
    report_path = paths.workspace_from_plan(plan) / "preflight" / f"{plan['plan_sha256']}.json"
    if not report_path.is_file():
        return []
    return [check["check"] for check in _load(report_path).get("checks", [])
            if isinstance(check, Mapping) and check.get("passed") is False]


def lane_kinds(lanes: list[str], host: str) -> dict[str, str]:
    """Each lane's registered Runtime kind on ``host``."""
    if not lanes or len(set(lanes)) != len(lanes) or not all(_NAME.fullmatch(lane) for lane in lanes):
        raise RuntimeError("lanes must be distinct registered resource profiles")
    resources = (load_host_profile(host) or {}).get("execution_resources") or {}
    unknown = [lane for lane in lanes if lane not in resources]
    if unknown:
        raise RuntimeError(f"lanes are not registered on {host}: {unknown}")
    return {lane: resources[lane]["runtime_kind"] for lane in lanes}


class Queue:
    """One group: its intent, its status file and the lane workers."""

    def __init__(self, cells: list[suite_cells.Cell], *, lanes: list[str], group_id: str,
                 treatment: suite_cells.Treatment, cells_sha256: str, resume: bool = False,
                 unsealed_pause_limit: int = UNSEALED_PAUSE_LIMIT, stagger_s: float = DEFAULT_STAGGER_S,
                 max_live: int | None = None, root: Path = PROJECT_ROOT) -> None:
        if not _NAME.fullmatch(group_id):
            raise ValueError("group_id must be one safe path component")
        if not 0 <= stagger_s <= 3600:
            raise ValueError("stagger_s must be in [0, 3600]")
        self.treatment, self.root, self.unsealed_pause_limit = treatment, root, unsealed_pause_limit
        if max_live is not None and not 1 <= max_live <= len(lanes):
            raise ValueError("max_live must be between 1 and the number of lanes")
        self.stagger_s, self.launch_gate, self.last_launch = stagger_s, threading.Lock(), None
        # More lanes than live cells lets a cap serve two Runtime kinds whose cells run out at different times.
        self.live = threading.BoundedSemaphore(max_live) if max_live is not None else None
        self.host = benchmark_spec.execution_host(treatment.host)
        self.kinds = lane_kinds(lanes, self.host)
        without_lane = sorted({cell.runtime_kind for cell in cells} - set(self.kinds.values()))
        if without_lane:
            raise RuntimeError(f"no lane registers the Runtime kind of some cells: {without_lane}")
        skip = (sealed_configs(treatment.experiment, treatment.cohort_id)
                | claimed_configs(treatment.experiment, treatment.cohort_id)) if resume else set()
        self.cells = {cell.config: cell for cell in cells if cell.config not in skip}
        self.report = treatment.service_report.expanduser().resolve(strict=True)
        self.report_sha256 = hashlib.sha256(self.report.read_bytes()).hexdigest()
        self.source = source_commit(root)
        self.group = groups_root(treatment.experiment) / group_id
        self.lock, self.blocked = threading.Lock(), threading.Event()
        self.pending = list(self.cells)
        self.launched: dict[str, Path] = {}
        sealed = sealed_configs(treatment.experiment, treatment.cohort_id) if resume else set()
        skipped = {cell.config for cell in cells} - set(self.cells)
        self.state: dict[str, Any] = {
            "schema": QUEUE_SCHEMA, "group_id": group_id, "experiment": treatment.experiment,
            "cohort_id": treatment.cohort_id, "source_commit": self.source, "pid": os.getpid(),
            "host": self.host, "lanes": dict(self.kinds), "cells_sha256": cells_sha256,
            "provider_profile": treatment.provider_profile, "rounds": treatment.rounds,
            "service_report": str(self.report), "service_report_sha256": self.report_sha256,
            "unsealed_pause_limit": unsealed_pause_limit, "stagger_s": stagger_s,
            "max_live": max_live, "started_unix_s": time.time(),
            "queue_status": "running", "dequeue_order": [],
            "skipped_sealed": sorted(skipped & sealed), "skipped_claimed_unsealed": sorted(skipped - sealed),
            "cells": {name: {"state": "pending", **cell.summary()} for name, cell in self.cells.items()},
        }

    # --- status ---------------------------------------------------------------------------------

    def persist(self) -> None:
        self.state["updated_unix_s"] = time.time()
        _atomic(self.group / "status.json", self.state)

    def record(self, config: str, **fields: Any) -> None:
        with self.lock:
            self.state["cells"][config].update(fields)
            self.persist()

    def pause(self, config: str | None, reason: str) -> None:
        with self.lock:
            self.blocked.set()
            self.state["queue_status"] = "paused"
            self.state.setdefault("first_error", {"config": config, "reason": reason})
            if config is not None:
                self.state["cells"][config].update(state="blocked", reason=reason)
            self.persist()

    def check_source(self) -> None:
        if source_commit(self.root) != self.source:
            raise RuntimeError("Queue source changed after its immutable intent")
        if hashlib.sha256(self.report.read_bytes()).hexdigest() != self.report_sha256:
            raise RuntimeError("Queue service report changed after its immutable intent")

    # --- one cell -------------------------------------------------------------------------------

    def await_launch_slot(self) -> None:
        """Space this group's launches at least ``stagger_s`` apart (heavy simulator start-up)."""
        with self.launch_gate:
            if self.last_launch is not None:
                wait = self.last_launch + self.stagger_s - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
            self.last_launch = time.monotonic()

    def execute(self, config: str, lane: str) -> None:
        self.check_source()
        cell = self.cells[config]
        raw = suite_cells.build_spec(cell, lane, self.treatment)
        spec = suite_cells.compose(cell, raw, self.treatment).resolve(strict=True)
        if load_run_spec(spec) != raw:
            raise RuntimeError("Composed RunSpec differs from the built one")
        plan = resolve_run_spec(raw, root=self.root)
        if plan.get("source") != {"commit": self.source, "dirty": False}:
            raise RuntimeError("Resolved cell is not bound to the queue's clean source")
        admission = plan.get("launch_admission", {})
        if admission.get("mode") != "single_attempt_cell" or admission.get("retry_allowed") is not False:
            raise RuntimeError("the queue requires Runtime-enforced single-attempt admission")
        if plan.get("runtime", {}).get("resource_binding", {}).get("profile_id") != lane:
            raise RuntimeError("Resolved cell does not use its lane resource")
        self.check_source()
        retry = self.state["cells"][config].get("preflight_retries", 0)
        log = self.group / (f"{config}-launch.log" if not retry else f"{config}-launch-r{retry}.log")
        with self.lock:
            if self.blocked.is_set():
                self.state["cells"][config]["state"] = "paused_before_launch"
                self.persist()
                return
            self.state["cells"][config].update(state="launching", log=str(log), spec=str(spec),
                                               plan_id=plan["plan_id"], plan_sha256=plan["plan_sha256"])
            self.persist()
        launch_error, returncode = None, None
        try:
            with log.open("x", encoding="utf-8") as stream:
                # The stagger is stamped next to the launch: status and log writes on a slow shared
                # filesystem after the gate used to eat into the spacing.
                self.await_launch_slot()
                if self.blocked.is_set():
                    self.record(config, state="paused_before_launch")
                    return
                process = subprocess.run([str(self.root / "scripts/ora"), "launch", str(spec), "--yes"],
                                         cwd=self.root, stdout=stream, stderr=subprocess.STDOUT,
                                         timeout=LAUNCH_TIMEOUT_SECONDS)
            returncode = process.returncode
            if returncode:
                launch_error = f"Launch/admission exit {returncode}; see {log}"
        except Exception as exc:  # a launch that failed to run is reported, not raised from the worker
            launch_error = str(exc)
        log_text = log.read_text(encoding="utf-8") if log.is_file() else ""
        launched = [line.split()[1] for line in log_text.splitlines()
                    if line.startswith("launched ") and len(line.split()) >= 2]
        if len(launched) != 1:
            failed = _preflight_failures(plan, returncode, log_text)
            self.record(config, launch_error=launch_error, launch_returncode=returncode,
                        attempt_consumed=False if failed else None, failed_checks=failed)
            raise RuntimeError(launch_error or f"Preflight refused before attempt creation: {failed}; see {log}")
        run_id = launched[0]
        with self.lock:
            self.launched[run_id] = paths.resolve_run_root(run_id)
        self.record(config, state="running", run_id=run_id, attempt_consumed=True, launch_returncode=returncode)
        try:
            self.record(config, **wait_for_terminal(run_id, plan))
        except UnsealedLanding as landing:
            with self.lock:
                unsealed = 1 + sum(1 for item in self.state["cells"].values()
                                   if item.get("state") == "unsealed_recorded")
            self.record(config, **landing.record)
            if unsealed >= self.unsealed_pause_limit:
                raise

    def requeue_refused_preflight(self, config: str, error: Exception) -> bool:
        """Put a cell that preflight refused before its attempt existed back at the end of the order."""
        with self.lock:
            cell = self.state["cells"][config]
            if (cell.get("attempt_consumed") is not False or not cell.get("failed_checks")
                    or cell.get("preflight_retries", 0) >= PREFLIGHT_RETRIES or self.blocked.is_set()):
                return False
            cell.update(state="pending", preflight_retries=cell.get("preflight_retries", 0) + 1,
                        last_refusal=str(error)[:300])
            self.pending.append(config)
            self.persist()
        time.sleep(PREFLIGHT_RETRY_DELAY_S)
        return True

    # --- lanes ----------------------------------------------------------------------------------

    def _next(self, lane: str) -> str | None:
        """The next pending cell for ``lane``; None when the group stops or nothing fits the lane."""
        with self.lock:
            if self.blocked.is_set():
                return None
            if (self.group / "drain").exists():
                self.blocked.set()
                self.state["queue_status"] = "paused"
                self.state.setdefault("first_error", {"config": None, "reason": "drained by operator"})
                self.persist()
                return None
            for index, config in enumerate(self.pending):
                if self.cells[config].runtime_kind == self.kinds[lane]:
                    del self.pending[index]
                    self.state["dequeue_order"].append(config)
                    self.state["cells"][config].update(state="preflight", lane=lane)
                    self.persist()
                    return config
            return None

    def worker(self, lane: str) -> None:
        while True:
            if self.live is not None:
                self.live.acquire()
            try:
                config = self._next(lane)
                if config is None:
                    return
                try:
                    self.execute(config, lane)
                except Exception as exc:
                    if not self.requeue_refused_preflight(config, exc):
                        self.pause(config, str(exc))
            finally:
                if self.live is not None:
                    self.live.release()

    def await_supervisors(self, poll: float = POLL_SECONDS) -> None:
        """Return only when no supervisor this group launched is alive."""
        while any(supervisor_alive(root, run_id) for run_id, root in list(self.launched.items())):
            time.sleep(poll)

    def run(self) -> int:
        self.group.mkdir(parents=True, exist_ok=False)
        _atomic(self.group / "intent.json", self.state)
        self.persist()
        threads = [threading.Thread(target=self.worker, args=(lane,), name=f"suite-{lane}") for lane in self.kinds]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.await_supervisors()
        with self.lock:
            unsettled = [name for name in self.cells if self.state["cells"][name]["state"] not in SETTLED_STATES]
            if unsettled and not self.blocked.is_set():
                self.blocked.set()
                self.state.setdefault("first_error", {
                    "config": unsettled[0], "reason": "Worker exited before the cell settled"})
            self.state["queue_status"] = "paused" if self.blocked.is_set() else "complete"
            self.state["finished_unix_s"] = time.time()
            self.persist()
        return 2 if self.blocked.is_set() else 0


# --- operator entry point -------------------------------------------------------------------------

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="launch every listed cell once across the lanes")
    run.add_argument("--cells", type=Path, required=True, help=f"cell list ({suite_cells.CELLS_SCHEMA})")
    run.add_argument("--experiment", required=True)
    run.add_argument("--cohort", required=True, help="single-attempt admission cohort")
    run.add_argument("--group", required=True, help="new group id under outputs/<experiment>/groups")
    run.add_argument("--lanes", nargs="+", required=True, help="registered resource profiles, one cell each")
    run.add_argument("--service-report", type=Path, required=True)
    run.add_argument("--host")
    run.add_argument("--provider-profile")
    run.add_argument("--provider-catalog", type=Path)
    run.add_argument("--rounds", type=int, help="one persistent K-round session per cell")
    run.add_argument("--seed-manifests", type=Path, help="RoboTwin: <dir>/<task>.json admits each seed")
    run.add_argument("--resume", action="store_true", help="skip the cohort's sealed and claimed cells")
    run.add_argument("--unsealed-pause-limit", type=int, default=UNSEALED_PAUSE_LIMIT)
    run.add_argument("--max-live", type=int,
                     help="at most this many cells live at once (default: one per lane)")
    run.add_argument("--stagger-s", type=float, default=DEFAULT_STAGGER_S,
                     help=f"seconds between two launches (default {DEFAULT_STAGGER_S:g})")
    run.add_argument("--yes", action="store_true", help="confirm live launches (required)")
    for name, text in (("drain", "stop new launches; running cells finish"), ("status", "print cell counts")):
        command = commands.add_parser(name, help=text)
        command.add_argument("--experiment", required=True)
        command.add_argument("--group", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Operator entry point: ``python -m ora.orchestrator.suite_queue``."""
    args = _parser().parse_args(argv)
    if args.command in {"drain", "status"}:
        group = groups_root(args.experiment) / args.group
        if args.command == "drain":
            (group / "drain").touch(exist_ok=True)
            print(f"drain requested: {group}")
            return 0
        status = _load(group / "status.json")
        counts: dict[str, int] = {}
        for cell in status["cells"].values():
            counts[cell["state"]] = counts.get(cell["state"], 0) + 1
        print(json.dumps({"queue_status": status["queue_status"], "cells": counts,
                          "first_error": status.get("first_error")}, indent=2))
        return 0
    if not args.yes:
        print("refusing to launch without --yes", file=sys.stderr)
        return 2
    cells, digest = suite_cells.load_cells(args.cells)
    treatment = suite_cells.Treatment(
        experiment=args.experiment, cohort_id=args.cohort, service_report=args.service_report, host=args.host,
        provider_profile=args.provider_profile, provider_catalog=args.provider_catalog, rounds=args.rounds,
        seed_manifests=args.seed_manifests)
    queue = Queue(cells, lanes=args.lanes, group_id=args.group, treatment=treatment, cells_sha256=digest,
                  resume=args.resume, unsealed_pause_limit=args.unsealed_pause_limit, stagger_s=args.stagger_s,
                  max_live=args.max_live)
    return queue.run()


if __name__ == "__main__":
    raise SystemExit(main())
