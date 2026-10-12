"""One single-attempt launch per cell across fixed lanes; never repair or reseal.

Generalized from the reviewed G8 queue: the selection, lane count and cohort
come from the caller, the safety checks do not.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Mapping

from examples.official_suite import run
from ora.orchestrator import paths
from ora.orchestrator.cli import resource_lease_for_owner
from ora.orchestrator.spec import load_run_spec, resolve_run_spec

POLL_SECONDS = 5.0
LAUNCH_TIMEOUT_SECONDS = 1800
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
TERMINAL_STATES = {"completed", "incomplete", "invalid", "blocked"}
QUEUE_SCHEMA = "ora.official-suite-queue.v1"


def _source_commit() -> str:
    if subprocess.run(["git", "status", "--porcelain"], cwd=run.ROOT, capture_output=True, text=True).stdout.strip():
        raise RuntimeError("queue requires a clean committed source")
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=run.ROOT, capture_output=True, text=True, check=True).stdout.strip()


def _atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sealed_configs(cohort_id: str) -> set[str]:
    """Configs earlier groups of this cohort already drove to sealed terminal evidence."""
    groups = run.workspace() / "groups"
    sealed = set()
    for status_path in sorted(groups.glob("*/status.json")) if groups.exists() else []:
        status = _load(status_path)
        if status.get("cohort_id") != cohort_id:
            continue
        sealed |= {name for name, cell in status.get("cells", {}).items() if cell.get("state") == "sealed_terminal"}
    return sealed


def _wait_for_terminal(run_id: str, plan: Mapping[str, Any]) -> dict[str, Any]:
    root = paths.resolve_run_root(run_id)
    recorded = _load(root / "plan.json")
    if recorded.get("source") != plan["source"] or recorded.get("plan_sha256") != plan["plan_sha256"]:
        raise RuntimeError("Launched run identity differs from the pre-launch plan")
    while True:
        lease = resource_lease_for_owner(run_id)
        result_path = root / "result.json"
        if result_path.is_file():
            result = _load(result_path)
            if lease is None:
                seal_path, evaluation_path = root / "seal.json", root / "evaluation.json"
                if result.get("state") not in TERMINAL_STATES or not seal_path.is_file():
                    raise RuntimeError(f"Run landed without a sealed terminal result: {run_id}")
                seal, evaluation = _load(seal_path), _load(evaluation_path)
                if (seal.get("run_id") != run_id or evaluation.get("status") not in {"valid", "incomplete"}
                        or seal.get("evaluation_status") != evaluation.get("status")
                        or seal.get("sealed_source_commit") != plan["source"]["commit"]
                        or seal.get("sealed_source_dirty") is not False or evaluation.get("run_id") != run_id):
                    raise RuntimeError(f"Run seal/evaluation identity is not valid: {run_id}")
                return {"state": "sealed_terminal", "run_id": run_id, "run_root": str(root),
                        "result": result["state"], "evaluation": evaluation["status"],
                        "official_success": result.get("official_success"),
                        "termination": result.get("termination")}
        elif lease is None:
            raise RuntimeError(f"Supervisor lease ended without a result: {run_id}")
        process_path = root / "process.json"
        if lease is not None and process_path.is_file():
            process = _load(process_path)
            if process.get("run_id") != run_id:
                raise RuntimeError(f"Supervisor process identity mismatch: {run_id}")
            try:
                os.kill(int(process["pid"]), 0)
            except ProcessLookupError as exc:
                raise RuntimeError(f"Supervisor exited with an unreleased lease: {run_id}") from exc
        time.sleep(POLL_SECONDS)


def ordered(cells, lanes, *, resume: bool, cohort_id: str) -> list[run.Cell]:
    if not lanes or len(set(lanes)) != len(lanes) or not all(_NAME.fullmatch(lane) for lane in lanes):
        raise RuntimeError("lanes must be distinct registered resource profiles")
    if not cells or len({cell.config for cell in cells}) != len(cells):
        raise RuntimeError("cells must be a non-empty selection with distinct configs")
    skip = sealed_configs(cohort_id) if resume else set()
    return [cell for cell in cells if cell.config not in skip]


def run_queue(cells, *, lanes, group_id: str, host=None, service_report=None,
              provider_profile=None, resume: bool = False) -> int:
    definition = run.definition()
    if definition.get("live_launch_authorized") is not True:
        raise RuntimeError("This suite is not authorized for live launch")
    if not _NAME.fullmatch(group_id):
        raise ValueError("group_id must be one safe path component")
    if service_report is None:
        raise RuntimeError("a resident-service readiness report is required to launch")
    cohort_id = definition["cohort_id"]
    queued = ordered(cells, lanes, resume=resume, cohort_id=cohort_id)
    report = service_report.expanduser().resolve(strict=True)
    report_digest = hashlib.sha256(report.read_bytes()).hexdigest()
    source = _source_commit()
    group = run.workspace() / "groups" / group_id
    group.mkdir(parents=True, exist_ok=False)
    lock = threading.Lock()
    blocked = threading.Event()
    state: dict[str, Any] = {
        "schema": QUEUE_SCHEMA, "group_id": group_id, "cohort_id": cohort_id, "source_commit": source,
        "pid": os.getpid(), "lanes": list(lanes), "host": run.declared_host(host),
        "service_report": str(report), "service_report_sha256": report_digest,
        "started_unix_s": time.time(), "queue_status": "running", "dequeue_order": [],
        "skipped_sealed": sorted({c.config for c in cells} - {c.config for c in queued}),
        "cells": {cell.config: {"state": "pending", "arm": cell.arm, "task": cell.task, "seed": cell.seed}
                  for cell in queued},
    }
    by_config = {cell.config: cell for cell in queued}

    def persist() -> None:
        state["updated_unix_s"] = time.time()
        try:
            _atomic(group / "status.json", state)
        except Exception:
            blocked.set()
            state["queue_status"] = "paused"
            raise

    def record(config: str, **fields: Any) -> None:
        with lock:
            state["cells"][config].update(fields)
            persist()

    def pause(config: str, error: Exception | str) -> None:
        with lock:
            blocked.set()
            state["queue_status"] = "paused"
            state.setdefault("first_error", {"config": config, "reason": str(error)})
            state["cells"][config].update(state="blocked", reason=str(error))
            persist()

    _atomic(group / "intent.json", state)
    persist()

    def check_source() -> None:
        if _source_commit() != source:
            raise RuntimeError("Queue source changed after its immutable intent")
        if hashlib.sha256(report.read_bytes()).hexdigest() != report_digest:
            raise RuntimeError("Queue service report changed after its immutable intent")

    def execute(config: str, lane: str) -> None:
        check_source()
        cell = by_config[config]
        kwargs = dict(host=host, service_report=report, provider_profile=provider_profile)
        raw = run.build_spec(cell, lane, **kwargs)
        plan = resolve_run_spec(raw, root=run.ROOT)
        if plan.get("source") != {"commit": source, "dirty": False}:
            raise RuntimeError("Resolved cell is not bound to the queue's clean source")
        admission = plan.get("launch_admission", {})
        if admission.get("mode") != "single_attempt_cell" or admission.get("retry_allowed") is not False:
            raise RuntimeError("The suite requires Runtime-enforced single-attempt admission")
        if plan.get("runtime", {}).get("resource_binding", {}).get("profile_id") != lane:
            raise RuntimeError("Resolved cell does not use its lane resource")
        spec = run.compose(cell, lane, **kwargs).resolve(strict=True)
        if load_run_spec(spec) != raw:
            raise RuntimeError("Composed RunSpec differs from the reviewed build_spec result")
        check_source()
        log = group / f"{config}-launch.log"
        with lock:
            if blocked.is_set():
                state["cells"][config]["state"] = "paused_before_launch"
                persist()
                return
            state["cells"][config].update(state="launching", log=str(log), spec=str(spec),
                                          plan_id=plan["plan_id"], plan_sha256=plan["plan_sha256"])
            persist()
        launch_error = None
        returncode = None
        try:
            with log.open("x", encoding="utf-8") as stream:
                process = subprocess.run([str(run.ROOT / "scripts/ora"), "launch", str(spec), "--yes"],
                                         cwd=run.ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                         timeout=LAUNCH_TIMEOUT_SECONDS)
            returncode = process.returncode
            if process.returncode:
                launch_error = f"Launch/admission exit {process.returncode}; see {log}"
        except Exception as exc:
            launch_error = str(exc)
        log_text = log.read_text(encoding="utf-8") if log.is_file() else ""
        launched = [line.split()[1] for line in log_text.splitlines()
                    if line.startswith("launched ") and len(line.split()) >= 2]
        if len(launched) != 1:
            failed = []
            if returncode == 2 and "preflight: BLOCKED" in log_text.splitlines():
                report_path = paths.workspace_from_plan(plan) / "preflight" / f"{plan['plan_sha256']}.json"
                if report_path.is_file():
                    preflight = _load(report_path)
                    failed = [check["check"] for check in preflight.get("checks", [])
                              if isinstance(check, Mapping) and check.get("passed") is False]
            record(config, launch_error=launch_error, launch_returncode=returncode, attempt_consumed=False if failed else None,
                   failed_checks=failed)
            raise RuntimeError(launch_error or f"Preflight refused before attempt creation: {failed}; see {log}")
        run_id = launched[0]
        record(config, state="running", run_id=run_id, attempt_consumed=True, launch_returncode=returncode)
        record(config, **_wait_for_terminal(run_id, plan))

    names = [cell.config for cell in queued]
    cursor = [0]

    def worker(lane: str) -> None:
        while True:
            with lock:
                if blocked.is_set() or cursor[0] >= len(names):
                    return
                config = names[cursor[0]]
                cursor[0] += 1
                state["dequeue_order"].append(config)
                state["cells"][config].update(state="preflight", lane=lane)
                persist()
            try:
                execute(config, lane)
            except Exception as exc:
                pause(config, exc)

    threads = [threading.Thread(target=worker, args=(lane,), name=f"suite-{lane}") for lane in lanes]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    with lock:
        if not blocked.is_set() and any(state["cells"][name]["state"] != "sealed_terminal" for name in names):
            blocked.set()
            state["queue_status"] = "paused"
            state.setdefault("first_error", {"config": next(n for n in names if state["cells"][n]["state"] != "sealed_terminal"),
                                             "reason": "Worker exited before the cell reached sealed terminal evidence"})
        state["queue_status"] = "paused" if blocked.is_set() else "complete"
        state["finished_unix_s"] = time.time()
        persist()
    return 2 if blocked.is_set() else 0
