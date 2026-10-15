"""Seed-scan lane execution: start stagger, lease-probe retry, seed watchdog, ledger validity.

``robotwin_evaluation qualify`` runs one isolated oracle worker per case. This
module owns how such a worker is started and supervised on a busy node, and
which of its ledger records sealing may count. It never imports the simulator.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

# Lanes start this far apart. Dozens of oracle workers initialising CUDA at once
# slowed nvidia-smi past the lease probe's timeout, and every lane whose claim
# saw a failed probe was dropped for the whole scan (18 of 64 lanes).
LANE_START_STAGGER_S = 3.0

# A claim refused only because the device probe failed is retried with backoff;
# every other refusal (slot already leased, card full) still fails at once.
PROBE_CLAIM_ATTEMPTS = 5
PROBE_RETRY_BACKOFF_S = 10.0

# A seed whose expert run gives no result for this long is recorded as
# rejected (classification oracle_timeout) and the worker resumes at the next
# seed. On one cluster a few seeds hung for hours in every scan that reached them,
# which froze their task's contiguous prefix (a seed
# with no result within the bound counts as rejected). The bound only has to
# separate a hang from a slow seed: with 70 workers per node on the slow
# remaining tasks, sound seeds took 27-46 min (first seed of a process, start-up
# included, up to 75 min), and 30/90 min bounds rejected 133 of them on
# 2026-10-03. Hung seeds never finish, so a generous bound costs little.
SEED_WATCHDOG_S = 7200
SEED_WATCHDOG_FIRST_S = 10800
SEED_WATCHDOG_POLL_S = 30.0
ORACLE_TIMEOUT_SCHEMA = "ora.robotwin-evaluation-oracle-timeout.v1"
WATCHDOG_RECORDER = "robotwin_evaluation.seed_watchdog"
#: Classifications that say nothing about the seed: the worker stops and
#: resumes at that same seed (``qualification._resume_requested_seed``).
UNSCANNED_CLASSIFICATIONS = frozenset({"infra_failure", "invalid_artifact"})


def claim_with_probe_retry(claim, plan, owner, pid, *, sleep=time.sleep):
    """Claim a lease, retrying only a refusal caused by an unreadable probe."""
    for attempt in range(PROBE_CLAIM_ATTEMPTS):
        try:
            return claim(plan, owner, pid)
        except RuntimeError as exc:
            if "rule=device_probe_unavailable" not in str(exc) or attempt == PROBE_CLAIM_ATTEMPTS - 1:
                raise
            sleep(PROBE_RETRY_BACKOFF_S * (attempt + 1))


def read_ledger_records(ledger: Path):
    """Parse a seed ledger; a torn final line from a running worker is skipped."""
    records = []
    for line in ledger.read_text().splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def valid_ledger_records(records):
    """Ledger records in order, without records that say nothing about their seed.

    A timeout says a seed is hung only if it ran past the bound in force now:
    earlier builds stopped sound slow seeds at 30 min (90 min for a process's
    first seed, start-up included; the first watchdog recorded no bound and
    used 30 min throughout). Such seeds count as unscanned and are scanned again.

    An ``infra_failure`` or ``invalid_artifact`` record says nothing about the
    seed either (the worker stops and resumes at that same seed); on
    2026-10-03 a GPU needing reset wrote 173 of them. They are dropped too."""
    def stands(record):
        if record.get("classification") in UNSCANNED_CLASSIFICATIONS:
            return False
        if record.get("recorded_by") != WATCHDOG_RECORDER:
            return True
        bound = SEED_WATCHDOG_FIRST_S if record.get("first_in_process") else SEED_WATCHDOG_S
        return float(record.get("limit_s", 1800)) >= bound
    return [record for record in records if stands(record)]


def _running_attempt(task_root: Path):
    """The newest attempt directory the worker opened but has not finalized."""
    attempts = task_root / "attempts"
    if not attempts.is_dir():
        return None
    pending = [d for d in attempts.glob("seed-*-try-*") if d.is_dir() and not (d / "attempt.json").exists()]
    return max(pending, key=lambda d: d.stat().st_mtime, default=None)


def _finalized_attempts(task_root: Path) -> int:
    attempts = task_root / "attempts"
    return sum(1 for _ in attempts.glob("seed-*-try-*/attempt.json")) if attempts.is_dir() else 0


def record_oracle_timeout(task_root: Path, attempt: Path, *, task: str, task_config: str,
                          limit_s: float, now: float, started: float, first_in_process: bool) -> int:
    """Finalize a hung attempt as rejected and append it to the worker's ledger.

    The worker resumes after the highest recorded seed, so a restarted worker
    moves on to the next seed. Returns the timed-out seed."""
    from ora.orchestrator.robotwin_evaluation import digest, save

    seed = int(attempt.name.split("-")[1])
    record = {"schema": ORACLE_TIMEOUT_SCHEMA, "task": task, "task_config": task_config,
              "requested_seed": seed, "started_at_unix": started, "finished_at_unix": now,
              "duration_seconds": now - started, "classification": "oracle_timeout", "accepted": False,
              "error": f"expert gave no result within {int(limit_s)} s; stopped by the scan watchdog",
              "recorded_by": WATCHDOG_RECORDER, "limit_s": limit_s, "first_in_process": first_in_process}
    record["record_sha256"] = digest(record)
    save(attempt / "attempt.json", record)
    with (task_root / "seed-ledger.jsonl").open("a", encoding="utf-8") as ledger:
        ledger.write(json.dumps(record, sort_keys=True) + "\n")
    return seed


def run_with_seed_watchdog(command, *, cwd, env, log, task_root: Path, task: str, task_config: str,
                           max_restarts: int, limit_s: float = SEED_WATCHDOG_S,
                           first_limit_s: float = SEED_WATCHDOG_FIRST_S,
                           poll_s: float = SEED_WATCHDOG_POLL_S, clock=time.time, sleep=time.sleep):
    """Run one case's worker; a seed stuck past its bound is rejected and the worker restarted.

    A process's first seed is bounded by ``first_limit_s`` (start-up included),
    later seeds by ``limit_s``. There is no other start-up or per-case timeout.
    Returns (returncode, timed-out seeds); 124 when more than ``max_restarts``
    seeds timed out."""
    timed_out, first_seen = [], {}
    while True:
        baseline = _finalized_attempts(task_root)
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        hung = None
        while process.poll() is None:
            sleep(poll_s)
            attempt = _running_attempt(task_root)
            if attempt is None:
                continue
            # Writes inside the attempt move its mtime; time the seed from when it was first seen.
            started = first_seen.setdefault(str(attempt), min(clock(), attempt.stat().st_mtime))
            first = _finalized_attempts(task_root) == baseline
            if clock() - started > (first_limit_s if first else limit_s):
                hung = attempt
                break
        if hung is None:
            return process.returncode, timed_out
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        if (hung / "attempt.json").exists():   # it finished while being stopped
            continue
        timed_out.append(record_oracle_timeout(
            task_root, hung, task=task, task_config=task_config, limit_s=first_limit_s if first else limit_s,
            now=clock(), started=first_seen[str(hung)], first_in_process=first))
        if len(timed_out) > max_restarts:
            return 124, timed_out
