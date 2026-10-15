"""Helpers for ``ora run``: confirmed long steps with a progress line, and resident services.

The services part does what an operator does by hand before a GPU task:
``model-services status``; ``start`` when nothing answers; wait for warmup;
reuse the readiness report that names the running instances, or ``qualify``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .paths import PROJECT_ROOT

#: Measured on the 2026-10-14 verification: both services ready ~170 s after start.
SERVICE_START_TIMEOUT_S = 900
#: A report binds for 24 h; reuse one only while it leaves this much for the
#: interactive session's own launch preflight.
REPORT_REUSE_MAX_AGE_S = 20 * 3600


def _clock(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes:02d}:{seconds:02d}"


def confirm(message: str, *, assume_yes: bool, stream=None, reader: Callable[[str], str] | None = None) -> None:
    """Show what is about to run; proceed on ``--yes`` or an interactive yes, refuse otherwise."""
    output = stream or sys.stderr
    print(message, file=output, flush=True)
    if assume_yes:
        return
    if reader is None:
        if not sys.stdin.isatty():
            raise RuntimeError("this step runs simulator episodes; rerun with --yes to confirm it")
        reader = input
    if reader("Proceed? [y/N] ").strip().lower() not in {"y", "yes"}:
        raise RuntimeError("not confirmed; nothing was started")


def run_with_progress(command: Sequence[str], *, label: str, estimate: str, log: Path, stream=None,
                      env: Mapping[str, str] | None = None, poll_s: float = 1.0,
                      popen=subprocess.Popen, accepted_codes: Sequence[int] = (0,)) -> str:
    """Run ``command`` with its output in ``log``; show elapsed time against ``estimate``; return stdout."""
    output = stream or sys.stderr
    log.parent.mkdir(parents=True, exist_ok=True)
    stdout_path = log.with_suffix(".stdout")
    live = hasattr(output, "isatty") and output.isatty()
    started = time.monotonic()
    last_line = started
    with log.open("ab") as errors, stdout_path.open("wb") as captured:
        process = popen([str(item) for item in command], cwd=PROJECT_ROOT, stdout=captured, stderr=errors,
                        env={**os.environ, **dict(env or {})})
        while process.poll() is None:
            elapsed = time.monotonic() - started
            if live:
                print(f"\r  {label}: {_clock(elapsed)} elapsed ({estimate}; log {log})", end="", file=output,
                      flush=True)
            elif time.monotonic() - last_line >= 60:
                print(f"  {label}: {_clock(elapsed)} elapsed ({estimate})", file=output, flush=True)
                last_line = time.monotonic()
            time.sleep(poll_s)
    elapsed = time.monotonic() - started
    if live:
        print("", file=output)
    result = stdout_path.read_text(errors="replace")
    if process.returncode not in accepted_codes:
        tail = (log.read_text(errors="replace").splitlines() + result.splitlines())[-20:]
        raise RuntimeError(f"{label} failed after {_clock(elapsed)} (exit {process.returncode}; log {log})\n"
                           + "\n".join(tail))
    print(f"  {label}: done in {_clock(elapsed)}", file=output, flush=True)
    return result


def _instances(health: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    from ora.agent.model_services.operator import FIELDS

    return {name: {key: item["health"].get(key) for key in FIELDS} for name, item in health.items()}


def reusable_report(directory: Path, health: Mapping[str, Any]) -> Path | None:
    """The newest readiness report that is valid now and names the running instances."""
    from ora.agent.model_services import operator

    if not all(item.get("ready") is True for item in health.values()):
        return None
    running = _instances(health)
    for path in sorted(directory.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            bound = operator.binding(path, "ora-run-reuse-probe", [])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        created = next(iter(bound["services"].values()), {}).get("qualification", {}).get("created_unix_s", 0)
        if time.time() - created > REPORT_REUSE_MAX_AGE_S:
            continue
        recorded = {name: {key: child.get(key) for key in operator.FIELDS}
                    for name, child in bound["services"].items()}
        if recorded == running:
            return path
    return None


def ensure_services(*, stream=None, timeout_s: float = SERVICE_START_TIMEOUT_S, poll_s: float = 5.0) -> Path:
    """Ready SAM3/Contact-GraspNet services and a readiness report for them; return the report."""
    from ora.agent.model_services import operator
    from . import paths
    from .model_services_cli import state_directory

    output = stream or sys.stderr
    state = state_directory(None)
    health = operator.health()
    if not all(item.get("ready") is True for item in health.values()):
        print("Starting the SAM3 and Contact-GraspNet services (~3 min, SAM3 warmup ~70 s) ...",
              file=output, flush=True)
        started = operator.lifecycle("start", root=paths.PROJECT_ROOT, state=state,
                                     runtime_state_root=paths.STATE_ROOT)
        logs = [item["log"] for item in started.values() if item.get("log")]
        begin = time.monotonic()
        while True:
            health = operator.health()
            if all(item.get("ready") is True for item in health.values()):
                break
            elapsed = time.monotonic() - begin
            if elapsed > timeout_s:
                raise RuntimeError(f"services not ready after {_clock(elapsed)}; see {', '.join(logs) or state}")
            waiting = ", ".join(name for name, item in health.items() if item.get("ready") is not True)
            print(f"  waiting for {waiting}: {_clock(elapsed)} elapsed", file=output, flush=True)
            time.sleep(poll_s)
    directory = state / "qualification"
    report = reusable_report(directory, health)
    if report is not None:
        print(f"Services ready; reusing readiness report {report}", file=output, flush=True)
        return report
    report = operator.qualify(directory)
    print(f"Services ready; readiness report {report}", file=output, flush=True)
    return report


__all__ = ["REPORT_REUSE_MAX_AGE_S", "SERVICE_START_TIMEOUT_S", "confirm", "ensure_services", "reusable_report", "run_with_progress"]
