"""Shared launch-environment and Linux process identity safeguards."""

from __future__ import annotations

import hashlib
import os
import signal
import time
from pathlib import Path
from typing import Any, Mapping


SERVICE_ENV_ALLOWLIST = frozenset({
    "PATH", "LD_LIBRARY_PATH", "CUDA_HOME", "CUDA_PATH", "HOME",
    "XDG_CACHE_HOME", "HF_HOME", "TORCH_HOME", "TRITON_CACHE_DIR",
    "TMPDIR", "TEMP", "TMP", "LANG", "LC_ALL", "OMP_NUM_THREADS",
    "MKL_NUM_THREADS", "NVIDIA_VISIBLE_DEVICES",
    "NVIDIA_DRIVER_CAPABILITIES",
})


def isolated_service_environment(
    *, worker_environment: Mapping[str, str],
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return an allowlisted, credential-free and offline worker environment."""

    environment = {
        name: value for name, value in os.environ.items()
        if name in SERVICE_ENV_ALLOWLIST and value
    }
    environment.update({str(key): str(value) for key, value in worker_environment.items()})
    if extra is not None:
        environment.update({str(key): str(value) for key, value in extra.items()})
    environment.update({
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "WANDB_MODE": "disabled",
        "TOKENIZERS_PARALLELISM": "false",
    })
    return environment


def process_alive(pid: Any) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        state = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()[2]
    except (OSError, IndexError):
        return False
    return state != "Z"


def _proc_start_ticks(pid: int) -> int:
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    _prefix, separator, suffix = raw.rpartition(")")
    fields = suffix.strip().split() if separator else []
    if len(fields) <= 19:
        raise RuntimeError("service process stat is malformed")
    return int(fields[19])


def capture_service_process_identity(
    pid: int, command: list[str], *, wait_s: float = 1.0,
) -> dict[str, Any]:
    """Bind a record to one exact Linux boot and process instance."""

    expected = b"\0".join(item.encode() for item in command) + b"\0"
    deadline = time.monotonic() + wait_s
    cmdline = b""
    while time.monotonic() < deadline:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
        if cmdline == expected:
            break
        if not process_alive(pid):
            break
        time.sleep(0.01)
    if cmdline != expected:
        raise RuntimeError("service process command differs from the launched command")
    session_id = os.getsid(pid)
    process_group_id = os.getpgid(pid)
    if session_id != pid or process_group_id != pid:
        raise RuntimeError("resident service does not own an isolated process group")
    return {
        "schema": "ora.resident-service-process-identity.v1",
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="utf-8",
        ).strip(),
        "proc_start_ticks": _proc_start_ticks(pid),
        "session_id": session_id,
        "process_group_id": process_group_id,
        "cmdline_sha256": hashlib.sha256(cmdline).hexdigest(),
        "executable": str(Path(f"/proc/{pid}/exe").resolve()),
    }


#: ``service_process_identity_state`` results.
IDENTITY_EXITED, IDENTITY_MATCHES = "exited", "matches"
IDENTITY_UNINSPECTABLE, IDENTITY_DRIFTED = "uninspectable", "drifted"


def service_process_identity_state(record: Mapping[str, Any]) -> str:
    """``exited``, ``matches``, ``uninspectable`` or ``drifted`` for ``record``'s process.

    ``uninspectable``: the PID is alive but this process may not read its
    ``/proc/<pid>/exe`` (permission denied). That happens when the service was
    started from another container, sandbox or user namespace; it says nothing
    about whether the PID is still our service, so it is never treated as a
    match, and it is reported separately from real drift (PID reuse, another
    command line or boot).
    """
    pid = record.get("pid")
    command = record.get("command")
    saved = record.get("process_identity")
    if not process_alive(pid):
        return IDENTITY_EXITED
    if (
        not isinstance(command, list)
        or not all(isinstance(item, str) for item in command)
        or not isinstance(saved, Mapping)
    ):
        return IDENTITY_DRIFTED
    try:
        current = capture_service_process_identity(pid, command, wait_s=0.05)
    except PermissionError:
        return IDENTITY_UNINSPECTABLE
    except (OSError, RuntimeError, ValueError):
        return IDENTITY_DRIFTED
    return IDENTITY_MATCHES if dict(saved) == current else IDENTITY_DRIFTED


def service_process_identity_matches(record: Mapping[str, Any]) -> bool:
    return service_process_identity_state(record) == IDENTITY_MATCHES


def uninspectable_message(record: Mapping[str, Any]) -> str:
    """What an operator does about a live service this process cannot inspect."""
    identity = record.get("process_identity") or {}
    return (
        f"resident service pid {record.get('pid')} is alive but this process cannot inspect it "
        "(permission denied on /proc/<pid>/exe: it was started from another container, sandbox or "
        "user namespace). Run model-services stop from the environment that started it; or confirm "
        f"/proc/{record.get('pid')}/cmdline is the recorded command and stop process group "
        f"{identity.get('process_group_id', record.get('pid'))} yourself (kill -TERM -- -<pgid>)"
    )


def terminate_service_process(
    record: Mapping[str, Any], *, timeout_s: float = 15.0,
) -> bool:
    """Stop only the exact isolated process instance bound by ``record``."""

    state = service_process_identity_state(record)
    if state == IDENTITY_EXITED:
        return True
    if state == IDENTITY_UNINSPECTABLE:
        raise RuntimeError(uninspectable_message(record))
    if state != IDENTITY_MATCHES:
        raise RuntimeError("refusing to stop a resident service with drifted identity "
                           "(the PID now belongs to another process or command)")
    pid = record["pid"]
    identity = record["process_identity"]
    process_group_id = identity["process_group_id"]
    os.killpg(process_group_id, signal.SIGTERM)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and process_alive(pid):
        time.sleep(0.1)
    if process_alive(pid):
        if not service_process_identity_matches(record):
            raise RuntimeError(
                "resident service identity drifted before forced termination"
            )
        os.killpg(process_group_id, signal.SIGKILL)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and process_alive(pid):
            time.sleep(0.05)
    return not process_alive(pid)


__all__ = [
    "IDENTITY_DRIFTED",
    "IDENTITY_EXITED",
    "IDENTITY_MATCHES",
    "IDENTITY_UNINSPECTABLE",
    "SERVICE_ENV_ALLOWLIST",
    "capture_service_process_identity",
    "isolated_service_environment",
    "process_alive",
    "service_process_identity_matches",
    "service_process_identity_state",
    "uninspectable_message",
    "terminate_service_process",
]
