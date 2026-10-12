"""Bounded GPU inventory and source-lease process attribution; no admission writes."""
from __future__ import annotations
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping
from .resources import ExecutionResourceError
from .resource_admission import _lease_key
# nvidia-smi slows with the number of CUDA processes on a node (about 13 s with
# ~100 MPS seed-scan workers on an H200 node); the bound only stops a wedged
# driver from hanging a claim.
GPU_MEMORY_PROBE_TIMEOUT_S = 60


def lease_owner_pids(lease_root: Path) -> dict[int, str]:
    """Map every recorded lease-owning pid to its lease key, without the lock.

    Read before the registry lock so the claim-time device probe can attribute
    GPU memory to our own leased process trees. It deliberately never deletes
    a stale record and never decides admission: pricing happens later under
    the lock against the authoritative active set, and an attributed key that
    is absent there is simply not credited.
    """

    owners: dict[int, str] = {}
    if not lease_root.is_dir():
        return owners
    for path in sorted(lease_root.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(record, Mapping):
            continue
        key = _lease_key(record)
        for name in ("supervisor_pid", "launcher_pid"):
            pid = record.get(name)
            if isinstance(pid, int) and not isinstance(pid, bool) and pid > 1:
                owners.setdefault(pid, key)
    return owners


def _process_ancestry(pid: int, limit: int = 64) -> list[int]:
    """Return ``pid`` and its ancestors, nearest first, read from ``/proc``."""

    chain: list[int] = []
    current = pid
    while current > 1 and len(chain) < limit:
        chain.append(current)
        try:
            stat = Path(f"/proc/{current}/stat").read_text(encoding="utf-8")
        except (OSError, ValueError):
            break
        # The command name sits in parentheses and may itself contain spaces
        # and ')', so the parent pid is the second field after the last one.
        fields = stat.rpartition(")")[2].split()
        if len(fields) < 2 or not fields[1].isdigit():
            break
        parent = int(fields[1])
        if parent in chain:
            break
        current = parent
    return chain


def _nvidia_smi_rows(query: str) -> list[list[str]]:
    """Return one nvidia-smi CSV query as trimmed fields, or refuse."""

    nvidia = shutil.which("nvidia-smi")
    if nvidia is None:
        raise ExecutionResourceError("nvidia-smi is unavailable")
    try:
        completed = subprocess.run(
            [nvidia, query, "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
            timeout=GPU_MEMORY_PROBE_TIMEOUT_S, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        # Operator-visible refusals never echo driver output: a wedged
        # nvidia-smi writes unbounded text that would land in run evidence.
        raise ExecutionResourceError(
            f"nvidia-smi did not answer ({type(exc).__name__})"
        ) from exc
    if completed.returncode:
        raise ExecutionResourceError(
            f"nvidia-smi exited {completed.returncode}"
        )
    return [
        [item.strip() for item in line.split(",")]
        for line in completed.stdout.splitlines() if line.strip()
    ]


def _gpu_memory_inventory(
    owner_pids: Mapping[int, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return per-device memory and the part of it our own leases hold.

    ``free_mib`` and ``total_mib`` are read exactly as preflight reads them.
    ``leased_mib`` maps a lease key to the device memory currently held by
    compute processes descended from that lease's launcher or supervisor pid.

    Every other byte in use is deliberately left unattributed and is therefore
    charged in full against a claim: a foreign job, another user's process, a
    resident service that holds no lease, driver and display reserve, and any
    of our own workers whose process tree no longer reaches its lease. Failing
    to recognise our own usage costs capacity; failing to see foreign usage
    would cost a run, so the attribution only ever credits what it can prove.
    """

    owners = dict(owner_pids or {})
    devices: dict[str, dict[str, Any]] = {}
    index_of_uuid: dict[str, str] = {}
    for fields in _nvidia_smi_rows(
        "--query-gpu=index,uuid,memory.free,memory.total",
    ):
        if len(fields) != 4 or not all(
            fields[position].isdigit() for position in (0, 2, 3)
        ):
            continue
        devices[fields[0]] = {
            "free_mib": int(fields[2]),
            "total_mib": int(fields[3]),
            "leased_mib": {},
        }
        index_of_uuid[fields[1]] = fields[0]
    if not devices:
        raise ExecutionResourceError("nvidia-smi reported no GPU memory")
    for fields in _nvidia_smi_rows(
        "--query-compute-apps=gpu_uuid,pid,used_memory",
    ):
        if len(fields) != 3 or not (fields[1].isdigit() and fields[2].isdigit()):
            continue
        device = index_of_uuid.get(fields[0])
        if device is None:
            continue
        key = next(
            (
                owners[ancestor]
                for ancestor in _process_ancestry(int(fields[1]))
                if ancestor in owners
            ),
            None,
        )
        if key is None:
            continue
        leased = devices[device]["leased_mib"]
        leased[key] = leased.get(key, 0) + int(fields[2])
    return devices
