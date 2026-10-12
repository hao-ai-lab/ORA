"""Per-run TMPDIR scratch outside every evidence tree the Harness scans.

The base defaults to ``<outputs root>/../.runtime-tmp``: a sibling of the
physical outputs root, so neither ``paths._output_run_candidates`` nor the
Viewer's ``**/runs/*/plan.json`` walk ever descends into it. Each run gets
``<base>/<12 hex of sha256(run_id)>``; the short key keeps the environment RPC
socket fallback (``<TMPDIR>/<16 hex>/worker.sock``) inside the AF_UNIX limit.

The launcher creates the run directory before the lease and records its owner;
the supervisor removes it when it exits. Preflight counts directories whose
owner process is gone (report only; nothing is deleted there).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping

from . import paths
from .shm_segments import pid_alive, pid_max

SCRATCH_DIRNAME = ".runtime-tmp"
SCRATCH_VARIABLES = ("TMPDIR", "TMP", "TEMP")
RUNTIME_SCRATCH_SCHEMA = "ora.runtime-scratch.v2"
OWNER_SCHEMA = "ora.runtime-scratch-owner.v1"
OWNER_FILE = ".ora-scratch-owner.json"
RUN_KEY_HEX = 12
#: rpc.py appends "/<16 hex>/worker.sock" to tempfile.gettempdir() when a run's
#: own transport path is too long for AF_UNIX; TMPDIR itself is <base>/<key>.
RPC_SOCKET_SUFFIX_BYTES = len("/" + "0" * RUN_KEY_HEX) + len("/" + "0" * 16 + "/worker.sock")


def default_base() -> Path:
    return paths.OUTPUTS_ROOT.parent / SCRATCH_DIRNAME


def run_key(run_id: str) -> str:
    return hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:RUN_KEY_HEX]


def socket_fallback_bytes(base: Path) -> int:
    return len(os.fsencode(str(base))) + RPC_SOCKET_SUFFIX_BYTES


def environment(record: Mapping[str, Any]) -> dict[str, str]:
    root = str(record["tmp_root"])
    return {name: root for name in SCRATCH_VARIABLES}


def _write_owner(directory: Path, owner: Mapping[str, Any]) -> None:
    temporary = directory / (OWNER_FILE + ".tmp")
    temporary.write_text(json.dumps(dict(owner), sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, directory / OWNER_FILE)


def _read_owner(directory: Path) -> dict[str, Any] | None:
    try:
        value = json.loads((directory / OWNER_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) and value.get("schema") == OWNER_SCHEMA else None


def stage(base: Path, run_id: str, *, source: str, host_id: str | None) -> dict[str, Any]:
    """Create this run's scratch directory and record the launcher as owner."""

    base.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory = base / run_key(run_id)
    directory.mkdir(mode=0o700, exist_ok=True)
    _write_owner(directory, {
        "schema": OWNER_SCHEMA, "run_id": run_id, "launcher_pid": os.getpid(),
        "supervisor_pid": None,
    })
    return {
        "schema": RUNTIME_SCRATCH_SCHEMA, "tmp_root": str(directory),
        "base": str(base), "run_key": directory.name, "source": source,
        "host_id": host_id, "environment_variables": list(SCRATCH_VARIABLES),
    }


def bind_owner(record: Mapping[str, Any], supervisor_pid: int) -> bool:
    """Name the supervisor as owner; best effort, the run is already spawned."""

    directory = Path(str(record["tmp_root"]))
    owner = _read_owner(directory)
    if owner is None:
        return False
    try:
        _write_owner(directory, {**owner, "supervisor_pid": int(supervisor_pid)})
    except OSError:
        return False
    return True


def discard(record: Mapping[str, Any]) -> bool:
    """Remove the directory ``stage`` returned, if its owner file still names it."""

    directory = Path(str(record["tmp_root"]))
    owner = _read_owner(directory)
    if owner is None or directory.name != run_key(str(owner.get("run_id"))):
        return False
    shutil.rmtree(directory, ignore_errors=True)
    return not directory.exists()


def release(run_root: Path) -> bool:
    """Remove the scratch directory this run's launch.json names, if it owns it.

    Silent by design: it runs after the landing seal, whose inventory includes
    the supervisor log.
    """

    try:
        launch = json.loads((run_root / "launch.json").read_text(encoding="utf-8"))
        record = launch["runtime_scratch"]
        directory = Path(str(record["tmp_root"]))
    except (OSError, ValueError, KeyError, TypeError):
        return False
    owner = _read_owner(directory)
    if (owner is None or owner.get("run_id") != launch.get("run_id")
            or directory.name != run_key(str(launch.get("run_id")))):
        return False
    shutil.rmtree(directory, ignore_errors=True)
    return not directory.exists()


def census(base: Path) -> dict[str, Any]:
    """Count run directories by owner liveness; report only."""

    try:
        entries = sorted(entry for entry in base.iterdir() if entry.is_dir())
    except FileNotFoundError:
        return {"base": str(base), "present": False}
    except OSError as exc:
        return {"base": str(base), "present": True, "error": str(exc)}
    limit = pid_max()
    live = residual = unrecognized = 0
    for entry in entries:
        owner = _read_owner(entry)
        pid = (owner.get("supervisor_pid") or owner.get("launcher_pid")) if owner else None
        if not isinstance(pid, int) or isinstance(pid, bool):
            unrecognized += 1
        elif pid_alive(pid, limit=limit):
            live += 1
        else:
            residual += 1
    return {"base": str(base), "present": True, "live_runs": live,
            "residual_dead_owner": residual, "unrecognized": unrecognized}
