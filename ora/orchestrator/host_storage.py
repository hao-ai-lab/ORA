"""Host storage preflight and launch-time scratch staging.

Every RunPlan receives the same four checks: evidence-workspace capacity,
TMPDIR scratch capacity (plus the RPC socket budget and a residual-directory
census), system-partition capacity where the run writes outside TMPDIR, and a
census of carb shared-memory segments. Each reports what it measured; a
failure blocks preflight, which precedes the lease and any cohort claim.

At launch, ``stage_launch`` creates the run's scratch directory and unlinks
this user's dead-PID carb segments, both before the lease. The supervisor
removes its scratch directory when it exits.

What this does not do: another user's stale carb segment cannot be removed,
so a Kit worker can still receive a colliding PID (run 683e36fb73af collided
with another user's segment). That case needs a worker-side guard.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Callable, Mapping

from ora.environment.rpc import MAX_UNIX_SOCKET_PATH_BYTES

from . import paths, resources, run_scratch, shm_segments

GIB = 1024 ** 3

#: Evidence workspace. Recorded RoboDojo runs wrote 1.3-5.3 GB each and one cohort
#: 47 GB; 20 GiB holds about four of the largest runs in flight at once.
DEFAULT_MINIMUM_WORKSPACE_FREE_GIB = 20
#: TMPDIR filesystem. Harness evidence writers stage beside their targets
#: (mkstemp dir=path.parent), so TMPDIR carries only library and simulator
#: scratch; 2 GiB is a margin floor against the 0-byte state that voided
#: cells, not a measured peak.
DEFAULT_MINIMUM_TMP_FREE_GIB = 2
#: "/", "/var/tmp" and "/tmp". The OptiX cache defaults to
#: /var/tmp/OptixCache_<user>, else /tmp (optix_host.h text embedded in
#: omni.hydra.rtx libnrend.so); it measured 21 MB for one user on a RoboDojo
#: host. 1 GiB is about fifty times that.
DEFAULT_MINIMUM_SYSTEM_FREE_GIB = 1
SYSTEM_PATHS = ("/", "/var/tmp", "/tmp")

#: Runs that write outside TMPDIR and therefore need the system check: Kit
#: runtimes (OptiX and Kit caches).
KIT_RUNTIME_KINDS = frozenset({"robodojo"})
#: What each Kit runtime still writes outside TMPDIR, as far as the source shows.
_KIT_SYSTEM_REASONS = {
    "robodojo": ("the bootstrap binds TMPDIR and the OptiX, XDG, CUDA and GL caches under its "
                 "runtime directory, but Kit's crash reporter keeps local dumps with no dump "
                 "directory set here (Kit default, not verified off /), and nothing that runs "
                 "before the bootstrap is redirected"),
}

EXECUTION_HOST_ENV = "ORA_EXECUTION_HOST"
HOST_STORAGE_KEY = "runtime_storage"
_FLOOR_KEYS = {
    "minimum_workspace_free_gib": DEFAULT_MINIMUM_WORKSPACE_FREE_GIB,
    "minimum_tmp_free_gib": DEFAULT_MINIMUM_TMP_FREE_GIB,
    "minimum_system_free_gib": DEFAULT_MINIMUM_SYSTEM_FREE_GIB,
}
# In-process Synthetic writes bounded JSON/image evidence, without simulator caches.
_SYNTHETIC_FLOORS = {"minimum_workspace_free_gib": 1.0, "minimum_tmp_free_gib": 0.25}
Check = Callable[[str, bool, Any], dict[str, Any]]


class StorageConfigError(RuntimeError):
    """The execution host or its runtime_storage block is malformed."""


class StorageProbeError(OSError):
    """A measured path cannot be attributed to a filesystem."""


def _free_bytes(path: Path) -> int:
    return int(shutil.disk_usage(path).free)


def _device(path: Path) -> int:
    return os.stat(path).st_dev


def _writable(path: Path) -> bool:
    return os.access(path, os.W_OK | os.X_OK)


def _execution_host(plan: Mapping[str, Any]) -> tuple[str | None, str]:
    """The host whose profile applies: where this process runs, else the plan's."""

    marker = os.environ.get(EXECUTION_HOST_ENV)
    if marker:
        return marker, "execution_marker"
    runtime = plan.get("runtime") if isinstance(plan.get("runtime"), Mapping) else {}
    required = runtime.get("required_host")
    if isinstance(required, str) and required:
        return required, "required_host"
    return None, "none"


def _host_profile(host_id: str) -> Mapping[str, Any] | None:
    try:
        return resources.load_host_profile(host_id)
    except resources.ExecutionResourceError as exc:
        raise StorageConfigError(str(exc)) from exc


def _floor(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise StorageConfigError(f"{HOST_STORAGE_KEY}.{name} must be a non-negative number")
    return value


def storage_config(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Floors and scratch base with their provenance, or StorageConfigError."""

    host, host_source = _execution_host(plan)
    profile = _host_profile(host) if host else None
    raw = profile.get(HOST_STORAGE_KEY) if profile else None
    if raw is not None and not isinstance(raw, Mapping):
        raise StorageConfigError(f"{HOST_STORAGE_KEY} must be a mapping")
    block = dict(raw or {})
    unknown = set(block) - {"tmp_root", *_FLOOR_KEYS}
    if unknown:
        raise StorageConfigError(f"{HOST_STORAGE_KEY} has unknown keys: {sorted(unknown)}")
    source = f"host_profile:{host}"
    synthetic = plan.get("runtime", {}).get("kind") == "synthetic"
    floors = {}
    for name, default in _FLOOR_KEYS.items():
        if name in block:
            floors[name] = (_floor(block[name], name), source)
        elif synthetic and name in _SYNTHETIC_FLOORS:
            floors[name] = (_SYNTHETIC_FLOORS[name], "synthetic_default")
        else:
            floors[name] = (default, "default")
    tmp_root = block.get("tmp_root")
    if tmp_root is None:
        base, base_source = run_scratch.default_base(), "workspace_default"
    else:
        if not isinstance(tmp_root, str) or "\x00" in tmp_root:
            raise StorageConfigError(f"{HOST_STORAGE_KEY}.tmp_root must be a path string")
        base, base_source = Path(tmp_root).expanduser(), source
        if not base.is_absolute() or base == Path(base.anchor):
            raise StorageConfigError(f"{HOST_STORAGE_KEY}.tmp_root must be an absolute non-root path")
    return {"host_id": host, "host_source": host_source, "floors": floors,
            "tmp_base": base.resolve(), "tmp_source": base_source}


def system_partition_required(plan: Mapping[str, Any]) -> tuple[bool, str]:
    runtime = plan.get("runtime") if isinstance(plan.get("runtime"), Mapping) else {}
    policy = plan.get("policy") if isinstance(plan.get("policy"), Mapping) else {}
    if runtime.get("kind") in KIT_RUNTIME_KINDS:
        return True, _KIT_SYSTEM_REASONS[str(runtime.get("kind"))]
    return False, "this run writes only below TMPDIR and the evidence workspace"


def _existing_ancestor(path: Path) -> Path:
    """The nearest existing ancestor; a dangling symlink is an error, not '/'."""

    cursor = path
    while True:
        if cursor.exists():
            return cursor
        if os.path.lexists(cursor):
            raise StorageProbeError(f"{cursor} is a dangling symlink")
        if cursor == cursor.parent:
            raise StorageProbeError(f"no existing ancestor for {path}")
        cursor = cursor.parent


def _capacity(path: Path, floor_gib: float) -> dict[str, Any]:
    probe = _existing_ancestor(path)
    free = _free_bytes(probe)
    return {
        "path": str(path), "filesystem_probe": str(probe),
        "free_bytes": free, "free_gib": round(free / GIB, 2),
        "minimum_free_gib": floor_gib, "disabled": floor_gib == 0,
        "passed": floor_gib == 0 or free >= floor_gib * GIB,
    }


def _measurement(measured: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in measured.items() if key != "passed"}


def _workspace_check(plan: Mapping[str, Any], config: Mapping[str, Any], check: Check) -> dict[str, Any]:
    floor, source = config["floors"]["minimum_workspace_free_gib"]
    try:
        resolved = paths.workspace_from_plan(plan)
        # Probe the logical outputs/<experiment> entry, so a dangling storage
        # link fails here instead of measuring whatever filesystem holds "/".
        measured = _capacity(paths.OUTPUTS_ROOT / str(plan["experiment"]), floor)
    except (OSError, RuntimeError) as exc:
        return check("evidence_workspace_capacity", False, {"error": str(exc), "floor_source": source})
    return check("evidence_workspace_capacity", measured["passed"], {
        **_measurement(measured), "resolved": str(resolved), "floor_source": source,
    })


def _tmpdir_check(config: Mapping[str, Any], check: Check, *, needs_rpc_socket: bool = True) -> dict[str, Any]:
    floor, source = config["floors"]["minimum_tmp_free_gib"]
    base = config["tmp_base"]
    socket_bytes = run_scratch.socket_fallback_bytes(base)
    detail = {
        "tmp_base": str(base), "per_run_directory": "<tmp_base>/<12 hex of sha256(run_id)>",
        "source": config["tmp_source"], "host_id": config["host_id"],
        "host_source": config["host_source"],
        "environment_variables": list(run_scratch.SCRATCH_VARIABLES),
        "rpc_socket_fallback_bytes": socket_bytes,
        "rpc_socket_limit_bytes": MAX_UNIX_SOCKET_PATH_BYTES, "rpc_socket_required": needs_rpc_socket, "floor_source": source,
        "scratch_runs": run_scratch.census(base),
    }
    try:
        measured = _capacity(base, floor)
    except OSError as exc:
        return check("runtime_tmpdir", False, {**detail, "error": str(exc)})
    writable = _writable(Path(measured["filesystem_probe"]))
    passed = measured["passed"] and writable and (not needs_rpc_socket or socket_bytes <= MAX_UNIX_SOCKET_PATH_BYTES)
    return check("runtime_tmpdir", passed, {**detail, **_measurement(measured), "writable": writable})


def _system_check(plan: Mapping[str, Any], config: Mapping[str, Any], check: Check) -> dict[str, Any]:
    required, reason = system_partition_required(plan)
    floor, source = config["floors"]["minimum_system_free_gib"]
    detail = {"required": required, "reason": reason, "minimum_free_gib": floor, "floor_source": source}
    if not required:
        return check("system_partition_capacity", True, detail)
    if floor == 0:
        return check("system_partition_capacity", True, {**detail, "disabled": True})
    try:
        groups: dict[int, list[str]] = {}
        for path in SYSTEM_PATHS:
            groups.setdefault(_device(_existing_ancestor(Path(path))), []).append(path)
        measured = [{"paths": group, **_capacity(Path(group[0]), floor)} for group in groups.values()]
    except OSError as exc:
        return check("system_partition_capacity", False, {**detail, "error": str(exc)})
    return check("system_partition_capacity", all(item["passed"] for item in measured), {
        **detail, "filesystems": measured,
    })


def _shm_check(check: Check) -> dict[str, Any]:
    counted = shm_segments.census(shm_segments.SHM_ROOT, uid=os.getuid())
    return check("shm_carb_segments", not counted["errors"], {
        **counted, "own_stale_removed_at": "launch, before the lease",
    })


def storage_checks(plan: Mapping[str, Any], *, check: Check) -> list[dict[str, Any]]:
    """Capacity, scratch and shared-memory checks for one RunPlan (no changes)."""

    try:
        config = storage_config(plan)
    except StorageConfigError as exc:
        return [check("host_storage_configuration", False, str(exc)), _shm_check(check)]
    return [
        _workspace_check(plan, config, check),
        _tmpdir_check(config, check, needs_rpc_socket=plan.get("runtime", {}).get("kind") != "synthetic"),
        _system_check(plan, config, check),
        _shm_check(check),
    ]


def stage_launch(plan: Mapping[str, Any], run_id: str) -> dict[str, Any]:
    """Create the run's scratch and sweep own dead-PID segments; before the lease."""

    config = storage_config(plan)
    scratch = run_scratch.stage(config["tmp_base"], run_id, source=config["tmp_source"],
                                host_id=config["host_id"])
    swept = shm_segments.sweep_own_stale(shm_segments.SHM_ROOT, uid=os.getuid())
    return {"runtime_scratch": scratch, "shm_sweep": swept}


def scratch_environment(staged: Mapping[str, Any]) -> dict[str, str]:
    return run_scratch.environment(staged["runtime_scratch"])


def discard_staged(staged: Mapping[str, Any]) -> bool:
    """Remove a staged scratch directory after a failed launch; segments stay swept."""
    return run_scratch.discard(staged["runtime_scratch"])


def bind_scratch_owner(staged: Mapping[str, Any], supervisor_pid: int) -> bool:
    return run_scratch.bind_owner(staged["runtime_scratch"], supervisor_pid)


def release_run_scratch(run_root: Path) -> bool:
    return run_scratch.release(run_root)
