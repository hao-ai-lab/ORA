"""Atomic resource claims with GPU probes outside the shared registry lock."""
from __future__ import annotations
import fcntl
import json
import os
import secrets
import socket
import time
from pathlib import Path
from typing import Any, Callable, Mapping
from .resources import ExecutionResourceError, resource_lease_key
from .resource_admission import _resource_refusal, _visible_gpu_ids
from .resource_inventory import lease_owner_pids


def claim_resource_lease(
    binding: Mapping[str, Any], *, run_id: str, plan_sha256: str, launcher_pid: int,
    lease_root: Path, lock_path: Path, schema: str,
    active_leases: Callable[[], list[Mapping[str, Any]]],
    memory_inventory: Callable[..., Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Recheck active claims under the lock after one bounded device probe.

    The probe is a snapshot, never an admission decision. Only keys still in
    the locked active set receive memory credit, and a new neighbour is charged
    its entire declared floor. Competing claims cannot bypass each other.
    """
    lease_root.mkdir(parents=True, exist_ok=True)
    inventory = None
    probe_error = None
    probed = False
    while True:
        with lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            active = active_leases()
            shared = bool(_visible_gpu_ids(binding)) and any(
                _visible_gpu_ids(lease) & _visible_gpu_ids(binding) for lease in active)
            if probed or not shared:
                refusal = _resource_refusal(binding, active, inventory=inventory,
                                            probe_error=probe_error)
                if refusal is not None:
                    raise RuntimeError(refusal)
                return _write_lease(binding, run_id=run_id, plan_sha256=plan_sha256,
                                    launcher_pid=launcher_pid, lease_root=lease_root,
                                    schema=schema)
        probed = True
        try:
            inventory = memory_inventory(lease_owner_pids(lease_root))
        except ExecutionResourceError as exc:
            probe_error = str(exc)


def _write_lease(binding: Mapping[str, Any], *, run_id: str, plan_sha256: str,
                 launcher_pid: int, lease_root: Path, schema: str) -> dict[str, Any]:
    """Write one admitted claim; caller holds the registry lock."""
    path = lease_root / (resource_lease_key(binding) + ".json")
    lease = {
        "schema": schema, "run_id": run_id, "plan_sha256": plan_sha256,
        "resource_class": binding["resource_class"], "slot_id": binding["slot_id"],
        "profile_id": binding["profile_id"], "host_id": binding.get("host_id"),
        "runtime_kind": binding.get("runtime_kind"),
        "worker_environment": dict(binding.get("worker_environment", {})),
        "minimum_free_memory_mib": binding.get("minimum_free_memory_mib"),
        "profile_sha256": binding.get("profile_sha256"),
        "claim_token": secrets.token_hex(16), "launcher_pid": int(launcher_pid),
        "supervisor_pid": None, "created_unix_s": time.time(),
        # The launcher and its supervisor run here. Only this node can tell
        # whether their PIDs are alive; see lease_off_node.
        "node": socket.gethostname(),
    }
    # Publish the claim whole. A reader outside the registry lock (preflight,
    # status, a queue waiting on its runs) prunes a lease it cannot parse, and
    # once pruned a claim between its O_EXCL create and its JSON write, so the
    # launcher found its own lease gone. The
    # temporary name is not *.json; the hard link refuses an existing claim
    # exactly as O_EXCL did.
    temporary = lease_root / f".{path.name}.{lease['claim_token']}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(lease, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {**lease, "lease_path": str(path)}


def lease_off_node(lease: Mapping[str, Any]) -> str | None:
    """The node a lease was claimed on, when that is not this node.

    A PID is only meaningful on the node that owns it: a lease scan from a login
    or scan node once judged every live cluster lease stale by a local PID and
    deleted it, and a queue then abandoned 22 running episodes.
    """
    node = lease.get("node")
    return node if isinstance(node, str) and node and node != socket.gethostname() else None
