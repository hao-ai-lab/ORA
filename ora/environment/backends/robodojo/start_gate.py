"""Node-level start gate and readback cache for RoboDojo Vulkan initialization (RDJ-10).

Every RoboDojo start used to run the Vulkan compatibility readback twice (the
launcher's preflight and the worker's bootstrap), each creating two Vulkan
instances that enumerate every GPU of the node, and then initialized Kit/RTX.
When several sessions started together on a loaded node, those driver-level
initializations queued behind each other: EXP11A/motion-ab on a shared simulator host
(2026-10-07) lost 33 starts before the first model call to 120 s readback and
15 s ``nvidia-smi`` timeouts, while one readback alone took 30 s.

Two node-local mechanisms remove that contention without changing what is
measured:

- ``gate(name, slots)`` admits at most ``slots`` holders at once through
  ``fcntl`` locks on files under the node-local runtime directory. A holder
  that dies releases its lock with the process.
- ``cached(key, produce)`` keeps one readback per node and boot: the key binds
  the kernel boot id, the NVIDIA driver version file and the caller's inputs
  (the installation fingerprints), so a reboot, a driver change or a changed
  installation measures again. Reads are lock-free; a miss measures inside the
  gate and publishes atomically.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

SCHEMA = "ora.robodojo-start-gate.v1"
#: Concurrent Vulkan/Kit initializations admitted per node.
INIT_SLOTS = 2
#: Readback holders per node: the readback enumerates every GPU, so one at a time.
READBACK_SLOTS = 1
POLL_S = 0.5


def root() -> Path:
    """Node-local, per-user directory (``/dev/shm`` is cleared on reboot)."""
    base = Path("/dev/shm") if Path("/dev/shm").is_dir() else Path("/tmp")
    path = base / f"ora-robodojo-{os.getuid()}"
    path.mkdir(mode=0o700, exist_ok=True)
    return path


def node_identity() -> dict[str, str]:
    """What a cached readback is valid for: this boot and this driver."""
    boot = Path("/proc/sys/kernel/random/boot_id")
    driver = Path("/proc/driver/nvidia/version")
    return {
        "boot_id": boot.read_text().strip() if boot.is_file() else "unknown",
        "driver_version_sha256": hashlib.sha256(driver.read_bytes()).hexdigest() if driver.is_file() else "unknown",
    }


@contextlib.contextmanager
def gate(name: str, slots: int, *, timeout_s: float, directory: Path | None = None) -> Iterator[dict[str, Any]]:
    """Hold one of ``slots`` node-wide locks named ``name``; the record names the slot and the wait."""
    if slots < 1:
        raise ValueError("a start gate needs at least one slot")
    folder = directory or root()
    started = time.monotonic()
    handles = [open(folder / f"{name}.slot{index}.lock", "a+") for index in range(slots)]
    held = None
    try:
        while held is None:
            for index, handle in enumerate(handles):
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                held = index
                break
            if held is None:
                if time.monotonic() - started >= timeout_s:
                    raise TimeoutError(f"start gate {name} stayed full for {timeout_s:.0f} s")
                time.sleep(POLL_S)
        yield {"schema": SCHEMA, "gate": name, "slot": held, "slots": slots,
               "waited_s": round(time.monotonic() - started, 3)}
    finally:
        for handle in handles:
            handle.close()  # closing releases a held flock


def cache_key(inputs: Mapping[str, Any]) -> str:
    material = json.dumps({"node": node_identity(), "inputs": inputs}, sort_keys=True)
    return hashlib.sha256(material.encode()).hexdigest()


def cached(name: str, inputs: Mapping[str, Any], produce: Callable[[], dict[str, Any]], *,
           timeout_s: float, directory: Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """The value for ``inputs`` on this node and boot, measured once; returns (value, provenance)."""
    folder = directory or root()
    key = cache_key(inputs)
    path = folder / f"{name}-{key}.json"

    def read() -> dict[str, Any] | None:
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            return None
        return record if record.get("key") == key and isinstance(record.get("value"), dict) else None

    record = read()
    if record is not None:
        return record["value"], {"schema": SCHEMA, "cache": "hit", "key": key,
                                 "measured_unix_s": record["measured_unix_s"], "waited_s": 0.0}
    with gate(name, READBACK_SLOTS, timeout_s=timeout_s, directory=folder) as held:
        record = read()  # another holder may have measured while this one waited
        if record is not None:
            return record["value"], {"schema": SCHEMA, "cache": "hit_after_wait", "key": key,
                                     "measured_unix_s": record["measured_unix_s"], "waited_s": held["waited_s"]}
        value = produce()
        record = {"key": key, "node": node_identity(), "measured_unix_s": time.time(), "value": value}
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(record, sort_keys=True))
        temporary.replace(path)
        return value, {"schema": SCHEMA, "cache": "measured", "key": key,
                       "measured_unix_s": record["measured_unix_s"], "waited_s": held["waited_s"]}


__all__ = ["INIT_SLOTS", "READBACK_SLOTS", "SCHEMA", "cache_key", "cached", "gate", "node_identity", "root"]
