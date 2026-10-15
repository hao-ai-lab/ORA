"""Stale carb shared-memory segments in ``/dev/shm``: census and own sweep.

carb names its string-interning segment and the paired POSIX semaphore by the
creating PID: ``carb-RStringInternals-<pid>`` and
``sem.carb-RStringInternals-<pid>``. A segment outlives a crashed Kit process,
and a later Kit process that receives the same PID collides with it.

Preflight only counts (``census``). The launcher unlinks this user's
segments whose PID provably names no process (``sweep_own_stale``), after
confirmation and before the lease. Other users' segments cannot be unlinked
from the sticky ``/dev/shm`` and are only counted.

PID liveness is judged in this process's PID namespace. A Kit process in a
container with its own PID namespace but a shared ``/dev/shm`` would look dead
here; such a host must not rely on this sweep.
"""

from __future__ import annotations

import os
import re
import stat
from collections import Counter
from pathlib import Path
from typing import Any

SHM_ROOT = Path("/dev/shm")
#: Exactly the two carb names observed on a RoboDojo host (4,705
#: segments and 2,353 semaphores). Seven digits bound the PID at
#: PID_MAX_LIMIT; a longer suffix is never parsed, so a crafted name such as
#: ``carb-RStringInternals-99999999999`` cannot reach ``os.kill``. ASCII digits
#: only: ``\d`` would also admit other Unicode digits that ``int`` accepts.
CARB_SEGMENT = re.compile(r"^(?:sem\.)?carb-RStringInternals-([0-9]{1,7})$")
#: PID_MAX_LIMIT for 64-bit Linux (include/linux/threads.h: 4 * 1024 * 1024);
#: /proc/sys/kernel/pid_max reads 4194304 on a typical simulator host.
PID_MAX_LIMIT = 4 * 1024 * 1024
PID_MAX_PATH = Path("/proc/sys/kernel/pid_max")
#: Bound the names copied into evidence; counts stay exact.
MAX_RECORDED_NAMES = 50


def pid_max() -> int:
    try:
        value = int(PID_MAX_PATH.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return PID_MAX_LIMIT
    return value if 0 < value <= PID_MAX_LIMIT else PID_MAX_LIMIT


def pid_alive(pid: int, *, limit: int) -> bool:
    """Only a PID that provably names no process counts as dead.

    A value outside ``1..limit`` is not a PID this kernel can issue, so it is
    treated as alive (never unlinked). ``os.kill`` raises OverflowError, not
    OSError, above the C ``pid_t`` range; that is caught here as well.
    """

    if pid <= 0 or pid > limit:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (OSError, OverflowError, ValueError):
        return True
    return True


def segment_owner(path: Path) -> int | None:
    """The owning uid of a regular shm file; anything else is never unlinked."""

    info = os.lstat(path)
    return info.st_uid if stat.S_ISREG(info.st_mode) else None


def _classify(root: Path, uid: int) -> tuple[list[tuple[str, str, int]], list[str]]:
    names = sorted(os.listdir(root))
    limit = pid_max()
    classified: list[tuple[str, str, int]] = []
    errors: list[str] = []
    for name in names:
        match = CARB_SEGMENT.match(name)
        if match is None:
            continue
        pid = int(match.group(1))
        try:
            owner = segment_owner(root / name)
        except FileNotFoundError:
            continue
        except OSError as exc:
            errors.append(f"{name}: {exc}")
            continue
        side = "own" if owner is not None and owner == uid else "foreign"
        classified.append((name, f"{side}_{'live' if pid_alive(pid, limit=limit) else 'stale'}", pid))
    return classified, errors


def _scan(root: Path, uid: int) -> tuple[dict[str, Any], list[str]]:
    """The census record plus this user's stale names (all of them)."""

    base = {"root": str(root), "present": True, "errors": []}
    try:
        classified, errors = _classify(root, uid)
    except FileNotFoundError:
        return {**base, "present": False}, []
    except OSError as exc:
        return {**base, "errors": [f"{root}: {exc}"]}, []
    counts = Counter(kind for _, kind, _ in classified)
    own_stale = [name for name, kind, _ in classified if kind == "own_stale"]
    return {
        **base,
        "own_stale": counts["own_stale"],
        "own_stale_names": own_stale[:MAX_RECORDED_NAMES],
        "own_live": counts["own_live"],
        "foreign_stale": counts["foreign_stale"],
        "foreign_stale_pids": len({pid for _, kind, pid in classified if kind == "foreign_stale"}),
        "foreign_live": counts["foreign_live"],
        "errors": errors,
    }, own_stale


def census(root: Path, *, uid: int) -> dict[str, Any]:
    """Count carb segments by owner and PID liveness without changing any."""

    return _scan(root, uid)[0]


def _unlink(path: Path) -> str | None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        return None
    except OSError as exc:
        return f"{path.name}: {exc}"
    return None


def sweep_own_stale(root: Path, *, uid: int) -> dict[str, Any]:
    """Unlink this user's segments whose PID is gone; report what happened."""

    counted, stale = _scan(root, uid)
    failures = {name: error for name in stale if (error := _unlink(root / name))}
    removed = [name for name in stale if name not in failures]
    return {
        **counted,
        "removed": len(removed),
        "removed_names": removed[:MAX_RECORDED_NAMES],
        "errors": [*counted["errors"], *failures.values()],
    }
