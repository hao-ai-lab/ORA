"""Resolve a path-only asset reference to the bytes and hash a RunPlan records.

Maintained policy assets name each other by path only. A ToolSurface component
is ``"../explicit_motion_v2/robotwin-visual_help.json"``, a context profile's
compiled descriptor is ``"ora/policy/embodiments/..."``; neither
carries a hand-written sha256. Integrity comes from ``policy.lock``, which CI
checks against the tree, and every loader hashes the bytes it actually reads
here, so the RunPlan still records ``{path, sha256}``: computed, never copied.

A reference resolves only when it

- is a non-empty path string (an inline ``{path, sha256}`` or ``{ref, sha256}``
  pin is refused, never verified and never ignored);
- stays inside the source tree it is resolved against;
- is listed in that tree's ``policy.lock``, so an untracked or uncovered file
  cannot be loaded as policy.
"""

from __future__ import annotations

import functools
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

#: The lock file, at the source-tree root, in ``sha256sum`` format.
LOCK_PATH = "policy.lock"
LOCK_LINE = re.compile(r"^(?P<sha256>[0-9a-f]{64})  (?P<path>[^\s].*)$")


@dataclass(frozen=True)
class LockedAsset:
    """One asset as a loader read it."""

    relative: str
    path: Path
    data: bytes
    sha256: str

    def record(self) -> dict[str, str]:
        """The ``{path, sha256}`` binding a RunPlan carries."""
        return {"path": self.relative, "sha256": self.sha256}


@functools.lru_cache(maxsize=8)
def _locked_paths(lock: str, _mtime_ns: int, _size: int) -> frozenset[str]:
    paths = set()
    for line in Path(lock).read_text(encoding="utf-8").splitlines():
        match = LOCK_LINE.match(line)
        if match is not None:
            paths.add(match["path"])
    return frozenset(paths)


def locked_paths(root: Path) -> frozenset[str] | None:
    """The asset paths ``root/policy.lock`` lists, or None when there is no lock."""
    lock = Path(root) / LOCK_PATH
    try:
        stat = lock.stat()
    except FileNotFoundError:
        return None
    return _locked_paths(str(lock.resolve()), stat.st_mtime_ns, stat.st_size)


def locked_asset(
    root: Path, reference: object, *, label: str,
    error: type[Exception] = ValueError, base: Path | None = None,
) -> LockedAsset:
    """Read the asset ``reference`` names and hash its bytes.

    ``reference`` is relative to ``base`` (default ``root``); ToolSurface
    components are relative to their manifest's directory, every other
    reference to the source-tree root. ``error`` is the caller's error type.
    """
    if isinstance(reference, dict):
        raise error(
            f"{label} must be a path string: inline sha256 pins are not accepted "
            f"(policy.lock carries asset integrity): {reference!r}"
        )
    if not isinstance(reference, str) or not reference.strip():
        raise error(f"{label} must be a non-empty path string")
    tree = Path(root).resolve()
    path = (Path(base) if base is not None else tree).joinpath(reference).resolve()
    if not path.is_relative_to(tree):
        raise error(f"{label} escapes the source tree: {reference}")
    relative = path.relative_to(tree).as_posix()
    listed = locked_paths(tree)
    if listed is None:
        raise error(f"{label} cannot be checked: {tree / LOCK_PATH} is absent; run ./scripts/ora lock")
    if relative not in listed:
        raise error(
            f"{label} is not in {LOCK_PATH}: {relative}; git add it and run ./scripts/ora lock"
        )
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise error(f"{label} is unavailable: {relative}") from exc
    return LockedAsset(relative=relative, path=path, data=data, sha256=hashlib.sha256(data).hexdigest())


__all__ = ["LOCK_LINE", "LOCK_PATH", "LockedAsset", "locked_asset", "locked_paths"]
