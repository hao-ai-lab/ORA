"""The generated ``policy.lock``: one sha256 per tracked policy asset.

``policy.lock`` at the repository root is the content inventory of every
policy asset the Harness may load by path. It is generated, never edited by
hand, and written in ``sha256sum`` text format (``<sha256>  <path>``, sorted
by path, LF line endings), so ``sha256sum -c policy.lock`` verifies a checkout
without the project installed.

``ora lock`` rewrites it from the tracked tree; ``ora lock
--check`` reports drift and exits nonzero. The check runs in CI, not
at run time: a RunPlan records the lock's
own sha256 (``policy_identity.policy_lock_sha256``) beside the source commit,
and loaders keep hashing the bytes they read.

Coverage is every git-tracked file matched by ``COVERAGE``: all of
``ora/policy/`` plus the path-loaded assets that live elsewhere,
each entry naming the loader that reads it. Only ``git ls-files`` entries
count, so untracked scratch never enters the lock; a new asset joins it once
it is added to the index. ``UNLOADED_REFERENCES`` records path references in
covered assets that no loader reads, which is why they are not covered.
External identities (task-contract source dependencies, model checkpoints,
simulator revisions) stay inline in their assets and are out of scope.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

# One definition of the file and its line format, shared with the loaders'
# membership check (agent/composition/locked_assets.py).
from ora.agent.composition.locked_assets import LOCK_LINE as _LINE, LOCK_PATH


class PolicyLockError(RuntimeError):
    """The lock cannot be rendered or read."""


@dataclass(frozen=True)
class Coverage:
    """One git pathspec of locked assets and the loader that reads them by path."""

    pathspec: str
    loader: str
    reason: str


COVERAGE: tuple[Coverage, ...] = (
    Coverage(
        "ora/policy",
        "every policy loader: ToolSurface manifests and components, prompt "
        "bundles and layers, context profiles, embodiment descriptors, cards "
        "and wording reviews, baseline catalogs, capability manifests, "
        "task contracts and runtime/control/motion profiles",
        "the policy asset tree",
    ),
    Coverage(
        ":(glob)ora/environment/benchmarks/*.json",
        "agent/composition/robodojo_layout_compatibility.py and "
        "environment/benchmarks/robodojo_layouts.py (LAYOUT_CATALOG_PATH)",
        "the RoboDojo layout-pool catalog that runtime.layout_schedule binds "
        "by sha256",
    ),
)


@dataclass(frozen=True)
class UnloadedReference:
    """A path prefix that covered assets name but no loader reads."""

    prefix: str
    reason: str


UNLOADED_REFERENCES: tuple[UnloadedReference, ...] = ()


@dataclass(frozen=True)
class LockDrift:
    """Differences between ``policy.lock`` and the tracked tree.

    ``missing``: covered tree assets the lock does not list (added assets).
    ``extra``: lock entries whose asset is no longer in the tree (deleted).
    ``changed``: assets whose bytes no longer match their lock entry.
    ``errors``: the lock is absent, malformed or not in canonical order.
    """

    missing: tuple[str, ...] = ()
    extra: tuple[str, ...] = ()
    changed: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def clean(self) -> bool:
        return not (self.missing or self.extra or self.changed or self.errors)

    def lines(self) -> list[str]:
        return [
            *self.errors,
            *(f"changed: {path}" for path in self.changed),
            *(f"missing from {LOCK_PATH}: {path}" for path in self.missing),
            *(f"extra in {LOCK_PATH}: {path}" for path in self.extra),
        ]


def covered_paths(root: Path, *, coverage: Sequence[Coverage] = COVERAGE) -> list[str]:
    """Return the sorted tracked asset paths that exist in the working tree."""

    root = Path(root)
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "--", *(item.pathspec for item in coverage)],
        capture_output=True, text=True, timeout=60, check=False,
    )
    if result.returncode:
        raise PolicyLockError(result.stderr.strip() or "git ls-files failed")
    paths = {item for item in result.stdout.split("\0") if item}
    return sorted(path for path in paths if (root / path).is_file() and not path.endswith(".py"))


def tree_digests(root: Path) -> dict[str, str]:
    """The sha256 of each covered asset's working-tree bytes."""

    digests: dict[str, str] = {}
    for path in covered_paths(root):
        if "\n" in path or "\\" in path or path != path.strip():
            raise PolicyLockError(f"asset path cannot be written in sha256sum format: {path!r}")
        digests[path] = hashlib.sha256((Path(root) / path).read_bytes()).hexdigest()
    return digests


def format_lock(digests: Mapping[str, str]) -> str:
    return "".join(f"{digests[path]}  {path}\n" for path in sorted(digests))


def render_lock(root: Path) -> str:
    """The canonical lock text for the tracked tree at ``root``."""

    return format_lock(tree_digests(root))


def write_lock(root: Path) -> Path:
    path = Path(root) / LOCK_PATH
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(render_lock(root).encode("utf-8"))
    temporary.replace(path)
    return path


def parse_lock(text: str) -> tuple[dict[str, str], list[str]]:
    """Parse lock text into ``{path: sha256}`` plus format errors."""

    errors: list[str] = []
    entries: dict[str, str] = {}
    if "\r" in text:
        errors.append(f"{LOCK_PATH} must use LF line endings")
    for number, line in enumerate(text.split("\n")[:-1], start=1):
        match = _LINE.match(line.rstrip("\r"))
        if match is None:
            errors.append(f"{LOCK_PATH}:{number}: malformed line")
        elif match["path"] in entries:
            errors.append(f"{LOCK_PATH}:{number}: duplicate path {match['path']}")
        else:
            entries[match["path"]] = match["sha256"]
    if text and not text.endswith("\n"):
        errors.append(f"{LOCK_PATH} must end with a newline")
    if not errors and text != format_lock(entries):
        errors.append(f"{LOCK_PATH} is not sorted by path")
    return entries, errors


def check_lock(root: Path) -> LockDrift:
    """Compare the committed lock with the tracked tree."""

    path = Path(root) / LOCK_PATH
    if not path.is_file():
        return LockDrift(errors=(f"{LOCK_PATH} is absent; run `ora lock`",))
    try:
        text = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        return LockDrift(errors=(f"{LOCK_PATH} is not UTF-8",))
    locked, errors = parse_lock(text)
    tree = tree_digests(root)
    return LockDrift(
        missing=tuple(sorted(set(tree) - set(locked))),
        extra=tuple(sorted(set(locked) - set(tree))),
        changed=tuple(sorted(item for item in set(tree) & set(locked) if tree[item] != locked[item])),
        errors=tuple(errors),
    )


def lock_sha256(root: Path) -> str | None:
    """The sha256 of the ``policy.lock`` bytes at ``root``, or None when absent."""

    path = Path(root) / LOCK_PATH
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command_lock(args: argparse.Namespace, root: Path) -> int:
    """``ora lock [--check] [--strip-pins]``."""

    if getattr(args, "strip_pins", False):
        from .policy_pins import command_strip

        return command_strip(root, check=getattr(args, "check", False), write_lock=write_lock)
    if not getattr(args, "check", False):
        path = write_lock(root)
        print(f"wrote {path.name}: {len(covered_paths(root))} assets, sha256 {lock_sha256(root)}")
        return 0
    drift = check_lock(root)
    if drift.clean:
        print(f"{LOCK_PATH} is current: sha256 {lock_sha256(root)}")
        return 0
    for line in drift.lines():
        print(line, file=sys.stderr)
    print(f"{LOCK_PATH} is stale; run `./scripts/ora lock` and commit it", file=sys.stderr)
    return 1
