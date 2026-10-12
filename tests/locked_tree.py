"""Give a scratch source tree the ``policy.lock`` its loaders require.

Loaders resolve an asset reference only when the tree's ``policy.lock`` lists
it. A test that builds a tree under ``tmp_path`` writes that lock here, from
the bytes it placed, in the same ``sha256sum`` format ``ora lock``
writes; the tree need not be a Git checkout.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from ora.orchestrator.policy_lock import LOCK_PATH, format_lock


def write_tree_lock(root: Path) -> Path:
    """List every non-Python file under ``root`` in ``root/policy.lock``."""

    root = Path(root)
    digests = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix != ".py" and path.name != LOCK_PATH
    }
    lock = root / LOCK_PATH
    lock.write_text(format_lock(digests), encoding="utf-8")
    return lock


__all__ = ["write_tree_lock"]
