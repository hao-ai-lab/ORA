"""Context Artifact index persistence: atomic, batched and deferred-durable.

The Session Artifact index is one JSON file rewritten whenever a reference is
registered. A lossless projection can register many references, so the
rewrite is coalesced to once per projection (``batched_persistence``), and its
``fsync`` follows the owning ArtifactStore's durability policy: inline for the
synchronous default, or behind a
:class:`~ora.agent.evidence.durability.DeferredDurability` barrier.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterator, Mapping


def atomic_json(path: Path, value: Mapping[str, Any], *, durability: Any = None) -> None:
    """Atomically replace ``path``; fsync inline or behind ``durability``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(
                json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True)
                + "\n"
            )
            stream.flush()
            if durability is None:
                os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    if durability is not None:
        durability.submit(path)


class BatchedIndexPersistence:
    """Mixin: ``_persist`` writes now, or once when the outermost batch exits."""

    _batch_depth = 0
    _batch_dirty = False

    def _write_index(self) -> None:  # pragma: no cover - implemented by owner
        raise NotImplementedError

    def _persist(self) -> None:
        if self._batch_depth:
            self._batch_dirty = True
            return
        self._write_index()

    @contextlib.contextmanager
    def batched_persistence(self) -> Iterator[None]:
        """Coalesce index rewrites inside one projection into a single write.

        In-memory resolution is unchanged. The file is rewritten once on exit
        (also on error) when anything was registered, before the caller can
        publish any reference minted inside the block.
        """

        self._batch_depth += 1
        try:
            yield
        finally:
            self._batch_depth -= 1
            if self._batch_depth == 0 and self._batch_dirty:
                self._batch_dirty = False
                self._write_index()
