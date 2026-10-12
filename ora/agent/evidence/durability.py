"""Deferred evidence durability: write in the loop, fsync behind a barrier.

Native evidence used to call ``fsync`` inside the policy loop for every event
line and every content-addressed Artifact (5–8 MB Provider requests on each
call). On a shared ext4 HDD each ``fsync`` waits for a journal commit that
queues behind unrelated writeback, so the loop spent seconds per call blocked
in ``jbd2_log_wait_commit``.

:class:`DeferredDurability` keeps every write, rename and link on the calling
thread, in the original order, so bytes and names are visible to every reader
(including the in-process evaluator and a live Viewer) exactly when they were
before. Only the ``fsync`` moves to one background thread. The guarantee is:

* **Process crash** (exception, SIGKILL, OOM): nothing is lost; completed
  writes live in the kernel page cache and reach disk through normal writeback.
* **Kernel crash or power loss**: files submitted after the last completed
  :meth:`DeferredDurability.barrier` may be missing, truncated or stale. A
  truncated content-addressed file fails its SHA-256 check, so evaluation
  fails closed instead of accepting it.
* After :meth:`barrier` returns, every file submitted before the call has been
  ``fsync``-ed, exactly as the synchronous path would have done.

The queue is bounded by ``max_pending`` distinct paths; :meth:`submit` blocks
when it is full, so a stalled disk throttles the loop instead of growing
memory. Repeated submissions of a path that has not started syncing coalesce.
Any ``fsync`` or snapshot failure is sticky: later submissions and barriers
raise :class:`DurabilityError`.

:meth:`replace_text` is the latest-wins variant for rewritable snapshots: the
worker writes the newest submitted text via a temporary file and
``os.replace``; an older pending text for the same path is dropped because the
newer one supersedes it.
"""

from __future__ import annotations

import os
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Callable


DEFAULT_MAX_PENDING = 64


class DurabilityError(RuntimeError):
    """A deferred evidence write or fsync failed; evidence is not durable."""


def fsync_path(path: Path) -> None:
    """Flush one file's data and metadata to stable storage."""

    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_text_replace(path: Path, text: str) -> None:
    """Replace ``path`` with ``text`` through a same-directory temporary file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class DeferredDurability:
    """One background thread that fsyncs already-written evidence files."""

    def __init__(
        self, *, max_pending: int = DEFAULT_MAX_PENDING,
        fsync: Callable[[Path], None] = fsync_path,
        replace: Callable[[Path, str], None] = write_text_replace,
    ) -> None:
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        self.max_pending = max_pending
        self._fsync = fsync
        self._replace = replace
        self._condition = threading.Condition()
        # path -> None (fsync) or str (latest snapshot text to write).
        self._pending: OrderedDict[Path, str | None] = OrderedDict()
        self._active = 0
        self._error: BaseException | None = None
        self._closed = False
        self.completed_fsyncs = 0
        self.completed_replacements = 0
        self.superseded_replacements = 0
        self._thread = threading.Thread(
            target=self._run, name="evidence-durability", daemon=True,
        )
        self._thread.start()

    # -- producer side -------------------------------------------------
    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise DurabilityError(
                f"deferred evidence durability failed: {self._error!r}"
            ) from self._error

    def _enqueue(self, path: Path, value: str | None) -> None:
        with self._condition:
            self._raise_if_failed()
            if self._closed:
                raise DurabilityError("deferred evidence durability is closed")
            if path in self._pending:
                if value is not None:
                    if isinstance(self._pending[path], str):
                        self.superseded_replacements += 1
                    self._pending[path] = value
                # A pending fsync already covers data written before it runs.
                self._condition.notify_all()
                return
            while len(self._pending) >= self.max_pending:
                self._condition.wait()
                self._raise_if_failed()
            self._pending[path] = value
            self._condition.notify_all()

    def submit(self, path: Path) -> None:
        """Schedule ``fsync(path)`` for data already written to ``path``."""

        self._enqueue(Path(path), None)

    def replace_text(self, path: Path, text: str) -> None:
        """Schedule an atomic latest-wins snapshot rewrite of ``path``."""

        self._enqueue(Path(path), text)

    def barrier(self) -> None:
        """Wait until every earlier submission is durable; raise on failure."""

        with self._condition:
            while self._pending or self._active:
                self._condition.wait()
            self._raise_if_failed()

    def close(self, *, raise_errors: bool = True) -> None:
        """Drain every submission, stop the worker and surface any failure.

        ``raise_errors=False`` is for cleanup paths that already propagate an
        earlier exception; the drain still completes before the worker stops.
        """

        with self._condition:
            while self._pending or self._active:
                self._condition.wait()
            self._closed = True
            self._condition.notify_all()
        self._thread.join()
        if raise_errors:
            with self._condition:
                self._raise_if_failed()

    @property
    def pending(self) -> int:
        with self._condition:
            return len(self._pending) + self._active

    # -- worker side ---------------------------------------------------
    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._closed:
                    self._condition.wait()
                if not self._pending:
                    return
                path, value = self._pending.popitem(last=False)
                self._active += 1
                self._condition.notify_all()
            try:
                if value is None:
                    self._fsync(path)
                else:
                    self._replace(path, value)
            except BaseException as exc:  # sticky; surfaced to the loop
                with self._condition:
                    if self._error is None:
                        self._error = exc
            finally:
                with self._condition:
                    self._active -= 1
                    if value is None:
                        self.completed_fsyncs += 1
                    else:
                        self.completed_replacements += 1
                    self._condition.notify_all()
