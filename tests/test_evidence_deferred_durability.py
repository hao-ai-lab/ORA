"""Deferred evidence durability keeps fsync off the Native policy loop.

The Native loop writes every event line and content-addressed Artifact on its
own thread and in its original order; only fsync runs on the durability worker.
These tests are deterministic: a blocked fake fsync proves that the loop
returns without waiting for storage, instead of timing wall clocks.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import pytest

from ora.agent.evidence import durability as durability_module
from ora.agent.evidence.durability import (
    DeferredDurability, DurabilityError,
)
from ora.agent.evidence.trajectory import (
    ArtifactStore, TrajectoryRecorder, canonical_bytes, read_agent_events,
)
from ora.agent.memory.artifacts import ContextArtifactCatalog


class GatedFsync:
    """Fake fsync that records its thread and waits for an explicit release."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.started = threading.Event()
        self.calls: list[tuple[Path, int]] = []
        self.lock = threading.Lock()

    def __call__(self, path: Path) -> None:
        with self.lock:
            self.calls.append((Path(path), threading.get_ident()))
        self.started.set()
        assert self.release.wait(10), "test never released the gated fsync"


def test_loop_returns_while_every_fsync_is_blocked(tmp_path: Path) -> None:
    gate = GatedFsync()
    durability = DeferredDurability(fsync=gate, max_pending=64)
    store = ArtifactStore(tmp_path, durability=durability)
    recorder = TrajectoryRecorder(
        tmp_path, "run", "policy-session", durability=durability,
    )
    loop_thread = threading.get_ident()
    refs = []
    for index in range(8):
        # A Provider request of several MB: the loop must not wait for it.
        payload = canonical_bytes({"call": index, "body": "x" * 200_000})
        refs.append(store.put_bytes(
            payload, kind="native_provider_wire_request", suffix=".json",
            media_type="application/json",
        ))
        recorder.append("provider_request", payload={"call": index},
                        artifact_ref=refs[-1])
    assert gate.started.wait(10)
    # Everything is already visible, hash-valid and in order before any fsync
    # has been allowed to finish.
    for ref in refs:
        data = (tmp_path / ref["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == ref["sha256"]
    events = read_agent_events(tmp_path)
    assert [event["seq"] for event in events] == list(range(8))
    assert all(
        events[index]["previous_event_sha256"] == events[index - 1]["event_sha256"]
        for index in range(1, 8)
    )
    assert durability.pending > 0
    gate.release.set()
    durability.barrier()
    assert durability.pending == 0
    synced = {path for path, _thread in gate.calls}
    assert {tmp_path / ref["path"] for ref in refs} <= synced
    assert tmp_path / "agent-events.jsonl" in synced
    assert all(thread != loop_thread for _path, thread in gate.calls)
    durability.close()


def test_synchronous_default_still_fsyncs_inline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    real_fsync = durability_module.os.fsync

    def recording_fsync(descriptor: int) -> None:
        calls.append(threading.get_ident())
        real_fsync(descriptor)

    monkeypatch.setattr("ora.agent.evidence.trajectory.os.fsync", recording_fsync)
    store = ArtifactStore(tmp_path)
    recorder = TrajectoryRecorder(tmp_path, "run", "policy-session")
    store.put_json({"a": 1}, kind="x")
    recorder.append("session_started", payload={})
    assert calls == [threading.get_ident(), threading.get_ident()]


def test_deferred_and_synchronous_evidence_bytes_match(tmp_path: Path) -> None:
    deferred_root, sync_root = tmp_path / "deferred", tmp_path / "sync"
    durability = DeferredDurability()
    stores = (
        ArtifactStore(deferred_root, durability=durability), ArtifactStore(sync_root),
    )
    for store in stores:
        for index in range(5):
            store.put_json({"index": index, "text": "ü" * index}, kind="k")
        store.put_json({"index": 0, "text": ""}, kind="k")  # duplicate content
    durability.close()
    listing = [
        sorted(
            (path.name, path.read_bytes())
            for path in (root / "native/agent-artifacts").iterdir()
        )
        for root in (deferred_root, sync_root)
    ]
    assert listing[0] == listing[1]
    assert not any(name.endswith(".tmp") for name, _data in listing[0])


def test_submission_queue_is_bounded(tmp_path: Path) -> None:
    gate = GatedFsync()
    durability = DeferredDurability(fsync=gate, max_pending=2)
    paths = [tmp_path / f"f{index}" for index in range(4)]
    for path in paths:
        path.write_text("x")
    durability.submit(paths[0])
    assert gate.started.wait(10)  # f0 is in flight, not pending
    durability.submit(paths[1])
    durability.submit(paths[2])
    blocked = threading.Event()
    finished = threading.Event()

    def producer() -> None:
        blocked.set()
        durability.submit(paths[3])
        finished.set()

    worker = threading.Thread(target=producer)
    worker.start()
    assert blocked.wait(10)
    assert not finished.wait(0.2)  # backpressure: the fourth file waits
    gate.release.set()
    assert finished.wait(10)
    worker.join()
    durability.close()
    assert [path for path, _thread in gate.calls] == paths


def test_repeated_submissions_coalesce(tmp_path: Path) -> None:
    gate = GatedFsync()
    durability = DeferredDurability(fsync=gate)
    first, second = tmp_path / "a", tmp_path / "b"
    first.write_text("a")
    second.write_text("b")
    durability.submit(first)
    assert gate.started.wait(10)
    # ``first`` is already syncing, so a new submission must queue again;
    # the three submissions of ``second`` collapse into one fsync.
    durability.submit(first)
    for _ in range(3):
        durability.submit(second)
    gate.release.set()
    durability.close()
    assert [path for path, _thread in gate.calls] == [first, first, second]


def test_failure_is_sticky_and_surfaces_at_barrier(tmp_path: Path) -> None:
    def failing(path: Path) -> None:
        raise OSError(5, "Input/output error")

    durability = DeferredDurability(fsync=failing)
    target = tmp_path / "x"
    target.write_text("x")
    durability.submit(target)
    with pytest.raises(DurabilityError):
        durability.barrier()
    with pytest.raises(DurabilityError):
        durability.submit(target)
    with pytest.raises(DurabilityError):
        durability.close()


def test_snapshot_replacement_is_latest_wins(tmp_path: Path) -> None:
    written: list[str] = []
    started, release = threading.Event(), threading.Event()

    def replace(path: Path, text: str) -> None:
        started.set()
        release.wait(10)
        written.append(text)
        durability_module.write_text_replace(path, text)

    durability = DeferredDurability(replace=replace)
    target = tmp_path / "native/agent-session.json"
    durability.replace_text(target, "v1\n")
    assert started.wait(10)
    for version in range(2, 6):
        durability.replace_text(target, f"v{version}\n")
    release.set()
    durability.barrier()
    assert written == ["v1\n", "v5\n"]
    assert durability.superseded_replacements == 3
    assert target.read_text() == "v5\n"
    assert not list(target.parent.glob("*.tmp"))
    durability.close()


def test_context_index_batches_one_rewrite_per_projection(tmp_path: Path) -> None:
    durability = DeferredDurability()
    store = ArtifactStore(tmp_path, durability=durability)
    catalog = ContextArtifactCatalog(tmp_path, "policy-session", store)
    writes: list[int] = []
    original = catalog.__class__._write_index

    def counting(self: ContextArtifactCatalog) -> None:
        writes.append(len(self._entries))
        original(self)

    catalog.__class__._write_index = counting  # type: ignore[method-assign]
    try:
        with catalog.batched_persistence():
            refs = [
                catalog.put_bytes(
                    f"payload-{index}".encode(), kind="context_tool_result",
                    suffix=".json", media_type="application/json",
                    round_index=1, action_step=index,
                )
                for index in range(4)
            ]
            # In-memory resolution is immediate inside the batch.
            assert catalog.resolve(refs[0]["ref"])[1] == b"payload-0"
        assert writes == [4]
    finally:
        catalog.__class__._write_index = original  # type: ignore[method-assign]
    durability.close()
    index = json.loads((tmp_path / "native/context-artifacts/index.json").read_text())
    assert sorted(index["artifacts"]) == sorted(ref["ref"] for ref in refs)
