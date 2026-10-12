"""Host-only replay persistence, recovery and Viewer cache integrity contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ora.environment import controller_replay as replay


def recorder(tmp_path: Path, backend: str = "robotwin") -> replay.ControllerReplayRecorder:
    return replay.ControllerReplayRecorder(
        output_root=tmp_path / "native", backend=backend,
        environment_identity={"seed": 7, "source_commit": "frozen-source"},
        camera_panels=["front"],
    )


def record(writer: replay.ControllerReplayRecorder, index: int, **kwargs):
    return writer.record(
        action_sequence=index,
        action={"action_id": f"action-{index}", "action": "move_pose"},
        controller="test_numeric_controller", arrays={"commands": [[index, 0.5]]},
        physics_steps=1, **kwargs,
    )


def read_header(writer):
    return json.loads(writer.manifest_path.read_text())


def resign_journal(writer, payload: bytes, count: int):
    """Retain valid byte/hash binding while testing semantic event validation."""
    writer.journal_path.write_bytes(payload)
    header = read_header(writer)
    header["journal"].update(bytes=len(payload), events=count,
                             sha256=hashlib.sha256(payload).hexdigest())
    writer.manifest_path.write_text(json.dumps(header))


@pytest.mark.parametrize("backend", ["robotwin", "libero_pro"])
def test_append_work_and_manifest_size_do_not_grow_with_history(tmp_path, monkeypatch, backend):
    writer = recorder(tmp_path, backend)
    record(writer, 0, metadata={"history_sentinel": "large-old-record", "payload": "x" * 8192})
    initial_header_size = writer.manifest_path.stat().st_size
    prefix = writer.journal_path.read_bytes()
    artifact = writer.root / "action-0000.npz"
    artifact_identity = (artifact.read_bytes(), artifact.stat().st_mtime_ns)
    original_jsonable = replay._jsonable
    history_visits = []

    def observe(value):
        if isinstance(value, dict) and value.get("history_sentinel") == "large-old-record":
            history_visits.append(value)
        return original_jsonable(value)

    monkeypatch.setattr(replay, "_jsonable", observe)
    for index in range(1, 65):
        record(writer, index)
        writer.update(index, execution_status="completed", settle_steps=2)
    # Updating an old action appends just its receipt, never its old payload.
    writer.update(0, execution_status="completed")
    assert not history_visits
    assert writer.journal_path.read_bytes().startswith(prefix)
    assert (artifact.read_bytes(), artifact.stat().st_mtime_ns) == artifact_identity
    assert writer.manifest_path.stat().st_size < initial_header_size + 100
    assert "entries" not in read_header(writer)
    loaded = replay.load_controller_replay_manifest(writer.manifest_path)
    assert len(loaded["entries"]) == 65
    assert all(entry["execution"]["status"] == "completed" for entry in loaded["entries"])


def test_round_order_and_late_receipts_reconstruct_exactly(tmp_path):
    writer = recorder(tmp_path)
    record(writer, 0)
    writer.begin_round(1)
    record(writer, 1)
    writer.update(1, execution_status="failed", execution_fault="controller stop")
    writer.update(0, execution_status="completed", settle_steps=4)
    entries = replay.load_controller_replay_manifest(writer.manifest_path)["entries"]
    assert [(entry["action_sequence"], entry["round_index"]) for entry in entries] == [(0, 0), (1, 1)]
    assert entries[0]["execution"] == {"status": "completed", "settle_steps": 4, "fault": None}
    assert entries[1]["execution"]["fault"] == "controller stop"
    assert replay.load_controller_replay_arrays(writer.manifest_path, entries[1])["commands"].tolist() == [[1, 0.5]]


def test_legacy_inline_manifest_remains_readable(tmp_path):
    writer = recorder(tmp_path)
    record(writer, 0)
    writer.update(0, execution_status="completed", settle_steps=3)
    legacy = replay.load_controller_replay_manifest(writer.manifest_path)
    legacy["schema"] = replay.LEGACY_MANIFEST_SCHEMA
    legacy.pop("journal")
    writer.manifest_path.write_text(json.dumps(legacy))
    writer.journal_path.unlink()
    assert replay.load_controller_replay_manifest(writer.manifest_path) == legacy


@pytest.mark.parametrize("tail", [b'{"seq":', b'{"seq":1,"type":"invalid-uncommitted-event"}\n'])
def test_reader_recovers_only_committed_prefix(tmp_path, tail):
    writer = recorder(tmp_path)
    record(writer, 0)
    expected = replay.load_controller_replay_manifest(writer.manifest_path)
    with writer.journal_path.open("ab") as stream:
        stream.write(tail)
    assert replay.load_controller_replay_manifest(writer.manifest_path) == expected


@pytest.mark.parametrize("damage", ["tamper", "reorder", "missing", "truncated", "count", "partial_commit"])
def test_committed_journal_damage_fails_closed(tmp_path, damage):
    writer = recorder(tmp_path)
    record(writer, 0)
    writer.update(0, execution_status="completed")
    payload = writer.journal_path.read_bytes()
    if damage == "tamper":
        writer.journal_path.write_bytes(payload.replace(b"completed", b"corrupted"))
    elif damage == "reorder":
        resign_journal(writer, b"".join(reversed(payload.splitlines(keepends=True))), 2)
    elif damage == "missing":
        writer.journal_path.unlink()
    elif damage == "truncated":
        writer.journal_path.write_bytes(payload[:-5])
    elif damage == "count":
        resign_journal(writer, payload, 1)
    else:
        resign_journal(writer, payload[:-1], 2)
    with pytest.raises(replay.ControllerReplayError):
        replay.load_controller_replay_manifest(writer.manifest_path)


@pytest.mark.parametrize("missing", [False, True])
def test_numeric_artifact_integrity_remains_required(tmp_path, missing):
    writer = recorder(tmp_path)
    entry = record(writer, 0)
    artifact = writer.output_root / entry["artifact"]["path"]
    if missing:
        artifact.unlink()
    else:
        artifact.write_bytes(b"tampered numeric controller targets")
    with pytest.raises(replay.ControllerReplayError):
        replay.load_controller_replay_manifest(writer.manifest_path)


@pytest.mark.parametrize("operation", ["record", "update", "round"])
def test_failed_commit_preserves_previous_prefix_and_disables_writer(tmp_path, monkeypatch, operation):
    writer = recorder(tmp_path)
    record(writer, 0)
    previous = replay.load_controller_replay_manifest(writer.manifest_path)
    original_atomic_json = replay._atomic_json

    def fail_publication(path, value):
        raise OSError("injected commit publication failure")

    monkeypatch.setattr(replay, "_atomic_json", fail_publication)
    with pytest.raises(OSError, match="publication failure"):
        if operation == "record":
            record(writer, 1)
        elif operation == "update":
            writer.update(0, execution_status="completed")
        else:
            writer.begin_round(1)
    monkeypatch.setattr(replay, "_atomic_json", original_atomic_json)
    assert replay.load_controller_replay_manifest(writer.manifest_path) == previous
    before = {path.name: path.read_bytes() for path in writer.root.iterdir() if path.is_file()}
    with pytest.raises(replay.ControllerReplayError, match="failed commit"):
        writer.update(0, execution_status="completed")
    with pytest.raises(replay.ControllerReplayError, match="failed commit"):
        writer.begin_round(2)
    with pytest.raises(replay.ControllerReplayError, match="failed commit"):
        record(writer, len(writer.entries))
    assert {path.name: path.read_bytes() for path in writer.root.iterdir() if path.is_file()} == before


def test_viewer_replay_identity_changes_after_each_commit(tmp_path):
    from viewer.replay import replay_identity

    writer = recorder(tmp_path)
    identities = [replay_identity(tmp_path)[0]]
    record(writer, 0)
    identities.append(replay_identity(tmp_path)[0])
    writer.update(0, execution_status="completed")
    identities.append(replay_identity(tmp_path)[0])
    writer.begin_round(1)
    identities.append(replay_identity(tmp_path)[0])
    assert len(set(identities)) == len(identities)


@pytest.mark.parametrize("damage", ["journal_tamper", "missing_numeric_artifact"])
def test_viewer_replay_identity_validates_evidence_before_cache_lookup(tmp_path, damage):
    from viewer.replay import replay_identity

    writer = recorder(tmp_path)
    entry = record(writer, 0)
    writer.update(0, execution_status="completed")
    assert replay_identity(tmp_path)[0]
    # Keep the manifest untouched: its digest alone cannot detect either fault.
    original_manifest = writer.manifest_path.read_bytes()
    if damage == "journal_tamper":
        writer.journal_path.write_bytes(writer.journal_path.read_bytes().replace(b"completed", b"corrupted"))
    else:
        (writer.output_root / entry["artifact"]["path"]).unlink()
    assert writer.manifest_path.read_bytes() == original_manifest
    with pytest.raises(replay.ControllerReplayError):
        replay_identity(tmp_path)


def test_cached_viewer_video_rejects_journal_tamper(tmp_path):
    from viewer.review_media import derived_root, find_execution_media

    writer = recorder(tmp_path)
    record(writer, 0)
    writer.update(0, execution_status="completed")
    cache = derived_root(tmp_path)
    cache.mkdir(parents=True)
    video = cache / "trajectory-forensic-fixture.mp4"
    video.write_bytes(b"fixture-video")
    timeline = cache / "timeline.jsonl"
    timeline.write_text(json.dumps({"frame_index": 0, "phase": "reset", "action_id": None}) + "\n")
    source_hashes = {"plan.json": "fixture-source-plan"}
    video.with_suffix(".json").write_text(json.dumps({
        "schema": "ora.viewer-trajectory-video.v2",
        "run_id": tmp_path.name, "replay_mode": "forensic", "source_hashes": source_hashes,
        "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        "frame_count": 1, "fps": 24, "continuous_execution": True,
        "controller_replay": True,
        "controller_replay_manifest_sha256": hashlib.sha256(writer.manifest_path.read_bytes()).hexdigest(),
        "execution_timeline": {"path": timeline.name, "sha256": hashlib.sha256(timeline.read_bytes()).hexdigest()},
    }))
    trajectory = {"source_hashes": source_hashes, "steps": []}
    assert find_execution_media(tmp_path, trajectory, replay_mode="forensic") is not None
    writer.journal_path.write_bytes(writer.journal_path.read_bytes().replace(b"completed", b"corrupted"))
    assert find_execution_media(tmp_path, trajectory, replay_mode="forensic") is None
