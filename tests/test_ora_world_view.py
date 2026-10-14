"""Read-only live pixels, causal capture lag and decoded browser binding."""
import base64
import hashlib
import json
import time
from pathlib import Path
from urllib.request import urlopen

import pytest

from viewer.live_world import WorldViewServer, _Prefix
from ora.environment.execution_capture import FRAME_SCHEMA, MANIFEST_SCHEMA

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jB9sAAAAASUVORK5CYII=")


class Run:
    def __init__(self, root: Path):
        self.root, self.seq, self.index = root, 0, 0
        root.mkdir()
        self.capture = root / "native/execution-capture"
        (self.capture / "frames").mkdir(parents=True)
        (root / "agent-events.jsonl").touch()
        (self.capture / "timeline.jsonl").touch()
        (root / "plan.json").write_text(json.dumps({"runtime": {"kind": "robotwin", "execution_capture": {"enabled": True}}}))
        self.manifest = {"schema": MANIFEST_SCHEMA, "identity": {"run_id": root.name},
                         "camera_panels": ["head_camera", "front_camera"],
                         "frame_count": 0, "status": "recording", "finalized": False}
        self.write_manifest()

    def write_manifest(self):
        path = self.capture / "manifest.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.manifest))
        temporary.replace(path)

    def artifact(self, value):
        path = self.root / f"native/artifact-{self.seq}.json"
        raw = json.dumps(value).encode()
        path.write_bytes(raw)
        return {"path": str(path.relative_to(self.root)), "sha256": hashlib.sha256(raw).hexdigest()}

    def event(self, kind, payload=None, *, artifact=None, append=True):
        event = {"schema": "ora.agent-event.v1", "run_id": self.root.name,
                 "seq": self.seq, "monotonic_ns": 1_000_000_000 + self.seq * 1_000_000,
                 "unix_s": time.time(), "type": kind, "payload": payload or {}, "artifact_ref": artifact}
        self.seq += 1
        if append:
            with (self.root / "agent-events.jsonl").open("a") as stream:
                stream.write(json.dumps(event) + "\n")
        return event

    def action(self, aid="action-one", *, skill=False):
        action = {"action_id": aid}
        if skill:
            action.update(schema="ora-skill-sequence-action.v1", arguments={"sequence": ["open", "lift", "verify"], "resume_from": 0})
        self.event("action_proposed", {"action_id": aid}, artifact=self.artifact(action))

    def receipt(self, aid="action-one", children=()):
        body = {"action_id": aid, "stage_receipts": [{"action_id": child} for child in children]}
        self.event("environment_receipt", {"action_id": aid}, artifact=self.artifact(body))

    def frame(self, aid=None, *, phase="reset", sim_step=0, append=True):
        images = {}
        for i, name in enumerate(self.manifest["camera_panels"]):
            path = self.capture / f"frames/{self.index:08d}-{i}.png"
            raw = PNG + str(self.index).encode()
            path.write_bytes(raw)
            images[name] = {"path": str(path.relative_to(self.capture)), "sha256": hashlib.sha256(raw).hexdigest()}
        frame = {"schema": FRAME_SCHEMA, "source": "original_execution", "visibility": "control_only",
                 "frame_index": self.index, "capture_index": self.index, "round_index": 0,
                 "action_id": aid, "phase": phase, "sim_step": sim_step, "sim_time_s": sim_step / 100,
                 "captured_unix_s": time.time(), "captured_monotonic_s": time.monotonic(), "images": images}
        self.index += 1
        if append:
            with (self.capture / "timeline.jsonl").open("a") as stream:
                stream.write(json.dumps(frame) + "\n")
            self.manifest["frame_count"] = self.index
            self.write_manifest()
        return frame


@pytest.fixture
def run(tmp_path):
    return Run(tmp_path / "run-one")


@pytest.fixture
def server(run):
    server = WorldViewServer()
    server.bind(run.root)
    yield server
    server.close()


def browser_state(server):
    return server.app.test_client().get("/api/live/status").get_json()


def ack(server, state, *, fetch=True):
    client = server.app.test_client()
    if fetch:
        for camera in state["frame"]["cameras"]:
            assert client.get(camera["url"]).status_code == 200
    return client.post("/api/live/ack", json={"binding_id": state["binding_id"], "ack_token": state["ack_token"],
                       "frame_index": state["frame"]["index"], "frame_hash": state["frame"]["hash"]},
                       headers={"Origin": "http://localhost"})


def test_loopback_service_starts_before_task_and_closes_idempotently():
    server = WorldViewServer().start()
    try:
        with urlopen(server.url.replace("/live", "/health"), timeout=2) as response:
            assert json.load(response) == {"service": "ora-live-world", "status": "ready"}
        assert server.status()["phase"] == "waiting"
        assert not server.status()["connected"]
    finally:
        server.close()
        server.close()
    with pytest.raises(ValueError, match="loopback"):
        WorldViewServer(host="0.0.0.0")


def test_first_image_requires_delivered_cameras_and_decode_ack(run, server):
    run.frame()
    state = browser_state(server)
    assert state["phase"] == "holding" and not state["visible"] and not state["connected"]
    assert ack(server, state, fetch=False).status_code == 409
    assert ack(server, state).status_code == 200
    assert server.status()["visible"] and server.status()["connected"]
    # Old source pixels do not imply browser disconnection during thinking.
    server.observer.frames[0]["record"]["captured_unix_s"] -= 100
    assert server.status()["connected"] and server.status()["frame"]["age_s"] >= 100


def test_async_writer_tail_stays_pending_until_exact_receipt_end(run, server):
    run.frame()
    run.action()
    run.receipt()
    run.event("provider_request")
    state = server.status()
    assert state["phase"] == "awaiting_capture" and state["agent_phase"] == "thinking"
    run.frame("action-one", phase="control", sim_step=3)
    assert server.status()["phase"] == "awaiting_capture"
    run.frame("action-one", phase="action_end", sim_step=5)
    state = server.status()
    assert state["phase"] == "holding" and not state["capture_pending"]
    assert state["frame"]["sim_step"] == 5


def test_skill_children_join_commitment_and_actual_stage_receipts(run, server):
    run.frame()
    run.action(skill=True)
    run.frame("action-one:00-open", phase="action_start")
    assert server.status()["phase"] == "executing"
    assert server.status()["frame"]["parent_action_id"] == "action-one"
    run.frame("action-one:00-open", phase="action_end", sim_step=2)
    run.receipt(children=["action-one:00-open", "action-one:01-lift"])
    run.event("provider_request")
    assert server.status()["capture_pending"]
    run.frame("action-one:01-lift", phase="action_end", sim_step=6)
    state = server.status()
    assert state["phase"] == "holding" and state["frame"]["parent_action_id"] == "action-one"


def test_old_binding_and_old_decoded_frame_cannot_become_current(run, server, tmp_path):
    run.frame()
    first = browser_state(server)
    assert ack(server, first).status_code == 200
    run.frame("action-two", phase="control", sim_step=1)
    second = browser_state(server)
    # A recently delivered older frame is truthful connection evidence, not
    # evidence that the latest frame was decoded.
    assert ack(server, first).status_code == 200
    assert server.status()["connected"] and not server.status()["visible"]
    assert ack(server, second).status_code == 200
    assert ack(server, first).status_code == 409
    other = Run(tmp_path / "run-two")
    server.bind(other.root)
    assert not server.status()["visible"] and not server.status()["connected"]
    assert ack(server, second, fetch=False).status_code == 409
    assert server.app.test_client().get(first["frame"]["cameras"][0]["url"]).status_code == 404


def test_browser_frame_is_pinned_while_background_observer_advances(run, server):
    run.frame()
    first = browser_state(server)
    for n in range(20):
        run.frame("action-one", phase="control", sim_step=n + 1)
    for _ in range(5):
        server.status()
    assert 0 not in server.observer.frames
    assert ack(server, first).status_code == 200
    assert server.status()["connected"] and not server.status()["visible"]


def test_torn_prefixes_wait_for_newline(run, server):
    event = run.event("provider_request", append=False)
    raw = json.dumps(event)
    path = run.root / "agent-events.jsonl"
    path.write_text(raw[:len(raw) // 2])
    assert server.status()["error"] is None and server.observer.event_seq == -1
    with path.open("a") as stream:
        stream.write(raw[len(raw) // 2:] + "\n")
    assert server.status()["agent_phase"] == "thinking"
    frame = run.frame(append=False)
    raw = json.dumps(frame)
    path = run.capture / "timeline.jsonl"
    path.write_text(raw[:-5])
    assert server.status()["frame"] is None
    with path.open("a") as stream:
        stream.write(raw[-5:] + "\n")
    assert server.status()["frame"]["index"] == 0


def test_prefix_drains_buffer_before_another_disk_read(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps({"index": i}) + "\n" for i in range(1000)))
    prefix = _Prefix(tmp_path, path.name)
    assert prefix.read(1) == [{"index": 0}]
    offset = prefix.offset
    for i in range(1, 100):
        assert prefix.read(1) == [{"index": i}]
        assert prefix.offset == offset


@pytest.mark.parametrize("damage", ["hash", "traversal", "symlink", "nan", "boolean", "identity"])
def test_invalid_original_capture_is_rejected(run, server, tmp_path, damage):
    frame = run.frame(append=False)
    image = frame["images"]["head_camera"]
    if damage == "hash":
        image["sha256"] = "0" * 64
    elif damage == "traversal":
        image["path"] = "../../plan.json"
    elif damage == "symlink":
        original = run.capture / image["path"]
        external = tmp_path / "external.png"
        external.write_bytes(original.read_bytes())
        original.unlink()
        original.symlink_to(external)
    elif damage == "nan":
        frame["sim_time_s"] = float("nan")
    elif damage == "boolean":
        frame["sim_step"] = True
    elif damage == "identity":
        run.manifest["identity"]["run_id"] = "another-run"
        run.write_manifest()
    (run.capture / "timeline.jsonl").write_text(json.dumps(frame) + "\n")
    state = server.status()
    assert state["phase"] == "error" and state["error"] and state["frame"] is None


def test_nonce_same_origin_hash_and_host_protect_ack(run, server):
    run.frame()
    state = browser_state(server)
    assert ack(server, state).status_code == 200
    client = server.app.test_client()
    body = {"binding_id": state["binding_id"], "ack_token": state["ack_token"],
            "frame_index": 0, "frame_hash": state["frame"]["hash"]}
    assert client.post("/api/live/ack", json=body, headers={"Origin": "https://example.com"}).status_code == 403
    body["ack_token"] = "wrong"
    assert client.post("/api/live/ack", json=body, headers={"Origin": "http://localhost"}).status_code == 409
    assert client.get("/health", headers={"Host": "evil.example"}).status_code == 403
    assert client.get("/health", headers={"Host": "[2001:db8::1]"}).status_code == 403
    assert client.get("/health", headers={"Host": "[::1]"}).status_code == 200


def test_finished_is_static_and_viewer_does_not_write_run(run, server):
    run.frame()
    run.action()
    run.frame("action-one", phase="action_end", sim_step=7)
    run.receipt()
    run.frame(phase="terminal", sim_step=7)
    run.event("official_verdict", {"official_success": True})
    run.event("run_terminated")
    before = {str(p): p.read_bytes() for p in run.root.rglob("*") if p.is_file()}
    state = browser_state(server)
    assert state["phase"] == "finished" and state["frame"]["sim_step"] == 7
    assert ack(server, state).status_code == 200
    assert server.status()["phase"] == "finished"
    assert {str(p): p.read_bytes() for p in run.root.rglob("*") if p.is_file()} == before


def test_final_missing_capture_boundary_is_unavailable_instead_of_false_hold(run, server):
    run.frame()
    run.action()
    run.receipt()
    run.event("run_terminated")
    run.manifest.update(finalized=True, status="incomplete")
    run.write_manifest()
    state = server.status()
    assert state["phase"] == "unavailable" and state["capture_pending"]
