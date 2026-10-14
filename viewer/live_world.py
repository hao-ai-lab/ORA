"""Exact-run, read-only observation of committed original execution pixels.

This service never imports a simulator, advances physics, or writes run evidence.
The finalized-video reader remains separate: an in-flight append prefix is not a
sealed video. UI phases describe capture lag separately from agent activity.
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import re
import secrets
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from flask import Flask, Response, abort, jsonify, request
from werkzeug.serving import WSGIRequestHandler, make_server

from ora.environment.execution_capture import FRAME_SCHEMA, MANIFEST_SCHEMA
from viewer.ui import TEMPLATES, render_page

EVENT_SCHEMA = "ora.agent-event.v1"
MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_IMAGE_BYTES = 8 * 1024 * 1024
READ_CHUNK_BYTES = 256 * 1024
MAX_EVENTS_PER_POLL = 128
MAX_FRAMES_PER_POLL = 8
MAX_ACTIONS = 256
MAX_SERVED_FRAMES = 4
BROWSER_HEARTBEAT_S = 5.0


def _number(value: Any, *, integer: bool = False, minimum: float = 0) -> float | int:
    if isinstance(value, bool) or not isinstance(value, int if integer else (int, float)):
        raise ValueError("capture clock/index has an invalid numeric type")
    if not math.isfinite(value) or value < minimum:
        raise ValueError("capture clock/index is not finite or is negative")
    return value


def _hash(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("artifact hash is invalid")
    return value


def _safe_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("artifact path is invalid")
    lexical = root / relative
    if any(part.is_symlink() for part in (lexical, *lexical.parents) if part != root and part.is_relative_to(root)):
        raise ValueError("artifact symlink is not an original committed file")
    path = lexical.resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("artifact escaped the bound run")
    return path


def _read_bytes(path: Path, limit: int) -> bytes:
    with path.open("rb") as stream:
        value = stream.read(limit + 1)
    if len(value) > limit:
        raise ValueError("artifact exceeds the live observer read limit")
    return value


def _json(path: Path) -> dict:
    value = json.loads(_read_bytes(path, MAX_JSON_BYTES))
    if not isinstance(value, dict):
        raise ValueError("artifact is not a JSON object")
    return value


class _Prefix:
    """Read only complete appended lines; retain bounded torn suffixes."""

    def __init__(self, root: Path, relative: str):
        self.root, self.relative = root, relative
        self.offset, self.partial, self.identity = 0, b"", None

    def read(self, limit: int) -> list[dict]:
        result = []

        def drain():
            while b"\n" in self.partial and len(result) < limit:
                raw, self.partial = self.partial.split(b"\n", 1)
                if len(raw) > MAX_JSON_BYTES:
                    raise ValueError("evidence line exceeds the live observer limit")
                if raw.strip():
                    value = json.loads(raw)
                    if not isinstance(value, dict):
                        raise ValueError("evidence line is not an object")
                    result.append(value)

        drain()
        if len(result) == limit:
            return result
        path = _safe_path(self.root, self.relative)
        try:
            with path.open("rb") as stream:
                stat = path.stat()
                identity = (stat.st_dev, stat.st_ino)
                if self.identity is not None and (self.identity != identity or stat.st_size < self.offset):
                    raise ValueError("append-only evidence was replaced or truncated")
                self.identity = identity
                stream.seek(self.offset)
                data = stream.read(READ_CHUNK_BYTES)
        except FileNotFoundError:
            return result
        self.offset += len(data)
        self.partial += data
        drain()
        if len(self.partial) > MAX_JSON_BYTES:
            raise ValueError("incomplete evidence line exceeds the live observer limit")
        return result


class _Observer:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.events = _Prefix(self.root, "agent-events.jsonl")
        self.timeline = _Prefix(self.root, "native/execution-capture/timeline.jsonl")
        self.manifest: dict = {}
        self.manifest_stamp = None
        self.plan: dict = {}
        self.error: str | None = None
        self.disabled = False
        self.source_phase = "waiting"
        self.current_action: str | None = None
        self.latest_receipt: str | None = None
        self.finished = False
        self.event_seq = -1
        self.event_mono = -1
        self.capture_index = -1
        self.frame_index = -1
        self.round_index = -1
        self.sim_time = -1.0
        self.sim_step = -1
        self.capture_mono = -1.0
        self.actions: OrderedDict[str, dict] = OrderedDict()
        self.parents: OrderedDict[str, str] = OrderedDict()
        self.ends: OrderedDict[str, bool] = OrderedDict()
        self.frames: OrderedDict[int, dict] = OrderedDict()
        self.calls: OrderedDict[str, str] = OrderedDict()

    @staticmethod
    def _remember(table: OrderedDict, key: Any, value: Any, limit: int = MAX_ACTIONS):
        table[key] = value
        table.move_to_end(key)
        while len(table) > limit:
            table.popitem(last=False)

    def _artifact(self, reference: Any) -> dict:
        if not isinstance(reference, dict):
            raise ValueError("source action/receipt artifact reference is missing")
        path = _safe_path(self.root, reference.get("path"))
        raw = _read_bytes(path, MAX_JSON_BYTES)
        if hashlib.sha256(raw).hexdigest() != _hash(reference.get("sha256")):
            raise ValueError("source action/receipt artifact hash differs")
        result = json.loads(raw)
        if not isinstance(result, dict):
            raise ValueError("source action/receipt artifact is invalid")
        return result

    def _event(self, event: dict):
        if event.get("schema") != EVENT_SCHEMA or event.get("run_id") != self.root.name:
            raise ValueError("agent event schema/run identity differs")
        seq = _number(event.get("seq"), integer=True)
        mono = _number(event.get("monotonic_ns"), integer=True)
        _number(event.get("unix_s"))
        if seq != self.event_seq + 1 or mono < self.event_mono:
            raise ValueError("agent event order moved backwards or skipped")
        self.event_seq, self.event_mono = seq, mono
        payload = event.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("agent event payload is invalid")
        kind = event.get("type")
        if kind == "provider_request":
            self.source_phase = "thinking"
        elif kind == "tool_call_proposed":
            name = str(payload.get("name") or "tool")
            self._remember(self.calls, event.get("tool_call_id"), name)
            self.source_phase = "planning" if payload.get("effect") == "mutation" else "read-only tool"
        elif kind == "action_proposed":
            aid = payload.get("action_id")
            if not isinstance(aid, str) or not aid or aid in self.actions:
                raise ValueError("source action identity is missing or duplicated")
            self._remember(self.actions, aid, {"expected_end": None, "mono": mono,
                                            "tool": self.calls.get(event.get("tool_call_id"), "action")})
            self._remember(self.parents, aid, aid)
            if event.get("artifact_ref") is not None:
                action = self._artifact(event["artifact_ref"])
                if action.get("action_id") != aid:
                    raise ValueError("action proposal artifact identity differs")
                if action.get("schema") == "ora-skill-sequence-action.v1":
                    args = action.get("arguments") or {}
                    sequence = args.get("sequence")
                    resume = _number(args.get("resume_from", 0), integer=True)
                    if not isinstance(sequence, list) or len(sequence) > 64 or any(not isinstance(stage, str) for stage in sequence):
                        raise ValueError("source skill commitment is invalid")
                    for index, stage in enumerate(sequence):
                        if index >= resume and stage not in {"stop", "verify"}:
                            self._remember(self.parents, f"{aid}:{index:02d}-{stage}", aid)
            self.current_action, self.source_phase = aid, "executing"
        elif kind == "environment_receipt":
            aid = payload.get("action_id")
            if aid not in self.actions:
                raise ValueError("receipt has no exact source action proposal")
            body = self._artifact(event.get("artifact_ref"))
            receipt = body.get("receipt", body)
            if not isinstance(receipt, dict) or receipt.get("action_id") != aid:
                raise ValueError("receipt body does not match its exact action")
            children = []
            for stage in receipt.get("stage_receipts", []):
                if not isinstance(stage, dict):
                    raise ValueError("receipt stage is invalid")
                child = stage.get("action_id")
                if child is not None:
                    if not isinstance(child, str) or not child:
                        raise ValueError("receipt child identity is invalid")
                    if child in self.parents and self.parents[child] != aid:
                        raise ValueError("receipt child identity belongs to another source action")
                    self._remember(self.parents, child, aid)
                    children.append(child)
            # Combined target-state phases also retain explicit action objects.
            for phase in receipt.get("phase_receipts", []):
                if not isinstance(phase, dict):
                    raise ValueError("receipt phase is invalid")
                action = phase.get("action")
                if isinstance(action, dict) and phase.get("receipt") is not None:
                    child = action.get("action_id")
                    if not isinstance(child, str) or not child:
                        raise ValueError("receipt phase action identity is invalid")
                    self._remember(self.parents, child, aid)
                    children.append(child)
            self.actions[aid]["expected_end"] = children[-1] if children else aid
            self.latest_receipt = aid
            if self.current_action == aid:
                self.current_action = None
            self.source_phase = "observing"
        elif kind in {"fresh_observation_published", "observation_received"}:
            self.source_phase = "observing"
        elif kind == "official_verdict":
            self.source_phase = "round result"
        elif kind in {"run_terminated", "session_closed"}:
            self.finished = True
            self.current_action = None
            self.source_phase = "finished"

    def _frame(self, record: dict):
        if record.get("schema") != FRAME_SCHEMA or record.get("source") != "original_execution":
            raise ValueError("capture schema/source is invalid")
        if record.get("visibility") != "control_only":
            raise ValueError("capture visibility is invalid")
        index = _number(record.get("frame_index"), integer=True)
        capture = _number(record.get("capture_index"), integer=True)
        round_index = _number(record.get("round_index"), integer=True)
        sim_time = _number(record.get("sim_time_s"))
        sim_step = _number(record.get("sim_step"), integer=True)
        _number(record.get("captured_unix_s"))
        capture_mono = _number(record.get("captured_monotonic_s"))
        if index != self.frame_index + 1 or capture <= self.capture_index or round_index < self.round_index:
            raise ValueError("capture identity/order is invalid")
        if round_index == self.round_index and (sim_time < self.sim_time or sim_step < self.sim_step):
            raise ValueError("capture simulation clock moved backwards")
        if capture_mono < self.capture_mono:
            raise ValueError("capture source clock moved backwards")
        aid = record.get("action_id")
        if aid is not None and (not isinstance(aid, str) or not aid):
            raise ValueError("capture action identity is invalid")
        images = record.get("images")
        cameras = self.manifest.get("camera_panels")
        if not isinstance(images, dict) or set(images) != set(cameras):
            raise ValueError("capture camera set differs from the bound manifest")
        image_data, total = {}, 0
        for camera, image in images.items():
            if not isinstance(image, dict):
                raise ValueError("capture image reference is invalid")
            raw = _read_bytes(_safe_path(self.root, "native/execution-capture/" + str(image.get("path", ""))), MAX_IMAGE_BYTES)
            total += len(raw)
            if total > MAX_IMAGE_BYTES:
                raise ValueError("capture frame exceeds the observer image budget")
            if hashlib.sha256(raw).hexdigest() != _hash(image.get("sha256")):
                raise ValueError("original capture image hash differs")
            if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError("original capture image is not a PNG")
            image_data[camera] = raw
        frame_hash = hashlib.sha256(json.dumps({name: image["sha256"] for name, image in sorted(images.items())},
                                              sort_keys=True).encode()).hexdigest()
        self._remember(self.frames, index, {"record": record, "hash": frame_hash, "data": image_data,
                                           "received": time.monotonic()}, MAX_SERVED_FRAMES)
        if record.get("phase") in {"action_end", "terminal"} and aid is not None:
            self._remember(self.ends, aid, True)
        self.frame_index, self.capture_index, self.round_index, self.sim_time = index, capture, round_index, sim_time
        self.sim_step, self.capture_mono = sim_step, capture_mono

    def poll(self):
        if self.error:
            return
        try:
            if not self.plan:
                try:
                    self.plan = _json(_safe_path(self.root, "plan.json"))
                except FileNotFoundError:
                    pass
                runtime = self.plan.get("runtime") or {}
                self.disabled = (runtime.get("execution_capture") or {}).get("enabled") is False or runtime.get("kind") == "synthetic"
            for event in self.events.read(MAX_EVENTS_PER_POLL):
                self._event(event)
            path = _safe_path(self.root, "native/execution-capture/manifest.json")
            try:
                stat = path.stat()
                stamp = stat.st_mtime_ns, stat.st_size
                if stamp != self.manifest_stamp:
                    manifest = _json(path)
                    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("identity", {}).get("run_id") != self.root.name:
                        raise ValueError("capture manifest schema/run identity differs")
                    cameras = manifest.get("camera_panels")
                    if not isinstance(cameras, list) or not 1 <= len(cameras) <= 4 or any(not isinstance(c, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", c) is None for c in cameras) or len(set(cameras)) != len(cameras):
                        raise ValueError("capture camera manifest is invalid")
                    _number(manifest.get("frame_count"), integer=True)
                    if not isinstance(manifest.get("finalized"), bool):
                        raise ValueError("capture manifest finalization flag is invalid")
                    if self.manifest and cameras != self.manifest["camera_panels"]:
                        raise ValueError("capture manifest camera set changed")
                    self.manifest, self.manifest_stamp = manifest, stamp
            except FileNotFoundError:
                pass
            if self.manifest:
                for frame in self.timeline.read(MAX_FRAMES_PER_POLL):
                    self._frame(frame)
                if self.manifest.get("error"):
                    self.error = "Capture writer: " + str(self.manifest["error"])
            try:
                status = _json(_safe_path(self.root, "status.json"))
                if status.get("state") in {"completed", "incomplete", "invalid", "blocked"}:
                    self.finished = True
            except FileNotFoundError:
                pass
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self.error = str(exc)

    def summary(self) -> dict:
        self.poll()
        pending = bool(self.latest_receipt and self.actions.get(self.latest_receipt, {}).get("expected_end") not in self.ends)
        frame = next(reversed(self.frames.values()), None) if self.frames else None
        if self.error:
            phase, message = "error", self.error
        elif self.disabled:
            phase, message = "unavailable", "Original world capture is disabled or unsupported for this run."
        elif pending and self.manifest.get("finalized") and self.frame_index + 1 >= self.manifest.get("frame_count", 0):
            phase, message = "unavailable", "Original capture ended before the measured action's final camera boundary."
        elif pending:
            phase, message = "awaiting_capture", "Measured result received; waiting for its original camera frames."
        elif frame is None:
            phase, message = ("unavailable", "Run ended before an original camera frame became available.") if self.finished else ("waiting", "Waiting for the simulator's first original camera frame.")
        elif self.finished:
            phase, message = "finished", "Run finished · last recorded world frame."
        elif self.current_action:
            aid = frame["record"].get("action_id")
            if aid == self.current_action or self.parents.get(aid) == self.current_action:
                phase, message = "executing", "Robot executing · original camera frames."
            else:
                phase, message = "awaiting_capture", "Robot executing; waiting for this action's original camera frame."
        else:
            phase, message = "holding", self.source_phase.capitalize() + " · world held between robot actions."
        return {"phase": phase, "status": phase, "message": message, "agent_phase": self.source_phase,
                "current_action_id": self.current_action, "latest_receipt_action_id": self.latest_receipt,
                "capture_pending": pending or phase == "awaiting_capture", "finished": self.finished,
                "error": self.error, "capture_status": self.manifest.get("status"), "frame": frame}


class _QuietHandler(WSGIRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(4.0)

    def log_request(self, code="-", size="-"):
        pass


class WorldViewServer:
    """A loopback-only browser view, bound explicitly to one run at a time."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        if not ipaddress.ip_address(host).is_loopback:
            raise ValueError("world view must bind a loopback address")
        if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
            raise ValueError("world view port is invalid")
        self.host, self.port = host, port
        self.lock = threading.RLock()
        self.observer: _Observer | None = None
        self.binding_id = secrets.token_urlsafe(24)
        self.ack_token = secrets.token_urlsafe(32)
        self.decoded: int | None = None
        self.delivered: OrderedDict[int, set[str]] = OrderedDict()
        self.offered: OrderedDict[int, dict] = OrderedDict()
        self.browser_seen = 0.0
        self.server = self.thread = None
        self.app = Flask(__name__, static_folder=str(Path(__file__).parent / "static"))
        self.app.config["MAX_CONTENT_LENGTH"] = 4096
        self._routes()

    @property
    def url(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"http://{host}:{self.port}/live"

    def start(self) -> "WorldViewServer":
        with self.lock:
            if self.server is not None:
                return self
            self.server = make_server(self.host, self.port, self.app, threaded=True, request_handler=_QuietHandler)
            self.port = self.server.server_port
            self.thread = threading.Thread(target=self.server.serve_forever, name="ora-world-view", daemon=True)
            self.thread.start()
        return self

    def bind(self, root: Path | None) -> None:
        with self.lock:
            self.binding_id, self.ack_token = secrets.token_urlsafe(24), secrets.token_urlsafe(32)
            self.decoded, self.browser_seen = None, 0.0
            self.delivered.clear()
            self.offered.clear()
            self.observer = None if root is None else _Observer(Path(root))

    def status(self) -> dict:
        with self.lock:
            state = self.observer.summary() if self.observer is not None else {
                "phase": "waiting", "status": "waiting", "message": "Choose a task to see its original world cameras.",
                "frame": None, "current_action_id": None, "capture_pending": False, "finished": False, "error": None,
            }
            source = state.pop("frame")
            frame = None
            if source is not None:
                record = source["record"]
                age = time.time() - record["captured_unix_s"]
                frame = {"index": record["frame_index"], "capture_index": record["capture_index"],
                         "round_index": record["round_index"], "action_id": record.get("action_id"),
                         "parent_action_id": self.observer.parents.get(record.get("action_id")),
                         "sim_step": record["sim_step"], "sim_time_s": record["sim_time_s"],
                         "age_s": max(0.0, age) if age >= -5 else None,
                         "received_age_s": max(0.0, time.monotonic() - source["received"]),
                         "hash": source["hash"], "cameras": [
                             {"name": name, "sha256": image["sha256"],
                              "url": f"/api/live/frame/{self.binding_id}/{record['frame_index']}/{source['hash']}/{name}"}
                             for name, image in record["images"].items()]}
            visible = frame is not None and self.decoded == frame["index"]
            return {**state, "url": self.url, "binding_id": self.binding_id,
                    "run_id": self.observer.root.name if self.observer else None,
                    "run_root": str(self.observer.root) if self.observer else None,
                    "frame": frame, "decoded_frame_index": self.decoded, "visible": visible,
                    "connected": self.decoded is not None and time.monotonic() - self.browser_seen < BROWSER_HEARTBEAT_S}

    def close(self) -> None:
        with self.lock:
            server, thread = self.server, self.thread
            self.server = self.thread = None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=2)

    def _routes(self):
        @self.app.before_request
        def local_host():
            # Host validation also rejects DNS rebinding. The forwarded local
            # port may differ, so localhost loopback authorities are allowed.
            try:
                host = urlsplit("//" + request.host).hostname
                valid = host == "localhost" or ipaddress.ip_address(host).is_loopback
            except (ValueError, TypeError):
                valid = False
            if not valid:
                abort(403)

        @self.app.after_request
        def headers(response):
            response.headers["Cache-Control"] = "no-store"
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' blob:; script-src 'self'; style-src 'self'; frame-ancestors 'none'"
            return response

        @self.app.get("/health")
        def health():
            return jsonify(service="ora-live-world", status="ready")

        @self.app.get("/live")
        def page():
            return render_page("ORA world view", TEMPLATES.get_template("live_world.html").render(), wide=True)

        @self.app.get("/api/live/status")
        def status():
            with self.lock:
                self.browser_seen = time.monotonic()
                state = self.status()
                # A frame URI already returned to a browser survives ordinary
                # observer advancement during HTTP transfer and image.decode.
                # At most four frames (32 MiB of original PNGs) are pinned.
                for index in list(self.offered):
                    if self.offered[index]["expires"] < time.monotonic():
                        del self.offered[index]
                if state["frame"] is not None:
                    index = state["frame"]["index"]
                    self.offered[index] = {"frame": self.observer.frames[index], "expires": time.monotonic() + 10.0}
                    self.offered.move_to_end(index)
                    while len(self.offered) > MAX_SERVED_FRAMES:
                        self.offered.popitem(last=False)
                return jsonify(**state, ack_token=self.ack_token)

        @self.app.get("/api/live/frame/<binding>/<int:index>/<frame_hash>/<camera>")
        def image(binding: str, index: int, frame_hash: str, camera: str):
            with self.lock:
                if binding != self.binding_id or self.observer is None:
                    abort(404)
                pin = self.offered.get(index)
                frame = pin["frame"] if pin and pin["expires"] >= time.monotonic() else self.observer.frames.get(index)
                if frame is None or frame["hash"] != frame_hash or camera not in frame["data"]:
                    abort(404)
                self.delivered.setdefault(index, set()).add(camera)
                while len(self.delivered) > MAX_SERVED_FRAMES:
                    self.delivered.popitem(last=False)
                return Response(frame["data"][camera], mimetype="image/png")

        @self.app.post("/api/live/ack")
        def ack():
            if request.headers.get("Origin") != request.host_url.rstrip("/"):
                abort(403)
            value = request.get_json(silent=True)
            if not isinstance(value, dict) or set(value) != {"binding_id", "ack_token", "frame_index", "frame_hash"}:
                abort(400)
            with self.lock:
                if value["binding_id"] != self.binding_id or not isinstance(value["ack_token"], str) or not secrets.compare_digest(value["ack_token"], self.ack_token):
                    abort(409)
                index = value["frame_index"]
                if isinstance(index, bool) or not isinstance(index, int) or self.observer is None:
                    abort(400)
                pin = self.offered.get(index)
                frame = pin["frame"] if pin and pin["expires"] >= time.monotonic() else self.observer.frames.get(index)
                if frame is None or frame["hash"] != value["frame_hash"] or (self.decoded is not None and index < self.decoded) or self.delivered.get(index) != set(frame["data"]):
                    abort(409)
                self.decoded, self.browser_seen = index, time.monotonic()
                return jsonify(ok=True, binding_id=self.binding_id, decoded_frame_index=index)


__all__ = ["WorldViewServer"]
