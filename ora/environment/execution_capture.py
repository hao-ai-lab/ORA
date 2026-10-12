"""Bounded, Control-only recording of pixels from the original execution.

The owner calls capture after a real simulator step. Only image copies happen
on that thread; PNG compression and artifact writes belong to one bounded
writer. Recording never supplies a verdict or advances the environment.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import queue
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping

CONFIG_SCHEMA = "ora.execution-capture-config.v1"
MANIFEST_SCHEMA = "ora.execution-capture-manifest.v1"
FRAME_SCHEMA = "ora.execution-capture-frame.v1"
MAX_FRAME_BYTES = 8 * 1024 * 1024


def resolve_execution_capture(value: Any = None) -> dict[str, Any]:
    defaults = {"schema": CONFIG_SCHEMA, "enabled": True, "fps": 10, "queue_capacity": 32}
    if value is None:
        return defaults
    if not isinstance(value, Mapping) or set(value) - set(defaults):
        raise ValueError("execution_capture has unknown fields or is not an object")
    result = {**defaults, **value}
    if result["schema"] != CONFIG_SCHEMA or not isinstance(result["enabled"], bool):
        raise ValueError("execution_capture schema/enabled is invalid")
    for name, low, high in (("fps", 1, 30), ("queue_capacity", 4, 32)):
        number = result[name]
        if isinstance(number, bool) or not isinstance(number, int) or not low <= number <= high:
            raise ValueError(f"execution_capture.{name} must be in [{low}, {high}]")
    return result


def _atomic(path: Path, data: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


class ExecutionCapture:
    """One recorder, one simulator producer, one bounded asynchronous writer."""

    def __init__(
        self, output_root: Path, *, backend: str, identity: Mapping[str, Any],
        cameras: list[str], config: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = resolve_execution_capture(config)
        self.root = output_root / "execution-capture"
        self.backend = backend
        self.identity = dict(identity)
        self.cameras = list(cameras)
        if not self.cameras or len(self.cameras) > 4 or len(set(self.cameras)) != len(self.cameras):
            raise ValueError("execution capture requires one to four distinct cameras")
        self._queue: queue.Queue = queue.Queue(self.config["queue_capacity"])
        self._stop = threading.Event()
        self._commit_lock = threading.Lock()
        self._closed = False
        self._accepting = bool(self.config["enabled"])
        self._thread: threading.Thread | None = None
        self._next_time: dict[int, float] = {}
        self.attempted = 0
        self.written = 0
        self.dropped: Counter[str] = Counter()
        self.boundaries: Counter[str] = Counter()
        self.rounds: dict[int, Counter[str]] = {}
        self.max_queue_occupancy = 0
        self.capture_wall_s = 0.0
        self.writer_wall_s = 0.0
        self.error: str | None = None
        self._last_manifest_time = 0.0
        if self.config["enabled"]:
            self.root.mkdir(parents=True, exist_ok=False)
            (self.root / "frames").mkdir()
            (self.root / "timeline.jsonl").touch()
            self._manifest(final=False)
            self._thread = threading.Thread(target=self._write_loop, daemon=True, name="execution-capture")
            self._thread.start()

    def due(self, *, round_index: int, sim_time_s: float) -> bool:
        return self._accepting and sim_time_s + 1e-9 >= self._next_time.get(round_index, 0.0)

    def capture(
        self, images: Callable[[], Mapping[str, Any]], *, round_index: int,
        action_id: str | None, action_sequence: int | None, action: str | None,
        sim_step: int, sim_time_s: float, step_unit: str, phase: str,
        force: bool = False,
    ) -> bool:
        if not self._accepting or (not force and not self.due(round_index=round_index, sim_time_s=sim_time_s)):
            return False
        started = time.monotonic()
        sequence = self.attempted
        self.attempted += 1
        self.rounds.setdefault(round_index, Counter())
        self._next_time[round_index] = sim_time_s + 1 / self.config["fps"]
        critical = phase != "control"
        # Leave two queue slots for reset/action/terminal boundaries.
        if not critical and self._queue.qsize() >= self.config["queue_capacity"] - 2:
            self.dropped["queue_full"] += 1
            return False
        try:
            if not math.isfinite(sim_time_s) or sim_step < 0 or round_index < 0:
                raise ValueError("capture clock is invalid")
            import numpy as np

            raw = images()
            copied = {}
            byte_count = 0
            for camera in self.cameras:
                array = np.asarray(raw[camera])
                if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
                    raise ValueError(f"capture camera {camera} is not uint8 RGB")
                byte_count += array.nbytes
                if byte_count > MAX_FRAME_BYTES:
                    raise ValueError("capture frame exceeds bounded pixel budget")
                copied[camera] = array.copy()
            record = {
                "schema": FRAME_SCHEMA, "capture_index": sequence,
                "round_index": round_index, "action_id": action_id,
                "action_sequence": action_sequence, "action": action,
                "sim_step": sim_step, "sim_time_s": sim_time_s,
                "step_unit": step_unit, "phase": phase,
                "captured_unix_s": time.time(), "captured_monotonic_s": started,
                "visibility": "control_only", "source": "original_execution",
            }
            # A saturated writer cannot block the control loop indefinitely.
            self._queue.put((record, copied), block=critical, timeout=0.25 if critical else None)
            self.max_queue_occupancy = max(self.max_queue_occupancy, self._queue.qsize())
            return True
        except queue.Full:
            self.dropped["boundary_queue_full" if critical else "queue_full"] += 1
        except Exception as exc:
            self.dropped["capture_error"] += 1
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.capture_wall_s += time.monotonic() - started
        return False

    @staticmethod
    def _encode(images: Mapping[str, Any]) -> dict[str, bytes]:
        from PIL import Image

        result = {}
        for camera, array in images.items():
            output = io.BytesIO()
            Image.fromarray(array).save(output, format="PNG", compress_level=1)
            result[camera] = output.getvalue()
        return result

    def _write_loop(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                record, images = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue
            started = time.monotonic()
            try:
                encoded = self._encode(images)
                with self._commit_lock:
                    if self._closed:
                        continue
                    paths = {}
                    for index, camera in enumerate(self.cameras):
                        data = encoded[camera]
                        relative = f"frames/{record['capture_index']:08d}-camera-{index}.png"
                        _atomic(self.root / relative, data)
                        paths[camera] = {
                            "path": relative, "sha256": hashlib.sha256(data).hexdigest(),
                            "pixel_sha256": hashlib.sha256(images[camera].tobytes()).hexdigest(),
                            "shape": list(images[camera].shape),
                        }
                    record["images"] = paths
                    record["frame_index"] = self.written
                    with (self.root / "timeline.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record, sort_keys=True) + "\n")
                    self.written += 1
                    self.boundaries[record["phase"]] += 1
                    self.rounds.setdefault(record["round_index"], Counter())[record["phase"]] += 1
                    self.writer_wall_s += time.monotonic() - started
                    if time.monotonic() - self._last_manifest_time >= 1:
                        self._manifest(final=False)
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                self._accepting = False
            finally:
                self._queue.task_done()

    def _manifest(self, *, final: bool) -> dict[str, Any]:
        missing_boundary = any(not counts["reset"] or not counts["terminal"] for counts in self.rounds.values())
        complete = final and self.written == self.attempted and self.written > 0 and not self.error and not missing_boundary
        result = {
            "schema": MANIFEST_SCHEMA, "backend": self.backend,
            "identity": self.identity, "config": self.config,
            "camera_panels": self.cameras, "status": "complete" if complete else "incomplete" if final else "recording",
            "finalized": final, "frame_count": self.written,
            "attempted_frames": self.attempted, "unwritten_frames": self.attempted - self.written,
            "dropped_by_reason": dict(self.dropped), "error": self.error,
            "boundaries": dict(self.boundaries), "round_boundaries": {str(k): dict(v) for k, v in self.rounds.items()},
            "max_queue_occupancy": self.max_queue_occupancy,
            "capture_wall_s": self.capture_wall_s, "writer_wall_s": self.writer_wall_s,
            "timeline": "timeline.jsonl", "visibility": "control_only",
            "capture_semantics": "original_execution_pixels_no_simulator_reconstruction",
            "success_authority": "none",
        }
        _atomic(self.root / "manifest.json", (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
        self._last_manifest_time = time.monotonic()
        return result

    def close(self, *, drain_timeout_s: float = 10.0) -> dict[str, Any] | None:
        if self._closed or not self.config["enabled"]:
            return None
        self._accepting = False
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=drain_timeout_s)
        # A writer still compressing cannot publish after this cutoff/sealing.
        with self._commit_lock:
            self._closed = True
            if self._thread is not None and self._thread.is_alive():
                self.error = self.error or "writer_drain_timeout"
            return self._manifest(final=True)


class PhysicsStepTap:
    """Observe completed steps of one native scene without replacing its identity.

    SAPIEN's instance method is read-only; its Python class supports a scoped
    override. Other scene instances still call the exact original method.
    The isolated worker owns installation and removal around one episode.
    """

    def __init__(self, scene: Any, after_step: Callable[[], None]) -> None:
        self.scene = scene
        self.owner = type(scene)
        self.original = self.owner.step
        self.error: str | None = None

        def step(instance: Any, *args: Any, **kwargs: Any) -> Any:
            result = self.original(instance, *args, **kwargs)
            if instance is self.scene:
                try:
                    after_step()
                except Exception as exc:
                    self.error = f"{type(exc).__name__}: {exc}"
            return result

        self.wrapper = step
        self.owner.step = step

    def close(self) -> None:
        if self.owner.step is self.wrapper:
            self.owner.step = self.original
