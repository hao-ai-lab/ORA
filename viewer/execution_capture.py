"""Read original captured pixels; never construct or advance a simulator."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from ora.environment.execution_capture import FRAME_SCHEMA, MANIFEST_SCHEMA


def capture_status(root: Path) -> dict[str, Any]:
    path = root / "native/execution-capture/manifest.json"
    if not path.is_file():
        return {}
    value = json.loads(path.read_text())
    if value.get("schema") != MANIFEST_SCHEMA:
        raise RuntimeError("original execution capture manifest schema is invalid")
    return value


def capture_ready(root: Path) -> bool:
    value = capture_status(root)
    # While the writer is active, the ordinary original-observation notebook
    # remains available. Continuous video compiles only from a fixed snapshot.
    return value.get("finalized") is True and value.get("frame_count", 0) > 0


def original_execution_source(root: Path) -> dict[str, Any] | None:
    manifest = capture_status(root)
    if not capture_ready(root):
        return None
    source = root / "native/execution-capture"
    timeline = source / "timeline.jsonl"
    raw = timeline.read_bytes()
    records = [json.loads(line) for line in raw.splitlines()]
    if len(records) != manifest["frame_count"]:
        raise RuntimeError("original capture frame count differs from its finalized manifest")
    previous_capture = -1
    rounds: dict[int, list[dict[str, Any]]] = {}
    for index, record in enumerate(records):
        if record.get("schema") != FRAME_SCHEMA or record.get("frame_index") != index:
            raise RuntimeError("original capture timeline is not contiguous")
        capture_index = record.get("capture_index")
        if not isinstance(capture_index, int) or capture_index <= previous_capture:
            raise RuntimeError("original capture attempt order is invalid")
        previous_capture = capture_index
        round_index = record.get("round_index")
        timestamp = record.get("sim_time_s")
        if (not isinstance(round_index, int) or round_index < 0
                or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp) or timestamp < 0):
            raise RuntimeError("original capture simulation clock is invalid")
        group = rounds.setdefault(round_index, [])
        if round_index != max(rounds):
            raise RuntimeError("original capture round order moved backwards")
        if group and timestamp < group[-1]["sim_time_s"]:
            raise RuntimeError("original capture simulation clock moved backwards")
        if set(record.get("images", {})) != set(manifest["camera_panels"]):
            raise RuntimeError("original capture camera set drifted")
        for image in record["images"].values():
            path = (source / image["path"]).resolve()
            if not path.is_relative_to(source.resolve()) or not path.is_file():
                raise RuntimeError("original capture image escaped its evidence root")
            if hashlib.sha256(path.read_bytes()).hexdigest() != image["sha256"]:
                raise RuntimeError("original capture image hash drifted")
        group.append(record)
    fps = manifest["config"]["fps"]
    playback = []
    for round_index, group in rounds.items():
        first, last = group[0]["sim_time_s"], group[-1]["sim_time_s"]
        current = 0
        for tick in range(math.ceil((last - first) * fps) + 1):
            timestamp = min(first + tick / fps, last)
            while current + 1 < len(group) and group[current + 1]["sim_time_s"] <= timestamp + 1e-9:
                current += 1
            playback.append({
                **group[current], "round_count": max(rounds) + 1,
                "cameras": manifest["camera_panels"],
                "playback_sim_time_s": timestamp,
                "frame_age_sim_s": max(0.0, timestamp - group[current]["sim_time_s"]),
                "capture_status": manifest["status"],
                "capture_interval_s": 1 / fps,
            })
    return {
        "root": source, "frames": playback, "frame_count": len(playback),
        "source_frame_count": len(records), "fps": fps,
        "camera_panels": manifest["camera_panels"],
        "kind": "same_session_execution_capture",
        "timeline_path": timeline.relative_to(root).as_posix(),
        "timeline_sha256": hashlib.sha256(raw).hexdigest(),
        "capture_semantics": manifest["capture_semantics"],
        "capture_status": manifest["status"],
        "sampling": {"mode": "original_simulation_clock_v1", "fps": fps,
                     "interpolation": "none", "gaps": "hold_original_frame_with_visible_age"},
    }
