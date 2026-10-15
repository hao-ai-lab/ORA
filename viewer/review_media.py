"""Verify existing execution media and map its frames to recorded rounds/actions."""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from viewer.execution_capture import capture_ready, original_execution_source
from viewer.process_timeline import sha256
from viewer.rounds import integer, iter_records, load_object, round_ledger


@lru_cache(maxsize=512)
def _stat_hash(path: str, identity: tuple[int, ...]) -> str:
    return sha256(Path(path))


def file_hash(path: Path) -> str | None:
    try:
        stat = path.stat()
        return _stat_hash(str(path.resolve()), (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
    except OSError:
        return None


def derived_root(root: Path) -> Path:
    root = root.resolve()
    workspace = root.parent.parent if root.parent.name == "runs" else root.parent
    path = workspace / "viewer-derived" / root.name
    if not path.resolve().is_relative_to((workspace / "viewer-derived").resolve()):
        raise ValueError("Viewer media cache escapes its workspace")
    return path


def _safe_file(parent: Path, relative: Any) -> Path | None:
    if not isinstance(relative, str):
        return None
    path = (parent / relative).resolve()
    return path if path.is_relative_to(parent.resolve()) and path.is_file() else None


def _controller_binding(root: Path, manifest: Mapping[str, Any]) -> bool:
    trace = root / "native/controller-replay/manifest.json"
    if not trace.is_file():
        return not manifest.get("controller_replay")
    expected = manifest.get("controller_replay_manifest_sha256")
    if expected is None:
        # Historical video manifests did not expose this binding separately.
        # Their unchanged seal binds the trace and every command artifact.
        seal = load_object(root / "seal.json")
        expected = next((row.get("sha256") for row in seal.get("inventory", [])
                         if isinstance(row, Mapping) and row.get("path") == "native/controller-replay/manifest.json"), None)
    if expected is None or file_hash(trace) != expected:
        return False
    from ora.environment.controller_replay import (
        ControllerReplayError, load_controller_replay_manifest,
    )

    try:
        entries = load_controller_replay_manifest(trace)["entries"]
    except (ControllerReplayError, OSError, ValueError, TypeError):
        return False
    for entry in entries:
        if not isinstance(entry, Mapping):
            return False
        reference = entry.get("artifact")
        if reference is None and entry.get("replayable") is False:
            continue
        if not isinstance(reference, Mapping):
            return False
        path = _safe_file(root / "native", reference.get("path"))
        if path is None or file_hash(path) != reference.get("sha256"):
            return False
    return True


def find_execution_media(root: Path, trajectory: Mapping[str, Any], *, replay_mode: str = "review") -> dict[str, Any] | None:
    """Reuse verified originals for review; reconstruction requires forensic mode.

    Original capture takes priority once finalized. Its raw frame timeline is
    resampled exactly as the encoder does before deriving video chapters.
    """
    if replay_mode not in {"review", "forensic"}:
        raise ValueError("unsupported Viewer replay mode")
    cache = derived_root(root)
    hashes = trajectory.get("source_hashes") or {}
    if not hashes.get("plan.json"):
        return None
    has_capture = capture_ready(root)
    capture = None
    candidates = sorted(cache.glob(f"trajectory-{replay_mode}-*.json"))
    if replay_mode == "review":
        candidates.extend(sorted(path for path in cache.glob("trajectory-*.json") if path not in candidates))
    for path in candidates:
        manifest = load_object(path)
        if manifest.get("schema") not in {"ora.viewer-trajectory-video.v1", "ora.viewer-trajectory-video.v2"}:
            continue
        if manifest.get("run_id") != root.name or manifest.get("replay_mode") not in {None, replay_mode}:
            continue
        kind = manifest.get("video_source_kind")
        reconstructed = (manifest.get("controller_replay") is True
                         or kind == "viewer_derived_controller_replay"
                         or manifest.get("visual_fidelity") == "unverified_simulator_reconstruction")
        if replay_mode == "review" and (reconstructed or (
            kind not in {"same_session_execution_capture", "same_session_robotwin_control_frames",
                         "original_observation_sequence", "action_boundary_observation_fallback", None}
        ) or (kind is None and manifest.get("continuous_execution"))):
            continue
        if has_capture and kind != "same_session_execution_capture":
            continue
        if kind == "same_session_execution_capture" and not has_capture:
            continue
        if any((manifest.get("source_hashes") or {}).get(name) != value for name, value in hashes.items()):
            continue
        video = _safe_file(cache, path.with_suffix(".mp4").name)
        if video is None or file_hash(video) != manifest.get("video_sha256"):
            continue
        count = integer(manifest.get("frame_count"))
        fps = manifest.get("fps")
        if fps is None and isinstance(manifest.get("seconds_per_frame"), (float, int)) and manifest["seconds_per_frame"] > 0:
            fps = 1 / manifest["seconds_per_frame"]
        if count is None or count <= 0 or not isinstance(fps, (float, int)) or isinstance(fps, bool) or not math.isfinite(fps) or fps <= 0:
            continue
        timeline = None
        if manifest.get("continuous_execution"):
            reference = manifest.get("execution_timeline") or {}
            timeline = _safe_file(cache, reference.get("path")) or _safe_file(root, reference.get("path"))
            if timeline is None or file_hash(timeline) != reference.get("sha256"):
                continue
            if kind == "same_session_execution_capture":
                if manifest.get("execution_capture_manifest_sha256") != file_hash(root / "native/execution-capture/manifest.json"):
                    continue
                if capture is None:
                    capture = original_execution_source(root)
                if (capture is None or count != capture["frame_count"] or fps != capture["fps"]
                        or manifest.get("source_frame_count") != capture["source_frame_count"]
                        or manifest.get("capture_status") != capture["capture_status"]
                        or reference.get("sha256") != capture["timeline_sha256"]):
                    continue
            elif not _controller_binding(root, manifest):
                continue
        elif kind == "same_session_execution_capture":
            continue
        result = {
            "path": video, "manifest_path": path, "manifest": manifest,
            "timeline_path": timeline, "fps": float(fps), "frame_count": count,
            "resolution": manifest.get("resolution"), "sha256": manifest["video_sha256"],
            "source_kind": kind or "action_boundary_observations",
        }
        try:
            result["chapters"] = (
                _capture_chapters(root, trajectory, capture)
                if kind == "same_session_execution_capture"
                else execution_chapters(root, trajectory, result)
            )
        except ValueError:
            continue
        return result
    return None


def _capture_chapters(root: Path, trajectory: Mapping[str, Any], capture: Mapping[str, Any]) -> list[dict]:
    """Keep native action/round identities without inventing policy notebook rows."""
    by_action = {step["action"]["id"]: step for step in trajectory.get("steps", [])
                 if step.get("action") and step["action"].get("id")}
    ledger = round_ledger(root)
    groups: list[dict] = []
    for index, frame in enumerate(capture["frames"]):
        action_id, round_index = frame.get("action_id"), frame["round_index"]
        step = by_action.get(action_id)
        if step is not None and ledger["is_session"]:
            rows = [row for row in ledger["rounds"] if row["round_index"] == round_index]
            bounds = rows[0]["policy_state_steps"] if rows else {}
            start, end = integer(bounds.get("start")), integer(bounds.get("end_exclusive"))
            if start is None or end is None or not start <= step["state_step"] < end:
                raise ValueError("original capture action conflicts with recorded round")
        phase = "execution" if action_id is not None else (
            "round_start" if frame["phase"] in {"reset", "round_reset", "initial"} else frame["phase"]
        )
        if groups and (groups[-1]["action_id"], groups[-1]["round_index"], groups[-1]["phase"]) == (action_id, round_index, phase):
            groups[-1]["end_frame"] = index + 1
            continue
        groups.append({
            "key": f"capture:{round_index}:{index}", "phase": phase,
            "start_frame": index, "end_frame": index + 1,
            "state_step": step.get("state_step") if step is not None else None,
            "round_index": round_index, "action_id": action_id,
            "control_action": frame.get("action"), "capture_phase": frame["phase"],
            "policy_action_mapped": step is not None,
        })
    for group in groups:
        group.update(start_s=group["start_frame"] / capture["fps"], end_s=group["end_frame"] / capture["fps"])
    return groups


def execution_chapters(root: Path, trajectory: Mapping[str, Any], media: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Frame ranges are half-open; round membership is bound by action identity."""
    ledger = round_ledger(root)
    by_step = {step["state_step"]: step for step in trajectory.get("steps", [])}
    by_action = {step["action"]["id"]: step for step in by_step.values() if step.get("action") and step["action"].get("id")}

    def round_for(step: Mapping[str, Any] | None) -> int | None:
        if not ledger["is_session"]:
            return 0
        if step is None:
            return None
        state = step.get("state_step")
        matches = []
        for row in ledger["rounds"]:
            bounds = row["policy_state_steps"]
            start, end = integer(bounds.get("start")), integer(bounds.get("end_exclusive"))
            if start is not None and end is not None and isinstance(state, int) and start <= state < end:
                matches.append(row["round_index"])
        if len(matches) != 1:
            raise ValueError("execution action does not have one recorded round")
        return matches[0]

    groups: list[dict] = []
    if media.get("timeline_path") is None:
        if media["frame_count"] != len(by_step):
            raise ValueError("observation video coverage changed")
        for index, step in enumerate(by_step.values()):
            groups.append({
                "key": f"observation:{step['state_step']}", "phase": "observation",
                "start_frame": index, "end_frame": index + 1, "state_step": step["state_step"],
                "round_index": round_for(step), "action_id": (step.get("action") or {}).get("id"),
            })
    else:
        seen: set[str] = set()
        count = 0
        for _line, frame in iter_records(media["timeline_path"]):
            if integer(frame.get("frame_index")) != count:
                raise ValueError("execution timeline is not contiguous")
            count += 1
            action_id = frame.get("action_id")
            step = by_action.get(action_id) if action_id is not None else None
            if action_id is not None and step is None:
                raise ValueError("execution timeline action is absent from the source trajectory")
            round_index = round_for(step) if step else None
            if integer(frame.get("round_index")) is not None and round_index is not None and frame["round_index"] != round_index:
                raise ValueError("execution timeline round conflicts with the recorded action")
            if groups and action_id is not None and groups[-1]["action_id"] == action_id:
                groups[-1]["end_frame"] = count
                continue
            if action_id is not None and action_id in seen:
                raise ValueError("execution action has disjoint frame ranges")
            if action_id is not None:
                seen.add(action_id)
            elif frame.get("phase") not in {"reset", "round_reset", "initial"}:
                raise ValueError("execution frame lacks a known action or reset boundary")
            groups.append({
                "key": f"action:{action_id}" if action_id else f"reset:{count - 1}",
                "phase": "execution" if action_id else "round_start",
                "start_frame": count - 1, "end_frame": count,
                "state_step": step.get("state_step") if step else None,
                "round_index": round_index, "action_id": action_id,
            })
        if count != media["frame_count"]:
            raise ValueError("execution video and timeline frame counts differ")
        # A reset frame is bound to the immediately following recorded action.
        # Multiple resets before an action are ambiguous, not invented rounds.
        for index, group in enumerate(groups):
            if group["action_id"] is None and index + 1 < len(groups) and groups[index + 1]["action_id"] is not None:
                group["round_index"] = groups[index + 1]["round_index"]
                group["state_step"] = groups[index + 1]["state_step"]
    for group in groups:
        group.update(start_s=group["start_frame"] / media["fps"], end_s=group["end_frame"] / media["fps"])
    return groups
