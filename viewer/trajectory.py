"""Deterministically compile heterogeneous run evidence into one viewer trajectory."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageDraw, ImageFont, ImageOps

from ora.orchestrator.control_audit import audit_libero_run
from viewer.execution_capture import capture_ready, capture_status, original_execution_source
from viewer.process_timeline import tool_images_by_step
from viewer.replay import (
    REPLAY_MODES,
    ViewerReplayError,
    _exclusive_replay_lock,
    ensure_controller_replay,
    replay_identity,
)


class TrajectoryVideoPendingError(RuntimeError):
    """The requested video first needs a simulator-side replay compilation."""


SCHEMA = "ora.viewer-trajectory.v1"
VIDEO_SCHEMA = "ora.viewer-trajectory-video.v1"
CONTINUOUS_VIDEO_SCHEMA = "ora.viewer-trajectory-video.v2"
ROBOTWIN_EXECUTION_FRAME_SCHEMA = (
    "ora.robotwin-operator-execution-frame.v1"
)
CONTROLLER_REPLAY_FRAME_SCHEMA = "ora.viewer-controller-replay-frame.v2"
RAW_SCHEMA = "ora.viewer-raw-trajectory.v1"
RAW_EVENT_SCHEMA = "ora.viewer-raw-trajectory-event.v1"
ANALYSIS_SCHEMA = "ora.viewer-run-analysis.v1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SESSION_ROUNDS_SCHEMA = "ora.viewer-session-rounds.v1"
RAW_STREAMS = {
    "runtime": "events.jsonl",
    "agent": "agent-events.jsonl",
    "transcript": "agent-transcript.jsonl",
}


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(root: Path, value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        return None
    return path.relative_to(root).as_posix()


def _state_images(root: Path, state: Mapping[str, Any]) -> list[dict[str, Any]]:
    images: list[dict[str, Any]] = []
    cameras = state.get("cameras")
    if not isinstance(cameras, Mapping):
        return images
    main = str(state.get("main_camera") or "")
    for name, camera in cameras.items():
        if not isinstance(camera, Mapping):
            continue
        files = camera.get("files")
        rgb = files.get("rgb_lossless_png") if isinstance(files, Mapping) else None
        relative = _artifact(root, rgb.get("path") if isinstance(rgb, Mapping) else None)
        if relative:
            images.append({
                "label": str(name), "path": relative, "kind": "rgb",
                "main": name == main, "visibility": "policy_visible",
                "sha256": _sha256(root / relative),
            })
    return sorted(images, key=lambda item: (not item["main"], item["label"]))


def _observation_images(
    root: Path, event: Mapping[str, Any], state: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Resolve current state images, then the pre-policy-state historical layout."""
    images = _state_images(root, state)
    if images:
        return images
    observation = state.get("observation") if isinstance(state.get("observation"), Mapping) else {}
    observation_id = event.get("observation_id", observation.get("id"))
    if not isinstance(observation_id, str):
        return []
    directory = (root / "policy-observations" / observation_id).resolve()
    if not directory.is_relative_to(root) or not directory.is_dir():
        return []
    context = event.get("context") if isinstance(event.get("context"), Mapping) else {}
    main = str(state.get("main_camera") or context.get("main_camera") or "")
    for camera in sorted(path for path in directory.iterdir() if path.is_dir()):
        rgb = next(
            (camera / name for name in ("rgb.png", "rgb-thumbnail.png", "rgb_lossless.png")
             if (camera / name).is_file()),
            None,
        )
        relative = _artifact(root, rgb.relative_to(root).as_posix()) if rgb else None
        if relative:
            images.append({
                "label": camera.name, "path": relative, "kind": "rgb",
                "main": camera.name == main, "visibility": "policy_visible",
                "sha256": _sha256(root / relative),
            })
    return sorted(images, key=lambda item: (not item["main"], item["label"]))


def _operator_image(root: Path, index: int) -> dict[str, Any] | None:
    directory = root / "native/operator-world-frames"
    if not directory.is_dir():
        return None
    matches = sorted(directory.glob(f"frame-{index:04d}*.png"))
    if not matches:
        return None
    relative = _artifact(root, matches[0].relative_to(root).as_posix())
    if not relative:
        return None
    return {
        "label": "operator world", "path": relative, "kind": "operator",
        "main": False, "visibility": "control_only",
        "sha256": _sha256(root / relative),
    }


def _grasp_plan_images(
    root: Path, event: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Resolve policy and review plan previews without crossing visibility."""

    query_ref = event.get("query_ref")
    valid_query_ref = bool(
        isinstance(query_ref, str)
        and query_ref.startswith("grasp-query-")
        and len(query_ref) == len("grasp-query-") + 24
        and all(character in "0123456789abcdef" for character in query_ref[12:])
    )
    count = event.get("candidate_count")
    suffix = f" · {count} candidates" if isinstance(count, int) else ""
    images: list[dict[str, Any]] = []

    policy_artifact = (
        event.get("policy_overlay_artifact")
        if isinstance(event.get("policy_overlay_artifact"), Mapping) else {}
    )
    policy_relative = _artifact(root, policy_artifact.get("path"))
    policy_expected = policy_artifact.get("sha256")
    if policy_relative is None and valid_query_ref:
        legacy = (
            root / "openeta-tool-artifacts" / "grasp-queries"
            / query_ref / "overlay.png"
        )
        if legacy.is_file():
            policy_relative = _artifact(root, legacy.relative_to(root).as_posix())
            policy_expected = event.get("overlay_sha256")
    if policy_relative is not None:
        policy_actual = _sha256(root / policy_relative)
        if not isinstance(policy_expected, str) or policy_actual == policy_expected:
            images.append({
                "label": f"grasp.plan policy contact sheet{suffix}",
                "path": policy_relative,
                "kind": "grasp_plan",
                "main": False,
                "visibility": "policy_visible",
                "sha256": policy_actual,
                "query_ref": query_ref,
            })

    review_artifact = (
        event.get("artifact") if isinstance(event.get("artifact"), Mapping) else {}
    )
    review_relative = _artifact(root, review_artifact.get("path"))
    if review_relative is not None:
        review_actual = _sha256(root / review_relative)
        review_expected = review_artifact.get("sha256")
        if not isinstance(review_expected, str) or review_actual == review_expected:
            images.append({
                "label": f"grasp.plan global input review{suffix}",
                "path": review_relative,
                "kind": "grasp_plan",
                "main": False,
                "visibility": "review_only",
                "sha256": review_actual,
                "query_ref": query_ref,
            })
    return images


def _grounding_image(
    root: Path, event: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Resolve the exact policy-visible text-grounding overlay by hash."""

    grounding_ref = event.get("grounding_ref")
    prefix = "grounding-query-"
    valid_ref = bool(
        isinstance(grounding_ref, str)
        and grounding_ref.startswith(prefix)
        and len(grounding_ref) == len(prefix) + 24
        and all(character in "0123456789abcdef" for character in grounding_ref[len(prefix):])
    )
    if not valid_ref:
        return None
    overlay = (
        root / "openeta-tool-artifacts" / "grounding-queries"
        / grounding_ref / "overlay.png"
    )
    if not overlay.is_file():
        return None
    relative = _artifact(root, overlay.relative_to(root).as_posix())
    if relative is None:
        return None
    actual = _sha256(root / relative)
    expected = event.get("overlay_sha256")
    if isinstance(expected, str) and actual != expected:
        return None
    count = event.get("candidate_count")
    suffix = f" · {count} targets" if isinstance(count, int) else ""
    camera = event.get("camera")
    camera_suffix = f" · {camera}" if isinstance(camera, str) else ""
    return {
        "label": f"vision.ground{camera_suffix}{suffix}",
        "path": relative,
        "kind": "grounding",
        "main": False,
        "visibility": "policy_visible",
        "sha256": actual,
        "grounding_ref": grounding_ref,
    }


def _grounding_images(
    root: Path, event: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Resolve the overview and every exact policy-visible candidate card."""

    overview = _grounding_image(root, event)
    if overview is None:
        return []
    images = [overview]
    grounding_ref = overview["grounding_ref"]
    query_root = (
        root / "openeta-tool-artifacts" / "grounding-queries" / grounding_ref
    )
    camera = event.get("camera")
    camera_suffix = f" · {camera}" if isinstance(camera, str) else ""
    previews = event.get("candidate_previews")
    if not isinstance(previews, list):
        return images
    for preview in previews:
        if not isinstance(preview, Mapping):
            continue
        rank = preview.get("rank")
        artifact = preview.get("artifact")
        if not isinstance(rank, int) or not isinstance(artifact, Mapping):
            continue
        relative_inside_query = artifact.get("path")
        if not isinstance(relative_inside_query, str):
            continue
        candidate_path = (query_root / relative_inside_query).resolve()
        if not candidate_path.is_relative_to(query_root.resolve()):
            continue
        relative = _artifact(root, candidate_path.relative_to(root).as_posix())
        if relative is None:
            continue
        actual = _sha256(root / relative)
        expected = artifact.get("sha256")
        if isinstance(expected, str) and actual != expected:
            continue
        images.append({
            "label": f"vision.ground T{rank} candidate enlargement{camera_suffix}",
            "path": relative,
            "kind": "grounding_candidate",
            "main": False,
            "visibility": "policy_visible",
            "sha256": actual,
            "grounding_ref": grounding_ref,
            "target_ref": preview.get("target_ref"),
            "rank": rank,
        })
    return images


def _agent_messages(call: Mapping[str, Any]) -> list[str]:
    messages: list[str] = []
    stdout = call.get("stdout")
    if not isinstance(stdout, str):
        return messages
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        item = event.get("item") if isinstance(event, Mapping) else None
        if (
            event.get("type") == "item.completed"
            and isinstance(item, Mapping)
            and item.get("type") == "agent_message"
            and isinstance(item.get("text"), str)
        ):
            messages.append(item["text"])
    return messages


def _action_summary(action: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(action, Mapping):
        return None
    arguments = action.get("arguments") if isinstance(action.get("arguments"), Mapping) else {}
    pose = arguments.get("pose") if isinstance(arguments.get("pose"), Mapping) else {}
    targets = arguments.get("targets") if isinstance(arguments.get("targets"), Mapping) else {}
    first_target = next(
        (value for value in targets.values() if isinstance(value, Mapping)), {}
    )
    target_pose = (
        first_target.get("target_pose")
        if isinstance(first_target.get("target_pose"), Mapping) else first_target
    )
    return {
        "id": action.get("action_id"),
        "kind": action.get("action"),
        "position_m": pose.get(
            "position_m", arguments.get("position_m", target_pose.get("position_m"))
        ),
        "orientation_xyzw": pose.get(
            "orientation_xyzw",
            arguments.get("orientation_xyzw", target_pose.get("orientation_xyzw")),
        ),
        "gripper": arguments.get(
            "gripper",
            {
                arm: value for arm, value in targets.items()
                if isinstance(value, str)
            } or None,
        ),
        "target_arms": list(targets),
        "arguments": arguments,
    }


def _receipt_summary(receipt: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(receipt, Mapping):
        return None
    nested = receipt.get("receipt")
    if isinstance(nested, Mapping):
        receipt = nested
    phases = receipt.get("phase_receipts") if isinstance(receipt.get("phase_receipts"), list) else []
    statuses = [
        {"phase": phase.get("phase"), "status": phase.get("status")}
        for phase in phases if isinstance(phase, Mapping)
    ]
    reached = receipt.get("reached", receipt.get("pose_reached"))
    execution_status = receipt.get("execution_status")
    if reached is None and execution_status in {"reached", "executed"}:
        reached = True
    elif reached is None and execution_status in {
        "not_reached", "planner_failed", "execution_fault", "not_executed",
    }:
        reached = False
    if reached is None and statuses:
        reached = all(item["status"] in {"reached", "executed", "completed", "not_requested"} for item in statuses)
    health = (
        receipt.get("control_health")
        if isinstance(receipt.get("control_health"), Mapping) else {}
    )
    arms = receipt.get("arms") if isinstance(receipt.get("arms"), Mapping) else {}
    arm_summaries = {
        str(arm): {
            "targeted": value.get("targeted"),
            "planner": value.get("planner"),
            "pose_error": value.get("pose_error"),
            "inactive_arm_drift": value.get("inactive_arm_drift"),
            "hold_controller": value.get("hold_controller"),
        }
        for arm, value in arms.items() if isinstance(value, Mapping)
    }
    targeted_errors = [
        value.get("pose_error") for value in arm_summaries.values()
        if value.get("targeted") is True and isinstance(value.get("pose_error"), Mapping)
    ]
    sparse_error = max(
        targeted_errors,
        key=lambda value: float(value.get("translation_m") or 0.0),
        default=None,
    )
    return {
        "action_id": receipt.get("action_id"),
        "consumed": receipt.get("consumed"),
        "reached": reached,
        "pose_reached": receipt.get("pose_reached"),
        "pose_residual": receipt.get("pose_residual"),
        "pose_error": receipt.get("pose_error", sparse_error),
        "planner": receipt.get("planner", {
            arm: value.get("planner") for arm, value in arm_summaries.items()
        } if arm_summaries else None),
        "status": receipt.get("status", execution_status),
        "execution_status": execution_status,
        "target_arms": receipt.get("target_arms"),
        "arms": arm_summaries,
        "synchronization": receipt.get("synchronization"),
        "settle_steps": receipt.get("settle_steps"),
        "controller_tracking": receipt.get("controller_tracking"),
        "terminal_after_action": receipt.get("terminal_after_action"),
        "aperture_mm": receipt.get("aperture_mm"),
        "reached_gripper": receipt.get("reached_gripper"),
        "state_changed": receipt.get("state_changed"),
        "steps_used": receipt.get("steps_used"),
        "phases": statuses,
        "control_health": {
            "status": health.get("status"),
            "codes": health.get("codes", []),
            "target_error_before_m": health.get("target_error_before_m"),
            "target_error_after_m": health.get("target_error_after_m"),
            "target_orientation_error_before_rad": health.get(
                "target_orientation_error_before_rad"
            ),
            "target_orientation_error_after_rad": health.get(
                "target_orientation_error_after_rad"
            ),
        } if health else None,
    }


def _round_index(rounds: list[dict[str, Any]], state_step: Any) -> int | None:
    if not isinstance(state_step, int):
        return None
    for item in rounds:
        bounds = item.get("policy_state_steps")
        if not isinstance(bounds, Mapping):
            continue
        start = bounds.get("start")
        end = bounds.get("end_exclusive")
        if isinstance(start, int) and isinstance(end, int) and start <= state_step < end:
            index = item.get("round_index")
            return index if isinstance(index, int) else None
    return None


def _round_projection(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    source = result.get("rounds")
    if not isinstance(source, list):
        return []
    rounds: list[dict[str, Any]] = []
    for ordinal, value in enumerate(source):
        if not isinstance(value, Mapping):
            continue
        bounds = value.get("policy_state_steps")
        rounds.append({
            "round_index": (
                value.get("round_index")
                if isinstance(value.get("round_index"), int) else ordinal
            ),
            "official_success": value.get("official_success"),
            "state": value.get("state"),
            "termination": value.get("termination"),
            "decisions": value.get("decisions"),
            "refused_proposals": value.get("refused_proposals"),
            "model_calls": value.get("model_calls"),
            "wall_duration_s": value.get("wall_duration_s"),
            "policy_state_steps": dict(bounds) if isinstance(bounds, Mapping) else {},
        })
    return rounds


def _analysis_candidates() -> list[Path]:
    return sorted((PROJECT_ROOT / "experiments").glob("*/viewer-analysis.json"))


def _load_run_analysis(
    root: Path, plan: Mapping[str, Any], *, candidates: list[Path] | None = None,
) -> dict[str, Any] | None:
    """Load one tracked, exact-source review overlay without touching run evidence."""

    source = plan.get("source") if isinstance(plan.get("source"), Mapping) else {}
    source_commit = source.get("commit")
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in candidates if candidates is not None else _analysis_candidates():
        value = _load(path)
        if (
            value.get("schema") == ANALYSIS_SCHEMA
            and value.get("run_id") == root.name
            and value.get("source_commit") == source_commit
        ):
            matches.append((path, value))
    if len(matches) > 1:
        raise ValueError(f"multiple Viewer analyses match run {root.name}")
    if not matches:
        return None
    path, value = matches[0]
    projected = dict(value)
    try:
        projected["source_path"] = path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        projected["source_path"] = str(path)
    projected["source_sha256"] = _sha256(path)
    projected["projection"] = "curated_navigation_only"
    return projected


def _session_round_projection(
    result: Mapping[str, Any], steps: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Project explicit persistent-session round evidence onto Viewer steps.

    A multi-round session intentionally has no aggregate environment verdict.
    The environment-owned Boolean remains attached to each round, while this
    projection makes the reset boundaries reviewable without changing the run.
    """
    session = result.get("session")
    rounds = result.get("rounds")
    if not isinstance(session, Mapping) or not isinstance(rounds, list) or not rounds:
        return None

    projected: list[dict[str, Any]] = []
    issues: list[str] = []
    previous_end = 0
    for expected_index, value in enumerate(rounds):
        if not isinstance(value, Mapping):
            issues.append(f"round_{expected_index}_not_object")
            continue
        raw_index = value.get("round_index")
        index = raw_index if isinstance(raw_index, int) and not isinstance(raw_index, bool) else expected_index
        bounds = value.get("policy_state_steps")
        start = bounds.get("start") if isinstance(bounds, Mapping) else None
        end = bounds.get("end_exclusive") if isinstance(bounds, Mapping) else None
        if (
            not isinstance(start, int) or isinstance(start, bool)
            or not isinstance(end, int) or isinstance(end, bool)
            or start < 0 or end <= start
        ):
            issues.append(f"round_{index}_invalid_step_range")
            start = end = None
        elif start < previous_end:
            issues.append(f"round_{index}_overlapping_step_range")
        if isinstance(end, int):
            previous_end = max(previous_end, end)
        verdict = value.get("official_success")
        projected.append({
            "round_index": index,
            "round_ordinal": index + 1,
            "state": value.get("state"),
            "termination": value.get("termination"),
            "official_success": verdict if isinstance(verdict, bool) else None,
            "decisions": value.get("decisions"),
            "refused_proposals": value.get("refused_proposals"),
            "model_calls": value.get("model_calls"),
            "sim_steps": (
                value.get("verdict", {}).get("sim_steps")
                if isinstance(value.get("verdict"), Mapping) else None
            ),
            "wall_duration_s": value.get("wall_duration_s"),
            "started_unix_s": value.get("started_unix_s"),
            "finished_unix_s": value.get("finished_unix_s"),
            "policy_state_steps": {
                "start": start, "end_exclusive": end,
            },
            "step_count": end - start if isinstance(start, int) and isinstance(end, int) else None,
        })

    for round_item in projected:
        bounds = round_item["policy_state_steps"]
        start, end = bounds["start"], bounds["end_exclusive"]
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        members = [
            step for step in steps
            if isinstance(step.get("state_step"), int)
            and start <= int(step["state_step"]) < end
        ]
        if len(members) != end - start:
            issues.append(f"round_{round_item['round_index']}_step_coverage_mismatch")
        for local_index, step in enumerate(members):
            step["round"] = {
                "round_index": round_item["round_index"],
                "round_ordinal": round_item["round_ordinal"],
                "round_count": len(projected),
                "local_step_index": local_index,
                "local_step_count": len(members),
            }

    recorded_count = result.get("round_count")
    recorded_successes = result.get("round_success_count")
    derived_successes = sum(item["official_success"] is True for item in projected)
    if isinstance(recorded_count, int) and recorded_count != len(projected):
        issues.append("round_count_mismatch")
    if isinstance(recorded_successes, int) and recorded_successes != derived_successes:
        issues.append("round_success_count_mismatch")
    return {
        "schema": SESSION_ROUNDS_SCHEMA,
        "cadence": session.get("cadence"),
        "reset_owner": session.get("reset_owner"),
        "budget_scope": session.get("budget_scope"),
        "agent_session_continuity": "persistent",
        "environment_session_continuity": "reset_between_rounds",
        "round_count": len(projected),
        "round_success_count": derived_successes,
        "success_within_k": any(item["official_success"] is True for item in projected),
        "all_rounds_succeeded": bool(projected) and derived_successes == len(projected),
        "aggregate_official_success": None,
        "rounds": projected,
        "integrity": {
            "status": "valid" if not issues else "warning",
            "issues": issues,
        },
    }


def _raw_timestamp(record: Mapping[str, Any]) -> float | None:
    for name in ("unix_s", "time", "started", "finished"):
        value = record.get(name)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _short(value: Any, *, width: int = 110) -> str:
    text = str(value or "").replace("\n", " ").strip()
    return textwrap.shorten(text, width=width, placeholder="…") if text else ""


def _raw_summary(record: Mapping[str, Any], event_type: str) -> str:
    if event_type == "observation":
        return _short(record.get("observation_id") or "fresh actual observation")
    if event_type in {"tool_result", "tool_error"}:
        tool = record.get("tool") or record.get("payload", {}).get("name")
        outcome = "result" if event_type == "tool_result" else "error"
        return f"{tool or 'tool'} · {outcome}"
    if event_type == "tool_call_proposed":
        payload = record.get("payload") if isinstance(record.get("payload"), Mapping) else {}
        return f"{payload.get('name', 'tool')} · {payload.get('effect', 'effect unknown')}"
    if event_type == "provider_response":
        payload = record.get("payload") if isinstance(record.get("payload"), Mapping) else {}
        message = payload.get("message") if isinstance(payload.get("message"), Mapping) else {}
        calls = message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []
        names = [
            call.get("function", {}).get("name") for call in calls
            if isinstance(call, Mapping) and isinstance(call.get("function"), Mapping)
        ]
        content = _short(message.get("content"), width=84)
        suffix = f" · tools {', '.join(str(name) for name in names if name)}" if names else ""
        return (content or str(payload.get("finish_reason") or "provider response")) + suffix
    if event_type == "provider_request":
        payload = record.get("payload") if isinstance(record.get("payload"), Mapping) else {}
        return f"{payload.get('provider_protocol', 'provider')} · {payload.get('model', 'model')}"
    if event_type in {"action_submitted", "action_proposed"}:
        action = record.get("action") if isinstance(record.get("action"), Mapping) else {}
        payload = record.get("payload") if isinstance(record.get("payload"), Mapping) else {}
        return str(action.get("action") or payload.get("action_id") or "world mutation")
    if event_type == "motion_receipt":
        value = record.get("record") if isinstance(record.get("record"), Mapping) else {}
        nested = value.get("receipt")
        if isinstance(nested, Mapping):
            value = nested
        phases = value.get("phase_receipts") if isinstance(value.get("phase_receipts"), list) else []
        statuses = [
            str(phase.get("status")) for phase in phases if isinstance(phase, Mapping)
        ]
        return " · ".join(statuses) or str(value.get("status") or "environment receipt")
    if event_type == "context_projected":
        payload = record.get("payload") if isinstance(record.get("payload"), Mapping) else {}
        return f"{payload.get('estimated_input_tokens', '—')} estimated input tokens"
    if event_type == "artifact":
        return "transcript record"
    return _short(
        record.get("message") or record.get("reason") or record.get("schema")
        or event_type
    )


def _raw_linkage(record: Mapping[str, Any]) -> dict[str, Any]:
    payload = record.get("payload") if isinstance(record.get("payload"), Mapping) else {}
    action = record.get("action") if isinstance(record.get("action"), Mapping) else {}
    receipt = record.get("record") if isinstance(record.get("record"), Mapping) else {}
    return {
        "event_id": record.get("event_id"),
        "causal_parent_id": record.get("causal_parent_id"),
        "model_call_id": record.get("model_call_id"),
        "tool_call_id": record.get("tool_call_id"),
        "action_id": (
            record.get("action_id") or action.get("action_id")
            or receipt.get("action_id") or payload.get("action_id")
        ),
        "observation_id": record.get("observation_id") or payload.get("observation_id"),
        "visibility": record.get("visibility"),
    }


def _raw_search_text(record: Mapping[str, Any]) -> str:
    parts: list[str] = []
    skipped = {"images_base64", "image_base64", "data", "blob", "image_url"}

    def collect(value: Any, key: str = "") -> None:
        if sum(len(part) for part in parts) >= 1800 or key in skipped:
            return
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                parts.append(str(child_key))
                collect(child, str(child_key))
        elif isinstance(value, list):
            for child in value[:24]:
                collect(child, key)
        elif isinstance(value, str):
            if value.startswith("data:image/"):
                return
            parts.append(value[:500])
        elif value is not None:
            parts.append(str(value))

    collect(record)
    return " ".join(parts)[:2000]


def _raw_artifacts(root: Path, value: Any) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    seen: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            relative = _artifact(root, item.get("path"))
            if relative and relative not in seen:
                seen.add(relative)
                artifacts.append({
                    "path": relative,
                    "sha256": item.get("sha256") or _sha256(root / relative),
                    "bytes": item.get("bytes", (root / relative).stat().st_size),
                    "kind": item.get("kind"),
                    "media_type": item.get("media_type"),
                })
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return artifacts


def _decode_raw_line(line_bytes: bytes) -> tuple[str, dict[str, Any] | None, str]:
    raw_bytes = line_bytes.rstrip(b"\r\n")
    raw = raw_bytes.decode("utf-8", errors="replace")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = None
    record = value if isinstance(value, dict) else None
    return raw, record, hashlib.sha256(line_bytes).hexdigest()


def _raw_lines(path: Path) -> list[tuple[str, dict[str, Any] | None, str]]:
    if not path.is_file():
        return []
    return [
        _decode_raw_line(line_bytes)
        for line_bytes in path.read_bytes().splitlines(keepends=True)
    ]


def _raw_line(
    path: Path, source_line: int,
) -> tuple[str, dict[str, Any] | None, str] | None:
    if not path.is_file():
        return None
    with path.open("rb") as source:
        for index, line_bytes in enumerate(source, start=1):
            if index == source_line:
                return _decode_raw_line(line_bytes)
    return None


def compile_raw_trajectory(root: Path) -> dict[str, Any]:
    """Index original JSONL records without copying full payloads into the index."""

    root = root.resolve()
    trajectory = compile_trajectory(root)
    plan = _load(root / "plan.json")
    result = _load(root / "result.json")
    launch = _load(root / "launch.json")
    sources: list[dict[str, Any]] = []
    indexed: list[dict[str, Any]] = []
    runtime_call_steps: dict[int, int] = {}

    for stream, relative in RAW_STREAMS.items():
        path = root / relative
        lines = _raw_lines(path)
        sources.append({
            "stream": stream, "path": relative, "present": path.is_file(),
            "line_count": len(lines), "bytes": path.stat().st_size if path.is_file() else 0,
            "sha256": _sha256(path),
        })
        current_runtime_step: int | None = None
        observation_ordinal = -1
        for source_line, (raw, record, line_sha256) in enumerate(lines, start=1):
            event_type = str(record.get("type")) if record else "invalid_json"
            if stream == "runtime":
                if event_type == "observation":
                    observation_ordinal += 1
                    state_ref = record.get("policy_state") if record else None
                    state_step = state_ref.get("step") if isinstance(state_ref, Mapping) else None
                    current_runtime_step = (
                        state_step if isinstance(state_step, int) else observation_ordinal
                    )
                observation_step = current_runtime_step
                call = record.get("call") if record else None
                if isinstance(call, int) and current_runtime_step is not None:
                    runtime_call_steps[call] = current_runtime_step
            elif stream == "agent":
                step = record.get("observation_step") if record else None
                observation_step = step if isinstance(step, int) else None
            else:
                call = record.get("call") if record else None
                observation_step = runtime_call_steps.get(call) if isinstance(call, int) else None
                event_type = event_type if event_type != "None" else "transcript_record"
                if event_type == "invalid_json" and record:
                    event_type = "transcript_record"
            linkage = _raw_linkage(record or {})
            round_index = _round_index(trajectory.get("rounds", []), observation_step)
            key = f"{stream}:{source_line}"
            indexed.append({
                "key": key,
                "stream": stream,
                "source_path": relative,
                "source_line": source_line,
                "line_sha256": line_sha256,
                "type": event_type,
                "timestamp_unix_s": _raw_timestamp(record or {}),
                "observation_step": observation_step,
                "round_index": round_index,
                "summary": _raw_summary(record or {}, event_type),
                "search_text": _raw_search_text(record or {}) or raw[:500],
                "artifact_count": len(_raw_artifacts(root, record or {})),
                "linkage": linkage,
                "error": (
                    event_type == "invalid_json"
                    or any(token in event_type.lower() for token in (
                        "error", "failed", "fault", "rejected", "stagnation",
                    ))
                ),
            })

    stream_order = {name: index for index, name in enumerate(RAW_STREAMS)}
    indexed.sort(key=lambda item: (
        item["timestamp_unix_s"] is None,
        item["timestamp_unix_s"] if item["timestamp_unix_s"] is not None else 0,
        stream_order[item["stream"]], item["source_line"],
    ))
    for index, item in enumerate(indexed):
        item["index"] = index
    timestamps = [
        item["timestamp_unix_s"] for item in indexed
        if item["timestamp_unix_s"] is not None
    ]
    cycles = [{
        "step": step.get("index"),
        "state_step": step.get("state_step"),
        "round_index": step.get("round_index"),
        "observation_id": step.get("observation_id"),
        "images": step.get("images", []),
        "after_images": step.get("after_images", []),
        "remaining_budget": step.get("remaining_budget"),
        "action": step.get("action"),
        "receipt": step.get("receipt"),
        "tool_count": len(step.get("tools", [])),
        "model_call_count": len(step.get("model_calls", [])),
        "warnings": step.get("warnings", []),
    } for step in trajectory.get("steps", [])]
    return {
        "schema": RAW_SCHEMA,
        "run_id": root.name,
        "experiment": plan.get("experiment"),
        "config": plan.get("config"),
        "milestone": plan.get("milestone"),
        "task": plan.get("task"),
        "seed": plan.get("seed"),
        "evidence_tier": plan.get("evidence_tier"),
        "source_commit": (
            plan.get("source", {}).get("commit")
            if isinstance(plan.get("source"), Mapping) else None
        ),
        "instruction": trajectory.get("instruction"),
        "summary": {
            "state": result.get("state"),
            "termination": result.get("termination"),
            "official_success": result.get("official_success"),
            "control_validity": trajectory.get("summary", {}).get(
                "control_validity"
            ),
            "control_issues": trajectory.get("summary", {}).get(
                "control_issues", []
            ),
            "event_count": len(indexed),
            "observation_cycles": len(cycles),
            "run_started_unix_s": result.get(
                "started_unix_s", launch.get("created_unix_s")
            ),
            "start_unix_s": min(timestamps) if timestamps else None,
            "end_unix_s": max(timestamps) if timestamps else None,
        },
        "sources": sources,
        "events": indexed,
        "cycles": cycles,
        "rounds": trajectory.get("rounds", []),
        "analysis": trajectory.get("analysis"),
        "projection": (
            "read_only_timestamp_merge; source path, line, line hash and stream order "
            "remain authoritative"
        ),
    }


def raw_trajectory_event(root: Path, stream: str, source_line: int) -> dict[str, Any]:
    """Return one exact source line plus its parsed record and safe artifact links."""

    root = root.resolve()
    relative = RAW_STREAMS.get(stream)
    if relative is None or source_line < 1:
        raise ValueError("invalid raw trajectory event identity")
    path = root / relative
    line = _raw_line(path, source_line)
    if line is None:
        raise ValueError("raw trajectory event does not exist")
    raw, record, line_sha256 = line
    return {
        "schema": RAW_EVENT_SCHEMA,
        "run_id": root.name,
        "key": f"{stream}:{source_line}",
        "stream": stream,
        "source_path": relative,
        "source_line": source_line,
        "source_sha256": _sha256(path),
        "line_sha256": line_sha256,
        "raw_json": raw,
        "record": record,
        "artifacts": _raw_artifacts(root, record or {}),
    }


def compile_trajectory(root: Path) -> dict[str, Any]:
    """Compile a read-only presentation projection; canonical evidence is untouched."""
    root = root.resolve()
    plan = _load(root / "plan.json")
    result = _load(root / "result.json")
    launch = _load(root / "launch.json")
    status = _load(root / "status.json")
    evaluation = _load(root / "evaluation.json")
    rounds = _round_projection(result)
    analysis = _load_run_analysis(root, plan)
    events = _jsonl(root / "events.jsonl")
    transcript = _jsonl(root / "agent-transcript.jsonl")
    agent_events = _jsonl(root / "agent-events.jsonl")
    returned_images = tool_images_by_step(root, agent_events)
    calls = {int(item["call"]): item for item in transcript if isinstance(item.get("call"), int)}
    native_requests = {
        str(item.get("model_call_id")): item for item in agent_events
        if item.get("type") == "provider_request" and item.get("model_call_id")
    }

    manifest = _load(root / "policy-state/manifest.json")
    state_records = manifest.get("steps") if isinstance(manifest.get("steps"), list) else []
    states: dict[int, dict[str, Any]] = {}
    for record in state_records:
        if not isinstance(record, Mapping) or not isinstance(record.get("step"), int):
            continue
        relative = record.get("path")
        path = (root / relative).resolve() if isinstance(relative, str) else root / "missing"
        if path.is_relative_to(root) and path.is_file():
            states[record["step"]] = _load(path)

    observation_indexes = [i for i, event in enumerate(events) if event.get("type") == "observation"]
    if not observation_indexes and states:
        # Older and small Synthetic runs may have a complete policy-state chain
        # without outer observation events. Reconstruct only the public causal
        # projection already present in adjacent immutable states.
        events = []
        ordered = sorted(states)
        for position, state_step in enumerate(ordered):
            state = states[state_step]
            observation = state.get("observation") if isinstance(state.get("observation"), Mapping) else {}
            events.append({
                "type": "observation", "observation_id": observation.get("id"),
                "policy_state": {"step": state_step},
            })
            if position + 1 < len(ordered):
                following = states[ordered[position + 1]]
                action = following.get("previous_public_action")
                receipt = following.get("previous_execution_receipt")
                if isinstance(action, Mapping):
                    events.append({"type": "action_submitted", "action": action})
                if isinstance(receipt, Mapping):
                    events.append({"type": "motion_receipt", "record": receipt})
        observation_indexes = [i for i, event in enumerate(events) if event.get("type") == "observation"]
    steps: list[dict[str, Any]] = []
    for ordinal, start in enumerate(observation_indexes):
        end = observation_indexes[ordinal + 1] if ordinal + 1 < len(observation_indexes) else len(events)
        group = events[start:end]
        observation_event = events[start]
        state_ref = observation_event.get("policy_state")
        state_step = state_ref.get("step") if isinstance(state_ref, Mapping) else ordinal
        state = states.get(state_step, {}) if isinstance(state_step, int) else {}
        action_event = next((event for event in group if event.get("type") == "action_submitted"), None)
        receipt_event = next((event for event in group if event.get("type") == "motion_receipt"), None)
        action = action_event.get("action") if isinstance(action_event, Mapping) else None
        receipt = receipt_event.get("record") if isinstance(receipt_event, Mapping) else None

        tools = []
        derived_images: list[dict[str, Any]] = []
        model_calls = []
        warnings = []
        for event in group:
            kind = event.get("type")
            if kind in {"tool_result", "tool_error"}:
                tools.append({
                    "tool": event.get("tool"), "ok": kind == "tool_result",
                    "arguments": event.get("arguments"),
                    "result": event.get("result") if kind == "tool_result" else event.get("error"),
                })
                if kind == "tool_error":
                    warnings.append("tool_error")
            elif kind == "openeta_view_derived":
                artifact = event.get("artifact") if isinstance(event.get("artifact"), Mapping) else {}
                relative = _artifact(root, artifact.get("path"))
                if relative:
                    identity = event.get("identity") if isinstance(event.get("identity"), Mapping) else {}
                    derived_images.append({"label": identity.get("camera", "derived"), "path": relative, "kind": identity.get("kind", "derived"), "main": False})
            elif kind == "grasp_plan_materialized":
                derived_images.extend(_grasp_plan_images(root, event))
            elif kind == "grounding_query_materialized":
                derived_images.extend(_grounding_images(root, event))
            elif kind == "agent_session_step_completed" and isinstance(event.get("call"), int):
                call = calls.get(event["call"], {})
                model_calls.append({
                    "call": event["call"], "message_kind": event.get("message_kind"),
                    "latency_s": event.get("latency_s"), "usage": event.get("usage"),
                    "messages": _agent_messages(call),
                })
            elif kind in {
                "action_rejected_stagnation", "action_stagnation_advisory",
            }:
                warnings.append(kind)
        stagnation = next((event.get("record") for event in group if event.get("type") == "stagnation_observation"), None)
        # `warning` was retired from the record; runs sealed before that still
        # carry it, and the Viewer reads heterogeneous history. Keep the branch.
        if isinstance(stagnation, Mapping) and stagnation.get("warning"):
            warnings.append("stagnation")

        native_step = state_step if isinstance(state_step, int) else ordinal
        for provider_event in agent_events:
            if (
                provider_event.get("type") != "provider_response"
                or provider_event.get("observation_step") != native_step
            ):
                continue
            payload = provider_event.get("payload") if isinstance(provider_event.get("payload"), Mapping) else {}
            message = payload.get("message") if isinstance(payload.get("message"), Mapping) else {}
            content = message.get("content")
            request_event = native_requests.get(str(provider_event.get("model_call_id")), {})
            started = request_event.get("unix_s")
            finished = provider_event.get("unix_s")
            model_calls.append({
                "call": provider_event.get("model_call_id"),
                "message_kind": "native_provider_response",
                "latency_s": (
                    finished - started
                    if isinstance(started, (int, float)) and isinstance(finished, (int, float))
                    else None
                ),
                "usage": payload.get("usage"),
                "messages": [content] if isinstance(content, str) and content else [],
                "reasoning": message.get("reasoning_content"),
            })

        if not model_calls and ordinal in calls:
            call = calls[ordinal]
            model_calls.append({
                "call": ordinal, "message_kind": "historical_turn_order",
                "latency_s": (
                    call.get("finished") - call.get("started")
                    if isinstance(call.get("finished"), (int, float))
                    and isinstance(call.get("started"), (int, float)) else None
                ),
                "usage": call.get("usage"), "messages": _agent_messages(call),
            })

        next_state = states.get(state_step + 1, {}) if isinstance(state_step, int) else {}
        images = _observation_images(root, observation_event, state) + derived_images
        image_hashes = {image.get("sha256") for image in images if image.get("sha256")}
        for image in returned_images.get(state_step, []):
            if image["sha256"] not in image_hashes:
                images.append(image)
                image_hashes.add(image["sha256"])
        operator = _operator_image(root, ordinal)
        if operator:
            images.append(operator)
        receipt_summary = _receipt_summary(receipt)
        health = (
            receipt_summary.get("control_health")
            if isinstance(receipt_summary, Mapping) else None
        )
        if isinstance(health, Mapping) and health.get("status") not in {None, "healthy"}:
            warnings.extend(str(code) for code in health.get("codes", []) if code)
        steps.append({
            "index": ordinal,
            "state_step": state_step,
            "round_index": _round_index(rounds, state_step),
            "observation_id": observation_event.get("observation_id"),
            "cause": state.get("cause"),
            "images": images,
            "after_images": _state_images(root, next_state),
            "proprioception": state.get("proprioception"),
            "remaining_budget": state.get("remaining_budget"),
            "model_calls": model_calls,
            "tools": tools,
            "action": _action_summary(action),
            "receipt": receipt_summary,
            "stagnation": stagnation,
            "warnings": warnings,
        })

    session_rounds = _session_round_projection(result, steps)
    source_files = ["plan.json", "events.jsonl", "agent-transcript.jsonl", "agent-events.jsonl", "policy-state/manifest.json", "result.json", "evaluation.json", "seal.json"]
    control_audit = audit_libero_run(root)
    return {
        "schema": SCHEMA,
        "run_id": root.name,
        "experiment": plan.get("experiment"),
        "config": plan.get("config"),
        "milestone": plan.get("milestone"),
        "task": plan.get("task"),
        "seed": plan.get("seed"),
        "evidence_tier": plan.get("evidence_tier"),
        "source_commit": (
            plan.get("source", {}).get("commit")
            if isinstance(plan.get("source"), Mapping) else None
        ),
        "instruction": next((state.get("task_instruction") for state in states.values() if state.get("task_instruction")), None),
        "policy": plan.get("policy"),
        "summary": {
            "state": status.get("state", result.get("state")),
            "termination": result.get("termination", status.get("termination")),
            "official_success": result.get("official_success", status.get("official_success")),
            "control_validity": control_audit["control_validity"],
            "control_issues": control_audit.get("issues", []),
            "evaluation": evaluation.get("status"),
            "decisions": result.get("decisions"),
            "refused_proposals": result.get("refused_proposals"),
            "model_calls": result.get("model_calls"),
            "wall_duration_s": result.get("wall_duration_s"),
            "run_started_unix_s": result.get(
                "started_unix_s", launch.get("created_unix_s")
            ),
        },
        "rounds": rounds,
        "analysis": analysis,
        "view_kind": (
            "persistent_multi_round_session" if session_rounds is not None
            else "single_episode"
        ),
        "session_rounds": session_rounds,
        "steps": steps,
        "source_hashes": {name: _sha256(root / name) for name in source_files},
        "video_source": {
            "continuous_execution": capture_ready(root) or (
                root / "native/operator-execution-frames/timeline.jsonl"
            ).is_file(),
            "kind": (
                "same_session_execution_capture" if capture_ready(root) else
                "same_session_robotwin_control_frames"
                if (root / "native/operator-execution-frames/timeline.jsonl").is_file()
                else "original_observation_sequence"
            ),
            "fidelity": "original_frames_only",
            "capture_status": capture_status(root).get("status"),
            "unrecorded_motion": "not_reconstructed",
        },
        "projection": "read_only_derived",
    }


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    names = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_wrapped(
    draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, *,
    width: int, font: ImageFont.ImageFont, fill: str, spacing: int = 5,
) -> int:
    lines: list[str] = []
    for paragraph in value.splitlines() or [""]:
        lines.extend(textwrap.wrap(paragraph, width=width) or [""])
    text = "\n".join(lines)
    draw.multiline_text(xy, text, font=font, fill=fill, spacing=spacing)
    box = draw.multiline_textbbox(xy, text, font=font, spacing=spacing)
    return box[3]


def _render_trajectory_frame(root: Path, trajectory: Mapping[str, Any], index: int) -> Image.Image:
    """Render one evidence-labelled frame without inferring task success."""
    steps = trajectory.get("steps") if isinstance(trajectory.get("steps"), list) else []
    step = steps[index]
    canvas = Image.new("RGB", (1280, 720), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    mono = _font(17)
    small = _font(14)
    tiny = _font(12)
    title = _font(24, bold=True)
    label = _font(13, bold=True)
    blue = "#256a8f"
    red = "#a33a3a"
    green = "#2d724c"
    ink = "#111111"
    muted = "#666666"
    line = "#cccccc"
    soft = "#f5f5f5"
    draw.rectangle((0, 0, 1279, 70), fill="#111111")
    round_info = step.get("round") if isinstance(step.get("round"), Mapping) else None
    if round_info:
        position = (
            f"ROUND {round_info.get('round_ordinal')}/{round_info.get('round_count')}  ·  "
            f"STEP {round_info.get('local_step_index', 0):02d}/"
            f"{max(0, int(round_info.get('local_step_count') or 1)-1):02d}"
        )
    else:
        position = f"STEP {index:02d}/{max(0, len(steps)-1):02d}"
    draw.text((24, 13), position, font=label, fill="#bbbbbb")
    heading = str(trajectory.get("instruction") or trajectory.get("task") or trajectory.get("run_id"))
    draw.text((24, 34), textwrap.shorten(heading, width=88, placeholder="…"), font=title, fill="#ffffff")

    images = [
        item for item in step.get("images", [])
        if isinstance(item, Mapping) and item.get("kind") in {"rgb", "operator"}
    ] if isinstance(step.get("images"), list) else []
    selected: list[Mapping[str, Any] | None] = [
        next((item for item in images if item.get("label") == camera and item.get("kind") == "rgb"), None)
        for camera in ("agentview", "wrist")
    ]
    if all(item is None for item in selected):
        main = next((item for item in images if item.get("main") is True), None)
        operator = next((item for item in images if item.get("kind") == "operator"), None)
        remainder = [item for item in images if item is not main and item is not operator]
        selected = [main or (remainder.pop(0) if remainder else None), operator or (remainder.pop(0) if remainder else None)]
    for slot, item in enumerate(selected):
        left = 24 + slot * 426
        box = (left, 94, left + 402, 526)
        draw.rectangle(box, fill=soft, outline=line, width=2)
        camera_label = str(item.get("label")) if isinstance(item, Mapping) else f"camera {slot+1}"
        visibility = str(item.get("visibility") or "") if isinstance(item, Mapping) else ""
        draw.text((left + 10, 102), camera_label.upper(), font=label, fill=ink)
        if visibility:
            draw.text((left + 240, 104), visibility.upper(), font=tiny, fill=muted)
        relative = item.get("path") if isinstance(item, Mapping) else None
        path = (root / relative).resolve() if isinstance(relative, str) else root / "missing"
        if path.is_relative_to(root) and path.is_file():
            try:
                if isinstance(item, Mapping) and item.get("sha256") != _sha256(path):
                    raise RuntimeError("original observation image hash drifted during rendering")
                with Image.open(path) as source:
                    rendered = ImageOps.contain(source.convert("RGB"), (382, 386), Image.Resampling.LANCZOS)
                x = left + 10 + (382 - rendered.width) // 2
                y = 132 + (386 - rendered.height) // 2
                canvas.paste(rendered, (x, y))
            except OSError:
                draw.text((left + 18, 180), "IMAGE DECODE FAILED", font=small, fill=red)
        else:
            draw.text((left + 18, 180), "NO CAMERA ARTIFACT", font=small, fill=muted)

    panel = (876, 94, 1256, 646)
    draw.rectangle(panel, fill=soft, outline=line, width=2)
    y = 110
    action = step.get("action") if isinstance(step.get("action"), Mapping) else None
    receipt = step.get("receipt") if isinstance(step.get("receipt"), Mapping) else None
    draw.text((894, y), "ACTION AFTER THIS OBSERVATION", font=label, fill=muted); y += 25
    if action:
        draw.text((894, y), str(action.get("kind") or "mutation"), font=mono, fill=ink); y += 27
        position = action.get("position_m")
        if isinstance(position, list):
            draw.text((894, y), "xyz  " + "  ".join(f"{float(v):+.3f}" for v in position), font=small, fill=ink); y += 23
        if action.get("gripper") is not None:
            draw.text((894, y), f"grip {action.get('gripper')}", font=small, fill=ink); y += 23
    else:
        draw.text((894, y), "no mutation", font=small, fill=muted); y += 25
    y += 10
    draw.text((894, y), "RECEIPT AFTER THAT ACTION", font=label, fill=muted); y += 25
    if receipt:
        reached = receipt.get("reached")
        status_color = green if reached is True else red if reached is False else muted
        draw.text((894, y), "REACHED" if reached is True else "NOT REACHED" if reached is False else "UNKNOWN", font=mono, fill=status_color); y += 27
        for phase in receipt.get("phases", []) if isinstance(receipt.get("phases"), list) else []:
            draw.text((894, y), f"{phase.get('phase')}: {phase.get('status')}", font=tiny, fill=ink); y += 20
        residual = receipt.get("pose_residual")
        error = receipt.get("pose_error")
        translation = (
            residual.get("translation_residual_m") if isinstance(residual, Mapping)
            else error.get("translation_m") if isinstance(error, Mapping) else None
        )
        if isinstance(translation, (int, float)):
            draw.text((894, y), f"residual {float(translation)*1000:.1f} mm", font=small, fill=status_color); y += 23
    else:
        draw.text((894, y), "no environment receipt", font=small, fill=muted); y += 24
    y += 12
    tools = step.get("tools") if isinstance(step.get("tools"), list) else []
    draw.text((894, y), "TOOLS", font=label, fill=muted); y += 23
    names = [str(item.get("tool") or "unknown") + ("" if item.get("ok") else " !") for item in tools]
    y = _draw_wrapped(draw, (894, y), ", ".join(names) or "none", width=38, font=tiny, fill=ink) + 15
    calls = step.get("model_calls") if isinstance(step.get("model_calls"), list) else []
    messages = [message for call in calls for message in call.get("messages", [])]
    if messages and y < 570:
        draw.text((894, y), "AGENT", font=label, fill=muted); y += 22
        _draw_wrapped(draw, (894, y), textwrap.shorten(messages[-1], width=220, placeholder="…"), width=42, font=tiny, fill=ink)

    draw.rectangle((0, 670, 1279, 719), fill="#111111")
    draw.text(
        (24, 634),
        "ORIGINAL OBSERVATIONS · 1.4 s / snapshot · motion between snapshots was not recorded",
        font=small, fill=blue,
    )
    remaining = step.get("remaining_budget") if isinstance(step.get("remaining_budget"), Mapping) else {}
    observation = textwrap.shorten(str(step.get("observation_id") or "unknown observation"), width=75, placeholder="…")
    draw.text((24, 686), observation, font=tiny, fill=muted)
    if round_info:
        draw.text((620, 686), "SAME AGENT SESSION · ENV RESET BETWEEN ROUNDS", font=tiny, fill="#d6a15f")
    draw.text((930, 686), f"{remaining.get('robot_decisions', '—')} decisions left", font=small, fill="#ffffff")
    draw.text((1175, 686), "1.4 s", font=tiny, fill="#bbbbbb")
    return canvas


def _continuous_execution_source(root: Path) -> dict[str, Any] | None:
    """Resolve same-session RoboTwin control frames without replaying the run."""

    shared = original_execution_source(root)
    if shared is not None:
        return shared
    timeline = root / "native/operator-execution-frames/timeline.jsonl"
    if not timeline.is_file():
        return None
    frames: list[dict[str, Any]] = []
    for expected, record in enumerate(_jsonl(timeline)):
        if (
            record.get("schema") != ROBOTWIN_EXECUTION_FRAME_SCHEMA
            or record.get("frame_index") != expected
        ):
            raise RuntimeError("RoboTwin execution timeline is not contiguous")
        path_value = record.get("path")
        relative = _artifact(root, path_value)
        # The backend's output root is ``<run>/native``.  Its timeline paths
        # are therefore correctly relative to that directory, while older
        # fixtures and imported runs may already store run-relative paths.
        if relative is None and isinstance(path_value, str):
            native_relative = (Path("native") / path_value).as_posix()
            relative = _artifact(root, native_relative)
        if relative is None:
            raise RuntimeError(f"RoboTwin execution frame {expected} is unavailable")
        frames.append({**record, "path": relative})
    if not frames:
        return None
    return {
        "root": root,
        "timeline_path": timeline.relative_to(root).as_posix(),
        "timeline_sha256": _sha256(timeline),
        "frame_count": len(frames),
        "camera_panels": list(frames[0].get("cameras") or []),
        "frames": frames,
        "capture_semantics": (
            "same_environment_session_actual_physics_control_frames"
        ),
        "kind": "same_session_robotwin_control_frames",
    }


def _controller_replay_source(
    root: Path, *, replay_mode: str = "review",
) -> dict[str, Any] | None:
    """Compile and resolve a derived same-source controller replay."""

    source_manifest = root / "native/controller-replay/manifest.json"
    if not source_manifest.is_file():
        return None
    replay_root, replay = ensure_controller_replay(root, replay_mode=replay_mode)
    timeline_value = replay.get("timeline")
    relative = (
        timeline_value.get("path") if isinstance(timeline_value, Mapping) else None
    )
    if not isinstance(relative, str):
        raise RuntimeError("controller replay timeline is unavailable")
    timeline = (replay_root / relative).resolve()
    if not timeline.is_relative_to(replay_root) or not timeline.is_file():
        raise RuntimeError("controller replay timeline escaped its derived workspace")
    sampling = replay.get("sampling")
    if not isinstance(sampling, Mapping):
        raise RuntimeError("controller replay sampling profile is unavailable")
    sampling = dict(sampling)
    sampling_mode = sampling.get("mode")
    if sampling_mode == "action_keyframes_v1":
        limit = sampling.get("max_frames_per_action")
        hold_steps = sampling.get("hold_steps_per_sample")
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 32
        ):
            raise RuntimeError("controller replay action sample limit is invalid")
        if (
            isinstance(hold_steps, bool)
            or not isinstance(hold_steps, int)
            or not 1 <= hold_steps <= 64
        ):
            raise RuntimeError("controller replay action hold limit is invalid")
        expected_result_semantics = (
            "derived_same_source_seed_controller_replay_action_keyframes"
        )
        expected_frame_semantics = "derived_controller_replay_action_keyframes"
    elif sampling_mode == "physics_stride_v1":
        stride = sampling.get("capture_stride")
        if (
            isinstance(stride, bool)
            or not isinstance(stride, int)
            or not 1 <= stride <= 64
        ):
            raise RuntimeError("controller replay capture stride is invalid")
        expected_result_semantics = (
            "derived_same_source_seed_controller_replay"
            if stride == 1
            else "derived_same_source_seed_controller_replay_temporally_subsampled"
        )
        expected_frame_semantics = (
            "derived_controller_replay_after_simulator_step"
            if stride == 1
            else "derived_controller_replay_temporally_subsampled"
        )
    else:
        raise RuntimeError("controller replay sampling mode is invalid")
    if replay.get("capture_semantics") != expected_result_semantics:
        raise RuntimeError("controller replay capture semantics are inconsistent")
    frames: list[dict[str, Any]] = []
    for expected, record in enumerate(_jsonl(timeline)):
        if (
            record.get("schema") != CONTROLLER_REPLAY_FRAME_SCHEMA
            or record.get("frame_index") != expected
        ):
            raise RuntimeError("controller replay timeline is not contiguous")
        if (
            record.get("sampling") != sampling
            or record.get("capture_semantics") != expected_frame_semantics
        ):
            raise RuntimeError("controller replay frame sampling metadata drifted")
        frame = (replay_root / str(record.get("path", ""))).resolve()
        if not frame.is_relative_to(replay_root) or not frame.is_file():
            raise RuntimeError(f"controller replay frame {expected} is unavailable")
        frames.append(dict(record))
    if not frames:
        raise RuntimeError("controller replay produced no frames")
    return {
        "root": replay_root,
        "timeline_path": timeline.relative_to(replay_root).as_posix(),
        "timeline_sha256": _sha256(timeline),
        "source_manifest_sha256": _sha256(source_manifest),
        "frame_count": len(frames),
        "camera_panels": list(replay.get("camera_panels") or []),
        "frames": frames,
        "sampling": sampling,
        "capture_semantics": replay.get("capture_semantics"),
        "kind": "viewer_derived_controller_replay",
    }


def _execution_step_map(
    trajectory: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], int]:
    steps = trajectory.get("steps") if isinstance(trajectory.get("steps"), list) else []
    by_action: dict[str, Mapping[str, Any]] = {}
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        action = step.get("action")
        action_id = action.get("id") if isinstance(action, Mapping) else None
        if isinstance(action_id, str):
            by_action[action_id] = step
    return by_action, len(by_action)


def _format_vector(value: Any, *, scale: float = 1.0, digits: int = 3) -> str:
    if not isinstance(value, list) or not all(
        isinstance(item, (int, float)) for item in value
    ):
        return "—"
    return "  ".join(f"{float(item) * scale:+.{digits}f}" for item in value)


def _render_continuous_execution_frame(
    root: Path, trajectory: Mapping[str, Any], record: Mapping[str, Any],
    *, source_root: Path, source_kind: str,
    action_steps: Mapping[str, Mapping[str, Any]], action_count: int,
) -> Image.Image:
    """Add action and measured-outcome context to one actual control frame."""

    canvas = Image.new("RGB", (1920, 960), "#121715")
    if source_kind == "same_session_execution_capture":
        panels = list(record["images"].items())
        panel_height = 438 if len(panels) > 2 else 880
        for slot, (camera, item) in enumerate(panels):
            source_path = (source_root / item["path"]).resolve()
            if not source_path.is_relative_to(source_root.resolve()) or _sha256(source_path) != item["sha256"]:
                raise RuntimeError("original capture image identity drifted during rendering")
            with Image.open(source_path) as source:
                rendered = ImageOps.contain(source.convert("RGB"), (630, panel_height - 28), Image.Resampling.LANCZOS)
            column, row = slot % 2, slot // 2
            left = column * 640 + (640 - rendered.width) // 2
            canvas.paste(rendered, (left, 48 + row * panel_height + (panel_height - rendered.height) // 2))
            ImageDraw.Draw(canvas).text((column * 640 + 14, 14 + row * panel_height), camera.upper(), font=_font(18), fill="white")
    else:
        relative = record.get("path")
        source_path = (source_root / str(relative)).resolve()
        if not source_path.is_relative_to(source_root) or not source_path.is_file():
            raise RuntimeError("continuous execution frame escapes its source")
        with Image.open(source_path) as source:
            montage = ImageOps.contain(source.convert("RGB"), (1280, 960), Image.Resampling.LANCZOS)
        canvas.paste(montage, ((1280 - montage.width) // 2, (960 - montage.height) // 2))
    draw = ImageDraw.Draw(canvas)
    mono = _font(18)
    small = _font(15)
    tiny = _font(13)
    title = _font(25, bold=True)
    label = _font(14, bold=True)
    ink = "#f4f0e7"
    muted = "#98a49f"
    orange = "#e5a15c"
    blue = "#61b3d4"
    green = "#72c493"
    red = "#ef7969"
    draw.rectangle((1280, 0, 1919, 959), fill="#121715")
    draw.line((1280, 0, 1280, 959), fill="#53615b", width=2)

    action_id = record.get("action_id")
    step = action_steps.get(str(action_id)) if isinstance(action_id, str) else None
    action = step.get("action") if isinstance(step, Mapping) and isinstance(step.get("action"), Mapping) else None
    if action is None and isinstance(action_id, str) and source_kind == "same_session_execution_capture":
        # A native skill sub-action need not have a policy-level notebook row.
        # Show only its recorded identity, without inventing a policy receipt.
        action = {"kind": record.get("action"), "id": action_id}
    receipt = step.get("receipt") if isinstance(step, Mapping) and isinstance(step.get("receipt"), Mapping) else None
    sequence = record.get("action_sequence")
    ordinal = int(sequence) + 1 if isinstance(sequence, int) and isinstance(action_id, str) else None
    phase = str(record.get("phase") or "unknown")
    control_step = record.get("control_step_index")

    derived_replay = source_kind == "viewer_derived_controller_replay"
    sampling = record.get("sampling") if isinstance(record.get("sampling"), Mapping) else {}
    sampling_mode = sampling.get("mode")
    capture_stride = sampling.get("capture_stride", 1)
    frame_limit = sampling.get("max_frames_per_action")
    draw.text(
        (1310, 24),
        (
            f"DERIVED CONTROLLER REPLAY · ACTION KEYFRAMES ≤{frame_limit}"
            if derived_replay and sampling_mode == "action_keyframes_v1"
            else
            f"DERIVED CONTROLLER REPLAY · TEMPORALLY SUBSAMPLED ×{capture_stride}"
            if derived_replay and sampling_mode == "physics_stride_v1" and capture_stride > 1
            else "DERIVED CONTROLLER REPLAY"
            if derived_replay else "ACTUAL EXECUTION"
        ),
        font=label, fill=blue,
    )
    round_info = step.get("round") if isinstance(step, Mapping) else None
    round_info = round_info if isinstance(round_info, Mapping) else {}
    round_index = record.get("round_index", round_info.get("round_index"))
    round_count = record.get("round_count", round_info.get("round_count"))
    round_actions = [value for value in action_steps.values()
                     if isinstance(value.get("round"), Mapping)
                     and value["round"].get("round_index") == round_index]
    round_action_sequence = record.get("round_action_sequence")
    round_action_count = record.get("round_action_count")
    if round_actions and step in round_actions:
        round_action_sequence = round_actions.index(step) if round_action_sequence is None else round_action_sequence
        round_action_count = len(round_actions) if round_action_count is None else round_action_count
    round_prefix = (
        f"ROUND {int(round_index) + 1}/{int(round_count)} · "
        if isinstance(round_index, int) and isinstance(round_count, int) else ""
    )
    if ordinal is not None:
        local = (
            f"CONTROL ACTION {ordinal:02d}"
            if source_kind == "same_session_execution_capture" else
            f"ACTION {int(round_action_sequence) + 1:02d}/{int(round_action_count):02d}"
            if isinstance(round_action_sequence, int)
            and isinstance(round_action_count, int) else
            f"ACTION {ordinal:02d}/{action_count:02d}"
        )
        position = round_prefix + local
    else:
        position = round_prefix + (
            "RESET" if phase in {"reset", "round_reset"} else phase.replace("_", " ").upper()
        )
    draw.text((1310, 52), position, font=title, fill=ink)
    phase_text = phase if control_step is None else f"{phase} · control {control_step}"
    draw.text((1310, 88), phase_text.upper(), font=small, fill=muted)
    y = 132
    draw.text((1310, y), "CURRENT ACTION", font=label, fill=orange); y += 30
    if action:
        draw.text((1310, y), str(action.get("kind") or "mutation"), font=title, fill=ink); y += 38
        arms = action.get("target_arms")
        if isinstance(arms, list) and arms:
            draw.text((1310, y), "arms  " + " + ".join(str(arm) for arm in arms), font=small, fill=ink); y += 27
        arguments = action.get("arguments") if isinstance(action.get("arguments"), Mapping) else {}
        targets = arguments.get("targets") if isinstance(arguments.get("targets"), Mapping) else {}
        if not targets:
            arm = arguments.get("arm")
            if isinstance(arm, str):
                draw.text((1310, y), f"arm  {arm}", font=small, fill=ink); y += 27
            position_m = arguments.get("position_m")
            if isinstance(position_m, list):
                draw.text((1310, y), f"xyz  {_format_vector(position_m)}", font=tiny, fill=ink); y += 22
            orientation = arguments.get("orientation_xyzw")
            if isinstance(orientation, list):
                draw.text((1310, y), "quat  " + _format_vector(orientation), font=tiny, fill=muted); y += 22
            if arguments.get("gripper") is not None:
                draw.text((1310, y), f"gripper  {arguments['gripper']}", font=small, fill=ink); y += 25
        for arm, target in targets.items():
            if isinstance(target, str):
                draw.text((1310, y), f"{arm} gripper  {target}", font=small, fill=ink); y += 25
                continue
            if not isinstance(target, Mapping):
                continue
            pose = target.get("target_pose") if isinstance(target.get("target_pose"), Mapping) else target
            draw.text((1310, y), f"{arm} xyz  {_format_vector(pose.get('position_m'))}", font=tiny, fill=ink); y += 22
            orientation = pose.get("orientation_xyzw")
            if isinstance(orientation, list):
                draw.text((1310, y), "quat  " + _format_vector(orientation), font=tiny, fill=muted); y += 22
    else:
        draw.text((1310, y), str(record.get("action") or ("reset" if phase in {"reset", "round_reset"} else "no active action")), font=mono, fill=muted); y += 30

    y += 16
    draw.text(
        (1310, y),
        "ORIGINAL RUN RECEIPT · NOT REPLAY RESULT"
        if derived_replay else "MEASURED OUTCOME",
        font=label, fill=orange,
    ); y += 30
    if receipt:
        recorded_execution = (
            record.get("recorded_execution")
            if isinstance(record.get("recorded_execution"), Mapping) else {}
        )
        status = str(
            receipt.get("execution_status")
            or receipt.get("status")
            or recorded_execution.get("status")
            or ("executed" if receipt.get("consumed") is True else None)
            or "unknown"
        )
        status_color = green if status in {"reached", "executed"} else red if status in {"not_reached", "planner_failed", "execution_fault", "not_executed"} else muted
        draw.text((1310, y), status.upper(), font=title, fill=status_color); y += 38
        arms = receipt.get("arms") if isinstance(receipt.get("arms"), Mapping) else {}
        for arm, value in arms.items():
            if not isinstance(value, Mapping):
                continue
            planner = value.get("planner") if isinstance(value.get("planner"), Mapping) else {}
            if value.get("targeted") is True:
                draw.text(
                    (1310, y),
                    f"{arm} planner  {planner.get('status', '—')} · {planner.get('path_steps', 0)} points",
                    font=tiny, fill=ink,
                ); y += 22
                error = value.get("pose_error") if isinstance(value.get("pose_error"), Mapping) else {}
                translation = error.get("translation_m")
                orientation = error.get("orientation_rad")
                if isinstance(translation, (int, float)):
                    detail = f"error  {float(translation)*1000:.1f} mm"
                    if isinstance(orientation, (int, float)):
                        detail += f" · {float(orientation)*180/3.141592653589793:.1f}°"
                    draw.text((1310, y), detail, font=tiny, fill=status_color); y += 22
            drift = value.get("inactive_arm_drift") if isinstance(value.get("inactive_arm_drift"), Mapping) else {}
            if value.get("targeted") is False and isinstance(drift.get("translation_m"), (int, float)):
                draw.text(
                    (1310, y),
                    f"{arm} inactive drift  {float(drift['translation_m'])*1000:.3f} mm",
                    font=tiny, fill=muted,
                ); y += 22
        settle = receipt.get("settle_steps")
        if isinstance(settle, int):
            draw.text((1310, y), f"settle steps  {settle}", font=tiny, fill=muted); y += 22
        steps_used = receipt.get("steps_used")
        if isinstance(steps_used, int):
            draw.text((1310, y), f"controller steps  {steps_used}", font=tiny, fill=muted); y += 22
        if receipt.get("terminal_after_action") is True:
            draw.text(
                (1310, y), "ORIGINAL RUN TERMINAL AFTER ACTION",
                font=label, fill=green,
            ); y += 26
    else:
        draw.text((1310, y), "No matching policy receipt" if source_kind == "same_session_execution_capture" else "available after action completion", font=small, fill=muted); y += 28

    calls = step.get("model_calls") if isinstance(step, Mapping) and isinstance(step.get("model_calls"), list) else []
    messages = [message for call in calls for message in call.get("messages", [])]
    if messages and y < 790:
        y += 14
        draw.text((1310, y), "AGENT CONTEXT", font=label, fill=orange); y += 26
        _draw_wrapped(
            draw, (1310, y), textwrap.shorten(messages[-1], width=250, placeholder="…"),
            width=58, font=tiny, fill=ink, spacing=4,
        )

    progress = 0.0 if ordinal is None or not action_count else ordinal / action_count
    draw.rectangle((1310, 878, 1890, 886), fill="#37413d")
    draw.rectangle((1310, 878, 1310 + int(580 * progress), 886), fill=orange)
    panels = record.get("cameras") if isinstance(record.get("cameras"), list) else []
    draw.text(
        (1310, 904), f"{len(panels)} synchronized cameras · control-only",
        font=tiny, fill=muted,
    )
    draw.text(
        (1310, 928),
        (
            "full controller execution · sampled frames · unverified reconstruction"
            if derived_replay and sampling_mode == "action_keyframes_v1"
            else "same source + seed · approximate timing · not original evidence"
            if derived_replay and sampling_mode == "physics_stride_v1" and capture_stride > 1
            else "same source + seed · derived, not original evidence"
            if derived_replay else "same simulator execution · not replayed"
        ),
        font=tiny, fill=blue,
    )
    if source_kind == "same_session_execution_capture":
        age = float(record.get("frame_age_sim_s", 0))
        detail = (f"ORIGINAL CAPTURE · round {int(record['round_index']) + 1} · "
                  f"simulation {record['playback_sim_time_s']:.3f} s · source step {record['sim_step']}")
        if record.get("capture_status") != "complete":
            detail += " · INCOMPLETE RECORDING"
        if age > 1.5 * float(record.get("capture_interval_s", 0.1)):
            detail += f" · FRAME GAP: last image {age:.2f} s old"
        draw.rectangle((0, 925, 1279, 959), fill="#8c2424" if age > 1.5 * float(record.get("capture_interval_s", 0.1)) else "#121715")
        draw.text((14, 936), detail, font=_font(14), fill="white")
    if derived_replay:
        draw.rectangle((0, 0, 1279, 36), fill="#8c2424")
        draw.text(
            (16, 8), "DIAGNOSTIC RECONSTRUCTION · MAY DIFFER FROM ORIGINAL RUN",
            font=label, fill="#ffffff",
        )
    return canvas


def _video_target(
    root: Path, trajectory: Mapping[str, Any], *, replay_mode: str = "review",
) -> tuple[str, Path, Path]:
    if replay_mode not in REPLAY_MODES:
        raise ValueError(f"unsupported Viewer replay mode: {replay_mode}")
    same_session_timeline = root / "native/operator-execution-frames/timeline.jsonl"
    replay_manifest = root / "native/controller-replay/manifest.json"
    if capture_ready(root):
        execution_identity: Mapping[str, Any] | None = {
            "kind": "same_session_execution_capture",
            "manifest_sha256": _sha256(root / "native/execution-capture/manifest.json"),
            "timeline_sha256": _sha256(root / "native/execution-capture/timeline.jsonl"),
            "reader_sha256": _sha256(PROJECT_ROOT / "viewer/execution_capture.py"),
        }
    elif same_session_timeline.is_file():
        execution_identity = {
            "kind": "same_session_robotwin_control_frames",
            "timeline_sha256": _sha256(same_session_timeline),
        }
    elif replay_mode == "forensic" and replay_manifest.is_file():
        execution_identity = {
            "kind": "viewer_derived_controller_replay",
            "replay_mode": replay_mode,
            "replay_identity_sha256": replay_identity(
                root, replay_mode=replay_mode,
            )[0],
        }
    elif replay_mode == "forensic":
        execution_identity = None
    else:
        execution_identity = {"kind": "original_observation_sequence"}
    identity_payload = {
        "video_contract": "original_review_or_explicit_diagnostic.v1",
        "renderer_sha256": _sha256(Path(__file__)),
        "trajectory": trajectory,
        "continuous_execution": execution_identity,
        "controller_replay_manifest_sha256": _sha256(replay_manifest),
        "controller_replay_reader_sha256": _sha256(
            PROJECT_ROOT / "ora/environment/controller_replay.py"
        ),
        "controller_replay_worker_sha256": _sha256(
            PROJECT_ROOT / "ora/environment/replay_worker.py"
        ),
    }
    identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    workspace = root.parent.parent if root.parent.name == "runs" else root.parent
    output = (
        workspace / "viewer-derived" / root.name
        / f"trajectory-{replay_mode}-{identity[:16]}.mp4"
    )
    return identity, output, output.with_suffix(".json")


def pending_replay_source(root: Path, *, replay_mode: str = "review") -> str | None:
    """Name the simulator-side replay a video build would have to compile first."""

    if replay_mode == "review" or capture_ready(root):
        return None
    if (root / "native/operator-execution-frames/timeline.jsonl").is_file():
        return None
    if (root / "native/controller-replay/manifest.json").is_file():
        _replay_id, replay_output = replay_identity(root, replay_mode=replay_mode)
        if not (replay_output / "manifest.json").is_file():
            return "controller replay"
    return None


def trajectory_video_cached(
    root: Path, trajectory: Mapping[str, Any], *, replay_mode: str = "review",
) -> bool:
    """Report exact-identity video cache state for an already-compiled trajectory."""

    _identity, output, manifest = _video_target(
        root, trajectory, replay_mode=replay_mode,
    )
    return output.is_file() and manifest.is_file()


def trajectory_video_status(
    root: Path, *, replay_mode: str = "review",
) -> dict[str, Any]:
    """Return exact current-identity cache status without creating a derived file."""
    root = root.resolve()
    trajectory = compile_trajectory(root)
    identity, output, manifest = _video_target(
        root, trajectory, replay_mode=replay_mode,
    )
    return {
        "trajectory_identity_sha256": identity,
        "video": output,
        "manifest": manifest,
        "cached": output.is_file() and manifest.is_file(),
    }


def compile_trajectory_video(
    root: Path, *, replay_mode: str = "review", allow_replay_compile: bool = True,
) -> tuple[Path, dict[str, Any]]:
    """Build or reuse an MP4 outside the canonical run evidence tree.

    ``allow_replay_compile=False`` serves cached or locally renderable videos
    only; when the build would first have to run a simulator-side controller
    replay, it raises
    :class:`TrajectoryVideoPendingError` instead of launching that worker.
    """
    root = root.resolve()
    trajectory = compile_trajectory(root)
    identity, output, manifest = _video_target(
        root, trajectory, replay_mode=replay_mode,
    )
    output_root = output.parent
    output_root.mkdir(parents=True, exist_ok=True)
    if output.is_file() and manifest.is_file():
        return output, _load(manifest)
    if not allow_replay_compile:
        pending = pending_replay_source(root, replay_mode=replay_mode)
        if pending is not None:
            raise TrajectoryVideoPendingError(
                f"{root.name}: the {pending} video is not compiled yet; run "
                "./scripts/dev/compile-viewer-videos outside the web process"
            )
    with _exclusive_replay_lock(output):
        if output.is_file() and manifest.is_file():
            return output, _load(manifest)
        return _compile_trajectory_video_locked(
            root, trajectory, identity, output, manifest, replay_mode=replay_mode,
        )


def _compile_trajectory_video_locked(
    root: Path, trajectory: Mapping[str, Any], identity: str,
    output: Path, manifest: Path, *, replay_mode: str,
) -> tuple[Path, dict[str, Any]]:
    output_root = output.parent
    execution = _continuous_execution_source(root)
    if execution is None and replay_mode == "forensic":
        execution = _controller_replay_source(root, replay_mode=replay_mode)
    steps = trajectory.get("steps") if isinstance(trajectory.get("steps"), list) else []
    if not steps and execution is None:
        raise RuntimeError("trajectory contains no observation steps")
    sampling = (
        dict(execution.get("sampling"))
        if execution is not None and isinstance(execution.get("sampling"), Mapping)
        else None
    )
    if execution is not None and execution.get("kind") == "same_session_execution_capture":
        execution_fps = float(execution["fps"])
    elif sampling is not None and sampling.get("mode") == "action_keyframes_v1":
        execution_fps = 10.0
    elif sampling is not None and sampling.get("mode") == "physics_stride_v1":
        execution_fps = 30 / int(sampling.get("capture_stride", 1))
    else:
        execution_fps = 30.0
    derived_replay_root = (
        Path(execution["root"])
        if execution is not None
        and execution.get("kind") == "viewer_derived_controller_replay"
        else None
    )
    derived_timeline = output.with_suffix(".timeline.jsonl")
    timeline_manifest_path = (
        derived_timeline.name
        if derived_replay_root is not None
        else execution.get("timeline_path") if execution is not None else None
    )
    with tempfile.TemporaryDirectory(
        prefix="trajectory-", dir=output_root,
    ) as temporary_name:
        temporary = Path(temporary_name)
        encoded = temporary / "trajectory.mp4"
        copied_timeline = temporary / "timeline.jsonl"
        if execution is not None:
            action_steps, action_count = _execution_step_map(trajectory)
            process = subprocess.Popen(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1920x960",
                    "-r", str(execution_fps), "-i", "-", "-an", "-c:v", "libx264",
                    "-crf", "20", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(encoded),
                ],
                stdin=subprocess.PIPE,
            )
            try:
                if process.stdin is None:  # pragma: no cover - subprocess contract
                    raise RuntimeError("ffmpeg raw-video stdin is unavailable")
                for execution_record in execution["frames"]:
                    frame = _render_continuous_execution_frame(
                        root, trajectory, execution_record,
                        source_root=execution["root"],
                        source_kind=str(execution["kind"]),
                        action_steps=action_steps, action_count=action_count,
                    )
                    process.stdin.write(frame.tobytes())
                process.stdin.close()
                return_code = process.wait()
                if return_code:
                    raise subprocess.CalledProcessError(return_code, process.args)
            finally:
                if process.stdin is not None and not process.stdin.closed:
                    process.stdin.close()
                if process.poll() is None:
                    process.kill()
                    process.wait()
            if derived_replay_root is not None:
                shutil.copyfile(
                    derived_replay_root / str(execution["timeline_path"]),
                    copied_timeline,
                )
        else:
            for index in range(len(steps)):
                frame = _render_trajectory_frame(root, trajectory, index)
                frame.save(temporary / f"frame-{index:04d}.png")
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-framerate", "5/7", "-start_number", "0",
                    "-i", str(temporary / "frame-%04d.png"),
                    "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(encoded),
                ],
                check=True,
            )
        if copied_timeline.is_file():
            os.replace(copied_timeline, derived_timeline)
        os.replace(encoded, output)
    record = {
            "schema": CONTINUOUS_VIDEO_SCHEMA if execution is not None else VIDEO_SCHEMA,
            "run_id": root.name,
            "trajectory_identity_sha256": identity,
            "trajectory_schema": trajectory.get("schema"),
            "frame_count": execution["frame_count"] if execution is not None else len(steps),
            "seconds_per_frame": (1 / execution_fps) if execution is not None else 1.4,
            "fps": execution_fps if execution is not None else 5 / 7,
            "resolution": [1920, 960] if execution is not None else [1280, 720],
            "video_sha256": _sha256(output),
            "source_hashes": trajectory.get("source_hashes"),
            "controller_replay_manifest_sha256": _sha256(root / "native/controller-replay/manifest.json"),
            "continuous_execution": execution is not None,
            "execution_capture_manifest_sha256": (
                _sha256(root / "native/execution-capture/manifest.json")
                if execution is not None and execution.get("kind") == "same_session_execution_capture"
                else None
            ),
            "capture_status": execution.get("capture_status") if execution is not None else None,
            "source_frame_count": execution.get("source_frame_count") if execution is not None else None,
            "video_source_kind": (
                execution["kind"] if execution is not None
                else "original_observation_sequence"
            ),
            "visual_fidelity": (
                "unverified_simulator_reconstruction"
                if derived_replay_root is not None else "original_frames_only"
            ),
            "unrecorded_motion": "not_reconstructed" if execution is None else None,
            "controller_replay": (
                execution is not None
                and execution["kind"] == "viewer_derived_controller_replay"
            ),
            "replay_mode": replay_mode if derived_replay_root is not None else None,
            "sampling": sampling,
            "camera_panels": execution["camera_panels"] if execution is not None else None,
            "execution_timeline": (
                {
                    "path": timeline_manifest_path,
                    "sha256": execution["timeline_sha256"],
                    "capture_semantics": execution["capture_semantics"],
                    "sampling": sampling,
                }
                if execution is not None else None
            ),
            "temporary_replay_frames_retained": False,
            "evidence_boundary": "read_only_viewer_derived_not_sealed_run_evidence",
        }
    temporary_manifest = manifest.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest)
    if derived_replay_root is not None:
        # Drop the bulky derived replay frames only after the video and its
        # manifest are durably in place, so a failed encode can retry from the
        # already-simulated replay instead of re-running the worker.
        shutil.rmtree(derived_replay_root, ignore_errors=True)
    return output, record
