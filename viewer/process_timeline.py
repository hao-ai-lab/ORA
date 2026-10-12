"""Source-addressed observations, model responses, tools and execution chapters.

This is a presentation projection, never an alternate canonical event stream.
Native calls retain their recorded interleaving. Managed summaries explicitly
declare that their cross-channel order and hidden reasoning are unavailable.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from viewer.rounds import PERSISTENT_CADENCE, integer, iter_records, round_ledger, scene_label


PROCESS_SCHEMA = "ora.viewer-process-timeline.v1"
CONTEXT_TYPES = frozenset({
    "context_compactor_attempt", "context_compactor_response", "context_condensation",
    "context_artifact_retrieval", "context_artifact_admission", "context_lifecycle_transition",
    "memory_snapshot_loaded", "memory_loaded", "memory_capture", "memory_candidate_created",
    "memory_published", "task_memory_published", "memory_review_completed",
})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for part in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def verified_artifact(root: Path, reference: Any) -> tuple[Path | None, str | None]:
    if not isinstance(reference, Mapping) or not isinstance(reference.get("path"), str):
        return None, "artifact_reference_missing"
    path = (root / reference["path"]).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        return None, "artifact_missing_or_outside_run"
    expected = reference.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        return None, "artifact_hash_missing"
    if sha256(path) != expected:
        return None, "artifact_hash_mismatch"
    return path, None


def image_from_event(root: Path, event: Mapping[str, Any], tool: str) -> tuple[dict | None, str | None]:
    reference = event.get("artifact_ref")
    path, issue = verified_artifact(root, reference)
    if path is None:
        return None, issue
    if reference.get("media_type") not in {"image/png", "image/jpeg", "image/webp"}:
        return None, "unsupported_tool_image_type"
    return {
        "path": path.relative_to(root.resolve()).as_posix(), "sha256": reference["sha256"],
        "label": tool or "tool image", "kind": "tool_image", "main": False,
        "visibility": "tool_returned", "recorded_visibility": event.get("visibility"),
        "tool_call_id": event.get("tool_call_id"), "model_call_id": event.get("model_call_id"),
    }, None


def tool_images_by_step(root: Path, events: Iterable[Mapping[str, Any]]) -> dict[int, list[dict]]:
    """Expose hash-verified Native images omitted by the legacy materializers."""
    proposals: dict[str, str] = {}
    result: dict[int, list[dict]] = defaultdict(list)
    for event in events:
        call = event.get("tool_call_id")
        payload = event.get("payload") or {}
        if event.get("type") == "tool_call_proposed" and isinstance(call, str):
            proposals[call] = str(payload.get("name") or "tool image")
        if event.get("type") != "tool_image" or integer(event.get("observation_step")) is None:
            continue
        image, _issue = image_from_event(root, event, proposals.get(call, "tool image"))
        if image is not None:
            result[event["observation_step"]].append(image)
    return dict(result)


def _short_payload(value: Any, limit: int = 1800) -> str:
    text = json.dumps(value, ensure_ascii=False, indent=2) if value is not None else "Not recorded"
    return text if len(text) <= limit else text[:limit] + "\n… [excerpt; open original event]"


def _phase(tool: str) -> str:
    if tool.startswith("vision.") or tool in {"observe", "mark_point", "preview_move"}:
        return "perception"
    if tool.startswith("grasp.") or tool.startswith("motion."):
        return "planning"
    if "memory" in tool.lower() or "artifact" in tool.lower():
        return "context"
    if tool in {"check_task", "finish_episode"}:
        return "check"
    return "tool"


def _source(line: int, event: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stream": "agent", "source_line": line, "event_id": event.get("event_id"),
        "seq": event.get("seq"), "causal_parent_id": event.get("causal_parent_id"),
        "model_call_id": event.get("model_call_id"), "tool_call_id": event.get("tool_call_id"),
    }


def _scene_note(scene: Any) -> str:
    """A scheduled round's scene: which layout, and whether the simulation was relaunched."""
    label = scene_label(scene)
    if label is None:
        return ""
    relaunched = scene.get("scene_reset_mode") == "simulation_relaunch"
    return f" · {'simulation relaunched on ' if relaunched else ''}{label}"


def compile_process_timeline(root: Path, trajectory: Mapping[str, Any] | None = None) -> dict[str, Any]:
    from viewer.trajectory import compile_trajectory

    root = root.resolve()
    trajectory = trajectory if trajectory is not None else compile_trajectory(root)
    ledger = round_ledger(root)
    steps = trajectory.get("steps") or []
    by_step = {item["state_step"]: item for item in steps}
    by_action = {item["action"]["id"]: item for item in steps if item.get("action") and item["action"].get("id")}
    if len(by_action) != sum(bool(item.get("action") and item["action"].get("id")) for item in steps):
        raise ValueError("duplicate Runtime action identity")
    round_by_step: dict[int, int] = {}
    for row in ledger["rounds"]:
        bounds = row["policy_state_steps"]
        start, end = integer(bounds.get("start")), integer(bounds.get("end_exclusive"))
        if start is not None and end is not None:
            for step in by_step:
                if start <= step < end:
                    if step in round_by_step:
                        raise ValueError("overlapping round state ranges")
                    round_by_step[step] = row["round_index"]
    if not ledger["is_session"]:
        round_by_step = dict.fromkeys(by_step, 0)

    groups: dict[int, list[dict]] = defaultdict(list)
    proposals: dict[str, tuple[int, dict]] = {}
    requests: dict[str, float] = {}
    errors: list[dict] = []
    unmapped: list[dict] = []
    last_step: int | None = None
    last_ordinal = -1
    step_ordinals = {step["state_step"]: index for index, step in enumerate(steps)}
    native_calls = 0
    native_actions: set[str] = set()

    for line, event in iter_records(root / "agent-events.jsonl"):
        kind = event.get("type")
        payload = event.get("payload") if isinstance(event.get("payload"), Mapping) else {}
        step = integer(event.get("observation_step"))
        if step in by_step:
            last_step = step
        contextual = kind in CONTEXT_TYPES
        target = step if step in by_step else last_step if contextual else None
        if kind == "provider_request":
            if isinstance(event.get("unix_s"), (float, int)):
                requests[str(event.get("model_call_id"))] = event["unix_s"]
            continue
        call = event.get("tool_call_id")
        if kind == "tool_call_proposed" and isinstance(call, str):
            if call in proposals:
                errors.append({"source": _source(line, event), "issue": "duplicate_tool_call_id"})
            proposals[call] = (line, event)
            continue
        if kind not in {"provider_response", "tool_result", "tool_image", "action_proposed", "environment_receipt"} and not contextual:
            continue
        if target not in by_step:
            unmapped.append({"source": _source(line, event), "type": kind, "issue": "no_recorded_observation_join"})
            continue
        ordinal = step_ordinals[target]
        if ordinal < last_ordinal:
            # Backreferences are retained as unplaced evidence, never reordered
            # into an earlier observation and presented as current cognition.
            unmapped.append({"source": _source(line, event), "type": kind, "issue": "nonmonotonic_observation_join"})
            continue
        last_ordinal = ordinal
        source = _source(line, event)
        block = {
            "key": f"agent:{line}", "phase": "context" if contextual else kind,
            "state_step": target, "round_index": round_by_step.get(target),
            "observation_id": by_step[target].get("observation_id"),
            "source_refs": [source], "unix_s": event.get("unix_s"),
            "model_call_id": event.get("model_call_id"), "tool_call_id": call,
            "images": [], "text": "", "reasoning": None, "issues": [],
            "order": "native_event_order", "action_id": None,
            "visibility": "recorded_model_output" if kind == "provider_response" else "recorded_tool_output",
        }
        if contextual:
            block.update(title=str(kind).replace("_", " "), text=_short_payload(payload),
                         placement="recorded_session_order" if step is None else "recorded_observation",
                         visibility="context_lifecycle_metadata")
        elif kind == "provider_response":
            native_calls += 1
            message = payload.get("message") if isinstance(payload.get("message"), Mapping) else {}
            content = message.get("content")
            reasoning = message.get("reasoning_content")
            started = requests.get(str(event.get("model_call_id")))
            finished = event.get("unix_s")
            block.update(
                phase="decision", title="Model decision", text=content if isinstance(content, str) else "",
                reasoning=reasoning if isinstance(reasoning, str) else None,
                accepted=payload.get("accepted"),
                latency_s=finished - started if isinstance(started, (float, int)) and isinstance(finished, (float, int)) else None,
                usage=payload.get("usage"),
            )
        elif kind == "tool_result":
            proposal = proposals.get(call)
            proposed = (proposal[1].get("payload") or {}) if proposal else {}
            tool = str(payload.get("name") or proposed.get("name") or "unknown tool")
            if proposal:
                block["source_refs"].insert(0, _source(*proposal))
            else:
                block["issues"].append("tool_proposal_missing")
            arguments = proposed.get("arguments")
            path, issue = verified_artifact(root, event.get("artifact_ref"))
            value = None
            if path:
                try:
                    value = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    issue = "tool_result_not_json"
            if issue:
                block["issues"].append(issue)
            block.update(
                phase=_phase(tool), title=tool, tool=tool, ok=payload.get("ok"),
                arguments=arguments, effect=proposed.get("effect"),
                text=_short_payload(arguments), result_excerpt=_short_payload(value),
                result_artifact=event.get("artifact_ref"),
            )
        elif kind == "tool_image":
            proposal = proposals.get(call)
            name = str((proposal[1].get("payload") or {}).get("name") or "tool image") if proposal else "tool image"
            image, issue = image_from_event(root, event, name)
            adjacent = groups[target][-1] if groups[target] else None
            if adjacent and adjacent.get("tool_call_id") == call and adjacent.get("result_artifact"):
                adjacent["source_refs"].append(source)
                if image:
                    adjacent["images"].append(image)
                if issue:
                    adjacent["issues"].append(issue)
                continue
            block.update(phase=_phase(name), title=f"{name} · image", images=[image] if image else [])
            if issue:
                block["issues"].append(issue)
        elif kind == "action_proposed":
            action_id = payload.get("action_id")
            bound = by_action.get(action_id)
            if bound is None or bound["state_step"] != target or action_id in native_actions:
                block.update(phase="unmatched", title="Unmatched action proposal")
                block["issues"].append("action_identity_join_failed")
            else:
                native_actions.add(action_id)
                block.update(phase="execution", title=bound["action"]["kind"], action_id=action_id,
                             text=_short_payload(bound["action"].get("arguments")), action=bound["action"], visibility="control_only")
        elif kind == "environment_receipt":
            action_id = payload.get("action_id")
            bound = by_action.get(action_id)
            if bound is None or bound["state_step"] != target:
                block.update(phase="unmatched", title="Unmatched execution receipt")
                block["issues"].append("receipt_identity_join_failed")
            else:
                block.update(phase="outcome", title="Measured outcome", action_id=action_id,
                             receipt=bound.get("receipt"), text=_short_payload(bound.get("receipt")), visibility="control_only")
        groups[target].append(block)

    events: list[dict] = []
    current_round: int | None = None
    opened_rounds: set[int] = set()

    def boundary(index: int, phase: str, state_step: int | None) -> dict:
        row = next(item for item in ledger["rounds"] if item["round_index"] == index)
        return {
            "key": f"round:{index}:{phase}", "phase": phase, "round_index": index,
            "state_step": state_step, "title": f"Round {index + 1} / {ledger['planned_rounds']}",
            "text": ((
                "Environment reset · AgentSession continues" if ledger["cadence"] == PERSISTENT_CADENCE
                else "Recorded round boundary · session continuity unestablished" if ledger["is_session"]
                else "Episode start"
            ) + _scene_note(row.get("scene"))) if phase == "round_start" else f"Official result: {row['outcome'].replace('_', ' ')} · {row['decisions'] if row['decisions'] is not None else '—'} actions",
            "round": row, "images": [], "source_refs": [{"path": row["source"], "round_index": index}],
            "order": "recorded_round_boundary", "issues": [], "visibility": "round_metadata",
        }

    for position, step in enumerate(steps):
        state_step = step["state_step"]
        index = round_by_step.get(state_step)
        if index != current_round:
            if current_round is not None:
                events.append(boundary(current_round, "round_end", steps[position - 1]["state_step"]))
            current_round = index
            if index is not None:
                opened_rounds.add(index)
                events.append(boundary(index, "round_start", state_step))
        rgb = [image for image in step.get("images", []) if image.get("kind") == "rgb"]
        events.append({
            "key": f"observation:{state_step}", "phase": "observation", "state_step": state_step,
            "round_index": index, "title": "Recorded observation", "text": str(step.get("observation_id") or ""),
            "images": rgb, "source_refs": [{"path": "policy-state/manifest.json", "state_step": state_step}],
            "order": "recorded_observation_boundary", "issues": [], "visibility": "policy_visible",
        })
        block_group = groups.get(state_step, [])
        if not block_group:
            # Managed or historical summaries are deliberately separate from a
            # Native call stream whose evidence has failed to join.
            for call_index, call in enumerate(step.get("model_calls") or []):
                if call.get("message_kind") == "native_provider_response":
                    continue
                block_group.append({
                    "key": f"summary:{state_step}:{call_index}", "phase": "decision", "state_step": state_step,
                    "round_index": index, "title": "Recorded model summary", "images": [], "source_refs": [],
                    "text": "\n\n".join(call.get("messages") or []), "reasoning": None,
                    "order": "partial_summary_order_unknown", "visibility": "recorded_model_output",
                    "issues": ["cross_channel_order_unavailable"], "model_call_id": call.get("call"),
                })
            projected_images = [image for image in step.get("images", []) if image.get("kind") not in {"rgb", "operator"}]
            if projected_images or step.get("tools"):
                block_group.append({
                    "key": f"tool-summary:{state_step}", "phase": "tool", "state_step": state_step,
                    "round_index": index, "title": "Recorded step tools · partial ordering", "images": projected_images,
                    "source_refs": [{"path": "events.jsonl", "state_step": state_step}],
                    "text": _short_payload([{"tool": tool.get("tool"), "arguments": tool.get("arguments"), "ok": tool.get("ok")} for tool in step.get("tools", [])]),
                    "order": "partial_summary_order_unknown", "visibility": "per_artifact_visibility",
                    "issues": ["cross_channel_order_unavailable"],
                })
        events.extend(block_group)
        action = step.get("action")
        if action and action.get("id") not in native_actions:
            events.append({
                "key": f"action:{action['id']}", "phase": "execution", "state_step": state_step,
                "round_index": index, "title": str(action.get("kind") or "Action"), "images": [],
                "source_refs": [{"path": "events.jsonl", "action_id": action["id"]}],
                "action_id": action["id"], "action": action, "text": _short_payload(action.get("arguments")),
                "order": "runtime_action_order", "issues": [], "visibility": "control_only",
            })
            if step.get("receipt"):
                events.append({
                    "key": f"receipt:{action['id']}", "phase": "outcome", "state_step": state_step,
                    "round_index": index, "title": "Measured outcome", "images": [],
                    "source_refs": [{"path": "events.jsonl", "action_id": action["id"]}],
                    "action_id": action["id"], "receipt": step["receipt"], "text": _short_payload(step["receipt"]),
                    "order": "runtime_action_order", "issues": [], "visibility": "control_only",
                })
    if current_round is not None:
        events.append(boundary(current_round, "round_end", steps[-1]["state_step"]))
    # An officially recorded empty round is still a chapter; it is not erased
    # merely because there is no observation/action to attach it to.
    for row in ledger["rounds"]:
        if row["round_index"] not in opened_rounds and row["outcome"] != "not_run":
            position = next((index for index, event in enumerate(events)
                             if isinstance(event.get("round_index"), int) and event["round_index"] > row["round_index"]), len(events))
            events[position:position] = [boundary(row["round_index"], "round_start", None), boundary(row["round_index"], "round_end", None)]
    execution_tools = {event.get("tool_call_id") for event in events if event["phase"] == "execution" and event.get("tool_call_id")}
    for event in events:
        event["covered_by_execution"] = bool(event.get("result_artifact") and event.get("tool_call_id") in execution_tools)
    source_hashes = dict(trajectory.get("source_hashes") or {})
    return {
        "schema": PROCESS_SCHEMA, "run_id": root.name, "source_commit": trajectory.get("source_commit"),
        "instruction": trajectory.get("instruction") or trajectory.get("task"),
        "round_ledger": ledger, "events": events, "source_hashes": source_hashes,
        "issues": errors, "unmapped_events": unmapped,
        "coverage": {
            "native_responses": native_calls,
            "reasoning_responses": sum(bool(event.get("reasoning")) for event in events),
            "tool_image_events": sum(len(event.get("images") or []) for event in events if event["phase"] != "observation"),
            "ordering": "native_event_order_within_observation" if native_calls else "partial_managed_or_historical_projection",
        },
        "evidence_boundary": "read_only_viewer_projection_not_sealed_evidence",
    }
