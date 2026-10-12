"""Synthetic run directories for the run report tests.

Each helper writes only the files the report reads (plan, launch, status,
result, rounds, evaluation, seal and ``agent-events.jsonl``) in the shapes the
Harness writes them; nothing here starts a simulator or a provider.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def round_record(
    index: int = 0, *, success: bool | None = False, termination: str = "robot_decision_budget",
    authority: str = "experiment_budget", upstream: str | None = None, reason: str | None = None,
    sample_valid: bool | None = True, decisions: int = 10, calls: int = 20, minutes: float = 5.0,
    start: float = 1000.0, state: str = "completed", refused: int = 0, **extra: Any,
) -> dict[str, Any]:
    return {
        "round_index": index, "official_success": success, "state": state,
        "termination": termination, "termination_authority": authority,
        "upstream_ending_condition": upstream, "behavioral_sample_valid": sample_valid,
        "decisions": decisions, "model_calls": calls, "refused_proposals": refused,
        "wall_duration_s": minutes * 60.0, "started_unix_s": start,
        "finished_unix_s": start + minutes * 60.0,
        "environment_status": {"reason": reason if reason is not None else termination},
        **extra,
    }


def usage(input_tokens: int, cached: int, output: int, reasoning: int = 0, *, legacy: bool = False) -> dict:
    """One provider_response usage block; ``legacy`` names the cache read only by its Anthropic field."""

    if legacy:
        return {"input_tokens": input_tokens, "cache_read_input_tokens": cached,
                "output_tokens": output, "reasoning_output_tokens": reasoning}
    return {"input_tokens": input_tokens, "cached_input_tokens": cached, "cache_read_input_tokens": cached,
            "output_tokens": output, "reasoning_output_tokens": reasoning,
            "total_tokens": input_tokens + output}


def _event(kind: str, payload: Mapping[str, Any], *, unix_s: float, call_id: str | None = None,
           artifact: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {"artifact_ref": artifact, "payload": dict(payload), "tool_call_id": call_id,
            "type": kind, "unix_s": unix_s}


def agent_events(
    run: Path, *, model: str = "gpt-6-astra", responses: Sequence[Mapping[str, Any] | None] = (),
    tools: Iterable[tuple] = (), observation_bytes: Sequence[int] = (), images: Sequence[int] = (),
    finishes: Iterable[tuple[float, Mapping[str, Any], bool]] = (), start: float = 1000.0,
) -> None:
    """Write ``agent-events.jsonl``.

    ``tools`` holds ``(name, ok)``, ``(name, False, code)`` or
    ``(name, False, code, "batch")``; a refused call's error code is written to
    its tool-result artifact, per item for a batch. ``finishes`` holds
    ``(unix_s, arguments, ok)`` finish_episode calls.
    """

    events = [_event("session_started", {"model": model}, unix_s=start)]
    for index, (text_bytes, image_count) in enumerate(zip(observation_bytes, images)):
        events.append(_event("context_projected", {
            "active_inline_images": image_count,
            "context_sources": [
                {"bytes": 3000, "kind": "harness_system", "role": "system"},
                {"bytes": text_bytes, "kind": "causal_observation", "role": "user"},
            ],
        }, unix_s=start + index))
    for index, value in enumerate(responses):
        payload = {"accepted": True} if value is None else {"accepted": True, "usage": dict(value)}
        events.append(_event("provider_response", payload, unix_s=start + 1 + index))
    artifacts = run / "native" / "agent-artifacts"
    for index, tool in enumerate(tools):
        name, ok = tool[0], tool[1]
        call_id = f"call-{index}"
        events.append(_event("tool_call_proposed", {"name": name, "arguments": {}}, unix_s=start + 2,
                             call_id=call_id))
        artifact = None
        if not ok:
            batch = len(tool) > 3 and tool[3] == "batch"  # a batch result names codes per item
            body = json.dumps({"ok": False, "marks": [{"error_code": tool[2]}]} if batch else
                              {"ok": False, "error": {"code": tool[2], "message": "refused"}}).encode()
            digest = hashlib.sha256(body).hexdigest()
            artifacts.mkdir(parents=True, exist_ok=True)
            (artifacts / f"{digest}.json").write_bytes(body)
            artifact = {"path": f"native/agent-artifacts/{digest}.json", "sha256": digest, "bytes": len(body)}
        events.append(_event("tool_result", {"name": name, "ok": ok}, unix_s=start + 2, call_id=call_id,
                             artifact=artifact))
    for index, (unix_s, arguments, ok) in enumerate(finishes):
        call_id = f"finish-{index}"
        events.append(_event("tool_call_proposed", {"name": "finish_episode", "arguments": dict(arguments)},
                             unix_s=unix_s, call_id=call_id))
        events.append(_event("tool_result", {"name": "finish_episode", "ok": ok}, unix_s=unix_s,
                             call_id=call_id))
    run.joinpath("agent-events.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in events), encoding="utf-8")


def write_run(
    namespace: Path, run_id: str, *, cohort: str = "cohort-a", task: str = "task_a",
    model: str = "gpt-6-astra", kind: str = "robodojo", rounds: Sequence[Mapping[str, Any]] | None = None,
    planned_rounds: int = 1, state: str = "completed", termination: str = "session_round_limit",
    authority: str = "experiment_budget", evaluation: str | None = "valid", eligible: bool = True,
    evaluation_extra: Mapping[str, Any] | None = None, result: bool = True, sealed: bool = True,
    events: Mapping[str, Any] | None = None, status: Mapping[str, Any] | None = None,
) -> Path:
    run = namespace / "runs" / run_id
    run.mkdir(parents=True)
    rounds = [round_record()] if rounds is None else [dict(item) for item in rounds]
    plan = {
        "launch_admission": {"cohort_id": cohort, "cell_id": f"{task}-l0-s0-closed"},
        "task": task, "runtime": {"kind": kind}, "session": {"rounds": planned_rounds},
        "policy": {"kind": "native_api", "model": model, "provider_profile": f"{model}-profile",
                   "reasoning_effort": "high"},
        "experiment_identity": {"family": "EXP11B", "branch": "robodojo", "generation": "g006"},
    }
    run.joinpath("plan.json").write_text(json.dumps(plan), encoding="utf-8")
    run.joinpath("launch.json").write_text(json.dumps({"created_unix_s": 900.0}), encoding="utf-8")
    decisions = sum(int(item.get("decisions") or 0) for item in rounds)
    calls = sum(int(item.get("model_calls") or 0) for item in rounds)
    status_record = {"state": state, "decisions": decisions, "model_calls": calls, "updated_unix_s": 1900.0}
    status_record.update(status or {})
    run.joinpath("status.json").write_text(json.dumps(status_record), encoding="utf-8")
    for item in rounds:
        folder = run / "rounds" / f"round-{int(item['round_index']):04d}"
        folder.mkdir(parents=True)
        folder.joinpath("result.json").write_text(json.dumps(item), encoding="utf-8")
    if result:
        record = {
            "state": state, "termination": termination, "termination_authority": authority,
            "decisions": status_record["decisions"], "model_calls": status_record["model_calls"],
            "wall_duration_s": sum(float(item.get("wall_duration_s") or 0) for item in rounds),
            "rounds": rounds, "round_count": len(rounds),
        }
        run.joinpath("result.json").write_text(json.dumps(record), encoding="utf-8")
    if evaluation is not None:
        body = {"status": evaluation, "reasons": [] if evaluation == "valid" else ["trusted_fault"],
                "learning_eligibility": {"eligible": eligible}, **dict(evaluation_extra or {})}
        run.joinpath("evaluation.json").write_text(json.dumps(body), encoding="utf-8")
    if sealed:
        run.joinpath("seal.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
    agent_events(run, model=model, **dict(events or {}))
    return run


def write_correction(namespace: Path, name: str, run: Path, *, validity: str = "infrastructure_invalid",
                     evaluation_sha256: str | None = None, seal_sha256: str | None = None) -> Path:
    """A record in the shape of EXP11B ``g006-E11BG6-001-conveyor.json``, citing the run's file hashes."""

    folder = namespace / "diagnostics" / "corrections"
    folder.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": "exp11b-robodojo.post-seal-correction.v1", "correction_id": name,
        "decided_by": "user", "reason": "environment did not present the task",
        "sealed_evidence_modified": False,
        "runs": [{
            "run_id": run.name, "cell_id": "task_a-l0-s0-closed",
            "seal_file_sha256": seal_sha256 or hashlib.sha256((run / "seal.json").read_bytes()).hexdigest(),
            "evaluation_file_sha256": (evaluation_sha256
                                       or hashlib.sha256((run / "evaluation.json").read_bytes()).hexdigest()),
            "as_sealed": {"evaluation_status": "valid", "learning_eligible": True},
            "corrected": {"validity": validity, "learning_eligible": False, "counts_as_model_attempt": False},
        }],
    }
    path = folder / f"{name}.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def tree_digest(root: Path) -> dict[str, str]:
    """Every file under ``root`` with its bytes' hash, to prove a reader wrote nothing."""

    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*")) if path.is_file()
    }
