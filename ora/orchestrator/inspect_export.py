"""Read-only Harness → Inspect Robots schema-v1 evidence projection.

The coordinator remains the execution/evaluation authority. This module never
drives a robot, invokes a model, re-evaluates a predicate or modifies a source
run. An Inspect ``success`` status means exportable execution, not task success.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

INSPECT_COMMIT = "7e4d1b7aee1c0d3cfc3a05a7492b9d12cda666f9"
EXPORT_SCHEMA = "ora.inspect-robots-export.v1"


class InspectExportError(ValueError):
    """Evidence cannot support the requested projection."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _iso(value: float) -> str:
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


class EvidenceReader:
    """Hash every consumed artifact; validate against a seal when available."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.consumed: dict[str, dict[str, Any]] = {}
        self.inventory: dict[str, dict[str, Any]] | None = None
        self.seal: dict[str, Any] | None = None
        if (self.root / "seal.json").is_file():
            self.seal = self.json("seal.json")
            records = self.seal["inventory"]
            if _sha(_canonical(records)) != self.seal["inventory_sha256"]:
                raise InspectExportError("seal inventory hash mismatch")
            self.inventory = {entry["path"]: entry for entry in records}
            if len(self.inventory) != len(records):
                raise InspectExportError("duplicate seal inventory paths")

    def read(self, name: str, expected: str | None = None) -> bytes:
        relative = Path(name)
        path = self.root / relative
        if relative.is_absolute() or ".." in relative.parts or not path.resolve().is_relative_to(self.root):
            raise InspectExportError("artifact escapes source run")
        data = path.read_bytes()
        digest = _sha(data)
        if expected is not None and digest != expected:
            raise InspectExportError(f"artifact reference hash mismatch: {name}")
        if self.inventory is not None:
            entry = self.inventory.get(name)
            if entry is None or entry["sha256"] != digest or entry["bytes"] != len(data):
                raise InspectExportError(f"consumed artifact differs from seal: {name}")
        self.consumed[name] = {"path": name, "sha256": digest, "bytes": len(data)}
        return data

    def json(self, name: str, expected: str | None = None) -> Any:
        return json.loads(self.read(name, expected))


def _transcript(reader: EvidenceReader, events: list[dict], round_record: dict,
                scene_id: str, frames: Path, *, images: bool) -> list[dict]:
    start = round_record["started_unix_s"]
    end = round_record["finished_unix_s"]
    selected = [e for e in events if start <= e.get("unix_s", -1) <= end]
    observations = [e for e in selected if e["type"] == "observation_received"]
    # Keyframes are explicitly sparse; no reconstructed or simulated frames.
    keyframes = {e["seq"] for e in observations[:1] + observations[-1:]}
    messages: list[dict] = []
    for event in selected:
        kind, payload = event["type"], event.get("payload", {})
        if kind == "provider_response" and payload.get("accepted") is True:
            raw = payload.get("message", {})
            messages.append({key: raw[key] for key in ("role", "content", "tool_calls") if key in raw})
        elif kind == "tool_result":
            ref = event.get("artifact_ref")
            content = reader.json(ref["path"], ref["sha256"]) if ref else payload
            messages.append({"role": "tool", "tool_call_id": event.get("tool_call_id"),
                             "name": payload.get("name"), "content": json.dumps(content, ensure_ascii=False)})
        elif kind == "operator_message" and isinstance(payload.get("message"), str):
            messages.append({"role": "user", "content": payload["message"]})
        elif images and event["seq"] in keyframes:
            from PIL import Image
            import numpy as np

            ref = payload["state_ref"]
            state = reader.json(ref["path"], ref["sha256"])
            step = int(event["seq"])
            content = [{"type": "text", "text": f"Recorded public observation, source event {step}. Sparse keyframe."}]
            for camera, value in sorted(state.get("cameras", {}).items()):
                if not camera.replace("_", "").replace("-", "").isalnum():
                    raise InspectExportError("invalid camera name")
                rgb = value["files"].get("rgb_lossless_png")
                if not rgb:
                    continue
                data = reader.read(rgb["path"], rgb["sha256"])
                array = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
                frames.mkdir(parents=True, exist_ok=True)
                np.save(frames / f"{scene_id}-e0_{camera}_{step:06d}.npy", array, allow_pickle=False)
                content.extend([
                    {"type": "text", "text": f"camera {camera!r} (step {step}):"},
                    {"type": "text", "text": "[image omitted: streamed camera frame]"},
                ])
            messages.append({"role": "user", "content": content})
    return messages


def _operator_messages(events: list[dict], round_index: int) -> tuple[list[dict], int]:
    """Retain delivered input and carry its persistent-context effect forward."""
    messages, cumulative_count = [], 0
    for event in events:
        if event.get("type") != "operator_message":
            continue
        payload = event.get("payload", {})
        delivered_round = payload.get("round_index")
        if type(delivered_round) is not int or delivered_round < 0:
            raise InspectExportError("operator message has no valid round identity")
        if delivered_round <= round_index:
            cumulative_count += 1
        if delivered_round == round_index:
            messages.append({"t": payload.get("observation_step"), "text": payload.get("message"),
                             "source": "harness_operator", **payload})
    return messages, cumulative_count


def export_run(run_root: Path, destination: Path, *, images: bool = True,
               comparison: str = "Source-bound Harness run; no cross-benchmark equivalence claimed") -> Path:
    """Write an Inspect EvalLog and a consumed-artifact provenance manifest.

    A sealed valid run gets autonomous per-round scores; assisted outcomes remain
    metadata without entering scores. A terminal unsealed or invalid run is an
    error with empty score entries; missing verdicts are never
    converted into failures. Only consumed files are checked, not the entire
    potentially multi-gigabyte seal inventory. Each persistent round is a
    distinct scene, never an independent epoch/trial replication claim.
    """
    root, destination = run_root.resolve(), destination.resolve()
    if destination.is_relative_to(root) or root.is_relative_to(destination):
        raise InspectExportError("derived export must be outside the source run")
    if destination.exists():
        raise InspectExportError("export destination already exists; use a new directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Publish only after every evidence check and sidecar write succeeds. Keep
    # staging on the destination filesystem so publication is a directory rename.
    with TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as staging:
        pending = Path(staging) / "export"
        _write_export(root, pending, images=images, comparison=comparison)
        if destination.exists():
            raise InspectExportError("export destination already exists; use a new directory")
        pending.rename(destination)
    return destination / "eval.json"


def _write_export(root: Path, destination: Path, *, images: bool, comparison: str) -> Path:
    """Build the complete derived projection in an unpublished directory."""
    reader = EvidenceReader(root)
    plan, result = reader.json("plan.json"), reader.json("result.json")
    if result.get("state") not in {"completed", "incomplete", "invalid", "blocked"}:
        raise InspectExportError("export requires a terminal result")
    run_id = result["run_id"]
    if result.get("plan_sha256") != plan.get("plan_sha256"):
        raise InspectExportError("result/plan identity mismatch")
    if reader.seal and reader.seal.get("run_id") != run_id:
        raise InspectExportError("seal/run identity mismatch")
    evaluation = reader.json("evaluation.json") if (root / "evaluation.json").is_file() else {}
    if evaluation and (evaluation.get("run_id") != run_id or evaluation.get("plan_sha256") != plan.get("plan_sha256")):
        raise InspectExportError("evaluation/run identity mismatch")
    valid = bool(reader.seal and reader.seal.get("evaluation_status") == "valid"
                 and evaluation.get("status") == "valid" and result["state"] == "completed")
    if evaluation.get("result_sha256") and evaluation["result_sha256"] != reader.consumed["result.json"]["sha256"]:
        raise InspectExportError("evaluation result digest mismatch")
    rounds = result.get("rounds") if isinstance(result.get("rounds"), list) else [result]
    expected_rounds = int((plan.get("session") or {}).get("rounds", 1))
    if valid and len(rounds) != expected_rounds:
        raise InspectExportError("valid run has incomplete round coverage")
    if not rounds:
        rounds = [result]
    events = ([json.loads(line) for line in reader.read("agent-events.jsonl").splitlines()]
              if (root / "agent-events.jsonl").is_file() else [])
    metrics = reader.json("runtime-metrics.json") if (root / "runtime-metrics.json").is_file() else {}
    destination.mkdir(parents=True)
    frames = destination / "frames" / "observations"
    samples, scores = [], []
    for index, record in enumerate(rounds):
        scene_id = f"round-{index:04d}"
        official = record.get("official_success")
        verdict = record.get("verdict") or {}
        completed_verdict = record.get("state") == "completed" and isinstance(official, bool)
        if valid and plan["runtime"]["kind"] != "synthetic" and not completed_verdict:
            raise InspectExportError("valid behavioral round is missing its official Boolean")
        if isinstance(verdict.get("official_success"), bool) and verdict["official_success"] != official:
            raise InspectExportError("round verdict disagrees with official result")
        operator_messages, delivered_count = _operator_messages(events, record.get("round_index", index))
        assisted = record.get("human_intervention") is True or bool(delivered_count)
        scoreable = valid and completed_verdict and record.get("behavioral_sample_valid") is not False and not assisted
        score = {"official_success": float(official)} if scoreable else {}
        if scoreable:
            scores.append(float(official))
        round_metadata = {key: record.get(key) for key in (
            "round_index", "seed", "decisions", "refused_proposals", "model_calls", "termination",
            "termination_authority", "behavioral_sample_valid", "protocol_comparable",
            "round_end_basis", "wall_duration_s", "human_intervention", "operator_message_count")}
        round_metadata.update(native_actions=verdict.get("native_actions"),
                              physics_steps=verdict.get("physics_steps"),
                              official_success=official, verdict_source=verdict.get("source"),
                              human_intervention=assisted,
                              operator_message_count=record.get("operator_message_count", delivered_count),
                              autonomous_scoreable=scoreable)
        messages = _transcript(reader, events, record, scene_id, frames, images=images)
        samples.append({
            "scene_id": scene_id, "status": "success" if valid else "error",
            "reduced": score, "epochs": [score],
            "error": None if valid else f"Harness {result['state']}; evaluation={evaluation.get('status', 'absent')}; sealed={bool(reader.seal)}",
            "instruction": plan.get("task_instruction"),
            "scene_metadata": {"run_id": run_id, "persistent_round_index": index,
                               "independent_replicate": False, "runtime_kind": plan["runtime"]["kind"]},
            "operator_judgements": [None], "judgement_sources": [None], "operator_notes": [None],
            "operator_messages": [operator_messages], "trial_metadata": [round_metadata],
            "termination_reasons": [record.get("termination")], "policy_transcripts": [messages],
        })
    policy = plan["policy"]
    start, finish = result["started_unix_s"], result["finished_unix_s"]
    duration = result.get("wall_duration_s", finish - start)
    if not math.isfinite(duration) or duration < 0:
        raise InspectExportError("invalid episode duration")
    log = {
        "version": 1, "status": "success" if valid else "error",
        "eval": {"task": plan["task"], "policy": policy.get("provider_profile", policy["kind"]),
                 "embodiment": plan["runtime"]["kind"], "created": _iso(start),
                 "inspect_robots_version": f"schema-v1-compatible@{INSPECT_COMMIT[:12]}",
                 "git_commit": plan["source"]["commit"], "seed": plan.get("seed"),
                 "max_steps": plan["budgets"]["max_robot_decisions"], "max_seconds": None,
                 "policy_config": {"model": policy.get("model"), "reasoning_effort": policy.get("reasoning_effort"),
                                   "tool_surface": policy.get("tool_surface", {}).get("profile"),
                                   "execution_authority": "Harness Supervisor", "step_unit": "Harness parent decision",
                                   "comparison": comparison},
                 "embodiment_info": {"is_simulated": plan["runtime"]["kind"] != "physical",
                                     "native_control_hz": plan["runtime"].get("control_hz"),
                                     "env_config": plan["runtime"].get("env_config")}},
        "results": {"total_scenes": len(samples), "total_trials": len(samples),
                    "errored_trials": 0 if valid else len(samples),
                    "metrics": {"official_success": sum(scores) / len(scores)} if scores else {}},
        "stats": {"started_at": _iso(start), "completed_at": _iso(finish), "duration_s": duration,
                  "total_steps": result.get("decisions", 0), "mean_inference_latency_s": None,
                  "frames_dir": "frames/observations" if frames.exists() else None},
        "samples": samples, "error": None if valid else samples[0]["error"],
    }
    log_path = destination / "eval.json"
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    provenance = {"schema": EXPORT_SCHEMA, "source_run_id": run_id, "source_root": str(root),
                  "source_commit": plan["source"]["commit"], "plan_sha256": plan["plan_sha256"],
                  "inspect_source_commit": INSPECT_COMMIT,
                  "verification": "consumed artifacts checked against seal inventory" if reader.seal else "unsealed terminal error; artifact hashes only",
                  "seal_sha256": reader.consumed.get("seal.json", {}).get("sha256"),
                  "consumed_artifacts": list(reader.consumed.values()), "eval_sha256": _sha(log_path.read_bytes()),
                  "workload": metrics.get("workload"), "latency": metrics.get("latency"),
                  "transcript_projection": "accepted assistant content/tool calls, tool results and delivered operator text; provider reasoning and wire requests omitted",
                  "score_projection": "autonomous behavioral rounds only; assisted official outcomes retained in trial metadata",
                  "media_projection": "first/last recorded policy RGB per round; sparse keyframes, not continuous execution video"}
    (destination / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    return log_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--no-images", action="store_true")
    args = parser.parse_args(argv)
    print(export_run(args.run_root, args.destination, images=not args.no_images))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
