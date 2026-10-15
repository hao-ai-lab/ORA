"""Issued Context reference boundaries and non-mutating historical review."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ora.agent.memory.artifacts import ContextArtifactCatalog
from ora.agent.evidence.trajectory import ArtifactStore
from ora.orchestrator.artifact_provenance import (
    context_artifact_entry_errors, verify_artifact_provenance,
)
from ora.orchestrator.evaluation import EvaluationError, evaluate
from ora.orchestrator.revalidation import _snapshot, revalidate_runs


@pytest.fixture
def evidence(tmp_path: Path):
    catalog = ContextArtifactCatalog(tmp_path, "session", ArtifactStore(tmp_path))
    ref = catalog.put_bytes(b"hello", kind="test", suffix=".txt", media_type="text/plain",
                            round_index=1, action_step=0)
    index = json.loads((tmp_path / "native/context-artifacts/index.json").read_text())
    request = {"messages": [{"role": "tool", "content": json.dumps({"recoverable_artifact": ref})}]}
    (tmp_path / "request.json").write_text(json.dumps(request))
    events = [
        {"type": "context_projected", "model_call_id": "call", "payload": {
            "passes": [{"artifact_refs": [ref]}]}},
        {"type": "provider_request", "model_call_id": "call", "payload": {
            "canonical_request_ref": {"path": "request.json"}}},
        {"type": "context_artifact_retrieval", "payload": {"status": "ok", "ref": ref["ref"]}},
    ]
    return tmp_path, events, index["artifacts"], ref, request


def test_model_prose_reasoning_and_rejected_tool_argument_are_not_publications(evidence):
    root, events, artifacts, ref, request = evidence
    bad = "artifact://short/" + "a" * 64
    request["messages"].extend([
        {"role": "assistant", "reasoning_content": bad, "content": bad,
         "tool_calls": [{"function": {"name": "artifact_read", "arguments": json.dumps({"ref": bad})}}]},
        {"role": "user", "content": "Documentation example: " + bad},
        {"role": "tool", "content": json.dumps({"ok": False, "error": {"message": bad}})},
    ])
    (root / "request.json").write_text(json.dumps(request))
    events.append({"type": "context_artifact_retrieval", "payload": {"status": "error", "ref": bad}})
    checked = verify_artifact_provenance(root, events, artifacts)
    assert checked["status"] == "valid"
    assert checked["issued_refs"] == checked["successful_retrieved_refs"] == 1


@pytest.mark.parametrize("origin", ["projection", "canonical", "retrieval"])
def test_unknown_issued_or_successfully_retrieved_reference_is_invalid(evidence, origin):
    root, events, artifacts, ref, request = evidence
    bad = dict(ref, ref="artifact://session/" + "b" * 64)
    if origin == "projection":
        events[0]["payload"]["passes"][0]["artifact_refs"] = [bad]
    elif origin == "canonical":
        request["messages"][0]["content"] = json.dumps({"recoverable_artifact": bad})
        (root / "request.json").write_text(json.dumps(request))
    else:
        events[2]["payload"]["ref"] = bad["ref"]
    checked = verify_artifact_provenance(root, events, artifacts)
    assert checked["status"] == "invalid"


def test_published_identity_and_historical_evidence_gaps_are_explicit(evidence):
    root, events, artifacts, ref, request = evidence
    events[0]["payload"]["passes"][0]["artifact_refs"][0]["bytes"] += 1
    assert "published_context_artifact_identity_mismatch" in verify_artifact_provenance(
        root, events, artifacts)["errors"]
    events[0]["payload"] = {}
    (root / "request.json").unlink()
    checked = verify_artifact_provenance(root, events, artifacts)
    assert checked["status"] == "insufficient_evidence"
    assert set(checked["evidence_gaps"]) == {
        "projection_artifact_inventory_missing", "canonical_request_origin_missing"}


def test_read_only_evaluation_and_report_preserve_every_source_file(tmp_path: Path):
    from tests.orchestrator.test_runtime_v1 import synthetic_plan
    from ora.orchestrator.coordinator import run_episode

    root = tmp_path / "run"
    root.mkdir()
    plan = synthetic_plan()
    (root / "plan.json").write_text(json.dumps(plan))
    run_episode(run_root=root, plan=plan, run_id="readonly-test")
    original = evaluate(root)
    before = _snapshot(root)
    assert evaluate(root, identity="backfill", persist=False) == original
    report = revalidate_runs([root], tmp_path / "review", workers=1)
    assert report["runs"][0]["status"] == "reviewed"
    assert report["runs"][0]["source_unchanged"] is True
    assert _snapshot(root) == before
    stored = json.loads((tmp_path / "review" / report["runs"][0]["report"]).read_text())
    assert stored["original_evaluation"] == original == stored["new_evaluation"]
    assert stored["original_files"] == before


@pytest.mark.parametrize("destination", ["inside", "ancestor", "other_formal", "runs"])
def test_revalidation_refuses_formal_or_overlapping_output(tmp_path: Path, destination: str):
    root = tmp_path / "source"
    root.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    (other / "plan.json").write_text("{}")
    out = {"inside": root / "audit", "ancestor": tmp_path,
           "other_formal": other / "audit", "runs": tmp_path / "runs" / "audit"}[destination]
    with pytest.raises(EvaluationError):
        revalidate_runs([root], out)
    assert not (out / "revalidation.json").exists()


def test_revalidation_defers_active_prefix(tmp_path: Path):
    root = tmp_path / "active"
    root.mkdir()
    (root / "plan.json").write_text("{}")
    before = _snapshot(root)
    report = revalidate_runs([root], tmp_path / "review")
    assert report["runs"][0]["status"] == "deferred_unfinished_prefix"
    assert _snapshot(root) == before


@pytest.mark.parametrize("damage", ["missing", "hash", "escape", "identity"])
def test_catalog_backing_file_integrity_still_fails_closed(evidence, damage):
    root, events, artifacts, ref, request = evidence
    index = {"policy_session_id": "session"}
    assert context_artifact_entry_errors(root, index, artifacts) == []
    entry = artifacts[ref["ref"]]
    path = root / entry["storage_ref"]["path"]
    if damage == "missing":
        path.unlink()
    elif damage == "hash":
        path.write_bytes(b"wrong")
    elif damage == "escape":
        entry["storage_ref"]["path"] = "../outside.txt"
    else:
        entry["policy_scope"] = "different-session"
    assert context_artifact_entry_errors(root, index, artifacts) == ["context_artifact_entry_invalid"]


@pytest.mark.parametrize("media_type,opener", [
    ("image/png", "observe"), ("text/plain", "artifact.read"),
    ("application/json", "artifact_read"),
])
def test_documented_compact_pointer_openers_match_expanded_catalog(evidence, media_type, opener):
    from ora.agent.memory.artifacts import recoverable_artifact_pointer

    root, events, artifacts, ref, request = evidence
    ref["media_type"] = artifacts[ref["ref"]]["media_type"] = media_type
    ref["open"] = {"tool": opener, "arguments": {"ref": ref["ref"]}}
    pointer = recoverable_artifact_pointer(ref)
    request["messages"][0]["content"] = json.dumps({"recoverable_artifact": pointer})
    (root / "request.json").write_text(json.dumps(request))
    assert verify_artifact_provenance(root, events, artifacts)["status"] == "valid"


@pytest.mark.parametrize("damage", ["wrong_mime", "wrong_decoder", "unknown_version", "expanded_string_opener"])
def test_compact_pointer_compatibility_does_not_admit_malformed_decoders(evidence, damage):
    from ora.agent.memory.artifacts import recoverable_artifact_pointer

    root, events, artifacts, ref, request = evidence
    pointer = recoverable_artifact_pointer(ref)
    if damage == "wrong_mime":
        pointer["mime"] = "application/json"
    elif damage == "wrong_decoder":
        pointer["open"] = "observe"
    elif damage == "unknown_version":
        pointer["v"] = 99
    else:
        pointer = {**ref, "open": "artifact.read"}
    request["messages"][0]["content"] = json.dumps({"recoverable_artifact": pointer})
    (root / "request.json").write_text(json.dumps(request))
    assert verify_artifact_provenance(root, events, artifacts)["errors"] == [
        "published_context_artifact_identity_mismatch"]


def test_host_path_check_ignores_home_inside_model_prose(tmp_path):
    from ora.orchestrator.evaluation_lossless import host_path_visible

    run_root = tmp_path / "runs" / "r1"
    # G004-03 / EXP12B g002 pick_dual_bottles: reasoning prose, not a path.
    assert not host_path_visible("the checker ignored grasp-flag/home/center-release", run_root)
    assert not host_path_visible("move back home/ and retry", run_root)
    assert host_path_visible('"path": "/home/alice/secrets.txt"', run_root)
    assert host_path_visible("cat /home/alice/notes", run_root)
    assert host_path_visible(f"see {run_root}/artifacts", run_root)
