"""Source-bound Inspect log semantics without robot or provider execution."""
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from ora.orchestrator.inspect_export import InspectExportError, export_run


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def seal(root):
    records = [{"path": f.name, "sha256": digest(f.read_bytes()), "bytes": f.stat().st_size}
               for f in sorted(root.iterdir()) if f.is_file() and f.name != "seal.json"]
    write(root / "seal.json", {"schema": "ora.seal.v2",
                              "run_id": "test-run", "evaluation_status": "valid", "inventory": records,
                              "inventory_sha256": digest(json.dumps(records, sort_keys=True, separators=(",", ":")).encode())})


@pytest.fixture
def run(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    plan = {"plan_sha256": "a" * 64, "task": "insert_tubes", "task_instruction": "Insert tubes.",
            "source": {"commit": "b" * 40}, "seed": 0, "session": {"rounds": 2},
            "runtime": {"kind": "robodojo", "control_hz": 25},
            "policy": {"kind": "native_api", "provider_profile": "deepseek_flash_openai_low_16k_v1"},
            "budgets": {"max_robot_decisions": 200}}
    rounds = [{"round_index": i, "state": "completed", "official_success": success,
               "started_unix_s": 1000 + i * 20, "finished_unix_s": 1010 + i * 20,
               "verdict": {"official_success": success, "source": "official fixture", "native_actions": 4},
               "termination": "native_action_limit", "protocol_comparable": True}
              for i, success in enumerate((False, True))]
    result = {"run_id": "test-run", "plan_sha256": plan["plan_sha256"], "state": "completed",
              "started_unix_s": 1000, "finished_unix_s": 1030, "wall_duration_s": 30,
              "decisions": 2, "rounds": rounds}
    write(root / "plan.json", plan)
    write(root / "result.json", result)
    write(root / "evaluation.json", {"run_id": "test-run", "plan_sha256": plan["plan_sha256"],
                                     "status": "valid", "result_sha256": digest((root / "result.json").read_bytes())})
    seal(root)
    return root


def test_false_is_scored_and_persistent_rounds_keep_identity(run, tmp_path):
    path = export_run(run, tmp_path / "export", images=False)
    value = json.loads(path.read_text())
    assert value["status"] == "success"
    assert value["samples"][0]["epochs"] == [{"official_success": 0.0}]
    assert value["samples"][1]["epochs"] == [{"official_success": 1.0}]
    assert value["results"]["metrics"]["official_success"] == 0.5
    assert all(s["scene_metadata"]["independent_replicate"] is False for s in value["samples"])
    assert value["stats"]["total_steps"] == 2  # parent decisions, not 8 native actions


@pytest.mark.parametrize("intervention_round,expected_metric", [(0, None), (1, 0.0)])
def test_assisted_outcomes_keep_verdicts_and_input_without_autonomous_scores(run, tmp_path, intervention_round, expected_metric):
    plan = json.loads((run / "plan.json").read_text())
    plan.update(experiment="ora-interactive", operator_control={"mode": "file_inbox_v1"})
    write(run / "plan.json", plan)
    result = json.loads((run / "result.json").read_text())
    result.update(human_intervention=True, operator_message_count=1, behavioral_sample_valid=False)
    for record in result["rounds"]:
        assisted = record["round_index"] >= intervention_round
        record.update(human_intervention=assisted, operator_message_count=int(assisted), behavioral_sample_valid=not assisted)
    write(run / "result.json", result)
    evaluation = json.loads((run / "evaluation.json").read_text())
    evaluation["result_sha256"] = digest((run / "result.json").read_bytes())
    write(run / "evaluation.json", evaluation)
    payload = {"schema": "ora.ora-operator-command.v1", "id": "steer-1", "type": "steer",
               "message": "Check alignment before inserting.", "run_id": "test-run", "plan_sha256": plan["plan_sha256"],
               "round_index": intervention_round, "observation_step": 3, "submitted_unix_ns": 1,
               "before_model_call_id": "model-call-000001"}
    (run / "agent-events.jsonl").write_text(json.dumps({
        "type": "operator_message", "seq": 0, "unix_s": 1001 + 20 * intervention_round, "payload": payload,
    }) + "\n")
    seal(run)
    before = {path.name: path.read_bytes() for path in run.iterdir()}
    value = json.loads(export_run(run, tmp_path / "export", images=False).read_text())
    assert value["status"] == "success"
    assert value["results"]["errored_trials"] == 0
    assert value["results"]["metrics"] == ({} if expected_metric is None else {"official_success": expected_metric})
    assert value["results"]["total_trials"] == 2
    for index, sample in enumerate(value["samples"]):
        assisted = index >= intervention_round
        assert sample["epochs"] == ([{}] if assisted else [{"official_success": 0.0}])
        metadata = sample["trial_metadata"][0]
        assert metadata["official_success"] is (index == 1)
        assert metadata["human_intervention"] is assisted
        assert metadata["operator_message_count"] == int(assisted)
        assert metadata["autonomous_scoreable"] is not assisted
        if index == intervention_round:
            message = sample["operator_messages"][0][0]
            assert message["text"] == payload["message"]
            assert message["source"] == "harness_operator"
            assert message["t"] == 3
            assert message["id"] == payload["id"]
            assert sample["policy_transcripts"][0] == [{"role": "user", "content": payload["message"]}]
        else:
            assert sample["operator_messages"] == [[]]
    assert before == {path.name: path.read_bytes() for path in run.iterdir()}


@pytest.mark.parametrize("field,value", [("human_intervention", True), ("behavioral_sample_valid", False)])
def test_assistance_or_invalid_sample_marker_excludes_round_without_changing_verdict(run, tmp_path, field, value):
    result = json.loads((run / "result.json").read_text())
    result["rounds"][1][field] = value
    write(run / "result.json", result)
    evaluation = json.loads((run / "evaluation.json").read_text())
    evaluation["result_sha256"] = digest((run / "result.json").read_bytes())
    write(run / "evaluation.json", evaluation)
    seal(run)
    exported = json.loads(export_run(run, tmp_path / "export", images=False).read_text())
    assert exported["results"]["metrics"] == {"official_success": 0.0}
    assert exported["samples"][1]["trial_metadata"][0]["official_success"] is True
    assert exported["samples"][1]["reduced"] == {}


def test_unsealed_prefix_never_becomes_behavioral_failure(run, tmp_path):
    (run / "seal.json").unlink()
    value = json.loads(export_run(run, tmp_path / "export", images=False).read_text())
    assert value["status"] == "error"
    assert value["results"]["metrics"] == {}
    assert value["results"]["errored_trials"] == 2
    assert all(s["epochs"] == [{}] for s in value["samples"])


def test_sealed_artifact_tampering_refused(run, tmp_path):
    result = json.loads((run / "result.json").read_text())
    result["rounds"][0]["official_success"] = True
    write(run / "result.json", result)
    with pytest.raises(InspectExportError, match="differs from seal"):
        export_run(run, tmp_path / "export", images=False)


def test_absent_official_verdict_refused_for_valid_behavior(run, tmp_path):
    result = json.loads((run / "result.json").read_text())
    result["rounds"][0]["official_success"] = None
    result["rounds"][0]["verdict"] = {}
    write(run / "result.json", result)
    evaluation = json.loads((run / "evaluation.json").read_text())
    evaluation["result_sha256"] = digest((run / "result.json").read_bytes())
    write(run / "evaluation.json", evaluation)
    seal(run)
    with pytest.raises(InspectExportError, match="missing its official Boolean"):
        export_run(run, tmp_path / "export", images=False)


def test_destination_cannot_touch_source_or_overwrite(run, tmp_path):
    with pytest.raises(InspectExportError, match="outside"):
        export_run(run, run / "derived", images=False)
    export_run(run, tmp_path / "export", images=False)
    with pytest.raises(InspectExportError, match="already exists"):
        export_run(run, tmp_path / "export", images=False)


@pytest.mark.parametrize("images", [False, True])
def test_real_inspect_reader_and_html(run, tmp_path, images):
    import os
    interpreter = os.environ.get("INSPECT_ROBOTS_TEST_PYTHON")
    if not interpreter:
        pytest.skip("separate pinned Inspect runtime is an explicit integration-test input")
    if images:
        from PIL import Image

        Image.new("RGB", (8, 8), (20, 100, 200)).save(run / "rgb.png")
        write(run / "observation.json", {"cameras": {"front": {"files": {
            "rgb_lossless_png": {"path": "rgb.png", "sha256": digest((run / "rgb.png").read_bytes())}
        }}}})
        state_ref = {"path": "observation.json", "sha256": digest((run / "observation.json").read_bytes())}
        (run / "agent-events.jsonl").write_text(json.dumps({
            "type": "observation_received", "seq": 1, "unix_s": 1001,
            "payload": {"state_ref": state_ref},
        }) + "\n")
        seal(run)
    path = export_run(run, tmp_path / "export", images=images)
    subprocess.run([interpreter, "-c", "from inspect_robots.log import read_eval_log; import sys; "
                    "v=read_eval_log(sys.argv[1]); assert v.results.metrics['official_success']==0.5", str(path)], check=True)
    html = path.with_suffix(".html")
    subprocess.run([interpreter, "-m", "inspect_robots.cli", "view", str(path), "-o", str(html)], check=True)
    assert "official_success" in html.read_text()
    if images:
        assert "data:image/png;base64," in html.read_text()


def test_failed_sidecar_write_leaves_destination_retryable(run, tmp_path, monkeypatch):
    destination = tmp_path / "export"
    before = {path.name: path.read_bytes() for path in run.iterdir()}
    original_write = Path.write_text

    def fail_provenance(path, *args, **kwargs):
        if path.name == "provenance.json":
            assert not destination.exists()
            assert (path.parent / "eval.json").is_file()
            raise OSError("injected sidecar write failure")
        return original_write(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "write_text", fail_provenance)
        with pytest.raises(OSError, match="sidecar write failure"):
            export_run(run, destination, images=False)
    assert not destination.exists()
    assert set(tmp_path.iterdir()) == {run}
    assert before == {path.name: path.read_bytes() for path in run.iterdir()}
    assert export_run(run, destination, images=False).is_file()
