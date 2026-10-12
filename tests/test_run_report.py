"""Read-only stop-reason and cost report over synthetic run directories.

Covers the ending classes, validity (evaluation, learning eligibility and
post-seal corrections), the policy finish outcome with and without the v2
finish contract, effort, provider tokens and API-equivalent cost, context
sizes, cohort/task/total aggregation, discovery and the CLI's read-only
boundary. Nothing here starts a simulator or a provider.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from ora.orchestrator import run_report
from ora.orchestrator.run_report_events import PRICING_PATH, api_equivalent_cost, load_pricing
from ora.orchestrator.run_report_run import classify_ending
from tests.run_report_fixture import round_record, tree_digest, usage, write_correction, write_run

ROOT = Path(__file__).resolve().parents[1]
ASTRA = {"gpt-6-astra": {"input": 10.0, "cached_input": 1.0, "output": 50.0}}


@pytest.mark.parametrize(("record", "ending"), [
    (round_record(success=True, termination="trusted_runtime_boundary", authority="environment",
                  upstream="official_success_latch", reason="official_success"), "success_latch"),
    (round_record(termination="trusted_runtime_boundary", authority="environment",
                  upstream="official_failure_latch"), "failure_latch"),
    (round_record(termination="trusted_runtime_boundary", authority="environment",
                  upstream="native_action_limit"), "native_action_limit"),
    (round_record(termination="trusted_runtime_boundary", authority="environment",
                  upstream="official_episode_end"), "episode_end"),
    (round_record(termination="robot_decision_budget"), "decision_budget"),
    (round_record(termination="model_call_budget"), "model_call_budget"),
    (round_record(termination="wall_time_budget"), "wall_time_budget"),
    (round_record(termination="policy_finished_episode", authority="policy_lifecycle"), "policy_finish"),
    (round_record(success=None, termination="operator_interrupt", authority="operator"), "operator_interrupt"),
    (round_record(success=None, termination="context_exhausted", authority="infrastructure"), "infrastructure"),
    (round_record(success=None, termination="trusted_fault", authority="infrastructure"), "infrastructure"),
    # A v2 abort admitted on recorded infrastructure keeps the policy literal.
    (round_record(success=None, termination="policy_finished_episode", authority="infrastructure",
                  sample_valid=False), "infrastructure"),
    # RoboTwin and LIBERO name the environment's own success, without an upstream field.
    (round_record(success=True, termination="trusted_runtime_boundary", authority="environment",
                  reason="official_success"), "success_latch"),
    (round_record(success=False, termination="trusted_runtime_boundary", authority="environment",
                  reason="official_success"), "environment_end"),
    (round_record(success=None, termination=None, authority=None, state="running"), "in_progress"),
])
def test_every_round_names_one_ending_class(record, ending):
    assert classify_ending(record) == ending


def test_shipped_pricing_table_prices_only_the_user_provided_models():
    pricing = load_pricing()
    assert PRICING_PATH.suffix == ".yaml" and pricing["label"] == "api_equivalent"
    assert pricing["currency"] == "USD" and pricing["unit"] == "per_million_tokens"
    assert pricing["models"] == {
        "gpt-6-astra": {"input": 10.0, "cached_input": 1.0, "output": 50.0, "source": pricing["models"]
                        ["gpt-6-astra"]["source"]},
        "gpt-6.1-sol": {"input": 2.0, "cached_input": 0.1, "output": 10.0, "source": pricing["models"]
                        ["gpt-6.1-sol"]["source"]},
    }


def test_cost_is_api_equivalent_and_null_for_an_unpriced_model():
    tokens = {"input": 3_000_000, "cached_input": 2_000_000, "output": 100_000}
    priced = api_equivalent_cost(tokens, "gpt-6-astra", {"models": ASTRA})
    assert priced["label"] == "api_equivalent_usd"
    assert priced["usd"] == pytest.approx(10.0 + 2.0 + 5.0)
    assert priced["components_usd"] == {"uncached_input": 10.0, "cached_input": 2.0, "output": 5.0}
    unpriced = api_equivalent_cost(tokens, "deepseek-flash", {"models": ASTRA})
    assert unpriced["usd"] is None and unpriced["priced"] is False and unpriced["model"] == "deepseek-flash"


def test_tokens_cache_share_and_cost_come_from_provider_responses(tmp_path):
    run = write_run(tmp_path, "run-1", events={"responses": [
        usage(1_000_000, 750_000, 2_000, 500), usage(1_000_000, 800_000, 1_000, 0, legacy=True), None,
    ]})
    report = run_report.build_report([run], pricing={"models": ASTRA}, corrections=[])
    tokens = report["runs"][0]["tokens"]
    assert tokens["source"] == "agent_events_provider_response"
    assert (tokens["input"], tokens["cached_input"], tokens["uncached_input"]) == (2_000_000, 1_550_000, 450_000)
    assert (tokens["output"], tokens["reasoning_output"]) == (3_000, 500)
    assert tokens["cache_share"] == pytest.approx(0.775)
    assert tokens["responses"] == 3 and tokens["responses_without_usage"] == 1
    cost = report["runs"][0]["cost"]
    assert cost["usd"] == pytest.approx(4.5 + 1.55 + 0.15)
    assert report["total"]["cost"]["usd"] == pytest.approx(cost["usd"])


def test_runtime_metrics_workload_is_the_token_fallback(tmp_path):
    run = write_run(tmp_path, "run-1")
    run.joinpath("runtime-metrics.json").write_text(json.dumps({"workload": {
        "input_tokens": 1000, "cached_input_tokens": 400, "output_tokens": 30, "reasoning_output_tokens": 5,
    }}), encoding="utf-8")
    tokens = run_report.build_report([run], pricing={"models": {}}, corrections=[])["runs"][0]["tokens"]
    assert tokens["source"] == "runtime_metrics_workload"
    assert (tokens["input"], tokens["cached_input"], tokens["output"]) == (1000, 400, 30)


def test_validity_counts_only_valid_evaluated_attempts(tmp_path):
    good = write_run(tmp_path, "good", rounds=[round_record(
        success=True, termination="trusted_runtime_boundary", authority="environment",
        upstream="official_success_latch")])
    stopped = write_run(tmp_path, "stopped", state="incomplete", termination="operator_interrupt",
                        authority="operator", evaluation="incomplete", eligible=False, rounds=[
                            round_record(success=None, termination="operator_interrupt", authority="operator")])
    report = run_report.build_report([good, stopped], pricing={"models": ASTRA}, corrections=[])
    runs = {item["run_id"]: item for item in report["runs"]}
    assert runs["good"]["validity"] == "valid" and runs["good"]["learning_eligible"] is True
    assert runs["stopped"]["validity"] == "incomplete" and runs["stopped"]["learning_eligible"] is False
    assert [item["valid_attempt"] for item in runs["stopped"]["attempts"]] == [False]
    total = report["total"]
    assert (total["attempts"], total["valid_attempts"], total["successes"]) == (2, 1, 1)
    assert total["success_rate"] == 1.0 and total["excluded_attempts"] == {"incomplete": 1}
    assert total["endings"] == {"operator_interrupt": 1, "success_latch": 1}
    assert total["learning_eligible_runs"] == 1


def test_a_post_seal_correction_from_the_namespace_marks_a_run_infrastructure_invalid(tmp_path):
    corrected = write_run(tmp_path, "conveyor", rounds=[round_record(
        success=True, termination="trusted_runtime_boundary", authority="environment",
        upstream="official_success_latch")])
    write_run(tmp_path, "other")
    path = write_correction(tmp_path, "C-001", corrected)
    report = run_report.build_report([tmp_path / "runs" / "conveyor", tmp_path / "runs" / "other"],
                                     pricing={"models": ASTRA})
    run = next(item for item in report["runs"] if item["run_id"] == "conveyor")
    assert run["validity"] == "infrastructure_invalid" and run["learning_eligible"] is False
    assert run["correction"]["correction_id"] == "C-001" and run["correction"]["binding"] == "matched"
    assert run["correction"]["verified_files"] == ["evaluation.json", "seal.json"]
    assert run["correction"]["mismatched_files"] == []
    assert run["evaluation"]["status"] == "valid"  # the sealed record is reported as sealed
    assert report["corrections"] == [{"path": str(path), "correction_id": "C-001", "runs": ["conveyor"]}]
    total = report["total"]
    assert total["excluded_attempts"] == {"infrastructure_invalid": 1}
    # The corrected run's sealed success and eligibility count nowhere; it is still listed.
    assert (total["valid_attempts"], total["successes"], total["learning_eligible_runs"]) == (1, 0, 1)
    assert total["validity"] == {"infrastructure_invalid": 1, "valid": 1}
    assert "infrastructure_invalid (C-001)" in run_report.render_markdown(report)


@pytest.mark.parametrize(("cited", "mismatched"), [
    ({"evaluation_sha256": "0" * 64}, ["evaluation.json"]),
    ({"seal_sha256": "1" * 64}, ["seal.json"]),
])
def test_a_correction_citing_other_sealed_bytes_is_applied_and_flagged(tmp_path, cited, mismatched):
    run = write_run(tmp_path, "conveyor")
    path = write_correction(tmp_path, "C-002", run, **cited)
    report = run_report.build_report([run], pricing={"models": ASTRA})
    record = report["runs"][0]
    assert record["validity"] == "infrastructure_invalid" and record["learning_eligible"] is False
    assert record["correction"]["binding"] == "mismatch"
    assert record["correction"]["mismatched_files"] == mismatched
    assert f"correction_sha256_mismatch:{mismatched[0]}" in record["warnings"]
    assert report["warnings"] == [{"path": str(path), "run_id": "conveyor",
                                   "error": f"cited sha256 differs from {mismatched[0]}"}]
    assert f"cited sha256 differs from {mismatched[0]}" in run_report.render_markdown(report)


def test_a_named_correction_folder_applies_and_a_foreign_file_is_only_a_warning(tmp_path):
    run = write_run(tmp_path, "conveyor")
    write_correction(tmp_path, "C-003", run)
    folder = tmp_path / "diagnostics" / "corrections"
    (folder / "notes.json").write_text('{"note": "operator notes"}', encoding="utf-8")
    report = run_report.build_report([run], pricing={"models": ASTRA}, corrections=[folder])
    assert report["runs"][0]["validity"] == "infrastructure_invalid" and report["errors"] == []
    assert report["warnings"] == [{"path": str(folder / "notes.json"),
                                   "error": "not a post-seal correction record (no runs list)"}]
    assert "Warning" in run_report.render_markdown(report)
    unchanged = run_report.build_report([run], pricing={"models": ASTRA}, corrections=[])
    assert unchanged["runs"][0]["validity"] == "valid" and unchanged["corrections"] == []


def test_v2_finish_summary_and_evaluation_report_are_read(tmp_path):
    finish = {"schema": "ora.policy-finish-summary.v1", "outcome": "give_up",
              "category": "unreachable", "reason": "the socket is outside the right arm workspace",
              "ended_round": True}
    run = write_run(tmp_path, "v2", rounds=[round_record(
        termination="policy_finished_episode", authority="policy_lifecycle", policy_finish=finish)])
    from_evaluation = write_run(tmp_path, "evaluation-only", rounds=[round_record(
        termination="policy_finished_episode", authority="policy_lifecycle")], evaluation_extra={
        "policy_finish": [{"round_index": 0, "outcome": "give_up", "category": "object_lost"}]})
    report = run_report.build_report([run, from_evaluation], pricing={"models": ASTRA}, corrections=[])
    first, second = (item["attempts"][0]["policy_finish"] for item in report["runs"])
    assert first == {"outcome": "give_up", "category": "unreachable",
                     "reason": "the socket is outside the right arm workspace", "source": "round_record"}
    assert second["category"] == "object_lost" and second["source"] == "evaluation"
    assert report["total"]["policy_finish"] == {"give_up": 2}
    assert report["total"]["give_up_categories"] == {"object_lost": 1, "unreachable": 1}


def test_a_released_finish_call_is_attributed_to_its_round_and_absence_is_not_an_error(tmp_path):
    rounds = [
        round_record(0, termination="policy_finished_episode", authority="policy_lifecycle", start=1000.0),
        round_record(1, termination="robot_decision_budget", start=2000.0),
    ]
    run = write_run(tmp_path, "released", rounds=rounds, planned_rounds=2, events={"finishes": [
        (1100.0, {"outcome": "failure", "reason": "cup fell off the table"}, True),
        (1050.0, {"outcome": "success", "reason": "x"}, False),
    ]})
    plain = write_run(tmp_path, "plain")
    report = run_report.build_report([run, plain], pricing={"models": ASTRA}, corrections=[])
    attempts = report["runs"][0]["attempts"]
    assert attempts[0]["policy_finish"] == {"outcome": "failure", "category": None,
                                            "reason": "cup fell off the table", "source": "tool_call"}
    assert attempts[1]["policy_finish"] is None
    assert report["runs"][1]["attempts"][0]["policy_finish"] is None


def test_effort_tool_mix_and_refusals_by_tool_and_code(tmp_path):
    run = write_run(tmp_path, "run-1", rounds=[round_record(decisions=7, calls=12, minutes=3.5, refused=2)],
                    events={"tools": [("move_pose", True), ("move_pose", False, "stale_observation"),
                                      ("compute", False, "invalid_arguments"),
                                      ("compute", False, "invalid_arguments"), ("vision_ground", True),
                                      ("mark_point", False, "depth_discontinuity", "batch")]})
    record = run_report.build_report([run], pricing={"models": ASTRA}, corrections=[])["runs"][0]
    effort = record["effort"]
    assert (effort["decisions"], effort["model_calls"], effort["refused_proposals"]) == (7, 12, 2)
    assert effort["wall_minutes"] == pytest.approx(3.5)
    assert effort["tool_calls"] == {"compute": 2, "mark_point": 1, "move_pose": 2, "vision_ground": 1}
    assert effort["refused_tool_calls"] == {
        "compute": {"count": 2, "codes": {"invalid_arguments": 2}},
        "mark_point": {"count": 1, "codes": {"depth_discontinuity": 1}},
        "move_pose": {"count": 1, "codes": {"stale_observation": 1}},
    }
    assert effort["refused_tool_call_total"] == 4


def test_context_sizes_are_request_input_tokens_and_observation_text(tmp_path):
    run = write_run(tmp_path, "run-1", events={
        "responses": [usage(100, 0, 1), usage(300, 0, 1), usage(200, 0, 1)],
        "observation_bytes": [2500, 2600, 2700], "images": [3, 7, 9]})
    context = run_report.build_report([run], pricing={"models": ASTRA}, corrections=[])["runs"][0]["context"]
    assert (context["request_input_tokens"]["p50"], context["request_input_tokens"]["max"]) == (200, 300)
    assert (context["observation_text_bytes"]["p50"], context["observation_text_bytes"]["max"]) == (2600, 2700)
    assert context["inline_images_per_request"]["max"] == 9


def test_cohort_task_and_total_aggregates(tmp_path):
    def success(index=0):
        return round_record(index, success=True, termination="trusted_runtime_boundary", authority="environment",
                            upstream="official_success_latch")

    write_run(tmp_path, "a1", cohort="g006", task="pick", rounds=[success()],
              events={"responses": [usage(1_000_000, 500_000, 1000)]})
    write_run(tmp_path, "a2", cohort="g006", task="place", events={"responses": [usage(2_000_000, 0, 0)]})
    write_run(tmp_path, "a3", cohort="g007", task="pick", rounds=[success()])
    write_run(tmp_path, "s1", cohort="g005", task="pick", model="gpt-6.1-sol")
    report = run_report.build_report(run_report.discover_runs(tmp_path, ["g006", "g007"]),
                                     pricing={"models": ASTRA}, corrections=[])
    cohorts = {item["key"]["cohort_id"]: item for item in report["cohorts"]}
    assert set(cohorts) == {"g006", "g007"}
    assert (cohorts["g006"]["runs"], cohorts["g006"]["successes"], cohorts["g006"]["valid_attempts"]) == (2, 1, 2)
    assert cohorts["g006"]["tokens"]["cache_share"] == pytest.approx(0.5 / 3, abs=1e-6)
    assert cohorts["g006"]["cost"]["usd"] == pytest.approx(5.0 + 0.5 + 0.05 + 20.0)
    tasks = {(item["key"]["model"], item["key"]["task"]): item for item in report["tasks"]}
    assert tasks[("gpt-6-astra", "pick")]["successes"] == 2 and tasks[("gpt-6-astra", "pick")]["cohorts"] == [
        "g006", "g007"]
    assert (report["total"]["valid_attempts"], report["total"]["successes"]) == (3, 2)


def test_faulted_and_in_progress_sessions_keep_their_unfinished_round(tmp_path):
    faulted = write_run(tmp_path, "faulted", planned_rounds=3, state="incomplete", termination="trusted_fault",
                        authority="infrastructure", evaluation="invalid", eligible=False,
                        rounds=[round_record(0, decisions=10, calls=20)],
                        status={"decisions": 14, "model_calls": 31})
    running = write_run(tmp_path, "running", planned_rounds=2, state="running", result=False, evaluation=None,
                        sealed=False, rounds=[round_record(0)])
    report = run_report.build_report([faulted, running], pricing={"models": ASTRA}, corrections=[])
    first, second = report["runs"]
    assert [item["ending"] for item in first["attempts"]] == ["decision_budget", "infrastructure"]
    assert (first["attempts"][1]["decisions"], first["attempts"][1]["model_calls"]) == (4, 11)
    assert first["validity"] == "invalid" and not any(item["valid_attempt"] for item in first["attempts"])
    assert second["validity"] == "in_progress" and second["state"] == "running" and second["sealed"] is False
    assert [item["ending"] for item in second["attempts"]] == ["decision_budget", "in_progress"]
    assert second["effort"]["wall_minutes"] == pytest.approx(1000.0 / 60.0)


def test_discovery_errors_and_unreadable_runs(tmp_path):
    write_run(tmp_path, "a1", cohort="g006")
    broken = tmp_path / "runs" / "broken"
    broken.mkdir()
    with pytest.raises(run_report.RunReportError, match="g404"):
        run_report.discover_runs(tmp_path, ["g006", "g404"])
    assert [path.name for path in run_report.discover_runs(tmp_path)] == ["a1"]
    report = run_report.build_report([tmp_path / "runs" / "a1", broken], pricing={"models": ASTRA}, corrections=[])
    assert [item["run_id"] for item in report["runs"]] == ["a1"]
    assert report["errors"] == [{"run_dir": str(broken), "error": "plan.json: missing"}]


def test_parallel_and_serial_reports_are_identical(tmp_path):
    runs = [write_run(tmp_path, f"r{index}", task=f"t{index % 2}", events={"responses": [usage(10 + index, 5, 1)]})
            for index in range(4)]
    serial = run_report.build_report(runs, pricing={"models": ASTRA}, corrections=[], jobs=1)
    parallel = run_report.build_report(runs, pricing={"models": ASTRA}, corrections=[], jobs=3)
    assert serial == parallel


def test_cli_writes_json_and_markdown_outside_runs_without_touching_evidence(tmp_path):
    run = write_run(tmp_path / "ns", "a1", cohort="g006", events={"responses": [usage(1_000_000, 0, 0)]})
    before = tree_digest(tmp_path / "ns")
    json_path, markdown_path = tmp_path / "out" / "report.json", tmp_path / "out" / "report.md"
    code = run_report.main(["--namespace", str(tmp_path / "ns"), "--cohort", "g006",
                            "--json", str(json_path), "--markdown", str(markdown_path)])
    assert code == 0 and tree_digest(tmp_path / "ns") == before
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["schema"] == run_report.REPORT_SCHEMA and report["runs"][0]["run_id"] == "a1"
    assert report["total"]["cost"]["usd"] == pytest.approx(10.0)
    text = markdown_path.read_text(encoding="utf-8")
    assert "API-equivalent" in text and "| g006 |" in text and "a1" in text
    with pytest.raises(SystemExit):
        run_report.main([str(run), "--json", str(run / "report.json")])
    assert tree_digest(tmp_path / "ns") == before


def test_script_wrapper_prints_markdown(tmp_path):
    run = write_run(tmp_path, "a1", cohort="g006")
    completed = subprocess.run([str(ROOT / "scripts/run-report"), str(run)], capture_output=True, text=True,
                               check=False, cwd=tmp_path)
    assert completed.returncode == 0, completed.stderr
    assert "API-equivalent" in completed.stdout and "| g006 |" in completed.stdout
