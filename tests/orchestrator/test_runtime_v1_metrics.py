"""Runtime v1 metrics contract regressions."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from ora.agent.context.request_size import measured_pass_record
from ora.orchestrator.metrics import _native_request_metrics, build_runtime_metrics, verify_runtime_metrics, write_runtime_metrics


def test_runtime_metrics_separate_context_workload_harness_input_and_latency(
    tmp_path: Path,
) -> None:
    run = tmp_path / "metrics-run"
    (run / "native").mkdir(parents=True)
    (run / "context-bundles").mkdir()
    (run / "plan.json").write_text(json.dumps({
        "plan_sha256": "metrics-plan", "plan_id": "metrics-test",
    }))
    (run / "result.json").write_text(json.dumps({
        "run_id": "metrics-run", "plan_sha256": "metrics-plan",
        "started_unix_s": 100.0, "finished_unix_s": 112.0,
        "wall_duration_s": 11.5,
    }))
    events = [
        {"type": "observation", "time": 100.0, "duration_s": 0.2},
        {"type": "action_submitted", "time": 104.0},
        {"type": "motion_receipt", "time": 105.0, "duration_s": 0.8},
        {"type": "observation", "time": 106.0, "duration_s": 0.3},
    ]
    (run / "events.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in events)
    )
    (run / "agent-events.jsonl").write_text(json.dumps({
        "type": "action_proposed", "model_call_id": "turn-0", "time": 103.6,
    }) + "\n")
    usage_events = [
        {
            "sequence": 0, "turn_id": "turn-0", "model_context_window": 1000,
            "last": {"input_tokens": 100, "cached_input_tokens": 80,
                     "cache_read_input_tokens": 80, "cache_creation_input_tokens": 10,
                     "cache_read_status": "reported", "cache_creation_status": "reported",
                     "output_tokens": 20, "reasoning_output_tokens": 5,
                     "total_tokens": 120},
            "total": {"input_tokens": 100, "cached_input_tokens": 80,
                      "output_tokens": 20, "reasoning_output_tokens": 5,
                      "total_tokens": 120},
        },
        {
            "sequence": 1, "turn_id": "turn-1", "model_context_window": 1000,
            "last": {"input_tokens": 150, "cached_input_tokens": 120,
                     "cache_read_input_tokens": 120, "cache_creation_input_tokens": 5,
                     "cache_read_status": "reported", "cache_creation_status": "reported",
                     "output_tokens": 30, "reasoning_output_tokens": 7,
                     "total_tokens": 180},
            "total": {"input_tokens": 250, "cached_input_tokens": 200,
                      "output_tokens": 50, "reasoning_output_tokens": 12,
                      "total_tokens": 300},
        },
    ]
    (run / "native/agent-session.json").write_text(json.dumps({
        "backend": "codex_app_server_v2", "thread_id": "thread-1",
        "process_invocations": 1, "process_restarts": 0,
        "usage_events": usage_events, "compaction_events": [], "rollovers": [],
        "calls": [
            {"call": 0, "turn_id": "turn-0", "message_kind": "initial",
             "project_message_chars": 80, "project_message_bytes": 100,
             "started_unix_s": 100.5, "finished_unix_s": 103.5,
             "latency_s": 3.0},
            {"call": 1, "turn_id": "turn-1", "message_kind": "observation_delta",
             "project_message_chars": 120, "project_message_bytes": 150,
             "started_unix_s": 106.5, "finished_unix_s": 109.0,
             "latency_s": 2.5},
        ],
    }))
    (run / "context-bundles/call-0000.json").write_text(json.dumps({
        "call": 0,
        "image_attachment": {
            "bytes": 2048, "derivation": {"maximum_size_px": [512, 512]},
        },
    }))

    metrics = write_runtime_metrics(run)
    assert metrics == build_runtime_metrics(run)
    assert metrics["context"]["peak_request_input_tokens"] == 150
    assert metrics["context"]["model_context_window_tokens"] == 1000
    assert metrics["context"]["peak_request_input_utilization"] == 0.15
    assert metrics["context"]["observed_input_headroom_tokens"] == 850
    assert metrics["context"]["terminal_turn_input_growth_tokens"]["p50"] == 50.0
    assert metrics["workload"]["input_tokens"] == 250
    assert metrics["workload"]["cached_input_tokens"] == 200
    assert metrics["workload"]["uncached_input_tokens"] == 50
    assert metrics["cache"] == {
        "measurement": "provider_reported_cache_telemetry",
        "status": "reported",
        "exchange_count": 2,
        "cache_read_telemetry_calls": 2,
        "cache_creation_telemetry_calls": 2,
        "cache_read_hit_calls": 2,
        "call_cache_read_hit_rate": 1.0,
        "input_tokens_with_cache_read_telemetry": 250,
        "cache_read_input_tokens": 200,
        "cache_creation_input_tokens": 15,
        "uncached_input_tokens": 35,
        "token_cache_read_ratio": 0.8,
    }
    assert metrics["turns"][0]["cache"]["read_hit"] is True
    assert metrics["turns"][0]["cache"]["read_ratio"] == 0.8
    assert metrics["harness_input"]["text"]["utf8_bytes"]["total"] == 250.0
    assert metrics["harness_input"]["text"]["token_count"] is None
    assert metrics["harness_input"]["images"]["count"] == 1
    assert metrics["latency"]["episode_wall_s"] == 11.5
    assert metrics["latency"]["agent_turn_s"]["p50"] == 2.75
    assert metrics["latency"]["turn_complete_to_action_publish_s"] == {
        "count": 1, "total": 0.5, "mean": 0.5, "p50": 0.5,
        "p95": 0.5, "min": 0.5, "max": 0.5,
        "pairing": "agent_action_proposed_model_call_id_to_outer_action_order",
        "proposing_turns": 1, "outer_actions": 1,
    }
    assert verify_runtime_metrics(run)["status"] == "valid"

    session_path = run / "native/agent-session.json"
    without_cache = json.loads(session_path.read_text())
    for event in without_cache["usage_events"]:
        for key in (
            "cache_read_input_tokens", "cache_creation_input_tokens",
            "cache_read_status", "cache_creation_status",
        ):
            event["last"].pop(key, None)
        event["last"]["cache_read_status"] = "missing"
        event["last"]["cache_creation_status"] = "missing"
    session_path.write_text(json.dumps(without_cache))
    assert build_runtime_metrics(run)["cache"]["status"] == "missing"
    session_path.write_text(json.dumps({
        **without_cache,
        "usage_events": usage_events,
    }))

    path = run / "runtime-metrics.json"
    tampered = json.loads(path.read_text())
    tampered["workload"]["input_tokens"] = 999
    path.write_text(json.dumps(tampered))
    assert verify_runtime_metrics(run)["status"] == "invalid"


def _measured_passes(*passes: tuple[str, int, int]) -> list[dict[str, Any]]:
    """Pass records as the Context pipeline writes them, from (name, pre, post) bytes."""
    return [
        measured_pass_record(name, (pre_bytes, set()), (post_bytes, set()))
        for name, pre_bytes, post_bytes in passes
    ]


def test_native_pass_byte_reduction_is_bytes_each_pass_removed_across_requests(
    tmp_path: Path,
) -> None:
    """`context.projection.pass_byte_reduction` summarizes `pre_bytes - post_bytes`.

    Every Context pass records the canonical request size before and after it
    runs; the report gives "the byte reduction of each L1-L4 pass" (Native
    Harness v1), so a pass that shrinks the request is a positive value.
    The values of one pass name are pooled over every native request.
    """
    projection, _communication = _native_request_metrics(tmp_path, [
        {"context_projection": {"passes": _measured_passes(
            ("l1_large_result_reference", 9_000, 6_000),
            ("l3_lifecycle_media_reference", 6_000, 1_000),
        )}},
        {"context_projection": {"passes": _measured_passes(
            ("l1_large_result_reference", 7_000, 6_500),
            ("l3_lifecycle_media_reference", 8_000, 2_000),
        )}},
    ], [])
    assert projection["pass_byte_reduction"] == {
        "l1_large_result_reference": {
            "count": 2, "total": 3500.0, "mean": 1750.0, "p50": 1750.0,
            "p95": 2875.0, "min": 500.0, "max": 3000.0, "increased_count": 0,
        },
        "l3_lifecycle_media_reference": {
            "count": 2, "total": 11000.0, "mean": 5500.0, "p50": 5500.0,
            "p95": 5950.0, "min": 5000.0, "max": 6000.0, "increased_count": 0,
        },
    }


def test_native_pass_byte_reduction_keeps_a_pass_that_grew_the_request(
    tmp_path: Path,
) -> None:
    """A pass that makes the request larger is a negative reduction, not dropped.

    Materializing recoverable payloads can grow the canonical request. Dropping
    those values reported such a pass as no data and overstated the mean
    reduction of a pass that sometimes grew it; `increased_count` names them.
    """
    projection, _communication = _native_request_metrics(tmp_path, [
        {"context_projection": {"passes": _measured_passes(
            ("recoverable_payload_materialization", 3_000, 4_000),
            ("l1_large_result_reference", 9_000, 6_000),
        )}},
        {"context_projection": {"passes": _measured_passes(
            ("recoverable_payload_materialization", 2_000, 2_500),
            ("l1_large_result_reference", 5_000, 6_000),
        )}},
    ], [])
    assert projection["pass_byte_reduction"] == {
        "l1_large_result_reference": {
            "count": 2, "total": 2000.0, "mean": 1000.0, "p50": 1000.0,
            "p95": 2800.0, "min": -1000.0, "max": 3000.0, "increased_count": 1,
        },
        "recoverable_payload_materialization": {
            "count": 2, "total": -1500.0, "mean": -750.0, "p50": -750.0,
            "p95": -525.0, "min": -1000.0, "max": -500.0, "increased_count": 2,
        },
    }


def test_native_pass_byte_reduction_counts_a_pass_that_removed_nothing(
    tmp_path: Path,
) -> None:
    """A pass that ran without shrinking the request is one zero reduction.

    That keeps it apart from a pass that never ran (no key). A record without
    a name or without both byte counts measures nothing and is not counted.
    """
    passes = [
        *_measured_passes(
            ("l3_lifecycle_media_reference", 4_000, 1_000),
            ("l4_read_only_result_reference", 1_000, 1_000),
        ),
        {"name": "l2_exact_duplicate_reference", "pre_bytes": 4_000},
        {"pre_bytes": 4_000, "post_bytes": 1_000},
    ]
    projection, _communication = _native_request_metrics(
        tmp_path, [{"context_projection": {"passes": passes}}], [],
    )
    assert projection["pass_byte_reduction"] == {
        "l3_lifecycle_media_reference": {
            "count": 1, "total": 3000.0, "mean": 3000.0, "p50": 3000.0,
            "p95": 3000.0, "min": 3000.0, "max": 3000.0, "increased_count": 0,
        },
        "l4_read_only_result_reference": {
            "count": 1, "total": 0.0, "mean": 0.0, "p50": 0.0,
            "p95": 0.0, "min": 0.0, "max": 0.0, "increased_count": 0,
        },
    }
