"""Provider list-price and cost-dashboard contracts."""

from __future__ import annotations

import json
from pathlib import Path

from viewer import app as viewer
from viewer.pricing import estimate_run_cost


def _run(tmp_path: Path, *, run_id: str = "priced-run") -> Path:
    root = tmp_path / "runs" / run_id
    root.mkdir(parents=True)
    (root / "plan.json").write_text(json.dumps({
        "experiment": "pricing-fixture",
        "config": "s24",
        "milestone": "M2",
        "task": "libero_object_swap-t00",
        "seed": 0,
        "evidence_tier": "behavioral",
        "runtime": {
            "kind": "libero_pro", "suite": "libero_object_swap", "task_id": 0,
        },
        "policy": {"kind": "native_api", "model": "k3"},
        "source": {"commit": "abc123"},
    }))
    (root / "status.json").write_text(json.dumps({
        "state": "completed", "official_success": True,
    }))
    (root / "evaluation.json").write_text(json.dumps({
        "status": "valid", "official_success": True,
    }))
    (root / "seal.json").write_text("{}")
    return root


def test_kimi_cost_uses_cache_categories_and_context_compactor(tmp_path: Path) -> None:
    root = _run(tmp_path)
    (root / "runtime-metrics.json").write_text(json.dumps({
        "schema": "ora.runtime-metrics.v1",
        "run_id": root.name,
        "workload": {
            "input_tokens": 1_000_000, "cached_input_tokens": 750_000,
            "uncached_input_tokens": 250_000, "output_tokens": 100_000,
        },
        "cache": {
            "status": "reported", "cache_read_input_tokens": 750_000,
            "input_tokens_with_cache_read_telemetry": 1_000_000,
        },
        "context": {"projection": {"context_compactor": {
            "call_count": 2,
            "input_tokens": {"total": 100_000},
            "cached_input_tokens": {"total": 20_000},
            "output_tokens": {"total": 10_000},
        }}},
    }))

    estimate = estimate_run_cost(root)

    assert estimate["status"] == "estimated"
    assert estimate["tokens"]["total"] == {
        "input_tokens": 1_100_000,
        "cache_read_input_tokens": 770_000,
        "uncached_input_tokens": 330_000,
        "output_tokens": 110_000,
    }
    assert estimate["cost"] == {
        "currency": "CNY",
        "cache_hit_input": 1.54,
        "cache_miss_input": 6.6,
        "output": 11.0,
        "total": 19.14,
    }


def test_live_cost_is_provisional_and_missing_cache_fails_closed(tmp_path: Path) -> None:
    root = _run(tmp_path, run_id="live-run")
    (root / "status.json").write_text(json.dumps({"state": "running"}))
    native = root / "native"
    native.mkdir()
    (native / "agent-session.json").write_text(json.dumps({
        "reported_totals": {
            "input_tokens": 100_000, "cached_input_tokens": 80_000,
            "output_tokens": 5_000,
        },
        "context_compactor_calls": [{"usage": {
            "input_tokens": 10_000, "cached_input_tokens": 0,
            "output_tokens": 1_000,
        }}],
    }))
    assert estimate_run_cost(root)["status"] == "provisional"

    (native / "agent-session.json").write_text(json.dumps({
        "reported_totals": {"input_tokens": 100_000, "output_tokens": 5_000},
    }))
    assert estimate_run_cost(root)["status"] == "missing_usage_or_cache_telemetry"


def test_cost_dashboard_excludes_non_behavioral_runs_by_registry_view(
    tmp_path: Path,
) -> None:
    root = _run(tmp_path)
    (root / "runtime-metrics.json").write_text(json.dumps({
        "run_id": root.name,
        "workload": {"input_tokens": 1_000_000, "output_tokens": 10_000},
        "cache": {
            "status": "reported", "cache_read_input_tokens": 900_000,
            "input_tokens_with_cache_read_telemetry": 1_000_000,
        },
    }))
    viewer.app.config["VIEWER_OUTPUT_ROOTS"] = (tmp_path,)
    try:
        client = viewer.app.test_client()
        response = client.get("/costs")
        cost_api = client.get("/api/run/priced-run/cost")
    finally:
        viewer.app.config.pop("VIEWER_OUTPUT_ROOTS", None)

    assert response.status_code == 200
    assert b"What did one robot task cost?" in response.data
    assert b"priced-run" in response.data
    assert b"\xc2\xa54.80" in response.data
    assert b"kimi-k3-cny-2026-07-28" in response.data
    assert cost_api.status_code == 200
    assert cost_api.json["cost"]["total"] == 4.8
