"""Versioned provider list-price estimates for immutable run telemetry.

This module never reads account balances or invoices.  It combines direct
provider token counters with a checked-in price snapshot so historical Viewer
pages do not silently change when a provider edits its public pricing page.
"""

from __future__ import annotations

import json
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


COST_ESTIMATE_SCHEMA = "ora.viewer-provider-cost-estimate.v1"
PRICE_CATALOG_SCHEMA = "ora.viewer-provider-price-catalog.v1"
CATALOG_PATH = Path(__file__).with_name("provider-pricing.json")


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _nonnegative_integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value < 0 or int(value) != value:
        return None
    return int(value)


def _summary_total(value: Any) -> int | None:
    if not isinstance(value, Mapping):
        return None
    return _nonnegative_integer(value.get("total"))


def load_price_catalog() -> dict[str, Any]:
    catalog = _load_object(CATALOG_PATH)
    if catalog.get("schema") != PRICE_CATALOG_SCHEMA:
        return {}
    return catalog


def _run_date(root: Path) -> str | None:
    for source, field in (
        (_load_object(root / "result.json"), "started_unix_s"),
        (_load_object(root / "launch.json"), "created_unix_s"),
    ):
        value = source.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return datetime.fromtimestamp(float(value), tz=timezone.utc).date().isoformat()
    stamp = root.name.rpartition("--")[2]
    try:
        return datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").date().isoformat()
    except ValueError:
        return None


def _price_for_model(
    model: str, catalog: Mapping[str, Any], *, run_date: str | None,
) -> dict[str, Any] | None:
    normalized = model.strip().casefold()
    prices = catalog.get("prices")
    if not isinstance(prices, list):
        return None
    matches = [
        item for item in prices
        if isinstance(item, Mapping)
        and normalized in {
            str(alias).strip().casefold()
            for alias in item.get("model_aliases", [])
            if isinstance(alias, str)
        }
    ]
    eligible = [
        item for item in matches
        if run_date is None
        or (
            isinstance(item.get("effective_date"), str)
            and item["effective_date"] <= run_date
            and (
                item.get("retired_date") is None
                or str(item["retired_date"]) > run_date
            )
        )
    ]
    if not eligible:
        return None
    return dict(sorted(eligible, key=lambda item: str(item["effective_date"]))[-1])


def _runtime_metric_tokens(metrics: Mapping[str, Any]) -> dict[str, Any] | None:
    workload = metrics.get("workload")
    cache = metrics.get("cache")
    if not isinstance(workload, Mapping) or not isinstance(cache, Mapping):
        return None
    main_input = _nonnegative_integer(workload.get("input_tokens"))
    main_output = _nonnegative_integer(workload.get("output_tokens"))
    main_cached = _nonnegative_integer(cache.get("cache_read_input_tokens"))
    cache_input = _nonnegative_integer(cache.get("input_tokens_with_cache_read_telemetry"))
    if (
        main_input is None or main_output is None or main_cached is None
        or cache_input != main_input or main_cached > main_input
        or cache.get("status") != "reported"
    ):
        return None

    context = metrics.get("context")
    projection = context.get("projection") if isinstance(context, Mapping) else None
    compactor = (
        projection.get("context_compactor")
        if isinstance(projection, Mapping) else None
    )
    auxiliary_input = auxiliary_cached = auxiliary_output = auxiliary_calls = 0
    if isinstance(compactor, Mapping):
        auxiliary_calls = _nonnegative_integer(compactor.get("call_count")) or 0
        auxiliary_input = _summary_total(compactor.get("input_tokens")) or 0
        auxiliary_cached = _summary_total(compactor.get("cached_input_tokens")) or 0
        auxiliary_output = _summary_total(compactor.get("output_tokens")) or 0
        if auxiliary_cached > auxiliary_input:
            return None
    return {
        "measurement": "sealed_runtime_metrics",
        "main": {
            "input_tokens": main_input,
            "cache_read_input_tokens": main_cached,
            "uncached_input_tokens": main_input - main_cached,
            "output_tokens": main_output,
        },
        "auxiliary": {
            "kind": "context_compactor",
            "calls": auxiliary_calls,
            "input_tokens": auxiliary_input,
            "cache_read_input_tokens": auxiliary_cached,
            "uncached_input_tokens": auxiliary_input - auxiliary_cached,
            "output_tokens": auxiliary_output,
        },
    }


def _live_session_tokens(session: Mapping[str, Any]) -> dict[str, Any] | None:
    total = session.get("reported_totals")
    if not isinstance(total, Mapping):
        return None
    main_input = _nonnegative_integer(total.get("input_tokens"))
    main_cached = _nonnegative_integer(total.get("cached_input_tokens"))
    main_output = _nonnegative_integer(total.get("output_tokens"))
    if (
        main_input is None or main_cached is None or main_output is None
        or main_cached > main_input
    ):
        return None
    auxiliary_input = auxiliary_cached = auxiliary_output = 0
    calls = session.get("context_compactor_calls")
    calls = calls if isinstance(calls, list) else []
    for call in calls:
        usage = call.get("usage") if isinstance(call, Mapping) else None
        if not isinstance(usage, Mapping):
            return None
        input_tokens = _nonnegative_integer(usage.get("input_tokens"))
        cached_tokens = _nonnegative_integer(usage.get("cached_input_tokens"))
        output_tokens = _nonnegative_integer(usage.get("output_tokens"))
        if (
            input_tokens is None or cached_tokens is None or output_tokens is None
            or cached_tokens > input_tokens
        ):
            return None
        auxiliary_input += input_tokens
        auxiliary_cached += cached_tokens
        auxiliary_output += output_tokens
    return {
        "measurement": "provisional_live_session",
        "main": {
            "input_tokens": main_input,
            "cache_read_input_tokens": main_cached,
            "uncached_input_tokens": main_input - main_cached,
            "output_tokens": main_output,
        },
        "auxiliary": {
            "kind": "context_compactor",
            "calls": len(calls),
            "input_tokens": auxiliary_input,
            "cache_read_input_tokens": auxiliary_cached,
            "uncached_input_tokens": auxiliary_input - auxiliary_cached,
            "output_tokens": auxiliary_output,
        },
    }


def _money(tokens: int, rate_per_million: Decimal) -> Decimal:
    return Decimal(tokens) * rate_per_million / Decimal(1_000_000)


def _decimal_number(value: Decimal) -> float:
    return float(value.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))


def estimate_run_cost(run_root: Path) -> dict[str, Any]:
    """Return a read-only list-price estimate for one run.

    Sealed ``runtime-metrics.json`` is preferred.  A running Native session can
    be shown provisionally without writing any derived file into the run.
    """

    root = run_root.resolve()
    plan = _load_object(root / "plan.json")
    policy = plan.get("policy") if isinstance(plan.get("policy"), Mapping) else {}
    model = str(policy.get("model") or "")
    catalog = load_price_catalog()
    run_date = _run_date(root)
    price = _price_for_model(model, catalog, run_date=run_date)
    base = {
        "schema": COST_ESTIMATE_SCHEMA,
        "run_id": root.name,
        "model": model,
        "run_date": run_date,
        "basis": "official_api_list_price_equivalent",
    }
    if price is None:
        return {**base, "status": "unpriced_model"}

    metrics = _load_object(root / "runtime-metrics.json")
    tokens = _runtime_metric_tokens(metrics) if metrics else None
    if tokens is None:
        session = _load_object(root / "native/agent-session.json")
        tokens = _live_session_tokens(session) if session else None
    if tokens is None:
        return {
            **base,
            "status": "missing_usage_or_cache_telemetry",
            "price": price,
        }

    main = tokens["main"]
    auxiliary = tokens["auxiliary"]
    total_cached = main["cache_read_input_tokens"] + auxiliary["cache_read_input_tokens"]
    total_uncached = main["uncached_input_tokens"] + auxiliary["uncached_input_tokens"]
    total_output = main["output_tokens"] + auxiliary["output_tokens"]
    rates = price.get("rates_per_million_tokens")
    if not isinstance(rates, Mapping):
        return {**base, "status": "invalid_price_catalog"}
    cached_rate = Decimal(str(rates["cache_hit_input"]))
    uncached_rate = Decimal(str(rates["cache_miss_input"]))
    output_rate = Decimal(str(rates["output"]))
    cached_cost = _money(total_cached, cached_rate)
    uncached_cost = _money(total_uncached, uncached_rate)
    output_cost = _money(total_output, output_rate)
    total_cost = cached_cost + uncached_cost + output_cost
    run_status = _load_object(root / "status.json")
    status = "estimated"
    if tokens["measurement"] == "provisional_live_session":
        status = (
            "estimated_session_fallback"
            if run_status.get("state") == "completed" else "provisional"
        )
    return {
        **base,
        "status": status,
        "measurement": tokens["measurement"],
        "price": price,
        "tokens": {
            "main": main,
            "auxiliary": auxiliary,
            "total": {
                "input_tokens": total_cached + total_uncached,
                "cache_read_input_tokens": total_cached,
                "uncached_input_tokens": total_uncached,
                "output_tokens": total_output,
            },
        },
        "cost": {
            "currency": price["currency"],
            "cache_hit_input": _decimal_number(cached_cost),
            "cache_miss_input": _decimal_number(uncached_cost),
            "output": _decimal_number(output_cost),
            "total": _decimal_number(total_cost),
        },
    }
