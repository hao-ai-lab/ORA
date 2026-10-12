"""Provider-reported token usage, its validation and cache telemetry.

`_usage` normalizes one usage record, `_usage_errors` names what makes it
inconsistent, `_cache_metrics` aggregates provider cache read/creation
telemetry and `_provider_metrics` derives the request context, cumulative
workload and cache reports from an AgentSession's usage events (or its calls
when it reports no events). `metrics` re-exports every name.
"""

from __future__ import annotations

from typing import Any, Mapping

from ora.orchestrator.metrics_values import _integer, _round, summarize


TOKEN_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
    "total_tokens",
)


def _usage(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    normalized = {field: _integer(value.get(field)) for field in TOKEN_FIELDS}
    if all(item is None for item in normalized.values()):
        return None
    input_tokens = normalized["input_tokens"]
    cached_tokens = normalized["cached_input_tokens"]
    normalized["uncached_input_tokens"] = (
        input_tokens - cached_tokens
        if input_tokens is not None and cached_tokens is not None
        and cached_tokens <= input_tokens else None
    )
    cache_read_status = value.get("cache_read_status")
    cache_read = _integer(value.get("cache_read_input_tokens"))
    if (
        "cache_read_status" not in value
        and "cache_read_input_tokens" not in value
        and cached_tokens is not None
    ):
        # Managed Codex historically exposed the provider's cache-read counter
        # only under the compatibility name ``cached_input_tokens``. That
        # field is itself direct provider telemetry, not an inferred hit.
        cache_read = cached_tokens
        cache_read_status = "reported"
    normalized["cache_read_input_tokens"] = cache_read
    normalized["cache_creation_input_tokens"] = _integer(
        value.get("cache_creation_input_tokens")
    )
    for field in ("cache_read_status", "cache_creation_status"):
        raw_status = (
            cache_read_status if field == "cache_read_status"
            else value.get(field)
        )
        normalized[field] = (
            raw_status if raw_status in {"reported", "missing"}
            else "missing" if field not in value
            else "invalid"
        )
    return normalized


def _usage_errors(
    usage: Mapping[str, Any] | None, *, label: str,
) -> list[str]:
    if usage is None:
        return [f"{label}:missing"]
    errors = []
    required = ("input_tokens", "cached_input_tokens", "output_tokens", "total_tokens")
    for field in required:
        if usage.get(field) is None:
            errors.append(f"{label}:{field}:missing")
    input_tokens = usage.get("input_tokens")
    cached_tokens = usage.get("cached_input_tokens")
    if (
        input_tokens is not None and cached_tokens is not None
        and cached_tokens > input_tokens
    ):
        errors.append(f"{label}:cached_exceeds_input")
    cache_read = usage.get("cache_read_input_tokens")
    cache_creation = usage.get("cache_creation_input_tokens")
    for kind in ("read", "creation"):
        status = usage.get(f"cache_{kind}_status")
        amount = usage.get(f"cache_{kind}_input_tokens")
        if status == "invalid":
            errors.append(f"{label}:cache_{kind}_status:invalid")
        if status == "missing" and amount is not None:
            errors.append(f"{label}:cache_{kind}_value_without_telemetry")
    if usage.get("cache_read_status") == "reported":
        if cache_read is None:
            errors.append(f"{label}:reported_cache_read_missing")
        elif input_tokens is not None and cache_read > input_tokens:
            errors.append(f"{label}:cache_read_exceeds_input")
        if cached_tokens is not None and cache_read != cached_tokens:
            errors.append(f"{label}:cache_read_compatibility_mismatch")
    if usage.get("cache_creation_status") == "reported":
        if cache_creation is None:
            errors.append(f"{label}:reported_cache_creation_missing")
        elif input_tokens is not None and cache_creation > input_tokens:
            errors.append(f"{label}:cache_creation_exceeds_input")
    if (
        input_tokens is not None and cache_read is not None and cache_creation is not None
        and cache_read + cache_creation > input_tokens
    ):
        errors.append(f"{label}:cache_categories_exceed_input")
    total_tokens = usage.get("total_tokens")
    output_tokens = usage.get("output_tokens")
    if (
        input_tokens is not None and output_tokens is not None
        and total_tokens is not None and total_tokens != input_tokens + output_tokens
    ):
        errors.append(f"{label}:total_mismatch")
    return errors


def _cache_metrics(exchanges: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    reported_reads: list[Mapping[str, Any]] = []
    reported_creations: list[Mapping[str, Any]] = []
    for exchange in exchanges:
        last = exchange.get("last")
        if not isinstance(last, Mapping):
            continue
        if last.get("cache_read_status") == "reported":
            reported_reads.append(last)
        if last.get("cache_creation_status") == "reported":
            reported_creations.append(last)
    read_pairs = [
        (int(item["input_tokens"]), int(item["cache_read_input_tokens"]), item)
        for item in reported_reads
        if _integer(item.get("input_tokens")) is not None
        and _integer(item.get("cache_read_input_tokens")) is not None
    ]
    read_tokens = [read for _input, read, _item in read_pairs]
    creation_tokens = [
        int(item["cache_creation_input_tokens"])
        for item in reported_creations
        if _integer(item.get("cache_creation_input_tokens")) is not None
    ]
    inputs = [input_tokens for input_tokens, _read, _item in read_pairs]
    total_read = sum(read_tokens)
    total_input = sum(inputs)
    reported_count = len(reported_reads)
    exchange_count = len(exchanges)
    status = (
        "reported" if exchange_count and reported_count == exchange_count
        else "partial" if reported_count else "missing"
    )
    hits = sum(value > 0 for value in read_tokens)
    uncached = 0
    for input_tokens, read, item in read_pairs:
        amount = input_tokens - read
        if item.get("cache_creation_status") == "reported":
            creation = _integer(item.get("cache_creation_input_tokens"))
            if creation is not None:
                amount -= creation
        uncached += amount
    return {
        "measurement": "provider_reported_cache_telemetry",
        "status": status,
        "exchange_count": exchange_count,
        "cache_read_telemetry_calls": reported_count,
        "cache_creation_telemetry_calls": len(reported_creations),
        "cache_read_hit_calls": hits,
        "call_cache_read_hit_rate": (
            _round(hits / reported_count) if reported_count else None
        ),
        "input_tokens_with_cache_read_telemetry": total_input if inputs else None,
        "cache_read_input_tokens": total_read if read_tokens else None,
        "cache_creation_input_tokens": (
            sum(creation_tokens) if creation_tokens else None
        ),
        "uncached_input_tokens": uncached if inputs else None,
        "token_cache_read_ratio": (
            _round(total_read / total_input) if total_input else None
        ),
    }


def _provider_metrics(
    session: Mapping[str, Any], errors: list[str],
) -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, dict[str, Any]],
]:
    backend = session.get("backend")
    raw_events = session.get("usage_events")
    raw_events = raw_events if isinstance(raw_events, list) else []
    exchanges: list[dict[str, Any]] = []
    terminal_by_turn: dict[str, dict[str, Any]] = {}
    previous_totals: dict[str, int] = {}

    for index, raw in enumerate(raw_events):
        if not isinstance(raw, Mapping):
            errors.append(f"usage_event:{index}:invalid")
            continue
        if raw.get("sequence") != index:
            errors.append(f"usage_event:{index}:sequence_mismatch")
        last = _usage(raw.get("last"))
        total = _usage(raw.get("total"))
        errors.extend(_usage_errors(last, label=f"usage_event:{index}:last"))
        errors.extend(_usage_errors(total, label=f"usage_event:{index}:total"))
        window = _integer(raw.get("model_context_window"))
        input_tokens = None if last is None else last.get("input_tokens")
        utilization = (
            input_tokens / window
            if input_tokens is not None and window not in {None, 0} else None
        )
        exchange = {
            "sequence": index,
            "turn_id": raw.get("turn_id"),
            "last": last,
            "total": total,
            "model_context_window": window,
            "request_input_utilization": _round(utilization),
        }
        exchanges.append(exchange)
        if isinstance(raw.get("turn_id"), str):
            terminal_by_turn[str(raw["turn_id"])] = exchange
        if total is not None:
            for field in TOKEN_FIELDS:
                amount = total.get(field)
                prior = previous_totals.get(field)
                if amount is not None and prior is not None and amount < prior:
                    errors.append(f"usage_event:{index}:total_not_monotonic:{field}")
                if amount is not None:
                    previous_totals[field] = amount

    calls = session.get("calls")
    calls = calls if isinstance(calls, list) else []
    if not exchanges and calls:
        cumulative = {field: 0 for field in TOKEN_FIELDS}
        cumulative_complete = {field: True for field in TOKEN_FIELDS}
        for index, call in enumerate(calls):
            if not isinstance(call, Mapping):
                continue
            last = _usage(call.get("usage"))
            errors.extend(_usage_errors(last, label=f"call:{index}:usage"))
            if last is not None:
                for field in TOKEN_FIELDS:
                    amount = last.get(field)
                    if amount is None:
                        cumulative_complete[field] = False
                    elif cumulative_complete[field]:
                        cumulative[field] += amount
            total = {
                field: cumulative[field] if cumulative_complete[field] else None
                for field in TOKEN_FIELDS
            }
            total["uncached_input_tokens"] = (
                total["input_tokens"] - total["cached_input_tokens"]
                if total["input_tokens"] is not None
                and total["cached_input_tokens"] is not None else None
            )
            exchange = {
                "sequence": index,
                "turn_id": call.get("turn_id") or f"call-{index:04d}",
                "last": last,
                "total": total,
                "model_context_window": None,
                "request_input_utilization": None,
            }
            exchanges.append(exchange)
            terminal_by_turn[str(exchange["turn_id"])] = exchange

    windows = sorted({
        int(item["model_context_window"])
        for item in exchanges if item.get("model_context_window") is not None
    })
    if len(windows) > 1:
        errors.append("provider:model_context_window_changed")
    request_inputs = [
        item["last"]["input_tokens"] for item in exchanges
        if isinstance(item.get("last"), Mapping)
        and item["last"].get("input_tokens") is not None
    ]
    utilizations = [
        item["request_input_utilization"] for item in exchanges
        if item.get("request_input_utilization") is not None
    ]
    final_total = exchanges[-1].get("total") if exchanges else None
    if not isinstance(final_total, Mapping):
        final_total = _usage(session.get("reported_totals"))
    provider_errors = [
        item for item in errors if item.startswith(("usage_event", "call", "provider"))
    ]
    quality = "reported" if exchanges and not provider_errors else (
        "partial" if exchanges else "missing"
    )
    window = windows[-1] if windows else None
    peak = max(request_inputs) if request_inputs else None
    context = {
        "measurement": "provider_reported_exchange_input",
        "model_context_window_tokens": window,
        "exchange_count": len(exchanges),
        "peak_request_input_tokens": peak,
        "peak_request_input_utilization": _round(max(utilizations)) if utilizations else None,
        "observed_input_headroom_tokens": (
            window - peak if window is not None and peak is not None else None
        ),
        "native_compactions": len(session.get("compaction_events", []))
        if isinstance(session.get("compaction_events"), list) else 0,
        "harness_rollovers": len(session.get("rollovers", []))
        if isinstance(session.get("rollovers"), list) else 0,
        "request_input_tokens": summarize(request_inputs),
    }
    workload = {
        "measurement": "provider_reported_cumulative_workload",
        "status": quality,
        **({field: final_total.get(field) for field in (*TOKEN_FIELDS, "uncached_input_tokens")}
           if isinstance(final_total, Mapping)
           else {field: None for field in (*TOKEN_FIELDS, "uncached_input_tokens")}),
    }
    workload["reasoning_output_is_subset_of_output"] = True
    workload["cached_input_is_subset_of_input"] = True
    workload["backend"] = backend
    return context, workload, _cache_metrics(exchanges), terminal_by_turn
