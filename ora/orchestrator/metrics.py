"""Deterministic Runtime workload, context and latency evaluation.

The AgentSession owns raw provider telemetry.  This module derives one stable
run-level report without claiming access to Codex's internal history or KV
cache.  It deliberately keeps provider workload, per-exchange request size and
Harness-owned additions as separate measurements.

Provider usage and cache telemetry live in `metrics_provider_usage`, native
context-projection aggregation in `metrics_native_projection` and the shared
value helpers in `metrics_values`; this module reads the run artifacts,
assembles the report and re-exports their names.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import os
import statistics
from ora.orchestrator.visual_usage import visual_usage

from pathlib import Path
from typing import Any, Iterable, Mapping
from ora.agent.evidence.policy_state import context_bundle_images
from ora.orchestrator.metrics_native_projection import _native_projection_metrics
from ora.orchestrator.metrics_provider_usage import (
    TOKEN_FIELDS, _cache_metrics, _provider_metrics, _usage, _usage_errors,
)
from ora.orchestrator.metrics_values import (
    _integer, _number, _percentile, _round, summarize,
)
from ora.agent.evidence import request_archive

# math, statistics, visual_usage, TOKEN_FIELDS, _cache_metrics, _usage_errors
# and _percentile are unused below; they remain importable from this module.


RUNTIME_METRICS_SCHEMA = "ora.runtime-metrics.v1"


class RuntimeMetricsError(RuntimeError):
    """A required run artifact cannot support deterministic metrics."""


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_object(path: Path, *, required: bool = False) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise RuntimeMetricsError(f"required metrics source is missing: {path.name}")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        if required:
            raise RuntimeMetricsError(f"invalid metrics source {path.name}: {exc}") from exc
        return {}
    if not isinstance(value, dict):
        if required:
            raise RuntimeMetricsError(f"metrics source is not an object: {path.name}")
        return {}
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    values: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            values.append(value)
    return values


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _collection_sha256(root: Path, paths: Iterable[Path]) -> tuple[int, str | None]:
    records = [
        {"path": path.relative_to(root).as_posix(), "sha256": _sha256(path)}
        for path in sorted(paths) if path.is_file()
    ]
    if not records:
        return 0, None
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return len(records), hashlib.sha256(encoded).hexdigest()


def _duration(start: Any, finish: Any) -> float | None:
    started = _number(start)
    finished = _number(finish)
    if started is None or finished is None or finished < started:
        return None
    return _round(finished - started)


def _event_time(value: Mapping[str, Any]) -> float | None:
    for field in ("time", "started", "finished"):
        number = _number(value.get(field))
        if number is not None:
            return number
    return None


def _context_bundles(run_root: Path) -> list[dict[str, Any]]:
    return [
        _load_object(path)
        for path in sorted((run_root / "context-bundles").glob("call-*.json"))
    ]


def _inline_image_payload_bytes(value: Any) -> list[int]:
    """Return decoded inline-image sizes from canonical or provider wire JSON."""

    decoded: list[int] = []

    def append_data(encoded: Any) -> None:
        if not isinstance(encoded, str):
            return
        try:
            decoded.append(len(base64.b64decode(encoded, validate=True)))
        except (ValueError, binascii.Error):
            return

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            block_type = item.get("type")
            if block_type == "image_url":
                image_url = item.get("image_url")
                url = image_url.get("url") if isinstance(image_url, Mapping) else image_url
                if isinstance(url, str) and url.startswith("data:image/"):
                    header, separator, encoded = url.partition(",")
                    if separator and header.endswith(";base64"):
                        append_data(encoded)
                return
            if block_type == "image":
                source = item.get("source")
                if (
                    isinstance(source, Mapping)
                    and source.get("type") == "base64"
                    and str(source.get("media_type", "")).startswith("image/")
                ):
                    append_data(source.get("data"))
                    return
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return decoded


def _artifact_path(run_root: Path, ref: Any) -> Path | None:
    return (
        run_root / ref["path"]
        if isinstance(ref, Mapping) and isinstance(ref.get("path"), str) else None
    )


def _native_request_inventory(
    run_root: Path, agent_events: list[dict[str, Any]],
) -> tuple[list[int], list[int], list[int]]:
    """Return request file sizes, inline image sizes and images per request.

    Images are counted in the recorded canonical request when it exists and in
    the provider wire request otherwise.
    """
    requests = [
        event for event in agent_events if event.get("type") == "provider_request"
    ]
    request_bytes: list[int] = []
    image_bytes: list[int] = []
    image_count_by_request: list[int] = []
    for event in requests:
        path = _artifact_path(run_root, event.get("artifact_ref"))
        if path is None or not request_archive.exists(run_root, str(path.relative_to(run_root))):
            continue
        request_bytes.append(len(request_archive.read_bytes(run_root, str(path.relative_to(run_root)))))
        payload = event.get("payload")
        canonical_ref = (
            payload.get("canonical_request_ref")
            if isinstance(payload, Mapping) else None
        )
        canonical_path = _artifact_path(run_root, canonical_ref)
        image_path = (canonical_path if canonical_path is not None
                      and request_archive.exists(run_root, str(canonical_path.relative_to(run_root))) else path)
        image_request = json.loads(request_archive.read_bytes(run_root, str(image_path.relative_to(run_root))))
        request_images = _inline_image_payload_bytes(image_request)
        image_bytes.extend(request_images)
        image_count_by_request.append(len(request_images))
    return request_bytes, image_bytes, image_count_by_request


def _native_response_bytes(
    run_root: Path, agent_events: list[dict[str, Any]],
) -> list[int]:
    wire_response_bytes: list[int] = []
    for event in agent_events:
        if event.get("type") != "provider_response":
            continue
        path = _artifact_path(run_root, event.get("artifact_ref"))
        if path is not None and path.is_file():
            wire_response_bytes.append(path.stat().st_size)
    return wire_response_bytes


def _native_request_metrics(
    run_root: Path, calls: list[dict[str, Any]], agent_events: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a native run's context-projection report and its wire communication sizes."""
    request_bytes, image_bytes, image_count_by_request = _native_request_inventory(
        run_root, agent_events,
    )
    wire_response_bytes = _native_response_bytes(run_root, agent_events)
    projection_metrics = _native_projection_metrics(calls, agent_events, request_bytes)
    return projection_metrics, {
        "request_utf8_bytes": summarize(request_bytes),
        "provider_wire_response_bytes": summarize(wire_response_bytes),
        "inline_images_per_request": summarize(image_count_by_request),
        "inline_image_bytes": summarize(image_bytes),
    }


def _harness_metrics(
    session: Mapping[str, Any], bundles: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[int, bool]]:
    calls = session.get("calls")
    calls = calls if isinstance(calls, list) else []
    characters = [
        int(call["project_message_chars"]) for call in calls
        if isinstance(call, Mapping) and _integer(call.get("project_message_chars")) is not None
    ]
    byte_counts = [
        int(call["project_message_bytes"]) for call in calls
        if isinstance(call, Mapping) and _integer(call.get("project_message_bytes")) is not None
    ]
    image_by_call: dict[int, bool] = {}
    image_bytes: list[int] = []
    maximum_sizes: set[tuple[int, int]] = set()
    image_count = 0
    for bundle in bundles:
        call = _integer(bundle.get("call"))
        attachments = context_bundle_images(bundle)
        image_count += len(attachments)
        if call is not None:
            image_by_call[call] = bool(attachments)
        for attachment in attachments:
            size = _integer(attachment.get("bytes"))
            if size is not None:
                image_bytes.append(size)
            derivation = attachment.get("derivation")
            dimensions = (
                derivation.get("maximum_size_px")
                if isinstance(derivation, Mapping) else None
            )
            if (
                isinstance(dimensions, list) and len(dimensions) == 2
                and all(_integer(item) is not None for item in dimensions)
            ):
                maximum_sizes.add((int(dimensions[0]), int(dimensions[1])))
    return {
        "text": {
            "measurement": "exact_project_owned_utf8",
            "characters": summarize(characters),
            "utf8_bytes": summarize(byte_counts),
            "later_turn_utf8_bytes": summarize(byte_counts[1:]),
            "token_count": None,
            "token_count_status": "missing",
            "token_count_reason": "Codex tokenizer and request serialization are not exposed",
        },
        "images": {
            "measurement": "exact_attachment_inventory",
            "count": image_count,
            "bytes": summarize(image_bytes),
            "maximum_size_px": [list(item) for item in sorted(maximum_sizes)],
            "token_count": None,
            "token_count_status": "missing",
            "token_count_reason": "Provider does not report per-modality token usage",
        },
    }, image_by_call


def _latency_metrics(
    result: Mapping[str, Any], session: Mapping[str, Any],
    events: list[dict[str, Any]], agent_events: list[dict[str, Any]],
) -> dict[str, Any]:
    calls = session.get("calls")
    calls = calls if isinstance(calls, list) else []
    turn_latencies = [
        float(call["latency_s"]) for call in calls
        if isinstance(call, Mapping) and _number(call.get("latency_s")) is not None
    ]
    observations = [item for item in events if item.get("type") == "observation"]
    actions = [item for item in events if item.get("type") == "action_submitted"]
    receipts = [item for item in events if item.get("type") == "motion_receipt"]
    tool_events = [
        item for item in events if item.get("type") in {"tool_result", "tool_error"}
    ]
    tool_durations = []
    for event in tool_events:
        value = _number(event.get("duration_s"))
        if value is None:
            value = _duration(event.get("started"), event.get("finished"))
        if value is not None:
            tool_durations.append(value)

    actuation_durations = []
    for event in receipts:
        value = _number(event.get("duration_s"))
        record = event.get("record")
        receipt = record.get("receipt") if isinstance(record, Mapping) else None
        receipt = receipt if isinstance(receipt, Mapping) else {}
        if value is None:
            value = _duration(receipt.get("started_unix_s"), receipt.get("finished_unix_s"))
        if value is not None:
            actuation_durations.append(value)

    observation_durations = [
        float(item["duration_s"]) for item in observations
        if _number(item.get("duration_s")) is not None
    ]
    cycle_durations = []
    for before, after in zip(observations, observations[1:]):
        duration = _duration(_event_time(before), _event_time(after))
        if duration is not None:
            cycle_durations.append(duration)

    observation_to_turn = []
    for observation, call in zip(observations, calls):
        if not isinstance(call, Mapping):
            continue
        duration = _duration(_event_time(observation), call.get("started_unix_s"))
        if duration is not None:
            observation_to_turn.append(duration)

    calls_by_turn = {
        str(call["turn_id"]): call for call in calls
        if isinstance(call, Mapping)
        and isinstance(call.get("turn_id"), str)
    }
    proposing_turns = [
        str(event["model_call_id"])
        for event in agent_events
        if event.get("type") == "action_proposed"
        and isinstance(event.get("model_call_id"), str)
    ]
    turn_to_action = []
    for model_call_id, action in zip(proposing_turns, actions):
        call = calls_by_turn.get(model_call_id)
        if call is None:
            continue
        duration = _duration(
            call.get("finished_unix_s"), _event_time(action)
        )
        if duration is not None:
            turn_to_action.append(duration)

    action_to_receipt = []
    for action, receipt in zip(actions, receipts):
        duration = _duration(_event_time(action), _event_time(receipt))
        if duration is not None:
            action_to_receipt.append(duration)

    receipt_to_observation = []
    for receipt, observation in zip(receipts, observations[1:]):
        duration = _duration(_event_time(receipt), _event_time(observation))
        if duration is not None:
            receipt_to_observation.append(duration)

    episode_wall = _number(result.get("wall_duration_s"))
    if episode_wall is None:
        episode_wall = _duration(result.get("started_unix_s"), result.get("finished_unix_s"))
    run_start = next((item for item in events if item.get("type") == "run_started"), None)
    first_agent_event = next(
        (item for item in events if str(item.get("type", "")).startswith("agent_session_")),
        None,
    )
    time_to_first_agent_event = (
        _duration(_event_time(run_start), _event_time(first_agent_event))
        if isinstance(run_start, Mapping) and isinstance(first_agent_event, Mapping)
        else None
    )
    return {
        "clock": {
            "duration_source": "monotonic" if result.get("wall_duration_s") is not None else "unix_difference",
            "unix_timestamps_are_ordering_evidence": True,
        },
        "episode_wall_s": _round(episode_wall),
        "agent_turn_s": summarize(turn_latencies),
        "tool_call_s": summarize(tool_durations),
        "environment_actuation_s": summarize(actuation_durations),
        "observation_pipeline_s": summarize(observation_durations),
        "decision_cycle_s": summarize(cycle_durations),
        "observation_to_turn_start_s": summarize(observation_to_turn),
        "turn_complete_to_action_publish_s": {
            **summarize(turn_to_action),
            "pairing": "agent_action_proposed_model_call_id_to_outer_action_order",
            "proposing_turns": len(proposing_turns),
            "outer_actions": len(actions),
        },
        "action_publish_to_receipt_s": summarize(action_to_receipt),
        "receipt_to_observation_s": summarize(receipt_to_observation),
        "time_to_first_agent_event_s": time_to_first_agent_event,
        "time_to_first_agent_event_status": (
            "unix_difference" if time_to_first_agent_event is not None else "missing"
        ),
    }


def _cache_attribution_name(
    projection: Mapping[str, Any], request_relation: Any,
    usage: Mapping[str, Any] | None, cache_read_tokens: int | None,
) -> str | None:
    """Name the request-structure cause of one call's cache behavior, if any."""
    cache_attribution_name = None
    if isinstance(projection.get("epoch_transition"), Mapping):
        cache_attribution_name = "explicit_projection_epoch"
    elif request_relation == "provider_image_window_epoch":
        cache_attribution_name = "provider_image_window_epoch"
    elif request_relation == "initial":
        cache_attribution_name = "initial_request"
    elif request_relation == "append_only":
        if (
            not isinstance(usage, Mapping)
            or usage.get("cache_read_status") != "reported"
            or cache_read_tokens is None
        ):
            cache_attribution_name = "append_only_cache_telemetry_missing"
        elif cache_read_tokens > 0:
            cache_attribution_name = "append_only_cache_read"
        else:
            cache_attribution_name = "opaque_provider_behavior"
    return cache_attribution_name


def _turn_cache(
    call: Mapping[str, Any], usage: Mapping[str, Any] | None,
    projection: Mapping[str, Any], cache_read_tokens: int | None,
) -> dict[str, Any]:
    return {
        "read_status": (
            usage.get("cache_read_status")
            if isinstance(usage, Mapping) else "missing"
        ),
        "creation_status": (
            usage.get("cache_creation_status")
            if isinstance(usage, Mapping) else "missing"
        ),
        "read_hit": (
            cache_read_tokens > 0
            if isinstance(usage, Mapping)
            and usage.get("cache_read_status") == "reported"
            and cache_read_tokens is not None
            else None
        ),
        "read_ratio": (
            _round(usage["cache_read_input_tokens"] / usage["input_tokens"])
            if isinstance(usage, Mapping)
            and usage.get("cache_read_status") == "reported"
            and _integer(usage.get("cache_read_input_tokens")) is not None
            and _integer(usage.get("input_tokens")) not in {None, 0}
            else None
        ),
        "stable_prefix_hash": (
            call.get("context_projection", {}).get("stable_prefix_hash")
            if isinstance(call.get("context_projection"), Mapping) else None
        ),
        "cache_epoch": (
            projection.get("cache_epoch")
        ),
    }


def _turn_metrics(
    calls: list[Any], terminal_by_turn: Mapping[str, Any],
    image_by_call: Mapping[int, bool],
) -> tuple[list[dict[str, Any]], list[int], dict[str, dict[str, int]]]:
    """Return the per-call turn records, request-input growth and cache attribution."""
    turns = []
    previous_input: int | None = None
    growth: list[int] = []
    cache_attribution: dict[str, dict[str, int]] = {}
    for index, call in enumerate(calls):
        if not isinstance(call, Mapping):
            continue
        turn_id = call.get("turn_id") or f"call-{index:04d}"
        exchange = terminal_by_turn.get(str(turn_id), {})
        usage = exchange.get("last") if isinstance(exchange, Mapping) else None
        if usage is None:
            usage = _usage(call.get("usage"))
        input_tokens = usage.get("input_tokens") if isinstance(usage, Mapping) else None
        delta = (
            input_tokens - previous_input
            if input_tokens is not None and previous_input is not None else None
        )
        if delta is not None and delta >= 0:
            growth.append(delta)
        if input_tokens is not None:
            previous_input = input_tokens
        projection = (
            call.get("context_projection")
            if isinstance(call.get("context_projection"), Mapping) else {}
        )
        request_relation = projection.get("request_relation")
        cache_read_tokens = (
            _integer(usage.get("cache_read_input_tokens"))
            if isinstance(usage, Mapping) else None
        )
        cache_attribution_name = _cache_attribution_name(
            projection, request_relation, usage, cache_read_tokens,
        )
        if cache_attribution_name is not None:
            bucket = cache_attribution.setdefault(cache_attribution_name, {
                "calls": 0, "input_tokens": 0, "cache_read_input_tokens": 0,
            })
            bucket["calls"] += 1
            if isinstance(usage, Mapping):
                bucket["input_tokens"] += _integer(usage.get("input_tokens")) or 0
                bucket["cache_read_input_tokens"] += (
                    _integer(usage.get("cache_read_input_tokens")) or 0
                )
        turn = {
            "call": index,
            "turn_id": turn_id,
            "message_kind": call.get("message_kind"),
            "project_message_chars": _integer(call.get("project_message_chars")),
            "project_message_bytes": _integer(call.get("project_message_bytes")),
            "automatic_image": image_by_call.get(index),
            "request_usage": usage,
            "cache": _turn_cache(call, usage, projection, cache_read_tokens),
            "request_input_growth_tokens": delta,
            "request_input_utilization": exchange.get("request_input_utilization")
            if isinstance(exchange, Mapping) else None,
            "latency_s": _number(call.get("latency_s")),
            "native_compaction_sequences": call.get("native_compaction_sequences", []),
        }
        if isinstance(request_relation, str):
            turn["cache"].update({
                "request_relation": request_relation,
                "attribution": cache_attribution_name,
                "protected_prefix_sha256": projection.get("protected_prefix_sha256"),
                "epoch_transition": projection.get("epoch_transition"),
            })
        turns.append(turn)
    return turns, growth, cache_attribution


def _cache_request_structure(
    cache_attribution: Mapping[str, Mapping[str, int]],
) -> dict[str, dict[str, Any]]:
    return {
        name: {
            **values,
            "token_cache_read_ratio": (
                _round(values["cache_read_input_tokens"] / values["input_tokens"])
                if values["input_tokens"] else None
            ),
        }
        for name, values in sorted(cache_attribution.items())
    }


def _measurement_quality(
    plan: Mapping[str, Any], provider_status: Any, context_status: str,
    cache: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "provider_usage": provider_status,
        "request_context": context_status,
        "harness_text": "exact_bytes",
        "harness_text_tokens": "missing",
        "image_inventory": "exact",
        "image_tokens": "missing",
        "codex_internal_history": (
            "not_applicable" if plan.get("policy", {}).get("kind") == "native_api" else "opaque"
        ),
        "codex_kv_cache": (
            "not_applicable" if plan.get("policy", {}).get("kind") == "native_api" else "opaque"
        ),
        "provider_cache_telemetry": cache["status"],
    }


def build_runtime_metrics(run_root: Path) -> dict[str, Any]:
    """Derive one immutable-semantics Runtime report from a completed run."""
    run_root = run_root.resolve()
    plan = _load_object(run_root / "plan.json", required=True)
    result = _load_object(run_root / "result.json", required=True)
    session = _load_object(run_root / "native/agent-session.json")
    events = _load_jsonl(run_root / "events.jsonl")
    agent_events = _load_jsonl(run_root / "agent-events.jsonl")
    bundles = _context_bundles(run_root)
    errors: list[str] = []
    context, workload, cache, terminal_by_turn = _provider_metrics(session, errors)
    harness, image_by_call = _harness_metrics(session, bundles)
    calls = session.get("calls")
    calls = calls if isinstance(calls, list) else []
    native_projection: dict[str, Any] | None = None
    native_communication: dict[str, Any] | None = None
    if plan.get("policy", {}).get("kind") == "native_api":
        native_projection, native_communication = _native_request_metrics(
            run_root, calls, agent_events
        )
        context["projection"] = native_projection
        harness["text"]["measurement"] = "exact_native_request_utf8_including_inline_images"
        harness["images"] = {
            "measurement": "exact_inline_data_url_inventory",
            "count": native_communication["inline_image_bytes"]["count"],
            "bytes": native_communication["inline_image_bytes"],
            "per_request": native_communication["inline_images_per_request"],
            "token_count": None,
            "token_count_status": "provider_aggregate_only",
        }
    turns, growth, cache_attribution = _turn_metrics(calls, terminal_by_turn, image_by_call)
    context["terminal_turn_input_growth_tokens"] = summarize(growth)
    if cache_attribution:
        cache["request_structure"] = _cache_request_structure(cache_attribution)
    provider_status = workload.get("status", "missing")
    context_status = (
        "provider_reported_with_window"
        if context.get("exchange_count") and context.get("model_context_window_tokens")
        else ("provider_reported_without_window" if context.get("exchange_count") else "missing")
    )
    quality = _measurement_quality(plan, provider_status, context_status, cache)
    source_paths = {
        "plan": run_root / "plan.json",
        "result": run_root / "result.json",
        "events": run_root / "events.jsonl",
        "agent_session": run_root / "native/agent-session.json",
        "context_manifest": run_root / "context-manifest.json",
    }
    bundle_paths = sorted((run_root / "context-bundles").glob("call-*.json"))
    bundle_count, bundle_sha256 = _collection_sha256(run_root, bundle_paths)
    return {
        "schema": RUNTIME_METRICS_SCHEMA,
        "run_id": result.get("run_id"),
        "plan_sha256": plan.get("plan_sha256"),
        "measurement_status": "complete" if (
            provider_status == "reported"
            and context_status == "provider_reported_with_window"
            and not errors
        )
        else ("partial" if session or events else "missing"),
        "measurement_errors": sorted(set(errors)),
        "measurement_quality": quality,
        "source_artifacts": {
            name: {"path": path.relative_to(run_root).as_posix(), "sha256": _sha256(path)}
            for name, path in source_paths.items()
        } | {
            "context_bundles": {
                "path": "context-bundles", "count": bundle_count,
                "sha256": bundle_sha256,
            },
        },
        "context": context,
        "workload": workload,
        "cache": cache,
        "harness_input": harness,
        "communication": native_communication,
        "latency": _latency_metrics(result, session, events, agent_events),
        "session": {
            "backend": session.get("backend"),
            "session_id": session.get("session_id"),
            "process_pid": session.get("process_pid"),
            "process_invocations": session.get("process_invocations"),
            "process_restarts": session.get("process_restarts"),
            "thread_id": session.get("thread_id"),
            "exit_reason": session.get("exit_reason"),
            "turns": len(calls),
        },
        "turns": turns,
    }


def write_runtime_metrics(run_root: Path) -> dict[str, Any]:
    """Build and atomically persist ``runtime-metrics.json``."""
    value = build_runtime_metrics(run_root)
    _atomic_json(run_root.resolve() / "runtime-metrics.json", value)
    return value


def verify_runtime_metrics(run_root: Path) -> dict[str, Any]:
    """Verify that persisted metrics equal a fresh deterministic derivation."""
    run_root = run_root.resolve()
    path = run_root / "runtime-metrics.json"
    try:
        persisted = _load_object(path, required=True)
        rebuilt = build_runtime_metrics(run_root)
    except RuntimeMetricsError as exc:
        return {"status": "invalid", "errors": [str(exc)], "metrics": {}}
    errors = []
    if persisted.get("schema") != RUNTIME_METRICS_SCHEMA:
        errors.append("runtime_metrics_schema_mismatch")
    if persisted != rebuilt:
        errors.append("runtime_metrics_derivation_mismatch")
    return {
        "status": "valid" if not errors else "invalid",
        "errors": errors,
        "metrics": persisted,
    }
