"""Provider tokens, tool calls and context sizes of one run, and their API-equivalent cost.

`scan_agent_events` reads a run's ``agent-events.jsonl`` once and keeps only
what the run report needs: provider-reported usage per response, the proposed
tool calls, the refused ones with their error codes (read from the small
tool-result artifact a refusal writes), the current-observation text size and
inline image count of each projected request, and the ``finish_episode``
calls. `metrics_tokens` is the fallback for a run whose events carry no usage
(the sealed ``runtime-metrics.json`` workload). `load_pricing` reads the
per-model price table and `api_equivalent_cost` prices a token total; a model
the table does not list is unpriced, never estimated. Everything here only
reads.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import yaml

from .metrics_values import _integer, _round

PRICING_PATH = Path(__file__).with_name("run_report_pricing.yaml")
PRICING_SCHEMA = "ora.api-pricing.v1"
COST_LABEL = "api_equivalent_usd"
PRICE_FIELDS = ("input", "cached_input", "output")
#: Event types the report reads; every other line is skipped unparsed.
_WANTED = (b'"provider_response"', b'"tool_call_proposed"', b'"tool_result"', b'"context_projected"',
           b'"session_started"')
#: A refusal's tool-result artifact is a few hundred bytes; anything larger is not read.
_ARTIFACT_LIMIT = 65536
_FINISH_TOOL = "finish_episode"
_REASON_LIMIT = 500


class PricingError(ValueError):
    """The price table is not a valid ``ora.api-pricing.v1`` document."""


def load_pricing(path: Path | None = None) -> dict[str, Any]:
    """The validated price table with its path and byte hash."""

    source = PRICING_PATH if path is None else Path(path)
    raw = source.read_bytes()
    data = yaml.safe_load(raw)
    if not isinstance(data, Mapping) or data.get("schema") != PRICING_SCHEMA:
        raise PricingError(f"{source}: schema must be {PRICING_SCHEMA}")
    models = data.get("models")
    if not isinstance(models, Mapping):
        raise PricingError(f"{source}: models must be a mapping")
    for model, prices in models.items():
        if not isinstance(prices, Mapping):
            raise PricingError(f"{source}: {model} must be a mapping")
        for field in PRICE_FIELDS:
            value = prices.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise PricingError(f"{source}: {model}.{field} must be a non-negative number")
    return {**dict(data), "models": {str(key): dict(value) for key, value in models.items()},
            "path": str(source), "sha256": hashlib.sha256(raw).hexdigest()}


def api_equivalent_cost(tokens: Mapping[str, Any], model: str | None, pricing: Mapping[str, Any]) -> dict[str, Any]:
    """What ``tokens`` would cost at the listed API price of ``model``; ``usd`` null when unknown."""

    models = pricing.get("models") if isinstance(pricing.get("models"), Mapping) else {}
    prices = models.get(model) if model is not None else None
    record: dict[str, Any] = {"label": COST_LABEL, "model": model, "priced": isinstance(prices, Mapping),
                              "usd": None, "components_usd": None}
    counts = [tokens.get(field) for field in ("input", "cached_input", "output")]
    if not isinstance(prices, Mapping) or any(not isinstance(value, int) for value in counts):
        return record
    input_tokens, cached, output = counts
    components = {
        "uncached_input": _round(max(input_tokens - cached, 0) * float(prices["input"]) / 1e6),
        "cached_input": _round(cached * float(prices["cached_input"]) / 1e6),
        "output": _round(output * float(prices["output"]) / 1e6),
    }
    record.update(usd=_round(sum(components.values())), components_usd=components)
    return record


def _first_int(*values: Any) -> int | None:
    for value in values:
        number = _integer(value)
        if number is not None:
            return number
    return None


def _details(usage: Mapping[str, Any], *names: str) -> Mapping[str, Any]:
    for name in names:
        value = usage.get(name)
        if isinstance(value, Mapping):
            return value
    return {}


def normalize_usage(usage: Any) -> dict[str, int] | None:
    """Input, cached input, output and reasoning tokens of one usage block, or ``None``.

    Harness events carry the normalized names (input includes cached input,
    output includes reasoning); raw OpenAI names are accepted as a fallback.
    """

    if not isinstance(usage, Mapping):
        return None
    input_tokens = _first_int(usage.get("input_tokens"), usage.get("prompt_tokens"))
    output = _first_int(usage.get("output_tokens"), usage.get("completion_tokens"))
    if input_tokens is None or output is None:
        return None
    prompt = _details(usage, "prompt_tokens_details", "input_tokens_details")
    completion = _details(usage, "completion_tokens_details", "output_tokens_details")
    cached = _first_int(usage.get("cached_input_tokens"), usage.get("cache_read_input_tokens"),
                        prompt.get("cached_tokens"))
    reasoning = _first_int(usage.get("reasoning_output_tokens"), completion.get("reasoning_tokens"))
    return {"input": input_tokens, "cached_input": cached or 0, "output": output, "reasoning_output": reasoning or 0}


def _result_code(result: Mapping[str, Any]) -> str | None:
    """``error.code``, else the first item ``error_code`` of a batch result (``mark_point`` marks)."""

    error = result.get("error")
    if isinstance(error, Mapping) and isinstance(error.get("code"), str) and error["code"]:
        return error["code"]
    for value in result.values():
        if isinstance(value, list):
            for item in value:
                if isinstance(item, Mapping) and isinstance(item.get("error_code"), str) and item["error_code"]:
                    return item["error_code"]
    return None


def _refusal_code(run_dir: Path, artifact: Any) -> str:
    """The error code of a refused tool result, read from its artifact inside the run."""

    if not isinstance(artifact, Mapping) or not isinstance(artifact.get("path"), str):
        return "unknown"
    path = (run_dir / artifact["path"]).resolve()
    if not path.is_relative_to(run_dir.resolve()) or not path.is_file() or path.stat().st_size > _ARTIFACT_LIMIT:
        return "unknown"
    try:
        result = json.loads(path.read_bytes())
    except (OSError, ValueError):
        return "unknown"
    return (_result_code(result) if isinstance(result, Mapping) else None) or "unknown"


def _observation_bytes(payload: Mapping[str, Any]) -> int | None:
    sources = payload.get("context_sources")
    if not isinstance(sources, list):
        return None
    sizes = [_integer(item.get("bytes")) for item in sources
             if isinstance(item, Mapping) and item.get("kind") == "causal_observation"]
    sizes = [value for value in sizes if value is not None]
    return sum(sizes) if sizes else None


def _finish(arguments: Any, unix_s: Any) -> dict[str, Any]:
    arguments = arguments if isinstance(arguments, Mapping) else {}
    reason = arguments.get("reason")
    return {
        "unix_s": float(unix_s) if isinstance(unix_s, (int, float)) else None,
        "outcome": arguments.get("outcome") if isinstance(arguments.get("outcome"), str) else None,
        "category": arguments.get("category") if isinstance(arguments.get("category"), str) else None,
        "reason": reason[:_REASON_LIMIT] if isinstance(reason, str) else None,
        "ok": None,
    }


def _empty_scan() -> dict[str, Any]:
    return {
        "present": False, "model": None, "unreadable_lines": 0,
        "usage": {"responses": 0, "responses_without_usage": 0, "input": 0, "cached_input": 0, "output": 0,
                  "reasoning_output": 0},
        "request_input_tokens": [], "observation_text_bytes": [], "inline_images_per_request": [],
        "tool_calls": Counter(), "refused": {}, "finish_calls": [],
    }


def _apply(scan: dict[str, Any], event: Mapping[str, Any], run_dir: Path, finishes: dict[str, dict]) -> None:
    kind, payload = event.get("type"), event.get("payload")
    payload = payload if isinstance(payload, Mapping) else {}
    if kind == "session_started" and scan["model"] is None and isinstance(payload.get("model"), str):
        scan["model"] = payload["model"]
    elif kind == "provider_response":
        totals = scan["usage"]
        totals["responses"] += 1
        usage = normalize_usage(payload.get("usage"))
        if usage is None:
            totals["responses_without_usage"] += 1
            return
        for field, value in usage.items():
            totals[field] += value
        scan["request_input_tokens"].append(usage["input"])
    elif kind == "context_projected":
        text_bytes = _observation_bytes(payload)
        if text_bytes is not None:
            scan["observation_text_bytes"].append(text_bytes)
        images = _integer(payload.get("active_inline_images"))
        if images is not None:
            scan["inline_images_per_request"].append(images)
    elif kind == "tool_call_proposed" and isinstance(payload.get("name"), str):
        scan["tool_calls"][payload["name"]] += 1
        if payload["name"] == _FINISH_TOOL:
            finishes[str(event.get("tool_call_id"))] = _finish(payload.get("arguments"), event.get("unix_s"))
    elif kind == "tool_result" and isinstance(payload.get("name"), str):
        finish = finishes.get(str(event.get("tool_call_id"))) if payload["name"] == _FINISH_TOOL else None
        if finish is not None:
            finish["ok"] = payload.get("ok") is True
        if payload.get("ok") is False:
            codes = scan["refused"].setdefault(payload["name"], Counter())
            codes[_refusal_code(run_dir, event.get("artifact_ref"))] += 1


def scan_agent_events(run_dir: Path) -> dict[str, Any]:
    """One pass over ``agent-events.jsonl``; an absent file yields an empty scan."""

    scan = _empty_scan()
    path = run_dir / "agent-events.jsonl"
    if not path.is_file():
        return scan
    scan["present"] = True
    finishes: dict[str, dict[str, Any]] = {}
    with path.open("rb") as stream:
        for line in stream:
            if not any(marker in line for marker in _WANTED):
                continue
            try:
                event = json.loads(line)
            except ValueError:
                scan["unreadable_lines"] += 1
                continue
            if isinstance(event, Mapping):
                _apply(scan, event, run_dir, finishes)
    scan["finish_calls"] = list(finishes.values())
    return scan


def metrics_tokens(run_dir: Path) -> dict[str, int] | None:
    """Token totals of the sealed ``runtime-metrics.json`` workload, or ``None``."""

    path = run_dir / "runtime-metrics.json"
    try:
        workload = json.loads(path.read_bytes()).get("workload") if path.is_file() else None
    except (OSError, ValueError, AttributeError):
        return None
    if not isinstance(workload, Mapping):
        return None
    return normalize_usage(workload)


__all__ = [
    "COST_LABEL", "PRICE_FIELDS", "PRICING_PATH", "PRICING_SCHEMA", "PricingError", "api_equivalent_cost",
    "load_pricing", "metrics_tokens", "normalize_usage", "scan_agent_events",
]
