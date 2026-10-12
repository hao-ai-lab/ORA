"""Bounded, content-free telemetry for the Native admission probe stages."""
from __future__ import annotations

import json
from typing import Any, Mapping


class ProviderProbeDiagnostics:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}

    def begin(self, stage: str) -> None:
        self.records[stage] = {"response_observed": False, "semantic_passed": None}

    def observe(self, stage: str, result: Any, credential_pool: list[str]) -> Mapping[str, Any]:
        probe = self.records[stage]
        serialized = json.dumps(result, ensure_ascii=False, default=str)
        if any(key in serialized for key in credential_pool):
            probe["failure_category"] = "credential_echo"
            raise RuntimeError("provider qualification echoed credential material")
        if not isinstance(result, Mapping):
            probe["failure_category"] = "malformed_response"
            raise RuntimeError("provider qualification response is not a mapping")
        message = result.get("message")
        message = message if isinstance(message, Mapping) else {}
        finish = result.get("finish_reason")
        probe.update(
            response_observed=True,
            finish_reason=finish if finish in ("stop", "length", "tool_calls", "content_filter", None) else "other",
            content_observed=bool(message.get("content")),
            reasoning_observed=bool(message.get("reasoning_content")),
            tool_call_count=len(message["tool_calls"]) if isinstance(message.get("tool_calls"), list) else 0,
        )
        usage = result.get("usage")
        probe["usage"] = {key: usage[key] for key in (
            "input_tokens", "output_tokens", "reasoning_output_tokens", "cached_input_tokens",
            "cache_read_input_tokens", "cache_creation_input_tokens",
        ) if type(usage.get(key)) is int and usage[key] >= 0} if isinstance(usage, Mapping) else {}
        if finish == "length":
            probe["failure_category"] = "output_cap"
        return result

    def semantic(self, stage: str, passed: bool) -> None:
        self.records[stage]["semantic_passed"] = bool(passed)
        if not passed:
            self.records[stage].setdefault("failure_category", "semantic_mismatch")

    def failure(self, stage: str, http_status: Any) -> None:
        self.records.setdefault(stage, {"response_observed": False, "semantic_passed": None}).setdefault(
            "failure_category", "http_error" if type(http_status) is int else "provider_exception",
        )
