"""One-call multimodal compactor for rollout-local Native context."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from ora.agent.contracts import (
    ModelProvider,
    PolicySessionError,
    SummaryResult,
)
from ora.agent.evidence.trajectory import ArtifactStore, canonical_bytes


COMPACTOR_SYSTEM_PROMPT = """You condense completed robot ActionSteps from one rollout.
Return exactly one JSON object with these seven keys:
task_progress, confirmed_facts, invalidated_beliefs, failed_attempts,
current_constraints, do_not_repeat, re_ground_next.

All fields are arrays of concise strings except failed_attempts, whose entries are
objects with exactly failure, evidence, and correction string fields.

Reconstruct intention/action -> tool receipt -> later visual evidence. Compare the
agent's belief with later images. Preserve empty grasps, wrong-object grasps,
slippage, false success beliefs, corrections, constraints, and unresolved state.
Use confirmed/inferred/unknown language when confidence matters. Do not narrate
frames one by one. Do not claim the official task verdict. Do not emit run IDs,
hashes, paths, historical pixels, world coordinates, or replayable target poses.
Treat repeated point, mask, and view queries within one search episode as one
consolidated attempt: preserve the accepted result, invalidated belief, and
remaining uncertainty, not a query-by-query inventory.
Express spatial state only qualitatively, for example "above the basket" or
"the descent stopped short". Never copy numeric positions, orientations,
distances, dimensions, apertures, pixel locations, axis assignments, or metric
units from the evidence. In particular, strings such as x=, y=, z=, xyz_m,
position_m, delta_mm, point_uv, quaternions, radians, millimeters, centimeters,
and meters are forbidden in the JSON. The retained exact observation and tools,
not this checkpoint, own all geometry. The next decision must re-ground against
the retained current observation.
Prefer at most two entries per array, each a short sentence. Use empty arrays
when there is nothing useful to preserve; do not fill every field by repeating
the same fact. Finish every array and the enclosing JSON object.
Return JSON only, without markdown."""


def compactor_system_prompt(*, checkpoint_target_estimated_tokens: int) -> str:
    """Bind the model's concise-output target below the hard admission limit."""

    if checkpoint_target_estimated_tokens <= 0:
        raise PolicySessionError("context checkpoint token budget must be positive")
    # Native context admission conservatively estimates dense JSON at one token
    # per three UTF-8 bytes. Give the model ten percent headroom so normal
    # wording variance does not turn the hard validator into the length control.
    target_bytes = int(checkpoint_target_estimated_tokens * 3 * 0.9)
    return COMPACTOR_SYSTEM_PROMPT + f"""

The complete minified JSON MUST be at most {target_bytes:,} UTF-8 bytes. This is
a strict concise-output target below the {checkpoint_target_estimated_tokens:,}-token
Harness target. When a previous checkpoint is supplied, rewrite and
consolidate it; do not append another history. Merge duplicates, delete stale or
resolved detail, and prioritize facts that can change the next decisions."""


def _assistant_text(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(block.get("text", ""))
            for block in content
            if isinstance(block, Mapping) and block.get("type") == "text"
        )
    raise PolicySessionError("context compactor response has no text content")


@dataclass
class RolloutCheckpointCompactor:
    """Execute and trace one fresh, tool-free provider call per transition."""

    provider: ModelProvider
    model: str
    reasoning_effort: str
    stream: bool
    artifacts: ArtifactStore
    event_sink: Callable[..., Mapping[str, Any]]
    checkpoint_target_estimated_tokens: int = 1200
    # Kimi's signed thinking and the JSON checkpoint share the generation
    # budget. Keep the visible checkpoint capped separately in the context
    # validator; this ceiling must leave room for both channels to reach stop.
    max_output_tokens: int = 8192
    call_sequence: int = 0
    calls: list[dict[str, Any]] = field(default_factory=list)

    def summarize(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        source_refs: Sequence[Mapping[str, Any]],
    ) -> SummaryResult:
        call_id = f"context-compactor-{self.call_sequence:06d}"
        self.call_sequence += 1
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": compactor_system_prompt(
                        checkpoint_target_estimated_tokens=(
                            self.checkpoint_target_estimated_tokens
                        ),
                    ),
                },
                *copy.deepcopy([dict(message) for message in messages]),
            ],
            "max_tokens": self.max_output_tokens,
            "stream": self.stream,
        }
        if self.reasoning_effort != "provider_default":
            request["reasoning_effort"] = self.reasoning_effort
        options_for = getattr(self.provider, "checkpoint_request_options", None)
        options = options_for(model=self.model) if callable(options_for) else {}
        if not isinstance(options, Mapping) or set(options) - {"reasoning_effort", "response_format"}:
            raise PolicySessionError("unsupported checkpoint request options")
        request.update(copy.deepcopy(dict(options)))
        if self.stream:
            request["stream_options"] = {"include_usage": True}
        canonical_ref = self.artifacts.put_bytes(
            canonical_bytes(request),
            kind="rollout_compactor_canonical_request",
            suffix=".json",
            media_type="application/json",
        )
        prepare = getattr(self.provider, "prepare_request", None)
        wire = prepare(request) if callable(prepare) else copy.deepcopy(request)
        if not isinstance(wire, Mapping):
            raise PolicySessionError("context compactor wire request is invalid")
        wire_ref = self.artifacts.put_bytes(
            canonical_bytes(wire),
            kind="rollout_compactor_wire_request",
            suffix=".json",
            media_type="application/json",
        )
        claim = self.event_sink(
            "context_compactor_attempt",
            visibility="derived",
            model_call_id=call_id,
            artifact_ref=wire_ref,
            payload={
                "call_kind": "context_compactor",
                "model": self.model,
                "provider_protocol": str(
                    getattr(self.provider, "protocol", "canonical_unknown")
                ),
                "canonical_request_ref": canonical_ref,
                "source_set_sha256": hashlib.sha256(
                    canonical_bytes(list(source_refs))
                ).hexdigest(),
            },
        )
        started = time.time()
        response_event_id: str | None = None
        try:
            response = self.provider.complete(request=request, model_call_id=call_id)
            returned_wire = response.get("wire_request")
            if returned_wire is not None and canonical_bytes(returned_wire) != canonical_bytes(wire):
                raise PolicySessionError(
                    "context compactor sent a wire request different from evidence"
                )
            attempts = response.get("attempts", 1)
            raw_ref = self.artifacts.put_json(
                response.get("raw_events", []),
                kind="rollout_compactor_wire_response",
            )
            finished = response.get("finished_unix_s")
            response_started = response.get("started_unix_s")
            latency = (
                float(finished) - float(response_started)
                if isinstance(finished, (int, float))
                and isinstance(response_started, (int, float))
                else time.time() - started
            )
            event = self.event_sink(
                "context_compactor_response",
                causal_parent_id=str(claim["event_id"]),
                visibility="derived",
                model_call_id=call_id,
                artifact_ref=raw_ref,
                payload={
                    "call_kind": "context_compactor",
                    "usage": response.get("usage"),
                    "finish_reason": response.get("finish_reason"),
                    "attempts": attempts,
                    "latency_s": round(latency, 6),
                    "response_id": response.get("response_id"),
                    "validation_pending": True,
                },
            )
            response_event_id = str(event["event_id"])
            if attempts != 1:
                raise PolicySessionError(
                    "context compactor provider performed more than one physical attempt"
                )
            if response.get("finish_reason") != "stop":
                raise PolicySessionError(
                    "context compactor response did not complete with stop: "
                    f"{response.get('finish_reason')!r}"
                )
            message = response.get("message")
            if not isinstance(message, Mapping) or message.get("role") != "assistant":
                raise PolicySessionError(
                    "context compactor response lacks an assistant message"
                )
            content = _assistant_text(message).strip()
            if not content:
                raise PolicySessionError("context compactor returned empty content")
            record = {
                "call_id": call_id,
                "attempt_event_id": claim["event_id"],
                "response_event_id": event["event_id"],
                "canonical_request_ref": canonical_ref,
                "wire_request_ref": wire_ref,
                "raw_response_ref": raw_ref,
                "usage": copy.deepcopy(response.get("usage")),
                "latency_s": round(latency, 6),
                "response_id": response.get("response_id"),
            }
            self.calls.append(record)
            return SummaryResult(
                content=content,
                source_refs=copy.deepcopy(list(source_refs)),
                source_set_sha256=hashlib.sha256(
                    canonical_bytes(list(source_refs))
                ).hexdigest(),
                metadata=record,
            )
        except Exception as exc:
            self.event_sink(
                "context_compactor_failed",
                causal_parent_id=response_event_id or str(claim["event_id"]),
                visibility="derived",
                model_call_id=call_id,
                payload={
                    "call_kind": "context_compactor",
                    "error": str(exc),
                },
            )
            if isinstance(exc, PolicySessionError):
                raise
            raise PolicySessionError(f"context_compactor_failed: {exc}") from exc


__all__ = ["COMPACTOR_SYSTEM_PROMPT", "RolloutCheckpointCompactor"]
