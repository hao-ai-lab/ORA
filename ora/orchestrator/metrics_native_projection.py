"""Context-projection aggregation for native API runs.

`_native_projection_metrics` turns the `context_projection` recorded with each
native call, together with the agent-event trace, into the
`context.projection` report: the projection summary, request relations and
cache epochs, the lossless v4 watermarks, transitions and per-action-step
sizes, and the context-compactor calls. `metrics._native_request_metrics`
measures the request and response files and calls it.
"""

from __future__ import annotations

from typing import Any, Mapping

from ora.orchestrator.metrics_values import _integer, summarize, summarize_signed
from ora.orchestrator.visual_usage import visual_usage


_LOSSLESS_PROJECTION_SCHEMA = "ora.native-context-projection.v6"


def _native_projection_metrics(
    calls: list[dict[str, Any]], agent_events: list[dict[str, Any]],
    request_bytes: list[int],
) -> dict[str, Any]:
    """Return the `context.projection` report of one native run."""
    projections = _context_projections(calls)
    projection_metrics = _projection_summary(projections, agent_events, request_bytes)
    relations = [
        projection.get("request_relation") for projection in projections
        if isinstance(projection.get("request_relation"), str)
    ]
    if relations:
        projection_metrics.update(_request_relation_metrics(projections, relations))
    lossless_projections = [
        projection for projection in projections
        if projection.get("schema")
        == _LOSSLESS_PROJECTION_SCHEMA
    ]
    if lossless_projections:
        projection_metrics["lossless_context_v4"] = _lossless_context_v4_metrics(
            calls, agent_events, lossless_projections,
        )
    compactor_responses = [
        event.get("payload") for event in agent_events
        if event.get("type") == "context_compactor_response"
        and isinstance(event.get("payload"), Mapping)
    ]
    if compactor_responses:
        projection_metrics["context_compactor"] = _context_compactor_metrics(
            compactor_responses,
        )
    return projection_metrics


def _context_projections(calls: list[dict[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        call.get("context_projection") for call in calls
        if isinstance(call, Mapping) and isinstance(call.get("context_projection"), Mapping)
    ]


def _pass_byte_reductions(projections: list[Mapping[str, Any]]) -> dict[str, list[int]]:
    pass_reductions: dict[str, list[int]] = {}
    for projection in projections:
        for item in projection.get("passes", []):
            if not isinstance(item, Mapping) or not isinstance(item.get("name"), str):
                continue
            before, after = _integer(item.get("pre_bytes")), _integer(item.get("post_bytes"))
            if before is not None and after is not None:
                pass_reductions.setdefault(item["name"], []).append(before - after)
    return pass_reductions


def _projection_summary(
    projections: list[Mapping[str, Any]], agent_events: list[dict[str, Any]],
    request_bytes: list[int],
) -> dict[str, Any]:
    pass_reductions = _pass_byte_reductions(projections)
    return {
        "visual_usage": visual_usage(agent_events),
        "resolved_episode_checkpoint_trigger_tokens": summarize(
            projection.get("resolved_episode_checkpoint_trigger_tokens") for projection in projections
        ),
        "profile": projections[-1].get("profile") if projections else None,
        "estimated_input_tokens": summarize(
            projection.get("estimated_input_tokens") for projection in projections
        ),
        "request_utf8_bytes": summarize(request_bytes),
        "canonical_request_utf8_bytes": summarize(
            projection.get("canonical_request_utf8_bytes")
            for projection in projections
        ),
        "provider_wire_request_utf8_bytes": summarize(
            projection.get("provider_wire_request_utf8_bytes")
            for projection in projections
        ),
        "pass_byte_reduction": {
            name: summarize_signed(values) for name, values in sorted(pass_reductions.items())
        },
        "stable_prefix_hashes": sorted({
            projection.get("stable_prefix_hash") for projection in projections
            if isinstance(projection.get("stable_prefix_hash"), str)
        }),
        "active_inline_images": summarize(
            projection.get("active_inline_images") for projection in projections
        ),
    }


def _request_relation_metrics(
    projections: list[Mapping[str, Any]], relations: list[str],
) -> dict[str, Any]:
    relation_counts = {
        relation: relations.count(relation) for relation in sorted(set(relations))
    }
    transitions = [
        projection.get("epoch_transition") for projection in projections
        if isinstance(projection.get("epoch_transition"), Mapping)
    ]
    return {
        "context_retention": projections[-1].get("context_retention"),
        "protected_prefix_hashes": sorted({
            projection.get("protected_prefix_sha256")
            for projection in projections
            if isinstance(projection.get("protected_prefix_sha256"), str)
        }),
        "request_relations": relation_counts,
        "cache_epochs": sorted({
            int(projection["cache_epoch"]) for projection in projections
            if _integer(projection.get("cache_epoch")) is not None
        }),
        "epoch_transition_count": len(transitions),
        "epoch_pre_estimated_input_tokens": summarize(
            item.get("pre_estimated_input_tokens") for item in transitions
        ),
        "epoch_post_estimated_input_tokens": summarize(
            item.get("post_estimated_input_tokens") for item in transitions
        ),
        "epoch_pre_request_utf8_bytes": summarize(
            item.get("pre_request_utf8_bytes") for item in transitions
        ),
        "epoch_post_request_utf8_bytes": summarize(
            item.get("post_request_utf8_bytes") for item in transitions
        ),
    }


def _lossless_context_v4_metrics(
    calls: list[dict[str, Any]], agent_events: list[dict[str, Any]],
    lossless_projections: list[Mapping[str, Any]],
) -> dict[str, Any]:
    lossless_transitions = [
        transition
        for projection in lossless_projections
        for transition in projection.get("projection_transitions", [])
        if isinstance(transition, Mapping)
    ]
    first_by_step = _first_lossless_projection_by_step(calls)
    retrieval_events = [
        event for event in agent_events
        if event.get("type") == "context_artifact_retrieval"
        and isinstance(event.get("payload"), Mapping)
    ]
    retrievals = [event["payload"] for event in retrieval_events]
    return {
        **_lossless_progress(lossless_projections),
        **_lossless_transition_counts(lossless_transitions),
        "token_estimate_statuses": _token_estimate_statuses(lossless_projections),
        "artifact_retrievals": {
            "count": len(retrievals),
            "accepted": sum(item.get("status") == "ok" for item in retrievals),
            "rejected": sum(
                item.get("status") == "rejected" for item in retrievals
            ),
        },
        "action_steps": [
            _action_step_metric(
                step, projection, provider_reported_input, retrieval_events,
            )
            for step, (projection, provider_reported_input)
            in sorted(first_by_step.items())
        ],
        "transitions": lossless_transitions,
    }


def _first_lossless_projection_by_step(
    calls: list[dict[str, Any]],
) -> dict[int, tuple[Mapping[str, Any], int | None]]:
    """Map each observation step to its first lossless projection and input usage."""
    first_by_step: dict[int, tuple[Mapping[str, Any], int | None]] = {}
    for call in calls:
        projection = call.get("context_projection")
        if (
            not isinstance(projection, Mapping)
            or projection.get("schema")
            != _LOSSLESS_PROJECTION_SCHEMA
        ):
            continue
        step = _integer(projection.get("observation_step"))
        if step is not None:
            usage = call.get("usage")
            provider_reported_input = (
                _integer(usage.get("input_tokens"))
                if isinstance(usage, Mapping) else None
            )
            first_by_step.setdefault(
                step, (projection, provider_reported_input),
            )
    return first_by_step


def _latest_action_step(
    lossless_projections: list[Mapping[str, Any]], field: str,
) -> int | None:
    return max((
        int(projection[field])
        for projection in lossless_projections
        if _integer(projection.get(field)) is not None
    ), default=None)


def _lossless_progress(lossless_projections: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Projection epochs, the five externalization watermarks and the two compaction counts."""
    return {
        "projection_epochs": sorted({
            int(projection["projection_epoch"])
            for projection in lossless_projections
            if _integer(projection.get("projection_epoch")) is not None
        }),
        "lossless_externalized_through_action_step": _latest_action_step(
            lossless_projections, "lossless_externalized_through_action_step",
        ),
        "lifecycle_media_externalized_through_action_step": _latest_action_step(
            lossless_projections, "lifecycle_media_externalized_through_action_step",
        ),
        "pressure_externalized_through_action_step": _latest_action_step(
            lossless_projections, "pressure_externalized_through_action_step",
        ),
        "episode_checkpointed_through_action_step": _latest_action_step(
            lossless_projections, "episode_checkpointed_through_action_step",
        ),
        "emergency_summarized_through_action_step": _latest_action_step(
            lossless_projections, "emergency_summarized_through_action_step",
        ),
        "episode_checkpoint_count": max(
            int(projection.get("episode_checkpoint_count", 0))
            for projection in lossless_projections
        ),
        "emergency_compaction_count": max(
            int(projection.get("emergency_compaction_count", 0))
            for projection in lossless_projections
        ),
    }


def _lossless_transition_counts(
    lossless_transitions: list[Mapping[str, Any]],
) -> dict[str, int]:
    return {
        "transition_count": len(lossless_transitions),
        "lossless_transition_count": sum(
            transition.get("kind") in {
                "lifecycle_read_only_externalization",
                "lifecycle_media_externalization",
                "pressure_externalization",
            }
            for transition in lossless_transitions
        ),
        "lifecycle_transition_count": sum(
            transition.get("kind") in {
                "lifecycle_media_externalization",
                "lifecycle_read_only_externalization",
            }
            for transition in lossless_transitions
        ),
        "pressure_transition_count": sum(
            transition.get("kind") == "pressure_externalization"
            for transition in lossless_transitions
        ),
        "episode_transition_count": sum(
            transition.get("kind") == "episode_capsule"
            for transition in lossless_transitions
        ),
        "episode_superseded_emergency_count": sum(
            transition.get("kind") == "episode_capsule"
            and isinstance(
                transition.get("superseded_emergency_checkpoint"), Mapping,
            )
            for transition in lossless_transitions
        ),
        "emergency_transition_count": sum(
            transition.get("kind") == "emergency_compaction"
            for transition in lossless_transitions
        ),
    }


def _token_estimate_statuses(
    lossless_projections: list[Mapping[str, Any]],
) -> dict[str, int]:
    return {
        status: sum(
            projection.get("token_estimate_status") == status
            for projection in lossless_projections
        )
        for status in sorted({
            str(projection["token_estimate_status"])
            for projection in lossless_projections
            if isinstance(projection.get("token_estimate_status"), str)
        })
    }


def _action_step_metric(
    step: int,
    projection: Mapping[str, Any],
    provider_reported_input_tokens: int | None,
    retrieval_events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Pre/post-transition sizes, usage anchor and retrievals of one action step."""
    transition = next((
        item for item in projection.get("projection_transitions", [])
        if isinstance(item, Mapping)
        and item.get("kind") in {
            "lifecycle_read_only_externalization",
            "lifecycle_media_externalization",
            "pressure_externalization",
        }
    ), {})
    post_images = projection.get("active_inline_images")
    post_media = projection.get("active_media_bytes")
    post_tokens = projection.get("estimated_input_tokens")
    post_request_bytes = projection.get("request_utf8_bytes")
    pre_images = transition.get("pre_inline_images", post_images)
    pre_media = transition.get("pre_media_bytes", post_media)
    pre_tokens = transition.get(
        "pre_estimated_input_tokens", post_tokens,
    )
    pre_request_bytes = transition.get(
        "pre_request_utf8_bytes", post_request_bytes,
    )
    pre_image_tokens = (
        int(pre_images) * 4096
        if _integer(pre_images) is not None else None
    )
    pre_text_tokens = (
        max(0, int(pre_tokens) - int(pre_image_tokens))
        if _integer(pre_tokens) is not None
        and pre_image_tokens is not None else None
    )
    step_retrievals = [
        event["payload"] for event in retrieval_events
        if _integer(event.get("observation_step")) == step
    ]
    return {
        "action_step": step,
        "pre_inline_images": pre_images,
        "post_inline_images": post_images,
        "pre_media_bytes": pre_media,
        "post_media_bytes": post_media,
        "pre_estimated_image_tokens": pre_image_tokens,
        "post_estimated_image_tokens": projection.get(
            "estimated_image_tokens"
        ),
        "pre_estimated_text_tokens": pre_text_tokens,
        "post_estimated_text_tokens": projection.get(
            "estimated_text_tokens"
        ),
        "pre_estimated_input_tokens": pre_tokens,
        "post_estimated_input_tokens": post_tokens,
        "provider_reported_input_tokens": (
            provider_reported_input_tokens
        ),
        "provider_usage_anchor_tokens": projection.get(
            "provider_usage_anchor_tokens"
        ),
        "provider_usage_anchor_projection_epoch": projection.get(
            "provider_usage_anchor_projection_epoch"
        ),
        "provider_usage_anchor_wire_request_sha256": projection.get(
            "provider_usage_anchor_wire_request_sha256"
        ),
        "provider_usage_anchor_applies_to_projection_epoch": (
            projection.get(
                "provider_usage_anchor_applies_to_projection_epoch"
            )
        ),
        "token_estimate_status": projection.get(
            "token_estimate_status"
        ),
        "episode_checkpoint_admission": projection.get(
            "episode_checkpoint_admission"
        ),
        "emergency_compaction_admission": projection.get(
            "emergency_compaction_admission"
        ),
        "pre_request_utf8_bytes": pre_request_bytes,
        "post_request_utf8_bytes": post_request_bytes,
        "canonical_request_utf8_bytes": projection.get(
            "canonical_request_utf8_bytes"
        ),
        "provider_wire_request_utf8_bytes": projection.get(
            "provider_wire_request_utf8_bytes"
        ),
        "actual_lossless_externalized_ratio": projection.get(
            "actual_lossless_externalized_ratio"
        ),
        "artifact_retrieval_count": len(step_retrievals),
        "artifact_retrieval_accepted": sum(
            item.get("status") == "ok" for item in step_retrievals
        ),
        "artifact_retrieval_rejected": sum(
            item.get("status") == "rejected" for item in step_retrievals
        ),
        "cache_epoch": projection.get("cache_epoch"),
        "projection_epoch": projection.get("projection_epoch"),
        "episode_checkpoint_count": projection.get(
            "episode_checkpoint_count"
        ),
        "emergency_compaction_count": projection.get(
            "emergency_compaction_count"
        ),
        "lossless_externalized_through_action_step": projection.get(
            "lossless_externalized_through_action_step"
        ),
        "lifecycle_media_externalized_through_action_step": (
            projection.get(
                "lifecycle_media_externalized_through_action_step"
            )
        ),
        "pressure_externalized_through_action_step": projection.get(
            "pressure_externalized_through_action_step"
        ),
        "episode_checkpointed_through_action_step": projection.get(
            "episode_checkpointed_through_action_step"
        ),
        "emergency_summarized_through_action_step": projection.get(
            "emergency_summarized_through_action_step"
        ),
    }


def _context_compactor_metrics(
    compactor_responses: list[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "call_count": len(compactor_responses),
        "latency_s": summarize(
            item.get("latency_s") for item in compactor_responses
        ),
        "input_tokens": summarize(
            item.get("usage", {}).get("input_tokens")
            for item in compactor_responses
            if isinstance(item.get("usage"), Mapping)
        ),
        "cached_input_tokens": summarize(
            item.get("usage", {}).get("cached_input_tokens")
            for item in compactor_responses
            if isinstance(item.get("usage"), Mapping)
        ),
        "output_tokens": summarize(
            item.get("usage", {}).get("output_tokens")
            for item in compactor_responses
            if isinstance(item.get("usage"), Mapping)
        ),
    }
