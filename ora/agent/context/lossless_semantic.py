"""The semantic layers of a lossless Context projection: L5 and L6.

Both summarize policy-visible history through the pipeline's compactor, and
both are admitted only by Provider-reported input tokens measured on the
projection's own epoch (``LosslessView.admission_input_tokens``):

* L5 :func:`admit_episode_capsule` replaces one closed Episode -- found by
  :func:`closed_episode_candidate` -- with a capsule once the input crosses the
  Episode watermark. Same-source retries are bounded and spaced.
* L6 :func:`admit_emergency_checkpoint` writes one cumulative emergency
  checkpoint over every completed step outside the exact causal tail, when no
  L5 candidate has priority and the input crosses the emergency watermark.

A rejected summary leaves the projection unchanged and is recorded on the
pipeline; the exact transcript of every summarized span stays recoverable.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .lossless_layers import TRANSITION_SCHEMA, LosslessView
from .projection import _inline_image_count, _inline_media_bytes_from_refs, _json_bytes

EPISODE_RETRY_ATTEMPTS = 4


# -- L5: closed-Episode capsule ------------------------------------------------


@dataclass(frozen=True)
class EpisodeAdmission:
    """L5's decision: its manifest record, watermark and priority over L6."""

    record: dict[str, Any]
    threshold: int
    has_priority: bool


def closed_episode_candidate(view: LosslessView) -> tuple[int, int, int] | None:
    """The next closed Episode L5 may summarize: ``(start, end, round)``.

    L5 owns closed Episode boundaries. An earlier L6 may have summarized only
    the prefix of the still-open Episode, so its frontier must not become L5's
    candidate start. The full Episode remains recoverable from
    ``lossless_working`` and can supersede that overlapping L6 slot once the
    Episode closes and clears the exact causal tail.
    """

    inventory = view.inventory
    completed_count = view.completed_count
    episode_start = view.pipeline.episode_checkpointed_action_step_count
    if episode_start >= completed_count:
        return None
    candidate_round = inventory[episode_start].get("session_round")
    starts_at_round_boundary = (
        episode_start == 0
        or inventory[episode_start - 1].get("session_round")
        != candidate_round
    )
    if not (
        isinstance(candidate_round, int)
        and candidate_round != view.current_round
        and starts_at_round_boundary
    ):
        return None
    candidate_end = episode_start
    while (
        candidate_end < completed_count
        and inventory[candidate_end].get("session_round") == candidate_round
    ):
        candidate_end += 1
    if (
        candidate_end > episode_start
        and candidate_end <= view.emergency_candidate
        and all(
            item.get("complete") is True
            for item in inventory[episode_start:candidate_end]
        )
    ):
        return (episode_start, candidate_end, candidate_round)
    return None


def admit_episode_capsule(
    view: LosslessView,
    candidate: tuple[int, int, int] | None,
    retry_key: str | None,
    *,
    summarize: Callable[..., dict[str, Any]],
) -> EpisodeAdmission:
    """Summarize the candidate Episode when the Provider crossed its watermark.

    Same-source retries are bounded (four attempts) and spaced by completed
    ActionSteps; ``retry_key`` identifies the candidate's exact source.
    """

    pipeline = view.pipeline
    post_lossless_pressure = view.pressure
    semantic_anchor_tokens = view.admission_input_tokens
    episode_threshold = pipeline.episode_checkpoint_trigger()
    episode_retry = {"attempts": 0, "next_completed_action_step": 0}
    episode_retry_reason = None
    if candidate is not None:
        episode_retry = pipeline.episode_retry_state.get(retry_key, episode_retry)
        if episode_retry["attempts"] >= EPISODE_RETRY_ATTEMPTS:
            episode_retry_reason = "same_episode_retry_exhausted"
        elif view.completed_count < episode_retry["next_completed_action_step"]:
            episode_retry_reason = "same_episode_retry_backoff"
    episode_admission = _episode_admission_record(
        view, candidate=candidate, retry_key=retry_key,
        retry=episode_retry, retry_reason=episode_retry_reason,
        anchor_tokens=semantic_anchor_tokens, threshold=episode_threshold,
    )
    if (
        view.allow_transition
        and semantic_anchor_tokens is not None
        and semantic_anchor_tokens >= episode_threshold
        and candidate is not None
        and episode_retry_reason is None
    ):
        assert retry_key is not None
        _summarize_episode(
            view, candidate, retry_key, episode_retry, episode_admission,
            summarize=summarize, post_lossless_pressure=post_lossless_pressure,
            anchor_tokens=semantic_anchor_tokens, threshold=episode_threshold,
        )
    # L6 is considered only after no closed Episode remains eligible for L5
    # in this projection epoch. A rejected/deferred L5 must not starve
    # emergency admission.
    has_priority = (candidate is not None
                    and episode_retry_reason is None
                    and episode_admission["status"] != "rejected")
    return EpisodeAdmission(
        record=episode_admission, threshold=episode_threshold,
        has_priority=has_priority,
    )


def _episode_admission_record(
    view: LosslessView, *, candidate: tuple[int, int, int] | None,
    retry_key: str | None, retry: Mapping[str, int], retry_reason: str | None,
    anchor_tokens: int | None, threshold: int,
) -> dict[str, Any]:
    pipeline = view.pipeline
    return {
        "authority": "provider_reported_input_tokens",
        "basis": "projection_entry_provider_usage",
        "status": "not_admitted",
        "provider_input_tokens": anchor_tokens,
        "provider_projection_epoch": pipeline.last_provider_projection_epoch,
        "provider_wire_request_sha256": (
            pipeline.last_provider_wire_request_sha256
        ),
        "episode_high_watermark_tokens": threshold,
        "retry_policy": "same_source_four_attempts_action_backoff_v1",
        "retry_source_sha256": retry_key,
        "attempts": retry["attempts"],
        "max_attempts": EPISODE_RETRY_ATTEMPTS,
        "next_retry_completed_action_step": retry["next_completed_action_step"],
        "eligible_session_round": (
            candidate[2] if candidate is not None else None
        ),
        "reason": (
            retry_reason if retry_reason is not None
            else "at_or_above_high_watermark"
            if anchor_tokens is not None
            and anchor_tokens >= threshold
            and candidate is not None
            else "no_closed_episode"
            if candidate is None
            else "below_high_watermark"
            if anchor_tokens is not None
            else "projection_epoch_unanchored"
            if pipeline.last_provider_input_tokens is not None
            else "provider_usage_unavailable"
        ),
    }


def _summarize_episode(
    view: LosslessView, candidate: tuple[int, int, int], retry_key: str,
    previous_retry: Mapping[str, int], episode_admission: dict[str, Any], *,
    summarize: Callable[..., dict[str, Any]], post_lossless_pressure: int,
    anchor_tokens: int, threshold: int,
) -> None:
    pipeline = view.pipeline
    episode_retry = {
        "attempts": previous_retry["attempts"] + 1,
        "next_completed_action_step": view.completed_count,
    }
    pipeline.episode_retry_state[retry_key] = episode_retry
    episode_admission["attempts"] = episode_retry["attempts"]
    episode_admission["status"] = "admitted"
    pre = view.size_snapshot()
    start_action_step, end_action_step, candidate_round = candidate
    try:
        semantic = summarize(
            messages=view.lossless_working,
            tools=view.tools,
            inventory=view.inventory,
            start_action_step=start_action_step,
            end_action_step=end_action_step,
        )
    except Exception as exc:
        # The stored retry record is this same dict, so the backoff persists.
        episode_retry["next_completed_action_step"] = (
            view.completed_count + 4 * 2 ** (episode_retry["attempts"] - 1)
        )
        episode_admission["next_retry_completed_action_step"] = episode_retry["next_completed_action_step"]
        pipeline.semantic_rejection_count += 1
        pipeline.last_semantic_rejection = {
            "layer": "episode_capsule_l5",
            "session_round": candidate_round,
            "projection_epoch": pipeline.cache_epoch,
            "provider_wire_request_sha256": (
                pipeline.last_provider_wire_request_sha256
            ),
            "error": str(exc),
            "disposition": "original_episode_retained",
        }
        episode_admission.update({
            "status": "rejected",
            "reason": "summary_rejected_original_retained",
            "error": str(exc),
        })
    else:
        _commit_episode_capsule(
            view, semantic, candidate, pre=pre,
            post_lossless_pressure=post_lossless_pressure,
            anchor_tokens=anchor_tokens, threshold=threshold,
        )


def _superseded_emergency_checkpoint(
    pipeline: Any, end_action_step: int,
) -> dict[str, Any] | None:
    """The active L6 checkpoint a new capsule fully covers, if any."""

    if not (
        pipeline.emergency_summarized_action_step_count
        > pipeline.episode_checkpointed_action_step_count
        and pipeline.emergency_checkpoint is not None
        and pipeline.emergency_summarized_action_step_count > 0
        and end_action_step
        >= pipeline.emergency_summarized_action_step_count
    ):
        return None
    return {
        "checkpoint_sha256": pipeline.emergency_checkpoint_sha256,
        "checkpoint_ref": copy.deepcopy(
            pipeline.emergency_checkpoint_ref
        ),
        "exact_transcript": copy.deepcopy(
            pipeline.emergency_transcript_ref
        ),
        "through_action_step_index": (
            pipeline.emergency_summarized_action_step_count - 1
        ),
    }


def _commit_episode_capsule(
    view: LosslessView, semantic: Mapping[str, Any],
    candidate: tuple[int, int, int], *, pre: Mapping[str, int],
    post_lossless_pressure: int, anchor_tokens: int, threshold: int,
) -> None:
    pipeline = view.pipeline
    start_action_step, end_action_step, candidate_round = candidate
    previous_episode_count = pipeline.episode_checkpoint_count
    superseded_emergency = _superseded_emergency_checkpoint(pipeline, end_action_step)
    pipeline.episode_checkpointed_action_step_count = end_action_step
    pipeline.episode_checkpointed_rounds = tuple([
        *pipeline.episode_checkpointed_rounds, candidate_round,
    ])
    pipeline.episode_checkpoint_count += 1
    pipeline.episode_capsules = tuple([
        *pipeline.episode_capsules, copy.deepcopy(semantic),
    ])
    pipeline.absorbed_action_step_count = max(
        pipeline.absorbed_action_step_count, end_action_step,
    )
    if superseded_emergency is not None:
        pipeline.emergency_checkpoint = None
        pipeline.emergency_checkpoint_sha256 = None
        pipeline.emergency_checkpoint_ref = None
        pipeline.emergency_transcript_ref = None
    pipeline.last_semantic_rejection = None
    if not view.transitions:
        pipeline.cache_epoch += 1
    view.adopt(view.render())
    view.transitions.append({
        "schema": TRANSITION_SCHEMA,
        "kind": "episode_capsule",
        "projection_epoch": pipeline.cache_epoch,
        "from_epoch": view.epoch_before,
        "to_epoch": pipeline.cache_epoch,
        "reasons": [
            "closed_episode_provider_high_watermark",
            *(
                ["superseded_overlapping_emergency_checkpoint"]
                if superseded_emergency is not None else []
            ),
        ],
        "session_round": candidate_round,
        "episode_checkpoint_count_before": previous_episode_count,
        "episode_checkpoint_count_after": pipeline.episode_checkpoint_count,
        "checkpointed_action_steps_before": start_action_step,
        "checkpointed_action_steps_after": end_action_step,
        "semantic_loss": True,
        "post_lossless_estimated_input_tokens": post_lossless_pressure,
        "semantic_admission_provider_input_tokens": anchor_tokens,
        "semantic_admission_provider_projection_epoch": (
            pipeline.last_provider_projection_epoch
        ),
        "semantic_admission_provider_wire_request_sha256": (
            pipeline.last_provider_wire_request_sha256
        ),
        "semantic_high_watermark_tokens": threshold,
        "pre_inline_images": pre["images"],
        "post_inline_images": _inline_image_count(view.request_view),
        "pre_media_bytes": pre["media_bytes"],
        "post_media_bytes": _inline_media_bytes_from_refs(view.working),
        "pre_request_utf8_bytes": pre["request_bytes"],
        "post_request_utf8_bytes": _json_bytes(view.request_view),
        "packet_ref": copy.deepcopy(semantic["packet_ref"]),
        "capsule_ref": copy.deepcopy(semantic["capsule_ref"]),
        "capsule_sha256": semantic["capsule_sha256"],
        "exact_transcript": copy.deepcopy(
            semantic["exact_transcript"]
        ),
        "superseded_emergency_checkpoint": superseded_emergency,
        "checkpoint_estimated_tokens": semantic[
            "checkpoint_estimated_tokens"
        ],
        "checkpoint_budget": copy.deepcopy(
            semantic["checkpoint_budget"]
        ),
        "compactor": copy.deepcopy(semantic["metadata"]),
        "post_estimated_input_tokens": view.pressure,
    })


# -- L6: emergency checkpoint --------------------------------------------------


def admit_emergency_checkpoint(
    view: LosslessView, episode: EpisodeAdmission, *,
    summarize: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    """Admit the L6 checkpoint at the emergency watermark; return its record."""

    pipeline = view.pipeline
    emergency_threshold = int(
        pipeline.hard_input_tokens
        * float(pipeline.context_management["emergency_fallback_ratio"])
    )
    emergency_anchor_tokens = (
        view.admission_input_tokens
        if episode.record["status"] != "admitted" else None
    )
    emergency_admission: dict[str, Any] = {
        "authority": "provider_reported_input_tokens",
        "basis": "projection_entry_provider_usage",
        "status": "not_admitted",
        "provider_input_tokens": emergency_anchor_tokens,
        "provider_projection_epoch": pipeline.last_provider_projection_epoch,
        "provider_wire_request_sha256": pipeline.last_provider_wire_request_sha256,
        "emergency_high_watermark_tokens": emergency_threshold,
        "reason": (
            "l5_has_priority" if episode.has_priority
            else "at_or_above_high_watermark"
            if emergency_anchor_tokens is not None
            and emergency_anchor_tokens >= emergency_threshold
            else "below_high_watermark"
            if emergency_anchor_tokens is not None
            else "projection_epoch_unanchored"
            if pipeline.last_provider_input_tokens is not None
            else "provider_usage_unavailable"
        ),
    }
    if (
        view.allow_transition
        and not episode.has_priority
        and emergency_anchor_tokens is not None
        and emergency_anchor_tokens >= emergency_threshold
        and view.emergency_candidate > max(
            pipeline.emergency_summarized_action_step_count,
            pipeline.episode_checkpointed_action_step_count,
        )
    ):
        emergency_admission["status"] = "admitted"
        _summarize_emergency(
            view, emergency_admission, summarize=summarize,
            anchor_tokens=emergency_anchor_tokens, threshold=emergency_threshold,
        )
    return emergency_admission


def _summarize_emergency(
    view: LosslessView, emergency_admission: dict[str, Any], *,
    summarize: Callable[..., dict[str, Any]], anchor_tokens: int, threshold: int,
) -> None:
    pipeline = view.pipeline
    before_emergency = max(
        pipeline.emergency_summarized_action_step_count,
        pipeline.episode_checkpointed_action_step_count,
    )
    pre_emergency_bytes = _json_bytes(view.request_view)
    try:
        semantic = summarize(
            messages=view.lossless_working,
            tools=view.tools,
            inventory=view.inventory,
            candidate_absorbed=view.emergency_candidate,
        )
    except Exception as exc:
        pipeline.semantic_rejection_count += 1
        pipeline.last_semantic_rejection = {
            "layer": "emergency_session_l6",
            "projection_epoch": pipeline.cache_epoch,
            "provider_wire_request_sha256": (
                pipeline.last_provider_wire_request_sha256
            ),
            "error": str(exc),
            "disposition": "current_projection_retained",
        }
        emergency_admission.update({
            "status": "rejected",
            "reason": "summary_rejected_projection_retained",
            "error": str(exc),
        })
    else:
        _commit_emergency_checkpoint(
            view, semantic, before_emergency=before_emergency,
            pre_emergency_bytes=pre_emergency_bytes,
            anchor_tokens=anchor_tokens, threshold=threshold,
        )


def _commit_emergency_checkpoint(
    view: LosslessView, semantic: Mapping[str, Any], *, before_emergency: int,
    pre_emergency_bytes: int, anchor_tokens: int, threshold: int,
) -> None:
    pipeline = view.pipeline
    emergency_candidate = view.emergency_candidate
    pipeline.emergency_summarized_action_step_count = emergency_candidate
    pipeline.emergency_compaction_count += 1
    pipeline.emergency_checkpoint = copy.deepcopy(semantic["checkpoint"])
    pipeline.emergency_checkpoint_sha256 = str(
        semantic["checkpoint_sha256"]
    )
    pipeline.emergency_checkpoint_ref = copy.deepcopy(
        semantic["checkpoint_ref"]
    )
    pipeline.emergency_transcript_ref = copy.deepcopy(
        semantic["exact_transcript"]
    )
    pipeline.absorbed_action_step_count = max(
        pipeline.absorbed_action_step_count, emergency_candidate,
    )
    pipeline.last_semantic_rejection = None
    if not view.transitions:
        pipeline.cache_epoch += 1
    view.adopt(view.render())
    target = int(
        pipeline.hard_input_tokens
        * float(pipeline.context_management["emergency_target_ratio"])
    )
    pipeline.pending_semantic_target_epoch = pipeline.cache_epoch
    pipeline.pending_semantic_target_tokens = target
    pipeline.last_semantic_target_validation = {
        "status": "provider_pending",
        "projection_epoch": pipeline.cache_epoch,
        "target_tokens": target,
        "enforcement": "telemetry_only_until_hard_limit",
    }
    view.transitions.append({
        "schema": TRANSITION_SCHEMA,
        "kind": "emergency_compaction",
        "projection_epoch": pipeline.cache_epoch,
        "from_epoch": view.epoch_before,
        "to_epoch": pipeline.cache_epoch,
        "reasons": ["provider_reported_emergency_high_watermark"],
        "emergency_summarized_action_steps_before": before_emergency,
        "emergency_summarized_action_steps_after": emergency_candidate,
        "semantic_loss": True,
        "semantic_admission_provider_input_tokens": anchor_tokens,
        "semantic_admission_provider_projection_epoch": (
            pipeline.last_provider_projection_epoch
        ),
        "semantic_admission_provider_wire_request_sha256": (
            pipeline.last_provider_wire_request_sha256
        ),
        "semantic_high_watermark_tokens": threshold,
        "pre_request_utf8_bytes": pre_emergency_bytes,
        "post_request_utf8_bytes": _json_bytes(view.request_view),
        "packet_ref": copy.deepcopy(semantic["packet_ref"]),
        "checkpoint_ref": copy.deepcopy(semantic["checkpoint_ref"]),
        "exact_transcript": copy.deepcopy(
            semantic["exact_transcript"]
        ),
        "checkpoint_sha256": semantic["checkpoint_sha256"],
        "checkpoint_estimated_tokens": semantic[
            "checkpoint_estimated_tokens"
        ],
        "checkpoint_budget": copy.deepcopy(
            semantic["checkpoint_budget"]
        ),
        "compactor": copy.deepcopy(semantic["metadata"]),
        "post_estimated_input_tokens": view.pressure,
        "target_input_tokens": target,
        "target_validation": "provider_pending",
    })
