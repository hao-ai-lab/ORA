"""Stateless L1-L4 rendering and semantic-slot composition for Context v4."""
from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

from ora.agent.memory.artifacts import ARTIFACT_READ_NAMES

from . import historical_observation, round_static_facts
from .projection import (
    _eligible_message_indexes, _externalize_historical_media,
    _externalize_tool_results, _message_event_ids,
    _tiered_semantic_projection,
)
from .request_size import active_sizer, measured_pass_record


def render_lossless(
    materialized: list[dict[str, Any]], *, mutation_tools: Sequence[str],
    spill_threshold_bytes: int, preview_bytes: int, active_eager_through: int,
    pressure_through: int, lifecycle_media_through: int,
    superseded_tool_image_event_ids: Sequence[str],
    episode_checkpointed_action_steps: int,
    episode_capsules: Sequence[Mapping[str, Any]],
    emergency_summarized_action_steps: int,
    emergency_checkpoint: Mapping[str, Any] | None,
    emergency_transcript_ref: Mapping[str, Any] | None,
    memo: dict[tuple[Any, ...], Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Render L1-L4 plus semantic slots over one materialized history.

    Every pass is a pure function of its input state and its own frontier
    (each deep-copies its input and registers nothing), and ``materialized`` is
    never edited. ``memo`` is scoped to one projection over one
    ``materialized`` list: a later render whose L1/L2 (and L3) frontiers are
    unchanged resumes from the stored post-L2 (post-L3) state instead of
    recomputing identical passes. Stored states are only ever pass inputs, so
    no caller receives an object that the memo keeps; pass records and refs are
    returned as copies.
    """

    measure = active_sizer().measure
    stages = memo if memo is not None else {}

    base_key = ("base",)
    if base_key not in stages:
        stages[base_key] = measure(materialized)
    base_size = stages[base_key]

    eager_key = ("l1_l2", active_eager_through)
    if eager_key not in stages:
        working, large_indexes, large_refs = _externalize_tool_results(
            materialized,
            through_action_steps=active_eager_through,
            mutation_tools=mutation_tools,
            selector="large",
            threshold_bytes=spill_threshold_bytes,
            preview_bytes=preview_bytes,
            exclude_tools=tuple(ARTIFACT_READ_NAMES),
        )
        after = measure(working)
        large_record = measured_pass_record(
            "l1_large_result_reference", base_size, after,
            changed_messages=large_indexes, artifact_refs=large_refs,
            externalized_event_ids=_message_event_ids(working, large_indexes),
        )
        working, duplicate_indexes, duplicate_refs = _externalize_tool_results(
            working,
            through_action_steps=active_eager_through,
            mutation_tools=mutation_tools,
            selector="duplicate",
            threshold_bytes=spill_threshold_bytes,
            preview_bytes=0,
            exclude_tools=tuple(ARTIFACT_READ_NAMES),
        )
        before, after = after, measure(working)
        duplicate_record = measured_pass_record(
            "l2_exact_duplicate_reference", before, after,
            changed_messages=duplicate_indexes, artifact_refs=duplicate_refs,
            externalized_event_ids=_message_event_ids(
                working, duplicate_indexes,
            ),
        )
        stages[eager_key] = (
            working, after, [large_record, duplicate_record],
            [*large_refs, *duplicate_refs],
        )
    working, after, eager_records, eager_refs = stages[eager_key]

    media_key = (
        "l3", active_eager_through, lifecycle_media_through,
        tuple(superseded_tool_image_event_ids),
    )
    if media_key not in stages:
        media_working, media_indexes, media_refs = _externalize_historical_media(
            working,
            through_action_steps=lifecycle_media_through,
            tool_image_event_ids=superseded_tool_image_event_ids,
        )
        # native_slots_v4: an earlier round's static-facts slot ages with its
        # round's media; the current round's slot never does.
        media_working, slot_indexes, slot_refs = (
            round_static_facts.externalize_prior_rounds(
                media_working, through_action_steps=lifecycle_media_through,
            )
        )
        if slot_indexes:
            media_indexes = sorted({*media_indexes, *slot_indexes})
            media_refs = [*media_refs, *slot_refs]
        media_after = measure(media_working)
        media_record = measured_pass_record(
            "l3_lifecycle_media_reference", after, media_after,
            changed_messages=media_indexes, artifact_refs=media_refs,
            externalized_event_ids=_message_event_ids(media_working, media_indexes),
        )
        stages[media_key] = (media_working, media_after, media_record, media_refs)
    working, after, media_record, media_refs = stages[media_key]

    passes: list[dict[str, Any]] = copy.deepcopy([*eager_records, media_record])
    artifact_refs: list[dict[str, Any]] = copy.deepcopy([*eager_refs, *media_refs])
    working, remaining_indexes, remaining_refs = _externalize_tool_results(
        working,
        through_action_steps=pressure_through,
        mutation_tools=mutation_tools,
        selector="remaining",
        threshold_bytes=spill_threshold_bytes,
        preview_bytes=preview_bytes,
    )
    before, after = after, measure(working)
    passes.append(measured_pass_record(
        "l4_read_only_result_reference", before, after,
        changed_messages=remaining_indexes, artifact_refs=remaining_refs,
        externalized_event_ids=_message_event_ids(
            working, remaining_indexes,
        ),
    ))
    working, observation_indexes, observation_refs = historical_observation.externalize(
        working, eligible_indexes=_eligible_message_indexes(
            working, through_action_steps=pressure_through,
        ),
    )
    before, after = after, measure(working)
    passes.append(measured_pass_record(
        "l4_observation_reference", before, after,
        changed_messages=observation_indexes, artifact_refs=observation_refs,
        externalized_event_ids=_message_event_ids(working, observation_indexes),
    ))
    projected = _tiered_semantic_projection(
        working,
        episode_checkpointed_action_steps=(
            episode_checkpointed_action_steps
        ),
        episode_capsules=episode_capsules,
        emergency_summarized_action_steps=(
            emergency_summarized_action_steps
        ),
        emergency_checkpoint=emergency_checkpoint,
        emergency_transcript_ref=emergency_transcript_ref,
    )
    return working, projected, {
        "passes": passes,
        "projection_checkpoint": copy.deepcopy(
            emergency_checkpoint
        ),
        "artifact_refs": [
            *artifact_refs, *remaining_refs, *observation_refs,
        ],
    }
