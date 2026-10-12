"""The layers of one lossless (Context v4/v5) projection.

``NativeContextPipeline._project_action_step_lossless_v4`` opens a
:class:`LosslessView` over the canonical history and then runs the layers in
order. Each layer moves at most one frontier on the pipeline, re-renders, and
records a projection transition when the Provider-visible request changed:

* L1/L2 :func:`externalize_aged_read_only_results` -- large and duplicate
  read-only results of ActionSteps that left the exact near window;
* L3 :func:`externalize_lifecycle_media` -- media outside the current-Episode
  media tail;
* L4 :func:`externalize_under_pressure` -- every remaining read-only result and
  older observation body, when inline images or estimated/reported tokens
  cross their thresholds;
* L5/L6 (:mod:`.lossless_semantic`) -- a closed-Episode capsule, then an
  emergency checkpoint, each admitted only at a Provider-reported watermark.

L1-L4 are lossless: every externalized payload stays recoverable. L5/L6 are
the only semantic layers. The pipeline owns the frontiers and checkpoint state;
the view owns this projection's working copies and request estimate.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ora.agent.contracts import PolicySessionError

from . import deferred_eager, tail_aging
from .lossless_render import render_lossless
from .projection import (
    IMAGE_PRESSURE_ESTIMATE_TOKENS, _action_step_inventory, _estimated_tokens,
    _exclude_reasoning, _inline_image_count, _inline_media_bytes_from_refs,
    _json_bytes, _materialize_recoverable_payloads, _message_event_ids,
    _observation_groups, _pass_record, _pressure_estimated_tokens,
    _provider_message,
)

TRANSITION_SCHEMA = "ora.context-projection-transition.v4"

Rendered = tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]


@dataclass
class LosslessView:
    """One projection's canonical inputs, frontiers and current render.

    ``lossless_working`` is the L1-L4 render (every slot still recoverable);
    ``working`` adds the L5/L6 semantic slots and is what the Provider sees.
    The ``active_*`` frontiers start at the pipeline's values and advance as
    layers admit transitions; :meth:`render` always renders the current ones.
    """

    pipeline: Any
    tools: Sequence[Mapping[str, Any]]
    allow_transition: bool
    raw: list[dict[str, Any]]
    materialized: list[dict[str, Any]]
    materialized_refs: list[Any]
    first_delivery_advertised_indexes: list[int]
    reasoning_changes: list[int]
    inventory: list[dict[str, Any]]
    groups: list[Any]
    completed_count: int
    causal_exact_tail: int
    near_tail: int
    aging: tail_aging.TailAging
    emergency_candidate: int
    active_eager_through: int
    active_pressure_through: int
    active_lifecycle_through: int
    active_superseded_events: tuple[str, ...]
    epoch_before: int
    render_memo: dict[tuple[Any, ...], Any] = field(default_factory=dict)
    transitions: list[dict[str, Any]] = field(default_factory=list)
    current_round: Any = None
    admission_input_tokens: int | None = None
    lossless_working: list[dict[str, Any]] = field(default_factory=list)
    working: list[dict[str, Any]] = field(default_factory=list)
    render_record: dict[str, Any] = field(default_factory=dict)
    provider_messages: list[dict[str, Any]] = field(default_factory=list)
    request_view: dict[str, Any] = field(default_factory=dict)
    pressure_raw: int = 0
    pressure: int = 0
    calibration_delta: int = 0
    token_estimate_status: str = ""

    def render(self) -> Rendered:
        """Render the materialized history at the current frontiers."""

        pipeline = self.pipeline
        # Renders inside one projection share unchanged L1-L3 stage results.
        return render_lossless(
            self.materialized, mutation_tools=pipeline.mutation_tools,
            spill_threshold_bytes=pipeline.spill_threshold_bytes,
            preview_bytes=pipeline.preview_bytes,
            active_eager_through=self.active_eager_through,
            pressure_through=self.active_pressure_through,
            lifecycle_media_through=self.active_lifecycle_through,
            superseded_tool_image_event_ids=self.active_superseded_events,
            episode_checkpointed_action_steps=pipeline.episode_checkpointed_action_step_count,
            episode_capsules=pipeline.episode_capsules,
            emergency_summarized_action_steps=pipeline.emergency_summarized_action_step_count,
            emergency_checkpoint=pipeline.emergency_checkpoint,
            emergency_transcript_ref=pipeline.emergency_transcript_ref,
            memo=self.render_memo,
        )

    def adopt(self, rendered: Rendered) -> None:
        """Make ``rendered`` current and re-estimate the request it implies."""

        self.lossless_working, self.working, self.render_record = rendered
        self.refresh_request()

    def refresh_request(self) -> None:
        self.provider_messages = [
            _provider_message(message) for message in self.working
        ]
        self.request_view = {
            "messages": self.provider_messages, "tools": list(self.tools),
        }
        self.pressure_raw = _pressure_estimated_tokens(self.request_view)
        (
            self.pressure, self.calibration_delta, self.token_estimate_status,
        ) = self._calibrated_pressure(self.pressure_raw)

    def provider_anchor_applies(self) -> bool:
        """Provider usage was measured on the current projection epoch."""

        pipeline = self.pipeline
        return bool(
            pipeline.last_provider_input_tokens is not None
            and pipeline.last_estimated_tokens is not None
            and pipeline.last_provider_projection_epoch == pipeline.cache_epoch
        )

    def _calibrated_pressure(self, raw_estimate: int) -> tuple[int, int, str]:
        pipeline = self.pipeline
        if self.provider_anchor_applies():
            assert pipeline.last_provider_input_tokens is not None
            assert pipeline.last_estimated_tokens is not None
            delta = pipeline.last_provider_input_tokens - pipeline.last_estimated_tokens
            return (
                max(1, raw_estimate + delta),
                delta,
                "provider_anchored_suffix_estimate",
            )
        return (
            max(1, raw_estimate),
            0,
            (
                "projection_epoch_unanchored"
                if pipeline.last_provider_input_tokens is not None
                else "provider_usage_unavailable"
            ),
        )

    def ensure_projection_epoch(self) -> None:
        """The first transition of this projection opens a new epoch."""

        if self.pipeline.cache_epoch == self.epoch_before:
            self.pipeline.cache_epoch += 1

    def size_snapshot(self) -> dict[str, int]:
        """Pre-transition sizes that a transition record reports."""

        return {
            "pressure": self.pressure,
            "images": _inline_image_count(self.request_view),
            "media_bytes": _inline_media_bytes_from_refs(self.working),
            "request_bytes": _json_bytes(self.request_view),
        }


def open_lossless_view(
    pipeline: Any, *, messages: list[dict[str, Any]],
    tools: Sequence[Mapping[str, Any]], allow_transition: bool,
) -> LosslessView:
    """Materialize recoverable payloads, check coverage and render once."""

    management = pipeline.context_management
    # Every caller passes a private deep copy and materialization copies
    # again before editing, so ``raw`` is only read here.
    raw = messages
    (
        materialized,
        materialized_refs,
        first_delivery_advertised_indexes,
    ) = _materialize_recoverable_payloads(
        raw, catalog=pipeline.context_artifacts,
        mutation_tools=pipeline.mutation_tools,
        advertise_tool_result_over_bytes=pipeline.spill_threshold_bytes,
        media_cache=pipeline.inline_media,
    )
    materialized, reasoning_changes = _exclude_reasoning(materialized)
    inventory = _action_step_inventory(
        materialized, mutation_tools=pipeline.mutation_tools,
        require_session_round=True,
    )
    groups = _observation_groups(materialized)
    if not groups:
        raise PolicySessionError("lossless ActionStep Context requires an observation")
    completed_count = len(groups) - 1
    causal_exact_tail = int(management["causal_exact_tail_action_steps"])
    # Causal/read-only bodies use one exact near window. Media has an
    # independent current-Episode tail below so visual history does not
    # enlarge the rewritten Provider suffix merely to preserve causal text.
    near_tail = max(causal_exact_tail, int(
        management["inline_media_tail_action_steps"]
    ))
    eligible_through = max(0, completed_count - near_tail)
    active_eager_through = pipeline.lifecycle_read_only_externalized_action_step_count
    if active_eager_through > eligible_through:
        raise PolicySessionError("read-only coverage enters protected near window")
    # v4 ages at every fresh observation; v5 batches the same frontiers.
    aging = tail_aging.plan_tail_aging(
        management=management, inventory=inventory,
        completed=completed_count, near_tail=near_tail,
        read_only_active=active_eager_through, allow_transition=allow_transition,
        media_active=pipeline.lifecycle_media_externalized_action_step_count,
    )
    emergency_candidate = max(0, completed_count - causal_exact_tail)
    _check_layer_coverage(pipeline, completed_count, emergency_candidate)
    view = LosslessView(
        pipeline=pipeline, tools=tools, allow_transition=allow_transition,
        raw=raw, materialized=materialized, materialized_refs=materialized_refs,
        first_delivery_advertised_indexes=first_delivery_advertised_indexes,
        reasoning_changes=reasoning_changes, inventory=inventory, groups=groups,
        completed_count=completed_count, causal_exact_tail=causal_exact_tail,
        near_tail=near_tail, aging=aging, emergency_candidate=emergency_candidate,
        active_eager_through=active_eager_through,
        active_pressure_through=pipeline.pressure_externalized_action_step_count,
        active_lifecycle_through=pipeline.lifecycle_media_externalized_action_step_count,
        active_superseded_events=pipeline.lifecycle_superseded_tool_image_event_ids,
        epoch_before=pipeline.cache_epoch,
    )
    view.adopt(view.render())
    # Capture pressure while its Provider wire/epoch binding is current.
    # L3/L4 below can rewrite that epoch on every physical action. Such a
    # rewrite invalidates size calibration for the new view, not the fact
    # that the source request already crossed a compaction watermark.
    # This admission anchor lives only for this projection; it cannot be
    # reused after a summary or in a later, unmeasured epoch.
    view.admission_input_tokens = (
        pipeline.last_provider_input_tokens if view.provider_anchor_applies() else None
    )
    return view


def _check_layer_coverage(
    pipeline: Any, completed_count: int, emergency_candidate: int,
) -> None:
    if pipeline.lifecycle_media_externalized_action_step_count > completed_count:
        raise PolicySessionError(
            "lifecycle media coverage exceeds completed ActionSteps"
        )
    if pipeline.pressure_externalized_action_step_count > completed_count:
        raise PolicySessionError(
            "pressure-GC coverage exceeds completed ActionSteps"
        )
    if pipeline.episode_checkpointed_action_step_count > completed_count:
        raise PolicySessionError("Episode capsule coverage exceeds completed history")
    if pipeline.emergency_summarized_action_step_count > emergency_candidate:
        raise PolicySessionError("emergency checkpoint coverage exceeds eligible history")


# -- L1-L4: lossless externalization -------------------------------------------


def externalize_aged_read_only_results(view: LosslessView) -> None:
    """L1/L2 run only after a step leaves the shared near window.

    Rewriting an already delivered result must announce its own cache
    transition.
    """

    if not (
        view.allow_transition
        and view.aging.read_only_through > view.active_eager_through
    ):
        return
    previous_eager_through = view.active_eager_through
    pre_eager_pressure = view.pressure
    pre_eager_bytes = _json_bytes(view.request_view)
    view.active_eager_through = view.aging.read_only_through
    view.pipeline.lifecycle_read_only_externalized_action_step_count = (
        view.active_eager_through
    )
    rendered = view.render()
    next_lossless_working = rendered[0]
    changed_indexes = [
        index for index, (before, after) in enumerate(
            zip(view.lossless_working, next_lossless_working)
        ) if _provider_message(before) != _provider_message(after)
    ]
    if not changed_indexes:
        return
    view.ensure_projection_epoch()
    view.adopt(rendered)
    view.transitions.append(deferred_eager.transition_record(
        projection_epoch=view.pipeline.cache_epoch, from_epoch=view.epoch_before,
        before=previous_eager_through, after=view.active_eager_through,
        event_ids=_message_event_ids(
            next_lossless_working, changed_indexes,
        ),
        pre_pressure=pre_eager_pressure, post_pressure=view.pressure,
        pre_bytes=pre_eager_bytes, post_bytes=_json_bytes(view.request_view),
        inline_images=_inline_image_count(view.request_view),
        media_bytes=_inline_media_bytes_from_refs(view.working),
    ))


def externalize_lifecycle_media(view: LosslessView) -> None:
    """L3 is an Episode-aware fixed media tail.

    Keep every image in the open ActionStep and the configured number of
    completed current-Episode steps; reset releases all prior-Episode media.
    This lifecycle is independent of the causal/read-only near window.
    """

    pipeline = view.pipeline
    view.current_round = view.inventory[-1]["session_round"]
    media_tail = int(
        pipeline.context_management["inline_media_tail_action_steps"]
    )
    desired_lifecycle_through = view.aging.media_through
    candidate_superseded_events: tuple[str, ...] = ()
    previous_superseded = set(view.active_superseded_events)
    next_superseded = previous_superseded | set(candidate_superseded_events)
    lifecycle_changed = (
        desired_lifecycle_through > view.active_lifecycle_through
        or next_superseded != previous_superseded
    )
    if not (view.allow_transition and lifecycle_changed):
        return
    pre = view.size_snapshot()
    previous_lifecycle = view.active_lifecycle_through
    pipeline.lifecycle_media_externalized_action_step_count = (
        desired_lifecycle_through
    )
    pipeline.lifecycle_superseded_tool_image_event_ids = tuple(
        sorted(next_superseded)
    )
    view.active_lifecycle_through = desired_lifecycle_through
    view.active_superseded_events = tuple(sorted(next_superseded))
    rendered = view.render()
    lifecycle_pass = next(
        item for item in rendered[2]["passes"]
        if item["name"] == "l3_lifecycle_media_reference"
    )
    if not lifecycle_pass["changed_messages"]:
        return
    view.ensure_projection_epoch()
    view.adopt(rendered)
    view.transitions.append({
        "schema": TRANSITION_SCHEMA,
        "kind": "lifecycle_media_externalization",
        "projection_epoch": pipeline.cache_epoch,
        "from_epoch": view.epoch_before,
        "to_epoch": pipeline.cache_epoch,
        "reasons": (
            ["recent_media_tail_advanced"]
            if desired_lifecycle_through > previous_lifecycle else []
        ) + (
            ["derived_media_superseded"]
            if next_superseded != previous_superseded else []
        ),
        "externalized_action_steps_before": previous_lifecycle,
        "externalized_action_steps_after": desired_lifecycle_through,
        "inline_media_tail_action_steps": media_tail,
        "protected_current_round": view.current_round,
        "superseded_tool_image_events_before": sorted(
            previous_superseded
        ),
        "superseded_tool_image_events_after": sorted(next_superseded),
        "semantic_loss": False,
        **_size_change(view, pre),
    })


def _size_change(view: LosslessView, pre: Mapping[str, int]) -> dict[str, int]:
    """The pre/post image, media, token and byte fields of a transition."""

    return {
        "pre_inline_images": pre["images"],
        "post_inline_images": _inline_image_count(view.request_view),
        "pre_media_bytes": pre["media_bytes"],
        "post_media_bytes": _inline_media_bytes_from_refs(view.working),
        "pre_estimated_input_tokens": pre["pressure"],
        "post_estimated_input_tokens": view.pressure,
        "pre_request_utf8_bytes": pre["request_bytes"],
        "post_request_utf8_bytes": _json_bytes(view.request_view),
    }


def _pressure_trigger_reasons(view: LosslessView, pre_images: int) -> list[str]:
    """Which L4 thresholds this request crosses; none while aging is pending."""

    pipeline = view.pipeline
    management = pipeline.context_management
    trigger_reasons: list[str] = []
    # Pending stride slack is not pressure; L4 waits for the next batch.
    admit_pressure = view.allow_transition and not view.aging.pending
    if admit_pressure and pre_images > int(
        management["trigger_inline_images"]
    ):
        trigger_reasons.append("inline_images")
    if admit_pressure and view.pressure > pipeline.estimate_pressure_trigger():
        trigger_reasons.append("estimated_tokens")
    provider_trigger = int(
        pipeline.hard_input_tokens
        * float(management["trigger_provider_input_ratio"])
    )
    if (
        admit_pressure
        and view.provider_anchor_applies()
        and pipeline.last_provider_input_tokens is not None
        and pipeline.last_provider_input_tokens >= provider_trigger
    ):
        trigger_reasons.append("provider_input_tokens")
    return trigger_reasons


def externalize_under_pressure(view: LosslessView) -> None:
    """L4 externalizes every aged read-only result and observation body."""

    pipeline = view.pipeline
    pre = view.size_snapshot()
    trigger_reasons = _pressure_trigger_reasons(view, pre["images"])
    if not (
        trigger_reasons
        and view.aging.read_only_through > view.active_pressure_through
    ):
        return
    previous = view.active_pressure_through
    pipeline.pressure_externalized_action_step_count = view.aging.read_only_through
    view.active_pressure_through = view.aging.read_only_through
    rendered = view.render()
    pressure_passes = [
        item for item in rendered[2]["passes"]
        if item["name"] in {"l4_read_only_result_reference", "l4_observation_reference"}
    ]
    if not any(item["changed_messages"] for item in pressure_passes):
        return
    view.ensure_projection_epoch()
    view.adopt(rendered)
    view.transitions.append({
        "schema": TRANSITION_SCHEMA,
        "kind": "pressure_externalization",
        "projection_epoch": pipeline.cache_epoch,
        "from_epoch": view.epoch_before,
        "to_epoch": pipeline.cache_epoch,
        "reasons": trigger_reasons,
        "externalized_action_steps_before": previous,
        "externalized_action_steps_after": view.active_pressure_through,
        "semantic_loss": False,
        **_size_change(view, pre),
    })


# -- Manifest sections without request hashing ---------------------------------


def projection_passes(view: LosslessView) -> list[dict[str, Any]]:
    """Materialization, the render's L1-L4 passes, then reasoning exclusion."""

    return [
        _pass_record(
            "recoverable_payload_materialization", view.raw, view.materialized,
            artifact_refs=view.materialized_refs,
            changed_messages=view.first_delivery_advertised_indexes,
        ),
        *view.render_record["passes"],
        _pass_record(
            "reasoning_exclusion", view.raw, view.materialized,
            changed_messages=view.reasoning_changes,
            replaced_event_ids=_message_event_ids(
                view.materialized, view.reasoning_changes,
            ),
        ),
    ]


def token_fields(view: LosslessView) -> dict[str, Any]:
    """Request estimates and the Provider usage anchor they are calibrated to."""

    pipeline = view.pipeline
    return {
        "estimated_input_tokens": view.pressure,
        "raw_estimated_input_tokens": _estimated_tokens(view.request_view),
        "pressure_estimated_input_tokens": view.pressure_raw,
        "provider_usage_anchor_tokens": pipeline.last_provider_input_tokens,
        "provider_usage_anchor_projection_epoch": (
            pipeline.last_provider_projection_epoch
        ),
        "provider_usage_anchor_wire_request_sha256": (
            pipeline.last_provider_wire_request_sha256
        ),
        "provider_usage_anchor_applies_to_projection_epoch": (
            view.provider_anchor_applies()
        ),
        "token_estimate_status": view.token_estimate_status,
        "estimate_calibration_delta_tokens": view.calibration_delta,
    }


def size_fields(view: LosslessView) -> dict[str, Any]:
    """Active request size and how much L1-L4 externalized."""

    request_view = view.request_view
    pre_lossless_bytes = _json_bytes({
        "messages": [_provider_message(message) for message in view.materialized],
        "tools": list(view.tools),
    })
    post_lossless_bytes = _json_bytes({
        "messages": [
            _provider_message(message) for message in view.lossless_working
        ],
        "tools": list(view.tools),
    })
    return {
        "request_utf8_bytes": _json_bytes(request_view),
        "active_inline_images": _inline_image_count(request_view),
        "active_media_bytes": _inline_media_bytes_from_refs(view.working),
        "registered_recoverable_artifact_count": len(view.materialized_refs),
        "first_delivery_advertised_message_count": len(
            view.first_delivery_advertised_indexes
        ),
        "pre_lossless_request_utf8_bytes": pre_lossless_bytes,
        "post_lossless_request_utf8_bytes": post_lossless_bytes,
        "actual_lossless_externalized_ratio": (
            max(0, pre_lossless_bytes - post_lossless_bytes)
            / pre_lossless_bytes
        ),
        "estimated_image_tokens": (
            _inline_image_count(request_view)
            * IMAGE_PRESSURE_ESTIMATE_TOKENS
        ),
        "estimated_text_tokens": max(
            0, view.pressure_raw - _inline_image_count(request_view)
            * IMAGE_PRESSURE_ESTIMATE_TOKENS,
        ),
    }


def _through_index(count: int) -> int | None:
    """The last covered ActionStep index of a leading-count frontier."""

    return count - 1 if count else None


def coverage_fields(view: LosslessView) -> dict[str, Any]:
    """Every frontier, semantic layer count and the ActionStep inventory."""

    pipeline = view.pipeline
    return {
        "lossless_externalized_through_action_step": _through_index(
            pipeline.lifecycle_media_externalized_action_step_count
        ),
        "lifecycle_media_externalized_through_action_step": _through_index(
            pipeline.lifecycle_media_externalized_action_step_count
        ),
        "pressure_externalized_through_action_step": _through_index(
            pipeline.pressure_externalized_action_step_count
        ),
        "episode_checkpointed_through_action_step": _through_index(
            pipeline.episode_checkpointed_action_step_count
        ),
        "episode_checkpointed_rounds": list(
            pipeline.episode_checkpointed_rounds
        ),
        "episode_checkpoint_count": pipeline.episode_checkpoint_count,
        "emergency_summarized_through_action_step": _through_index(
            pipeline.emergency_summarized_action_step_count
        ),
        "emergency_compaction_count": pipeline.emergency_compaction_count,
        "semantic_rejection_count": pipeline.semantic_rejection_count,
        "last_semantic_rejection": copy.deepcopy(
            pipeline.last_semantic_rejection
        ),
        "retained_exact_causal_action_steps": min(
            view.causal_exact_tail,
            view.completed_count - max(
                pipeline.episode_checkpointed_action_step_count,
                pipeline.emergency_summarized_action_step_count,
            ),
        ),
        "protected_inline_media_action_steps": [
            item["action_step_index"] for item in view.inventory
            if (
                item.get("complete") is not True
                or item.get("action_step_index", -1)
                >= pipeline.lifecycle_media_externalized_action_step_count
            )
        ],
        "open_action_step": view.inventory[-1]["action_step_index"],
        "action_step_inventory": copy.deepcopy(view.inventory),
    }
