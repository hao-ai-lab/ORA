"""Stateful Native-provider Context lifecycle: lossless v4 and rewrite projection."""

from __future__ import annotations

import copy
import functools
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, NamedTuple, Sequence

from ora.agent.contracts import PolicySessionError, SummaryProvider
from ora.agent.context.management import (
    ContextManagementConfigError, validate_context_management,
)
from ora.agent.evidence.trajectory import ArtifactStore, canonical_bytes
from ora.agent.memory.artifacts import (
    ContextArtifactCatalog, LOSSLESS_CONTEXT_RETENTION, recoverable_artifact_open,
)

from .projection import (
    CHECKPOINT_FIELDS,
    CONTEXT_PROFILES, CONTEXT_RETENTIONS, EMERGENCY_CHECKPOINT_SCHEMA,
    EPISODE_CAPSULE_SCHEMA, IMAGE_PRESSURE_ESTIMATE_TOKENS,
    ROLLOUT_PACKET_SCHEMA, SUMMARY_POLICIES,
    ContextCompactionRejected, _action_step_inventory,
    _causal_projection_violations,
    _compact_old_causal_observations, _compactor_history_message,
    _compactor_private_violations, _estimated_tokens,
    _exclude_reasoning, _externalize_tool_results,
    _inline_media_bytes_from_refs, _json_bytes,
    _lossless_invariant_view, _materialize_recoverable_payloads,
    _message_event_ids, _observation_groups, _packet_message_blocks,
    _packet_source_event_ids, _pass_record,
    _provider_message, _semantic_capsule_slot,
    _spill_large_tool_results,
    _strip_historical_images, _validate_checkpoint,
)


from . import deferred_eager, tail_aging, thresholds  # noqa: F401
from . import lossless_layers, lossless_semantic
from .inline_media import InlineMediaCache
from .lossless_render import render_lossless  # noqa: F401  re-exported
from .request_size import CanonicalSizer, with_projection_sizer
from .thresholds import ESTIMATE_PRESSURE_WINDOW_SHARE  # re-exported


def _with_batched_artifact_index(method: Any) -> Any:
    """Rewrite the Session Artifact index once per projection, not per ref."""

    @functools.wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        batch = getattr(self.context_artifacts, "batched_persistence", None)
        if not callable(batch):
            return method(self, *args, **kwargs)
        with batch():
            return method(self, *args, **kwargs)

    return wrapper


class _LosslessInvariants(NamedTuple):
    """What the lossless verification proved, for the manifest."""

    canonical_causal_tail: Any
    before_sha256: str
    after_sha256: str


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _episode_retry_key(
    candidate: tuple[int, int, int] | None, inventory: Sequence[Mapping[str, Any]],
) -> str | None:
    """Identity of one L5 candidate's exact source, which bounds its retries."""

    if candidate is None:
        return None
    start, end, _candidate_round = candidate
    return _canonical_sha256({
        "coverage": list(candidate),
        "source_events": [item["source_event_ids"] for item in inventory[start:end]],
    })


_STATIC_PROMPT_SOURCES = {"harness_system", "project_instructions", "episode_task"}


def _request_identity_parts(
    *, provider_messages: list[dict[str, Any]], working: list[dict[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    is_static_protected: Callable[[Mapping[str, Any]], bool],
) -> dict[str, Any]:
    """Prefix hashes, protected sources and history of one projected request."""

    protected_indexes = [
        index for index, message in enumerate(working)
        if message.get("_protected_context") is True
    ]
    stable_prefix: list[dict[str, Any]] = []
    for message in provider_messages:
        if message.get("role") != "system":
            break
        stable_prefix.append(message)
    protected_messages = [
        provider_messages[index] for index in protected_indexes
        if is_static_protected(working[index])
    ]
    return {
        "stable_prefix_hash": _canonical_sha256({
            "messages": stable_prefix, "tools": list(tools),
        }),
        "protected_prefix_sha256": _canonical_sha256({
            "messages": protected_messages, "tools": list(tools),
        }),
        "protected_prefix_message_count": len(protected_messages),
        "source_messages": {
            str(message.get("_context_source")): _provider_message(message)
            for message in working
            if isinstance(message.get("_context_source"), str)
            and message.get("_rollout_checkpoint") is not True
        },
        "history_messages": [
            provider_messages[index] for index in range(len(provider_messages))
            if index not in protected_indexes
        ],
    }


def _source_sha256(source_messages: Mapping[str, Any], name: str) -> str | None:
    return (
        _canonical_sha256(source_messages[name])
        if name in source_messages else None
    )


def _lossless_request_identity(view: lossless_layers.LosslessView) -> dict[str, Any]:
    """Request identity fields of a lossless manifest, in manifest order."""

    parts = _request_identity_parts(
        provider_messages=view.provider_messages, working=view.working,
        tools=view.tools,
        is_static_protected=lambda message: (
            message.get("_context_source") in _STATIC_PROMPT_SOURCES
        ),
    )
    source_messages = parts["source_messages"]
    history_messages = parts["history_messages"]
    return {
        "request_sha256": _canonical_sha256(view.request_view),
        "stable_prefix_hash": parts["stable_prefix_hash"],
        "protected_prefix_sha256": parts["protected_prefix_sha256"],
        "protected_prefix_message_count": parts["protected_prefix_message_count"],
        "tool_schema_sha256": _canonical_sha256(list(view.tools)),
        "history_sha256": _canonical_sha256(history_messages),
        "history_utf8_bytes": len(canonical_bytes(history_messages)),
        "stable_system_sha256": _canonical_sha256(
            source_messages.get("harness_system", [])
        ),
        "project_context_sha256": _source_sha256(
            source_messages, "project_instructions",
        ),
        "episode_task_sha256": _source_sha256(source_messages, "episode_task"),
    }


@dataclass
class NativeContextPipeline:
    profile: str
    artifacts: ArtifactStore
    context_window_tokens: int
    max_output_tokens: int
    summary_policy: str = "disabled"
    summary_provider: SummaryProvider | None = None
    spill_threshold_bytes: int = 16 * 1024
    preview_bytes: int = 4 * 1024
    last_estimated_tokens: int | None = None
    last_provider_input_tokens: int | None = None
    last_provider_projection_epoch: int | None = None
    last_provider_wire_request_sha256: str | None = None
    pending_semantic_target_epoch: int | None = None
    pending_semantic_target_tokens: int | None = None
    last_semantic_target_validation: Mapping[str, Any] | None = None
    retention: str = "rewrite_compact_v1"
    context_management: Mapping[str, Any] | None = None
    context_artifacts: ContextArtifactCatalog | None = None
    mutation_tools: Sequence[str] = (
        "move_pose", "close_gripper", "open_gripper",
    )
    cache_epoch: int = 0
    absorbed_action_step_count: int = 0
    lifecycle_media_externalized_action_step_count: int = 0
    lifecycle_read_only_externalized_action_step_count: int = 0
    lifecycle_superseded_tool_image_event_ids: tuple[str, ...] = ()
    pressure_externalized_action_step_count: int = 0
    episode_checkpointed_action_step_count: int = 0
    episode_checkpointed_rounds: tuple[int, ...] = ()
    episode_checkpoint_count: int = 0
    episode_capsules: tuple[Mapping[str, Any], ...] = ()
    emergency_summarized_action_step_count: int = 0
    emergency_compaction_count: int = 0
    emergency_checkpoint: Mapping[str, Any] | None = None
    emergency_checkpoint_sha256: str | None = None
    emergency_checkpoint_ref: Mapping[str, Any] | None = None
    emergency_transcript_ref: Mapping[str, Any] | None = None
    last_semantic_rejection: Mapping[str, Any] | None = None
    semantic_rejection_count: int = 0
    # Same-source L5 retries are bounded and spaced by completed ActionSteps.
    # This state survives projection/cache changes within the PolicySession.
    episode_retry_state: dict[str, dict[str, int]] = field(default_factory=dict)
    # Exact size memo and verified payload admissions; never evidence state.
    canonical_sizer: CanonicalSizer = field(
        default_factory=CanonicalSizer, init=False, repr=False, compare=False,
    )
    inline_media: InlineMediaCache = field(
        default_factory=InlineMediaCache, init=False, repr=False, compare=False,
    )

    def __post_init__(self) -> None:
        if self.profile not in CONTEXT_PROFILES:
            raise PolicySessionError(f"unsupported native context profile: {self.profile}")
        if self.retention not in CONTEXT_RETENTIONS:
            raise PolicySessionError(f"unsupported native context retention: {self.retention}")
        if self.summary_policy not in SUMMARY_POLICIES:
            raise PolicySessionError(f"unsupported summary policy: {self.summary_policy}")
        if self.retention == LOSSLESS_CONTEXT_RETENTION:
            if self.summary_policy != "rollout_checkpoint_v1":
                raise PolicySessionError(
                    "action_step_lossless_v4 requires summary_policy=rollout_checkpoint_v1"
                )
            try:
                raw = validate_context_management(self.context_management)
            except ContextManagementConfigError as exc:
                raise PolicySessionError(str(exc)) from exc
            if self.summary_provider is None:
                raise PolicySessionError("lossless Context fallback lacks a compactor")
            if self.context_artifacts is None:
                raise PolicySessionError("lossless Context lacks an Artifact catalog")
            self.context_management = raw
            self.spill_threshold_bytes = int(raw["eager_large_result_bytes"])
            self.preview_bytes = int(raw["eager_preview_bytes"])
            if hasattr(self.summary_provider, "checkpoint_target_estimated_tokens"):
                setattr(
                    self.summary_provider,
                    "checkpoint_target_estimated_tokens",
                    int(raw["checkpoint_target_estimated_tokens"]),
                )
        elif self.context_management is not None:
            raise PolicySessionError(
                "context_management is supported only by action_step_lossless_v4"
            )
        if self.context_window_tokens <= self.max_output_tokens + 2048:
            raise PolicySessionError("native context window leaves no safe input budget")

    @property
    def target_input_tokens(self) -> int:
        available = self.context_window_tokens - self.max_output_tokens
        return int(available * 0.75)

    @property
    def hard_input_tokens(self) -> int:
        return self.context_window_tokens - self.max_output_tokens - 2048

    def project(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        observation_step: int,
    ) -> dict[str, Any]:
        working = [copy.deepcopy(dict(message)) for message in messages]
        if self.retention == LOSSLESS_CONTEXT_RETENTION:
            return self._project_action_step_lossless_v4(
                messages=working,
                tools=tools,
                observation_step=observation_step,
            )
        passes: list[dict[str, Any]] = []

        before = copy.deepcopy(working)
        working, image_changes = _strip_historical_images(
            working,
            current_step=observation_step,
            replacement_text=(
                "[historical image omitted from the active causal context]"
                if self.profile == "native_causal_context_v1" else None
            ),
        )
        passes.append(_pass_record(
            "historical_image_index", before, working,
            changed_messages=image_changes,
            replaced_event_ids=_message_event_ids(working, image_changes),
        ))

        if self.profile == "native_causal_context_v1":
            before = copy.deepcopy(working)
            working, compacted_observations = _compact_old_causal_observations(
                working, current_step=observation_step,
            )
            passes.append(_pass_record(
                "causal_observation_allowlist", before, working,
                changed_messages=compacted_observations,
                replaced_event_ids=_message_event_ids(
                    working, compacted_observations,
                ),
            ))

        before = copy.deepcopy(working)
        working, spill_refs = _spill_large_tool_results(
            working,
            artifacts=self.artifacts,
            threshold_bytes=self.spill_threshold_bytes,
            preview_bytes=self.preview_bytes,
        )
        spill_indexes = [
            index for index, message in enumerate(working)
            if isinstance(message.get("_artifact_ref"), Mapping)
            and message["_artifact_ref"] in spill_refs
        ]
        passes.append(_pass_record(
            "l1_spill", before, working, artifact_refs=spill_refs,
            externalized_event_ids=_message_event_ids(working, spill_indexes),
        ))

        before = copy.deepcopy(working)
        working, reasoning_changes = _exclude_reasoning(working)
        passes.append(_pass_record(
            "reasoning_exclusion", before, working,
            changed_messages=reasoning_changes,
            replaced_event_ids=_message_event_ids(working, reasoning_changes),
        ))
        provider_messages = [_provider_message(message) for message in working]
        if self.profile == "native_causal_context_v1":
            violations = _causal_projection_violations([
                message for message in provider_messages
                if message.get("role") != "assistant"
            ])
            if violations:
                raise PolicySessionError(
                    "causal_context_forbidden_fields: " + ", ".join(violations[:12])
                )
        request_view = {"messages": provider_messages, "tools": list(tools)}
        raw_estimated = _estimated_tokens(request_view)
        calibration_delta = (
            self.last_provider_input_tokens - self.last_estimated_tokens
            if self.last_provider_input_tokens is not None
            and self.last_estimated_tokens is not None
            else 0
        )
        estimated = max(1, raw_estimated + calibration_delta)
        if estimated > self.hard_input_tokens:
            raise PolicySessionError(
                f"context_exhausted: estimated {estimated} tokens, hard limit {self.hard_input_tokens}"
            )
        stable_prefix = []
        for message in provider_messages:
            if message.get("role") != "system":
                break
            stable_prefix.append(message)
        stable_prefix_hash = hashlib.sha256(canonical_bytes({
            "messages": stable_prefix,
            "tools": list(tools),
        })).hexdigest()
        source_messages = {
            str(message.get("_context_source")): _provider_message(message)
            for message in working
            if isinstance(message.get("_context_source"), str)
        }
        protected_indexes = [
            index for index, message in enumerate(working)
            if message.get("_protected_context") is True
        ]
        history_messages = [
            provider_messages[index] for index in range(len(provider_messages))
            if index not in protected_indexes
        ]
        request_hash = hashlib.sha256(canonical_bytes(request_view)).hexdigest()
        return {
            "messages": provider_messages,
            "tools": list(tools),
            "manifest": {
                "schema": "ora.native-context-projection.v1",
                "profile": self.profile,
                "summary_policy": self.summary_policy,
                "observation_step": observation_step,
                "target_input_tokens": self.target_input_tokens,
                "hard_input_tokens": self.hard_input_tokens,
                "estimated_input_tokens": estimated,
                "raw_estimated_input_tokens": raw_estimated,
                "provider_usage_anchor_tokens": self.last_provider_input_tokens,
                "estimate_calibration_delta_tokens": calibration_delta,
                "request_utf8_bytes": _json_bytes(request_view),
                "stable_prefix_hash": stable_prefix_hash,
                "stable_system_sha256": (
                    hashlib.sha256(canonical_bytes(source_messages["harness_system"])).hexdigest()
                    if "harness_system" in source_messages else stable_prefix_hash
                ),
                "project_context_sha256": (
                    hashlib.sha256(canonical_bytes(source_messages["project_instructions"])).hexdigest()
                    if "project_instructions" in source_messages else None
                ),
                "episode_task_sha256": (
                    hashlib.sha256(canonical_bytes(source_messages["episode_task"])).hexdigest()
                    if "episode_task" in source_messages else None
                ),
                "tool_schema_sha256": hashlib.sha256(canonical_bytes(list(tools))).hexdigest(),
                "history_sha256": hashlib.sha256(canonical_bytes(history_messages)).hexdigest(),
                "history_utf8_bytes": len(canonical_bytes(history_messages)),
                "request_sha256": request_hash,
                "protected_context_sources": sorted(source_messages),
                "causal_envelope": (
                    "ora.causal-observation.v1"
                    if self.profile == "native_causal_context_v1" else None
                ),
                "causal_forbidden_field_scan": (
                    "passed" if self.profile == "native_causal_context_v1" else None
                ),
                "cache_epoch": 0,
                "passes": passes,
            },
        }

    def _lossless_episode_capsule(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        inventory: Sequence[Mapping[str, Any]],
        start_action_step: int,
        end_action_step: int,
    ) -> dict[str, Any]:
        """Summarize exactly one closed Episode from its L1-L4 projection."""

        assert self.summary_provider is not None
        assert self.context_artifacts is not None
        groups = _observation_groups(messages)
        if not 0 <= start_action_step < end_action_step < len(groups):
            raise ContextCompactionRejected("episode capsule coverage is invalid")
        selected_inventory = inventory[start_action_step:end_action_step]
        rounds = {item.get("session_round") for item in selected_inventory}
        if len(rounds) != 1 or any(item.get("complete") is not True for item in selected_inventory):
            raise ContextCompactionRejected("episode capsule must cover one closed Episode")
        session_round = next(iter(rounds))
        if not isinstance(session_round, int):
            raise ContextCompactionRejected("episode capsule lacks its Session round")
        start = groups[start_action_step][0]
        end = groups[end_action_step][0]
        exact_messages = [
            _compactor_history_message(message) for message in messages[start:end]
        ]
        transcript_ref = self.context_artifacts.put_bytes(
            canonical_bytes(exact_messages),
            kind="episode_policy_visible_transcript",
            suffix=".json",
            media_type="application/json",
            round_index=session_round,
            action_step=start_action_step,
        )
        verdict_message = next((
            message for message in messages[start:end]
            if message.get("_context_source")
            == f"official_round_result_{session_round}"
        ), None)
        if verdict_message is None:
            raise ContextCompactionRejected(
                "episode capsule lacks its exact official round result"
            )
        raw_verdict = verdict_message.get("content")
        try:
            verdict = json.loads(
                str(raw_verdict).split("<official_round_result>\n", 1)[1]
                .split("\n</official_round_result>", 1)[0]
            )
        except (IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ContextCompactionRejected(
                "episode capsule official round result is malformed"
            ) from exc
        if (
            not isinstance(verdict, Mapping)
            or verdict.get("round") != session_round
            or not isinstance(verdict.get("passed"), bool)
        ):
            raise ContextCompactionRejected(
                "episode capsule official round result identity drift"
            )
        protected = [
            message for message in messages[:groups[0][0]]
            if message.get("_protected_context") is True
        ]
        task_message = next((
            _provider_message(message) for message in protected
            if message.get("_context_source") == "episode_task"
        ), None)
        project_message = next((
            _provider_message(message) for message in protected
            if message.get("_context_source") == "project_instructions"
        ), None)
        if task_message is None or project_message is None:
            raise ContextCompactionRejected(
                "episode capsule lacks protected task/project semantics"
            )
        header = {
            "exact_task": task_message.get("content"),
            "embodiment_semantics": project_message.get("content"),
            "tool_semantics": copy.deepcopy(list(tools)),
            "session_round": session_round,
            "instruction": (
                "Summarize only this closed Episode's durable decision evidence. "
                "Do not repeat identifiers, coordinates, hashes, or the official verdict; "
                "the exact transcript remains recoverable separately."
            ),
        }
        packet_blocks: list[dict[str, Any]] = [{
            "type": "text",
            "text": "<episode_compaction_input>\n" + json.dumps(
                header, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            ) + "\n</episode_compaction_input>",
        }]
        compactor_source = [
            message for message in messages[start:end]
            if message.get("_protected_context") is not True
        ]
        packet_blocks.extend(_packet_message_blocks(
            section="closed_episode_post_lossless_projection",
            messages=compactor_source,
        ))
        compactor_messages = [{"role": "user", "content": packet_blocks}]
        private = _compactor_private_violations(compactor_messages)
        if private:
            raise ContextCompactionRejected(
                "episode_compaction_private_evidence: " + ", ".join(private[:12])
            )
        source_event_ids = _packet_source_event_ids(compactor_source)
        source_refs = [{"event_id": event_id} for event_id in source_event_ids]
        packet_body = {
            "schema": ROLLOUT_PACKET_SCHEMA,
            "layer": "episode_capsule_l5",
            "session_round": session_round,
            "coverage": {
                "from_action_step_index": start_action_step,
                "through_action_step_index": end_action_step - 1,
                "action_step_count": end_action_step - start_action_step,
            },
            "source_refs": source_refs,
            "exact_transcript": transcript_ref,
            "compactor_messages": compactor_messages,
            "trigger": "closed_episode_provider_high_watermark",
        }
        packet_sha = hashlib.sha256(canonical_bytes(packet_body)).hexdigest()
        packet_ref = self.artifacts.put_json(
            {**packet_body, "packet_sha256": packet_sha},
            kind="episode_context_capsule_packet",
        )
        summary = self.summary_provider.summarize(
            messages=compactor_messages, source_refs=source_refs,
        )
        if canonical_bytes(list(summary.source_refs)) != canonical_bytes(source_refs):
            raise ContextCompactionRejected(
                "episode compactor source references changed across the call"
            )
        if summary.source_set_sha256 != hashlib.sha256(
            canonical_bytes(source_refs)
        ).hexdigest():
            raise ContextCompactionRejected(
                "episode compactor source-set identity is invalid"
            )
        checkpoint = _validate_checkpoint(summary.content)
        checkpoint_estimated_tokens = _estimated_tokens(checkpoint)
        checkpoint_budget = self._checkpoint_budget(
            checkpoint_estimated_tokens=checkpoint_estimated_tokens,
            metadata=summary.metadata,
        )
        capsule_body = {
            "schema": EPISODE_CAPSULE_SCHEMA,
            "revision": self.episode_checkpoint_count + 1,
            "session_round": session_round,
            "coverage": packet_body["coverage"],
            "official_verdict": copy.deepcopy(dict(verdict)),
            "exact_transcript": {
                **copy.deepcopy(transcript_ref),
                "open": recoverable_artifact_open(transcript_ref),
            },
            "source_refs": source_refs,
            "packet_sha256": packet_sha,
            "summary": checkpoint,
        }
        capsule_sha = hashlib.sha256(canonical_bytes(capsule_body)).hexdigest()
        capsule_ref = self.artifacts.put_json(
            {**capsule_body, "capsule_sha256": capsule_sha},
            kind="episode_context_capsule",
        )
        return {
            **capsule_body,
            "capsule_sha256": capsule_sha,
            "capsule_ref": capsule_ref,
            "packet_ref": packet_ref,
            "metadata": copy.deepcopy(dict(summary.metadata)),
            "checkpoint_estimated_tokens": checkpoint_estimated_tokens,
            "checkpoint_budget": checkpoint_budget,
        }

    def _checkpoint_budget(
        self, *, checkpoint_estimated_tokens: int,
        metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        assert isinstance(self.context_management, Mapping)
        usage = metadata.get("usage")
        provider_output_tokens = (
            usage.get("output_tokens") if isinstance(usage, Mapping) else None
        )
        if (
            isinstance(provider_output_tokens, bool)
            or not isinstance(provider_output_tokens, int)
            or provider_output_tokens < 0
        ):
            provider_output_tokens = None
        soft_max = int(self.context_management["checkpoint_max_estimated_tokens"])
        observed_tokens = (
            provider_output_tokens
            if provider_output_tokens is not None
            else checkpoint_estimated_tokens
        )
        return {
            "enforcement": "telemetry_only",
            "authority": (
                "provider_reported_output_tokens"
                if provider_output_tokens is not None
                else "harness_model_dependent_estimate"
            ),
            "provider_output_tokens": provider_output_tokens,
            "estimated_tokens": checkpoint_estimated_tokens,
            "target_estimated_tokens": int(
                self.context_management["checkpoint_target_estimated_tokens"]
            ),
            "soft_max_estimated_tokens": soft_max,
            "status": (
                "within_soft_max" if observed_tokens <= soft_max
                else "soft_max_exceeded"
            ),
        }

    def _lossless_emergency_checkpoint(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        inventory: Sequence[Mapping[str, Any]],
        candidate_absorbed: int,
        observation_step: int,
    ) -> dict[str, Any]:
        """Create the bounded L6 emergency checkpoint."""

        assert self.summary_provider is not None
        assert isinstance(self.context_management, Mapping)
        assert self.context_artifacts is not None
        groups = _observation_groups(messages)
        semantic_frontier = max(
            self.emergency_summarized_action_step_count,
            self.episode_checkpointed_action_step_count,
        )
        if candidate_absorbed <= semantic_frontier:
            raise PolicySessionError("semantic fallback has no new complete ActionStep")
        absorbed_start_action_step = semantic_frontier
        active_emergency_projection = (
            self.emergency_summarized_action_step_count
            > self.episode_checkpointed_action_step_count
        )
        if active_emergency_projection and (
            self.emergency_checkpoint is None
            or self.emergency_checkpoint_sha256 is None
            or self.emergency_checkpoint_ref is None
            or self.emergency_transcript_ref is None
        ):
            raise PolicySessionError(
                "active emergency coverage lacks its checkpoint evidence"
            )
        absorbed_start = groups[absorbed_start_action_step][0]
        absorbed_end = groups[candidate_absorbed][0]
        protected = [
            message for message in messages[:groups[0][0]]
            if message.get("_protected_context") is True
        ]
        task_message = next((
            _provider_message(message) for message in protected
            if message.get("_context_source") == "episode_task"
        ), None)
        project_message = next((
            _provider_message(message) for message in protected
            if message.get("_context_source") == "project_instructions"
        ), None)
        if task_message is None or project_message is None:
            raise ContextCompactionRejected(
                "semantic fallback lacks protected task/project semantics"
            )
        header = {
            "exact_task": task_message.get("content"),
            "embodiment_semantics": project_message.get("content"),
            "tool_semantics": copy.deepcopy(list(tools)),
            "previous_checkpoint": (
                None if not active_emergency_projection else {
                    field: copy.deepcopy(self.emergency_checkpoint[field])
                    for field in CHECKPOINT_FIELDS
                }
            ),
            "instruction": (
                "This is the L6 emergency fallback after L1-L5. Produce one cumulative "
                "checkpoint from existing Episode Capsules and only the newly eligible "
                "post-lossless ActionSteps. Do not copy identifiers, coordinates, hashes, "
                "or official verdicts."
            ),
        }
        packet_blocks: list[dict[str, Any]] = [{
            "type": "text",
            "text": "<rollout_compaction_input>\n" + json.dumps(
                header, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            ) + "\n</rollout_compaction_input>",
        }]
        capsule_messages = [
            _semantic_capsule_slot(capsule) for capsule in self.episode_capsules
            if capsule.get("coverage", {}).get("through_action_step_index", -1)
            >= (
                self.emergency_summarized_action_step_count
                if active_emergency_projection else 0
            )
        ]
        if capsule_messages:
            packet_blocks.extend(_packet_message_blocks(
                section="existing_episode_capsules", messages=capsule_messages,
            ))
        newly_absorbed = [
            message for message in messages[absorbed_start:absorbed_end]
            if message.get("_protected_context") is not True
        ]
        packet_blocks.extend(_packet_message_blocks(
            section="newly_absorbed_post_lossless_action_steps",
            messages=newly_absorbed,
        ))
        compactor_messages = [{"role": "user", "content": packet_blocks}]
        private = _compactor_private_violations(compactor_messages)
        if private:
            raise ContextCompactionRejected(
                "rollout_compaction_private_evidence: " + ", ".join(private[:12])
            )
        semantic_source_messages = [
            message for message in messages[
                groups[absorbed_start_action_step][0]:absorbed_end
            ]
            if message.get("_protected_context") is not True
        ]
        source_event_ids = _packet_source_event_ids(semantic_source_messages)
        source_refs = [{"event_id": event_id} for event_id in source_event_ids]
        # The current visible ref is cumulative. Replacing an older L6 slot
        # must never make an earlier exact ActionStep transcript undiscoverable
        # to the policy.
        absorbed_ref = self.context_artifacts.put_bytes(
            canonical_bytes([
                _compactor_history_message(message)
                for message in messages[
                    groups[0][0]:absorbed_end
                ]
            ]),
            kind="emergency_policy_visible_transcript",
            suffix=".json", media_type="application/json",
            round_index=None,
            action_step=absorbed_start_action_step,
        )
        packet_body = {
            "schema": ROLLOUT_PACKET_SCHEMA,
            "layer": "emergency_session_l6",
            "previous_checkpoint_sha256": (
                self.emergency_checkpoint_sha256
                if active_emergency_projection else None
            ),
            "semantic_frontier_before": absorbed_start_action_step,
            "absorbed_action_steps": copy.deepcopy(list(
                inventory[
                    absorbed_start_action_step:candidate_absorbed
                ]
            )),
            "source_refs": source_refs,
            "exact_transcript": {
                **copy.deepcopy(absorbed_ref),
                "open": recoverable_artifact_open(absorbed_ref),
            },
            "compactor_messages": compactor_messages,
            "trigger": "post_l5_emergency_high_watermark",
        }
        packet_sha = hashlib.sha256(canonical_bytes(packet_body)).hexdigest()
        packet_ref = self.artifacts.put_json(
            {**packet_body, "packet_sha256": packet_sha},
            kind="lossless_context_semantic_packet",
        )
        summary = self.summary_provider.summarize(
            messages=compactor_messages, source_refs=source_refs,
        )
        if canonical_bytes(list(summary.source_refs)) != canonical_bytes(source_refs):
            raise ContextCompactionRejected(
                "context compactor source references changed across the call"
            )
        if summary.source_set_sha256 != hashlib.sha256(
            canonical_bytes(source_refs)
        ).hexdigest():
            raise ContextCompactionRejected(
                "context compactor source-set identity is invalid"
            )
        checkpoint = _validate_checkpoint(summary.content)
        checkpoint_estimated_tokens = _estimated_tokens(checkpoint)
        checkpoint_budget = self._checkpoint_budget(
            checkpoint_estimated_tokens=checkpoint_estimated_tokens,
            metadata=summary.metadata,
        )
        checkpoint_body = {
            "schema": EMERGENCY_CHECKPOINT_SCHEMA,
            "revision": self.emergency_compaction_count + 1,
            "previous_checkpoint_sha256": (
                self.emergency_checkpoint_sha256
                if active_emergency_projection else None
            ),
            "packet_sha256": packet_sha,
            "coverage": {
                "absorbed_action_step_count": candidate_absorbed,
                "through_action_step_index": candidate_absorbed - 1,
            },
            "source_refs": source_refs,
            "checkpoint": checkpoint,
        }
        checkpoint_sha = hashlib.sha256(
            canonical_bytes(checkpoint_body)
        ).hexdigest()
        checkpoint_ref = self.artifacts.put_json(
            {**checkpoint_body, "checkpoint_sha256": checkpoint_sha},
            kind="emergency_context_checkpoint",
        )
        return {
            "checkpoint": checkpoint,
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_ref": checkpoint_ref,
            "packet_ref": packet_ref,
            "metadata": copy.deepcopy(dict(summary.metadata)),
            "checkpoint_estimated_tokens": checkpoint_estimated_tokens,
            "checkpoint_budget": checkpoint_budget,
            "exact_transcript": copy.deepcopy(absorbed_ref),
        }

    def estimate_pressure_trigger(self) -> int:
        """Forwarder; the layer shares and their order live in ``thresholds``."""

        return thresholds.estimate_pressure_trigger(
            self.context_management, self.hard_input_tokens,
        )

    def episode_checkpoint_trigger(self) -> int:
        """Forwarder; see :mod:`ora.agent.context.thresholds`."""

        return thresholds.episode_checkpoint_trigger(
            self.context_management, self.hard_input_tokens,
        )

    @_with_batched_artifact_index
    @with_projection_sizer
    def _project_action_step_lossless_v4(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        observation_step: int,
        allow_transition: bool = True,
    ) -> dict[str, Any]:
        """Project eager L1-L4, closed-Episode L5, then emergency L6.

        :mod:`.lossless_layers` documents what each layer may rewrite.
        """

        assert isinstance(self.context_management, Mapping)
        assert self.context_artifacts is not None
        view = lossless_layers.open_lossless_view(
            self, messages=messages, tools=tools,
            allow_transition=allow_transition,
        )
        lossless_layers.externalize_aged_read_only_results(view)
        lossless_layers.externalize_lifecycle_media(view)
        lossless_layers.externalize_under_pressure(view)
        candidate = lossless_semantic.closed_episode_candidate(view)
        episode = lossless_semantic.admit_episode_capsule(
            view, candidate, _episode_retry_key(candidate, view.inventory),
            summarize=self._lossless_episode_capsule,
        )
        emergency_admission = lossless_semantic.admit_emergency_checkpoint(
            view, episode, summarize=functools.partial(
                self._lossless_emergency_checkpoint,
                observation_step=observation_step,
            ),
        )
        invariants = self._verify_lossless_projection(view)
        return {
            "messages": view.provider_messages,
            "tools": list(tools),
            "manifest": self._lossless_manifest(
                view, observation_step=observation_step, episode=episode,
                emergency_admission=emergency_admission, invariants=invariants,
            ),
        }

    def _verify_lossless_projection(
        self, view: lossless_layers.LosslessView,
    ) -> _LosslessInvariants:
        """Refuse a request over the hard limit or one that lost semantics."""

        authoritative_current_tokens = (
            self.last_provider_input_tokens if view.provider_anchor_applies() else None
        )
        if (
            authoritative_current_tokens is not None
            and authoritative_current_tokens > self.hard_input_tokens
        ):
            raise PolicySessionError(
                "context_exhausted: Provider reported "
                f"{authoritative_current_tokens} input tokens, hard limit "
                f"{self.hard_input_tokens}"
            )
        violations = _causal_projection_violations([
            message for message in view.provider_messages
            if message.get("role") != "assistant"
        ])
        if violations:
            raise PolicySessionError(
                "causal_context_forbidden_fields: " + ", ".join(violations[:12])
            )

        groups, materialized = view.groups, view.materialized
        tail_start = (
            groups[view.emergency_candidate][0]
            if view.emergency_candidate < len(groups) else len(materialized)
        )
        canonical_causal_tail = _lossless_invariant_view(
            materialized[tail_start:]
        )
        projected_causal_tail = _lossless_invariant_view(
            view.lossless_working[tail_start:]
        )
        if canonical_bytes(canonical_causal_tail) != canonical_bytes(
            projected_causal_tail
        ):
            raise PolicySessionError(
                "lossless projector changed the exact causal ActionStep tail"
            )
        invariant_before_sha256 = _canonical_sha256(
            _lossless_invariant_view(materialized)
        )
        invariant_after_sha256 = _canonical_sha256(
            _lossless_invariant_view(view.lossless_working)
        )
        if invariant_before_sha256 != invariant_after_sha256:
            raise PolicySessionError(
                "lossless projector changed normalized policy-visible semantics"
            )
        return _LosslessInvariants(
            canonical_causal_tail=canonical_causal_tail,
            before_sha256=invariant_before_sha256,
            after_sha256=invariant_after_sha256,
        )

    def _lossless_manifest(
        self, view: lossless_layers.LosslessView, *, observation_step: int,
        episode: lossless_semantic.EpisodeAdmission,
        emergency_admission: Mapping[str, Any], invariants: _LosslessInvariants,
    ) -> dict[str, Any]:
        assert isinstance(self.context_management, Mapping)
        return {
            "schema": "ora.native-context-projection.v6",
            "profile": self.profile,
            "context_retention": self.retention,
            "summary_policy": self.summary_policy,
            "context_management": copy.deepcopy(dict(self.context_management)),
            "resolved_near_tail_action_steps": view.near_tail,
            "resolved_media_retention_policy": view.aging.media_policy,
            **view.aging.manifest_fields(),
            "resolved_inline_image_target": int(
                self.context_management["trigger_inline_images"]
            ),
            "resolved_episode_checkpoint_trigger_tokens": episode.threshold,
            "resolved_episode_checkpoint_trigger_formula": (
                thresholds.episode_checkpoint_trigger_formula(self.context_management)
            ),
            "observation_step": observation_step,
            "target_input_tokens": self.target_input_tokens,
            "hard_input_tokens": self.hard_input_tokens,
            **lossless_layers.token_fields(view),
            "episode_checkpoint_admission": episode.record,
            "emergency_compaction_admission": emergency_admission,
            "emergency_target_validation": copy.deepcopy(
                self.last_semantic_target_validation,
            ),
            **lossless_layers.size_fields(view),
            **_lossless_request_identity(view),
            "cache_epoch": self.cache_epoch,
            "projection_epoch": self.cache_epoch,
            "projection_transitions": view.transitions,
            "epoch_transition": view.transitions[-1] if view.transitions else None,
            **lossless_layers.coverage_fields(view),
            "causal_exact_tail_sha256": _canonical_sha256(
                invariants.canonical_causal_tail
            ),
            "lossless_invariants_before_sha256": invariants.before_sha256,
            "lossless_invariants_after_sha256": invariants.after_sha256,
            "emergency_checkpoint_sha256": (
                self.emergency_checkpoint_sha256
            ),
            "emergency_exact_transcript_ref": copy.deepcopy(
                self.emergency_transcript_ref
            ),
            "checkpoint_slot_count": sum(
                message.get("_episode_context_capsule") is True
                or message.get("_emergency_context_checkpoint") is True
                for message in view.working
            ),
            "causal_envelope": "ora.causal-observation.v1",
            "causal_forbidden_field_scan": "passed",
            "passes": lossless_layers.projection_passes(view),
        }

    def observe_provider_usage(
        self,
        *,
        estimated_tokens: int,
        input_tokens: Any,
        projection_epoch: Any = None,
        provider_wire_request_sha256: Any = None,
    ) -> None:
        """Anchor estimates to exact Provider usage for the request just sent."""

        if (
            isinstance(estimated_tokens, bool)
            or not isinstance(estimated_tokens, int)
            or estimated_tokens < 1
        ):
            raise PolicySessionError("Provider usage estimate anchor is invalid")
        if not (
            isinstance(input_tokens, int)
            and not isinstance(input_tokens, bool)
            and input_tokens >= 0
        ):
            if (
                self.pending_semantic_target_epoch is not None
                and projection_epoch == self.pending_semantic_target_epoch
            ):
                target = self.pending_semantic_target_tokens
                self.last_semantic_target_validation = {
                    "status": "failed",
                    "projection_epoch": projection_epoch,
                    "target_tokens": target,
                    "input_tokens": None,
                    "provider_wire_request_sha256": provider_wire_request_sha256,
                    "reason": "provider_input_tokens_unavailable",
                    "enforcement": "telemetry_only_until_hard_limit",
                }
                self.pending_semantic_target_epoch = None
                self.pending_semantic_target_tokens = None
            return
        if projection_epoch is None:
            if self.retention == LOSSLESS_CONTEXT_RETENTION:
                raise PolicySessionError(
                    "lossless Provider usage lacks its projection epoch"
                )
            projection_epoch = self.cache_epoch
        if (
            isinstance(projection_epoch, bool)
            or not isinstance(projection_epoch, int)
            or projection_epoch < 0
            ):
            raise PolicySessionError("Provider usage projection epoch is invalid")
        if (
            self.retention == LOSSLESS_CONTEXT_RETENTION
            and projection_epoch != self.cache_epoch
        ):
            raise PolicySessionError(
                "lossless Provider usage projection epoch is stale"
            )
        if (
            self.retention == LOSSLESS_CONTEXT_RETENTION
            and provider_wire_request_sha256 is None
        ):
            raise PolicySessionError(
                "lossless Provider usage lacks its wire request identity"
            )
        if provider_wire_request_sha256 is not None and (
            not isinstance(provider_wire_request_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", provider_wire_request_sha256) is None
        ):
            raise PolicySessionError("Provider usage wire request identity is invalid")
        self.last_estimated_tokens = estimated_tokens
        self.last_provider_input_tokens = input_tokens
        self.last_provider_projection_epoch = projection_epoch
        self.last_provider_wire_request_sha256 = provider_wire_request_sha256
        if (
            self.pending_semantic_target_epoch == projection_epoch
            and self.pending_semantic_target_tokens is not None
        ):
            target = self.pending_semantic_target_tokens
            self.last_semantic_target_validation = {
                "status": "passed" if input_tokens <= target else "failed",
                "projection_epoch": projection_epoch,
                "target_tokens": target,
                "input_tokens": input_tokens,
                "provider_wire_request_sha256": provider_wire_request_sha256,
                "enforcement": "telemetry_only_until_hard_limit",
            }
            self.pending_semantic_target_epoch = None
            self.pending_semantic_target_tokens = None
