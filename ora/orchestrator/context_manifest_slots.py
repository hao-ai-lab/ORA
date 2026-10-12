"""Context-manifest sections for the Native slot compositions.

``native_slots_v3`` (and ``_v4``) place the harness system text,
project instructions and episode task in fixed slots ahead of the dynamic
history. ``context_manifest.build_context_manifest`` builds the sections every
policy shares and passes them here as ``common``; these builders add the
native context description, the exact static sources and the slot order.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ora.agent.context.delivery_trim import RULE_ABSENT_LISTING, declared_rules
from ora.agent.memory.artifacts import LOSSLESS_CONTEXT_RETENTION
from ora.agent.native.constants import COMPOSITION_CONTEXT_CAPABILITIES
from ora.agent.native.session import assemble_native_v3_static_context


@dataclass(frozen=True)
class _NativeV3Layout:
    """Retention family and slot positions of one ``native_slots_v3`` run."""

    action_step_retention: bool
    semantic_checkpoint: bool
    lossless_context: bool
    checkpoint_order: int | None
    history_order: int


def _native_v3_layout(policy: Mapping[str, Any]) -> _NativeV3Layout:
    retention = policy.get("context_retention", "rewrite_compact_v1")
    # The lossless retention is this tree's only ActionStep retention and its
    # only one with a semantic checkpoint slot.
    lossless_context = retention == LOSSLESS_CONTEXT_RETENTION
    action_step_retention = lossless_context
    semantic_checkpoint = lossless_context
    checkpoint_order = 3 if semantic_checkpoint else None
    history_order = 3 + int(semantic_checkpoint)
    return _NativeV3Layout(
        action_step_retention=action_step_retention,
        semantic_checkpoint=semantic_checkpoint,
        lossless_context=lossless_context,
        checkpoint_order=checkpoint_order,
        history_order=history_order,
    )


def _native_slots_v3_manifest(
    common: Mapping[str, Any], plan: Mapping[str, Any],
    policy: Mapping[str, Any], compiled_policy: Mapping[str, Path],
    context: Mapping[str, Any], definitions: list[dict[str, Any]],
    surface_capabilities: Sequence[str] = (),
) -> dict[str, Any]:
    """Manifest of a ``native_slots_v3``/``v4`` session with its retention layers."""

    harness_path = compiled_policy["harness_system"]
    project_path = compiled_policy["project_instructions"]
    task_path = compiled_policy["episode_task"]
    assembled = assemble_native_v3_static_context(
        harness_system=harness_path.read_text(encoding="utf-8"),
        project_instructions=project_path.read_text(encoding="utf-8"),
        episode_task_template=task_path.read_text(encoding="utf-8"),
        task_instruction=plan["task_instruction"],
        task_id=plan["task"],
    )
    sources = assembled["sources"]
    layout = _native_v3_layout(policy)
    native_context = _native_v3_context(
        plan, policy, assembled, layout,
        surface_capabilities=surface_capabilities,
    )
    context_order = _native_v3_context_order(
        assembled, context, definitions, layout,
    )
    return {
        **common,
        "native_context": native_context,
        "context_sources": sources,
        "context_order": context_order,
        "dynamic_call_evidence": "agent-events.jsonl",
    }


def _native_v3_context(
    plan: Mapping[str, Any], policy: Mapping[str, Any],
    assembled: Mapping[str, Any], layout: _NativeV3Layout,
    surface_capabilities: Sequence[str] = (),
) -> dict[str, Any]:
    lossless_context = layout.lossless_context
    sources = assembled["sources"]
    composition = policy.get("context_composition", "native_slots_v3")
    # observation_delivery_trim_v2: observations omit every byte-identical field
    # without naming it and list only slot fields with no current value.
    trimmed = RULE_ABSENT_LISTING in declared_rules(surface_capabilities)
    return {
        "active": True,
        "managed_system_developer_scaffold": False,
        "canonical_message_ir": "openai_chatml",
        "provider_wire_protocol": policy["provider"],
        "context_composition": composition,
        **({"context_capabilities": list(COMPOSITION_CONTEXT_CAPABILITIES[composition]),
            "round_static_facts": {
                "slot": "one protected user message after each Session round's first observation",
                "observation_listing": (
                    "current_actual.static_facts.absent" if trimmed
                    else "current_actual.static_facts.unchanged"),
                "split": "byte_identical_delivered_value",
                "aging": "never in its round; released with its round's media at reset",
            }} if composition in COMPOSITION_CONTEXT_CAPABILITIES else {}),
        "initial_observation": "automatic",
        "automatic_observation_image": True,
        "context_profile": "native_causal_context_v1",
        "context_retention": policy.get(
            "context_retention", "rewrite_compact_v1"
        ),
        "causal_observation_schema": (
            "ora.causal-observation.v1"
        ),
        "causal_pair_retention": _native_v3_causal_pair_retention(layout),
        "actual_image_growth": {
            "initial_blocks": 2,
            "new_blocks_per_mutation": 2 if layout.action_step_retention else 4,
            "duplicated_input_blocks_per_mutation": (
                0 if layout.action_step_retention else 2
            ),
        },
        "cache_epoch_policy": (
            "recoverable_artifacts_then_episode_capsules_then_emergency"
            if lossless_context else None
        ),
        "provider_reasoning_retention": "canonical_evidence_only",
        "provider_projection": _native_v3_provider_projection(lossless_context),
        "summary_policy": policy.get("summary_policy", "disabled"),
        "context_management": policy.get("context_management"),
        "checkpoint_schema": None,
        "semantic_projection_schemas": (
            {
                "episode_capsule": (
                    "ora.episode-context-capsule.v1"
                ),
                "emergency_checkpoint": (
                    "ora.emergency-context-checkpoint.v1"
                ),
                "exact_transcript_reader": "artifact.read",
            }
            if lossless_context else None
        ),
        "recoverable_artifacts": (
            {
                "ref_schema": "artifact://<policy_session_id>/<sha256>",
                "session_scoped": True,
                "survives_episode_reset": True,
                "cross_session_access": "fail_closed",
                "image_reader": "observe(ref=...)",
                "structured_reader": "artifact.read",
                "host_paths_model_visible": False,
            }
            if lossless_context else None
        ),
        "memory_policy": policy.get("memory_policy", "disabled"),
        "protected_episode_task": {
            "task_id": plan["task"],
            "task_sha256": assembled["task_sha256"],
            "source_sha256": sources[2]["sha256"],
            "compaction_status": "always_preserved_exact",
        },
        "context_window_tokens": policy["context_window_tokens"],
        "max_output_tokens": policy["max_output_tokens"],
        "canonical_trace": "agent-events.jsonl",
        "projection_evidence": "context_projected events",
        "credentials_recorded": False,
    }


def _native_v3_causal_pair_retention(layout: _NativeV3Layout) -> str:
    return (
        "latest three completed ActionSteps plus the current open "
        "ActionStep remain outside semantic L5/L6 coverage; their "
        "causal fields stay exact while older recoverable media may be "
        "represented by typed references"
        if layout.lossless_context
        else "until_next_mutation"
    )


def _native_v3_provider_projection(lossless_context: bool) -> dict[str, Any]:
    return {
        "mode": "strict_allowlist",
        "allows": [
            "initial_actual_images", "new_action_result_actual_images",
            "retained_action_input_actual_images",
            "minimal_atomic_action_effect", "current_eef_pose",
            "current_aperture", "candidate_preview", "point_queries",
            "necessary_tool_errors", "remaining_decision_budgets",
            *(
                ["artifact_logical_refs", "artifact_content_hashes"]
                if lossless_context else []
            ),
        ],
        "excludes": [
            "audit_ids",
            *(
                ["host_artifact_paths", "provider_private_reasoning"]
                if lossless_context else
                [
                    "artifact_paths", "hashes",
                    "historical_reasoning", "official_outer_verdict",
                ]
            ),
            "timestamps", "worker_transport", "full_receipts",
        ],
    }


def _native_v3_context_order(
    assembled: Mapping[str, Any],
    context: Mapping[str, Any], definitions: list[dict[str, Any]],
    layout: _NativeV3Layout,
) -> list[dict[str, Any]]:
    """Static slots, the optional checkpoint slot, then history."""

    lossless_context = layout.lossless_context
    sources = assembled["sources"]
    return sorted([
        {
            "order": 0, "kind": "harness_system", "role": "system",
            "exact_text": assembled["system_text"],
            "exact_sha256": sources[0]["sha256"],
        },
        {
            "order": 1, "kind": "project_instructions", "role": "user",
            "exact_text": assembled["project_text"],
            "exact_sha256": sources[1]["sha256"],
        },
        {
            "order": 2, "kind": "episode_task", "role": "user",
            "exact_text": assembled["episode_text"],
            "exact_sha256": sources[2]["sha256"],
        },
        {
            "order": layout.history_order, "kind": (
                "action_step_retained_exact_history"
                if layout.action_step_retention
                else "compact_assistant_tool_history"
            ),
            "role": "mixed", "example": None,
        },
        {
            "order": layout.history_order + 1, "kind": (
                "new_causal_actual_sequence"
                if layout.action_step_retention
                else "current_causal_observation"
            ),
            "role": "user", "example": context,
        },
        {
            "order": layout.history_order + 2, "kind": "canonical_tool_schemas",
            "delivery": "separate provider tools field",
            "definitions": definitions,
        },
    ] + ([{
        "order": layout.checkpoint_order,
        "kind": "tiered_episode_and_emergency_context",
        "role": "user",
        "status": "zero_or_more_episode_capsules_and_at_most_one_emergency",
        "supersession": (
            "emergency_subsumes_covered_capsules_and_exposes_cumulative_exact_ref"
        ),
    }] if layout.semantic_checkpoint else []), key=lambda item: int(item["order"]))
