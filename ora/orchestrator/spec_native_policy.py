"""native_api policy fields of a RunSpec: tools, context and observation.

The Harness itself runs the native_api loop, so this resolver owns the fields
that shape each Provider request: the tool allowlist, the context profile and
retention (with its ``context_management`` budget), token limits, and the
observation/composition contract. The Provider protocol check runs first and
the endpoint check right after the tool allowlist; both live in
``spec_native_provider.py``.
"""

from __future__ import annotations

from typing import Any

from ora.agent.composition import ResolvedToolSurface
from ora.agent.context.management import (
    ContextManagementConfigError,
    validate_context_management,
)
from ora.agent.context.projection import (
    ACTION_STEP_RETENTIONS, CONTEXT_PROFILES, CONTEXT_RETENTIONS, SUMMARY_POLICIES,
)
from ora.agent.memory.artifacts import LOSSLESS_CONTEXT_RETENTION
from ora.agent.native.constants import (
    NATIVE_CAUSAL_SLOTS, NATIVE_SLOTS_V4,
)

from .spec_native_provider import (
    resolve_native_provider_endpoint,
    resolve_native_provider_protocol,
)
from .spec_policy import require_tool_allowlist
from .spec_values import RunSpecError, _positive_int


def resolve_native_policy(
    policy: dict[str, Any], resolved_surface: ResolvedToolSurface,
    runtime_spec: Any,
) -> None:
    """native_api: Provider, tools, context retention, budgets, observation."""

    resolve_native_provider_protocol(policy, runtime_spec)
    tools = policy.setdefault(
        "tool_allowlist", [item["name"] for item in resolved_surface.definitions]
    )
    require_tool_allowlist(tools, resolved_surface)
    resolve_native_provider_endpoint(policy)
    context_retention = _resolve_native_context(policy)
    _resolve_native_token_budgets(policy)
    _resolve_native_observation(policy, context_retention)


def _resolve_native_context(policy: dict[str, Any]) -> Any:
    """Context profile, retention and summary policy; returns the retention."""

    policy.setdefault("context_profile", "native_minimal_v1")
    if policy["context_profile"] not in CONTEXT_PROFILES:
        raise RunSpecError("unsupported native context profile")
    context_retention = policy.get("context_retention")
    if context_retention is not None and context_retention not in CONTEXT_RETENTIONS:
        raise RunSpecError("unsupported native context retention")
    policy.setdefault("summary_policy", "disabled")
    if policy["summary_policy"] not in SUMMARY_POLICIES:
        raise RunSpecError(
            "Runtime v1 requires summary_policy=disabled or the explicit "
            "rollout_checkpoint_v1 treatment"
        )
    _resolve_context_management(policy, context_retention)
    return context_retention


def _resolve_context_management(
    policy: dict[str, Any], context_retention: Any,
) -> None:
    """Only an ActionStep retention takes ``context_management``."""

    if context_retention in ACTION_STEP_RETENTIONS:
        try:
            management = validate_context_management(policy.get("context_management"))
        except ContextManagementConfigError as exc:
            raise RunSpecError(f"policy.{exc}") from exc
        if policy["summary_policy"] != "rollout_checkpoint_v1":
            raise RunSpecError(
                f"{context_retention} requires summary_policy=rollout_checkpoint_v1"
            )
        policy["context_management"] = dict(management)
    else:
        if policy.get("context_management") is not None:
            raise RunSpecError(
                "policy.context_management requires an ActionStep retention"
            )
        if policy["summary_policy"] != "disabled":
            raise RunSpecError(
                "Runtime v1 requires summary_policy=disabled outside "
                f"{LOSSLESS_CONTEXT_RETENTION}"
            )


def _resolve_native_token_budgets(policy: dict[str, Any]) -> None:
    """Context window, output and per-observation call limits."""

    for field, default in (
        ("context_window_tokens", None),
        ("max_output_tokens", 8192),
        ("max_model_calls_per_observation", 12),
    ):
        if default is not None:
            policy.setdefault(field, default)
        policy[field] = _positive_int(policy.get(field), f"policy.{field}")
    if policy.get("max_inline_images_per_request") is not None:
        # Only the action_step_pressure_* retentions capped inline images;
        # they compose only at the commits that composed them.
        raise RunSpecError(
            "policy.max_inline_images_per_request is not supported by any "
            "context retention of this tree"
        )
    if policy["context_window_tokens"] <= policy["max_output_tokens"] + 2048:
        raise RunSpecError("native context window leaves no safe input budget")


def _resolve_native_observation(
    policy: dict[str, Any], context_retention: Any,
) -> None:
    """Streaming, images, first observation and the context composition."""

    for field, default in (("stream", True), ("supports_images", True)):
        policy.setdefault(field, default)
        if not isinstance(policy[field], bool):
            raise RunSpecError(f"policy.{field} must be boolean")
    policy.setdefault("initial_observation", "automatic")
    if policy["initial_observation"] not in {"automatic", "tool_pull"}:
        raise RunSpecError(
            "native_api policy.initial_observation must be automatic or tool_pull"
        )
    context_composition = policy.get("context_composition")
    if context_composition is not None and context_composition not in NATIVE_CAUSAL_SLOTS:
        raise RunSpecError(
            "native_api policy.context_composition must be native_slots_v3 or native_slots_v4"
        )
    if (
        context_retention in ACTION_STEP_RETENTIONS
        and context_composition not in NATIVE_CAUSAL_SLOTS
    ):
        raise RunSpecError(
            f"{context_retention} requires context_composition=native_slots_v3"
        )
    # round_static_facts_slot_v1 protects its slot through the lossless layers.
    if (
        context_composition == NATIVE_SLOTS_V4
        and context_retention != LOSSLESS_CONTEXT_RETENTION
    ):
        raise RunSpecError(
            f"native_slots_v4 requires context_retention={LOSSLESS_CONTEXT_RETENTION}"
        )
    if context_retention in ACTION_STEP_RETENTIONS:
        if policy["context_profile"] != "native_causal_context_v1":
            raise RunSpecError("ActionStep retention requires native causal context")
        if policy.get("initial_observation", "automatic") != "automatic":
            raise RunSpecError("ActionStep retention requires automatic observations")
        if policy.get("supports_images", True) is not True:
            raise RunSpecError("ActionStep retention requires an image-capable provider")
    if policy.get("fresh_observation_gate") is not None:
        raise RunSpecError("native_api policy.fresh_observation_gate is not supported")
    policy.setdefault("require_live_marker", True)
    if not isinstance(policy["require_live_marker"], bool):
        raise RunSpecError("policy.require_live_marker must be boolean")
