"""Construct one policy session without coupling the Coordinator to a provider."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from ora.agent.contracts import (
    ModelProvider, PolicySessionError,
)
from ora.agent.native.session import NativePlanner
from ora.agent.native.policy_session import NativePolicySession
from ora.agent.sessions import ScriptedPolicySession
from ora.agent.providers import build_native_provider
from ora.agent.providers.credentials import (
    credential_ring, qualified_credential_index,
)
from ora.agent.scripted import ScriptedPlanner


def _credential_ring(
    api_key: str, standby: list[str], slot_id: Any,
) -> tuple[list[str], int]:
    """Spread a launch batch over the host-local credential ring.

    Concurrent sessions occupy distinct host-exclusive slots, so the slot
    ordinal picks each session's sticky starting key; exhaustion rotation
    then walks the rest of the ring in order.
    """

    return credential_ring(api_key, standby, slot_id)


def ensure_policy_session(
    value: Any, *, kind: str, supports_images: bool,
) -> Any:
    """Enforce one typed PolicySession at the orchestrator boundary."""

    if hasattr(value, "capabilities") and hasattr(value, "accept_runtime_event"):
        return value
    if kind == "native_api":
        return NativePolicySession(value, supports_images=supports_images)
    return ScriptedPolicySession(value)


def policy_session_hints(policy: Mapping[str, Any]) -> dict[str, Any]:
    """Return orchestration hints without exposing backend selection branches."""

    kind = str(policy.get("kind", ""))
    redactions: list[str] = []
    if kind == "native_api":
        redactions = [
            str(policy["base_url_env"]),
            str(policy["api_key_env"]),
            *(str(item) for item in policy.get("proxy_env", [])),
        ]
    return {
        "session_kind": kind,
        "supports_images": bool(policy.get("supports_images", False)),
        "redacted_environment_names": tuple(redactions),
    }


def build_policy_session(
    *,
    plan: Mapping[str, Any],
    run_root: Path,
    run_id: str,
    workspace: Path,
    control_root: Path,
    events_path: Path,
    compiled_policy: Mapping[str, Path],
    tools: Any,
    state_root: Path,
    record_context_bundle: Any,
    provider_override: ModelProvider | None = None,
    compactor_provider_override: ModelProvider | None = None,
) -> Any:
    policy = plan["policy"]
    if policy["kind"] == "scripted":
        return ScriptedPolicySession(ScriptedPlanner(
            tools=tools,
            actions=policy["actions"],
            state_probe=bool(policy.get("state_probe", False)),
        ))
    if policy["kind"] != "native_api":
        raise PolicySessionError(f"unsupported policy kind: {policy['kind']}")
    provider = provider_override
    if provider is None:
        base_url = os.environ.get(policy["base_url_env"])
        api_key = os.environ.get(policy["api_key_env"])
        if not base_url or not api_key:
            raise PolicySessionError(
                f"native provider environment is incomplete: "
                f"{policy['base_url_env']}, {policy['api_key_env']}"
            )
        if base_url.rstrip("/") != policy["endpoint"]:
            raise PolicySessionError(
                f"native provider endpoint drift: expected {policy['endpoint']}"
            )
        # Host-local rotation pool: NATIVE_MODEL_API_KEYS may list additional
        # comma-separated credentials used only when the active one is
        # exhausted or rejected. Values stay in the operator environment.
        standby = [
            item.strip()
            for item in os.environ.get(
                f"{policy['api_key_env']}S", "",
            ).split(",")
            if item.strip() and item.strip() != api_key
        ]
        selection = None
        qualification_path = run_root / "preflight.json"
        formal_plan = bool(plan.get("plan_id") and plan.get("plan_sha256"))
        if policy.get("require_live_marker", True) and (formal_plan or qualification_path.exists()):
            if not qualification_path.is_file():
                raise PolicySessionError("native credential qualification report is missing")
            pool, _ = credential_ring(api_key, standby, selected_index=0)
            selection = qualified_credential_index(
                plan, json.loads(qualification_path.read_text(encoding="utf-8")),
                len(pool),
            )
        ring, start = credential_ring(
            api_key, standby,
            (plan.get("runtime") or {}).get("resource_binding", {}).get("slot_id"),
            selected_index=selection,
        )
        provider = build_native_provider(
            policy["provider"], base_url=policy["endpoint"], api_key=ring[0],
            standby_api_keys=tuple(ring[1:]),
            credential_start_index=start,
            recovery_policy=policy.get("provider_recovery", "legacy_v1"),
            recovery_window_s=policy.get("provider_recovery_window_s"),
            timeout_seconds=float(policy.get("timeout_s", 900)),
            auth_scheme=policy.get("auth_scheme"),
            output_token_parameter=policy.get("output_token_parameter"),
            tool_strictness=policy.get("tool_strictness"),
            reasoning_field=policy.get("reasoning_field"),
            request_options=policy.get("request_options"),
            continuation_replay=policy.get(
                "provider_continuation_replay", "exact_v1",
            ),
            session_affinity_id=(
                f"ora:{run_id}:native"
                if policy.get("provider_session_affinity")
                == "x_session_id_run_v1"
                else None
            ),
        )
    return NativePolicySession(NativePlanner(
        run_root=run_root,
        run_id=run_id,
        tools=tools,
        provider=provider,
        project_instructions_path=compiled_policy["codex_project"],
        episode_prefix_path=compiled_policy["episode_prefix"],
        model=policy["model"],
        reasoning_effort=policy["reasoning_effort"],
        context_profile=policy["context_profile"],
        context_window_tokens=policy["context_window_tokens"],
        context_retention=policy.get("context_retention", "rewrite_compact_v1"),
        context_management=policy.get("context_management"),
        max_inline_images_per_request=policy.get(
            "max_inline_images_per_request"
        ),
        compactor_provider=compactor_provider_override,
        max_output_tokens=policy["max_output_tokens"],
        stream=policy["stream"],
        supports_images=policy["supports_images"],
        initial_observation=policy.get("initial_observation", "automatic"),
        summary_policy=policy["summary_policy"],
        max_model_calls_per_observation=int(policy.get("max_model_calls_per_observation", 12)),
        context_composition=policy.get("context_composition"),
        harness_system_path=compiled_policy.get("harness_system"),
        task_id=str(plan["task"]),
    ), supports_images=bool(policy["supports_images"]))


__all__ = ["build_policy_session", "policy_session_hints"]
