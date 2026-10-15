"""The Synthetic Native RunSpec several runtime tests drive through the real resolver.

It is the ``synthetic-native-atomic-causal-cache-v1`` cell the retired EXP01B
package composed (``native_atomic_causal_v1`` surface, ``native_slots_v3``
Context, Kimi K3 medium Provider values), with that package's three policy
layers kept as test fixtures under ``tests/data/prompts``. A test overrides the
Context or budgets it exercises; the Provider is always replaced by a fake.
"""
from __future__ import annotations

from typing import Any

from ora.agent.composition import resolve_tool_surface
from ora.agent.composition.environment_context import resolve_environment_prompt_profile
from ora.environment import synthetic_public_environment_context
from ora.orchestrator.paths import PROJECT_ROOT

CONFIG = "synthetic-native-atomic-causal-cache-v1"
PROMPTS = "tests/data/prompts"


def synthetic_spec(config: str = CONFIG, *, seed: int = 0) -> dict[str, Any]:
    """One Synthetic Native cell on ``native_atomic_causal_v1``; nothing is written."""
    if config != CONFIG:
        raise ValueError(f"only {CONFIG} is composed here")
    surface = resolve_tool_surface("native_atomic_causal_v1")
    runtime: dict[str, Any] = {
        "kind": "synthetic", "transport": "in_process_direct",
        "terminal_actions": 6, "main_camera": "agentview",
        "policy_cameras": ["agentview", "wrist"],
        "controlled_end_effector": "right",
        "workspace_bounds_m": {"x": [-0.6, 0.6], "y": [-0.4, 0.4], "z": [0.7, 1.45]},
    }
    public_environment_context = synthetic_public_environment_context(camera_names=("agentview", "wrist"))
    environment_binding = resolve_environment_prompt_profile(public_environment_context, project_root=PROJECT_ROOT)
    runtime["public_environment_context"] = public_environment_context
    runtime["environment_prompt_profile"] = {
        key: environment_binding[key]
        for key in ("schema", "profile_id", "profile_path", "profile_sha256", "metadata_sha256")
    }
    policy = {
        "kind": "native_api", "provider": "anthropic_messages",
        "provider_profile": "kimi_code_k3_anthropic_medium_v1", "model": "k3",
        "model_env": "NATIVE_MODEL_ID", "endpoint": "https://api.kimi.com/coding",
        "base_url_env": "NATIVE_MODEL_BASE_URL", "api_key_env": "NATIVE_MODEL_API_KEY",
        "proxy_env": ["HTTPS_PROXY", "https_proxy"], "reasoning_effort": "medium",
        "context_window_tokens": 262144, "max_output_tokens": 8192, "stream": True,
        "supports_images": True,
        "provider_qualification_schema": "ora.native-provider-probe.v1",
        "provider_qualification_version": 1,
        "context_profile": "native_causal_context_v1",
        "context_composition": "native_slots_v3",
        "context_retention": "append_only_epoch_v1",
        "summary_policy": "disabled",
        "memory_policy": "disabled",
        "require_live_marker": True,
        "timeout_s": 900,
        "tool_allowlist": [item["name"] for item in surface.definitions],
        "tool_surface": {"profile": surface.profile_id, "component_overrides": {}},
        "action_codec": "runtime_action_v3",
        "initial_observation": "automatic",
    }
    policy_context = {
        "schema": "ora.policy-context.v3",
        "harness_system": [f"{PROMPTS}/harness-system-append-only-v1.md"],
        "project_instructions": [
            environment_binding["environment_policy"],
            environment_binding["embodiment_policy"],
        ],
        "episode_task": [f"{PROMPTS}/episode-task-v1.md"],
        "tool_manual": [f"{PROMPTS}/tool-schema-delivery-v1.md"],
    }
    return {
        "schema": "ora.run-spec.v1",
        "experiment": "native-harness-exp01b", "config": config,
        "output_workspace": "outputs/native-harness-exp01b", "milestone": "NATIVE-EXP01B-SYNTHETIC",
        "task": "synthetic_atomic_causal",
        "task_instruction": "Validate atomic pose and gripper actions with automatic causal observations.",
        "seed": seed, "evidence_tier": "diagnostic",
        "policy": policy, "runtime": runtime,
        "budgets": {"max_robot_decisions": 6, "max_model_calls": 18, "max_sim_steps": 6, "wall_time_s": 900},
        "policy_context": policy_context,
        "qualification": {"required": False}, "prerequisite": None,
    }
