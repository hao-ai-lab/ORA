"""One canonical scaffold for planned and persisted agent-context metadata.

``build_context_manifest`` resolves the run's advertised tools, builds the sections every policy kind shares, then adds the context
order of the selected policy: a Native slot composition (built in
``context_manifest_slots``), the Native v1 single system prefix, or a managed
Codex/Claude Code session.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ora.agent.context.codex import build_context_policy
from ora.agent.native.session import assemble_native_v3_static_context
from ora.agent.memory.artifacts import LOSSLESS_CONTEXT_RETENTION
from ora.agent.native.constants import NATIVE_CAUSAL_SLOTS
from ora.agent.tools.robot import tool_definitions
from ora.agent.composition import resolve_tool_surface
from ora.environment import (
    ENVIRONMENT_SESSION_SCHEMA, ENVIRONMENT_STATUS_SCHEMA, RPC_SCHEMA,
)
from ora.orchestrator.paths import WORKSPACE_ROOT

from .context_manifest_slots import _native_slots_v3_manifest


SCHEMA = "ora.context-manifest.v1"


def context_placeholder(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Build the explicit non-runtime preview used by the context CLI."""
    policy = plan["policy"]
    tools = policy.get("tool_allowlist", ["robot.submit_action"])
    main_camera = str(plan["runtime"].get("main_camera", "head_camera"))
    full = {
        "schema": "ora.context.v1",
        "preview": True,
        "observation_fresh": True,
        "main_camera": main_camera,
        "main_image": f"inputs/current/{main_camera}/rgb-thumbnail.png",
        "proprioception": "<current EEF, gripper and compact joints>",
        "previous_execution": "<previous public motion receipt or null>",
        "remaining_budget": {
            "robot_decisions": plan["budgets"]["max_robot_decisions"],
            "model_calls": plan["budgets"]["max_model_calls"],
            "wall_time_s": plan["budgets"]["wall_time_s"],
        },
        "runtime_notice": None,
        "policy_state": {
            "step": 0,
            "path": "policy-state/steps/step-0000.json",
            "sha256": "<snapshot-file-sha256>",
            "state_sha256": "<canonical-state-sha256>",
        },
        "manual": "contract/robot-manual.md" if "robot.read_manual" in tools else None,
        "task": "contract/task.md",
        "forbidden_modalities_absent": [
            "actor_state", "contact", "expert_program", "object_gt_pose",
            "segmentation", "success_predicate", "task_source",
        ],
    }
    return build_context_policy(
        policy.get("context_projection", "current_observation_v1")
    ).project(full)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class _AdvertisedTools:
    """The resolved ToolSurface and the exact definitions the policy sees."""

    allowed: list[Any]
    surface_config: Any
    surface_profile: Any
    resolved_surface: Any
    definitions: list[dict[str, Any]]


def build_context_manifest(
    plan: Mapping[str, Any],
    *,
    compiled_policy: Mapping[str, Path],
    context: Mapping[str, Any],
    run_id: str | None = None,
    public_history: Sequence[str] = (),
    workspace_root: Path = WORKSPACE_ROOT,
) -> dict[str, Any]:
    """Describe context ownership, ordering, surfaces and exact project text.

    The advertised tools are resolved first; the
    sections every policy shares follow; the policy kind then selects the
    context order: a Native slot composition, the Native v1 single system
    prefix, or none.
    """
    policy = plan["policy"]
    tools = _advertised_tools(plan, policy, compiled_policy)
    common = _common_sections(plan, policy, tools, run_id=run_id)
    if policy["kind"] == "native_api":
        if policy.get("context_composition") in NATIVE_CAUSAL_SLOTS:
            return _native_slots_v3_manifest(
                common, plan, policy, compiled_policy, context,
                tools.definitions,
                surface_capabilities=tuple(getattr(tools.resolved_surface, "capabilities", ()) or ()),
            )
        return _native_v1_manifest(
            common, plan, policy, compiled_policy, context, tools.definitions,
        )
    return {
        **common,
        "context_order": [],
        "dynamic_call_evidence": None,
    }


def _advertised_tools(
    plan: Mapping[str, Any], policy: Mapping[str, Any],
    compiled_policy: Mapping[str, Path],
) -> _AdvertisedTools:
    """Resolve the ToolSurface and label each advertised tool's state effect."""

    allowed = list(policy.get("tool_allowlist", ["robot.submit_action"]))
    surface_config = policy.get("tool_surface", {})
    surface_profile = (
        surface_config.get("profile", "robot_runtime_v1")
        if isinstance(surface_config, Mapping) else "robot_runtime_v1"
    )
    surface_overrides = (
        surface_config.get("component_overrides", {})
        if isinstance(surface_config, Mapping) else {}
    )
    resolved_surface = resolve_tool_surface(
        surface_profile, component_overrides=surface_overrides,
    )
    mutations = set(resolved_surface.mutation_tools)
    lifecycle = set(resolved_surface.lifecycle_tools)
    planned_definitions = plan.get("policy_composition", {}).get(
        "tool_surface", {}
    ).get("definitions")
    exact_definitions = (
        planned_definitions
        if isinstance(planned_definitions, list)
        else tool_definitions(
            allowed, manual_path=compiled_policy.get("tool_manual"),
            profile_id=surface_profile,
            component_overrides=surface_overrides,
        )
    )
    definitions = []
    for definition in exact_definitions:
        definitions.append({
            **definition,
            "state_effect": (
                "state_changing_terminal_for_observation"
                if definition["name"] in mutations
                else "policy_lifecycle" if definition["name"] in lifecycle
                else "read_only"
            ),
        })
    return _AdvertisedTools(
        allowed=allowed,
        surface_config=surface_config,
        surface_profile=surface_profile,
        resolved_surface=resolved_surface,
        definitions=definitions,
    )


def _common_sections(
    plan: Mapping[str, Any], policy: Mapping[str, Any], tools: _AdvertisedTools,
    *, run_id: str | None,
) -> dict[str, Any]:
    """The manifest sections every policy kind shares, in their fixed order."""

    allowed = tools.allowed
    common = {
        "schema": SCHEMA,
        "run_id": run_id,
        "plan_id": plan["plan_id"],
        "plan_sha256": plan["plan_sha256"],
        "experiment": plan["experiment"],
        "config": plan["config"],
        "planner": _planner_section(policy),
        "environment": _environment_section(plan),
        "policy_context": plan["policy_context"],
        "policy_composition": plan.get("policy_composition"),
        "tool_surface": _tool_surface_section(policy, tools),
        "state_system": {
            "schema": "ora.policy-state.v1",
            "manifest_schema": "ora.policy-state-manifest.v1",
            "manifest_path": "policy-state/manifest.json",
            "step_template": "policy-state/steps/step-NNNN.json",
            "single_writer": "trusted supervisor",
            "historical_policy_access": "observation.dump_state" in allowed,
        },
        "context_bundle_system": {
            "schema": "ora.context-bundle.v1",
            "path_template": "context-bundles/call-NNNN.json",
            "transcript_linkage": "agent-transcript.jsonl",
        },
        "audit_system": _audit_system_section(),
        "agent_session_system": _agent_session_section(policy),
        "workspace_surface": {
            "root": "sandbox/workspace",
            "project_instructions": "AGENTS.md",
            "episode_prefix_evidence": "contract/episode-prefix.md",
            "manual": "contract/robot-manual.md" if "robot.read_manual" in allowed else None,
            "task": "contract/task.md",
            "current_context": "inputs/current-context.json",
            "current_observation": "inputs/current/",
            "writable_scratch": "scratch/",
            "runtime_capabilities": "contract/runtime-capabilities.json",
            "execution_cwd": "scratch/",
        },
    }
    if "context_retention" in policy:
        common["planner"]["context_retention"] = policy["context_retention"]
        common["agent_session_system"]["context_retention"] = policy[
            "context_retention"
        ]
    if "context_management" in policy:
        common["planner"]["context_management"] = policy["context_management"]
        common["agent_session_system"]["context_management"] = policy[
            "context_management"
        ]
    return common


def _planner_section(policy: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": policy["kind"],
        "provider": policy["provider"],
        "provider_profile": policy.get("provider_profile"),
        "endpoint": policy.get("endpoint"),
        "model": policy["model"],
        "reasoning_effort": policy["reasoning_effort"],
        "stream": policy.get("stream"),
        "supports_images": policy.get("supports_images"),
        "max_inline_images_per_request": policy.get(
            "max_inline_images_per_request"
        ),
        "agent_session_backend": policy.get("agent_session_backend"),
        "context_rollover_policy": policy.get("context_rollover_policy"),
        "context_projection": policy.get("context_projection"),
        "context_profile": policy.get("context_profile"),
        "context_window_tokens": policy.get("context_window_tokens"),
        "max_output_tokens": policy.get("max_output_tokens"),
        "summary_policy": policy.get("summary_policy"),
    }


def _environment_section(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "session_contract": ENVIRONMENT_SESSION_SCHEMA,
        "backend": plan["runtime"]["kind"],
        "transport": plan["runtime"].get("transport", "in_process_direct"),
        "process_isolated": plan["runtime"].get("transport") == "local_rpc",
        "runtime_python_env": plan["runtime"].get("python_env"),
        "transport_protocol": (
            RPC_SCHEMA if plan["runtime"].get("transport") == "local_rpc" else None
        ),
        "session_evidence": (
            "native/environment-session.json"
            if plan["runtime"].get("transport") == "local_rpc"
            else None
        ),
        "exclusive_writer": True,
        "public_observation_schema": "ora-observation.v1",
        "state_change_boundary": "execute(action)",
        "official_verdict_owner": "environment",
        "status_contract": ENVIRONMENT_STATUS_SCHEMA,
        "status_policy": plan["runtime"].get(
            "environment_signal_policy", "lifecycle_only"
        ),
        "status_visibility": "control_only",
        "policy_forbidden_inputs": [
            "actor_state", "contact", "expert_program", "object_gt_pose",
            "segmentation", "success_predicate", "task_source",
        ],
    }


def _tool_surface_section(
    policy: Mapping[str, Any], tools: _AdvertisedTools,
) -> dict[str, Any]:
    surface_config = tools.surface_config
    resolved_surface = tools.resolved_surface
    return {
        "profile": tools.surface_profile,
        "composition_sha256": surface_config.get("composition_sha256")
        if isinstance(surface_config, Mapping) else None,
        "protocol": "trusted_direct",
        "advertised_in_order": tools.definitions,
        "state_changing_tools": list(resolved_surface.mutation_tools),
        "lifecycle_tools": list(resolved_surface.lifecycle_tools),
        **({
            "observation_tools": list(resolved_surface.observation_tools),
            "call_effect_profile": resolved_surface.call_effect_profile,
            "effect_classification": "per_call_arguments",
        } if policy.get("context_composition") in NATIVE_CAUSAL_SLOTS else {}),
        "only_state_changing_tool": (
            resolved_surface.mutation_tools[0]
            if len(resolved_surface.mutation_tools) == 1 else None
        ),
        "policy_state_reader": (
            "observation.dump_state"
            if "observation.dump_state" in tools.allowed else None
        ),
    }


def _audit_system_section() -> dict[str, Any]:
    return {
        "runtime_metrics": "runtime-metrics.json",
        "context_audit": "context-audit.json",
        "coding_audit": "native/coding-audit.jsonl",
    }


def _agent_session_section(policy: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": (
            "ora.native-policy-session.v1"
            if policy["kind"] == "native_api"
            else "ora.agent-session.v2"
        ),
        "path": "native/agent-session.json",
        "backend": policy.get("agent_session_backend"),
        "rollover_policy": policy.get("context_rollover_policy"),
        "logical_session_persistent": policy.get("context_rollover_policy") == "native_only",
    }


def _native_v1_manifest(
    common: Mapping[str, Any], plan: Mapping[str, Any],
    policy: Mapping[str, Any], compiled_policy: Mapping[str, Path],
    context: Mapping[str, Any], definitions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Native v1: project policy and task prefix share one system message."""

    project_path = compiled_policy["codex_project"]
    prefix_path = compiled_policy["episode_prefix"]
    task_instruction = plan["task_instruction"].strip()
    exact_system_text = (
        project_path.read_text(encoding="utf-8").strip()
        + "\n\n"
        + prefix_path.read_text(encoding="utf-8").strip().replace(
            "{{TASK}}", task_instruction
        )
    )
    return {
        **common,
        "native_context": {
            "active": True,
            "managed_system_developer_scaffold": False,
            "canonical_message_ir": "openai_chatml",
            "provider_wire_protocol": policy["provider"],
            "initial_observation": policy.get("initial_observation", "automatic"),
            "automatic_observation_image": (
                policy["supports_images"]
                and policy.get("initial_observation", "automatic") == "automatic"
            ),
            "context_profile": policy["context_profile"],
            "summary_policy": policy["summary_policy"],
            "context_window_tokens": policy["context_window_tokens"],
            "max_output_tokens": policy["max_output_tokens"],
            "target_input_fraction": 0.75,
            "hard_input_reserve_tokens": 2048,
            "canonical_trace": "agent-events.jsonl",
            "projection_evidence": "context_projected events",
            "provider_request_evidence": "native/agent-artifacts",
            "credentials_recorded": False,
        },
        "context_order": [
            {
                "order": 0,
                "owner": "Robot Harness",
                "kind": "stable_system_policy_prefix",
                "source_layers": [
                    *plan["policy_context"]["codex_project"],
                    *plan["policy_context"]["episode_prefix"],
                ],
                "compiled_sha256": {
                    "project": _sha256(project_path),
                    "episode_prefix": _sha256(prefix_path),
                },
                "exact_text": exact_system_text,
                "exact_sha256": hashlib.sha256(
                    exact_system_text.encode("utf-8")
                ).hexdigest(),
                "placement_note": (
                    "Native v1 places both project policy and the task-bound "
                    "episode prefix in the sole system message; this differs "
                    "from managed Codex and is retained pending prompt-layer review."
                ),
            },
            {
                "order": 1,
                "owner": "Robot Harness",
                "kind": "current_observation_and_budget",
                "example": context,
            },
            {
                "order": 2,
                "owner": "Robot Harness",
                "kind": "openai_function_tools",
                "definitions": definitions,
            },
        ],
        "dynamic_call_evidence": "agent-events.jsonl",
    }


