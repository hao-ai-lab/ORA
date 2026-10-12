"""Prompt layers and policy composition of a RunSpec.

``bind_policy_context`` binds the ordered prompt-layer files the policy reads
(``policy_context``) to their content hashes: the typed v3 channels of a
Native slot composition, or the three legacy managed channels.
``compose_policy`` then records what the policy is offered -- the advertised
ToolSurface with its derived layers and the action codec -- as one hashed
``policy_composition``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

from ora.agent.composition import (
    POLICY_CONTEXT_V3_SCHEMA, V2_CHANNELS,
    ResolvedToolSurface,
)
from ora.agent.native.constants import NATIVE_CAUSAL_SLOTS
from ora.agent.composition.environment_context import (
    EnvironmentContextError,
    resolve_environment_prompt_profile,
)
from ora.agent.memory.artifacts import (
    LOSSLESS_CONTEXT_RETENTION,
    recoverable_artifact_definitions,
    recoverable_artifact_result_schemas,
)
from ora.agent.tools.robot import manual_sections

from .environment_spec import resolve_typed_environment_prompt_profile
from .spec_values import RunSpecError, _mapping, _resource_list, json_sha256


def bind_policy_context(
    spec: Mapping[str, Any], policy: Mapping[str, Any],
    runtime: dict[str, Any], resolved_surface: ResolvedToolSurface, *,
    profile_id: str, root: Path,
) -> dict[str, Any]:
    """Return the hash-bound prompt layers."""

    policy_context = _mapping(spec.get("policy_context"), "policy_context")
    if resolved_surface.supports("current_public_robot_facts_v1"):
        _require_public_geometry_delivery(policy_context, runtime, root=root)
    if policy_context.get("schema") == POLICY_CONTEXT_V3_SCHEMA:
        bound_policy_context = _bind_typed_policy_context(
            policy_context, policy, runtime, resolved_surface, root=root,
        )
    else:
        bound_policy_context = _bind_legacy_policy_context(
            policy_context, policy, root=root,
        )
    bound_policy_context["composition_sha256"] = json_sha256(bound_policy_context)
    return bound_policy_context


def _require_public_geometry_delivery(
    policy_context: Mapping[str, Any], runtime: Mapping[str, Any], *,
    root: Path,
) -> None:
    """The public geometry must arrive through its compiled embodiment card."""

    binding = runtime.get("environment_prompt_profile")
    public = runtime.get("public_environment_context")
    if not isinstance(binding, Mapping) or not isinstance(public, Mapping):
        raise RunSpecError("public geometry delivery requires its reviewed environment prompt profile")
    try:
        geometry_profile = resolve_environment_prompt_profile(public, project_root=root, profile_id=binding.get("profile_id"))
    except EnvironmentContextError as exc:
        raise RunSpecError(str(exc)) from exc
    compiled = geometry_profile.get("compiled_embodiment")
    if not isinstance(compiled, Mapping) or compiled.get("descriptor") != runtime["public_geometry_descriptor"]:
        raise RunSpecError("public geometry ToolSurface and compiled embodiment must be selected together")
    project_sources = policy_context.get("project_instructions", policy_context.get("codex_project"))
    if not isinstance(project_sources, list) or geometry_profile["embodiment_policy"] not in project_sources:
        raise RunSpecError("public geometry descriptor must be delivered through its exact compiled embodiment card")


def _bind_typed_policy_context(
    policy_context: Mapping[str, Any], policy: Mapping[str, Any],
    runtime: Mapping[str, Any], resolved_surface: ResolvedToolSurface, *,
    root: Path,
) -> dict[str, Any]:
    """Typed v3 channels of a Native slot composition."""

    expected_channels = {"schema", *V2_CHANNELS}
    if set(policy_context) != expected_channels:
        raise RunSpecError(
            f"typed policy_context fields must be exactly {sorted(expected_channels)}"
        )
    if policy["kind"] != "native_api":
        raise RunSpecError("typed policy_context is currently supported only by native_api")
    # A v3 policy_context binds either causal slot layout: native_slots_v4 is
    # native_slots_v3 plus the round static-facts slot (0059 D3b).
    if policy.get("context_composition") not in NATIVE_CAUSAL_SLOTS:
        raise RunSpecError(
            "policy_context requires policy.context_composition="
            + " or ".join(sorted(NATIVE_CAUSAL_SLOTS))
        )
    _require_native_slots_v3(policy, runtime, resolved_surface)
    if not resolved_surface.observation_tools or not resolved_surface.call_effect_profile:
        raise RunSpecError(
            "typed native slots require a ToolSurface observation/effect contract"
        )
    resolved_environment_profile = resolve_typed_environment_prompt_profile(runtime, root=root)
    project_sources = policy_context["project_instructions"]
    expected_project_sources = [
        resolved_environment_profile["environment_policy"],
        resolved_environment_profile["embodiment_policy"],
    ]
    # The environment-policy/embodiment pair stays the protected tail;
    # a unified prompt bundle may deliver its shared core layer ahead of
    # it in the same project channel (EXP08A bundle contract).
    project_sources_match = (
        isinstance(project_sources, list)
        and len(project_sources) >= 2
        and project_sources[-2:] == expected_project_sources
    )
    if not project_sources_match:
        raise RunSpecError(
            "policy_context project sources do not match the environment profile"
        )
    bound_policy_context = {
        "schema": POLICY_CONTEXT_V3_SCHEMA,
        **{
            channel: _resource_list(
                root, policy_context[channel], f"policy_context.{channel}"
            )
            for channel in V2_CHANNELS
        },
    }
    return bound_policy_context


def _require_native_slots_v3(
    policy: Mapping[str, Any], runtime: Mapping[str, Any],
    resolved_surface: ResolvedToolSurface,
) -> None:
    """native_slots_v3 binds the observation, codec and end-effector contract."""

    if policy.get("initial_observation") != "automatic":
        raise RunSpecError("native_slots_v3 requires initial_observation=automatic")
    if policy.get("context_profile") != "native_causal_context_v1":
        raise RunSpecError(
            "native_slots_v3 requires context_profile=native_causal_context_v1"
        )
    expected_codec = (
        "motion_action_v3"
        if resolved_surface.supports("motion_intent_router_v1")
        else "runtime_action_v3"
    )
    if policy.get("action_codec") != expected_codec:
        raise RunSpecError(
            f"native_slots_v3 requires action_codec={expected_codec}"
        )
    dual_arm_motion = (
        resolved_surface.supports("enhanced_move_pose_v1")
        and resolved_surface.supports("dual_arm_sparse_targets_v1")
    )
    if (
        resolved_surface.runtime_id != "native_atomic_causal"
        and not dual_arm_motion
    ):
        raise RunSpecError(
            "native_slots_v3 requires a native_atomic_causal or explicit "
            "dual-arm two-layer ToolSurface"
        )
    if tuple(resolved_surface.mutation_tools) != (
        "move_pose", "close_gripper", "open_gripper",
    ) and not resolved_surface.supports("enhanced_move_pose_v1"):
        raise RunSpecError(
            "native_slots_v3 requires an atomic or registered motion contract"
        )
    if dual_arm_motion and runtime.get("controlled_end_effector") is not None:
        raise RunSpecError(
            "dual-arm Native motion selects the end effector per tool call"
        )
    if not dual_arm_motion and runtime.get("controlled_end_effector") != "right":
        raise RunSpecError(
            "native_slots_v3 requires environment-bound controlled_end_effector=right"
        )


def _bind_legacy_policy_context(
    policy_context: Mapping[str, Any], policy: Mapping[str, Any], *,
    root: Path,
) -> dict[str, Any]:
    """The legacy managed channels: project, episode prefix and tool manual."""

    expected_channels = {"codex_project", "episode_prefix", "tool_manual"}
    if set(policy_context) != expected_channels:
        raise RunSpecError(
            f"policy_context channels must be exactly {sorted(expected_channels)}"
        )
    if policy.get("context_composition") is not None:
        raise RunSpecError("legacy policy_context must not set context_composition")
    bound_policy_context = {
        channel: _resource_list(
            root, policy_context[channel], f"policy_context.{channel}"
        )
        for channel in sorted(expected_channels)
    }
    return bound_policy_context


def compose_policy(
    policy: Mapping[str, Any], resolved_surface: ResolvedToolSurface,
    bound_policy_context: Mapping[str, Any], *, root: Path,
) -> dict[str, Any]:
    """Hash what the policy is offered: prompt layers, tools and codec."""

    _, tool_surface_snapshot = _advertised_tool_surface(
        policy, resolved_surface, bound_policy_context, root=root,
    )
    policy_composition = {
        "schema": "ora.policy-composition.v2",
        "prompt_layers": bound_policy_context,
        "tool_surface": tool_surface_snapshot,
        "action_codec": policy["action_codec"],
        "initial_observation": policy.get("initial_observation"),
    }
    policy_composition["composition_sha256"] = json_sha256(policy_composition)
    return policy_composition


def _advertised_tool_surface(
    policy: Mapping[str, Any], resolved_surface: ResolvedToolSurface,
    bound_policy_context: Mapping[str, Any], *, root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The allowlisted definitions plus the layers derived from the plan.

    Lossless Context Artifacts add their reader tools, and
    ``robot.read_manual`` gets the composed manual's section headings as an
    enum. Returns the advertised definitions and the ToolSurface snapshot.
    """

    advertised_definitions = resolved_surface.select(policy.get("tool_allowlist"))
    derived_layers: list[dict[str, Any]] = []
    if policy.get("context_retention") == LOSSLESS_CONTEXT_RETENTION:
        from ora.agent.memory.artifacts import artifact_read_name

        _artifact_reader = artifact_read_name(advertised_definitions)
        advertised_definitions = recoverable_artifact_definitions(
            advertised_definitions
        )
        derived_layers.append({
            "kind": "recoverable_context_artifacts_v1",
            "tools": ["observe", _artifact_reader],
            "effects": {
                "observe(ref=...)": "read_only",
                _artifact_reader: "read_only",
            },
            "logical_ref": "artifact://<policy_session_id>/<sha256>",
        })
    if any(item["name"] == "robot.read_manual" for item in advertised_definitions):
        derived_layers.append(_manual_section_enum(
            advertised_definitions, bound_policy_context, root=root,
        ))
    tool_surface_snapshot = copy.deepcopy(resolved_surface.snapshot())
    tool_surface_snapshot["definitions"] = advertised_definitions
    if policy.get("context_retention") == LOSSLESS_CONTEXT_RETENTION:
        tool_surface_snapshot["result_schemas"] = (
            recoverable_artifact_result_schemas(
                tool_surface_snapshot["result_schemas"]
            )
        )
    tool_surface_snapshot["derived_layers"] = derived_layers
    tool_surface_snapshot["advertised_composition_sha256"] = json_sha256({
        "definitions": advertised_definitions,
        "result_schemas": tool_surface_snapshot["result_schemas"],
        "mutation_tools": tool_surface_snapshot["mutation_tools"],
        "lifecycle_tools": tool_surface_snapshot["lifecycle_tools"],
        **({
            "observation_tools": tool_surface_snapshot["observation_tools"],
            "call_effect_profile": tool_surface_snapshot["call_effect_profile"],
        } if "observation_tools" in tool_surface_snapshot else {}),
        "derived_layers": derived_layers,
    })
    return advertised_definitions, tool_surface_snapshot


def _manual_section_enum(
    advertised_definitions: list[dict[str, Any]],
    bound_policy_context: Mapping[str, Any], *, root: Path,
) -> dict[str, Any]:
    """Set the manual tool's section enum in place; return its derived layer."""

    sections: list[str] = []
    for resource in bound_policy_context["tool_manual"]:
        path = (root / resource["path"]).resolve()
        sections.extend(manual_sections(path))
    if len(sections) != len(set(sections)):
        raise RunSpecError("composed manual section headings must be unique")
    for definition in advertised_definitions:
        if definition["name"] == "robot.read_manual":
            definition["inputSchema"]["properties"]["section"]["enum"] = sections
    return {
        "kind": "manual_section_enum",
        "tool": "robot.read_manual",
        "sections": sections,
        "source_sha256": [
            item["sha256"] for item in bound_policy_context["tool_manual"]
        ],
    }
