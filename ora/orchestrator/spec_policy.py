"""The policy fields every RunSpec policy kind shares.

``resolve_run_spec`` calls ``resolve_policy_surface`` for the kind, model,
Provider, ToolSurface and action codec, then the kind's own resolver from
``POLICY_KIND_RESOLVERS`` in ``spec.py``: ``resolve_scripted_policy`` here
and ``resolve_native_policy`` for native_api
(``spec_native_policy.py``). Every resolver sets defaults on, and normalizes,
the caller's own ``policy`` copy in place.
"""

from __future__ import annotations

from typing import Any, Mapping

from ora.agent.composition import (
    ResolvedToolSurface,
    ToolSurfaceError,
    resolve_tool_surface,
)

from .spec_values import RunSpecError, _mapping

POLICY_KINDS = {"scripted", "native_api"}
#: The action codecs this tree accepts.
ACTION_CODECS = frozenset({
    "runtime_action_v3", "target_state_v4", "motion_action_v3",
})


def resolve_policy_surface(
    spec: Mapping[str, Any],
) -> tuple[dict[str, Any], ResolvedToolSurface, str]:
    """Return the policy copy, its resolved ToolSurface and the profile id."""

    policy = dict(_mapping(spec.get("policy"), "policy"))
    if policy.get("kind") not in POLICY_KINDS:
        raise RunSpecError(
            "policy.kind must be scripted or native_api"
        )
    if policy["kind"] == "native_api":
        policy.setdefault("reasoning_effort", "provider_default")
    for field in ("model", "provider", "reasoning_effort"):
        if not isinstance(policy.get(field), str) or not policy[field]:
            raise RunSpecError(f"policy.{field} must be non-empty")
    profile_id, resolved_surface = _bind_tool_surface(spec, policy)
    _resolve_action_codec(policy, profile_id, resolved_surface)
    return policy, resolved_surface, profile_id


def _bind_tool_surface(
    spec: Mapping[str, Any], policy: dict[str, Any],
) -> tuple[str, ResolvedToolSurface]:
    """Resolve the selected ToolSurface profile and record its identity."""

    default_surface = "robot_runtime_v2"
    raw_surface = policy.setdefault("tool_surface", {
        "profile": default_surface, "component_overrides": {},
    })
    if not isinstance(raw_surface, Mapping) or set(raw_surface) - {
        "profile", "component_overrides",
    }:
        raise RunSpecError("policy.tool_surface must contain profile and component_overrides")
    profile_id = raw_surface.get("profile", default_surface)
    overrides = raw_surface.get("component_overrides", {})
    if not isinstance(profile_id, str) or not isinstance(overrides, Mapping):
        raise RunSpecError("policy.tool_surface profile/overrides are invalid")
    try:
        resolved_surface = resolve_tool_surface(
            profile_id, component_overrides=overrides,
        )
    except ToolSurfaceError as exc:
        raise RunSpecError(str(exc)) from exc
    policy["tool_surface"] = {
        "profile": profile_id,
        "component_overrides": dict(overrides),
        "composition_sha256": resolved_surface.composition_sha256,
        "manifest_sha256": resolved_surface.manifest_sha256,
    }
    return profile_id, resolved_surface


def _resolve_action_codec(
    policy: dict[str, Any], profile_id: str,
    resolved_surface: ResolvedToolSurface,
) -> None:
    """Default the action codec and require the one the ToolSurface compiles to."""

    policy.setdefault(
        "action_codec",
        "runtime_action_v3" if profile_id == "robot_runtime_v1" else "target_state_v4",
    )
    if policy["action_codec"] not in ACTION_CODECS:
        raise RunSpecError(
            "policy.action_codec must be runtime_action_v3, target_state_v4 "
            "or motion_action_v3"
        )
    if (
        policy["action_codec"] == "motion_action_v3"
        and not resolved_surface.supports("dual_arm_sparse_targets_v1")
        # EXP08A: the single-arm flat form compiles to the same sparse v3
        # action internally (targets.right), so it satisfies the sparse
        # contract without a dual-arm wrapper.
        and not resolved_surface.supports("single_arm_flat_targets_v1")
    ):
        raise RunSpecError(
            "sparse motion action requires a dual-arm sparse-target ToolSurface"
        )
    if profile_id == "robot_runtime_v1" and policy["action_codec"] != "runtime_action_v3":
        raise RunSpecError("robot_runtime_v1 requires runtime_action_v3")
    if (
        profile_id == "robot_runtime_v2"
        and policy["action_codec"] != "target_state_v4"
    ):
        raise RunSpecError(f"{profile_id} requires target_state_v4")


def require_tool_allowlist(
    tools: Any, resolved_surface: ResolvedToolSurface,
) -> None:
    """Require unique, known tool names that keep every mutation tool."""

    known_tools = {item["name"] for item in resolved_surface.definitions}
    required_mutations = set(resolved_surface.mutation_tools)
    if (
        not isinstance(tools, list)
        or len(tools) != len(set(tools))
        or not set(tools).issubset(known_tools)
        or not required_mutations.issubset(tools)
    ):
        raise RunSpecError(
            "policy.tool_allowlist must be unique, known and include mutation tools"
        )


def resolve_scripted_policy(
    policy: dict[str, Any], resolved_surface: ResolvedToolSurface,
    runtime_spec: Any,
) -> None:
    """scripted: a fixed action list; the allowlist follows ``state_probe``."""

    actions = policy.get("actions")
    if not isinstance(actions, list) or not actions:
        raise RunSpecError("scripted policy requires at least one action")
    policy.setdefault("state_probe", False)
    if not isinstance(policy["state_probe"], bool):
        raise RunSpecError("policy.state_probe must be boolean")
    if policy["state_probe"]:
        policy["tool_allowlist"] = [
            "observation.dump_state", "robot.submit_action"
        ]
    else:
        policy["tool_allowlist"] = list(resolved_surface.mutation_tools)
