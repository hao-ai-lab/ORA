"""Atomic default composition for new Native robot RunSpecs.

Explicit ToolSurfaces retain their historical behavior. Baseline selection is
not host qualification, service startup or authority to launch an episode.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Mapping

import yaml

from ora.environment.benchmarks.robotwin import TASK_CONFIGS
from ora.agent.composition.baseline import (
    BASELINE_ID, BASELINE_CATALOGS, load_native_baseline,
)

# Baselines whose RoboTwin contract admits either official scene configuration.
SCENE_CONFIGURATION_CHOICE_BASELINES = frozenset({BASELINE_ID})
# Native robot environments a baseline catalog may carry an instance for.
NATIVE_BASELINE_KINDS = frozenset({"libero_pro", "robotwin", "robodojo"})
from ora.agent.composition.environment_context import (
    resolve_environment_prompt_profile,
)
from ora.agent.composition.task_contract import resolve_task_contract_layer
from ora.agent.composition.tool_surface import resolve_tool_surface
from ora.environment.definitions import resolve_runtime_descriptor


def _fill(target: dict[str, Any], defaults: Mapping[str, Any]) -> None:
    for key, value in defaults.items():
        if key not in target:
            target[key] = copy.deepcopy(value)
        elif isinstance(value, Mapping) and isinstance(target[key], dict):
            _fill(target[key], value)


def _check_runtime_contract(
    runtime: Mapping[str, Any], template: Mapping[str, Any],
    prefix: str = "runtime",
) -> None:
    for key, expected in template.items():
        # The timeouts and the execution host are launch bindings: a baseline
        # instance names a default, the RunSpec may bind another registered one.
        if key not in runtime or key in {"startup_timeout_s", "request_timeout_s", "required_host"}:
            continue
        actual = runtime[key]
        if isinstance(expected, Mapping) and isinstance(actual, Mapping):
            _check_runtime_contract(actual, expected, f"{prefix}.{key}")
        elif actual != expected:
            raise ValueError(f"policy.baseline conflicts with {prefix}.{key}")


def expand_native_defaults(spec: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    """Expand one selected baseline without mutating caller-owned input."""
    result = copy.deepcopy(dict(spec))
    policy, runtime = result.get("policy"), result.get("runtime")
    if not isinstance(policy, dict) or not isinstance(runtime, dict):
        return result
    requested = policy.get("baseline")
    if "baseline" in policy and (not isinstance(requested, str) or requested not in BASELINE_CATALOGS):
        raise ValueError("policy.baseline must be a registered id string")
    eligible = policy.get("kind") == "native_api" and runtime.get("kind") in NATIVE_BASELINE_KINDS
    if requested is not None and not eligible:
        raise ValueError("policy.baseline requires a supported Native robot environment")
    if not eligible or (requested is None and "tool_surface" in policy):
        return result
    catalog = load_native_baseline(root, requested or BASELINE_ID)
    if runtime["kind"] not in catalog["instances"]:
        # A catalog without an instance for this environment leaves an
        # omitted-baseline RunSpec as written.
        if requested is not None:
            raise ValueError(f"policy.baseline {catalog['id']} has no {runtime['kind']} instance")
        return result
    instance = catalog["instances"][runtime["kind"]]
    surface = {"profile": instance["tool_surface"], "component_overrides": {}}
    bundle_ref = instance["prompt_bundle"]
    raw_surface = policy.get("tool_surface", surface)
    if not isinstance(raw_surface, Mapping) or (
        raw_surface.get("profile") != surface["profile"]
        or raw_surface.get("component_overrides", {}) != surface["component_overrides"]
        or set(raw_surface) - {"profile", "component_overrides"}
    ):
        raise ValueError("policy.baseline tool_surface must match its unmodified instance")
    # Baseline semantics are one unit. Custom treatments choose an explicit
    # ToolSurface instead of silently inheriting a misleading baseline label.
    for field in (
        "action_codec", "context_profile", "context_composition", "context_retention",
        "summary_policy", "memory_policy", "provider_continuation_replay",
        "initial_observation", "context_management",
        "require_live_marker",
        "provider_recovery",
    ):
        if field in catalog["policy"] and field in policy and policy[field] != catalog["policy"][field]:
            raise ValueError(f"policy.baseline conflicts with policy.{field}")
    if "policy_context" in result:
        raise ValueError(
            "policy.baseline owns policy_context; choose an explicit custom surface"
        )
    if policy.get("supports_images") is not True:
        raise ValueError("policy.baseline requires explicit supports_images=true")
    if "tool_allowlist" in policy:
        names = [
            item["name"]
            for item in resolve_tool_surface(
                surface["profile"], component_overrides=surface["component_overrides"],
            ).definitions
        ]
        if policy["tool_allowlist"] != names:
            raise ValueError("policy.baseline requires its complete ordered tool_allowlist")
    _fill(policy, catalog["policy"])
    policy["tool_surface"] = surface
    template = dict(instance["runtime"])
    if runtime.get("kind") == "robodojo" and runtime.get("task_variant") == "swap_T":
        from ora.environment.benchmarks.robodojo_asset_revision import (
            asset_revision_for, SWAP_T_ASSET_ROOT_ENV,
        )
        # A reviewed task owns its exact asset pin and asset-root binding;
        # this is not a general permission to override baseline revisions.
        template.update(required_asset_revision=asset_revision_for("swap_T"),
                        asset_root_env=SWAP_T_ASSET_ROOT_ENV)
    task_config = runtime.get("task_config")
    if (runtime.get("kind") == "robotwin"
            and (requested is None or requested in SCENE_CONFIGURATION_CHOICE_BASELINES)
            and isinstance(task_config, str) and task_config in TASK_CONFIGS):
        # Both official scene configurations evaluate the same composition;
        # the instance names the default one and the RunSpec may select the
        # other. A frozen baseline keeps its exact prior contract, which for
        # the ones listed below included this choice.
        template.pop("task_config", None)
    _check_runtime_contract(runtime, template)
    _fill(runtime, template)
    # Resolve metadata from the actual controller/cameras, never from a
    # caller-supplied public-context assertion.
    unasserted_runtime = {
        key: value for key, value in runtime.items()
        if key != "public_environment_context"
    }
    descriptor = resolve_runtime_descriptor(
        unasserted_runtime, action_codec=policy["action_codec"],
    )
    public = {
        "schema": "ora.public-environment-context.v1",
        "environment_id": descriptor.environment_id,
        "embodiment_id": descriptor.embodiment_id,
        "robot_models": list(descriptor.robot_models),
        "controller": descriptor.controller,
        "public_cameras": list(descriptor.public_cameras),
    }
    binding = resolve_environment_prompt_profile(
        public, project_root=root, profile_id=instance["environment_prompt_profile"],
    )
    prompt_binding = {k: binding[k] for k in (
        "schema", "profile_id", "profile_path", "profile_sha256", "metadata_sha256",
    )}
    for key, expected in (
        ("public_environment_context", public), ("environment_prompt_profile", prompt_binding),
    ):
        if key in runtime and runtime[key] != expected:
            raise ValueError(f"policy.baseline conflicts with runtime.{key}")
        runtime[key] = expected
    bundle_path = root / bundle_ref
    bundle_raw = bundle_path.read_bytes()
    layers = yaml.safe_load(bundle_raw)["layers"]
    # The episode-task template already tells the policy to treat a separately
    # advertised completion contract as the objective. Advertise it here, or
    # record why it was withheld: an undelivered contract that leaves no trace
    # is indistinguishable from a task that never had one.
    contract_layer, contract_disclosure = resolve_task_contract_layer(runtime, root=root)
    result["policy_context"] = {
        "schema": "ora.policy-context.v3",
        "harness_system": list(layers["planner"]["sources"]),
        "project_instructions": [
            *layers["core"]["sources"],
            binding["environment_policy"], binding["embodiment_policy"],
        ],
        "episode_task": [
            layers["episode_task"]["source"],
            *([contract_layer] if contract_layer is not None else []),
        ],
        "tool_manual": list(layers["tool_manual"]["sources"]),
    }
    policy["task_contract"] = contract_disclosure
    scope = catalog["required_service_capabilities"]
    if "prerequisite" not in result:
        result["prerequisite"] = {
            "schema": "ora.model-service-set-binding.v2",
            "capability_scope": list(scope), "status": "unbound_preflight_required",
        }
    prerequisite = result["prerequisite"]
    if not isinstance(prerequisite, Mapping) or (
        prerequisite.get("schema") != "ora.model-service-set-binding.v2"
        or prerequisite.get("capability_scope") != scope
    ):
        raise ValueError("policy.baseline requires its complete resident-service prerequisite")
    if "services" in prerequisite and (
        set(prerequisite) != {"schema", "capability_scope", "services"}
        or not isinstance(prerequisite["services"], Mapping)
        or set(prerequisite["services"]) != {"segmentation", "grasp_proposals"}
        or not all(
            isinstance(child, Mapping) for child in prerequisite["services"].values()
        )
    ):
        raise ValueError(
            "policy.baseline resident-service prerequisite has an invalid service set"
        )
    policy["baseline"] = {
        "id": catalog["id"], "instance": instance["id"],
        "selection": "explicit" if requested is not None else "omitted_tool_surface",
        "path": catalog["path"], "sha256": catalog["sha256"],
        "prompt_bundle": {
            "path": bundle_ref,
            "sha256": hashlib.sha256(bundle_raw).hexdigest(),
        },
    }
    return result
