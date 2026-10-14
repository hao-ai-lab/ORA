"""Host-safe local CPU composition; no simulator, Provider or service startup.

The local ORA consumer selects this composition explicitly. The shared server
baseline remains unchanged. Task text, pinned source inputs, Provider policy
and execution host are caller-owned inputs, checked again by normal preflight.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Mapping

from .baseline import load_native_baseline
from .environment_context import resolve_environment_prompt_profile
from .prompt_bundle import load_prompt_bundle


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOCAL_COMPOSITION = {
    "id": "ora_local_cpu_v1",
    "tool_surface": "local_atomic_cpu_v1",
    "prompt_bundle": "ora/policy/prompt_bundles/local_cpu_v1/libero-panda-native-api.yaml",
    "environment_prompt_profile": "libero_pro_local_cpu_v1",
    "action_codec": "runtime_action_v3",
    "environment_driver": "direct_libero_v1",
    "required_host": "local-cpu",
    "require_gpu": False,
}
SOURCE_FIELDS = (
    "required_libero_pro_revision", "required_data_revision", "python_env",
    "libero_pro_root_env", "data_root_env",
)
BASE_SUITES = ("libero_object", "libero_goal", "libero_spatial", "libero_10")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CONTEXT_FIELDS = (
    "context_profile", "context_composition", "context_retention", "summary_policy",
    "memory_policy", "provider_continuation_replay", "initial_observation", "timeout_s",
    "max_model_calls_per_observation", "context_management", "require_live_marker", "provider_recovery",
)


def compose_local_spec(*, task: Mapping[str, Any], source: Mapping[str, Any],
                       provider: Mapping[str, Any], seed: int = 0,
                       resources: Mapping[str, Any] | None = None,
                       root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Compose one source-bound original LIBERO task for bounded CPU OSC.

    Asset reads are the only I/O. Resolution does not consume launch admission
    or claim platform/motion qualification. The public local command supplies
    the official task text and any reviewed BDDL wording acknowledgment.
    """
    if not all(isinstance(value, Mapping) for value in (task, source, provider)):
        raise ValueError("task, source and provider must be mappings")
    suite, task_id = task.get("suite"), task.get("task_id")
    if suite not in BASE_SUITES or isinstance(task_id, bool) or not isinstance(task_id, int) or not 0 <= task_id < 10:
        raise ValueError("local composition requires an original LIBERO suite and task_id in [0,10)")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 50:
        raise ValueError("local LIBERO initial state must be in [0,50)")
    language = task.get("task_language")
    if not isinstance(language, str) or not language.strip():
        raise ValueError("local composition requires source-bound task_language")
    if set(source) != set(SOURCE_FIELDS):
        raise ValueError(f"local source inputs must be exactly {list(SOURCE_FIELDS)}")
    for name in SOURCE_FIELDS:
        value = source[name]
        pattern = r"[0-9a-f]{40}" if name.startswith("required_") else _ENV_NAME.pattern
        if not isinstance(value, str) or re.fullmatch(pattern, value) is None:
            raise ValueError(f"invalid local source input: {name}")
    maintained = load_native_baseline(root)
    source_contract = maintained["instances"]["libero_pro"]["runtime"]
    for name in SOURCE_FIELDS[:2]:
        if source[name] != source_contract[name]:
            raise ValueError(f"local CPU bundle has not been reviewed for another {name}")
    fixed_policy = {"tool_surface", "baseline", "action_codec", *_CONTEXT_FIELDS}
    if set(provider) & fixed_policy or provider.get("kind") != "native_api" or provider.get("supports_images") is not True:
        raise ValueError("local composition requires an image-capable native Provider policy without composition overrides")
    placement = dict(resources or {})
    if set(placement) - {"required_host"}:
        raise ValueError("local CPU resources accept only required_host; no GPU or simulator-slot binding")
    host = placement.get("required_host", DEFAULT_LOCAL_COMPOSITION["required_host"])
    if host != DEFAULT_LOCAL_COMPOSITION["required_host"]:
        raise ValueError("local CPU composition requires the registered local-cpu host")

    bundle = load_prompt_bundle(root / DEFAULT_LOCAL_COMPOSITION["prompt_bundle"])
    layers = bundle["layers"]
    public = copy.deepcopy(bundle["bindings"]["environment_prompt_profile"]["public_environment_context"])
    binding = resolve_environment_prompt_profile(
        public, project_root=root, profile_id=DEFAULT_LOCAL_COMPOSITION["environment_prompt_profile"],
    )
    # Share the maintained Context defaults, not its GPU tools, runtime,
    # embodiment or service prerequisites. Every resulting value is explicit
    # in the local RunSpec and its content-addressed RunPlan.
    shared = maintained["policy"]
    policy = {**copy.deepcopy(dict(provider)),
              **{key: copy.deepcopy(shared[key]) for key in _CONTEXT_FIELDS},
              "action_codec": DEFAULT_LOCAL_COMPOSITION["action_codec"],
              "tool_surface": {"profile": DEFAULT_LOCAL_COMPOSITION["tool_surface"], "component_overrides": {}}}
    runtime = {
        "kind": "libero_pro", "transport": "local_rpc", **dict(source),
        "environment_driver": DEFAULT_LOCAL_COMPOSITION["environment_driver"],
        "required_host": host, "require_gpu": False,
        "suite": suite, "task_id": task_id, "expected_task_language": language,
        "main_camera": "agentview", "policy_cameras": ["agentview", "wrist"],
        "controlled_end_effector": "right", "startup_timeout_s": 600,
        "request_timeout_s": 1800, "servo_max_steps": 80,
        "workspace_profile": "libero_pro_panda_tabletop_v1",
        "public_environment_context": public,
        "environment_prompt_profile": {key: binding[key] for key in (
            "schema", "profile_id", "profile_path", "profile_sha256", "metadata_sha256",
        )},
    }
    bddl_language = task.get("bddl_task_language", language)
    if bddl_language != language:
        ack = task.get("task_language_divergence_ack")
        if (not isinstance(ack, Mapping) or set(ack) != {"bddl_task_language", "policy_task_language", "review"}
                or ack["bddl_task_language"] != bddl_language or ack["policy_task_language"] != language
                or not isinstance(ack["review"], str) or not ack["review"].strip()):
            raise ValueError("differing official and BDDL task wording requires its exact reviewed acknowledgment")
        runtime["task_language_divergence_ack"] = copy.deepcopy(dict(ack))
    label = f"{suite}-t{task_id:02d}"
    return {
        "schema": "ora.run-spec.v1",
        "experiment": "ora-interactive", "output_workspace": "outputs/ora-interactive",
        "config": f"local-{label}-s{seed}", "milestone": "LIBERO-LOCAL-CPU",
        "task": label, "task_instruction": language, "seed": seed, "evidence_tier": "behavioral",
        "policy": policy, "runtime": runtime,
        "policy_context": {
            "schema": "ora.policy-context.v3",
            "harness_system": list(layers["planner"]["sources"]),
            "project_instructions": [*layers["core"]["sources"], binding["environment_policy"], binding["embodiment_policy"]],
            "episode_task": [layers["episode_task"]["source"]],
            "tool_manual": list(layers["tool_manual"]["sources"]),
        },
        "budgets": {"max_robot_decisions": 80, "max_model_calls": 200, "max_sim_steps": 10000,
                    "wall_time_s": 21600},
        "qualification": {"required": False},
    }


__all__ = ["DEFAULT_LOCAL_COMPOSITION", "SOURCE_FIELDS", "compose_local_spec"]
