"""Trusted episode coordinator for every registered environment session.

`run_episode` is the entry point. It binds one RunPlan to an
EnvironmentSession, a composed ToolRuntime and a started PolicySession
(`_bind_sessions`), then runs `EpisodeLoop` from `episode_loop.py`, which owns
the observe -> policy turn -> execute loop, round boundaries and the result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ora.agent.composition import materialize_policy
from ora.agent.context import ContextProjector
from ora.agent.evidence.policy_state import PolicyStateStore
from ora.agent.memory.artifacts import build_recoverable_artifact_runtime
from ora.agent.tools import build_tool_runtime
from ora.environment import (
    EnvironmentSession,
    build_environment_session,
    resolve_runtime_descriptor,
    validate_restartable_environment_session,
)
from ora.orchestrator.episode_bindings import (
    SupervisorError, _qualified_runtime_binding,
)

from .episode_loop import EpisodeLoop, EpisodeSessions
from .episode_records import _atomic_json
from .live import LiveInteractionPort
from .policy_factory import (
    build_policy_session,
    ensure_policy_session,
    policy_session_hints,
)
from .spec import PROJECT_ROOT

# Names this module exposed before the episode loop moved to the
# `episode_*` modules; callers and tests still import them from here.
import copy  # noqa: F401
import json  # noqa: F401
import os  # noqa: F401
import shutil  # noqa: F401
import signal  # noqa: F401
import threading  # noqa: F401
import time  # noqa: F401
from dataclasses import replace  # noqa: F401
from typing import Sequence  # noqa: F401

from ora.agent import (  # noqa: F401
    ActionProposal,
    PolicyRuntimeEvent,
    PolicyTurnInput,
)
from ora.agent.context.management import LOSSLESS_CONTEXT_MANAGEMENT_SCHEMAS  # noqa: F401
from ora.contracts import (  # noqa: F401
    SKILL_SEQUENCE_ACTION_SCHEMA,
)
from ora.environment import validate_environment_status  # noqa: F401
from ora.orchestrator.episode_bindings import (  # noqa: F401
    ROUND_COMPARABILITY_FIELDS, _round_comparability, robodojo_official_complete,
)

from .action_executor import execute_runtime_action  # noqa: F401
from .context_manifest import build_context_manifest, context_placeholder  # noqa: F401
from .decision_accounting import is_zero_write_refusal  # noqa: F401
from .episode_observation import _session_round_budget_fields  # noqa: F401
from .episode_records import _append, _status  # noqa: F401
from .episode_result import _quiesce_policy_after_fault  # noqa: F401
from .faults import classify_harness_fault  # noqa: F401
from .interaction import EpisodeInteractionState, InteractionStateError  # noqa: F401
from .live import HarnessLiveEvent, NullLiveInteractionPort  # noqa: F401
from .operator_stop import act_unless_stopped  # noqa: F401
from .robodojo_round_scene import (  # noqa: F401
    layout_change_sentence, require_round_scene, round_scene_field,
)
from .round_aggregates import run_round_aggregates  # noqa: F401
from .round_isolation import RoundIsolationError, require_round_isolation  # noqa: F401
from .run_artifacts import (  # noqa: F401  re-exported for the Runtime v1 tests
    _main_observation_image, _reconciled_provider_model_calls,
)
from .skill_sequence_job import run_skill_sequence  # noqa: F401
from .stagnation import StagnationGuard  # noqa: F401
from .termination import (  # noqa: F401
    TerminationAuthority,
    TerminationDecision,
    budget_boundary,
    environment_boundary,
    infrastructure_boundary,
    operator_boundary,
    policy_outcome_decision,
)
from .verdict_guard import OFFICIAL_VERDICT_CHECKER_DIVERGENCE, RoundVerdictGuard  # noqa: F401
from .wait_accounting import RoundActionLedger  # noqa: F401


class EpisodeCoordinator:
    """Reusable façade over the behavior-preserving episode function.

    Dependency injection lives here so callers do not need to import a concrete
    Codex, Native, Scripted, or environment implementation.
    """

    def __init__(
        self,
        *,
        policy_session_factory: Any = build_policy_session,
        environment_session_factory: Any = build_environment_session,
        live_port: LiveInteractionPort | None = None,
    ) -> None:
        self._policy_session_factory = policy_session_factory
        self._environment_session_factory = environment_session_factory
        self._live_port = live_port

    def run(
        self,
        *,
        run_root: Path,
        plan: Mapping[str, Any],
        run_id: str,
        provider_override: Any = None,
    ) -> dict[str, Any]:
        return run_episode(
            run_root=run_root,
            plan=plan,
            run_id=run_id,
            provider_override=provider_override,
            policy_session_factory=self._policy_session_factory,
            environment_session_factory=self._environment_session_factory,
            live_port=self._live_port,
        )


#: Comparability fields the environment publishes on its own verdict and the
#: round record mirrors verbatim. They are never invented here: a backend whose
#: protocol has no such notion simply publishes none, and the round record then
#: carries nulls, which every reporting layer must treat as "not comparable"
#: rather than as an omission it may ignore.


def run_episode(
    *, run_root: Path, plan: Mapping[str, Any], run_id: str,
    provider_override: Any = None,
    policy_session_factory: Any = build_policy_session,
    environment_session_factory: Any = build_environment_session,
    live_port: LiveInteractionPort | None = None,
) -> dict[str, Any]:
    """Run one RunPlan episode (or multi-round session) and return its result.

    A failure while binding the sessions propagates to the caller; once the
    loop runs, every failure becomes a trusted fault result instead.
    """

    episode = EpisodeLoop(
        run_root=run_root, plan=plan, run_id=run_id, live_port=live_port,
    )
    sessions = _bind_sessions(
        episode,
        provider_override=provider_override,
        policy_session_factory=policy_session_factory,
        environment_session_factory=environment_session_factory,
    )
    return episode.run(sessions)


def _bind_sessions(
    episode: EpisodeLoop, *, provider_override: Any,
    policy_session_factory: Any, environment_session_factory: Any,
) -> EpisodeSessions:
    """Construct the environment, compose the policy surface, start the policy."""

    plan = episode.plan
    runtime_config = plan["runtime"]
    qualified_runtime_binding = _qualified_runtime_binding(
        episode.run_root, runtime_config,
    )
    session_hints = policy_session_hints(plan["policy"])
    runtime, runtime_descriptor = _open_environment(
        episode, runtime_config,
        session_hints=session_hints,
        qualified_runtime_binding=qualified_runtime_binding,
        environment_session_factory=environment_session_factory,
    )
    compiled_policy = materialize_policy(
        plan["policy_context"], project_root=PROJECT_ROOT,
        destination=episode.run_root / "compiled-policy",
    )
    prompt_path = compiled_policy["episode_prefix"]
    manual_path = compiled_policy["tool_manual"]
    policy = plan["policy"]
    allowed_tools = policy.get("tool_allowlist", ["robot.submit_action"])
    state_store = PolicyStateStore(
        run_root=episode.run_root, run_id=episode.run_id, plan_sha256=plan["plan_sha256"]
    )
    projector = ContextProjector(
        workspace=episode.workspace, control_root=episode.control,
        prompt_path=prompt_path,
        manual_path=manual_path,
        project_instructions_path=compiled_policy["codex_project"],
        expose_manual="robot.read_manual" in allowed_tools,
        main_camera=str(runtime_config.get("main_camera", "head_camera")),
        run_root=episode.run_root,
        camera_roles=runtime_descriptor.get("camera_roles") or {},
        view_config=plan.get("policy_composition", {}).get("tool_surface", {}).get("view_config"),
        capabilities=tuple(plan.get("policy_composition", {}).get("tool_surface", {}).get("capabilities") or ()),
    )
    tools = _compose_tool_runtime(
        episode, runtime, runtime_config,
        manual_path=manual_path,
        allowed_tools=allowed_tools, state_store=state_store,
    )
    planner = policy_session_factory(
        plan=plan,
        run_root=episode.run_root,
        run_id=episode.run_id,
        workspace=episode.workspace,
        control_root=episode.control,
        events_path=episode.events,
        compiled_policy=compiled_policy,
        tools=tools,
        state_root=state_store.root,
        record_context_bundle=state_store.register_context_bundle,
        provider_override=provider_override,
    )
    planner = ensure_policy_session(
        planner,
        kind=session_hints["session_kind"],
        supports_images=session_hints["supports_images"],
    )
    planner.start()
    return EpisodeSessions(
        runtime_config=runtime_config, runtime=runtime,
        runtime_descriptor=runtime_descriptor, compiled_policy=compiled_policy,
        state_store=state_store, projector=projector, tools=tools,
        planner=planner,
    )


def _open_environment(
    episode: EpisodeLoop, runtime_config: Mapping[str, Any], *,
    session_hints: Mapping[str, Any], qualified_runtime_binding: Any,
    environment_session_factory: Any,
) -> tuple[EnvironmentSession, dict[str, Any]]:
    """Construct the EnvironmentSession and prove it matches the RunPlan."""

    plan = episode.plan
    expected_environment_descriptor = resolve_runtime_descriptor(
        runtime_config,
        action_codec=plan["policy"].get("action_codec"),
    )
    runtime: EnvironmentSession = environment_session_factory(
        runtime_config=runtime_config,
        output_root=episode.native,
        task=plan["task"],
        seed=plan["seed"],
        max_sim_steps=plan["budgets"]["max_sim_steps"],
        redact_environment_names=session_hints["redacted_environment_names"],
        action_codec=plan["policy"].get("action_codec"),
        qualified_runtime_binding=qualified_runtime_binding,
    )
    if episode.multi_round:
        runtime = validate_restartable_environment_session(runtime)
    describe = getattr(runtime, "describe", None)
    runtime_descriptor = (
        dict(describe()) if callable(describe)
        else expected_environment_descriptor.as_dict()
    )
    if runtime_descriptor != expected_environment_descriptor.as_dict():
        raise SupervisorError("environment runtime descriptor differs from RunPlan")
    _atomic_json(episode.native / "environment-runtime-descriptor.json", runtime_descriptor)
    return runtime, runtime_descriptor


def _compose_tool_runtime(
    episode: EpisodeLoop, runtime: EnvironmentSession,
    runtime_config: Mapping[str, Any], *, manual_path: Any,
    allowed_tools: Any, state_store: PolicyStateStore,
) -> Any:
    """Build the policy ToolRuntime with its Artifact reader, as the RunPlan composed it."""

    plan = episode.plan
    policy = plan["policy"]
    tools = build_tool_runtime(
        control_root=episode.control, manual_path=manual_path, event_path=episode.events,
        workspace_bounds_m=runtime_config["workspace_bounds_m"],
        allowed_tools=allowed_tools,
        state_root=state_store.root,
        profile_id=policy.get("tool_surface", {}).get(
            "profile", "robot_runtime_v1"
        ),
        component_overrides=policy.get("tool_surface", {}).get(
            "component_overrides", {}
        ),
        prerequisite_binding=(
            plan.get("prerequisite")
            if isinstance(plan.get("prerequisite"), Mapping) else None
        ),
        controlled_end_effector=runtime_config.get("controlled_end_effector"),
        motion_probe_callback=(
            getattr(runtime, "probe_motion", None)
            if callable(getattr(runtime, "probe_motion", None)) else None
        ),
        align_plan_callback=(
            getattr(runtime, "align_plan", None)
            if callable(getattr(runtime, "align_plan", None)) else None
        ),
    )
    tools = build_recoverable_artifact_runtime(
        plan, run_root=episode.run_root, event_path=episode.events, base_runtime=tools,
    )
    planned_tool_definitions = plan.get("policy_composition", {}).get(
        "tool_surface", {}
    ).get("definitions")
    if (
        isinstance(planned_tool_definitions, list)
        and tools.definitions() != planned_tool_definitions
    ):
        raise SupervisorError("Runtime tool definitions differ from the RunPlan composition")
    return tools
