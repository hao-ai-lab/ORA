"""Runtime, budget and session sections of a RunSpec.

``resolve_runtime`` checks the fields every environment shares (resource
slot, transport, execution capture, action workspace and public
geometry), then the selected environment's own fields through
``RUNTIME_KIND_RESOLVERS``. Several ToolSurface bindings depend on the
runtime, so it returns the ToolSurface those bindings produced alongside the
runtime. ``resolve_budgets`` and ``resolve_session`` follow, in that order.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Mapping

from ora.agent.composition import ResolvedToolSurface
from ora.agent.composition.reach_reference_schema import INTER_ARM_SCREEN_CAPABILITY
from ora.agent.native.constants import NATIVE_CAUSAL_SLOTS
from ora.contracts.workspace import WorkspaceError, normalize_workspace_bounds
from ora.environment.controllers.profile_capabilities import (
    OLDEST_COMPOSED_CONTROL_PROFILE, bound_control_profile_version,
)
from ora.environment.definitions import (
    SPARSE_MOTION_QUALIFIED_ENVIRONMENTS,
    sparse_motion_physics_refusal,
)
from ora.environment.workspace_profiles import resolve_workspace_profile

from .resources import ExecutionResourceError, resolve_execution_resource
from .spec_values import RunSpecError, _mapping, _positive_int

RUNTIME_KINDS = {"synthetic", "robotwin", "libero_pro", "robodojo"}
ENVIRONMENT_TRANSPORTS = {"in_process_direct", "local_rpc"}
ENVIRONMENT_SIGNAL_POLICIES = {"lifecycle_only", "task_recoverability_v1"}
SESSION_CADENCES = {"persistent_agent_k_rounds_v1"}


def resolve_runtime(
    spec: Mapping[str, Any], policy: dict[str, Any],
    resolved_surface: ResolvedToolSurface, *, task: str, seed: int,
    task_instruction: str, root: Path,
) -> tuple[dict[str, Any], ResolvedToolSurface]:
    """Return the runtime copy and the ToolSurface bound to it."""

    runtime = dict(_mapping(spec.get("runtime"), "runtime"))
    if runtime.get("kind") not in RUNTIME_KINDS:
        raise RunSpecError("runtime.kind must be synthetic, robotwin, libero_pro or robodojo")
    _bind_runtime_profile_paths(runtime, root=root)
    _require_sparse_motion_qualification(policy, runtime)
    _bind_runtime_policy_contracts(spec, policy, runtime, root=root)
    _bind_execution_resource(runtime)
    _resolve_transport_and_signals(runtime, task=task)
    raw_workspace_profile = runtime.get("workspace_profile")
    _bind_execution_capture(runtime)
    _resolve_workspace_bounds(runtime, raw_workspace_profile)
    resolved_surface = _bind_public_geometry(runtime, policy, resolved_surface)
    RUNTIME_KIND_RESOLVERS[runtime["kind"]](
        runtime, policy, resolved_surface,
        task=task, seed=seed, task_instruction=task_instruction,
    )
    _validate_runtime_extension(runtime)
    _require_surface_runtime(policy, runtime, resolved_surface)
    return runtime, resolved_surface


def _bind_runtime_profile_paths(runtime: dict[str, Any], *, root: Path) -> None:
    """A motion or control profile given by path becomes its ``{path, sha256}`` record.

    The record hashes the bytes read now. A RunSpec may instead carry the
    record form a RunPlan holds; the factory and preflight verify it.
    """

    from ora.agent.composition.baseline import RUNTIME_PROFILE_FIELDS
    from ora.agent.composition.locked_assets import locked_asset

    for field in RUNTIME_PROFILE_FIELDS:
        if isinstance(runtime.get(field), str):
            runtime[field] = locked_asset(
                root, runtime[field], label=f"runtime.{field}", error=RunSpecError,
            ).record()


def _require_sparse_motion_qualification(
    policy: Mapping[str, Any], runtime: Mapping[str, Any],
) -> None:
    if policy["action_codec"] in SPARSE_MOTION_QUALIFIED_ENVIRONMENTS and (
        runtime.get("kind")
        not in SPARSE_MOTION_QUALIFIED_ENVIRONMENTS[policy["action_codec"]]
    ):
        raise RunSpecError(
            f"sparse motion codec {policy['action_codec']} has no "
            "source-bound motion qualification for "
            f"{runtime.get('kind')}; qualified: "
            + ", ".join(
                SPARSE_MOTION_QUALIFIED_ENVIRONMENTS[policy["action_codec"]]
            )
        )
    # A row's scripted record also binds the physics device it ran on
    # (definitions.SPARSE_MOTION_QUALIFIED_PHYSICS_DEVICES).
    physics_refusal = sparse_motion_physics_refusal(policy["action_codec"], runtime)
    if physics_refusal is not None:
        raise RunSpecError(physics_refusal)


def _bind_runtime_policy_contracts(
    spec: Mapping[str, Any], policy: dict[str, Any], runtime: dict[str, Any],
    *, root: Path,
) -> None:
    """Bind the policy fields that need the runtime: the task contract."""

    if "semantic_manifest" in policy:
        raise RunSpecError("policy.semantic_manifest is not supported")
    # A deliverable contract that is not delivered, and leaves no disclosure,
    # reads exactly like a task that never had one. Asserted on the one path
    # every RunSpec takes rather than in the package that happened to omit it.
    from ora.agent.composition.task_contract import (
        TaskContractError, bind_task_contract,
    )

    try:
        bind_task_contract(spec, policy, runtime, root=root)
    except TaskContractError as exc:
        raise RunSpecError(str(exc)) from exc


def _bind_execution_resource(runtime: dict[str, Any]) -> None:
    """Replace the selected resource profile with its resolved slot binding."""

    if "resource_binding" in runtime:
        raise RunSpecError(
            "runtime.resource_binding is resolver-owned; select resource_profile"
        )
    if "resource_concurrency_override" in runtime:
        raise RunSpecError(
            "runtime.resource_concurrency_override is not an operator authority; "
            "use distinct source-tracked resource profiles or wait for the "
            "exclusive lease"
        )
    resource_profile = runtime.pop("resource_profile", None)
    if resource_profile is not None:
        try:
            runtime["resource_binding"] = resolve_execution_resource(
                resource_profile,
                runtime_kind=str(runtime["kind"]),
                required_host=(
                    str(runtime["required_host"])
                    if isinstance(runtime.get("required_host"), str)
                    else None
                ),
            )
        except ExecutionResourceError as exc:
            raise RunSpecError(str(exc)) from exc


def _resolve_transport_and_signals(runtime: dict[str, Any], *, task: str) -> None:
    runtime.setdefault("transport", "in_process_direct")
    if runtime["transport"] not in ENVIRONMENT_TRANSPORTS:
        raise RunSpecError(
            f"runtime.transport must be one of {sorted(ENVIRONMENT_TRANSPORTS)}"
        )
    runtime.setdefault("environment_signal_policy", "lifecycle_only")
    if runtime["environment_signal_policy"] not in ENVIRONMENT_SIGNAL_POLICIES:
        raise RunSpecError(
            "runtime.environment_signal_policy must be one of "
            f"{sorted(ENVIRONMENT_SIGNAL_POLICIES)}"
        )
    if (
        runtime["environment_signal_policy"] != "lifecycle_only"
        and runtime.get("kind") != "robotwin"
    ):
        raise RunSpecError(
            "task_recoverability_v1 is currently supported only by robotwin"
        )
    if (
        runtime["environment_signal_policy"] == "task_recoverability_v1"
        and task != "adjust_bottle"
    ):
        raise RunSpecError(
            "task_recoverability_v1 is currently defined only for adjust_bottle"
        )


def _bind_execution_capture(runtime: dict[str, Any]) -> None:
    """Resolve execution capture; refuse the settings of the retired Servo tool."""

    from ora.environment.execution_capture import resolve_execution_capture
    retired = sorted(key for key in ("servo_profile", "episode_physics_budget", "servo_worker_abi")
                     if key in runtime)
    if retired:
        raise RunSpecError(f"bounded Servo is not part of this tree; remove runtime.{', runtime.'.join(retired)}")
    try:
        if runtime["kind"] in {"robotwin", "libero_pro", "robodojo"}:
            runtime["execution_capture"] = resolve_execution_capture(runtime.get("execution_capture"))
    except ValueError as exc:
        raise RunSpecError(str(exc)) from exc


def _resolve_workspace_bounds(
    runtime: dict[str, Any], raw_workspace_profile: Any,
) -> None:
    """Bind a named workspace profile, or normalize explicit bounds."""

    if raw_workspace_profile is not None:
        if "workspace_bounds_m" in runtime:
            raise RunSpecError(
                "runtime must select workspace_profile or workspace_bounds_m, not both"
            )
        if not isinstance(raw_workspace_profile, str) or not raw_workspace_profile:
            raise RunSpecError("runtime.workspace_profile must be a stable profile id")
        try:
            workspace_profile = resolve_workspace_profile(raw_workspace_profile)
        except ValueError as exc:
            raise RunSpecError(str(exc)) from exc
        if workspace_profile["environment_id"] != runtime["kind"]:
            raise RunSpecError("action workspace profile does not match runtime.kind")
        runtime["workspace_profile"] = workspace_profile
        runtime["workspace_bounds_m"] = workspace_profile["bounds_m"]
    else:
        bounds = _mapping(
            runtime.get("workspace_bounds_m"), "runtime.workspace_bounds_m"
        )
        try:
            runtime["workspace_bounds_m"] = normalize_workspace_bounds(bounds)
        except WorkspaceError as exc:
            raise RunSpecError(str(exc)) from exc


def _bind_public_geometry(
    runtime: dict[str, Any], policy: dict[str, Any],
    resolved_surface: ResolvedToolSurface,
) -> ResolvedToolSurface:
    """Publish the workspace bounds and any public geometry descriptor."""

    from ora.agent.composition.public_parameters import bind_public_workspace
    resolved_surface = bind_public_workspace(resolved_surface, runtime["workspace_bounds_m"])
    policy["tool_surface"]["composition_sha256"] = resolved_surface.composition_sha256
    if resolved_surface.supports("current_public_robot_facts_v1"):
        geometry_descriptor = resolved_surface.view_config["public_geometry_descriptor"]
        if "public_geometry_descriptor" in runtime and runtime["public_geometry_descriptor"] != geometry_descriptor:
            raise RunSpecError("runtime public geometry must match the selected ToolSurface descriptor")
        runtime["public_geometry_descriptor"] = copy.deepcopy(geometry_descriptor)
    elif "public_geometry_descriptor" in runtime:
        raise RunSpecError("public geometry descriptor requires its complete delivery surface")
    return resolved_surface


def _resolve_robotwin_runtime(
    runtime: dict[str, Any], policy: Mapping[str, Any],
    resolved_surface: ResolvedToolSurface, *, task: str, seed: int,
    task_instruction: str,
) -> None:
    if runtime["transport"] != "local_rpc":
        raise RunSpecError("robotwin runtime.transport must be local_rpc")
    for field in ("root_env", "python_env", "required_revision", "task_config"):
        if not isinstance(runtime.get(field), str) or not runtime[field]:
            raise RunSpecError(f"runtime.{field} must be non-empty")
    runtime.setdefault("startup_timeout_s", 600)
    runtime.setdefault("request_timeout_s", 1800)
    for field in ("startup_timeout_s", "request_timeout_s"):
        runtime[field] = _positive_int(runtime[field], f"runtime.{field}")
    if policy["action_codec"] == "motion_action_v3":
        _require_sparse_motion_topology(runtime)
    elif "control_profile" in runtime:
        raise RunSpecError(
            "runtime.control_profile is valid only with a sparse motion action"
        )
    _bind_reach_check_inter_arm_screen(runtime, policy, resolved_surface)


#: RoboTwin runtime field -> the ToolSurface capability it is derived from
#: (the reach check's other-arm screen).
REACH_SCREEN_FIELDS = {
    "reach_check_inter_arm_screen": INTER_ARM_SCREEN_CAPABILITY,
}


def _bind_reach_check_inter_arm_screen(
    runtime: dict[str, Any], policy: Mapping[str, Any],
    resolved_surface: ResolvedToolSurface,
) -> None:
    """Set the screen field exactly when the bound surface declares its capability.

    A surface without the capability adds no key; a field set by hand must
    agree with the surface.
    """

    declared = resolved_surface.supports(INTER_ARM_SCREEN_CAPABILITY)
    for field, capability in REACH_SCREEN_FIELDS.items():
        if field in runtime and (runtime[field] is not True or not declared):
            raise RunSpecError(
                f"runtime.{field} disagrees with the ToolSurface: it is derived from "
                f"{capability}, and is set (true) exactly when the surface declares it"
            )
        if declared:
            runtime[field] = True
    if declared and policy["action_codec"] != "motion_action_v3":
        raise RunSpecError(
            "the reach-check inter-arm screen requires a sparse motion action "
            "(runtime.control_profile carries the dual_arm_collision_check it uses)"
        )


def _require_sparse_motion_topology(runtime: Mapping[str, Any]) -> None:
    """Sparse motion binds a control profile and the qualified cameras."""

    control_profile = runtime.get("control_profile")
    if not isinstance(control_profile, Mapping) or set(control_profile) != {
        "path", "sha256",
    }:
        raise RunSpecError(
            "sparse motion action requires runtime.control_profile path and sha256"
        )
    if (
        not isinstance(control_profile.get("path"), str)
        or not control_profile["path"]
        or not isinstance(control_profile.get("sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", control_profile["sha256"]) is None
    ):
        raise RunSpecError("runtime.control_profile binding is invalid")
    version = bound_control_profile_version(control_profile)
    if version is not None and version < OLDEST_COMPOSED_CONTROL_PROFILE:
        raise RunSpecError(
            f"runtime.control_profile v{version} is older than the oldest control "
            f"profile this tree composes (v{OLDEST_COMPOSED_CONTROL_PROFILE}); a run "
            "bound to it composes and evaluates only at the commits that composed it"
        )
    if runtime.get("policy_observer_camera", False) is not False:
        raise RunSpecError("runtime.policy_observer_camera is not supported")
    expected_cameras = ["head_camera", "front_camera", "left_camera", "right_camera"]
    if runtime.get("policy_cameras") != expected_cameras:
        raise RunSpecError(
            "sparse motion policy cameras must match the qualified RoboTwin order"
        )
    if runtime.get("main_camera") != "head_camera":
        raise RunSpecError("sparse motion main_camera must be head_camera")


def _resolve_robodojo_runtime(
    runtime: dict[str, Any], policy: Mapping[str, Any],
    resolved_surface: ResolvedToolSurface, *, task: str, seed: int,
    task_instruction: str,
) -> None:
    # The unified RoboDojo instance drives the sparse v3 motion codec; the
    # registry's SPARSE_MOTION_QUALIFIED_ENVIRONMENTS row (bound to the
    # scripted RoboDojo motion-qualification record) is what actually
    # admits it, so this branch only excludes the codecs the backend has
    # no executor for.
    if policy["action_codec"] not in {"runtime_action_v3", "motion_action_v3"}:
        raise RunSpecError(
            "RoboDojo supports runtime_action_v3 and motion_action_v3"
        )
    if task != runtime.get("task_variant") or seed != runtime.get("eval_seed"):
        raise RunSpecError("RoboDojo task/eval_seed must match the episode identity")
    runtime.setdefault("startup_timeout_s", 600)
    runtime.setdefault("request_timeout_s", 180)
    runtime.setdefault("close_timeout_s", 30)
    if runtime.get("expected_task_language") != task_instruction:
        raise RunSpecError("RoboDojo task instruction must match official environment language")
    # The staged contact route follows the surface whose help describes it.
    from ora.environment.benchmarks.robodojo import CONTACT_APPROACH_STAGED
    route = (CONTACT_APPROACH_STAGED if resolved_surface.supports(
        "robodojo_staged_contact_approach_v1") else None)
    if runtime.get("contact_approach_route", route) != route:
        raise RunSpecError("runtime.contact_approach_route disagrees with the ToolSurface")
    if route is not None:
        runtime["contact_approach_route"] = route
    recovery = ("retract_then_unwind" if resolved_surface.supports(
        "robodojo_retract_then_unwind_recovery_v1") else None)
    if runtime.get("recovery_route", recovery) != recovery:
        raise RunSpecError("runtime.recovery_route disagrees with the ToolSurface")
    if recovery is not None:
        runtime["recovery_route"] = recovery


def _resolve_libero_pro_runtime(
    runtime: dict[str, Any], policy: Mapping[str, Any],
    resolved_surface: ResolvedToolSurface, *, task: str, seed: int,
    task_instruction: str,
) -> None:
    from .libero_configuration import validate_libero_configuration
    validate_libero_configuration(runtime)


def _resolve_synthetic_runtime(
    runtime: dict[str, Any], policy: Mapping[str, Any],
    resolved_surface: ResolvedToolSurface, *, task: str, seed: int,
    task_instruction: str,
) -> None:
    if runtime["transport"] != "in_process_direct":
        raise RunSpecError("synthetic runtime.transport must be in_process_direct")
    runtime.setdefault("terminal_actions", 1)
    runtime["terminal_actions"] = _positive_int(
        runtime["terminal_actions"], "runtime.terminal_actions"
    )


# One resolver per runtime.kind; each validates and defaults that
# environment's own fields after the shared runtime checks above.
RUNTIME_KIND_RESOLVERS = {
    "robotwin": _resolve_robotwin_runtime,
    "robodojo": _resolve_robodojo_runtime,
    "libero_pro": _resolve_libero_pro_runtime,
    "synthetic": _resolve_synthetic_runtime,
}


def _validate_runtime_extension(runtime: Mapping[str, Any]) -> None:
    from ora.environment.definitions import (
        EnvironmentDefinitionError,
        validate_runtime_extension,
    )

    try:
        validate_runtime_extension(runtime)
    except EnvironmentDefinitionError as exc:
        raise RunSpecError(str(exc)) from exc


def _require_surface_runtime(
    policy: Mapping[str, Any], runtime: Mapping[str, Any],
    resolved_surface: ResolvedToolSurface,
) -> None:
    """Runtime requirements a ToolSurface or a Native benchmark imposes."""

    if runtime.get("kind") != "robotwin":
        for field, capability in REACH_SCREEN_FIELDS.items():
            if resolved_surface.supports(capability) or field in runtime:
                raise RunSpecError(
                    f"{capability} and runtime.{field} screen RoboTwin's reach check "
                    "and require runtime.kind robotwin"
                )
    if (
        policy["kind"] == "native_api"
        and runtime["kind"] in {"libero_pro", "robodojo"}
        and policy["supports_images"] is not True
    ):
        raise RunSpecError("native robot benchmark policies require supports_images=true")


def resolve_budgets(
    spec: Mapping[str, Any], policy: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    budgets = dict(_mapping(spec.get("budgets"), "budgets"))
    for field in ("max_robot_decisions", "max_model_calls", "max_sim_steps", "wall_time_s"):
        budgets[field] = _positive_int(budgets.get(field), f"budgets.{field}")
    if runtime["kind"] == "synthetic" and runtime["terminal_actions"] > budgets["max_sim_steps"]:
        raise RunSpecError("runtime.terminal_actions must not exceed max_sim_steps")
    return budgets


def resolve_session(
    spec: Mapping[str, Any], policy: Mapping[str, Any],
    runtime: Mapping[str, Any], resolved_surface: ResolvedToolSurface,
    *, seed: int,
) -> dict[str, Any] | None:
    """The optional persistent multi-round session; ``None`` for one episode."""

    session = None
    if "session" in spec:
        session = _persistent_session(
            spec, policy, runtime, resolved_surface, seed=seed,
        )
    # Upstream's CPU reset respawns rigid objects that an in-place restart
    # leaves gravity- and collision-disabled; a CPU session's later rounds must
    # go through the scheduled relaunch (robodojo_physics_attestation).
    from .robodojo_physics_attestation import cpu_round_restart_refusal
    restart_refusal = cpu_round_restart_refusal(runtime, rounds=1 if session is None else session["rounds"])
    if restart_refusal is not None:
        raise RunSpecError(restart_refusal)
    if runtime["kind"] == "robodojo" and runtime.get("layout_schedule") is not None and (
            session is None or len(runtime["layout_schedule"]["rounds"]) != session["rounds"]):
        raise RunSpecError("runtime.layout_schedule must schedule exactly one layout per session round")
    return session


def _persistent_session(
    spec: Mapping[str, Any], policy: Mapping[str, Any],
    runtime: Mapping[str, Any], resolved_surface: ResolvedToolSurface,
    *, seed: int,
) -> dict[str, Any]:
    raw_session = dict(_mapping(spec.get("session"), "session"))
    if not {"cadence", "rounds"}.issubset(raw_session) or not set(
        raw_session
    ).issubset({"cadence", "rounds", "round_seeds"}):
        raise RunSpecError(
            "session fields must be cadence, rounds, and optional round_seeds"
        )
    if raw_session.get("cadence") not in SESSION_CADENCES:
        raise RunSpecError(
            f"session.cadence must be one of {sorted(SESSION_CADENCES)}"
        )
    rounds = _positive_int(raw_session.get("rounds"), "session.rounds")
    # A one-round session (K=1) keeps the session wrapper,
    # result schema and per-round evidence; the Supervisor never restarts
    # after the last round, so it simply ends after round 0.
    if rounds < 1 or rounds > 32:
        raise RunSpecError("session.rounds must be in [1,32]")
    if policy["kind"] != "native_api":
        raise RunSpecError(
            "persistent multi-round sessions currently require policy.kind=native_api"
        )
    # RoboDojo joined on its scripted sparse-motion qualification, whose
    # round-isolation case resets the same plant into a second round and
    # re-checks the layout identities, verdict rotation and command
    # budget.
    if runtime["kind"] not in {
        "synthetic", "libero_pro", "robotwin", "robodojo",
    }:
        raise RunSpecError(
            "persistent multi-round sessions currently support synthetic, "
            "libero_pro, robotwin and robodojo"
        )
    if policy.get("context_composition") not in NATIVE_CAUSAL_SLOTS:
        raise RunSpecError(
            "persistent multi-round sessions require native_slots_v3 or native_slots_v4"
        )
    if "finish_episode" not in resolved_surface.lifecycle_tools:
        raise RunSpecError(
            "persistent multi-round sessions require a finish_episode lifecycle tool"
        )
    round_seeds = raw_session.get("round_seeds")
    if round_seeds is not None:
        if not isinstance(round_seeds, list) or len(round_seeds) != rounds:
            raise RunSpecError(
                "session.round_seeds must contain exactly session.rounds values"
            )
        if any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or item < 0
            for item in round_seeds
        ):
            raise RunSpecError(
                "session.round_seeds values must be non-negative integers"
            )
        if round_seeds[0] != seed:
            raise RunSpecError(
                "session.round_seeds[0] must equal the top-level seed"
            )
    return {
        "cadence": raw_session["cadence"],
        "rounds": rounds,
        "budget_scope": "per_round",
        "reset_owner": "supervisor",
        **(
            {"round_seeds": list(round_seeds)}
            if round_seeds is not None else {}
        ),
    }
