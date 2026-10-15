"""Canonical composition root for registered external-world sessions."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .rpc import LocalRpcEnvironmentSession
from .synthetic import SyntheticSession

from .definitions import EnvironmentDefinitionError, environment_definition
from .controllers.hybrid_motion import load_hybrid_motion_profile
from .controllers.embodiment_control import load_embodiment_control_profile
from .session import DescribedEnvironmentSession


PROJECT_ROOT = Path(__file__).resolve().parents[2]
#: RoboTwin runtime fields of the read-only reach check's other-arm screen
#: (integration 0059 M6), derived by the RunSpec from the ToolSurface.
ROBOTWIN_REACH_SCREEN_FIELDS = ("reach_check_inter_arm_screen",)


class EnvironmentFactoryError(RuntimeError):
    """A RunPlan cannot be bound to a supported environment backend."""


@dataclass(frozen=True)
class _BuildContext:
    runtime: Mapping[str, Any]
    output_root: Path
    task: str
    seed: int
    max_sim_steps: int
    redacted_environment_names: Sequence[str]
    qualified_runtime_binding: Mapping[str, Any] | None
    action_codec: str | None


def _required_environment_path(
    runtime: Mapping[str, Any], field: str, *, preserve_symlink: bool = False,
) -> Path:
    name = runtime.get(field)
    if not isinstance(name, str) or not name:
        raise EnvironmentFactoryError(f"backend requires runtime.{field}")
    value = os.environ.get(name)
    if not value:
        raise EnvironmentFactoryError(f"missing {name}")
    path = Path(os.path.abspath(os.path.expanduser(value)))
    # Executing a resolved venv/conda Python symlink loses the environment
    # prefix and its site-packages. Data/source roots remain canonicalized.
    return path if preserve_symlink else path.resolve()


def _synthetic(context: _BuildContext) -> object:
    runtime = context.runtime
    main_camera = str(runtime.get("main_camera", "head_camera"))
    configured_public = runtime.get("public_environment_context")
    configured_controller = (
        configured_public.get("controller")
        if isinstance(configured_public, Mapping) else None
    )
    return SyntheticSession(
        context.output_root,
        context.task,
        context.seed,
        max_actions=int(runtime.get("terminal_actions", 1)),
        main_camera=main_camera,
        camera_names=tuple(runtime.get("policy_cameras", [main_camera])),
        action_codec=context.action_codec or "target_state_v4",
        public_controller=(
            str(configured_controller) if configured_controller else None
        ),
    )


def _public_geometry_config(runtime: Mapping[str, Any]) -> dict[str, Any]:
    configured = runtime.get("public_geometry_descriptor")
    if configured is None:
        return {}
    import hashlib
    import json

    path = (PROJECT_ROOT / configured["path"]).resolve()
    if not path.is_relative_to(PROJECT_ROOT) or hashlib.sha256(path.read_bytes()).hexdigest() != configured["sha256"]:
        raise EnvironmentFactoryError("public robot geometry descriptor source drift")
    # The RunSpec resolver validated the complete descriptor. Re-read the exact
    # hash-bound source here, without importing Agent composition into Environment.
    descriptor = json.loads(path.read_text())
    from ora.contracts.robot_geometry import GEOMETRY_BINDING_SCHEMA
    geometry = {gripper["end_effector_id"]: gripper["contact_geometry"] for gripper in descriptor["geometry"]["grippers"]}
    binding = {"schema": GEOMETRY_BINDING_SCHEMA, "descriptor_sha256": configured["sha256"], "geometry": geometry}
    return {"public_robot_geometry": binding}


def validate_robotwin_control_binding(
    runtime: Mapping[str, Any], *, action_codec: str | None,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any] | None:
    """Check source and public topology without opening a simulator session.

    Preflight and factory use the same binding rules, so a passing launch
    cannot discover a known source mismatch only after consuming admission.
    """

    control_profile = None
    if action_codec == "motion_action_v3":
        configured_profile = runtime.get("control_profile")
        if not isinstance(configured_profile, Mapping):
            raise EnvironmentFactoryError(
                "sparse motion action requires runtime.control_profile"
            )
        profile_path = configured_profile.get("path")
        profile_sha256 = configured_profile.get("sha256")
        if not isinstance(profile_path, str) or not isinstance(profile_sha256, str):
            raise EnvironmentFactoryError("runtime.control_profile binding is invalid")
        candidate = (project_root / profile_path).resolve()
        if project_root not in candidate.parents:
            raise EnvironmentFactoryError("runtime.control_profile escapes the project")
        control_profile = load_embodiment_control_profile(candidate, profile_sha256)
        profile = control_profile["profile"]
        if (
            profile["embodiment_id"] != "aloha_agilex"
            or profile["source_binding"]["robotwin_revision"]
            != runtime.get("required_revision")
        ):
            raise EnvironmentFactoryError(
                "control profile does not match RoboTwin source or embodiment: "
                f"profile revision={profile['source_binding']['robotwin_revision']}, "
                f"runtime revision={runtime.get('required_revision')}, "
                f"embodiment={profile['embodiment_id']} (expected aloha_agilex)"
            )
        descriptor = environment_definition("robotwin").resolve_descriptor(
            runtime, action_codec=action_codec,
        )
        effectors = descriptor.controlled_end_effectors or {}
        if set(effectors) != set(profile["arms"]):
            raise EnvironmentFactoryError(
                "control profile arms do not match the Runtime descriptor"
            )
        for arm, record in profile["arms"].items():
            public = effectors.get(arm, {})
            if (
                public.get("controlled_point") != record["controlled_point"]
                or public.get("planner_route") != arm
                or record["planner_route"] != f"robot.{arm}_plan_path"
            ):
                raise EnvironmentFactoryError(
                    f"control profile route does not match descriptor arm {arm}"
                )
        descriptor_groups = {
            (
                str(group.get("group_id")),
                tuple(group.get("members", [])),
                bool(group.get("synchronous_pose_execution")),
            )
            for group in descriptor.control_groups
        }
        profile_groups = {
            (
                str(group.get("group_id")),
                tuple(group.get("members", [])),
                bool(group.get("synchronous_pose_execution")),
            )
            for group in profile["control_groups"]
        }
        if descriptor_groups != profile_groups:
            raise EnvironmentFactoryError(
                "control profile synchronization does not match Runtime descriptor"
            )
    return control_profile


def _robotwin(context: _BuildContext) -> object:
    runtime = context.runtime
    root = _required_environment_path(runtime, "root_env")
    python = _required_environment_path(
        runtime, "python_env", preserve_symlink=True,
    )
    control_profile = validate_robotwin_control_binding(
        runtime, action_codec=context.action_codec, project_root=PROJECT_ROOT,
    )
    return LocalRpcEnvironmentSession(
        python_executable=python,
        project_root=PROJECT_ROOT,
        output_root=context.output_root,
        socket_root=PROJECT_ROOT / ".harness-runtime/env",
        startup_timeout_s=float(runtime["startup_timeout_s"]),
        request_timeout_s=float(runtime["request_timeout_s"]),
        redacted_environment_names=context.redacted_environment_names,
        worker_environment=dict(
            context.runtime.get("resource_binding", {}).get(
                "worker_environment", {}
            )
        ),
        worker_config={
            "backend": "robotwin",
            "execution_capture": runtime.get("execution_capture"),
            "robotwin_root": str(root),
            "output_root": str(context.output_root.resolve()),
            "task": context.task,
            "seed": context.seed,
            "episode": 0,
            "task_config": runtime["task_config"],
            "required_revision": runtime["required_revision"],
            "workspace_bounds_m": runtime["workspace_bounds_m"],
            "max_sim_steps": context.max_sim_steps,
            "operator_world_video": True,
            "operator_execution_video": bool(
                runtime.get("operator_execution_video", False)
            ),
            "environment_signal_policy": runtime.get(
                "environment_signal_policy", "lifecycle_only"
            ),
            "action_codec": context.action_codec,
            "control_profile": control_profile,
            **_public_geometry_config(runtime),
            # The reach check's other-arm screen (0059 M6): the RunSpec sets
            # it only for a surface that declares the screen, so every other
            # worker config keeps its keys.
            **{field: runtime[field] for field in ROBOTWIN_REACH_SCREEN_FIELDS if field in runtime},
        },
    )


def _libero(context: _BuildContext) -> object:
    runtime = context.runtime
    python = _required_environment_path(
        runtime, "python_env", preserve_symlink=True,
    )
    roots = {
        field: _required_environment_path(runtime, field)
        for field in ("libero_pro_root_env", "data_root_env")
    }
    motion_planner = None
    motion_profile = None
    # Sparse motion v3 on LIBERO reuses the qualified hybrid executors, so it
    # carries the CuRobo planner and motion profile bindings into the
    # isolated worker.
    if context.action_codec == "motion_action_v3":
        configured = runtime.get("motion_planner")
        if not isinstance(configured, Mapping):
            raise EnvironmentFactoryError(
                f"{context.action_codec} requires runtime.motion_planner"
            )
        motion_planner = {
            **dict(configured),
            "python_executable": str(_required_environment_path(
                configured, "python_env", preserve_symlink=True,
            )),
            "curobo_root": str(_required_environment_path(configured, "root_env")),
        }
        configured_profile = runtime.get("motion_profile")
        if not isinstance(configured_profile, Mapping):
            raise EnvironmentFactoryError(
                f"{context.action_codec} requires runtime.motion_profile"
            )
        profile_path = configured_profile.get("path")
        profile_sha256 = configured_profile.get("sha256")
        if not isinstance(profile_path, str) or not isinstance(profile_sha256, str):
            raise EnvironmentFactoryError("runtime.motion_profile binding is invalid")
        candidate = (PROJECT_ROOT / profile_path).resolve()
        if PROJECT_ROOT not in candidate.parents:
            raise EnvironmentFactoryError("runtime.motion_profile escapes the project")
        motion_profile = load_hybrid_motion_profile(
            candidate, profile_sha256,
        )
    return LocalRpcEnvironmentSession(
        python_executable=python,
        project_root=PROJECT_ROOT,
        output_root=context.output_root,
        socket_root=PROJECT_ROOT / ".harness-runtime/env",
        startup_timeout_s=float(runtime["startup_timeout_s"]),
        request_timeout_s=float(runtime["request_timeout_s"]),
        redacted_environment_names=context.redacted_environment_names,
        worker_environment=dict(
            context.runtime.get("resource_binding", {}).get(
                "worker_environment", {}
            )
        ),
        worker_config={
            "backend": "libero_pro",
            "execution_capture": runtime.get("execution_capture"),
            "environment_driver": runtime.get("environment_driver", "direct_libero_v1"),
            "libero_pro_root": str(roots["libero_pro_root_env"]),
            "data_root": str(roots["data_root_env"]),
            "output_root": str(context.output_root.resolve()),
            "task": context.task,
            "seed": context.seed,
            "suite": runtime["suite"],
            "task_id": runtime["task_id"],
            "robustness_case": runtime.get("robustness_case"),
            "expected_task_language": runtime["expected_task_language"],
            "required_libero_pro_revision": runtime["required_libero_pro_revision"],
            "required_data_revision": runtime["required_data_revision"],
            "workspace_bounds_m": runtime["workspace_bounds_m"],
            "workspace_profile": runtime.get("workspace_profile"),
            "max_sim_steps": context.max_sim_steps,
            "servo_max_steps": runtime["servo_max_steps"],
            "qualified_controller_binding": context.qualified_runtime_binding,
            "action_codec": context.action_codec,
            "motion_planner": motion_planner,
            "motion_profile": motion_profile,
            **_public_geometry_config(runtime),
        },
    )


def _robodojo_worker_environment(
    runtime: Mapping[str, Any],
) -> dict[str, str]:
    """Bind the isolated RoboDojo worker to exactly one physical device.

    RoboDojo selects its device twice: ``CUDA_VISIBLE_DEVICES`` masks the
    physics/CUDA side (the runtime then always names ``cuda:0``) while the Kit
    renderer takes an *unmasked* Vulkan ordinal from ``renderer_gpu``. A
    formal run additionally carries a resolved ``resource_binding`` whose
    ``worker_environment`` is the host profile's authority on the slot. All
    three must name one device, otherwise a slot lease would guard a card the
    worker does not actually use.
    """

    device = str(runtime["cuda_visible_devices"])
    environment = {"CUDA_VISIBLE_DEVICES": device}
    binding = runtime.get("resource_binding")
    if isinstance(binding, Mapping):
        bound = binding.get("worker_environment")
        if not isinstance(bound, Mapping) or bound.get(
            "CUDA_VISIBLE_DEVICES"
        ) != device:
            raise EnvironmentFactoryError(
                "RoboDojo resource slot and runtime.cuda_visible_devices "
                "select different devices"
            )
        environment.update({str(key): str(value) for key, value in bound.items()})
    return environment


def build_robodojo_worker_session(
    *, runtime: Mapping[str, Any], output_root: Path, task: str, seed: int,
    max_sim_steps: int, action_codec: str | None,
    qualified_runtime_binding: Mapping[str, Any] | None = None,
    redacted_environment_names: Sequence[str] = (),
) -> LocalRpcEnvironmentSession:
    """Start the isolated RoboDojo worker exactly as the factory does.

    Scripted motion qualification needs this bootstrap before the codec's
    admission row exists, so it is reachable without the descriptor gate;
    formal runs still enter through ``build_environment_session``.
    """

    root = _required_environment_path(runtime, "root_env")
    assets = _required_environment_path(runtime, "asset_root_env")
    python = _required_environment_path(runtime, "python_env", preserve_symlink=True)
    worker_environment = _robodojo_worker_environment(runtime)
    worker = dict(runtime)
    worker.update(_public_geometry_config(runtime))
    worker.update({
        "backend": "robodojo", "robodojo_root": str(root),
        "output_root": str(output_root.resolve()),
        "runtime_directory": str(python.parent.parent.parent),
        "accept_eula": True,
        "asset_root": str(assets), "task": task, "eval_seed": seed,
        "max_sim_steps": max_sim_steps, "action_codec": action_codec,
        "qualification_binding": qualified_runtime_binding,
    })
    return LocalRpcEnvironmentSession(
        python_executable=python, project_root=PROJECT_ROOT,
        output_root=output_root,
        socket_root=PROJECT_ROOT / ".harness-runtime/env",
        startup_timeout_s=float(runtime["startup_timeout_s"]),
        request_timeout_s=float(runtime["request_timeout_s"]),
        close_timeout_s=float(runtime["close_timeout_s"]),
        redacted_environment_names=redacted_environment_names,
        worker_environment=worker_environment,
        worker_config=worker,
    )


def _robodojo(context: _BuildContext) -> object:
    return build_robodojo_worker_session(
        runtime=context.runtime, output_root=context.output_root, task=context.task,
        seed=context.seed, max_sim_steps=context.max_sim_steps,
        action_codec=context.action_codec,
        qualified_runtime_binding=context.qualified_runtime_binding,
        redacted_environment_names=context.redacted_environment_names,
    )


_BUILDERS: dict[tuple[str, str], Callable[[_BuildContext], object]] = {
    ("synthetic", "in_process_direct"): _synthetic,
    ("robotwin", "local_rpc"): _robotwin,
    ("libero_pro", "local_rpc"): _libero,
    ("robodojo", "local_rpc"): _robodojo,
}


def build_environment_session(
    *,
    runtime_config: Mapping[str, Any],
    output_root: Path,
    task: str,
    seed: int,
    max_sim_steps: int,
    redact_environment_names: Sequence[str] = (),
    action_codec: str | None = None,
    qualified_runtime_binding: Mapping[str, Any] | None = None,
) -> DescribedEnvironmentSession:
    """Resolve one definition and instantiate only its selected backend."""

    kind = runtime_config.get("kind")
    transport = runtime_config.get("transport", "in_process_direct")
    if not isinstance(kind, str) or not isinstance(transport, str):
        raise EnvironmentFactoryError("runtime kind and transport must be strings")
    try:
        definition = environment_definition(kind)
        builder = _BUILDERS[(kind, transport)]
    except (KeyError, EnvironmentDefinitionError) as exc:
        raise EnvironmentFactoryError(
            f"unsupported environment backend/transport: {kind!r}/{transport!r}"
        ) from exc
    try:
        descriptor = definition.resolve_descriptor(
            runtime_config, action_codec=action_codec,
        )
    except EnvironmentDefinitionError as exc:
        raise EnvironmentFactoryError(str(exc)) from exc
    delegate = builder(_BuildContext(
        runtime=runtime_config,
        output_root=output_root,
        task=task,
        seed=seed,
        max_sim_steps=max_sim_steps,
        redacted_environment_names=redact_environment_names,
        qualified_runtime_binding=qualified_runtime_binding,
        action_codec=action_codec,
    ))
    return DescribedEnvironmentSession(
        delegate=delegate,
        planned_descriptor=descriptor,
        runtime_config=runtime_config,
        action_codec=action_codec,
    )


__all__ = ["EnvironmentFactoryError", "build_environment_session", "build_robodojo_worker_session"]
