"""Construct source-bound LIBERO drivers inside the isolated worker."""
from __future__ import annotations
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable
from ora.environment.controllers.libero_osc import libero_osc_action_contract


def build_environment(
    *, rlinf_root: Path | None, libero_pro_root: Path, data_root: Path,
    runtime_config_root: Path, suite: str, task_id: int, seed: int,
    max_sim_steps: int, robustness_case: str | None = None,
    environment_driver: str = "rlinf_v1",
) -> tuple[Any, dict[str, Any]]:
    """Construct the selected source-bound driver, without importing RPent."""
    from .session import (
        LiberoProError, _prepare_pinned_libero_imports, _git,
        libero_environment_init_params, _camera_metadata_env_factory,
        libero_public_environment_context,
    )

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    runtime_bindings = _prepare_pinned_libero_imports(
        libero_pro_root=libero_pro_root, data_root=data_root,
        runtime_config_root=runtime_config_root, suite=suite,
    )
    from robosuite.controllers import load_controller_config
    import robosuite
    import robosuite.models.grippers.panda_gripper as panda_gripper_module
    if environment_driver == "direct_libero_v1":
        from libero.libero import benchmark
        runtime_bindings.pop("rlinf_import_variant", None)
        runtime_bindings["environment_driver"] = environment_driver
    elif environment_driver == "rlinf_v1":
        if rlinf_root is None:
            raise LiberoProError("legacy RLinf driver requires its source root")
        rlinf_text = str(rlinf_root)
        if rlinf_text in sys.path:
            sys.path.remove(rlinf_text)
        sys.path.insert(0, rlinf_text)
        from omegaconf import OmegaConf
        from rlinf.envs.libero.libero_env import LiberoEnv
        from rlinf.envs.libero.utils import benchmark
    else:
        raise LiberoProError("unsupported LIBERO environment driver")
    from .registration import ensure_perturbation_suites

    from ora.environment.benchmarks.libero_pro import CORE_SUITES

    if suite in CORE_SUITES:
        ensure_perturbation_suites(benchmark)

    benchmark_suite = benchmark.get_benchmark(suite)()
    if not 0 <= task_id < benchmark_suite.get_num_tasks():
        raise LiberoProError(f"task_id {task_id} is outside suite {suite}")
    trials = len(benchmark_suite.get_task_init_states(task_id))
    if trials <= 0:
        raise LiberoProError(f"suite {suite} task {task_id} has no init states")
    first_id = sum(
        len(benchmark_suite.get_task_init_states(index)) for index in range(task_id)
    )
    reset_id = first_id + (seed % trials)
    task = benchmark_suite.get_task(task_id)
    robustness_bddl: Path | None = None
    robustness_config: dict[str, Any] | None = None
    if robustness_case is not None:
        from .robustness import (
            ROBUSTNESS_CODE_REVISION,
            parse_and_validate_robustness_bddl,
            robustness_bddl_path,
        )

        actual_revision = _git(libero_pro_root, "rev-parse", "HEAD")
        if actual_revision != ROBUSTNESS_CODE_REVISION:
            raise LiberoProError(
                "BDDL robustness requires the paired LIBERO-Pro runtime: "
                f"expected {ROBUSTNESS_CODE_REVISION}, got {actual_revision}"
            )
        robustness_bddl = robustness_bddl_path(
            data_root, case=robustness_case, suite=suite,
            task_name=str(task.name),
        )
        robustness_config = parse_and_validate_robustness_bddl(
            robustness_bddl, case=robustness_case, suite=suite,
            task_name=str(task.name),
        )
    init_params = libero_environment_init_params(max_sim_steps=max_sim_steps)
    controller_action_contract = libero_osc_action_contract(
        load_controller_config(default_controller="OSC_POSE")
    )
    robosuite_root = Path(str(robosuite.__file__)).resolve().parent
    controller_config_file = robosuite_root / "controllers/config/osc_pose.json"
    panda_gripper_file = robosuite_root / "models/assets/grippers/panda_gripper.xml"
    for dependency in (controller_config_file, panda_gripper_file):
        if not dependency.is_file():
            raise LiberoProError(f"RoboSuite control dependency is unavailable: {dependency}")
    if environment_driver == "direct_libero_v1":
        from libero.libero.envs import OffScreenRenderEnv
        from .direct import DirectLiberoEnvironment
        active_bddl = robustness_bddl or Path(benchmark_suite.get_task_bddl_file_path(task_id))
        factory = lambda: OffScreenRenderEnv(
            bddl_file_name=str(active_bddl), **{**init_params, "ignore_done": True})
        if robustness_bddl is not None:
            from .robustness import robustness_env_factory
            factory = robustness_env_factory(factory, case=str(robustness_case), suite=suite,
                task_name=str(task.name), bddl_path=robustness_bddl, seed=seed)
        environment = DirectLiberoEnvironment(
            factory=_camera_metadata_env_factory(factory),
            init_states=benchmark_suite.get_task_init_states(task_id),
            first_reset_id=first_id, reset_id=reset_id, seed=seed,
            task_language=str(task.language), max_sim_steps=max_sim_steps)
    else:
        config = OmegaConf.create({
            "env_type": "libero",
            "task_suite_name": suite,
            "auto_reset": False,
            "ignore_terminations": False,
            "max_steps_per_rollout_epoch": max_sim_steps,
            "max_episode_steps": max_sim_steps,
            "use_rel_reward": False,
            "use_step_penalty": False,
            "reward_coef": 1.0,
            "reset_gripper_open": True,
            "is_eval": True,
            "seed": seed,
            "group_size": 1,
            "use_fixed_reset_state_ids": True,
            "use_ordered_reset_state_ids": True,
            "specific_reset_id": reset_id,
            "video_cfg": {
                "save_video": False, "info_on_video": False,
                "video_base_dir": "/tmp/ora-libero-pro",
            },
            "init_params": init_params,
        })
        class HarnessLiberoEnv(LiberoEnv):
            def get_env_fn_params(self, env_idx: Any = None) -> list[dict[str, Any]]:
                params = super().get_env_fn_params(env_idx)
                if robustness_bddl is not None:
                    for item in params:
                        item["bddl_file_name"] = str(robustness_bddl)
                return params

            def get_env_fns(self) -> list[Callable[[], Any]]:
                factories = super().get_env_fns()
                if robustness_bddl is not None:
                    from .robustness import robustness_env_factory

                    factories = [
                        robustness_env_factory(
                            factory, case=str(robustness_case), suite=suite,
                            task_name=str(task.name), bddl_path=robustness_bddl,
                            seed=seed,
                        )
                        for factory in factories
                    ]
                return [_camera_metadata_env_factory(factory) for factory in factories]

        environment = HarnessLiberoEnv(
            cfg=config, num_envs=1, seed_offset=0, total_num_processes=1,
            worker_info=None,
        )
    public_environment_context = libero_public_environment_context(
        task, init_params=init_params,
    )
    init_states_path = (
        Path(benchmark.get_libero_path("init_states"))
        / suite / str(task.init_states_file)
    )
    active_bddl = (
        robustness_bddl
        if robustness_bddl is not None
        else Path(benchmark_suite.get_task_bddl_file_path(task_id))
    )
    robustness_runtime_module: Path | None = None
    robustness_parser_module: Path | None = None
    if robustness_case is not None:
        robustness_runtime_module = (
            libero_pro_root / "libero/libero/envs/robustness_perturbations.py"
        ).resolve()
        robustness_parser_module = (
            libero_pro_root / "libero/libero/envs/perturbation_config.py"
        ).resolve()
    return environment, {
        "suite": suite,
        "task_id": task_id,
        "task_language": str(task.language).strip(),
        "task_name": str(task.name),
        "bddl_file": str(active_bddl),
        "init_states_file": str(init_states_path),
        "reset_id": reset_id,
        "trial_count": trials,
        **({"rlinf_root": str(rlinf_root)} if environment_driver == "rlinf_v1" else {}),
        "environment_driver": environment_driver,
        "libero_pro_root": str(libero_pro_root),
        "data_root": str(data_root),
        "runtime_bindings": {
            **runtime_bindings,
            **({"rlinf_module": str(Path(sys.modules["rlinf.envs.libero.libero_env"].__file__).resolve())}
               if environment_driver == "rlinf_v1" else
               {"direct_driver_module": str(Path(__file__).with_name("direct.py").resolve())}),
            "robosuite_module": str(Path(robosuite.__file__).resolve()),
            "controller_config_file": str(controller_config_file.resolve()),
            "panda_gripper_module": str(
                Path(panda_gripper_module.__file__).resolve()
            ),
            "panda_gripper_file": str(panda_gripper_file.resolve()),
            **(
                {
                    "robustness_runtime_module": str(robustness_runtime_module),
                    "robustness_parser_module": str(robustness_parser_module),
                }
                if robustness_runtime_module is not None
                and robustness_parser_module is not None else {}
            ),
        },
        "controller_action_contract": controller_action_contract,
        "public_environment_context": public_environment_context,
        "robustness": (
            None if robustness_case is None else {
                "case": robustness_case,
                "config_sha256": hashlib.sha256(json.dumps(
                    robustness_config, ensure_ascii=False, sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")).hexdigest(),
                "official_predicate_changed": False,
            }
        ),
    }
