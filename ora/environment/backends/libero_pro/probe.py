"""External-interpreter metadata probe for the pinned LIBERO-Pro runtime.

Both generic preflight and the LIBERO-Pro experiment CLI execute this module in
the configured simulator Python.  Keeping the probe here prevents two embedded
scripts from drifting while preserving the process-isolation boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

from .registration import ensure_perturbation_suites
from .robustness import (
    parse_and_validate_robustness_bddl,
    robustness_bddl_path,
)
from .session import (
    _prepare_pinned_libero_imports,
    bddl_task_language,
    libero_environment_init_params,
    libero_osc_action_contract,
    libero_public_environment_context,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probe(
    *,
    rlinf_root: Path | None,
    libero_pro_root: Path,
    data_root: Path,
    runtime_config_root: Path,
    suite: str,
    task_id: int,
    expected_task_language: str | None = None,
    robustness_case: str | None = None,
    include_init_states: bool = False,
    include_controller: bool = False,
    environment_driver: str = "rlinf_v1",
) -> dict[str, Any]:
    """Resolve one task and optional controller binding without starting MuJoCo."""

    if environment_driver not in {"rlinf_v1", "direct_libero_v1"}:
        raise ValueError("unsupported LIBERO environment driver")
    if environment_driver == "rlinf_v1" and rlinf_root is None:
        raise ValueError("legacy RLinf probe requires its source root")
    rlinf_root = rlinf_root.resolve() if rlinf_root is not None else None
    libero_pro_root = libero_pro_root.resolve()
    data_root = data_root.resolve()
    if environment_driver == "rlinf_v1":
        sys.path.insert(0, str(rlinf_root))
    module_names = [
        "ora.environment.worker",
        "numpy",
        "PIL",
        "yaml",
        "torch",
    ]
    modules = {name: importlib.import_module(name) for name in module_names}
    runtime_bindings = _prepare_pinned_libero_imports(
        libero_pro_root=libero_pro_root,
        data_root=data_root,
        runtime_config_root=runtime_config_root,
        suite=suite,
    )
    if environment_driver == "direct_libero_v1":
        modules["ora.environment.backends.libero_pro.direct"] = importlib.import_module(
            "ora.environment.backends.libero_pro.direct")
        modules["libero.libero.envs"] = importlib.import_module("libero.libero.envs")
        from libero.libero import benchmark
        runtime_bindings.pop("rlinf_import_variant", None)
        runtime_bindings["environment_driver"] = environment_driver
    else:
        modules["omegaconf"] = importlib.import_module("omegaconf")
        modules["rlinf.envs.libero.libero_env"] = importlib.import_module("rlinf.envs.libero.libero_env")
        from rlinf.envs.libero.utils import benchmark
    modules["libero.libero"] = importlib.import_module("libero.libero")

    from ora.environment.benchmarks.libero_pro import CORE_SUITES

    if suite in CORE_SUITES:
        ensure_perturbation_suites(benchmark)
    selected = benchmark.get_benchmark(suite)()
    task = selected.get_task(task_id)
    active_bddl = Path(selected.get_task_bddl_file_path(task_id))
    robustness = None
    if robustness_case is not None:
        active_bddl = robustness_bddl_path(
            data_root,
            case=robustness_case,
            suite=suite,
            task_name=str(task.name),
        )
        config = parse_and_validate_robustness_bddl(
            active_bddl,
            case=robustness_case,
            suite=suite,
            task_name=str(task.name),
        )
        robustness = {
            "case": robustness_case,
            "bddl_file": str(active_bddl),
            "config_sha256": hashlib.sha256(json.dumps(
                config, sort_keys=True, separators=(",", ":"),
            ).encode()).hexdigest(),
        }
    actual_language = str(task.language).strip()
    # The benchmark registry derives its task language from the BDDL file
    # name. Perturbation suites keep the base file name while rewriting the
    # goal, so the registry language can describe a different task than the
    # evaluated predicate. This binds the policy-visible instruction to
    # the BDDL file's own (:language ...); it does not parse or compare
    # the (:goal ...) predicate itself. Report the file's language from the
    # bound BDDL so the launch gate can refuse that divergence.
    bddl_language = bddl_task_language(Path(str(active_bddl)))
    result: dict[str, Any] = {
        "executable": sys.executable,
        "python": platform.python_version(),
        "suite": suite,
        "task_id": task_id,
        "task_name": str(task.name),
        "task_language": actual_language,
        "expected_task_language": expected_task_language,
        "language_matches": (
            expected_task_language is None
            or actual_language == expected_task_language.strip()
        ),
        "bddl_task_language": bddl_language,
        "bddl_language_matches": (
            None if bddl_language is None or expected_task_language is None
            else bddl_language == expected_task_language.strip()
        ),
        "bddl_file": str(active_bddl),
        "init_states_file": str(
            Path(benchmark.get_libero_path("init_states"))
            / suite / str(task.init_states_file)
        ),
        "robustness": robustness,
        **({"rlinf_root": str(rlinf_root)} if environment_driver == "rlinf_v1" else {}),
        "environment_driver": environment_driver,
        "runtime_bindings": runtime_bindings,
        "public_environment_context": libero_public_environment_context(
            task, init_params=libero_environment_init_params(max_sim_steps=1),
        ),
        "modules": {
            name: getattr(module, "__file__", None)
            for name, module in modules.items()
        },
    }
    if include_init_states:
        result["init_state_count"] = len(selected.get_task_init_states(task_id))
    if include_controller:
        import numpy as np
        import robosuite.models.grippers.panda_gripper as panda_gripper_module
        from robosuite.controllers import load_controller_config
        from robosuite.models.grippers import PandaGripper

        robosuite = importlib.import_module("robosuite")
        modules["robosuite"] = robosuite
        controller_contract = libero_osc_action_contract(
            load_controller_config(default_controller="OSC_POSE")
        )
        robosuite_root = Path(str(robosuite.__file__)).resolve().parent
        controller_config = robosuite_root / "controllers/config/osc_pose.json"
        gripper_file = robosuite_root / "models/assets/grippers/panda_gripper.xml"
        gripper = object.__new__(PandaGripper)
        gripper.current_action = np.zeros(1, dtype=np.float64)
        result.update({
            "controller_action_contract": controller_contract,
            "gripper_action_probe": {
                "close": PandaGripper.format_action(
                    gripper, np.asarray([1.0]),
                ).tolist(),
                "zero_hold": PandaGripper.format_action(
                    gripper, np.asarray([0.0]),
                ).tolist(),
                "reopen": PandaGripper.format_action(
                    gripper, np.asarray([-1.0]),
                ).tolist(),
            },
            "controller_binding": {
                "robosuite_module": str(Path(str(robosuite.__file__)).resolve()),
                "robosuite_module_sha256": _sha256(
                    Path(str(robosuite.__file__)).resolve()
                ),
                "controller_config_file": str(controller_config.resolve()),
                "controller_config_sha256": _sha256(controller_config),
                "panda_gripper_module": str(
                    Path(str(panda_gripper_module.__file__)).resolve()
                ),
                "panda_gripper_module_sha256": _sha256(
                    Path(str(panda_gripper_module.__file__)).resolve()
                ),
                "panda_gripper_file": str(gripper_file.resolve()),
                "panda_gripper_sha256": _sha256(gripper_file),
            },
        })
        result["modules"]["robosuite"] = str(robosuite.__file__)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rlinf-root", type=Path)
    parser.add_argument("--environment-driver", choices=("rlinf_v1", "direct_libero_v1"), default="rlinf_v1")
    parser.add_argument("--libero-pro-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--runtime-config-root", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--expected-task-language")
    parser.add_argument("--robustness-case")
    parser.add_argument("--include-init-states", action="store_true")
    parser.add_argument("--include-controller", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = probe(
        rlinf_root=args.rlinf_root,
        environment_driver=args.environment_driver,
        libero_pro_root=args.libero_pro_root,
        data_root=args.data_root,
        runtime_config_root=args.runtime_config_root,
        suite=args.suite,
        task_id=args.task_id,
        expected_task_language=args.expected_task_language,
        robustness_case=args.robustness_case,
        include_init_states=args.include_init_states,
        include_controller=args.include_controller,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
