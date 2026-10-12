"""RoboDojo runtime selection used by infrastructure qualification.

Moved from the robodojo-compat experiment package on 2026-09-28
(relocation U6). Qualification output stays under the ``robodojo-compat``
workspace so existing records are found where they were written.
"""

from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from ora.environment.benchmarks.robodojo import (
    NATIVE_ACTION_LIMITS, REVISIONS, REVISION_FIELDS, public_context, selected_layout,
)
from ora.environment.benchmarks.robodojo_layout_catalog import build_layout_schedule
from ora.environment.benchmarks.robodojo_physics import (
    CPU_PHYSICS_DEVICE, CUDA_PHYSICS_DEVICE, physics_device_for,
)
from ora.orchestrator.paths import experiment_workspace
from ora.orchestrator.resources import declared_execution_host

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = experiment_workspace("robodojo-compat")
RUNTIME_DIRECTORY = Path.home() / "robodojo-runtime"


def configure_environment() -> None:
    """Supply documented local defaults without changing the shared workspace."""
    defaults = {
        "ROBODOJO_ROOT": RUNTIME_DIRECTORY / "source/RoboDojo",
        "ROBODOJO_ASSET_ROOT": RUNTIME_DIRECTORY / "assets/Assets",
        "ROBODOJO_PYTHON": RUNTIME_DIRECTORY / "env/bin/python",
    }
    for name, value in defaults.items():
        os.environ.setdefault(name, str(value))


def execution_host() -> str:
    """The RoboDojo host this launcher runs on: its host marker.

    The runtime binds the marker's host so preflight (marker == required_host)
    and the slot lookup agree; an unset or unregistered marker is refused here
    rather than by a later lease or preflight failure.
    """
    return declared_execution_host()


def runtime_configuration(gpu: str, task: str = "arrange_largest_number", layout_id: int = 0) -> dict[str, Any]:
    """The default RoboDojo runtime of ``task`` rendered on physical GPU ``gpu``.

    ``layout_id`` selects the official layout ordinal of evaluation seed 0
    (default 0), resolved by upstream's numeric sort and hashed from the pinned
    Assets; an ordinal the Assets do not hold is refused.

    Default activation (user decision 2026-10-04): ``physics_device`` is the
    task's own (``robodojo_physics.physics_device_for``), ``cpu`` for the
    CPU-physics scenes and ``cuda:0`` otherwise. ``gpu`` stays the one CUDA
    device the worker sees and the Vulkan renderer in both cases.
    """
    configure_environment()
    from ora.environment.benchmarks.robodojo_asset_revision import (
        asset_revision_for, SWAP_T_ASSET_ROOT_ENV,
    )
    asset_variable = "ROBODOJO_ASSET_ROOT"
    if task == "swap_T":
        asset_variable = SWAP_T_ASSET_ROOT_ENV
        base = Path(os.environ["ROBODOJO_ASSET_ROOT"]).resolve().parent.parent
        os.environ.setdefault(asset_variable, str(base / "assets-35efbc7d/Assets"))
    assets = Path(os.environ[asset_variable]).resolve()
    layout = selected_layout(assets, task=task, eval_seed=0, layout_id=layout_id)
    return {
        "kind": "robodojo", "transport": "local_rpc",
        "root_env": "ROBODOJO_ROOT", "python_env": "ROBODOJO_PYTHON",
        "asset_root_env": asset_variable,
        **{REVISION_FIELDS[name]: revision for name, revision in REVISIONS.items()},
        "required_asset_revision": asset_revision_for(task),
        "required_host": execution_host(), "require_gpu": True,
        "task_variant": task, "eval_seed": 0, "layout_id": layout_id,
        "layout_file": layout.relative_to(assets).as_posix(),
        "layout_sha256": hashlib.sha256(layout.read_bytes()).hexdigest(),
        "num_envs": 1, "env_config": "arx_x5", "sensor_profile": "rgbd_metric_v1",
        "physics_dt": 0.004, "control_hz": 25,
        "native_action_limit": NATIVE_ACTION_LIMITS[task], "native_actions_per_command": 20,
        "physics_device": physics_device_for(task), "renderer_gpu": int(gpu), "cuda_visible_devices": gpu,
        "main_camera": "head", "policy_cameras": ["head", "left_wrist", "right_wrist"],
        "workspace_bounds_m": {"x": [-1.5, 1.5], "y": [-1.5, 1.5], "z": [-0.2, 2.0]},
        "startup_timeout_s": 600, "request_timeout_s": 600, "close_timeout_s": 30,
        "public_environment_context": public_context(),
    }


def qualification_device_label(gpu: str, physics_device: str) -> str:
    """The device part of a RoboDojo card or lane-gate file name.

    ``cuda<N>`` for CUDA physics on physical GPU N, exactly the names every
    record before 2026-10-04 carries; ``cpu-gpu<N>`` for CPU physics rendered
    on GPU N. The two never collide, so a CPU card or gate cannot shadow a
    CUDA one in a namespace or a reused namespace.
    """
    if physics_device == CUDA_PHYSICS_DEVICE:
        return f"cuda{gpu}"
    if physics_device == CPU_PHYSICS_DEVICE:
        return f"cpu-gpu{gpu}"
    raise ValueError(f"unknown RoboDojo physics device {physics_device!r}")


def robodojo_layout_schedule(task: str, rounds: int, *, layout_ids: Sequence[int] | None = None) -> dict[str, Any]:
    """The default per-round layout schedule of a persistent RoboDojo session.

    Default activation (user decision 2026-10-01): every round of a K-round
    session starts on a new reviewed layout. The schedule is the first
    ``rounds`` entries of the task's reviewed pool, or an explicit selection
    inside it, each hashed from the pinned Assets. A task without a reviewed
    pool is refused; replaying round 0 is the explicit ``repeat_round0_layout``
    treatment of the builder, never a fallback here.
    """
    configure_environment()
    return build_layout_schedule(
        task, rounds, asset_root=Path(os.environ["ROBODOJO_ASSET_ROOT"]),
        project_root=PROJECT_ROOT, layout_ids=layout_ids,
    )


def with_layout_schedule(runtime: dict[str, Any], schedule: Mapping[str, Any]) -> dict[str, Any]:
    """Bind a schedule and its round-0 layout fields onto one RoboDojo runtime."""
    first = schedule["rounds"][0]
    return {**runtime, "layout_id": first["layout_id"], "layout_file": first["layout_file"],
            "layout_sha256": first["layout_sha256"], "layout_schedule": copy.deepcopy(dict(schedule))}
