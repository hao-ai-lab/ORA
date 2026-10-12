"""Per-task RoboDojo physics device: one host-safe owner, no simulator imports.

User decision 2026-10-04: nine RoboDojo scenes resolve IsaacLab ``cpu``
physics by default; every other task keeps ``cuda:0``. ``cpu`` names the
IsaacLab simulation and tensor device, not pure CPU dynamics: upstream
``BaseEnv.setup_physics`` calls ``overwrite_gpu_setting(1)`` on every device,
so PhysX may still solve on the worker's one visible GPU, which also renders
and runs CuRobo. The lane changes the tensor API (CPU copies instead of GPU
tensors), the broadphase (MBP), USD readback instead of Fabric, and which scene
objects upstream creates. ``physics_scene_probe`` records the actual mode.

The worker never reads this module. ``benchmarks/robodojo.py`` imports it only
inside ``validate_runtime_extension``, which every card and scene dependency
projection strips, and ``robodojo_kernel_review.KERNEL_REVIEWED_EXCLUSIONS``
records that review. The device a worker actually runs with is bound by
``CONFIGURATION_FIELDS.physics_device`` and by the resolved environment
configuration (``sim.device``) in its source manifest.
"""
from __future__ import annotations

import contextlib
from types import MappingProxyType
from typing import Any, Iterator, Mapping, MutableMapping

CUDA_PHYSICS_DEVICE = "cuda:0"
CPU_PHYSICS_DEVICE = "cpu"
#: Every device a RoboDojo runtime may declare; the first is the default.
PHYSICS_DEVICES = (CUDA_PHYSICS_DEVICE, CPU_PHYSICS_DEVICE)

#: Why each scene kind needs CPU physics (upstream ee67a146, IsaacLab afca7b09).
CPU_PHYSICS_REASONS: Mapping[str, str] = MappingProxyType({
    "articulation": (
        "scene ArticulationObject.initialize reads dof_properties through numpy, which crashes "
        "on CUDA tensors (E11BG6-Q001)"
    ),
    "conveyor": (
        "the GPU pipeline drops PhysxSurfaceVelocity belt contacts, so the belt never moves "
        "objects on cuda:0 (E11BG6-001)"
    ),
    "fluid": "upstream SceneManager never creates Fluid objects on a CUDA physics device",
})
#: The nine CPU-physics scenes and their scene kind. A task absent here is ``cuda:0``.
CPU_PHYSICS_SCENES: Mapping[str, str] = MappingProxyType({
    "fill_egg_holder": "articulation",
    "make_toast": "articulation",
    "press_by_number": "articulation",
    "store_laptop_and_headphones": "articulation",
    "swap_blocks": "articulation",
    "match_and_pick_from_conveyor": "conveyor",
    "pick_from_conveyor_by_image": "conveyor",
    "pour_by_language": "fluid",
    "pour_liquid_into_cup": "fluid",
})
#: Task -> reason, the reviewed default-activation table.
CPU_PHYSICS_TASKS: Mapping[str, str] = MappingProxyType({
    task: CPU_PHYSICS_REASONS[kind] for task, kind in CPU_PHYSICS_SCENES.items()
})


def physics_device_for(task: str) -> str:
    """The IsaacLab physics device ``task`` resolves by default: ``cpu`` or ``cuda:0``."""

    return CPU_PHYSICS_DEVICE if task in CPU_PHYSICS_SCENES else CUDA_PHYSICS_DEVICE


def scene_kind_for(task: str) -> str:
    """``articulation``, ``conveyor`` or ``fluid`` for a CPU scene, else ``rigid``."""

    return CPU_PHYSICS_SCENES.get(task, "rigid")



#: The qualification-only physics scene probe's worker report, shared by the
#: worker (``backends/robodojo/physics_scene_probe.py``) and its host launcher
#: (``orchestrator/robodojo_physics_probe.py``), which must not import a
#: process-isolated simulator backend.
WORKER_REPORT_SCHEMA = "ora.robodojo-physics-scene-probe-worker.v1"
WORKER_REPORT_NAME = "probe-worker.json"


@contextlib.contextmanager
def probe_only_admission(task: str | None, limit: int | None,
                         limits: MutableMapping[str, int]) -> Iterator[dict[str, Any] | None]:
    """Admit an unadmitted task into this process's native-action limits for the probe only.

    The caller passes ``benchmarks.robodojo.NATIVE_ACTION_LIMITS``, so this table keeps
    no project import and adds nothing to the worker kernel's import closure.
    """
    if limit is None:
        yield None
        return
    if not isinstance(task, str) or not task or task in limits:
        raise ValueError("a probe-only limit applies only to an unadmitted task")
    limits[task] = int(limit)
    try:
        yield {"task": task, "limit": int(limit), "admission": "unadmitted; qualification probe only"}
    finally:
        limits.pop(task, None)

__all__ = [
    "CPU_PHYSICS_DEVICE", "CPU_PHYSICS_REASONS", "CPU_PHYSICS_SCENES", "CPU_PHYSICS_TASKS",
    "CUDA_PHYSICS_DEVICE", "PHYSICS_DEVICES", "WORKER_REPORT_NAME", "WORKER_REPORT_SCHEMA",
    "physics_device_for", "probe_only_admission", "scene_kind_for",
]
