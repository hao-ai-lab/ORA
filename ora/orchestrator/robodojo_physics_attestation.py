"""Host-side physics-mode attestation and round rules of the RoboDojo CPU-physics lane.

No worker or kernel code reads this module. It checks what a worker already
publishes: the resolved environment configuration (``sim.device``, Fabric) in
its source manifest. ``device_binding.physics_cuda_*`` names the worker's one
visible CUDA device (renderer, CuRobo, forced PhysX GPU dynamics); it is not
proof of where the IsaacLab tensor API lives, so the attestation here reads
the resolved configuration instead.

A CUDA manifest written before this attestation may lack ``sim.device``; CUDA
was the only device then, so a missing value is admitted for ``cuda:0`` only.
A CPU runtime must prove ``sim.device == "cpu"`` and no Fabric.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ora.environment.benchmarks.robodojo import RoboDojoBenchmarkError
from ora.environment.benchmarks.robodojo_physics import CPU_PHYSICS_DEVICE, CUDA_PHYSICS_DEVICE

CPU_ROUND_RESTART_RULE = "robodojo_cpu_in_place_restart"
DEVICE_INVALID = "robodojo_physics_device_invalid"
FABRIC_ENABLED = "robodojo_physics_fabric_enabled"
#: The run's own artifacts cannot show whether PhysX solved on the GPU; the
#: qualification probe records ``overwrite_gpu_setting`` and the scene flags.
GPU_DYNAMICS_NOTE = "not_attested_by_the_run; see physics_scene_probe"


def cpu_round_restart_refusal(runtime: Mapping[str, Any], *, rounds: int) -> str | None:
    """Refuse a CPU-physics RoboDojo session of K > 1 rounds without a layout schedule.

    On CPU, upstream's reset respawns every non-articulation object, and an
    in-place restart reuses the destroyed prims' paths: the objects come back
    visible but gravity- and possibly collision-disabled, and the identity
    checks still pass. A scheduled round relaunches the simulation, which is
    correct, so CPU sessions are single-round or layout-scheduled.
    """
    if (runtime.get("kind") != "robodojo" or runtime.get("physics_device") != CPU_PHYSICS_DEVICE
            or runtime.get("layout_schedule") is not None or int(rounds) <= 1):
        return None
    return (
        f"a CPU-physics RoboDojo session of {int(rounds)} rounds needs a reviewed layout_schedule: "
        "upstream's CPU reset leaves respawned rigid objects gravity- and collision-disabled after an "
        "in-place round restart, so every later round must relaunch on a scheduled layout; "
        f"rule={CPU_ROUND_RESTART_RULE}"
    )


def resolved_sim_device(source: Mapping[str, Any]) -> str | None:
    config = source.get("resolved_environment_config")
    sim = config.get("sim") if isinstance(config, Mapping) else None
    device = sim.get("device") if isinstance(sim, Mapping) else None
    return None if device is None else str(device)


def _fabric(source: Mapping[str, Any]) -> bool:
    config = source.get("resolved_environment_config")
    sim = config.get("sim") if isinstance(config, Mapping) else None
    return isinstance(sim, Mapping) and sim.get("use_fabric") is True


def physics_attestation_errors(runtime: Mapping[str, Any], source: Mapping[str, Any]) -> list[str]:
    """Evaluation reason codes: the resolved device differs from the declared one, or CPU ran Fabric."""

    declared = runtime.get("physics_device", CUDA_PHYSICS_DEVICE)
    resolved = resolved_sim_device(source)
    if resolved != declared and not (resolved is None and declared == CUDA_PHYSICS_DEVICE):
        return [DEVICE_INVALID]
    if declared == CPU_PHYSICS_DEVICE and _fabric(source):
        return [FABRIC_ENABLED]
    return []


def physics_mode(runtime: Mapping[str, Any], source: Mapping[str, Any]) -> dict[str, Any]:
    declared = runtime.get("physics_device", CUDA_PHYSICS_DEVICE)
    resolved = resolved_sim_device(source)
    return {
        "physics_device": declared,
        "isaaclab_sim_device": resolved,
        "tensor_api": None if resolved is None else ("cpu" if resolved == CPU_PHYSICS_DEVICE else "cuda"),
        "fabric": _fabric(source),
        "physx_gpu_dynamics": GPU_DYNAMICS_NOTE,
    }


def require_physics_attestation(runtime: Mapping[str, Any], source: Mapping[str, Any]) -> dict[str, Any]:
    """The physics mode, or a refusal naming the drift (revalidation of a scene smoke)."""

    errors = physics_attestation_errors(runtime, source)
    if errors:
        raise RoboDojoBenchmarkError(
            f"scene physics device drift: declared {runtime.get('physics_device')!r}, resolved "
            f"sim.device {resolved_sim_device(source)!r}, fabric {_fabric(source)} ({', '.join(errors)})"
        )
    return physics_mode(runtime, source)


def physics_record(plan: Mapping[str, Any], native_root: Path) -> dict[str, Any] | None:
    """The physics-mode record an evaluation and its seal carry for a RoboDojo run."""

    runtime = plan.get("runtime") or {}
    if runtime.get("kind") != "robodojo":
        return None
    path = Path(native_root) / "source-manifest.json"
    try:
        source = json.loads(path.read_text(encoding="utf-8")) if path.is_file() and not path.is_symlink() else None
    except (OSError, ValueError):
        source = None
    if not isinstance(source, Mapping):
        return {**physics_mode(runtime, {}), "source_manifest": "absent", "errors": []}
    return {**physics_mode(runtime, source), "source_manifest": "source-manifest.json",
            "errors": physics_attestation_errors(runtime, source)}


__all__ = [
    "CPU_ROUND_RESTART_RULE", "DEVICE_INVALID", "FABRIC_ENABLED", "cpu_round_restart_refusal",
    "physics_attestation_errors", "physics_mode", "physics_record", "require_physics_attestation",
    "resolved_sim_device",
]
