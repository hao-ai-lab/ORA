"""Physics scene probe of one RoboDojo scene, inside the isolated RoboDojo interpreter.

Qualification only (CPU-physics lane, 2026-10-04). No policy, Provider or
ToolSurface reaches this module, and no worker entry point imports it, so it is
outside the worker kernel and its edits stale no card or gate.
``ora.orchestrator.robodojo_physics_probe`` builds the worker
configuration from the default runtime of a task on one leased slot, starts
this module with the RoboDojo interpreter and judges what it records; see that
module for the operator command.

Contract:

* ``session.start()`` resets the task on layout 0 exactly as a formal run does.
* The physics mode is read back once: IsaacLab's simulation device, Fabric,
  ``/physics/suppressReadback``, ``/physics/fabricEnabled``, PhysX
  ``get_overwrite_gpu_setting()`` (upstream forces 1, GPU, on every device),
  the physics scenes' GPU-dynamics and broadphase attributes, and
  ``SimulationManager.get_physics_sim_device()``.
* The plant then holds ``--ticks`` native actions through
  ``session._conformance_tick({})`` (the retained arm and gripper targets, no
  new command) and stops early only on the official terminal.
* Every ``--sample-every`` ticks, and at tick 0 and the last tick, it records
  private scene state: every layout instance's world position, articulation
  joint positions, and fluid particle counts and bounds. Conveyor belts
  (``PhysxSurfaceVelocityAPI`` prims) are read once with their world bounds.
  This is qualification evidence and never policy-visible.
* ``--in-place-restart`` additionally restarts round 1 in place and reads each
  rigid object's gravity and collision flags before and after: the CPU
  in-place restart hazard of ``robodojo_physics_attestation``. Report only.
* ``--probe-only-native-action-limit`` admits an unadmitted task into this
  process only (the two fluid scenes used it before their 2026-10-04
  admission), with its upstream ``step_lim``; the Runtime's admission table is
  unchanged everywhere else.
* In a scene with a liquid it also records, at tick 0 and at the end of the
  hold, the raw inputs the official fluid predicate reads
  (``func_parser.is_A_fluid_in_B``): each labelled liquid's particle positions
  as ``get_particle_positions()`` returns them and its layout init pose, and
  each labelled rigid or geometry instance's env-relative pose and local
  oriented-bbox vertices. A read-only offline replay of the predicate uses
  them; the judge does not.
* Every reader failure is recorded in the sample, never raised; only a boot or
  hold failure is the worker's ``error``. The report is
  ``probe-worker.json`` under ``--output``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ora.environment.benchmarks.robodojo_physics import (
    WORKER_REPORT_NAME, WORKER_REPORT_SCHEMA, probe_only_admission,
)

VISIBILITY = "qualification_only_never_policy_visible"
#: Kit sometimes never returns from close; the evidence is already on disk.
CLOSE_TIMEOUT_S = 60.0
#: Ticks held after the in-place restart, so a gravity-free object has time to show it.
RESTART_HOLD_TICKS = 10
#: Layout sections whose labelled instances the fluid predicate can name as a
#: container or a bottle (upstream ``LayoutManager.get_instance_name`` order).
CONTAINER_SECTIONS = ("Rigid", "Dynamic", "Geometry", "Articulation")

Reader = Callable[[Any], Any]


@dataclass(frozen=True)
class Readers:
    """Scene readers; each takes the live session and returns JSON values."""

    physics_mode: Reader
    objects: Reader
    articulations: Reader
    fluids: Reader
    belts: Reader
    rigid_flags: Reader
    fluid_inputs: Reader | None = None


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m ora.environment.backends.robodojo.physics_scene_probe",
        description="Hold one RoboDojo scene and record its physics (qualification only).")
    parser.add_argument("--config", type=Path, required=True, help="worker configuration JSON from the launcher")
    parser.add_argument("--output", type=Path, required=True, help="probe cell directory")
    parser.add_argument("--ticks", type=_positive, required=True, help="native actions to hold")
    parser.add_argument("--sample-every", type=_positive, default=5, help="sample interval in native actions")
    parser.add_argument("--probe-only-native-action-limit", type=_positive, default=None,
                        help="admit an unadmitted task into this process only, with its upstream step_lim")
    parser.add_argument("--in-place-restart", action="store_true",
                        help="also restart round 1 in place and read rigid-object physics flags")
    return parser.parse_args(argv)


def _floats(value: Any) -> list[float]:
    """A tensor, array or any non-string sequence as plain floats (CPU copy for a torch tensor).

    Upstream keeps layout values such as a liquid's ``init_pos`` as OmegaConf
    ``ListConfig``, a ``Sequence`` that is neither a list nor a tuple.
    """
    for name in ("detach", "cpu"):
        if hasattr(value, name):
            value = getattr(value, name)()
    if hasattr(value, "tolist"):
        value = value.tolist()
    flat: list[float] = []
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            stack.extend(reversed(item))
        elif hasattr(item, "__len__") and hasattr(item, "__getitem__") and not isinstance(item, (str, bytes)):
            # pxr Gf vectors (a belt's Vec3f velocity, a bound's Vec3d corner)
            # index like sequences but are not registered as one.
            stack.extend(reversed([item[index] for index in range(len(item))]))
        else:
            flat.append(float(item))
    return flat


def _safe(reader: Reader, session: Any) -> Any:
    try:
        return reader(session)
    except Exception as exc:  # noqa: BLE001 - a reader failure is evidence, not a probe failure
        return {"error": f"{type(exc).__name__}: {exc}"}


# --- live readers (RoboDojo interpreter only) ------------------------------------------------------

def physics_mode_readback(session: Any) -> dict[str, Any]:
    """What device and pipeline PhysX and IsaacLab actually run with."""
    environment = session.environment
    record: dict[str, Any] = {
        "isaaclab_sim_device": str(getattr(getattr(getattr(environment, "sim", None), "sim", None), "device", None)
                                   or getattr(environment, "device", None)),
        "env_device": str(getattr(environment, "device", None)),
        "use_fabric": getattr(environment, "use_fabric", None),
    }
    try:
        import carb

        settings = carb.settings.get_settings()
        record.update(suppress_readback=settings.get("/physics/suppressReadback"),
                      fabric_enabled_setting=settings.get("/physics/fabricEnabled"),
                      cuda_device_setting=settings.get("/physics/cudaDevice"),
                      update_to_usd=settings.get("/physics/updateToUsd"),
                      update_particles_to_usd=settings.get("/physics/updateParticlesToUsd"))
    except Exception as exc:  # noqa: BLE001
        record["settings_error"] = f"{type(exc).__name__}: {exc}"
    try:
        from omni.physx import get_physx_interface

        record["overwrite_gpu_setting"] = get_physx_interface().get_overwrite_gpu_setting()
        record["overwrite_gpu_setting_meaning"] = "-1 schema, 0 force CPU, 1 force GPU"
    except Exception as exc:  # noqa: BLE001
        record["overwrite_gpu_setting_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import omni.usd
        from pxr import PhysxSchema, UsdPhysics

        scenes = []
        for prim in omni.usd.get_context().get_stage().Traverse():
            if prim.IsA(UsdPhysics.Scene):
                api = PhysxSchema.PhysxSceneAPI(prim)
                scenes.append({"path": str(prim.GetPath()),
                               "enable_gpu_dynamics": api.GetEnableGPUDynamicsAttr().Get(),
                               "broadphase": str(api.GetBroadphaseTypeAttr().Get()),
                               "solver": str(api.GetSolverTypeAttr().Get())})
        record["physics_scenes"] = scenes
    except Exception as exc:  # noqa: BLE001
        record["physics_scenes_error"] = f"{type(exc).__name__}: {exc}"
    try:
        from isaacsim.core.simulation_manager import SimulationManager

        record["simulation_manager_physics_sim_device"] = str(SimulationManager.get_physics_sim_device())
    except Exception as exc:  # noqa: BLE001
        record["simulation_manager_error"] = f"{type(exc).__name__}: {exc}"
    return record


def _layout(session: Any) -> Any:
    return session.environment.scene_manager.layout_manager


def object_positions(session: Any) -> dict[str, Any]:
    """Every layout instance's world position (root pose), keyed by instance name."""
    layout = _layout(session)
    objects: dict[str, Any] = {}
    for name, kind in sorted(layout.instance_type_by_env[0].items()):
        entry: dict[str, Any] = {"kind": str(kind)}
        try:
            obj = layout.get_scene_object(0, name)
            position = None
            if obj is not None and hasattr(obj, "get_world_pose"):
                position = obj.get_world_pose()[0]
            if position is None:
                position = layout.get_instance_pose(env_idx=0, inst_name=name, relative=False)[0]
            entry["position_m"] = _floats(position)[:3]
        except Exception as exc:  # noqa: BLE001
            entry["error"] = f"{type(exc).__name__}: {exc}"
        objects[str(name)] = entry
    return objects


def articulation_joints(session: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, obj in session.environment.scene_manager._articulation_objects[0].items():
        result[str(name)] = {"dof_names": [str(item) for item in getattr(obj, "dof_names", []) or []],
                             "joint_positions": _floats(obj.get_joint_positions())}
    return result


def fluid_particles(session: Any) -> dict[str, Any]:
    fluids = getattr(session.environment.scene_manager, "_fluid_objects", None) or [{}]
    result: dict[str, Any] = {}
    for name, obj in fluids[0].items():
        positions = obj.get_particle_positions()[0]
        values = _floats(positions)
        rows = [values[index:index + 3] for index in range(0, len(values) - len(values) % 3, 3)]
        finite = all(math.isfinite(value) for value in values)
        entry: dict[str, Any] = {"count": len(rows), "finite": finite,
                                 "init_position_m": _floats(getattr(obj, "init_pos", []))[:3]}
        if rows and finite:
            entry["local_min_m"] = [min(row[axis] for row in rows) for axis in range(3)]
            entry["local_max_m"] = [max(row[axis] for row in rows) for axis in range(3)]
        result[str(name)] = entry
    return result


def _rows(values: list[float]) -> list[list[float]]:
    return [values[index:index + 3] for index in range(0, len(values) - len(values) % 3, 3)]


def fluid_evaluator_inputs(session: Any) -> dict[str, Any]:
    """The raw inputs of the official fluid predicate, for a read-only offline replay; empty without a liquid."""
    scene = session.environment.scene_manager
    fluids = (getattr(scene, "_fluid_objects", None) or [{}])[0]
    if not fluids:
        return {}
    layout = scene.layout_manager
    result: dict[str, Any] = {"env_origin_m": _floats(scene.env_origins[0])[:3], "fluids": {}, "containers": {}}
    for record in layout.get_layout_records(0, "Fluid"):
        obj = fluids.get(record["inst_name"])
        if obj is None:
            continue
        result["fluids"][str(record.get("label"))] = {
            "inst_name": str(record["inst_name"]),
            "local_positions_m": _rows(_floats(obj.get_particle_positions()[0])),
            "init_position_m": _floats(obj.init_pos)[:3], "init_orientation_wxyz": _floats(obj.init_ori)[:4]}
    for section in CONTAINER_SECTIONS:
        for record in layout.get_layout_records(0, section):
            name = str(record["inst_name"])
            entry: dict[str, Any] = {"inst_name": name, "kind": section}
            try:
                position, orientation = layout.get_instance_pose(env_idx=0, inst_name=name)
                vertices = layout.get_instance_bbox_vertices(inst_name=name, env_idx=0)
                entry.update(position_m=_floats(position)[:3], orientation_wxyz=_floats(orientation)[:4],
                             bbox_vertices_m=None if vertices is None else _rows(_floats(vertices)))
            except Exception as exc:  # noqa: BLE001 - one unreadable instance is evidence, not a probe failure
                entry["error"] = f"{type(exc).__name__}: {exc}"
            result["containers"][str(record.get("label"))] = entry
    return result


def conveyor_belts(session: Any) -> list[dict[str, Any]]:
    """Every PhysxSurfaceVelocityAPI prim, its velocity and world-aligned bounds."""
    import omni.usd
    from pxr import Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    belts = []
    for prim in stage.Traverse():
        if "PhysxSurfaceVelocityAPI" not in [str(item) for item in prim.GetAppliedSchemas()]:
            continue
        velocity = prim.GetAttribute("physxSurfaceVelocity:surfaceVelocity")
        box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        belts.append({"path": str(prim.GetPath()),
                      "surface_velocity": _floats(velocity.Get()) if velocity and velocity.Get() is not None else None,
                      "world_aabb_min": _floats(box.GetMin()), "world_aabb_max": _floats(box.GetMax())})
    return belts


def rigid_physics_flags(session: Any) -> dict[str, Any]:
    """Gravity and collision flags under each rigid instance's prim (the CPU restart hazard)."""
    import omni.usd
    from pxr import Usd

    stage = omni.usd.get_context().get_stage()
    layout = _layout(session)
    result: dict[str, Any] = {}
    for name, kind in sorted(layout.instance_type_by_env[0].items()):
        if str(kind) != "rigid":
            continue
        obj = layout.get_scene_object(0, name)
        path = getattr(obj, "prim_path", None) or getattr(obj, "_prim_path", None)
        prim = stage.GetPrimAtPath(str(path)) if path else None
        if prim is None or not prim.IsValid():
            result[str(name)] = {"error": f"no prim for {path!r}"}
            continue
        gravity, collision = [], []
        for item in Usd.PrimRange(prim):
            disabled = item.GetAttribute("physxRigidBody:disableGravity")
            enabled = item.GetAttribute("physics:collisionEnabled")
            if disabled and disabled.Get() is not None:
                gravity.append(bool(disabled.Get()))
            if enabled and enabled.Get() is not None:
                collision.append(bool(enabled.Get()))
        result[str(name)] = {"disable_gravity": gravity, "collision_enabled": collision}
    return result


def default_readers() -> Readers:
    return Readers(physics_mode=physics_mode_readback, objects=object_positions,
                   articulations=articulation_joints, fluids=fluid_particles, belts=conveyor_belts,
                   rigid_flags=rigid_physics_flags, fluid_inputs=fluid_evaluator_inputs)


# --- probe -----------------------------------------------------------------------------------------

def _native_action(session: Any) -> int | None:
    try:
        return int(session.environment.take_action_cnt[0])
    except Exception:  # noqa: BLE001
        return None


def take_sample(session: Any, readers: Readers, *, tick: int) -> dict[str, Any]:
    return {"tick": tick, "native_action": _native_action(session),
            "physics_steps": getattr(session, "sim_steps", None),
            "objects": _safe(readers.objects, session), "articulations": _safe(readers.articulations, session),
            "fluids": _safe(readers.fluids, session)}


def _hold(session: Any, readers: Readers, *, ticks: int, sample_every: int,
          first_tick: int = 0) -> tuple[list[dict[str, Any]], int, bool]:
    samples, held = [], 0
    for tick in range(1, ticks + 1):
        if session.is_terminal:
            break
        session._conformance_tick({})
        held = tick
        if tick % sample_every == 0 or tick == ticks:
            samples.append(take_sample(session, readers, tick=first_tick + tick))
    if held and (not samples or samples[-1]["tick"] != first_tick + held):
        samples.append(take_sample(session, readers, tick=first_tick + held))
    return samples, held, held < ticks


def run_probe(session: Any, *, ticks: int, sample_every: int, in_place_restart: bool = False,
              readers: Readers | None = None) -> dict[str, Any]:
    """Reset, read the physics mode, hold ``ticks`` native actions while sampling, optionally restart in place."""
    readers = readers or default_readers()
    started = time.time()
    reset = session.start()
    report: dict[str, Any] = {
        "start_s": round(time.time() - started, 3),
        "reset_native_action_limit": (reset or {}).get("native_action_limit") if isinstance(reset, Mapping) else None,
        "physics_mode": _safe(readers.physics_mode, session), "belts": _safe(readers.belts, session),
    }
    before_flags = _safe(readers.rigid_flags, session) if in_place_restart else None
    samples = [take_sample(session, readers, tick=0)]
    if readers.fluid_inputs is not None:
        report["fluid_evaluator_inputs"] = {"reset": _safe(readers.fluid_inputs, session)}
    held_samples, held, early = _hold(session, readers, ticks=ticks, sample_every=sample_every)
    report.update(samples=samples + held_samples, ticks_held=held, terminal_before_hold_end=early,
                  hold_s=round(time.time() - started, 3))
    if readers.fluid_inputs is not None:
        report["fluid_evaluator_inputs"]["end"] = _safe(readers.fluid_inputs, session)
    if in_place_restart:
        session.restart_round(1, seed=0)
        after_flags = _safe(readers.rigid_flags, session)
        after, _, _ = _hold(session, readers, ticks=RESTART_HOLD_TICKS, sample_every=RESTART_HOLD_TICKS)
        report["in_place_restart"] = {
            "before": before_flags, "after": after_flags,
            "samples_after": [take_sample(session, readers, tick=0), *after],
            "claim_boundary": ("report only: whether rigid objects keep gravity and collision across an in-place "
                               "round restart; formal CPU sessions never restart in place"),
        }
    return report


def _write_report(output: Path, report: Mapping[str, Any]) -> None:
    (output / WORKER_REPORT_NAME).write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
                                             encoding="utf-8")


def _bounded_close(session: Any, timeout_s: float = CLOSE_TIMEOUT_S) -> bool:
    thread = threading.Thread(target=session.close, daemon=True)
    thread.start()
    thread.join(timeout_s)
    return not thread.is_alive()


def _session(config: Mapping[str, Any], output_root: Path) -> Any:
    from .session import RobodojoSession

    return RobodojoSession(config=config, output_root=output_root)


def run_worker(config: Mapping[str, Any], output: Path, *, ticks: int, sample_every: int,
               in_place_restart: bool = False, probe_only_native_action_limit: int | None = None,
               session_factory: Callable[..., Any] | None = None,
               readers: Readers | None = None) -> dict[str, Any]:
    """Start the plant, probe, record and close; a probe failure is recorded, never raised."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"schema": WORKER_REPORT_SCHEMA, "task": config.get("task"),
                              "physics_device": config.get("physics_device"), "ticks_requested": ticks,
                              "sample_every": sample_every, "error": None, "completed": False,
                              "ticks_held": 0, "terminal_before_hold_end": False, "samples": [],
                              "visibility": VISIBILITY, "probe_only_native_action_limit": None}
    session = None
    from ora.environment.benchmarks.robodojo import NATIVE_ACTION_LIMITS

    with probe_only_admission(config.get("task"), probe_only_native_action_limit,
                              NATIVE_ACTION_LIMITS) as admission:
        report["probe_only_native_action_limit"] = admission
        try:
            session = (session_factory or _session)(config=config, output_root=Path(config["output_root"]))
            report.update(run_probe(session, ticks=ticks, sample_every=sample_every,
                                    in_place_restart=in_place_restart, readers=readers))
            report["completed"] = True
        except Exception as exc:  # noqa: BLE001 - recorded for the host report, then the plant is closed
            report.update(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        finally:
            _write_report(output, report)
            if session is not None:
                report["closed_within_timeout"] = _bounded_close(session)
            _write_report(output, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    report = run_worker(config, arguments.output, ticks=arguments.ticks, sample_every=arguments.sample_every,
                        in_place_restart=arguments.in_place_restart,
                        probe_only_native_action_limit=arguments.probe_only_native_action_limit)
    return 0 if report["error"] is None else 1


__all__ = [
    "CONTAINER_SECTIONS", "Readers", "VISIBILITY", "WORKER_REPORT_NAME", "WORKER_REPORT_SCHEMA", "default_readers",
    "fluid_evaluator_inputs", "main",
    "parse_arguments", "probe_only_admission", "run_probe", "run_worker", "take_sample",
]


if __name__ == "__main__":  # pragma: no cover - runs in the RoboDojo interpreter
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Kit keeps non-daemon threads after close; the official worker exits hard too.
    os._exit(code)
