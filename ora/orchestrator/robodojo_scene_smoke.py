"""Bounded RoboDojo scene smoke on the runtime a default RunSpec resolves to.

One worker on one qualified slot: device identity, loaded public geometry and
RGB-D, read-only current-pose IK and one held current-pose composite per arm,
then a same-scene round reset -- or, for a layout-scheduled session, one
simulation relaunch per later scheduled layout with its own checks
(``robodojo_layout_smoke``). No Provider, terminal sweep or behavioral claim;
exact failures survive in the report.

Maintained copy (integration 0062 U12) of the earlier B2 RoboDojo
scene smoke; the plan and report
keep that smoke's schemas and fields, so ``robodojo_revalidation`` admits them
unchanged. The runtime is ``robodojo_spec.qualification_runtime``, the one a
RunSpec built by ``robodojo_spec`` resolves to, so a lane gate made from this
smoke admits that plan. Evidence goes to ``outputs/<experiment>/qualification/
robodojo-scene-smoke-*``; ``robodojo_lane_qualification revalidate`` reads it.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence

from ora.environment.factory import build_environment_session
from ora.orchestrator.action_executor import execute_runtime_action
from ora.orchestrator.cli import (
    assign_resource_lease, claim_resource_lease, release_resource_lease,
)
from ora.orchestrator.paths import PROJECT_ROOT, experiment_workspace
from ora.orchestrator.robodojo_card_qualification import _prepare, _sensor_evidence
from ora.orchestrator.spec import json_sha256

PLAN_SCHEMA = "ora.robodojo-b2-smoke-plan.v1"
REPORT_SCHEMA = "ora.robodojo-b2-smoke.v1"
OWNER_PREFIX = "robodojo-scene-smoke"
ARMS = ("left", "right")
#: Files whose bytes are the smoke's checker, hashed into its plan.
CHECKER_FILES = (
    "ora/orchestrator/robodojo_scene_smoke.py",
    "ora/orchestrator/robodojo_layout_smoke.py",
    "ora/environment/benchmarks/robodojo_layout_reach.py",
)


def _file_sha256(relative: str) -> str:
    return hashlib.sha256((PROJECT_ROOT / relative).read_bytes()).hexdigest()

#: ``checks.round_reset`` of an unscheduled CPU-physics smoke: not run, because
#: a CPU session is single-round and never restarts in place.
CPU_SINGLE_ROUND_RESET = "not_applicable_cpu_single_round"


def smoke_plan(runtime: Mapping[str, Any], prepared: Mapping[str, Any]) -> dict[str, Any]:
    """The content-addressed plan the report and the lane gate bind."""
    schedule = runtime.get("layout_schedule")
    layout_ids = None if schedule is None else [entry["layout_id"] for entry in schedule["rounds"]]
    checker = {path: _file_sha256(path) for path in CHECKER_FILES}
    plan = {
        "schema": PLAN_SCHEMA, "runtime": dict(runtime), "harness_commit": prepared["commit"],
        "worker_kernel_sha256": prepared["kernel"], "purpose": "launch_smoke",
        "coverage_policy": "representative-v1",
        "dependency_closure": {"worker_kernel": "orchestrator.robodojo_qualification.WORKER_KERNEL_PATHS",
                               "case_runtime": dict(runtime), "checker_files": checker},
        "producer_dependency_hash": prepared["kernel"], "checker_hash": json_sha256(checker),
        "cases": [{"case_id": f"{runtime['task_variant']}-scene-sequence",
                   "case_key": json_sha256({"runtime": runtime, "kernel": prepared["kernel"]}),
                   "repeats": 1, "decision": "execute"}],
        "reuse": [], "parallel_execution": {"lane": runtime["resource_profile"], "cpu_count": os.cpu_count()},
        "estimated_wall_minutes": 4 if layout_ids is None else 4 + 2 * (len(layout_ids) - 1),
        "estimate_basis": "One worker startup, two held composites, a reset or one relaunch per later layout "
                          "(35-60 s each) and a bounded close.",
        "nondeterminism_evidence": None, "max_parent_actions": 2, "max_native_actions": 80,
        "task_solving": False,
    }
    if layout_ids is not None:
        plan["layout_schedule"] = layout_ids
    plan["plan_sha256"] = json_sha256(plan)
    return plan


def _worker_gpus(pid: int) -> list[str]:
    rows = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"],
                                   text=True, timeout=15)
    return sorted({row.split(",")[1].strip() for row in rows.splitlines() if row.split(",")[0].strip() == str(pid)})


def _held_composites(session: Any, report: dict[str, Any], root: Path, observation: Mapping[str, Any],
                     save) -> Mapping[str, Any]:
    """Read-only current-pose IK and one held current-pose target_state per arm."""
    for index, arm in enumerate(ARMS):
        state = observation["proprioception"]["end_effectors"][arm]
        pose = state["pose_xyz_xyzw"]
        align = session.align_plan({"schema": "ora.align-plan-request.v1",
                                    "target_ref": "smoke-current-pose", "approach_side": "top",
                                    "arm": arm, "target_poses_xyzw": [pose]})
        report.setdefault("alignment", {})[arm] = align
        save()
        assert align["candidates"][0]["reachable"] is True, f"{arm} current pose not reachable"
        action = {"schema": "ora-action.v4", "observation_id": observation["observation_id"],
                  "action_id": f"smoke-{index:04d}", "action": "target_state",
                  "bound_observation_sha256": json_sha256(observation),
                  "arguments": {"arm": arm, "pose": {"position_m": pose[:3], "orientation_xyzw": pose[3:]},
                                "gripper": state["gripper_command"]}}
        receipt = execute_runtime_action(session, action)
        report["actions"].append(receipt)
        save()
        assert [row["status"] for row in receipt["phase_receipts"]] == ["reached", "executed"], (
            f"{arm} composite phase did not finish")
        report["checks"][f"{arm}_composite"] = True
        observation, _ = session.observe(root / "observations")
    report["checks"]["read_only_alignment"] = True
    return observation


def smoke(task: str, *, gpu: str, experiment: str, resource_profile: str | None = None, rounds: int = 1,
          layout_schedule: Sequence[int] | str | None = None, layout_id: int | None = None) -> dict[str, Any]:
    """Run the bounded scene smoke of one lane; return its report (also written to disk).

    ``layout_id`` k > 0 smokes the one-round episode on official layout k;
    the reset's ``task_instruction`` is the official instruction of that
    layout, which the lane gate binds when the layout generates it.
    """
    from ora.orchestrator.robodojo_spec import qualification_runtime

    layout = {} if layout_id is None else {"layout_id": layout_id}
    runtime = qualification_runtime(task, gpu=gpu, resource_profile=resource_profile, rounds=rounds,
                                    layout_schedule=layout_schedule, **layout)
    prepared = _prepare(gpu, task, runtime, OWNER_PREFIX)
    root = experiment_workspace(experiment) / "qualification" / prepared["owner"]
    root.mkdir(parents=True)
    plan = smoke_plan(runtime, prepared)
    (root / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"plan": str(root / "plan.json"), "estimated_wall_minutes": plan["estimated_wall_minutes"]}),
          flush=True)
    owner = prepared["owner"]
    lease = claim_resource_lease(plan, owner, os.getpid())
    session = None
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA, "passed": False, "plan_sha256": plan["plan_sha256"],
        "source_commit": prepared["commit"], "environment": prepared["stack"],
        "nvidia_driver_version": prepared["driver_version"], "task": task, "gpu": gpu,
        "resource_profile": runtime["resource_profile"], "checks": {}, "actions": [],
        "physics_device": runtime["physics_device"],
        "claim_boundary": "Device/geometry/sequence smoke only; no official terminal, behavioral success or full "
                          "qualification claim." + ("" if runtime["physics_device"] == "cuda:0" else
                          " CPU physics: the in-place round_reset proves lifecycle counters and identities only, not"
                          " the physics of respawned rigid objects (gravity/collision may be disabled after an"
                          " in-place CPU restart)."),
    }
    if "layout_schedule" in plan:
        report["layout_schedule"] = list(plan["layout_schedule"])

    def save():
        (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    try:
        assign_resource_lease(lease, os.getpid())
        save()
        print(f"Starting {task} on GPU {gpu}: {root}", flush=True)
        session = build_environment_session(
            runtime_config=runtime, output_root=root / "environment", task=task, seed=0,
            max_sim_steps=min(1000, int(runtime["native_action_limit"]) * 10), action_codec="motion_action_v3")
        reset = dict(session.start())
        report["reset"] = reset
        save()
        worker_pid = session.delegate.process.pid
        devices = _worker_gpus(worker_pid)
        report["worker_residency"] = {"pid": worker_pid, "gpu_uuids": devices}
        assert devices == [reset["device_binding"]["selected_gpu_uuid"]], "CUDA/renderer/worker GPU mismatch"
        report["checks"]["device_identity"] = True
        observation, images = session.observe(root / "observations")
        report["sensors"] = _sensor_evidence(observation, images)
        assert observation.get("robot_geometry_verification", {}).get("passed") is True, (
            "loaded public geometry not verified")
        report["checks"]["loaded_geometry"] = True
        report["checks"]["rgbd"] = True
        report["ready_unix_s"] = time.time()
        save()
        _held_composites(session, report, root, observation, save)
        report["control_end_unix_s"] = time.time()
        if "layout_schedule" in runtime:
            from ora.orchestrator.robodojo_layout_smoke import scheduled_rounds
            scheduled_rounds(session, report, root, runtime, pid=worker_pid, save=save,
                             sensor_evidence=_sensor_evidence)
        elif runtime["physics_device"] != "cuda:0":
            # A CPU lane runs one round (rule robodojo_cpu_in_place_restart), so
            # its sessions never restart in place; upstream's CPU in-place
            # restart can leave respawned objects unsettled (2026-10-05:
            # fill_egg_holder, pour_liquid_into_cup). Nothing to qualify there.
            report["restart"] = None
            report["checks"]["round_reset"] = CPU_SINGLE_ROUND_RESET
        else:
            session.abort("qualification_boundary")
            restart = session.restart_round(1, seed=0)
            report["restart"] = restart
            assert restart["round_isolation"]["passed"] is True
            report["checks"]["round_reset"] = True
        report["passed"] = True
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        started = time.monotonic()
        try:
            if session is not None:
                session.close()
            report["close_elapsed_s"] = time.monotonic() - started
        except Exception as exc:
            report["passed"] = False
            report["close_error"] = str(exc)
        finally:
            save()
            release_resource_lease(owner, owner_pid=os.getpid(), claim_token=lease["claim_token"])
            print(json.dumps({"passed": report["passed"], "report": str(root / "report.json"),
                              "checks": report["checks"]}), flush=True)
    return report
