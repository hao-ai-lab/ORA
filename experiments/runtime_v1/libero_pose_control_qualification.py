#!/usr/bin/env python3
"""Source-bound real LIBERO/Panda qualification for Pose/Frame Contract v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from ora.contracts.pose import compile_pose_request
from ora.environment.backends.libero_pro.session import (
    LiberoProSession,
    libero_osc_action_contract,
)
from ora.environment.workspace_profiles import resolve_workspace_profile


SCHEMA = "ora.runtime-v1-libero-pose-control-qualification.v1"
TOPDOWN_XYZW = [math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0]
SIDE_XYZW = [0.5, 0.5, -0.5, -0.5]


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments], check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _quat_multiply(left: list[float], right: list[float]) -> list[float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return [
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ]


def _z_rotation(angle_rad: float) -> list[float]:
    return [0.0, 0.0, math.sin(angle_rad / 2), math.cos(angle_rad / 2)]


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _controller_qualification() -> dict[str, Any]:
    """Rebuild the same source binding required at session reset."""

    import numpy as np
    import robosuite
    import robosuite.models.grippers.panda_gripper as panda_gripper_module
    from robosuite.controllers import load_controller_config
    from robosuite.models.grippers import PandaGripper

    contract = libero_osc_action_contract(
        load_controller_config(default_controller="OSC_POSE")
    )
    robosuite_module = Path(str(robosuite.__file__)).resolve()
    robosuite_root = robosuite_module.parent
    controller_config = robosuite_root / "controllers/config/osc_pose.json"
    panda_module = Path(str(panda_gripper_module.__file__)).resolve()
    panda_xml = robosuite_root / "models/assets/grippers/panda_gripper.xml"
    gripper = object.__new__(PandaGripper)
    gripper.current_action = np.zeros(1, dtype=np.float64)
    close = PandaGripper.format_action(gripper, np.asarray([1.0])).tolist()
    hold = PandaGripper.format_action(gripper, np.asarray([0.0])).tolist()
    reopen = PandaGripper.format_action(gripper, np.asarray([-1.0])).tolist()
    if (
        any(abs(close[index] - hold[index]) > 1e-9 for index in range(2))
        or not close[0] < 0 < close[1]
        or any(abs(value) > 1e-9 for value in reopen)
    ):
        raise RuntimeError("Panda actuator-target hold qualification failed")
    return {
        "schema": "ora.libero-controller-qualification.v1",
        "contract_sha256": contract["contract_sha256"],
        "input_min": contract["input_min"],
        "input_max": contract["input_max"],
        "output_min": contract["output_min"],
        "output_max": contract["output_max"],
        "servo_clip": contract["servo_clip"],
        "gripper_action_probe": {
            "close": close, "zero_hold": hold, "reopen": reopen,
        },
        "binding": {
            "robosuite_module": str(robosuite_module),
            "robosuite_module_sha256": _file_sha256(robosuite_module),
            "controller_config_file": str(controller_config.resolve()),
            "controller_config_sha256": _file_sha256(controller_config),
            "panda_gripper_module": str(panda_module),
            "panda_gripper_module_sha256": _file_sha256(panda_module),
            "panda_gripper_file": str(panda_xml.resolve()),
            "panda_gripper_sha256": _file_sha256(panda_xml),
        },
    }


def _session(args: argparse.Namespace, scenario: str) -> LiberoProSession:
    profile = resolve_workspace_profile("libero_pro_panda_tabletop_v1")
    return LiberoProSession(
        rlinf_root=args.rlinf_root,
        libero_pro_root=args.libero_pro_root,
        data_root=args.data_root,
        output_root=args.output_root / scenario,
        task="libero_object_swap-t00-pose-frame-contract-v2",
        seed=0,
        suite="libero_object_swap",
        task_id=0,
        expected_task_language="pick up the alphabet soup and place it in the basket",
        required_rlinf_revision=args.required_rlinf_revision,
        required_libero_pro_revision=args.required_libero_pro_revision,
        required_data_revision=args.required_data_revision,
        workspace_bounds_m=profile["bounds_m"],
        workspace_profile=profile,
        qualified_controller_binding=args.controller_qualification,
        max_sim_steps=4000,
        servo_max_steps=80,
    )


def _move(
    session: LiberoProSession, *, action_id: str, request: dict[str, Any],
) -> dict[str, Any]:
    position, orientation, _gripper = session._current_state()
    current = position.tolist() + orientation.tolist()
    compiled = compile_pose_request(
        request, current_pose_xyz_xyzw=current,
        controlled_point="robosuite_grip_site",
    )
    target = compiled["target_pose_xyz_xyzw"]
    receipt = session.execute({
        "schema": "ora-action.v3",
        "observation_id": f"qualification:{action_id}",
        "action_id": action_id,
        "action": "move_pose",
        "bound_observation_sha256": "qualification-direct-state-readback",
        "arguments": {
            "arm": "right",
            "position_m": target[:3],
            "orientation_xyzw": target[3:],
        },
    })
    return {
        "authored_request": request,
        "compiled_pose": compiled,
        "receipt": receipt,
        "compiled_hash_identity": (
            receipt.get("compiled_pose_sha256")
            == compiled["compiled_pose_sha256"]
        ),
    }


def _gripper(
    session: LiberoProSession, *, action_id: str, target: float,
) -> dict[str, Any]:
    return session.execute({
        "schema": "ora-action.v3",
        "observation_id": f"qualification:{action_id}",
        "action_id": action_id,
        "action": "set_gripper",
        "bound_observation_sha256": "qualification-direct-state-readback",
        "arguments": {"arm": "right", "gripper": target},
    })


def _passed_motion(record: dict[str, Any]) -> bool:
    receipt = record["receipt"]
    return bool(
        record["compiled_hash_identity"]
        and receipt.get("motion_status") == "reached"
        and receipt.get("control_health", {}).get("status") == "healthy"
        and receipt.get("steps_used", 80) < 80
    )


def _run_fresh(
    args: argparse.Namespace, scenario: str,
    body: Callable[[LiberoProSession], dict[str, Any]],
) -> dict[str, Any]:
    session = _session(args, scenario)
    try:
        reset = session.start()
        result = body(session)
        return {"source_manifest": reset["source_manifest"], **result}
    finally:
        session.close()


def _free_translation(args: argparse.Namespace) -> dict[str, Any]:
    def body(session: LiberoProSession) -> dict[str, Any]:
        move = _move(
            session, action_id="free-translation",
            request={
                "translation_delta_mm": [20, 0, 20],
                "translation_delta_frame": "world",
            },
        )
        return {"passed": _passed_motion(move), "moves": [move]}
    return _run_fresh(args, "free-translation", body)


def _unequal_axis_long_translation(args: argparse.Namespace) -> dict[str, Any]:
    """Qualify curved endpoint-OSC progress without a straight-line guard."""

    def body(session: LiberoProSession) -> dict[str, Any]:
        stage = _move(
            session, action_id="unequal-axis-stage",
            request={"target_xyz_world_m": [-0.35, 0.15, 0.24]},
        )
        move = _move(
            session, action_id="unequal-axis-target",
            request={"target_xyz_world_m": [-0.40, 0.0, 0.30]},
        )
        tracking_ref = move["receipt"]["tracking_evidence"]
        tracking_path = session.output_root / tracking_ref["path"]
        tracking = json.loads(tracking_path.read_text(encoding="utf-8"))
        max_cross_axis_drift_m = max(
            float(sample["cross_axis_drift_m"])
            for sample in tracking["samples"]
        )
        guard = move["receipt"]["control_health"]["tracking_guard"]
        return {
            "passed": (
                _passed_motion(stage)
                and _passed_motion(move)
                and max_cross_axis_drift_m > 0.02
                and guard["outcome"] == "not_triggered"
                and guard["thresholds"]["cross_axis_drift"]
                == "telemetry_only_no_stop_authority"
            ),
            "expected": (
                "an unequal-axis endpoint move may curve more than 2 cm from "
                "the geometric line while continuing to the target"
            ),
            "max_cross_axis_drift_m": max_cross_axis_drift_m,
            "moves": [stage, move],
        }

    return _run_fresh(args, "unequal-axis-long-translation", body)


def _pure_rotations(args: argparse.Namespace) -> dict[str, Any]:
    def body(session: LiberoProSession) -> dict[str, Any]:
        moves = []
        for label, angle in (("small", 0.10), ("quarter", math.pi / 2), ("near-pi", math.radians(175))):
            _position, orientation, _gripper_state = session._current_state()
            target = _quat_multiply(_z_rotation(angle), orientation.tolist())
            moves.append(_move(
                session, action_id=f"rotation-{label}",
                request={"target_orientation_world_xyzw": target},
            ))
        near_pi = moves[-1]
        near_receipt = near_pi["receipt"]
        near_guard = near_receipt["control_health"]["tracking_guard"]
        return {
            "passed": (
                all(_passed_motion(move) for move in moves[:2])
                and near_pi["compiled_hash_identity"]
                and near_pi["compiled_pose"]["rotation_angle_deg"] > 170
                and near_receipt["motion_status"] in {"reached", "not_reached"}
                and near_receipt["control_health"]["status"] != "fault"
                and near_guard["outcome"] != "terminal_fault"
                and near_receipt["motion_realization"][
                    "realized_translation_distance_m"
                ] <= (
                    near_guard["thresholds"]["position_error_growth_stop_m"]
                    + near_guard["thresholds"]["gross_single_tick_translation_m"]
                    + 1e-6
                )
                and near_receipt["steps_used"] <= 80
            ),
            "expected": (
                "small and 90-degree rotations reach tolerance; a legal "
                "near-180-degree request may return truthful bounded not_reached"
            ),
            "moves": moves,
        }
    return _run_fresh(args, "pure-rotations", body)


def _micro_translation_large_rotation(args: argparse.Namespace) -> dict[str, Any]:
    def body(session: LiberoProSession) -> dict[str, Any]:
        _position, orientation, _gripper_state = session._current_state()
        target = _quat_multiply(
            _z_rotation(math.radians(175)), orientation.tolist(),
        )
        move = _move(
            session, action_id="micro-plus-near-pi",
            request={
                "translation_delta_mm": [10, 0, 0],
                "translation_delta_frame": "world",
                "target_orientation_world_xyzw": target,
            },
        )
        compiled = move["compiled_pose"]
        receipt = move["receipt"]
        guard = receipt["control_health"]["tracking_guard"]
        return {
            "passed": (
                move["compiled_hash_identity"]
                and math.isclose(
                    compiled["translation_distance_m"], 0.01, abs_tol=1e-9,
                )
                and compiled["rotation_angle_deg"] > 170
                and receipt["motion_status"] in {"reached", "not_reached"}
                and receipt["control_health"]["status"] != "fault"
                and guard["outcome"] != "terminal_fault"
                and receipt["motion_realization"][
                    "realized_translation_distance_m"
                ] <= 0.01 + guard["thresholds"]["travel_overshoot_stop_m"] + 1e-6
                and receipt["steps_used"] <= 80
            ),
            "expected": (
                "a legal micro-translation plus near-180-degree request remains "
                "spatially bounded and reports reached or truthful not_reached"
            ),
            "moves": [move],
        }
    return _run_fresh(args, "micro-translation-large-rotation", body)


def _gripper_target_hold(args: argparse.Namespace) -> dict[str, Any]:
    def body(session: LiberoProSession) -> dict[str, Any]:
        opened = _gripper(session, action_id="open", target=1.0)
        open_move = _move(
            session, action_id="open-target-hold",
            request={"translation_delta_mm": [0, 0, 15]},
        )
        closed = _gripper(session, action_id="close", target=0.0)
        closed_move = _move(
            session, action_id="closed-target-hold",
            request={"translation_delta_mm": [0, 0, -15]},
        )
        moves = [open_move, closed_move]
        invariants = [
            move["receipt"]["command_invariant"] for move in moves
        ]
        return {
            "passed": (
                opened.get("motion_status") == "reached"
                and closed.get("motion_status") == "reached"
                and all(_passed_motion(move) for move in moves)
                and all(item.get("verified") is True for item in invariants)
                and all(
                    item.get("held_dimension") == "gripper_actuator_target"
                    for item in invariants
                )
            ),
            "set_gripper_receipts": [opened, closed],
            "moves": moves,
        }
    return _run_fresh(args, "gripper-target-hold", body)


def _near_table(args: argparse.Namespace) -> dict[str, Any]:
    def body(session: LiberoProSession) -> dict[str, Any]:
        position, _orientation, _gripper_state = session._current_state()
        raise_move = _move(
            session, action_id="near-table-raise",
            request={"target_xyz_world_m": [float(position[0]), float(position[1]), 0.30]},
        )
        down = _move(
            session, action_id="near-table-down",
            request={
                "target_xyz_world_m": [float(position[0]), float(position[1]), 0.12],
                "target_orientation_world_xyzw": TOPDOWN_XYZW,
            },
        )
        return {
            "passed": all(
                move["compiled_hash_identity"]
                and move["receipt"].get("control_health", {}).get("status")
                in {"healthy", "degraded"}
                and move["receipt"].get("steps_used", 80) < 80
                for move in (raise_move, down)
            ),
            "moves": [raise_move, down],
        }
    return _run_fresh(args, "near-table", body)


def _recoverable_guard(args: argparse.Namespace) -> dict[str, Any]:
    def body(session: LiberoProSession) -> dict[str, Any]:
        _position, orientation, _gripper_state = session._current_state()
        stage = _move(
            session, action_id="guard-stage",
            request={
                "target_xyz_world_m": [-0.09, -0.045, 0.16],
                "target_orientation_world_xyzw": orientation.tolist(),
            },
        )
        turn = _move(
            session, action_id="guard-turn",
            request={
                "target_xyz_world_m": [-0.04, -0.045, 0.10],
                "target_orientation_world_xyzw": SIDE_XYZW,
            },
        )
        recovery = _move(
            session, action_id="guard-recovery",
            request={
                "target_xyz_world_m": [-0.12, -0.06, 0.20],
                "target_orientation_world_xyzw": TOPDOWN_XYZW,
            },
        )
        moves = [stage, turn, recovery]
        guards = [
            move["receipt"].get("control_health", {}).get(
                "tracking_guard", {}
            )
            for move in moves
        ]
        return {
            "passed": (
                all(move["compiled_hash_identity"] for move in moves)
                and any(
                    guard.get("outcome") == "recoverable_stop"
                    for guard in guards
                )
                and all(
                    guard.get("outcome") != "terminal_fault"
                    for guard in guards
                )
                and session.status().get("terminal") is False
                and recovery["receipt"].get("steps_used", 80) < 80
            ),
            "expected": (
                "the source-bound large-orientation-change sequence must "
                "exercise a recoverable guard stop without ending the session"
            ),
            "moves": moves,
        }
    return _run_fresh(args, "recoverable-guard", body)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--rlinf-root", type=Path, required=True)
    parser.add_argument("--libero-pro-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--required-rlinf-revision", required=True)
    parser.add_argument("--required-libero-pro-revision", required=True)
    parser.add_argument("--required-data-revision", required=True)
    args = parser.parse_args()
    args.project_root = args.project_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.report = args.report.expanduser().resolve()
    started = time.time()
    source = {
        "commit": _git(args.project_root, "rev-parse", "HEAD"),
        "dirty": bool(_git(
            args.project_root, "status", "--porcelain", "--untracked-files=normal",
        )),
    }
    if source["dirty"]:
        report = {
            "schema": SCHEMA, "passed": False, "source": source,
            "error": "qualification requires a clean committed source",
        }
    else:
        try:
            args.controller_qualification = _controller_qualification()
            scenarios = {
                "free_translation": _free_translation(args),
                "unequal_axis_long_translation": (
                    _unequal_axis_long_translation(args)
                ),
                "pure_rotations": _pure_rotations(args),
                "micro_translation_large_rotation": _micro_translation_large_rotation(args),
                "near_table": _near_table(args),
                "gripper_target_hold": _gripper_target_hold(args),
                "recoverable_guard": _recoverable_guard(args),
            }
            report = {
                "schema": SCHEMA,
                "created_unix_s": time.time(),
                "duration_s": time.time() - started,
                "source": source,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "passed": all(item["passed"] for item in scenarios.values()),
                "scenarios": scenarios,
                "terminal_fault_authority": {
                    "live_fault_was_not_deliberately_induced": True,
                    "deterministic_qualification": (
                        "tests/test_runtime_v1.py::"
                        "test_libero_control_health_fault_stops_after_gross_divergence"
                    ),
                },
            }
        except Exception as exc:
            report = {
                "schema": SCHEMA,
                "created_unix_s": time.time(),
                "duration_s": time.time() - started,
                "source": source,
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
    _atomic_json(args.report, report)
    print(json.dumps({
        "schema": report["schema"],
        "passed": report["passed"],
        "duration_s": report.get("duration_s"),
        "source": report.get("source"),
        "scenarios": {
            name: value.get("passed")
            for name, value in report.get("scenarios", {}).items()
        },
        **({"error": report["error"]} if "error" in report else {}),
    }, ensure_ascii=False, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
