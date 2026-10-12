"""Synthetic RoboDojo native roots for the motion-metric and control-replay tests.

The writer reproduces the layout a sealed run keeps under ``native/``:
``reset.json`` (with its ``actual_source_fingerprint``), ``source-manifest.json``,
``receipts/action-NNNN.json`` and, per action, ``controller/action-NNNN.json``
(one sample per native tick) beside ``controller/action-NNNN.jsonl`` (one line
per physics step). Physics clocks are contiguous inside a round and jump at
every round start, exactly as the session's reset leaves them. Applied arm
targets default to the measured joints, so the commanded speed of a tick
equals its measured speed unless a test supplies its own.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

TRACE_SCHEMA = "ora.robodojo-controller-trace.v1"
RECEIPT_SCHEMA = "ora.robodojo-sparse-motion-receipt.v1"
ZERO = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
HOME_XYZ = {"left": (-0.3, -0.35, 1.0), "right": (0.3, -0.35, 1.0)}
IDENTITY_XYZW = (0.0, 0.0, 0.0, 1.0)
FINGERPRINT = {"loaded_files": {"Assets/Robots/x5/curobo.yml": "0" * 64}, "installed_packages": {"isaaclab": "1"},
               "resolved_environment_config_sha256": "1" * 64}
REVISIONS = {"robodojo": "r" * 40}


def state(q: Sequence[float] = ZERO, xyz: Sequence[float] | None = None, *, arm: str = "right",
          gripper: float = 1.0, target: str = "open") -> dict[str, Any]:
    position = list(xyz if xyz is not None else HOME_XYZ[arm])
    return {"arm_joint_qpos": [float(v) for v in q], "pose_xyz_xyzw": position + list(IDENTITY_XYZW),
            "gripper": gripper, "gripper_finger_qpos": [0.044 * gripper, 0.044 * gripper],
            "gripper_command_target": target,
            "gripper_feedback": {"requested_opening_normalized": 0.0 if target == "close" else 1.0}}


def still(arm: str, q: Sequence[float] = ZERO, xyz: Sequence[float] | None = None,
          **kwargs: Any) -> list[dict[str, Any]]:
    return [state(q, xyz, arm=arm, **kwargs) for _ in range(10)]


def joint_ramp(start: Sequence[float], per_step: Sequence[float]) -> list[list[float]]:
    """Ten physics-step joint vectors advancing ``per_step`` from ``start`` (exclusive)."""
    return [[s + (k + 1) * d for s, d in zip(start, per_step)] for k in range(10)]


def moving(arm: str, joints: Sequence[Sequence[float]], z: Sequence[float] | None = None,
           **kwargs: Any) -> list[dict[str, Any]]:
    base = HOME_XYZ[arm]
    heights = list(z) if z is not None else [base[2]] * 10
    return [state(q, (base[0], base[1], heights[k]), arm=arm, **kwargs) for k, q in enumerate(joints)]


def control(**targets: Sequence[float]) -> dict[str, Any]:
    return {f"{arm}_arm_joint_state": {"position": [float(v) for v in q]} for arm, q in targets.items()}


def tick(segment: str, control_dict: Mapping[str, Any], *, left: Sequence[Mapping[str, Any]] | None = None,
         right: Sequence[Mapping[str, Any]] | None = None, applied: Sequence[Mapping[str, Any]] | None = None,
         command: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """One native tick; ``command`` is the resolved SDK command journaled per step (default: the control)."""
    return {"segment": segment, "control": dict(control_dict),
            "left": list(left) if left is not None else still("left"),
            "right": list(right) if right is not None else still("right"),
            "applied": list(applied) if applied is not None else None,
            "command": dict(command) if command is not None else dict(control_dict)}


def segment_entry(name: str, native_actions: int, goal_z: float | None = None, arm: str = "right") -> dict[str, Any]:
    base = HOME_XYZ[arm]
    goal = None if goal_z is None else [base[0], base[1], goal_z, *IDENTITY_XYZW]
    return {"segment": name, "native_actions": native_actions, "target_pose_xyz_xyzw": goal}


def receipt(action: str, *, status: str = "reached", code: str | None = None,
            arguments: Mapping[str, Any] | None = None, arms: Mapping[str, Any] | None = None,
            **extra: Any) -> dict[str, Any]:
    arm_effects = {arm: {"segments": [], "interaction": None, "targeted": False} for arm in ("left", "right")}
    for arm, effect in (arms or {}).items():
        arm_effects[arm] = {**arm_effects[arm], "targeted": True, **dict(effect)}
    return {"schema": RECEIPT_SCHEMA, "action": action, "execution_status": status, "failure_code": code,
            "arguments": dict(arguments or {}), "arms": arm_effects,
            "target_arms": [arm for arm, value in arm_effects.items() if value["targeted"]], **extra}


def pose_target(xyz: Sequence[float]) -> dict[str, Any]:
    return {"position_m": [float(v) for v in xyz], "orientation_xyzw": list(IDENTITY_XYZW)}


def action(receipt_body: Mapping[str, Any], ticks: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    return {"receipt": dict(receipt_body), "ticks": list(ticks)}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _applied(item: Mapping[str, Any], step: int) -> dict[str, Any]:
    if item.get("applied") is not None:
        return dict(item["applied"][step])
    return {f"{arm}_arm_joint_state": list(item[arm][step]["arm_joint_qpos"]) for arm in ("left", "right")}


def _write_action(native: Path, sequence: int, round_index: int, body: Mapping[str, Any],
                  clock: int, native_action: int) -> tuple[int, int]:
    action_id = f"action-{sequence:04d}"
    _write_json(native / "receipts" / f"action-{sequence:04d}.json",
                {**body["receipt"], "action_id": action_id, "round_index": round_index,
                 "native_actions": len(body["ticks"])})
    samples, lines = [], []
    for item in body["ticks"]:
        native_action += 1
        for step in range(10):
            clock += 1
            lines.append({"action_id": action_id, "segment": item["segment"], "physics_clock": clock,
                          "native_action": native_action, "command": item["command"],
                          "applied_position_targets": _applied(item, step),
                          "measured": {"left": item["left"][step], "right": item["right"][step]}})
        samples.append({"native_action": native_action, "physics_steps": clock, "segment": item["segment"],
                        "control": item["control"],
                        "measured": {"left": item["left"][-1], "right": item["right"][-1]}})
    _write_json(native / "controller" / f"action-{sequence:04d}.json",
                {"schema": TRACE_SCHEMA, "action_id": action_id, "samples": samples,
                 "physics_journal": {"path": f"controller/action-{sequence:04d}.jsonl", "sha256": "0" * 64}})
    journal = native / "controller" / f"action-{sequence:04d}.jsonl"
    journal.write_text("".join(json.dumps(line, sort_keys=True) + "\n" for line in lines), encoding="utf-8")
    return clock, native_action


def write_native(root: Path, rounds: Sequence[Sequence[Mapping[str, Any]]], *,
                 task: str = "stack_blocks") -> Path:
    """Write one run directory (``root/native``) and return the run directory."""
    native = root / "native"
    _write_json(native / "reset.json", {"schema": "ora.robodojo-reset.v1", "task": task,
                                        "round_index": 0, "actual_source_fingerprint": FINGERPRINT})
    _write_json(native / "source-manifest.json", {"source_revisions": REVISIONS})
    sequence, clock = 0, 100
    for round_index, actions in enumerate(rounds):
        clock += 500  # a reset advances physics outside every journaled tick
        native_action = 0
        for body in actions:
            clock, native_action = _write_action(native, sequence, round_index, body, clock, native_action)
            sequence += 1
    return root
