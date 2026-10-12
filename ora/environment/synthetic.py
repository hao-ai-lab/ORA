"""Synthetic M0 implementation of the external-world session contract."""

from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any, Mapping

from ora.contracts.action import validate_runtime_action
from .base import ENVIRONMENT_STATUS_SCHEMA, validate_environment_status


def synthetic_public_environment_context(
    *, camera_names: tuple[str, ...], action_codec: str = "target_state_v4",
    controller: str | None = None,
) -> dict[str, Any]:
    """Describe the public synthetic embodiment selected by the environment."""

    return {
        "schema": "ora.public-environment-context.v1",
        "environment_id": "synthetic",
        "embodiment_id": "synthetic_target_state",
        "robot_models": ["SyntheticTargetState"],
        "controller": controller or action_codec,
        "public_cameras": list(camera_names),
    }


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_npy_f32(path: Path, rows: int, columns: int, value: float) -> None:
    """Write a small C-order NPY without depending on NumPy in M0."""

    header = repr({"descr": "<f4", "fortran_order": False, "shape": (rows, columns)})
    prefix = b"\x93NUMPY" + bytes([1, 0])
    padding = 16 - ((len(prefix) + 2 + len(header) + 1) % 16)
    encoded = (header + " " * padding + "\n").encode("latin1")
    path.write_bytes(
        prefix
        + struct.pack("<H", len(encoded))
        + encoded
        + struct.pack("<f", value) * (rows * columns)
    )


def _initial_end_effectors() -> dict[str, dict[str, Any]]:
    return {
        "left": {"pose_xyz_xyzw": [0, 0, 1, 0, 0, 0, 1], "gripper": 1.0},
        "right": {"pose_xyz_xyzw": [0, 0, 1, 0, 0, 0, 1], "gripper": 1.0},
    }


@dataclass
class SyntheticSession:
    """Small deterministic EnvironmentSession used only by synthetic M0."""

    output_root: Path
    task: str
    seed: int
    max_actions: int = 1
    main_camera: str = "head_camera"
    camera_names: tuple[str, ...] = ("head_camera",)
    action_codec: str = "target_state_v4"
    public_controller: str | None = None
    action_sequence: int = 0
    round_action_sequence: int = 0
    observation_sequence: int = 0
    round_index: int = 0
    terminal: bool = False
    abort_reason: str | None = None
    last_receipt: dict[str, Any] | None = None
    end_effectors: dict[str, dict[str, Any]] = field(
        default_factory=_initial_end_effectors
    )

    def start(self) -> dict[str, Any]:
        if self.max_actions <= 0:
            raise ValueError("max_actions must be positive")
        if self.main_camera not in self.camera_names:
            raise ValueError("main_camera must be included in camera_names")
        self.output_root.mkdir(parents=True, exist_ok=True)
        return {
            "schema": "ora.synthetic-reset.v1",
            "task": self.task,
            "seed": self.seed,
            "public_environment_context": synthetic_public_environment_context(
                camera_names=self.camera_names,
                action_codec=self.action_codec,
                controller=self.public_controller,
            ),
        }

    def restart_round(
        self, round_index: int, *, seed: int | None = None,
    ) -> dict[str, Any]:
        if round_index != self.round_index + 1:
            raise ValueError("synthetic round reset must advance by exactly one")
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError("synthetic round seed must be a non-negative integer")
            self.seed = seed
        self.round_index = round_index
        self.round_action_sequence = 0
        self.terminal = False
        self.abort_reason = None
        self.last_receipt = None
        self.end_effectors = _initial_end_effectors()
        reset = {
            "schema": "ora.synthetic-round-reset.v1",
            "task": self.task,
            "seed": self.seed,
            "round_index": round_index,
            "public_environment_context": synthetic_public_environment_context(
                camera_names=self.camera_names,
                action_codec=self.action_codec,
                controller=self.public_controller,
            ),
        }
        _atomic_json(
            self.output_root / "round-resets" / f"round-{round_index:04d}.json",
            reset,
        )
        return reset

    def observe(self, destination: Path) -> tuple[dict[str, Any], list[Path]]:
        observation_id = f"synthetic:{self.task}:{self.observation_sequence:04d}"
        target = destination / observation_id
        from PIL import Image
        camera_records: dict[str, Any] = {}
        image_paths: list[Path] = []
        for camera_name in self.camera_names:
            camera = target / camera_name
            camera.mkdir(parents=True)
            rgb = camera / "rgb.png"
            depth = camera / "depth-m.npy"
            calibration = camera / "calibration.json"
            image = Image.new("RGB", (4, 3))
            pixels = image.load()
            corner_colours = {
                (0, 0): (255, 0, 0), (3, 0): (0, 255, 0),
                (0, 2): (0, 0, 255), (3, 2): (255, 255, 0),
            }
            for y in range(3):
                for x in range(4):
                    pixels[x, y] = corner_colours.get(
                        (x, y), (32 + x * 20, 32 + y * 30, 64)
                    )
            image.save(rgb, format="PNG")
            _write_npy_f32(depth, 3, 4, 1.0)
            _atomic_json(calibration, {
                "intrinsic_cv": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "extrinsic_cv": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                "cam2world_gl": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                "depth_unit": "metres",
            })
            files = {
                "rgb_lossless_png": {"path": f"{camera_name}/rgb.png", "sha256": _file_hash(rgb), "bytes": rgb.stat().st_size},
                "depth_numpy_m_f32": {"path": f"{camera_name}/depth-m.npy", "sha256": _file_hash(depth), "bytes": depth.stat().st_size},
                "calibration": {"path": f"{camera_name}/calibration.json", "sha256": _file_hash(calibration), "bytes": calibration.stat().st_size},
            }
            camera_records[camera_name] = {
                "width": 4, "height": 3,
                "mount": "fixed_world" if camera_name not in {"left_camera", "right_camera"} else camera_name.replace("_camera", "_wrist"),
                "calibration_dynamic": camera_name in {"left_camera", "right_camera"},
                "modalities": ["rgb", "depth", "calibration"], "files": files,
            }
            image_paths.append(rgb)
        payload = {
            "schema": "ora-observation.v1",
            "observation_id": observation_id,
            "task": self.task,
            "sequence": self.observation_sequence,
            "main_camera": self.main_camera,
            "cameras": camera_records,
            "proprioception": {
                "end_effectors": json.loads(json.dumps(self.end_effectors))
            },
            "previous_execution_receipt": self.last_receipt,
            "forbidden_modalities_absent": [
                "actor_state",
                "object_gt_pose",
                "segmentation",
                "contact",
                "success_predicate",
                "expert_program",
                "task_source",
            ],
        }
        _atomic_json(target / "observation.json", payload)
        self.observation_sequence += 1
        return payload, image_paths

    def execute(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        action = validate_runtime_action(payload)
        arm = action["arguments"]["arm"]
        before = json.loads(json.dumps(self.end_effectors))
        if action["action"] == "move_pose":
            self.end_effectors[arm]["pose_xyz_xyzw"] = (
                list(action["arguments"]["position_m"])
                + list(action["arguments"]["orientation_xyzw"])
            )
        else:
            self.end_effectors[arm]["gripper"] = float(action["arguments"]["gripper"])
        after = json.loads(json.dumps(self.end_effectors))
        gripper_before = float(before[arm]["gripper"])
        gripper_after = float(after[arm]["gripper"])
        pose_translation_residual = sum(
            (float(before[arm]["pose_xyz_xyzw"][index]) - float(after[arm]["pose_xyz_xyzw"][index])) ** 2
            for index in range(3)
        ) ** 0.5
        gripper_residual = abs(gripper_before - gripper_after)
        self.action_sequence += 1
        self.round_action_sequence += 1
        self.terminal = self.round_action_sequence >= self.max_actions
        receipt = {
            "schema": "ora.synthetic-motion-receipt.v1",
            "action": action,
            "state_changed": True,
            "before": before,
            "after": after,
            "commanded_pose_xyz_xyzw": (
                after[arm]["pose_xyz_xyzw"]
                if action["action"] == "move_pose" else None
            ),
            "reached_pose_xyz_xyzw": after[arm]["pose_xyz_xyzw"],
            "reached_gripper": gripper_after,
            "gripper_evidence": {
                "measured_before": gripper_before,
                "measured_after": gripper_after,
                "closure_delta": gripper_before - gripper_after,
                "closure_residual": (
                    abs(float(action["arguments"]["gripper"]) - gripper_after)
                    if action["action"] == "set_gripper" else None
                ),
                "grasp_success": None,
            },
            "preserved_dimension": (
                {
                    "field": "gripper", "before": gripper_before,
                    "after": gripper_after, "residual": gripper_residual,
                    "verified": gripper_residual <= 1e-3,
                }
                if action["action"] == "move_pose"
                else {
                    "field": "eef_pose", "translation_residual_m": pose_translation_residual,
                    "verified": pose_translation_residual <= 0.005,
                }
            ),
            "semantic_action_count": self.round_action_sequence,
        }
        path = self.output_root / f"receipts/action-{self.action_sequence - 1:04d}.json"
        _atomic_json(path, receipt)
        record = {
            "path": path.relative_to(self.output_root).as_posix(),
            "sha256": _file_hash(path),
            "receipt": receipt,
        }
        self.last_receipt = record
        return record

    @property
    def is_terminal(self) -> bool:
        return bool(self.status()["terminal"])

    def status(self) -> dict[str, Any]:
        terminal = bool(self.terminal)
        reason = (
            self.abort_reason
            if self.abort_reason is not None
            else "synthetic_action_limit" if terminal else None
        )
        return validate_environment_status({
            "schema": ENVIRONMENT_STATUS_SCHEMA,
            "phase": "terminal" if terminal else "running",
            "terminal": terminal,
            "reason": reason,
            "source": "synthetic_session",
            "policy_visibility": "control_only",
            "signals": [],
        })

    def persist_official_verdict(self) -> dict[str, Any]:
        verdict = {
            "schema": "ora.synthetic-verdict.v1",
            "official_success": None,
            "diagnostic_complete": True,
        }
        _atomic_json(self.output_root / "official-verdict.json", verdict)
        return verdict

    def abort(self, _reason: str) -> None:
        self.abort_reason = _reason
        self.terminal = True

    def finalize_operator_video(self) -> dict[str, Any] | None:
        return None

    def close(self) -> None:
        return None
