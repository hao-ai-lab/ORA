"""Shared PublicEpisode fixture for retention and holding dispatch tests.

This module now only hosts the fixture: its own gate-v2 transport-permit tests
went with the last surfaces declaring that capability.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from ora.agent.composition import resolve_tool_surface
from ora.agent.tools.runtime import ManipulationToolRuntime


POSE = [0.2, 0.0, 1.0, 0, 0, 0, 1]


class PublicEpisode:
    """Inject only public sensor snapshots; run the real compiler, gate and result schemas."""

    def __init__(self, tmp_path: Path, monkeypatch, *, profile) -> None:
        self.root = tmp_path
        self.states = {}
        self.submitted = []
        rng = np.random.default_rng(73)
        self.cloud = rng.uniform(-0.01, 0.01, (180, 3)) + [0.23, 0, 0.96]
        self.geometry = {}
        self.advance()
        runtime = object.__new__(ManipulationToolRuntime)
        self.runtime = runtime
        runtime.surface = resolve_tool_surface(profile)
        runtime.allowed_tools = tuple(d["name"] for d in runtime.surface.definitions)
        runtime.control_root = tmp_path
        runtime.state_root = tmp_path / "policy-state"
        runtime.event_path = tmp_path / "events.jsonl"
        runtime.point_path = tmp_path / "points.json"
        runtime.preview_path = tmp_path / "previews.json"
        runtime._known_plan_refs = {}
        runtime._submitted_mutations = []
        runtime.workspace_bounds_m = {"x": [-1, 1], "y": [-1, 1], "z": [0, 2]}
        runtime._current_grip_state = lambda: (self.control, tmp_path, self.current, self.pose, 0.02)
        runtime._current_target_geometry = lambda **_: {}
        runtime._target_geometry_and_state = lambda *, target_ref: (
            self.geometry[int(target_ref.removeprefix("target-"))],
            self.states[int(target_ref.removeprefix("target-"))],
        )
        runtime._persist_action = lambda _path, _control, value: self.submitted.append(copy.deepcopy(value))
        monkeypatch.setattr("ora.agent.tools.runtime.core.load_policy_state", lambda _root, step=None:
                            self.current if step is None else self.states[step])
        monkeypatch.setattr("ora.agent.tools.runtime.core.verify_policy_state", lambda *_args, **_kw:
                            {"status": "valid", "steps": list(self.states.values())})

    @property
    def current(self):
        return self.states[max(self.states)]

    @property
    def pose(self):
        return self.current["proprioception"]["end_effectors"]["right"]["pose_xyz_xyzw"]

    @property
    def control(self):
        step = self.current["step"]
        return {"observation_id": f"obs-{step}", "observation_sha256": f"{step:064x}", "generation": step, "open": True}

    def advance(self, pose=None, *, steps=4, changed=True, status="reached", object_z=None):
        previous = self.current if self.states else None
        step = len(self.states)
        action = copy.deepcopy(self.submitted[-1]) if self.submitted else None
        pose = list(pose if pose is not None else (self.pose if self.states else POSE))
        state = {
            "step": step, "state_sha256": f"{step + 100:064x}",
            "previous_state_sha256": previous["state_sha256"] if previous else None,
            "cause": "after_action" if previous else "reset",
            "observation": {"id": f"obs-{step}", "manifest": {"sha256": f"{step:064x}"}},
            "proprioception": {"end_effectors": {"right": {"pose_xyz_xyzw": pose, "gripper_opening_m": 0.02}}},
            "cameras": {"agentview": {"files": {name: {"sha256": value * 64} for name, value in (
                ("rgb_lossless_png", "a"), ("depth_numpy_m_f32", "b"), ("calibration", "c"))}}},
            "previous_public_action": action,
            "previous_execution_receipt": None if action is None else {
                "schema": "ora.libero-sparse-motion-receipt.v1",
                "action_id": action["action_id"], "consumed": True, "execution_status": status,
                "state_changed": changed, "steps_used": steps,
            },
        }
        self.states[step] = state
        points = self.cloud + [0, 0, (pose[2] - 1) if object_z is None else object_z]
        self.geometry[step] = {
            "observation_id": f"obs-{step}", "source_camera": "agentview", "voxel_size_m": 0.003,
            "centroid_world_m": points.mean(axis=0).tolist(), "voxel_points_world_m": points.tolist(),
            "source_quality": {"depth_valid_fraction": 1.0, "depth_inlier_fraction": 1.0,
                               "primary_component_fraction": 1.0, "touches_image_boundary": False},
        }
        return state

    def move(self, z):
        return self.runtime.invoke("move_pose", {"interaction": "precision", "position_m": [0.2, 0, z],
                                                 "orientation_xyzw": POSE[3:]})

    def verify(self, before, current=None):
        return self.runtime.invoke("grasp.verify_retention", {
            "before_target_ref": f"target-{before}",
            "current_target_ref": f"target-{self.current['step'] if current is None else current}",
        })

    def close_probe_verify(self):
        self.runtime.invoke("close_gripper", {"target_ref": "target-0"})
        self.advance()
        self.move(1.025)
        self.advance([*POSE[:2], 1.025, *POSE[3:]])
        return self.verify(1)
