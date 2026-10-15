"""Shared RoboTwin dual-arm Runtime contracts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ora.agent.composition import (
    ToolSurfaceError,
    validate_tool_help_contract,
)
from ora.contracts import MOTION_ACTION_SCHEMA
from ora.contracts.action import ActionValidationError, validate_runtime_action
from ora.environment.backends.robotwin.qualification import (
    ARTICULATED_MODEL_NAMES,
    DEFAULT_TASKS,
    TASK_MODEL_ATTRIBUTES,
    _articulated_variant,
)
from ora.environment.backends.robotwin.session import (
    DEFERRED_SESSION_TASKS,
    SUPPORTED_TASKS,
    RoboTwinLiveSession,
)
from ora.orchestrator.spec import resolve_run_spec


class _DualArmRobot:
    def __init__(self) -> None:
        self.grippers = {"left": 1.0, "right": 1.0}

    def left_plan_path(self, *_args, **_kwargs):
        return {"status": "Success", "position": np.zeros((2, 7))}

    def right_plan_path(self, *_args, **_kwargs):
        return {"status": "Success", "position": np.zeros((2, 7))}

    def get_left_gripper_val(self):
        return self.grippers["left"]

    def get_right_gripper_val(self):
        return self.grippers["right"]


class _DualArmEnvironment:
    def __init__(self) -> None:
        self.robot = _DualArmRobot()
        self.poses = {
            "left": [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            "right": [0.2, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        }
        self.take_action_cnt = 0
        self.step_lim = 20
        self.eval_success = False
        self.actions: list[list[float]] = []

    def setup_demo(self, **_kwargs):
        return None

    def check_success(self):
        return False

    def get_arm_pose(self, arm):
        return self.poses[arm]

    def get_obs(self):
        camera = {
            "rgb": np.zeros((3, 4, 3), dtype=np.uint8),
            "depth": np.full((3, 4), 1000.0, dtype=np.float32),
            "intrinsic_cv": np.eye(3),
            "extrinsic_cv": np.eye(4),
            "cam2world_gl": np.eye(4),
        }
        return {
            "observation": {"head_camera": camera},
            "endpose": {
                "left_endpose": self.poses["left"],
                "left_gripper": self.robot.grippers["left"],
                "right_endpose": self.poses["right"],
                "right_gripper": self.robot.grippers["right"],
            },
            "joint_action": {
                "left_arm": [0.0] * 6,
                "left_gripper": self.robot.grippers["left"],
                "right_arm": [0.0] * 6,
                "right_gripper": self.robot.grippers["right"],
            },
        }

    def take_action(self, ee_action, action_type):
        assert action_type == "ee"
        values = [float(value) for value in ee_action]
        self.actions.append(values)
        self.poses["left"] = values[:7]
        self.robot.grippers["left"] = values[7]
        self.poses["right"] = values[8:15]
        self.robot.grippers["right"] = values[15]
        self.take_action_cnt += 1

    def close_env(self, clear_cache=False):
        assert clear_cache is False


def _action(observation_id: str, action_id: str, arguments: dict) -> dict:
    action = "set_gripper" if "gripper" in arguments else "move_pose"
    return {
        "schema": "ora-action.v3",
        "observation_id": observation_id,
        "action_id": action_id,
        "action": action,
        "arguments": arguments,
        "bound_observation_sha256": "bound",
    }


def _motion_action(
    observation_id: str, action_id: str, action: str, arguments: dict,
) -> dict:
    return {
        "schema": MOTION_ACTION_SCHEMA,
        "observation_id": observation_id,
        "action_id": action_id,
        "action": action,
        "arguments": arguments,
        "bound_observation_sha256": "bound",
    }


def test_blocks_ranking_size_is_registered_for_session_and_oracle() -> None:
    assert "blocks_ranking_size" in SUPPORTED_TASKS
    assert "blocks_ranking_size" in DEFAULT_TASKS


def test_every_qualification_task_is_admitted_or_explicitly_deferred() -> None:
    # The spot check above passed for a year while the two sets drifted to
    # 14 against 50, because it sampled one long-registered task. Thirty-six
    # tasks then qualified seeds and failed at worker startup instead. Both
    # sets stay independent by design, but every registered task must be in
    # exactly one of them, so a gap is a declaration rather than a silence.
    assert set(DEFAULT_TASKS) == SUPPORTED_TASKS | DEFERRED_SESSION_TASKS
    assert not SUPPORTED_TASKS & DEFERRED_SESSION_TASKS


def test_qualification_registers_every_pinned_task_in_both_structures() -> None:
    assert len(DEFAULT_TASKS) == len(set(DEFAULT_TASKS))
    assert set(DEFAULT_TASKS) == set(TASK_MODEL_ATTRIBUTES)
    assert len(DEFAULT_TASKS) == 50


def test_qualification_model_attribute_entries_are_well_formed() -> None:
    for task, entries in TASK_MODEL_ATTRIBUTES.items():
        assert isinstance(entries, tuple), task
        for entry in entries:
            assert isinstance(entry, tuple), (task, entry)
            assert 2 <= len(entry) <= 4, (task, entry)
            name_field, id_field, *constants = entry
            constant_name = constants[0] if constants else None
            constant_id = constants[1] if len(constants) > 1 else None
            assert name_field is None or isinstance(name_field, str), (task, entry)
            assert id_field is None or isinstance(id_field, str), (task, entry)
            if name_field is None:
                assert isinstance(constant_name, str) and constant_name, (task, entry)
            else:
                assert constant_name is None, (task, entry)
            if id_field is None:
                assert isinstance(constant_id, int), (task, entry)
            else:
                assert constant_id is None, (task, entry)


def test_articulated_variant_matches_the_upstream_index_and_name_fallback(
    tmp_path: Path,
) -> None:
    model_root = tmp_path / "036_cabinet"
    for name in ("46653", "7310", "visual"):
        (model_root / name).mkdir(parents=True)
    assert _articulated_variant(model_root, 0).name == "7310"
    assert _articulated_variant(model_root, 1).name == "46653"
    # put_object_cabinet passes the variant directory name, not an index.
    assert _articulated_variant(model_root, 46653).name == "46653"
    with pytest.raises(ValueError):
        _articulated_variant(model_root, 999)
    assert "036_cabinet" in ARTICULATED_MODEL_NAMES


def test_dual_arm_actions_hold_the_inactive_arm_target(tmp_path: Path) -> None:
    environment = _DualArmEnvironment()
    session = RoboTwinLiveSession(
        robotwin_root=tmp_path / "RoboTwin",
        output_root=tmp_path / "native",
        task="blocks_ranking_size",
        actual_seed=100,
        episode=0,
        task_factory=lambda _task: environment,
        task_args_factory=lambda **_kwargs: {},
    )
    session.start()
    first, _images = session.observe(tmp_path / "observations")
    session.execute(_motion_action(
        first["observation_id"],
        "left-move",
        "move_free_space",
        {
            "arm": "left",
            "target_pose": {
                "position_m": [0.05, 0.0, 1.0],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        },
    ))
    left_command = environment.actions[-1]
    assert left_command[:3] == [0.05, 0.0, 1.0]
    assert left_command[8:15] == [0.2, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    assert left_command[15] == 1.0

    second, _images = session.observe(tmp_path / "observations")
    session.execute(_motion_action(
        second["observation_id"],
        "right-close",
        "set_gripper",
        {"arm": "right", "gripper": 0.0},
    ))
    right_command = environment.actions[-1]
    assert right_command[:7] == left_command[:7]
    assert right_command[7] == 1.0
    assert right_command[8:15] == left_command[8:15]
    assert right_command[15] == 0.0
    session.close()


def test_motion_action_rejects_a_non_atomic_servo_delta() -> None:
    with pytest.raises(ActionValidationError, match="exceeds 0.05 m"):
        validate_runtime_action(_motion_action(
            "observation-0",
            "too-large",
            "servo_delta",
            {
                "arm": "left",
                "target_pose": {
                    "position_m": [0.1, 0.0, 1.0],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "delta_pose": {
                    "translation_m": [0.051, 0.0, 0.0],
                    "rotation_rotvec_rad": [0.0, 0.0, 0.0],
                },
                "frame": "world",
                "max_iterations": 20,
            },
        ))


def test_tool_help_contract_rejects_incomplete_model_visible_help() -> None:
    definition = {
        "name": "inspect",
        "description": (
            "Inspect evidence. Use when: evidence is needed. Effect: read-only. "
            "Returns: one value. Does not: change the world. "
            "On error: correct the field and retry."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Public value name to inspect.",
                },
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    }
    result = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    missing_section = json.loads(json.dumps(definition))
    missing_section["description"] = missing_section["description"].replace(
        "Does not:", "Missing:"
    )
    with pytest.raises(ToolSurfaceError, match="help lacks sections"):
        validate_tool_help_contract([missing_section], {"inspect": result})
    undocumented = json.loads(json.dumps(definition))
    del undocumented["inputSchema"]["properties"]["name"]["description"]
    with pytest.raises(ToolSurfaceError, match="lacks a parameter description"):
        validate_tool_help_contract([undocumented], {"inspect": result})
    with pytest.raises(ToolSurfaceError, match="must cover exactly"):
        validate_tool_help_contract([definition], {})


def test_place_mouse_pad_is_registered_for_session_and_oracle() -> None:
    assert "place_mouse_pad" in SUPPORTED_TASKS
    assert "place_mouse_pad" in DEFAULT_TASKS


def test_expert_only_evaluator_state_is_restored_before_the_reset_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three predicates read attributes upstream sets only inside play_once.

    An ORA rollout never runs the expert, so without this the
    reset probe raises AttributeError and the task surfaces as an
    infrastructure fault instead of an official verdict. The restored values
    must be the same deterministic functions of the reset scene the expert
    computes, so this test pins the expressions rather than merely asserting
    the attributes exist.
    """

    import sys
    import types
    from ora.environment.backends.robotwin import session as module

    envs = types.ModuleType("envs")
    envs_utils = types.ModuleType("envs.utils")
    envs_base = types.ModuleType("envs._base_task")
    envs_utils.get_face_prod = lambda quaternion, a, b: quaternion[0]
    envs_base.ArmTag = lambda value: f"arm:{value}"
    monkeypatch.setitem(sys.modules, "envs", envs)
    monkeypatch.setitem(sys.modules, "envs.utils", envs_utils)
    monkeypatch.setitem(sys.modules, "envs._base_task", envs_base)

    class _Pose:
        def __init__(self, position: list[float], quaternion: list[float]) -> None:
            self.p = position
            self.q = quaternion

    class _Actor:
        def __init__(self, pose: _Pose) -> None:
            self._pose = pose

        def get_pose(self) -> _Pose:
            return self._pose

    # open_laptop selects by the sign of the laptop face product.
    laptop_env = types.SimpleNamespace(
        laptop=_Actor(_Pose([0.0, 0.0, 0.0], [0.4, 0.0, 0.0, 0.0])),
    )
    module._restore_expert_only_evaluator_state(laptop_env, "open_laptop")
    assert laptop_env.arm_tag == "arm:left"
    laptop_env.laptop = _Actor(_Pose([0.0, 0.0, 0.0], [-0.4, 0.0, 0.0, 0.0]))
    module._restore_expert_only_evaluator_state(laptop_env, "open_laptop")
    assert laptop_env.arm_tag == "arm:right"

    # place_object_scale and put_object_cabinet select by the object's x sign,
    # and the cabinet task also records the object's spawn height.
    for task in ("place_object_scale", "put_object_cabinet"):
        right = types.SimpleNamespace(object=_Actor(_Pose([0.2, 0.0, 0.77], [1, 0, 0, 0])))
        module._restore_expert_only_evaluator_state(right, task)
        assert right.arm_tag == "arm:right"
        left = types.SimpleNamespace(object=_Actor(_Pose([-0.2, 0.0, 0.77], [1, 0, 0, 0])))
        module._restore_expert_only_evaluator_state(left, task)
        assert left.arm_tag == "arm:left"
    assert right.origin_z == 0.77

    # Tasks without expert-only evaluator state are untouched.
    other = types.SimpleNamespace()
    module._restore_expert_only_evaluator_state(other, "click_bell")
    assert not hasattr(other, "arm_tag")


def test_official_success_latches_on_a_transient_mid_motion_predicate() -> None:
    """Upstream latches success per physics step; our executors must too.

    Base_Task.take_action steps the scene and calls check_success() after
    each step, setting eval_success permanently the first time it holds. Our
    motion transaction steps the scene itself and bypasses that entry point,
    so a predicate true only mid-motion -- a hammer touching a block before
    it rebounds -- would score false here and true upstream.
    """

    from ora.environment.backends.robotwin.session import (
        RoboTwinLiveSession,
    )

    class _Transient:
        """True on the third poll only, as a rebounding contact would be."""

        def __init__(self) -> None:
            self.eval_success = False
            self.polls = 0

        def check_success(self) -> bool:
            self.polls += 1
            return self.polls == 3

    environment = _Transient()
    for _ in range(6):
        RoboTwinLiveSession._latch_official_success(environment)
    assert environment.eval_success is True
    # Once latched the predicate is no longer polled: upstream stops acting
    # on a finished episode rather than re-testing a decayed state.
    assert environment.polls == 3

    # A predicate that raises mid-motion is not a verdict either way.
    class _Raising:
        eval_success = False

        def check_success(self) -> bool:
            raise RuntimeError("actor pose unavailable between control writes")

    raising = _Raising()
    RoboTwinLiveSession._latch_official_success(raising)
    assert raising.eval_success is False

    # Backends without the predicate are untouched.
    class _Bare:
        eval_success = False

    bare = _Bare()
    RoboTwinLiveSession._latch_official_success(bare)
    assert bare.eval_success is False
