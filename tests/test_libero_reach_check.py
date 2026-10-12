"""LIBERO answers the internal read-only reach check with CuRobo IK (integration 0059 M2).

The reach check (``contracts/reach_check.py``) asks one arm whether it can hold
exact world-frame poses; RoboTwin and RoboDojo answer it with their own IK, and
until M2 LIBERO had no ``reach_check_batch`` at all, so ``grasp_plan`` on LIBERO
could not screen its candidates (H200 BOOT-01). LIBERO now answers with one
CuRobo v0.7.8 IK solve per pose in the isolated planner worker -- the IK stage
of the MotionGen planner its global ``move_pose`` segment plans through --
with joint limits and Panda self-collision and no world model.

Offline and layered:

* the worker's ``ik`` command over a CPU stand-in for CuRobo (no Torch);
* the host-side client contract for that command;
* the session's ``reach_check_batch`` over a fake IK planner, through the real
  frame conversion ``move_pose`` uses (world grip site -> CuRobo base-frame
  panda_hand) and the real shared controller;
* the closed LIBERO reason set and the surface binder that selects it.

The agent edge with a LIBERO bridge is in ``test_libero_reach_ports.py``.
"""
from __future__ import annotations

import copy
import math
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ora.contracts.reach_check import (
    BACKEND_REASONS, LIBERO_REASONS, REQUEST_SCHEMA, RESULT_SCHEMA, ROBODOJO_REASONS, ROBOTWIN_REASONS,
    SHARED_REASONS, START_UNREACHABLE,
)
from ora.contracts.refusal import EnvironmentRefusal
from ora.environment.backends.libero_pro import reach_check as libero_reach
from ora.environment.backends.libero_pro.common import LiberoProError
from ora.environment.backends.libero_pro.planned_motion import LiberoPlannedMotionMixin
from ora.environment.controllers.curobo_ik import (
    CUROBO_IK_RESPONSE_SCHEMA, validate_curobo_ik_request, validate_curobo_ik_response,
)
from ora.environment.controllers.curobo_motion import CuroboPlannerError

NAMES = [f"panda_joint{index}" for index in range(1, 8)]
MEASURED = [0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.8]
#: The robosuite Panda limits (the same table ``test_motion_runtime``'s CPU CuRobo uses).
LIMITS = [[-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
          [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]]
DOWN = [1.0, 0.0, 0.0, 0.0]  # grip site +Z pointing down: 180 degrees about world x


def _matrix(pose):
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
    transform[:3, 3] = pose[:3]
    return transform


def _pose(transform):
    return [*map(float, transform[:3, 3]), *map(float, Rotation.from_matrix(transform[:3, :3]).as_quat())]


#: A yawed, offset robot base and a grip site that is not a pure translation of
#: the hand, so a frame error in either direction changes the numbers.
WORLD_FROM_BASE = _pose(_matrix([-0.66, 0.05, 0.912, *Rotation.from_euler("z", 0.3).as_quat()]))
HAND_FROM_GRIP = _pose(_matrix([0.0, 0.0, 0.1034, *Rotation.from_euler("z", math.pi / 2).as_quat()]))
WORLD_FROM_HAND = _pose(_matrix([-0.2, 0.0, 1.25, *DOWN]))
WORLD_FROM_GRIP = _pose(_matrix(WORLD_FROM_HAND) @ _matrix(HAND_FROM_GRIP))
BASE_FROM_HAND = _pose(np.linalg.inv(_matrix(WORLD_FROM_BASE)) @ _matrix(WORLD_FROM_HAND))
TOLERANCES = {"position_tolerance_m": 0.005, "orientation_tolerance_rad": 0.05}


def _state():
    return {"joint_names": list(NAMES), "joint_positions": list(MEASURED),
            "base_pose_world_xyzw": list(WORLD_FROM_BASE), "hand_pose_world_xyzw": list(WORLD_FROM_HAND),
            "grip_site_pose_world_xyzw": list(WORLD_FROM_GRIP)}


def _base_hand_target(grip_world):
    """Independent of the session code: base <- world <- grip, then grip -> hand."""
    return _pose(np.linalg.inv(_matrix(WORLD_FROM_BASE)) @ _matrix(grip_world)
                 @ np.linalg.inv(_matrix(HAND_FROM_GRIP)))


class FakeIk:
    """CuRobo IK stand-in in the base frame; the client contract checks every answer.

    Reaches a hand position within ``reach_m`` of the base origin. A solution
    moves joint 1 by 0.1 rad from its seed, and its forward kinematics is the
    requested hand pose, optionally turned by ``turn_rad`` about hand x.
    """

    def __init__(self, *, reach_m=0.9, turn_rad=0.0, start=None, step=None):
        self.reach_m, self.turn_rad, self.step = reach_m, turn_rad, step
        self.start = list(BASE_FROM_HAND if start is None else start)
        self.requests: list[dict[str, Any]] = []

    def solve_ik(self, request):
        validated = validate_curobo_ik_request(request)
        self.requests.append(copy.deepcopy(validated))
        if self.step is not None:
            self.step()
        results, ends = [], []
        for query in validated["queries"]:
            start = query["from_index"]
            seed = validated["joint_positions"] if start is None else ends[start]
            if seed is None:
                results.append({"solved": False, "skipped": True, "joint_positions": None,
                                "ee_pose_base_xyzw": None, "position_error_m": None, "rotation_error": None})
                ends.append(None)
                continue
            target = query["target_pose_base_xyzw"]
            if float(np.linalg.norm(target[:3])) > self.reach_m:
                results.append({"solved": False, "skipped": False, "joint_positions": None,
                                "ee_pose_base_xyzw": None, "position_error_m": 0.2, "rotation_error": 0.3})
                ends.append(None)
                continue
            joints = [seed[0] + 0.1, *seed[1:]]
            reached = _matrix(target) @ _matrix([0, 0, 0, *Rotation.from_euler("x", self.turn_rad).as_quat()])
            results.append({"solved": True, "skipped": False, "joint_positions": joints,
                            "ee_pose_base_xyzw": _pose(reached), "position_error_m": 0.0, "rotation_error": 0.0})
            ends.append(joints)
        response = {"schema": CUROBO_IK_RESPONSE_SCHEMA, "status": "solved", "world_model": "none",
                    "solver": {"num_seeds": 32, "position_threshold_m": 0.005, "rotation_threshold": 0.05,
                               "attempts_per_query": 1},
                    "joint_names": list(NAMES), "joint_limits": copy.deepcopy(LIMITS),
                    "start_ee_pose_base_xyzw": list(self.start), "results": results, "solve_time_s": 0.01}
        return validate_curobo_ik_response(response, validated)


class _Session(LiberoPlannedMotionMixin, libero_reach.LiberoReachCheckMixin):
    """The two real mixins over a stand-in plant; nothing here can step physics."""

    def __init__(self, planner):
        self.motion_planner = planner
        self.world_mutation_sequence = 0
        self.environment = SimpleNamespace(_elapsed_steps=7)

    def _require_environment(self):
        return self.environment

    def _motion_state(self):
        return _state()


def _query(pose, interaction="free_space", from_index=None):
    return {"pose_xyzw": list(pose), "interaction": interaction, "from_index": from_index}


def _ask(session, *queries, arm="right"):
    return session.align_plan({"schema": REQUEST_SCHEMA, "arm": arm, "queries": [dict(q) for q in queries]})


def _grip(position, orientation=DOWN):
    return [*position, *orientation]


# --------------------------------------------------------------------------
# Session: the frame move_pose commands, the real controller, a fake IK
# --------------------------------------------------------------------------

def test_libero_solves_the_grip_site_pose_move_pose_commands_in_curobo_s_hand_frame():
    planner = FakeIk()
    session = _Session(planner)
    target = _grip([-0.1, 0.1, 1.05])
    answer = _ask(session, _query(target))
    assert answer["schema"] == RESULT_SCHEMA and answer["arm"] == "right"
    assert answer["ik_model"] == libero_reach.IK_MODEL == "libero_curobo_ik_robot_only_v1"
    assert answer["matches_move_pose_admission"] is False and answer["motion_planner_called"] is False
    [sent] = planner.requests[0]["queries"]
    # The planner is asked for panda_hand in the base frame: the grip site
    # target composed with the measured hand-to-grip transform, exactly as the
    # global move_pose segment compiles it.
    assert sent["target_pose_base_xyzw"][:3] == pytest.approx(_base_hand_target(target)[:3], abs=1e-9)
    assert abs(float(np.dot(sent["target_pose_base_xyzw"][3:], _base_hand_target(target)[3:]))) == \
        pytest.approx(1.0, abs=1e-9)
    assert planner.requests[0]["joint_positions"] == MEASURED
    [row] = answer["results"]
    assert row["reachable"] is True and row["reason"] is None and row["route"] is None
    assert row["position_error_m"] == pytest.approx(0.0, abs=1e-9)
    assert row["orientation_error_rad"] == pytest.approx(0.0, abs=1e-6)
    assert row["joint_delta_rad"] == pytest.approx(0.1)
    # joint 4 sits 0.8 rad below its upper stop -0.0698 at -2.2 ... the smallest margin is joint 7's
    assert row["joint_limit_margin_rad"] == pytest.approx(min(
        min(value - low, high - value) for value, low, high in zip(
            [MEASURED[0] + 0.1, *MEASURED[1:]], LIMITS[0], LIMITS[1]))
    )
    for phrase in ("CuRobo", "joint limits", "self-collision", "no table, object or scene",
                   "no trajectory", "OSC"):
        assert phrase in answer["claim_boundary"], phrase


def test_a_chained_approach_starts_from_its_pregrasp_solution_and_an_orphan_is_start_unreachable():
    planner = FakeIk()
    session = _Session(planner)
    pregrasp, approach = _grip([-0.1, 0.1, 1.05]), _grip([-0.1, 0.1, 0.98])
    far = _grip([0.6, 0.6, 1.05])
    answer = _ask(session, _query(pregrasp), _query(approach, "contact_approach", 0),
                  _query(far), _query(approach, "contact_approach", 2))
    sent = planner.requests[0]["queries"]
    assert [item["from_index"] for item in sent] == [None, 0, None, 2]
    first, chained, failed, orphan = answer["results"]
    assert first["reachable"] and chained["reachable"]
    # 0.1 rad from the chained seed, not 0.2 from the measured joints.
    assert chained["joint_delta_rad"] == pytest.approx(0.1)
    assert failed == {"reachable": False, "reason": "ik_failed", "route": None, "position_error_m": None,
                      "orientation_error_rad": None, "joint_limit_margin_rad": None, "joint_delta_rad": None}
    assert orphan["reachable"] is False and orphan["reason"] == START_UNREACHABLE


def test_a_solution_whose_grip_site_misses_the_arrival_bound_is_named():
    # CuRobo judges the hand; a turn of 0.06 rad about the hand also swings the
    # grip site 0.1034 m away by about 6 mm.
    answer = _ask(_Session(FakeIk(turn_rad=0.06)), _query(_grip([-0.1, 0.1, 1.05])),
                  _query(_grip([-0.1, 0.1, 1.0]), "contact_approach", 0))
    row, chained = answer["results"]
    assert row["reachable"] is False and row["reason"] == "fk_residual_exceeded"
    assert row["orientation_error_rad"] == pytest.approx(0.06, abs=1e-6)
    assert row["position_error_m"] == pytest.approx(0.1034 * math.sin(0.06), abs=2e-4)
    assert row["joint_delta_rad"] == pytest.approx(0.1)
    assert chained["reason"] == START_UNREACHABLE


def test_a_turn_inside_the_arrival_bound_is_reachable():
    [row] = _ask(_Session(FakeIk(turn_rad=0.02)), _query(_grip([-0.1, 0.1, 1.05])))["results"]
    assert row["reachable"] is True and row["orientation_error_rad"] == pytest.approx(0.02, abs=1e-6)


def test_curobo_fk_disagreeing_with_the_measured_hand_is_a_typed_answer():
    shifted = [BASE_FROM_HAND[0] + 0.01, *BASE_FROM_HAND[1:]]
    answer = _ask(_Session(FakeIk(start=shifted)), _query(_grip([-0.1, 0.1, 1.05])),
                  _query(_grip([0.6, 0.6, 1.05])))
    assert [row["reason"] for row in answer["results"]] == ["fk_calibration_mismatch"] * 2
    assert not any(row["reachable"] for row in answer["results"])


def test_the_check_refuses_to_answer_if_the_simulation_moved():
    session = _Session(None)
    session.motion_planner = FakeIk(step=lambda: setattr(session.environment, "_elapsed_steps", 8))
    with pytest.raises(LiberoProError, match="read-only"):
        _ask(session, _query(_grip([-0.1, 0.1, 1.05])))


def test_the_bridge_serves_only_the_reach_check_and_only_the_right_arm():
    session = _Session(FakeIk())
    with pytest.raises(LiberoProError, match="only the internal reach check"):
        session.align_plan({"schema": "ora.align-plan-request.v1", "target_ref": "t",
                            "approach_side": "top", "arm": "right", "target_poses_xyzw": []})
    with pytest.raises(EnvironmentRefusal) as refused:
        _ask(session, _query(_grip([-0.1, 0.1, 1.05])), arm="left")
    assert refused.value.field == "arm" and refused.value.allowed == ["right"]
    with pytest.raises(LiberoProError, match="drifted"):
        session.align_plan({"schema": REQUEST_SCHEMA, "arm": "right", "queries": [], "extra": 1})


def test_no_planner_or_a_planner_error_is_a_fault_not_an_answer():
    with pytest.raises(LiberoProError, match="CuRobo planner is not active"):
        _ask(_Session(None), _query(_grip([-0.1, 0.1, 1.05])))

    class Broken(FakeIk):
        def solve_ik(self, request):
            return {"schema": CUROBO_IK_RESPONSE_SCHEMA, "status": "planner_error", "world_model": "none",
                    "solve_time_s": 0.0, "error": {"type": "RuntimeError", "message": "cuda"}}

    with pytest.raises(LiberoProError, match="CuRobo IK failed: RuntimeError"):
        _ask(_Session(Broken()), _query(_grip([-0.1, 0.1, 1.05])))


def test_the_session_class_serves_the_bridge_through_its_reach_mixin():
    from ora.environment.backends.libero_pro.session import LiberoProSession

    assert issubclass(LiberoProSession, libero_reach.LiberoReachCheckMixin)
    assert LiberoProSession.align_plan is libero_reach.LiberoReachCheckMixin.align_plan


def test_the_real_session_answers_the_bridge_without_stepping_or_commanding(tmp_path, monkeypatch):
    """The real LiberoProSession over the CPU plant of ``test_libero_correctness_repair``."""
    from test_libero_correctness_repair import TARGET, _session

    session, plant = _session(tmp_path, monkeypatch)
    # That fixture's base is the world origin and its hand sits at (0.2, 0, 0.9).
    planner = FakeIk(reach_m=1.5, start=[0.2, 0.0, 0.9, 0.0, 0.0, 0.0, 1.0])
    session.motion_planner = planner
    steps, mutations = session.environment._elapsed_steps, session.world_mutation_sequence
    answer = session.align_plan({"schema": REQUEST_SCHEMA, "arm": "right", "queries": [_query(TARGET)]})
    [row] = answer["results"]
    assert row["reachable"] is True and answer["ik_model"] == libero_reach.IK_MODEL
    # grip site (0.2, 0, 0.8) under hand (0.2, 0, 0.9): the hand target is 0.1 m above the grip target
    assert planner.requests[0]["queries"][0]["target_pose_base_xyzw"][:3] == pytest.approx(
        [TARGET[0], TARGET[1], TARGET[2] + 0.1])
    assert plant.commands == [] and session.environment._elapsed_steps == steps
    assert session.world_mutation_sequence == mutations and session.action_sequence == 0


# --------------------------------------------------------------------------
# Client contract for the worker's ik command
# --------------------------------------------------------------------------

def _ik_request(*queries):
    return {"joint_names": list(NAMES), "joint_positions": list(MEASURED),
            "queries": [{"target_pose_base_xyzw": [0.4, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0], "from_index": None},
                        *queries]}


@pytest.mark.parametrize("change", [
    lambda r: r.update(joint_names=list(reversed(NAMES))),
    lambda r: r["joint_positions"].__setitem__(2, float("nan")),
    lambda r: r.update(queries=[]),
    lambda r: r.update(queries=r["queries"] * 17),
    lambda r: r["queries"].append({"target_pose_base_xyzw": [0.4, 0, 0.5, 0, 0, 0, 0.5], "from_index": None}),
    lambda r: r["queries"].append({"target_pose_base_xyzw": [0.4, 0, 0.5, 0, 0, 0, 1], "from_index": 1}),
    lambda r: r["queries"].append({"target_pose_base_xyzw": [0.4, 0, 0.5, 0, 0, 0, 1], "from_index": True}),
    lambda r: r["queries"].append({"target_pose_base_xyzw": [0.4, 0, 0.5, 0, 0, 0, 1]}),
    lambda r: r.update(scene={}),
])
def test_the_ik_request_contract_is_closed(change):
    request = _ik_request()
    change(request)
    with pytest.raises(CuroboPlannerError):
        validate_curobo_ik_request(request)


def test_the_ik_response_must_answer_every_query_consistently():
    request = validate_curobo_ik_request(_ik_request(
        {"target_pose_base_xyzw": [0.4, 0.0, 0.45, 1.0, 0.0, 0.0, 0.0], "from_index": 0}))
    good = FakeIk().solve_ik(request)
    assert validate_curobo_ik_response(good, request) == good
    for drift in (
        lambda r: r["results"].pop(),
        lambda r: r["results"][0].update(skipped=True),          # skipped though its seed exists
        lambda r: r["results"][0].update(joint_positions=None),  # solved without joints
        lambda r: r.update(world_model="scene"),
        lambda r: r.update(joint_limits=[LIMITS[1], LIMITS[0]]),
        lambda r: r["results"][1].update(rotation_error=-1.0),
    ):
        broken = copy.deepcopy(good)
        drift(broken)
        with pytest.raises(CuroboPlannerError):
            validate_curobo_ik_response(broken, request)
    failed = copy.deepcopy(good)
    failed["results"][0].update(solved=False, joint_positions=None, ee_pose_base_xyzw=None)
    with pytest.raises(CuroboPlannerError, match="skipped"):  # its chained query must then be skipped
        validate_curobo_ik_response(failed, request)
    failed["results"][1] = {"solved": False, "skipped": True, "joint_positions": None, "ee_pose_base_xyzw": None,
                            "position_error_m": None, "rotation_error": None}
    assert validate_curobo_ik_response(failed, request)["results"][1]["skipped"] is True


def test_the_client_sends_the_ik_command_and_validates_the_answer(tmp_path):
    from ora.environment.controllers.curobo_motion import CuroboPlannerClient

    client = object.__new__(CuroboPlannerClient)
    client._process = SimpleNamespace()  # treated as started
    sent = []
    client.start = lambda: {}
    client._invalidate_worker = lambda: sent.append("invalidated")

    def exchange(payload):
        sent.append(copy.deepcopy(payload))
        body = {key: value for key, value in payload.items() if key not in {"schema", "command"}}
        return FakeIk().solve_ik(body)

    client._exchange = exchange
    answer = client.solve_ik(_ik_request())
    assert sent[0]["command"] == "ik" and sent[0]["schema"] == "ora.curobo-motion-request.v1"
    assert answer["status"] == "solved"
    client._exchange = lambda payload: {"schema": "drift"}
    with pytest.raises(CuroboPlannerError):
        client.solve_ik(_ik_request())
    assert sent[-1] == "invalidated"


# --------------------------------------------------------------------------
# Worker: the ik command over a CPU stand-in for CuRobo
# --------------------------------------------------------------------------

def _cpu_ik_worker(monkeypatch, *, reachable, failed_error=None):
    from test_motion_runtime import _CpuTensor
    from ora.environment.backends.libero_pro.curobo_worker import Worker

    class Pose:
        def __init__(self, *, position, quaternion):
            self.position, self.quaternion = position, quaternion

    class JointState:
        @staticmethod
        def from_position(position, *, joint_names):
            return SimpleNamespace(position=position, joint_names=joint_names)

    def fk(joints):
        """Toy kinematics: the hand sits at (q1, q2, q3) with the down orientation."""
        return [*joints[:3], *DOWN]

    class Solver:
        num_seeds, position_threshold, rotation_threshold = 32, 0.005, 0.05

        def __init__(self):
            self.calls = []

        def solve_single(self, goal, retract_config=None, seed_config=None, return_seeds=1, num_seeds=None,
                         use_nn_seed=True, newton_iters=None, link_poses=None):
            target = goal.position.tolist()[0]
            seed = seed_config.tolist()
            self.calls.append({"target": target, "retract": retract_config.tolist(), "seed": seed,
                               "use_nn_seed": use_nn_seed, "return_seeds": return_seeds,
                               "num_seeds": num_seeds})
            ok = reachable(target)
            solution = [*target, *seed[0][0][3:]] if ok else [9.0] * 7
            return SimpleNamespace(success=_CpuTensor([[ok]]), solution=_CpuTensor([[solution]]),
                                   position_error=_CpuTensor([[0.0 if ok else failed_error or 0.3]]),
                                   rotation_error=_CpuTensor([[0.0 if ok else failed_error or 0.2]]))

    class MotionGen:
        def __init__(self):
            self.ik_solver = Solver()
            self.planned = []
            self.ik_diagnostic_attempts = ["stale"]
            self.ik_diagnostic_attempts_dropped = 3
            self.rollout_fn = SimpleNamespace(bound_constraint=SimpleNamespace(
                joint_limits=SimpleNamespace(position=_CpuTensor(LIMITS))))

        def compute_kinematics(self, state):
            pose = fk(state.position.tolist()[0])
            return SimpleNamespace(ee_pose=SimpleNamespace(
                position=_CpuTensor([pose[:3]]), quaternion=_CpuTensor([[pose[6], *pose[3:6]]])))

        def plan_single(self, *args):
            self.planned.append(args)
            raise AssertionError("the reach check never plans")

    for name in ("curobo", "curobo.types"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    for name, members in {"curobo.types.math": {"Pose": Pose},
                          "curobo.types.robot": {"JointState": JointState}}.items():
        module = ModuleType(name)
        module.__dict__.update(members)
        monkeypatch.setitem(sys.modules, name, module)
    worker = object.__new__(Worker)
    worker.tensor_args = SimpleNamespace(to_device=_CpuTensor)
    worker.motion_gen = MotionGen()
    return worker


def test_the_worker_seeds_each_solve_from_the_measured_joints_or_its_chained_solution(monkeypatch):
    worker = _cpu_ik_worker(monkeypatch, reachable=lambda target: target[2] > 0.3)
    request = {"schema": "ora.curobo-motion-request.v1", "command": "ik", "request_id": "r1",
               "joint_names": list(NAMES), "joint_positions": list(MEASURED), "queries": [
                   {"target_pose_base_xyzw": [0.4, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0], "from_index": None},
                   {"target_pose_base_xyzw": [0.4, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0], "from_index": 0},
                   {"target_pose_base_xyzw": [0.4, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0], "from_index": None},
                   {"target_pose_base_xyzw": [0.4, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0], "from_index": 2}]}
    response = worker.solve_ik(request)
    validated = validate_curobo_ik_request({k: v for k, v in request.items()
                                            if k not in {"schema", "command", "request_id"}})
    assert validate_curobo_ik_response(response, validated) == response
    calls = worker.motion_gen.ik_solver.calls
    assert len(calls) == 3  # the orphan of a failed solve is never solved
    first, chained, failed = calls
    # The CPU tensor stand-in is float32, like the device tensors.
    assert np.allclose(first["seed"], [[MEASURED]]) and np.allclose(first["retract"], [MEASURED])
    solution = [0.4, 0.0, 0.5, *MEASURED[3:]]
    assert np.allclose(chained["seed"], [[solution]]) and np.allclose(chained["retract"], [solution])
    assert np.asarray(first["seed"]).shape == (1, 1, 7) and np.asarray(first["retract"]).shape == (1, 7)
    assert all(call["use_nn_seed"] is False and call["return_seeds"] == 1 and call["num_seeds"] is None
               for call in calls)
    results = response["results"]
    assert [item["solved"] for item in results] == [True, True, False, False]
    assert [item["skipped"] for item in results] == [False, False, False, True]
    assert results[0]["joint_positions"] == pytest.approx(solution)
    assert results[0]["ee_pose_base_xyzw"] == pytest.approx([0.4, 0.0, 0.5, *DOWN])
    assert results[2]["joint_positions"] is None and results[2]["ee_pose_base_xyzw"] is None
    assert response["start_ee_pose_base_xyzw"] == pytest.approx([*MEASURED[:3], *DOWN])
    assert response["solver"] == {"num_seeds": 32, "position_threshold_m": 0.005, "rotation_threshold": 0.05,
                                  "attempts_per_query": 1}
    assert np.allclose(response["joint_limits"], LIMITS)
    assert worker.motion_gen.planned == []
    # The plan diagnostics observer never carries a reach solve into the next plan.
    assert worker.motion_gen.ik_diagnostic_attempts == [] and worker.motion_gen.ik_diagnostic_attempts_dropped == 0


def test_a_malformed_ik_request_is_a_planner_error_before_any_solve(monkeypatch):
    worker = _cpu_ik_worker(monkeypatch, reachable=lambda target: True)
    response = worker.solve_ik({"joint_names": list(NAMES), "joint_positions": list(MEASURED), "queries": []})
    assert response["status"] == "planner_error" and response["error"]["type"] == "CuroboPlannerError"
    assert validate_curobo_ik_response(response, None)["status"] == "planner_error"
    assert worker.motion_gen.ik_solver.calls == []


def test_a_non_finite_error_of_an_unsolved_query_is_reported_unknown_not_a_planner_fault(monkeypatch):
    """A failed solve's best-seed error is evidence only; NaN must not retire the shared planner worker."""
    worker = _cpu_ik_worker(monkeypatch, reachable=lambda target: False, failed_error=float("nan"))
    request = {"joint_names": list(NAMES), "joint_positions": list(MEASURED), "queries": [
        {"target_pose_base_xyzw": [0.4, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0], "from_index": None}]}
    response = worker.solve_ik(request)
    assert response["status"] == "solved"
    assert validate_curobo_ik_response(response, validate_curobo_ik_request(request)) == response
    assert response["results"][0] == {"solved": False, "skipped": False, "joint_positions": None,
                                      "ee_pose_base_xyzw": None, "position_error_m": None,
                                      "rotation_error": None}


def test_the_worker_dispatches_the_ik_command():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1]
              / "ora/environment/backends/libero_pro/curobo_worker.py").read_text(encoding="utf-8")
    assert 'elif command == "ik":' in source and "worker.solve_ik(request)" in source


# --------------------------------------------------------------------------
# The closed LIBERO reason set and its surface selection
# --------------------------------------------------------------------------

def test_the_libero_reason_set_is_closed_and_admitted_by_the_controller():
    assert LIBERO_REASONS == ("ik_failed", "fk_residual_exceeded", "fk_calibration_mismatch")
    assert set(LIBERO_REASONS) <= BACKEND_REASONS
    assert not set(LIBERO_REASONS) & set(SHARED_REASONS)
    source = (libero_reach.__file__)
    text = open(source, encoding="utf-8").read()
    for reason in LIBERO_REASONS:
        assert f'"{reason}"' in text, reason
    # The released sets are unchanged.
    assert ROBOTWIN_REASONS == ("joint_limit", "singular", "not_converged")
    assert "fk_calibration_mismatch" in ROBODOJO_REASONS


def test_the_grasp_annotation_selects_the_libero_set_from_its_identity_capability():
    """Same field shapes as the released binder; only the stage reason enum differs.

    The released binder file is a hashed component of three released surfaces,
    so LIBERO binds through its sibling module and both files are named.
    """
    from ora.agent.composition import grasp_ik_reachability_libero_schema as libero_binder
    from ora.agent.composition import grasp_ik_reachability_schema as binder
    from ora.agent.composition import resolve_tool_surface
    from ora.agent.composition.tool_surface import ToolSurfaceError

    assert "libero_pro_rgbd_v1" not in binder.BACKEND_REASONS  # released file untouched
    # A kept surface whose grasp_plan result carries no annotation yet (no LIBERO one is left).
    released = resolve_tool_surface("robotwin_toolchain_b2_closed_noservo_v1")
    assert not released.supports(binder.CAPABILITY)
    capabilities = [binder.CAPABILITY, "libero_pro_rgbd_v1"]
    assert libero_binder.declared(capabilities)
    libero = copy.deepcopy(dict(released.result_schemas))
    descriptors = libero_binder.bind_libero_grasp_ik_reachability(libero, capabilities)
    assert [item["ref"] for item in descriptors] == [
        "ora/agent/composition/grasp_ik_reachability_schema.py",
        "ora/agent/composition/grasp_ik_reachability_libero_schema.py"]
    twin = copy.deepcopy(dict(released.result_schemas))
    binder.bind_grasp_ik_reachability(twin, [binder.CAPABILITY, "robotwin_rgbd_v1"])

    def stages(results):
        candidate = results["grasp_plan"]["$defs"]["candidate"]["properties"]["ik_reachability"]
        return [candidate["properties"][stage]["properties"]["reason"] for stage in ("pregrasp", "approach")]

    for reason in stages(libero):
        assert reason["enum"] == [*SHARED_REASONS, *LIBERO_REASONS, None]
    for reason in stages(twin):
        assert reason["enum"] == [*SHARED_REASONS, *ROBOTWIN_REASONS, None]
        reason["enum"] = [*SHARED_REASONS, *LIBERO_REASONS, None]
    assert libero == twin  # every other field is the released shape
    with pytest.raises(ToolSurfaceError, match="exactly one backend"):
        libero_binder.bind_libero_grasp_ik_reachability(
            copy.deepcopy(dict(released.result_schemas)), [*capabilities, "robotwin_rgbd_v1"])
    with pytest.raises(ToolSurfaceError, match="overwrite"):
        libero_binder.bind_libero_grasp_ik_reachability(libero, capabilities)
