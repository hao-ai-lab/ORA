"""Original IK sample evidence survives direct motion without changing admission."""
from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from ora.environment.backends.libero_pro.planning_diagnostics import (
    capture_attempt, install_ik_observer, reset_ik_observer, summarize,
)
from ora.environment.backends.libero_pro.session import LiberoProSession
from ora.environment.controllers.curobo_motion import validate_curobo_response
from tests.agent.motion.test_motion_runtime import _CpuTensor, _action, _cpu_curobo_worker, _pose, _robot_only_request


def _original_ik():
    positions = np.zeros((4, 1, 7))
    positions[2, 0, 0] = 3.0
    raw = SimpleNamespace(action=SimpleNamespace(position=_CpuTensor(positions)),
        metrics=SimpleNamespace(
            position_error=_CpuTensor([[.04], [.001], [.001], [.001]]),
            rotation_error=_CpuTensor([[.01], [.01], [.01], [.01]]),
            feasible=_CpuTensor([[True], [False], [False], [True]])))
    answer = SimpleNamespace(success=_CpuTensor([[True, False]]))
    limits = SimpleNamespace(position=_CpuTensor([[-2.] * 7, [2.] * 7]))
    solver = SimpleNamespace(
        position_threshold=.005, rotation_threshold=.05,
        rollout_fn=SimpleNamespace(kinematics=SimpleNamespace(
            joint_names=[f"panda_joint{i}" for i in range(1, 8)])),
        solver=SimpleNamespace(safety_rollout=SimpleNamespace(
            bound_constraint=SimpleNamespace(joint_limits=limits))))
    calls = []

    def original(*args):
        calls.append(args)
        return answer

    solver._get_result = original
    return SimpleNamespace(ik_solver=solver), raw, answer, calls


@pytest.mark.parametrize("explicit_horizon", [False, True])
def test_observer_preserves_all_original_seeds_without_changing_result_or_call(explicit_horizon):
    motion_gen, raw, answer, calls = _original_ik()
    if not explicit_horizon:
        raw.action.position = _CpuTensor(raw.action.position.numpy()[:, 0, :])
    original_q = raw.action.position.numpy()
    original_masks = raw.metrics.feasible.numpy()
    install_ik_observer(motion_gen)
    goal = object()
    returned = motion_gen.ik_solver._get_result(4, raw, goal, 2)
    assert returned is answer
    assert calls == [(4, raw, goal, 2)]
    np.testing.assert_array_equal(raw.action.position.numpy(), original_q)
    np.testing.assert_array_equal(raw.metrics.feasible.numpy(), original_masks)
    record = motion_gen.ik_diagnostic_attempts[0]
    assert record["candidate_counts"] == {
        "optimized": 4, "returned": 2, "returned_successful": 1,
        "converged": 3, "feasible": 2, "accepted": 1,
    }
    assert record["rejection_counts"] == {
        "position_tolerance": 1, "rotation_tolerance": 0, "solver_infeasible": 2,
        "outside_position_limits": 1, "self_collision": None,
    }
    assert record["accepted"] == [False, False, False, True]
    assert record["joint_positions"][2][0] == 3.0
    # The in-bounds infeasible candidate does not become a claimed collision.
    assert record["unknown_reasons"] == ["per_term_constraint_costs_not_retained_by_sdk"]


@pytest.mark.parametrize("failure", ["metrics", "shape", "order"])
def test_diagnostic_failure_preserves_original_answer_and_does_not_retry(failure):
    motion_gen, raw, answer, calls = _original_ik()
    install_ik_observer(motion_gen)
    if failure == "metrics":
        raw.metrics.position_error = None
    elif failure == "shape":
        raw.action.position = _CpuTensor(np.zeros((4, 2, 7)))
    else:
        motion_gen.ik_solver.rollout_fn.kinematics.joint_names.reverse()
    assert motion_gen.ik_solver._get_result(4, raw, None, 2) is answer
    assert len(calls) == 1
    record = motion_gen.ik_diagnostic_attempts[0]
    assert record["status"] == "unavailable"
    summary, evidence = summarize(motion_gen, {
        "planner_status": "MotionGenStatus.IK_FAIL", "reachable": False,
    }, _robot_only_request())
    assert summary["details_available"] is False
    assert summary["candidate_counts"] == {}
    assert "ik_metric_capture_failed" in summary["unknown_reasons"]
    assert evidence["extra_solver_calls"] == 0


def test_original_extraction_error_remains_an_error():
    motion_gen, raw, _, calls = _original_ik()

    def original(*args):
        calls.append(args)
        raise RuntimeError("original extraction failed")

    motion_gen.ik_solver._get_result = original
    install_ik_observer(motion_gen)
    with pytest.raises(RuntimeError, match="original extraction failed"):
        motion_gen.ik_solver._get_result(4, raw, None, 2)
    assert len(calls) == 1
    assert motion_gen.ik_diagnostic_attempts == []


def test_joint_goal_cannot_reuse_previous_target_ik_diagnostics(monkeypatch):
    worker = _cpu_curobo_worker(monkeypatch)
    motion_gen, raw, answer, _ = _original_ik()
    worker.motion_gen.ik_diagnostic_attempts = [capture_attempt(motion_gen.ik_solver, 4, raw, answer)]
    response = validate_curobo_response(worker.plan(_robot_only_request("js")))
    assert response["reachable"] is True
    assert response["planning_diagnostics"]["phase"] == "joint_space"
    assert response["planning_diagnostics"]["candidate_counts"] == {}
    assert response["planning_diagnostic_evidence"]["attempts"] == []


def test_direct_move_failure_keeps_sample_diagnostics_and_zero_execution(tmp_path):
    motion_gen, raw, answer, _ = _original_ik()
    reset_ik_observer(motion_gen)
    raw.metrics.feasible = _CpuTensor([[False]] * 4)
    answer.success = _CpuTensor([[False, False]])
    motion_gen.ik_diagnostic_attempts.append(capture_attempt(motion_gen.ik_solver, 4, raw, answer))
    response = {"schema": "ora.curobo-motion-response.v1",
        "status": "not_reachable", "reachable": False, "world_model": "none",
        "valid_query": True, "planner_status": "MotionGenStatus.IK_FAIL"}
    response["planning_diagnostics"], response["planning_diagnostic_evidence"] = summarize(
        motion_gen, response, _robot_only_request())
    requests = []

    class Planner:
        def plan(self, request):
            requests.append(deepcopy(request))
            return validate_curobo_response({**response, "request": request})

    session = object.__new__(LiberoProSession)
    session.execution_capture = {"enabled": False}
    session.motion_planner = Planner()
    session.output_root = tmp_path
    session.action_sequence = session.sim_steps = session.world_mutation_sequence = 0
    session.last_receipt = None
    session.workspace_bounds_m = {"x": [-.6, .6], "y": [-.5, .5], "z": [-.1, 1.3]}
    session._require_environment = lambda: SimpleNamespace()
    session._current_state = lambda: (np.array([.4, 0., .5]), np.array([0., 0., 0., 1.]), .2)
    session._motion_state = lambda: {
        "joint_names": [f"panda_joint{i}" for i in range(1, 8)], "joint_positions": [0.] * 7,
        "base_pose_world_xyzw": [0., 0., 0., 0., 0., 0., 1.],
        "hand_pose_world_xyzw": [.4, 0., .5, 0., 0., 0., 1.],
        "grip_site_pose_world_xyzw": [.4, 0., .5, 0., 0., 0., 1.],
    }
    receipt = session._execute_planned_motion(_action(
        "move_free_space", {"arm": "right", "target_pose": _pose()}))
    assert len(requests) == 1
    assert receipt["steps_used"] == 0 and receipt["state_changed"] is False
    assert receipt["feedback"]["planning_failure_reason"] == "MotionGenStatus.IK_FAIL"
    public = receipt["planner"]["planning_diagnostics"]
    assert public["phase"] == "ik" and public["candidate_counts"]["accepted"] == 0
    assert public["rejection_counts"]["self_collision"] is None
    assert "joint_positions" not in json.dumps(public)
    path = tmp_path / receipt["planner"]["canonical_artifact"]["path"]
    saved = json.loads(path.read_text())
    assert saved["planning_diagnostic_evidence"]["attempts"][0]["joint_positions"] == raw.action.position.numpy()[:, 0, :].tolist()
    assert session.sim_steps == session.world_mutation_sequence == 0
