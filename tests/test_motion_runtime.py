"""Shared hybrid-motion Runtime contracts."""

from __future__ import annotations

import json
import math
import sys
from copy import deepcopy
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pytest

from ora.agent.evidence.policy_state import _forbidden_policy_key
from ora.agent.evidence.trajectory import _forbidden_path
from ora.agent.tools.grasp import Sam3ApiSegmenter
from ora.contracts import (
    MOTION_ACTION_SCHEMA,
    compile_servo_delta,
    validate_runtime_action,
)
from ora.contracts.action import ActionValidationError
from ora.environment.backends.libero_pro.session import LiberoProSession
from ora.environment.controllers.curobo_motion import (
    CuroboPlannerError,
    validate_curobo_request,
    validate_curobo_response,
)
from ora.environment.controllers.hybrid_motion import (
    HybridMotionProfileError,
    aggregate_motion_contact,
    bounded_joint_position_command,
    load_hybrid_motion_profile,
    motion_stop_reason,
    validate_hybrid_motion_profile,
)
from ora.environment.workspace_profiles import resolve_workspace_profile
from ora.orchestrator.spec import RunSpecError, resolve_run_spec
from ora.orchestrator.stagnation import StagnationGuard


ROOT = Path(__file__).resolve().parents[1]
MOTION_PROFILE = (
    ROOT / "ora/policy/embodiments/franka-panda/motion-profile-v2.json"
)
MOTION_PROFILE_SHA256 = (
    "30df8c9ba91abc23c8ebdbf3977bf20604441e6bbe853ee6dad618074f8548ea"
)
CUROBO_WORKER = (
    ROOT / "ora/environment/backends/libero_pro/curobo_worker.py"
)


def _action(
    name: str, arguments: dict[str, object], *, sequence: int = 0,
) -> dict[str, object]:
    return {
        "schema": MOTION_ACTION_SCHEMA,
        "observation_id": f"observation-{sequence}",
        "action_id": f"action-{sequence}",
        "action": name,
        "arguments": arguments,
        "bound_observation_sha256": "bound",
    }


def _pose() -> dict[str, list[float]]:
    return {
        "position_m": [0.4, 0.0, 0.5],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }


@pytest.mark.parametrize("outcome", [
    "ready", "timeout", "invalid_json", "response_identity", "close_error", "interrupt",
])
def test_curobo_preflight_closes_its_worker_even_when_readiness_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, outcome: str,
) -> None:
    from ora.environment.controllers import curobo_motion
    from ora.orchestrator import preflight

    absent = {
        "libero_pro_root_env": "TEST_CUROBO_PREFLIGHT_ABSENT_LIBERO",
        "data_root_env": "TEST_CUROBO_PREFLIGHT_ABSENT_DATA",
        "python_env": "TEST_CUROBO_PREFLIGHT_ABSENT_SIMULATOR",
    }
    for name in absent.values():
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TEST_CUROBO_PREFLIGHT_PYTHON", str(tmp_path / "planner-python"))
    monkeypatch.setenv("TEST_CUROBO_PREFLIGHT_ROOT", str(tmp_path / "planner-root"))
    events = []
    worker_outputs = []

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            self.output = kwargs["output_root"]
            worker_outputs.append(self.output)
            events.append("created")

        def start(self) -> dict[str, Any]:
            events.append("spawned")
            if outcome == "interrupt":
                raise KeyboardInterrupt("CPU readiness interruption")
            if outcome not in {"ready", "close_error"}:
                raise CuroboPlannerError(f"CPU readiness {outcome}")
            return {"worker": {"pid": 123}, "test_fixture_only": True}

        def close(self) -> None:
            assert self.output.is_dir(), "close must precede temporary workspace cleanup"
            events.append("closed")
            if outcome == "close_error":
                raise CuroboPlannerError("CPU readiness cleanup failed")

    def forbidden_process(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("CPU preflight fixture must not launch any process")

    monkeypatch.setattr(curobo_motion, "CuroboPlannerClient", FakeClient)
    monkeypatch.setattr(preflight.subprocess, "run", forbidden_process)
    monkeypatch.setattr(preflight, "_git", forbidden_process)
    plan = {
        "policy": {"action_codec": "motion_action_v3"},
        "runtime": {**absent, "kind": "libero_pro", "require_gpu": False,
            "motion_planner": {
                "python_env": "TEST_CUROBO_PREFLIGHT_PYTHON",
                "root_env": "TEST_CUROBO_PREFLIGHT_ROOT",
                "required_revision": "a" * 40,
            }},
    }
    if outcome == "interrupt":
        with pytest.raises(KeyboardInterrupt, match="CPU readiness interruption"):
            preflight._libero_pro_runtime(plan)
    else:
        checks = preflight._libero_pro_runtime(plan)
        planner = next(row for row in checks if row["check"] == "curobo_motion_planner")
        assert planner["passed"] is (outcome == "ready")
        if outcome != "ready":
            assert "CPU readiness" in planner["detail"]
    assert events == ["created", "spawned", "closed"]
    assert len(worker_outputs) == 1 and not worker_outputs[0].exists()


def test_hybrid_contact_estimate_is_safe_aggregate_not_contact_truth() -> None:
    public_receipt = {"feedback": {"contact_estimated": True}}
    assert _forbidden_path(public_receipt) is None
    assert _forbidden_policy_key(public_receipt) is None

    privileged = {"feedback": {"contact_estimated": {"body": "object-1"}}}
    assert _forbidden_path(privileged) == "$.feedback.contact_estimated"
    assert _forbidden_policy_key(privileged) == "$.feedback.contact_estimated"

    compiler_geometry = {
        "compiler": {
            "contact_graspnet_gripper_depth_m": 0.1034,
            "controlled_point_to_contact_baseline_m": 0.0064,
        },
        "candidate": {"compiler_checks": {
            "contact_baseline_reconstruction": "passed",
        }},
    }
    assert _forbidden_path(compiler_geometry) == (
        "$.compiler.contact_graspnet_gripper_depth_m"
    )
    assert _forbidden_path(
        compiler_geometry, allow_public_grasp_contact_pose=True,
    ) is None
    assert _forbidden_path({
        "compiler": {"contact_graspnet_gripper_depth_m": 0.1035},
    }, allow_public_grasp_contact_pose=True) == (
        "$.compiler.contact_graspnet_gripper_depth_m"
    )
    assert _forbidden_path({
        "elsewhere": {"controlled_point_to_contact_baseline_m": 0.0064},
    }, allow_public_grasp_contact_pose=True) == (
        "$.elsewhere.controlled_point_to_contact_baseline_m"
    )
    assert _forbidden_path({
        "candidate": {"compiler_checks": {
            "contact_baseline_reconstruction": {"body": "object-1"},
        }},
    }, allow_public_grasp_contact_pose=True) == (
        "$.candidate.compiler_checks.contact_baseline_reconstruction"
    )


def test_forbidden_terms_match_word_boundaries_not_substrings() -> None:
    # Regression for the v13 seal incident: the policy-safe staged-motion
    # receipt fields embed the controller phase noun "precontact", which the
    # bare substring rule misread as contact-sensing evidence and used to
    # invalidate trace evaluation for nine completed runs.
    policy_safe_receipt = {
        "motion_strategy": {
            "precontact_public_tcp_poses": [[0.1, 0.2, 0.3]],
            "mode": "global_precontact_then_local",
        },
        "controller_tracking": {"phases": {"global_precontact": {"completed": True}}},
    }
    assert _forbidden_path(policy_safe_receipt) is None

    # Boundary-delimited terms stay forbidden, including plural forms.
    assert _forbidden_path({"contact": {"body": "object-1"}}) == "$.contact"
    assert _forbidden_path({"contact_force_n": 1.2}) == "$.contact_force_n"
    assert _forbidden_path({"contacts": []}) == "$.contacts"
    assert _forbidden_path({"task_sources": ["x"]}) == "$.task_sources"
    assert _forbidden_path({"object_gt": {}}) == "$.object_gt"
    assert _forbidden_path({"segmentation_mask": "m"}) == "$.segmentation_mask"

    # Word-internal embeddings are public vocabulary, not evidence leaks.
    assert _forbidden_path({"presegmentation_stage": 1}) is None
    assert _forbidden_path({"precontact_offset_m": 0.05}) is None

    # The policy-state scanner shares the boundary semantics.
    assert _forbidden_policy_key({
        "motion_strategy": {"precontact_public_tcp_poses": [[0.1, 0.2, 0.3]]},
        "controller_tracking": {"phases": {"global_precontact": {}}},
    }) is None
    assert _forbidden_policy_key({"contact": {}}) == "$.contact"
    assert _forbidden_policy_key({"official_success": True}) == "$.official_success"

    # The policy-authored staged-approach argument is public motion
    # vocabulary in both scanners; values beneath it are still scanned.
    assert _forbidden_path({"targets": {"left": {"contact_approach": "top_down"}}}) is None
    assert _forbidden_policy_key({"contact_approach": "top_down"}) is None
    assert _forbidden_path(
        {"contact_approach": {"object_gt": {}}}
    ) == "$.contact_approach.object_gt"

    # The staged phase label `local_contact` is exempt only under `phases`;
    # elsewhere the standalone term stays forbidden, and phase values are
    # still scanned.
    phases_receipt = {"controller_tracking": {"phases": {"local_contact": {"completed": True}}}}
    assert _forbidden_path(phases_receipt) is None
    assert _forbidden_policy_key(phases_receipt) is None
    assert _forbidden_path({"local_contact": True}) == "$.local_contact"
    assert _forbidden_policy_key({"local_contact": True}) == "$.local_contact"
    assert _forbidden_path({
        "controller_tracking": {"phases": {"local_contact": {"object_gt": {}}}},
    }) == "$.controller_tracking.phases.local_contact.object_gt"


def test_libero_90_range_is_validated_by_runspec() -> None:
    from test_native_baseline import native_spec

    spec = native_spec("libero_pro")
    spec["runtime"].update(suite="libero_90", task_id=89)
    assert resolve_run_spec(spec, root=ROOT)["runtime"]["task_id"] == 89
    spec["runtime"]["task_id"] = 90
    with pytest.raises(RunSpecError, match=r"\[0,89\]"):
        resolve_run_spec(spec, root=ROOT)


def test_resident_sam_binding_accepts_the_motion_surface() -> None:
    segmenter = Sam3ApiSegmenter(expected_binding={
        "schema": "ora.segmentation-service-binding.v1",
        "tool_surface": "libero_placement_direct_overhead_v1",
        "endpoint": "http://127.0.0.1:8114",
        "instance_id": "model-instance-test",
        "pid": 123,
        "started_unix_s": 1.0,
        "adapter_sha256": "a" * 64,
        "model": {"family": "sam3"},
        "load_count": 1,
        "warmup_complete": True,
        "qualification": {
            "sha256": "b" * 64,
            "status": "passed",
            "created_unix_s": 1.0,
        },
    })
    try:
        assert segmenter.expected_binding["tool_surface"] == (
            "libero_placement_direct_overhead_v1"
        )
    finally:
        segmenter.close()


def test_hybrid_workspace_profile_names_the_actual_controller_pair() -> None:
    profile = resolve_workspace_profile("libero_pro_panda_hybrid_motion_v1")
    assert profile["provenance"]["controller"] == (
        "CUROBO_JOINT_POSITION+OSC_POSE"
    )
    assert profile["semantics"] == "hard_action_envelope_not_reachability"


def test_servo_delta_compiles_world_and_grip_site_frames() -> None:
    current = [
        0.4, 0.0, 0.5, 0.0, 0.0,
        math.sin(math.pi / 4), math.cos(math.pi / 4),
    ]
    tool = compile_servo_delta({
        "translation_m": [0.01, 0.0, 0.0],
        "rotation_rotvec_rad": [0.0, 0.0, 0.0],
        "frame": "grip_site",
    }, current_pose_xyz_xyzw=current)
    assert tool["resolved_translation_world_m"] == pytest.approx([0.0, 0.01, 0.0])
    world = compile_servo_delta({
        "translation_m": [0.01, 0.0, 0.0],
        "rotation_rotvec_rad": [0.0, 0.0, 0.0],
        "frame": "world",
    }, current_pose_xyz_xyzw=current)
    assert world["resolved_translation_world_m"] == pytest.approx([0.01, 0.0, 0.0])


@pytest.mark.parametrize(
    ("delta", "iterations"),
    [([0.051, 0.0, 0.0], 1), ([0.0, 0.0, 0.0], 21)],
)
def test_motion_action_rejects_unbounded_servo(
    delta: list[float], iterations: int,
) -> None:
    with pytest.raises(ActionValidationError):
        validate_runtime_action(_action("servo_delta", {
            "arm": "right", "target_pose": _pose(),
            "delta_pose": {
                "translation_m": delta,
                "rotation_rotvec_rad": [0.0, 0.0, 0.0],
            },
            "frame": "world", "max_iterations": iterations,
        }))


def test_gripper_and_recovery_contracts_are_discrete() -> None:
    close = validate_runtime_action(_action(
        "set_gripper", {"arm": "right", "gripper": 0.0},
    ))
    assert close["arguments"]["gripper"] == 0.0
    with pytest.raises(ActionValidationError):
        validate_runtime_action(_action(
            "set_gripper", {"arm": "right", "gripper": 0.4},
        ))
    recovered = validate_runtime_action(_action(
        "recover_motion", {"arm": "right", "strategy": "backtrack_last_safe"},
    ))
    assert recovered["arguments"]["strategy"] == "backtrack_last_safe"


def test_curobo_request_is_single_goal_and_has_no_scene_fields() -> None:
    request = validate_curobo_request({
        "joint_names": [f"panda_joint{index}" for index in range(1, 8)],
        "joint_positions": [0.0] * 7,
        "target_pose_base_xyzw": [0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0],
    })
    assert set(request) == {
        "joint_names", "joint_positions", "target_pose_base_xyzw",
    }
    with pytest.raises(CuroboPlannerError, match="fields drifted"):
        validate_curobo_request({**request, "voxel_pitch_m": 0.01})
    with pytest.raises(CuroboPlannerError, match="exactly one goal"):
        validate_curobo_request({
            **request, "goal_joint_positions": [0.0] * 7,
        })


def test_curobo_worker_uses_absent_world_not_empty_world_config() -> None:
    source = CUROBO_WORKER.read_text(encoding="utf-8")
    assert '"franka.yml", None, self.tensor_args' in source
    assert "WorldConfig()" not in source


@pytest.mark.parametrize("reason", ["IK_FAIL", "SELF_COLLISION"])
def test_curobo_unreachable_and_self_collision_are_typed(reason: str) -> None:
    response = validate_curobo_response({
        "schema": "ora.curobo-motion-response.v1",
        "status": "not_reachable", "reachable": False,
        "world_model": "none", "planner_status": reason,
    })
    assert response["reachable"] is False
    assert response["planner_status"] == reason


def test_curobo_malformed_trajectory_fails_closed() -> None:
    with pytest.raises(CuroboPlannerError, match="malformed"):
        validate_curobo_response({
            "schema": "ora.curobo-motion-response.v1",
            "status": "planned", "reachable": True, "world_model": "none",
            "trajectory": {
                "joint_names": [f"panda_joint{index}" for index in range(1, 8)],
                "positions": [[float("nan")] * 7], "dt_s": 0.02,
            },
        })


class _CpuTensor:
    """Tiny NumPy-backed tensor boundary; never imports Torch or CuRobo."""

    def __init__(self, values: Any) -> None:
        self.array = np.asarray(values, dtype=np.float32)
        self.shape, self.dtype = self.array.shape, self.array.dtype

    def detach(self) -> Any:
        return self

    def cpu(self) -> Any:
        return self

    def tolist(self) -> Any:
        return self.array.tolist()

    def numpy(self) -> Any:
        return self.array.copy()

    def item(self) -> Any:
        return self.array.item()

    def unsqueeze(self, axis: int) -> Any:
        return _CpuTensor(np.expand_dims(self.array, axis))

    def __getitem__(self, index: Any) -> Any:
        return _CpuTensor(self.array[index])


def _cpu_curobo_worker(monkeypatch: pytest.MonkeyPatch) -> Any:
    from ora.environment.backends.libero_pro.curobo_worker import (
        Worker, _motion_gen_class,
    )

    class MotionGenStatus(Enum):
        INVALID_START_STATE_JOINT_LIMITS = "joint limits"
        INVALID_START_STATE_SELF_COLLISION = "self collision"

    class JointState:
        @staticmethod
        def from_position(position: Any, *, joint_names: list[str]) -> Any:
            return SimpleNamespace(position=position, joint_names=joint_names)

    class Constraint:
        def __init__(self, evaluate: Any) -> None:
            self.enabled = True
            self.weight = _CpuTensor([1000.0])
            self.evaluate = evaluate

        def forward(self, state: Any) -> Any:
            return _CpuTensor([[self.evaluate(state)]])

    class Rollout:
        cuda_graph_instance = False
        start_state = None

        def __init__(self) -> None:
            self.constraint_cfg = SimpleNamespace(
                primitive_collision_cfg=None, bound_cfg=object(), self_collision_cfg=object(),
            )
            self.bound_constraint = Constraint(lambda state: float(np.maximum(
                limits[0] - state.position.array, 0,
            ).sum() + np.maximum(state.position.array - limits[1], 0).sum()))
            limits = np.asarray([
                [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
                [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
            ], dtype=np.float32)
            self.bound_constraint.joint_limits = SimpleNamespace(position=_CpuTensor(limits))
            self.bound_constraint.activation_distance = _CpuTensor([0.0] * 4)
            self.robot_self_collision_constraint = Constraint(
                lambda spheres: float(spheres.array[0, 0, 0] > 1.0),
            )
            self.dynamics_model = SimpleNamespace(forward=lambda _start, q: SimpleNamespace(
                state_seq=SimpleNamespace(position=q), robot_spheres=q,
            ))

        def constraint_fn(self, state: Any, *, use_batch_env: bool) -> Any:
            assert use_batch_env is False
            total = self.bound_constraint.forward(state.state_seq).item()
            total += self.robot_self_collision_constraint.forward(state.robot_spheres).item()
            return SimpleNamespace(constraint=_CpuTensor([[total]]), feasible=_CpuTensor([[total == 0]]))

    class MotionGen:
        def __init__(self) -> None:
            self.rollout_fn = Rollout()
            self.routes: list[str] = []

        def compute_kinematics(self, q: Any) -> Any:
            return SimpleNamespace(ee_pose=SimpleNamespace(
                position=_CpuTensor([[0.4, 0.0, 0.5]]), quaternion=_CpuTensor([[1.0, 0, 0, 0]]),
            ))

        def _plan(self, q: Any, route: str) -> Any:
            self.routes.append(route)
            valid, status = self.check_start_state(q)
            return SimpleNamespace(
                success=_CpuTensor([valid]), valid_query=valid, status=status,
                interpolation_dt=0.02,
                get_interpolated_plan=lambda: SimpleNamespace(
                    position=q.position, velocity=None, joint_names=q.joint_names,
                ),
            )

        def plan_single(self, q: Any, _goal: Any, _config: Any) -> Any:
            return self._plan(q, "pose")

        def plan_single_js(self, q: Any, _goal: Any, _config: Any) -> Any:
            return self._plan(q, "js")

    for name in ("curobo", "curobo.types", "curobo.wrap", "curobo.wrap.reacher"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    for name, members in {
        "curobo.types.math": {"Pose": SimpleNamespace},
        "curobo.types.robot": {"JointState": JointState},
        "curobo.wrap.reacher.motion_gen": {"MotionGenPlanConfig": SimpleNamespace},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(members)
        monkeypatch.setitem(sys.modules, name, module)
    worker = object.__new__(Worker)
    worker.tensor_args = SimpleNamespace(to_device=_CpuTensor)
    worker.motion_gen = _motion_gen_class(MotionGen, MotionGenStatus)()
    return worker


def _robot_only_request(route: str = "pose") -> dict[str, Any]:
    request = {
        "joint_names": [f"panda_joint{i}" for i in range(1, 8)],
        "joint_positions": [0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0],
    }
    if route == "pose":
        request["target_pose_base_xyzw"] = [0.4, 0, 0.5, 0, 0, 0, 1]
    else:
        request["goal_joint_positions"] = list(request["joint_positions"])
    return request


@pytest.mark.parametrize("route", ["pose", "js"])
@pytest.mark.parametrize("invalid", ["joint", "self", "both"])
def test_curobo_start_constraints_reject_then_plan_in_same_worker(
    monkeypatch: pytest.MonkeyPatch, route: str, invalid: str,
) -> None:
    worker = _cpu_curobo_worker(monkeypatch)
    request = _robot_only_request(route)
    if invalid in {"joint", "both"}:
        # The source-bound EXP08G q5 boundary value, never clamped.
        request["joint_positions"][4] = -2.8974711869540637
    if invalid in {"self", "both"}:
        request["joint_positions"][0] = 1.5
    original = deepcopy(request)
    refused = validate_curobo_response(worker.plan(request))
    expected = "JOINT_LIMITS" if invalid != "self" else "SELF_COLLISION"
    assert refused["planner_status"] == f"MotionGenStatus.INVALID_START_STATE_{expected}"
    assert refused["status"] == "not_reachable" and refused["valid_query"] is False
    assert refused["reachable"] is False and "trajectory" not in refused
    assert refused["request"] == original == request
    diagnostic = refused["start_state"]
    assert diagnostic["joint_positions"] == np.asarray(request["joint_positions"], dtype=np.float32).tolist()
    assert diagnostic["constraints"]["world_collision"]["configured"] is False
    assert diagnostic["constraints"]["self_collision"]["feasible"] == (invalid == "joint")
    assert diagnostic["constraints"]["joint_limits"]["feasible"] == (invalid == "self")
    rollout = worker.motion_gen.rollout_fn
    assert not hasattr(rollout, "primitive_collision_constraint")
    for constraint in (rollout.bound_constraint, rollout.robot_self_collision_constraint):
        assert constraint.enabled is True
        assert constraint.weight.tolist() == [1000.0]
    succeeded = validate_curobo_response(worker.plan(_robot_only_request(route)))
    assert succeeded["status"] == "planned" and succeeded["valid_query"] is True
    assert succeeded["start_state"]["status"] == "valid"
    assert worker.motion_gen.routes == [route, route]


@pytest.mark.parametrize("route", ["pose", "js"])
@pytest.mark.parametrize("fault", [
    "missing_bound", "missing_self", "disabled_self", "nan_weight", "nan_bound",
    "nan_self", "bound_exception", "self_exception", "aggregate_exception", "world_drift",
])
def test_curobo_start_diagnostic_fault_preserves_constraints_and_worker(
    monkeypatch: pytest.MonkeyPatch, route: str, fault: str,
) -> None:
    worker = _cpu_curobo_worker(monkeypatch)
    rollout = worker.motion_gen.rollout_fn
    bound, self_collision = rollout.bound_constraint, rollout.robot_self_collision_constraint
    def crash(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("injected diagnostic fault")

    with monkeypatch.context() as patch:
        if fault == "missing_bound":
            patch.delattr(rollout, "bound_constraint")
        elif fault == "missing_self":
            patch.delattr(rollout, "robot_self_collision_constraint")
        elif fault == "disabled_self":
            patch.setattr(self_collision, "enabled", False)
        elif fault == "nan_weight":
            patch.setattr(self_collision, "weight", _CpuTensor([float("nan")]))
        elif fault == "world_drift":
            patch.setattr(rollout.constraint_cfg, "primitive_collision_cfg", object())
        elif fault.startswith("nan_"):
            patch.setattr(bound if fault == "nan_bound" else self_collision, "forward", lambda *_: _CpuTensor([[float("nan")]]))
        else:
            owner = rollout if fault == "aggregate_exception" else bound if fault == "bound_exception" else self_collision
            patch.setattr(owner, "constraint_fn" if owner is rollout else "forward", crash)
        before = [(term.enabled, term.weight.tolist()) for term in (bound, self_collision)]
        result = validate_curobo_response(worker.plan(_robot_only_request(route)))
        assert result["status"] == "planner_error"
        assert result["reachable"] is False and "trajectory" not in result
        assert result["request"] == _robot_only_request(route)
        assert result["start_state"]["status"] == "checking"
        after = [(term.enabled, term.weight.tolist()) for term in (bound, self_collision)]
        for (before_enabled, before_weight), (after_enabled, after_weight) in zip(before, after, strict=True):
            assert before_enabled == after_enabled
            np.testing.assert_equal(before_weight, after_weight)
    assert validate_curobo_response(worker.plan(_robot_only_request(route)))["status"] == "planned"


@pytest.mark.parametrize("change", [
    {"status": "planner_error"}, {"status": "invented"},
    {"reachable": False}, {"valid_query": False},
    {"planner_status": "MotionGenStatus.IK_FAIL"},
    {"planner_status": ["success"]},
    {"error": {"type": "AttributeError", "message": "missing primitive collision constraint"}},
])
def test_curobo_response_rejects_contradictory_success(
    monkeypatch: pytest.MonkeyPatch, change: dict[str, Any],
) -> None:
    response = _cpu_curobo_worker(monkeypatch).plan(_robot_only_request())
    with pytest.raises(CuroboPlannerError):
        validate_curobo_response({**response, **change})


def test_curobo_valid_start_does_not_imply_valid_goal_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _cpu_curobo_worker(monkeypatch).plan(_robot_only_request())
    response.pop("trajectory")
    response.update(
        status="not_reachable", reachable=False, valid_query=False,
        planner_status="MotionGenStatus.INVALID_PARTIAL_POSE_COST_METRIC",
    )
    assert validate_curobo_response(response)["start_state"]["valid"] is True


@pytest.mark.parametrize("malformed", ["names", "quaternion", "nonfinite", "extra"])
def test_curobo_worker_rejects_request_before_touching_constraints(
    monkeypatch: pytest.MonkeyPatch, malformed: str,
) -> None:
    worker = _cpu_curobo_worker(monkeypatch)
    request = _robot_only_request()
    if malformed == "names":
        request["joint_names"] = list(reversed(request["joint_names"]))
    elif malformed == "quaternion":
        request["target_pose_base_xyzw"][6] = 0.0
    elif malformed == "nonfinite":
        request["joint_positions"][4] = float("nan")
    else:
        request["scene"] = {}
    response = validate_curobo_response(worker.plan(request))
    assert response["status"] == "planner_error" and response["reachable"] is False
    assert response["request"] is None and response["start_state"] is None
    assert worker.motion_gen.routes == []
    assert validate_curobo_response(worker.plan(_robot_only_request()))["reachable"] is True


def test_planner_request_ignores_closed_gripper_and_has_no_payload(
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class Planner:
        def plan(self, request: Mapping[str, Any]) -> dict[str, Any]:
            captured.update(request)
            return {
                "schema": "ora.curobo-motion-response.v1",
                "status": "not_reachable", "reachable": False,
                "world_model": "none", "planner_status": "IK_FAIL",
                "start_ee_pose_base_xyzw": [
                    0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0,
                ],
            }

    session = object.__new__(LiberoProSession)
    session.execution_capture = {"enabled": False}  # This fixture has no live camera worker.
    session.motion_planner = Planner()
    session.output_root = tmp_path
    session.action_sequence = 0
    session.sim_steps = 0
    session.last_receipt = None
    session.workspace_bounds_m = {
        "x": [-0.6, 0.6], "y": [-0.5, 0.5], "z": [-0.1, 1.3],
    }
    session._require_environment = lambda: SimpleNamespace()
    session._current_state = lambda: (
        np.array([0.4, 0.0, 0.5]),
        np.array([0.0, 0.0, 0.0, 1.0]),
        0.2,
    )
    session._motion_state = lambda: {
        "joint_names": [f"panda_joint{index}" for index in range(1, 8)],
        "joint_positions": [0.0] * 7,
        "base_pose_world_xyzw": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "hand_pose_world_xyzw": [0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0],
        "grip_site_pose_world_xyzw": [
            0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0,
        ],
    }
    receipt = session._execute_planned_motion(_action(
        "move_free_space", {"arm": "right", "target_pose": _pose()},
    ))
    assert set(captured) == {
        "joint_names", "joint_positions", "target_pose_base_xyzw",
    }
    assert receipt["motion_status"] == "refused"
    assert receipt["feedback"]["planning_failure_reason"] == "IK_FAIL"


@pytest.mark.parametrize("reason,valid_query", [
    ("planner_error", None),
    ("MotionGenStatus.INVALID_START_STATE_JOINT_LIMITS", False),
    ("MotionGenStatus.INVALID_START_STATE_SELF_COLLISION", False),
    ("MotionGenStatus.IK_FAIL", True),
])
def test_planner_failure_without_fk_is_consistent_in_probe_and_execute(
    tmp_path: Path, reason: str, valid_query: bool | None,
) -> None:
    response = {
        "schema": "ora.curobo-motion-response.v1",
        "status": "planner_error" if reason == "planner_error" else "not_reachable",
        "reachable": False, "world_model": "none", "valid_query": valid_query,
    }
    if reason == "planner_error":
        response["error"] = {
            "type": "AttributeError",
            "message": "'ArmReacher' object has no attribute 'primitive_collision_constraint'",
        }
    else:
        response["planner_status"] = reason

    class Planner:
        def plan(self, request: Mapping[str, Any]) -> dict[str, Any]:
            return validate_curobo_response({**response, "request": dict(request)})

    session = object.__new__(LiberoProSession)
    session.execution_capture = {"enabled": False}  # This fixture has no live camera worker.
    session.motion_planner = Planner()
    session.output_root = tmp_path
    session.action_sequence = 0
    session.sim_steps = 0
    session.last_receipt = None
    session.world_mutation_sequence = 0
    session.motion_probe_sequence = 0
    session.plan_references = {}
    session.workspace_bounds_m = {
        "x": [-0.6, 0.6], "y": [-0.5, 0.5], "z": [-0.1, 1.3],
    }
    session._require_environment = lambda: SimpleNamespace()
    session._current_state = lambda: (
        np.array([0.4, 0.0, 0.5]),
        np.array([0.0, 0.0, 0.0, 1.0]),
        0.2,
    )
    session._motion_state = lambda: {
        "joint_names": [f"panda_joint{index}" for index in range(1, 8)],
        "joint_positions": [0.0] * 7,
        "base_pose_world_xyzw": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "hand_pose_world_xyzw": [0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0],
        "grip_site_pose_world_xyzw": [
            0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0,
        ],
    }
    receipt = session._execute_planned_motion(_action(
        "move_free_space", {"arm": "right", "target_pose": _pose()},
    ))
    assert receipt["motion_status"] == "refused"
    assert receipt["state_changed"] is False
    assert receipt["steps_used"] == 0
    assert receipt["feedback"]["planning_failure_reason"] == reason
    assert receipt["planner"]["status"] == response["status"]
    assert receipt["planner"]["valid_query"] is valid_query
    execution_evidence = json.loads((
        tmp_path / receipt["planner"]["canonical_artifact"]["path"]
    ).read_text())
    assert execution_evidence["request"]["joint_positions"] == [0.0] * 7
    for sequence in range(2):
        result = session.probe_motion({
            "schema": "ora.motion-probe-request.v1",
            "observation_id": "observation-0", "observation_sha256": "0" * 64,
            "queries": [{"query_id": "query-0", "arm": "right", "pose": _pose()}],
            "variant_policy": "exact",
        })
        assert result["schema"] == "ora.motion-probe-result.v1"
        assert result["robot_decisions_consumed"] == 0
        assert result["world_state_changed"] is False
        row = result["results"][0]
        assert row["reachable"] is False and row["plan_ref"] is None
        variant = row["variants"][0]
        assert variant["failure"]["code"] == reason
        assert variant["planner"]["status"] == reason
        assert variant["planner"]["valid_query"] is valid_query
        if reason == "planner_error":
            assert variant["planner"]["error"] == {"stage": "planner", **response["error"]}
        probe_evidence = json.loads((
            tmp_path / f"curobo/probes/request-{sequence:06d}.json"
        ).read_text())
        assert probe_evidence == execution_evidence
    assert session.world_mutation_sequence == session.sim_steps == 0
    assert session.plan_references == {}


def test_full_joint_plan_is_canonical_but_not_policy_visible(tmp_path: Path) -> None:
    session = object.__new__(LiberoProSession)
    session.execution_capture = {"enabled": False}  # This fixture has no live camera worker.
    session.output_root = tmp_path
    session.action_sequence = 0
    summary = session._persist_planner_evidence({
        "schema": "ora.curobo-motion-response.v1",
        "status": "planned", "reachable": True,
        "world_model": "none", "planner_status": "SUCCESS",
        "trajectory": {
            "joint_names": [f"panda_joint{index}" for index in range(1, 8)],
            "positions": [[0.0] * 7, [0.05] * 7], "dt_s": 0.02,
        },
    })
    assert summary["world_model"] == "none"
    assert summary["trajectory"]["waypoint_count"] == 2
    assert "positions" not in summary["trajectory"]
    artifact = tmp_path / summary["canonical_artifact"]["path"]
    assert "0.05" in artifact.read_text(encoding="utf-8")


def test_motion_profile_is_hash_bound_and_gripper_is_fixed() -> None:
    binding = load_hybrid_motion_profile(MOTION_PROFILE, MOTION_PROFILE_SHA256)
    profile = binding["profile"]
    assert profile["gripper"]["open_directive"] == 0.0
    assert profile["gripper"]["close_directive"] == 1.0
    assert profile["servo"]["max_iterations"] == 20
    assert profile["sensing"]["force_telemetry_threshold_n"] == 40.0
    drifted = dict(profile)
    drifted["gripper"] = {**profile["gripper"], "close_directive": 0.5}
    with pytest.raises(HybridMotionProfileError):
        validate_hybrid_motion_profile(drifted)


def test_contact_aggregates_blocked_closure_without_payload_inference() -> None:
    assert aggregate_motion_contact(
        external_force_xyz=[0.0, 0.0, 5.0], force_threshold_n=10.0,
        closing=True, measured_aperture_normalized=0.4,
        closure_contact_min_normalized=0.08,
    ) is True
    assert aggregate_motion_contact(
        external_force_xyz=[0.0, 0.0, 5.0], force_threshold_n=10.0,
        closing=False, measured_aperture_normalized=0.4,
        closure_contact_min_normalized=0.08,
    ) is False


def test_joint_executor_stop_is_operator_owned_and_force_is_telemetry(
    tmp_path: Path,
) -> None:
    sentinel = tmp_path / "control/stop.request"
    assert motion_stop_reason(
        stop_request_path=sentinel,
    ) is None
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("operator\n", encoding="utf-8")
    assert motion_stop_reason(
        stop_request_path=sentinel,
    ) == "operator_stop_requested"


def test_joint_position_command_is_bounded() -> None:
    command = bounded_joint_position_command(
        [0.0] * 7, [0.2, -0.2, 0.025, 0.0, 0.0, 0.0, 0.0],
        maximum_step_rad=0.05,
    )
    assert command == pytest.approx([1.0, -1.0, 0.5, 0.0, 0.0, 0.0, 0.0])


def test_backtrack_state_follows_each_non_recovery_action_start() -> None:
    session = object.__new__(LiberoProSession)
    session.execution_capture = {"enabled": False}  # This fixture has no live camera worker.
    session.last_backtrack_joint_state = None
    state = {"joint_positions": [0.1] * 7}
    session._motion_state = lambda: dict(state)
    session._remember_backtrack_state()
    assert session.last_backtrack_joint_state == [0.1] * 7
    state["joint_positions"] = [-0.2] * 7
    session._remember_backtrack_state()
    assert session.last_backtrack_joint_state == [-0.2] * 7


def test_the_observation_delta_no_longer_compares_motion_layers(
    tmp_path: Path,
) -> None:
    """Two actions are never judged equivalent by the Harness.

    The retired guard decided whether a `move_free_space`, `servo_delta` or
    `recover_motion` proposal repeated the previous one, using position and
    orientation epsilons that were bound to no profile. That comparison, and
    the repetition streak built on it, are gone; what the Supervisor records
    after an action is one image distance.
    """

    guard = StagnationGuard(tmp_path / "stagnation.json")
    assert not hasattr(guard, "_equivalent")
    assert not hasattr(guard, "is_high_confidence")
    assert not any(
        hasattr(guard, name) for name in (
            "position_epsilon_m", "orientation_epsilon_deg", "gripper_epsilon",
            "eef_motion_epsilon_m", "image_phash_hamming_max",
            "warning_at", "high_confidence_at",
        )
    )


def _curobo_diagnostic_response():
    return {
        "schema": "ora.curobo-motion-response.v1",
        "status": "not_reachable", "reachable": False,
        "world_model": "none", "planner_status": "MotionGenStatus.IK_FAIL",
        "planning_diagnostics": {
            "phase": "ik", "outcome": "no_converged_candidate",
            "candidate_counts": {"returned": 32, "accepted": 0, "requested": None},
            "rejection_counts": {"position_error": 32, "self_collision": None},
            "details_available": True, "recovery": ["Consider an intermediate target."],
        },
        "planning_diagnostic_evidence": {"attempts": [{"joint_positions": [[.1] * 7]}]},
    }


def test_curobo_wire_preserves_unknown_counts_and_private_evidence_without_aliasing():
    response = _curobo_diagnostic_response()
    validated = validate_curobo_response(response)
    assert validated == response
    assert validated["planning_diagnostics"]["rejection_counts"]["self_collision"] is None
    assert validated["reachable"] is False
    validated["planning_diagnostics"]["candidate_counts"]["returned"] = 0
    validated["planning_diagnostic_evidence"]["attempts"][0]["joint_positions"][0][0] = 9
    assert response["planning_diagnostics"]["candidate_counts"]["returned"] == 32
    assert response["planning_diagnostic_evidence"]["attempts"][0]["joint_positions"][0][0] == .1


@pytest.mark.parametrize("change", [
    {"phase": []}, {"phase": "unrecognized"}, {"details_available": 1},
    {"candidate_counts": {"returned": True}}, {"candidate_counts": {"returned": -1}},
    {"rejection_counts": {"self_collision": float("nan")}},
    {"recovery": ["hint"] * 5}, {"unknown_reasons": "unknown"},
    {"raw_seed_positions": [[0.0] * 7]},
])
def test_curobo_wire_rejects_malformed_planning_diagnostics(change):
    response = _curobo_diagnostic_response()
    response["planning_diagnostics"].update(change)
    with pytest.raises(CuroboPlannerError, match="diagnostics are malformed"):
        validate_curobo_response(response)


@pytest.mark.parametrize("evidence", [
    [], None, {"invalid": float("nan")}, {"invalid": object()}, {"oversized": "x" * 1_000_001},
])
def test_curobo_wire_rejects_invalid_or_oversized_raw_diagnostic_evidence(evidence):
    response = _curobo_diagnostic_response()
    response["planning_diagnostic_evidence"] = evidence
    with pytest.raises(CuroboPlannerError, match="diagnostic evidence is invalid or oversized"):
        validate_curobo_response(response)
