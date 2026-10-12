"""Report measured IK filter counts without inventing missing solver causes."""
from __future__ import annotations

from types import SimpleNamespace
import hashlib
import json

import numpy as np
import pytest

from ora.environment.backends.robotwin.planning import planner_public_summary
from ora.environment.backends.robotwin.session import RoboTwinLiveSession
from ora.environment.controllers.profile_capabilities import PROFILE_VERSION_IDS
from test_robotwin_correctness_repair import vendored_planner_with_fake_solver


def _plan(solutions, successes, *, expose_limits=True):
    # Execute the actual vendored class with the existing CPU-only solver port.
    planner = vendored_planner_with_fake_solver(ValueError)
    namespace = planner._plan_pose_via_nearest_ik.__globals__
    tensor = namespace["torch"].tensor

    class IK:
        def __init__(self, _config):
            self.kinematics = SimpleNamespace()
            if expose_limits:
                self.kinematics.get_joint_limits = lambda: SimpleNamespace(
                    position=[[-1] * 6, [1] * 6],
                )

        def solve_single(self, *args, **kwargs):
            return SimpleNamespace(
                success=successes, solution=tensor(solutions),
                position_error=None, rotation_error=None,
            )

    namespace["IKSolver"] = IK
    return planner._plan_pose_via_nearest_ik(object(), [0.0] * 6, object())


def test_actual_ik_filters_count_each_rejection_once_before_deduplication():
    result = _plan([
        [float("nan")] * 6, [.1, 0, 0, 0, 0, 0], [.2, 0, 0, 0, 0, 0],
        [2., 0, 0, 0, 0, 0], [.1, 0, 0, 0, 0, 0],
    ], [False, True, False, True, True])
    ik = result["ik"]
    assert ik["constraint_rejection_counts"] == {
        "non_finite_solution": 1, "solver_unsuccessful": 1, "joint_limit": 1,
    }
    assert ik["returned_solution_count"] == 5
    assert ik["valid_solution_count"] == 2
    assert ik["deduplicated_solution_count"] == ik["planned_candidate_count"] == 1
    assert np.allclose(result["candidates"][0]["ik_joint_goal"], [.1, 0, 0, 0, 0, 0])
    rows = ik["candidate_validation"]
    assert [row["candidate_index"] for row in rows] == list(range(5))
    assert [row["rejection"] for row in rows] == [
        "non_finite_solution", None, "solver_unsuccessful", "joint_limit", None,
    ]
    assert rows[3]["violating_joints"][0]["excess_rad"] == 1.0
    assert rows[3]["violating_joints"][0]["value_rad"] == 2.0
    assert rows[2]["joint_limit_check"] == "unknown"
    assert ik["planned_source_indices"] == [1]
    compact = planner_public_summary(result)["planning_diagnostics"]
    assert compact["details_available"] is True
    assert compact["candidate_counts"]["returned"] == 5
    assert compact["rejection_counts"] == ik["constraint_rejection_counts"]
    assert "candidate_validation" not in compact and "violating_joints" not in repr(compact)


def test_all_solver_failures_keep_counts_without_claiming_a_collision_cause():
    result = _plan([[.1] * 6, [.2] * 6], [False, False])
    assert result["planner_status"] == "IK_FAIL"
    assert result["ik"]["constraint_rejection_counts"] == {
        "non_finite_solution": 0, "solver_unsuccessful": 2, "joint_limit": 0,
    }
    assert "constraint_rejection_counts" not in planner_public_summary(result)
    public = planner_public_summary(result, include_constraint_rejection_counts=True)
    assert public["constraint_rejection_counts"]["solver_unsuccessful"] == 2
    assert "collision" not in public["constraint_rejection_counts"]
    compact = public["planning_diagnostics"]
    assert compact["outcome"] == "no_valid_endpoint"
    assert compact["unknown_reasons"] == ["solver_failure_detail"]
    assert "does not identify collision" in compact["recovery"][0]
    assert [row["solver_failure_detail"] for row in result["ik"]["candidate_validation"]] == ["unknown", "unknown"]


def test_missing_limit_and_misaligned_success_data_are_unmeasured_not_zero():
    result = _plan([[.1] * 6, [.2] * 6], [True], expose_limits=False)
    assert result["ik"]["constraint_rejection_counts"] == {
        "non_finite_solution": 0, "solver_unsuccessful": None, "joint_limit": None,
    }
    public = planner_public_summary(result, include_constraint_rejection_counts=True)
    assert public["constraint_rejection_counts"] == {"non_finite_solution": 0}
    missing = planner_public_summary(
        {"status": "Fail", "planner_status": "IK_FAIL"},
        include_constraint_rejection_counts=True,
    )
    assert "constraint_rejection_counts" not in missing
    assert missing["planning_diagnostics"]["candidate_counts"]["returned"] is None
    assert missing["planning_diagnostics"]["rejection_counts"]["joint_limit"] is None
    assert missing["planning_diagnostics"]["details_available"] is False
    assert result["ik"]["candidate_validation"][0]["solver_success"] is None
    assert result["ik"]["candidate_validation"][0]["joint_limit_check"] == "unknown"


def test_selected_topk_retains_compact_ik_diagnostics_without_seed_rows():
    result = _plan([[.1] * 6, [.2] * 6], [False, False])
    selected = {"status": "Fail", "topk_selection": {"ik": result["ik"]}}
    public = planner_public_summary(selected)
    assert public["planning_diagnostics"]["candidate_counts"]["retained"] == 0
    assert "candidate_validation" not in public["planning_diagnostics"]


def test_missing_ik_counters_remain_unknown_not_false_zero():
    result = {"status": "Fail", "planner_status": "IK_FAIL", "ik": {
        "seed_count": True, "returned_solution_count": -1,
        "constraint_rejection_counts": {"solver_unsuccessful": "128"},
    }}
    compact = planner_public_summary(result)["planning_diagnostics"]
    assert compact["candidate_counts"]["requested"] is None
    assert compact["candidate_counts"]["returned"] is None
    assert compact["rejection_counts"]["solver_unsuccessful"] is None
    assert compact["outcome"] == "unknown"


def test_failed_ik_keeps_full_seed_records_in_hashed_artifact(tmp_path):
    raw = _plan([[.1] * 6, [2.0] * 6], [False, True])
    assert raw["planner_status"] == "IK_FAIL"
    session = object.__new__(RoboTwinLiveSession)
    session.output_root = tmp_path
    session.source_manifest = {"robotwin_revision": "test-source"}
    session.control_profile = {"sha256": "a" * 64}
    identity = {"kind": "action", "action_id": "act-1", "action_sequence": 1}
    session._record_ik_candidate_evidence(raw, arm="left", identity=identity)
    reference = raw["ik_validation_artifact"]
    path = tmp_path / reference["path"]
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == reference["sha256"]
    saved = json.loads(data)
    assert saved["identity"] == identity
    assert saved["candidate_validation"][0]["rejection"] == "solver_unsuccessful"
    assert saved["candidate_validation"][1]["rejection"] == "joint_limit"
    assert len(saved["candidate_validation"][1]["violating_joints"]) == 6
    session._record_ik_candidate_evidence(raw, arm="left", identity=identity)
    assert path.read_bytes() == data
    public = session._planner_record(raw, called=True)
    assert public["ik_validation_artifact"] == reference
    assert "candidate_validation" not in public
    assert "candidate_validation" not in public["planning_diagnostics"]


@pytest.mark.parametrize("version", [9, 10])
def test_probe_exposes_measured_ik_counts_only_on_new_control_profile(version):
    counts = {"non_finite_solution": 0, "solver_unsuccessful": 128, "joint_limit": 0}
    raw = {"schema": "robotwin.curobo-plan-result.v1", "status": "Fail",
           "planner_status": "IK_FAIL", "ik": {"constraint_rejection_counts": counts}}
    environment = SimpleNamespace(robot=SimpleNamespace(left_plan_multi_path=lambda targets: {
        "schema": "robotwin.curobo-topk-batch-result.v1", "results": [raw] * len(targets),
    }))
    session = object.__new__(RoboTwinLiveSession)
    session.control_profile = {"profile": {"profile_id": PROFILE_VERSION_IDS[version], "plan_validation": {}}}
    session.workspace_bounds_m = {"x": [-1, 1], "y": [-1, 1], "z": [0, 2]}
    session.world_mutation_sequence = 0
    session._require_environment = lambda: environment
    session._uses_motion_transaction_profile = lambda: False
    session._measured_arm_joints = lambda _env, _arm: [0] * 6
    session._public_pose_wxyz = lambda _env, _arm: [0, 0, 1, 1, 0, 0, 0]
    session._public_to_planner_wxyz = lambda _arm, pose: pose
    session._arm_joint_limits = lambda _env, _arm: [[-1, 1]] * 6
    session._arm_profile = lambda _arm: {"planner_joint_names": [f"j{i}" for i in range(6)]}
    result = session.probe_motion({
        "schema": "ora.motion-probe-request.v1", "observation_id": "obs-1",
        "observation_sha256": "a" * 64, "variant_policy": "exact",
        "queries": [{"query_id": "q", "arm": "left", "pose": {
            "position_m": [0, 0, 1], "orientation_xyzw": [0, 0, 0, 1],
        }}],
    })
    planner = result["results"][0]["variants"][0]["planner"]
    if version == 10:
        assert planner["constraint_rejection_counts"] == counts
    else:
        assert "constraint_rejection_counts" not in planner
    assert result["world_state_changed"] is False and result["robot_decisions_consumed"] == 0
