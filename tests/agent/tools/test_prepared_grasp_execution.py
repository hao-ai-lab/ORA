"""Prepared grasp regression: source binding, routing, staged effects and evidence."""

from dataclasses import replace
import json

import numpy as np
import pytest

from ora.agent.tools import skill_driver, skill_ledger, skill_retention
from ora.contracts import MOTION_ACTION_V3_SCHEMA, validate_runtime_action
from ora.orchestrator.action_executor import execute_skill_sequence
from tests.agent.tools.test_manipulation_skill_driver import (
    CONTACT, _runtime, _publish_grasp_candidate, _observation_id, _submitted, _driven,
)
from tests.agent.tools.test_skill_sequence_executor import _session


def _prepared(tmp_path):
    host = _runtime(tmp_path)
    host.surface = replace(host.surface, capabilities=(*host.surface.capabilities, skill_retention.CAPABILITY))
    _publish_grasp_candidate(host, observation_id=_observation_id(host))
    host.motion_probe_callback = lambda request: {
        "schema": "ora.motion-probe-result.v1",
        "results": [{"query_id": query["query_id"], "arm": query["arm"],
                     "reachable": True, "variants": []} for query in request["queries"]],
    }
    return host


def _begin(host):
    return host.invoke("grasp_execute", {"candidate_id": "grasp-a1", "rationale": "chosen target; opening and short lift are appropriate"})


def test_prepared_grasp_commits_candidate_poses_opening_and_bounded_lift(tmp_path):
    host = _prepared(tmp_path)
    result = _begin(host)
    assert result["committed_sequence"] == list(skill_ledger.PREPARED_GRASP_STAGES)
    assert result["cost"] == {"robot_decisions": 1, "dispatched_stage_actions": 5}
    commitment = skill_driver.load_commitment(host)
    assert commitment["grippers"] == {"open": 1.0, "close": 0.0}
    assert commitment["poses"]["pregrasp"]["position_m"] == [.2, .1, .96]
    assert commitment["poses"]["approach"] == CONTACT
    assert commitment["poses"]["lift"]["position_m"] == pytest.approx([.2, .1, .93])
    assert commitment["poses"]["lift"]["orientation_xyzw"] == CONTACT["orientation_xyzw"]


@pytest.mark.parametrize("arm", ["left", "right"])
def test_robotwin_prepared_stages_route_every_action_to_candidate_tcp(tmp_path, arm):
    from ora.agent.composition import resolve_tool_surface
    host = _prepared(tmp_path)
    host.surface = resolve_tool_surface("robotwin_toolchain_b2_closed_noservo_v1")
    host.allowed_tools = tuple(item["name"] for item in host.surface.definitions)
    path = host.artifact_root / "grasp-queries/grasp-query-fixture/manifest.json"
    body = json.loads(path.read_text())
    body["identity"]["compiler"] = {"arm": arm, "controlled_point_semantics": "parallel_jaw_grip_center_tcp"}
    path.write_text(json.dumps(body))
    _begin(host)
    action = _submitted(host)
    commitment = skill_driver.load_commitment(host)
    observation = {"observation_id": "post-close"}
    (host.control_root / "observation.json").write_text(json.dumps(observation))
    for index, stage in enumerate(skill_ledger.PREPARED_GRASP_STAGES[:-1]):
        result = host.drive_skill_stage(action, stage=stage, stage_index=index,
            observation=observation, observation_root=host.control_root,
            completed=[{"stage": "close", "status": "executed"}] if stage == "lift" else [],
            capture_observation=lambda: (observation, host.control_root))
        driven = validate_runtime_action(result["action"])
        assert driven["schema"] == MOTION_ACTION_V3_SCHEMA
        assert set(driven["arguments"]["targets"]) == {arm}
        target = driven["arguments"]["targets"][arm]
        assert target == (stage if stage in {"open", "close"} else commitment["poses"][stage])


def test_missing_pregrasp_is_refused_before_opening_or_submission(tmp_path):
    from ora.agent.tools.robot import ToolError

    host = _prepared(tmp_path)
    path = host.artifact_root / "grasp-queries/grasp-query-fixture/manifest.json"
    candidate = json.loads(path.read_text())
    del candidate["compiled_candidates"][0]["pregrasp_pose"]
    path.write_text(json.dumps(candidate))
    with pytest.raises(ToolError, match="pregrasp_pose"):
        _begin(host)
    assert not (host.control_root / "submitted-action.json").exists()


def test_a_submitted_subset_cannot_skip_open_and_pregrasp(tmp_path):
    host = _prepared(tmp_path)
    _begin(host)
    action = _submitted(host)
    action["arguments"]["sequence"] = ["approach", "close", "verify"]
    refused = _driven(host, action, "approach", 0, {})
    assert refused["kind"] == "refusal"
    assert refused["reason"] == "committed_stage_sequence_mismatch"


@pytest.mark.parametrize("failed_lift", [False, True])
def test_prepared_sequence_executes_all_stages_and_records_post_close_baseline(tmp_path, failed_lift):
    host = _prepared(tmp_path)
    _begin(host)
    backend = _session(tmp_path / "backend", max_actions=16)
    original_execute = backend.execute

    def execute(payload):
        result = original_execute(payload)
        if failed_lift and "-lift" in payload["action_id"]:
            result["receipt"].pop("commanded_pose_xyz_xyzw", None)
        return result

    backend.execute = execute
    root = tmp_path / "observations"
    initial, _ = backend.observe(root)

    def capture():
        observation, _ = backend.observe(root)
        return observation, root / observation["observation_id"]

    receipt, (final, _) = execute_skill_sequence(
        backend, _submitted(host), initial_observation=initial,
        initial_observation_root=root / initial["observation_id"],
        drive_stage=host.drive_skill_stage, build_receipt=host.build_skill_receipt,
        capture_observation=capture,
    )
    assert receipt["status"] == ("aborted" if failed_lift else "completed")
    assert receipt["dispatched_stage_actions"] == 5
    evidence = receipt["skill_receipt"]["observation_evidence"]
    assert evidence[0]["role"] == "post_close_before_lift"
    assert evidence[0]["observation_id"] != final["observation_id"]
    if failed_lift:
        assert receipt["stop_reason"] == "lift_not_reached"
        assert receipt["skill_receipt"]["verification"] is None
    else:
        assert receipt["skill_receipt"]["verification"]["lift_probe"]["surface_motion"]["status"] == "unknown"
    assert skill_driver.load_ledger(host)["phase"] == receipt["status"]


@pytest.mark.parametrize("profile", [
    "robotwin_toolchain_b2_closed_noservo_v2", "robodojo_placement_closed_skills_v2",
    "robotwin_toolchain_b2_closed_noservo_v1",
])
def test_maintained_defaults_bind_prepared_grasp_and_release_disclosure(profile):
    from ora.agent.composition import resolve_tool_surface
    surface = resolve_tool_surface(profile)
    assert surface.supports(skill_retention.CAPABILITY)
    help_text = next(d["description"] for d in surface.definitions if d["name"] == "grasp_execute")
    assert "releases anything currently held" in help_text
    assert "five stage actions" in help_text
    assert "completed sequence only records execution" in help_text


@pytest.mark.parametrize("mutation", ["invalid_quaternion", "outside_workspace", "no_approach"])
def test_prepared_geometry_rejects_before_opening(tmp_path, mutation):
    from ora.agent.tools.robot import ToolError
    host = _prepared(tmp_path)
    path = host.artifact_root / "grasp-queries/grasp-query-fixture/manifest.json"
    body = json.loads(path.read_text())
    candidate = body["compiled_candidates"][0]
    if mutation == "invalid_quaternion":
        candidate["contact_pose"]["orientation_xyzw"] = [0., 0., 0., 0.]
    elif mutation == "outside_workspace":
        candidate["contact_pose"]["position_m"][2] = 1.44  # lift would exit the workspace
    else:
        candidate["pregrasp_pose"] = candidate["contact_pose"]
    path.write_text(json.dumps(body))
    with pytest.raises(ToolError):
        _begin(host)
    assert not (host.control_root / "submitted-action.json").exists()


@pytest.mark.parametrize("case,expected", [
    ("moving", "rigid_motion_supported"), ("stationary", "stationary_surface_supported"),
    ("both", "unknown"), ("occluded", "unknown"), ("baseline_moved", "unknown"),
    ("tiny_lift", "unknown"),
])
def test_surface_hypotheses_preserve_motion_and_ambiguity(case, expected):
    x, y = np.meshgrid(np.linspace(-.02, .02, 9), np.linspace(-.02, .02, 9))
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    delta = np.array([0., 0., .003 if case == "tiny_lift" else .03])
    after = points + delta
    before = points.copy()
    if case == "stationary":
        after = points.copy()
    elif case == "both":
        after = np.vstack((points, after))
    elif case == "occluded":
        after = after[:4]
    elif case == "baseline_moved":
        before += [.1, 0., 0.]
    evidence = skill_retention.compare_surfaces(
        points=points, before_scene=before, after_scene=after,
        before_position=np.zeros(3), after_position=delta,
        before_rotation=np.eye(3), after_rotation=np.eye(3),
    )
    assert evidence["status"] == expected
    assert "semantic_identity_unverified" in evidence["scope"]
