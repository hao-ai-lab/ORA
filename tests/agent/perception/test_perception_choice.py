"""Agent-owned decisions over source-bound public perception, across both embodiments."""

import json

import pytest

from ora.agent.composition import validate_schema_instance
from ora.agent.tools import retention
from ora.agent.tools.robot import ToolError

#: The kept LIBERO default (single arm, flat targets).
LIBERO = "libero_placement_direct_overhead_v1"

def advisory_run(tmp_path):
    """The kept RoboTwin surface that ships the advisory capability with grasp_verify_retention."""
    from tests.environments.robotwin.test_robotwin_placement_adapter import RoboTwinPublicRun
    run = RoboTwinPublicRun(tmp_path, bind_submitted=True)
    assert run.runtime.surface.supports(retention.ADVISORY_CAPABILITY)
    return run


def _submitted(run):
    return json.loads((run.runtime.control_root / "submitted-action.json").read_text())


def _object(run):
    return run.frames[-1]["regions"]["observer_camera"]["object"]


def _lift(run, z=1.025):
    from tests.environments.robotwin.test_robotwin_placement_adapter import ABSOLUTE, pose_target
    probe = pose_target("left", z)
    run.runtime.invoke("move_pose", {"targets": {"left": {**probe, **ABSOLUTE, "interaction": "precision"}}})
    run.publish({"left": probe}, name="move_pose")


def test_unknown_is_evidence_and_never_writes_or_reads_a_motion_permit(tmp_path):
    run = advisory_run(tmp_path)
    anchor = _object(run)
    run.runtime.invoke("set_gripper", {"targets": {"left": "close"}, "target_ref": anchor})
    run.publish({"left": "close"})
    _lift(run)
    # The pre-close anchor is not a transport baseline: the comparison is unknown.
    result = run.runtime.invoke("grasp_verify_retention", {
        "arm": "left", "before_target_ref": anchor, "current_target_ref": _object(run)})
    assert result["status"] == "unknown"
    assert result["assessment_role"] == "advisory_only"
    assert "execution_gate" not in result and "recovery" not in result
    assert result["surface_comparison"] and result["thresholds"]
    from tests.environments.robotwin.test_robotwin_placement_adapter import ABSOLUTE, pose_target
    run.runtime.invoke("move_pose", {"targets": {"left": {
        **pose_target("left", 1.15), **ABSOLUTE, "interaction": "free_space"}}})
    assert _submitted(run)["arguments"]["interactions"] == {"left": "free_space"}


def test_repeat_close_and_optional_unverified_intent_do_not_require_geometry(tmp_path):
    run = advisory_run(tmp_path)
    submitted = []
    persist = run.runtime._persist_action

    def record(path, control, value):
        submitted.append(value)
        return persist(path, control, value)

    run.runtime._persist_action = record
    run.runtime._current_target_geometry = lambda **_: pytest.fail("closure must not require segmentation")
    first = run.runtime.invoke("set_gripper", {"targets": {"left": "close"},
                                               "target_ref": "old-or-unavailable-intent"})
    assert first["intent"] == {"target_ref": "old-or-unavailable-intent", "role": "agent_annotation_only"}
    run.publish({"left": "close"})
    second = run.runtime.invoke("set_gripper", {"targets": {"left": "close"}})
    assert second["intent"]["target_ref"] is None
    assert len(submitted) == 2
    assert all(a["arguments"]["targets"] == {"left": "close"} for a in submitted)


def test_advisory_preserves_current_target_and_round_source_checks(tmp_path):
    run = advisory_run(tmp_path)
    run.runtime.invoke("set_gripper", {"targets": {"left": "close"}})
    run.publish({"left": "close"})
    before = _object(run)
    _lift(run)
    with pytest.raises(ToolError, match="current observation"):
        run.runtime.invoke("grasp_verify_retention", {
            "arm": "left", "before_target_ref": before, "current_target_ref": before})
    run.runtime.reset_round(1)
    run.publish(reset=True)
    # A baseline from the previous round never pairs with a current target.
    with pytest.raises(ToolError, match="current round"):
        run.runtime.invoke("grasp_verify_retention", {
            "arm": "left", "before_target_ref": before, "current_target_ref": _object(run)})
    from tests.environments.robotwin.test_robotwin_placement_adapter import ABSOLUTE
    with pytest.raises(ToolError):
        run.runtime.invoke("move_pose", {"targets": {"left": {
            "position_m": [100, 0, 1], "orientation_xyzw": [0, 0, 0, 1], **ABSOLUTE}}})
    with pytest.raises(ToolError):
        run.runtime.invoke("move_pose", {"plan_ref": "unknown-plan"})


def test_absolute_and_delta_contact_targets_have_identical_execution(tmp_path):
    from tests.environments.robotwin.test_robotwin_placement_adapter import POSES
    run = advisory_run(tmp_path)
    runtime, pose, submitted = run.runtime, POSES["left"], []
    runtime._persist_action = lambda _p, _c, value: submitted.append(value)
    absolute = {"targets": {"left": {"position_mode": "absolute", "position_m": [*pose[:2], pose[2] + .01],
                                     "orientation_mode": "absolute_quaternion", "orientation_xyzw": pose[3:],
                                     "interaction": "contact_approach"}}}
    delta = {"targets": {"left": {"position_mode": "delta", "translation_m": [0.0, 0.0, .01],
                                  "position_frame": "world", "orientation_mode": "hold_current",
                                  "interaction": "contact_approach"}}}
    schema = next(d["inputSchema"] for d in runtime.surface.definitions if d["name"] == "move_pose")
    validate_schema_instance(absolute, schema)
    validate_schema_instance(delta, schema)
    runtime.invoke("move_pose", absolute)
    runtime.invoke("move_pose", delta)
    assert submitted[0]["arguments"] == submitted[1]["arguments"]


def test_dual_close_is_arm_local_and_still_consumes_observation_once(tmp_path):
    run = advisory_run(tmp_path)
    with pytest.raises(ToolError):
        run.runtime.invoke("set_gripper", {"targets": {}})
    run.runtime.invoke("set_gripper", {"targets": {"left": "close"}})
    with pytest.raises(ToolError):
        run.runtime.invoke("set_gripper", {"targets": {"right": "close"}})
    run.publish({"left": "close"})
    run.runtime.invoke("set_gripper", {"targets": {"left": "close"}})


def test_libero_absolute_and_delta_contact_targets_have_identical_execution(tmp_path, monkeypatch):
    from tests.support.holding_fixtures import PublicEpisode
    ep = PublicEpisode(tmp_path, monkeypatch, profile=LIBERO)
    pose = ep.pose
    absolute = {"position_mode": "absolute", "position_m": [*pose[:2], pose[2] + .01],
                "orientation_mode": "absolute_quaternion", "orientation_xyzw": pose[3:],
                "interaction": "contact_approach"}
    delta = {"position_mode": "delta", "translation_m": [0.0, 0.0, .01], "position_frame": "world",
             "orientation_mode": "hold_current", "interaction": "contact_approach"}
    schema = next(d["inputSchema"] for d in ep.runtime.surface.definitions if d["name"] == "move_pose")
    validate_schema_instance(absolute, schema)
    validate_schema_instance(delta, schema)
    ep.runtime.invoke("move_pose", absolute)
    ep.runtime.invoke("move_pose", delta)
    assert ep.submitted[0]["arguments"] == ep.submitted[1]["arguments"]
