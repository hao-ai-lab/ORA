"""Visible borders, joined camera pairs and finite evidence-backed recovery."""

import copy
from types import SimpleNamespace

import pytest

from ora.agent.composition import resolve_tool_surface, validate_schema_instance
from ora.agent.tools import retention, retention_recovery as recovery
from ora.agent.tools.robot import ToolError
from tests.agent.perception.test_retention_correctness import POSE, cloud, geometry, timeline


def comparison(*, stationary=False, extra_stationary=None):
    states = timeline([*POSE[:2], POSE[2] + .025, *POSE[3:]])
    points = cloud()
    targets = {
        "a0": (geometry(1, points), states[1]),
        "a1": (geometry(2, points + [0, 0, 0 if stationary else .025]), states[2]),
    }
    for item in targets.values():
        item[0]["source_quality"]["touches_image_boundary"] = True
    pairs = []
    if extra_stationary is not None:
        targets.update(w0=(geometry(1, points), states[1]),
                       w1=(geometry(2, points + [0, 0, 0 if extra_stationary else .025]), states[2]))
        for key in ("w0", "w1"):
            targets[key][0]["source_camera"] = "wrist"
        pairs.append({"before_target_ref": "w0", "current_target_ref": "w1"})
    return states, targets, pairs


#: The kept surface that ships grasp_verify_retention, as an advisory (no motion permit).
SURFACE = "robotwin_toolchain_b2_closed_noservo_v1"


def advisory(result):
    """What the runtime delivers on an advisory surface (``runtime/holding.py``)."""
    result = copy.deepcopy(result)
    result.update(schema=retention.ADVISORY_SCHEMA, assessment_role="advisory_only")
    result.pop("recovery", None)
    return result


def compute(states, targets, pairs):
    return recovery.compare_views(arm="right", before_ref="a0", current_ref="a1",
        additional_pairs=pairs, load_target=targets.__getitem__, load_step=states.__getitem__,
        current_state=states[2])


@pytest.mark.parametrize("stationary,status", [(False, "consistent"), (True, "inconsistent")])
def test_border_surfaces_are_measured_without_relaxing_motion_thresholds(stationary, status):
    states, targets, pairs = comparison(stationary=stationary)
    result = compute(states, targets, pairs)
    assert result["status"] == status
    assert result["surface_comparison"]["rigid_p90_error_m"] is not None
    assert result["quality_flags"] == ["before_image_border", "current_image_border"]
    assert result["thresholds"]["minimum_rigid_overlap_fraction"] == .9


def changed_view_comparison():
    """A partly-followed target whose one camera moved past the negative bound."""
    states = timeline([*POSE[:2], POSE[2] + .05, *POSE[3:]])
    points = cloud()
    targets = {"a0": (geometry(1, points), states[1]),
               "a1": (geometry(2, points + [0, 0, .04]), states[2])}
    for item in targets.values():
        item[0]["camera_origin_world_m"] = [.30, 0, .96]
    return states, targets, []


def test_suppressed_negative_view_change_recovery_names_the_pair_to_re_observe():
    """The recovery record and the reason must name one executable step.

    The suppression itself is correct and stays: a changed view of the visible
    surfaces is not evidence against retention. What the record adds is the
    camera and the baseline step whose viewpoint makes the pair comparable.
    """
    result = compute(*changed_view_comparison())
    assert result["status"] == "unknown"
    assert result["evidence_class"] == "visibility_insufficient"
    assert result["recovery"]["next_step"] == (
        "inspect_same_camera_baseline_and_current_or_use_remaining_local_probe")
    remedy = result["reason"].split(";", 1)[1]
    assert result["source_cameras"]["before"] in remedy
    assert f"step_{result['recovery']['baseline_step']}_" in remedy
    delivered = advisory(result)
    surface = resolve_tool_surface(SURFACE)
    validate_schema_instance(delivered, surface.result_schemas["grasp_verify_retention"])
    # The remedy is worthless unless it survives the model projection, together
    # with the measured angle and the threshold it failed. The advisory surface
    # drops the permit-era recovery record, so the reason carries the remedy.
    from ora.agent.native.session import NativePlanner
    planner = object.__new__(NativePlanner)
    planner.tools = SimpleNamespace(surface=surface)
    visible = planner._causal_tool_result("grasp_verify_retention", delivered)
    assert visible["reason"] == result["reason"]
    assert remedy in visible["reason"]
    assert (visible["surface_comparison"]["view_direction_change_rad"]
            > visible["thresholds"]["maximum_negative_view_direction_change_rad"] > 0)
    assert visible["source_cameras"]["before"] == result["source_cameras"]["before"]


def test_heavy_occlusion_does_not_pass_from_a_small_matching_fragment():
    states, targets, pairs = comparison()
    targets["a1"][0]["voxel_points_world_m"] = targets["a1"][0]["voxel_points_world_m"][:16]
    assert compute(states, targets, pairs)["status"] == "unknown"


def test_near_threshold_fit_failure_reports_failed_acceptance_not_model_separation(monkeypatch):
    states, targets, pairs = comparison()
    # Source-bound Task 9 shape: stationary and rigid models ARE separated,
    # but the visible surfaces miss the fit/overlap acceptance criteria.
    values = iter([(0.0047416, .90625, .85742), (.031602, .1, .1), (.003, .9, .9)])
    monkeypatch.setattr(recovery.retention, "_surface_distance", lambda *_a, **_k: next(values))
    result = compute(states, targets, pairs)
    assert result["status"] == "unknown"
    assert result["reason"] == "rigid_surface_fit_and_overlap_fail_acceptance"
    assert result["surface_comparison"]["model_separation_m"] > .02


def test_camera_conflict_is_not_hidden_by_selecting_a_positive():
    result = compute(*comparison(extra_stationary=True))
    assert result["status"] == "unknown"
    assert result["evidence_class"] == "conflicting_views"
    assert [pair["status"] for pair in result["comparisons"]] == ["consistent", "inconsistent"]


def test_complementary_pair_preserves_witness_and_original_anchor():
    states, targets, pairs = comparison(extra_stationary=False)
    targets["a1"][0]["source_camera"] = "different-camera"
    result = compute(states, targets, pairs)
    assert result["status"] == "consistent"
    assert result["anchor_before_target_ref"] == "a0"
    assert result["before_target_ref"] == "w0" and result["current_target_ref"] == "w1"
    assert result["source_cameras"] == {"before": "wrist", "current": "wrist"}
    targets["w0"] = (targets["w0"][0], {**states[1], "state_sha256": "wrong-state"})
    with pytest.raises(ToolError, match="same baseline"):
        compute(states, targets, pairs)


def test_historical_camera_selection_is_read_only_and_round_bound(tmp_path):
    from tests.environments.robotwin.test_robotwin_placement_adapter import RoboTwinPublicRun
    run = RoboTwinPublicRun(tmp_path, bind_submitted=True)
    tools = run.runtime
    tools.invoke("set_gripper", {"targets": {"right": "open"}})
    run.publish({"right": "open"})
    current = copy.deepcopy(tools._current_grip_state()[0])
    result = tools.invoke("observe", {"step": 0, "views": ["left_camera"]})
    assert result["status"] == "actual_historical_observation"
    assert result["observation_step"] == 0
    from tests.agent.perception.test_batch_segmentation import BatchBackend
    tools.segmentation_backend = BatchBackend()  # the kept surface segments by point batch
    target = tools.invoke("vision_segment", {"step": 0, "camera": "left_camera", "points_uv": [[48, 48]]})
    geometry, state = tools._target_geometry_and_state(target_ref=target["candidates"][0]["target_ref"])
    assert state["step"] == 0
    assert tools._current_grip_state()[0] == current
    from ora.agent.native.session import NativePlanner
    planner = object.__new__(NativePlanner)
    planner.tools = SimpleNamespace(surface=tools.surface)
    visible = planner._causal_tool_result("observe", result)
    assert visible["status"] == "actual_historical_observation"
    assert visible["observation_step"] == 0
    assert visible["views"] == ["left_camera"]
    tools.reset_round(1)
    run.publish(reset=True)
    with pytest.raises(ToolError, match="current round"):
        tools.invoke("observe", {"step": 0, "views": ["left_camera"]})


def test_recovery_definitions_advertise_real_camera_pair_workflow():
    surface = resolve_tool_surface(SURFACE)
    tool = next(item for item in surface.definitions if item["name"] == "grasp_verify_retention")
    assert "additional_pairs" in tool["inputSchema"]["properties"]
    assert "observe(step=...) and vision_segment(step=...)" in tool["description"]
    result = compute(*comparison())
    validate_schema_instance(advisory(result), surface.result_schemas["grasp_verify_retention"])
