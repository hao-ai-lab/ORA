"""Public surface motion cannot become an attachment or semantic oracle."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ora.agent.composition import resolve_tool_surface, validate_schema_instance
from ora.agent.tools import retention
from ora.agent.tools.runtime import ManipulationToolRuntime
from ora.agent.tools.robot import ToolError


POSE = [0.2, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
#: The kept surface that serves retention: every camera pair it compares is one
#: full-reference (v4) verification, published as a ``comparisons`` entry.
RETENTION_SURFACE = "robotwin_toolchain_b2_closed_noservo_v1"


def _validate(result: dict) -> None:
    """Admit one verification as the kept surface publishes it per camera pair."""
    from ora.agent.tools import retention_recovery

    schema = resolve_tool_surface(RETENTION_SURFACE).result_schemas["grasp_verify_retention"]
    item = {**result, "quality_flags": [], "evidence_class": retention_recovery.evidence_class(result)}
    validate_schema_instance(item, schema["properties"]["comparisons"]["items"])


def public_state(step: int, pose=None, *, previous=None, action=None, receipt=None) -> dict:
    return {
        "step": step, "state_sha256": f"{1000 + step:064x}",
        "previous_state_sha256": None if previous is None else previous["state_sha256"],
        "cause": "reset" if previous is None else "after_action",
        "observation": {"id": f"obs-{step}", "manifest": {"sha256": f"{step:064x}"}},
        "proprioception": {"end_effectors": {"right": {
            "pose_xyz_xyzw": list(POSE if pose is None else pose), "gripper_opening_m": 0.02,
        }}},
        "cameras": {"agentview": {"files": {name: {"sha256": value * 64} for name, value in (
            ("rgb_lossless_png", "a"), ("depth_numpy_m_f32", "b"), ("calibration", "c"))}}},
        "previous_public_action": action, "previous_execution_receipt": receipt,
    }


def public_action(source: dict, name: str, pose=None) -> dict:
    return {
        "schema": "ora-motion-action.v3", "action_id": f"action-{source['step']}",
        "observation_id": source["observation"]["id"],
        "bound_observation_sha256": source["observation"]["manifest"]["sha256"],
        "action": "set_gripper" if name in {"open", "close"} else "move_pose",
        "arguments": ({"targets": {"right": name}} if name in {"open", "close"} else {
            "targets": {"right": {"position_m": pose[:3], "orientation_xyzw": pose[3:]}},
            "interactions": {"right": "precision"},
        }),
    }


def receipt(action: dict, *, steps=4, changed=True, status="reached") -> dict:
    return {"action_id": action["action_id"], "consumed": True,
            "execution_status": status, "state_changed": changed, "steps_used": steps}


def cloud() -> np.ndarray:
    rng = np.random.default_rng(841)
    points = rng.normal(size=(240, 3))
    points /= np.linalg.norm(points, axis=1)[:, None]
    return points * [0.012, 0.022, 0.017] + [0.24, 0, 0.96]


def geometry(step: int, points: np.ndarray) -> dict:
    return {"observation_id": f"obs-{step}", "source_camera": "agentview",
            "centroid_world_m": np.mean(points, axis=0).tolist(),
            "voxel_points_world_m": points.tolist(), "voxel_size_m": 0.003,
            "source_quality": {"depth_valid_fraction": 1.0, "depth_inlier_fraction": 1.0,
                               "primary_component_fraction": 1.0, "touches_image_boundary": False}}


def timeline(pose: list) -> dict[int, dict]:
    zero = public_state(0)
    close = public_action(zero, "close")
    one = public_state(1, previous=zero, action=close, receipt=receipt(close))
    move = public_action(one, "move_pose", pose)
    two = public_state(2, pose, previous=one, action=move, receipt=receipt(move))
    return {0: zero, 1: one, 2: two}


def verify(states: dict, before: dict, current: dict) -> dict:
    result = retention.verify_retention(
        arm="right", before_ref="target-1", current_ref="target-2",
        before_geometry=before, current_geometry=current,
        before_state=states[1], current_state=states[2], load_step=states.__getitem__,
        full_reference=True,
    )
    _validate(result)
    return result


@pytest.mark.parametrize("delta,object_delta,status", [
    (0.025, 0.025, "consistent"), (0.016, -0.008, "inconsistent"),
    (0.025, 0.0, "inconsistent"), (0.006, 0.006, "unknown"),
])
def test_rigid_stationary_opposed_and_low_excitation(delta, object_delta, status) -> None:
    pose = [*POSE[:2], POSE[2] + delta, *POSE[3:]]
    points = cloud()
    result = verify(timeline(pose), geometry(1, points), geometry(2, points + [0, 0, object_delta]))
    assert result["status"] == status
    assert result["baseline"]["closed_interval_verified"]
    assert result["baseline"]["close_action_id"] == "action-0"
    assert result["surface_comparison"]["centroid_is_material_point"] is False


@pytest.mark.parametrize("object_delta, expected_status", [
    (0.025, "consistent"), (0.0, "inconsistent"), (-0.015, "inconsistent"),
])
def test_full_measured_reference_removes_independent_sampling_holes(
    object_delta, expected_status,
) -> None:
    rng = np.random.default_rng(157)
    points = rng.normal(size=(1600, 3))
    points /= np.linalg.norm(points, axis=1)[:, None]
    points = points * [.06, .09, .08] + [.24, 0, .96]
    # Identical measured material points in a different raster/voxel order:
    # thinning both reference sets must not manufacture a surface discrepancy.
    current = points[rng.permutation(len(points))] + [0, 0, object_delta]
    states = timeline([*POSE[:2], POSE[2] + .025, *POSE[3:]])
    kwargs = dict(arm="right", before_ref="target-1", current_ref="target-2",
                  before_geometry=geometry(1, points), current_geometry=geometry(2, current),
                  before_state=states[1], current_state=states[2], load_step=states.__getitem__)
    result = retention.verify_retention(**kwargs, full_reference=True)
    _validate(result)
    assert result["status"] == expected_status
    assert result["surface_comparison"]["before_point_count"] == 1600
    assert result["surface_comparison"]["before_query_count"] == 512
    assert result["thresholds"]["surface_noise_radius_m"] == pytest.approx(.0045)
    if object_delta == .025:
        assert result["surface_comparison"]["rigid_p90_error_m"] < 1e-12
        legacy = retention.verify_retention(**kwargs)
        assert legacy["schema"] == retention.SCHEMA
        assert legacy["status"] == "unknown"


def test_full_reference_does_not_complete_a_missing_visible_surface() -> None:
    points = cloud()
    current = points[points[:, 0] > np.median(points[:, 0])] + [0, 0, .025]
    states = timeline([*POSE[:2], POSE[2] + .025, *POSE[3:]])
    result = retention.verify_retention(
        arm="right", before_ref="target-1", current_ref="target-2",
        before_geometry=geometry(1, points), current_geometry=geometry(2, current),
        before_state=states[1], current_state=states[2], load_step=states.__getitem__,
        full_reference=True,
    )
    assert result["status"] == "unknown"
    assert result["surface_comparison"]["rigid_overlap_before"] < .9


def test_pure_rotation_of_offset_asymmetric_target_can_support_rigid_motion() -> None:
    pose = [*POSE[:3], 0, 0, 1, 0]
    points = cloud()
    rotated = (points - POSE[:3]) @ np.diag([-1, -1, 1]) + POSE[:3]
    result = verify(timeline(pose), geometry(1, points), geometry(2, rotated))
    assert result["status"] == "consistent"
    assert result["geometry"]["expected_target_motion_m"] > 0.07


def test_changed_visible_hemisphere_during_rotation_is_unknown() -> None:
    pose = [*POSE[:3], 0, 0, 1, 0]
    points = cloud()
    # The camera sees a different part of a rigid object; a centroid is not a material point.
    rotated = (points - POSE[:3]) @ np.diag([-1, -1, 1]) + POSE[:3]
    result = verify(timeline(pose), geometry(1, points[points[:, 0] > 0.235]),
                    geometry(2, rotated[rotated[:, 0] > 0.155]))
    assert result["status"] == "unknown"


@pytest.mark.parametrize("damage", ["camera", "boundary", "depth", "sparse", "receipt", "open", "reset", "gap"])
def test_incomparable_or_unjoined_evidence_never_authorizes_transport(damage) -> None:
    points = cloud()
    states = timeline([*POSE[:2], 1.025, *POSE[3:]])
    before, current = geometry(1, points), geometry(2, points + [0, 0, 0.025])
    if damage == "camera":
        current["source_camera"] = "wrist"
    elif damage == "boundary":
        current["source_quality"]["touches_image_boundary"] = True
    elif damage == "depth":
        current["source_quality"]["depth_valid_fraction"] = 0.4
    elif damage == "sparse":
        current["voxel_points_world_m"] = points[:4].tolist()
    elif damage == "receipt":
        states[2]["previous_execution_receipt"]["action_id"] = "another"
    elif damage == "open":
        states[2]["previous_public_action"]["action"] = "set_gripper"
        states[2]["previous_public_action"]["arguments"] = {"targets": {"right": "open"}}
    elif damage == "reset":
        states[2]["cause"] = "reset"
    else:
        states[2]["previous_state_sha256"] = "f" * 64
    result = verify(states, before, current)
    assert result["status"] == "unknown"
    assert result["comparison_eligible"] is False


def test_symmetric_rotation_without_target_displacement_is_unknown() -> None:
    points = cloud() - [0.04, 0, -0.04]
    # Pin centroid exactly to TCP, making rotation unobservable for this hypothesis.
    points += np.asarray(POSE[:3]) - np.mean(points, axis=0)
    result = verify(timeline([*POSE[:3], 0, 0, 1, 0]), geometry(1, points), geometry(2, points))
    assert result["status"] == "unknown"


def _timeline_with_second_gripper_command(pose: list, directive: str) -> dict[int, dict]:
    """Close at step 1, issue `directive` at step 2, then move at step 3."""
    zero = public_state(0)
    close = public_action(zero, "close")
    one = public_state(1, previous=zero, action=close, receipt=receipt(close))
    again = public_action(one, directive)
    two = public_state(2, previous=one, action=again, receipt=receipt(again))
    move = public_action(two, "move_pose", pose)
    three = public_state(3, pose, previous=two, action=move, receipt=receipt(move))
    return {0: zero, 1: one, 2: two, 3: three}


def _verify_over(states: dict, before: dict, current: dict, *, current_step: int) -> dict:
    result = retention.verify_retention(
        arm="right", before_ref="target-1", current_ref="target-2",
        before_geometry=before, current_geometry=current,
        before_state=states[1], current_state=states[current_step],
        load_step=states.__getitem__, full_reference=True,
    )
    _validate(result)
    return result


def test_redundant_close_keeps_the_post_close_interval_alive() -> None:
    """A second close on an already-closed gripper is not a state change.

    The operator-visible symptom this covers: two arms each holding a bottle
    reported `unknown` with `gripper_command_changed_since_baseline` purely
    because the policy had re-issued `close` inside the compared interval.
    """
    pose = [*POSE[:2], POSE[2] + 0.025, *POSE[3:]]
    points = cloud()
    states = _timeline_with_second_gripper_command(pose, "close")
    result = _verify_over(
        states, geometry(1, points), geometry(3, points + [0, 0, 0.025]), current_step=3,
    )
    assert result["baseline"]["closed_interval_verified"] is True
    assert result["baseline"]["close_action_id"] == "action-0"
    assert result["comparison_eligible"] is True
    assert result["status"] == "consistent"


def test_open_inside_the_interval_still_ends_it() -> None:
    pose = [*POSE[:2], POSE[2] + 0.025, *POSE[3:]]
    points = cloud()
    states = _timeline_with_second_gripper_command(pose, "open")
    result = _verify_over(
        states, geometry(1, points), geometry(3, points + [0, 0, 0.025]), current_step=3,
    )
    assert result["baseline"]["closed_interval_verified"] is False
    assert result["reason"] == "gripper_opened_since_baseline"
    assert result["comparison_eligible"] is False
    assert result["status"] == "unknown"


@pytest.mark.parametrize("wrapped", [False, True])
def test_real_retention_dispatch_accepts_direct_and_hash_bound_wrapped_receipts(tmp_path: Path, monkeypatch, wrapped) -> None:
    points = cloud()
    states = timeline([*POSE[:2], 1.025, *POSE[3:]])
    if wrapped:
        for step in (1, 2):
            data = states[step]["previous_execution_receipt"]
            path = tmp_path / f"receipt-{step}.json"
            path.write_text(json.dumps(data))
            states[step]["previous_execution_receipt"] = {
                "path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "receipt": data,
            }
    runtime = object.__new__(ManipulationToolRuntime)
    runtime.surface = resolve_tool_surface(RETENTION_SURFACE)
    runtime.allowed_tools = tuple(d["name"] for d in runtime.surface.definitions)
    runtime.event_path = tmp_path / "events.jsonl"
    runtime.state_root = tmp_path / "policy-state"
    runtime.artifact_root = tmp_path / "artifacts"
    runtime._current_grip_state = lambda: (
        {"observation_id": "obs-2", "observation_sha256": f"{2:064x}"}, None, None, None, None)
    evidence = {"target-1": (geometry(1, points), states[1]),
                "target-2": (geometry(2, points + [0, 0, 0.025]), states[2])}
    runtime._target_geometry_and_state = lambda *, target_ref: evidence[target_ref]
    for module in ("ora.agent.tools.runtime.core", "ora.agent.tools.saved_steps"):
        monkeypatch.setattr(f"{module}.verify_policy_state", lambda *_args, **_kw:
                            {"status": "valid", "steps": list(states.values())})
    result = runtime.invoke("grasp_verify_retention", {"arm": "right", "before_target_ref": "target-1", "current_target_ref": "target-2"})
    assert result["status"] == "consistent"
    if wrapped:
        (tmp_path / "receipt-2.json").write_text("{}")
        assert runtime.invoke("grasp_verify_retention", {
            "arm": "right", "before_target_ref": "target-1", "current_target_ref": "target-2"})["status"] == "unknown"
    evidence["target-2"][0]["observation_id"] = "stale"
    with pytest.raises(ToolError, match="current observation"):
        runtime.invoke("grasp_verify_retention", {"arm": "right", "before_target_ref": "target-1", "current_target_ref": "target-2"})


def _changed_view_pair(camera_origin_world_m: list) -> dict:
    """A partly-followed target seen from a viewpoint that moved past the bound."""
    pose = [*POSE[:2], POSE[2] + 0.05, *POSE[3:]]
    points = cloud()
    before, current = geometry(1, points), geometry(2, points + [0, 0, 0.04])
    for item in (before, current):
        item["camera_origin_world_m"] = list(camera_origin_world_m)
    return verify(timeline(pose), before, current)


def test_suppressed_negative_view_change_names_the_camera_and_baseline_step() -> None:
    """The suppression stands, but it must hand the policy a step to execute.

    Observed shape: view_change 1.0086 rad against the 0.349 threshold left the
    policy with a cause and no way to make the pair comparable again.
    """
    result = _changed_view_pair([0.30, 0.0, 0.96])
    assert result["status"] == "unknown"  # A changed view still cannot say inconsistent.
    # The schema-validated measurement and its threshold reach the policy too.
    assert (result["surface_comparison"]["view_direction_change_rad"]
            > result["thresholds"]["maximum_negative_view_direction_change_rad"] > 0)
    reason, remedy = result["reason"].split(";", 1)
    assert reason == "changed_view_of_visible_surfaces_prevents_negative_retention_claim"
    assert remedy.startswith("remedy=")
    assert result["source_cameras"]["before"] in remedy
    assert result["source_cameras"]["before"] == result["source_cameras"]["current"]
    assert f"step_{result['baseline']['before_state_step']}_" in remedy
    assert "verify_again" in remedy
