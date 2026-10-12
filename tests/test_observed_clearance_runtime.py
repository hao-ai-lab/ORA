"""observed_clearance_reference_v1 on the live RoboDojo surface: inputs, binding and delivery.

The measurement itself is ``test_observed_clearance.py``. Here the Tool
Runtime gathers its inputs from what the policy already receives -- the
observation's public depth and calibration (hash-checked), measured state of
both arms, the bound public geometry descriptor, and the policy's own
localization of the target it named when the jaws closed -- and every
``move_pose``, ``move_to``, ``rotate_wrist`` and ``rotate_pitch`` target carries
the read-only ``observed_clearance`` its declared result schema admits. The
move is submitted unchanged and a failure of the reference never blocks it.
Offline: a synthetic top-down depth scene and the fake Dojo plant.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ora.agent.composition import resolve_tool_surface
from ora.agent.composition.observed_clearance_schema import TOOLS
from ora.agent.tools import observed_clearance as oc
from ora.agent.tools import observed_clearance_runtime as rt

DOJO = "robodojo_placement_closed_skills_v2"
ARX = Path(__file__).resolve().parents[1] / "ora/policy/embodiments/arx-x5"
#: The descriptor the live Dojo surface binds (v5 from the EXP11A g013 kernel unit).
DESCRIPTOR = ARX / "descriptor-public-geometry-v5.json"
#: link6 approach (+x) straight down; jaw line along world y.
DOWN = [0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)]
TABLE_Z, RIM_TOP = 0.75, 0.842
#: A basket wall 5 mm thick beside the jaws' x span.
WALL = ((0.046, -0.20, TABLE_Z), (0.051, 0.20, RIM_TOP))


def _render(camera_to_world, intrinsic, size, boxes):
    from test_observed_clearance import _render as render

    return render(camera_to_world, intrinsic, size, boxes)


def _write_scene(root: Path, *, left_pose, right_pose, left_close=False, left_opening=1.0) -> dict:
    """A Dojo-shaped observation of a table and one basket wall, three depth cameras."""
    from test_observed_clearance import _camera

    root.mkdir(parents=True)
    cameras = {}
    specs = {"head": ((0.0, -0.35, 1.6), (0.0, 0.25, -1.0)),
             "left_wrist": ((-0.6, 0.0, 1.5), (0.0, 0.0, -1.0)),
             "right_wrist": ((0.6, 0.0, 1.5), (0.0, 0.0, -1.0))}
    for name, (position, look) in specs.items():
        size = (400, 300)
        camera_to_world, intrinsic = _camera(position, look=look, size=size, focal=300.0)
        folder = root / name
        folder.mkdir()
        np.save(folder / "depth-m.npy", _render(camera_to_world, intrinsic, size, [WALL]), allow_pickle=False)
        (folder / "calibration.json").write_text(json.dumps({
            "intrinsic_cv": intrinsic.tolist(), "extrinsic_cv": np.linalg.inv(camera_to_world).tolist(),
            "cam2world_cv": camera_to_world.tolist()}))
        (folder / "rgb.png").write_bytes(b"")
        cameras[name] = {"files": {
            role: {"path": f"{name}/{file}", "sha256": hashlib.sha256((folder / file).read_bytes()).hexdigest()}
            for role, file in (("depth_numpy_m_f32", "depth-m.npy"), ("calibration", "calibration.json"),
                               ("rgb_lossless_png", "rgb.png"))}}
    def effector(pose, close, opening):
        finger = 0.044 * opening
        return {"pose_xyz_xyzw": list(pose), "gripper_finger_qpos": [finger, finger],
                "gripper_command_target": "close" if close else "open", "gripper": opening,
                "arm_joint_qpos": [0.0] * 6,
                "gripper_feedback": {"arrival_check": {"tolerance_normalized": 0.03}}}
    observation = {
        "observation_id": "robodojo:scene:round-0:observation-0007", "cameras": cameras,
        "proprioception": {"end_effectors": {
            "left": effector(left_pose, left_close, left_opening),
            "right": effector(right_pose, False, 1.0)}}}
    (root / "observation.json").write_text(json.dumps(observation))
    return observation


def _geometry():
    return rt.load_jaw_geometry(DESCRIPTOR)


HOME_RIGHT = [0.6, -0.3, 1.1, *DOWN]


def test_the_live_surface_binds_descriptor_v5():
    binding = resolve_tool_surface(DOJO).view_config["public_geometry_descriptor"]
    assert (Path(__file__).resolve().parents[1] / binding["path"]).resolve() == DESCRIPTOR.resolve()


def test_the_descriptor_supplies_both_jaws_and_their_wrist_cameras():
    geometry = _geometry()
    assert set(geometry) == {"left", "right"}
    left = geometry["left"]
    assert [finger.opening_direction for finger in left.fingers] == [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]
    assert left.empty_close_reference == pytest.approx(0.185185185)
    assert left.contact_center_local == pytest.approx((0.1501, 0.0, 0.0))
    assert left.wrist_cameras == ("left_wrist",)


def test_a_target_below_the_observed_rim_reports_it_from_the_observation(tmp_path):
    observation = _write_scene(tmp_path / "obs", left_pose=[-0.3, -0.3, 1.1, *DOWN], right_pose=HOME_RIGHT)
    target = [0.0, 0.0, 0.80 + 0.15757, *DOWN]
    result = rt.annotate_observation(observation, tmp_path / "obs", arm="left", target_pose=target,
                                     geometry=_geometry(), held={})
    assert result["status"] == "measured"
    assert result["clearance_m"] == pytest.approx(0.80 - RIM_TOP, abs=0.005)
    assert result["highest_surface"]["observation_id"] == observation["observation_id"]
    assert result["held_object"] is None


def test_a_drifted_depth_file_is_not_read(tmp_path):
    observation = _write_scene(tmp_path / "obs", left_pose=[-0.3, -0.3, 1.1, *DOWN], right_pose=HOME_RIGHT)
    for name in observation["cameras"]:
        observation["cameras"][name]["files"]["depth_numpy_m_f32"]["sha256"] = "0" * 64
    result = rt.annotate_observation(observation, tmp_path / "obs", arm="left",
                                     target_pose=[0.0, 0.0, 1.0, *DOWN], geometry=_geometry(), held={})
    assert (result["status"], result["reason"]) == ("unknown", "depth_unavailable")


def test_held_extent_follows_the_jaws_only_while_they_hold(tmp_path):
    # An object 8 cm tall localized under jaws that closed at 1.0 m.
    attach = [0.0, 0.0, 1.0, *DOWN]
    localization = {"status": "localized", "aabb_min_xyz_m": [-0.02, -0.02, 0.80],
                    "aabb_max_xyz_m": [0.02, 0.02, 0.88], "support_z_m": 0.79}
    corners, recentered = rt.held_local_corners(localization, attach, (0.1501, 0.0, 0.0))
    assert recentered is False
    world = corners @ oc.rotation(DOWN).T + np.array(attach[:3])
    assert world[:, 2].min() == pytest.approx(0.79)  # bottom lowered to the reported support
    target = [0.2, 0.0, 1.2, *DOWN]
    held = {"left": rt.HeldExtent(target_ref="target-abc", local_corners=corners)}
    holding = _write_scene(tmp_path / "holding", left_pose=attach, right_pose=HOME_RIGHT, left_close=True,
                           left_opening=0.5)
    result = rt.annotate_observation(holding, tmp_path / "holding", arm="left", target_pose=target,
                                     geometry=_geometry(), held=held)
    assert result["footprint_lowest_part"] == "held_object"
    assert result["footprint_lowest_m"][2] == pytest.approx(0.99, abs=1e-3)
    assert result["held_object"] == {"target_ref": "target-abc", "extent_included": True,
                                     "recentered_under_jaws": False}
    empty_close = _write_scene(tmp_path / "empty", left_pose=attach, right_pose=HOME_RIGHT, left_close=True,
                               left_opening=0.19)
    result = rt.annotate_observation(empty_close, tmp_path / "empty", arm="left", target_pose=target,
                                     geometry=_geometry(), held=held)
    assert result["footprint_lowest_part"] == "fingers" and result["held_object"] is None


def test_a_localization_away_from_the_jaws_is_moved_under_them_and_says_so():
    attach = [0.0, 0.0, 1.0, *DOWN]
    far = {"status": "localized", "aabb_min_xyz_m": [0.30, 0.30, 0.80], "aabb_max_xyz_m": [0.34, 0.34, 0.88]}
    corners, recentered = rt.held_local_corners(far, attach, (0.1501, 0.0, 0.0))
    world = corners @ oc.rotation(DOWN).T + np.array(attach[:3])
    assert recentered is True and world[:, :2].mean(axis=0) == pytest.approx([0.0, 0.0], abs=1e-9)
    assert rt.held_local_corners({"status": "insufficient_depth", "aabb_min_xyz_m": None,
                                  "aabb_max_xyz_m": None}, attach, (0.1501, 0.0, 0.0)) is None


# --------------------------------------------------------------------------
# The Tool Runtime on the live Dojo surface
# --------------------------------------------------------------------------

def _dojo_runtime(tmp_path, *, declared=True, left_close=False, left_opening=1.0):
    from test_move_pose_reach import _dojo_runtime as runtime_for

    runtime = runtime_for(tmp_path)
    if not declared:
        runtime.surface = replace(runtime.surface, capabilities=tuple(
            item for item in runtime.surface.capabilities if item != rt.V2_CAPABILITY))
    source = tmp_path / "scene"
    observation = _write_scene(source, left_pose=[-0.3, -0.3, 1.1, *DOWN], right_pose=HOME_RIGHT,
                               left_close=left_close, left_opening=left_opening)
    pointer, _source, released = runtime._current()
    observation = {**copy.deepcopy(released), **observation,
                   "proprioception": {**released["proprioception"], **observation["proprioception"]}}
    pointer = {**pointer, "observation_sha256": hashlib.sha256(
        json.dumps(observation, sort_keys=True).encode()).hexdigest()}
    runtime._current = lambda: (pointer, source, observation)
    runtime.align_plan_callback = None
    return runtime


def _consume(runtime):
    """The environment takes the submitted action; the same observation stays current here."""
    (runtime.control_root / "submitted-action.json").unlink()


def _move_pose(runtime, position, orientation=DOWN):
    return runtime.invoke("move_pose", {"targets": {"left": {
        "interaction": "free_space", "position_mode": "absolute", "position_m": list(position),
        "orientation_mode": "absolute_quaternion", "orientation_xyzw": list(orientation)}}})


def test_move_pose_carries_the_observed_clearance_its_schema_admits(tmp_path):
    runtime = _dojo_runtime(tmp_path)
    result = _move_pose(runtime, [0.0, 0.0, 0.80 + 0.15757])
    clearance = result["compiled_pose"]["targets"]["left"]["observed_clearance"]
    assert clearance["status"] == "measured"
    assert clearance["clearance_m"] == pytest.approx(0.80 - RIM_TOP, abs=0.005)
    submitted = json.loads((runtime.control_root / "submitted-action.json").read_text())
    assert submitted["arguments"]["targets"]["left"]["position_m"] == pytest.approx([0.0, 0.0, 0.95757])


@pytest.mark.parametrize(("tool", "arguments"), [
    ("move_to", {"targets": {"left": {"position_m": [0.0, 0.0, 0.95757]}}}),
    ("rotate_wrist", {"targets": {"left": 20.0}}),
    ("rotate_pitch", {"targets": {"left": -10.0}}),
])
def test_every_pose_primitive_target_carries_it(tmp_path, tool, arguments):
    runtime = _dojo_runtime(tmp_path)
    result = runtime.invoke(tool, arguments)
    clearance = result["compiled_pose"]["targets"]["left"]["observed_clearance"]
    assert clearance["status"] in {"measured", "unknown"}
    assert clearance["claim_boundary"] == oc.CLAIM_BOUNDARY


def test_set_gripper_close_with_a_target_ref_names_the_held_extent_until_open(tmp_path, monkeypatch):
    runtime = _dojo_runtime(tmp_path, left_close=True, left_opening=0.5)
    localization = {"status": "localized", "aabb_min_xyz_m": [-0.32, -0.32, 0.86],
                    "aabb_max_xyz_m": [-0.28, -0.28, 0.94], "support_z_m": 0.86}
    from ora.agent.tools import grasp

    monkeypatch.setattr(grasp, "load_segmented_target", lambda **_kwargs: (
        {"localization": localization}, None, None))
    runtime.invoke("set_gripper", {"targets": {"left": "close"}, "target_ref": "target-0123456789abcdef01234567"})
    _consume(runtime)
    result = _move_pose(runtime, [0.0, 0.0, 1.2])
    clearance = result["compiled_pose"]["targets"]["left"]["observed_clearance"]
    assert clearance["held_object"] == {"target_ref": "target-0123456789abcdef01234567",
                                        "extent_included": True, "recentered_under_jaws": False}
    assert clearance["footprint_lowest_part"] == "held_object"
    _consume(runtime)
    runtime.invoke("set_gripper", {"targets": {"left": "open"}})
    assert "left" not in getattr(runtime, "_observed_clearance_holds")


def test_grasp_execute_records_its_committed_target_and_contact_pose(tmp_path):
    from types import SimpleNamespace

    host = SimpleNamespace(surface=resolve_tool_surface(DOJO), control_root=tmp_path)
    contact = {"position_m": [0.1, -0.05, 0.93], "orientation_xyzw": DOWN}
    (tmp_path / "manipulation-skill-commitment.json").write_text(json.dumps({
        "skill": "grasp_execute", "arm": "right", "source_ref": "target-0123456789abcdef01234567",
        "poses": {"approach": contact, "lift": contact}}))
    rt.note_grasp_execute(host)
    assert host._observed_clearance_holds["right"] == {
        "target_ref": "target-0123456789abcdef01234567", "source": "grasp_execute",
        "attach_pose": [0.1, -0.05, 0.93, *DOWN]}
    (tmp_path / "manipulation-skill-commitment.json").write_text("{not json")
    rt.note_grasp_execute(host)  # a record it cannot read never raises
    rt.note_gripper(host, {"right": "close"}, target_ref="target-x", observation={})
    assert "right" not in host._observed_clearance_holds


def test_jaws_holding_without_a_named_target_report_the_extent_unavailable(tmp_path):
    runtime = _dojo_runtime(tmp_path, left_close=True, left_opening=0.5)
    clearance = _move_pose(runtime, [0.0, 0.0, 1.2])["compiled_pose"]["targets"]["left"]["observed_clearance"]
    assert clearance["held_object"] == {"target_ref": None, "extent_included": False,
                                        "recentered_under_jaws": False}


def test_a_failing_reference_never_blocks_the_move(tmp_path, monkeypatch):
    runtime = _dojo_runtime(tmp_path)

    def broken(*_args, **_kwargs):
        raise RuntimeError("depth reader died")

    monkeypatch.setattr(rt, "annotate_observation", broken)
    result = _move_pose(runtime, [0.0, 0.0, 1.0])
    assert result["status"] == "submitted"
    assert result["compiled_pose"]["targets"]["left"]["observed_clearance"] == {
        "status": "unknown", "reason": "reference_failed", "radius_m": oc.RADIUS_M,
        "claim_boundary": oc.CLAIM_BOUNDARY}


def test_a_surface_without_the_capability_submits_as_released(tmp_path):
    runtime = _dojo_runtime(tmp_path, declared=False)
    result = _move_pose(runtime, [0.0, 0.0, 1.0])
    assert "observed_clearance" not in result["compiled_pose"]["targets"]["left"]


def test_the_live_dojo_surface_declares_it():
    dojo = resolve_tool_surface(DOJO)
    assert dojo.supports(rt.V2_CAPABILITY)
    for name in TOOLS:
        targets = dojo.result_schemas[name]["properties"]["compiled_pose"]["properties"]["targets"]
        assert "observed_clearance" in targets["additionalProperties"]["properties"]
    help_text = next(item["description"] for item in dojo.definitions if item["name"] == "move_pose")
    assert "observed_clearance" in help_text
