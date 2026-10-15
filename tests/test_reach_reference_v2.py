"""RoboTwin move targets carry a reach pre-check that names the IK that answered it (0059 M1, M3).

On RoboDojo ``move_pose_reach_along_request_v1`` replays the route move_pose
takes. On RoboTwin the same internal reach check is one damped-least-squares
descent with joint limits only (``robotwin_single_seed_damped_least_squares_v1``)
and ``move_pose`` admits through a 128-seed collision-checked CuRobo IK, so the
v2 generation carries the backend's own ``ik_model``,
``matches_move_pose_admission`` and claim, plus the measured disagreement
(about 5% false refusals, 4% false admissions, from the align_plan measurement
of the same solver). ``reach_nearest_orientation_v2`` sweeps with the same check
and holds the grip-centre TCP itself, the zero jaw offset of a grip-centre
frame. Offline: the real RoboTwin reach-check path (session ``align_plan`` ->
``plan_alignment`` -> ``check_reach`` -> adapter ``reach_check_batch``) over a
toy arm whose joints are its pose, and the RoboTwin default surface that
declares the ports (its move_pose help describes the endpoint-screened check,
M6; the toy adapter answers with the unscreened model).
"""
from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ora.agent.composition import reach_reference_schema as schema
from ora.agent.composition import resolve_tool_surface
from ora.agent.composition.move_pose_reach_schema import LINE_FRACTIONS
from ora.agent.composition.profile_catalog import PROFILE_CATALOG
from ora.agent.composition.reach_orientation_schema import NEAREST_PROPERTY, ROTATIONS_DEG
from ora.agent.composition.tool_surface import ToolSurfaceError, validate_schema_instance
from ora.agent.context.public_facts import PublicFactsError
from ora.agent.tools import jaw_center, move_pose_reach, reach_reference
from ora.agent.tools.robot import ToolError
from ora.contracts.reach_check import ROBOTWIN_INTER_ARM_IK_MODELS
from test_reach_check import ARRIVAL, _robotwin_adapter

PORTS = Path(__file__).resolve().parents[1] / "ora/policy/tool_surfaces/closed_skills_ports_v1"
ROBOTWIN_DEFAULT = "robotwin_toolchain_b2_closed_noservo_v2"
ROBOTWIN_MODEL = "robotwin_single_seed_damped_least_squares_v1"
SCREENED_MODEL = ROBOTWIN_INTER_ARM_IK_MODELS["endpoint"]
BOUNDS = {"x": [-1.0, 1.0], "y": [-1.0, 1.0], "z": [0.0, 2.0]}
IDENTITY = [0.0, 0.0, 0.0, 1.0]
CURRENT = [0.0, 0.0, 0.9, *IDENTITY]
#: Toy arm: joints are [x, y, z, rotvec]; z above 1.2 m or any rotation
#: component beyond 0.5 rad is out of joint range.
LIMITS = [[-1.0, 1.0], [-1.0, 1.0], [0.0, 1.2], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]]
NEW_CAPABILITIES = (schema.MOVE_POSE_CAPABILITY, schema.PRIMITIVE_CAPABILITY, schema.NEAREST_CAPABILITY)


def _quaternion(rotvec):
    from scipy.spatial.transform import Rotation

    return [float(item) for item in Rotation.from_rotvec(rotvec).as_quat()]


def _robotwin_bridge(monkeypatch, requests=None):
    """Session align_plan over the real controllers and the real adapter methods."""
    from ora.environment.backends.robotwin import session as robotwin_session

    adapter = _robotwin_adapter(LIMITS)
    monkeypatch.setattr(robotwin_session, "RoboTwinMotionBackendAdapter", lambda **kwargs: adapter)
    fake = SimpleNamespace(_require_environment=lambda: object(), control_profile=ARRIVAL,
                           _motion_backend_adapter=lambda **kwargs: adapter)

    def bridge(request):
        if requests is not None:
            requests.append(copy.deepcopy(request))
        return robotwin_session.RoboTwinLiveSession.align_plan(fake, request)
    return bridge


@pytest.fixture
def robotwin():
    return resolve_tool_surface(ROBOTWIN_DEFAULT)


def _host(surface, bridge):
    return SimpleNamespace(surface=surface, align_plan_callback=bridge, workspace_bounds_m=BOUNDS)


def _target_schema(surface, tool):
    return surface.result_schemas[tool]["properties"]["compiled_pose"]["properties"]["targets"][
        "additionalProperties"]["properties"]["reach_along_request"]


def _plan(surface, bridge, target, *, interaction="free_space"):
    return move_pose_reach.plan_target(
        _host(surface, bridge), "left", current_pose=list(CURRENT), target_pose=list(target),
        interaction=interaction, orientation_mode="absolute_quaternion",
        observation={"proprioception": {"end_effectors": {"left": {"gripper_command_target": "open"}}}})


# --------------------------------------------------------------------------
# Activation: only the v2 default generation declares it (integration 0059 unit A0)
# --------------------------------------------------------------------------

def test_only_the_robotwin_and_libero_v2_defaults_declare_the_ports():
    declaring = {profile for profile, path in PROFILE_CATALOG.items()
                 if set(json.loads(path.read_text(encoding="utf-8")).get("capabilities", [])) & set(NEW_CAPABILITIES)}
    # LIBERO joined with M2; RoboDojo keeps v1 (parity exceptions P0059-D01-D03).
    # LIBERO's overhead successor inherits the ports.
    assert declaring == {ROBOTWIN_DEFAULT, "libero_placement_direct_overhead_v1"}
    for profile in declaring:
        declared = set(json.loads(PROFILE_CATALOG[profile].read_text(encoding="utf-8"))["capabilities"])
        assert set(NEW_CAPABILITIES) <= declared, profile


def test_the_candidate_binds_the_v2_reference_on_move_pose_and_the_primitives(robotwin):
    for tool in ("move_pose", "move_to", "rotate_wrist", "rotate_pitch"):
        reach = _target_schema(robotwin, tool)
        assert reach == {**schema.REACH_REFERENCE_PROPERTY, "oneOf": reach["oneOf"]}, tool
        solved, unsolved, unavailable = reach["oneOf"]
        for variant in (solved, unsolved):
            assert {"ik_model", "matches_move_pose_admission", "measured_disagreement",
                    "claim_boundary"} <= set(variant["required"])
        assert unsolved["properties"]["nearest_reachable_orientation"] == NEAREST_PROPERTY
        assert "nearest_reachable_orientation" in unsolved["required"]
        assert unavailable["required"] == ["status", "failure"]
    slots = {item["slot"] for item in robotwin.components}
    assert {f"derived:{name}" for name in NEW_CAPABILITIES} <= slots
    # The jaw turn stays an ARX X5 capability.
    assert not robotwin.supports(move_pose_reach.JAW_CAPABILITY)
    assert not robotwin.supports(move_pose_reach.REACH_CAPABILITY)


# --------------------------------------------------------------------------
# The real RoboTwin reach check, through the v2 annotation
# --------------------------------------------------------------------------

def test_a_solved_target_names_the_approximate_check_and_its_measured_error(robotwin, monkeypatch):
    requests = []
    target = [0.1, 0.05, 0.95, *IDENTITY]
    pose, notes = _plan(robotwin, _robotwin_bridge(monkeypatch, requests), target)
    assert pose == target
    assert len(requests) == 1 and len(requests[0]["queries"]) == 1  # no jaw turn is tried
    reach = notes["reach_along_request"]
    assert reach["target_reachable"] is True
    assert reach["ik_model"] == ROBOTWIN_MODEL and reach["matches_move_pose_admission"] is False
    assert reach["measured_disagreement"] == {"refused_given_move_pose_admits": 0.05,
                                              "reachable_given_move_pose_refuses": 0.04}
    assert "not the 128-seed collision-checked CuRobo IK move_pose plans through" in reach["claim_boundary"]
    validate_schema_instance(reach, _target_schema(robotwin, "move_pose"))


def test_an_unsolved_target_samples_its_path_and_finds_the_nearest_turn_at_the_tcp(robotwin, monkeypatch):
    requests = []
    # 40 degrees about y: the rotation joint stops at 0.5 rad, so the target fails.
    target = [0.1, 0.05, 0.95, *_quaternion([0.0, 0.7, 0.0])]
    pose, notes = _plan(robotwin, _robotwin_bridge(monkeypatch, requests), target)
    assert pose == target
    reach = notes["reach_along_request"]
    validate_schema_instance(reach, _target_schema(robotwin, "move_pose"))
    assert reach["target_reachable"] is False and reach["reason"] in {"joint_limit", "not_converged"}
    assert reach["ik_model"] == ROBOTWIN_MODEL and reach["matches_move_pose_admission"] is False
    assert reach["claim_boundary"].startswith(reach_reference.SAMPLING_CLAIM)
    assert reach["claim_boundary"].endswith("or reach one reported unreachable.")
    assert len(reach["samples"]) == len(LINE_FRACTIONS)
    assert 0.0 < reach["farthest_reachable_fraction"] < 1.0
    nearest = reach["nearest_reachable_orientation"]
    assert nearest["status"] == "found" and nearest["rotation_from_request_deg"] in ROTATIONS_DEG
    assert nearest["claim_boundary"] == schema.NEAREST_CLAIM
    # Grip-centre frame: the jaw centre is the TCP itself, so the turn keeps the
    # requested position exactly (a descriptor offset would have moved it).
    assert nearest["jaw_center_m"] == pytest.approx(target[:3], abs=1e-5)
    assert nearest["pose"]["position_m"] == pytest.approx(target[:3], abs=1e-5)
    # Every sweep request went to the same backend check.
    assert all(request["arm"] == "left" for request in requests)


def test_a_target_no_turn_can_hold_reports_none_within_the_sweep(robotwin, monkeypatch):
    target = [0.1, 0.05, 1.5, *IDENTITY]  # beyond the toy z stop in every orientation
    _pose, notes = _plan(robotwin, _robotwin_bridge(monkeypatch), target)
    nearest = notes["reach_along_request"]["nearest_reachable_orientation"]
    assert nearest["status"] == "none_within_sweep"
    assert nearest["queries"] == len(ROTATIONS_DEG) * 12
    assert nearest["jaw_center_m"] == pytest.approx(target[:3])


def test_the_primitives_carry_the_same_v2_reference(robotwin, monkeypatch):
    host = _host(robotwin, _robotwin_bridge(monkeypatch))
    solved = move_pose_reach.primitive_reference(
        host, "left", {"target_pose_xyz_xyzw": [0.1, 0.05, 0.95, *IDENTITY]}, current_pose=CURRENT,
        interaction="free_space")["reach_along_request"]
    assert solved["target_reachable"] is True and solved["ik_model"] == ROBOTWIN_MODEL
    unsolved = move_pose_reach.primitive_reference(
        host, "left", {"target_pose_xyz_xyzw": [0.1, 0.05, 1.5, *IDENTITY]}, current_pose=CURRENT,
        interaction="precision")["reach_along_request"]
    assert unsolved["target_reachable"] is False and unsolved["measured_disagreement"] is not None
    assert unsolved["nearest_reachable_orientation"]["status"] == "none_within_sweep"
    for tool, value in (("move_to", solved), ("rotate_wrist", unsolved)):
        validate_schema_instance(value, _target_schema(robotwin, tool))


def test_a_backend_whose_check_is_the_admission_gets_no_measured_disagreement():
    answer = {"ik_model": "robodojo_sdk_ik_endpoint_selection_v2", "matches_move_pose_admission": True,
              "claim_boundary": "route IK"}
    assert reach_reference.basis(answer)["measured_disagreement"] is None
    unmeasured = {**answer, "ik_model": "some_other_approximate_ik", "matches_move_pose_admission": False}
    assert reach_reference.basis(unmeasured)["measured_disagreement"] is None
    assert reach_reference.solved(answer)["claim_boundary"] == "route IK"


def test_an_unavailable_reference_never_blocks_the_move(robotwin):
    def broken(request):
        raise RuntimeError("worker died")

    target = [0.1, 0.05, 0.95, *IDENTITY]
    pose, notes = _plan(robotwin, broken, target)
    assert pose == target
    assert notes == {"reach_along_request": {"status": "unavailable", "failure": "RuntimeError"}}
    validate_schema_instance(notes["reach_along_request"], _target_schema(robotwin, "move_pose"))


def test_a_failed_sweep_leaves_the_rest_of_the_reference(robotwin, monkeypatch):
    real = _robotwin_bridge(monkeypatch)

    def sweep_breaks(request):
        if len(request["queries"]) == 12:
            raise ToolError("bridge closed", code="environment_contract")
        return real(request)

    _pose, notes = _plan(robotwin, sweep_breaks, [0.1, 0.05, 1.5, *IDENTITY])
    reach = notes["reach_along_request"]
    assert reach["ik_model"] == ROBOTWIN_MODEL
    assert reach["nearest_reachable_orientation"] == {"status": "unavailable", "failure": "environment_contract"}


# --------------------------------------------------------------------------
# The jaw centre of a grip-centre frame
# --------------------------------------------------------------------------

def test_grip_centre_frames_hold_the_controlled_point_and_the_x5_keeps_its_contact_centre(robotwin):
    aloha = robotwin.view_config
    assert jaw_center.jaw_center_offsets(aloha) == {"left": [0.0, 0.0, 0.0], "right": [0.0, 0.0, 0.0]}
    # The descriptor formula needs a field the ALOHA descriptor does not publish.
    from ora.agent.context.contact_point_projection import jaw_contact_center_offsets
    with pytest.raises(PublicFactsError, match="reviewed jaw contact centre"):
        jaw_contact_center_offsets(aloha)
    panda = resolve_tool_surface("libero_placement_direct_overhead_v1").view_config
    assert jaw_center.jaw_center_offsets(panda) == {"right": [0.0, 0.0, 0.0]}
    x5 = resolve_tool_surface("robodojo_placement_closed_skills_v2").view_config
    assert jaw_center.jaw_center_offsets(x5) == jaw_contact_center_offsets(x5)
    assert jaw_center.jaw_center_offsets(x5)["left"] == pytest.approx([0.1501, 0.0, 0.0])


def test_a_grip_centre_convention_must_match_its_descriptor(robotwin):
    descriptor = jaw_center.load_descriptor(robotwin.view_config)
    wrong = copy.deepcopy(descriptor)
    wrong["control"]["controlled_points"][0]["description"] = "wrist_flange"
    with pytest.raises(jaw_center.JawCenterError, match="is not a tool_center"):
        jaw_center.grip_centre_offsets(wrong, "aloha_grip_center_tcp")
    with pytest.raises(jaw_center.JawCenterError, match="drifted"):
        jaw_center.jaw_center_offsets({**robotwin.view_config, "public_geometry_descriptor": {
            **robotwin.view_config["public_geometry_descriptor"], "sha256": "0" * 64}})
    with pytest.raises(jaw_center.JawCenterError, match="no jaw-centre rule"):
        jaw_center.jaw_center_offsets({"controlled_frame_convention": "unknown_frame"})


# --------------------------------------------------------------------------
# Binders
# --------------------------------------------------------------------------

def test_the_binders_refuse_what_they_cannot_describe(robotwin):
    released = resolve_tool_surface("robotwin_toolchain_b2_closed_noservo_v1")
    results = copy.deepcopy(dict(released.result_schemas))
    base = list(released.capabilities)
    with pytest.raises(ToolSurfaceError, match="alternatives"):
        schema.bind_move_pose_reach_reference(copy.deepcopy(results), [*base, "move_pose_reach_along_request_v1"])
    with pytest.raises(ToolSurfaceError, match="enhanced_move_pose_v1"):
        schema.bind_move_pose_reach_reference(copy.deepcopy(results), [])
    with pytest.raises(ToolSurfaceError, match="alternatives"):
        schema.bind_pose_primitive_reach_reference(copy.deepcopy(results),
                                                   [*base, "pose_primitive_reach_along_request_v1"])
    with pytest.raises(ToolSurfaceError, match="semantic_pose_primitives_v1"):
        schema.bind_pose_primitive_reach_reference(copy.deepcopy(results), [])
    with pytest.raises(ToolSurfaceError, match=schema.MOVE_POSE_CAPABILITY):
        schema.bind_reach_nearest_orientation_reference(copy.deepcopy(results), base, released.view_config)
    full = [*base, schema.MOVE_POSE_CAPABILITY]
    with pytest.raises(ToolSurfaceError, match="v2 reach_along_request"):
        schema.bind_reach_nearest_orientation_reference(copy.deepcopy(results), full, released.view_config)
    with pytest.raises(ToolSurfaceError, match="known jaw centre"):
        schema.bind_reach_nearest_orientation_reference(
            copy.deepcopy(results), full, {**released.view_config, "controlled_frame_convention": "other"})
    with pytest.raises(ToolSurfaceError, match="alternatives"):
        schema.bind_reach_nearest_orientation_reference(
            copy.deepcopy(results), [*full, "reach_nearest_orientation_v1"], released.view_config)
    bound = copy.deepcopy(dict(robotwin.result_schemas))
    with pytest.raises(ToolSurfaceError, match="overwrite"):
        schema.bind_move_pose_reach_reference(bound, list(robotwin.capabilities))


# --------------------------------------------------------------------------
# What the policy reads
# --------------------------------------------------------------------------

def _help(surface, tool):
    return next(item["description"] for item in surface.definitions if item["name"] == tool)


def test_the_help_states_the_backend_model_and_its_measured_error_from_one_source(robotwin):
    from ora.environment.backends.robotwin import motion_adapter

    move_pose = _help(robotwin, "move_pose")
    # The default declares the endpoint screen (M6): its model was never
    # measured against move_pose's admission, and the help says so.
    assert schema.MEASURED_DISAGREEMENT[SCREENED_MODEL] is None
    assert "has not been measured, so measured_disagreement is null" in move_pose
    assert f"ik_model {SCREENED_MODEL}, matches_move_pose_admission false" in move_pose
    assert ROBOTWIN_MODEL + "," not in move_pose
    # The unscreened model id the screen builds on is the one the RoboTwin adapter reports.
    assert f'"ik_model": "{ROBOTWIN_MODEL}"' in open(motion_adapter.__file__, encoding="utf-8").read()
    assert "approximate and is not what move_pose runs" in move_pose
    sweep = re.search(r"\((\d+) to (\d+) degrees in (\d+)-degree steps", move_pose)
    assert sweep and tuple(range(int(sweep[1]), int(sweep[2]) + 1, int(sweep[3]))) == ROTATIONS_DEG
    # The RoboDojo wording that claims the route move_pose takes is not delivered.
    assert "on its route" not in move_pose and "route move_pose takes" not in move_pose
    for tool in ("move_to", "rotate_wrist", "rotate_pitch"):
        assert "same approximate pre-check as move_pose's" in _help(robotwin, tool)
    for name in ("robotwin-move-pose-reach-reference-inter-arm-endpoint-help-v1.json",
                 "robotwin-reach-nearest-orientation-help-v1.json"):
        assert (PORTS / name).is_file()


def test_the_released_robodojo_reference_is_unchanged():
    from ora.agent.composition.move_pose_reach_schema import REACH_PROPERTY

    dojo = resolve_tool_surface("robodojo_placement_closed_skills_v2")
    assert not any(dojo.supports(name) for name in NEW_CAPABILITIES)
    reach = _target_schema(dojo, "move_pose")
    assert "ik_model" not in json.dumps(reach)
    assert reach["description"] == REACH_PROPERTY["description"]
    assert math.isclose(move_pose_reach.JOINT6_MARGIN_FLOOR_RAD, round(math.pi / 4, 6))
    assert np.isclose(len(LINE_FRACTIONS), 7)
