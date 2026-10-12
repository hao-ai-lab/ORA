"""The RoboTwin reach check's other-arm screen, wired per surface (integration 0059, M6).

User decision 2026-10-04: the screen checks the reached configuration alone
(``robotwin_reach_check_inter_arm_screen_v1``). The RunSpec derives
``runtime.reach_check_inter_arm_screen`` from the bound surface, the factory
and worker pass it to ``RoboTwinLiveSession``, and the adapter's
``reach_check_batch`` screens every solved row. The screen names its own
``ik_model`` with an unmeasured (null) disagreement, its own reason and claim
boundary, and its help restates M1's sentence for the screened model.

Offline: the real RoboTwin reach-check path (session ``align_plan`` ->
``plan_alignment`` -> ``check_reach`` -> adapter ``reach_check_batch``) over a
two-arm toy whose joints are its TCP pose and whose chain is shoulder + hand,
with the RoboTwin control profile's ``dual_arm_collision_check``; the surface is
the RoboTwin default, which declares the endpoint screen.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ora.agent.composition import reach_reference_schema as schema
from ora.agent.composition import tool_surface
from ora.agent.composition.profile_catalog import PROFILE_CATALOG
from ora.agent.composition.tool_surface import ToolSurfaceError, validate_schema_instance
from ora.agent.tools import move_pose_reach, reach_reference
from ora.contracts.reach_check import (
    REQUEST_SCHEMA, ROBOTWIN_INTER_ARM_IK_MODELS, ROBOTWIN_INTER_ARM_REASONS, ROBOTWIN_REASONS, SHARED_REASONS,
)
from ora.environment.backends.robotwin import reach_inter_arm
from ora.orchestrator import spec_runtime
from ora.orchestrator.spec_values import RunSpecError

ROOT = Path(__file__).resolve().parents[1]
CONTROL_PROFILE_PATH = ROOT / "ora/policy/embodiments/aloha-agilex/control-profile-v10.json"
COLLISION = json.loads(CONTROL_PROFILE_PATH.read_text(encoding="utf-8"))["dual_arm_collision_check"]
ARRIVAL = {"position_tolerance_m": 0.005, "orientation_tolerance_rad": 0.05}
IDENTITY = [0.0, 0.0, 0.0, 1.0]
#: Each toy arm: a fixed shoulder and a hand at its first three joints.
SHOULDERS = {"left": [-0.3, 0.0, 1.0], "right": [0.3, 0.0, 1.0]}
MEASURED = {"left": [-0.3, 0.0, 0.8, 0.0, 0.0, 0.0], "right": [0.3, 0.0, 0.8, 0.0, 0.0, 0.0]}
LIMITS = [[-1.0, 1.0], [-1.0, 1.0], [0.0, 1.2], [-3.0, 3.0], [-3.0, 3.0], [-3.0, 3.0]]
#: The left hand's straight path to here would pass through the right hand; the end is 0.3 m clear.
CROSSING = [0.6, 0.0, 0.8, *IDENTITY]
#: 50 mm from the right hand: inside the 95 mm sphere-centre distance at the end itself.
TOO_CLOSE = [0.25, 0.0, 0.8, *IDENTITY]
CLEAR = [-0.2, 0.05, 0.85, *IDENTITY]
UNSCREENED = "robotwin_single_seed_damped_least_squares_v1"
SCREEN = schema.INTER_ARM_SCREEN_CAPABILITY
FLAGS = {None: False, "endpoint": True}
#: The RoboTwin default: M1-M5 ports with the endpoint screen's help in M1's place.
DEFAULT = "robotwin_toolchain_b2_closed_noservo_v2"
PORTS = ROOT / "ora/policy/tool_surfaces/closed_skills_ports_v1"


def _adapter(mode):
    """The real adapter methods over the two-arm toy, with a session carrying the screen flag."""
    from ora.environment.backends.robotwin.motion_adapter import RoboTwinMotionBackendAdapter
    from scipy.spatial.transform import Rotation

    adapter = object.__new__(RoboTwinMotionBackendAdapter)
    adapter._environment = object()
    adapter._session = SimpleNamespace(
        reach_check_inter_arm_screen=FLAGS[mode],
        control_profile={"profile": {"dual_arm_collision_check": COLLISION, "arrival_check": ARRIVAL}},
        _pinocchio_chain_points=lambda environment, arm, joints: [SHOULDERS[arm], [float(v) for v in joints[:3]]],
    )
    adapter.read_joint_positions = lambda arm: list(MEASURED[arm])
    adapter.joint_limits = lambda arm: LIMITS
    adapter.fk_public_tcp_pose = lambda arm, joints: [*joints[:3], *Rotation.from_rotvec(joints[3:6]).as_quat()]
    return adapter


def _bridge(monkeypatch, mode, requests=None):
    """Session align_plan over the real controllers and the toy adapter."""
    from ora.environment.backends.robotwin import session as robotwin_session

    adapter = _adapter(mode)
    fake = SimpleNamespace(_require_environment=lambda: object(), control_profile=adapter._session.control_profile,
                           _motion_backend_adapter=lambda **kwargs: adapter)

    def bridge(request):
        if requests is not None:
            requests.append(copy.deepcopy(request))
        return robotwin_session.RoboTwinLiveSession.align_plan(fake, request)
    return bridge


def _ask(monkeypatch, mode, *queries):
    request = {"schema": REQUEST_SCHEMA, "arm": "left", "queries": [
        {"pose_xyzw": list(pose), "interaction": "free_space", "from_index": start} for pose, start in queries]}
    return _bridge(monkeypatch, mode)(request)


# --------------------------------------------------------------------------
# The adapter: the screen checks the end configuration
# --------------------------------------------------------------------------

def test_without_the_flag_the_check_is_the_unscreened_one(monkeypatch):
    answer = _ask(monkeypatch, None, (CROSSING, None), (TOO_CLOSE, None))
    assert answer["ik_model"] == UNSCREENED
    assert [row["reachable"] for row in answer["results"]] == [True, True]  # no collision model
    assert "joint limits only: no collision model" in answer["claim_boundary"]


def test_the_endpoint_screen_passes_a_crossing_path_with_a_clear_end_and_refuses_a_close_end(monkeypatch):
    answer = _ask(monkeypatch, "endpoint", (CROSSING, None), (TOO_CLOSE, None), (CLEAR, 1), (CLEAR, 0))
    crossing, close, orphan, chained = answer["results"]
    assert crossing["reachable"] is True and crossing["reason"] is None
    assert close == {**close, "reachable": False, "reason": "inter_arm_collision_endpoint"}
    # The solved numbers of the blocked configuration stay; no chained query starts from it.
    assert close["position_error_m"] <= ARRIVAL["position_tolerance_m"]
    assert orphan["reachable"] is False and orphan["reason"] == "start_unreachable"
    assert chained["reachable"] is True
    assert answer["ik_model"] == ROBOTWIN_INTER_ARM_IK_MODELS["endpoint"]
    assert answer["matches_move_pose_admission"] is False
    assert answer["claim_boundary"] == reach_inter_arm.CLAIM_BOUNDARIES["endpoint"]
    assert answer["motion_planner_called"] is False


def test_the_right_arm_is_screened_against_the_held_left_arm(monkeypatch):
    request = {"schema": REQUEST_SCHEMA, "arm": "right", "queries": [
        {"pose_xyzw": [-0.25, 0.0, 0.8, *IDENTITY], "interaction": "free_space", "from_index": None}]}
    row = _bridge(monkeypatch, "endpoint")(request)["results"][0]
    assert row["reachable"] is False and row["reason"] == "inter_arm_collision_endpoint"


# --------------------------------------------------------------------------
# Session, worker and factory
# --------------------------------------------------------------------------

def _live_session(tmp_path, **kwargs):
    from ora.environment.backends.robotwin.session import RoboTwinLiveSession
    from ora.environment.controllers.embodiment_control import load_embodiment_control_profile

    binding = load_embodiment_control_profile(
        CONTROL_PROFILE_PATH, hashlib.sha256(CONTROL_PROFILE_PATH.read_bytes()).hexdigest())
    output = tmp_path / f"output-{sum(1 for _ in tmp_path.glob('output-*'))}"  # one replay root each
    return RoboTwinLiveSession(
        robotwin_root=tmp_path, output_root=output, task="blocks_ranking_size", actual_seed=100,
        episode=0, action_codec="motion_action_v3", control_profile=binding,
        task_factory=lambda _task: object(), **kwargs)


def test_the_session_takes_a_valid_flag_and_refuses_the_rest(tmp_path):
    from ora.environment.backends.robotwin.session import RoboTwinLiveError, RoboTwinLiveSession

    assert _live_session(tmp_path).reach_check_inter_arm_screen is False
    assert _live_session(tmp_path, reach_check_inter_arm_screen=True).reach_check_inter_arm_screen is True
    with pytest.raises(RoboTwinLiveError, match="Boolean"):
        _live_session(tmp_path, reach_check_inter_arm_screen="true")
    stub = SimpleNamespace(reach_check_inter_arm_screen=True,
                           control_profile={"profile": {"arrival_check": ARRIVAL}})
    with pytest.raises(RoboTwinLiveError, match="dual_arm_collision_check"):
        RoboTwinLiveSession._validate_reach_check_screen(stub)


@pytest.mark.parametrize("mode", list(FLAGS))
def test_the_runtime_fields_reach_the_isolated_session_constructor(tmp_path, monkeypatch, mode):
    from ora.environment import factory, worker
    from ora.environment.execution_capture import resolve_execution_capture

    screen = FLAGS[mode]
    runtime = dict(kind="robotwin", execution_capture=resolve_execution_capture(None), startup_timeout_s=1,
                   request_timeout_s=1, task_config="demo_randomized", required_revision="source",
                   workspace_bounds_m={"x": [-1, 1], "y": [-1, 1], "z": [0, 2]},
                   **({"reach_check_inter_arm_screen": True} if screen else {}))
    context = factory._BuildContext(runtime=runtime, output_root=tmp_path / "native", task="fixture-task", seed=0,
                                    max_sim_steps=10, redacted_environment_names=(), qualified_runtime_binding=None,
                                    action_codec="runtime_action_v3")
    monkeypatch.setattr(factory, "_required_environment_path", lambda *a, **kw: tmp_path)
    monkeypatch.setattr(factory, "LocalRpcEnvironmentSession", lambda **kwargs: SimpleNamespace(**kwargs))
    config = {**factory._robotwin(context).worker_config, "shared_root": str(tmp_path)}
    # A runtime without the screen keeps every worker-config key it had.
    assert {key for key in config if "inter_arm" in key} == {key for key in runtime if "inter_arm" in key}
    monkeypatch.setattr("ora.environment.backends.robotwin.session.RoboTwinLiveSession",
                        lambda **kwargs: kwargs)
    constructed = worker._build_session(config)
    assert constructed["reach_check_inter_arm_screen"] is screen


# --------------------------------------------------------------------------
# RunSpec derivation
# --------------------------------------------------------------------------

def _surface(*capabilities):
    return SimpleNamespace(capabilities=tuple(capabilities), supports=lambda name: name in capabilities)


SPARSE = {"action_codec": "motion_action_v3"}


@pytest.mark.parametrize("capabilities,expected", [
    ((), {}),
    ((SCREEN,), {"reach_check_inter_arm_screen": True}),
])
def test_the_runtime_fields_follow_the_surface(capabilities, expected):
    runtime = {"kind": "robotwin"}
    spec_runtime._bind_reach_check_inter_arm_screen(runtime, SPARSE, _surface(*capabilities))
    assert runtime == {"kind": "robotwin", **expected}  # absent capability -> no key at all
    again = dict(runtime)
    spec_runtime._bind_reach_check_inter_arm_screen(again, SPARSE, _surface(*capabilities))
    assert again == runtime  # a hand-set field that agrees is kept


def test_the_runspec_refuses_what_the_surface_does_not_say():
    for runtime, capabilities in (
        ({"reach_check_inter_arm_screen": True}, ()),
        ({"reach_check_inter_arm_screen": False}, ()),
        ({"reach_check_inter_arm_screen": False}, (SCREEN,)),
        ({"reach_check_inter_arm_screen": "true"}, (SCREEN,)),
    ):
        with pytest.raises(RunSpecError, match="disagrees with the ToolSurface"):
            spec_runtime._bind_reach_check_inter_arm_screen({"kind": "robotwin", **runtime}, SPARSE,
                                                            _surface(*capabilities))
    with pytest.raises(RunSpecError, match="requires a sparse motion action"):
        spec_runtime._bind_reach_check_inter_arm_screen({"kind": "robotwin"}, {"action_codec": "runtime_action_v3"},
                                                        _surface(SCREEN))
    for runtime, capabilities in (({"kind": "libero_pro"}, (SCREEN,)),
                                  ({"kind": "robodojo", "reach_check_inter_arm_screen": True}, ())):
        with pytest.raises(RunSpecError, match="require runtime.kind robotwin"):
            spec_runtime._require_surface_runtime({"kind": "native_api"}, runtime, _surface(*capabilities))


def test_a_fresh_robotwin_runspec_carries_exactly_what_its_surface_declares():
    from ora.agent.composition import resolve_tool_surface
    from ora.orchestrator.spec import resolve_run_spec
    from test_native_baseline import native_spec

    spec = native_spec("robotwin")
    plan = resolve_run_spec(spec, root=ROOT)
    surface = resolve_tool_surface(plan["policy"]["tool_surface"]["profile"])
    for field, capability in spec_runtime.REACH_SCREEN_FIELDS.items():
        if surface.supports(capability):
            assert plan["runtime"][field] is True
        else:
            assert field not in plan["runtime"]
    spec["runtime"]["reach_check_inter_arm_screen"] = not surface.supports(SCREEN)
    with pytest.raises(RunSpecError, match="reach_check_inter_arm_screen"):
        resolve_run_spec(spec, root=ROOT)


# --------------------------------------------------------------------------
# What the policy reads: the screened model, null disagreement, its help
# --------------------------------------------------------------------------

def _screened_candidate(mode):
    """The registered surface that declares ``mode``'s screen, resolved."""
    surface = tool_surface.resolve_tool_surface(DEFAULT)
    assert schema.inter_arm_screen_mode(list(surface.capabilities)) == mode
    assert surface.supports("grasp_plan_ik_reachability_annotation_v1")
    return surface


def _help(surface, tool):
    return next(item["description"] for item in surface.definitions if item["name"] == tool)


def _target_schema(surface):
    return surface.result_schemas["move_pose"]["properties"]["compiled_pose"]["properties"]["targets"][
        "additionalProperties"]["properties"]["reach_along_request"]


def _plan(surface, bridge, target):
    host = SimpleNamespace(surface=surface, align_plan_callback=bridge,
                           workspace_bounds_m={"x": [-1.0, 1.0], "y": [-1.0, 1.0], "z": [0.0, 2.0]})
    return move_pose_reach.plan_target(
        host, "left", current_pose=[*MEASURED["left"][:3], *IDENTITY], target_pose=list(target),
        interaction="free_space", orientation_mode="absolute_quaternion",
        observation={"proprioception": {"end_effectors": {"left": {"gripper_command_target": "open"}}}})


def test_the_screened_models_are_listed_unmeasured():
    for model in ROBOTWIN_INTER_ARM_IK_MODELS.values():
        assert model in schema.MEASURED_DISAGREEMENT and schema.MEASURED_DISAGREEMENT[model] is None
        assert reach_reference.basis({"ik_model": model, "matches_move_pose_admission": False}) == {
            "ik_model": model, "matches_move_pose_admission": False, "measured_disagreement": None}
    assert schema.MEASURED_DISAGREEMENT[UNSCREENED] is not None
    assert reach_inter_arm.IK_MODELS == ROBOTWIN_INTER_ARM_IK_MODELS


@pytest.mark.parametrize("mode", ["endpoint"])
def test_move_pose_reports_the_screened_check_with_null_disagreement(monkeypatch, mode):
    surface = _screened_candidate(mode)
    bridge = _bridge(monkeypatch, mode)
    _pose, notes = _plan(surface, bridge, CLEAR)
    reach = notes["reach_along_request"]
    assert reach["target_reachable"] is True
    assert reach["ik_model"] == ROBOTWIN_INTER_ARM_IK_MODELS[mode]
    assert reach["matches_move_pose_admission"] is False and reach["measured_disagreement"] is None
    assert reach["claim_boundary"] == reach_inter_arm.CLAIM_BOUNDARIES[mode]
    validate_schema_instance(reach, _target_schema(surface))
    _pose, notes = _plan(surface, bridge, CROSSING)
    assert notes["reach_along_request"]["target_reachable"] is (mode == "endpoint")
    _pose, notes = _plan(surface, bridge, TOO_CLOSE)
    blocked = notes["reach_along_request"]
    assert blocked["target_reachable"] is False and blocked["reason"] == reach_inter_arm.REASONS[mode]
    assert blocked["measured_disagreement"] is None
    assert blocked["claim_boundary"].endswith(reach_inter_arm.CLAIM_BOUNDARIES[mode])
    validate_schema_instance(blocked, _target_schema(surface))


@pytest.mark.parametrize("mode", ["endpoint"])
def test_the_screened_help_replaces_m1_and_says_the_disagreement_is_unmeasured(mode):
    surface = _screened_candidate(mode)
    move_pose = _help(surface, "move_pose")
    model = ROBOTWIN_INTER_ARM_IK_MODELS[mode]
    assert f"ik_model {model}, matches_move_pose_admission false" in move_pose
    assert "has not been measured, so measured_disagreement is null" in move_pose
    assert f"reason {reach_inter_arm.REASONS[mode]}" in move_pose
    assert "held at its measured joints" in move_pose
    # None of the unscreened check's model or measured figures is delivered.
    assert UNSCREENED not in move_pose and "about 5%" not in move_pose and "about 4%" not in move_pose
    assert "not the path to it" in move_pose
    # M3 follows it unchanged and refers to "the same pre-check".
    assert "that the same pre-check solves" in move_pose
    for tool in ("move_to", "rotate_wrist", "rotate_pitch"):
        assert "same approximate pre-check as move_pose's" in _help(surface, tool)
    layer = json.loads((PORTS / Path(schema.INTER_ARM_HELP[mode]).name).read_text(encoding="utf-8"))
    assert set(layer) == {"append"} and "move_pose" in layer["append"]
    # The binder accepts it (over the grasp_plan reasons as they were before the screen widened them).
    results = copy.deepcopy(dict(surface.result_schemas))
    for stage in results["grasp_plan"]["$defs"]["candidate"]["properties"]["ik_reachability"]["properties"].values():
        if isinstance(stage, dict) and "reason" in stage.get("properties", {}):
            stage["properties"]["reason"]["enum"] = [*SHARED_REASONS, *ROBOTWIN_REASONS, None]
    schema.bind_inter_arm_screen(surface.definitions, results, list(surface.capabilities))


@pytest.mark.parametrize("mode", ["endpoint"])
def test_the_binder_widens_the_grasp_reasons_and_refuses_what_it_cannot_describe(mode):
    surface = _screened_candidate(mode)
    capabilities = list(surface.capabilities)
    # Resolution already bound the screen (tool_surface_resolution.bind_capability_contracts).
    bound = surface.result_schemas["grasp_plan"]["$defs"]["candidate"]["properties"]["ik_reachability"]["properties"]
    for stage in ("pregrasp", "approach"):
        assert bound[stage]["properties"]["reason"]["enum"] == [
            *SHARED_REASONS, *ROBOTWIN_INTER_ARM_REASONS[mode], None]
    assert any(item["slot"] == f"derived:{SCREEN}" for item in surface.components)

    def unscreened_results():
        value = copy.deepcopy(dict(surface.result_schemas))
        for stage in value["grasp_plan"]["$defs"]["candidate"]["properties"]["ik_reachability"]["properties"].values():
            if isinstance(stage, dict) and "reason" in stage.get("properties", {}):
                stage["properties"]["reason"]["enum"] = [*SHARED_REASONS, *ROBOTWIN_REASONS, None]
        return value

    results = unscreened_results()
    stages = results["grasp_plan"]["$defs"]["candidate"]["properties"]["ik_reachability"]["properties"]
    assert stages["pregrasp"]["properties"]["reason"]["enum"] == [*SHARED_REASONS, *ROBOTWIN_REASONS, None]
    descriptor = schema.bind_inter_arm_screen(surface.definitions, results, capabilities)
    assert descriptor["slot"] == f"derived:{SCREEN}"
    assert descriptor["ref"] == "ora/agent/composition/reach_reference_schema.py"
    for stage in ("pregrasp", "approach"):
        assert stages[stage]["properties"]["reason"]["enum"] == [
            *SHARED_REASONS, *ROBOTWIN_INTER_ARM_REASONS[mode], None]
    with pytest.raises(ToolSurfaceError, match="unscreened RoboTwin stage reason set"):
        schema.bind_inter_arm_screen(surface.definitions, results, capabilities)  # twice
    fresh = unscreened_results()
    with pytest.raises(ToolSurfaceError, match="robotwin_rgbd_v1"):
        schema.bind_inter_arm_screen(surface.definitions, fresh, [c for c in capabilities if c != "robotwin_rgbd_v1"])
    with pytest.raises(ToolSurfaceError, match="reach-check consumer"):
        schema.bind_inter_arm_screen(surface.definitions, fresh, ["robotwin_rgbd_v1", SCREEN])
    unscreened =[{**item, "description": item["description"].replace(
        ROBOTWIN_INTER_ARM_IK_MODELS[mode], UNSCREENED)} for item in surface.definitions]
    with pytest.raises(ToolSurfaceError, match=Path(schema.INTER_ARM_HELP[mode]).name):
        schema.bind_inter_arm_screen(unscreened, fresh, capabilities)


# --------------------------------------------------------------------------
# Registered surfaces: help, reason set and binder agree with the declaration
# --------------------------------------------------------------------------

def _declared():
    for profile_id, path in sorted(PROFILE_CATALOG.items()):
        capabilities = json.loads(Path(path).read_text(encoding="utf-8")).get("capabilities", [])
        yield profile_id, capabilities


def test_every_registered_screen_surface_delivers_the_screened_help_reasons_and_derivation():
    from ora.agent.composition import resolve_tool_surface

    for profile_id, capabilities in _declared():
        mode = schema.inter_arm_screen_mode(capabilities)
        if mode is None:
            continue
        surface = resolve_tool_surface(profile_id)
        slot = f"derived:{SCREEN}"
        assert slot in {item["slot"] for item in surface.components}, (
            f"{profile_id}: register reach_reference_schema.bind_inter_arm_screen in bind_capability_contracts")
        if surface.supports(schema.MOVE_POSE_CAPABILITY):
            move_pose = _help(surface, "move_pose")
            assert f"ik_model {ROBOTWIN_INTER_ARM_IK_MODELS[mode]}" in move_pose, profile_id
            assert UNSCREENED not in move_pose, profile_id
        if surface.supports("grasp_plan_ik_reachability_annotation_v1"):
            stage = surface.result_schemas["grasp_plan"]["$defs"]["candidate"]["properties"]["ik_reachability"][
                "properties"]["pregrasp"]["properties"]["reason"]
            assert stage["enum"] == [*SHARED_REASONS, *ROBOTWIN_INTER_ARM_REASONS[mode], None], profile_id


def test_no_registered_surface_without_the_screen_delivers_a_screened_model():
    screened = tuple(ROBOTWIN_INTER_ARM_IK_MODELS.values())
    for profile_id, capabilities in _declared():
        if schema.inter_arm_screen_mode(capabilities) is not None:
            continue
        text = Path(PROFILE_CATALOG[profile_id]).read_text(encoding="utf-8")
        assert not any(Path(name).name in text for name in schema.INTER_ARM_HELP.values()), profile_id
        assert not any(model in text for model in screened), profile_id
