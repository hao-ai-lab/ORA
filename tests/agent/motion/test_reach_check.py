"""The internal reach check and grasp_plan's pooled IK selection, on both dual-arm backends.

EXP11A g003 recorded 13 of 24 grasp_execute calls refused at the precommit
check. grasp_plan now checks its whole compiled pool with each backend's
existing read-only IK -- on RoboDojo the SDK IK on the move_pose route the
interaction selects, the approach chained onto its pregrasp -- before the
display cap. The check is internal: by decision (2026-09-29) no ToolSurface
declares ``reach_check`` as a policy tool, and these tests hold that too.

Everything here is offline: the RoboDojo session is the fake SDK plant, the
RoboTwin adapter gets a toy forward-kinematics model, and the tool runtime is
driven through its real dispatcher and result-schema admission.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from tests.support.surface_copies import scratch_profile_root
from ora.agent.composition import resolve_tool_surface
from ora.agent.tools import reach_check as policy_reach
from ora.agent.tools.robot import ToolError
from ora.contracts.alignment import REQUEST_SCHEMA as ALIGN_REQUEST_SCHEMA
from ora.contracts.reach_check import (
    REQUEST_SCHEMA, RESULT_SCHEMA, ReachCheckRequestDrift,
)
from ora.contracts.refusal import EnvironmentRefusal
from ora.environment.controllers.alignment import plan_alignment
from ora.environment.controllers.reach_check import check_reach

LIVE = (
    "robotwin_toolchain_b2_closed_noservo_v2",
    "robodojo_placement_closed_skills_v2",
)
ARRIVAL = {"profile": {"arrival_check": {"position_tolerance_m": 0.005, "orientation_tolerance_rad": 0.05}}}
IDENTITY = [0.0, 0.0, 0.0, 1.0]


def _query(pose, interaction="free_space", from_index=None):
    return {"pose_xyzw": list(pose), "interaction": interaction, "from_index": from_index}


def _request(queries, *, arm="left"):
    return {"schema": REQUEST_SCHEMA, "arm": arm, "queries": [dict(item) for item in queries]}


# --------------------------------------------------------------------------
# Surfaces
# --------------------------------------------------------------------------

@pytest.mark.parametrize("profile", LIVE)
def test_the_live_surfaces_select_the_pooled_ik_check_without_declaring_reach_check(profile):
    from ora.agent.composition.reach_reference_schema import INTER_ARM_SCREEN_CAPABILITY
    from ora.agent.composition.tool_surface import PROFILE_CATALOG

    surface = resolve_tool_surface(profile)
    names = [item["name"] for item in surface.definitions]
    assert "reach_check" not in names and "reach_check" not in surface.result_schemas
    # The 0059 M6 other-arm screen configures the internal check; it declares no tool.
    assert not any("reach_check" in capability for capability in surface.capabilities
                   if capability != INTER_ARM_SCREEN_CAPABILITY)
    assert surface.supports(policy_reach.GRASP_ANNOTATION_CAPABILITY)
    manifest = json.loads(PROFILE_CATALOG[profile].read_text())
    assert "supplementary_definitions" not in manifest
    # The shared file 18 surfaces name, frozen ones included, keeps its bytes. The
    # live surfaces name its wait-v1 successor (EXP11A g004), which adds only the
    # wait_ticks definition beside it.
    shared = PROFILE_CATALOG[profile].parent.parent / "explicit_motion_v3"
    assert hashlib.sha256((shared / "robotwin-additional-definitions.json").read_bytes()).hexdigest() == (
        "3422060964dfeb9955231105ebc4bb495f7151ada212384157d2062646ae1360")
    assert (PROFILE_CATALOG[profile].parent / manifest["additional_definitions"]).resolve() == (
        shared / "robotwin-additional-definitions-wait-v1.json").resolve()
    released = json.loads((shared / "robotwin-additional-definitions.json").read_text())["tools"]
    live = json.loads((shared / "robotwin-additional-definitions-wait-v1.json").read_text())["tools"]
    assert [tool for tool in live if tool["name"] != "wait_ticks"] == released
    grasp_help = next(entry["description"] for entry in surface.definitions if entry["name"] == "grasp_plan")
    assert "IK reference at planning time, not a guarantee" in grasp_help
    assert "pool of up to 32" in grasp_help
    for definition in surface.definitions:  # help never names the undeclared tool
        assert "reach_check" not in definition["description"], definition["name"]


def test_a_supplementary_layer_cannot_redefine_a_tool(tmp_path, monkeypatch):
    from ora.agent.composition import tool_surface

    source = tool_surface.PROFILE_CATALOG["robotwin_toolchain_b2_closed_noservo_v2"]
    folder = source.parent.relative_to(tool_surface.PROFILE_ROOT)
    manifest = json.loads(source.read_text())
    manifest["supplementary_definitions"] = ["__reach_check_duplicate_definitions.json"]
    # A scratch copy of the catalog tree with its own lock: the checkout is never written.
    profile_root = scratch_profile_root(monkeypatch, tmp_path, {
        f"{folder}/__reach_check_duplicate_definitions.json": {"tools": [{
            "name": "align_plan", "description": "x", "inputSchema": {"type": "object"}}]},
        f"{folder}/__reach_check_duplicate_manifest.json": manifest,
    }, copy_tree=True)
    monkeypatch.setitem(tool_surface.PROFILE_CATALOG, "robotwin_toolchain_b2_closed_noservo_v2",
                        profile_root / folder / "__reach_check_duplicate_manifest.json")
    with pytest.raises(tool_surface.ToolSurfaceError, match="redefine"):
        tool_surface.resolve_tool_surface("robotwin_toolchain_b2_closed_noservo_v2")


def test_the_annotation_reason_set_is_the_backend_check_s_own_set():
    from ora.agent.composition.reach_reference_schema import inter_arm_screen_mode
    from ora.contracts.reach_check import (
        ROBODOJO_REASONS, ROBOTWIN_INTER_ARM_REASONS, ROBOTWIN_REASONS, SHARED_REASONS,
    )

    for profile in LIVE:
        surface = resolve_tool_surface(profile)
        stage = surface.result_schemas["grasp_plan"]["$defs"]["candidate"]["properties"][
            "ik_reachability"]["properties"]["pregrasp"]["properties"]["reason"]
        # A RoboTwin surface with the 0059 M6 other-arm screen publishes the screened set.
        screen = inter_arm_screen_mode(surface.capabilities)
        backend = (ROBODOJO_REASONS if profile.startswith("robodojo")
                   else ROBOTWIN_INTER_ARM_REASONS[screen] if screen else ROBOTWIN_REASONS)
        assert stage["enum"] == [*SHARED_REASONS, *backend, None]


# --------------------------------------------------------------------------
# Shared controller
# --------------------------------------------------------------------------

class _Adapter:
    def __init__(self, rows=None):
        self.calls = []
        self.rows = rows

    def reach_check_batch(self, arm, queries, **tolerances):
        self.calls.append((arm, copy.deepcopy(queries), tolerances))
        rows = self.rows or [{"reachable": True, "reason": None, "route": None, "position_error_m": 0.001,
                              "orientation_error_rad": 0.002, "joint_limit_margin_rad": 0.5,
                              "joint_delta_rad": 0.1} for _ in queries]
        return {"ik_model": "stub_ik", "matches_move_pose_admission": False,
                "claim_boundary": "stub", "results": rows}


def test_the_bridge_routes_reach_checks_and_leaves_alignment_requests_alone():
    adapter = _Adapter()
    answer = plan_alignment(
        _request([_query([0.1, 0, 0.9, *IDENTITY]), _query([0.1, 0, 0.85, *IDENTITY], "contact_approach", 0)]),
        adapter=adapter, control_profile=ARRIVAL, available_arms=("left", "right"))
    assert answer["schema"] == RESULT_SCHEMA and answer["motion_planner_called"] is False
    assert answer["matches_move_pose_admission"] is False
    assert adapter.calls[0][1][1]["from_index"] == 0
    assert adapter.calls[0][2] == {"position_tolerance_m": 0.005, "orientation_tolerance_rad": 0.05}
    align = SimpleNamespace(solve_batch_reachability=lambda arm, poses, **kw: [
        {"reachable": True, "position_error_m": 0.0, "orientation_error_rad": 0.0,
         "iterations": 1, "claim_boundary": "IK"} for _ in poses])
    aligned = plan_alignment(
        {"schema": ALIGN_REQUEST_SCHEMA, "target_ref": "t", "approach_side": "top", "arm": "left",
         "target_poses_xyzw": [[0.1, 0, 0.9, *IDENTITY]]},
        adapter=align, control_profile=ARRIVAL, available_arms=("left", "right"))
    assert aligned["schema"] == "ora.align-plan-result.v1"


@pytest.mark.parametrize(("request_", "field"), [
    (_request([_query([0, 0, 1, *IDENTITY])], arm="middle"), "arm"),
    (_request([_query([0, 0, 1, *IDENTITY])] * 17), "queries"),
    (_request([_query([0, 0, 1, *IDENTITY], "sprint")]), "queries[0].interaction"),
    (_request([_query([0, 0, 1, *IDENTITY], from_index=0)]), "queries[0].from_index"),
])
def test_the_controller_refuses_correctable_input(request_, field):
    with pytest.raises(EnvironmentRefusal) as refused:
        check_reach(request_, adapter=_Adapter(), control_profile=ARRIVAL, available_arms=("left", "right"))
    assert refused.value.field == field


def test_the_controller_rejects_backend_drift():
    with pytest.raises(ReachCheckRequestDrift):
        check_reach({**_request([_query([0, 0, 1, *IDENTITY])]), "extra": 1}, adapter=_Adapter(),
                    control_profile=ARRIVAL, available_arms=("left",))
    bad = _Adapter(rows=[{"reachable": False, "reason": "made_up", "route": None, "position_error_m": None,
                          "orientation_error_rad": None, "joint_limit_margin_rad": None,
                          "joint_delta_rad": None}])
    with pytest.raises(ReachCheckRequestDrift, match="undeclared"):
        check_reach(_request([_query([0, 0, 1, *IDENTITY])]), adapter=bad, control_profile=ARRIVAL,
                    available_arms=("left",))


# --------------------------------------------------------------------------
# RoboTwin: the damped-least-squares IK align_plan already uses
# --------------------------------------------------------------------------

def _robotwin_adapter(limits):
    """The real adapter methods over a toy arm whose joints are its pose."""
    from ora.environment.backends.robotwin.motion_adapter import RoboTwinMotionBackendAdapter
    from scipy.spatial.transform import Rotation

    adapter = object.__new__(RoboTwinMotionBackendAdapter)
    # The unscreened check: the session's surface did not ask for the 0059 M6
    # other-arm screen (test_robotwin_reach_inter_arm_wiring.py covers it).
    adapter._session = SimpleNamespace(reach_check_inter_arm_screen=False,
                                       reach_check_inter_arm_path_screen=False)
    adapter.read_joint_positions = lambda arm: [0.0, 0.0, 0.9, 0.0, 0.0, 0.0]
    adapter.joint_limits = lambda arm: limits
    adapter.fk_public_tcp_pose = lambda arm, joints: [*joints[:3], *Rotation.from_rotvec(joints[3:6]).as_quat()]
    return adapter


def test_robotwin_answers_chains_and_names_its_weaker_claim(monkeypatch):
    from ora.environment.backends.robotwin import session as robotwin_session

    limits = [[-1.0, 1.0], [-1.0, 1.0], [0.0, 1.2], [-3.0, 3.0], [-3.0, 3.0], [-3.0, 3.0]]
    adapter = _robotwin_adapter(limits)
    monkeypatch.setattr(robotwin_session, "RoboTwinMotionBackendAdapter", lambda **kwargs: adapter)
    # Since the RoboTwin session split every motion mixin builds its adapter
    # through ``_motion_backend_adapter``, so the stand-in session carries it.
    fake = SimpleNamespace(_require_environment=lambda: object(), control_profile=ARRIVAL,
                           _motion_backend_adapter=lambda **kwargs: adapter)
    answer = robotwin_session.RoboTwinLiveSession.align_plan(fake, _request([
        _query([0.1, 0.05, 0.95, *IDENTITY]),                                # inside the toy limits
        _query([0.1, 0.05, 1.5, *IDENTITY]),                                 # z beyond the joint stop
        _query([0.12, 0.05, 0.95, *IDENTITY], "contact_approach", 0),       # chained onto 0
        _query([0.1, 0.05, 1.0, *IDENTITY], "contact_approach", 1),         # chained onto a refusal
    ]))
    assert answer["ik_model"] == "robotwin_single_seed_damped_least_squares_v1"
    assert answer["matches_move_pose_admission"] is False
    reachable, blocked, chained, orphan = answer["results"]
    assert reachable["reachable"] is True and reachable["route"] is None
    assert reachable["position_error_m"] <= 0.005
    assert reachable["joint_limit_margin_rad"] == pytest.approx(0.25, abs=1e-3)
    assert reachable["joint_delta_rad"] == pytest.approx(0.1, abs=1e-3)
    assert blocked["reachable"] is False and blocked["reason"] == "joint_limit"
    assert blocked["joint_limit_margin_rad"] == pytest.approx(0.0, abs=1e-9)
    assert chained["reachable"] is True
    assert chained["joint_delta_rad"] == pytest.approx(0.02, abs=1e-3)  # from the chained start
    assert orphan == {**orphan, "reachable": False, "reason": "start_unreachable", "position_error_m": None}
    assert "not the 128-seed collision-checked CuRobo IK move_pose plans through" in answer["claim_boundary"]


# --------------------------------------------------------------------------
# RoboDojo: the SDK IK selection on the modelled move_pose route
# --------------------------------------------------------------------------

@pytest.fixture
def dojo(tmp_path, monkeypatch):
    import tests.environments.robodojo.test_robodojo_backend as backend
    backend.fake_sdk_candidate_boundary.__wrapped__(monkeypatch)
    instance = backend.sparse_session(tmp_path)
    try:
        yield instance
    finally:
        instance.close()


def _dojo(dojo, *queries):
    answer = dojo.align_plan(_request(queries))
    assert answer["schema"] == RESULT_SCHEMA and answer["arm"] == "left"
    assert answer["ik_model"] == "robodojo_sdk_ik_endpoint_selection_v2"
    assert answer["matches_move_pose_admission"] is True
    return answer["results"]


def test_robodojo_admits_a_pose_on_the_global_route_without_writing(dojo):
    [row] = _dojo(dojo, _query([0.02, 0.0, 0.8, *IDENTITY]))
    assert row["reachable"] is True and row["reason"] is None and row["route"] == "global"
    assert row["position_error_m"] == pytest.approx(0.0, abs=1e-9)
    assert row["joint_limit_margin_rad"] == pytest.approx(3.14 - 0.8)
    assert row["joint_delta_rad"] == pytest.approx(0.02)
    assert dojo.environment.commands == [] and dojo.sim_steps == 0


def test_robodojo_models_the_route_each_interaction_selects(dojo):
    near, far = [0.02, 0.0, 0.8, *IDENTITY], [0.2, 0.0, 0.8, *IDENTITY]
    rows = _dojo(dojo, _query(near, "precision"), _query(far, "precision"),
                 _query(near, "contact_approach"), _query(far, "contact_approach"))
    assert [row["route"] for row in rows] == ["local", "global", "local", None]
    assert rows[3]["reason"] == "route_contact_approach_exceeds_local_bound"
    dojo.contact_approach_route = "global_precontact_then_local"
    [staged] = _dojo(dojo, _query(far, "contact_approach"))
    assert staged["reachable"] is True and staged["route"] == "global_precontact_then_local"
    assert staged["joint_delta_rad"] == pytest.approx(0.2)


def test_robodojo_chains_an_approach_onto_its_pregrasp_with_the_local_jump_limit(dojo):
    pregrasp, approach = [0.3, 0.0, 0.8, *IDENTITY], [0.3, 0.0, 0.76, *IDENTITY]
    rows = _dojo(dojo, _query(pregrasp), _query(approach, "contact_approach", 0))
    assert [row["route"] for row in rows] == ["global", "local"]
    assert rows[1]["reachable"] is True and rows[1]["joint_delta_rad"] == pytest.approx(0.04)
    # Unchained, the same approach is 0.3 m from the measured pose: not local.
    [alone] = _dojo(dojo, _query(approach, "contact_approach"))
    assert alone["reason"] == "route_contact_approach_exceeds_local_bound"
    dojo.local_joint_jump_limit_rad = 0.01
    rows = _dojo(dojo, _query(pregrasp), _query(approach, "contact_approach", 0))
    assert rows[1]["reason"] == "local_ik_branch_jump"
    dojo.environment.robot_manager.fail_ik = True
    rows = _dojo(dojo, _query(pregrasp), _query(approach, "contact_approach", 0))
    assert [row["reason"] for row in rows] == ["ik_failed", "start_unreachable"]


def test_robodojo_names_each_refusal_the_executor_would_hit(dojo, monkeypatch):
    from ora.environment.backends.robodojo import ik

    manager = dojo.environment.robot_manager
    original = manager.solve_ik

    def aliased(target_pose, env_idx, robot):
        result = original(target_pose, env_idx, robot)
        result["joint_value"][5] += 2 * math.pi
        return result

    manager.solve_ik = aliased
    assert _dojo(dojo, _query([0.02, 0.0, 0.8, *IDENTITY]))[0]["reason"] == "joint_limit"
    manager.solve_ik = original

    limit = dojo.global_joint_jump_limit_rad
    dojo.global_joint_jump_limit_rad = 0.001
    [jumped] = _dojo(dojo, _query([0.02, 0.0, 0.8, *IDENTITY]))
    assert jumped["reason"] == "ik_branch_jump" and jumped["joint_delta_rad"] == pytest.approx(0.02)
    dojo.global_joint_jump_limit_rad = limit

    monkeypatch.setattr(ik, "self_collision_model", lambda *a: SimpleNamespace(
        straight_path=lambda start, end: {"status": "blocked"}))
    [blocked] = _dojo(dojo, _query([0.02, 0.0, 0.8, *IDENTITY]))
    assert blocked["reason"] == "self_collision_path" and blocked["position_error_m"] is None
    assert dojo.environment.commands == []

    # The same executor screen now includes the other arm held at its measured joints.
    monkeypatch.setattr(ik, "self_collision_model", lambda *a: SimpleNamespace(
        straight_path=lambda start, end: {"status": "clear"}))
    monkeypatch.setattr(ik, "inter_arm_screen", lambda session: SimpleNamespace(
        straight_paths=lambda starts, ends: {"status": "inter_arm_collision", "min_clearance_m": -0.01}))
    [other_arm] = _dojo(dojo, _query([0.02, 0.0, 0.8, *IDENTITY]))
    assert other_arm["reason"] == "inter_arm_collision_path" and dojo.environment.commands == []


def test_an_unsettled_measured_frame_is_a_typed_answer_not_an_internal_fault(dojo, monkeypatch):
    from ora.environment.backends.robodojo import reach_check as dojo_reach

    monkeypatch.setattr(dojo_reach, "fk_calibrated", lambda *a: False)
    rows = _dojo(dojo, _query([0.02, 0.0, 0.8, *IDENTITY]), _query([0.03, 0.0, 0.8, *IDENTITY]))
    assert {row["reason"] for row in rows} == {"fk_calibration_mismatch"}


# --------------------------------------------------------------------------
# Policy edge, through the real dispatcher and result admission
# --------------------------------------------------------------------------

def _dojo_runtime(tmp_path, grasp_backend=None):
    import tests.environments.robodojo.test_robodojo_closed_skills_default as parity
    runtime = parity.tool_runtime(tmp_path)
    runtime.surface = resolve_tool_surface("robodojo_placement_closed_skills_v2")
    runtime.allowed_tools = tuple(item["name"] for item in runtime.surface.definitions)
    if grasp_backend is not None:
        runtime.grasp_backend = grasp_backend
    return runtime


def _local_x(orientation):
    x, y, z, w = orientation
    return [1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)]


def _local_y(orientation):
    x, y, z, w = orientation
    return [2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)]


def _stub_bridge(requests, *, reachable=lambda query: True, reason="ik_failed",
                 ik_model="robodojo_sdk_ik_endpoint_selection_v1", parity=True):
    def bridge(request):
        requests.append(copy.deepcopy(request))
        assert request["schema"] == REQUEST_SCHEMA
        rows = []
        for query in request["queries"]:
            start = query["from_index"]
            if start is not None and not rows[start]["reachable"]:
                rows.append({"reachable": False, "reason": "start_unreachable", "route": None,
                             "position_error_m": None, "orientation_error_rad": None,
                             "joint_limit_margin_rad": None, "joint_delta_rad": None})
                continue
            ok = reachable(query)
            rows.append({"reachable": ok, "reason": None if ok else reason,
                         "route": "global" if parity else None,
                         "position_error_m": 0.0012345678 if ok else None,
                         "orientation_error_rad": 0.0 if ok else None,
                         "joint_limit_margin_rad": 1.0 if ok else None,
                         "joint_delta_rad": 0.2 if ok else None})
        return {"schema": RESULT_SCHEMA, "arm": request["arm"], "ik_model": ik_model,
                "matches_move_pose_admission": parity, "motion_planner_called": False,
                "claim_boundary": "stub IK", "results": rows}
    return bridge


def test_the_policy_cannot_call_the_undeclared_tool(tmp_path):
    runtime = _dojo_runtime(tmp_path)
    runtime.align_plan_callback = _stub_bridge([])
    with pytest.raises(ToolError) as refused:
        runtime.invoke("reach_check", {"arm": "left", "poses": [
            {"position_m": [0.1, 0.0, 0.8], "orientation_xyzw": IDENTITY}]})
    assert refused.value.code == "tool_unavailable"


def test_the_check_answers_in_order_and_never_asks_ik_outside_the_workspace(tmp_path):
    runtime = _dojo_runtime(tmp_path)
    requests = []
    runtime.align_plan_callback = _stub_bridge(requests, reachable=lambda query: query["pose_xyzw"][0] < 0.3)
    outside = [runtime.workspace_bounds_m["x"][1] + 1.0, 0.0, 0.8]
    result = policy_reach.check_queries("right", [
        _query([0.1, 0.0, 0.8, *IDENTITY]),
        _query([*outside, *IDENTITY]),
        _query([0.35, 0.0, 0.8, *IDENTITY], "precision"),
        _query([0.1, 0.0, 0.76, *IDENTITY], "contact_approach", 1),
        _query([0.1, 0.0, 0.76, *IDENTITY], "contact_approach", 2),
    ], bridge=runtime.align_plan_callback, workspace_bounds_m=runtime.workspace_bounds_m)
    assert [row["index"] for row in result["results"]] == [0, 1, 2, 3, 4]
    assert result["reachable_indices"] == [0]
    assert result["matches_move_pose_admission"] is True
    assert result["results"][0]["position_error_m"] == 0.00123
    assert [row["reason"] for row in result["results"]] == [
        None, "outside_workspace", "ik_failed", "start_unreachable", "start_unreachable"]
    [sent] = requests
    assert len(sent["queries"]) == 3  # 0, 2 and 4; 1 is outside and 3 is chained onto it
    assert sent["queries"][1]["interaction"] == "precision"
    assert sent["queries"][2]["from_index"] == 1  # renumbered from 2
    assert not (runtime.control_root / "submitted-action.json").exists()


def test_a_run_without_the_ik_bridge_says_so(tmp_path):
    runtime = _dojo_runtime(tmp_path)
    runtime.align_plan_callback = None
    with pytest.raises(ToolError) as refused:
        policy_reach.check_queries("left", [_query([0.1, 0.0, 0.8, *IDENTITY])],
                                   bridge=runtime.align_plan_callback,
                                   workspace_bounds_m=runtime.workspace_bounds_m)
    assert refused.value.code == "tool_unavailable"


def test_a_backend_refusal_becomes_a_field_naming_error(tmp_path):
    runtime = _dojo_runtime(tmp_path)

    def refuse(request):
        raise EnvironmentRefusal("arm is not an arm of this embodiment", field="arm",
                                 received=request["arm"], allowed=["left"])

    runtime.align_plan_callback = refuse
    with pytest.raises(ToolError) as refused:
        policy_reach.check_queries("right", [_query([0.1, 0.0, 0.8, *IDENTITY])],
                                   bridge=runtime.align_plan_callback,
                                   workspace_bounds_m=runtime.workspace_bounds_m)
    assert refused.value.code == "invalid_arguments" and refused.value.field == "arm"


# --------------------------------------------------------------------------
# grasp_plan: check the whole pool before the cap, order, never drop
# --------------------------------------------------------------------------

def _tilted(angle_deg):
    radians = math.radians(angle_deg)
    return [math.sin(radians), 0.0, -math.cos(radians)]


#: (score, approach tilt from straight down in degrees, x offset)
POOL = [(0.9, 180.0, 0.0), (0.8, 40.0, 0.0), (0.7, 0.0, 0.0), (0.6, 20.0, 0.0), (0.5, 0.0, 0.1)]


class PoolProposer:
    """Five X5 candidates; honours the requested count and records it."""
    backend_id = "contact_graspnet_test_double"
    backend_version = "v1"

    def __init__(self):
        self.requested = []

    def configuration(self):
        return {"pregrasp_offset_m": 0.04}

    def propose(self, **kwargs):
        self.requested.append(kwargs["max_candidates"])
        candidates = []
        for index, (score, tilt, x) in enumerate(POOL):
            approach = _tilted(tilt)
            origin = [x, 0.0, 0.8]
            candidates.append({
                "candidate_id": f"grasp-pool{index:020d}",
                "source_index": index, "rank": index,
                "source_gripper_origin_xyz_m": origin,
                "source_pregrasp_origin_xyz_m": [o - a * 0.04 for o, a in zip(origin, approach)],
                "approach_world": approach, "jaw_world": [0.0, 1.0, 0.0],
                "gripper_opening_m": 0.05, "backend_score": score,
                "pose_convention": "contact_graspnet_panda_gripper_origin_opencv_v2",
                "checks": {"workspace": "passed", "aperture": "passed",
                           "ik": "not_checked", "collision": "not_checked"},
                "execution_ready": False,
            })
        return candidates[:kwargs["max_candidates"]], {"request_id": "grasp-pool", "rejected": []}


def _pool_ik(query):
    """Everything but the upward approach; the x=0.1 candidate only as its jaw flip."""
    orientation = query["pose_xyzw"][3:]
    if _local_x(orientation)[2] > 0.5:
        return False
    if abs(query["pose_xyzw"][0] - 0.1) < 0.05 and _local_y(orientation)[1] > 0:
        return False
    return True


def _pool_grasp(tmp_path, bridge, arm="left"):
    backend = PoolProposer()
    runtime = _dojo_runtime(tmp_path, grasp_backend=backend)
    runtime.align_plan_callback = bridge
    grounded = runtime.invoke("vision_ground", {"camera": "head", "text": "red mug", "max_candidates": 1})
    result = runtime.invoke("grasp_plan", {"target_ref": grounded["candidates"][0]["target_ref"], "arm": arm})
    return runtime, backend, result


def test_the_whole_pool_is_checked_then_reachable_and_near_vertical_come_first(tmp_path):
    requests = []
    runtime, backend, result = _pool_grasp(tmp_path, _stub_bridge(requests, reachable=_pool_ik))
    assert backend.requested == [policy_reach.POOL_LIMIT]
    ids = [f"grasp-pool{index:020d}" for index in range(len(POOL))]
    # Reachable within 30 degrees (2, 3, 4 in backend order), then reachable
    # beyond it (1), then unreachable (0); the cap keeps the first three.
    assert [item["candidate_id"] for item in result["candidates"]] == [ids[2], ids[3], ids[4]]
    assert [item["rank"] for item in result["candidates"]] == [0, 1, 2]
    by_id = {item["candidate_id"]: item["ik_reachability"] for item in result["candidates"]}
    assert by_id[ids[2]]["member"] == "exact" and by_id[ids[4]]["member"] == "jaw_flip"
    assert by_id[ids[3]]["approach_tilt_from_vertical_deg"] == pytest.approx(20.0, abs=0.1)
    assert by_id[ids[2]]["pregrasp"] == {"reachable": True, "reason": None}
    summary = result["ik_reachability"]
    assert summary["status"] == "annotated" and summary["matches_move_pose_admission"] is True
    assert summary["checked_candidate_count"] == 5 and summary["reachable_candidate_count"] == 4
    # Exact member for all five (one chunk of pregrasp/approach pairs), then --
    # because this surface declares grasp_plan_jaw_symmetry_headroom_v1 from
    # revision 9 -- the jaw flip for all five too. Without that capability the
    # flip pass covers only the two the exact member did not admit
    # (test_grasp_jaw_symmetry.test_the_released_selection_is_unchanged_without_a_rule).
    assert [len(item["queries"]) for item in requests] == [10, 10]
    assert all(item["arm"] == "left" for item in requests)
    pair = requests[0]["queries"][:2]
    assert [q["interaction"] for q in pair] == ["free_space", "contact_approach"]
    assert [q["from_index"] for q in pair] == [None, 0]
    manifest_path = next(runtime.artifact_root.glob("grasp-queries/*/manifest.json"))
    record = json.loads(manifest_path.read_text())["candidate_selection"]
    assert [item["candidate_id"] for item in record["pool"]] == [ids[2], ids[3], ids[4], ids[1], ids[0]]
    assert record["displayed_candidate_ids"] == [ids[2], ids[3], ids[4]]
    assert record["evidence"]["pool_size"] == 5 and record["evidence"]["elapsed_s"] >= 0
    compiled = json.loads(manifest_path.read_text())["compiled_candidates"]
    assert [item["candidate_id"] for item in compiled] == [ids[2], ids[3], ids[4]]
    assert not (runtime.control_root / "submitted-action.json").exists()


def test_an_ik_failure_keeps_the_backend_order_and_the_cap(tmp_path):
    def broken(request):
        raise RuntimeError("worker died")

    _runtime, _backend, result = _pool_grasp(tmp_path, broken)
    ids = [f"grasp-pool{index:020d}" for index in range(3)]
    assert [item["candidate_id"] for item in result["candidates"]] == ids
    assert result["ik_reachability"]["status"] == "unavailable"
    assert result["ik_reachability"]["checked_candidate_count"] == 0
    assert all(item["ik_reachability"]["reachable"] is None for item in result["candidates"])


def test_a_surface_without_the_capability_keeps_the_released_cap(tmp_path):
    from dataclasses import replace

    backend = PoolProposer()
    runtime = _dojo_runtime(tmp_path, grasp_backend=backend)
    runtime.surface = replace(runtime.surface, capabilities=tuple(
        item for item in runtime.surface.capabilities if item != policy_reach.GRASP_ANNOTATION_CAPABILITY))
    assert policy_reach.grasp_candidate_selection(runtime, "left") is None
    result = {"candidates": []}
    policy_reach.publish_grasp_selection(runtime, result, {})
    assert "ik_reachability" not in result


def test_robotwin_grasp_plan_carries_the_selection_through_result_admission(tmp_path):
    from ora.agent.tools import tool_names
    from tests.agent.perception.test_grounding_grasp_runtime import _runtime, _select_correctness_surface

    runtime, _ = _runtime(tmp_path)
    _select_correctness_surface(runtime, "robotwin_toolchain_b2_closed_noservo_v2")
    control, root, observation, pose, gripper = runtime._current_grip_state()
    observation["cameras"] = {"head_camera": observation["cameras"]["agentview"]}
    runtime._current_grip_state = lambda: (control, root, observation, pose, gripper)
    requests = []
    runtime.align_plan_callback = _stub_bridge(
        requests, ik_model="robotwin_single_seed_damped_least_squares_v1", parity=False)
    target = runtime.invoke(tool_names.as_advertised(runtime.allowed_tools, "vision_ground"),
                            {"camera": "head_camera", "text": "red object"})["candidates"][0]["target_ref"]
    result = runtime.invoke("grasp_plan", {"arm": "right", "target_ref": target})
    assert result["candidates"]
    assert all(item["ik_reachability"]["reachable"] is True for item in result["candidates"])
    assert result["ik_reachability"]["matches_move_pose_admission"] is False
    assert requests and requests[0]["arm"] == "right"


def test_the_jaw_flip_is_the_precommit_s_rotation_about_local_x():
    from ora.environment.backends.robodojo.sparse_motion import (
        JAW_SYMMETRY_AXIS, rotate_about_local_axis,
    )
    from ora.agent.tools.alignment import frame_local_axes

    orientation = [0.1, 0.2, 0.3, math.sqrt(1 - 0.14)]
    mine = policy_reach._member(orientation, "jaw_flip", frame_local_axes("arx_x5_link6")["approach"])
    theirs = rotate_about_local_axis(orientation, JAW_SYMMETRY_AXIS, math.pi)
    assert abs(float(np.dot(mine, theirs))) == pytest.approx(1.0)
