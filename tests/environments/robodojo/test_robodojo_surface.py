"""RoboDojo/ARX X5 ToolSurface, observe gate and workspace profile."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from tests.environments.robodojo.test_robodojo_backend import fake_sdk_candidate_boundary  # noqa: F401

from ora.agent.composition import tool_surface as surface_module
from ora.agent.composition.tool_surface import (
    ToolSurfaceError, resolve_tool_surface, validate_schema_instance,
)
from ora.agent.tools.runtime import core as runtime_core
from ora.agent.tools.runtime.core import ManipulationToolRuntime, ToolError
from ora.environment.workspace_profiles import (
    ROBODOJO_ARX_X5_TABLETOP_V1, resolve_workspace_profile,
)

import tests.environments.robodojo.test_robodojo_backend as backend
import tests.environments.robodojo.test_robodojo_perception as perception
ROOT = Path(__file__).resolve().parents[3]
MANUAL = ROOT / "tests/data/prompts/tool-schema-delivery-v1.md"
ROBODOJO_CAMERAS = ["head", "left_wrist", "right_wrist"]
ROBOTWIN_CAMERAS = ["head_camera", "front_camera", "left_camera", "right_camera"]
# Tools whose definition and result contract must stay byte-identical with
# the RoboTwin default surface: they carry no embodiment or camera fact.
SHARED_TOOLS = ("compute", "report_issue", "finish_episode")
BANNED_TERMS = (
    "RoboTwin", "ALOHA", "aloha", "observer_camera", "head_camera",
    "front_camera", "left_camera", "right_camera", "grip_center", "TCP",
    "CuRobo",
)


ROBODOJO = "robodojo_placement_closed_skills_v2"
ROBOTWIN = "robotwin_toolchain_b2_closed_noservo_v2"


def _robodojo():
    return resolve_tool_surface(ROBODOJO)


def _robotwin():
    return resolve_tool_surface(ROBOTWIN)


def _definition(surface, name):
    return next(item for item in surface.definitions if item["name"] == name)


def test_robodojo_surface_mirrors_the_robotwin_topology_with_x5_facts() -> None:
    robodojo, robotwin = _robodojo(), _robotwin()
    assert [d["name"] for d in robodojo.definitions] == [
        d["name"] for d in robotwin.definitions
    ]
    assert len(robodojo.definitions) == 19
    assert robodojo.mutation_tools == robotwin.mutation_tools
    assert robodojo.lifecycle_tools == robotwin.lifecycle_tools
    assert robodojo.observation_tools == robotwin.observation_tools
    assert robodojo.call_effect_profile == robotwin.call_effect_profile
    assert robodojo.runtime_id == "native_atomic_causal"
    assert robodojo.supports("policy_tool_help_v1")
    for capability in (
        "robodojo_rgbd_v1", "contact_graspnet_robodojo_x5_link6_compiler_v1",
        "dual_arm_grasp_plan_v1", "dual_arm_sparse_targets_v1",
        "motion_action_v3", "automatic_current_auxiliary_contact_sheet_v1",
        "robodojo_sdk_ik_joint_interpolation_v1_non_parity",
    ):
        assert robodojo.supports(capability), capability
    for capability in (
        "robotwin_rgbd_v1", "policy_observer_camera_v1",
        "contact_graspnet_robotwin_aloha_grip_center_compiler_v3",
        "robotwin_aloha_public_rgbd_topdown_grasp_v1",
        "topk_ik_joint_planning_v1", "analytic_tcp_local_cartesian_v1",
    ):
        assert not robodojo.supports(capability), capability
    assert robodojo.view_config["default_observe_views"] == ROBODOJO_CAMERAS
    assert robodojo.view_config["automatic_actual_views"] == ROBODOJO_CAMERAS
    assert robodojo.view_config["controlled_frame_convention"] == "arx_x5_link6"
    assert robodojo.view_config["public_geometry_descriptor"]["path"].startswith(
        "ora/policy/embodiments/arx-x5/")


def test_shared_tools_stay_byte_identical_and_x5_trims_replace_every_robotwin_fact() -> None:
    robodojo, robotwin = _robodojo(), _robotwin()
    for name in SHARED_TOOLS:
        assert _definition(robodojo, name) == _definition(robotwin, name), name
        assert robodojo.result_schemas.get(name) == robotwin.result_schemas.get(name)
    for definition in robodojo.definitions:
        text = definition["description"] + json.dumps(definition["inputSchema"])
        text += json.dumps(robodojo.result_schemas.get(definition["name"]))
        hits = [term for term in BANNED_TERMS if term in text]
        assert not hits, (definition["name"], hits)
    observe = _definition(robodojo, "observe")["inputSchema"]
    assert observe["properties"]["views"]["items"]["enum"] == ROBODOJO_CAMERAS
    assert _definition(robodojo, "mark_point")["inputSchema"]["properties"]["view"]["enum"] == ROBODOJO_CAMERAS
    for name in ("vision_segment", "vision_ground"):
        assert _definition(robodojo, name)["inputSchema"]["properties"]["camera"]["enum"] == ROBODOJO_CAMERAS
        assert robodojo.result_schemas[name]["properties"]["camera"]["enum"] == ROBODOJO_CAMERAS
    move_pose = _definition(robodojo, "move_pose")["description"]
    # The v1 route reports an aliased IK branch as plan_rejected, and keeps
    # planner_failed for the SDK solve itself failing.
    rejected, failed = (
        move_pose.split("plan_rejected", 1)[1].split("not_reached", 1)[0],
        move_pose.split("planner_failed", 1)[1].split("plan_rejected", 1)[0],
    )
    assert "ik_branch_jump" in rejected and "local_ik_branch_jump" in rejected
    assert "branch" not in failed
    assert "link6" in _definition(robodojo, "grasp_plan")["description"]


def _x5_grasp_result(tmp_path: Path) -> dict[str, object]:
    """The public grasp_plan result the live RoboDojo runtime admits for the X5."""
    import tests.environments.robodojo.test_robodojo_closed_skills_default as parity
    from tests.agent.motion.test_reach_check import _stub_bridge

    runtime = parity.tool_runtime(tmp_path)
    runtime.surface = _robodojo()
    runtime.allowed_tools = tuple(item["name"] for item in runtime.surface.definitions)
    runtime.align_plan_callback = _stub_bridge([])
    grounded = runtime.invoke("vision_ground", {"camera": "head", "text": "red mug", "max_candidates": 1})
    result = runtime.invoke("grasp_plan", {"target_ref": grounded["candidates"][0]["target_ref"], "arm": "left"})
    # The delivered images ride beside the admitted result, outside its contract.
    result.pop("images_base64", None)
    return result


def test_x5_grasp_result_passes_the_robodojo_override_and_fails_robotwin(tmp_path: Path) -> None:
    from ora.agent.tools.grasp import ALOHA_NOMINAL_MAXIMUM_OPENING_M

    result = _x5_grasp_result(tmp_path)
    assert result["camera"] == "head" and result["arm"] == "left"
    assert result["compiler"]["embodiment_descriptor_id"] == "arx_x5_robodojo_v1"
    assert result["compiler"]["physical_jaw_aperture_calibrated"] is False
    assert result["compiler"]["aperture_caveat"] == perception.ARX_X5_APERTURE_CAVEAT
    assert result["candidates"][0]["compiler_checks"]["aperture_calibration"] == (
        perception.ARX_X5_APERTURE_CAVEAT
    )
    validate_schema_instance(result, _robodojo().result_schemas["grasp_plan"])
    with pytest.raises(ToolSurfaceError):
        validate_schema_instance(result, _robotwin().result_schemas["grasp_plan"])
    # The declared opening bound is the 80 mm X5 admission limit, not ALOHA's
    # nominal maximum; the compiler enforces it (test_robodojo_perception), the
    # result validator only checks types, enums, consts and required fields.
    opening = _robodojo().result_schemas["grasp_plan"]["$defs"]["candidate"][
        "properties"]["gripper_opening_m"]
    assert opening["maximum"] == 0.08
    assert _robotwin().result_schemas["grasp_plan"]["$defs"]["candidate"][
        "properties"]["gripper_opening_m"]["maximum"] == ALOHA_NOMINAL_MAXIMUM_OPENING_M
    compiler = _robodojo().result_schemas["grasp_plan"]["properties"]["compiler"]
    assert compiler["properties"]["controlled_point_semantics"]["const"] == "robodojo_link6_pose"
    assert "physical_jaw_aperture_calibrated" in compiler["required"]


def test_x5_command_target_is_published_once_and_read_by_every_consumer(
    tmp_path: Path,
) -> None:
    """One published category serves observe and the Native causal actual."""
    from types import SimpleNamespace

    from ora.agent.native.session import NativePlanner

    session = backend.sparse_session(tmp_path / "backend")
    observations = tmp_path / "backend" / "observations"
    project = ManipulationToolRuntime._dual_arm_gripper_target
    payload, paths = session.observe(observations)
    effectors = payload["proprioception"]["end_effectors"]
    # The session is the publisher; the tool layer only reads what it stated,
    # for either backend, and never re-derives a category of its own.
    for arm in ("left", "right"):
        assert effectors[arm]["gripper_command_target"] == "open"
        assert project(effectors[arm]) == "open"
    assert project({"gripper_command_target": "close"}) == "close"
    assert project({"gripper_command": 0.0}) == "unknown"

    session.execute(backend.v3(
        session, tmp_path / "backend", kind="set_gripper",
        targets={"left": "close"},
    ))
    effectors = session.observe(observations)[0][
        "proprioception"]["end_effectors"]
    assert project(effectors["left"]) == "close"
    assert project(effectors["right"]) == "open"


def test_native_dual_arm_causal_accepts_a_robodojo_proprioception(
    tmp_path: Path,
) -> None:
    """The join the first live RoboDojo K3 launch failed before any model call.

    `dual_arm_sparse_targets_v1` requires a categorical command target for both
    arms in the causal actual observation; the RoboDojo session publishes it,
    so the Native check reads it rather than raising on the first observation.
    """
    from types import SimpleNamespace

    from ora.agent.native.session import NativePlanner

    session = backend.sparse_session(tmp_path / "backend")
    session.execute(backend.v3(
        session, tmp_path / "backend", kind="set_gripper",
        targets={"left": "close"},
    ))
    payload, _paths = session.observe(tmp_path / "backend" / "observations")
    effectors = payload["proprioception"]["end_effectors"]

    planner = object.__new__(NativePlanner)
    planner.run_root = tmp_path / "run"
    planner.tools = SimpleNamespace(surface=_robodojo())
    # The live surface's camera and public-fact delivery is covered by the
    # current-visual tests; this is the end-effector join it ends in.
    actual = planner._actual_end_effector_snapshot({
        "policy_state": {"step": 0, "path": "policy-state.json"},
        "proprioception": payload["proprioception"],
    }, [], {})
    assert actual["end_effectors"] == {
        arm: {
            "pose_xyz_xyzw": [round(float(value), 6)
                              for value in effectors[arm]["pose_xyz_xyzw"]],
            "gripper_command_target": "close" if arm == "left" else "open",
        }
        for arm in ("left", "right")
    }


def _runtime(tmp_path: Path, profile: str) -> ManipulationToolRuntime:
    control = tmp_path / "control"
    control.mkdir(exist_ok=True)
    return ManipulationToolRuntime(
        control_root=control, manual_path=MANUAL,
        event_path=tmp_path / "events.jsonl",
        workspace_bounds_m=perception.BOUNDS, state_root=None,
        tool_surface_profile=profile, controlled_end_effector="right",
    )


def _publish_robodojo_observation(tmp_path: Path) -> str:
    from ora.contracts.robot_geometry import GEOMETRY_VERIFICATION_SCHEMA

    root, observation = perception._synthetic_robodojo_observation(
        tmp_path / "observations",
    )
    manifest = root / "observation.json"
    # The live surface delivers public robot facts only for geometry verified
    # against its descriptor.
    observation["robot_geometry_verification"] = {
        "schema": GEOMETRY_VERIFICATION_SCHEMA, "scope": "loaded_robot_geometry_only", "passed": True,
        "descriptor_sha256": _robodojo().view_config["public_geometry_descriptor"]["sha256"],
    }
    manifest.write_text(json.dumps(observation, sort_keys=True, indent=2) + "\n")
    control = tmp_path / "control"
    control.mkdir(exist_ok=True)
    (control / "current-observation.json").write_text(json.dumps({
        "observation_id": observation["observation_id"],
        "observation_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "source_root": str(root), "generation": 0, "open": True,
    }), encoding="utf-8")
    return str(observation["observation_id"])


def test_observe_serves_robodojo_cameras_only_under_its_capability(
    tmp_path: Path, ready_model_services: None,
) -> None:
    observation_id = _publish_robodojo_observation(tmp_path)
    runtime = _runtime(tmp_path, ROBODOJO)
    result = runtime.invoke("observe", {})
    assert result["observation_id"] == observation_id
    assert result["views"] == ROBODOJO_CAMERAS
    assert len(result["images_base64"]) == 3
    assert result["view_dimensions"] == {
        name: {"width": 64, "height": 48} for name in ROBODOJO_CAMERAS
    }
    effectors = result["public_robot_facts"]["end_effectors"]
    assert set(effectors) == {"left", "right"}
    # The published RoboDojo command is the executed normalized directive, so
    # the public read is the category it stands for, never "unknown".
    for arm in ("left", "right"):
        assert effectors[arm]["gripper"]["command_target"] == "open"
    wrist = runtime.invoke("observe", {"views": ["right_wrist"]})
    assert wrist["views"] == ["right_wrist"] and len(wrist["images_base64"]) == 1
    with pytest.raises(ToolError) as rejected:
        runtime.invoke("observe", {"views": ["head_camera"]})
    assert rejected.value.as_dict()["constraints"] == {"enum": ROBODOJO_CAMERAS, "violated": "enum"}
    with pytest.raises(ToolError, match="items must be unique"):
        runtime.invoke("observe", {"views": ["head", "head"]})

    robotwin = _runtime(tmp_path, ROBOTWIN)
    with pytest.raises(ToolError) as unknown:
        robotwin.invoke("observe", {"views": ["head"]})
    assert unknown.value.as_dict()["constraints"] == {"enum": ROBOTWIN_CAMERAS, "violated": "enum"}
    assert "outside the input enum" in unknown.value.as_dict()["message"]


def test_observe_without_a_dual_arm_rgbd_capability_keeps_the_libero_views() -> None:
    libero = resolve_tool_surface("libero_placement_direct_overhead_v1")
    assert not libero.supports("robotwin_rgbd_v1")
    assert not libero.supports(runtime_core.ROBODOJO_RGBD_CAPABILITY)
    views = _definition(libero, "observe")["inputSchema"]["properties"]["views"]
    assert "agentview" in json.dumps(views) and "head" not in views["items"].get("enum", [])


def test_robodojo_workspace_profile_covers_the_table_and_reset_poses() -> None:
    profile = resolve_workspace_profile(ROBODOJO_ARX_X5_TABLETOP_V1)
    assert profile["environment_id"] == "robodojo"
    assert profile["embodiment_id"] == "arx_x5"
    assert profile["frame"] == "robodojo_episode_local"
    assert profile["semantics"] == "hard_action_envelope_not_reachability"
    assert profile["provenance"]["backend_revision"] == (
        "ee67a1468510da7624a089164402359f2afc72c8"
    )
    bounds = profile["bounds_m"]
    # Table top: 1.4 x 1.1 m centred at (0, -0.05), surface z = 0.765.
    for x, y in ((-0.7, -0.6), (0.7, 0.5)):
        assert bounds["x"][0] <= x <= bounds["x"][1]
        assert bounds["y"][0] <= y <= bounds["y"][1]
    assert bounds["z"][0] < 0.765 < bounds["z"][1]
    for x, y, z in ((-0.3, -0.352, 0.922), (0.3, -0.352, 0.922)):
        assert bounds["x"][0] <= x <= bounds["x"][1]
        assert bounds["y"][0] <= y <= bounds["y"][1]
        assert bounds["z"][0] <= z <= bounds["z"][1]
    assert profile["profile_sha256"] == resolve_workspace_profile(
        ROBODOJO_ARX_X5_TABLETOP_V1,
    )["profile_sha256"]
    with pytest.raises(ValueError, match="unknown action workspace profile"):
        resolve_workspace_profile("robodojo_unknown_profile")


def test_sheet_camera_counts_follow_the_projection_layouts(monkeypatch) -> None:
    from ora.agent.native import observation_projection

    base = {
        "schema": "ora.policy-view-config.v1",
        "rgb_resolution": "native", "derived_resolution": [256, 256],
        "default_observe_views": ["head"],
        "coordinate_convention": "opencv_top_left_xy",
        "automatic_actual_views": ["head", "auxiliary_contact_sheet"],
    }
    capabilities = ["automatic_current_auxiliary_contact_sheet_v1"]
    two = {"view_config": {**base, "auxiliary_contact_sheet_cameras": ["a", "b"]},
           "capabilities": capabilities}
    with pytest.raises(ToolSurfaceError, match="admitted counts: 3, 4"):
        surface_module._resolve_view_config(two)
    monkeypatch.setitem(
        observation_projection.AUXILIARY_CONTACT_SHEET_LAYOUTS, 2,
        ("one_by_two_test_layout", 2, 1),
    )
    assert surface_module._resolve_view_config(two)["auxiliary_contact_sheet_cameras"] == ["a", "b"]
