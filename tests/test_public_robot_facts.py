"""Public-data production fixtures; independent CI auditing lives elsewhere."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import base64
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np

from ora.agent.composition import resolve_tool_surface
from ora.agent.context.action_views import materialize_post_action_observation, resolve_post_action_images
from ora.agent.context.public_facts import (
    PublicFactsError, project_current_public_facts, public_end_effectors, world_axes,
)
from ora.agent.context.projection import causal_provider_value, _causal_projection_violations
from observation_fixtures import observation_fixture as _observation


ROOT = Path(__file__).resolve().parents[1]


#: A public-facts composition without complete current visual delivery. Its
#: registered example (``*_placement_local_vla_v3``) left the tree with VLA
#: support, and so did the explicit-motion v3 base it was derived from; the
#: maintained default of each backend now carries the capability set, minus
#: complete current visual delivery, beside the former example's view
#: configuration, so both producers stay covered.
_PUBLIC_FACTS_ONLY = {
    "libero": dict(
        base="libero_placement_direct_overhead_v1",
        dropped={"full_current_visual_v1"},
        added=set(),
        view_config={"automatic_actual_views": ["agentview", "wrist"], "coordinate_convention": "opencv_top_left_xy",
                     "default_observe_views": ["agentview", "wrist"], "derived_resolution": [256, 256],
                     "rgb_resolution": "native", "schema": "ora.policy-view-config.v1",
                     "current_observation_delivery": "all_public_v1", "controlled_frame_convention": "panda_grip_site",
                     "public_geometry_descriptor": "ora/policy/embodiments/franka-panda/descriptor-public-geometry-v8-controller-feedback-v1-session-split-v1.json"}),
    "robotwin": dict(
        base="robotwin_toolchain_b2_closed_noservo_v2",
        dropped={"full_current_visual_v1"},
        added={"policy_observer_camera_v1"},
        view_config={"automatic_actual_views": ["observer_camera", "auxiliary_contact_sheet"],
                     "coordinate_convention": "opencv_top_left_xy", "default_observe_views": ["observer_camera"],
                     "derived_resolution": [256, 256], "rgb_resolution": "native",
                     "schema": "ora.policy-view-config.v1", "current_observation_delivery": "all_public_v1",
                     "controlled_frame_convention": "aloha_grip_center_tcp",
                     "auxiliary_contact_sheet_cameras": ["head_camera", "front_camera", "left_camera", "right_camera"],
                     "public_geometry_descriptor": "ora/policy/embodiments/aloha-agilex/descriptor-public-geometry-v5-session-split-v1.json"}),
}


def _public_facts_surface(backend):
    import dataclasses

    spec = _PUBLIC_FACTS_ONLY[backend]
    base = resolve_tool_surface(spec["base"])
    view = dict(spec["view_config"])
    descriptor = view["public_geometry_descriptor"]
    view["public_geometry_descriptor"] = {
        "path": descriptor, "sha256": hashlib.sha256((ROOT / descriptor).read_bytes()).hexdigest()}
    capabilities = tuple(sorted((set(base.capabilities) - spec["dropped"]) | spec["added"]))
    results = copy.deepcopy(dict(base.result_schemas))
    if "observer_camera" in view["automatic_actual_views"]:
        cameras = results["observe"]["properties"]["public_robot_facts"]["properties"]["cameras"]
        cameras["properties"]["observer_camera"] = {"$ref": "#/$defs/camera"}
    return dataclasses.replace(base, capabilities=capabilities, view_config=view, result_schemas=results)


def fixture(root, backend="libero", sequence=1):
    surface = (resolve_tool_surface("robodojo_placement_closed_skills_v2") if backend == "robodojo"
               else _public_facts_surface(backend))
    cameras = (["head", "left_wrist", "right_wrist"] if backend == "robodojo" else ["agentview", "wrist"] if backend == "libero" else
               ["observer_camera", "head_camera", "front_camera", "left_camera", "right_camera"])
    observation, source = _observation(root, cameras, sequence=sequence)
    if backend == "libero":
        effectors = {"right": {"pose_xyz_xyzw": [0, 0, 1, 0, 0, 0, 1],
            "pose_frame": "robosuite_grip_site", "gripper": .5,
            "gripper_finger_qpos": [.012, -.028]}}
    else:
        effectors = {arm: {"pose_xyz_xyzw": [0, 0, 1, 0, 0, 0, 1],
            "controlled_point_semantics": "parallel_jaw_grip_center_tcp", "gripper_command_target": "close",
            "gripper_state": {"commanded_opening_normalized": 0., "drive_target_opening_normalized": 0.,
                "measured_opening_normalized": .18181818, "measured_primary_joint_position": 0.}}
            for arm in ("left", "right")}
    if backend == "robodojo":
        from ora.contracts.gripper_feedback import gripper_feedback
        from ora.environment.controllers.robodojo_arx import public_feedback_contract
        effectors = {arm: {"pose_xyz_xyzw": [0, 0, 1, 0, 0, 0, 1],
            "pose_frame": "robodojo_episode_local_ee_link", "gripper": 1.,
            "gripper_finger_qpos": [.044, .0345], "gripper_command": .8,
            "gripper_command_target": "close", "gripper_feedback": gripper_feedback(
                requested_opening_normalized=0., drive_target_opening_normalized=.8,
                measured_opening_normalized=1., effective_target_opening_normalized=.01/.054,
                tolerance_normalized=.03, drive_target_source="resolved_sdk_joint_position_target",
                measurement_source="primary_finger_encoder", pose_hold_semantics="preserve_last_applied_drive_target")}
            for arm in ("left", "right")}
        observation["controller_feedback"] = public_feedback_contract()
    observation["proprioception"] = {"end_effectors": effectors}
    from ora.contracts.robot_geometry import GEOMETRY_VERIFICATION_SCHEMA
    observation["robot_geometry_verification"] = {
        "schema": GEOMETRY_VERIFICATION_SCHEMA,
        "descriptor_sha256": surface.view_config["public_geometry_descriptor"]["sha256"],
        "scope": "loaded_robot_geometry_only", "passed": True,
    }
    from ora.agent.evidence.policy_state import REQUIRED_ABSENT
    observation["forbidden_modalities_absent"] = sorted(REQUIRED_ABSENT)
    observation["main_camera"] = cameras[0]
    for name in cameras:
        calibration = {"intrinsic_cv": [[10, 0, 6], [0, 10, 5], [0, 0, 1]],
            "extrinsic_cv": [[1, 0, 0, sequence], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            "cam2world_cv": [[1, 0, 0, -sequence], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            "source_camera": "private_adapter_name", "object_gt": {"must_not_escape": True}}
        path = source / name / "calibration.json"
        path.write_text(json.dumps(calibration))
        observation["cameras"][name]["files"]["calibration"] = {
            "path": name + "/calibration.json", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        depth = source / name / "depth.npy"
        np.save(depth, np.ones((10, 12), dtype=np.float32))
        observation["cameras"][name]["files"]["depth_numpy_m_f32"] = {
            "path": name + "/depth.npy", "sha256": hashlib.sha256(depth.read_bytes()).hexdigest()}
    (source / "observation.json").write_text(json.dumps(observation))
    return surface, observation, source


@pytest.mark.parametrize("backend", ["libero", "robotwin"])
def test_source_numbers_and_privacy_survive_public_projection(tmp_path, backend):
    surface, observation, source = fixture(tmp_path, backend)
    value = project_current_public_facts(observation=observation, observation_root=source, view_config=surface.view_config)
    assert set(value["cameras"]) == set(observation["cameras"])
    assert value["observation_id"] == observation["observation_id"]
    for camera in value["cameras"].values():
        assert camera["calibration"]["intrinsic_cv"] == [[10, 0, 6], [0, 10, 5], [0, 0, 1]]
        assert camera["calibration"]["world_to_camera_cv"][0][3] == 1
    serialized = json.dumps(value)
    for private in ("private_adapter_name", "object_gt", "sha256", '"path"'):
        assert private not in serialized
    assert not _causal_projection_violations(causal_provider_value(value))
    grip = value["end_effectors"]["right"]["gripper"]
    if backend == "libero":
        assert grip["measured_finger_displacements_m"] == [.012, -.028]
        assert grip["measured_aperture_mm"] == 40
    else:
        assert grip["commanded_opening_normalized"] == 0
        assert grip["measured_opening_normalized"] == .18181818
        assert grip["grasp_evidence"] == "not_established_by_joint_measurement"


def test_panda_publishes_the_commanded_gripper_beside_the_measured_one(tmp_path):
    """A measurement without its command cannot answer "did my close take?".

    RoboTwin and ARX both publish the commanded target beside the encoder
    reading; Panda published only the reading, so a LIBERO policy comparing
    the two had one of the two numbers.
    """

    surface, observation, source = fixture(tmp_path, "libero")
    observation["proprioception"]["end_effectors"]["right"].update({
        "gripper_command": 0.0, "gripper_command_target": "close",
    })
    (source / "observation.json").write_text(json.dumps(observation))
    value = project_current_public_facts(
        observation=observation, observation_root=source,
        view_config=surface.view_config, gripper_command_delivered=True,
    )
    grip = value["end_effectors"]["right"]["gripper"]
    assert grip["commanded_opening_normalized"] == 0.0
    assert grip["command_target"] == "close"
    # The measurement stays exactly what it was; the command is beside it,
    # never instead of it.
    assert grip["measured_aperture_mm"] == 40
    assert grip["measured_opening_normalized"] == .5


def test_panda_command_target_stays_absent_rather_than_guessed(tmp_path):
    """Before any gripper command, the target is unknown, not "open"."""

    surface, observation, source = fixture(tmp_path, "libero")
    value = project_current_public_facts(
        observation=observation, observation_root=source,
        view_config=surface.view_config, gripper_command_delivered=True,
    )
    grip = value["end_effectors"]["right"]["gripper"]
    assert grip["commanded_opening_normalized"] is None
    assert grip["command_target"] is None


def test_panda_gripper_command_stays_out_of_surfaces_that_never_declared_it(tmp_path):
    """A frozen surface's closed result schema must see the shape it pinned."""

    surface, observation, source = fixture(tmp_path, "libero")
    value = project_current_public_facts(
        observation=observation, observation_root=source,
        view_config=surface.view_config,
    )
    assert set(value["end_effectors"]["right"]["gripper"]) == {
        "measured_finger_displacements_m", "measured_aperture_mm",
        "measured_opening_normalized", "measurement_semantics",
        "grasp_evidence",
    }


def test_axes_are_controlled_frame_specific_not_robot_guesses():
    # Independent quarter-turn about world Z: Panda closing +X becomes +Y;
    # ALOHA approach +X becomes +Y and its jaw line +Y becomes -X.
    pose = [0, 0, 1, 0, 0, math.sqrt(.5), math.sqrt(.5)]
    assert world_axes(pose, "panda_grip_site")["closing_world"] == pytest.approx([0, 1, 0])
    assert world_axes(pose, "panda_grip_site")["approach_world"] == pytest.approx([0, 0, 1])
    assert world_axes(pose, "aloha_grip_center_tcp")["approach_world"] == pytest.approx([0, 1, 0])
    assert world_axes(pose, "aloha_grip_center_tcp")["closing_world"] == pytest.approx([-1, 0, 0])


def test_sheet_coordinates_and_observation_bound_extrinsics(tmp_path):
    surface, observation, source = fixture(tmp_path, "robotwin")
    value = project_current_public_facts(observation=observation, observation_root=source, view_config=surface.view_config)
    assert value["cameras"]["head_camera"]["native_pixel_to_image_offset_xy"] == [0, 24]
    assert value["cameras"]["front_camera"]["native_pixel_to_image_offset_xy"] == [12, 24]
    assert value["cameras"]["left_camera"]["native_pixel_to_image_offset_xy"] == [0, 58]
    assert value["cameras"]["right_camera"]["native_pixel_to_image_offset_xy"] == [12, 58]
    _, later, later_source = fixture(tmp_path, "robotwin", 2)
    second = project_current_public_facts(observation=later, observation_root=later_source, view_config=surface.view_config)
    assert value["cameras"]["left_camera"]["calibration"]["world_to_camera_cv"][0][3] == 1
    assert second["cameras"]["left_camera"]["calibration"]["world_to_camera_cv"][0][3] == 2


@pytest.mark.parametrize("action", [None, "move_pose", "close_gripper"])
@pytest.mark.parametrize("backend", ["libero", "robotwin"])
def test_initial_and_every_action_deliver_one_complete_current_set(tmp_path, backend, action):
    surface, observation, source = fixture(tmp_path, backend)
    previous = None if action is None else {"schema": "other",
        "action": action, "observation_id": "observation-0", "action_id": "a0",
        "arguments": {"mode": "delta", "targets": {"right": {}}}}
    kwargs = dict(run_root=tmp_path, workspace=tmp_path / "workspace", observation=observation,
        observation_root=source, policy_state={"step": 1, "state_sha256": "state-hash"}, previous_action=previous,
        view_config=surface.view_config, camera_roles={"left_camera": "left_wrist", "right_camera": "right_wrist", "wrist": "right_wrist"})
    first = materialize_post_action_observation(**kwargs)
    second = materialize_post_action_observation(**kwargs)
    assert first == second
    assert len(first["images"]) == 2
    assert first["status"] == "complete"
    assert not first["missing_wrist_views"]
    images = resolve_post_action_images({"post_action_observation": first}, root=tmp_path, path_key="run_path")
    assert [image["camera"] for image in images] == surface.view_config["automatic_actual_views"]


def test_missing_camera_and_changed_calibration_fail_closed(tmp_path):
    surface, observation, source = fixture(tmp_path)
    config = dict(surface.view_config, automatic_actual_views=["agentview"])
    with pytest.raises(PublicFactsError, match="each physical camera"):
        project_current_public_facts(observation=observation, observation_root=source, view_config=config)
    (source / "wrist/calibration.json").write_text("{}")
    with pytest.raises(PublicFactsError, match="hash mismatch"):
        project_current_public_facts(observation=observation, observation_root=source, view_config=surface.view_config)


def test_frame_mismatch_and_missing_numbers_do_not_become_zero(tmp_path):
    surface, observation, _ = fixture(tmp_path)
    state = copy.deepcopy(observation["proprioception"])
    state["end_effectors"]["right"]["pose_frame"] = "unbound_frame"
    with pytest.raises(PublicFactsError, match="controlled point"):
        public_end_effectors(state, "panda_grip_site")
    del state["end_effectors"]["right"]["gripper_finger_qpos"]
    state["end_effectors"]["right"]["pose_frame"] = "robosuite_grip_site"
    with pytest.raises(PublicFactsError, match="both measured"):
        public_end_effectors(state, "panda_grip_site")


def published_fixture(tmp_path, backend):
    from ora.agent.evidence.policy_state import PolicyStateStore
    from ora.agent.context.codex import ContextProjector
    from test_native_causal_runtime import PROMPT, MANUAL, PROJECT

    surface, observation, source = fixture(tmp_path, backend)
    store = PolicyStateStore(tmp_path, "public-data", "plan")
    budgets = {"model_calls": 11, "robot_decisions": 7, "wall_time_s": 179.5}
    store.publish(observation=observation, observation_root=source, cause="reset",
        task_instruction="Inspect public robot evidence.", remaining_budget=budgets, previous_public_action=None)
    projector = ContextProjector(tmp_path / "workspace", tmp_path / "control", PROMPT, MANUAL, PROJECT,
        main_camera=observation["main_camera"], run_root=tmp_path, view_config=surface.view_config)
    pointer = projector.publish(observation=observation, observation_root=source,
        task_instruction="Inspect public robot evidence.", remaining_budget=budgets, policy_state=store.steps[-1])
    return surface, pointer, projector, store


@pytest.mark.parametrize("backend", ["libero", "robotwin", "robodojo"])
@pytest.mark.parametrize("retention", ["action_step_lossless_v4", "rewrite_compact_v1"])
def test_complete_initial_facts_reach_both_native_provider_bodies(tmp_path, backend, retention):
    from ora.agent.native.session import NativePlanner
    from ora.agent.providers.native import OpenAICompatibleProvider, translate_anthropic_request

    surface, context, _, _ = published_fixture(tmp_path, backend)
    planner = NativePlanner.__new__(NativePlanner)
    planner.engine = SimpleNamespace()
    planner.run_root, planner.supports_images = tmp_path, True
    planner.tools = SimpleNamespace(surface=surface, controlled_end_effector="right")
    planner.last_actual_snapshot, planner.pending_receipt_body, planner.pending_receipt = None, None, None
    planner.actual_sequence, planner.context_retention = 0, retention
    planner._current_observation_id = lambda: context["public_robot_facts"]["observation_id"]
    message = planner._causal_observation_message(context=context, terminal_lifecycle=False)
    envelope = json.loads(message["content"][0]["text"])
    actual = envelope["current_actual"]
    assert actual["public_robot_facts"] == causal_provider_value(context["public_robot_facts"])
    if retention == "action_step_lossless_v4" and surface.supports("observation_delivery_trim_v2"):
        # The trimmed lossless delivery states freshness only where it differs
        # from comparison.available, which an initial observation leaves false.
        assert "fresh_for_this_action" not in actual["current_views"]
        assert envelope["comparison"]["available"] is False
    else:
        assert actual["current_views"]["fresh_for_this_action"] is False
    assert set(actual) >= {"public_robot_facts", "current_views", "view_dimensions"}
    assert not set(actual) & {"aperture_mm", "eef_pose_xyz_xyzw", "end_effectors"}
    assert envelope["remaining_budget"]["wall_time_s"] == 179.5
    assert not _causal_projection_violations(message["content"])
    assert len([b for b in message["content"] if b["type"] == "image_url"]) == (3 if backend == "robodojo" else 2)
    request = {"model": "offline", "max_tokens": 64, "messages": [message]}
    for body in (OpenAICompatibleProvider.prepare_request(request), translate_anthropic_request(request)):
        text = json.dumps(body)
        assert "world_to_camera_cv" in text and "measured_opening_normalized" in text
        if backend == "robodojo":
            visible = [json.loads(block["text"]) for msg in body["messages"]
                       for block in msg["content"] if block.get("type") == "text" and block["text"].lstrip().startswith("{")]
            facts = next(v["current_actual"]["public_robot_facts"] for v in visible if "current_actual" in v)
            grip = facts["end_effectors"]["right"]["gripper"]["feedback"]
            assert grip["requested_opening_normalized"] == 0.
            assert grip["drive_target_opening_normalized"] == .8
            assert grip["arrival_check"]["tolerance_normalized"] == .03
            assert facts["controller_feedback"]["gripper_drive_limit"]["maximum_normalized_offset"] == .2
        for image in context["post_action_observation"]["images"]:
            assert base64.b64encode((tmp_path / image["run_path"]).read_bytes()).decode() in text


@pytest.mark.parametrize("backend", ["libero", "robotwin"])
def test_observe_current_and_history_use_selected_snapshot_calibration(tmp_path, backend, monkeypatch):
    from ora.agent.tools.runtime import ManipulationToolRuntime
    from ora.agent.composition import validate_schema_instance
    from test_native_causal_runtime import MANUAL
    from placement_fixtures import SelectedMask

    surface, context, projector, store = published_fixture(tmp_path, backend)
    # The runtime composes the fixture surface, which no registered profile carries.
    monkeypatch.setattr("ora.agent.tools.robot.resolve_tool_surface", lambda *args, **kwargs: surface)
    mock = SelectedMask()
    runtime = ManipulationToolRuntime(control_root=projector.control_root, manual_path=MANUAL,
        event_path=tmp_path / "events.jsonl", state_root=store.root,
        workspace_bounds_m={"x": [-3, 3], "y": [-3, 3], "z": [0, 3]},
        tool_surface_profile=surface.profile_id, controlled_end_effector="right",
        segmentation_backend=mock, grounding_backend=mock, grasp_backend=mock)
    camera = "wrist" if backend == "libero" else "right_camera"
    initial = runtime.invoke("observe", {"views": [camera]})
    assert set(initial["public_robot_facts"]["cameras"]) == {camera}
    assert initial["public_robot_facts"]["cameras"][camera]["native_pixel_to_image_offset_xy"] == [0, 0]
    _, observation, source = fixture(tmp_path, backend, 2)
    store.publish(observation=observation, observation_root=source, cause="after_action",
        task_instruction="Inspect", remaining_budget={}, previous_public_action=None)
    pointer = projector.publish(observation=observation, observation_root=source,
        task_instruction="Inspect", remaining_budget={}, policy_state=store.steps[-1])
    prior_control = (projector.control_root / "current-observation.json").read_bytes()
    current = runtime.invoke("observe", {"views": [camera]})
    historical = runtime.invoke("observe", {"step": 0, "views": [camera]})
    for result, identity, offset, role in ((historical, "observation-1", 1, "historical"),
                                          (current, "observation-2", 2, "current")):
        facts = result["public_robot_facts"]
        assert facts["observation_id"] == result["observation_id"] == identity
        assert facts["snapshot_role"] == role
        assert facts["cameras"][camera]["calibration"]["world_to_camera_cv"][0][3] == offset
        validate_schema_instance({k: v for k, v in result.items() if k != "images_base64"},
            surface.result_schemas["observe"])
        assert not set(result) & {"aperture_mm", "aperture_limits_mm", "end_effectors"}
    assert prior_control == (projector.control_root / "current-observation.json").read_bytes()
    assert pointer["public_robot_facts"]["observation_id"] == "observation-2"


@pytest.mark.parametrize("damage", ["path", "json", "matrix", "private_field"])
def test_calibration_and_result_contract_reject_bad_evidence(tmp_path, damage):
    from ora.agent.composition import validate_schema_instance
    surface, observation, source = fixture(tmp_path)
    reference = observation["cameras"]["wrist"]["files"]["calibration"]
    if damage == "path":
        reference["path"] = "../outside.json"
    elif damage in {"json", "matrix"}:
        path = source / "wrist/calibration.json"
        data = json.loads(path.read_text())
        data["extrinsic_cv"][3][0] = 1
        path.write_text("[]" if damage == "json" else json.dumps(data))
        reference["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (source / "observation.json").write_text(json.dumps(observation))
    if damage != "private_field":
        with pytest.raises(PublicFactsError):
            project_current_public_facts(observation=observation, observation_root=source, view_config=surface.view_config)
    else:
        value = project_current_public_facts(observation=observation, observation_root=source, view_config=surface.view_config)
        value["end_effectors"]["right"]["object_gt"] = [1, 2, 3]
        with pytest.raises(Exception, match="object_gt"):
            validate_schema_instance({"success": True, "observation_id": observation["observation_id"],
                "views": ["agentview"], "view_dimensions": {}, "public_robot_facts": value}, surface.result_schemas["observe"])


def test_the_last_held_object_offset_reaches_the_observation_with_its_age(tmp_path):
    # One backend suffices: the delivery point is one backend-neutral branch in
    # _actual_snapshot.
    """An offset only a tool receipt carries is unavailable to a policy that
    stops calling that tool.

    place.plan now returns held_object_offset, but in the sealed 2026-09-11
    LIBERO-10 run the second round never called place.plan at all: it issued
    four "ground object, ground destination" pairs separated only by move_pose,
    using text grounding as its transport feedback. Delivering the offset only
    in a place receipt saves nothing there.

    The observation therefore carries the last offset the tool layer produced,
    with the observation it was measured at and how many observations have
    elapsed since. No new perception runs: this is the policy's own prior
    measurement, re-shown and explicitly aged, so the policy can decide whether
    it still trusts it.
    """
    from ora.agent.native.session import NativePlanner

    backend = "robotwin"
    fixture_surface, context, _, _ = published_fixture(tmp_path, backend)
    # This contract belongs to a composition that declares the capability; the
    # RoboTwin prepared-grasp qualification surface is the one kept declaration.
    # It keeps the fixture's camera set, which that surface does not publish.
    import dataclasses
    declared = resolve_tool_surface("robotwin_toolchain_b2_closed_noservo_v1")
    assert declared.supports("held_object_offset_v1")
    surface = dataclasses.replace(declared, view_config=fixture_surface.view_config)
    planner = NativePlanner.__new__(NativePlanner)
    planner.engine = SimpleNamespace()
    planner.run_root, planner.supports_images = tmp_path, True
    measured = {
        "frame": "world", "translation_m": [0.01, -0.02, 0.12],
        "basis": "visible_object_points_median_minus_tcp_position",
        "observation_id": context["public_robot_facts"]["observation_id"],
        "observation_step": 0,
        "claim_boundary": "Derived, not a measurement of the object pose.",
    }
    planner.tools = SimpleNamespace(
        surface=surface, controlled_end_effector="right",
        last_held_object_offset=measured,
    )
    planner.last_actual_snapshot, planner.pending_receipt_body, planner.pending_receipt = None, None, None
    planner.actual_sequence, planner.context_retention = 0, "action_step_lossless_v4"
    planner._current_observation_id = lambda: context["public_robot_facts"]["observation_id"]
    message = planner._causal_observation_message(context=context, terminal_lifecycle=False)
    actual = json.loads(message["content"][0]["text"])["current_actual"]

    carried = actual.get("held_object_offset")
    assert carried is not None, "the observation withholds the offset already measured"
    assert carried["translation_m"] == measured["translation_m"]
    assert carried["measured_at_observation_id"] == measured["observation_id"]
    assert carried["observations_since"] == 0
    assert "derived" in carried["claim_boundary"].lower()
    assert not _causal_projection_violations(message["content"])


@pytest.mark.parametrize("backend", ["libero", "robotwin"])
def test_a_profile_without_full_current_visual_keeps_its_own_calibration(tmp_path, backend):
    """The two producers are gated by two different capabilities.

    The fixture composition declares ``current_public_robot_facts_v1`` and
    not ``full_current_visual_v1``, so no ``camera_geometry`` is ever delivered
    to it. Delegating the calibration unconditionally would leave this frozen
    composition with a pointer into a field that does not exist - the silent
    degradation this guard exists to prevent. The delegation is therefore bound
    to ``view_config``, which is always present, not to the possibly absent
    ``camera_geometry`` itself.
    """
    from ora.agent.composition import validate_schema_instance

    surface, context, _, _ = published_fixture(tmp_path, backend)
    assert not surface.supports("full_current_visual_v1")
    assert "current_visual_delivery" not in surface.view_config
    assert "current_visual_observation" not in context and "current_visual" not in context

    facts = context["public_robot_facts"]
    assert facts["cameras"]
    for entry in facts["cameras"].values():
        calibration = entry["calibration"]
        assert calibration["status"] == "reported"
        assert calibration["intrinsic_cv"] == [[10, 0, 6], [0, 10, 5], [0, 0, 1]]
        assert calibration["world_to_camera_cv"][0][3] == 1
        assert calibration["camera_axes"] == "x_right_y_down_z_forward"
        assert calibration["world_position_unit"] == "metre"
    validate_schema_instance({"success": True, "observation_id": facts["observation_id"],
        "views": [next(iter(facts["cameras"]))], "view_dimensions": {}, "public_robot_facts": facts},
        surface.result_schemas["observe"])


def test_a_historical_snapshot_never_points_at_the_current_camera_geometry(tmp_path):
    """``camera_geometry`` only ever describes the current observation.

    A historical ``observe`` answers from an earlier snapshot whose calibration
    differs, so it must keep its own numbers even while the live composition is
    delivering ``camera_geometry`` for the current one. A pointer there would
    hand the policy a wrong number, which is worse than a missing one.
    """
    from ora.agent.context.calibration_ownership import (
        delegate_calibration_to_camera_geometry,
    )
    from ora.agent.context.public_facts import project_current_public_facts

    surface = resolve_tool_surface("libero_placement_direct_overhead_v1")
    assert surface.supports("full_current_visual_v1")
    _, observation, source = fixture(tmp_path, "libero", sequence=7)
    # Rebind the fixture observation to the active composition's descriptor.
    observation["robot_geometry_verification"]["descriptor_sha256"] = (
        surface.view_config["public_geometry_descriptor"]["sha256"])
    (source / "observation.json").write_text(json.dumps(observation))
    for role, delegated in (("current", True), ("historical", False)):
        facts = project_current_public_facts(observation=observation, observation_root=source,
            view_config=surface.view_config, snapshot_role=role)
        value = delegate_calibration_to_camera_geometry(facts, view_config=surface.view_config)
        for entry in value["cameras"].values():
            calibration = entry["calibration"]
            if delegated:
                assert calibration["status"] == "reported_in_camera_geometry"
            else:
                assert calibration["status"] == "reported"
                assert calibration["world_to_camera_cv"][0][3] == 7


def test_delegation_without_full_visual_delivery_is_an_identity(tmp_path):
    """No pointer is ever emitted into a field the payload does not publish.

    The rule reads ``view_config``, which is always present, rather than testing
    for the ``camera_geometry`` object it would point at. A composition without
    complete current visual delivery is returned unchanged rather than degraded.
    """
    from ora.agent.context.calibration_ownership import (
        delegate_calibration_to_camera_geometry,
    )
    from ora.agent.context.public_facts import project_current_public_facts

    surface, observation, source = fixture(tmp_path, "libero")
    assert not surface.supports("full_current_visual_v1")
    assert "current_visual_delivery" not in surface.view_config
    facts = project_current_public_facts(observation=observation, observation_root=source,
        view_config=surface.view_config)
    assert delegate_calibration_to_camera_geometry(facts, view_config=surface.view_config) == facts
    assert delegate_calibration_to_camera_geometry(facts, view_config=None) == facts
