"""Complete public RGB-D reaches Native and managed wire inputs before compression."""
from __future__ import annotations

import base64
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ora.agent.tools import tool_names
from PIL import Image

from ora.agent.composition.baseline import load_native_baseline
from ora.agent.composition.tool_surface import resolve_tool_surface
from ora.agent.context.action_views import PostActionObservationError, resolve_post_action_images
from ora.agent.context.codex import ContextProjector
from ora.agent.context.current_visual import materialize_current_visual
from ora.agent.context.projection import _externalize_tool_results
from ora.agent.contracts import PolicySessionError
from ora.agent.evidence.policy_state import PolicyStateStore
from ora.agent.native.session import NativePlanner
from ora.agent.providers.native import OpenAICompatibleProvider, translate_anthropic_request
from ora.environment.synthetic import SyntheticSession

ROOT = Path(__file__).resolve().parents[1]


def published(root, backend, *, after_servo=False):
    from test_native_causal_runtime import PROMPT, MANUAL, PROJECT
    instance = load_native_baseline(ROOT)["instances"][backend]
    surface = resolve_tool_surface(instance["tool_surface"])
    cameras = instance["runtime"]["policy_cameras"]
    environment = SyntheticSession(root / "environment", "visual", 0,
        main_camera=cameras[0], camera_names=tuple(cameras))
    environment.start()
    observation, _ = environment.observe(root / "source")
    source = root / "source" / observation["observation_id"]
    # Synthetic sensors stand in for the two maintained robot workers; supply
    # their explicit measured state and descriptor verification, never infer it
    # from a commanded gripper value.
    assert surface.supports("current_public_robot_facts_v1")
    if backend == "libero_pro":
        effectors = {"right": {"pose_xyz_xyzw": [0, 0, 1, 0, 0, 0, 1],
            "pose_frame": "robosuite_grip_site", "gripper": .5,
            "gripper_command_target": "open", "gripper_finger_qpos": [.012, -.028]}}
    else:
        effectors = {arm: {"pose_xyz_xyzw": [0, 0, 1, 0, 0, 0, 1],
            "controlled_point_semantics": "parallel_jaw_grip_center_tcp",
            "gripper_command_target": "open", "gripper": .18181818,
            "gripper_state": {"commanded_opening_normalized": 1.,
                "drive_target_opening_normalized": 1.,
                "measured_opening_normalized": .18181818,
                "measured_primary_joint_position": 0.}}
            for arm in ("left", "right")}
    if backend == "robodojo":
        # The RoboDojo default instance (direct_v4): the ARX link6 frame and the
        # typed gripper feedback the worker publishes (test_public_robot_facts).
        from ora.contracts.gripper_feedback import gripper_feedback
        from ora.environment.controllers.robodojo_arx import public_feedback_contract
        effectors = {arm: {"pose_xyz_xyzw": [0, 0, 1, 0, 0, 0, 1],
            "pose_frame": "robodojo_episode_local_ee_link", "gripper": 1.,
            "gripper_finger_qpos": [.044, .0345], "gripper_command": .8,
            "gripper_command_target": "open", "gripper_feedback": gripper_feedback(
                requested_opening_normalized=1., drive_target_opening_normalized=1.,
                measured_opening_normalized=1., effective_target_opening_normalized=1.,
                tolerance_normalized=.03, drive_target_source="resolved_sdk_joint_position_target",
                measurement_source="primary_finger_encoder", pose_hold_semantics="preserve_last_applied_drive_target")}
            for arm in ("left", "right")}
        observation["controller_feedback"] = public_feedback_contract()
    observation["proprioception"]["end_effectors"] = effectors
    from ora.contracts.robot_geometry import GEOMETRY_VERIFICATION_SCHEMA
    observation["robot_geometry_verification"] = {
        "schema": GEOMETRY_VERIFICATION_SCHEMA,
        "descriptor_sha256": surface.view_config["public_geometry_descriptor"]["sha256"],
        "scope": "loaded_robot_geometry_only", "passed": True,
    }
    # Nontrivial public calibration, including a moving optical origin, catches
    # the DS-R06 mistake of silently substituting EEF position for the camera.
    for index, camera in enumerate(cameras):
        rgb = source / camera / "rgb.png"
        with Image.open(rgb) as image:
            pixels = image.copy()
        pixels.putpixel((1, 1), (index * 31, 117, 29))
        pixels.save(rgb)
        observation["cameras"][camera]["files"]["rgb_lossless_png"].update(
            sha256=hashlib.sha256(rgb.read_bytes()).hexdigest(), bytes=rgb.stat().st_size)
        calibration = source / camera / "calibration.json"
        value = json.loads(calibration.read_text())
        value["intrinsic_cv"][0][0] = 50.25 + index
        value["extrinsic_cv"][0][3] = -0.123 - index
        value["extrinsic_cv"][1][3] = -0.234
        value["extrinsic_cv"][2][3] = -1.345
        if backend in {"robotwin", "robodojo"}:
            value["extrinsic_cv"] = value["extrinsic_cv"][:3]
        value["object_gt_pose"] = [999, 999, 999]  # Never propagate arbitrary fields.
        calibration.write_text(json.dumps(value))
        observation["cameras"][camera]["files"]["calibration"].update(
            sha256=hashlib.sha256(calibration.read_bytes()).hexdigest(), bytes=calibration.stat().st_size)
        depth = source / camera / "depth-m.npy"
        np.save(depth, np.array([[0, np.nan, 1, 2], [1, 2, 3, 4], [2, 3, 4, 5]], dtype=np.float32))
        observation["cameras"][camera]["files"]["depth_numpy_m_f32"].update(
            sha256=hashlib.sha256(depth.read_bytes()).hexdigest(), bytes=depth.stat().st_size)
    (source / "observation.json").write_text(json.dumps(observation))
    store = PolicyStateStore(root, "visual-test", "plan-test")
    store.publish(observation=observation, observation_root=source, cause="mutation" if after_servo else "initial",
        task_instruction="Inspect", previous_public_action=None, remaining_budget={})
    projector = ContextProjector(root / "workspace", root / "control", PROMPT, MANUAL, PROJECT,
        main_camera=cameras[0], run_root=root, view_config=surface.view_config,
        camera_roles={name: role for name, role in {"wrist": "right_wrist", "left_camera": "left_wrist", "right_camera": "right_wrist",
                                                    "left_wrist": "left_wrist", "right_wrist": "right_wrist"}.items() if name in cameras})
    context = projector.publish(observation=observation, observation_root=source,
        task_instruction="Inspect", remaining_budget={}, policy_state=store.steps[-1],
        previous_action={"schema": "ora-servo-action.v1", "action": "servo", "action_id": "servo-1",
            "observation_id": "earlier", "arguments": {"mode": "hold", "arms": ["right"]}} if after_servo else None)
    return context, projector, store, surface, source


def planner_for(root, context, surface):
    planner = NativePlanner.__new__(NativePlanner)
    planner.run_root = root
    planner.engine = SimpleNamespace()
    planner.supports_images = True
    planner.tools = SimpleNamespace(surface=surface, controlled_end_effector="right")
    planner.pending_receipt = planner.pending_receipt_body = planner.last_actual_snapshot = None
    planner.actual_sequence = 0
    planner.context_retention = "action_step_lossless_v4"
    planner._current_observation_id = lambda: context["current_visual_observation"]["observation_id"]
    return planner


@pytest.mark.parametrize("backend", ["libero_pro", "robotwin"])
@pytest.mark.parametrize("after_servo", [False, True])
def test_all_current_rgb_and_calibration_reach_both_native_protocols(tmp_path, backend, after_servo):
    context, _, _, surface, source = published(tmp_path, backend, after_servo=after_servo)
    planner = planner_for(tmp_path, context, surface)
    message = planner._causal_observation_message(context=context, terminal_lifecycle=False)
    current = json.loads(message["content"][0]["text"])["current_actual"]
    cameras = load_native_baseline(ROOT)["instances"][backend]["runtime"]["policy_cameras"]
    from ora.agent.context.projection import causal_provider_value
    # The Native facts equal the causal facts.
    assert current["public_robot_facts"] == causal_provider_value(context["public_robot_facts"])
    grip = current["public_robot_facts"]["end_effectors"]["right"]["gripper"]
    if backend == "libero_pro":
        assert grip["measured_aperture_mm"] == 40
        assert grip["measured_finger_displacements_m"] == [.012, -.028]
    else:
        assert grip["commanded_opening_normalized"] == 1
        assert grip["measured_opening_normalized"] == .18181818
    assert set(current["camera_geometry"]) == set(cameras)
    assert len([b for b in message["content"] if b["type"] == "image_url"]) == len(cameras)
    assert all(dim == {"width": 4, "height": 3} for dim in current["view_dimensions"].values())
    assert "object_gt_pose" not in json.dumps(current)
    for index, camera in enumerate(cameras):
        geometry = current["camera_geometry"][camera]
        assert geometry["intrinsic_cv"][0][0] == 50.25 + index
        # observation_delivery_trim_v2 keeps every pose representation
        # (v1's single-pose rule is covered in test_delivery_trim_runtime).
        assert geometry["cam2world_cv"][0][3] == pytest.approx(.123 + index)
        assert geometry["camera_origin_world_m"] == pytest.approx([.123 + index, .234, 1.345])
        assert geometry["camera_axes_world"]["forward"] == [0, 0, 1]
    from ora.agent.context.projection import _causal_projection_violations
    from ora.agent.evidence.trajectory import _forbidden_path
    assert _causal_projection_violations(message["content"]) == []
    assert _forbidden_path(message["content"]) is None
    request = {"model": "offline", "max_tokens": 64, "messages": [message]}
    for body in (OpenAICompatibleProvider.prepare_request(request), translate_anthropic_request(request)):
        wire = json.dumps(body)
        for camera in cameras:
            assert base64.b64encode((source / camera / "rgb.png").read_bytes()).decode() in wire
        assert "50.25" in wire and "1.345" in wire
        assert "public_robot_facts" in wire and "measured_opening_normalized" in wire
    images = resolve_post_action_images(context, root=tmp_path, path_key="run_path")
    assert {image["modality"] for image in images} == {"rgb"}
    assert context["current_visual_observation"]["depth_delivery"] == "on_demand"


@pytest.mark.parametrize("backend", ["libero_pro", "robotwin"])
@pytest.mark.parametrize("after_servo", [False, True])
def test_native_slots_v4_delivers_every_camera_through_the_observation_and_its_slot(
    tmp_path, backend, after_servo,
):
    """D3b: every image stays in the observation; calibration sits in the round slot."""
    from ora.agent.context import round_static_facts as slot
    from ora.agent.native.engine import NativeSessionEngine

    context, _, _, surface, source = published(tmp_path, backend, after_servo=after_servo)
    context = {**context, "remaining_budget": {"session_round": 1, "session_rounds": 1}}
    legacy_planner = planner_for(tmp_path, context, surface)
    legacy = legacy_planner._causal_observation_message(context=context, terminal_lifecycle=False)
    planner = planner_for(tmp_path, context, surface)
    planner.engine = NativeSessionEngine()
    planner.actual_sequence = 0
    planner.context_composition = slot.COMPOSITION
    message = planner._causal_observation_message(context=context, terminal_lifecycle=False)
    cameras = load_native_baseline(ROOT)["instances"][backend]["runtime"]["policy_cameras"]
    images = [block for block in message["content"] if block["type"] == "image_url"]
    assert images == [block for block in legacy["content"] if block["type"] == "image_url"]
    assert len(images) == len(cameras)
    current = json.loads(message["content"][0]["text"])["current_actual"]
    # observation_delivery_trim_v2: an omitted field is not named; only a slot
    # field with no current value would be listed.
    assert "camera_geometry" not in current
    assert current["static_facts"] == {"round_slot": 1}
    planner._append_round_static_facts({"event_id": "agent-event-000001"})
    slot_text = planner.messages[-1]["content"][0]["text"]
    fields = json.loads(slot_text)["round_static_facts"]["current_actual"]
    restored = slot.restore(current, fields)
    assert restored == json.loads(legacy["content"][0]["text"])["current_actual"]
    assert set(restored["camera_geometry"]) == set(cameras)
    from ora.agent.context.projection import (
        _causal_projection_violations, _provider_message,
    )
    assert _causal_projection_violations(
        [_provider_message(message), _provider_message(planner.messages[-1])]) == []
    request = {"model": "offline", "max_tokens": 64, "messages": [message, planner.messages[-1]]}
    for body in (OpenAICompatibleProvider.prepare_request(request), translate_anthropic_request(request)):
        wire = json.dumps(body)
        for camera in cameras:
            assert base64.b64encode((source / camera / "rgb.png").read_bytes()).decode() in wire
        assert "50.25" in wire and "1.345" in wire


def test_current_visual_requires_bound_unmodified_sensor_artifacts(tmp_path):
    context, _, _, _, _ = published(tmp_path, "libero_pro")
    record = context["current_visual_observation"]
    image = tmp_path / record["images"][0]["run_path"]
    image.chmod(0o644)
    image.write_bytes(b"tampered")
    with pytest.raises(PostActionObservationError, match="hash"):
        resolve_post_action_images(context, root=tmp_path, path_key="run_path")
    stale = copy.deepcopy(context)
    stale["policy_state"]["step"] += 1
    with pytest.raises(PostActionObservationError, match="current policy state"):
        resolve_post_action_images(stale, root=tmp_path, path_key="run_path")


def obs(step, count=2):
    return {"role": "user", "_observation": {"step": step}, "_full_current_visual": True,
        "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{step}-{i}"}} for i in range(count)]}


@pytest.mark.parametrize("selector", ["large", "duplicate"])
def test_current_step_results_are_not_replaced_by_progressive_disclosure(selector):
    payload = json.dumps({"measurements": [123] * 15000})
    ref = {"sha256": "a" * 64}
    tool = {"role": "tool", "name": "vision.segment", "content": payload, "_context_artifact_ref": ref}
    messages = [obs(0), obs(1), tool, copy.deepcopy(tool)]
    projected, changed, _ = _externalize_tool_results(messages, through_action_steps=2,
        mutation_tools=(), selector=selector, threshold_bytes=50000, preview_bytes=0)
    assert projected == messages and changed == []


@pytest.mark.parametrize("selector", ["large", "duplicate", "unchanged"])
@pytest.mark.parametrize("media_tail", [0, 2])
def test_lossless_pipeline_delivers_current_result_then_compresses_it_as_history(
    tmp_path, selector, media_tail,
):
    from test_lossless_context_runtime import (
        MANAGEMENT, TOOLS, _pipeline, _messages, _exchange_for_test, _tool_content,
    )
    from ora.agent.native.request_assembly import NativeRequestAssembler
    from ora.agent.providers.native import FakeProvider
    from ora.orchestrator.metrics import _native_request_metrics
    pipeline = _pipeline(tmp_path / "run", management={
        **MANAGEMENT, "inline_media_tail_action_steps": media_tail, "trigger_inline_images": 4,
    })
    assembler = NativeRequestAssembler(
        "test", "provider_default", 2048, False, FakeProvider([]),
    )
    messages = _messages(0)
    messages[-1]["_full_current_visual"] = True
    payload = json.dumps({
        "status": "actual",
        "diagnostic": "L" * (60000 if selector == "large" else 4000),
    }, sort_keys=True, separators=(",", ":"))
    if selector == "duplicate":
        messages.extend(_exchange_for_test(
            "observe", "original", payload, decision="First read.",
        ))
    messages.extend(_exchange_for_test(
        "observe", "large-current", payload, decision="Inspect all current evidence.",
    ))
    first = pipeline.project(messages=messages, tools=TOOLS, observation_step=0)
    assembler.prepare(first)
    visible = json.loads(_tool_content(first["messages"], "large-current"))
    visible.pop("recoverable_artifact", None)
    assert visible == json.loads(payload)
    assert "externalized" not in visible
    # Four actual advances move this result outside the three-step near window.
    messages.extend(_messages(4)[6:])
    messages[-1]["_full_current_visual"] = True
    frozen = pipeline._project_action_step_lossless_v4(
        messages=messages, tools=TOOLS, observation_step=4,
        allow_transition=False,
    )
    assembler.prepare(frozen)
    assert frozen["manifest"]["projection_transitions"] == []
    assert frozen["messages"][:len(first["messages"])] == first["messages"]
    second = pipeline.project(messages=messages, tools=TOOLS, observation_step=4)
    assembler.prepare(second)
    historical = json.loads(_tool_content(second["messages"], "large-current"))
    if selector == "unchanged":
        assert historical == json.loads(payload)
    else:
        assert historical["externalized"] is True
        _, recovered = pipeline.context_artifacts.resolve(
            historical["recoverable_artifact"]["ref"],
        )
        assert recovered.decode() == payload
    transitions = second["manifest"]["projection_transitions"]
    assert any(
        t["kind"] == "lifecycle_read_only_externalization" for t in transitions
    ) == (selector != "unchanged")
    assert second["manifest"]["cache_epoch"] == (
        1
    )
    metrics, _ = _native_request_metrics(
        tmp_path / "run", [{"context_projection": second["manifest"]}], [],
    )
    lossless = metrics["lossless_context_v4"]
    expected_transitions = int(selector != "unchanged") + 1
    assert lossless["lossless_transition_count"] == expected_transitions
    assert lossless["lifecycle_transition_count"] == expected_transitions
    # L1/L2 and L3 share one epoch when both change this request. Repeating
    # the same projection must not advance it again.
    repeated = pipeline.project(messages=messages, tools=TOOLS, observation_step=4)
    assembler.prepare(repeated)
    assert repeated["manifest"]["projection_transitions"] == []
    assert repeated["manifest"]["cache_epoch"] == second["manifest"]["cache_epoch"]
    assert repeated["messages"] == second["messages"]
    messages[4]["content"] = "silently rewritten historical decision"
    with pytest.raises(PolicySessionError, match="cache_prefix_rewritten"):
        assembler.prepare(pipeline.project(
            messages=messages, tools=TOOLS, observation_step=4,
        ))


def test_missing_declared_camera_and_modified_calibration_cannot_silently_deliver_subset(tmp_path):
    context, projector, store, _, source = published(tmp_path, "libero_pro")
    observation = json.loads((source / "observation.json").read_text())
    args = dict(run_root=tmp_path, workspace=projector.workspace, source_root=source,
        policy_state=store.steps[-1], observation_id=observation["observation_id"],
        observation_sha256=context["current_visual_observation"]["observation_sha256"],
        required_views=("agentview", "wrist"))
    with pytest.raises(PostActionObservationError, match="declared current camera"):
        materialize_current_visual(cameras={"agentview": observation["cameras"]["agentview"]}, **args)
    path = source / "wrist/calibration.json"
    path.write_text("{}")
    with pytest.raises(PostActionObservationError, match="hash mismatch"):
        materialize_current_visual(cameras=observation["cameras"], **args)


def test_robotwin_default_excludes_observer_at_collection_tools_and_actual_input(tmp_path):
    from ora.agent.composition.tool_surface import validate_schema_instance, ToolSurfaceError
    from ora.orchestrator.defaults import expand_native_defaults
    from test_native_baseline import native_spec
    expected = ["head_camera", "front_camera", "left_camera", "right_camera"]
    for kind in ("native_api",):
        spec = native_spec("robotwin")
        spec["policy"]["kind"] = kind
        expanded = expand_native_defaults(spec, root=ROOT)
        assert expanded["runtime"]["policy_cameras"] == expected
        assert expanded["runtime"]["policy_observer_camera"] is False
        assert expanded["runtime"]["main_camera"] == "head_camera"
        surface = resolve_tool_surface(expanded["policy"]["tool_surface"]["profile"])
        assert surface.supports("current_public_robot_facts_v1")
        from ora.agent.composition.embodiment import load_embodiment_descriptor, render_embodiment_card
        descriptor_path = ROOT / surface.view_config["public_geometry_descriptor"]["path"]
        compiled_card = render_embodiment_card(load_embodiment_descriptor(descriptor_path))
        context_paths = expanded["policy_context"]["project_instructions"]
        assert any((ROOT / path).read_text() == compiled_card for path in context_paths), (
            "selected public descriptor must deliver its exact compiled card")
        assert not surface.supports("policy_observer_camera_v1")
        assert "observer_camera" not in json.dumps(surface.definitions)
        assert "observer_camera" not in json.dumps(surface.result_schemas)
        schemas = {tool["name"]: tool["inputSchema"] for tool in surface.definitions}
        for name, arguments in {
            "observe": {"views": ["observer_camera"]},
            "vision_segment": {"camera": "observer_camera", "points_uv": [[1, 1]]},
            "vision_ground": {"camera": "observer_camera", "text": "cup"},
            "mark_point": {"point_id": "p", "view": "observer_camera", "x": 1, "y": 1},
        }.items():
            with pytest.raises(ToolSurfaceError):
                validate_schema_instance(arguments, schemas[name], root_path="$input")
    context, projector, store, _, source = published(tmp_path, "robotwin")
    observation = json.loads((source / "observation.json").read_text())
    observation["cameras"]["observer_camera"] = observation["cameras"]["head_camera"]
    with pytest.raises(PostActionObservationError, match="undeclared camera"):
        materialize_current_visual(run_root=tmp_path, workspace=projector.workspace,
            source_root=source, cameras=observation["cameras"], policy_state=store.steps[-1],
            observation_id=observation["observation_id"], observation_sha256=None,
            required_views=tuple(expected))


@pytest.mark.parametrize("backend", ["libero_pro", "robotwin"])
def test_calibration_numbers_have_exactly_one_owner_in_the_delivered_observation(tmp_path, backend):
    """Two modules render the same calibration artifact; only one may publish it.

    Before this contract, ``camera_geometry`` and ``public_robot_facts`` each
    wrote the same intrinsic and world-to-camera matrices into the same
    ``current_actual``. They were byte-identical on every observation of every
    request, so the second copy bought the policy nothing and cost it context.
    ``camera_geometry`` is the richer producer - it alone derives
    ``cam2world_cv``, ``camera_origin_world_m`` and ``camera_axes_world`` - so
    it keeps the numbers, and the facts keep the pixel-space mapping they alone
    own plus an explicit pointer to the owner.
    """
    context, _, _, surface, _ = published(tmp_path, backend)
    planner = planner_for(tmp_path, context, surface)
    message = planner._causal_observation_message(context=context, terminal_lifecycle=False)
    current = json.loads(message["content"][0]["text"])["current_actual"]
    geometry = current["camera_geometry"]
    facts = current["public_robot_facts"]["cameras"]
    assert set(geometry) == set(facts) == set(surface.view_config["automatic_actual_views"])
    # A bare status reads as "unavailable" to a policy, so the payload names the
    # owner -- once, beside cameras, because it is the same sentence for every
    # camera but the name and this object carries one entry per camera per
    # retained observation.
    owner = current["public_robot_facts"]["calibration_owner"]
    assert owner == {"field": "camera_geometry.<camera>",
                     "intrinsic_cv": "intrinsic_cv",
                     "world_to_camera_cv": "extrinsic_cv"}
    for name, entry in facts.items():
        calibration = entry["calibration"]
        assert calibration["status"] == "reported_in_camera_geometry"
        assert set(calibration) == {"status", "pixel_basis"}
        assert calibration["pixel_basis"] == "native_rgb_before_sheet_translation"
        # The once-stated mapping resolves against this very payload, for each
        # camera, substituting that entry's own key for <camera>.
        resolved = current
        for step in owner["field"].replace("<camera>", name).split("."):
            resolved = resolved[step]
        assert isinstance(resolved[owner["intrinsic_cv"]], list)
        assert len(resolved[owner["intrinsic_cv"]]) == 3
        assert len(resolved[owner["world_to_camera_cv"]]) == 4
        # The facts keep exactly the pixel-space mapping they uniquely own.
        assert {"width", "height", "image_view", "native_pixel_to_image_offset_xy",
                "resampling", "pixel_coordinates"} <= set(entry)
    # Each matrix now appears once in the whole delivered observation.
    delivered = json.dumps(current)
    for name in geometry:
        for matrix in (geometry[name]["intrinsic_cv"], geometry[name]["extrinsic_cv"]):
            assert delivered.count(json.dumps(matrix)) == 1
    from ora.agent.context.projection import _causal_projection_violations
    from ora.agent.evidence.trajectory import _forbidden_path
    assert _causal_projection_violations(message["content"]) == []
    assert _forbidden_path(message["content"]) is None
