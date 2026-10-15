"""Audited RoboDojo closed-skill default delivery through the shared resolver, dispatcher and wire."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ora.agent.composition.embodiment import (
    audit_embodiment_sources, load_embodiment_descriptor, render_embodiment_card,
)
from ora.agent.composition.embodiment_geometry import load_public_geometry_binding
from ora.agent.composition.task_contract import resolve_task_contract_layer, with_task_contract
from ora.agent.composition.tool_surface import resolve_tool_surface, validate_schema_instance
from ora.agent.tools.runtime import ManipulationToolRuntime
from ora.agent.tools.robot import ToolError
from ora.agent.composition.robodojo_task_contract import (
    REGISTRY_PATH, verify_contract_dependencies,
)
from ora.orchestrator.spec import RunSpecError, resolve_run_spec
from tests.environments.robodojo.test_robodojo_admission import _runtime
from tests.environments.robodojo.test_robodojo_perception import (
    BOUNDS, FakeHeadGrounding, FakeX5Proposer, _synthetic_robodojo_observation,
)

ROOT = Path(__file__).resolve().parents[3]
SURFACE = "robodojo_placement_closed_skills_v2"
REGISTRY = json.loads((ROOT / REGISTRY_PATH).read_text())["tasks"]


def runtime_for(task):
    from ora.agent.composition.environment_context import resolve_environment_prompt_profile
    from ora.environment.benchmarks.robodojo import public_context
    record = json.loads((ROOT / REGISTRY[task]["record"]).read_text())
    runtime = _runtime()
    runtime.update(task_variant=task, expected_task_language=record["instruction"]["text"],
        layout_file=record["layout"]["layout_file"], layout_sha256=record["layout"]["layout_sha256"],
        native_action_limit=record["native_action_limit"])
    metadata = public_context()
    environment = resolve_environment_prompt_profile(metadata, project_root=ROOT,
        profile_id="robodojo_public_explicit_v6")
    runtime["public_environment_context"] = metadata
    runtime["environment_prompt_profile"] = {key: environment[key] for key in
        ("schema", "profile_id", "profile_path", "profile_sha256", "metadata_sha256")}
    runtime["public_geometry_descriptor"] = environment["compiled_embodiment"]["descriptor"]
    return runtime


def spec_for(task):
    runtime = runtime_for(task)
    return with_task_contract({
        "schema": "ora.run-spec.v1", "experiment": "runtime-v1",
        "config": "dojo-contract", "milestone": "M0", "output_workspace": "outputs/runtime-v1",
        "task": task, "task_instruction": runtime["expected_task_language"], "seed": 0,
        "evidence_tier": "diagnostic", "runtime": runtime,
        "policy": {"kind": "native_api", "model": "offline-test", "provider": "openai_compatible",
            "endpoint": "https://example.invalid/v1", "supports_images": True,
            "context_window_tokens": 262144, "max_output_tokens": 8192,
            "reasoning_effort": "none", "action_codec": "motion_action_v3",
            "context_composition": "native_slots_v3", "context_profile": "native_causal_context_v1",
            "initial_observation": "automatic", "controlled_end_effector": "right",
            "tool_surface": {"profile": SURFACE, "component_overrides": {}}},
        "budgets": {"max_robot_decisions": 1, "max_model_calls": 1, "max_sim_steps": 20, "wall_time_s": 10},
        "policy_context": {"schema": "ora.policy-context.v3",
            "harness_system": ["ora/policy/planners/native/policy-v5.md"],
            "project_instructions": ["ora/policy/environments/robodojo/boundary-v2.md",
                "ora/policy/embodiments/arx-x5/manual-public-geometry-v5.md"],
            "episode_task": ["ora/policy/prompt_bundles/templates/episode-task-v2.md"],
            "tool_manual": ["ora/policy/prompt_bundles/templates/tool-schema-delivery-v1.md"]},
        "qualification": {"required": False},
    }, root=ROOT)


@pytest.mark.parametrize("task", ["insert_tubes", "stack_blocks", "pack_objects_into_box"])
def test_shared_runspec_requires_actual_contract_delivery(task):
    value = spec_for(task)
    expected = value["policy"].pop("task_contract")
    plan = resolve_run_spec(value, root=ROOT)
    assert plan["policy"]["task_contract"] == expected
    assert expected["delivered"] and expected["record_sha256"] == hashlib.sha256(
        (ROOT / REGISTRY[task]["record"]).read_bytes()).hexdigest()
    value["policy_context"]["episode_task"].pop()
    with pytest.raises(RunSpecError, match="episode_task does not carry"):
        resolve_run_spec(value, root=ROOT)


@pytest.mark.parametrize("field", ["required_revision", "required_asset_revision", "layout_sha256", "layout_file"])
def test_different_source_or_layout_cannot_borrow_the_contract(field):
    value = spec_for("stack_blocks")
    value["runtime"][field] = "different-source"
    with pytest.raises(RuntimeError, match="reviewed contract differs"):
        resolve_task_contract_layer(value["runtime"], root=ROOT)


def test_preflight_checks_evaluator_bytes_and_never_delivers_them(tmp_path):
    record = json.loads((ROOT / REGISTRY["stack_blocks"]["record"]).read_text())
    local = tmp_path / "project"
    path = local / REGISTRY["stack_blocks"]["record"]
    path.parent.mkdir(parents=True)
    source = tmp_path / "upstream"
    file = source / record["source"]["source_path"]
    file.parent.mkdir(parents=True)
    file.write_text("fixture evaluator bytes")
    record["source"]["file_sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
    record["evaluator_dependencies"] = []
    path.write_text(json.dumps(record))
    disclosure = {"backend": "robodojo", "delivered": True,
        "record": str(path.relative_to(local)), "record_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    assert verify_contract_dependencies(disclosure, project_root=local,
        source_root=source, asset_root=tmp_path)["checked_dependencies"] == 1
    file.write_text("drift")
    with pytest.raises(RuntimeError, match="dependency drift"):
        verify_contract_dependencies(disclosure, project_root=local, source_root=source, asset_root=tmp_path)


def test_full_body_card_keeps_nominal_facts_separate_from_verified_collision_geometry():
    surface = resolve_tool_surface(SURFACE)
    ref = surface.view_config["public_geometry_descriptor"]
    descriptor = load_embodiment_descriptor(ROOT / ref["path"])
    validate_schema_instance(descriptor, json.loads((ROOT / "ora/policy/embodiments/descriptor-schema-v2.json").read_text()))
    assert audit_embodiment_sources(descriptor, project_root=ROOT)["passed"]
    card = render_embodiment_card(descriptor)
    assert card == (ROOT / "ora/policy/embodiments/arx-x5/manual-public-geometry-v5.md").read_text()
    for term in ("142.6", "157.6", "0.1501", "88.8", "0.185", '"X":', '"Y":', '"Z":', "uncalibrated"):
        assert term in card
    assert load_public_geometry_binding(ref, project_root=ROOT)["geometry"]
    assert (len(card.encode()) + 3) // 4 <= 1200


class BothArmsProposer(FakeX5Proposer):
    def propose(self, **kwargs):
        assert kwargs["compiler_identity"]["id"] in {
            f"contact_graspnet_to_robodojo_x5_{arm}_link6_v2" for arm in ("left", "right")}
        checked = copy.deepcopy(kwargs)
        checked["compiler_identity"]["id"] = "contact_graspnet_to_robodojo_x5_left_link6_v1"
        return super().propose(**checked)


def tool_runtime(tmp_path):
    source, observation = _synthetic_robodojo_observation(tmp_path / "observations")
    control = tmp_path / "control"
    control.mkdir()
    runtime = ManipulationToolRuntime(control_root=control,
        manual_path=ROOT / "ora/policy/prompt_bundles/templates/tool-schema-delivery-v1.md",
        event_path=tmp_path / "events.jsonl", workspace_bounds_m=BOUNDS, state_root=None,
        tool_surface_profile=SURFACE, controlled_end_effector="right",
        segmentation_backend=FakeHeadGrounding(), grounding_backend=FakeHeadGrounding(),
        grasp_backend=BothArmsProposer())
    pointer = {"open": True, "observation_id": observation["observation_id"],
        "observation_path": str(source / "observation.json")}
    runtime._current = lambda: (pointer, source, observation)
    return runtime


@pytest.mark.parametrize("arm", ["left", "right"])
def test_real_dispatcher_delivers_complete_arx_grasp_result(tmp_path, arm):
    runtime = tool_runtime(tmp_path)
    grounded = runtime.invoke("vision_ground", {"camera": "head", "text": "red mug", "max_candidates": 1})
    result = runtime.invoke("grasp_plan", {"target_ref": grounded["candidates"][0]["target_ref"], "arm": arm})
    assert result["status"] == "candidates"
    assert result["compiler"]["controlled_point_semantics"] == "robodojo_link6_pose"
    assert result["compiler"]["physical_jaw_aperture_calibrated"] is False
    assert result["compiler"]["aperture_caveat"]
    assert result["arm"] == arm
    assert result["review_evidence"]["primary_input_required"]
    assert "backend_input_scope=multiview_fused_rgbd" in result["review_evidence"]["claim_boundary"]
    assert f"{arm}_wrist" in result["review_evidence"]["claim_boundary"]
    for candidate in result["candidates"]:
        assert candidate["compiler_checks"]["aperture_calibration"]
        assert candidate["compiler_checks"]["reachability"] == "delegated_to_move_pose"
        assert "motion_probe" not in candidate
    assert not (runtime.control_root / "submitted-action.json").exists()


def test_grounding_rejects_service_overflow_before_backend_call(tmp_path):
    runtime = tool_runtime(tmp_path)
    calls = []
    runtime.grounding_backend.ground = lambda **kwargs: calls.append(kwargs)
    with pytest.raises(ToolError):
        runtime.invoke("vision_ground", {"camera": "head", "text": "red mug", "max_candidates": 6})
    assert calls == []


def test_three_camera_calibration_refreshes_after_action_and_round_reset(tmp_path):
    """Changed K/optical origin must reach both wire formats, never old EEF data."""
    import base64
    from ora.agent.context.codex import ContextProjector
    from ora.agent.context.current_visual import visual_feedback
    from ora.agent.evidence.policy_state import PolicyStateStore, REQUIRED_ABSENT
    from ora.agent.providers.native import OpenAICompatibleProvider, translate_anthropic_request
    from tests.agent.context.test_current_visual_delivery import planner_for
    from tests.agent.native.test_native_causal_runtime import PROMPT, MANUAL, PROJECT

    surface = resolve_tool_surface(SURFACE)
    store = PolicyStateStore(tmp_path, "dojo-camera-test", "plan-test")
    projector = ContextProjector(tmp_path / "workspace", tmp_path / "control",
        PROMPT, MANUAL, PROJECT, run_root=tmp_path, main_camera="head",
        view_config=surface.view_config)
    previous_calibration = None
    for stage, round_index in enumerate((0, 0, 1)):
        source, observation = _synthetic_robodojo_observation(tmp_path / f"stage-{stage}")
        observation.update(observation_id=f"dojo-round-{round_index}-observation-{stage}",
            sequence=stage, round_index=round_index, forbidden_modalities_absent=sorted(REQUIRED_ABSENT),
            robot_geometry_verification={"schema": "ora.loaded-robot-geometry-verification.v1",
                "descriptor_sha256": surface.view_config["public_geometry_descriptor"]["sha256"],
                "scope": "loaded_robot_geometry_only", "passed": True})
        expected = {}
        for index, (name, camera) in enumerate(observation["cameras"].items()):
            path = source / camera["files"]["calibration"]["path"]
            value = json.loads(path.read_text())
            value["intrinsic_cv"][0][0] = 80 + stage * 10 + index
            value["extrinsic_cv"][0][3] = -0.123 - stage - index
            value["object_gt_pose"] = [999, 999, 999]
            path.write_text(json.dumps(value))
            camera["files"]["calibration"].update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                                   bytes=path.stat().st_size)
            expected[name] = value
        (source / "observation.json").write_text(json.dumps(observation))
        store.publish(observation=observation, observation_root=source,
            cause="after_action" if stage == 1 else "reset", task_instruction="Inspect",
            previous_public_action=None, remaining_budget={})
        context = projector.publish(observation=observation, observation_root=source,
            task_instruction="Inspect", remaining_budget={}, policy_state=store.steps[-1])
        planner = planner_for(tmp_path, context, surface)
        message = planner._causal_observation_message(context=context, terminal_lifecycle=False)
        current = json.loads(message["content"][0]["text"])["current_actual"]
        geometry = current["camera_geometry"]
        assert set(geometry) == {"head", "left_wrist", "right_wrist"}
        assert geometry != previous_calibration
        previous_calibration = geometry
        assert len([b for b in message["content"] if b["type"] == "image_url"]) == 3
        for name in geometry:
            assert geometry[name]["intrinsic_cv"] == expected[name]["intrinsic_cv"]
            assert geometry[name]["extrinsic_cv"] == expected[name]["extrinsic_cv"]
            assert np.allclose(np.array(geometry[name]["cam2world_cv"]) @
                               np.array(geometry[name]["extrinsic_cv"]), np.eye(4))
        assert "object_gt_pose" not in json.dumps(message)
        request = {"model": "offline", "max_tokens": 64, "messages": [message]}
        for wire in (OpenAICompatibleProvider.prepare_request(request), translate_anthropic_request(request)):
            serialized = json.dumps(wire)
            for name in geometry:
                assert base64.b64encode((source / name / "rgb.png").read_bytes()).decode() in serialized
        # The native default trims the per-camera conventions that never change
        # (observation_delivery_trim_v2); every value it does deliver is the
        # visual feedback's value for the same calibration.
        full = visual_feedback(context["current_visual_observation"])["camera_geometry"]
        assert {name: {key: full[name][key] for key in fields} for name, fields in geometry.items()} == geometry



def test_native_slots_v4_never_serves_a_changed_calibration_from_the_slot(tmp_path):
    """D3b: a changed camera is inline, an unchanged one is named, a reset re-captures."""
    from ora.agent.context import round_static_facts as slot
    from ora.agent.context.codex import ContextProjector
    from ora.agent.evidence.policy_state import PolicyStateStore, REQUIRED_ABSENT
    from ora.agent.native.engine import NativeSessionEngine
    from tests.agent.context.test_current_visual_delivery import planner_for
    from tests.agent.native.test_native_causal_runtime import PROMPT, MANUAL, PROJECT

    surface = resolve_tool_surface(SURFACE)
    store = PolicyStateStore(tmp_path, "dojo-slot-test", "plan-test")
    projector = ContextProjector(tmp_path / "workspace", tmp_path / "control",
        PROMPT, MANUAL, PROJECT, run_root=tmp_path, main_camera="head",
        view_config=surface.view_config)
    planner = None
    # Stage 1 moves only the left wrist camera; stage 2 is a new round.
    for stage, (round_index, moved) in enumerate(((1, ()), (1, ("left_wrist",)), (2, ()))):
        source, observation = _synthetic_robodojo_observation(tmp_path / f"stage-{stage}")
        observation.update(observation_id=f"dojo-round-{round_index}-observation-{stage}",
            sequence=stage, round_index=round_index - 1, forbidden_modalities_absent=sorted(REQUIRED_ABSENT),
            robot_geometry_verification={"schema": "ora.loaded-robot-geometry-verification.v1",
                "descriptor_sha256": surface.view_config["public_geometry_descriptor"]["sha256"],
                "scope": "loaded_robot_geometry_only", "passed": True})
        for name, camera in observation["cameras"].items():
            path = source / camera["files"]["calibration"]["path"]
            value = json.loads(path.read_text())
            value["extrinsic_cv"][0][3] = -0.5 - (0.01 * stage if name in moved else 0.0)
            path.write_text(json.dumps(value))
            camera["files"]["calibration"].update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                                   bytes=path.stat().st_size)
        (source / "observation.json").write_text(json.dumps(observation))
        store.publish(observation=observation, observation_root=source,
            cause="after_action" if stage == 1 else "reset", task_instruction="Inspect",
            previous_public_action=None, remaining_budget={})
        context = projector.publish(observation=observation, observation_root=source,
            task_instruction="Inspect", remaining_budget={"session_round": round_index},
            policy_state=store.steps[-1])
        if planner is None:
            planner = planner_for(tmp_path, context, surface)
            planner.engine = NativeSessionEngine()
            planner.actual_sequence = 0
            planner.context_composition = slot.COMPOSITION
        planner._current_observation_id = lambda context=context: context[
            "current_visual_observation"]["observation_id"]
        message = planner._causal_observation_message(context=context, terminal_lifecycle=False)
        planner._append_round_static_facts({"event_id": f"agent-event-{stage:06d}"})
        current = json.loads(message["content"][0]["text"])["current_actual"]
        # The default declares observation_delivery_trim_v2: a byte-identical
        # field is left out without being named (static_facts.absent, listed
        # only when non-empty, names slot fields with no current value).
        assert "unchanged" not in current["static_facts"] and "absent" not in current["static_facts"]
        fields = planner.engine.round_static_facts["fields"]
        restored = slot.restore(current, fields)
        for name in ("head", "left_wrist", "right_wrist"):
            expected = -0.5 - (0.01 * stage if name in moved else 0.0)
            assert restored["camera_geometry"][name]["extrinsic_cv"][0][3] == pytest.approx(expected)
        if stage == 1:
            assert set(current["camera_geometry"]) == {"left_wrist"}
            assert current["camera_geometry"]["left_wrist"]["extrinsic_cv"][0][3] == pytest.approx(-0.51)
        else:
            assert "camera_geometry" not in current
    assert [message["_round_static_facts"]["round"] for message in planner.messages] == [1, 2]


def test_delivered_routing_examples_validate_and_compute_dispatches(tmp_path):
    runtime = tool_runtime(tmp_path)
    definitions = {d["name"]: d for d in runtime.surface.definitions}
    # Run literal model-visible call examples through the selected schemas,
    # then exercise pure computation without submitting a robot action. The
    # default's move_pose description carries no literal call example.
    for name in ("compute",):
        examples = [json.loads(line) for line in definitions[name]["description"].splitlines()
                    if line.startswith("{")]
        assert examples
        for arguments in examples:
            validate_schema_instance(arguments, definitions[name]["inputSchema"])
            if name == "compute":
                result = runtime.invoke(name, arguments)
                values = {entry["name"]: entry["value"] for entry in result["outputs"]}
                assert values == ({"distance_m": 0.5} if len(arguments["outputs"]) == 1
                                  else {"point": [.13, .2, .3], "angle": 90.0})
    assert not (runtime.control_root / "submitted-action.json").exists()
    camera = definitions["vision_ground"]["inputSchema"]["properties"]["camera"]
    assert set(camera["enum"]) == set(runtime.surface.view_config["automatic_actual_views"])
    with pytest.raises(ToolError):
        runtime.invoke("vision_ground", {"camera": "front", "text": "red mug"})


@pytest.mark.parametrize(("reason", "residual"), [("ik_failed", None), ("fk_residual_exceeded", .1)])
def test_unreachable_alignment_survives_actual_dispatcher_without_fabricated_residuals(tmp_path, reason, residual):
    from types import SimpleNamespace
    from ora.environment.controllers.alignment import plan_alignment

    runtime = tool_runtime(tmp_path)
    grounded = runtime.invoke("vision_ground", {"camera": "head", "text": "red mug", "max_candidates": 1})
    adapter = SimpleNamespace(solve_batch_reachability=lambda arm, poses, **kw: [
        {"reachable": False, "position_error_m": residual, "orientation_error_rad": residual,
         "iterations": 0, "reason": reason, "claim_boundary": "IK only; no path or grasp guarantee"}
        for _ in poses
    ])
    runtime.align_plan_callback = lambda request: plan_alignment(request, adapter=adapter,
        available_arms=("left", "right"), control_profile={"profile": {"arrival_check": {
            "position_tolerance_m": .005, "orientation_tolerance_rad": .05}}})
    result = runtime.invoke("align_plan", {
        "target_ref": grounded["candidates"][0]["target_ref"], "approach_side": "top", "arm": "both"})
    assert result["motion_planner_called"] is False
    assert {item["arm"] for item in result["candidates"]} == {"left", "right"}
    for item in result["candidates"]:
        assert item["reachable"] is False
        assert item["position_error_m"] == residual
        assert item["orientation_error_rad"] == residual
        assert item["rejection_reason"] == reason
    assert not (runtime.control_root / "submitted-action.json").exists()


@pytest.mark.parametrize("arm", ["left", "right"])
@pytest.mark.parametrize("damage", [None, "position", "orientation", "interaction", "action_id", "boolean"])
def test_command_tree_accepts_only_canonical_motion_roundoff(arm, damage):
    from ora.agent.evidence.command_tree import command_leaves
    from ora.contracts.action import validate_runtime_action

    action = {"schema": "ora-motion-action.v3", "action": "move_pose",
        "action_id": "move-1", "observation_id": "obs-0", "bound_observation_sha256": "a" * 64,
        "arguments": {"interactions": {arm: "precision"}, "targets": {arm: {
            "position_m": [.094188, -.113189, .9177],
            "orientation_xyzw": [-.0037816611109993685, .009549075616998406,
                                 .8985376840778501, .43877618951592673]}}}}
    # Repeated worker validation can alternate the final bit of a unit quaternion.
    receipt = {"schema": "ora.robodojo-sparse-motion-receipt.v1",
        "action": "move_pose", "action_id": "move-1", "consumed": True,
        "arguments": validate_runtime_action(validate_runtime_action(action))["arguments"]}
    target = receipt["arguments"]["targets"][arm]
    if damage == "position": target["position_m"][0] += 1e-6
    elif damage == "orientation": target["orientation_xyzw"][0] += 1e-6
    elif damage == "interaction": receipt["arguments"]["interactions"][arm] = "free_space"
    elif damage == "action_id": receipt["action_id"] = "different-action"
    elif damage == "boolean": target["position_m"][0] = False
    if damage is None:
        assert list(command_leaves(action, receipt, lambda value: value)) == [(action, receipt)]
    else:
        with pytest.raises(ValueError):
            list(command_leaves(action, receipt, lambda value: value))
