"""Dual-arm holding isolation, real RoboTwin receipt stages and placement handoff."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest
import numpy as np
from PIL import Image

from ora.agent.composition import resolve_tool_surface, validate_schema_instance
from ora.agent.tools.runtime import ManipulationToolRuntime
from ora.agent.tools.placement_runtime import stage_evidence
from ora.agent.tools.robot import ToolError
from ora.agent.context import ContextProjector
from ora.agent.evidence.policy_state import PolicyStateStore, REQUIRED_ABSENT, verify_policy_state

from ora.agent.native.gateway import NativeToolGateway, openai_tool_definitions
from ora.agent.providers.native import _anthropic_tool_alias, translate_anthropic_request
from ora.agent.tools import tool_names
from ora.contracts.robot_geometry import GEOMETRY_VERIFICATION_SCHEMA
from ora.agent.tools.placement_arguments import destination_fields

from placement_fixtures import SelectedMask, _hash, _json, MANUAL, PROMPT, PROJECT, materialize_segmented_target


#: The kept RoboTwin surface that still declares place_plan and
#: grasp_verify_retention (the runtime_v1 prepared-grasp qualification binds it).
PROFILE = "robotwin_toolchain_b2_closed_noservo_v1"
#: The fixture publishes the observer view its callers name and the views the
#: kept RoboTwin surfaces admit.
FIXED_CAMERAS = ("observer_camera", "head_camera", "front_camera")
SHA = "a" * 64
ABSOLUTE = {"position_mode": "absolute", "orientation_mode": "absolute_quaternion"}
POSES = {"left": [-0.2, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
         "right": [0.2, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]}


def _measurement():
    return {"name": "visible_clearance_m", "scalar": 0.02, "vector": None, "text": None}


def _result():
    """A concrete RoboTwin place_plan result for one arm."""
    return {
        "schema": "ora.robotwin-place-plan.v1", "status": "candidates", "arm": "left",
        "reason": "Selected visible regions are geometrically compatible.",
        "observation_id": "episode-000:observation-0007", "observation_sha256": SHA,
        "assessment_ref": "place-check-" + "b" * 32,
        "selected_regions": [{"role": "object", "target_ref": "target-object", "camera": "left_camera",
                              "rgb_sha256": SHA, "depth_sha256": SHA, "calibration_sha256": SHA}],
        "stage_evidence": {
            "status": "confirmed", "reason": "Latest gripper directive is close.",
            "latest_gripper_action_id": "action-0003", "latest_gripper_directive": "close",
            "post_open_pose_actions": 0, "last_pose_execution_status": "reached"},
        "checks": {key: {"status": "consistent", "reason": "Visible evidence only.",
                         "measurements": [_measurement()]}
                   for key in ("region", "support", "orientation", "protected_objects")},
        "geometry_measurements": [_measurement()],
        "evidence": {"path": "placement/assessment.json", "sha256": SHA,
                     "overlay_path": "placement/overlay.png", "overlay_sha256": SHA},
        "claim_boundary": "Public selected-region geometry only, not stable support or success.",
        "phase": "preplace",
        "candidates": [{
            "candidate_id": "candidate-0",
            "move_pose": {"targets": {"left": {"position_m": [0.0, 0.1, 0.2],
                                               "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                                               **ABSOLUTE, "interaction": "precision"}}},
            "measurements": [_measurement()],
            "claim_boundary": "One numeric step; no scene/payload collision proof.",
        }],
    }


def control(step=0):
    return {"observation_id": f"obs-{step}", "observation_sha256": f"{step:064x}",
            "generation": step, "open": True}


def action(targets, *, name="set_gripper", step=0, interactions=None):
    return {"schema": "ora-motion-action.v3", "action": name,
            "action_id": f"action-{step}", "observation_id": f"obs-{step}",
            "bound_observation_sha256": f"{step:064x}",
            "arguments": {"targets": targets, **({"interactions": interactions} if interactions else {})}}


def pose_target(arm, z):
    return {"position_m": [*POSES[arm][:2], z], "orientation_xyzw": POSES[arm][3:]}


def history(targets, *, name="set_gripper", step=0, status="executed", changed=True):
    request = action(targets, name=name, step=step)
    receipt = {"schema": "ora.robotwin-sparse-motion-receipt.v7",
        "action_id": request["action_id"], "action": name, "arguments": request["arguments"],
        "target_arms": list(targets), "consumed": True, "state_changed": changed,
        "execution_status": status,
        "arms": {arm: {"targeted": arm in targets, "command_preservation": {
            "gripper_command_target_after": targets.get(arm) if name == "set_gripper" else "open"}}
            for arm in POSES}}
    return {"previous_public_action": request, "previous_execution_receipt": receipt}


def test_robotwin_stage_uses_selected_arm_executed_gripper_receipt_not_reached():
    states = [history({"left": "close"}), history({"right": "open"}, step=1)]
    stage = stage_evidence(states, "pre_release", arm="left")
    assert stage["status"] == "confirmed" and stage["latest_gripper_directive"] == "close"
    states.append(history({"left": "open"}, step=2))
    assert stage_evidence(states, "post_release", arm="left")["status"] == "confirmed"
    states.append(history({"right": pose_target("right", 1.03)}, name="move_pose", step=3, status="reached"))
    assert stage_evidence(states, "post_exit", arm="left")["status"] == "contradicted"
    states.append(history({"left": pose_target("left", 1.03)}, name="move_pose", step=4, status="not_reached"))
    stage = stage_evidence(states, "post_exit", arm="left")
    assert stage["status"] == "confirmed" and stage["last_pose_execution_status"] == "not_reached"


@pytest.mark.parametrize("damage", ["schema", "status", "arm", "directive", "consumed", "changed", "arguments"])
def test_robotwin_stage_rejects_mismatched_or_unexecuted_receipts(damage):
    state = history({"left": "open"})
    receipt = state["previous_execution_receipt"]
    if damage == "schema": receipt["schema"] = "unrecognized"
    elif damage == "status": receipt["execution_status"] = "reached"
    elif damage == "arm": receipt["arms"]["left"]["targeted"] = False
    elif damage == "directive": receipt["arms"]["left"]["command_preservation"]["gripper_command_target_after"] = "close"
    elif damage == "consumed": receipt["consumed"] = False
    elif damage == "changed": receipt["state_changed"] = False
    else: receipt["arguments"] = {"targets": {"right": "open"}}
    assert stage_evidence([state], "post_release", arm="left")["status"] == "unknown"


def _renormalized_pose_history():
    from ora.contracts.action import validate_runtime_action

    # Exact published quaternion from H's ad6b6f3 basket step 4. The isolated
    # worker validates the already validated public action once more.
    target = {"position_m": [-0.2, 0.0, 1.025], "orientation_xyzw": [
        6.2232349999989555e-06, -2.542137999999573e-06,
        0.7141351863138801, 0.7000078111158825]}
    state = history({"left": target}, name="move_pose", step=1, status="reached")
    request = state["previous_public_action"]
    request["arguments"]["interactions"] = {"left": "contact_approach"}
    receipt = state["previous_execution_receipt"]
    receipt["arguments"] = validate_runtime_action(request)["arguments"]
    assert receipt["arguments"] != request["arguments"]
    return state


def test_robotwin_stage_accepts_exact_worker_renormalization_without_rewriting_evidence():
    states = [history({"left": "close"}), _renormalized_pose_history(),
              history({"left": "open"}, step=2)]
    before = copy.deepcopy(states)
    result = stage_evidence(states, "post_release", arm="left")
    assert result["status"] == "confirmed"
    assert states == before


@pytest.mark.parametrize("damage", ["position", "orientation", "interaction", "arm",
    "action_schema", "receipt_schema", "consumed", "status", "invalid_action"])
def test_robotwin_worker_normalization_does_not_admit_other_receipt_drift(damage):
    state = _renormalized_pose_history()
    receipt, request = state["previous_execution_receipt"], state["previous_public_action"]
    if damage == "position": receipt["arguments"]["targets"]["left"]["position_m"][0] += 1e-12
    elif damage == "orientation": receipt["arguments"]["targets"]["left"]["orientation_xyzw"][0] += 1e-12
    elif damage == "interaction": receipt["arguments"]["interactions"]["left"] = "precision"
    elif damage == "arm": receipt["target_arms"] = ["right"]
    elif damage == "action_schema": request["schema"] = "unknown"
    elif damage == "receipt_schema": receipt["schema"] = "unknown"
    elif damage == "consumed": receipt["consumed"] = False
    elif damage == "status": receipt["execution_status"] = "executed"
    else: request["arguments"]["interactions"] = {}
    states = [history({"left": "close"}), state, history({"left": "open"}, step=2)]
    assert stage_evidence(states, "post_release", arm="left")["status"] == "unknown"


@pytest.mark.parametrize("damage", ["gripper", "plan_ref", "camera", "arm", "authority"])
def test_robotwin_placement_result_rejects_non_atomic_or_privileged_payloads(damage):
    value = _result()
    candidate = value["candidates"][0]
    schema = resolve_tool_surface(PROFILE).result_schemas["place_plan"]
    validate_schema_instance(value, schema)
    targets = schema["properties"]["candidates"]["items"]["properties"]["move_pose"]["properties"]["targets"]
    # One selected arm: the cardinality is declared; the schema subset leaves it to the runtime.
    assert (targets["minProperties"], targets["maxProperties"]) == (1, 1)
    if damage == "gripper": candidate["move_pose"]["targets"]["left"]["gripper"] = "open"
    elif damage == "plan_ref": candidate["move_pose"] = {"plan_ref": "plan-hidden"}
    elif damage == "camera": value["selected_regions"][0]["camera"] = "agentview"
    elif damage == "arm": del value["arm"]
    else: value["release_allowed"] = True
    with pytest.raises(Exception):
        validate_schema_instance(value, schema)


def test_robotwin_placement_native_wire():
    surface = resolve_tool_surface(PROFILE)
    names = [item["name"] for item in surface.definitions]
    runtime = SimpleNamespace(surface=surface, allowed_tools=names)
    runtime.definitions = lambda: ManipulationToolRuntime.definitions(runtime)
    assert NativeToolGateway(runtime).delivered() == list(surface.definitions)
    canonical = openai_tool_definitions(names, definitions=runtime.definitions())
    anthropic = translate_anthropic_request({"model": "offline", "max_tokens": 128,
        "messages": [{"role": "user", "content": "Inspect the selected region."}], "tools": canonical})
    wire = {item["name"]: item for item in anthropic["tools"]}
    for definition in surface.definitions:
        if definition["name"] in {"place_plan", "grasp_verify_retention", "set_gripper"}:
            projected = wire[_anthropic_tool_alias(definition["name"])]
            assert projected["input_schema"] == definition["inputSchema"]
            assert projected["description"] == definition["description"]


class RoboTwinPublicRun:
    """Real public timeline, source-bound masks and distinct measured arm poses."""

    def __init__(self, root, *, profile=PROFILE, bind_submitted=False):
        self.root, self.frames = root, []
        self.bind_submitted = bind_submitted
        self.grippers = {arm: "open" for arm in POSES}
        self.store = PolicyStateStore(root, "robotwin-placement-fixture", "a" * 64)
        self.projector = ContextProjector(root / "workspace", root / "control", PROMPT, MANUAL,
            PROJECT, expose_manual=False, main_camera="observer_camera")
        fixture = SelectedMask()
        self.runtime = ManipulationToolRuntime(control_root=root / "control", manual_path=MANUAL,
            event_path=root / "events.jsonl", state_root=self.store.root,
            workspace_bounds_m={"x": [-1, 1], "y": [-1, 1], "z": [0, 2]},
            tool_surface_profile=profile, controlled_end_effector="right",
            segmentation_backend=fixture, grounding_backend=fixture, grasp_backend=fixture)
        self.publish()

    def publish(self, targets=None, *, name="set_gripper", reset=False):
        step = len(self.frames)
        source = self.root / "observations" / f"obs-{step}"
        cameras = {}
        yy, xx = np.mgrid[:96, :96]
        obj = (xx - 48) ** 2 + (yy - 48) ** 2 <= 9 ** 2
        support = (xx >= 12) & (xx < 84) & (yy >= 12) & (yy < 84) & ~obj
        for camera in (*FIXED_CAMERAS, "left_camera", "right_camera"):
            directory = source / camera
            directory.mkdir(parents=True)
            rgb = np.full((96, 96, 3), [120, 126, 130], dtype=np.uint8)
            rgb[obj] = [190, 54, 31]
            Image.fromarray(rgb).save(directory / "rgb.png")
            depth = np.full((96, 96), .8, dtype=np.float32)
            depth[obj] = .795
            np.save(directory / "depth-m.npy", depth, allow_pickle=False)
            _json(directory / "calibration.json", {
                "intrinsic_cv": [[200., 0., 48.], [0., 200., 48.], [0., 0., 1.]],
                "extrinsic_cv": [[1., 0., 0., 0.], [0., -1., 0., 0.],
                                 [0., 0., -1., 1.2], [0., 0., 0., 1.]], "depth_unit": "metres"})
            files = {}
            for role, filename in (("rgb_lossless_png", "rgb.png"),
                ("depth_numpy_m_f32", "depth-m.npy"), ("calibration", "calibration.json")):
                path = directory / filename
                files[role] = {"path": path.relative_to(source).as_posix(),
                    "sha256": _hash(path), "bytes": path.stat().st_size}
            cameras[camera] = {"width": 96, "height": 96, "mount": "fixed" if camera in FIXED_CAMERAS else "wrist",
                "modalities": ["rgb", "depth"], "files": files}
        poses = copy.deepcopy(POSES)
        if step and not reset:
            poses = copy.deepcopy(self.frames[-1]["poses"])
        if name == "move_pose":
            for arm, target in (targets or {}).items():
                poses[arm] = [*target["position_m"], *target["orientation_xyzw"]]
        if self.bind_submitted:
            if reset:
                self.grippers = {arm: "open" for arm in POSES}
            elif name == "set_gripper":
                self.grippers.update(targets or {})
        observation = {"schema": "ora-observation.v1", "observation_id": source.name,
            "sequence": step, "main_camera": "observer_camera", "forbidden_modalities_absent": sorted(REQUIRED_ABSENT),
            "cameras": cameras, "proprioception": {"end_effectors": {
                arm: {"pose_xyz_xyzw": pose, "gripper_command_target": self.grippers[arm]} for arm, pose in poses.items()}}}
        descriptor = (self.runtime.surface.view_config or {}).get("public_geometry_descriptor")
        if descriptor:
            # The fixture robot is the descriptor's grip-centre robot: public robot facts project.
            observation["robot_geometry_verification"] = {
                "schema": GEOMETRY_VERIFICATION_SCHEMA, "descriptor_sha256": descriptor["sha256"],
                "scope": "loaded_robot_geometry_only", "passed": True}
            for effector in observation["proprioception"]["end_effectors"].values():
                effector["controlled_point_semantics"] = "parallel_jaw_grip_center_tcp"
        _json(source / "observation.json", observation)
        request = receipt = None
        if targets:
            joined = history(targets, name=name, step=step-1, status="executed" if name == "set_gripper" else "reached")
            request, raw = joined["previous_public_action"], joined["previous_execution_receipt"]
            if self.bind_submitted:
                request = json.loads((self.root / "control/submitted-action.json").read_text())
                raw.update(action_id=request["action_id"], action=request["action"],
                           arguments=copy.deepcopy(request["arguments"]), controller_tracking={
                               "schema": "ora.motion-execution.v1",
                               "backend": "robotwin", "sampled_steps": 4,
                               "physics_step_attempts": 4, "execution_started": True})
            request["bound_observation_sha256"] = _hash(self.frames[-1]["root"] / "observation.json")
            receipt_path = self.root / "native" / "receipts" / f"action-{step-1}.json"
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            _json(receipt_path, raw)
            receipt = {"path": receipt_path.relative_to(self.root / "native").as_posix(),
                "sha256": _hash(receipt_path), "receipt": raw}
        snapshot = self.store.publish(observation=observation, observation_root=source,
            cause="reset" if reset or not step else "after_action", task_instruction="place the selected item",
            previous_public_action=request, previous_execution_receipt=receipt,
            remaining_budget={"robot_decisions": 100-step, "model_calls": 300-step})
        self.projector.publish(observation=observation, observation_root=source,
            task_instruction="place the selected item", remaining_budget=snapshot["remaining_budget"],
            policy_state=self.store.steps[-1], previous_execution_receipt=receipt)
        regions = {}
        for camera in cameras:
            regions[camera] = {}
            directory = source / camera
            for label, mask in (("object", obj), ("support", support)):
                y, x = np.argwhere(mask)[len(np.argwhere(mask)) // 2]
                manifest, _ = materialize_segmented_target(artifact_root=self.runtime.artifact_root,
                    observation_id=source.name, camera=camera, rgb_path=directory / "rgb.png",
                    depth_path=directory / "depth-m.npy", calibration_path=directory / "calibration.json",
                    point_uv=(int(x), int(y)), label=label, backend=SelectedMask(mask))
                regions[camera][label] = manifest["target_ref"]
        self.frames.append({"root": source, "regions": regions, "poses": poses})
        assert verify_policy_state(self.root, verify_artifacts=True)["status"] == "valid"

    def args(self, arm="left", camera="head_camera", *, tool="place_plan", mode="preplace"):
        regions = self.frames[-1]["regions"][camera]
        schema = next(item["inputSchema"] for item in self.runtime.surface.definitions
                      if tool_names.canonical(item["name"]) == tool_names.canonical(tool))
        return {"arm": arm, "object_ref": regions["object"],
                **destination_fields(schema, kind="support_surface", support_ref=regions["support"]),
                "orientation_requirement": "none",
                "phase" if tool_names.canonical(tool) == "place_plan" else "stage": mode}


@pytest.mark.parametrize("tool", ["grasp_verify_retention", "place_plan"])
@pytest.mark.parametrize("arm", [[], {}, "both"])
def test_robotwin_arm_input_type_errors_are_structured_before_ledger_or_proposal(tmp_path, tool, arm):
    run = RoboTwinPublicRun(tmp_path)
    if tool == "place_plan":
        arguments = {**run.args(), "arm": arm}
    else:
        arguments = {"arm": arm, "before_target_ref": "before", "current_target_ref": "current"}
    with pytest.raises(ToolError) as error:
        run.runtime.invoke(tool, arguments)
    assert error.value.code == "invalid_arguments"
    assert not (run.runtime.control_root / "submitted-action.json").exists()


def test_robotwin_placement_real_public_timeline_selected_tcp_cameras_and_sparse_candidates(tmp_path):
    run = RoboTwinPublicRun(tmp_path)
    run.publish({"left": "close", "right": "close"})
    snapshot = (run.store.root / "manifest.json").read_bytes()
    results = []
    for arm in ("left", "right"):
        result = run.runtime.invoke("place_plan", run.args(arm, camera=f"{arm}_camera"))
        results.append(result)
        assert result["arm"] == arm and result["status"] == "candidates"
        assert result["stage_evidence"]["status"] == "confirmed"
        assert all(source["camera"] == f"{arm}_camera" for source in result["selected_regions"])
        assert result["images_base64"]
        schema = next(item["inputSchema"] for item in run.runtime.surface.definitions if item["name"] == "move_pose")
        for candidate in result["candidates"]:
            assert set(candidate["move_pose"]["targets"]) == {arm}
            validate_schema_instance(candidate["move_pose"], schema)
    left = results[0]["candidates"][0]["move_pose"]["targets"]["left"]["position_m"]
    right = results[1]["candidates"][0]["move_pose"]["targets"]["right"]["position_m"]
    assert np.subtract(left, right) == pytest.approx(np.subtract(POSES["left"][:3], POSES["right"][:3]))
    assert (run.store.root / "manifest.json").read_bytes() == snapshot
    assert not (run.runtime.control_root / "submitted-action.json").exists()


def test_robotwin_placement_stale_sources_and_reset_fail_closed(tmp_path):
    run = RoboTwinPublicRun(tmp_path)
    run.publish({"left": "close"})
    old_args = run.args()
    assert run.runtime.invoke("place_plan", old_args)["stage_evidence"]["status"] == "confirmed"
    run.publish({"left": "open", "right": "open"})
    with pytest.raises(ToolError) as stale:
        run.runtime.invoke("place_plan", old_args)
    assert stale.value.code == "stale_observation"
    current_args = run.args()
    run.runtime.reset_round(1)
    run.publish(reset=True)
    with pytest.raises(ToolError) as reset:
        run.runtime.invoke("place_plan", current_args)
    assert reset.value.code == "stale_observation"


def test_robotwin_retention_estimator_same_round_gate(tmp_path):
    run = RoboTwinPublicRun(tmp_path, bind_submitted=True)
    anchor = run.frames[-1]["regions"]["observer_camera"]["object"]
    run.runtime.invoke("set_gripper", {"targets": {"left": "close"}, "target_ref": anchor})
    run.publish({"left": "close"})
    before = run.frames[-1]["regions"]["observer_camera"]["object"]  # the post-close baseline
    probe = pose_target("left", 1.025)
    run.runtime.invoke("move_pose", {"targets": {"left": {**probe, **ABSOLUTE, "interaction": "precision"}}})
    run.publish({"left": probe}, name="move_pose")
    result = run.runtime.invoke("grasp_verify_retention", {"arm": "left", "before_target_ref": before,
        "current_target_ref": run.frames[-1]["regions"]["observer_camera"]["object"]})
    assert result["status"] == "inconsistent"  # static region did not follow measured left TCP.
    run.runtime.reset_round(1)
    run.publish(reset=True)
    with pytest.raises(ToolError) as reset:
        run.runtime.invoke("grasp_verify_retention", {"arm": "left", "before_target_ref": before,
            "current_target_ref": run.frames[-1]["regions"]["observer_camera"]["object"]})
    assert reset.value.code == "stale_observation"


def skill_history(states, *, action_id='skill-1'):
    """A sparse skill envelope with the same recorded actions as the executor."""
    sequence = ['approach', 'close', 'verify']
    parent = {'schema': 'ora-skill-sequence-action.v1', 'action': 'execute_skill_sequence',
              'action_id': action_id, 'observation_id': 'obs-0', 'bound_observation_sha256': '0' * 64,
              'arguments': {'skill': 'grasp_execute', 'candidate_id': 'grasp-test', 'rationale': 'test',
                            'sequence': sequence, 'resume_from': 0, 'stop_before_commit': False}}
    stages = []
    for index, state in enumerate(states):
        child = copy.deepcopy(state['previous_public_action'])
        child['action_id'] = f'{action_id}:{index:02d}-{sequence[index]}'
        child['observation_id'] = 'obs-0'
        child['bound_observation_sha256'] = '0' * 64
        receipt = copy.deepcopy(state['previous_execution_receipt'])
        receipt['action_id'] = child['action_id']
        if child['action'] == 'move_pose':
            child['arguments']['interactions'] = {arm: 'contact_approach' for arm in child['arguments']['targets']}
            receipt['arguments'] = copy.deepcopy(child['arguments'])
        stages.append({'stage': sequence[index], 'action_id': child['action_id'], 'action': child,
                       'status': 'executed', 'receipt': receipt})
    stages.append({'stage': 'verify', 'action_id': None, 'status': 'executed'})
    receipt = {'schema': 'ora.manipulation-skill-sequence-receipt.v1', 'action': parent,
               'action_id': action_id, 'consumed': True, 'status': 'completed', 'stage_receipts': stages}
    return {'previous_public_action': parent, 'previous_execution_receipt': receipt}


@pytest.mark.parametrize('arm', ['left', 'right'])
def test_place_history_reads_skill_commands_and_isolates_opposite_arm(arm):
    other = 'right' if arm == 'left' else 'left'
    skill = skill_history([history({other: pose_target(other, 1.0)}, name='move_pose', status='reached'),
                           history({other: 'close'})])
    states = [history({arm: 'close'}), skill]
    assert stage_evidence(states, 'pre_release', arm=arm)['status'] == 'confirmed'
    result = stage_evidence(states, 'pre_release', arm=other)
    assert result['status'] == 'confirmed'
    assert result['latest_gripper_action_id'] == 'skill-1:01-close'


@pytest.mark.parametrize('damage', ['missing', 'id', 'arguments', 'order', 'unconsumed'])
def test_place_history_does_not_accept_broken_skill_children(damage):
    skill = skill_history([history({'left': pose_target('left', 1)}, name='move_pose', status='reached'),
                           history({'left': 'close'})])
    child = skill['previous_execution_receipt']['stage_receipts'][1]
    if damage == 'missing': child.pop('receipt')
    elif damage == 'id': child['receipt']['action_id'] = 'different'
    elif damage == 'arguments': child['receipt']['arguments'] = {'targets': {'left': 'open'}}
    elif damage == 'order': child['stage'] = 'approach'
    else: child['receipt']['consumed'] = False
    assert stage_evidence([skill], 'pre_release', arm='left')['status'] == 'unknown'

@pytest.mark.parametrize('damage', [None, 'phase_action', 'phase_receipt'])
def test_historical_nested_target_state_skill_preserves_command_evidence(damage):
    from ora.contracts import TARGET_STATE_ACTION_SCHEMA, target_state_phases
    # The pre-fix driver wrote target_state children without a stage.action.
    skill = skill_history([history({'right': pose_target('right', 1)}, name='move_pose', status='reached'),
                           history({'right': 'close'})])
    for stage in skill['previous_execution_receipt']['stage_receipts'][:2]:
        old = stage.pop('action')
        arguments = {'arm': 'right', **({'gripper': 0.0} if old['action'] == 'set_gripper'
                                       else {'pose': old['arguments']['targets']['right']})}
        target = {**old, 'schema': TARGET_STATE_ACTION_SCHEMA, 'action': 'target_state',
                  'arguments': arguments}
        phase = target_state_phases(target)[0]
        leaf = {'schema': 'ora.robotwin-motion-receipt.v1',
                'action_id': phase['action_id'], 'action': phase['action'],
                'arguments': phase['arguments'], 'consumed': True, 'state_changed': True,
                'motion_status': 'reached', 'steps_used': 1}
        stage['receipt'] = {'schema': 'ora.composite-motion-receipt.v1',
            'action_id': target['action_id'], 'action': target, 'consumed': True,
            'phase_receipts': [{'action': phase, 'receipt': leaf, 'status': 'reached'}]}
    phase = skill['previous_execution_receipt']['stage_receipts'][1]['receipt']['phase_receipts'][0]
    if damage == 'phase_action':
        phase['action'] = {**phase['action'], 'action_id': 'drift'}
    elif damage == 'phase_receipt':
        phase['receipt']['arguments'] = {'arm': 'right', 'gripper': 1.0}
    initial = {'observation': {'id': 'obs-0', 'manifest': {'sha256': '0' * 64}}}
    result = stage_evidence([initial, skill], 'pre_release', arm='right')
    assert result['status'] == ('confirmed' if damage is None else 'unknown')
    if damage is None:
        assert result['latest_gripper_directive'] == 'close'
        assert result['latest_gripper_action_id'] == 'skill-1:01-close:gripper'
