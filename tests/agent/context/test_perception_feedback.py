"""H6 source-safe decision feedback, placement evidence and actual protocol delivery."""
import base64
import copy
import json
from types import SimpleNamespace

import pytest

from ora.agent.composition import resolve_tool_surface, validate_schema_instance
from ora.agent.tools.runtime import ManipulationToolRuntime
from ora.agent.native.gateway import NativeToolGateway
from ora.agent.native.session import NativePlanner
from ora.agent.providers.native import translate_anthropic_request

#: The kept surface that declares the placement and retention tools.
PLACEMENT_SURFACE = 'robotwin_toolchain_b2_closed_noservo_v1'


def test_unconfirmed_stage_keeps_valid_geometry_candidates(tmp_path):
    from tests.environments.robotwin.test_robotwin_placement_adapter import RoboTwinPublicRun
    run = RoboTwinPublicRun(tmp_path, profile=PLACEMENT_SURFACE)
    result = run.runtime.invoke('place_plan', run.args('left', camera='left_camera'))
    assert result['stage_evidence']['status'] != 'confirmed'
    assert result['status'] == 'unknown'
    assert result['geometry_measurements'] and result['candidates']
    assert result['uncertainty_reasons']
    assert result['image_mapping'][0]['image_index'] == 0
    for c in result['candidates']:
        validate_schema_instance(c['move_pose'], next(d['inputSchema'] for d in run.runtime.surface.definitions if d['name'] == 'move_pose'))


def test_retention_threshold_margins_keep_missing_values_and_conflict(tmp_path):
    from tests.environments.robotwin.test_robotwin_placement_adapter import RoboTwinPublicRun
    run = RoboTwinPublicRun(tmp_path, profile=PLACEMENT_SURFACE)
    run.publish({'left': 'close'})
    old = run.frames[-1]['regions']['left_camera']['object']
    run.publish()
    current = run.frames[-1]['regions']['left_camera']['object']
    result = run.runtime.invoke('grasp_verify_retention', {'arm': 'left',
        'before_target_ref': old, 'current_target_ref': current})
    assert result['assessment_role'] == 'advisory_only'
    pair = result['pair_feedback'][0]
    assert pair['status'] == result['status']
    assert pair['threshold_gaps']
    for margin in pair['threshold_gaps']:
        if margin['measured'] is not None:
            sign = 1 if margin['criterion'] == 'minimum' else -1
            assert margin['acceptance_margin'] == pytest.approx(sign * (margin['measured'] - margin['threshold']))
    assert 'execution_gate' not in result


def test_batch_result_images_match_candidates_and_current_or_historical_view(tmp_path):
    from tests.environments.robotwin.test_robotwin_placement_adapter import RoboTwinPublicRun
    from tests.agent.perception.test_batch_segmentation import BatchBackend
    run = RoboTwinPublicRun(tmp_path, profile=PLACEMENT_SURFACE)
    # No schema is patched in, so this also checks the runtime honours the
    # batch result contract the resolved surface declares.
    run.runtime.segmentation_backend = BatchBackend()
    result = run.runtime.invoke('vision_segment', {'camera': 'head_camera', 'points_uv': [[20, 20], [60, 60]]})
    assert len(result['images_base64']) == len(result['candidates']) + 1
    for candidate, mapping in zip(result['candidates'], result['image_mapping'][1:]):
        assert mapping['target_ref'] == candidate['target_ref']
        assert mapping['candidate_id'] == candidate['candidate_id']
    run.publish({'left': 'close'})
    old = run.runtime.invoke('observe', {'step': 0, 'views': ['head_camera']})
    session = NativePlanner.__new__(NativePlanner)
    session.tools = run.runtime
    projected = session._causal_tool_result('observe', old)
    assert projected['image_mapping'][0]['observation_step'] == 0
    assert projected['measurement_contract'] == old['measurement_contract']
    assert projected['status'] == 'actual_historical_observation'


def test_grasp_global_review_and_two_provider_wire(tmp_path):
    from tests.agent.perception.test_grounding_grasp_runtime import _runtime, _select_correctness_surface
    from ora.agent.evidence.trajectory import ArtifactStore
    from ora.agent.native.constants import NATIVE_SLOTS_V3
    from ora.agent.providers.native import OpenAICompatibleProvider

    runtime, _ = _runtime(tmp_path)
    _select_correctness_surface(runtime, 'libero_placement_direct_overhead_v1')
    target = runtime.invoke('vision_ground', {'camera': 'agentview', 'text': 'red object'})['candidates'][0]['target_ref']
    result = runtime.invoke('grasp_plan', {'target_ref': target})
    assert result['scene_review']['status'] == 'available'
    assert [m['role'] for m in result['image_mapping']] == ['local_contact_candidates', 'global_public_scene_candidates']
    assert len(result['images_base64']) == 2
    # The manifest says what this surface does with the review: it delivers it.
    manifest = json.loads(next(runtime.artifact_root.glob('grasp-queries/*/manifest.json')).read_text())
    assert manifest['review_visualization']['policy_visible'] is True
    assert ('observed_extent' in manifest['review_visualization']) is runtime.surface.supports(
        'observed_extent_scene_review_v1')
    planner = NativePlanner.__new__(NativePlanner)
    planner.tools, planner.context_composition = runtime, NATIVE_SLOTS_V3
    planner.engine = SimpleNamespace(messages=[])
    planner.artifacts = ArtifactStore(tmp_path / 'wire')
    planner.context_artifacts = None
    planner.supports_images = True
    planner._append = lambda *a, **kw: {'event_id': 'event-fixture'}
    planner.gateway = NativeToolGateway(runtime)  # every recorded result passes the gateway
    image_message = planner._record_tool_result(call_id='call1', model_call_id='model1', step=0,
        proposal_event={'event_id': 'proposal1'}, name='grasp_plan', result=result)
    assert image_message is not None
    messages = [{'role': 'user', 'content': 'Inspect the grasp candidates.'},
        {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'call1', 'type': 'function',
            'function': {'name': 'grasp_plan', 'arguments': json.dumps({'target_ref': target})}}]},
        *planner.messages, image_message]
    request = {'model': 'offline', 'max_tokens': 64, 'messages': messages}
    openai = OpenAICompatibleProvider.prepare_request(request)
    anthropic = translate_anthropic_request(request)
    for body in (openai, anthropic):
        text = json.dumps(body)
        assert 'scene_review' in text and 'global_public_scene_candidates' in text
        assert result['candidates'][0]['candidate_id'] in text
        for encoded in result['images_base64']:
            assert encoded in text


def test_the_live_dojo_review_delivers_the_observed_extent_cell(tmp_path):
    """observed_extent_scene_review_v1 through the real dispatcher and delivery.

    The live RoboDojo surface's image 1 is the 1168x768 review whose top-left
    cell is the observed extent.
    """
    import base64
    import io

    from PIL import Image

    import tests.environments.robodojo.test_robodojo_closed_skills_default as parity
    from tests.agent.motion.test_reach_check import _stub_bridge
    from tests.environments.robodojo.test_x5_topdown import _NoLearned
    from ora.agent.tools import x5_topdown
    from ora.agent.tools.grasp import AlohaHybridGraspProposer

    class Learned(_NoLearned):
        def propose(self, **kwargs):
            _, diagnostics = super().propose(**kwargs)
            return parity.BothArmsProposer().propose(**{
                **kwargs, 'compiler_identity': {**kwargs['compiler_identity'],
                                                'id': 'contact_graspnet_to_robodojo_x5_left_link6_v2'}})[0], diagnostics

    runtime = parity.tool_runtime(tmp_path / 'live')
    runtime.surface = resolve_tool_surface('robodojo_placement_closed_skills_v2')
    runtime.allowed_tools = tuple(item['name'] for item in runtime.surface.definitions)
    runtime.align_plan_callback = _stub_bridge([])
    # The live surface fuses the X5 top-down family (v2 compiler).
    runtime.grasp_backend = AlohaHybridGraspProposer(
        contact_backend=Learned(), backend_id=x5_topdown.HYBRID_BACKEND_ID)
    target = runtime.invoke('vision_ground', {'camera': 'head', 'text': 'red mug', 'max_candidates': 1})
    result = runtime.invoke('grasp_plan', {'target_ref': target['candidates'][0]['target_ref'], 'arm': 'left'})
    assert [m['role'] for m in result['image_mapping']] == [
        'local_contact_candidates', 'global_public_scene_candidates']
    with Image.open(io.BytesIO(base64.b64decode(result['images_base64'][1]))) as image:
        assert image.size == (1168, 768)
    manifest = json.loads(next(runtime.artifact_root.glob('grasp-queries/*/manifest.json')).read_text())
    review = manifest['review_visualization']
    assert review['policy_visible'] is True
    assert 'review_renderer' in manifest['identity']
    assert review['observed_extent']['status'] == 'available'
    assert review['observed_extent']['camera_order'] == ['head', 'left_wrist']


def test_paired_retention_images_keep_exact_reference_and_camera_labels(tmp_path):
    from tests.environments.robotwin.test_robotwin_placement_adapter import RoboTwinPublicRun
    run = RoboTwinPublicRun(tmp_path, profile=PLACEMENT_SURFACE)
    run.publish({'left': 'close'})
    old = run.frames[-1]['regions']['left_camera']['object']
    run.publish()
    current = run.frames[-1]['regions']['left_camera']['object']
    result = run.runtime.invoke('grasp_verify_retention', {'arm': 'left',
        'before_target_ref': old, 'current_target_ref': current})
    assert result['pair_feedback'][0]['image_status'] == 'available'
    assert len(result['images_base64']) == 1
    assert result['pair_feedback'][0]['before_camera'] == 'left_camera'
    assert result['pair_feedback'][0]['current_target_ref'] == current


def test_unconfirmed_stage_does_not_invent_closed_gripper_for_numeric_plan(tmp_path):
    from tests.environments.robotwin.test_robotwin_placement_adapter import RoboTwinPublicRun
    run = RoboTwinPublicRun(tmp_path, profile=PLACEMENT_SURFACE)
    run.publish({'left': 'open'})
    result = run.runtime.invoke('place_plan', run.args('left'))
    assert result['stage_evidence']['latest_gripper_directive'] == 'open'
    assert result['geometry_measurements']
    assert result['candidates'] == []
    assert 'selected_gripper_closed' in result['candidate_prerequisites']


def test_every_tool_that_passes_the_contract_gate_has_a_contract():
    """The gate and the table were two literals for one fact, and they drifted.

    Canonicalising `enrich`'s input moved the gate to `motion_probe` and left
    the table keyed `motion.probe`, so a `motion.probe` call passed the gate and
    raised `KeyError` on the next line -- reaching a policy as a non-retryable
    `internal_tool_error` on the 47 registered profiles that still declare it,
    and doing it after the tool had already done its work.

    The gate is the table's key set now. This asserts they have not become two
    things again, and that every name in it is one the dispatch can reach.
    """

    import inspect
    import re

    from ora.agent.tools import perception_feedback, tool_names

    source = inspect.getsource(perception_feedback.enrich)
    table = re.search(r"contracts = \{(.*?)\n        \}", source, re.S)
    assert table, "the measurement-contract table moved; this test lost its subject"
    keys = set(re.findall(r'\n *"([a-z_.]+)": "', table.group(1)))

    assert keys == set(perception_feedback.MEASUREMENT_CONTRACTS)
    for name in keys:
        assert tool_names.canonical(name) == name, (
            f"{name} is the advertised spelling; enrich canonicalises before "
            f"indexing this table, so the key can never be reached"
        )
