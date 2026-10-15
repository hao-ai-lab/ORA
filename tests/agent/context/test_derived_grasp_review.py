"""Derived review metadata survives result admission and the next provider request."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ora.agent.tools import tool_names

from ora.agent.composition.tool_surface import resolve_tool_surface, validate_schema_instance, ToolSurfaceError
from ora.agent.evidence.trajectory import ArtifactStore
from ora.agent.native.constants import NATIVE_SLOTS_V3
from ora.agent.native.gateway import NativeToolGateway
from ora.agent.native.session import NativePlanner
from ora.agent.providers.native import OpenAICompatibleProvider, translate_anthropic_request

ROOT = Path(__file__).resolve().parents[3]
SURFACES = {'libero': 'libero_placement_direct_overhead_v1',
            'robotwin': 'robotwin_toolchain_b2_closed_noservo_v2'}


def assert_next_request(tmp_path, runtime, result):
    planner = NativePlanner.__new__(NativePlanner)
    planner.tools, planner.context_composition = runtime, NATIVE_SLOTS_V3
    planner.engine = SimpleNamespace(messages=[])
    planner.artifacts = ArtifactStore(tmp_path / 'wire')
    planner.context_artifacts = None
    planner.supports_images = True
    planner._append = lambda *a, **kw: {'event_id': 'fixture-event'}
    planner.gateway = NativeToolGateway(runtime)  # every recorded result passes the gateway
    image_message = planner._record_tool_result(call_id='call1', model_call_id='model1', step=0,
        proposal_event={'event_id': 'proposal1'}, name='grasp_plan', result=result)
    messages = [{'role': 'user', 'content': 'Inspect candidates.'},
        {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'call1', 'type': 'function',
            'function': {'name': 'grasp_plan', 'arguments': '{}'}}]}, *planner.messages, image_message]
    request = {'model': 'offline', 'max_tokens': 64, 'messages': messages}
    for body in (OpenAICompatibleProvider.prepare_request(request), translate_anthropic_request(request)):
        text = json.dumps(body)
        assert 'global_public_scene_candidates' in text
        assert result['candidates'][0]['candidate_id'] in text
        assert 'internal_tool_error' not in text
        for encoded in result['images_base64']:
            assert encoded in text


@pytest.mark.parametrize('backend', ['libero', 'robotwin'])
def test_complete_grasp_result_reaches_next_provider_request(tmp_path, backend):
    from tests.agent.perception.test_grounding_grasp_runtime import _runtime, _select_correctness_surface
    if backend == 'libero':
        runtime, _ = _runtime(tmp_path)
        _select_correctness_surface(runtime, SURFACES['libero'])
        target = runtime.invoke(tool_names.as_advertised(runtime.allowed_tools, 'vision_ground'), {'camera': 'agentview', 'text': 'red object'})['candidates'][0]['target_ref']
        args = {'target_ref': target}
    else:
        runtime, _ = _runtime(tmp_path)
        _select_correctness_surface(runtime, SURFACES['robotwin'])
        control, root, observation, pose, gripper = runtime._current_grip_state()
        observation['cameras'] = {'head_camera': observation['cameras']['agentview']}
        runtime._current_grip_state = lambda: (control, root, observation, pose, gripper)
        target = runtime.invoke(tool_names.as_advertised(runtime.allowed_tools, 'vision_ground'), {'camera': 'head_camera', 'text': 'red object'})['candidates'][0]['target_ref']
        args = {'arm': 'right', 'target_ref': target}
    result = runtime.invoke(tool_names.as_advertised(runtime.allowed_tools, 'grasp_plan'), args)
    assert result.get('scene_review', {}).get('status') == 'available', result
    assert result['candidates']
    assert [m['camera'] for m in result['image_mapping']] == ['', '']
    validate_schema_instance({k: v for k, v in result.items() if k != 'images_base64'}, runtime.surface.result_schemas[
        tool_names.as_advertised(runtime.surface.result_schemas, 'grasp_plan')])
    assert_next_request(tmp_path, runtime, result)


@pytest.mark.parametrize('backend', ['libero', 'robotwin'])
def test_derived_metadata_does_not_open_physical_camera_boundary(backend):
    surface = resolve_tool_surface(SURFACES[backend])
    schema = surface.result_schemas[tool_names.as_advertised(surface.result_schemas, 'grasp_plan')]['properties']['image_mapping']
    item = {'image_index': 0, 'role': 'local_contact_candidates', 'camera': '',
            'candidate_id': '', 'target_ref': '', 'observation_step': None}
    validate_schema_instance([item], schema)
    for change in ({'role': 'saved_public_view'}, {'camera': 'observer_camera'}, {'camera': 'invented_camera'}):
        with pytest.raises(ToolSurfaceError):
            validate_schema_instance([{**item, **change}], schema)
    observe = surface.result_schemas[tool_names.as_advertised(surface.result_schemas, 'observe')]['properties']['image_mapping']
    with pytest.raises(ToolSurfaceError):
        validate_schema_instance([{**item, 'role': 'saved_public_view'}], observe)


def test_all_derived_review_roles_and_named_pointcloud_remain_usable():
    # The kept surfaces declaring the derived-review tools: RoboTwin's
    # prepared-grasp v1 surface carries grasp_verify_retention and place_plan
    # (no kept surface declares place_verify); LIBERO's default observe names
    # its point cloud.
    surface = resolve_tool_surface('robotwin_toolchain_b2_closed_noservo_v1')
    roles = {'grasp_verify_retention': 'V0_before_current_masks',
             'place_plan': 'public_geometry_review'}
    for name, role in roles.items():
        item = {'image_index': 0, 'role': role, 'camera': '', 'candidate_id': '',
                'target_ref': '', 'observation_step': None}
        validate_schema_instance([item], surface.result_schemas[name]['properties']['image_mapping'])
    surface = resolve_tool_surface(SURFACES['libero'])
    item = {'image_index': 0, 'role': 'saved_public_view', 'camera': 'pointcloud_top',
            'candidate_id': '', 'target_ref': '', 'observation_step': 0}
    validate_schema_instance([item], surface.result_schemas[tool_names.as_advertised(surface.result_schemas, 'observe')]['properties']['image_mapping'])


@pytest.mark.parametrize('backend', ['libero', 'robotwin'])
def test_mark_point_binds_the_batch_panel_mapping_instead_of_the_derived_review_one(backend):
    """These surfaces publish the batched mark_point, which names its own panel.

    The derived-review entry mark_point used to carry was the label
    `perception_feedback.enrich` applies when a result has images and no mapping
    of its own. A batched call always builds one, and it says what the image
    actually is -- one composite panel, bound to a `source_index` rather than to
    its position. On LIBERO that also ends `pointcloud_top` appearing here,
    which is the same loss as the batch refusing a point-cloud view to mark on:
    recorded in `docs/reference/public-interfaces.md`, not silent.
    """
    surface = resolve_tool_surface(SURFACES[backend])
    schema = surface.result_schemas[tool_names.as_advertised(surface.result_schemas, 'mark_point')]['properties']['image_mapping']
    panel = {'image_index': 0, 'role': 'mark_batch_panel', 'camera': 'agentview',
             'source_index': 0, 'candidate_id': None, 'target_ref': None,
             'observation_step': 0}
    validate_schema_instance([panel], schema)
    validate_schema_instance([{**panel, 'source_index': None}], schema)
    for change in ({'role': 'public_geometry_review'}, {'role': 'saved_public_view'}):
        with pytest.raises(ToolSurfaceError):
            validate_schema_instance([{**panel, **change}], schema)
