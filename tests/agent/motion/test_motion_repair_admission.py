"""Selected-pregrasp admission and mode-companion delivery without simulator steps."""
from dataclasses import replace
import json

import pytest

from ora.agent.composition import resolve_tool_surface
from ora.agent.tools import skill_driver, motion_arguments, grasp_feasibility
from ora.agent.tools.robot import ToolError
from tests.agent.tools.test_prepared_grasp_execution import _prepared, _begin
from tests.agent.tools.test_manipulation_skill_driver import CONTACT


@pytest.mark.parametrize('profile', ['robotwin_toolchain_b2_closed_noservo_v2', 'robotwin_toolchain_b2_closed_noservo_v1',
    'robodojo_placement_closed_skills_v2'])
def test_current_defaults_deliver_mode_companions_and_precommit_check(profile):
    surface = resolve_tool_surface(profile)
    assert surface.supports(grasp_feasibility.CAPABILITY)
    move = next(t for t in surface.definitions if t['name'] == 'move_pose')
    props = move['inputSchema']['properties']
    targets = props.get('targets', {}).get('properties', {'right': move['inputSchema']})
    for schema in targets.values():
        fields = schema['properties']
        assert 'translation_m AND position_frame' in fields['position_mode']['description']
        assert 'orientation_frame' in fields['orientation_mode']['description']
        assert 'Never inferred' in fields['position_frame']['description']
    grasp = next(t for t in surface.definitions if t['name'] == 'grasp_execute')
    assert 'before opening' in grasp['description'].lower() or 'Before opening' in grasp['description']


def test_all_missing_companions_are_returned_in_one_refusal():
    with pytest.raises(ToolError) as exc:
        motion_arguments.arm_target({'position_mode': 'delta',
            'orientation_mode': 'axis_angle_delta'}, arm='targets.left')
    assert exc.value.field == 'targets.left.translation_m'
    assert exc.value.constraints['missing_fields'] == ['translation_m', 'position_frame',
        'rotation_axis', 'rotation_angle_deg', 'orientation_frame']
    assert exc.value.constraints['effect'] == 'no_action_submitted'


@pytest.mark.parametrize('reachable', [True, False, None])
def test_selected_grasp_probes_the_jaw_equivalence_class_before_any_commit(tmp_path, reachable):
    host = _prepared(tmp_path)
    host.surface = replace(host.surface, capabilities=(*host.surface.capabilities, grasp_feasibility.CAPABILITY))
    requests = []
    def probe(request):
        assert skill_driver.load_ledger(host) is None
        assert skill_driver.load_commitment(host) is None
        assert not (host.control_root / 'submitted-action.json').exists()
        requests.append(request)
        return {'schema': 'ora.motion-probe-result.v1', 'results': [{
            'query_id': q['query_id'], 'arm': q['arm'], 'reachable': reachable,
            'variants': [{'reachable': reachable, 'failure': {'code': 'planner_failed'}}] if reachable is False else [],
        } for q in request['queries']]}
    host.motion_probe_callback = probe
    if reachable is True:
        result = _begin(host)
        assert result['precommit_motion_check']['query_count'] == len(requests[0]['queries'])
        assert result['cost']['robot_decisions'] == 1
        assert skill_driver.load_commitment(host)['precommit_motion_check']['summary']['reachable'] is True
        # A backend that reports reachability without naming which member of the
        # class it admitted leaves the compiled representative in place.
        assert result['precommit_motion_check']['selected_variant_id'] is None
        assert 'adopted_variant_stages' not in result['precommit_motion_check']
        assert result['precommit_motion_check']['pose_selection']['reason'] == 'members_not_named'
    else:
        with pytest.raises(ToolError) as exc:
            _begin(host)
        assert exc.value.code == ('grasp_pregrasp_unreachable' if reachable is False else 'environment_contract')
        assert skill_driver.load_ledger(host) is None
        assert skill_driver.load_commitment(host) is None
        assert not (host.control_root / 'submitted-action.json').exists()
    assert len(requests) == 1
    # A parallel jaw holds the same two faces whichever way round it is rolled,
    # so refusing on one compiled orientation refused grasps the arm could make.
    assert requests[0]['variant_policy'] == 'parallel_jaw_equivalents'
    assert [q['query_id'] for q in requests[0]['queries']] == ['pregrasp', 'approach']
    assert requests[0]['queries'][0]['pose']['position_m'] == [.2, .1, .96]
    assert requests[0]['observation_sha256']


def _libero_members(orientation):
    # libero_pro/session.py motion probe: q * quat(z, pi) = [y, -x, w, -z].
    x, y, z, w = orientation
    return [('variant-00-exact-+0deg', list(orientation)),
            ('variant-01-jaw-flip-180deg', [y, -x, w, -z])]


def _robotwin_members(orientation):
    from ora.environment.backends.robotwin.session import RoboTwinLiveSession
    return RoboTwinLiveSession._orientation_variants_xyzw(
        orientation, policy='parallel_jaw_equivalents')


def _robodojo_members(orientation):
    import math
    from ora.environment.backends.robodojo.sparse_motion import (
        JAW_SYMMETRY_AXIS, rotate_about_local_axis)
    return [('variant-00-exact-+0deg', list(orientation)),
            ('variant-01-jaw-flip-180deg',
             rotate_about_local_axis(orientation, JAW_SYMMETRY_AXIS, math.pi))]


MEMBERS = {'libero': _libero_members, 'robotwin': _robotwin_members,
           'robodojo': _robodojo_members}
FLIP = {'libero': 'variant-01-jaw-flip-180deg', 'robotwin': 'variant-03-approach-x-+180deg',
        'robodojo': 'variant-01-jaw-flip-180deg'}
EXACT = {'libero': 'variant-00-exact-+0deg', 'robotwin': 'variant-00-approach-x-+0deg',
         'robodojo': 'variant-00-exact-+0deg'}


def _backend_probe(backend, admitted, prefer):
    """A motion-probe result in the shape each backend actually publishes.

    Each admitted variant row and the query's selection carry the target as a
    7-element ``[x, y, z, qx, qy, qz, qw]`` list with a plan handle, exactly as
    libero_pro/session.py, robotwin/session.py and robodojo/sparse_motion.py
    return it. ``prefer`` stands in for the backend's own ranking.
    """
    def probe(request):
        results = []
        for q in request['queries']:
            stage, position = q['query_id'], list(q['pose']['position_m'])
            rows = []
            for index, (variant_id, orientation) in enumerate(MEMBERS[backend](q['pose']['orientation_xyzw'])):
                if variant_id in admitted[stage]:
                    rows.append({'variant_id': variant_id, 'reachable': True,
                                 'target_pose_xyzw': [*position, *orientation],
                                 'plan_ref': f'plan-{stage}-{index}', 'trajectory_duration_s': 1.0})
                else:
                    rows.append({'variant_id': variant_id, 'reachable': False,
                                 'failure': {'code': 'solver_unsuccessful', 'message': 'no plan'}})
            ranked = [row for pick in prefer.get(stage, []) for row in rows
                      if row['variant_id'] == pick and row['reachable']]
            ranked += [row for row in rows if row['reachable'] and row not in ranked]
            selected = ranked[0] if ranked else None
            results.append({'query_id': stage, 'arm': q['arm'], 'reachable': selected is not None,
                            'selected_variant_id': selected and selected['variant_id'],
                            'plan_ref': selected and selected['plan_ref'],
                            'selected_target_pose_xyzw': selected and selected['target_pose_xyzw'],
                            'variants': rows})
        return {'schema': 'ora.motion-probe-result.v1', 'results': results}
    return probe


def _probed_host(tmp_path, backend, admitted, prefer=None):
    host = _prepared(tmp_path)
    host.surface = replace(host.surface, capabilities=(*host.surface.capabilities, grasp_feasibility.CAPABILITY))
    host.motion_probe_callback = _backend_probe(backend, admitted, prefer or {})
    return host


@pytest.mark.parametrize('backend', sorted(MEMBERS))
def test_an_admitted_jaw_flip_in_the_backend_shape_is_adopted(tmp_path, backend):
    # Only the jaw flip is admitted, as when the compiled roll solves nowhere.
    # Before this was fixed the list-shaped selection was read as "not named",
    # the grasp proceeded on the compiled orientation that had just failed and
    # selected_variant_id was reported as None on every backend.
    flip = FLIP[backend]
    host = _probed_host(tmp_path, backend, {'pregrasp': {flip}, 'approach': {flip}})
    compiled = list(CONTACT['orientation_xyzw'])
    expected = dict(MEMBERS[backend](compiled))[flip]
    check = _begin(host)['precommit_motion_check']
    assert check['selected_variant_id'] == flip
    assert check['adopted_variant_stages'] == ['pregrasp', 'approach']
    assert check['pose_selection'] == {
        'decision': 'adopted_equivalent', 'reason': 'compiled_representative_not_admitted',
        'variant_id': flip, 'backend_selected_variants': {'pregrasp': flip, 'approach': flip},
        'compiled_orientation_xyzw': compiled, 'adopted_orientation_xyzw': expected,
    }
    poses = skill_driver.load_commitment(host)['poses']
    # Execution must run the orientation that was admitted, not the one that
    # was refused, and the lift has to follow the contact it is derived from.
    for stage in ('pregrasp', 'approach', 'lift'):
        assert poses[stage]['orientation_xyzw'] == expected
    assert poses['pregrasp']['position_m'] == [.2, .1, .96]
    assert poses['approach']['position_m'] == CONTACT['position_m']
    assert poses['lift']['position_m'] == pytest.approx([.2, .1, .93])


@pytest.mark.parametrize('backend', sorted(MEMBERS))
def test_the_compiled_orientation_stays_when_every_stage_admits_it(tmp_path, backend):
    # The backend ranks by its own motion cost and may prefer the flip even
    # when the compiled roll also solves; that alone does not change the grasp.
    flip, exact = FLIP[backend], EXACT[backend]
    both = {exact, flip}
    host = _probed_host(tmp_path, backend, {'pregrasp': both, 'approach': both},
                        prefer={'pregrasp': [flip], 'approach': [flip]})
    check = _begin(host)['precommit_motion_check']
    assert check['selected_variant_id'] == exact
    assert 'adopted_variant_stages' not in check
    assert check['pose_selection'] == {
        'decision': 'kept_compiled', 'reason': 'compiled_representative_admitted',
        'variant_id': exact, 'backend_selected_variants': {'pregrasp': flip, 'approach': flip},
    }
    poses = skill_driver.load_commitment(host)['poses']
    assert poses['approach'] == CONTACT
    assert poses['pregrasp']['orientation_xyzw'] == CONTACT['orientation_xyzw']


def test_a_member_common_to_both_stages_is_adopted_over_differing_selections(tmp_path):
    # RoboTwin ranks each query separately, so the two stages can select
    # different slack members while both admit the same one.
    minus, plus = 'variant-01-approach-x--10deg', 'variant-02-approach-x-+10deg'
    host = _probed_host(tmp_path, 'robotwin', {'pregrasp': {minus}, 'approach': {minus, plus}},
                        prefer={'approach': [plus]})
    check = _begin(host)['precommit_motion_check']
    assert check['selected_variant_id'] == minus
    assert check['pose_selection']['backend_selected_variants'] == {'pregrasp': minus, 'approach': plus}
    expected = dict(_robotwin_members(CONTACT['orientation_xyzw']))[minus]
    poses = skill_driver.load_commitment(host)['poses']
    assert poses['pregrasp']['orientation_xyzw'] == poses['approach']['orientation_xyzw'] == expected


@pytest.mark.parametrize('backend', sorted(MEMBERS))
def test_stages_admitted_under_different_equivalents_are_refused(tmp_path, backend):
    flip, exact = FLIP[backend], EXACT[backend]
    host = _probed_host(tmp_path, backend, {'pregrasp': {flip}, 'approach': {exact}})
    with pytest.raises(ToolError) as exc:
        _begin(host)
    # Rolling the approach one way and the close the other is not a grasp.
    assert exc.value.code == 'grasp_pregrasp_unreachable'
    # Only another candidate can succeed (EXP11B census D5).
    assert exc.value.as_dict()['retry_requires_changed_arguments'] is True
    assert exc.value.constraints['stage_variants'] == {'pregrasp': flip, 'approach': exact}
    assert exc.value.constraints['stage_admitted_variants'] == {'pregrasp': [flip], 'approach': [exact]}
    assert skill_driver.load_commitment(host) is None
    assert not (host.control_root / 'submitted-action.json').exists()


def test_the_compiled_pose_is_recognised_in_its_normalised_backend_echo():
    # motion.probe normalises and the backend may sign-flip the quaternion;
    # prepared geometry admits a norm within 1e-3 of one.
    compiled = [.2, .1, .9, 0., 0., 0., .9995]
    assert grasp_feasibility._same_pose(compiled, [.2, .1, .9, 0., 0., 0., 1.])
    assert grasp_feasibility._same_pose(compiled, [.2, .1, .9, -0., -0., -0., -1.])
    assert not grasp_feasibility._same_pose(compiled, [.2, .1, .9, 0., 0., 1., 0.])
    assert not grasp_feasibility._same_pose(compiled, [.2, .1, .91, 0., 0., 0., 1.])


def test_a_selected_pose_in_no_published_shape_is_a_contract_fault(tmp_path):
    # No backend returns a Mapping target; reading one as "not named" is how
    # the admitted equivalent was silently dropped, so it now fails by name.
    host = _prepared(tmp_path)
    host.surface = replace(host.surface, capabilities=(*host.surface.capabilities, grasp_feasibility.CAPABILITY))
    host.motion_probe_callback = lambda request: {
        'schema': 'ora.motion-probe-result.v1', 'results': [{
            'query_id': q['query_id'], 'arm': q['arm'], 'reachable': True,
            'selected_variant_id': 'variant-01-jaw-flip-180deg',
            'selected_target_pose_xyzw': dict(q['pose']), 'variants': [],
        } for q in request['queries']]}
    with pytest.raises(ToolError) as exc:
        _begin(host)
    assert exc.value.code == 'environment_contract'
    assert skill_driver.load_commitment(host) is None
    assert not (host.control_root / 'submitted-action.json').exists()


def test_missing_bridge_refuses_before_opening(tmp_path):
    host = _prepared(tmp_path)
    host.surface = replace(host.surface, capabilities=(*host.surface.capabilities, grasp_feasibility.CAPABILITY))
    host.motion_probe_callback = None
    with pytest.raises(ToolError) as exc:
        _begin(host)
    assert exc.value.code == 'tool_unavailable'
    assert skill_driver.load_ledger(host) is None
    assert not (host.control_root / 'submitted-action.json').exists()


def test_terminal_settle_preserves_joint_failure_when_tcp_arrived():
    from ora.agent.native.receipt_projection import _public_terminal_settle, public_motion_failure_code
    result = _public_terminal_settle({'segment': {'execution': {
        'tracking_error_max_abs_rad': .01,
        'terminal_settle': {'joint_tolerance_rad': .005, 'ticks': 64, 'max_ticks': 64,
            'exhausted': True, 'endpoint_error': {'position_m': .0001, 'orientation_rad': .001, 'object_gt': 'SECRET'},
            'endpoint_tolerance': {'position_m': .004, 'orientation_rad': .02}},
    }}})['terminal_settle']
    assert result['joint_within_tolerance'] is False
    assert result['budget_exhausted'] is True
    assert result['joint_error_max_abs_rad'] == .01
    assert result['tcp_endpoint_error']['position_m'] == .0001
    assert 'SECRET' not in json.dumps(result)
    assert public_motion_failure_code('execution', 'terminal_tracking_timeout') == 'controller_timeout'


def test_rejected_link_spheres_keep_names_and_world_centers_through_public_projection():
    from tests.environments.robotwin.test_robotwin_contact_recovery import binding, path
    from ora.environment.backends.robotwin.planning import validate_inter_arm_clearance, RoboTwinPlanRejected
    from ora.agent.evidence.plan_feedback import public_plan_validation
    with pytest.raises(RoboTwinPlanRejected) as exc:
        validate_inter_arm_clearance({'right': path([.2, .09])},
            measured_joints={'left': [0, 0, 1], 'right': [.2, 0, 1]},
            fk_chain_points=lambda arm, joints: [list(joints)],
            profile=binding()['profile']['dual_arm_collision_check'],
            link_names={'left': ['fl_link6'], 'right': ['fr_link6']})
    details = exc.value.details
    details['link_identity']['object_gt'] = 'SECRET'
    result = public_plan_validation({'status': 'rejected', 'code': exc.value.code, 'details': details})
    spheres = result['clearance']['link_identity']
    assert spheres['frame'] == 'world'
    assert spheres['left']['name'] == 'fl_link6'
    assert spheres['right']['name'] == 'fr_link6'
    assert spheres['left']['center_m'] == [0, 0, 1]
    assert spheres['right']['center_m'] == [.09, 0, 1]
    assert spheres['left']['radius_m'] == .045
    assert 'SECRET' not in json.dumps(result)


@pytest.mark.parametrize('learned', [False, True])
def test_aloha_oversize_rejection_discloses_its_compiler_bound(monkeypatch, learned):
    import numpy as np
    from types import SimpleNamespace
    from ora.agent.tools import grasp
    # The bound is derived from the pinned URDF and config rather than written
    # as a literal, so the test asks for the ALOHA compiler by name and reads
    # the number off it. Pinning the digits here is what made the previous bound
    # look deliberate when it was a stale literal.
    bound = grasp.ALOHA_NOMINAL_MAXIMUM_OPENING_M
    compiler = next(dict(v) for v in grasp.GRASP_COMPILER_IDENTITIES.values()
                    if v.get('maximum_opening_m') == bound
                    and v.get('embodiment_descriptor_id') == 'aloha_agilex')
    x, y = np.meshgrid(np.linspace(-.1, .1, 12), np.linspace(-.1, .1, 12))
    points = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, .9)])
    if learned:
        proposer = grasp.ContactGraspNetApiProposer()
        body = {'schema': 'ora.model-service.grasp-proposals.v1',
            'request_id': 'offline', 'metrics': {'queue_ms': 0, 'inference_ms': 0, 'total_ms': 0},
            'poses_camera_npy_base64': grasp._encode_npy(np.eye(4)[None]),
            'scores_npy_base64': grasp._encode_npy(np.array([.9])),
            'openings_m_npy_base64': grasp._encode_npy(np.array([.12])),
            'contact_points_camera_npy_base64': grasp._encode_npy(np.zeros((1, 3)))}
        monkeypatch.setattr(grasp, '_require_response_service_identity', lambda **kwargs: None)
        monkeypatch.setattr(grasp, '_verify_model_identity', lambda *args: {})
        monkeypatch.setattr(proposer._client, 'post', lambda *args, **kwargs: SimpleNamespace(
            raise_for_status=lambda: None, json=lambda: body))
    else:
        proposer = grasp.PcaTopDownGraspProposer()
    candidates, diagnostics = proposer.propose(scene_points_camera_m=points,
        target_points_camera_m=points, camera_to_world=np.eye(4),
        target_ref='test', workspace_bounds_m={'x': [-2, 2], 'y': [-2, 2], 'z': [-2, 2]},
        max_candidates=3, compiler_identity=compiler)
    assert not candidates
    assert diagnostics['rejected']
    for rejection in diagnostics['rejected']:
        assert rejection['maximum_opening_m'] == bound
        assert rejection['predicted_opening_m' if learned else 'required_opening_m'] > bound
        assert rejection['limit_semantics'] == 'compiler_admission_not_measured_physical_aperture'


def test_both_arm_omissions_preserve_values_and_accept_one_complete_correction():
    from copy import deepcopy
    request = {'left': {'position_mode': 'delta', 'translation_m': [0, 0, -.04],
                       'orientation_mode': 'hold_current'},
               'right': {'position_mode': 'hold_current', 'orientation_mode': 'axis_angle_delta',
                         'rotation_axis': 'z', 'rotation_angle_deg': 15}}
    original = deepcopy(request)
    with pytest.raises(ToolError) as exc:
        motion_arguments.arm_targets(request)
    detail = exc.value.constraints
    assert detail['missing_paths'] == ['targets.left.position_frame', 'targets.right.orientation_frame']
    assert detail['preserved_targets'] == original == request
    assert detail['effect'] == 'no_action_submitted'
    corrected = deepcopy(detail['preserved_targets'])
    corrected['left']['position_frame'] = 'world'
    corrected['right']['orientation_frame'] = 'end_effector'
    result = motion_arguments.arm_targets(corrected)
    assert result['left']['position']['translation_m'] == [0, 0, -.04]
    assert result['right']['orientation']['angle_deg'] == 15


def test_developing_dojo_default_delivers_source_controller_rules():
    from pathlib import Path
    from ora.environment.controllers.robodojo_arx import public_feedback_contract
    root = Path(__file__).resolve().parents[3]
    # The RoboDojo default instance (integration 0059 unit A0) replaced the
    # developing-surface entry; it restates angles in degrees.
    requirement = json.loads((root / 'tests/data/default-expectations.json').read_text())['instances']['robodojo']
    surface = resolve_tool_surface(requirement['tool_surface'])
    assert all(surface.supports(cap) for cap in requirement['required_capabilities'])
    gripper = next(t for t in surface.definitions if t['name'] == 'set_gripper')
    delivered = json.loads(gripper['description'].split('Controller feedback contract: ', 1)[1])
    from ora.agent.context.policy_units import policy_degrees
    assert surface.supports('policy_angles_degrees_v1')
    assert delivered == policy_degrees(public_feedback_contract())
    from ora.contracts.robodojo_arx import ARX_X5_CONTRACT
    from ora.environment.controllers.robodojo_arx import ARX_X5_CONTRACT as executor_contract
    assert executor_contract is ARX_X5_CONTRACT
    assert delivered['gripper_tolerance_normalized'] == .03
    assert delivered['gripper_drive_limit']['maximum_normalized_offset'] == .2
