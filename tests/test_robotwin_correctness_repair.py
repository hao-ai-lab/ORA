"""CPU production-path regressions for RoboTwin terminal and motion correctness.

The fake plant supplies only joints, public RGB-D and the environment-owned
predicate. These tests certify control/evidence contracts, not task success.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ora.environment.backends.robotwin import session as session_module
from ora.environment.backends.robotwin.motion_adapter import RoboTwinMotionBackendAdapter
from ora.environment.backends.robotwin.planning import (
    RoboTwinPlanRejected, select_topk_validated_plan_set,
    validate_inter_arm_clearance, validate_single_arm_plan,
)
from ora.environment.backends.robotwin.session import RoboTwinLiveError, RoboTwinLiveSession
from ora.environment.controllers.motion_transaction import select_motion_strategy
from ora.orchestrator.action_executor import execute_runtime_action

ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / 'ora/policy/embodiments/aloha-agilex/control-profile-v10.json'
PROFILE = json.loads(PROFILE_PATH.read_text())
WORKSPACE = {'x': [-1., 1.], 'y': [-1., 1.], 'z': [0., 2.]}


class Plant:
    def __init__(self, *, success_step=None, gain=1.):
        self.q = {arm: np.zeros(6) for arm in ('left', 'right')}
        self.targets = {arm: np.zeros(6) for arm in ('left', 'right')}
        self.steps = 0
        self.success_step = success_step
        self.gain = gain
        self.eval_success = False
        self.take_action_cnt = 0
        self.step_lim = 100
        self.calls = []
        self.observation_reads = 0
        self.predicate_reads = 0
        self.scene = SimpleNamespace(step=self.step)
        self.robot = SimpleNamespace(set_arm_joints=self.write, set_gripper=self.write_gripper)
        for arm in ('left', 'right'):
            setattr(self.robot, f'{arm}_entity', SimpleNamespace(get_qpos=lambda a=arm: self.q[a].copy()))
            setattr(self.robot, f'{arm}_homestate', [0.] * 6)

    def write(self, position, velocity, arm):
        assert not self.eval_success, 'joint command after official terminal'
        self.calls.append(('joint', arm, self.steps))
        self.targets[arm] = np.asarray(position)

    def write_gripper(self, value, arm, per_step=0.):
        assert not self.eval_success, 'gripper command after official terminal'
        self.calls.append(('gripper', arm, self.steps))

    def set_gripper(self, set_tag, **kwargs):
        path = {'num_step': 300, 'per_step': .01, 'result': np.linspace(0., 1., 300)}
        return (copy.deepcopy(path), copy.deepcopy(path)) if set_tag == 'together' else path

    def step(self):
        assert not self.eval_success, 'physics after official terminal'
        self.steps += 1
        for arm in self.q:
            self.q[arm] += self.gain * (self.targets[arm] - self.q[arm])

    def check_success(self):
        self.predicate_reads += 1
        return self.steps == self.success_step

    def pose(self, arm, joints=None):
        q = self.q[arm] if joints is None else np.asarray(joints)
        xyz = q[:3] + np.asarray([-.2 if arm == 'left' else .2, 0., .8])
        return [*xyz, *Rotation.from_rotvec(q[3:]).as_quat()]

    def get_arm_pose(self, arm):
        p = self.pose(arm)
        return [*p[:3], p[6], *p[3:6]]

    def _update_render(self):
        pass

    def close_env(self, **kwargs):
        pass

    def get_obs(self):
        self.observation_reads += 1
        return {
            'observation': {'head_camera': {
                'rgb': np.full((3, 4, 3), self.steps, dtype=np.uint8),
                'depth': np.full((3, 4), 1000. + self.steps, dtype=np.float32),
                'intrinsic_cv': np.eye(3), 'extrinsic_cv': np.eye(4), 'cam2world_gl': np.eye(4),
            }},
            'endpose': {
                f'{arm}_{key}': (self.get_arm_pose(arm) if key == 'endpose' else .5)
                for arm in self.q for key in ('endpose', 'gripper')
            },
            'joint_action': {f'{arm}_arm': self.q[arm].copy() for arm in self.q},
        }


def live_session(tmp_path, plant):
    live = RoboTwinLiveSession(
        robotwin_root=tmp_path, output_root=tmp_path / 'native', task='hanging_mug',
        actual_seed=0, episode=0, action_codec='motion_action_v3',
        control_profile={'profile': copy.deepcopy(PROFILE), 'sha256': hashlib.sha256(PROFILE_PATH.read_bytes()).hexdigest()},
        workspace_bounds_m=WORKSPACE, task_factory=lambda task: plant,
    )
    live.environment = plant
    live._measured_arm_joints = lambda env, arm: env.q[arm].tolist()
    live._public_pose_wxyz = lambda env, arm: env.get_arm_pose(arm)
    live._pinocchio_public_pose_xyzw = lambda env, arm, joints: env.pose(arm, joints)
    live._pinocchio_link6_pose_xyzw = lambda env, arm, joints: env.pose(arm, joints)
    live._pinocchio_chain_points = lambda env, arm, joints: [[-.3 if arm == 'left' else .3, 0., .8]]
    live._arm_joint_limits = lambda env, arm: [[-2., 2.]] * 6
    return live


def action(kind='set_gripper', arguments=None):
    return {
        'schema': 'ora-motion-action.v3', 'action_id': 'action-0',
        'observation_id': 'obs-0', 'bound_observation_sha256': '0' * 64,
        'action': kind, 'arguments': arguments or {'targets': {'left': 'open'}},
    }


@pytest.mark.parametrize('success_step', [1, 5])
@pytest.mark.parametrize('targets', [{'left': 'open'}, {'left': 'open', 'right': 'open'}])
def test_gripper_first_success_freezes_snapshot_and_stops_same_action(tmp_path, success_step, targets):
    plant = Plant(success_step=success_step)
    live = live_session(tmp_path, plant)
    receipt = live.execute(action(arguments={'targets': targets}))['receipt']
    assert plant.steps == success_step
    assert receipt['controller_tracking']['sampled_steps'] == success_step
    assert receipt['controller_tracking']['planned_stages'] == 300
    assert receipt['execution_status'] == 'executed'
    assert receipt['execution_fault'] is None
    assert receipt['terminal_after_action'] is True
    assert live.status()['reason'] == 'official_success'
    frozen = copy.deepcopy(live.official_terminal)
    assert frozen['physics_step_index'] == success_step
    assert plant.observation_reads == 1
    before_reads = plant.predicate_reads
    # Deliberately mutate the fake backing buffers without invoking physics.
    # Subsequent observation/query/persist must still use the first snapshot.
    plant.steps = 250
    plant.q['left'][0] = .5
    plant.success_step = None
    first, images = live.observe(tmp_path / 'policy')
    second, _ = live.observe(tmp_path / 'policy')
    assert first['proprioception'] == second['proprioception']
    assert first['proprioception']['end_effectors']['left']['pose_xyz_xyzw'][0] == -.2
    from PIL import Image
    assert np.asarray(Image.open(images[0])).min() == success_step
    verdict = live.persist_official_verdict()
    assert verdict['official_success'] is True
    assert verdict['official_terminal'] == frozen == live.official_terminal
    assert plant.observation_reads == 1
    assert plant.predicate_reads == before_reads
    with pytest.raises(RoboTwinLiveError, match='already terminal'):
        live.execute(action())
    with pytest.raises(RoboTwinLiveError, match='overwrite'):
        live.persist_official_verdict()


def test_never_true_gripper_finishes_and_remains_false(tmp_path):
    plant = Plant()
    live = live_session(tmp_path, plant)
    receipt = live.execute(action())['receipt']
    assert plant.steps == 300
    assert receipt['terminal_after_action'] is False
    assert live.official_terminal is None
    live.abort('test_boundary')
    before = plant.predicate_reads
    # A query is not an official physical transition, even if an impure fake
    # predicate would newly say true without another simulator step.
    plant.success_step = plant.steps
    assert live.persist_official_verdict()['official_success'] is False
    assert plant.predicate_reads == before
    assert plant.eval_success is False


def test_round_reset_clears_latch_without_rewriting_previous_terminal(tmp_path, monkeypatch):
    plant = Plant(success_step=1)
    live = live_session(tmp_path, plant)
    live.execute(action())
    live.persist_official_verdict()
    previous = tmp_path / 'native' / 'official-terminal' / 'round-0000'
    frozen = {p.name: p.read_bytes() for p in previous.iterdir()}
    verdict = tmp_path / 'native' / 'official-verdict.json'
    verdict.rename(tmp_path / 'first-verdict.json')
    fresh = Plant(success_step=2)
    monkeypatch.setattr(live, '_construct_environment', lambda seed: fresh)
    monkeypatch.setattr(live, '_pinocchio_context', lambda env, arm: None)
    live.restart_round(1, seed=1)
    assert live.official_terminal is live.terminal_observation is None
    assert live.physics_step_sequence == 0
    assert live.status()['terminal'] is False
    live.execute(action())
    assert fresh.steps == 2
    assert live.official_terminal['round_index'] == 1
    assert live.official_terminal['physics_step_index'] == 2
    assert live.persist_official_verdict()['official_success'] is True
    assert {p.name: p.read_bytes() for p in previous.iterdir()} == frozen


@pytest.mark.parametrize('route', ['global', 'local'])
def test_pose_executor_stops_before_endpoint_or_next_local_iteration(tmp_path, monkeypatch, route):
    plant = Plant(success_step=1, gain=.2)
    live = live_session(tmp_path, plant)
    monkeypatch.setattr(RoboTwinMotionBackendAdapter, 'public_tcp_jacobian', lambda self, arm, joints: np.eye(6))
    if route == 'global':
        paths = {'left_arm': {'position': np.asarray([[.01, 0, 0, 0, 0, 0], [.02, 0, 0, 0, 0, 0]]),
                              'velocity': np.zeros((2, 6))}}
        report, arrays, replay = live._execute_motion_transaction_dense_action(
            environment=plant, control_sequence=paths, action=action('move_pose'),
            endpoint_public_tcp_targets={'left': plant.pose('left', [.02, 0, 0, 0, 0, 0])},
        )
    else:
        target = plant.pose('right'); target[0] += .02
        report, arrays, replay = live._execute_local_cartesian_action(
            environment=plant, target_pose_xyzw={'left': plant.pose('left'), 'right': target}, action=action('move_pose'),
        )
    assert report['completed'] is True
    assert report['abort'] is None
    assert report['termination']['reason'] == 'environment_terminal'
    assert plant.steps == 1
    assert arrays['measured_left_tcp__pose_xyzw'].shape[0] == 1


def test_dual_local_arrived_arm_does_not_block_peer(tmp_path, monkeypatch):
    plant = Plant()
    live = live_session(tmp_path, plant)
    monkeypatch.setattr(RoboTwinMotionBackendAdapter, 'public_tcp_jacobian', lambda self, arm, joints: np.eye(6))
    original = RoboTwinMotionBackendAdapter.compute_local_joint_target
    solved = []
    def solve(self, arm, *args):
        solved.append(arm)
        assert arm == 'right'
        return original(self, arm, *args)
    monkeypatch.setattr(RoboTwinMotionBackendAdapter, 'compute_local_joint_target', solve)
    target = plant.pose('right'); target[0] += .01
    report, _, _ = live._execute_local_cartesian_action(
        environment=plant, target_pose_xyzw={'left': plant.pose('left'), 'right': target}, action=action('move_pose'),
    )
    assert report['completed'] is True
    assert solved
    assert plant.q['left'].tolist() == [0.] * 6
    assert abs(plant.pose('right')[0] - target[0]) <= PROFILE['local_cartesian']['position_tolerance_m']


def test_local_rotation_router_does_not_admit_beyond_executor_bound():
    maximum = PROFILE['local_cartesian']['maximum_rotation_rad']
    for offset, expected in [(0., 'local'), (5e-6, 'global_then_local_settle')]:
        angle = maximum + offset
        target = [-.2, 0., .8, math.sin(angle / 2), 0., 0., math.cos(angle / 2)]
        decision = select_motion_strategy(
            action='move_pose', start_public_tcp_poses={'left': [-.2, 0., .8, 0., 0., 0., 1.]},
            target_public_tcp_poses={'left': target}, interactions={'left': 'precision'},
            capabilities=RoboTwinMotionBackendAdapter.capabilities, profile=PROFILE['strategy_router'],
        )
        assert decision.strategy == expected


def test_local_prewrite_admission_is_rejection_not_execution_fault(tmp_path, monkeypatch):
    live = live_session(tmp_path, Plant())
    def reject(**kwargs):
        raise RoboTwinLiveError('local target exceeds source-bound envelope')
    monkeypatch.setattr(live, '_execute_local_cartesian_action', reject)
    receipt = live.execute(action('move_pose', {
        'targets': {'left': {'position_m': [-.2, 0., .801], 'orientation_xyzw': [0, 0, 0, 1]}},
        'interactions': {'left': 'contact_approach'},
    }))['receipt']
    assert receipt['execution_status'] == 'plan_rejected'
    assert receipt['execution_fault'] is None
    assert receipt['state_changed'] is False
    assert live.environment.steps == 0


def test_target_state_success_skips_conditional_gripper_without_fault():
    class Runtime:
        def __init__(self):
            self.terminal = False
            self.calls = []
        def status(self):
            return {'terminal': self.terminal, 'reason': 'official_success' if self.terminal else None}
        def execute(self, phase):
            assert not self.terminal
            self.calls.append(phase['action'])
            self.terminal = True
            return {'commanded_pose_xyz_xyzw': [0, 0, .8, 0, 0, 0, 1],
                    'reached_pose_xyz_xyzw': [0, 0, .8, 0, 0, 0, 1]}
    runtime = Runtime()
    result = execute_runtime_action(runtime, {
        'schema': 'ora-action.v4', 'action_id': 'a', 'observation_id': 'o',
        'bound_observation_sha256': '0' * 64, 'action': 'target_state',
        'arguments': {'arm': 'left', 'pose': {'position_m': [0, 0, .8], 'orientation_xyzw': [0, 0, 0, 1]}, 'gripper': 0.},
    })
    assert runtime.calls == ['move_pose']
    assert result['phase_receipts'][1]['status'] == 'skipped_environment_terminal'
    assert result['phase_receipts'][1]['receipt'] is None


def branch_candidates(arm='left'):
    # Both IK branches have the same public TCP endpoint. The short one sweeps
    # an elbow through the peer; the longer one goes around it.
    sign = -1. if arm == 'left' else 1.
    positions = [
        [[0, 0, 0, 0, 0, 0], [.01, 0, sign * .2, 0, 0, 0], [.02, 0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0, 0], [.006, 0, -sign * .2, 0, 0, 0], [.014, 0, -sign * .2, 0, 0, 0], [.02, 0, 0, 0, 0, 0]],
    ]
    return {'schema': 'robotwin.curobo-plan-result.v2', 'status': 'Success',
            'joint_names': PROFILE['arms'][arm]['planner_joint_names'],
            'candidates': [
                {'schema': 'robotwin.curobo-plan-result.v1', 'status': 'Success',
                 'planner_status': 'SUCCESS', 'ik_candidate_index': index,
                 'position': np.asarray(q, dtype=float), 'velocity': np.zeros_like(q, dtype=float),
                 'interpolation_dt_s': .01}
                for index, q in enumerate(positions)
            ]}


def validate_branch(raw, arm='left'):
    fk = lambda q: [float(q[0]), 0, 1, 0, 0, 0, 1]
    return validate_single_arm_plan(
        raw, expected_joint_names=PROFILE['arms'][arm]['planner_joint_names'],
        measured_start_joints=[0.] * 6, measured_start_pose_xyzw=fk([0.] * 6),
        target_pose_xyzw=fk([.02] * 6), joint_limits=[[-2., 2.]] * 6,
        fk_public_pose=fk, workspace_bounds_m=WORKSPACE, profile=PROFILE['plan_validation'],
    )


@pytest.mark.parametrize('dual', [False, True])
def test_full_candidate_admission_recovers_safe_alternate_including_dual_pair(dual):
    arms = ('left', 'right') if dual else ('left',)
    def admit(plans):
        return validate_inter_arm_clearance(
            plans, measured_joints={'left': [0.] * 6, 'right': [0.] * 6},
            fk_chain_points=lambda arm, q: [[0, (.2 if arm == 'left' else -.2 if dual else 0.) + q[2], 1]],
            profile=PROFILE['dual_arm_collision_check'],
        )
    selected, plans, admission = select_topk_validated_plan_set(
        {arm: branch_candidates(arm) for arm in arms},
        validators={arm: lambda raw, a=arm: validate_branch(raw, a) for arm in arms},
        set_validator=admit,
    )
    assert admission['status'] == 'accepted'
    indices = tuple(selected[arm]['ik_candidate_index'] for arm in arms)
    assert indices != (0,) * len(arms)
    assert all(selected[a]['topk_selection']['combination_rejections'] for a in arms)
    assert admit(plans)['status'] == 'accepted'


def test_no_safe_candidate_still_rejects_and_records_every_candidate():
    def reject(plans):
        raise RoboTwinPlanRejected('inter_arm_collision', 'all paths collide', details={'minimum': 0.})
    with pytest.raises(RoboTwinPlanRejected) as caught:
        select_topk_validated_plan_set(
            {'left': branch_candidates()}, validators={'left': validate_branch}, set_validator=reject,
        )
    assert caught.value.code == 'topk_no_admitted_plan'
    assert len(caught.value.details['candidates']) == 2
    assert all(item['code'] == 'inter_arm_collision' for item in caught.value.details['candidates'])


def test_probe_and_execution_admit_safe_alternate_before_selection(tmp_path):
    plant = Plant()
    live = live_session(tmp_path, plant)
    raw = branch_candidates()
    live._public_to_planner_wxyz = lambda arm, pose: pose
    live._pinocchio_public_pose_xyzw = lambda env, arm, joints: [float(joints[0]) - .2, 0, .8, 0, 0, 0, 1]
    live._pinocchio_link6_pose_xyzw = live._pinocchio_public_pose_xyzw
    live._pinocchio_chain_points = lambda env, arm, joints: [[0, .2 + joints[2] if arm == 'left' else 0., .8]]
    plant.robot.left_plan_path = lambda *args, **kwargs: copy.deepcopy(raw)
    request = {
        'schema': 'ora.motion-probe-request.v1', 'observation_id': 'o',
        'observation_sha256': '0' * 64, 'variant_policy': 'exact',
        'queries': [{'query_id': 'q', 'arm': 'left',
                     'pose': {'position_m': [-.18, 0, .8], 'orientation_xyzw': [0, 0, 0, 1]}}],
    }
    probe = live.probe_motion(request)
    assert probe['results'][0]['reachable'] is True
    reference = live.plan_references[probe['results'][0]['plan_ref']]
    assert reference['planner_result']['ik_candidate_index'] == 1
    assert plant.steps == 0
    # Drive the actual submission admission path as well as the read-only probe.
    proposal = action('move_pose', {
        'targets': {'left': request['queries'][0]['pose']}, 'interactions': {'left': 'free_space'},
    })
    state = live._admit_sparse_motion(action=proposal, receipt_path=live.output_root/'receipts/action-0000.json', started=time.time())
    live._route_motion_strategy(state)
    assert live._plan_and_validate(state) is True
    assert state.planner_results['left']['ik_candidate_index'] == 1
    assert not state.plan_rejections
    assert plant.steps == 0


def test_global_success_skips_local_settle_and_returns_normal_receipt(tmp_path, monkeypatch):
    plant = Plant(success_step=1, gain=.2)
    live = live_session(tmp_path, plant)
    live._public_to_planner_wxyz = lambda arm, pose: pose
    raw = branch_candidates()
    for candidate in raw['candidates']:
        candidate['position'][:, 2] = 0.
    plant.robot.left_plan_path = lambda *args, **kwargs: copy.deepcopy(raw)
    def forbidden(**kwargs):
        pytest.fail('local settle dispatched after global official success')
    monkeypatch.setattr(live, '_execute_local_cartesian_action', forbidden)
    result = live.execute(action('move_pose', {
        'targets': {'left': {'position_m': [-.18, 0., .8], 'orientation_xyzw': [0., 0., 0., 1.]}},
        'interactions': {'left': 'precision'},
    }))
    receipt = result['receipt']
    assert receipt['execution_status'] == 'executed'
    assert receipt['execution_fault'] is None
    assert receipt['terminal_after_action'] is True
    assert receipt['arms']['left']['pose_error']['translation_m'] > .01
    assert receipt['controller_tracking']['termination']['reason'] == 'environment_terminal'
    assert plant.steps == 1


def vendored_planner_with_fake_solver(error, *, all_candidates_error=False):
    tree = ast.parse((ROOT/'ora/environment/backends/robotwin/vendored/planner.py').read_text())
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == 'CuroboPlanner')
    class Tensor:
        def __init__(self, data): self.data = np.asarray(data)
        def cuda(self): return self
        def reshape(self, *shape): return Tensor(self.data.reshape(*shape))
        def unsqueeze(self, axis): return Tensor(np.expand_dims(self.data, axis))
        def detach(self): return self
        def to(self, *args): return self
        def numpy(self): return self.data
    class IK:
        def __init__(self, cfg):
            self.kinematics = SimpleNamespace(get_joint_limits=lambda: SimpleNamespace(position=[[-1] * 6, [1] * 6]))
        def solve_single(self, *args, **kwargs):
            return SimpleNamespace(success=[True, True], solution=Tensor([[.1, 0, 0, 0, 0, 0], [-.1, .2, 0, 0, 0, 0]]), position_error=[0, 0], rotation_error=[0, 0])
    class MotionGen:
        tensor_args = object()
        def __init__(self): self.calls = 0
        def plan_single_js(self, *args):
            self.calls += 1
            if self.calls > 1 or all_candidates_error:
                raise error('candidate solver exception')
            return {'status': 'Success', 'position': np.zeros((2, 6)), 'velocity': np.zeros((2, 6)), 'interpolation_dt_s': .01, 'timings_s': {'total': .001}}
    namespace = {
        'np': np, 'time': time, 'IKSolver': IK,
        'IKSolverConfig': SimpleNamespace(load_from_robot_config=lambda *args, **kwargs: None),
        'torch': SimpleNamespace(tensor=lambda data, **kwargs: Tensor(data), float32='float32'),
        'JointState': SimpleNamespace(from_position=lambda *args, **kwargs: object()),
        'MotionGenPlanConfig': lambda **kwargs: kwargs, '_planner_value': lambda value: value,
        '_motion_result': lambda result, **kwargs: result,
    }
    exec(compile(ast.Module(body=[cls], type_ignores=[]), '<production-vendored-planner>', 'exec'), namespace)
    planner = object.__new__(namespace['CuroboPlanner'])
    planner.motion_gen_config = SimpleNamespace(robot_cfg={})
    planner.world_config = {}
    planner.active_joints_name = PROFILE['arms']['left']['planner_joint_names']
    planner.motion_gen = MotionGen()
    return planner


@pytest.mark.parametrize('error', [AttributeError, ValueError])
def test_late_recoverable_planner_error_preserves_prior_candidate(error):
    planner = vendored_planner_with_fake_solver(error)
    result = planner._plan_pose_via_nearest_ik(object(), [0.] * 6, object())
    assert result['status'] == 'Success'
    assert len(result['candidates']) == 2
    assert result['candidates'][0]['status'] == 'Success'
    assert result['candidates'][1]['planner_status'] == 'PLANNER_ERROR'
    assert result['candidate_error_count'] == 1


def test_all_candidate_errors_remain_planner_errors_not_unreachable():
    planner = vendored_planner_with_fake_solver(AttributeError, all_candidates_error=True)
    result = planner._plan_pose_via_nearest_ik(object(), [0.] * 6, object())
    assert result['status'] == 'Fail'
    assert result['planner_status'] == 'PLANNER_ERROR'
    assert result['valid_query'] is False
    assert result['candidate_error_count'] == 2


def test_device_runtime_error_does_not_reuse_planner_candidates():
    planner = vendored_planner_with_fake_solver(RuntimeError)
    with pytest.raises(RuntimeError, match='candidate solver exception'):
        planner._plan_pose_via_nearest_ik(object(), [0.] * 6, object())
    assert planner.motion_gen.calls == 2
