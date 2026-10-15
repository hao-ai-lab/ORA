"""Original pixel integrity, bounded failure, backend pass-through and playback."""
import hashlib
import json
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from ora.environment.execution_capture import ExecutionCapture, PhysicsStepTap, resolve_execution_capture
from ora.environment.backends.libero_pro.session import _CameraMetadataEnvironment
from viewer.execution_capture import original_execution_source


def record(recorder, pixels, phase='control', at=0.0, round_index=0):
    return recorder.capture(lambda: {'agentview': pixels, 'wrist': pixels},
        round_index=round_index, action_id='a1' if phase == 'control' else None,
        action_sequence=1, action='move_pose', sim_step=round(at * 100),
        sim_time_s=at, step_unit='test', phase=phase, force=True)


def make(tmp_path, **config):
    return ExecutionCapture(tmp_path / 'native', backend='test', identity={'run_id': 'r'},
                            cameras=['agentview', 'wrist'], config=config)


def test_original_pixels_are_copied_and_round_clock_resampled(tmp_path):
    recorder = make(tmp_path)
    pixels = np.full((4, 5, 3), 23, dtype=np.uint8)
    original_hash = hashlib.sha256(pixels.tobytes()).hexdigest()
    record(recorder, pixels, 'reset')
    pixels[:] = 91
    record(recorder, pixels, at=.3)
    record(recorder, pixels, 'terminal', .3)
    record(recorder, pixels, 'reset', round_index=1)
    record(recorder, pixels, 'terminal', .1, round_index=1)
    result = recorder.close()
    assert result['status'] == 'complete'
    source = original_execution_source(tmp_path)
    assert source['source_frame_count'] == 5
    assert source['frame_count'] == 6
    assert source['frames'][1]['frame_age_sim_s'] == pytest.approx(.1)
    first = json.loads((recorder.root / 'timeline.jsonl').read_text().splitlines()[0])
    image = first['images']['agentview']
    assert image['pixel_sha256'] == original_hash
    assert np.asarray(Image.open(recorder.root / image['path'])).min() == 23
    (recorder.root / image['path']).write_bytes(b'corrupt')
    with pytest.raises(RuntimeError, match='hash drifted'):
        original_execution_source(tmp_path)


def test_bounded_queue_and_late_encoder_cannot_mutate_finalized_evidence(tmp_path, monkeypatch):
    from ora.environment.controller_replay import (
        ControllerReplayRecorder, load_controller_replay_manifest,
    )

    recorder = make(tmp_path, queue_capacity=4)
    journal = ControllerReplayRecorder(tmp_path / 'native', backend='libero_pro',
        environment_identity={}, camera_panels=['agentview', 'wrist'])
    entered, release = threading.Event(), threading.Event()
    encode = recorder._encode
    def stalled(images):
        entered.set()
        release.wait(3)
        return encode(images)
    monkeypatch.setattr(recorder, '_encode', stalled)
    pixels = np.zeros((2, 2, 3), dtype=np.uint8)
    record(recorder, pixels, 'reset')
    assert entered.wait(1)
    assert record(recorder, pixels, at=.1)
    assert record(recorder, pixels, at=.2)
    assert not record(recorder, pixels, at=.3)
    assert record(recorder, pixels, 'terminal', .4)
    # Current Servo producers persist controller receipts while the
    # independent image encoder is saturated. Numeric evidence must continue.
    for index in range(32):
        journal.record(action_sequence=index,
            action={'action_id': f'burst-{index}', 'action': 'move_pose'},
            controller='fixture-controller', arrays={'commands': [[index, 0.]]}, physics_steps=1)
        assert not record(recorder, pixels, at=.5 + index * .1)
        journal.update(index, execution_status='completed')
    entries = load_controller_replay_manifest(journal.manifest_path)['entries']
    assert len(entries) == 32
    assert all(entry['execution']['status'] == 'completed' for entry in entries)
    assert journal.manifest_path.stat().st_size < 2048
    result = recorder.close(drain_timeout_s=.01)
    assert result['status'] == 'incomplete'
    assert result['dropped_by_reason']['queue_full'] == 33
    assert result['unwritten_frames'] == result['attempted_frames'] == 37
    assert result['max_queue_occupancy'] <= 4
    assert result['error'] == 'writer_drain_timeout'
    before = {p.name: p.read_bytes() for p in recorder.root.iterdir() if p.is_file()}
    release.set()
    recorder._thread.join(2)
    assert not recorder._thread.is_alive()
    assert before == {p.name: p.read_bytes() for p in recorder.root.iterdir() if p.is_file()}
    assert not list((recorder.root / 'frames').iterdir())


def test_missing_terminal_and_disabled_capture(tmp_path):
    recorder = make(tmp_path)
    record(recorder, np.zeros((1, 1, 3), dtype=np.uint8), 'reset')
    assert recorder.close()['status'] == 'incomplete'
    disabled = make(tmp_path / 'disabled', enabled=False)
    assert not record(disabled, None)
    assert disabled.close() is None
    assert not disabled.root.exists()
    assert resolve_execution_capture()['enabled'] is True
    with pytest.raises(ValueError):
        resolve_execution_capture({'fps': True})


def test_scene_tap_preserves_steps_returns_rng_and_restores():
    class Scene:
        def __init__(self): self.steps = 0
        def step(self):
            self.steps += 1
            return self.steps
    original = Scene.step
    scene, other = Scene(), Scene()
    calls = []
    state = np.random.get_state()
    tap = PhysicsStepTap(scene, lambda: calls.append(scene.steps))
    assert scene.step() == 1
    assert other.step() == 1
    assert calls == [1]
    assert np.array_equal(state[1], np.random.get_state()[1])
    tap.close()
    assert Scene.step is original
    tap = PhysicsStepTap(scene, lambda: 1 / 0)
    assert scene.step() == 2
    assert 'ZeroDivisionError' in tap.error
    tap.close()
    assert Scene.step is original


def test_libero_capture_preserves_command_result_and_reset_clock(tmp_path):
    pixels = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    raw = {'agentview_image': pixels, 'robot0_eye_in_hand_image': pixels}
    class Delegate:
        sim = SimpleNamespace(data=SimpleNamespace(time=3.), model=SimpleNamespace(opt=SimpleNamespace(timestep=.01)))
        commands = []
        def step(self, command):
            self.commands.append(command)
            self.sim.data.time += .1
            return raw, 0, False, {'unchanged': True}
        def reset(self): return raw
    delegate = Delegate()
    wrapped = _CameraMetadataEnvironment(delegate)
    wrapped.harness_execution_capture = {'op': 'init', 'output_root': str(tmp_path / 'native'), 'config': {}, 'identity': {}}
    wrapped.harness_execution_capture = {'op': 'reset', 'round_index': 0, 'raw_images': raw}
    wrapped.harness_execution_capture = {'op': 'action_start', 'context': {'action_id': 'a1'}}
    command = np.zeros(7)
    result = wrapped.step(command)
    assert delegate.commands[0] is command
    assert result[0] is raw and result[3] == {'unchanged': True}
    wrapped.harness_execution_capture = {'op': 'terminal'}
    wrapped.reset()
    wrapped.step(command)  # reset warmup is deliberately outside captured rounds
    wrapped.harness_execution_capture = {'op': 'reset', 'round_index': 1, 'raw_images': raw}
    wrapped.harness_execution_capture = {'op': 'terminal'}
    wrapped.harness_execution_capture = {'op': 'close'}
    rows = [json.loads(x) for x in (tmp_path / 'native/execution-capture/timeline.jsonl').read_text().splitlines()]
    control = [x for x in rows if x['phase'] == 'control']
    assert len(control) == 1
    assert control[0]['sim_step'] == 10 and control[0]['action_id'] == 'a1'
    assert [x['sim_time_s'] for x in rows if x['phase'] == 'reset'] == [0, 0]
    first = rows[0]['images']['agentview']
    assert first['pixel_sha256'] == hashlib.sha256(pixels[::-1].tobytes()).hexdigest()


def test_libero_replay_same_mode_retains_controller_state(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, 'robosuite.controllers', SimpleNamespace(load_controller_config=lambda **kw: kw))
    robot = SimpleNamespace(action_dim=8, _load_controller=lambda: calls.append('reload'))
    wrapped = _CameraMetadataEnvironment(SimpleNamespace(robots=[robot], _action_dim=7))
    wrapped.harness_replay_controller_mode = 'OSC_POSE'
    wrapped.harness_replay_controller_mode = 'OSC_POSE'
    assert calls == []
    wrapped.harness_replay_controller_mode = 'JOINT_POSITION'
    wrapped.harness_replay_controller_mode = 'JOINT_POSITION'
    wrapped.harness_replay_controller_mode = 'OSC_POSE'
    assert calls == ['reload', 'reload']


@pytest.mark.parametrize("has_policy_steps", [True, False])
def test_original_capture_compiles_video_without_simulator(tmp_path, monkeypatch, has_policy_steps):
    from viewer.trajectory import compile_trajectory_video
    run = tmp_path / 'runs/capture-video'
    (run / 'policy-state/steps').mkdir(parents=True)
    (run / 'plan.json').write_text(json.dumps({'experiment': 'test', 'task': 'test', 'runtime': {'kind': 'libero_pro'}, 'source': {'commit': 'test'}}))
    (run / 'policy-state/steps/step-0000.json').write_text(json.dumps({'step': 0, 'cause': 'reset', 'observation': {'id': 'o'}, 'cameras': {}}))
    (run / 'policy-state/manifest.json').write_text(json.dumps({'steps': [{'step': 0, 'path': 'policy-state/steps/step-0000.json'}]}))
    if not has_policy_steps:
        (run / "policy-state/manifest.json").unlink()
    recorder = make(run)
    pixels = np.zeros((4, 4, 3), dtype=np.uint8)
    record(recorder, pixels, 'reset')
    record(recorder, pixels, 'terminal', .2)
    recorder.close()
    monkeypatch.setattr('viewer.trajectory._controller_replay_source', lambda *_a, **_kw: pytest.fail('simulator reconstruction requested'))
    video, manifest = compile_trajectory_video(run)
    assert video.is_file()
    assert manifest['visual_fidelity'] == 'original_frames_only'
    assert manifest['source_frame_count'] == 2
    assert manifest['capture_status'] == 'complete'


def test_libero_media_failure_stays_inside_worker(tmp_path):
    wrapped = _CameraMetadataEnvironment(SimpleNamespace())
    wrapped.harness_execution_capture = {'op': 'init', 'output_root': str(tmp_path), 'identity': {}, 'config': {'fps': -1}}
    assert wrapped.harness_execution_capture['initialized'] is False
    assert 'ValueError' in wrapped.harness_execution_capture['error']
    wrapped.harness_execution_capture = {'op': 'close'}


def test_libero_close_waits_for_worker_acknowledgement(tmp_path):
    from ora.environment.backends.libero_pro.session import LiberoProSession
    calls = []
    vector = SimpleNamespace(set_env_attr=lambda key,value: calls.append(('send', value['op'])),
        get_env_attr=lambda key: (calls.append(('ack', key)) or [{'initialized': True, 'finalized': True, 'error': None}]))
    session = object.__new__(LiberoProSession)
    session.execution_capture = {'enabled': True}
    session.environment = SimpleNamespace(env=vector)
    session.output_root = tmp_path
    session._capture_command('close')
    assert calls == [('send', 'close'), ('ack', 'harness_execution_capture')]
    assert not (tmp_path / 'execution-capture-failure.json').exists()


def test_premature_verdict_does_not_create_terminal_capture():
    from ora.environment.backends.robotwin.session import RoboTwinLiveSession, RoboTwinLiveError
    calls = []
    session = SimpleNamespace(_require_environment=lambda: object(), status=lambda: {'terminal': False},
                              _capture_execution_boundary=calls.append)
    with pytest.raises(RoboTwinLiveError, match='before episode termination'):
        RoboTwinLiveSession.persist_official_verdict(session)
    assert calls == []


@pytest.mark.parametrize('backend', ['robotwin', 'libero_pro'])
def test_fresh_consumers_capture_by_default_and_preserve_explicit_disable(backend):
    from pathlib import Path
    from test_native_baseline import native_spec
    from ora.orchestrator.spec import resolve_run_spec

    spec = native_spec(backend)
    root = Path(__file__).resolve().parents[1]
    assert resolve_run_spec(spec, root=root)['runtime']['execution_capture'] == resolve_execution_capture()
    spec['runtime']['execution_capture'] = {'enabled': False}
    assert resolve_run_spec(spec, root=root)['runtime']['execution_capture']['enabled'] is False


def test_libero_missing_finalize_ack_is_persisted_as_auxiliary_failure(tmp_path):
    from ora.environment.backends.libero_pro.session import LiberoProSession
    session = object.__new__(LiberoProSession)
    session.execution_capture = {'enabled': True}
    session.environment = SimpleNamespace(env=SimpleNamespace(set_env_attr=lambda *a: None,
        get_env_attr=lambda name: [{'initialized': True, 'finalized': False, 'error': None}]))
    session.output_root = tmp_path
    session._capture_command('close')
    failure = json.loads((tmp_path / 'execution-capture-failure.json').read_text())
    assert failure['component'] == 'original_execution_capture'
    assert 'did not finalize' in failure['error']


@pytest.mark.parametrize('backend', ['robotwin', 'libero_pro'])
@pytest.mark.parametrize('enabled', [True, False])
def test_capture_configuration_reaches_isolated_backend_constructor(tmp_path, monkeypatch, backend, enabled):
    from ora.environment import factory, worker

    capture = resolve_execution_capture({'enabled': enabled, 'fps': 6, 'queue_capacity': 8})
    runtime = dict(kind=backend, execution_capture=capture, startup_timeout_s=1,
        request_timeout_s=1, task_config='demo_randomized', required_revision='source',
        workspace_bounds_m={'x': [-1, 1], 'y': [-1, 1], 'z': [0, 2]},
        suite='libero_spatial', task_id=0, expected_task_language='fixture',
        required_libero_pro_revision='libero-source',
        required_data_revision='data-source', servo_max_steps=80)
    context = factory._BuildContext(runtime=runtime, output_root=tmp_path / 'native',
        task='fixture-task', seed=0, max_sim_steps=10, redacted_environment_names=(),
        qualified_runtime_binding=None, action_codec='runtime_action_v3')
    monkeypatch.setattr(factory, '_required_environment_path', lambda *a, **kw: tmp_path)
    monkeypatch.setattr(factory, 'LocalRpcEnvironmentSession', lambda **kwargs: SimpleNamespace(**kwargs))
    rpc = (factory._robotwin if backend == 'robotwin' else factory._libero)(context)
    config = {**rpc.worker_config, 'shared_root': str(tmp_path)}
    path = f'ora.environment.backends.{backend}.session.'
    name = 'RoboTwinLiveSession' if backend == 'robotwin' else 'LiberoProSession'
    monkeypatch.setattr(path + name, lambda **kwargs: kwargs)
    constructed = worker._build_session(config)
    assert constructed['execution_capture'] == capture


@pytest.mark.parametrize('enabled', [True, False])
def test_robodojo_capture_default_and_isolated_config_binding(tmp_path, monkeypatch, enabled):
    from pathlib import Path
    from test_robodojo_b2_parity import spec_for
    from ora.orchestrator.spec import resolve_run_spec
    from ora.environment import factory, worker

    spec = spec_for('insert_tubes')
    root = Path(__file__).resolve().parents[1]
    assert resolve_run_spec(spec, root=root)['runtime']['execution_capture'] == resolve_execution_capture()
    spec['runtime']['execution_capture'] = {'enabled': enabled, 'fps': 6}
    runtime = resolve_run_spec(spec, root=root)['runtime']
    monkeypatch.setattr(factory, '_required_environment_path', lambda *a, **kw: tmp_path / 'bin/python')
    monkeypatch.setattr(factory, 'LocalRpcEnvironmentSession', lambda **kwargs: SimpleNamespace(**kwargs))
    rpc = factory.build_robodojo_worker_session(runtime=runtime, output_root=tmp_path / 'native',
        task='insert_tubes', seed=0, max_sim_steps=5000, action_codec=spec['policy']['action_codec'])
    config = {**rpc.worker_config, 'shared_root': str(tmp_path)}
    monkeypatch.setattr('ora.environment.backends.robodojo.session.RobodojoSession',
                        lambda **kwargs: kwargs)
    assert worker._build_session(config)['config']['execution_capture'] == resolve_execution_capture(
        {'enabled': enabled, 'fps': 6})


@pytest.mark.parametrize('enabled', [True, False])
def test_robodojo_real_tick_samples_without_extra_physics_and_finalizes_rounds(tmp_path, monkeypatch, enabled):
    from ora.environment.backends.robodojo import bridge
    from ora.environment.backends.robodojo.session import RobodojoSession, CAMERAS

    class Parent:
        def step(self, **kwargs):
            pass

    class Plant(bridge.EvaluationLifecycle, Parent):
        def __init__(self):
            self.steps = 500
            self.take_action_cnt = [0]
            self.step_lim = 100
            self.reward_registered = True
            self.end_flag = [False]
            self._official_terminal = None
            self.unstable_envs = set()
            self.rewards = 0
            self.pending = []
            self.robot_manager = SimpleNamespace(control_manager=SimpleNamespace(
                prev_control=[{}], push=self.push, pop=lambda _: [self.pending.pop(0)],
                get_empty=lambda _: not self.pending))
            self.reward_manager = SimpleNamespace(step=self.reward_step)
            self.sim = SimpleNamespace(sim=SimpleNamespace(current_time_step_index=self.steps))

        def push(self, ids, commands):
            self.pending.extend(commands[0])

        def interpolate(self, control):
            return [control] * 10

        def sim_step(self, **kwargs):
            self.steps += 1
            self.sim.sim.current_time_step_index = self.steps

        def reward_step(self, **kwargs):
            self.rewards += 1

        def advance_official_terminal(self):
            return {'official_success': False}

        def get_obs(self):
            return {'vision': {source: {'color': np.full((8, 10, 3), self.steps % 255, dtype=np.uint8)}
                               for source in CAMERAS.values()}}

    plant = Plant()
    session = RobodojoSession(config={'execution_capture': {'enabled': enabled, 'fps': 30}}, output_root=tmp_path)
    session.environment = plant
    session.reset_physics_steps = plant.steps
    session.source_hash = 'a' * 64
    monkeypatch.setattr(bridge, '_fixed_meta_control', lambda command, *a: (command, command))
    session._execute_without_capture = lambda payload: plant.native_tick({'left_arm': {'position': [1]}})
    session._start_execution_capture()
    try:
        assert session.execute({'action_id': 'a1', 'action': 'move_pose'}) == {'official_success': False}
        assert plant.steps == 510 and plant.take_action_cnt == [1] and plant.rewards == 1
        session._capture_execution_boundary('terminal')
        session.round_index = 1
        session.reset_physics_steps = plant.steps
        session._start_execution_capture()
        session.execute({'action_id': 'a2', 'action': 'move_pose'})
        session._capture_execution_boundary('terminal')
    finally:
        session.finalize_operator_video()
    assert plant.steps == 520 and plant.rewards == 2
    assert plant.execution_capture_callback is None
    if not enabled:
        assert not (tmp_path / 'execution-capture').exists()
        return
    rows = [json.loads(line) for line in (tmp_path / 'execution-capture/timeline.jsonl').read_text().splitlines()]
    controls = [row for row in rows if row['phase'] == 'control']
    assert {row['round_index'] for row in controls} == {0, 1}
    assert {row['action_id'] for row in controls} == {'a1', 'a2'}
    assert all(0 < row['sim_step'] <= 10 for row in controls)
    assert all(row['action_id'] is None for row in rows if row['phase'] == 'reset')
    for row in rows:
        if row['phase'] == 'control':
            for frame in row['images'].values():
                pixels = np.asarray(Image.open(tmp_path / 'execution-capture' / frame['path']))
                assert np.all(pixels == (500 + row['round_index'] * 10 + row['sim_step']) % 255)
    before = (tmp_path / 'execution-capture/timeline.jsonl').read_bytes()
    session.finalize_operator_video()
    assert (tmp_path / 'execution-capture/timeline.jsonl').read_bytes() == before
