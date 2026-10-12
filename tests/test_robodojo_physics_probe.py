"""The qualification-only RoboDojo physics scene probe (CPU-physics lane).

The worker is driven with a fake session and fake scene readers, and the host
judges synthetic samples: conveyor objects at belt height and advancing,
articulations finite and holding, fluid particles present and finite, and no
object below its support. Nothing here starts Isaac or claims a real lease.
"""
from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from ora.environment.backends.robodojo import physics_scene_probe as worker
from ora.environment.benchmarks.robodojo import NATIVE_ACTION_LIMITS
from ora.orchestrator import robodojo_physics_probe as probe

ROOT = Path(__file__).resolve().parents[1]
BELT = {'path': '/World/envs/env_0/dynamic/conveyor/conveyor_0_6/ConveyorTrack/Belt',
        'surface_velocity': [-0.1, 0.0, 0.0], 'world_aabb_min': [-2.078, -0.275, 0.7466],
        'world_aabb_max': [1.8493, 0.625, 0.7622]}


class FakeSession:
    def __init__(self, *, terminal_after=None, fail_start=False):
        self.ticks, self.closed, self.restarts = 0, False, []
        self.terminal_after, self.fail_start = terminal_after, fail_start
        self.sim_steps = 0
        self.environment = SimpleNamespace(take_action_cnt=[0])
        self.limit_seen = None

    def start(self):
        if self.fail_start:
            raise RuntimeError('boot failed')
        self.limit_seen = dict(NATIVE_ACTION_LIMITS)
        return {'native_action_limit': 400}

    @property
    def is_terminal(self):
        return self.terminal_after is not None and self.ticks >= self.terminal_after

    def _conformance_tick(self, control):
        assert control == {}
        self.ticks += 1
        self.sim_steps += 10
        self.environment.take_action_cnt[0] += 1

    def restart_round(self, index, *, seed=None):
        self.restarts.append((index, seed))
        return {'round_index': index}

    def close(self):
        self.closed = True


def _readers(session_ticks=lambda s: s.ticks):
    return worker.Readers(
        physics_mode=lambda s: {'isaaclab_sim_device': 'cpu', 'use_fabric': False, 'suppress_readback': False},
        objects=lambda s: {'box_0_1': {'kind': 'rigid', 'position_m': [0.004 * session_ticks(s), 0.0, 0.773]}},
        articulations=lambda s: {'toaster_0_2': {'dof_names': ['lever'], 'joint_positions': [0.01]}},
        fluids=lambda s: {},
        belts=lambda s: [BELT],
        rigid_flags=lambda s: {'box_0_1': {'disable_gravity': [s.ticks > 1000], 'collision_enabled': [True]}},
    )


# --- worker --------------------------------------------------------------------------------------

def test_worker_arguments_require_paths_and_positive_counts():
    parsed = worker.parse_arguments(['--config', 'c.json', '--output', 'o', '--ticks', '7', '--sample-every', '2'])
    assert (parsed.config, parsed.output, parsed.ticks, parsed.sample_every) == (Path('c.json'), Path('o'), 7, 2)
    assert parsed.probe_only_native_action_limit is None and parsed.in_place_restart is False
    for bad in (['--ticks', '0'], ['--ticks', '3', '--sample-every', '0']):
        with pytest.raises(SystemExit):
            worker.parse_arguments(['--config', 'c.json', '--output', 'o', *bad])


def test_the_worker_holds_samples_and_closes(tmp_path):
    session = FakeSession()
    report = worker.run_worker({'task': 'press_by_number', 'output_root': str(tmp_path / 'environment')},
                               tmp_path, ticks=7, sample_every=3, session_factory=lambda **_: session,
                               readers=_readers())
    assert report['error'] is None and report['completed'] is True and session.closed
    assert session.ticks == report['ticks_held'] == 7 and report['ticks_requested'] == 7
    assert [row['tick'] for row in report['samples']] == [0, 3, 6, 7]
    assert [row['native_action'] for row in report['samples']] == [0, 3, 6, 7]
    assert report['physics_mode']['isaaclab_sim_device'] == 'cpu'
    assert report['belts'] == [BELT]
    assert report['samples'][-1]['objects']['box_0_1']['position_m'][0] == pytest.approx(0.028)
    assert report['visibility'] == 'qualification_only_never_policy_visible'
    assert json.loads((tmp_path / worker.WORKER_REPORT_NAME).read_text())['ticks_held'] == 7


def test_an_early_terminal_stops_the_hold_and_is_recorded(tmp_path):
    session = FakeSession(terminal_after=4)
    report = worker.run_worker({'output_root': str(tmp_path / 'e')}, tmp_path, ticks=10, sample_every=5,
                               session_factory=lambda **_: session, readers=_readers())
    assert report['ticks_held'] == 4 and report['terminal_before_hold_end'] is True
    assert report['samples'][-1]['tick'] == 4 and report['completed'] is True


def test_a_reader_failure_is_evidence_and_a_boot_failure_is_an_error(tmp_path):
    readers = _readers()
    broken = worker.Readers(**{**readers.__dict__, 'fluids': lambda s: (_ for _ in ()).throw(KeyError('fluid'))})
    report = worker.run_worker({'output_root': str(tmp_path / 'a')}, tmp_path / 'a', ticks=2, sample_every=1,
                               session_factory=lambda **_: FakeSession(), readers=broken)
    assert report['error'] is None and report['samples'][0]['fluids'] == {'error': "KeyError: 'fluid'"}
    session = FakeSession(fail_start=True)
    report = worker.run_worker({'output_root': str(tmp_path / 'b')}, tmp_path / 'b', ticks=2, sample_every=1,
                               session_factory=lambda **_: session, readers=_readers())
    assert report['error'] == 'RuntimeError: boot failed' and report['completed'] is False and session.closed


#: An upstream task module the Runtime does not admit (its upstream step_lim is 400). The two
#: fluid tasks that used this path are admitted since 2026-10-04, so the mechanism is exercised
#: on an unadmitted name.
UNADMITTED = 'pour_liquid_into_cup_random'


def test_a_probe_only_limit_exists_only_inside_the_worker_run(tmp_path):
    session = FakeSession()
    assert UNADMITTED not in NATIVE_ACTION_LIMITS
    report = worker.run_worker({'task': UNADMITTED, 'output_root': str(tmp_path / 'e')}, tmp_path,
                               ticks=1, sample_every=1, probe_only_native_action_limit=400,
                               session_factory=lambda **_: session, readers=_readers())
    assert session.limit_seen[UNADMITTED] == 400
    assert UNADMITTED not in NATIVE_ACTION_LIMITS
    assert report['probe_only_native_action_limit'] == {'task': UNADMITTED, 'limit': 400,
                                                        'admission': 'unadmitted; qualification probe only'}
    # An admitted task, the fluid two included, takes no probe-only limit.
    with pytest.raises(ValueError, match='only to an unadmitted task'):
        worker.run_worker({'task': 'pour_liquid_into_cup', 'output_root': str(tmp_path / 'f')}, tmp_path / 'f',
                          ticks=1, sample_every=1, probe_only_native_action_limit=400,
                          session_factory=lambda **_: FakeSession(), readers=_readers())


def test_the_in_place_restart_diagnostic_reads_object_physics_flags_before_and_after(tmp_path):
    session = FakeSession()
    report = worker.run_worker({'output_root': str(tmp_path / 'e')}, tmp_path, ticks=2, sample_every=1,
                               in_place_restart=True, session_factory=lambda **_: session, readers=_readers())
    restart = report['in_place_restart']
    assert session.restarts == [(1, 0)]
    assert restart['before']['box_0_1'] == {'disable_gravity': [False], 'collision_enabled': [True]}
    assert set(restart) >= {'before', 'after', 'samples_after', 'claim_boundary'}


def test_the_worker_module_stays_outside_the_worker_kernel_and_its_import_closure():
    from ora.orchestrator.robodojo_kernel import ROBODOJO_MOTION_KERNEL_PATHS, WORKER_KERNEL_PATHS
    from ora.orchestrator.robodojo_kernel_review import worker_import_closure

    assert probe.WORKER_SOURCE not in WORKER_KERNEL_PATHS and probe.WORKER_SOURCE not in ROBODOJO_MOTION_KERNEL_PATHS
    closure = worker_import_closure()
    assert probe.WORKER_SOURCE not in closure
    assert not any(probe.WORKER_SOURCE in targets for targets in closure.values())
    assert (ROOT / probe.WORKER_SOURCE).is_file() and probe.WORKER_MODULE == worker.__name__


class ConfigSequence(Sequence):
    """Stand-in for OmegaConf's ListConfig: a Sequence that is no list or tuple and has no tolist()."""

    def __init__(self, items):
        self._items = list(items)

    def __getitem__(self, index):
        return self._items[index]

    def __len__(self):
        return len(self._items)


class FakeFluid:
    def __init__(self, rows, init_pos, init_ori=(1.0, 0.0, 0.0, 0.0)):
        self.rows = rows
        self.init_pos, self.init_ori = ConfigSequence(init_pos), ConfigSequence(init_ori)

    def get_particle_positions(self):
        return self.rows, None, None


class FakeLayout:
    """The four layout-manager calls the official fluid predicate makes."""

    def __init__(self, records, poses, boxes):
        self.records, self.poses, self.boxes = records, poses, boxes

    def get_layout_records(self, env_idx, object_type):
        assert env_idx == 0
        return self.records.get(object_type, [])

    def get_instance_pose(self, env_idx, label=None, inst_name=None, relative=True):
        assert env_idx == 0 and relative is True
        return self.poses[inst_name]

    def get_instance_bbox_vertices(self, inst_name, env_idx):
        return self.boxes[inst_name]


ROWS = [[0.0, 0.01, -0.05], [0.002, -0.003, 0.03]]


def _fluid_session(fluids):
    layout = FakeLayout(
        records={'Rigid': [{'label': 'bottle', 'inst_name': 'wuliangye_0_1'}, {'label': 'cup', 'inst_name': 'mug_16_2'}],
                 'Geometry': [{'label': 'camera_stand', 'inst_name': 'camera_stand_0_3'}],
                 'Fluid': [{'label': 'wine', 'inst_name': 'wuliangye_0_4'}] if fluids else []},
        poses={'wuliangye_0_1': ([-0.1, -0.19, 0.825], ConfigSequence([1.0, 0.0, 0.0, 0.0])),
               'mug_16_2': ([-0.01, -0.01, 0.803], [1.0, 0.0, 0.0, 0.0]),
               'camera_stand_0_3': ([0.0, -0.47, 0.765], [0.707, -0.707, 0.0, 0.0])},
        boxes={'wuliangye_0_1': [[-0.04, -0.04, -0.06], [0.04, 0.04, 0.16]],
               'mug_16_2': [[-0.05, -0.03, -0.04], [0.05, 0.03, 0.04]], 'camera_stand_0_3': None})
    scene = SimpleNamespace(_fluid_objects=[fluids], env_origins=[[0.0, 0.0, 0.0]], layout_manager=layout)
    return SimpleNamespace(environment=SimpleNamespace(scene_manager=scene))


def test_floats_accept_any_non_string_sequence_such_as_an_omegaconf_list():
    assert worker._floats(ConfigSequence([0.25, ConfigSequence([0.5, 1])])) == [0.25, 0.5, 1.0]
    assert worker._floats((1, [2.0, (3,)])) == [1.0, 2.0, 3.0]


class GfVector:
    """Indexes like pxr ``Gf.Vec3f`` without registering as a ``Sequence``; ``float()`` refuses it."""

    def __init__(self, *items):
        self._items = items

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        return self._items[index]


def test_floats_accept_a_gf_vector_such_as_a_belt_surface_velocity():
    # 2026-10-05: float() on the belt's Vec3f velocity failed, so both
    # conveyor cells reported no belt prim.
    assert worker._floats(GfVector(-0.1, -0.0, -0.0)) == [-0.1, -0.0, -0.0]
    assert worker._floats([GfVector(1, 2), 3]) == [1.0, 2.0, 3.0]


def test_the_fluid_reader_takes_the_config_sequence_init_pose_upstream_stores():
    session = _fluid_session({'wuliangye_0_4': FakeFluid(ROWS, init_pos=[-0.0997, -0.1865, 0.8508])})
    assert worker.fluid_particles(session) == {'wuliangye_0_4': {
        'count': 2, 'finite': True, 'init_position_m': [-0.0997, -0.1865, 0.8508],
        'local_min_m': [0.0, -0.003, -0.05], 'local_max_m': [0.002, 0.01, 0.03]}}


def test_the_fluid_evaluator_inputs_are_what_the_official_predicate_reads():
    session = _fluid_session({'wuliangye_0_4': FakeFluid(ROWS, init_pos=[-0.0997, -0.1865, 0.8508])})
    inputs = worker.fluid_evaluator_inputs(session)
    assert inputs['env_origin_m'] == [0.0, 0.0, 0.0]
    assert inputs['fluids'] == {'wine': {'inst_name': 'wuliangye_0_4', 'local_positions_m': ROWS,
                                         'init_position_m': [-0.0997, -0.1865, 0.8508],
                                         'init_orientation_wxyz': [1.0, 0.0, 0.0, 0.0]}}
    assert inputs['containers']['cup'] == {'inst_name': 'mug_16_2', 'kind': 'Rigid',
                                           'position_m': [-0.01, -0.01, 0.803],
                                           'orientation_wxyz': [1.0, 0.0, 0.0, 0.0],
                                           'bbox_vertices_m': [[-0.05, -0.03, -0.04], [0.05, 0.03, 0.04]]}
    assert inputs['containers']['bottle']['orientation_wxyz'] == [1.0, 0.0, 0.0, 0.0]
    assert inputs['containers']['camera_stand']['bbox_vertices_m'] is None
    # A scene without a liquid records nothing.
    assert worker.fluid_evaluator_inputs(_fluid_session({})) == {}


def test_the_worker_records_fluid_evaluator_inputs_at_reset_and_at_the_end(tmp_path):
    readers = worker.Readers(**{**_readers().__dict__, 'fluid_inputs': lambda s: {'tick': s.ticks}})
    report = worker.run_worker({'output_root': str(tmp_path / 'e')}, tmp_path, ticks=3, sample_every=1,
                               session_factory=lambda **_: FakeSession(), readers=readers)
    assert report['fluid_evaluator_inputs'] == {'reset': {'tick': 0}, 'end': {'tick': 3}}
    assert worker.default_readers().fluid_inputs is worker.fluid_evaluator_inputs
    plain = worker.run_worker({'output_root': str(tmp_path / 'p')}, tmp_path / 'p', ticks=1, sample_every=1,
                              session_factory=lambda **_: FakeSession(), readers=_readers())
    assert 'fluid_evaluator_inputs' not in plain


# --- host judgement --------------------------------------------------------------------------------

def _sample(tick, objects=None, articulations=None, fluids=None):
    return {'tick': tick, 'native_action': tick, 'objects': objects or {}, 'articulations': articulations or {},
            'fluids': fluids or {}}


def _worker(samples, *, mode=None, belts=(), error=None, held=None):
    return {'schema': worker.WORKER_REPORT_SCHEMA, 'error': error, 'completed': error is None,
            'ticks_requested': samples[-1]['tick'], 'ticks_held': samples[-1]['tick'] if held is None else held,
            'terminal_before_hold_end': False,
            'physics_mode': mode or {'isaaclab_sim_device': 'cpu', 'use_fabric': False, 'suppress_readback': False},
            'belts': list(belts), 'samples': samples}


def _belt_objects(speed, z=0.7734):
    return [_sample(t, {'game_machine_0_2': {'kind': 'rigid', 'position_m': [0.45 + speed * t, 0.04, z]},
                        'conveyor_0_6': {'kind': 'dynamic', 'position_m': [0.0, 0.0, 0.5]},
                        'Ground_0_9': {'kind': 'ground', 'position_m': [0.0, 0.0, -5.0 * t]}})
            for t in (0, 50, 100)]


def test_a_moving_conveyor_passes_and_a_stalled_or_sunk_one_fails():
    passed = probe.judge_cell('pick_from_conveyor_by_image', _worker(_belt_objects(0.004), belts=[BELT]),
                              declared_device='cpu')
    assert passed['scene_kind'] == 'conveyor' and passed['passed'], passed['failures']
    assert set(passed['checks']) == {'worker_completed', 'physics_mode', 'supported', 'conveyor'}
    speed = passed['checks']['conveyor']['objects']['game_machine_0_2']['speed_m_per_action']
    assert speed == pytest.approx(0.004)
    stalled = probe.judge_cell('pick_from_conveyor_by_image', _worker(_belt_objects(0.0), belts=[BELT]),
                               declared_device='cpu')
    assert not stalled['passed'] and stalled['checks']['conveyor']['passed'] is False
    no_belt = probe.judge_cell('pick_from_conveyor_by_image', _worker(_belt_objects(0.004)), declared_device='cpu')
    assert 'no surface-velocity belt' in ' '.join(no_belt['failures'])
    sunk = _belt_objects(0.004)
    sunk[-1]['objects']['game_machine_0_2']['position_m'][2] = 0.70
    result = probe.judge_cell('pick_from_conveyor_by_image', _worker(sunk, belts=[BELT]), declared_device='cpu')
    assert result['checks']['supported']['passed'] is False and result['checks']['conveyor']['passed'] is False


def test_an_articulation_must_stay_finite_and_hold_its_joints():
    def run(values):
        return probe.judge_cell('make_toast', _worker([
            _sample(t, {'toaster_0_1': {'kind': 'articulation', 'position_m': [0.1, 0.2, 0.8]}},
                    {'toaster_0_1': {'dof_names': ['lever', 'knob'], 'joint_positions': value}})
            for t, value in zip((0, 25, 50), values)]), declared_device='cpu')

    held = run(([0.0, 0.1], [0.001, 0.1], [0.002, 0.1]))
    assert held['scene_kind'] == 'articulation' and held['passed'], held['failures']
    assert held['checks']['articulation']['articulations']['toaster_0_1']['max_drift'] == pytest.approx(0.002)
    assert not run(([0.0, 0.1], [0.3, 0.1], [0.6, 0.1]))['passed']
    assert not run(([0.0, 0.1], [math.nan, 0.1], [0.0, 0.1]))['passed']
    empty = probe.judge_cell('make_toast', _worker([_sample(0), _sample(50)]), declared_device='cpu')
    assert 'no articulation' in ' '.join(empty['failures'])


def test_fluid_particles_must_exist_stay_finite_and_keep_their_count():
    def run(rows):
        return probe.judge_cell('pour_liquid_into_cup', _worker([
            _sample(t, fluids={'wuliangye_0_3': {'count': count, 'finite': finite}})
            for t, (count, finite) in zip((0, 25, 50), rows)]), declared_device='cpu')

    assert run(((5120, True),) * 3)['passed']
    assert run(((5120, True),) * 3)['scene_kind'] == 'fluid'
    assert not run(((0, True),) * 3)['passed']
    assert not run(((5120, True), (5000, True), (5120, True)))['passed']
    assert not run(((5120, True), (5120, False), (5120, True)))['passed']
    assert not probe.judge_cell('pour_liquid_into_cup', _worker([_sample(0), _sample(5)]), declared_device='cpu')['passed']


def test_the_physics_mode_must_be_the_declared_one_without_fabric():
    samples = [_sample(0), _sample(50)]
    ok = probe.judge_cell('stack_blocks', _worker(samples, mode={'isaaclab_sim_device': 'cuda:0'}),
                          declared_device='cuda:0')
    assert ok['scene_kind'] == 'rigid' and ok['passed'], ok['failures']
    for mode in ({'isaaclab_sim_device': 'cuda:0'}, {'isaaclab_sim_device': 'cpu', 'use_fabric': True},
                 {'isaaclab_sim_device': 'cpu', 'suppress_readback': True}, {'error': 'no readback'}):
        result = probe.judge_cell('stack_blocks', _worker(samples, mode=mode), declared_device='cpu')
        assert result['checks']['physics_mode']['passed'] is False and not result['passed']


def test_an_object_may_not_fall_below_its_support_or_vanish():
    def run(last):
        return probe.judge_cell('stack_blocks', _worker([
            _sample(0, {'block_0_1': {'kind': 'rigid', 'position_m': [0.1, 0.1, 0.78]}}),
            _sample(50, last)], mode={'isaaclab_sim_device': 'cuda:0'}), declared_device='cuda:0')

    assert run({'block_0_1': {'kind': 'rigid', 'position_m': [0.1, 0.1, 0.775]}})['passed']
    assert not run({'block_0_1': {'kind': 'rigid', 'position_m': [0.1, 0.1, 0.70]}})['passed']
    assert not run({'block_0_1': {'kind': 'rigid', 'position_m': [0.1, 0.1, -0.5]}})['passed']
    assert not run({})['passed']
    assert not run({'block_0_1': {'kind': 'rigid', 'position_m': [0.1, math.inf, 0.78]}})['passed']


def test_a_worker_error_or_a_short_hold_fails_the_cell():
    samples = [_sample(0), _sample(50)]
    mode = {'isaaclab_sim_device': 'cuda:0'}
    assert not probe.judge_cell('stack_blocks', _worker(samples, mode=mode, error='RuntimeError: x'),
                                declared_device='cuda:0')['passed']
    assert not probe.judge_cell('stack_blocks', _worker(samples, mode=mode, held=10),
                                declared_device='cuda:0')['passed']
    assert not probe.judge_cell('stack_blocks', None, declared_device='cuda:0')['passed']


# --- host launcher -----------------------------------------------------------------------------------

def test_cells_pair_a_task_with_a_slot_and_run_in_parallel():
    arguments = probe.parse_arguments(['--cell', 'make_toast:robodojo-gpu4-d1', '--cell',
                                       'pick_from_conveyor_by_image:robodojo-gpu5-d2', '--output', 'out'])
    assert arguments.cell == [('make_toast', 'robodojo-gpu4-d1'), ('pick_from_conveyor_by_image', 'robodojo-gpu5-d2')]
    assert arguments.ticks is None and arguments.sample_every == 5 and arguments.physics_device is None
    for bad in (['--cell', 'make_toast'], ['--cell', 'not_a_task:robodojo-gpu4-d1'], []):
        with pytest.raises(SystemExit):
            probe.parse_arguments([*bad, '--output', 'out'])
    assert probe.parse_arguments(['--output', 'out', '--report-only']).cell == []
    assert probe.hold_ticks('match_and_pick_from_conveyor', None) == 100
    assert probe.hold_ticks('make_toast', None) == probe.hold_ticks('pour_by_language', None) == 50
    assert probe.hold_ticks('make_toast', 7) == 7
    assert probe.cell_directory('make_toast', 'robodojo-gpu4-d1') == 'make_toast--robodojo-gpu4-d1'


@pytest.fixture
def fixture_assets(monkeypatch, tmp_path):
    folder = tmp_path / 'Assets/Eval_Layout/RoboDojo/arx_x5/0'
    folder.mkdir(parents=True)
    for task in (*NATIVE_ACTION_LIMITS, *probe.PROBE_ONLY_NATIVE_ACTION_LIMITS):
        (folder / f'{task}_0.json').write_text(json.dumps({'task': task}))
    monkeypatch.setenv('ROBODOJO_ASSET_ROOT', str(tmp_path / 'Assets'))
    monkeypatch.setenv('ORA_EXECUTION_HOST', 'test-host')


def test_the_probe_runtime_is_the_tasks_default_on_the_leased_slot(fixture_assets):
    runtime = probe.probe_runtime('press_by_number', 'robodojo-gpu5-d2')
    assert (runtime['physics_device'], runtime['cuda_visible_devices'], runtime['renderer_gpu']) == ('cpu', '5', 5)
    assert runtime['resource_binding']['profile_id'] == 'robodojo-gpu5-d2'
    assert runtime['execution_capture'] == {'enabled': False}
    assert 'infrastructure_qualification' not in runtime
    assert probe.probe_runtime('stack_blocks', 'robodojo-gpu4-d1')['physics_device'] == 'cuda:0'
    # A comparison override is explicit and recorded by the launcher.
    assert probe.probe_runtime('pick_from_conveyor_by_image', 'robodojo-gpu4-d1',
                               physics_device='cuda:0')['physics_device'] == 'cuda:0'


def test_the_admitted_fluid_tasks_probe_on_their_own_limit_without_probe_only_admission(fixture_assets):
    """Admitted on 2026-10-04 (0059 pour admission): the probe-only table no longer holds them."""
    assert probe.PROBE_ONLY_NATIVE_ACTION_LIMITS == {}
    for task, limit in (('pour_by_language', 800), ('pour_liquid_into_cup', 400)):
        assert NATIVE_ACTION_LIMITS[task] == limit and probe._probe_only_limit(task) is None
        assert probe.parse_cell(f'{task}:robodojo-gpu4-d3') == (task, 'robodojo-gpu4-d3')
        runtime = probe.probe_runtime(task, 'robodojo-gpu4-d3')
        assert (runtime['task_variant'], runtime['native_action_limit'], runtime['physics_device']) == (
            task, limit, 'cpu')


def test_an_unadmitted_task_gets_its_upstream_limit_for_the_probe_only(monkeypatch, fixture_assets):
    """The mechanism stays for the next unadmitted scene; its limit exists inside the probe only."""
    monkeypatch.setattr(probe, 'PROBE_ONLY_NATIVE_ACTION_LIMITS', MappingProxyType({UNADMITTED: 400}))
    folder = Path(os.environ['ROBODOJO_ASSET_ROOT']) / 'Eval_Layout/RoboDojo/arx_x5/0'
    (folder / f'{UNADMITTED}_0.json').write_text(json.dumps({'task': UNADMITTED}))
    assert probe.parse_cell(f'{UNADMITTED}:robodojo-gpu4-d3') == (UNADMITTED, 'robodojo-gpu4-d3')
    assert probe._probe_only_limit(UNADMITTED) == 400
    runtime = probe.probe_runtime(UNADMITTED, 'robodojo-gpu4-d3')
    assert (runtime['task_variant'], runtime['native_action_limit']) == (UNADMITTED, 400)
    assert UNADMITTED not in NATIVE_ACTION_LIMITS


def test_the_worker_command_names_the_qualification_module():
    command = probe.worker_command(Path('/py'), config=Path('c.json'), pass_root=Path('p'), ticks=100,
                                   sample_every=5, probe_only_limit=None, in_place_restart=False)
    assert command == ['/py', '-m', probe.WORKER_MODULE, '--config', 'c.json', '--output', 'p',
                       '--ticks', '100', '--sample-every', '5']
    command = probe.worker_command(Path('/py'), config=Path('c.json'), pass_root=Path('p'), ticks=50,
                                   sample_every=5, probe_only_limit=800, in_place_restart=True)
    assert command[-3:] == ['--probe-only-native-action-limit', '800', '--in-place-restart']


def test_report_only_judges_every_cell_directory(tmp_path):
    good = tmp_path / probe.cell_directory('pick_from_conveyor_by_image', 'robodojo-gpu4-d1')
    bad = tmp_path / probe.cell_directory('make_toast', 'robodojo-gpu5-d1')
    for root, value in ((good, _worker(_belt_objects(0.004), belts=[BELT])),
                        (bad, _worker([_sample(0), _sample(50)]))):
        root.mkdir()
        (root / worker.WORKER_REPORT_NAME).write_text(json.dumps(value))
        (root / 'launch.json').write_text(json.dumps({'task': root.name.split('--')[0], 'physics_device': 'cpu',
                                                      'resource_profile': root.name.split('--')[1]}))
    assert probe.main(['--output', str(tmp_path), '--report-only']) == 1
    report = json.loads((tmp_path / probe.REPORT_NAME).read_text())
    assert report['schema'] == probe.REPORT_SCHEMA and report['passed'] is False
    assert {cell['task']: cell['passed'] for cell in report['cells']} == {
        'pick_from_conveyor_by_image': True, 'make_toast': False}
    assert report['pass_criteria']
