"""The RoboDojo CPU-physics lane: per-task device, admission, identities and attestation.

User decision 2026-10-04: nine RoboDojo scenes resolve IsaacLab ``cpu``
physics by default (five articulated, two conveyor, two fluid); every other
task keeps ``cuda:0``. The selection must never move the CUDA evidence: the
per-task table lives in a host-only module outside the worker kernel, so the
card and scene-gate dependency projections of every CUDA task stay
byte-identical. Nothing here starts Isaac, a worker, a Provider or a lease.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
from pathlib import Path

import pytest

from ora.environment.benchmarks import robodojo
from ora.environment.benchmarks import robodojo_physics as physics
from ora.orchestrator import robodojo_physics_attestation as attestation
from ora.orchestrator import robodojo_revalidation as rv

ROOT = Path(__file__).resolve().parents[1]
PHYSICS_MODULE = 'ora/environment/benchmarks/robodojo_physics.py'
#: Written out independently of the table so the table cannot vouch for itself.
ARTICULATED = ('fill_egg_holder', 'make_toast', 'press_by_number', 'store_laptop_and_headphones', 'swap_blocks')
CONVEYOR = ('match_and_pick_from_conveyor', 'pick_from_conveyor_by_image')
FLUID = ('pour_by_language', 'pour_liquid_into_cup')
CPU_TASKS = (*ARTICULATED, *CONVEYOR, *FLUID)


def _runtime(task: str, device: str) -> dict:
    """An otherwise admissible runtime for `task` (the admission fixture's shape)."""
    return {
        'kind': 'robodojo', 'transport': 'local_rpc', 'root_env': 'ROBODOJO_ROOT',
        'python_env': 'ROBODOJO_PYTHON', 'asset_root_env': 'ROBODOJO_ASSET_ROOT',
        **{field: robodojo.REVISIONS[name] for name, field in robodojo.REVISION_FIELDS.items()},
        'task_variant': task, 'eval_seed': 0, 'layout_id': 0,
        'layout_file': f'Eval_Layout/RoboDojo/arx_x5/0/{task}_0.json',
        'layout_sha256': 'a' * 64, 'num_envs': 1, 'env_config': 'arx_x5',
        'sensor_profile': 'rgbd_metric_v1', 'physics_dt': 0.004, 'control_hz': 25,
        'native_action_limit': robodojo.NATIVE_ACTION_LIMITS[task], 'native_actions_per_command': 20,
        'main_camera': 'head', 'policy_cameras': list(robodojo.PUBLIC_CAMERAS),
        'required_host': 'test-host', 'require_gpu': True, 'physics_device': device,
        'renderer_gpu': 4, 'cuda_visible_devices': '4', 'startup_timeout_s': 600,
        'request_timeout_s': 180, 'close_timeout_s': 30,
        'workspace_bounds_m': {'x': [-1, 1], 'y': [-1, 1], 'z': [0, 2]},
        'infrastructure_qualification': {'path': 'outputs/x/qualification/fixture.json', 'sha256': 'b' * 64},
    }


# --- the per-task table and admission ---------------------------------------------------------

def test_nine_scenes_resolve_cpu_physics_and_every_other_task_cuda():
    assert set(physics.CPU_PHYSICS_TASKS) == set(CPU_TASKS)
    assert physics.PHYSICS_DEVICES == ('cuda:0', 'cpu')
    for task in CPU_TASKS:
        assert physics.physics_device_for(task) == 'cpu'
        assert physics.CPU_PHYSICS_TASKS[task].strip()
    for task in set(robodojo.NATIVE_ACTION_LIMITS) - set(CPU_TASKS):
        assert physics.physics_device_for(task) == 'cuda:0'
    assert {physics.scene_kind_for(task) for task in ARTICULATED} == {'articulation'}
    assert {physics.scene_kind_for(task) for task in CONVEYOR} == {'conveyor'}
    assert {physics.scene_kind_for(task) for task in FLUID} == {'fluid'}
    assert physics.scene_kind_for('stack_blocks') == 'rigid'
    # The two fluid scenes are admitted since 2026-10-04 (reviewed contracts, passing fluid probe).
    assert set(CPU_TASKS) <= set(robodojo.NATIVE_ACTION_LIMITS)


@pytest.mark.parametrize('task', CPU_TASKS)
def test_a_cpu_scene_admits_only_cpu_physics(task):
    robodojo.validate_runtime_extension(_runtime(task, 'cpu'))
    with pytest.raises(robodojo.RoboDojoBenchmarkError, match="physics_device must be 'cpu'"):
        robodojo.validate_runtime_extension(_runtime(task, 'cuda:0'))


@pytest.mark.parametrize('task', ['stack_blocks', 'play_tic_tac_toe', 'fold_clothes'])
def test_a_cuda_scene_admits_only_cuda_physics(task):
    robodojo.validate_runtime_extension(_runtime(task, 'cuda:0'))
    with pytest.raises(robodojo.RoboDojoBenchmarkError, match="physics_device must be 'cuda:0'"):
        robodojo.validate_runtime_extension(_runtime(task, 'cpu'))


def test_cpu_physics_still_requires_the_gpu_renderer():
    runtime = _runtime('press_by_number', 'cpu')
    runtime['require_gpu'] = False
    with pytest.raises(robodojo.RoboDojoBenchmarkError, match='require_gpu'):
        robodojo.validate_runtime_extension(runtime)


# --- CUDA evidence stays current ----------------------------------------------------------------

def test_the_table_is_a_reviewed_exclusion_outside_every_kernel_and_dependency_map():
    from ora.orchestrator.robodojo_kernel import ROBODOJO_MOTION_KERNEL_PATHS, WORKER_KERNEL_PATHS
    from ora.orchestrator.robodojo_kernel_review import (
        KERNEL_REVIEWED_EXCLUSIONS, kernel_manifest_gaps, kernel_review_drift, worker_import_closure,
    )

    assert PHYSICS_MODULE not in WORKER_KERNEL_PATHS and PHYSICS_MODULE not in ROBODOJO_MOTION_KERNEL_PATHS
    assert 'validate_runtime_extension' in KERNEL_REVIEWED_EXCLUSIONS[PHYSICS_MODULE]
    # Reached only through the lazy import inside the projected admission function.
    closure = worker_import_closure()
    assert PHYSICS_MODULE in closure['ora/environment/benchmarks/robodojo.py']
    assert not kernel_manifest_gaps() and not kernel_review_drift()
    assert PHYSICS_MODULE not in rv.DEPENDENCIES
    # The table imports nothing from the project, so it pulls nothing into the closure.
    assert closure[PHYSICS_MODULE] == ()


def _benchmark_tree() -> ast.Module:
    return ast.parse((ROOT / rv.BENCHMARK).read_text())


def test_the_benchmark_edit_lives_only_inside_projected_regions():
    """Only validate_runtime_extension imports the table; no new module-level binding exists."""
    tree = _benchmark_tree()
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
               and node.module == 'robodojo_physics']
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == 'validate_runtime_extension')
    assert imports and all(node in list(ast.walk(function)) for node in imports)
    names = {target.id for node in tree.body if isinstance(node, (ast.Assign, ast.AnnAssign))
             for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
             if isinstance(target, ast.Name)}
    assert not {name for name in names if 'PHYSICS' in name and name != 'PHYSICS_DT_S'
                and name != 'PHYSICS_STEPS_PER_NATIVE_ACTION'}


def _worktree_source(commit, path):
    """`git show` of the working tree: a missing file fails as git does."""
    if not (ROOT / path).is_file():
        raise subprocess.CalledProcessError(128, ['git', 'show', f'{commit}:{path}'])
    return (ROOT / path).read_bytes()


def test_the_card_never_follows_the_table_into_its_control_closure(monkeypatch):
    """The card's terminal/control closure is read from the working tree: the table is not a member."""
    monkeypatch.setattr(rv, 'source_bytes', _worktree_source)
    closure = rv.control_closure('worktree', 'press_by_number')
    # The benchmark module's lazy import is an edge, but the card never follows it:
    # only followed members are hashed, and the benchmark itself is hashed projected.
    assert PHYSICS_MODULE in closure[rv.BENCHMARK] and PHYSICS_MODULE not in closure
    assert PHYSICS_MODULE not in rv.terminal_dependencies('worktree', 'press_by_number')


# --- default activation of the resolver ---------------------------------------------------------

@pytest.fixture
def fixture_assets(monkeypatch, tmp_path):
    assets = tmp_path / 'Assets'
    folder = assets / 'Eval_Layout/RoboDojo/arx_x5/0'
    folder.mkdir(parents=True)
    for task in robodojo.NATIVE_ACTION_LIMITS:
        (folder / f'{task}_0.json').write_text(json.dumps({'task': task}))
    monkeypatch.setenv('ROBODOJO_ASSET_ROOT', str(assets))
    monkeypatch.setenv('ROBODOJO_SWAP_T_ASSET_ROOT', str(assets))
    monkeypatch.setenv('ORA_EXECUTION_HOST', 'test-host')
    return assets


def test_the_runtime_resolver_emits_each_tasks_own_physics_device(fixture_assets):
    from ora.orchestrator.robodojo_runtime_config import runtime_configuration

    for task in robodojo.NATIVE_ACTION_LIMITS:
        runtime = runtime_configuration('4', task)
        assert runtime['physics_device'] == ('cpu' if task in CPU_TASKS else 'cuda:0'), task
        assert (runtime['cuda_visible_devices'], runtime['renderer_gpu']) == ('4', 4)


# --- RunSpec, preflight, evaluation and seal -------------------------------------------------------

def test_cpu_multi_round_sessions_need_a_layout_schedule():
    runtime = {'kind': 'robodojo', 'physics_device': 'cpu'}
    message = attestation.cpu_round_restart_refusal(runtime, rounds=2)
    assert message and 'rule=robodojo_cpu_in_place_restart' in message and 'layout_schedule' in message
    assert attestation.cpu_round_restart_refusal(runtime, rounds=1) is None
    assert attestation.cpu_round_restart_refusal({**runtime, 'layout_schedule': {'rounds': [1, 2]}}, rounds=2) is None
    assert attestation.cpu_round_restart_refusal({**runtime, 'physics_device': 'cuda:0'}, rounds=3) is None
    assert attestation.cpu_round_restart_refusal({'kind': 'robotwin'}, rounds=3) is None


def test_a_cpu_lane_is_motion_qualified_only_by_its_own_scripted_record(monkeypatch):
    from ora.environment import definitions

    runtime = {'kind': 'robodojo', 'physics_device': 'cpu'}
    # The CPU row landed with its own 24-case record (test-cluster, 2026-10-05).
    assert definitions.SPARSE_MOTION_QUALIFIED_PHYSICS_DEVICES[('motion_action_v3', 'robodojo')] == ('cuda:0', 'cpu')
    assert definitions.sparse_motion_physics_refusal('motion_action_v3', runtime) is None
    # Without it, the CUDA record alone never qualified a CPU lane.
    monkeypatch.setitem(definitions.SPARSE_MOTION_QUALIFIED_PHYSICS_DEVICES, ('motion_action_v3', 'robodojo'),
                        ('cuda:0',))
    message = definitions.sparse_motion_physics_refusal('motion_action_v3', runtime)
    assert message and 'physics device cpu' in message and 'cuda:0' in message
    assert definitions.sparse_motion_physics_refusal('motion_action_v3', {**runtime, 'physics_device': 'cuda:0'}) is None
    assert definitions.sparse_motion_physics_refusal('runtime_action_v3', runtime) is None
    assert definitions.sparse_motion_physics_refusal('motion_action_v3', {'kind': 'robotwin'}) is None
    # The codec/environment table itself is unchanged.
    assert definitions.SPARSE_MOTION_QUALIFIED_ENVIRONMENTS['motion_action_v3'] == ('robotwin', 'libero_pro', 'robodojo')


@pytest.mark.skipif(not (Path(os.environ.get('ROBODOJO_ASSET_ROOT', str(Path.home() / "robodojo-runtime/assets/Assets")))
                          / 'Eval_Layout/RoboDojo/arx_x5/0').is_dir(), reason='pinned RoboDojo Assets absent')
def test_a_cpu_cell_plan_is_refused_until_its_motion_record_lands_then_by_round_count(monkeypatch):
    """On the maintained RoboDojo builder (the frozen EXP11A launcher left this tree; it composes at c20299be)."""
    from ora.environment import definitions
    from ora.orchestrator import robodojo_spec
    from ora.orchestrator.spec import RunSpecError, resolve_run_spec

    gate = {'path': 'outputs/robodojo-cpu-fixture/qualification/revalidation/x.json', 'sha256': '0' * 64}
    spec = robodojo_spec.build_spec('press_by_number', experiment='robodojo-cpu-fixture', gpu='4',
                                    resource_profile='robodojo-gpu4-d1', infrastructure_qualification=gate, rounds=1)
    assert spec['runtime']['physics_device'] == 'cpu'
    assert spec['runtime']['infrastructure_qualification']['path'].endswith('x.json')
    monkeypatch.setitem(definitions.SPARSE_MOTION_QUALIFIED_PHYSICS_DEVICES, ('motion_action_v3', 'robodojo'),
                        ('cuda:0',))
    with pytest.raises(RunSpecError, match='rule=sparse_motion_physics_unqualified'):
        resolve_run_spec(spec, root=ROOT)
    monkeypatch.undo()
    plan = resolve_run_spec(spec, root=ROOT)
    assert plan['runtime']['physics_device'] == 'cpu' and plan['session']['rounds'] == 1
    spec['session']['rounds'] = 2
    with pytest.raises(RunSpecError, match='rule=robodojo_cpu_in_place_restart'):
        resolve_run_spec(spec, root=ROOT)


def _source(device=None, fabric=None):
    sim = {} if device is None else {'device': device}
    if fabric is not None:
        sim['use_fabric'] = fabric
    return {'resolved_environment_config': {'sim': sim}}


def test_the_resolved_simulation_device_must_be_the_declared_physics_device():
    cuda, cpu = {'physics_device': 'cuda:0'}, {'physics_device': 'cpu'}
    assert attestation.physics_attestation_errors(cuda, _source('cuda:0')) == []
    # A CUDA manifest from before the attestation carries no device; CUDA was the only device then.
    assert attestation.physics_attestation_errors(cuda, {}) == []
    assert attestation.physics_attestation_errors(cpu, _source('cpu')) == []
    assert attestation.physics_attestation_errors(cpu, {}) == ['robodojo_physics_device_invalid']
    assert attestation.physics_attestation_errors(cpu, _source('cuda:0')) == ['robodojo_physics_device_invalid']
    assert attestation.physics_attestation_errors(cuda, _source('cpu')) == ['robodojo_physics_device_invalid']
    assert attestation.physics_attestation_errors(cpu, _source('cpu', True)) == ['robodojo_physics_fabric_enabled']
    mode = attestation.physics_mode(cpu, _source('cpu'))
    assert mode == {'physics_device': 'cpu', 'isaaclab_sim_device': 'cpu', 'tensor_api': 'cpu', 'fabric': False,
                    'physx_gpu_dynamics': 'not_attested_by_the_run; see physics_scene_probe'}
    assert attestation.physics_mode(cuda, _source('cuda:0'))['tensor_api'] == 'cuda'
    assert attestation.require_physics_attestation(cpu, _source('cpu')) == mode
    with pytest.raises(robodojo.RoboDojoBenchmarkError, match='physics device'):
        attestation.require_physics_attestation(cpu, _source('cuda:0'))


def test_the_evaluation_and_seal_record_the_physics_device(tmp_path):
    native = tmp_path / 'native'
    native.mkdir()
    (native / 'source-manifest.json').write_text(json.dumps(_source('cpu')))
    record = attestation.physics_record({'runtime': {'kind': 'robodojo', 'physics_device': 'cpu'}}, native)
    assert record['physics_device'] == 'cpu' and record['isaaclab_sim_device'] == 'cpu'
    assert record['errors'] == []
    missing = attestation.physics_record({'runtime': {'kind': 'robodojo', 'physics_device': 'cpu'}}, tmp_path / 'x')
    assert missing['isaaclab_sim_device'] is None and missing['source_manifest'] == 'absent'
    assert attestation.physics_record({'runtime': {'kind': 'robotwin'}}, native) is None
    evaluation = (ROOT / 'ora/orchestrator/evaluation.py').read_text()
    assert '"robodojo_physics"' in evaluation and '"physics_device"' in evaluation


def test_revalidation_checks_the_smokes_physics_and_keeps_cuda_checks_and_coverage():
    assert rv.CHECKS == ('device_identity', 'loaded_geometry', 'rgbd', 'left_composite',
                         'right_composite', 'read_only_alignment', 'round_reset')
    source = (ROOT / 'ora/orchestrator/robodojo_revalidation.py').read_text()
    assert 'require_physics_attestation' in source
    cuda = rv.scene_coverage({'physics_device': 'cuda:0'})
    assert cuda == {'fresh': list(rv.CHECKS) + ['bounded_close'],
                    'reused': ['official_terminal', 'unchanged native motion/control implementation'],
                    'not_established': ['task success', 'new full motion sweep', 'fresh per-task native timeout',
                                        'fresh nonzero pose/gripper displacement']}
    cpu = rv.scene_coverage({'physics_device': 'cpu'})
    # A single-round CPU smoke does not restart in place (2026-10-05).
    assert cpu['fresh'] == [check for check in cuda['fresh'] if check != 'round_reset']
    assert cpu['reused'] == cuda['reused']
    assert cpu['not_established'][:4] == cuda['not_established']
    assert 'in-place' in cpu['not_established'][4]
    scheduled = rv.scene_coverage({'physics_device': 'cpu', 'layout_schedule': {'rounds': []}})
    assert 'round_reset' in scheduled['fresh']


@pytest.mark.parametrize('rounds, schedule, passed', [(2, None, False), (1, None, True), (2, {'rounds': [1, 2]}, True)])
def test_preflight_rechecks_the_cpu_round_rule_on_the_exact_plan(monkeypatch, rounds, schedule, passed):
    from ora.orchestrator import preflight

    for name in ('ROBODOJO_ROOT', 'ROBODOJO_ASSET_ROOT', 'ROBODOJO_PYTHON'):
        monkeypatch.delenv(name, raising=False)
    runtime = {**_runtime('press_by_number', 'cpu'), 'layout_schedule': schedule}
    checks = preflight._robodojo_runtime({'runtime': runtime, 'session': {'rounds': rounds}}, runner=None)
    selected = [item for item in checks if item['check'] == 'robodojo_cpu_round_restart']
    assert len(selected) == 1 and selected[0]['passed'] is passed
    cuda = preflight._robodojo_runtime({'runtime': _runtime('stack_blocks', 'cuda:0'), 'session': {'rounds': 2}},
                                       runner=None)
    assert not [item for item in cuda if item['check'] == 'robodojo_cpu_round_restart']
