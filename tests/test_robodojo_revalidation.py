"""Offline evidence reuse preserves origins and refuses dependency/raw drift."""
import copy
import hashlib
import json
from pathlib import Path
import pytest
from ora.orchestrator import robodojo_revalidation as rv
from ora.environment.benchmarks.robodojo import RoboDojoBenchmarkError


def test_physical_dependencies_ignore_only_host_schema_checker(monkeypatch):
    monkeypatch.setattr(rv,'DEPENDENCIES',(rv.BENCHMARK,'worker.py'))
    values={'old':b'LIMIT=20\ndef validate_runtime_extension(x):\n return 1\n',
            'new':b'LIMIT=20\ndef validate_runtime_extension(x):\n return 2\n',
            'changed':b'LIMIT=21\ndef validate_runtime_extension(x):\n return 2\n'}
    monkeypatch.setattr(rv,'source_bytes',lambda commit,path: values[commit] if path==rv.BENCHMARK else b'worker')
    assert rv.dependency_map('old')==rv.dependency_map('new')
    assert rv.dependency_map('old')!=rv.dependency_map('changed')


def test_selected_task_limit_reuse_retains_the_measured_limit_and_controller(monkeypatch):
    monkeypatch.setattr(rv, 'DEPENDENCIES', (rv.BENCHMARK,))
    sources = {
        'old': b'NATIVE_ACTION_LIMITS={"fold_clothes":500}\nCONTROL_HZ=25\n',
        'added': b'NATIVE_ACTION_LIMITS={"fold_clothes":500,"build_tower":1050}\nCONTROL_HZ=25\n',
        'changed': b'NATIVE_ACTION_LIMITS={"fold_clothes":501}\nCONTROL_HZ=25\n',
        'controller': b'NATIVE_ACTION_LIMITS={"fold_clothes":500}\nCONTROL_HZ=26\n',
        'missing': b'NATIVE_ACTION_LIMITS={"build_tower":1050}\nCONTROL_HZ=25\n',
    }
    monkeypatch.setattr(rv, 'source_bytes', lambda commit, path: sources[commit])
    old = rv.dependency_map('old', 'fold_clothes')
    assert old == rv.dependency_map('added', 'fold_clothes')
    assert old != rv.dependency_map('changed', 'fold_clothes')
    assert old != rv.dependency_map('controller', 'fold_clothes')
    assert rv.dependency_map('old') != rv.dependency_map('added')
    with pytest.raises(RoboDojoBenchmarkError, match='task limit missing'):
        rv.dependency_map('missing', 'fold_clothes')


def test_auxiliary_guard_projection_only_reuses_reviewed_inactive_execution(monkeypatch):
    monkeypatch.setattr(rv, 'DEPENDENCIES', (rv.SESSION,))
    old = b'class RobodojoSession:\n def targets(self):\n  result = self.target_buffers()\n  return result\n'
    added = old.replace(b'  return result', b'  if getattr(self.environment, "interact", False):\n   from .interaction_control import support_position_targets\n   result.update(support_position_targets(self))\n  return result')
    sources = {'old':old, 'added':added,
               'target_changed':added.replace(b'self.target_buffers()', b'self.other_buffers()'),
               'guard_changed':added.replace(b'"interact", False', b'"interact", True')}
    monkeypatch.setattr(rv, 'source_bytes', lambda commit, path: sources[commit])
    assert rv.dependency_map('old', 'fold_clothes') == rv.dependency_map('added', 'fold_clothes')
    for task in (None, 'imitate_sorting_sequence', 'make_kong', 'future_task'):
        assert rv.dependency_map('old', task) != rv.dependency_map('added', task)
    for changed in ('target_changed', 'guard_changed'):
        assert rv.dependency_map('old', 'fold_clothes') != rv.dependency_map(changed, 'fold_clothes')


def test_terminal_dependency_detects_lifecycle_change(monkeypatch):
    monkeypatch.setattr(rv,'TERMINAL_FILES',())
    monkeypatch.setattr(rv,'TERMINAL_METHODS',('is_terminal',))
    monkeypatch.setattr(rv,'source_bytes',lambda commit,path:
        ('class RobodojoSession:\n def is_terminal(self):\n  return '+('False' if commit=='changed' else 'True')+'\n').encode())
    assert rv.terminal_dependencies('old')==rv.terminal_dependencies('docs_only')
    assert rv.terminal_dependencies('old')!=rv.terminal_dependencies('changed')


def test_artifact_hash_and_containment(tmp_path):
    p=tmp_path/'outputs/cell/report.json';p.parent.mkdir(parents=True);p.write_text('{"passed": true}')
    ref=rv.reference(p,tmp_path)
    assert rv.read_reference(ref,tmp_path)[0]['passed'] is True
    p.write_text('{"passed": false}')
    with pytest.raises(RoboDojoBenchmarkError,match='hash mismatch'):rv.read_reference(ref,tmp_path)
    with pytest.raises(RoboDojoBenchmarkError,match='shared outputs'):
        rv.read_reference({'path':'outputs/../report.json','sha256':'0'*64},tmp_path)
    link=p.parent/'link.json';link.symlink_to(p)
    with pytest.raises(RoboDojoBenchmarkError,match='symlink'):
        rv.read_reference({'path':'outputs/cell/link.json','sha256':hashlib.sha256(p.read_bytes()).hexdigest()},tmp_path)


def test_certificate_rechecks_raw_evidence_across_unrelated_commits(tmp_path,monkeypatch):
    record={'schema':rv.SCHEMA,'original_card':{'path':'original/card','sha256':'old'},
        'scene_smoke':{'path':'scene/report','sha256':'new'},'producer_dependencies':{'worker':'same'},
        'terminal_dependencies':{'terminal':'same'},'coverage':{'fresh':['scene'],'reused':['terminal']},
        'environment':{},'device_binding':{'selected_physical_index':7},
        'raw_artifacts':[{'path':'source','sha256':'original'}], 'harness_commit':'original-checker'}
    path=tmp_path/'outputs/certificate.json';path.parent.mkdir();path.write_text(json.dumps(record))
    observed=[]
    def evaluate(runtime,**kwargs):
        observed.append(kwargs)
        return {**copy.deepcopy(record),'harness_commit':kwargs['harness_commit']}
    monkeypatch.setattr(rv,'evaluate',evaluate)
    runtime={'infrastructure_qualification':rv.reference(path,tmp_path)}
    view=rv.load(runtime,workspace_root=tmp_path,harness_commit='later-docs')
    assert view['harness_commit']=='later-docs'
    assert observed[0]['card_reference']==record['original_card']
    altered=copy.deepcopy(record);altered['raw_artifacts'][0]['sha256']='changed'
    monkeypatch.setattr(rv,'evaluate',lambda *a,**k:altered)
    with pytest.raises(RoboDojoBenchmarkError,match='raw_artifacts drift'):
        rv.load(runtime,workspace_root=tmp_path,harness_commit='later-docs')


def test_admitted_revalidation_view_reaches_worker_boundary(tmp_path):
    from ora.orchestrator import coordinator
    view={'schema':rv.SCHEMA,'passed':True,'coverage':{'reused':['official_terminal']}}
    (tmp_path/'preflight.json').write_text(json.dumps({'checks':[{
        'check':'robodojo_infrastructure_qualification','passed':True,'detail':view}]}))
    assert coordinator._qualified_runtime_binding(tmp_path,{'kind':'robodojo'})==view
    (tmp_path/'preflight.json').write_text(json.dumps({'checks':[{
        'check':'robodojo_infrastructure_qualification','passed':False,'detail':view}]}))
    with pytest.raises(coordinator.SupervisorError,match='infrastructure binding'):
        coordinator._qualified_runtime_binding(tmp_path,{'kind':'robodojo'})


@pytest.mark.parametrize('damage', [None, 'collision', 'controlled_frame', 'observed_hash'])
def test_wording_refresh_reuses_only_identical_measured_geometry(tmp_path, monkeypatch, damage):
    root = Path(__file__).resolve().parents[1]
    directory = 'ora/policy/embodiments/arx-x5/'
    # The original card is the kept descriptor with only its wording changed;
    # the source commit that held it is stood in for by ``source_bytes``.
    old_path, new_path = directory+'descriptor-public-geometry-wording-predecessor.json', directory+'descriptor-public-geometry-v5.json'
    current = json.loads((root/new_path).read_text())
    old = copy.deepcopy(current)
    old['display_name'] += ' (earlier wording)'
    old_bytes = json.dumps(old, indent=2).encode()
    geometry = {g['end_effector_id']:g['contact_geometry'] for g in old['geometry']['grippers']}
    verification = {'passed':True, 'descriptor_sha256':rv.digest(old_bytes),
        'observed_geometry_sha256':rv.digest(json.dumps(geometry,sort_keys=True,separators=(',',':')).encode())}
    if damage == 'collision':
        current['geometry']['grippers'][0]['contact_geometry']['fingers'][0]['collision_bounds_m']['maximum'][0] += .01
    elif damage == 'controlled_frame':
        current['control']['controlled_points'][0]['offset_m'][0] = .01
    elif damage == 'observed_hash':
        verification['observed_geometry_sha256'] = '0'*64
    path = tmp_path/new_path
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(current))
    monkeypatch.setattr(rv, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setattr(rv, 'source_bytes', lambda commit, relative: old_bytes)
    kwargs = dict(current={'path':new_path,'sha256':rv.digest(path.read_bytes())},
        original={'path':old_path,'sha256':rv.digest(old_bytes)}, source_commit='original', verification=verification)
    if damage:
        with pytest.raises(RoboDojoBenchmarkError):
            rv.revalidate_geometry_binding(**kwargs)
    else:
        result = rv.revalidate_geometry_binding(**kwargs)
        assert result['original_descriptor'] == kwargs['original']
        assert result['current_descriptor'] == kwargs['current']
        assert result['observed_geometry_sha256'] == verification['observed_geometry_sha256']


def test_card_reuse_excludes_only_exact_inactive_cloth_setup_and_keeps_scene_dependency(monkeypatch):
    old = b'''class EvaluationLifecycle:
 def _post_setup_scene(self, sim):
  super()._post_setup_scene(sim)
  self.obs_manager.initialize(self)
 def official_tick(self):
  return self.reward_manager.step()
'''
    new = b'''def _prepare_cuda_garments(manager, num_envs):
 manager.spawn_scene_objects(0)

class EvaluationLifecycle:
 def _post_setup_scene(self, sim):
  super()._post_setup_scene(sim)
  if self.task_name == "fold_clothes":
   _prepare_cuda_garments(self.scene_manager, self.num_envs)
  self.obs_manager.initialize(self)
 def official_tick(self):
  return self.reward_manager.step()
'''
    project = rv._bridge_terminal_source
    assert project(old, 'hang_mugs') == project(new, 'hang_mugs')
    assert project(old, 'fold_clothes') != project(new, 'fold_clothes')
    assert project(old, None) != project(new, None)
    assert project(old, 'hang_mugs') != project(new.replace(b'== "fold_clothes"', b'!= "fold_clothes"'), 'hang_mugs')
    assert project(old, 'hang_mugs') != project(new.replace(b'reward_manager.step()', b'reward_manager.reset()'), 'hang_mugs')
    path = 'ora/environment/backends/robodojo/bridge.py'
    monkeypatch.setattr(rv, 'DEPENDENCIES', (path,))
    monkeypatch.setattr(rv, 'source_bytes', lambda commit, path: old if commit == 'old' else new)
    assert rv.dependency_map('old') != rv.dependency_map('new')
    assert rv.dependency_map('old', 'fold_clothes') != rv.dependency_map('new', 'fold_clothes')
    assert rv.dependency_map('old', 'insert_tubes') == rv.dependency_map('new', 'insert_tubes')


def test_terminal_dependency_resolves_refactored_owner_and_binds_helpers(monkeypatch):
    monkeypatch.setattr(rv, 'TERMINAL_FILES', ())
    monkeypatch.setattr(rv, 'TERMINAL_METHODS', ('restart_round',))
    owner_path = rv.TERMINAL_OWNERS['RobodojoRoundMixin'][0]
    session = b"from .round_lifecycle import RobodojoRoundMixin\nclass RobodojoSession(RobodojoRoundMixin):\n pass\n"
    mixin = b"LIMIT = 1\ndef helper():\n return LIMIT\nclass RobodojoRoundMixin:\n def restart_round(self):\n  return self._check()\n def _check(self):\n  return helper()\n"
    sources = {rv.SESSION: session, owner_path: mixin, rv.BENCHMARK: b'LIMIT=20'}
    monkeypatch.setattr(rv, 'source_bytes', lambda commit, path: sources[path])
    original = rv.terminal_dependencies('source')
    assert f'{rv.SESSION}:_check' in original
    for changed in (mixin.replace(b'return helper()', b'return False'),
                    mixin.replace(b'LIMIT = 1', b'LIMIT = 2'),
                    mixin.replace(b'return LIMIT', b'return 3')):
        sources[owner_path] = changed
        assert rv.terminal_dependencies('source') != original
    sources[owner_path] = mixin.replace(b' def _check(self):', b' def _missing(self):')
    with pytest.raises(RoboDojoBenchmarkError, match='helper _check has no declared owner'):
        rv.terminal_dependencies('source')


@pytest.mark.parametrize('session,owner', [
    (b'class RobodojoSession(Unknown):\n pass\n', b''),
    (b'class RobodojoSession(RobodojoRoundMixin):\n pass\n', b''),
    (b'from .round_lifecycle import RobodojoRoundMixin\nclass RobodojoSession(RobodojoRoundMixin):\n pass\n',
     b'class RobodojoRoundMixin:\n pass\n'),
    (b'from .round_lifecycle import RobodojoRoundMixin\nclass RobodojoSession(RobodojoRoundMixin):\n def restart_round(self):\n  return True\n',
     b'class RobodojoRoundMixin:\n def restart_round(self):\n  return False\n'),
])
def test_terminal_dependency_rejects_unknown_missing_and_ambiguous_owners(monkeypatch, session, owner):
    monkeypatch.setattr(rv, 'TERMINAL_FILES', ())
    monkeypatch.setattr(rv, 'TERMINAL_METHODS', ('restart_round',))
    sources = {rv.SESSION: session, rv.TERMINAL_OWNERS['RobodojoRoundMixin'][0]: owner}
    monkeypatch.setattr(rv, 'source_bytes', lambda commit, path: sources[path])
    with pytest.raises(RoboDojoBenchmarkError, match='owner'):
        rv.terminal_dependencies('source')


def test_terminal_dependency_historical_session_does_not_read_future_mixins(monkeypatch):
    monkeypatch.setattr(rv, 'TERMINAL_FILES', ())
    monkeypatch.setattr(rv, 'TERMINAL_METHODS', ('restart_round',))
    sources = {rv.SESSION: b'class RobodojoSession:\n def restart_round(self):\n  return True\n',
               rv.BENCHMARK: b'LIMIT=20'}
    monkeypatch.setattr(rv, 'source_bytes', lambda commit, path: sources[path])
    assert f'{rv.SESSION}:restart_round' in rv.terminal_dependencies('historical')


def test_terminal_dependency_includes_imported_ik_and_reachability(monkeypatch):
    monkeypatch.setattr(rv, 'TERMINAL_FILES', (rv.SPARSE,))
    monkeypatch.setattr(rv, 'TERMINAL_METHODS', ('is_terminal',))
    def source(commit, path):
        if path == rv.SESSION:
            return b'class RobodojoSession:\n def is_terminal(self):\n  return False\n'
        if path == rv.SPARSE:
            return b'from .ik import solve_ik\n'
        # Helpers are parsed and bound by AST, so the fixture differs in code.
        changed = ' changed' if commit == 'ik_changed' and path.endswith('/ik.py') else ''
        return f'SOURCE = {path + changed!r}\n'.encode()
    monkeypatch.setattr(rv, 'source_bytes', source)
    old = rv.terminal_dependencies('old')
    assert 'ora/environment/backends/robodojo/ik.py' in old
    assert 'ora/environment/backends/robodojo/reachability.py' in old
    assert old != rv.terminal_dependencies('ik_changed')


def _git_sources(sources):
    """A ``source_bytes`` stand-in that fails like ``git show`` on a missing path."""
    import subprocess

    def read(commit, path):
        try:
            return sources[commit][path]
        except KeyError:
            raise subprocess.CalledProcessError(128, ['git', 'show', f'{commit}:{path}']) from None
    return read


def test_terminal_dependency_binds_the_executor_helper_closure(monkeypatch):
    backend = 'ora/environment/backends/robodojo/'
    bridge = backend + 'bridge.py'
    monkeypatch.setattr(rv, 'TERMINAL_FILES', (rv.SPARSE, bridge))
    monkeypatch.setattr(rv, 'TERMINAL_METHODS', ('is_terminal',))
    sparse = (b'from .bridge import RobodojoError\n'
              b'def dispatch(self, kind):\n'
              b'    if kind == "wait":\n'
              b'        from .wait import execute_wait\n'
              b'        return execute_wait(self)\n'
              b'    if getattr(self.environment, "interact", False):\n'
              b'        from .interaction_control import policy_targets\n'
              b'    from .recovery import execute_recovery\n'
              b'    return execute_recovery(self)\n')
    base = {rv.SESSION: b'class RobodojoSession:\n def is_terminal(self):\n  return False\n',
            rv.BENCHMARK: b'LIMIT=20', rv.SPARSE: sparse,
            bridge: b'class RobodojoError(Exception):\n pass\n',
            backend + 'wait.py': b'HOLD = 1\n',
            backend + 'interaction_control.py': b'def policy_targets(s):\n return {}\n',
            # recovery reaches its own helper, and a task-projected TERMINAL_FILES member.
            backend + 'recovery.py': b'from . import unwind\nfrom .bridge import RobodojoError\n',
            backend + 'unwind.py': b'STEP = 0.2\n'}
    sources = {'old': base,
               'comment': {**base, backend + 'recovery.py': base[backend + 'recovery.py'] + b'# note\n'},
               'recovery': {**base, backend + 'recovery.py': base[backend + 'recovery.py'] + b'LEVELS = 3\n'},
               'unwind': {**base, backend + 'unwind.py': b'STEP = 0.3\n'},
               'wait': {**base, backend + 'wait.py': b'HOLD = 2\n'},
               'historical': {**base, rv.SPARSE: b'from .bridge import RobodojoError\n'}}
    monkeypatch.setattr(rv, 'source_bytes', _git_sources(sources))
    old = rv.terminal_dependencies('old', 'build_tower')
    for helper in ('recovery.py', 'unwind.py', 'wait.py'):
        assert backend + helper in old, helper
    # The guarded interactive import stays projected out for a reviewed
    # noninteractive task, and bound for any other one.
    assert backend + 'interaction_control.py' not in old
    assert backend + 'interaction_control.py' in rv.terminal_dependencies('old', None)
    # A TERMINAL_FILES member keeps its task projection: it is not re-hashed whole.
    assert old[bridge] == rv.digest(rv._bridge_terminal_source(base[bridge], 'build_tower'))
    for changed in ('recovery', 'unwind', 'wait'):
        assert rv.terminal_dependencies(changed, 'build_tower') != old, changed
    # Helpers are bound by AST: a comment-only edit leaves the card current.
    assert rv.terminal_dependencies('comment', 'build_tower') == old
    # A producer whose executor had no such helpers differs by key, not by a crash.
    assert rv.terminal_dependencies('historical', 'build_tower') != old


BACKEND = 'ora/environment/backends/robodojo/'


def _worktree_sources(monkeypatch):
    import subprocess

    def worktree(commit, path):
        file = rv.PROJECT_ROOT / path
        if not file.is_file():
            raise subprocess.CalledProcessError(128, ['git', 'show', f'{commit}:{path}'])
        return file.read_bytes()
    monkeypatch.setattr(rv, 'source_bytes', worktree)


def _covered(deps, path):
    return path in deps or f'{path}:module_setup' in deps


def test_terminal_and_scene_dependencies_stay_inside_the_worker_kernel(monkeypatch):
    from ora.orchestrator.robodojo_kernel import WORKER_KERNEL_PATHS

    _worktree_sources(monkeypatch)
    kernel = set(WORKER_KERNEL_PATHS)
    assert set(rv.TERMINAL_FILES) <= kernel
    assert {path for path, _ in rv.TERMINAL_OWNERS.values()} <= kernel
    assert kernel <= set(rv.DEPENDENCIES)
    # What a card leaves out, the scene smoke still binds: a card is reused
    # only through a fresh smoke, whose DEPENDENCIES cover the whole kernel.
    assert set(rv.CARD_EXCLUSIONS) <= set(rv.DEPENDENCIES)
    for task in ('build_tower', None):
        files = {key for key in rv.terminal_dependencies('worktree', task) if ':' not in key}
        assert files <= kernel, sorted(files - kernel)


def test_a_card_binds_everything_its_terminal_and_control_closure_reaches(monkeypatch):
    """Every module the session or TERMINAL_FILES reach is bound or reviewed.

    The adversarial review edited each of these files and the card stayed
    current while only the scene smoke went stale; each must be a key.
    """
    from ora.orchestrator.import_closure import import_closure, tree_reader
    from ora.orchestrator.robodojo_kernel import WORKER_KERNEL_PATHS

    _worktree_sources(monkeypatch)
    kernel = set(WORKER_KERNEL_PATHS)
    reviewed_mutations = {
        'ora/environment/controllers/' + name for name in (
            'joint_path_clearance.py', 'gripper_stall.py', 'motion_transaction.py', 'reachability.py')
    } | {'ora/contracts/' + name for name in ('motion.py', 'pose.py', 'robodojo_arx.py', 'wait.py')
         } | {BACKEND + name for name in ('recovery.py', 'wait.py', 'bootstrap.py', 'feedback.py',
                                          'camera_capture.py', 'staged_contact.py')}
    for task in ('build_tower', None):
        deps = rv.terminal_dependencies('worktree', task)
        closure = rv.control_closure('worktree', task)
        assert reviewed_mutations <= set(deps), sorted(reviewed_mutations - set(deps))
        unbound = [path for path in closure
                   if not _covered(deps, path) and path != rv.SESSION]
        assert not unbound, unbound
        imported = {target for targets in closure.values() for target in targets}
        assert set(rv.CARD_EXCLUSIONS) <= imported, sorted(set(rv.CARD_EXCLUSIONS) - imported)
    # The executor itself -- sparse_motion and what it imports without passing
    # through another terminal file -- never reaches a reviewed exclusion.
    others = set(rv.TERMINAL_FILES) - {rv.SPARSE}
    executor = import_closure(
        (rv.SPARSE,), read=tree_reader(rv.PROJECT_ROOT),
        follow=lambda importer, imported: (
            importer not in others and (imported in kernel or imported.startswith(BACKEND))
            and not imported.endswith('/__init__.py')))
    deps = rv.terminal_dependencies('worktree', None)
    assert not set(executor) & set(rv.CARD_EXCLUSIONS)
    assert [path for path in executor if not _covered(deps, path)] == []


def test_terminal_dependencies_are_cached_per_full_commit(monkeypatch):
    monkeypatch.setattr(rv, 'TERMINAL_FILES', ())
    monkeypatch.setattr(rv, 'TERMINAL_METHODS', ('is_terminal',))
    reads = []

    def source(commit, path):
        reads.append(path)
        return b'class RobodojoSession:\n def is_terminal(self):\n  return False\n'
    monkeypatch.setattr(rv, 'source_bytes', source)
    rv._cached_terminal_dependencies.cache_clear()
    try:
        first = rv.terminal_dependencies('e' * 40, 'build_tower')
        count = len(reads)
        second = rv.terminal_dependencies('e' * 40, 'build_tower')
        assert len(reads) == count and second == first and second is not first
        second['tampered'] = 'x'
        assert 'tampered' not in rv.terminal_dependencies('e' * 40, 'build_tower')
        rv.terminal_dependencies('short-ref', 'build_tower')
        assert len(reads) > count
    finally:
        rv._cached_terminal_dependencies.cache_clear()
