"""Generation identity is immutable evidence and never a new launch authority.

The mechanism is tested against a synthetic lineage in
``tests/data/identity-registry-fixture.json``: family EXP01A has a main line
(g001-g004) with a ``low`` fork and a ``vla`` fork, and family EXP01B receives
EXP01A/main/g003's plan-bound runs by reviewed migration, then continues in
its own namespace with a two-parent generation. The shipped registry is only
checked for loadability.
"""
from __future__ import annotations

import copy
import functools
import json
from pathlib import Path

import pytest
import yaml

from ora.orchestrator import experiment_identity as identity
from ora.orchestrator import spec as runtime
from ora.orchestrator.cli import main

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / 'tests' / 'data' / 'identity-registry-fixture.json'

#: Reviewed fixture migrations: (cohort, source, recorded generation, shown generation).
MIGRATIONS = [
    ('study-a-g003', '3' * 40, 'EXP01A/main/g003', 'EXP01B/main/g001'),
    ('study-a-g003-fill', '4' * 40, 'EXP01A/main/g003', 'EXP01B/main/g001'),
]


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding='utf-8'))


def _write_catalog(root: Path, catalog: dict) -> None:
    target = root / identity.CATALOG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(catalog))


@pytest.fixture
def catalog_root(tmp_path):
    """A scratch project root whose registry is the fixture catalog."""
    root = tmp_path / 'catalog-root'
    _write_catalog(root, _fixture())
    return root


@pytest.fixture
def fixture_catalog(catalog_root):
    return identity.load_catalog(catalog_root)


@pytest.fixture
def viewer_client(catalog_root, monkeypatch):
    """The Viewer app reading the fixture catalog instead of the shipped registry."""
    from viewer import app as viewer
    from viewer import experiment_identity as viewer_identity
    from viewer import registry as viewer_registry
    loaded = identity.load_catalog(catalog_root)
    monkeypatch.setattr(viewer_registry, '_identity_catalog', lambda: loaded)
    monkeypatch.setattr(viewer_identity, 'load_catalog', lambda: loaded)

    def client(roots):
        monkeypatch.setattr(viewer, 'viewer_run_roots', lambda: roots)
        return viewer.app.test_client()
    return client


def test_shipped_registry_loads():
    catalog = identity.load_catalog(ROOT)
    assert catalog['schema'] == identity.CATALOG_SCHEMA
    for key, node in catalog['generations'].items():
        assert identity.identity_key(node['identity']) == key


def test_fixture_registry_loads_with_forks_and_multi_parent_generations(fixture_catalog):
    nodes = fixture_catalog['generations']
    parents = {k: v['identity']['parents'] for k, v in nodes.items()}
    # Forks start at g001 on their own branch and name where they came from.
    assert parents['EXP01A/low/g001'] == ['EXP01A/main/g002']
    assert parents['EXP01A/vla/g001'] == ['EXP01A/main/g001']
    # A migrated generation lists the recorded generation it relabels.
    for *_, recorded, target in MIGRATIONS:
        assert recorded in parents[target]
    assert parents['EXP01B/main/g003'] == ['EXP01B/main/g002', 'EXP01A/low/g001']
    assert {v['status'] for v in nodes.values()} == {'development', 'historical'}


@pytest.fixture
def project(tmp_path, monkeypatch):
    catalog = _fixture()
    catalog['namespaces']['identity-test'] = 'EXP01A'
    _write_catalog(tmp_path, catalog)
    for name, text in [('prompt.md', 'Move once.'), ('manual.md', '## Overview\nMove safely.'),
                       ('agent.md', 'Agent rules.')]:
        (tmp_path / name).write_text(text)
    monkeypatch.setattr(runtime, '_git', lambda *a, **k: 'a' * 40 if a[0] == 'rev-parse' else '')
    monkeypatch.setattr(runtime, 'PROJECT_ROOT', tmp_path)
    spec = {
        'schema': runtime.SCHEMA, 'experiment': 'identity-test', 'config': 'cell',
        'milestone': 'M0', 'output_workspace': 'outputs/identity-test',
        'task': 'synthetic', 'task_instruction': 'Move once.', 'seed': 0,
        'evidence_tier': 'diagnostic',
        'policy': {'kind': 'scripted', 'model': 'scripted', 'provider': 'trusted',
                   'reasoning_effort': 'none', 'actions': [{'x': 1}]},
        'runtime': {'kind': 'synthetic', 'workspace_bounds_m': {
            'x': [-1, 1], 'y': [-1, 1], 'z': [0, 2]}},
        'budgets': {'max_robot_decisions': 1, 'max_model_calls': 1,
                    'max_sim_steps': 1, 'wall_time_s': 10},
        'policy_context': {'codex_project': ['agent.md'], 'episode_prefix': ['prompt.md'],
                           'tool_manual': ['manual.md']},
        'qualification': {'required': False},
    }
    return tmp_path, catalog, spec


def test_new_identity_hashes_and_inherits_while_old_plan_is_unchanged(project):
    root, catalog, spec = project
    old = runtime.resolve_run_spec(spec, root=root)
    old_bytes = json.dumps(old, sort_keys=True)
    selected = catalog['generations']['EXP01A/main/g004']['identity']
    declaration = root / 'experiments/identity_test/experiment.yaml'
    declaration.parent.mkdir(parents=True)
    declaration.write_text(yaml.safe_dump({'experiment': 'identity-test',
                                           'experiment_identity': selected}))
    new = runtime.resolve_run_spec(spec, root=root)
    assert new['experiment_identity'] == selected
    assert new['plan_sha256'] != old['plan_sha256']
    for key in ('experiment', 'output_workspace', 'policy', 'runtime', 'policy_context'):
        assert new[key] == old[key]
    assert 'launch_admission' not in new
    assert runtime.validate_run_plan(new) == new
    tampered = copy.deepcopy(new)
    tampered['experiment_identity']['generation'] = 'g005'
    with pytest.raises(runtime.RunSpecError, match='hash'):
        runtime.validate_run_plan(tampered)
    # Validation of persisted plans does not depend on any future catalog.
    (root / identity.CATALOG_PATH).write_text('not a catalog')
    assert runtime.validate_run_plan(old) == old
    assert runtime.validate_run_plan(new) == new
    assert json.dumps(old, sort_keys=True) == old_bytes


def test_conflicting_or_unregistered_new_identity_refused(project):
    root, catalog, spec = project
    node = catalog['generations']['EXP01A/main/g004']['identity']
    spec['experiment_identity'] = copy.deepcopy(node)
    spec['experiment_identity']['generation'] = 'g999'
    with pytest.raises(runtime.RunSpecError, match='not registered'):
        runtime.resolve_run_spec(spec, root=root)
    declaration = root / 'experiments/identity_test/experiment.yaml'
    declaration.parent.mkdir(parents=True)
    declaration.write_text(yaml.safe_dump({'experiment': 'identity-test', 'experiment_identity': node}))
    with pytest.raises(runtime.RunSpecError, match='conflicts'):
        runtime.resolve_run_spec(spec, root=root)


def test_identity_of_another_family_is_refused_for_the_namespace(catalog_root):
    selected = identity.load_catalog(catalog_root)['generations']['EXP01B/main/g002']['identity']
    with pytest.raises(ValueError, match='namespace is not registered'):
        identity.bind_identity({'experiment': 'study-a', 'experiment_identity': selected}, root=catalog_root)
    assert identity.bind_identity({'experiment': 'study-b', 'experiment_identity': selected},
                                  root=catalog_root) == selected
    assert identity.bind_identity({'experiment': 'study-a'}, root=catalog_root) is None


@pytest.mark.parametrize('mutation', ['cycle', 'duplicate', 'missing_parent', 'wrong_family',
                                      'key_mismatch', 'unknown_family', 'bad_status', 'bad_namespace'])
def test_catalog_rejects_ambiguous_or_invalid_lineage(tmp_path, mutation):
    catalog = _fixture()
    nodes = catalog['generations']
    if mutation == 'cycle':
        nodes['EXP01A/main/g001']['identity']['parents'] = ['EXP01A/main/g003']
    elif mutation == 'duplicate':
        catalog['legacy_aliases'].append(copy.deepcopy(catalog['legacy_aliases'][0]))
    elif mutation == 'missing_parent':
        nodes['EXP01A/low/g001']['identity']['parents'] = ['EXP01A/missing/g001']
    elif mutation == 'wrong_family':
        catalog['namespaces']['study-a'] = 'EXP99Z'
    elif mutation == 'key_mismatch':
        nodes['EXP01A/main/g009'] = nodes.pop('EXP01A/main/g004')
    elif mutation == 'unknown_family':
        del catalog['families']['EXP01B']
    elif mutation == 'bad_status':
        nodes['EXP01A/main/g004']['status'] = 'active'
    else:
        catalog['namespaces']['Study A'] = 'EXP01A'
    _write_catalog(tmp_path, catalog)
    with pytest.raises(ValueError):
        identity.load_catalog(tmp_path)


def test_legacy_projection_is_source_exact_and_does_not_absorb_other_treatments(fixture_catalog):
    catalog = fixture_catalog
    alias = next(a for a in catalog['legacy_aliases'] if a['experiment'] == 'shared-pilot')
    plan = {'experiment': alias['experiment'], 'source': {'commit': alias['source_commit']}}
    original = copy.deepcopy(plan)
    assert identity.project_identity(plan, catalog)['generation_id'] == 'EXP01A/vla/g001'
    assert plan == original
    # The shared namespace is not registered: another source stays unmapped.
    plan['source']['commit'] = 'b' * 40
    assert identity.project_identity(plan, catalog)['binding'] == 'unmapped'
    # A registered namespace at an unknown source has family membership only.
    plan['experiment'] = 'study-a'
    assert identity.project_identity(plan, catalog) == {
        'binding': 'family_only', 'family': 'EXP01A', 'generation_id': None}
    plan['experiment_identity'] = {'schema': 'broken'}
    assert identity.project_identity(plan, catalog)['binding'] == 'invalid'


def test_source_declarations_select_successors_without_reopening_admission(catalog_root):
    catalog = identity.load_catalog(catalog_root)
    for namespace, key in [('study-a', 'EXP01A/main/g004'), ('study-b', 'EXP01B/main/g003')]:
        selected = catalog['generations'][key]['identity']
        declaration = catalog_root / 'experiments' / namespace.replace('-', '_') / 'experiment.yaml'
        declaration.parent.mkdir(parents=True)
        declaration.write_text(yaml.safe_dump({'experiment': namespace, 'experiment_identity': selected}))
        spec = {'experiment': namespace, 'launch_admission': {'cohort_id': 'unchanged', 'retry_allowed': False}}
        before = copy.deepcopy(spec)
        assert identity.bind_identity(spec, root=catalog_root) == selected
        assert spec == before


def _write_run(base: Path, namespace: str, stamp: str, plan: dict, result: dict) -> Path:
    run = base / namespace / 'runs' / f'{namespace}--{stamp}'
    run.mkdir(parents=True)
    (run / 'plan.json').write_text(json.dumps(plan))
    (run / 'result.json').write_text(json.dumps(result))
    return run


def test_viewer_groups_legacy_generations_and_preserves_outcomes(tmp_path, fixture_catalog, viewer_client):
    roots = []
    for namespace, cohort, verdict in [('study-a', 'study-a-g002', True),
                                       ('study-a-low', 'study-a-low-g001', False)]:
        alias = next(a for a in fixture_catalog['legacy_aliases']
                     if a['experiment'] == namespace and a['cohort_id'] == cohort)
        roots.append(_write_run(tmp_path, namespace, '20260101T000000Z', {
            'experiment': namespace, 'source': {'commit': alias['source_commit']},
            'launch_admission': {'cohort_id': alias['cohort_id']},
            'runtime': {'kind': 'robotwin'}, 'task': 'sample', 'evidence_tier': 'behavioral'},
            {'state': 'completed', 'official_success': verdict}))
    original = [(r / 'plan.json').read_bytes() for r in roots]
    client = viewer_client(roots)
    payload = client.get('/api/experiments/EXP01A').get_json()
    assert len(payload['runs']) == 2
    assert {r['official_success'] for r in payload['runs']} == {True, False}
    assert {r['experiment_identity']['generation_id'] for r in payload['runs']} == {
        'EXP01A/main/g002', 'EXP01A/low/g001'}
    page = client.get('/?view=all&family=EXP01A&generation=EXP01A/low/g001')
    assert page.status_code == 200
    assert roots[1].name.encode() in page.data and roots[0].name.encode() not in page.data
    assert client.get('/experiments/EXP01A').status_code == 200
    assert b'EXP01B' in client.get('/experiments').data
    assert client.get('/experiments/EXP99Z').status_code == 404
    assert [(r / 'plan.json').read_bytes() for r in roots] == original


@pytest.fixture
def cli_on_fixture(catalog_root, monkeypatch):
    monkeypatch.setattr(identity, 'command_experiments',
                        functools.partial(identity.command_experiments, root=catalog_root))


def test_cli_family_lookup_is_read_only(cli_on_fixture, capsys):
    assert main(['experiments', 'EXP01A', '--json']) == 0
    value = json.loads(capsys.readouterr().out)
    assert {'EXP01A/main/g001', 'EXP01A/low/g001', 'EXP01A/vla/g001'} <= set(value['generations'])
    assert value['namespaces'] == {'study-a': 'EXP01A', 'study-a-low': 'EXP01A'}
    assert main(['experiments']) == 0
    listing = capsys.readouterr().out
    assert 'EXP01B/main/g003\tdevelopment\tparents=EXP01B/main/g002,EXP01A/low/g001' in listing
    assert 'EXP01A/main/g001\thistorical\tparents=root' in listing
    assert main(['experiments', 'EXP99Z']) == 2


def test_shipped_registry_is_inspectable_from_the_cli(capsys):
    assert main(['experiments', '--json']) == 0
    assert json.loads(capsys.readouterr().out)['schema'] == 'ora.experiment-lineage-view.v1'


def test_long_reverse_ordered_lineage_has_no_recursion_limit(tmp_path):
    catalog = _fixture()
    for number in reversed(range(1, 2001)):
        key = f'EXP01A/long/g{number:03d}'
        catalog['generations'][key] = {'status': 'development', 'label': 'Long lineage test', 'identity': {
            'schema': identity.SCHEMA, 'family': 'EXP01A', 'branch': 'long',
            'generation': f'g{number:03d}',
            'parents': [f'EXP01A/long/g{number-1:03d}'] if number > 1 else []}}
    _write_catalog(tmp_path, catalog)
    assert identity.load_catalog(tmp_path)['generations']['EXP01A/long/g2000']


@pytest.mark.parametrize('value', [
    {'schema': 'wrong', 'family': 'EXP01A', 'branch': 'main', 'generation': 'g001', 'parents': []},
    {'schema': identity.SCHEMA, 'family': 'EXP01A', 'branch': 'main', 'generation': 'g000', 'parents': []},
    {'schema': identity.SCHEMA, 'family': 'exp01a', 'branch': 'main', 'generation': 'g001', 'parents': []},
    {'schema': identity.SCHEMA, 'family': 'EXP01A', 'branch': 'Main', 'generation': 'g001', 'parents': []},
    {'schema': identity.SCHEMA, 'family': 'EXP01A', 'branch': 'main', 'generation': 'g01', 'parents': []},
    {'schema': identity.SCHEMA, 'family': 'EXP01A', 'branch': 'main', 'generation': 'g002',
     'parents': ['EXP01A/main/g002']},
    {'schema': identity.SCHEMA, 'family': 'EXP01A', 'branch': 'main', 'generation': 'g002',
     'parents': ['EXP01A/main/g001', 'EXP01A/main/g001']},
    {'schema': identity.SCHEMA, 'family': 'EXP01A', 'branch': 'main', 'generation': 'g002',
     'parents': ['EXP01A/main/g000']},
    {'schema': identity.SCHEMA, 'family': 'EXP01A', 'branch': 'main', 'generation': 'g001',
     'parents': [], 'extra': True},
])
def test_identity_structure_is_validated_without_the_catalog(value):
    with pytest.raises(ValueError):
        identity.validate_identity(value)


def _bound_plan(catalog, cohort, source, recorded):
    return {'experiment': 'study-a', 'source': {'commit': source, 'dirty': False},
            'launch_admission': {'cohort_id': cohort},
            'experiment_identity': copy.deepcopy(catalog['generations'][recorded]['identity'])}


@pytest.mark.parametrize('cohort,source,recorded,target', MIGRATIONS)
def test_reviewed_migration_relabels_exact_plan_bound_runs_without_touching_plans(
        fixture_catalog, cohort, source, recorded, target):
    catalog = fixture_catalog
    plan = _bound_plan(catalog, cohort, source, recorded)
    original = copy.deepcopy(plan)
    projected = identity.project_identity(plan, catalog)
    assert projected == {
        'binding': 'reviewed_migration', 'family': 'EXP01B', 'generation_id': target,
        'identity': catalog['generations'][target]['identity'],
        'recorded': {'binding': 'plan', 'family': 'EXP01A', 'generation_id': recorded,
                     'identity': original['experiment_identity']}}
    assert plan == original
    # Only the exact reviewed record moves: another source, cohort, namespace or
    # recorded identity (including the same key with other parents) keeps its plan binding.
    changes = [{'source': {'commit': 'b' * 40, 'dirty': False}},
               {'launch_admission': {'cohort_id': cohort + '-r2'}},
               {'experiment': 'study-a-low'},
               {'experiment_identity': catalog['generations']['EXP01A/main/g002']['identity']},
               {'experiment_identity': {**original['experiment_identity'], 'parents': []}}]
    for change in changes:
        other = {**copy.deepcopy(original), **change}
        kept = identity.project_identity(other, catalog)
        assert kept['binding'] == 'plan' and kept['family'] == 'EXP01A'
        assert kept['identity'] == other['experiment_identity']


def test_reviewed_migration_can_relabel_a_historical_alias(tmp_path):
    catalog = _fixture()
    alias = next(a for a in catalog['legacy_aliases'] if a['cohort_id'] == 'study-a-low-g001')
    catalog['reviewed_migrations'].append({
        'experiment': alias['experiment'], 'cohort_id': alias['cohort_id'],
        'source_commit': alias['source_commit'], 'recorded_generation_id': alias['generation_id'],
        'generation_id': 'EXP01B/main/g003', 'decided': '2026-01-03', 'review': 'Fixture alias relabel.'})
    _write_catalog(tmp_path, catalog)
    loaded = identity.load_catalog(tmp_path)
    plan = {'experiment': alias['experiment'], 'source': {'commit': alias['source_commit']},
            'launch_admission': {'cohort_id': alias['cohort_id']}}
    projected = identity.project_identity(plan, loaded)
    assert projected['binding'] == 'reviewed_migration'
    assert projected['generation_id'] == 'EXP01B/main/g003'
    assert projected['recorded']['binding'] == 'historical_alias'
    assert projected['recorded']['generation_id'] == 'EXP01A/low/g001'


def test_catalog_without_reviewed_migrations_still_loads(tmp_path):
    catalog = _fixture()
    del catalog['reviewed_migrations']
    _write_catalog(tmp_path, catalog)
    loaded = identity.load_catalog(tmp_path)
    alias = loaded['legacy_aliases'][0]
    plan = {'experiment': alias['experiment'], 'source': {'commit': alias['source_commit']},
            'launch_admission': {'cohort_id': alias['cohort_id']}}
    assert identity.project_identity(plan, loaded)['binding'] == 'historical_alias'


@pytest.mark.parametrize('mutation', ['duplicate', 'unknown_target', 'unknown_recorded', 'self', 'short_source',
                                      'foreign_namespace', 'extra_field', 'undated', 'unreviewed', 'not_a_list'])
def test_catalog_rejects_invalid_reviewed_migration(tmp_path, mutation):
    catalog = _fixture()
    migrations = catalog['reviewed_migrations']
    entry = migrations[0]
    if mutation == 'duplicate':
        migrations.append({**copy.deepcopy(entry), 'generation_id': 'EXP01B/main/g002'})
    elif mutation == 'unknown_target':
        entry['generation_id'] = 'EXP01B/main/g999'
    elif mutation == 'unknown_recorded':
        entry['recorded_generation_id'] = 'EXP01A/main/g999'
    elif mutation == 'self':
        entry['generation_id'] = entry['recorded_generation_id']
    elif mutation == 'short_source':
        entry['source_commit'] = entry['source_commit'][:8]
    elif mutation == 'foreign_namespace':
        entry['experiment'] = 'study-b'
    elif mutation == 'extra_field':
        entry['note'] = 'unreviewed field'
    elif mutation == 'undated':
        entry['decided'] = 'yesterday'
    elif mutation == 'unreviewed':
        entry['review'] = ''
    else:
        catalog['reviewed_migrations'] = {'0': entry}
    _write_catalog(tmp_path, catalog)
    with pytest.raises(ValueError):
        identity.load_catalog(tmp_path)


def test_cli_lists_a_family_with_its_reviewed_migrations(cli_on_fixture, capsys):
    assert main(['experiments', 'EXP01B', '--json']) == 0
    value = json.loads(capsys.readouterr().out)
    assert set(value['generations']) == {'EXP01B/main/g001', 'EXP01B/main/g002', 'EXP01B/main/g003'}
    assert value['namespaces'] == {'study-b': 'EXP01B'}
    assert value['legacy_aliases'] == []
    assert {(m['cohort_id'], m['recorded_generation_id'], m['generation_id'])
            for m in value['reviewed_migrations']} == {(c, r, t) for c, _, r, t in MIGRATIONS}
    assert main(['experiments', 'EXP01A', '--json']) == 0
    value = json.loads(capsys.readouterr().out)
    # The recorded family lists the migrations away from it, not their targets.
    assert len(value['reviewed_migrations']) == len(MIGRATIONS)
    assert 'EXP01B/main/g001' not in value['generations']


def test_viewer_lists_migrated_runs_under_the_new_family_and_shows_the_recorded_identity(
        tmp_path, fixture_catalog, viewer_client):
    roots = []
    for index, (cohort, source, recorded, _) in enumerate(MIGRATIONS):
        plan = _bound_plan(fixture_catalog, cohort, source, recorded)
        plan.update(runtime={'kind': 'robotwin'}, task='sample', evidence_tier='behavioral')
        roots.append(_write_run(tmp_path, 'study-a', f'2026010{index + 1}T000000Z', plan,
                                {'state': 'incomplete', 'official_success': None}))
    # The same recorded generation at an unreviewed source stays under EXP01A.
    plan = _bound_plan(fixture_catalog, 'study-a-g003', 'b' * 40, 'EXP01A/main/g003')
    plan.update(runtime={'kind': 'robotwin'}, task='sample', evidence_tier='behavioral')
    roots.append(_write_run(tmp_path, 'study-a', '20260109T000000Z', plan,
                            {'state': 'completed', 'official_success': False}))
    original = [(r / 'plan.json').read_bytes() for r in roots]
    client = viewer_client(roots)
    runs = client.get('/api/experiments/EXP01B').get_json()['runs']
    assert sorted((r['experiment_identity']['recorded']['generation_id'], r['experiment_identity']['generation_id'])
                  for r in runs) == [('EXP01A/main/g003', 'EXP01B/main/g001')] * 2
    kept = client.get('/api/experiments/EXP01A').get_json()['runs']
    assert [r['experiment_identity']['binding'] for r in kept] == ['plan']
    assert b'2 runs recorded under EXP01A' in client.get('/experiments/EXP01A').data
    listing = client.get('/?view=all&family=EXP01B&generation=EXP01B/main/g001')
    assert listing.status_code == 200
    assert roots[0].name.encode() in listing.data and roots[2].name.encode() not in listing.data
    assert b'recorded EXP01A/main/g003' in listing.data
    assert [(r / 'plan.json').read_bytes() for r in roots] == original
