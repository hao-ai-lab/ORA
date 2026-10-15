"""The two RoboDojo kernel path sets, and how they relate.

Each set is a content-hashed gate: editing a member requires assessing the
changed dependency and recording qualified reuse or a bounded live regression. So the
comment beside them is the document a maintainer consults when placing a new
file -- and it was wrong. It asserted that the worker set's first eight
entries "are exactly" the motion set. They are not: the motion set is a
nine-of-eighteen subset whose ordering is independent, sharing none of the
worker set's first eight entries.

A maintainer following that claim places a new file by position instead of by
meaning. `ora/contracts/pose.py` sits on the worker's executed servo
path and was in neither set, which is exactly the failure that claim invites,
until the import-closure check below listed it in the worker set.
"""

from __future__ import annotations

from ora.orchestrator.robodojo_qualification import (
    WORKER_KERNEL_PATHS, ROBODOJO_MOTION_KERNEL_PATHS,
)


def test_the_motion_set_is_a_subset_and_not_a_prefix() -> None:
    worker = list(WORKER_KERNEL_PATHS)
    motion = list(ROBODOJO_MOTION_KERNEL_PATHS)
    assert set(motion) <= set(worker), sorted(set(motion) - set(worker))
    # Pin the relation the comment now states, so a future edit that makes the
    # motion set a genuine prefix has to update the comment with it. The two
    # tuples share members near the front -- four of the first eight -- which
    # is exactly why reading membership off a position looks plausible and is
    # wrong.
    assert worker[: len(motion)] != motion
    assert set(worker[:8]) & set(motion[:8])
    assert sorted(set(motion) - set(worker[: len(motion)]))


def test_both_sets_are_closed_and_free_of_duplicates() -> None:
    for label, paths in (
        ("worker", WORKER_KERNEL_PATHS),
        ("motion", ROBODOJO_MOTION_KERNEL_PATHS),
    ):
        listed = list(paths)
        assert len(listed) == len(set(listed)), label
    # Only the worker set is sorted. The motion set leads with the host-safe
    # benchmark module and then runs alphabetically, and nothing depends on
    # either order -- which is the point: order carries no meaning here, so
    # do not let a reader infer one.
    assert list(WORKER_KERNEL_PATHS) == sorted(WORKER_KERNEL_PATHS)
    assert list(ROBODOJO_MOTION_KERNEL_PATHS) != sorted(
        ROBODOJO_MOTION_KERNEL_PATHS
    )


def test_every_member_is_a_tracked_source_file_that_exists() -> None:
    from ora.orchestrator.paths import PROJECT_ROOT

    for path in {*WORKER_KERNEL_PATHS, *ROBODOJO_MOTION_KERNEL_PATHS}:
        resolved = PROJECT_ROOT / path
        assert resolved.is_file(), path
        assert not resolved.is_symlink(), path


def test_historical_kernel_uses_its_own_manifest_and_requires_every_file(tmp_path):
    import hashlib
    import subprocess
    import pytest
    from ora.orchestrator.robodojo_kernel import worker_kernel_sha256
    from ora.environment.benchmarks.robodojo import RoboDojoBenchmarkError

    def git(*args):
        return subprocess.check_output(['git', *args], cwd=tmp_path, text=True).strip()
    git('init', '-q')
    git('config', 'user.name', 'Test')
    git('config', 'user.email', 'test@example.invalid')
    path = 'ora/environment/backends/robodojo/session.py'
    file = tmp_path/path
    file.parent.mkdir(parents=True)
    file.write_text('class RobodojoSession: pass\n')
    manifest = tmp_path/'ora/orchestrator/robodojo_qualification.py'
    manifest.parent.mkdir(parents=True)
    manifest.write_text(f'WORKER_KERNEL_PATHS = ({path!r},)\n')
    git('add', '.')
    git('commit', '-qm', 'historical monolith')
    old = git('rev-parse', 'HEAD')
    encoded = path.encode(); data = file.read_bytes()
    expected = hashlib.sha256(len(encoded).to_bytes(4, 'big') + encoded
                              + len(data).to_bytes(8, 'big') + data).hexdigest()
    assert worker_kernel_sha256(old, project_root=tmp_path) == expected
    file.unlink()
    git('add', '-u'); git('commit', '-qm', 'missing declared owner')
    with pytest.raises(RoboDojoBenchmarkError, match='cannot resolve'):
        worker_kernel_sha256(git('rev-parse', 'HEAD'), project_root=tmp_path)
    assert worker_kernel_sha256(old, project_root=tmp_path) == expected


# The manifest is a hand-written list; the worker's import graph is not. The
# EXP11A g005/g006 recovery commits edited ``backends/robodojo/recovery.py``,
# which ``session.py`` imports, and every card and lane gate stayed "current"
# because neither recovery.py nor ``wait.py`` was listed. The tests below
# derive the import closure statically and hold the manifest to it.

BACKEND = 'ora/environment/backends/robodojo/'


def _write_tree(root, files):
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def test_import_closure_follows_every_static_project_import(tmp_path):
    from ora.orchestrator.import_closure import import_closure, tree_reader

    _write_tree(tmp_path, {
        'ora/__init__.py': '',
        'ora/pkg/__init__.py': 'from .heavy import everything\n',
        'ora/pkg/heavy.py': 'everything = 1\n',
        'ora/pkg/entry.py': (
            'import json\nimport numpy as np\nfrom env.global_configs import X\n'
            'from . import sibling\nfrom .helpers import run\n'
            'from ora.other import value\n'
            'def later(kind):\n'
            '    if kind == "lazy":\n'
            '        from .lazy import go\n'
            '        return go\n'
            '    import ora.deep.module\n'),
        'ora/pkg/sibling.py': 'from ..other import value\n',
        'ora/pkg/helpers.py': '',
        # Names an attribute of the package, so the package's own module runs.
        'ora/pkg/lazy.py': 'from ora.pkg import everything\n',
        'ora/other.py': '',
        'ora/deep/__init__.py': 'import ora.pkg.heavy\n',
        'ora/deep/module.py': '',
    })
    entry = 'ora/pkg/entry.py'
    closure = import_closure((entry,), read=tree_reader(tmp_path))
    # Function-level and conditional imports count; third-party ones do not.
    assert closure[entry] == (
        'ora/deep/module.py', 'ora/other.py',
        'ora/pkg/helpers.py', 'ora/pkg/lazy.py',
        'ora/pkg/sibling.py',
    )
    assert closure['ora/pkg/lazy.py'] == ('ora/pkg/__init__.py',)
    # A submodule import binds that submodule, not every parent package.
    assert 'ora/deep/__init__.py' not in closure
    assert set(closure) == {
        entry, 'ora/deep/module.py', 'ora/other.py',
        'ora/pkg/helpers.py', 'ora/pkg/lazy.py',
        'ora/pkg/sibling.py', 'ora/pkg/__init__.py',
        'ora/pkg/heavy.py',
    }

    pruned = import_closure((entry,), read=tree_reader(tmp_path),
                            follow=lambda importer, imported: not imported.endswith('/lazy.py'))
    assert not {'ora/pkg/lazy.py', 'ora/pkg/heavy.py'} & set(pruned)

    local = import_closure((entry,), read=tree_reader(tmp_path), package='ora.pkg')
    assert 'ora/other.py' not in local and 'ora/deep/module.py' not in local
    assert 'ora/pkg/heavy.py' in local


def test_import_closure_resolves_constant_dynamic_imports_and_import_time_chains(tmp_path):
    import pytest
    from ora.orchestrator.import_closure import (
        ImportClosureError, import_closure, tree_reader,
    )

    _write_tree(tmp_path, {
        'ora/__init__.py': '',
        'ora/pkg/__init__.py': 'from .aggregate import X\n',
        'ora/pkg/aggregate.py': '',
        'ora/pkg/entry.py': (
            'import importlib\nfrom importlib import import_module\n'
            'from .eager import E\n'
            'def later(task):\n'
            '    importlib.import_module("ora.pkg.dyn")\n'
            '    import_module("ora.pkg.named")\n'
            '    __import__("ora.pkg.dunder")\n'
            '    importlib.import_module(f"task.RoboDojo.tasks.{task}")\n'
            '    from .lazy import L\n'),
        'ora/pkg/eager.py': '', 'ora/pkg/lazy.py': '',
        'ora/pkg/dyn.py': '', 'ora/pkg/named.py': '',
        'ora/pkg/dunder.py': '',
    })
    entry = 'ora/pkg/entry.py'
    closure = import_closure((entry,), read=tree_reader(tmp_path))
    assert {'ora/pkg/dyn.py', 'ora/pkg/named.py',
            'ora/pkg/dunder.py'} <= set(closure[entry])
    # What Python executes on import: module-level imports plus every parent
    # package's __init__, and not the function-level ones.
    chain = import_closure((entry,), read=tree_reader(tmp_path),
                           module_level_only=True, implicit_parents=True)
    assert set(chain) == {entry, 'ora/__init__.py', 'ora/pkg/__init__.py',
                          'ora/pkg/aggregate.py', 'ora/pkg/eager.py'}

    # A module name the source does not fix statically is refused, not
    # skipped; only an f-string whose fixed prefix names a third-party
    # top-level package (bootstrap's task import) is left alone.
    for body in ('importlib.import_module(".dyn", "ora.pkg")\n',
                 'def f(name):\n    return importlib.import_module(f"ora.pkg.{name}")\n',
                 'def f(name):\n    return importlib.import_module(name)\n',
                 'def f(parts):\n    return __import__(".".join(parts))\n',
                 'def f(task):\n    return importlib.import_module(f"{task}.tasks")\n',
                 'def f():\n    return importlib.import_module()\n'):
        (tmp_path / entry).write_text('import importlib\n' + body)
        with pytest.raises(ImportClosureError, match='dynamic'):
            import_closure((entry,), read=tree_reader(tmp_path))


def test_import_closure_refuses_what_it_cannot_resolve(tmp_path):
    import pytest
    from ora.orchestrator.import_closure import (
        ImportClosureError, import_closure, tree_reader,
    )

    _write_tree(tmp_path, {'ora/pkg/entry.py': 'def f():\n    from .missing import x\n'})
    with pytest.raises(ImportClosureError, match='ora.pkg.missing'):
        import_closure(('ora/pkg/entry.py',), read=tree_reader(tmp_path))
    with pytest.raises(ImportClosureError, match='absent.py'):
        import_closure(('ora/pkg/absent.py',), read=tree_reader(tmp_path))


def test_manifest_gaps_prune_only_reviewed_dispatcher_edges(tmp_path):
    from ora.orchestrator.robodojo_kernel_review import (
        ROBODOJO_WORKER_ENTRY_POINTS, kernel_manifest_gaps,
    )

    worker, factory, bootstrap, session = ROBODOJO_WORKER_ENTRY_POINTS
    files = {
        worker: ('def build(backend):\n'
                 '    if backend == "robotwin":\n'
                 '        from ora.environment.backends.robotwin.session import S\n'
                 '    from ora.environment.backends.robodojo.session import RobodojoSession\n'),
        factory: 'from .controllers.hybrid_motion import load\n',
        bootstrap: '',
        session: ('from . import sparse_motion\n'
                  'from ora.environment.controllers.robodojo_arx import C\n'
                  'from ora.contracts.refusal import V\n'),
        BACKEND + 'sparse_motion.py': 'def run():\n    from .recovery import execute_recovery\n',
        BACKEND + 'recovery.py': '',
        'ora/environment/backends/robotwin/session.py': 'from ...controllers.servo import S\n',
        'ora/environment/controllers/robodojo_arx.py': '',
        'ora/environment/controllers/hybrid_motion.py': '',
        'ora/environment/controllers/servo.py': '',
        'ora/environment/novel.py': '',
        'ora/contracts/refusal.py': '',
        'ora/contracts/unlisted.py': '',
    }
    _write_tree(tmp_path, files)
    listed = (worker, factory, bootstrap, session, BACKEND + 'sparse_motion.py',
              'ora/environment/controllers/robodojo_arx.py', BACKEND + 'recovery.py')
    # The reviewed other-backend edges are pruned; everything else the worker
    # reaches is listed or a reviewed exclusion.
    assert kernel_manifest_gaps(tmp_path, manifest=listed[:-1]) == (BACKEND + 'recovery.py',)
    assert kernel_manifest_gaps(tmp_path, manifest=listed) == ()
    # A reviewed exclusion is not bound, but its own imports are still
    # followed: a new unlisted import inside it is a gap.
    (tmp_path / 'ora/contracts/refusal.py').write_text(
        'from ora.contracts.unlisted import U\n')
    assert kernel_manifest_gaps(tmp_path, manifest=listed) == ('ora/contracts/unlisted.py',)
    (tmp_path / 'ora/contracts/refusal.py').write_text('')
    # A new dispatcher import is unreviewed until listed or pruned.
    (tmp_path / factory).write_text(files[factory] + 'from .novel import N\n')
    assert kernel_manifest_gaps(tmp_path, manifest=listed) == ('ora/environment/novel.py',)


def test_worker_kernel_covers_the_whole_worker_import_closure():
    from ora.orchestrator.robodojo_kernel_review import (
        kernel_manifest_gaps, kernel_review_drift,
    )

    gaps = kernel_manifest_gaps()
    assert not gaps, f'imported by the RoboDojo worker but neither listed nor reviewed: {gaps}'
    drift = kernel_review_drift()
    assert not drift, drift


def test_worker_closure_reaches_the_executor_helpers_and_no_other_backend():
    from ora.orchestrator.robodojo_kernel_review import (
        ROBODOJO_WORKER_ENTRY_POINTS, worker_import_closure,
    )

    closure = worker_import_closure()
    assert BACKEND + 'session.py' in closure['ora/environment/worker.py']
    for helper in (BACKEND + 'recovery.py', BACKEND + 'wait.py', BACKEND + 'staged_contact.py',
                   'ora/contracts/robodojo_arx.py', 'ora/contracts/wait.py',
                   'ora/contracts/pose.py', 'ora/contracts/skill_sequence.py'):
        assert helper in closure, helper
    # skill_sequence decides which skill-sequence children the session admits
    # and in what order, and no qualification check exercises it: bound.
    assert 'ora/contracts/skill_sequence.py' in WORKER_KERNEL_PATHS
    assert not [path for path in closure if path.startswith('ora/environment/backends/')
                and not path.startswith(BACKEND)]
    assert set(ROBODOJO_WORKER_ENTRY_POINTS) <= set(WORKER_KERNEL_PATHS)


#: Calls that only declare values at import time. Anything else a module
#: executes on import (environment, sys.path, patching, I/O) must not hide in
#: a module the kernel does not bind.
DECLARATIVE_CALLS = frozenset({
    'dataclass', 'field', 'range', 'frozenset', 'tuple', 'dict', 'set', 'list',
    'len', 'sorted', 'enumerate', 'zip', 'str', 'int', 'float', 'bool',
})


def _import_time_calls(tree):
    import ast

    local = {node.name for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    pending = list(tree.body)
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            pending.extend(node.decorator_list)
            pending.extend(default for default in node.args.defaults + node.args.kw_defaults if default)
            continue
        if isinstance(node, ast.Lambda):
            continue
        if isinstance(node, ast.Call):
            func = node.func
            if not ((isinstance(func, ast.Name) and (func.id in DECLARATIVE_CALLS or func.id in local))
                    or (isinstance(func, ast.Attribute) and func.attr in {'items', 'keys', 'values'})):
                yield ast.unparse(func)
        pending.extend(ast.iter_child_nodes(node))


def test_unbound_modules_on_the_worker_startup_chain_only_declare():
    """Parent ``__init__`` and pruned dispatcher imports still run on import.

    The closure leaves them out because they decide nothing; this checks the
    premise on the chain Python actually executes for ``-m
    ora.environment.worker`` and the RoboDojo backend import.
    """
    import ast
    from ora.orchestrator.import_closure import import_closure, tree_reader
    from ora.orchestrator.paths import PROJECT_ROOT
    from ora.orchestrator.robodojo_kernel_review import ROBODOJO_WORKER_ENTRY_POINTS

    chain = import_closure(ROBODOJO_WORKER_ENTRY_POINTS, read=tree_reader(PROJECT_ROOT),
                           module_level_only=True, implicit_parents=True)
    unbound = sorted(set(chain) - set(WORKER_KERNEL_PATHS))
    for expected in ('ora/environment/__init__.py',
                     'ora/environment/controllers/__init__.py',
                     'ora/environment/controllers/curobo_motion.py',
                     'ora/environment/controllers/hybrid_motion.py',
                     'ora/environment/definitions.py',
                     'ora/environment/synthetic.py'):
        assert expected in unbound, expected
    effects = {path: calls for path in unbound
               if (calls := sorted(set(_import_time_calls(ast.parse((PROJECT_ROOT / path).read_text())))))}
    assert not effects, effects


def test_the_motion_set_binds_the_recovery_its_gate_executes():
    from ora.orchestrator.paths import PROJECT_ROOT

    gate = (PROJECT_ROOT / BACKEND / 'motion_qualification.py').read_text()
    assert '"recover_motion"' in gate
    assert BACKEND + 'recovery.py' in ROBODOJO_MOTION_KERNEL_PATHS
    assert 'ora/contracts/robodojo_arx.py' in ROBODOJO_MOTION_KERNEL_PATHS


def _kernel_repo(tmp_path):
    import subprocess

    def git(*args):
        return subprocess.check_output(['git', *args], cwd=tmp_path, text=True).strip()
    git('init', '-q')
    git('config', 'user.name', 'Test')
    git('config', 'user.email', 'test@example.invalid')

    def commit(files, message):
        _write_tree(tmp_path, files)
        git('add', '-A')
        git('commit', '-qm', message)
        return git('rev-parse', 'HEAD')
    return commit


def _manifest(*paths):
    return {'ora/orchestrator/robodojo_kernel.py': f'WORKER_KERNEL_PATHS = {tuple(paths)!r}\n'}


def test_a_record_survives_only_a_manifest_extension_by_unchanged_files(tmp_path):
    from ora.orchestrator.robodojo_kernel import worker_kernel_extension, worker_kernel_sha256

    def extension(recorded, plan, measured_commit=None):
        return worker_kernel_extension(recorded, measured_commit=measured_commit or measured,
                                       plan_commit=plan, project_root=tmp_path)

    a, b, c = 'ora/a.py', 'ora/b.py', 'ora/c.py'
    commit = _kernel_repo(tmp_path)
    measured = commit({a: 'A = 1\n', b: 'B = 1\n', c: 'C = 1\n', **_manifest(a, c)}, 'measured')
    recorded = worker_kernel_sha256(measured, project_root=tmp_path)
    extended = commit(_manifest(a, b, c), 'list b')
    assert worker_kernel_sha256(extended, project_root=tmp_path) != recorded
    assert extension(recorded, extended) == (b,)
    # Delete one, add one: not an extension, even though every byte agrees.
    swapped = commit(_manifest(a, b), 'unlist c, list b')
    assert extension(recorded, swapped) is None
    same_manifest = commit(_manifest(a, c), 'relist the measured set')
    assert extension(recorded, same_manifest) is None
    commit(_manifest(a, b, c), 'list b again')
    added_changed = commit({b: 'B = 2\n'}, 'edit b')
    assert extension(recorded, added_changed) is None
    measured_changed = commit({a: 'A = 2\n', b: 'B = 1\n'}, 'edit a')
    assert extension(recorded, measured_changed) is None
    assert extension('0' * 64, extended) is None
    assert extension(recorded, extended, measured_commit='c' * 40) is None


def test_an_added_file_absent_at_measurement_refuses_the_extension(tmp_path):
    from ora.orchestrator.robodojo_kernel import worker_kernel_extension, worker_kernel_sha256

    a, b = 'ora/a.py', 'ora/b.py'
    commit = _kernel_repo(tmp_path)
    measured = commit({a: 'A = 1\n', **_manifest(a)}, 'measured')
    recorded = worker_kernel_sha256(measured, project_root=tmp_path)
    extended = commit({b: 'B = 1\n', **_manifest(a, b)}, 'add and list b')
    assert worker_kernel_extension(recorded, measured_commit=measured, plan_commit=extended,
                                   project_root=tmp_path) is None


def test_record_identity_reports_how_it_admitted_the_kernel(tmp_path, monkeypatch):
    """End to end through the real admission code, against a real history."""
    import pytest
    from ora.environment.benchmarks.robodojo import RoboDojoBenchmarkError
    from ora.orchestrator import robodojo_kernel, robodojo_qualification as qualification

    a, b, c = 'ora/a.py', 'ora/b.py', 'ora/c.py'
    commit = _kernel_repo(tmp_path)
    measured = commit({a: 'A = 1\n', b: 'B = 1\n', c: 'C = 1\n', **_manifest(a, c)}, 'measured')
    unrelated = commit({'README.md': 'docs\n'}, 'unrelated')
    extended = commit(_manifest(a, b, c), 'list b')
    swapped = commit(_manifest(a, b), 'unlist c')
    monkeypatch.setattr(robodojo_kernel, '_PROJECT_ROOT', tmp_path)
    recorded = robodojo_kernel.worker_kernel_sha256(measured)
    record = {'schema': 'schema', 'passed': True, 'harness_commit': measured,
              'worker_kernel_sha256': recorded}
    snapshot = dict(record)

    def admit(plan):
        return qualification._validate_record_identity(record, schema='schema', harness_commit=plan,
                                                       label='card')
    assert admit(unrelated) == {
        'mode': 'identical_digest', 'measured_commit': measured, 'plan_commit': unrelated,
        'recorded_worker_kernel_sha256': recorded, 'plan_worker_kernel_sha256': recorded,
        'added_paths': []}
    assert admit(extended) == {
        'mode': 'manifest_extension_byte_identical', 'measured_commit': measured,
        'plan_commit': extended, 'recorded_worker_kernel_sha256': recorded,
        'plan_worker_kernel_sha256': robodojo_kernel.worker_kernel_sha256(extended),
        'added_paths': [b]}
    with pytest.raises(RoboDojoBenchmarkError, match='kernel changed'):
        admit(swapped)
    assert record == snapshot
