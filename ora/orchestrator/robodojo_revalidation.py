"""Offline reuse of unchanged Dojo terminal evidence plus a current scene smoke.

This is representative launch admission. It retains the original card and GPU
identity, and never turns a two-action smoke into a full terminal/motion sweep.
"""
from __future__ import annotations
import ast
import functools
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from ora.environment.benchmarks.robodojo import (
    RoboDojoBenchmarkError, configuration_binding, card_configuration_binding,
    source_revisions, validate_scene_registration,
)
from . import robodojo_qualification as legacy

SCHEMA = 'ora.robodojo-revalidation.v1'
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = 'ora/environment/benchmarks/robodojo.py'
SESSION = 'ora/environment/backends/robodojo/session.py'
INTERACTION = 'ora/environment/backends/robodojo/interaction_control.py'
SPARSE = 'ora/environment/backends/robodojo/sparse_motion.py'
# Pinned upstream tasks whose constructors do not enable auxiliary playback.
# Future tasks are conservative until their interaction semantics are reviewed.
NONINTERACTIVE_TASKS = frozenset((
    'arrange_largest_number', 'build_tower', 'classify_objects',
    'classify_objects_by_language', 'fold_clothes', 'hang_mugs', 'insert_tubes',
    'organize_table', 'pack_objects_into_box', 'push_T', 'put_bottles_into_dustbin',
    'stack_bowls', 'stack_blocks',
))
# Only source-declared, directly imported bases are admitted. Historical
# monolithic sessions need none of these files to exist.
TERMINAL_OWNERS = {
    'RobodojoRoundMixin': ('ora/environment/backends/robodojo/round_lifecycle.py', 'round_lifecycle'),
    'RobodojoObservationMixin': ('ora/environment/backends/robodojo/observation.py', 'observation'),
    'RobodojoExecutionCaptureMixin': ('ora/environment/backends/robodojo/execution_capture.py', 'execution_capture'),
}

# The scene smoke binds the whole worker kernel -- which is held to the
# worker's import closure (orchestrator/robodojo_kernel_review.py) -- plus the
# host-side action executor the smoke drives. Former extras the closure check
# moved into the kernel are no longer repeated here.
DEPENDENCIES = tuple(sorted(set(legacy.WORKER_KERNEL_PATHS) | {
    'ora/orchestrator/action_executor.py',
}))
# Terminal/control closure members a card deliberately does not bind, with the
# review. They are leaves: their own imports are not followed either. The
# safety premise is that a card is only ever reused through a fresh scene
# smoke whose DEPENDENCIES bind the whole kernel, so every entry here must be
# a DEPENDENCIES member (tests/environments/robodojo/test_robodojo_revalidation.py checks it).
CARD_EXCLUSIONS = {
    'ora/environment/backends/robodojo/motion_adapter.py':
        'align_plan planning adapter (probe_motion runs through sparse_motion, not this); its plans '
        'are returned to the policy, never commanded',
    'ora/environment/controllers/alignment.py':
        'read-only align_plan planning over that same adapter',
    'ora/environment/backends/robodojo/robot_asset_geometry.py':
        'read-only loaded-geometry measurement at start; each scene smoke re-measures it and '
        'revalidate_geometry_binding compares it',
    'ora/contracts/robot_geometry.py':
        'the verification contract of that same read-only geometry measurement',
    'ora/environment/backends/robodojo/graphics_compat.py':
        'process-local Vulkan allocation limit, not terminal or control code; the scene smoke that '
        'every card reuse requires binds it through DEPENDENCIES, and preflight re-compares its readback',
}
TERMINAL_FILES = (
    'ora/environment/base.py',
    'ora/environment/backends/robodojo/bridge.py',
    'ora/environment/backends/robodojo/common.py',
    'ora/environment/backends/robodojo/sparse_motion.py',
    'ora/environment/controllers/robodojo_arx.py',
)
SPARSE = 'ora/environment/backends/robodojo/sparse_motion.py'
TERMINAL_METHODS = (
    '_clock', '_reset_environment', '_scene_registration', '_bind_reset_state',
    'restart_round', '_official_lifecycle_state', '_require_started',
    '_run_native_control', '_conformance_tick', '_persist_receipt',
    'is_terminal', 'status', '_fault', 'abort', '_official_outcome',
    'persist_official_verdict', 'close',
)
#: Mirrors robodojo_scene_smoke.CPU_SINGLE_ROUND_RESET (this checker imports no producer).
CPU_SINGLE_ROUND_RESET = 'not_applicable_cpu_single_round'
CHECKS = ('device_identity','loaded_geometry','rgbd','left_composite',
          'right_composite','read_only_alignment','round_reset')
# A CPU-physics smoke's in-place round_reset proves lifecycle counters and
# identities only: upstream's CPU reset leaves respawned rigid objects
# gravity/collision-disabled there, which no smoke check observes. CPU sessions
# are therefore single-round or layout-scheduled (robodojo_physics_attestation).
CPU_ROUND_RESET_BOUNDARY = ('CPU in-place round_reset object physics (respawned rigid objects may be '
                            'gravity/collision-disabled; CPU sessions run one round or a layout schedule)')


def revalidate_geometry_binding(current: dict, original: dict, *, source_commit: str,
                                verification: dict) -> dict:
    """Reuse measured collision geometry across wording/provenance revisions.

    Compare the actual worker payload and controlled-frame geometry, not card
    IDs. Nominal pad/encoder facts never become measured aperture evidence.
    """
    require(verification.get('passed') is True
            and verification.get('descriptor_sha256') == original.get('sha256'),
            'original loaded geometry is not bound to the original descriptor')
    if current == original:
        return {'original_descriptor': original, 'current_descriptor': current,
                'comparison': 'identical_descriptor_binding'}
    from ora.agent.composition.embodiment import validate_embodiment_descriptor
    from ora.agent.composition.embodiment_geometry import load_public_geometry_binding
    old_bytes = source_bytes(source_commit, original['path'])
    require(digest(old_bytes) == original['sha256'], 'original geometry descriptor source drift')
    old = validate_embodiment_descriptor(json.loads(old_bytes))
    new_binding = load_public_geometry_binding(current, project_root=PROJECT_ROOT)
    new = validate_embodiment_descriptor(json.loads((PROJECT_ROOT/current['path']).read_text()))
    old_geometry = {g['end_effector_id']:g['contact_geometry'] for g in old['geometry']['grippers']}
    require(old_geometry == new_binding['geometry'], 'physical robot collision geometry changed')
    def frames(descriptor):
        return [{k:v for k,v in point.items() if k != 'description'}
                for point in descriptor['control']['controlled_points']]
    require(old['topology'] == new['topology'] and frames(old) == frames(new),
            'robot topology or controlled-frame geometry changed')
    measured_hash = digest(json.dumps(old_geometry,sort_keys=True,separators=(',',':')).encode())
    require(verification.get('observed_geometry_sha256') == measured_hash,
            'loaded geometry measurement does not match the original source geometry')
    return {'original_descriptor':original, 'current_descriptor':current,
            'comparison':'identical_measured_collision_geometry_and_controlled_frames',
            'observed_geometry_sha256':measured_hash,
            'claim_boundary':'original loaded-geometry measurement; new wording and nominal parameters are not physical aperture calibration'}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RoboDojoBenchmarkError(message)


def source_bytes(commit: str, path: str) -> bytes:
    return subprocess.check_output(['git','show',f'{commit}:{path}'],cwd=PROJECT_ROOT)


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def dependency_map(commit: str, task: str | None = None) -> dict[str, str]:
    result = {}
    for path in DEPENDENCIES:
        if path == INTERACTION and task in NONINTERACTIVE_TASKS:
            continue
        try:
            data = source_bytes(commit,path)
        except subprocess.CalledProcessError as exc:
            # Source topology itself is a dependency. Preserve an explicit
            # absence marker rather than dropping a missing historical file
            # or treating its absence as proof of equivalent execution.
            from .robodojo_kernel import worker_kernel_paths
            declared = worker_kernel_paths(commit, project_root=PROJECT_ROOT)
            if path not in legacy.WORKER_KERNEL_PATHS or path in declared:
                raise RoboDojoBenchmarkError(f'missing declared execution dependency {path}') from exc
            result[path] = 'absent_from_producer_manifest'
            continue
        if path in (SESSION, SPARSE):
            data = ast.dump(_project_noninteractive(ast.parse(data), task), include_attributes=False).encode()
        if path.endswith('/robodojo/bridge.py') and task is not None:
            data = _bridge_terminal_source(data, task)
        if path == BENCHMARK:
            # Host-only schema admission does not execute in the plant. All
            # worker-consumed constants, task limits and bindings stay hashed.
            tree = ast.parse(data)
            tree.body = [n for n in tree.body if not (
                isinstance(n, ast.FunctionDef) and n.name == 'validate_runtime_extension')]
            _select_task_limit(tree, task)
            data = ast.dump(tree,include_attributes=False).encode()
        result[path] = digest(data)
    return result


def _project_noninteractive(tree: ast.Module, task: str | None) -> ast.Module:
    """Exclude only guarded auxiliary playback for reviewed noninteractive tasks.

    Imports stay inside the exact inactive guard. Else arms, altered guards,
    target control, all other methods and unknown/new tasks remain bound.
    """
    if task not in NONINTERACTIVE_TASKS:
        return tree
    guards = {ast.dump(ast.parse(f'getattr({owner}.environment, "interact", False)', mode='eval').body,
                       include_attributes=False) for owner in ('self', 'session')}

    class Project(ast.NodeTransformer):
        def visit_If(self, node):
            if not node.orelse and ast.dump(node.test, include_attributes=False) in guards:
                return None
            return self.generic_visit(node)
    return Project().visit(tree)


def _select_task_limit(tree: ast.Module, task: str | None) -> None:
    """Other tasks' limits cannot invalidate this scene or measured card.

    Every remaining constant/function stays bound, and the selected limit
    must exist exactly once. With no task selection retain the full map.
    """
    if task is None:
        return
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name)
                and t.id == 'NATIVE_ACTION_LIMITS' for t in node.targets):
            require(isinstance(node.value, ast.Dict), 'task limits must be a literal map')
            pairs = [(k, v) for k, v in zip(node.value.keys, node.value.values)
                     if isinstance(k, ast.Constant) and k.value == task]
            require(len(pairs) == 1, 'measured task limit missing or ambiguous')
            node.value.keys = [pairs[0][0]]
            node.value.values = [pairs[0][1]]


def _bridge_terminal_source(data: bytes, measured_task: str | None) -> bytes:
    """Project the exact non-cloth card's inactive, scene-only setup extension.

    A cloth scene keeps the full bridge dependency. Only the reviewed
    fold-clothes initialization branch is inactive for a card measured on a
    different task; native control, verdict, reset and all other bridge code
    remain compared. A fold-clothes card keeps the entire bridge dependency.
    """
    tree = ast.parse(data)
    if measured_task is not None and measured_task != 'fold_clothes':
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.FunctionDef) and node.name == '_prepare_cuda_garments')]
        expected = ast.parse('if self.task_name == "fold_clothes":\n    _prepare_cuda_garments(self.scene_manager, self.num_envs)').body[0]
        expected_dump = ast.dump(expected, include_attributes=False)
        for cls in tree.body:
            if isinstance(cls, ast.ClassDef) and cls.name == 'EvaluationLifecycle':
                for method in cls.body:
                    if isinstance(method, ast.FunctionDef) and method.name == '_post_setup_scene':
                        method.body = [node for node in method.body
                            if ast.dump(node, include_attributes=False) != expected_dump]
    return ast.dump(tree, include_attributes=False).encode()


def terminal_dependencies(commit: str, measured_task: str | None = None) -> dict[str, str]:
    """Bind terminal methods to declared owners, helpers and module globals.

    Resolve source without importing simulator code. A new/ambiguous base or
    missing method is an admission error, never an omitted dependency. Owner
    module setup (including imports, constants and standalone helpers) stays
    bound alongside the transitive self-method closure. Historical sessions
    declaring no mixins are resolved entirely from their original session.
    Source at a full commit id never changes, so those results are cached;
    every caller receives its own copy.
    """
    if re.fullmatch(r'[0-9a-f]{40}', str(commit)) is not None:
        return dict(_cached_terminal_dependencies(commit, measured_task))
    return _terminal_dependencies(commit, measured_task)


@functools.lru_cache(maxsize=64)
def _cached_terminal_dependencies(commit: str, measured_task: str | None) -> dict[str, str]:
    return _terminal_dependencies(commit, measured_task)


def _terminal_dependencies(commit: str, measured_task: str | None) -> dict[str, str]:
    result = {}
    for p in TERMINAL_FILES:
        data = source_bytes(commit, p)
        if p.endswith('/robodojo/bridge.py'):
            data = _bridge_terminal_source(data, measured_task)
        elif p == SPARSE:
            data = ast.dump(_project_noninteractive(ast.parse(data), measured_task), include_attributes=False).encode()
        result[p] = digest(data)
    modules: dict[str, ast.Module] = {}

    def module(path: str) -> ast.Module:
        if path not in modules:
            try:
                modules[path] = _project_noninteractive(ast.parse(source_bytes(commit, path)), measured_task)
            except (OSError, subprocess.CalledProcessError, SyntaxError) as exc:
                raise RoboDojoBenchmarkError(
                    f'cannot resolve terminal owner source {path} at {commit}'
                ) from exc
        return modules[path]

    def owner(path: str, name: str) -> ast.ClassDef:
        matches = [n for n in module(path).body if isinstance(n, ast.ClassDef) and n.name == name]
        require(len(matches) == 1, f'terminal owner {name} is missing or ambiguous in {path}')
        cls = matches[0]
        require(not cls.keywords and not cls.decorator_list,
                f'terminal owner {name} has unsupported dynamic class construction')
        return cls

    session = owner(SESSION, 'RobodojoSession')
    classes = [(SESSION, session)]
    for base in session.bases:
        require(isinstance(base, ast.Name) and base.id in TERMINAL_OWNERS,
                'terminal session has an undeclared method owner')
        path, imported_module = TERMINAL_OWNERS[base.id]
        bindings = [n for n in module(SESSION).body if isinstance(n, ast.ImportFrom)
                    and n.level == 1 and n.module == imported_module
                    and any(a.name == base.id and a.asname in (None, base.id) for a in n.names)]
        require(len(bindings) == 1, f'terminal owner {base.id} import binding is missing or ambiguous')
        cls = owner(path, base.id)
        require(not cls.bases, f'terminal owner {base.id} has undeclared inherited methods')
        classes.append((path, cls))
    require(len({cls.name for _, cls in classes}) == len(classes), 'duplicate terminal method owner')
    methods: dict[str, tuple[str, ast.FunctionDef]] = {}
    for path, cls in classes:
        for node in cls.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                require(node.name not in methods, f'ambiguous terminal method owner for {node.name}')
                methods[node.name] = (path, node)
            else:
                require(isinstance(node, (ast.Expr, ast.Pass)),
                        f'terminal owner {cls.name} has unsupported class state')
    pending = list(TERMINAL_METHODS)
    seen: set[str] = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        require(name in methods, f'terminal method {name} has no declared owner')
        seen.add(name)
        path, method = methods[name]
        result[f'{SESSION}:{name}'] = digest(ast.dump(method, include_attributes=False).encode())
        # A bound-method callback (not only a direct call) is part of the
        # closure; properties such as is_terminal also enter here.
        for node in ast.walk(method):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == 'self':
                if node.attr in methods:
                    pending.append(node.attr)
                elif isinstance(node.ctx, ast.Load):
                    # Session fields are allowed. Only an unresolved
                    # direct method call is unambiguously a missing helper.
                    require(not any(isinstance(call, ast.Call) and call.func is node
                                    for call in ast.walk(method)),
                            f'terminal helper {node.attr} has no declared owner')
    for path, tree in modules.items():
        setup = ast.Module(body=[n for n in tree.body if not isinstance(n, ast.ClassDef)
                                and not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
                                         and isinstance(n.value.value, str))], type_ignores=[])
        result[f'{path}:module_setup'] = digest(ast.dump(setup, include_attributes=False).encode())
    if any(isinstance(node, ast.ImportFrom) and node.level == 1 and node.module == 'interaction_control'
           for tree in modules.values() for node in ast.walk(tree)):
        result[INTERACTION] = _module_digest(source_bytes(commit, INTERACTION), measured_task)
    # The capture mixin introduces executable helpers outside the owner
    # modules. Old monolithic sources do not import or require this file.
    if any(cls.name == 'RobodojoExecutionCaptureMixin' for _, cls in classes):
        support = 'ora/environment/execution_capture.py'
        result[support] = _module_digest(source_bytes(commit, support), measured_task)
    # Task limits, ending classifiers and reset validation are imported
    # globals, and cannot be excluded merely because method ASTs match.
    benchmark = ast.parse(source_bytes(commit, BENCHMARK))
    benchmark.body = [n for n in benchmark.body if not (
        isinstance(n, ast.FunctionDef) and n.name == 'validate_runtime_extension')]
    _select_task_limit(benchmark, measured_task)
    result[BENCHMARK] = digest(ast.dump(benchmark, include_attributes=False).encode())
    sparse_tree = ast.parse(source_bytes(commit, SPARSE)) if SPARSE in TERMINAL_FILES else ast.Module(body=[], type_ignores=[])
    if any(isinstance(node, ast.ImportFrom) and node.level == 1 and node.module == 'ik'
           for node in ast.walk(sparse_tree)):
        for name in ('ik.py', 'reachability.py'):
            path = 'ora/environment/backends/robodojo/' + name
            result[path] = _module_digest(source_bytes(commit, path), measured_task)
    closure, dumps = _control_closure(commit, measured_task)
    projected = {SESSION, BENCHMARK, *TERMINAL_FILES, *(path for path, _ in TERMINAL_OWNERS.values())}
    result.update({path: digest(dumps[path]) for path in sorted(closure) if path not in projected})
    return result


def _module_digest(data: bytes, measured_task: str | None) -> str:
    """AST digest, so a comment or formatting edit leaves the card current."""
    return digest(_module_dump(data, measured_task))


def _module_dump(data: bytes, measured_task: str | None) -> bytes:
    tree = _project_noninteractive(ast.parse(data), measured_task)
    return ast.dump(tree, include_attributes=False).encode()


def control_closure(commit: str, measured_task: str | None = None) -> dict[str, tuple[str, ...]]:
    """The terminal/control import closure a card binds, read at ``commit``."""
    return _control_closure(commit, measured_task)[0]


def _control_closure(commit: str, measured_task: str | None) -> tuple[dict[str, tuple[str, ...]], dict[str, bytes]]:
    """Follow what the session and ``TERMINAL_FILES`` import, through them.

    The card witnessed control and the official terminal, so every module
    those roots reach through kernel or RoboDojo-backend paths is bound --
    the executor's recovery/wait/staged-contact siblings, the controllers and
    contracts they call, bootstrap and camera capture on the reset path.
    Session, owner, ``TERMINAL_FILES`` and benchmark modules are traversed but
    keep their projections instead of a whole-file digest. ``CARD_EXCLUSIONS``
    are reviewed leaves; guarded interactive imports stay projected out for
    reviewed noninteractive tasks. A producer that imported fewer modules
    differs by key.
    """
    from .import_closure import ImportClosureError, import_closure
    from .robodojo_kernel import ROBODOJO_BACKEND_PREFIX

    kernel = set(legacy.WORKER_KERNEL_PATHS)
    dumps: dict[str, bytes] = {}

    def read(path: str) -> bytes | None:
        try:
            data = source_bytes(commit, path)
        except subprocess.CalledProcessError:
            return None
        tree = _project_noninteractive(ast.parse(data), measured_task)
        dumps[path] = ast.dump(tree, include_attributes=False).encode()
        return ast.unparse(tree).encode()

    def follow(importer: str, imported: str) -> bool:
        return ((imported in kernel or imported.startswith(ROBODOJO_BACKEND_PREFIX))
                and imported not in CARD_EXCLUSIONS and not imported.endswith('/__init__.py'))

    try:
        closure = import_closure((SESSION, *TERMINAL_FILES), read=read, follow=follow)
    except (ImportClosureError, SyntaxError) as exc:
        raise RoboDojoBenchmarkError(
            f'cannot resolve the RoboDojo terminal/control closure at {commit}: {exc}'
        ) from exc
    return closure, dumps


def read_reference(ref: dict, root: Path) -> tuple[dict, Path]:
    relative = Path(ref['path'])
    require(not relative.is_absolute() and '..' not in relative.parts
            and relative.parts[:1] == ('outputs',), 'revalidation artifact must be in shared outputs')
    path = root / relative
    # The reference stays a portable `outputs/...` locator, and this root is the
    # authority whenever it resolves normally. Where the home filesystem ran out
    # of capacity the locator is a symlink onto another volume, so that case
    # falls back to the managed output roots rather than being refused; the
    # lexical checks above still bar absolute paths and parent traversal.
    if path.resolve().is_relative_to((root / 'outputs').resolve()):
        cursor = path
        while cursor != root:
            require(not cursor.is_symlink(), 'revalidation artifact uses a symlink')
            require(cursor.parent != cursor, 'revalidation artifact root mismatch')
            cursor = cursor.parent
    else:
        from ora.orchestrator.paths import managed_evidence_path

        managed_evidence_path(path, label='revalidation artifact')
    data = path.read_bytes()
    require(digest(data)==ref['sha256'], 'revalidation artifact hash mismatch')
    value = json.loads(data)
    require(isinstance(value,dict),'revalidation artifact must be a record')
    return value,path


def reference(path: Path, root: Path) -> dict[str,str]:
    from ora.orchestrator.paths import portable_output_locator

    return {'path':portable_output_locator(path, root), 'sha256':digest(path.read_bytes())}


def scene_language(runtime: dict, smoke_runtime: dict, readback: str) -> None:
    """Hold a plan's instruction to the instruction its scene smoke read back.

    A smoke that bound ``expected_task_language`` admits exactly that
    instruction, and its reset must have read it back. A smoke that bound
    none (a single layout whose instruction the layout generates) read the official instruction back; a plan admitted on it must
    carry exactly that readback, and revalidating the unbound lane itself
    certifies the readback (``task_instruction`` on the certificate).
    """
    bound = smoke_runtime.get('expected_task_language')
    language = runtime.get('expected_task_language')
    require(bound is None or language==bound, 'scene expected_task_language drift')
    require(readback==(readback if language is None else language), 'scene evaluator/language mismatch')


def validate_scene(runtime: dict, report: dict, plan: dict, root: Path, card: dict) -> dict:
    require(report.get('schema')=='ora.robodojo-b2-smoke.v1' and report.get('passed') is True,
            'scene smoke did not pass')
    cpu_single_round = (runtime.get('physics_device','cuda:0')!='cuda:0' and runtime.get('layout_schedule') is None)
    checks = report.get('checks',{})
    required = [k for k in CHECKS if not (cpu_single_round and k=='round_reset')]
    require(all(checks.get(k) is True for k in required),'scene smoke checks incomplete')
    if cpu_single_round:
        # A CPU session is single-round (rule robodojo_cpu_in_place_restart):
        # its smoke does not restart in place, and an older one that did may stay.
        require(checks.get('round_reset') in (True, CPU_SINGLE_ROUND_RESET),'scene smoke checks incomplete')
    original_plan = dict(plan); recorded_hash = original_plan.pop('plan_sha256')
    from .spec import json_sha256
    require(json_sha256(original_plan)==recorded_hash==report['plan_sha256'],'scene plan hash drift')
    require(plan['harness_commit']==report['source_commit'],'scene producer source drift')
    require(configuration_binding(runtime)==configuration_binding(plan['runtime']), 'scene configuration drift')
    # The smoke's physical evidence depends on the scheduled files, not on the
    # catalog revision they were selected from (a new pool for another task).
    require(_scheduled_layouts(runtime)==_scheduled_layouts(plan['runtime']),'scene layout_schedule drift')
    reset=report['reset']; restart=report['restart']
    require(restart is not None or cpu_single_round,'scene smoke round reset missing')
    require(report['task']==runtime['task_variant'] and report['gpu']==runtime['cuda_visible_devices'], 'scene task/device drift')
    require(reset['reward_registered'] is True, 'scene evaluator/language mismatch')
    scene_language(runtime, plan['runtime'], reset['task_instruction'])
    registered=validate_scene_registration(reset['scene_registration'],native_action_limit=runtime['native_action_limit'])
    require(restart is None or restart['round_isolation']['passed'] is True,'round isolation failed')
    source_path=root/'environment/source-manifest.json'; source=json.loads(source_path.read_text())
    from .robodojo_physics_attestation import require_physics_attestation
    physics=require_physics_attestation(runtime,source)
    scheduled=None
    if runtime.get('layout_schedule') is None:
        require(restart is None or validate_scene_registration(restart['scene_registration'],native_action_limit=runtime['native_action_limit'])==registered,
                'repeated reset changed scene identities')
    else:
        # A scheduled round is a new layout in a relaunched simulation, so its
        # identities differ by design; its own checks replace the comparison.
        from .robodojo_layout_revalidation import validate_scheduled_scene
        sync_path=root/'environment/render-synchronization.jsonl'
        synchronizations=[json.loads(line) for line in sync_path.read_text().splitlines() if line.strip()]
        scheduled=validate_scheduled_scene(runtime,report,source,synchronizations)
    require(digest(source_path.read_bytes())==reset['source_manifest']['sha256'],'scene source manifest hash drift')
    require(source['source_revisions']==source_revisions(runtime),'scene external source drift')
    require(source['layout_file']['sha256']==runtime['layout_sha256'],'scene layout content drift')
    fingerprint={'loaded_files':source['loaded_files'],'installed_packages':source['installed_packages'],
                 'resolved_environment_config_sha256':digest(json.dumps(source['resolved_environment_config'],sort_keys=True,allow_nan=False).encode())}
    require(fingerprint==reset['actual_source_fingerprint'],'scene actual fingerprint drift')
    legacy._validate_source_fingerprint(fingerprint,label='scene smoke')
    require(legacy._joined('actual_source_fingerprint',fingerprint)==legacy._joined('actual_source_fingerprint',card['actual_source_fingerprint']),
            'terminal evidence installation/assets differ from scene smoke')
    legacy._validate_compiled_model(reset['runtime_embodiment'],label='scene smoke')
    require(reset['runtime_embodiment']==card['runtime_embodiment'],'compiled articulation/camera geometry drift')
    device={k:v for k,v in reset['device_binding'].items() if k!='startup_log_prefix'}
    legacy._validate_device_binding(device,runtime=runtime)
    require(report['worker_residency']['gpu_uuids']==[device['selected_gpu_uuid']],'scene worker residency mismatch')
    process=json.loads((root/'environment/environment-session.json').read_text())
    require(process['state']=='closed' and process['worker_returncode'] is not None and report['close_elapsed_s']<=60,
            'scene worker bounded close incomplete')
    environment={'python_executable':process['python_executable'],'python_binary_sha256':process['python_sha256'],
                 'versions':{k:source['installed_packages'][k] for k in card['environment']['versions']}}
    require(environment==card['environment'],'terminal interpreter differs from scene smoke')
    legacy._validate_environment_readback(environment,label='scene smoke')
    observations=sorted((root/'observations').glob('*/observation.json'))
    require(len(observations)>=3,'scene observations missing')
    first=json.loads(observations[0].read_text())
    require(first.get('robot_geometry_verification',{}).get('passed') is True,'loaded geometry evidence missing')
    geometry_reuse = revalidate_geometry_binding(runtime['public_geometry_descriptor'],
        plan['runtime']['public_geometry_descriptor'], source_commit=report['source_commit'],
        verification=first['robot_geometry_verification'])
    require(set(report['sensors'])=={'head','left_wrist','right_wrist'},'scene RGB-D camera coverage incomplete')
    for sensor in report['sensors'].values():
        require(sensor['shape']==[480,640] and sensor['finite_positive_fraction']>0.5,'scene RGB-D is invalid')
    require(len(report['actions'])==2,'scene composite coverage incomplete')
    for arm,action in zip(('left','right'),report['actions']):
        phases=action['phase_receipts']
        require([p['status'] for p in phases]==['reached','executed'],'scene target-state phases failed')
        for phase in phases:
            rawref=phase['receipt']; rawpath=root/'environment'/rawref['path']
            require(rawpath.resolve().is_relative_to((root/'environment').resolve()),'scene receipt path escapes')
            require(digest(rawpath.read_bytes())==rawref['sha256'],'scene receipt hash mismatch')
            raw=json.loads(rawpath.read_text())
            require(raw['action_id']==phase['action']['action_id'] and raw['arguments']['arm']==arm,'scene receipt command/arm mismatch')
            require(raw['command_invariant']['verified'] is True,'scene held-command invariant failed')
        require(report['alignment'][arm]['candidates'][0]['reachable'] is True,'scene current-pose IK failed')
    return {'passed':True,'environment':environment,'device_binding':device,
            'geometry_reuse':geometry_reuse,
            'graphics_compatibility':source['graphics_compatibility'],
            'source_revisions':source['source_revisions'],'actual_source_fingerprint':fingerprint,
            'configuration':configuration_binding(runtime),'layout_sha256':runtime['layout_sha256'],
            'task_instruction':reset['task_instruction'],**({} if scheduled is None else {'scheduled_layouts':scheduled}),
            **({} if runtime.get('physics_device','cuda:0')=='cuda:0' else {'physics_mode':physics})}


def evaluate(runtime: dict, *, workspace_root: Path, harness_commit: str, card_reference: dict, smoke_reference: dict) -> dict:
    raw_card,_=legacy._read_record(card_reference,workspace_root=workspace_root,label='original terminal card')
    from ora.environment.benchmarks.robodojo import REVISION_FIELDS
    from ora.environment.benchmarks.robodojo_asset_revision import unchanged_card_assets
    require(unchanged_card_assets(raw_card['source_revisions'], source_revisions(runtime),
                                 measured_task=raw_card['measured_task']['task']),
            'terminal card external dependencies changed; targeted qualification required')
    # Validate the historical record against its ORIGINAL device/config/source.
    old_runtime={**runtime,**raw_card['card_configuration'],
                 **{REVISION_FIELDS[k]: v for k, v in raw_card['source_revisions'].items()}}
    card=legacy.load_card_qualification(old_runtime,workspace_root=workspace_root,
        harness_commit=raw_card['harness_commit'],reference=card_reference)
    require(terminal_dependencies(card['harness_commit'], card['measured_task']['task'])==terminal_dependencies(harness_commit, card['measured_task']['task']),
            'official terminal/control dependency changed; targeted requalification required')
    old_config=dict(card['card_configuration']); current_config=card_configuration_binding(runtime)
    for field in ('cuda_visible_devices','renderer_gpu'):
        old_config.pop(field); current_config.pop(field)
    require(old_config==current_config,'terminal timing/controller configuration drift')
    report,path=read_reference(smoke_reference,workspace_root)
    producer = dependency_map(report['source_commit'], runtime['task_variant'])
    current = dependency_map(harness_commit, runtime['task_variant'])
    changed = sorted(k for k in set(producer) | set(current) if producer.get(k) != current.get(k))
    require(not changed, 'scene smoke execution dependencies require review: ' + ', '.join(changed))
    plan=json.loads((path.parent/'plan.json').read_text())
    view=validate_scene(runtime,report,plan,path.parent,card)
    return {**view,'schema':SCHEMA,'harness_commit':harness_commit,
            'worker_kernel_sha256':legacy.worker_kernel_sha256(harness_commit),
            'original_card':card_reference,'original_card_source':card['harness_commit'],
            'original_card_device':card['device_binding'],'card_measured_task':card['measured_task'],
            'scene_smoke':smoke_reference,'scene_source':report['source_commit'],
            'raw_artifacts':[reference(p,workspace_root) for p in sorted(path.parent.rglob('*.json'))],
            'producer_dependencies':dependency_map(report['source_commit'], runtime['task_variant']),
            'terminal_dependencies':terminal_dependencies(card['harness_commit'], card['measured_task']['task']),
            'coverage':scene_coverage(runtime)}


def scene_coverage(runtime: dict) -> dict:
    """What a revalidation certificate establishes fresh, reuses, and does not establish.

    A CUDA certificate's coverage is unchanged by the CPU-physics lane; a CPU
    certificate adds the in-place round_reset boundary.
    """
    cpu_single_round = runtime.get('physics_device','cuda:0')!='cuda:0' and runtime.get('layout_schedule') is None
    coverage={'fresh':[k for k in CHECKS if not (cpu_single_round and k=='round_reset')]+['bounded_close']+(
                  [] if runtime.get('layout_schedule') is None else list(_schedule_checks())),
              'reused':['official_terminal','unchanged native motion/control implementation'],
              'not_established':['task success','new full motion sweep','fresh per-task native timeout',
                                 'fresh nonzero pose/gripper displacement']}
    if runtime.get('physics_device','cuda:0')!='cuda:0':
        coverage['not_established']=[*coverage['not_established'],CPU_ROUND_RESET_BOUNDARY]
    return coverage


def _scheduled_layouts(runtime: dict) -> dict | None:
    schedule = runtime.get('layout_schedule')
    return None if schedule is None else {key: value for key, value in schedule.items() if key != 'selection'}


def _schedule_checks() -> tuple[str, ...]:
    from .robodojo_layout_revalidation import SCHEDULE_CHECKS
    return SCHEDULE_CHECKS


def load(runtime: dict, *, workspace_root: Path, harness_commit: str) -> dict:
    record,_=read_reference(runtime['infrastructure_qualification'],workspace_root)
    require(record.get('schema')==SCHEMA,'unknown revalidation schema')
    view=evaluate(runtime,workspace_root=workspace_root,harness_commit=harness_commit,
                  card_reference=record['original_card'],smoke_reference=record['scene_smoke'])
    # This certificate is a checker product. Re-evaluate its raw inputs at every
    # admission; a later unrelated source commit never rewrites the certificate.
    for key in ('producer_dependencies','terminal_dependencies','coverage','environment','device_binding','raw_artifacts'):
        require(record.get(key)==view[key],f'revalidation {key} drift')
    return view
