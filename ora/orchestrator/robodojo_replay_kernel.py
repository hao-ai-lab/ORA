"""Whether a control replay on this checkout reproduces a run's RoboDojo physics.

A control replay feeds recorded native controls through the session; it never
runs the executor. So a worker kernel that differs from the run's can still
replay the run when every difference is off the replay path. Every file of
either commit's worker-kernel manifest is compared:

* A file in ``CONTROL_REPLAY_EXECUTOR_ONLY`` is an executor or probe module the
  replay imports without running, each with a reviewed reason. Only the
  top-level names the replay reaches must bind identically (every binding of a
  name, AST without line numbers), and its module-level statements that bind
  no name must be unchanged. A reached name bound at neither commit refuses.
* Any other differing file may only add: every top-level name the run's
  version binds is bound identically, ``__all__`` may only grow, new names
  may appear, and no module-level statement that binds no name may change.

``identical_kernel`` and ``executor_drift_admitted`` admit the replay; anything
else refuses the launch. The module docstring is ignored; comments are not in
the AST.
"""

from __future__ import annotations

import ast
import subprocess
from collections import defaultdict
from typing import Any, Mapping

from .paths import PROJECT_ROOT

_BACKEND = "ora/environment/backends/robodojo"
_SESSION_ONLY = "referenced only by session construction; the executor code itself never runs in a control replay"

#: Worker-kernel modules a control replay imports but does not run: path ->
#: (reached top-level names, reason). ``session``, ``bridge``,
#: ``round_lifecycle``, ``feedback`` and ``observation`` reference nothing else
#: in them on the replay path (``tests/test_robodojo_motion_replay.py`` scans).
CONTROL_REPLAY_EXECUTOR_ONLY: Mapping[str, tuple[tuple[str, ...], str]] = {
    f"{_BACKEND}/sparse_motion.py": (
        ("FREE_SPACE_CLEARANCE_M", "LOCAL_JOINT_JUMP_LIMIT_RAD", "GLOBAL_JOINT_JUMP_LIMIT_RAD",
         "STAGED_CONTACT_STRATEGY", "gripper_closed_floor"),
        "sparse executor; the replay reaches the constants session.__init__ validates against and "
        "gripper_closed_floor through feedback.gripper_state"),
    f"{_BACKEND}/recovery.py": (("RETRACT_THEN_UNWIND", "configured_route"),
                                "recovery executor; the replay reaches configured_route in session.__init__"),
    f"{_BACKEND}/motion_stall.py": ((), "stall monitor of the sparse executor; " + _SESSION_ONLY),
    f"{_BACKEND}/staged_contact.py": ((), "staged contact route of the sparse executor; " + _SESSION_ONLY),
    f"{_BACKEND}/wait.py": ((), "wait executor; " + _SESSION_ONLY),
    f"{_BACKEND}/ik.py": ((), "SDK IK selection for execute and probes; session, bridge, round_lifecycle, feedback "
                              "and observation do not reference it"),
    "ora/contracts/planning_diagnostics.py": (
        (), "planner-diagnostics schema for receipts and probes; no replay-path module references it"),
}
#: Names the replay-path modules reference in executor-only modules from code a
#: control replay never calls: (module, name) -> the calling path.
CONTROL_REPLAY_UNREACHED: Mapping[tuple[str, str], str] = {
    ("sparse_motion", "execute_sparse_motion_v3"): "session._execute_without_capture (policy execute path)",
    ("sparse_motion", "solve_ik"): "session._execute_without_capture (runtime action v3 move_pose)",
    ("sparse_motion", "probe_motion"): "session.probe_motion (read-only motion probe)",
}
ADMITTED_KERNELS = ("identical_kernel", "executor_drift_admitted")


def _source_at(commit: str, path: str) -> bytes | None:
    try:
        result = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=PROJECT_ROOT, capture_output=True,
                                check=False)
    except OSError:
        return None
    return result.stdout if result.returncode == 0 else None


def _bound_names(node: ast.stmt) -> list[tuple[str, str]] | None:
    """(name, binding dump) for a top-level statement that binds names; ``None`` for any other."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [(node.name, ast.dump(node))]
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        module = f"{getattr(node, 'level', 0)}:{getattr(node, 'module', None)}"
        return [((alias.asname or alias.name).split(".")[0], f"import {module}:{alias.name}") for alias in node.names]
    if isinstance(node, ast.Assign) and all(isinstance(target, ast.Name) for target in node.targets):
        return [(target.id, ast.dump(node)) for target in node.targets]
    if isinstance(node, (ast.AnnAssign, ast.AugAssign)) and isinstance(node.target, ast.Name):
        return [(node.target.id, ast.dump(node))]
    return None


def module_shape(source: bytes | None) -> tuple[dict[str, tuple[str, ...]], tuple[str, ...]]:
    """Every binding of every top-level name, and the other module-level statements (no docstring)."""
    bindings: defaultdict[str, list[str]] = defaultdict(list)
    others: list[str] = []
    for index, node in enumerate(ast.parse(source or b"").body):
        if index == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            continue
        names = _bound_names(node)
        if names is None:
            others.append(ast.dump(node))
        for name, dump in names or ():
            bindings[name].append(dump)
    return {name: tuple(dumps) for name, dumps in bindings.items()}, tuple(others)


def _all_names(source: bytes | None) -> set[str] | None:
    """The literal ``__all__`` of a module, or ``None`` when it is absent or not a literal."""
    for node in ast.parse(source or b"").body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            try:
                return {str(value) for value in ast.literal_eval(node.value)}
            except ValueError:
                return None
    return None


def add_only_changes(run_source: bytes | None, source: bytes | None) -> list[str]:
    """Names the run bound that now bind differently, and changed non-binding statements."""
    before, before_other = module_shape(run_source)
    after, after_other = module_shape(source)
    changed = [name for name, dumps in sorted(before.items()) if after.get(name) != dumps and name != "__all__"]
    exported, now = _all_names(run_source), _all_names(source)
    if "__all__" in before and (exported is None or now is None or not exported <= now):
        changed.append("__all__")
    return changed + (["<module-level statements>"] if before_other != after_other else [])


def executor_only_changes(path: str, run_source: bytes | None, source: bytes | None) -> list[str]:
    """Reached names that bind differently (or nowhere), and changed non-binding statements."""
    reached = CONTROL_REPLAY_EXECUTOR_ONLY[path][0]
    before, before_other = module_shape(run_source)
    after, after_other = module_shape(source)
    changed = [name if name in before or name in after else f"{name} (bound at neither commit)"
               for name in reached if before.get(name) != after.get(name) or name not in before]
    return changed + (["<module-level statements>"] if before_other != after_other else [])


def kernel_admission(run_commit: str | None, commit: str | None) -> dict[str, Any]:
    """Whether this checkout replays the run's physics: identical kernel, or reviewed executor drift."""
    from ora.environment.benchmarks.robodojo import RoboDojoBenchmarkError

    from .robodojo_kernel import worker_kernel_paths

    try:
        paths = sorted(set(worker_kernel_paths(run_commit, project_root=PROJECT_ROOT))
                       | set(worker_kernel_paths(commit, project_root=PROJECT_ROOT))) if run_commit and commit else None
    except RoboDojoBenchmarkError:
        paths = None
    if paths is None:
        return {"status": "unresolvable", "run_commit": run_commit, "commit": commit}
    sources = {path: (_source_at(run_commit, path), _source_at(commit, path)) for path in paths}
    differing = [path for path, (before, after) in sources.items() if before != after]
    refused = {}
    for path in differing:
        rule = executor_only_changes if path in CONTROL_REPLAY_EXECUTOR_ONLY else None
        changes = rule(path, *sources[path]) if rule else add_only_changes(*sources[path])
        if changes:
            refused[path] = changes
    status = ("identical_kernel" if not differing else
              "executor_drift_admitted" if not refused else "control_path_changed")
    return {"status": status, "run_commit": run_commit, "commit": commit, "differing": differing,
            "executor_only": [path for path in differing if path in CONTROL_REPLAY_EXECUTOR_ONLY],
            "add_only": [path for path in differing if path not in CONTROL_REPLAY_EXECUTOR_ONLY],
            "refused_changes": refused}


def admission_note(admission: Mapping[str, Any] | None) -> str | None:
    """The report's top-level notice when the replay kernel is not the run's own."""
    if not admission or admission.get("status") != "executor_drift_admitted":
        return None
    return ("The RoboDojo worker kernel differs from the run's; the replay was admitted because every "
            f"difference is AST-equivalent on the replay path ({len(admission.get('differing', []))} files: "
            "reviewed executor-only modules and add-only changes).")


__all__ = [
    "ADMITTED_KERNELS", "CONTROL_REPLAY_EXECUTOR_ONLY", "CONTROL_REPLAY_UNREACHED", "add_only_changes",
    "admission_note", "executor_only_changes", "kernel_admission", "module_shape",
]
