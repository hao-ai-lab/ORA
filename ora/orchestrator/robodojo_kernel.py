"""Source dependency closure and content-addressed RoboDojo worker identity."""
from __future__ import annotations
import hashlib
import ast
import re
import subprocess
from pathlib import Path
from typing import Sequence
from ora.environment.benchmarks.robodojo import RoboDojoBenchmarkError

#: The tracked Harness sources that decide how the isolated RoboDojo worker
#: behaves. An infrastructure qualification is bound to the *content* of this
#: set at the RunPlan's own commit, not to the commit the record happened to
#: be measured on, so an unrelated commit cannot forge a failure of a 36-minute
#: live measurement while any edit inside the worker must.
#:
#: ``ROBODOJO_MOTION_KERNEL_PATHS`` -- the source paths the scripted
#: sparse-motion gate binds -- is a *subset* of this set, not a prefix of it.
#: This set is sorted and that one is not, so the two orderings do not line
#: up: membership must be checked, never read off a position.
#: ``tests/test_robodojo_kernel_paths.py`` asserts the containment and the
#: non-prefix relation, because an earlier version of this comment claimed a
#: prefix and a maintainer placing a new file by that claim is how
#: ``contracts/pose.py`` -- on the worker's executed servo path -- ended up in
#: neither set until the import-closure check below listed it here. When
#: adding a file, decide its membership in each set on what
#: that set means, and add it to both if both meanings apply.
#:
#: The motion subset covers the session wrapper and its bridge, camera
#: capture, geometry helpers, motion adapter, sparse-motion executor, the
#: scripted motion module whose X5 mount/reach constants the infrastructure
#: probe's unreachable-pose target is derived from, and the ARX X5 controller
#: contract. The rest of this set is what the infrastructure probes exercise
#: beyond motion:
#:
#: * ``environment/worker.py`` is the isolated process that constructs the
#:   session from the RPC config and dispatches every method;
#: * ``environment/rpc.py`` is that process's transport and lifecycle -- the
#:   ``bounded_close`` check is a statement about this module;
#: * ``environment/factory.py`` composes the worker config and the
#:   ``CUDA_VISIBLE_DEVICES``/``renderer_gpu`` mask the recorded
#:   ``device_binding`` asserts;
#: * ``environment/base.py`` validates every status the session publishes,
#:   which is what ``official_terminal`` is read from;
#: * ``environment/benchmarks/robodojo.py`` owns the layout selection, the
#:   public context and the native-action/physics-step contract the worker
#:   enforces;
#: * ``environment/controllers/motion_transaction.py`` writes the plan and
#:   receipt artifacts the probes read back;
#: * ``contracts/action.py`` and ``contracts/motion.py`` are the action
#:   envelope and servo compilation the worker validates against;
#: * ``backends/robodojo/graphics_compat.py`` is the process-local Vulkan
#:   allocation-limit repair whose readback the record stores and preflight
#:   re-compares.
#:
#: Membership is checked, not only reviewed: every module in the worker's static
#: import closure (``worker_import_closure``, from
#: ``ROBODOJO_WORKER_ENTRY_POINTS``) must be listed here or excluded, with its
#: reason, in ``KERNEL_REVIEWED_EXCLUSIONS``; every import a shared dispatcher
#: makes for another backend must be named in ``DISPATCH_PRUNED_EDGES``.
#: ``tests/test_robodojo_kernel_paths.py`` fails with the unreviewed paths or
#: edges, and with any table entry that no longer matches the source. The
#: executor modules ``recovery.py`` and ``wait.py`` were imported but unlisted,
#: so their edits left every record current; the check added them with
#: ``robot_asset_geometry.py``, the ``alignment``, ``motion_progress``,
#: ``plan_validation``, ``reach_check`` and ``reachability`` controllers, and
#: the contracts ``robodojo_arx`` (ARX_X5_CONTRACT moved there from the listed
#: controller in 7c7e6a8a), ``pose`` (executed through
#: ``motion.compile_servo_delta``), ``wait``, ``reach_check``,
#: ``robot_geometry`` and ``skill_sequence`` (it decides which skill-sequence
#: children the session admits, in what order). ``composed_tool`` joined for the
#: same reason when composed tools reached the live path, with
#: ``sequence_admission``, the session's child admission moved out of
#: ``session.py`` (and so also in the motion subset the session belonged to).
#:
#: Deliberately outside the set: the entries of ``KERNEL_REVIEWED_EXCLUSIONS``;
#: ``agent/composition/embodiment.py`` and the tracked ARX X5 card, which
#: ``_validate_embodiment_activation`` recompiles and rehashes live on every
#: load; and the ``robodojo_card_qualification`` probe driver, which does not
#: run at launch -- a new or renamed entry in ``QUALIFICATION_CHECKS``
#: invalidates every older record through the checks comparison instead.
WORKER_KERNEL_PATHS = (
    "ora/contracts/acceptance.py",
    "ora/contracts/action.py",
    "ora/contracts/composed_tool.py",
    "ora/contracts/gripper_feedback.py",
    "ora/contracts/motion.py",
    "ora/contracts/planning_diagnostics.py",
    "ora/contracts/pose.py",
    "ora/contracts/reach_check.py",
    "ora/contracts/robodojo_arx.py",
    "ora/contracts/robot_geometry.py",
    "ora/contracts/skill_sequence.py",
    "ora/contracts/wait.py",
    "ora/environment/backends/robodojo/__init__.py",
    "ora/environment/backends/robodojo/bootstrap.py",
    "ora/environment/backends/robodojo/bridge.py",
    "ora/environment/backends/robodojo/camera_capture.py",
    "ora/environment/backends/robodojo/common.py",
    "ora/environment/backends/robodojo/execution_capture.py",
    "ora/environment/backends/robodojo/feedback.py",
    "ora/environment/backends/robodojo/graphics_compat.py",
    "ora/environment/backends/robodojo/ik.py",
    "ora/environment/backends/robodojo/inter_arm.py",
    "ora/environment/backends/robodojo/interaction_control.py",
    "ora/environment/backends/robodojo/layout_rounds.py",
    "ora/environment/backends/robodojo/motion_adapter.py",
    "ora/environment/backends/robodojo/motion_checks.py",
    "ora/environment/backends/robodojo/motion_qualification.py",
    "ora/environment/backends/robodojo/motion_stall.py",
    "ora/environment/backends/robodojo/observation.py",
    "ora/environment/backends/robodojo/reach_check.py",
    "ora/environment/backends/robodojo/reachability.py",
    "ora/environment/backends/robodojo/recovery.py",
    "ora/environment/backends/robodojo/robot_asset_geometry.py",
    "ora/environment/backends/robodojo/round_lifecycle.py",
    "ora/environment/backends/robodojo/sequence_admission.py",
    "ora/environment/backends/robodojo/session.py",
    "ora/environment/backends/robodojo/sparse_motion.py",
    "ora/environment/backends/robodojo/staged_contact.py",
    "ora/environment/backends/robodojo/start_gate.py",
    "ora/environment/backends/robodojo/wait.py",
    "ora/environment/base.py",
    "ora/environment/benchmarks/robodojo.py",
    "ora/environment/benchmarks/robodojo_layouts.py",
    "ora/environment/controllers/alignment.py",
    "ora/environment/controllers/convergence_request.py",
    "ora/environment/controllers/gripper_stall.py",
    "ora/environment/controllers/joint_path_clearance.py",
    "ora/environment/controllers/measured_convergence.py",
    "ora/environment/controllers/motion_progress.py",
    "ora/environment/controllers/motion_transaction.py",
    "ora/environment/controllers/plan_validation.py",
    "ora/environment/controllers/reach_check.py",
    "ora/environment/controllers/reachability.py",
    "ora/environment/controllers/robodojo_arx.py",
    "ora/environment/execution_capture.py",
    "ora/environment/factory.py",
    "ora/environment/rpc.py",
    "ora/environment/worker.py",
)


ROBODOJO_MOTION_KERNEL_PATHS = (
    "ora/contracts/acceptance.py",
    "ora/contracts/gripper_feedback.py",
    "ora/environment/backends/robodojo/feedback.py",
    "ora/contracts/planning_diagnostics.py",
    "ora/environment/backends/robodojo/reachability.py",
    "ora/environment/controllers/motion_transaction.py",
    # The request admission and stage loop of execute_measured_convergence,
    # which motion_transaction.py defines; they are bound with that module.
    "ora/environment/controllers/convergence_request.py",
    "ora/environment/controllers/measured_convergence.py",
    "ora/environment/backends/robodojo/ik.py",
    # ik.py screens every admitted IK candidate's straight joint path with this
    # module, so it decides which endpoint the motion gate executes.
    "ora/environment/controllers/joint_path_clearance.py",
    # The lockstep inter-arm admission of segments that drive both arms
    # (dual-arm moves and two-arm recoveries) refuses before any write.
    "ora/environment/backends/robodojo/inter_arm.py",
    # The gate's own driver reads NATIVE_ACTION_LIMITS to size the episode and
    # to judge the official preterminal boundary case, so the benchmark
    # contract is part of what it measures.
    "ora/environment/benchmarks/robodojo.py",
    # The per-round layout schedule the session and bridge bind round 0 and
    # every scheduled restart to (g008), and its worker-side relaunch checks.
    "ora/environment/benchmarks/robodojo_layouts.py",
    "ora/environment/backends/robodojo/bridge.py",
    "ora/environment/backends/robodojo/round_lifecycle.py",
    "ora/environment/backends/robodojo/layout_rounds.py",
    "ora/environment/backends/robodojo/observation.py",
    "ora/environment/backends/robodojo/motion_checks.py",
    "ora/environment/backends/robodojo/execution_capture.py",
    "ora/environment/backends/robodojo/bootstrap.py",
    # The node start gate bootstrap and the readback hold (RDJ-10).
    "ora/environment/backends/robodojo/start_gate.py",
    "ora/environment/backends/robodojo/camera_capture.py",
    "ora/environment/backends/robodojo/common.py",
    "ora/environment/backends/robodojo/interaction_control.py",
    "ora/environment/backends/robodojo/motion_adapter.py",
    "ora/environment/backends/robodojo/motion_qualification.py",
    "ora/environment/backends/robodojo/session.py",
    "ora/environment/backends/robodojo/sequence_admission.py",
    "ora/environment/backends/robodojo/sparse_motion.py",
    # The gate's recover_home_after_matrix and recover_backtrack_after_move
    # cases run recover_motion, which the executor dispatches into this module.
    "ora/environment/backends/robodojo/recovery.py",
    # The executor's staged contact route (opt-in per runtime).
    "ora/environment/backends/robodojo/staged_contact.py",
    # Every sparse segment is driven by this module and ends on its joint stall
    # rule, which feeds the shared per-sample monitor.
    "ora/environment/backends/robodojo/motion_stall.py",
    "ora/environment/controllers/motion_progress.py",
    "ora/environment/controllers/gripper_stall.py",
    "ora/environment/controllers/robodojo_arx.py",
    # The ARX X5 controller contract's values; the controller module above
    # only re-exports them since 7c7e6a8a.
    "ora/contracts/robodojo_arx.py",
    "ora/environment/execution_capture.py",
)

#: The RoboDojo backend package; ``robodojo_kernel_review`` holds the manifest
#: to the worker's import closure starting here.
ROBODOJO_BACKEND_PREFIX = "ora/environment/backends/robodojo/"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def worker_kernel_paths(commit: str, *, project_root: Path) -> tuple[str, ...]:
    """Read the producer's literal manifest, without executing historical code.

    The manifest moved from qualification.py to this module. A historical
    digest must use its own ordered paths; missing files are never skipped.
    """
    for owner in ("ora/orchestrator/robodojo_kernel.py",
                  "ora/orchestrator/robodojo_qualification.py"):
        result = subprocess.run(["git", "show", f"{commit}:{owner}"], cwd=project_root,
                                capture_output=True, check=False)
        if result.returncode:
            continue
        tree = ast.parse(result.stdout)
        declarations = [node.value for node in tree.body if isinstance(node, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "WORKER_KERNEL_PATHS"
                                for t in node.targets)]
        if not declarations:
            continue
        try:
            paths = ast.literal_eval(declarations[0]) if len(declarations) == 1 else None
        except (ValueError, TypeError):
            paths = None
        if (not isinstance(paths, tuple) or not paths
                or any(not isinstance(p, str) or not p.startswith("ora/")
                       or ".." in Path(p).parts or not p.endswith(".py") for p in paths)
                or len(set(paths)) != len(paths)):
            raise RoboDojoBenchmarkError("invalid source-bound RoboDojo kernel manifest")
        return paths
    raise RoboDojoBenchmarkError(f"cannot resolve source-bound RoboDojo kernel manifest at {commit}")


def worker_kernel_sha256(commit: str, *, project_root: Path | None = None) -> str:
    """Hash the RoboDojo worker kernel as it stood at one commit.

    ``git show <commit>:<path>`` resolves the tracked content at any commit in
    history, so the same digest is computable at measurement time, at launch
    and at seal. Length-prefixing both the path and its bytes keeps the
    concatenation unambiguous.
    """

    _require_commit(commit)
    root = _PROJECT_ROOT if project_root is None else project_root
    return _paths_sha256(commit, worker_kernel_paths(commit, project_root=root), root)


def _require_commit(commit: str) -> None:
    if re.fullmatch(r"[0-9a-f]{7,40}", str(commit)) is None:
        raise RoboDojoBenchmarkError(
            "a RoboDojo worker kernel hash needs an exact commit id"
        )


def _source(commit: str, relative: str, root: Path) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative}"], cwd=root,
        capture_output=True, check=False,
    )
    if result.returncode:
        raise RoboDojoBenchmarkError(
            f"cannot resolve the RoboDojo worker kernel path {relative} at {commit}"
        )
    return result.stdout


def _paths_sha256(commit: str, paths: Sequence[str], root: Path) -> str:
    digest = hashlib.sha256()
    for relative in paths:
        data = _source(commit, relative, root)
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def worker_kernel_extension(
    recorded: str, *, measured_commit: str, plan_commit: str,
    project_root: Path | None = None,
) -> tuple[str, ...] | None:
    """The files a manifest grew by, when a record may cross that growth.

    Adding a file to ``WORKER_KERNEL_PATHS`` changes the plan-commit digest of
    every earlier record, even when not one executed byte differs. Such a
    record stays admissible exactly when (1) ``recorded`` is the digest of its
    measurement commit's own manifest, (2) the plan commit's manifest strictly
    contains that manifest, (3) the measured members, in their measured order,
    hash to ``recorded`` at the plan commit too, and (4) every added member
    exists at both commits with identical bytes. The result is then the sorted
    added paths. Anything else -- a changed, removed or unreadable file, a
    shrunk or merely different manifest -- is ``None``.
    """

    root = _PROJECT_ROOT if project_root is None else project_root
    try:
        _require_commit(measured_commit)
        _require_commit(plan_commit)
        measured = worker_kernel_paths(measured_commit, project_root=root)
        planned = worker_kernel_paths(plan_commit, project_root=root)
        if not set(measured) < set(planned):
            return None
        if (_paths_sha256(measured_commit, measured, root) != recorded
                or _paths_sha256(plan_commit, measured, root) != recorded):
            return None
        added = tuple(sorted(set(planned) - set(measured)))
        if any(_source(measured_commit, path, root) != _source(plan_commit, path, root)
               for path in added):
            return None
        return added
    except RoboDojoBenchmarkError:
        return None
