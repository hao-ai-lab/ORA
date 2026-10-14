"""Canonical tracked-source, shared evidence-workspace and local-state paths.

Linked Git worktrees isolate source checkouts, not Runtime evidence.  By
default every worktree belonging to one repository resolves the checkout that
owns the common Git directory as its physical workspace.  RunSpecs retain the
portable logical path ``outputs/<experiment>`` while runs, the resource-lease
registry and the reconstructable run index live under that shared workspace.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Iterator, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT_ENV = "ORA_WORKSPACE_ROOT"
#: Where run evidence is stored, independent of the workspace that holds leases and the run index. A host
#: with several Runtime workspaces (one per node, so leases stay node-local) sets it to one shared directory
#: so every experiment namespace has a single home.
OUTPUTS_ROOT_ENV = "ORA_OUTPUTS_ROOT"
EXTRA_OUTPUT_ROOTS_ENV = "ORA_EXTRA_OUTPUT_ROOTS"


def _git_dir(project_root: Path) -> Path | None:
    marker = project_root.resolve() / ".git"
    if marker.is_dir():
        return marker.resolve()
    if not marker.is_file():
        return None
    try:
        prefix, value = marker.read_text(encoding="utf-8").strip().split(":", 1)
    except (OSError, ValueError):
        return None
    if prefix.casefold() != "gitdir":
        return None
    path = Path(value.strip()).expanduser()
    return (path if path.is_absolute() else marker.parent / path).resolve()


def git_common_checkout_root(project_root: Path = PROJECT_ROOT) -> Path:
    """Return the checkout that owns the repository's common Git directory."""

    root = project_root.expanduser().resolve()
    git_dir = _git_dir(root)
    if git_dir is None:
        return root
    common = git_dir
    commondir = git_dir / "commondir"
    if commondir.is_file():
        try:
            value = Path(commondir.read_text(encoding="utf-8").strip())
            common = (
                value if value.is_absolute() else git_dir / value
            ).resolve()
        except OSError:
            return root
    return common.parent if common.name == ".git" else root


def resolve_workspace_root(
    project_root: Path = PROJECT_ROOT,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve the one physical Runtime workspace shared by linked worktrees."""

    values = os.environ if environ is None else environ
    configured = str(values.get(WORKSPACE_ROOT_ENV, "")).strip()
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            raise RuntimeError(f"{WORKSPACE_ROOT_ENV} must be an absolute path")
        resolved = path.resolve()
        if resolved == Path(resolved.anchor):
            raise RuntimeError(f"{WORKSPACE_ROOT_ENV} may not be a filesystem root")
        return resolved
    return git_common_checkout_root(project_root)


def resolve_outputs_root(
    workspace_root: Path,
    *,
    project_root: Path = PROJECT_ROOT,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Use shared host storage without relocating active lease/state files.

    Precedence: an explicit ``ORA_OUTPUTS_ROOT`` (an existing absolute
    directory) wins; otherwise an explicit workspace override stays
    self-contained (``<workspace>/outputs``); otherwise the selector local to
    Git's common directory, which fresh linked branches inherit without
    editing a RunSpec or relying on a shell's environment; otherwise
    ``<workspace>/outputs``.
    """
    values = os.environ if environ is None else environ
    explicit = str(values.get(OUTPUTS_ROOT_ENV, "")).strip()
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            raise RuntimeError(f"{OUTPUTS_ROOT_ENV} must be an absolute path")
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise RuntimeError(f"{OUTPUTS_ROOT_ENV} must name an existing directory: {exc}") from exc
        if not resolved.is_dir() or resolved == Path(resolved.anchor):
            raise RuntimeError(f"{OUTPUTS_ROOT_ENV} must be an existing non-root directory")
        return resolved
    if str(values.get(WORKSPACE_ROOT_ENV, "")).strip():
        return workspace_root / "outputs"
    selector = git_common_checkout_root(project_root) / ".git/ora-output-storage.json"
    if not selector.exists():
        return workspace_root / "outputs"
    try:
        record = json.loads(selector.read_text(encoding="utf-8"))
        if (
            not isinstance(record, dict)
            or set(record) != {"schema", "outputs_root"}
            or record["schema"] != "ora.output-storage.v1"
            or not isinstance(record["outputs_root"], str)
        ):
            raise ValueError("invalid storage selector schema")
        root = Path(record["outputs_root"]).expanduser()
        if not root.is_absolute() or root == Path(root.anchor):
            raise ValueError("outputs_root must be an absolute non-root directory")
        resolved = root.resolve(strict=True)
        if not resolved.is_dir() or resolved == Path(resolved.anchor):
            raise ValueError("outputs_root must be an existing non-root directory")
        return resolved
    except (OSError, ValueError, RuntimeError) as exc:
        raise RuntimeError(f"invalid output storage selector {selector}: {exc}") from exc


WORKSPACE_ROOT = resolve_workspace_root()
OUTPUTS_ROOT = resolve_outputs_root(WORKSPACE_ROOT)
STATE_ROOT = WORKSPACE_ROOT / ".harness-runtime"
RUN_INDEX_ROOT = STATE_ROOT / "run-index"
LEGACY_RUNS_ROOT = STATE_ROOT / "runs"


def stable_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise RuntimeError(f"{label} must be a stable id")
    return value


def experiment_workspace(experiment: str) -> Path:
    identifier = stable_id(experiment, "experiment")
    return (OUTPUTS_ROOT / identifier).resolve()


def linked_worktree_roots(project_root: Path = PROJECT_ROOT) -> tuple[Path, ...]:
    """Return live Git worktree roots for legacy-output discovery."""

    try:
        completed = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return (project_root.resolve(),)
    roots = []
    if completed.returncode == 0:
        for line in completed.stdout.splitlines():
            if not line.startswith("worktree "):
                continue
            candidate = Path(line.removeprefix("worktree ").strip()).expanduser()
            if candidate.is_absolute():
                roots.append(candidate.resolve())
    return tuple(dict.fromkeys(roots or [project_root.resolve()]))


def checkout_root_for(path: Path, *, default: Path | None = None) -> Path:
    """The checkout root one generated artefact's binding is recorded against.

    An artefact binding is `path relative to root` plus a digest, and the
    recorded value must stay `outputs/...` wherever the workspace physically
    sits. Binding against the launching checkout alone forced every servo
    qualification into the worktree it ran from, which is how the same label
    came to exist in up to four worktrees with nothing naming an authority.

    ``default`` is the caller's own root and is tried first, so a module that
    rebinds ``PROJECT_ROOT`` still decides for paths under it. When nothing
    matches, ``default`` is returned unchanged and the caller raises the error
    it always raised rather than a new one from here.
    """

    resolved = path.resolve()
    candidates = [root for root in (default, PROJECT_ROOT) if root is not None]
    candidates.extend(outputs.parent for outputs in output_roots())
    for root in dict.fromkeys(candidate.resolve() for candidate in candidates):
        if resolved.is_relative_to(root):
            return root
    return (default or PROJECT_ROOT).resolve()


def output_roots(
    *,
    include_legacy_worktrees: bool = True,
    environ: Mapping[str, str] | None = None,
) -> tuple[Path, ...]:
    """Return the shared root plus readable legacy/imported output roots."""

    roots = [OUTPUTS_ROOT.resolve()]
    if include_legacy_worktrees:
        roots.extend(
            (worktree / "outputs").resolve()
            for worktree in linked_worktree_roots()
            if (worktree / "outputs").is_dir()
        )
    values = os.environ if environ is None else environ
    for item in str(values.get(EXTRA_OUTPUT_ROOTS_ENV, "")).split(os.pathsep):
        if not item.strip():
            continue
        candidate = Path(item.strip()).expanduser()
        if not candidate.is_absolute():
            raise RuntimeError(f"{EXTRA_OUTPUT_ROOTS_ENV} entries must be absolute")
        roots.append(candidate.resolve())
    return tuple(dict.fromkeys(roots))


def trusted_output_roots() -> tuple[Path, ...]:
    """Roots a managed evidence file may resolve into.

    ``outputs/<experiment>`` is a symlink on a host whose home filesystem ran
    out of capacity, so a managed workspace legitimately resolves outside
    ``WORKSPACE_ROOT``. The target is trusted only because the workspace's own
    directory points at it; an unrelated path is still refused.
    """

    roots: list[Path] = []
    for output_root in output_roots():
        if not output_root.is_dir():
            continue
        roots.append(output_root.resolve())
        for child in sorted(output_root.iterdir()):
            if child.is_symlink() and child.is_dir():
                roots.append(child.resolve())
    return tuple(dict.fromkeys(roots))


def _registration_link(cursor: Path) -> bool:
    """Whether ``cursor`` is this workspace's ``outputs/<experiment>`` link onto that experiment's storage.

    With ``ORA_OUTPUTS_ROOT`` (or a selector) storing every namespace outside the
    workspace, ``link_logical_workspace`` makes the workspace's logical name a link onto
    ``<outputs root>/<experiment>``. The link is trusted because the workspace's own
    directory points at exactly that experiment's storage; any other target is not.
    """

    if not cursor.is_symlink() or not cursor.is_dir():
        return False
    if cursor.parent.resolve() != (WORKSPACE_ROOT / "outputs").resolve():
        return False
    try:
        storage = (OUTPUTS_ROOT / cursor.name).resolve(strict=True)
    except OSError:
        return False
    return cursor.resolve() == storage


def managed_evidence_path(path: Path | str, *, label: str) -> Path:
    """Resolve one evidence path, refusing anything outside a managed root.

    A recorded locator must name the real file, so a symlinked artefact or an
    intermediate symlinked directory is still refused: otherwise two locators
    could name one set of bytes, or a target could be swapped after the hash
    was taken. The one admitted hop is `outputs/<experiment>` itself: the link the
    capacity migration put onto another volume, or the workspace's registration
    link onto the configured outputs root (``_registration_link``).
    """

    supplied = Path(path).absolute()
    resolved = supplied.resolve()
    roots = trusted_output_roots()
    owning = next((root for root in roots if resolved.is_relative_to(root)), None)
    if owning is None:
        raise ValueError(f"{label} must live in a managed experiment output root")
    output_parents = {root.resolve() for root in output_roots()}
    cursor = supplied
    while cursor != cursor.parent and cursor.resolve() != owning:
        if _registration_link(cursor):
            cursor = cursor.parent
            continue
        if cursor.is_symlink():
            if not (cursor.parent.resolve() in output_parents and cursor.is_dir()):
                raise ValueError(f"{label} must not use a symlink below its output root")
            if cursor.resolve() not in roots:
                raise ValueError(f"{label} root symlink leaves the managed output roots")
        cursor = cursor.parent
    return resolved


def portable_output_locator(path: Path | str, root: Path) -> str:
    """Express one managed artefact as the portable `outputs/...` locator.

    Resolving first and subtracting `root` is what the readers and writers used
    to do, and it breaks wherever the capacity migration made
    `outputs/<experiment>` a link onto another volume: the resolved path is no
    longer under this checkout at all. The link's own name is still the
    locator, so map back through it instead of through the filesystem.

    Where ``ORA_OUTPUTS_ROOT`` stores namespaces outside the workspace
    and ``root`` is the workspace, an artefact under the outputs root maps to
    ``outputs/<experiment>/...`` and the workspace's registration link for that
    experiment is created if it is missing (as ``register_run`` does), so readers
    that resolve the locator against the workspace reach the same bytes.
    """

    supplied = Path(path).absolute()
    resolved = supplied.resolve()
    outputs = (Path(root) / "outputs").resolve()
    if supplied.is_relative_to(outputs):
        return ("outputs" / supplied.relative_to(outputs)).as_posix()
    if resolved.is_relative_to(outputs):
        return ("outputs" / resolved.relative_to(outputs)).as_posix()
    if outputs.is_dir():
        for child in sorted(outputs.iterdir()):
            if not child.is_symlink() or not child.is_dir():
                continue
            target = child.resolve()
            if resolved.is_relative_to(target):
                return ("outputs" / Path(child.name) / resolved.relative_to(target)).as_posix()
    storage = OUTPUTS_ROOT.resolve()
    if (
        Path(root).resolve() == WORKSPACE_ROOT.resolve()
        and storage != outputs
        and resolved.is_relative_to(storage)
        and resolved != storage
    ):
        relative = resolved.relative_to(storage)
        link_logical_workspace(relative.parts[0])
        return ("outputs" / relative).as_posix()
    raise ValueError("artefact has no portable outputs locator under this root")


def workspace_record_locator(path: Path | str, *, label: str) -> str:
    """The workspace-relative ``outputs/...`` locator of one qualification record.

    A record pair stores ``path`` relative to the workspace, and readers resolve
    it against ``WORKSPACE_ROOT``. With ``ORA_OUTPUTS_ROOT`` storing namespaces
    outside the workspace, the record a qualification step just wrote lives
    under the outputs root; ``portable_output_locator`` maps it through the
    namespace's registration link. The locator must reach exactly the same
    file, so a record kept anywhere else is still refused.
    """

    supplied = Path(path).absolute()
    resolved = supplied.resolve()
    try:
        locator = portable_output_locator(supplied, WORKSPACE_ROOT)
    except ValueError:
        locator = None
    if locator is None or (WORKSPACE_ROOT / locator).resolve() != resolved:
        raise ValueError(
            f"{label} must live in the shared workspace or the configured outputs root "
            f"({WORKSPACE_ROOT.resolve()} or {OUTPUTS_ROOT.resolve()}): {resolved}")
    return locator


def workspace_topology() -> dict[str, Any]:
    """Return local-only path provenance for operators and launch evidence."""

    return {
        "schema": "ora.workspace-topology.v1",
        "source_worktree": str(PROJECT_ROOT.resolve()),
        "workspace_root": str(WORKSPACE_ROOT.resolve()),
        "outputs_root": str(OUTPUTS_ROOT.resolve()),
        "state_root": str(STATE_ROOT.resolve()),
        "discovery_output_roots": [str(path) for path in output_roots()],
        "workspace_root_env": WORKSPACE_ROOT_ENV,
        "output_storage_selector": str(
            git_common_checkout_root() / ".git/ora-output-storage.json"
        ),
    }


def workspace_from_plan(plan: Mapping[str, Any]) -> Path:
    experiment = stable_id(plan.get("experiment"), "experiment")
    expected = Path("outputs") / experiment
    configured = Path(str(plan.get("output_workspace", "")))
    if configured != expected:
        raise RuntimeError(f"output_workspace must be {expected.as_posix()}")
    return experiment_workspace(experiment)


def new_run_root(plan: Mapping[str, Any], run_id: str) -> Path:
    identifier = stable_id(run_id, "run-id")
    return workspace_from_plan(plan) / "runs" / identifier


def _index_path(run_id: str) -> Path:
    return RUN_INDEX_ROOT / f"{stable_id(run_id, 'run-id')}.json"


def link_logical_workspace(experiment: str, storage: Path | None = None) -> Path:
    """Make ``outputs/<experiment>`` in the workspace root reach the experiment's storage.

    The resolver reads generated resources through the logical path
    (``spec._generated_resource``), so when an output-storage selector moves
    storage elsewhere the logical name is a symlink to it. Launch creates it
    (``register_run``); a builder that writes such a resource before any run
    (``benchmark_spec.compose``) creates it first. Returns the storage path.
    """

    identifier = stable_id(experiment, "experiment")
    target = experiment_workspace(identifier) if storage is None else storage
    logical_experiment = WORKSPACE_ROOT / "outputs" / identifier
    if logical_experiment.resolve() != target:
        logical_experiment.parent.mkdir(parents=True, exist_ok=True)
        try:
            logical_experiment.symlink_to(target, target_is_directory=True)
        except FileExistsError:
            if logical_experiment.resolve() != target:
                raise RuntimeError("logical output namespace conflicts with configured storage")
    return target


def register_run(run_id: str, plan: Mapping[str, Any], root: Path) -> None:
    identifier = stable_id(run_id, "run-id")
    expected = new_run_root(plan, identifier).resolve()
    if root.resolve() != expected:
        raise RuntimeError("run root does not match the plan output workspace")
    link_logical_workspace(plan["experiment"], expected.parent.parent)
    record = {
        "schema": "ora.run-index.v2",
        "run_id": identifier,
        "experiment": plan["experiment"],
        # Store the logical namespace, not the resolved storage mount. outputs
        # (or one experiment directory) may be an operator-owned symlink.
        "root": (Path("outputs") / plan["experiment"] / "runs" / identifier).as_posix(),
        "workspace_root": str(WORKSPACE_ROOT.resolve()),
        "source_worktree": str(PROJECT_ROOT.resolve()),
    }
    path = _index_path(identifier)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _indexed_root(run_id: str) -> Path | None:
    path = _index_path(run_id)
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        relative = Path(record["root"])
        experiment = stable_id(record["experiment"], "experiment")
    except (OSError, KeyError, TypeError, RuntimeError, json.JSONDecodeError):
        return None
    if relative.is_absolute() or ".." in relative.parts:
        return None
    if record.get("schema") == "ora.run-index.v2":
        expected_relative = Path("outputs") / experiment / "runs" / run_id
        if relative != expected_relative:
            return None
        root = (OUTPUTS_ROOT / experiment / "runs" / run_id).resolve()
    else:
        root = (PROJECT_ROOT / relative).resolve()
    if (
        root.name != run_id
        or root.parent.name != "runs"
        or root.parent.parent != experiment_workspace(experiment)
    ):
        return None
    return root if root.is_dir() else None


def _output_run_candidates(outputs: Path, run_id: str = "*") -> Iterator[Path]:
    """Follow declared experiment storage links, retaining exact run layout.

    Recursive glob does not descend into symlinked experiment directories.
    The explicit experiment/runs pattern does; a nested run link still must
    resolve under that experiment's own physical root.
    """
    candidates = list(outputs.glob(f"*/runs/{run_id}"))
    candidates.extend(outputs.glob(f"**/runs/{run_id}"))
    for path in candidates:
        root = path.resolve()
        if (
            path.is_dir()
            and root.name == path.name
            and root.parent.name == "runs"
            and root.parent.parent == path.parent.parent.resolve()
            and (path / "plan.json").is_file()
        ):
            yield root


def _occupied(path: Path) -> bool:
    """One ``lstat``: only a missing entry or a non-directory parent is free.

    Any other failure (permission, I/O) propagates, so spawn refuses rather
    than admitting a run-id it could not check.
    """

    try:
        os.lstat(path)
    except (FileNotFoundError, NotADirectoryError):
        return False
    return True


def existing_run_location(plan: Mapping[str, Any], run_id: str) -> Path | None:
    """Where ``run_id`` is already taken, or ``None``, without listing a directory.

    Spawn only has to show that a freshly minted run-id is unused. The run
    index is keyed by run-id alone and every spawn registers there, so it
    answers for a run in any namespace or layout. Then ``new_run_root``, each
    discovery root's exact ``<root>/<plan experiment>/runs/<run_id>`` and the
    legacy state root get one ``lstat`` each; a dangling link counts as taken.
    The namespace is the plan's experiment id, never the name of a resolved
    storage link, which may differ (``outputs/<experiment>`` can link to a
    directory of another name). ``resolve_run_root`` keeps its recursive
    discovery for reading existing runs. Spawn used to reuse it and walked
    every run artefact below all discovery roots, cold network storage
    included (median 10 s, outliers 53-118 s, 2026-09-30). The caller's
    ``mkdir`` of the run root without ``exist_ok`` remains the exclusive claim
    against a concurrent spawn of the same id.
    """

    identifier = stable_id(run_id, "run-id")
    experiment = stable_id(plan.get("experiment"), "experiment")
    candidates = [_index_path(identifier), new_run_root(plan, identifier)]
    candidates.extend(outputs / experiment / "runs" / identifier for outputs in output_roots())
    candidates.append(LEGACY_RUNS_ROOT / identifier)
    return next((path for path in candidates if _occupied(path)), None)


def resolve_run_root(run_id: str) -> Path:
    identifier = stable_id(run_id, "run-id")
    indexed = _indexed_root(identifier)
    if indexed is not None:
        return indexed
    candidates: list[Path] = []
    for outputs in output_roots():
        if not outputs.is_dir():
            continue
        candidates.extend(_output_run_candidates(outputs, identifier))
    legacy = (LEGACY_RUNS_ROOT / identifier).resolve()
    if legacy.is_dir():
        candidates.append(legacy)
    unique = list(dict.fromkeys(candidates))
    if len(unique) != 1:
        qualifier = "not found" if not unique else "ambiguous across output workspaces"
        raise RuntimeError(f"run-id {identifier} is {qualifier}")
    return unique[0]


def iter_run_roots() -> Iterator[Path]:
    seen: set[Path] = set()
    candidates: list[Path] = []
    for outputs in output_roots():
        if outputs.is_dir():
            candidates.extend(_output_run_candidates(outputs))
    if LEGACY_RUNS_ROOT.is_dir():
        candidates.extend(LEGACY_RUNS_ROOT.iterdir())
    for root in sorted((path.resolve() for path in candidates if path.is_dir()), reverse=True):
        if root not in seen:
            seen.add(root)
            yield root


def workspace_for_run(root: Path) -> Path:
    resolved = root.resolve()
    if resolved.parent.name == "runs" and (resolved / "plan.json").is_file():
        return resolved.parent.parent
    plan = json.loads((resolved / "plan.json").read_text(encoding="utf-8"))
    return workspace_from_plan(plan)
