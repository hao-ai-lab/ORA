"""Restricted geometry evaluator: a pure function over already-delivered numbers.

The policy writes one short block of numeric Python; this module runs it with no
filesystem, no network, no environment access and no state carried between calls.
It exists because measured teacher traces spend their longest reasoning segments
hand-deriving rotation matrices, camera back-projections and axis extrapolations,
and because that arithmetic competence does not survive distillation to a small
student while a named operator does.

The boundary this module does **not** cross: it never supplies a quantity the
Harness could have delivered. Coordinate frames, canonical target geometry,
calibration, controlled-point poses and reachability are delivered by their
owning tools. This evaluator is for the residual combinations that only the
policy can name.

Three enforced layers, in order:

1. An AST allowlist. Only assignment, expression, literal, operator, subscript,
   call and attribute nodes survive; imports, definitions, loops, comprehensions
   and dunder attributes are rejected before anything executes.
2. A namespace built from an explicit callable table. Module objects are never
   exposed, because ``numpy.load`` and ``numpy.fromfile`` read files.
3. A warm worker process that forks per request. The child installs
   ``RLIMIT_DATA``/``RLIMIT_CPU``, evaluates in a fresh empty namespace, writes
   its result to a pipe and exits. Nothing survives into the next call.

Two measured facts shape layer 3 and must not be "simplified" away:

* ``RLIMIT_AS`` set anywhere in a process that later imports numpy breaks the C
  extension load with a misleading "do not import numpy from its source
  directory" error. The real cause is OpenBLAS reserving per-thread buffer
  address space sized to the CPU count. ``OPENBLAS_NUM_THREADS=1`` fixes it;
  ``RLIMIT_DATA`` is used here regardless because it bounds the heap rather than
  the reserved address space.
* Starting a fresh interpreter per call costs ~0.55 s. Forking from a warm
  worker costs ~14 ms and keeps the fresh-namespace guarantee.
"""

from __future__ import annotations
from .compute_bindings import _digest

import ast
import atexit
import hashlib
import json
import os
import resource
import select
import signal
import subprocess
import sys
import threading
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from .robot import ToolError


CAPABILITY = "policy_restricted_geometry_compute_v1"
SCHEMA = "ora.restricted-geometry-compute.v1"

from .compute_grammar import (  # noqa: E402,F401 - the evaluator's allow-list, shared with its published grammar
    CHILD_CPU_SECONDS, CHILD_DATA_LIMIT_BYTES, MAX_BINDINGS, MAX_OUTPUTS, MAX_POW_EXPONENT,
    MAX_SOURCE_BYTES, MAX_STATEMENTS, MAX_VALUE_ELEMENTS, RESERVED_NAMES, WALL_TIMEOUT_S,
    WORKER_START_TIMEOUT_S, _ALLOWED_NODES, _INSTANCE_ATTRIBUTES, _NAMED_REJECTIONS,
    _REJECTION_RECOVERY, TARGET_RECOVERY, grammar,
)


def _reject(
    message: str, *, received: Any = None, field: str = "expressions",
    **constraints: Any,
) -> ToolError:
    """Every policy-fixable refusal, naming the field the policy must repair.

    ``field`` defaults to the block because that is where most refusals land.
    A binding refusal names its own entry instead: a union that swallows which
    half failed is the defect this tool's argument shape exists to avoid.
    """

    return ToolError(
        message, code="invalid_arguments", retryable=True,
        field=field, received=received, constraints=constraints or None,
    )


def _literal_exponent(node: ast.AST) -> float | None:
    """A numeric literal, optionally signed (``-2`` parses as a unary minus)."""

    sign = 1
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        sign = -1 if isinstance(node.op, ast.USub) else 1
        node = node.operand
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return sign * node.value
    return None


def validate_source(source: str) -> ast.Module:
    """Parse and allowlist one block. Every rejection is policy-fixable."""

    if not isinstance(source, str) or not source.strip():
        raise _reject("expressions must be a non-empty block of numeric Python")
    encoded = len(source.encode("utf-8"))
    if encoded > MAX_SOURCE_BYTES:
        raise _reject(
            "block is longer than the accepted source budget",
            received=encoded, maximum_bytes=MAX_SOURCE_BYTES,
            recovery="Use bindings for input arrays and prebound Rotation/geom helpers to shorten the block.",
        )
    try:
        tree = ast.parse(source, filename="<compute>", mode="exec")
    except SyntaxError as error:
        # The help says "encode line breaks as \\n", right for a JSON argument
        # wire. A wire that passes string parameters raw (vLLM's DeepSeek DSML)
        # then delivers those two characters instead of a line break: read them
        # as the line breaks the policy was told to write. The recorded tool
        # arguments keep the original text.
        repaired = None
        if "\\n" in source:
            try:
                repaired = ast.parse(source.replace("\\n", "\n"), filename="<compute>", mode="exec")
            except SyntaxError:
                repaired = None
        if repaired is None:
            raise _reject(
                f"block is not valid Python: {error.msg}",
                received=error.text.strip() if error.text else None,
                line=error.lineno,
                **({"recovery": "expressions contains a literal backslash followed by n where a "
                                "line break was meant. Put each statement on its own line or "
                                "separate statements with semicolons."}
                   if "\\n" in source else {}),
            ) from error
        tree = repaired

    if len(tree.body) > MAX_STATEMENTS:
        raise _reject(
            "block has more statements than the accepted budget",
            received=len(tree.body), maximum_statements=MAX_STATEMENTS,
            recovery="Use prebound Rotation/geom helpers or split independent calculations into separate calls.",
        )

    for node in ast.walk(tree):
        kind = type(node)
        if kind in _NAMED_REJECTIONS:
            raise _reject(
                f"{_NAMED_REJECTIONS[kind]} is not available in this evaluator. "
                "Use prebound np, Rotation, math and geom directly; write expressions without imports or definitions.",
                received=_NAMED_REJECTIONS[kind],
                allowed_statements=grammar()["statements"], allowed_forms=grammar()["forms"],
                **({"recovery": _REJECTION_RECOVERY[kind]}
                   if kind in _REJECTION_RECOVERY else {}),
            )
        if kind not in _ALLOWED_NODES:
            raise _reject(
                f"{kind.__name__} is not available in this evaluator",
                received=kind.__name__, allowed_statements=grammar()["statements"],
                allowed_forms=grammar()["forms"], not_available=grammar()["not_available"],
            )
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise _reject(
                    "attributes beginning with an underscore are not reachable",
                    received=node.attr,
                )
            if node.attr not in _reachable_attributes():
                raise _reject(
                    f"'{node.attr}' is not a reachable attribute in this evaluator",
                    received=node.attr, reachable=sorted(_reachable_attributes()),
                )
        if isinstance(node, ast.Assign):
            # `x, y, z, w = q` is how a quaternion is read, and refusing it
            # bought nothing: unpacking binds plain names too, one per element.
            for target in _assign_targets(node):
                if not isinstance(target, ast.Name):
                    raise _reject(
                        "assignment targets must be plain names, or a tuple of them; "
                        "subscript and attribute assignment are not available",
                        received=type(target).__name__, allowed_statements=grammar()["statements"],
                        recovery=TARGET_RECOVERY,
                    )
                if target.id in RESERVED_NAMES:
                    raise _reject(
                        "this name is bound by the evaluator and cannot be reassigned",
                        received=target.id, reserved=sorted(RESERVED_NAMES),
                    )
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            exponent = _literal_exponent(node.right)
            if exponent is None:
                raise _reject(
                    "the exponent of ** must be a numeric literal (a sign is allowed: x ** -2)",
                    received=type(node.right).__name__,
                )
            if abs(exponent) > MAX_POW_EXPONENT:
                raise _reject(
                    "the exponent of ** is outside the accepted range",
                    received=exponent, maximum_absolute_exponent=MAX_POW_EXPONENT,
                )
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if len(node.value) > 32:
                raise _reject(
                    "string literals are limited to short axis and sequence labels",
                    received=len(node.value), maximum_length=32,
                )
    return tree


# --------------------------------------------------------------------------
# Layer 2 — namespace built from an explicit callable table
# --------------------------------------------------------------------------

from .compute_namespace import (  # noqa: F401
    _LINALG, _NUMPY, _ROTATION, _GEOMETRY, _MATH, _Table, _namespace,
)


_REACHABLE_ATTRIBUTES: frozenset[str] | None = None


def _reachable_attributes() -> frozenset[str]:
    """Every attribute name layer 1 admits: the tables, plus value conversions.

    Derived from the tables rather than restated, so adding an entry to a table
    cannot leave it unreachable and removing one cannot leave it reachable.
    """

    global _REACHABLE_ATTRIBUTES
    if _REACHABLE_ATTRIBUTES is None:
        table_names: set[str] = set()
        for table in (_NUMPY, _LINALG, _ROTATION, _GEOMETRY, _MATH):
            table_names |= set(object.__getattribute__(table, "_entries"))
        _REACHABLE_ATTRIBUTES = frozenset(table_names | _INSTANCE_ATTRIBUTES)
    return _REACHABLE_ATTRIBUTES


# --------------------------------------------------------------------------
# Result serialisation — fixed shape, explicit kind
# --------------------------------------------------------------------------

class _NotNumeric(Exception):
    """A binding that is a legitimate intermediate, not a returnable measurement."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _serialise(name: str, value: Any) -> dict[str, Any]:
    """One output entry.

    The key names are fixed (``name``/``value``/``kind``) and the identifier the
    policy chose appears only as a *value*. Trajectory and context auditors scan
    key paths for forbidden fragments, so a free ``{identifier: value}`` mapping
    would let a block named ``contact`` invalidate the whole run.

    A non-numeric binding raises :class:`_NotNumeric` so the caller can list it
    as an intermediate instead of failing the block. A binding that *is* numeric
    but unusable (non-finite, oversized, too many dimensions) is an error: the
    policy asked for a measurement and must be told it cannot have it.
    """

    if isinstance(value, Rotation):
        raise _NotNumeric("rotation object; call as_quat, as_matrix, as_rotvec or as_euler")
    if isinstance(value, np.ndarray) and value.ndim == 0:
        # np.where(c, a, b) and np.array(x) of scalars are 0-d arrays, not tensors.
        value = value.item()
    if isinstance(value, (bool, np.bool_)):
        return {"name": name, "value": bool(value), "kind": "boolean"}
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        if number != number or number in (float("inf"), float("-inf")):
            raise _reject(f"'{name}' is not a finite number; check division by zero, input scale and units", received=str(value))
        # Rounded on the same rule as a float array below. Unrounded, the same
        # quantity reads as 0.30000000000000004 when the block binds it alone
        # and 0.3 when it is an element of a vector, and a result schema cannot
        # state a precision that depends on how the policy asked. Nine decimals
        # is nanometres at this embodiment's scale.
        return {"name": name, "value": round(number, 9), "kind": "scalar"}

    try:
        array = np.asarray(value)
    except Exception:  # noqa: BLE001 - any object numpy cannot view is an intermediate
        raise _NotNumeric(f"{type(value).__name__} is not a numeric value") from None
    if array.dtype.kind not in "fiub":
        raise _NotNumeric(f"{array.dtype} is not a numeric dtype")
    if array.size > MAX_VALUE_ELEMENTS:
        raise _reject(
            f"'{name}' has more elements than one result may carry",
            received=int(array.size), maximum_elements=MAX_VALUE_ELEMENTS,
            recovery="Return a slice or summary, or use outputs to select the small final results instead of this temporary array.",
        )
    if not np.all(np.isfinite(array)):
        raise _reject(f"'{name}' contains a non-finite entry; check division by zero, input scale and units", received=name)

    # Rounding is a float operation. `np.round` on a boolean array raises
    # `UFuncTypeError: Cannot cast ufunc 'multiply' output ... to dtype('bool')`,
    # which named nothing the policy could act on -- and `b` is admitted by the
    # dtype check ten lines above, so the module accepted a value it could not
    # return. `Compare` is an allowed node, so `clear = margins > 0.01` is the
    # natural way to ask whether something fits; it failed on the way back.
    # Integers and unsigned integers are already exact, so only floats round.
    listed = (
        np.round(array, 9) if array.dtype.kind == "f" else array
    ).tolist()
    if array.ndim == 1:
        kind = {3: "vector3", 4: "vector4"}.get(array.shape[0], "vector")
    elif array.ndim == 2:
        kind = "matrix3x3" if array.shape == (3, 3) else "matrix"
    else:
        raise _reject(
            f"'{name}' has more than two dimensions",
            received=int(array.ndim), maximum_dimensions=2,
        )
    return {"name": name, "value": listed, "kind": kind}


def _assign_targets(node: ast.Assign) -> list[ast.AST]:
    """Every name an assignment binds, unpacking one level of tuple or list."""

    out: list[ast.AST] = []
    for target in node.targets:
        if isinstance(target, (ast.Tuple, ast.List)):
            out.extend(target.elts)
        else:
            out.append(target)
    return out


def _consumed_names(tree: ast.Module) -> frozenset[str]:
    """Names some statement reads, which makes them scaffolding, not answers.

    Writing a rotation matrix by element binds nine names before anything uses
    them. Counting those against the returned-measurement ceiling refused half
    the blocks a live policy wrote, and each refusal was followed by the same
    computation spelled with fewer names -- so the cap cost a round trip and
    changed no answer. A name a later line reads is still reported by name
    among the intermediates; it is simply not one of the answers.
    """

    read: set[str] = set()
    for statement in tree.body:
        targets = (
            {t.id for t in _assign_targets(statement) if isinstance(t, ast.Name)}
            if isinstance(statement, ast.Assign) else set()
        )
        for node in ast.walk(statement):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id not in targets:
                    read.add(node.id)
    return frozenset(read)


def _collect(namespace: Mapping[str, Any], last: Any, bound_keys: Sequence[str],
             consumed: frozenset[str] = frozenset(),
             selected: Sequence[str] | None = None) -> dict[str, Any]:
    """Partition the block's bindings into returned measurements and intermediates.

    Intermediates are named rather than dropped. Silent omission is the defect
    class this whole unit exists to remove; a block whose only binding is a
    rotation object must be told so, not handed an empty result.
    """

    bound = [(name, namespace[name]) for name in bound_keys if name in namespace]
    if last is not None:
        bound.append(("result", last))

    outputs: list[dict[str, Any]] = []
    intermediates: list[str] = []
    if selected is not None:
        values = dict(bound)
        missing = [name for name in selected if name not in values]
        if missing:
            raise _reject("outputs names a value the block did not produce",
                          field="outputs", received=missing, available=list(values))
        intermediates.extend(
            f"{name} (not selected for output)" for name, _ in bound if name not in selected
        )
        bound = [(name, values[name]) for name in selected]
    for name, value in bound:
        try:
            outputs.append(_serialise(name, value))
        except _NotNumeric as skipped:
            intermediates.append(f"{name} ({skipped.reason})")

    # Over the ceiling, demote scaffolding before refusing. A name a later line
    # reads is working state; a name nothing reads is what the block was for.
    # Four of eight refusals in one batch were blocks building a single answer
    # out of a rotation matrix written element by element, and each was rewritten
    # with fewer names and accepted -- so the ceiling cost a round trip and
    # changed no answer. Demoting is not dropping: every demoted name is listed.
    if len(outputs) > MAX_OUTPUTS and consumed:
        kept, demoted = [], []
        for entry in outputs:
            (demoted if entry["name"] in consumed else kept).append(entry)
        while demoted and len(kept) + len(demoted) > MAX_OUTPUTS:
            intermediates.append(f"{demoted.pop(0)['name']} (read by a later line)")
        outputs = kept + demoted
        outputs.sort(key=lambda entry: [e["name"] for e in bound].index(entry["name"])
                     if any(e["name"] == entry["name"] for e in []) else 0)
        outputs = [e for name, _ in bound for e in outputs if e["name"] == name]

    if len(outputs) > MAX_OUTPUTS:
        raise _reject(
            "the block returns more measurements than one result may carry",
            received=len(outputs), maximum_outputs=MAX_OUTPUTS,
            recovery="Set outputs to the final variable names you need, or pack related values into one small vector.",
        )
    if not outputs:
        raise _reject(
            "the block bound no numeric measurement to return; "
            + ("its bindings are intermediates: " + "; ".join(intermediates)
               if intermediates else "it bound nothing"),
            received=len(bound),
        )
    payload: dict[str, Any] = {"outputs": outputs}
    if intermediates:
        payload["intermediates_not_returned"] = "; ".join(intermediates)
    return payload


# --------------------------------------------------------------------------
# Layer 3 — warm worker, fork per request
# --------------------------------------------------------------------------

#: Attributes whose first call resolves a lazy import inside numpy.
#:
#: ``ndarray.sum`` and its siblings dispatch through ``numpy.core._methods``,
#: which imports on first use. The child has empty builtins, so that import
#: raises ``KeyError: '__import__'`` -- a dependency's internal failure standing
#: in for a reason the policy could act on, and the same accident this module
#: already refuses to rely on as a defence. Ten of the forty-eight entries in
#: ``_INSTANCE_ATTRIBUTES`` were advertised and unusable because of it, ``clip``
#: among them, which is the second most frequent primitive in the measured
#: corpus.
#:
#: The parent has real builtins, so calling each one here once resolves the
#: import before any fork and every child inherits it. This is what the warm
#: parent is for; it was importing numpy and never exercising these paths.
_LAZY_IMPORT_METHODS = (
    "all", "any", "clip", "max", "mean", "min", "prod", "std", "sum", "var",
)


def _resolve_lazy_numpy_imports() -> None:
    """Call each lazily-importing method once, in the parent, before any fork."""

    probe = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    for name in _LAZY_IMPORT_METHODS:
        method = getattr(probe, name)
        method(0.0, 1.0) if name == "clip" else method()


_resolve_lazy_numpy_imports()


def _child_evaluate(
    tree: ast.Module, bindings: Mapping[str, Any], outputs: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Runs inside the forked child. Never returns to the parent's namespace."""

    resource.setrlimit(resource.RLIMIT_DATA, (CHILD_DATA_LIMIT_BYTES,) * 2)
    resource.setrlimit(resource.RLIMIT_CPU, (CHILD_CPU_SECONDS, CHILD_CPU_SECONDS))
    namespace = _namespace()
    for key, value in bindings.items():
        namespace[key] = np.asarray(value) if isinstance(value, list) else value

    # Assigned names in source order. Injected bindings are inputs the policy
    # already holds, so they are not echoed back as outputs unless reassigned.
    assigned: list[str] = []
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            for target in _assign_targets(statement):
                if isinstance(target, ast.Name) and target.id not in assigned:
                    assigned.append(target.id)

    last: Any = None
    body = list(tree.body)
    if body and isinstance(body[-1], ast.Expr):
        tail = body.pop()
        exec(compile(ast.Module(body=body, type_ignores=[]), "<compute>", "exec"), namespace)
        last = eval(compile(ast.Expression(tail.value), "<compute>", "eval"), namespace)
    else:
        exec(compile(tree, "<compute>", "exec"), namespace)
    return _collect(namespace, last, assigned, _consumed_names(tree), selected=outputs)


def _worker_main() -> None:
    """Warm worker: import once, fork per request, relay the child's answer."""

    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.stdout.write(json.dumps({"ready": True}) + "\n")
    sys.stdout.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except ValueError:
            sys.stdout.write(json.dumps({"error": {
                "code": "internal_tool_error", "message": "malformed worker request",
            }}) + "\n")
            sys.stdout.flush()
            continue
        sys.stdout.write(json.dumps(_fork_and_run(request)) + "\n")
        sys.stdout.flush()


def _fork_and_run(request: Mapping[str, Any]) -> dict[str, Any]:
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        try:
            os.close(read_fd)
            tree = validate_source(request.get("source", ""))
            payload = _child_evaluate(tree, request.get("bindings") or {}, request.get("outputs"))
        except ToolError as error:
            payload = {"error": error.as_dict()}
        except MemoryError:
            payload = {"error": {
                "code": "invalid_arguments", "retryable": True, "field": "expressions",
                "message": "the block asked for more memory than one evaluation may use",
                "constraints": {"maximum_bytes": CHILD_DATA_LIMIT_BYTES},
            }}
        except BaseException as error:  # noqa: BLE001 - reported, never raised across the fork
            payload = {"error": {
                "code": "invalid_arguments", "retryable": True, "field": "expressions",
                "message": f"{type(error).__name__}: {str(error)[:200]}",
                "constraints": {"allowed": "the prebound names np, Rotation, math, geom, pi, "
                                           "float, int, abs, max, min and round, your bindings "
                                           "and names the block assigns; no print, return, "
                                           "list(), str() or imports"},
            }}
        try:
            os.write(write_fd, json.dumps(payload).encode("utf-8"))
        finally:
            os._exit(0)

    os.close(write_fd)
    chunks: list[bytes] = []
    deadline_hit = False
    try:
        while True:
            ready, _, _ = select.select([read_fd], [], [], WALL_TIMEOUT_S)
            if not ready:
                deadline_hit = True
                break
            chunk = os.read(read_fd, 65536)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(read_fd)
        if deadline_hit:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        _, status = os.waitpid(pid, 0)

    if deadline_hit:
        return {"error": {
            "code": "invalid_arguments", "retryable": True, "field": "expressions",
            "message": "the block did not finish inside the evaluation budget",
            "constraints": {"maximum_seconds": WALL_TIMEOUT_S},
        }}
    if not chunks:
        signalled = os.WTERMSIG(status) if os.WIFSIGNALED(status) else None
        if signalled == signal.SIGXCPU:
            return {"error": {
                "code": "invalid_arguments", "retryable": True, "field": "expressions",
                "message": "the block used more processor time than one evaluation may use",
                "constraints": {"maximum_cpu_seconds": CHILD_CPU_SECONDS},
            }}
        return {"error": {
            "code": "internal_tool_error", "retryable": False,
            "message": "the evaluator produced no result",
        }}
    return json.loads(b"".join(chunks).decode("utf-8"))


# --------------------------------------------------------------------------
# Provenance — what a receipt needs to be recomputed offline
# --------------------------------------------------------------------------



class RestrictedComputeRuntime:
    """Owns the warm worker. One instance per policy session."""

    def __init__(self) -> None:
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()

    def _ensure_worker(self) -> subprocess.Popen[str]:
        if self._process is not None and self._process.poll() is None:
            return self._process
        environment = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1",
        }
        process = subprocess.Popen(
            [sys.executable, "-m", "ora.agent.tools.compute", "--worker"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            env=environment, text=True, bufsize=1,
        )
        handshake = process.stdout.readline() if process.stdout else ""
        if not handshake.strip():
            process.kill()
            raise ToolError(
                "the geometry evaluator did not start",
                code="internal_tool_error", retryable=False,
            )
        self._process = process
        return process

    def evaluate(
        self, source: str, bindings: Mapping[str, Any] | None = None,
        *, outputs: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Validate in-process, then evaluate in a forked child of the worker."""

        validate_source(source)  # fail fast with a policy-visible reason
        if outputs is not None:
            if (not isinstance(outputs, list) or not 1 <= len(outputs) <= MAX_OUTPUTS
                or any(not isinstance(name, str) or not name.isidentifier() or name.startswith("_")
                       for name in outputs) or len(set(outputs)) != len(outputs)):
                raise _reject("outputs must list 1 to 12 distinct final variable names (or result for the last expression)",
                              field="outputs", received=outputs, maximum_outputs=MAX_OUTPUTS)
        from .compute_arguments import _split_bindings

        values, provenance = _split_bindings(bindings or {})
        request = json.dumps({"source": source, "bindings": values, "outputs": outputs})
        with self._lock:
            worker = self._ensure_worker()
            assert worker.stdin is not None and worker.stdout is not None
            worker.stdin.write(request + "\n")
            worker.stdin.flush()
            answer = worker.stdout.readline()
        if not answer.strip():
            self.close()
            raise ToolError(
                "the geometry evaluator stopped responding",
                code="internal_tool_error", retryable=False,
            )
        payload = json.loads(answer)
        error = payload.get("error")
        if error:
            raise ToolError(
                error.get("message", "the block could not be evaluated"),
                code=error.get("code", "invalid_arguments"),
                retryable=bool(error.get("retryable", True)),
                field=error.get("field"), received=error.get("received"),
                constraints=error.get("constraints"),
            )
        # Recorded in evidence, never shown to the model and never charged
        # against the delivery budget: `native/session.py` pops `_runtime` from
        # the visible message and `skill_receipt.py` excludes it from the
        # measured body. The source is bound by digest rather than echoed,
        # because both auditors open strings that begin with `{` or `[`; the
        # block itself is already recorded verbatim in the tool call.
        payload["_runtime"] = {
            "provenance": {
                "schema": SCHEMA,
                "source_sha256": _digest(source),
                "inputs": provenance,
                **({"selected_outputs": list(outputs)} if outputs is not None else {}),
            }
        }
        return payload

    def close(self) -> None:
        process, self._process = self._process, None
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=5)


_SHARED: RestrictedComputeRuntime | None = None
_SHARED_LOCK = threading.Lock()


def shared_runtime() -> RestrictedComputeRuntime:
    """The process-wide evaluator, closed at interpreter exit.

    A handler cannot own a per-session instance today: neither
    `ManipulationToolRuntime` nor `RobotToolRuntime` has a teardown hook and
    nothing calls `close()`, so a per-session worker would leak one process per
    episode for the lifetime of the Harness.

    Sharing one worker costs nothing the isolation contract depends on. Freshness
    comes from the fork -- every request evaluates in a new child with a new
    empty namespace and its own rlimits -- not from the identity of the parent,
    and `evaluate` already serialises on its own lock. Bind a per-session
    instance instead once a session teardown hook exists.
    """

    global _SHARED
    with _SHARED_LOCK:
        if _SHARED is None:
            _SHARED = RestrictedComputeRuntime()
            atexit.register(_SHARED.close)
        return _SHARED


def handle(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Forwarder. The tool-call edge lives in :mod:`compute_arguments`.

    The kernel is the sandbox and the evaluator; reading either published
    argument spelling and joining a result to a surface are a different job and
    a different module. Kept here because callers and tests already name it.
    """

    from .compute_arguments import handle as _handle

    return _handle(arguments)


if __name__ == "__main__":  # pragma: no cover - worker entry point
    if "--worker" in sys.argv:
        _worker_main()
