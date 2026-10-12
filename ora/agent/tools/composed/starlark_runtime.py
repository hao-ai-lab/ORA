"""Deterministic Starlark evaluation for composed tool bodies.

A composed tool body is reviewed, offline-authored Starlark whose only contact
with the world is a fixed set of host symbols such as ``sdk.move`` or
``sdk.check.hold``. This module owns that language boundary and nothing above
it: parsing, the symbol table, evaluation limits and journaled replay. It
defines no SDK function, tool, stage or receipt.

Replay is the execution model of the composed tool design. Starlark has neither
generators nor exceptions, so a body cannot pause at an action. Instead every
host call is journaled. Re-running an entry with the same inputs and journal
returns each recorded result in order; the first *action* past the end of the
journal stops interpretation and is handed to the caller as :class:`Suspended`.
The caller performs it, appends the outcome with :meth:`HostRequest.resolve`
and runs again. A recorded call that no longer matches what the body asks for
is ``replay_divergence``, never a silently different path.

Host symbols come in three kinds. A *query* reads the world and is journaled;
an *action* changes it and suspends interpretation; a *pure* symbol is a
deterministic host function over plain data (a declaration constructor, a
result constructor, vector arithmetic). Pure calls are neither journaled nor
replayed: they are recomputed on every interpretation, may run while the
module loads, and are bound once with the program rather than per run.

Facts measured against ``starlark-pyo3`` 2026.1.2 that shape this module:

* ``Dialect.standard()`` has no ``while``, no top-level ``if``/``for``, no
  f-strings, no annotations and no ``load``; the standard globals have no
  ``print``, clock, randomness or ``set``. ``struct`` comes from the
  ``StructType`` extension and is the only extension enabled.
* Recursion is *permitted* by starlark-rust. ``max_callstack`` bounds its
  depth; it is not rejected statically.
* Values cross the boundary as JSON data both ways. An object with no JSON form
  is refused, and a NaN or infinity leaving Starlark arrives as ``None``. Host
  results are therefore checked to be finite plain data before Starlark sees
  them, and a body's own non-finite arithmetic is a known lossy edge.
* ``check_cancelled`` is polled about every 1000 bytecode instructions, so one
  tick of ``max_ticks`` is that many instructions.
* Allocation is not bounded: ``[0] * n`` is one instruction. These limits stop
  authoring mistakes such as an unbounded ``range``, not an adversary. Code the
  policy writes at run time goes to ``compute``, never here.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import starlark as sl

HostKind = Literal["query", "action", "pure"]

ERROR_CODES = (
    "parse_error",
    "load_forbidden",
    "reserved_identifier",
    "missing_entrypoint",
    "missing_global",
    "host_call_at_load",
    "action_in_read_only_entry",
    "host_error",
    "host_refusal",
    "non_data_value",
    "replay_divergence",
    "tick_budget_exhausted",
    "deadline_exceeded",
    "callstack_exceeded",
    "contract_fail",
    "evaluation_error",
)

_SYMBOL = re.compile(r"^[a-z_][a-z0-9_]*(\.[a-z_][a-z0-9_]*)*$")
_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")
_GLOBAL = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RESERVED = "__host_"
_DIALECT = sl.Dialect.standard()
_GLOBALS = sl.Globals.extended_by([sl.LibraryExtension.StructType])
_DETAIL_CHARS = 2000


class StarlarkProgramError(Exception):
    """A body could not be loaded or did not finish; ``code`` is in ERROR_CODES.

    A host function refuses on purpose (for example a spent query budget) by
    raising this with ``host_refusal`` and a machine-readable ``detail``; any
    other exception from a host function becomes ``host_error``.
    """

    def __init__(self, code: str, detail: str) -> None:
        assert code in ERROR_CODES, code
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class StarlarkLimits:
    max_ticks: int = 10_000
    max_callstack: int = 64
    wall_s: float = 30.0


@dataclass(frozen=True)
class HostCall:
    symbol: str
    args: list[Any]
    kwargs: dict[str, Any]
    result: Any


@dataclass(frozen=True)
class HostRequest:
    """An action the body asked for that the journal does not yet contain."""

    symbol: str
    args: list[Any]
    kwargs: dict[str, Any]

    def resolve(self, result: Any) -> HostCall:
        return HostCall(self.symbol, self.args, self.kwargs, plain_data(result, f"{self.symbol} result"))


@dataclass(frozen=True)
class Completed:
    value: Any
    journal: tuple[HostCall, ...]
    ticks: int


@dataclass(frozen=True)
class Suspended:
    request: HostRequest
    journal: tuple[HostCall, ...]
    ticks: int


def plain_data(value: Any, where: str) -> Any:
    """Return ``value`` as finite JSON data (tuples become lists) or refuse it."""

    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise StarlarkProgramError("non_data_value", f"{where}: non-finite float")
        return value
    if isinstance(value, (list, tuple)):
        return [plain_data(item, f"{where}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, Mapping):
        out = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise StarlarkProgramError("non_data_value", f"{where}: key {key!r} is not a string")
            out[key] = plain_data(item, f"{where}.{key}")
        return out
    raise StarlarkProgramError("non_data_value", f"{where}: {type(value).__name__} is not plain data")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class _Abort(Exception):
    """Raised inside a host callable to stop interpretation; never escapes."""


@dataclass
class _Run:
    queries: Mapping[str, Callable[..., Any]]
    allow_actions: bool
    journal: list[HostCall]
    deadline: float
    max_ticks: int
    cursor: int = 0
    ticks: int = 0
    pending: HostRequest | None = None
    fault: StarlarkProgramError | None = None
    entry: str = field(default="")


def _prelude(symbols: Sequence[str]) -> str:
    tree: dict[str, Any] = {}
    for index, symbol in enumerate(symbols):
        node = tree
        *parents, leaf = symbol.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = f"{_RESERVED}{index}"

    def render(node: Any) -> str:
        if isinstance(node, str):
            return node
        return "struct(" + ", ".join(f"{key} = {render(child)}" for key, child in node.items()) + ")"

    return "".join(f"{name} = {render(node)}\n" for name, node in tree.items())


def _validate_host(host: Mapping[str, HostKind]) -> None:
    for symbol, kind in host.items():
        if not _SYMBOL.match(symbol) or symbol.startswith(_RESERVED):
            raise ValueError(f"host symbol {symbol!r} is not a dotted lowercase name")
        if kind not in ("query", "action", "pure"):
            raise ValueError(f"host symbol {symbol!r} has kind {kind!r}")
        if any(other.startswith(symbol + ".") for other in host):
            raise ValueError(f"host symbol {symbol!r} is also a namespace")


class StarlarkProgram:
    """One loaded body: frozen module, its host table and its source digest."""

    def __init__(
        self,
        *,
        filename: str,
        source: str,
        host: Mapping[str, HostKind],
        entrypoints: Collection[str],
        limits: StarlarkLimits,
        pure: Mapping[str, Callable[..., Any]] | None = None,
        read_globals: Collection[str] = (),
    ) -> None:
        _validate_host(host)
        pure = dict(pure or {})
        declared_pure = {symbol for symbol, kind in host.items() if kind == "pure"}
        if set(pure) != declared_pure:
            raise ValueError(f"pure must bind exactly {sorted(declared_pure)}, got {sorted(pure)}")
        for name in read_globals:
            if not _GLOBAL.match(name):
                raise ValueError(f"global {name!r} is not an identifier")
        if _RESERVED in source:
            raise StarlarkProgramError("reserved_identifier", f"{_RESERVED}* names belong to the host")
        for entry in entrypoints:
            if not _IDENTIFIER.match(entry):
                raise ValueError(f"entrypoint {entry!r} is not an identifier")
        self.filename = filename
        self.digest = "sha256:" + hashlib.sha256(source.encode("utf-8")).hexdigest()
        self.host = dict(host)
        self.entrypoints = tuple(entrypoints)
        self.limits = limits
        self._pure = pure
        self._lock = threading.Lock()
        self._run: _Run | None = None
        self._called_at_load: str | None = None
        self._load_fault: StarlarkProgramError | None = None

        try:
            body = sl.parse(filename, source, _DIALECT)
        except sl.StarlarkError as exc:
            raise StarlarkProgramError("parse_error", _detail(exc)) from None
        loads = body.loads()
        if loads:
            raise StarlarkProgramError("load_forbidden", f"{filename} loads {[item.module_id for item in loads]}")

        symbols = list(self.host)
        module = sl.Module()
        for index, symbol in enumerate(symbols):
            module.add_callable(f"{_RESERVED}{index}", self._router(symbol))
        options = sl.EvalOptions(max_callstack_size=limits.max_callstack)
        try:
            sl.eval_with(options, module, sl.parse("<host>", _prelude(symbols), _DIALECT), _GLOBALS)
            sl.eval_with(options, module, body, _GLOBALS)
        except sl.StarlarkError as exc:
            if self._load_fault is not None:
                raise self._load_fault from None
            if self._called_at_load is not None:
                raise StarlarkProgramError("host_call_at_load", f"{self._called_at_load} called at top level") from None
            raise StarlarkProgramError("evaluation_error", _detail(exc)) from None
        if self.entrypoints:
            probe = "[" + ", ".join(f"type({entry})" for entry in self.entrypoints) + "]\n"
            try:
                kinds = sl.eval(module, sl.parse("<entrypoints>", probe, _DIALECT), _GLOBALS)
            except sl.StarlarkError as exc:
                raise StarlarkProgramError("missing_entrypoint", _detail(exc)) from None
            missing = [entry for entry, kind in zip(self.entrypoints, kinds) if kind != "function"]
            if missing:
                raise StarlarkProgramError("missing_entrypoint", f"not functions: {missing}")
        self.globals: dict[str, Any] = {}
        for name in read_globals:
            try:
                value = sl.eval(module, sl.parse("<globals>", f"{name}\n", _DIALECT), _GLOBALS)
            except sl.StarlarkError as exc:
                raise StarlarkProgramError("missing_global", f"{name}: {_detail(exc)}") from None
            self.globals[name] = plain_data(value, name)
        self._module = module.freeze()

    def run(
        self,
        entry: str,
        *,
        args: Sequence[Any] = (),
        kwargs: Mapping[str, Any] | None = None,
        queries: Mapping[str, Callable[..., Any]],
        allow_actions: bool,
        journal: Sequence[HostCall] = (),
        limits: StarlarkLimits | None = None,
    ) -> Completed | Suspended:
        """Interpret ``entry`` from the start, replaying ``journal`` first."""

        if entry not in self.entrypoints:
            raise ValueError(f"{entry!r} is not a declared entrypoint of {self.filename}")
        expected = {symbol for symbol, kind in self.host.items() if kind == "query"}
        if set(queries) != expected:
            raise ValueError(f"queries must bind exactly {sorted(expected)}, got {sorted(queries)}")
        call_args = plain_data(list(args), "args")
        call_kwargs = plain_data(dict(kwargs or {}), "kwargs")
        limits = limits or self.limits
        state = _Run(
            queries=queries,
            allow_actions=allow_actions,
            journal=list(journal),
            deadline=time.monotonic() + limits.wall_s,
            max_ticks=limits.max_ticks,
            entry=entry,
        )

        def cancelled() -> bool:
            state.ticks += 1
            if state.ticks > state.max_ticks:
                state.fault = StarlarkProgramError("tick_budget_exhausted", f"over {state.max_ticks} ticks")
            elif time.monotonic() > state.deadline:
                state.fault = StarlarkProgramError("deadline_exceeded", f"over {limits.wall_s} s")
            return state.fault is not None

        options = sl.EvalOptions(check_cancelled=cancelled, max_callstack_size=limits.max_callstack)
        with self._lock:
            self._run = state
            try:
                value = self._module.call_with(options, entry, *call_args, **call_kwargs).value
            except sl.StarlarkError as exc:
                if state.pending is not None and state.fault is None:
                    return Suspended(state.pending, tuple(state.journal), state.ticks)
                if state.fault is not None:
                    raise state.fault from None
                raise _classify(exc) from None
            finally:
                self._run = None
        if state.cursor != len(state.journal):
            raise StarlarkProgramError(
                "replay_divergence", f"{entry} finished after {state.cursor} of {len(state.journal)} recorded calls"
            )
        return Completed(plain_data(value, f"{entry} return"), tuple(state.journal), state.ticks)

    def _router(self, symbol: str) -> Callable[..., Any]:
        kind = self.host[symbol]

        def route(*args: Any, **kwargs: Any) -> Any:
            state = self._run
            if kind == "pure":
                try:
                    return self._call_pure(symbol, args, kwargs)
                except StarlarkProgramError as fault:
                    if state is None:
                        self._load_fault = self._load_fault or fault
                    else:
                        state.fault = state.fault or fault
                    raise _Abort(fault.code) from None
            if state is None:
                self._called_at_load = symbol
                raise _Abort(symbol)
            try:
                return self._dispatch(state, symbol, kind, plain_data(list(args), symbol), plain_data(kwargs, symbol))
            except StarlarkProgramError as fault:
                state.fault = state.fault or fault
                raise _Abort(fault.code) from None

        return route

    def _call_pure(self, symbol: str, args: Sequence[Any], kwargs: Mapping[str, Any]) -> Any:
        call_args = plain_data(list(args), symbol)
        call_kwargs = plain_data(dict(kwargs), symbol)
        try:
            result = self._pure[symbol](*call_args, **call_kwargs)
        except StarlarkProgramError:
            raise
        except Exception as exc:  # the pure function's own refusal, reported by name
            raise StarlarkProgramError("host_error", f"{symbol}: {type(exc).__name__}: {exc}") from None
        return plain_data(result, f"{symbol} result")

    def _dispatch(self, state: _Run, symbol: str, kind: HostKind, args: list, kwargs: dict) -> Any:
        if state.cursor < len(state.journal):
            recorded = state.journal[state.cursor]
            if (recorded.symbol, _canonical(recorded.args), _canonical(recorded.kwargs)) != (
                symbol,
                _canonical(args),
                _canonical(kwargs),
            ):
                raise StarlarkProgramError(
                    "replay_divergence",
                    f"call {state.cursor}: recorded {recorded.symbol}, body asked for {symbol}",
                )
            state.cursor += 1
            return recorded.result
        if kind == "action":
            if not state.allow_actions:
                raise StarlarkProgramError("action_in_read_only_entry", f"{state.entry} called {symbol}")
            state.pending = HostRequest(symbol, args, kwargs)
            raise _Abort("suspend")
        try:
            result = state.queries[symbol](*args, **kwargs)
        except StarlarkProgramError:
            raise  # a deliberate refusal (host_refusal) keeps its code
        except Exception as exc:  # the host function's own failure, reported by name
            raise StarlarkProgramError("host_error", f"{symbol}: {type(exc).__name__}: {exc}") from None
        result = plain_data(result, f"{symbol} result")
        state.journal.append(HostCall(symbol, args, kwargs, result))
        state.cursor += 1
        return result


def load_program(
    source: str,
    *,
    filename: str,
    host: Mapping[str, HostKind],
    entrypoints: Collection[str] = ("plan", "execute"),
    limits: StarlarkLimits | None = None,
    pure: Mapping[str, Callable[..., Any]] | None = None,
    read_globals: Collection[str] = (),
) -> StarlarkProgram:
    """Parse, check and freeze one body; query and action calls during loading are refused.

    ``pure`` binds every host symbol of kind ``"pure"``; ``read_globals`` names
    top-level values (such as a declaration) copied out as plain data before
    the module freezes, available as ``program.globals``.
    """

    return StarlarkProgram(
        filename=filename,
        source=source,
        host=host,
        entrypoints=entrypoints,
        limits=limits or StarlarkLimits(),
        pure=pure,
        read_globals=read_globals,
    )


def _detail(exc: BaseException) -> str:
    return str(exc)[:_DETAIL_CHARS]


def _classify(exc: sl.StarlarkError) -> StarlarkProgramError:
    text = str(exc)
    if "call stack overflow" in text:
        return StarlarkProgramError("callstack_exceeded", _detail(exc))
    if re.search(r"^error: fail:", text, re.MULTILINE):
        return StarlarkProgramError("contract_fail", _detail(exc))
    if "serde::serialize" in text:
        return StarlarkProgramError("non_data_value", _detail(exc))
    return StarlarkProgramError("evaluation_error", _detail(exc))
