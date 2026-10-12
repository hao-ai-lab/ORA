"""Where each default tool is served and how its proposal reaches the backend.

The tool half of `ora baselines --explain`, derived from code:

* a tool's handler is read from the tool runtime's own `invoke` dispatch table
  (located with `ast`) and resolved to the callable it names;
* a state-changing tool's proposal comes from a static walk of its handler
  that follows only the branches the resolved surface's capabilities select
  and records the action envelopes the walk can build;
* the Supervisor executor for each action schema is read from the branches of
  `episode_actions.dispatch_action`, the transport client from the environment
  factory, and the backend session class from the isolated worker.

Where the code admits no exact answer the record says so and gives the
generic route. Backend session modules are parsed, never imported. Nothing is
written or started; the only code run, besides imports of host-safe project
modules, is a project predicate whose inputs are the resolved surface and
constants.
"""

from __future__ import annotations

import ast
import functools
import importlib
import importlib.util
import inspect
import re
import textwrap
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping

from ora.agent.composition.tool_surface import ResolvedToolSurface
from ora.agent.tools import factory as tool_factory
from ora.agent.tools import tool_names
from ora.environment import factory as environment_factory
from ora.environment import worker as environment_worker
from ora.environment.session import DescribedEnvironmentSession

from . import episode_actions
from .episode_loop import EpisodeLoop

# Handlers reach their proposal through helpers; past this depth a walk is
# exploring unrelated utilities, not the submission path.
MAX_WALK_DEPTH = 8
# The Runtime action envelope (ora/contracts/action.py). A literal
# with these keys is a proposal, not a result or receipt that happens to carry
# "schema" and "action".
_ENVELOPE_KEYS = {"schema", "action", "arguments", "bound_observation_sha256"}
_PROJECT_PACKAGE = "ora."
_NEVER_IMPORTED = "ora.environment.backends."


class BaselineExplainError(RuntimeError):
    """The explanation cannot be derived, or the request is invalid."""


# -- small code-location helpers ------------------------------------------------

def _relative(path: str | Path, root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def dotted(value: Any) -> str:
    return f"{value.__module__}:{value.__qualname__}"


@functools.lru_cache(maxsize=None)
def _parsed(function: Callable[..., Any]) -> tuple[ast.AST, int]:
    """The function's own AST node and the file line its source starts on."""
    lines, start = inspect.getsourcelines(function)
    tree = ast.parse(textwrap.dedent("".join(lines)))
    return tree.body[0], start


def location(function: Callable[..., Any], root: Path) -> str:
    """`path:line` of the `def` line itself, past any decorators."""
    node, start = _parsed(function)
    return f"{_relative(inspect.getsourcefile(function), root)}:{start + node.lineno - 1}"


def callable_record(function: Callable[..., Any], root: Path) -> dict[str, str]:
    return {"callable": dotted(function), "location": location(function, root)}


def _resolve_import(node: ast.ImportFrom, package: str | None) -> str:
    return importlib.util.resolve_name("." * node.level + (node.module or ""), package)


def _module_file(module: str, root: Path) -> Path:
    """Locate a backend module by name without importing it."""
    return root / (module.replace(".", "/") + ".py")


# -- the tool runtime and its dispatch table ------------------------------------

def _literal_dict(function: Callable[..., Any], *, value_types: tuple[type, ...]) -> ast.Dict:
    """The first dict literal in ``function`` keyed by strings onto ``value_types``."""
    node, _start = _parsed(function)
    for child in ast.walk(node):
        if (isinstance(child, ast.Dict) and len(child.keys) > 1
                and all(isinstance(k, ast.Constant) and isinstance(k.value, str)
                        for k in child.keys)
                and all(isinstance(v, value_types) for v in child.values)):
            return child
    raise BaselineExplainError(f"no literal dispatch table in {dotted(function)}")


def _tool_runtime_class(surface: ResolvedToolSurface) -> type:
    table = _literal_dict(tool_factory.build_tool_runtime, value_types=(ast.Name,))
    for key, value in zip(table.keys, table.values):
        if key.value == surface.runtime_id:
            return tool_factory.build_tool_runtime.__globals__[value.id]
    raise BaselineExplainError(f"no tool runtime registered for {surface.runtime_id}")


def _dispatch_table(runtime_class: type) -> dict[str, ast.expr]:
    table = _literal_dict(runtime_class.invoke, value_types=(ast.Attribute, ast.Lambda))
    return {key.value: value for key, value in zip(table.keys, table.values)}


def _handler(entry: ast.expr, runtime_class: type) -> tuple[Callable[..., Any], list, str | None]:
    """Resolve one dispatch entry: ``self._x``, or a lambda forwarding to one call.

    Also returns the handler's name for the runtime instance (``self`` for a
    method, the first parameter of a module function the lambda passes it to).
    """
    bound: list = []
    target, arguments = entry, []
    if isinstance(entry, ast.Lambda):
        if not isinstance(entry.body, ast.Call):
            raise BaselineExplainError("dispatch lambda does not forward to one call")
        target, arguments = entry.body.func, entry.body.args
        bound = [arg.value for arg in arguments if isinstance(arg, ast.Constant)]
    if not (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)):
        raise BaselineExplainError("dispatch entry is not a method or module function")
    owner = target.value.id
    holder = (runtime_class if owner == "self"
              else runtime_class.invoke.__globals__[owner])
    function = inspect.unwrap(getattr(holder, target.attr))
    if owner == "self":
        return function, bound, "self"
    passes_runtime = bool(arguments) and isinstance(arguments[0], ast.Name) and arguments[0].id == "self"
    parameters = list(inspect.signature(function).parameters)
    return function, bound, parameters[0] if passes_runtime and parameters else None


# -- static proposal scan --------------------------------------------------------

class _Context:
    """Name resolution for one function: its local imports, then its globals."""

    def __init__(self, function: Callable[..., Any], node: ast.AST, runtime_name: str | None):
        self.function, self.runtime_name = function, runtime_name
        # Function-local ``from x import y``, resolved only when a name is used.
        self.imports: dict[str, tuple[str, str]] = {}
        # Locals assigned a surface test (``semantic_v3 = self.surface.supports(...)``).
        self.truths: dict[str, bool] = {}
        package = function.__module__.rpartition(".")[0] or None
        for child in ast.walk(node):
            if isinstance(child, ast.ImportFrom):
                base = _resolve_import(child, package)
                for alias in child.names:
                    self.imports[alias.asname or alias.name] = (base, alias.name)

    def lookup(self, name: str) -> Any:
        if name in self.imports:
            return _import_member(*self.imports[name])
        return self.function.__globals__.get(name)


def _import_member(base: str, name: str) -> Any:
    """Import a project member a handler imports locally; never a backend module."""
    if not base.startswith(_PROJECT_PACKAGE) or base.startswith(_NEVER_IMPORTED):
        return None
    try:
        module = importlib.import_module(base)
        if hasattr(module, name):
            return getattr(module, name)
        return importlib.import_module(f"{base}.{name}")
    except Exception:  # noqa: BLE001 - an unimportable member resolves nothing
        return None


def _is_surface_attribute(node: ast.AST, attributes: tuple[str, ...]) -> bool:
    return (isinstance(node, ast.Attribute) and node.attr in attributes
            and isinstance(node.value, ast.Attribute) and node.value.attr == "surface"
            and isinstance(node.value.value, ast.Name))


def _is_surface_expression(node: ast.AST) -> bool:
    """``runtime.surface`` or ``getattr(runtime, "surface", ...)``."""
    if isinstance(node, ast.Attribute):
        return node.attr == "surface" and isinstance(node.value, ast.Name)
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "getattr" and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant) and node.args[1].value == "surface")


class _ProposalScan:
    """Collect the action envelopes a handler can persist on one resolved surface."""

    def __init__(self, runtime_class: type, surface: ResolvedToolSurface):
        self.runtime_class, self.surface = runtime_class, surface
        self.found: set[tuple[str, str]] = set()
        self.unresolved = False
        self._seen: set[Any] = set()
        submit = getattr(runtime_class, "_atomic_submit", None)
        self.default_schema = (
            inspect.signature(submit).parameters["schema"].default if submit else None
        )

    def function(self, function: Callable[..., Any], runtime_name: str | None, depth: int) -> None:
        code = getattr(function, "__code__", None)
        if code is None or code in self._seen or depth > MAX_WALK_DEPTH:
            return
        self._seen.add(code)
        node, _start = _parsed(function)
        context = _Context(function, node, runtime_name)
        self._block(getattr(node, "body", []), context, depth)

    def truth(self, node: ast.AST, context: _Context) -> bool | None:
        """Evaluate a capability/profile test against this surface; None if not one."""
        if isinstance(node, ast.Name):
            return context.truths.get(node.id)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            inner = self.truth(node.operand, context)
            return None if inner is None else not inner
        if isinstance(node, ast.BoolOp):
            values = [self.truth(value, context) for value in node.values]
            decisive = isinstance(node.op, ast.Or)
            if decisive in values:
                return decisive
            return (not decisive) if all(value is not None for value in values) else None
        if (isinstance(node, ast.Call) and _is_surface_attribute(node.func, ("supports",))
                and len(node.args) == 1 and isinstance(node.args[0], ast.Constant)):
            return node.args[0].value in self.surface.capabilities
        if isinstance(node, ast.Call):
            return self._predicate(node, context)
        if not (isinstance(node, ast.Compare) and len(node.ops) == 1):
            return None
        left, operator, right = node.left, node.ops[0], node.comparators[0]
        if (isinstance(operator, (ast.In, ast.NotIn)) and isinstance(left, ast.Constant)
                and _is_surface_attribute(right, ("capabilities",))):
            return (left.value in self.surface.capabilities) == isinstance(operator, ast.In)
        if (isinstance(operator, (ast.Eq, ast.NotEq)) and isinstance(right, ast.Constant)
                and _is_surface_attribute(left, ("profile_id", "runtime_id"))):
            equal = getattr(self.surface, left.attr) == right.value
            return equal == isinstance(operator, ast.Eq)
        return None

    def _predicate(self, node: ast.Call, context: _Context) -> bool | None:
        """Call a project predicate whose only inputs are this surface and constants."""
        target, runtime_name = self._target(node, context)
        values = [*node.args, *(item.value for item in node.keywords)]
        # Only a module function: a runtime method would need the instance.
        if (target is None or runtime_name == "self" or not values
                or any(item.arg is None for item in node.keywords)):
            return None
        if not all(isinstance(item, ast.Constant) or _is_surface_expression(item)
                   for item in values):
            return None

        def value(item: ast.expr) -> Any:
            return item.value if isinstance(item, ast.Constant) else self.surface

        try:
            result = target(*map(value, node.args),
                            **{item.arg: value(item.value) for item in node.keywords})
        except Exception:  # noqa: BLE001 - a predicate that cannot run decides nothing
            return None
        return result if isinstance(result, bool) else None

    def _block(self, statements: list[ast.stmt], context: _Context, depth: int) -> bool:
        """Walk statements in order; True once every path through them has left."""
        for statement in statements:
            if isinstance(statement, ast.If):
                verdict = self.truth(statement.test, context)
                if verdict is None:
                    self._expression(statement.test, context, depth)
                    ends_body = self._block(statement.body, context, depth)
                    ends = self._block(statement.orelse, context, depth) and ends_body
                else:
                    ends = self._block(statement.body if verdict else statement.orelse,
                                       context, depth)
                if ends:
                    return True
                continue
            nested = [getattr(statement, field, None) for field in ("body", "orelse", "finalbody")]
            if any(isinstance(block, list) for block in nested):
                for field, child in ast.iter_fields(statement):
                    if field not in {"body", "orelse", "finalbody", "handlers"}:
                        self._expression(child, context, depth)
                for block in nested:
                    if isinstance(block, list):
                        self._block(block, context, depth)
                for handler in getattr(statement, "handlers", []):
                    self._block(handler.body, context, depth)
                continue
            self._expression(statement, context, depth)
            if (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)):
                verdict = self.truth(statement.value, context)
                if verdict is not None:
                    context.truths[statement.targets[0].id] = verdict
            if isinstance(statement, (ast.Return, ast.Raise)):
                return True
        return False

    def _expression(self, node: Any, context: _Context, depth: int) -> None:
        if isinstance(node, list):
            for item in node:
                self._expression(item, context, depth)
            return
        if not isinstance(node, ast.AST):
            return
        if isinstance(node, ast.IfExp):
            verdict = self.truth(node.test, context)
            branches = ([node.test, node.body, node.orelse] if verdict is None
                        else [node.body if verdict else node.orelse])
            self._expression(branches, context, depth)
            return
        if isinstance(node, ast.Call):
            self._call(node, context, depth)
        elif isinstance(node, ast.Dict):
            self._envelope(node, context)
        for child in ast.iter_child_nodes(node):
            self._expression(child, context, depth)

    def _strings(self, node: ast.AST, context: _Context) -> set[str]:
        if isinstance(node, ast.IfExp):
            verdict = self.truth(node.test, context)
            if verdict is not None:
                return self._strings(node.body if verdict else node.orelse, context)
            return self._strings(node.body, context) | self._strings(node.orelse, context)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return {node.value}
        if isinstance(node, ast.Name) and isinstance(context.lookup(node.id), str):
            return {context.lookup(node.id)}
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            owner = context.lookup(node.value.id)
            if isinstance(owner, ModuleType) and isinstance(getattr(owner, node.attr, None), str):
                return {getattr(owner, node.attr)}
        return set()

    def _record(self, action: ast.AST | None, schema: ast.AST | None, context: _Context) -> None:
        if not (isinstance(action, ast.Constant) and isinstance(action.value, str)):
            self.unresolved = True
            return
        schemas = (self._strings(schema, context) if schema is not None
                   else {self.default_schema} if self.default_schema else set())
        if not schemas:
            self.unresolved = True
        for value in schemas:
            self.found.add((action.value, value))

    def _envelope(self, node: ast.Dict, context: _Context) -> None:
        keys = {key.value: value for key, value in zip(node.keys, node.values)
                if isinstance(key, ast.Constant) and isinstance(key.value, str)}
        if _ENVELOPE_KEYS <= set(keys) and isinstance(keys["action"], ast.Constant):
            self._record(keys["action"], keys["schema"], context)

    def _call(self, node: ast.Call, context: _Context, depth: int) -> None:
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "_atomic_submit":
            keywords = {item.arg: item.value for item in node.keywords if item.arg}
            self._record(keywords.get("action_name"), keywords.get("schema"), context)
        target, runtime_name = self._target(node, context)
        if target is not None:
            self.function(target, runtime_name, depth + 1)

    def _target(self, node: ast.Call, context: _Context) -> tuple[Any, str | None]:
        func = node.func
        passes_runtime = (context.runtime_name is not None and bool(node.args)
                          and isinstance(node.args[0], ast.Name)
                          and node.args[0].id == context.runtime_name)
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            if func.value.id == context.runtime_name:
                method = getattr(self.runtime_class, func.attr, None)
                method = inspect.unwrap(method) if callable(method) else None
                return (method, "self") if _project_function(method) else (None, None)
            owner = context.lookup(func.value.id)
            if isinstance(owner, ModuleType):
                return _module_target(getattr(owner, func.attr, None), passes_runtime)
        if isinstance(func, ast.Name):
            return _module_target(context.lookup(func.id), passes_runtime)
        return None, None


def _project_function(value: Any) -> bool:
    return (inspect.isfunction(value)
            and str(getattr(value, "__module__", "")).startswith(_PROJECT_PACKAGE)
            and not str(value.__module__).startswith(_NEVER_IMPORTED))


def _module_target(value: Any, passes_runtime: bool) -> tuple[Any, str | None]:
    value = inspect.unwrap(value) if callable(value) else value
    if not _project_function(value) or not value.__module__.startswith("ora.agent."):
        return None, None
    parameters = list(inspect.signature(value).parameters)
    return value, (parameters[0] if passes_runtime and parameters else None)


# -- the Supervisor and environment route ---------------------------------------

def _dispatch_branches(root: Path) -> list[dict[str, Any]]:
    """Read `dispatch_action`'s schema branches: which executor runs which proposal."""
    function = episode_actions.dispatch_action
    namespace = function.__globals__
    node, _start = _parsed(function)
    statement = next((item for item in node.body if isinstance(item, ast.If)), None)
    branches: list[dict[str, Any]] = []
    while isinstance(statement, ast.If):
        schemas: set[str] = set()
        test = statement.test
        for child in ast.walk(test.comparators[0] if isinstance(test, ast.Compare) else test):
            if isinstance(child, ast.Name) and isinstance(namespace.get(child.id), str):
                schemas.add(namespace[child.id])
        if not schemas:
            raise BaselineExplainError("dispatch_action branch names no action schema")
        branches.append({"schemas": sorted(schemas),
                         **_first_call(statement.body, namespace, root)})
        orelse = statement.orelse
        statement = orelse[0] if len(orelse) == 1 and isinstance(orelse[0], ast.If) else None
        if statement is None and orelse:
            branches.append({"schemas": None, **_first_call(orelse, namespace, root)})
    if not branches or branches[-1]["schemas"] is not None:
        raise BaselineExplainError("dispatch_action has no default executor branch")
    return branches


def _first_call(body: list[ast.stmt], namespace: Mapping[str, Any], root: Path) -> dict[str, str]:
    for statement in body:
        for child in ast.walk(statement):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                value = namespace.get(child.func.id)
                if inspect.isfunction(value):
                    record = callable_record(value, root)
                    return {"executor": record["callable"], "location": record["location"]}
    raise BaselineExplainError("dispatch_action branch calls no executor")


def _executor_for(schema: str, branches: list[dict[str, Any]]) -> str:
    return next(branch["executor"] for branch in branches
                if branch["schemas"] is None or schema in branch["schemas"])


def _builder_facts(runtime: Mapping[str, Any]) -> tuple[type, str]:
    """The transport client class and worker backend id the factory builds."""
    key = (runtime["kind"], runtime.get("transport", "in_process_direct"))
    builder = environment_factory._BUILDERS.get(key)
    if builder is None:
        raise BaselineExplainError(f"no environment builder for {key}")
    node, _start = _parsed(builder)
    client, backend = None, None
    for child in ast.walk(node):
        if (isinstance(child, ast.Return) and isinstance(child.value, ast.Call)
                and isinstance(child.value.func, ast.Name)):
            client = builder.__globals__.get(child.value.func.id)
            # A builder that only delegates (RoboDojo's returns
            # build_robodojo_worker_session(...)) is read through its delegate.
            if inspect.isfunction(client):
                node, _start = _parsed(client)
                client = None
                for inner in ast.walk(node):
                    if (isinstance(inner, ast.Return) and isinstance(inner.value, ast.Call)
                            and isinstance(inner.value.func, ast.Name)):
                        client = builder.__globals__.get(inner.value.func.id)
                break
    for child in ast.walk(node):
        if isinstance(child, ast.Dict):
            for item, value in zip(child.keys, child.values):
                if (isinstance(item, ast.Constant) and item.value == "backend"
                        and isinstance(value, ast.Constant)):
                    backend = value.value
    if not isinstance(client, type) or backend is None:
        raise BaselineExplainError(f"environment builder {dotted(builder)} is not static")
    return client, backend


def _backend_session(backend: str, root: Path) -> dict[str, Any]:
    """Find the worker's session class for ``backend`` by parsing, never importing."""
    function = environment_worker._build_session
    node, _start = _parsed(function)
    package = function.__module__.rpartition(".")[0]
    for child in ast.walk(node):
        test = getattr(child, "test", None)
        if not (isinstance(child, ast.If) and isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name) and test.left.id == "backend"
                and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value == backend):
            continue
        for statement in child.body:
            if isinstance(statement, ast.ImportFrom):
                module = _resolve_import(statement, package)
                name = statement.names[0].name
                path = _module_file(module, root)
                tree = ast.parse(path.read_text(encoding="utf-8"))
                cls = next(item for item in tree.body
                           if isinstance(item, ast.ClassDef) and item.name == name)
                execute = next((item for item in cls.body if isinstance(item, ast.FunctionDef)
                                and item.name == "execute"), None)
                return {
                    "class": f"{module}:{name}",
                    "location": f"{_relative(path, root)}:{cls.lineno}",
                    "execute": (f"{_relative(path, root)}:{execute.lineno}" if execute
                                else "inherited, resolved in the isolated worker"),
                }
    raise BaselineExplainError(f"environment worker builds no session for {backend!r}")


def _proposal_route(runtime: Mapping[str, Any], root: Path) -> dict[str, Any]:
    client, backend = _builder_facts(runtime)
    return {
        "persisted_to": "control/submitted-action.json",
        "supervisor": [
            callable_record(EpisodeLoop._execute_proposal, root),
            callable_record(EpisodeLoop._accept_submitted_action, root),
            callable_record(episode_actions.dispatch_action, root),
        ],
        "dispatch_branches": _dispatch_branches(root),
        "environment": [
            callable_record(DescribedEnvironmentSession.execute, root),
            callable_record(client.execute, root),
        ],
        "transport": runtime.get("transport"),
        "worker_backend": backend,
        "backend_session": _backend_session(backend, root),
    }


def _summary(description: str) -> str:
    """The model-visible description's opening statement, before its help markers."""
    head = description.split(" Use when:", 1)[0].strip()
    match = re.match(r"(.+?[.!?])(\s|$)", head)
    return match.group(1) if match else head


def _tool(definition: Mapping[str, Any], table: Mapping[str, ast.expr], runtime_class: type,
          surface: ResolvedToolSurface, route: Mapping[str, Any], codec: str,
          root: Path) -> dict[str, Any]:
    name = definition["name"]
    effect = _effect(name, surface)
    entry = table.get(tool_names.canonical(name))
    if entry is None:
        raise BaselineExplainError(f"tool {name!r} has no entry in the dispatch table")
    function, bound, runtime_name = _handler(entry, runtime_class)
    record: dict[str, Any] = {
        "name": name, "effect": effect, "summary": _summary(str(definition["description"])),
        "proposal": None,
        "handler": {**callable_record(function, root), "bound_arguments": bound},
    }
    if effect != "state_changing_proposal":
        return record
    scan = _ProposalScan(runtime_class, surface)
    scan.function(function, runtime_name, 0)
    tail = f"EnvironmentSession.execute on {route['backend_session']['class'].split(':')[1]}"
    actions = [{"action": action, "schema": schema,
                "executor": _executor_for(schema, route["dispatch_branches"])}
               for action, schema in sorted(scan.found)]
    executors = " | ".join(sorted({item["executor"].split(":")[1] for item in actions}))
    generic = (f"submitted as a {codec} proposal (action type not resolved statically) -> "
               f"Supervisor dispatch_action -> {tail}")
    if actions and not scan.unresolved:
        record["proposal"] = {"resolution": "static", "actions": actions,
                              "route": f"{executors} -> {tail}"}
    else:
        # Whatever the walk did resolve is still shown; the rest takes the
        # generic route rather than a guessed one.
        record["proposal"] = {"resolution": "partial" if actions else "generic",
                              "actions": actions, "route": generic}
    return record


def _effect(name: str, surface: ResolvedToolSurface) -> str:
    if name in surface.lifecycle_tools:
        return "lifecycle"
    if name in surface.mutation_tools:
        return "state_changing_proposal"
    if name in surface.observation_tools:
        return "read_only_observation"
    return "read_only"


def explain_tools(surface: ResolvedToolSurface, runtime: Mapping[str, Any], codec: str,
                  root: Path) -> dict[str, Any]:
    """The tool runtime, the shared proposal route and one record per surface tool."""
    runtime_class = _tool_runtime_class(surface)
    table = _dispatch_table(runtime_class)
    route = _proposal_route(runtime, root)
    return {
        "tool_runtime": {"class": dotted(runtime_class),
                         "location": location(runtime_class, root),
                         "dispatch": location(runtime_class.invoke, root)},
        "proposal_route": route,
        "tools": [_tool(item, table, runtime_class, surface, route, codec, root)
                  for item in surface.definitions],
    }
