"""Static project import closure, resolved from source without importing it.

A hand-written source manifest drifts when code moves into a new module: the
RoboDojo worker kernel listed ``sparse_motion.py`` but not the ``recovery.py``
and ``wait.py`` it dispatches into, so executor edits left every qualification
"current". This module derives the closure from the import statements
themselves, so a manifest can be held to it by a test or by a checker that
reads historical source through ``git show``.

Resolution rules, deliberately narrow and fail-closed:

* every ``import``/``from`` statement counts, including ones inside functions,
  conditionals, ``try`` blocks and ``TYPE_CHECKING`` guards -- a lazy import
  still executes, and an unexecuted one only over-binds;
* ``importlib.import_module("x.y")`` and ``__import__("x.y")`` with a constant
  name count as ``import x.y``. Any other argument is an error, because the
  module cannot be resolved statically: a relative constant, a variable, a
  computed string, or an f-string, except an f-string whose literal prefix
  already fixes a top-level package outside ``package`` (RoboDojo's
  ``f"task.RoboDojo.tasks.{task}"``);
* only names inside ``package`` (default ``ora``) are resolved;
  third-party and upstream-simulator imports are ignored;
* ``from P import name`` binds ``P/name.py`` when ``name`` is a submodule and
  ``P/__init__.py`` otherwise;
* a project import that names no source file is an error, never a skip,
  including an optional ``try: ... except ImportError`` import.

By default importing a submodule does not also bind every parent package's
``__init__``. Those run on import; ``implicit_parents=True`` adds them, and
``module_level_only=True`` keeps only imports that execute at import time, which
together give the chain Python actually executes when a module is imported.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Callable, Iterable, Iterator

#: Returns a project-relative file's bytes, or ``None`` when it is absent.
Reader = Callable[[str], "bytes | None"]
#: ``follow(importer, imported)`` decides whether the closure visits an edge's target.
Follow = Callable[[str, str], bool]

_DEFERRED = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)


class ImportClosureError(ValueError):
    """A project import or root names a module the source does not contain."""


def tree_reader(root: Path) -> Reader:
    """Read project-relative files from one working tree."""

    def read(relative: str) -> bytes | None:
        path = Path(root) / relative
        return path.read_bytes() if path.is_file() else None
    return read


class _Resolver:
    def __init__(self, read: Reader, package: str) -> None:
        self._read = read
        self._cache: dict[str, bytes | None] = {}
        self.package = package

    def source(self, relative: str) -> bytes | None:
        if relative not in self._cache:
            self._cache[relative] = self._read(relative)
        return self._cache[relative]

    def inside(self, dotted: str) -> bool:
        return dotted == self.package or dotted.startswith(self.package + ".")

    def module_file(self, dotted: str) -> str | None:
        base = dotted.replace(".", "/")
        return base + ".py" if self.source(base + ".py") is not None else None

    def package_file(self, dotted: str) -> str | None:
        init = dotted.replace(".", "/") + "/__init__.py"
        return init if self.source(init) is not None else None

    def require(self, dotted: str, importer: str) -> str:
        found = self.module_file(dotted) or self.package_file(dotted)
        if found is None:
            raise ImportClosureError(f"{importer} imports {dotted}, which has no source file")
        return found

    def from_import(self, dotted: str, names: Iterable[str], importer: str) -> set[str]:
        module = self.module_file(dotted)
        if module is not None:
            return {module}
        found: set[str] = set()
        for name in names:
            child = f"{dotted}.{name}"
            submodule = name != "*" and (self.module_file(child) or self.package_file(child))
            found.add(submodule or self.require(dotted, importer))
        return found

    def parents(self, relative: str) -> set[str]:
        """Existing ``__init__`` files of every package above ``relative``."""

        parts = relative.split("/")[:-1]
        if relative.endswith("/__init__.py"):
            parts = parts[:-1]
        found = set()
        for depth in range(1, len(parts) + 1):
            dotted = ".".join(parts[:depth])
            init = self.package_file(dotted) if self.inside(dotted) else None
            if init is not None:
                found.add(init)
        return found


def _own_package(relative: str) -> list[str]:
    return relative[: -len(".py")].split("/")[:-1]


def _nodes(tree: ast.AST, module_level_only: bool) -> Iterator[ast.AST]:
    if not module_level_only:
        yield from ast.walk(tree)
        return
    pending = [tree]
    while pending:
        node = pending.pop()
        yield node
        pending.extend(child for child in ast.iter_child_nodes(node)
                       if not isinstance(child, _DEFERRED))


def _dynamic_name(node: ast.Call, relative: str, resolver: _Resolver) -> str | None:
    """The constant module name of an ``import_module``/``__import__`` call."""

    func = node.func
    dynamic = ((isinstance(func, ast.Name) and func.id in {"__import__", "import_module"})
               or (isinstance(func, ast.Attribute) and func.attr == "import_module"
                   and isinstance(func.value, ast.Name) and func.value.id == "importlib"))
    if not dynamic:
        return None
    argument = node.args[0] if node.args else None
    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
        if argument.value.startswith("."):
            raise ImportClosureError(f"{relative} has a relative dynamic import {argument.value!r}")
        return argument.value
    if isinstance(argument, ast.JoinedStr) and argument.values:
        head = argument.values[0]
        prefix = head.value if isinstance(head, ast.Constant) and isinstance(head.value, str) else ""
        top = prefix.split(".")[0]
        if "." in prefix and top and top != resolver.package.split(".")[0]:
            return None
    raise ImportClosureError(
        f"{relative} has a dynamic import whose module cannot be resolved statically"
    )


def module_imports(relative: str, resolver: _Resolver, *,
                   module_level_only: bool = False) -> tuple[str, ...]:
    """Project-relative files named by one module's import statements."""

    source = resolver.source(relative)
    if source is None:
        raise ImportClosureError(f"import closure root {relative} has no source file")
    found: set[str] = set()
    for node in _nodes(ast.parse(source, filename=relative), module_level_only):
        if isinstance(node, ast.Call):
            dotted = _dynamic_name(node, relative, resolver)
            if dotted is not None and resolver.inside(dotted):
                found.add(resolver.require(dotted, relative))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if resolver.inside(alias.name):
                    found.add(resolver.require(alias.name, relative))
        elif isinstance(node, ast.ImportFrom):
            found |= _from_import(node, relative, resolver)
    return tuple(sorted(found))


def _from_import(node: ast.ImportFrom, relative: str, resolver: _Resolver) -> set[str]:
    names = [alias.name for alias in node.names]
    if not node.level:
        dotted = node.module or ""
        return resolver.from_import(dotted, names, relative) if resolver.inside(dotted) else set()
    package = _own_package(relative)
    if node.level - 1 > len(package):
        raise ImportClosureError(f"{relative} has a relative import above the project")
    base = package[: len(package) - (node.level - 1)]
    if node.module is not None:
        dotted = ".".join([*base, node.module])
        return resolver.from_import(dotted, names, relative) if resolver.inside(dotted) else set()
    # ``from . import x`` names a package, never a module file.
    dotted = ".".join(base)
    if not resolver.inside(dotted):
        return set()
    return {resolver.module_file(f"{dotted}.{name}") or resolver.package_file(f"{dotted}.{name}")
            or resolver.package_file(dotted) or resolver.require(f"{dotted}.{name}", relative)
            for name in names}


def import_closure(
    roots: Iterable[str], *, read: Reader, package: str = "ora",
    follow: Follow | None = None, module_level_only: bool = False,
    implicit_parents: bool = False,
) -> dict[str, tuple[str, ...]]:
    """Map every visited module to all the project modules it imports.

    Values list every resolved import, followed or not; ``follow`` only
    decides which targets are visited next, so a caller can classify the
    edges it pruned. Roots are always visited. Keys are project-relative
    paths, values sorted tuples.
    """

    resolver = _Resolver(read, package)
    closure: dict[str, tuple[str, ...]] = {}
    pending = list(roots)
    if implicit_parents:
        pending.extend(parent for root in tuple(pending) for parent in resolver.parents(root))
    while pending:
        relative = pending.pop()
        if relative in closure:
            continue
        edges = set(module_imports(relative, resolver, module_level_only=module_level_only))
        if implicit_parents:
            edges |= {parent for target in tuple(edges) for parent in resolver.parents(target)}
        closure[relative] = tuple(sorted(edges))
        pending.extend(target for target in closure[relative]
                       if follow is None or follow(relative, target))
    return closure
