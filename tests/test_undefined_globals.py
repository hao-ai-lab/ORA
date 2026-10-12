"""Every global name a function reads must be bound by its module.

A refactor or merge can leave a call to a deleted helper, or move code out of
the function that defined a local it reads. Python only fails when that line
runs, which an error path may never do under test.
"""

from __future__ import annotations

import builtins
import symtable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_IMPLICIT = set(dir(builtins)) | {
    "__file__", "__name__", "__doc__", "__spec__", "__package__", "__builtins__",
    "__loader__", "__path__", "__annotations__", "__dict__", "__class__",
}


def _unbound_reads(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    if "import *" in source:
        return []
    module = symtable.symtable(source, str(path), "exec")
    bound = {
        symbol.get_name() for symbol in module.get_symbols()
        if symbol.is_assigned() or symbol.is_imported() or symbol.is_namespace()
    }
    found: list[str] = []

    def visit(table: symtable.SymbolTable, scope: str) -> None:
        for child in table.get_children():
            name = f"{scope}.{child.get_name()}" if scope else child.get_name()
            for symbol in child.get_symbols():
                if (
                    symbol.is_referenced() and symbol.is_global()
                    and symbol.get_name() not in bound
                    and symbol.get_name() not in _IMPLICIT
                ):
                    found.append(f"{path.relative_to(ROOT)}:{name}:{symbol.get_name()}")
            visit(child, name)

    visit(module, "")
    return found


def test_functions_read_only_globals_their_module_binds() -> None:
    unbound = [
        item
        for path in sorted((ROOT / "ora").rglob("*.py"))
        for item in _unbound_reads(path)
    ]
    assert unbound == []
