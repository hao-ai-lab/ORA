"""The tool runtime keeps no state it only ever writes.

`_latest_motion_probe_results` was assigned in four places and read in none.
It cost a line in `__init__`, a line in every round reset and two lines in the
tools that populated it, and it answered no question, because the only reader
it ever had was a test asserting that the reset cleared it. State like that is
worse than dead code: a reader who finds it assumes something consumes it and
preserves it through every refactor.

This guard is deliberately narrow. It asks only about attributes assigned
somewhere under `ora/agent/tools/`, and it counts a read anywhere in
`ora/` -- an attribute written by one tool module and consumed by
another is live, which is why `_active_mark_point_batch` (written in
`runtime/core.py`, read in `mark_point.py` and `mark_point_overlay.py`) is not a
finding. A per-file version of this check called that attribute dead.

Tests are not counted as readers on purpose. A test that stores a value and
asserts it was cleared proves only that the field exists; if production never
reads the field, the test is describing its own fixture.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "ora"
SCOPE = "ora/agent/tools/"


def _attribute_use() -> tuple[dict[str, set[str]], set[str]]:
    """Map every private attribute name to where it is stored, and what is read."""

    stores: dict[str, set[str]] = {}
    loads: set[str] = set()
    for path in sorted(PACKAGE.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - a syntax error is its own failure
            continue
        relative = path.relative_to(ROOT).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute) or not node.attr.startswith("_"):
                continue
            if isinstance(node.ctx, ast.Store):
                stores.setdefault(node.attr, set()).add(relative)
            else:
                # Load and Del both prove something depends on the value being
                # there, so neither counts as a write.
                loads.add(node.attr)
    return stores, loads


def test_no_tool_runtime_attribute_is_only_ever_written() -> None:
    """An attribute nothing reads is removed, not preserved."""

    stores, loads = _attribute_use()
    write_only = {
        name: sorted(paths)
        for name, paths in stores.items()
        if name not in loads and any(path.startswith(SCOPE) for path in paths)
    }
    assert not write_only, (
        f"these attributes are assigned but never read: {write_only}. Delete the "
        "field and its assignments; if something was supposed to consume it, the "
        "missing reader is the bug."
    )
