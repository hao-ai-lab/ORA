"""The published ``compute`` grammar, generated from the evaluator's allow-list.

Under ``compute_grammar_from_evaluator_v1`` the ``compute`` description gains
one generated section, appended after every hand-written layer and delimited by
its own first sentence ("Evaluator grammar (generated from the evaluator's own
allow-list)."). Its text is ``agent.tools.compute_grammar.grammar_text()``: the
statements, operators and forms the evaluator admits, the constructs it refuses
by name (augmented assignment and subscript or attribute assignment among
them), the reserved names, the attributes of a produced value and the exponent
rule. The evaluator's refusals read the same tables, so the description and
the refusal cannot disagree. Hand-written prose stays in the component files;
this section is never edited by hand.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

CAPABILITY = "compute_grammar_from_evaluator_v1"
REQUIRES = "policy_restricted_geometry_compute_v1"


def bind_compute_grammar(
    definitions: list[dict[str, Any]], capabilities: Sequence[str],
) -> dict[str, str]:
    """Append the generated grammar section to the ``compute`` description, in place."""

    from ..tools.compute_grammar import grammar_text
    from .tool_surface import ToolSurfaceError

    if REQUIRES not in capabilities:
        raise ToolSurfaceError(f"{CAPABILITY} requires {REQUIRES}")
    entries = [item for item in definitions if item.get("name") == "compute"]
    if len(entries) != 1:
        raise ToolSurfaceError(f"{CAPABILITY} requires exactly one declared compute tool")
    section = grammar_text()
    description = str(entries[0].get("description", "")).rstrip()
    if section not in description:
        entries[0]["description"] = f"{description}\n{section}"
    digest = hashlib.sha256()
    for path in (Path(__file__), Path(__file__).parents[1] / "tools" / "compute_grammar.py"):
        digest.update(path.read_bytes())
    return {
        "slot": f"derived:{CAPABILITY}",
        "ref": "ora/agent/composition/compute_grammar_schema.py",
        "sha256": digest.hexdigest(),
    }


__all__ = ["CAPABILITY", "bind_compute_grammar"]
