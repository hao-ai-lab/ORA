"""One spelling per tool on a surface that opts in.

The map itself lives in ``agent/tools/tool_names.py`` and is not restated here
or in a manifest: a second copy of it is a copy that can disagree with the
first. A surface opts in with ``canonical_tool_names_v1``; frozen surfaces do
not, and keep the dotted names their sealed runs were shown.

The rewrite happens during composition rather than in the component files
because two of the layers a live profile binds are frozen catalogs it
hash-references, so their bytes cannot change. Rewriting only the editable ones
would leave the rule with an exception on whichever backend binds a frozen
layer -- and it would move component hashes that nothing else about this change
touches.
"""
from __future__ import annotations

import re
from typing import Any, Mapping

CANONICAL_TOOL_NAMES_CAPABILITY = "canonical_tool_names_v1"


def canonical_tool_name(name: str) -> str:
    from ora.agent.tools.tool_names import canonical

    return canonical(name)


def canonicalise_component(value: Any, *, nested: bool) -> dict[str, Any]:
    """Rewrite a resolved component's tool-name keys to the canonical spelling.

    A description layer keys its tools one level down, under ``replace`` and its
    siblings; a schema or result layer keys them at the top. Both are rewritten
    here rather than in the files because two of the layers a live profile binds
    are frozen catalogs it hash-references, so their bytes cannot change -- and
    editing the editable ones instead would leave the rule with an exception
    that only shows up on whichever backend happens to bind a frozen layer.
    """

    if not isinstance(value, Mapping):
        return value
    if nested:
        return {
            str(operation): (
                {canonical_tool_name(str(name)): text for name, text in entries.items()}
                if isinstance(entries, Mapping) else entries
            )
            for operation, entries in value.items()
        }
    return {canonical_tool_name(str(name)): entry for name, entry in value.items()}


def canonicalise_prose(value: Any) -> Any:
    """Rewrite tool names inside model-visible text, not only in the keys.

    Renaming the keys alone leaves every ``Use when:`` near-miss clause pointing
    at a name the surface no longer advertises -- "this is not grasp.plan; call
    grasp.plan if ..." beside a tool called ``grasp_plan``. That is worse than
    the wire alias this rename removes: the alias at least disclosed its
    mapping. Measured on the two live surfaces before this ran: 30 references in
    descriptions and 15 in input-schema descriptions.

    The names are matched whole, and every one of them is a tool this surface
    declares, so there is no prose in which the string means something else.
    """

    from ora.agent.tools.tool_names import DOTTED_TOOL_NAMES

    if isinstance(value, str):
        for dotted, canonical_name in DOTTED_TOOL_NAMES.items():
            value = re.sub(rf"(?<![\w.]){re.escape(dotted)}(?![\w])", canonical_name, value)
        return value
    if isinstance(value, Mapping):
        return {key: canonicalise_prose(item) for key, item in value.items()}
    if isinstance(value, list):
        return [canonicalise_prose(item) for item in value]
    return value
