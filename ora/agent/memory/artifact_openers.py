"""One advertised spelling for Context Artifact recovery."""
from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

ARTIFACT_READ_NAMES = frozenset({"artifact.read", "artifact_read"})


def recoverable_artifact_open(reference: Mapping[str, Any]) -> dict[str, Any]:
    """Use the catalog's delivered decoder; accept legacy unannotated refs."""
    image = str(reference.get("media_type", "")).startswith("image/")
    opener = reference.get("open")
    name = opener.get("tool") if isinstance(opener, Mapping) else None
    if name is None:
        name = "observe" if image else "artifact.read"
    if not isinstance(name, str) or name not in ({"observe"} if image else ARTIFACT_READ_NAMES):
        raise ValueError("recoverable Artifact decoder does not match media type")
    return {"tool": name, "arguments": {"ref": reference["ref"]}}


def artifact_read_name(definitions: Sequence[Mapping[str, Any]]) -> str:
    """The spelling this surface would give the reader, had it composed it.

    The reader is appended after composition, so `canonical_tool_names_v1`
    never reached it and a surface spelling every other tool in snake_case
    still advertised one dotted name. Inferred from the delivered definitions
    rather than from a capability list, because what a policy is handed is what
    it has to spell back.
    """

    from ..tools import tool_names

    for item in definitions:
        name = str(item.get("name", ""))
        if name and name in tool_names.CANONICAL_TOOL_NAMES:
            return "artifact_read"
    return "artifact.read"


def public_artifact_ref(
    entry: Mapping[str, Any], *, reader_name: str,
    round_index: int | None = None, action_step: int | None = None,
) -> dict[str, Any]:
    reference = {
        key: copy.deepcopy(entry.get(key)) for key in (
            "schema", "ref", "sha256", "kind", "media_type", "bytes",
            "policy_scope",
        )
    } | {"round": round_index, "action_step": action_step}
    # Preserve the shape of legacy refs while carrying the active surface's
    # spelling through eager markers, pressure eviction and checkpoints.
    if reader_name != "artifact.read":
        reference["open"] = {
            "tool": "observe" if str(entry.get("media_type", "")).startswith("image/")
            else reader_name,
            "arguments": {"ref": reference["ref"]},
        }
    return reference
