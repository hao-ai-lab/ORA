"""Generate a surface's description and schema layers from exact sentence edits.

A live surface corrects its help with its own layer, generated from that
surface's composed text by exact sentence edits; the shared files the old
sentences live in are hash-bound by frozen surfaces and keep their bytes
(docs/guides/tool-surface-authoring.md, "Help that says what the handler does").
This module is that generator. A wording review lists the edits, each with its
Chinese reference rendering; the layers it names are the edits applied to a
surface's text as composed *before* capability binders run, because a binder
appends generated text (compute grammar, controller feedback, the abort class
list, "Null means omitted.") after every layer and would otherwise appear twice.

Which text the edits apply to is the review's ``edits_apply_to``:
``base_profile`` (the default) composes the base surface; ``successor_before_layers``
composes the registered successor with every layer except the review's own two
slots. A generated layer replaces whole descriptions, so on a successor that
appends help of its own (ports, IK selection, wait,
the inter-arm screen) the layer must be generated from that text and ordered last, or
it would erase the appends.

    uv run --locked python -m ora.agent.composition.description_edits REVIEW [--check]

writes, or with ``--check`` verifies, the two layer files beside REVIEW.
"""

from __future__ import annotations

import argparse
import copy
import functools
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import tool_surface
from .canonical_names import canonicalise_prose
from .schema_validation import ToolSurfaceError
from .tool_surface_resolution import (
    apply_canonical_tool_names, exclude_and_order_tools, load_ordered_components,
    load_tool_definitions,
)

REVIEW_SCHEMA = "ora.description-edit-review.v1"
DESCRIPTION = "description"


APPLY_TO_BASE = "base_profile"
APPLY_TO_SUCCESSOR = "successor_before_layers"


def composed_before_binders(
    profile_id: str, *, omit_slots: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """The registered surface's definitions after its layers, before any binder.

    ``omit_slots`` drops those description and schema slots from the
    composition, which is how a successor's own generated layers are left out
    of the text they are generated from.
    """

    manifest_path, allowed_root, manifest, _revision = tool_surface._load_manifest(profile_id)
    if omit_slots:
        manifest = copy.deepcopy(manifest)
        for order in ("description_order", "schema_order"):
            manifest[order] = [slot for slot in manifest[order] if slot not in set(omit_slots)]
        manifest["components"] = {
            slot: spec for slot, spec in manifest["components"].items() if slot not in set(omit_slots)
        }
    load = functools.partial(
        tool_surface._component, profile_root=manifest_path.parent, allowed_root=allowed_root,
    )
    base, _descriptors = load_tool_definitions(manifest, load)
    canonical_names = apply_canonical_tool_names(manifest, base)
    base, excluded = exclude_and_order_tools(manifest, base)
    description_layers, schema_layers, _results, _components = load_ordered_components(
        manifest, None, load, canonical_names=canonical_names,
    )
    definitions = tool_surface.compose_tool_definitions(
        base, description_layers, schema_replacements=schema_layers, excluded=set(excluded),
    )
    return [canonicalise_prose(item) for item in definitions] if canonical_names else definitions


def _pointer_parent(schema: dict[str, Any], pointer: str) -> tuple[dict[str, Any], str]:
    """The object holding the string a JSON pointer names inside an input schema."""

    parts = [part.replace("~1", "/").replace("~0", "~") for part in pointer.split("/")[1:]]
    if not parts:
        raise ToolSurfaceError(f"edit pointer {pointer!r} names no field")
    node: Any = schema
    for part in parts[:-1]:
        node = node[int(part)] if isinstance(node, list) else node[part]
    if not isinstance(node, dict) or not isinstance(node.get(parts[-1]), str):
        raise ToolSurfaceError(f"edit pointer {pointer!r} does not name a string")
    return node, parts[-1]


def _replace_once(text: str, before: str, after: str, *, where: str) -> str:
    count = text.count(before)
    if count != 1 or before == after:
        raise ToolSurfaceError(
            f"{where}: the edited sentence must occur exactly once and change "
            f"(found {count} times): {before[:80]!r}"
        )
    return text.replace(before, after)


def apply_sentence_edits(
    definitions: Sequence[Mapping[str, Any]], edits: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``({"replace": {tool: text}}, {tool: input_schema})`` for the edited tools."""

    by_name = {str(item["name"]): copy.deepcopy(dict(item)) for item in definitions}
    described: dict[str, str] = {}
    schemas: dict[str, dict[str, Any]] = {}
    for index, edit in enumerate(edits):
        tool, field = str(edit["tool"]), str(edit["field"])
        if tool not in by_name:
            raise ToolSurfaceError(f"edit {index} names undeclared tool {tool!r}")
        where = f"edit {index} ({tool} {field})"
        if field == DESCRIPTION:
            current = described.get(tool, by_name[tool]["description"])
            described[tool] = _replace_once(current, edit["before"], edit["after"], where=where)
            continue
        schema = schemas.setdefault(tool, copy.deepcopy(by_name[tool]["inputSchema"]))
        parent, key = _pointer_parent(schema, field)
        parent[key] = _replace_once(parent[key], edit["before"], edit["after"], where=where)
    return {"replace": dict(sorted(described.items()))}, dict(sorted(schemas.items()))


def load_review(path: Path) -> dict[str, Any]:
    review = json.loads(path.read_text(encoding="utf-8"))
    if review.get("schema") != REVIEW_SCHEMA:
        raise ToolSurfaceError(f"{path} is not a {REVIEW_SCHEMA} record")
    return review


def edit_base_text(review: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The composed text a review's edits apply to (see ``edits_apply_to``)."""

    target = review.get("edits_apply_to", APPLY_TO_BASE)
    if target == APPLY_TO_BASE:
        return composed_before_binders(str(review["base_profile"]))
    if target == APPLY_TO_SUCCESSOR:
        return composed_before_binders(
            str(review["successor_profile"]), omit_slots=tuple(review["slots"].values()),
        )
    raise ToolSurfaceError(f"unknown edits_apply_to {target!r}")


def generated_layers(review: Mapping[str, Any]) -> dict[str, Any]:
    """The layer files a review names, keyed by file name, from the text its edits apply to."""

    description, schemas = apply_sentence_edits(edit_base_text(review), review["edits"])
    layers = review["layers"]
    return {str(layers["description"]): description, str(layers["schemas"]): schemas}


def render(value: Any) -> str:
    """The committed byte form of a generated layer."""

    return json.dumps(value, indent=2, ensure_ascii=False) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("review", type=Path)
    parser.add_argument("--check", action="store_true", help="verify instead of writing")
    arguments = parser.parse_args(argv)
    review = load_review(arguments.review)
    stale = []
    for name, value in generated_layers(review).items():
        target = arguments.review.parent / name
        if arguments.check:
            if not target.is_file() or target.read_text(encoding="utf-8") != render(value):
                stale.append(name)
        else:
            target.write_text(render(value), encoding="utf-8")
    if stale:
        print("stale generated layers: " + ", ".join(stale))
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - authoring entry point
    raise SystemExit(main())
