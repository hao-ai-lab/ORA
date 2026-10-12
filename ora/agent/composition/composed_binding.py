"""Composed tool binding: a ToolSurface serves tools from compiled composed-tool documents.

Split out of :mod:`.tool_surface` (which re-exports :func:`bind_composed_tools`) when workspace
tools and nested closures made the binding longer; the composed tool surface spec §20 describes
what it pins.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def bind_composed_tools(
    manifest: Mapping[str, Any], definitions: list[dict[str, Any]],
    results: dict[str, Mapping[str, Any]], capabilities: Sequence[str],
    overrides: Mapping[str, Any] | None = None,
) -> tuple[dict[str, dict[str, str]], list[dict[str, str]]]:
    """Serve the tools ``manifest["composed_tools"]`` names from their compiled documents.

    ``{"grasp_execute": "grasp_execute/c1"}`` replaces that tool's definition
    and result schema with the library entry's compiled contract for the
    surface's arm profile (``dual_arm`` when the surface declares
    ``dual_arm_sparse_targets_v1``). The name must already be a tool of this
    surface and must equal the library tool's name, so the public name, the
    compiled help and the dispatch path cannot diverge. Both library files are
    recorded as components by path and sha256. ``overrides`` (the
    ``composed:<tool>`` component overrides) bind another version under the
    same rules, or add a library tool this surface does not declare (a task
    tool): its compiled definition is appended, it becomes a mutation tool and
    its binding records ``scope: added``. The manifest binds declared tools only.

    An override may also name a tool outside the library by absolute path,
    ``/<root>/<tool>/<version>`` (a practice agent's workspace): the binding
    then records that ``root`` and the source sha256, and its descriptors the
    absolute paths. Every tool a bound tool reaches through ``tool`` stages is
    pinned under ``nested`` (directory and source sha256) and recorded as a
    component, so the plan's composition identity covers the whole closure.
    """

    from .tool_surface import COMPOSED_TOOLS_CAPABILITY, ToolSurfaceError, file_sha256

    raw = manifest.get("composed_tools")
    if overrides:
        raw = {**(dict(raw) if isinstance(raw, Mapping) else {}), **dict(overrides)}
    if raw is None:
        return {}, []
    if not isinstance(raw, Mapping) or not raw:
        raise ToolSurfaceError("composed_tools must map tool names to <tool>/<version>")
    if COMPOSED_TOOLS_CAPABILITY not in capabilities:
        raise ToolSurfaceError(f"composed_tools requires the {COMPOSED_TOOLS_CAPABILITY} capability")
    from ora.agent.tools.composed.library import LIBRARY_ROOT, LibraryError, load_tool, nested_closure

    arm_profile = "dual_arm" if "dual_arm_sparse_targets_v1" in capabilities else "single_arm"
    by_name = {item["name"]: index for index, item in enumerate(definitions)}
    bindings: dict[str, dict[str, str]] = {}
    descriptors: list[dict[str, str]] = []
    for name, ref in raw.items():
        added = name not in by_name
        if added and name not in (overrides or {}):
            raise ToolSurfaceError(f"composed_tools binds {name!r}, which this surface does not declare")
        external = isinstance(ref, str) and ref.startswith("/") and name in (overrides or {})
        if external:
            directory = Path(ref)
            tool_name, version = directory.parent.name, directory.name
        elif not isinstance(ref, str) or ref.count("/") != 1:
            raise ToolSurfaceError(f"composed_tools[{name!r}] must be <tool>/<version> or an absolute tool directory")
        else:
            tool_name, version = ref.split("/")
            directory = LIBRARY_ROOT / tool_name / version
        if tool_name != name:
            raise ToolSurfaceError(f"composed_tools[{name!r}] must bind a library tool of the same name")
        try:
            tool = load_tool(directory)
            compiled = tool.compiled(arm_profile)
            closure = nested_closure(tool)
            for nested in closure.values():
                nested.compiled(arm_profile)
        except (LibraryError, OSError) as exc:
            raise ToolSurfaceError(f"composed_tools[{name!r}]: {exc}") from None
        if added:
            definitions.append({})
            by_name[name] = len(definitions) - 1
        definitions[by_name[name]] = copy.deepcopy(compiled["definition"])
        results[name] = copy.deepcopy(compiled["result_schema"])
        bindings[name] = {
            "tool": tool_name, "version": version, "arm_profile": arm_profile,
            "compiled_sha256": compiled["compiled_sha256"], **({"scope": "added"} if added else {}),
            **({"root": str(directory.parent.parent), "source_sha256": tool.program.digest} if external else {}),
            **({"nested": {
                nested_ref: {"directory": str(nested.directory), "source_sha256": nested.program.digest,
                             "compiled_sha256": nested.compiled(arm_profile)["compiled_sha256"]}
                for nested_ref, nested in sorted(closure.items())
            }} if closure else {}),
        }
        for member in (tool, *[closure[key] for key in sorted(closure)]):
            for path in (member.directory / "tool.star", member.compiled_path(arm_profile)):
                inside = path.is_relative_to(LIBRARY_ROOT)
                descriptors.append({
                    "slot": f"composed:{name}",
                    "ref": path.relative_to(LIBRARY_ROOT.parent.parent.parent).as_posix() if inside else str(path),
                    "sha256": file_sha256(path),
                })
    return bindings, descriptors
