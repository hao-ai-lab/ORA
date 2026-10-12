"""Copy a ToolSurface manifest elsewhere and keep its components resolving.

Components are path strings relative to the manifest's directory. A test that
writes a manifest copy beside itself rewrites each component path so it still
names the original file; nothing is hashed by hand.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

#: Manifest fields that hold one component path, a list of them or a map of them.
SINGLE = ("base_definitions", "additional_definitions")
LISTS = ("supplementary_definitions",)
MAPS = ("components", "source_artifacts")


def component_paths(manifest: dict[str, Any], directory: Path) -> dict[str, Path]:
    """``{slot: absolute file}`` for every component the manifest names."""

    found: dict[str, Path] = {}
    for key in SINGLE:
        if isinstance(manifest.get(key), str):
            found[key] = (directory / manifest[key]).resolve()
    for key in LISTS:
        for index, ref in enumerate(manifest.get(key) or []):
            found[f"{key}[{index}]"] = (directory / ref).resolve()
    for key in MAPS:
        for slot, ref in (manifest.get(key) or {}).items():
            found[slot if key == "components" else f"source:{slot}"] = (directory / ref).resolve()
    return found


def rebase_manifest(manifest: dict[str, Any], source_dir: Path, target_dir: Path) -> dict[str, Any]:
    """A copy whose component paths, read from ``target_dir``, name the same files."""

    def rebase(ref: str) -> str:
        return os.path.relpath((Path(source_dir) / ref).resolve(), Path(target_dir).resolve())

    value = copy.deepcopy(manifest)
    for key in SINGLE:
        if isinstance(value.get(key), str):
            value[key] = rebase(value[key])
    for key in LISTS:
        if isinstance(value.get(key), list):
            value[key] = [rebase(ref) for ref in value[key]]
    for key in MAPS:
        if isinstance(value.get(key), dict):
            value[key] = {slot: rebase(ref) for slot, ref in value[key].items()}
    return value


def content_sha256(snapshot: dict[str, Any]) -> str:
    """Pin-insensitive digest of a resolved ToolSurface snapshot or a plan's surface.

    Every hash and every path beside its own hash is masked (the RunPlan
    ``model_visible`` mask), so the digest names what a surface composes --
    definitions, result schemas, tool roles, capabilities, view settings,
    revision and provenance prose -- and not how its assets pin each other.
    """
    from ora.orchestrator.model_visible import canonical_sha256, mask_pins

    return canonical_sha256(mask_pins(snapshot))


def scratch_profile_root(monkeypatch: Any, tmp_path: Path, files: dict[str, Any], *,
                         copy_tree: bool = False) -> Path:
    """Bind ``<tmp>/ora/policy/tool_surfaces`` as the catalog root, with its own lock.

    ``files`` maps paths under that root to JSON values written there. With
    ``copy_tree`` the checkout's ToolSurface assets are copied in first, so a
    test manifest may also name shipped components. The scratch tree's
    ``policy.lock`` lists everything in it, which is what the loader requires
    before it reads a component.
    """
    import json
    import shutil

    from ora.agent.composition import tool_surface
    from locked_tree import write_tree_lock

    root = Path(tmp_path) / "scratch-tree"
    profile_root = root / "ora/policy/tool_surfaces"
    if copy_tree:
        shutil.copytree(tool_surface.PROFILE_ROOT, profile_root)
    for relative, value in files.items():
        target = profile_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value), encoding="utf-8")
    write_tree_lock(root)
    monkeypatch.setattr(tool_surface, "PROFILE_ROOT", profile_root)
    return profile_root


__all__ = ["LISTS", "MAPS", "SINGLE", "component_paths", "content_sha256", "rebase_manifest",
           "scratch_profile_root"]
