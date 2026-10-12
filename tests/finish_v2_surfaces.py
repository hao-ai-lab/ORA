"""Test-local ToolSurfaces that declare the finish/abort v2 contract.

The real v2 surface generations are registered later (integration 0059 unit
A0). Until then a test copies a registered manifest into ``tmp_path``, swaps
the v1 finish capabilities for the v2 ones and lets every component resolve
from its original file by path: nothing in the repository is written and no
sha256 is spelled by hand.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from ora.agent.composition import tool_surface
from ora.agent.composition.finish_outcomes_schema import ABORT_CAPABILITY, CAPABILITY
from surface_copies import rebase_manifest

#: The current default surface of each backend in scope.
DEFAULTS = {
    "robotwin": "robotwin_toolchain_b2_closed_noservo_v2",
    "libero": "libero_placement_direct_overhead_v1",
    "robodojo": "robodojo_placement_closed_skills_v2",
}


def install(monkeypatch, tmp_path: Path, source_profile: str, *, abort: bool = True,
            profile_id: str | None = None, extra: Iterable[str] = (),
            drop: Iterable[str] = ()) -> str:
    """Register a v2 copy of ``source_profile`` for this test; return its id."""

    source = tool_surface.PROFILE_CATALOG[source_profile]
    manifest = json.loads(source.read_text(encoding="utf-8"))
    profile = profile_id or f"{source_profile}__finish_v2_test"
    removed = set(drop)
    capabilities = [name for name in manifest["capabilities"] if name not in removed]
    for name in (CAPABILITY, *((ABORT_CAPABILITY,) if abort else ()), *extra):
        if name not in capabilities:
            capabilities.append(name)
    folder = tmp_path / "finish-v2-surfaces"
    folder.mkdir(exist_ok=True)
    manifest = rebase_manifest(manifest, source.parent, folder)
    manifest.update(profile_id=profile, capabilities=capabilities)
    path = folder / f"{profile}.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    monkeypatch.setitem(tool_surface.PROFILE_CATALOG, profile, path)
    # Components stay where they are; the loading boundary is widened for
    # this test only so the copy may reference them.
    monkeypatch.setattr(tool_surface, "PROFILE_ROOT", Path(path.anchor))
    return profile


def resolve(monkeypatch, tmp_path: Path, backend: str, **kwargs: Any):
    """The resolved v2 copy of one backend's current default surface."""

    return tool_surface.resolve_tool_surface(install(monkeypatch, tmp_path, DEFAULTS[backend], **kwargs))


__all__ = ["DEFAULTS", "install", "resolve"]
