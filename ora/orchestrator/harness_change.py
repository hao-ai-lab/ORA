"""A harness change: a recorded change on top of a maintained baseline.

A ``harness-change`` record (schema ``ora.harness-change.v1``) names
the baseline it changes and, per Runtime kind, an **overlay** relative to that
baseline's instance:

- ``tool_surface``: a registered successor of the instance's surface
  (``profile_catalog.COMPOSED_SUCCESSORS``), or the instance's own;
- ``composed_tools``: ``{tool: "<tool>/<version>"}``, bound on that surface as
  ``composed:<tool>`` component overrides (``bind_composed_tools``);
- ``task_tools``: ``{task: {tool: "<tool>/<version>"}}``, composed tools bound
  only on that task's episodes (a RoboTwin task name, or ``<suite>-tNN`` for
  LIBERO, as :func:`task_key` derives it); a tool the surface does not declare
  is added there and nowhere else;
- ``prompt_bundle``: another tracked bundle under ``ora/policy/prompt_bundles/``.

Records are policy assets under ``ora/policy/harness_changes/``.

The overlay is cumulative, so a record alone fixes the harness; ``parent``
links the record it was derived from, for lineage. A RunSpec binds a record
as ``policy.harness_change: {path, sha256}`` (a tracked file, so the plan's
source commit fixes its bytes); ``defaults.expand_native_defaults`` applies it
and records ``policy.baseline.selection: harness_change``. A record is
immutable once launched.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "ora.harness-change.v1"
FIELDS = frozenset({"schema", "id", "baseline", "parent", "overlay", "summary", "sources"})
OVERLAY_FIELDS = frozenset({"tool_surface", "composed_tools", "task_tools", "prompt_bundle"})
BUNDLE_ROOT = "ora/policy/prompt_bundles/"
CHANGES_ROOT = "ora/policy/harness_changes/"
_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{2,79}$")
_REF = re.compile(r"^([a-z][a-z0-9_]*)/(c[0-9]+)$")
_TASK = re.compile(r"^[a-z0-9_]+(-t[0-9]{2})?$")


class HarnessChangeError(ValueError):
    """A harness-change record or its binding is invalid."""


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise HarnessChangeError(message)


def _binding(value: Any, field: str) -> tuple[str, str]:
    _expect(isinstance(value, Mapping) and set(value) == {"path", "sha256"}
            and isinstance(value["path"], str) and isinstance(value["sha256"], str),
            f"{field} must be {{path, sha256}}")
    path = Path(value["path"])
    _expect(value["path"].startswith(CHANGES_ROOT) and value["path"].endswith(".json")
            and ".." not in path.parts, f"{field}.path must be a {CHANGES_ROOT} record")
    return value["path"], value["sha256"]


def _tool_refs(tools: Any, label: str) -> None:
    _expect(isinstance(tools, Mapping) and tools and all(
        isinstance(ref, str) and (match := _REF.match(ref)) is not None and match.group(1) == name
        for name, ref in tools.items()), f"{label} maps a tool to <tool>/c<n>")


def task_key(runtime: Mapping[str, Any]) -> str | None:
    """The task a RunSpec runtime plays, as ``task_tools`` keys it."""
    if isinstance(runtime.get("suite"), str) and isinstance(runtime.get("task_id"), int):
        return f"{runtime['suite']}-t{runtime['task_id']:02d}"
    return runtime.get("task") if isinstance(runtime.get("task"), str) else None


def parse_change(document: Any) -> dict[str, Any]:
    """Validate a decoded record; returns it unchanged."""
    from ora.agent.composition.baseline import BASELINE_CATALOGS
    from .defaults import NATIVE_BASELINE_KINDS

    _expect(isinstance(document, Mapping) and document.get("schema") == SCHEMA, f"schema must be {SCHEMA}")
    _expect(set(document) == FIELDS, f"fields must be exactly {sorted(FIELDS)}")
    _expect(isinstance(document["id"], str) and bool(_ID.match(document["id"])), "id must match [a-z0-9._-]{3,80}")
    _expect(document["baseline"] in BASELINE_CATALOGS, "baseline must be a maintained baseline id")
    if document["parent"] is not None:
        _binding(document["parent"], "parent")
    overlay = document["overlay"]
    _expect(isinstance(overlay, Mapping) and overlay and set(overlay) <= NATIVE_BASELINE_KINDS,
            f"overlay maps Runtime kinds {sorted(NATIVE_BASELINE_KINDS)} to changes")
    for kind, change in overlay.items():
        _expect(isinstance(change, Mapping) and change and set(change) <= OVERLAY_FIELDS,
                f"overlay.{kind} holds some of {sorted(OVERLAY_FIELDS)}")
        if "tool_surface" in change:
            _expect(isinstance(change["tool_surface"], str), f"overlay.{kind}.tool_surface must be a profile id")
        if "composed_tools" in change:
            _tool_refs(change["composed_tools"], f"overlay.{kind}.composed_tools")
        if "task_tools" in change:
            per_task = change["task_tools"]
            _expect(isinstance(per_task, Mapping) and per_task and all(
                isinstance(task, str) and _TASK.match(task) for task in per_task),
                f"overlay.{kind}.task_tools maps tasks (task name or <suite>-tNN) to composed tools")
            for task, tools in per_task.items():
                _tool_refs(tools, f"overlay.{kind}.task_tools.{task}")
                _expect(not set(tools) & set(change.get("composed_tools", {})),
                        f"overlay.{kind}.task_tools.{task} rebinds a tool composed_tools already binds")
        if "prompt_bundle" in change:
            bundle = change["prompt_bundle"]
            _expect(isinstance(bundle, str) and bundle.startswith(BUNDLE_ROOT) and bundle.endswith(".yaml")
                    and ".." not in Path(bundle).parts, f"overlay.{kind}.prompt_bundle must be a {BUNDLE_ROOT} file")
    _expect(isinstance(document["summary"], str) and document["summary"].strip() != "", "summary must be text")
    _expect(isinstance(document["sources"], list) and all(isinstance(item, str) and item for item in document["sources"]),
            "sources lists attribution record references")
    return dict(document)


def load_change(root: Path, binding: Any) -> tuple[dict[str, Any], dict[str, str]]:
    """The record a ``{path, sha256}`` binding names, checked against its bytes."""
    path, sha256 = _binding(binding, "policy.harness_change")
    file = Path(root) / path
    _expect(file.is_file(), f"harness change {path} does not exist")
    raw = file.read_bytes()
    _expect(hashlib.sha256(raw).hexdigest() == sha256, f"harness change {path} does not match its sha256")
    record = parse_change(json.loads(raw))
    return record, {"id": record["id"], "path": path, "sha256": sha256}


def apply_overlay(record: Mapping[str, Any], instance: Mapping[str, Any], kind: str,
                  *, root: Path, task: str | None = None) -> tuple[dict[str, Any], str]:
    """The ToolSurface binding and prompt bundle path of ``instance`` under ``record``.

    Without an overlay for ``kind`` the instance is unchanged; ``task`` selects
    that task's ``task_tools``, which no other task receives. The surface is
    resolved here so an unbindable composed tool fails when the RunSpec is
    composed, not when the policy first calls it.
    """
    from ora.agent.composition.profile_catalog import COMPOSED_SUCCESSORS
    from ora.agent.composition.tool_surface import (
        COMPOSED_OVERRIDE_PREFIX, ToolSurfaceError, resolve_tool_surface,
    )

    change = record["overlay"].get(kind, {})
    profile = change.get("tool_surface", instance["tool_surface"])
    _expect(profile == instance["tool_surface"] or COMPOSED_SUCCESSORS.get(profile) == instance["tool_surface"],
            f"overlay.{kind}.tool_surface {profile} is not a registered successor of {instance['tool_surface']}")
    tools = {**change.get("composed_tools", {}), **change.get("task_tools", {}).get(task, {})}
    overrides = {f"{COMPOSED_OVERRIDE_PREFIX}{name}": ref for name, ref in tools.items()}
    try:
        resolve_tool_surface(profile, component_overrides=overrides)
    except (ValueError, ToolSurfaceError) as exc:
        raise HarnessChangeError(f"overlay.{kind}: {exc}") from None
    bundle = change.get("prompt_bundle", instance["prompt_bundle"])
    _expect((Path(root) / bundle).is_file(), f"overlay.{kind}.prompt_bundle {bundle} does not exist")
    return {"profile": profile, "component_overrides": overrides}, bundle


def bind_change(path: Path | str, kind: str, *, root: Path,
                task: str | None = None) -> tuple[dict[str, str], dict[str, Any]]:
    """The ``{path, sha256}`` binding of a tracked record and the surface it gives ``kind`` (on ``task``)."""
    from ora.agent.composition.baseline import load_native_baseline

    file = Path(path)
    relative = (file.resolve().relative_to(Path(root).resolve()) if file.is_absolute() else file).as_posix()
    _expect((Path(root) / relative).is_file(), f"harness change {relative} does not exist")
    binding = {"path": relative, "sha256": hashlib.sha256((Path(root) / relative).read_bytes()).hexdigest()}
    record, _ref = load_change(root, binding)
    instance = load_native_baseline(root, record["baseline"])["instances"].get(kind)
    _expect(instance is not None, f"baseline {record['baseline']} has no {kind} instance")
    surface, _bundle = apply_overlay(record, instance, kind, root=root, task=task)
    return binding, surface


__all__ = ["CHANGES_ROOT", "HarnessChangeError", "SCHEMA", "apply_overlay", "bind_change", "load_change",
           "parse_change", "task_key"]
