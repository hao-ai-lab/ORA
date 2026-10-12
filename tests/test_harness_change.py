"""Harness-change records: a recorded overlay on a maintained baseline.

No run is launched: a harness change is checked through the default expansion
of a task-free probe RunSpec (the one ``baseline_explain`` uses).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from ora.agent.composition.tool_surface import ToolSurfaceError, resolve_tool_surface
from ora.orchestrator import defaults, harness_change
from ora.orchestrator.paths import PROJECT_ROOT

COMPOSED = "robotwin_toolchain_b2_closed_composed_v1"
DEFAULT_ROBOTWIN = "robotwin_toolchain_b2_closed_noservo_v2"


# --- composed-tool component overrides ---------------------------------------------------------------

def test_a_composed_override_rebinds_through_every_resolver_call():
    plain = resolve_tool_surface(COMPOSED)
    same = resolve_tool_surface(COMPOSED, component_overrides={"composed:grasp_execute": "grasp_execute/c1"})
    assert same.composed_tools == plain.composed_tools
    assert same.composition_sha256 == plain.composition_sha256
    with pytest.raises(ToolSurfaceError, match="grasp_execute/c9"):
        resolve_tool_surface(COMPOSED, component_overrides={"composed:grasp_execute": "grasp_execute/c9"})
    with pytest.raises(ToolSurfaceError, match="composed_tools_v1"):
        resolve_tool_surface(DEFAULT_ROBOTWIN, component_overrides={"composed:grasp_execute": "grasp_execute/c1"})


# --- harness-change records ------------------------------------------------------------------------

def _record(**overrides):
    record = {"schema": harness_change.SCHEMA, "id": "rt-composed-c1", "baseline": defaults.BASELINE_ID,
              "parent": None, "overlay": {"robotwin": {"tool_surface": COMPOSED,
                                                       "composed_tools": {"grasp_execute": "grasp_execute/c1"}}},
              "summary": "serve grasp_execute from the composed library", "sources": []}
    record.update(overrides)
    return record


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    """A source tree with the package and lock; tests add records to its harness_changes."""
    tree = tmp_path_factory.mktemp("tree")
    shutil.copytree(PROJECT_ROOT / "ora", tree / "ora")
    shutil.copy2(PROJECT_ROOT / "policy.lock", tree / "policy.lock")
    return tree


def _write(root: Path, record, name="change.json"):
    raw = json.dumps(record).encode()
    (root / harness_change.CHANGES_ROOT / name).write_bytes(raw)
    return {"path": f"{harness_change.CHANGES_ROOT}{name}", "sha256": hashlib.sha256(raw).hexdigest()}


def _probe(environment="robotwin", **policy):
    return {"policy": {"kind": "native_api", "supports_images": True, **policy}, "runtime": {"kind": environment}}


def test_a_harness_change_overlays_the_default_baseline_and_is_recorded(root):
    binding = _write(root, _record())
    expanded = defaults.expand_native_defaults(_probe(harness_change=binding), root=root)
    policy = expanded["policy"]
    assert "harness_change" not in policy
    assert policy["tool_surface"] == {"profile": COMPOSED,
                                      "component_overrides": {"composed:grasp_execute": "grasp_execute/c1"}}
    assert policy["baseline"]["selection"] == "harness_change"
    assert policy["baseline"]["harness_change"] == {"id": "rt-composed-c1", **binding}
    plain = defaults.expand_native_defaults(_probe(), root=root)["policy"]
    assert plain["baseline"]["selection"] == "omitted_tool_surface"
    assert plain["tool_surface"]["profile"] == DEFAULT_ROBOTWIN
    assert plain["baseline"]["prompt_bundle"] == policy["baseline"]["prompt_bundle"]
    # A Runtime kind the overlay does not name keeps its instance.
    libero = defaults.expand_native_defaults(_probe("libero_pro", harness_change=binding), root=root)["policy"]
    assert libero["tool_surface"] == defaults.expand_native_defaults(
        _probe("libero_pro"), root=root)["policy"]["tool_surface"]


def test_a_harness_change_is_refused_when_its_bytes_or_overlay_are_wrong(root):
    binding = _write(root, _record())
    with pytest.raises(harness_change.HarnessChangeError, match="does not match its sha256"):
        defaults.expand_native_defaults(_probe(harness_change={**binding, "sha256": "0" * 64}), root=root)
    bad_surface = _write(root, _record(overlay={"robotwin": {"tool_surface": "robodojo_placement_closed_composed_v1"}}),
                         "surface.json")
    with pytest.raises(harness_change.HarnessChangeError, match="not a registered successor"):
        defaults.expand_native_defaults(_probe(harness_change=bad_surface), root=root)
    unbindable = _write(root, _record(overlay={"robotwin": {"composed_tools": {"grasp_execute": "grasp_execute/c1"}}}),
                        "unbindable.json")
    with pytest.raises(harness_change.HarnessChangeError, match="composed_tools_v1"):
        defaults.expand_native_defaults(_probe(harness_change=unbindable), root=root)
    with pytest.raises(ValueError, match="without an explicit tool_surface"):
        defaults.expand_native_defaults(
            _probe(harness_change=binding, tool_surface={"profile": COMPOSED, "component_overrides": {}}), root=root)


@pytest.mark.parametrize("field, value, message", [
    ("baseline", "native_full_v1", "maintained baseline"),
    ("overlay", {"robotwin": {"prompt_bundle": "/etc/passwd"}}, "prompt_bundles"),
    ("overlay", {"robotwin": {"composed_tools": {"grasp_execute": "open_gripper/c1"}}}, "<tool>/c<n>"),
    ("overlay", {"robotwin": {"memory": "x"}}, "holds some of"),
    ("overlay", {}, "maps Runtime kinds"),
    ("parent", {"path": "../x.json", "sha256": "0"}, "harness_changes/ record"),
    ("parent", {"path": "outputs/x.json", "sha256": "0"}, "harness_changes/ record"),
    ("summary", " ", "summary"),
])
def test_harness_change_records_are_validated(field, value, message):
    with pytest.raises(harness_change.HarnessChangeError, match=message):
        harness_change.parse_change(_record(**{field: value}))
