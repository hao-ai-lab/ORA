"""Maintained policy assets name each other by path only (stage 2, units L2 and L3).

A composable asset -- one a loader in this tree reaches without a refusal --
holds no hand-written asset-to-asset sha256: ``policy.lock`` carries asset
integrity and each loader records the sha256 of the bytes it reads. An
embodiment descriptor names its repository sources by path (the RunPlan's
source commit identifies their bytes) and a wording review attests the card
text by ``card_sha256`` alone. The inline hashes left are external identities,
human-review pins (card text, task-contract layer text), generated compiled
card manifests, or sit in frozen assets, which compose only from their launch
commit and keep their pins.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from tests.support.locked_tree import write_tree_lock
from tests.support.surface_copies import scratch_profile_root
from ora.agent.composition import baseline, tool_surface
from ora.agent.composition.embodiment import EmbodimentDescriptorError
from ora.agent.composition.embodiment_geometry import bind_public_geometry_descriptor
from ora.agent.composition.locked_assets import locked_asset
from ora.agent.composition.task_contract_review import load_review_record
from ora.orchestrator import policy_lock, policy_pins

ROOT = Path(__file__).resolve().parents[2]
ACTIVATION = json.loads((ROOT / "tests/data/default-expectations.json").read_text(encoding="utf-8"))
#: The kinds an inline 64-hex literal may still have; every other literal in a
#: composable asset is an asset pin the strip should have removed.
REMAINING = {"external", "human_review", "generated_card_manifest", "frozen"}
#: Composable literals that equal a locked asset's bytes yet are not asset pins.
COINCIDENT: set[tuple[str, str]] = set()


def test_every_composable_asset_is_path_only():
    plans = policy_pins.plan_strip(ROOT)
    assert plans == [], "inline pins in composable assets; run ./scripts/ora lock --strip-pins: " + \
        ", ".join(f"{plan.path} ({len(plan.pins)})" for plan in plans)
    assert policy_pins.command_strip(ROOT, check=True, write_lock=policy_lock.write_lock) == 0


def test_inline_hashes_left_are_external_human_review_generated_or_frozen():
    rows = policy_pins.remaining_literals(ROOT)
    assert {row["kind"] for row in rows} <= REMAINING
    unclassified = [row for row in rows if row["kind"] == "unclassified"]
    assert unclassified == []
    pins = [(row["path"], row["key"], row["asset_pin"]) for row in rows
            if row["composed"] and row["asset_pin"]
            and row["kind"] != "human_review"
            and (row["path"], row["key"]) not in COINCIDENT]
    assert pins == [], "a composable asset still pins another asset by hash"
    # The ratchet only turns one way: frozen assets keep theirs, composed ones never gain one.
    assert any(row["kind"] == "frozen" for row in rows)
    assert not any(row["kind"] == "frozen" and row["composed"] for row in rows)
    # Unit L3: a composed descriptor pins external sources only; a wording review only its card.
    composed = {row["path"] for row in rows if row["composed"]}
    for path in composed:
        value = json.loads((ROOT / path).read_text())
        if "/wording-review-" in path:
            assert set(value) == {"schema", "decision", "reviewer", "card_sha256", "scope"}, path
        if "embodiment-descriptor" in str(value.get("schema", "")):
            for source in value["provenance"]["sources"]:
                assert not ({"revision", "sha256"} & set(source)) if "local_path" in source else (
                    {"revision", "sha256"} <= set(source)), (path, source["source_id"])


def test_the_composable_set_is_what_this_tree_can_load():
    holders = policy_pins.composable_pin_holders(ROOT)
    for manifest in tool_surface.PROFILE_CATALOG.values():
        assert holders[manifest.resolve().relative_to(ROOT).as_posix()] == "tool_surface_manifest"
    for catalog in baseline.BASELINE_CATALOGS.values():
        assert holders[catalog] == "baseline_catalog"
    for registry in (ACTIVATION["task_contracts"][name]["registry"] for name in ("robotwin", "robodojo")):
        assert registry in holders
    for frozen in ACTIVATION["frozen_task_contracts"].values():
        assert frozen["path"] not in holders
    frozen_catalogs = [path for path in (ROOT / "ora/policy/baselines").glob("*.json")
                       if path.relative_to(ROOT).as_posix() not in baseline.BASELINE_CATALOGS.values()]
    assert not {path.relative_to(ROOT).as_posix() for path in frozen_catalogs} & set(holders)
    # Unit L3: the descriptors and wording reviews those assets bind, and the
    # ARX descriptor the RoboDojo embodiment activation compiles.
    from ora.orchestrator.robodojo_embodiment import DESCRIPTOR_PATH
    assert holders[DESCRIPTOR_PATH.relative_to(ROOT).as_posix()] == "embodiment_descriptor"
    for manifest in tool_surface.PROFILE_CATALOG.values():
        geometry = (json.loads(manifest.read_text()).get("view_config") or {}).get("public_geometry_descriptor")
        if geometry:
            assert holders[geometry] in {"embodiment_descriptor", "geometry_descriptor"}
    for path in (ROOT / "ora/policy/environments").glob("*/context-profile-*.json"):
        profile = json.loads(path.read_text())
        if "compiled_embodiment" in profile:
            assert holders[profile["compiled_embodiment"]["descriptor"]] == "embodiment_descriptor"
            assert holders[profile["compiled_embodiment"]["review"]] == "embodiment_wording_review"


def test_compiled_card_manifests_are_generated_from_their_path_only_descriptors():
    """``compiled-v*/manifest.json`` is write_compiled_card's output; no ``.py`` bytes reach it."""
    from ora.agent.composition.embodiment import compiled_card_files

    manifests = sorted((ROOT / policy_pins.EMBODIMENTS).glob("*/compiled-v*/manifest.json"))
    assert len(manifests) >= 1
    for manifest in manifests:
        recorded = json.loads(manifest.read_text())
        descriptors = [path for path in manifest.parent.parent.glob("descriptor-*.json")
                       if json.loads(path.read_text())["descriptor_id"] == recorded["descriptor_id"]]
        assert len(descriptors) == 1, manifest
        card, _manifest, text = compiled_card_files(descriptors[0].read_bytes(), project_root=ROOT)
        assert text == manifest.read_text() and card == (manifest.parent / "manual.md").read_text()
        statuses = {check["status"] for check in recorded["source_audit"]["checks"]}
        assert statuses <= {"repository_source", "content_addressed_external"}
        assert all("expected_sha256" not in check for check in recorded["source_audit"]["checks"]
                   if check["status"] == "repository_source")


def test_frozen_assets_keep_their_inline_pins():
    """Frozen bytes are evidence of what ran; this tree refuses them before loading.

    Written over whatever frozen assets the tree still holds, so removing them
    (stage 0.2) leaves the test meaningful.
    """

    holders = policy_pins.composable_pin_holders(ROOT)
    catalogs = [path for path in (ROOT / "ora/policy/baselines").glob("*.json")
                if path.relative_to(ROOT).as_posix() not in holders]
    for path in catalogs:
        for instance in json.loads(path.read_text())["instances"].values():
            for key in ("motion_profile", "control_profile"):
                if key in instance.get("runtime", {}):
                    assert set(instance["runtime"][key]) == {"path", "sha256"}, path
    for frozen in ACTIVATION["frozen_task_contracts"].values():
        if frozen["path"].startswith("ora/policy/environments/robodojo/") and (ROOT / frozen["path"]).is_file():
            assert all("record_sha256" in entry
                       for entry in json.loads((ROOT / frozen["path"]).read_text())["tasks"].values())


# ---- the loaders -----------------------------------------------------------------

def test_a_reference_resolves_to_the_bytes_read_and_their_sha256():
    relative = "ora/policy/embodiments/franka-panda/motion-profile-v2.json"
    asset = locked_asset(ROOT, relative, label="motion profile")
    assert asset.record() == {"path": relative, "sha256": hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()}
    assert asset.data == (ROOT / relative).read_bytes()


@pytest.mark.parametrize("reference,message", [
    ("../outside.json", "escapes the source tree"),
    ("ora/../../outside.json", "escapes the source tree"),
    ("ora/policy/../../README.md", "is not in policy.lock"),
    ("README.md", "is not in policy.lock"),
    ("ora/policy/does-not-exist.json", "is not in policy.lock"),
    ({"path": "ora/policy/retired-assets.json", "sha256": "0" * 64}, "must be a path string"),
    ("", "non-empty path string"),
])
def test_a_reference_outside_the_tree_or_the_lock_or_inline_is_refused(reference, message):
    with pytest.raises(ValueError, match=message):
        locked_asset(ROOT, reference, label="asset")


def test_every_converted_loader_refuses_an_inline_pin(monkeypatch, tmp_path):
    descriptor = "ora/policy/embodiments/arx-x5/descriptor-public-geometry-v5.json"
    inline = {"path": descriptor, "sha256": locked_asset(ROOT, descriptor, label="d").sha256}
    with pytest.raises(EmbodimentDescriptorError, match="inline sha256 pins are not accepted"):
        bind_public_geometry_descriptor(inline, project_root=ROOT)
    review = "ora/policy/environments/robotwin/task-contracts/reviews/adjust_bottle-review-v3.json"
    with pytest.raises(ValueError, match="must be a path string"):
        load_review_record(ROOT, {"path": review, "sha256": "0" * 64})
    assert load_review_record(ROOT, review)[1] == locked_asset(ROOT, review, label="r").sha256
    # A catalog that still pins its runtime profiles inline is refused, not verified.
    catalog = json.loads((ROOT / baseline.CATALOG_PATH).read_text())
    for instance in catalog["instances"].values():
        for key in baseline.RUNTIME_PROFILE_FIELDS:
            if key in instance["runtime"]:
                relative = instance["runtime"][key]
                instance["runtime"][key] = {"path": relative, "sha256": locked_asset(ROOT, relative, label=key).sha256}
    inline_catalog = tmp_path / "inline-catalog.json"
    inline_catalog.write_text(json.dumps(catalog))
    monkeypatch.setitem(baseline.BASELINE_CATALOGS, baseline.BASELINE_ID, str(inline_catalog))
    with pytest.raises(ValueError, match="must be a path string"):
        baseline.load_native_baseline(ROOT)


def _definitions() -> dict:
    return {"tools": [{"name": "inspect", "description": "x", "inputSchema": {"type": "object"}}]}


def _manifest(base: object) -> dict:
    return {
        "schema": "ora.tool-surface-profile.v1", "profile_id": "scratch_surface",
        "revision": 1, "runtime_id": "native_atomic_causal", "base_definitions": base,
        "tool_order": ["inspect"], "components": {}, "description_order": [], "schema_order": [],
        "result_order": [], "mutation_tools": [], "lifecycle_tools": [], "capabilities": [],
    }


@pytest.mark.parametrize("base,message", [
    ({"ref": "base.json", "sha256": "0" * 64}, "must be a path string"),
    ("../../../../../outside.json", "unavailable"),
    ("unlocked.json", "is not in policy.lock"),
])
def test_the_tool_surface_loader_refuses_inline_escaping_or_unlocked_components(monkeypatch, tmp_path, base, message):
    root = scratch_profile_root(monkeypatch, tmp_path, {
        "scratch/base.json": _definitions(), "scratch/manifest.json": _manifest(base),
    })
    (root / "scratch/unlocked.json").write_text(json.dumps(_definitions()))  # written after the lock
    monkeypatch.setitem(tool_surface.PROFILE_CATALOG, "scratch_surface", root / "scratch/manifest.json")
    with pytest.raises(tool_surface.ToolSurfaceError, match=message):
        tool_surface.resolve_tool_surface("scratch_surface")


def test_lock_check_catches_a_stale_lock_after_an_asset_edit(tmp_path):
    """Loaders check membership and hash what they read; CI checks the lock against the bytes."""

    root = tmp_path / "repository"
    surfaces = root / "ora/policy/tool_surfaces/scratch"
    surfaces.mkdir(parents=True)
    (surfaces / "base.json").write_text(json.dumps(_definitions()))
    (surfaces / "manifest.json").write_text(json.dumps(_manifest("base.json")))
    for args in (("init", "-q"), ("add", "."), ("-c", "user.name=F", "-c", "user.email=f@example.invalid",
                                                 "commit", "-qm", "fixture")):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    policy_lock.write_lock(root)
    assert policy_lock.check_lock(root).clean
    edited = _definitions()
    edited["tools"][0]["description"] = "edited"
    (surfaces / "base.json").write_text(json.dumps(edited))
    drift = policy_lock.check_lock(root)
    assert drift.changed == ("ora/policy/tool_surfaces/scratch/base.json",)
    # The loader still reads the edited bytes and records their hash; the gate is the lock check.
    asset = locked_asset(root, "ora/policy/tool_surfaces/scratch/base.json", label="base")
    assert json.loads(asset.data) == edited
    policy_lock.write_lock(root)
    assert policy_lock.check_lock(root).clean


# ---- the strip ---------------------------------------------------------------------

MANIFEST_TEXT = """{
  "schema": "ora.tool-surface-profile.v1",
  "provenance": {"derived_from": "old.json", "source_manifest_sha256": "%(old)s", "change": "x"},
  "base_definitions": {
    "ref": "base.json",
    "sha256": "%(base)s"
  },
  "components": {"help": {"sha256": "%(help)s", "ref": "../shared/help.json"}, "kept": "kept.json"},
  "view_config": {"public_geometry_descriptor": {"path": "ora/policy/d.json", "sha256": "%(d)s"}}
}
"""


def test_the_strip_rewrites_only_the_pins_and_keeps_the_formatting():
    digests = {"old": "1" * 64, "base": "2" * 64, "help": "3" * 64, "d": "4" * 64}
    text = MANIFEST_TEXT % digests
    path = "ora/policy/tool_surfaces/a/manifest.json"
    stripped, pins = policy_pins._strip_value(path, "tool_surface_manifest", json.loads(text))
    assert {pin.key for pin in pins} == {
        "base_definitions/sha256", "components/help/sha256",
        "view_config/public_geometry_descriptor/sha256", "provenance/source_manifest_sha256"}
    assert {pin.target for pin in pins if pin.verified} == {
        "ora/policy/tool_surfaces/a/base.json", "ora/policy/tool_surfaces/shared/help.json",
        "ora/policy/d.json"}
    rewritten = policy_pins._rewrite_text(text, pins)
    assert json.loads(rewritten) == stripped
    assert stripped["base_definitions"] == "base.json"
    assert stripped["components"] == {"help": "../shared/help.json", "kept": "kept.json"}
    assert stripped["view_config"]["public_geometry_descriptor"] == "ora/policy/d.json"
    assert stripped["provenance"] == {"derived_from": "old.json", "change": "x"}
    assert '"schema": "ora.tool-surface-profile.v1",\n' in rewritten
    assert policy_pins._strip_value(path, "tool_surface_manifest", stripped)[1] == []


def test_the_strip_refuses_a_stale_pin_instead_of_reviving_its_asset(monkeypatch, tmp_path):
    relative = "ora/policy/tool_surfaces/a/manifest.json"
    (tmp_path / relative).parent.mkdir(parents=True)
    (tmp_path / "ora/policy/tool_surfaces/a/base.json").write_text("{}")
    text = MANIFEST_TEXT % {"old": "1" * 64, "base": "2" * 64, "help": "3" * 64, "d": "4" * 64}
    (tmp_path / relative).write_text(text)
    write_tree_lock(tmp_path)
    monkeypatch.setattr(policy_pins, "composable_pin_holders", lambda root: {relative: "tool_surface_manifest"})
    with pytest.raises(policy_pins.StalePinError, match="base_definitions/sha256 names"):
        policy_pins.plan_strip(tmp_path)
    assert (tmp_path / relative).read_text() == text


def test_the_model_visible_digest_reads_a_profile_descriptor_in_either_form(tmp_path):
    """Path-only profiles and pre-L2 ``{path, sha256}`` profiles yield the same embodiment item.

    ``scripts/dev/compare-model-visible`` applies this checkout's ``model_visible``
    to trees on both sides of the conversion.
    """
    from ora.orchestrator.model_visible import model_visible_items
    from ora.orchestrator.spec import resolve_run_spec
    from tests.support.run_specs import native_spec

    plan = resolve_run_spec(native_spec("libero_pro"), root=ROOT)
    items = model_visible_items(plan, root=ROOT)["embodiment"]
    assert set(items) == {"compiled_embodiment", "public_geometry_descriptor"}
    profile = json.loads((ROOT / plan["runtime"]["environment_prompt_profile"]["profile_path"]).read_text())
    descriptor = profile["compiled_embodiment"]["descriptor"]
    assert isinstance(descriptor, str)
    profile["compiled_embodiment"]["descriptor"] = {
        "path": descriptor, "sha256": locked_asset(ROOT, descriptor, label="d").sha256}
    for relative, data in ((plan["runtime"]["environment_prompt_profile"]["profile_path"], json.dumps(profile)),
                           (descriptor, (ROOT / descriptor).read_text())):
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative).write_text(data)
    assert model_visible_items(plan, root=tmp_path)["embodiment"] == items


DESCRIPTOR_TEXT = """{
  "schema": "ora.embodiment-descriptor.v2",
  "provenance": {
    "runtime_truth": "compiled_model",
    "sources": [
      {
        "source_id": "external",
        "kind": "robot_urdf",
        "uri": "robot-asset://x",
        "revision": "abc",
        "sha256": "%(external)s",
        "supports": ["a"]
      },
      {
        "source_id": "adapter",
        "kind": "environment_runtime_adapter",
        "uri": "repo://ora/a.py",
        "local_path": "ora/a.py",
        "revision": "content-addressed",
        "sha256": "%(adapter)s",
        "supports": ["b"]
      }
    ]
  }
}
"""


def test_the_strip_names_repository_sources_by_path_and_keeps_external_identities():
    text = DESCRIPTOR_TEXT % {"external": "1" * 64, "adapter": "2" * 64}
    path = "ora/policy/embodiments/x/descriptor-v1.json"
    for kind, verified in (("embodiment_descriptor", True), ("geometry_descriptor", False)):
        stripped, pins = policy_pins._strip_value(path, kind, json.loads(text))
        # A loader audits an embodiment_descriptor's provenance, so its byte pin is
        # checked before removal; a geometry-only descriptor's never was.
        assert [(pin.key, pin.verified) for pin in pins] == [
            ("provenance/sources/1/revision", False), ("provenance/sources/1/sha256", verified)]
        rewritten = policy_pins._rewrite_text(text, pins)
        assert json.loads(rewritten) == stripped
        assert stripped["provenance"]["sources"][0] == json.loads(text)["provenance"]["sources"][0]
        assert set(stripped["provenance"]["sources"][1]) == {"source_id", "kind", "uri", "local_path", "supports"}
    assert policy_pins.path_only_descriptor(json.loads(text)) == stripped
    assert policy_pins.classify(path, "provenance/sources/0/sha256", json.loads(text)) == "external"
    assert policy_pins.classify(path, "provenance/sources/1/sha256", json.loads(text)) is None


def test_the_strip_verifies_a_wording_review_against_its_descriptor(monkeypatch, tmp_path):
    descriptor = "ora/policy/embodiments/x/descriptor-v1.json"
    review = "ora/policy/embodiments/x/wording-review-v1.json"
    (tmp_path / descriptor).parent.mkdir(parents=True)
    (tmp_path / descriptor).write_text("{}")
    text = json.dumps({"schema": "ora.embodiment-wording-review.v1", "decision": "accepted",
                       "reviewer": "codex_source_review", "descriptor_sha256": "0" * 64,
                       "card_sha256": "3" * 64, "scope": "s"}, indent=2)
    (tmp_path / review).write_text(text)
    write_tree_lock(tmp_path)
    monkeypatch.setattr(policy_pins, "composable_pin_holders", lambda root: {review: "embodiment_wording_review"})
    monkeypatch.setattr(policy_pins, "review_targets", lambda root: {review: descriptor})
    with pytest.raises(policy_pins.StalePinError, match="descriptor_sha256 names"):
        policy_pins.plan_strip(tmp_path)
    (tmp_path / review).write_text(text.replace("0" * 64, hashlib.sha256(b"{}").hexdigest()))
    [plan] = policy_pins.plan_strip(tmp_path)
    assert json.loads(plan.text) == {key: value for key, value in json.loads(text).items() if key != "descriptor_sha256"}


def test_a_wording_review_that_pins_descriptor_bytes_is_refused(monkeypatch, tmp_path):
    """The review attests the card text; a descriptor_sha256 beside it is refused, never verified."""
    import shutil
    from ora.agent.composition.environment_context import (
        EnvironmentContextError, resolve_environment_prompt_profile,
    )
    from ora.environment.benchmarks.robodojo import public_context

    profile_path = "ora/policy/environments/robodojo/context-profile-public-explicit-v6.json"
    profile = json.loads((ROOT / profile_path).read_text())
    for relative in ("ora/policy/environments", "ora/policy/embodiments/arx-x5"):
        shutil.copytree(ROOT / relative, tmp_path / relative)
    for source in json.loads((ROOT / profile["compiled_embodiment"]["descriptor"]).read_text())["provenance"]["sources"]:
        if "local_path" in source:
            (tmp_path / source["local_path"]).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / source["local_path"], tmp_path / source["local_path"])
    shutil.copyfile(ROOT / "policy.lock", tmp_path / "policy.lock")
    resolve_environment_prompt_profile(public_context(), project_root=tmp_path, profile_id=profile["profile_id"])
    review = tmp_path / profile["compiled_embodiment"]["review"]
    value = json.loads(review.read_text())
    value["descriptor_sha256"] = hashlib.sha256((tmp_path / profile["compiled_embodiment"]["descriptor"]).read_bytes()).hexdigest()
    review.write_text(json.dumps(value))
    with pytest.raises(EnvironmentContextError, match="wording review does not accept"):
        resolve_environment_prompt_profile(public_context(), project_root=tmp_path, profile_id=profile["profile_id"])
