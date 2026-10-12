"""Semantic review is distinct from source binding and public projection."""
import json
from pathlib import Path

import pytest

from locked_tree import write_tree_lock
from ora.agent.composition.task_contract import (
    REGISTRIES, TaskContractError, resolve_task_contract_layer,
)
from ora.agent.composition.task_contract_review import (
    audit_registry, main, render_public_contract, review_errors,
)

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "ora/policy/environments/robotwin/task-contracts/reviews/rotate_qrcode-review-v3.json"


def test_default_qr_delivery_is_corrected_and_private_review_is_not_a_layer():
    record = json.loads(REVIEW.read_text())
    layer, disclosure = resolve_task_contract_layer({"kind": "robotwin", "task": "rotate_qrcode",
        "required_revision": record["verified_revisions"][0]}, root=ROOT)
    assert layer.endswith("rotate_qrcode-completion-v3.md")
    assert (ROOT / layer).read_text() == render_public_contract(record)
    assert disclosure["semantic_review_status"] == "review_record_valid"
    assert "source_dependencies" not in disclosure and "evidence" not in disclosure


def test_batch_audit_covers_all_fifty_reviewed_tasks():
    report = audit_registry(ROOT, "robotwin")
    assert report["counts"] == {"review_record_valid": 50}
    assert len(report["tasks"]) == 50
    assert report["witness_artifacts_checked"] is False
    for backend in ("libero_pro", "robodojo"):
        assert audit_registry(ROOT, backend)["status"] == "no_registry"


@pytest.mark.parametrize("damage,expected", [
    ("old_flat_claim", "contradictory_public_claim:lying flat"),
    ("no_source", "missing_source_dependencies"),
    ("no_witness", "missing_evidence"),
    ("unchecked_geometry", "unchecked_assets_and_frames"),
    ("extra_clause", "public_clause_without_source_mapping"),
    ("private_threshold", "public_text_contains_private_detail_or_placeholder"),
])
def test_regeneration_rejects_contradiction_and_unreviewed_additions(damage, expected):
    record = json.loads(REVIEW.read_text())
    if damage == "old_flat_claim": record["public_clauses"][0] = "The board must end lying flat."
    if damage == "no_source": record["source_dependencies"] = []
    if damage == "no_witness": record["evidence"] = []
    if damage == "unchecked_geometry": record["checks"]["assets_and_frames"]["status"] = "pending"
    if damage == "extra_clause": record["public_clauses"].append("Keep holding the sign.")
    if damage == "private_threshold": record["public_clauses"][0] = "Reach z 0.75."
    assert expected in review_errors(record)


def test_valid_review_renders_only_allowlisted_prose():
    record = json.loads(REVIEW.read_text())
    record["private_note"] = "SECRET"
    assert review_errors(record) == []
    assert "SECRET" not in render_public_contract(record)
    assert "source_dependencies" not in render_public_contract(record)


def test_runtime_refuses_semantic_record_changed_without_registration(tmp_path):
    """The registry names the review by path; an edit is caught by its content and by the lock.

    The runtime refuses a review that no longer validates against its task and
    layer, and ``ora lock --check`` reports the edited bytes against
    the committed lock.
    """
    registry = json.loads((ROOT / REGISTRIES["robotwin"]).read_text())
    entry = registry["tasks"]["rotate_qrcode"]
    for relative in (REGISTRIES["robotwin"], entry["path"], entry["semantic_review"]):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    lock = write_tree_lock(tmp_path).read_text()
    (tmp_path / entry["semantic_review"]).write_text('{}')
    with pytest.raises(TaskContractError, match="semantic review invalid"):
        resolve_task_contract_layer({"kind": "robotwin", "task": "rotate_qrcode",
            "required_revision": entry["verified_revisions"][0]}, root=tmp_path)
    assert write_tree_lock(tmp_path).read_text() != lock


def test_template_is_pending_and_strict_batch_accepts_complete_review(capsys):
    assert main(["template", "--task", "new_task"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["status"] == "needs_review"
    assert "semantic_review_incomplete" in review_errors(record)
    assert main(["audit", "--root", str(ROOT), "--require-reviewed"]) == 0


def test_strict_batch_still_rejects_unreviewed_and_unregistered_entries(tmp_path, monkeypatch, capsys):
    import shutil
    relative = "ora/policy/environments/robotwin/task-contracts"
    shutil.copytree(ROOT / relative, tmp_path / relative)
    # A test-local registry derived from the current one: thirteen tasks
    # dropped and thirty-six left without a semantic review. The auditor reads
    # only the task names in contract_source, so the fifty-task roster stands
    # in for the retired source package.
    from ora.environment.benchmarks.robotwin import TASKS

    registry = json.loads((ROOT / REGISTRIES["robotwin"]).read_text())
    source = tmp_path / registry["contract_source"]
    source.mkdir(parents=True)
    for task in TASKS:
        (source / f"{task}.json").write_text("{}\n")
    others = sorted(set(registry["tasks"]) - {"rotate_qrcode"})
    for task in others[:13]:
        del registry["tasks"][task]
    for task in others[13:]:
        del registry["tasks"][task]["semantic_review"]
    fixture = relative + "/registry-unreviewed-fixture.json"
    (tmp_path / fixture).write_text(json.dumps(registry, indent=2), encoding="utf-8")
    write_tree_lock(tmp_path)
    monkeypatch.setitem(REGISTRIES, "robotwin", fixture)
    report = audit_registry(tmp_path, "robotwin")
    assert report["counts"] == {"needs_semantic_review": 36, "not_registered": 13, "review_record_valid": 1}
    assert main(["audit", "--root", str(tmp_path), "--require-reviewed"]) == 1


def test_batch_preserves_witness_limits_instead_of_claiming_fifty_physical_qualifications():
    registry = json.loads((ROOT / REGISTRIES["robotwin"]).read_text())
    records = {task: json.loads((ROOT / entry["semantic_review"]).read_text())
               for task, entry in registry["tasks"].items()}
    assert all(r["witness_scope"]["expert_post_play_success"] for r in records.values())
    assert records["beat_block_hammer"]["witness_scope"]["hold_passed"] is False
    assert "after hold" in records["beat_block_hammer"]["witness_scope"]["visual_interpretation"]
    assert "omitted selected asset IDs" in records["click_alarmclock"]["limitations"]
    assert all(r["witness_scope"]["claim"] == "one_recorded_configuration_not_all_variant_physical_qualification"
               for r in records.values())


def test_semantic_regressions_are_not_reintroduced():
    registry = json.loads((ROOT / REGISTRIES["robotwin"]).read_text())
    def public(task):
        return (ROOT / registry["tasks"][task]["path"]).read_text()
    assert "started on the left toward the nearer" in public("place_dual_shoes")
    assert "started on the right toward the farther" in public("place_dual_shoes")
    assert "toe pointing to the robot's left" in public("place_shoe")
    assert "hamburger on the left" in public("place_burger_fries")
    assert "fries on the right" in public("place_burger_fries")
    assert "compartment" not in public("place_cans_plasticbox")
    for task in ("pick_diverse_bottles", "pick_dual_bottles"):
        assert "top of each bottle" not in public(task)
        assert "upright" not in public(task)
    for task in ("shake_bottle", "shake_bottle_horizontally"):
        assert "shake it back and forth" in public(task)
