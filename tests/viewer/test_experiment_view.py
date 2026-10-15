from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from viewer.app import build_parser
from viewer.experiment_view import (
    SCHEMA,
    audit_experiment_view,
    audit_experiment_view_files,
    render_experiment_view,
    validate_experiment_view,
)


def _definition() -> dict:
    return {
        "experiment": "viewer-fixture",
        "accepted_evidence": {
            "cell_a": {
                "run_id": "run-a", "source_commit": "source-a",
                "official_success": True,
            },
            "cell_b": {
                "run_id": "run-b", "source_commit": "source-b",
                "official_success": False,
            },
            "excluded": {"run_id": "run-x", "reason": "infrastructure_incomplete"},
        },
    }


def _cell(identifier: str, evidence: str, **extra: str) -> dict:
    return {
        "id": identifier, "accepted_evidence": evidence,
        "label": f"Cell {identifier}", "role": "fixture role",
        "summary": "A source-tracked fixture.", **extra,
    }


def _manifest(layout: str, cells: list[dict]) -> dict:
    return {
        "schema": SCHEMA, "experiment": "viewer-fixture", "layout": layout,
        "title": "Viewer fixture", "subtitle": "Four layout qualification.",
        "claim_boundary": "Cells remain separate and are never pooled.",
        "cells": cells,
        "findings": [{
            "id": "bounded_finding", "layer": "harness_feedback",
            "epistemic_status": "inference", "priority": "diagnostic",
            "title": "Bounded finding", "summary": "This is an inference.",
            "evidence_cells": [cell["id"] for cell in cells],
        }],
        "exclusions": [{
            "evidence": "excluded", "reason": "infrastructure_incomplete",
            "summary": "Not a behavioral sample.",
        }],
    }


def _run(
    root: Path, run_id: str, source: str, success: bool | None,
    *, result: dict | None = None,
) -> SimpleNamespace:
    root.mkdir(parents=True)
    (root / "result.json").write_text(json.dumps(result or {
        "official_success": success, "decisions": 37, "model_calls": 61,
        "wall_duration_s": 3661,
    }), encoding="utf-8")
    return SimpleNamespace(
        root=root, run_id=run_id, source_commit=source,
        official_success=success, control_validity="control-valid",
        experiment="viewer-fixture",
    )


def _metrics(_root: Path) -> dict:
    return {
        "context_peak": 41000, "images_mean": 13.25,
        "cache_token_ratio": 0.9,
        "cost": {"total_cost": 12.5, "currency": "CNY"},
    }


def test_experiment_view_validates_all_four_layout_contracts() -> None:
    definition = _definition()
    manifests = [
        _manifest("single_episode", [_cell("a", "cell_a")]),
        _manifest("persistent_rounds", [_cell("a", "cell_a")]),
        _manifest("diagnostic_set", [
            _cell("a", "cell_a"), _cell("b", "cell_b"),
        ]),
        _manifest("matrix", [
            _cell("a", "cell_a", row="seed_0", column="control"),
            _cell("b", "cell_b", row="seed_0", column="treatment"),
        ]),
    ]
    assert all(validate_experiment_view(value, definition) == () for value in manifests)

    two_single = _manifest("single_episode", [
        _cell("a", "cell_a"), _cell("b", "cell_b"),
    ])
    assert "layout single_episode: expected exactly one cell" in validate_experiment_view(
        two_single, definition,
    )
    incomplete_matrix = _manifest("matrix", [_cell("a", "cell_a", row="seed_0")])
    assert any("cells[0].column" in issue for issue in validate_experiment_view(
        incomplete_matrix, definition,
    ))


def test_experiment_view_rejects_unsafe_and_ambiguous_manifests() -> None:
    definition = _definition()
    base = _manifest("diagnostic_set", [
        _cell("a", "cell_a"), _cell("b", "cell_b"),
    ])
    unsafe = copy.deepcopy(base)
    unsafe["title"] = "<style>body{display:none}</style>"
    assert any("HTML/CSS markup is forbidden" in issue for issue in validate_experiment_view(
        unsafe, definition,
    ))
    duplicate = copy.deepcopy(base)
    duplicate["cells"][1]["id"] = "a"
    assert any("duplicate a" in issue for issue in validate_experiment_view(
        duplicate, definition,
    ))
    conflict = copy.deepcopy(base)
    conflict["exclusions"][0]["evidence"] = "cell_a"
    assert any("accepted/excluded conflict" in issue for issue in validate_experiment_view(
        conflict, definition,
    ))
    unknown = copy.deepcopy(base)
    unknown["cells"][1]["accepted_evidence"] = "missing"
    assert any("unknown key missing" in issue for issue in validate_experiment_view(
        unknown, definition,
    ))
    illegal = copy.deepcopy(base)
    illegal["layout"] = "custom_html"
    assert any("layout: expected one of" in issue for issue in validate_experiment_view(
        illegal, definition,
    ))


def test_diagnostic_renderer_preserves_order_and_derives_metrics(tmp_path: Path) -> None:
    experiment = tmp_path / "experiments/viewer_fixture"
    experiment.mkdir(parents=True)
    definition = _definition()
    manifest = _manifest("diagnostic_set", [
        _cell("b", "cell_b"), _cell("a", "cell_a"),
    ])
    (experiment / "experiment.yaml").write_text(yaml.safe_dump(definition), encoding="utf-8")
    (experiment / "viewer.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    runs = [
        _run(tmp_path / "runs/run-a", "run-a", "source-a", True),
        _run(tmp_path / "runs/run-b", "run-b", "source-b", False),
    ]
    projection = render_experiment_view(
        experiment / "experiment.yaml", definition, runs, _metrics,
    )
    assert projection.present is True and projection.audit is not None
    assert projection.audit.passed is True
    assert projection.html.index("Cell b") < projection.html.index("Cell a")
    assert projection.html.count("<video controls") == 2
    assert "37" in projection.html and "61" in projection.html
    assert "41.0k" in projection.html and "13.2" in projection.html
    assert "90.0% · CNY 12.50" in projection.html
    assert "run-x" in projection.html
    assert "source-bound" in projection.html
    assert "Chat + action notebook" in projection.html
    assert "Exact chat trace" in projection.html


def test_diagnostic_renderer_exposes_persistent_round_chats(tmp_path: Path) -> None:
    definition = _definition()
    manifest = _manifest("diagnostic_set", [
        _cell("a", "cell_a"), _cell("b", "cell_b"),
    ])
    experiment = tmp_path / "experiments/viewer_fixture"
    experiment.mkdir(parents=True)
    definition_path = experiment / "experiment.yaml"
    definition_path.write_text(yaml.safe_dump(definition), encoding="utf-8")
    (experiment / "viewer.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    session_result = {
        "official_success": None, "decisions": 3, "model_calls": 7,
        "wall_duration_s": 30,
        "rounds": [{
            "official_success": False, "decisions": 3, "model_calls": 7,
            "wall_duration_s": 30, "termination": "policy_finished_episode",
            "policy_state_steps": {"start": 12, "end_exclusive": 15},
        }],
    }
    runs = [
        _run(tmp_path / "runs/run-a", "run-a", "source-a", None, result=session_result),
        _run(tmp_path / "runs/run-b", "run-b", "source-b", False),
    ]
    projection = render_experiment_view(definition_path, definition, runs, _metrics)
    assert projection.audit is not None and projection.audit.passed is True
    assert "0/1 rounds succeeded" in projection.html
    assert "Round 1 chat" in projection.html
    assert "/run/run-a#step-12" in projection.html


def test_missing_run_and_source_mismatch_are_distinct_audit_states(tmp_path: Path) -> None:
    definition = _definition()
    manifest = _manifest("single_episode", [_cell("a", "cell_a")])
    path = tmp_path / "viewer.yaml"
    missing = audit_experiment_view(path, manifest, definition, [])
    assert missing.passed is True
    assert missing.unavailable_run_ids == ("run-a",)

    drifted = _run(tmp_path / "run-a", "run-a", "wrong-source", True)
    mismatch = audit_experiment_view(path, manifest, definition, [drifted])
    assert mismatch.passed is False
    assert mismatch.unavailable_run_ids == ()
    assert "source mismatch" in mismatch.evidence_issues[0]


def test_persistent_renderer_keeps_round_verdicts_without_aggregate(tmp_path: Path) -> None:
    definition = _definition()
    manifest = _manifest("persistent_rounds", [_cell("a", "cell_a")])
    experiment = tmp_path / "experiments/viewer_fixture"
    experiment.mkdir(parents=True)
    definition_path = experiment / "experiment.yaml"
    definition_path.write_text(yaml.safe_dump(definition), encoding="utf-8")
    (experiment / "viewer.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    result = {
        "official_success": None, "decisions": 3, "model_calls": 7,
        "wall_duration_s": 30,
        "rounds": [
            {"official_success": True, "decisions": 1, "model_calls": 2,
             "wall_duration_s": 10, "termination": "official_success"},
            {"official_success": False, "decisions": 2, "model_calls": 5,
             "wall_duration_s": 20, "termination": "round_budget_exhausted"},
        ],
    }
    run = _run(tmp_path / "runs/run-a", "run-a", "source-a", None, result=result)
    projection = render_experiment_view(definition_path, definition, [run], _metrics)
    assert projection.audit is not None and projection.audit.passed is True
    assert "1/2 rounds succeeded" in projection.html
    assert "Aggregate official verdict is intentionally absent" in projection.html
    assert "/run/run-a#step-" not in projection.html
    assert "unscored" in projection.html

    result["official_success"] = True
    (run.root / "result.json").write_text(json.dumps(result), encoding="utf-8")
    assert audit_experiment_view(
        experiment / "viewer.yaml", manifest, definition, [run],
    ).passed is False


def test_experiment_view_file_audit_and_cli_flag(tmp_path: Path) -> None:
    experiment = tmp_path / "experiments/viewer_fixture"
    experiment.mkdir(parents=True)
    definition = _definition()
    manifest = _manifest("single_episode", [_cell("a", "cell_a")])
    (experiment / "experiment.yaml").write_text(yaml.safe_dump(definition), encoding="utf-8")
    (experiment / "viewer.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    run = _run(tmp_path / "runs/run-a", "run-a", "source-a", True)
    reports = audit_experiment_view_files(tmp_path, [run])
    assert len(reports) == 1 and reports[0].passed is True
    assert build_parser().parse_args(["--audit-experiment-views"]).audit_experiment_views


def test_experiment_view_absence_preserves_generic_experiment_fallback() -> None:
    projection = render_experiment_view(None, _definition(), [], _metrics)
    assert projection.present is False
    assert projection.html == "" and projection.audit is None
