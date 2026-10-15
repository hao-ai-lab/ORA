"""Optional full-plan launch binding, inspection recursion and live rechecks."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import ModuleType

import pytest

from ora.orchestrator import spec as runtime


EXPERIMENT = "launch-binding-fixture"
MODULE = "experiments.launch_binding_fixture.run"
COMMIT = "a" * 40
ROOT = Path(__file__).resolve().parents[2]


def _spec() -> dict:
    return {
        "schema": runtime.SCHEMA, "experiment": EXPERIMENT,
        "config": "cell", "milestone": "M0", "task": "synthetic",
        "task_instruction": "Move once.", "seed": 0, "evidence_tier": "diagnostic",
        "output_workspace": f"outputs/{EXPERIMENT}",
        "policy": {"kind": "scripted", "model": "scripted", "provider": "trusted",
                   "reasoning_effort": "none", "actions": [{"x": 1}]},
        "runtime": {"kind": "synthetic", "workspace_bounds_m": {
            "x": [-1, 1], "y": [-1, 1], "z": [0, 2]}},
        "budgets": {"max_robot_decisions": 1, "max_model_calls": 1,
                    "max_sim_steps": 1, "wall_time_s": 10},
        "policy_context": {
            "codex_project": ["ora/policy/planners/native/project-v1.md"],
            "episode_prefix": ["ora/policy/core/thin-react-v1.md"],
            "tool_manual": ["ora/policy/embodiments/aloha-agilex/manual-v1.md"],
        },
        "qualification": {"required": False},
        "launch_admission": {"schema": runtime.LAUNCH_ADMISSION_SCHEMA,
            "mode": "single_attempt_cell", "cohort_id": "fixture", "cell_id": "cell",
            "retry_allowed": False, "gate": f"outputs/{EXPERIMENT}/gates/cell.json"},
    }


@pytest.fixture
def binding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = ModuleType(MODULE)
    monkeypatch.setitem(sys.modules, MODULE, module)
    gate = tmp_path / f"outputs/{EXPERIMENT}/gates/cell.json"
    gate.parent.mkdir(parents=True)
    gate.write_text('{"test_fixture_only":true}\n')
    policy = {"cells": {"cell": {
        "cohort_id": "fixture", "cell_id": "cell", "resource_profile": "fixture",
        "slot_id": "fixture", "gate": f"outputs/{EXPERIMENT}/gates/cell.json",
    }}}
    monkeypatch.setattr(runtime, "_source_launch_policy", lambda *a, **k: copy.deepcopy(policy))
    monkeypatch.setattr(runtime, "resolve_workspace_root", lambda *_: tmp_path)
    monkeypatch.setattr(runtime, "_git", lambda *a, **k: COMMIT if a[0] == "rev-parse" else "")
    # The legacy runtime/gate check is independently covered. Keep the actual
    # ignored-artifact binding and full RunSpec resolver under test here.
    monkeypatch.setattr(runtime, "_validate_launch_gate_record", lambda **_: None)
    return module, gate


def test_full_plan_hook_runs_after_resolution_without_mutating_the_plan(binding):
    module, _gate = binding
    calls = []

    def validate(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        kwargs["plan"]["policy"]["model"] = "hook-local-copy-only"

    module.validate_launch_admission_plan = validate
    plan = runtime.resolve_run_spec(_spec(), root=ROOT)
    assert len(calls) == 1
    assert calls[0]["config"] == "cell" and calls[0]["source_commit"] == COMMIT
    assert calls[0]["plan"] == plan
    assert plan["policy"]["model"] == "scripted"
    assert len(plan["plan_sha256"]) == 64 and plan["confirmation_code"]
    assert plan["policy_context"]["composition_sha256"]
    assert plan["policy_composition"] and plan["source"]["dirty"] is False


@pytest.mark.parametrize("damage", ["seed", "budget", "policy"])
def test_resolver_rejects_validly_encoded_but_unqualified_full_composition(binding, damage):
    module, _gate = binding

    def validate(*, plan, **_):
        if (plan["seed"] != 0 or plan["budgets"]["max_robot_decisions"] != 1
            or plan["policy"]["model"] != "scripted"):
            raise ValueError("fixed composition drift")

    module.validate_launch_admission_plan = validate
    source = _spec()
    if damage == "seed":
        source["seed"] = 1
    elif damage == "budget":
        source["budgets"]["max_robot_decisions"] = 2
    else:
        source["policy"]["model"] = "another-model"
    with pytest.raises(runtime.RunSpecError, match="full-plan binding.*fixed composition drift"):
        runtime.resolve_run_spec(source, root=ROOT)


def test_preview_skips_hook_and_can_be_resolved_inside_the_validator(binding):
    module, _gate = binding
    calls = []
    preview = _spec()
    preview["launch_admission"] = {"schema": runtime.LAUNCH_ADMISSION_SCHEMA, "mode": "disabled"}

    def validate(*, plan, **_):
        calls.append(plan["plan_sha256"])
        expected = runtime.resolve_run_spec(preview, root=ROOT)
        assert expected["seed"] == plan["seed"]

    module.validate_launch_admission_plan = validate
    runtime.resolve_run_spec(preview, root=ROOT)
    assert calls == []
    runtime.resolve_run_spec(_spec(), root=ROOT)
    assert len(calls) == 1


def test_package_without_full_plan_hook_keeps_legacy_admission(binding):
    plan = runtime.resolve_run_spec(_spec(), root=ROOT)
    assert runtime.verify_launch_admission(plan, root=ROOT) == "live"


@pytest.mark.parametrize("failure", [RuntimeError, ValueError, KeyError, TypeError, OSError])
def test_full_plan_hook_errors_fail_closed(binding, failure):
    module, _gate = binding

    def validate(**_):
        raise failure("invalid full composition")

    module.validate_launch_admission_plan = validate
    with pytest.raises(runtime.RunSpecError, match="full-plan binding"):
        runtime.resolve_run_spec(_spec(), root=ROOT)


@pytest.mark.parametrize("validator", [False, lambda **_: False])
def test_noncallable_or_boolean_returning_validator_is_not_a_pass(binding, validator):
    module, _gate = binding
    module.validate_launch_admission_plan = validator
    with pytest.raises(runtime.RunSpecError, match="full-plan binding"):
        runtime.resolve_run_spec(_spec(), root=ROOT)


@pytest.mark.parametrize("snapshot", [False, True])
def test_live_prespawn_rechecks_complete_plan_including_snapshot_fallback(binding, tmp_path, snapshot):
    module, gate = binding
    plan = runtime.resolve_run_spec(_spec(), root=ROOT)
    run_root = tmp_path / "test-run"
    if snapshot:
        target = run_root / "qualification/launch-gate.json"
        target.parent.mkdir(parents=True)
        target.write_bytes(gate.read_bytes())
        gate.write_text('{"requalified_elsewhere":true}\n')
    calls = []

    def validate(*, plan, **_):
        calls.append(plan["seed"])
        if plan["seed"] != 0:
            raise RuntimeError("seed is outside the qualified cell")

    module.validate_launch_admission_plan = validate
    assert runtime.verify_launch_admission(plan, root=ROOT, run_root=run_root) == (
        "snapshot" if snapshot else "live")
    plan["seed"] = 1
    with pytest.raises(runtime.RunSpecError, match="seed is outside"):
        runtime.verify_launch_admission(plan, root=ROOT, run_root=run_root)
    assert calls == [0, 1]


def test_backfill_and_disabled_inspection_never_run_full_plan_hook(binding):
    module, _gate = binding
    plan = runtime.resolve_run_spec(_spec(), root=ROOT)

    def unexpected(**_):
        pytest.fail("historical/inspection verification ran the live full-plan hook")

    module.validate_launch_admission_plan = unexpected
    assert runtime.verify_launch_admission(plan, root=ROOT, admission="backfill") == "live"
    plan["launch_admission"] = {"schema": runtime.LAUNCH_ADMISSION_SCHEMA, "mode": "disabled"}
    assert runtime.verify_launch_admission(plan, root=ROOT) == "absent"
