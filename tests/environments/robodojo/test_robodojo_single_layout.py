"""One RoboDojo episode on an official layout k of evaluation seed 0 (EXP14A g002, S3).

An unscheduled, one-round RoboDojo runtime may name any official layout
ordinal whose canonical file it binds; the worker resets round 0 (and an
in-place round reset) on that layout; the lane gate of that layout is tagged
``l<k>``; and the default builder composes the episode only where the reviewed
layout-portability table extends the task's registry-v8 contract to that
layout's bytes. Tasks whose instruction the layout generates take it from the
scene-smoke readback their lane gate certifies. Layout schedules of persistent
sessions are unchanged (tests/environments/robodojo/test_robodojo_layout_schedule.py). No simulator:
sessions, plants and qualification records are fakes; tests marked
``needs_assets`` read the pinned Assets layouts when present.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ora.agent.composition.locked_assets import locked_paths
from ora.agent.composition.robodojo_layout_compatibility import (
    INSTRUCTION_LAYOUT_READBACK, LAYOUT_PORTABILITY_PATH, PORTABILITY_SCHEMA,
)
from ora.agent.composition.robodojo_task_contract import REGISTRY_PATH, resolve_reviewed_contract
from ora.environment.backends.robodojo.session import RobodojoError, RobodojoSession
from ora.environment.benchmarks import robodojo
from ora.environment.benchmarks.robodojo_layouts import layout_cell_tag, layout_file_for
from ora.orchestrator import robodojo_lane_qualification as lane, robodojo_spec
from ora.orchestrator import robodojo_revalidation as rv
from ora.orchestrator.spec import resolve_run_spec

ROOT = Path(__file__).resolve().parents[3]
REGISTRY = json.loads((ROOT / REGISTRY_PATH).read_text())["tasks"]
TABLE = json.loads((ROOT / LAYOUT_PORTABILITY_PATH).read_text())
ASSETS = Path(os.environ.get("ROBODOJO_ASSET_ROOT", str(Path.home() / "robodojo-runtime/assets/Assets")))
needs_assets = pytest.mark.skipif(not (ASSETS / "Eval_Layout/RoboDojo/arx_x5/0").is_dir(),
                                  reason="pinned RoboDojo Assets absent")
#: The EXP14A RSI tasks whose instruction gen_instruction builds from the layout's objects.
READBACK_TASKS = {"stack_blocks_by_language", "classify_objects_by_language", "general_pickup"}
QUALIFICATION = {"path": "outputs/single-layout-fixture/qualification/revalidation/gate.json", "sha256": "0" * 64}
EXPERIMENT = "single-layout-fixture"


@pytest.fixture(autouse=True)
def _registered_execution_host(monkeypatch):
    """RoboDojo runtimes bind the marker's registered host; tests run on the fixture host."""
    monkeypatch.setenv("ORA_EXECUTION_HOST", "test-host")


@pytest.fixture(autouse=True)
def _restore_service_endpoints():
    saved = {variable: os.environ.pop(variable, None) for _, variable in robodojo_spec.SERVICE_VARIABLES}
    yield
    for variable, value in saved.items():
        if value is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = value


# --- host-safe admission -------------------------------------------------------------------------

def test_admission_takes_one_official_layout_whose_canonical_file_the_runtime_binds():
    from tests.environments.robodojo.test_robodojo_admission import _runtime

    base = _runtime()
    robodojo.validate_runtime_extension(base)  # layout 0 exactly as before
    single = {**base, "layout_id": 3, "layout_file": layout_file_for("arrange_largest_number", 0, 3)}
    robodojo.validate_runtime_extension(single)
    for change in ({"layout_file": base["layout_file"]}, {"layout_id": -1}, {"layout_id": True},
                   {"layout_file": layout_file_for("arrange_largest_number", 0, 4)}):
        with pytest.raises(robodojo.RoboDojoBenchmarkError, match="official layout ordinal"):
            robodojo.validate_runtime_extension({**single, **change})


def test_an_ordinal_the_pinned_assets_do_not_hold_is_refused(tmp_path, monkeypatch):
    from ora.orchestrator.robodojo_runtime_config import runtime_configuration

    folder = tmp_path / "Assets/Eval_Layout/RoboDojo/arx_x5/0"
    folder.mkdir(parents=True)
    for ordinal in range(5):
        (folder / f"stack_bowls_{ordinal}.json").write_text(json.dumps({"ordinal": ordinal}))
    monkeypatch.setenv("ROBODOJO_ASSET_ROOT", str(tmp_path / "Assets"))
    runtime = runtime_configuration("0", "stack_bowls", 4)
    assert (runtime["layout_id"], runtime["layout_file"]) == (4, layout_file_for("stack_bowls", 0, 4))
    assert runtime["layout_sha256"] == hashlib.sha256((folder / "stack_bowls_4.json").read_bytes()).hexdigest()
    assert runtime_configuration("0", "stack_bowls")["layout_id"] == 0
    with pytest.raises(robodojo.RoboDojoBenchmarkError, match="unavailable"):
        runtime_configuration("0", "stack_bowls", 5)


# --- the worker ------------------------------------------------------------------------------------

def test_the_worker_binds_round_zero_to_the_runplans_one_layout():
    from ora.environment.backends.robodojo.bridge import manifest_bindings

    assert manifest_bindings({"action_codec": "motion_action_v3"}, "stack_bowls")["layout_id"] == 0
    assert manifest_bindings({"action_codec": "motion_action_v3", "layout_id": 3}, "stack_bowls")["layout_id"] == 3
    for layout_id in (-1, True, "3"):
        with pytest.raises(RobodojoError, match="layout_id"):
            manifest_bindings({"action_codec": "motion_action_v3", "layout_id": layout_id}, "stack_bowls")


def test_an_unscheduled_session_resets_and_restarts_in_place_on_its_one_layout(tmp_path):
    from tests.environments.robodojo.test_robodojo_backend import Plant, _fake_source_manifest

    class SeedPlant(Plant):
        def reset(self, seed):
            assert len(seed) == 1 and seed[0] in self.harness_layout_ids
            self.seeds = [*getattr(self, "seeds", []), seed[0]]
            super().reset([0])

    plant = SeedPlant()
    session = RobodojoSession(
        config={"workspace_bounds_m": {"x": [-1, 1], "y": [-1, 1], "z": [0, 2]}, "layout_id": 3},
        output_root=tmp_path / "native",
        environment_factory=lambda config: (plant, SimpleNamespace(close=lambda: None),
                                            _fake_source_manifest(tmp_path)))
    session._calibration = lambda name: (np.array([[50, 0, 5], [0, 50, 4], [0, 0, 1]]), np.eye(4))
    try:
        start = session.start()
        assert plant.harness_layout_ids == (3,) and plant.seeds == [3]
        assert start["layout_id"] == 3 and "layout_verification" not in start
        session.abort("qualification_boundary")
        session.persist_official_verdict()
        (session.output_root / "official-verdict.json").unlink()
        reset = session.restart_round(1, seed=0)
        assert reset["layout_id"] == 3 and reset["round_isolation"]["passed"] is True
        assert plant.seeds == [3, 3]
    finally:
        session.close()


# --- lane gate tags ----------------------------------------------------------------------------------

def test_the_lane_gate_and_cell_name_the_single_layout(monkeypatch, tmp_path):
    from tests.environments.robodojo.test_robodojo_lane_qualification import _configured

    assert layout_cell_tag(None) == "l0" and layout_cell_tag(None, 3) == "l3"
    assert layout_cell_tag({"rounds": [{"layout_id": 23}, {"layout_id": 29}]}, 7) == "l23.29"
    with pytest.raises(robodojo.RoboDojoBenchmarkError, match="one layout ordinal"):
        layout_cell_tag(None, -1)
    workspace = tmp_path / "outputs/lane-fixture"
    monkeypatch.setattr(lane, "experiment_workspace", lambda name: workspace)
    assert lane.gate_path("lane-fixture", _configured()) == workspace / "gates/stack_bowls-l0-cuda0.json"
    assert lane.gate_path("lane-fixture", _configured(layout_id=3)) == workspace / "gates/stack_bowls-l3-cuda0.json"
    assert lane.gate_path("lane-fixture", _configured(layout_id=3, physics_device="cpu", cuda_visible_devices="4")) \
        == workspace / "gates/stack_bowls-l3-cpu-gpu4.json"


def test_a_single_layout_lane_reads_only_smokes_of_that_layout(monkeypatch, tmp_path):
    from tests.environments.robodojo.test_robodojo_lane_qualification import _configured, _smoke_dir

    workspace = tmp_path / "outputs/lane-fixture"
    monkeypatch.setattr(lane, "experiment_workspace", lambda name: workspace)
    layout0, layout3 = _configured(), _configured(layout_id=3, layout_file="f3", layout_sha256="s3")
    _smoke_dir(workspace, "robodojo-scene-smoke-l0", runtime=layout0, mtime=5)
    chosen = _smoke_dir(workspace, "robodojo-scene-smoke-l3", runtime=layout3, mtime=1)
    assert lane.smoke_candidates("lane-fixture", layout3) == [chosen]
    assert [path.parent.name for path in lane.smoke_candidates("lane-fixture", layout0)] == [
        "robodojo-scene-smoke-l0"]


def test_the_lane_entry_points_pass_the_single_layout(monkeypatch, capsys, tmp_path):
    from ora.orchestrator import robodojo_scene_smoke as scene

    calls = []
    monkeypatch.setattr(lane, "revalidate", lambda task, **kwargs: calls.append(kwargs) or tmp_path / "g.json")
    monkeypatch.setattr(scene, "smoke", lambda task, **kwargs: calls.append(kwargs) or {"passed": True})
    common = ["--gpu", "0", "--experiment", "lane-fixture", "--layout", "3"]
    assert lane.main(["smoke", "stack_bowls", *common, "--yes"]) == 0
    assert lane.main(["revalidate", "stack_bowls", *common, "--card", str(tmp_path / "card.json")]) == 0
    assert [(call["layout_id"], call["rounds"], call["layout_schedule"]) for call in calls] == [
        (3, 1, None), (3, 1, None)]
    seen = []
    monkeypatch.setattr(robodojo_spec, "qualification_runtime", lambda task, **kwargs: seen.append(kwargs) or {})
    lane._runtime("stack_bowls", gpu="0", resource_profile=None, rounds=1, layout_schedule=None)
    lane._runtime("stack_bowls", gpu="0", resource_profile=None, rounds=1, layout_schedule=None, layout_id=3)
    assert "layout_id" not in seen[0] and seen[1]["layout_id"] == 3


# --- the reviewed layout-portability table ---------------------------------------------------------

def test_the_portability_table_extends_registry_v8_records_unchanged_from_layout_zero():
    assert TABLE["schema"] == PORTABILITY_SCHEMA and TABLE["registry"] == REGISTRY_PATH
    assert (TABLE["robodojo_revision"], TABLE["assets_revision"]) == (
        robodojo.REVISIONS["robodojo"], robodojo.REVISIONS["assets"])
    assert LAYOUT_PORTABILITY_PATH in locked_paths(ROOT)
    assert set(TABLE["tasks"]) <= set(REGISTRY)
    for task, entry in TABLE["tasks"].items():
        record = json.loads((ROOT / REGISTRY[task]["record"]).read_text())
        assert entry["record"] == REGISTRY[task]["record"]
        assert entry["contract_layer_sha256"] == record["contract_layer"]["sha256"]
        assert [row["layout_id"] for row in entry["layouts"]] == list(range(10))
        assert all(row["layout_file"] == layout_file_for(task, 0, row["layout_id"]) for row in entry["layouts"])
        assert entry["layouts"][0] == {"layout_id": 0, "layout_file": record["layout"]["layout_file"],
                                       "layout_sha256": record["layout"]["layout_sha256"]}
    assert {task for task, entry in TABLE["tasks"].items()
            if entry["instruction"] == INSTRUCTION_LAYOUT_READBACK} == READBACK_TASKS


@needs_assets
def test_every_audited_layout_is_the_pinned_upstream_ordinal_and_bytes():
    for task, entry in TABLE["tasks"].items():
        for row in entry["layouts"]:
            path = robodojo.selected_layout(ASSETS, task=task, eval_seed=0, layout_id=row["layout_id"])
            assert path.relative_to(ASSETS.resolve()).as_posix() == row["layout_file"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row["layout_sha256"]


# --- contract delivery -----------------------------------------------------------------------------------

def _contract_runtime(task, layout_id, **changes):
    record = json.loads((ROOT / REGISTRY[task]["record"]).read_text())
    rows = {row["layout_id"]: row for row in TABLE["tasks"].get(task, {}).get("layouts", [])}
    row = rows.get(layout_id, {"layout_file": layout_file_for(task, 0, layout_id), "layout_sha256": "e" * 64})
    return {"task_variant": task, "required_revision": record["source"]["robodojo_revision"],
            "required_asset_revision": record["layout"]["assets_revision"], "eval_seed": 0,
            "layout_id": layout_id, "layout_file": row["layout_file"], "layout_sha256": row["layout_sha256"],
            "expected_task_language": record["instruction"]["text"], **changes}


def _resolve(runtime):
    return resolve_reviewed_contract(runtime, project_root=ROOT, disclosure={"backend": "robodojo"})


@pytest.mark.parametrize("task", sorted(TABLE["tasks"]))
def test_an_audited_record_is_delivered_unchanged_on_each_audited_layout(task):
    layer0, disclosure0 = _resolve(_contract_runtime(task, 0))
    assert "layout_portability" not in disclosure0
    for layout_id in range(1, 10):
        runtime = _contract_runtime(task, layout_id)
        layer, disclosure = _resolve(runtime)
        assert layer == layer0 == REGISTRY[task]["layer"] and disclosure["delivered"] is True
        assert {key: value for key, value in disclosure.items() if key != "layout_portability"} == disclosure0
        portability = disclosure["layout_portability"]
        assert (portability["table"], portability["layout_id"], portability["layout_sha256"]) == (
            LAYOUT_PORTABILITY_PATH, layout_id, runtime["layout_sha256"])
        assert portability["instruction_source"] == TABLE["tasks"][task]["instruction"]
        other = {**runtime, "expected_task_language": "A layout-generated instruction."}
        if task in READBACK_TASKS:
            assert _resolve(other)[0] == layer
        else:
            with pytest.raises(RuntimeError, match="task instruction differs"):
                _resolve(other)


def test_an_unaudited_task_layout_or_byte_is_refused_its_contract():
    with pytest.raises(RuntimeError, match="hang_mugs is not audited for layout portability"):
        _resolve(_contract_runtime("hang_mugs", 1))
    with pytest.raises(RuntimeError, match="layout 10 of stack_bowls is not an audited portable layout"):
        _resolve(_contract_runtime("stack_bowls", 10))
    with pytest.raises(RuntimeError, match="differs from its audited file or bytes"):
        _resolve(_contract_runtime("stack_bowls", 2, layout_sha256="f" * 64))
    with pytest.raises(RuntimeError, match="differs from its audited file or bytes"):
        _resolve(_contract_runtime("stack_bowls", 2, layout_file=layout_file_for("stack_bowls", 0, 3)))
    # Layout 0 keeps its reviewed binding: other bytes are refused as before.
    with pytest.raises(RuntimeError, match="differs from runtime layout_sha256"):
        _resolve(_contract_runtime("stack_bowls", 0, layout_sha256="f" * 64))


# --- RunSpec composition -----------------------------------------------------------------------------------

def _spec(task, **kwargs):
    kwargs.setdefault("gpu", "0")
    kwargs.setdefault("infrastructure_qualification", QUALIFICATION)
    return robodojo_spec.build_spec(task, experiment=EXPERIMENT, **kwargs)


@needs_assets
def test_a_fixed_instruction_task_composes_its_layout_with_the_record_instruction():
    record = robodojo_spec.task_contract("stack_bowls")["record"]
    assert _spec("stack_bowls", layout_id=0) == _spec("stack_bowls")
    spec = _spec("stack_bowls", layout_id=3)
    runtime, row = spec["runtime"], TABLE["tasks"]["stack_bowls"]["layouts"][3]
    assert spec["config"] == "stack_bowls-l3-s0" and "session" not in spec
    assert {key: runtime[key] for key in ("layout_id", "layout_file", "layout_sha256")} == row
    assert runtime["expected_task_language"] == spec["task_instruction"] == record["instruction"]["text"]
    plan = resolve_run_spec(spec, root=ROOT)
    assert plan["runtime"]["layout_id"] == 3
    assert plan["policy"]["task_contract"]["layout_portability"]["layout_id"] == 3
    qualification = robodojo_spec.qualification_runtime("stack_bowls", gpu="0", layout_id=3)
    assert qualification["layout_id"] == 3
    assert qualification["expected_task_language"] == record["instruction"]["text"]


@needs_assets
def test_a_single_layout_spec_refuses_what_its_layout_cannot_carry():
    with pytest.raises(ValueError, match="needs the scene-smoke lane gate of that layout"):
        _spec("stack_bowls", layout_id=3, infrastructure_qualification={"card": QUALIFICATION,
                                                                        "task": QUALIFICATION})
    with pytest.raises(ValueError, match="one round"):
        _spec("stack_bowls", layout_id=3, rounds=3)
    with pytest.raises(ValueError, match="not audited for layout portability"):
        _spec("hang_mugs", layout_id=1)
    with pytest.raises(ValueError, match="not an audited portable layout"):
        _spec("stack_bowls", layout_id=10)
    with pytest.raises(ValueError, match="non-negative official ordinal"):
        robodojo_spec.runtime_for("stack_bowls", gpu="0", layout_id=-2)
    with pytest.raises(ValueError, match="renders its reviewed instruction on every layout"):
        robodojo_spec.runtime_for("stack_bowls", gpu="0", layout_id=3, task_instruction="Another.")


def _gate(tmp_path, monkeypatch, *, task, layout_id, instruction):
    monkeypatch.setattr(robodojo_spec, "WORKSPACE_ROOT", tmp_path)
    record = tmp_path / f"outputs/{EXPERIMENT}/qualification/revalidation/{task}-{layout_id}.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps({"schema": rv.SCHEMA, "task_instruction": instruction,
                                  "configuration": {"task_variant": task, "layout_id": layout_id}}))
    return {"path": record.relative_to(tmp_path).as_posix(),
            "sha256": hashlib.sha256(record.read_bytes()).hexdigest()}


@needs_assets
def test_a_layout_generated_instruction_comes_from_the_lane_gate_readback(tmp_path, monkeypatch):
    qualification = robodojo_spec.qualification_runtime("general_pickup", gpu="0", layout_id=2)
    assert qualification["layout_id"] == 2 and "expected_task_language" not in qualification
    instruction = "Pick up the pearl seashell by 10 cm."
    gate = _gate(tmp_path, monkeypatch, task="general_pickup", layout_id=2, instruction=instruction)
    spec = _spec("general_pickup", layout_id=2, infrastructure_qualification=gate)
    assert spec["runtime"]["expected_task_language"] == spec["task_instruction"] == instruction
    assert spec["config"] == "general_pickup-l2-s0"
    plan = resolve_run_spec(spec, root=ROOT)
    assert plan["policy"]["task_contract"]["layout_portability"]["instruction_source"] == "layout_readback"
    other = _gate(tmp_path, monkeypatch, task="general_pickup", layout_id=3, instruction=instruction)
    with pytest.raises(ValueError, match="binds no scene-smoke instruction of layout 2"):
        _spec("general_pickup", layout_id=2, infrastructure_qualification=other)
    with pytest.raises(ValueError, match="unreadable"):
        _spec("general_pickup", layout_id=2, infrastructure_qualification={**gate, "sha256": "1" * 64})
    with pytest.raises(ValueError, match="not a card/task pair"):
        _spec("general_pickup", layout_id=2, infrastructure_qualification={"card": gate, "task": gate})


def test_the_spec_entry_point_passes_the_single_layout(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(robodojo_spec, "build_spec", lambda task, **kwargs: seen.update(kwargs) or {"spec": task})
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"path": "outputs/x.json", "sha256": "0" * 64}))
    assert robodojo_spec.main(["spec", "stack_bowls", "--experiment", EXPERIMENT, "--gpu", "0", "--layout", "3",
                               "--qualification-reference", str(gate)]) == 0
    assert seen["layout_id"] == 3 and seen["rounds"] is None and seen["layout_schedule"] is None


# --- revalidation: the readback binding ------------------------------------------------------------------

def test_a_plan_is_held_to_the_instruction_its_smoke_bound_or_read_back():
    bound, unbound = {"expected_task_language": "Stack."}, {}
    rv.scene_language({"expected_task_language": "Stack."}, bound, "Stack.")
    with pytest.raises(rv.RoboDojoBenchmarkError, match="drift"):
        rv.scene_language({"expected_task_language": "Other."}, bound, "Stack.")
    with pytest.raises(rv.RoboDojoBenchmarkError, match="drift"):
        rv.scene_language({}, bound, "Stack.")
    with pytest.raises(rv.RoboDojoBenchmarkError, match="language mismatch"):
        rv.scene_language({"expected_task_language": "Stack."}, bound, "Read back otherwise.")
    # An unbound smoke certifies its readback, and admits only a plan that carries exactly it.
    rv.scene_language({}, unbound, "Pick up the scissors by 10 cm.")
    rv.scene_language({"expected_task_language": "Pick up the scissors by 10 cm."}, unbound,
                      "Pick up the scissors by 10 cm.")
    with pytest.raises(rv.RoboDojoBenchmarkError, match="language mismatch"):
        rv.scene_language({"expected_task_language": "Pick up the shovel by 10 cm."}, unbound,
                          "Pick up the scissors by 10 cm.")
