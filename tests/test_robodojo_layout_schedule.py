"""Per-round layout schedule of a persistent RoboDojo session, without Isaac.

Covers the host-safe schedule contract and catalog, the worker's relaunch
branch on a fake plant (call order, records, isolation proof and its
refusals), the bridge's in-process relaunch order and the legacy branch it
must leave untouched.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ora.environment.backends.robodojo import bridge, layout_rounds
from ora.environment.backends.robodojo.bridge import EvaluationLifecycle
from ora.environment.backends.robodojo.session import RobodojoError, RobodojoSession
from ora.environment.benchmarks import robodojo, robodojo_layouts as layouts
from ora.agent.composition import robodojo_layout_compatibility as compatibility
from ora.agent.composition.robodojo_layout_compatibility import verify_scheduled_contract_layouts
from ora.environment.benchmarks import robodojo_layout_catalog as catalog_module
from ora.environment.benchmarks.robodojo_layout_catalog import build_layout_schedule, load_layout_catalog
from ora.orchestrator import robodojo_layout_evaluation as evaluation_layouts
from test_robodojo_backend import Plant

ROOT = Path(__file__).resolve().parents[1]
TASK = "stack_bowls"
LAYOUT_IDS = (21, 37, 51)  # fixture-Assets ordinals, independent of the reviewed pools
INSTRUCTION = "Stack the three bowls together."


def _layout(ordinal: int) -> dict:
    return {
        "Rigid": {"bowl": [{"label": f"bowl{index}", "category": "bowl", "category_idx": 10,
                            "default_pos": [0.1 * index - 0.1 + ordinal / 1000, -0.1, 0.79],
                            "default_ori": [1.0, 0.0, 0.0, 0.0]} for index in range(3)]},
        "Geometry": {"camera_stand": [{"category_idx": 0, "default_pos": [0.0, -0.47, 0.715]}]},
        "Table": {"default": "material_0122"},
    }


def _assets(tmp_path: Path) -> Path:
    assets = tmp_path / "assets"
    folder = assets / "Eval_Layout/RoboDojo/arx_x5/0"
    folder.mkdir(parents=True)
    for ordinal in range(55):
        (folder / f"{TASK}_{ordinal}.json").write_text(json.dumps(_layout(ordinal), sort_keys=True))
    # Another task module: never part of the stack_bowls ordinal.
    (folder / f"{TASK}_random_0.json").write_text("{}")
    return assets


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _schedule(assets: Path, ids=LAYOUT_IDS) -> dict:
    return {
        "schema": layouts.LAYOUT_SCHEDULE_SCHEMA, "eval_seed": 0, "env_seed_policy": "layout_id",
        "rounds": [{"layout_id": ordinal, "layout_file": layouts.layout_file_for(TASK, 0, ordinal),
                    "layout_sha256": _sha(assets / layouts.layout_file_for(TASK, 0, ordinal))}
                   for ordinal in ids],
        "selection": {"catalog": layouts.LAYOUT_CATALOG_PATH, "catalog_sha256": "c" * 64},
    }


def _round0(schedule: dict) -> dict:
    first = schedule["rounds"][0]
    return {"layout_id": first["layout_id"], "layout_file": first["layout_file"],
            "layout_sha256": first["layout_sha256"]}


# --- host-safe schedule contract ---------------------------------------------

def test_a_schedule_names_one_distinct_official_layout_per_round(tmp_path):
    schedule = _schedule(_assets(tmp_path))
    validated = layouts.validate_layout_schedule(schedule, task=TASK, eval_seed=0, rounds=3, **_round0(schedule))
    assert validated == schedule and validated is not schedule
    assert layouts.layout_cell_tag(validated) == "l21.37.51" and layouts.layout_cell_tag(None) == "l0"
    assert layouts.scheduled_layout(validated, 2)["layout_id"] == 51
    with pytest.raises(robodojo.RoboDojoBenchmarkError, match="outside"):
        layouts.scheduled_layout(validated, 3)


def _drift(name):
    def change(schedule):
        rounds = schedule["rounds"]
        if name == "repeat":
            rounds[1] = dict(rounds[0])
        elif name == "same_bytes":
            rounds[2]["layout_sha256"] = rounds[1]["layout_sha256"]
        elif name == "random_task_file":
            rounds[1]["layout_file"] = "Eval_Layout/RoboDojo/arx_x5/0/stack_bowls_random_37.json"
        elif name == "other_seed":
            rounds[1]["layout_file"] = "Eval_Layout/RoboDojo/arx_x5/1/stack_bowls_37.json"
        elif name == "catalog":
            schedule["selection"]["catalog"] = "another-catalog.json"
        elif name == "policy":
            schedule["env_seed_policy"] = "eval_seed"
        elif name == "single_round":
            del rounds[1:]
        elif name == "extra_field":
            rounds[0]["layout_note"] = "x"
    return change


@pytest.mark.parametrize("name", ["repeat", "same_bytes", "random_task_file", "other_seed", "catalog",
                                  "policy", "single_round", "extra_field"])
def test_a_malformed_or_repeating_schedule_is_refused(tmp_path, name):
    schedule = _schedule(_assets(tmp_path))
    round0 = _round0(schedule)
    _drift(name)(schedule)
    with pytest.raises(robodojo.RoboDojoBenchmarkError):
        layouts.validate_layout_schedule(schedule, task=TASK, eval_seed=0, **round0)


def test_round0_fields_and_session_length_must_match_the_schedule(tmp_path):
    schedule = _schedule(_assets(tmp_path))
    with pytest.raises(robodojo.RoboDojoBenchmarkError, match="rounds\\[0\\]"):
        layouts.validate_layout_schedule(schedule, task=TASK, eval_seed=0,
                                         **{**_round0(schedule), "layout_id": 0})
    with pytest.raises(robodojo.RoboDojoBenchmarkError, match="per session round"):
        layouts.validate_layout_schedule(schedule, task=TASK, eval_seed=0, rounds=2, **_round0(schedule))


def test_runtime_admission_takes_one_official_layout_or_a_schedule_that_starts_on_it(tmp_path):
    """EXP14A g002: an unscheduled one-round runtime may name any official ordinal whose file it binds."""
    from test_robodojo_admission import _runtime

    schedule = _schedule(_assets(tmp_path))
    runtime = {**_runtime(), "task_variant": TASK, "native_action_limit": 800, **_round0(schedule)}
    robodojo.validate_runtime_extension(runtime)
    for layout_id in (37, -1, True, "21"):
        with pytest.raises(robodojo.RoboDojoBenchmarkError, match="official layout ordinal"):
            robodojo.validate_runtime_extension({**runtime, "layout_id": layout_id})
    robodojo.validate_runtime_extension({**runtime, "layout_schedule": schedule})
    with pytest.raises(robodojo.RoboDojoBenchmarkError, match="rounds\\[0\\]"):
        robodojo.validate_runtime_extension({**runtime, "layout_id": 37, "layout_schedule": schedule})


def test_selected_layout_follows_the_upstream_ordinal_and_ignores_the_random_task(tmp_path):
    assets = _assets(tmp_path)
    for ordinal in (0, 21, 54):
        assert robodojo.selected_layout(assets, task=TASK, eval_seed=0, layout_id=ordinal).name == f"{TASK}_{ordinal}.json"
    with pytest.raises(robodojo.RoboDojoBenchmarkError, match="unavailable"):
        robodojo.selected_layout(assets, task=TASK, eval_seed=0, layout_id=55)


def test_worker_start_hashes_every_scheduled_file_and_refuses_drift(tmp_path):
    assets = _assets(tmp_path)
    schedule = _schedule(assets)
    files = layout_rounds.scheduled_layout_files(assets, TASK, schedule)
    assert [item["layout_id"] for item in files] == list(LAYOUT_IDS)
    assert layout_rounds.scheduled_layout_files(assets, TASK, None) is None
    (assets / schedule["rounds"][2]["layout_file"]).write_text("{}")
    with pytest.raises(layout_rounds.LayoutScheduleError, match="51"):
        layout_rounds.scheduled_layout_files(assets, TASK, schedule)


# --- reviewed catalog and contract appendix -----------------------------------

#: EXP11A g012 task additions (survey 2026-10-02): default K = 3 order and fallback.
G012_POOLS = {
    "classify_objects": ((49, 54, 37), 50),
    "arrange_largest_number": ((16, 10, 6), 8),
    "organize_table": ((17, 14, 39), 19),
    "build_tower": ((2, 19, 32), 49),
}
REGISTRY = "ora/policy/environments/robodojo/task-contracts/registry-v8.json"

def test_the_reviewed_catalog_schedules_its_tasks_and_refuses_unreviewed_ones(monkeypatch):
    catalog, digest = load_layout_catalog(ROOT)
    assert digest == _sha(ROOT / layouts.LAYOUT_CATALOG_PATH)
    pool = [entry["layout_id"] for entry in catalog["tasks"][TASK]["pool"]]
    # 2026-10-01: 21 and 37 (bowls) and 28 (a bottle) left the pools after the
    # first scene smokes found objects beyond the measured reach envelope.
    assert pool == [23, 29, 51]
    bottles = catalog["tasks"]["put_bottles_into_dustbin"]
    assert bottles["status"] == catalog_module.REVIEWED
    assert [entry["layout_id"] for entry in bottles["pool"]] == [17, 2, 25, 21]
    # EXP11A g012 task additions: default order, then the explicit-only fallback.
    for task, (default, fallback) in G012_POOLS.items():
        assert catalog["tasks"][task]["status"] == catalog_module.REVIEWED
        assert [entry["layout_id"] for entry in catalog["tasks"][task]["pool"]] == [*default, fallback]
    with pytest.raises(robodojo.RoboDojoBenchmarkError, match="no reviewed layout pool"):
        build_layout_schedule("stack_blocks", 2, asset_root=ROOT / "missing-assets", project_root=ROOT)
    # Only layouts 0, 7 and 41 render its reviewed instruction: the pool is those
    # three, layout 0 last (user decision 2026-10-01), with no fallback.
    by_language = catalog["tasks"]["classify_objects_by_language"]
    assert by_language["status"] == catalog_module.REVIEWED
    assert [entry["layout_id"] for entry in by_language["pool"]] == [7, 41, 0]
    with pytest.raises(robodojo.RoboDojoBenchmarkError, match="inside the reviewed pool"):
        build_layout_schedule(TASK, 2, asset_root=ROOT / "missing-assets", project_root=ROOT, layout_ids=[23, 21])
    # A pool still under survey is refused, never replaced by layout 0.
    pending = {**catalog, "tasks": {**catalog["tasks"], "put_bottles_into_dustbin": {
        **bottles, "status": catalog_module.PENDING_REVIEW}}}
    monkeypatch.setattr(catalog_module, "load_layout_catalog", lambda root: (pending, digest))
    with pytest.raises(robodojo.RoboDojoBenchmarkError, match="pending review"):
        build_layout_schedule("put_bottles_into_dustbin", 3, asset_root=ROOT / "missing-assets", project_root=ROOT)


def test_agent_composition_reads_the_same_catalog_and_appendix_schemas():
    """The contract check repeats the catalog names because Agent imports no Environment module."""
    assert compatibility.LAYOUT_CATALOG_PATH == layouts.LAYOUT_CATALOG_PATH
    assert compatibility.CATALOG_SCHEMA == catalog_module.CATALOG_SCHEMA
    assert compatibility.REVIEWED == catalog_module.REVIEWED
    appendix = json.loads((ROOT / "ora/policy/environments/robodojo/task-contracts/reviews/"
                                  "stack_bowls-layout-compatibility-v1.json").read_text())
    assert appendix["schema"] == compatibility.APPENDIX_SCHEMA


def _catalog_schedule(ids=(23, 29, 51), task=TASK) -> dict:
    catalog, digest = load_layout_catalog(ROOT)
    pool = {entry["layout_id"]: entry for entry in catalog["tasks"][task]["pool"]}
    return {"schema": layouts.LAYOUT_SCHEDULE_SCHEMA, "eval_seed": 0, "env_seed_policy": "layout_id",
            "rounds": [dict(pool[ordinal]) for ordinal in ids],
            "selection": {"catalog": layouts.LAYOUT_CATALOG_PATH, "catalog_sha256": digest}}


def test_the_contract_appendix_extends_the_reviewed_record_to_every_scheduled_layout():
    registry = json.loads((ROOT / REGISTRY).read_text())
    entry = registry["tasks"][TASK]
    schedule = _catalog_schedule()
    runtime = {"task_variant": TASK, "eval_seed": 0, "layout_schedule": schedule}
    disclosure = verify_scheduled_contract_layouts(runtime, record_path=entry["record"],
                                                   project_root=ROOT)
    assert [item["layout_id"] for item in disclosure["scheduled_layouts"]] == [23, 29, 51]
    # The stack_bowls evaluator reads bowl poses only: no scheduled evaluator asset.
    assert disclosure["evaluator_dependencies"] == []
    appendix = json.loads((ROOT / disclosure["appendix"]).read_text())
    rows = {row["ordinal"]: row for row in appendix["layouts"]}
    # Each row records its bowl model's metadata hash as audit evidence; the
    # scheduled layouts keep layout 0's model 10.
    assert {rows[ordinal]["bowl_metadata"]["path"] for ordinal in (0, 23, 29, 51)} == {
        "Object/RoboDojo/Rigid/bowl/00010/metadata.json"}
    assert {rows[ordinal]["bowl_metadata"]["sha256"] for ordinal in (0, 23, 29, 51)} == {
        "e6c6fd201b871a8c57661c027b5ad5d50f71a9b62543515b9bb26ee1513493a5"}
    # Every eval_seed-0 layout was audited: same instruction, three upright
    # bowls of one model, no clutter and the reviewed fixtures.
    assert sorted(rows) == list(range(55)) and all(row["compatible"] for row in rows.values())
    assert max(row["max_bowl_tilt_deg"] for row in rows.values()) <= 0.3
    assert rows[0]["layout_sha256"] == json.loads((ROOT / entry["record"]).read_text())["layout"]["layout_sha256"]
    with pytest.raises(RuntimeError, match="this reviewed contract record"):
        # The appendix names the record it extends by path; any other record is refused.
        verify_scheduled_contract_layouts(runtime, record_path=entry["record"] + ".other",
                                          project_root=ROOT)
    drifted = deepcopy(schedule)
    drifted["selection"]["catalog_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="another layout catalog"):
        verify_scheduled_contract_layouts({**runtime, "layout_schedule": drifted}, record_path=entry["record"],
                                          project_root=ROOT)


def test_put_bottles_scheduled_layouts_disclose_the_bottle_metadata_the_evaluator_reads():
    """The review-v2 record binds layout 0's bottle models; scheduled layouts bring their own."""
    registry = json.loads((ROOT / REGISTRY).read_text())
    entry = registry["tasks"]["put_bottles_into_dustbin"]
    runtime = {"task_variant": "put_bottles_into_dustbin", "eval_seed": 0,
               "layout_schedule": _catalog_schedule((17, 2, 25), "put_bottles_into_dustbin")}
    disclosure = verify_scheduled_contract_layouts(runtime, record_path=entry["record"],
                                                   project_root=ROOT)
    models = {item["path"].split("/")[-2]: item["sha256"] for item in disclosure["evaluator_dependencies"]}
    assert models == {
        "00000": "97aa47eadb3d1251bb5841c0eb0c467be48d9ac287bdbb58fbbcdbd0948f584c",
        "00002": "4330cd08f46bdf328c3bffa94e217a3185a57506664358787613515c69f7c9cc",
        "00005": "04add3df2e5f5359b034ddaf27098ab974dcab577657fd220fe74b9923b5e6e0",
        "00025": "7a9ab0555f8f30caba6e97d147540e6cf0f48027b934724f996e1bcffe0173dd",
        "00041": "9ad91a5a25c7e8ce0746e68f1e1849cbb656a819bee06f2c20e7c1e08ff40972",
        "00050": "9e9144fa52a31c225770da306cf76a1a196cccb3469ce5a79a336d54c0f5dd2e",
        "00059": "eb0df723799dd005896e6522af6006fa37b9f25a8fc08d18bcffbf59f970b8e0",
    }
    fallback = {**runtime, "layout_schedule": _catalog_schedule((21, 2, 25), "put_bottles_into_dustbin")}
    paths = {item["path"].split("/")[-2] for item in verify_scheduled_contract_layouts(
        fallback, record_path=entry["record"], project_root=ROOT)["evaluator_dependencies"]}
    assert "00060" in paths


@pytest.mark.parametrize("task", G012_POOLS)
def test_g012_appendices_extend_each_record_to_its_pool_and_bind_every_object_metadata(task):
    """Each record binds every layout-0 object's metadata; every audited row binds its own layout's."""
    default, fallback = G012_POOLS[task]
    entry = json.loads((ROOT / REGISTRY).read_text())["tasks"][task]
    record = json.loads((ROOT / entry["record"]).read_text())
    runtime = {"task_variant": task, "eval_seed": 0, "layout_schedule": _catalog_schedule(default, task)}
    disclosure = verify_scheduled_contract_layouts(runtime, record_path=entry["record"],
                                                   project_root=ROOT)
    assert [item["layout_id"] for item in disclosure["scheduled_layouts"]] == list(default)
    appendix = json.loads((ROOT / disclosure["appendix"]).read_text())
    assert appendix["contract_layer"] == {"path": entry["layer"], "sha256": record["contract_layer"]["sha256"]}
    assert appendix["instruction"]["text"] == record["instruction"]["text"]
    rows = {row["ordinal"]: row for row in appendix["layouts"]}
    assert sorted(rows) == sorted([0, *default, fallback]) and all(row["compatible"] for row in rows.values())
    # The layout-0 row reproduces the record's own asset set, byte for byte.
    assert rows[0]["layout_sha256"] == record["layout"]["layout_sha256"]
    assert ({(item["path"], item["sha256"]) for item in rows[0]["evaluator_asset_dependencies"]}
            == {(item["path"], item["sha256"]) for item in record["evaluator_dependencies"] if item["root"] == "assets"})
    scheduled = {(item["path"], item["sha256"]) for ordinal in default
                 for item in rows[ordinal]["evaluator_asset_dependencies"]}
    assert {(item["path"], item["sha256"]) for item in disclosure["evaluator_dependencies"]} == scheduled
    # The fallback is admissible only by explicit selection inside the pool.
    explicit = {**runtime, "layout_schedule": _catalog_schedule((fallback, *default[1:]), task)}
    verify_scheduled_contract_layouts(explicit, record_path=entry["record"], project_root=ROOT)
    with pytest.raises(robodojo.RoboDojoBenchmarkError, match="inside the reviewed pool"):
        build_layout_schedule(task, 3, asset_root=ROOT / "missing-assets", project_root=ROOT,
                              layout_ids=[0, *default[1:]])


def test_the_by_language_pool_is_the_three_layouts_that_render_its_reviewed_instruction():
    """Round 3 replays the reviewed layout 0: an exception scoped to this pool entry, not a rule change."""
    task = "classify_objects_by_language"
    entry = json.loads((ROOT / REGISTRY).read_text())["tasks"][task]
    record = json.loads((ROOT / entry["record"]).read_text())
    runtime = {"task_variant": task, "eval_seed": 0, "layout_schedule": _catalog_schedule((7, 41, 0), task)}
    disclosure = verify_scheduled_contract_layouts(runtime, record_path=entry["record"],
                                                   project_root=ROOT)
    assert [item["layout_id"] for item in disclosure["scheduled_layouts"]] == [7, 41, 0]
    assert disclosure["scheduled_layouts"][-1]["layout_sha256"] == record["layout"]["layout_sha256"]
    appendix = json.loads((ROOT / disclosure["appendix"]).read_text())
    rows = {row["ordinal"]: row for row in appendix["layouts"]}
    assert sorted(rows) == [0, 7, 41] and all(row["compatible"] for row in rows.values())
    for row in rows.values():  # the worker refuses a round whose instruction differs from round 0
        assert row["rendered_instruction"] == record["instruction"]["text"] and row["instruction_equals_reviewed"]
        assert row["baskets_left_middle_right"] and row["fixtures_equal_reviewed_layout"]
    assert ({(item["path"], item["sha256"]) for item in rows[0]["evaluator_asset_dependencies"]}
            == {(item["path"], item["sha256"]) for item in record["evaluator_dependencies"] if item["root"] == "assets"})
    catalog, _digest = load_layout_catalog(ROOT)
    assert catalog["tasks"][task]["review"]["fallback"].startswith("None is possible")
    # No other layout is in the pool, so no other selection is admissible.
    with pytest.raises(robodojo.RoboDojoBenchmarkError, match="inside the reviewed pool"):
        build_layout_schedule(task, 3, asset_root=ROOT / "missing-assets", project_root=ROOT, layout_ids=[7, 41, 1])


def test_g012_appendix_rows_hold_the_layout_dependent_contract_facts():
    """The contract sentences that could change with the layout hold on every audited row."""
    def rows(task):
        path = ROOT / f"ora/policy/environments/robodojo/task-contracts/reviews/{task}-layout-compatibility-v1.json"
        return json.loads(path.read_text())["layouts"]

    for row in rows("classify_objects"):  # a separate basket for each of three categories
        assert row["three_distinct_categories"] and len(row["object_groups"]) == 3
        assert not any("action_camera/1" in note for note in row["mass_load_notes"])
    for row in rows("arrange_largest_number"):  # the k-th largest digit on the k-th pad from world -X
        assert row["one_pad_per_piece"] and row["pads_ordered_from_minus_x"] and row["digits_unique"]
        assert len(row["pieces"]) == 4 and {piece["texture_group"] for piece in row["pieces"]} == {0}
    for row in rows("organize_table"):  # keyboard larger than the frame; reset facing satisfies the checks
        assert row["keyboard_larger_than_frame"] and row["reset_mouse_local_y_from_world_x_deg"] < 30
        assert row["objects"]["garage"]["category_idx"] == 8 and row["figurine_xy_tolerance_m"] == 0.02
        assert 0.225 < row["alarm_rest_height_over_drawer_m"] < 0.3
        assert all(row["fixtures"][name]["equal_reviewed_layout"] for name in ("frame", "cube_cushion"))
    for row in rows("build_tower"):  # one upright pair on each side; the long boards where layout 0 has them
        assert row["models_as_reviewed"] and row["pairs_on_opposite_sides"] and row["boards_at_reviewed_pose"]
        assert row["max_upright_block_tilt_deg"] < 1 and row["roof_local_y_from_up_deg"] < 1


def test_preflight_rehashes_the_disclosed_scheduled_evaluator_assets(tmp_path):
    from ora.agent.composition.robodojo_task_contract import verify_contract_dependencies

    registry = json.loads((ROOT / REGISTRY).read_text())
    entry = registry["tasks"]["put_bottles_into_dustbin"]
    record = json.loads((ROOT / entry["record"]).read_text())
    source, assets = tmp_path / "source", tmp_path / "assets"
    for dependency in [{"root": "robodojo", "path": record["source"]["source_path"]},
                       *record["evaluator_dependencies"]]:
        target = (source if dependency["root"] == "robodojo" else assets) / dependency["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"fixture")
    fixture_sha = hashlib.sha256(b"fixture").hexdigest()
    record_bytes = json.dumps({**record, "source": {**record["source"], "file_sha256": fixture_sha},
                               "evaluator_dependencies": [{**item, "sha256": fixture_sha}
                                                          for item in record["evaluator_dependencies"]]}).encode()
    (tmp_path / "record.json").write_bytes(record_bytes)
    extra = assets / "Object/RoboDojo/Rigid/bottle/00005/metadata.json"
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_bytes(b"fixture")
    disclosure = {"backend": "robodojo", "delivered": True, "record": "record.json",
                  "record_sha256": hashlib.sha256(record_bytes).hexdigest(),
                  "layout_compatibility": {"evaluator_dependencies": [
                      {"root": "assets", "path": "Object/RoboDojo/Rigid/bottle/00005/metadata.json",
                       "sha256": fixture_sha}]}}
    checked = verify_contract_dependencies(disclosure, project_root=tmp_path, source_root=source, asset_root=assets)
    assert checked["checked_dependencies"] == len(record["evaluator_dependencies"]) + 2
    extra.write_bytes(b"drifted")
    with pytest.raises(RuntimeError, match="bottle/00005"):
        verify_contract_dependencies(disclosure, project_root=tmp_path, source_root=source, asset_root=assets)


ASSET_ROOT = Path.home() / "robodojo-runtime/assets/Assets"


@pytest.mark.skipif(not (ASSET_ROOT / "Eval_Layout/RoboDojo/arx_x5/0").is_dir(), reason="pinned RoboDojo Assets absent")
@pytest.mark.parametrize("task, default", [(TASK, [23, 29, 51]), ("put_bottles_into_dustbin", [17, 2, 25]),
                                           *((task, list(pool[0])) for task, pool in G012_POOLS.items()),
                                           ("classify_objects_by_language", [7, 41, 0])])
def test_the_catalog_and_appendix_match_the_pinned_assets(task, default):
    schedule = build_layout_schedule(task, 3, asset_root=ASSET_ROOT, project_root=ROOT)
    assert [entry["layout_id"] for entry in schedule["rounds"]] == default
    appendix = json.loads((ROOT / f"ora/policy/environments/robodojo/task-contracts/reviews/"
                                  f"{task}-layout-compatibility-v1.json").read_text())
    for row in appendix["layouts"]:
        assert _sha(ASSET_ROOT / row["layout_file"]) == row["layout_sha256"]
        for dependency in [*row.get("evaluator_asset_dependencies", []), *filter(None, [row.get("bowl_metadata")])]:
            assert _sha(ASSET_ROOT / dependency["path"]) == dependency["sha256"]


# --- reach footprint of the layout objects (scene-smoke targets) ---------------

def _box(x: float, y: float, z: float) -> list[list[float]]:
    return [[sx * x, sy * y, sz * z] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]


def _reach_assets(tmp_path: Path, layout: dict) -> Path:
    from ora.environment.benchmarks.robodojo_layout_reach import object_metadata_path

    assets = tmp_path / "reach-assets"
    for category, index, box in (("bowl", 10, _box(0.06, 0.06, 0.025)), ("bottle", 25, _box(0.03, 0.03, 0.1))):
        path = assets / object_metadata_path(category, index)
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"geometry": {"aligned_bbox": {"vertices": box}}}))
    (assets / "layout.json").write_text(json.dumps(layout))
    return assets


def test_reach_objects_are_the_free_rigid_bodies_the_policy_moves(tmp_path):
    from ora.environment.benchmarks.robodojo_layout_reach import layout_reach_objects

    bowl = {"label": "bowl0", "category_idx": 10, "default_pos": [0.3, -0.05, 0.79], "default_ori": [1, 0, 0, 0]}
    layout = {
        "Rigid": {"bowl": [bowl], "ball": [{"label": "ball", "type": "cluttered", "category_idx": 3,
                                            "default_pos": [0.0, 0.3, 0.8]}]},
        "Geometry": {"dustbin": [{"label": "dustbin", "category_idx": 0, "default_pos": [-0.63, -0.1, 0.4]}],
                     "basket": [{"label": "basket", "category_idx": 0, "default_pos": [0.0, 0.0, 0.77]}]},
        "Articulation": {"drawer": [{"label": "drawer", "category_idx": 0, "default_pos": [0.5, 0.2, 0.77]}]},
        "Dynamic": {"conveyor": [{"label": "conveyor", "category_idx": 0, "default_pos": [0.0, 0.6, 0.4]}]},
    }
    assets = _reach_assets(tmp_path, layout)
    objects = layout_reach_objects(assets, "layout.json")
    assert [item["label"] for item in objects] == ["bowl0"]
    assert objects[0]["metadata"] == {
        "path": "Object/RoboDojo/Rigid/bowl/00010/metadata.json",
        "sha256": _sha(assets / "Object/RoboDojo/Rigid/bowl/00010/metadata.json")}
    # An in-scope object the pinned assets cannot size fails closed.
    layout["Rigid"]["bowl"].append({**bowl, "label": "bowl1", "category_idx": 11})
    (assets / "layout.json").write_text(json.dumps(layout))
    with pytest.raises(robodojo.RoboDojoBenchmarkError, match="bowl/00011/metadata.json is missing"):
        layout_reach_objects(assets, "layout.json")


def test_the_footprint_follows_the_asset_box_scale_and_orientation(tmp_path):
    from ora.environment.benchmarks.robodojo_layout_reach import footprint

    bowl = footprint({"default_pos": [0.3, -0.05, 0.79], "default_ori": [1, 0, 0, 0]},
                     {"geometry": {"aligned_bbox": {"vertices": _box(0.06, 0.06, 0.025)}}})
    assert bowl["half_extents_m"] == [0.06, 0.06]
    assert [point["bearing_deg"] for point in bowl["points"]] == [0, 45, 90, 135, 180, 225, 270, 315]
    assert all(np.hypot(point["xy_m"][0] - 0.3, point["xy_m"][1] + 0.05) == pytest.approx(0.06, abs=1e-6)
               for point in bowl["points"])
    lying = {"default_pos": [0.0, 0.0, 0.795], "default_ori": [np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0]}
    bottle = {"geometry": {"aligned_bbox": {"vertices": _box(0.03, 0.03, 0.1)}}}
    # Lying on its side (90 degrees about x): the 0.2 m axis is horizontal along world y.
    points = {point["point"]: point["xy_m"] for point in footprint(lying, bottle)["points"]}
    assert footprint(lying, bottle)["half_extents_m"] == [0.03, 0.1]
    assert points["edge0"] == pytest.approx([0.03, 0.0]) and points["edge2"] == pytest.approx([0.0, -0.1])
    assert points["edge4"] == pytest.approx([-0.03, 0.0]) and points["edge6"] == pytest.approx([0.0, 0.1])
    # The layout scale stretches the asset box before it is turned.
    assert footprint({**lying, "scale": [2.0, 1.0, 1.0]}, bottle)["half_extents_m"] == [0.06, 0.1]
    # A further 90-degree yaw lays the long axis along world x.
    yawed = {**lying, "default_ori": [0.5, 0.5, 0.5, 0.5]}
    ends = sorted(point["xy_m"][0] for point in footprint(yawed, bottle)["points"])
    assert ends[0] == pytest.approx(-0.1) and ends[-1] == pytest.approx(0.1)


def test_footprint_base_distances_measure_the_far_rim_from_one_arm():
    from ora.environment.benchmarks.robodojo_layout_reach import footprint, footprint_base_distances

    instance = {"default_pos": [0.3, -0.05, 0.79], "default_ori": [1, 0, 0, 0]}
    item = {"position_m": instance["default_pos"],
            "footprint": footprint(instance, {"geometry": {"aligned_bbox": {"vertices": _box(0.06, 0.06, 0.025)}}})}
    distances = footprint_base_distances(item)
    assert distances["centre_nearer_base_m"] == pytest.approx(0.40)
    assert distances["footprint_arm"] == "right" and distances["footprint_nearer_base_m"] == pytest.approx(0.46)


def test_the_reach_footprint_uses_the_worker_arm_base_mounts():
    from ora.environment.backends.robodojo import motion_checks
    from ora.environment.benchmarks.robodojo_layout_reach import X5_BASE_MOUNTS_XY_M

    assert sorted(X5_BASE_MOUNTS_XY_M.values()) == sorted(mount[:2] for mount in motion_checks.X5_BASE_MOUNTS_M)
    assert X5_BASE_MOUNTS_XY_M["left"][0] < X5_BASE_MOUNTS_XY_M["right"][0]


@pytest.mark.skipif(not (ASSET_ROOT / "Eval_Layout/RoboDojo/arx_x5/0").is_dir(), reason="pinned RoboDojo Assets absent")
def test_the_catalog_bowl_reach_annotation_is_the_centre_case_of_the_footprint():
    from ora.environment.benchmarks.robodojo_layout_reach import (
        footprint_base_distances, layout_reach_objects,
    )

    review = load_layout_catalog(ROOT)[0]["tasks"][TASK]
    annotated = {0: review["reference_layout"],
                 **{int(key): value for key, value in review["review"]["layouts"].items()}}
    for layout_id, annotation in annotated.items():
        distances = [footprint_base_distances(item) for item in
                     layout_reach_objects(ASSET_ROOT, layouts.layout_file_for(TASK, 0, layout_id))]
        centre = max(item["centre_nearer_base_m"] for item in distances)
        assert round(centre, 3) == annotation["max_nearer_base_distance_m"]
        # The 0.12 m bowl's far rim lies 0.06 m beyond its centre.
        assert all(item["footprint_nearer_base_m"] >= item["centre_nearer_base_m"] + 0.055 for item in distances)


# --- worker relaunch branch on a fake plant -------------------------------------

class RelaunchPlant(Plant):
    """A plant whose scheduled rounds relaunch: new clock, new instance names."""

    def __init__(self, assets: Path):
        super().__init__()
        self.assets = assets
        self.step_lim = 800
        self.loaded = None
        self.events: list[tuple] = []
        self.counter = 0
        self.obs_manager = SimpleNamespace(instruction=[INSTRUCTION])

    def _layout(self, ordinal):
        return json.loads((self.assets / layouts.layout_file_for(TASK, 0, ordinal)).read_text())

    def reset(self, seed):
        from ora.environment.benchmarks.robodojo import build_scene_registration

        assert len(seed) == 1 and seed[0] in self.harness_layout_ids
        self.events.append(("reset", seed[0]))
        self.loaded = seed[0]
        self.take_action_cnt, self.reward_registered = [0], True
        self.success, self.end_flag, self._official_terminal = [True], [False], None
        self.resets = getattr(self, "resets", 0) + 1
        instances = {}
        for section, kind in (("Rigid", "rigid"), ("Geometry", "geometry")):
            for category, items in self._layout(seed[0])[section].items():
                for _ in items:
                    self.counter += 1
                    instances[f"{category}_{self.counter}"] = kind
        self.scene_registration = build_scene_registration(
            instances, evaluator_registration={"check_list": True, "final_check_list": False,
                                               "trigger_check_list": False},
            native_action_limit=self.step_lim)
        self.robot_manager.reset_home()

    def relaunch_for_layout(self, layout_id):
        self.events.append(("release_and_close", self.loaded))
        before = self.sim.sim.current_time_step_index
        time_before = before * 0.004
        previous_offset = self.harness_simulation_time_offset_s
        self.harness_simulation_time_offset_s = previous_offset + time_before
        self.sim.sim.current_time_step_index = 320  # a new simulation context restarts its clock
        previous = self.loaded
        self.reset([layout_id])
        return {"schema": layout_rounds.RELAUNCH_SCHEMA, "previous_layout_id": previous, "layout_id": layout_id,
                "sim_closed": True, "stage_replaced": True,
                "stage_before": {"stage": "1:anon:a", "env_prim_present": True},
                "stage_after_close": {"stage": "2:anon:b", "env_prim_present": False},
                "physics_clock_before_close": before, "physics_clock_after_reset": 320,
                "simulation_time_before_close_s": time_before,
                "previous_simulation_time_offset_s": previous_offset,
                "simulation_time_offset_s": self.harness_simulation_time_offset_s,
                "simulation_time_after_reset_s": 320 * 0.004, "wall_s": 41.0}

    def loaded_layout_state(self):
        layout = self._layout(self.loaded)
        return {"env_seed": self.loaded, "layout": layout, "object_counts": layouts.layout_object_counts(layout)}


def _scheduled_session(tmp_path, *, plant=None, schedule=None):
    assets = _assets(tmp_path)
    schedule = schedule or _schedule(assets)
    plant = plant or RelaunchPlant(assets)
    first = schedule["rounds"][0]
    source = {"schema": "ora.robodojo-source-manifest.v1",
              "source_revisions": {"robodojo": "1" * 40},
              "layout_file": {"path": str(assets / first["layout_file"]), "sha256": first["layout_sha256"]},
              "layout_schedule": layout_rounds.scheduled_layout_files(assets, TASK, schedule)}
    session = RobodojoSession(
        config={"task": TASK, "workspace_bounds_m": {"x": [-1, 1], "y": [-1, 1], "z": [0, 2]},
                **_round0(schedule), "layout_schedule": schedule},
        output_root=tmp_path / "native",
        environment_factory=lambda config: (plant, SimpleNamespace(close=lambda: None), source),
    )
    session._calibration = lambda name: (np.array([[50, 0, 5], [0, 50, 4], [0, 0, 1]]), np.eye(4))
    return session, plant, assets


def _end_round(session):
    session.abort("qualification_boundary")
    verdict = session.persist_official_verdict()
    (session.output_root / "official-verdict.json").unlink()
    return verdict


def test_each_scheduled_round_starts_a_relaunched_new_layout(tmp_path):
    session, plant, _assets_root = _scheduled_session(tmp_path)
    start = session.start()
    try:
        assert plant.events == [("reset", 21)] and plant.harness_layout_ids == LAYOUT_IDS
        assert start["schema"] == layout_rounds.START_RESET_V2_SCHEMA
        assert (start["layout_id"], start["env_seed"], start["physics_clock_epoch"],
                start["scene_reset_mode"], start["seed"]) == (21, 21, 0, "initial_launch", 0)
        assert start["layout_verification"]["passed"] is True
        assert start["simulation_time_offset_s"] == 0.0 and plant.harness_simulation_time_offset_s == 0.0
        assert start["layout_verification"]["checks"] == {
            "layout_matches_round_schedule": True, "scene_objects_match_layout": True}
        # A scheduled observation id names the round, never the layout id.
        observation, _ = session.observe(tmp_path / "observations")
        assert observation["observation_id"] == "robodojo:stack_bowls:round-0:observation-0000"
        verdict = _end_round(session)
        assert verdict["schema"] == layout_rounds.VERDICT_V3_SCHEMA
        assert (verdict["layout_id"], verdict["env_seed"], verdict["physics_clock_epoch"]) == (21, 21, 0)
        clocks = [start["physics_steps_before_policy"]]
        for index, ordinal in enumerate(LAYOUT_IDS[1:], start=1):
            reset = session.restart_round(index)
            assert plant.events[-2:] == [("release_and_close", LAYOUT_IDS[index - 1]), ("reset", ordinal)]
            assert reset["schema"] == layout_rounds.ROUND_RESET_V2_SCHEMA
            assert (reset["layout_id"], reset["env_seed"], reset["physics_clock_epoch"], reset["seed"]) == (
                ordinal, ordinal, index, 0)
            assert reset["scene_reset_mode"] == "simulation_relaunch" and reset["relaunch"]["layout_id"] == ordinal
            assert "pre_reset_home_joint_write" not in reset
            isolation = reset["round_isolation"]
            assert isolation["passed"] is True and isolation["failures"] == []
            assert sorted(isolation["checks"]) == sorted(layout_rounds.SCHEDULED_ISOLATION_CHECKS)
            assert isolation["measured"]["layout_object_counts"] == {
                "articulation": 0, "garment": 0, "geometry": 1, "rigid": 3}
            assert json.loads((session.output_root / f"round-resets/round-{index:04d}.json").read_text()) == reset
            assert session.observe(tmp_path / "observations")[0]["observation_id"].startswith(
                f"robodojo:stack_bowls:round-{index}:")
            clocks.append(reset["physics_steps_before_policy"])
            verdict = _end_round(session)
            assert (verdict["layout_id"], verdict["physics_clock_epoch"], verdict["round_index"]) == (
                ordinal, index, index)
        # Each relaunch starts a new simulation clock: monotone only within an epoch,
        # while the annotator offset accumulates every closed epoch's own time.
        assert clocks == [500, 320, 320]
        offsets = [json.loads((session.output_root / f"round-resets/round-{index:04d}.json").read_text())[
            "simulation_time_offset_s"] for index in (1, 2)]
        assert offsets == pytest.approx([2.0, 2.0 + 1.28])
        with pytest.raises(RobodojoError, match="outside the layout schedule"):
            session.restart_round(3)
    finally:
        session.close()


@pytest.mark.parametrize("fault, check", [
    ("wrong_layout_loaded", "layout_matches_round_schedule"),
    ("object_missing", "scene_objects_match_layout"),
    ("instruction_drift", "task_instruction_unchanged"),
    ("camera_rebound", "plant_embodiment_unchanged"),
    ("stage_survived", "simulation_relaunched"),
    ("clock_not_the_new_reset", "simulation_relaunched"),
    ("offset_not_chained", "simulation_relaunched"),
    ("file_changed", "layout_matches_round_schedule"),
])
def test_a_scheduled_round_that_is_not_the_planned_new_scene_is_refused(tmp_path, fault, check):
    session, plant, assets = _scheduled_session(tmp_path)
    session.start()
    try:
        _end_round(session)
        relaunch = plant.relaunch_for_layout
        if fault == "wrong_layout_loaded":
            plant.loaded_layout_state = lambda: {"env_seed": 37, "layout": plant._layout(21),
                                                 "object_counts": {"articulation": 0, "garment": 0,
                                                                   "geometry": 1, "rigid": 3}}
        elif fault == "object_missing":
            plant.loaded_layout_state = lambda: {"env_seed": 37, "layout": plant._layout(37),
                                                 "object_counts": {"articulation": 0, "garment": 0,
                                                                   "geometry": 1, "rigid": 2}}
        elif fault == "instruction_drift":
            def relaunch_and_reword(layout_id):
                result = relaunch(layout_id)
                plant.obs_manager.instruction = ["Stack the bowls."]
                return result
            plant.relaunch_for_layout = relaunch_and_reword
        elif fault == "camera_rebound":
            def relaunch_and_move_camera(layout_id):
                result = relaunch(layout_id)
                plant.camera_manager.cameras[0][0].prim_path = "/World/envs/env_0/Camera_9/optical"
                return result
            plant.relaunch_for_layout = relaunch_and_move_camera
        elif fault == "stage_survived":
            # The flags say replaced; the measured stage says the old scene is still there.
            plant.relaunch_for_layout = lambda layout_id: {
                **relaunch(layout_id), "stage_after_close": {"stage": "1:anon:a", "env_prim_present": True}}
        elif fault == "clock_not_the_new_reset":
            plant.relaunch_for_layout = lambda layout_id: {**relaunch(layout_id), "physics_clock_after_reset": 9999}
        elif fault == "offset_not_chained":
            # The offset the cameras would be compared with is not previous + time at close.
            def relaunch_with_wrong_offset(layout_id):
                record = relaunch(layout_id)
                plant.harness_simulation_time_offset_s = record["simulation_time_offset_s"] = 0.0
                return record
            plant.relaunch_for_layout = relaunch_with_wrong_offset
        elif fault == "file_changed":
            (assets / layouts.layout_file_for(TASK, 0, 37)).write_text(json.dumps(_layout(99)))
        with pytest.raises(RobodojoError, match="did not start an isolated new layout"):
            session.restart_round(1)
        record = json.loads((session.output_root / "round-resets/round-0001.json").read_text())
        assert record["round_isolation"]["checks"][check] is False
        assert session.fault_reason == "robodojo_round_reset_isolation_failed"
    finally:
        session.close()


def test_round0_must_load_its_scheduled_layout(tmp_path):
    session, plant, _assets_root = _scheduled_session(tmp_path)
    plant.loaded_layout_state = lambda: {"env_seed": 21, "layout": plant._layout(0),
                                         "object_counts": {"articulation": 0, "garment": 0,
                                                           "geometry": 1, "rigid": 3}}
    try:
        with pytest.raises(RobodojoError, match="round 0 did not load its scheduled layout"):
            session.start()
        assert session.fault_reason == "robodojo_start_failed"
    finally:
        session.close()


def test_fold_clothes_cannot_follow_a_layout_schedule(tmp_path):
    schedule = _schedule(_assets(tmp_path))
    with pytest.raises(robodojo.RoboDojoBenchmarkError, match="not relaunch-safe"):
        layouts.validate_layout_schedule(schedule, task="fold_clothes", eval_seed=0, **_round0(schedule))
    lifecycle = object.__new__(EvaluationLifecycle)
    lifecycle.task_name, lifecycle.harness_layout_ids, lifecycle.sim = "fold_clothes", (21, 37), object()
    with pytest.raises(RobodojoError, match="not relaunch-safe"):
        lifecycle.relaunch_for_layout(37)


def test_an_unscheduled_session_binds_its_one_layout_and_refuses_a_malformed_one(tmp_path):
    """EXP14A g002: the unscheduled round 0 is the RunPlan's layout_id (0 when unset)."""
    session = RobodojoSession(config={"task": TASK, "layout_id": 21}, output_root=tmp_path / "native",
                              environment_factory=lambda config: None)
    assert session.layout_id == 21 and session.layout_schedule is None
    assert RobodojoSession(config={"task": TASK}, output_root=tmp_path / "default",
                           environment_factory=lambda config: None).layout_id == 0
    for layout_id in (-1, True, "21"):
        with pytest.raises(RobodojoError, match="layout_id"):
            RobodojoSession(config={"task": TASK, "layout_id": layout_id}, output_root=tmp_path / "bad",
                            environment_factory=lambda config: None)


# --- bridge relaunch order -------------------------------------------------------

def test_relaunch_releases_capture_closes_sim_and_stage_then_resets_on_the_new_layout(monkeypatch):
    events = []

    class ParentTask:
        def close(self):
            events.append("task_env_close")
            self.sim = None

        def reset(self, seed=None, options=None):
            events.append(("upstream_reset", self.sim is None))
            self.sim = SimpleNamespace(sim=SimpleNamespace(current_time_step_index=300, current_time=1.2))
            self.reward_manager.check_list = [[]]

    class Lifecycle(EvaluationLifecycle, ParentTask):
        def setup_scene(self):
            events.append("settle")

        def run_reward(self):
            self.reward_manager.check_list[0].append("official_predicate")

    stages = iter([{"stage": "1:anon:a", "env_prim_present": True}, {"stage": "2:anon:b", "env_prim_present": False}])
    monkeypatch.setattr(bridge, "_stage_state", lambda: next(stages))
    monkeypatch.setattr(bridge, "_cuda_memory", lambda: {"allocated_mib": 1})
    lifecycle = object.__new__(Lifecycle)
    lifecycle.num_envs, lifecycle.step_lim = 1, 800
    lifecycle.sim = SimpleNamespace(sim=SimpleNamespace(current_time_step_index=9000, current_time=36.0))
    lifecycle.harness_simulation_time_offset_s = 1.5
    lifecycle.harness_layout_ids, lifecycle.harness_loaded_layout_id = (21, 37, 51), 21
    lifecycle.scene_manager = SimpleNamespace(layout_manager=SimpleNamespace(
        set_saved_layout=lambda env_idx, layout: events.append(("bind_layout", layout["ordinal"])),
        instance_type_by_env=[{"bowl_9": "rigid"}], get_scene_object=lambda env_idx, name: object()))
    lifecycle.seed_manager = SimpleNamespace(get_seed_scene_info=lambda seed: {"ordinal": seed})
    lifecycle.obs_manager = SimpleNamespace(reset=lambda: events.append("observation_reset"))
    lifecycle.robot_manager = SimpleNamespace(set_origin_endpose=lambda: None, set_robot_init_state=lambda: None)
    lifecycle.reward_manager = SimpleNamespace(check_list=[[]], final_check_list=[[]], trigger_check_list=[[]],
                                               init_state=lambda: None)
    lifecycle.capture_manager = SimpleNamespace(tiled_cameras=[object()])
    lifecycle._release_capture_products = lambda: (events.append("release_capture_products"),
                                                   setattr(lifecycle.capture_manager, "tiled_cameras", []))
    # An in-place reset cannot change the layout: CUDA keeps the round-0 objects.
    with pytest.raises(RobodojoError, match="needs relaunch_for_layout"):
        lifecycle.reset(seed=[37])
    with pytest.raises(RobodojoError, match="scheduled"):
        lifecycle.relaunch_for_layout(22)
    record = lifecycle.relaunch_for_layout(37)
    assert events == ["release_capture_products", "observation_reset", "task_env_close", ("bind_layout", 37),
                      ("upstream_reset", True), "observation_reset", "settle"]
    assert lifecycle.env_seeds == [37] and lifecycle.harness_loaded_layout_id == 37
    assert record["schema"] == layout_rounds.RELAUNCH_SCHEMA
    assert (record["previous_layout_id"], record["layout_id"], record["physics_clock_after_reset"]) == (21, 37, 300)
    assert record["stage_after_close"]["env_prim_present"] is False
    assert (record["sim_closed"], record["stage_replaced"], record["physics_clock_before_close"]) == (True, True, 9000)
    # The annotator offset is set before the new context exists: every earlier epoch's time.
    assert (record["simulation_time_before_close_s"], record["previous_simulation_time_offset_s"],
            record["simulation_time_offset_s"], record["simulation_time_after_reset_s"]) == (36.0, 1.5, 37.5, 1.2)
    assert lifecycle.harness_simulation_time_offset_s == 37.5


def test_relaunch_refuses_when_the_old_scene_survives_close(monkeypatch):
    class ParentTask:
        def close(self):
            self.sim = None

    class Lifecycle(EvaluationLifecycle, ParentTask):
        pass

    monkeypatch.setattr(bridge, "_stage_state", lambda: {"stage": "1:anon:a", "env_prim_present": True})
    lifecycle = object.__new__(Lifecycle)
    lifecycle.sim = SimpleNamespace(sim=SimpleNamespace(current_time_step_index=1, current_time=0.004))
    lifecycle.harness_layout_ids = (21, 37)
    lifecycle.obs_manager = SimpleNamespace(reset=lambda: None)
    lifecycle.capture_manager = SimpleNamespace(tiled_cameras=[])
    with pytest.raises(RobodojoError, match="survived the relaunch close"):
        lifecycle.relaunch_for_layout(37)


# --- camera time across a relaunch (smoke b2-robodojo-smoke-20261001T050812Z-ab2ac6f1) ----------------

ROUND0_FINAL_TIME = 3.536000167950988   # last round-0 synchronization, physics step 884
ROUND1_RESET_TIME = 3.3760001603513956  # first observe after the relaunch, physics step 844
ANNOTATOR_TIME = 6.912000328302383      # what every camera's IsaacReadSimulationTime reported


def _recorded_cameras():
    """Three cameras rendering the relaunched epoch: accumulated annotator time, exact poses."""
    state = {"renders": 0}

    def render():
        state["renders"] += 1

    manager = SimpleNamespace(num_envs=1, tiled_cameras=[
        SimpleNamespace(_render_product_path=f"/camera/{index}",
                        _harness_camera_params=SimpleNamespace(get_data=lambda: {
                            "cameraViewTransform": np.eye(4).reshape(-1).tolist()}),
                        _harness_simulation_time=SimpleNamespace(get_data=lambda: {"simulationTime": ANNOTATOR_TIME}),
                        _harness_reference_time=SimpleNamespace(get_data=lambda: {
                            "referenceTimeNumerator": 241 + state["renders"], "referenceTimeDenominator": 30}))
        for index in range(3)])
    return manager, render, state


def test_relaunched_cameras_match_their_epoch_clock_through_the_offset(tmp_path):
    from ora.environment.backends.robodojo.camera_capture import synchronize_render_products

    manager, render, state = _recorded_cameras()
    common = dict(manager=manager, render=render, physics_clock=lambda: 844,
                  simulation_time=lambda: ROUND1_RESET_TIME,
                  physical_poses=lambda: [np.diag([1., -1., -1., 1.])] * 3, output_root=tmp_path)
    # Without the offset this is the smoke's failure: 6.912 never equals 3.376.
    with pytest.raises(TimeoutError, match="current physics frame"):
        synchronize_render_products(**common)
    unscheduled = json.loads((tmp_path / "render-synchronization.jsonl").read_text().splitlines()[0])
    assert "simulation_time_offset_s" not in unscheduled and unscheduled["matched"] is False
    state["renders"] = 0
    record = synchronize_render_products(**common, simulation_time_offset_s=ROUND0_FINAL_TIME)
    assert record["matched"] is True and state["renders"] == 2
    assert record["simulation_time_offset_s"] == ROUND0_FINAL_TIME
    assert record["samples"][-1]["cameras"][0]["simulation_time_s"] == ANNOTATOR_TIME
    # The host join accepts exactly this record against the reset chain, and nothing else.
    resets = [{"simulation_time_offset_s": 0.0}, {"simulation_time_offset_s": ROUND0_FINAL_TIME}]
    assert evaluation_layouts.render_time_errors([record], resets) == []
    assert evaluation_layouts.render_time_errors([unscheduled], resets) == ["robodojo_render_time_offset_invalid"]
    shifted = {**record, "samples": [{"cameras": [{"simulation_time_s": ANNOTATOR_TIME + 0.04}]}]}
    assert evaluation_layouts.render_time_errors([shifted], resets) == ["robodojo_render_time_offset_invalid"]


def test_the_bridge_passes_the_offset_only_to_a_scheduled_session(monkeypatch):
    from ora.environment.backends.robodojo import camera_capture

    seen = []
    monkeypatch.setattr(camera_capture, "synchronize_render_products", lambda **kwargs: seen.append(kwargs))
    lifecycle = object.__new__(EvaluationLifecycle)
    lifecycle.capture_manager, lifecycle.render, lifecycle.harness_output_root = object(), lambda: None, Path(".")
    lifecycle.camera_manager = SimpleNamespace(cameras=[[]])
    lifecycle.obs_manager = SimpleNamespace(get_obs=lambda env_idx_list: {0: {"vision": {}}})
    lifecycle.get_obs()
    lifecycle.harness_simulation_time_offset_s = ROUND0_FINAL_TIME
    lifecycle.get_obs()
    assert "simulation_time_offset_s" not in seen[0]
    assert seen[1]["simulation_time_offset_s"] == ROUND0_FINAL_TIME
