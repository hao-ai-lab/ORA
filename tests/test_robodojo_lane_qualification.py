"""Maintained RoboDojo lane qualification: card, task, scene smoke and lane gate (integration 0062 U12).

Every producer measures the runtime a RunSpec built by robodojo_spec resolves
to, so the records it writes admit that plan; a layout-scheduled session is
qualified on every layout through the scene smoke and its lane gate. The frozen
experiments/toolchain_b2_robodojo smoke, layout_smoke and revalidate scripts
stay at their launch commit. No simulator: sessions, leases and the checker
are fakes; the scheduled-round tests are the frozen layout smoke's, run on the
maintained copy.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

from ora.environment.benchmarks.robodojo import configuration_binding
from ora.orchestrator import (
    robodojo_lane_qualification as lane, robodojo_layout_smoke as layout_smoke,
    robodojo_revalidation as rv, robodojo_scene_smoke as scene, robodojo_spec,
)
from ora.orchestrator.robodojo_layout_revalidation import SCHEDULE_CHECKS
from ora.orchestrator.spec import resolve_run_spec

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(os.environ.get("ROBODOJO_ASSET_ROOT", str(Path.home() / "robodojo-runtime/assets/Assets")))
needs_assets = pytest.mark.skipif(not (ASSETS / "Eval_Layout/RoboDojo/arx_x5/0").is_dir(),
                                  reason="pinned RoboDojo Assets absent")
QUALIFICATION = {"path": "outputs/lane-fixture/qualification/fixture.json", "sha256": "0" * 64}


@pytest.fixture(autouse=True)
def _registered_execution_host(monkeypatch):
    """RoboDojo runtimes bind the marker's registered host; tests run on the fixture host."""
    monkeypatch.setenv("ORA_EXECUTION_HOST", "test-host")


# --- the runtime every producer measures ---------------------------------------------------------------

@needs_assets
@pytest.mark.parametrize("task, rounds", [("stack_bowls", 1), ("stack_bowls", 3), ("hang_mugs", 1)])
def test_the_qualification_runtime_is_the_runtime_the_plan_resolves_to(task, rounds):
    plan = resolve_run_spec(robodojo_spec.build_spec(
        task, experiment="lane-fixture", gpu="0", infrastructure_qualification=QUALIFICATION,
        rounds=None if rounds == 1 else rounds), root=ROOT)["runtime"]
    runtime = robodojo_spec.qualification_runtime(task, gpu="0", rounds=rounds)
    assert configuration_binding(runtime) == configuration_binding(plan)
    for field in ("public_geometry_descriptor", "expected_task_language", "public_environment_context",
                  "environment_prompt_profile", "resource_binding", "physics_device"):
        assert runtime[field] == plan[field], field
    assert runtime.get("layout_schedule") == plan.get("layout_schedule")
    assert "infrastructure_qualification" not in runtime and "workspace_profile" not in runtime


@needs_assets
def test_the_cli_default_card_runtime_measures_other_bounds_than_a_default_plan():
    """Why the lane producers exist: robodojo_card_qualification without a runtime measures compat bounds."""
    from ora.orchestrator.robodojo_runtime_config import runtime_configuration

    lane_runtime = robodojo_spec.qualification_runtime("stack_bowls", gpu="0")
    compat = runtime_configuration("0", "stack_bowls")
    assert lane_runtime["workspace_bounds_m"] != compat["workspace_bounds_m"]
    assert {key for key, value in configuration_binding(lane_runtime).items()
            if configuration_binding(compat)[key] != value} == {"workspace_bounds_m"}


@needs_assets
def test_a_cpu_lane_runtime_keeps_its_physics_device_and_the_plans_configuration(monkeypatch):
    from ora.environment import definitions

    monkeypatch.setitem(definitions.SPARSE_MOTION_QUALIFIED_PHYSICS_DEVICES, ("motion_action_v3", "robodojo"),
                        ("cuda:0", "cpu"))
    plan = resolve_run_spec(robodojo_spec.build_spec(
        "press_by_number", experiment="lane-fixture", gpu="4", resource_profile="robodojo-gpu4-d1",
        infrastructure_qualification=QUALIFICATION), root=ROOT)["runtime"]
    runtime = robodojo_spec.qualification_runtime("press_by_number", gpu="4", resource_profile="robodojo-gpu4-d1")
    assert runtime["physics_device"] == "cpu"
    assert configuration_binding(runtime) == configuration_binding(plan)


@needs_assets
def test_a_scheduled_session_refuses_a_round_zero_card_task_pair():
    pair = {"card": QUALIFICATION, "task": QUALIFICATION}
    assert robodojo_spec.build_spec("stack_bowls", experiment="lane-fixture", gpu="0",
                                    infrastructure_qualification=pair)
    with pytest.raises(ValueError, match="scene-smoke lane gate"):
        robodojo_spec.build_spec("stack_bowls", experiment="lane-fixture", gpu="0",
                                 infrastructure_qualification=pair, rounds=3)


# --- card and task records ---------------------------------------------------------------------------------

def _fixed_runtime(monkeypatch, runtime):
    seen = []

    def fake(task, **kwargs):
        seen.append((task, kwargs))
        return {**runtime, "task_variant": task}
    monkeypatch.setattr(robodojo_spec, "qualification_runtime", fake)
    return seen


def test_card_and_task_records_measure_the_lane_runtime(monkeypatch, tmp_path):
    from ora.orchestrator import robodojo_card_qualification as producer

    base = {"physics_device": "cuda:0", "resource_binding": {"slot": "gpu4"}, "cuda_visible_devices": "4"}
    seen = _fixed_runtime(monkeypatch, base)
    calls = []
    monkeypatch.setattr(producer, "qualify_card", lambda gpu, task, **kwargs: calls.append(
        ("card", gpu, task, kwargs)) or tmp_path / "card.json")
    monkeypatch.setattr(producer, "qualify_task", lambda gpu, task, **kwargs: calls.append(
        ("task", gpu, task, kwargs)) or tmp_path / "task.json")
    assert lane.card("hang_mugs", gpu="4", resource_profiles=["robodojo-gpu4-d1"]) == tmp_path / "card.json"
    _, gpu, task, kwargs = calls[-1]
    assert (gpu, task, kwargs["parallel_resource_profiles"], kwargs["owner_prefix"]) == (
        "4", "hang_mugs", None, "robodojo-lane-card")
    assert kwargs["runtime"]["resource_binding"] == {"slot": "gpu4"}
    lane.card("hang_mugs", gpu="4", resource_profiles=["robodojo-gpu4-d1", "robodojo-gpu4-d2"])
    kwargs = calls[-1][3]
    assert kwargs["parallel_resource_profiles"] == ("robodojo-gpu4-d1", "robodojo-gpu4-d2")
    assert "resource_binding" not in kwargs["runtime"]
    with pytest.raises(ValueError, match="two for parallel"):
        lane.card("hang_mugs", gpu="4", resource_profiles=["a", "b", "c"])
    assert lane.task_record("stack_bowls", gpu="4", card_record=tmp_path / "card.json") == tmp_path / "task.json"
    kwargs = calls[-1][3]
    assert kwargs["card_qualification"] == tmp_path / "card.json" and kwargs["owner_prefix"] == "robodojo-lane-task"
    assert kwargs["runtime"]["task_variant"] == "stack_bowls"
    assert [entry[1]["rounds"] for entry in seen] == [1, 1, 1]
    _fixed_runtime(monkeypatch, {**base, "physics_device": "cpu"})
    lane.card("press_by_number", gpu="4")
    assert calls[-1][3]["owner_prefix"] == "robodojo-lane-card-cpu"


def test_the_card_workspace_records_are_referenced_by_the_lane_route(monkeypatch, tmp_path):
    """The card producer writes its records under ``outputs/robodojo-compat/qualification``.

    That is a workspace, not a RunSpec namespace; the lane route references
    the records from there and revalidation reads them.
    """
    from test_robodojo_operator import _stub_probe_environment
    from ora.orchestrator import robodojo_card_qualification as producer
    from ora.orchestrator.paths import experiment_workspace
    from ora.orchestrator.robodojo_runtime_config import runtime_configuration

    assert producer.OUTPUT_ROOT == experiment_workspace("robodojo-compat")

    # The real card producer, with the probe stubbed (no interpreter, GPU or worker).
    _stub_probe_environment(producer, tmp_path, monkeypatch, device="6")
    workspace = tmp_path / "outputs/robodojo-compat"
    monkeypatch.setattr(producer, "OUTPUT_ROOT", workspace)
    layout = tmp_path / "Assets/Eval_Layout/RoboDojo/arx_x5/0/hang_mugs_0.json"
    layout.parent.mkdir(parents=True)
    layout.write_text("{}")
    monkeypatch.setenv("ROBODOJO_ASSET_ROOT", str(tmp_path / "Assets"))
    runtime = runtime_configuration("6", "hang_mugs")
    _fixed_runtime(monkeypatch, runtime)
    card = lane.card("hang_mugs", gpu="6")
    assert card.is_relative_to(workspace / "qualification") and json.loads(card.read_text())["passed"]

    # The lane route references it from that workspace, and revalidation reads it.
    monkeypatch.setattr(lane, "WORKSPACE_ROOT", tmp_path)
    reference = lane.card_reference(card)
    assert reference["path"].startswith("outputs/robodojo-compat/qualification/robodojo-lane-card-")
    lanes = tmp_path / "outputs/lane-fixture"
    monkeypatch.setattr(lane, "experiment_workspace", lambda name: lanes if name == "lane-fixture" else None)
    monkeypatch.setattr(lane, "_source_commit", lambda: "d" * 40)
    _smoke_dir(lanes, "robodojo-scene-smoke-a", runtime={**runtime, "task_variant": "hang_mugs"}, gpu="6")
    seen = {}

    def evaluate(runtime_, *, workspace_root, harness_commit, card_reference, smoke_reference):
        seen["card"] = card_reference
        return {"schema": rv.SCHEMA, "coverage": rv.scene_coverage(runtime_)}
    monkeypatch.setattr(lane, "evaluate", evaluate)
    gate = lane.revalidate("hang_mugs", gpu="6", experiment="lane-fixture", card=card)
    assert seen["card"] == reference and gate.is_file()


def test_a_card_reference_comes_from_the_record_or_a_card_gate(monkeypatch, tmp_path):
    from ora.environment.benchmarks.robodojo import CARD_QUALIFICATION_SCHEMA

    monkeypatch.setattr(lane, "WORKSPACE_ROOT", tmp_path)
    record = tmp_path / "outputs/robodojo-compat/qualification/card/card-qualification.json"
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({"schema": CARD_QUALIFICATION_SCHEMA, "passed": True}))
    expected = rv.reference(record, tmp_path)
    assert lane.card_reference(record) == expected
    assert expected["path"] == "outputs/robodojo-compat/qualification/card/card-qualification.json"
    gate = tmp_path / "card-gate.json"
    gate.write_text(json.dumps({"qualification": expected}))
    assert lane.card_reference(gate) == expected
    record.write_text(json.dumps({"schema": CARD_QUALIFICATION_SCHEMA, "passed": False}))
    with pytest.raises(ValueError, match="not a passing"):
        lane.card_reference(record)
    gate.write_text(json.dumps({"path": "x"}))
    with pytest.raises(ValueError, match="neither"):
        lane.card_reference(gate)


# --- the scene smoke ---------------------------------------------------------------------------------------

_SENSORS = {name: {"shape": [480, 640], "finite_positive_fraction": 1.0}
            for name in ("head", "left_wrist", "right_wrist")}
_POSE = [0.0, -0.3, 1.0, 0.0, 0.0, 0.0, 1.0]


class _Session:
    """A worker that answers every scene check; ``schedule`` relaunches its later layouts."""

    def __init__(self, schedule=None, *, unreachable=()):
        self.schedule, self.unreachable, self.calls, self.closed = schedule, set(unreachable), [], False
        self.delegate = type("Delegate", (), {"process": type("Process", (), {"pid": 4242})()})()

    def start(self):
        return {"device_binding": {"selected_gpu_uuid": "GPU-fixture"}}

    def observe(self, destination):
        self.calls.append(("observe",))
        return {"observation_id": f"obs-{len(self.calls)}", "robot_geometry_verification": {"passed": True},
                "proprioception": {"end_effectors": {arm: {"pose_xyz_xyzw": _POSE, "gripper_command": 1.0}
                                                     for arm in ("left", "right")}}}, []

    def align_plan(self, request):
        ref = request["target_ref"]
        self.calls.append(("align", ref, request["target_poses_xyzw"][0][:3]))
        arms = ["left", "right"] if request["arm"] == "both" else [request["arm"]]
        return {"candidates": [{"arm": arm, "reachable": ref not in self.unreachable
                                and (ref, arm) not in self.unreachable}
                               for arm in arms for _ in request["target_poses_xyzw"]]}

    def abort(self, reason):
        self.calls.append(("abort", reason))

    def restart_round(self, index, seed=None):
        self.calls.append(("restart", index, seed))
        if self.schedule is None:
            return {"round_index": index, "round_isolation": {"passed": True}}
        entry = self.schedule[index]
        return {"round_index": index, "scene_reset_mode": "simulation_relaunch", "layout_id": entry["layout_id"],
                "layout_sha256": entry["layout_sha256"], "round_isolation": {"passed": True}}

    def close(self):
        self.closed = True


#: A 0.12 m-wide, 0.05 m-high bowl model (the metadata shape of Rigid/bowl/00010).
_BOWL_BOX = [[x, y, z] for x in (-0.06, 0.06) for y in (-0.06, 0.06) for z in (-0.025, 0.025)]


def _smoke_layouts(tmp_path, monkeypatch):
    assets = tmp_path / "assets"
    metadata = assets / "Object/RoboDojo/Rigid/bowl/00010/metadata.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(json.dumps({"geometry": {"aligned_bbox": {"vertices": _BOWL_BOX}}}))
    schedule = []
    for ordinal in (21, 37, 51):
        relative = f"Eval_Layout/RoboDojo/arx_x5/0/stack_bowls_{ordinal}.json"
        (assets / relative).parent.mkdir(parents=True, exist_ok=True)
        (assets / relative).write_text(json.dumps({
            "Rigid": {"bowl": [{"label": f"bowl{index}", "category_idx": 10, "default_pos": [0.1 * index, -0.1, 0.79],
                                "default_ori": [1.0, 0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]} for index in range(3)],
                      # A scattered distractor (no asset metadata here): never a reach object.
                      "ball": [{"label": "ball", "type": "cluttered", "category_idx": 3,
                                "default_pos": [0.0, 0.3, 0.8]}]},
            # Static fixtures: the floor-standing dustbin and the camera stand are not reach objects.
            "Geometry": {"dustbin": [{"label": "dustbin", "category_idx": 0, "default_pos": [-0.63, -0.1, 0.4]}],
                         "camera_stand": [{"label": "camera_stand", "category_idx": 0,
                                           "default_pos": [0.0, -0.47, 0.765]}]}}))
        schedule.append({"layout_id": ordinal, "layout_file": relative, "layout_sha256": f"{ordinal:02d}" * 32})
    monkeypatch.setenv("ROBODOJO_ASSET_ROOT", str(assets))
    return {"asset_root_env": "ROBODOJO_ASSET_ROOT", "layout_schedule": {"rounds": schedule}}, schedule


def _fake_smoke_host(monkeypatch, tmp_path, runtime, session):
    _fixed_runtime(monkeypatch, runtime)
    leases = []
    monkeypatch.setattr(scene, "_prepare", lambda gpu, task, runtime, prefix: {
        "commit": "c" * 40, "kernel": "k" * 64, "stack": {"python": "fixture"}, "driver_version": "580.0",
        "owner": f"{prefix}-fixture"})
    monkeypatch.setattr(scene, "experiment_workspace", lambda name: tmp_path / "outputs" / name)
    monkeypatch.setattr(scene, "claim_resource_lease", lambda plan, owner, pid: leases.append(owner) or
                        {"claim_token": "t"})
    monkeypatch.setattr(scene, "assign_resource_lease", lambda lease, pid: None)
    monkeypatch.setattr(scene, "release_resource_lease", lambda owner, **kwargs: leases.append(("released", owner)))
    monkeypatch.setattr(scene, "build_environment_session", lambda **kwargs: session)
    monkeypatch.setattr(scene, "_worker_gpus", lambda pid: ["GPU-fixture"])
    monkeypatch.setattr(scene, "_sensor_evidence", lambda observation, images: dict(_SENSORS))
    monkeypatch.setattr(scene, "execute_runtime_action", lambda session, action: {
        "action_id": action["action_id"], "phase_receipts": [{"status": "reached"}, {"status": "executed"}]})
    monkeypatch.setattr(layout_smoke, "worker_gpu_memory_mib", lambda pid: 9000)
    return leases


_LANE = {"task_variant": "stack_bowls", "cuda_visible_devices": "0", "resource_profile": "robodojo-gpu0",
         "physics_device": "cuda:0", "native_action_limit": 800}


def test_the_unscheduled_smoke_writes_every_check_the_lane_gate_reads(monkeypatch, tmp_path):
    session = _Session()
    leases = _fake_smoke_host(monkeypatch, tmp_path, _LANE, session)
    report = scene.smoke("stack_bowls", gpu="0", experiment="lane-fixture")
    root = tmp_path / "outputs/lane-fixture/qualification/robodojo-scene-smoke-fixture"
    assert report == json.loads((root / "report.json").read_text())
    assert report["passed"] is True and report["schema"] == "ora.robodojo-b2-smoke.v1"
    assert set(rv.CHECKS) <= {name for name, value in report["checks"].items() if value is True}
    plan = json.loads((root / "plan.json").read_text())
    assert plan["schema"] == "ora.robodojo-b2-smoke-plan.v1"
    assert report["plan_sha256"] == plan["plan_sha256"] and plan["harness_commit"] == report["source_commit"]
    from ora.orchestrator.spec import json_sha256
    assert json_sha256({key: value for key, value in plan.items() if key != "plan_sha256"}) == plan["plan_sha256"]
    assert set(plan["dependency_closure"]["checker_files"]) == set(scene.CHECKER_FILES)
    assert [call for call in session.calls if call[0] in ("abort", "restart")] == [
        ("abort", "qualification_boundary"), ("restart", 1, 0)]
    assert len(report["actions"]) == 2 and set(report["alignment"]) == {"left", "right"}
    assert session.closed and leases == ["robodojo-scene-smoke-fixture", ("released", "robodojo-scene-smoke-fixture")]
    assert "layout_schedule" not in report and "close_elapsed_s" in report


def test_the_scheduled_smoke_relaunches_every_layout_and_writes_the_schedule_checks(monkeypatch, tmp_path):
    runtime, schedule = _smoke_layouts(tmp_path, monkeypatch)
    session = _Session(schedule)
    _fake_smoke_host(monkeypatch, tmp_path, {**_LANE, **runtime}, session)
    report = scene.smoke("stack_bowls", gpu="0", experiment="lane-fixture", rounds=3)
    assert report["passed"] is True and report["layout_schedule"] == [21, 37, 51]
    assert all(report["checks"][name] is True for name in (*rv.CHECKS, *SCHEDULE_CHECKS))
    assert [call for call in session.calls if call[0] in ("abort", "restart")] == [
        ("abort", "qualification_boundary"), ("restart", 1, 0), ("abort", "qualification_boundary"),
        ("restart", 2, 0)]
    assert [record["layout_id"] for record in report["rounds"]] == [37, 51]


def test_a_failing_smoke_keeps_its_report_and_releases_its_lease(monkeypatch, tmp_path):
    session = _Session(unreachable={("smoke-current-pose", "right")})
    leases = _fake_smoke_host(monkeypatch, tmp_path, _LANE, session)
    with pytest.raises(AssertionError, match="right current pose not reachable"):
        scene.smoke("stack_bowls", gpu="0", experiment="lane-fixture")
    report = json.loads((tmp_path / "outputs/lane-fixture/qualification/robodojo-scene-smoke-fixture"
                         / "report.json").read_text())
    assert report["passed"] is False and report["error"]["type"] == "AssertionError"
    assert session.closed and leases[-1] == ("released", "robodojo-scene-smoke-fixture")


# --- the scheduled rounds (the frozen layout smoke's checks, on the maintained copy) -------------------------------

def test_the_smoke_top_down_pose_points_the_arx_approach_axis_down():
    from scipy.spatial.transform import Rotation

    for yaw in layout_smoke.TOP_DOWN_YAWS_DEG:
        rotation = Rotation.from_quat(layout_smoke.top_down_orientation(yaw))  # xyzw
        assert rotation.apply([1, 0, 0]) == pytest.approx([0, 0, -1], abs=1e-9)
        radians = math.radians(yaw)
        assert rotation.apply([0, 1, 0]) == pytest.approx([-math.sin(radians), math.cos(radians), 0], abs=1e-9)


def _run_scheduled_rounds(tmp_path, monkeypatch, *, unreachable=()):
    runtime, schedule = _smoke_layouts(tmp_path, monkeypatch)
    monkeypatch.setattr(layout_smoke, "worker_gpu_memory_mib", lambda pid: 9000 + pid)
    session = _Session(schedule, unreachable=unreachable)
    report = {"checks": {}}
    try:
        layout_smoke.scheduled_rounds(session, report, tmp_path, runtime, pid=1, save=lambda: None,
                                      sensor_evidence=lambda observation, images: dict(_SENSORS))
    except AssertionError as exc:
        return session, report, str(exc)
    return session, report, None


def test_scheduled_rounds_aim_only_at_reach_objects_and_record_each_footprint_point(tmp_path, monkeypatch):
    session, report, error = _run_scheduled_rounds(tmp_path, monkeypatch)
    assert error is None and report["restart"] is report["rounds"][0]["restart"]
    assert all(report["checks"][name] is True for name in SCHEDULE_CHECKS)
    bowl = report["rounds"][1]["object_reach"]["bowl2"]
    assert bowl["reachable_arms"] == ["left", "right"] and bowl["footprint_half_extents_m"] == [0.06, 0.06]
    assert [point["point"] for point in bowl["points"]] == ["centre", *(f"edge{index}" for index in range(8))]
    assert bowl["points"][3]["position_m"] == pytest.approx([0.2, -0.04, 0.99])
    refs = {call[1] for call in session.calls if call[0] == "align" and call[1].startswith("smoke-top-down-")}
    assert {ref.split("-")[3] for ref in refs} == {"bowl0", "bowl1", "bowl2"} and len(refs) == 3 * 9


def test_an_object_counts_only_through_one_arm_that_covers_its_whole_footprint(tmp_path, monkeypatch):
    _, report, error = _run_scheduled_rounds(tmp_path, monkeypatch,
                                             unreachable={("smoke-top-down-bowl1-edge0", "left")})
    assert error is None and report["round0_object_reach"]["bowl1"]["reachable_arms"] == ["right"]
    _, report, error = _run_scheduled_rounds(tmp_path / "split", monkeypatch, unreachable={
        ("smoke-top-down-bowl1-edge0", "left"), ("smoke-top-down-bowl1-edge4", "right")})
    assert report["round0_object_reach"]["bowl1"]["reachable_arms"] == []
    assert report["checks"]["scheduled_object_reach"] is False
    assert "round 0 bowl1: left misses edge0@0deg; right misses edge4@180deg" in error


# --- revalidation and the lane gate ---------------------------------------------------------------------------

def _smoke_dir(workspace, name, *, runtime, layout_schedule=None, passed=True, gpu="0", mtime=1):
    root = workspace / "qualification" / name
    root.mkdir(parents=True)
    (root / "plan.json").write_text(json.dumps({"runtime": runtime}))
    report = root / "report.json"
    report.write_text(json.dumps({"schema": "ora.robodojo-b2-smoke.v1", "task": runtime["task_variant"],
                                  "gpu": gpu, "passed": passed, "layout_schedule": layout_schedule}))
    os.utime(report, (mtime, mtime))
    return report


def _configured(**changes):
    base = {"task_variant": "stack_bowls", "eval_seed": 0, "layout_id": 0, "layout_file": "f", "layout_sha256": "s",
            "num_envs": 1, "env_config": "arx_x5", "sensor_profile": "rgbd_metric_v1", "physics_dt": 0.004,
            "control_hz": 25, "native_action_limit": 800, "native_actions_per_command": 20,
            "physics_device": "cuda:0", "renderer_gpu": 0, "cuda_visible_devices": "0", "main_camera": "head",
            "policy_cameras": ["head", "left_wrist", "right_wrist"],
            "workspace_bounds_m": {"x": [-0.75, 0.75], "y": [-0.65, 0.55], "z": [0.72, 1.4]},
            "startup_timeout_s": 600, "request_timeout_s": 600, "close_timeout_s": 30}
    return {**base, **changes}


def test_revalidation_certifies_the_newest_smoke_of_exactly_this_lane_and_writes_its_gate(monkeypatch, tmp_path):
    runtime = _configured()
    _fixed_runtime(monkeypatch, runtime)
    workspace = tmp_path / "outputs/lane-fixture"
    monkeypatch.setattr(lane, "experiment_workspace", lambda name: workspace)
    monkeypatch.setattr(lane, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(lane, "_source_commit", lambda: "d" * 40)
    chosen = _smoke_dir(workspace, "robodojo-scene-smoke-a", runtime=runtime, mtime=1)
    _smoke_dir(workspace, "robodojo-scene-smoke-bounds", runtime=_configured(
        workspace_bounds_m={"x": [-1.5, 1.5], "y": [-1.5, 1.5], "z": [-0.2, 2.0]}), mtime=5)
    _smoke_dir(workspace, "robodojo-scene-smoke-failed", runtime=runtime, passed=False, mtime=6)
    _smoke_dir(workspace, "robodojo-scene-smoke-gpu4", runtime=runtime, gpu="4", mtime=7)
    _smoke_dir(workspace, "robodojo-scene-smoke-scheduled", runtime=runtime, layout_schedule=[23, 29], mtime=8)
    _smoke_dir(workspace, "b2-robodojo-smoke-frozen", runtime=runtime, mtime=9)
    assert lane.smoke_candidates("lane-fixture", runtime) == [chosen]
    card = tmp_path / "card-gate.json"
    card.write_text(json.dumps({"qualification": {"path": "outputs/card.json", "sha256": "1" * 64}}))
    seen = {}

    def evaluate(runtime_, *, workspace_root, harness_commit, card_reference, smoke_reference):
        seen.update(commit=harness_commit, card=card_reference, smoke=smoke_reference["path"])
        return {"schema": rv.SCHEMA, "coverage": rv.scene_coverage(runtime_)}
    monkeypatch.setattr(lane, "evaluate", evaluate)
    gate = lane.revalidate("stack_bowls", gpu="0", experiment="lane-fixture", card=card)
    assert gate == workspace / "gates/stack_bowls-l0-cuda0.json"
    assert seen == {"commit": "d" * 40, "card": {"path": "outputs/card.json", "sha256": "1" * 64},
                    "smoke": "outputs/lane-fixture/qualification/robodojo-scene-smoke-a/report.json"}
    binding = json.loads(gate.read_text())
    record = tmp_path / binding["path"]
    assert record.parent == workspace / "qualification/revalidation"
    value = json.loads(record.read_text())
    assert value["checker_source_commit"] == "d" * 40 and len(value["checker_sha256"]) == 64
    assert robodojo_spec.qualification_reference(gate) == binding
    assert set(binding) == {"path", "sha256"}


def test_a_scheduled_lane_gate_names_its_layouts_and_needs_a_smoke_of_that_schedule(monkeypatch, tmp_path):
    schedule = {"rounds": [{"layout_id": 23}, {"layout_id": 29}, {"layout_id": 51}]}
    runtime = _configured(layout_schedule=schedule)
    _fixed_runtime(monkeypatch, runtime)
    workspace = tmp_path / "outputs/lane-fixture"
    monkeypatch.setattr(lane, "experiment_workspace", lambda name: workspace)
    _smoke_dir(workspace, "robodojo-scene-smoke-one-round", runtime=runtime)
    with pytest.raises(ValueError, match="no passing stack_bowls/GPU0/cuda:0 scene smoke"):
        lane.revalidate("stack_bowls", gpu="0", experiment="lane-fixture", card=tmp_path / "unused.json", rounds=3)
    assert lane.gate_path("lane-fixture", runtime) == workspace / "gates/stack_bowls-l23.29.51-cuda0.json"
    assert lane.gate_path("lane-fixture", _configured(physics_device="cpu", cuda_visible_devices="4")) == (
        workspace / "gates/stack_bowls-l0-cpu-gpu4.json")


def test_the_entry_point_confirms_simulator_commands_and_dispatches_revalidation(monkeypatch, capsys, tmp_path):
    with pytest.raises(SystemExit):
        lane.main(["smoke", "stack_bowls", "--gpu", "0", "--experiment", "lane-fixture"])
    with pytest.raises(SystemExit):
        lane.main(["card", "stack_bowls", "--gpu", "0"])
    calls = []
    monkeypatch.setattr(lane, "revalidate", lambda task, **kwargs: calls.append((task, kwargs)) or tmp_path / "g.json")
    assert lane.main(["revalidate", "stack_bowls", "--gpu", "0", "--experiment", "lane-fixture", "--card",
                      str(tmp_path / "card.json"), "--rounds", "3", "--layouts", "23", "29", "51"]) == 0
    assert calls == [("stack_bowls", {"gpu": "0", "experiment": "lane-fixture", "card": tmp_path / "card.json",
                                      "resource_profile": None, "rounds": 3, "layout_schedule": [23, 29, 51]})]
    assert capsys.readouterr().out.strip() == str(tmp_path / "g.json")


def test_a_cpu_lane_reads_only_its_cpu_smoke_and_writes_its_cpu_gate(monkeypatch, tmp_path):
    """A newer CUDA smoke of the same task and GPU never certifies a CPU-physics lane."""
    runtime = _configured(task_variant="press_by_number", physics_device="cpu", cuda_visible_devices="4",
                          renderer_gpu=4)
    _fixed_runtime(monkeypatch, runtime)
    workspace = tmp_path / "outputs/lane-fixture"
    monkeypatch.setattr(lane, "experiment_workspace", lambda name: workspace)
    monkeypatch.setattr(lane, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(lane, "_source_commit", lambda: "d" * 40)
    cpu = _smoke_dir(workspace, "robodojo-scene-smoke-cpu", runtime=runtime, gpu="4", mtime=1)
    _smoke_dir(workspace, "robodojo-scene-smoke-cuda", runtime={**runtime, "physics_device": "cuda:0"}, gpu="4",
               mtime=2)
    seen = {}
    monkeypatch.setattr(lane, "evaluate", lambda runtime_, **kwargs: seen.update(
        smoke=kwargs["smoke_reference"]["path"]) or {"schema": rv.SCHEMA})
    card = tmp_path / "card-gate.json"
    card.write_text(json.dumps({"qualification": {"path": "outputs/card.json", "sha256": "1" * 64}}))
    gate = lane.revalidate("press_by_number", gpu="4", experiment="lane-fixture", card=card)
    assert seen["smoke"] == cpu.relative_to(tmp_path).as_posix()
    assert gate == workspace / "gates/press_by_number-l0-cpu-gpu4.json"
