"""The maintained suite queue (suite_cells + suite_queue), offline.

Cells are built by benchmark_spec against fixture readiness reports and seed
manifests; launch, lease and supervisor are fakes that write the run files the
Runtime would. These carry the cluster queue lessons: a released lease with a
result is not yet a landing while the supervisor seals (BOOT-09), an
action-free trusted fault or a lone unsealed landing does not stop a group,
a preflight refusal that consumed no attempt is retried, and a drained or
resumed group never launches a claimed cell again.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ora.orchestrator import suite_cells, suite_queue
from test_benchmark_spec import _official_manifest, _readiness_report, workspace  # noqa: F401 (fixture)

SOURCE = "f" * 40


def _cells(*entries):
    return suite_cells.parse_cells({"schema": suite_cells.CELLS_SCHEMA, "cells": list(entries)})


LIBERO = [{"environment": "libero_pro", "suite": "libero_10_swap", "task_id": 6, "seed": 0},
          {"environment": "libero", "suite": "libero_object", "task_id": 0, "seed": 1},
          {"environment": "libero_pro", "suite": "libero_goal_task", "task_id": 2, "seed": 4}]


# --- cell lists -----------------------------------------------------------------------------------

def test_a_cell_list_names_cells_only_and_ids_follow_the_builder():
    cells = _cells(*LIBERO, {"environment": "robotwin", "task": "adjust_bottle", "seed": 100002,
                             "task_config": "demo_clean"})
    assert [cell.config for cell in cells] == [
        "lp-libero_10_swap-t06-s0", "lb-libero_object-t00-s1", "lp-libero_goal_task-t02-s4",
        "rt-clean-adjust_bottle-s100002"]
    assert [cell.runtime_kind for cell in cells] == ["libero_pro"] * 3 + ["robotwin"]


@pytest.mark.parametrize("entry, match", [
    ({"environment": "robodojo", "task": "push_T", "seed": 0}, "environment must be one of"),
    ({"environment": "libero", "suite": "libero_object", "task_id": 0}, r"missing \['seed'\]"),
    ({"environment": "libero", "suite": "libero_object", "task_id": 0, "seed": 0, "rounds": 3}, "unknown"),
    ({"environment": "robotwin", "task": "adjust_bottle", "seed": 1, "task_config": "demo_x"}, "task_config"),
    ({"environment": "libero", "suite": "libero_object", "task_id": 0, "seed": 0, "baseline": "native_full_v1"},
     "baseline must be one of"),
])
def test_a_cell_list_refuses_what_is_not_a_cell(entry, match):
    with pytest.raises(ValueError, match=match):
        _cells(entry)


def test_a_robotwin_cell_takes_the_official_episode_instruction_unless_it_names_its_own(workspace, monkeypatch):  # noqa: F811
    from ora.environment.benchmarks.robotwin_instructions import official_episode_instruction

    report = _readiness_report(workspace / "reports", monkeypatch)
    manifest = _official_manifest(workspace / "manifests", "place_a2b_left", "demo_randomized", (100004, 100001))
    named = {"environment": "robotwin", "task": "place_a2b_left", "seed": 100004, "seed_manifest": str(manifest),
             "episode_instruction": "Put the coffee box left of the bell"}
    cell, = _cells(named)
    treatment = suite_cells.Treatment(experiment="suite-queue-fixture", cohort_id="c", service_report=report)
    spec = suite_cells.build_spec(cell, "robotwin-gpu4", treatment)
    assert spec["task_instruction"] == "Put the coffee box left of the bell"
    assert spec["runtime"]["task_instruction_source"] == "episode_name"
    for bad in ({"environment": "robotwin", "task": "adjust_bottle", "seed": 1, "episode_instruction": " "},
                {"environment": "libero_pro", "suite": "libero_10_task", "task_id": 1, "seed": 1,
                 "episode_instruction": "x"}):
        with pytest.raises(ValueError, match="episode_instruction|unknown"):
            _cells(bad)
    without, = _cells({key: value for key, value in named.items() if key != "episode_instruction"})
    spec = suite_cells.build_spec(without, "robotwin-gpu4", treatment)
    assert spec["task_instruction"] == official_episode_instruction("place_a2b_left", 100004, "demo_randomized")
    assert spec["runtime"]["task_instruction_source"] == "official_episode_unseen_v1"
    outside, = _cells({**{key: value for key, value in named.items() if key != "episode_instruction"},
                       "seed": 100001})
    with pytest.raises(Exception, match="episode names lack"):
        suite_cells.build_spec(outside, "robotwin-gpu4", treatment)


def test_an_expert_unsolved_cell_replaces_its_seed_manifest():
    unsolved = {"reason": "RPent pair", "evidence": ["/scan/a.json"]}
    cell, = _cells({"environment": "robotwin", "task": "adjust_bottle", "seed": 100004, "expert_unsolved": unsolved})
    assert cell.summary()["expert_unsolved"] is True
    for bad in ({**unsolved, "evidence": []}, {"reason": "", "evidence": ["x"]}, {"evidence": ["x"]}):
        with pytest.raises(ValueError, match="expert_unsolved"):
            _cells({"environment": "robotwin", "task": "adjust_bottle", "seed": 1, "expert_unsolved": bad})
    with pytest.raises(ValueError, match="replaces seed_manifest"):
        _cells({"environment": "robotwin", "task": "adjust_bottle", "seed": 1, "expert_unsolved": unsolved,
                "seed_manifest": "/m.json"})


def test_admission_is_per_cell_so_ids_must_be_distinct():
    with pytest.raises(ValueError, match="distinct"):
        _cells(LIBERO[0], dict(LIBERO[0]))


def test_cells_build_through_benchmark_spec_under_the_queue_treatment(workspace, monkeypatch):  # noqa: F811
    report = _readiness_report(workspace / "reports", monkeypatch)
    manifests = workspace / "manifests"
    _official_manifest(manifests, "adjust_bottle", "demo_clean", (100002,))
    (manifests / "adjust_bottle-demo_clean.json").rename(manifests / "adjust_bottle.json")
    treatment = suite_cells.Treatment(experiment="suite-queue-fixture", cohort_id="suite-queue-fixture-g001",
                                      service_report=report, rounds=3, seed_manifests=manifests)
    robotwin, libero = _cells({"environment": "robotwin", "task": "adjust_bottle", "seed": 100002,
                               "task_config": "demo_clean"}, LIBERO[1])
    spec = suite_cells.build_spec(robotwin, "robotwin-gpu4", treatment)
    assert spec["runtime"]["task_config"] == "demo_clean" and spec["session"]["round_seeds"] == [100002] * 3
    assert spec["launch_admission"]["cell_id"] == robotwin.config
    assert suite_cells.compose(robotwin, spec, treatment).is_file()
    spec = suite_cells.build_spec(libero, "libero-gpu4b", treatment)
    assert (spec["milestone"], spec["runtime"]["suite"], spec["config"]) == (
        "LIBERO", "libero_object", "lb-libero_object-t00-s1")


def test_a_cell_may_name_a_registered_baseline_and_otherwise_takes_the_default(workspace, monkeypatch):  # noqa: F811
    """Both arms of a paired comparison can share one list: the cell's baseline is its only difference."""
    report = _readiness_report(workspace / "reports", monkeypatch)
    treatment = suite_cells.Treatment(experiment="suite-queue-fixture", cohort_id="suite-queue-fixture-g001",
                                      service_report=report)
    default, named = _cells(LIBERO[1], {**LIBERO[1], "config": "lb-libero_object-t00-s1-named",
                                        "baseline": "native_full_explicit_motion_direct_v4"})
    spec = suite_cells.build_spec(default, "libero-gpu4b", treatment)
    assert "baseline" not in spec["policy"]
    assert {item["tool_surface"] for item in spec["prerequisite"]["services"].values()} == {
        "libero_placement_direct_overhead_v1"}
    spec = suite_cells.build_spec(named, "libero-gpu4b", treatment)
    assert spec["policy"]["baseline"] == "native_full_explicit_motion_direct_v4"
    # The readiness report binds to the selected baseline's surface, which launch preflight compares.
    surfaces = {item["tool_surface"] for item in spec["prerequisite"]["services"].values()}
    assert surfaces == {"libero_placement_direct_overhead_v1"}
    assert spec["config"] == "lb-libero_object-t00-s1-named"


# --- one run's landing ----------------------------------------------------------------------------

class FakeRuntime:
    """Run roots, leases and supervisors the queue reads; launches that write them."""

    def __init__(self, tmp_path: Path, monkeypatch) -> None:
        self.runs, self.alive, self.leases = tmp_path / "runs", {}, set()
        self.outcomes: dict[str, str] = {}
        self.launches: list[str] = []
        self.refusals: dict[str, int] = {}
        monkeypatch.setattr(suite_queue.paths, "resolve_run_root", lambda run_id: self.runs / run_id)
        monkeypatch.setattr(suite_queue, "resource_lease_for_owner",
                            lambda run_id: {"owner": run_id} if run_id in self.leases else None)
        monkeypatch.setattr(suite_queue, "supervisor_alive", lambda root, run_id: self.alive.get(run_id, 0) > 0
                            and not self._tick(run_id))
        monkeypatch.setattr(suite_queue, "source_commit", lambda root=None: SOURCE)
        monkeypatch.setattr(suite_queue, "resolve_run_spec", self.plan)
        monkeypatch.setattr(suite_queue.subprocess, "run", self.launch)
        monkeypatch.setattr(suite_queue.paths, "workspace_from_plan",
                            lambda plan: suite_queue.benchmark_spec.experiment_root(plan["experiment"]))
        monkeypatch.setattr(suite_queue, "POLL_SECONDS", 0.0)
        monkeypatch.setattr(suite_queue, "PREFLIGHT_RETRY_DELAY_S", 0.0)

    def _tick(self, run_id: str) -> bool:
        """A live supervisor seals on its last tick, then exits."""
        self.alive[run_id] -= 1
        if self.alive[run_id] == 0 and self.outcomes.get(run_id) == "sealed_late":
            self._seal(self.runs / run_id, run_id)
        return False

    @staticmethod
    def plan(raw, root=None):
        return {"source": {"commit": SOURCE, "dirty": False}, "launch_admission": raw["launch_admission"],
                "runtime": {"resource_binding": {"profile_id": raw["runtime"]["resource_profile"]}},
                "plan_id": "plan-" + raw["config"], "plan_sha256": hashlib.sha256(raw["config"].encode()).hexdigest(),
                "experiment": raw["experiment"], "output_workspace": raw["output_workspace"]}

    def _seal(self, root: Path, run_id: str) -> None:
        (root / "seal.json").write_text(json.dumps({
            "run_id": run_id, "evaluation_status": "valid", "sealed_source_commit": SOURCE,
            "sealed_source_dirty": False}))

    def land(self, run_id: str, plan, outcome: str) -> None:
        root = self.runs / run_id
        root.mkdir(parents=True)
        (root / "plan.json").write_text(json.dumps({"source": plan["source"], "plan_sha256": plan["plan_sha256"]}))
        if outcome == "action_free_fault":
            result = {"state": "incomplete", "termination": "trusted_fault", "decisions": 0}
            evaluation = {"run_id": run_id, "status": "invalid", "reasons": ["trusted_fault"]}
        else:
            result = {"state": "completed", "termination": "policy_finished_episode", "decisions": 5,
                      "official_success": True}
            evaluation = {"run_id": run_id, "status": "invalid" if outcome == "unsealed" else "valid",
                          "reasons": ["evaluator_refusal"] if outcome == "unsealed" else []}
        (root / "result.json").write_text(json.dumps(result))
        (root / "evaluation.json").write_text(json.dumps(evaluation))
        if outcome == "sealed":
            self._seal(root, run_id)
        if outcome == "sealed_late":
            self.alive[run_id], self.outcomes[run_id] = 3, outcome

    def launch(self, command, *, cwd, stdout, stderr, timeout):
        spec = json.loads(Path(command[2]).read_text())
        config = spec["config"]
        if self.refusals.get(config, 0) > 0:
            self.refusals[config] -= 1
            stdout.write("preflight: BLOCKED\n")
            plan = self.plan(spec)
            report = suite_queue.paths.workspace_from_plan(plan) / "preflight" / f"{plan['plan_sha256']}.json"
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps({"checks": [{"check": "provider_marker", "passed": False}]}))
            return type("Done", (), {"returncode": 2})()
        run_id = f"run-{len(self.launches)}"
        self.launches.append(config)
        self.land(run_id, self.plan(spec), self.outcomes.get(config, "sealed"))
        self.outcomes[run_id] = self.outcomes.get(config, "sealed")
        stdout.write(f"launched {run_id} pid=1\n")
        return type("Done", (), {"returncode": 0})()


def _plan(config="cell"):
    return FakeRuntime.plan({"launch_admission": {}, "runtime": {"resource_profile": "lane"}, "config": config,
                             "experiment": "suite-queue-fixture", "output_workspace": "outputs/suite-queue-fixture"})


def test_a_released_lease_with_a_result_waits_for_the_live_supervisor_seal(tmp_path, monkeypatch):
    fake = FakeRuntime(tmp_path, monkeypatch)
    fake.land("run-late", _plan(), "sealed_late")
    settled = suite_queue.wait_for_terminal("run-late", _plan(), poll=0)
    assert settled["state"] == "sealed_terminal" and fake.alive["run-late"] == 0


def test_an_unsealed_landing_is_declared_only_after_its_supervisor_exits(tmp_path, monkeypatch):
    fake = FakeRuntime(tmp_path, monkeypatch)
    fake.land("run-u", _plan(), "unsealed")
    fake.alive["run-u"] = 2
    with pytest.raises(suite_queue.UnsealedLanding) as landing:
        suite_queue.wait_for_terminal("run-u", _plan(), poll=0)
    assert fake.alive["run-u"] == 0 and landing.value.record["evaluation_reasons"] == ["evaluator_refusal"]


def test_an_action_free_trusted_fault_settles_without_a_seal(tmp_path, monkeypatch):
    fake = FakeRuntime(tmp_path, monkeypatch)
    fake.land("run-f", _plan(), "action_free_fault")
    assert suite_queue.wait_for_terminal("run-f", _plan(), poll=0)["state"] == "trusted_fault_recorded"


# --- a group --------------------------------------------------------------------------------------

@pytest.fixture
def queue_env(workspace, monkeypatch):  # noqa: F811
    report = _readiness_report(workspace / "reports", monkeypatch)
    fake = FakeRuntime(workspace, monkeypatch)
    treatment = suite_cells.Treatment(experiment="suite-queue-fixture", cohort_id="suite-queue-fixture-g001",
                                      service_report=report)

    def queue(cells, lanes=("libero-gpu4b", "libero-gpu5"), group="g", **kwargs):
        kwargs.setdefault("stagger_s", 0.0)
        return suite_queue.Queue(_cells(*cells), lanes=list(lanes), group_id=group, treatment=treatment,
                                 cells_sha256="0" * 64, root=workspace, **kwargs)
    return fake, queue


def _status(queue):
    return json.loads((queue.group / "status.json").read_text())


def test_a_group_launches_each_cell_once_and_completes(queue_env):
    fake, make = queue_env
    queue = make(LIBERO)
    assert queue.run() == 0
    status = _status(queue)
    assert status["queue_status"] == "complete" and sorted(fake.launches) == sorted(c.config for c in _cells(*LIBERO))
    assert {cell["state"] for cell in status["cells"].values()} == {"sealed_terminal"}
    assert set(status["lanes"]) == {"libero-gpu4b", "libero-gpu5"}


def test_a_lone_unsealed_landing_and_an_action_free_fault_do_not_stop_the_group(queue_env):
    fake, make = queue_env
    configs = [cell.config for cell in _cells(*LIBERO)]
    fake.outcomes.update({configs[0]: "unsealed", configs[1]: "action_free_fault"})
    queue = make(LIBERO)
    assert queue.run() == 0
    states = {name: cell["state"] for name, cell in _status(queue)["cells"].items()}
    assert states == {configs[0]: "unsealed_recorded", configs[1]: "trusted_fault_recorded",
                      configs[2]: "sealed_terminal"}


def test_the_limit_th_unsealed_landing_pauses_the_group(queue_env):
    fake, make = queue_env
    fake.outcomes.update({cell.config: "unsealed" for cell in _cells(*LIBERO)})
    queue = make(LIBERO, lanes=("libero-gpu4b",), unsealed_pause_limit=2)
    assert queue.run() == 2
    status = _status(queue)
    assert status["queue_status"] == "paused" and len(fake.launches) == 2
    assert "without a sealed terminal result" in status["first_error"]["reason"]


def test_a_preflight_refusal_without_an_attempt_goes_back_to_the_end(queue_env):
    fake, make = queue_env
    first = _cells(*LIBERO)[0].config
    fake.refusals[first] = 1
    queue = make(LIBERO, lanes=("libero-gpu4b",))
    assert queue.run() == 0, _status(queue).get("first_error")
    status = _status(queue)
    assert status["cells"][first]["preflight_retries"] == 1 and fake.launches[-1] == first
    assert status["dequeue_order"] == [first, *fake.launches[:2], first]
    # Each attempt keeps its own launch log (the EXP12B queue reopened the first one and paused).
    assert {path.name for path in queue.group.glob(f"{first}-launch*.log")} == {
        f"{first}-launch.log", f"{first}-launch-r1.log"}


def test_a_drained_group_launches_nothing_new(queue_env):
    fake, make = queue_env
    queue = make(LIBERO)
    original = queue._next

    def drain_first(lane):
        (queue.group / "drain").touch()
        return original(lane)
    queue._next = drain_first
    assert queue.run() == 2 and fake.launches == []
    assert _status(queue)["first_error"]["reason"] == "drained by operator"


def test_resume_skips_the_cohort_claimed_cells_and_lanes_must_fit_the_cells(queue_env):
    fake, make = queue_env
    claimed = _cells(*LIBERO)[0].config
    admissions = suite_queue.benchmark_spec.experiment_root("suite-queue-fixture") / "launch-admissions"
    (admissions / "suite-queue-fixture-g001").mkdir(parents=True)
    (admissions / "suite-queue-fixture-g001" / f"{claimed}.json").write_text("{}")
    queue = make(LIBERO, group="resumed", resume=True)
    assert queue.run() == 0 and claimed not in fake.launches and len(fake.launches) == 2
    assert _status(queue)["skipped_claimed_unsealed"] == [claimed]
    with pytest.raises(RuntimeError, match="no lane registers"):
        make([*LIBERO, {"environment": "robotwin", "task": "adjust_bottle", "seed": 1}], group="mixed")
    with pytest.raises(RuntimeError, match="not registered"):
        make(LIBERO, lanes=("libero-gpu99",), group="nolane")


def test_launches_of_one_group_are_staggered(queue_env, monkeypatch):
    fake, make = queue_env
    starts = []
    original = fake.launch

    def timed(*args, **kwargs):
        starts.append(suite_queue.time.monotonic())
        return original(*args, **kwargs)
    monkeypatch.setattr(suite_queue.subprocess, "run", timed)
    queue = make(LIBERO, lanes=("libero-gpu4b", "libero-gpu5", "libero-gpu3"), stagger_s=0.2)
    assert queue.run() == 0 and len(starts) == 3
    gaps = [later - earlier for earlier, later in zip(sorted(starts), sorted(starts)[1:])]
    assert min(gaps) >= 0.19 and _status(queue)["stagger_s"] == 0.2


def test_max_live_caps_live_cells_below_the_lane_count(queue_env, monkeypatch):
    fake, make = queue_env
    live, peak = [0], [0]
    original = suite_queue.Queue.execute

    def counted(self, config, lane):
        live[0] += 1
        peak[0] = max(peak[0], live[0])
        suite_queue.time.sleep(0.05)
        try:
            return original(self, config, lane)
        finally:
            live[0] -= 1
    monkeypatch.setattr(suite_queue.Queue, "execute", counted)
    queue = make(LIBERO, lanes=("libero-gpu4b", "libero-gpu5", "libero-gpu3"), max_live=2)
    assert queue.run() == 0 and peak[0] == 2 and len(fake.launches) == 3
    assert _status(queue)["max_live"] == 2
    with pytest.raises(ValueError, match="max_live"):
        make(LIBERO, lanes=("libero-gpu4b",), max_live=2, group="toomany")


def test_the_entry_point_refuses_a_launch_without_yes(capsys):
    assert suite_queue.main(["run", "--cells", "x.json", "--experiment", "e", "--cohort", "c", "--group", "g",
                             "--lanes", "a", "--service-report", "r.json"]) == 2
    assert "--yes" in capsys.readouterr().err
