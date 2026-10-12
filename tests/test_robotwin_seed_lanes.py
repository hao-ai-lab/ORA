"""Host-only RoboTwin seed-scan lane execution: lease-probe retry and ledger reading."""

from __future__ import annotations

import pytest

from ora.orchestrator import robotwin_seed_lanes as lanes


def test_lease_claim_retries_only_an_unreadable_device_probe():
    sleeps, calls = [], []
    def flaky(plan, owner, pid):
        calls.append(owner)
        if len(calls) < 3:
            raise RuntimeError("co-residency refused: cannot read GPU memory "
                               "(nvidia-smi did not answer (TimeoutExpired)); rule=device_probe_unavailable")
        return {"claim_token": owner}
    assert lanes.claim_with_probe_retry(flaky, {}, "lane", 1, sleep=sleeps.append) == {"claim_token": "lane"}
    assert len(calls) == 3 and sleeps == [lanes.PROBE_RETRY_BACKOFF_S, 2 * lanes.PROBE_RETRY_BACKOFF_S]
    def leased(plan, owner, pid):
        calls.append("leased")
        raise RuntimeError("slot already leased; rule=slot_lease")
    with pytest.raises(RuntimeError, match="slot_lease"):
        lanes.claim_with_probe_retry(leased, {}, "lane", 1, sleep=sleeps.append)
    assert calls.count("leased") == 1
    def never(plan, owner, pid):
        raise RuntimeError("rule=device_probe_unavailable")
    with pytest.raises(RuntimeError, match="device_probe_unavailable"):
        lanes.claim_with_probe_retry(never, {}, "lane", 1, sleep=lambda seconds: None)


def test_a_running_ledger_is_read_without_its_torn_final_line(tmp_path):
    ledger = tmp_path / "seed-ledger.jsonl"
    ledger.write_text('{"requested_seed": 100000}\n\n{"requested_seed": 100001}\n{"requested_se')
    assert [r["requested_seed"] for r in lanes.read_ledger_records(ledger)] == [100000, 100001]


_WORKER = (
    "import json, sys, time, pathlib\n"
    "root = pathlib.Path(sys.argv[1]); ledger = root / 'seed-ledger.jsonl'\n"
    "rows = [json.loads(l) for l in ledger.read_text().splitlines()] if ledger.exists() else []\n"
    "seed = max([r['requested_seed'] for r in rows], default=99999) + 1\n"
    "while seed < {limit}:\n"
    "    attempt = root / 'attempts' / f'seed-{{seed:06d}}-try-00'\n"
    "    attempt.mkdir(parents=True)\n"
    "    time.sleep({delay})\n"
    "    rec = {{'requested_seed': seed, 'accepted': True, 'classification': 'accepted'}}\n"
    "    (attempt / 'attempt.json').write_text(json.dumps(rec))\n"
    "    with ledger.open('a') as f: f.write(json.dumps(rec) + '\\n')\n"
    "    seed += 1\n"
)


def _watch(tmp_path, *, limit, delay, **bounds):
    """Run a fixture worker (resuming after its highest recorded seed) under the watchdog."""
    import sys
    task_root = tmp_path / "out" / "click_bell"
    worker = tmp_path / "worker.py"
    worker.write_text(_WORKER.format(limit=limit, delay=delay))
    with (tmp_path / "worker.log").open("xb") as log:
        result = lanes.run_with_seed_watchdog(
            [sys.executable, str(worker), str(task_root)], cwd=tmp_path, env=None, log=log,
            task_root=task_root, task="click_bell", task_config="demo_clean", max_restarts=3, **bounds)
    return task_root, result


def test_seed_watchdog_rejects_a_hung_seed_and_resumes_at_the_next(tmp_path):
    """A seed with no result past the bound becomes oracle_timeout; the worker restarts after it."""
    import json
    task_root, (code, timed_out) = _watch(tmp_path, limit=100003, delay="60 if seed == 100001 else 0",
                                          limit_s=1.0, poll_s=0.1)
    assert code == 0 and timed_out == [100001]
    rows = [json.loads(l) for l in (task_root / "seed-ledger.jsonl").read_text().splitlines()]
    assert [(r["requested_seed"], r["classification"]) for r in rows] == [
        (100000, "accepted"), (100001, "oracle_timeout"), (100002, "accepted")]
    hung = json.loads((task_root / "attempts/seed-100001-try-00/attempt.json").read_text())
    assert hung["accepted"] is False and hung["recorded_by"] == lanes.WATCHDOG_RECORDER
    assert hung["limit_s"] == 1.0 and hung["first_in_process"] is False


def test_seed_watchdog_gives_a_process_first_seed_the_start_up_bound(tmp_path):
    """The first seed of a worker process carries its start-up and gets the longer bound."""
    _, (code, timed_out) = _watch(tmp_path, limit=100002, delay="1.0 if seed == 100000 else 0",
                                  limit_s=0.3, first_limit_s=5.0, poll_s=0.05)
    assert code == 0 and timed_out == []   # 1 s of start-up is under the first-seed bound


def test_seed_watchdog_bounds_are_two_and_three_hours():
    assert (lanes.SEED_WATCHDOG_S, lanes.SEED_WATCHDOG_FIRST_S) == (7200, 10800)


def test_sealing_ignores_timeouts_recorded_under_shorter_bounds():
    timeout = {"classification": "oracle_timeout", "accepted": False, "recorded_by": lanes.WATCHDOG_RECORDER}
    real = {"classification": "accepted", "accepted": True}
    records = [dict(timeout, requested_seed=100000),                                   # first watchdog: no bound recorded
               dict(real, requested_seed=100001),
               dict(timeout, requested_seed=100002, first_in_process=False, limit_s=1800),
               dict(timeout, requested_seed=100003, first_in_process=True, limit_s=5400),
               dict(timeout, requested_seed=100004, first_in_process=False, limit_s=lanes.SEED_WATCHDOG_S),
               dict(timeout, requested_seed=100005, first_in_process=True, limit_s=lanes.SEED_WATCHDOG_FIRST_S),
               dict(timeout, requested_seed=100006, first_in_process=False, limit_s=lanes.SEED_WATCHDOG_FIRST_S),
               {"classification": "infra_failure", "accepted": False, "requested_seed": 100007},
               {"classification": "invalid_artifact", "accepted": False, "requested_seed": 100007},
               {"classification": "oracle_exception", "accepted": False, "requested_seed": 100007}]
    kept = [(r["requested_seed"], r["classification"]) for r in lanes.valid_ledger_records(records)]
    # Only timeouts that ran past the bound now in force count as hung seeds;
    # infrastructure and artifact failures say nothing about the seed.
    assert kept == [(100001, "accepted"), (100004, "oracle_timeout"), (100005, "oracle_timeout"),
                    (100006, "oracle_timeout"), (100007, "oracle_exception")]


def test_a_sealed_manifest_leaves_out_records_that_say_nothing_about_the_seed(tmp_path):
    import json
    from ora.orchestrator import robotwin_evaluation as run
    from tests.test_robotwin_evaluation_setting import _ledger_record
    plan = {"robotwin_revision": "cf", "source_commit": "c" * 40, "qualification": {
        "producer_dependency_hash": "dep", "cases": [
            {"task": "click_bell", "start_requested_seed": 100000, "exclusive_seed_limit": 100003,
             "accepted_target": 2}]}}
    ledger = tmp_path / "click_bell" / "click_bell" / "seed-ledger.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text("\n".join(json.dumps(r) for r in (
        _ledger_record(100000, accepted=False, classification="infra_failure"),
        _ledger_record(100000, accepted=True, hold=(True, 250, None)),
        {"requested_seed": 100001, "accepted": False, "classification": "oracle_timeout",
         "recorded_by": lanes.WATCHDOG_RECORDER, "limit_s": 1800, "first_in_process": False},
        _ledger_record(100002, accepted=False, classification="oracle_exception"),
    )) + "\n")
    manifest = json.loads(run.seed_manifests(tmp_path, plan, "plan", "demo_clean")["click_bell"].read_text())
    assert manifest["accepted_seeds"] == [100000] and manifest["evaluated_seeds"] == [100000, 100002]
    assert [r["classification"] for r in manifest["rejected"]] == ["oracle_exception"]
