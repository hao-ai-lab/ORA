"""Lease reads never delete a live lease: not from another node, not while it is being claimed.

On a cluster a lease scan from the login node judged every live 281 lease stale and
deleted it; the A/B queue then abandoned 22 running episodes (EXP12A, 2026-10-02).
An unlocked reader also pruned a claim it read between its O_EXCL create and its
JSON write, and the launcher found its own lease gone (EXP13A g001, 2026-10-10).
"""
from __future__ import annotations

import json
import os
import socket

import pytest

from ora.orchestrator import cli, lease_claim


def _lease(path, **fields):
    value = {"schema": cli.LEASE_SCHEMA, "run_id": "run-x", "supervisor_pid": 2 ** 22 + 12345,
             "launcher_pid": 2 ** 22 + 12345, **fields}
    path.write_text(json.dumps(value))
    return path


def test_a_lease_claimed_on_another_node_is_never_judged_stale_here(tmp_path):
    path = _lease(tmp_path / "slot.json", node="some-other-node")
    lease = cli._read_resource_lease(path)
    assert lease is not None and lease["run_id"] == "run-x" and path.exists()
    assert lease_claim.lease_off_node(lease) == "some-other-node"


def test_a_dead_lease_claimed_on_this_node_is_still_pruned(tmp_path):
    path = _lease(tmp_path / "slot.json", node=socket.gethostname())
    assert cli._read_resource_lease(path) is None and not path.exists()


def test_a_live_lease_on_this_node_is_kept(tmp_path):
    path = _lease(tmp_path / "slot.json", node=socket.gethostname(), supervisor_pid=os.getpid())
    assert cli._read_resource_lease(path) is not None


def test_new_leases_record_their_node(tmp_path, monkeypatch):
    monkeypatch.setattr(lease_claim, "resource_lease_key", lambda binding: "robotwin_sim--rt-g0s0")
    binding = {"resource_class": "robotwin_sim", "slot_id": "rt-g0s0", "profile_id": "robotwin-g0s0",
               "host_id": "test-cluster", "worker_environment": {"CUDA_VISIBLE_DEVICES": "0"}}
    lease = lease_claim._write_lease(binding, run_id="run-x", plan_sha256="a" * 64, launcher_pid=os.getpid(),
                                     lease_root=tmp_path, schema=cli.LEASE_SCHEMA)
    assert lease["node"] == socket.gethostname()


BINDING = {"resource_class": "robotwin_sim", "slot_id": "rt-g0s0", "profile_id": "robotwin-g0s0",
           "host_id": "test-cluster", "worker_environment": {"CUDA_VISIBLE_DEVICES": "0"}}


def _claim(lease_root):
    return lease_claim._write_lease(BINDING, run_id="run-x", plan_sha256="a" * 64, launcher_pid=os.getpid(),
                                    lease_root=lease_root, schema=cli.LEASE_SCHEMA)


def test_a_claim_is_published_whole_so_an_unlocked_reader_never_prunes_it(tmp_path, monkeypatch):
    monkeypatch.setattr(lease_claim, "resource_lease_key", lambda binding: "robotwin_sim--rt-g0s0")
    seen, dump = [], json.dump

    def read_then_dump(value, stream, **kwargs):
        # An unlocked reader (preflight, status, a queue) scans the registry mid-claim.
        seen.append(sorted(path.name for path in tmp_path.glob("*.json")))
        for path in tmp_path.glob("*.json"):
            cli._read_resource_lease(path)
        dump(value, stream, **kwargs)

    monkeypatch.setattr(json, "dump", read_then_dump)
    lease = _claim(tmp_path)
    assert seen == [[]]
    assert cli._read_resource_lease(tmp_path / "robotwin_sim--rt-g0s0.json")["claim_token"] == lease["claim_token"]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["robotwin_sim--rt-g0s0.json"]


def test_a_held_slot_still_refuses_a_second_claim(tmp_path, monkeypatch):
    monkeypatch.setattr(lease_claim, "resource_lease_key", lambda binding: "robotwin_sim--rt-g0s0")
    first = _claim(tmp_path)
    with pytest.raises(FileExistsError):
        _claim(tmp_path)
    assert cli._read_resource_lease(tmp_path / "robotwin_sim--rt-g0s0.json")["claim_token"] == first["claim_token"]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["robotwin_sim--rt-g0s0.json"]
