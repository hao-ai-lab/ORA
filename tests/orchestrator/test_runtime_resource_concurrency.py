from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from ora.environment.rpc import (
    EnvironmentRpcError,
    LocalRpcEnvironmentSession,
    _worker_environment,
)
from ora.orchestrator import resource_inventory
from ora.orchestrator import cli
from ora.orchestrator import resources as resource_profiles
from ora.orchestrator.resources import (
    ExecutionResourceError,
    resolve_execution_resource,
)


ROOT = Path(__file__).resolve().parents[2]


def _plan(binding: dict | None, digest: str = "a" * 64) -> dict:
    runtime = {
        "kind": (
            str(binding["runtime_kind"]) if binding is not None else "libero_pro"
        ),
        "required_host": "test-host",
    }
    if binding is not None:
        runtime["resource_binding"] = binding
    return {"plan_sha256": digest, "runtime": runtime}


#: A resident-model-service slot on GPU 7 beside the source-tracked simulator
#: slots. The source host profile registers none since VLA support left the
#: tree, so co-residency across resource classes is checked on a copy of it.
MODEL_SERVICE_SLOT = """  model-service-gpu7:
    runtime_kind: model_service
    resource_class: learned_model_service
    slot_id: gpu7
    worker_environment:
      CUDA_VISIBLE_DEVICES: "7"
    minimum_free_memory_mib: 12288
"""


@pytest.fixture(autouse=True)
def model_service_host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    hosts = tmp_path / "source/ora/orchestrator/hosts"
    hosts.mkdir(parents=True)
    for path in resource_profiles.HOSTS_ROOT.glob("*.yaml"):
        text = path.read_text(encoding="utf-8")
        if path.name == "test-host.yaml":
            assert "\nexecution_resources:\n" in text
            text = text.replace("\nexecution_resources:\n", "\nexecution_resources:\n" + MODEL_SERVICE_SLOT, 1)
        (hosts / path.name).write_text(text, encoding="utf-8")
    monkeypatch.setattr(resource_profiles, "HOSTS_ROOT", hosts)
    return hosts


@pytest.fixture
def lease_registry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> Path:
    root = tmp_path / "leases"
    monkeypatch.setattr(cli, "LEASE_ROOT", root)
    monkeypatch.setattr(cli, "LEASE_LOCK_PATH", root / ".registry.lock")
    monkeypatch.setattr(cli, "LEASE_PATH", tmp_path / "active-run.json")
    return root


def test_source_tracked_resource_profiles_bind_exact_gpu_slots() -> None:
    robotwin_gpu0 = resolve_execution_resource(
        "robotwin-gpu0", runtime_kind="robotwin", required_host="test-host",
    )
    robotwin_gpu4 = resolve_execution_resource(
        "robotwin-gpu4", runtime_kind="robotwin", required_host="test-host",
    )
    robotwin_gpu3 = resolve_execution_resource(
        "robotwin-gpu3", runtime_kind="robotwin", required_host="test-host",
    )
    robotwin_gpu7 = resolve_execution_resource(
        "robotwin-gpu7", runtime_kind="robotwin", required_host="test-host",
    )
    robotwin_gpu6 = resolve_execution_resource(
        "robotwin-gpu6", runtime_kind="robotwin", required_host="test-host",
    )
    resources = {
        index: resolve_execution_resource(
            f"libero-gpu{index}",
            runtime_kind="libero_pro",
            required_host="test-host",
        )
        for index in range(1, 6)
    }
    assert {
        item["resource_class"] for item in resources.values()
    } == {"libero_pro_sim"}
    assert {
        index: item["slot_id"] for index, item in resources.items()
    } == {index: f"gpu{index}" for index in range(1, 6)}
    assert {
        index: item["worker_environment"] for index, item in resources.items()
    } == {
        index: {
            "CUDA_VISIBLE_DEVICES": str(index),
            "MUJOCO_EGL_DEVICE_ID": str(index),
        }
        for index in range(1, 6)
    }
    assert len({item["profile_sha256"] for item in resources.values()}) == 5
    service_gpu7 = resolve_execution_resource(
        "model-service-gpu7", runtime_kind="model_service", required_host="test-host",
    )
    assert robotwin_gpu0["resource_class"] == "robotwin_sim"
    assert robotwin_gpu0["slot_id"] == "gpu0"
    assert robotwin_gpu0["worker_environment"] == {"CUDA_VISIBLE_DEVICES": "0"}
    assert robotwin_gpu0["minimum_free_memory_mib"] == 20480
    assert robotwin_gpu4["resource_class"] == "robotwin_sim"
    assert robotwin_gpu4["slot_id"] == "gpu4"
    assert robotwin_gpu4["worker_environment"] == {"CUDA_VISIBLE_DEVICES": "4"}
    assert robotwin_gpu4["minimum_free_memory_mib"] == 20480
    assert robotwin_gpu0["profile_sha256"] != robotwin_gpu4["profile_sha256"]
    assert [
        robotwin_gpu3["slot_id"], robotwin_gpu6["slot_id"],
        robotwin_gpu7["slot_id"],
    ] == ["gpu3", "gpu6", "gpu7"]
    assert robotwin_gpu3["worker_environment"] == {"CUDA_VISIBLE_DEVICES": "3"}
    assert robotwin_gpu6["worker_environment"] == {"CUDA_VISIBLE_DEVICES": "6"}
    assert robotwin_gpu7["worker_environment"] == {"CUDA_VISIBLE_DEVICES": "7"}
    assert len({
        robotwin_gpu0["profile_sha256"], robotwin_gpu3["profile_sha256"],
        robotwin_gpu6["profile_sha256"],
        robotwin_gpu7["profile_sha256"],
    }) == 4
    assert service_gpu7["resource_class"] == "learned_model_service"
    assert service_gpu7["slot_id"] == "gpu7"
    assert service_gpu7["worker_environment"] == {"CUDA_VISIBLE_DEVICES": "7"}
    with pytest.raises(ExecutionResourceError, match="not robotwin"):
        resolve_execution_resource(
            "libero-gpu1", runtime_kind="robotwin", required_host="test-host",
        )


def test_single_attempt_launch_claim_is_atomic_and_plan_bound(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    workspace = tmp_path / "outputs/experiment"
    monkeypatch.setattr(cli.paths, "workspace_from_plan", lambda _plan: workspace)
    plan = {
        "experiment": "experiment",
        "config": "cell-r0",
        "plan_id": "experiment-m0-abc",
        "plan_sha256": "a" * 64,
        "source": {"commit": "source", "dirty": False},
        "launch_admission": {
            "schema": "ora.run-launch-admission.v1",
            "mode": "single_attempt_cell",
            "cohort_id": "cohort-v1",
            "cell_id": "r0",
            "retry_allowed": False,
        },
    }

    def attempt() -> tuple[str, object]:
        try:
            return "ok", cli._claim_launch_admission(plan)
        except RuntimeError as exc:
            return "blocked", str(exc)

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _index: attempt(), range(2)))
    assert [kind for kind, _value in outcomes].count("ok") == 1
    assert [kind for kind, _value in outcomes].count("blocked") == 1
    claim_path = workspace / "launch-admissions/cohort-v1/r0.json"
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    assert claim["plan_sha256"] == plan["plan_sha256"]
    successful = next(value for kind, value in outcomes if kind == "ok")
    assert isinstance(successful, dict)
    assert successful["path"] == str(claim_path)
    assert successful["claim_sha256"] == hashlib.sha256(
        claim_path.read_bytes()
    ).hexdigest()
    with pytest.raises(RuntimeError, match="forbids retries"):
        cli._require_launchable(plan, retry=True)
    cli._require_launchable(plan, retry=False)
    context_only = {
        **plan,
        "launch_admission": {
            "schema": "ora.run-launch-admission.v1",
            "mode": "disabled",
        },
    }
    with pytest.raises(RuntimeError, match="context-only"):
        cli._require_launchable(context_only, retry=False)


def test_robotwin_gpu0_coexists_with_libero_gpu_slots(
    lease_registry: Path,
) -> None:
    robotwin = resolve_execution_resource(
        "robotwin-gpu0", runtime_kind="robotwin", required_host="test-host",
    )
    libero = resolve_execution_resource(
        "libero-gpu1", runtime_kind="libero_pro", required_host="test-host",
    )
    first = cli._create_lease(_plan(robotwin), "run-robotwin-gpu0", os.getpid())
    second = cli._create_lease(_plan(libero), "run-libero-gpu1", os.getpid())
    assert {item["run_id"] for item in cli._active_leases()} == {
        "run-robotwin-gpu0", "run-libero-gpu1",
    }
    cli._release_lease(
        "run-robotwin-gpu0", claim_token=str(first["claim_token"]),
    )
    cli._release_lease(
        "run-libero-gpu1", claim_token=str(second["claim_token"]),
    )
    assert list(lease_registry.glob("*.json")) == []


def test_three_robotwin_slots_can_be_leased_together(
    lease_registry: Path,
) -> None:
    bindings = [
        resolve_execution_resource(
            f"robotwin-gpu{gpu}", runtime_kind="robotwin",
            required_host="test-host",
        )
        for gpu in (6, 3, 7)
    ]
    leases = [
        cli._create_lease(_plan(binding), f"run-gpu{gpu}", os.getpid())
        for binding, gpu in zip(bindings, (6, 3, 7))
    ]
    assert {item["run_id"] for item in cli._active_leases()} == {
        "run-gpu6", "run-gpu3", "run-gpu7",
    }
    with pytest.raises(RuntimeError, match="already leased"):
        cli._create_lease(_plan(bindings[1]), "run-gpu3-duplicate", os.getpid())
    for lease in leases:
        cli._release_lease(
            str(lease["run_id"]), claim_token=str(lease["claim_token"]),
        )
    assert list(lease_registry.glob("*.json")) == []


def test_disjoint_slots_coexist_and_same_slot_is_exclusive(
    lease_registry: Path,
) -> None:
    gpu1 = resolve_execution_resource(
        "libero-gpu1", runtime_kind="libero_pro", required_host="test-host",
    )
    gpu2 = resolve_execution_resource(
        "libero-gpu2", runtime_kind="libero_pro", required_host="test-host",
    )
    first = cli._create_lease(_plan(gpu1), "run-gpu1", os.getpid())
    second = cli._create_lease(_plan(gpu2), "run-gpu2", os.getpid())
    assert {item["run_id"] for item in cli._active_leases()} == {
        "run-gpu1", "run-gpu2",
    }
    with pytest.raises(RuntimeError, match="already leased"):
        cli._create_lease(_plan(gpu1), "run-gpu1-b", os.getpid())
    cli._release_lease("run-gpu1", claim_token=str(first["claim_token"]))
    cli._release_lease("run-gpu2", claim_token=str(second["claim_token"]))
    assert list(lease_registry.glob("*.json")) == []


def test_five_disjoint_libero_slots_can_coexist(lease_registry: Path) -> None:
    resources = [
        resolve_execution_resource(
            f"libero-gpu{index}",
            runtime_kind="libero_pro",
            required_host="test-host",
        )
        for index in range(1, 6)
    ]
    leases = [
        cli._create_lease(_plan(binding), f"run-gpu{index}", os.getpid())
        for index, binding in enumerate(resources, start=1)
    ]
    assert {item["run_id"] for item in cli._active_leases()} == {
        f"run-gpu{index}" for index in range(1, 6)
    }
    for index, lease in enumerate(leases, start=1):
        cli._release_lease(
            f"run-gpu{index}", claim_token=str(lease["claim_token"]),
        )


def _gpu_memory(
    monkeypatch: pytest.MonkeyPatch, *, free_mib: int, total_mib: int,
    index: str = "7", leased_mib: dict[str, int] | None = None,
) -> None:
    """Pin the claim-time GPU probe instead of reading the live host.

    ``leased_mib`` is what the probe attributed to our own leases' process
    trees, keyed by ``resource_class--slot_id``. Everything else in use on the
    device stays unattributed and is charged against the claim.
    """

    held = dict(leased_mib or {})
    monkeypatch.setattr(
        cli, "_gpu_memory_inventory",
        lambda owner_pids=None: {
            index: {
                "free_mib": free_mib, "total_mib": total_mib,
                "leased_mib": dict(held),
            },
        },
    )


def _unavailable_gpu_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    def _refuse(owner_pids: object = None) -> dict:
        raise ExecutionResourceError("nvidia-smi is unavailable")

    monkeypatch.setattr(cli, "_gpu_memory_inventory", _refuse)


def _cross_class_pair() -> tuple[dict, dict]:
    robotwin = resolve_execution_resource(
        "robotwin-gpu7", runtime_kind="robotwin", required_host="test-host",
    )
    service = resolve_execution_resource(
        "model-service-gpu7", runtime_kind="model_service", required_host="test-host",
    )
    assert robotwin["worker_environment"]["CUDA_VISIBLE_DEVICES"] == "7"
    assert service["worker_environment"]["CUDA_VISIBLE_DEVICES"] == "7"
    assert robotwin["minimum_free_memory_mib"] == 20480
    assert service["minimum_free_memory_mib"] == 12288
    return robotwin, service


def test_co_residency_is_admitted_when_declared_floors_fit(
    lease_registry: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    robotwin, service = _cross_class_pair()
    _gpu_memory(monkeypatch, free_mib=45000, total_mib=46068)
    first = cli._create_lease(_plan(robotwin), "run-robotwin", os.getpid())
    assert first["minimum_free_memory_mib"] == 20480
    second = cli._create_lease(_plan(service), "service-model", os.getpid())
    assert {item["run_id"] for item in cli._active_leases()} == {
        "run-robotwin", "service-model",
    }
    cli._release_lease("run-robotwin", claim_token=str(first["claim_token"]))
    cli._release_lease("service-model", claim_token=str(second["claim_token"]))


def test_foreign_device_memory_is_charged_against_a_co_resident_claim(
    lease_registry: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact state the retired total-memory arithmetic admitted.

    A 46068 MiB card carries a foreign 30000 MiB process, so 16068 MiB is
    free. One RoboTwin lease is held and declares 20480 MiB, but its worker
    has not allocated yet, so the probe attributes nothing on the device to
    us. A 12288 MiB claim arrives. Both retired checks passed --
    ``20480 + 12288 <= 46068`` against the device total, and
    ``16068 >= 12288`` for the claimant alone -- yet the card cannot hold the
    RoboTwin worker once it allocates. Capacity is now measured against what
    the device can still give: 16068 MiB free plus 0 MiB our own leases hold.
    """

    robotwin, service = _cross_class_pair()
    _gpu_memory(monkeypatch, free_mib=16068, total_mib=46068)
    first = cli._create_lease(_plan(robotwin), "run-robotwin", os.getpid())
    # The retired rule's own arithmetic, spelled out so the regression is
    # visible if anyone reinstates it.
    assert 20480 + 12288 <= 46068 and 16068 >= 12288
    with pytest.raises(RuntimeError) as blocked:
        cli._create_lease(_plan(service), "service-model", os.getpid())
    message = str(blocked.value)
    assert "rule=device_capacity" in message
    assert "16068 MiB free plus 0 MiB already held by its own leases" in message
    assert "16068 MiB admissible" in message
    assert "32768 MiB required" in message
    assert "unattributed use" in message
    cli._release_lease("run-robotwin", claim_token=str(first["claim_token"]))


def test_our_own_leases_allocation_is_credited_back_but_capped_at_its_floor(
    lease_registry: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Attributed usage is admissible again; overshoot buys nothing.

    Same 16068 MiB free card, but now the 30000 MiB belongs to the RoboTwin
    lease's own process tree. Its declared 20480 MiB floor is charged in full
    on the left, so at most 20480 MiB of what it holds may be credited back --
    crediting the whole 30000 MiB would let a worker that overshot its floor
    buy capacity the device does not have.
    """

    robotwin, service = _cross_class_pair()
    _gpu_memory(
        monkeypatch, free_mib=16068, total_mib=46068,
        leased_mib={"robotwin_sim--gpu7": 30000},
    )
    first = cli._create_lease(_plan(robotwin), "run-robotwin", os.getpid())
    # 16068 free + min(20480, 30000) credited = 36548 >= 20480 + 12288.
    second = cli._create_lease(_plan(service), "service-model", os.getpid())
    cli._release_lease("run-robotwin", claim_token=str(first["claim_token"]))
    cli._release_lease("service-model", claim_token=str(second["claim_token"]))
    # One MiB less free and the uncapped credit would still have admitted it.
    _gpu_memory(
        monkeypatch, free_mib=12287, total_mib=46068,
        leased_mib={"robotwin_sim--gpu7": 30000},
    )
    third = cli._create_lease(_plan(robotwin), "run-robotwin-b", os.getpid())
    with pytest.raises(RuntimeError) as blocked:
        cli._create_lease(_plan(service), "service-model-b", os.getpid())
    assert "rule=device_capacity" in str(blocked.value)
    assert "32767 MiB admissible" in str(blocked.value)
    cli._release_lease("run-robotwin-b", claim_token=str(third["claim_token"]))


def test_co_residency_refused_when_declared_floors_exceed_the_device(
    lease_registry: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    robotwin, service = _cross_class_pair()
    _gpu_memory(monkeypatch, free_mib=32000, total_mib=32767)
    first = cli._create_lease(_plan(robotwin), "run-robotwin", os.getpid())
    with pytest.raises(RuntimeError) as blocked:
        cli._create_lease(_plan(service), "service-model", os.getpid())
    message = str(blocked.value)
    assert "execution device is full" in message
    assert "rule=device_capacity" in message
    assert "GPU 7 has 32000 MiB free" in message
    assert "32767 MiB total" in message
    assert "1 active lease(s) declare 20480 MiB" in message
    assert "needs 12288 MiB more (32768 MiB required)" in message
    assert "already leased" not in message
    cli._release_lease("run-robotwin", claim_token=str(first["claim_token"]))


def test_co_residency_refused_without_a_live_gpu_probe(
    lease_registry: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing or failing nvidia-smi fails closed, never open."""

    robotwin, service = _cross_class_pair()
    first = cli._create_lease(_plan(robotwin), "run-robotwin", os.getpid())
    _unavailable_gpu_probe(monkeypatch)
    with pytest.raises(RuntimeError) as blocked:
        cli._create_lease(_plan(service), "service-model", os.getpid())
    assert "rule=device_probe_unavailable" in str(blocked.value)
    assert "nvidia-smi is unavailable" in str(blocked.value)
    # Re-patch rather than monkeypatch.undo(): undo would also revert the
    # lease_registry fixture and point the claim at the live host registry.
    _gpu_memory(monkeypatch, free_mib=45000, total_mib=46068)
    second = cli._create_lease(_plan(service), "service-model", os.getpid())
    cli._release_lease("run-robotwin", claim_token=str(first["claim_token"]))
    cli._release_lease("service-model", claim_token=str(second["claim_token"]))


def test_probe_failures_do_not_echo_driver_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A refusal names the failure, not whatever the driver printed.

    A wedged driver writes unbounded text to stderr, and a claim refusal is
    operator-facing and reaches run evidence.
    """

    fake = tmp_path / "nvidia-smi"
    fake.write_text(
        "#!/bin/sh\n"
        "echo \"Unable to determine the device handle: $(head -c 4096 </dev/zero"
        " | tr '\\0' 'x')\" >&2\n"
        "exit 9\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    monkeypatch.setattr(cli.shutil, "which", lambda _name: str(fake))
    with pytest.raises(ExecutionResourceError) as failed:
        resource_inventory._nvidia_smi_rows("--query-gpu=index")
    assert str(failed.value) == "nvidia-smi exited 9"


def test_same_class_co_residency_is_priced_like_every_other_neighbour(
    lease_registry: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capacity applies to every co-resident claim, whatever the classes are.

    Two slots of one class consume the same memory two classes do, so the
    accounting cannot be skipped for them: a same-class claim on a device that
    is already full is refused, and it depends on the same live probe.
    """

    first_slot = resolve_execution_resource(
        "robotwin-gpu7", runtime_kind="robotwin", required_host="test-host",
    )
    second_slot = resolve_execution_resource(
        "robotwin-gpu7b", runtime_kind="robotwin", required_host="test-host",
    )
    _gpu_memory(monkeypatch, free_mib=45000, total_mib=46068)
    first = cli._create_lease(_plan(first_slot), "run-gpu7", os.getpid())
    second = cli._create_lease(_plan(second_slot), "run-gpu7b", os.getpid())
    with pytest.raises(RuntimeError) as taken:
        cli._create_lease(_plan(second_slot), "run-gpu7b-again", os.getpid())
    assert "rule=slot_lease" in str(taken.value)
    assert "already leased" in str(taken.value)
    cli._release_lease("run-gpu7b", claim_token=str(second["claim_token"]))
    # The same same-class claim on a device a foreign process has filled.
    _gpu_memory(monkeypatch, free_mib=4000, total_mib=46068)
    with pytest.raises(RuntimeError) as full:
        cli._create_lease(_plan(second_slot), "run-gpu7b-full", os.getpid())
    assert "rule=device_capacity" in str(full.value)
    # And with no live probe at all it fails closed exactly as a cross-class
    # claim does, rather than keeping a guarantee the arithmetic cannot hold.
    _unavailable_gpu_probe(monkeypatch)
    with pytest.raises(RuntimeError) as blind:
        cli._create_lease(_plan(second_slot), "run-gpu7b-blind", os.getpid())
    assert "rule=device_probe_unavailable" in str(blind.value)
    cli._release_lease("run-gpu7", claim_token=str(first["claim_token"]))


def test_a_claim_that_shares_no_device_never_reaches_the_probe(
    lease_registry: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The uncontended path is unchanged: no probe, no nvidia-smi dependency."""

    robotwin = resolve_execution_resource(
        "robotwin-gpu7", runtime_kind="robotwin", required_host="test-host",
    )
    libero = resolve_execution_resource(
        "libero-gpu1", runtime_kind="libero_pro", required_host="test-host",
    )

    def _explode(owner_pids: object = None) -> dict:
        raise AssertionError("an uncontended claim must not probe the device")

    monkeypatch.setattr(cli, "_gpu_memory_inventory", _explode)
    first = cli._create_lease(_plan(robotwin), "run-robotwin", os.getpid())
    second = cli._create_lease(_plan(libero), "run-libero", os.getpid())
    cli._release_lease("run-robotwin", claim_token=str(first["claim_token"]))
    cli._release_lease("run-libero", claim_token=str(second["claim_token"]))


def test_the_device_probe_runs_before_the_registry_lock_is_taken(
    lease_registry: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wedged driver must not stall the shared lease registry.

    The claim reads the authoritative lease set under the lock, releases it to
    probe, and decides again under a fresh lock, so the probe never runs while
    the registry is held and the decision is still made against a locked read.
    """

    order: list[str] = []
    robotwin, service = _cross_class_pair()
    first = cli._create_lease(_plan(robotwin), "run-robotwin", os.getpid())
    active_leases = cli._active_leases

    def _record_leases() -> list:
        order.append("locked_read")
        return active_leases()

    def _probe(owner_pids: object = None) -> dict:
        order.append("probe")
        return {
            "7": {"free_mib": 45000, "total_mib": 46068, "leased_mib": {}},
        }

    monkeypatch.setattr(cli, "_active_leases", _record_leases)
    monkeypatch.setattr(cli, "_gpu_memory_inventory", _probe)
    second = cli._create_lease(_plan(service), "service-model", os.getpid())
    assert order == ["locked_read", "probe", "locked_read"]
    monkeypatch.setattr(cli, "_active_leases", active_leases)
    cli._release_lease("run-robotwin", claim_token=str(first["claim_token"]))
    cli._release_lease("service-model", claim_token=str(second["claim_token"]))


def test_device_memory_is_attributed_to_the_lease_that_owns_the_process(
    lease_registry: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only memory held by our own leases' process trees is credited."""

    lease_registry.mkdir(parents=True)
    (lease_registry / "robotwin_sim--gpu7.json").write_text(json.dumps({
        "schema": cli.LEASE_SCHEMA,
        "run_id": "run-robotwin",
        "resource_class": "robotwin_sim",
        "slot_id": "gpu7",
        "profile_id": "robotwin-gpu7",
        "minimum_free_memory_mib": 20480,
        "worker_environment": {"CUDA_VISIBLE_DEVICES": "7"},
        "launcher_pid": os.getpid(),
        "supervisor_pid": os.getpid(),
    }))
    assert resource_inventory.lease_owner_pids(cli.LEASE_ROOT) == {os.getpid(): "robotwin_sim--gpu7"}
    worker = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
    )
    try:
        assert resource_inventory._process_ancestry(worker.pid)[:2] == [
            worker.pid, os.getpid(),
        ]
        rows = {
            "--query-gpu=index,uuid,memory.free,memory.total": [
                ["7", "GPU-seven", "16068", "46068"],
            ],
            "--query-compute-apps=gpu_uuid,pid,used_memory": [
                # Our own leased worker, a foreign process, and a device we
                # do not know about.
                ["GPU-seven", str(worker.pid), "6000"],
                ["GPU-seven", "1", "30000"],
                ["GPU-other", str(worker.pid), "1000"],
            ],
        }
        monkeypatch.setattr(resource_inventory, "_nvidia_smi_rows", lambda query: rows[query])
        inventory = cli._gpu_memory_inventory(resource_inventory.lease_owner_pids(cli.LEASE_ROOT))
    finally:
        worker.terminate()
        worker.wait(timeout=30)
    assert inventory == {
        "7": {
            "free_mib": 16068, "total_mib": 46068,
            "leased_mib": {"robotwin_sim--gpu7": 6000},
        },
    }


def test_fixture_host_card_patterns_still_fit_the_uniform_capacity_rule(
    lease_registry: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pricing every co-resident claim must not refuse what the fixture host already runs.

    Cards 0, 3 and 4 register the heaviest mix: one 20480 MiB RoboTwin slot,
    two 6144 MiB ones and five 2048 MiB LIBERO slots, 43008 MiB declared in
    total against a 49140 MiB card. Cards 6 and 7 register the largest
    declared sum, 20480 + 4x6144 = 45056 MiB. Both are checked against the
    strictest snapshot: an idle card whose 653 MiB is the driver's own
    reserve, where no worker has allocated anything the rule could credit.
    """

    idle_free = 49140 - 653
    _gpu_memory(monkeypatch, free_mib=idle_free, total_mib=49140, index="0")
    card0 = [
        resolve_execution_resource(
            profile, runtime_kind=kind, required_host="test-host",
        )
        for profile, kind in (
            ("robotwin-gpu0", "robotwin"), ("robotwin-gpu0b", "robotwin"),
            ("robotwin-gpu0c", "robotwin"), ("libero-gpu0", "libero_pro"),
            ("libero-gpu0b", "libero_pro"), ("libero-gpu0c", "libero_pro"),
            ("libero-gpu0d", "libero_pro"), ("libero-gpu0e", "libero_pro"),
        )
    ]
    assert sum(item["minimum_free_memory_mib"] for item in card0) == 43008
    leases = [
        cli._create_lease(_plan(binding), f"run-card0-{index}", os.getpid())
        for index, binding in enumerate(card0)
    ]
    assert len(cli._active_leases()) == 8
    for lease in leases:
        cli._release_lease(
            str(lease["run_id"]), claim_token=str(lease["claim_token"]),
        )

    _gpu_memory(monkeypatch, free_mib=idle_free, total_mib=49140, index="6")
    card6 = [
        resolve_execution_resource(
            profile, runtime_kind="robotwin", required_host="test-host",
        )
        for profile in (
            "robotwin-gpu6", "robotwin-gpu6b", "robotwin-gpu6c",
            "robotwin-gpu6d", "robotwin-gpu6e",
        )
    ]
    assert sum(item["minimum_free_memory_mib"] for item in card6) == 45056
    leases = [
        cli._create_lease(_plan(binding), f"run-card6-{index}", os.getpid())
        for index, binding in enumerate(card6)
    ]
    # the fixture host registers no LIBERO slot on card 6, so this is the ruling's
    # arithmetic rather than a registered profile: 45056 + 2048 = 47104 MiB
    # against 48487 MiB admissible on an otherwise idle 49140 MiB card.
    libero_shaped = {
        "schema": "ora.execution-resource-binding.v1",
        "profile_id": "libero-gpu6-hypothetical",
        "host_id": "test-host",
        "runtime_kind": "libero_pro",
        "resource_class": "libero_pro_sim",
        "slot_id": "gpu6",
        "worker_environment": {
            "CUDA_VISIBLE_DEVICES": "6", "MUJOCO_EGL_DEVICE_ID": "6",
        },
        "minimum_free_memory_mib": 2048,
    }
    assert cli._resource_refusal(
        libero_shaped, cli._active_leases(),
        inventory={
            "6": {
                "free_mib": idle_free, "total_mib": 49140, "leased_mib": {},
            },
        },
    ) is None
    for lease in leases:
        cli._release_lease(
            str(lease["run_id"]), claim_token=str(lease["claim_token"]),
        )


def test_a_binding_without_a_usable_floor_refuses_co_residency(
    lease_registry: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unpriceable claimant fails closed, exactly like an unpriceable lease."""

    robotwin, _service = _cross_class_pair()
    first = cli._create_lease(_plan(robotwin), "run-robotwin", os.getpid())
    unpriced = {
        "schema": "ora.execution-resource-binding.v1",
        "profile_id": "unregistered-gpu7",
        "host_id": "test-host",
        "runtime_kind": "model_service",
        "resource_class": "learned_model_service",
        "slot_id": "gpu7x",
        "worker_environment": {"CUDA_VISIBLE_DEVICES": "7"},
        "minimum_free_memory_mib": None,
    }
    refusal = cli._resource_refusal(
        unpriced, cli._active_leases(),
        inventory={
            "7": {"free_mib": 45000, "total_mib": 46068, "leased_mib": {}},
        },
    )
    assert "rule=missing_declared_floor" in str(refusal)
    assert "unregistered-gpu7" in str(refusal)
    # The same claim on an uncontended device keeps its previous path.
    assert cli._resource_refusal(
        {**unpriced, "worker_environment": {"CUDA_VISIBLE_DEVICES": "5"}},
        cli._active_leases(),
    ) is None
    cli._release_lease("run-robotwin", claim_token=str(first["claim_token"]))


def test_a_device_that_is_not_an_nvidia_smi_index_refuses_co_residency(
    lease_registry: Path,
) -> None:
    """MIG and UUID device selectors are refused, never index-matched."""

    robotwin, service = _cross_class_pair()
    first = cli._create_lease(_plan(robotwin), "run-robotwin", os.getpid())
    # A binding that names only a MIG or UUID device shares no device string
    # with an index-named lease, so it is not even seen as co-resident; that
    # device-identity gap predates this rule and is recorded in the reference.
    both = {
        **service,
        "worker_environment": {"CUDA_VISIBLE_DEVICES": "7,MIG-GPU-7/1/0"},
    }
    refusal = cli._resource_refusal(
        both, cli._active_leases(),
        inventory={
            "7": {"free_mib": 45000, "total_mib": 46068, "leased_mib": {}},
        },
    )
    assert "rule=device_identity_unsupported" in str(refusal)
    cli._release_lease("run-robotwin", claim_token=str(first["claim_token"]))


def test_active_lease_without_a_recorded_floor_is_priced_from_its_profile(
    lease_registry: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Records that predate the lease field fall back to the host profile."""

    robotwin, service = _cross_class_pair()
    lease_registry.mkdir(parents=True)
    (lease_registry / "robotwin_sim--gpu7.json").write_text(json.dumps({
        "schema": cli.LEASE_SCHEMA,
        "run_id": "legacy-record",
        "resource_class": "robotwin_sim",
        "slot_id": "gpu7",
        "profile_id": "robotwin-gpu7",
        "host_id": "test-host",
        "runtime_kind": "robotwin",
        "worker_environment": {"CUDA_VISIBLE_DEVICES": "7"},
        "launcher_pid": os.getpid(),
        "supervisor_pid": None,
    }))
    assert cli._memory_floor_mib(cli._active_leases()[0]) == 20480
    _gpu_memory(monkeypatch, free_mib=32000, total_mib=46068)
    with pytest.raises(RuntimeError) as blocked:
        cli._create_lease(_plan(service), "service-model", os.getpid())
    assert "1 active lease(s) declare 20480 MiB" in str(blocked.value)
    _gpu_memory(monkeypatch, free_mib=45000, total_mib=46068)
    claimed = cli._create_lease(_plan(service), "service-model", os.getpid())
    cli._release_lease(
        "service-model", claim_token=str(claimed["claim_token"]),
    )
    assert robotwin["profile_id"] == "robotwin-gpu7"


def test_active_lease_whose_profile_hash_drifted_is_not_repriced(
    lease_registry: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Another checkout must not reprice a live lease by editing its profile.

    A record without the floor field is priced by re-resolving the profile it
    names, but only when the record's own ``profile_sha256`` still matches
    what this source resolves. A repriced or renamed profile is unpriceable,
    not cheaper.
    """

    _robotwin, service = _cross_class_pair()
    lease_registry.mkdir(parents=True)
    (lease_registry / "robotwin_sim--gpu7.json").write_text(json.dumps({
        "schema": cli.LEASE_SCHEMA,
        "run_id": "repriced-record",
        "resource_class": "robotwin_sim",
        "slot_id": "gpu7",
        "profile_id": "robotwin-gpu7",
        "host_id": "test-host",
        "runtime_kind": "robotwin",
        "worker_environment": {"CUDA_VISIBLE_DEVICES": "7"},
        "profile_sha256": "0" * 64,
        "launcher_pid": os.getpid(),
        "supervisor_pid": None,
    }))
    assert cli._memory_floor_mib(cli._active_leases()[0]) is None
    _gpu_memory(monkeypatch, free_mib=45000, total_mib=46068)
    with pytest.raises(RuntimeError) as blocked:
        cli._create_lease(_plan(service), "service-model", os.getpid())
    assert "rule=unpriced_active_lease" in str(blocked.value)
    assert "repriced-record" in str(blocked.value)


def test_active_lease_with_an_unresolvable_profile_refuses_co_residency(
    lease_registry: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unpriceable neighbour fails closed instead of counting as zero."""

    _robotwin, service = _cross_class_pair()
    lease_registry.mkdir(parents=True)
    (lease_registry / "robodojo_sim--gpu7.json").write_text(json.dumps({
        "schema": cli.LEASE_SCHEMA,
        "run_id": "foreign-source-record",
        "resource_class": "robodojo_sim",
        "slot_id": "gpu7",
        "profile_id": "retired-unregistered-robodojo-gpu7",
        "host_id": "test-host",
        "runtime_kind": "robodojo",
        "worker_environment": {"CUDA_VISIBLE_DEVICES": "7"},
        "launcher_pid": os.getpid(),
        "supervisor_pid": None,
    }))
    _gpu_memory(monkeypatch, free_mib=45000, total_mib=46068)
    with pytest.raises(RuntimeError) as blocked:
        cli._create_lease(_plan(service), "service-model", os.getpid())
    message = str(blocked.value)
    assert "rule=unpriced_active_lease" in message
    assert "foreign-source-record" in message


def test_exclusive_host_refuses_and_is_refused_in_both_directions(
    lease_registry: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _gpu_memory(monkeypatch, free_mib=40000, total_mib=46068)
    robotwin, _service = _cross_class_pair()
    slot = cli._create_lease(_plan(robotwin), "run-robotwin", os.getpid())
    with pytest.raises(RuntimeError) as blocked:
        cli._create_lease(_plan(None, digest="d" * 64), "run-x", os.getpid())
    assert "rule=exclusive_host" in str(blocked.value)
    cli._release_lease("run-robotwin", claim_token=str(slot["claim_token"]))
    exclusive = cli._create_lease(
        _plan(None, digest="d" * 64), "run-exclusive", os.getpid(),
    )
    with pytest.raises(RuntimeError) as reverse:
        cli._create_lease(_plan(robotwin), "run-robotwin-b", os.getpid())
    assert "rule=exclusive_host" in str(reverse.value)
    cli._release_lease(
        "run-exclusive", claim_token=str(exclusive["claim_token"]),
    )


def test_concurrent_same_slot_claim_has_exactly_one_winner(
    lease_registry: Path,
) -> None:
    gpu1 = resolve_execution_resource(
        "libero-gpu1", runtime_kind="libero_pro", required_host="test-host",
    )
    barrier = threading.Barrier(2)

    def claim(run_id: str) -> dict | str:
        barrier.wait()
        try:
            return cli._create_lease(_plan(gpu1), run_id, os.getpid())
        except RuntimeError as exc:
            return str(exc)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(claim, ("race-a", "race-b")))
    winners = [item for item in results if isinstance(item, dict)]
    failures = [item for item in results if isinstance(item, str)]
    assert len(winners) == len(failures) == 1
    assert "already leased" in failures[0]
    cli._release_lease(
        str(winners[0]["run_id"]),
        claim_token=str(winners[0]["claim_token"]),
    )


def test_implicit_exclusive_and_legacy_leases_block_all_slots(
    lease_registry: Path,
) -> None:
    gpu1 = resolve_execution_resource(
        "libero-gpu1", runtime_kind="libero_pro", required_host="test-host",
    )
    exclusive = cli._create_lease(_plan(None), "run-exclusive", os.getpid())
    with pytest.raises(RuntimeError, match="already leased"):
        cli._create_lease(_plan(gpu1), "run-gpu1", os.getpid())
    cli._release_lease(
        "run-exclusive", claim_token=str(exclusive["claim_token"]),
    )
    cli.LEASE_PATH.write_text(json.dumps({
        "run_id": "legacy-run", "pid": os.getpid(),
    }))
    with pytest.raises(RuntimeError, match="legacy-run"):
        cli._create_lease(_plan(gpu1), "run-gpu1", os.getpid())
    cli.LEASE_PATH.unlink()


def test_runspec_cannot_self_authorize_an_exclusive_conflict(
    lease_registry: Path,
) -> None:
    exclusive = cli._create_lease(
        _plan(None, digest="b" * 64), "run-exclusive", os.getpid(),
    )
    gpu1 = resolve_execution_resource(
        "libero-gpu1", runtime_kind="libero_pro", required_host="test-host",
    )
    attempted = _plan(gpu1, digest="c" * 64)
    attempted["runtime"]["resource_concurrency_override"] = {
        "authorized": True,
        "authorization_id": "self-asserted",
    }
    with pytest.raises(RuntimeError, match="already leased"):
        cli._create_lease(attempted, "run-gpu1", os.getpid())
    cli._release_lease(
        "run-exclusive", claim_token=str(exclusive["claim_token"]),
    )


def test_stale_slot_is_reclaimed(lease_registry: Path) -> None:
    gpu1 = resolve_execution_resource(
        "libero-gpu1", runtime_kind="libero_pro", required_host="test-host",
    )
    lease_registry.mkdir(parents=True)
    stale = lease_registry / "libero_pro_sim--gpu1.json"
    stale.write_text(json.dumps({
        "schema": cli.LEASE_SCHEMA,
        "run_id": "stale",
        "launcher_pid": 999_999_999,
        "supervisor_pid": None,
    }))
    current = cli._create_lease(_plan(gpu1), "current", os.getpid())
    assert not any(item["run_id"] == "stale" for item in cli._active_leases())
    cli._release_lease("current", claim_token=str(current["claim_token"]))


def _rpc_session(root: Path, name: str, gpu: str) -> LocalRpcEnvironmentSession:
    output = root / name
    return LocalRpcEnvironmentSession(
        python_executable=Path(sys.executable),
        project_root=ROOT,
        output_root=output,
        socket_root=root / "transport",
        worker_config={
            "backend": "synthetic", "output_root": str(output),
            "task": name, "seed": int(gpu),
        },
        worker_environment={"CUDA_VISIBLE_DEVICES": gpu},
        startup_timeout_s=10,
        request_timeout_s=10,
    )


def test_two_local_rpc_sessions_use_independent_config_and_socket_paths(
    tmp_path: Path,
) -> None:
    first = _rpc_session(tmp_path, "first", "1")
    second = _rpc_session(tmp_path, "second", "2")
    with ThreadPoolExecutor(max_workers=2) as pool:
        resets = list(pool.map(lambda session: session.start(), (first, second)))
    try:
        assert {item["task"] for item in resets} == {"first", "second"}
        assert first.transport_root != second.transport_root
        assert first.socket_path.is_socket() and second.socket_path.is_socket()
        assert first.config_path.is_file() and second.config_path.is_file()
        first_config = json.loads(first.config_path.read_text())
        second_config = json.loads(second.config_path.read_text())
        assert first_config["session_id"] == first.session_id
        assert second_config["session_id"] == second.session_id
        assert first_config["task"] == "first"
        assert second_config["task"] == "second"
    finally:
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda session: session.close(), (first, second)))
    assert not first.transport_root.exists()
    assert not second.transport_root.exists()


def test_local_rpc_falls_back_to_short_socket_for_long_worktree_path(
    tmp_path: Path,
) -> None:
    output = tmp_path / "native"
    requested = tmp_path / ("long-worktree-component-" * 6) / "transport"
    session = LocalRpcEnvironmentSession(
        python_executable=Path(sys.executable),
        project_root=ROOT,
        output_root=output,
        socket_root=requested,
        worker_config={
            "backend": "synthetic", "output_root": str(output),
            "task": "long-socket", "seed": 0,
        },
        startup_timeout_s=10,
        request_timeout_s=10,
    )
    assert session.socket_root_fallback is True
    assert session.socket_root != requested.resolve()
    assert len(os.fsencode(str(session.socket_path))) <= 107
    selected = session.transport_root
    try:
        session.start()
        record = json.loads((output / "environment-session.json").read_text())
        assert record["socket_root_requested"] == str(requested.resolve())
        assert record["socket_root_selected"] == str(session.socket_root)
        assert record["socket_root_fallback"] is True
        assert Path(record["socket"]).is_socket()
    finally:
        session.close()
    assert not selected.exists()


def test_worker_environment_overrides_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NATIVE_MODEL_API_KEY", "secret")
    environment = _worker_environment(
        ROOT, {"CUDA_VISIBLE_DEVICES": "2", "MUJOCO_EGL_DEVICE_ID": "2"},
    )
    assert environment["CUDA_VISIBLE_DEVICES"] == "2"
    assert environment["MUJOCO_EGL_DEVICE_ID"] == "2"
    assert "NATIVE_MODEL_API_KEY" not in environment
    with pytest.raises(EnvironmentRpcError, match="not allowlisted"):
        _worker_environment(ROOT, {"NATIVE_MODEL_API_KEY": "secret"})


def test_host_profiles_declare_each_resource_name_once() -> None:
    """A repeated key silently replaces the earlier profile.

    YAML keeps the last duplicate, so re-declaring an existing slot with a
    different memory floor changes the original binding with no error and no
    diff at the point of use - which is how a 20480 MiB floor briefly became
    6144 while a new set of co-resident slots was being added.
    """

    import collections
    import re
    from pathlib import Path

    source = Path("tests/data/hosts/test-host.yaml").read_text(
        encoding="utf-8",
    )
    names = re.findall(r"^  ([A-Za-z][\w.-]*):$", source, re.M)
    duplicated = sorted(
        name for name, count in collections.Counter(names).items() if count > 1
    )
    assert duplicated == []


def test_same_physical_gpu_cross_class_claim_requires_measured_capacity(
    lease_registry: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    robotwin = resolve_execution_resource(
        "robotwin-gpu0", runtime_kind="robotwin", required_host="test-host",
    )
    alias = {
        **robotwin,
        "profile_id": "model-gpu0",
        "runtime_kind": "model_service",
        "resource_class": "learned_model_service",
    }
    first = cli._create_lease(_plan(robotwin), "run-robotwin", os.getpid())
    monkeypatch.setattr(cli, "_gpu_memory_inventory", lambda owner_pids=None: {
        "0": {"total_mib": 48 * 1024, "free_mib": 0},
    })
    with pytest.raises(RuntimeError, match="device_capacity"):
        cli._create_lease(_plan(alias), "service-model", os.getpid())
    monkeypatch.setattr(cli, "_gpu_memory_inventory", lambda owner_pids=None: {
        "0": {"total_mib": 48 * 1024, "free_mib": 48 * 1024},
    })
    second = cli._create_lease(_plan(alias), "service-model", os.getpid())
    assert len(cli._active_leases()) == 2
    cli._release_lease("service-model", claim_token=str(second["claim_token"]))
    cli._release_lease(
        "run-robotwin", claim_token=str(first["claim_token"]),
    )


def test_a_live_run_must_declare_its_execution_host(monkeypatch) -> None:
    from ora.orchestrator.checks import execution_host_check
    from ora.orchestrator.resources import ExecutionResourceError, declared_execution_host

    undeclared = execution_host_check({"kind": "robotwin"}, "test-host")
    assert undeclared["passed"] is False
    assert undeclared["detail"]["expected"] == "runtime.required_host is not declared"
    assert execution_host_check({"required_host": "test-arm"}, "test-host")["passed"] is False
    assert execution_host_check({"required_host": "test-arm"}, "test-arm")["passed"] is True
    # A matching marker is not enough: the host must have a source-tracked profile.
    unregistered = execution_host_check({"required_host": "unregistered-host"}, "unregistered-host")
    assert unregistered["passed"] is False and "unavailable" in unregistered["detail"]["profile"]
    monkeypatch.setenv("ORA_EXECUTION_HOST", "test-arm")
    assert declared_execution_host() == "test-arm"
    assert declared_execution_host("test-arm") == "test-arm"
    with pytest.raises(ExecutionResourceError, match="does not match"):
        declared_execution_host("test-host")
