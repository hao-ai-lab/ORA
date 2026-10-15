"""RoboDojo installer and infrastructure qualification without SDK or provider calls."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess

import pytest

from ora.environment.benchmarks import robodojo
from ora.orchestrator import robodojo_runtime_config as config


@pytest.mark.parametrize("foreign_base", [False, True])
def test_robodojo_installer_validates_python_base_and_editable_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, foreign_base: bool,
) -> None:
    from ora.environment.backends.robodojo.installation.setup_runtime import Installer, PYTHON_VERSION

    installer = Installer(tmp_path / "runtime")
    base = installer.root / f"python/cpython-{PYTHON_VERSION}-linux-x86_64-gnu/bin/python3.11"
    base.parent.mkdir(parents=True)
    base.write_bytes(b"fixture independent Python binary")
    marker = "__editable__.fixture.__path_hook__"
    result = {
        "version": PYTHON_VERSION, "executable": str(installer.python),
        "base_executable": str(base), "prefix": str(installer.root / "env"),
        "base_prefix": str(tmp_path / "external") if foreign_base else str(base.parent.parent),
        "stdlib": str(base.parent.parent / "lib/python3.11"),
        "purelib": str(installer.root / "env/lib/python3.11/site-packages"),
        "ssl_module_origin": "built-in", "sys_path": [marker],
        "editable_path_hooks": {marker: {
            "finder_file": str(installer.root / "env/finder.py"),
            "mapping": {"fixture": str(installer.source / "fixture")}, "namespaces": {},
        }},
    }
    calls = []

    def execute(*args, **kwargs):
        calls.append(args)
        return json.dumps(result) if kwargs.get("capture") else None

    monkeypatch.setattr(installer, "run", execute)
    if foreign_base:
        with pytest.raises(RuntimeError, match="escapes the independent runtime root"):
            installer.python_runtime()
        assert not (installer.root / "python-runtime.json").exists()
    else:
        installer.python_runtime()
        assert json.loads((installer.root / "python-runtime.json").read_text())["all_python_paths_within_runtime_root"]
    assert calls[0] == ("uv", "venv", "--python", base, "--seed", installer.root / "env")


def test_robodojo_installer_initializes_cone_mode_before_asset_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ora.environment.backends.robodojo.installation.setup_runtime import ASSET_REV, Installer

    installer = Installer(tmp_path / "runtime")
    (installer.assets / ".git").mkdir(parents=True)
    calls = []
    monkeypatch.setattr(installer, "run", lambda *args, **_kwargs: calls.append(args) or "")
    installer.clone("https://example.invalid/assets", installer.assets, ASSET_REV)
    cone = calls.index(("git", "sparse-checkout", "init", "--cone"))
    selected = calls.index(("git", "sparse-checkout", "set", "Assets"))
    checkout = calls.index(("git", "checkout", "--detach", ASSET_REV))
    assert cone < selected < checkout


@pytest.mark.parametrize("noncanonical_asset", [False, True])
def test_robodojo_installer_validates_exact_asset_pointer_blobs_only(noncanonical_asset: bool) -> None:
    from ora.environment.backends.robodojo.installation.setup_runtime import validate_asset_pointer_blobs

    oid = hashlib.sha256(b"payload").hexdigest()
    files = [{"name": name, "oid_type": "sha256", "oid": oid, "size": 7}
             for name in ("Assets/first", "Assets/second")]
    pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{oid}\nsize 7\n".encode()
    head_blobs = {"asset-head-blob": pointer + (b"\n" if noncanonical_asset else b""),
                  "excluded-checkpoint-blob": b"noncanonical excluded checkpoint pointer"}
    tracked = {item["name"]: "asset-head-blob" for item in files}
    if noncanonical_asset:
        with pytest.raises(RuntimeError, match="Noncanonical HEAD Asset pointer"):
            validate_asset_pointer_blobs(files, tracked, head_blobs)
    else:
        validate_asset_pointer_blobs(files, tracked, head_blobs)


def test_robodojo_static_doctor_accepts_relocated_tmp_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ora.environment.backends.robodojo.installation.setup_runtime import Installer

    installer = Installer(tmp_path / "Data1 runtime")
    original = installer.source / "scripts/internal/verify_install.sh"
    original.parent.mkdir(parents=True)
    raw = ('ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"\n'
           'echo inventory >/tmp/robodojo_tasks_verify.json\n'
           "with open('/tmp/robodojo_tasks_verify.json', encoding='utf-8') as f:\n"
           'rm -f /tmp/robodojo_tasks_verify.json\n')
    original.write_text(raw)
    calls = []
    monkeypatch.setattr(installer, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    installer.doctor()
    adapted = (installer.root / "tmp/verify_install.isolated.sh").read_text()
    assert original.read_text() == raw
    assert str(installer.root / "tmp/robodojo_tasks_verify.json") in adapted
    assert calls[0][0][2:5] == ("--skip-conda", "--skip-isaac", "--skip-policy")
    assert json.loads((installer.root / "doctor-provenance.json").read_text())["upstream_source_modified"] is False


def _qualification(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, activation,
                   task: str = "arrange_largest_number") -> Path:
    assets = tmp_path / "Assets"
    layout = assets / f"Eval_Layout/RoboDojo/arx_x5/0/{task}_0.json"
    layout.parent.mkdir(parents=True)
    layout.write_text('{"private_fixture_layout": true}')
    monkeypatch.setenv("ROBODOJO_ASSET_ROOT", str(assets))
    monkeypatch.setattr(config, "OUTPUT_ROOT", tmp_path / "outputs/robodojo-compat")
    monkeypatch.setenv("ORA_EXECUTION_HOST", "test-host")
    if task == "arrange_largest_number":
        # Composition fixture uses the registry's reviewed identity. This does not
        # attest the fake layout bytes; dependency verification has separate tests.
        original_runtime = config.runtime_configuration
        review = json.loads((config.PROJECT_ROOT / (
            "ora/policy/environments/robodojo/task-contracts/reviews/"
            + task + "-review-v2.json")).read_text())
        def fixture_runtime(gpu, selected_task):
            value = original_runtime(gpu, selected_task)
            value["layout_sha256"] = review["layout"]["layout_sha256"]
            return value
        monkeypatch.setattr(config, "runtime_configuration", fixture_runtime)
    runtime = config.runtime_configuration("2", task)
    qualification = {
        "schema": robodojo.QUALIFICATION_SCHEMA, "passed": True,
        "harness_commit": subprocess.run(
            ["git", "-C", str(config.PROJECT_ROOT), "rev-parse", "HEAD"],
            text=True, capture_output=True, check=True,
        ).stdout.strip(),
        "source_revisions": robodojo.source_revisions(runtime),
        "task": task, "eval_seed": 0, "layout_id": 0,
        "layout_file": runtime["layout_file"], "layout_sha256": runtime["layout_sha256"],
        "cuda_visible_devices": "2", "sensor_profile": runtime["sensor_profile"],
        "configuration": robodojo.configuration_binding(runtime),
        "checks": {name: True for name in robodojo.QUALIFICATION_CHECKS},
        "task_instruction": "Arrange the numbers from left to right to form the largest possible number, and place them on the pad.",
        "actual_source_fingerprint": {
            "loaded_files": {"env_cfg/arx_x5.yml": "a" * 64},
            "installed_packages": robodojo.WORKER_PACKAGE_VERSIONS,
            "resolved_environment_config_sha256": "b" * 64,
        },
        "environment": {
            "python_executable": "/fixture/bin/python", "python_binary_sha256": "c" * 64,
            "versions": {**robodojo.WORKER_PACKAGE_VERSIONS, "warp-lang": "1.9.1"},
        },
    }
    path = tmp_path / "outputs/robodojo-compat/qualification/fixture.json"
    qualification["embodiment_activation"] = activation(path.parent)
    qualification["device_binding"] = dict(activation.device_binding)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(qualification))
    return path


@pytest.mark.parametrize("driver", ["595.71.05", "595.84", "malformed"])
def test_robodojo_qualification_rejects_bad_driver_before_lease_or_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, driver: str,
) -> None:
    from ora.orchestrator import robodojo_card_qualification as module

    monkeypatch.setattr(module, "OUTPUT_ROOT", tmp_path / "probes")
    monkeypatch.setattr(module, "runtime_configuration", lambda gpu, task: {})
    monkeypatch.setattr(module, "_source_commit", lambda: "a" * 40)
    monkeypatch.setattr(
        module, "worker_kernel_sha256", lambda commit, **_kwargs: "d" * 64,
    )
    monkeypatch.setattr(module, "_python_stack", lambda: {})
    monkeypatch.setenv("ROBODOJO_PYTHON", str(tmp_path / "env/bin/python"))
    monkeypatch.setattr(module, "inspect_graphics_compatibility", lambda *_args: None)
    monkeypatch.setenv("ORA_EXECUTION_HOST", "test-host")
    monkeypatch.setattr(module.subprocess, "check_output", lambda *_args, **_kwargs: f"0, 40000, {driver}\n")

    def forbidden(*_args, **_kwargs):
        pytest.fail("known incompatible driver crossed the lease/worker boundary")

    monkeypatch.setattr(module, "claim_resource_lease", forbidden)
    monkeypatch.setattr(module, "_episode", forbidden)
    with pytest.raises(ValueError, match="driver"):
        module.qualify("0")
    assert not module.OUTPUT_ROOT.exists()


def test_robodojo_qualification_releases_its_lease_when_final_evidence_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, robodojo_activation,
) -> None:
    from ora.orchestrator import robodojo_card_qualification as module

    reference = _qualification(tmp_path, monkeypatch, robodojo_activation)
    binding = json.loads(reference.read_text())
    monkeypatch.setattr(module, "OUTPUT_ROOT", tmp_path / "probes")
    monkeypatch.setattr(module, "_source_commit", lambda: binding["harness_commit"])
    monkeypatch.setattr(
        module, "worker_kernel_sha256", lambda commit, **_kwargs: "d" * 64,
    )
    monkeypatch.setattr(module, "_python_stack", lambda: binding["environment"])
    monkeypatch.setenv("ORA_EXECUTION_HOST", "test-host")
    monkeypatch.setattr(module.subprocess, "check_output", lambda *_args, **_kwargs: "2, 40000, 580.65.06\n")
    monkeypatch.setattr(module, "claim_resource_lease", lambda *_args: {"claim_token": "fixture-token"})
    monkeypatch.setattr(module, "assign_resource_lease", lambda *_args: None)
    released = []
    monkeypatch.setattr(module, "release_resource_lease", lambda *args, **kwargs: released.append((args, kwargs)))
    reset = {
        "task_instruction": binding["task_instruction"],
        "actual_source_fingerprint": binding["actual_source_fingerprint"],
        "source_binding": {"source_revisions": binding["source_revisions"]},
        "control_binding": {"control_hz": 25},
        "runtime_embodiment": {"collector": "fixture"},
    }
    monkeypatch.setattr(module, "_episode", lambda *_args, **_kwargs: {
        "passed": True, "reset": reset, "native_timeout_exercised": True,
        "checks": {name: True for name in robodojo.QUALIFICATION_CHECKS},
        "scene_registration": {"instances": {"mug_0": "rigid"}},
    })
    original_write = module._write

    def write(path, value):
        if path.name == "qualification.json":
            raise OSError("fixture disk failure")
        original_write(path, value)

    monkeypatch.setattr(module, "_write", write)
    with pytest.raises(OSError, match="disk failure"):
        module.qualify("2")
    assert len(released) == 1
    assert released[0][1]["claim_token"] == "fixture-token"


def _stub_probe_environment(
    module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *,
    device: str = "2",
) -> list[dict]:
    """Reach the probe body without an interpreter, a GPU or a worker."""

    from ora.orchestrator import robodojo_embodiment as embodiment

    monkeypatch.setattr(module, "OUTPUT_ROOT", tmp_path / "probes")
    monkeypatch.setattr(module, "_source_commit", lambda: "a" * 40)
    # The record is admitted by worker-kernel content; the fixture commit has
    # no git history behind it, so the digest is supplied here.
    monkeypatch.setattr(
        module, "worker_kernel_sha256", lambda commit, **_kwargs: "d" * 64,
    )
    monkeypatch.setattr(module, "_python_stack", lambda: {"stub": True})
    monkeypatch.setenv("ORA_EXECUTION_HOST", "test-host")
    monkeypatch.setattr(
        module.subprocess, "check_output",
        lambda *_args, **_kwargs: f"{device}, 40000, 580.65.06\n",
    )
    monkeypatch.setattr(
        module, "claim_resource_lease", lambda *_args: {"claim_token": "token"},
    )
    monkeypatch.setattr(module, "assign_resource_lease", lambda *_args: None)
    monkeypatch.setattr(
        module, "release_resource_lease", lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        embodiment, "create_activation", lambda root, episodes: {"stub": True},
    )
    seen: list[dict] = []

    def episode(runtime, root, *, scope="card"):
        seen.append(dict(runtime))
        names = (robodojo.CARD_QUALIFICATION_CHECKS if scope == "card"
                 else robodojo.TASK_QUALIFICATION_CHECKS)
        registration = {
            "schema": robodojo.SCENE_REGISTRATION_SCHEMA,
            "instances": {"mug_0": "rigid"},
            "instance_identities_sha256": robodojo.scene_identities_sha256(
                {"mug_0": "rigid"},
            ),
            "registered_instance_count": 1,
            "evaluator_registration": {
                "check_list": True, "final_check_list": False,
                "trigger_check_list": False,
            },
            "native_action_limit": int(runtime["native_action_limit"]),
        }
        return {
            "passed": True, "native_timeout_exercised": True, "scope": scope,
            "checks": {
                **{name: True for name in names},
                **{name: True for name in robodojo.QUALIFICATION_CHECKS},
            },
            "scene_registration": registration,
            "repeated_reset_scene_registration": registration,
            "reset": {
                "task_instruction": "Hang all the mugs on the mug rack.",
                "actual_source_fingerprint": {"loaded_files": {}},
                "source_binding": {
                    "source_revisions": robodojo.source_revisions(runtime),
                },
                "control_binding": {"control_hz": 25},
                "runtime_embodiment": {
                    "collector": (
                        "initialized_isaac_articulation_and_optical_camera_prim"
                    ),
                    "arms": {"left": {}, "right": {}},
                    "cameras": {
                        "head": {}, "left_wrist": {}, "right_wrist": {},
                    },
                },
                "device_binding": {"selected_gpu_uuid": "GPU-fixture"},
            },
        }

    monkeypatch.setattr(module, "_episode", episode)
    return seen


def test_robodojo_qualification_default_still_uses_this_package_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The compat surface is unchanged by the runtime parameter."""

    from ora.orchestrator import robodojo_card_qualification as module

    assets = tmp_path / "Assets"
    layout = assets / "Eval_Layout/RoboDojo/arx_x5/0/hang_mugs_0.json"
    layout.parent.mkdir(parents=True)
    layout.write_text('{"private_fixture_layout": true}')
    monkeypatch.setenv("ROBODOJO_ASSET_ROOT", str(assets))
    seen = _stub_probe_environment(module, tmp_path, monkeypatch)

    destination = module.qualify("2", "hang_mugs")
    record = json.loads(destination.read_text())
    expected = config.runtime_configuration("2", "hang_mugs")
    assert record["configuration"] == robodojo.configuration_binding(expected)
    assert record["configuration"]["workspace_bounds_m"] == {
        "x": [-1.5, 1.5], "y": [-1.5, 1.5], "z": [-0.2, 2.0],
    }
    assert record["cuda_visible_devices"] == "2"
    assert seen and seen[0]["workspace_bounds_m"] == (
        expected["workspace_bounds_m"]
    )
    # The record stores the RoboDojo worker-kernel content that admission
    # compares, and keeps the measurement commit only as lineage.
    assert record["worker_kernel_sha256"] == "d" * 64
    assert record["harness_commit"] == "a" * 40
    plan = json.loads((destination.parent / "probe-plan.json").read_text())
    assert plan["worker_kernel_sha256"] == "d" * 64
    # The evidence directory keeps this package's own owner prefix.
    assert destination.parent.name.startswith("robodojo-infra-")


def test_robodojo_qualification_records_a_caller_supplied_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A foreign Runtime -- another device and another envelope -- is bound.

    ``load_infrastructure_qualification`` compares the recorded
    ``configuration`` field for field, and ``CONFIGURATION_FIELDS`` contains
    ``workspace_bounds_m``, so the record has to be produced from the caller's
    own Runtime rather than from this package's configuration.
    """

    from ora.orchestrator import robodojo_card_qualification as module

    seen = _stub_probe_environment(module, tmp_path, monkeypatch, device="3")
    layouts = tmp_path / "Assets/Eval_Layout/RoboDojo/arx_x5/0"
    layouts.mkdir(parents=True)
    (layouts / "hang_mugs_0.json").write_text('{"private_fixture_layout": true}')
    monkeypatch.setenv("ROBODOJO_ASSET_ROOT", str(tmp_path / "Assets"))
    runtime = config.runtime_configuration("3", "hang_mugs")
    runtime["workspace_bounds_m"] = {
        "x": [-0.75, 0.75], "y": [-0.65, 0.55], "z": [0.72, 1.40],
    }
    runtime["resource_binding"] = {
        "profile_id": "robodojo-gpu3", "slot_id": "gpu3",
        "worker_environment": {"CUDA_VISIBLE_DEVICES": "3"},
    }

    destination = module.qualify(
        "3", "hang_mugs", runtime=runtime, owner_prefix="exp08a-robodojo-infra",
    )
    record = json.loads(destination.read_text())
    assert record["configuration"] == robodojo.configuration_binding(runtime)
    assert record["configuration"]["workspace_bounds_m"] == (
        runtime["workspace_bounds_m"]
    )
    assert record["cuda_visible_devices"] == "3"
    assert seen and seen[0]["resource_binding"] == runtime["resource_binding"]
    assert destination.parent.name.startswith("exp08a-robodojo-infra-")


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda runtime: runtime.update({"cuda_visible_devices": "5"}),
         "does not select the requested GPU"),
        (lambda runtime: runtime.update({"renderer_gpu": 5}),
         "does not select the requested GPU"),
        (lambda runtime: runtime.update({"task_variant": "hang_mugs"}),
         "does not select the requested task"),
        (lambda runtime: runtime.update({
            "infrastructure_qualification": {"path": "p", "sha256": "e" * 64},
        }), "may not carry a qualification binding"),
    ],
)
def test_robodojo_qualification_refuses_an_inconsistent_supplied_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutate, message: str,
) -> None:
    from ora.orchestrator import robodojo_card_qualification as module

    monkeypatch.setattr(module, "OUTPUT_ROOT", tmp_path / "probes")
    monkeypatch.setattr(
        module, "_source_commit",
        lambda: pytest.fail("an invalid Runtime must never reach the probes"),
    )
    runtime = {
        "cuda_visible_devices": "3", "renderer_gpu": 3,
        "task_variant": "pack_objects_into_box",
    }
    mutate(runtime)
    with pytest.raises(RuntimeError, match=message):
        module.qualify("3", "pack_objects_into_box", runtime=runtime)
    assert not (tmp_path / "probes").exists()


@pytest.mark.parametrize(
    "envelope,expected",
    [
        ("compat", [1.49, 1.49, 1.99]),
        ("tabletop", [0.74, 0.54, 1.39]),
    ],
)
def test_the_unreachable_probe_target_follows_the_runtime_it_is_given(
    envelope: str, expected: list[float],
) -> None:
    """The planner-refusal probe must survive whichever envelope it runs under.

    ``RobodojoSession.execute`` raises ``pose target exceeds the qualified
    workspace`` before it ever calls ``solve_ik``, so a target outside
    ``workspace_bounds_m`` proves nothing about the planner and aborts the
    infrastructure episode instead.  The compatibility box and the
    source-owned ``robodojo_arx_x5_tabletop_v1`` envelope EXP08A resolves are
    2.2 m apart on z, so pin both halves under both: inside the envelope with
    the full derivation margin, and past the X5's reach from either mount.
    """

    from ora.orchestrator.robodojo_card_qualification import (
        UNREACHABLE_PROBE_MARGIN_M, unreachable_pose_probe_target,
    )
    from ora.environment.backends.robodojo.common import workspace_violation
    from ora.environment.backends.robodojo.motion_qualification import (
        X5_BASE_MOUNTS_M, X5_LINK6_REACH_BOUND_M,
    )
    from ora.environment.workspace_profiles import (
        ROBODOJO_ARX_X5_TABLETOP_V1, resolve_workspace_profile,
    )

    bounds = (
        {"x": [-1.5, 1.5], "y": [-1.5, 1.5], "z": [-0.2, 2.0]}
        if envelope == "compat"
        else resolve_workspace_profile(ROBODOJO_ARX_X5_TABLETOP_V1)["bounds_m"]
    )
    target = unreachable_pose_probe_target(bounds)
    assert target == expected
    # The hard action envelope must admit it, with the full derived margin.
    assert workspace_violation(target, bounds) is None
    for index, axis in enumerate(("x", "y", "z")):
        low, high = bounds[axis]
        assert min(target[index] - low, high - target[index]) >= (
            UNREACHABLE_PROBE_MARGIN_M - 1e-9
        )
    # And the SDK IK must have no answer: link6 cannot stand farther from its
    # own fixed base than the URDF chain it hangs on, whichever arm is asked.
    for mount in X5_BASE_MOUNTS_M:
        assert math.dist(target, mount) > 1.5 * X5_LINK6_REACH_BOUND_M


def test_an_envelope_with_no_unreachable_corner_is_refused_before_the_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Runtime that cannot host a provable IK refusal has no valid probe.

    The derivation is required, not merely attempted: an envelope small enough
    that every corner sits inside the X5's reach would silently turn the
    planner-refusal probe into a reachable move.  That must fail closed, and
    before an Isaac worker is paid for.
    """

    from ora.orchestrator import robodojo_card_qualification as module
    from ora.orchestrator import robodojo_probe_episode

    # The probe episode is where a worker would boot.
    monkeypatch.setattr(
        robodojo_probe_episode, "build_environment_session",
        lambda **_kwargs: pytest.fail("a Runtime without a valid probe booted a worker"),
    )
    runtime = {
        "native_action_limit": 100, "task_variant": "hang_mugs",
        # Entirely within the left mount's reach sphere.
        "workspace_bounds_m": {
            "x": [-0.4, -0.2], "y": [-0.5, -0.4], "z": [0.80, 0.90],
        },
    }
    with pytest.raises(RuntimeError, match="not provable"):
        module._episode(runtime, tmp_path / "reset-0")


@pytest.mark.parametrize("failure", [None, "episode", "second_claim"])
def test_card_parallel_workers_reserve_distinct_slots_and_join_before_release(
    tmp_path, monkeypatch, failure,
):
    from threading import Barrier
    from ora.orchestrator import robodojo_card_qualification as module

    _stub_probe_environment(module, tmp_path, monkeypatch, device="6")
    layout = tmp_path / "Assets/Eval_Layout/RoboDojo/arx_x5/0/hang_mugs_0.json"
    layout.parent.mkdir(parents=True)
    layout.write_text('{}')
    monkeypatch.setenv("ROBODOJO_ASSET_ROOT", str(tmp_path / "Assets"))
    runtime = config.runtime_configuration("6", "hang_mugs")
    original_episode = module._episode
    barrier = Barrier(2, timeout=5)
    claimed, assigned, released, finished = [], [], [], []

    def claim(plan, owner, pid):
        assert not finished
        if failure == "second_claim" and claimed:
            raise RuntimeError("fixture second slot unavailable")
        binding = plan["runtime"]["resource_binding"]
        claimed.append((owner, binding["slot_id"]))
        return {"claim_token": owner}

    def episode(selected, root, *, scope):
        assert len(claimed) == len(assigned) == 2
        assert not released
        assert scope == "card"
        assert selected["native_action_limit"] == runtime["native_action_limit"]
        assert selected["resource_binding"]["worker_environment"] == {"CUDA_VISIBLE_DEVICES": "6"}
        # Neither episode can complete until both actually enter the probe.
        barrier.wait()
        try:
            if failure == "episode" and root.name == "reset-0":
                raise RuntimeError("fixture episode failed")
            return original_episode(selected, root, scope=scope)
        finally:
            finished.append(root.name)

    def release(owner, *, owner_pid, claim_token):
        assert claim_token == owner
        assert len(finished) == (0 if failure == "second_claim" else 2)
        released.append(owner)

    monkeypatch.setattr(module, "claim_resource_lease", claim)
    monkeypatch.setattr(module, "assign_resource_lease", lambda lease, pid: assigned.append(lease))
    monkeypatch.setattr(module, "release_resource_lease", release)
    monkeypatch.setattr(module, "_episode", episode)

    def run_probe():
        return module.qualify_card(
            "6", "hang_mugs", runtime=runtime,
            parallel_resource_profiles=("robodojo-gpu6", "robodojo-gpu6b"),
        )

    if failure:
        with pytest.raises(RuntimeError, match="fixture"):
            run_probe()
    else:
        destination = run_probe()
        record = json.loads(destination.read_text())
        assert record["passed"] and all(record["checks"].values())
        assert record["measured_task"]["native_timeout_exercised"] is True
        assert record["card_configuration"] == robodojo.card_configuration_binding(runtime)
        for index, slot in enumerate(("gpu6", "gpu6b")):
            plan = json.loads((destination.parent / f"reset-{index}/probe-plan.json").read_text())
            assert plan["runtime"]["resource_binding"]["slot_id"] == slot
            assert plan["episodes"] == 1 and plan["episode_index"] == index
    assert {owner for owner, _ in claimed} == set(released)
    assert [slot for _, slot in claimed] == (["gpu6"] if failure == "second_claim" else ["gpu6", "gpu6b"])
    assert "resource_binding" not in runtime


@pytest.mark.parametrize("profiles, message", [
    (("robodojo-gpu6", "robodojo-gpu6"), "distinct resource slots"),
    (("robodojo-gpu6", "robodojo-gpu7"), "same CUDA and renderer GPU"),
    (("robodojo-gpu6",), "exactly two resource profiles"),
])
def test_parallel_card_rejects_duplicate_or_cross_device_slots(profiles, message):
    from ora.orchestrator import robodojo_card_qualification as module

    runtime = {**dict.fromkeys(robodojo.CARD_CONFIGURATION_FIELDS),
               "kind": "robodojo", "required_host": "test-host",
               "cuda_visible_devices": "6", "renderer_gpu": 6}
    # The resolver must reject the invalid scheduling choice before leases or
    # a simulator are touched; configuration comparison does not need assets.
    with pytest.raises(RuntimeError, match=message):
        module._card_episode_runtimes(runtime, profiles)


def test_qualification_entry_point_requires_confirmation_before_any_episode(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from ora.orchestrator import robodojo_card_qualification as module

    calls = []
    monkeypatch.setattr(module, "qualify_card", lambda *args, **kwargs: calls.append((args, kwargs)) or "card")
    with pytest.raises(SystemExit):
        module.main(["qualify-card", "--gpu", "2"])
    assert calls == []
    assert module.main(["qualify-card", "--gpu", "2", "--task", "hang_mugs", "--yes"]) == 0
    assert calls == [(("2", "hang_mugs"), {})]
    assert capsys.readouterr().out.strip() == "card"


def test_probe_host_copies_match_the_worker_source() -> None:
    """The host may not import the worker backend, so its copies are checked here."""
    from ora.environment.backends.robodojo import common, motion_checks
    from ora.orchestrator import robodojo_probe_episode as probe

    assert probe.X5_LINK6_REACH_BOUND_M == motion_checks.X5_LINK6_REACH_BOUND_M
    assert probe.X5_BASE_MOUNTS_M == motion_checks.X5_BASE_MOUNTS_M
    bounds = {"x": [-0.75, 0.75], "y": [-0.65, 0.55], "z": [0.72, 1.40]}
    for position in ([0, 0, 1.0], [0.8, 0, 1.0], [0, -0.7, 1.0], [0, 0, 1.5],
                     [0.75, 0.55, 1.40], [-0.76, 0.6, 0.5]):
        assert probe.workspace_violation(position, bounds) == common.workspace_violation(position, bounds)
    with pytest.raises(RuntimeError, match="workspace_bounds_m is required"):
        probe.workspace_violation([0, 0, 0], None)


@pytest.mark.parametrize("binding,admitted", [
    ({"minimum_free_memory_mib": 10240}, True),
    (None, False),
])
def test_probe_admission_uses_the_slot_memory_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, binding, admitted: bool,
) -> None:
    from ora.orchestrator import robodojo_card_qualification as module

    runtime = {"cuda_visible_devices": "2", "renderer_gpu": 2, "task_variant": "hang_mugs"}
    if binding is not None:
        runtime["resource_binding"] = binding
    monkeypatch.setattr(module, "OUTPUT_ROOT", tmp_path / "probes")
    monkeypatch.setattr(module, "_source_commit", lambda: "a" * 40)
    monkeypatch.setattr(module, "worker_kernel_sha256", lambda commit, **_kwargs: "d" * 64)
    monkeypatch.setattr(module, "_python_stack", lambda: {})
    monkeypatch.setenv("ORA_EXECUTION_HOST", "test-host")
    monkeypatch.setattr(module.subprocess, "check_output", lambda *_args, **_kwargs: "2, 15000, 580.65.06\n")

    def past_memory_gate(*_args, **_kwargs):
        raise LookupError("reached the driver check")

    monkeypatch.setattr(module, "validate_runtime_driver", past_memory_gate)
    expected = (LookupError, "reached the driver check") if admitted else (
        RuntimeError, "insufficient free memory")
    with pytest.raises(expected[0], match=expected[1]):
        module._prepare("2", "hang_mugs", runtime, "robodojo-card")


def test_robodojo_installer_graphics_compat_pins_match_the_worker() -> None:
    from ora.environment.backends.robodojo import graphics_compat
    from ora.environment.backends.robodojo.installation import setup_runtime

    assert setup_runtime.VULKAN_PROFILES_REV == graphics_compat.SOURCE_REVISION
    assert setup_runtime.COMPAT_PATCH_SHA256 == graphics_compat.PATCH_SHA256
    assert setup_runtime.COMPAT_PROFILE_NAME == graphics_compat.PROFILE
    here = Path(setup_runtime.__file__).parent
    profile = here / f"{setup_runtime.COMPAT_PROFILE_NAME}.json"
    assert hashlib.sha256(profile.read_bytes()).hexdigest() == setup_runtime.COMPAT_PROFILE_SHA256


def test_robodojo_installer_graphics_compat_writes_a_receipt_the_worker_accepts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from ora.environment.backends.robodojo import graphics_compat
    from ora.environment.backends.robodojo.installation import setup_runtime

    installer = setup_runtime.Installer(tmp_path / "runtime")
    installer.python.parent.mkdir(parents=True)
    installer.python.touch()
    work = installer.root / "graphics-compat-build"
    (work / "Vulkan-Profiles/.git").mkdir(parents=True)
    calls = []

    def execute(*args, **kwargs):
        calls.append(tuple(str(arg) for arg in args))
        if args[:2] == ("git", "rev-parse"):
            return setup_runtime.VULKAN_PROFILES_REV
        if args[-1] == "--version":
            return f"{args[0]} fixture 1.0"
        if args[:2] == ("cmake", "--build"):
            layer = work / "build/layer"
            layer.mkdir(parents=True)
            (layer / "libVkLayer_khronos_profiles.so").write_bytes(b"fixture layer")
            (layer / "VkLayer_khronos_profiles.json").write_text(json.dumps({
                "file_format_version": "1.2.1",
                "layer": {"name": "VK_LAYER_KHRONOS_profiles", "type": "GLOBAL",
                          "library_path": "./libVkLayer_khronos_profiles.so"}}))
        return "" if kwargs.get("capture") else None

    monkeypatch.setattr(installer, "run", execute)
    monkeypatch.setattr(setup_runtime.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1))
    installer.graphics_compat(jobs=2)
    assert ("git", "apply", str(Path(setup_runtime.__file__).parent / "vulkan-profiles-core-id.patch")) in calls
    assert any(call[:2] == ("cmake", "-S") and "-DUPDATE_DEPS=ON" in call for call in calls)
    receipt, files = graphics_compat.installation(installer.root)
    assert receipt["schema"] == "ora.robodojo-graphics-compat-installation.v1"
    assert set(files) == {"layer", "profile", "manifest", "patch"}
    assert not any(call[:2] == ("git", "clone") for call in calls)

    def rejected(*args, **kwargs):
        if args and str(args[0]) == str(installer.python):
            raise subprocess.CalledProcessError(1, [str(arg) for arg in args])
        return execute(*args, **kwargs)

    # A failed worker readback withdraws the receipt so ora setup does not skip the stage next time.
    import shutil
    import subprocess

    shutil.rmtree(work / "build")
    monkeypatch.setattr(installer, "run", rejected)
    with pytest.raises(subprocess.CalledProcessError):
        installer.graphics_compat(jobs=2)
    assert not (installer.root / "graphics-compat/installation.json").exists()


def test_setup_root_has_no_host_default(monkeypatch, tmp_path):
    from ora.environment.backends.robodojo.installation import setup_runtime

    assert setup_runtime.default_root({}) is None
    assert setup_runtime.default_root({"ROBODOJO_RUNTIME_ROOT": str(tmp_path)}) == tmp_path
    assert setup_runtime.default_root({"ROBODOJO_ROOT": f"{tmp_path}/source/RoboDojo"}) == tmp_path
    assert setup_runtime.default_root({"ROBODOJO_ROOT": "/elsewhere/RoboDojo"}) is None
    monkeypatch.delenv("ROBODOJO_RUNTIME_ROOT", raising=False)
    monkeypatch.delenv("ROBODOJO_ROOT", raising=False)
    with pytest.raises(SystemExit):
        setup_runtime.main(["doctor"])


def test_setup_runtime_retries_git_only_on_a_lingering_index_lock(tmp_path, monkeypatch):
    import subprocess
    from types import SimpleNamespace

    from ora.environment.backends.robodojo.installation import setup_runtime

    installer = setup_runtime.Installer(tmp_path / "runtime")
    lock = "fatal: Unable to create '/x/.git/index.lock': File exists.\n"
    results = [SimpleNamespace(returncode=128, stdout=None, stderr=lock),
               SimpleNamespace(returncode=0, stdout=None, stderr="")]
    calls = []
    monkeypatch.setattr(setup_runtime.subprocess, "run", lambda command, **kwargs: (calls.append(command),
                                                                                    results.pop(0))[1])
    monkeypatch.setattr(setup_runtime.time, "sleep", lambda seconds: None)
    installer.run("git", "sparse-checkout", "set", "Assets", cwd=tmp_path)
    assert len(calls) == 2
    results[:] = [SimpleNamespace(returncode=128, stdout=None, stderr="fatal: not a git repository\n")]
    with pytest.raises(subprocess.CalledProcessError):
        installer.run("git", "status", cwd=tmp_path)
    assert len(calls) == 3


def test_robodojo_manifest_tolerates_a_checkout_another_stage_is_still_cloning(tmp_path: Path) -> None:
    # sources, assets and sdk run in parallel and each ends with manifest(); the
    # 2026-10-15 single-GPU run failed sdk on a submodule whose clone had begun
    # (git init done, no HEAD yet).
    import json
    import subprocess

    from ora.environment.backends.robodojo.installation.setup_runtime import Installer

    installer = Installer(tmp_path / "runtime")
    cloning = installer.source / "XPolicyLab"
    cloning.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(cloning)], check=True)
    (installer.source / ".git").mkdir()
    installer.root.mkdir(parents=True, exist_ok=True)
    installer.manifest()
    checkouts = json.loads((installer.root / "install-manifest.json").read_text())["checkouts"]
    assert checkouts["XPolicyLab"] == {"head": None, "status": None, "incomplete": True}
