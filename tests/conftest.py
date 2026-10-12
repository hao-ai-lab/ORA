"""Keep every pytest process outside the shared Runtime evidence workspace."""

from __future__ import annotations

import os
import hashlib
import json
import pytest
import shutil
import tempfile
from pathlib import Path


_WORKSPACE_ENV = "ORA_WORKSPACE_ROOT"
_TEST_WORKSPACE = Path(tempfile.mkdtemp(prefix="ora-pytest-workspace-"))

# This runs before test modules import ora.orchestrator.paths.  That
# module resolves workspace constants at import time, so setting the variable
# in a fixture would be too late and would still let collection-time imports
# point at the operator's shared evidence workspace.
os.environ[_WORKSPACE_ENV] = str(_TEST_WORKSPACE)
# Execution host profiles are user-provided; tests bind the fixture hosts
# (test-host, test-cluster, test-arm) before ora.orchestrator.resources
# resolves HOSTS_ROOT at import time.
os.environ["ORA_HOSTS_DIR"] = str(Path(__file__).resolve().parent / "data" / "hosts")


def pytest_sessionfinish(session, exitstatus: int) -> None:
    """Remove only the exact temporary workspace allocated above."""

    del session, exitstatus
    shutil.rmtree(_TEST_WORKSPACE, ignore_errors=True)


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "host_storage_real: run orchestrator.host_storage against the test's own "
        "fake probes instead of the hermetic stub",
    )


@pytest.fixture(autouse=True)
def _hermetic_host_storage(request, monkeypatch: pytest.MonkeyPatch, tmp_path_factory):
    """Keep the orchestrator's host-storage calls off real disks and /dev/shm.

    This stubs the ``host_storage`` entry points preflight and launch call:
    the checks, launch staging, owner binding, discard and release. It does
    not reach code that runs in a separate worker process. Tests marked
    ``host_storage_real`` replace the probes underneath with their own fakes.
    """

    if request.node.get_closest_marker("host_storage_real"):
        return
    from ora.orchestrator import host_storage, run_scratch

    names = ("evidence_workspace_capacity", "runtime_tmpdir",
             "system_partition_capacity", "shm_carb_segments")

    def checks(plan, *, check):
        return [check(name, True, "hermetic pytest stub") for name in names]

    def stage(plan, run_id):
        directory = tmp_path_factory.mktemp("runtime-scratch")
        return {
            "runtime_scratch": {
                "schema": run_scratch.RUNTIME_SCRATCH_SCHEMA, "tmp_root": str(directory),
                "base": str(directory.parent), "run_key": run_scratch.run_key(run_id),
                "source": "hermetic pytest stub", "host_id": None,
                "environment_variables": list(run_scratch.SCRATCH_VARIABLES),
            },
            "shm_sweep": {"root": "hermetic pytest stub", "present": False, "removed": 0,
                          "removed_names": [], "errors": []},
        }

    monkeypatch.setattr(host_storage, "storage_checks", checks)
    monkeypatch.setattr(host_storage, "stage_launch", stage)
    monkeypatch.setattr(host_storage, "bind_scratch_owner", lambda staged, pid: True)
    monkeypatch.setattr(host_storage, "discard_staged", lambda staged: False)
    monkeypatch.setattr(host_storage, "release_run_scratch", lambda run_root: False)


@pytest.fixture
def robodojo_activation():
    """Build explicitly fake compiled-model evidence for host admission tests."""
    from ora.orchestrator.robodojo_embodiment import CARD_ROOT, create_activation
    from ora.environment.benchmarks.robodojo import REVISIONS
    from ora.environment.controllers.robodojo_arx import controller_binding

    device = {
        "schema": "ora.robodojo-device-binding.v1", "renderer_vulkan_index": 2,
        "renderer_cuda_index": 0, "renderer_uuid_prefix": "123456789abcdef0",
        "renderer_pci_bus_hex": "31", "selected_physical_index": 2,
        "selected_gpu_uuid": "GPU-12345678-9abc-def0-1234-56789abcdef0",
        "physics_cuda_logical_index": 0, "physics_cuda_uuid": "12345678-9abc-def0-1234-56789abcdef0",
        "selected_pci_bus_id": "00000000:31:00.0",
        "renderer_identity_resolution": "unique_nvml_uuid_prefix_and_pci_bus",
    }

    def build(root: Path):
        def write(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, sort_keys=True) + "\n")
            return hashlib.sha256(path.read_bytes()).hexdigest()

        descriptor = json.loads((CARD_ROOT / "descriptor-v1.json").read_text())
        external = next(item for item in descriptor["provenance"]["sources"]
                        if item["source_id"] == "robodojo_x5_robot_config")
        compiled = {
            "collector": "initialized_isaac_articulation_and_optical_camera_prim",
            "position_frame": "episode_local_world", "quaternion_order": "xyzw",
            "arms": {side: {
                "fixed_base": True, "arm_joint_names": [f"joint{index}" for index in range(1, 7)],
                "gripper_joint_names": ["joint7", "joint8"],
                "compiled_joint_names": [f"joint{index}" for index in range(1, 9)],
                "controlled_link": "link6", "compiled_body_names": ["base_link", "link6", "camera"],
            } for side in ("left", "right")},
            "cameras": {name: {"source_name": f"cam_{name}", "mount_link": mount}
                        for name, mount in (("head", None), ("left_wrist", "robot0/camera"),
                                            ("right_wrist", "robot1/camera"))},
        }
        episodes = []
        for index in range(2):
            episode_root = root / f"reset-{index}"
            log_path = episode_root / "environment/kit-startup-readback.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("Explicit fake GPU startup evidence for host admission tests.\n")
            actual_device = {**device, "startup_log_prefix": {
                "path": log_path.name, "sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
            }}
            digest = write(episode_root / "environment/source-manifest.json", {
                "loaded_files": {"Assets/Robots/x5/robot_config.yml": external["sha256"]},
                "source_revisions": REVISIONS,
                "device_binding": actual_device,
            })
            reset = {"source_manifest": {"path": "source-manifest.json", "sha256": digest},
                     "control_binding": controller_binding(), "runtime_embodiment": compiled,
                     "device_binding": actual_device}
            write(episode_root / "environment/reset.json", reset)
            sensors = {name: {"finite_positive_fraction": 1.0, "shape": [8, 10]}
                       for name in compiled["cameras"]}
            report = {"passed": True, "reset": reset, "initial_sensors": sensors, "repeated_sensors": sensors,
                      "worker_device_residency": {"worker_pid": 1000 + index, "gpu_uuids": [device["selected_gpu_uuid"]],
                                                   "captured_unix_s": 1_700_000_000.0,
                                                   "source": "nvidia-smi query-compute-apps"}}
            write(episode_root / "probe-report.json", report)
            episodes.append(report)
        return create_activation(root, episodes)

    build.device_binding = device
    return build


@pytest.fixture
def ready_model_services(monkeypatch: pytest.MonkeyPatch):
    """Stand in for the resident SAM3 and Contact-GraspNet services.

    Building a ToolRuntime whose surface enables segmentation, grounding or
    grasp proposals asks each resident service for readiness over HTTP. A test
    about tool definitions or observation projection must not pass or fail on
    whether those services happen to be running on the host. runtime.core binds
    the client classes at import, so the patch targets runtime.core, not grasp.
    """
    from ora.agent.tools.runtime import core as runtime_core

    class Ready:
        def __init__(self, **kwargs):
            self.expected_binding = kwargs.get("expected_binding")

        def require_ready(self):
            return {
                "instance_id": "fixture", "pid": 0, "adapter_sha256": "0" * 64,
                "model": "fixture", "warmup_complete": True,
            }

        def close(self):
            pass

    monkeypatch.setattr(runtime_core, "Sam3ApiSegmenter", Ready)
    monkeypatch.setattr(runtime_core, "ContactGraspNetApiProposer", Ready)
