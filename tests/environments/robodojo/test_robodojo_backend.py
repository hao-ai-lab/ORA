"""RoboDojo policy boundary and lifecycle checks without an Isaac process."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from ora.environment.backends.robodojo.bridge import EvaluationLifecycle, renderer_table_binding
from ora.environment.backends.robodojo import bridge
from ora.environment.backends.robodojo import graphics_compat
from ora.environment.backends.robodojo import session as session_module
from ora.environment.backends.robodojo.common import (
    CLOSED_COMMAND_FLOOR, gripper_command_target,
)
from ora.environment.backends.robodojo.session import RobodojoError, RobodojoSession


def test_resolved_sdk_control_holds_targets_under_feedback_and_repeated_reads():
    class FeedbackControl:
        def get_action(self, manager, env_idx):
            return {"left_arm": {"position": [0.2]}, "left_ee": {"position": [manager.measured + .01]},
                    "right_ee": {"position": [manager.measured + .01]}}

    manager = SimpleNamespace(measured=0.0)
    previous = {"left_arm": {"position": [0.1]}, "left_ee": {"position": [.044]},
                "right_ee": {"position": [.044]}}
    original = FeedbackControl()
    command, targets = bridge._fixed_meta_control(original, manager, previous, {"left_arm": {}})
    assert isinstance(command, FeedbackControl)
    assert targets == {**previous, "left_arm": {"position": [.2]}}
    manager.measured = -.02
    assert command.get_action(manager, env_idx=0) == targets
    command.get_action(manager, env_idx=0)["left_ee"]["position"][0] = 100
    assert command.get_action(manager, env_idx=0) == targets
    # An explicit gripper action retains the SDK's feedback-limited command.
    changed, targets = bridge._fixed_meta_control(FeedbackControl(), manager, previous, {"left_ee": {}})
    assert targets["left_ee"]["position"] == [-.01]
    assert targets["left_arm"] == previous["left_arm"]
    assert targets["right_ee"] == previous["right_ee"]
    assert FeedbackControl().get_action(manager, env_idx=0)["right_ee"]["position"] == [-.01]


@pytest.mark.parametrize("method", ["get_functional_points", "get_support_points"])
def test_private_nested_evaluator_geometry_reads_use_cpu_adapter(method):
    root = np.array([3.25, -4.5, 2.75, 1, 0, 0, 0], dtype=np.float32)
    obj = SimpleNamespace(get_state=lambda is_relative: {"root_pose": root},
                          env_origin=np.array([3, -5, 2], dtype=np.float32))

    class Layout:
        instance_type_by_env = [{"rack": "geometry"}]

        def get_scene_object(self, env_idx, inst_name):
            return obj

        def get_instance_pose(self, **kwargs):
            raise RuntimeError("unadapted GPU geometry arithmetic")

        def get_functional_points(self, **kwargs):
            return self.get_instance_pose(**kwargs)

        def get_support_points(self, **kwargs):
            return self.get_instance_pose(**kwargs)

    delegate = Layout()
    with pytest.raises(RuntimeError, match="unadapted"):
        getattr(delegate, method)(env_idx=0, inst_name="rack")
    result = getattr(bridge.EvaluatorLayoutInput(delegate), method)(env_idx=0, inst_name="rack")
    np.testing.assert_array_equal(result[0], [.25, .5, .75])
    np.testing.assert_array_equal(root, [3.25, -4.5, 2.75, 1, 0, 0, 0])


@pytest.mark.parametrize("fault", [None, "resolution", "lens", "offset"])
def test_intrinsics_follow_effective_render_projection_not_raw_vertical_aperture(fault):
    from ora.environment.backends.robodojo.camera_capture import renderer_intrinsics

    parameters = {
        "cameraModel": "pinhole", "cameraApertureOffset": [0, 0],
        "cameraAperture": [20, 12], "cameraFocalLength": 9,
        "renderProductResolution": [640, 480],
        "cameraProjection": [.9, 0, 0, 0, 0, 1.2, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0],
    }
    if fault == "resolution":
        parameters["renderProductResolution"] = [480, 640]
    elif fault == "lens":
        parameters["cameraModel"] = "fisheye"
    elif fault == "offset":
        parameters["cameraApertureOffset"] = [.1, 0]
    if fault:
        with pytest.raises(ValueError):
            renderer_intrinsics(parameters, (640, 480))
    else:
        np.testing.assert_allclose(renderer_intrinsics(parameters, (640, 480)),
                                   [[288, 0, 320], [0, 288, 240], [0, 0, 1]])


def test_private_evaluator_pose_copy_preserves_values_and_leaves_physics_manager_untouched():
    from ora.environment.backends.robodojo.bridge import EvaluatorLayoutInput

    class DevicePose:
        def __init__(self, value):
            self.value = value

        def __array__(self):
            raise TypeError("a GPU pose cannot be converted directly")

        def detach(self):
            return self

        def cpu(self):
            return SimpleNamespace(numpy=lambda: self.value)

    position = np.array([.125, -.25, .875], dtype=np.float32)
    rotation = np.array([0, 1, 0, 0], dtype=np.float64)
    poses = (DevicePose(position), DevicePose(rotation))
    calls = []

    def get_pose(*args, **kwargs):
        calls.append((args, kwargs))
        return poses

    delegate = SimpleNamespace(get_instance_pose=get_pose, private_marker=object())
    adapter = EvaluatorLayoutInput(delegate)
    result = adapter.get_instance_pose(env_idx=0, inst_name="fixture", relative=True)
    assert calls == [((), {"env_idx": 0, "label": None, "inst_name": "fixture", "relative": True})]
    for actual, expected in zip(result, (position, rotation)):
        assert actual.dtype == expected.dtype and np.array_equal(actual, expected)
        assert not np.shares_memory(actual, expected)
    assert delegate.get_instance_pose()[0] is poses[0]
    assert adapter.private_marker is delegate.private_marker


def test_private_geometry_pose_uses_same_origin_subtraction_on_cpu():
    from ora.environment.backends.robodojo.bridge import EvaluatorLayoutInput

    root = np.array([3.25, -4.5, 2.75, 1, 0, 0, 0], dtype=np.float32)
    origin = np.array([3, -5, 2], dtype=np.float32)
    calls = []

    def get_state(*, is_relative):
        calls.append(is_relative)
        if is_relative:
            raise RuntimeError("mixed CPU/GPU origin")
        return {"root_pose": root}

    obj = SimpleNamespace(get_state=get_state, env_origin=origin)
    delegate = SimpleNamespace(instance_type_by_env=[{"pad": "geometry"}],
                               get_scene_object=lambda env_idx, inst_name: obj)
    position, rotation = EvaluatorLayoutInput(delegate).get_instance_pose(0, inst_name="pad")
    np.testing.assert_array_equal(position, np.array([.25, .5, .75], dtype=np.float32))
    np.testing.assert_array_equal(rotation, root[3:])
    assert position.dtype == root.dtype and calls == [False]
    assert root[0] == 3.25 and origin[0] == 3


@pytest.mark.parametrize("fault", [None, "timeout", "physics"])
def test_texture_wait_requires_real_readiness_without_physics_advance(tmp_path, monkeypatch, fault):
    from ora.environment.backends.robodojo import camera_capture

    clock = {"frames": 0, "physics": 17, "time": 0.0}
    monkeypatch.setattr(camera_capture.time, "monotonic", lambda: clock["time"])
    manager = SimpleNamespace(num_envs=1, tiled_cameras=[
        SimpleNamespace(_render_product_path=f"/camera/{index}") for index in range(3)])

    def render():
        clock["frames"] += 1
        clock["time"] += 1
        if fault == "physics":
            clock["physics"] += 1

    def run():
        camera_capture.wait_for_render_assets(
            manager=manager, render=render,
            assets_loading=lambda: fault == "timeout" or clock["frames"] < 2,
            stage_status=lambda: {"loading": ["local textures", 2, 3]},
            physics_clock=lambda: clock["physics"], output_root=tmp_path, timeout_s=3)

    if fault is None:
        run()
        assert clock["frames"] == 2 and clock["physics"] == 17
        evidence = [json.loads(line) for line in (tmp_path / "render-readiness.jsonl").read_text().splitlines()]
        assert evidence[0]["assets_loading"] is True
        assert evidence[-1]["assets_loading"] is False
    else:
        with pytest.raises(TimeoutError if fault == "timeout" else ValueError):
            run()
        assert clock["frames"] <= 3


def test_crash_report_guard_precedes_sdk_startup_and_preserves_local_dumps(tmp_path, monkeypatch):
    monkeypatch.setattr(bridge.os, "environ", dict(bridge.os.environ))
    monkeypatch.setattr(sys, "path", list(sys.path))
    for variable in bridge._CRASH_REPORT_ENV:
        monkeypatch.setenv(variable, "unsafe-inherited-value")
    root = tmp_path / "source"
    experience = root / "third_party/IsaacLab/apps/isaaclab.python.kit"
    experience.parent.mkdir(parents=True)
    experience.touch()
    output = tmp_path / "episode"
    monkeypatch.setattr(bridge, "source_manifest", lambda config: {"loaded_files": {}})
    monkeypatch.setattr(graphics_compat, "activate", lambda root: {"files": {}})
    monkeypatch.setitem(sys.modules, "env.global_configs", SimpleNamespace())

    class StopBeforeSDK(RuntimeError):
        pass

    def launcher(**kwargs):
        assert kwargs["headless"] is True and kwargs["enable_cameras"] is True
        assert kwargs["fast_shutdown"] is False
        assert kwargs["device"] == "cuda:0" and kwargs["experience"] == str(experience)
        assert "--/renderer/activeGpu=2" in kwargs["kit_args"]
        arguments = kwargs["kit_args"].split()
        assert "--/crashreporter/url=" in arguments
        assert "--/app/uploadDumpsOnStartup=false" in arguments
        assert "--/crashreporter/skipOldDumpUpload=true" in arguments
        assert "--/crashreporter/devOnlyOverridePrivacyAndForceUpload=false" in arguments
        assert "--/crashreporter/enabled=true" in arguments
        assert "--/crashreporter/preserveDump=true" in arguments
        assert "--/privacy/performance=false" in arguments
        # Kit launches the telemetry transmitter during extension startup, before
        # the post-start privacy load; it must already see every category off.
        assert "--/privacy/usage=false" in arguments
        assert "--/privacy/personalization=false" in arguments
        assert "--/telemetry/enableAnonymousData=false" in arguments
        assert "--/telemetry/enableAnonymousAppName=false" in arguments
        privacy = output / "crash-reporting-privacy.toml"
        assert f"--/structuredLog/privacySettingsFile={privacy}" in arguments
        assert bridge.tomllib.loads(privacy.read_text()) == {
            "privacy": {"performance": False, "personalization": False, "usage": False}}
        assert {key: bridge.os.environ[key] for key in bridge._CRASH_REPORT_ENV} == bridge._CRASH_REPORT_ENV
        raise StopBeforeSDK

    monkeypatch.setitem(sys.modules, "isaaclab.app", SimpleNamespace(AppLauncher=launcher))
    with pytest.raises(StopBeforeSDK):
        bridge.build_environment({"runtime_directory": str(tmp_path / "runtime"), "robodojo_root": str(root),
                                  "asset_root": str(tmp_path / "assets"), "output_root": str(output),
                                  "renderer_gpu": 2, "physics_device": "cuda:0"})


@pytest.mark.parametrize("mutation", [None, "layer_bytes", "extra_property", "wrong_limit", "manifest", "identity", "readback_limit", "patch"])
def test_vulkan_compatibility_requires_exact_installation_and_preserves_device_identity(tmp_path, monkeypatch, mutation):
    import hashlib

    monkeypatch.setattr(graphics_compat.os, "environ", {})
    root = tmp_path / "graphics-compat"
    root.mkdir()
    layer, profile, manifest = (root / name for name in ("layer.so", "profile.json", "manifest.json"))
    layer.write_bytes(b"fixture layer")
    limit = 123 if mutation == "wrong_limit" else graphics_compat.CAP
    properties = {"VkPhysicalDeviceMaintenance3Properties": {"maxMemoryAllocationSize": limit}}
    if mutation == "extra_property":
        properties["VkPhysicalDeviceVulkan11Properties"] = {"deviceUUID": "invented"}
    profile.write_text(json.dumps({
        "capabilities": {"allocation": {"properties": properties}},
        "profiles": {graphics_compat.PROFILE: {"capabilities": ["allocation"]}},
    }))
    manifest.write_text(json.dumps({"layer": {
        "name": "VK_LAYER_KHRONOS_profiles", "type": "GLOBAL", "library_path": str(layer),
        "enable_environment": {"GLOBAL_ENABLE" if mutation == "manifest" else "ROBODOJO_VULKAN_COMPAT_ENABLE": "1"},
        "disable_environment": {"ROBODOJO_VULKAN_COMPAT_DISABLE": ""},
    }}))
    patch = root / "core-id.patch"
    patch.write_bytes((Path(__file__).resolve().parents[3] / "ora/environment/backends/robodojo/installation/vulkan-profiles-core-id.patch").read_bytes())
    receipt = {
        "patches": [{"path": patch.name, "sha256": hashlib.sha256(patch.read_bytes()).hexdigest()}],
        "schema": "ora.robodojo-graphics-compat-installation.v1",
        "source_url": "https://github.com/KhronosGroup/Vulkan-Profiles.git",
        "source_revision": graphics_compat.SOURCE_REVISION,
        **{role: {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
           for role, path in (("layer", layer), ("profile", profile), ("manifest", manifest))},
    }
    (root / "installation.json").write_text(json.dumps(receipt))
    if mutation == "patch":
        patch.write_bytes(b"unreviewed patch")
    if mutation == "layer_bytes":
        layer.write_bytes(b"altered layer")
    before = [{"device_uuid": "a" * 32, "physical_properties_sha256": "b" * 64,
               "max_memory_allocation_size": 2**64 - 1}]
    after = deepcopy(before)
    after[0]["max_memory_allocation_size"] = 1 if mutation == "readback_limit" else graphics_compat.CAP
    if mutation == "identity":
        after[0]["device_uuid"] = "0" * 32

    def readback(*, expected_layer=None):
        if expected_layer is not None:
            assert expected_layer == layer
            assert graphics_compat.os.environ["VK_KHRONOS_PROFILES_EMULATE_PORTABILITY"] == "false"
            assert graphics_compat.os.environ["VK_KHRONOS_PROFILES_SIMULATE_CAPABILITIES"] == "SIMULATE_PROPERTIES_BIT"
            return after
        return before

    monkeypatch.setattr(graphics_compat, "properties", readback)
    if mutation is not None:
        with pytest.raises(graphics_compat.GraphicsCompatibilityError):
            graphics_compat.readback(tmp_path)
    else:
        value = graphics_compat.readback(tmp_path)
        assert value["before"] == before and value["after"] == after
        assert value["files"] == {**{key: receipt[key] for key in ("layer", "profile", "manifest")}, "patch": receipt["patches"][0]}


@pytest.mark.parametrize("kind,mutation", [("rgb", None), ("depth", None),
    ("depth", "dtype"), ("rgb", "shape"), ("rgb", "empty"), ("rgb", "multi_env")])
def test_camera_readback_preserves_original_pixels_and_rejects_malformed_buffers(tmp_path, kind, mutation):
    from ora.environment.backends.robodojo.camera_capture import install_single_environment_readback

    channels = 4 if kind == "rgb" else 1
    values = np.arange(3 * 5 * channels, dtype=np.uint8 if kind == "rgb" else np.float32).reshape(3, 5, channels)
    if kind == "depth":
        values[0, 0, 0] = np.inf
    expected = values.copy()
    if mutation == "dtype":
        values = values.astype(np.float64)
    if mutation == "shape":
        values = values.transpose(1, 0, 2)
    if mutation == "empty":
        values = values[:0]

    class Annotator:
        def get_data(self, *, device):
            assert device == "cpu"
            return {"data": values, "info": {"fixture": True}}

    view = SimpleNamespace(prims=[object()], camera_resolution=(5, 3), tiled_resolution=(5, 3),
                           _render_product_path="/Render/OriginalCamera", _annotators={
                               "rgba" if kind == "rgb" else "distance_to_image_plane": Annotator()})
    manager = SimpleNamespace(num_envs=2 if mutation == "multi_env" else 1, tiled_cameras=[view] * 3)
    if mutation == "multi_env":
        with pytest.raises(ValueError, match="one environment"):
            install_single_environment_readback(manager, tmp_path)
        return
    install_single_environment_readback(manager, tmp_path)
    if mutation:
        with pytest.raises(ValueError, match="unexpected shape/dtype"):
            view.get_data(kind)
    else:
        actual, info = view.get_data(kind)
        np.testing.assert_array_equal(actual, expected[None])
        values.fill(0)
        np.testing.assert_array_equal(actual, expected[None])
        assert info == {"fixture": True}
    evidence = json.loads((tmp_path / "camera-buffer-readback.json").read_text())
    assert evidence["buffers"][0]["pixels_resampled"] is False


@pytest.mark.parametrize("changed", [None, *bridge._CRASH_REPORT_SETTINGS,
                                    "/structuredLog/privacySettingsFile", "environment", "consent_file"])
def test_crash_report_readback_rejects_upload_or_diagnostic_setting_drift(tmp_path, monkeypatch, changed):
    for key in bridge._CRASH_REPORT_ENV:
        monkeypatch.setenv(key, "inherited")
    _, privacy = bridge._configure_crash_reporting(tmp_path)
    actual = {**bridge._CRASH_REPORT_SETTINGS, "/structuredLog/privacySettingsFile": str(privacy)}
    if changed == "environment":
        monkeypatch.setenv("OMNI_CRASHREPORTER_URL", "https://example.invalid/upload")
    elif changed == "consent_file":
        privacy.chmod(0o644)
        privacy.write_text("[privacy]\nperformance = true\n")
    elif changed is not None:
        actual[changed] = not actual[changed] if isinstance(actual[changed], bool) else "unexpected"
    if changed is not None:
        with pytest.raises(RobodojoError, match="crash-report privacy"):
            bridge._crash_reporting_readback(actual, privacy)
    else:
        evidence = bridge._crash_reporting_readback(actual, privacy)
        assert evidence["uploads_disabled"] is True and evidence["local_dumps_preserved"] is True
        assert evidence["privacy_file"]["path"] == privacy.name
        assert len(evidence["privacy_file"]["sha256"]) == 64


@pytest.mark.parametrize("corruption", [None, "wrong_cuda", "wrong_index", "ambiguous_prefix", "two_active", "no_active"])
def test_renderer_readback_requires_actual_active_uuid_and_cuda_agreement(corruption):
    gpu_uuid = "GPU-a0161c6f-1234-5678-1234-123456789abc"
    table = "\n".join([
        "| 2 | NVIDIA RTX 5880 | Yes: 0 | | 49386 MB | 10de | 0 |",
        "|   |                 |        | |          | 26b3 | a0161c6f.. |",
        "|   |                 |        | |          | 38   |             |",
    ])
    inventory = [{"index": 2, "uuid": gpu_uuid, "pci_bus_id": "00000000:38:00.0"}]
    selected_index, cuda_uuid = 2, gpu_uuid
    if corruption == "wrong_cuda":
        cuda_uuid = "GPU-00000000-1234-5678-1234-123456789abc"
    elif corruption == "wrong_index":
        selected_index = 0
    elif corruption == "ambiguous_prefix":
        inventory.append({**inventory[0], "index": 3, "uuid": gpu_uuid[:-1] + "d"})
    elif corruption == "two_active":
        table += "\n" + table
    elif corruption == "no_active":
        table = table.replace("Yes: 0", "      ")
    if corruption is not None:
        with pytest.raises(RobodojoError):
            renderer_table_binding(table, selected_index=selected_index, cuda_uuid=cuda_uuid, inventory=inventory)
        return
    binding = renderer_table_binding(table, selected_index=selected_index, cuda_uuid=cuda_uuid, inventory=inventory)
    assert binding["selected_gpu_uuid"] == gpu_uuid
    assert binding["renderer_vulkan_index"] == 2 and binding["physics_cuda_logical_index"] == 0


class Robot:
    def __init__(self, side):
        self.type = "target"
        self.robot_name = side
        self.entity_origin_pose = [0, 0, 0, 1, 0, 0, 0]
        self.arm_name, self.gripper_name = f"{side}_arm", f"{side}_ee"
        self.gripper_scale = [-0.01, 0.044]
        self.gripper_move = {"sign": 1, "mimic": ["joint8", 1, 0]}
        self.ee_type = "gripper"
        self.ee_link_name = "link6"
        self.arm_joints_name = [f"joint{index}" for index in range(1, 7)]
        self.gripper_joints_name = ["joint7", "joint8"]
        self.arm_joint_indices = list(range(6))
        self.gripper_joint_indices = [6, 7]


class Manager:
    def __init__(self):
        self.robot_list = [Robot("left"), Robot("right")]
        self.ik_solver = {robot.robot_name: SimpleNamespace(manager=self, robot=robot)
                          for robot in self.robot_list}
        self.robot_key = [SimpleNamespace(
            is_fixed_base=True, joint_names=[f"joint{index}" for index in range(1, 9)],
            body_names=["base_link", "link6", "camera"],
        ) for _ in self.robot_list]
        self.fail_ik = False
        # The synthetic home joint vector, which the fake kinematics below read
        # back as the link6 pose; ``set_home`` moves it so a test can place the
        # fake plant inside a specific action envelope.
        self.home = [0, 0, 0.8, 0, 0, 0]
        self.joints = {robot.arm_name: list(self.home) for robot in self.robot_list}
        self.fingers = {robot.arm_name: [0.044, 0.044] for robot in self.robot_list}
        self.control_manager = SimpleNamespace(prev_control=[{
            key: {"position": list(value)}
            for robot in self.robot_list
            for key, value in ((self.process_name(robot.arm_name), self.joints[robot.arm_name]),
                               (self.process_name(robot.gripper_name), self.fingers[robot.arm_name]))
        }])
        self.unreachable = None
        for articulation in self.robot_key:
            articulation.data = SimpleNamespace(joint_pos_target=np.array([[*self.home, .044, .044]]),
                                                joint_pos_limits=np.array([[[-3.14, 3.14]] * 8]))
        for robot, articulation in zip(self.robot_list, self.robot_key):
            articulation.write_joint_state_to_sim = (
                lambda position, velocity, joint_ids=None, env_ids=None, _arm=robot.arm_name:
                self.write_joint_state(_arm, position, joint_ids))

    def write_joint_state(self, arm, position, joint_ids=None):
        """Teleport ``arm`` joints like Isaac Lab's ``write_joint_state_to_sim``."""
        ids = list(range(6)) if joint_ids is None else list(joint_ids)
        for column, index in enumerate(ids):
            if index < 6:
                self.joints[arm][index] = float(position[0][column])

    def set_home(self, joints):
        """Rebind the synthetic home pose and settle the plant on it."""
        self.home = [float(value) for value in joints]
        self.reset_home()

    def reset_home(self):
        for robot in self.robot_list:
            self.joints[robot.arm_name] = list(self.home)
            self.fingers[robot.arm_name] = [0.044, 0.044]
            self.control_manager.prev_control[0][self.process_name(robot.arm_name)] = {"position": list(self.home)}
            self.control_manager.prev_control[0][self.process_name(robot.gripper_name)] = {"position": [0.044, 0.044]}
        for articulation in self.robot_key:
            articulation.data.joint_pos_target[0] = [*self.home, .044, .044]

    @staticmethod
    def process_name(name):
        return name + "_joint_state"

    def get_joint(self, robot, env_idx_list=None):
        return {0: self.joints[robot.arm_name]}

    def get_real_endpose(self, robot, env_idx_list=None):
        # Fake kinematics: joints 0-2 are the link6 position, joints 3-5 the
        # xyz quaternion components on the w >= 0 hemisphere.
        joints = self.joints[robot.arm_name]
        vector = [float(value) for value in joints[3:6]]
        w = math.sqrt(max(0.0, 1.0 - sum(value * value for value in vector)))
        return {0: list(joints[:3]) + [w] + vector}

    def get_end_effector_real_val(self, robot, env_idx_list=None):
        return {0: self.fingers[robot.arm_name]}

    def solve_ik(self, target_pose, env_idx, robot):
        if self.fail_ik or (self.unreachable is not None and self.unreachable(target_pose)):
            return {"status": "Fail"}
        sign = -1.0 if target_pose[3] < 0 else 1.0
        return {"status": "Success", "joint_value": list(target_pose[:3]) + [sign * v for v in target_pose[4:7]]}


class Plant:
    def __init__(self):
        self.robot_manager = Manager()
        camera_names = ["cam_head", "cam_left_wrist", "cam_right_wrist"]
        mounts = [None, "robot0/camera", "robot1/camera"]
        self.camera_manager = SimpleNamespace(
            camera_names=[camera_names],
            cameras=[[SimpleNamespace(prim_path=f"/World/envs/env_0/{mount + '/' if mount else ''}Camera_{index}/optical")
                      for index, mount in enumerate(mounts)]],
            camera_config={name: SimpleNamespace(camera={"mount_link": mount})
                           for name, mount in zip(camera_names, mounts)},
        )
        self.sim = SimpleNamespace(_sim_step_counter=500)
        self.sim.sim = SimpleNamespace(current_time_step_index=500)
        self.take_action_cnt = [0]
        self.step_lim = 1050
        self.reward_registered = False
        # The upstream episode-lifecycle latches the round-isolation audit
        # measures: ``success`` is the failure latch (True until the evaluator
        # marks a failure) and ``end_flag`` the episode's own terminal flag.
        self.success = [True]
        self.end_flag = [False]
        self._official_terminal = None
        # The layout instances upstream registers as scene objects, and the
        # monotone id counter ``SceneManager.reload_env_scene`` regenerates
        # them from. ``rename_on_reset`` reproduces exactly that upstream
        # behaviour, so a test can watch the identity check fail closed.
        self.layout_generation = 0
        self.rename_on_reset = False
        self.scene_registration = None
        self.obs_manager = SimpleNamespace(instruction=["Pick up the displayed object by 10 cm."])
        self.terminal_at = None
        self.commands = []
        self.closed = False
        self.leak_clock = False

    def reset(self, seed):
        from ora.environment.benchmarks.robodojo import (
            build_scene_registration,
        )

        assert seed == [0]
        self.take_action_cnt = [0]
        self.reward_registered = True
        self.success = [True]
        self.end_flag = [False]
        self._official_terminal = None
        self.resets = getattr(self, "resets", 0) + 1
        # Upstream only reloads the saved layout on a *later* reset of a live
        # plant, so the first reset keeps generation 0 either way.
        if self.rename_on_reset and getattr(self, "resets", 0) > 1:
            self.layout_generation += 1
        suffix = self.layout_generation
        self.scene_registration = build_scene_registration(
            {f"mug_{suffix}": "rigid", f"rack_{suffix}": "geometry"},
            evaluator_registration={
                "check_list": True, "final_check_list": False,
                "trigger_check_list": False,
            },
            native_action_limit=self.step_lim,
        )
        self.robot_manager.reset_home()

    def native_tick(self, control):
        self.commands.append(deepcopy(control))
        self.take_action_cnt[0] += 1
        manager = self.robot_manager
        fraction = getattr(self, "step_fraction", 1.0)
        for robot in manager.robot_list:
            arm_key, ee_key = manager.process_name(robot.arm_name), manager.process_name(robot.gripper_name)
            if arm_key in control:
                manager.joints[robot.arm_name] = [
                    current + fraction * (target - current)
                    for current, target in zip(manager.joints[robot.arm_name], control[arm_key]["position"])
                ]
            if ee_key in control:
                limits = manager.robot_key[manager.robot_list.index(robot)].data.joint_pos_limits[0]
                manager.fingers[robot.arm_name] = [
                    max(float(limits[index][0]), value)
                    for index, value in zip(robot.gripper_joint_indices, control[ee_key]["position"])
                ]
        manager.control_manager.prev_control[0].update(deepcopy(control))
        for robot, articulation in zip(manager.robot_list, manager.robot_key):
            targets = manager.control_manager.prev_control[0]
            articulation.data.joint_pos_target[0] = (
                targets[manager.process_name(robot.arm_name)]["position"]
                + targets[manager.process_name(robot.gripper_name)]["position"]
            )
        for _ in range(10):
            self.sim._sim_step_counter += 1
            self.sim.sim.current_time_step_index += 1
            callback = getattr(self, "controller_trace_callback", None)
            if callback is not None:
                callback(deepcopy(manager.control_manager.prev_control[0]))
        if self.take_action_cnt[0] == self.terminal_at:
            return {"official_success": True, "native_actions": self.take_action_cnt[0]}
        if self.take_action_cnt[0] == self.step_lim:
            return {"official_success": False, "native_actions": self.step_lim}
        return None

    def get_obs(self):
        self.sim._sim_step_counter += int(self.leak_clock)
        self.sim.sim.current_time_step_index += int(self.leak_clock)
        return {
            "object_gt": "must not reach the public manifest",
            "vision": {name: {"color": np.full((8, 10, 3), 100, dtype=np.uint8),
                              "depth": np.full((8, 10), 1.0, dtype=np.float32)}
                       for name in ("cam_head", "cam_left_wrist", "cam_right_wrist")},
        }

    def close(self):
        self.closed = True


def _fake_source_manifest(tmp_path):
    """A source manifest whose layout file really exists on disk.

    The round-isolation audit re-hashes the layout the session was started
    against, so the fixture has to bind a real file rather than a name.
    """

    layout = tmp_path / "layout" / "hang_mugs_0.json"
    layout.parent.mkdir(parents=True, exist_ok=True)
    if not layout.is_file():
        layout.write_text('{"objects": []}', encoding="utf-8")
    return {
        "schema": "ora.robodojo-source-manifest.v1",
        "source_revisions": {"robodojo": "1" * 40},
        "layout_file": {
            "path": str(layout),
            "sha256": hashlib.sha256(layout.read_bytes()).hexdigest(),
        },
    }


@pytest.fixture
def session(tmp_path):
    plant = Plant()
    instance = RobodojoSession(
        config={"workspace_bounds_m": {"x": [-1, 1], "y": [-1, 1], "z": [0, 2]}},
        output_root=tmp_path / "native",
        environment_factory=lambda config: (
            plant, SimpleNamespace(close=lambda: None), _fake_source_manifest(tmp_path),
        ),
    )
    instance._calibration = lambda name: (np.array([[50, 0, 5], [0, 50, 4], [0, 0, 1]]), np.eye(4))
    instance.start()
    try:
        yield instance
    finally:
        instance.close()


def action(session, tmp_path, *, kind="move_pose", arm="left", value=None):
    observation, _ = session.observe(tmp_path / "observations")
    arguments = ({"arm": arm, "position_m": [0.02, 0, 0.8], "orientation_xyzw": [0, 0, 0, 1]}
                 if kind == "move_pose" else {"arm": arm, "gripper": 0.0 if value is None else value})
    return {"schema": "ora-action.v3", "observation_id": observation["observation_id"],
            "action_id": f"action-{session.action_sequence}", "action": kind, "arguments": arguments,
            "bound_observation_sha256": "a" * 64}


def test_observation_is_measured_immutable_rgbd_and_omits_private_state(session, tmp_path):
    manager = session.environment.robot_manager
    manager.fingers["left_arm"] = [0.017, 0.017]
    payload, paths = session.observe(tmp_path / "observations")
    assert payload["proprioception"]["end_effectors"]["left"]["gripper"] == pytest.approx(0.5)
    assert payload["proprioception"]["end_effectors"]["left"]["gripper_command"] == 1.0
    assert "object_gt" not in payload
    assert len(paths) == 3
    assert session.environment.sim._sim_step_counter == 500
    assert all(path.stat().st_mode & 0o222 == 0 for path in paths)
    assert not session.is_terminal
    assert session.persist_official_verdict()["official_success"] is None


def test_pose_action_keeps_gripper_and_other_arm_commands(session, tmp_path):
    before = deepcopy(session.environment.robot_manager.control_manager.prev_control)
    receipt = session.execute(action(session, tmp_path))["receipt"]
    assert receipt["motion_status"] == "reached"
    assert receipt["native_actions"] == 1 and receipt["physics_steps"] == 10
    assert set(session.environment.commands[0]) == {"left_arm_joint_state"}
    after = session.environment.robot_manager.control_manager.prev_control
    assert before[0]["right_arm_joint_state"] == after[0]["right_arm_joint_state"]
    assert before[0]["left_ee_joint_state"] == after[0]["left_ee_joint_state"]
    next_observation, _ = session.observe(tmp_path / "observations")
    reference = next_observation["previous_execution_receipt"]
    assert reference["receipt"] == receipt
    assert json.loads((session.output_root / reference["path"]).read_text()) == receipt


def test_sdk_target_mismatch_aborts_even_when_retained_intent_is_unchanged(session, tmp_path):
    original = session._applied_position_targets

    def corrupted():
        values = original()
        values["left_ee_joint_state"] = [.03, .03]
        return values

    session._applied_position_targets = corrupted
    with pytest.raises(RobodojoError, match="SDK applied"):
        session.execute(action(session, tmp_path))
    assert session.abort_reason == "robodojo_applied_command_mismatch"
    assert session.gripper_directives["left"] == 1.0
    journal = session.output_root / "controller/action-0000.jsonl"
    evidence = json.loads(journal.read_text().splitlines()[0])
    assert evidence["applied_position_targets"]["left_ee_joint_state"] == [.03, .03]
    assert session.persist_official_verdict()["official_success"] is None


@pytest.mark.parametrize('mode', ['session', 'sparse'])
@pytest.mark.parametrize('corruption', [None, 'auxiliary', 'held_target'])
def test_interactive_arm_executes_and_is_journaled_without_becoming_a_policy_target(
    request, tmp_path, mode, corruption,
):
    session = request.getfixturevalue(mode)
    plant = session.environment
    plant.interact = True
    manager = plant.robot_manager
    robot = Robot('support')
    robot.type = 'support'
    manager.robot_list.append(robot)
    manager.robot_key.append(deepcopy(manager.robot_key[0]))
    manager.joints[robot.arm_name] = list(manager.home)
    manager.fingers[robot.arm_name] = [.044, .044]
    support_key = manager.process_name(robot.arm_name)
    manager.control_manager.prev_control[0].update({
        support_key: {'position': list(manager.home)},
        manager.process_name(robot.gripper_name): {'position': [.044, .044]},
    })
    native_tick = plant.native_tick

    def autonomous_tick(control):
        auxiliary = list(manager.home)
        auxiliary[0] = .1
        return native_tick({**control, support_key: {'position': auxiliary}})
    plant.native_tick = autonomous_tick
    if corruption:
        readback = session._applied_position_targets
        def damaged():
            result = readback()
            result[support_key if corruption == 'auxiliary' else 'right_arm_joint_state'][0] += .01
            return result
        session._applied_position_targets = damaged
    command = (action(session, tmp_path, kind='set_gripper', value=.5) if mode == 'session'
               else v3(session, tmp_path, kind='set_gripper', targets={'left': 'close'}))
    if corruption:
        with pytest.raises(RobodojoError, match='SDK applied'):
            session.execute(command)
        assert session.abort_reason == 'robodojo_applied_command_mismatch'
    else:
        receipt = session.execute(command)['receipt']
        assert receipt['command_invariant']['verified'] is True
        assert all('support' not in key for field in ('command_targets_before', 'command_targets_after')
                   for key in receipt[field])
        observation, _ = session.observe(tmp_path/'post-observation')
        assert set(observation['proprioception']['end_effectors']) == {'left', 'right'}
    samples = [json.loads(line) for line in (session.output_root/'controller/action-0000.jsonl').read_text().splitlines()]
    assert samples[0]['command'][support_key]['position'][0] == .1
    assert support_key in samples[0]['applied_position_targets']


def test_native_two_tool_path_projects_actual_robodojo_state_and_receipt(session, tmp_path):
    from ora.agent.context import ContextProjector
    from ora.agent.evidence.policy_state import PolicyStateStore, verify_policy_state
    from ora.agent.native.session import NativePlanner
    from ora.agent.providers.native import FakeProvider
    from ora.agent.tools.robot import RobotToolRuntime

    root = Path(__file__).resolve().parents[3]
    # Sample policy text copied from the robodojo-compat package (relocation U6).
    rules = root / "tests/data/robodojo/planner-v1.md"
    prompt = root / "tests/data/robodojo/core-v1.md"
    manual = root / "ora/policy/embodiments/arx-x5/compiled-v1/manual.md"
    store = PolicyStateStore(tmp_path, "robodojo-native-boundary", "a" * 64)
    projector = ContextProjector(
        tmp_path / "workspace", tmp_path / "control", prompt, manual, rules,
        main_camera="head", expose_manual=False,
    )
    tools = RobotToolRuntime(
        tmp_path / "control", manual, tmp_path / "events.jsonl", session.config["workspace_bounds_m"],
        allowed_tools=["observation.dump_state", "robot.submit_action"], state_root=store.root,
    )

    def publish(previous=None):
        payload, _ = session.observe(tmp_path / "observations")
        source = tmp_path / "observations" / payload["observation_id"]
        shared = dict(observation=payload, observation_root=source,
                      task_instruction="Pick up the displayed object by 10 cm.",
                      remaining_budget={"robot_decisions": 199, "model_calls": 398},
                      previous_execution_receipt=previous)
        snapshot = store.publish(**shared, cause="reset" if previous is None else "after_action",
                                 previous_public_action=None)
        record = store.steps[-1]
        state_ref = {"step": snapshot["step"], "path": record["path"],
                     "sha256": record["sha256"], "state_sha256": snapshot["state_sha256"]}
        return projector.publish(**shared, policy_state=state_ref)

    pointer = publish()
    calls = [("observation.dump_state", {}), ("robot.submit_action", {
        "action": "move_pose", "arm": "right", "position_m": [0.02, 0, 0.8],
        "orientation_xyzw": [0, 0, 0, 1],
    })]
    provider = FakeProvider([
        {"message": {"role": "assistant", "content": "", "tool_calls": [{
            "id": f"call-{index}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }]}, "usage": {"input_tokens": 100, "cached_input_tokens": 0,
                       "output_tokens": 10, "reasoning_output_tokens": 0, "total_tokens": 110}}
        for index, (name, arguments) in enumerate(calls)
    ])
    planner = NativePlanner(
        tmp_path, "robodojo-native-boundary", tools, provider, rules, prompt,
        "fake", "native_minimal_v1", 32768, max_output_tokens=2048,
    )
    try:
        outcome = planner.run_observation(
            task_instruction="Pick up the displayed object by 10 cm.", context=pointer,
            image_path=projector.workspace / pointer["main_image"],
        )
        assert outcome["submitted"] is True
        assert outcome["action"]["arguments"]["arm"] == "right"
        initial = [message for message in provider.requests[0]["messages"]
                   if message["role"] == "user"][-1]
        assert any(part["type"] == "image_url" for part in initial["content"])
        receipt = session.execute(outcome["action"])
        planner.record_environment_receipt(receipt)
        publish(receipt)
        state = tools.invoke("observation.dump_state", {})
        assert state["image_camera"] == "head" and state["image_base64"]
        assert state["camera_inventory"]["left_wrist"]["calibration_dynamic"] is True
        assert state["camera_inventory"]["right_wrist"]["modalities"] == ["rgb", "depth", "calibration"]
        assert state["state"]["previous_execution_receipt"] == receipt
        assert state["state"]["proprioception"]["end_effectors"]["right"]["pose_xyz_xyzw"][:3] == [0.02, 0, 0.8]
        assert verify_policy_state(tmp_path)["status"] == "valid"
    finally:
        planner.close()


def test_gripper_action_holds_pose_target_and_other_arm(session, tmp_path):
    before = deepcopy(session.pose_targets)
    receipt = session.execute(action(session, tmp_path, kind="set_gripper"))["receipt"]
    assert receipt["reached_gripper"] == 0
    assert set(session.environment.commands[0]) == {"left_ee_joint_state"}
    assert session.pose_targets == before


@pytest.mark.parametrize("corruption", [None, "missing_collector", "missing_link", "wrong_camera_mount", "source_hash"])
def test_embodiment_activation_requires_bound_compiled_runtime_evidence(session, tmp_path, corruption):
    import hashlib
    from ora.orchestrator.robodojo_embodiment import CARD_ROOT, create_activation
    from ora.agent.composition.embodiment import descriptor_activation_ready
    from ora.environment.controllers.robodojo_arx import controller_binding

    descriptor = json.loads((CARD_ROOT / "descriptor-v1.json").read_text())
    provenance = descriptor["provenance"]["sources"][1]
    qualification_root = tmp_path / "activation-probe"
    reports = []
    for index in range(2):
        environment_root = qualification_root / f"reset-{index}" / "environment"
        environment_root.mkdir(parents=True)
        source = {"source_revisions": {"assets": provenance["revision"]},
                  "loaded_files": {"Assets/Robots/x5/robot_config.yml": provenance["sha256"]}}
        if corruption == "source_hash":
            source["loaded_files"]["Assets/Robots/x5/robot_config.yml"] = "0" * 64
        source_path = environment_root / "source-manifest.json"
        source_path.write_text(json.dumps(source))
        runtime = session._runtime_embodiment()
        if corruption == "missing_collector":
            runtime.pop("collector")
        elif corruption == "missing_link":
            runtime["arms"]["left"]["compiled_body_names"].remove("link6")
        elif corruption == "wrong_camera_mount":
            runtime["cameras"]["left_wrist"]["mount_link"] = "robot1/camera"
        reset = {"source_manifest": {"sha256": hashlib.sha256(source_path.read_bytes()).hexdigest()},
                 "control_binding": controller_binding(), "runtime_embodiment": runtime}
        (environment_root / "reset.json").write_text(json.dumps(reset))
        sensors = {name: {"finite_positive_fraction": 1.0, "shape": [8, 10]}
                   for name in ("head", "left_wrist", "right_wrist")}
        report = {"passed": True, "reset": reset, "initial_sensors": sensors, "repeated_sensors": sensors}
        (environment_root.parent / "probe-report.json").write_text(json.dumps(report))
        reports.append(report)
    if corruption is not None:
        with pytest.raises(RuntimeError, match="activation rejected"):
            create_activation(qualification_root, reports)
        assert not (qualification_root / "embodiment/activation.json").exists()
        return
    binding = create_activation(qualification_root, reports)
    activation_path = qualification_root / binding["activation_record"]["path"]
    activation = json.loads(activation_path.read_text())
    assert descriptor_activation_ready(descriptor, activation_record=activation) is True
    assert hashlib.sha256(activation_path.read_bytes()).hexdigest() == binding["activation_record"]["sha256"]
    assert activation_path.stat().st_mode & 0o222 == 0


def test_failed_ik_does_not_step_or_change_gripper(session, tmp_path):
    session.environment.robot_manager.fail_ik = True
    before = deepcopy(session.environment.robot_manager.control_manager.prev_control)
    receipt = session.execute(action(session, tmp_path))["receipt"]
    assert receipt["motion_status"] == "ik_failed"
    assert receipt["native_actions"] == receipt["physics_steps"] == 0
    assert before == session.environment.robot_manager.control_manager.prev_control
    assert session.environment.commands == []


def test_official_terminal_is_snapshot_and_rejects_postterminal_actions(session, tmp_path):
    session.environment.terminal_at = 1
    proposal = action(session, tmp_path)
    session.execute(proposal)
    assert session.persist_official_verdict()["official_success"] is True
    session.environment.success = [False]  # Mutating an upstream eligibility flag cannot revise the snapshot.
    for _ in range(4):
        assert session.status()["reason"] == "official_success"
    session.observe(tmp_path / "observations")
    with pytest.raises(RobodojoError, match="cannot accept"):
        session.execute(proposal)
    session.abort("operator_stop")
    assert session.persist_official_verdict()["official_success"] is True
    assert session.environment.take_action_cnt == [1]


def test_official_exhaustion_is_false_but_outer_budget_is_incomplete(session, tmp_path):
    session.environment.take_action_cnt = [1049]
    session.sim_steps = 10490
    session.execute(action(session, tmp_path, kind="set_gripper"))
    verdict = session.persist_official_verdict()
    assert verdict["terminal"] and verdict["official_success"] is False
    assert verdict["native_actions"] == 1050


def test_policy_budget_reports_the_standing_official_predicate(session, tmp_path):
    """A budget stop is a behavioral boundary, so it carries a Boolean.

    RoboDojo latches success inside ``RewardManager.step``, which runs after
    every native action, and nothing advances physics outside a native tick.
    So an episode with no official terminal transition has a standing official
    predicate that was already evaluated after the last physics step and was
    not a success. That is also why the Boolean on this path is always false:
    a success would have produced the terminal transition instead.
    """

    session.execute(action(session, tmp_path))
    assert session.persist_official_verdict()["verdict_basis"] == "episode_incomplete"
    session.abort("robot_decision_budget")
    verdict = session.persist_official_verdict()
    assert verdict["official_success"] is False and verdict["terminal"] is False
    assert verdict["verdict_basis"] == "standing_official_predicate_at_policy_budget"
    assert verdict["fault_reason"] is None
    assert verdict["abort_reason"] == "robot_decision_budget"


def test_exhausted_episode_physics_budget_reports_the_same_predicate(session, tmp_path):
    session.max_sim_steps = 10
    session.execute(action(session, tmp_path))
    assert session.is_terminal
    verdict = session.persist_official_verdict()
    assert verdict["official_success"] is False and verdict["terminal"] is False
    assert verdict["verdict_basis"] == "standing_official_predicate_at_policy_budget"
    assert verdict["fault_reason"] is None


def test_official_success_on_the_last_decision_outranks_the_budget_stop(session, tmp_path):
    """The only reachable true verdict at a budget edge is a real transition."""

    session.environment.terminal_at = 1
    session.execute(action(session, tmp_path))
    session.abort("robot_decision_budget")
    verdict = session.persist_official_verdict()
    assert verdict["official_success"] is True and verdict["terminal"] is True
    assert verdict["verdict_basis"] == "official_terminal_transition"
    assert verdict["fault_reason"] is None and verdict["abort_reason"] is None


def test_infrastructure_fault_never_becomes_an_official_task_failure(session, tmp_path):
    session.environment.leak_clock = True
    with pytest.raises(RobodojoError, match="advanced physics"):
        session.observe(tmp_path / "observations")
    session.abort("robot_decision_budget")  # A later budget cannot relabel a fault.
    verdict = session.persist_official_verdict()
    assert verdict["official_success"] is None and verdict["terminal"] is False
    assert verdict["verdict_basis"] == "infrastructure_abort"
    assert verdict["fault_reason"] == "observation_advanced_physics"
    assert verdict["abort_reason"] == "observation_advanced_physics"


def test_observation_clock_violation_and_stale_action_fail_closed(session, tmp_path):
    proposal = action(session, tmp_path)
    proposal["observation_id"] = "stale"
    with pytest.raises(RobodojoError, match="latest unconsumed"):
        session.execute(proposal)
    session.environment.leak_clock = True
    with pytest.raises(RobodojoError, match="advanced physics"):
        session.observe(tmp_path / "observations")
    assert session.is_terminal


def test_official_transition_requires_registration_and_never_repolls_terminal():
    lifecycle = object.__new__(EvaluationLifecycle)
    lifecycle.reward_registered = False
    lifecycle._official_terminal = None
    with pytest.raises(RobodojoError, match="not been initialized"):
        lifecycle.advance_official_terminal()
    lifecycle.reward_registered = True
    lifecycle.take_action_cnt, lifecycle.success, lifecycle.end_flag = [200], [True], [False]
    lifecycle.step_lim = 200
    calls = []
    lifecycle.reward_manager = SimpleNamespace(get_reward=lambda final_check: calls.append(final_check) or [1.0])
    assert lifecycle.advance_official_terminal()["official_success"] is True
    assert lifecycle.advance_official_terminal()["official_success"] is True
    assert calls == [True]  # Success on the last native action takes precedence over exhaustion.


def test_interpolation_preserves_upstream_eight_ramp_two_hold_contract():
    lifecycle = object.__new__(EvaluationLifecycle)
    lifecycle.obs_manager = SimpleNamespace(collect_interval=10)
    lifecycle.robot_manager = Manager()
    sequence = lifecycle.interpolate({"left_arm_joint_state": {"position": [0.09, 0, 0.8, 0, 0, 0]}})
    assert len(sequence) == 10
    assert sequence[0]["left_arm_joint_state"]["position"][0] == pytest.approx(0.01)
    assert sequence[7]["left_arm_joint_state"]["position"][0] == pytest.approx(0.08)
    assert sequence[8]["left_arm_joint_state"]["position"][0] == pytest.approx(0.09)
    assert sequence[9] == sequence[8]


def test_every_reset_reinitializes_reward_after_task_reset_and_before_support():
    events = []

    class ParentTask:
        def reset(self, seed=None, options=None):
            events.append("task_reset_clears_checks")
            self.reward_manager.check_list = [[]]

    class Lifecycle(EvaluationLifecycle, ParentTask):
        def setup_scene(self):
            events.append("settle")

        def run_reward(self):
            events.append("register_reward")
            self.reward_manager.check_list[0].append("official_predicate")

        def get_score(self):
            events.append("register_score")

        def query_support_arm_traj(self, env_idx):
            events.append("support_init")

    lifecycle = object.__new__(Lifecycle)
    lifecycle.num_envs = 1
    lifecycle.interact = True
    lifecycle.step_lim = 1050
    lifecycle.scene_manager = SimpleNamespace(layout_manager=SimpleNamespace(
        set_saved_layout=lambda env_idx, layout: events.append("bind_layout"),
        instance_type_by_env=[{"mug_0": "rigid"}],
        get_scene_object=lambda env_idx, inst_name: object()))
    lifecycle.seed_manager = SimpleNamespace(get_seed_scene_info=lambda seed: {})
    lifecycle.obs_manager = SimpleNamespace(reset=lambda: events.append("observation_reset"))
    lifecycle.robot_manager = SimpleNamespace(set_origin_endpose=lambda: events.append("origin"),
                                             set_robot_init_state=lambda: events.append("controller_initial_state"))
    lifecycle.reward_manager = SimpleNamespace(check_list=[[]], final_check_list=[[]], trigger_check_list=[[]],
                                               init_state=lambda: events.append("predicate_initial_state"))
    lifecycle.capture_manager = SimpleNamespace(tiled_cameras=[])
    lifecycle._release_capture_products = lambda: events.append("release_previous_capture_products")
    for index in range(2):
        events.clear()
        lifecycle.reset(seed=[0])
        assert events == (["release_previous_capture_products"] if index else []) + [
            "bind_layout", "task_reset_clears_checks", "observation_reset", "settle", "origin",
            "controller_initial_state", "predicate_initial_state", "register_reward", "register_score", "support_init"]
        # The upstream reset appends new views; the next reset must release them first.
        lifecycle.capture_manager.tiled_cameras = [object()]
        assert len(lifecycle.reward_manager.check_list[0]) == 1
        assert lifecycle.success == [True] and lifecycle.end_flag == [False]


def _layout_reloading_lifecycle(events):
    """Model SceneManager.reload_env_scene: a reload of the saved layout renames every
    record from a monotone id counter while the CUDA pipeline keeps the round-0
    rigid objects registered under their original names."""
    registry = {}
    counter = {"value": 0}

    class Layout:
        def __init__(self):
            self.instance_type_by_env = [{}]

        def set_saved_layout(self, env_idx, layout):
            return None

        def load_saved_layout(self):
            self.instance_type_by_env[0] = {}
            for category, kind in (("mug_8", "rigid"), ("mug_5", "rigid"), ("cup_holder_0", "geometry")):
                counter["value"] += 1
                self.instance_type_by_env[0][f"{category}_{counter['value']}"] = kind

        def get_scene_object(self, env_idx, inst_name):
            return registry.get(inst_name)

    class ParentTask:
        def reset(self, seed=None, options=None):
            scene = self.scene_manager
            if self.sim is None:
                self.sim = object()
                scene.layout_manager.load_saved_layout()
                registry.update({name: object() for name, kind in scene.layout_manager.instance_type_by_env[0].items()
                                 if kind == "rigid"})
                scene.setup_scene = True
            events.append(("upstream_reset", scene.setup_scene))
            if not scene.setup_scene:
                scene.layout_manager.load_saved_layout()
            for name, kind in scene.layout_manager.instance_type_by_env[0].items():
                if kind == "geometry":
                    registry[name] = object()
            scene.setup_scene = False
            self.reward_manager.check_list = [[]]

    class Lifecycle(EvaluationLifecycle, ParentTask):
        def setup_scene(self):
            events.append("settle")

        def run_reward(self):
            self.reward_manager.check_list[0].append("official_predicate")

    lifecycle = object.__new__(Lifecycle)
    lifecycle.num_envs, lifecycle.sim = 1, None
    lifecycle.step_lim = 1050
    lifecycle.scene_manager = SimpleNamespace(layout_manager=Layout(), setup_scene=False)
    lifecycle.seed_manager = SimpleNamespace(get_seed_scene_info=lambda seed: {})
    lifecycle.obs_manager = SimpleNamespace(reset=lambda: None)
    lifecycle.robot_manager = SimpleNamespace(set_origin_endpose=lambda: None, set_robot_init_state=lambda: None)
    lifecycle.reward_manager = SimpleNamespace(check_list=[[]], final_check_list=[[]], trigger_check_list=[[]],
                                               init_state=lambda: None)
    lifecycle.capture_manager = SimpleNamespace(tiled_cameras=[])
    return lifecycle, registry


def test_later_resets_keep_the_layout_identities_the_upstream_reload_would_rename():
    events = []
    lifecycle, registry = _layout_reloading_lifecycle(events)
    lifecycle.reset(seed=[0])
    round0 = dict(lifecycle.scene_manager.layout_manager.instance_type_by_env[0])
    assert round0 == {"mug_8_1": "rigid", "mug_5_2": "rigid", "cup_holder_0_3": "geometry"}
    lifecycle.reset(seed=[0])
    assert lifecycle.scene_manager.layout_manager.instance_type_by_env[0] == round0
    assert set(round0) <= set(registry)
    # The first reset runs upstream's own post-initialization path; the bridge
    # restores that flag before every later reset of the same plant.
    assert [event for event in events if isinstance(event, tuple)] == [("upstream_reset", True)] * 2
    assert lifecycle.scene_manager.setup_scene is False and lifecycle.reward_registered is True


def test_reset_fails_closed_when_layout_records_no_longer_name_scene_objects():
    events = []
    lifecycle, _registry = _layout_reloading_lifecycle(events)
    lifecycle._keep_layout_identities = lambda: None
    lifecycle.reset(seed=[0])
    with pytest.raises(RobodojoError, match="no scene object after reset: mug_5_5, mug_8_4"):
        lifecycle.reset(seed=[0])
    assert "settle" not in events[events.index(("upstream_reset", False)):]
    assert lifecycle.reward_registered is False


def test_native_tick_rejects_terminal_before_any_reward_update():
    lifecycle = object.__new__(EvaluationLifecycle)
    lifecycle._official_terminal = {"official_success": True}
    lifecycle.end_flag = [True]
    with pytest.raises(RobodojoError, match="cannot accept"):
        lifecycle.native_tick({})


def test_close_is_idempotent(session):
    calls = []
    session.app.close = lambda: calls.append("app_close")
    session.close()
    session.close()
    assert calls == ["app_close"]
    assert session.environment.closed


@pytest.mark.parametrize("fault", [None, "delayed", "split_frames", "pose_lag", "timeout", "physics_leak"])
def test_render_barrier_joins_all_products_to_current_physics_without_stepping(tmp_path, fault):
    from ora.environment.backends.robodojo.camera_capture import synchronize_render_products

    state = {"renders": 0, "physics": 250}

    def render():
        state["renders"] += 1
        if fault == "physics_leak":
            state["physics"] += 1

    def stamp(index):
        late = index == 2 and (fault == "timeout" or (fault == "delayed" and state["renders"] == 1))
        return {"simulationTime": 0.8 if late else 1.0}

    def reference(index):
        split = fault == "split_frames" and index == 2 and state["renders"] == 1
        return {"referenceTimeNumerator": state["renders"] - int(split), "referenceTimeDenominator": 60}

    manager = SimpleNamespace(num_envs=1, tiled_cameras=[
        SimpleNamespace(_render_product_path=f"/camera/{index}",
                        _harness_camera_params=SimpleNamespace(get_data=lambda: {
                            "cameraViewTransform": np.eye(4).reshape(-1).tolist()}),
                        _harness_simulation_time=SimpleNamespace(get_data=lambda i=index: stamp(i)),
                        _harness_reference_time=SimpleNamespace(get_data=lambda i=index: reference(i)))
        for index in range(3)])
    def physical_poses():
        pose = np.diag([1., -1., -1., 1.])
        if fault == "pose_lag" and state["renders"] < 3:
            pose[0, 3] = .5
        return [pose] * 3

    kwargs = dict(manager=manager, render=render, physics_clock=lambda: state["physics"],
                  physical_poses=physical_poses,
                  simulation_time=lambda: 1.0, output_root=tmp_path, max_renders=3)
    if fault in {"timeout", "physics_leak"}:
        with pytest.raises((TimeoutError, ValueError)):
            synchronize_render_products(**kwargs)
    else:
        assert synchronize_render_products(**kwargs)["matched"] is True
        assert state["renders"] == (3 if fault == "pose_lag" else 2)
        assert state["physics"] == 250
    record = json.loads((tmp_path / "render-synchronization.jsonl").read_text())
    assert record["matched"] is (fault not in {"timeout", "physics_leak"})
    if fault == "timeout":
        assert len(record["samples"]) == 3


@pytest.mark.parametrize("stale", [False, True])
def test_calibration_rejects_rendered_pose_from_before_robot_motion(tmp_path, stale):
    from ora.environment.backends.robodojo.camera_capture import renderer_intrinsics

    parameters = {"cameraModel": "pinhole", "renderProductResolution": [640, 480],
                  "cameraProjection": np.diag([.9, 1.2, 1., 1.]).reshape(-1).tolist(),
                  "cameraAperture": [2., 2.], "cameraApertureOffset": [0., 0.],
                  "cameraViewTransform": np.eye(4).reshape(-1).tolist()}
    intrinsic = renderer_intrinsics(parameters, (640, 480))
    # The physical wrist has moved half a metre, while its render product
    # still reports the old optical transform. A static camera would miss this.
    physical = np.array([[.5 if stale else 0., 0., 0.]])
    camera = SimpleNamespace(get_intrinsics_matrix=lambda: intrinsic,
                             _prim_view=SimpleNamespace(get_world_poses=lambda **kw: (physical, [[1., 0., 0., 0.]])))
    view = SimpleNamespace(_render_product_path="/camera/wrist", camera_resolution=(640, 480),
                           _harness_camera_params=SimpleNamespace(get_data=lambda: parameters),
                           _harness_simulation_time=SimpleNamespace(get_data=lambda: {"simulationTime": 1.0}),
                           _harness_reference_time=SimpleNamespace(get_data=lambda: {"referenceTimeNumerator": 1,
                                                                                  "referenceTimeDenominator": 60}))
    session = RobodojoSession(config={}, output_root=tmp_path)
    session.environment = SimpleNamespace(
        camera_manager=SimpleNamespace(camera_names=[["wrist"]], cameras=[[camera]],
                                       get_camera_intrinsics=lambda *a, **kw: intrinsic),
        capture_manager=SimpleNamespace(tiled_cameras=[view]), use_fabric=True,
        sim=SimpleNamespace(scene=SimpleNamespace(env_origins=np.zeros((1, 3)))))
    if stale:
        with pytest.raises(RobodojoError, match="rendered camera pose differs"):
            session._calibration("wrist")
    else:
        session._calibration("wrist")
        assert session.calibration_readbacks["wrist"]["current_physical_pose_verified"] is True


# --- sparse motion v3 over the same fake SDK boundary ---------------------

@pytest.fixture(autouse=True)
def fake_sdk_candidate_boundary(monkeypatch):
    """Keep fake plant kinematics outside the real bounded candidate selector."""
    from ora.environment.backends.robodojo import ik, reachability
    from ora.environment.backends.robodojo.common import native_pose

    def candidates(planner, current, target, base):
        result = planner.manager.solve_ik(native_pose(target), 0, planner.robot)
        success = result["status"] == "Success"
        return {"joint_names": planner.robot.arm_joints_name,
                "solutions": [result.get("joint_value", current)], "success": [success],
                "feasible": [success], "position_tolerance_m": .001, "orientation_tolerance_rad": .02}

    def fk(session, arm, q):
        vector = list(q[3:6])
        return list(q[:3]) + vector + [math.sqrt(max(0.0, 1.0 - sum(x * x for x in vector)))]

    monkeypatch.setattr(ik, "sdk_candidates", candidates)
    monkeypatch.setattr(reachability, "fk_link6", fk)
    # The fake plant has no SDK collision spheres; the straight-path screen is
    # covered in test_robodojo_ik_continuity with planar and pinned X5 models.
    monkeypatch.setattr(ik, "self_collision_model",
                        lambda *a: SimpleNamespace(straight_path=lambda start, end: {"status": "clear"}))
    # Nor a second placed sphere model: the inter-arm screen is covered in
    # test_robodojo_inter_arm_clearance; executor refusals in ..._admission.
    monkeypatch.setattr(ik, "inter_arm_screen", lambda session: CLEAR_INTER_ARM)


#: A stand-in inter-arm screen that finds every lockstep path clear.
CLEAR_INTER_ARM = SimpleNamespace(straight_paths=lambda starts, ends: {
    "status": "clear", "min_clearance_m": 1.0, "required_clearance_m": 0.005, "start_clearance_m": 1.0,
    "limiting_links": {"left": "link6", "right": "link6"}, "limiting_fraction": 1.0,
    "moving_arms": sorted(arm for arm in starts if list(starts[arm]) != list(ends[arm]))})

SPARSE_SCHEMA = "ora.robodojo-sparse-motion-receipt.v1"


def sparse_session(tmp_path, **config):
    plant = Plant()
    instance = RobodojoSession(
        config={"workspace_bounds_m": {"x": [-1, 1], "y": [-1, 1], "z": [0, 2]},
                "action_codec": "motion_action_v3", **config},
        output_root=tmp_path / "native",
        environment_factory=lambda config: (
            plant, SimpleNamespace(close=lambda: None), _fake_source_manifest(tmp_path),
        ),
    )
    instance._calibration = lambda name: (np.array([[50, 0, 5], [0, 50, 4], [0, 0, 1]]), np.eye(4))
    instance.start()
    return instance


@pytest.fixture
def sparse(tmp_path):
    instance = sparse_session(tmp_path)
    try:
        yield instance
    finally:
        instance.close()


def pose(position, orientation=(0, 0, 0, 1)):
    return {"position_m": list(position), "orientation_xyzw": list(orientation)}


def v3(session, tmp_path, *, kind="move_pose", targets=None, interaction="free_space", plan_ref=None):
    observation, _ = session.observe(tmp_path / "observations")
    return v3_for(observation, session, kind=kind, targets=targets, interaction=interaction, plan_ref=plan_ref)


def v3_for(observation, session, *, kind="move_pose", targets=None, interaction="free_space", plan_ref=None):
    if plan_ref is not None:
        arguments = {"plan_ref": plan_ref}
    elif kind in {"move_pose", "move_free_space"}:
        arguments = {"targets": targets, "interactions": {arm: interaction for arm in targets}}
    else:
        arguments = {"targets": targets}
    return {"schema": "ora-motion-action.v3", "observation_id": observation["observation_id"],
            "action_id": f"action-{session.action_sequence}", "action": kind, "arguments": arguments,
            "bound_observation_sha256": "a" * 64}


def probe_request(session, tmp_path, position, *, policy="exact", arm="left", orientation=(0, 0, 0, 1)):
    observation, _ = session.observe(tmp_path / "observations")
    return observation, {
        "schema": "ora.motion-probe-request.v1", "observation_id": observation["observation_id"],
        "observation_sha256": "a" * 64, "variant_policy": policy,
        "queries": [{"query_id": "q0", "arm": arm, "pose": pose(position, orientation)}],
    }


def test_sparse_v3_requires_the_motion_codec(session, tmp_path):
    with pytest.raises(RobodojoError, match="requires the motion_action_v3 codec"):
        session.execute(v3(session, tmp_path, targets={"left": pose([0.02, 0, 0.8])}))
    with pytest.raises(RobodojoError, match="requires the motion_action_v3 codec"):
        session.probe_motion(probe_request(session, tmp_path, [0.02, 0, 0.8])[1])
    with pytest.raises(RobodojoError, match="action_codec"):
        RobodojoSession(config={"action_codec": "target_state_v4"}, output_root=tmp_path / "x")


def test_sparse_global_move_pose_reaches_and_holds_the_other_dimensions(sparse, tmp_path):
    before = deepcopy(sparse.environment.robot_manager.control_manager.prev_control[0])
    receipt = sparse.execute(v3(sparse, tmp_path, targets={"left": pose([0.02, 0.05, 0.85])}))["receipt"]
    assert receipt["schema"] == SPARSE_SCHEMA and receipt["action"] == "move_pose"
    assert receipt["execution_status"] == "reached" and receipt["motion_strategy"] == "global"
    assert receipt["executor"] == ["robodojo_sdk_ik_joint_interpolation"]
    assert receipt["failure_stage"] is None and receipt["failure_code"] is None
    assert receipt["native_actions"] == 1 and receipt["physics_steps"] == 10 and receipt["state_changed"]
    assert receipt["target_arms"] == ["left"] and receipt["synchronization"]["mode"] == "single_arm"
    left, right = receipt["arms"]["left"], receipt["arms"]["right"]
    assert left["targeted"] and left["interaction"] == "free_space"
    assert left["pose_error"]["translation_m"] == pytest.approx(0.0)
    assert left["plan_validation"]["parity"] == "non_parity_motion"
    assert left["segments"][0]["segment"] == "move" and left["segments"][0]["ik_status"] == "Success"
    assert right["targeted"] is False and right["inactive_arm_drift"]["translation_m"] == 0.0
    assert receipt["command_invariant"]["verified"] is True
    assert set(sparse.environment.commands[0]) == {"left_arm_joint_state"}
    after = sparse.environment.robot_manager.control_manager.prev_control[0]
    assert before["left_ee_joint_state"] == after["left_ee_joint_state"]
    assert before["right_arm_joint_state"] == after["right_arm_joint_state"]
    assert sparse.pose_targets["left"] == [0.02, 0.05, 0.85, 0, 0, 0, 1]
    assert sparse.last_safe_joints["left"] == [0, 0, 0.8, 0, 0, 0]
    assert sparse.world_mutation_sequence == 1 and sparse.world_mutation_actions == ["move_pose"]
    trace = json.loads((sparse.output_root / receipt["control_trace"]["path"]).read_text())
    assert trace["schema"] == "ora.robodojo-controller-trace.v1"
    assert [sample["segment"] for sample in trace["samples"]] == ["move"]
    journal = (sparse.output_root / trace["physics_journal"]["path"]).read_text().splitlines()
    assert len(journal) == 10 and json.loads(journal[0])["segment"] == "move"
    persisted = json.loads((sparse.output_root / "receipts/action-0000.json").read_text())
    assert persisted == receipt
    next_observation, _ = sparse.observe(tmp_path / "observations")
    assert next_observation["previous_execution_receipt"]["receipt"] == receipt


def test_sparse_free_space_lifts_translates_and_descends_within_one_budget(sparse, tmp_path):
    receipt = sparse.execute(v3(sparse, tmp_path, kind="move_free_space",
                                targets={"left": pose([0.10, 0.05, 0.80])}))["receipt"]
    assert receipt["execution_status"] == "reached" and receipt["motion_strategy"] == "global"
    segments = receipt["arms"]["left"]["segments"]
    assert [item["segment"] for item in segments] == ["lift", "translate", "descend"]
    assert segments[0]["target_pose_xyz_xyzw"][:3] == pytest.approx([0.0, 0.0, 0.85])
    assert segments[1]["target_pose_xyz_xyzw"][:3] == pytest.approx([0.10, 0.05, 0.85])
    assert segments[2]["target_pose_xyz_xyzw"][:3] == pytest.approx([0.10, 0.05, 0.80])
    assert all(item["reached"] and item["native_actions"] == 1 for item in segments)
    assert receipt["native_actions"] == 3 and receipt["physics_steps"] == 30
    assert receipt["native_action_budget"] == 20
    commands = [command["left_arm_joint_state"]["position"][:3] for command in sparse.environment.commands]
    assert commands == [pytest.approx([0, 0, 0.85]), pytest.approx([0.1, 0.05, 0.85]), pytest.approx([0.1, 0.05, 0.8])]
    # Degenerate segments are skipped: a purely vertical goal has no translate leg.
    receipt = sparse.execute(v3(sparse, tmp_path, kind="move_free_space",
                                targets={"left": pose([0.10, 0.05, 0.85])}))["receipt"]
    assert [item["segment"] for item in receipt["arms"]["left"]["segments"]] == ["lift", "descend"]
    assert receipt["arms"]["left"]["segments"][0]["target_pose_xyz_xyzw"][2] == pytest.approx(0.90)


def test_sparse_precision_routes_local_only_within_the_servo_envelope(sparse, tmp_path):
    local = sparse.execute(v3(sparse, tmp_path, targets={"left": pose([0, 0, 0.806])},
                              interaction="precision"))["receipt"]
    assert local["execution_status"] == "reached" and local["motion_strategy"] == "local"
    assert local["executor"] == ["robodojo_sdk_ik_local_cartesian_step"]
    assert local["arms"]["left"]["segments"][0]["segment"] == "local"
    assert local["arms"]["left"]["segments"][0]["servo_frame"] == "world"
    assert local["arms"]["left"]["plan_validation"] is None
    far = sparse.execute(v3(sparse, tmp_path, targets={"left": pose([0.2, 0, 0.806])},
                            interaction="precision"))["receipt"]
    assert far["motion_strategy"] == "global"
    contact = sparse.execute(v3(sparse, tmp_path, targets={"left": pose([0.2, 0, 0.798])},
                                interaction="contact_approach"))["receipt"]
    assert contact["execution_status"] == "reached" and contact["motion_strategy"] == "local"
    # A boundary-exact 15-degree request routes local; one above the contract goes global.
    from ora.environment.backends.robodojo.sparse_motion import within_local_envelope

    assert within_local_envelope(0.05, math.radians(15.0))
    assert not within_local_envelope(0.05, math.radians(15.0) + 1e-10)


def test_sparse_large_contact_approach_is_refused_without_writing(sparse, tmp_path):
    receipt = sparse.execute(v3(sparse, tmp_path, targets={"left": pose([0, 0, 0.6])},
                                interaction="contact_approach"))["receipt"]
    assert receipt["execution_status"] == "plan_rejected"
    assert receipt["failure_stage"] == "route"
    assert receipt["failure_code"] == "route_contact_approach_exceeds_local_bound"
    assert receipt["executor"] is None and receipt["motion_strategy"] is None
    assert receipt["state_changed"] is False and receipt["native_actions"] == 0
    assert sparse.environment.commands == [] and sparse.world_mutation_sequence == 0
    assert receipt["arms"]["left"]["pose_error"]["translation_m"] == pytest.approx(0.2)
    trace = json.loads((sparse.output_root / receipt["control_trace"]["path"]).read_text())
    assert trace["samples"] == []
    mixed = sparse.execute({**v3(sparse, tmp_path, targets={"left": pose([0, 0, 0.81]), "right": pose([0, 0, 0.81])}),
                            "arguments": {"targets": {"left": pose([0, 0, 0.81]), "right": pose([0, 0, 0.81])},
                                          "interactions": {"left": "free_space", "right": "precision"}}})["receipt"]
    assert mixed["execution_status"] == "plan_rejected" and mixed["failure_code"] == "mixed_interactions"
    outside = sparse.execute(v3(sparse, tmp_path, targets={"left": pose([0, 0, 3.0])}))["receipt"]
    assert outside["execution_status"] == "plan_rejected" and outside["failure_code"] == "workspace_target_z"


def test_sparse_ik_failure_is_planner_failed_only_before_any_execution(sparse, tmp_path):
    sparse.environment.robot_manager.fail_ik = True
    receipt = sparse.execute(v3(sparse, tmp_path, targets={"left": pose([0.1, 0, 0.8])}))["receipt"]
    assert receipt["execution_status"] == "planner_failed" and receipt["failure_stage"] == "planner"
    assert receipt["failure_code"] == "ik_failed:move" and receipt["executor"] is None
    assert receipt["native_actions"] == 0 and sparse.environment.commands == []
    assert receipt["arms"]["left"]["segments"][0]["joint_target"] is None
    sparse.environment.robot_manager.fail_ik = False
    sparse.environment.robot_manager.unreachable = lambda target: target[0] > 0.05
    partial = sparse.execute(v3(sparse, tmp_path, kind="move_free_space",
                                targets={"left": pose([0.10, 0.05, 0.80])}))["receipt"]
    assert partial["execution_status"] == "not_reached" and partial["failure_code"] == "ik_failed:translate"
    assert partial["native_actions"] == 1 and partial["executor"] == ["robodojo_sdk_ik_joint_interpolation"]
    assert [item["segment"] for item in partial["arms"]["left"]["segments"]] == ["lift", "translate"]


def test_sparse_budget_exhaustion_and_branch_jump_fail_closed(sparse, tmp_path):
    sparse.environment.step_fraction = 0.05
    receipt = sparse.execute(v3(sparse, tmp_path, targets={"left": pose([0.3, 0, 0.8])}))["receipt"]
    assert receipt["execution_status"] == "not_reached"
    assert receipt["failure_stage"] == "execution"
    assert receipt["failure_code"] == "native_action_budget_exhausted"
    assert receipt["native_actions"] == 20 and receipt["physics_steps"] == 200
    sparse.environment.step_fraction = 1.0
    sparse.local_joint_jump_limit_rad = 0.001
    current = sparse._proprioception()["end_effectors"]["left"]["pose_xyz_xyzw"]
    jump = sparse.execute(v3(sparse, tmp_path, targets={"left": pose([current[0] + 0.01, 0, 0.8])},
                             interaction="precision"))["receipt"]
    assert jump["execution_status"] == "plan_rejected" and jump["failure_code"] == "local_ik_branch_jump"
    assert jump["native_actions"] == 0
    rejected = jump["arms"]["left"]["segments"][0]
    assert rejected["joint_target"] is not None and rejected["native_actions"] == 0
    assert rejected["ik_selection"]["joint_jump_limit_rad"] == .001
    assert rejected["ik_selection"]["limiting_joint"] == "joint1"
    assert jump["ik_attempts"][0]["candidates"][0]["rejection"] == "joint_jump"



def test_sparse_global_branch_refusal_preserves_selected_endpoint(sparse, tmp_path):
    sparse.global_joint_jump_limit_rad = .001
    receipt = sparse.execute(v3(sparse, tmp_path, targets={"left": pose([.2, 0, .8])}))["receipt"]
    assert receipt["execution_status"] == "plan_rejected"
    assert receipt["failure_code"].startswith("ik_branch_jump:")
    assert receipt["native_actions"] == 0
    segment = receipt["arms"]["left"]["segments"][0]
    attempt = receipt["ik_attempts"][0]
    assert segment["joint_target"] == attempt["joint_target"]
    assert segment["joint_target"] is not None
    assert segment["ik_selection"]["outcome"] == "joint_jump"
    assert attempt["candidates"][0]["rejection"] == "joint_jump"
    assert receipt["arms"]["left"]["planning_diagnostics"]["candidate_counts"]["continuous"] == 0

def test_sparse_gripper_targets_hold_pose_targets(sparse, tmp_path):
    before_targets = deepcopy(sparse.pose_targets)
    receipt = sparse.execute(v3(sparse, tmp_path, kind="set_gripper", targets={"left": "close"}))["receipt"]
    assert receipt["execution_status"] == "reached" and receipt["executor"] == ["robodojo_sdk_gripper_actuator"]
    assert receipt["motion_strategy"] is None and receipt["arms"]["left"]["gripper_error"] == pytest.approx(0.0)
    assert receipt["arms"]["left"]["pose_error"] is None
    assert receipt["after"]["left"]["gripper"] == pytest.approx(0.0)
    assert receipt["command_invariant"] == {
        "schema": "ora.command-invariant.v2", "controlled_dimension": "gripper",
        "held_dimension": "eef_pose_target", "verified": True,
        "applied_targets": "per_physics_step_sdk_joint_position_buffer",
    }
    assert set(sparse.environment.commands[0]) == {"left_ee_joint_state"}
    assert sparse.pose_targets == before_targets and sparse.gripper_directives["left"] == 0.0
    assert sparse.world_mutation_actions == ["set_gripper"]
    both = sparse.execute(v3(sparse, tmp_path, kind="set_gripper", targets={"left": "open", "right": "close"}))["receipt"]
    assert both["execution_status"] == "reached" and both["target_arms"] == ["left", "right"]
    assert set(sparse.environment.commands[-1]) == {"left_ee_joint_state", "right_ee_joint_state"}


def test_sparse_recovery_returns_to_the_last_safe_then_home_configuration(sparse, tmp_path):
    untouched = sparse.execute(v3(sparse, tmp_path, kind="recover_motion", targets={"left": "backtrack_last_safe"}))["receipt"]
    assert untouched["execution_status"] == "reached" and untouched["executor"] == ["robodojo_sdk_joint_interpolation"]
    sparse.execute(v3(sparse, tmp_path, targets={"left": pose([0.05, 0.05, 0.85])}))
    sparse.execute(v3(sparse, tmp_path, targets={"left": pose([0.10, 0.10, 0.90])}))
    back = sparse.execute(v3(sparse, tmp_path, kind="recover_motion", targets={"left": "backtrack_last_safe"}))["receipt"]
    assert back["execution_status"] == "reached" and back["motion_strategy"] == "global"
    assert back["after"]["left"]["pose_xyz_xyzw"][:3] == pytest.approx([0.05, 0.05, 0.85])
    assert back["arms"]["left"]["segments"][0]["recovery"] == "backtrack_last_safe"
    # Recovery itself never rewrites the backtrack anchor.
    assert sparse.last_safe_poses["left"][:3] == pytest.approx([0.05, 0.05, 0.85])
    home = sparse.execute(v3(sparse, tmp_path, kind="recover_motion", targets={"left": "home"}))["receipt"]
    assert home["execution_status"] == "reached"
    assert home["after"]["left"]["pose_xyz_xyzw"][:3] == pytest.approx([0.0, 0.0, 0.8])
    sparse.home_joints.pop("right")
    missing = sparse.execute(v3(sparse, tmp_path, kind="recover_motion", targets={"right": "home"}))["receipt"]
    assert missing["execution_status"] == "plan_rejected" and missing["failure_code"] == "recovery_state_unavailable"


def test_sparse_dual_arm_targets_execute_in_one_native_tick(sparse, tmp_path):
    receipt = sparse.execute(v3(sparse, tmp_path, targets={"left": pose([0, 0, 0.84]), "right": pose([0, 0, 0.86])}))["receipt"]
    assert receipt["execution_status"] == "reached" and receipt["target_arms"] == ["left", "right"]
    assert receipt["synchronization"] == {"mode": "dual_arm_common_native_tick", "requested": True, "executed": True}
    assert set(sparse.environment.commands[0]) == {"left_arm_joint_state", "right_arm_joint_state"}
    assert receipt["native_actions"] == 1
    assert receipt["arms"]["right"]["pose_error"]["translation_m"] == pytest.approx(0.0)
    assert sparse.gripper_directives == {"left": 1.0, "right": 1.0}


def test_sparse_probe_and_plan_reference_lifecycle(sparse, tmp_path):
    clock = sparse.environment.sim.sim.current_time_step_index
    observation, request = probe_request(sparse, tmp_path, [0.06, 0, 0.8])
    probed = sparse.probe_motion(request)
    assert probed["schema"] == "ora.motion-probe-result.v1"
    assert probed["world_state_changed"] is False and probed["robot_decisions_consumed"] == 0
    row = probed["results"][0]
    assert row["reachable"] and row["selected_variant_id"] == "variant-00-exact-+0deg"
    assert row["selected_target_pose_xyzw"] == [0.06, 0, 0.8, 0, 0, 0, 1]
    assert row["variants"][0]["planner"]["parity"] == "non_parity_motion"
    plan_ref = row["plan_ref"]
    assert plan_ref.startswith("plan-") and len(plan_ref) == 37
    assert sparse.environment.sim.sim.current_time_step_index == clock
    stored = sparse.plan_references[plan_ref]
    manifest = json.loads((sparse.output_root / stored["artifact"]["path"]).read_text())
    assert manifest["backend_id"] == "robodojo" and manifest["embodiment_id"] == "arx_x5"
    assert manifest["validation"]["parity"] == "non_parity_motion"
    receipt = sparse.execute(v3_for(observation, sparse, plan_ref=plan_ref))["receipt"]
    assert receipt["execution_status"] == "reached" and receipt["arms"]["left"]["plan_ref"] == plan_ref
    assert receipt["arms"]["left"]["segments"][0]["source"] == "motion_probe_plan_ref"
    assert receipt["after"]["left"]["pose_xyz_xyzw"][:3] == pytest.approx([0.06, 0, 0.8])
    # A gripper-only interval keeps the probed plan consumable.
    observation, request = probe_request(sparse, tmp_path, [0.02, 0, 0.8])
    plan_ref = sparse.probe_motion(request)["results"][0]["plan_ref"]
    sparse.execute(v3_for(observation, sparse, kind="set_gripper", targets={"left": "close"}))
    survived = sparse.execute(v3(sparse, tmp_path, plan_ref=plan_ref))["receipt"]
    assert survived["execution_status"] == "reached"
    # Any other world write expires it, and an unknown reference never executes.
    observation, request = probe_request(sparse, tmp_path, [0.04, 0, 0.8])
    plan_ref = sparse.probe_motion(request)["results"][0]["plan_ref"]
    sparse.execute(v3_for(observation, sparse, targets={"left": pose([0.03, 0, 0.8])}, interaction="precision"))
    with pytest.raises(RobodojoError, match="stale"):
        sparse.execute(v3(sparse, tmp_path, plan_ref=plan_ref))
    with pytest.raises(RobodojoError, match="unknown"):
        sparse.execute(v3(sparse, tmp_path, plan_ref="plan-" + "0" * 32))
    # Jaw-flip equivalents rotate 180 degrees about the link6 approach axis (+X).
    _observation, request = probe_request(sparse, tmp_path, [0.05, 0, 0.8], policy="parallel_jaw_equivalents")
    variants = sparse.probe_motion(request)["results"][0]["variants"]
    assert [item["variant_id"] for item in variants] == ["variant-00-exact-+0deg", "variant-01-jaw-flip-180deg"]
    assert variants[1]["target_pose_xyzw"][3:] == pytest.approx([1, 0, 0, 0])
    sparse.environment.robot_manager.fail_ik = True
    _observation, request = probe_request(sparse, tmp_path, [0.05, 0, 0.8])
    unreachable = sparse.probe_motion(request)["results"][0]
    assert unreachable["reachable"] is False and unreachable["plan_ref"] is None
    assert unreachable["variants"][0]["failure"]["code"] == "Fail"
    sparse.environment.robot_manager.fail_ik = False
    _observation, request = probe_request(sparse, tmp_path, [0.05, 0, 3.0])
    with pytest.raises(RobodojoError, match="outside the action workspace"):
        sparse.probe_motion(request)
    with pytest.raises(RobodojoError, match="drifted"):
        sparse.probe_motion({**request, "extra": True})
    _observation, request = probe_request(sparse, tmp_path, [0.05, 0, 0.8])
    original = sparse.environment.robot_manager.solve_ik

    def leaking(target_pose, env_idx, robot):
        sparse.environment.sim.sim.current_time_step_index += 1
        return original(target_pose, env_idx, robot)

    sparse.environment.robot_manager.solve_ik = leaking
    with pytest.raises(RobodojoError, match="probe advanced physics"):
        sparse.probe_motion(request)
    assert sparse.is_terminal and sparse.abort_reason == "motion_probe_advanced_physics"


def test_sparse_official_terminal_during_execution_snapshots_the_verdict(sparse, tmp_path):
    sparse.environment.terminal_at = 1
    receipt = sparse.execute(v3(sparse, tmp_path, kind="move_free_space",
                                targets={"left": pose([0.10, 0.05, 0.80])}))["receipt"]
    assert receipt["native_actions"] == 1 and receipt["terminal_after_action"] is True
    assert receipt["execution_status"] == "not_reached"
    assert receipt["failure_code"] == "episode_terminal_during_execution"
    assert sparse.persist_official_verdict()["official_success"] is True
    with pytest.raises(RobodojoError, match="cannot accept"):
        sparse.execute(v3(sparse, tmp_path, targets={"left": pose([0.1, 0, 0.8])}))
    assert sparse.persist_official_verdict()["official_success"] is True


def test_restart_round_isolates_plans_targets_and_lifecycle(sparse, tmp_path):
    sparse.execute(v3(sparse, tmp_path, targets={"left": pose([0.05, 0.05, 0.85])}))
    sparse.execute(v3(sparse, tmp_path, kind="set_gripper", targets={"left": "close"}))
    _observation, request = probe_request(sparse, tmp_path, [0.06, 0, 0.85])
    plan_ref = sparse.probe_motion(request)["results"][0]["plan_ref"]
    sparse.abort("operator_stop")
    assert sparse.is_terminal
    with pytest.raises(RobodojoError, match="advance by exactly one"):
        sparse.restart_round(2)
    with pytest.raises(RobodojoError, match="seed 0"):
        sparse.restart_round(1, seed=5)
    sparse.persist_official_verdict()
    with pytest.raises(RobodojoError, match="archived"):
        sparse.restart_round(1)
    (sparse.output_root / "official-verdict.json").unlink()
    reset = sparse.restart_round(1, seed=0)
    assert reset["schema"] == "ora.robodojo-round-reset.v1"
    assert reset["round_index"] == 1 and reset["seed"] == 0 and reset["reward_registered"] is True
    assert reset["artifact_sequences_preserved"] == {
        "observation_sequence": 3, "action_sequence": 2, "motion_probe_sequence": 1,
    }
    assert reset["public_environment_context"]["environment_id"] == "robodojo"
    assert sparse.environment.harness_round_index == 1 and sparse.environment.resets == 2
    assert sparse.round_index == 1 and not sparse.is_terminal and sparse.abort_reason is None
    assert sparse.plan_references == {} and sparse.world_mutation_sequence == 0
    assert sparse.pose_targets["left"][:3] == [0, 0, 0.8] and sparse.gripper_directives["left"] == 1.0
    assert sparse.last_safe_joints["left"] == sparse.home_joints["left"] == [0, 0, 0.8, 0, 0, 0]
    assert sparse.sim_steps == 0 and sparse.last_receipt is None
    assert json.loads((sparse.output_root / "round-resets/round-0001.json").read_text()) == reset
    observation, _ = sparse.observe(tmp_path / "observations")
    assert observation["round_index"] == 1 and observation["sequence"] == 3
    assert observation["previous_execution_receipt"] is None
    with pytest.raises(RobodojoError, match="unknown"):
        sparse.execute({**v3(sparse, tmp_path, plan_ref=plan_ref)})
    receipt = sparse.execute(v3(sparse, tmp_path, targets={"left": pose([0.02, 0, 0.8])}))["receipt"]
    assert receipt["round_index"] == 1 and receipt["action_id"] == "action-2"
    assert (sparse.output_root / "receipts/action-0002.json").exists()
    assert sparse.persist_official_verdict()["round_index"] == 1
    with pytest.raises(RobodojoError, match="integer"):
        sparse.restart_round(True)


def _second_round(sparse, tmp_path):
    """Drive one round to a verdict and reset into round 1."""

    sparse.execute(v3(sparse, tmp_path, targets={"left": pose([0.05, 0.05, 0.85])}))
    sparse.abort("operator_stop")
    sparse.persist_official_verdict()
    (sparse.output_root / "official-verdict.json").unlink()
    return sparse.restart_round(1, seed=0)


def test_a_second_round_starts_from_a_measured_restored_world(sparse, tmp_path):
    """The round-reset record proves the restore instead of asserting it.

    Every field below is read back from the plant or from this session's own
    carried state after the reset: the official evaluator was rebuilt, its
    counters and failure latch are back at their reset values, the layout the
    session was started against still hashes the same, both arms returned to
    the pose the session's first reset measured, and no per-episode channel
    kept a value.
    """

    reset = _second_round(sparse, tmp_path)
    isolation = reset["round_isolation"]

    assert isolation["schema"] == "ora.robodojo-round-isolation.v1"
    assert isolation["passed"] is True and isolation["failures"] == []
    assert isolation["checks"] == {
        "arm_home_restored": True,
        "carried_channels_cleared": True,
        "episode_counters_zeroed": True,
        "failure_latch_cleared": True,
        "layout_identity_unchanged": True,
        "official_lifecycle_rebound": True,
        "official_terminal_cleared": True,
    }
    measured = isolation["measured"]
    # reward_registered is measured on the plant, not written as a literal:
    # the round-reset record republishes exactly what the audit read.
    assert measured["reward_registered"] is True
    assert reset["reward_registered"] is measured["reward_registered"]
    assert measured["step_lim"] == measured["native_action_limit"] == 1050
    assert measured["take_action_cnt"] == 0 and measured["end_flag"] is False
    assert measured["failure_latch_clear"] is True
    assert measured["stale_channels"] == []
    assert measured["layout_sha256"] == measured["layout_sha256_at_start"]
    assert measured["max_home_position_error_m"] == 0.0
    assert measured["physics_steps_before_policy"] == reset[
        "physics_steps_before_policy"
    ]
    assert sorted(measured["cleared_channels"]) == sorted(
        session_module.PER_ROUND_CLEARED_CHANNELS
    )
    # And the world really is back: the same record is on disk for the join.
    assert json.loads(
        (sparse.output_root / "round-resets/round-0001.json").read_text()
    ) == reset


@pytest.mark.parametrize(
    "channel", sorted(session_module.PER_ROUND_CLEARED_CHANNELS)
)
def test_a_stale_value_in_any_carried_channel_refuses_the_round(
    sparse, tmp_path, monkeypatch, channel,
):
    """Every declared per-round channel is proved empty, one by one.

    The failure is an infrastructure fault -- the session records one and the
    round keeps a null official verdict -- not a task outcome.
    """

    stale = {
        "action_sequences": [{"parent": "stale"}],
        "abort_reason": "carried_stop",
        "boundary_reason": "carried_boundary",
        "fault_reason": "carried_fault",
        "last_observation_id": "robodojo:observation-0000",
        "last_receipt": {"action_id": "action-0"},
        "official_terminal": {"official_success": True},
        "plan_references": {"plan-0": {"stale": True}},
        "sim_steps": 40,
        "world_mutation_actions": ["move_free_space"],
        "world_mutation_sequence": 3,
    }[channel]
    original = type(sparse)._bind_reset_state

    def leaking(self):
        state = original(self)
        setattr(self, channel, stale)
        return state

    monkeypatch.setattr(type(sparse), "_bind_reset_state", leaking)

    with pytest.raises(RobodojoError, match="not restore an isolated world"):
        _second_round(sparse, tmp_path)

    record = json.loads(
        (sparse.output_root / "round-resets/round-0001.json").read_text()
    )
    isolation = record["round_isolation"]
    assert isolation["passed"] is False
    assert isolation["measured"]["stale_channels"] == [channel]
    assert any(channel in failure for failure in isolation["failures"])
    # An infrastructure fault is recorded, so the round keeps a null official
    # verdict. When the leaked channel is fault_reason itself, the session
    # keeps the first reason rather than overwriting it.
    assert sparse.fault_reason == (
        stale if channel == "fault_reason"
        else "robodojo_round_reset_isolation_failed"
    )


@pytest.mark.parametrize(
    "field, value, failed_check",
    [
        ("take_action_cnt", [7], "episode_counters_zeroed"),
        ("end_flag", [True], "episode_counters_zeroed"),
        ("success", [False], "failure_latch_cleared"),
        ("_official_terminal", {"official_success": False},
         "official_terminal_cleared"),
    ],
)
def test_an_unrestored_official_lifecycle_refuses_the_round(
    sparse, tmp_path, monkeypatch, field, value, failed_check,
):
    """A contradictory round result never reaches the new round's policy.

    This is the defect class the guarantee exists for: an evaluator latch or
    episode counter that survives the reset would make the freshly reset world
    answer with the previous round's outcome.
    """

    plant = sparse.environment
    original = type(plant).reset

    def leaking(self, seed):
        original(self, seed)
        setattr(self, field, value)

    monkeypatch.setattr(type(plant), "reset", leaking)

    with pytest.raises(RobodojoError, match="not restore an isolated world"):
        _second_round(sparse, tmp_path)

    record = json.loads(
        (sparse.output_root / "round-resets/round-0001.json").read_text()
    )
    assert record["round_isolation"]["checks"][failed_check] is False


def test_a_changed_layout_file_refuses_the_round(sparse, tmp_path):
    """The world is only restored if it is restored from the *same* layout."""

    layout = Path(sparse.source["layout_file"]["path"])
    layout.write_text('{"objects": ["moved"]}', encoding="utf-8")

    with pytest.raises(RobodojoError, match="layout_identity_unchanged"):
        _second_round(sparse, tmp_path)

    record = json.loads(
        (sparse.output_root / "round-resets/round-0001.json").read_text()
    )
    measured = record["round_isolation"]["measured"]
    assert measured["layout_sha256"] != measured["layout_sha256_at_start"]


def test_an_arm_that_does_not_return_home_refuses_the_round(
    sparse, tmp_path, monkeypatch,
):
    """A plant that resets its evaluator but not its robot is not restored.

    The Harness writes the session-home joints before the upstream reset; a
    plant on which neither that write nor the reset moves the arm still fails
    round isolation, so the guard is unchanged.
    """

    manager = sparse.environment.robot_manager
    monkeypatch.setattr(
        type(manager), "reset_home", lambda self: None, raising=True,
    )
    monkeypatch.setattr(
        type(manager), "write_joint_state", lambda self, *a, **k: None, raising=True,
    )
    sparse.execute(v3(sparse, tmp_path, targets={"left": pose([0.30, 0.20, 1.10])}))
    sparse.abort("operator_stop")
    sparse.persist_official_verdict()
    (sparse.output_root / "official-verdict.json").unlink()

    with pytest.raises(RobodojoError, match="arm_home_restored"):
        sparse.restart_round(1, seed=0)

    record = json.loads(
        (sparse.output_root / "round-resets/round-0001.json").read_text()
    )
    measured = record["round_isolation"]["measured"]
    assert measured["max_home_position_error_m"] > record["round_isolation"][
        "tolerances"
    ]["home_position_m"]


def test_motion_backend_adapter_conforms_to_the_shared_transaction(sparse, tmp_path):
    from ora.environment.controllers.motion_transaction import (
        MotionTransactionError, execute_measured_convergence, validate_motion_backend_adapter,
    )

    adapter = sparse.motion_backend_adapter()
    capabilities = validate_motion_backend_adapter(adapter)
    assert capabilities.backend_id == "robodojo" and capabilities.embodiment_id == "arx_x5"
    assert capabilities.arms == ("left", "right") and capabilities.controlled_point == "link6"
    assert capabilities.global_joint_planning and capabilities.local_cartesian
    assert capabilities.synchronized_groups == (("left", "right"),)
    assert adapter.global_planning_parity == "sdk_ik_joint_interpolation_not_curobo_planning"
    start = adapter.capture_action_start()
    assert start["joints"]["left"] == [0, 0, 0.8, 0, 0, 0]
    target = adapter.compile_public_target("left", [0.1, 0, 0.8, 0, 0, 0, 1])
    assert target == [0.1, 0, 0.8, 1, 0, 0, 0]
    plan = adapter.plan_global("left", target, start)
    assert plan["reachable"] and plan["parity"] == "sdk_ik_joint_interpolation_not_curobo_planning"
    assert plan["trajectory"]["positions"] == [[0.1, 0, 0.8, 0, 0, 0]]
    batch = adapter.plan_batch_global("left", [target, target], start)
    assert len(batch["results"]) == 2
    assert adapter.joint_limits("left") == [[-3.14, 3.14]] * 6
    assert adapter.validate_collision({"left": plan}, start)["status"] == "not_validated"
    with pytest.raises(MotionTransactionError, match="no forward kinematics"):
        adapter.fk_control_pose("left", [0] * 6)
    with pytest.raises(MotionTransactionError, match="no arm"):
        adapter.read_joint_positions("middle")
    local = adapter.compute_local_joint_target("right", [0] * 6, [0, 0, 0.8, 0, 0, 0, 1], [0, 0.01, 0.8, 0, 0, 0, 1], {})
    assert local == {"position": [0, 0.01, 0.8, 0, 0, 0], "velocity": [0.0] * 6}
    sparse.environment.robot_manager.fail_ik = True
    with pytest.raises(MotionTransactionError, match="local IK failed"):
        adapter.compute_local_joint_target("right", [0] * 6, [0, 0, 0.8, 0, 0, 0, 1], [0, 0.01, 0.8, 0, 0, 0, 1], {})
    sparse.environment.robot_manager.fail_ik = False
    report, arrays = execute_measured_convergence(
        adapter=adapter, paths={"left": {"position": [[0.1, 0, 0.8, 0, 0, 0]], "velocity": [[0.0] * 6]}},
        all_arms=("left", "right"), mode="global",
        profile={"joint_convergence_tolerance_rad": 0.01, "maximum_physics_steps_per_waypoint": 4,
                 "maximum_total_physics_steps": 40, "maximum_tcp_step_m": 0.5, "workspace_margin_m": 0.0},
        workspace_bounds_m=sparse.config["workspace_bounds_m"],
    )
    assert report["completed"] and report["backend"] == "robodojo" and report["sampled_steps"] == 1
    assert arrays["measured_left_tcp__pose_xyzw"][-1][:3] == pytest.approx([0.1, 0, 0.8])
    assert sparse.sim_steps == 10 and sparse.environment.take_action_cnt == [1]


def test_sparse_receipts_project_into_the_unified_envelope(sparse, tmp_path):
    from ora.agent.native.receipt_projection import (
        ReceiptProjectionContext, resolve_receipt_effect_projector,
    )

    receipt = sparse.execute(v3(sparse, tmp_path, targets={"left": pose([0.02, 0, 0.8])}))["receipt"]
    projector = resolve_receipt_effect_projector(receipt["schema"])
    assert projector is not None
    projected = projector(receipt, ReceiptProjectionContext(last_action_id=receipt["action_id"], last_action_public_name="move_pose"))
    assert projected["status"] == "reached" and projected["executor"] == ["robodojo_sdk_ik_joint_interpolation"]
    assert projected["motion_strategy"] == "global" and projected["detail"]["target_arms"] == ["left"]
    assert "control_trace" not in json.dumps(projected)


def test_sparse_global_route_refuses_aliased_ik_branches_before_writing(sparse, tmp_path):
    manager = sparse.environment.robot_manager
    original = manager.solve_ik

    def aliased(target_pose, env_idx, robot):
        result = original(target_pose, env_idx, robot)
        if result["status"] == "Success":
            result["joint_value"][5] += 2 * math.pi
        return result

    manager.solve_ik = aliased
    receipt = sparse.execute(v3(sparse, tmp_path, targets={"left": pose([0.02, 0, 0.8])}))["receipt"]
    # The aliased coordinate is outside the actual finite joint limits, so it
    # is now rejected before jump admission instead of treating it as valid IK.
    assert receipt["execution_status"] == "planner_failed" and receipt["failure_stage"] == "planner"
    assert receipt["failure_code"] == "ik_failed:move" and receipt["native_actions"] == 0
    assert receipt["ik_attempts"][0]["candidates"][0]["rejection"] == "joint_limits"
    assert receipt["executor"] is None and sparse.environment.commands == []
    assert receipt["arms"]["left"]["pose_error"]["translation_m"] == pytest.approx(0.02)
    _observation, request = probe_request(sparse, tmp_path, [0.02, 0, 0.8])
    row = sparse.probe_motion(request)["results"][0]
    assert row["reachable"] is False and row["variants"][0]["failure"]["code"] == "Fail"
    manager.solve_ik = original
    with pytest.raises(RobodojoError, match="supported ranges"):
        RobodojoSession(config={"action_codec": "motion_action_v3", "global_joint_jump_limit_rad": 0.5},
                        output_root=tmp_path / "bad")


def test_sparse_gripper_close_converges_to_the_measured_jaw_floor(sparse, tmp_path):
    manager = sparse.environment.robot_manager
    for articulation in manager.robot_key:
        articulation.data.joint_pos_limits[0][6] = [0.0, 0.044]
        articulation.data.joint_pos_limits[0][7] = [0.0, 0.044]
    receipt = sparse.execute(v3(sparse, tmp_path, kind="set_gripper", targets={"left": "close"}))["receipt"]
    assert receipt["execution_status"] == "reached" and receipt["native_actions"] == 1
    assert receipt["gripper_floors"]["left"] == pytest.approx(0.01 / 0.054)
    assert receipt["gripper_targets_effective"]["left"] == pytest.approx(0.01 / 0.054)
    assert receipt["gripper_directives"]["left"] == 0.0
    assert receipt["after"]["left"]["gripper"] == pytest.approx(0.01 / 0.054)
    assert receipt["arms"]["left"]["gripper_error"] == pytest.approx(0.0)
    assert manager.control_manager.prev_control[0]["left_ee_joint_state"]["position"] == [-0.01, -0.01]


def test_x5_command_target_reads_the_command_scale_not_a_finger_measurement():
    # Source identities: the command scale is robot_config.yml's
    # gripper_scale [-0.01, 0.044]; X5A.urdf limits joint7/joint8 to
    # [0, 0.044], 10 mm above the command origin, so a fully closed jaw
    # floors near 0.19 on the command scale rather than at 0.0.
    assert CLOSED_COMMAND_FLOOR == pytest.approx(0.01 / 0.054, abs=1e-6)
    assert gripper_command_target(1.0) == "open"
    assert gripper_command_target(0.0) == "close"
    # A future path publishing the executed floored command still reads close.
    assert gripper_command_target(CLOSED_COMMAND_FLOOR) == "close"
    assert gripper_command_target(0.5) == "unknown"
    assert gripper_command_target(None) == "unknown"
    assert gripper_command_target(True) == "unknown"
    assert gripper_command_target(float("nan")) == "unknown"


def test_proprioception_publishes_the_commanded_gripper_category_per_arm(sparse, tmp_path):
    """The session states each arm's directive; no consumer re-derives one."""
    manager = sparse.environment.robot_manager
    for articulation in manager.robot_key:
        articulation.data.joint_pos_limits[0][6] = [0.0, 0.044]
        articulation.data.joint_pos_limits[0][7] = [0.0, 0.044]
    payload, _paths = sparse.observe(tmp_path / "observations")
    effectors = payload["proprioception"]["end_effectors"]
    assert set(effectors) == {"left", "right"}
    for arm in ("left", "right"):
        assert effectors[arm]["gripper_command_target"] == "open"

    sparse.execute(v3(sparse, tmp_path, kind="set_gripper", targets={"left": "close"}))
    effectors = sparse.observe(tmp_path / "observations")[0][
        "proprioception"]["end_effectors"]
    assert effectors["left"]["gripper_command_target"] == "close"
    assert effectors["right"]["gripper_command_target"] == "open"
    # The jaw is empty and fully closed, so the measurement rests on the
    # articulation's own floor while the written command stays at 0.0. The
    # published category follows the command, and a floor reading is published
    # as a measurement only: the observation still asserts no contact evidence.
    assert effectors["left"]["gripper"] == pytest.approx(CLOSED_COMMAND_FLOOR)
    assert effectors["left"]["gripper_command"] == 0.0
    assert "contact" in payload["forbidden_modalities_absent"]
    assert not [
        key for key in effectors["left"]
        if "contact" in key or "grasp" in key or "retention" in key
    ]

    sparse.execute(v3(sparse, tmp_path, kind="set_gripper", targets={"left": "open"}))
    # Pin the fingers at the closed floor an obstructed open would leave them
    # at: the category is the directive, so it must read open regardless.
    manager.fingers["left_arm"] = [0.0, 0.0]
    effectors = sparse.observe(tmp_path / "observations")[0][
        "proprioception"]["end_effectors"]
    assert effectors["left"]["gripper"] == pytest.approx(CLOSED_COMMAND_FLOOR)
    assert effectors["left"]["gripper_command_target"] == "open"


def test_runtime_action_v3_set_gripper_publishes_the_same_category(session, tmp_path):
    receipt = session.execute(
        action(session, tmp_path, kind="set_gripper", value=0.0),
    )["receipt"]
    assert receipt["after"]["left"]["gripper_command_target"] == "close"
    assert receipt["after"]["right"]["gripper_command_target"] == "open"
    assert receipt["before"]["left"]["gripper_command_target"] == "open"
    effectors = session.observe(tmp_path / "observations")[0][
        "proprioception"]["end_effectors"]
    assert effectors["left"]["gripper_command_target"] == "close"
    assert effectors["right"]["gripper_command_target"] == "open"


def test_round_reset_writes_the_session_home_joints_before_the_upstream_reset(
    sparse, tmp_path, monkeypatch,
):
    """g004 classify_objects/build_tower: the upstream reset drives the arm home
    and cannot pass an obstruction on the home side (100 N m saturated, joints
    1-3 motionless), so round isolation failed. The Harness now writes the
    session-home joint state first and records it in the reset evidence."""

    manager = sparse.environment.robot_manager
    monkeypatch.setattr(type(manager), "reset_home", lambda self: None, raising=True)
    sparse.execute(v3(sparse, tmp_path, targets={"left": pose([0.30, 0.20, 1.10])}))
    moved = list(manager.joints["left_arm"])
    sparse.abort("operator_stop")
    sparse.persist_official_verdict()
    (sparse.output_root / "official-verdict.json").unlink()

    reset = sparse.restart_round(1, seed=0)

    assert reset["round_isolation"]["passed"] is True
    write = reset["pre_reset_home_joint_write"]
    assert write["method"] == "articulation_joint_state_write"
    left = write["arms"]["left"]
    assert left["joints_before"] == pytest.approx(moved)
    assert left["joint_distance_before_rad"] > 0.1
    assert left["joints_after_write"] == pytest.approx(left["home_joints"])
    assert manager.joints["left_arm"] == pytest.approx(left["home_joints"])
    record = json.loads((sparse.output_root / "round-resets/round-0001.json").read_text())
    assert record["pre_reset_home_joint_write"] == write


def test_round_reset_releases_the_previous_capture_products_before_upstream_recreates_them():
    events = []

    class Annotator:
        def __init__(self, name):
            self.name = name

        def detach(self, paths):
            events.append(("detach", self.name, tuple(paths)))

    class Product:
        def __init__(self, index):
            self.index = index

        def destroy(self):
            events.append(("destroy", self.index))

    views = [SimpleNamespace(
        _render_product_path=f"/rp/{index}", _annotators={"rgb": Annotator(f"rgb{index}")},
        _harness_camera_params=Annotator(f"params{index}"), _harness_simulation_time=Annotator(f"time{index}"),
        _harness_reference_time=Annotator(f"ref{index}"),
    ) for index in range(3)]
    manager = SimpleNamespace(
        tiled_cameras=list(views), tiled_render_products=[Product(index) for index in range(3)],
        annotator=[[], [], []], annotator_type=[["rgb"], ["rgb"], ["rgb"]], annotator_device=[[], [], []],
        _output_buffers={0: {}, 1: {}, 2: {}},
    )
    lifecycle = object.__new__(EvaluationLifecycle)
    lifecycle.capture_manager = manager
    lifecycle._release_capture_products()
    assert [event for event in events if event[0] == "destroy"] == [("destroy", 0), ("destroy", 1), ("destroy", 2)]
    assert ("detach", "params1", ("/rp/1",)) in events and ("detach", "rgb2", ("/rp/2",)) in events
    assert events.index(("detach", "rgb0", ("/rp/0",))) < events.index(("destroy", 0))
    assert manager.tiled_cameras == [] and manager.tiled_render_products == []
    assert manager.annotator == [] and manager.annotator_type == [] and manager.annotator_device == []
    assert manager._output_buffers == {}


def test_a_plant_that_cannot_report_its_lifecycle_refuses_the_round(
    sparse, tmp_path, monkeypatch,
):
    """Unreadable is refused, not assumed clean.

    A plant that cannot report its own episode lifecycle cannot prove it was
    restored, so the lifecycle checks fail rather than being skipped.
    """

    plant = sparse.environment
    original = type(plant).reset

    def leaking(self, seed):
        original(self, seed)
        del self.end_flag

    monkeypatch.setattr(type(plant), "reset", leaking)

    with pytest.raises(RobodojoError, match="not restore an isolated world"):
        _second_round(sparse, tmp_path)

    isolation = json.loads(
        (sparse.output_root / "round-resets/round-0001.json").read_text()
    )["round_isolation"]
    assert isolation["measured"]["reward_registered"] is False
    assert isolation["checks"]["official_lifecycle_rebound"] is False
    assert isolation["checks"]["episode_counters_zeroed"] is False
    assert isolation["checks"]["failure_latch_cleared"] is False


def test_source_manifest_attests_the_codec_the_worker_was_bound_to():
    """The manifest reports the resolved binding, never a compat-era literal.

    A RoboDojo cell that runs the motion codec used to attest
    ``runtime_action_v3`` because the worker returned a literal. The
    evaluator compares the manifest against the RunPlan's own codec, so the
    lie invalidated every landed episode at seal, long after the run.
    """

    bound = bridge.manifest_bindings({"action_codec": "motion_action_v3"}, "hang_mugs")
    assert bound["action_codec"] == "motion_action_v3"
    assert bound["official_native_action_limit"] == 800
    assert bound["sensor_profile"] == "rgbd_metric_v1"
    compat = bridge.manifest_bindings({"action_codec": "runtime_action_v3"}, "hang_mugs")
    assert compat["action_codec"] == "runtime_action_v3"


def test_source_manifest_refuses_an_unresolved_or_drifting_binding():
    """Every divergence is refused at worker startup, not discovered at seal."""

    with pytest.raises(bridge.RobodojoError):
        bridge.manifest_bindings({}, "hang_mugs")
    with pytest.raises(bridge.RobodojoError):
        bridge.manifest_bindings({"action_codec": "target_state_v4"}, "hang_mugs")
    for field, value in (
        ("sensor_profile", "rgb_only_v1"), ("physics_dt", 0.002), ("control_hz", 30),
        ("eval_seed", 1), ("layout_id", -1), ("layout_id", True), ("num_envs", 2),
        ("native_action_limit", 600),
    ):
        with pytest.raises(bridge.RobodojoError):
            bridge.manifest_bindings(
                {"action_codec": "motion_action_v3", field: value}, "hang_mugs",
            )
    # EXP14A g002: an unscheduled worker binds the RunPlan's one official layout;
    # its file and bytes are checked against the pinned Assets at start.
    assert bridge.manifest_bindings({"action_codec": "motion_action_v3", "layout_id": 2}, "hang_mugs")[
        "layout_id"] == 2
@pytest.mark.parametrize("upstream_renames", [False, True])
def test_repeated_reset_identity_is_measured_on_one_live_plant(
    tmp_path, upstream_renames,
):
    """The one scene fact two fresh workers can never establish.

    ``bridge._keep_layout_identities`` -- the workaround for upstream
    ``SceneManager.reload_env_scene`` regenerating every instance name from
    LayoutManager's monotone id counter -- runs only on a second reset of a
    *live* plant, i.e. only from ``restart_round``. The long infrastructure
    probe's two episodes are two fresh worker processes, so it never reaches
    this path; the per-task qualification layer does, and publishes both
    registrations so the comparison is evidence rather than an assumption.
    """

    plant = Plant()
    plant.rename_on_reset = upstream_renames
    instance = RobodojoSession(
        config={"workspace_bounds_m": {"x": [-1, 1], "y": [-1, 1], "z": [0, 2]}},
        output_root=tmp_path / "native",
        environment_factory=lambda config: (
            plant, SimpleNamespace(close=lambda: None), _fake_source_manifest(tmp_path),
        ),
    )
    instance._calibration = lambda name: (np.array([[50, 0, 5], [0, 50, 4], [0, 0, 1]]), np.eye(4))
    first = instance.start()["scene_registration"]
    assert first["instances"] == {"mug_0": "rigid", "rack_0": "geometry"}
    assert first["registered_instance_count"] == 2
    # The task's own registered limit, read from the environment rather than
    # restated: this is what the task layer proves in place of the official
    # terminal transition it cannot afford to witness.
    assert first["native_action_limit"] == plant.step_lim
    assert first["evaluator_registration"] == {
        "check_list": True, "final_check_list": False, "trigger_check_list": False,
    }

    repeated = instance.restart_round(1)["scene_registration"]
    if upstream_renames:
        assert repeated["instances"] == {"mug_1": "rigid", "rack_1": "geometry"}
        assert repeated != first
        assert (repeated["instance_identities_sha256"]
                != first["instance_identities_sha256"])
    else:
        assert repeated == first


def test_target_state_two_phases_share_one_consumed_observation(session, tmp_path):
    from ora.orchestrator.action_executor import execute_runtime_action
    parent = action(session, tmp_path)
    parent.update(schema="ora-action.v4", action="target_state", arguments={
        "arm": "left", "pose": {"position_m": [.02, 0, .8], "orientation_xyzw": [0, 0, 0, 1]},
        "gripper": 0.,
    })
    receipt = execute_runtime_action(session, parent)
    assert [r["status"] for r in receipt["phase_receipts"]] == ["reached", "executed"]
    assert session.action_sequences == []
    assert session.last_observation_id is None
    with pytest.raises(RobodojoError, match="unconsumed"):
        execute_runtime_action(session, parent)


def test_sequence_rejects_changed_phase_replay_and_uncommitted_child(session, tmp_path):
    from ora.contracts.action import target_state_phases
    parent = action(session, tmp_path)
    parent.update(schema="ora-action.v4", action="target_state", arguments={"arm": "left", "gripper": 0.})
    session.begin_action_sequence(parent)
    phase = target_state_phases(parent)[0]
    changed = deepcopy(phase)
    changed["arguments"]["gripper"] = 1.
    with pytest.raises(RobodojoError, match="changed its committed"):
        session.execute(changed)
    session.execute(phase)
    with pytest.raises(RobodojoError, match="duplicated"):
        session.execute(phase)
    session.end_action_sequence(parent["action_id"])
    assert not session.action_sequences


def test_closed_skill_consumes_bound_children_and_cleans_up_on_fault(session, tmp_path):
    from ora.orchestrator.action_executor import execute_skill_sequence
    parent = action(session, tmp_path)
    parent.update(schema="ora-skill-sequence-action.v1", action="execute_skill_sequence", arguments={
        "skill": "grasp_execute", "candidate_id": "grasp-test", "rationale": "source-bound test",
        "stop_before_commit": False, "sequence": ["approach", "close", "verify"], "resume_from": 0,
    })
    def drive(action, *, stage, stage_index, **kwargs):
        if stage == "close":
            raise RuntimeError("injected driver fault")
        return {"kind": "mutation", "action": {
            **action, "schema": "ora-action.v4", "action": "target_state",
            "action_id": f"{action['action_id']}:{stage_index:02d}-{stage}",
            "arguments": {"arm": "left", "pose": {"position_m": [.02, 0, .8], "orientation_xyzw": [0, 0, 0, 1]}},
        }}
    reports = []
    with pytest.raises(RuntimeError, match="injected"):
        execute_skill_sequence(session, parent, initial_observation={}, initial_observation_root=tmp_path,
                               drive_stage=drive, build_receipt=lambda **kw: reports.append(kw) or {},
                               capture_observation=lambda: ({"observation_id": "fresh"}, tmp_path))
    assert not session.action_sequences
    assert len(reports) == 1
    assert len(session.environment.commands) == 1


def current_pose_skill(session, observation, root, *, arm="left"):
    """Drive the real skill driver at the measured current pose.

    A copy of the qualification smoke's helper, which stays byte-identical
    in its package because the smoke's dependency closure hashes that file.
    """
    from functools import partial
    from ora.agent.composition import resolve_tool_surface
    from ora.agent.tools import skill_driver, skill_ledger
    from ora.orchestrator.action_executor import execute_skill_sequence
    from ora.orchestrator.spec import json_sha256

    root = Path(root)
    control = root / "control"
    control.mkdir(parents=True)
    host = SimpleNamespace(control_root=control, artifact_root=root / "artifacts",
                           surface=resolve_tool_surface("robodojo_placement_closed_skills_v2"))
    candidate_id = "qualification-current-pose"
    observation_id = observation["observation_id"]
    state = skill_ledger.begin(skill="grasp_execute", candidate_id=candidate_id,
                               rationale="Protocol regression at the measured current pose",
                               observation_id=observation_id, observation_sha256=json_sha256(observation),
                               stop_before_commit=False)
    skill_driver.store_ledger(host, state)
    xyz_xyzw = observation["proprioception"]["end_effectors"][arm]["pose_xyz_xyzw"]
    target = {"position_m": xyz_xyzw[:3], "orientation_xyzw": xyz_xyzw[3:]}
    candidate_root = host.artifact_root / "grasp-queries" / "qualification"
    candidate_root.mkdir(parents=True)
    (candidate_root / "manifest.json").write_text(json.dumps({
        "query_ref": "qualification", "identity": {"observation_id": observation_id,
                                                   "target_ref": candidate_id, "compiler": {"arm": arm}},
        "compiled_candidates": [{"candidate_id": candidate_id, "contact_pose": target}],
    }))
    skill_driver.store_commitment(host, {"skill": "grasp_execute", "candidate_id": candidate_id,
                                         "arm": arm, "observation_id": observation_id,
                                         "poses": {"approach": target}, "grippers": {"close": 0.0}})
    parent = {"schema": "ora-skill-sequence-action.v1",
              "action": "execute_skill_sequence", "action_id": "smoke-skill",
              "observation_id": observation_id, "bound_observation_sha256": json_sha256(observation),
              "arguments": {key: state[key] for key in
                            ("skill", "candidate_id", "rationale", "stop_before_commit", "sequence")}}
    parent["arguments"]["resume_from"] = 0

    def capture():
        fresh, _ = session.observe(root / "observations")
        return fresh, root / "observations" / fresh["observation_id"]

    receipt, latest = execute_skill_sequence(session, parent,
        initial_observation=observation, initial_observation_root=root,
        drive_stage=partial(skill_driver.drive_skill_stage, host),
        build_receipt=partial(skill_driver.build_skill_receipt, host),
        capture_observation=capture)
    assert receipt["status"] == "completed", receipt["stop_reason"]
    assert receipt["dispatched_stage_actions"] == 2
    assert [row["stage"] for row in receipt["stage_receipts"]] == ["approach", "close", "verify"]
    assert [row["action"]["schema"] for row in receipt["stage_receipts"][:2]] == [
        "ora-motion-action.v3", "ora-motion-action.v3"]
    assert receipt["skill_receipt"]["ledger_released"] is True
    return receipt, latest[0]


@pytest.mark.parametrize("arm", ["left", "right"])
def test_real_sparse_skill_driver_executes_committed_sequence(sparse, tmp_path, arm):
    observation, _ = sparse.observe(tmp_path / "observations")
    receipt, fresh = current_pose_skill(sparse, observation, tmp_path / "skill", arm=arm)
    assert receipt["status"] == "completed"
    assert fresh["observation_id"] != observation["observation_id"]
    assert sparse.action_sequences == []
    assert sparse.action_sequence == 2
    assert sparse.gripper_directives[arm] == 0.0


@pytest.mark.parametrize("corruption", ["observation_id", "bound_observation_sha256", "action_id", "schema", "codec"])
def test_sparse_skill_admission_rejects_drift_without_consuming_child(sparse, tmp_path, corruption):
    parent = v3(sparse, tmp_path, targets={"left": pose([0, 0, .8])})
    child = deepcopy(parent)
    parent.update(schema="ora-skill-sequence-action.v1", action="execute_skill_sequence", arguments={
        "skill": "grasp_execute", "candidate_id": "grasp-test", "rationale": "source-bound test",
        "stop_before_commit": False, "sequence": ["approach", "close", "verify"], "resume_from": 0,
    })
    child["action_id"] = parent["action_id"] + ":00-approach"
    sparse.begin_action_sequence(parent)
    changed = deepcopy(child)
    if corruption == "codec":
        sparse.action_codec = "runtime_action_v3"
    elif corruption == "schema":
        changed.update(schema="ora-action.v3", arguments={
            "arm": "left", **pose([0, 0, .8]),
        })
    else:
        changed[corruption] = "b" * 64
    with pytest.raises(RobodojoError):
        sparse.execute(changed)
    assert sparse.action_sequences[-1]["next"] == 0
    assert sparse.environment.commands == []
    sparse.action_codec = "motion_action_v3"
    sparse.execute(child)
    with pytest.raises(RobodojoError, match="duplicated"):
        sparse.execute(child)
    sparse.end_action_sequence(parent["action_id"])


def test_cuda_garment_prim_is_created_before_physics_and_retained_until_close():
    class Manager:
        spawnable_object_types = ['rigid', 'articulation', 'garment']
        def __init__(self):
            self.physics_started = False
            self.garments = {}
            self._garment_objects = [self.garments]
            self.created = 0
        def spawn_scene_objects(self, env_id, exclude_types=None):
            if 'garment' not in (exclude_types or []):
                assert not self.physics_started, 'cloth must exist before PhysX starts'
                self.garments[env_id] = SimpleNamespace(
                    sample_mesh_vertices=lambda: ('world-particles', 'mesh-particles', None, None),
                    get_world_pose=lambda: ('prim-position', 'prim-quaternion'))
                self.created += 1
        def clear_scene_objects(self, env_id, exclude_types=None):
            if 'garment' not in (exclude_types or []):
                self.garments.pop(env_id, None)
    manager = Manager()
    bridge._prepare_cuda_garments(manager, 1)
    original = manager.garments[0]
    assert original.sample_mesh_vertices() == ('world-particles', 'mesh-particles', 'prim-position', 'prim-quaternion')
    manager.physics_started = True
    for _ in range(2):
        manager.clear_scene_objects(0, exclude_types=['rigid', 'articulation'])
        manager.spawn_scene_objects(0, exclude_types=['rigid', 'articulation'])
        assert manager.garments[0] is original
    assert manager.created == 1
    manager.clear_scene_objects(0)
    assert manager.garments == {}


@pytest.mark.parametrize('arm', ['left', 'right'])
def test_sdk_limited_close_keeps_intent_through_observation_and_receipt(sparse, tmp_path, monkeypatch, arm):
    """Recorded SDK witness: first q=.044, requested q=-.01 -> drive q=.0332.

    An obstructed second finger differs; it must not become the limiter's
    reference or turn the close directive into an unknown/open command.
    """
    from ora.agent.context.public_facts import public_end_effectors
    from ora.agent.native.receipt_projection import (
        ReceiptProjectionContext, resolve_receipt_effect_projector,
    )
    from ora.agent.context.projection import causal_provider_value
    manager = sparse.environment.robot_manager
    robot = sparse.robots[arm]
    key = manager.process_name(robot.gripper_name)
    for articulation in manager.robot_key:
        articulation.data.joint_pos_limits[0][6:] = [[0., .044], [0., .044]]
    native_tick = sparse.environment.native_tick
    measured = [.044, .0345]
    manager.fingers[robot.arm_name] = measured[:]

    def obstructed_tick(control):
        # Model the SDK postprocessor, not the pre-SDK command buffer.
        resolved = deepcopy(control)
        if key in resolved:
            requested = resolved[key]['position'][0]
            resolved[key]['position'] = [max(measured[0] - .2 * .054,
                min(measured[0] + .2 * .054, requested))] * 2
        result = native_tick(resolved)
        manager.fingers[robot.arm_name] = measured[:]
        return result

    monkeypatch.setattr(sparse.environment, 'native_tick', obstructed_tick)
    record = sparse.execute(v3(sparse, tmp_path, kind='set_gripper', targets={arm: 'close'}))['receipt']
    assert record['execution_status'] == 'not_reached'
    # The blocked jaws never move, so the shared stall rule ends the command
    # after three still native actions instead of the 20-action budget.
    assert record['failure_code'] == 'gripper_stalled_before_target'
    assert record['native_actions'] == 3
    assert record['arms'][arm]['gripper_outcome']['sdk_limited_drive_target_normalized'] == pytest.approx(.8)
    # The second finger stands at the limited target (.0345 m ~ .82), so the
    # outcome names the primary-relative hold beside both finger readings.
    assert record['arms'][arm]['gripper_outcome']['primary_finger_limit_holds_close'] is True
    assert record['arms'][arm]['gripper_outcome']['measured_finger_displacements_m'] == pytest.approx(measured)
    assert record['after'][arm]['gripper_command_target'] == 'close'
    feedback = record['after'][arm]['gripper_feedback']
    assert feedback['requested_opening_normalized'] == 0.
    assert feedback['drive_target_opening_normalized'] == pytest.approx(.8)
    # E11G12-001: both fingers' mean, not the pinned first finger's 1.0.
    assert feedback['measured_opening_normalized'] == pytest.approx((sum(measured) / 2 + .01) / .054)
    assert feedback['finger_lateral_offset_m'] == pytest.approx((.044 - .0345) / 2)
    assert feedback['effective_target_opening_normalized'] == pytest.approx(.01/.054)
    assert feedback['arrival_check']['tolerance_normalized'] == .03
    assert feedback['arrival_check']['within_tolerance'] is False
    # Every capability except the 0059 delivery rules: under
    # within_payload_deduplication_v1 this feedback is read from the current
    # facts instead (tests/agent/context/test_payload_dedup.py).
    from ora.agent.native.effect_delivery import DELIVERY_CAPABILITIES
    surface = ReceiptProjectionContext(None, 'set_gripper', lambda name: name not in DELIVERY_CAPABILITIES)
    public = causal_provider_value(resolve_receipt_effect_projector(record['schema'])(record, surface))['detail']
    assert public['gripper_feedback'][arm]['requested_command'] == 'close'
    assert public['gripper_feedback'][arm]['drive_target_opening_normalized'] == pytest.approx(.8)
    assert public['acceptance_checks'][0]['tolerance']['opening_normalized'] == .03
    assert public['execution_window'] == {'native_actions_used': 3, 'native_action_budget': 20}
    payload, _ = sparse.observe(tmp_path / 'observations')
    state = public_end_effectors(payload['proprioception'], 'arx_x5_link6')[arm]['gripper']
    assert state['commanded_opening_normalized'] == 0.
    assert state['feedback'] == feedback
    assert payload['controller_feedback']['gripper_drive_limit']['maximum_normalized_offset'] == .2
    damaged = deepcopy(record)
    del damaged['after'][arm]['gripper_feedback']['effective_target_opening_normalized']
    with pytest.raises(ValueError, match='fields differ'):
        resolve_receipt_effect_projector(record['schema'])(damaged, surface)
    damaged = deepcopy(record)
    del damaged['after'][arm]['gripper_feedback']
    with pytest.raises(ValueError, match='both arm gripper snapshots'):
        resolve_receipt_effect_projector(record['schema'])(damaged, surface)
    # Next pose holds the already applied target, and the original intent.
    held = deepcopy(manager.control_manager.prev_control[0][key])
    next_record = sparse.execute(v3(sparse, tmp_path, targets={arm: pose([.01, 0, .8])}))['receipt']
    assert manager.control_manager.prev_control[0][key] == held
    assert next_record['after'][arm]['gripper_feedback']['requested_command'] == 'close'
    assert next_record['acceptance_checks'][0]['tolerance'] == {'position_m': .01, 'orientation_rad': .1}


def test_legacy_dojo_close_uses_physical_floor_and_pose_checks(session, tmp_path):
    manager = session.environment.robot_manager
    for articulation in manager.robot_key:
        articulation.data.joint_pos_limits[0][6:] = [[0., .044], [0., .044]]
    record = session.execute(action(session, tmp_path, kind='set_gripper'))['receipt']
    assert record['motion_status'] == 'reached'
    assert record['native_actions'] == 1
    assert record['acceptance_checks'][0]['arms']['left']['within_tolerance']['opening_normalized'] is True
    record = session.execute(action(session, tmp_path))['receipt']
    assert record['acceptance_checks'][0]['stage'] == 'final_endpoint'
