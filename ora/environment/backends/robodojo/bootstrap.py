"""Isolated Isaac application boot and exact RoboDojo environment construction."""
from __future__ import annotations
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib
from typing import Any, Mapping

#: Process caches and scratch redirected under the run's runtime directory
#: before Isaac starts. OptiX reads OPTIX_CACHE_PATH (the optix_host.h text
#: embedded in omni.hydra.rtx libnrend.so); unset, its cache lands in
#: /var/tmp/OptixCache_<user>, else /tmp, on the system partition.
PROCESS_PATHS = (("XDG_CACHE_HOME", "cache/xdg"), ("XDG_DATA_HOME", "data/xdg"),
                 ("XDG_CONFIG_HOME", "config/xdg"), ("XDG_RUNTIME_DIR", "runtime/xdg"),
                 ("CUDA_CACHE_PATH", "cache/cuda"), ("__GL_SHADER_DISK_CACHE_PATH", "cache/gl"),
                 ("TMPDIR", "tmp"), ("OPTIX_CACHE_PATH", "cache/optix"))


def bind_process_paths(runtime_directory: Path) -> dict[str, str]:
    """Create and export every PROCESS_PATHS directory for this process."""
    for variable, relative in PROCESS_PATHS:
        directory = runtime_directory / relative
        directory.mkdir(parents=True, exist_ok=True)
        if variable == "XDG_RUNTIME_DIR":
            directory.chmod(0o700)
        os.environ[variable] = str(directory)
    return {variable: os.environ[variable] for variable, _ in PROCESS_PATHS}


#: Longest wait for a Kit/Vulkan initialization slot; inside the 600 s start limit.
KIT_INIT_GATE_TIMEOUT_S = 300
#: nvidia-smi answered in 6-13 s on a loaded simulator host (2026-10-07); 15 s failed starts.
NVIDIA_SMI_TIMEOUT_S = 60


def nvidia_smi(arguments: list[str]) -> str:
    """One bounded ``nvidia-smi`` query, retried once after a timeout (RDJ-10)."""
    for attempt in range(2):
        try:
            return subprocess.check_output(["nvidia-smi", *arguments], text=True, timeout=NVIDIA_SMI_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            if attempt:
                raise
    raise AssertionError("unreachable")


def build_environment(config: Mapping[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    """Launch one source-verified Isaac application and original RoboDojo task."""
    from .bridge import (EvaluationLifecycle, RobodojoError, _CRASH_REPORT_SETTINGS,
                         _configure_crash_reporting, _crash_reporting_readback,
                         renderer_table_binding, source_manifest)
    runtime_directory = Path(config["runtime_directory"]).resolve()
    if not runtime_directory.is_absolute() or runtime_directory == Path("/"):
        raise RobodojoError("runtime_directory must be an explicit non-root path")
    bind_process_paths(runtime_directory)
    omni_config = runtime_directory / "config/omniverse"
    omni_config.mkdir(parents=True, exist_ok=True)
    os.environ["OMNI_CONFIG_PATH"] = str(omni_config)
    omni_paths = {"data_root": runtime_directory / "data/ov", "cache_root": runtime_directory / "cache/ov",
                  "logs_root": runtime_directory / "logs/ov"}
    for directory in omni_paths.values():
        directory.mkdir(parents=True, exist_ok=True)
    config_text = "[paths]\n" + "".join(f"{key} = {json.dumps(str(value))}\n" for key, value in omni_paths.items())
    omni_config_file = omni_config / "omniverse.toml"
    if omni_config_file.exists():
        if tomllib.loads(omni_config_file.read_text()).get("paths") != {key: str(value) for key, value in omni_paths.items()}:
            raise RobodojoError("dedicated Omniverse path configuration differs from the runtime binding")
    else:
        omni_config_file.write_text(config_text)
    if config.get("accept_eula") is True:
        os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    crash_report_args, crash_privacy_file = _configure_crash_reporting(Path(config["output_root"]).resolve())
    manifest = source_manifest(config)
    from .graphics_compat import activate, verify_activation
    from . import start_gate

    # RDJ-10: concurrent Vulkan/Kit initializations queue in the driver; admit a
    # bounded number per node from the readback through SDK startup.
    # A start that fails before Kit exists ends the worker process, which releases the slot.
    init_gate = start_gate.gate("kit-init", start_gate.INIT_SLOTS, timeout_s=KIT_INIT_GATE_TIMEOUT_S)
    gate_record = init_gate.__enter__()
    graphics = activate(runtime_directory)
    manifest["graphics_compatibility"] = graphics
    manifest["loaded_files"].update({
        f"graphics-compat/{role}": reference["sha256"]
        for role, reference in graphics["files"].items()
    })
    root = Path(config["robodojo_root"]).resolve()
    sys.path.insert(0, str(root))
    # The global module imports only os; bind roots before importing upstream managers.
    globals_module = importlib.import_module("env.global_configs")
    assets = str(Path(config["asset_root"]).resolve())
    globals_module.ASSETS_PATH = assets
    globals_module.ROBOTS_PATH = str(Path(assets) / "Robots")
    globals_module.OBJECTS_PATH = str(Path(assets) / "Object/RoboDojo")
    from isaaclab.app import AppLauncher

    startup_log = Path(config["output_root"]).resolve() / "kit-startup.log"
    if startup_log.exists():
        raise RobodojoError("refusing to reuse an existing Kit startup log")
    startup_log.parent.mkdir(parents=True, exist_ok=True)
    # This exact IsaacLab revision selects its unified experience for every
    # mode, including headless camera rendering. The legacy file remains in
    # its tree but omits the current RTX initialization/stability settings.
    experience = root / "third_party/IsaacLab/apps/isaaclab.python.kit"
    if not experience.is_file():
        raise RobodojoError("the pinned IsaacLab default experience is missing")
    launcher = AppLauncher(
        headless=True, enable_cameras=True, fast_shutdown=False, device=str(config.get("physics_device", "cuda:0")),
        experience=str(experience),
        kit_args=(f"--/renderer/activeGpu={int(config.get('renderer_gpu', 0))} --/renderer/multiGpu/enabled=false "
                  f"--/renderer/multiGpu/maxGpuCount=1 --/renderer/activeCudaGpus=0 --/renderer/useCudaDeviceOrder=true "
                  f"--/plugins/carb.tasking.plugin/suppressTaskException=false "
                  f"--/log/file={startup_log} --portable-root {runtime_directory / 'kit'} {crash_report_args}"),
    )
    app = launcher.app
    try:
        import hashlib
        import carb.settings

        verify_activation(runtime_directory, graphics)
        init_gate.__exit__(None, None, None)
        manifest["start_gate"] = gate_record

        import omni.structuredlog

        settings = carb.settings.get_settings()
        privacy_before = {key: settings.get(key) for key in _CRASH_REPORT_SETTINGS}
        settings.set_string("/structuredLog/privacySettingsFile", str(crash_privacy_file))
        if not omni.structuredlog.IStructuredLogSettings().load_privacy_settings():
            raise RobodojoError("SDK could not load the explicit episode privacy file")
        # This SDK build retains performance=True even after a successful
        # privacy-file load. Apply the episode's explicit opt-out in this
        # isolated process; the upload URL has already been empty since startup.
        for category in ("performance", "personalization", "usage"):
            settings.set_bool(f"/privacy/{category}", False)
        manifest["crash_reporting"] = _crash_reporting_readback(settings, crash_privacy_file)
        manifest["crash_reporting"]["before_explicit_privacy_load"] = privacy_before
        # RoboDojo owns external IsaacSim cameras rather than IsaacLab Camera
        # objects, so it must register their render requirement explicitly.
        settings.set_bool("/isaaclab/render/rtx_sensors", True)
        manifest["render_settings"] = {key: settings.get(key) for key in (
            "/isaaclab/cameras_enabled", "/isaaclab/render/offscreen",
            "/isaaclab/render/rtx_sensors", "/isaaclab/render/active_viewport",
            "/app/asyncRendering", "/omni/replicator/asyncRendering",
        )}
        if (manifest["render_settings"]["/isaaclab/render/offscreen"] is not True
                or manifest["render_settings"]["/isaaclab/render/rtx_sensors"] is not True):
            raise RobodojoError("RoboDojo external RGB-D sensors require offscreen RTX rendering")
        import torch
        import carb.tokens

        tokens = carb.tokens.get_tokens_interface()
        resolved_paths = {}
        for name in ("omni_global_cache", "omni_global_data", "omni_global_logs", "cache", "data", "logs"):
            resolved = tokens.resolve("${" + name + "}")
            path = Path(resolved)
            if not resolved or "${" in resolved or not path.is_absolute() or not path.resolve().is_relative_to(runtime_directory):
                raise RobodojoError(f"Kit runtime path escapes the dedicated runtime: {name}")
            resolved_paths[name] = str(path.resolve())
        manifest["runtime_paths"] = {"kit_tokens": resolved_paths, "runtime_directory": str(runtime_directory),
                                     "xdg_paths": {key: os.environ[key] for key in (
                                         "XDG_CACHE_HOME", "XDG_DATA_HOME", "XDG_CONFIG_HOME", "XDG_RUNTIME_DIR")},
                                     "omniverse_configuration": str(omni_config_file),
                                     "optix_cache": os.environ["OPTIX_CACHE_PATH"]}

        if torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
            raise RobodojoError("the isolated worker must have exactly one logical CUDA device")
        gpu_output = nvidia_smi(["--query-gpu=index,uuid,pci.bus_id", "--format=csv,noheader,nounits"])
        inventory = []
        for line in gpu_output.splitlines():
            index, gpu_uuid, pci_bus_id = [item.strip() for item in line.split(",")]
            inventory.append({"index": int(index), "uuid": gpu_uuid, "pci_bus_id": pci_bus_id})
        log_prefix = startup_log.read_bytes()
        device_binding = renderer_table_binding(
            log_prefix.decode("utf-8", errors="replace"), selected_index=int(config["renderer_gpu"]),
            cuda_uuid=str(torch.cuda.get_device_properties(0).uuid), inventory=inventory,
        )
        # Kit keeps appending its live log; preserve the exact readback prefix separately.
        prefix_path = startup_log.with_name("kit-startup-readback.log")
        with prefix_path.open("xb") as stream:
            stream.write(log_prefix)
        prefix_path.chmod(0o444)
        device_binding["startup_log_prefix"] = {"path": prefix_path.name,
                                                "sha256": hashlib.sha256(log_prefix).hexdigest()}
        manifest["device_binding"] = device_binding
        # Upstream exposes this module flag; XDG_CACHE_HOME does not redirect it.
        # Set it before importing task modules that construct CuRobo objects.
        import curobo._src.runtime as curobo_runtime

        curobo_cache = runtime_directory / "cache/curobo"
        curobo_cache.mkdir(parents=True, exist_ok=True)
        curobo_runtime.cache_dir = str(curobo_cache)
        manifest["runtime_paths"]["curobo_cache"] = curobo_runtime.cache_dir
        from omegaconf import OmegaConf
        from utils.pipeline_utils import process_config, process_randomization

        task = manifest["task"]
        eval_config = OmegaConf.load(root / "env_cfg/arx_x5.yml")
        eval_config.update({"task_name": task, "seed": 0, "num_envs": 1})
        composed = OmegaConf.create({
            name: OmegaConf.load(root / f"env_cfg/{name}/{eval_config.config[name]}.yml")
            for name in ("sim", "scene", "robot", "camera")
        })
        composed.task_env = OmegaConf.load(root / f"task/RoboDojo/config/{task}.yml")
        composed.eval_cfg = eval_config
        composed = process_randomization(composed)
        composed, _ = process_config(composed, task_name=task)
        composed.sim.scene.num_envs = 1
        composed.sim.device = str(config.get("physics_device", "cuda:0"))
        composed.sim.seed = [0]
        composed.camera.default_frequency = 25
        composed.eval_cfg.observation.vision.update({
            "depth": True, "approximate_depth": False, "intrinsic_matrix": True, "extrinsic_matrix": True,
        })
        for camera in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
            composed.camera.annotator[camera].distance_to_image_plane_capture = {
                "type": "distance_to_image_plane", "device": "cpu",
            }
        if float(composed.sim.dt) != 0.004 or int(composed.sim.decimation) != 1:
            raise RobodojoError("upstream task physics cadence has changed")
        task_class = getattr(importlib.import_module(f"task.RoboDojo.tasks.{task}"), task)
        cls = type("HarnessRoboDojoTask", (EvaluationLifecycle, task_class), {})
        import faulthandler

        with (Path(config["output_root"]) / "environment-initialization-stack.txt").open("w") as stack_log:
            faulthandler.dump_traceback_later(120, file=stack_log)
            try:
                environment = cls(composed, app)
            finally:
                faulthandler.cancel_dump_traceback_later()
        environment.harness_output_root = Path(config["output_root"])
        manifest["camera_capture"] = {
            "adapter": "single_environment_numpy_rgbd_v1", "num_envs": 1,
            "original_annotators": True, "pixels_resampled": False,
            "buffer_readback": "camera-buffer-readback.json",
            "texture_wait": "after_original_capture_reset_before_first_observation",
            "texture_wait_timeout_s": 180,
            "texture_wait_evidence": "render-readiness.jsonl",
            "intrinsics": "original_render_product_camera_params_projection",
            "frame_synchronization": "current_physics_time_and_common_reference_time_v1",
            "frame_synchronization_max_renders": 8,
            "frame_synchronization_min_renders": 2,
            "frame_synchronization_evidence": "render-synchronization.jsonl",
            "optical_pose_check": "render_product_matches_current_physical_prim_atol_1e-4",
        }
        manifest["resolved_environment_config"] = OmegaConf.to_container(composed, resolve=True)
        manifest["evaluator_input_adapter"] = "private_pose_device_copy_numpy_v2"
        manifest["held_control_adapter"] = "per_step_resolved_meta_control_v1"
        return environment, app, manifest
    except BaseException:
        import traceback

        diagnostic = traceback.format_exc()
        (Path(config["output_root"]) / "startup-error.txt").write_text(diagnostic)
        print(diagnostic, file=sys.stderr, flush=True)
        init_gate.__exit__(None, None, None)  # idempotent; a failure before here exits the worker, which releases it
        app.close()
        raise
