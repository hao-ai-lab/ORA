"""Policy-neutral extraction of the pinned RoboDojo EvalEnv lifecycle.

The reference is RoboDojo ee67a1468510da7624a089164402359f2afc72c8,
src/eval_client/eval_env.py. Task classes and reward predicates stay upstream.
No websocket client, policy loop, abandoned-seed replacement or process restart
is imported. Isaac-dependent imports occur only inside ``build_environment``.
"""

from __future__ import annotations

from copy import deepcopy
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tomllib
from typing import Any, Mapping

import numpy as np


class RobodojoError(RuntimeError):
    """A failed source, sensor, controller or environment qualification."""


def _fixed_meta_control(meta_control: Any, robot_manager: Any,
                        previous: Mapping[str, Any], requested: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Resolve one SDK command, preserving dimensions absent from the proposal.

    ControlManager.pop fills missing dimensions and runs MetaControl's
    measurement-dependent gripper limiter on them. Restore those held targets
    and freeze this instance's resolved action before RobotManager reads it
    again. Explicit gripper commands retain the original limiter.
    """
    resolved = deepcopy(meta_control.get_action(robot_manager, env_idx=0))
    for key, target in previous.items():
        if key not in requested:
            resolved[key] = deepcopy(target)

    def get_action(robot_manager: Any, env_idx: int) -> dict[str, Any]:
        if env_idx != 0:
            raise RobodojoError("resolved control belongs to the single environment")
        return deepcopy(resolved)

    # Keep the original MetaControl identity required by RobotManager. This
    # changes only this queued instance, never the upstream class or module.
    meta_control.get_action = get_action
    return meta_control, deepcopy(resolved)


class EvaluatorLayoutInput:
    """Copy private GPU poses to the original NumPy evaluator's input device."""

    def __init__(self, delegate: Any):
        self._delegate = delegate

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def get_functional_points(self, *args: Any, **kwargs: Any) -> Any:
        # Run the original algorithm with nested pose reads on this adapter.
        return type(self._delegate).get_functional_points(self, *args, **kwargs)

    def get_support_points(self, *args: Any, **kwargs: Any) -> Any:
        return type(self._delegate).get_support_points(self, *args, **kwargs)

    def get_instance_pose(self, env_idx: int, label: str | None = None,
                          inst_name: str | None = None, relative: bool = True) -> tuple[np.ndarray, np.ndarray]:
        resolved_name = inst_name if inst_name is not None else self._delegate.get_instance_name(env_idx, label)
        kinds = getattr(self._delegate, "instance_type_by_env", {})
        if kinds and kinds[env_idx].get(resolved_name) == "geometry":
            obj = self._delegate.get_scene_object(env_idx, resolved_name)
            root = obj.get_state(is_relative=False)["root_pose"]
            if hasattr(root, "detach"):
                root = root.detach().cpu().numpy()
            root = np.asarray(root).copy()
            if hasattr(obj, "env_origin"):
                origin = obj.env_origin
                if hasattr(origin, "detach"):
                    origin = origin.detach().cpu().numpy()
                # Match Geometry.get_state(is_relative=True) and the original
                # layout manager's geometry branch, including float32 math.
                root[:3] -= np.asarray(origin, dtype=root.dtype)[:3]
            pose = root[:3], root[3:]
        else:
            pose = self._delegate.get_instance_pose(env_idx=env_idx, label=label,
                                                    inst_name=inst_name, relative=relative)
        values = []
        for value, dimension in zip(pose, (3, 4), strict=True):
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            copied = np.asarray(value).copy()
            if copied.shape != (dimension,) or not np.isfinite(copied).all():
                raise RobodojoError("official evaluator received an invalid measured object pose")
            values.append(copied)
        return values[0], values[1]


_CRASH_REPORT_SETTINGS = {
    "/app/uploadDumpsOnStartup": False,
    "/crashreporter/skipOldDumpUpload": True,
    "/crashreporter/url": "",
    "/crashreporter/devOnlyOverridePrivacyAndForceUpload": False,
    "/privacy/performance": False,
    # omni.kit.telemetry starts its transmitter during extension startup, before
    # the post-start privacy load, and the IsaacLab experience enables anonymous
    # data (all consent levels). Every category must be off on the command line.
    "/privacy/usage": False,
    "/privacy/personalization": False,
    "/telemetry/enableAnonymousData": False,
    "/telemetry/enableAnonymousAppName": False,
    "/crashreporter/preserveDump": True,
    "/crashreporter/enabled": True,
}
_CRASH_REPORT_ENV = {
    "OMNI_CRASHREPORTER_URL": "",
    "OMNI_CRASHREPORTER_SKIPOLDDUMPUPLOAD": "1",
    "OMNI_CRASHREPORTER_PRESERVEDUMP": "1",
    "OMNI_CRASHREPORTER_ENABLED": "1",
}
_CRASH_PRIVACY_TEXT = "[privacy]\nperformance = false\npersonalization = false\nusage = false\n"


def _configure_crash_reporting(output_root: Path) -> tuple[str, Path]:
    """Block uploads before SDK import while retaining local crash diagnostics."""
    os.environ.update(_CRASH_REPORT_ENV)
    output_root.mkdir(parents=True, exist_ok=True)
    privacy_file = output_root / "crash-reporting-privacy.toml"
    with privacy_file.open("x") as stream:
        stream.write(_CRASH_PRIVACY_TEXT)
    privacy_file.chmod(0o444)
    settings = {**_CRASH_REPORT_SETTINGS, "/structuredLog/privacySettingsFile": str(privacy_file)}
    # AppLauncher splits kit_args on whitespace. Fail before starting the SDK
    # if the dedicated path cannot be represented by that upstream interface.
    if any(character.isspace() for character in str(privacy_file)):
        raise RobodojoError("crash-report privacy path must not contain whitespace")
    return " ".join(f"--{key}={str(value).lower() if isinstance(value, bool) else value}"
                    for key, value in settings.items()), privacy_file


def _crash_reporting_readback(settings: Any, privacy_file: Path) -> dict[str, Any]:
    """Reject effective SDK settings that could upload or discard local dumps."""
    import hashlib

    expected = {**_CRASH_REPORT_SETTINGS, "/structuredLog/privacySettingsFile": str(privacy_file)}
    actual = {key: settings.get(key) for key in expected}
    for key, value in expected.items():
        if type(actual[key]) is not type(value) or actual[key] != value:
            raise RobodojoError(f"crash-report privacy setting differs from the worker binding: {key}; expected {value!r}, read {actual[key]!r} ({type(actual[key]).__name__})")
    if (any(os.environ.get(key) != value for key, value in _CRASH_REPORT_ENV.items())
            or privacy_file.read_text() != _CRASH_PRIVACY_TEXT):
        raise RobodojoError("crash-report privacy environment or consent file changed during startup")
    return {"settings": actual, "environment": dict(_CRASH_REPORT_ENV),
            "privacy_file": {"path": privacy_file.name,
                             "sha256": hashlib.sha256(privacy_file.read_bytes()).hexdigest()},
            "local_dumps_preserved": True, "uploads_disabled": True}


def renderer_table_binding(log_text: str, *, selected_index: int, cuda_uuid: str,
                           inventory: list[dict[str, Any]]) -> dict[str, Any]:
    """Join Kit's actual active-device table with full physical CUDA identities."""
    lines = log_text.splitlines()
    active = []
    for index, line in enumerate(lines):
        columns = [part.strip() for part in line.split("|")]
        if len(columns) < 9 or not columns[1].isdigit() or not columns[3].startswith("Yes:"):
            continue
        if index + 2 >= len(lines):
            raise RobodojoError("active renderer device table is incomplete")
        uuid_columns = [part.strip() for part in lines[index + 1].split("|")]
        bus_columns = [part.strip() for part in lines[index + 2].split("|")]
        if len(uuid_columns) < 9 or len(bus_columns) < 9:
            raise RobodojoError("active renderer identity rows are incomplete")
        prefix = uuid_columns[7].lower().removesuffix("..").replace("-", "")
        if not re.fullmatch(r"[0-9a-f]{8,32}", prefix):
            raise RobodojoError("active renderer UUID prefix is invalid")
        active.append({"renderer_vulkan_index": int(columns[1]),
                       "renderer_cuda_index": int(columns[3].split(":", 1)[1].strip()),
                       "renderer_uuid_prefix": prefix, "renderer_pci_bus_hex": bus_columns[6].lower()})
    # Multiple tables can signal device reinitialization; do not silently choose one.
    if len(active) != 1:
        raise RobodojoError("exactly one active renderer device table entry is required")
    observed = active[0]
    def normalized_uuid(value: str) -> str:
        return value.lower().removeprefix("gpu-").replace("-", "")
    candidates = [item for item in inventory
                  if normalized_uuid(item["uuid"]).startswith(observed["renderer_uuid_prefix"])
                  and int(item["pci_bus_id"].split(":")[-2], 16) == int(observed["renderer_pci_bus_hex"], 16)]
    if (len(candidates) != 1 or candidates[0]["index"] != selected_index
            or observed["renderer_vulkan_index"] != selected_index or observed["renderer_cuda_index"] != 0
            or normalized_uuid(candidates[0]["uuid"]) != normalized_uuid(cuda_uuid)):
        raise RobodojoError("actual Vulkan/CUDA GPU identity differs from the selected physical GPU")
    return {"schema": "ora.robodojo-device-binding.v1", **observed,
            "selected_physical_index": selected_index, "selected_gpu_uuid": candidates[0]["uuid"],
            "physics_cuda_logical_index": 0, "physics_cuda_uuid": cuda_uuid,
            "selected_pci_bus_id": candidates[0]["pci_bus_id"],
            "renderer_identity_resolution": "unique_nvml_uuid_prefix_and_pci_bus"}


def revision(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode:
        raise RobodojoError(f"source is not a Git checkout: {path}")
    return result.stdout.strip()


def manifest_bindings(config: Mapping[str, Any], task: str) -> dict[str, Any]:
    """Resolve the bindings this worker was configured with, or refuse them.

    The manifest attests what the worker actually ran under. Reporting a
    literal here let a codec-changing cell attest the compat-era default, and
    the divergence surfaced only at seal -- after the episodes had already
    run. Everything the RunPlan may choose is read back from the resolved
    config; everything pinned to the upstream revision is the worker's own
    fact, and a RunPlan that declares something else is refused at startup
    rather than reported back as if it had been measured.
    """

    from ora.environment.benchmarks.robodojo import (
        ACTION_CODECS, CONTROL_HZ, NATIVE_ACTION_LIMITS, PHYSICS_DT_S,
        SENSOR_PROFILE,
    )

    from ora.environment.benchmarks.robodojo_layouts import configured_layout_schedule

    if task not in NATIVE_ACTION_LIMITS:
        raise RobodojoError("unsupported RoboDojo task")
    codec = config.get("action_codec")
    if codec not in ACTION_CODECS:
        raise RobodojoError("unsupported RoboDojo first-stage binding: action_codec")
    # Round 0 is the RunPlan's one layout (layout 0 when unset), or the first
    # layout of a validated per-round schedule.
    schedule = configured_layout_schedule({**config, "task": task})
    first = config.get("layout_id", 0) if schedule is None else schedule["rounds"][0]["layout_id"]
    if type(first) is not int or first < 0:
        raise RobodojoError("unsupported RoboDojo first-stage binding: layout_id")
    selected = {"eval_seed": 0, "layout_id": first, "num_envs": 1, "sensor_profile": SENSOR_PROFILE}
    for field, pinned in selected.items():
        if config.get(field, pinned) != pinned:
            raise RobodojoError(f"unsupported RoboDojo first-stage binding: {field}")
    for field, pinned in (("physics_dt", PHYSICS_DT_S), ("control_hz", CONTROL_HZ)):
        if config.get(field, pinned) != pinned:
            raise RobodojoError(
                f"RunPlan {field} differs from the pinned RoboDojo simulation cadence"
            )
    limit = NATIVE_ACTION_LIMITS[task]
    if config.get("native_action_limit", limit) != limit:
        raise RobodojoError("RoboDojo native action limit differs from the selected task")
    return {
        **selected, "action_codec": codec,
        "physics_dt_s": PHYSICS_DT_S, "control_hz": CONTROL_HZ,
        "official_native_action_limit": limit,
    }


def source_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    import hashlib

    root = Path(config["robodojo_root"]).resolve()
    assets = Path(config["asset_root"]).resolve()
    roots = {
        "robodojo": root, "isaaclab": root / "third_party/IsaacLab",
        "curobo": root / "third_party/curobo", "xpolicylab": root / "XPolicyLab",
        "assets": assets,
    }
    required_keys = {
        "robodojo": "required_revision", "isaaclab": "required_isaaclab_revision",
        "curobo": "required_curobo_revision", "xpolicylab": "required_xpolicylab_revision",
        "assets": "required_asset_revision",
    }
    actual, required = {}, {}
    for name, path in roots.items():
        expected = str(config.get(required_keys[name], ""))
        if len(expected) != 40 or any(char not in "0123456789abcdef" for char in expected):
            raise RobodojoError(f"missing exact source pin: {required_keys[name]}")
        actual[name], required[name] = revision(path), expected
        if actual[name] != expected:
            raise RobodojoError(f"source revision mismatch: {name}")
        # Ignore untracked caches/assets; tracked source modifications alter semantics.
        dirty = subprocess.run(
            ["git", "-C", str(path), "diff", "HEAD", "--", "."],
            capture_output=True, check=True,
        )
        if dirty.stdout:
            raise RobodojoError(f"tracked source is modified: {name}")
    from ora.environment.benchmarks.robodojo import selected_layout
    task = config.get("task", "arrange_largest_number")
    bindings = manifest_bindings(config, task)
    layout = selected_layout(assets, task=task, eval_seed=0, layout_id=bindings["layout_id"])
    layout_hash = hashlib.sha256(layout.read_bytes()).hexdigest()
    if config.get("layout_sha256") is not None and layout_hash != config["layout_sha256"]:
        raise RobodojoError("selected layout content differs from the RunPlan")
    # Every scheduled round's file is hashed now, so a drifted later layout
    # fails before round 0 rather than at its own round boundary.
    from .layout_rounds import scheduled_layout_files
    scheduled = scheduled_layout_files(assets, task, config.get("layout_schedule"))
    loaded_files = {}
    for relative in ("Robots/x5/robot_config.yml", "Robots/x5/curobo_tmp.yml", "Robots/x5/curobo.yml"):
        path = assets / relative
        if not path.is_file():
            raise RobodojoError(f"required robot configuration is missing: {relative}")
        loaded_files["Assets/" + relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    template = (assets / "Robots/x5/curobo_tmp.yml").read_text()
    expected_curobo = template.replace("${ASSETS_PATH}", str(root)).replace("$ASSETS_PATH", str(root))
    if (assets / "Robots/x5/curobo.yml").read_text() != expected_curobo:
        raise RobodojoError("derived CuRobo configuration differs from exact path-template expansion")
    for relative in ("env_cfg/arx_x5.yml", "env_cfg/sim/sim_config.yml", "env_cfg/scene/default.yml",
                     "env_cfg/robot/dual_x5.yml", "env_cfg/camera/camera_config.yml",
                     f"task/RoboDojo/config/{task}.yml", "task/RoboDojo/config/_task.yml",
                     "third_party/IsaacLab/apps/isaaclab.python.kit"):
        loaded_files[relative] = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    packages = {}
    for package in ("isaacsim", "torch", "numpy", "warp-lang", "nvidia-curobo", "starlette"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    if not str(packages["isaacsim"]).startswith("5.1.") or sys.version_info[:2] != (3, 11):
        raise RobodojoError("the isolated worker requires Python 3.11 and Isaac Sim 5.1")
    return {
        "schema": "ora.robodojo-source-manifest.v1",
        "task": task, **bindings,
        "source_revisions": actual, "required_revisions": required,
        "layout_file": {"path": str(layout), "sha256": layout_hash},
        **({} if scheduled is None else {"layout_schedule": scheduled}),
        "loaded_files": loaded_files, "installed_packages": packages,
    }


def _prepare_cuda_garments(manager: Any, num_envs: int) -> None:
    """Author cloth before the first physics start and retain its particle view.

    Upstream creates CUDA garments during reload, after PhysX has initialized;
    Isaac Sim 5.1 then cannot bind the particle-cloth tensor. Rigid/articulation
    objects already use retained prims upstream. Apply that creation ordering to
    garments while keeping their original initialize/apply_saved_pose methods,
    asset bytes, physical parameters and task evaluator.
    """
    spawn, clear = manager.spawn_scene_objects, manager.clear_scene_objects
    for env_id in range(num_envs):
        spawn(env_id=env_id, exclude_types=[kind for kind in manager.spawnable_object_types if kind != "garment"])
        for garment in manager._garment_objects[env_id].values():
            original_sample = garment.sample_mesh_vertices

            def sample_with_pose(sample: Any = original_sample, obj: Any = garment) -> Any:
                world_points, mesh_points, position, orientation = sample()
                # Upstream's CUDA branch returns None for the transform while
                # its original evaluator needs the garment-local axes. Supply
                # the same USD prim pose used by the CPU branch; keep particle
                # positions, metadata, thresholds and evaluator code intact.
                if position is None or orientation is None:
                    measured_position, measured_orientation = obj.get_world_pose()
                    if position is None:
                        position = measured_position
                    if orientation is None:
                        orientation = measured_orientation
                return world_points, mesh_points, position, orientation

            garment.sample_mesh_vertices = sample_with_pose

    def retained_spawn(env_id: int, exclude_types: Any = None) -> Any:
        return spawn(env_id=env_id, exclude_types=[*(exclude_types or []), "garment"])

    def retained_clear(env_id: int, exclude_types: Any = None) -> Any:
        # Unfiltered close still destroys every object, including garments.
        return clear(env_id=env_id, exclude_types=None if exclude_types is None else [*exclude_types, "garment"])

    manager.spawn_scene_objects = retained_spawn
    manager.clear_scene_objects = retained_clear


class EvaluationLifecycle:
    """Mixin placed before the original task class in the worker's MRO."""

    def setup_sim_cfg(self, config: Any) -> None:
        super().setup_sim_cfg(config)
        if self.sim_cfg.wait_for_textures is not True:
            raise RobodojoError("upstream reset texture-wait default changed")
        # TaskEnv creates its render products only AFTER DirectRLEnv.reset.
        # Headless rendering has no viewport to complete asset loading there.
        # Preserve the wait below, after the original capture_manager.reset.
        self.sim_cfg.wait_for_textures = False

    def __init__(self, config: Any, app: Any):
        super().__init__(config, app)
        from env.observation_manager.obs_manager import ObsManager
        from env.seed_manager.seed_manager import SeedManager

        self.eval_cfg = config.eval_cfg
        self.task_name = self.eval_cfg.task_name
        self.eval_seed = int(self.eval_cfg.seed)
        self.success = [True] * self.num_envs
        self.end_flag = [False] * self.num_envs
        self.take_action_cnt = [0] * self.num_envs
        self.unstable_envs: set[int] = set()
        self.obs_manager = ObsManager(
            obs_config=deepcopy(self.eval_cfg.observation), num_envs=self.num_envs,
            dt=self.dt, task_name=self.task_name, description_cfg={"seen": 1},
            seeds_per_env=self.env_seed_list,
        )
        self.scene_manager.layout_manager.replay = True
        self.seed_manager = SeedManager(self.eval_cfg)
        self.seed_manager.init_eval()
        self.reward_registered = False
        #: This scene's published registration, rebuilt by every reset. None
        #: before the first reset, so a read that precedes one fails closed.
        self.scene_registration: dict[str, Any] | None = None
        self._official_terminal: dict[str, Any] | None = None

    def _post_setup_scene(self, sim: Any) -> None:
        super()._post_setup_scene(sim)
        if self.task_name == "fold_clothes":
            _prepare_cuda_garments(self.scene_manager, self.num_envs)
        self.obs_manager.initialize(self)

    def _release_capture_products(self) -> None:
        """Drop the previous round's render products before upstream recreates them.

        TiledCaptureManager.reset appends three new CameraViews on every
        reset and never releases the old ones (its own cleanup targets an
        unused tiled product), so a second reset of the same plant would
        leave six views and stale products behind the three-camera invariant.
        """
        manager = self.capture_manager
        for view in manager.tiled_cameras:
            path = view._render_product_path
            for name in ("_harness_camera_params", "_harness_simulation_time", "_harness_reference_time"):
                annotator = getattr(view, name, None)
                if annotator is not None:
                    annotator.detach([path])
            for annotator in getattr(view, "_annotators", {}).values():
                annotator.detach([path])
        for product in manager.tiled_render_products:
            product.destroy()
        manager.tiled_render_products.clear()
        manager.tiled_cameras.clear()
        manager.annotator.clear()
        manager.annotator_type.clear()
        manager.annotator_device.clear()
        manager._output_buffers.clear()

    def _keep_layout_identities(self) -> None:
        """Route a later reset of the same plant through upstream's first-reset path.

        SceneManager.reload_env_scene reloads the saved layout whenever
        ``setup_scene`` is False and regenerates every instance name from
        LayoutManager's monotone id counter, while the CUDA pipeline never
        respawns rigid or articulation objects: they stay registered under
        their round-0 names, so the reloaded records would name objects the
        official evaluator and the layout stability check cannot resolve.
        The session's layout is fixed, so the records bound at initialization
        stay exact; the flag makes upstream reuse them like its first reset.
        """
        self.scene_manager.setup_scene = True

    def _verify_scene_identities(self) -> dict[str, str]:
        """Require every registered layout instance to resolve, and name them.

        The returned name/kind map is this scene's identity. It is what a
        repeated reset of the same plant is compared on, because upstream
        ``SceneManager.reload_env_scene`` regenerates instance names from
        LayoutManager's monotone counter and ``_keep_layout_identities`` is
        the workaround; publishing the map is what lets a task qualification
        record show the workaround actually held for *this* scene.
        """
        from ora.environment.benchmarks.robodojo import (
            SCENE_TRACKED_INSTANCE_KINDS,
        )

        layout = self.scene_manager.layout_manager
        # Fluid objects are disabled upstream on CUDA and never registered.
        instances = {
            name: kind for name, kind in layout.instance_type_by_env[0].items()
            if kind in SCENE_TRACKED_INSTANCE_KINDS
        }
        missing = sorted(
            name for name in instances if layout.get_scene_object(0, name) is None
        )
        if missing:
            raise RobodojoError("layout instances have no scene object after reset: " + ", ".join(missing))
        return {str(name): str(kind) for name, kind in instances.items()}

    def reset(self, seed: Any = None, options: Any = None) -> None:
        # The per-environment seed is the layout ordinal, as upstream evaluation
        # binds it; only a layout the session scheduled is admitted.
        layouts = tuple(getattr(self, "harness_layout_ids", (0,)))
        if (not isinstance(seed, list) or len(seed) != 1 or type(seed[0]) is not int
                or seed[0] not in layouts or self.num_envs != 1):
            raise RobodojoError("only a scheduled evaluation-seed-0 layout of the selected task is qualified")
        live = getattr(self, "sim", None) is not None
        if live and seed[0] != getattr(self, "harness_loaded_layout_id", seed[0]):
            # An in-place reload keeps the round-0 objects (CUDA never respawns
            # rigid or articulation objects), so a new layout needs a relaunch.
            raise RobodojoError("a different layout needs relaunch_for_layout, not an in-place reset")
        capture_manager = getattr(self, "capture_manager", None)
        if capture_manager is not None and capture_manager.tiled_cameras:
            self._release_capture_products()
        if live:
            self._keep_layout_identities()
        self.success, self.end_flag, self.take_action_cnt = [True], [False], [0]
        self.unstable_envs = set()
        self.reward_registered = False
        self.scene_registration = None
        self._official_terminal = None
        self.env_seeds = list(seed)
        self.harness_loaded_layout_id = seed[0]
        self.scene_manager.layout_manager.set_saved_layout(0, self.seed_manager.get_seed_scene_info(seed[0]))
        # The task reset clears all reward registrations and predicate history.
        super().reset(seed=seed, options=options)
        instances = self._verify_scene_identities()
        self.obs_manager.reset()
        self.setup_scene()
        self.robot_manager.set_origin_endpose()
        self.robot_manager.set_robot_init_state()
        self.reward_manager.init_state()
        self.run_reward()
        if hasattr(self, "get_score"):
            self.get_score()
        checks = self.reward_manager
        if not any((checks.check_list[0], checks.final_check_list[0], checks.trigger_check_list[0])):
            raise RobodojoError("official task did not register success checks")
        # Publish this scene's registration beside the reset: which layout
        # instances exist and resolved, which official evaluator check lists
        # the task registered, and the native-action limit the environment
        # itself reports. One boot establishes all of it, and it is the entire
        # per-scene surface a task qualification record rests on.
        from ora.environment.benchmarks.robodojo import (
            build_scene_registration,
        )

        self.scene_registration = build_scene_registration(
            instances,
            evaluator_registration={
                "check_list": bool(checks.check_list[0]),
                "final_check_list": bool(checks.final_check_list[0]),
                "trigger_check_list": bool(checks.trigger_check_list[0]),
            },
            native_action_limit=int(self.step_lim),
        )
        self.reward_registered = True
        if getattr(self, "interact", False) and hasattr(self, "query_support_arm_traj"):
            self.query_support_arm_traj(env_idx=0)

    def relaunch_for_layout(self, layout_id: int) -> dict[str, Any]:
        """Start a scheduled layout as a new episode inside the same Kit process.

        An in-place reload cannot change objects on CUDA, so this takes
        upstream's own between-batch path (``main.py``: ``env.close()`` then
        ``env.reset(seed)``): drop the Harness capture annotators first (the
        upstream destroy does not detach them), run the TaskEnv close body --
        managers, timeline, ``sim.close()`` and a new USD stage, never the app
        -- and reset on the layout, which launches a new simulation context
        whose scene manager loads that saved layout. The robot manager and its
        CuRobo planners survive; cameras and render products are rebuilt.
        fold_clothes is refused: its garment preparation wraps scene-manager
        and garment methods in ``_post_setup_scene`` and is not relaunch-safe.
        """
        import time

        layouts = tuple(getattr(self, "harness_layout_ids", ()))
        if type(layout_id) is not int or layout_id not in layouts:
            raise RobodojoError("relaunch_for_layout accepts only a scheduled layout")
        if getattr(self, "task_name", None) == "fold_clothes":
            raise RobodojoError("fold_clothes garment preparation is not relaunch-safe")
        if getattr(self, "sim", None) is None:
            raise RobodojoError("there is no live simulation to relaunch")
        started = time.monotonic()
        previous = getattr(self, "harness_loaded_layout_id", None)
        clock_before = int(self.sim.sim.current_time_step_index)
        # Read at close from the context's own clock, not from the last
        # observation: physics steps can follow it.
        time_before = float(self.sim.sim.current_time)
        previous_offset = float(getattr(self, "harness_simulation_time_offset_s", None) or 0.0)
        stage_before = _stage_state()
        capture_manager = getattr(self, "capture_manager", None)
        if capture_manager is not None and capture_manager.tiled_cameras:
            self._release_capture_products()
        self.obs_manager.reset()
        super().close()
        stage_after = _stage_state()
        # BaseEnv.close prints and continues when close_stage fails, so the
        # outcome is measured: no simulation, and a stage without the old scene.
        sim_closed = getattr(self, "sim", None) is None
        stage_replaced = bool(stage_before["env_prim_present"]) and not stage_after["env_prim_present"]
        if not (sim_closed and stage_replaced):
            raise RobodojoError("the previous simulation or its USD scene survived the relaunch close")
        # The camera annotators' IsaacReadSimulationTime keeps counting across
        # the relaunch while the new SimulationContext clock restarts at 0
        # (smoke 2026-10-01: 3.536 s + 3.376 s read as 6.912 s), so the new
        # epoch's annotator time is offset by every earlier epoch's total.
        self.harness_simulation_time_offset_s = previous_offset + time_before
        self.reset(seed=[layout_id])
        return {"schema": "ora.robodojo-simulation-relaunch.v1",
                "previous_layout_id": previous, "layout_id": layout_id,
                "sim_closed": sim_closed, "stage_replaced": stage_replaced,
                "stage_before": stage_before, "stage_after_close": stage_after,
                "physics_clock_before_close": clock_before,
                "physics_clock_after_reset": int(self.sim.sim.current_time_step_index),
                "simulation_time_before_close_s": time_before,
                "previous_simulation_time_offset_s": previous_offset,
                "simulation_time_offset_s": self.harness_simulation_time_offset_s,
                "simulation_time_after_reset_s": float(self.sim.sim.current_time),
                "wall_s": round(time.monotonic() - started, 3), "cuda_memory": _cuda_memory()}

    def loaded_layout_state(self) -> dict[str, Any]:
        """The saved layout and env seed the live scene was built from, plus its object counts."""
        manager = self.scene_manager
        return {"env_seed": int(self.env_seeds[0]),
                "layout": deepcopy(manager.layout_manager.saved_layouts[0]),
                "object_counts": {"rigid": len(manager._rigid_and_dynamic_objects[0]),
                                  "articulation": len(manager._articulation_objects[0]),
                                  "garment": len(manager._garment_objects[0]),
                                  "geometry": len(manager._geometry_objects[0])}}

    def setup_scene(self) -> None:
        from .camera_capture import attach_camera_parameters, install_single_environment_readback, wait_for_render_assets
        from isaacsim.core.simulation_manager import SimulationManager
        import omni.usd

        context = omni.usd.get_context()
        attach_camera_parameters(self.capture_manager)
        # A persistent session resets the same plant once per round; each
        # reset keeps its own readiness record instead of reopening the first.
        round_index = int(getattr(self, "harness_round_index", 0))
        wait_for_render_assets(
            manager=self.capture_manager, render=self.render,
            assets_loading=SimulationManager.assets_loading,
            stage_status=lambda: {"loading": list(context.get_stage_loading_status()),
                                  "streaming": bool(context.get_stage_streaming_status())},
            physics_clock=lambda: int(self.sim.sim.current_time_step_index),
            output_root=self.harness_output_root,
            record_name=("render-readiness.jsonl" if round_index == 0
                         else f"render-readiness-round-{round_index:04d}.jsonl"),
        )
        # The original NumPy predicates consume CPU arrays. Physics continues
        # using the unwrapped manager; only the evaluator's private input is
        # copied from CUDA without changing values or predicate definitions.
        self.reward_manager.func_parser.layout_manager = EvaluatorLayoutInput(self.scene_manager.layout_manager)

        install_single_environment_readback(self.capture_manager, self.harness_output_root)
        self.scene_manager.apply_saved_poses(env_idx_list=[0])
        if not self.scene_manager.layout_manager.layout_valid[0]:
            raise RobodojoError("official layout is invalid")
        stable, unstable = self.scene_manager.layout_manager.check_layout_stability(self)
        if not stable or 0 in unstable:
            raise RobodojoError("official layout did not settle stably")
        for _ in range(10):
            self.render()
        for index in range(200):
            self.sim_step()
            if index % 5 == 0:
                self.render()
                self.obs_manager.get_obs()

    def get_obs(self) -> dict[str, Any]:
        from .camera_capture import physical_camera_pose, synchronize_render_products

        # Set only by a layout-scheduled session (relaunch_for_layout).
        offset = getattr(self, "harness_simulation_time_offset_s", None)
        synchronize_render_products(
            manager=self.capture_manager, render=self.render,
            physics_clock=lambda: int(self.sim.sim.current_time_step_index),
            simulation_time=lambda: float(self.sim.sim.current_time),
            physical_poses=lambda: [physical_camera_pose(camera, use_fabric=self.use_fabric)
                                    for camera in self.camera_manager.cameras[0]],
            output_root=self.harness_output_root,
            **({} if offset is None else {"simulation_time_offset_s": offset}),
        )
        return deepcopy(self.obs_manager.get_obs(env_idx_list=[0])[0])

    def mark_env_unstable(self, env_idx: int) -> None:
        self.unstable_envs.add(env_idx)

    def advance_official_terminal(self) -> dict[str, Any] | None:
        if not self.reward_registered:
            raise RobodojoError("official reward lifecycle has not been initialized")
        if self._official_terminal is not None:
            return deepcopy(self._official_terminal)
        final = self.take_action_cnt[0] >= self.step_lim or not self.success[0]
        reward = self.reward_manager.get_reward(final_check=final)[0]
        if reward > 1 - 1e-3:
            self.end_flag[0], self.success[0] = True, True
        elif final:
            self.end_flag[0], self.success[0] = True, False
        if self.end_flag[0]:
            self._official_terminal = {"official_success": bool(self.success[0]), "native_actions": self.take_action_cnt[0]}
        return deepcopy(self._official_terminal)

    def native_tick(self, control: Mapping[str, Any]) -> dict[str, Any] | None:
        if self._official_terminal is not None or self.end_flag[0]:
            raise RobodojoError("an officially terminated episode cannot accept an action")
        if not self.reward_registered:
            raise RobodojoError("reward registration is required before action execution")
        if self.take_action_cnt[0] >= self.step_lim:
            raise RobodojoError("native action limit was reached without a terminal transition")
        self.take_action_cnt[0] += 1
        sequence = self.interpolate(control)
        manager = self.robot_manager.control_manager
        manager.push([0], [sequence])
        for requested in sequence:
            previous = deepcopy(manager.prev_control[0])
            commands = manager.pop([0])
            commands[0], resolved = _fixed_meta_control(commands[0], self.robot_manager, previous, requested)
            manager.prev_control[0] = resolved
            super().step(meta_control_list=commands)
            self.sim_step(render=False)
            capture = getattr(self, "execution_capture_callback", None)
            if capture is not None:
                capture()
            callback = getattr(self, "controller_trace_callback", None)
            if callback is not None:
                callback(deepcopy(manager.prev_control[0]))
        if not manager.get_empty([0]):
            raise RobodojoError("native control queue did not drain at the fixed cadence")
        self.reward_manager.step(env_idx_list=[0])
        if getattr(self, "interact", False):
            if hasattr(self, "query_support_arm_traj"):
                self.query_support_arm_traj(env_idx=0)
            if hasattr(self, "check_support_arm_stable"):
                self.check_support_arm_stable(env_idx=0)
        if self.unstable_envs:
            raise RobodojoError("official environment became unstable")
        return self.advance_official_terminal()

    def interpolate(self, control: Mapping[str, Any]) -> list[dict[str, Any]]:
        count = int(self.obs_manager.collect_interval)
        if count != 10:
            raise RobodojoError("native control cadence must be 10 physics steps")
        sequence = [deepcopy(dict(control)) for _ in range(count)]
        for robot in self.robot_manager.robot_list:
            if robot.type != "target" and hasattr(self, "support_arm_action"):
                for step in sequence:
                    if self.support_arm_action[0]:
                        step.update(self.support_arm_action[0].pop(0))
            for key, gripper in ((self.robot_manager.process_name(robot.arm_name), False),
                                 (self.robot_manager.process_name(robot.gripper_name), True)):
                if key not in control:
                    continue
                if gripper:
                    current = self.robot_manager.get_end_effector_real_val(robot, env_idx_list=[0])[0][0]
                    target = control[key]["position"][0]
                else:
                    current = np.asarray(self.robot_manager.get_joint(robot, env_idx_list=[0])[0])
                    target = np.asarray(control[key]["position"])
                for index in range(count):
                    value = ((1 - (index + 1) / 9) * current + (index + 1) / 9 * target
                             if index < 8 else target)
                    if gripper:
                        value = float(np.clip(value, *robot.gripper_scale))
                        sequence[index][key]["position"] = [value, value * robot.gripper_move["mimic"][1] + robot.gripper_move["mimic"][2]]
                    else:
                        sequence[index][key]["position"] = np.asarray(value).tolist()
        return sequence


def _stage_state() -> dict[str, Any]:
    """The context stage's identity and whether the environment prims exist.

    A relaunch must close a stage that holds ``/World/envs/env_0`` and leave
    one that does not. The cache id is read, never inserted, so measuring the
    old stage cannot keep it alive.
    """
    import omni.usd
    from pxr import UsdUtils

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return {"stage": None, "env_prim_present": False}
    stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
    return {"stage": f"{stage_id}:{stage.GetRootLayer().identifier}",
            "env_prim_present": bool(stage.GetPrimAtPath("/World/envs/env_0").IsValid())}


def _cuda_memory() -> dict[str, int]:
    """This worker's CUDA memory after a relaunch, so a per-round leak is measured."""
    import torch

    free, total = torch.cuda.mem_get_info()
    return {"allocated_mib": int(torch.cuda.memory_allocated() // 2**20),
            "reserved_mib": int(torch.cuda.memory_reserved() // 2**20),
            "device_free_mib": int(free // 2**20), "device_total_mib": int(total // 2**20)}


def build_environment(config: Mapping[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    from .bootstrap import build_environment as build
    return build(config)
