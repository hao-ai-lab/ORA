"""Fail-closed adapter for the BDDL-driven LIBERO-Pro robustness cases.

The perturbation algorithms belong to the pinned LIBERO-Pro source checkout.
This module binds those algorithms to Robot Harness' one-environment RLinf
worker lifecycle and validates the dataset-authored configuration before any
simulator state is changed.
"""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping
from xml.etree import ElementTree

from ora.environment.benchmarks.libero_pro import (
    BASE_SUITES as ROBUSTNESS_BASE_SUITES,
    LIBERO_PRO_CODE_REVISION as ROBUSTNESS_CODE_REVISION,
    ROBUSTNESS_CASE_FOLDERS,
    ROBUSTNESS_CASES,
    ROBUSTNESS_SCHEMA,
)

ROBUSTNESS_EVENT_SCHEMA = "ora.libero-pro-robustness-event.v1"

_TOP_LEVEL_FIELDS = {
    "case", "category", "source_suite", "source_task", "bddl_only", "detail",
    "lighting", "observation", "camera", "runtime_move", "object_appearance",
    "object_placement", "object_shape", "initial_pose",
}
_SECTION_FIELDS = {
    "lighting": {"enabled", "mode", "ambient", "diffuse", "specular"},
    "observation": {"enabled", "mode", "brightness", "sigma"},
    "camera": {"enabled", "mode", "name", "pos", "quat"},
    "runtime_move": {
        "enabled", "mode", "object", "step", "trigger", "distance_threshold",
        "fallback_step", "delta_xy", "min_delta_norm",
    },
    "object_appearance": {
        "enabled", "mode", "targets", "target", "rgba",
        "override_material_rgba", "disable_material_texture",
        "apply_visible_geom_rgba",
    },
    "object_placement": {
        "enabled", "mode", "reference", "front_direction", "targets", "target",
        "offsets_xy", "target_quat", "target_quats", "z_offset", "z_offsets",
        "mesh_scale", "min_reference_distance", "pin_targets", "avoid_targets",
        "avoid_target", "forbidden_objects", "min_avoid_distance",
        "between_object", "between_delta_xy", "min_between_distance",
    },
    "object_shape": {"enabled", "mode", "target", "targets", "mesh_scale"},
    "initial_pose": {
        "enabled", "mode", "target", "delta_xy", "min_delta_norm", "yaw",
    },
}
_ACTIVE_SECTIONS = {
    "visual_noise_glare": {
        "lighting": "dark_lighting", "observation": "dark_noise",
    },
    "camera_view_angle": {"camera": "agentview_angle"},
    "runtime_object_move": {"runtime_move": "qpos_delta_xy"},
    "object_texture": {"object_appearance": "table_like_material"},
    "view_occlusion": {"object_placement": "front_occlusion"},
    "object_shape": {"object_shape": "mesh_scale"},
    "initial_pose_position_angle": {"initial_pose": "delta_xy_yaw"},
}


class LiberoProRobustnessError(RuntimeError):
    """A robustness config or application violated the pinned contract."""


class RobustnessEpisode:
    """Drive and validate one worker-side robustness lifecycle over RLinf RPC."""

    def __init__(self, case: str) -> None:
        if case not in ROBUSTNESS_CASES:
            raise LiberoProRobustnessError(
                f"unsupported LIBERO robustness case: {case!r}"
            )
        self.case = case
        self.event_sequence = 0

    @staticmethod
    def _transport(environment: Any) -> Any:
        transport = getattr(environment, "env", None)
        if transport is None:
            raise LiberoProRobustnessError(
                "LIBERO robustness worker transport is unavailable"
            )
        return transport

    def state(self, environment: Any) -> dict[str, Any]:
        getter = getattr(self._transport(environment), "get_env_attr", None)
        if not callable(getter):
            raise LiberoProRobustnessError(
                "LIBERO robustness state transport is unavailable"
            )
        values = getter("harness_robustness_state")
        state = values[0] if isinstance(values, list) and len(values) == 1 else None
        if not isinstance(state, Mapping):
            raise LiberoProRobustnessError(
                "LIBERO robustness worker returned invalid state"
            )
        return dict(state)

    def start(self, environment: Any) -> dict[str, Any]:
        state = self.state(environment)
        if (
            state.get("case") != self.case
            or state.get("static_applied") is not True
            or state.get("official_predicate_changed") is not False
        ):
            raise LiberoProRobustnessError(
                "LIBERO robustness application did not satisfy its reset contract"
            )
        setter = getattr(self._transport(environment), "set_env_attr", None)
        if not callable(setter):
            raise LiberoProRobustnessError(
                "LIBERO robustness arming transport is unavailable"
            )
        setter("harness_robustness_armed", True)
        state = self.state(environment)
        if state.get("armed") is not True:
            raise LiberoProRobustnessError("LIBERO robustness runtime did not arm")
        return state

    def poll_event(self, environment: Any) -> dict[str, Any] | None:
        event = self.state(environment).get("runtime_event")
        if event is None:
            return None
        if not isinstance(event, Mapping):
            raise LiberoProRobustnessError(
                "LIBERO robustness runtime event is malformed"
            )
        sequence = event.get("sequence")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence <= 0:
            raise LiberoProRobustnessError(
                "LIBERO robustness event has invalid sequence"
            )
        if sequence <= self.event_sequence:
            return None
        if sequence != self.event_sequence + 1:
            raise LiberoProRobustnessError(
                "LIBERO robustness event sequence is not contiguous"
            )
        self.event_sequence = sequence
        return dict(event)


def _json_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _joint_layout(model: Any) -> list[dict[str, Any]]:
    """Capture one complete named MuJoCo qpos/qvel layout."""

    layout = []
    names: set[str] = set()
    joint_count = int(model.njnt)
    for joint_id in range(joint_count):
        name = model.joint_id2name(joint_id)
        if not name or str(name) in names:
            raise LiberoProRobustnessError(
                "MuJoCo state layout contains an unnamed or duplicate joint"
            )
        name = str(name)
        names.add(name)
        qpos_address = int(model.jnt_qposadr[joint_id])
        qvel_address = int(model.jnt_dofadr[joint_id])
        qpos_end = (
            int(model.jnt_qposadr[joint_id + 1])
            if joint_id + 1 < joint_count else int(model.nq)
        )
        qvel_end = (
            int(model.jnt_dofadr[joint_id + 1])
            if joint_id + 1 < joint_count else int(model.nv)
        )
        if qpos_end <= qpos_address or qvel_end <= qvel_address:
            raise LiberoProRobustnessError("MuJoCo joint state width is invalid")
        layout.append({
            "joint": name,
            "qpos_address": qpos_address,
            "qpos_width": qpos_end - qpos_address,
            "qvel_address": qvel_address,
            "qvel_width": qvel_end - qvel_address,
        })
    return layout


def robustness_bddl_path(
    data_root: Path, *, case: str, suite: str, task_name: str,
) -> Path:
    """Resolve one exact dataset-authored robustness BDDL path."""

    if case not in ROBUSTNESS_CASE_FOLDERS:
        raise LiberoProRobustnessError(f"unsupported robustness case: {case!r}")
    if suite not in ROBUSTNESS_BASE_SUITES:
        raise LiberoProRobustnessError(
            f"robustness suite must be a base LIBERO suite, got {suite!r}"
        )
    if not task_name or Path(task_name).name != task_name:
        raise LiberoProRobustnessError("robustness task name is not a stable filename")
    path = (
        data_root.resolve() / "bddl_files" / ROBUSTNESS_CASE_FOLDERS[case]
        / "bddl" / suite / f"{task_name}.bddl"
    )
    if not path.is_file():
        raise LiberoProRobustnessError(f"robustness BDDL is unavailable: {path}")
    return path


def _number(value: Any, field: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LiberoProRobustnessError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise LiberoProRobustnessError(f"{field} is outside its finite range")
    return result


def _vector(
    value: Any, field: str, length: int, *, positive: bool = False,
) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise LiberoProRobustnessError(f"{field} must contain {length} values")
    result = [_number(item, field) for item in value]
    if positive and any(item <= 0 for item in result):
        raise LiberoProRobustnessError(f"{field} values must be positive")
    return result


def _names(value: Any, field: str) -> list[str]:
    values = value if isinstance(value, list) else [value]
    if not values or any(not isinstance(item, str) or not item for item in values):
        raise LiberoProRobustnessError(f"{field} must name at least one scene entity")
    return values


def validate_robustness_config(
    config: Mapping[str, Any], *, case: str, suite: str, task_name: str,
) -> dict[str, Any]:
    """Validate one closed seven-case BDDL config and return canonical JSON.

    Unknown fields, case drift and inactive-section drift are rejected. This is
    deliberately stricter than the upstream convenience parser, which accepts
    defaults useful for authoring but unsafe for a benchmark claim.
    """

    value = dict(config)
    if set(value) != _TOP_LEVEL_FIELDS:
        raise LiberoProRobustnessError(
            "robustness config fields differ: "
            f"missing={sorted(_TOP_LEVEL_FIELDS - set(value))}, "
            f"extra={sorted(set(value) - _TOP_LEVEL_FIELDS)}"
        )
    if case not in ROBUSTNESS_CASES or value["case"] != case or value["category"] != case:
        raise LiberoProRobustnessError("robustness case/category binding drifted")
    if (
        value["source_suite"] != suite
        or not isinstance(value["source_task"], str)
        or value["source_task"].casefold() != task_name.casefold()
    ):
        raise LiberoProRobustnessError("robustness source suite/task binding drifted")
    if value["bddl_only"] is not True:
        raise LiberoProRobustnessError("robustness BDDL must declare bddl_only=true")
    if not isinstance(value["detail"], str) or not value["detail"].strip():
        raise LiberoProRobustnessError("robustness detail must be non-empty")

    expected_active = _ACTIVE_SECTIONS[case]
    for name, allowed_fields in _SECTION_FIELDS.items():
        section = value.get(name)
        if not isinstance(section, Mapping):
            raise LiberoProRobustnessError(f"robustness section {name} must be an object")
        section = dict(section)
        extra = set(section) - allowed_fields
        if extra or not {"enabled", "mode"}.issubset(section):
            raise LiberoProRobustnessError(
                f"robustness section {name} has invalid fields: {sorted(extra)}"
            )
        active = name in expected_active
        if section["enabled"] is not active:
            raise LiberoProRobustnessError(f"robustness section {name} enabled state drifted")
        expected_mode = expected_active.get(name, "none")
        if section["mode"] != expected_mode:
            raise LiberoProRobustnessError(
                f"robustness section {name} mode must be {expected_mode!r}"
            )

    if case == "visual_noise_glare":
        lighting = value["lighting"]
        for field in ("ambient", "diffuse", "specular"):
            _number(lighting[field], f"lighting.{field}", minimum=0.0)
        observation = value["observation"]
        brightness = _number(observation["brightness"], "observation.brightness")
        if not 0 < brightness <= 1:
            raise LiberoProRobustnessError("observation.brightness must be in (0,1]")
        _number(observation["sigma"], "observation.sigma", minimum=0.0)
    elif case == "camera_view_angle":
        camera = value["camera"]
        if camera.get("name") != "agentview":
            raise LiberoProRobustnessError("camera perturbation may change only agentview")
        _vector(camera.get("pos"), "camera.pos", 3)
        quat = _vector(camera.get("quat"), "camera.quat", 4)
        if math.sqrt(sum(item * item for item in quat)) <= 1e-8:
            raise LiberoProRobustnessError("camera.quat must be non-zero")
    elif case == "runtime_object_move":
        runtime = value["runtime_move"]
        _names(runtime.get("object"), "runtime_move.object")
        if runtime.get("trigger") != "near_grasp":
            raise LiberoProRobustnessError("runtime move trigger must be near_grasp")
        threshold = _number(
            runtime.get("distance_threshold"), "runtime_move.distance_threshold",
            minimum=0.0,
        )
        if not 0 < threshold <= 0.09:
            raise LiberoProRobustnessError(
                "runtime move distance threshold must be in (0,0.09] m"
            )
        if runtime.get("fallback_step") != 160:
            raise LiberoProRobustnessError("runtime move fallback_step must be 160")
        _vector(runtime.get("delta_xy"), "runtime_move.delta_xy", 2)
        _number(runtime.get("min_delta_norm"), "runtime_move.min_delta_norm", minimum=0.0)
    elif case == "object_texture":
        appearance = value["object_appearance"]
        _names(appearance.get("targets", appearance.get("target")), "object_appearance.targets")
        _vector(appearance.get("rgba"), "object_appearance.rgba", 4)
        for field in (
            "override_material_rgba", "disable_material_texture",
            "apply_visible_geom_rgba",
        ):
            if not isinstance(appearance.get(field), bool):
                raise LiberoProRobustnessError(f"object_appearance.{field} must be Boolean")
    elif case == "view_occlusion":
        placement = value["object_placement"]
        _names(placement.get("targets", placement.get("target")), "object_placement.targets")
        _names(placement.get("reference"), "object_placement.reference")
        _vector(placement.get("front_direction"), "object_placement.front_direction", 2)
        offsets = placement.get("offsets_xy")
        if offsets and isinstance(offsets, list) and not isinstance(offsets[0], list):
            offsets = [offsets]
        if not isinstance(offsets, list) or not offsets:
            raise LiberoProRobustnessError("object_placement.offsets_xy must be non-empty")
        for offset in offsets:
            _vector(offset, "object_placement.offsets_xy", 2)
        _vector(placement.get("mesh_scale"), "object_placement.mesh_scale", 3, positive=True)
        if placement.get("pin_targets") is not True:
            raise LiberoProRobustnessError("view occluders must declare pin_targets=true")
    elif case == "object_shape":
        shape = value["object_shape"]
        _names(shape.get("targets", shape.get("target")), "object_shape.target")
        _vector(shape.get("mesh_scale"), "object_shape.mesh_scale", 3, positive=True)
    elif case == "initial_pose_position_angle":
        pose = value["initial_pose"]
        _names(pose.get("target"), "initial_pose.target")
        _vector(pose.get("delta_xy"), "initial_pose.delta_xy", 2)
        _number(pose.get("min_delta_norm"), "initial_pose.min_delta_norm", minimum=0.0)
        _number(pose.get("yaw"), "initial_pose.yaw")

    # A JSON round trip rejects parser-specific objects and fixes the exact
    # representation hashed into Run evidence.
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise LiberoProRobustnessError("robustness config is not canonical JSON") from exc


def parse_and_validate_robustness_bddl(
    bddl_path: Path, *, case: str, suite: str, task_name: str,
) -> dict[str, Any]:
    """Use the pinned upstream parser, then enforce the Harness contract."""

    from libero.libero.envs.perturbation_config import (
        parse_bddl_perturbation_config,
    )

    config = parse_bddl_perturbation_config(bddl_path)
    if not isinstance(config, Mapping) or not config:
        raise LiberoProRobustnessError(
            f"BDDL has no parsed perturbation_config: {bddl_path}"
        )
    return validate_robustness_config(
        config, case=case, suite=suite, task_name=task_name,
    )


class RobustnessEnvironment:
    """Apply one validated BDDL perturbation inside the environment worker."""

    def __init__(
        self, delegate: Any, *, case: str, suite: str, task_name: str,
        bddl_path: Path, seed: int,
    ) -> None:
        from libero.libero.envs import robustness_perturbations as runtime

        self._delegate = delegate
        self._runtime = runtime
        self._case = case
        self._suite = suite
        self._task_name = task_name
        self._bddl_path = bddl_path.resolve()
        self._seed = int(seed)
        self._config = parse_and_validate_robustness_bddl(
            self._bddl_path, case=case, suite=suite, task_name=task_name,
        )
        self._config_sha256 = _json_hash(self._config)
        self._spec = runtime.infer_spec_from_config(
            case, suite, task_name, self._config,
            bddl_path=self._bddl_path, init_suite=suite,
        )
        self._options = runtime.PerturbationRuntimeOptions()
        self._vec_like = SimpleNamespace(envs=[SimpleNamespace(_env=delegate)])
        self._armed = False
        self._sim_step = 0
        self._observation_sequence = 0
        self._static_applied = False
        self._static_info: dict[str, Any] = {}
        self._lighting_info: Any = None
        self._camera_info: Any = None
        self._runtime_event: dict[str, Any] | None = None
        self._init_state_adaptation: dict[str, Any] | None = None
        self._model_recompile: dict[str, Any] | None = None
        self._source_joint_layout: list[dict[str, Any]] | None = None
        self._recompiled_joint_layout: list[dict[str, Any]] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    @property
    def harness_robustness_armed(self) -> bool:
        return self._armed

    @harness_robustness_armed.setter
    def harness_robustness_armed(self, value: Any) -> None:
        if value is not True:
            raise LiberoProRobustnessError("robustness runtime can only be armed once")
        if self._armed or not self._static_applied:
            raise LiberoProRobustnessError(
                "robustness runtime must be statically applied before one-time arming"
            )
        self._armed = True
        self._sim_step = 0

    @property
    def harness_robustness_state(self) -> dict[str, Any]:
        return {
            "schema": ROBUSTNESS_SCHEMA,
            "case": self._case,
            "suite": self._suite,
            "task_name": self._task_name,
            "bddl_file": str(self._bddl_path),
            "config": self._config,
            "config_sha256": self._config_sha256,
            "armed": self._armed,
            "sim_step": self._sim_step,
            "observation_sequence": self._observation_sequence,
            "static_applied": self._static_applied,
            "application": {
                "static": self._static_info,
                "lighting": self._lighting_info,
                "camera": self._camera_info,
                "init_state_adaptation": self._init_state_adaptation,
                "model_recompile": self._model_recompile,
            },
            "runtime_event": self._runtime_event,
            "official_predicate_changed": False,
        }

    def reset(self) -> Any:
        self._armed = False
        self._sim_step = 0
        self._observation_sequence = 0
        self._static_applied = False
        self._static_info = {}
        self._lighting_info = None
        self._camera_info = None
        self._runtime_event = None
        self._init_state_adaptation = None
        self._model_recompile = None
        self._source_joint_layout = None
        self._recompiled_joint_layout = None
        observation = self._delegate.reset()
        self._model_recompile = self._recompile_configured_mesh_scale()
        if self._model_recompile is not None:
            observation = self._fresh_observation(observation)
        return self._transform_observation(observation)

    def set_init_state(self, init_state: Any) -> Any:
        adapted_state = self._adapt_shared_init_state(init_state)
        observation = self._delegate.set_init_state(adapted_state)
        self._apply_static_once()
        return self._transform_observation(self._fresh_observation(observation))

    def step(self, action: Any) -> Any:
        self._maybe_move_runtime_target()
        observation, reward, done, info = self._delegate.step(action)
        pinned = self._runtime.apply_pinned_object_placements(
            self._vec_like, self._static_info,
        )
        if pinned:
            observation = self._fresh_observation(observation)
        observation = self._transform_observation(observation)
        if self._armed:
            self._sim_step += 1
        return observation, reward, done, info

    def _apply_static_once(self) -> None:
        if self._static_applied:
            raise LiberoProRobustnessError(
                "robustness static configuration was applied more than once"
            )
        runtime = self._runtime
        if self._case == "object_shape":
            if self._model_recompile is None:
                raise LiberoProRobustnessError(
                    "object shape was not compiled into the MuJoCo model"
                )
            self._static_info = {
                "object_appearance": None,
                "object_shape": {
                    "mode": "xml_precompile_mesh_scale_v1",
                    "targets": self._model_recompile["targets"],
                    "mesh_scale_multiplier": self._model_recompile[
                        "mesh_scale_multiplier"
                    ],
                    "changed_meshes": [
                        item["mesh"] for item in self._model_recompile["meshes"]
                    ],
                    "model_recompile": self._model_recompile,
                },
                "object_placement": None,
                "initial_pose": None,
            }
        else:
            application_spec = self._spec
            if self._case == "view_occlusion" and self._model_recompile is not None:
                config = deepcopy(self._config)
                # Mesh scale has already changed both visual and collision
                # geometry during model compilation. Prevent the upstream
                # post-compile helper from reporting a second ineffective
                # model.mesh_scale mutation.
                config["object_placement"]["mesh_scale"] = []
                application_spec = replace(
                    self._spec, perturbation_config=config,
                )
            self._static_info = runtime.apply_static_bddl_config_perturbations(
                self._vec_like, application_spec, self._options,
            )
            if self._case == "view_occlusion" and self._model_recompile is not None:
                placement = self._static_info.get("object_placement")
                if not isinstance(placement, dict):
                    raise LiberoProRobustnessError(
                        "occlusion placement evidence is unavailable"
                    )
                placement["model_recompile"] = self._model_recompile
        self._lighting_info = runtime.apply_sim_lighting(
            self._vec_like, self._spec, self._options,
        ) if runtime.active_config_section(self._config, "lighting") else None
        self._camera_info = runtime.apply_camera_pose(
            self._vec_like, self._spec, self._options,
        ) if runtime.active_config_section(self._config, "camera") else None
        self._static_applied = True

    def _recompile_configured_mesh_scale(self) -> dict[str, Any] | None:
        """Apply BDDL mesh scale in MJCF before MuJoCo compiles geometry.

        Mutating ``model.mesh_scale`` after compilation changes an exposed
        numeric array but does not reliably rebuild rendered vertices or the
        collision acceleration structures. The paired upstream helper does
        that late mutation; the Harness instead recompiles the exact model XML
        and verifies the resulting model values before accepting the case.
        """

        if self._case == "object_shape":
            section = self._config["object_shape"]
        elif self._case == "view_occlusion":
            section = self._config["object_placement"]
        else:
            return None
        raw_scale = section.get("mesh_scale")
        if not raw_scale:
            return None

        import numpy as np

        scale = np.asarray(
            _vector(raw_scale, "mesh_scale", 3, positive=True),
            dtype=np.float64,
        )
        targets = _names(
            section.get("targets", section.get("target")), "mesh targets",
        )
        inner = getattr(self._delegate, "env", None)
        if inner is None or not hasattr(getattr(inner, "model", None), "get_xml"):
            raise LiberoProRobustnessError(
                "mesh-scale application lacks the compiled task model XML"
            )
        source_model = inner.sim.model
        self._source_joint_layout = _joint_layout(source_model)
        source_state = np.asarray(
            self._delegate.get_sim_state(), dtype=np.float64,
        ).copy()
        source_body_transforms = []
        for body_id in range(int(source_model.nbody)):
            body_name = source_model.body_id2name(body_id)
            if not body_name:
                raise LiberoProRobustnessError(
                    "MuJoCo model contains an unnamed body before recompilation"
                )
            source_body_transforms.append({
                "body": str(body_name),
                "pos": np.asarray(source_model.body_pos[body_id]).copy(),
                "quat": np.asarray(source_model.body_quat[body_id]).copy(),
            })
        mesh_names: list[str] = []
        for target in targets:
            mesh_ids = self._runtime.body_mesh_ids(inner, target)
            if not mesh_ids:
                raise LiberoProRobustnessError(
                    f"mesh-scale target has no mesh geometry: {target}"
                )
            for mesh_id in mesh_ids:
                name = inner.sim.model.mesh_id2name(mesh_id)
                if not name:
                    raise LiberoProRobustnessError(
                        f"mesh-scale target has an unnamed mesh: {target}"
                    )
                mesh_names.append(str(name))
        if len(mesh_names) != len(set(mesh_names)):
            raise LiberoProRobustnessError(
                "mesh-scale targets resolve to overlapping mesh assets"
            )

        root = ElementTree.fromstring(inner.model.get_xml())
        asset = root.find("asset")
        if asset is None:
            raise LiberoProRobustnessError("task model XML has no asset section")
        mesh_elements: dict[str, Any] = {}
        for element in asset.findall("mesh"):
            name = element.get("name")
            if name in mesh_names:
                if name in mesh_elements:
                    raise LiberoProRobustnessError(
                        f"task model XML repeats mesh asset: {name}"
                    )
                mesh_elements[str(name)] = element
        if set(mesh_elements) != set(mesh_names):
            raise LiberoProRobustnessError(
                "task model XML is missing a configured target mesh"
            )

        changes = []
        for name in mesh_names:
            element = mesh_elements[name]
            before = np.fromstring(element.get("scale", "1 1 1"), sep=" ")
            if before.shape != (3,) or not np.isfinite(before).all():
                raise LiberoProRobustnessError(
                    f"task model XML has invalid mesh scale: {name}"
                )
            after = before * scale
            element.set("scale", " ".join(f"{value:.17g}" for value in after))
            changes.append({
                "mesh": name,
                "before": before.tolist(),
                "after": after.tolist(),
            })

        self._delegate.reset_from_xml_string(
            ElementTree.tostring(root, encoding="unicode"),
        )
        recompiled_inner = getattr(self._delegate, "env", None)
        recompiled_model = recompiled_inner.sim.model
        self._recompiled_joint_layout = _joint_layout(recompiled_model)
        current_by_joint = {
            item["joint"]: item for item in self._recompiled_joint_layout
        }
        if set(item["joint"] for item in self._source_joint_layout) != set(
            current_by_joint
        ):
            raise LiberoProRobustnessError(
                "mesh-only XML recompilation changed the MuJoCo joint set"
            )
        current_state = np.asarray(
            self._delegate.get_sim_state(), dtype=np.float64,
        ).copy()
        source_nq = int(source_model.nq)
        source_nv = int(source_model.nv)
        if (
            source_state.size != 1 + source_nq + source_nv
            or current_state.size
            != 1 + int(recompiled_model.nq) + int(recompiled_model.nv)
        ):
            raise LiberoProRobustnessError(
                "mesh-only XML recompilation changed state dimensions"
            )
        current_state[0] = source_state[0]
        for source_joint in self._source_joint_layout:
            target_joint = current_by_joint[source_joint["joint"]]
            if (
                source_joint["qpos_width"] != target_joint["qpos_width"]
                or source_joint["qvel_width"] != target_joint["qvel_width"]
            ):
                raise LiberoProRobustnessError(
                    f"mesh-only XML recompilation changed joint width: "
                    f"{source_joint['joint']}"
                )
            source_qpos = 1 + int(source_joint["qpos_address"])
            target_qpos = 1 + int(target_joint["qpos_address"])
            qpos_width = int(source_joint["qpos_width"])
            current_state[target_qpos : target_qpos + qpos_width] = source_state[
                source_qpos : source_qpos + qpos_width
            ]
            source_qvel = 1 + source_nq + int(source_joint["qvel_address"])
            target_qvel = (
                1 + int(recompiled_model.nq) + int(target_joint["qvel_address"])
            )
            qvel_width = int(source_joint["qvel_width"])
            current_state[target_qvel : target_qvel + qvel_width] = source_state[
                source_qvel : source_qvel + qvel_width
            ]
        for transform in source_body_transforms:
            body_id = int(recompiled_model.body_name2id(transform["body"]))
            recompiled_model.body_pos[body_id] = transform["pos"]
            recompiled_model.body_quat[body_id] = transform["quat"]
        self._delegate.set_state(current_state)
        recompiled_inner.sim.forward()
        self._delegate._post_process()
        self._delegate._update_observables(force=True)
        verified = []
        for change in changes:
            mesh_id = int(recompiled_inner.sim.model.mesh_name2id(change["mesh"]))
            actual = np.asarray(
                recompiled_inner.sim.model.mesh_scale[mesh_id], dtype=np.float64,
            )
            if not np.allclose(actual, change["after"], rtol=1e-9, atol=1e-12):
                raise LiberoProRobustnessError(
                    f"recompiled mesh scale differs: {change['mesh']}"
                )
            verified.append(actual.tolist())
        return {
            "mode": "xml_precompile_mesh_scale_v1",
            "targets": targets,
            "mesh_scale_multiplier": scale.tolist(),
            "meshes": [
                {**item, "verified": actual}
                for item, actual in zip(changes, verified, strict=True)
            ],
            "configured_mesh_assets_recompiled": True,
            "source_joint_layout_sha256": _json_hash(self._source_joint_layout),
            "recompiled_joint_layout_sha256": _json_hash(
                self._recompiled_joint_layout
            ),
            "restored_body_transforms": len(source_body_transforms),
            "precompile_reset_state_restored": True,
        }

    def _adapt_shared_init_state(self, init_state: Any) -> Any:
        """Insert reset-authored state for BDDL-only occluder free joints.

        The dataset intentionally shares the clean task init tensors. The
        occlusion BDDL adds one or more free objects, so the clean flattened
        MuJoCo vector is shorter by 7 qpos + 6 qvel values per object. MuJoCo's
        joint order follows the generated task model and therefore the added
        free joint is not guaranteed to be a suffix. Copy the clean qpos/qvel
        values into the complement of the explicitly named occluder ranges and
        retain deterministic BDDL reset values only for those ranges.
        """

        if self._model_recompile is not None:
            return self._remap_recompiled_init_state(init_state)
        if self._case != "view_occlusion":
            self._init_state_adaptation = {
                "mode": "identity", "input_values": len(init_state),
            }
            return init_state
        import numpy as np

        placement = self._config["object_placement"]
        targets = _names(
            placement.get("targets", placement.get("target")),
            "object_placement.targets",
        )
        extra_count = len(targets)
        inner = getattr(self._delegate, "env", None)
        sim = getattr(inner, "sim", None)
        if sim is None:
            raise LiberoProRobustnessError("occlusion init adaptation lacks MuJoCo state")
        model = sim.model
        base_nq = int(model.nq) - 7 * extra_count
        base_nv = int(model.nv) - 6 * extra_count
        source = np.asarray(init_state, dtype=np.float64).reshape(-1)
        if base_nq < 0 or base_nv < 0 or source.size != 1 + base_nq + base_nv:
            raise LiberoProRobustnessError(
                "shared clean init state does not match appended occluder dimensions"
            )
        object_dict = getattr(inner, "objects_dict", {})
        target_joints = []
        target_qpos_indices: set[int] = set()
        target_qvel_indices: set[int] = set()
        for target in targets:
            obj = object_dict.get(target) if isinstance(object_dict, Mapping) else None
            joints = getattr(obj, "joints", None)
            if not joints:
                raise LiberoProRobustnessError(
                    f"occlusion target has no free joint: {target}"
                )
            joint = str(joints[-1])
            joint_id = int(model.joint_name2id(joint))
            if int(model.jnt_type[joint_id]) != 0:
                raise LiberoProRobustnessError(
                    f"occlusion target joint is not MuJoCo free type: {target}"
                )
            qpos_address = int(model.jnt_qposadr[joint_id])
            dof_address = int(model.jnt_dofadr[joint_id])
            qpos_indices = set(range(qpos_address, qpos_address + 7))
            qvel_indices = set(range(dof_address, dof_address + 6))
            if (
                qpos_address < 0
                or qpos_address + 7 > int(model.nq)
                or dof_address < 0
                or dof_address + 6 > int(model.nv)
                or target_qpos_indices.intersection(qpos_indices)
                or target_qvel_indices.intersection(qvel_indices)
            ):
                raise LiberoProRobustnessError(
                    "occlusion target free-joint layout is invalid or overlapping"
                )
            target_qpos_indices.update(qpos_indices)
            target_qvel_indices.update(qvel_indices)
            target_joints.append({
                "object": target, "joint": joint,
                "qpos_address": qpos_address, "dof_address": dof_address,
            })
        if (
            len(target_qpos_indices) != 7 * extra_count
            or len(target_qvel_indices) != 6 * extra_count
        ):
            raise LiberoProRobustnessError(
                "occlusion BDDL did not add the expected free-joint dimensions"
            )
        base_qpos_indices = [
            index for index in range(int(model.nq))
            if index not in target_qpos_indices
        ]
        base_qvel_indices = [
            index for index in range(int(model.nv))
            if index not in target_qvel_indices
        ]
        if len(base_qpos_indices) != base_nq or len(base_qvel_indices) != base_nv:
            raise LiberoProRobustnessError(
                "occlusion clean-state complement has invalid dimensions"
            )
        current = np.asarray(self._delegate.get_sim_state(), dtype=np.float64).copy()
        if current.size != 1 + int(model.nq) + int(model.nv):
            raise LiberoProRobustnessError("perturbed reset state has invalid dimensions")
        current[0] = source[0]
        current[1 + np.asarray(base_qpos_indices, dtype=np.int64)] = source[
            1 : 1 + base_nq
        ]
        current[
            1 + int(model.nq) + np.asarray(base_qvel_indices, dtype=np.int64)
        ] = source[1 + base_nq :]
        self._init_state_adaptation = {
            "mode": "insert_bddl_reset_free_joints_v1",
            "input_values": int(source.size),
            "output_values": int(current.size),
            "base_nq": base_nq,
            "base_nv": base_nv,
            "appended_qpos": 7 * extra_count,
            "appended_qvel": 6 * extra_count,
            "targets": target_joints,
        }
        return current

    def _remap_recompiled_init_state(self, init_state: Any) -> Any:
        """Map a pinned init tensor across an XML-recompiled named layout."""

        import numpy as np

        source_layout = self._source_joint_layout
        current_layout = self._recompiled_joint_layout
        if not source_layout or not current_layout:
            raise LiberoProRobustnessError(
                "recompiled model has no captured named state layout"
            )
        excluded: set[str] = set()
        target_joints: list[dict[str, Any]] = []
        if self._case == "view_occlusion":
            placement = self._config["object_placement"]
            targets = _names(
                placement.get("targets", placement.get("target")),
                "object_placement.targets",
            )
            inner = getattr(self._delegate, "env", None)
            object_dict = getattr(inner, "objects_dict", {})
            for target in targets:
                obj = object_dict.get(target) if isinstance(object_dict, Mapping) else None
                joints = getattr(obj, "joints", None)
                if not joints:
                    raise LiberoProRobustnessError(
                        f"occlusion target has no free joint: {target}"
                    )
                joint = str(joints[-1])
                source_target = next(
                    (item for item in source_layout if item["joint"] == joint),
                    None,
                )
                current_target = next(
                    (item for item in current_layout if item["joint"] == joint),
                    None,
                )
                if (
                    source_target is None
                    or current_target is None
                    or source_target["qpos_width"] != 7
                    or source_target["qvel_width"] != 6
                    or current_target["qpos_width"] != 7
                    or current_target["qvel_width"] != 6
                ):
                    raise LiberoProRobustnessError(
                        f"occlusion target is not a stable free joint: {target}"
                    )
                excluded.add(joint)
                target_joints.append({"object": target, "joint": joint})

        source_base = [
            item for item in source_layout if item["joint"] not in excluded
        ]
        current_by_name = {item["joint"]: item for item in current_layout}
        if set(item["joint"] for item in source_base) - set(current_by_name):
            raise LiberoProRobustnessError(
                "recompiled model lost a clean-state joint"
            )
        source_nq = sum(int(item["qpos_width"]) for item in source_base)
        source_nv = sum(int(item["qvel_width"]) for item in source_base)
        source = np.asarray(init_state, dtype=np.float64).reshape(-1)
        if source.size != 1 + source_nq + source_nv:
            raise LiberoProRobustnessError(
                "pinned init tensor differs from the captured clean joint layout"
            )
        current = np.asarray(self._delegate.get_sim_state(), dtype=np.float64).copy()
        sim = getattr(getattr(self._delegate, "env", None), "sim", None)
        model = getattr(sim, "model", None)
        if model is None or current.size != 1 + int(model.nq) + int(model.nv):
            raise LiberoProRobustnessError(
                "recompiled reset state has invalid dimensions"
            )
        current[0] = source[0]
        qpos_cursor = 1
        qvel_cursor = 1 + source_nq
        mappings = []
        for source_joint in source_base:
            target_joint = current_by_name[source_joint["joint"]]
            if (
                source_joint["qpos_width"] != target_joint["qpos_width"]
                or source_joint["qvel_width"] != target_joint["qvel_width"]
            ):
                raise LiberoProRobustnessError(
                    f"recompiled joint width drifted: {source_joint['joint']}"
                )
            qpos_width = int(source_joint["qpos_width"])
            qvel_width = int(source_joint["qvel_width"])
            qpos_target = 1 + int(target_joint["qpos_address"])
            qvel_target = 1 + int(model.nq) + int(target_joint["qvel_address"])
            current[qpos_target : qpos_target + qpos_width] = source[
                qpos_cursor : qpos_cursor + qpos_width
            ]
            current[qvel_target : qvel_target + qvel_width] = source[
                qvel_cursor : qvel_cursor + qvel_width
            ]
            qpos_cursor += qpos_width
            qvel_cursor += qvel_width
            mappings.append({
                "joint": source_joint["joint"],
                "source_qpos_address": source_joint["qpos_address"],
                "target_qpos_address": target_joint["qpos_address"],
                "source_qvel_address": source_joint["qvel_address"],
                "target_qvel_address": target_joint["qvel_address"],
            })
        if qpos_cursor != 1 + source_nq or qvel_cursor != source.size:
            raise LiberoProRobustnessError("init-state joint mapping was incomplete")
        self._init_state_adaptation = {
            "mode": "named_joint_recompile_mapping_v2",
            "input_values": int(source.size),
            "output_values": int(current.size),
            "source_layout_sha256": _json_hash(source_layout),
            "target_layout_sha256": _json_hash(current_layout),
            "mapped_joint_count": len(mappings),
            "reordered_joint_count": sum(
                item["source_qpos_address"] != item["target_qpos_address"]
                or item["source_qvel_address"] != item["target_qvel_address"]
                for item in mappings
            ),
            "excluded_bddl_joints": target_joints,
        }
        return current

    def _maybe_move_runtime_target(self) -> None:
        if not self._armed or not self._spec.runtime_move or self._runtime_event:
            return
        runtime = self._runtime
        _configured_step, object_name, delta_xy, min_delta_norm = (
            runtime.runtime_move_settings(self._spec, self._options)
        )
        trigger = runtime.runtime_move_trigger_status(
            self._vec_like, object_name, self._sim_step, self._spec, self._options,
        )
        if trigger is None:
            return
        application = runtime.move_object_xy(
            self._vec_like, object_name, delta_xy, self._options,
            min_delta_norm=min_delta_norm, trigger_info=trigger,
        )
        self._runtime_event = {
            "schema": ROBUSTNESS_EVENT_SCHEMA,
            "sequence": 1,
            "case": self._case,
            "sim_step": self._sim_step,
            "application": application,
        }

    def _fresh_observation(self, fallback: Any) -> Any:
        inner = getattr(self._delegate, "env", None)
        get_observations = getattr(inner, "_get_observations", None)
        return get_observations() if callable(get_observations) else fallback

    def _transform_observation(self, observation: Any) -> Any:
        if not isinstance(observation, Mapping) or self._spec.observation_perturbation is None:
            return observation
        import numpy as np

        pixels = {
            key: np.asarray(value)[None].copy()
            for key, value in observation.items()
            if key.endswith("_image")
            and isinstance(value, np.ndarray)
            and value.ndim == 3
            and value.shape[-1] == 3
        }
        if not pixels:
            raise LiberoProRobustnessError(
                "observation perturbation found no RGB camera images"
            )
        transformed = self._runtime.apply_case_to_observation(
            {"pixels": pixels}, self._spec,
            self._seed + self._observation_sequence, self._options,
        )
        result = dict(observation)
        for key, batch in transformed["pixels"].items():
            result[key] = batch[0]
        self._observation_sequence += 1
        return result


def robustness_env_factory(
    factory: Callable[[], Any], *, case: str, suite: str, task_name: str,
    bddl_path: Path, seed: int,
) -> Callable[[], RobustnessEnvironment]:
    """Wrap an RLinf worker factory without importing MuJoCo on the host."""

    def build() -> RobustnessEnvironment:
        return RobustnessEnvironment(
            factory(), case=case, suite=suite, task_name=task_name,
            bddl_path=bddl_path, seed=seed,
        )

    return build
