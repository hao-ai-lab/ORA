"""Five-tool robot interface with one terminal, state-changing submission."""

from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ora.agent.composition import (
    ResolvedToolSurface,
    resolve_tool_surface,
    validate_schema_instance,
)
from ora.agent.evidence.policy_state import PolicyStateError, load_policy_state
from ora.contracts.pose import compile_pose_request
from ora.contracts.workspace import (
    is_monotonic_workspace_recovery,
    workspace_state,
)
# Re-exported: tool modules import the shared envelope from here.
from .tool_errors import ToolError, require_exactly_one_of, require_known_arguments


ROUND_START_ISOLATION_SCHEMA = "ora.round-start-isolation.v1"
#: Control-root files that hold exactly one round of policy-visible state.
#: ``RobotToolRuntime.reset_round`` archives every one of them and
#: ``round_start_isolation`` proves none of them survived; a surface that adds
#: such a file extends ``per_round_control_files`` rather than either list.
PER_ROUND_CONTROL_FILES = (
    "current-observation.json",
    "submitted-action.json",
    "policy-task-check.json",
    "policy-finish.json",
)


TOOL_DEFINITIONS = (
    {
        "name": "observation.dump_state",
        "description": (
            "Read the latest or one saved policy-visible state step. Returns public state, "
            "camera/modality inventory and the selected step's main-camera image."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"step": {"type": ["integer", "null"], "minimum": 0}},
            "additionalProperties": False,
        },
    },
    {
        "name": "vision.inspect",
        "description": "Inspect a current or saved policy-visible camera image or crop.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "camera": {"type": "string"},
                "modality": {"type": "string", "enum": ["rgb", "depth"]},
                "crop": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4},
                "resolution": {"type": "string", "enum": ["thumbnail", "full"]},
                "step": {"type": ["integer", "null"], "minimum": 0},
            },
            "required": ["camera"],
            "additionalProperties": False,
        },
    },
    {
        "name": "geometry.query",
        "description": "Back-project pixels, project world points, or summarize region depth.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "camera": {"type": "string"},
                "points": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}, "minItems": 1, "maxItems": 16},
                "region": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4},
                "world_points_m": {"type": "array", "items": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}, "minItems": 1, "maxItems": 16},
                "window_px": {"type": "integer", "minimum": 0, "maximum": 15},
                "step": {"type": ["integer", "null"], "minimum": 0},
            },
            "required": ["camera"],
            "oneOf": [
                {"required": ["points"]},
                {"required": ["region"]},
                {"required": ["world_points_m"]},
            ],
            "additionalProperties": False,
        },
    },
    {
        "name": "robot.read_manual",
        "description": "Read one named section of the robot manual.",
        "inputSchema": {
            "type": "object",
            "properties": {"section": {"type": "string"}},
            "required": ["section"],
            "additionalProperties": False,
        },
    },
    {
        "name": "robot.submit_action",
        "description": "Submit one pose-only motion or one close-in-place gripper command.",
        "inputSchema": {
            "oneOf": [
                {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "const": "move_pose"},
                        "arm": {"type": "string", "enum": ["left", "right"]},
                        "position_m": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                        "orientation_xyzw": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                    },
                    "required": ["action", "arm", "position_m", "orientation_xyzw"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "const": "set_gripper"},
                        "arm": {"type": "string", "enum": ["left", "right"]},
                        "gripper": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["action", "arm", "gripper"],
                    "additionalProperties": False,
                },
            ],
        },
    },
)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manual_sections(path: Path) -> tuple[str, ...]:
    """Return the exact ordered level-two headings from one bound manual."""

    text = path.read_text(encoding="utf-8")
    sections = tuple(
        line[3:].strip() for line in text.splitlines() if line.startswith("## ")
    )
    if not sections or any(not section for section in sections):
        raise ToolError("manual must contain non-empty level-two sections")
    if len(sections) != len(set(sections)):
        raise ToolError("manual level-two sections must be unique")
    return sections


def tool_definitions(
    allowed_tools: Sequence[str] | None = None, *, manual_path: Path | None = None,
    surface: ResolvedToolSurface | None = None,
    profile_id: str = "robot_runtime_v1",
    component_overrides: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build the advertised tool surface for one concrete Runtime manual."""

    resolved = surface or resolve_tool_surface(
        profile_id, component_overrides=component_overrides,
    )
    base_definitions = resolved.definitions
    selected = (
        {item["name"] for item in base_definitions}
        if allowed_tools is None else set(allowed_tools)
    )
    definitions = [
        copy.deepcopy(item) for item in base_definitions if item["name"] in selected
    ]
    if manual_path is not None and any(
        definition["name"] == "robot.read_manual" for definition in definitions
    ):
        sections = list(manual_sections(manual_path))
        for definition in definitions:
            if definition["name"] == "robot.read_manual":
                definition["inputSchema"]["properties"]["section"]["enum"] = sections
                break
    return definitions


def _append(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(value), ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _finite_vector(value: Any, length: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ToolError(f"{label} must contain {length} numbers")
    output = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
            raise ToolError(f"{label} must contain finite numbers")
        output.append(float(item))
    return output


@dataclass
class RobotToolRuntime:
    control_root: Path
    manual_path: Path
    event_path: Path
    workspace_bounds_m: Mapping[str, list[float]]
    allowed_tools: Sequence[str] | None = None
    state_root: Path | None = None
    tool_surface_profile: str = "robot_runtime_v1"
    tool_surface_overrides: Mapping[str, Any] | None = None
    controlled_end_effector: str | None = None
    motion_probe_callback: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None
    align_plan_callback: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None

    def __post_init__(self) -> None:
        self.control_root = self.control_root.resolve()
        self.manual_path = self.manual_path.resolve()
        self.event_path = self.event_path.resolve()
        self.state_root = None if self.state_root is None else self.state_root.resolve()
        if self.controlled_end_effector is not None and not self.controlled_end_effector:
            raise ToolError("controlled_end_effector must be a non-empty name")
        if not self.manual_path.is_file():
            raise ToolError(f"manual is unavailable: {self.manual_path}")
        self.surface = resolve_tool_surface(
            self.tool_surface_profile,
            component_overrides=self.tool_surface_overrides,
        )
        from ora.agent.composition.public_parameters import bind_public_workspace
        self.surface = bind_public_workspace(self.surface, self.workspace_bounds_m)
        self._manual_sections = (
            manual_sections(self.manual_path)
            if any(
                item["name"] == "robot.read_manual"
                for item in self.surface.definitions
            )
            else ()
        )
        known = {item["name"] for item in self.surface.definitions}
        allowed = known if self.allowed_tools is None else set(self.allowed_tools)
        required_mutations = set(self.surface.mutation_tools)
        if not allowed.issubset(known) or not required_mutations.issubset(allowed):
            raise ToolError(
                "allowed_tools must be known and include the surface mutation tools"
            )
        self.allowed_tools = tuple(
            item["name"] for item in self.surface.definitions if item["name"] in allowed
        )

    def definitions(self) -> list[dict[str, Any]]:
        return tool_definitions(
            self.allowed_tools, manual_path=self.manual_path, surface=self.surface,
        )

    def per_round_control_files(self) -> tuple[str, ...]:
        """Control-root files that hold exactly one round's policy state.

        ``reset_round`` archives every name this returns and
        ``round_start_isolation`` proves none of them survived into the new
        round, so a surface that adds a policy-visible control file declares
        it once here instead of in two places that can drift apart. The
        canonical example is ``policy-task-check.json``: ``check_task`` reads
        whatever that file says, so a copy left behind by a round reset would
        answer the new round with the previous round's official outcome.
        """

        return PER_ROUND_CONTROL_FILES

    def round_registry_names(self) -> tuple[str, ...]:
        """In-memory attributes that must be empty when a round starts."""

        return ()

    def residual_round_registries(self) -> list[str]:
        """Which of ``round_registry_names`` still hold a value."""

        return sorted(
            name for name in self.round_registry_names() if getattr(self, name, None)
        )

    def round_start_isolation(self) -> dict[str, Any]:
        """Measure whether any policy-visible channel survived a round reset.

        Read-only: it inspects the Control root and this runtime's own
        registries and reports what it found. The Supervisor calls it before
        the new round's first observation is published and refuses to let the
        policy act when it does not pass.
        """

        residual_files = sorted(
            name for name in self.per_round_control_files()
            if (self.control_root / name).exists()
        )
        residual_registries = self.residual_round_registries()
        failures = []
        if residual_files:
            failures.append(
                "control files carrying a previous round: "
                + ", ".join(residual_files)
            )
        if residual_registries:
            failures.append(
                "tool registries carrying a previous round: "
                + ", ".join(residual_registries)
            )
        return {
            "schema": ROUND_START_ISOLATION_SCHEMA,
            "passed": not failures,
            "audited_control_files": list(self.per_round_control_files()),
            "audited_runtime_registries": list(self.round_registry_names()),
            "residual_control_files": residual_files,
            "residual_runtime_registries": residual_registries,
            "failures": failures,
        }

    def reset_round(self, next_round_index: int) -> None:
        """Archive ephemeral control state before a Supervisor-owned world reset."""

        if (
            isinstance(next_round_index, bool)
            or not isinstance(next_round_index, int)
            or next_round_index <= 0
        ):
            raise ToolError("next_round_index must be a positive integer")
        archive = self.control_root / "rounds" / f"round-{next_round_index - 1:04d}"
        archive.mkdir(parents=True, exist_ok=True)
        for name in self.per_round_control_files():
            source = self.control_root / name
            if source.is_file():
                target = archive / name
                if target.exists():
                    raise ToolError(f"refusing to overwrite round control state: {target}")
                os.replace(source, target)

    def classify_call_effect(
        self, name: str, arguments: Mapping[str, Any]
    ) -> str:
        """Classify calls for surfaces that advertise argument-aware effects."""

        del arguments
        if name in self.surface.lifecycle_tools:
            return "lifecycle"
        if name in self.surface.mutation_tools:
            return "mutation"
        return "read_only"

    def invoke(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        started = time.time()
        started_monotonic_ns = time.monotonic_ns()
        from . import mutation_compensation

        taken = mutation_compensation.snapshot(self, name, arguments)
        try:
            if name not in self.allowed_tools:
                raise ToolError(
                    f"tool is disabled by this experiment config: {name}",
                    code="tool_unavailable", retryable=False, field="tool",
                    received=name, constraints={"allowed": list(self.allowed_tools)},
                )
            if name == "observation.dump_state":
                result = self._dump_state(arguments)
            elif name == "vision.inspect":
                result = self._inspect(arguments)
            elif name == "geometry.query":
                result = self._geometry(arguments)
            elif name == "robot.read_manual":
                result = self._manual(arguments)
            elif name == "robot.submit_action":
                result = self._submit(arguments)
            else:
                raise ToolError(
                    f"unknown tool: {name}", code="unknown_tool", retryable=False,
                    field="tool", received=name,
                    constraints={"allowed": list(self.allowed_tools)},
                )
            public_result = {
                key: value for key, value in result.items()
                if key not in {"image_base64", "_runtime"}
            }
            result_schema = self.surface.result_schemas.get(name)
            if result_schema is not None:
                validate_schema_instance(public_result, result_schema)
        except Exception as exc:
            # Remove only what this call persisted; a pending action survives.
            mutation_compensation.restore(self, taken)
            finished = time.time()
            duration_s = round(
                max(0, time.monotonic_ns() - started_monotonic_ns) / 1e9, 6
            )
            structured = (
                exc.as_dict() if isinstance(exc, ToolError)
                else ToolError(str(exc), code="internal_tool_error", retryable=False).as_dict()
            )
            _append(self.event_path, {
                "type": "tool_error", "tool": name, "arguments": dict(arguments),
                "error": structured, "time": finished,
                "started": started, "finished": finished,
                "duration_s": duration_s, "duration_clock": "monotonic",
            })
            raise
        finished = time.time()
        duration_s = round(
            max(0, time.monotonic_ns() - started_monotonic_ns) / 1e9, 6
        )
        try:
            _append(self.event_path, {
                "type": "tool_result", "tool": name, "arguments": dict(arguments),
                "result": public_result,
                "started": started, "finished": finished,
                "duration_s": duration_s, "duration_clock": "monotonic",
            })
        except OSError:
            # The action file is only an unexecuted proposal at this boundary.
            # Never let a mutation survive if its pre-dispatch evidence failed.
            if self.classify_call_effect(name, arguments) == "mutation":
                (self.control_root / "submitted-action.json").unlink(missing_ok=True)
            raise
        return result

    def _dump_state(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if set(arguments) - {"step"}:
            raise ToolError("observation.dump_state accepts only optional step")
        step = arguments.get("step")
        if step is not None and (isinstance(step, bool) or not isinstance(step, int) or step < 0):
            raise ToolError("step must be a non-negative integer or null")
        if self.state_root is None:
            raise ToolError("policy-state evidence is unavailable")
        try:
            state = load_policy_state(self.state_root.parent, step)
        except PolicyStateError as exc:
            raise ToolError(str(exc)) from exc
        main_camera = state.get("main_camera", "head_camera")
        camera = state.get("cameras", {}).get(main_camera)
        if not isinstance(camera, Mapping):
            raise ToolError(f"policy-state main camera is unavailable: {main_camera}")
        reference = camera.get("files", {}).get("rgb_lossless_png")
        if not isinstance(reference, Mapping) or not isinstance(reference.get("path"), str):
            raise ToolError("policy-state main image reference is invalid")
        run_root = self.state_root.parent.resolve()
        image_path = (run_root / reference["path"]).resolve()
        if not image_path.is_file() or not image_path.is_relative_to(run_root):
            raise ToolError("policy-state main image escapes the run")
        if _hash(image_path) != reference.get("sha256"):
            raise ToolError("policy-state main image hash drift")
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((512, 512))
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        encoded = stream.getvalue()
        return {
            "state": state,
            "camera_inventory": {
                name: {
                    "mount": record.get("mount"),
                    "calibration_dynamic": record.get("calibration_dynamic", False),
                    "modalities": record.get("modalities", []),
                    "size": [record.get("width"), record.get("height")],
                }
                for name, record in sorted(state.get("cameras", {}).items())
                if isinstance(record, Mapping)
            },
            "image_camera": main_camera,
            "image_source_sha256": reference["sha256"],
            "image_sha256": hashlib.sha256(encoded).hexdigest(),
            "image_base64": base64.b64encode(encoded).decode(),
        }

    def _current(self) -> tuple[dict[str, Any], Path, dict[str, Any]]:
        pointer = self.control_root / "current-observation.json"
        if not pointer.is_file():
            raise ToolError(
                "no current observation", code="environment_unavailable",
                retryable=False,
            )
        control = json.loads(pointer.read_text(encoding="utf-8"))
        if control.get("open") is not True:
            # Its own code: stale_observation elsewhere refuses a value bound to
            # an earlier observation and is retryable with a changed value.
            raise ToolError(
                "the current observation is closed: an action was submitted on it "
                "or its round ended, and no fresh observation is published yet",
                code="action_pending_fresh_observation", retryable=False,
            )
        root = Path(control["source_root"]).resolve()
        if not root.is_dir() or not root.is_relative_to(self.control_root.parent.resolve()):
            raise ToolError("observation root escapes the job boundary", code="artifact_integrity_fault")
        manifest = root / "observation.json"
        if _hash(manifest) != control.get("observation_sha256"):
            raise ToolError("observation manifest hash drift", code="artifact_hash_mismatch")
        observation = json.loads(manifest.read_text(encoding="utf-8"))
        return control, root, observation

    def _select_observation(
        self, step: Any,
    ) -> tuple[int | None, dict[str, Any] | None, Path, dict[str, Any]]:
        if step is None:
            control, root, observation = self._current()
            reference = control.get("policy_state", {})
            selected = reference.get("step") if isinstance(reference, Mapping) else None
            if self.state_root is None or not isinstance(selected, int):
                raise ToolError(
                    "current policy-state evidence is unavailable",
                    code="artifact_not_found", retryable=False,
                )
            try:
                state = load_policy_state(self.state_root.parent, selected)
            except PolicyStateError as exc:
                raise ToolError(
                    str(exc), code="artifact_not_found", retryable=False,
                ) from exc
            if (
                state.get("state_sha256") != reference.get("state_sha256")
                or state.get("observation", {}).get("id") != control.get("observation_id")
            ):
                raise ToolError(
                    "current policy-state reference drift",
                    code="artifact_hash_mismatch", retryable=False,
                )
            return selected, state, root, observation
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ToolError(
                "step must be a non-negative integer or null", field="step",
                received=step, constraints={"minimum": 0, "nullable": True},
            )
        if self.state_root is None:
            raise ToolError(
                "policy-state evidence is unavailable", code="artifact_not_found",
                retryable=False,
            )
        try:
            state = load_policy_state(self.state_root.parent, step)
        except PolicyStateError as exc:
            from .saved_steps import saved_step_refusal
            raise saved_step_refusal(self, step, exc) from exc
        manifest = state.get("observation", {}).get("manifest", {})
        relative = manifest.get("path") if isinstance(manifest, Mapping) else None
        if not isinstance(relative, str):
            raise ToolError("saved observation reference is invalid",
                            code="artifact_integrity_fault", retryable=False)
        run_root = self.state_root.parent.resolve()
        manifest_path = (run_root / relative).resolve()
        if not manifest_path.is_file() or not manifest_path.is_relative_to(run_root):
            raise ToolError("saved observation is unavailable",
                            code="artifact_integrity_fault", retryable=False)
        if _hash(manifest_path) != manifest.get("sha256"):
            raise ToolError("saved observation hash drift", code="artifact_hash_mismatch", retryable=False)
        observation = json.loads(manifest_path.read_text(encoding="utf-8"))
        return step, state, manifest_path.parent, observation

    @staticmethod
    def _camera(root: Path, observation: Mapping[str, Any], camera: Any) -> tuple[dict[str, Any], Path, Path, Path]:
        if not isinstance(camera, str) or camera not in observation.get("cameras", {}):
            raise ToolError("camera is not policy-visible")
        record = observation["cameras"][camera]
        files = record.get("files", {})
        roles = ("rgb_lossless_png", "depth_numpy_m_f32", "calibration")
        paths = []
        for role in roles:
            ref = files.get(role)
            if not isinstance(ref, Mapping) or not isinstance(ref.get("path"), str):
                raise ToolError(f"camera lacks {role}", code="artifact_integrity_fault")
            path = (root / ref["path"]).resolve()
            if not path.is_file() or not path.is_relative_to(root):
                raise ToolError("camera artifact escapes observation root", code="artifact_integrity_fault")
            if _hash(path) != ref.get("sha256"):
                raise ToolError("camera artifact hash drift", code="artifact_hash_mismatch")
            paths.append(path)
        return record, paths[0], paths[1], paths[2]

    def _inspect(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if set(arguments) - {"camera", "modality", "crop", "resolution", "step"} or "camera" not in arguments:
            raise ToolError("vision.inspect fields are camera, optional modality, crop, resolution and step")
        step, state, root, observation = self._select_observation(arguments.get("step"))
        record, rgb_path, depth_path, calibration_path = self._camera(root, observation, arguments["camera"])
        modality = arguments.get("modality", "rgb")
        if modality not in {"rgb", "depth"}:
            raise ToolError(
                "modality must be rgb or depth", field="modality", received=modality,
                constraints={"enum": ["rgb", "depth"]},
            )
        from PIL import Image
        depth_stats = None
        if modality == "rgb":
            image = Image.open(rgb_path).convert("RGB")
        else:
            import numpy as np
            depth = np.load(depth_path, allow_pickle=False).astype(np.float32)
            valid = depth[np.isfinite(depth) & (depth > 1e-6)]
            if valid.size:
                low, high = np.percentile(valid, [2, 98])
                high = max(float(high), float(low) + 1e-6)
                normalized = np.clip((depth - low) / (high - low), 0, 1)
                normalized[~np.isfinite(depth) | (depth <= 1e-6)] = 0
                color = np.stack([
                    normalized,
                    1.0 - np.abs(2.0 * normalized - 1.0),
                    1.0 - normalized,
                ], axis=-1)
                image = Image.fromarray((color * 255).astype(np.uint8), mode="RGB")
                depth_stats = {
                    "valid_count": int(valid.size),
                    "valid_fraction": float(valid.size / depth.size),
                    "min_m": float(np.min(valid)),
                    "median_m": float(np.median(valid)),
                    "max_m": float(np.max(valid)),
                    "visualization_percentile_m": [float(low), float(high)],
                }
            else:
                image = Image.fromarray(np.zeros((*depth.shape, 3), dtype=np.uint8), mode="RGB")
                depth_stats = {"valid_count": 0, "valid_fraction": 0.0}
        crop = arguments.get("crop")
        if crop is not None:
            if not isinstance(crop, list) or len(crop) != 4 or not all(isinstance(v, int) and not isinstance(v, bool) for v in crop):
                raise ToolError("crop must be integer [x0,y0,x1,y1]")
            x0, y0, x1, y1 = crop
            if not (0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height):
                raise ToolError("crop is outside the source image")
            image = image.crop((x0, y0, x1, y1))
        resolution = arguments.get("resolution", "thumbnail")
        if resolution not in {"thumbnail", "full"}:
            raise ToolError("resolution must be thumbnail or full")
        if resolution == "thumbnail":
            image.thumbnail((512, 512))
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        encoded = stream.getvalue()
        roi = crop or [0, 0, int(record["width"]), int(record["height"])]
        return {
            "step": step,
            "source_state_sha256": state.get("state_sha256") if state else None,
            "camera": arguments["camera"],
            "modality": modality,
            "source_size": [int(record["width"]), int(record["height"])],
            "returned_size": [image.width, image.height],
            "source_roi_xyxy": roi,
            "coordinate_convention": "opencv_top_left_uv",
            "source_image_sha256": _hash(rgb_path if modality == "rgb" else depth_path),
            "source_artifacts": {
                "rgb": record.get("files", {}).get("rgb_lossless_png"),
                "depth": record.get("files", {}).get("depth_numpy_m_f32"),
                "calibration": record.get("files", {}).get("calibration"),
            },
            "calibration_sha256": _hash(calibration_path),
            "depth_stats": depth_stats,
            "image_bytes": len(encoded),
            "image_sha256": hashlib.sha256(encoded).hexdigest(),
            "image_base64": base64.b64encode(encoded).decode(),
        }

    def _geometry(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        require_known_arguments(
            arguments, allowed=("camera", "points", "region", "world_points_m", "window_px", "step"),
            required=("camera",), message="geometry.query fields are camera plus points or region",
        )
        require_exactly_one_of(
            arguments, ("points", "region", "world_points_m"),
            message="provide exactly one of points, region or world_points_m",
        )
        from . import perception_arguments as pixels, saved_steps

        step, state, root, observation = self._select_observation(
            saved_steps.route_current(self, arguments.get("step")))
        record, _rgb, depth_path, calibration_path = self._camera(root, observation, arguments["camera"])
        width, height = int(record["width"]), int(record["height"])
        import numpy as np
        depth = np.load(depth_path, allow_pickle=False)
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        if "region" in arguments:
            box = pixels.region_box(arguments["region"], width=width, height=height)
            x0, y0, x1, y1 = box
            values = depth[y0:y1, x0:x1]
            valid = values[np.isfinite(values) & (values > 1e-6)]
            result: dict[str, Any] = {
                "region": box, "valid_count": int(valid.size),
                "valid_fraction": float(valid.size / values.size),
            }
            if valid.size:
                result["depth_m"] = {
                    "min": float(np.min(valid)), "median": float(np.median(valid)),
                    "max": float(np.max(valid)),
                }
            return {
                "step": step,
                "source_state_sha256": state.get("state_sha256") if state else None,
                "camera": arguments["camera"], "region_result": result,
            }
        window = arguments.get("window_px", 3)
        if not isinstance(window, int) or isinstance(window, bool) or not 0 <= window <= 15:
            raise ToolError("window_px must be in [0,15]")
        intrinsic = np.asarray(calibration["intrinsic_cv"], dtype=np.float64)
        extrinsic = np.asarray(calibration["extrinsic_cv"], dtype=np.float64)
        if intrinsic.shape != (3, 3) or extrinsic.shape not in {(3, 4), (4, 4)}:
            raise ToolError("invalid calibration matrices", code="invalid_camera_calibration")
        homogeneous = np.vstack([extrinsic, [0, 0, 0, 1]]) if extrinsic.shape == (3, 4) else extrinsic
        if "world_points_m" in arguments:
            world_points = arguments["world_points_m"]
            if not isinstance(world_points, list) or not 1 <= len(world_points) <= 16:
                raise ToolError("world_points_m must contain 1..16 [x,y,z] points")
            projected = []
            for index, raw_point in enumerate(world_points):
                point = _finite_vector(raw_point, 3, "world_points_m point")
                camera_point = homogeneous @ np.asarray([*point, 1.0], dtype=np.float64)
                pixels.finite_camera_point(camera_point, raw_point, index)
                z = float(camera_point[2])
                if z <= 1e-9:
                    projected.append({
                        "world_xyz_m": point, "camera_xyz_m": camera_point[:3].tolist(),
                        "in_front": False, "in_frame": False,
                    })
                    continue
                u = float(intrinsic[0, 0] * camera_point[0] / z + intrinsic[0, 2])
                v = float(intrinsic[1, 1] * camera_point[1] / z + intrinsic[1, 2])
                if not pixels.finite_pixel(u, v):  # in front, but projects to no pixel
                    projected.append({"world_xyz_m": point, "in_front": True, "in_frame": False,
                                      "camera_xyz_m": [float(value) for value in camera_point[:3]]})
                    continue
                rounded = [int(round(u)), int(round(v))]
                in_frame = 0 <= rounded[0] < int(record["width"]) and 0 <= rounded[1] < int(record["height"])
                scene_depth = None
                residual = None
                if in_frame:
                    sampled = float(depth[rounded[1], rounded[0]])
                    if math.isfinite(sampled) and sampled > 1e-6:
                        scene_depth = sampled
                        residual = z - sampled
                projected.append({
                    "world_xyz_m": point,
                    "camera_xyz_m": [float(value) for value in camera_point[:3]],
                    "pixel_uv": [u, v],
                    "pixel_uv_rounded": rounded,
                    "in_front": True,
                    "in_frame": in_frame,
                    "scene_depth_m": scene_depth,
                    "depth_residual_m": residual,
                })
            return {
                "step": step,
                "source_state_sha256": state.get("state_sha256") if state else None,
                "camera": arguments["camera"],
                "calibration_sha256": _hash(calibration_path),
                "world_points": projected,
            }
        points = arguments["points"]
        if not isinstance(points, list) or not 1 <= len(points) <= 16:
            raise ToolError("points must contain 1..16 source [u,v] pixels")
        inverse = np.linalg.inv(homogeneous)
        results = []
        for index, point in enumerate(points):
            u, v = pixels.query_pixel(point, index, width=width, height=height)
            patch = depth[max(0, v-window):min(depth.shape[0], v+window+1), max(0, u-window):min(depth.shape[1], u+window+1)]
            valid = patch[np.isfinite(patch) & (patch > 1e-6)]
            if not valid.size:
                results.append({"pixel_uv": [u, v], "valid": False})
                continue
            z = float(np.median(valid))
            x = (u - intrinsic[0, 2]) * z / intrinsic[0, 0]
            y = (v - intrinsic[1, 2]) * z / intrinsic[1, 1]
            world = inverse @ np.asarray([x, y, z, 1.0])
            results.append({
                "pixel_uv": [u, v], "valid": True, "depth_median_m": z,
                "valid_count": int(valid.size),
                "world_xyz_m": [float(world[i]) for i in range(3)],
            })
        return {
            "step": step,
            "source_state_sha256": state.get("state_sha256") if state else None,
            "camera": arguments["camera"], "points": results,
        }

    def _manual(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if set(arguments) != {"section"} or not isinstance(arguments.get("section"), str):
            raise ToolError("robot.read_manual requires section")
        text = self.manual_path.read_text(encoding="utf-8")
        requested = arguments["section"]
        if requested not in self._manual_sections:
            raise ToolError(
                "unknown manual section", code="unknown_section", field="section",
                received=arguments["section"],
                constraints={"sections": list(self._manual_sections)},
            )
        lines = text.splitlines()
        start = next(
            index for index, line in enumerate(lines)
            if line.startswith("## ") and line[3:].strip() == requested
        )
        end = next(
            (index for index in range(start + 1, len(lines)) if lines[index].startswith("## ")),
            len(lines),
        )
        content = "\n".join(lines[start:end]).strip()
        return {"section": requested, "content": content, "manual_sha256": _hash(self.manual_path)}

    def _submit(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        target = self.control_root / "submitted-action.json"
        if target.exists():
            raise ToolError(
                "an action is already submitted for this observation",
                code="duplicate_action", retryable=False,
            )
        if self.surface.profile_id == "robot_runtime_v2":
            return self._submit_target_state(arguments, target)
        action_kind = arguments.get("action")
        if action_kind not in {"move_pose", "set_gripper"}:
            raise ToolError(
                "action must be move_pose or set_gripper", field="action",
                received=action_kind, constraints={"enum": ["move_pose", "set_gripper"]},
            )
        if arguments.get("arm") not in {"left", "right"}:
            raise ToolError(
                "arm must be left or right", field="arm", received=arguments.get("arm"),
                constraints={"enum": ["left", "right"]},
            )
        control, _root, observation = self._current()
        arm = str(arguments["arm"])
        if action_kind == "move_pose":
            expected = {"action", "arm", "position_m", "orientation_xyzw"}
            if set(arguments) != expected:
                raise ToolError(
                    f"move_pose requires {sorted(expected)}", field="arguments",
                    received=sorted(arguments), constraints={"required": sorted(expected)},
                )
            position = _finite_vector(arguments["position_m"], 3, "position_m")
            orientation = _finite_vector(arguments["orientation_xyzw"], 4, "orientation_xyzw")
            norm = math.sqrt(sum(value * value for value in orientation))
            if not 0.98 <= norm <= 1.02:
                raise ToolError(
                    "orientation_xyzw must be normalized", field="orientation_xyzw",
                    received=orientation, constraints={"norm_min": 0.98, "norm_max": 1.02},
                )
            current_position = self._observed_arm_position(observation, arm)
            target_workspace = workspace_state(position, self.workspace_bounds_m)
            recovery = (
                current_position is not None
                and is_monotonic_workspace_recovery(
                    current_position, position, self.workspace_bounds_m,
                )
            )
            if not target_workspace["inside"] and not recovery:
                for index, axis in enumerate(("x", "y", "z")):
                    details = target_workspace["violations"].get(axis)
                    if details is not None:
                        interval = details["allowed"]
                        raise ToolError(
                            f"position {axis} is outside workspace bounds",
                            code="out_of_bounds", field=f"position_m[{index}]",
                            received=position[index],
                            constraints={"minimum": interval[0], "maximum": interval[1]},
                        )
            action_arguments = {
                "arm": arm, "position_m": position,
                "orientation_xyzw": orientation,
            }
        else:
            expected = {"action", "arm", "gripper"}
            if set(arguments) != expected:
                raise ToolError(
                    f"set_gripper requires {sorted(expected)}", field="arguments",
                    received=sorted(arguments), constraints={"required": sorted(expected)},
                )
            gripper = arguments["gripper"]
            if isinstance(gripper, bool) or not isinstance(gripper, (int, float)) or not math.isfinite(gripper) or not 0 <= gripper <= 1:
                raise ToolError(
                    "gripper must be in [0,1]", field="gripper", received=gripper,
                    constraints={"minimum": 0, "maximum": 1},
                )
            action_arguments = {"arm": arm, "gripper": float(gripper)}
        action = {
            "schema": "ora-action.v3",
            "observation_id": control["observation_id"],
            "action_id": f"action-{control['generation']}",
            "action": action_kind,
            "arguments": action_arguments,
            "bound_observation_sha256": control["observation_sha256"],
        }
        return self._persist_action(target, control, action)

    def _submit_target_state(
        self, arguments: Mapping[str, Any], target: Path,
    ) -> dict[str, Any]:
        from .target_state import _submit_target_state as invoke
        return invoke(self, arguments, target)

    @staticmethod
    def _observed_arm_position(
        observation: Mapping[str, Any], arm: str,
    ) -> list[float] | None:
        state = observation.get("proprioception", {}).get(
            "end_effectors", {}
        ).get(arm)
        pose = state.get("pose_xyz_xyzw") if isinstance(state, Mapping) else None
        if not isinstance(pose, list) or len(pose) != 7:
            return None
        try:
            return [float(item) for item in pose[:3]]
        except (TypeError, ValueError):
            return None

    def _persist_action(
        self, target: Path, control: dict[str, Any], action: Mapping[str, Any],
    ) -> dict[str, Any]:
        try:
            descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise ToolError(
                "an action is already submitted for this observation",
                code="duplicate_action", retryable=False,
            ) from exc
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(action, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        control["open"] = False
        temporary = (self.control_root / "current-observation.json").with_suffix(".json.tmp")
        temporary.write_text(json.dumps(control, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, self.control_root / "current-observation.json")
        return {"accepted": True, "directive": "action_submitted", "action_id": action["action_id"]}
