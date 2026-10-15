"""The default manipulation tool runtime over immutable Harness observations.

``ManipulationToolRuntime`` keeps its dataclass fields, construction,
tool dispatch (``invoke``) and per-round reset here. Each tool family's
handlers live in a mixin module beside this one: ``observe``,
``perception``, ``grasp``, ``holding``, ``skills``, ``motion_submit``,
``motion``, ``move_pose``, ``move_to`` and ``lifecycle``;
``common`` holds what several of them share. It serves the
``native_atomic_causal`` runtime id that ToolSurfaces and RunPlans name.
"""

from __future__ import annotations

from ora.agent.tools.dual_arm_surface import (
    PER_ROUND_RUNTIME_REGISTRIES, GROUNDING_RESULT_FIELDS,
    ROBODOJO_RGBD_CAPABILITY, ARX_X5_GRASP_COMPILER_CAPABILITY,
    dual_arm_grasp_scene_cameras, dual_arm_grasp_compiler_id, conform_to_result_schema,
)

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

from ora.agent.composition import resolve_tool_surface, validate_schema_instance
from ora.agent.composition.tool_surface import ToolSurfaceError
from ora.agent.evidence.policy_state import (
    PolicyStateError,
    load_policy_state,
    verify_policy_state,
)
from ora.agent.evidence.receipt_binding import canonical_public_receipt
from ora.agent.model_services.binding import service_client_arguments
from ora.contracts import (
    MOTION_ACTION_V3_SCHEMA,
    compile_enhanced_move_pose,
    validate_runtime_action,
)
from ora.contracts.pose import (
    PoseCompilationError,
    canonicalize_quaternion,
    compile_pose_request,
    matrix_quaternion,
    quaternion_matrix,
)
from ora.contracts.workspace import (
    is_monotonic_workspace_recovery,
    workspace_state,
)
from ora.contracts.refusal import EnvironmentRefusal
from ora.agent.tools.robot import RobotToolRuntime, ToolError, _append, _finite_vector
from ora.agent.tools import finish_outcomes, finish_task_check, mutation_compensation, retention, perception_feedback, tool_errors
from ora.agent.tools import grasp, grounding_preview, motion_arguments, move_pose_reach, observed_clearance_runtime, pose_target_annotations, scene_extent_review, target_refusals, tool_names, wait_tool, world_axes, x5_topdown
from ora.agent.tools.motion_arguments import (
    project_copyable_approach_targets as _project_copyable_grasp_approach_targets,
)
from ora.agent.tools import mark_point as _mark_point
from ora.agent.tools.mark_point import (
    MARK_POINT_BATCH_COLOURS,
    MARK_POINT_DEPTH_WINDOW_RADIUS,
    MARK_POINT_VERIFY_TOLERANCE_M,
    opencv_camera_point_to_world as _opencv_camera_point_to_world,
    robust_depth_measurement as _robust_depth_measurement,
)
from ora.agent.tools.geometry import (
    fuse_rgbd_pointcloud_sources, materialize_two_view_pointcloud,
)
from ora.agent.tools.grasp import (
    AlohaHybridGraspProposer,
    ContactGraspNetApiProposer,
    DepthPromptSegmenter,
    DEPTH_INLIER_GEOMETRY_CAPABILITY,
    GraspProposalBackend,
    GRASP_COMPILER_IDENTITIES,
    GRASP_QUERY_SCHEMA,
    GRASP_FUSION_MAX_DEPTH_M,
    GRASP_FUSION_VOXEL_SIZE_M,
    GROUNDING_MAX_CANDIDATES,
    PANDA_GRIP_SITE_COMPILER_IDENTITY,
    PcaTopDownGraspProposer,
    PointSegmentationBackend,
    Sam3ApiSegmenter,
    TextGroundingBackend,
    _canonical_hash,
    grasp_plan_public_result,
    grasp_public_result,
    grounding_public_result,
    load_segmented_target,
    materialize_grasp_query,
    materialize_grounded_targets,
    segmented_target_geometry,
    perception_selection_evidence,
    validate_segmented_target_source,
)


LEGACY_OPENETA_VIEW_CONFIG = {
    "schema": "ora.policy-view-config.v1",
    "rgb_resolution": [512, 512],
    "derived_resolution": [512, 512],
    "automatic_actual_views": ["agentview", "wrist"],
    "default_observe_views": ["agentview", "wrist"],
    "coordinate_convention": "opencv_top_left_xy",
}

ROBOTWIN_ALOHA_HYBRID_GRASP_CAPABILITY = (
    "robotwin_aloha_public_rgbd_topdown_grasp_v1"
)


MODEL_SERVICE_SET_BINDING_SCHEMA_V2 = (
    "ora.model-service-set-binding.v2"
)


def _grasp_hover_offset_m(surface: Any, *, grasp_plan_enabled: bool) -> float:
    """Select the source-bound pre-grasp distance without changing old surfaces."""

    if surface.supports(ROBOTWIN_ALOHA_HYBRID_GRASP_CAPABILITY):
        if not grasp_plan_enabled:
            raise ToolError(
                "ALOHA public RGB-D grasp staging requires grasp_plan on the ToolSurface",
                code="tool_surface_capability_mismatch",
                retryable=False,
            )
        return 0.10
    return 0.04 if grasp_plan_enabled else 0.08


from ora.agent.tools.runtime_artifacts import (  # noqa: F401  re-exported for mark_point and mark_point_overlay
    _artifact_record, _atomic_json, _png_base64,
)

# Every name importable from this module stays importable: the imports
# above and the re-exports below. The mixins import what they use themselves.
from ora.agent.tools.runtime.common import (  # noqa: F401  re-exported
    RETENTION_REVIEW_CAMERA_ORDER, ROBODOJO_POLICY_CAMERAS, _matrix_quat, _quat_matrix,
)
from ora.agent.tools.runtime.move_pose import ENHANCED_MOVE_POSE_EXAMPLES  # noqa: F401  re-exported
from ora.agent.tools.runtime.observe import ObserveRuntimeMixin
from ora.agent.tools.runtime.perception import PerceptionRuntimeMixin
from ora.agent.tools.runtime.grasp import GraspRuntimeMixin
from ora.agent.tools.runtime.holding import HoldingRuntimeMixin
from ora.agent.tools.runtime.skills import SkillRuntimeMixin
from ora.agent.tools.runtime.motion_submit import MotionSubmissionRuntimeMixin
from ora.agent.tools.runtime.motion import MotionRuntimeMixin
from ora.agent.tools.runtime.move_pose import MovePoseRuntimeMixin
from ora.agent.tools.runtime.move_to import MoveToRuntimeMixin
from ora.agent.tools.runtime.lifecycle import LifecycleRuntimeMixin


@dataclass
class ManipulationToolRuntime(
    RobotToolRuntime,
    ObserveRuntimeMixin,
    PerceptionRuntimeMixin,
    GraspRuntimeMixin,
    HoldingRuntimeMixin,
    SkillRuntimeMixin,
    MotionSubmissionRuntimeMixin,
    MotionRuntimeMixin,
    MovePoseRuntimeMixin,
    MoveToRuntimeMixin,
    LifecycleRuntimeMixin,
):
    """Shared perception/motion dispatcher across versioned compatible surfaces."""

    segmentation_backend: PointSegmentationBackend | None = None
    grounding_backend: TextGroundingBackend | None = None
    grasp_backend: GraspProposalBackend | None = None
    prerequisite_binding: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.surface.runtime_id != "native_atomic_causal":
            raise ToolError(
                "ManipulationToolRuntime requires a compatible perception surface"
            )
        self.artifact_root = self.control_root.parent / "openeta-tool-artifacts"
        self.point_path = self.control_root / "openeta-points.json"
        self.preview_path = self.control_root / "openeta-previews.json"
        self._active_mark_point_batch: tuple[dict[str, Any], ...] | None = None
        self._known_plan_refs: dict[str, dict[str, Any]] = {}
        # Ordered log of submitted mutation names; plan references record their
        # position in it so gripper-only intervals can keep a probed plan alive
        # The environment remains the execution
        # authority and re-validates joints before consuming any plan_ref.
        self._submitted_mutations: list[str] = []
        # Last held-object offset a placement assessment produced, re-shown
        # in the observation with its age so a policy that stops calling
        # place.plan does not lose it.
        self.last_held_object_offset: dict[str, Any] | None = None
        self.view_config = dict(self.surface.view_config or LEGACY_OPENETA_VIEW_CONFIG)
        # allowed_tools carries whatever this surface advertises, so a literal
        # dotted membership test turns the capability off on a renamed surface
        # without raising anything.
        segmentation_enabled = tool_names.declares(self.allowed_tools, "vision_segment")
        grounding_enabled = tool_names.declares(self.allowed_tools, "vision_ground")
        grasp_plan_enabled = tool_names.declares(self.allowed_tools, "grasp_plan")
        grasp_enabled = grasp_plan_enabled
        prerequisite = (
            dict(self.prerequisite_binding)
            if isinstance(self.prerequisite_binding, Mapping) else {}
        )
        expected_segmentation: Mapping[str, Any] | None = None
        expected_grasp: Mapping[str, Any] | None = None
        binding_scope_verified = False
        if isinstance(self.prerequisite_binding, Mapping) and (
            segmentation_enabled or grounding_enabled or grasp_enabled
        ):
            prerequisite_schema = prerequisite.get("schema")
            if prerequisite_schema == "ora.segmentation-service-binding.v1":
                expected_segmentation = dict(prerequisite)
            elif prerequisite_schema == MODEL_SERVICE_SET_BINDING_SCHEMA_V2:
                # A capability-scoped set binding is valid
                # for any surface that supports every named capability, so a
                # successor profile no longer needs the experiment-side
                # binding rename hack. Service identity (endpoint, process,
                # checkpoint hashes, qualification) stays fully enforced by
                # the service clients.
                services = prerequisite.get("services")
                scope = prerequisite.get("capability_scope")
                if (
                    set(prerequisite) != {
                        "schema", "capability_scope", "services",
                    }
                    or not isinstance(scope, list)
                    or not scope
                    or not all(
                        isinstance(value, str) and value for value in scope
                    )
                    or not isinstance(services, Mapping)
                    or set(services) != {"segmentation", "grasp_proposals"}
                    or not all(
                        isinstance(value, Mapping) for value in services.values()
                    )
                ):
                    raise ValueError(
                        "capability-scoped model-service set binding is invalid"
                    )
                unsupported = [
                    value for value in scope if not self.surface.supports(value)
                ]
                if unsupported:
                    raise ToolError(
                        "model-service binding capability scope is not "
                        "supported by the tool surface: "
                        + ", ".join(sorted(unsupported)),
                        code="tool_surface_capability_mismatch",
                        retryable=False,
                    )
                binding_scope_verified = True
                expected_segmentation = dict(services["segmentation"])
                expected_grasp = dict(services["grasp_proposals"])
            elif prerequisite_schema == "ora.model-service-set-binding.v1":
                services = prerequisite.get("services")
                if (
                    set(prerequisite) != {
                        "schema", "tool_surface", "services",
                    }
                    or prerequisite.get("tool_surface") != self.surface.profile_id
                    or not isinstance(services, Mapping)
                    or set(services) != {"segmentation", "grasp_proposals"}
                    or not all(
                        isinstance(value, Mapping) for value in services.values()
                    )
                ):
                    raise ValueError("model-service set binding is invalid")
                expected_segmentation = dict(services["segmentation"])
                expected_grasp = dict(services["grasp_proposals"])
            else:
                raise ValueError("unsupported perception prerequisite binding")
        for binding in (expected_segmentation, expected_grasp):
            if (
                binding is not None
                and not binding_scope_verified
                and binding.get("tool_surface") != self.surface.profile_id
            ):
                raise ToolError(
                    "model-service prerequisite is bound to a different tool surface"
                )
        if self.segmentation_backend is None:
            self.segmentation_backend = (
                Sam3ApiSegmenter(**service_client_arguments(
                    expected_segmentation, "ORA_SEGMENTATION_URL", "http://127.0.0.1:8114",
                ))
                if segmentation_enabled or grounding_enabled else DepthPromptSegmenter()
            )
        if self.grounding_backend is None:
            if grounding_enabled and callable(getattr(self.segmentation_backend, "ground", None)):
                self.grounding_backend = self.segmentation_backend  # type: ignore[assignment]
            elif grounding_enabled:
                self.grounding_backend = Sam3ApiSegmenter(**service_client_arguments(
                    expected_segmentation, "ORA_SEGMENTATION_URL", "http://127.0.0.1:8114",
                ))
        if self.grasp_backend is None:
            contact_backend = (
                ContactGraspNetApiProposer(**service_client_arguments(
                    expected_grasp, "ORA_GRASP_URL", "http://127.0.0.1:8115",
                ), hover_offset_m=_grasp_hover_offset_m(
                    self.surface, grasp_plan_enabled=grasp_plan_enabled,
                ))
                if grasp_enabled else None
            )
            hybrid = x5_topdown.hybrid_backend_arguments(self.surface)  # ALOHA or X5 fused family
            self.grasp_backend = (
                AlohaHybridGraspProposer(contact_backend=contact_backend, **hybrid)
                if contact_backend is not None and hybrid is not None
                else contact_backend
                if contact_backend is not None
                else PcaTopDownGraspProposer()
            )
        service_bindings: dict[str, Any] = {}
        if segmentation_enabled or grounding_enabled:
            readiness = getattr(self.segmentation_backend, "require_ready", None)
            if callable(readiness):
                service_bindings["segmentation"] = readiness()
        if grounding_enabled and self.grounding_backend is not self.segmentation_backend:
            readiness = getattr(self.grounding_backend, "require_ready", None)
            if callable(readiness):
                service_bindings["grounding"] = readiness()
        if grasp_enabled:
            readiness = getattr(self.grasp_backend, "require_ready", None)
            if callable(readiness):
                service_bindings["grasp_proposals"] = readiness()
        if service_bindings:
            _append(self.event_path, {
                "type": "resident_model_services_bound",
                "tool_surface": self.surface.profile_id,
                "services": service_bindings,
                "time": time.time(),
            })

    def definitions(self) -> list[dict[str, Any]]:
        return self.surface.select(self.allowed_tools)

    def per_round_control_files(self) -> tuple[str, ...]:
        """The base control files plus this surface's own per-round artifacts."""

        return (
            *super().per_round_control_files(),
            self.point_path.name,
            self.preview_path.name,
        )

    def round_registry_names(self) -> tuple[str, ...]:
        """In-memory references a Supervisor-owned world reset invalidates.

        Every name below refers to something the reset just destroyed: plans
        the environment has dropped, the mutation ledger those references are
        judged fresh against, and the open mark-point batch. Keeping one would let a stale ``plan_ref``
        pass this runtime's freshness rule and reach a backend that can only
        reject it, and that rejection is raised outside the recoverable-error
        classification, so a leaked reference becomes a Runtime fault rather
        than a retryable receipt.
        """

        return tuple(PER_ROUND_RUNTIME_REGISTRIES)

    def reset_round(self, next_round_index: int) -> None:
        super().reset_round(next_round_index)
        archive = self.control_root / "rounds" / f"round-{next_round_index - 1:04d}"
        for source in (self.point_path, self.preview_path):
            if source.is_file():
                if (archive / source.name).exists():
                    raise ToolError(f"refusing to overwrite round control state: {source.name}")
                os.replace(source, archive / source.name)
        self._active_mark_point_batch = None
        # Every reference below names something the Supervisor-owned world
        # reset just destroyed: plans the environment has dropped and the
        # mutation ledger those references are judged fresh against.
        # Keeping them would let a
        # stale plan_ref pass this runtime's freshness rule and reach a
        # backend that can only reject it, and the rejection is raised
        # outside the recoverable-error classification, so a leaked
        # reference becomes a Runtime fault rather than a retryable receipt.
        self._known_plan_refs.clear()
        self._submitted_mutations.clear()
        # Cleared through the declared roster so a new per-round registry is
        # named once and is provably empty at the next round's first turn.
        for name, empty in PER_ROUND_RUNTIME_REGISTRIES.items():
            setattr(self, name, copy.deepcopy(empty))

    def classify_call_effect(
        self, name: str, arguments: Mapping[str, Any]
    ) -> str:
        """Classify the actual call, not merely its advertised tool name."""

        del arguments
        if name in self.surface.lifecycle_tools:
            return "lifecycle"
        return "mutation" if name in self.surface.mutation_tools else "read_only"

    def invoke(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        started = time.time()
        taken = mutation_compensation.snapshot(self, name, arguments)
        try:
            if name not in self.allowed_tools:
                raise ToolError(
                    f"tool is disabled: {name}", code="tool_unavailable", retryable=False,
                    field="tool", received=name, constraints={"allowed": list(self.allowed_tools)},
                )
            if name == "finish_episode":
                # A latched success first: once check_task confirmed it, no
                # other finish refusal can help.
                finish_task_check.refuse_latched(self, arguments)
                finish_outcomes.admit(self, arguments)
            if self.surface.supports(retention.ADVISORY_CAPABILITY):
                self._validate_public_arguments(name, arguments)
            handlers = {
                "observe": self._observe,
                "mark_point": self._mark_point,
                "geometry_query": self._geometry,
                "compute": self._restricted_compute,
                "vision_segment": self._vision_segment,
                "vision_ground": self._vision_ground,
                "grasp_plan": self._grasp_plan,
                "grasp_verify_retention": self._grasp_verify_retention,
                "grasp_execute": lambda values: self._closed_skill("grasp_execute", values),
                "place_execute": lambda values: self._closed_skill("place_execute", values),
                "place_resume": lambda values: self._closed_skill("place_resume", values),
                "place_plan": self._place_plan,
                "align_plan": self._align_plan,
                "move_to": self._move_to,
                "rotate_wrist": self._motion_rotate_wrist_v2,
                "rotate_pitch": self._motion_rotate_pitch_v2,
                "preview_move": self._atomic_preview_move,
                "move_pose": self._atomic_move_pose,
                "set_gripper": self._motion_set_gripper,
                "close_gripper": self._atomic_close_gripper,
                "open_gripper": self._atomic_open_gripper,
                "recover_pose": self._motion_recover_pose_v2,
                "wait_ticks": lambda values: wait_tool.submit(self, values),
                "report_issue": self._report_issue,
                "check_task": self._check_task,
                "finish_episode": self._finish,
            }
            # A frozen surface advertises the dotted spelling and an active
            # one the canonical spelling; both reach the same handler. Handlers
            # are registered once, under the canonical name, so a tool added
            # later cannot be reachable under one spelling and missing under the
            # other.
            result = handlers[tool_names.canonical(name)](arguments)
            perception_feedback.enrich(self, name, arguments, result)
            visible = {
                key: value for key, value in result.items()
                if key not in {"image_base64", "images_base64", "_runtime"}
            }
            if self.surface.result_schemas.get(name) is not None:
                validate_schema_instance(visible, self.surface.result_schemas[name])
        except Exception as exc:
            mutation_compensation.restore(self, taken)
            structured = exc.as_dict() if isinstance(exc, ToolError) else ToolError(
                str(exc), code="internal_tool_error", retryable=False,
            ).as_dict()
            _append(self.event_path, {
                "type": "tool_error", "tool": name, "arguments": dict(arguments),
                "error": structured, "time": time.time(),
            })
            raise
        try:
            _append(self.event_path, {
                "type": "tool_result", "tool": name, "arguments": dict(arguments),
                "result": visible, "started": started, "finished": time.time(),
            })
        except OSError:
            if self.classify_call_effect(name, arguments) == "mutation":
                (self.control_root / "submitted-action.json").unlink(missing_ok=True)
            raise
        return result

    def _validate_public_arguments(self, name: str, arguments: Mapping[str, Any]) -> None:
        """Keep schema/arm validation independent of holding admission.

        Callers name a tool canonically. This surface may advertise the dotted
        spelling, and a mismatch here raises StopIteration rather than a
        ToolError, so the lookup compares canonical forms on both sides.
        """
        definition = tool_names.definition_for(self.surface.definitions, name)
        try:
            validate_schema_instance(
                dict(arguments), definition["inputSchema"], root_path="$input",
            )
        except ToolSurfaceError as error:
            raise tool_errors.schema_refusal(error) from error
