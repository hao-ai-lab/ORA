"""``preview_move``, ``move_pose`` and the semantic pose primitives.

``move_pose`` in its atomic and sparse forms, and the ``rotate_wrist``,
``rotate_pitch`` and ``move_to`` primitives compiled through it.

A mixin of ``core.ManipulationToolRuntime``; it holds
methods only, no dataclass fields.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Callable, Mapping

from ora.contracts import (
    MOTION_ACTION_V3_SCHEMA, compile_enhanced_move_pose,
)

from ora.agent.tools import (
    motion_arguments, motion_refusals, move_pose_reach, pose_target_annotations, world_axes,
)
from ora.agent.tools.robot import ToolError
from ora.agent.tools.runtime_artifacts import _atomic_json, _png_base64


# EXP06A closing repair A5: every rejected move_pose argument payload gets
# these worked examples in the structured error so a format mistake costs one
# call, not a trial-and-error loop. Kept in one place and asserted against
# compile_enhanced_move_pose by a regression test so they cannot drift from
# the accepted contract.
ENHANCED_MOVE_POSE_EXAMPLES: tuple[dict[str, Any], ...] = (
    {"targets": {"right": {
        "interaction": "free_space",
        "position": {"mode": "absolute", "position_m": [-0.10, 0.02, 0.95]},
        "orientation": {"mode": "hold_current"},
    }}},
    {"targets": {"left": {
        "interaction": "precision",
        "position": {
            "mode": "delta",
            "translation_m": [0.0, 0.0, -0.03],
            "frame": "world",
        },
        "orientation": {"mode": "hold_current"},
    }}},
    {"targets": {"right": {
        "interaction": "contact_approach",
        "position": {"mode": "hold_current"},
        "orientation": {
            "mode": "axis_angle_delta",
            "axis": "x",
            "angle_deg": 25.0,
            "frame": "end_effector",
        },
    }}},
    {"targets": {"left": {
        "position_m": [-0.12, 0.05, 0.92],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }}},
    {"plan_ref": "plan-0123456789abcdef0123456789abcdef"},
)


class MovePoseRuntimeMixin:
    """``preview_move``, ``move_pose`` and the semantic pose primitives."""

    def _atomic_preview_move(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        control, root, observation, current_pose, _gripper = self._current_grip_state()
        arm = self._atomic_arm(observation)
        target, compiled = self._atomic_compiled_target(
            arguments, current_pose=current_pose, arm=arm,
        )
        preview_id = "preview-" + hashlib.sha256(json.dumps({
            "observation": control["observation_id"],
            "action": "move_pose",
            "arguments": target,
        }, sort_keys=True).encode()).hexdigest()[:16]
        previews = self._load_previews()
        previews[preview_id] = {
            "observation_id": control["observation_id"],
            "action": "move_pose",
            "arguments": target,
            **({"compiled_pose": compiled} if compiled is not None else {}),
        }
        _atomic_json(self.preview_path, previews)
        preview_image = self._annotated_rgb(
            root=root,
            observation=observation,
            camera="agentview",
            labelled_world_points=[
                ("target", list(target["position_m"]), (255, 0, 255))
            ],
            identity_kind="move_preview",
        )
        result = {
            "status": "candidate",
            "preview_id": preview_id,
            "views": ["candidate_agentview"],
            "images_base64": [_png_base64(preview_image)],
            "_runtime": {"directive": "preview_only"},
        }
        if compiled is not None:
            result["compiled_pose"] = compiled
        if self.surface.view_config is not None:
            result["view_dimensions"] = {
                "candidate_agentview": self._image_dimensions(preview_image),
            }
        return result

    def _atomic_move_pose(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if self.surface.supports("enhanced_move_pose_v1"):
            return self._motion_move_pose_v2(arguments)
        self._ensure_mutation_available("move_pose")
        control, _root, observation, current_pose, _gripper = self._current_grip_state()
        if set(arguments) == {"preview_id"}:
            preview = self._load_previews().get(arguments["preview_id"])
            if (
                not isinstance(preview, Mapping)
                or preview.get("observation_id") != control["observation_id"]
                or preview.get("action") != "move_pose"
                or not isinstance(preview.get("arguments"), Mapping)
            ):
                raise ToolError("preview_id is unknown or stale", code="stale_preview")
            target = dict(preview["arguments"])
            compiled = (
                dict(preview["compiled_pose"])
                if isinstance(preview.get("compiled_pose"), Mapping)
                else None
            )
        else:
            target, compiled = self._atomic_compiled_target(
                arguments,
                current_pose=current_pose,
                arm=self._atomic_arm(observation),
            )
        self._validate_target_workspace(
            {"pose": {
                "position_m": target["position_m"],
                "orientation_xyzw": target["orientation_xyzw"],
            }},
            current_position_m=current_pose[:3],
        )
        return self._atomic_submit(
            control=control, action_name="move_pose", arguments=target,
            compiled_pose=compiled,
        )

    def _motion_move_pose_v2(
        self, arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Compile one sparse dual-arm ergonomic request to planner targets."""

        try:
            return self._motion_move_pose_v2_admit(arguments)
        except ToolError as exc:
            if (
                exc.code == "invalid_motion_arguments"
                and "examples" not in exc.constraints
            ):
                # Written in the spelling this surface's own schema admits.
                exc.constraints["examples"] = motion_refusals.move_pose_examples(
                    self.surface, ENHANCED_MOVE_POSE_EXAMPLES,
                )
            raise

    def _motion_move_pose_v2_admit(
        self, arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        self._ensure_mutation_available("move_pose")
        if "plan_ref" in arguments and self.surface.supports("direct_pose_submission_v1"):
            raise ToolError(
                "Submit an explicit current pose target",
                code="invalid_motion_arguments", field="plan_ref",
            )
        payload_ref = arguments.get("payload_ref")
        if payload_ref is not None:
            raise ToolError(
                "payload_ref is deferred from the motion-intent surface",
                code="unsupported_motion_extension", field="payload_ref",
            )
        core_arguments = {
            key: value for key, value in arguments.items() if key != "payload_ref"
        }
        targets = self._surface_targets(core_arguments)
        # One place either published spelling is read. The flat form names the
        # choice as an enum so a malformed position names its field instead of
        # returning "must match exactly one input branch" from three levels down.
        targets = world_axes.resolve_targets(self.surface, motion_arguments.arm_targets(targets))
        control, root, observation, _pose, _gripper = self._current_grip_state()
        current = self._surface_arm_states(observation)
        compiled_targets: dict[str, Any] = {}
        action_targets: dict[str, Any] = {}
        action_interactions: dict[str, str] = {}
        for arm, request in targets.items():
            if not isinstance(request, Mapping):
                raise ToolError(
                    f"{arm} move_pose target must be an object",
                    code="invalid_motion_arguments",
                )
            interaction = request.get("interaction", "free_space")
            if interaction not in {"free_space", "precision", "contact_approach"}:
                raise ToolError(
                    f"{arm} interaction must be free_space, precision, or contact_approach",
                    code="invalid_motion_arguments", field="interaction",
                )
            pose_request = {
                key: value for key, value in request.items() if key != "interaction"
            }
            try:
                compiled = compile_enhanced_move_pose(
                    pose_request,
                    current_pose_xyz_xyzw=current[arm],
                    controlled_point="end_effector",
                )
            except (TypeError, ValueError) as exc:
                raise ToolError(
                    str(exc), code="invalid_motion_arguments",
                    field=motion_refusals.target_field(self.surface, arm),
                ) from exc
            target_pose = {
                "position_m": list(compiled["target_pose_xyz_xyzw"][:3]),
                "orientation_xyzw": list(compiled["target_pose_xyz_xyzw"][3:]),
            }
            self._validate_target_workspace(
                {"pose": target_pose}, current_position_m=current[arm][:3],
                field=motion_refusals.target_field(self.surface, arm, {
                    "absolute": "position_m", "delta": "translation_m",
                }.get(str(compiled["position_mode"]))),
            )
            reference: dict[str, Any] = {}
            if move_pose_reach.declared(self.surface):
                compiled, target_pose, reference = move_pose_reach.apply_reference(
                    self, arm, compiled, current_pose=current[arm],
                    interaction=str(interaction), observation=observation,
                )
            action_targets[arm] = target_pose
            action_interactions[arm] = str(interaction)
            public = self._public_sparse_pose_compilation(compiled)
            public.update({
                "position_mode": compiled["position_mode"],
                "orientation_mode": compiled["orientation_mode"],
                "position_delta_frame": compiled["position_delta_frame"],
                "orientation_delta_frame": compiled["orientation_delta_frame"],
            })
            public.update(pose_target_annotations.merged(self, arm, compiled, reference, observation, root))
            compiled_targets[arm] = public
        return self._atomic_submit(
            control=control,
            action_name="move_pose",
            arguments={
                "targets": action_targets,
                "interactions": action_interactions,
            },
            compiled_pose={"targets": compiled_targets},
            schema=MOTION_ACTION_V3_SCHEMA,
        )

    def _submit_pose_primitive_v2(
        self, arguments: Mapping[str, Any], *, primitive: str,
        request_for_target: Callable[[Any], Mapping[str, Any]],
        flat_scalar_key: str | None = None,
    ) -> dict[str, Any]:
        """Compile a restricted semantic primitive through canonical move_pose."""

        self._ensure_mutation_available(primitive)
        targets = self._surface_targets(arguments, scalar_key=flat_scalar_key)
        control, root, observation, _pose, _gripper = self._current_grip_state()
        current = self._surface_arm_states(observation)
        compiled_targets: dict[str, Any] = {}
        action_targets: dict[str, Any] = {}
        for arm, raw_target in targets.items():
            try:
                request = request_for_target(raw_target)
                compiled = compile_enhanced_move_pose(
                    request,
                    current_pose_xyz_xyzw=current[arm],
                    controlled_point="end_effector",
                )
            except (TypeError, ValueError) as exc:
                raise motion_refusals.primitive_refusal(
                    exc, primitive=primitive,
                    field=motion_refusals.primitive_field(self.surface, arm, primitive),
                ) from exc
            target_pose = {
                "position_m": list(compiled["target_pose_xyz_xyzw"][:3]),
                "orientation_xyzw": list(compiled["target_pose_xyz_xyzw"][3:]),
            }
            self._validate_target_workspace(
                {"pose": target_pose}, current_position_m=current[arm][:3],
                field=(motion_refusals.target_field(self.surface, arm, "position_m")
                       if primitive == "move_to"
                       else motion_refusals.primitive_field(self.surface, arm, primitive)),
            )
            action_targets[arm] = target_pose
            public = self._public_sparse_pose_compilation(compiled)
            public.update({
                "semantic_primitive": primitive,
                "position_mode": compiled["position_mode"],
                "orientation_mode": compiled["orientation_mode"],
                "position_delta_frame": compiled["position_delta_frame"],
                "orientation_delta_frame": compiled["orientation_delta_frame"],
            })
            reach = move_pose_reach.primitive_reference(
                self, arm, compiled, current_pose=current[arm],
                interaction="free_space" if primitive == "move_to" else "precision")
            public.update(pose_target_annotations.merged(self, arm, compiled, reach, observation, root))
            compiled_targets[arm] = public
        return self._atomic_submit(
            control=control,
            action_name="move_pose",
            arguments={
                "targets": action_targets,
                "interactions": {
                    arm: (
                        "free_space" if primitive == "move_to" else "precision"
                    )
                    for arm in action_targets
                },
            },
            compiled_pose={
                "semantic_primitive": primitive,
                "canonical_action": "move_pose",
                "targets": compiled_targets,
            },
            schema=MOTION_ACTION_V3_SCHEMA,
            public_action_name=primitive,
        )

    def _motion_move_to_primitive_v2(
        self, arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        def request(raw_target: Any) -> Mapping[str, Any]:
            if not isinstance(raw_target, Mapping) or set(raw_target) != {
                "position_m",
            }:
                raise ValueError(
                    "each move_to target must contain only position_m"
                )
            return {
                "position": {
                    "mode": "absolute", "position_m": raw_target["position_m"],
                },
                "orientation": {"mode": "hold_current"},
            }

        return self._submit_pose_primitive_v2(
            arguments, primitive="move_to", request_for_target=request,
        )

    def _motion_rotate_wrist_v2(
        self, arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        # rotate_wrist rotates about the embodiment's APPROACH axis to align
        # the jaw-closing line. ALOHA's local frame puts approach on +X;
        # the Panda grip site puts approach on +Z (x is the jaw-separation
        # axis) — the semantic axis is a declared embodiment fact, never a
        # copied letter (EXP08A parity boundary).
        return self._motion_rotate_primitive_v2(
            arguments, primitive="rotate_wrist",
            axis=(
                "z" if self.surface.supports("panda_grip_site_axes_v1")
                else "x"
            ),
        )

    def _motion_rotate_pitch_v2(
        self, arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        return self._motion_rotate_primitive_v2(
            arguments, primitive="rotate_pitch", axis="y",
        )

    def _motion_rotate_primitive_v2(
        self, arguments: Mapping[str, Any], *, primitive: str, axis: str,
    ) -> dict[str, Any]:
        def request(raw_target: Any) -> Mapping[str, Any]:
            if (
                isinstance(raw_target, bool)
                or not isinstance(raw_target, (int, float))
                or not math.isfinite(float(raw_target))
            ):
                raise ValueError(
                    f"each {primitive} target must be a finite angle in degrees"
                )
            if abs(float(raw_target)) > motion_refusals.ANGLE_BOUND_DEG:
                raise motion_refusals.PrimitiveAngleError(
                    f"each {primitive} target must be within [-180, 180] degrees"
                )
            return {
                "position": {"mode": "hold_current"},
                "orientation": {
                    "mode": "axis_angle_delta",
                    "axis": axis,
                    "angle_deg": float(raw_target),
                    "frame": "end_effector",
                },
            }

        return self._submit_pose_primitive_v2(
            arguments, primitive=primitive, request_for_target=request,
            flat_scalar_key="angle_deg",
        )
