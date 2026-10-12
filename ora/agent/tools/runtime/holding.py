"""``grasp_verify_retention``.

Retention verification over immutable public RGB-D evidence.

A mixin of ``core.ManipulationToolRuntime``; it holds
methods only, no dataclass fields.

``load_policy_state`` and ``verify_policy_state`` come from ``common``,
which reads them from ``core`` at call time.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from ora.agent.evidence.policy_state import PolicyStateError
from ora.agent.evidence.receipt_binding import canonical_public_receipt

from ora.agent.tools import retention, target_refusals
from ora.agent.tools.grasp import (
    DEPTH_INLIER_GEOMETRY_CAPABILITY, load_segmented_target, segmented_target_geometry,
    validate_segmented_target_source,
)
from ora.agent.tools.runtime.common import load_policy_state, verify_policy_state
from ora.agent.tools.robot import ToolError


class HoldingRuntimeMixin:
    """``grasp_verify_retention``."""

    def _target_geometry_and_state(
        self, *, target_ref: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Load one immutable target from its own public RGB-D observation."""

        if self.state_root is None:
            raise ToolError(
                "policy-state evidence is unavailable",
                code="artifact_not_found", retryable=False,
            )
        try:
            target, _mask, _overlay = load_segmented_target(
                artifact_root=self.artifact_root, target_ref=target_ref,
            )
        except ValueError as exc:
            raise target_refusals.unavailable_target(exc, target_ref=target_ref) from exc
        identity = target.get("identity")
        observation_id = (
            identity.get("observation_id")
            if isinstance(identity, Mapping) else None
        )
        camera = identity.get("camera") if isinstance(identity, Mapping) else None
        if not isinstance(observation_id, str) or not isinstance(camera, str):
            raise ToolError(
                "target_ref lacks a bound public observation",
                code="target_ref_unavailable", field="target_ref", received=target_ref,
            )
        manifest_path = self.state_root / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ToolError(
                "policy-state manifest is unavailable",
                code="artifact_not_found", retryable=False,
            ) from exc
        matches = [
            item for item in manifest.get("steps", [])
            if isinstance(item, Mapping)
            and item.get("observation_id") == observation_id
            and isinstance(item.get("step"), int)
        ]
        if len(matches) != 1:
            raise ToolError(
                "target_ref observation is unavailable in policy state",
                code="artifact_not_found", field="target_ref",
            )
        try:
            state = load_policy_state(
                self.state_root.parent, int(matches[0]["step"]),
            )
        except PolicyStateError as exc:
            raise ToolError(
                str(exc), code="artifact_hash_mismatch", retryable=False,
            ) from exc
        _step, selected_state, root, observation = self._select_observation(
            int(matches[0]["step"])
        )
        if (
            selected_state is None
            or state.get("state_sha256") != selected_state.get("state_sha256")
            or state.get("observation", {}).get("id") != observation_id
        ):
            raise ToolError(
                "target_ref policy-state binding drifted",
                code="artifact_hash_mismatch", retryable=False,
            )
        _record, _rgb, depth_path, calibration_path = self._camera(
            root, observation, camera,
        )
        try:
            if retention.enabled(self.surface):
                validate_segmented_target_source(
                    target, observation_id=observation_id, rgb_path=_rgb,
                    depth_path=depth_path, calibration_path=calibration_path,
                )
            geometry = segmented_target_geometry(
                artifact_root=self.artifact_root,
                target_ref=target_ref,
                depth_path=depth_path,
                calibration_path=calibration_path,
                voxel_size_m=0.003 if retention.enabled(self.surface) else 0.01,
                use_depth_inliers=self.surface.supports(DEPTH_INLIER_GEOMETRY_CAPABILITY),
                rgb_path=_rgb,
            )
        except ValueError as exc:
            raise ToolError(
                str(exc), code="target_geometry_unavailable", retryable=True,
                field="target_ref", received=target_ref,
            ) from exc
        if retention.enabled(self.surface):
            import numpy as np

            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            extrinsic = np.asarray(calibration["extrinsic_cv"], dtype=np.float64)
            if extrinsic.shape == (3, 4):
                extrinsic = np.vstack([extrinsic, [0.0, 0.0, 0.0, 1.0]])
            geometry["camera_origin_world_m"] = np.linalg.inv(extrinsic)[:3, 3].tolist()
            summary = target.get("summary", {})
            localization = target.get("localization", {})
            bbox = summary.get("bbox_xyxy", [])
            geometry.setdefault("source_quality", {
                "depth_valid_fraction": float(summary.get("depth_valid_fraction", 0.0)),
                "depth_inlier_fraction": float(localization.get("depth_inlier_fraction", 0.0)),
                "primary_component_fraction": float(localization.get("primary_component_fraction", 0.0)),
                "touches_image_boundary": len(bbox) != 4 or bbox[0] <= 0 or bbox[1] <= 0
                or bbox[2] >= int(_record["width"]) or bbox[3] >= int(_record["height"]),
            })
        return geometry, state

    def _grasp_verify_retention(
        self, arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Compare target/TCP motion using only immutable public RGB-D evidence."""

        surface = getattr(self, "surface", None)
        dual = surface is not None and surface.supports("dual_arm_grasp_plan_v1")
        provided = set(arguments)
        from ora.agent.tools import retention_recovery
        recoverable = retention_recovery.enabled(surface)
        required = {"before_target_ref", "current_target_ref"}
        if dual:
            required = required | {"arm"}
        optional = {"arm", "additional_pairs"} if recoverable else {"arm"}
        if not required <= provided or provided - required - optional:
            raise ToolError(
                "grasp_verify_retention requires "
                + " and ".join(sorted(required))
                + ("" if dual else " (arm defaults to the single right arm)"),
                code="invalid_arguments",
            )
        arm = arguments.get("arm", "right")
        before_ref = arguments.get("before_target_ref")
        current_ref = arguments.get("current_target_ref")
        if arm not in {"left", "right"}:
            raise ToolError("arm must be left or right", field="arm")
        if not isinstance(before_ref, str) or not isinstance(current_ref, str):
            raise ToolError("target references must be strings")
        control, _root, _observation, _pose, _gripper = self._current_grip_state()
        before_geometry, before_state = self._target_geometry_and_state(
            target_ref=before_ref,
        )
        current_geometry, current_state = self._target_geometry_and_state(
            target_ref=current_ref,
        )
        if current_geometry.get("observation_id") != control["observation_id"]:
            raise ToolError(
                "current_target_ref must come from the current observation",
                code="stale_observation", field="current_target_ref", received=current_ref, retryable=True,
                retry_requires_changed_arguments=True,
            )

        if surface is not None and retention.enabled(surface):
            if surface.supports(retention.ADVISORY_CAPABILITY):
                # Preserve H's strict round-local reference boundary when
                # estimating retention, independently of motion admission.
                from ora.agent.tools.placement_runtime import _timeline
                states, round_start = _timeline(self, control)
                if not states[round_start]["step"] <= before_state["step"] < current_state["step"]:
                    raise ToolError("retention references must advance within the current round", code="stale_observation")
            run_root = self.state_root.parent
            cache = {before_state["step"]: before_state, current_state["step"]: current_state}
            timeline = None

            def normalized(state: Mapping[str, Any]) -> Mapping[str, Any]:
                return {**state, "previous_execution_receipt": canonical_public_receipt(state, run_root)}

            def load_step(step: int) -> Mapping[str, Any]:
                nonlocal timeline
                if step not in cache:
                    if timeline is None:
                        checked = verify_policy_state(run_root, verify_artifacts=False)
                        if checked["status"] != "valid":
                            raise ToolError("retention public-state history is invalid", code="artifact_hash_mismatch")
                        timeline = checked["steps"]
                    if not 0 <= step < len(timeline):
                        raise ToolError("retention state is unavailable", code="artifact_not_found")
                    cache[step] = timeline[step]
                return normalized(cache[step])

            if recoverable:
                result = retention_recovery.compare_views(
                    arm=arm, before_ref=before_ref, current_ref=current_ref,
                    additional_pairs=arguments.get("additional_pairs", []),
                    load_target=lambda ref: self._target_geometry_and_state(target_ref=ref),
                    load_step=load_step, current_state=current_state,
                )
            else:
                result = retention.verify_retention(
                    arm=arm, before_ref=before_ref, current_ref=current_ref,
                    before_geometry=before_geometry, current_geometry=current_geometry,
                    before_state=normalized(before_state), current_state=normalized(current_state),
                    load_step=load_step,
                    full_reference=surface.supports(retention.FULL_REFERENCE_CAPABILITY),
                )
            if surface.supports(retention.ADVISORY_CAPABILITY):
                result["schema"] = retention.ADVISORY_SCHEMA
                result["assessment_role"] = "advisory_only"
                # Source-paired measurements stand alone. The old recovery
                # protocol describes a permit/budget that this surface lacks.
                result.pop("recovery", None)
            return result

        raise ToolError(
            "grasp_verify_retention requires the source-bound retention capability",
            code="tool_surface_capability_mismatch", retryable=False,
        )
