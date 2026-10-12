"""The causal current observation a Native observation message shows.

A mixin of ``session.NativePlanner``; it holds methods only, no dataclass
fields. ``_actual_snapshot`` resolves, from one immutable policy-state
snapshot, the policy-safe images (configured actual views, the auxiliary
contact sheet, Servo post-action images or the complete current visual), the
carried public extras and the measured end-effector state in the shape the
ToolSurface declares.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping

from ora.agent.context import causal_provider_value
from ora.agent.context.action_views import (
    PostActionObservationError, resolve_post_action_images,
)
from ora.agent.contracts import PolicySessionError

from .observation_projection import (
    AUXILIARY_CONTACT_SHEET_CAMERAS,
    AUXILIARY_CONTACT_SHEET_VIEW,
    ObservationProjectionError,
    materialize_auxiliary_contact_sheet,
)


class NativeActualSnapshotMixin:
    """The causal current observation a Native observation message shows."""

    def _actual_snapshot(self, context: Mapping[str, Any]) -> dict[str, Any]:
        """Resolve policy-safe actual images from the immutable state snapshot."""

        state_ref, snapshot, cameras = self._load_actual_state(context)
        surface = getattr(self.tools, "surface", None)
        view_config = getattr(surface, "view_config", None)
        selected_views = (
            list(view_config["automatic_actual_views"])
            if isinstance(view_config, Mapping) else sorted(cameras)
        )
        try:
            images = resolve_post_action_images(context, root=self.run_root, path_key="run_path")
        except PostActionObservationError as exc:
            raise PolicySessionError(str(exc)) from exc
        post_action = context.get("post_action_observation")
        extra = self._actual_snapshot_extras(context, snapshot, surface)
        if post_action is not None:
            self._bind_post_action_snapshot(context, state_ref, snapshot, post_action)
            selected_views = []
            extra["post_action_observation"] = copy.deepcopy(post_action)
        from ora.agent.context.current_visual import (
            full_visual_enabled, materialize_current_visual,
        )
        if full_visual_enabled(view_config):
            if not self.supports_images:
                raise PolicySessionError("complete current visual delivery requires image support")
            try:
                full = materialize_current_visual(
                    run_root=self.run_root, workspace=self.run_root / "native",
                    source_root=self.run_root, cameras=cameras, policy_state=state_ref,
                    required_views=tuple(view_config["automatic_actual_views"]),
                    observation_id=snapshot.get("observation", {}).get("id", ""),
                    observation_sha256=snapshot.get("observation", {}).get("manifest", {}).get("sha256"),
                )
                images = resolve_post_action_images(
                    {"policy_state": state_ref, "current_visual_observation": full},
                    root=self.run_root, path_key="run_path",
                )
            except (PostActionObservationError, KeyError, ValueError, OSError) as exc:
                raise PolicySessionError(f"complete current visual delivery failed: {exc}") from exc
            extra["current_visual_observation"] = full
            selected_views = []
        for name in selected_views:
            image = self._actual_view_image(context, name, cameras, surface, view_config)
            if image is not None:
                images.append(image)
        if not images:
            raise PolicySessionError("causal actual observation has no RGB images")
        return self._actual_end_effector_snapshot(context, images, extra)

    def _load_actual_state(
        self, context: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], dict[str, Any], Mapping[str, Any]]:
        """Read the bound policy-state snapshot and its camera table."""

        state_ref = context.get("policy_state")
        if not isinstance(state_ref, Mapping) or not isinstance(state_ref.get("path"), str):
            raise PolicySessionError("causal context lacks a policy-state snapshot")
        state_path = (self.run_root / state_ref["path"]).resolve()
        if (
            not state_path.is_file()
            or state_path.is_symlink()
            or not state_path.is_relative_to(self.run_root)
        ):
            raise PolicySessionError("causal policy-state snapshot is unavailable")
        try:
            snapshot = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PolicySessionError("causal policy-state snapshot is invalid") from exc
        cameras = snapshot.get("cameras")
        if not isinstance(cameras, Mapping):
            raise PolicySessionError("causal policy-state snapshot lacks cameras")
        return state_ref, snapshot, cameras

    def _actual_snapshot_extras(
        self, context: Mapping[str, Any], snapshot: Mapping[str, Any], surface: Any,
    ) -> dict[str, Any]:
        """The carried held-object offset and the current public robot facts."""

        extra = {}
        # The offset is the policy's own prior measurement, re-shown with how
        # stale it is. Nothing is re-measured here: a policy that stopped
        # calling place.plan would otherwise have to re-run text grounding on
        # every object after every move just to know how far it still is.
        if surface is not None and surface.supports("held_object_offset_v1"):
            carried = getattr(self.tools, "last_held_object_offset", None)
            if isinstance(carried, Mapping) and carried.get("observation_id"):
                extra["held_object_offset"] = {
                    "frame": carried["frame"],
                    "translation_m": list(carried["translation_m"]),
                    "basis": carried["basis"],
                    "measured_at_observation_id": carried["observation_id"],
                    "observations_since": max(
                        0,
                        self._observation_step(context)
                        - int(carried.get("observation_step", 0)),
                    ),
                    "claim_boundary": carried["claim_boundary"],
                }
        if surface is not None and surface.supports("current_public_robot_facts_v1"):
            facts = context.get("public_robot_facts")
            if not isinstance(facts, Mapping) or facts.get("observation_id") != snapshot.get("observation", {}).get("id"):
                raise PolicySessionError("current public robot facts lack their observation binding")
            extra["public_robot_facts"] = causal_provider_value(facts)
        return extra

    def _bind_post_action_snapshot(
        self, context: Mapping[str, Any], state_ref: Mapping[str, Any],
        snapshot: Mapping[str, Any], post_action: Mapping[str, Any],
    ) -> None:
        """Servo post-action images must share the snapshot's observation."""

        if not self.supports_images:
            raise PolicySessionError("Servo post-action observation requires image support")
        if (
            snapshot.get("state_sha256") != state_ref.get("state_sha256")
            or snapshot.get("observation", {}).get("id") != post_action.get("observation_id")
            or snapshot.get("observation", {}).get("manifest", {}).get("sha256") != post_action.get("observation_sha256")
            or snapshot.get("proprioception") != context.get("proprioception")
        ):
            raise PolicySessionError("Servo images and measurements do not share an observation")

    def _actual_view_image(
        self, context: Mapping[str, Any], name: Any, cameras: Mapping[str, Any],
        surface: Any, view_config: Any,
    ) -> dict[str, Any] | None:
        """One configured actual view; ``None`` when its camera has no RGB file."""

        if name == AUXILIARY_CONTACT_SHEET_VIEW:
            return self._auxiliary_contact_sheet_image(context, name, cameras, surface, view_config)
        camera = cameras.get(name)
        if not isinstance(camera, Mapping):
            raise PolicySessionError(
                f"configured causal actual view is unavailable: {name}"
            )
        files = camera.get("files") if isinstance(camera, Mapping) else None
        if not isinstance(files, Mapping):
            return None
        rgb = files.get("rgb_lossless_png") or files.get("rgb")
        if not isinstance(rgb, Mapping) or not isinstance(rgb.get("path"), str):
            return None
        path = (self.run_root / rgb["path"]).resolve()
        if not path.is_file() or path.is_symlink() or not path.is_relative_to(self.run_root):
            raise PolicySessionError(f"causal actual image is unavailable: {name}")
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
        configured_resolution = (
            view_config.get("rgb_resolution")
            if isinstance(view_config, Mapping) else "native"
        )
        if configured_resolution != "native" and [width, height] != list(
            configured_resolution
        ):
            raise PolicySessionError(
                f"causal actual view resolution differs from ToolSurface: {name}"
            )
        return {
            "camera": str(name), "path": path,
            "width": width, "height": height,
        }

    def _auxiliary_contact_sheet_image(
        self, context: Mapping[str, Any], name: Any, cameras: Mapping[str, Any],
        surface: Any, view_config: Any,
    ) -> dict[str, Any]:
        """Derive the auxiliary contact sheet from its source cameras."""

        if surface is None or not surface.supports(
            "automatic_current_auxiliary_contact_sheet_v1"
        ):
            raise PolicySessionError(
                "auxiliary contact sheet lacks ToolSurface capability"
            )
        configured_sources = (
            view_config.get("auxiliary_contact_sheet_cameras")
            if isinstance(view_config, Mapping) else None
        )
        sheet_cameras = (
            tuple(configured_sources)
            if isinstance(configured_sources, list) and configured_sources
            else AUXILIARY_CONTACT_SHEET_CAMERAS
        )
        source_paths: dict[str, Path] = {}
        for source_name in sheet_cameras:
            source_camera = cameras.get(source_name)
            source_files = (
                source_camera.get("files")
                if isinstance(source_camera, Mapping) else None
            )
            source_rgb = (
                source_files.get("rgb_lossless_png") or source_files.get("rgb")
                if isinstance(source_files, Mapping) else None
            )
            if not isinstance(source_rgb, Mapping) or not isinstance(
                source_rgb.get("path"), str,
            ):
                raise PolicySessionError(
                    f"auxiliary contact sheet source is unavailable: {source_name}"
                )
            source_paths[source_name] = (
                self.run_root / source_rgb["path"]
            ).resolve()
        try:
            derived = materialize_auxiliary_contact_sheet(
                run_root=self.run_root,
                observation_step=self._observation_step(context),
                camera_paths=source_paths,
                cameras=sheet_cameras,
            )
        except ObservationProjectionError as exc:
            raise PolicySessionError(str(exc)) from exc
        return {
            "camera": name,
            "path": derived["path"],
            "width": derived["width"],
            "height": derived["height"],
        }

    def _actual_end_effector_snapshot(
        self, context: Mapping[str, Any], images: list[dict[str, Any]],
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        """Attach the measured end-effector state in the surface's arm shape."""

        proprioception = context.get("proprioception")
        end_effectors = (
            proprioception.get("end_effectors")
            if isinstance(proprioception, Mapping) else None
        )
        if not isinstance(end_effectors, Mapping):
            raise PolicySessionError("causal actual observation lacks end-effector state")
        surface = getattr(self.tools, "surface", None)
        if (
            surface is not None
            and surface.supports("dual_arm_sparse_targets_v1")
        ):
            measured = self._dual_arm_target_state(end_effectors, surface)
            return {
                "step": self._observation_step(context),
                "images": images,
                "end_effectors": measured,
                **extra,
            }
        bound = getattr(self.tools, "controlled_end_effector", None)
        if not isinstance(bound, str):
            available = [
                str(name) for name, value in end_effectors.items()
                if isinstance(value, Mapping)
            ]
            if len(available) != 1:
                raise PolicySessionError("causal context requires one bound end effector")
            bound = available[0]
        eef = end_effectors.get(bound)
        if not isinstance(eef, Mapping):
            raise PolicySessionError("bound end effector is absent from causal context")
        pose = eef.get("pose_xyz_xyzw")
        aperture = eef.get("gripper")
        if not (
            isinstance(pose, list) and len(pose) == 7
            and isinstance(aperture, (int, float)) and not isinstance(aperture, bool)
        ):
            raise PolicySessionError("causal end-effector state is invalid")
        return {
            "step": self._observation_step(context),
            "images": images,
            "eef_pose_xyz_xyzw": [round(float(value), 6) for value in pose],
            "aperture_mm": round(float(aperture) * 80.0, 3),
            **extra,
        }

    @staticmethod
    def _dual_arm_target_state(
        end_effectors: Mapping[str, Any], surface: Any,
    ) -> dict[str, dict[str, Any]]:
        """Both arms' poses and command targets on a sparse-target surface."""

        measured: dict[str, dict[str, Any]] = {}
        for arm in ("left", "right"):
            eef = end_effectors.get(arm)
            pose = eef.get("pose_xyz_xyzw") if isinstance(eef, Mapping) else None
            command = (
                eef.get("gripper_command_target")
                if isinstance(eef, Mapping) else None
            )
            if (
                not isinstance(pose, list) or len(pose) != 7
                or command not in {"open", "close", "unknown"}
            ):
                raise PolicySessionError(
                    f"dual-arm causal {arm} command-target state is invalid"
                )
            measured[arm] = {
                "pose_xyz_xyzw": [round(float(value), 6) for value in pose],
                "gripper_command_target": command,
            }
            # The measured jaw separation is the only quantity the policy can
            # compare against an object's extents; the command target alone is
            # an echo of its own last request. Carry it whenever the backend
            # measured one, and carry the explicit null when it could not.
            if surface.supports("measured_jaw_separation_v1") and "gripper_opening_m" in eef:
                opening = eef["gripper_opening_m"]
                if opening is not None and not (
                    isinstance(opening, (int, float)) and not isinstance(opening, bool)
                    and math.isfinite(float(opening)) and float(opening) >= 0.0
                ):
                    raise PolicySessionError(
                        f"dual-arm causal {arm} measured jaw separation is invalid"
                    )
                measured[arm]["gripper_opening_m"] = (
                    None if opening is None else round(float(opening), 6)
                )
            semantics = eef.get("controlled_point_semantics")
            if semantics is not None:
                if semantics != "parallel_jaw_grip_center_tcp":
                    raise PolicySessionError(
                        f"dual-arm causal {arm} controlled point is invalid"
                    )
                measured[arm]["controlled_point_semantics"] = semantics
        return measured
