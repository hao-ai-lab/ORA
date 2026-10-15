"""Materialize the deliberately small policy context for one observation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from PIL import Image

from .action_views import (
    PostActionObservationError,
    materialize_post_action_observation,
)
from .public_facts import (
    PublicFactsError, controlled_point_projection_delivery, full_public_delivery,
    gripper_command_delivery,
)
from .contact_point_projection import project_policy_public_facts


class ContextError(RuntimeError):
    """A policy-visible observation or context resource is invalid."""


class ContextPolicy(Protocol):
    """Minimal projection seam for future context-policy experiments."""

    def project(self, context: Mapping[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class CurrentObservationContextPolicy:
    """Preserve Runtime v1's current-observation projection exactly."""

    def project(self, context: Mapping[str, Any]) -> dict[str, Any]:
        return dict(context)


def build_context_policy(name: str) -> ContextPolicy:
    """Resolve a named, RunSpec-visible context projection."""
    if name == "current_observation_v1":
        return CurrentObservationContextPolicy()
    raise ContextError(f"unknown context projection: {name}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


@dataclass
class ContextProjector:
    """Expose one head image plus local RGB-D files, without simulator state."""

    workspace: Path
    control_root: Path
    prompt_path: Path
    manual_path: Path
    project_instructions_path: Path
    main_camera: str = "head_camera"
    expose_manual: bool = True
    project_instruction_filename: str = "AGENTS.md"
    run_root: Path | None = None
    view_config: Mapping[str, Any] | None = None
    camera_roles: Mapping[str, str] | None = None
    #: The bound ToolSurface's capabilities; they select the facts' delivery rules.
    capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.workspace = self.workspace.resolve()
        self.control_root = self.control_root.resolve()
        self.prompt_path = self.prompt_path.resolve()
        self.manual_path = self.manual_path.resolve()
        self.project_instructions_path = self.project_instructions_path.resolve()
        if self.project_instruction_filename not in {"AGENTS.md", "CLAUDE.md"}:
            raise ContextError("unsupported managed-Agent project instruction filename")
        for path in (self.prompt_path, self.manual_path, self.project_instructions_path):
            if not path.is_file():
                raise ContextError(f"missing Harness resource: {path}")
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.control_root.mkdir(parents=True, exist_ok=True)
        for relative in ("contract", "inputs", "scratch"):
            (self.workspace / relative).mkdir(parents=True, exist_ok=True)
        manual_target = self.workspace / "contract/robot-manual.md"
        if self.expose_manual:
            shutil.copyfile(self.manual_path, manual_target)
            manual_target.chmod(0o444)
        elif manual_target.exists():
            manual_target.unlink()
        prefix_target = self.workspace / "contract/episode-prefix.md"
        project_target = self.workspace / self.project_instruction_filename
        shutil.copyfile(self.prompt_path, prefix_target)
        shutil.copyfile(self.project_instructions_path, project_target)
        prefix_target.chmod(0o444)
        project_target.chmod(0o444)
        stale_git_link = self.workspace / "scratch/.git"
        if stale_git_link.exists() or stale_git_link.is_symlink():
            stale_git_link.unlink()
        for name, target in {
            self.project_instruction_filename: f"../{self.project_instruction_filename}",
            "contract": "../contract",
            "inputs": "../inputs",
        }.items():
            link = self.workspace / "scratch" / name
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(target)

    def publish(
        self,
        *,
        observation: Mapping[str, Any],
        observation_root: Path,
        task_instruction: str,
        remaining_budget: Mapping[str, Any],
        policy_state: Mapping[str, Any],
        runtime_notice: Mapping[str, Any] | None = None,
        previous_execution_receipt: Mapping[str, Any] | None = None,
        previous_action: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if observation.get("schema") != "ora-observation.v1":
            raise ContextError("observation schema mismatch")
        cameras = observation.get("cameras")
        if not isinstance(cameras, Mapping) or self.main_camera not in cameras:
            raise ContextError(f"main camera is unavailable: {self.main_camera}")
        absent = set(observation.get("forbidden_modalities_absent", []))
        required_absent = {
            "actor_state", "object_gt_pose", "segmentation", "contact",
            "success_predicate", "expert_program", "task_source",
        }
        if not required_absent.issubset(absent):
            raise ContextError("observation lacks the privilege-boundary declaration")
        source_root = observation_root.resolve()
        if not source_root.is_dir() or source_root.is_symlink():
            raise ContextError("observation root is unavailable or symlinked")
        observation_id = str(observation.get("observation_id", ""))
        generation = hashlib.sha256(
            json.dumps(observation, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        post_action = None
        public_facts = None
        if full_public_delivery(self.view_config):
            try:
                public_facts = project_policy_public_facts(
                    observation=observation, observation_root=source_root, view_config=self.view_config,
                    gripper_command_delivered=gripper_command_delivery(self.view_config),
                    controlled_point_projection=controlled_point_projection_delivery(self.view_config),
                )
            except PublicFactsError as exc:
                raise ContextError(str(exc)) from exc
            # This payload may also carry camera_geometry, which publishes the
            # identical matrices for the identical observation. Give them one
            # owner here, where both producers are assembled.
            from .calibration_ownership import delegate_calibration_to_camera_geometry
            public_facts = delegate_calibration_to_camera_geometry(
                public_facts, view_config=self.view_config)
            from .policy_delivery import deliver_public_facts
            public_facts = deliver_public_facts(public_facts, capabilities=self.capabilities)
        if public_facts is not None:
            if self.run_root is None or self.view_config is None:
                raise ContextError("fresh current views require a bound run root and view configuration")
            try:
                post_action = materialize_post_action_observation(
                    run_root=self.run_root, workspace=self.workspace,
                    observation=observation, observation_root=source_root,
                    policy_state=policy_state, previous_action=previous_action,
                    view_config=self.view_config, camera_roles=self.camera_roles or {},
                )
            except PostActionObservationError as exc:
                raise ContextError(str(exc)) from exc
        target = self.workspace / "inputs/current"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source_root, target, symlinks=False)
        full_image = target / self.main_camera / "rgb.png"
        if not full_image.is_file():
            raise ContextError(f"main camera RGB is unavailable: {full_image}")
        thumbnail = target / self.main_camera / "rgb-thumbnail.png"
        with Image.open(full_image) as image:
            image = image.convert("RGB")
            image.thumbnail((512, 512))
            image.save(thumbnail, format="PNG", optimize=True)
        for path in target.rglob("*"):
            if path.is_file():
                path.chmod(0o444)
        task_path = self.workspace / "contract/task.md"
        if task_path.exists():
            task_path.chmod(0o644)
        task_path.write_text(task_instruction.strip() + "\n", encoding="utf-8")
        task_path.chmod(0o444)
        pointer = {
            "schema": "ora.context.v1",
            "observation_fresh": True,
            "main_camera": self.main_camera,
            "main_image": f"inputs/current/{self.main_camera}/rgb-thumbnail.png",
            "proprioception": observation.get("proprioception"),
            "previous_execution": (
                dict(previous_execution_receipt)
                if previous_execution_receipt is not None
                else observation.get("previous_execution_receipt")
            ),
            "remaining_budget": dict(remaining_budget),
            "runtime_notice": None if runtime_notice is None else dict(runtime_notice),
            "policy_state": dict(policy_state),
            "manual": "contract/robot-manual.md" if self.expose_manual else None,
            "task": "contract/task.md",
            "forbidden_modalities_absent": sorted(absent),
        }
        candidate_receipt = pointer["previous_execution"]
        if post_action is not None:
            pointer["post_action_observation"] = post_action
            pointer["image_paths"] = [image["path"] for image in post_action["images"]]
        from .current_visual import full_visual_enabled, materialize_current_visual, visual_feedback
        if full_visual_enabled(self.view_config):
            if self.run_root is None:
                raise ContextError("complete current visual delivery requires a run root")
            try:
                full = materialize_current_visual(
                    run_root=self.run_root, workspace=self.workspace, source_root=source_root,
                    cameras=cameras, policy_state=policy_state, observation_id=observation_id,
                    required_views=tuple(self.view_config["automatic_actual_views"]),
                    observation_sha256=hashlib.sha256((source_root / "observation.json").read_bytes()).hexdigest(),
                )
            except (PostActionObservationError, KeyError, ValueError, OSError) as exc:
                raise ContextError(f"complete current visual delivery failed: {exc}") from exc
            pointer["current_visual_observation"] = full
            pointer["current_visual"] = visual_feedback(full)
            pointer["image_paths"] = [image["path"] for image in full["images"]]

        if self.run_root is not None and self.view_config is not None:
            from .skill_observations import materialize_skill_observations

            skill_images = materialize_skill_observations(
                receipt=candidate_receipt, run_root=self.run_root, workspace=self.workspace,
                camera_names=self.view_config["automatic_actual_views"],
            )
            if skill_images:
                pointer["skill_observations"] = skill_images
                pointer.setdefault("image_paths", []).extend(item["path"] for item in skill_images)

        if public_facts is not None:
            pointer["public_robot_facts"] = public_facts
        _atomic_json(self.workspace / "inputs/current-context.json", pointer)
        control = {
            "observation_id": observation_id,
            "generation": generation,
            "policy_state": dict(policy_state),
            "source_root": str(target),
            "observation_manifest": str(target / "observation.json"),
            "observation_sha256": _sha256(target / "observation.json"),
            "open": True,
        }
        _atomic_json(self.control_root / "current-observation.json", control)
        submitted = self.control_root / "submitted-action.json"
        if submitted.exists():
            submitted.unlink()
        return pointer

    def close_observation(self) -> None:
        path = self.control_root / "current-observation.json"
        if not path.is_file():
            return
        value = json.loads(path.read_text(encoding="utf-8"))
        value["open"] = False
        _atomic_json(path, value)
