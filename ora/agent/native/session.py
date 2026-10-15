"""Native policy session over the shared provider, context and tool ports."""

from __future__ import annotations

import copy
import json
import math
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ora.agent.context import (
    NativeContextPipeline, causal_provider_value,
    RolloutCheckpointCompactor, episode_index_keyframe,
)
from ora.agent.context.action_views import (
    PostActionObservationError, resolve_post_action_images,
)
from ora.agent.context.payload_dedup import (
    CAPABILITY as DEDUP_CAPABILITY, hoist_camera_geometry_conventions,
)
from ora.agent.context import round_static_facts
from ora.agent.context.delivery_trim import declared_rules
from ora.agent.context.delivery_trim_native import (
    trim_observation_envelope, trim_view_freshness,
)
from ora.agent.native.receipt_projection import (
    ReceiptProjectionContext,
    resolve_receipt_effect_projector,
    public_motion_effect_envelope,
    public_motion_failure_code,
)
from ora.agent.contracts import (
    ModelProvider,
    PolicySessionError,
)
from ora.agent.tools.robot import RobotToolRuntime, ToolError
from ora.agent.evidence.durability import DeferredDurability
from ora.agent.evidence.trajectory import (
    ArtifactStore, TrajectoryRecorder, canonical_bytes, image_data_url,
)
from ora.agent.memory.artifacts import (
    ContextArtifactCatalog,
    LOSSLESS_CONTEXT_RETENTION,
)
from ora.contracts import (
    POSE_ORIENTATION_TOLERANCE_RAD,
    POSE_POSITION_TOLERANCE_M,
)

from .stage_guard import native_turn_instruction
from .constants import (
    COMPOSITION_CONTEXT_CAPABILITIES, NATIVE_CAUSAL_SLOTS, NATIVE_SLOTS_V4,
    TOOL_IMAGE_CALL_LABELS,
)
from .composition import assemble_native_v3_static_context
from .gateway import NativeToolGateway, openai_tool_definitions
from .request_assembly import NativeRequestAssembler
from .engine import NativeSessionEngine
from .tool_loop import NativeTurnLoop, _arguments, _tool_error
from .session_manifest import SessionManifestWriter
from .operator_messages import NativeOperatorMessagesMixin
from .actual_snapshot import NativeActualSnapshotMixin
# Re-exported: importable from this module since before the causal
# current-observation snapshot moved to ``actual_snapshot``.
from .observation_projection import (  # noqa: F401
    AUXILIARY_CONTACT_SHEET_CAMERAS,
    AUXILIARY_CONTACT_SHEET_VIEW,
    ObservationProjectionError,
    materialize_auxiliary_contact_sheet,
)


def _declared_result_properties(schema: Mapping[str, Any]) -> tuple[str, ...]:
    """Collect the declared top-level result fields across schema branches."""

    found: set[str] = set()
    properties = schema.get("properties")
    if isinstance(properties, Mapping):
        found.update(str(key) for key in properties)
    for keyword in ("oneOf", "anyOf", "allOf"):
        branches = schema.get(keyword)
        if not isinstance(branches, list):
            continue
        for branch in branches:
            if isinstance(branch, Mapping):
                found.update(_declared_result_properties(branch))
    return tuple(sorted(found))


@dataclass
class NativePlanner(NativeActualSnapshotMixin, NativeOperatorMessagesMixin):
    """Observation-scoped policy session over one canonical message IR."""

    run_root: Path
    run_id: str
    tools: Any
    provider: ModelProvider
    project_instructions_path: Path
    episode_prefix_path: Path
    model: str
    context_profile: str
    context_window_tokens: int
    reasoning_effort: str = "provider_default"
    max_output_tokens: int = 8192
    stream: bool = True
    supports_images: bool = True
    initial_observation: str = "automatic"
    summary_policy: str = "disabled"
    max_model_calls_per_observation: int = 12
    context_composition: str | None = None
    harness_system_path: Path | None = None
    task_id: str | None = None
    context_retention: str = "rewrite_compact_v1"
    context_management: Mapping[str, Any] | None = None
    max_inline_images_per_request: int | None = None
    compactor_provider: ModelProvider | None = None
    policy_session_id: str = field(default_factory=lambda: f"native-policy-{uuid.uuid4().hex}")

    def __post_init__(self) -> None:
        self.run_root = self.run_root.resolve()
        self.project_instructions_path = self.project_instructions_path.resolve()
        self.episode_prefix_path = self.episode_prefix_path.resolve()
        if self.harness_system_path is not None:
            self.harness_system_path = self.harness_system_path.resolve()
        paths = [self.project_instructions_path, self.episode_prefix_path]
        if self.harness_system_path is not None:
            paths.append(self.harness_system_path)
        for path in paths:
            if not path.is_file():
                raise PolicySessionError(f"native policy context is unavailable: {path}")
        if self.initial_observation not in {"automatic", "tool_pull"}:
            raise PolicySessionError(
                "native initial_observation must be automatic or tool_pull"
            )
        if self.context_composition not in {None, *NATIVE_CAUSAL_SLOTS}:
            raise PolicySessionError("unsupported native context composition")
        if (
            self.context_composition == NATIVE_SLOTS_V4
            and self.context_retention != LOSSLESS_CONTEXT_RETENTION
        ):
            raise PolicySessionError(
                "native_slots_v4 requires lossless ActionStep context retention"
            )
        if self.context_composition in NATIVE_CAUSAL_SLOTS:
            if self.harness_system_path is None:
                raise PolicySessionError("native_slots_v3 requires harness_system_path")
            if self.initial_observation != "automatic":
                raise PolicySessionError("native_slots_v3 requires automatic observation")
            if self.context_profile != "native_causal_context_v1":
                raise PolicySessionError("native_slots_v3 requires causal context profile")
            if self.context_retention not in {
                "rewrite_compact_v1", LOSSLESS_CONTEXT_RETENTION,
            }:
                raise PolicySessionError("native_slots_v3 has unsupported context retention")
            mutations = tuple(getattr(self.tools.surface, "mutation_tools", ()))
            if mutations != (
                "move_pose", "close_gripper", "open_gripper",
            ) and not self.tools.surface.supports("enhanced_move_pose_v1"):
                raise PolicySessionError(
                    "native_slots_v3 requires atomic or registered motion tools"
                )
        if self.max_inline_images_per_request is not None:
            # Only the action_step_pressure_* retentions capped inline images;
            # they compose only at the commits that composed them.
            raise PolicySessionError(
                "max_inline_images_per_request is not supported by any context "
                "retention of this tree"
            )
        # Writes stay ordered on this thread; fsync waits for round/close barriers.
        self.durability = DeferredDurability()
        self.manifest_writer = SessionManifestWriter(
            self.run_root / "native/agent-session.json", self.durability,
        )
        self.artifacts = ArtifactStore(self.run_root, durability=self.durability)
        self.recorder = TrajectoryRecorder(
            self.run_root, self.run_id, self.policy_session_id,
            durability=self.durability,
        )
        self.context_artifacts = None
        bind_context_artifacts = getattr(
            self.tools, "bind_context_artifacts", None,
        )
        if self.context_retention == LOSSLESS_CONTEXT_RETENTION:
            if not callable(bind_context_artifacts):
                raise PolicySessionError(
                    "lossless Context retention lacks its Artifact reader"
                )
            self.context_artifacts = ContextArtifactCatalog(
                self.run_root, self.policy_session_id, self.artifacts,
            )
            bind_context_artifacts(
                self.context_artifacts, event_sink=self._append,
            )
        self.gateway = NativeToolGateway(self.tools)
        self.tool_schemas = [dict(item) for item in self.gateway.definitions()]
        compactor = None
        if self.context_retention == LOSSLESS_CONTEXT_RETENTION:
            provider = self.compactor_provider
            if provider is None:
                provider = copy.copy(self.provider)
                if hasattr(provider, "_owns_client"):
                    setattr(provider, "_owns_client", False)
            if hasattr(provider, "max_retries"):
                setattr(provider, "max_retries", 0)
            compactor = RolloutCheckpointCompactor(
                provider=provider,
                model=self.model,
                reasoning_effort=self.reasoning_effort,
                stream=self.stream,
                artifacts=self.artifacts,
                event_sink=self._append,
            )
        self.rollout_compactor = compactor
        self.pipeline = NativeContextPipeline(
            profile=self.context_profile,
            artifacts=self.artifacts,
            context_window_tokens=self.context_window_tokens,
            max_output_tokens=self.max_output_tokens,
            retention=self.context_retention,
            summary_policy=self.summary_policy,
            summary_provider=compactor,
            context_management=self.context_management,
            mutation_tools=tuple(getattr(self.tools.surface, "mutation_tools", ())),
            context_artifacts=self.context_artifacts,
        )
        self.request_assembler = NativeRequestAssembler(
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            max_output_tokens=self.max_output_tokens,
            stream=self.stream,
            provider=self.provider,
        )
        self.turn_loop = NativeTurnLoop()
        self.project_template = self.project_instructions_path.read_text(
            encoding="utf-8"
        ).strip()
        self.episode_prefix_template = self.episode_prefix_path.read_text(
            encoding="utf-8"
        ).strip()
        self.harness_system_template = (
            self.harness_system_path.read_text(encoding="utf-8").strip()
            if self.harness_system_path is not None else None
        )
        self.engine = NativeSessionEngine()
        self.start()

    @property
    def messages(self) -> list[dict[str, Any]]:
        return self.engine.messages

    @property
    def context_sources(self) -> list[dict[str, Any]]:
        return self.engine.context_sources

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.engine.calls

    @property
    def usage_events(self) -> list[dict[str, Any]]:
        return self.engine.usage_events

    @property
    def reported_totals(self) -> dict[str, int]:
        return self.engine.reported_totals

    @property
    def call_sequence(self) -> int:
        return self.engine.call_sequence

    @call_sequence.setter
    def call_sequence(self, value: int) -> None:
        self.engine.call_sequence = value

    @property
    def closed(self) -> bool:
        return self.engine.closed

    @closed.setter
    def closed(self, value: bool) -> None:
        self.engine.closed = value

    @property
    def interrupted(self) -> bool:
        return self.engine.interrupted

    @interrupted.setter
    def interrupted(self, value: bool) -> None:
        self.engine.interrupted = value

    @property
    def pending_receipt(self) -> dict[str, Any] | None:
        return self.engine.pending_receipt

    @pending_receipt.setter
    def pending_receipt(self, value: dict[str, Any] | None) -> None:
        self.engine.pending_receipt = value

    @property
    def pending_receipt_body(self) -> dict[str, Any] | None:
        return self.engine.pending_receipt_body

    @pending_receipt_body.setter
    def pending_receipt_body(self, value: dict[str, Any] | None) -> None:
        self.engine.pending_receipt_body = value

    @property
    def last_actual_snapshot(self) -> dict[str, Any] | None:
        return self.engine.last_actual_snapshot

    @last_actual_snapshot.setter
    def last_actual_snapshot(self, value: dict[str, Any] | None) -> None:
        self.engine.last_actual_snapshot = value

    @property
    def actual_sequence(self) -> int:
        return self.engine.actual_sequence

    @actual_sequence.setter
    def actual_sequence(self, value: int) -> None:
        self.engine.actual_sequence = value

    @property
    def last_event_id(self) -> str | None:
        return self.engine.last_event_id

    @last_event_id.setter
    def last_event_id(self, value: str | None) -> None:
        self.engine.last_event_id = value

    @property
    def last_action_event_id(self) -> str | None:
        return self.engine.last_action_event_id

    @last_action_event_id.setter
    def last_action_event_id(self, value: str | None) -> None:
        self.engine.last_action_event_id = value

    @property
    def last_action_step(self) -> int | None:
        return self.engine.last_action_step

    @last_action_step.setter
    def last_action_step(self, value: int | None) -> None:
        self.engine.last_action_step = value

    @property
    def last_action_id(self) -> str | None:
        return self.engine.last_action_id

    @last_action_id.setter
    def last_action_id(self, value: str | None) -> None:
        self.engine.last_action_id = value

    @property
    def last_action_public_name(self) -> str | None:
        return self.engine.last_action_public_name

    @last_action_public_name.setter
    def last_action_public_name(self, value: str | None) -> None:
        self.engine.last_action_public_name = value

    @property
    def last_fresh_observation_event_id(self) -> str | None:
        return self.engine.last_fresh_observation_event_id

    @last_fresh_observation_event_id.setter
    def last_fresh_observation_event_id(self, value: str | None) -> None:
        self.engine.last_fresh_observation_event_id = value

    @property
    def system_text(self) -> str | None:
        return self.engine.system_text

    @system_text.setter
    def system_text(self, value: str | None) -> None:
        self.engine.system_text = value

    @property
    def bound_task_instruction(self) -> str | None:
        return self.engine.bound_task_instruction

    @bound_task_instruction.setter
    def bound_task_instruction(self, value: str | None) -> None:
        self.engine.bound_task_instruction = value

    @property
    def task_sha256(self) -> str | None:
        return self.engine.task_sha256

    @task_sha256.setter
    def task_sha256(self, value: str | None) -> None:
        self.engine.task_sha256 = value

    @property
    def started(self) -> bool:
        return self.engine.started

    @started.setter
    def started(self, value: bool) -> None:
        self.engine.started = value

    def _append(self, event_type: str, **fields: Any) -> dict[str, Any]:
        fields.setdefault("causal_parent_id", self.last_event_id)
        event = self.recorder.append(event_type, **fields)
        self.last_event_id = str(event["event_id"])
        return event

    def start(self) -> None:
        if self.started:
            return
        provider_protocol = str(getattr(self.provider, "protocol", "canonical_unknown"))
        payload = {
            "provider_protocol": provider_protocol,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "context_profile": self.context_profile,
            "context_retention": self.context_retention,
            "context_management": self.context_management,
            "max_inline_images_per_request": self.max_inline_images_per_request,
            "context_window_tokens": self.context_window_tokens,
            "max_output_tokens": self.max_output_tokens,
            "stream": self.stream,
            "supports_images": self.supports_images,
            "initial_observation": self.initial_observation,
            "automatic_observation_image": (
                self.supports_images and self.initial_observation == "automatic"
            ),
            "summary_policy": self.summary_policy,
            "context_composition": self.context_composition,
        }
        if self.context_composition in COMPOSITION_CONTEXT_CAPABILITIES:
            payload["context_capabilities"] = list(
                COMPOSITION_CONTEXT_CAPABILITIES[self.context_composition]
            )
        event = self.recorder.append("session_started", payload=payload)
        self.last_event_id = str(event["event_id"])
        self.started = True
        self._write_session_manifest(exit_reason=None)

    def _write_session_manifest(self, *, exit_reason: str | None) -> None:
        provider_protocol = str(getattr(self.provider, "protocol", "canonical_unknown"))
        body = {
            "schema": "ora.native-policy-session.v1",
            "backend": f"native_{provider_protocol}_v1",
            "provider_protocol": provider_protocol,
            "session_id": self.policy_session_id,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "initial_observation": self.initial_observation,
            "context_composition": self.context_composition,
            "context_retention": self.context_retention,
            "context_management": self.context_management,
            "max_inline_images_per_request": self.max_inline_images_per_request,
            "automatic_observation_image": (
                self.supports_images and self.initial_observation == "automatic"
            ),
            "model_context_window": self.context_window_tokens,
            "context_sources": sorted(
                self.context_sources, key=lambda item: int(item["order"])
            ),
            "calls": self.calls,
            "context_compactor_calls": (
                [] if self.rollout_compactor is None
                else copy.deepcopy(self.rollout_compactor.calls)
            ),
            "usage_events": self.usage_events,
            "reported_totals": self.reported_totals,
            "process_pid": os.getpid(),
            "process_invocations": 1,
            "process_restarts": 0,
            "thread_id": None,
            "exit_reason": exit_reason,
        }
        self.manifest_writer.write(body, final=exit_reason is not None)

    @staticmethod
    def _observation_step(context: Mapping[str, Any]) -> int:
        state = context.get("policy_state")
        if not isinstance(state, Mapping) or not isinstance(state.get("step"), int):
            raise PolicySessionError("native context lacks a bound policy-state step")
        return int(state["step"])

    @staticmethod
    def _post_action_view_feedback(current: Mapping[str, Any]) -> dict[str, Any]:
        from ora.agent.context.current_visual import visual_feedback
        full = current.get("current_visual_observation")
        feedback = visual_feedback(full) if isinstance(full, Mapping) else {}
        record = current.get("post_action_observation")
        if not isinstance(record, Mapping):
            return feedback
        return {**feedback, "current_views": {
            "mode": record["mode"], "selected_arms": record["selected_arms"],
            "status": record["status"], "missing_wrist_views": record["missing_wrist_views"],
            "views": [image["camera"] for image in record["images"]],
            "fresh_for_this_action": record.get("action_id") is not None,
        }}

    @staticmethod
    def _receipt_record(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
        nested = receipt.get("receipt")
        return nested if isinstance(nested, Mapping) else receipt

    @staticmethod
    def _pose_residual(
        target: Sequence[Any], actual: Sequence[Any],
    ) -> tuple[float, float]:
        translation = math.sqrt(sum(
            (float(target[index]) - float(actual[index])) ** 2
            for index in range(3)
        ))
        dot = abs(sum(
            float(target[index]) * float(actual[index]) for index in range(3, 7)
        ))
        orientation = 2.0 * math.acos(min(1.0, max(0.0, dot)))
        return translation, orientation

    def _minimal_action_effect(self, receipt: Mapping[str, Any]) -> dict[str, Any]:
        record = self._receipt_record(receipt)
        raw_action = record.get("action")
        if isinstance(raw_action, Mapping):
            action_name = raw_action.get("action")
            arguments = raw_action.get("arguments")
        else:
            action_name = raw_action
            arguments = record.get("arguments")
        arguments = arguments if isinstance(arguments, Mapping) else {}
        projector = resolve_receipt_effect_projector(record.get("schema"))
        if projector is not None:
            # Environment-specific receipt contracts project through the
            # schema-keyed registry; the shared session keeps no per-backend
            # branch (cross-embodiment parity boundary).
            surface = getattr(getattr(self, "tools", None), "surface", None)
            return projector(record, ReceiptProjectionContext(
                last_action_id=getattr(self, "last_action_id", None),
                last_action_public_name=getattr(
                    self, "last_action_public_name", None,
                ),
                surface_supports=(
                    surface.supports if surface is not None else None
                ),
            ))
        if action_name in {"move_pose", "move_free_space", "servo_delta", "recover_motion"}:
            target = record.get("commanded_pose_xyz_xyzw")
            actual = record.get("reached_pose_xyz_xyzw")
            if not (
                isinstance(target, list) and len(target) == 7
                and isinstance(actual, list) and len(actual) == 7
            ):
                motion_status = record.get("motion_status")
                feedback = record.get("feedback")
                reason = (
                    feedback.get("planning_failure_reason")
                    if isinstance(feedback, Mapping) else None
                )
                if motion_status == "refused" and isinstance(reason, str) and reason:
                    return causal_provider_value(public_motion_effect_envelope(
                        action=str(action_name), status="plan_rejected", executor=None,
                        motion_strategy=record.get("motion_strategy"),
                        failure_stage="planner",
                        failure_code=public_motion_failure_code("planner", reason),
                        round_terminated=record.get("terminal_after_action"),
                    ))
                return {"action": str(action_name), "status": "world_state_unknown"}
            translation, orientation = self._pose_residual(target, actual)
            return {
                "action": str(action_name),
                "status": (
                    "executed"
                    if translation <= POSE_POSITION_TOLERANCE_M
                    and orientation <= POSE_ORIENTATION_TOLERANCE_RAD
                    else "not_reached"
                ),
                "target_pose_xyz_xyzw": [round(float(value), 6) for value in target],
                "actual_pose_xyz_xyzw": [round(float(value), 6) for value in actual],
                "position_error_m": round(translation, 6),
                "orientation_error_rad": round(orientation, 6),
            }
        if action_name == "set_gripper":
            gripper = record.get("gripper_evidence")
            before = gripper.get("measured_before") if isinstance(gripper, Mapping) else None
            after = gripper.get("measured_after") if isinstance(gripper, Mapping) else None
            target = arguments.get("gripper")
            label = "close_gripper" if target == 0.0 else "open_gripper"
            if not all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in (before, after)
            ):
                return {"action": label, "status": "world_state_unknown"}
            return {
                "action": label,
                "status": "executed",
                "before_aperture_mm": round(float(before) * 80.0, 3),
                "after_aperture_mm": round(float(after) * 80.0, 3),
            }
        return {"action": "unknown", "status": "world_state_unknown"}

    def _skill_observation_blocks(self) -> list[dict[str, Any]]:
        from ora.agent.context.skill_observations import (
            materialize_skill_observations, native_blocks,
        )

        surface = getattr(self.tools, "surface", None)
        if surface is None or not surface.supports("prepared_grasp_execution_v2"):
            return []
        images = materialize_skill_observations(
            receipt=self.pending_receipt_body, run_root=self.run_root,
            workspace=self.run_root / "native",
            camera_names=surface.view_config["automatic_actual_views"],
        )
        return native_blocks(images, run_root=self.run_root)

    def _causal_observation_message(
        self, *, context: Mapping[str, Any], terminal_lifecycle: bool,
    ) -> dict[str, Any]:
        if self.context_retention != LOSSLESS_CONTEXT_RETENTION:
            return self._rewrite_causal_observation_message(
                context=context,
                terminal_lifecycle=terminal_lifecycle,
            )
        current = self._actual_snapshot(context)
        receipt = self.pending_receipt_body
        before = self.last_actual_snapshot if receipt is not None else None
        if receipt is not None and before is None:
            raise PolicySessionError("mutation receipt lacks its pre-action actual snapshot")
        current_sequence = self.actual_sequence + 1
        current["actual_sequence"] = current_sequence
        before_sequence = (
            before.get("actual_sequence") if isinstance(before, Mapping) else None
        )
        if before is not None and not isinstance(before_sequence, int):
            raise PolicySessionError("causal input actual lacks a stable sequence")
        effect = self._minimal_action_effect(receipt) if receipt is not None else None
        current_actual = {"actual_sequence": current_sequence}
        current_actual.update(self._post_action_view_feedback(current))
        if self._delivery_trim():
            current_actual = trim_view_freshness(current_actual, paired=before is not None)
        if "public_robot_facts" in current:
            current_actual["public_robot_facts"] = copy.deepcopy(current["public_robot_facts"])
        elif "end_effectors" in current:
            current_actual["end_effectors"] = copy.deepcopy(
                current["end_effectors"]
            )
        else:
            current_actual.update({
                "eef_pose_xyz_xyzw": current["eef_pose_xyz_xyzw"],
                "aperture_mm": current["aperture_mm"],
            })
        # current_actual is assembled field by field, so a snapshot key that is
        # not named here never reaches the policy and nothing reports it.
        if "held_object_offset" in current:
            current_actual["held_object_offset"] = copy.deepcopy(
                current["held_object_offset"]
            )
        if getattr(
            getattr(self.tools, "surface", None), "view_config", None,
        ) is not None:
            current_actual["view_dimensions"] = {
                item["camera"]: {
                    "width": item["width"], "height": item["height"],
                }
                for item in current["images"]
            }
        current_actual = self._delivered_current_actual(context, current_actual)
        envelope = {
            "current_actual": current_actual,
            "last_action_effect": effect,
            "comparison": (
                {
                    "available": True,
                    "action_input_actual_sequence": before_sequence,
                    "action_result_actual_sequence": current_sequence,
                    "instruction": (
                        "Compare the retained ACTION INPUT ACTUAL with this ACTION "
                        "RESULT ACTUAL before choosing the next tool."
                    ),
                }
                if before is not None else {
                    "available": False,
                    "instruction": "Use this initial actual evidence to choose the next tool.",
                }
            ),
            "remaining_budget": {
                key: context.get("remaining_budget", {}).get(key)
                for key in (
                    "robot_decisions", "model_calls", "wall_time_s",
                    "native_actions_limit", "native_actions_remaining",
                )
                if key in context.get("remaining_budget", {})
            },
            "terminal_lifecycle": terminal_lifecycle,
        }
        round_notice = context.get("runtime_notice")
        if (
            isinstance(round_notice, Mapping)
            and round_notice.get("kind") == "session_round_start"
        ):
            envelope["session_round"] = {
                key: copy.deepcopy(round_notice[key])
                for key in (
                    "kind", "round", "total_rounds", "budget_scope", "instruction",
                )
                if key in round_notice
            }
        if self._delivery_trim():
            envelope = trim_observation_envelope(envelope)
        content: list[dict[str, Any]] = [{
            "type": "text",
            "text": json.dumps(envelope, ensure_ascii=False, sort_keys=True),
        }]
        current_images = {
            item["camera"]: item["path"] for item in current["images"]
        }
        camera_order = [item["camera"] for item in current["images"]]
        if before is not None:
            before_images = {
                item["camera"]: item["path"] for item in before["images"]
            }
            if set(before_images) != set(current_images) and not (
                current.get("post_action_observation") or before.get("post_action_observation")
            ):
                raise PolicySessionError("causal actual camera set changed across mutation")
            label = "ACTION RESULT ACTUAL"
        else:
            label = "INITIAL ACTUAL"
        for camera in camera_order:
            content.extend([
                {
                    "type": "text",
                    "text": (
                        f"{label} · sequence {current_sequence} · {camera} "
                        "· latest at this message"
                    ),
                },
                {"type": "image_url", "image_url": {
                    "url": image_data_url(current_images[camera]),
                }},
            ])
        content.extend(self._skill_observation_blocks())
        message = {
            "role": "user",
            "content": content,
            "_observation": {
                "step": current["step"],
                "state_ref": dict(context["policy_state"]),
                "observation_id": self._current_observation_id(),
                "keyframe": True,
                "receipt": self.pending_receipt,
            },
            "_full_current_visual": "current_visual_observation" in current,
            "_causal_observation": {
                "schema": "ora.causal-observation.v1",
                "step": current["step"],
                "paired": before is not None,
                "cameras": camera_order,
                "input_actual_sequence": before_sequence,
                "result_actual_sequence": current_sequence,
                "new_actual_image_blocks": len(current_images),
                **({"post_action_observation": current["post_action_observation"]}
                   if "post_action_observation" in current else {}),
            },
            "_session_round": context.get("remaining_budget", {}).get(
                "session_round"
            ),
        }
        self.last_actual_snapshot = current
        self.actual_sequence = current_sequence
        return message

    def _delivery_trim_rules(self) -> frozenset[str]:
        """The trim rules the bound surface declares (``observation_delivery_trim_v2``)."""

        surface = getattr(getattr(self, "tools", None), "surface", None)
        return declared_rules(getattr(surface, "capabilities", ()) or ())

    def _delivery_trim(self) -> bool:
        """Whether the bound surface declares a delivery-trim generation."""

        return bool(self._delivery_trim_rules())

    def _delivered_current_actual(
        self, context: Mapping[str, Any], current_actual: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply the declared once-per-payload rules to one assembled ``current_actual``."""

        surface = getattr(getattr(self, "tools", None), "surface", None)
        if surface is not None and surface.supports(DEDUP_CAPABILITY):
            current_actual = hoist_camera_geometry_conventions(current_actual)
        if getattr(self, "context_composition", None) == NATIVE_SLOTS_V4:
            current_actual = self._round_static_facts_view(context, current_actual)
        return current_actual

    def _round_static_facts_view(
        self, context: Mapping[str, Any], current_actual: dict[str, Any],
    ) -> dict[str, Any]:
        """``round_static_facts_slot_v1``: point fields the round's slot states at it.

        The first observation of a Session round captures the slot; the
        message itself is appended right after that observation by
        :meth:`_append_round_static_facts`.
        """

        round_index = context.get("remaining_budget", {}).get("session_round")
        if isinstance(round_index, bool) or not isinstance(round_index, int) or round_index < 1:
            raise PolicySessionError("native_slots_v4 observation lacks its Session round")
        # observation_delivery_trim_v2: omit every byte-identical field without
        # naming it, and list only the slot fields with no current value.
        listing = (round_static_facts.LISTING_ABSENT if self._delivery_trim()
                   else round_static_facts.LISTING_UNCHANGED)
        state = self.engine.round_static_facts
        if state is None or state["round"] != round_index:
            fields = round_static_facts.capture(current_actual)
            state = {"round": round_index, "fields": fields}
            self.engine.round_static_facts = state
            self.engine.pending_round_static_facts = round_static_facts.slot_message(
                round_index=round_index, actual_sequence=current_actual["actual_sequence"],
                fields=fields, listing=listing,
            )
        return round_static_facts.reduce(
            current_actual, state["fields"], round_index=round_index, listing=listing,
        )

    def _append_round_static_facts(self, observation_event: Mapping[str, Any]) -> None:
        """Append the round's slot right after the observation that captured it.

        Its source is that observation's event, so every L5/L6 packet and the
        ActionStep inventory name the same events.
        """

        pending = getattr(self.engine, "pending_round_static_facts", None)
        if pending is None:
            return
        self.engine.pending_round_static_facts = None
        pending["_source_event_ids"] = [str(observation_event["event_id"])]
        self.messages.append(pending)

    def _rewrite_causal_observation_message(
        self, *, context: Mapping[str, Any], terminal_lifecycle: bool,
    ) -> dict[str, Any]:
        """Preserve the original v3 rewrite/compact projection byte-for-byte."""

        current = self._actual_snapshot(context)
        receipt = self.pending_receipt_body
        before = self.last_actual_snapshot if receipt is not None else None
        if receipt is not None and before is None:
            raise PolicySessionError("mutation receipt lacks its pre-action actual snapshot")
        effect = self._minimal_action_effect(receipt) if receipt is not None else None
        current_actual = self._post_action_view_feedback(current)
        if "public_robot_facts" in current:
            current_actual["public_robot_facts"] = copy.deepcopy(current["public_robot_facts"])
        elif "end_effectors" in current:
            current_actual["end_effectors"] = copy.deepcopy(
                current["end_effectors"]
            )
        else:
            current_actual.update({
                "eef_pose_xyz_xyzw": current["eef_pose_xyz_xyzw"],
                "aperture_mm": current["aperture_mm"],
            })
        if getattr(
            getattr(self.tools, "surface", None), "view_config", None,
        ) is not None:
            current_actual["view_dimensions"] = {
                item["camera"]: {
                    "width": item["width"], "height": item["height"],
                }
                for item in current["images"]
            }
        envelope = {
            "current_actual": current_actual,
            "last_action_effect": effect,
            "comparison": (
                {
                    "available": True,
                    "instruction": (
                        "Compare BEFORE ACTUAL with AFTER ACTUAL / CURRENT before "
                        "choosing the next tool."
                    ),
                }
                if before is not None else {
                    "available": False,
                    "instruction": "Use CURRENT ACTUAL evidence to choose the next tool.",
                }
            ),
            "remaining_budget": {
                key: context.get("remaining_budget", {}).get(key)
                for key in (
                    "robot_decisions", "model_calls", "wall_time_s",
                    "native_actions_limit", "native_actions_remaining",
                )
                if key in context.get("remaining_budget", {})
            },
            "terminal_lifecycle": terminal_lifecycle,
        }
        round_notice = context.get("runtime_notice")
        if (
            isinstance(round_notice, Mapping)
            and round_notice.get("kind") == "session_round_start"
        ):
            envelope["session_round"] = {
                key: copy.deepcopy(round_notice[key])
                for key in (
                    "kind", "round", "total_rounds", "budget_scope", "instruction",
                )
                if key in round_notice
            }
        content: list[dict[str, Any]] = [{
            "type": "text",
            "text": json.dumps(envelope, ensure_ascii=False, sort_keys=True),
        }]
        current_images = {
            item["camera"]: item["path"] for item in current["images"]
        }
        camera_order = [item["camera"] for item in current["images"]]
        if before is not None:
            before_images = {
                item["camera"]: item["path"] for item in before["images"]
            }
            if set(before_images) != set(current_images) and not (
                current.get("post_action_observation") or before.get("post_action_observation")
            ):
                raise PolicySessionError("causal actual camera set changed across mutation")
            for camera in camera_order:
                if camera in before_images:
                    content.extend([
                        {"type": "text", "text": f"BEFORE ACTUAL · {camera}"},
                        {"type": "image_url", "image_url": {
                            "url": image_data_url(before_images[camera]),
                        }},
                    ])
                content.extend([
                    {"type": "text", "text": f"AFTER ACTUAL / CURRENT · {camera}"},
                    {"type": "image_url", "image_url": {
                        "url": image_data_url(current_images[camera]),
                    }},
                ])
        else:
            for camera in camera_order:
                content.extend([
                    {"type": "text", "text": f"CURRENT ACTUAL · {camera}"},
                    {"type": "image_url", "image_url": {
                        "url": image_data_url(current_images[camera]),
                    }},
                ])
        content.extend(self._skill_observation_blocks())
        message = {
            "role": "user",
            "content": content,
            "_observation": {
                "step": current["step"],
                "state_ref": dict(context["policy_state"]),
                "observation_id": self._current_observation_id(),
                "keyframe": True,
                "receipt": self.pending_receipt,
            },
            "_full_current_visual": "current_visual_observation" in current,
            "_causal_observation": {
                "schema": "ora.causal-observation.v1",
                "step": current["step"],
                "paired": before is not None,
                "cameras": camera_order,
                **({"post_action_observation": current["post_action_observation"]}
                   if "post_action_observation" in current else {}),
            },
        }
        self.last_actual_snapshot = current
        return message

    def _bind_system(self, task_instruction: str) -> None:
        task = task_instruction.strip()
        if not task:
            raise PolicySessionError("native task instruction must be non-empty")
        if self.bound_task_instruction is not None:
            if task != self.bound_task_instruction:
                raise PolicySessionError("native task instruction changed inside one session")
            return
        if self.context_composition in NATIVE_CAUSAL_SLOTS:
            assembled = assemble_native_v3_static_context(
                harness_system=str(self.harness_system_template),
                project_instructions=self.project_template,
                episode_task_template=self.episode_prefix_template,
                task_instruction=task,
                task_id=self.task_id or self.run_id,
            )
            self.system_text = str(assembled["system_text"])
            self.task_sha256 = str(assembled["task_sha256"])
            self.messages.extend(copy.deepcopy(assembled["messages"]))
            self.context_sources.extend(copy.deepcopy(assembled["sources"]))
            self.bound_task_instruction = task
            return
        prefix = self.episode_prefix_template.replace("{{TASK}}", task)
        self.system_text = self.project_template + "\n\n" + prefix
        self.messages.append({"role": "system", "content": self.system_text})
        self.bound_task_instruction = task

    def _turn_instruction(self) -> str:
        return native_turn_instruction(self.tools.surface)

    def _observation_message(
        self, *, task_instruction: str, context: Mapping[str, Any], image_path: Path,
        terminal_lifecycle: bool = False,
    ) -> dict[str, Any]:
        if self.context_composition in NATIVE_CAUSAL_SLOTS:
            return self._causal_observation_message(
                context=context, terminal_lifecycle=terminal_lifecycle,
            )
        step = self._observation_step(context)
        state = context["policy_state"]
        previous_execution = context.get("previous_execution")
        previous_receipt = (
            previous_execution.get("receipt")
            if isinstance(previous_execution, Mapping) else None
        )
        previous_action = (
            previous_receipt.get("action")
            if isinstance(previous_receipt, Mapping) else None
        )
        keyframe = episode_index_keyframe(
            step=step, previous_action=previous_action,
        )
        text = {
            "task_instruction": task_instruction,
            "current_observation": context,
            "instruction": self._turn_instruction(),
        }
        content: list[dict[str, Any]] = [{
            "type": "text",
            "text": json.dumps(text, ensure_ascii=False, sort_keys=True),
        }]
        try:
            post_action_images = resolve_post_action_images(context, root=self.run_root, path_key="run_path")
        except PostActionObservationError as exc:
            raise PolicySessionError(str(exc)) from exc
        if post_action_images:
            if not self.supports_images:
                raise PolicySessionError("current post-action views require image support")
            for item in post_action_images:
                content.extend([
                    {"type": "text", "text": f"ACTION RESULT ACTUAL · {item['camera']}"},
                    {"type": "image_url", "image_url": {"url": image_data_url(item["path"])}},
                ])
        elif self.supports_images and self.initial_observation == "automatic":
            if not image_path.is_file():
                raise PolicySessionError(f"native observation image is unavailable: {image_path}")
            content.append({"type": "image_url", "image_url": {"url": image_data_url(image_path)}})
        message = {
            "role": "user",
            "content": content,
            "_observation": {
                "step": step,
                "state_ref": dict(state),
                "observation_id": self._current_observation_id(),
                "keyframe": keyframe,
                "receipt": self.pending_receipt,
            },
        }
        return message

    def _current_observation_id(self) -> str | None:
        path = self.tools.control_root / "current-observation.json"
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value.get("observation_id") if isinstance(value, Mapping) else None

    def _record_tool_result(
        self,
        *,
        call_id: str,
        model_call_id: str,
        step: int,
        proposal_event: Mapping[str, Any],
        name: str,
        result: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        # Every result passes the gateway once, before it is recorded or shown.
        result = self.gateway.record_proposed_result(name, proposal_event, result)
        canonical_result = copy.deepcopy(dict(result))
        image_b64 = canonical_result.pop("image_base64", None)
        images_b64 = canonical_result.pop("images_base64", None)
        if image_b64 is not None and images_b64 is not None:
            raise PolicySessionError("tool result cannot mix image_base64 and images_base64")
        raw_images = [image_b64] if image_b64 is not None else images_b64 or []
        if not isinstance(raw_images, list) or not all(
            isinstance(item, str) for item in raw_images
        ):
            raise PolicySessionError("tool returned invalid image payloads")
        if self.context_composition in NATIVE_CAUSAL_SLOTS:
            visible = self._causal_tool_result(name, canonical_result)
        else:
            visible = copy.deepcopy(canonical_result)
            visible.pop("_runtime", None)
        tool_message = {
            "role": "tool",
            "tool_call_id": call_id,
            "name": name,
            "content": json.dumps(visible, ensure_ascii=False, sort_keys=True),
        }
        self.messages.append(tool_message)
        ref = self.artifacts.put_json(
            canonical_result if self.context_composition in NATIVE_CAUSAL_SLOTS else visible,
            kind="native_tool_result",
        )
        result_event = self._append(
            "tool_result",
            causal_parent_id=str(proposal_event["event_id"]),
            observation_step=step,
            model_call_id=model_call_id,
            tool_call_id=call_id,
            payload={
                "name": name,
                "ok": (
                    bool(visible["ok"])
                    if isinstance(visible.get("ok"), bool)
                    else (
                        "error" not in visible
                        and visible.get("status") not in {
                            "error", "failed", "invalid", "rejected",
                        }
                    )
                ),
            },
            artifact_ref=ref,
        )
        tool_message["_source_event_ids"] = [str(result_event["event_id"])]
        tool_message["_artifact_ref"] = ref
        if raw_images:
            try:
                import base64
            except Exception as exc:
                raise PolicySessionError("tool returned invalid image_base64") from exc
            image_events = []
            recoverable_image_refs: list[dict[str, Any]] = []
            image_media_types: list[str] = []
            for index, encoded in enumerate(raw_images):
                try:
                    image = base64.b64decode(encoded, validate=True)
                except Exception as exc:
                    raise PolicySessionError("tool returned invalid image_base64") from exc
                historical_media_type = canonical_result.get("media_type")
                media_type = (
                    str(historical_media_type)
                    if canonical_result.get("status")
                    == "actual_historical_artifact"
                    and isinstance(historical_media_type, str)
                    and historical_media_type.startswith("image/")
                    else "image/png"
                )
                suffix = ".png" if media_type == "image/png" else ".img"
                image_ref = self.artifacts.put_bytes(
                    image, kind="native_tool_image", suffix=suffix,
                    media_type=media_type,
                )
                image_media_types.append(media_type)
                if self.context_artifacts is not None:
                    logical_ref = (
                        f"artifact://{self.policy_session_id}/"
                        f"{image_ref['sha256']}"
                    )
                    try:
                        entry, stored = self.context_artifacts.resolve(logical_ref)
                    except FileNotFoundError:
                        reference = self.context_artifacts.register_storage_ref(
                            image_ref,
                            kind="context_image",
                            round_index=None,
                            action_step=step,
                        )
                    else:
                        if (
                            stored != image
                            or entry.get("media_type") != media_type
                        ):
                            raise PolicySessionError(
                                "historical tool image differs from its existing "
                                "Artifact identity"
                            )
                        reference = self.context_artifacts.public_ref(
                            entry, round_index=None, action_step=step,
                        )
                    recoverable_image_refs.append(reference)
                image_events.append(self._append(
                    "tool_image", visibility="exact", observation_step=step,
                    model_call_id=model_call_id, tool_call_id=call_id,
                    artifact_ref=image_ref, payload={"image_index": index},
                ))
            if self.supports_images:
                image_label = (
                    f"HISTORICAL ACTUAL observation step {result['observation_step']}"
                    if name == "observe" and result.get("status") == "actual_historical_observation"
                    else "CANDIDATE preview"
                    if self.context_composition in NATIVE_CAUSAL_SLOTS and name == "preview_move"
                    else "CURRENT ACTUAL tool observation"
                    if self.context_composition in NATIVE_CAUSAL_SLOTS and name == "observe"
                    else "READ-ONLY tool feedback"
                )
                returned_by = f"returned by {name} ({call_id})"
                return {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": (
                            # tool_image_call_labels_v1: name the call, so
                            # the images of parallel calls are not ambiguous.
                            f"{image_label}: {len(raw_images)} image(s) {returned_by}."
                            if TOOL_IMAGE_CALL_LABELS in COMPOSITION_CONTEXT_CAPABILITIES.get(
                                self.context_composition, ())
                            else f"{image_label}: {len(raw_images)} image(s)."
                            if self.context_composition in NATIVE_CAUSAL_SLOTS else
                            f"{len(raw_images)} image(s) {returned_by}."
                        )},
                        *[
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{encoded}",
                                },
                            }
                            for encoded, media_type in zip(
                                raw_images, image_media_types, strict=True,
                            )
                        ],
                    ],
                    "_tool_image_step": step,
                    "_tool_image_name": name,
                    "_tool_image_call_id": call_id,
                    "_source_event_ids": [str(event["event_id"]) for event in image_events],
                    **({
                        "_context_image_refs": recoverable_image_refs,
                    } if recoverable_image_refs else {}),
                }
        return None

    def _causal_tool_result(
        self, name: str, result: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Strict model projection; canonical result remains in the artifact store."""

        error = result.get("error")
        if isinstance(error, Mapping):
            # Public recovery fields apply to every active consumer. The shared
            # filter still removes private state and host provenance recursively.
            visible_error = {
                key: copy.deepcopy(error[key])
                for key in (
                    "code", "message", "retryable", "retry_requires_changed_arguments",
                    "infrastructure", "field", "received", "constraints", "repetition",
                )
                if error.get(key) is not None
            }
            return {"error": causal_provider_value(visible_error)}
        # result_schemas is keyed by what this surface advertises. A miss here
        # does not raise -- it falls through to the unprojected result -- so the
        # lookup resolves the spelling rather than depending on the caller's.
        from ora.agent.tools import tool_names

        schemas = self.tools.surface.result_schemas
        result_schema = schemas.get(tool_names.as_advertised(schemas, name))
        if isinstance(result_schema, Mapping):
            properties = _declared_result_properties(result_schema)
            if not properties:
                raise PolicySessionError(
                    f"public result schema for {name} declares no result properties"
                )
            # ToolRuntime already validates the complete public result against
            # this schema. Native causal projection mechanically follows the
            # union of its declared branches, then applies the shared causal
            # provenance filter. This supports new surfaces without adding a
            # second tool-name switch while keeping audit IDs out of prompts.
            visible = causal_provider_value(
                {key: result[key] for key in properties if key in result},
                declared=frozenset(properties),
            )
            return visible
        if name in {
            "move_pose", "move_free_space", "servo_delta", "set_gripper",
            "close_gripper", "open_gripper", "recover_motion",
        }:
            return {
                key: copy.deepcopy(result[key])
                for key in ("status", "action") if key in result
            }
        if name == "check_task":
            if result.get("success") is False:
                return {
                    "status": "unavailable",
                    "reason": result.get("reason"),
                    "retryable": bool(result.get("retryable", False)),
                }
            return {"task_success": bool(result.get("task_success", False))}
        if name == "report_issue":
            return {"status": "recorded"}
        if name == "finish_episode":
            return {"outcome": result.get("outcome")}
        return {"status": str(result.get("status", "returned"))}

    def _execute_aggregated_read_only_batch(
        self, *, parsed: Sequence[tuple[
            dict[str, Any], str | None, dict[str, Any] | None,
            str | None, Exception | None,
        ]], assistant: dict[str, Any], response_event: Mapping[str, Any],
        model_call_id: str, step: int,
    ) -> bool:
        """Let a tool runtime aggregate same-tool read-only visual feedback."""

        names = {name for _call, name, _args, _effect, _error in parsed}
        if len(parsed) < 2 or len(names) != 1 or None in names or any(
            args is None or effect != "read_only"
            or parse_error is not None
            for _call, _name, args, effect, parse_error in parsed
        ):
            return False
        name = next(iter(names))
        assert isinstance(name, str)
        arguments = [dict(args) for _call, _name, args, _effect, _error in parsed if args]
        if not self.gateway.begin_read_only_feedback_batch(name, arguments):
            return False

        records: list[tuple[str, Mapping[str, Any], str, dict[str, Any]]] = []
        for call, name, args, effect, _parse_error in parsed:
            call_id = str(call["id"])
            proposal_event = self._append(
                "tool_call_proposed",
                causal_parent_id=str(response_event["event_id"]),
                observation_step=step,
                model_call_id=model_call_id,
                tool_call_id=call_id,
                payload={
                    "name": name, "arguments": args, "effect": effect,
                    "batch_valid": True, "visual_feedback": "combined",
                },
            )
            assistant.setdefault("_source_event_ids", []).append(
                str(proposal_event["event_id"])
            )
            try:
                result = dict(self.gateway.execute(name, args or {}))
            except Exception as exc:
                result = _tool_error(exc)
            records.append((call_id, proposal_event, name, result))

        feedback = dict(self.gateway.finish_read_only_feedback_batch(
            name, [record[3] for record in records]
        ))
        if set(feedback) - {"image_base64", "images_base64"}:
            raise PolicySessionError("read-only batch feedback contains visible fields")
        if feedback:
            target = next(
                (
                    record for record in reversed(records)
                    if record[3].get("success") is True
                ),
                None,
            )
            if target is None:
                raise PolicySessionError(
                    "read-only batch produced feedback without a successful result"
                )
            target[3].update(feedback)

        tool_images: list[dict[str, Any]] = []
        for call_id, proposal_event, name, result in records:
            image_message = self._record_tool_result(
                call_id=call_id,
                model_call_id=model_call_id,
                step=step,
                proposal_event=proposal_event,
                name=name,
                result=result,
            )
            if image_message is not None:
                tool_images.append(image_message)
        if len(tool_images) > 1:
            raise PolicySessionError("read-only batch emitted duplicate feedback images")
        self.messages.extend(tool_images)
        return True

    def run_observation(
        self,
        *,
        task_instruction: str,
        context: Mapping[str, Any],
        image_path: Path,
        allow_reminder: bool = True,
        terminal_lifecycle: bool = False,
        image_paths: tuple[Path, ...] = (),
    ) -> dict[str, Any]:
        if image_paths:
            try:
                verified = resolve_post_action_images(context, root=self.run_root, path_key="run_path")
            except PostActionObservationError as exc:
                raise PolicySessionError(str(exc)) from exc
            if tuple(item["path"].resolve() for item in verified) != tuple(Path(path).resolve() for path in image_paths):
                raise PolicySessionError("policy image vector differs from action-bound fresh views")
        return self.turn_loop.run(
            self,
            task_instruction=task_instruction,
            context=context,
            image_path=image_path,
            allow_reminder=allow_reminder,
            terminal_lifecycle=terminal_lifecycle,
        )

    def preserve_observation_without_policy_turn(
        self,
        *,
        task_instruction: str,
        context: Mapping[str, Any],
        image_path: Path,
        terminal_lifecycle: bool = True,
        allow_reminder: bool = False,
        image_paths: tuple[Path, ...] = (),
    ) -> None:
        """Append one actual observation without making a Provider call.

        A terminal Environment boundary can forbid a further policy turn while
        a persistent AgentSession still needs the post-action actual evidence
        before the Supervisor resets the next round.  Materializing that
        observation here closes the causal ActionStep and consumes its receipt
        without granting the policy another chance to mutate the world.
        """

        del allow_reminder, image_paths
        if self.closed or self.interrupted:
            raise PolicySessionError("native policy session is not active")
        self._bind_system(task_instruction)
        step = self._observation_step(context)
        observation = self._observation_message(
            task_instruction=task_instruction,
            context=context,
            image_path=image_path,
            terminal_lifecycle=terminal_lifecycle,
        )
        self.messages.append(observation)
        self.pending_receipt = None
        self.pending_receipt_body = None
        observation_event = self._append(
            "observation_received",
            observation_step=step,
            payload={
                "observation_id": observation["_observation"]["observation_id"],
                "state_ref": observation["_observation"]["state_ref"],
                "provider_turn": False,
                **({
                    "causal_pair": bool(
                        observation.get("_causal_observation", {}).get("paired")
                    ),
                    "causal_cameras": list(
                        observation.get("_causal_observation", {}).get("cameras", [])
                    ),
                } if self.context_composition in NATIVE_CAUSAL_SLOTS else {}),
            },
        )
        source_event_ids = [str(observation_event["event_id"])]
        receipt_metadata = observation["_observation"].get("receipt")
        if (
            isinstance(receipt_metadata, Mapping)
            and isinstance(receipt_metadata.get("event_id"), str)
        ):
            source_event_ids.append(str(receipt_metadata["event_id"]))
        if self.last_fresh_observation_event_id is not None:
            source_event_ids.append(self.last_fresh_observation_event_id)
        observation["_source_event_ids"] = source_event_ids
        self._append_round_static_facts(observation_event)
        self._append(
            "observation_preserved_without_policy_turn",
            causal_parent_id=str(observation_event["event_id"]),
            observation_step=step,
            visibility="derived",
            payload={
                "reason": "terminal_round_boundary",
                "terminal_lifecycle": terminal_lifecycle,
            },
        )
        self._write_session_manifest(exit_reason=None)

    def act(self, **kwargs: Any) -> Mapping[str, Any]:
        return self.run_observation(**kwargs)

    def record_environment_receipt(self, receipt: Mapping[str, Any]) -> None:
        ref = self.artifacts.put_json(receipt, kind="environment_receipt")
        top_level_action_id = receipt.get("action_id")
        nested_receipt = receipt.get("receipt")
        nested_receipt_action_id = (
            nested_receipt.get("action_id")
            if isinstance(nested_receipt, Mapping) else None
        )
        nested_action = (
            nested_receipt.get("action")
            if isinstance(nested_receipt, Mapping) else None
        )
        nested_action_id = (
            nested_action.get("action_id")
            if isinstance(nested_action, Mapping) else None
        )
        action_ids = {
            value for value in (
                top_level_action_id,
                nested_receipt_action_id,
                nested_action_id,
            )
            if isinstance(value, str) and value
        }
        if len(action_ids) > 1:
            raise PolicySessionError(
                "environment receipt contains conflicting action identities"
            )
        action_id = next(iter(action_ids), None)
        if (
            isinstance(self.last_action_id, str)
            and self.last_action_id
            and action_id != self.last_action_id
        ):
            raise PolicySessionError(
                "environment receipt does not match the submitted action"
            )
        event = self._append(
            "environment_receipt",
            causal_parent_id=self.last_action_event_id,
            visibility="exact",
            observation_step=self.last_action_step,
            payload={
                "receipt_id": receipt.get("receipt_id") or receipt.get("id")
                or f"receipt-sha256-{ref['sha256'][:16]}",
                "action_id": action_id,
            },
            artifact_ref=ref,
        )
        self.pending_receipt = {
            "receipt_ref": ref,
            "event_id": event["event_id"],
            "action_id": event["payload"]["action_id"],
        }
        self.pending_receipt_body = copy.deepcopy(dict(receipt))

    def record_fresh_observation(self, context: Mapping[str, Any]) -> None:
        state = context.get("policy_state")
        step = state.get("step") if isinstance(state, Mapping) else None
        event = self._append(
            "fresh_observation_published",
            observation_step=step if isinstance(step, int) else None,
            payload={
                "observation_id": self._current_observation_id(),
                "state_ref": state,
                "after_receipt": self.pending_receipt,
            },
        )
        self.last_fresh_observation_event_id = str(event["event_id"])

    def record_official_verdict(self, verdict: Mapping[str, Any] | None) -> None:
        ref = None if verdict is None else self.artifacts.put_json(
            verdict, kind="official_verdict"
        )
        event = self._append(
            "official_verdict",
            visibility="exact" if verdict is not None else "opaque",
            payload=None if verdict is None else {
                "official_success": verdict.get("official_success"),
            },
            artifact_ref=ref,
        )
        self.durability.barrier()  # round end: the Episode's evidence is durable
        if (
            self.context_retention == LOSSLESS_CONTEXT_RETENTION
            and isinstance(verdict, Mapping)
            and isinstance(verdict.get("official_success"), bool)
        ):
            round_number = next((
                message.get("_session_round")
                for message in reversed(self.messages)
                if isinstance(message.get("_session_round"), int)
            ), None)
            self.messages.append({
                "role": "user",
                "content": "<official_round_result>\n" + json.dumps({
                    "round": round_number,
                    "passed": verdict["official_success"],
                    "instruction": (
                        "This completed-round result is exact. Reflect on durable "
                        "task-local lessons."
                    ),
                }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n</official_round_result>",
                "_protected_context": True,
                "_context_source": f"official_round_result_{round_number}",
                "_source_event_ids": [str(event["event_id"])],
            })

    def interrupt(self, reason: str = "operator_interrupt") -> None:
        self.interrupted = True
        self._append("session_interrupted", visibility="derived", payload={"reason": reason})
        if (
            self.rollout_compactor is not None
            and self.rollout_compactor.provider is not self.provider
        ):
            self.rollout_compactor.provider.close()
        self.provider.close()

    def close(self) -> None:
        if self.closed:
            return
        try:
            self._append("session_closed", visibility="derived", payload={"reason": "closed"})
            self._write_session_manifest(exit_reason="closed")
            self.durability.barrier()  # durable before evaluation or sealing
        finally:
            self.closed = True
            self.durability.close(raise_errors=False)
            if (
                self.rollout_compactor is not None
                and self.rollout_compactor.provider is not self.provider
            ):
                self.rollout_compactor.provider.close()
            self.provider.close()
