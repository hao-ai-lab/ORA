"""Canonical policy trace, content-addressed artifacts and SFT export."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from .operator_messages import operator_trace_errors
from ora.agent.evidence import request_archive


AGENT_EVENT_SCHEMA = "ora.agent-event.v1"
SFT_EXPORT_SCHEMA = "ora.sft-export.v1"
VISIBILITY = {"exact", "derived", "opaque"}
FORBIDDEN_FRAGMENTS = {
    "actor_state",
    "object_gt",
    "segmentation",
    "contact",
    "success_predicate",
    "expert_program",
    "task_source",
}
# Forbidden terms match on word boundaries (start/end of the key or an
# adjacent non-alphanumeric character), with plural forms included.  A term
# embedded inside a larger word does not match: the controller phase noun
# "precontact" is public motion vocabulary, not contact-sensing evidence,
# and the bare substring rule invalidated nine sealed v13 runs over the
# policy-safe field `motion_strategy.precontact_public_tcp_poses`.
_FORBIDDEN_PATTERNS = tuple(
    re.compile(rf"(?<![a-z0-9]){re.escape(fragment)}(?:s|es)?(?![a-z0-9])")
    for fragment in sorted(FORBIDDEN_FRAGMENTS)
)


def _matches_forbidden_fragment(lowered: str) -> bool:
    return any(pattern.search(lowered) for pattern in _FORBIDDEN_PATTERNS)


# Exact policy-authored motion-parameter names that legitimately contain a
# forbidden term on a word boundary.  `contact_approach` is the public
# staged-approach strategy argument the policy itself writes
# (`staged_contact_approach_v1`); it cannot leak state the policy never saw.
# Values under these keys are still scanned recursively.
PUBLIC_MOTION_VOCABULARY_KEYS = frozenset({"contact_approach"})


# Exact Panda grip-site compiler constants; the ALOHA dual-arm surfaces
# publish a restricted compiler projection instead, so their geometry never
# reaches provider-visible results.  The exemption stays value-pinned and
# fail-closed: an unregistered geometry value remains forbidden evidence.
PUBLIC_CONTACT_GEOMETRY_VALUES = {
    "contact_graspnet_gripper_depth_m": (frozenset({0.1034}), ".compiler"),
    "controlled_point_to_contact_baseline_m": (frozenset({0.0064}), ".compiler"),
    "contact_baseline_reconstruction": (frozenset({"passed"}), ".compiler_checks"),
}
PUBLIC_GRASP_CONTACT_POSE_CAPABILITY = "public_grasp_contact_pose_v1"


class TrajectoryError(RuntimeError):
    """Canonical agent evidence is missing, corrupt or unsafe to export."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def json_semantic_equal(left: Any, right: Any) -> bool:
    """Compare decoded JSON values without treating 2 and 2.0 as substitutions.

    Raw artifacts and their hashes remain byte-exact.  This helper is only for
    joins between independently decoded/serialized JSON evidence planes, where
    JSON has one number type even though Python preserves integer versus float
    syntax.  Booleans remain distinct from numbers and non-finite values never
    compare equal because they are not valid JSON.
    """

    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if isinstance(left, float) and not math.isfinite(left):
            return False
        if isinstance(right, float) and not math.isfinite(right):
            return False
        return left == right
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            json_semantic_equal(left[key], right[key]) for key in left
        )
    if (
        isinstance(left, Sequence) and not isinstance(left, (str, bytes, bytearray))
        and isinstance(right, Sequence)
        and not isinstance(right, (str, bytes, bytearray))
    ):
        return len(left) == len(right) and all(
            json_semantic_equal(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    return type(left) is type(right) and left == right


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(
                json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass
class ArtifactStore:
    """Store generated native-agent payloads by content hash."""

    run_root: Path
    # ``None`` keeps the synchronous fsync-before-link path. A
    # :class:`~ora.agent.evidence.durability.DeferredDurability`
    # keeps the same write/link order but moves the fsync behind its barrier.
    durability: Any = None

    def __post_init__(self) -> None:
        self.run_root = self.run_root.resolve()
        self.root = self.run_root / "native/agent-artifacts"
        self.root.mkdir(parents=True, exist_ok=True)

    def put_bytes(
        self,
        data: bytes,
        *,
        kind: str,
        suffix: str = ".bin",
        media_type: str = "application/octet-stream",
    ) -> dict[str, Any]:
        digest = sha256_bytes(data)
        safe_suffix = suffix if suffix.startswith(".") else f".{suffix}"
        target = self.root / f"{digest}{safe_suffix}"
        if target.exists():
            if target.read_bytes() != data:
                raise TrajectoryError(f"artifact hash collision: {digest}")
        else:
            descriptor, name = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".tmp", dir=self.root,
            )
            temporary = Path(name)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(data)
                    stream.flush()
                    if self.durability is None:
                        os.fsync(stream.fileno())
                linked = False
                try:
                    os.link(temporary, target)
                    linked = True
                except FileExistsError:
                    if target.read_bytes() != data:
                        raise TrajectoryError(f"artifact hash collision: {digest}")
            finally:
                temporary.unlink(missing_ok=True)
            if linked and self.durability is not None:
                self.durability.submit(target)
        return {
            "kind": kind,
            "path": target.relative_to(self.run_root).as_posix(),
            "sha256": digest,
            "bytes": len(data),
            "media_type": media_type,
        }

    def put_json(self, value: Any, *, kind: str) -> dict[str, Any]:
        return self.put_bytes(
            canonical_bytes(value), kind=kind, suffix=".json", media_type="application/json"
        )

    def reference(self, path: Path, *, kind: str, media_type: str) -> dict[str, Any]:
        resolved = path.resolve()
        if not resolved.is_file() or not resolved.is_relative_to(self.run_root):
            raise TrajectoryError(f"artifact reference escapes run: {path}")
        return {
            "kind": kind,
            "path": resolved.relative_to(self.run_root).as_posix(),
            "sha256": file_sha256(resolved),
            "bytes": resolved.stat().st_size,
            "media_type": media_type,
        }


@dataclass
class TrajectoryRecorder:
    """Append-only, fsync-backed event recorder with a forward hash chain."""

    run_root: Path
    run_id: str
    policy_session_id: str
    # ``None`` fsyncs every line before returning. A DeferredDurability keeps
    # the write+flush (and so the hash chain and line order) on this thread and
    # coalesces the fsync behind its barrier.
    durability: Any = None

    def __post_init__(self) -> None:
        self.run_root = self.run_root.resolve()
        self.path = self.run_root / "agent-events.jsonl"
        if self.path.exists():
            raise TrajectoryError("refusing to append to an existing agent-events.jsonl")
        self.seq = 0
        self.previous_event_sha256: str | None = None

    def append(
        self,
        event_type: str,
        *,
        visibility: str = "exact",
        causal_parent_id: str | None = None,
        observation_step: int | None = None,
        model_call_id: str | None = None,
        tool_call_id: str | None = None,
        payload: Any = None,
        artifact_ref: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if visibility not in VISIBILITY:
            raise TrajectoryError(f"invalid event visibility: {visibility}")
        body = {
            "schema": AGENT_EVENT_SCHEMA,
            "seq": self.seq,
            "event_id": f"agent-event-{self.seq:06d}",
            "type": event_type,
            "run_id": self.run_id,
            "policy_session_id": self.policy_session_id,
            "observation_step": observation_step,
            "model_call_id": model_call_id,
            "tool_call_id": tool_call_id,
            "causal_parent_id": causal_parent_id,
            "visibility": visibility,
            "unix_s": time.time(),
            "monotonic_ns": time.monotonic_ns(),
            "previous_event_sha256": self.previous_event_sha256,
            "payload": payload,
            "artifact_ref": None if artifact_ref is None else dict(artifact_ref),
        }
        event = {**body, "event_sha256": sha256_bytes(canonical_bytes(body))}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            if self.durability is None:
                os.fsync(stream.fileno())
        if self.durability is not None:
            self.durability.submit(self.path)
        self.previous_event_sha256 = event["event_sha256"]
        self.seq += 1
        return event


def read_agent_events(run_root: Path) -> list[dict[str, Any]]:
    path = run_root.resolve() / "agent-events.jsonl"
    if not path.is_file():
        raise TrajectoryError("agent-events.jsonl is missing")
    events: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TrajectoryError(f"invalid agent event JSON at line {index + 1}") from exc
        if not isinstance(value, dict):
            raise TrajectoryError(f"agent event must be an object at line {index + 1}")
        events.append(value)
    return events


def _is_public_grasp_contact_pose(value: Any) -> bool:
    """Recognize only the exact policy-visible grasp-plan pose contract."""

    if not isinstance(value, Mapping) or set(value) != {
        "position_m", "orientation_xyzw",
    }:
        return False
    position = value.get("position_m")
    orientation = value.get("orientation_xyzw")
    if not isinstance(position, list) or len(position) != 3:
        return False
    if not isinstance(orientation, list) or len(orientation) != 4:
        return False
    return all(
        isinstance(component, (int, float))
        and not isinstance(component, bool)
        and math.isfinite(component)
        for component in [*position, *orientation]
    )


def _allows_public_grasp_contact_pose(plan: Mapping[str, Any]) -> bool:
    surface = plan.get("policy_composition", {}).get("tool_surface", {})
    capabilities = surface.get("capabilities", []) if isinstance(surface, Mapping) else []
    return (
        isinstance(capabilities, list)
        and PUBLIC_GRASP_CONTACT_POSE_CAPABILITY in capabilities
    )


def _forbidden_path(
    value: Any,
    *,
    path: str = "$",
    allow_public_grasp_contact_pose: bool = False,
    safe_field_names: frozenset[str] = frozenset(),
) -> str | None:
    from .motion_feedback import is_public_contact_feedback_field

    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            aggregate_contact = (
                lowered in {"contact", "contact_estimated"}
                and (path.endswith(".feedback") or path.endswith(".motion_feedback"))
                and (isinstance(child, bool) or child is None)
            )
            geometry_rule = PUBLIC_CONTACT_GEOMETRY_VALUES.get(lowered)
            public_contact_geometry = False
            if allow_public_grasp_contact_pose and geometry_rule is not None:
                expected_values, required_parent = geometry_rule
                public_contact_geometry = (
                    path.endswith(required_parent)
                    and isinstance(child, (int, float, str))
                    and not isinstance(child, bool)
                    and child in expected_values
                )
            public_grasp_contact_pose = (
                allow_public_grasp_contact_pose
                and lowered == "contact_pose"
                and _is_public_grasp_contact_pose(child)
            )
            # The staged controller phase labels live under `phases`; the
            # `local_contact` label names the local contact-approach stage
            # and its value is public tracking evidence, not contact sensing.
            staged_phase_label = (
                lowered == "local_contact" and path.endswith(".phases")
            )
            if (
                lowered not in safe_field_names
                and lowered not in PUBLIC_MOTION_VOCABULARY_KEYS
                and not staged_phase_label
                and not is_public_contact_feedback_field(path, lowered, child)
                and not aggregate_contact
                and not public_contact_geometry
                and not public_grasp_contact_pose
                and _matches_forbidden_fragment(lowered)
            ):
                # The explicit absence declaration is safe evidence, not a leak.
                if lowered != "forbidden_modalities_absent":
                    return f"{path}.{key}"
            found = _forbidden_path(
                child,
                path=f"{path}.{key}",
                allow_public_grasp_contact_pose=allow_public_grasp_contact_pose,
                safe_field_names=safe_field_names,
            )
            if found:
                return found
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found = _forbidden_path(
                child,
                path=f"{path}[{index}]",
                allow_public_grasp_contact_pose=allow_public_grasp_contact_pose,
                safe_field_names=safe_field_names,
            )
            if found:
                return found
    elif isinstance(value, str) and value.lstrip().startswith(("{", "[")):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            pass
        else:
            return _forbidden_path(
                decoded,
                path=f"{path}<json>",
                allow_public_grasp_contact_pose=allow_public_grasp_contact_pose,
                safe_field_names=safe_field_names,
            )
    return None


def _artifact_refs(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        if (
            isinstance(value.get("path"), str)
            and isinstance(value.get("sha256"), str)
            and isinstance(value.get("bytes"), int)
        ):
            yield value
        for child in value.values():
            yield from _artifact_refs(child)
    elif isinstance(value, list):
        for child in value:
            yield from _artifact_refs(child)


def verify_agent_trace(run_root: Path) -> dict[str, Any]:
    """Validate canonical identity, causality, pairing and artifact integrity."""

    root = run_root.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    try:
        events = read_agent_events(root)
    except TrajectoryError as exc:
        return {
            "status": "invalid", "errors": [str(exc)], "warnings": [],
            "event_count": 0,
        }
    previous_hash: str | None = None
    known_ids: set[str] = set()
    pending_tools: dict[str, dict[str, Any]] = {}
    seen_tool_ids: set[str] = set()
    mutation_by_model_call: dict[str, int] = {}
    requests: set[str] = set()
    responses: set[str] = set()
    request_steps: dict[str, Any] = {}
    request_refs: dict[str, Any] = {}
    request_ids: dict[str, Any] = {}
    failed_attempt_counts: dict[str, int] = {}
    terminal_attempt_failures: set[str] = set()
    accepted_mutations: dict[str, set[str]] = {}
    action_steps: set[Any] = set()
    awaiting_fresh: dict[str, Any] | None = None
    mutation_tools = {"robot.submit_action"}
    allow_public_grasp_contact_pose = False
    safe_field_names: frozenset[str] = frozenset()
    plan: dict[str, Any] = {}
    plan_path = root / "plan.json"
    if plan_path.is_file():
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            surface = plan.get("policy_composition", {}).get("tool_surface", {})
            allow_public_grasp_contact_pose = _allows_public_grasp_contact_pose(plan)
            configured = surface.get("mutation_tools")
            if isinstance(configured, list) and all(isinstance(name, str) for name in configured):
                mutation_tools = set(configured)
            capabilities = surface.get("capabilities")
            view_config = surface.get("view_config")
            automatic_views = (
                view_config.get("automatic_actual_views")
                if isinstance(view_config, Mapping) else None
            )
            if (
                isinstance(capabilities, list)
                and "automatic_current_auxiliary_contact_sheet_v1" in capabilities
                and isinstance(automatic_views, list)
                and "auxiliary_contact_sheet" in automatic_views
            ):
                # This exact field names a hash-bound RGB composition.  It is
                # not simulator contact evidence.  Descendants are still
                # scanned normally, so the exception cannot hide privileged
                # payloads beneath the derived-view label.
                safe_field_names = frozenset({"auxiliary_contact_sheet"})
        except (OSError, json.JSONDecodeError):
            errors.append("plan:invalid")

    errors.extend(operator_trace_errors(events, plan if plan_path.is_file() else {}))
    for index, event in enumerate(events):
        if event.get("schema") != AGENT_EVENT_SCHEMA:
            errors.append(f"event:{index}:schema")
        if event.get("seq") != index:
            errors.append(f"event:{index}:sequence")
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or event_id in known_ids:
            errors.append(f"event:{index}:event_id")
        else:
            known_ids.add(event_id)
        if event.get("previous_event_sha256") != previous_hash:
            errors.append(f"event:{index}:previous_hash")
        body = {key: value for key, value in event.items() if key != "event_sha256"}
        expected = sha256_bytes(canonical_bytes(body))
        if event.get("event_sha256") != expected:
            errors.append(f"event:{index}:hash")
        previous_hash = event.get("event_sha256") if isinstance(event.get("event_sha256"), str) else None
        parent = event.get("causal_parent_id")
        if parent is not None and parent not in known_ids:
            errors.append(f"event:{index}:causal_parent")
        artifacts = list(_artifact_refs(event.get("artifact_ref")))
        artifacts.extend(_artifact_refs(event.get("payload")))
        for artifact in artifacts:
            relative = artifact.get("path")
            if not isinstance(relative, str):
                errors.append(f"event:{index}:artifact_path")
            else:
                path = (root / relative).resolve()
                if not path.is_file() or not path.is_relative_to(root):
                    errors.append(f"event:{index}:artifact_missing")
                elif file_sha256(path) != artifact.get("sha256"):
                    errors.append(f"event:{index}:artifact_hash")
                elif artifact.get("media_type") == "application/json":
                    try:
                        artifact_value = json.loads(path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        errors.append(f"event:{index}:artifact_json")
                    else:
                        forbidden_artifact = _forbidden_path(
                            artifact_value,
                            allow_public_grasp_contact_pose=allow_public_grasp_contact_pose,
                            safe_field_names=safe_field_names,
                        )
                        if forbidden_artifact:
                            errors.append(
                                f"event:{index}:privileged_artifact_field:{forbidden_artifact}"
                            )

        event_type = event.get("type")
        call_id = event.get("model_call_id")
        tool_id = event.get("tool_call_id")
        payload = event.get("payload")
        forbidden_payload = _forbidden_path(
            payload,
            allow_public_grasp_contact_pose=allow_public_grasp_contact_pose,
            safe_field_names=safe_field_names,
        )
        if forbidden_payload:
            errors.append(f"event:{index}:privileged_field:{forbidden_payload}")
        if event_type == "provider_request" and isinstance(call_id, str):
            if awaiting_fresh is not None:
                errors.append(f"event:{index}:provider_request_before_fresh_observation")
            requests.add(call_id)
            request_steps[call_id] = event.get("observation_step")
            request_refs[call_id] = event.get("artifact_ref")
            request_ids[call_id] = event.get("event_id")
        elif event_type == "provider_attempt_failed":
            sent_sha256 = payload.get("request_sha256") if isinstance(payload, Mapping) else None
            recorded_sha256 = (request_refs.get(call_id) or {}).get("sha256")
            if (
                not isinstance(call_id, str) or call_id not in requests
                or call_id in responses or call_id in terminal_attempt_failures
                or event.get("observation_step") != request_steps.get(call_id)
                or event.get("causal_parent_id") != request_ids.get(call_id)
            ):
                errors.append(f"event:{index}:provider_attempt_pairing")
            if not isinstance(payload, Mapping) or (
                payload.get("schema") != "ora.provider-attempt-failure.v1"
                or payload.get("model_call_id") != call_id
                or payload.get("tool_execution_started") is not False
                or type(payload.get("attempt")) is not int
                or payload.get("attempt") != failed_attempt_counts.get(call_id, 0) + 1
                or type(payload.get("retry_scheduled")) is not bool
                or sent_sha256 != recorded_sha256
                or event.get("artifact_ref") is None
            ):
                errors.append(f"event:{index}:provider_attempt_contract")
            if isinstance(call_id, str):
                failed_attempt_counts[call_id] = failed_attempt_counts.get(call_id, 0) + 1
                if isinstance(payload, Mapping) and payload.get("retry_scheduled") is False:
                    terminal_attempt_failures.add(call_id)
        elif event_type == "provider_response" and isinstance(call_id, str):
            if call_id in terminal_attempt_failures:
                errors.append(f"event:{index}:response_after_terminal_attempt_failure")
            if call_id in failed_attempt_counts and (
                not isinstance(payload, Mapping)
                or payload.get("attempts") != failed_attempt_counts[call_id] + 1
            ):
                errors.append(f"event:{index}:provider_attempt_count")
            responses.add(call_id)
            if request_steps.get(call_id) != event.get("observation_step"):
                errors.append(f"event:{index}:cross_observation_provider_response")
        elif event_type == "tool_call_proposed":
            if not isinstance(tool_id, str) or tool_id in seen_tool_ids:
                errors.append(f"event:{index}:tool_call_id")
            else:
                seen_tool_ids.add(tool_id)
                pending_tools[tool_id] = event
            if (
                isinstance(payload, Mapping)
                and payload.get("name") in mutation_tools
                and payload.get("batch_valid") is not False
                and payload.get("state_effect") != "read_only"
            ):
                if isinstance(call_id, str):
                    mutation_by_model_call[call_id] = mutation_by_model_call.get(call_id, 0) + 1
                    accepted_mutations.setdefault(call_id, set()).add(str(tool_id))
        elif event_type == "tool_result":
            if not isinstance(tool_id, str) or tool_id not in pending_tools:
                errors.append(f"event:{index}:orphan_tool_result")
            else:
                proposed = pending_tools.pop(tool_id)
                if proposed.get("model_call_id") != call_id:
                    errors.append(f"event:{index}:cross_model_call_tool_result")
                if proposed.get("observation_step") != event.get("observation_step"):
                    errors.append(f"event:{index}:cross_observation_tool_result")
        elif event_type == "action_proposed":
            if (
                not isinstance(call_id, str)
                or not isinstance(tool_id, str)
                or tool_id not in accepted_mutations.get(call_id, set())
            ):
                errors.append(f"event:{index}:action_without_unique_mutation")
            step = event.get("observation_step")
            if step in action_steps:
                errors.append(f"event:{index}:multiple_actions_for_observation")
            action_steps.add(step)
            awaiting_fresh = {"step": step, "receipt": False}
        elif event_type == "environment_receipt":
            if awaiting_fresh is None:
                errors.append(f"event:{index}:receipt_without_action")
            else:
                awaiting_fresh["receipt"] = True
        elif event_type == "fresh_observation_published" and awaiting_fresh is not None:
            if awaiting_fresh.get("receipt") is not True:
                errors.append(f"event:{index}:fresh_observation_without_receipt")
            prior_step = awaiting_fresh.get("step")
            fresh_step = event.get("observation_step")
            if isinstance(prior_step, int) and (
                not isinstance(fresh_step, int) or fresh_step <= prior_step
            ):
                errors.append(f"event:{index}:observation_not_fresh")
            awaiting_fresh = None

    for tool_id in sorted(pending_tools):
        errors.append(f"missing_tool_result:{tool_id}")
    for call_id, count in mutation_by_model_call.items():
        if count > 1:
            errors.append(f"multiple_mutations:{call_id}")
    for call_id in sorted(responses - requests):
        errors.append(f"response_without_request:{call_id}")
    for call_id in sorted(requests - responses):
        errors.append(f"request_without_response:{call_id}")
    if awaiting_fresh is not None:
        errors.append("action_without_fresh_observation")
    return {
        "status": "valid" if not errors else "invalid",
        "errors": errors,
        "warnings": warnings,
        "event_count": len(events),
        "final_event_sha256": previous_hash,
    }


def _iter_provider_exchanges(events: Sequence[Mapping[str, Any]]) -> Iterable[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    requests: dict[str, Mapping[str, Any]] = {}
    for event in events:
        call_id = event.get("model_call_id")
        if not isinstance(call_id, str):
            continue
        if event.get("type") == "provider_request":
            requests[call_id] = event
        elif event.get("type") == "provider_response" and call_id in requests:
            yield requests[call_id], event


def _canonical_provider_request_ref(event: Mapping[str, Any]) -> Any:
    payload = event.get("payload")
    canonical = payload.get("canonical_request_ref") if isinstance(payload, Mapping) else None
    return canonical if isinstance(canonical, Mapping) else event.get("artifact_ref")


def _surface_tool_sets(value: Any) -> tuple[set[str], set[str]]:
    """Locate the resolved ToolSurface snapshot's mutation/lifecycle sets."""

    if isinstance(value, Mapping):
        if (
            isinstance(value.get("profile_id"), str)
            and isinstance(value.get("mutation_tools"), list)
        ):
            return (
                {str(item) for item in value.get("mutation_tools", [])},
                {str(item) for item in value.get("lifecycle_tools", [])},
            )
        for child in value.values():
            found = _surface_tool_sets(child)
            if found != (set(), set()):
                return found
    elif isinstance(value, list):
        for child in value:
            found = _surface_tool_sets(child)
            if found != (set(), set()):
                return found
    return set(), set()


def _decision_attribution(
    assistant: Mapping[str, Any],
    mutation_tools: set[str],
    lifecycle_tools: set[str],
) -> dict[str, Any]:
    """Attribute one exported decision (EXP08A training-readiness boundary)."""

    calls = assistant.get("tool_calls")
    names: list[str] = []
    if isinstance(calls, list):
        for call in calls:
            function = call.get("function") if isinstance(call, Mapping) else None
            name = function.get("name") if isinstance(function, Mapping) else None
            if isinstance(name, str):
                names.append(name)
    executor: str | None = None
    if any(name in mutation_tools for name in names):
        kind = "robot_action"
    elif any(name in lifecycle_tools for name in names):
        kind = "lifecycle"
    else:
        kind = "read_only_or_reflection"
    return {
        "decision_kind": kind,
        "delegated_executor": executor,
        "tool_call_names": names,
    }


def export_sft(run_root: Path, destination: Path) -> Path:
    """Export validated native exchanges as OpenAI-compatible tool-use JSONL."""

    root = run_root.resolve()
    seal_path = root / "seal.json"
    if seal_path.is_file():
        seal_record = json.loads(seal_path.read_text(encoding="utf-8"))
        for record in seal_record.get("inventory", []):
            if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
                raise TrajectoryError("sealed inventory is invalid")
            source = (root / record["path"]).resolve()
            if (
                not source.is_file()
                or not source.is_relative_to(root)
                or file_sha256(source) != record.get("sha256")
            ):
                raise TrajectoryError(f"sealed source artifact drift: {record['path']}")
    verification = verify_agent_trace(root)
    if verification["status"] != "valid":
        raise TrajectoryError("invalid agent trace: " + ", ".join(verification["errors"]))
    evaluation_path = root / "evaluation.json"
    if not evaluation_path.is_file():
        raise TrajectoryError(
            "SFT export requires a completed run evaluation; evaluate the "
            "run first"
        )
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if evaluation.get("status") != "valid":
        raise TrajectoryError(
            "SFT export refuses an invalid run evaluation: "
            + ", ".join(str(item) for item in evaluation.get("reasons", []))
        )
    eligibility = evaluation.get("learning_eligibility")
    if not isinstance(eligibility, Mapping) or eligibility.get("eligible") is not True:
        reasons = (
            eligibility.get("reasons", [])
            if isinstance(eligibility, Mapping) else ["missing"]
        )
        raise TrajectoryError(
            "SFT export refuses a learning-ineligible run: "
            + ", ".join(str(item) for item in reasons)
        )
    events = read_agent_events(root)
    # Export v1 has no qualified Servo training contract. Inspect exact action
    # artifacts as well as evaluation so a stale/forged eligibility Boolean
    # cannot relabel an unqualified local attempt as a fine-control target.
    for event in events:
        if event.get("type") != "action_proposed":
            continue
        reference = event.get("artifact_ref")
        if not isinstance(reference, Mapping) or not isinstance(reference.get("path"), str):
            continue  # Trace verification owns malformed/missing artifacts.
        action_path = (root / reference["path"]).resolve()
        if not action_path.is_relative_to(root) or not action_path.is_file():
            continue
        proposed = json.loads(action_path.read_text(encoding="utf-8"))
        if isinstance(proposed, Mapping) and proposed.get("schema") == "ora-servo-action.v1":
            raise TrajectoryError("SFT export v1 does not support unqualified Servo execution evidence")
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    allow_public_grasp_contact_pose = _allows_public_grasp_contact_pose(plan)
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    mutation_tools, lifecycle_tools = _surface_tool_sets(plan)
    records: list[dict[str, Any]] = []
    attributions: list[dict[str, Any]] = []
    loss_masks: list[dict[str, Any]] = []
    for request_event, response_event in _iter_provider_exchanges(events):
        # Older native traces used the provider_request artifact for both the
        # canonical and wire request. New protocol adapters preserve each
        # separately and SFT always consumes the canonical OpenAI/ChatML IR.
        request_ref = _canonical_provider_request_ref(request_event)
        if not isinstance(request_ref, Mapping) or not isinstance(request_ref.get("path"), str):
            raise TrajectoryError("provider request lacks an exact artifact")
        request_bytes = request_archive.verified_bytes(root, request_ref)  # file or post-seal archive
        if request_bytes is None:
            raise TrajectoryError("canonical provider request artifact drift")
        request = json.loads(request_bytes.decode("utf-8"))
        response = response_event.get("payload")
        if not isinstance(request, Mapping) or not isinstance(response, Mapping):
            raise TrajectoryError("provider exchange is not exportable")
        if response.get("accepted") is False:
            # Rejected malformed Provider output remains exact audit evidence,
            # but is never silently promoted into an SFT conversation.
            continue
        assistant = response.get("message")
        if not isinstance(assistant, Mapping):
            raise TrajectoryError("provider response lacks an assistant message")
        assistant = {
            key: value for key, value in assistant.items()
            if key != "reasoning_content" and not str(key).startswith("_")
        }
        if assistant.get("role") != "assistant":
            raise TrajectoryError("SFT target must be a Language-Model assistant response")
        record = {
            "messages": [*list(request.get("messages", [])), assistant],
            "tools": list(request.get("tools", [])),
        }
        forbidden = _forbidden_path(
            record,
            allow_public_grasp_contact_pose=allow_public_grasp_contact_pose,
        )
        if forbidden:
            raise TrajectoryError(f"privileged field in SFT export: {forbidden}")
        attributions.append({
            "index": len(records),
            **_decision_attribution(assistant, mutation_tools, lifecycle_tools),
        })
        loss_masks.append({
            "index": len(records),
            "target_author": "language_model",
            "target_message_index": len(record["messages"]) - 1,
            "message_loss_weights": [0] * (len(record["messages"]) - 1) + [1],
        })
        records.append(record)

    destination = destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    jsonl = destination / "conversations.openai-tools.jsonl"
    with jsonl.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    attribution_jsonl = destination / "attribution.jsonl"
    with attribution_jsonl.open("w", encoding="utf-8") as stream:
        for item in attributions:
            stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    attribution_counts: dict[str, int] = {}
    loss_mask_jsonl = destination / "loss-mask.jsonl"
    with loss_mask_jsonl.open("w", encoding="utf-8") as stream:
        for item in loss_masks:
            stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    for item in attributions:
        attribution_counts[item["decision_kind"]] = (
            attribution_counts.get(item["decision_kind"], 0) + 1
        )
    manifest = {
        "schema": SFT_EXPORT_SCHEMA,
        "run_id": result.get("run_id"),
        "source_commit": plan.get("source", {}).get("commit"),
        "trace_final_sha256": verification.get("final_event_sha256"),
        "context_profile": plan.get("policy", {}).get("context_profile"),
        "context_projection_sha256": sha256_bytes(canonical_bytes([
            event.get("payload") for event in events
            if event.get("type") == "context_projected"
        ])),
        "exact_request_sha256": [
            (_canonical_provider_request_ref(event) or {}).get("sha256")
            for event in events
            if event.get("type") == "provider_request"
        ],
        "official_success": result.get("official_success"),
        "termination": result.get("termination"),
        "record_count": len(records),
        "attribution_counts": attribution_counts,
        "loss_mask_contract": {
            "level": "message",
            "target": "final_language_model_assistant_response_only",
            "history_tools_observations_and_receipts": "context_only",
            "tokenizer_compilation_required": True,
        },
        "files": [{
            "path": jsonl.name,
            "sha256": file_sha256(jsonl),
            "bytes": jsonl.stat().st_size,
        }, {
            "path": attribution_jsonl.name,
            "sha256": file_sha256(attribution_jsonl),
            "bytes": attribution_jsonl.stat().st_size,
        }, {
            "path": loss_mask_jsonl.name,
            "sha256": file_sha256(loss_mask_jsonl),
            "bytes": loss_mask_jsonl.stat().st_size,
        }],
    }
    if "context_retention" in plan.get("policy", {}):
        manifest["context_retention"] = plan["policy"]["context_retention"]
    _atomic_json(destination / "manifest.json", manifest)
    return destination


def image_data_url(path: Path) -> str:
    """Return one deterministic PNG/JPEG provider content URL."""

    suffix = path.suffix.lower()
    media = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{media};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"
