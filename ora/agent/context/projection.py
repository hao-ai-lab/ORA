"""Pure inventory, projection, and validation primitives for Native Context."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ora.agent.contracts import PolicySessionError, SummaryProvider
from ora.agent.context.management import (
    ContextManagementConfigError,
    validate_context_management_v4,
)
from ora.agent.evidence.trajectory import ArtifactStore, canonical_bytes
from ora.agent.memory.artifacts import (
    ARTIFACT_READ_NAMES,
    CONTEXT_ARTIFACT_REF_SCHEMA,
    ContextArtifactCatalog,
    LOSSLESS_CONTEXT_RETENTION,
    RECOVERABLE_ARTIFACT_ENVELOPE_SCHEMA,
    RECOVERABLE_ARTIFACT_TAG,
    decode_recoverable_artifact_marker,
    encode_recoverable_artifact_marker,
    recoverable_artifact_pointer,
    recoverable_artifact_open,
    validate_recoverable_artifact_envelope,
    validate_recoverable_artifact_pointer,
)
from ora.agent.providers.continuation import (
    PROVIDER_CONTINUATION_MESSAGE_KEY,
)

from . import historical_observation, round_static_facts
from .inline_media import InlineMediaCache
from .preview_shortfall import _preview_shortfall
from .request_size import estimated_tokens_from_bytes, json_bytes, pass_record


# The Context values this tree composes.
CONTEXT_PROFILES = {"native_minimal_v1", "native_causal_context_v1"}
CONTEXT_RETENTIONS = {"rewrite_compact_v1", LOSSLESS_CONTEXT_RETENTION}
SUMMARY_POLICIES = {"disabled", "rollout_checkpoint_v1"}
IMAGE_PRESSURE_ESTIMATE_TOKENS = 4096
# Retentions that close physical ActionSteps; each takes an exact
# ``context_management`` and the native causal context.
ACTION_STEP_RETENTIONS = {LOSSLESS_CONTEXT_RETENTION}
ROLLOUT_PACKET_SCHEMA = "ora.rollout-compaction-packet.v1"
EPISODE_CAPSULE_SCHEMA = "ora.episode-context-capsule.v1"
EMERGENCY_CHECKPOINT_SCHEMA = "ora.emergency-context-checkpoint.v1"
CHECKPOINT_FIELDS = (
    "task_progress",
    "confirmed_facts",
    "invalidated_beliefs",
    "failed_attempts",
    "current_constraints",
    "do_not_repeat",
    "re_ground_next",
)


class ContextCompactionRejected(PolicySessionError):
    """The untrusted compactor response could not form an admissible checkpoint."""


def _json_bytes(value: Any) -> int:
    return json_bytes(value)


def _estimated_tokens(value: Any) -> int:
    return estimated_tokens_from_bytes(json_bytes(value))


def _pressure_estimated_tokens(value: Any) -> int:
    """Estimate request pressure without tokenizing inline base64 as prose."""

    image_count = 0

    def scrub(item: Any) -> Any:
        nonlocal image_count
        if isinstance(item, Mapping):
            if item.get("type") == "image_url" and "image_url" in item:
                image_count += 1
                return {
                    "type": "image_url",
                    "image_payload": "externalized_for_estimate",
                }
            return {str(key): scrub(child) for key, child in item.items()}
        if isinstance(item, list):
            return [scrub(child) for child in item]
        return item

    return _estimated_tokens(scrub(value)) + image_count * IMAGE_PRESSURE_ESTIMATE_TOKENS


def _inline_image_count(value: Any) -> int:
    if isinstance(value, Mapping):
        return int(value.get("type") == "image_url") + sum(
            _inline_image_count(child) for child in value.values()
        )
    if isinstance(value, list):
        return sum(_inline_image_count(child) for child in value)
    return 0


def _message_event_ids(messages: Sequence[Mapping[str, Any]], indexes: Sequence[int] | None = None) -> list[str]:
    selected = range(len(messages)) if indexes is None else indexes
    found: set[str] = set()
    for index in selected:
        raw = messages[index].get("_source_event_ids")
        if isinstance(raw, list):
            found.update(item for item in raw if isinstance(item, str))
    return sorted(found)


def _packet_source_event_ids(
    messages: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Require every packet message to carry current-rollout evidence identity."""

    found: set[str] = set()
    for index, message in enumerate(messages):
        raw = message.get("_source_event_ids")
        if not isinstance(raw, list) or not raw:
            raise PolicySessionError(
                f"rollout compaction packet message {index} lacks source events"
            )
        for event_id in raw:
            if not isinstance(event_id, str) or not re.fullmatch(
                r"agent-event-[0-9]{6}", event_id,
            ):
                raise PolicySessionError(
                    "rollout compaction packet has an invalid source event id"
                )
            found.add(event_id)
    return sorted(found)


_pass_record = pass_record


def _provider_message(message: Mapping[str, Any]) -> dict[str, Any]:
    """Strip Harness metadata while retaining opaque provider continuity."""

    return {
        key: copy.deepcopy(value) for key, value in message.items()
        if (
            not str(key).startswith("_")
            or key == PROVIDER_CONTINUATION_MESSAGE_KEY
        ) and key != "reasoning_content"
    }


def _exclude_reasoning(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[int]]:
    output = copy.deepcopy(messages)
    changed = []
    for index, message in enumerate(output):
        if "reasoning_content" in message:
            message.pop("reasoning_content")
            changed.append(index)
    return output, changed


def _strip_historical_images(
    messages: list[dict[str, Any]], *, current_step: int,
    replacement_text: str | None = None,
) -> tuple[list[dict[str, Any]], list[int]]:
    changed: list[int] = []
    output: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        item = copy.deepcopy(message)
        metadata = item.get("_observation")
        step = metadata.get("step") if isinstance(metadata, Mapping) else item.get("_tool_image_step")
        content = item.get("content")
        if (
            item.get("role") == "user"
            and isinstance(content, list)
            and isinstance(step, int)
            and step != current_step
        ):
            next_content = []
            removed = False
            for block in content:
                if isinstance(block, Mapping) and block.get("type") == "image_url":
                    removed = True
                    continue
                next_content.append(block)
            if removed:
                next_content.append({
                    "type": "text",
                    "text": replacement_text or (
                        f"[historical observation image omitted; recover step {step} "
                        "with observation.dump_state or vision.inspect]"
                    ),
                })
                item["content"] = next_content
                changed.append(index)
        output.append(item)
    return output, changed


def _compact_old_causal_observations(
    messages: list[dict[str, Any]], *, current_step: int,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Keep the action/tool transcript while removing superseded world snapshots."""

    output = copy.deepcopy(messages)
    changed: list[int] = []
    for index, message in enumerate(output):
        metadata = message.get("_causal_observation")
        if not isinstance(metadata, Mapping) or metadata.get("step") == current_step:
            continue
        message["content"] = [{
            "type": "text",
            "text": "[earlier actual observation omitted; later actions and tool results follow]",
        }]
        changed.append(index)
    return output, changed


_CAUSAL_FORBIDDEN_KEYS = {
    "schema", "schema_version", "action_id", "receipt_id",
    "run_id", "event_id", "causal_parent_id", "plan_id", "sha256", "hash",
    "path", "artifact_path", "timestamp", "timestamps", "started_unix_s",
    "finished_unix_s", "worker", "transport", "transport_status",
    "reached", "verified", "state_changed",
    "object_gt", "object_gt_pose", "contact", "success_predicate",
    "task_source", "private_mask", "official_outcome",
}

# Tool results retain operational provenance in canonical evidence, but the
# provider view needs only decision-relevant semantics.  Keep this policy next
# to the causal-context validator so result projection and final request audit
# cannot drift into separate hand-maintained tool-name projectors.
# Two different things were in one set. ``diagnostics`` is content a producer
# builds for the policy -- grasp.plan's candidate ledger lives there -- so a
# result schema that declares it makes it promised evidence. ``backend`` is the
# tool's own model identity, version and config hash: provenance the policy
# cannot act on, and sixty schema sites declare it as an artefact of how those
# schemas were written rather than as a promise. It stays stripped.
#
# EXPIRY: when the v4 surface generation lands, remove ``backend`` from those
# declarations and delete this split -- at that point a declaration means a
# promise everywhere and the special case is dead weight.
_CAUSAL_RESULT_PROVENANCE_KEYS = {"backend"}
_CAUSAL_RESULT_CONTENT_KEYS = {"diagnostics"}
_CAUSAL_RESULT_OPERATIONAL_KEYS = _CAUSAL_RESULT_PROVENANCE_KEYS | _CAUSAL_RESULT_CONTENT_KEYS


def _logical_artifact_identity(value: Mapping[str, Any]) -> bool:
    reference = value.get("ref")
    digest = value.get("sha256")
    return bool(
        isinstance(reference, str)
        and isinstance(digest, str)
        and re.fullmatch(
            r"artifact://"
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}/[0-9a-f]{64}",
            reference,
        )
        and reference.endswith("/" + digest)
        and "path" not in value
        and "artifact_path" not in value
    )


def _recoverable_artifact_envelope_identity(value: Mapping[str, Any]) -> bool:
    if value.get("schema") != RECOVERABLE_ARTIFACT_ENVELOPE_SCHEMA:
        return False
    try:
        validate_recoverable_artifact_envelope(value)
    except ValueError:
        return False
    return True


def causal_provider_value(value: Any, *, declared: frozenset[str] = frozenset()) -> Any:
    """Recursively remove audit/provenance fields from one tool-result value.

    ``declared`` names the top-level properties a public result schema already
    promises. Those survive the *operational* filter, which exists to drop
    provenance that leaks in undeclared, not to overrule a schema. Without it the
    three layers disagree: a producer builds ``diagnostics`` deliberately for the
    policy — ``grasp_plan_public_result`` even adds
    ``compiler_returned_candidate_count`` to it — the result schema declares the
    property, and the projection then removes the whole block, so the candidate
    ledger a policy would need in order to cite an untried candidate never
    arrives.

    The privilege filter is not relaxed. ``_CAUSAL_FORBIDDEN_KEYS`` still applies
    to declared properties, and ``declared`` is deliberately not propagated into
    nested values: a nested ``backend`` stays provenance rather than becoming
    promised evidence.
    """

    if isinstance(value, Mapping):
        logical_artifact = _logical_artifact_identity(value)
        recoverable_envelope = _recoverable_artifact_envelope_identity(value)
        return {
            str(key): causal_provider_value(child)
            for key, child in value.items()
            if (
                str(key) not in _CAUSAL_FORBIDDEN_KEYS
                or logical_artifact and str(key) in {"schema", "sha256"}
                or recoverable_envelope and str(key) == "schema"
            )
            and (
                str(key) not in _CAUSAL_RESULT_OPERATIONAL_KEYS
                or (str(key) in declared
                    and str(key) not in _CAUSAL_RESULT_PROVENANCE_KEYS)
            )
        }
    if isinstance(value, list):
        return [causal_provider_value(child) for child in value]
    return copy.deepcopy(value)


def _causal_projection_violations(value: Any, *, location: str = "messages") -> list[str]:
    violations: list[str] = []
    if isinstance(value, Mapping):
        logical_artifact = _logical_artifact_identity(value)
        recoverable_envelope = _recoverable_artifact_envelope_identity(value)
        if logical_artifact and value.get("schema") not in {
            None, CONTEXT_ARTIFACT_REF_SCHEMA,
        }:
            violations.append(f"{location}.schema")
        if (
            value.get("schema") == RECOVERABLE_ARTIFACT_ENVELOPE_SCHEMA
            and not recoverable_envelope
        ):
            violations.append(f"{location}.schema")
        for raw_key, child in value.items():
            key = str(raw_key)
            if key in _CAUSAL_FORBIDDEN_KEYS and not (
                logical_artifact and key in {"schema", "sha256"}
                or recoverable_envelope and key == "schema"
            ):
                violations.append(f"{location}.{key}")
            violations.extend(
                _causal_projection_violations(child, location=f"{location}.{key}")
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            violations.extend(
                _causal_projection_violations(child, location=f"{location}[{index}]")
            )
    elif isinstance(value, str) and value.lstrip().startswith(("{", "[")):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, (Mapping, list)):
            violations.extend(
                _causal_projection_violations(decoded, location=f"{location}<json>")
            )
    return violations


_COMPACTOR_PRIVATE_KEYS = {
    "object_gt", "object_gt_pose", "contact", "success_predicate",
    "task_source", "private_mask", "official_outcome", "official_success",
    "official_verdict", "task_success",
}


def _compactor_private_violations(
    value: Any, *, location: str = "packet",
) -> list[str]:
    violations: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            if key in _COMPACTOR_PRIVATE_KEYS:
                violations.append(f"{location}.{key}")
            violations.extend(
                _compactor_private_violations(child, location=f"{location}.{key}")
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            violations.extend(
                _compactor_private_violations(child, location=f"{location}[{index}]")
            )
    elif isinstance(value, str) and value.lstrip().startswith(("{", "[")):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, (Mapping, list)):
            violations.extend(
                _compactor_private_violations(
                    decoded, location=f"{location}<json>"
                )
            )
    return violations


def _spill_large_tool_results(
    messages: list[dict[str, Any]],
    *,
    artifacts: ArtifactStore,
    threshold_bytes: int,
    preview_bytes: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output: list[dict[str, Any]] = []
    refs: list[dict[str, Any]] = []
    for message in messages:
        item = copy.deepcopy(message)
        if item.get("role") == "tool" and isinstance(item.get("content"), str):
            raw = item["content"].encode("utf-8")
            if len(raw) > threshold_bytes:
                existing = item.get("_artifact_ref")
                ref = dict(existing) if isinstance(existing, Mapping) else artifacts.put_bytes(
                    raw,
                    kind="native_tool_result",
                    suffix=".txt",
                    media_type="text/plain",
                )
                preview = raw[:preview_bytes].decode("utf-8", errors="ignore")
                item["content"] = json.dumps({
                    # The exact result remains bound to this message through
                    # the private ``_artifact_ref`` metadata and projection
                    # manifest.  Provider-visible context must not expose the
                    # artifact store path or content hash: those values are
                    # audit provenance, not decision-relevant tool output.
                    "complete_result_recorded": True,
                    "preview": preview,
                    "truncated": True,
                    "retrieve_with": "repeat the read-only tool call to refresh the complete result",
                }, ensure_ascii=False, sort_keys=True)
                item["_artifact_ref"] = ref
                refs.append(ref)
        output.append(item)
    return output, refs


def _materialize_recoverable_payloads(
    messages: list[dict[str, Any]],
    *,
    catalog: ContextArtifactCatalog,
    mutation_tools: Sequence[str],
    advertise_tool_result_over_bytes: int,
    media_cache: InlineMediaCache | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[int]]:
    """Write payloads and advertise only references worth their wire cost."""

    media = media_cache if media_cache is not None else InlineMediaCache()
    media.bind(catalog)
    output = copy.deepcopy(messages)
    mutation_names = set(mutation_tools)
    refs: list[dict[str, Any]] = []
    advertised_indexes: set[int] = set()
    groups = _observation_groups(output)
    group_by_message: dict[int, int] = {}
    round_by_message: dict[int, int] = {}
    for action_step, (start, end) in enumerate(groups):
        round_value = output[start].get("_session_round")
        for index in range(start, end):
            group_by_message[index] = action_step
            if isinstance(round_value, int) and not isinstance(round_value, bool):
                round_by_message[index] = round_value
    for index, item in enumerate(output):
        action_step = group_by_message.get(index)
        round_index = round_by_message.get(index)
        observation_ref = historical_observation.materialize(
            item, catalog=media, round_index=round_index, action_step=action_step,
        )
        if observation_ref is not None:
            refs.append(observation_ref)
        slot_ref = round_static_facts.materialize(
            item, catalog=media, round_index=round_index, action_step=action_step,
        )
        if slot_ref is not None:
            refs.append(slot_ref)
        content = item.get("content")
        if isinstance(content, list):
            existing_refs = item.get("_context_image_refs")
            if existing_refs is not None and not isinstance(existing_refs, list):
                raise PolicySessionError("inline media Artifact inventory is invalid")
            image_refs: list[dict[str, Any]] = []
            next_content: list[Any] = []
            image_index = 0
            for block in content:
                if not isinstance(block, Mapping) or block.get("type") != "image_url":
                    next_content.append(copy.deepcopy(block))
                    continue
                reference = media.admit_image(
                    block.get("image_url"), existing_refs=existing_refs,
                    image_index=image_index, round_index=round_index,
                    action_step=action_step,
                )
                pointer = recoverable_artifact_pointer(reference)
                marker = encode_recoverable_artifact_marker(pointer)
                previous = next_content[-1] if next_content else None
                previous_text = (
                    previous.get("text") if isinstance(previous, Mapping)
                    and previous.get("type") == "text" else None
                )
                if isinstance(previous_text, str) and previous_text.startswith(
                    f"<{RECOVERABLE_ARTIFACT_TAG}>",
                ):
                    try:
                        advertised = decode_recoverable_artifact_marker(previous_text)
                    except ValueError as exc:
                        raise PolicySessionError(
                            "inline media recovery marker is invalid"
                        ) from exc
                    if canonical_bytes(advertised) != canonical_bytes(pointer):
                        raise PolicySessionError(
                            "inline media recovery marker identity drift"
                        )
                else:
                    next_content.append({"type": "text", "text": marker})
                    advertised_indexes.add(index)
                next_content.append(copy.deepcopy(block))
                image_refs.append(reference)
                refs.append(reference)
                image_index += 1
            if isinstance(existing_refs, list) and len(existing_refs) != image_index:
                raise PolicySessionError("inline media Artifact inventory drift")
            if image_refs:
                item["content"] = next_content
                item["_context_image_refs"] = image_refs
        if item.get("role") == "tool" and isinstance(content, str):
            media_type = "text/plain"
            suffix = ".txt"
            parsed: Any = None
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                pass
            else:
                media_type = "application/json"
                suffix = ".json"
            reference = media.put_bytes(
                content.encode("utf-8"),
                kind="context_tool_result",
                suffix=suffix,
                media_type=media_type,
                round_index=round_index,
                action_step=action_step,
            )
            item["_context_artifact_ref"] = reference
            item["_context_original_content"] = content
            if (
                item.get("name") not in mutation_names
                # A read already returns the original artifact identity and
                # requested bytes. Advertising this response's wrapper would
                # create another read instead of delivering the reopened data.
                # Keep the private backing reference for historical pressure GC.
                and item.get("name") not in ARTIFACT_READ_NAMES
                and not _tool_result_has_exact_causal_outcome(item)
                and len(content.encode("utf-8")) > advertise_tool_result_over_bytes
            ):
                pointer = recoverable_artifact_pointer(reference)
                if isinstance(parsed, Mapping):
                    advertised = parsed.get("recoverable_artifact")
                    if advertised is not None:
                        try:
                            advertised = validate_recoverable_artifact_pointer(
                                advertised,
                            )
                        except ValueError as exc:
                            raise PolicySessionError(
                                "tool-result recovery pointer is invalid"
                            ) from exc
                        if canonical_bytes(advertised) != canonical_bytes(pointer):
                            raise PolicySessionError(
                                "tool-result recovery pointer identity drift"
                            )
                    else:
                        parsed = {
                            **copy.deepcopy(dict(parsed)),
                            "recoverable_artifact": pointer,
                        }
                    item["content"] = json.dumps(
                        parsed, ensure_ascii=False, sort_keys=True,
                        separators=(",", ":"),
                    )
                else:
                    item["content"] = json.dumps({
                        "payload": parsed if media_type == "application/json" else content,
                        "recoverable_artifact": pointer,
                    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                advertised_indexes.add(index)
            refs.append(reference)
    unique = {
        str(reference["ref"]): reference for reference in refs
        if isinstance(reference.get("ref"), str)
    }
    return (
        output,
        [copy.deepcopy(unique[key]) for key in sorted(unique)],
        sorted(advertised_indexes),
    )


def _eligible_message_indexes(
    messages: Sequence[Mapping[str, Any]], *, through_action_steps: int,
) -> set[int]:
    groups = _observation_groups(messages)
    eligible: set[int] = set()
    for start, end in groups[:through_action_steps]:
        eligible.update(range(start, end))
    return eligible


def _tool_result_has_exact_causal_outcome(message: Mapping[str, Any]) -> bool:
    if message.get("name") in {"check_task", "finish_episode"}:
        return True
    content = message.get("content")
    if not isinstance(content, str):
        return True
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return False
    return isinstance(value, Mapping) and isinstance(value.get("error"), Mapping)


def _recoverable_tool_marker(
    message: Mapping[str, Any], reference: Mapping[str, Any], *, preview_bytes: int,
) -> str:
    content = str(message.get("_context_original_content", message.get("content", "")))
    preview = content.encode("utf-8")[:preview_bytes].decode(
        "utf-8", errors="ignore",
    )
    status: dict[str, Any] = {}
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, Mapping):
        # A ToolSurface may declare a bounded semantic summary. Keep all of
        # its queries visible rather than cutting a large solver result in
        # the middle of the first query's error arrays. The full result is
        # still the exact immutable Artifact addressed below.
        summary = value.get("summary")
        if preview_bytes > 0 and isinstance(summary, (Mapping, list)):
            summary_text = json.dumps(summary, ensure_ascii=False, sort_keys=True,
                                      separators=(",", ":"))
            if len(summary_text.encode("utf-8")) <= 16384:
                preview = summary_text
        for key in (
            "status", "success", "ok", "observation_id", "views",
            "view_dimensions",
        ):
            if key in value:
                status[key] = copy.deepcopy(value[key])
    pointer = recoverable_artifact_pointer(reference)
    return json.dumps({
        **status,
        **_preview_shortfall(content, preview, value),
        "recoverable_artifact": pointer,
        "preview": preview,
        "externalized": True,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _externalize_tool_results(
    messages: list[dict[str, Any]],
    *,
    through_action_steps: int,
    mutation_tools: Sequence[str],
    selector: str,
    threshold_bytes: int,
    preview_bytes: int,
    exclude_tools: Sequence[str] = (),
) -> tuple[list[dict[str, Any]], list[int], list[dict[str, Any]]]:
    # Copy-on-write: unchanged messages are shared with the input, which no
    # lossless pass edits in place; a replaced message gets its own dict.
    output = list(messages)
    eligible = _eligible_message_indexes(
        output, through_action_steps=through_action_steps,
    )
    groups = _observation_groups(output)
    if groups and any(message.get("_full_current_visual") is True for message in output[groups[-1][0]:]):
        eligible.difference_update(range(groups[-1][0], len(output)))
    mutation_names = set(mutation_tools)
    excluded_names = set(exclude_tools)
    content_counts: dict[str, int] = {}
    first_content_index: dict[str, int] = {}
    for index in eligible:
        message = output[index]
        if message.get("role") == "tool" and isinstance(message.get("content"), str):
            reference = message.get("_context_artifact_ref")
            digest = (
                str(reference.get("sha256")) if isinstance(reference, Mapping)
                else hashlib.sha256(message["content"].encode("utf-8")).hexdigest()
            )
            content_counts[digest] = content_counts.get(digest, 0) + 1
            first_content_index.setdefault(digest, index)
    changed: list[int] = []
    refs: list[dict[str, Any]] = []
    for index in sorted(eligible):
        message = output[index]
        content = message.get("content")
        reference = message.get("_context_artifact_ref")
        if (
            message.get("role") != "tool"
            or not isinstance(content, str)
            or not isinstance(reference, Mapping)
            or message.get("_lossless_externalized") is True
            or message.get("name") in mutation_names
            or message.get("name") in excluded_names
            or _tool_result_has_exact_causal_outcome(message)
        ):
            continue
        original = str(message.get("_context_original_content", content))
        digest = str(reference.get("sha256"))
        selected = (
            selector == "large" and len(original.encode("utf-8")) > threshold_bytes
            or selector == "duplicate"
            and content_counts.get(digest, 0) > 1
            and first_content_index.get(digest) != index
            or selector == "remaining"
        )
        if not selected:
            continue
        marker = _recoverable_tool_marker(
            message, reference, preview_bytes=preview_bytes,
        )
        if (
            selector != "large"
            and len(marker.encode("utf-8")) >= len(original.encode("utf-8"))
        ):
            continue
        message = dict(message)
        output[index] = message
        message["content"] = marker
        message["_lossless_externalized"] = True
        changed.append(index)
        refs.append(copy.deepcopy(dict(reference)))
    return output, changed, refs


def _externalize_historical_media(
    messages: list[dict[str, Any]], *, through_action_steps: int,
    tool_image_event_ids: Sequence[str] = (),
) -> tuple[list[dict[str, Any]], list[int], list[dict[str, Any]]]:
    # Copy-on-write, as in ``_externalize_tool_results``.
    output = list(messages)
    eligible = _eligible_message_indexes(
        output, through_action_steps=through_action_steps,
    )
    selected_tool_images = set(tool_image_event_ids)
    eligible.update(
        index for index, message in enumerate(output)
        if isinstance(message.get("_source_event_ids"), list)
        and any(
            isinstance(event_id, str) and event_id in selected_tool_images
            for event_id in message["_source_event_ids"]
        )
    )
    changed: list[int] = []
    refs: list[dict[str, Any]] = []
    for index in sorted(eligible):
        message = output[index]
        content = message.get("content")
        image_refs = message.get("_context_image_refs")
        if not isinstance(content, list) or not isinstance(image_refs, list):
            continue
        reference_index = 0
        next_content: list[Any] = []
        removed = False
        for block in content:
            if isinstance(block, Mapping) and block.get("type") == "image_url":
                if reference_index >= len(image_refs) or not isinstance(
                    image_refs[reference_index], Mapping,
                ):
                    raise PolicySessionError(
                        "historical image lacks its recoverable Artifact reference"
                    )
                reference = copy.deepcopy(dict(image_refs[reference_index]))
                reference_index += 1
                refs.append(reference)
                previous = next_content[-1] if next_content else None
                previous_text = (
                    previous.get("text") if isinstance(previous, Mapping)
                    and previous.get("type") == "text" else None
                )
                if not isinstance(previous_text, str) or not previous_text.startswith(
                    f"<{RECOVERABLE_ARTIFACT_TAG}>",
                ):
                    raise PolicySessionError(
                        "historical image lacks its recovery pointer"
                    )
                try:
                    pointer = decode_recoverable_artifact_marker(previous_text)
                except ValueError as exc:
                    raise PolicySessionError(
                        "historical image recovery pointer is invalid"
                    ) from exc
                if pointer["ref"] != reference.get("ref"):
                    raise PolicySessionError(
                        "historical image recovery pointer identity drift"
                    )
                removed = True
            else:
                next_content.append(copy.deepcopy(block))
        if reference_index != len(image_refs):
            raise PolicySessionError("historical image Artifact inventory drift")
        if removed:
            message = dict(message)
            output[index] = message
            message["content"] = next_content
            message["_lossless_media_externalized"] = True
            changed.append(index)
    return output, changed, refs


def _inline_media_bytes_from_refs(messages: Sequence[Mapping[str, Any]]) -> int:
    total = 0
    for message in messages:
        content = message.get("content")
        references = message.get("_context_image_refs")
        if not isinstance(content, list) or not isinstance(references, list):
            continue
        inline_count = sum(
            isinstance(block, Mapping) and block.get("type") == "image_url"
            for block in content
        )
        if inline_count != len(references):
            continue
        total += sum(
            int(reference.get("bytes", 0))
            for reference in references if isinstance(reference, Mapping)
        )
    return total


def _lossless_invariant_view(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize recoverable payloads by identity while keeping policy prose exact."""

    normalized: list[dict[str, Any]] = []
    for message in messages:
        visible = _provider_message(message)
        tool_ref = message.get("_context_artifact_ref")
        slot_content = round_static_facts.invariant_content(message)
        if message.get("role") == "tool" and isinstance(tool_ref, Mapping):
            visible["content"] = {
                "recoverable_sha256": tool_ref.get("sha256"),
                "media_type": tool_ref.get("media_type"),
                "bytes": tool_ref.get("bytes"),
            }
        elif slot_content is not None:
            visible["content"] = slot_content
        elif isinstance(message.get("content"), list):
            content: list[Any] = []
            observation_block = historical_observation.invariant_block(message)
            for block_index, block in enumerate(message["content"]):
                if block_index == 0 and observation_block is not None:
                    block = observation_block
                if (
                    isinstance(block, Mapping)
                    and block.get("type") == "text"
                    and str(block.get("text", "")).startswith(
                        "[recoverable historical image externalized]"
                    )
                ):
                    continue
                if isinstance(block, Mapping) and block.get("type") == "image_url":
                    continue
                content.append(copy.deepcopy(block))
            image_refs = message.get("_context_image_refs")
            if isinstance(image_refs, list):
                content.extend({
                    "type": "recoverable_image_identity",
                    "sha256": reference.get("sha256"),
                    "media_type": reference.get("media_type"),
                    "bytes": reference.get("bytes"),
                } for reference in image_refs if isinstance(reference, Mapping))
            visible["content"] = content
        normalized.append(visible)
    return normalized


def _observation_groups(messages: Sequence[Mapping[str, Any]]) -> list[tuple[int, int]]:
    starts = [
        index for index, message in enumerate(messages)
        if message.get("role") == "user" and isinstance(message.get("_observation"), Mapping)
    ]
    groups: list[tuple[int, int]] = []
    for offset, start in enumerate(starts):
        end = starts[offset + 1] if offset + 1 < len(starts) else len(messages)
        groups.append((start, end))
    return groups


def episode_index_keyframe(
    *, step: int, previous_action: Mapping[str, Any] | None,
) -> bool:
    """Decide the ``keyframe`` flag of one observation's Harness metadata.

    The flag selected the steps the ``native_context_v1`` profile's collapsed
    ``<episode_index>`` highlighted. That profile composes only at the commits
    that composed it; the flag stays on the observation metadata and nothing
    it decides is stated to the policy.

    It also read the per-action runtime notice, keeping an observation whose
    guard raised `warning`, `error` or `stagnation`, or whose head image moved
    more than four hash bits. The notice no longer carries any of those: the
    guard's booleans were retired, and its surviving image distance is not
    routed to the policy context. Step cadence and the previous action are
    what remain readable here.
    """

    if step == 0 or step % 10 == 0:
        return True
    return bool(
        isinstance(previous_action, Mapping)
        and (
            previous_action.get("action") == "set_gripper"
            or isinstance(previous_action.get("arguments"), Mapping)
            and previous_action["arguments"].get("gripper") is not None
        )
    )


def _action_step_inventory(
    messages: Sequence[Mapping[str, Any]],
    *,
    mutation_tools: Sequence[str],
    require_session_round: bool = False,
) -> list[dict[str, Any]]:
    """Validate observation groups as complete or open physical ActionSteps."""

    groups = _observation_groups(messages)
    mutation_names = set(mutation_tools)
    inventory: list[dict[str, Any]] = []
    for index, (start, end) in enumerate(groups):
        observation = messages[start].get("_observation")
        if not isinstance(observation, Mapping):
            raise PolicySessionError("ActionStep lacks its input actual observation")
        session_round = messages[start].get("_session_round")
        if require_session_round and (
            isinstance(session_round, bool)
            or not isinstance(session_round, int)
            or session_round < 1
        ):
            raise PolicySessionError("ActionStep lacks its positive Session round")
        if not isinstance(session_round, int) or isinstance(session_round, bool):
            session_round = 1
        calls_by_id: dict[str, tuple[str, Mapping[str, Any]]] = {}
        nameless_calls: set[str] = set()
        tool_results: dict[str, tuple[str | None, Any]] = {}
        for message in messages[start:end]:
            if message.get("role") == "assistant":
                calls = message.get("tool_calls")
                if isinstance(calls, list):
                    for call in calls:
                        function = call.get("function") if isinstance(call, Mapping) else None
                        name = function.get("name") if isinstance(function, Mapping) else None
                        call_id = call.get("id") if isinstance(call, Mapping) else None
                        if not isinstance(call_id, str) or not call_id:
                            raise PolicySessionError(
                                "ActionStep tool call lacks a stable provider id"
                            )
                        if call_id in calls_by_id:
                            raise PolicySessionError(
                                f"ActionStep {index} reuses tool-call id {call_id}"
                            )
                        if not isinstance(name, str) or not name:
                            nameless_calls.add(call_id)
                            name = "invalid_tool_call"  # Inventory label only; raw history stays exact.
                        calls_by_id[call_id] = (name, call)
            if message.get("role") == "tool" and isinstance(
                message.get("tool_call_id"), str
            ):
                call_id = str(message["tool_call_id"])
                if call_id in tool_results:
                    raise PolicySessionError(
                        f"ActionStep {index} duplicates tool result {call_id}"
                    )
                content = message.get("content")
                if isinstance(content, str):
                    try:
                        content = json.loads(content)
                    except json.JSONDecodeError:
                        pass
                result_name = message.get("name")
                tool_results[call_id] = (
                    str(result_name) if isinstance(result_name, str) else None,
                    content,
                )
        if set(calls_by_id) != set(tool_results):
            raise PolicySessionError(
                f"ActionStep {index} does not pair every tool call and result"
            )
        for call_id, (name, _call) in calls_by_id.items():
            result_name, _content = tool_results[call_id]
            if call_id in nameless_calls and not (
                result_name == "invalid_tool_call" and isinstance(_content, Mapping)
                and _content.get("ok") is not True and _content.get("success") is not True
                and isinstance(_content.get("error"), Mapping)
                and _content["error"].get("code") == "invalid_tool_call"
                and _content["error"].get("retryable") is True
            ):
                raise PolicySessionError("ActionStep nameless tool call lacks its invalid_tool_call refusal")
            if result_name is not None and result_name != name:
                raise PolicySessionError(
                    f"ActionStep {index} tool result name differs from its call"
                )
        mutation_calls = [
            (name, call_id) for call_id, (name, _call) in calls_by_id.items()
            if name in mutation_names
        ]
        accepted_mutations = [
            (name, call_id) for name, call_id in mutation_calls
            if isinstance(tool_results[call_id][1], Mapping)
            and not isinstance(tool_results[call_id][1].get("error"), Mapping)
            and tool_results[call_id][1].get("status") in {
                "submitted", "action_submitted",
            }
        ]
        if len(accepted_mutations) > 1:
            raise PolicySessionError(
                f"ActionStep {index} contains more than one robot mutation"
            )
        complete = index < len(groups) - 1
        post_actual = None
        if complete:
            next_observation = messages[groups[index + 1][0]].get("_observation")
            if not isinstance(next_observation, Mapping):
                raise PolicySessionError(
                    f"ActionStep {index} lacks its post-action actual observation"
                )
            post_actual = next_observation.get("step")
            # After a no-action outcome (model_call_limit, ...) the Supervisor's
            # continue_same_observation recovery presents the same observation
            # again: same step and identity, no action, no receipt between.
            presented_again = (
                not accepted_mutations
                and post_actual == observation.get("step")
                and isinstance(observation.get("observation_id"), str)
                and next_observation.get("observation_id")
                == observation["observation_id"]
            )
            if isinstance(observation.get("step"), int) and not presented_again and (
                not isinstance(post_actual, int)
                or post_actual != int(observation["step"]) + 1
            ):
                raise PolicySessionError(
                    f"ActionStep {index} post-action observation is not the next step"
                )
            if accepted_mutations:
                receipt = next_observation.get("receipt")
                if (
                    not isinstance(receipt, Mapping)
                    or not isinstance(receipt.get("event_id"), str)
                    or not receipt["event_id"]
                    or not isinstance(receipt.get("action_id"), str)
                    or not receipt["action_id"]
                ):
                    raise PolicySessionError(
                        f"ActionStep {index} mutation lacks an identity-bound environment receipt"
                    )
        source_ids = _message_event_ids(messages, range(start, end))
        inventory.append({
            "action_step_index": index,
            "session_round": session_round,
            "observation_step": observation.get("step"),
            "post_observation_step": post_actual,
            "complete": complete,
            "message_start": start,
            "message_end": end,
            "message_count": end - start,
            "inline_images": _inline_image_count(list(messages[start:end])),
            "mutation": accepted_mutations[0][0] if accepted_mutations else None,
            "mutation_tool_call_id": (
                accepted_mutations[0][1] if accepted_mutations else None
            ),
            "receipt_event_id": (
                messages[groups[index + 1][0]].get("_observation", {})
                .get("receipt", {}).get("event_id")
                if complete and accepted_mutations else None
            ),
            "receipt_action_id": (
                messages[groups[index + 1][0]].get("_observation", {})
                .get("receipt", {}).get("action_id")
                if complete and accepted_mutations else None
            ),
            "source_event_ids": source_ids,
        })
    return inventory


def _packet_message_blocks(
    *, section: str, messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten canonical history into one tool-free multimodal packet message."""

    blocks: list[dict[str, Any]] = [{
        "type": "text", "text": f"<{section}>"
    }]
    for message in messages:
        visible = _compactor_history_message(message)
        content = visible.pop("content", "")
        descriptor = copy.deepcopy(visible)
        blocks.append({
            "type": "text",
            "text": json.dumps(
                descriptor, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
        })
        if isinstance(content, str):
            if content:
                blocks.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, Mapping):
                    continue
                if block.get("type") == "image_url":
                    blocks.append(copy.deepcopy(dict(block)))
                elif block.get("type") == "text" and isinstance(block.get("text"), str):
                    blocks.append({"type": "text", "text": str(block["text"])})
        else:
            blocks.append({
                "type": "text",
                "text": json.dumps(content, ensure_ascii=False, sort_keys=True),
            })
    blocks.append({"type": "text", "text": f"</{section}>"})
    return blocks


def _compactor_history_message(message: Mapping[str, Any]) -> dict[str, Any]:
    """Return policy-visible history minus fields reserved from the Compactor."""

    visible = _provider_message(message)
    visible.pop(PROVIDER_CONTINUATION_MESSAGE_KEY, None)
    # The main policy may call the faithful task checker, but the rollout
    # checkpoint is explicitly forbidden from carrying an official outcome.
    # Preserve tool adjacency and evidence identity while withholding the
    # checker's Boolean from this separate model boundary. Arbitrary
    # ``task_success`` fields elsewhere remain privilege violations.
    if visible.get("role") == "tool" and visible.get("name") == "check_task":
        content = visible.get("content")
        if isinstance(content, str):
            try:
                decoded = json.loads(content)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, Mapping) and "task_success" in decoded:
                visible["content"] = json.dumps({
                    "status": "task_check_result_withheld_from_compactor",
                    "reason": "official_outcome_reserved",
                }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return visible


_CHECKPOINT_FORBIDDEN = re.compile(
    r"(?:\brun[_ -]?id\b|\bsha-?256\b|"
    r"\b(?:official[_-](?:success|verdict|outcome)|task[_-]success)\b|"
    r"\bpixel coordinates?\b|\bworld coordinates?\b|"
    r"\b(?:target|historical) pose\b|"
    r"\b(?:pose_xyz|position_m|delta_mm|point_uv|xyz_m)\b|"
    r"\b[xyz]\s*(?:=|~)|"
    r"(?:/home/|outputs/|native/agent-artifacts/)|\b[0-9a-f]{64}\b)",
    re.IGNORECASE,
)

_CHECKPOINT_OUTCOME_ASSERTION = re.compile(
    r"\b(?:official[ ]+(?:success|verdict|outcome)|task[ ]+success)\b"
    r"\s*(?:(?:is|was|=|:|returned|reported|indicated|showed)\s*)"
    r"(?:true|false|success(?:ful)?|fail(?:ed|ure)?)\b",
    re.IGNORECASE,
)


def checkpoint_forbidden_fragments(value: Mapping[str, Any]) -> list[str]:
    """Return deterministic replay/audit fragments rejected from a checkpoint."""

    serialized = json.dumps(value, ensure_ascii=False)
    return sorted({
        *(match.group(0) for match in _CHECKPOINT_FORBIDDEN.finditer(serialized)),
        *(match.group(0) for match in _CHECKPOINT_OUTCOME_ASSERTION.finditer(serialized)),
    })


def _validate_checkpoint(
    content: str, *, max_estimated_tokens: int | None = None,
) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ContextCompactionRejected(
            "context compactor returned invalid JSON"
        ) from exc
    if not isinstance(value, dict) or set(value) != set(CHECKPOINT_FIELDS):
        raise ContextCompactionRejected(
            "context compactor checkpoint fields do not match the v1 schema"
        )
    value = {field: value[field] for field in CHECKPOINT_FIELDS}
    for field in CHECKPOINT_FIELDS:
        entries = value[field]
        if not isinstance(entries, list):
            raise ContextCompactionRejected(
                f"context checkpoint {field} must be an array"
            )
        if field == "failed_attempts":
            for entry in entries:
                if not isinstance(entry, dict) or set(entry) != {
                    "failure", "evidence", "correction",
                } or not all(isinstance(entry[key], str) for key in entry):
                    raise ContextCompactionRejected(
                        "context checkpoint failed_attempts entry is invalid"
                    )
        elif not all(isinstance(entry, str) for entry in entries):
            raise ContextCompactionRejected(
                f"context checkpoint {field} entry is invalid"
            )
    if not any(value[field] for field in CHECKPOINT_FIELDS):
        raise ContextCompactionRejected(
            "context compactor returned an empty checkpoint"
        )
    forbidden = checkpoint_forbidden_fragments(value)
    if forbidden:
        raise ContextCompactionRejected(
            "context checkpoint contains forbidden replay/audit data: "
            + ", ".join(forbidden)
        )
    if (
        max_estimated_tokens is not None
        and _estimated_tokens(value) > max_estimated_tokens
    ):
        raise ContextCompactionRejected(
            "context checkpoint exceeds its Harness token budget"
        )
    return copy.deepcopy(value)


def _semantic_capsule_slot(capsule: Mapping[str, Any]) -> dict[str, Any]:
    envelope = {
        "schema": EPISODE_CAPSULE_SCHEMA,
        "session_round": capsule["session_round"],
        "coverage": copy.deepcopy(capsule["coverage"]),
        "official_verdict": copy.deepcopy(capsule["official_verdict"]),
        "exact_transcript": copy.deepcopy(capsule["exact_transcript"]),
        "summary": {
            field: copy.deepcopy(capsule["summary"][field])
            for field in CHECKPOINT_FIELDS
        },
    }
    return {
        "role": "user",
        "content": (
            "<episode_context_capsule>\n"
            + json.dumps(
                envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            )
            + "\n</episode_context_capsule>"
        ),
        "_protected_context": True,
        "_context_source": f"episode_context_capsule_{capsule['session_round']}",
        "_episode_context_capsule": True,
    }


def _tiered_semantic_projection(
    messages: Sequence[Mapping[str, Any]],
    *,
    episode_checkpointed_action_steps: int,
    episode_capsules: Sequence[Mapping[str, Any]],
    emergency_summarized_action_steps: int,
    emergency_checkpoint: Mapping[str, Any] | None,
    emergency_transcript_ref: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Project exact static prefix, semantic slots, then the untouched tail."""

    groups = _observation_groups(messages)
    if not groups:
        return copy.deepcopy([dict(message) for message in messages])
    absorbed = max(
        episode_checkpointed_action_steps, emergency_summarized_action_steps,
    )
    if absorbed >= len(groups):
        raise PolicySessionError("semantic projection must retain an open ActionStep")
    prefix = [
        copy.deepcopy(dict(message)) for message in messages[:groups[0][0]]
        if message.get("_protected_context") is True
    ]
    slots: list[dict[str, Any]] = []
    active_emergency_projection = (
        emergency_summarized_action_steps > episode_checkpointed_action_steps
    )
    if active_emergency_projection:
        if emergency_checkpoint is None or emergency_transcript_ref is None:
            raise PolicySessionError("emergency projection lacks its checkpoint")
        exact_text = json.dumps({
            "schema": EMERGENCY_CHECKPOINT_SCHEMA,
            "summary": {
                field: emergency_checkpoint[field] for field in CHECKPOINT_FIELDS
            },
            "exact_transcript": {
                **copy.deepcopy(dict(emergency_transcript_ref)),
                "open": recoverable_artifact_open(emergency_transcript_ref),
            },
        }, ensure_ascii=False, sort_keys=False, separators=(",", ":"))
        slots.append({
            "role": "user",
            "content": (
                "<emergency_context_checkpoint>\n" + exact_text
                + "\n</emergency_context_checkpoint>"
            ),
            "_protected_context": True,
            "_context_source": "emergency_context_checkpoint",
            "_emergency_context_checkpoint": True,
        })
    for capsule in episode_capsules:
        coverage = capsule.get("coverage")
        through = coverage.get("through_action_step_index") if isinstance(
            coverage, Mapping,
        ) else None
        if (
            active_emergency_projection
            and isinstance(through, int)
            and through < emergency_summarized_action_steps
        ):
            continue
        slots.append(_semantic_capsule_slot(capsule))
    tail_start = groups[absorbed][0]
    # native_slots_v4: a summarized span never takes the current round's
    # protected static-facts slot with it.
    carried = round_static_facts.current_round_slots(
        messages[groups[0][0]:tail_start],
        current_round=messages[groups[-1][0]].get("_session_round"),
    )
    return [*prefix, *slots, *carried, *copy.deepcopy(messages[tail_start:])]
