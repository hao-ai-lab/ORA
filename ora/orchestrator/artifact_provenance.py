"""Audit Context recovery references at their Harness-owned publication boundary."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ora.agent.memory.artifacts import (
    CONTEXT_ARTIFACT_REF_SCHEMA, validate_recoverable_artifact_pointer,
)
from .evidence_io import _hash
from ora.agent.evidence import request_archive


def _published_pointers(request: Mapping[str, Any]) -> list[Any]:
    pointers: list[Any] = []
    for message in request.get("messages", []):
        # Assistant prose, reasoning and proposed tool arguments are model output.
        if not isinstance(message, Mapping) or message.get("role") not in {"user", "tool"}:
            continue
        content = message.get("content")
        texts = [content] if isinstance(content, str) else [
            block.get("text") for block in content or []
            if isinstance(block, Mapping) and block.get("type") == "text"
        ]
        for text in texts:
            if not isinstance(text, str):
                continue
            tag = next((tag for tag in (
                "recoverable_artifact", "episode_context_capsule",
                "emergency_context_checkpoint",
            ) if text.startswith(f"<{tag}>")), None)
            if tag:
                if not text.endswith(f"</{tag}>"):
                    pointers.append(None)
                    continue
                text = text[len(tag) + 2:-(len(tag) + 3)].strip()
            try:
                body = json.loads(text)
            except ValueError:
                if tag:
                    pointers.append(None)
                continue
            if not isinstance(body, Mapping):
                continue
            if tag == "recoverable_artifact":
                pointers.append(body.get("artifact", body))
            elif tag:
                pointers.append(body.get("exact_transcript"))
            else:
                # Only reserved recovery fields are publication, never arbitrary
                # payload/preview/error strings (which may quote rejected input).
                for key in ("recoverable_artifact", "recoverable_observation"):
                    if key in body:
                        pointer = body[key]
                        pointers.append(pointer.get("artifact", pointer)
                                        if isinstance(pointer, Mapping) else pointer)
    return pointers


def verify_artifact_provenance(
    run_root: Path, events: Sequence[Mapping[str, Any]], artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    """Check issued references and successful reads; report unavailable origins.

    The caller separately verifies trace hashes and every catalog backing file.
    Publication is established by projection inventories and normalized recovery
    fields, not a URI-shaped substring anywhere in a provider request.
    """
    errors: set[str] = set()
    insufficient: set[str] = set()
    issued: set[str] = set()
    retrieved: set[str] = set()
    request_count = 0
    projections = {
        event.get("model_call_id"): event.get("payload", {}) for event in events
        if event.get("type") == "context_projected"
    }

    def check(pointer: Any) -> None:
        ref = pointer.get("ref") if isinstance(pointer, Mapping) else None
        if not isinstance(ref, str) or ref not in artifacts:
            errors.add("unknown_published_context_artifact_ref")
            return
        issued.add(ref)
        entry = artifacts[ref]
        if not isinstance(entry, Mapping) or any(
            pointer[key] != entry.get(key)
            for key in ("sha256", "bytes", "media_type", "policy_scope", "kind")
            if key in pointer
        ):
            errors.add("published_context_artifact_identity_mismatch")
        if "v" in pointer:
            # The compact v1 pointer uses mime and an opener name, whereas
            # expanded public refs use media_type and an optional call mapping.
            try:
                validate_recoverable_artifact_pointer(pointer)
            except ValueError:
                errors.add("published_context_artifact_identity_mismatch")
            if pointer.get("mime") != entry.get("media_type"):
                errors.add("published_context_artifact_identity_mismatch")
        else:
            opener = pointer.get("open")
            if opener is not None:
                arguments = opener.get("arguments") if isinstance(opener, Mapping) else None
                tool = opener.get("tool") if isinstance(opener, Mapping) else None
                expected = {"observe"} if str(entry.get("media_type", "")).startswith("image/") else {"artifact.read", "artifact_read"}
                if not isinstance(arguments, Mapping) or arguments.get("ref") != ref or tool not in expected:
                    errors.add("published_context_artifact_identity_mismatch")

    for event in events:
        payload = event.get("payload", {})
        if not isinstance(payload, Mapping):
            continue
        if event.get("type") == "context_projected":
            passes = payload.get("passes")
            if not isinstance(passes, list):
                insufficient.add("projection_artifact_inventory_missing")
            else:
                for item in passes:
                    if isinstance(item, Mapping):
                        for pointer in item.get("artifact_refs", []):
                            check(pointer)
            for transition in payload.get("projection_transitions", []):
                if isinstance(transition, Mapping) and "exact_transcript" in transition:
                    check(transition["exact_transcript"])
            if payload.get("emergency_exact_transcript_ref") is not None:
                check(payload["emergency_exact_transcript_ref"])
        elif event.get("type") == "context_artifact_retrieval":
            if payload.get("status") == "ok":
                ref = payload.get("ref")
                if not isinstance(ref, str) or ref not in artifacts:
                    errors.add("successful_context_artifact_retrieval_unknown")
                else:
                    retrieved.add(ref)
        elif event.get("type") == "provider_request":
            request_count += 1
            projection = projections.get(event.get("model_call_id"))
            if not isinstance(projection, Mapping):
                insufficient.add("request_projection_origin_missing")
            canonical = payload.get("canonical_request_ref")
            # Historical OpenAI wire requests were already normalized ChatML.
            if not isinstance(canonical, Mapping):
                canonical = event.get("artifact_ref")
            relative = canonical.get("path") if isinstance(canonical, Mapping) else None
            path = (run_root / relative).resolve() if isinstance(relative, str) else None
            if (path is None or not path.is_relative_to(run_root.resolve())
                    or not request_archive.exists(run_root, relative)):
                insufficient.add("canonical_request_origin_missing")
                continue
            try:
                request = json.loads(request_archive.read_bytes(run_root, relative))
            except (ValueError, OSError, request_archive.RequestArchiveError):
                insufficient.add("canonical_request_origin_unreadable")
                continue
            if not isinstance(request, Mapping) or not isinstance(request.get("messages"), list):
                insufficient.add("canonical_request_origin_unknown")
                continue
            # Anthropic wire-only history cannot establish canonical roles.
            if any(isinstance(m, Mapping) and isinstance(m.get("content"), list)
                   and any(isinstance(b, Mapping) and b.get("type") in {"tool_use", "tool_result"}
                           for b in m["content"]) for m in request["messages"]):
                insufficient.add("canonical_request_origin_unknown")
                continue
            for pointer in _published_pointers(request):
                check(pointer)
    return {
        "schema": "ora.context-artifact-provenance.v1",
        "status": "invalid" if errors else "insufficient_evidence" if insufficient else "valid",
        "errors": sorted(errors), "evidence_gaps": sorted(insufficient),
        "provider_requests": request_count, "issued_refs": len(issued),
        "successful_retrieved_refs": len(retrieved),
    }


def context_artifact_entry_errors(
    run_root: Path, artifact_index: Mapping[str, Any], artifacts: Mapping[str, Any],
) -> list[str]:
    """Validate every catalog entry even when no request happens to use it."""
    for logical_ref, entry in artifacts.items():
        storage = entry.get("storage_ref") if isinstance(entry, Mapping) else None
        relative = storage.get("path") if isinstance(storage, Mapping) else None
        path = (
            (run_root / relative).resolve()
            if isinstance(relative, str) else None
        )
        if (
            not isinstance(logical_ref, str)
            or not isinstance(entry, Mapping)
            or entry.get("schema") != CONTEXT_ARTIFACT_REF_SCHEMA
            or logical_ref != entry.get("ref")
            or not logical_ref.startswith(
                f"artifact://{artifact_index.get('policy_session_id')}/"
            )
            or not logical_ref.endswith("/" + str(entry.get("sha256")))
            or entry.get("policy_scope")
            != artifact_index.get("policy_session_id")
            or not isinstance(entry.get("kind"), str)
            or not entry.get("kind")
            or not isinstance(entry.get("media_type"), str)
            or not entry.get("media_type")
            or not isinstance(entry.get("sources"), list)
            or not entry.get("sources")
            or any(
                not isinstance(source, Mapping)
                or source.get("round") is not None
                and (
                    isinstance(source.get("round"), bool)
                    or not isinstance(source.get("round"), int)
                )
                or source.get("action_step") is not None
                and (
                    isinstance(source.get("action_step"), bool)
                    or not isinstance(source.get("action_step"), int)
                )
                for source in entry.get("sources", [])
            )
            or not isinstance(storage, Mapping)
            or storage.get("sha256") != entry.get("sha256")
            or storage.get("bytes") != entry.get("bytes")
            or storage.get("media_type") != entry.get("media_type")
            or path is None
            or Path(relative).is_absolute()
            or not path.is_relative_to(run_root)
            or not path.is_file()
            or path.is_symlink()
            or _hash(path) != entry.get("sha256")
            or path.stat().st_size != entry.get("bytes")
        ):
            return ["context_artifact_entry_invalid"]

    return []
