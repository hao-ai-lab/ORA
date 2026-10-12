"""Deterministic per-turn context, coding and camera-disclosure audit."""

from __future__ import annotations

import hashlib
import json
import os
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Mapping
from ora.agent.evidence.policy_state import context_bundle_images

from .metrics import build_runtime_metrics


CONTEXT_AUDIT_SCHEMA = "ora-context-audit.v2"
TYPED_POLICY_CONTEXT_SCHEMAS = {"ora.policy-context.v3"}


class ContextAuditError(RuntimeError):
    """Run evidence cannot support a deterministic context audit."""


def _load(path: Path, *, required: bool = False) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise ContextAuditError(f"required audit source is missing: {path.name}")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContextAuditError(f"invalid audit source {path.name}: {exc}") from exc
    return value if isinstance(value, dict) else {}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    values = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            values.append(value)
    return values


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_ref(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": _sha256(path),
    }


def policy_delivery_inventory(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Describe authored-source delivery without equating storage with visibility.

    Native typed context is fully assembled by the Harness, so its provider
    placement is exact. Managed Codex/Claude project inputs are known at the
    product boundary, while their final provider placement remains product-
    owned. A tool manual with zero eager injections is never called visible
    merely because it was materialized and hash-bound.
    """

    policy_context = plan.get("policy_context", {})
    typed_context = policy_context.get("schema") in TYPED_POLICY_CONTEXT_SCHEMAS
    policy_kind = plan.get("policy", {}).get("kind")
    channel_specs = (
        {
            "harness_system": (
                "harness-system.md", 1, "system", True, "exact",
                "native_system_message", "conversation_start",
            ),
            "project_instructions": (
                "project-instructions.md", 1, "user", True, "exact",
                "native_project_context_user_message", "conversation_start",
            ),
            "episode_task": (
                "episode-task.md", 1, "user", True, "exact",
                "native_episode_task_user_message", "conversation_start",
            ),
            "tool_manual": (
                "tool-manual.md", 0, None, False, "not_delivered",
                "manifest_only", "never",
            ),
        }
        if typed_context else {
            "codex_project": (
                "AGENTS.md", 1, None, None, "managed_product_owned",
                "managed_project_document", "conversation_start",
            ),
            "episode_prefix": (
                "episode-prefix.md", 1, "user", None,
                "managed_product_owned", "managed_episode_input",
                "conversation_start",
            ),
            "tool_manual": (
                "robot-manual.md", 0, None, False, "on_demand_only",
                "workspace_file_not_eagerly_delivered", "on_demand",
            ),
        }
    )
    layers = []
    for channel, spec in channel_specs.items():
        (
            compiled_name, injection_count, provider_role, provider_visible,
            visibility_status, delivery, delivery_timing,
        ) = spec
        for order, item in enumerate(policy_context.get(channel, [])):
            layers.append({
                "channel": channel,
                "order": order,
                "source_path": item.get("path"),
                "source_bytes": item.get("bytes"),
                "source_sha256": item.get("sha256"),
                "compiled_name": compiled_name,
                "injection_count": injection_count,
                "provider_role": provider_role,
                "provider_visible": provider_visible,
                "provider_visibility_status": visibility_status,
                "delivery": delivery,
                "delivery_timing": delivery_timing,
            })
    task = str(plan.get("task_instruction", ""))
    layers.append({
        "channel": "task", "order": 0, "source_path": "plan.json:task_instruction",
        "source_bytes": len(task.encode("utf-8")),
        "source_sha256": hashlib.sha256(task.encode("utf-8")).hexdigest(),
        "injection_count": 1,
        "provider_role": "user" if typed_context else None,
        "provider_visible": True if typed_context else None,
        "provider_visibility_status": (
            "exact" if typed_context else "managed_product_owned"
        ),
        "delivery": (
            "rendered_into_episode_task"
            if typed_context else "rendered_into_managed_episode_input"
        ),
        "delivery_timing": "conversation_start",
    })
    if policy_kind == "native_api":
        tool_delivery = {
            "provider_role": "tools",
            "provider_visible": True,
            "provider_visibility_status": "exact",
            "delivery": "provider_tools_field",
            "delivery_timing": "every_model_call",
        }
    else:
        tool_delivery = {
            "provider_role": None,
            "provider_visible": None,
            "provider_visibility_status": "runtime_specific",
            "delivery": "runtime_tool_configuration",
            "delivery_timing": "runtime_specific",
        }
    return {
        "policy_context_schema": policy_context.get("schema"),
        "policy_kind": policy_kind,
        "policy_layers": layers,
        "tool_schema": tool_delivery,
    }


def _policy_layers(root: Path, plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    layers = []
    for item in policy_delivery_inventory(plan)["policy_layers"]:
        item = dict(item)
        compiled_name = item.pop("compiled_name", None)
        if isinstance(compiled_name, str):
            # Runs can outlive the checkout. Prefer materialized policy evidence.
            compiled = root / "compiled-policy" / compiled_name
            item.update({
                "compiled_path": compiled.relative_to(root).as_posix(),
                "compiled_bytes": (
                    compiled.stat().st_size if compiled.is_file() else None
                ),
                "compiled_sha256": _sha256(compiled),
            })
        layers.append(item)
    return layers


def _tool_images(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    images = []
    for index, event in enumerate(events):
        result = event.get("result")
        if event.get("type") != "tool_result" or not isinstance(result, Mapping):
            continue
        image_hash = result.get("image_sha256")
        if isinstance(image_hash, str):
            size = result.get("returned_size")
            images.append({
                "event_index": index,
                "tool": event.get("tool"),
                "camera": result.get("camera"),
                "modality": result.get("modality", "rgb"),
                "step": result.get("step"),
                "size_px": size,
                "bytes": result.get("image_bytes"),
                "sha256": image_hash,
            })
    return images


def _repetition(values: list[str]) -> dict[str, Any]:
    counts = Counter(value for value in values if value)
    repeated = [
        {"identity": identity, "count": count, "repetitions_after_first": count - 1}
        for identity, count in sorted(counts.items()) if count > 1
    ]
    return {
        "groups": repeated,
        "repetitions_after_first": sum(item["repetitions_after_first"] for item in repeated),
    }


def build_context_audit(run_root: Path) -> dict[str, Any]:
    """Build an external audit without claiming access to Codex-owned internals."""
    root = run_root.resolve()
    plan = _load(root / "plan.json", required=True)
    result = _load(root / "result.json", required=True)
    session = _load(root / "native/agent-session.json")
    metrics = build_runtime_metrics(root)
    events = _jsonl(root / "events.jsonl")
    coding = _jsonl(root / "native/coding-audit.jsonl")
    bundles = [_load(path) for path in sorted((root / "context-bundles").glob("call-*.json"))]
    tool_images = _tool_images(events)
    coding_by_turn = {str(item.get("turn_id")): item for item in coding if item.get("turn_id")}
    bundle_by_call = {item.get("call"): item for item in bundles}

    turns = []
    for turn in metrics.get("turns", []):
        call = int(turn.get("call", len(turns)))
        bundle = bundle_by_call.get(call, {})
        message = bundle.get("message") if isinstance(bundle.get("message"), Mapping) else {}
        attachment = bundle.get("image_attachment")
        usage = turn.get("request_usage") if isinstance(turn.get("request_usage"), Mapping) else {}
        command_record = coding_by_turn.get(str(turn.get("turn_id")), {})
        message_bytes = turn.get("project_message_bytes")
        turns.append({
            "call": call,
            "message_kind": turn.get("message_kind"),
            "process_pid": session.get("process_pid"),
            "thread_id": session.get("thread_id"),
            "turn_id": turn.get("turn_id"),
            "provider": {
                "input_tokens": usage.get("input_tokens"),
                "cached_input_tokens": usage.get("cached_input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "reasoning_output_tokens": usage.get("reasoning_output_tokens"),
                "context_window_tokens": metrics.get("context", {}).get("model_context_window_tokens"),
                "request_input_utilization": turn.get("request_input_utilization"),
                "measurement": "provider_reported",
            },
            "harness_text": {
                "characters": turn.get("project_message_chars"),
                "utf8_bytes": message_bytes,
                "sha256": message.get("sha256"),
                "estimated_tokens": (
                    round(float(message_bytes) / 4.0, 2)
                    if isinstance(message_bytes, int) else None
                ),
                "token_status": "estimated_bytes_divided_by_four",
            },
            "automatic_image": (
                {
                    "path": attachment.get("path"),
                    "bytes": attachment.get("bytes"),
                    "sha256": attachment.get("sha256"),
                    "source": attachment.get("source"),
                    "maximum_size_px": attachment.get("derivation", {}).get("maximum_size_px"),
                }
                if isinstance(attachment, Mapping) else None
            ),
            **({"automatic_images": context_bundle_images(bundle),
                **{field: bundle[field] for field in ("post_action_observation", "current_visual_observation") if field in bundle}}
               if "image_attachments" in bundle else {}),
            "coding": {
                "cwd": command_record.get("cwd"),
                "commands": command_record.get("commands", []),
                "scratch": command_record.get("scratch", {}),
                "immutable_inputs_unchanged": command_record.get("immutable_inputs_unchanged"),
            },
            "latency_s": turn.get("latency_s"),
            "native_compaction_sequences": turn.get("native_compaction_sequences", []),
        })

    calls = session.get("calls") if isinstance(session.get("calls"), list) else []
    repeated_shell = []
    all_commands = []
    repeated_reads = []
    for item in coding:
        for command in item.get("commands", []):
            identity = str(command.get("command", ""))
            all_commands.append(identity)
            actions = command.get("command_actions", [])
            if isinstance(actions, list) and any(
                isinstance(action, Mapping) and action.get("type") == "read"
                for action in actions
            ):
                repeated_reads.append(identity)
    repeated_shell = _repetition(all_commands)

    read_only_tool_identities = []
    action_generation = 0
    seen_tool: Counter[tuple[int, str]] = Counter()
    repeated_tool_calls = []
    for event in events:
        if event.get("type") == "action_submitted":
            action_generation += 1
            continue
        if event.get("type") != "tool_result" or event.get("tool") == "robot.submit_action":
            continue
        identity = hashlib.sha256(json.dumps(
            {"tool": event.get("tool"), "arguments": event.get("arguments")},
            sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest()
        key = (action_generation, identity)
        seen_tool[key] += 1
        if seen_tool[key] > 1:
            repeated_tool_calls.append({
                "tool": event.get("tool"), "arguments": event.get("arguments"),
                "action_generation": action_generation,
            })
        read_only_tool_identities.append(identity)

    auto_hashes = [
        str(image.get("sha256", ""))
        for item in bundles for image in context_bundle_images(item)
    ]
    message_hashes = [
        str(item.get("message", {}).get("sha256", ""))
        for item in bundles if isinstance(item.get("message"), Mapping)
    ]
    input_tokens = metrics.get("workload", {}).get("input_tokens")
    cached_tokens = metrics.get("workload", {}).get("cached_input_tokens")
    cache_ratio = (
        cached_tokens / input_tokens
        if isinstance(input_tokens, int) and input_tokens > 0 and isinstance(cached_tokens, int)
        else None
    )
    later = [
        int(item["harness_text"]["utf8_bytes"])
        for item in turns[1:] if isinstance(item["harness_text"].get("utf8_bytes"), int)
    ]
    main_camera = str(plan.get("runtime", {}).get("main_camera", "head_camera"))
    auxiliary = [item for item in tool_images if item.get("camera") != main_camera]
    delivery_inventory = policy_delivery_inventory(plan)
    layers = _policy_layers(root, plan)
    tool_definitions = _load(root / "context-manifest.json").get(
        "tool_surface", {}
    ).get("advertised_in_order", [])
    tool_encoded = json.dumps(
        tool_definitions, sort_keys=True, separators=(",", ":")
    ).encode()
    observation_count = sum(event.get("type") == "observation" for event in events)
    model_facing_bundles = [
        item for item in bundles
        if item.get("message", {}).get("kind")
        in {"initial_user_message", "resume_user_message"}
    ]
    action_aware = any("post_action_observation" in item or "current_visual_observation" in item for item in bundles)
    required_image_counts = [
        len((item.get("current_visual_observation") or item["post_action_observation"])["images"]) + len(item.get("skill_observations", []))
        if isinstance(item.get("current_visual_observation") or item.get("post_action_observation"), Mapping)
        else int(bool(plan.get("policy", {}).get("auto_image", True))) if action_aware else 1
        for item in model_facing_bundles
    ]
    return {
        "schema": CONTEXT_AUDIT_SCHEMA,
        "run_id": result.get("run_id"),
        "plan_sha256": plan.get("plan_sha256"),
        "semantics": {
            "cached_tokens_are_not_redundant_new_input": True,
            "cumulative_input_is_provider_workload_not_context_occupancy": True,
            "harness_token_estimate_is_not_provider_tokenization": True,
        },
        "policy_layers": layers,
        "tool_schema": {
            "bytes": len(tool_encoded),
            "sha256": hashlib.sha256(tool_encoded).hexdigest(),
            "injection_count": 1,
            **delivery_inventory["tool_schema"],
        },
        "turns": turns,
        "session": metrics.get("session", {}),
        "provider_workload": metrics.get("workload", {}),
        "request_pressure": metrics.get("context", {}),
        "latency": metrics.get("latency", {}),
        "thresholds": {
            "later_message_median_lt_bytes": 2500,
            "later_message_max_lt_bytes": 4000,
            "observed_later_median_bytes": statistics.median(later) if later else None,
            "observed_later_max_bytes": max(later) if later else None,
            "passed": bool(later) and statistics.median(later) < 2500 and max(later) < 4000,
        },
        "camera_disclosure": {
            "automatic_policy_camera": main_camera,
            "automatic_images": len(auto_hashes),
            "fresh_observations_published": observation_count,
            "model_facing_fresh_observations": len(model_facing_bundles),
            "automatic_images_per_fresh_observation_required": None if action_aware else 1,
            **({"automatic_view_contract": "servo_defaults_plus_selected_public_wrists_v1",
                "required_image_counts_by_observation": required_image_counts}
               if action_aware else {}),
            "automatic_disclosure_passed": (
                len(auto_hashes) == sum(required_image_counts)
                and all(len(context_bundle_images(item)) == required
                        for item, required in zip(model_facing_bundles, required_image_counts, strict=True))
            ),
            "tool_images": tool_images,
            "auxiliary_camera_tool_images": auxiliary,
        },
        "coding": {
            "calls": coding,
            "command_count": len(all_commands),
            "scratch_artifact_generations": sum(
                len(item.get("scratch", {}).get("created", []))
                + len(item.get("scratch", {}).get("modified", []))
                for item in coding
            ),
            "integrity_passed": bool(coding) and all(
                item.get("immutable_inputs_unchanged") is True
                and not item.get("scratch", {}).get("reserved_paths_changed")
                and not item.get("outside_workspace_path_violations")
                for item in coding
            ),
        },
        "verifiable_redundancy": {
            "harness_repeated_exact_messages": _repetition(message_hashes),
            "repeated_automatic_images": _repetition(auto_hashes),
            "repeated_tool_images": _repetition([str(item["sha256"]) for item in tool_images]),
            "repeated_read_only_tool_calls_without_action": repeated_tool_calls,
            "repeated_shell_commands": repeated_shell,
            "repeated_shell_reads": _repetition(repeated_reads),
            "provider_cached_ratio": cache_ratio,
            "provider_cached_ratio_classification": "cache_reuse_not_redundant_new_input",
        },
        "source_artifacts": {
            relative: _file_ref(root, root / relative)
            for relative in (
                "plan.json", "result.json", "events.jsonl", "context-manifest.json",
                "native/agent-session.json", "native/coding-audit.jsonl",
                "native/app-server-protocol.jsonl",
            )
        },
    }


def write_context_audit(run_root: Path) -> dict[str, Any]:
    value = build_context_audit(run_root)
    target = run_root.resolve() / "context-audit.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    return value


def verify_context_audit(run_root: Path) -> dict[str, Any]:
    try:
        persisted = _load(run_root.resolve() / "context-audit.json", required=True)
        rebuilt = build_context_audit(run_root)
    except ContextAuditError as exc:
        return {"status": "invalid", "errors": [str(exc)], "audit": {}}
    errors = [] if persisted == rebuilt else ["context_audit_drift"]
    if persisted.get("schema") != CONTEXT_AUDIT_SCHEMA:
        errors.append("context_audit_schema_mismatch")
    return {"status": "valid" if not errors else "invalid", "errors": errors, "audit": persisted}
