"""Lossless projection cost stays bound to text plus newly admitted media.

The golden digests below were produced by the pre-fix implementation at
5d771e5c (every ``passes[*].pre_bytes`` and ``pre_lossless_request_utf8_bytes``
computed by serializing the full image-bearing history). They bind the complete
projected messages, the manifest after request assembly, the Provider wire
request, the Context Artifact index and every stored Artifact file.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from ora.agent.context.native import NativeContextPipeline
from ora.agent.contracts import PolicySessionError, SummaryResult
from ora.agent.evidence.trajectory import (
    ArtifactStore, TrajectoryError, canonical_bytes,
)
from ora.agent.memory.artifacts import ContextArtifactCatalog
from ora.agent.native.request_assembly import NativeRequestAssembler
from ora.agent.providers.native import OpenAICompatibleProvider

MANAGEMENT = {
    "schema": "ora.native-context-management.v4",
    "causal_exact_tail_action_steps": 3, "inline_media_tail_action_steps": 2,
    "eager_large_result_bytes": 50_000, "eager_preview_bytes": 2_048,
    "eager_exact_deduplication": True, "trigger_inline_images": 12,
    "trigger_estimated_tokens": 6_000, "trigger_provider_input_ratio": 0.70,
    "episode_checkpoint_trigger_tokens": 10_000,
    "episode_checkpoint_trigger_ratio": 0.2, "emergency_fallback_ratio": 0.80,
    "emergency_target_ratio": 0.65, "checkpoint_target_estimated_tokens": 1200,
    "checkpoint_max_estimated_tokens": 1600,
}
TOOLS = [{"type": "function", "function": {
    "name": name, "description": name, "parameters": {"type": "object"},
}} for name in ("move_pose", "observe", "grasp_plan")]
CHECKPOINT = {
    "task_progress": ["progress"], "confirmed_facts": ["fact"],
    "invalidated_beliefs": ["belief"],
    "failed_attempts": [{"failure": "f", "evidence": "e", "correction": "c"}],
    "current_constraints": ["constraint"], "do_not_repeat": ["repeat"],
    "re_ground_next": ["ground"],
}
GOLDEN_STEPS = 12
GOLDEN_FIRST_ROUND = 3
GOLDEN_IMAGE_BYTES = 3_000


class _Summary:
    def __init__(self) -> None:
        self.calls = 0

    def summarize(self, *, messages: Any, source_refs: Any) -> SummaryResult:
        self.calls += 1
        refs = copy.deepcopy(list(source_refs))
        return SummaryResult(
            content=json.dumps(CHECKPOINT), source_refs=refs,
            source_set_sha256=hashlib.sha256(canonical_bytes(refs)).hexdigest(),
            metadata={"call_id": f"summary-{self.calls}", "attempts": 1},
        )


def _image_bytes(label: str, size: int) -> bytes:
    seed = hashlib.sha256(label.encode()).digest()
    blocks = [
        hashlib.sha256(seed + index.to_bytes(8, "big")).digest()
        for index in range(size // 32 + 1)
    ]
    return b"".join(blocks)[:size]


def _url(data: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _pipeline(root: Path, *, window: int = 110_000) -> NativeContextPipeline:
    artifacts = ArtifactStore(root)
    return NativeContextPipeline(
        profile="native_causal_context_v1", artifacts=artifacts,
        context_window_tokens=window, max_output_tokens=2_048,
        retention="action_step_lossless_v4", summary_policy="rollout_checkpoint_v1",
        summary_provider=_Summary(), context_management=MANAGEMENT,
        context_artifacts=ContextArtifactCatalog(root, "policy-session-1", artifacts),
        mutation_tools=("move_pose",),
    )


def _protected() -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": "system " * 40, "_protected_context": True,
         "_context_source": "harness_system"},
        {"role": "user", "content": "embodiment " * 30, "_protected_context": True,
         "_context_source": "project_instructions"},
        {"role": "user", "content": "stack the blocks", "_protected_context": True,
         "_context_source": "episode_task"},
    ]


def _observation(step: int, session_round: int, size: int) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": json.dumps({
                "current_actual": {
                    "step": step, "joints": [0.1 * index for index in range(20)],
                    "note": "measured " * 20,
                },
                "observation_text": f"observation {step}",
            })},
            {"type": "image_url", "image_url": {"url": _url(_image_bytes(f"obs-{step}-a", size))}},
            {"type": "image_url", "image_url": {"url": _url(_image_bytes(f"obs-{step}-b", size))}},
        ],
        "_observation": {"step": step, "receipt": (
            {"event_id": f"receipt-{step}", "action_id": f"action-{step - 1}"}
            if step else None
        )},
        "_session_round": session_round,
        "_source_event_ids": [f"agent-event-{step * 10:06d}"],
    }


def _exchange(
    step: int, size: int, catalog: ContextArtifactCatalog, tool_images: int = 3,
) -> list[dict[str, Any]]:
    event = step * 10
    read_id, image_id, mutation_id = f"plan-{step}", f"view-{step}", f"move-{step}"
    datas = [_image_bytes(f"tool-{step}-{index}", size) for index in range(tool_images)]
    image_message: dict[str, Any] = {
        "role": "user",
        "content": [
            {"type": "text", "text": f"READ-ONLY tool feedback: {tool_images} image(s)."},
            *[{"type": "image_url", "image_url": {"url": _url(data)}} for data in datas],
        ],
        "_tool_image_step": step, "_tool_image_name": "observe",
        "_tool_image_call_id": image_id,
        "_source_event_ids": [f"agent-event-{event + 5:06d}"],
    }
    if step % 2 == 0:
        # Session-registered tool images take the resolve-and-compare path.
        image_message["_context_image_refs"] = [
            catalog.register_storage_ref(
                catalog.artifacts.put_bytes(
                    data, kind="native_tool_image", suffix=".png",
                    media_type="image/png",
                ),
                kind="context_image", round_index=None, action_step=step,
            )
            for data in datas
        ]
    return [
        {
            "role": "assistant", "content": f"decision {step}",
            "reasoning_content": "think " * 40,
            "tool_calls": [{"id": read_id, "type": "function",
                            "function": {"name": "grasp_plan", "arguments": "{}"}}],
            "_source_event_ids": [f"agent-event-{event + 1:06d}"],
        },
        {
            "role": "tool", "tool_call_id": read_id, "name": "grasp_plan",
            "content": json.dumps({
                "status": "actual",
                "candidates": [{"index": index, "pose": [0.01 * index] * 7}
                               for index in range(6)],
                "diagnostic": "L" * 60_000 if step % 4 == 1 else "same read-only payload",
            }),
            "_source_event_ids": [f"agent-event-{event + 2:06d}"],
        },
        {
            "role": "assistant", "content": f"look {step}",
            "tool_calls": [{"id": image_id, "type": "function",
                            "function": {"name": "observe", "arguments": "{}"}}],
            "_source_event_ids": [f"agent-event-{event + 3:06d}"],
        },
        {
            "role": "tool", "tool_call_id": image_id, "name": "observe",
            "content": json.dumps({"status": "actual", "images": tool_images}),
            "_source_event_ids": [f"agent-event-{event + 4:06d}"],
        },
        image_message,
        {
            "role": "assistant", "content": f"act {step}",
            "tool_calls": [{"id": mutation_id, "type": "function", "function": {
                "name": "move_pose", "arguments": json.dumps({"step": step}),
            }}],
            "_source_event_ids": [f"agent-event-{event + 6:06d}"],
        },
        {
            "role": "tool", "tool_call_id": mutation_id, "name": "move_pose",
            "content": json.dumps({"status": "submitted", "step": step}),
            "_source_event_ids": [f"agent-event-{event + 7:06d}"],
        },
    ]


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _history(
    pipeline: NativeContextPipeline, *, steps: int, first_round: int, size: int,
    project: Any = None,
) -> list[dict[str, Any]]:
    """Append-shaped two-round transcript; ``project`` sees every live call."""

    messages: list[dict[str, Any]] = _protected()
    for step in range(steps):
        if step == first_round:
            messages.append({
                "role": "user",
                "content": '<official_round_result>{"passed":false,"round":1}'
                           "</official_round_result>",
                "_protected_context": True,
                "_context_source": "official_round_result_1",
                "_source_event_ids": ["agent-event-900000"],
            })
        messages.append(_observation(step, 1 if step < first_round else 2, size))
        if project is not None:
            project(messages, step)
        exchange = _exchange(step, size, pipeline.context_artifacts)
        # Tool-triggered projection: the read-only result is paired, the
        # mutation is not yet proposed.
        messages.extend(exchange[:5])
        if project is not None:
            project(messages, step)
        messages.extend(exchange[5:])
    return messages


def _drive(root: Path) -> dict[str, Any]:
    pipeline = _pipeline(root)
    assembler = NativeRequestAssembler(
        model="deepseek-flash", reasoning_effort="low", max_output_tokens=2_048,
        stream=True, provider=OpenAICompatibleProvider(
            base_url="https://provider.invalid", api_key="test", client=object(),
        ),
    )
    projections: list[str] = []
    kinds: set[str] = set()

    def project(messages: list[dict[str, Any]], step: int) -> None:
        projected = pipeline.project(messages=messages, tools=TOOLS, observation_step=step)
        prepared = assembler.prepare(projected)
        manifest = projected["manifest"]
        kinds.update(item["kind"] for item in manifest["projection_transitions"])
        projections.append(_digest({
            "messages": projected["messages"], "tools": projected["tools"],
            "manifest": manifest, "wire": prepared.wire,
        }))
        pipeline.observe_provider_usage(
            estimated_tokens=manifest["pressure_estimated_input_tokens"],
            input_tokens=manifest["pressure_estimated_input_tokens"] + 17,
            projection_epoch=manifest["projection_epoch"],
            provider_wire_request_sha256=hashlib.sha256(
                canonical_bytes(prepared.wire)
            ).hexdigest(),
        )

    _history(
        pipeline, steps=GOLDEN_STEPS, first_round=GOLDEN_FIRST_ROUND,
        size=GOLDEN_IMAGE_BYTES, project=project,
    )
    files = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*")) if path.is_file()
    }
    return {
        "projections": projections, "files_sha256": _digest(files),
        "transition_kinds": sorted(kinds),
        "episode_checkpoints": pipeline.episode_checkpoint_count,
    }


# Namespace migration audit: request contents differ only in schema names,
# their content hashes and derived byte/token counts. Projection transitions
# and checkpoint choices match the pre-migration recording.
GOLDEN: dict[str, Any] = {
    "episode_checkpoints": 0,
    "files_sha256": "152dec12457d818d9b3ac81889828c1f637c5f3a7b83df06f62ed866714134d6",
    "projections": [
        "a2a90f24100cf07ab2144d4bb47f110d4852f1ec9d04a8fb5554127107d57a9c",
        "0cf0f8f5b40ed6dc1a947244097875e6850dbbfce743a134207f54b1210a4dbb",
        "a501fcb8c608589d3afb1902e9b0c429b1f1f6767710c6966065f5cf2bc613f4",
        "e7880941d160f22ff5ca12c60dd4317df9f5685357b2940aaf0518242fa99932",
        "03a573d809c63d6b8c77e74e66a956096071f76d63b2c76703dfcfc8606fd4b5",
        "923c33f9627362235dc6d6a5215b540d1eac37c0f54b535f822ed734452c7644",
        "64c824dab8b31ea916fe3d4e206856b72087a3b323ffc5b8edd386aec7081de4",
        "e18a78f8aeb2cf9eb543f5619f44d6aaffc04116c84c7e5d6cc4dae916af4669",
        "72410322602550f3726616c3b7cd9040482225ed7a9eef7c39678ee0e8d93bfc",
        "5ec71974ae2cb6ccb40d2553c112703d7d1210acf88b130fff292a0921bd23e6",
        "3f279e6bff43b1faf3964e8acd14a023050b4813cce382bb89ab7b9b8e507da6",
        "02cc9e7924e10663aae153223154b3db31450684f2c0694eb9210bb48e2f3a32",
        "c5442922c6a9333b068c3c3b8ef00cda8362a0102dcd04f8cf2129b361eb5622",
        "06b1593f663df25a23bff28d81269abb6bef0f81a52d29f4a6350b5576780a73",
        "aba93600ac4aeafabc9f47f7367a478388bfc4213e45b96909245d412b85d32e",
        "b9588bdb954353590dbdfd3ef0655359fabe934848561e3433b63486a6a8d145",
        "1f4d1c3ad0962a74a1c0c568d564fbc74a7c6d82cc2ee2f12cc0998da89dda8f",
        "7a00244edf066e2b0216c6b8bd1608389d48b5aedec1829b880b21945ed60a63",
        "e5856c0296885a62c6c5e5e4106884f567e3b8f3b58e06ef66a90e80dc32ecea",
        "de9d07c17e9bf4e376889247e3a0977b3d62fd10a2f611340b76d2df59e53952",
        "7e3382c180486b429a0060dd330549101da6315f05fa1f6cf3fceb3d268613a7",
        "a8921a0e1c42929a83167340c53754609be403a69b8cd5d5b9eeb321b7d2f918",
        "ff72f09c5a324de1a2e642815b1160969301bcc64660cbb646defe5d4f4bfd55",
        "f762a61f5612a434ee9780a134eaaf0fbb3b515c212622388a9ffeba0504575f"
    ],
    "transition_kinds": [
        "emergency_compaction",
        "lifecycle_media_externalization",
        "lifecycle_read_only_externalization",
        "pressure_externalization"
    ]
}


def test_projection_matches_the_audited_namespace_migration_snapshot(tmp_path: Path) -> None:
    assert _drive(tmp_path / "run") == GOLDEN


class _Counters:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from ora.agent.context import inline_media, native, projection, request_size

        self.serialized: list[int] = []
        self.decodes = 0
        self.store_reads = 0

        def counting_canonical(value: Any) -> bytes:
            encoded = canonical_bytes(value)
            self.serialized.append(len(encoded))
            return encoded

        for module in (native, projection, request_size):
            monkeypatch.setattr(module, "canonical_bytes", counting_canonical)
        decode = inline_media._decode_image_data_url

        def counting_decode(value: Any) -> Any:
            self.decodes += 1
            return decode(value)

        monkeypatch.setattr(inline_media, "_decode_image_data_url", counting_decode)
        put, resolve = ArtifactStore.put_bytes, ContextArtifactCatalog.resolve

        def counting_put(store: ArtifactStore, data: bytes, **kwargs: Any) -> Any:
            self.store_reads += 1
            return put(store, data, **kwargs)

        def counting_resolve(catalog: ContextArtifactCatalog, ref: str) -> Any:
            self.store_reads += 1
            return resolve(catalog, ref)

        monkeypatch.setattr(ArtifactStore, "put_bytes", counting_put)
        monkeypatch.setattr(ContextArtifactCatalog, "resolve", counting_resolve)

    def reset(self) -> None:
        self.serialized, self.decodes, self.store_reads = [], 0, 0


def test_repeat_projection_work_excludes_historical_media(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _pipeline(tmp_path / "run", window=1_048_576)
    messages = _history(pipeline, steps=24, first_round=24, size=20_000)
    messages.append(_observation(24, 1, 20_000))
    first = pipeline.project(messages=messages, tools=TOOLS, observation_step=24)
    counters = _Counters(monkeypatch)
    second = pipeline.project(messages=messages, tools=TOOLS, observation_step=24)
    assert second["messages"] == first["messages"]
    manifest = second["manifest"]
    pre, post = (
        manifest["pre_lossless_request_utf8_bytes"],
        manifest["post_lossless_request_utf8_bytes"],
    )
    assert pre > 5 * post  # the history is dominated by historical media
    # Nothing is decoded, re-read or re-hashed from the Artifact store again.
    assert counters.decodes == 0
    assert counters.store_reads == 0
    # No serialization carries historical media: each is bounded by the
    # materialized history's text or by the post-lossless request (which holds
    # the active images). The pre-fix implementation serialized the whole
    # image-bearing history (``pre``) about twenty times per projection.
    media = sum(
        len(block["image_url"]["url"])
        for message in messages if isinstance(message.get("content"), list)
        for block in message["content"] if block.get("type") == "image_url"
    )
    assert media > pre // 2
    materialized = manifest["passes"][0]["post_bytes"]  # adds recoverable originals
    assert max(counters.serialized) <= max(post, materialized - media)
    assert sum(counters.serialized) < 4 * pre

    counters.reset()
    exchange = _exchange(24, 20_000, pipeline.context_artifacts)
    messages.extend(exchange)
    messages.append(_observation(25, 1, 20_000))
    pipeline.project(messages=messages, tools=TOOLS, observation_step=25)
    # Only the five images first seen in this projection are admitted: two
    # observation images plus three tool images; step 24 registers its tool
    # images, which still take one resolve-and-compare admission each.
    assert counters.decodes == 5
    assert max(counters.serialized) <= pre


def _legacy_ids(value: Any) -> list[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        raw = value.get("_source_event_ids")
        if isinstance(raw, list):
            found.update(item for item in raw if isinstance(item, str))
        for child in value.values():
            found.update(_legacy_ids(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_legacy_ids(child))
    return sorted(found)


def test_canonical_sizer_matches_serialization_exactly() -> None:
    from ora.agent.context.request_size import CanonicalSizer

    data_url = _url(_image_bytes("sizer", 5_000))
    odd_url = "data:text/plain," + 'quote " back \\ tab \t é 中  ' * 200
    value = [
        {"role": "user", "_source_event_ids": ["a", 3, "b"], "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": odd_url},
            {"nested": ({"_source_event_ids": ["hidden-in-tuple"]}, data_url)},
        ]},
        {"f": 2.5, "k": None, "b": True, "same": data_url, "short": "data:x"},
        {"_source_event_ids": "not-a-list", "child": [{"_source_event_ids": ["c"]}]},
    ]
    sizer = CanonicalSizer()
    size, ids = sizer.measure(value)
    assert size == len(canonical_bytes(value))
    assert sorted(ids) == _legacy_ids(value)
    assert "hidden-in-tuple" not in ids
    sizer.rotate()
    assert sizer.measure(value)[0] == size
    with pytest.raises(ValueError):
        sizer.measure({"nan": float("nan"), "image": data_url})


def test_admission_cache_revalidates_changed_artifact_files(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path / "run", window=1_048_576)
    messages = _history(pipeline, steps=3, first_round=3, size=2_000)
    messages.append(_observation(3, 1, 2_000))
    pipeline.project(messages=messages, tools=TOOLS, observation_step=3)
    registered = messages[-4]["_context_image_refs"][0]  # step-2 session tool image
    entry, _data = pipeline.context_artifacts.resolve(registered["ref"])
    stored = pipeline.context_artifacts.run_root / entry["storage_ref"]["path"]
    stored.write_bytes(b"tampered")
    with pytest.raises(PolicySessionError, match="cannot be resolved"):
        pipeline.project(messages=messages, tools=TOOLS, observation_step=3)

    fresh = _pipeline(tmp_path / "second", window=1_048_576)
    messages = _history(fresh, steps=2, first_round=2, size=2_000)
    messages.append(_observation(2, 1, 2_000))
    fresh.project(messages=messages, tools=TOOLS, observation_step=2)
    observation_data = _image_bytes("obs-2-a", 2_000)
    target = fresh.artifacts.root / (
        hashlib.sha256(observation_data).hexdigest() + ".png"
    )
    target.write_bytes(b"tampered")
    with pytest.raises(TrajectoryError, match="hash collision"):
        fresh.project(messages=messages, tools=TOOLS, observation_step=2)
