"""Context management v6: the absolute L5 watermark (user decision 2026-10-04).

"所有的 L5 改成 15 万": every maintained configuration admits an L5 Episode
Capsule at 150,000 Provider-input tokens, whatever the bound window. v5's
max(150,000, 0.60 x usable) put it at 618,086 on a 1M DeepSeek window, where
it effectively never fired. L4 and L6 are unchanged. v4/v5 keep their recorded
window-scaled watermark so sealed runs and frozen archives read back exactly.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ora.agent.composition.baseline import BASELINE_CATALOGS
from ora.agent.context import thresholds
from ora.agent.context.management import (
    CONTEXT_MANAGEMENT_V5_FIELD_SPECS,
    CONTEXT_MANAGEMENT_V6_FIELD_SPECS,
    CONTEXT_MANAGEMENT_V6_SCHEMA,
    ContextManagementConfigError,
    validate_context_management,
)
from ora.agent.context.native import NativeContextPipeline
from ora.agent.evidence.trajectory import ArtifactStore
from ora.agent.memory.artifacts import ContextArtifactCatalog
from ora.orchestrator.context_coverage import coverage_violations
from test_lossless_context_runtime import (
    CHECKPOINT, TOOLS, Summary, _messages, _project_with_input_usage,
)

ROOT = Path(__file__).resolve().parents[1]
ACTIVATION = json.loads((ROOT / "tests/data/default-expectations.json").read_text())
#: The maintained block, as fresh configurations receive it.
V6 = ACTIVATION["native_policy"]["context_management"]
#: The same figures under the v5 schema this change replaced.
V5 = {
    **V6, "schema": "ora.native-context-management.v5",
    "episode_checkpoint_trigger_ratio": 0.60,
}
ONE_MILLION = 1_048_576
#: DeepSeek's usable input with the default 16,384-token output budget.
ONE_MILLION_USABLE = ONE_MILLION - 16_384 - 2_048


def _bound(root: Path, management, *, context_window: int, summary=None):
    """A lossless pipeline bound like a maintained 16K-output Provider."""
    artifacts = ArtifactStore(root)
    return NativeContextPipeline(
        profile="native_causal_context_v1", artifacts=artifacts,
        context_window_tokens=context_window, max_output_tokens=16_384,
        retention="action_step_lossless_v4",
        summary_policy="rollout_checkpoint_v1",
        summary_provider=summary or Summary(CHECKPOINT),
        context_management=management,
        context_artifacts=ContextArtifactCatalog(root, "policy-session-1", artifacts),
        mutation_tools=("move_pose", "close_gripper", "open_gripper"),
    )


def test_every_maintained_configuration_selects_the_absolute_watermark():
    catalogs = {
        baseline_id: json.loads((ROOT / path).read_text())["policy"]["context_management"]
        for baseline_id, path in BASELINE_CATALOGS.items()
    }
    for owner, management in {"default-activation": V6, **catalogs}.items():
        assert management["schema"] == CONTEXT_MANAGEMENT_V6_SCHEMA, owner
        assert management["episode_checkpoint_trigger_tokens"] == 150_000, owner
        assert "episode_checkpoint_trigger_ratio" not in management, owner
        assert validate_context_management(management) == management, owner


def test_l5_is_150k_not_618k_on_a_1m_window(tmp_path):
    pipeline = _bound(tmp_path, V6, context_window=ONE_MILLION)
    assert pipeline.hard_input_tokens == ONE_MILLION_USABLE == 1_030_144
    assert pipeline.episode_checkpoint_trigger() == 150_000
    manifest = pipeline.project(messages=_messages(0), tools=TOOLS, observation_step=0)["manifest"]
    assert manifest["resolved_episode_checkpoint_trigger_tokens"] == 150_000
    assert manifest["resolved_episode_checkpoint_trigger_formula"] == "absolute_tokens"
    assert coverage_violations(manifest, manifest["action_step_inventory"]) == []
    # The replaced v5 watermark on the same binding.
    assert thresholds.episode_checkpoint_trigger(V5, ONE_MILLION_USABLE) == 618_086


def test_l5_admits_a_closed_episode_at_150k_on_a_1m_window(tmp_path):
    pipeline = _bound(tmp_path, V6, context_window=ONE_MILLION)
    below = _project_with_input_usage(pipeline, _messages(5), 5, 149_999)
    admission = below["manifest"]["episode_checkpoint_admission"]
    assert admission["reason"] == "below_high_watermark"
    assert admission["episode_high_watermark_tokens"] == 150_000
    assert below["manifest"]["episode_checkpoint_count"] == 0
    admitted = _project_with_input_usage(pipeline, _messages(5), 5, 150_000)
    assert admitted["manifest"]["episode_checkpoint_count"] == 1
    capsule = next(item for item in admitted["manifest"]["projection_transitions"]
                   if item["kind"] == "episode_capsule")
    assert capsule["semantic_high_watermark_tokens"] == 150_000


def test_l5_is_150k_on_a_200k_window_where_v5_agreed(tmp_path):
    pipeline = _bound(tmp_path, V6, context_window=200_000)
    assert pipeline.hard_input_tokens == 181_568
    assert pipeline.episode_checkpoint_trigger() == 150_000
    assert thresholds.episode_checkpoint_trigger(V5, 181_568) == 150_000


@pytest.mark.parametrize("window", [200_000, 262_144, ONE_MILLION])
def test_l4_and_l6_are_unchanged(tmp_path, window):
    v6 = _bound(tmp_path / "v6", V6, context_window=window)
    v5 = _bound(tmp_path / "v5", V5, context_window=window)
    assert V6["emergency_fallback_ratio"] == 0.80 and V6["emergency_target_ratio"] == 0.65
    assert v6.estimate_pressure_trigger() == v5.estimate_pressure_trigger()
    emergency = v6.project(messages=_messages(0), tools=TOOLS, observation_step=0)[
        "manifest"]["emergency_compaction_admission"]
    assert emergency["emergency_high_watermark_tokens"] == int(v6.hard_input_tokens * 0.80)


def test_token_order_of_l4_and_l5_holds_only_below_about_620k():
    """Decided, not guaranteed: above ~620K usable L4's token trigger passes L5."""
    share = thresholds.ESTIMATE_PRESSURE_WINDOW_SHARE
    assert thresholds.estimate_pressure_trigger(V6, 618_000) < 150_000
    assert thresholds.estimate_pressure_trigger(V6, ONE_MILLION_USABLE) == int(
        ONE_MILLION_USABLE * share) == 249_792
    assert thresholds.episode_checkpoint_trigger(V6, ONE_MILLION_USABLE) == 150_000


def test_v6_is_v5_without_the_l5_ratio():
    assert set(CONTEXT_MANAGEMENT_V5_FIELD_SPECS) - set(CONTEXT_MANAGEMENT_V6_FIELD_SPECS) == {
        "episode_checkpoint_trigger_ratio"}
    assert set(CONTEXT_MANAGEMENT_V6_FIELD_SPECS) < set(CONTEXT_MANAGEMENT_V5_FIELD_SPECS)
    with pytest.raises(ContextManagementConfigError, match="extra=\\['episode_checkpoint_trigger_ratio'\\]"):
        validate_context_management({**V6, "episode_checkpoint_trigger_ratio": 0.6})
    stride_free = {key: value for key, value in V6.items()
                   if key != "tail_aging_stride_action_steps"}
    with pytest.raises(ContextManagementConfigError, match="missing=\\['tail_aging_stride_action_steps'\\]"):
        validate_context_management(stride_free)
    with pytest.raises(ContextManagementConfigError, match="L4 estimate floor"):
        validate_context_management({**V6, "episode_checkpoint_trigger_tokens": 48_000})
    with pytest.raises(ContextManagementConfigError, match="must be positive"):
        validate_context_management({**V6, "episode_checkpoint_trigger_tokens": 0})
    with pytest.raises(ContextManagementConfigError, match="emergency target"):
        validate_context_management({**V6, "emergency_target_ratio": 0.75})


def test_recorded_v5_evidence_keeps_its_window_scaled_watermark(tmp_path):
    """HEAD still composes and revalidates v5 exactly; only maintained configs moved."""
    assert validate_context_management(V5)["schema"].endswith(".v5")
    pipeline = _bound(tmp_path, V5, context_window=ONE_MILLION)
    manifest = pipeline.project(messages=_messages(0), tools=TOOLS, observation_step=0)["manifest"]
    assert manifest["resolved_episode_checkpoint_trigger_tokens"] == 618_086
    assert manifest["resolved_episode_checkpoint_trigger_formula"] == (
        "max(absolute_tokens, int(hard_input_tokens * ratio))")
    inventory = manifest["action_step_inventory"]
    assert coverage_violations(manifest, inventory) == []
    # Evidence cannot be read back under the other schema's watermark.
    retimed = {**manifest, "resolved_episode_checkpoint_trigger_tokens": 150_000}
    assert "lossless_context_resolved_threshold_invalid" in coverage_violations(retimed, inventory)
    relabelled = {**manifest, "context_management": {
        key: value for key, value in V6.items()}}
    assert "lossless_context_resolved_threshold_invalid" in coverage_violations(relabelled, inventory)
