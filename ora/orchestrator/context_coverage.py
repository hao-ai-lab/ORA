"""Source-version-aware validation of lossless Context coverage evidence."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from ora.agent.context.thresholds import episode_checkpoint_trigger


def coverage_violations(
    projection: Mapping[str, Any], inventory: Sequence[Mapping[str, Any]],
) -> list[str]:
    reasons: list[str] = []
    completed = sum(item.get("complete") is True for item in inventory)
    management = projection.get("context_management", {})
    if not isinstance(management, Mapping):
        return ["lossless_context_lifecycle_coverage_invalid"]
    lifecycle = projection.get("lifecycle_media_externalized_through_action_step")
    tail = management.get("inline_media_tail_action_steps")
    if "resolved_near_tail_action_steps" not in projection:
        # Frozen source evidence retains its original fixed-tail semantics.
        current_round = inventory[-1].get("session_round")
        current_completed = sum(item.get("complete") is True and item.get(
            "session_round"
        ) == current_round for item in inventory)
        count = completed - min(tail, current_completed) if type(tail) is int else 0
        if lifecycle != (count - 1 if count > 0 else None):
            reasons.append("lossless_context_lifecycle_coverage_invalid")
        return reasons
    causal = management.get("causal_exact_tail_action_steps")
    target = management.get("trigger_inline_images")
    if type(tail) is not int or type(causal) is not int or type(target) is not int:
        return ["lossless_context_near_window_invalid"]
    near = max(tail, causal)
    eligible = max(0, completed - near)
    media_policy = projection.get("resolved_media_retention_policy")
    if projection.get("resolved_near_tail_action_steps") != near or projection.get(
        "resolved_inline_image_target"
    ) != target:
        reasons.append("lossless_context_near_window_invalid")
    if media_policy == "current_round_action_step_tail_v1":
        current_round = inventory[-1].get("session_round")
        current_completed = sum(
            item.get("complete") is True
            and item.get("session_round") == current_round
            for item in inventory
        )
        media_coverage = completed - min(tail, current_completed)
        expected_lifecycle = media_coverage - 1 if media_coverage else None
        if lifecycle != expected_lifecycle or projection.get(
            "protected_inline_media_action_steps"
        ) != [
            item["action_step_index"] for item in inventory
            if item["action_step_index"] >= media_coverage
        ]:
            reasons.append("lossless_context_lifecycle_coverage_invalid")
    elif media_policy == "current_round_action_step_tail_stride_v1":
        if not _stride_media_coverage_admitted(management, lifecycle, inventory) or (
            projection.get("protected_inline_media_action_steps") != [
                item["action_step_index"] for item in inventory
                if item["action_step_index"] >= (
                    lifecycle + 1 if type(lifecycle) is int else 0
                )
            ]
        ):
            reasons.append("lossless_context_lifecycle_coverage_invalid")
    elif media_policy is None:
        # Target-based v4 evidence emitted before fixed Episode media tails
        # were restored. Keep its exact source semantics revalidatable.
        if projection.get("protected_inline_media_action_steps") != [
            item["action_step_index"] for item in inventory
            if item["action_step_index"] >= eligible
        ]:
            reasons.append("lossless_context_near_window_invalid")
        if lifecycle is not None and (
            type(lifecycle) is not int or not 0 <= lifecycle < eligible
        ):
            reasons.append("lossless_context_lifecycle_coverage_invalid")
    else:
        reasons.append("lossless_context_lifecycle_coverage_invalid")
    pressure = projection.get("pressure_externalized_through_action_step")
    if pressure is not None and (
        type(pressure) is not int or not 0 <= pressure < eligible
    ):
        reasons.append("lossless_context_lifecycle_coverage_invalid")
    frontier = max((projection.get(field) if type(projection.get(field)) is int else -1)
                   for field in ("lifecycle_media_externalized_through_action_step",
                                 "episode_checkpointed_through_action_step",
                                 "emergency_summarized_through_action_step"))
    inline = sum(item.get("inline_images", 0) for item in inventory
                 if item["action_step_index"] > frontier)
    target_violation = (
        media_policy is None and inline > target and frontier + 1 < eligible
    )
    if projection.get("active_inline_images") != inline or target_violation:
        reasons.append("lossless_context_image_target_invalid")
    hard = projection.get("hard_input_tokens")
    if type(hard) is not int or projection.get(
        "resolved_episode_checkpoint_trigger_tokens"
    ) != episode_checkpoint_trigger(management, hard):
        reasons.append("lossless_context_resolved_threshold_invalid")
    return reasons


def _stride_media_coverage_admitted(
    management: Mapping[str, Any], lifecycle: Any,
    inventory: Sequence[Mapping[str, Any]],
) -> bool:
    """Context v5 media may lag its minimum tail by less than one stride.

    The lag only ever keeps current-Episode media inline: a reset releases
    every prior-Episode image in the same projection, as v4 does.
    """

    stride = management.get("tail_aging_stride_action_steps")
    tail = management.get("inline_media_tail_action_steps")
    if type(stride) is not int or stride < 1 or type(tail) is not int:
        return False
    if lifecycle is not None and (type(lifecycle) is not int or lifecycle < 0):
        return False
    completed = sum(item.get("complete") is True for item in inventory)
    current_round = inventory[-1].get("session_round")
    current_completed = sum(
        item.get("complete") is True and item.get("session_round") == current_round
        for item in inventory
    )
    covered = lifecycle + 1 if lifecycle is not None else 0
    lag = completed - min(tail, current_completed) - covered
    return 0 <= lag < stride and (
        covered >= completed
        or inventory[covered].get("session_round") == current_round
    )
