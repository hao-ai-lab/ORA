"""Host-safe identities for the pinned RoboTwin 2.0 randomized evaluation."""

from __future__ import annotations

from typing import Any, Mapping


TASKS = (
    "adjust_bottle", "beat_block_hammer", "blocks_ranking_rgb", "blocks_ranking_size",
    "click_alarmclock", "click_bell", "dump_bin_bigbin", "grab_roller",
    "handover_block", "handover_mic", "hanging_mug", "lift_pot", "move_can_pot",
    "move_pillbottle_pad", "move_playingcard_away", "move_stapler_pad", "open_laptop",
    "open_microwave", "pick_diverse_bottles", "pick_dual_bottles", "place_a2b_left",
    "place_a2b_right", "place_bread_basket", "place_bread_skillet", "place_burger_fries",
    "place_can_basket", "place_cans_plasticbox", "place_container_plate", "place_dual_shoes",
    "place_empty_cup", "place_fan", "place_mouse_pad", "place_object_basket",
    "place_object_scale", "place_object_stand", "place_phone_stand", "place_shoe",
    "press_stapler", "put_bottles_dustbin", "put_object_cabinet", "rotate_qrcode",
    "scan_object", "shake_bottle", "shake_bottle_horizontally", "stack_blocks_three",
    "stack_blocks_two", "stack_bowls_three", "stack_bowls_two", "stamp_seal", "turn_switch",
)

# Exact task_config/demo_randomized.yml values, not invented ranges. The pinned
# _base_task.py consumes table height as U(-0.03, 0), NOT U(-0.03, +0.03).
RANDOMIZATION = {
    "random_background": True,
    "cluttered_table": True,
    "clean_background_rate": 0.02,
    "random_head_camera_dis": 0,
    "random_table_height": 0.03,
    "random_light": True,
    "crazy_random_light_rate": 0.02,
}


#: task_config/demo_clean.yml: every randomization off; the leaderboard's Easy.
CLEAN_RANDOMIZATION = {
    "random_background": False,
    "cluttered_table": False,
    "clean_background_rate": 1,
    "random_head_camera_dis": 0,
    "random_table_height": 0,
    "random_light": False,
    "crazy_random_light_rate": 0,
}
#: The two official evaluation scene configurations and their randomization.
TASK_CONFIGS = {"demo_clean": CLEAN_RANDOMIZATION, "demo_randomized": RANDOMIZATION}


def validate_task_config(name: str, config: Mapping[str, Any]) -> None:
    """Reject scene-setting drift without importing a simulator."""
    if name not in TASK_CONFIGS:
        raise ValueError("task_config must be demo_clean or demo_randomized")
    if config.get("domain_randomization") != TASK_CONFIGS[name]:
        raise ValueError(f"RoboTwin official {name} domain_randomization settings differ")
    if config.get("embodiment") != ["aloha-agilex"]:
        raise ValueError("RoboTwin evaluation requires the aloha-agilex embodiment")


def validate_randomized_config(config: Mapping[str, Any]) -> None:
    """Reject scene-setting drift without importing a simulator."""
    validate_task_config("demo_randomized", config)


def evaluation_protocol(*, outer_seed: int = 0, task_config: str = "demo_randomized") -> dict[str, Any]:
    """Describe script/eval_policy.py's scene selection, not a policy RunPlan.

    The official loop is the same under both scene configurations; they differ
    in the randomization the scene applies, so each has its own accepted-seed
    list.
    """
    if isinstance(outer_seed, bool) or not isinstance(outer_seed, int) or outer_seed < 0:
        raise ValueError("outer_seed must be a non-negative integer")
    if task_config not in TASK_CONFIGS:
        raise ValueError("task_config must be demo_clean or demo_randomized")
    return {
        "schema": "ora.robotwin-evaluation-protocol.v1",
        "task_config": task_config,
        "eval_mode": True,
        "is_test": True,
        "background_split": "unseen",
        "domain_randomization": dict(TASK_CONFIGS[task_config]),
        "outer_seed": outer_seed,
        "start_requested_seed": 100000 * (1 + outer_seed),
        "accepted_trials_per_task": 100,
        "seed_selection": "ascending requested seed; expert plan_success AND post-play check_success; reset accepted seed for policy",
        "expert_rejection_counts_as_policy_trial": False,
        "infrastructure_failure_is_expert_rejection": False,
        "policy_reset_per_trial": True,
        "official_success": "upstream eval_success/check_success",
        "task_count": len(TASKS),
    }
