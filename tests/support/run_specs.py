"""Minimal Native RunSpecs shared by the composition, orchestrator and backend tests."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def native_spec(environment: str) -> dict:
    instruction = "Move the selected object to the destination."
    runtime = {"kind": environment}
    if environment == "libero_pro":
        runtime.update({
            "suite": "libero_spatial", "task_id": 0,
            "expected_task_language": instruction,
            "motion_planner": {"device": "cuda:0"},
        })
    return {
        "schema": "ora.run-spec.v1",
        "experiment": "native-baseline-fixture", "config": environment,
        "milestone": "NATIVE", "task": "fixture-task",
        "task_instruction": instruction, "seed": 0,
        "evidence_tier": "diagnostic",
        "output_workspace": "outputs/native-baseline-fixture",
        "policy": {
            "kind": "native_api", "provider": "openai_compatible",
            "model": "test-model", "endpoint": "https://example.invalid/v1",
            "supports_images": True, "context_window_tokens": 262144,
            "max_output_tokens": 8192,
        },
        "runtime": runtime,
        "budgets": {
            "max_robot_decisions": 80, "max_model_calls": 200,
            "max_sim_steps": 10000, "wall_time_s": 600,
        },
    }


def robodojo_native_spec(task: str = "stack_bowls") -> dict:
    """A fresh RoboDojo Native RunSpec: only the per-run choices, no surface.

    The task, its reviewed layout (the registry record's), its native-action
    limit, its physics device, the GPU and the qualification binding stay the
    RunSpec's; everything else comes from the direct_v4 RoboDojo instance.
    """
    from ora.agent.composition.robodojo_task_contract import REGISTRY_PATH, load_task_contract
    from ora.environment.benchmarks.robodojo import NATIVE_ACTION_LIMITS
    from ora.environment.benchmarks.robodojo_physics import physics_device_for

    entry = json.loads((ROOT / REGISTRY_PATH).read_text())["tasks"][task]
    record = load_task_contract(task, bindings={task: entry}, project_root=ROOT)["record"]
    layout, instruction = record["layout"], record["instruction"]["text"]
    spec = native_spec("robotwin")
    spec.update(task=task, task_instruction=instruction)
    spec["runtime"] = {
        "kind": "robodojo", "task_variant": task, "eval_seed": layout["eval_seed"],
        "layout_id": layout["ordinal"], "layout_file": layout["layout_file"],
        "layout_sha256": layout["layout_sha256"], "native_action_limit": NATIVE_ACTION_LIMITS[task],
        "physics_device": physics_device_for(task), "renderer_gpu": 0, "cuda_visible_devices": "0",
        "expected_task_language": instruction,
        "infrastructure_qualification": {"path": "robodojo-compat/qualification/fixture.json",
                                         "sha256": "0" * 64},
    }
    return spec
