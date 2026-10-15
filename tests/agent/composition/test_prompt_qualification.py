from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from ora.orchestrator.spec import resolve_run_spec
from ora.agent.composition import materialize_policy
from ora.agent.composition.environment_context import (
    EnvironmentContextError, resolve_environment_prompt_profile,
)
from ora.agent.composition.tool_surface import resolve_tool_surface
from ora.agent.tools.robot import RobotToolRuntime
from tests.support.native_synthetic_spec import synthetic_spec


ROOT = Path(__file__).resolve().parents[3]


def test_environment_metadata_selects_embodiment_profile_and_rejects_drift() -> None:
    metadata = {
        "schema": "ora.public-environment-context.v1",
        "environment_id": "libero_pro",
        "embodiment_id": "franka_panda",
        "robot_models": ["Panda"],
        "controller": "CUROBO_JOINT_POSITION+OSC_POSE",
        "public_cameras": ["agentview", "wrist"],
    }
    resolved = resolve_environment_prompt_profile(metadata, project_root=ROOT)
    assert resolved["profile_id"] == "libero_pro_public_direct_v1"
    assert resolved["embodiment_policy"].endswith("franka-panda/manual-public-geometry-v3.md")
    drifted = {**metadata, "robot_models": ["Sawyer"]}
    with pytest.raises(EnvironmentContextError, match="exactly one"):
        resolve_environment_prompt_profile(drifted, project_root=ROOT)


def _synthetic_native_spec() -> dict:
    spec = synthetic_spec()
    # The Synthetic cell's recorded retention is no longer accepted.
    spec["policy"]["context_retention"] = "rewrite_compact_v1"
    return spec


def test_typed_v2_hides_source_paths() -> None:
    v2 = resolve_run_spec(_synthetic_native_spec(), root=ROOT)
    with tempfile.TemporaryDirectory() as temporary:
        v2_compiled = materialize_policy(
            v2["policy_context"], project_root=ROOT,
            destination=Path(temporary) / "v2",
        )
        for channel in ("harness_system", "project_instructions", "episode_task"):
            text = v2_compiled[channel].read_text(encoding="utf-8")
            assert "<!-- policy-layer:" not in text
            assert "ora/" not in text
            assert "tests/data/" not in text
        assert v2_compiled["tool_manual"].name == "tool-manual.md"
        assert "OpenETA" not in v2_compiled["tool_manual"].read_text(
            encoding="utf-8"
        )


def test_robot_runtime_v2_effect_and_nested_observation_contract() -> None:
    surface = resolve_tool_surface("robot_runtime_v2")
    assert surface.mutation_tools == ("robot.submit_action",)
    runtime = object.__new__(RobotToolRuntime)
    runtime.surface = surface
    assert runtime.classify_call_effect("observation.dump_state", {}) == "read_only"
    assert runtime.classify_call_effect("robot.submit_action", {"arm": "left"}) == (
        "mutation"
    )


def test_robot_runtime_v2_rejects_legacy_action_codec() -> None:
    spec = _synthetic_native_spec()
    spec["policy"]["tool_surface"] = {
        "profile": "robot_runtime_v2",
        "component_overrides": {},
    }
    spec["policy"]["action_codec"] = "runtime_action_v3"
    spec["policy"]["tool_allowlist"] = [
        definition["name"]
        for definition in resolve_tool_surface(
            "robot_runtime_v2"
        ).definitions
    ]
    with pytest.raises(ValueError, match="requires target_state_v4"):
        resolve_run_spec(spec, root=ROOT)
