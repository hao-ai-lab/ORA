"""Offline CPU composition and delivery; no simulator or Provider calls."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from ora.agent.composition.ora_local import DEFAULT_LOCAL_COMPOSITION, compose_local_spec
from ora.agent.composition.prompt_bundle import audit_prompt_bundle, load_prompt_bundle
from ora.agent.composition.tool_surface import resolve_tool_surface
from ora.environment.benchmarks.libero_pro import LIBERO_PRO_CODE_REVISION, LIBERO_PRO_DATA_REVISION
from ora.environment.benchmarks.libero_pro_language import base_bddl_language, base_task_language
from ora.orchestrator.benchmark_spec import BASE_LANGUAGE_DIVERGENCE_REVIEW
from ora.orchestrator.run_spec_inputs import provider_policy
from ora.orchestrator.spec import resolve_run_spec


ROOT = Path(__file__).resolve().parents[2]


def _inputs() -> dict:
    language, bddl = base_task_language("libero_goal", 0), base_bddl_language("libero_goal", 0)
    task = {"suite": "libero_goal", "task_id": 0, "task_language": language, "bddl_task_language": bddl}
    if bddl != language:
        task["task_language_divergence_ack"] = {
            "bddl_task_language": bddl, "policy_task_language": language,
            "review": BASE_LANGUAGE_DIVERGENCE_REVIEW,
        }
    return {
        "task": task,
        "source": {"required_libero_pro_revision": LIBERO_PRO_CODE_REVISION,
                   "required_data_revision": LIBERO_PRO_DATA_REVISION,
                   "python_env": "LIBERO_PRO_PYTHON", "libero_pro_root_env": "LIBERO_PRO_ROOT",
                   "data_root_env": "LIBERO_PRO_DATA_ROOT"},
        "provider": provider_policy(), "resources": {"required_host": "local-cpu"},
    }


def test_local_bundle_delivers_audited_cpu_tools_and_embodiment() -> None:
    bundle = load_prompt_bundle(ROOT / DEFAULT_LOCAL_COMPOSITION["prompt_bundle"])
    audit = audit_prompt_bundle(bundle, project_root=ROOT)
    assert audit["passed"], [check for check in audit["checks"] if not check["passed"]]
    surface = resolve_tool_surface(DEFAULT_LOCAL_COMPOSITION["tool_surface"])
    assert surface.mutation_tools == ("move_pose", "close_gripper", "open_gripper")
    assert surface.supports("policy_tool_help_v1")
    assert surface.runtime_id == "native_atomic_causal"


def test_local_fresh_plan_is_cpu_only_and_preserves_source_and_provider() -> None:
    inputs = _inputs()
    untouched = copy.deepcopy(inputs)
    spec = compose_local_spec(**inputs)
    plan = resolve_run_spec(spec, root=ROOT)
    assert inputs == untouched
    assert plan["experiment"] == "ora-interactive"
    assert plan["task_instruction"] == inputs["task"]["task_language"]
    assert plan["policy"]["action_codec"] == "runtime_action_v3"
    assert plan["policy"]["provider_profile"] == "deepseek_flash_openai_low_16k_v1"
    assert plan["policy"]["max_output_tokens"] == 16384
    assert plan["policy"]["memory_policy"] == "disabled"
    assert plan["runtime"]["environment_driver"] == "direct_libero_v1"
    assert plan["runtime"]["require_gpu"] is False
    assert plan["runtime"]["required_host"] == "local-cpu"
    assert plan["runtime"]["required_libero_pro_revision"] == LIBERO_PRO_CODE_REVISION
    assert plan["runtime"]["required_data_revision"] == LIBERO_PRO_DATA_REVISION
    assert plan["runtime"]["environment_prompt_profile"]["profile_id"] == "libero_pro_local_cpu_v1"
    assert not ({"motion_planner", "local_vla", "resource_profile"} & set(plan["runtime"]))
    assert not plan.get("prerequisite")
    assert spec["policy_context"]["project_instructions"][-1].endswith("manual-local-cpu-v1.md")
    assert plan["runtime"].get("task_language_divergence_ack") == inputs["task"].get("task_language_divergence_ack")


@pytest.mark.parametrize("change,error", [
    ({"seed": 50}, "initial state"),
    ({"resources": {"required_host": "local-cpu", "resource_profile": "gpu"}}, "resources"),
    ({"resources": {"required_host": "m2"}}, "local-cpu"),
])
def test_local_composition_refuses_wrong_seed_or_gpu_treatment(change: dict, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        compose_local_spec(**{**_inputs(), **change})


def test_local_composition_requires_exact_language_review_and_source_inputs() -> None:
    inputs = _inputs()
    inputs["task"]["bddl_task_language"] = "different BDDL wording"
    inputs["task"].pop("task_language_divergence_ack", None)
    with pytest.raises(ValueError, match="acknowledgment"):
        compose_local_spec(**inputs)
    inputs = _inputs()
    inputs["source"]["motion_planner"] = {"device": "cuda:0"}
    with pytest.raises(ValueError, match="source inputs"):
        compose_local_spec(**inputs)
    inputs = _inputs()
    inputs["source"]["required_data_revision"] = "f" * 40
    with pytest.raises(ValueError, match="not been reviewed"):
        compose_local_spec(**inputs)


def test_local_composition_cannot_inject_provider_composition_overrides() -> None:
    inputs = _inputs()
    inputs["provider"]["tool_surface"] = {"profile": "other"}
    with pytest.raises(ValueError, match="composition overrides"):
        compose_local_spec(**inputs)


def test_local_result_schemas_accept_real_atomic_runtime_and_preserve_pose_units(tmp_path: Path) -> None:
    # The existing synthetic public-camera fixture exercises the real tool
    # handlers and their new result schemas without loading a simulator.
    from tests.agent.native.test_native_causal_runtime import _runtime

    runtime = _runtime(tmp_path, tool_surface_profile="local_atomic_cpu_v1")
    actual = runtime.invoke("observe", {})
    assert actual["success"] is True
    runtime.invoke("mark_point", {"point_id": "point", "view": "agentview", "x": 2, "y": 1})
    preview = runtime.invoke("preview_move", {
        "translation_delta_mm": [10.0, 0.0, 0.0], "translation_delta_frame": "grip_site",
        "grip_site_z_axis_world": [0.0, 0.0, -1.0],
    })
    assert preview["compiled_pose"]["translation_distance_m"] == pytest.approx(0.01)
    assert preview["compiled_pose"]["rotation_angle_deg"] == pytest.approx(180.0)
    submitted = runtime.invoke("move_pose", {"preview_id": preview["preview_id"]})
    assert submitted["compiled_pose"] == preview["compiled_pose"]
