"""Control plane preflight contract regressions."""

from __future__ import annotations
import hashlib
import json
import os
import subprocess
from pathlib import Path
import pytest
from ora.orchestrator.preflight import _libero_controller_contract_check, _offscreen_graphics_runtime, _provider_marker, _resident_service_prerequisite, run_preflight
from ora.environment import graphics_probe
from ora.environment.backends.libero_pro.session import libero_osc_action_contract
from ora.orchestrator.provider_probe_stages import PROBE_ANSWER_BUDGET
from tests.orchestrator.test_control_plane import (
    _graphics_stub,
    _install_fake_services,
    _service_set_plan,
)


def test_preflight_rejects_a_drifted_resident_segmentation_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = {
        "schema": "ora.segmentation-service-binding.v1",
        "tool_surface": "segmentation_localize_v1",
        "endpoint": "http://127.0.0.1:8114",
        "instance_id": "bound-instance",
    }

    class DriftedSegmenter:
        def __init__(self, **kwargs):
            assert kwargs["expected_binding"] == binding

        def require_ready(self):
            raise ValueError("SAM3 runtime service differs from bound instance_id")

        def close(self):
            pass

    from ora.agent.tools import grasp

    monkeypatch.setattr(grasp, "Sam3ApiSegmenter", DriftedSegmenter)
    check = _resident_service_prerequisite({
        "policy": {"tool_surface": {"profile": "segmentation_localize_v1"}},
        "prerequisite": binding,
    })
    assert check == {
        "check": "resident_service_prerequisite",
        "passed": False,
        "detail": "SAM3 runtime service differs from bound instance_id",
    }


def test_offscreen_graphics_preflight_names_a_missing_egl_stack(
    tmp_path: Path,
) -> None:
    report = {
        "schema": graphics_probe.GRAPHICS_PROBE_SCHEMA,
        "backend": "egl",
        "egl_devices": 0,
        "usable": False,
        "error": "eglQueryDevicesEXT enumerated 0 devices",
    }
    stub = _graphics_stub(tmp_path, f"echo '{json.dumps(report)}'")
    check = _offscreen_graphics_runtime(stub, os.environ)
    assert check["check"] == "offscreen_graphics_runtime"
    assert check["passed"] is False
    assert check["detail"]["egl_devices"] == 0
    assert "0 devices" in check["detail"]["error"]

    report["egl_devices"] = 2
    report["usable"] = True
    report["error"] = None
    healthy = _graphics_stub(tmp_path, f"echo '{json.dumps(report)}'")
    assert _offscreen_graphics_runtime(healthy, os.environ)["passed"] is True


def test_offscreen_graphics_preflight_rejects_an_unrecognized_report(
    tmp_path: Path,
) -> None:
    stub = _graphics_stub(tmp_path, "echo 'Could not import OffScreenRenderEnv' >&2\nexit 1")
    check = _offscreen_graphics_runtime(stub, os.environ)
    assert check["passed"] is False
    assert check["detail"]["returncode"] == 1
    assert "OffScreenRenderEnv" in check["detail"]["stderr_tail"]


def test_graphics_probe_reports_a_contradictory_backend_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.setenv("PYOPENGL_PLATFORM", "glx")
    report = graphics_probe.probe()
    assert report["usable"] is False
    assert report["egl_devices"] is None
    assert "contradicts" in report["error"]


def test_graphics_probe_accepts_a_non_egl_offscreen_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MUJOCO_GL", "osmesa")
    monkeypatch.setenv("PYOPENGL_PLATFORM", "osmesa")
    report = graphics_probe.probe()
    assert report["usable"] is True
    assert report["egl_devices"] is None
    assert report["error"] is None


def test_preflight_verifies_every_service_in_a_v2_set_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list = []
    _install_fake_services(monkeypatch, seen)
    check = _resident_service_prerequisite(_service_set_plan())
    assert check["passed"] is True
    # The defect this replaces: a set binding fell through the single-schema
    # comparison and returned "not required" without contacting anything.
    assert check["detail"] != "not required"
    assert sorted(seen) == ["CGN", "SAM3"]
    assert sorted(check["detail"]["services"]) == ["grasp_proposals", "segmentation"]


def test_preflight_rejects_a_v2_set_binding_whose_grasp_service_drifted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_services(monkeypatch, [])
    check = _resident_service_prerequisite(_service_set_plan(
        drifted="ora.grasp-proposal-service-binding.v1",
    ))
    assert check["passed"] is False
    assert "differs from bound instance_id" in check["detail"]


def test_preflight_reports_an_unownable_prerequisite_as_unverified() -> None:
    check = _resident_service_prerequisite({
        "policy": {"tool_surface": {"profile": "libero_placement_direct_overhead_v1"}},
        "prerequisite": {
            "schema": "ora.experiment-owned-prerequisite.v1",
            "tool_surface": "libero_placement_direct_overhead_v1",
        },
    })
    assert check["passed"] is True
    assert check["detail"]["verified"] is False
    assert check["detail"]["schema"] == "ora.experiment-owned-prerequisite.v1"


def test_libero_preflight_recomputes_controller_units_and_binding(
    tmp_path: Path,
) -> None:
    contract = libero_osc_action_contract({
        "control_delta": True,
        "input_min": -1,
        "input_max": 1,
        "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
        "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
    })
    robosuite_root = tmp_path / "robosuite"
    controller_path = robosuite_root / "controllers/osc_pose.json"
    gripper_path = robosuite_root / "models/panda_gripper.xml"
    gripper_module_path = robosuite_root / "models/grippers/panda_gripper.py"
    module_path = robosuite_root / "__init__.py"
    controller_config = {
        "control_delta": True,
        "input_min": -1,
        "input_max": 1,
        "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
        "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
    }
    for path, content in (
        (module_path, "# robosuite\n"),
        (controller_path, json.dumps(controller_config) + "\n"),
        (gripper_module_path, "# Panda gripper\n"),
        (gripper_path, "<mujoco/>\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    observed = {
        "controller_action_contract": contract,
        "gripper_action_probe": {
            "close": [-0.01, 0.01],
            "zero_hold": [-0.01, 0.01],
            "reopen": [0.0, 0.0],
        },
        "controller_binding": {
            "robosuite_module": str(module_path),
            "robosuite_module_sha256": digest(module_path),
            "controller_config_file": str(controller_path),
            "controller_config_sha256": digest(controller_path),
            "panda_gripper_module": str(gripper_module_path),
            "panda_gripper_module_sha256": digest(gripper_module_path),
            "panda_gripper_file": str(gripper_path),
            "panda_gripper_sha256": digest(gripper_path),
        },
    }
    accepted = _libero_controller_contract_check(observed)
    assert accepted["passed"] is True
    assert accepted["detail"]["schema"] == (
        "ora.libero-controller-qualification.v1"
    )
    stale_scale = json.loads(json.dumps(observed))
    stale_scale["controller_action_contract"]["output_max"][3] = 0.1
    rejected = _libero_controller_contract_check(stale_scale)
    assert rejected["passed"] is False
    assert "drifted" in rejected["detail"]
    drifted_file = json.loads(json.dumps(observed))
    drifted_file["controller_binding"]["controller_config_sha256"] = "0" * 64
    assert _libero_controller_contract_check(drifted_file)["passed"] is False
    mismatched_file = json.loads(json.dumps(observed))
    mismatched_file["controller_action_contract"] = libero_osc_action_contract({
        **controller_config,
        "output_min": [-0.04, -0.04, -0.04, -0.4, -0.4, -0.4],
        "output_max": [0.04, 0.04, 0.04, 0.4, 0.4, 0.4],
    })
    rejected_file_contract = _libero_controller_contract_check(mismatched_file)
    assert rejected_file_contract["passed"] is False
    assert "bound config file" in rejected_file_contract["detail"]
    drifted_gripper = json.loads(json.dumps(observed))
    drifted_gripper["gripper_action_probe"]["zero_hold"] = [0.0, 0.0]
    rejected_gripper = _libero_controller_contract_check(drifted_gripper)
    assert rejected_gripper["passed"] is False
    assert "sign-integrator" in rejected_gripper["detail"]
    missing_binding = {"controller_action_contract": contract}
    assert _libero_controller_contract_check(missing_binding)["passed"] is False


def test_native_preflight_marker_allows_reasoning_before_output(monkeypatch) -> None:
    observed: dict = {}

    class ProbeProvider:
        def __init__(
            self, *, base_url: str, api_key: str, timeout_seconds: float,
        ) -> None:
            assert base_url == "https://provider.invalid/v1"
            assert api_key == "test-secret"
            assert timeout_seconds == 900.0

        def complete(self, *, model_call_id: str, request: dict) -> dict:
            observed.update(request)
            marker = request["messages"][0]["content"].rsplit(": ", 1)[-1]
            return {"message": {"role": "assistant", "content": marker}}

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        "ora.agent.providers.native.OpenAICompatibleProvider", ProbeProvider
    )
    monkeypatch.setenv("TEST_NATIVE_BASE_URL", "https://provider.invalid/v1")
    monkeypatch.setenv("TEST_NATIVE_API_KEY", "test-secret")
    check = _provider_marker({"policy": {
        "kind": "native_api",
        "require_live_marker": True,
        "base_url_env": "TEST_NATIVE_BASE_URL",
        "api_key_env": "TEST_NATIVE_API_KEY",
        "endpoint": "https://provider.invalid/v1",
        "model": "reasoning-model",
        "provider": "openai_compatible",
        "reasoning_effort": "high",
        "stream": True,
        "supports_images": False,
    }}, subprocess.run)
    assert check["passed"] is True
    assert observed["max_tokens"] == PROBE_ANSWER_BUDGET
    assert observed["reasoning_effort"] == "high"
    assert observed["stream_options"] == {"include_usage": True}


def test_native_preflight_probes_an_openai_loopback_gateway_as_the_session_will(monkeypatch) -> None:
    """The probe client carries the plan's output-token name and a preflight affinity id."""
    observed: dict = {}

    class Probe:
        def complete(self, *, model_call_id, request):
            observed.setdefault("requests", []).append(request)
            marker = request["messages"][0]["content"].rsplit(": ", 1)[-1]
            return {"message": {"role": "assistant", "content": marker}}

        def close(self):
            pass

    def build(protocol, **kwargs):
        observed.update(kwargs, protocol=protocol)
        return Probe()

    monkeypatch.setattr("ora.agent.providers.build_native_provider", build)
    monkeypatch.setenv("TEST_NATIVE_BASE_URL", "http://127.0.0.1:8319/v1")
    monkeypatch.setenv("TEST_NATIVE_API_KEY", "test-secret")
    check = _provider_marker({"policy": {
        "kind": "native_api", "base_url_env": "TEST_NATIVE_BASE_URL",
        "api_key_env": "TEST_NATIVE_API_KEY", "endpoint": "http://127.0.0.1:8319/v1",
        "endpoint_transport": "loopback_http_v1", "model": "gpt-6-astra",
        "provider": "openai_compatible", "reasoning_effort": "high", "stream": True,
        "supports_images": False, "output_token_parameter": "max_completion_tokens",
        "provider_session_affinity": "x_session_id_run_v1",
    }}, subprocess.run)
    assert check["passed"] is True
    assert observed["protocol"] == "openai_compatible"
    assert observed["output_token_parameter"] == "max_completion_tokens"
    assert observed["session_affinity_id"].startswith("ora:preflight:")
    assert check["detail"]["session_affinity"] == "x_session_id_run_v1"


@pytest.mark.parametrize("fails", [False, True])
def test_native_preflight_marker_isolates_candidates_and_redacts_full_pool(monkeypatch, fails):
    observed = {}
    keys = ("fixture-primary-only", "fixture-standby-a", "fixture-standby-b")

    class Probe:
        def complete(self, *, model_call_id, request):
            if fails:
                raise RuntimeError("rejected " + ",".join(keys))
            marker = request["messages"][0]["content"].rsplit(": ", 1)[-1]
            return {"message": {"role": "assistant", "content": marker}}

        def close(self):
            observed["closed"] = True

    def build(*args, **kwargs):
        observed.update(kwargs)
        return Probe()

    monkeypatch.setattr("ora.agent.providers.build_native_provider", build)
    monkeypatch.setenv("TEST_NATIVE_BASE_URL", "https://provider.invalid")
    monkeypatch.setenv("TEST_NATIVE_API_KEY", keys[0])
    monkeypatch.setenv("TEST_NATIVE_API_KEYS", ",".join(keys[1:]))
    check = _provider_marker({"policy": {
        "kind": "native_api", "base_url_env": "TEST_NATIVE_BASE_URL",
        "api_key_env": "TEST_NATIVE_API_KEY", "endpoint": "https://provider.invalid",
        "model": "fixture-model", "provider": "anthropic_messages",
        "reasoning_effort": "high", "stream": True, "supports_images": False,
    }}, subprocess.run)
    assert observed["api_key"] == keys[0]
    assert "standby_api_keys" not in observed  # One full qualification per candidate.
    assert check["detail"]["credential_selection"]["pool_size"] == 3
    assert observed["closed"] is True
    assert check["passed"] is not fails
    assert all(key not in json.dumps(check) for key in keys)


@pytest.mark.parametrize("answer, normalized", [("red", "red"), ("Vermilion", "vermilion"), ("Red.", "red"),
                                                ("orange-red", "orange-red"), ("Reddish", "reddish")])
def test_native_preflight_requires_image_request_for_image_capability(monkeypatch, answer, normalized) -> None:
    # One-word answers in the red family pass; a stochastic synonym or a
    # trailing full stop is not a provider fault.
    calls: list[dict] = []

    class ProbeProvider:
        def __init__(
            self, *, base_url: str, api_key: str, timeout_seconds: float,
        ) -> None:
            assert base_url == "https://provider.invalid/api/anthropic"
            assert api_key == "test-secret"
            assert timeout_seconds == 900.0

        def complete(self, *, model_call_id: str, request: dict) -> dict:
            calls.append(request)
            if model_call_id == "preflight-marker":
                marker = request["messages"][0]["content"].rsplit(": ", 1)[-1]
                content = marker
            else:
                assert model_call_id == "preflight-image"
                assert request["messages"][0]["content"][1]["type"] == "image_url"
                image_url = request["messages"][0]["content"][1]["image_url"]["url"]
                assert image_url.startswith("data:image/png;base64,")
                assert len(image_url) > 220
                content = answer
            return {
                "message": {
                    "role": "assistant",
                    "content": content,
                    "reasoning_content": "checked pixels" if model_call_id == "preflight-image" else "",
                },
                "finish_reason": "stop",
                "usage": {"input_tokens": 10, "output_tokens": 4},
            }

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        "ora.agent.providers.native.AnthropicMessagesProvider", ProbeProvider
    )
    monkeypatch.setenv("TEST_NATIVE_BASE_URL", "https://provider.invalid/api/anthropic")
    monkeypatch.setenv("TEST_NATIVE_API_KEY", "test-secret")
    check = _provider_marker({"policy": {
        "kind": "native_api",
        "require_live_marker": True,
        "base_url_env": "TEST_NATIVE_BASE_URL",
        "api_key_env": "TEST_NATIVE_API_KEY",
        "endpoint": "https://provider.invalid/api/anthropic",
        "model": "vision-model",
        "provider": "anthropic_messages",
        "reasoning_effort": "provider_default",
        "stream": True,
        "supports_images": True,
    }}, subprocess.run)
    assert check["passed"] is True
    assert check["detail"]["image_request_required"] is True
    assert check["detail"]["image_response_observed"] is True
    assert check["detail"]["image_semantics_verified"] is True
    assert check["detail"]["image_answer"] == normalized
    assert check["detail"]["image_finish_reason"] == "stop"
    assert check["detail"]["image_reasoning_observed"] is True
    assert check["detail"]["image_usage_observed"] is True
    assert len(calls) == 2
    assert calls[1]["max_tokens"] == PROBE_ANSWER_BUDGET


def test_native_preflight_probe_schema_requires_real_tool_continuity(monkeypatch) -> None:
    calls: list[tuple[str, dict]] = []

    class ProbeProvider:
        def __init__(
            self, *, base_url: str, api_key: str, timeout_seconds: float,
        ) -> None:
            assert base_url == "https://provider.invalid/api/anthropic"
            assert api_key == "test-secret"
            assert timeout_seconds == 900.0

        def complete(self, *, model_call_id: str, request: dict) -> dict:
            calls.append((model_call_id, request))
            if model_call_id == "preflight-marker":
                marker = request["messages"][0]["content"].rsplit(": ", 1)[-1]
                return {
                    "message": {"role": "assistant", "content": marker},
                    "finish_reason": "stop",
                }
            if model_call_id == "preflight-continuity-tool":
                token = request["messages"][0]["content"].split(
                    "with token ", 1,
                )[1].split(".", 1)[0]
                return {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": "continuity-call-1",
                            "type": "function",
                            "function": {
                                "name": "preflight_echo",
                                "arguments": json.dumps({"token": token}),
                            },
                        }],
                        "provider_continuation": {
                            "provider": "anthropic_messages",
                            "blocks": [{"type": "redacted_thinking", "data": "opaque"}],
                        },
                    },
                    "finish_reason": "tool_calls",
                }
            assert model_call_id == "preflight-continuity-resume"
            assert request["messages"][1]["provider_continuation"]["blocks"] == [
                {"type": "redacted_thinking", "data": "opaque"},
            ]
            assert request["messages"][2]["tool_call_id"] == "continuity-call-1"
            marker = request["messages"][3]["content"].rsplit("exactly ", 1)[-1]
            return {
                "message": {"role": "assistant", "content": marker},
                "finish_reason": "stop",
            }

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        "ora.agent.providers.native.AnthropicMessagesProvider", ProbeProvider
    )
    monkeypatch.setenv("TEST_NATIVE_BASE_URL", "https://provider.invalid/api/anthropic")
    monkeypatch.setenv("TEST_NATIVE_API_KEY", "test-secret")
    check = _provider_marker({"policy": {
        "kind": "native_api",
        "require_live_marker": True,
        "base_url_env": "TEST_NATIVE_BASE_URL",
        "api_key_env": "TEST_NATIVE_API_KEY",
        "endpoint": "https://provider.invalid/api/anthropic",
        "model": "continuity-model",
        "provider": "anthropic_messages",
        "reasoning_effort": "medium",
        "stream": True,
        "supports_images": False,
        "provider_qualification_schema": "ora.native-provider-probe.v1",
        "provider_qualification_version": 1,
    }}, subprocess.run)
    assert check["passed"] is True
    assert check["detail"]["continuity_required"] is True
    assert check["detail"]["continuity_tool_call_valid"] is True
    assert check["detail"]["continuity_passed"] is True
    assert [model_call_id for model_call_id, _request in calls] == [
        "preflight-marker",
        "preflight-continuity-tool",
        "preflight-continuity-resume",
    ]


def test_native_preflight_rejects_nonempty_but_incorrect_image_answer(monkeypatch) -> None:
    class ProbeProvider:
        def __init__(
            self, *, base_url: str, api_key: str, timeout_seconds: float,
        ) -> None:
            pass

        def complete(self, *, model_call_id: str, request: dict) -> dict:
            if model_call_id == "preflight-marker":
                marker = request["messages"][0]["content"].rsplit(": ", 1)[-1]
                content = marker
            else:
                content = "white"
            return {"message": {"role": "assistant", "content": content}}

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        "ora.agent.providers.native.AnthropicMessagesProvider", ProbeProvider
    )
    monkeypatch.setenv("TEST_NATIVE_BASE_URL", "https://provider.invalid/api/anthropic")
    monkeypatch.setenv("TEST_NATIVE_API_KEY", "test-secret")
    check = _provider_marker({"policy": {
        "kind": "native_api",
        "require_live_marker": True,
        "base_url_env": "TEST_NATIVE_BASE_URL",
        "api_key_env": "TEST_NATIVE_API_KEY",
        "endpoint": "https://provider.invalid/api/anthropic",
        "model": "text-only-model",
        "provider": "anthropic_messages",
        "reasoning_effort": "provider_default",
        "stream": True,
        "supports_images": True,
    }}, subprocess.run)
    assert check["passed"] is False
    assert check["detail"]["marker_observed"] is True
    assert check["detail"]["image_response_observed"] is True
    assert check["detail"]["image_semantics_verified"] is False
    assert check["detail"]["image_answer"] == "white"


def test_host_storage_is_hermetic_unless_a_test_opts_in() -> None:
    """The autouse conftest stub keeps every other test off real disks and /dev/shm."""

    from ora.orchestrator import host_storage
    from ora.orchestrator.checks import check

    assert {item["detail"] for item in host_storage.storage_checks({}, check=check)} == {
        "hermetic pytest stub"
    }
    staged = host_storage.stage_launch({}, "hermetic-run")
    assert staged["shm_sweep"]["removed"] == 0
    assert host_storage.release_run_scratch(Path("/nonexistent")) is False
