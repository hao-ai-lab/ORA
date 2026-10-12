"""Two-call registered-provider grasp diagnostic using the existing canonical Provider adapter.

This is a bounded Provider/tool diagnostic, not a NativePolicySession, full task
or K2 rollout. The caller alone submits the validated proposal to Supervisor.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from ora.agent.context import causal_provider_value
from ora.agent.context.public_facts import project_current_public_facts
from ora.agent.evidence.trajectory import image_data_url
from ora.agent.native.receipt_projection import ReceiptProjectionContext, resolve_receipt_effect_projector
from ora.agent.providers.native import build_native_provider
from ora.agent.providers.profiles import BUILTIN_PROVIDER_CATALOG, load_provider_profiles

SOURCE = Path(__file__).resolve().parents[2]
KIMI_PROFILE = "kimi_direct_k3_anthropic_high_v1"
DEEPSEEK_PROFILE = "deepseek_flash_openai_low_16k_v1"
PROFILE_CATALOGS = {
    KIMI_PROFILE: SOURCE / "ora/agent/providers/kimi-direct-profiles.yaml",
    DEEPSEEK_PROFILE: BUILTIN_PROVIDER_CATALOG,
}
EXPECTED_BINDINGS = {
    KIMI_PROFILE: ("anthropic_messages", "https://api.kimi.com/coding", "k3", "high", 8192, True, True),
    DEEPSEEK_PROFILE: ("openai_compatible", "https://api.deepseek.com", "deepseek-flash", "low", 16384, True, True),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_probe_binding(profile_id: str) -> dict[str, Any]:
    if profile_id not in PROFILE_CATALOGS:
        raise ValueError("unsupported grasp diagnostic Profile")
    catalog = PROFILE_CATALOGS[profile_id]
    profile = load_provider_profiles(catalog)[profile_id]
    if tuple(profile[key] for key in ("protocol", "endpoint", "model", "reasoning_effort",
            "max_output_tokens", "stream", "supports_images")) != EXPECTED_BINDINGS[profile_id]:
        raise ValueError("registered diagnostic Profile changed")
    return {"profile_id": profile_id, "profile": profile,
            "catalog": {"path": str(catalog.relative_to(SOURCE)), "sha256": _sha(catalog)},
            "probe_source": {"path": str(Path(__file__).relative_to(SOURCE)), "sha256": _sha(Path(__file__))}}


def _image_payloads(value: Any) -> list[str]:
    if isinstance(value, list):
        return [item for child in value for item in _image_payloads(child)]
    if not isinstance(value, dict):
        return []
    if value.get("type") == "image_url":
        return [value["image_url"]["url"].split(",", 1)[1]]
    if value.get("type") == "image":
        return [value["source"]["data"]]
    return [item for child in value.values() for item in _image_payloads(child)]


class ProviderGraspProbe:
    def __init__(self, binding: dict, output: Path, caseid: str):
        if binding != load_probe_binding(binding["profile_id"]):
            raise ValueError("Provider diagnostic source binding changed")
        profile = binding["profile"]
        if (os.environ.get(profile["model_env"]) != profile["model"]
                or os.environ.get(profile["base_url_env"]) != profile["endpoint"]):
            raise ValueError("Provider diagnostic requires exact bound model and endpoint environment")
        key = os.environ.get(profile["api_key_env"])
        if not key:
            raise ValueError("Provider diagnostic credential is unavailable")
        self.output, self.root = Path(output).resolve(), Path(output).resolve() / "provider-probe"
        self.binding, self.caseid, self.secret = copy.deepcopy(binding), caseid, key
        self.provider = build_native_provider(profile["protocol"], base_url=profile["endpoint"],
            api_key=key, timeout_seconds=180, recovery_policy="legacy_v1")
        self.provider.max_retries = 0
        self.calls: list[dict] = []
        self.messages: list[dict] = []
        self.images: list[dict] = []
        self.phase = "ready"
        self._save("binding.json", {**binding, "case_id": caseid, "logical_call_limit": 2,
            "transport_retries": 0, "request_timeout_seconds": 180,
            "scope": "bounded_provider_plus_shared_grasp_tool_not_native_session_or_task_benchmark"})

    def _save(self, name: str, value: Any) -> None:
        text = json.dumps(value, ensure_ascii=False, indent=2)
        # Credentials never belong to request/response evidence, even on error.
        if self.secret in text:
            raise ValueError("credential material refused in diagnostic evidence")
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / name).open("x") as stream:
            stream.write(text + "\n")

    def _image(self, root: Path, path: str, digest: str, *, role: str, camera: str,
               observation_id: str) -> list[dict]:
        resolved = (root / path).resolve()
        if not resolved.is_relative_to(root.resolve()) or _sha(resolved) != digest:
            raise ValueError("diagnostic RGB source identity mismatch")
        self.images.append({"role": role, "camera": camera, "observation_id": observation_id,
                            "path": str(resolved), "sha256": digest})
        return [{"type": "text", "text": f"{role} RGB: {camera}; observation {observation_id}"},
                {"type": "image_url", "image_url": {"url": image_data_url(resolved)}}]

    def _observation(self, host: Any, observation: dict, root: Path) -> list[dict]:
        views = ["head_camera", f"{self.arm}_camera"]
        facts = project_current_public_facts(observation=observation, observation_root=root,
            view_config=host.surface.view_config, selected_views=views)
        blocks = [{"type": "text", "text": json.dumps(causal_provider_value(facts))}]
        for view in views:
            files = observation["cameras"][view]["files"]
            ref = files.get("rgb_lossless_png") or files["rgb"]
            blocks += self._image(root, ref["path"], ref["sha256"], role="current", camera=view,
                                  observation_id=observation["observation_id"])
        return blocks

    def _call(self, phase: str, *, tool_choice: str) -> dict:
        if len(self.calls) >= 2:
            raise ValueError("Provider diagnostic call budget exhausted")
        index = len(self.calls) + 1
        profile = self.binding["profile"]
        request = {"model": profile["model"], "max_tokens": profile["max_output_tokens"],
                   "reasoning_effort": profile["reasoning_effort"], "stream": profile["stream"],
                   "messages": copy.deepcopy(self.messages), "tools": self.tools,
                   "tool_choice": tool_choice}
        wire = self.provider.prepare_request(request)
        payloads = _image_payloads(request)
        if payloads != _image_payloads(wire):
            raise ValueError("Provider image translation changed original RGB payloads")
        delivery = [hashlib.sha256(base64.b64decode(value, validate=True)).hexdigest() for value in payloads]
        if delivery != [item["sha256"] for item in self.images]:
            raise ValueError("Provider image delivery differs from source RGB order or bytes")
        self._save(f"call-{index}-request.json", request)
        self._save(f"call-{index}-wire-request.json", wire)
        self._save(f"call-{index}-images.json", {"sources": self.images,
            "wire_image_sha256": delivery, "original_bytes_verified": True})
        self.calls.append({"phase": phase, "status": "started"})
        try:
            result = self.provider.complete(request=request, model_call_id=f"{self.caseid}-{phase}")
        except Exception as exc:
            failure = getattr(exc, "provider_failure", {})
            self.calls[-1].update(status="failed", exception_type=type(exc).__name__,
                                  attempts=failure.get("attempts"), usage=None)
            self._save(f"call-{index}-failure.json", self.calls[-1])
            self.phase = "failed"
            raise RuntimeError(f"Provider {phase} request failed; see bounded failure evidence") from None
        self._save(f"call-{index}-response.json", result)
        # Anthropic returns its wire body; OpenAI's adapter exposes that body
        # through prepare_request only. Keep the original response unmodified.
        wire_mismatch = (profile["protocol"] == "anthropic_messages" and result.get("wire_request") != wire)
        if wire_mismatch or result.get("attempts") != 1:
            self.phase = "failed"
            raise ValueError("Provider wire or attempt count differs from bounded request")
        self.calls[-1].update(status="completed", attempts=result["attempts"], usage=result.get("usage"),
                             response_id=result.get("response_id"), finish_reason=result.get("finish_reason"))
        if result.get("output_rejection") or result.get("finish_reason") not in {"stop", "tool_calls"}:
            self.phase = "failed"
            raise ValueError("Provider diagnostic requires a complete uncapped response")
        return result["message"]

    def propose(self, host: Any, case: dict, observation: dict, observation_root: Path) -> dict:
        if self.phase != "ready":
            raise ValueError("Provider diagnostic proposal cannot be repeated")
        self.phase, self.arm = "proposing", case["arm"]
        candidate = case["candidate"]
        self.candidate_id = candidate["candidate_id"]
        definition = next(item for item in host.definitions() if item["name"] == "grasp_execute")
        self.tools = [{"type": "function", "function": {"name": definition["name"],
            "description": definition["description"], "parameters": definition["inputSchema"]}}]
        public_candidate = {key: copy.deepcopy(candidate[key]) for key in (
            "candidate_id", "pregrasp_pose", "contact_pose", "gripper_opening_m", "score",
        ) if key in candidate}
        self.messages = [{"role": "system", "content": (
            "You are performing one authorized bounded grasp-tool verification. This is not a full task. "
            "The operator supplied one current-observation-bound candidate fixture. Inspect the public state, "
            "images and candidate, then call grasp_execute exactly once for that candidate with a rationale. "
            "Opening and a 30 mm upward probe are authorized. Do not change candidate, retry, or call other tools. "
            "Subsequently assess only the actual executed stages and evidence; do not claim task success.\n\n"
            + host.manual_path.read_text())}, {"role": "user", "content": [
                {"type": "text", "text": json.dumps({"arm": self.arm, "candidate": public_candidate,
                    "fixture": "operator_supplied_public_geometry", "diagnostic_mode": case["mode"]})},
                *self._observation(host, observation, observation_root)]}]
        self._save("tool-and-manual.json", {"tool": definition, "manual_path": str(host.manual_path),
                                            "manual_sha256": _sha(host.manual_path)})
        message = self._call("propose", tool_choice="auto")
        calls = message.get("tool_calls", [])
        if len(calls) != 1 or calls[0].get("function", {}).get("name") != "grasp_execute":
            self.phase = "failed"
            raise ValueError("Provider must propose exactly one permitted grasp_execute")
        args = json.loads(calls[0]["function"]["arguments"])
        if (not isinstance(args, dict) or set(args) != {"candidate_id", "rationale"}
                or args["candidate_id"] != self.candidate_id or not isinstance(args["rationale"], str)
                or not args["rationale"].strip()):
            self.phase = "failed"
            raise ValueError("Provider proposal is not bound to the permitted candidate")
        self.messages.append(copy.deepcopy(message))  # includes adapter-owned continuation verbatim
        self.tool_call_id, self.phase = calls[0]["id"], "proposed"
        return args

    def review(self, host: Any, receipt: dict, observation: dict, observation_root: Path,
               images: list[dict]) -> dict:
        if self.phase != "proposed":
            raise ValueError("Provider review requires exactly one preceding proposal")
        self.phase = "reviewing"
        record = receipt.get("receipt", receipt)
        project = resolve_receipt_effect_projector(record.get("schema"))
        if (record.get("skill") != "grasp_execute" or project is None
                or record.get("candidate_id") != self.candidate_id):
            raise ValueError("Provider review requires a shared grasp skill receipt")
        effect = project(record, ReceiptProjectionContext(record.get("action_id"), "grasp_execute", host.surface.supports))
        self.messages.append({"role": "tool", "tool_call_id": self.tool_call_id,
                              "name": "grasp_execute", "content": json.dumps(effect)})
        blocks = self._observation(host, observation, observation_root)
        if len(images) > 2:
            raise ValueError("Provider grasp baseline permits at most two original images")
        for item in images:
            if item.get("role") != "post_close_before_lift":
                raise ValueError("unexpected grasp baseline image role")
            blocks += self._image(self.output, item["run_path"], item["sha256"],
                role="historical_post_close_before_lift", camera=item["camera"], observation_id=item["observation_id"])
        blocks.append({"type": "text", "text": (
            "Return a concise assessment of which stages actually ran, measured lift and gripper evidence, "
            "what the images support, and remaining unknowns. Distinguish an empty grasp or failed approach "
            "from unavailable evidence. Do not infer stable holding or task success from completion. "
            "No more actions or tool calls are permitted.")})
        self.messages.append({"role": "user", "content": blocks})
        message = self._call("review", tool_choice="none")
        if message.get("tool_calls") or not isinstance(message.get("content"), str) or not message["content"].strip():
            self.phase = "failed"
            raise ValueError("Provider review must contain assessment text and zero tool calls")
        self.phase = "reviewed"
        report = {"case_id": self.caseid, "profile_id": self.binding["profile_id"], "logical_calls": len(self.calls),
            "transport_attempts": sum(item["attempts"] for item in self.calls), "calls": self.calls,
            "assessment": message["content"], "projected_skill_effect": effect,
            "official_task_verdict_claimed": False, "assessment_is_model_judgement_not_qualification": True}
        self._save("report.json", report)
        return report

    def close(self) -> None:
        self.provider.close()
