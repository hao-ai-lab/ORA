"""Composed tools: the tool-call edge and the executor-side start.

``_composed_tool`` is what ``invoke`` calls for a tool the surface binds to a
composed library entry: the closed-skill lifecycle refusals, ``plan()`` over
the current observation, the commitment (with the host state the body never
sees) written beside the control state, and one submitted action.
``start_composed_tool`` is the trusted executor's half: it reloads that
commitment, checks it against the submitted action, and returns the
:class:`~ora.agent.tools.composed.executor.Execution` the
orchestrator drives.

``composed_read`` is ``sdk.read``: one read tool called through ``invoke``
exactly as the policy would call it, on the current observation or, during
execution, on an observation captured after the last dispatched action (the
runtime's current observation is overridden for that one call). Tools bound
from a practice workspace load from the binding's ``root`` and nested tools
from the binding's pinned closure, each checked against its source sha256.

A mixin of ``core.ManipulationToolRuntime``; it holds methods only, no
dataclass fields.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

from ora.agent.tools.robot import ToolError
from ora.contracts import COMPOSED_TOOL_ACTION, COMPOSED_TOOL_ACTION_SCHEMA, validate_runtime_action


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class ComposedToolRuntimeMixin:
    """The tool runtime's side of composed tools."""

    def _composed_binding(self, name: str) -> Mapping[str, Any]:
        binding = (getattr(self.surface, "composed_tools", None) or {}).get(name)
        if binding is None:
            raise ToolError(f"{name} is not a composed tool on this surface", code="tool_unavailable", retryable=False)
        return binding

    def _composed_library_tool(self, binding: Mapping[str, Any]) -> Any:
        from ora.agent.tools.composed.library import LIBRARY_ROOT, load_tool

        cache = self.__dict__.setdefault("_composed_tool_cache", {})
        root = Path(binding["root"]) if binding.get("root") else LIBRARY_ROOT
        key = (str(root), binding["tool"], binding["version"])
        if key not in cache:
            tool = load_tool(root / binding["tool"] / binding["version"])
            if binding.get("source_sha256") and tool.program.digest != binding["source_sha256"]:
                raise ToolError(f"{binding['tool']}/{binding['version']} source differs from its binding",
                                code="internal_tool_error", retryable=False)
            cache[key] = tool
        return cache[key]

    def _composed_nested_tool(self, binding: Mapping[str, Any], ref: str) -> Any:
        """A tool stage's tool, from the pinned closure of the surface binding."""

        from ora.agent.tools.composed.library import load_tool

        pinned = (binding.get("nested") or {}).get(ref)
        if pinned is None:
            raise ToolError(f"{ref} is not pinned by the binding of {binding['tool']}", code="internal_tool_error",
                            retryable=False)
        cache = self.__dict__.setdefault("_composed_tool_cache", {})
        key = ("nested", pinned["directory"])
        if key not in cache:
            tool = load_tool(Path(pinned["directory"]))
            if tool.program.digest != pinned["source_sha256"]:
                raise ToolError(f"{ref} source differs from its pinned sha256", code="internal_tool_error",
                                retryable=False)
            cache[key] = tool
        return cache[key]

    @contextlib.contextmanager
    def _observation_override(self, observation: Mapping[str, Any], root: Path) -> Iterator[dict[str, Any]]:
        """Make a captured, unpublished observation the current one for the calls inside."""

        manifest = Path(root) / "observation.json"
        published = json.loads((Path(self.control_root) / "current-observation.json").read_text(encoding="utf-8"))
        control = {
            "open": True, "observation_id": str(observation["observation_id"]),
            "observation_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "source_root": str(Path(root).resolve()), "generation": published.get("generation"),
        }
        previous = self.__dict__.get("_current_override")
        self.__dict__["_current_override"] = (control, Path(root).resolve(), dict(observation))
        try:
            yield control
        finally:
            self.__dict__["_current_override"] = previous

    def composed_read(
        self, tool: str, arguments: Mapping[str, Any], *, observation: Mapping[str, Any] | None = None,
        root: Path | None = None,
    ) -> dict[str, Any]:
        """``sdk.read``: one read tool's policy-visible result, or its structured refusal."""

        from ora.agent.tools.composed.sdk import READ_TOOLS

        if tool not in READ_TOOLS:
            raise ValueError(f"sdk.read reaches only {list(READ_TOOLS)}")
        context = (self._observation_override(observation, root) if observation is not None and root is not None
                   else contextlib.nullcontext())
        try:
            with context:
                result = self.invoke(tool, dict(arguments))
        except ToolError as error:
            reported = error.as_dict()
            return {"ok": False, "error": {key: reported.get(key) for key in ("code", "message", "field", "retryable")}}
        visible = {key: value for key, value in result.items()
                   if key not in {"image_base64", "images_base64", "_runtime"}}
        return {"ok": True, "result": json.loads(json.dumps(visible, default=str))}

    def _composed_tool(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        from ora.agent.tools import skill_driver, skill_ledger
        from ora.agent.tools.composed import executor
        from ora.agent.tools.composed.host import COMMITMENT_FILENAME, PlanPort
        from ora.agent.tools.composed.tunables import load_tunables

        binding = self._composed_binding(name)
        self._ensure_mutation_available(name)
        active = skill_driver.reconcile(self)
        if active is not None and active["phase"] not in {"completed", "aborted", "terminated"}:
            raise ToolError(
                "a committed skill sequence is already in flight; resume or abandon it before starting another",
                code="skill_sequence_active", retryable=False,
                constraints={"skill": active["skill"], "phase": active["phase"],
                             "pending_stage": skill_ledger.pending_stage(active)},
            )
        control, root, observation, _pose, _gripper = self._current_grip_state()
        tool = self._composed_library_tool(binding)
        plan = PlanPort(self, control=control, root=root, observation=observation, binding=binding)
        commitment, result = executor.submit(
            tool, arm_profile=binding["arm_profile"], arguments=arguments, port=plan.port(),
            tunables=load_tunables(), call_id=f"action-{control['generation']}",
            observation_id=str(control["observation_id"]), observation_sha256=str(control["observation_sha256"]),
        )
        if commitment.compiled_sha256 != binding["compiled_sha256"]:
            raise ToolError(f"{name} compiled identity differs from the surface binding",
                            code="internal_tool_error", retryable=False)
        _write_json(Path(self.control_root) / COMMITMENT_FILENAME,
                    {**commitment.to_json(), "host_state": plan.host_state})
        action = validate_runtime_action({
            "schema": COMPOSED_TOOL_ACTION_SCHEMA,
            "observation_id": control["observation_id"],
            "action_id": commitment.call_id,
            "action": COMPOSED_TOOL_ACTION,
            "arguments": {
                "tool": commitment.tool, "version": commitment.version, "arm_profile": commitment.arm_profile,
                "compiled_sha256": commitment.compiled_sha256, "rationale": commitment.rationale,
                "committed_sequence": list(result["committed_sequence"]),
            },
            "bound_observation_sha256": control["observation_sha256"],
        })
        self._persist_action(Path(self.control_root) / "submitted-action.json", dict(control), action)
        return {**result, "_runtime": {"directive": "action_submitted", "action_id": action["action_id"]}}

    def start_composed_tool(
        self,
        action: Mapping[str, Any],
        *,
        capture: Callable[[], tuple[dict[str, Any], Path]],
        latest: Callable[[], tuple[dict[str, Any], Path]],
        fresh: Callable[[], tuple[dict[str, Any], Path]] | None = None,
    ) -> Any:
        """The Execution for one submitted composed tool action, bound to this runtime.

        ``fresh`` returns an observation taken after the last dispatched action
        (capturing one when needed); ``sdk.read`` and nested tools plan on it.
        """

        from ora.agent.tools.composed import executor
        from ora.agent.tools.composed.host import COMMITMENT_FILENAME, ExecutePort

        path = Path(self.control_root) / COMMITMENT_FILENAME
        if not path.is_file():
            raise ValueError("composed tool commitment is missing")
        stored = json.loads(path.read_text(encoding="utf-8"))
        commitment = executor.Commitment.from_json(stored)
        arguments = action["arguments"]
        if (
            commitment.call_id != action["action_id"]
            or commitment.observation_id != action["observation_id"]
            or commitment.observation_sha256 != action["bound_observation_sha256"]
            or (commitment.tool, commitment.version, commitment.arm_profile, commitment.compiled_sha256)
            != (arguments["tool"], arguments["version"], arguments["arm_profile"], arguments["compiled_sha256"])
        ):
            raise ValueError("composed tool action drifted from its committed call")
        binding = self._composed_binding(commitment.tool)
        if binding["compiled_sha256"] != commitment.compiled_sha256:
            raise ValueError("composed tool commitment is not the surface-bound version")
        tool = self._composed_library_tool(binding)
        if tool.program.digest != commitment.source_sha256:
            raise ValueError("composed tool source changed after the call")
        box: dict[str, Any] = {}
        port = ExecutePort(
            self, host_state=stored.get("host_state") or {}, capture=capture, latest=latest,
            fresh=fresh, binding=binding,
            completed=lambda: [
                {"stage": item.stage, "status": item.outcome["status"], "reason": item.outcome["reason"],
                 "measurements": item.outcome["measurements"]}
                for item in box["execution"].dispatched
            ],
        )
        box["execution"] = executor.Execution(
            tool, commitment, port.port(),
            receipt_context=lambda: {
                "fresh_observation_id": latest()[0].get("observation_id"),
                "held_object_offset": getattr(self, "last_held_object_offset", None),
            },
        )
        return box["execution"]
