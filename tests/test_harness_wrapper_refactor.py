"""Shared Harness-wrapper contracts and compatibility gates."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from ora.agent import (
    ActionProposal,
    LifecycleRequest,
    NoAction,
    PolicyRuntimeEvent,
    PolicyTurnInput,
    policy_outcome_from_mapping,
)
from ora.agent.context import ContextBundle, ContextSlot, ContextSource
from ora.agent.native.gateway import NativeToolGateway
from ora.environment import (
    EnvironmentDefinitionError, environment_definitions, resolve_runtime_descriptor,
)
from ora.orchestrator import EpisodeCoordinator
from ora.orchestrator.faults import classify_harness_fault
from ora.orchestrator.interaction import (
    EpisodeInteractionState,
    InteractionStateError,
)
from ora.agent.tools.robot import RobotToolRuntime


def test_policy_turn_is_immutable_and_runtime_projection_is_detached(tmp_path: Path) -> None:
    source = {"observation": {"pixels": [1, 2]}, "hidden": None}
    turn = PolicyTurnInput(
        task_id="task-1",
        task_instruction="  pick the object  ",
        context=source,
        image_path=tmp_path / "current.png",
        observation_id="observation-1",
        observation_step=1,
        budget={"remaining_actions": 3},
    )
    source["observation"]["pixels"].append(3)
    assert turn.task_instruction == "pick the object"
    assert tuple(turn.context["observation"]["pixels"]) == (1, 2)
    with pytest.raises(TypeError):
        turn.context["new"] = True  # type: ignore[index]

    runtime = turn.runtime_kwargs()
    runtime["context"]["observation"]["pixels"].append(9)
    assert tuple(turn.context["observation"]["pixels"]) == (1, 2)


def test_policy_outcomes_are_one_tagged_union() -> None:
    action = policy_outcome_from_mapping({
        "submitted": True,
        "attempts": 2,
        "action": {"action_id": "action-1"},
    })
    no_action = policy_outcome_from_mapping({
        "submitted": False, "termination": "policy_no_action", "attempts": 1,
    })
    lifecycle = policy_outcome_from_mapping({
        "submitted": False,
        "termination": "policy_finished_episode",
        "attempts": 1,
    })
    assert isinstance(action, ActionProposal) and action.kind == "action"
    assert isinstance(no_action, NoAction) and no_action.kind == "no_action"
    assert isinstance(lifecycle, LifecycleRequest) and lifecycle.kind == "lifecycle"


def test_interaction_state_enforces_receipt_then_fresh_observation(tmp_path: Path) -> None:
    state = EpisodeInteractionState("run-1", tmp_path / "interaction.json")
    state.start()
    state.publish_observation(observation_id="observation-0", step=0)
    state.begin_policy()
    state.record_policy_outcome(kind="action", model_calls=1, action_id="action-0")
    state.begin_execution()
    state.record_execution(receipt_id="receipt-0")
    assert state.phase == "fresh_observation_required"
    assert state.fresh_observation_required is True
    with pytest.raises(InteractionStateError):
        state.begin_policy()
    state.publish_observation(observation_id="observation-1", step=1)
    assert state.phase == "observation_ready"
    assert state.pending_action_id is None
    assert json.loads((tmp_path / "interaction.json").read_text())["sequence"] == 7


def test_post_dispatch_fault_is_unknown_and_never_replayed() -> None:
    fault = classify_harness_fault(
        ConnectionError("lost acknowledgement"),
        phase="executing",
        layer="environment",
        cause_event_id="event-7",
    )
    assert fault.code == "mutation_outcome_unknown"
    assert fault.world_state_certainty == "unknown"
    assert fault.retryable is False
    assert fault.requires_fresh_observation is True
    assert fault.recoverable_at == "operator"
    confirmed = classify_harness_fault(
        RuntimeError("camera unavailable"), phase="fresh_observation_required",
        layer="environment",
    )
    assert confirmed.world_state_certainty == "changed"
    assert confirmed.code == "post_mutation_observation_failure"
    assert confirmed.recoverable_at == "next_observation"


def test_environment_registry_is_explicit_and_descriptor_rejects_drift() -> None:
    definitions = environment_definitions()
    assert [item.environment_id for item in definitions] == [
        "libero_pro", "robodojo", "robotwin", "synthetic",
    ]
    descriptor = resolve_runtime_descriptor({
        "kind": "libero_pro",
        "transport": "local_rpc",
        "required_rlinf_revision": "rlinf-rev",
    }, action_codec="target_state_v4")
    assert descriptor.embodiment_id == "franka_panda"
    assert descriptor.controller == "OSC_POSE"
    assert descriptor.public_cameras == ("agentview", "wrist")
    assert descriptor.revisions == {"rlinf_revision": "rlinf-rev"}
    with pytest.raises(EnvironmentDefinitionError, match="identity drift"):
        resolve_runtime_descriptor(
            {"kind": "libero_pro", "transport": "local_rpc"},
            observed_public_context={
                "environment_id": "robotwin",
                "embodiment_id": "franka_panda",
                "robot_models": ["Panda"],
                "controller": "OSC_POSE",
                "public_cameras": ["agentview"],
            },
        )


def test_host_layers_do_not_import_process_isolated_simulator_backends() -> None:
    root = Path(__file__).resolve().parents[1]
    violations = []
    for package in (root / "ora/orchestrator", root / "viewer"):
        for path in package.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                module = (
                    node.module if isinstance(node, ast.ImportFrom) else None
                )
                names = (
                    [item.name for item in node.names]
                    if isinstance(node, ast.Import) else []
                )
                imported = [module] if module else names
                if any(
                    name.startswith("ora.environment.backends")
                    for name in imported
                ):
                    violations.append(path.relative_to(root).as_posix())
    assert violations == []

    assert not (root / "ora/agent/managed_coder.py").exists()
    assert not (root / "ora/agent/managed_session.py").exists()


def test_context_bundle_has_stable_order_sources_and_hash() -> None:
    task = ContextSource("episode_task", "user", 2, "protected", "exact task")
    system = ContextSource("harness_system", "system", 0, "protected", "system")
    bundle = ContextBundle(slots=(
        ContextSlot(task, "exact task"),
        ContextSlot(system, "system"),
    ))
    assert [item["_context_source"] for item in bundle.messages()] == [
        "harness_system", "episode_task",
    ]
    assert bundle.sha256() == ContextBundle(slots=bundle.slots).sha256()


def test_gateway_and_direct_calls_share_one_tool_runtime() -> None:
    class Runtime:
        def __init__(self) -> None:
            self.calls = []

        def definitions(self):
            return [{"name": "read", "description": "read", "inputSchema": {}}]

        def invoke(self, name, arguments):
            self.calls.append((name, arguments))
            return {"ok": True}

    runtime = Runtime()
    gateway = NativeToolGateway(runtime)
    called = gateway.execute("read", {"step": 1})
    assert gateway.delivered() == runtime.definitions()
    assert runtime.calls == [("read", {"step": 1})]
    assert called == {"ok": True}


def test_removed_compatibility_modules_are_not_shipped() -> None:
    root = Path(__file__).resolve().parents[1]
    assert EpisodeCoordinator.__module__ == "ora.orchestrator.coordinator"
    removed_paths = (
        root / "control/__init__.py",
        root / "environment/__init__.py",
        root / "ora/composition/__init__.py",
        root / "ora/environment/action.py",
        root / "ora/agent/active_memory.py",
        root / "ora/agent/file_memory.py",
        root / "ora/agent/memory.py",
        root / "ora/agent/recoverable_artifacts.py",
        root / "ora/policy/composition.py",
        root / "ora/policy/environment_context.py",
        root / "ora/agent/codex/__init__.py",
        root / "ora/agent/claude_code/__init__.py",
        root / "ora/agent/managed/__init__.py",
        root / "ora/agent/wire/__init__.py",
    )
    assert all(not path.exists() for path in removed_paths)
    fresh = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from ora.orchestrator.context_manifest import "
                "build_context_manifest; "
                "import ora.agent.native.gateway, "
                "ora.agent.providers.native, "
                "ora.agent.providers.profiles, "
                "ora.agent.composition, "
                "ora.agent.composition.environment_context"
            ),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert fresh.returncode == 0, fresh.stderr


def test_policy_runtime_event_freezes_trusted_payload() -> None:
    payload = {"receipt": {"consumed": True}}
    event = PolicyRuntimeEvent(kind="environment_receipt", payload=payload)
    payload["receipt"]["consumed"] = False
    assert event.payload["receipt"]["consumed"] is True
    mutable = event.mutable_payload()
    mutable["receipt"]["consumed"] = False
    assert event.payload["receipt"]["consumed"] is True


def test_mutation_proposal_is_discarded_when_evidence_write_fails(
    tmp_path: Path, monkeypatch,
) -> None:
    manual = tmp_path / "manual.md"
    manual.write_text("# Manual\n\n## Safety\nRead first.\n", encoding="utf-8")
    control = tmp_path / "control"
    control.mkdir()
    runtime = RobotToolRuntime(
        control_root=control,
        manual_path=manual,
        event_path=tmp_path / "events.jsonl",
        workspace_bounds_m={
            "x": [-1.0, 1.0], "y": [-1.0, 1.0], "z": [0.0, 2.0],
        },
        allowed_tools=["robot.submit_action"],
    )
    assert not hasattr(runtime, "call")

    def stage(_arguments):
        (control / "submitted-action.json").write_text("{}\n", encoding="utf-8")
        return {"accepted": True}

    runtime._submit = stage  # type: ignore[method-assign]
    monkeypatch.setattr(
        "ora.agent.tools.robot._append",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )
    with pytest.raises(OSError, match="disk full"):
        runtime.invoke("robot.submit_action", {"action": "set_gripper"})
    assert not (control / "submitted-action.json").exists()


def test_namespace_dependencies_and_coordinator_are_backend_neutral() -> None:
    root = Path(__file__).resolve().parents[1]
    forbidden = {
        root / "ora/agent": (
            "control", "environment", "ora.environment",
            "ora.orchestrator",
        ),
        root / "ora/environment": (
            "control", "ora.agent", "ora.orchestrator",
        ),
    }
    violations = []
    for package, prefixes in forbidden.items():
        for path in package.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                module = (
                    node.module if isinstance(node, ast.ImportFrom)
                    else None
                )
                names = (
                    [alias.name for alias in node.names]
                    if isinstance(node, ast.Import) else [module] if module else []
                )
                for name in names:
                    if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
                        violations.append(f"{path.relative_to(root)}:{node.lineno}:{name}")
    assert violations == []

    coordinator = (root / "ora/orchestrator/coordinator.py").read_text(
        encoding="utf-8"
    )
    assert "CodexPlanner" not in coordinator
    assert "NativePlanner" not in coordinator
    assert 'policy["kind"]' not in coordinator
    assert "environment.backends" not in coordinator

    native_session = (root / "ora/agent/native/session.py").read_text(
        encoding="utf-8"
    )
    native_loop = (root / "ora/agent/native/tool_loop.py").read_text(
        encoding="utf-8"
    )
    assert "def _run_turn_loop" not in native_session
    assert "provider.complete" in native_loop
    assert "gateway.execute" in native_loop

    deprecated_imports = (
        "control",
        "environment",
        "ora.agent_session",
        "ora.agent.active_memory",
        "ora.agent.file_memory",
        "ora.agent.recoverable_artifacts",
        "ora.app_server_session",
        "ora.capabilities",
        "ora.composition",
        "ora.config_inspection",
        "ora.context",
        "ora.managed_policy_trace",
        "ora.mcp_server",
        "ora.memory",
        "ora.metadata",
        "ora.native_context",
        "ora.native_planner",
        "ora.agent.native.planner",
        "ora.native_provider",
        "ora.native_provider_profiles",
        "ora.openeta_tools",
        "ora.planner",
        "ora.policy_session",
        "ora.policy.composition",
        "ora.policy.environment_context",
        "ora.prompt",
        "ora.prompt_qualification",
        "ora.state",
        "ora.tool_runtime_factory",
        "ora.tool_surface",
        "ora.tools",
        "ora.trajectory",
        "ora.two_view_geometry",
    )
    maintained_roots = (
        root / "ora/agent",
        root / "ora/contracts",
        root / "ora/environment",
        root / "ora/orchestrator",
        root / "experiments",
        root / "viewer",
        root / "tests",
    )
    deprecated_uses = []
    for maintained_root in maintained_roots:
        for path in maintained_root.rglob("*.py"):
            if path == Path(__file__).resolve():
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                elif isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    names = [node.value]
                for name in names:
                    is_module_string = isinstance(node, ast.Constant)
                    # A wire schema is not a deprecated Python module. The
                    # public namespace migration makes their prefixes overlap.
                    if is_module_string and name == "ora.context.v1":
                        continue
                    if any(
                        name.startswith(prefix + ".")
                        or (not is_module_string and name == prefix)
                        for prefix in deprecated_imports
                    ):
                        deprecated_uses.append(
                            f"{path.relative_to(root)}:{node.lineno}:{name}"
                        )
    assert deprecated_uses == []

    removed_source_paths = [
        path for path in deprecated_imports
        if (root / (path.replace(".", "/") + ".py")).is_file()
        or (root / path.replace(".", "/") / "__init__.py").is_file()
    ]
    assert removed_source_paths == []
    import ora.agent.composition as composition

    assert not hasattr(composition, "CHANNELS")
    wrapper = (root / "scripts/ora").read_text(encoding="utf-8")
    assert "-m ora.orchestrator.cli" in wrapper
    assert "-m control.cli" not in wrapper
