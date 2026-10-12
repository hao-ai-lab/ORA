"""Layer-2 contract tests: drive a composed tool through scripted outcomes.

:func:`run` plays one call end to end against :class:`FakeWorld`: ``submit``,
then every action the body requests, each lowered to its motion v3 envelope
and validated by the same contract the trusted executor uses, then resolved
with a scripted outcome. :func:`scenarios` enumerates, for every action the
nominal run dispatches, each non-executed outcome, an operator interrupt
before it and a terminal environment after it. :func:`violations` lists every
invariant a finished run breaks; an admitted tool has none in any scenario.

``sdk.read`` answers ``{"ok": False}`` unless a test scripts it through
``FakeWorld.reads`` (``{tool: result or callable(arguments)}``), and a ``tool``
stage runs its nested tool against the same fake world (library resolution
next to the calling tool), so nested dispatches are scripted like any other.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ora.agent.tools import skill_receipt

from . import sdk
from .executor import (
    STOP_ENVIRONMENT_TERMINAL, STOP_OPERATOR_INTERRUPT, ActionRequest, Commitment, Execution, Finished, Port,
    submit,
)
from .library import ComposedTool
from .tunables import TunableTable

OBSERVATION_ID = "obs-contract-0001"
OBSERVATION_SHA256 = "c" * 64


@dataclass
class FakeWorld:
    """Public robot state that follows the directives it executes."""

    state: dict[str, dict[str, Any]] = field(default_factory=lambda: {
        arm: {
            "arm": arm,
            "tcp_position_m": [0.3, 0.2 if arm == "left" else -0.2, 0.25],
            "tcp_orientation_xyzw": [0.0, 1.0, 0.0, 0.0],
            "gripper_directive": "open",
        }
        for arm in ("left", "right")
    })
    queries: list[str] = field(default_factory=list)
    #: Query functions by SDK symbol that replace the defaults below.
    extra: dict[str, Any] = field(default_factory=dict)
    #: Scripted sdk.read results by read tool: a result dict, or a callable of the arguments.
    reads: dict[str, Any] = field(default_factory=dict)
    captures: int = 0
    #: The tool whose library root nested refs resolve next to (set by run()).
    near: Any = None

    def port(self) -> Port:
        def robot_state(arm: str) -> Mapping[str, Any]:
            self.queries.append(arm)
            return dict(self.state[arm])

        def grasp_target(candidate_id: str) -> Mapping[str, Any]:
            contact = {"position_m": [0.2, 0.1, 0.9], "orientation_xyzw": [0.0, 1.0, 0.0, 0.0]}
            return {"ok": True, "target": {
                "candidate_id": candidate_id, "arm": "right", "source_ref": "target-1",
                "poses": {
                    "pregrasp": {**contact, "position_m": [0.2, 0.1, 0.96]},
                    "approach": contact,
                    "lift": {**contact, "position_m": [0.2, 0.1, 0.93]},
                },
            }}

        def precommit_probe(target: Mapping[str, Any]) -> Mapping[str, Any]:
            return {"ok": True, "checked": True, "poses": target["poses"],
                    "evidence": {"probed_stages": ["pregrasp", "approach"], "summary": "reachable"}}

        def capture(role: str) -> Mapping[str, Any]:
            self.captures += 1
            return {"role": role, "observation_root": f"fake/{self.captures}",
                    "observation_id": f"obs-fake-{self.captures}", "observation_sha256": "e" * 64}

        def check_grasp(target: Mapping[str, Any], stage: str, hold: bool = False) -> Mapping[str, Any]:
            verification: dict[str, Any] = {"strength": "unavailable"}
            if hold:
                from .hold import hold_verdict

                verification["hold"] = hold_verdict(verification)
            return {"status": "executed", "reason": "verified_against_fake",
                    "measurements": [{"quantity": "closure_strength", "value": "unavailable"}],
                    "verification": verification}

        def ask(question: str, images: Any = (), values: Any = None) -> Mapping[str, Any]:
            return {"status": "unavailable", "question": question, "answer": None, "reason": "no_runtime_model_binding"}

        def read(tool: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
            scripted = self.reads.get(tool)
            if scripted is None:
                return {"ok": False, "error": {"code": "unavailable_in_contract_test", "message": tool,
                                               "field": None, "retryable": False}}
            return {"ok": True, "result": scripted(arguments) if callable(scripted) else scripted}

        return Port(queries={
            "sdk.robot_state": robot_state, "sdk.grasp.target": grasp_target,
            "sdk.grasp.precommit_probe": precommit_probe, "sdk.capture": capture,
            "sdk.check.grasp": check_grasp, "sdk.ask": ask, "sdk.read": read, **self.extra,
        }, nested=self.nested)

    def nested(self, ref: str, arguments: Mapping[str, Any], *, call_id: str) -> tuple[Any, Any, Port]:
        from .library import resolve_tool
        from .tunables import load_tunables

        tool = resolve_tool(ref, near=self.near)
        arm_profile = "dual_arm" if "dual_arm" in tool.declaration.arm_profiles else tool.declaration.arm_profiles[0]
        commitment, _result = submit(
            tool, arm_profile=arm_profile, arguments={"rationale": "nested step", **dict(arguments)},
            port=self.port(), tunables=load_tunables(), call_id=call_id, observation_id=OBSERVATION_ID,
            observation_sha256=OBSERVATION_SHA256,
        )
        return tool, commitment, self.port()

    def apply(self, request: Mapping[str, Any]) -> dict[str, Any]:
        arm = request["arm"]
        if request["primitive"] == "gripper":
            self.state[arm]["gripper_directive"] = request["directive"]
        else:
            self.state[arm]["tcp_position_m"] = list(request["pose"]["position_m"])
            self.state[arm]["tcp_orientation_xyzw"] = list(request["pose"]["orientation_xyzw"])
        return {"status": "executed", "reason": "executed", "measurements": []}


@dataclass(frozen=True)
class Scenario:
    name: str
    outcomes: Mapping[int, str] = field(default_factory=dict)  # action index -> non-executed status
    interrupt_before: int | None = None
    terminal_after: int | None = None


@dataclass
class Run:
    scenario: Scenario
    commitment: Commitment
    result: Mapping[str, Any]
    finished: Finished
    requests: list[ActionRequest]
    envelopes: list[Mapping[str, Any]]


def run(
    tool: ComposedTool,
    *,
    arm_profile: str,
    arguments: Mapping[str, Any],
    tunables: TunableTable,
    scenario: Scenario = Scenario("nominal"),
    world: FakeWorld | None = None,
    outcome: Callable[[ActionRequest, FakeWorld], Mapping[str, Any] | None] | None = None,
) -> Run:
    world = world or FakeWorld()
    world.near = tool.directory.parent.parent
    commitment, result = submit(
        tool, arm_profile=arm_profile, arguments=arguments, port=world.port(), tunables=tunables,
        call_id="action-contract", observation_id=OBSERVATION_ID, observation_sha256=OBSERVATION_SHA256,
    )
    execution = Execution(tool, commitment, world.port())
    requests: list[ActionRequest] = []
    envelopes: list[Mapping[str, Any]] = []
    while True:
        step = execution.next()
        if isinstance(step, Finished):
            finished = step
            break
        if scenario.interrupt_before == step.index:
            finished = execution.stop(STOP_OPERATOR_INTERRUPT)
            break
        requests.append(step)
        envelopes.append(sdk.lower_motion_v3(
            step.request, action_id=step.action_id, observation_id=OBSERVATION_ID,
            observation_sha256=OBSERVATION_SHA256,
        ))
        scripted = scenario.outcomes.get(step.index)
        if scripted is not None:
            resolved = {"status": scripted, "reason": f"scripted_{scripted}", "measurements": []}
        else:
            resolved = (outcome(step, world) if outcome else None) or world.apply(step.request)
        execution.resolve(resolved)
        if scenario.terminal_after == step.index:
            finished = execution.stop(STOP_ENVIRONMENT_TERMINAL)
            break
    return Run(scenario, commitment, result, finished, requests, envelopes)


def scenarios(nominal: Run) -> list[Scenario]:
    """Every single-fault variation of a nominal run."""

    out = [Scenario("nominal")]
    for index in range(len(nominal.requests)):
        for status in ("not_reached", "unavailable", "refused"):
            out.append(Scenario(f"action{index}_{status}", outcomes={index: status}))
        out.append(Scenario(f"interrupt_before_action{index}", interrupt_before=index))
        out.append(Scenario(f"terminal_after_action{index}", terminal_after=index))
    return out


def violations(tool: ComposedTool, done: Run) -> list[str]:
    declaration = tool.declaration
    finished = done.finished
    receipt = finished.receipt
    found: list[str] = []

    def expect(condition: bool, message: str) -> None:
        if not condition:
            found.append(f"{done.scenario.name}: {message}")

    expect(receipt["schema"] == skill_receipt.SCHEMA, "receipt schema")
    expect(receipt["status"] in skill_receipt.TERMINAL_STATUSES, "receipt status vocabulary")
    expect(receipt["rationale"] == done.commitment.rationale, "rationale echoed into the receipt")
    declared = [item.name for item in declaration.stages]
    names = [item["stage"] for item in finished.stages]
    expect(len(set(names)) == len(names) and all(name in declared for name in names),
           "receipt entries name distinct declared stages")
    expect([declared.index(name) for name in names] == sorted(declared.index(name) for name in names),
           "receipt entries follow declared order")
    expect(all(item["status"] in skill_receipt.STAGE_STATUSES for item in finished.stages), "stage status vocabulary")
    expect(len(finished.dispatched) <= declaration.budgets.primitive_actions, "dispatch budget")
    expect(receipt["cost"]["robot_decisions"] == 1
           and receipt["cost"]["dispatched_stage_actions"] == len(finished.dispatched),
           "cost counts one decision and every dispatched action")
    expect(all(item.action_id.split(":")[-1].split(".")[0] == f"{declared.index(item.stage):02d}-{item.stage}"
               for item in finished.dispatched), "child action ids carry the declared stage index")
    expect(all(item.stage in declaration.stage_primitives for item in finished.dispatched), "actions name declared stages")
    expect(receipt["delivery"]["body_bytes"] <= skill_receipt.RECEIPT_BUDGET_BYTES, "receipt byte budget")
    expect(receipt["delivery"]["stages_with_elided_measurements"] == 0, "no measurement elided")
    expect(receipt["composed_tool"]["compiled_sha256"] == done.commitment.compiled_sha256, "compiled identity bound")
    expect(len(done.envelopes) == len(finished.dispatched), "every dispatched action was lowered and validated")

    if finished.status == "completed":
        expect(finished.stop_reason == "sequence_completed", "a completed run reports sequence_completed")
        last = {item.stage: item.outcome["status"] for item in finished.dispatched
                if declaration.stage_primitives.get(item.stage) != "tool"}
        expect(all(status == "executed" for status in last.values()),
               "completed although a stage's last attempt did not execute")
    else:
        expect(bool(finished.stop_reason), "a stopped run names its stop reason")
        later = [item for item in finished.stages if item.get("action_id") is None and item["status"] == "skipped"]
        expect(len(later) <= 1, "at most the pending stage is reported skipped")
    scenario = done.scenario
    if scenario.interrupt_before is not None:
        expect(finished.stop_reason == STOP_OPERATOR_INTERRUPT and finished.status == "aborted",
               "operator interrupt aborts")
        expect(len(finished.dispatched) == scenario.interrupt_before, "nothing dispatched after the interrupt")
        expect(finished.stages[-1]["status"] == "skipped" and finished.stages[-1]["reason"] == STOP_OPERATOR_INTERRUPT,
               "the stage that did not start is reported skipped by the interrupt")
    if scenario.terminal_after is not None:
        expect(finished.status == "terminated" and finished.stop_reason == STOP_ENVIRONMENT_TERMINAL,
               "a terminal environment terminates the sequence")
        expect(len(finished.dispatched) == scenario.terminal_after + 1, "nothing dispatched after the terminal")
        last = finished.dispatched[-1]
        record = next(item for item in finished.stages if item.get("action_id") == last.action_id)
        expect(record["status"] == "terminated", "the stage the episode ended during is reported terminated")
    for index, status in scenario.outcomes.items():
        if index < len(finished.dispatched):
            stage = finished.dispatched[index].stage
            if declaration.stage_primitives.get(stage) == "tool":
                continue  # a tool stage reports the nested tool's summary, not one of its actions
            record = next(item for item in finished.stages if item["stage"] == stage)
            retried = any(item.stage == stage for item in finished.dispatched[index + 1:])
            expect(retried or record["status"] == status, f"stage {stage} reports the {status} outcome")
    return found
