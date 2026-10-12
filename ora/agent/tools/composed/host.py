"""What the SDK's queries mean inside the manipulation tool runtime.

:class:`PlanPort` serves ``plan()`` at tool-call time from the current
observation; :class:`ExecutePort` serves ``execute()`` inside the trusted
executor from the observations it captures. Grasp queries reuse the
closed-skill implementation itself -- candidate lookup and freshness, the
prepared-grasp commitment, the pre-commit motion probe and the verify-stage
measurement -- so a composed grasp and ``grasp_execute`` cannot disagree about
geometry or evidence. The full native commitment never crosses into Starlark:
the body sees the arm, candidate, source reference and stage poses, and the
host keeps the rest in ``host_state``, which the tool runtime persists with
the commitment and the executor reloads.

``sdk.read`` is the runtime's ``composed_read`` in both ports; during
execution it reads a fresh observation. A ``tool`` stage's nested tool plans
through a :class:`PlanPort` over that same fresh observation (made current for
the call) and executes through its own :class:`ExecutePort`.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ora.agent.tools import localization, skill_driver, skill_retention
from ora.agent.tools.tool_errors import ToolError

from .executor import Port

COMMITMENT_FILENAME = "composed-tool-commitment.json"
GRASP_TARGET = "grasp_target"
GRASP_POSES = ("pregrasp", "approach", "lift")


def robot_state(observation: Mapping[str, Any], arm: str) -> dict[str, Any]:
    """Deployable public state of one arm in an observation."""

    effectors = observation.get("proprioception", {}).get("end_effectors", {})
    record = effectors.get(arm) if isinstance(effectors, Mapping) else None
    if not isinstance(record, Mapping):
        raise ValueError(f"the observation publishes no end effector for arm {arm!r}")
    pose = record.get("pose_xyz_xyzw")
    position = list(pose[:3]) if isinstance(pose, list) and len(pose) == 7 else None
    orientation = list(pose[3:]) if isinstance(pose, list) and len(pose) == 7 else None
    command = record.get("gripper")
    return {
        "arm": arm,
        "tcp_position_m": position,
        "tcp_orientation_xyzw": orientation,
        "gripper_command": float(command) if isinstance(command, (int, float)) and not isinstance(command, bool) else None,
        "gripper_opening_m": localization.measured_jaw_separation_m(record),
    }


def refusal(error: ToolError) -> dict[str, Any]:
    reported = error.as_dict()
    return {
        "code": reported["code"], "message": reported["message"], "field": reported["field"],
        "retryable": bool(reported["retryable"]), "constraints": dict(reported["constraints"] or {}),
    }


def _public_target(commitment: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": commitment["candidate_id"],
        "arm": commitment["arm"],
        "source_ref": commitment.get("source_ref"),
        "poses": {name: copy.deepcopy(commitment["poses"][name]) for name in GRASP_POSES},
    }


class PlanPort:
    """Call-time queries over the tool runtime's current observation."""

    def __init__(self, host: Any, *, control: Mapping[str, Any], root: Path, observation: Mapping[str, Any],
                 binding: Mapping[str, Any] | None = None):
        self.host = host
        self.control = control
        self.root = root
        self.observation = observation
        self.binding = binding
        self.host_state: dict[str, Any] = {}

    def port(self) -> Port:
        return Port(queries={
            "sdk.robot_state": lambda arm: robot_state(self.observation, arm),
            "sdk.grasp.target": self.grasp_target,
            "sdk.grasp.precommit_probe": self.precommit_probe,
            "sdk.read": lambda tool, arguments: self.host.composed_read(tool, arguments),
        })

    def grasp_target(self, candidate_id: Any) -> dict[str, Any]:
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("candidate_id must be a non-empty string")
        try:
            # The prepared-grasp commitment begin_skill derives: candidate lookup,
            # freshness, workspace/orientation gate, localization, semantics and
            # retention reference, always with the pregrasp and lift targets.
            commitment = skill_driver._commit_geometry(
                self.host, skill="grasp_execute", candidate_id=candidate_id,
                observation_id=str(self.control["observation_id"]), control=self.control, root=self.root,
                observation=self.observation, prepared=True,
            )
        except ToolError as error:
            return {"ok": False, "refusal": refusal(error)}
        self.host_state[GRASP_TARGET] = commitment
        return {"ok": True, "target": _public_target(commitment)}

    def precommit_probe(self, target: Any) -> dict[str, Any]:
        from ora.agent.tools import grasp_feasibility

        commitment = self.host_state.get(GRASP_TARGET)
        if commitment is None or not isinstance(target, Mapping) or target.get("candidate_id") != commitment["candidate_id"]:
            raise ValueError("sdk.grasp.precommit_probe needs the target sdk.grasp.target returned")
        surface = self.host.surface
        if not (surface.supports(skill_retention.CAPABILITY) and surface.supports(grasp_feasibility.CAPABILITY)):
            return {"ok": True, "checked": False, "poses": _public_target(commitment)["poses"], "evidence": None}
        try:
            evidence = grasp_feasibility.check_pregrasp(self.host, commitment)
        except ToolError as error:
            return {"ok": False, "refusal": refusal(error)}
        commitment["precommit_motion_check"] = evidence
        return {"ok": True, "checked": True, "poses": _public_target(commitment)["poses"], "evidence": evidence}


class ExecutePort:
    """Execution-time queries inside the trusted executor."""

    def __init__(
        self,
        host: Any,
        *,
        host_state: Mapping[str, Any],
        capture: Callable[[], tuple[dict[str, Any], Path]],
        latest: Callable[[], tuple[dict[str, Any], Path]],
        completed: Callable[[], Sequence[Mapping[str, Any]]],
        fresh: Callable[[], tuple[dict[str, Any], Path]] | None = None,
        binding: Mapping[str, Any] | None = None,
    ):
        self.host = host
        self.host_state = dict(host_state)
        self.capture_observation = capture
        self.latest = latest
        self.fresh = fresh or latest
        self.completed = completed
        self.binding = binding
        self.captures: dict[str, dict[str, Any]] = {}

    def port(self) -> Port:
        return Port(queries={
            "sdk.robot_state": lambda arm: robot_state(self.latest()[0], arm),
            "sdk.capture": self.capture,
            "sdk.check.grasp": self.check_grasp,
            "sdk.ask": self.ask,
            "sdk.read": self.read,
        }, nested=self.nested if self.binding is not None else None)

    def read(self, tool: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        observation, root = self.fresh()
        return self.host.composed_read(tool, arguments, observation=observation, root=root)

    def nested(self, ref: str, arguments: Mapping[str, Any], *, call_id: str) -> tuple[Any, Any, Port, Any]:
        """Plan a tool stage's pinned tool on the fresh observation; raise ToolError when it refuses."""

        from . import executor
        from .tunables import load_tunables

        assert self.binding is not None
        tool = self.host._composed_nested_tool(self.binding, ref)
        observation, root = self.fresh()
        with self.host._observation_override(observation, root) as control:
            plan = PlanPort(self.host, control=control, root=Path(root), observation=observation, binding=self.binding)
            commitment, _result = executor.submit(
                tool, arm_profile=self.binding["arm_profile"],
                arguments={"rationale": f"step of {self.binding['tool']}", **dict(arguments)},
                port=plan.port(), tunables=load_tunables(), call_id=call_id,
                observation_id=str(control["observation_id"]), observation_sha256=str(control["observation_sha256"]),
            )
        box: dict[str, Any] = {}
        port = ExecutePort(
            self.host, host_state=plan.host_state, capture=self.capture_observation, latest=self.latest,
            fresh=self.fresh, binding=self.binding,
            completed=lambda: [
                {"stage": item.stage, "status": item.outcome["status"], "reason": item.outcome["reason"],
                 "measurements": item.outcome["measurements"]}
                for item in box["execution"].dispatched
            ] if "execution" in box else [],
        )
        return tool, commitment, port.port(), lambda execution: box.__setitem__("execution", execution)

    def ask(self, question: str, images: Sequence[str] = (), values: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """One question-bank question to the runtime model, as a structured, checked answer.

        Only an admitted question is asked, and only through the host's
        ``model_judge``; without one the answer is ``unavailable`` and the
        body must carry on without it.
        """

        from .questions import QuestionError, load_question

        try:
            asked = load_question(question)
            prompt = asked.render(dict(values or {}), list(images))
        except QuestionError as exc:
            raise ValueError(str(exc)) from None
        if not asked.admitted:
            return {"status": "unavailable", "question": asked.ref, "answer": None, "reason": "question_not_admitted"}
        judge = getattr(self.host, "model_judge", None)
        if not callable(judge):
            return {"status": "unavailable", "question": asked.ref, "answer": None, "reason": "no_runtime_model_binding"}
        latest = self.latest()[0]
        return asked.answer(judge(asked, prompt, [{"role": role, "observation_id": latest.get("observation_id")}
                                                  for role in images], dict(values or {})))

    def capture(self, role: str) -> dict[str, Any]:
        fresh, root = self.capture_observation()
        baseline = skill_retention.capture_baseline(fresh, root)
        baseline["role"] = role
        self.captures[role] = baseline
        return {key: baseline[key] for key in ("role", "observation_root", "observation_id", "observation_sha256")}

    def check_grasp(self, target: Any, stage: str, hold: bool = False) -> dict[str, Any]:
        commitment = self.host_state.get(GRASP_TARGET)
        if commitment is None or not isinstance(target, Mapping) or target.get("candidate_id") != commitment["candidate_id"]:
            raise ValueError("sdk.check.grasp needs the committed grasp target")
        fresh, root = self.capture_observation()
        measured = skill_driver._measure(self.host, "verify", fresh, commitment, self.completed())
        baseline = self.captures.get("post_close_before_lift")
        if baseline is not None:
            measured["verification"]["lift_probe"] = skill_retention.measure(
                self.host, commitment={**commitment, "lift_baseline": baseline},
                observation=fresh, observation_root=root,
            )
        if hold:
            from .hold import hold_verdict

            measured["verification"]["hold"] = hold_verdict(measured["verification"])
        return {
            "status": measured["status"], "reason": measured["reason"],
            "measurements": measured["measurements"], "verification": measured["verification"],
        }
