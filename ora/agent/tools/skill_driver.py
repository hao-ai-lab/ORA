"""The tool-runtime half of a closed skill: commit, drive, verify, release.

Where this sits
---------------

``skill_ledger`` is the state; this module is the driver that pumps it, and
``orchestrator/action_executor.execute_skill_sequence`` is the loop that calls
the driver. The split is:

* the **tool runtime** (here) owns the geometry, because it holds the candidate
  artifacts and their observation binding, and it owns the ledger, because the
  ledger lives beside its other control state;
* the **trusted executor** owns the stage loop and the dispatch boundary; and
* the **coordinator** owns the turn, so it is the only thing that can execute
  several Supervisor actions without opening one.

It cannot be only the coordinator: the coordinator is Agent-neutral and has no
candidate geometry, and driving stages from there would write actions the
AgentSession never proposed -- ``record_environment_receipt`` refuses a receipt
whose ``action_id`` is not the session's ``last_action_id``, so every driven
stage would end the episode. It cannot be only the session loop either: the
Native turn loop returns as soon as a mutation submits, so a driver there would
own manipulation semantics inside ``agent/native/``, which owns the provider
loop and not the tool contract, and it would not run for a managed runtime
sharing the same surface.

How a committed sequence resumes
--------------------------------

It mostly does not need to. The stages between the decision and the receipt all
run inside one coordinator execution with no Agent turn between them, so the
only thing that genuinely crosses a turn boundary is ``place_execute``'s
requested pre-release stop. That stop ends the submitted action; the policy sees
the ordinary post-action receipt and fresh observation -- the close view it
asked for is that observation, not a second image the skill attaches -- and
``place_resume`` re-submits the same commitment with the completed prefix
skipped. Resuming takes ``continue`` or ``abandon`` and nothing else, because
the stop returned a view, not the candidate choice back.

Not wedging
-----------

``build_skill_receipt`` releases the ledger on terminal executor paths,
including before a raised backend fault propagates. A requested pre-release
stop retains the commitment until the policy continues or abandons it; other
motion calls must wait for that decision. That covers everything the
executor can see; it does not cover a sequence that died with the executor --
a crashed worker, a killed run -- which would otherwise leave a ``running``
ledger no later call could clear. So the ledger is also reconciled by whatever
call comes next, in the holding gate's pending/reconcile shape: a ledger whose
committed observation is no longer the current one, with no action still
pending, has already been decided somewhere this process cannot see and is
released as abandoned. A skill that can wedge is worse than three explicit
calls, so neither half of that is optional.
"""

from __future__ import annotations

from ora.agent.tools.skill_geometry import (
    _pose, _lifted, validate_prepared_targets,
)

import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ora.contracts import (
    SKILL_SEQUENCE_ACTION_SCHEMA, TARGET_STATE_ACTION_SCHEMA, MOTION_ACTION_V3_SCHEMA,
    validate_runtime_action,
)

from . import (
    jaw_hold, localization, skill_ledger, skill_receipt, skill_retention, target_identity, tool_names,
)
from .skill_candidates import _advertised, _grasp_candidate, _place_candidate
from .robot import ToolError


#: The ToolSurface capability a profile must declare to advertise these tools.
#: It names what the driver actually needs: one public end effector, the
#: selected motion envelope for each backend, and the two planners
#: whose candidates the skills consume.
CAPABILITY = "closed_manipulation_skills_v1"

LEDGER_FILENAME = "manipulation-skill-ledger.json"

#: The committed stage targets, written once at the moment the policy decided
#: and replayed for every stage afterwards. They are stored rather than
#: recomputed because the stages run while the action is still pending, when
#: `_current()` correctly refuses to serve a policy read, and because replaying
#: what was committed is what "the policy is asked once" means.
COMMITMENT_FILENAME = "manipulation-skill-commitment.json"

#: Jaw directives in the v4 envelope: 0 is closed, 1 is the open profile. These
#: are the same two values `_atomic_gripper` submits.
GRIPPER_CLOSED = 0.0
GRIPPER_OPEN = 1.0

CLAIM_BOUNDARY = (
    "one committed sequence of trusted executions and the public measurements "
    "taken after it; motion arrival and jaw separation beside visible extents "
    "are evidence, not grasp, placement, retention or official task success"
)


# --------------------------------------------------------------------------
# Ledger storage and reconciliation
# --------------------------------------------------------------------------

def ledger_path(host: Any) -> Path:
    return Path(host.control_root) / LEDGER_FILENAME


def load_ledger(host: Any) -> dict[str, Any] | None:
    path = ledger_path(host)
    if not path.is_file():
        return None
    return skill_ledger.validate_state(
        json.loads(path.read_text(encoding="utf-8"))
    )


def store_ledger(host: Any, state: Mapping[str, Any] | None) -> None:
    path = ledger_path(host)
    if state is None:
        path.unlink(missing_ok=True)
        return
    _write_json(path, state)


def load_commitment(host: Any) -> dict[str, Any] | None:
    path = Path(host.control_root) / COMMITMENT_FILENAME
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def store_commitment(host: Any, value: Mapping[str, Any]) -> None:
    _write_json(Path(host.control_root) / COMMITMENT_FILENAME, value)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def reconcile(host: Any) -> dict[str, Any] | None:
    """Release a ledger whose sequence can no longer report.

    Called before committing or resuming a skill, which is where a stuck ledger
    would otherwise block the policy. A live sequence is one whose committed
    observation is still the current one, or one whose action is still pending
    execution. Anything else has already been decided somewhere this process
    cannot see, and holding it would block every later call with a decision the
    policy cannot take back.
    """

    state = load_ledger(host)
    # A stopped sequence is *waiting*, not stalled, and it waits across exactly
    # the observation change this function otherwise treats as proof that
    # nothing is coming: the stop ends its action, the coordinator publishes the
    # close view, and the policy resumes on that turn. Releasing it here would
    # discard the decision at the one moment it is meant to be picked up.
    if state is None or state["phase"] in {"completed", "aborted", "terminated", "stopped"}:
        return state
    control_path = Path(host.control_root) / "current-observation.json"
    if not control_path.is_file():
        return state
    control = json.loads(control_path.read_text(encoding="utf-8"))
    pending = (Path(host.control_root) / "submitted-action.json").is_file()
    if pending or control.get("observation_id") == state["observation_id"]:
        return state
    released, _transition = skill_ledger.abandon(
        state, "sequence_did_not_report_before_the_next_observation",
    )
    store_ledger(host, released)
    return released


# --------------------------------------------------------------------------
# Paused sequence mutation admission
# --------------------------------------------------------------------------

def ensure_paused_mutation_available(host: Any, tool: str) -> None:
    """Keep a paused placement's release decision exclusive across tools."""

    if not host.surface.supports(CAPABILITY) or tool == "place_resume":
        return
    state = load_ledger(host)
    if state is None or state["phase"] != "stopped":
        return
    recovery_tool = _advertised(host, "place_resume")
    raise ToolError(
        f"{tool} is unavailable while placement is paused before release; use "
        f"{recovery_tool} with decision continue to finish it, or abandon before "
        "using another motion or VLA tool. Read-only inspection remains available.",
        code="skill_sequence_paused", retryable=False,
        constraints={
            "skill": state["skill"], "phase": state["phase"],
            "pending_stage": skill_ledger.pending_stage(state),
            "recovery_tool": recovery_tool,
            "decisions": ["continue", "abandon"],
        },
    )


# --------------------------------------------------------------------------
# Committing one skill call
# --------------------------------------------------------------------------

def _require_capability(host: Any) -> None:
    if not host.surface.supports(CAPABILITY):
        raise ToolError(
            f"this ToolSurface does not declare {CAPABILITY}",
            code="tool_surface_capability_mismatch", retryable=False,
        )


def begin_skill(host: Any, skill: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Commit one skill call and submit its sequence as a single action."""

    _require_capability(host)
    host._ensure_mutation_available(skill)
    allowed = {"candidate_id", "rationale"} | (
        {"stop_before_commit"} if skill == "place_execute" else set()
    )
    if set(arguments) - allowed or not {"candidate_id", "rationale"} <= set(arguments):
        raise ToolError(
            f"{skill} requires candidate_id and rationale",
            code="invalid_arguments",
        )
    skill_receipt.validate_rationale(arguments["rationale"])  # before any lookup or probe
    active = reconcile(host)
    if active is not None and active["phase"] not in {"completed", "aborted", "terminated"}:
        raise ToolError(
            "a committed skill sequence is already in flight; resume or abandon "
            "it before starting another",
            code="skill_sequence_active", retryable=False,
            constraints={
                "skill": active["skill"], "phase": active["phase"],
                "pending_stage": skill_ledger.pending_stage(active),
            },
        )
    control, root, observation, _pose_state, _gripper = host._current_grip_state()
    observation_id = str(control["observation_id"])
    commitment = _commit_geometry(
        host, skill=skill, candidate_id=str(arguments["candidate_id"]),
        observation_id=observation_id, control=control, root=root,
        observation=observation,
    )
    from . import grasp_feasibility
    if (skill == "grasp_execute"
            and host.surface.supports(skill_retention.CAPABILITY)
            and host.surface.supports(grasp_feasibility.CAPABILITY)):
        commitment["precommit_motion_check"] = grasp_feasibility.check_pregrasp(host, commitment)
    state = skill_ledger.begin(
        skill=skill, candidate_id=str(arguments["candidate_id"]),
        rationale=str(arguments["rationale"]),
        observation_id=observation_id,
        observation_sha256=str(control["observation_sha256"]),
        stop_before_commit=bool(arguments.get("stop_before_commit", False)),
        prepared_grasp=host.surface.supports(skill_retention.CAPABILITY),
    )
    store_ledger(host, state)
    store_commitment(host, commitment)
    return _submit(host, control, state, resume_from=0, commitment=commitment)


def resume_skill(host: Any, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Continue or abandon a sequence stopped for its pre-release view."""

    _require_capability(host)
    decision = arguments.get("decision")
    if set(arguments) != {"decision"} or decision not in {"continue", "abandon"}:
        raise ToolError(
            "place_resume requires decision continue or abandon",
            code="invalid_arguments", field="decision", received=decision,
            constraints={"enum": ["continue", "abandon"]},
        )
    state = reconcile(host)
    if state is None or state["phase"] != "stopped":
        raise ToolError(
            "no committed sequence is waiting at its pre-release stop",
            code="skill_ledger_state", retryable=False,
        )
    if decision == "abandon":
        released, transition = skill_ledger.abandon(state, "policy_abandoned_before_release")
        store_ledger(host, released)
        return {
            "success": True, "status": "aborted", "transition": transition,
            "skill": released["skill"], "candidate_id": released["candidate_id"],
            "released": True, "claim_boundary": CLAIM_BOUNDARY,
        }
    host._ensure_mutation_available("place_resume")
    commitment = load_commitment(host)
    if commitment is None or commitment.get("candidate_id") != state["candidate_id"]:
        raise ToolError(
            "the stopped sequence's committed targets are unavailable; abandon it "
            "and plan again",
            code="skill_commitment_unavailable", retryable=False,
        )
    control, _root, _observation, _pose_state, _gripper = host._current_grip_state()
    stop_observation_id = commitment.get("stop_observation_id")
    if not stop_observation_id or stop_observation_id != control["observation_id"]:
        raise ToolError(
            "the current observation no longer matches the placement's pre-release "
            f"stop; use {_advertised(host, 'place_resume')} with decision abandon "
            "and plan again from the current observation",
            code="skill_resume_observation_mismatch", retryable=False,
            constraints={
                "stop_observation_id": stop_observation_id,
                "current_observation_id": control["observation_id"],
                "recovery_tool": _advertised(host, "place_resume"),
                "decisions": ["abandon"],
            },
        )
    resumed, _transition = skill_ledger.resume(state)
    store_ledger(host, resumed)
    # The commitment is replayed, not recomputed. The stop returned a view; it
    # did not return the candidate choice, and re-deriving here against a moved
    # world would quietly change what the policy already decided.
    return _submit(
        host, control, resumed,
        resume_from=len(resumed["completed"]), commitment=commitment,
    )


def _submit(
    host: Any, control: Mapping[str, Any], state: Mapping[str, Any], *,
    resume_from: int, commitment: Mapping[str, Any],
) -> dict[str, Any]:
    """Write the one action this decision authorizes."""

    action = validate_runtime_action({
        "schema": SKILL_SEQUENCE_ACTION_SCHEMA,
        "observation_id": control["observation_id"],
        "action_id": f"action-{control['generation']}",
        "action": "execute_skill_sequence",
        "arguments": {
            "skill": state["skill"],
            "candidate_id": state["candidate_id"],
            "rationale": state["rationale"],
            "stop_before_commit": bool(state["stop_before_commit"]),
            "sequence": list(state["sequence"]),
            "resume_from": int(resume_from),
        },
        "bound_observation_sha256": control["observation_sha256"],
    })
    host._persist_action(
        Path(host.control_root) / "submitted-action.json", dict(control), action,
    )
    remaining = list(state["sequence"])[resume_from:]
    return {
        "success": True,
        "status": "submitted",
        "skill": state["skill"],
        "candidate_id": state["candidate_id"],
        "committed_sequence": list(state["sequence"]),
        "driving_stages": remaining,
        # A policy budgeting robot decisions has to be able to count. The whole
        # sequence is one decision, the way a combined v4 target state is; the
        # simulator advances once per dispatched stage, which is a different
        # budget and is stated rather than left to be inferred.
        "cost": {
            "robot_decisions": 1,
            "dispatched_stage_actions": sum(
                1 for stage in remaining
                if stage not in skill_ledger.READ_ONLY_STAGES
            ),
        },
        **({"precommit_motion_check": commitment["precommit_motion_check"]}
           if "precommit_motion_check" in commitment else {}),
        "target_ref": str(commitment.get("source_ref") or ""),
        "claim_boundary": CLAIM_BOUNDARY,
        "_runtime": {
            "directive": "action_submitted", "action_id": action["action_id"],
        },
    }


def _place_target(candidate: Mapping[str, Any]) -> Any:
    """The world pose one place candidate proposes, in either published shape."""

    # The persisted planner computation uses target_pose. move_pose belongs to
    # its public projection and to older candidate artifacts.
    if isinstance(candidate.get("target_pose"), Mapping):
        return candidate["target_pose"]
    move_pose = candidate.get("move_pose")
    if not isinstance(move_pose, Mapping):
        return None
    targets = move_pose.get("targets")
    if isinstance(targets, Mapping):
        arm_target = next(iter(targets.values()), None)
        return arm_target if isinstance(arm_target, Mapping) else None
    return move_pose


# --------------------------------------------------------------------------
# Driving one stage
# --------------------------------------------------------------------------

def drive_skill_stage(
    host: Any, action: Mapping[str, Any], *, stage: str, stage_index: int,
    observation: Mapping[str, Any], observation_root: Path,
    completed: Sequence[Mapping[str, Any]],
    capture_observation: Callable[[], tuple[dict[str, Any], Path]],
) -> dict[str, Any]:
    """Produce one stage's action or measurement from the committed candidate."""

    del observation, observation_root
    arguments = action["arguments"]
    state = load_ledger(host)
    commitment = load_commitment(host)
    if (
        state is None or commitment is None
        or state["skill"] != arguments["skill"]
        or commitment.get("candidate_id") != arguments["candidate_id"]
    ):
        return {
            "kind": "refusal", "status": "refused",
            "reason": "committed_ledger_is_missing_or_belongs_to_another_skill",
        }
    if (arguments["sequence"] != state["sequence"]
            or not 0 <= stage_index < len(state["sequence"])
            or state["sequence"][stage_index] != stage):
        return {"kind": "refusal", "status": "refused", "reason": "committed_stage_sequence_mismatch"}
    try:
        _confirm_candidate_still_bound(host, state)
    except ToolError as refusal:
        return {"kind": "refusal", "status": "refused", "reason": refusal.code}
    if stage in skill_ledger.READ_ONLY_STAGES:
        fresh, _root = capture_observation()
        measured = _measure(host, stage, fresh, commitment, completed)
        if state["skill"] == "grasp_execute" and "lift" in state["sequence"]:
            probe = skill_retention.measure(
                host, commitment=commitment, observation=fresh, observation_root=_root,
            )
            measured["verification"]["lift_probe"] = probe
        return measured
    if stage == "lift":
        if not completed or completed[-1].get("stage") != "close" or completed[-1].get("status") != "executed":
            return {"kind": "refusal", "status": "refused", "reason": "lift_requires_executed_close"}
        fresh, root = capture_observation()
        commitment = {**commitment, "lift_baseline": skill_retention.capture_baseline(fresh, root)}
        store_commitment(host, commitment)
    pose = commitment["poses"].get(stage)
    gripper = commitment["grippers"].get(stage)
    if pose is None and gripper is None:
        return {
            "kind": "refusal", "status": "refused",
            "reason": f"stage_{stage}_has_no_committed_target",
        }
    sparse = (
        host.surface.supports("dual_arm_sparse_targets_v1")
        or host.surface.supports("motion_intent_router_v1")
    )
    if sparse:
        # Candidate coordinates are public TCP coordinates. They must enter the
        # same sparse adapter as manual motion, where TCP-to-planner conversion,
        # controller validation and measured jaw evidence are owned. Every
        # surface declaring the sparse topology also declares the router.
        if pose is not None and gripper is not None:
            raise ValueError("a sparse skill stage must separate pose and gripper")
        arm = commitment["arm"]
        interaction = "contact_approach" if stage in {"approach", "descend"} else (
            "precision" if stage in {"settle", "lift"} else "free_space"
        )
        return {
            "kind": "mutation",
            "action": {
                "schema": MOTION_ACTION_V3_SCHEMA,
                "observation_id": action["observation_id"],
                "action_id": f"{action['action_id']}:{stage_index:02d}-{stage}",
                "action": "move_pose" if pose is not None else "set_gripper",
                "arguments": {
                    "targets": {arm: pose if pose is not None else (
                        "close" if gripper == GRIPPER_CLOSED else "open"
                    )},
                    **({"interactions": {arm: interaction}} if pose is not None else {}),
                },
                "bound_observation_sha256": action["bound_observation_sha256"],
            },
        }
    return {
        "kind": "mutation",
        "action": {
            "schema": TARGET_STATE_ACTION_SCHEMA,
            "observation_id": action["observation_id"],
            "action_id": f"{action['action_id']}:{stage_index:02d}-{stage}",
            "action": "target_state",
            "arguments": {
                "arm": commitment["arm"],
                **({"pose": pose} if pose is not None else {}),
                **({"gripper": gripper} if gripper is not None else {}),
            },
            "bound_observation_sha256": action["bound_observation_sha256"],
        },
    }


def _commit_geometry(
    host: Any, *, skill: str, candidate_id: str, observation_id: str,
    control: Mapping[str, Any], root: Path, observation: Mapping[str, Any],
    prepared: bool | None = None,
) -> dict[str, Any]:
    """Derive every stage target once, at the moment the policy decided.

    Every number here comes from the one observation the candidate was produced
    from. Deriving them later, against a world the sequence itself has moved,
    would quietly change what the policy chose; deriving them now is what makes
    "one committed sequence" a fact about the targets and not only about the
    call count.
    """

    from .runtime.core import _grasp_hover_offset_m

    standoff = _grasp_hover_offset_m(
        host.surface, grasp_plan_enabled=tool_names.declares(host.allowed_tools, "grasp_plan"),
    )
    if skill == "grasp_execute":
        resolved = _grasp_candidate(host, candidate_id, observation_id)
        contact = _pose(resolved["candidate"].get("contact_pose"), "contact_pose", candidate_id)
        if prepared is None:
            prepared = host.surface.supports(skill_retention.CAPABILITY)
        pregrasp = (
            _pose(resolved["candidate"].get("pregrasp_pose"), "pregrasp_pose", candidate_id)
            if prepared else None
        )
        poses = {"approach": contact}
        if prepared:
            poses.update(pregrasp=pregrasp, lift=_lifted(contact, skill_retention.LIFT_M))
            validate_prepared_targets(poses, host.workspace_bounds_m, candidate_id)
        # A dual-arm embodiment has no single atomic arm, so asking the
        # observation for one refused every closed grasp on RoboTwin before it
        # reached the simulator -- the skill had never run against a backend
        # with two. The candidate already knows: it was compiled for one arm's
        # controlled point and is meaningless on the other.
        return {
            "skill": skill, "candidate_id": candidate_id,
            "arm": resolved["arm"] or host._atomic_arm(observation),
            "observation_id": observation_id,
            "source_ref": resolved["target_ref"],
            "poses": poses,
            "grippers": {
                "close": GRIPPER_CLOSED,
                **({"open": GRIPPER_OPEN} if prepared else {}),
            },
            "localization": _target_localization(
                host, control=control, root=root, observation=observation,
                target_ref=resolved["target_ref"],
            ),
            "semantics": target_identity.resolve_target_semantics(
                host.artifact_root, resolved["target_ref"],
            ),
            **({"retention_reference": skill_retention.reference(
                host, control=control, root=root, observation=observation,
                target_ref=resolved["target_ref"],
            )} if prepared else {}),
        }
    resolved = _place_candidate(host, candidate_id, observation_id)
    if resolved["phase"] != "lower":
        raise ToolError(
            "place_execute requires a lower-phase candidate; use the preplace or exit "
            "candidate's move_pose first, then plan lower on the fresh observation",
            code="invalid_arguments", field="candidate_id", received=candidate_id,
            constraints={"required_phase": "lower", "received_phase": resolved["phase"]},
        )
    place = _pose(_place_target(resolved["candidate"]), "place target pose", candidate_id)
    return {
        "skill": skill, "candidate_id": candidate_id,
        "arm": resolved.get("arm") or host._atomic_arm(observation),
        "observation_id": observation_id,
        "source_ref": resolved["assessment_ref"],
        "poses": {
            "transport": _lifted(place, standoff),
            "descend": place,
            # The jaws opening is not the end of the motion. Withdrawing to the
            # same standoff advances the world once more and clears the gripper
            # from the object, so the verification measures a settled scene and
            # not the instant of release through the fingers.
            "settle": _lifted(place, standoff),
        },
        "grippers": {"release": GRIPPER_OPEN},
        "localization": None,
    }


def _confirm_candidate_still_bound(host: Any, state: Mapping[str, Any]) -> None:
    """Refuse by name if the committed candidate is no longer what it was.

    The candidate artifacts are immutable and content-addressed, so this is not
    a re-derivation: it is the check that the evidence the commitment rests on
    is still present and still names the observation it was bound to. A stage
    that cannot confirm that must not execute against what is left.
    """

    resolver = (
        _grasp_candidate if state["skill"] == "grasp_execute" else _place_candidate
    )
    resolver(host, str(state["candidate_id"]), str(state["observation_id"]))


def _target_localization(
    host: Any, *, control: Mapping[str, Any], root: Path,
    observation: Mapping[str, Any], target_ref: str,
) -> dict[str, Any] | None:
    try:
        return localization.localize(host._current_target_geometry(
            control=control, root=root, observation=observation,
            target_ref=target_ref,
        ))
    except (ToolError, KeyError, ValueError, OSError):
        return None


def _measure(
    host: Any, stage: str, observation: Mapping[str, Any],
    commitment: Mapping[str, Any], completed: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Measure a read-only stage against the freshly captured observation."""

    if stage == skill_ledger.POLICY_VISIBLE_STOP:
        return {
            "kind": "measurement", "status": "stopped",
            "reason": "stopped_for_close_view_before_release",
            "measurements": [{
                "quantity": "stop_observation_id",
                "value": str(observation.get("observation_id") or ""),
            }],
        }
    hold_signal = getattr(getattr(host, "surface", None), "supports", lambda _name: False)(jaw_hold.CAPABILITY)
    del host
    effectors = observation.get("proprioception", {}).get("end_effectors", {})
    record = effectors.get(commitment["arm"]) if isinstance(effectors, Mapping) else None
    evidence = localization.closure_evidence(
        gripper_opening_m=localization.measured_jaw_separation_m(record),
        localization=commitment.get("localization"),
    )
    if hold_signal and evidence["strength"] == "unavailable":
        evidence["jaw_hold"] = jaw_hold.verify_stage_evidence(completed, commitment["arm"], record)
    semantics = commitment.get("semantics")
    semantics = (
        semantics if isinstance(semantics, Mapping)
        else target_identity.unnamed_target()
    )
    return {
        "kind": "measurement", "status": "executed",
        "reason": f"verified_against_{observation.get('observation_id')}",
        "measurements": [
            {"quantity": "closure_strength", "value": evidence["strength"]},
            *([{"quantity": "jaw_hold", "value": evidence["jaw_hold"]["held"]}] if "jaw_hold" in evidence else []),
            {"quantity": "gripper_opening_m", "value": evidence["gripper_opening_m"]},
            {
                "quantity": "target_largest_extent_m",
                "value": evidence["target_largest_extent_m"],
            },
            # What the policy named, beside what the jaws measured. Neither is
            # a verdict on the other, and the pair is the whole evidence for a
            # question the geometry cannot reach.
            {"quantity": "committed_target_phrase", "value": semantics["phrase"]},
            {"quantity": "committed_target_score", "value": semantics["score"]},
            {
                "quantity": "committed_target_observation_id",
                "value": semantics["observation_id"],
            },
            {"quantity": "committed_target_selection", "value": semantics["selection"]},
        ],
        "verification": evidence,
    }


# --------------------------------------------------------------------------
# Reporting and releasing
# --------------------------------------------------------------------------

def build_skill_receipt(
    host: Any, *, action: Mapping[str, Any], status: str,
    stages: Sequence[Mapping[str, Any]], observation: Mapping[str, Any],
    stop_reason: str,
) -> dict[str, Any]:
    """Advance the ledger to its terminal phase and assemble one receipt.

    Called on every exit path the executor has, which is what makes the ledger
    safe to hold across an action at all: whatever happened, the sequence
    reports and the ledger stops being a decision the policy cannot take back.
    """

    state = load_ledger(host)
    verification: Mapping[str, Any] | None = None
    if state is not None:
        for item in stages:
            if skill_ledger.pending_stage(state) != item["stage"]:
                continue
            state, _transition = skill_ledger.record(
                state, stage=str(item["stage"]), status=str(item["status"]),
                reason=str(item["reason"]), action_id=item.get("action_id"),
                measurements=item.get("measurements", ()),
            )
        if status == "terminated":
            # A pre-stage terminal check records a skipped stage. The skill
            # still ends by lifecycle, independent of that stage's outcome.
            state["phase"] = "terminated"
        if status != "stopped" and state["phase"] not in {"completed", "aborted", "terminated"}:
            state, _transition = skill_ledger.abandon(state, stop_reason)
        if state["phase"] == "stopped":
            commitment = load_commitment(host)
            if commitment is not None and commitment.get("candidate_id") == state["candidate_id"]:
                # The coordinator publishes this exact final capture. Another
                # observation after it invalidates replay of the release stages,
                # including when a world mutation bypassed the policy tool gate.
                store_commitment(host, {
                    **commitment,
                    "stop_observation_id": observation.get("observation_id"),
                })
        store_ledger(host, state)
    for item in stages:
        if isinstance(item.get("verification"), Mapping):
            verification = item["verification"]
    receipt = skill_receipt.assemble(
        skill=str(action["arguments"]["skill"]),
        status=status,
        rationale=str(action["arguments"]["rationale"]),
        candidate_id=str(action["arguments"]["candidate_id"]),
        observation_id=str(action["observation_id"]),
        observation_sha256=str(action["bound_observation_sha256"]),
        stages=[
            skill_receipt.stage(
                str(item["stage"]), str(item["status"]), str(item["reason"]),
                action_id=item.get("action_id"),
                measurements=item.get("measurements", ()),
            )
            for item in stages
        ],
        verification=verification,
        held_object_offset=getattr(host, "last_held_object_offset", None),
        # Current images use the ordinary observation path. Prepared grasps
        # separately bind a historical post-close baseline below.
        images=(),
        claim_boundary=CLAIM_BOUNDARY,
    )
    receipt["stop_reason"] = stop_reason
    receipt["fresh_observation_id"] = str(observation.get("observation_id") or "")
    receipt["cost"] = {
        "robot_decisions": 1,
        "dispatched_stage_actions": sum(
            1 for item in stages if item.get("action_id") is not None
        ),
        "semantics": (
            "the Harness robot-decision budget counts this whole skill as one "
            "decision; the simulator advanced once per dispatched stage action"
        ),
    }
    receipt["ledger_released"] = (
        state is None or state["phase"] in {"completed", "aborted", "terminated"}
    )
    commitment = load_commitment(host)
    if isinstance(commitment, Mapping) and "precommit_motion_check" in commitment:
        receipt["precommit_motion_check"] = commitment["precommit_motion_check"]
    baseline = commitment.get("lift_baseline") if isinstance(commitment, Mapping) else None
    if isinstance(baseline, Mapping) and action["arguments"]["skill"] == "grasp_execute":
        receipt["observation_evidence"] = [{
            "arm": commitment["arm"],
            **{key: baseline[key] for key in (
                "role", "observation_root", "observation_id", "observation_sha256",
            )},
        }]
    # `assemble` measured the receipt it returned; the four fields above made it
    # a different object. Re-measuring is the whole point of that field -- a
    # recorded size that describes an earlier version of the body is exactly the
    # size claim about a different object the receipt module warns against.
    for _ in range(4):
        measured = skill_receipt.body_bytes(receipt)
        if receipt["delivery"]["body_bytes"] == measured:
            break
        receipt["delivery"]["body_bytes"] = max(
            measured, receipt["delivery"]["body_bytes"],
        )
    return receipt
