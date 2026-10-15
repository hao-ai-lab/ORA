"""The tool-runtime half of a closed skill: commit, drive, verify, release.

One policy call has to turn into several Supervisor executions without asking
the policy again, and the parts of that are split across three owners. This file
covers the part the tool runtime owns: resolving the chosen candidate against
the observation that produced it, committing one action for the whole sequence,
producing each stage's target from the committed candidate, and advancing and
releasing the ledger.

The releases are the tests worth reading. A sequence that stalls holding a
decision the policy cannot take back is worse than the three explicit calls
these skills replace, so the ledger is reconciled by any later call rather than
only by the writer that committed it. `build_skill_receipt` releases terminal
paths while a requested pre-release stop retains its commitment until an
explicit continue or abandon decision.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from ora.agent.context import ContextProjector
from ora.agent.tools import localization, skill_driver, skill_ledger
from ora.agent.tools.runtime import ManipulationToolRuntime
from ora.agent.tools.robot import ToolError
from ora.contracts import SKILL_SEQUENCE_ACTION_SCHEMA
from ora.environment import SyntheticSession


ROOT = Path(__file__).resolve().parents[3]
MANUAL = ROOT / "ora/policy/embodiments/aloha-agilex/manual-v1.md"
PROMPT = ROOT / "tests/data/prompts/startup-prompt-v1.md"
AGENTS = ROOT / "ora/policy/planners/native/project-v1.md"
BOUNDS = {"x": [-0.6, 0.6], "y": [-0.4, 0.4], "z": [0.7, 1.45]}

CONTACT = {"position_m": [0.2, 0.1, 0.9], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}
PLACE = {"position_m": [0.3, -0.1, 0.85], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}


def _runtime(tmp_path: Path):
    source = tmp_path / "source"
    session = SyntheticSession(
        tmp_path / "native", "skill_driver_synthetic", 0,
        main_camera="agentview", camera_names=("agentview", "wrist"),
    )
    session.start()
    observation, _images = session.observe(source)
    control = tmp_path / "control"
    projector = ContextProjector(
        tmp_path / "workspace", control, PROMPT, MANUAL, AGENTS,
        expose_manual=False, main_camera="agentview",
    )
    projector.publish(
        observation=observation,
        observation_root=source / observation["observation_id"],
        task_instruction="test",
        remaining_budget={"robot_decisions": 8, "model_calls": 8},
        policy_state={
            "step": 0, "path": "unused", "sha256": "unused",
            "state_sha256": "unused",
        },
    )
    runtime = ManipulationToolRuntime(
        control_root=control, manual_path=MANUAL,
        event_path=tmp_path / "events.jsonl", workspace_bounds_m=BOUNDS,
        state_root=None, tool_surface_profile="native_atomic_causal_v1",
        controlled_end_effector="right",
    )
    # A single-arm target-state surface that does not declare the skills. The
    # fixture grants exactly what a declaring profile would: the capability and
    # the three names in `allowed_tools`.
    runtime.surface = replace(
        runtime.surface,
        capabilities=(*runtime.surface.capabilities, skill_driver.CAPABILITY),
    )
    runtime.allowed_tools = (
        *runtime.allowed_tools, "grasp_execute", "place_execute", "place_resume",
    )
    return runtime


def _observation_id(runtime) -> str:
    control = json.loads(
        (runtime.control_root / "current-observation.json").read_text(encoding="utf-8")
    )
    return str(control["observation_id"])


def _publish_grasp_candidate(runtime, *, observation_id: str, candidate_id="grasp-a1"):
    root = runtime.artifact_root / "grasp-queries" / "grasp-query-fixture"
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps({
        "query_ref": "grasp-query-fixture",
        "identity": {"observation_id": observation_id, "target_ref": "target-1"},
        "compiled_candidates": [{
            "candidate_id": candidate_id, "rank": 0,
            "pregrasp_pose": {
                "position_m": [0.2, 0.1, 0.96],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "contact_pose": dict(CONTACT),
            "gripper_opening_m": 0.04,
        }],
    }), encoding="utf-8")


def _publish_grounding(runtime, *, observation_id: str, target_ref="target-1",
                       text="tomato sauce can", score=0.94):
    root = runtime.artifact_root / "grounding-queries" / "grounding-query-fixture"
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps({
        "grounding_ref": "grounding-query-fixture",
        "identity": {"text": text, "camera": "agentview",
                     "observation_id": observation_id},
        "candidates": [{"rank": 0, "target_ref": target_ref,
                        "backend_score": score}],
    }), encoding="utf-8")


def _publish_place_candidate(runtime, *, observation_id: str, candidate_id="P0",
                             assessment="place-check-fixture"):
    root = runtime.artifact_root / "placement"
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{assessment}.json").write_text(json.dumps({
        "identity": {"observation_id": observation_id, "arguments": {"phase": "lower"}},
        "computation": {
            "phase": "lower",
            "candidates": [{"candidate_id": candidate_id, "move_pose": dict(PLACE)}],
        },
    }), encoding="utf-8")


def _submitted(runtime) -> dict:
    return json.loads(
        (runtime.control_root / "submitted-action.json").read_text(encoding="utf-8")
    )


def test_a_grasp_commits_one_action_for_its_whole_sequence(tmp_path):
    runtime = _runtime(tmp_path)
    _publish_grasp_candidate(runtime, observation_id=_observation_id(runtime))

    result = runtime.invoke("grasp_execute", {
        "candidate_id": "grasp-a1",
        "rationale": "the only candidate clear of the neighbouring mug",
    })

    assert result["status"] == "submitted"
    assert result["committed_sequence"] == list(skill_ledger.GRASP_STAGES)
    assert result["_runtime"]["directive"] == "action_submitted"
    action = _submitted(runtime)
    assert action["schema"] == SKILL_SEQUENCE_ACTION_SCHEMA
    assert action["arguments"]["sequence"] == ["approach", "close", "verify"]
    assert action["arguments"]["resume_from"] == 0


def test_the_submitted_result_states_what_one_call_costs(tmp_path):
    """A policy budgeting 80 decisions has to be able to count."""

    runtime = _runtime(tmp_path)
    _publish_grasp_candidate(runtime, observation_id=_observation_id(runtime))

    result = runtime.invoke("grasp_execute", {
        "candidate_id": "grasp-a1", "rationale": "clear approach corridor",
    })

    assert result["cost"]["robot_decisions"] == 1
    assert result["cost"]["dispatched_stage_actions"] == 2


def test_a_place_states_its_four_dispatched_stages(tmp_path):
    runtime = _runtime(tmp_path)
    _publish_place_candidate(runtime, observation_id=_observation_id(runtime))

    result = runtime.invoke("place_execute", {
        "candidate_id": "P0", "rationale": "widest hull margin of the three patches",
    })

    assert result["committed_sequence"] == [
        "transport", "descend", "release", "settle", "verify",
    ]
    assert result["cost"]["robot_decisions"] == 1
    assert result["cost"]["dispatched_stage_actions"] == 4


def test_a_candidate_from_an_earlier_observation_is_refused_by_name(tmp_path):
    """The candidate binds to the observation that produced it."""

    runtime = _runtime(tmp_path)
    _publish_grasp_candidate(runtime, observation_id="observation-that-has-passed")

    with pytest.raises(ToolError) as refused:
        runtime.invoke("grasp_execute", {
            "candidate_id": "grasp-a1", "rationale": "stale geometry",
        })

    assert refused.value.code == "stale_candidate_observation"
    assert not (runtime.control_root / "submitted-action.json").is_file()
    assert skill_driver.load_ledger(runtime) is None


def test_two_current_place_assessments_with_the_same_id_are_ambiguous(tmp_path):
    """Resolving by recency would execute a candidate nobody looked at."""

    runtime = _runtime(tmp_path)
    observation_id = _observation_id(runtime)
    _publish_place_candidate(runtime, observation_id=observation_id, assessment="place-check-a")
    _publish_place_candidate(runtime, observation_id=observation_id, assessment="place-check-b")

    with pytest.raises(ToolError) as refused:
        runtime.invoke("place_execute", {
            "candidate_id": "P0", "rationale": "widest hull margin",
        })

    assert refused.value.code == "ambiguous_candidate_id"


def test_asking_a_grasp_to_stop_is_refused_at_the_tool_boundary(tmp_path):
    runtime = _runtime(tmp_path)
    _publish_grasp_candidate(runtime, observation_id=_observation_id(runtime))

    with pytest.raises(ToolError, match="candidate_id and rationale"):
        runtime.invoke("grasp_execute", {
            "candidate_id": "grasp-a1", "rationale": "r",
            "stop_before_commit": True,
        })


def test_a_second_skill_is_refused_while_one_is_in_flight(tmp_path):
    runtime = _runtime(tmp_path)
    _publish_grasp_candidate(runtime, observation_id=_observation_id(runtime))
    runtime.invoke("grasp_execute", {
        "candidate_id": "grasp-a1", "rationale": "clear approach corridor",
    })

    with pytest.raises(ToolError) as refused:
        runtime.invoke("grasp_execute", {
            "candidate_id": "grasp-a1", "rationale": "clear approach corridor",
        })

    assert refused.value.code == "skill_sequence_active"


def test_a_ledger_whose_sequence_never_reported_is_released_not_wedged(tmp_path):
    """A crashed sequence must not hold a decision no later call can clear."""

    runtime = _runtime(tmp_path)
    observation_id = _observation_id(runtime)
    _publish_grasp_candidate(runtime, observation_id=observation_id)
    runtime.invoke("grasp_execute", {
        "candidate_id": "grasp-a1", "rationale": "clear approach corridor",
    })
    # The world moved on without the sequence ever reporting: the action was
    # consumed and a new observation was published.
    (runtime.control_root / "submitted-action.json").unlink()
    control = json.loads(
        (runtime.control_root / "current-observation.json").read_text(encoding="utf-8")
    )
    control["observation_id"] = "a-later-observation"
    (runtime.control_root / "current-observation.json").write_text(
        json.dumps(control), encoding="utf-8",
    )

    released = skill_driver.reconcile(runtime)

    assert released["phase"] == "aborted"
    assert skill_driver.load_ledger(runtime)["phase"] == "aborted"


def test_a_stopped_sequence_survives_the_observation_it_is_waiting_for(tmp_path):
    """The stop ends its action; the close view is the next observation.

    Reconciliation treats a new observation with no pending action as proof that
    a running sequence is never going to report. A stopped one is in exactly
    that state by design, and releasing it here would discard the decision at
    the one moment the policy is meant to pick it up.
    """

    runtime = _runtime(tmp_path)
    observation_id = _observation_id(runtime)
    _publish_place_candidate(runtime, observation_id=observation_id)
    runtime.invoke("place_execute", {
        "candidate_id": "P0", "rationale": "widest hull margin",
        "stop_before_commit": True,
    })
    action = _submitted(runtime)
    runtime.build_skill_receipt(
        action=action, status="stopped", observation={"observation_id": "the-close-view"},
        stop_reason="stopped_for_close_view_before_release",
        stages=[
            {"stage": "transport", "status": "executed", "reason": "executed",
             "action_id": "a:00-transport", "measurements": []},
            {"stage": "descend", "status": "executed", "reason": "executed",
             "action_id": "a:01-descend", "measurements": []},
            {"stage": "stop", "status": "stopped", "reason": "stopped",
             "action_id": None, "measurements": []},
        ],
    )
    (runtime.control_root / "submitted-action.json").unlink()
    control = json.loads(
        (runtime.control_root / "current-observation.json").read_text(encoding="utf-8")
    )
    control["observation_id"] = "the-close-view"
    control["open"] = True
    (runtime.control_root / "current-observation.json").write_text(
        json.dumps(control), encoding="utf-8",
    )

    assert skill_driver.reconcile(runtime)["phase"] == "stopped"
    assert runtime.invoke("place_resume", {"decision": "continue"})["status"] == "submitted"


def test_a_pending_action_is_not_reconciled_away(tmp_path):
    """Reconciliation must not release a sequence that is still executing."""

    runtime = _runtime(tmp_path)
    _publish_grasp_candidate(runtime, observation_id=_observation_id(runtime))
    runtime.invoke("grasp_execute", {
        "candidate_id": "grasp-a1", "rationale": "clear approach corridor",
    })
    control = json.loads(
        (runtime.control_root / "current-observation.json").read_text(encoding="utf-8")
    )
    control["observation_id"] = "a-later-observation"
    (runtime.control_root / "current-observation.json").write_text(
        json.dumps(control), encoding="utf-8",
    )

    assert skill_driver.reconcile(runtime)["phase"] == "running"


def _driven(runtime, action, stage, index, observation):
    return runtime.drive_skill_stage(
        action, stage=stage, stage_index=index, observation=observation,
        observation_root=runtime.control_root, completed=[],
        capture_observation=lambda: (observation, runtime.control_root),
    )


def test_each_grasp_stage_is_produced_from_the_committed_candidate(tmp_path):
    runtime = _runtime(tmp_path)
    _publish_grasp_candidate(runtime, observation_id=_observation_id(runtime))
    runtime.invoke("grasp_execute", {
        "candidate_id": "grasp-a1", "rationale": "clear approach corridor",
    })
    action = _submitted(runtime)
    observation = {"observation_id": "fresh-1", "proprioception": {
        "end_effectors": {"right": {"gripper_opening_m": 0.0009}},
    }}

    approach = _driven(runtime, action, "approach", 0, observation)
    close = _driven(runtime, action, "close", 1, observation)
    verify = _driven(runtime, action, "verify", 2, observation)

    assert approach["kind"] == "mutation"
    assert approach["action"]["arguments"]["pose"]["position_m"] == CONTACT["position_m"]
    assert approach["action"]["action_id"] == f"{action['action_id']}:00-approach"
    assert close["action"]["arguments"]["gripper"] == skill_driver.GRIPPER_CLOSED
    assert "pose" not in close["action"]["arguments"]
    assert verify["kind"] == "measurement"
    assert verify["verification"]["gripper_opening_m"] == pytest.approx(0.0009)


def test_verify_reads_the_libero_finger_displacements_when_there_is_no_aperture(tmp_path):
    """LIBERO publishes two prismatic finger displacements, not an aperture.

    The separation is their summed magnitude, the same conversion the public
    facts layer already applies. Reading only the RoboTwin key left the
    absorbed verify reporting `unavailable` on every LIBERO episode while the
    measurement sat one field away.
    """

    runtime = _runtime(tmp_path)
    _publish_grasp_candidate(runtime, observation_id=_observation_id(runtime))
    runtime.invoke("grasp_execute", {
        "candidate_id": "grasp-a1", "rationale": "clear approach corridor",
    })
    action = _submitted(runtime)
    # Jaws met: 1.18 mm + 1.17 mm is below the 3 mm near-closed bound, so this
    # is the negative signal the run was losing, not merely a missing number.
    closed = {"observation_id": "fresh-1", "proprioception": {"end_effectors": {
        "right": {"gripper_finger_qpos": [0.0011815762989788993,
                                          -0.0011679928755845951],
                  "gripper": 0.5099},
    }}}
    verify = _driven(runtime, action, "verify", 2, closed)
    assert verify["verification"]["gripper_opening_m"] == pytest.approx(0.0023495691745634946)
    # Only the target extents are now missing; the separation is no longer.
    assert verify["verification"]["missing"] == ["target_extents_m"]


def test_a_jaw_separation_under_the_near_closed_bound_is_a_strong_negative():
    """The negative branch the LIBERO runs were losing, given both inputs.

    2.35 mm of separation is below the 3 mm near-closed bound, so the jaws met:
    evidence that nothing of the target's measured size is between them.
    """

    evidence = localization.closure_evidence(
        gripper_opening_m=0.0023495691745634946,
        localization={
            "schema": localization.SCHEMA,
            "principal_extents_m": [0.1586882214457184,
                                    0.06582657322217246,
                                    0.044610149665408064],
        },
    )
    assert evidence["strength"] == "strong_negative"
    assert "missing" not in evidence


def test_the_measured_jaw_separation_reads_either_backend_and_never_the_command():
    from_robotwin = {"gripper_opening_m": 0.0412}
    from_libero = {"gripper_finger_qpos": [0.04000015011914127,
                                           -0.03999216510420057],
                   "gripper": 0.9999}
    assert localization.measured_jaw_separation_m(from_robotwin) == pytest.approx(0.0412)
    assert localization.measured_jaw_separation_m(from_libero) == pytest.approx(0.07999231522334184)
    # A normalized command is not a length and must not stand in for one.
    assert localization.measured_jaw_separation_m({"gripper": 0.5099}) is None
    assert localization.measured_jaw_separation_m({"gripper_finger_qpos": [0.02]}) is None
    assert localization.measured_jaw_separation_m(None) is None


def test_verify_prefers_a_published_aperture_over_the_finger_displacements(tmp_path):
    runtime = _runtime(tmp_path)
    _publish_grasp_candidate(runtime, observation_id=_observation_id(runtime))
    runtime.invoke("grasp_execute", {
        "candidate_id": "grasp-a1", "rationale": "clear approach corridor",
    })
    action = _submitted(runtime)
    both = {"observation_id": "fresh-1", "proprioception": {"end_effectors": {
        "right": {"gripper_opening_m": 0.0009,
                  "gripper_finger_qpos": [0.02, -0.02]},
    }}}
    verify = _driven(runtime, action, "verify", 2, both)
    assert verify["verification"]["gripper_opening_m"] == pytest.approx(0.0009)


def test_the_place_stages_lift_transport_and_settle_off_the_place_pose(tmp_path):
    runtime = _runtime(tmp_path)
    _publish_place_candidate(runtime, observation_id=_observation_id(runtime))
    runtime.invoke("place_execute", {
        "candidate_id": "P0", "rationale": "widest hull margin",
    })
    action = _submitted(runtime)
    observation = {"observation_id": "fresh-1", "proprioception": {"end_effectors": {}}}

    transport = _driven(runtime, action, "transport", 0, observation)
    descend = _driven(runtime, action, "descend", 1, observation)
    release = _driven(runtime, action, "release", 2, observation)
    settle = _driven(runtime, action, "settle", 3, observation)

    place_z = PLACE["position_m"][2]
    assert transport["action"]["arguments"]["pose"]["position_m"][2] > place_z
    assert descend["action"]["arguments"]["pose"]["position_m"][2] == pytest.approx(place_z)
    assert release["action"]["arguments"]["gripper"] == skill_driver.GRIPPER_OPEN
    assert settle["action"]["arguments"]["pose"]["position_m"][2] > place_z


def test_a_stage_whose_candidate_went_stale_refuses_rather_than_executing(tmp_path):
    """The refusal reaches the receipt by name instead of moving the robot."""

    runtime = _runtime(tmp_path)
    _publish_grasp_candidate(runtime, observation_id=_observation_id(runtime))
    runtime.invoke("grasp_execute", {
        "candidate_id": "grasp-a1", "rationale": "clear approach corridor",
    })
    action = _submitted(runtime)
    _publish_grasp_candidate(runtime, observation_id="something-else")

    driven = _driven(runtime, action, "approach", 0, {"observation_id": "fresh-1"})

    assert driven["kind"] == "refusal"
    assert driven["reason"] == "stale_candidate_observation"


def test_the_receipt_releases_the_ledger_and_states_the_cost(tmp_path):
    runtime = _runtime(tmp_path)
    _publish_grasp_candidate(runtime, observation_id=_observation_id(runtime))
    runtime.invoke("grasp_execute", {
        "candidate_id": "grasp-a1", "rationale": "clear approach corridor",
    })
    action = _submitted(runtime)

    receipt = runtime.build_skill_receipt(
        action=action, status="completed", observation={"observation_id": "fresh-1"},
        stop_reason="sequence_completed",
        stages=[
            {"stage": "approach", "status": "executed", "reason": "executed",
             "action_id": f"{action['action_id']}:00-approach", "measurements": []},
            {"stage": "close", "status": "executed", "reason": "executed",
             "action_id": f"{action['action_id']}:01-close", "measurements": []},
            {"stage": "verify", "status": "executed", "reason": "measured",
             "action_id": None, "measurements": []},
        ],
    )

    assert receipt["schema"] == "ora.skill-receipt.v1"
    assert receipt["status"] == "completed"
    assert receipt["rationale"] == "clear approach corridor"
    assert receipt["cost"] == {
        "robot_decisions": 1, "dispatched_stage_actions": 2,
        "semantics": receipt["cost"]["semantics"],
    }
    assert receipt["ledger_released"] is True
    assert skill_driver.load_ledger(runtime)["phase"] == "completed"
    # The recorded size describes the receipt that carries it, not the one
    # `assemble` returned before the driver added its four fields.
    from ora.agent.tools import skill_receipt

    assert receipt["delivery"]["body_bytes"] == skill_receipt.body_bytes(receipt)
    assert receipt["delivery"]["body_bytes"] < skill_receipt.DELIVERY_CEILING_BYTES


def test_an_aborted_sequence_releases_the_ledger_too(tmp_path):
    runtime = _runtime(tmp_path)
    _publish_grasp_candidate(runtime, observation_id=_observation_id(runtime))
    runtime.invoke("grasp_execute", {
        "candidate_id": "grasp-a1", "rationale": "clear approach corridor",
    })
    action = _submitted(runtime)

    receipt = runtime.build_skill_receipt(
        action=action, status="aborted", observation={"observation_id": "fresh-1"},
        stop_reason="approach_not_reached",
        stages=[{
            "stage": "approach", "status": "not_reached",
            "reason": "endpoint_residual_exceeded",
            "action_id": f"{action['action_id']}:00-approach", "measurements": [],
        }],
    )

    assert receipt["status"] == "aborted"
    assert receipt["stop_reason"] == "approach_not_reached"
    assert receipt["ledger_released"] is True
    assert skill_driver.load_ledger(runtime)["phase"] == "aborted"


def test_a_skill_receipt_carries_no_images_of_its_own(tmp_path):
    """Protected current-step images end an episode when they overflow."""

    runtime = _runtime(tmp_path)
    _publish_place_candidate(runtime, observation_id=_observation_id(runtime))
    runtime.invoke("place_execute", {
        "candidate_id": "P0", "rationale": "widest hull margin",
        "stop_before_commit": True,
    })
    action = _submitted(runtime)

    receipt = runtime.build_skill_receipt(
        action=action, status="stopped", observation={"observation_id": "fresh-1"},
        stop_reason="stopped_for_close_view_before_release",
        stages=[
            {"stage": "transport", "status": "executed", "reason": "executed",
             "action_id": f"{action['action_id']}:00-transport", "measurements": []},
            {"stage": "descend", "status": "executed", "reason": "executed",
             "action_id": f"{action['action_id']}:01-descend", "measurements": []},
            {"stage": "stop", "status": "stopped", "reason": "stopped",
             "action_id": None, "measurements": []},
        ],
    )

    assert receipt["image_mapping"] == []
    assert "image_base64" not in receipt and "images_base64" not in receipt
    assert skill_driver.load_ledger(runtime)["phase"] == "stopped"
    assert receipt["ledger_released"] is False


def test_resuming_takes_a_decision_and_not_the_candidate_choice(tmp_path):
    runtime = _runtime(tmp_path)
    observation_id = _observation_id(runtime)
    _publish_place_candidate(runtime, observation_id=observation_id)
    runtime.invoke("place_execute", {
        "candidate_id": "P0", "rationale": "widest hull margin",
        "stop_before_commit": True,
    })
    action = _submitted(runtime)
    runtime.build_skill_receipt(
        action=action, status="stopped", observation={"observation_id": observation_id},
        stop_reason="stopped_for_close_view_before_release",
        stages=[
            {"stage": "transport", "status": "executed", "reason": "executed",
             "action_id": "a:00-transport", "measurements": []},
            {"stage": "descend", "status": "executed", "reason": "executed",
             "action_id": "a:01-descend", "measurements": []},
            {"stage": "stop", "status": "stopped", "reason": "stopped",
             "action_id": None, "measurements": []},
        ],
    )
    (runtime.control_root / "submitted-action.json").unlink()
    control = json.loads(
        (runtime.control_root / "current-observation.json").read_text(encoding="utf-8")
    )
    control["open"] = True
    (runtime.control_root / "current-observation.json").write_text(
        json.dumps(control), encoding="utf-8",
    )

    with pytest.raises(ToolError, match="continue or abandon"):
        runtime.invoke("place_resume", {"candidate_id": "P1"})

    resumed = runtime.invoke("place_resume", {"decision": "continue"})

    assert resumed["status"] == "submitted"
    assert resumed["driving_stages"] == ["release", "settle", "verify"]
    # release and settle move the world; verify is read-only
    assert resumed["cost"]["dispatched_stage_actions"] == 2
    assert _submitted(runtime)["arguments"]["resume_from"] == 3
    assert _submitted(runtime)["arguments"]["candidate_id"] == "P0"


def test_abandoning_a_stopped_sequence_releases_it_without_moving(tmp_path):
    runtime = _runtime(tmp_path)
    observation_id = _observation_id(runtime)
    _publish_place_candidate(runtime, observation_id=observation_id)
    runtime.invoke("place_execute", {
        "candidate_id": "P0", "rationale": "widest hull margin",
        "stop_before_commit": True,
    })
    action = _submitted(runtime)
    runtime.build_skill_receipt(
        action=action, status="stopped", observation={"observation_id": observation_id},
        stop_reason="stopped_for_close_view_before_release",
        stages=[
            {"stage": "transport", "status": "executed", "reason": "executed",
             "action_id": "a:00-transport", "measurements": []},
            {"stage": "descend", "status": "executed", "reason": "executed",
             "action_id": "a:01-descend", "measurements": []},
            {"stage": "stop", "status": "stopped", "reason": "stopped",
             "action_id": None, "measurements": []},
        ],
    )
    (runtime.control_root / "submitted-action.json").unlink()

    result = runtime.invoke("place_resume", {"decision": "abandon"})

    assert result["status"] == "aborted"
    assert result["released"] is True
    assert skill_driver.load_ledger(runtime)["phase"] == "aborted"
    assert not (runtime.control_root / "submitted-action.json").is_file()


def _paused_placement(runtime):
    observation_id = _observation_id(runtime)
    _publish_place_candidate(runtime, observation_id=observation_id)
    runtime.invoke("place_execute", {
        "candidate_id": "P0", "rationale": "inspect before release",
        "stop_before_commit": True,
    })
    action = _submitted(runtime)
    runtime.build_skill_receipt(
        action=action, status="stopped", observation={"observation_id": observation_id},
        stop_reason="stopped_for_close_view_before_release",
        stages=[
            {"stage": stage, "status": "stopped" if stage == "stop" else "executed",
             "reason": "executed", "action_id": None if stage == "stop" else f"a:{stage}",
             "measurements": []}
            for stage in ("transport", "descend", "stop")
        ],
    )
    (runtime.control_root / "submitted-action.json").unlink()
    path = runtime.control_root / "current-observation.json"
    control = json.loads(path.read_text())
    control["open"] = True
    path.write_text(json.dumps(control))


@pytest.mark.parametrize("tool,arguments", [
    ("grasp_execute", {"candidate_id": "grasp-a1", "rationale": "recover"}),
    ("place_execute", {"candidate_id": "P0", "rationale": "recover"}),
    ("open_gripper", {}),
    ("close_gripper", {}),
])
def test_paused_placement_requires_explicit_handoff_before_another_mutation(
    tmp_path, tool, arguments,
):
    runtime = _runtime(tmp_path)
    _paused_placement(runtime)
    runtime.allowed_tools = (*runtime.allowed_tools, tool, "open_gripper")
    committed = skill_driver.load_commitment(runtime)

    with pytest.raises(ToolError) as rejected:
        runtime.invoke(tool, arguments)

    assert rejected.value.code == "skill_sequence_paused"
    assert rejected.value.constraints["recovery_tool"] == "place_resume"
    assert rejected.value.constraints["decisions"] == ["continue", "abandon"]
    assert skill_driver.load_ledger(runtime)["phase"] == "stopped"
    assert skill_driver.load_commitment(runtime) == committed
    assert not (runtime.control_root / "submitted-action.json").exists()
    assert runtime.invoke("observe", {})["observation_id"] == _observation_id(runtime)
    runtime.invoke("place_resume", {"decision": "abandon"})
    assert runtime.invoke("open_gripper", {})["status"] == "submitted"


@pytest.mark.parametrize("binding", ["changed", "missing"])
def test_resume_refuses_stale_or_unbound_stop_but_allows_abandon(tmp_path, binding):
    runtime = _runtime(tmp_path)
    _paused_placement(runtime)
    if binding == "changed":
        path = runtime.control_root / "current-observation.json"
        control = json.loads(path.read_text())
        control["observation_id"] = "after-external-mutation"
        path.write_text(json.dumps(control))
    else:
        commitment = skill_driver.load_commitment(runtime)
        commitment.pop("stop_observation_id")
        skill_driver.store_commitment(runtime, commitment)

    with pytest.raises(ToolError) as rejected:
        runtime.invoke("place_resume", {"decision": "continue"})

    assert rejected.value.code == "skill_resume_observation_mismatch"
    assert rejected.value.constraints["decisions"] == ["abandon"]
    assert skill_driver.load_ledger(runtime)["phase"] == "stopped"
    assert not (runtime.control_root / "submitted-action.json").exists()
    assert runtime.invoke("place_resume", {"decision": "abandon"})["released"] is True


def test_a_surface_without_the_capability_refuses_the_skill(tmp_path):
    runtime = _runtime(tmp_path)
    runtime.surface = replace(
        runtime.surface,
        capabilities=tuple(
            name for name in runtime.surface.capabilities
            if name != skill_driver.CAPABILITY
        ),
    )

    with pytest.raises(ToolError) as refused:
        runtime.invoke("grasp_execute", {
            "candidate_id": "grasp-a1", "rationale": "clear approach corridor",
        })

    assert refused.value.code == "tool_surface_capability_mismatch"


def test_verify_reports_the_phrase_the_policy_committed_to(tmp_path):
    """Geometry alone cannot say whether the right object was grasped.

    The first B2 run grasped, re-grounded, and discovered mid-episode that a
    can it had been treating as the target was a different object. The verify
    stage measured jaw separation and visible extents and never once said what
    the policy had named -- so the semantic question had no evidence path
    inside the skill at all, and the policy paid a fresh grounding round after
    every grasp to ask it.

    What travels is the commitment, not a verdict: the phrase, its score, the
    view and the observation it was grounded in. Whether the thing in the jaws
    is that thing is a judgment, and the fresh full frames the next turn
    already carries are what it is judged from.
    """

    runtime = _runtime(tmp_path)
    observation_id = _observation_id(runtime)
    _publish_grasp_candidate(runtime, observation_id=observation_id)
    _publish_grounding(runtime, observation_id=observation_id,
                       text="tomato sauce can", score=0.94)
    runtime.invoke("grasp_execute", {
        "candidate_id": "grasp-a1", "rationale": "clear approach corridor",
    })
    action = _submitted(runtime)
    observation = {"observation_id": "fresh-1", "proprioception": {
        "end_effectors": {"right": {"gripper_finger_qpos": [0.02, -0.02]}},
    }}

    verify = _driven(runtime, action, "verify", 2, observation)
    measured = {item["quantity"]: item["value"] for item in verify["measurements"]}
    assert measured["committed_target_phrase"] == "tomato sauce can"
    assert measured["committed_target_score"] == pytest.approx(0.94)
    assert measured["committed_target_observation_id"] == observation_id
    # No verdict: naming the target is evidence, deciding is not the tool's.
    assert not any(
        key.startswith("semantic_") or key.endswith("_correct")
        for key in measured
    )


def test_a_target_chosen_by_point_says_so_instead_of_inventing_a_phrase(tmp_path):
    """A point-selected target has no phrase, and that is itself informative."""

    runtime = _runtime(tmp_path)
    observation_id = _observation_id(runtime)
    _publish_grasp_candidate(runtime, observation_id=observation_id)
    runtime.invoke("grasp_execute", {
        "candidate_id": "grasp-a1", "rationale": "clear approach corridor",
    })
    action = _submitted(runtime)
    observation = {"observation_id": "fresh-1", "proprioception": {
        "end_effectors": {"right": {"gripper_finger_qpos": [0.02, -0.02]}},
    }}
    verify = _driven(runtime, action, "verify", 2, observation)
    measured = {item["quantity"]: item["value"] for item in verify["measurements"]}
    assert measured["committed_target_phrase"] is None
    assert measured["committed_target_selection"] == "not_grounded_from_a_phrase"


def test_a_dual_arm_candidate_commits_to_the_arm_it_was_compiled_for(tmp_path):
    """A two-armed embodiment has no single atomic arm, and the plan knows.

    Every closed grasp on RoboTwin was refused before it reached the
    simulator: the commitment asked the observation for "the" end effector and
    ALOHA publishes two, so `embodiment_binding` came back with
    `available_effectors: [left, right]` six times across three cells and the
    policy routed around the skill with raw motion. The skill had never run
    against a backend with two arms.

    The candidate carries it. A Contact-GraspNet proposal is compiled onto one
    arm's controlled point and is meaningless on the other, so that arm is a
    property of the plan rather than of the moment it executes in.
    """

    runtime = _runtime(tmp_path)
    observation_id = _observation_id(runtime)
    root = runtime.artifact_root / "grasp-queries" / "dual-arm-query"
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps({
        "query_ref": "dual-arm-query",
        "identity": {
            "observation_id": observation_id, "target_ref": "target-1",
            "compiler": {"arm": "left",
                         "id": "contact_graspnet_to_robotwin_aloha_left_grip_center_v3"},
        },
        "compiled_candidates": [{
            "candidate_id": "grasp-dual", "rank": 0,
            "pregrasp_pose": {"position_m": [0.2, 0.1, 0.96],
                              "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
            "contact_pose": dict(CONTACT),
            "gripper_opening_m": 0.04,
        }],
    }), encoding="utf-8")

    runtime.invoke("grasp_execute", {
        "candidate_id": "grasp-dual", "rationale": "left arm reaches the block",
    })
    action = _submitted(runtime)
    # Two public effectors: the observation cannot name one, and does not have to.
    observation = {"observation_id": "fresh-1", "proprioception": {"end_effectors": {
        "left": {"gripper_opening_m": 0.04},
        "right": {"gripper_opening_m": 0.04},
    }}}
    approach = _driven(runtime, action, "approach", 0, observation)
    assert approach["action"]["arguments"]["arm"] == "left"
    close = _driven(runtime, action, "close", 1, observation)
    assert close["action"]["arguments"]["arm"] == "left"


@pytest.mark.parametrize('profile', ['robotwin_toolchain_b2_closed_noservo_v1'])
@pytest.mark.parametrize('arm', ['left', 'right'])
def test_closed_grasp_enters_the_selected_public_tcp_motion_codec(tmp_path, profile, arm):
    from ora.agent.composition import resolve_tool_surface
    from ora.contracts import validate_runtime_action, MOTION_ACTION_V3_SCHEMA

    runtime = _runtime(tmp_path)
    runtime.surface = resolve_tool_surface(profile)
    runtime.allowed_tools = tuple(item['name'] for item in runtime.surface.definitions)
    _publish_grasp_candidate(runtime, observation_id=_observation_id(runtime))
    path = runtime.artifact_root / 'grasp-queries/grasp-query-fixture/manifest.json'
    candidate = json.loads(path.read_text())
    candidate['identity']['compiler'] = {'arm': arm, 'controlled_point_semantics': 'parallel_jaw_grip_center_tcp'}
    path.write_text(json.dumps(candidate))
    runtime.motion_probe_callback = lambda request: {
        "schema": "ora.motion-probe-result.v1",
        "results": [{"query_id": q["query_id"], "arm": q["arm"],
                     "reachable": True, "variants": []} for q in request["queries"]],
    }
    runtime.invoke('grasp_execute', {'candidate_id': 'grasp-a1', 'rationale': 'selected public TCP'})
    action = _submitted(runtime)
    observation = {'observation_id': 'fresh'}
    approach = validate_runtime_action(_driven(runtime, action, 'approach', 2, observation)['action'])
    close = validate_runtime_action(_driven(runtime, action, 'close', 3, observation)['action'])
    assert approach['schema'] == close['schema'] == MOTION_ACTION_V3_SCHEMA
    assert approach['arguments'] == {'targets': {arm: CONTACT}, 'interactions': {arm: 'contact_approach'}}
    assert close['arguments'] == {'targets': {arm: 'close'}}
    # The RoboTwin session bound to the composed control profile executes the
    # driven close on the selected arm only.
    from tests.environments.robotwin.test_robotwin_motion_v2 import transaction_plant, transaction_session
    plant = transaction_plant()
    backend, calls = transaction_session(tmp_path / 'backend', plant)
    receipt = backend.execute(close)['receipt']
    assert receipt['schema'] == 'ora.robotwin-sparse-motion-receipt.v7'
    assert receipt['target_arms'] == [arm]
    assert set(plant.gripper_writes) == {arm}
    assert calls == []


@pytest.mark.parametrize('phase', ['preplace', 'lower', 'exit'])
def test_place_consumes_persisted_target_pose_and_candidate_arm(tmp_path, phase):
    from ora.agent.composition import resolve_tool_surface

    runtime = _runtime(tmp_path)
    runtime.surface = resolve_tool_surface('robotwin_toolchain_b2_closed_noservo_v1')
    runtime.allowed_tools = tuple(item['name'] for item in runtime.surface.definitions)
    root = runtime.artifact_root / 'placement'
    root.mkdir(parents=True)
    (root / 'place-check-persisted.json').write_text(json.dumps({
        'identity': {'observation_id': _observation_id(runtime), 'arguments': {'phase': phase, 'arm': 'left'}},
        'computation': {'phase': phase, 'candidates': [{'candidate_id': 'P0', 'target_pose': PLACE}]},
    }))
    arguments = {'candidate_id': 'P0', 'rationale': 'selected lower candidate'}
    if phase != 'lower':
        with pytest.raises(ToolError) as error:
            runtime.invoke('place_execute', arguments)
        assert error.value.constraints['required_phase'] == 'lower'
        assert not (runtime.control_root / 'submitted-action.json').exists()
    else:
        runtime.invoke('place_execute', arguments)
        action = _submitted(runtime)
        descend = _driven(runtime, action, 'descend', 1, {'observation_id': 'fresh'})['action']
        assert descend['arguments']['targets'] == {'left': PLACE}


@pytest.mark.parametrize("stage_status", ["terminated", "skipped"])
def test_terminal_world_releases_skill_commitment_without_abort_or_resume(tmp_path, stage_status):
    runtime = _runtime(tmp_path)
    _publish_grasp_candidate(runtime, observation_id=_observation_id(runtime))
    runtime.invoke("grasp_execute", {
        "candidate_id": "grasp-a1", "rationale": "clear approach corridor",
    })
    action = _submitted(runtime)
    receipt = runtime.build_skill_receipt(
        action=action, status="terminated", observation={"observation_id": "fresh-1"},
        stop_reason="environment_terminal",
        stages=[{
            "stage": "approach", "status": stage_status, "reason": "environment_terminal",
            "action_id": None if stage_status == "skipped" else f"{action['action_id']}:00-approach",
            "measurements": [],
        }],
    )
    state = skill_driver.load_ledger(runtime)
    assert state["phase"] == "terminated"
    assert skill_driver.reconcile(runtime)["phase"] == "terminated"
    assert receipt["status"] == "terminated"
    assert receipt["ledger_released"] is True
    from ora.agent.tools import skill_ledger, skill_receipt
    assert skill_ledger.pending_stage(state) is None
    with pytest.raises(ToolError, match="only a stopped skill"):
        skill_ledger.resume(state)
    assert receipt["delivery"]["body_bytes"] == skill_receipt.body_bytes(receipt)
