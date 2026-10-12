"""The committed-sequence action: what a single policy decision may commit to.

This action is the thing that makes "one call" true. Everything a later stage
does is authorized here, at the one moment the policy was asked, so the checks
that matter are the ones that stop a submitted action from meaning something
other than what the policy chose: a reordered sequence that releases before it
descends, a stop the policy did not request, a resume that restarts somewhere
other than the stop it is resuming from.
"""

from __future__ import annotations

import pytest

from ora.contracts import (
    SKILL_SEQUENCE_ACTION_SCHEMA, validate_runtime_action,
)
from ora.contracts.skill_sequence import (
    MAX_RATIONALE_CHARS, SKILL_SEQUENCE_STAGES, validate_skill_sequence_action,
)


def _action(**overrides):
    arguments = {
        "skill": "grasp_execute",
        "candidate_id": "grasp-abc123",
        "rationale": "G1 is the only candidate clear of the neighbouring mug",
        "stop_before_commit": False,
        "sequence": ["approach", "close", "verify"],
        "resume_from": 0,
    }
    arguments.update(overrides.pop("arguments", {}))
    payload = {
        "schema": SKILL_SEQUENCE_ACTION_SCHEMA,
        "observation_id": "observation-0001",
        "action_id": "action-3",
        "action": "execute_skill_sequence",
        "arguments": arguments,
        "bound_observation_sha256": "a" * 64,
    }
    payload.update(overrides)
    return payload


def test_the_shared_action_validator_routes_the_committed_sequence():
    """It must arrive through `validate_runtime_action`, not a private door."""

    action = validate_runtime_action(_action())

    assert action["schema"] == SKILL_SEQUENCE_ACTION_SCHEMA
    assert action["arguments"]["sequence"] == ["approach", "close", "verify"]
    assert action["arguments"]["resume_from"] == 0


def test_a_reordered_sequence_is_refused():
    """Releasing before descending is the shape this check exists to stop."""

    with pytest.raises(ValueError, match="ordered subset"):
        validate_skill_sequence_action(_action(arguments={
            "skill": "place_execute",
            "sequence": ["transport", "release", "descend", "settle", "verify"],
        }))


def test_a_stage_outside_the_skill_is_refused():
    with pytest.raises(ValueError, match="ordered subset"):
        validate_skill_sequence_action(_action(arguments={
            "sequence": ["approach", "release", "verify"],
        }))


def test_a_grasp_may_not_declare_a_stop():
    with pytest.raises(ValueError, match="pre-release stop"):
        validate_skill_sequence_action(_action(arguments={
            "stop_before_commit": True,
        }))


def test_the_stop_stage_and_the_stop_request_must_agree():
    """A stop stage nobody asked for would consult a policy expecting none."""

    with pytest.raises(ValueError, match="present exactly when"):
        validate_skill_sequence_action(_action(arguments={
            "skill": "place_execute",
            "stop_before_commit": False,
            "sequence": ["transport", "descend", "stop", "release", "settle"],
        }))


def test_a_resume_must_restart_immediately_after_the_stop():
    with pytest.raises(ValueError, match="pre-release stop"):
        validate_skill_sequence_action(_action(arguments={
            "skill": "place_execute",
            "stop_before_commit": True,
            "sequence": ["transport", "descend", "stop", "release", "settle", "verify"],
            "resume_from": 1,
        }))


def test_a_resume_after_the_stop_is_accepted():
    action = validate_skill_sequence_action(_action(arguments={
        "skill": "place_execute",
        "stop_before_commit": True,
        "sequence": ["transport", "descend", "stop", "release", "settle", "verify"],
        "resume_from": 3,
    }))

    assert action["arguments"]["resume_from"] == 3


def test_the_rationale_is_required_and_bounded():
    with pytest.raises(ValueError, match="rationale"):
        validate_skill_sequence_action(_action(arguments={"rationale": "  "}))
    with pytest.raises(ValueError, match="at most"):
        validate_skill_sequence_action(
            _action(arguments={"rationale": "x" * (MAX_RATIONALE_CHARS + 1)}),
        )


def test_no_geometry_may_ride_along_in_the_action():
    """Stage poses belong to the tool runtime that owns the candidate artifact."""

    with pytest.raises(ValueError, match="exactly"):
        validate_skill_sequence_action(_action(arguments={
            "pose": {"position_m": [0.1, 0.2, 0.3]},
        }))


def test_both_skills_declare_the_stage_vocabulary_the_ledger_uses():
    from ora.agent.tools import skill_ledger

    assert SKILL_SEQUENCE_STAGES["grasp_execute"] == skill_ledger.PREPARED_GRASP_STAGES
    assert SKILL_SEQUENCE_STAGES["place_execute"] == skill_ledger.PLACE_STAGES
