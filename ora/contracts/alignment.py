"""Versioned alignment-request envelope shared by Agent and Environment.

The policy side synthesises approach poses and the environment side ranks them
by inverse kinematics. Both need the same field names, the same side vocabulary
and the same batch ceiling, and neither may import the other: ``agent`` is
forbidden from importing ``ora.environment`` outright, not merely from
importing a backend. So the shared half lives here, next to the action envelope
it travels beside, and carries no session, adapter or backend implementation.
"""

from __future__ import annotations

REQUEST_SCHEMA = "ora.align-plan-request.v1"
RESULT_SCHEMA = "ora.align-plan-result.v1"

#: Named directions toward the target object, never toward an arm. A side is a
#: property of the object being approached, so the vocabulary is the same on a
#: single-arm and a dual-arm embodiment.
APPROACH_SIDES = ("top", "left", "right", "front", "back")

#: One request may carry this many candidate poses. The ceiling exists because
#: the batch is answered from forward kinematics per pose; it is a cost bound,
#: not a claim that more poses would be less correct.
MAXIMUM_POSES = 12

REQUEST_FIELDS = frozenset(
    {"schema", "target_ref", "approach_side", "arm", "target_poses_xyzw"}
)


class AlignmentRequestDrift(ValueError):
    """The request shape is not the one both sides agreed to serve.

    Distinct from ``EnvironmentRefusal`` on purpose. A refusal is something the
    policy can correct and is delivered to it as retryable ``invalid_arguments``;
    drift means the two sides were built from different revisions of this file,
    which no policy action can repair.
    """
