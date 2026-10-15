"""RoboDojo admission of the children of a trusted action sequence.

The session brackets a closed-skill sequence or a v4 target state, and admits
each child against the innermost frame. A closed skill's children are
enumerated in advance from its stage list, and a target-state phase must be
exactly its committed command.
"""

from __future__ import annotations

from typing import Any, Mapping

from ora.contracts.action import MOTION_ACTION_V3_SCHEMA

from .bridge import RobodojoError

#: The sequence parents the session brackets: closed skills and v4 target states.
SEQUENCE_PARENT_SCHEMAS = frozenset({"ora-skill-sequence-action.v1", "ora-action.v4"})


def admit_sequence_child(frame: dict[str, Any], action: Mapping[str, Any], *, action_codec: str) -> None:
    """Admit one child of the innermost sequence frame or raise RobodojoError."""

    parent = frame["parent"]
    if any(action[key] != parent[key] for key in ("observation_id", "bound_observation_sha256")):
        raise RobodojoError("sequence child changed its observation binding")
    identity = action["action_id"]
    remaining = frame["children"][frame["next"]:]
    if identity not in remaining:
        raise RobodojoError("sequence child is uncommitted, duplicated or reordered")
    index = frame["children"].index(identity)
    expected = frame["exact"].get(identity)
    if expected is not None and action != expected:
        raise RobodojoError("target-state phase changed its committed command")
    if expected is None:
        # The shared semantic skill driver emits sparse v3 directly;
        # legacy target-state stages enter a nested, exact phase frame.
        sparse_stage = (
            action_codec == "motion_action_v3"
            and action["schema"] == MOTION_ACTION_V3_SCHEMA
            and action["action"] in {"move_pose", "set_gripper"}
        )
        if action["schema"] != "ora-action.v4" and not sparse_stage:
            raise RobodojoError("skill stage requires target-state v4 or supported sparse motion v3")
    frame["next"] = index + 1
