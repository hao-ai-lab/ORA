"""RoboDojo admission of the children of a trusted action sequence.

The session brackets a closed-skill sequence, a composed tool or a v4 target
state, and admits each child against the innermost frame. A closed skill's
children are enumerated in advance from its stage list, and a target-state
phase must be exactly its committed command. A composed body may retry a
stage or skip one, so its children cannot be enumerated: what is admitted is
``<parent>:<index:02d>-<stage>`` (``.<attempt>`` on a retry) naming a
committed stage at or after the last one, attempts counted one by one, each
as sparse motion v3 pose or gripper motion.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from ora.contracts.action import MOTION_ACTION_V3_SCHEMA

from .bridge import RobodojoError

#: The sequence parents the session brackets: closed skills, composed tools and v4 target states.
SEQUENCE_PARENT_SCHEMAS = frozenset({
    "ora-skill-sequence-action.v1", "ora-composed-tool-action.v1", "ora-action.v4",
})


def admit_composed_child(frame: dict[str, Any], action: Mapping[str, Any], *, action_codec: str) -> None:
    parent = frame["parent"]
    match = re.fullmatch(re.escape(parent["action_id"]) + r":(\d{2})-([a-z0-9_]+)(?:\.(\d+))?", action["action_id"])
    sequence = parent["arguments"]["committed_sequence"]
    if match is None:
        raise RobodojoError("composed child id does not name a committed stage")
    index, stage, attempt = int(match.group(1)), match.group(2), int(match.group(3) or 1)
    if index >= len(sequence) or sequence[index] != stage:
        raise RobodojoError("composed child names a stage outside its commitment")
    last_index, last_attempt = frame["composed"]
    if index < last_index or (index == last_index and attempt != last_attempt + 1) or (
        index > last_index and attempt != 1
    ):
        raise RobodojoError("composed child is duplicated or reordered")
    if not (action_codec == "motion_action_v3" and action["schema"] == MOTION_ACTION_V3_SCHEMA
            and action["action"] in {"move_pose", "set_gripper"}):
        raise RobodojoError("composed stage requires supported sparse motion v3")
    frame["composed"] = (index, attempt)


def admit_sequence_child(frame: dict[str, Any], action: Mapping[str, Any], *, action_codec: str) -> None:
    """Admit one child of the innermost sequence frame or raise RobodojoError."""

    parent = frame["parent"]
    if any(action[key] != parent[key] for key in ("observation_id", "bound_observation_sha256")):
        raise RobodojoError("sequence child changed its observation binding")
    identity = action["action_id"]
    if frame.get("composed") is not None:
        return admit_composed_child(frame, action, action_codec=action_codec)
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
