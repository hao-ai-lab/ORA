"""Constants and helpers shared by ``core`` and its tool-family mixins.

``core`` re-exports the constants and pose helpers, so imports from it
keep working.
"""

from __future__ import annotations

from typing import Any

from ora.contracts.pose import matrix_quaternion, quaternion_matrix


# RoboDojo publishes three metric RGB-D cameras under its own names; the
# capability gates them exactly like robotwin_rgbd_v1 gates the RoboTwin set.
ROBODOJO_POLICY_CAMERAS = frozenset({"head", "left_wrist", "right_wrist"})


RETENTION_REVIEW_CAMERA_ORDER = (
    "observer_camera", "head_camera", "front_camera",
    "left_camera", "right_camera",
    "head", "left_wrist", "right_wrist",
)


def _quat_matrix(xyzw: list[float]) -> Any:
    import numpy as np

    return np.asarray(quaternion_matrix(xyzw), dtype=float)


def _matrix_quat(matrix: Any) -> list[float]:
    return matrix_quaternion(matrix.tolist())


def load_policy_state(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Call ``core.load_policy_state``, looked up when called.

    Tests replace ``load_policy_state`` and ``verify_policy_state`` on the
    ``core`` module, some of them with a stand-in host object that has none
    of the runtime's methods, so the mixins that read policy state import these
    two forwarders instead of the evidence functions.
    """

    from ora.agent.tools.runtime import core

    return core.load_policy_state(*args, **kwargs)


def verify_policy_state(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Call ``core.verify_policy_state``, looked up when called."""

    from ora.agent.tools.runtime import core

    return core.verify_policy_state(*args, **kwargs)
