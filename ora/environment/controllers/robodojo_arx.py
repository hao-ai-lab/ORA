"""Host-safe ARX X5 controller binding and source-fact validation.

Shared values live in contracts so policy composition need not import the
Environment package. These existing controller entry points retain their API.
"""
from __future__ import annotations

from typing import Any, Mapping

from ora.contracts.robodojo_arx import (
    ARX_X5_CONTRACT, controller_binding, public_feedback_contract,
)


def validate_robot_facts(facts: Mapping[str, Any]) -> None:
    expected = {"joint_dof": 6, "ee_link": "link6", "ee_type": "gripper",
                "gripper_scale": [-0.01, 0.044], "gripper_sign": 1.0,
                "gripper_mimic_multiplier": 1.0, "gripper_mimic_offset": 0.0}
    for key, value in expected.items():
        if facts.get(key) != value:
            raise ValueError(f"ARX X5 source/instance control binding drift: {key}")
