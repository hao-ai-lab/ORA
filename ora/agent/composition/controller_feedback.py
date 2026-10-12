"""Bind source-owned controller facts to the maintained Dojo tool surface."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from ora.contracts import robodojo_arx


CAPABILITY = "robodojo_controller_feedback_v1"


def bind_controller_feedback(definitions: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    """Controller rules are compiled from their owner, not copied into prompts."""
    contract = robodojo_arx.public_feedback_contract()
    rule = contract["gripper_drive_limit"]
    explanation = (
        f" Controller arrival uses position error <= {contract['position_tolerance_m']:g} m, "
        f"orientation error <= {contract['orientation_tolerance_rad']:g} rad and normalized gripper error "
        f"<= {contract['gripper_tolerance_normalized']:g}, within at most "
        f"{contract['native_actions_per_command']} native actions per command. "
        "Gripper feedback distinguishes requested directive, applied drive target, measured opening (both "
        "fingers' mean) and effective arrival target (the actual joint-limit floor for close). "
        "For explicit gripper commands, the SDK limits the applied target to within "
        f"{rule['maximum_normalized_offset']:g} normalized units of the current FIRST finger encoder, "
        "then derives both mimic drive targets from that limited value. Repeating close with an unchanged "
        "first-finger measurement does not request a more closed applied endpoint. A pose-only action "
        "holds the last APPLIED gripper target; it does not continue recomputing the close limiter. "
        "A blocked close can therefore remain a requested close while the applied target is intermediate; "
        "neither encoder arrival nor an intermediate target establishes contact or retention. "
        "Exact current targets, residuals and thresholds accompany observation and action feedback. "
        "Controller feedback contract: " + json.dumps(contract, sort_keys=True, separators=(",", ":"))
    )
    # One owning tool carries the static rule. Other motion tools identify the
    # held-target semantics without repeating the complete controller table.
    gripper = next((item for item in definitions if item["name"] == "set_gripper"), None)
    if gripper is None:
        from .tool_surface import ToolSurfaceError
        raise ToolSurfaceError("ARX controller feedback requires the set_gripper tool")
    gripper["description"] += explanation
    for item in definitions:
        if item["name"] in {"move_pose", "move_to", "rotate_pitch", "rotate_wrist", "recover_pose"}:
            item["description"] += (
                " The previous applied gripper drive target is held, including an intermediate "
                "SDK-limited target. Requested close and applied target are separate in current feedback."
            )
    return [
        {"slot": f"derived:{CAPABILITY}:{path.stem}",
         "ref": "ora/" + str(path).split("/ora/", 1)[1],
         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in (Path(__file__), Path(robodojo_arx.__file__))
    ]
