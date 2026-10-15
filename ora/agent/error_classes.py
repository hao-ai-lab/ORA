"""Which tool-error codes are infrastructure failures, in one table.

A dependency-free leaf module: ``agent.tools.tool_errors`` labels every
envelope ``infrastructure: true|false`` from it, and
``agent.composition.abort_evidence_schema`` counts exactly these codes as abort
evidence. The gate used to keep its own list while
infrastructure faults raised as ``invalid_arguments`` (the current grip-site
pose missing, observation and camera artifacts drifting or escaping their
root, invalid calibration matrices) never reached it; those faults now carry
codes from this table.
"""

from __future__ import annotations

#: Error codes that say the Harness, a model service, the read-only bridge or a
#: sensor failed -- never the policy's input.
INFRASTRUCTURE_CODES: dict[str, str] = {
    "internal_tool_error": "an unexpected exception inside a tool, its model service or the read-only bridge",
    "environment_contract": "the environment's read-only bridge answered outside the agreed envelope",
    "environment_unavailable": "the Harness had no current observation to bind",
    "artifact_hash_mismatch": "a Harness-written artifact no longer matched its recorded hash",
    "artifact_integrity_fault": "a Harness-written artifact was missing or outside its root",
    "invalid_camera_calibration": "a published camera calibration was not a valid camera model",
    "pose_state_unavailable": "the measured end-effector pose was unavailable",
    "embodiment_binding": "the bound embodiment facts were missing or inconsistent with the observation",
    "grounding_unavailable": "the grounding model service was not bound",
    "tool_capability_unavailable": "the bound model service lacks a capability the surface declares",
    "tool_surface_capability_mismatch": "the tool surface and the runtime disagreed",
}


__all__ = ["INFRASTRUCTURE_CODES"]
