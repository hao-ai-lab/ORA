"""Placement refusals carry the measured quantity beside the bound (EXP11A g003, B3).

g003's five RoboDojo ``place_plan`` calls returned three ``unknown`` results and
two errors, none with the number that decided it: a basket's front handle slot
fitted as support at 65.1 degrees (bound 15), a bowl rim without enough plane
coverage, a wrist-view held object with 117,088 retained depth points (bound
100,000), and the same region named as support and boundary.
"""
from __future__ import annotations

import numpy as np
import pytest

from ora.agent.tools import placement_refusal
from ora.agent.tools.placement_depth import project_geometry
from ora.agent.tools.robot import ToolError


def test_the_retained_point_refusal_names_the_count_region_and_camera():
    """The recorded wrist-view held bowl: 117,088 mask pixels, all with valid depth."""
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask.reshape(-1)[:117_088] = 1
    depth = np.full((480, 640), 0.11, dtype=np.float32)
    intrinsic = [[320.0, 0, 320], [0, 320.0, 240], [0, 0, 1]]
    with pytest.raises(placement_refusal.RetainedPointBoundsError) as caught:
        project_geometry(mask, depth, intrinsic, np.eye(4).tolist(), observation_id="o", target_ref="t",
                         camera="right_wrist")
    assert isinstance(caught.value, ValueError)  # frozen callers still catch it as before
    error = placement_refusal.retained_point_error(caught.value, role="object", camera="right_wrist").as_dict()
    assert error["code"] == "placement_evidence_unavailable" and error["retryable"] is True
    assert error["field"] == "object_ref" and error["received"] == 117_088
    assert error["constraints"]["maximum_retained_points"] == 100_000
    assert error["constraints"]["minimum_retained_points"] == 16
    assert error["constraints"]["source_camera"] == "right_wrist"
    assert "117088" in error["message"] and "head camera" in error["message"]


def test_a_reused_region_names_the_field_and_the_reference():
    error = placement_refusal.duplicate_reference_error(
        [("object", "target-a"), ("support", "target-b"), ("boundary", "target-b")])
    assert isinstance(error, ToolError)
    value = error.as_dict()
    assert value["field"] == "destination_boundary_ref" and value["received"] == "target-b"
    assert value["constraints"] == {"same_reference_as": "destination_support_ref"}
    assert placement_refusal.duplicate_reference_error([("object", "a"), ("support", "b")]) is None
