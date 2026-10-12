"""Pure quaternion conversion and policy-safe compiler identity projection."""
from __future__ import annotations
import math
from typing import Any, Mapping

def _rotation_matrix_to_quaternion_xyzw(matrix: Any) -> list[float]:
    import numpy as np

    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (3, 3) or not np.isfinite(value).all():
        raise ValueError("grasp orientation matrix is invalid")
    trace = float(np.trace(value))
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2.0
        x = (value[2, 1] - value[1, 2]) / scale
        y = (value[0, 2] - value[2, 0]) / scale
        z = (value[1, 0] - value[0, 1]) / scale
        w = 0.25 * scale
    else:
        index = int(np.argmax(np.diag(value)))
        if index == 0:
            scale = math.sqrt(1.0 + value[0, 0] - value[1, 1] - value[2, 2]) * 2.0
            x = 0.25 * scale
            y = (value[0, 1] + value[1, 0]) / scale
            z = (value[0, 2] + value[2, 0]) / scale
            w = (value[2, 1] - value[1, 2]) / scale
        elif index == 1:
            scale = math.sqrt(1.0 + value[1, 1] - value[0, 0] - value[2, 2]) * 2.0
            x = (value[0, 1] + value[1, 0]) / scale
            y = 0.25 * scale
            z = (value[1, 2] + value[2, 1]) / scale
            w = (value[0, 2] - value[2, 0]) / scale
        else:
            scale = math.sqrt(1.0 + value[2, 2] - value[0, 0] - value[1, 1]) * 2.0
            x = (value[0, 2] + value[2, 0]) / scale
            y = (value[1, 2] + value[2, 1]) / scale
            z = 0.25 * scale
            w = (value[1, 0] - value[0, 1]) / scale
    quaternion = np.asarray([x, y, z, w], dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm <= 1e-9:
        raise ValueError("grasp orientation quaternion is degenerate")
    quaternion /= norm
    if quaternion[3] < 0:
        quaternion = -quaternion
    return [float(item) for item in quaternion]


def _public_grasp_compiler_identity(
    compiler_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Project compiler provenance without private reconstruction constants."""

    public = {
        "id": str(compiler_identity["id"]),
        "embodiment_descriptor_id": str(
            compiler_identity["embodiment_descriptor_id"]
        ),
        "controlled_point": str(compiler_identity["controlled_point"]),
        "controlled_point_semantics": str(
            compiler_identity["controlled_point_semantics"]
        ),
    }
    arm = compiler_identity.get("arm")
    if arm is not None:
        if arm not in {"left", "right"}:
            raise ValueError("grasp compiler arm is invalid")
        public["arm"] = str(arm)
    if "physical_jaw_aperture_calibrated" in compiler_identity:
        public["physical_jaw_aperture_calibrated"] = bool(
            compiler_identity["physical_jaw_aperture_calibrated"]
        )
        if not public["physical_jaw_aperture_calibrated"]:
            public["aperture_caveat"] = str(compiler_identity["aperture_caveat"])
        # Where the admission ceiling came from, and where the published
        # per-candidate opening came from, are two different provenances and
        # only one of them is an embodiment fact. Publishing both stops the
        # caveat from being read as a claim about the predicted width.
        if "maximum_opening_source" in compiler_identity:
            public["maximum_opening_source"] = str(
                compiler_identity["maximum_opening_source"]
            )
        if "predicted_opening_source" in compiler_identity:
            public["predicted_opening_source"] = str(
                compiler_identity["predicted_opening_source"]
            )
    return public
