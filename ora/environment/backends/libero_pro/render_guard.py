"""Refuse LIBERO observations whose offscreen frames are not a rendering.

On H200 nodes (driver 595), two processes rendering through one EGL
device, or EGL beside SAPIEN Vulkan load on the same GPU, returned frames of
uninitialised memory: NaN or out-of-range depth, depth pinned at the far plane,
and noise or near-black RGB. The worker checks each saved LIBERO observation
before the policy sees it and stops the episode as an infrastructure fault.

Kept out of ``session.py``: that file is a content-addressed provenance source
of the Panda embodiment descriptors, so the check lives at the worker edge.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence


#: Hand cameras may legitimately face the distant background (EXP12B g002
#: goal_swap t03: the wrist view of the wall had median depth 372 m), so the
#: far-plane test applies only to the fixed scene cameras that always see the table.
HAND_CAMERAS = frozenset({"wrist", "robot0_eye_in_hand"})


def frame_corruption(metric_depth: Any, rgb: Any, far_m: float | None,
                     *, scene_camera: bool = True) -> str | None:
    """Why one saved frame is not a rendering of the scene, else None."""
    import numpy as np

    depth = np.asarray(metric_depth, dtype=np.float64)
    if depth.size == 0 or not np.isfinite(depth).all() or depth.min() < 0.0:
        return "depth is not finite and non-negative"
    if scene_camera and far_m is not None and float(np.median(depth)) >= 0.5 * float(far_m):
        return "median depth at the far plane"
    if float(np.asarray(rgb, dtype=np.float64).std()) < 5.0:
        return "RGB is uniform"
    return None


def observation_corruption(image_paths: Sequence[Path]) -> str | None:
    """Check every camera directory a LIBERO observe() wrote (rgb.png beside depth-m.npy)."""
    import numpy as np
    from PIL import Image

    for rgb_path in image_paths:
        camera = Path(rgb_path).parent
        calibration = json.loads((camera / "calibration.json").read_text(encoding="utf-8"))
        far = (calibration.get("valid_depth_range_m") or {}).get("far")
        reason = frame_corruption(np.load(camera / "depth-m.npy", allow_pickle=False),
                                  np.asarray(Image.open(rgb_path)), far,
                                  scene_camera=camera.name not in HAND_CAMERAS)
        if reason is not None:
            return f"{camera.name}: {reason}"
    return None
