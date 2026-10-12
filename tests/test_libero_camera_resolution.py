"""The rendered camera size and the intrinsics that deproject it must agree.

``harness_camera_meta`` in ``libero_pro/session.py`` computes the intrinsic
matrix that every depth deprojection, mark, grasp pose and placement target is
derived from. It holds its own literal ``256`` rather than reading the size the
environment was actually constructed with. The first test records that: it is
the reason raising the render size is not a one-line change, because a 3x3
matrix computed for the wrong image size is wrong in a way that looks entirely
plausible -- plausible focal lengths, plausible principal point, every
deprojected metre off by the ratio. The last test keeps the render guard that
refuses a corrupted offscreen frame.
"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
# The construction parameters live in ``common.py`` and the worker wrapper's
# intrinsics in ``worker_environment.py``; read the session module set.
SESSION_MODULES = tuple(
    ROOT / "ora/environment/backends/libero_pro" / name
    for name in ("session.py", "common.py", "worker_environment.py")
)


# --------------------------------------------------------------------------
# Characterization: why this cannot be a one-line constant change.
# --------------------------------------------------------------------------

def test_todays_intrinsics_hold_their_own_literal_size():
    """``harness_camera_meta`` does not read the size it was constructed with.

    Two independent ``256`` literals govern one image: the construction
    parameters and the intrinsics. Changing only the first would leave every
    intrinsic describing an image that no longer exists.
    """

    source = "\n".join(path.read_text(encoding="utf-8") for path in SESSION_MODULES)
    construction = re.search(r'"camera_heights": (\d+)', source)
    assert construction is not None and construction.group(1) == "256"

    meta_body = source[source.index("def harness_camera_meta"):]
    meta_body = meta_body[: meta_body.index("@property", 10)]
    assert re.search(r"^\s+height = 256$", meta_body, re.MULTILINE), (
        "the intrinsics still pin their own literal; raising the render size "
        "without changing this makes every deprojected metre wrong"
    )


def test_a_corrupted_offscreen_frame_is_refused_before_the_policy_sees_it(tmp_path):
    # H200 / driver 595: two EGL renderers on one GPU returned frames of
    # uninitialised memory (NaN or far-plane depth, noise or black RGB).
    import json
    import numpy as np
    from PIL import Image
    from ora.environment.backends.libero_pro.render_guard import (
        frame_corruption, observation_corruption,
    )

    scene = np.random.default_rng(0).integers(0, 255, (16, 16, 3)).astype(np.uint8)
    assert frame_corruption(np.full((16, 16), 1.2), scene, 530.49) is None
    assert frame_corruption(np.full((16, 16), np.nan), scene, 530.49) == "depth is not finite and non-negative"
    assert frame_corruption(np.full((16, 16), 529.8), scene, 530.49) == "median depth at the far plane"
    # A hand camera facing the background is a rendering (EXP12B g002 goal_swap t03).
    assert frame_corruption(np.full((16, 16), 372.0), scene, 530.49, scene_camera=False) is None
    assert frame_corruption(np.full((16, 16), np.nan), scene, 530.49, scene_camera=False) == "depth is not finite and non-negative"
    assert frame_corruption(np.full((16, 16), 1.2), np.zeros((16, 16, 3)), 530.49) == "RGB is uniform"
    camera = tmp_path / "agentview"
    camera.mkdir()
    Image.fromarray(scene).save(camera / "rgb.png")
    np.save(camera / "depth-m.npy", np.full((16, 16), np.nan, dtype=np.float32))
    (camera / "calibration.json").write_text(json.dumps({"valid_depth_range_m": {"near": 0.01, "far": 530.49}}))
    assert observation_corruption([camera / "rgb.png"]) == "agentview: depth is not finite and non-negative"
