# ALOHA-AgileX RoboTwin Bootstrap Card

## Embodiment

The environment uses a dual-arm ALOHA-AgileX embodiment. Each end effector is
reported as position `[x,y,z]` in metres followed by quaternion `[x,y,z,w]`.
Gripper command `0.0` is closed and `1.0` is open.

## Cameras

`head_camera` is the fixed high global view and is attached automatically for
each fresh observation. `front_camera` is a lower fixed frontal view.
`left_camera` and `right_camera` are wrist-mounted. Other policy-visible views
must be requested with `vision.inspect`; the trusted observer camera is never
policy-visible.

## Saved state

`observation.dump_state` reads the latest or a numbered immutable public state
step. It returns proprioception, the prior public action/receipt, camera
artifact indexes and one `head_camera` image. It never advances the simulator
and never contains the official success predicate or privileged object state.

## Frames

Pixels use OpenCV top-left `(u,v)`. Depth arrays contain metric metres.
`intrinsic_cv` maps camera coordinates to source-image pixels.
`extrinsic_cv` maps world/base homogeneous coordinates into camera
coordinates, so back-projection uses its inverse. Robot action positions use
the same world/base frame returned by `geometry.query`.

## Actions

`robot.submit_action` accepts exactly one of two action shapes:

- `move_pose` commands one arm to one absolute EEF pose while preserving its
  current measured gripper value.
- `set_gripper` changes one arm's gripper while fixing that arm at its current
  measured EEF pose.

The execution receipt reports the measured before/after gripper values,
closure delta/residual, and whether the unspecified dimension remained fixed.
It does not expose contacts, object ground truth, or inferred grasp success. A
successful tool call consumes the current observation; the Runtime executes it
and returns a new observation on the next outer turn.

## Safety

Remain inside the workspace bounds shown by the run plan. Prefer short,
observable motions, approach objects with clearance, and re-localize after
every motion. Do not infer that a planner receipt means the task succeeded.
