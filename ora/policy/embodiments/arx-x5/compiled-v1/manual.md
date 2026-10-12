# Embodiment Card v1

## Body and command frame

This deployment uses a 6-DoF serial arm and a 6-DoF serial arm on a fixed base.
Policy motion controls `left_link6`: the model-defined tool center point.
Policy motion controls `right_link6`: the model-defined tool center point.
Position values use metre in the world frame. Measured controlled-point poses use quaternion order `[x,y,z,w]`; advertised tools define their own action-orientation arguments.
The resolved ToolSurface owns exact action arguments, enforced limits and result thresholds.

## Cameras and policy-visible state

- `head` is fixed in the scene and provides RGB and metric depth; use calibration supplied with each observation.
- `left_wrist` is moving with a manipulator and provides RGB and metric depth; use calibration supplied with each observation.
- `right_wrist` is moving with a manipulator and provides RGB and metric depth; use calibration supplied with each observation.
- `eef_pose_xyz_xyzw` reports the measured controlled-point pose in world-frame metres and an `xyzw` quaternion from the current observation.

## Motion and geometry

The policy action abstraction is cartesian target.
Internal controller increments and gains are not the policy action contract.
Use the current Runtime-enforced bounds; do not assume a universal reach box.
Use bounded motions and current execution evidence; source geometry is not a collision guarantee.
Treat camera calibration (`camera_calibration`) and workspace bounds (`workspace_bounds`) as dynamic facts from the current Runtime, observation or ToolSurface.

## Evidence limits

- Motion completion does not establish contact.
- A commanded gripper closure does not identify contact.
- Gripper closure or aperture alone does not prove object retention.
- A geometric target is not an executed state or a collision guarantee.
