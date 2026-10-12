# ALOHA-AgileX compiled embodiment v2

## Identity and topology

Fixed-base ALOHA-AgileX: left 6-DoF arm, right 6-DoF arm; parallel-jaw grippers.

## Controlled points

`left_tcp` is the controlled reference, not a fingertip. Its local offset is [0.12,0,0] m from the declared manipulator flange; commands already target it. Do not add this offset again.
`right_tcp` is the controlled reference, not a fingertip. Its local offset is [0.12,0,0] m from the declared manipulator flange; commands already target it. Do not add this offset again.

## Frames and units

Positions are world metres; measured quaternions are [x,y,z,w]. Transform local geometry by the current measured pose; local lengths are not world-Z corrections.

## End-effector axes

`left_tcp` local approach [1, 0, 0]; jaw-closing line [0, 1, 0]. Current world directions are supplied with measured state.
`right_tcp` local approach [1, 0, 0]; jaw-closing line [0, 1, 0]. Current world directions are supplied with measured state.

## Motion capabilities

The ToolSurface owns motion, workspace, window and acceptance limits. Holding a command does not freeze measured state.

## Gripper semantics

`left_gripper`: no single aperture is asserted across the tapered fingers.
At reference finger qpos [0, 0] m, finger collision extent along local approach from the controlled point is [-50.43, 35.57] mm.
At that reference opening, the mesh centerline gap at TCP depth is 4.8964 mm; the gap varies with depth and is not a general minimum grasp width.
`right_gripper`: no single aperture is asserted across the tapered fingers.
At reference finger qpos [0, 0] m, finger collision extent along local approach from the controlled point is [-50.43, 35.57] mm.
At that reference opening, the mesh centerline gap at TCP depth is 4.8964 mm; the gap varies with depth and is not a general minimum grasp width.

## Sensors and cameras

`head_camera`: fixed RGB-D; use calibration from the same observation.
`front_camera`: fixed RGB-D; use calibration from the same observation.
`left_camera`: arm-mounted moving RGB-D; use calibration from the same observation.
`right_camera`: arm-mounted moving RGB-D; use calibration from the same observation.

## Preservation and evidence limits

Robot geometry, command closure and encoder aperture establish neither contact nor retention. Endpoint arrival is not collision clearance or task success.

## Dynamic facts and provenance

Current pose, finger measurements versus command targets, world axes, calibration, remaining budgets and execution thresholds come from current observation, Runtime or ToolSurface. Asset hashes and review/verification stay manifest-only.
