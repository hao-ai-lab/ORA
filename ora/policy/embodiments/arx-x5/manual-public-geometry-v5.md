# ARX X5 compiled embodiment v2

## Identity and topology

Fixed-base ARX X5: left 6-DoF arm, right 6-DoF arm; parallel-jaw grippers.

## Controlled points

`left_link6` is the left arm's controlled point, its wrist flange, not a fingertip; commands and measured poses already refer to it, so add no offset.
`right_link6` is the right arm's controlled point, its wrist flange, not a fingertip; commands and measured poses already refer to it, so add no offset.

## Frames and units

Positions are world metres; measured quaternions are [x,y,z,w]. Transform local geometry by the current measured pose; local lengths are not world-Z corrections.

## End-effector axes

`left_link6` local approach [1, 0, 0]; jaw-closing line [0, 1, 0]. Local Z is perpendicular to both. Current world directions are supplied with measured state.
`right_link6` local approach [1, 0, 0]; jaw-closing line [0, 1, 0]. Local Z is perpendicular to both. Current world directions are supplied with measured state.

## Motion capabilities

The ToolSurface owns motion, workspace, window and acceptance limits. Holding a command does not freeze measured state.

## Gripper semantics

`left_gripper`: no single aperture is asserted across the tapered fingers.
At reference finger qpos [0, 0] m, finger collision extent along local approach from the controlled point is [71.57, 157.57] mm.
`finger_1` collision bounds in that controlled-point frame are {"X":[71.57,157.57],"Y":[0.4016,37.7516],"Z":[-30.6401,30.3599]} mm; opening direction is [0.0, 1.0, 0.0]. These are bounds at the stated reference opening, not current contact surfaces.
`finger_2` collision bounds in that controlled-point frame are {"X":[71.57,157.57],"Y":[-37.7556,-0.4056],"Z":[-30.6401,30.3599]} mm; opening direction is [0.0, -1.0, 0.0]. These are bounds at the stated reference opening, not current contact surfaces.
Nominal inner pad region in the controlled-point frame: 142.6–157.6 mm along the approach axis and -10.4–8 mm along local Z. The jaw contact centre is +0.1501 m along the approach axis from `left_link6`. Both are nominal source geometry, not a calibrated physical contact point.
Nominal full-open pad separation is 88.8 mm. Physical pad aperture remains uncalibrated. The public encoder reading is the mean of both finger joints on the normalized command scale (empty-close reference about 0.185, full-open 1); it is not measured pad separation, and a pair shifted to one side is no evidence of holding or release. A close directive is not a request for a measured zero, and reaching the encoder floor proves neither contact nor retention.
`right_gripper`: no single aperture is asserted across the tapered fingers.
At reference finger qpos [0, 0] m, finger collision extent along local approach from the controlled point is [71.57, 157.57] mm.
`finger_1` collision bounds in that controlled-point frame are {"X":[71.57,157.57],"Y":[0.4016,37.7516],"Z":[-30.6401,30.3599]} mm; opening direction is [0.0, 1.0, 0.0]. These are bounds at the stated reference opening, not current contact surfaces.
`finger_2` collision bounds in that controlled-point frame are {"X":[71.57,157.57],"Y":[-37.7556,-0.4056],"Z":[-30.6401,30.3599]} mm; opening direction is [0.0, -1.0, 0.0]. These are bounds at the stated reference opening, not current contact surfaces.
Nominal inner pad region in the controlled-point frame: 142.6–157.6 mm along the approach axis and -10.4–8 mm along local Z. The jaw contact centre is +0.1501 m along the approach axis from `right_link6`. Both are nominal source geometry, not a calibrated physical contact point.
Nominal full-open pad separation is 88.8 mm. Physical pad aperture remains uncalibrated. The public encoder reading is the mean of both finger joints on the normalized command scale (empty-close reference about 0.185, full-open 1); it is not measured pad separation, and a pair shifted to one side is no evidence of holding or release. A close directive is not a request for a measured zero, and reaching the encoder floor proves neither contact nor retention.

## Sensors and cameras

`head`: fixed RGB-D; use calibration from the same observation.
`left_wrist`: arm-mounted moving RGB-D; use calibration from the same observation.
`right_wrist`: arm-mounted moving RGB-D; use calibration from the same observation.

## Preservation and evidence limits

Robot geometry, command closure and encoder aperture establish neither contact nor retention. Endpoint arrival is not collision clearance or task success.

## Dynamic facts and provenance

Current pose, finger measurements versus command targets, world axes, calibration, remaining budgets and execution thresholds come from current observation, Runtime or ToolSurface. Asset hashes and review/verification stay manifest-only.
