# Synthetic Target-State Embodiment v1

The qualification stand exposes one controlled grip site and a parallel-jaw
gripper. Positions are world-frame metres and orientations are quaternions
`[x,y,z,w]`. The resolved ToolSurface defines the exact controllable fields and
workspace limits.

For Runtime compatibility, the public qualification observation carries both
`left` and `right` end-effector state. In this embodiment the resolved action
surface controls `right` as the single grip site. Treat `left` as a
non-actionable qualification fixture, not as a second available arm.

`agentview` is a fixed scene camera and `wrist` moves with the grip site. Both
provide aligned RGB, metric depth and current calibration through the public
observation boundary. Pixels use top-left `(u,v)` coordinates. A marked scene
surface is not automatically the controlled grip site or an object grasp point.

The stand validates target-state submission and observation sequencing; it
does not establish contact, grasp retention, placement or task success. Verify
only claims supported by a fresh public observation.
