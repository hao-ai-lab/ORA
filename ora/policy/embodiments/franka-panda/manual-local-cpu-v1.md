# Franka Panda local OSC embodiment

## Identity and topology

One fixed-base, seven-degree-of-freedom Franka Panda arm has one symmetric
parallel-jaw gripper. This composition uses a bounded Cartesian controller.

## Controlled points

The controlled point is the RoboSuite grip site between the fingers, not the
hand-body origin, a fingertip contact point, or a point on an object.

## Frames and units

Absolute positions use world-frame metres and quaternions use `[x,y,z,w]`.
Translation deltas use the units and frame explicitly advertised in the tool
schema. World deltas follow fixed world axes; grip-site deltas follow the
measured local axes at action start. Omitted pose dimensions hold their
measured action-start values. Image pixels start at the top-left.

## End-effector axes

The grip-site frame has local `+X`, `+Y`, and `+Z` axes. An absolute tool-axis
direction is expressed in the world frame. A top-down orientation normally has
local `+Z` pointing toward world `-Z`. Tool-axis directions and a quaternion are
alternative orientation representations.

## Motion capabilities

The controller tracks a bounded Cartesian pose target with operational-space
control. It can change position, orientation, or both. Relative targets resolve
from the measured action-start pose. Candidate images project a target into
the measured scene; they do not predict physical execution. No global motion
planner, collision planner, learned grasp service, or visual policy is present.
Use the advertised action workspace and controller outcome as the executable
bounds. Simulator time advances during executed controls, not model thinking.

## Gripper semantics

The fingers open and close symmetrically around the grip site. A pose command
holds the current gripper directive; a gripper command holds the current pose
target. Contact and dynamics can still change measured state. Reported
aperture is a nominal proxy: normalized gripper state multiplied by 80 mm,
not an independently calibrated physical jaw-distance measurement. Aperture
alone proves neither contact nor retention.

## Sensors and cameras

`agentview` is a fixed RGB-D scene camera. `wrist` is a moving eye-in-hand RGB-D
camera. Point measurements use the selected actual observation's depth and
calibration. A wrist measurement must not reuse calibration across an action.
Derived point-cloud views are renderings of measured camera depth, not extra
sensors or object segmentation.

## Preservation and evidence limits

Holding a command target does not freeze physical state. A completed action's
receipt and next actual observation describe execution. Endpoint arrival is
not collision clearance, contact, grasp retention, correct placement, or task
success. Only the environment's official termination supplies task success.

## Dynamic facts and provenance

Topology and frame conventions come from the pinned RoboSuite Panda and
environment sources. The source-bound control profile identifies their executable
owners. Current camera calibration, pose, nominal aperture, workspace bounds,
tracking residuals, and action outcome remain observation or Runtime facts.
No object ground truth, task predicate, contact labels, or expert actions are
provided to the policy.
