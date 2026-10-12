# __REQUIRED__ Embodiment

## Identity and topology

__REQUIRED__: Name the body, base mobility, manipulators, sides, joint DoF, and
end-effectors.

## Controlled points

__REQUIRED__: Name the public controlled point for every manipulator and what it
is not.

## Frames and units

__REQUIRED__: Define world/base/local frames, position units, orientation
representation and order, delta-frame timing, and omission semantics.

## End-effector axes

__REQUIRED__: Define approach, jaw-closing or tool axes only when source-bound;
otherwise state that current pose evidence is required.

## Motion capabilities

__REQUIRED__: Describe stable controllable dimensions, single/multi-arm
synchronization, and controller-level non-guarantees without naming tools.

## Gripper semantics

__REQUIRED__: Separate public command targets, measured state, contact, and
retention. Do not turn a private normalized actuator value into an aperture.

## Sensors and cameras

__REQUIRED__: List every advertised public camera, mounting, qualitative role,
modalities, and calibration lifetime. Keep current visibility and coverage
dynamic.

## Preservation and evidence limits

__REQUIRED__: State command-preservation semantics and distinguish planner
acceptance or endpoint arrival from collision clearance, contact, grasp, and
task success.

## Dynamic facts and provenance

__REQUIRED__: Name which facts remain observation/Runtime-owned and state the
source-bound status of stable facts without exposing repository paths or hashes
to the policy.
