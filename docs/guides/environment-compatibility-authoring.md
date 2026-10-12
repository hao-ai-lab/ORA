# Environment and embodiment compatibility

Audit source facts, frames and units before writing a Bundle or ToolSurface. This applies to new backends, robots, controllers, task families and upgrades of existing sources. Use the [compatibility record](../templates/environment-compatibility-record.md); it references the fact owners rather than creating independent parameter definitions.

## Order of work

1. Pin environment, task, robot, controller, asset and dependency sources.
2. Audit task completion, official verdict authority, termination and scoring units.
3. Inventory body constants, coordinate frames, physical units and normalized encodings.
4. Identify each conversion owner and verify it against independent known values.
5. Author environment/embodiment/task layers, Bundle and ToolSurface only after that audit.
6. Check fresh resolved delivery, then obtain the smallest necessary physical qualification.

A required unknown blocks its capability. Optional measurements may explicitly be unknown/unavailable, with that state preserved in the schema, help and result. Never substitute zero, a guessed constant or another arm's measurement. Existing sealed evidence and live source bindings remain unchanged.

## Task and verdict audit

Record benchmark/revision, suite/task/variant, seed/layout, reset and randomization. Read the full official predicate and its called helpers in the trusted adapter review. Distinguish event-latched success, current-state success and terminal-state success; identify any release, stability or return-to-home requirement.

Publish qualitative completion conditions in the episode task. Keep hidden coordinates, thresholds, object GT, contacts, expert code and predicate implementation private. A successful grasp, motion arrival or tool call does not imply official task success.

Separate true/false/unknown verdicts, environment termination, policy stop, budget exhaustion and infrastructure faults. State episode/round/seed/task denominators, exclusions and retries. K rounds in one persistent session are not K independent episodes. Keep model calls, tool calls, Supervisor decisions, native actions, physics steps, reset work and wall time distinct.

## Constants and dimensional conversions

| Category | Required facts |
|---|---|
| Topology | Body/version, arm names, joint names/order/count, active and passive joints. |
| Controlled point | Actual controlled link/EEF, TCP/contact center and camera frames. |
| Frames | World/base/episode/tool/camera axes, handedness, origin and transform direction. |
| Motion | Absolute/delta/velocity semantics, orientation encoding, limits, bounds and hold rules. |
| Gripper | Command versus measurement, opening direction, encoder zero, full versus half width, calibration. |
| Geometry | Link-to-TCP/contact transforms, nominal finger/pad geometry and the controller's consumed approximation. |
| Sensors | Mounts, resolution, RGB order, depth scale/invalid values, intrinsics/extrinsics and timing. |
| Time | Control dt, simulation dt/substeps, sampling boundaries and reset/episode accounting. |

For every constant record source revision/location, value/unit, frame/arm, conditions, nominal/calibrated/measured status, unique configuration owner, actual consumers and verification evidence. Similar numbers are not necessarily the same physical quantity.

For every conversion record source field/unit/encoding → formula → target field/unit/range → owning function → invalid/saturation behavior → independent known-value check. Convert once, not independently in the codec, tool, Supervisor and worker.

Check meters versus millimeters, degrees versus radians, Euler order, xyzw versus wxyz, joint order, full versus single-finger aperture, normalized UV versus pixels, resize/crop intrinsics, and native versus physics steps. Normalization is an encoding, not a physical unit. Use a linear mapping only when calibration establishes it; otherwise retain explicitly named raw readings and unknown physical quantities.

For a link-local offset, `p_contact_world = p_link_world + R_world_link * t_link_contact`. Adding a world-Z constant is not an equivalent transformation under rotation. Test nonzero values, nonidentity poses, both arms, boundaries and invalid inputs.

## Delivery and acceptance

Embodiment owns stable body facts; environment owns lifecycle/verdict authority; episode task owns completion conditions; ToolSurface owns syntax, units and tool-local recovery; dynamic context owns measurements and receipts. Planner/system text owns its role and protocol. See [Bundle authoring](prompt-bundle-authoring.md) and [ToolSurface authoring](tool-surface-authoring.md).

Verify every fact through its actual consumers: descriptor/card, profile, Bundle, tool view/compiler, worker/controller, receipt, observation and Viewer/export where applicable. A correct source file is insufficient if the policy never receives it.

Test verdict true/false/unknown, terminal action refusal, failed-stage skipping and close behavior. Record actual CPU/GPU/memory/slot capacity, parallel schedule, expected wall time and evidence reuse. New physical execution needs a clean committed source and fresh preflight; this guide does not create launch authority.
