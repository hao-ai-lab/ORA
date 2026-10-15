# RoboTwin v10 clearance and blocked recovery

The `robotwin_aloha_contact_recovery_dual_arm_v10` control profile separates
planning clearance from the physical execution boundary and adds a bounded
response to measured joint blocking. Its predecessor v9 assets remain frozen.
The v7 experiment/default bindings and live qualification are separate gates;
offline tests do not establish the physical effectiveness of retreat.

## Clearance

The source-bound robot link spheres have radius 45 mm. A new planned path
normally requires 95 mm between sphere centres; measured execution retains
the 90 mm threshold. This is robot-versus-robot clearance, not an observed
scene collision world.

When the action starts below the execution threshold, planning and local/global
execution use one immutable action-start measurement. Every stage must stay at
or above that distance minus 1 mm. The endpoint must increase the distance.
The reference is never reset at each waypoint or local iteration: many small
inward updates cannot accumulate a larger inward excursion. Receipt details
include `escape_allowed`, `current_center_distance_m`, the candidate distance
and both thresholds. An accepted escape can still be stopped by workspace,
joint, gross-motion or episode/terminal constraints.

The interval from 90 mm to 95 mm remains subject to normal planning clearance.
The current implementation does not silently add another escape exception for
that interval; its disposition remains a separate integration decision.

## Blocking and feedback

On v10, the measured convergence loop counts actual physics samples for each
commanded arm against an unchanged joint target. Sixteen consecutive samples
with joint change below 0.001 rad and error above the selected convergence
tolerance stop the remaining path with `not_reached` / `blocked`. A changed
target or renewed joint movement resets the count. The 64-step waypoint limit
remains the fallback. Earlier profiles retain their previous wait behavior.

The receipt records `blocked_at_stage`, `frozen_steps`, `progress` and
`stall_shape`, plus the arms and measured joint error supporting the decision.
Progress is the zero-based blocked stage divided by the number of planned
stages. `from_start` means the first 10%; `creeping` requires three earlier
waypoints with waits above four steps; `abrupt` requires at least twelve
earlier waypoints whose first twelve waits are at most two steps. Other
prefixes retain `stall_shape: null`. These are measured execution descriptions;
they do not identify the contacting object or prove a contact force.

## Recovery

If the preceding Move receipt was blocked, `recover_motion` first attempts a local
20 mm retreat opposite the recorded approach direction for affected selected
arms. Other selected arms hold their measured pose during that phase; gripper
directives remain unchanged. This is part of the same Supervisor-admitted
action and the same physical-step/terminal accounting.

For a global path, the direction comes from the last nonzero segment of the
validated public-TCP path at or before the blocked stage. For a local action,
it comes from the commanded local TCP displacement. The receipt preserves the
source action and segment identity. This direction is not a collision normal
and does not guarantee release.

Only a completed retreat permits global planning to the originally requested
previous/home goal. Both planners then use a fresh common measured snapshot.
The original action-start `before` state remains in the receipt. Both phases,
their control rows, their sample counts and their failures remain in canonical
evidence, including when the later global plan is rejected. A stopped or
terminal retreat does not proceed to global recovery.

A blocked stage zero or a pure rotation may supply no nonzero linear approach
segment. Missing or invalid direction evidence returns
`blocked_recovery_direction_unavailable` before any retreat write. A fresh
observation and an explicitly selected bounded local move remain available;
the Runtime does not invent a world axis or hidden contact information.
An automatic retreat outside the workspace similarly returns
`blocked_recovery_retreat_outside_workspace` without invoking global planning.

## Ownership and verification

`controllers/inter_arm_clearance.py` owns the common immutable clearance rule;
`controllers/motion_progress.py` owns measured blocking and stall descriptions.
RoboTwin planning, global convergence and local Cartesian execution consume
these rules. v10
values live in its hash-bound control profile.

`tests/environments/robotwin/test_robotwin_contact_recovery.py` uses deterministic plants to verify
escape admission, cumulative inward rejection, execution-versus-planning
thresholds, exact 16-step blocking, terminal precedence, source-derived
retreat direction, fresh planning snapshots and preserved prefix evidence.
The pinned-source RoboTwin motion sweep and v7 behavioral triggers remain
required before production activation. No live motion was run to produce
these offline checks.


## Public rejected-plan measurements

Recognized `inter_arm_collision` rejections retain their public explanation and
robot-only `clearance` measurements through the shared plan-feedback projection,
including single-plan failures and bounded candidate outcomes. `stage_index`
and `link_pair` locate the predicted rejection on the planned trajectory;
center distances and the current/required/planning/execution thresholds describe
the immutable admission decision. They are link-sphere center distances, not
an end-effector gap, measured collision or object geometry. The projection
neither changes the planning/execution margins nor permits an escape that the
controller rejected. Unknown or malformed numbers are omitted. The boundary
and Native OpenAI/Anthropic delivery regression live in
`tests/agent/context/test_plan_feedback.py`.
