# RoboTwin Environment Boundary v4

The exact task language supplied for the current episode is the objective.
Current policy-visible scene evidence is authoritative; task names and prior
layouts do not provide hidden geometry or a solution.

The public boundary exposes selected rendered RGB-D sensors, calibration and
robot proprioception. Camera mounting, robot topology and command frames belong
to the embodiment card.

Simulator object state, actor identities, segmentation, contacts, task source,
expert programs, demonstrations and the success-predicate implementation are
private and unavailable.

Only the environment evaluator at episode termination supplies the official
Boolean verdict. Motion completion and the policy lifecycle never replace it.
An in-episode completion aid, where one is advertised, is advisory and does not
bind that verdict in either direction.

## What this boundary does not guarantee

Measured robot and scene state can change without a command from you. Contact,
gravity and settling all act between one observation and the next, so a changed
measurement does not prove a command took effect and an unchanged one does not
prove that nothing happened.

A rendered view reports what that sensor sees from where it is. An object
missing from one view may be outside its field, behind another object, or past
the usable depth range. Missing from a view is not missing from the scene, and
this boundary exposes no channel that could confirm a scene defect; change
viewpoint rather than reasoning about whether the rendering is intact.

Nothing here announces that the objective has become unreachable. The absence
of such a signal is not evidence that the episode is still recoverable.
