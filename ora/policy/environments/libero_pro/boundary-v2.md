# LIBERO-Pro Environment Boundary v2

This layer defines the environment boundary, not robot controls or a task
strategy.

The exact episode task is the objective. Current observations and advertised
tool results are the only public evidence about the current scene. Camera
mounting, command frames and robot geometry are defined by the embodiment and
tool contracts.

The following are private: BDDL contents, initialization regions, object
ground-truth poses or identities, segmentation, contacts, demonstrations,
expert actions, benchmark source and the success-predicate implementation. Do
not request or reconstruct private state from suite, task or file names.

Only LIBERO environment termination supplies the official Boolean verdict.
Motion results and the policy lifecycle report do not replace that verdict.
The ToolSurface determines whether an official verdict is visible to the
policy. An in-episode completion aid, where one is advertised, is advisory and
does not bind that verdict in either direction.

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
