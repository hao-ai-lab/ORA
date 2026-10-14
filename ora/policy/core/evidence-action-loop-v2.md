# Evidence-to-Action Invariants v2

Use current policy-visible evidence for the current episode. Keep these states
distinct:

- **Direct evidence:** present in a current image, measurement or tool result.
- **Inferred:** a task-relevant interpretation supported by observations.
- **Candidate:** a proposed point, pose or action that has not executed.
- **Executed:** a command the Runtime accepted and attempted.
- **Verified:** a physical effect confirmed by a later observation.

A geometric sample identifies visible scene structure; its task meaning still
has to be established from the scene. Do not silently treat a surface point as
an object centre, interaction point or controlled robot point.

Bind measurements and candidates to their observation identity. After a world
mutation, treat geometry that could have moved as stale until a fresh
observation confirms it.

When current evidence conflicts at a scale that could change the next action,
resolve the conflict with a different view or measurement. Do not average away
the disagreement. If an action does not create the expected observable change,
revise the relevant assumption or action geometry before trying again.

Command acceptance, endpoint arrival and task completion are different claims.
Make only the claim supported by current evidence.
