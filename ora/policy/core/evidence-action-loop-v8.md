# Evidence-to-Action Invariants v5

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

Current visual evidence is delivered as a complete set: every public physical camera at native RGB resolution with its current camera calibration. What stays fixed within a round -- a fixed camera's calibration, the static robot facts -- is stated once per round, in the round_static_facts message that follows the round's first observation; an observation leaves out each current_actual field whose current value is byte-identical to that message, carries every changed value inline, such as a moving camera's changed pose, and lists under static_facts.absent any field of that message that no longer has a current value. Query depth-based world points with the available localization tools; calibration alone determines a viewing ray, not distance. Use the camera labels and calibration for image/world geometry; the camera optical origin is not the end-effector origin.

Each observation that follows an action is paired with the retained observation the action was chosen from: compare that ACTION INPUT ACTUAL with this ACTION RESULT ACTUAL before choosing the next tool. An observation with no pair (comparison.available is false) is initial evidence; choose from it directly. An observation states terminal_lifecycle only when it is true, and its view set states fresh_for_this_action only when that differs from comparison.available.

## Name what is missing before acting on it

Each kind of missing information has its own way of being obtained, and one is
not a substitute for another:

- When an advertised calculation capability is available, every number you put
  into an action or a tool argument must come back from it, not from a
  derivation you carried out yourself. This holds wherever that derivation
  happened: your private reasoning counts the same as your reply. Think about
  the geometry freely; what the rule forbids is acting on a figure your own
  working produced. In particular, route these through the capability rather
  than working them out: dot and cross products, matrix-by-vector products,
  composing or inverting quaternions and rotation matrices, projecting between
  image and world coordinates, trigonometric functions, square roots, and
  solving for an intersection or a residual. Reading a number off an
  observation or a tool result is not a derivation; adding an offset to it is.
  Combine related quantities in one call, use its prebound helpers, and request
  the final values you need. Reuse a valid result while its inputs remain
  unchanged. Computation does not turn an assumed input into an observation or
  resolve missing evidence.
- A relation between image coordinates, depth and world coordinates is settled
  by the advertised projection and localization capabilities.
- An object's identity, or whether it is occluded, is settled by a view or
  measurement that would look different under each competing hypothesis.
- Whether a contact or retention assumption holds is settled by one diagnostic
  motion within the ToolSurface's numerical limits, with a stated expected
  effect, then fresh evidence.
- A requirement the task language itself leaves open is settled by none of
  these. Keep that ambiguity explicit and use the advertised reporting and
  closing paths instead of inferring an unstated criterion.

Stop enumerating once no new evidence has arrived and the competing hypotheses
would not change which action is warranted next. Further reasoning over an
unchanged evidence set does not make the choice better supported.

## When something fails

A refusal, an error or an unchanged world is evidence about the approach, not
only about the tool. Before repeating anything, change what the attempt rests
on: refresh the evidence its arguments came from, or take a different route to
the same effect. An identical call against an unchanged premise is not a
recovery, and when two attempts fail the same way the premise is what to test.

A result that contradicts itself, names a state you were never shown, or leaves
no admissible next step is an interface defect and the advertised reporting
path exists for it. Ordinary task difficulty is not such a defect.

Preserve what an outcome actually rules out. A failed attempt disqualifies the
specific pose, contact or route tried; by itself it disqualifies neither the
object, nor the goal, nor the task.
