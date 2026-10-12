# RoboDojo Environment Boundary v2

The exact task language supplied for the current episode is the objective.
Current policy-visible scene evidence is authoritative; task names, layouts
and prior episodes do not supply hidden geometry or a solution.

Alongside that language, this environment supplies a public completion
contract for the current task: the official criteria restated in ordinary
words. It says what the finished result must be and nothing about how to
reach it; it is not a plan, an ordering, a hint or a solution. It adds no
requirement the criteria do not carry and withholds none that they do, and it
never stands in for the official verdict.

The scene is a simulated Isaac Sim tabletop in an episode-local world frame:
`+X` (red), `+Y` (green) and `+Z` (blue), `+Z` upward, positions in metres.
One 1.4 m by 1.1 m table stands with its top near z = 0.76 m (table body
centred at z = 0.74 m). The robot is mounted on the table's `-Y` side, so
`+Y` runs from the robot toward the far edge. The simulator's native w-first
quaternion order is never policy-visible; public orientations use the
`[x,y,z,w]` order stated by the embodiment card.

The public boundary exposes three RGB-D cameras with
per-observation calibration and robot proprioception. Camera mounting, robot
topology and command frames belong to the embodiment card.

Simulator object and actor state, ground-truth poses, segmentation, contacts,
evaluator objects, reward internals, the success-predicate implementation,
task source and the teleoperated source demonstrations are private and
unavailable.

Only RoboDojo's reward manager supplies the official Boolean verdict,
read through the advertised verdict query at a valid state boundary and
recorded by the Runtime at episode termination. Motion completion, closed
jaws, a plausible arrangement and the policy lifecycle never replace it.

The episode ends at official success, or when the native-action allowance
reported with the other remaining allowances is spent: the task's own
native-action limit, or a lower physics cap where one applies. It can also end
earlier, with the verdict false, when the task reaches a permanent official
failure condition; which conditions a task carries and their thresholds are
not policy-visible. Every robot command reserves up to a fixed number of
native actions (20) from that allowance; the receipt reports how many were
consumed, and a refused command consumes none.

Where the task's completion contract requires it, both arms must be measured
back at the poses they held when the episode reset, and they must be there at
the same instant that everything else the task requires is also true. A
finished arrangement left with an arm still out over the table is then not a
success, and neither is a retraction taken before or after that finished state
instead of together with it. Those reset poses are themselves the reference,
and the allowance around them is not policy-visible. The advertised recovery
action's home route is the deliberate way back.
