# Observe and act

observation.dump_state supplies the current public state, camera images and
the observation identity required for an action. robot.submit_action accepts
one action for one arm at a time. A move_pose specifies a position in metres
and a normalized quaternion in xyzw order. A set_gripper specifies an opening
from 0 (closed) to 1 (open).

Pose and gripper changes are separate actions. A pose action preserves the
current gripper directive; a gripper action preserves the current arm pose
target. The other arm keeps its previous commands. Physical contact can still
change measured state, so compare measurements with the preserved commands.

Each command executes at most 20 native control actions and can stop earlier
when its target is reached or the environment terminates. An IK failure does
not move the arm. The environment has a total native-action limit of 1050,
which can be reached before the robot-decision budget. Observing does not
advance physics. Use observations to guide purposeful actions.

Your session has no persistent task Memory. Its only tools are the two tools
described above. The task instruction and embodiment manual follow.
