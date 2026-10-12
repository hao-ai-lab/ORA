# RoboTwin Policy Boundary v1

RoboTwin supplies policy-visible RGB-D observations and proprioception through
the Robot Harness. Simulator object state, segmentation, contacts, task source,
expert programs, and the success predicate remain outside the policy boundary.

Only RoboTwin `eval_success/check_success` provides the official task verdict.
A successful motion receipt means the action executed; it does not mean the
task succeeded.
