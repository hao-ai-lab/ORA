# grasp_execute c1 -- the prepared closed-skill grasp, recomposed.
#
# The same commitment, probe, stages and evidence as the native prepared grasp
# (skill_driver begin_skill / drive_skill_stage with prepared_grasp_execution_v2):
# the candidate's pregrasp and contact poses, a 3 cm world-up lift at the
# contact orientation, the parallel-jaw pre-commit probe when the surface
# declares it, the post-close baseline, and the verify stage's closure and lift
# evidence, plus the fused check.hold verdict under verification.hold (spec
# 4.6). Equivalence with the native sequence is a layer-3 test.

TOOL = composed_tool(
    name = "grasp_execute",
    version = "c1",
    family = "grasp",
    summary = "Execute your selected grasp through open, pregrasp, approach, close, lift and verify.",
    use_when = "grasp_plan returned candidates on the current observation and you selected one, the gripper may open at its current position, and a short upward probe is appropriate; use manual motion for articulation or a grasp that must not lift.",
    does_not = [
        "change the candidate, retry a stage, or stop for a decision halfway",
        "infer an empty gripper before opening; opening releases anything held",
        "certify attachment, stable transport or task success; a completed sequence only records execution",
    ],
    inputs = {
        "candidate_id": candidate_input(
            "grasp",
            "The candidate_id of one candidate from a grasp_plan call on the current observation. " +
            "A candidate from an earlier observation is refused as stale rather than executed against " +
            "geometry the robot has moved through.",
        ),
        "rationale": rationale_input(),
    },
    stages = [
        stage("open", primitive = "gripper", summary = "open the gripper, releasing anything held"),
        stage("pregrasp", primitive = "move", summary = "move to the candidate's pregrasp pose in free space"),
        stage("approach", primitive = "move", summary = "follow the approach to the contact pose"),
        stage("close", primitive = "gripper", summary = "close the gripper"),
        stage("lift", primitive = "move", summary = "request 3 cm world-up at the same orientation"),
        stage("verify", primitive = "check", summary = "measure jaw separation beside the target's visible extent and the lift's effect"),
    ],
    budgets = budgets(primitive_actions = 5, queries = 5),
    arm_profiles = ["single_arm", "dual_arm"],
    refusals = [
        refusal("stale_candidate_observation", "a candidate from an earlier observation"),
        refusal("candidate_unavailable", "a candidate the planner did not publish"),
        refusal("candidate_geometry_unavailable", "a pregrasp or contact pose that is missing, non-finite or not a unit quaternion, that differs in orientation or does not stand off, or a pose outside the action workspace"),
        refusal("grasp_pregrasp_unreachable", "either probed pose unreachable, or the two reachable only under different parallel-jaw equivalents"),
        refusal("environment_contract", "a backend equivalent that moved a committed position", retryable = False),
    ],
    refers_to = ["grasp_plan"],
    claim_boundary = "one committed sequence of trusted executions and the public measurements taken after it; motion arrival and jaw separation beside visible extents are evidence, not grasp, placement, retention or official task success",
    evidence = ["precommit_motion_check"],
)

def _refused(refusal):
    return sdk.refuse(
        refusal["code"], refusal["message"], field = refusal["field"],
        retryable = refusal["retryable"], constraints = refusal["constraints"],
    )

def plan(ctx, inputs, params):
    found = sdk.grasp.target(inputs["candidate_id"])
    if not found["ok"]:
        return _refused(found["refusal"])
    probe = sdk.grasp.precommit_probe(found["target"])
    if not probe["ok"]:
        return _refused(probe["refusal"])
    target = found["target"]
    evidence = {"precommit_motion_check": probe["evidence"]} if probe["checked"] else {}
    return sdk.commit(target = target, poses = probe["poses"], arm = target["arm"], evidence = evidence)

def execute(ctx, c):
    arm = c["arm"]
    poses = c["poses"]
    steps = [
        ("open", lambda: sdk.gripper(arm, "open", stage = "open")),
        ("pregrasp", lambda: sdk.move(arm, poses["pregrasp"], interaction = "free_space", stage = "pregrasp")),
        ("approach", lambda: sdk.move(arm, poses["approach"], interaction = "contact_approach", stage = "approach")),
        ("close", lambda: sdk.gripper(arm, "close", stage = "close")),
    ]
    for name, act in steps:
        if act()["status"] != "executed":
            return sdk.abort(name + "_not_reached")
    sdk.capture("post_close_before_lift")
    if sdk.move(arm, poses["lift"], interaction = "precision", stage = "lift")["status"] != "executed":
        return sdk.abort("lift_not_reached")
    sdk.check.grasp(c["target"], stage = "verify", hold = True)
    return sdk.complete()
