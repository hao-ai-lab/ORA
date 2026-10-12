# close_gripper c1 -- the minimal composed tool: one gripper stage.
#
# Recomposes the atomic close_gripper (explicit_motion_v1/v2) over the SDK:
# one close directive on the chosen arm, with the optional target_ref echoed
# as an agent annotation. The holding-gate variant, where a surface requires
# a current target_ref before closing, is not part of c1.

TOOL = composed_tool(
    name = "close_gripper",
    version = "c1",
    family = "gripper",
    summary = "Close the selected gripper.",
    use_when = "the jaws are positioned around what you intend to hold, or another close is useful; repeat a close only after a fresh observation.",
    does_not = [
        "require segmentation or check the target_ref annotation against anything",
        "establish contact, attachment, retention or task success",
        "report jaw separation; a closed jaw is not a grasp",
    ],
    inputs = {
        "arm": arm_input(),
        "rationale": rationale_input(),
        "target_ref": annotation_input("What you believe the jaws are closing on."),
    },
    stages = [
        stage("close", primitive = "gripper", summary = "close the gripper"),
    ],
    budgets = budgets(primitive_actions = 1),
    arm_profiles = ["single_arm", "dual_arm"],
)

def plan(ctx, inputs, params):
    return sdk.commit(arm = inputs["arm"])

def execute(ctx, c):
    closed = sdk.gripper(c["arm"], "close", stage = "close")
    if closed["status"] != "executed":
        return sdk.abort("close_not_reached")
    return sdk.complete()
