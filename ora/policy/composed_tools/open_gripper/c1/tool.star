# open_gripper c1 -- the minimal composed tool: one gripper stage.
#
# Recomposes the atomic open_gripper (explicit_motion_v1/v2) over the SDK:
# one open directive on the chosen arm, one robot decision, one staged
# receipt. It exists to exercise the toolchain end to end, not to replace
# the atomic command on any surface.

TOOL = composed_tool(
    name = "open_gripper",
    version = "c1",
    family = "gripper",
    summary = "Open the selected gripper.",
    use_when = "you are choosing to release an object or to reset the jaws, and nothing else should move.",
    does_not = [
        "establish separation, stable support, correct placement or task success",
        "report jaw separation or what, if anything, was released",
    ],
    inputs = {
        "arm": arm_input(),
        "rationale": rationale_input(),
    },
    stages = [
        stage("open", primitive = "gripper", summary = "open the gripper"),
    ],
    budgets = budgets(primitive_actions = 1),
    arm_profiles = ["single_arm", "dual_arm"],
)

def plan(ctx, inputs, params):
    return sdk.commit(arm = inputs["arm"])

def execute(ctx, c):
    opened = sdk.gripper(c["arm"], "open", stage = "open")
    if opened["status"] != "executed":
        return sdk.abort("open_not_reached")
    return sdk.complete()
