# Policy assets

This directory is an immutable asset library, not a second Runtime and not a
list of competing defaults. Version suffixes identify the exact content that a
source-bound RunPlan used; they are deliberately retained for reproducibility.

## How one asset becomes active

The eight current Compute-bearing Panda/ALOHA profiles select versioned
`*-routing-v1-manifest.json` successors. Shared calculation routing lives in
`tool_routing_v1/shared-help.json`; backend and atomic/closed selection overlays
own the planner handoff examples. Earlier Compute v2 assets are preserved.
See [Compute usage](../../docs/guides/compute-usage.md) for the exact call and
[public interfaces](../../docs/reference/public-interfaces.md) for selection
and recovery semantics.

```text
experiment config / RunSpec
  -> orchestrator validates omitted-field defaults
  -> policy_context and tool_surface profiles select exact resources
  -> composition resolves content and hashes
  -> expanded RunPlan records every selected identity
```

There is no implicit “highest version wins” rule. A file called `*-v3` is not
active merely because it has the largest suffix. Use `ora plan` or
the experiment's `plan` command to inspect the effective composition.

## Directory ownership

| Directory | Content | Selected by |
|---|---|---|
| `core/` | Shared evidence/action-loop policy text | `policy_context` composition |
| `planners/codex/` | Codex project document assets | managed Codex composition |
| `planners/native/` | Native system/project policy assets | Native context composition |
| `embodiments/` | Robot descriptors and manuals | embodiment/manual binding |
| `environments/` | Policy-safe environment boundary/context assets | environment context profile |
| `tool_surfaces/` | Immutable tool manifests, schemas and descriptions | `policy.tool_surface.profile` |
| `baselines/` | Provider-independent product default composition | Native robot RunSpec default expansion |
| `composed_tools/` | Composed tools: `<name>/<version>/tool.star` and its generated `compiled/<arm_profile>.json`; the question bank under `questions/<id>/<version>.yaml` | a manifest's `composed_tools` binding (`robotwin_toolchain_b2_closed_composed_v1`, `robodojo_placement_closed_composed_v1`); no baseline selects them |
| `primitives/tunables/` | Primitive tunable table: the parameters a composed tool may set, with defaults, bounds and qualification | composed tool headers (`tunable(id)`) |

`embodiments/aloha-agilex/` holds the versioned dual-arm asset family:
`control-profile-v1..v9.json` and `manual-v1..v11.md` are immutable,
version-numbered treatments. Only the version a RunPlan or bundle explicitly
binds is active; earlier versions stay byte-frozen for reproducibility and are
never edited in place. This tree composes and executes control profile v10
only (`OLDEST_COMPOSED_CONTROL_PROFILE`): a RunSpec or RoboTwin session bound
to v1-v9 is refused, and a recorded run bound to one is evaluated at its source
commit. The current control profile additionally carries a
`source_binding` that content-addresses the vendored RoboTwin planner chain
(`environment/backends/robotwin/vendored/`), so a planner or installer change
requires refreshing that binding through `./scripts/refresh-vendored-pins`.

Embodiment descriptors name the repository-local Runtime sources behind each
fact by path only (`local_path`, no `revision` or `sha256`); the RunPlan's
source commit identifies their bytes. An edit or a behavior-neutral split of a
cited source no longer touches the descriptor; only a renamed or removed source
needs its `local_path` updated, which the source audit reports. External
sources (MJCF, URDF, USD, meshes, robot bindings) keep `revision` and `sha256`.

## Naming assets by path; the lock

Assets this tree composes name each other by path only: a ToolSurface
component is `"../explicit_motion_v2/robotwin-visual_help.json"`, a context
profile's compiled descriptor and wording review are repository-relative
paths, a baseline catalog's `runtime.motion_profile` and
`runtime.control_profile` are paths, the RoboTwin registry names each
`semantic_review` by path, a RoboDojo registry entry is exactly
`{record, layer}`, a descriptor names its repository sources by path and a
wording review attests only the card text (`card_sha256`). None carries a
hand-written sha256 for another asset or a repository source.
`policy.lock` at the repository root holds one sha256 per tracked asset. A
loader reads only assets the lock lists and records the sha256 of the bytes it
read in the RunPlan; an inline `{path, sha256}` or `{ref, sha256}` is refused
([path-only asset references](../../docs/reference/configuration-and-evidence.md#path-only-asset-references)).

Hashes that remain inline are human-review pins of reviewed model-visible text
(`card_sha256` in profiles and wording reviews, task-contract layer-text
hashes), generated compiled card manifests and external identities (descriptor
external sources, task-contract source dependencies, model checkpoints,
simulator revisions, vendored-payload bindings). Frozen assets keep their old
inline pins: they compose only from their launch commit, and this tree refuses
them before loading.

To edit an asset:

1. Change the file, or add it and `git add` it.
2. Refer to it by path from the assets that use it. Their bytes do not change
   when it does.
3. Run `./scripts/ora lock` and commit `policy.lock` in the same
   commit. CI runs `lock --check`.
4. Run `scripts/compare-model-visible <base> HEAD` to see what the model now
   receives.

`./scripts/ora lock --strip-pins` rewrites any composable asset that
still holds inline pins, for example after a merge from an older branch,
regenerates the compiled card manifest of any descriptor it rewrote, and then
regenerates the lock. `--strip-pins --check` only reports. A shared file is
named by every surface that uses it, so changing it changes all of them. Add a
layer when only one surface should change.

Shared Runtime code may read assets only through the composition layer under
`ora.agent.composition`. Experiment-specific prompts remain under
their experiment package and never become shared defaults by being copied here.

## Current omitted-field baseline

New `native_api` RunSpecs for `libero_pro`, `robotwin` or `robodojo` that omit
`policy.tool_surface` select
`native_full_explicit_motion_direct_v4` (integration 0059 unit A0, 2026-10-04).
The same selection can be made explicitly with
`policy.baseline: native_full_explicit_motion_direct_v4`. Its owner is
[`baselines/native-full-explicit-motion-direct-v4.json`](baselines/native-full-explicit-motion-direct-v4.json).
It differs from `native_full_explicit_motion_direct_v3` (frozen 2026-10-04,
launch commit `c0a8d96f`) in `context_composition: native_slots_v4`, Context
management v6 (its absolute 150,000-token L5 watermark, an in-place revision of
2026-10-04), the v2
ToolSurfaces `libero_placement_explicit_motion_direct_v2` (on LIBERO since
integration 0084 its overhead successor, below),
`robotwin_toolchain_b2_closed_noservo_v2` and `robodojo_placement_closed_skills_v2`,
the `prompt_bundles/current_v4/` bundles (Native core v7), and a RoboDojo
instance for Native consumers (`robodojo_arx_x5`: the ARX X5 runtime template,
`robodojo_public_explicit_v6`; the task, layout, device, GPU and qualification
stay the RunSpec's). Each v2 surface is its v1 base manifest plus the 0059 units
and the published-limits layers last (see below). direct_v3 differed from
`native_full_explicit_motion_direct_v2` (frozen 2026-10-02) only
in Native Context management v5 with `tail_aging_stride_action_steps: 4`;
`native_full_explicit_motion_direct_v1` (frozen 2026-09-30) and
`native_full_explicit_motion_v3` (below) precede it. direct_v4's
instances select:

- `libero_placement_direct_overhead_v1` (direct driver, 17 tools with
  `wait_ticks` and the M2 reach features: `libero_placement_explicit_motion_direct_v2`
  plus `overhead_footprint_localization_v1`, whose `vision_ground` /
  `vision_segment` localizations add the footprint centre, view tilt and an
  overhead wrist pose; the default since integration 0084 after
  EXP12B/placement-centre/g001, in the same `current_v4` bundles; direct_v2
  stays selectable by an explicit `tool_surface`),
  `robotwin_toolchain_b2_closed_noservo_v2` (B2 closed skill `grasp_execute`
  with probe IK counts, `align_plan` and the shared `wait_ticks`, 19 tools) under `demo_randomized`
  by default with `demo_clean` selectable, and `robodojo_placement_closed_skills_v2`
  (19 tools), plus derived `artifact.read`.
  `place_plan`/`place_execute`/`place_resume`/`place_verify`/`grasp_verify_retention`
  are retired by undeclaration (user decisions 2026-10-04); the frozen
  predecessors keep their inventories;
- `motion_action_v3`, causal `native_slots_v4` (static facts once per round),
  automatic current observation, and `action_step_lossless_v4` Context with
  exact provider continuation;
- shared planner/core sources and environment-bound manuals from
  `prompt_bundles/current_v4/`, including compiled geometry cards;
- all current public cameras and closed measured-state/calibration facts;
- same-credential Native Provider recovery within 180 seconds and remaining
  round wall time, including active requests and waits;
- Memory disabled, all physical current RGB enabled, Observer disabled;
- complete resident segmentation/grasp prerequisites, unbound until an
  operator supplies qualified service identities.

Fresh managed robot configurations inherit the same selected surface; an
unregistered baseline id is refused, and explicit ToolSurfaces retain their
selection.

The Context catalog keeps the open visual step plus two completed steps from
the current Episode. Its 24-image value is a soft L4 pressure trigger. The L5 watermark is
an absolute 150,000 tokens (Context management v6, user decision 2026-10-04;
direct_v3 and earlier used the larger of 150,000 and 0.60 of usable input),
gated by exact Provider usage. These are product defaults, not revised EXP05C evidence.
Provider/model, endpoint, image capability, context window, task, seed, budgets,
resource assignment and live qualifications remain explicit. Selecting the
baseline is not permission to launch. Use `ora baselines --json` to
inspect its catalog and `plan` or `context` for an exact resolved task.

Fresh managed robot specifications use the matching public-data surface and
compiled-card project context. Their products still own provider transport.
The frozen catalogs (`native_full_v1`, explicit-motion v1–v3
and direct v1–v2) keep their files, but each composes only from its launch commit
([maintained and frozen baselines](../../docs/reference/configuration-and-evidence.md#maintained-and-frozen-baselines)).
Baseline-defining overrides are rejected; custom treatments
select an explicit ToolSurface and context composition instead. Outside these
robot defaults, the fallback validation values remain:

- tool surface: `robot_runtime_v2`;
- action codec: selected from the tool surface (`target_state_v4` for
  `robot_runtime_v2`, not the full robot baseline);
- Memory: `disabled`;
- managed context projection: `current_observation_v1`;
- Native context profile outside the full robot baseline: `native_minimal_v1`;
- provider/model, task, endpoint, context window and experiment treatment:
  always explicit.

Behavioral studies should explicitly name their baseline or every
treatment-defining field. Historical profiles and their hashes are unchanged.
The full-baseline environment profiles are selected by exact ID, not by changing
the legacy implicit environment-profile preference.

## Tool authoring gate

The experimental `libero_placement_support_v1` profile explicitly selects the
G holding admission plus read-only public `place.plan` and `place.verify`.
Its immutable `placement_support_v1/` assets reuse the original component files,
add two closed contracts and append tool-local selection reminders. The
semantic manifest is `manipulation_placement_exp08h_v1`; neither it nor the
holding predecessor changes `native_full_v1`. See the
[placement reference](../../docs/reference/public-interfaces.md#placement-and-holding-evidence).

The separate `robotwin_placement_support_v1` profile and
`manipulation_placement_robotwin_exp08h_v1` manifest adapt those read-only
placement tools to explicit left/right TCPs and sparse dual-arm proposals.
Its arm-local holding ledger uses the v2 capability; shared geometry remains
unchanged. The profile includes EXP08F's candidate-preview grounding contract
and preserves the historical RoboTwin profile/component bytes.

The separate `libero_placement_support_v2` and `robotwin_placement_support_v2`
profiles select `public_rgbd_placement_depth_edges_v1` and the matching v2
semantic manifests. Only their two placement descriptions change: originals
are preserved, flagged depth-edge pixels remain uncertain, and numerical
candidates are conditional rather than full-object fit certificates. All v1
assets and inherited non-placement definitions remain byte-identical.

Every new ToolSurface declares `policy_tool_help_v1`. Its canonical definition
must tell the model when to use each tool, its actual effect, immediate and
deferred results, non-guarantees and error recovery; every nested input field
has a description and every tool has a public result schema. The resolver
enforces this contract before composition. Historical profiles are explicit
source-frozen exemptions, not templates for new tools. See
[`docs/guides/tool-surface-authoring.md`](../../docs/guides/tool-surface-authoring.md).

`tool_surfaces/published_limits_v1/` (integration 0059, units N1 phase 1 and N2)
is not a surface of its own. `limits.json` is the one table that maps every
enforced number in policy-visible tool help to the owner the code enforces;
`tests/test_published_limits.py` reads it for the frozen v1 defaults and the
current v2 ones. Each `*-review.json` is the wording review of one v2 default
(`robotwin_toolchain_b2_closed_noservo_v2`,
`libero_placement_explicit_motion_direct_v2` -- generation `retained` since
integration 0084 --, `robodojo_placement_closed_skills_v2`) or of LIBERO's
current `libero_placement_direct_overhead_v1` (`libero-overhead-review.json`,
whose base is direct_v2): its base profile, its exact sentence
edits with their Chinese references, and the tools whose refusal sentence a
binder owns (`finish_episode`, written by the F1 binder). The `*-help.json`
and `*-schemas.json` beside it are generated by
`python -m ora.agent.composition.description_edits <review>` from the
v2 surface's own text before them (`edits_apply_to: successor_before_layers`).
Unit A0 registered each v2 manifest with these two layers last in
`description_order` and `schema_order`, because a generated layer replaces
whole descriptions and would erase any append after it; regenerate them and
`policy.lock` whenever an earlier layer changes
([reference](../../docs/reference/public-interfaces.md#published-limits-and-stated-refusals)).

`robotwin_dual_layer_v1` is the Native v3 RoboTwin profile imported from the
EXP05A tool-semantics work and adapted to ALOHA dual arms. It exposes the four
existing read tools plus `move_free_space`, `servo_delta` and `set_gripper`,
declares the RobotRuntime observation/effect contract, and selects only
`head_camera` for the automatic causal actual while preserving all four
cameras for on-demand reads. Both measured arm states are projected, but each
mutation remains a sequential single-arm proposal. Both motion tools use
RoboTwin's configured pose planner; their distinction is absolute long transit
versus bounded relative local correction, not planner-versus-OSC. See the
[ToolSurface authoring guide](../../docs/guides/tool-surface-authoring.md).

The released opt-in `grasp_plan_v1` profile is a successor to
`motion_dual_layer_v1`. It preserves the four existing mutation tools and adds
only `vision.ground` and `grasp.plan` as read-only, current-observation-bound
calls. A returned grasp candidate compiles directly into the existing
`move_free_space` and `servo_delta` inputs; it is not reachability or grasp
success evidence. `vision.ground` is deliberately a candidate-retrieval and
inspection boundary: it narrows the full-scene search space, then delivers a
numbered overview followed by one labelled raw/mask enlargement per candidate
so the Agent can make the semantic choice. These are exact-pixel inspection
views, not higher-resolution observations, calibrated probabilities or an
identity verdict from the grounding backend. Revision 7 also makes
`grasp.plan` self-describing as an operating pack: it names prerequisite and
freshness, input/ranking semantics, every execution-relevant result, the
ordinary motion-and-verification sequence and failure-specific recovery. The
description explicitly distinguishes audit references, predicted geometry and
backend scores from policy-callable persistence, commands and success.
The learned proposal is source-camera-only. Candidate compilation and rejection
happen before the policy contact sheet is rendered, so one malformed proposal
cannot abort valid alternatives. Fused agentview+wrist geometry is optional,
separately labelled human-review evidence and never a prerequisite for the
policy result.

Correctness successors are selected explicitly from `tool_surfaces/correctness_v1/`.
Motion-only, perception/retention, holding and combined Panda placement profiles
have distinct content hashes. Retention v3 is a public surface-motion estimator;
the holding gates that once admitted motion on it left this tree. Existing default/frozen
manifests retain their selection. The shared Panda descriptor names its local
adapter sources by path, so an implementation change does not touch it or the
compiled robot facts. See the [current contract](../../docs/reference/public-interfaces.md#correctness-successor-surfaces).

The separate `correctness_v2/` assets bind depth-inlier target geometry v2 and
full-reference retention v4. They retain the first candidate's v1/v3 assets for
source-bound replay and do not change the robot motion or holding probe bounds.

The explicit `placement_unified_v3/` pair composes these corrections with H's
placement-only depth-edge uncertainty. Panda uses holding v2; ALOHA uses the
new dual-arm v3 capability with independent v2 ledgers. Original prompt-layer
text and historical assets remain unchanged. The selected v3 prompt bundles
and semantic manifests bind the combined surfaces, not a transferred live pass.

The explicit `placement_unified_v4/` pair preserves the v3 physical criteria
and adds the reviewed tool-contract repairs: combined closed grounding result,
accurate Native-visible Returns sections and opt-in public recovery errors.
It has separate v4 capability manifests and prompt-bundle selectors; v3 bytes
remain source-bound. Each 20-tool robot profile becomes 21 Native tools after
the existing lossless Artifact wrapper adds artifact.read and historical
observe(ref). The current experiment qualifies that complete advertisement.

## Finish contract v2 capabilities

`policy_finish_outcomes_v2` and `policy_abort_infrastructure_round_v2`
(user decision 2026-10-04, integration 0059 units F1-F4;
`agent/composition/finish_outcomes_schema.py`) are reserved for the v2 surface
generations of RoboTwin, LIBERO and RoboDojo, which unit A0 registered; no
other manifest declares them, and no component file changed.
They rewrite `finish_episode` to `success` and `give_up` (reason required,
give_up `category` from `unreachable`, `object_lost`, `scene_damaged`,
`repeated_failure`, `other`; `failure` and `attempts` retired), add `abort`
admitted only on this round's recorded infrastructure failure, and rewrite the
released `recover_pose` clauses that named the failure exit. They replace
`policy_failure_exit_withheld_v1` and
`policy_abort_requires_infrastructure_evidence_v1` (both since retired from
this tree with the surfaces that declared them). Every word of the new policy-visible text is generated by that
module, with the numbers taken from the constants the runtime enforces; the
contract and records are in the
[interface reference](../../docs/reference/public-interfaces.md).


## Plan A v7 candidate

The two `placement_unified_v7` surfaces retain the complete H tool inventory.
They declared bounded `servo`; this tree withholds it, so they resolve without
the tool. Six shared prompt bundles cover LIBERO/Panda and
RoboTwin/ALOHA for Native, Codex and Claude Code. The candidate
`baselines/native-full-plan-a-v7.json` binds them with motion-profile-v2 and
control-profile-v10. Its provider, host and task remain caller-owned.

The released catalog selector is unchanged. Frozen profiles are not
requalified against changed source bytes.

## Generic manipulation retention revision 2

The `libero_manipulation_v1` and `robotwin_manipulation_v1` aliases select
revision-2 manifests and versioned help/input/result components. Their
`grasp.verify_retention` requires a post-close baseline and continuous
closed-command evidence, and compensates measured TCP rotation. Revision-1
assets remain immutable. H placement v3/v4, advisory and recovery v5 retain
their existing selections. See the
[public contract](../../docs/reference/public-interfaces.md) for thresholds,
unknown cases and visible-centroid limitations.


The Dojo B2 diagnostic now selects the audited v4 closed surface, ARX public
geometry v3 card/descriptor and Dojo public-explicit v3 environment profile.
V1 source assets remain available for their exact-source history. The shared
Dojo task registry binds semantic completion layers and source/layout evidence;
builders must compose its layer after binding the actual runtime.
Panda's source-binding-only descriptor v4 retains the v3 rendered card and
geometry; current explicit-motion/B2 consumers select its v5/v6 environment
bindings. It does not change Panda motion or aperture semantics.

Core v5 is retained as the explicit Dojo v4 dependency; unrelated prompt bundles keep their existing selection.

`robodojo_placement_closed_skills_v1` and
`prompt_bundles/closed_skills_v1/robodojo-x5-native_api-v1.yaml` are the
registered RoboDojo successor on the RoboTwin closed-skills line. EXP11A g003
selected surface revision 3; EXP11A g004 selected revision 6, which integrates
the four g004 units; revision 8 adds the motion stall
help (`robodojo-motion-stall-help-v1.json` on `move_pose`: every Dojo arm
motion stops on the joint stall rule of
`environment/backends/robodojo/motion_stall.py`, holds its measured joints and
reports `controller_stalled`) and `robodojo-recovery-help-v2.json`; the developing
`robodojo_toolchain_b2_closed_noservo_v4` (revision 10) appends the same
`move_pose` help. Revision 9 (EXP11A g008) declares
`grasp_plan_jaw_symmetry_headroom_v1`, so `grasp_plan` publishes each candidate
in the member of its parallel-jaw class with the larger X5 joint6 margin and
reports `ik_reachability.jaw_symmetry` and `joint_headroom`
(`robodojo-grasp-jaw-symmetry-help-v1.json`), and replaces the jaw-hold help
with `robodojo-grasp-hold-help-v2.json`, which names a close that never left the
open position `unlikely`. Revision 10 (EXP11A g009) declares
`move_pose_jaw_symmetry_headroom_v1` and `move_pose_reach_along_request_v1`, so
`move_pose` checks each compiled target with the read-only reach check, submits
an empty-jaw `free_space` absolute target in its parallel-jaw turn when the
requested turn does not solve or has under pi/4 rad of joint6 margin and the
turn has more, and reports `jaw_symmetry` and `reach_along_request` under
`compiled_pose.targets.<arm>` (`robodojo-move-pose-reach-help-v1.json`). The B2 v4 surface lacks the pooled IK annotation
the member choice rides on and does not declare it. Revision 11 (EXP11A g009
perception review fixes) declares
`grounding_candidate_depth_profile_v1`, so every `vision_ground` and
`vision_segment` candidate card (1024x800) adds metric elevation and top views
of the candidate's own depth points under the unchanged crops
(`agent/tools/grounding_preview.py`), and `observed_extent_scene_review_v1`,
so the `grasp_plan` global review's top-left cell shows the unclipped observed
extent of the fused review cloud (top x-y view shaded by height above a front
x-z view, one scale, the action workspace outlined) instead of repeating the
contact sheet (`agent/tools/scene_extent_review.py`); no help text changed.
Revision 12 (EXP11A g011) declares
`policy_failure_exit_withheld_v1`, so `finish_episode` has no `failure`
outcome -- only `success` after `check_task` true and `abort` for an
infrastructure condition -- and a round otherwise ends at the benchmark's own
ending or an experiment per-round budget (a capability since retired from this
tree). Revision 13 (EXP11A g012) declares
`pose_primitive_reach_along_request_v1`, so every `move_to`,
`rotate_wrist` and `rotate_pitch` target carries the same `reach_along_request`
as `move_pose` (`robodojo-pose-primitive-reach-help-v1.json`). Revision 14
(EXP11A g013 tool fixes) declares
`policy_abort_requires_infrastructure_evidence_v1`, so `abort` is admitted only
when the run's own event log holds an infrastructure-class failure recorded in
this round (`agent/composition/abort_evidence_schema.py`,
`agent/tools/abort_evidence.py`). Revision 15 (EXP11A g013 tool fixes) declares
`reach_nearest_orientation_v1`, so an unsolved target's `reach_along_request`
also carries `nearest_reachable_orientation`, the smallest reachable turn of
the requested orientation with the nominal jaw centre held
(`agent/tools/reach_orientation.py`,
`robodojo-reach-nearest-orientation-help-v1.json`). Revision 16 (EXP11A g013
observed clearance) declares `observed_clearance_reference_v1`, so every
`move_pose`, `move_to`, `rotate_wrist` and `rotate_pitch` target carries a
read-only `observed_clearance` -- the jaw footprint's lowest point at the
target minus the highest surface the current public depth images show beneath
it or within 2 cm, with where that surface was observed, or `unknown` with a
reason (`agent/tools/observed_clearance.py`,
`agent/tools/observed_clearance_runtime.py`,
`robodojo-observed-clearance-help-v1.json`). Revision 17 (EXP11A g013
orientation) declares `pose_orientation_from_world_axes_v1`, so
`move_pose` takes `orientation_mode: world_axes` with world `approach_world`
and `jaw_line_world` directions compiled into the absolute quaternion, and every
`move_pose`, `move_to`, `rotate_wrist` and `rotate_pitch` target reports
`target_axes_world` from the quaternion it submits
(`agent/composition/world_axes_schema.py`, `agent/tools/world_axes.py`,
`robodojo-move-pose-world-axes-help-v1.json`). The three `move_pose` help
sentences of revisions 15-17 are appended in that order after every other Dojo
layer, and the result-field binders run in a fixed order (pose-primitive reach,
nearest orientation, observed clearance, then world axes after the
`finish_episode` binders), each adding its own field and refusing to overwrite
one. Revision 18 (EXP11A g013 kernel) declares no new
capability and carries the kernel unit's policy-visible text -- Dojo wording
help v2 (the other-arm screen in `align_plan`, `move_to`, `move_pose` and
`recover_pose`; the two-finger outcome fields of `set_gripper`), recovery help
v3 (`recovery_inter_arm_collision_path`), grasp-hold help v3, observe results
v4 and ARX descriptor v5 in `view_config`. Revision 19 (EXP11B,
2026-10-02) declares `optional_argument_null_as_omitted_v1`
(`agent/composition/null_optional_arguments.py`, bound after every other
binder and again on the final delivered tool list), so every optional member
at every depth accepts null meaning omitted, root unions are delivered flat and
both gateways drop those nulls before dispatch; the B2 sibling (revision 19),
the LIBERO default (revision 33) and the developing B2 Dojo surface (revision
11) declare it too, and no component file changed. The current revision is 20
(EXP11B, 2026-10-03): it binds the B2 sibling's two description-consistency
layers in the B2 prefix and `robodojo-description-consistency-{help,schemas}-v1.json`
after `robodojo_topdown_help` and `robodojo_wording_schemas`, correcting help
that contradicted its handlers (`grasp_plan` reachability against its own IK
selection, `geometry_query` missing depth, `mark_point` disagreement,
`recover_pose`'s withheld failure exit, defaults stated as values, one owner
for the compute rule); the B2 sibling (revision 20), the LIBERO default
(revision 34) and the developing B2 Dojo surface (revision 12) bind their own
([reference](../../docs/reference/public-interfaces.md#tool-description-consistency)).
The same four revisions declare `compute_grammar_from_evaluator_v1`, which
appends the generated evaluator grammar section to `compute`, and apply the
null-as-omitted rule at every depth of the final delivered tool list.
No capability or result schema changed beyond these. It is held to the B2 closed arm
RoboTwin ran (`robotwin_toolchain_b2_closed_noservo_v1`, including
`target_principal_frame_v1`), with the same B2 component files, then Dojo
backend layers in `tool_surfaces/robodojo_closed_skills_v1/`: the Dojo wording
and schema layers; observe result v3;
`robodojo_retract_then_unwind_recovery_v1` (the RunSpec derives
`runtime.recovery_route`) and `gripper_stall_hold_evidence_v1` with their help
layers; the X5 top-down grasp family and pooled `grasp_plan` IK selection; and,
shared with the RoboTwin closed-skills surface, the `wait_ticks` tool and
`target_support_height_v1`. Revision 7 (user decision 2026-09-29) retires
`grasp_verify_retention`, `place_plan`, `place_execute` and `place_resume` by
undeclaration: they are in `exclude_tools`, their handlers and shared layers
stay, and the loader skips a layer entry for an excluded tool. It declares 19
tools: the B2 closed arm's 22 less those four, plus `wait_ticks`; `reach_check`
is not declared. `robotwin_placement_closed_skills_v1` revision 20 and
`libero_placement_closed_skills_v1` revision 17 retire the placement trio the
same way (20 and 18 tools) and declare `target_support_height_v1`
(`tool_surfaces/target_support_height_v1/`, with a LIBERO-camera
`libero-results.json`), so the support height `place_plan` measured stays on
vision localization. The three surfaces no longer declare the capabilities only
the retired tools read (`flat_placement_arguments_v1`, the placement support,
dual-arm support and depth-edge capabilities,
`placement_refusal_measurements_v1`,
`public_rgbd_grasp_retention_verification_v4`, `held_object_offset_v1`). The surface and
bundle bind the semantic-capability manifest
`capabilities/manipulation_closed_skills_robodojo_v1.json` (instances
`robodojo_x5_closed` and the RoboTwin closed-skills `robotwin_aloha_closed`,
shared planner/core layers). `developing_surfaces.robodojo` and the B2 Dojo
package keep v4. Revision 2 declares `robodojo_staged_contact_approach_v1`, from
which the RunSpec derives the Dojo staged contact route.

The bundle binds the non-default environment profile
`robodojo_public_explicit_v6`: environment layer
`environments/robodojo/boundary-v2.md` and the ARX v5 card
(`embodiments/arx-x5/manual-public-geometry-v5.md`, descriptor
`descriptor-public-geometry-v5.json`; until EXP11A g013 profile v5 and card
v4, which differ only in the encoder sentence: v5 states the two-finger mean). Boundary v1 claimed an episode ends only
at official success or the native-action limit; v2 adds the permanent official
failure ending without its thresholds and applies the arm-return requirement
only where the task's completion contract states it. Card v4 differs from v3
only in its projection (one controlled point, the pad interval along link6
local Z, the jaw contact centre +0.1501 m along the approach axis). Boundary v1,
card v3 and profile v3 stay unchanged for g003, the B2 bundles and recorded
runs. The RoboDojo `imitate_sorting_sequence` contract moves to
`imitate_sorting_sequence-completion-v2.md` (review record v2): the same clauses
plus a tool-neutral note that the scripted demonstration advances only while
native actions are consumed.

## Closed-skills ports for the next RoboTwin and LIBERO generations

`tool_surfaces/closed_skills_ports_v1/` holds the policy-visible text of the
integration-0059 ports of RoboDojo closed-skills features to the RoboTwin and
LIBERO defaults. Unit A0 declared the capabilities and appended the components
on the v2 defaults, following `tests/closed_skills_port_candidates.py` `UNITS`
(on RoboTwin the M1 slot binds the M6 screened variant,
`robotwin-move-pose-reach-reference-inter-arm-endpoint-help-v1.json`).

| Component | Surface | Capability it explains |
|---|---|---|
| `robotwin-move-pose-reach-reference-help-v1.json` | RoboTwin | `move_pose_reach_along_request_v2`, `pose_primitive_reach_along_request_v2` (approximate DLS pre-check, about 5% false refusals and 4% false admissions against `move_pose`'s CuRobo IK) |
| `robotwin-reach-nearest-orientation-help-v1.json` | RoboTwin | `reach_nearest_orientation_v2` (grip-centre TCP held) |
| `observed-clearance-grip-centre-help-v1.json` | RoboTwin, LIBERO | `observed_clearance_reference_v2` (holding = close plus measured jaw separation above 3 mm) |
| `robotwin-move-pose-world-axes-help-v1.json`, `libero-move-pose-world-axes-help-v1.json` | RoboTwin, LIBERO | released `pose_orientation_from_world_axes_v1` on each embodiment's axes |
| `libero-grounding-depth-profile-help-v1.json` | LIBERO | released `grounding_candidate_depth_profile_v1`: the `vision_ground` card size 1024x800 (a `replace` of the released text) |
| `libero-move-pose-reach-reference-help-v1.json` | LIBERO | `move_pose_reach_along_request_v2`, `pose_primitive_reach_along_request_v2` on M2's CuRobo IK (`libero_curobo_ik_robot_only_v1`; joint limits and self-collision only; disagreement with `move_pose` not yet measured) |
| `libero-reach-nearest-orientation-help-v1.json` | LIBERO | `reach_nearest_orientation_v2` (grip site held) |
| `libero-grasp-plan-ik-selection-help-v1.json` | LIBERO | `grasp_plan_ik_reachability_annotation_v1` with `libero_pro_rgbd_v1` (a `replace` of the released `grasp_plan` text: one sentence repointed, the IK-selection paragraph appended) |

`observed_extent_scene_review_v1` needs no text. Generated layers are
regenerated, never hand-edited, when the text they replace moves; the port
tests compare them. Chinese reference renderings are in
[the reference](../../docs/reference/public-interfaces.md#closed-skills-ports-to-the-next-robotwin-and-libero-generations-integration-0059).

## Compute usage and routing defaults

The eight maintained Panda/ALOHA Compute-bearing surfaces resolve current routing manifests,
sharing Compute v2 input/output selection, numeric helpers and executable
examples. Earlier manifests remain available. The current core and prompt
bundles retain the target's selection. See the
[usage guide](../../docs/guides/compute-usage.md).

## Local CPU ORA composition

The default local ORA entry selects `local_atomic_cpu_v1` and the
`local_cpu_v1/libero-panda-native-api.yaml` bundle explicitly through
`agent/composition/ora_local.py`. It reuses the atomic v3 OSC action compiler
with nine tools and no GPU planner, VLA or resident perception service.
`control-local-cpu-v1.json`, `manual-local-cpu-v1.md` and the
`libero_pro_local_cpu_v1` context profile bind the reviewed Panda/controller
facts; the legacy atomic manifests and the shared server baseline stay byte
stable. Context settings come from the current maintained baseline; local
provider, source, host and task inputs are checked by normal resolution and
preflight. See the [current local contract](../../docs/reference/public-interfaces.md#ora-local-default)
and [source compatibility audit](../../docs/design/ora-local-cpu-compatibility.md).
