# Policy assets

This directory is an immutable asset library, not a second Runtime and not a
list of competing defaults. Version suffixes identify the exact content that a
source-bound RunPlan used; they are deliberately retained for reproducibility.

## How one asset becomes active

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
| `prompt_bundles/` | Ordered prompt layers per backend and consumer | baseline instances and `ora local` |
| `planners/native/` | Native system/project policy assets | Native context composition |
| `embodiments/` | Robot descriptors and manuals | embodiment/manual binding |
| `environments/` | Policy-safe environment boundary/context assets | environment context profile |
| `tool_surfaces/` | Immutable tool manifests, schemas and descriptions | `policy.tool_surface.profile` |
| `baselines/` | Provider-independent product default composition | Native robot RunSpec default expansion |

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

## Default baseline

New `native_api` RunSpecs for `libero_pro`, `robotwin` or `robodojo` that omit
`policy.tool_surface` select `native_full_explicit_motion_direct_v4`, the only
registered baseline. The same selection can be made explicitly with
`policy.baseline: native_full_explicit_motion_direct_v4`. Its owner is
[`baselines/native-full-explicit-motion-direct-v4.json`](baselines/native-full-explicit-motion-direct-v4.json).
Its instances select:

- `libero_placement_direct_overhead_v1` on LIBERO/LIBERO-Pro (direct driver,
  17 tools, with `overhead_footprint_localization_v1`: `vision_ground` and
  `vision_segment` localizations add the footprint centre, view tilt and an
  overhead wrist pose);
- `robotwin_toolchain_b2_closed_noservo_v2` on RoboTwin (closed skill
  `grasp_execute` with probe IK counts, `align_plan` and the shared
  `wait_ticks`, 19 tools), under `demo_randomized` by default with
  `demo_clean` selectable;
- `robodojo_placement_closed_skills_v2` on RoboDojo (ARX X5, 19 tools);
- plus the derived `artifact.read` tool on every backend.

All three use `motion_action_v3`, causal `native_slots_v4` context (static
facts once per round), automatic current observation, `action_step_lossless_v4`
retention with exact provider continuation, the `prompt_bundles/current_v4/`
bundles (shared planner/core sources, environment-bound manuals and compiled
geometry cards), all current public cameras with measured-state and calibration
facts, same-credential provider recovery, Memory disabled, and the resident
segmentation and grasp prerequisites, unbound until an operator supplies
qualified service identities.

The Context catalog keeps the open visual step plus two completed steps of the
current episode. Its 24-image value is a soft L4 pressure trigger; the L5
watermark is an absolute 150,000 tokens, gated by exact provider usage.
Provider/model, endpoint, image capability, context window, task, seed,
budgets, resource assignment and live qualifications remain explicit.
Selecting the baseline is not permission to launch. Use `ora baselines --json`
to inspect its catalog and `plan` or `context` for an exact resolved task.

Baseline-defining overrides are rejected; custom treatments select an explicit
ToolSurface and context composition instead. Outside these robot defaults, the
fallback validation values are:

- tool surface: `robot_runtime_v2`;
- action codec: selected from the tool surface (`target_state_v4` for
  `robot_runtime_v2`);
- Memory: `disabled`;
- Native context profile outside the robot baseline: `native_minimal_v1`;
- provider/model, task, endpoint, context window and experiment treatment:
  always explicit.

## Registered surfaces

`agent/composition/profile_catalog.py` registers every ToolSurface this tree
resolves:

| Profile | Role |
|---|---|
| `libero_placement_direct_overhead_v1`, `robotwin_toolchain_b2_closed_noservo_v2`, `robodojo_placement_closed_skills_v2` | the default baseline's instances |
| `robotwin_toolchain_b2_closed_noservo_v1` | the earlier RoboTwin closed-skill surface, which keeps `place_plan`, `place_execute`, `place_resume` and `grasp_verify_retention`; no baseline selects it |
| `robot_runtime_v1`, `robot_runtime_v2` | the runtime_v1 qualification ladder and the RunSpec fallback |
| `native_atomic_causal_v1` | the Synthetic Native run |
| `local_atomic_cpu_v1` | `ora local` |

The surface directories under `tool_surfaces/` are layered: the default
manifests pull components from `explicit_motion_v1`..`v3`, `public_data_v3`,
`compute_v2`, `delivery_trim_v2`, `closed_skills_ports_v1`,
`published_limits_v1` and the other component directories, so a `v1` name
is not by itself a retired asset.

## Tool help contract and published limits

Every new ToolSurface declares `policy_tool_help_v1`. Its canonical definition
must tell the model when to use each tool, its actual effect, immediate and
deferred results, non-guarantees and error recovery; every nested input field
has a description and every tool has a public result schema. The resolver
enforces this contract before composition. See
[`docs/guides/tool-surface-authoring.md`](../../docs/guides/tool-surface-authoring.md).

`tool_surfaces/published_limits_v1/` is not a surface of its own.
`limits.json` is the one table that maps every enforced number in
policy-visible tool help to the owner the code enforces;
`tests/test_published_limits.py` reads it. Each `*-review.json` is the
wording review of one default surface: its base profile, its exact sentence
edits with their Chinese reference renderings, and the tools whose refusal
sentence a binder owns (`finish_episode`). The `*-help.json` and
`*-schemas.json` beside it are generated by
`python -m ora.agent.composition.description_edits <review>` from the
surface's own text before them (`edits_apply_to: successor_before_layers`).
Each default manifest lists these two layers last in `description_order` and
`schema_order`, because a generated layer replaces whole descriptions and
would erase any append after it; regenerate them and `policy.lock` whenever
an earlier layer changes
([reference](../../docs/reference/public-interfaces.md#published-limits-and-stated-refusals)).

## Finish contract v2 capabilities

`policy_finish_outcomes_v2` and `policy_abort_infrastructure_round_v2`
(`agent/composition/finish_outcomes_schema.py`) are declared by the v2 default
surfaces of RoboTwin, LIBERO and RoboDojo. They rewrite `finish_episode` to
`success` and `give_up` (reason required, give_up `category` from
`unreachable`, `object_lost`, `scene_damaged`, `repeated_failure`,
`other`), add `abort` admitted only on this round's recorded infrastructure
failure, and rewrite the `recover_pose` clauses that named the failure exit.
Every word of the new policy-visible text is generated by that module, with the
numbers taken from the constants the runtime enforces; the contract and
records are in the
[interface reference](../../docs/reference/public-interfaces.md).

## Closed-skills ports on the RoboTwin and LIBERO defaults

`tool_surfaces/closed_skills_ports_v1/` holds the policy-visible text of the
RoboDojo closed-skills features ported to the RoboTwin and LIBERO defaults.
The v2 defaults declare the capabilities and append the components, following
`tests/closed_skills_port_candidates.py` `UNITS` (on RoboTwin the reach slot
binds the other-arm screened variant,
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
tests compare them.

## Compute usage and routing

The default surfaces share Compute v2 input/output selection, numeric helpers
and executable examples; shared calculation routing lives in
`tool_routing_v1/shared-help.json`, and backend overlays own the planner
handoff examples. See the [usage guide](../../docs/guides/compute-usage.md).

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
and [source compatibility audit](../../docs/reference/ora-local-cpu-compatibility.md).
