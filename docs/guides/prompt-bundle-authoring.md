# Prompt Bundle authoring guide

ORA authors a robot policy as one versioned bundle of independently
owned layers. The purpose is not to make the prompt longer. It is to ensure
that every fact has one owner, every first-action fact is actually visible, and
adding a robot cannot silently mix task strategy, camera folklore, tool syntax,
or private evaluator details into a convenient system prompt.

The machine contract is `ora.prompt-bundle-authoring.v1`. It is an
offline authoring and preflight surface; creating or auditing a bundle does not
select it in a RunPlan, start a simulator, or modify a historical experiment.

The product `native_full_v1` catalog explicitly registers the two
`prompt_bundles/native_full/` bundles as inputs to Native robot default
expansion. Merely adding another bundle does not enroll it. The resolver binds
the chosen bundle and each expanded source by hash; archived and example
bundles retain their existing selection boundary.

## The bundle and its delivery

Every bundle contains these layers in this semantic order:

| Layer | Owns | Must not own |
|---|---|---|
| Planner | Policy-runtime role, turn protocol, runtime-wide unavailable facilities | Task, benchmark, robot, camera, frame, or individual tool names |
| Core | Evidence/inference/candidate/executed/verified distinctions, freshness, replanning and claim boundaries | Provider features, task tactics, robot facts, tool syntax |
| Environment | Task authority, public/private state, reset/termination and official-verdict owner | Robot topology, camera mounting, frames, tool syntax, evaluator implementation |
| Embodiment | Body topology, controlled points, frames, axes, stable motion/gripper semantics, sensor mounting and physical non-guarantees | Task objective, evaluator, individual tool names or experiment identity |
| Episode task | Exact instruction and optional public task contract | Generic robot manual, hidden predicate, canonical coordinates, expert trajectory |
| Tool manual delivery | Whether a separate manual is eager, on demand, or manifest-only | A second drifting copy of tool syntax |
| ToolSurface | Exact tool purpose, selection boundary, inputs, effect, returns, non-guarantees and recovery | General policy philosophy or benchmark solution |
| Dynamic Runtime | Current observation, calibration, measured state, receipt, budget and recoverability | Stable facts that should have been present before the first action |

ORA owns the Native request roles and records the exact canonical and provider-wire requests.

| Semantic layer | Native delivery |
|---|---|
| Planner | Protected `harness_system` / provider system input. |
| Core/environment/embodiment | Protected project instructions. |
| Episode task | Protected episode-task input. |
| Tools | Provider tools field on each call. |
| Measurements/history | Causal current state and retained interaction history. |
| Manifest-only manual | Bound evidence, not automatically provider-visible. |

### Physical prefix template for new bundles

The semantic table above is not permission to concatenate all Markdown into
one large system prompt. New compositions must map the bundle into this
stable-to-dynamic physical template:

| Stability tier | Content | Native placement | Change rule |
|---|---|---|---|
| Global/session-stable | Planner identity and only the Runtime facts that cannot change inside the session | Protected `harness_system`; Native v2 may inline separately hashed session `runtime_facts` | Version the source; do not patch it mid-session |
| Project-stable | Core, environment and eager embodiment facts needed before the first action | Protected user `project_instructions` | Keep exact ordered source/hash inventory |
| Episode-stable | Exact task and optional public completion contract | Protected first user `episode_task` | Render `{{TASK}}` once; later turns use task identity, not repeated task prose |
| Optional protected state | An admitted rollout checkpoint only at a context epoch | Protected current user slot | Never imply hidden or automatic Memory |
| Growing history | Assistant decisions, tool calls/results, receipts and prior observations retained by policy | Runtime projection after protected sources | Retention/compaction is a separate treatment axis |
| Current state | Fresh observation, budget, lifecycle and recoverability notices | Final causal observation or `dynamic_runtime` user tail | Append/update only here; never rewrite stable sources |
| Tool catalog | Canonical names, descriptions, input and public result schemas | Separate provider `tools` field | Deterministic order and independent hash; schema/help changes create a new ToolSurface |

For Native v3, the concrete protected sequence is
`harness_system → project_instructions → episode_task`.
An admitted rollout checkpoint may follow those sources at an explicit
condensation epoch. Projected ActionStep history and the current causal
observation follow; tools remain outside the message list. The provider adapter
may serialize that canonical request differently—for example Anthropic uses a
top-level `system` field—but the exact provider wire is evidence, not an
authoring guess.

The current Native catalog is intentionally eager: all tools on the selected
surface are projected in the provider `tools` field on every call. There is no
progressive tool disclosure today. A future dynamic catalog would be a separate
policy treatment and would need to preserve discovery, schema hashes, tool-call
continuity and comparable evidence.

## What moves where in the current RoboTwin wording

The standard changes ownership, not necessarily the underlying fact. Concrete
examples from the current engineering prompt are:

| Existing wording or topic | Standard owner | Standard form |
|---|---|---|
| “Use only the fourteen advertised tools” | Planner | “Control one episode through the advertised tools”; the exact fourteen names come only from ToolSurface delivery. |
| “`observer_camera` is widest; head/front are partial; wrist views move” | Embodiment | Put mounting, stable qualitative role and calibration lifetime under `Sensors and cameras`. |
| “If observer is absent, move an unused open arm high/far” | Embodiment capability + tool-local recovery | Embodiment says a wrist camera may be physically repositioned for broader coverage and that this is not optical zoom. The relevant motion/observation help explains when that action is useful. It is not a universal system-policy tactic. |
| “Lowering world z cannot rotate the gripper” | ToolSurface + embodiment frames | Embodiment defines world z and the end-effector axes; the position-only tool says that it preserves orientation and gives this near-miss counterexample. |
| “Use wrist roll versus pitch versus full pose” | ToolSurface | Each overlapping tool owns a positive selection test, adjacent counterexample, effect and recovery. The planner does not duplicate the catalog. |
| “A close target is not a grasp” | Embodiment + gripper tool | Embodiment separates command, measurement, contact and retention; the gripper tool states its exact immediate/deferred evidence. |
| “Only the official checker decides success” | Environment + checker tool | Environment owns verdict authority; the tool owns only the exact Boolean call/result contract. |
| “Fresh evidence contradicted the old plan” | Core | One cross-task rule: revise the relevant belief or geometry before repeating a risky action. Do not list every observed failure episode. |

This is also the boundary for behavior-specific advice. “Observe more,” “raise
the camera,” or “always inspect every view” is not appropriate global system
text. Stable sensor capabilities and occlusion limits belong to the embodiment;
a concrete uncertainty-to-view decision belongs to tool help; the current
visibility result remains dynamic evidence. Only a genuinely cross-task
invariant such as “current conflicting evidence invalidates the old plan”
belongs in the core.

## Embodiment card format

Every new bundle-format embodiment card has exactly these nine level-two
sections, in order:

1. `Identity and topology`
2. `Controlled points`
3. `Frames and units`
4. `End-effector axes`
5. `Motion capabilities`
6. `Gripper semantics`
7. `Sensors and cameras`
8. `Preservation and evidence limits`
9. `Dynamic facts and provenance`

The authoring template is
`ora/policy/prompt_bundles/templates/embodiment-v1.md`. The preflight
requires all nine headings, checks that every public camera in the selected
environment profile appears in the card, and requires at least one descriptor
or control-profile binding.

The card describes the body; it does not teach API syntax. “The end effector
can change orientation while holding position” belongs here. “Call
`rotate_pitch({targets: ...})`” belongs only in that tool's canonical schema and
description. For a two-arm embodiment, the card states sparse selection,
synchronization and omission semantics; the ToolSurface states the exact
`targets` object.

The current standard examples preserve the distinctive facts from both
embodiments without activating new defaults:

| Node | Panda example | ALOHA example |
|---|---|---|
| Topology | One fixed 7-DoF serial arm, symmetric parallel jaws | Fixed dual body, independent left/right arms and parallel jaws |
| Controlled point | RoboSuite grip site | Per-arm public planner `end_effector` |
| Axes | Grip-site axes; top-down normally maps local +Z to world -Z | +X approach, +Y jaw separation, +Z lateral |
| Motion | Position/orientation/combined target; action-start relative/hold rules | Sparse one-/two-arm planning from one snapshot; all planners succeed before synchronized execution |
| Gripper | Actuator-target preservation is not invariant measured aperture | Public open/close/unknown command target; private `[0,1]` cache is not aperture |
| Cameras | Fixed `agentview`, moving `wrist` | Optional observer, partial head/front, moving left/right wrists |
| Evidence | Tracking outcome is not collision/contact/success | Omitted command is not physical immobility; inactive drift remains measured evidence |

The concrete non-default cards are
`franka-panda/manual-v4.md` and `aloha-agilex/manual-v7.md`. Their matching
environment profiles end in `standard_bundle_v1`. A numerically newer file or
profile is never selected automatically.

## Project instructions, indexes and manuals

Project instructions are stable eager semantics needed before the first
action; they are not primarily a context-compression index. Native robot
sessions currently have no general shell or file reader, so a referenced file
is useless unless an advertised tool can retrieve it. Native first-action facts
must therefore be in the visible planner/core/environment/embodiment/task
layers or in the eager tool schemas.

A `tool_manual` source is still materialized and hash-bound for composition
evidence, but that does not make it visible. Use one of the exact delivery
claims:

- `manifest_only`: not injected and no reader tool;
- `on_demand`: the named reader exists on the selected ToolSurface;


Do not copy ToolSurface descriptions into a manual. Before retiring an old
manual, relocate every unique body fact to the embodiment, every environment
fact to the boundary, and every tool-local fact to the canonical tool
definition.

## Authoring workflow

Create a draft manifest:

```bash
./scripts/dev/audit-prompt-bundle template /tmp/my-robot-bundle.yaml \
  --bundle-id my_robot_native_v1
```

Copy the Markdown templates under
`ora/policy/prompt_bundles/templates/` into versioned shared or
experiment-owned locations, resolve every `__REQUIRED__`, and bind an exact
environment prompt profile plus ToolSurface. Then run the simulator-free
preflight:

```bash
./scripts/dev/audit-prompt-bundle audit /tmp/my-robot-bundle.yaml
./scripts/dev/audit-prompt-bundle audit \
  ora/policy/prompt_bundles/examples/libero-panda-native-v1.yaml
./scripts/dev/audit-prompt-bundle audit \
  ora/policy/prompt_bundles/examples/robotwin-aloha-native-v1.yaml
```

The audit fails on:

- unknown/missing manifest fields, unresolved sentinels or unsafe paths;
- wrong layer delivery in the Native request;
- missing or reordered embodiment nodes and missing public cameras;
- environment metadata/profile/manual mismatches;
- a new ToolSurface without `policy_tool_help_v1`;
- a manual that claims on-demand visibility without a reader;
- environment, embodiment or exact tool vocabulary in the wrong authored
  layer;
- task templates without exactly one `{{TASK}}` slot; or
- descriptor/control-profile embodiment drift.

The audit emits exact source hashes, ToolSurface composition hash, layer order
and a delivery inventory with tri-state provider visibility. It also inventories
estimated tokens against the existing per-layer review targets and detects
long exact sentences duplicated across semantic layers. Token targets are
review prompts rather than correctness gates: a multi-arm/five-camera card may
remain longer when removing text would remove an owned first-action fact.

The preflight is a necessary authoring gate, not Runtime qualification or
behavioral evidence. A bundle becomes active only when a separately reviewed
RunSpec/RunPlan selects its exact sources and hashes.

## Naming assets by path

A bundle names its layers, descriptor and control profile by path and holds no
hashes. The assets it selects follow the same rule
([path-only asset references](../reference/configuration-and-evidence.md#path-only-asset-references)).
An environment prompt profile names its compiled descriptor and wording review
by path:

```json
"compiled_embodiment": {
  "descriptor": "ora/policy/embodiments/arx-x5/descriptor-public-geometry-v5.json",
  "review": "ora/policy/embodiments/arx-x5/wording-review-public-geometry-v5.json",
  "card_sha256": "<sha256 of the reviewed card text>",
  "runtime_verification": "loaded_robot_geometry_before_policy_v1"
}
```

`card_sha256` stays: it is the human-review pin that says this exact card text
was reviewed. The wording review carries the same `card_sha256` and nothing
about the descriptor's bytes (unit L3): the card must be the descriptor's
compiled projection, so a descriptor edit that leaves the card unchanged needs
no new review. A descriptor names its repository provenance sources by path
only. A baseline catalog names
`runtime.motion_profile` and `runtime.control_profile` by path. The resolver
hashes every file it reads and records `{path, sha256}` in the RunPlan.

To change a layer, profile or card, edit or add the file, `git add` it, run
`./scripts/ora lock` and commit `policy.lock` with it. A changed card
also needs its new `card_sha256` in the profile and a new wording review; that
is a human-review step, not part of the lock. The loader refuses an inline
`{path, sha256}` binding, and it refuses a path that `policy.lock` does not
list.

## Review and documentation contract

Treat these as different experimental axes even when they change in one file:

- semantic wording or fact ownership;
- provider role/physical slot;
- ToolSurface names, descriptions, schemas, order or eager/deferred disclosure;
- observation delivery;
- history retention, Memory or condensation; and
- controller/action execution semantics.

A review must say which axis changed and freeze the others or name the joint
treatment. “Same prompt” is insufficient unless source text, semantic owner,
provider role, delivery timing and visibility all match. “Same tools” is
insufficient unless canonical order, descriptions, input/public-result schemas
and disclosure mode match.

Any future prompt/context change updates, in the same review unit:

1. the mapped current reference in `docs/reference/`;
2. `docs/context-assembly.md` when physical order, visibility, retention or
   evidence changes;
3. this guide when authoring shape, ownership or preflight changes;
4. `docs/guides/tool-surface-authoring.md` when tool-help/schema rules change;
5. the policy asset README when selection/default behavior changes; and
6. the experiment README when an experiment-owned prompt or treatment changes.

Historical prompt assets and experiment traces are never edited to conform to
the latest template. A new bundle versions the changed source, binds an exact
environment/embodiment/ToolSurface tuple, records delivery inventory and hashes,
and passes `scripts/dev/audit-prompt-bundle` before live qualification.

## Versioning and experiments

Never rewrite a layer bound to historical evidence. Add a versioned source,
new environment profile or experiment-owned layer, then bind it in a separately
named treatment. Prompt-only, ToolSurface-only, observation-delivery and
controller changes remain different experimental axes.

Use fresh resolved configurations to verify actual default selection. Historical compositions remain bound to their recorded source and do not inherit current wording or delivery changes.
