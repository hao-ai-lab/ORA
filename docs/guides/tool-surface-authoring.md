# ToolSurface authoring and model-visible help

ORA treats every policy-visible tool definition as executable model
documentation. A fixed tool catalog does not need a separate `help` tool: the
canonical description, input schema, public result schema and structured error
must together tell the policy when the tool is useful, what it changes, what it
returns and what its success cannot prove.

This guide is the required authoring contract for every new ToolSurface. The
source of truth remains the resolved ToolSurface, whose RunPlan records the
hash of every component it read; `CLAUDE.md`,
`AGENTS.md`, experiment prompts and transport adapters must not maintain a
second copy of individual tool syntax.

## Required model-visible description

The Native full product profiles use one shared description component and one
embodiment component with disjoint tool ownership. Do not append another
overriding help chain to those profiles; introduce a new version when semantics
change. They opt into `canonical_tool_order_v1`: `tool_order` must contain every
effective base/additional tool exactly once, after exclusions. Missing, duplicate
or unknown names fail resolution. Older profiles retain their original strict
base ordering and hashes.

Every tool on a surface declaring the `policy_tool_help_v1` capability uses
these exact sections, in this order:

```text
<One-sentence purpose.>
Use when: <decision condition and useful stage>.
Effect: <read-only, world mutation, lifecycle, and persisted local state>.
Returns: <immediate public fields/images and any later receipt/observation>.
Does not: <important non-guarantees and forbidden inference>.
On error: <evidence-preserving correction or recovery>.
```

The purpose and `Use when` text must distinguish the tool from adjacent tools.
Review the final resolved definition and a captured provider projection when
available, rather than an unselected base file. Review producer/consumer pairs
together: a producer must name an available downstream tool, explain when to
choose it over an atomic alternative, and show a schema-valid complete handoff.
Keep trigger examples distinct from argument examples. Do not make every tool
mandatory; copying an existing valid value and an unnecessary repeated query
are useful negative selection examples. Apply shared rules once and bind the
actual backend cameras, frames, measurements and optional capabilities.
`Effect` must describe actual semantics rather than its name: a read-only tool
may persist an observation-bound measurement record, while a mutation tool
usually returns `submitted` before the Environment has executed it. `Returns`
must separate the immediate tool result from a later Supervisor receipt and
fresh observation. `Does not` carries authority boundaries such as “not object
ground truth,” “not reachability,” and “not task success.” `On error` must give
the policy a valid next step without teaching a benchmark solution.

Do not put general ReAct advice, task strategy, simulator-private facts,
successful trajectories or provider-specific syntax in a tool description.
Those belong to the core policy, episode task, Environment/embodiment contract
or provider adapter.

## Input schema contract

The input schema is the tool's machine-readable `--help`:

- the root is an object with `additionalProperties: false`;
- every property, including nested object properties, has a non-empty
  `description` stating semantics, units, frame and omission/default behavior;
- closed vocabularies use `enum`; do not advertise a free-form string and rely
  on a Runtime error to reveal the accepted values;
- bounded numbers and collections declare `minimum`, `maximum`, `minItems` and
  `maxItems` whenever the Runtime enforces them;
- tuple order is explicit, for example `[x,y,z]`, `[x,y,z,w]` or `[u,v]`;
- required fields are listed exactly, and mutually exclusive shapes use a
  schema branch rather than a hidden mode switch;
- on a surface declaring `optional_argument_null_as_omitted_v1` every
  optional member at every depth also accepts null, meaning omitted, and the
  delivered tool is one flat object; composition and delivery add it and the
  gateways drop the null, so do not hand-edit `"null"` into a component, give
  an optional member a null meaning of its own there, or rely on a root union
  for exclusivity (state it in the description and enforce it in the handler);
- no policy parameter may expose privileged state or an unenforced physical
  guarantee.

Schema and Runtime validation must accept the same values. Tightening only the
schema or only the handler is a contract defect. Composition refuses an input
keyword the Runtime validator does not enforce (`lint_input_schema`); a generic
schema mismatch is already reported with its field, rule and bound, so a
handler need not repeat a check the schema states.

## Public result and error contract

Every advertised tool owns one public result schema. The canonical ToolRuntime
validates the text/JSON result after private image payloads and Runtime
directives are removed. This prevents accidental field drift or privileged
data leakage. Image and other multimodal blocks remain separate artifacts; the
description states when they are returned.

Provider protocols do not all expose an output-schema field, so the
model-visible `Returns` sentence is mandatory even when a Runtime result schema
exists. Managed MCP, Native Anthropic and Native OpenAI projections must all
derive their name, description and input schema from the same resolved
ToolSurface.

Errors use the shared structured ToolError boundary
(`agent/tools/tool_errors.py`): stable `code`, concise message, `retryable`,
`retry_requires_changed_arguments` when known, and only relevant `field`,
`received` or `constraints`. `retryable` means a later call of this tool can
still succeed (after a corrected request or a changed public state); it is not
a resend instruction. An argument refusal code (`ARGUMENT_REFUSAL_CODES`)
additionally states that the unchanged request is refused again; pass
`retry_requires_changed_arguments` explicitly only when a raising site knows
better than its code. Express a mutually exclusive or unknown-argument rule with
the shared `require_*` helpers, which report argument names and the violated
rule, never values. An error should enumerate a closed choice or current bound
when that helps the policy correct the call. It must not reveal object
identities, hidden predicates, exact privileged contacts or simulator
internals. The gateway, not the tool, appends the repeated-identical-failure
count; see the
[structured error reference](../reference/public-interfaces.md#structured-tool-errors-and-repeated-identical-requests).

## Effect, freshness and authority audit

For each tool, review all of the following before activation:

| Question | Required evidence |
|---|---|
| When should the policy select it? | Distinct `Use when` text and at least one selection test or fixture |
| What state can it change? | Manifest mutation/lifecycle/observation classification agrees with ToolRuntime dispatch |
| Which observation binds the call? | Current/stale behavior and artifact identity are explicit |
| What is returned immediately? | Public result schema and multimodal block contract |
| What arrives later? | Receipt/fresh-observation cadence for mutations |
| What cannot be inferred? | Explicit non-guarantees and authority owner |
| How does failure recover? | Structured error and valid retry/replan/stop path |
| Is transport parity preserved? | Resolved canonical definitions match the Native provider projection |

Proposal-producing perception tools must also document common
**successful-but-wrong** outcomes. Backend success only means that a valid
candidate or mask was produced. Text grounding is a recall-oriented candidate
set: rank and score order candidates within one call but do not certify rank
0. Point segmentation treats a click as a surface prompt rather than a
semantic label; a boundary, shadow or background click can validly return the
table, robot or neighbouring object. Help must identify the evidence to inspect
(overlay, prompt cross, box, mask area/shape and depth), the rejection rule and
a valid farther-interior/complementary-view retry. Dynamic-camera help must
also say that pixels and calibration are observation-bound: after a wrist
moves, old wrist pixels cannot be reused as current hand-eye geometry.

## Adding or changing a tool

1. Work on the owning feature/experiment branch. Do not rewrite a
   frozen ToolSurface bound to historical evidence; create a versioned
   component/profile or an explicitly source-bound successor revision.
2. Add the tool once to the canonical ToolSurface and classify its effect in
   the manifest. Update the canonical definitions used by Native.
3. Write the five-section description and fully documented closed input schema.
4. Add the public result schema and validate every successful result path,
   including unavailable/pending variants and multimodal-field removal.
5. Add structured invalid, stale, unavailable and terminal-path tests. For a
   mutation, test proposal → Supervisor execution → receipt → fresh
   observation; tool submission alone is insufficient.
6. Check Native projection for both provider protocols. Bind exact definitions
   and hashes in the RunPlan.
7. Declare `policy_tool_help_v1`. Resolution fails if any section, parameter
   description or result schema is missing. Source-frozen profiles are listed
   explicitly as legacy exemptions; a new catalog entry cannot inherit that
   status implicitly.
8. Update the owning current reference, policy README and tests index as
   appropriate. Run the focused tests and the locked full suite before merging.
9. Keep the declared set and the dispatch set joined. A tool named in a
   profile's `tool_order` must resolve to a handler in the runtime that
   `runtime_id` selects, because `allowed_tools` defaults to every declared
   tool and the loader validates a manifest only against itself.
   `tests/test_declared_tool_handler_join.py` checks this across the whole
   catalog. For `ManipulationToolRuntime` the dispatch table is `invoke`
   in `agent/tools/runtime/core.py`; the handler itself goes in the mixin
   module of its tool family (`runtime/observe.py`, `runtime/motion.py` and so
   on). The grasp family's admission checks and service calls stay in
   `runtime/grasp.py`.

### Retiring a tool

A handler may be deleted only when no registered profile still declares the
tool. The loader never consults the dispatch table, so a tool left advertised
without a handler resolves and composes normally and then fails on first call:
`ManipulationToolRuntime.invoke` subscripts a name-to-method dict, and the
`KeyError` is rewritten by the surrounding handler as a non-retryable
`internal_tool_error`. Nothing offline catches that, and the cost lands on a
live episode.

Run `tests/test_declared_tool_handler_join.py` before the deletion, not after.
It names every profile that would be left advertising the tool. If the set is
empty the handler is unreachable and may go; if it is not, the choice is to
stop, or to retire those profiles as a separate and explicitly approved
disposition. Preserving a frozen manifest's bytes keeps its evidence identity
valid, but it does not keep the profile runnable — those are different claims,
and a retirement that relies on the first should say plainly that it is giving
up the second.

Undeclaring a tool from a live surface also retires the capabilities that only
its code paths read, decided from the `supports(...)` call sites rather than the
name. The 2026-09-29 retirement of `place_plan`, `place_execute` and
`place_resume` (and, on RoboDojo, `grasp_verify_retention`) from the three live
closed-skills surfaces is the worked case. `public_rgbd_grasp_retention_recovery_v5`
reads like retention but gates `observe`/`vision_segment` `step`, and
`public_rgbd_retention_advisory_v1` gates `set_gripper`'s `target_ref`
annotation and pre-dispatch argument validation for every tool; both stay.
`held_object_offset_v1` reads like a general observation field, but only
`place_plan` sets its value, so it went with the tool. A quantity the retired
tool delivered must reach the policy through a surviving tool first: the
support height moved to vision localization (`target_support_height_v1`). The
manifest lists the tool in `exclude_tools`; shared layers need no edit because
the loader skips an excluded tool's entries.
`tests/test_closed_skill_surface_contract.py` checks tool help, prompt layers,
environment profiles and task contracts for either spelling.

A live manifest changes a tool its frozen base catalog defines by *overriding*
it, never by editing that catalog. `description_order`, `schema_order` and
`result_order` are applied in order and a later component wins for the same tool
name, so a new component plus a `revision` bump is how a maintained profile
takes a new input schema, help text or public result while the shared file it
names keeps its bytes. `explicit_motion_v3` is the worked case: its
`base_definitions` names the `explicit_motion_v2` catalog by path, and its
per-point `vision.segment`, batched `mark_point` and flat `place.*` contracts
all arrive this way. A new profile generation is for a changed tool *set* or a
changed identity, not for a changed schema.

`resolve_tool_surface` runs its steps in a fixed order (definition catalogs;
canonical names, `exclude_tools` and `tool_order`; source artifacts and ordered
components; composition; result schemas; tool semantics; view config; identity;
capability binders), each a function in
`ora/agent/composition/tool_surface_resolution.py`, or in
`tool_surface.py` where it needs another resolver helper.

When a shared runtime result changes shape, gate it on a capability instead of
replacing it. Every frozen surface that publishes the old result schema is the
identity of a sealed run and will reject the new shape on its first call, so the
runtime must be able to serve both: publish a second capability, keep the
released implementation reachable, select between them from the declared
capability alone, and prove the old path still returns the released bytes rather
than assuming the extraction preserved them. Record such evidence from a fixture
with no numerical ties: a fixture with square masks gives two equal in-plane
eigenvalues, so `principal_axis_world` becomes whichever axis the host's LAPACK
returns and the golden holds only on the host that recorded it.

Components are named by path only. A manifest's `base_definitions`,
`additional_definitions`, each `supplementary_definitions` entry, each
`source_artifacts` entry and each `components` slot is a string path relative
to the manifest's directory, and `view_config.public_geometry_descriptor` is a
repository-relative path:

```json
"base_definitions": "../explicit_motion_v2/robotwin-definitions.json",
"components": {
  "visual_help": "../explicit_motion_v2/robotwin-visual_help.json"
},
"view_config": {
  "public_geometry_descriptor": "ora/policy/embodiments/aloha-agilex/descriptor-public-geometry-v5-session-split-v1.json"
}
```

Do not write a sha256 beside a component: the loader refuses
`{"ref": ..., "sha256": ...}`. Integrity comes from `policy.lock`. The loader
reads each component only if the lock lists it, hashes the bytes it read, and
records `{slot, ref, sha256}` in the resolved surface and the RunPlan
([path-only asset references](../reference/configuration-and-evidence.md#path-only-asset-references)).

To edit or add a component:

1. Edit the file, or create it and `git add` it.
2. Name it by path in the manifest.
3. Run `./scripts/ora lock` and commit `policy.lock` with the change.
   CI fails on a stale lock.
4. Run `scripts/compare-model-visible <base> HEAD` to see which delivered tool
   definitions moved.

A shared component is named, not pinned, by every surface that uses it, so
editing it changes all of them, including registered older surfaces. A frozen
surface keeps its exact identity only at its launch commit. When only one
surface should change, add a layer instead.
`./scripts/ora lock --strip-pins` converts a manifest that still
carries `{ref, sha256}` components, for example one merged from a branch that
predates this rule. It checks every pin against the file's bytes first.

A manifest may add tools with an `additional_definitions` component instead of
copying an entire base catalog. References may reuse another registered
profile's file under `ora/policy/tool_surfaces/`, but resolution
rejects paths outside that root and paths `policy.lock` does not list. Use this
only for exact shared components; the derived profile must own separate
changed descriptions, schemas or result contracts.

A live surface adds a *new* tool without rewriting a shared
`additional_definitions` file through `supplementary_definitions`: a list of
component paths, each a `{"tools": [...]}` catalog appended after the base and
additional definitions. A supplementary tool may not redefine an existing name
(resolution fails); changing an existing tool is still a description, schema
or result layer. No registered surface uses it at present: `reach_check` was
added this way and then undeclared by decision (retiring a tool means not
declaring it), which is why the layer, not an edit of the shared file, was the
right shape -- `explicit_motion_v3/robotwin-additional-definitions.json` is
named by 18 manifests, frozen ones included, and kept its bytes throughout.
When a sibling surface must reuse the list (the RoboDojo closed-skills surface
follows the RoboTwin one), compare it entry by entry as
`tests/test_robodojo_closed_skill_surface_contract.py` does.

A Dojo layer that *replaces* a whole description erases every shared append
ordered before it. Re-apply the shared append file under a `robodojo_` slot
after the replacing layer rather than copying its text into the replacement;
the parity test records such slots in `DESCRIPTION_REAPPLIED`.

Changing a tool name, description, schema, public result, effect or error
meaning is a policy behavior change. It requires a new hash-bound composition
and cannot be retroactively attributed to an earlier run.


## Restricted geometry evaluator audit (`compute`, B-II candidate)

`compute` is a read-only pure function over already-delivered numbers, built as a
B-II candidate and **not selected by any default baseline**. It is recorded here
because its result shape is dictated by contracts this guide owns, and because
two of those contracts protect run validity rather than one call.

**It is not a substitute for delivery.** A quantity the Harness can compute
*without knowing the policy's intent* must be delivered by its owning tool;
`compute` is only for combinations the policy must first decide in order to
name. The review test: if a `compute` usage could be precomputed by a function
that does not know the policy's intent, it is a delivery gap — fix the delivery,
do not widen the evaluator. Without this line the accepted-call table grows
without bound.

Authoring requirements specific to this shape:

- **Fixed result keys.** `evidence/trajectory.py::_forbidden_path` scans key
  paths, so a `{identifier: value}` mapping lets a block that binds `contact`
  invalidate the run and block sealing. The result is
  `{"outputs": [{"name", "value", "kind"}]}` and the policy's identifier appears
  only as a value.
- **Never echo the source block.** `_forbidden_path` and
  `context/projection.py::_causal_projection_violations` both `json.loads`
  strings that begin with `{` or `[`. A returned source string could therefore
  be opened as an object carrying a forbidden key.
- **Name skipped bindings.** A non-numeric intermediate is reported in
  `intermediates_not_returned`, never dropped silently; silent omission is the
  defect class this unit exists to remove.
- **Declare every returned property at the result schema's top level.**
  `native/session.py::_declared_result_properties` projects only declared
  top-level properties, and drops the rest without an error.
- **Fit the decision-critical values in 2048 bytes at the top level**, because
  an externalised result is previewed by a byte prefix.
- **Every policy-fixable failure is `invalid_arguments`.** Syntax, rejected
  construct, unknown name, malformed reference, timeout, memory ceiling and
  oversized result all carry `field`, `received` and the violated bound.
  `internal_tool_error` is reserved for evaluator faults the policy cannot fix.
- **`Use when:` must require batching.** One call evaluates every quantity that
  step needs. Per-quantity round trips cost more than the hand derivation they
  replace, because each call re-reads the whole cached context.

Effect classification needs no code: a tool absent from the manifest's
`mutation_tools` and `lifecycle_tools` is machine-provably `read_only` under
`native_atomic_causal`, and a handler that returns no `_runtime.directive`
consumes a model call without consuming a robot decision.

### Enforced boundaries the kernel learned the hard way

- **Attributes are an allowlist, not a dunder denylist.** Layer 2 withholds
  `numpy.load` and `numpy.fromfile` by never exposing a module object, but
  `ndarray.tofile`, `ndarray.dump` and `ndarray.ctypes` are the same capability
  reached from a value the block already produced, and
  `Rotation.identity().random()` reaches a classmethod through an instance.
  Rejecting only names beginning with `_` left all of those reachable, and
  randomness in particular contradicts the receipt being re-executable offline.
  Empty builtins is not a substitute: numpy happens to fail the file writers
  with `KeyError: '__import__'` from its own internal import, which is an
  accident of the dependency and reaches the policy as a message it cannot act
  on. The allowlist is derived from the callable tables plus the numeric
  conversions of an already-produced value, so a table entry cannot be added and
  left unreachable.
- **Provenance belongs under `_runtime`, and the placement is forced.** `schema`
  and `sha256` are both in `context/projection.py::_CAUSAL_FORBIDDEN_KEYS`, so a
  receipt left in the visible half fails the causal projection audit.
  `native/session.py` pops `_runtime` before building the tool message and
  `skill_receipt.py` excludes it from the measured body, so the receipt is
  recorded as evidence without being shown to the model or charged against
  delivery. Audit a result the way session.py builds it; auditing the raw
  payload tests a string that is never put on the wire.
- **A reference form must not be mandatory.** A binding is either a bare value
  or `{"value": ..., "from": "<origin>"}`. Requiring the annotated form would
  stop the policy bringing its own constant, which is legitimate and common. The
  receipt records each input's declared origin and digest beside the digest of
  the block, which is what an offline recomputation needs; the block itself is
  already recorded verbatim in the tool call, so the receipt binds it rather
  than echoing it.
- **A tool that owns a subprocess needs a teardown hook that exists.** Neither
  `ManipulationToolRuntime` nor `RobotToolRuntime` has one and nothing calls
  `close()`, so a per-session worker leaks one process per episode. Until a
  session teardown hook exists, bind the process-wide instance closed at
  interpreter exit; per-call freshness comes from the per-request fork into a new
  empty namespace, not from the identity of the parent.

## Planning-time IK selection in `grasp_plan`

`grasp_plan` on the live dual-arm surfaces checks its whole compiled pool (up to
32) with the backend's internal read-only IK check before the display cap and
orders reachable, then within 30 degrees of vertical, then backend order. The
check is deliberately not a policy tool: no surface declares `reach_check`
(user decision, 2026-09-29) and this tree has no handler for it, so the policy
sees only the annotation. Never drop a proposal below the cap and never block it on the
annotation: an unavailable IK bridge keeps the backend order and says
`status: unavailable`. The help must call the annotation an IK reference at
planning time, not a guarantee, and must not name the undeclared tool. Nor may
it keep the unannotated surfaces' "Reachability is delegated to move_pose or
grasp_execute; this call does not run a motion probe" beside that paragraph:
EXP11B revision 20 of the Dojo surface replaced it with a pointer to the IK
selection paragraph, and `max_candidates` names the IK-ordered pool. A grasp
result field that depends on a backend's refusal vocabulary is derived from the
closed per-backend set in `contracts/reach_check.py`
(`grasp_ik_reachability_schema.py`), not restated.

A binder's file hash is a component of every surface it binds. Adding a backend
to a released binder therefore changes released surfaces, which is why LIBERO
binds through the sibling
`grasp_ik_reachability_libero_schema.py`, keyed on its identity capability
`libero_pro_rgbd_v1`, with the released binder's constants and stage helper and
a test that the two bindings differ only in the reason enum. Its help also
states that this backend's check is not `move_pose`'s admission and has no
measured disagreement yet, instead of reusing another backend's figures.

Publish the member that will execute. An annotation that names a better member
while the pose fields keep another is a pose the policy, the copyable
`move_pose` target and `grasp_execute` will all run anyway: EXP11A g007 annotated
`member: exact` on a wall pinch whose compiled orientation solved 0.14 rad from
the X5 joint6 stop, and the precommit kept it because it was admitted. Under
`grasp_plan_jaw_symmetry_headroom_v1` the selection rewrites `pregrasp_pose` and
`contact_pose` to the chosen member's exact quaternions and says so in
`ik_reachability.jaw_symmetry`. A derived quantity a row does not carry (the
X5 joint6 margin) is published only with the argument that isolates it from
the row's own numbers, and is null when that argument does not hold; one that
would need a guessed IK branch (an elbow-extension margin) is left out rather
than estimated.

## Help that says what the handler does

Read a sentence against the handler before shipping it; the EXP11B g003 audit
found these defect classes on the current surfaces
([corrections](../reference/public-interfaces.md#tool-description-consistency)):

- **A per-entry outcome is not a refusal.** When a handler reports a problem
  inside a successful result (`geometry_query` points `valid: false`, a
  `mark_point` batch entry with `error_code: inconsistent_mark`), the help says
  which entry fails and what the call still returns; "refused" or "fails
  explicitly" belong only to errors that end the call.
- **State a default as its value.** On a surface declaring
  `optional_argument_null_as_omitted_v1` write "default 3", not "omit for
  three": null and omission both select it, and the composed "Null means
  omitted." then needs no second step. Advice that assumes another starting
  value ("raise it from one") contradicts the default and is task guidance.
- **One owner for a rule.** A selection rule that belongs to one tool lives in
  that tool's `Use when`; another tool points to it in one sentence rather than
  repeating it. Keep a repetition only when the tool needs it to be
  self-contained: its own result fields, a producer's complete handoff, its
  Returns section, or a pointer to the single owner of shared numbers (the
  workspace bounds live only in `move_pose`).
- **Capability-gated behaviour gates its wording.** A sentence true without a
  capability (no IK in `grasp_plan`, a failure exit in `finish_episode`) must
  not survive on a surface that declares it.

Correct a live surface with its own description or schema layer generated from
its composed text by exact sentence edits, never by editing the shared file the
sentence lives in: those files are named by older registered surfaces as well,
and an edit would change what each of them delivers.

## A feature lands on every backend's default

A Harness behaviour that is not about one backend's identity, embodiment or
mechanism is declared on all three defaults (RoboTwin, LIBERO-Pro, RoboDojo).
`ci/backend-parity.json` classifies every capability and tool a default
declares, and `tests/test_backend_parity.py` (always selected) refuses:

- an unclassified name;
- a `generic` or `generic_needs_backend_code` name missing from a default
  without an exception that cites a decision record containing the exception
  id; a missing backend hook is named (`missing_hook`: a file that `defines` a
  function, a constant that `contains` a value, or a module glob that
  `mentions` an identifier) with its follow-up unit;
- an exception that has expired: the backend declares the name, or the hook it
  waited for now exists -- so whoever writes the backend code also declares the
  capability;
- a `backend_specific` name declared outside its `applies_to`;
- a default swap that drops a name of the default it replaced without a
  recorded retirement (`previous_defaults`, updated in the swap itself).

A successor capability covers its predecessor (`supersedes`), and a default may
not declare both. When a port is written for one backend first, keep it
capability-gated and add its exception in the same change, so the next default
swap cannot silently leave it behind.

## Every number in help has an owner

An enforced number that a tool's help states — a bound the handler refuses
past, a routing threshold, a fixed parameter such as a lift height or a tick
length — is one row of
`ora/policy/tool_surfaces/published_limits_v1/limits.json`. The row
names the owner the code enforces: a constant, a profile value by JSON pointer,
a source literal matched exactly once, or a value derived from those. It also
lists each sentence that states the number, per surface.
`tests/test_published_limits.py` composes the current defaults and the
unregistered successor texts and compares every stated number with its owner
after unit conversion. A census refuses any digit in a composed tool text that
neither a row nor a reviewed informational entry covers
([reference](../reference/public-interfaces.md#published-limits-and-stated-refusals)).

- Add the row in the change that adds the number. A measured figure (a
  latency, a median distance, a free-air timing) or an illustrative example is
  an `informational` entry with its reason, never a limit.
- A number no repository code enforces gets an `unenforced` or `external`
  owner. An example is the RoboDojo SDK's IK seed count and residual
  tolerances, which are read from the SDK at run time. Such a number may
  remain on a frozen surface as a strict expected failure with its reason. It
  may not appear on a successor.
- State refusals as rules. On a successor generation `On error:` opens with
  `Refused:` and lists:
  - what the handler refuses before anything is submitted, with the table's
    numbers;
  - what the backend rejects before any controller write.

  It never says what to try instead. A per-entry failure or a dropped
  candidate is not a refusal and says so.
- A capability binder runs after every layer. Edit binder-written text in its
  binder: the compute grammar, ARX controller feedback, the abort class list
  and the workspace JSON. A review lists such tools
  under `refusals_owned_elsewhere`, and a text written for a capability the
  base surface lacks lists it under `presupposes_capabilities`.
- Generate a surface's layers from its review with
  `uv run --locked python -m ora.agent.composition.description_edits <review> [--check]`.
  It applies each exact sentence edit to the text as composed before the
  binders: the base profile's, or with `edits_apply_to: successor_before_layers`
  the registered successor's own text before its two published-limits slots.
  Each edit carries its reason and its Chinese reference. A generated layer
  replaces whole descriptions, so it is the last description and schema slot of
  the manifest; regenerate it, and re-pin the manifest, whenever an earlier layer
  or the base text changes.

Phase 2 (not built) generates these sentences from the same owners, so a
changed constant changes the help in the same commit.

## Publishing a capability the runtime already has

Before writing a tool, check whether the handler exists and only the dispatch or
the declaration is missing. `RobotToolRuntime._geometry` has served world-to-pixel
projection, pixel-to-world back-projection and region depth all along, declared
as `geometry.query` by `robot_runtime_v1`, `robot_runtime_v2`,
`robot_runtime_v3`, `robot_runtime_native_prompt_v2` and
`robotwin_dual_layer_v1` -- five of the 89 registered profiles. (Three of 31
*directories*: `robot_runtime_v2/` registers three profiles. Count the catalog,
not the tree.) `ManipulationToolRuntime`
inherits the method but overrides `invoke` without delegating, so every surface
built on it could not dispatch the name however it was spelled.

The measurable consequence is the reason this section exists. Across 177 sealed
runs, 51 (29%) show the policy deriving `u = cx + fx * x / z` by hand against
intrinsics it read off calibration, and those are precisely the runs whose
surface omits the tool; the five runs that had it called it instead. A
capability gap and a publication gap look identical from the trace and are
fixed differently, so establish which one you have before building anything.

New declarations use the snake_case name, `geometry_query`. The three surfaces
that already froze `geometry.query` keep that spelling: a frozen surface is
evidence of what a past run was shown, and re-spelling it would rewrite that.

Both active `*_placement_explicit_motion_v3` surfaces now publish it, through an
`additional_definitions` component plus one `result_order` component each, so no
frozen catalog is touched. Two authoring decisions there are general.

**A new name is a new contract, so inherit the shape rather than the defects.**
The frozen input schema is a top-level `oneOf` whose branches carry `required`
alone. `agent/native/gateway.py::_openai_parameters` flattens a root union only
when every branch carries `properties`, so this one returns verbatim and reaches
the provider — and `validate_schema_instance` then answers a missing operation
with "must match exactly one branch", which is run H6-013. The published schema
declares the three operations as sibling properties with `required: ["camera"]`
and leaves exclusivity to the handler, which already raises a structured
`invalid_arguments` naming all three fields. The union enforced nothing the
handler does not, and cost the diagnosis. This is the same repair
`flat_placement_arguments_v1` made; reaching for `anyOf` instead would restore
the union under a different keyword. Because the new key is distinct, the three
frozen surfaces keep their `oneOf` and their bytes — re-spelling is what would
have made the change expensive, not flattening.

**A released result schema is proof the shape was accepted, not that it is
closed.** The frozen one declares `region_result` as a bare `{"type": "object"}`
and `points` as a bare array, which cannot catch field drift the contract exists
to catch. The published schema closes all three branches, with each branch's
`required` set as the intersection of the handler's own variants — a projected
point behind the camera returns no `pixel_uv` at all. A tightened result schema
is enforced on live calls, so it has to be exercised against real handler output
before it ships rather than asserted: `tests/test_geometry_query_publication.py`
drives all three operations and both per-entry variants through `_geometry`.

## A tool name is prompt vocabulary

Prompt-bundle audits forbid every declared tool name, as a whole word and
case-insensitively, in the planner, core, environment, embodiment and
episode-task layers, so that tool syntax has one owner. A tool therefore cannot
take a common English word the layers already use. The shared motion-v3 `wait`
action is submitted by `wait_ticks`: every planner layer already says "wait for
the next actual observation", and naming the tool `wait` failed the
`planner_single_owner_vocabulary` check for every current bundle. The same rule
keeps task-contract notes tool-neutral: the RoboDojo `imitate_sorting_sequence`
note says the demonstration advances only while native actions are consumed and
asks the robot to wait, but names no tool, so it stays true on a surface
without `wait_ticks`. Check a proposed name against the rendered layers with the
prompt-bundle audit before publishing it.
