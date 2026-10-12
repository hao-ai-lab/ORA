# Test system

`test_output_storage.py` checks host-local shared output selectors, explicit
workspace overrides, logical run indexing, linked discovery and rejection of
conflicting namespaces or escaped runs. These checks use temporary filesystems.

`test_run_id_uniqueness.py` checks that spawn finds a taken run-id at its exact
locations (run index, planned root, each discovery root's
`<plan experiment>/runs/<run_id>`, legacy root, a dangling link) with every
directory listing forbidden; that the namespace is the plan's experiment even
when `outputs/<experiment>` links to a differently named directory; that an
uncheckable location refuses while a non-directory parent is free; that
`resolve_run_root` keeps its recursive reach; and that `_spawn` refuses a taken
id before any lease without calling `resolve_run_root`. It uses temporary
filesystems and stubs the launcher.

`test_yaml_parse_cache.py` checks the content-addressed YAML parse cache: one
parse per distinct content (shared across paths, none for a same-bytes
rewrite), a same-size in-place rewrite visible at once across 200 rounds,
caller-owned copies over a frozen shared view (including `!!pairs` values),
refusal without caching of unreadable, malformed or recursive documents, the
libyaml-first loader with its `SafeLoader` fallback, identical values and types
from both loaders for every tracked host profile, experiment declaration and
prompt bundle, and one host-profile parse across repeated resource resolutions
that still fail closed on malformed, recursive or non-UTF-8 profiles.

`test_kit_pid_collision.py` checks that only `robodojo`/`robolab` workers look
for another user's `carb-RStringInternals-<own pid>` segment (removing their own
stale copy, reading `SHM_ROOT` at call time), report it and exit with 73 before
Kit starts, and that the RPC launcher relaunches only those backends, at most
three times, recording worker PID and segment in a field only they carry; other
exit codes and other backends are unchanged. It also checks that the RoboDojo
bootstrap exports `OPTIX_CACHE_PATH` under its runtime directory. Worker
processes are real, Kit never starts and every `/dev/shm` is a temporary
directory. `test_robodojo_runtime.py` runs the real worker under the `robodojo`
backend, so its fake worker points `SHM_ROOT` at a temporary directory as well;
the conftest host-storage stub does not reach worker processes.

`test_host_storage_preflight.py` checks the host storage preflight for every
runtime kind: inclusive capacity floors with measured values, the system
partition only for Kit runtimes and the managed Codex/Claude Code CLIs (one
measurement per device),
0 as a disabled floor, host-profile selection by execution host, fail-closed
`runtime_storage`, the scratch RPC socket budget, a dangling workspace link,
the exact carb name pattern and out-of-range PIDs for own and foreign segments,
a report-only preflight census, the launch-time own dead-PID sweep before the
lease, per-run scratch outside the scanned evidence trees with owner binding,
release, removal after a failed lease or cohort claim and residual census, the
public `resources.load_host_profile`, and `TMPDIR`/`TMP`/`TEMP` delivery. Probes are
faked; no real disk, `/dev/shm` segment or evidence workspace is touched. The
autouse fixture in `conftest.py` stubs the orchestrator's `host_storage` entry
points for every other test; it does not reach a separate worker process.

`test_experiment_identity.py` covers hash-bound identity inheritance, immutable historical validation, ambiguous/cyclic lineage rejection, exact-source alias isolation and Viewer family navigation without simulator startup.

`test_native_harness_openai.py` checks strict OpenAI tool aliases, streamed/unary
response restoration, historical reasoning replay and collision rejection, and
the OpenAI-compatible `x_session_id_run_v1` affinity: the client sends one
stable `X-Session-ID` only when bound and refuses it off HTTP loopback, RunPlan
resolution admits the mode only on a loopback gateway, and a loopback profile
may declare it with `output_token_parameter`; under either selector the
recorded wire request is the body every attempt sends, so failed attempts join
their `provider_request`.
Provider boundary regressions cover rejection without edited assistant replay,
fragmented tool names, bounded clean-EOF recovery with attempt evidence,
non-object Anthropic inputs and exact string arguments in unsigned history.

`test_robodojo_b2_parity.py` also covers command-history recovery after an
earlier evidence gap and repeated v3 quaternion normalization through the
shared command tree. Invalid directives, missing execution, source mismatch
and material pose changes remain rejected. These are offline interface
regressions; they do not establish physical grasp or release success.

`test_robotwin_evaluation_setting.py` provides host-only regression coverage for
the official 50-task roster, evaluation seed domain, the two official scene
configurations (`demo_clean`, `demo_randomized`) and their randomization,
oracle unseen scene mode, private failure/distractor evidence, bounded parallel
prepare with a per-task accepted-seed target, the sealed per-task seed
manifests `qualify` writes from the worker ledger, the `oracle_exception`
rejection of an expert exception from RoboTwin task code, and the evaluation
revision's equality with the vendored install. It does not initialize a
simulator or claim physical qualification coverage.

`test_robotwin_seed_segments.py` covers seed-scan segments with fixture
manifests and ledgers: offset and explicit `task:offset[:count]` cases, the
first-N merge over a contiguous prefix (gaps, refused conflicts, revision and
task-level closure checks, admission by `seed_admission`), interim seals of a
running ledger and the `merge`/`seal` entry points.
`test_robotwin_seed_lanes.py` covers lane execution: the lease-probe retry,
reading a ledger with a torn final line, the seed watchdog against a fixture
worker process and the sealing filter. Neither starts an oracle worker.

The test suite protects stable Runtime boundaries, not implementation trivia.
Run all deterministic tests from the repository root:

```bash
uv run --locked pytest -q
```

Use `uv run --locked pytest --collect-only -q` to inspect individual test
functions. Function names and docstrings live with the executable source; they
are not copied into this README.

The controller-feedback regressions cover SDK post-limit close versus intent,
both Dojo arms, held-target pose continuation, legacy close-floor convergence,
paired arrival bounds, LIBERO sparse wrapping, Native OpenAI/Anthropic and
managed Codex/Claude delivery, malformed evidence and dual-arm correction.
They use recorded numeric witnesses and fake plants, with no simulator or
provider invocation. `test_parameter_delivery_guards.py` checks actual values
surviving projection as well as the static failure-evidence registry, and
registers every RoboDojo IK candidate rejection code with the `ik_selection`
numbers it must deliver (a `pitch_joint_wrap` refusal carries the joint, its
measured and commanded values and the interval).

## Machine-owned inventory

[`test-manifest.json`](test-manifest.json) is the complete module inventory.
Every tracked `tests/test_*.py` module appears exactly once with its path and
one human-readable contract.
`tests/test_control_plane_cli.py::test_documented_script_and_test_module_inventory_is_current`
checks that the manifest lists exactly the set of `tests/test_*.py` modules.

## Suite responsibilities

`test_request_archive.py` builds a sealed run fixture and checks the post-seal
provider-request archive: only requests are archived, reads and `export` return
the sealed bytes, seal inventory records stay satisfied, runtime metrics read
through the archive, the 2026-10-09 all-JSON format converts to requests-only,
and nothing changes without `tar`/`zstd` or a seal. It needs the two
executables and skips otherwise.

`test_viewer_process.py` checks that list pages defer full media projection
and a slow discovery scan receives its complete cache lifetime after finishing.
These checks use CPU fixtures and start no simulator or model service.

`test_runtime_startup_contracts.py` checks preflight/factory agreement for exact
RoboTwin controller bindings, shared preflight/tool endpoint selection with real
Panda/ALOHA client constructors, mismatched override rejection and immutable
single-attempt refusal diagnostics through CLI stderr/exit status. It starts no
simulator or resident service and replaces network health I/O with fixtures.

`test_task_contract_review.py` checks all fifty v3 review records, corrected QR
and task-specific semantic regressions, bounded witness
claims, private/public separation and rejection of contradictory/unmapped clauses.
`test_robodojo_task_contracts.py` checks that the default registry-v8 delivers
only reviewed layers within the admitted task set and native-action limits, and
holds the first four contracted tasks' records and layers to their review rules;
`test_robodojo_spec.py` covers the maintained 42-task roster and each fluid
task's one-round CPU composition and K >= 2 refusal.
`test_robodojo_admission.py` also pins the admitted RoboDojo task set: it equals
all 42 official simulation tasks (the two fluid tasks since 2026-10-04), each
limit equals an independently written upstream `step_lim` table, every admission
path (RunPlan validation, physics ceiling, layout selection, worker manifest
binding, session) accepts the 25 tasks added on 2026-10-02 and the two fluid
tasks added on 2026-10-04 with their own layout, limit and physics device and
refuses unknown, `_random` and case-changed names; with the same mounted
source and Assets it re-reads the official list and each `step_lim` from the
pinned revision and checks that only the fluid tasks' layout 0 has a Fluid
section and that they resolve CPU physics.
`test_robodojo_single_layout.py` covers one unscheduled RoboDojo episode on an
official layout k (EXP14A g002): admission of k's canonical file, the worker's
round-0 and in-place reset binding, `l<k>` lane gates and cells, the
layout-portability table v2 (layouts 0-9) against registry-v8 (and, with the
Assets mounted, the pinned layout bytes), contract delivery and refusals, spec composition with the
record or lane-gate readback instruction, and the revalidation readback rule.
It needs no simulator; `test_robodojo_layout_schedule.py` keeps the scheduled
sessions.
`test_plan_feedback.py`
checks mixed planner/admission rejection, bounded single/dual-arm candidate
summaries, privacy and propagation through both Native Provider wire formats.
These are offline checks, not new physical or Agent-success qualification.

The public-data-v3 regression group is offline and source-bound:

- `test_public_robot_facts.py`: all-camera image/calibration joins, current and
  historical wrist transforms, native sheet pixels, Native Provider and managed
  input delivery, actual observe results and private-field rejection.
- `test_public_geometry_delivery.py`: closed geometry schema, deterministic
  cards, loaded-geometry comparison/drift, evidence retention, fresh defaults
  and command/encoder delivery.
- `test_controlled_point_projection.py`: each calibrated camera's public TCP
  pixel, in-image flag and depth for every arm, including points off the image
  and behind the camera; delivered and admitted by result schemas only on
  surfaces that declare `controlled_point_projection`, with arithmetic checked
  against the measured ALOHA and Panda wrist geometry. On the live RoboDojo
  surface, recorded ARX calibration and link6 poses check both the link6 pixels
  (behind each own wrist camera) and the descriptor jaw-contact-centre pixels,
  and the frozen Dojo predecessor keeps the closed camera entry.
- `test_libero_gripper_feedback.py`: a LIBERO jaw command ends early when the
  measured opening stalls short of its target and reports
  `gripper_stalled_before_target` with the outcome in physical units, while a
  free-air close never stalls; the live direct surface constrains the
  segmentation camera, states gripper timing in `set_gripper` help and
  publishes the commanded gripper beside the encoder only where declared. The
  traces are measured ticks, so this checks the stall rule, not simulator
  contact dynamics.
- `test_robodojo_gripper_feedback.py`: the shared stall rule on recorded ARX
  native-action traces (free air arrives, a close on an object ends three still
  native actions after the jaws stop, a creeping close still ends before the
  budget), one public code per dual-arm jaw command, the fake-plant receipt's
  per-arm `gripper_outcome` and its policy projection, and live `set_gripper`
  help numbers bound to the executed rule and ARX contract.
- `test_robodojo_live_surface_repairs.py`: the live Dojo `align_plan` result
  enum equals the rejection reasons the reachability adapter can produce
  (including `ik_branch_jump`), and
  historical observe admits exactly the cameras each live surface's observe
  schema advertises.
- `test_robodojo_perception.py`: the RoboDojo three-camera perception path and
  ARX X5 grasp geometry on synthetic RoboDojo-layout observations. It also
  holds `grounding_candidate_depth_profile_v1`: the profiled candidate card is
  1024x800 with the released 1024x576 pixels above a strip that draws the
  candidate's own depth-inlier points (count equal to its localization's) in a
  window of `max(0.10 m, 2.5 x largest extent)`; the dashed `support_z_m` line
  appears only when the surface declares the support fields and the value lies
  in the window; target refs and the policy-visible preview keys are unchanged;
  only a surface declaring the capability selects it through the real
  dispatcher.
  For `observed_extent_scene_review_v1` a synthetic head-depth patch below the
  0.72 m workspace floor and background beside the workspace are counted and
  drawn by the observed-extent cell while the three clipped panels keep their
  records and their pixels below each title; the canvas stays 1168x768, the
  query identity gains only `review_renderer`, and an unusable source leaves an
  `unavailable` cell.
- `test_robodojo_closed_skill_surface_contract_v2.py`: the RoboDojo default
  `robodojo_placement_closed_skills_v2` (integration 0059 unit A0) has the
  RoboTwin v2 sibling's tools and effect classes, the capability difference is
  an enumerated list of backend facts and version splits, and definition bytes
  stay under a reviewed ceiling against the sibling.
- `test_robodojo_closed_skill_surface_contract.py`: the RoboDojo default
  `robodojo_placement_closed_skills_v2` carries Dojo backend facts, not
  RoboTwin ones: its `grasp_plan` IK annotation and `align_plan` rejection
  reasons admit exactly what the Dojo check and adapter return, its view config
  binds the Dojo cameras and the `arx_x5_link6` controlled frame and otherwise
  equals the `robotwin_toolchain_b2_closed_noservo_v2` sibling's, and no
  RoboTwin/ALOHA camera, frame, planner or evidence term survives in its help
  or input schemas.
- `test_overhead_view.py`: `overhead_footprint_localization_v1`: the footprint
  centre of an overhead outline (mug rim plus handle) is its occupancy-box
  centre and is classified overhead, a side camera is oblique, too few points
  give the empty shape; `overhead_view_pose` puts the wrist camera 30 cm above
  the target looking straight down from any current grip pose; only oblique
  measurements receive it, and undeclaring surfaces receive no footprint field.
- `test_target_support_height.py`: `target_support_height_v1` on ray-cast
  RGB-D (a box on a table and on a shelf, three viewpoints): the support-plane
  estimate recovers the support height, visible top and mid-height that the
  visible-surface `center_xyz_m` misses, missing support evidence publishes
  nulls, and delivery follows the declared capabilities: the LIBERO, RoboTwin
  and Dojo default surfaces declare it with the principal frame.
- `test_arx_embodiment_card_v4.py`: the live ARX descriptor (v5) card keeps
  the v4 wording (one controlled-point name, the local Z pad axis, the
  +0.1501 m jaw contact centre).
- `test_arx_embodiment_card_v5.py`: the ARX descriptor v5 card states the
  encoder sentence (both fingers' mean on the normalized command scale; a
  one-sided pair is no hold or release evidence) and is what the live Dojo
  bundle, profile v6, surface and descriptor registry pin.
- `test_robodojo_two_finger_jaw.py`: the ARX public jaw reading is both fingers'
  mean (EXP11A g012 E11G12-001). The versioned feedback source carries the
  signed finger lateral offset, recorded divergent readings are reproduced,
  symmetric fingers publish the released value, the gripper outcome names the
  SDK primary-relative hold, the stall rule waits for both fingers, the jaw
  probe compares both fingers, jaw_hold stays unknown for a one-sided pair,
  observe delivers the reading through the live result schema, and control and
  journal replays compare the jaw reading only under one measurement source.
- `test_robodojo_probe_ik_feedback.py`: D3L-R002. A v2 motion-probe request
  keeps the SDK IK candidate counts, per-filter rejections and the retained
  diagnostic reference on each variant's planner record; the v1 record keeps
  the released shape the frozen unified probe schema accepts (and would
  refuse the v2 record); only the declaring surface sends v2, and its
  `grasp_execute` pregrasp refusal summary names the counts. A
  `pitch_joint_wrap` rejection is counted, not dropped.
- `test_robotwin_probe_ik_feedback.py`: 0059. RoboTwin answers the same v2
  request on RoboDojo's candidate set and unit: `returned` and `valid_endpoint`
  from the vendored CuRobo solve's own IK summary, `continuous` 0 only when no
  endpoint is valid and otherwise null (`joint_jump_filter_not_applied`), the
  three RoboDojo-only filters null in `constraint_rejection_counts`, outcomes
  `admitted`, `no_valid_endpoint` or `no_admitted_trajectory`. The v1 record
  stays byte-identical, malformed requests and pre-v10 control profiles are
  refused, the agent summary withholds a count any variant left unknown, the
  pregrasp refusal names the counts, and no registered RoboTwin surface sends v2.
- `test_robodojo_observe_result.py`: The live RoboDojo observe result admits
  what every Dojo observation carries (three Dojo cameras, the ARX jaw branch
  and the closed controller_feedback contract) in the recorded EXP11A g003
  imitate_sorting_sequence shape, through the real dispatcher, and the released
  v2 result is pinned as rejecting it.
- `test_robodojo_inter_arm_clearance.py`: the inter-arm sphere screen (EXP11A
  g012 E11G12-004a). Both arms' solver spheres sit at their world bases with
  unbuffered radii plus the solver's world buffer; lockstep straight joint
  lines are refused below the 5 mm margin unless they leave an existing
  overlap; placement fails closed. With the pinned X5 assets, recorded g012
  moves that stalled against the other arm are refused and an arrived
  neighbouring move is admitted. IK candidate selection skips and names paths
  that meet the other arm, which moves from its measured joints to the target
  the plant still applies to it, an arm moving with the other is left to the
  lockstep screen, and a one-armed session screens nothing.
- `test_robodojo_inter_arm_admission.py`: every RoboDojo arm-motion admission
  asks the inter-arm screen before writing -- single-arm global and local moves
  (`inter_arm_collision:<segment>`, public `planner_failed` with the outcome,
  links and clearance), dual-arm segments in lockstep, motion probes, the
  one-arm recovery order, two-arm recovery refused before motion or stopped
  mid-route (`recovery_inter_arm_collision_path`, passed through publicly), the
  released straight recovery, a refused back-off probe that drives no back-off,
  level-2 admission on the yaw-first order and the reach-check reason. Fake
  plant only.
- `test_robodojo_recovery.py`: RoboDojo recovery from the recorded EXP11A g003
  wound joint states: the stall rule separates the recorded blocked and moving
  traces, unwind orders fall back straight/fold/yaw on the SDK sphere screen
  (pinned X5 assets when present), the released route stops a blocked arm early
  as controller_stalled, retract-then-unwind frees the recorded state and
  finishes on a second call, an all-blocked plan refuses before any write,
  public codes and recovery_progress reach the policy, and the RunSpec derives
  the route only from the declaring surface. EXP11A g005 audit D3: an arm a
  whole turn from home recovers in place to the FK-confirmed nearest home
  equivalent on both routes (literal home would need a 2π sweep), the receipt
  records `unwind_target`, and literal home is kept when FK does not confirm
  within 1e-6 m / 1e-5 rad (review U1). A stored value outside its limits
  gains no equivalent (joint6 at 3.1432), a pitch-joint turn that would leave
  [-π, π] is dropped as `pitch_vetoed_turns`, a stored target already farther
  out is flagged as `stored_pitch_wrap`, and non-finite stored joints refuse as
  `recovery_state_unavailable` before any write.
- `test_robodojo_motion_stall.py`: EXP11A D2. RoboDojo arm motion stops on the
  shared `JointBlockedMonitor` fed once per native action and holds where it
  stopped. Synthetic joint trajectories: an arm still closing on its target
  never stalls, three still native actions (every joint under 0.005 rad) stall
  and one moving action resets the count, and the compact evidence names
  measured joints, joint residual, still actions and the hold. On the fake plant
  a move, local step, free-space descent, dual-arm move and probed plan
  reference pressed into a table end as `controller_stalled` with
  `arms.<arm>.motion_progress` and a journaled `stall_hold` native action; a
  stalled arm's applied targets stay at its measured joints through the next
  `wait` and `set_gripper` with no further creep, also when it stalls on the
  last budgeted action (retained command rewritten) or while the other arm keeps
  moving; an arm still moving at the window end keeps
  `native_action_budget_exhausted`; recovery keeps no copy of its own rule; the
  projection never calls a stall `controller_timeout`, a `grasp_execute` stage
  carries `motion_progress` and the public code, and both Dojo surfaces' help
  states the stall and hold.
- `test_jaw_hold_evidence.py`: The jaw_hold verify signal classifies recorded
  g003 close-then-lift jaw readings as likely, unlikely or unknown with the
  readings and tolerance beside it, never guesses a missing tolerance, calls a
  close that never left the open position `unlikely` (`jaws_never_closed`, the
  recorded EXP11A g007 +31.7 min false positive) while a mid-travel stall stays
  `likely`, and appears only on the declaring surface beside the unavailable
  closure strength.
- `test_placement_refusal_measurements.py`: the retained-point refusal names
  count, region field, camera and bounds, and a reused region names its field
  and reference.
- `test_wait_action.py`: The shared motion-v3 wait: the contract admits exactly
  1..20 ticks; RoboDojo holds every command, sends empty SDK commands and
  advances exactly the requested native actions (stopping at an official
  terminal); RoboTwin steps ten physics steps per tick without a drive write
  through the session's own sparse helpers and receipt; the wait block reaches
  the policy without its schema; the three live surfaces publish one wait_ticks
  contract and the tool names its bound on refusal.
- `test_libero_wait_action.py`: 0059. LIBERO executes the shared wait: one tick
  is one control step (one `env.step`, 25 physics steps of 2 ms read from the
  live worker; any other control period is refused before a step), each tick
  sends the contract-bound zero-delta OSC_POSE command (all zeros under the
  pinned ranges; pose goal = measured pose, gripper directive unchanged), and the sparse receipt carries `target_arms: []`, the
  hold command invariant, measured drift, OSC-bound control health and the
  shared `wait` block with `physics_steps_per_tick: 25`. It stops at the
  official termination or the Harness step budget, faults on a changed
  directive, persists, counts as a world mutation and replays as
  `libero_osc_pose`. Only waits take the unpinned execution-capture route;
  `WAIT_ENVIRONMENTS` and the v3 descriptor advertise `wait`; learning
  eligibility, the control audit and the LIBERO placement stage history judge
  the wait under its own contract.
- `test_reach_check.py`: the internal read-only reach check and the pooled
  `grasp_plan` IK selection on the three live dual-arm surfaces, none of which
  declares `reach_check` (the runtime refuses it). It covers the closed
  query/row contract routed through `plan_alignment`, RoboTwin
  damped-least-squares answers with chaining and
  `matches_move_pose_admission: false`, RoboDojo SDK IK on the route each
  interaction selects with chained starts and every executor refusal named,
  the agent edge's workspace pre-check, the loader's refusal of a
  supplementary layer that redefines a tool, and a whole-pool check before the
  display cap that orders reachable, then near-vertical, then backend order
  without dropping a candidate.
- `test_libero_reach_check.py`: integration 0059 M2. LIBERO answers the
  internal reach check with one CuRobo v0.7.8 IK solve per pose in the
  isolated planner worker (CPU CuRobo stand-in: measured or chained seed as
  retract and seed, orphans skipped, no plan), compiled from the world grip
  site into CuRobo's base-frame `panda_hand` exactly as the global `move_pose`
  segment compiles it, judged back at the grip site against 0.005 m / 0.05 rad,
  with `ik_failed`, `fk_residual_exceeded`, `start_unreachable` and a typed
  `fk_calibration_mismatch`; the bridge serves only the reach-check schema and
  the right arm, the real session never commands or steps, the host client
  contract is closed, and the LIBERO reason set binds through its own module
  while the released binder file keeps its bytes.
- `test_libero_probe_ik_feedback.py`: integration 0059 M2. A v2 motion probe on
  LIBERO keeps each variant's CuRobo IK seed population (from that variant's
  own plan) as `constraint_rejection_counts`, `planning_diagnostics`
  `candidate_counts {returned, valid_endpoint}` and an `ik_validation_artifact`,
  the three fields RoboDojo adds; the v1 record and the pinned
  `motion_probe.py` are unchanged.
- `test_libero_reach_ports.py`: integration 0059 M1-M3 on the LIBERO default
  generation (`libero_placement_direct_overhead_v1`) over the real LIBERO bridge and a stand-in CuRobo IK: the pooled
  `grasp_plan` selection on the single-arm surface (right arm, LIBERO reason
  set, never dropping a candidate), the v2 `move_pose` and primitive reach
  pre-check naming LIBERO's IK with a null measured disagreement, the
  nearest-orientation sweep holding the grip site, and LIBERO help that states
  its own model and no other backend's numbers.
- `test_grasp_jaw_symmetry.py`: on the live RoboDojo surface `grasp_plan`
  checks both members of each candidate's parallel-jaw class and publishes the
  reachable one with more X5 joint6 (wrist-roll) margin, then the smaller joint
  move, then the compiled one, with `ik_reachability.jaw_symmetry` and
  `joint_headroom`; built from the EXP11A g007 +38.8 min probe joints. It also
  pins that the published poses are what `grasp_execute` commits, that joint6
  is reported only when the measured joints isolate it, and that RoboTwin and
  the released selection are unchanged.
- `test_move_pose_reach.py`: on the live RoboDojo surface `move_pose` checks
  each compiled target with the read-only reach check before submission. An
  empty-jaw `free_space` absolute target is submitted in its parallel-jaw turn
  when the requested turn does not solve or has under pi/4 rad of X5 joint6
  margin and the turn has more (built from the EXP11A g008 round-1 receipts:
  joint6 -2.465 and -3.007 from home, then a refused 5.5 rad branch jump); held
  or unknown jaws, relative orientations and local moves are never turned; an
  unsolved target reports the farthest gap-free fraction of its straight path;
  the dispatcher submits the turn it reports; RoboTwin surfaces do not declare it.
- `test_policy_abort_evidence.py`: the round-signal classifier behind the
  evidence-gated `abort` (`policy_abort_infrastructure_round_v2`) counts only an
  infrastructure-class failure in the run's event log since the round's reset:
  every distinct g012 tool-error and receipt failure shape is not one, each
  Harness-marked fault and infrastructure tool-error code is; the g013
  kernel unit's other-arm refusals (`inter_arm_collision:<segment>`,
  `recovery_inter_arm_collision_path`) are ordinary refusals and leave `abort`
  refused.
- `test_finish_outcomes.py`: the finish contract v2 (`policy_finish_outcomes_v2`,
  `policy_abort_infrastructure_round_v2`) on test-local v2 copies of the
  RoboTwin, LIBERO and RoboDojo defaults (`finish_v2_surfaces.py` copies a
  registered manifest into `tmp_path` and resolves every component at its
  pinned hash): `success` and `give_up` (plus evidence-gated `abort`), a
  required reason refused retryably with the field named when empty, a token,
  a placeholder or a success reason without an evidence source, a required
  give_up category, `failure` and `attempts` retired, no help naming the
  retired failure exit; a malformed argument is a policy refusal, never an
  `internal_tool_error` that would admit a later abort; CJK reasons and loose
  evidence wording; the current defaults keep their contract. The abort
  refusal is retryable without the changed-arguments flag, lists the counted
  classes and the signals read, and the same abort is admitted once a fault is
  recorded; a latched success is refused first and is retryable; an
  unconfirmed success is retryable only while the round is not terminal.
- `test_finish_candidates.py`: the give_up record of plan candidates offered
  and not tried this round, by candidate id on RoboTwin/RoboDojo and by
  approximate `move_pose` pose match on LIBERO, and whether the reason names
  them (an id is untried only when no offering of it was tried); never a
  refusal.
- `test_policy_finish_rounds.py`: Synthetic sessions on a test-local v2 surface:
  give_up is a valid policy-lifecycle sample, an admitted abort ends only its
  round as an invalid infrastructure sample with no verdict, and the summary
  reaches the decision event, round records, `result.json` and
  `evaluation.json`, bound there to the archived finish record; a finish on a
  model-call overrun turn is still recorded; a RoboDojo session with an
  admitted abort round completes; a surface without the capability keeps its
  records.
- `test_pose_primitive_reach.py`: on the live RoboDojo surface `move_to`,
  `rotate_wrist` and `rotate_pitch` carry the `move_pose` reach reference
  (`pose_primitive_reach_along_request_v1`) on their own routes, submit
  unchanged, and RoboTwin surfaces do not declare it.
- `test_reach_nearest_orientation.py`: on the live RoboDojo surface an
  unsolved `move_pose`/`move_to`/rotation target's `reach_along_request` carries
  `nearest_reachable_orientation` (`reach_nearest_orientation_v1`): each turn
  tilts the approach by its angle about an axis perpendicular to it and holds
  the jaw centre, one 12-query request per rotation stops at the first
  reachable one (the recorded g012 drawer-top target at 45 degrees), the larger
  joint margin wins, the action is unchanged and a failed sweep leaves the rest
  of the reference intact; against the v2 reach check a turn refused only as
  `inter_arm_collision_path` is never reachable, also through the RoboDojo
  backend's own reach check.
- `test_observed_clearance.py`: the `observed_clearance_reference_v1`
  measurement on ray-cast synthetic depth -- footprint lowest point at the
  target, a rim within `radius_m` reported with its position and support, robot
  points of the measured jaws excluded, `unknown` for missing depth or an
  unobserved footprint, measurement-only wording, bounded full-resolution cost.
- `test_observed_clearance_runtime.py`: on the live RoboDojo surface every
  `move_pose`, `move_to`, `rotate_wrist` and `rotate_pitch` target carries
  `observed_clearance` from hash-checked public depth, the bound descriptor and
  measured state; the held extent comes from the policy's own localization of
  the target it named at close and only while the jaws measure as holding; a
  failing reference never blocks the move; the live surface declares v2 (whose
  X5 jaw keeps v1's loader) and binds ARX descriptor v5.
- `test_pose_world_axes.py`: on the live RoboDojo surface
  (`pose_orientation_from_world_axes_v1`) the `move_pose` `world_axes` mode
  round-trips world approach and jaw-line directions through the compiled
  quaternion (approach `[0,0,-1]` with several jaw lines), refuses near-zero
  and near-parallel directions by field and bound before submission, and every
  `move_pose`/`move_to`/`rotate_*` target echoes `target_axes_world` from the
  quaternion it submits; the three EXP11A g012 upward quaternions stay legal
  and echo approach `[0,0,1]`; the released RoboTwin v1 surface does not
  declare it.
- `test_reach_reference_v2.py`, `test_observed_clearance_v2.py`,
  `test_closed_skills_ports_surfaces.py`, `test_robotwin_reach_inter_arm.py`:
  the integration-0059 ports of the RoboDojo closed-skills features onto the
  RoboTwin and LIBERO defaults (`robotwin_toolchain_b2_closed_noservo_v2`,
  `libero_placement_direct_overhead_v1`) and their composed successors, which
  declare them. M1/M3: the v2 reach pre-check names the backend's `ik_model`,
  `matches_move_pose_admission`, measured disagreement and claim (RoboTwin's
  single-seed DLS, about 5% false refusals and 4% false admissions), through
  the real RoboTwin reach-check path, and the nearest-orientation sweep holds
  the grip-centre TCP (zero jaw offset). M4: `observed_clearance_reference_v2`
  jaw models for ALOHA and Panda from their own public facts. M5/V1/V2: the
  released capabilities compose on both defaults with each embodiment's axes
  and camera names, the LIBERO help names the profiled card size, every port
  component is bound by a registered surface, and the real dispatcher delivers
  every port through result-schema admission.
  M6: the RoboTwin other-arm screen for reach-check solutions
  (endpoint-only by default, straight joint path opt-in), with
  `test_robotwin_reach_inter_arm_wiring.py` holding its activation: the
  RunSpec derives the runtime fields from the surface capabilities, the
  factory/worker/session carry them, the adapter screens every solved row and
  names the screened `ik_model` with null measured disagreement, the screened
  help replaces M1's sentence, the binder widens the `grasp_plan` reason enum,
  and every registered screen surface is checked for all three.
- `test_robodojo_g013_surface.py`: the combined EXP11A g013 Dojo surface
  (revision 17) declares all four g013 capabilities after revision 13's 78;
  every `move_pose`/`move_to`/`rotate_*` target schema carries
  `reach_along_request` (unsolved form with `nearest_reachable_orientation`),
  `observed_clearance` and `target_axes_world`; `finish_episode` keeps the
  abort-evidence contract; the three `move_pose` help sentences appear once each
  in revision order; the binders run in one deterministic order; and the Tool
  Runtime reaches the per-target annotations only through
  `pose_target_annotations.merged`, which refuses to overwrite a field.
- `test_x5_topdown.py`: the ARX X5 top-down grasp family fused with
  Contact-GraspNet on the live RoboDojo surface -- support-aware contact
  height on synthetic clouds (flat 15 mm object, 34 mm cube, tall object, no
  visible support), nominal-aperture and link6 workspace-edge rejections,
  link6 compilation, the unchanged ALOHA rule and one IK-ordered pool.
- `test_public_field_denylist.py`: policy-visible field names never collide with
  the agent-trace privileged-field denylist. It runs the verifier's own
  `_forbidden_path` over current public robot facts on every surface that
  projects the controlled point (so `jaw_center_pixels` on Dojo), over RoboDojo
  sparse motion, jaw and recovery receipt envelopes as Native delivers them, and
  over every declared result and input field outside the verifier's value-checked
  exemptions; the retired `jaw_contact_center_pixels` still fails.
- `test_robodojo_staged_contact.py`: the opt-in staged Dojo contact approach.
  With `contact_approach_route` a large `contact_approach` runs a `precontact`
  global segment then the local step, discloses the per-arm pre-contact pose and
  hand-off, reports a pre-contact IK stop without writes and a post-write local
  refusal as `not_reached`; free-space, precision and small approaches ignore it;
  without the route the released refusal holds, and a surface lacking
  `robodojo_staged_contact_approach_v1` refuses the field at RunSpec resolution.
- `test_robodojo_motion_metrics.py`: U1 v2 L0 metric definitions
  (`robodojo-motion-metrics.v2`) on synthetic native journals written by
  `tests/robodojo_journal_fixture.py`. Each metric's numerator and denominator
  and the edge of its definition: speed samples across a tick boundary only
  within a round, the closed-command and holding-mode cohorts, commanded
  speed from applied targets, moving ticks, out-of-step over constant-target
  segments, sag against the receipt goal height, held-then-lost by jaw floor
  and by the object at the jaw contact centre, arrival over every pose command,
  native actions per command, global/local segment and round, three still ticks
  at a timeout, tracking error against the endpoint or planned waypoints, and
  the loader's refusals. Commanded speed counts only a tick's own steps (a
  boundary snap back to the measured joints is not a violation). It reads no
  recorded run.
- `test_robodojo_motion_replay.py`: the `control` replay worker driven by a
  fake session whose physics reaches each recorded target and whose gripper
  command follows its directives (directive restoration, zero divergence,
  first divergent tick, early terminal, object-pose read errors, source and
  fingerprint refusal, journals in the original layout) and the host
  launcher's arguments, output containment, identity, kernel admission, lease
  release, process-group kill on timeout and on SIGINT (short `sleep`
  subprocesses), held target-arm seeding, divergence comparisons with cause
  hints and report-only `replay`, `metrics-after-replay` and `determinism`
  reports. Kernel admission is tested on synthetic sources, and the real
  executor-only table is pinned against a scan of `session`, `bridge`,
  `round_lifecycle`, `feedback` and `observation`. It starts no Isaac and claims
  no real lease.
- `test_motion_parameter_delivery.py`: executed-stage residual/threshold
  equality, strictness, unavailable evidence and actual bound workspace help.
- `test_official_verdict_guard.py`: the round-boundary guard between the
  persisted official verdict and the policy-visible `check_task` snapshot. A
  snapshot that said `task_success: true` against a recorded
  `official_success: false` ends the round as an infrastructure fault with a
  divergence record and a `runtime_fault` event; agreeing records stay a
  behavioral sample; a late or shadowed latch is republished into the checker
  rather than nulled. Uses Synthetic sessions and a scripted provider, so it
  checks the coordinator wiring, not simulator predicates.
- `test_packaging.py`: `pyproject.toml` discovers every `ora` and
  `viewer` package, names only existing experiment packages, and lists every
  tracked runtime data file under `ora` as package data, so an
  installed wheel can start `ora`.
- `test_embodiment_pin_activation.py`: maintained baselines resolve to
  descriptors that audit path-only; every context profile resolves; every
  descriptor is composed or code-registered and audits path-only, and no
  geometry-only descriptor remains.
- `test_frozen_composition.py`: the RunSpec resolver refuses a RoboTwin
  control profile older than the oldest composed and the policy values removed
  from its accepted sets.
- `test_provider_recovery_deadline.py`: same-credential deadline and run-budget
  propagation for both protocols, local HTTP stalls/trickles, partial-response
  recovery, child reaping, redaction and single-shot preflight; the real
  transport worker sends an OpenAI-compatible `X-Session-ID` only when bound;
  a loopback gateway's HTTP 408 naming an upstream stream disconnect is a
  retried stream break within the same budget, any other 408 stays terminal.

These checks make no behavioral-success, external Provider or full
simulator-motion qualification claim.

The root suite is organized by durable ownership:

`test_native_baseline.py` protects the complete Native robot default selection
(direct_v4 on LIBERO, RoboTwin and RoboDojo), canonical tool order and
fail-closed live prerequisites without calling a provider or simulator.

`test_baseline_explain.py` holds `ora baselines --explain` to the code
it describes: every tool of every default surface in surface order, each handler
resolving to a real callable at the stated line, a proposal route for every
state-changing tool, the maintained baselines with their selectors, the
`--environment` filter and round-tripping `--json`. It also keeps `baselines`
and `baselines --json` byte-identical and runs the command against an isolated
workspace to show it writes no file.

`test_policy_lock.py` checks the generated `policy.lock` on throwaway Git
repositories: exactly the tracked covered assets in sorted `sha256sum` format,
independent of creation order, untracked scratch excluded; `lock --check`
naming a modified, an added and a deleted asset and refusing an absent,
malformed or unsorted lock. On the checkout it requires every coverage entry to
name its loader and match tracked files, every path a loader reads by path
(catalog constants, task-contract records, capability-manifest bundles,
experiment prompt layers) to be covered, and every path a
covered asset names to be covered or declared unloaded. It never rewrites the
committed lock; `lock --check` in CI holds that.
`test_model_visible_identity.py` checks the RunPlan `policy_identity` block
(`policy_lock_sha256`, `model_visible_sha256`) beside an unchanged
`{commit, dirty}` source, persisted-plan validation with and without the
block, that execution placement leaves the digest unchanged, and, on a scratch
copy of `ora/` with its own lock,
that pin-only edits (a manifest provenance note, a component moved to a new
path, an external descriptor provenance pin, which no other asset pins since
L3) leave
`model_visible_sha256` unchanged while a tool description, a prompt layer or
an embodiment card moves it.
`test_policy_path_only.py` is the stage-2 L2/L3 ratchet: every composable asset
(registered ToolSurfaces, unretired context profiles, selectable baselines,
maintained task-contract registries, the RoboDojo layout catalog and
appendices, `tests/data/default-expectations.json`, and the descriptors and wording
reviews they bind plus the RoboDojo activation descriptor) names other assets
by path only, a composed descriptor names its repository sources by path and
a composed wording review pins only `card_sha256`, every inline hash left is
classified as external, human-review, generated card manifest or frozen,
frozen assets (retired descriptors included) keep their pins, the compiled
card manifests equal their regeneration, the loaders refuse inline pins,
references that escape the tree or are not in `policy.lock` and a review that
pins descriptor bytes, `lock --check` catches an edit made after the lock, and
`lock --strip-pins` keeps formatting, refuses a stale pin a loader verified and
strips a geometry-only descriptor's unverified ones. Test fixtures that build a scratch
tree give it a lock with `tests/locked_tree.py`, and
`tests/surface_copies.py` rebases copied manifests and computes the
pin-insensitive `content_sha256` that replaced composition-hash goldens.

`test_motion_transaction_v1.py` covers post-close retention eligibility and
TCP rotation compensation across LIBERO right and both RoboTwin arms, including
intervening opening/reset, insufficient motion and corrupt historical evidence.

| Area | Representative modules | Responsibility |
|---|---|---|
| Control and qualification | `test_control_plane*.py`, `test_runtime_v1*.py`, `test_artifact_provenance.py`, `test_robotwin_prepared_grasp_qualification.py` | Planning, exact CLI, source gates, leases, evidence, sealing and repository hygiene; issued Artifact provenance, immutable historical review and bounded RoboTwin prepared-grasp evidence checks. |
| Policy runtimes | `test_native_harness*.py` | Native and managed lifecycle, provider transport, tool pairing and wire integrity; real Codex through a local fake provider/MCP, exact namespace inventory, child-schema/compaction-alias rejection and flat/mixed legacy trace compatibility. |
| Provider attempt recovery | `test_native_harness.py`, `test_native_harness_anthropic.py`, `test_native_harness_openai.py` | Bounded Anthropic stream recovery, credential redaction, request/artifact/attempt joins, OpenAI partial-stream recovery and TLS/final-error boundaries, including bounded pre-response bad-record-MAC retry. |
| Operator stop during a policy turn | `test_operator_stop_cancellation.py` | An explicit stop that lands during projection, a provider stream or a completed reply ends the session with operator authority: no later request, retry or tool dispatch, no executed action and no valid behavioral sample; a closed-client error from any policy runtime is not a `trusted_fault`, while the identical failure without a stop stays infrastructure. Scripted providers only. |
| Post-action observation record | `test_stagnation_measurement_boundary.py` | One measured head-image distance and no judgement: why the retired guard could not be repaired under the sparse motion codec and receipt, that an identical head image is not evidence that nothing moved, that no notice selects a stalled-progress instruction, that the collapsed `<episode_index>` carries no record, and the retention keyframe condition. |
| Context | `test_lossless_context_runtime*.py`, `test_rollout_context_runtime.py` | Recoverable projection, semantic fallback and persistent rounds; a no-action ActionStep closes on the same observation presented again. |
| Robot actions and tools | `test_pose_frame_contract.py`, `test_motion_runtime.py`, `test_grounding_grasp_runtime.py`, `test_tool_help_contract.py`, `test_tool_description_consistency.py`, `test_published_limits.py` | Canonical action compilation, controller evidence, proposals and model-visible contracts; on the current surfaces, help that matches its handler (EXP11B description consistency); every enforced number in tool help equal to its owner through one table (`policy/tool_surfaces/published_limits_v1/limits.json`), with the current surfaces' known mismatches as strict expected failures and the unregistered successor texts (0059 N1/N2, with their refusal sentences) free of them. |
| Holding and placement | `test_placement_geometry.py`, `test_placement_argument_shapes.py`, `test_robotwin_placement_adapter.py` | Public partial-geometry proposals, bounded front-cavity headroom with all observed points retained, actual public-state/receipt stage joins, explicit-arm sparse proposals and unchanged baseline transports. Both `place.*` argument spellings normalize to one selection with the bound schema deciding which is legal, the two active default surfaces publish the flat one and are checked for a surviving union after composition and after provider projection, and the conditional-opening and unequal-length rejections name their field rather than reproducing a union. CPU tests do not certify real GPU planning or physical placement. |
| Closed manipulation skills | `test_manipulation_skill_ledger.py`, `test_manipulation_skill_driver.py`, `test_closed_skill_surface_contract.py`, `test_closed_skill_help_drift_repair.py`, `test_skill_sequence_action.py`, `test_skill_sequence_executor.py`, `test_skill_receipt_delivery.py`, `test_target_localization.py`, `test_verifier_retirement_capture.py`, `test_placement_rejection_ledger.py`, `test_grasp_candidate_palette.py` | One policy decision per skill: a grasp never yields, a place yields once before the release. Staged receipts measured against the delivery ceilings with the threshold's own serializer, per-source image binding and a mandatory bounded rationale. The canonical localization contract the retired verifiers' measurements moved into, the place.plan rejection ledger, and palettes pinned to the counts they bound. Trusted terminal signals produce irreversible `terminated` receipts, release skill ownership and prevent every subsequent stage while preserving measured controller evidence. CPU tests do not certify physical grasping, placement or retention. |
| Composed tools | `test_starlark_runtime.py`, `test_composed_tool_contract.py`, `test_composed_tool_execution.py`, `test_composed_grasp_equivalence.py`, `test_composed_tool_live_path.py`, `test_composed_hold_and_questions.py`, `test_composed_linking.py` | The Starlark language boundary; reads, nested tools, value inputs and workspace tools bound by absolute path; headers, the SDK, the tunable table and the compiler; the executor's one-decision sequences with bounded retries, budgets and closed-skill receipt conventions; layer-1 static checks and the layer-2 contract of every library tool; layer-3 equivalence of grasp_execute c1 with the native prepared grasp; surface binding, the composed action, dispatch and delivery; check.hold and the question bank. Synthetic backends and fakes only: nothing here certifies a composed tool on a robot. |
| Harness changes | `test_harness_change.py`, `test_composed_task_tools.py`, `test_benchmark_spec.py` | Composed version overrides and task tools added on one task, harness-change records in the default expansion and in a built RunSpec (surface, service binding, recorded selection). Offline: nothing here runs a candidate or measures a gain. |
| Holding and placement | `test_placement_geometry.py`, `test_placement_argument_shapes.py`, `test_robotwin_placement_adapter.py` | Public partial-geometry proposals, bounded front-cavity headroom with all observed points retained, actual public-state/receipt stage joins, explicit-arm sparse proposals and unchanged baseline transports. Both `place.*` argument spellings normalize to one selection with the bound schema deciding which is legal, and the conditional-opening and unequal-length rejections name their field rather than reproducing a union. CPU tests do not certify real GPU planning or physical placement. |
| Placement depth uncertainty | `test_placement_depth.py`, `test_placement_depth_tool_surface.py` | Local discontinuity evidence, preserved originals, retained-point projection, genuine far-tip ambiguity, conditional fit labels, immutable v1 surfaces and rehashed/recomputed v2 records. CPU counterexamples do not substitute for live evidence. |
| Post-action image delivery | `test_servo_observation_delivery.py` | Fresh default and selected-wrist views after an action, explicit missing-wrist disclosure, fail-closed stale images, verified image bytes and immutable multi-image runtime inputs (`agent/context/action_views.py`). |
| Historical Servo replay | `test_servo_historical_replay.py` | Historical replay rejects recorded Servo actions before simulator construction while legacy `servo_delta` payloads keep their bytes. |
| Plan A v7 feedback | `test_tool_boundary_recovery.py`, `test_environment_rpc_refusal.py`, `test_correctness_feedback.py`, `test_robotwin_contact_recovery.py`, `test_robotwin_ik_feedback.py` | Typed pre-execution refusals, truthful physical causes, fixed-start escape and frozen-target stalls, measured IK rejection counts and valid variant references. |
| Prompt authoring | `test_prompt_bundle_authoring.py` | Modular prompt-bundle schema, layer ownership, embodiment format and offline audit gate. |
| LIBERO execution correctness | `test_libero_correctness_repair.py` | First termination, planner/source binding and worker lifecycle; profile-v2 joint/TCP settling with fixed budget, precision-only force stops through sparse dispatch, sensor-site force conversion, executed-command drift and unchanged frozen-v1 behavior. Waypoint interruption and unfinished endpoint convergence remain distinct from measured settling through raw and model-visible feedback. Controller-boundary fakes do not qualify simulator dynamics. |
| Ownership audit | `test_ownership_audit.py` | Deterministic behavioral-ownership report: selection versus delegation counts, hidden planner/controller work and re-observation boundaries from append-only run evidence. |
| Vendored planner | `test_vendored_install.py` | Deterministic vendored RoboTwin planner install identity, MIT attribution and v10 pin refresh; temporary bare computation preserves external Git objects and frozen v1–v9 profile bytes and registry entries. |
| Motion transaction | `test_motion_transaction_v1.py`, `test_profile_capabilities.py` | Backend-neutral motion execution, strategy routing, plan validation and the capability registry. |
| RoboTwin motion v2 | `test_robotwin_motion_v2*.py`, `test_robotwin_planning_v4.py` | Dual-arm sparse control, grasp compiler filtering, vendored planner candidate contracts. |
| Environments | `test_robotwin_dual_arm_runtime.py`, `test_libero_pro_robustness.py` | Backend isolation and benchmark qualification. The RoboTwin file also pins the executable/deferred partition, oracle-qualification task/asset registry and the reset-time restoration of evaluator state that upstream assigns only inside its expert script. |
| Original capture and recording | `test_execution_capture.py`, `test_controller_replay_journal.py`, `test_viewer_visual_fidelity.py` | Shared bounded capture/journal, original pixels and default Viewer fidelity. |
| Evidence and presentation | `test_experiment_view.py`, `test_viewer_session_rounds.py`, `test_viewer_process.py`, `test_viewer_pricing.py`, `test_control_audit.py` | Source-bound read models, round semantics, ordered tool/image chapters, read-only media routes, CPU video exports, pricing and control-health evidence. |
| Embodiment-declared geometry | `test_alignment_pose_synthesis.py` | Approach-pose synthesis obeys each embodiment's declared `local_axes` rather than a fixed rotation column order, resolved through the one reader that owns axis values. The two maintained embodiments disagree -- aloha-agilex approaches along local +X, franka-panda along +Z -- and the wrong assumption points the jaw-separation line at the target while reporting no error. |
| Alignment acceptance band | `test_alignment_acceptance_band.py` | `align_plan` solves at the local servo controller's convergence bound, 0.003 m / 0.03 rad, while the RoboTwin backend's own 128-seed IK accepts 0.005 m / 0.05 rad and the same profile's `arrival_check` calls that distance arrived. A pose whose best configuration lands in between is refused with a `position_error_m` inside the bound everything downstream uses. The band is pinned from both sides with the exact shipped numbers so that widening either threshold is a deliberate act that revisits the claim boundary; the restated planner thresholds are checked against the planner source; and two structural properties are recorded -- `joint_limit` is reported only when the limits actually bind, and the solver's signature admits no collision model of any kind, so every collision refusal the backend makes is one it could not have made. |
| Reachability acceptance | `test_alignment_acceptance_band.py` | `align_plan` accepts on `arrival_check`, the bound that decides whether a pose counts as reached, not on `local_cartesian`, the local servo controller's convergence bound. The two disagree by 2 mm and 0.02 rad, and reading the servo bound refused every pose whose best solution landed in that band while the backend planner would have solved it. Pinned against the vendored planner's own literals from the other side. |
| Jaw-symmetric member | `test_grasp_jaw_symmetry.py` | On the ARX X5 a parallel-jaw grasp turned 180 degrees about the approach axis differs only in joint6, so the live RoboDojo `grasp_plan` publishes the reachable member with the larger joint6 margin (EXP11A g007: the compiled member closed 0.14 rad from the joint6 stop and every carry move after it was refused). The joint6 margin is the reach check's minimum joint-limit margin whenever the measured joints bound joints 1-5 inside the X5 limit table; RoboTwin keeps the released selection. |
| move_pose IK reference | `test_move_pose_reach.py` | Before submission `move_pose` asks the backend's read-only reach check about each compiled target. On the ARX X5 an empty-jaw absolute move whose requested turn solved within pi/4 rad of the joint6 stop, or not at all, is submitted as its 180-degree jaw turn when that has more margin (EXP11A g008: joint6 -2.465/-3.007 from home, then a refused 5.5 rad branch jump); an unsolved target reports how far along its straight path the same route solves. Execution and refusals are unchanged, and an unavailable IK reference never blocks the move. |
| Reach check | `test_reach_check.py` | The internal (undeclared) reach check answers exact poses with each backend's read-only IK (RoboDojo: the SDK IK on the move_pose route the interaction selects, chained starts; RoboTwin: the damped-least-squares solve, declared weaker than move_pose), and `grasp_plan` checks its whole compiled pool before the display cap, ordering reachable, then within 30 degrees of vertical, then backend order. |
| LIBERO reach check | `test_libero_reach_check.py`, `test_libero_probe_ik_feedback.py`, `test_libero_reach_ports.py` | LIBERO answers the internal reach check with one CuRobo IK solve per grip-site pose (the IK solver of the planner the global `move_pose` segment uses; joint limits and self-collision, no world model; `matches_move_pose_admission: false`), a v2 motion probe keeps the IK seed counts of each variant's own plan, and the LIBERO default generation runs the pooled `grasp_plan` selection and the v2 reach pre-checks on it. Offline: a CPU CuRobo stand-in and the real session over a CPU plant; no measured disagreement exists yet. |
| Gripper codec parity | `test_set_gripper_codec_parity.py` | `set_gripper` and the pair it replaces emit the same action codec on each active surface; every kept surface advertising `set_gripper` declares its arm topology, so it always takes the shared v3 path. The close-intent `target_ref` survives the replacement, labelled as annotation and never evidence. |
| Grounding palette | `test_grounding_candidate_palette.py` | One colour table bounds how many grounding candidates can be drawn, and both renderings and both admission points derive the ceiling from it. Two inline three-colour tuples used to decide it silently: the overview zipped non-strictly and drew nothing past the third, the enlarged preview indexed modulo three and painted the fourth in the first's colour. |
| Canonical tool names | `test_canonical_tool_names.py` | Provider function names are legal, and the dispatch boundary resolves the dotted spelling frozen surfaces advertise onto the canonical one a handler is registered under. Normalising at the boundary rather than registering each handler twice keeps a later tool from being reachable under one spelling and missing under the other. It also simulates the rename against each live surface and requires every name-indexed consumer downstream of dispatch to still resolve, which is the precondition for renaming the manifests. |
| Package integrity | `test_harness_wrapper_refactor.py`, `test_module_size_ratchet.py` | Canonical imports and retired façades. The size ratchet records every module already past 800 lines and lets an oversized one grow only by a wiring allowance, so new work lands in its own module; a module that shrinks below the threshold loses its exemption and cannot silently regain it. The host-layer import boundary in `test_harness_wrapper_refactor.py` covers every module under `ora/orchestrator/` and `viewer/`. |
| Unbound global reads | `test_undefined_globals.py` | Every global name a `ora` function reads is bound by its module (symtable, no imports executed). A merge or extraction that leaves a call to a deleted helper, or moves code away from the function that defined a local, fails here instead of as a NameError on an untested path (three such were found after refactor stage 1 batch 1). |
| Prompt bundle quantities | `test_bundle_quantities_ratchet.py` | A live bundle states a physical quantity as a number with a unit. Fails when a vague magnitude word describes one and no number appears in the same sentence. Scans only live bundles, derived from the baseline catalog minus frozen ones, because several frozen manuals are regenerated byte-for-byte by `render_geometry_card` and demanding edits there demands rewriting sealed evidence. Every recorded violation carries a mandatory reason. |

The manifest is authoritative when a module spans more than one area.

Native transport regressions exercise both provider protocols with injected
pre-response TLS EOFs, certificate failures and partial streams; they check
bounded retries and request-bound failure telemetry without live requests.
`test_tool_boundary_recovery.py` checks correctable Artifact inputs, complete
provider tool schemas, public recovery fields and refusal evidence. Its
malformed-JSON regression runs the Native lossless turn loop through the next
request, including signed OpenAI continuation via mock transport, without a
live provider or simulator.
`test_tool_error_feedback.py` checks the shared ToolError envelope
(`retryable` unchanged, `retry_requires_changed_arguments` on argument
refusals), the name-only `require_*` argument rules on EXP11A g016's
`geometry_query` call, and the per-session repeated-identical-failure count in
the Native loop (appended to the new result only; every request a strict
extension of the last) and the managed MCP façade, and `report_issue`
refusals that name `message` and the non-whitespace rule without echoing it.
`test_compute_grammar.py` checks the generated compute grammar and its refusals.
`test_infrastructure_classification.py` checks the one infrastructure table behind
the envelope flag and the abort gate.
The 0059 refusal-envelope modules check the open-bugs triage fixes, each without
a simulator or provider: `test_target_and_grasp_refusals.py` (target refs,
typed model-service failures, withheld `grasp_execute` candidates, rationale
order, candidate geometry, the plan-time lift check and one meaning for
`stale_observation`), `test_observation_and_perception_refusals.py` (current and
out-of-range saved steps, history overlays, pixel/region/world-point and
`mark_point` refusals, public-facts faults), `test_motion_refusal_envelopes.py`
(`move_pose` examples against every enhanced surface's delivered schema and the
compiler, public angle/quaternion bounds, primitive naming, the workspace field,
undeclared tools) and `test_argument_envelopes_before_dispatch.py`
(`artifact_read`/`observe(ref)` argument refusals and the pointer bound,
integral-float coercion in both gateways, the retryable abort refusal).
`test_mutation_compensation.py` checks that a mutation failing after submission
leaves no action behind and is logged, while a pending action survives a refusal.
`test_schema_refusals.py` checks the structured generic input-schema refusal
(field path, violated keyword and bound, received shape) and the input keyword lint.
`test_openai_tool_strictness.py` checks the profile-gated `"strict": false`
statement on the OpenAI-compatible wire and that the recorded wire equals the
sent body.
`test_null_optional_arguments.py` checks `optional_argument_null_as_omitted_v1`
on the three default instances' surfaces and the EXP11A/EXP11B Dojo surface:
nullable optional arguments on every protocol, null dropped by both gateways
before dispatch, placeholders still refused and non-declaring surfaces
unchanged.
RoboTwin motion regressions distinguish certified pre-execution refusals from
partial execution.
Retention regressions also check phase-appropriate recovery instructions
and explicit fit/overlap failure reasons without changing numeric thresholds.

The unified H candidate extends LIBERO admission to 22 cases, including
post-close binding and two actual one-use transports; RoboTwin retains its
22/23-case sets. New correctness modules cover first termination, candidate
admission, geometry inliers, full-reference retention and single/dual-arm
holding. Credential tests qualify each key independently and verify private
launch handoff without exposing secrets. CPU qualification fixtures remain
control-flow tests, not live simulator evidence. The older counts in historical
records are not relabelled as unified-v3 passes.

## Experiment-derived tests

Modules that began as experiment checks use product names and live at
`tests/`; their contract belongs to the shared Runtime. Examples include
lossless Context from EXP05C, hybrid motion from EXP05A, grounding/grasp
planning from EXP05B and persistent rounds from EXP03E.

## Determinism and authority

- Tests never call a live provider, simulator or physical robot.
- Runtime writes use pytest temporary workspaces, never operator evidence.
- Tests may validate frozen source metadata but cannot upgrade historical
  evidence or invent a task verdict.
- Only the environment's official boundary supplies Boolean success.

## Adding a test

1. add the module as `tests/test_<name>.py`;
2. add its row (path and a responsibility-level contract, not one row per test
   function) to `test-manifest.json` and describe it in this README;
   `tests/test_control_plane_cli.py::test_documented_script_and_test_module_inventory_is_current`
   checks that the manifest equals the set of `tests/test_*.py` modules; and
3. run it with `uv run --locked pytest -q tests/test_<name>.py`, then the
   complete locked suite.

Renaming or deleting a module updates the same manifest row and README entry.

Operator command ownership is documented separately in
[`scripts/README.md`](../scripts/README.md).

The unified EXP08H contract regressions additionally exercise zero/nonzero
RoboTwin and LIBERO grounding results through production dispatch and Native
projection, recoverable holding errors, all 21 advertised tools, fixed eight
cell identities including seed 884, and the four-whole-session queue ceiling.

`test_retention_recovery.py` covers the opt-in v5 retention recovery contract:
border evidence, occlusion and conflicting camera counterexamples, exact source
joins, current-round historical observe/segment, Native historical projection,
and probe renewal only after new consistent evidence. These CPU fixtures do
not certify object identity, attachment or real-task success.

`test_perception_choice.py` protects the advisory-retention capability through
single/dual-arm dispatch: unknown measurements cannot lock motion, repeat close
needs no segmentation, source checks remain enforced, and absolute/enhanced
contact targets compile identically.

`test_batch_segmentation.py` checks one image preparation with independent
point prompts, a per-point mask reserve under one shared delivery cap, stable
deduplication, disclosed component filtering, partial failures, source-bound
target/cache integrity, history isolation and the loopback batch client wire.
The declared capability alone selects the batch contract. Under
`grounding_candidate_depth_profile_v1` the v2 batch identity gains
`depth_profile` while target refs, the ledger, the overview and every published
preview key stay as released, and each 1024x800 card keeps the released pixels
above its depth strip. These CPU fixtures neither load SAM weights
nor measure real semantic candidate coverage.

`test_libero_camera_resolution.py` records the LIBERO session's two independent
`256` literals -- the construction parameters and the intrinsics -- because
changing only the first would leave every deprojected metre wrong while the
matrix still looked plausible, and keeps the render guard that refuses a
corrupted offscreen frame.
`test_libero_mesa_lanes.py` keeps the software-rendered (Mesa) cluster LIBERO
lanes, their closed worker environment and their preflight device selection.

`test_grounding_candidate_bandwidth.py` checks what bounds one `vision.ground`
result. Two characterization tests pin the renderer: the overview draws a fourth
candidate, and the preview helper's rank-3/rank-0 colour collision is recorded.
The rest derive the ceiling of 32 from the shipped palette
(`grounding_palette`), measure the worst-case result against the 50,000-byte
delivery threshold in both directions, and require the live preview loop to be
sliced by the preview cap. The depth-profile card
publishes the same preview record keys at the same byte length, so the measured
preview-record budget still holds. These are CPU fixtures:
they neither load a grounding model nor measure real candidate quality.

`test_mark_point_batch.py` checks the bounded eight-pixel `mark_point` batch:
the ceiling is derived from the overlay colour table rather than chosen, the
largest result the schema permits is measured against the 50,000-byte delivery
threshold, eight marks cost one image slot, and attribution is carried by each
entry's colour and `source_index` rather than by image position, which
individual block eviction would break. One unmeasurable pixel fails only its own
entry, and every entry reports its depth evidence on success and on failure. The
same module drives the older parallel-call overlay aggregation end to end, which
is the evidence that it is still reachable and was therefore not deleted. It
also holds the surface boundary: exactly the two active default profiles declare
`mark_point_batch_v1`, each binds the published input and result schemas by
object equality rather than retyping them, and neither publishes a point-cloud
view, because the batch refuses one and a surface cannot publish both call
forms. These are synthetic CPU fixtures: they measure result size and
attribution, not real scene geometry.

`test_perception_feedback.py` resolves both final H v6 profiles, checks
unconfirmed-stage geometry preservation, signed retention margins and paired
source images, candidate/image mapping and historical observation projection.
It also exercises actual Native tool-result assembly into both Provider wire
formats and managed MCP, proving delivery of global grasp evidence, and that the
grasp-query manifest records that delivery as `policy_visible: true`. Through
the real dispatcher the live RoboDojo surface delivers the review with its
observed-extent cell while the B2 Dojo arm keeps the released cell. These
checks provide no live behavioral success claim.

The EXP08H placement contracts also verify separate v6 failure-cohort admission,
rejection of substituted v5 treatment, and batch-point Provider advertisement.


`test_grasp_candidate_motion_probe.py` checks that direct-pose grasp plans on
the maintained LIBERO-Pro and RoboTwin defaults emit candidates without invoking
a probe; the probe and trajectory-handle contracts left with the last surface
exposing `motion.probe`.
`test_tool_runtime_dead_state.py` guards the same package against state nothing
consumes: an attribute assigned under `agent/tools` must be read somewhere in
`ora`, counting cross-module readers so that a value written by one
tool module and used by another is not mistaken for dead.
`test_declared_tool_handler_join.py` joins the other direction, across every
profile in the catalog: a tool a surface advertises must resolve to a handler in
the runtime that serves it. The loader checks a manifest only against itself, so
without this a retired handler leaves selectable profiles dispatching to
`internal_tool_error` in a live episode and no offline test notices.
`test_geometry_query_publication.py` takes the case that join exposed to its
other end. `_geometry` has served forward projection, back-projection and region
depth all along; only the dispatch and the declaration were missing, and sealed
traces show 51 of 177 runs deriving `u = cx + fx * x / z` by hand instead. The
module checks what the two default surfaces now hand a policy -- the snake_case
name, each backend's own camera enum, read-only by absence from both effect
lists, and no `oneOf` reaching the provider, because `gateway.py` can only
flatten a root union whose branches carry `properties` and this one carries
`required` alone. It then drives all three result branches through the real
handler on the synthetic camera, because the published result schema closes
what the released one left as a bare object and a too-strict result schema
fails on a live call rather than offline. Each of those results is then put
through `causal_provider_value` with its declared property set, because
validated by the runtime and delivered to the model are different claims and
both filters between them are silent. The frozen surfaces keep
`geometry.query`: that spelling is what their sealed runs were shown, and the
sets are derived from `PROFILE_CATALOG` rather than quoted.
`test_correctness_feedback.py` covers the public contact-feedback exceptions in
state and trace validation, including rejection of wrong paths, wrong types,
nonfinite thresholds and nested privileged fields.

`test_backend_parity.py`, `test_native_baseline.py` and
`test_parameter_delivery_guards.py` check fresh consumers and reviewed default
settings; several modules read the expected defaults from
`tests/data/default-expectations.json`. Update those expectations only with a
reviewed default change; do not bless a regression by regenerating expectations.

Since 2026-10-04 (`native_full_explicit_motion_direct_v4`, integration 0059
unit A0) the default carries three instances: the v2 surfaces
`robotwin_toolchain_b2_closed_noservo_v2`,
`libero_placement_explicit_motion_direct_v2` and
`robodojo_placement_closed_skills_v2`, Context `native_slots_v4` and the
`current_v4` bundles; since integration 0084 the LIBERO instance is direct_v2's
successor `libero_placement_direct_overhead_v1` (the overhead footprint
localization). RoboDojo is Native-only (`instances.robodojo.consumers`).
`test_backend_parity.py` reads `ci/backend-parity.json`:
every capability and tool a default declares is classified; a generic one is on
every backend or excepted with a reason that names the exception; an exception
fails as stale once the backend declares the name or its missing hook exists;
and a default swap keeps a superset of the default it replaced, minus recorded
retirements; exception ids are `P<integration>-<letter><nn>`. `test_published_limits.py`
composes the registered surfaces of generations `current`, `retained` (LIBERO
direct_v2 since 0084) and `candidate`, and the frozen v1 ones, and checks that
the published-limits layers are the last two slots of each generated manifest.

Since 2026-09-29 (`native_full_explicit_motion_direct_v2`) the baseline tests verify
that a fresh RoboTwin consumer receives the B2 closed-skill surface with its
22 tools and 72 capabilities, `demo_randomized` scenes and a bundle that
declares that surface, and that `native_full_explicit_motion_direct_v1` stays
selectable as a frozen predecessor. Since 2026-10-02
(`native_full_explicit_motion_direct_v3`) they verify that the default differs
from the frozen `native_full_explicit_motion_direct_v2` only in Native Context
management v5 with stride 4, and that direct_v2 still resolves to v4. Since 2026-10-01 each default instance
in `tests/data/default-expectations.json` also pins the view settings no tool or capability name carries
(`view_config.controlled_point_projection: per_camera_pixel_v1`), so a default
swap cannot silently stop delivering the per-camera TCP pixel.
Parameter-delivery guards include the DS-R06 current-camera geometry ratchet:
public calibration is saved but not delivered by the Native observation renderer
on the two maintained backends. This recorded gap is not a passing repair claim.

`test_current_visual_delivery.py` covers complete current RGB and calibration
for LIBERO and RoboTwin, both Native wire protocols, managed dispatch and
attachment metrics, immutable image identities, explicit capacity failure and
current-to-historical read-only result projection. Depth remains queried. It
uses synthetic sensor fixtures only and makes no physical qualification claim.
It covers `agent/native/session.py` and `agent/native/actual_snapshot.py`, which
resolves the causal current observation's images, carried extras and measured
end-effector state.

Integration 0036 advanced those expectations to the reviewed v3 default and
retired the eight repaired Servo disclosure gaps; with bounded Servo removed
from this tree, the Servo guards of `test_parameter_delivery_guards.py` skip.
Controller values, frozen profiles and historical evidence are unchanged.
`test_servo_prompt_templates.py` checks that the fresh default templates
declare no Servo and that the frozen Servo-baseline templates are refused at
resolution and resolve without `servo`.


Integration 0037 selects all approved public-data changes with Observer off. `test_successor_default_delivery.py` independently checks the six fresh
consumers and negative missing-facts/old-card cases.
`test_derived_grasp_review.py` exercises real tool return and next-request serialization for both
Native protocols, including derived and physical image identity.

## Contract delivery guards (2026-09-11)

`tests/test_task_contract_delivery.py` asserts the delivery invariant for reviewed
RoboTwin completion contracts: registered plus source-pinned means actually sent,
and every withheld case names its own reason in the plan. A contract that exists,
is not delivered and leaves no record is the failure it exists to prevent.
Managed legacy contexts must carry the contract in their compiled episode_prefix;
an unused episode_task alias cannot bypass the omission guard.

## Held-object offset delivery (2026-09-11)

`tests/test_public_robot_facts.py` asserts the causal observation carries the
last held-object offset with the observation it was measured at and how many
observations have elapsed. The receipt half reaches only a policy that calls
place.plan; the sealed LIBERO-10 round two never did. The assessment already computed that offset and only
the sibling `geometry` key was projected, so the policy could not carry a prior
measurement forward and re-ran text grounding on every object after every move:
14 of 21 `vision.ground` calls in the sealed 2026-09-11 spatial run.

## Closed grasp and place skills (2026-09-11)

`tests/test_manipulation_skill_ledger.py` pins where each closed skill consults
the policy and where it does not. `grasp_execute` runs approach, close and
verify as one committed sequence and never yields; asking it to stop is refused
rather than ignored, because a dropped request leaves the policy waiting for a
view that never arrives. `place_execute` yields exactly once, immediately before
the release, and resuming returns the remaining stages as they stood rather than
reopening the candidate choice. Mutating stages fail the skill closed; verifying
stages do not, because failing to measure is not failing to act.

`tests/test_closed_skill_surface_contract.py` checks the closed-skill surfaces
this tree composes: no help names a tool the surface does not declare, driver
messages use the surface's own spelling, and the three live closed surfaces are
checked for the 2026-09-29 retirement of `place_plan`, `place_execute`,
`place_resume` and `grasp_verify_retention`: neither spelling reaches their
tool lists, help, bound prompt layers, environment profiles or task contracts,
and the handlers remain.

`tests/test_manipulation_skill_driver.py` covers the tool-runtime half: the
chosen candidate is resolved against the observation that produced it and
refused by name otherwise, one action is committed for the whole sequence with
its decision cost stated on the result, every stage target is derived once at
commit and replayed rather than re-derived against a world the sequence itself
moved, and the ledger is released twice over -- by the receipt on every exit the
executor has, and by the next call when the sequence died without one. Two place
assessments of the current observation that both publish `P0` are refused as
ambiguous rather than resolved by recency, because `place.plan` numbers its
candidates per call and picking the newer one would execute a candidate nobody
looked at.

`tests/test_skill_sequence_action.py` pins the action that makes "one call"
true. Everything a later stage does is authorized at the one moment the policy
was asked, so the action must be an ordered subset of its skill's declared
stages -- a reordering would release before it descended -- must carry the stop
stage exactly when the policy requested one, must resume only at the stage after
that stop, and must carry no geometry: stage poses belong to the tool runtime
that owns the candidate artifact and its observation binding.

`tests/test_skill_sequence_executor.py` covers the trusted stage loop that
runs between the submitted action and the receipt the policy reads on its next
turn. Two properties are load-bearing: a mutating stage that did not happen
stops the sequence, and the sequence reports however it ends -- refused,
terminal, interrupted, or with the backend raising -- because a skill that can
end without releasing its ledger is worse than the three explicit calls it
replaced. One coordinator execution stays one robot decision, and the receipt
carries the submitted `action_id` rather than a stage's, because the
AgentSession rejects a receipt that does not match the action it proposed.

`tests/test_skill_receipt_delivery.py` measures those receipts with the same
serializer the delivery threshold uses, so image payloads and `_runtime` are
excluded exactly as they are before the tool message is built. The worst case --
six stages at the per-stage measurement cap, a full verification block, the
held-object offset and four image bindings -- is 24,596 bytes against the 50,000
`eager_large_result_bytes` ceiling. Over budget, the oldest stages give up their
measurements first and each elision is recorded.

`tests/test_verifier_retirement_capture.py` asserts that every tool a
registered profile declares has a dispatch handler, and that the measurements
of the retired `grasp.verify_retention`, `place.verify` and `vision.offset`
verifiers stay importable. `test_target_localization.py`
holds the contract their measurements moved into, including the jaw-separation-
beside-extents evidence and its explicit unavailability on backends that publish
no jaw separation at all.

`tests/test_placement_rejection_ledger.py`
asserts an empty `place.plan` candidate list still carries the counts, hull
margin and observed lowest point that explain why it is empty.
`tests/test_grasp_candidate_palette.py` pins a palette to the candidate count it
silently bounds. CPU tests here do not certify physical grasping or placement.
## Restricted geometry evaluator (2026-09-11)

`tests/test_compute_tool.py` holds the evaluator's rejection,
isolation and leak contracts rather than any experiment's behaviour.

Two of its cases guard run validity rather than one call, and both assert
against the real scanners instead of a local reimplementation — a
reimplementation would keep passing after the scanners change:

- a block binding a name matching `evidence/trajectory.py::FORBIDDEN_FRAGMENTS`
  (`contact`, `segmentation`, `task_source`, `object_gt`, `actor_state`) must
  leave `_forbidden_path` clean, because a hit there invalidates the run and
  blocks sealing;
- no result may carry a JSON-parseable string, because
  `context/projection.py::_causal_projection_violations` opens strings that
  begin with `{` or `[`.

The suite is offline and starts one process — the evaluator's own warm worker,
which costs a single import and is reused across cases.

Its publication cases resolve the two live surfaces rather than a fixture. The
result schema is validated against the visible half of the result — `_runtime`
popped, the way `native/session.py` assembles the tool message — because a
published schema that nothing checks against the delivered shape is a guard that
cannot fail. The binding-refusal cases are parametrised on the field each
refusal must name: naming only the block would reproduce the union error the
entry form removes.


RoboDojo development modules (`test_robodojo_*`) cover backend lifecycle,
round reset, action/observation admission, source/qualification dependency
identities, isolated transport, public perception and task contracts. The backend
module additionally checks trusted v4/closed-skill admission, phase drift, replay
and exception cleanup without starting Isaac. Shared skill-executor and driver
checks cover existing maintained backends. Robot USD and SDK FK require a separate
bounded isolated smoke; unit tests do not certify loaded physics geometry.

`test_robodojo_layout_schedule.py` covers the per-round layout schedule of a
persistent RoboDojo session without Isaac: the schedule contract (repeats,
`_random_` files, other seeds, round 0 and session length refused), the
reviewed catalog and the hash-bound stack_bowls contract appendix (a
skip-if-absent check re-hashes them against the pinned Assets), worker-start
hashing, the put_bottles appendix's per-layout bottle-metadata disclosure and
its preflight re-hash, the reach footprint of the layout objects (scope,
asset-box scale and orientation, base distances, the worker's base mounts and,
skip-if-absent, the catalog's bowl annotation as its centre case), the four EXP11A g012 pools (classify_objects,
arrange_largest_number, organize_table, build_tower: default order and
explicit-only fallback, each appendix's layout-0 row equal to its record's
asset set, the layout-dependent contract facts per audited row; the
classify_objects_by_language pool 7, 41, 0 of the only layouts that render its
reviewed instruction, round 3 on layout 0 and no fallback), the scheduled relaunch branch on a fake plant (call order, v2/v3
records, a restarting clock epoch and each scheduled isolation refusal) and the
bridge's relaunch order. `test_robodojo_layout_evaluation.py` evaluates a
recorded scheduled three-round shape and one perturbation at a time, keeps the g003-g007
fixture shape valid, and covers the Supervisor's scene comparison, the round
notice sentence and the scheduled lane-gate certificate (including refusal of
centre-only object-reach records). Neither proves the
in-process relaunch on a real simulator; the scene smoke does.

`test_robodojo_start_gate.py` covers RDJ-10 offline: the node start gate's
slots and timeout, the per-node/boot readback cache (one measurement, a new one
after a reboot, driver change or installation change, nothing published on a
failed measurement), the script entry loading the gate by path, and the bounded
`nvidia-smi` retry.

`test_robodojo_cpu_physics.py` covers the RoboDojo CPU-physics lane offline:
the nine-scene table and each task's admission pin (and `require_gpu`), the
table as a reviewed kernel exclusion that no kernel list, scene dependency map
or card control closure binds, the benchmark edit confined to
`validate_runtime_extension`, the resolver default, the RunSpec refusals on the
maintained builder (sparse-motion physics key, then a K = 2 unscheduled CPU
session) and the matching preflight check, the
physics attestation and evaluation/seal record, and revalidation's unchanged
CUDA checks and coverage with the CPU `not_established` boundary. Device-keyed
names (`cuda<N>`, `cpu-gpu<N>`) and the CPU lane's smoke choice and gate are covered by
`test_robodojo_spec.py` and `test_robodojo_lane_qualification.py`.
`test_robodojo_physics_probe.py` drives the qualification-only physics scene
probe worker with a fake session and readers (samples, early terminal, reader
and boot failures, a probe-only limit for an unadmitted task and its refusal for
an admitted one, the in-place restart flags, the
module outside the worker kernel, OmegaConf-style sequences in the fluid reader,
and the fluid predicate's raw inputs recorded at reset and at the end of the
hold) and judges synthetic samples: conveyor travel
and belt height, articulation hold, fluid particle counts, the declared physics
mode, supports, short holds, cells, the probe runtime (the admitted fluid tasks
on their own limit; the probe-only table is empty) and report-only judging.

`test_robodojo_revalidation.py` verifies physical dependency invalidation, host-only checker reuse, terminal lifecycle drift, artifact containment/hash checks and original evidence identities across later commits. It also binds a card's whole terminal/control import closure by AST digest, with only reviewed `CARD_EXCLUSIONS` left out, keeps terminal and scene dependencies inside the worker kernel, and caches per full commit.


`test_robodojo_b2_parity.py` covers shared RunSpec contract delivery/source guards,
complete ARX results through the actual dispatcher, source-bound body parameters
and changed three-camera calibration across initial, after-action and next-round
Native/managed projections. It also preserves the v2 composition identity while
checking fresh v4 selection and literal model-visible examples against the
resolved schemas and pure-compute dispatcher. `test_robodojo_revalidation.py` additionally rejects
collision/frame changes and false measurement hashes when reusing geometry
evidence across a wording/provenance-only descriptor revision.

The B2/G5 repair regressions additionally cover nested target-state arrival
stops, RoboTwin skill dispatch through the public-TCP adapter on both arms,
copyable preplace/exit results on closed surfaces, persisted lower candidates,
and joined skill command history with corrupted children kept unknown.
Backend fixtures exercise the real adapter with a synthetic robot; these tests
do not qualify physical grasp or placement outcomes.

Compute usage v2 extends `test_compute_tool.py` with large/small-magnitude
normalization, quaternion order and transform direction, helper isolation,
explicit final-output selection and missing-output errors. The published JSON
examples are validated and executed through all eight applicable surfaces,
including causal-projection validation. These CPU tests do not assess model
adoption or physical outcomes.


`tests/test_tool_routing_contract.py` validates exact model-visible grasp/place
pose handoff examples against all eight selected consumer schemas, retains
rejection of the old incomplete shape, and checks atomic/closed tool routing.


Dojo B2 repair regressions exercise unreachable/null and FK-residual alignment
through the real dispatcher, applied close versus intermediate drive labels,
and malformed evidence clearing stale directives. `test_robodojo_runtime.py`
joins legacy nested and fresh sparse skill children to real temporary hashed
controller/physics journals and rejects tampering. `test_skill_sequence_executor.py`
checks Dojo sparse arrival, close verification and incomplete-open stopping;
`test_receipt_projection_failed_arms.py` preserves Dojo convergence/terminal
codes. `test_lossless_context_runtime_triggers.py` checks same-source L5 retry spacing,
exhaustion, later admission and emergency L6 during backoff.
`test_lossless_context_runtime_cost.py` binds a two-round, image-bearing history
(pre-registered and fresh images, L1–L4 and L6) to digests produced by the
pre-fix serializing projection, then bounds serialization, decode and Artifact
reads on repeat projections and checks that a changed Artifact file is
re-verified. `test_evidence_deferred_durability.py` holds a fake fsync closed
and proves the loop still writes, links and hash-chains every Artifact and
event (visible and hash-valid) without waiting, then checks barriers,
coalescing, bounded backpressure, sticky failures, latest-wins snapshots and
one Context index rewrite per projection.
`test_native_session_manifest_incremental.py` compares incremental
`agent-session.json` text with the reference serializer over randomized
bodies and a real Native session, and binds the 120 s snapshot interval. `test_lossless_render_reuse.py` counts
pass executions across the four renders of one transition-heavy projection,
checks copy-on-write passes never edit their input, and compares element-wise
list sizing with serialization. `test_lossless_context_stride.py` drives an
18-ActionStep, two-calls-per-step synthetic episode through the real pipeline
and request assembler: v4 rewrites delivered history on 16 steps, Context v5
with stride 4 on 4 (every other request is an exact-prefix extension, on both
OpenAI-compatible and Anthropic wire), and stride 1 is byte-identical to v4.
It also bounds inline media to five completed steps, resolves every aged
reference, shows the stride slack never admits L4 while a Provider-anchored
L5 capsule is still admitted mid-stride, checks the reset release, v5 coverage
evidence and schema validation. These are CPU
contract/evidence checks, not new simulator qualification or task-success data.
`test_context_management_v6.py` binds the absolute L5 watermark of Context v6
(user decision 2026-10-04): every maintained configuration selects v6 with
150,000 tokens; a 1M-window binding resolves and admits L5 at 150,000 instead
of v5's 618,086, a 200K binding at 150,000; L4 and L6 are unchanged; v6
validation; and recorded v4/v5 evidence keeps and revalidates only under its
window-scaled watermark.

The shared Context near-window regressions in `test_context_near_window.py`
cover image-target retention across resets, protected overflow, exact historical
public-body recovery for all three robot backends, invariant tampering and
source-aware seal coverage. `test_visual_usage_metrics.py` separates automatic
observations, current reads and historical reopens by causal event identity.
These tests cover `orchestrator/metrics.py` and its
`metrics_native_projection.py`, `metrics_provider_usage.py` and
`metrics_values.py` helper modules.

`test_execution_capture.py` also exercises RoboDojo RunSpec→factory→worker
configuration, the actual native-tick sampling hook, unchanged physics/reward
counts, original pixels, per-round action joins, disable and idempotent flush.
Its fake plant replaces SDK I/O; it is not Isaac renderer qualification.

Publication guards include the four camera-adapted Dojo surface
versions and validate each arm's delivered parameter descriptions as well as
top-level help. Frozen surface bytes remain unchanged. Failure-code coverage
distinguishes private native-budget reasons normalized to `controller_timeout`
from public gripper and official lifecycle stops.

The suite also keeps source-bound queue helper
imports working through current resource admission, audits the current Panda
v4 descriptor, covers Dojo in explicit environment/codec tables, and checks
three/four-camera acceptance plus deterministic cross-class memory admission.

`test_model_services_cli.py` checks that `scripts/model-services` dispatches
start/status/stop/qualify to the shared operator with the workspace state
directory and the Runtime state root, and that `status` exits 1 until both
services are ready.
`test_benchmark_spec.py` covers the maintained RoboTwin, LIBERO-Pro and LIBERO
builder offline: default-instance plans with a fresh spec's model-visible items,
RoboTwin `runtime.task` and contract delivery, refused LIBERO Servo keys, original
LIBERO base-suite cells and their base language, K-round sessions, the
expert-unsolved seed exception (admitted only on scan evidence of an expert
failure, refused once the evidence changes),
preflight's seed-admission rule, host and slot registration, readiness-report
refusals and the content-addressed seed-manifest import.
`test_suite_queue.py` covers the maintained suite queue offline with fake
launch, lease and supervisor: cell lists, waiting for the seal while the
supervisor lives (BOOT-09), recorded action-free faults and unsealed landings
up to the pause limit, preflight requeue with a new launch log, drain, resume
and lane/kind checks.
`test_robodojo_operator.py` covers the RoboDojo installer and the product
infrastructure qualification (`ora/orchestrator/robodojo_card_qualification.py`
and its entry point).

The expected defaults (`default_experiment_provider` in
`tests/data/default-expectations.json`) also bind new experiment Provider selection to
Low 16K; the tests check explicit High/package overrides and verify Low delivery in
the current G8 and RoboDojo successor specs.

Low Provider default checks also verify the dedicated credential variable in
resolved RoboTwin and composed RoboDojo configurations; test fixtures contain
no live secrets.


`test_robodojo_probe_timeouts.py` checks that the RoboDojo worker graphics readback, its host-side inspection and the preflight Python-stack probe use one 120 s bound: a writeback storm on the shared disk pushed the 30 s readback to 34.6 s while healthy.

Prepared grasp regressions (`test_prepared_grasp_execution.py`) cover maintained
defaults, geometry refusal before opening, committed stages, semantic routing,
failed-lift capture and rigid/stationary/unknown public depth evidence. RoboTwin
left/right checks bind the selected arm, candidate TCP/pregrasp poses, semantic
routing and 30 mm trial lift to the existing shared sequence.
`test_skill_observation_delivery.py` checks hash/path integrity, selected-arm
views, actual Native requests and managed image/context-bundle identity.
CPU fixtures do not qualify physical grasping or retained transport.



`test_artifact_provenance.py` separates Harness-issued projection/canonical
recovery pointers and successful reads from model-authored reasoning, prose and
rejected tool arguments. It preserves missing-file, hash, path and identity
failure checks, reports historical origin gaps explicitly and verifies that
read-only evaluation/revalidation leaves every original file unchanged. Output
inside or overlapping a formal run is refused, and unfinished prefixes are
deferred. `test_control_plane_cli.py` binds this command to the exact public CLI,
documentation order and dispatch contract.

`test_robotwin_prepared_grasp_qualification.py` checks the bounded four-case
qualification evidence contract: stopping versus physical coverage, stage order,
selected-arm routing, missing historical images and the difference between
measured zero jaw aperture and normalized actuator position. Its CPU assertions validate the checker; only
separately recorded simulator cases establish physical coverage. No fixture
creates a grasp-success or task-success claim.

`test_provider_prepared_grasp_probe.py` checks the restricted two-call Kimi/DeepSeek Low
diagnostic, exact candidate/tool admission, public receipt and image delivery,
and refusal of additional mutations. Its mock HTTP exercises both real protocol adapters without establishing a
live Provider result; live cases retain separate source-bound requests and responses.


- `test_provider_continuation_binding.py`: exact provider continuation identity across dispatch, missing tool IDs and both protocol projections.


The five large test families are organized by protected contract:

| Family | Modules |
|---|---|
| Control plane | Base planning/gates/sealing; `_preflight`, `_viewer`, `_workspace`, `_cli` |
| Runtime v1 | Base lifecycle; `_state`, `_metrics`, `_tools`, `_libero` |
| Native Harness | Base composition/recovery; `_anthropic`, `_openai`, `_sft`, `_loop` |
| Lossless Context | Base projection; `_triggers` |
| RoboTwin motion v2 | Base execution; `_grasp`, `_probe`, `_profiles`, `_sparse`, `_surface` |

Shared helpers and constants remain in each base module and are explicitly
imported by its children, preserving existing cross-module fixture imports.
Current test bodies and parametrization are migrated intact; predecessor-only
tests are not restored. These splits
improve navigation; they do not establish a measured
full-suite speedup.

`test_inspect_export.py` checks read-only Inspect export identity and evidence
location. `test_robodojo_kernel_paths.py` checks isolated kernel source bindings
and holds `WORKER_KERNEL_PATHS` to the worker's static import closure
(`orchestrator/import_closure.py`, `robodojo_kernel_review.py`). Every reached
module must be listed, reviewed-excluded or behind a reviewed pruned dispatcher
edge; an excluded module's own imports are still checked. Stale table entries
fail, and unbound modules on the worker's import-time chain may only make
declarative calls. It also covers readmitting a record whose manifest only grew
by unchanged files, end to end through the real admission code, and the
`kernel_admission` report it returns.
`test_b2_collaboration.py` checks collaborator CLI defaults and service inputs.
These use offline fixtures; they do not certify simulator or service readiness.

`test_robodojo_ik_continuity.py` covers per-seed feasibility, joint limits and FK
residuals, raw-coordinate jump admission before distance ranking, named-joint
extraction, bounded SDK seed counts and local/global bounds without a simulator.
It covers the pitch-joint rule (EXP11A g005 audit D3, review U1; X5 joint2-joint5):
a candidate taking joint3 from 2.9 to 3.3 rad (or -2.9 to -3.3) is refused as
`pitch_joint_wrap`, one returning from 3.3 to 2.9 (or -3.3 to -2.9) is
admitted, one winding it more than 1e-3 rad farther is refused, a missing mask
is derived from the revolute joint names, the policy text names no joint, and
joint1 may still wind. Canonical-first ordering applies only between
candidates that differ in canonicity: a continuous candidate inside [-π, π]
beats a nearer one past π only within the session's local joint-jump bound,
canonical candidates keep their nearest order among themselves, a canonical
candidate beyond the jump bound is still refused, and a blocked canonical path
falls back to the nearest remaining candidate.
It also covers the SDK-sphere straight joint-path self-collision screen: pair
selection, start-overlap tolerance, fail-closed model binding, next-candidate
fallback, the no-target refusal, and the recorded EXP11A g001 near-base sweep
when the pinned RoboDojo X5 assets are present.


`test_libero_planning_diagnostics.py` verifies the instance-local CuRobo observer,
per-seed convergence and feasibility, bounded evidence, and unchanged original
solver result when diagnostics are unavailable.

- `test_motion_repair_admission.py`: selected-pregrasp admission before any commitment or action, parallel-jaw member adoption from the LIBERO, RoboTwin and RoboDojo list-shaped probe results (compiled kept when admitted, common member adopted, no common member refused, non-list targets a contract fault), current mode companion help and one-refusal recovery, terminal joint/TCP evidence, named collision spheres and learned/PCA compiler aperture bounds. Uses fake read-only bridges, no simulator allocation.

`test_contact_graspnet_input_policy.py` drives the resident Contact-GraspNet
adapter against a stub estimator with upstream's control flow: each distinct
crop point is fed once in a request-seeded order, a crop above 20000 distinct
points keeps upstream's exact subsample, a crop below 16 distinct points yields
no region, and the input regime reaches the response, health and the client's
private diagnostics. No checkpoint, CUDA or upstream source tree is needed.

`test_grasp_rejection_schema.py` joins actual PCA/learned rejection diagnostics
to every current declaring grasp result schema, checks numerical boundaries,
and preserves undeclared historical schemas. It covers the optional
compiler-opening feedback boundary added during the 0048 return.

## EXP11A g004 orchestration unit (2026-09-29)

`test_decision_accounting.py` pins which executed receipts spend a
robot decision: only a receipt certifying zero writes (`state_changed: false`
with a present, zero native count) or a target-state composite whose every
dispatched phase is such a refusal does not. It drives a Synthetic episode in
which two refusals leave the two-decision allowance intact, and checks the
separate `refused_proposals` counter on result, status, interaction state and
events. `test_native_action_allowance.py` pins the effective
native-action limit, the receipt-count sum, the unknown-not-guessed remainder
and Native delivery of the two new remaining-budget fields.

`test_round_wait_accounting.py` (EXP11A g013, E11G12-005) pins the
per-round `wait_accounting` block: waits and terminal waits (after the last
non-wait proposal, executed or refused) with the native actions their receipts
consumed, a budget-exhausted run of consecutive waits still terminal, waits
interleaved with mutations not terminal, unknown rather than guessed counts,
the `events.jsonl` replay, the session totals and their evaluation
reconciliation, all on synthetic sequences and one Synthetic session.

`test_provider_recovery_deadline.py` gains the bounded stream-break contract
(`STREAM_BREAK_RETRIES`). Its credential-echo cases drive a real
transport worker whose stream echoes the key and then breaks: the call must
fail closed after one request, on both protocols and on the legacy path, with
no secret in the attempt record. `test_robodojo_round_ending.py` replaces
the pin on the escaped failure-latch ending with positive contracts for
`official_failure_latch` and the matching environment status reason;
`test_robodojo_task_contracts.py` checks that the live bundle binds boundary
v2; `test_compute_tool.py` covers
escaped line breaks without widening the evaluator allowlist.

## Run stop-reason and cost report (integration 0059)

`test_run_report.py` covers `orchestrator/run_report.py` and
`scripts/run-report` on synthetic run directories written by
`tests/run_report_fixture.py`; no simulator or provider starts. It pins one
ending class per round (RoboDojo upstream latches and native-action limit,
RoboTwin/LIBERO environment endings, Harness budgets, policy finish, operator
interrupt, infrastructure including an admitted v2 abort, in-progress), validity
from `evaluation.json` and learning eligibility with post-seal correction
records discovered under `<namespace>/diagnostics/corrections/` (cited seal
and evaluation hashes verified, a mismatch flagged; corrected runs listed with
their correction id and excluded from success and eligibility counts), the policy
finish outcome and give_up category from the v2 round summary, the evaluation
report or an accepted `finish_episode` call (absent without error), refused tool
calls by tool and code, provider tokens with the cache share and the
API-equivalent cost of the shipped price table (unpriced models null), context
sizes, cohort/model-task/total aggregates, the unfinished round of a faulted or
running session, and that the CLI writes JSON and Markdown outside every source
run without changing evidence.
## Summary line, degrees and once-per-payload delivery (2026-10-04)

Integration 0059 units D1, D2 and D3a are capability-gated and no registered
surface declares them yet, so their tests add the capabilities locally:
`tests/delivery_lessons_fixtures.py` provides `with_capabilities` for runtime
projection and `composed_with` for composition. It also builds RoboDojo,
RoboTwin, LIBERO and closed-skill receipts with the contract owners' builders.

- `test_action_summary.py` pins the D1 line per backend shape. Every figure in
  the line must be a rounding of a value in the effect, no advice word may
  appear, and the rest of the effect stays unchanged.
- `test_policy_units.py` holds the D2 table to every radian key a producer can
  write, checks conversion and free text, and checks both gateways and composed
  help text on test-local degree surfaces.
- `test_payload_dedup.py` assembles a real Native RoboDojo action-result
  observation and resolves every D3a pointer inside it against the legacy copy;
  it also restores the Native `camera_geometry` conventions stated once under
  `current_actual.conventions` to the legacy `current_actual`.
- `test_delivery_legacy_bytes.py` checks that surfaces without the capabilities
  keep exact bytes.

`test_robodojo_backend.py`'s SDK-limited close case now projects with every
capability except these three, because under D3a the feedback it reads
moves to the current facts.


## Each arm's own translation (E11BG6-004, 2026-10-04)

`test_robodojo_arm_translation.py` projects trimmed copies of recorded EXP11B
receipts (`tests/data/robodojo/exp11b-arm-translation-receipts.json`: a left-,
a right- and a both-arm `move_pose`). Without `per_arm_realized_translation_v1`
the left move still reports the recorded right-arm zero; with it each targeted
arm reports its own translation in `detail.arms`. When the recorded run 77dc is
on the host, every one of its 26 left-arm receipts is checked as well; elsewhere
that case skips.

## Static facts once per round (D3b, 2026-10-04)

`native_slots_v4` cannot yet be selected through a RunSpec (registering it in
`orchestrator/spec_composition.py` and `spec_runtime.py` is integration 0058
unit A0's), so its tests build the session directly or set the composition on a
resolved synthetic plan.

- `test_round_static_facts.py` covers the pure split (slot capture, `reduce`,
  `restore`, the naming and identifier rules, the lossless marker) and the
  Native session: one slot per round, appended after its observation, composed
  with D3a.
- `test_round_static_facts_context.py` drives the stride episode of
  `test_lossless_context_stride.py` with slots: no extra rewrite, the current
  slot byte-exact through every batch, the previous slot aged at reset, the L6
  carry, plus one- and three-round synthetic Supervisor episodes that evaluate
  valid.
- `test_current_visual_delivery.py`, `test_parameter_delivery_guards.py` and
  `test_robodojo_b2_parity.py` each gained a `native_slots_v4` case beside their
  unchanged v3 assertions: every image stays in the observation and every
  calibration reaches the request through the observation or its slot, and a
  changed camera is never served from the slot.


## A skipped retract is not the arm's planning outcome (E11G7-001, 2026-10-04)

`test_robodojo_retract_diagnostics.py` projects trimmed copies of recorded
EXP11B receipts (`tests/data/robodojo/exp11b-retract-diagnostic-receipts.json`:
home recoveries whose retract was skipped after `ik_failed` or
`ik_branch_jump`, reached and not reached, one with a planned retract, a
both-arm recovery with one of each, and a refused `move_pose`). Without
`per_arm_realized_translation_v1` a reached recovery still carries the recorded
`no_valid_endpoint` diagnostic; with it only an arm whose last IK attempt
belongs to a skipped retract loses its `planning_diagnostics`. When the 47
recorded EXP11B runs are on the host, all 3,196 of their receipts are projected
with and without the capability: only those diagnostics change, 78 of them
`no_valid_endpoint` on reached recoveries in 31 runs (90 in all). Elsewhere that
case skips.

## Tool image labels name the call (census A4, 2026-10-04)

`test_tool_image_call_labels.py` runs one synthetic Supervisor episode per
composition whose single turn calls `mark_point`, `preview_move` and `observe`
twice each. Under `native_slots_v4` (`tool_image_call_labels_v1`) every image
message reads `<kind>: N image(s) returned by <tool> (<call id>).`; under
`native_slots_v3` the labels keep their recorded bare-kind bytes and the session
records no Context capability. `test_round_static_facts_context.py` now expects
both v4 capabilities at `session_started`.

## Delivery trim (observation_delivery_trim_v2, 2026-10-05)

The three default surfaces declare the capability, so its tests resolve the
real surfaces; `robotwin_toolchain_b2_closed_noservo_v1` stands in for "not
declared".

- `test_delivery_trim.py` covers each rule on its own: the camera and
  frame-label defaults, the arrival-tolerance equality, untargeted arms,
  capability gating, and the Native rules of `delivery_trim_native.py` (view
  freshness, the envelope) with core v8's statements.
- `test_delivery_trim_prose.py` holds every fixed text in
  `delivery_trim_prose.ENTRIES` to its producer (constant import, or the
  literal in the producer's source for inline texts) and to the composed
  description of its tool on each listed v2 surface.
- `test_delivery_trim_runtime.py` builds the Native observation through the
  real session on the direct_v4 surfaces (reusing `test_current_visual_delivery`'s
  synthetic publication), checks that the published observation, calibration
  and canonical tool result keep every byte, ages trimmed observations in a
  lossless projection to check that the recoverable observation reopens to
  exactly the delivered text, and checks that an image marker keeps its full
  form.

`test_robodojo_closed_skill_surface_contract_v2.py` expects the
`delivery_trim_help` slot before the published-limits layers and carries the
reviewed byte ceilings with the layer's measured growth.
`test_current_visual_delivery.py`, `test_parameter_delivery_guards.py`,
`test_native_baseline.py`, `test_bundle_quantities_ratchet.py` and
`test_model_visible_identity.py` now expect the direct_v4 Native forms: one
pose matrix and its owner, the `absent` listing, and core v8.
