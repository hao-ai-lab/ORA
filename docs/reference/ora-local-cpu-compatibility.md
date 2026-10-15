# ORA local CPU compatibility record

Status: source facts reviewed before authoring the local bundle; local rendering,
reset and device qualification remain separate. This record follows the
[environment compatibility guide](../guides/environment-compatibility-authoring.md).
It authorizes no simulator or provider launch and makes no Mac behavioral claim.

## Sources and scope

The maintained source contract is LIBERO-Pro code
`2b910b5b5f53016bef9907632f6f840f1ce2229c` and data
`c86fc3b8293185a6f373677018ff3e37f8391602`, owned by
`ora/environment/benchmarks/libero_pro.py`. Task identity and public
language come from that module and `libero_pro_language.py`; the isolated worker
checks the actual loaded task against those inputs. RoboSuite 1.4.1 Panda,
PandaGripper and OSC_POSE are the existing source-paired body/controller.

This treatment selects the existing direct driver, atomic v3 actions and bounded
OSC executor. It removes the full baseline's CuRobo and resident perception
dependencies; it does not alter their archived or maintained configurations.
`agent/composition/ora_local.py` owns the local composition, while the ordinary
server baseline remains unchanged. Local runs retain the `ora-interactive` namespace and normal Supervisor
admission, leases, evidence and sealing.

## Tasks, authority and counts

| Fact | Source and consumer | Reviewed boundary |
| --- | --- | --- |
| Base-suite task identity and instruction | `environment/benchmarks/libero_pro_language.py`; local spec and first task message | Official filename language; task id is a registry index, not an inferred visual label. The source BDDL language is recorded separately when wording differs. |
| Public completion clause | Exact supplied official instruction; existing `task_language_divergence_ack` review for base wording differences | No hidden goal atoms, target coordinates or solution trajectory enters the prompt. |
| Official success | `environment/backends/libero_pro/direct.py:DirectLiberoEnvironment.step`; `session.py:persist_official_verdict` | LIBERO's returned termination Boolean, separate from horizon truncation, policy finish and infrastructure faults. The agent's checker reads that authority. |
| Reset | `direct.py:reset` | One source init state, seed bound to that state; 15 zero-pose/open-gripper settling controls precede policy execution and are not counted as policy actions. No auto-reset. |
| One sample | One interactive run/round and source seed | Human assistance remains marked; this local demo is not a full-suite benchmark comparison. Retry creates a new run and never rewrites sealed evidence. |
| Clocks | `direct.py`, `local_motion.py`, Runtime metrics | Model/tool/decision counts, native controls, physics time and wall time remain separate. Thinking does not advance simulation. |

## Body, constants, frames and conversion owners

The following table references existing executable owners rather than making a
second parameter catalog. No controller bounds or robot geometry are changed.

| Quantity | Source / exact semantics | Conversion owner and consumers | Evidence / remaining gap |
| --- | --- | --- | --- |
| Body / controlled point | `common.py:libero_environment_init_params`, source RoboSuite Panda/PandaGripper; one fixed seven-joint right arm; `grip_site` | `observation.py`, `local_motion.py`; no hand-origin or object-contact offset added by local composition | Existing control-qualification/source checks; fresh local source/device check still required. |
| Pose | World metres and xyzw quaternion; relative inputs in world or measured grip-site frame | Atomic `pose-v2-input-schemas.json`, shared pose compiler, action v3 validator and OSC executor | Existing `test_native_causal_runtime.py` nonzero 10 mm and nonidentity 180° fixtures; no new conversion. |
| Controller scaling | `environment/controllers/libero_osc.py`; delta OSC, 0.025 m / 0.10 rad per-tick servo clipping; actual input/output scales loaded from source config | `compile_libero_osc_pose_command`; worker queue consumes normalized seven-element controls | Existing independent controller-scale fixtures; local rendering/physics smoke remains pending. |
| Pose arrival | `contracts/action.py` and `local_motion.py`; measured position/angular residuals, separate terminal tracking envelope | Existing controller and receipt projection | Endpoint arrival is not collision, contact, grasp or task success. |
| Gripper | `local_motion.py`: action v3 opening 0 closed / 1 open compiles to `1 - 2 * opening`; pose action input 0 preserves actuator directive | Existing executor; legacy atomic observe reports normalized opening × nominal 80 mm as `aperture_mm`, not a separately calibrated measurement | Encoder/actuator target and physical jaw separation are distinct; the local help/card label the nominal proxy. Physical contact and retention are not inferred. No learned-grasp geometry is activated. |
| Atomic preservation | `contracts/action.py`, `local_motion.py` | Pose-only preserves gripper directive; gripper-only holds measured EEF pose | Contact can still move measured held dimensions; no combined pose-and-gripper request. |
| Workspace | `environment/workspace_profiles.py`: `libero_pro_panda_tabletop_v1` | Runtime resolver, action validation, observation and atomic tool preview | Hard engineering envelope; not a reachability or scene-collision certificate. |
| Cameras | `common.py`: `agentview` fixed; source `robot0_eye_in_hand` exposed as `wrist`; 256×256 RGB-D | `observation.py`, `camera_geometry.py`, ToolSurface view config | Wrist calibration is observation-bound; actual RGB-D/render correctness must be checked locally. |
| Pixels / depth | OpenCV top-left [x,y] pixels at returned view size; metric depth and per-observation calibration | Existing `mark_point`/camera geometry; no extra resizing or depth conversion in composition | Invalid depth remains unavailable, never guessed. |
| Control timing | Existing LIBERO OSC controls at 20 Hz; simulator dt read from loaded source | Direct driver, local executor, capture and metrics | One native control is not one physics substep or model call. |
| Platform / graphics | Named external interpreter; platform graphics selection belongs to worker construction | Public ORA local entry passes explicit worker environment; no provider secrets needed by simulator | Darwin/GLFW and Linux software rendering require device validation. MuJoCo installation alone is not qualification. |

## Composition and delivery mapping

Planner/core reuse current canonical cross-task sources. Environment lifecycle
reuses the current LIBERO boundary. The new local embodiment card states only
the audited OSC/Panda facts above and binds a source-reference control profile;
it does not activate a CuRobo motion profile or a physical grasp compiler.
The isolated worker substitutes the version-matched `opencv-python-headless`
distribution for RoboSuite's desktop `opencv-python` dependency. Both expose
`cv2`; the selected offscreen environment never calls OpenCV's desktop renderer
or the private segmentation-display path. This avoids a desktop `libGL`
dependency without changing image/depth/calibration handling.
The task slot receives the exact official source instruction once. All nine
atomic tool schemas/effects reuse `native_atomic_causal_v3`; a new
`local_atomic_cpu_v1` manifest adds current required help without changing the
old manifest. No segmentation, grounding, VLA or grasp-model endpoint is bound.

The local composition explicitly selects action v3, the local bundle/profile,
disabled Task Memory and causal ActionStep context. Provider choice and endpoint
are caller inputs; credentials are environment references, never prompt text.
Runtime observations/receipts remain dynamic and authoritative.

## Acceptance and limits

Before authoring, task/evaluator, body/control point, units, conversion direction,
camera names and authority owners above were checked against current sources.
No required source fact was filled from another robot or an old success video.
The authored bundle must pass the simulator-free bundle audit and a freshly
resolved RunPlan must select the local surface/context/profile, no GPU services
and the exact source revisions. Focused tests cover input rejection, source and
task identity, provider/resource propagation, first-message delivery and preserved
server defaults. Existing atomic and OSC unit evidence is reusable because their
implementation and schemas are unchanged.

Fresh local interpreter, source/assets, rendering, measured motion, reset and
cleanup remain necessary device evidence. Until obtained, source composition and
offline tests establish compatibility intent, not successful Mac operation.
No benchmark success or cross-platform qualification is claimed by this record.
