# Supported configurations

These are supported integrations, not success-rate claims. A configuration absent from this table is untested, not necessarily unsupported.

| Environment | Suites / tasks | Status | Runtime |
|---|---|---|---|
| Synthetic | One-action infrastructure demonstration; M0 qualification | Supported | CPU, no simulator |
| LIBERO | Spatial, Object, Goal, LIBERO-10 | Supported | Panda; direct LIBERO worker |
| LIBERO-Pro | Source-paired Obj/Pos/Sem/Task variants | Supported | Panda; authoritative BDDL/init-state data |
| RoboTwin | 50 tasks, `demo_clean` and `demo_randomized` | Supported | ALOHA; SAPIEN worker |
| RoboDojo | Tasks in the shipped registry and reviewed layout pool | Supported | ARX; Isaac worker |

LIBERO-Pro Env perturbations are not claimed reproducible by the core dataset path. RoboDojo tasks/layouts must be in the source-bound compatibility catalog. A task's presence does not promise that a selected model can solve it.

## Model connections

| Interface | Selection | Requirements |
|---|---|---|
| OpenAI-compatible Chat Completions | `openai_compatible` | Tool calls; images when selected by the observation contract |
| Anthropic Messages | `anthropic_messages` | Tool use; images when selected by the observation contract |
| Self-hosted model | Either protocol above | Explicit model ID, endpoint, credentials and compatible tool/observation support |

`ora providers` lists the built-in profiles. A custom profile may select model limits and wire options; it cannot change the Supervisor's authority. A profile records a supported protocol binding, not a promise that every account has access to that model.

## Platform and hardware

ORA's canonical environment is Python 3.11 on Linux. Synthetic uses CPU only. The benchmark integrations use Linux x86_64, NVIDIA GPUs, separate simulator interpreters and the graphics libraries described on their installation pages. macOS and Windows simulator execution have not been qualified.

GPU memory depends on scene assets, renderer, motion planner and perception services. There is no single minimum VRAM claim for all three environments. Declare GPU slots in a host profile and let preflight check capacity and qualification; do not infer readiness from `nvidia-smi` alone.

Benchmarks, simulator SDKs, model weights and assets are downloaded separately and keep their upstream licenses. ORA does not distribute them.
