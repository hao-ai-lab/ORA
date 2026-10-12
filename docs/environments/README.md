# Robot environments

Complete the [Synthetic first episode](../quickstart.md) before installing a
simulator. ORA stays in its locked host environment; each simulator runs under
its own explicit interpreter.

| Environment | Use it for | Start here |
|---|---|---|
| LIBERO / LIBERO-Pro | Panda tasks and source-paired robustness cases | [Install and run one task](libero.md) |
| RoboTwin | Dual-arm manipulation in clean or randomized scenes | [Install and run one task](robotwin.md) |
| RoboDojo | Isaac Sim manipulation tasks | [Install and run one task](robodojo.md) |

## Setup sequence

1. Install the pinned simulator and assets using its guide.
2. Register the host and required perception services using [common setup](common.md).
3. Run environment-specific doctor discovery.
4. Obtain the source-bound controller, seed, service, and provider qualification
   required by the task.
5. Compose a fresh RunSpec, inspect it, and explicitly launch the episode.

Installation is not qualification. Each environment guide covers interpreter
variables, source pins, assets, and one-task commands. The
[support matrix](../support.md) records what was checked.

## Results

Only the environment's official boundary supplies the manipulation verdict.
A model's statement that it finished is not a success label. Synthetic reports
`official_success: null` and has no benchmark score.
