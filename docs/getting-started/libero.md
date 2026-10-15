# Get started with LIBERO / LIBERO-Pro

[LIBERO](https://libero-project.github.io/) is a set of Panda-arm kitchen and
table tasks; [LIBERO-Pro](https://github.com/Zxy-MLlab/LIBERO-PRO) adds
source-paired robustness variants of the same tasks (swapped objects or
positions, paraphrased instructions, changed tasks). ORA drives the robot
through perception, grasp-planning and motion tools; LIBERO's own termination
check decides success.

There are two ways to run it:

| | One NVIDIA GPU | Computer without a GPU |
|---|---|---|
| Machine | x86_64 Linux, NVIDIA GPU (24 GB is enough for LIBERO) | macOS on Apple Silicon or Linux |
| Tools the model gets | full set: segmentation, grasp proposals, CuRobo motion planning | reduced set: bounded moves, gripper, RGB-D, point marking |
| Install | `./scripts/ora setup --env libero` | `./scripts/ora setup` |
| Run | `./scripts/ora run --env libero --task libero_goal/0 --confirm-launch` | `./scripts/ora local --confirm-launch` |
| Use it for | benchmark-comparable runs | trying ORA on your own computer |

Both need ORA itself (`git clone https://github.com/hao-ai-lab/ORA.git
&& cd ORA && uv sync --all-extras --locked`) and a DeepSeek API key.

## On one GPU

```bash
./scripts/ora setup --env libero
export DEEPSEEK_LOW_API_KEY=sk-...
./scripts/ora run --env libero --task libero_goal/0 --confirm-launch
```

- **Install** (~35 GB, mostly downloads; CuRobo builds in about 6 minutes):
  LIBERO-Pro sources and task data at pinned revisions, a Python 3.11
  simulator environment with CUDA 12.8 PyTorch, CuRobo built for your GPU,
  the SAM3 and Contact-GraspNet perception services, and the terminal UI.
  Check the [tools to install first](single-gpu.md#2-user-space-tools).
- **Run.** `ora run` starts the perception services if needed (about
  3 minutes the first time), writes the run's specification and opens the
  terminal UI. Press Enter on the task in the picker. LIBERO needs no
  one-time qualification.
- **Result** on an RTX PRO 6000: `libero_goal/0` ("open the middle drawer of
  the cabinet") reached the official success in 5 decisions and 11 model
  calls; the episode took about 4 minutes.

### Choosing a task

`--task <suite>/<id>`, with `--seed 0..49` for the initial state (default 0):

| Suites | Tasks |
|---|---|
| `libero_spatial`, `libero_object`, `libero_goal`, `libero_10` | ids 0–9 in each |
| LIBERO-Pro: `<suite>_task`, `<suite>_swap`, `<suite>_lan`, `<suite>_object` (for example `libero_goal_swap`) | the same ids, with the perturbation applied |

```bash
./scripts/ora run --env libero --task libero_spatial/3 --seed 7 --confirm-launch
./scripts/ora run --env libero --task libero_10_lan/2 --confirm-launch
```

## On a computer without a GPU

```bash
./scripts/ora setup                       # ~2 GB: LIBERO sources, a CPU simulator environment, the terminal UI
export DEEPSEEK_LOW_API_KEY=sk-...
./scripts/ora doctor                      # "ready for task selection"
./scripts/ora local --confirm-launch      # pick a suite and task in the picker
```

This uses a reduced tool set, so its results are not comparable with the GPU
setup. On Linux, `libero_goal/0` reached the official success after 40 actions
and 70 model calls in about 21 minutes. Details: [Run ORA on your computer](local.md).

## While it runs

Enter sends the agent a message (the run is then marked human-assisted), Esc
stops it, Ctrl-C leaves the UI with the run still going. Replay any run with
`./scripts/ora watch RUN_ID`. See [Watch, steer and inspect](single-gpu.md#6-watch-steer-and-inspect).

## More

- [One-GPU guide](single-gpu.md): requirements, GPU and driver notes, troubleshooting.
- [Explicit LIBERO installation](../environments/libero.md): each pinned source and package, for a custom or multi-GPU host.
