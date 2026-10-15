# Get started with RoboTwin

[RoboTwin 2.0](https://robotwin-platform.github.io/) is a dual-arm (ALOHA)
manipulation benchmark with 50 tasks in clean or randomized scenes. ORA drives
both arms through perception, grasp-planning and motion tools; RoboTwin's own
`check_success` decides success.

| | |
|---|---|
| Machine | x86_64 Linux with one NVIDIA GPU, 32 GB or more |
| Disk | ~57 GB (14 GB of asset archives, unpacked to 16 GB) |
| Install | `./scripts/ora setup --env robotwin` |
| Run | `./scripts/ora run --env robotwin --task click_bell --confirm-launch` |

You need ORA itself (`git clone https://github.com/hao-ai-lab/ORA.git &&
cd ORA && uv sync --all-extras --locked`) and a DeepSeek API key.

## Install and run

```bash
./scripts/ora setup --env robotwin
export DEEPSEEK_LOW_API_KEY=sk-...
./scripts/ora run --env robotwin --task click_bell --confirm-launch
```

- **Install.** Setup installs, mostly as downloads:
  - RoboTwin at its pinned revision with ORA's vendored planner install;
  - a Python 3.10 environment with CUDA 12.8 PyTorch and the upstream SAPIEN and MPLib;
  - CuRobo built for your GPU (about 6 minutes);
  - the RoboTwin 2.0 assets;
  - the SAM3 and Contact-GraspNet services;
  - the terminal UI.

  Check the [tools to install first](single-gpu.md#2-user-space-tools).
- **First run of a task.** RoboTwin needs a seed that the upstream expert
  policy can solve on your machine. `ora run` shows an estimate, asks, then
  scans the task's seeds and keeps the first that works: about 2 minutes for
  `click_bell`. Later runs of that task reuse the record.
- **Run.** The perception services start if needed (about 3 minutes the first
  time). Then the terminal UI opens with the task; press Enter in the picker.
- **Result** on an RTX PRO 6000: `click_bell` reached the official success in 4
  decisions and 12 model calls; the episode took about 4 minutes. GPU memory
  peaked at 8 GB for the simulator plus 8–10 GB for the perception services.

## Choosing a task

`--task <name>`; `--task-config demo_randomized` (default, cluttered scenes
with random textures) or `demo_clean`; `--seed` picks a seed the qualification
accepted. The 50 tasks:

`adjust_bottle` `beat_block_hammer` `blocks_ranking_rgb` `blocks_ranking_size`
`click_alarmclock` `click_bell` `dump_bin_bigbin` `grab_roller` `handover_block`
`handover_mic` `hanging_mug` `lift_pot` `move_can_pot` `move_pillbottle_pad`
`move_playingcard_away` `move_stapler_pad` `open_laptop` `open_microwave`
`pick_diverse_bottles` `pick_dual_bottles` `place_a2b_left` `place_a2b_right`
`place_bread_basket` `place_bread_skillet` `place_burger_fries` `place_can_basket`
`place_cans_plasticbox` `place_container_plate` `place_dual_shoes` `place_empty_cup`
`place_fan` `place_mouse_pad` `place_object_basket` `place_object_scale`
`place_object_stand` `place_phone_stand` `place_shoe` `press_stapler`
`put_bottles_dustbin` `put_object_cabinet` `rotate_qrcode` `scan_object`
`shake_bottle` `shake_bottle_horizontally` `stack_blocks_three` `stack_blocks_two`
`stack_bowls_three` `stack_bowls_two` `stamp_seal` `turn_switch`

```bash
./scripts/ora run --env robotwin --task stack_blocks_two --task-config demo_clean --confirm-launch
```

Some tasks have few seeds the expert can solve. If the scan finds none, widen
it with `--candidates 8`.

## While it runs

Enter sends the agent a message (the run is then marked human-assisted), Esc
stops it, Ctrl-C leaves the UI with the run still going. Replay any run with
`./scripts/ora watch RUN_ID`. See [Watch, steer and inspect](single-gpu.md#6-watch-steer-and-inspect).

## More

- [One-GPU guide](single-gpu.md): requirements, GPU and driver notes, troubleshooting.
- [Explicit RoboTwin installation](../environments/robotwin.md): each pinned source, asset and seed-scan command, for a custom or multi-GPU host.
