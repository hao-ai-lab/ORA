# RoboDojo: install and run one episode

Read [shared setup](common.md) for host slots, provider profiles and perception services. RoboDojo uses its Isaac Sim environment, X5 arms and the upstream CuRobo v2 fork. The official task success function supplies the verdict.

## Install the isolated runtime

The maintained [installer reference](../../ora/environment/backends/robodojo/installation/INSTALL.md) contains the exact sources, asset revisions, package manifest and graphics patch. Its tested recipe uses Python 3.11.15, Isaac Sim 5.1.0.0 and Torch 2.7.0/CUDA 12.8. Use a permanent root:

```bash
export ORA_DOJO_INSTALL=/srv/ora/robodojo
./scripts/setup-robodojo sources --root "$ORA_DOJO_INSTALL"
./scripts/setup-robodojo assets --root "$ORA_DOJO_INSTALL"
./scripts/setup-robodojo sdk --root "$ORA_DOJO_INSTALL"
./scripts/setup-robodojo resolve --root "$ORA_DOJO_INSTALL"
./scripts/setup-robodojo packages --root "$ORA_DOJO_INSTALL"
./scripts/setup-robodojo bind-assets --root "$ORA_DOJO_INSTALL"
./scripts/setup-robodojo doctor --root "$ORA_DOJO_INSTALL"
./scripts/setup-robodojo manifest --root "$ORA_DOJO_INSTALL"
export ROBODOJO_ROOT="$ORA_DOJO_INSTALL/source/RoboDojo"
export ROBODOJO_ASSET_ROOT="$ORA_DOJO_INSTALL/assets/Assets"
export ROBODOJO_PYTHON="$ORA_DOJO_INSTALL/env/bin/python"
uv run --locked ora doctor --env robodojo
```

`sources`, `assets` and `sdk` may run independently. Later stages depend on those inputs. Installation does not launch a simulator or qualify physics. The installer preserves source/package/asset receipts and supports reuse of matching components. Isaac SDK and assets are large separate downloads; consult their licenses and check local disk capacity first. No portable VRAM or installation-time minimum has been established by ORA.

## Qualify the lane

Inspect supported task identities:

```bash
uv run --locked python -m ora.orchestrator.robodojo_spec tasks
```

On the GPU/slot registered in your host profile, obtain a card qualification and a task record. These are simulator-only development checks, not model episodes:

```bash
uv run --locked python -m ora.orchestrator.robodojo_lane_qualification card stack_bowls \
  --gpu 1 --resource-profile robodojo-gpu1 --yes
uv run --locked python -m ora.orchestrator.robodojo_lane_qualification task stack_bowls \
  --gpu 1 --resource-profile robodojo-gpu1 --card PATH_TO_CARD_RECORD --yes
```

The card check performs two full-length episodes (the operator's estimate is 35–56 minutes). Reuse a still-valid card record whose actual source, controller, device and asset dependencies match. A task record admits the default layout 0 only. For another layout or a multi-round session, use the lane operator's `smoke` and offline `revalidate` commands with the exact schedule; see their `--help`. Qualification evidence belongs under the selected shared workspace, outside Git.

## Compose and launch

Obtain a [perception readiness report](common.md#install-the-shared-perception-services), then:

```bash
uv run --locked python -m ora.orchestrator.robodojo_spec compose stack_bowls \
  --experiment dojo-example --gpu 1 --resource-profile robodojo-gpu1 \
  --task-qualification PATH_TO_TASK_RECORD \
  --service-report PATH_TO_READINESS_REPORT --cohort dojo-example-g001
```

Use the printed path in the [review/launch commands](common.md#run-and-inspect-one-composed-episode). The composer inherits the selected default tools, provider, task instruction and official task-specific physics ceiling. The result records the official boolean verdict separately from infrastructure faults. Check actual elapsed time and tokens in the run rather than treating the budget as expected duration.

A missing Isaac import usually means the wrong interpreter or an incomplete SDK stage. Rendering failures require checking Vulkan/EGL and driver components. Asset verification failures should be repaired through the installer; do not replace source pins or qualification hashes to force admission. Physics devices and layout schedules are task-bound, so evidence for one device/layout cannot silently qualify another.
