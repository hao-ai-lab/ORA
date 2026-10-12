# RoboTwin: install and run one episode

Read [shared setup](common.md) for host slots, model profiles, perception services and launch review. The default RoboTwin surface uses the ALOHA dual-arm robot, RGB-D perception and the vendored CuRobo planner. The simulator's `check_success` supplies the verdict.

## Source, interpreter and planner

```bash
export ORA_INSTALL_ROOT=/srv/ora
mkdir -p "$ORA_INSTALL_ROOT/sources" "$ORA_INSTALL_ROOT/envs"
git clone https://github.com/RoboTwin-Platform/RoboTwin.git "$ORA_INSTALL_ROOT/sources/RoboTwin"
git -C "$ORA_INSTALL_ROOT/sources/RoboTwin" checkout --detach c3ddfa8b97d5519efa828b075999bd0006778e5e
uv run --locked python -m ora.environment.backends.robotwin.vendored_install \
  --robotwin-root "$ORA_INSTALL_ROOT/sources/RoboTwin"
git -C "$ORA_INSTALL_ROOT/sources/RoboTwin" rev-parse HEAD
# Expected: cf4872cc90538fd4bca761c8db912d8f65e49f2b
uv venv --python 3.10 --seed "$ORA_INSTALL_ROOT/envs/robotwin"
uv pip install --python "$ORA_INSTALL_ROOT/envs/robotwin/bin/python" \
  torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python "$ORA_INSTALL_ROOT/envs/robotwin/bin/python" \
  -r "$ORA_INSTALL_ROOT/sources/RoboTwin/script/requirements.txt" \
  numpy==1.26.4 torchvision==0.19.1 setuptools==69.5.1
git clone https://github.com/NVlabs/curobo.git "$ORA_INSTALL_ROOT/sources/curobo"
git -C "$ORA_INSTALL_ROOT/sources/curobo" checkout --detach d64c4b005459db10c5dd867d8b30a87d5bda9bdb
# CUDA_HOME must select a toolkit compatible with this Torch environment.
uv pip install --python "$ORA_INSTALL_ROOT/envs/robotwin/bin/python" \
  -e "$ORA_INSTALL_ROOT/sources/curobo" --no-build-isolation
uv pip install --python "$ORA_INSTALL_ROOT/envs/robotwin/bin/python" warp-lang==1.12.0
export ROBOTWIN_ROOT="$ORA_INSTALL_ROOT/sources/RoboTwin"
export ROBOTWIN_PYTHON="$ORA_INSTALL_ROOT/envs/robotwin/bin/python"
export CUROBO_ROOT="$ORA_INSTALL_ROOT/sources/curobo"
export CUROBO_PYTHON="$ROBOTWIN_PYTHON"
```

The installer creates a deterministic source commit. Do not edit the planner to bypass the revision gate. The original identity text retained inside the installer is part of that commit's reproducibility.

## Assets

Download `background_texture.zip`, `embodiments.zip` and `objects.zip` from [TianxingChen/RoboTwin2.0](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0), selecting and recording an immutable dataset revision. Training checkpoints and demonstration trajectories are unnecessary. Record archive SHA-256 values, extract all three into `$ROBOTWIN_ROOT/assets`, then regenerate absolute robot paths:

```bash
(cd "$ROBOTWIN_ROOT" && "$ROBOTWIN_PYTHON" script/update_embodiment_config_path.py)
uv run --locked ora doctor --env robotwin
```

The `assets/background_texture/unseen/` pool must be complete. Moving the installation requires regenerating embodiment paths. SAPIEN requires working Vulkan libraries, not merely a visible CUDA GPU.

## Qualify one seed

On your registered `robotwin-gpu0` slot, prepare a bounded scan for one task. This runs the upstream expert, not the language model:

```bash
uv run --locked python -m ora.orchestrator.robotwin_evaluation prepare \
  --robotwin-root "$ROBOTWIN_ROOT" --robotwin-python "$ROBOTWIN_PYTHON" \
  --tasks adjust_bottle --slots robotwin-gpu0 \
  --accepted-per-task 1 --candidate-count 3 --task-config demo_clean
uv run --locked python -m ora.orchestrator.robotwin_evaluation qualify \
  --plan PATH_TO_PRINTED_PLAN --confirm QUALIFY-ROBOTWIN:PRINTED_HASH
```

The immutable plan gives the estimate and confirmation hash. Candidate probes can take minutes; this bounded scan is separate from a full 50-task qualification. The resulting `seed-manifests/adjust_bottle.json` names accepted seeds and binds the source/configuration. Select a seed actually present in it. Reuse still-valid evidence instead of rescanning on every launch. See [CLI reference](../guides/cli-reference.md#robotwin-evaluation-preparation).

## Compose and launch

After qualifying perception services, substitute their report path, the seed manifest and an accepted seed:

```bash
uv run --locked python -m ora.orchestrator.benchmark_spec compose robotwin adjust_bottle \
  --seed ACCEPTED_SEED --task-config demo_clean \
  --qualification PATH_TO_SEED_MANIFEST \
  --experiment robotwin-example --resource-profile robotwin-gpu0 \
  --service-report PATH_TO_READINESS_REPORT --cohort robotwin-example-g001
```

Use the printed path in the [review/launch commands](common.md#run-and-inspect-one-composed-episode). The result contains the official boolean verdict; completion alone does not imply success. `demo_randomized` needs qualification for that configuration. The composer inherits the current default tools/prompts and an 80-decision, 200-model-call, 10,000-sim-step, six-hour ceiling. Actual duration and tokens are reported in the episode evidence.
