# LIBERO / LIBERO-Pro: install and run one episode

Read [shared setup](common.md) first. Both base LIBERO suites and the Pro perturbations use the direct LIBERO driver with the Panda robot. Official LIBERO termination supplies the verdict. The default surface uses CuRobo planning, RGB-D perception and overhead footprint localization.

## Source, data and interpreter

```bash
export ORA_INSTALL_ROOT=/srv/ora
mkdir -p "$ORA_INSTALL_ROOT/sources" "$ORA_INSTALL_ROOT/envs"
git clone https://github.com/Zxy-MLlab/LIBERO-PRO.git "$ORA_INSTALL_ROOT/sources/LIBERO-PRO"
git -C "$ORA_INSTALL_ROOT/sources/LIBERO-PRO" checkout --detach 2b910b5b5f53016bef9907632f6f840f1ce2229c
GIT_LFS_SKIP_SMUDGE=1 git clone --no-checkout \
  https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro "$ORA_INSTALL_ROOT/sources/LIBERO-Pro-data"
git -C "$ORA_INSTALL_ROOT/sources/LIBERO-Pro-data" checkout --detach c86fc3b8293185a6f373677018ff3e37f8391602
uv venv --python 3.11 --seed "$ORA_INSTALL_ROOT/envs/libero"
uv pip install --python "$ORA_INSTALL_ROOT/envs/libero/bin/python" \
  torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python "$ORA_INSTALL_ROOT/envs/libero/bin/python" \
  numpy==1.26.4 mujoco==3.8.1 robosuite==1.4.1 gym==0.26.2 gymnasium==0.29.1 \
  bddl==3.6.0 einops==0.8.1 cloudpickle==3.1.1 pyyaml==6.0.2 pillow==12.2.0 \
  imageio==2.37.4 imageio-ffmpeg==0.6.0 opencv-python==4.11.0.86 scipy==1.16.1 \
  h5py==3.14.0 termcolor==3.1.0 matplotlib==3.11.1 packaging==26.2 easydict==1.13
export LIBERO_PRO_PYTHON="$ORA_INSTALL_ROOT/envs/libero/bin/python"
export LIBERO_PRO_ROOT="$ORA_INSTALL_ROOT/sources/LIBERO-PRO"
export LIBERO_PRO_DATA_ROOT="$ORA_INSTALL_ROOT/sources/LIBERO-Pro-data"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
```

The base suites read BDDL/init-state files from the code checkout; Pro reads the pinned dataset. The task definitions and init-state files are required, while demonstration downloads are skipped. Avoid global `LIBERO_TYPE`, `LIBERO_CONFIG_PATH` and `PYTHONPATH`; ORA creates its own isolated path map. The separate worker needs neither RLinf nor a training framework.

Install CuRobo `d64c4b005459db10c5dd867d8b30a87d5bda9bdb` in its own compatible Torch/CUDA interpreter using the [RoboTwin planner recipe](robotwin.md#source-interpreter-and-planner). Set `CUROBO_ROOT` and `CUROBO_PYTHON`. That planner environment can be shared; it does not require installing RoboTwin when used only for LIBERO.

```bash
uv run --locked ora doctor --env libero_pro
```

`doctor` checks package discovery and directories. It does not qualify rendering. The worker's render guard rejects corrupted RGB/depth; check EGL/CUDA device correspondence before allocating concurrent slots. Start with one LIBERO renderer per GPU and keep its graphics selector explicit.

### Software-rendered (Mesa) slots

NVIDIA EGL tolerates one LIBERO renderer per GPU, and on some drivers a SAPIEN (RoboTwin) process anywhere on the same machine corrupts its frames. To pack several LIBERO episodes on a host, or to run LIBERO beside RoboTwin, render on the CPU with Mesa llvmpipe instead. Declare a slot whose `worker_environment` selects Mesa's EGL vendor:

```yaml
  libero-mesa-gpu0:
    runtime_kind: libero_pro
    resource_class: libero_pro_sim
    slot_id: lpm-gpu0-s0
    worker_environment:
      CUDA_VISIBLE_DEVICES: "0"          # CuRobo's GPU; robosuite also needs the EGL device listed here
      MUJOCO_EGL_DEVICE_ID: "0"
      MUJOCO_GL: egl
      PYOPENGL_PLATFORM: egl
      __EGL_VENDOR_LIBRARY_FILENAMES: /usr/share/glvnd/egl_vendor.d/50_mesa.json
      LIBGL_ALWAYS_SOFTWARE: "1"
      LP_NUM_THREADS: "2"
    minimum_free_memory_mib: 2048
```

Mesa exposes one EGL device, numbered 0, and robosuite requires `MUJOCO_EGL_DEVICE_ID` to appear in `CUDA_VISIBLE_DEVICES`. A slot whose planner runs on GPU K therefore lists `"K,0"`, and lease admission also checks the slot's free-memory floor on GPU 0. Each lane costs CPU (about two cores at `LP_NUM_THREADS=2`) rather than GPU rendering. Mesa and NVIDIA images differ slightly (depth matches; RGB does not bit-match), so keep one renderer per scored cohort and do not pool their results.

## Compose and launch

Register a `libero-gpu0` slot and obtain a [perception readiness report](common.md#install-the-shared-perception-services). For a Pro object-swap task:

```bash
uv run --locked python -m ora.orchestrator.benchmark_spec compose libero_pro libero_object_swap 0 \
  --seed 0 --experiment libero-example --resource-profile libero-gpu0 \
  --service-report PATH_TO_READINESS_REPORT --cohort libero-example-g001
```

For the original LIBERO Spatial suite:

```bash
uv run --locked python -m ora.orchestrator.benchmark_spec compose libero libero_spatial 0 \
  --seed 0 --experiment libero-base-example --resource-profile libero-gpu0 \
  --service-report PATH_TO_READINESS_REPORT --cohort libero-base-example-g001
```

Use the printed path in the [review/launch commands](common.md#run-and-inspect-one-composed-episode). The source-bound task index and init-state selection are resolved by the composer. The run records the official verdict and the source/data pins. These recipes inherit the current default and an 80-decision, 200-model-call, 10,000-sim-step, six-hour ceiling; they do not promise a particular task success rate. Actual runtime and token use appear in the evidence.

If MuJoCo reports missing EGL, check driver graphics libraries and the slot's `MUJOCO_EGL_DEVICE_ID`. If the wrong LIBERO imports, clear inherited package selectors before starting the isolated worker. A data-path error should be fixed in the installation, not by weakening the source gate.
