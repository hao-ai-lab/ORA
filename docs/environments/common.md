# Before running a simulator

Complete the [Synthetic quickstart](../quickstart.md) first. ORA's locked environment stays separate from simulator and perception environments. The recipes below target Linux x86_64 with an NVIDIA driver, a compatible CUDA toolkit, Git LFS, a C++ toolchain, FFmpeg and working headless graphics. External packages and assets retain their upstream licenses.

## Register the host and a slot

Copy `examples/hosts/example-host.yaml` to a host-local directory as `<host-id>.yaml`. Change `id`, `execution_marker.required_value` and GPU selectors to match your machine. Export `ORA_HOSTS_DIR` to that directory, `ORA_EXECUTION_HOST` to the ID, and `ORA_PROJECT_ROOT` to the clean ORA checkout. Keep absolute installation paths in your shell or a private `.env`, using the variable names in the template. Do not put credentials in a host profile.

### Workspace and evidence roots

The runtime workspace holds `.harness-runtime/` (resource leases and the run index); evidence goes to the outputs root, `<workspace>/outputs` by default. A linked Git worktree defaults to the workspace of the checkout that owns the common `.git` directory, not to itself, so leases and runs from every worktree meet there. Set `ORA_WORKSPACE_ROOT` (absolute) to choose the workspace explicitly, for example one per node so leases stay node-local, and `ORA_OUTPUTS_ROOT` (an existing absolute directory) to keep evidence from several workspaces in one place. `uv run --locked ora workspace` prints the resolved roots. Read leases only on the machine that runs them.

Declare only slots you can actually run. Each slot binds a backend, a physical GPU and a minimum free-memory floor. An unprofiled run takes a host-wide lease. Start with one renderer per GPU; measure before increasing concurrency. LIBERO's EGL device numbering must be checked against CUDA numbering on the actual host.

## Configure the model

Simulator recipes use a provider profile rather than the quickstart's `ORA_*` convenience variables. Inspect available profiles with:

```bash
uv run --locked ora providers --json
uv run --locked ora baselines --explain
```

The selected default profile is `deepseek_flash_openai_low_16k_v1`. It reads `DEEPSEEK_LOW_API_KEY`, `DEEPSEEK_BASE_URL` and `DEEPSEEK_MODEL_ID`; URL and model must equal its registration. To use another service, copy a profile from `ora/agent/providers/builtin-profiles.yaml` into a credential-free catalog with schema `ora.native-provider-catalog.v1`, change its endpoint/model, and pass `--provider-profile ID --provider-catalog PATH` to the composer. Name an environment variable in `api_key_env`; store the value only in your shell. HTTPS is required except for loopback HTTP. Set `SSL_CERT_FILE` when your service uses a private CA.

Preflight probes a live robot provider's text, image, tool-call and continuation support. This can incur API usage. A provider registration alone is not qualification. See [configuration](../reference/configuration-and-evidence.md).

## Install the shared perception services

Default simulator surfaces use SAM3 and Contact-GraspNet. They are separate resident services, not dependencies of the Synthetic quickstart. Choose a permanent writable installation root:

```bash
export ORA_INSTALL_ROOT=/srv/ora
mkdir -p "$ORA_INSTALL_ROOT/sources" "$ORA_INSTALL_ROOT/envs" "$ORA_INSTALL_ROOT/models"
git clone https://github.com/capgym/cap-x.git "$ORA_INSTALL_ROOT/sources/cap-x"
git -C "$ORA_INSTALL_ROOT/sources/cap-x" checkout --detach 53e9966d7a8e2fa7494676772bccc35280f5c0ed
git -C "$ORA_INSTALL_ROOT/sources/cap-x" submodule update --init --recursive \
  capx/third_party/sam3 capx/third_party/contact_graspnet_pytorch
uv venv --python 3.11 --seed "$ORA_INSTALL_ROOT/envs/perception"
uv pip install --python "$ORA_INSTALL_ROOT/envs/perception/bin/python" \
  torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$ORA_INSTALL_ROOT/envs/perception/bin/python" \
  numpy==1.26.4 scipy==1.16.1 setuptools==78.1.0 fastapi==0.116.1 uvicorn==0.35.0 \
  timm==1.0.28 einops==0.8.1 decord==0.6.0 pycocotools==2.0.10 omegaconf==2.3.0 \
  iopath==0.1.10 ftfy==6.1.1 pyrender==0.1.45 trimesh==4.7.4 pillow==12.2.0 \
  pyyaml==6.0.2 huggingface-hub==1.27.0 safetensors==0.8.0 \
  opencv-python-headless==4.11.0.86 psutil==7.2.2 regex==2026.9.10
```

The submodule pins are SAM3 `6fe87d64a5beb9084923d7a9e002741178635b09` and Contact-GraspNet `da3dcfb2f53e43b186083ee4a9d1e232f73efc98`. Obtain `sam3.pt` through the official [facebook/sam3](https://huggingface.co/facebook/sam3) access and license process; place it under `models/`. The service checks these SHA-256 values:

| File | SHA-256 |
|---|---|
| SAM3 `sam3.pt` | `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e` |
| GraspNet `checkpoints/contact_graspnet/checkpoints/model.pt` | `39fc3439d5814043ba64e0715c127c2e8aca6ea376c22614e11af1bc89317762` |
| GraspNet `checkpoints/contact_graspnet/config.yaml` | `7ae6cc15c726e8ec2c46b0c799d58ca95caeb9ce9f261aaaf3e467712df76ac5` |

Copy the `B2_SAM_*` and `B2_GRASP_*` bindings in `.env.example` into your private host environment. Point them to this interpreter, checkout and weights; choose physical GPU indices and unused loopback ports. These environment-variable names are retained by the service operator.

```bash
./scripts/model-services start
./scripts/model-services status
./scripts/model-services qualify
```

`status` remains nonzero during model loading. `qualify` prints a content-addressed readiness report; pass that path to the episode composer. It checks wire/model readiness, not grasp quality, and expires after 24 hours or a service restart. Do not restart services with active leases. Source, weights and configuration hashes must match the current adapter.

Download size, VRAM and installation time depend on your cache, device and upstream assets; no universal minimum has been measured. Inspect available capacity before downloading or allocating slots. Keep installation manifests and generated qualification outside Git.

## Run and inspect one composed episode

Composers print the RunSpec path. Review it before launching:

```bash
uv run --locked ora plan PATH_TO_SPEC
uv run --locked ora preflight PATH_TO_SPEC
uv run --locked ora launch PATH_TO_SPEC
uv run --locked ora status RUN_ID
uv run --locked ora watch RUN_ID
./scripts/open-viewer
```

`plan` and `launch` print the plan's `confirmation code` (the last six hex digits of its `plan sha256`); launch asks you to type it. Evidence lands under `outputs/<experiment>/runs/<run_id>/` (or the selected outputs root). A landed run records `result.json`, evaluation and a seal or seal-refusal reason. Read the official verdict and infrastructure status separately. The robot episode budgets are ceilings, not expected duration; provider speed, task length and retries determine elapsed time and token usage.

`ora launch` and `scripts/model-services start` start detached processes and return. Those processes must outlive the shell that started them. Under a batch scheduler such as Slurm, every process of a job step (`srun`) is killed when the step ends, detached or not: keep the step alive until the run lands (for example, run the launcher in `tmux` inside one long-lived `srun --overlap` step that waits for the supervisor PID in the run's `process.json`), and keep the services' step alive for as long as runs use them. Stop a run with `ora stop RUN_ID` before ending its step; killing the step leaves the run unlanded.

| Problem | Check |
|---|---|
| Missing interpreter or module | Use the explicit isolated interpreter; run `ora doctor --env ENV --json`. |
| EGL/Vulkan initialization | Install the driver's graphics libraries; verify the device mapping, not just `nvidia-smi`. |
| Asset/source mismatch | Restore the pinned source and verified asset tree; do not suppress preflight. |
| Provider 401, 429 or tool/vision failure | Check profile variable names, exact endpoint/model and service capability. Never paste the key into an issue. |
| Context overflow | Check the provider's real context limit and output reservation in the profile. |
| Episode appears stuck | Inspect `ora status`, `ora watch` and the exact run's supervisor log; use `ora stop RUN_ID` for that run. |
