# Isolated RoboDojo installation

The experiment uses Python 3.11.15, Isaac Sim 5.1.0.0, Torch 2.7.0 with CUDA
12.8, and the upstream CuRobo v2 fork. The Harness keeps its own locked
interpreter. Installing these packages does not qualify physics, rendering,
robot control, camera geometry or task verdicts.

Choose an installation root, for example `/path/to/robodojo-runtime`. Its
bindings are:

```sh
export ROBODOJO_ROOT=/path/to/robodojo-runtime/source/RoboDojo
export ROBODOJO_ASSET_ROOT=/path/to/robodojo-runtime/assets/Assets
export ROBODOJO_PYTHON=/path/to/robodojo-runtime/env/bin/python
```

## Fixed sources

| Component | Revision |
|---|---|
| RoboDojo | `ee67a1468510da7624a089164402359f2afc72c8` |
| IsaacLab | `afca7b09d60d8beb9c1cb28b43066499940b969b` |
| CuRobo | `d17b54ce32cba095c0b000c4c58777075d11de0e` |
| XPolicyLab | `432f82b1758c5b1202e42a3dfe014546dbc50871` |
| HF RoboDojo assets | `a14409d7fae673c00499e01fd88b4457df6351b1` |

The asset revision identifies the selected installation data. It does not
claim that an upstream published result used that revision.

## Setup stages

From this experiment's checkout:

```sh
./scripts/setup-robodojo sources
./scripts/setup-robodojo assets
./scripts/setup-robodojo sdk
./scripts/setup-robodojo resolve
./scripts/setup-robodojo packages
./scripts/setup-robodojo bind-assets
./scripts/setup-robodojo doctor
./scripts/setup-robodojo manifest
```

The first three stages are independent and may run concurrently. `packages`
requires both source and SDK completion; asset binding requires source, SDK
and asset completion. Every stage accepts `--root PATH`. Use a permanent
location before binding assets. None of these stages starts a simulator,
policy service, CUDA kernel or GPU qualification job.

The installer redirects UV, pip, Hugging Face, XDG, CUDA, GL and temporary
storage beneath the selected root. It clears inherited Python/conda selection
and uses an explicit independent interpreter. `sdk` first installs/selects
the full CPython base under `python/`, then passes its absolute binary path
to virtual-environment creation. `python-runtime` repeats the path/version
readback and fails if the binary, base prefix, standard library, external SSL extension
or any Python search path resolves outside the runtime root. Its receipt is
`python-runtime.json`. Existing environments, package
caches and Physical Demo installations are left intact. System prerequisites
such as Git LFS, a C++ toolchain and the host graphics driver are inspected
separately; the installer does not invoke sudo or change the host driver.

The initial reference-workstation virtual environment had discovered an
existing Home-based CPython despite the install-directory setting. Before
qualification, the matching complete base was independently copied to the data
volume and all 4,850 files and 1,048 symlinks verified; only this new environment's interpreter link and
`pyvenv.cfg` were repointed. `python-base-provenance.json` records the donor,
base/binary hashes and exact migration, with no donor changes or package
rebuild. Future installs use the explicit base selection described above.

Unlike upstream `install.sh`, this installer updates submodules to gitlinks
without `--remote`. It installs the tracked IsaacLab packages directly,
without optional RL framework extras or editor-configuration mutations.
CuRobo uses its default `cuda.core` runtime compilation backend; the deprecated
`CUROBO_USE_PYBIND=1` build path is not selected.

The source-pinned IsaacLab package requires `starlette==0.49.1`; RoboDojo's
outer installer subsequently forces `0.45.3`. This installation chooses
`0.45.3` to retain RoboDojo and Isaac Sim's FastAPI contract. The installer
uses a single explicit resolver override and records the resulting IsaacLab
`pip check` conflict in `pip-check.json`. Only that exact declared mismatch is
accepted; other conflicts fail installation. This does not claim a clean
`pip check`. No source or installed package metadata is patched. The
headless qualification must establish actual import and execution behavior.
Other simulator pins are retained in the generated
`constraints.txt`. `installation.in` and `installation.lock` record the full resolver input and
solution. The final `pip-freeze.txt` and `packages.json` record installed versions.

## Verified local package reuse

An existing same-version installation can save substantial download time.
Reuse is optional and precedes the `sdk` stage:

```sh
./scripts/setup-robodojo donor-wheels \
  --donor-site-packages /path/to/donor-env/lib/python3.11/site-packages
```

Each selected IsaacSim/Torch/NVIDIA distribution is checked against every
hashed installed RECORD payload. A missing or altered member rejects that
distribution. Passing distributions become independent wheels under the new
root; installed scripts and metadata are recreated by pip. Their new archive
hashes and donor paths are recorded in `donor-wheel-provenance.json`. These
are derived wheels, not byte-identical copies of the original wheel archives.

The inspected reference-workstation donor's Kit cache lacked 17 original Python files under
the bundled `packaging` package. To reconstruct that distribution safely:

```sh
python3 ora/environment/backends/robodojo/installation/rebuild_sdk_wheel.py \
  --root /path/to/robodojo-runtime \
  --donor-site-packages /path/to/donor-env/lib/python3.11/site-packages
```

This reads the exact NVIDIA archive's own RECORD through HTTPS range
requests, verifies every member against that record, and fetches only missing
or changed members. It writes a new independent wheel and provenance that
distinguishes member-hash verification from verification of the original ZIP
hash. The donor remains unchanged. Downloading the original distribution is
the fallback when range requests or member verification fail.

## Asset completeness and generated paths

Asset setup downloads all `Assets/**`, including `Traj`, from the immutable
HF revision. The four directories checked by upstream `init_assets.sh` are
insufficient for all tasks: `imitate_sorting_sequence` loads support-arm
trajectories from `Assets/Traj`. Training data and checkpoints outside
`Assets/` are excluded. A plain sparse clone avoids HF's unreliable partial
clone/promisor fetch path; the exact revision is fetched before checkout.
Cone mode is initialized explicitly so root `.gitattributes` metadata is
present and normal Git LFS filtering remains active. On the reference workstation's Git version,
cone selection uses `sparse-checkout init --cone` before `set Assets`.

The source's `Assets` link points to the independent asset tree. Only the two
untracked `Assets/Robots/{franka,x5}/curobo.yml` files are generated from their
tracked templates. The installer refuses to overwrite a tracked output.
Each generated path is individually ignored in the asset clone's local Git
exclude file; tracked modifications remain visible. `asset-bindings.json`
records the template Git blob, template SHA-256, root substitution and output
SHA-256. Both upstream checkouts must remain clean.

Asset setup verifies cached object hashes, canonical pointers from the exact
tracked `HEAD:Assets/**` blobs, and SHA-256/size of every hydrated Asset path.
It requires a clean asset checkout. `assets-verification.json` records each
check's scope/status, revision, path count and unique object totals;
`verify-assets` repeats this check
without fetching or changing the source selection. Since 2026-10-04 Hugging
Face returns 404 for every `swap_T` evaluation layout at the pinned revision
(162 objects, eval seeds 0-2; `UPSTREAM_PURGED_ASSETS`). A later installation
leaves them as pointers, still checks their pointer blobs and lists each path
and object id under `upstream_purged_unhydrated`; `swap_T` cannot run there,
every other task's assets are fully verified. The raw LFS inventory is
retained as `assets-lfs.txt`; the installation manifest records actual source
and asset revisions. Budget separately for
LFS objects, the working tree, derived wheels, installed packages and caches.
The remote asset total is not a peak local disk budget.

The fixed HF revision also contains seven noncanonical empty-file pointers
under excluded `ckpt/**`. A whole-repository pointer fsck reports them even
with the fetch exclusion; this installation makes no whole-repository fsck
claim. Cache checks use `fsck --objects`, and pointer checks read only exact
tracked Asset blobs, so no Asset pointer error is suppressed. The original
out-of-scope finding is retained in `assets-out-of-scope-pointer-audit.json`.

## Static validation and subsequent qualification

Run the static doctor after package and asset completion:

```sh
./scripts/setup-robodojo doctor
"$ROBODOJO_PYTHON" -m pip check
```

The doctor runs an external copy of upstream `verify_install.sh` with only
its source-root assignment and hard-coded `/tmp` inventory pathname adapted.
Both hashes and exact substitutions are recorded in `doctor-provenance.json`;
the upstream checkout remains unchanged. The independent interpreter is first
in PATH, and Isaac, conda and policy checks are explicitly skipped.
`--skip-conda` alone would still invoke conda for Isaac imports. Those imports
require a separate check through the explicit worker interpreter. A passing
doctor is only a static asset/configuration check.

`install-manifest.json`, package/provenance records, `asset-bindings.json`,
`device-preparation.json` and installation logs stay beneath the independent
installation root. Device inventory records observation time; it grants no
GPU allocation. Physics, 5.1 RTX/RGB-D, first-use CuRobo compilation, native
task control and official verdict tests follow the Runtime lease and exact
source qualification process. Their raw evidence and formal run outputs use
the shared Runtime workspace.

The worker explicitly selects the pinned launcher's actual default,
`apps/isaaclab.python.kit`, with headless mode and cameras enabled. The retained
legacy `isaaclab.python.headless.rendering.kit` is not the default at this
revision and lacks its current RTX initialization settings. The selected
experience file is included in the loaded-source hashes.

The worker disables both previous-dump and new-dump uploads before SDK
startup, preserving local diagnostics. It uses Carbonite's documented
upload-URL, skip-old-upload and preservation controls plus an immutable
episode-local privacy file, then verifies the actual settings before task
initialization. This avoids relying on the selected experience's telemetry
defaults. See the [Carbonite crash-reporting configuration](https://docs.omniverse.nvidia.com/kit/docs/carbonite/167.3/docs/CrashReporter.html).

## Process-local graphics compatibility

The unmodified reference-workstation driver `595.71.05` reports
`maxMemoryAllocationSize = UINT64_MAX`, which causes Isaac Sim 5.1 RTX
scene-database startup to overflow. Three original probes failed before
reset. The [upstream issue and finite-limit workaround](https://github.com/isaac-sim/IsaacSim/issues/568#issuecomment-4415180467)
identify a process-local repair using the official Khronos Vulkan Profiles
layer. No host-driver, SDK version, RoboDojo task or physics change is needed.

The installed layer is built locally under
`/path/to/robodojo-runtime/graphics-compat` from
`KhronosGroup/Vulkan-Profiles@a400d0b10059b2cc3cd285475593eee79975a5c4`
(tag `vulkan-sdk-1.4.341.0`) and its pinned `scripts/known_good.json`
dependencies. Apply the tracked
[core-identity generator patch](vulkan-profiles-core-id.patch)
before building: the generator otherwise omits `VkPhysicalDeviceIDProperties`
when the device supports core Vulkan 1.1 without the instance extension being
listed as a device extension. The patch preserves the driver's actual UUID,
LUID and node-mask values; it does not replace them with profile data.

`scripts/setup-robodojo graphics-compat --root ROOT` performs these steps (opt-in;
needs Git, CMake, a C++17 compiler and network access for `known_good.json`): it
clones the pinned revision under `ROOT/graphics-compat-build/`, applies the patch,
generates with `jsonschema` in a separate build venv, builds only `ProfilesLayer`,
and writes `ROOT/graphics-compat/` with the tracked profile and the receipt.
On fs-mbz (driver 595.91.07, 2026-10-13) the stage's layer passed the readback
and driver gate on all eight GPUs.

`installation.json` records the exact patch, generated source, dependencies,
compiler, local build commands, layer/profile/manifest hashes and verification
receipt. Reproduce its `build.reproduction_commands` in a fresh installation.
Generation must precede the layer compilation; install only the layer target.
The profile contains exactly one override:
`VkPhysicalDeviceMaintenance3Properties.maxMemoryAllocationSize = 4292870144`.

Activation is worker-local: `ROBODOJO_VULKAN_COMPAT_ENABLE=1`, the named
`VP_ROBODOJO_RTX_ALLOCATION_COMPAT` profile, `SIMULATE_PROPERTIES_BIT`, and
`VK_KHRONOS_PROFILES_EMULATE_PORTABILITY=false`. The scoped implicit-layer
manifest uses that explicit enable variable. Its directory is passed through
both `VK_ADD_IMPLICIT_LAYER_PATH` and the worker's `XDG_DATA_DIRS` for the
system and older SDK loaders. Nothing is added to a shell profile, global
Vulkan directory, Python startup hook or SDK source tree.

Before activation, a disposable child checks real properties with and without
the layer, validates installation hashes and rejects any change to the other
recorded physical properties or device identities. Running the query in a
child avoids preloading the system Vulkan loader into the SDK process.
After SDK startup, the worker checks the actual mapped library and unchanged
installation bytes. The source fingerprint includes all four compatibility
file roles, including the tracked patch. Qualification and preflight bind the
same readback; neither treats this repair as a passing simulator qualification.

The local exhaustive verification compared all eight devices through both
the system and Isaac-bundled loaders. Device-property text differed only in
the allocation limit and its Vulkan 1.1 alias; the full GPU-0 formats, queues
and features matched, and disabling the layer restored original properties.
Live scene/reset/control acceptance remains required. The source-bound results
are in `experiments/robodojo_compat/qualification.md` at commit `c20299be`
(the package left the tree on 2026-10-04; read it with
`git show c20299be:experiments/robodojo_compat/qualification.md`).
