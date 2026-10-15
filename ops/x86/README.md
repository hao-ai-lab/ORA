# x86_64 worker environment recipes

Direct-dependency recipes for the isolated LIBERO-Pro worker and the resident
perception services on a Linux x86_64 NVIDIA host, pinned by version. Torch and
Torchvision are installed separately from the matching CUDA wheel index, and
neither file is a portable transitive lock: freeze the resolved environment on
your host. `ora setup --env libero|perception` installs them; the
[single-GPU guide](../../docs/getting-started/single-gpu.md) lists the explicit
commands.
