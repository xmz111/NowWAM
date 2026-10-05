# Setup

[Overview](README.md) · [Prerequisites](#system-prerequisites) ·
[Install](#install) · [Evaluate](#evaluate-and-resume)

Use Python 3.11 on x86-64 Linux with an NVIDIA GPU and a working CUDA driver.
Run commands from the repository root. Installation downloads several GB of
packages and tens of GB of weights/assets; leave sufficient disk space.
No cluster account or JAX installation is needed.

The model and LIBERO environments use the Torch CUDA 11.8 build;
RoboCasa uses a separate CUDA 12.6 build.

## System prerequisites

On Ubuntu or Debian:

```bash
sudo apt-get update
sudo apt-get install -y git build-essential python3.11-venv libgl1 libegl1 libglib2.0-0 libmagickwand-dev
nvidia-smi
```

Python 3.11 must already be installed; its package availability depends on the
distribution. The installer checks Python, system libraries and the GPU
before downloading model assets. It does not change the system driver, use
sudo, or modify shell startup files.

Klein uses the [FLUX.2 VAE](https://huggingface.co/black-forest-labs/FLUX.2-dev).
Accept its upstream model terms if required and
authenticate with Hugging Face before downloading gated files. An access
denial stops installation; it does not select different weights.

## Install

Choose one checkpoint, or use `all` to prepare all three:

```bash
python3.11 scripts/install.py Klein-LIBERO
# python3.11 scripts/install.py ZImage-LIBERO
# python3.11 scripts/install.py Klein-RoboCasa-100shot
# python3.11 scripts/install.py all
source .venv/bin/activate
```

The installer creates the model environment in `.venv` and prepares only
the simulator environments needed by the selected checkpoints. LIBERO and
RoboCasa use incompatible MuJoCo/robosuite versions, so the evaluator starts
the appropriate interpreter automatically; no manual environment switching
is needed.

It installs packages, checks dependency compatibility, downloads pinned
checkpoints/Qwen/VAEs and simulator assets, applies the visible preparation
patches, records asset hashes, checks simulator resets, and runs a separate
one-case policy smoke evaluation. Any failed step stops installation.

LIBERO checks one reset per suite; RoboCasa checks all 24 tasks at seed 0.
These checks confirm basic startup and observations, not benchmark scores.
The complete fresh-machine CUDA installation is still under validation.

To install packages without downloading weights/assets or running simulation:

```bash
python3.11 scripts/install.py all --dependencies-only
```

After the model environment exists, `.venv/bin/hf auth login` is available
for authentication. Re-run the normal installer to continue.

## What is installed

- `checkpoints/NowWAM/<model>`: EMA policy weights, config and normalization
  statistics from [xmz111/NowWAM](https://huggingface.co/xmz111/NowWAM).
- `assets/Qwen3-4B`: [text encoder and tokenizer](https://huggingface.co/Qwen/Qwen3-4B).
- `assets/flux2/ae.safetensors` or `assets/Z-Image`: matching VAE;
  [Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image) also needs its transformer
  configuration, not base DiT weights.
- `assets/LIBERO-plus`, `assets/DIAL`,
  `assets/robocasa-gr1-tabletop-tasks`: pinned simulator source/assets.
- `assets/osmesa`: isolated LIBERO renderer, installed using the explicit
  package lock in [requirements/osmesa-linux-64.txt](requirements/osmesa-linux-64.txt).
- `assets/<model>-manifest.json`: local asset identity for evaluation/resume.

Download revisions and archive checksums are recorded in [scripts/install.py](scripts/install.py).
Upstream licenses and access conditions apply separately.

FLUX.2's packaging pins differ from this evaluator's dependencies.
`scripts/flux2-dependencies.patch` changes only its Torch, torchvision and
safetensors requirements; its model and VAE source are unchanged. The
installer installs its remaining declared dependencies and runs `pip check`.

LIBERO preparation enables loading trusted upstream initial-state files on
PyTorch 2.6+, creates a local config, and extracts the pinned asset archive
without preserving the producer's parent directories. OSMesa is used only
for LIBERO; RoboCasa uses EGL.

DIAL preparation removes nine interactive debugger calls. RoboCasa uses the
upstream non-fixture asset registries and fills absent public fixture files
separately; existing fixture XML files are preserved, then the two changes in
`scripts/robocasa-fixtures.patch` are applied. No object-sampling patch is
added.

## Evaluate and resume

```bash
python -m eval Klein-LIBERO --limit 4
python -m eval Klein-LIBERO --gpus 0,1 --lanes 2
```

The full command omits `--limit`. Each lane owns a policy and simulator;
start with the default single lane if GPU memory is limited.

Repeat an evaluation command with the same output directory to resume.
Re-running installation reuses matching downloads and checks existing
manifests; it refuses a different source revision, changed protected assets,
or a mismatched archive. Do not rerun installation while an evaluation is
using those environments.

For custom paths, see `python -m eval.run --help`.
