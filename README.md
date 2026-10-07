<div align="center">

<h1>NowWAM</h1>

<h3>Beyond Future Prediction:<br>Denoising as Generative Adaptation for Robot Control</h3>

<p>Zanyi Wang, Yuheng Lei, Dengyang Jiang, Ping Luo, Mengdi Wang, Zhixuan Liang, and Shilong Liu</p>

<p>
  <a href="https://arxiv.org/abs/2609.28339"><img src="https://img.shields.io/badge/arXiv-2609.28339-b31b1b" alt="Paper on arXiv"></a>
  <a href="https://xmz111.github.io/NowWAM/"><img src="https://img.shields.io/badge/Project-NowWAM-2563eb" alt="Project page"></a>
  <a href="https://huggingface.co/xmz111/NowWAM"><img src="https://img.shields.io/badge/Hugging_Face-Weights-ffcc4d" alt="Model weights"></a>
</p>

<p>
  <a href="https://arxiv.org/pdf/2609.28339">Paper PDF</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#evaluation">Evaluation</a> ·
  <a href="#training">Training</a> ·
  <a href="#citation">Citation</a>
</p>

</div>

NowWAM adapts pretrained generative models to robot control through
current-observation denoising, without a separate future-target branch.
We provide model weights, training and evaluation code.

📦 [Klein weights](#checkpoints) · 🏋️ [LIBERO training](#training) · 🤖 [LIBERO-Plus & RoboCasa evaluation](#evaluation)

## Checkpoints

- [Klein LIBERO](https://huggingface.co/xmz111/NowWAM/tree/main/Klein-LIBERO)
- [Klein RoboCasa](https://huggingface.co/xmz111/NowWAM/tree/main/Klein-RoboCasa-100shot)

## Quick start

Use Python 3.11 on x86-64 Linux with an NVIDIA GPU and a working CUDA driver.
On Ubuntu/Debian, install the system packages first:

```bash
sudo apt-get update
sudo apt-get install -y git build-essential python3.11-venv ffmpeg libgl1 libegl1 libglib2.0-0 libmagickwand-dev
```

Accept the model access terms for the [FLUX.2 VAE](https://huggingface.co/black-forest-labs/FLUX.2-dev)
if required, then install and authenticate:

```bash
git clone https://github.com/xmz111/NowWAM.git
cd NowWAM
python3.11 scripts/install.py Klein-LIBERO --dependencies-only
.venv/bin/hf auth login
python3.11 scripts/install.py Klein-LIBERO
source .venv/bin/activate
python -m eval Klein-LIBERO --limit 4
```

The installer downloads the weights and assets, prepares the simulator, and
runs a one-episode test. Replace `Klein-LIBERO` with `Klein-RoboCasa-100shot`
to evaluate RoboCasa. Use `all` with the installer to prepare both checkpoints;
evaluate each by name. Training code currently covers Klein on LIBERO.
Simulator environments are selected automatically. Do not reinstall during a run.

## Evaluation

Full evaluation, with optional parallel lanes:

```bash
python -m eval Klein-LIBERO
# Or use two GPUs with two lanes each:
python -m eval Klein-LIBERO --gpus 0,1 --lanes 2
```

Results are saved under `outputs/<model>`. Re-run the same command to resume.
Short tests use a separate output directory. Each parallel lane holds its own
policy; start with one lane if GPU memory is limited.
Keep the installed environment and assets unchanged while resuming.
For results from an older evaluator, use a new directory with `--output outputs/new-run`.

## Training

Run from the repository root. Use four GPUs with DDP (tested on 4 × H200,
141 GB each) and allow 300 GB for assets and checkpoints. Keep training in
its own environment; do not run `pip install -e .` inside it.
Accept any required [base-model access terms](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B)
and the VAE terms above before downloading.

```bash
python3.11 -m venv .venv-train
source .venv-train/bin/activate
pip install -r requirements/train.txt
hf auth login
```

Download the [LIBERO v2.1 training data](https://huggingface.co/datasets/yuanty/LIBERO-fastwam)
and base models once:

<details>
<summary>Download commands</summary>

```bash
hf download yuanty/LIBERO-fastwam --repo-type dataset \
  --revision ee018b997c430bb12b5bf3c892d744798c5a2f91 \
  --include 'libero_*_no_noops_lerobot.tar.gz' --local-dir data/archives
for suite in spatial object goal 10; do
  tar -xzf "data/archives/libero_${suite}_no_noops_lerobot.tar.gz" -C data
done

hf download black-forest-labs/FLUX.2-klein-base-4B flux-2-klein-base-4b.safetensors \
  --revision a3b4f4849157f664bdbc776fd7453c2783562f4d --local-dir assets/FLUX.2-klein-base-4B
hf download black-forest-labs/FLUX.2-dev ae.safetensors \
  --revision 26afe3a78bb242c0a8bb181dcc8937bb16e5c66c --local-dir assets/FLUX.2-dev
hf download Qwen/Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --include '*.json' '*.safetensors' 'merges.txt' 'vocab.json' 'tokenizer.model' --local-dir assets/Qwen3-4B

[ -d assets/sources/flux2/.git ] || git clone --no-checkout https://github.com/black-forest-labs/flux2.git assets/sources/flux2
git -C assets/sources/flux2 checkout --detach 50fe5162777813d869182b139e83b10743caef15
python -m training.prepare
```

</details>

Train with the recipe in [training/libero.json](training/libero.json):

```bash
OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  torchrun --standalone --nproc-per-node=4 -m training
```

Repeat the training command to resume, or append `outputs/new-run` for a new run.

EMA exports are saved under `outputs/libero/exports/`. After the evaluation
setup above, evaluate an export using `.venv`, not `.venv-train`:

```bash
.venv/bin/python -m eval outputs/libero/exports/step_032830 --gpus 0,1 --lanes 2
```

## Citation

```bibtex
@article{wang2026nowwam,
  title   = {Beyond Future Prediction: Denoising as Generative Adaptation for Robot Control},
  author  = {Wang, Zanyi and Lei, Yuheng and Jiang, Dengyang and Luo, Ping and Wang, Mengdi and Liang, Zhixuan and Liu, Shilong},
  journal = {arXiv preprint arXiv:2609.28339},
  year    = {2026}
}
```

## Acknowledgements

We thank FastWAM and [ImageWAM](https://github.com/yuyangalin/ImageWAM) for their open-source work, and TRC for compute support.
