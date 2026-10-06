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
  <a href="#citation">Citation</a>
</p>

</div>

NowWAM adapts pretrained generative models to robot control through
current-observation denoising, without a separate future-target branch.
This repository provides PyTorch inference and evaluation code.

## Checkpoints

- [Klein LIBERO](https://huggingface.co/xmz111/NowWAM/tree/main/Klein-LIBERO)
- [ZImage LIBERO](https://huggingface.co/xmz111/NowWAM/tree/main/ZImage-LIBERO)
- [Klein RoboCasa](https://huggingface.co/xmz111/NowWAM/tree/main/Klein-RoboCasa-100shot)

## Quick start

Use Python 3.11 on Linux with an NVIDIA GPU. See the [installation guide](SETUP.md) for
system packages and upstream model access requirements.

```bash
git clone https://github.com/xmz111/NowWAM.git
cd NowWAM
python3.11 scripts/install.py Klein-LIBERO
source .venv/bin/activate
python -m eval Klein-LIBERO --limit 4
```

The installer prepares the checkpoint, encoders and simulator. To use another
checkpoint, replace `Klein-LIBERO` in both commands with `ZImage-LIBERO` or
`Klein-RoboCasa-100shot`. Use `all` with the installer to prepare all three.

## Evaluation

Omit `--limit` to evaluate the full benchmark:

```bash
python -m eval Klein-LIBERO
```

For parallel evaluation:

```bash
python -m eval Klein-LIBERO --gpus 0,1 --lanes 2
```

Results are saved under `outputs/<model>`. Re-run the same command to resume.
Short tests use a separate output directory. Each parallel lane holds its own
policy; start with one lane if GPU memory is limited.

## Code

- [Policy](nowwam/policy.py): model loading and action inference.
- [Klein](nowwam/klein.py) / [Z-Image](nowwam/zimage.py): model implementations.
- [LIBERO](eval/libero.py) / [RoboCasa](eval/robocasa.py): simulator adapters.
- [Evaluator](eval/run.py): parallel evaluation and results.
- [Installer](scripts/install.py) / [Setup](SETUP.md): environments and assets.

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

Our implementation builds on [ImageWAM](https://github.com/yuyangalin/ImageWAM).
We thank the authors of [FLUX.2](https://github.com/black-forest-labs/flux2),
[Z-Image](https://github.com/Tongyi-MAI/Z-Image), [Qwen](https://github.com/QwenLM/Qwen3),
[Diffusers](https://github.com/huggingface/diffusers),
[LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus),
[DIAL](https://github.com/xpeng-robotics/DIAL),
[RoboCasa GR1](https://github.com/robocasa/robocasa-gr1-tabletop-tasks), and
[robosuite](https://github.com/ARISE-Initiative/robosuite) for their open-source work.

ImageWAM-derived code retains Copyright (c) 2026 Yuyang "Alice.L" and its
[MIT license](LICENSE). Third-party code, models and assets retain their
respective licenses; NowWAM checkpoint licenses are included on Hugging Face.
