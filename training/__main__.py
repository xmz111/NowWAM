"""LIBERO training: tested on 4 x H200 (141 GB VRAM each), using DDP.

Minimum VRAM has not been measured. The tested host had 32 CPUs and 320 GB RAM.
Run from the repository root: torchrun --standalone --nproc-per-node=4 -m training
"""

import argparse
import fcntl
import logging
import os
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from ._core.trainer import Wan22Trainer
from ._core.utils import misc
from ._core.utils.checkpoint_transaction import contract, select
from ._core.utils.logging_config import setup_logging
from .initialization import task_parameters, warmup_factor
from .export import export_policy


class Trainer(Wan22Trainer):
    def _build_scheduler(self, scheduler_type, total_train_steps, warmup_steps=0):
        if scheduler_type != "constant" or warmup_steps != 2110:
            raise ValueError("This recipe uses a constant LR with 2110 warmup updates")
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=warmup_factor)

    def save_checkpoint(self):
        result = super().save_checkpoint()
        if self.accelerator.is_main_process and self.global_step in {
            28140,
            32830,
            37520,
            42210,
        }:
            model = self.accelerator.unwrap_model(self.model)
            ema_path = Path(self.weights_dir) / f"ema_step_{self.global_step:06d}.pt"
            destination = Path(self.output_dir) / "exports" / f"step_{self.global_step:06d}"
            export_policy(
                model, ema_path, Path(self.output_dir) / "dataset_stats.json", destination
            )
        self.accelerator.wait_for_everyone()
        return result


def configuration(data, assets, output):
    cfg = OmegaConf.load(Path(__file__).with_name("libero.json"))
    cfg.output_dir = str(output.resolve())
    cfg.model.flux2_src_path = str(assets.resolve() / "sources/flux2")
    cfg.data.train.dataset_dirs = [
        str(data.resolve() / Path(p).name) for p in cfg.data.train.dataset_dirs
    ]
    for key in (
        "flux2_model_path",
        "ae_model_path",
        "qwen3_model_spec",
        "action_dit_pretrained_path",
    ):
        relative = Path(cfg.model[key]).relative_to("assets")
        cfg.model[key] = str(assets.resolve() / relative)
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=Path("outputs/libero"))
    args = parser.parse_args()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world != 4:
        raise ValueError(
            "Use torchrun --nproc-per-node=4; this recipe requires global batch size 64"
        )
    cfg = configuration(Path("data"), Path("assets"), args.output)
    for path in [
        *cfg.data.train.dataset_dirs,
        cfg.model.flux2_model_path,
        cfg.model.ae_model_path,
        cfg.model.qwen3_model_spec,
        cfg.model.action_dit_pretrained_path,
    ]:
        if not Path(path).exists():
            raise FileNotFoundError(path)
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".checkpoint.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cfg.resume = select(args.output / "checkpoints", contract(cfg, world))
    setup_logging(log_level=logging.INFO, is_main_process=rank == 0)
    misc.register_work_dir(cfg.output_dir)
    if rank == 0:
        OmegaConf.save(cfg, args.output / "config.yaml")
    model = instantiate(cfg.model, model_dtype=torch.float32, device=f"cuda:{local_rank}")
    tensors = task_parameters()
    # Model parameters have a canonical first name and may also have MoT aliases.
    aliases = dict(model.named_parameters(remove_duplicate=False))
    with torch.no_grad():
        for name, value in tensors.items():
            target = aliases[name]
            assert target.shape == value.shape
            target.copy_(value.to(device=target.device, dtype=target.dtype))
    dataset = instantiate(cfg.data.train)
    trainer = Trainer(model, dataset, val_dataset=None, cfg=cfg)
    try:
        trainer.train()
        if rank == 0:
            (args.output / "TRAINING_DONE").touch()
    finally:
        trainer.accelerator.end_training()


if __name__ == "__main__":
    main()
