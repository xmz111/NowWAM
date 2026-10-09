"""RoboCasa 100-shot training: tested on 4 x H200 (141 GB each), using DDP.

Run: torchrun --standalone --nproc-per-node=4 -m training.robocasa
"""

import argparse
import fcntl
import logging
import os
from pathlib import Path
import shutil
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.nn import functional as F

from training.__main__ import Trainer as LiberoTrainer, configuration
from training._core.trainer import Wan22Trainer
from training._core.models.backbones.imagewam import ImageWAM
from training._core.utils import misc
from training._core.utils.checkpoint_transaction import contract, select
from training._core.utils.logging_config import setup_logging
from training.export import ema_policy_state


def task_parameters():
    # Preserve the pretrained action bias and the fixed head initialization.
    base_rng = np.random.default_rng(1234)
    base_rng.uniform(-1 / np.sqrt(7), 1 / np.sqrt(7), size=(7, 1024))
    bias = base_rng.uniform(-1 / np.sqrt(7), 1 / np.sqrt(7), size=1024).astype(np.float32)
    modulation = base_rng.uniform(-1 / np.sqrt(1024), 1 / np.sqrt(1024),
                                  size=(1024, 2048)).astype(np.float32)
    rng = np.random.default_rng(42)

    def weight(fi, fo):
        return (rng.standard_normal((fi, fo)) / np.sqrt(fi)).astype(np.float32).T

    values = {
        "mot.mixtures.action.action_encoder.weight": weight(47, 1024),
        "mot.mixtures.action.action_encoder.bias": bias,
        "mot.mixtures.action.head.linear.weight": weight(1024, 47),
        "mot.mixtures.action.head.adaLN_modulation.1.weight": modulation.T,
        "proprio_encoder.weight": weight(76, 7680),
        "proprio_encoder.bias": np.zeros(7680, np.float32),
    }
    return {name: torch.from_numpy(np.ascontiguousarray(value)) for name, value in values.items()}


def flow_loss(self, sample, tiled=False):
    inputs = self.build_inputs_flux2(sample, tiled=tiled)
    video = inputs["target_latent"].to(torch.bfloat16)
    action = inputs["action"].to(torch.bfloat16)
    inputs["text_hidden_states"] = inputs["text_hidden_states"].to(torch.bfloat16)

    def corrupt(clean, scheduler):
        noise = torch.randn_like(clean)
        t = scheduler.sample_training_t(len(clean), clean.device, torch.float32)
        sigma = (t / scheduler.num_train_timesteps).view(-1, 1, 1)
        noisy = ((1 - sigma) * clean.float() + sigma * noise.float()).to(torch.bfloat16)
        return noisy, noise - clean, t

    noisy_video, target_video, video_t = corrupt(video, self.train_video_scheduler_dift)
    noisy_action, target_action, action_t = corrupt(action, self.train_action_scheduler)
    ids = inputs["target_img_ids"]
    video_pre = self.video_expert.pre_dit(
        x=noisy_video[:, :0], timestep=video_t / 1000,
        context=inputs["text_hidden_states"], context_mask=inputs["text_attention_mask"],
        ref_image_hidden_states=noisy_video, target_img_ids=ids[:, :0], ref_img_ids=ids,
    )
    action_pre = self.action_expert.pre_dit(action_tokens=noisy_action, timestep=action_t / 1000)
    mask = self._build_mot_attention_mask_flux2(
        batch_size=len(video), txt_len=int(video_pre["txt_len"]), target_len=0,
        cond_len=int(video_pre["cond_len"]), action_len=16, device=video.device,
        text_attention_mask=video_pre["text_mask"],
    )
    tokens = self.mot(
        embeds_all={"video": video_pre["tokens"], "action": action_pre["tokens"]},
        attention_mask=mask, freqs_all={"video": video_pre["freqs"]},
        context_all={"video": None, "action": {"ids": action_pre["ids"]}},
        t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
    )
    pred_video = self.video_expert.post_dit_ref(tokens["video"], video_pre)
    pred_action = self.action_expert.post_dit(tokens["action"], action_pre)
    video_loss = F.mse_loss(pred_video.float(), target_video.float())
    action_loss = self._compute_action_loss_per_sample(
        pred_action=pred_action, target_action=target_action,
        action_is_pad=inputs["action_is_pad"], action_dim_is_pad=None,
    )
    action_loss = (action_loss * self.train_action_scheduler.training_weight(action_t)).mean()
    loss = .5 * video_loss + action_loss
    return loss, {"loss_video": float((.5 * video_loss).detach()),
                  "loss_action": float(action_loss.detach())}


class RoboCasaModel(ImageWAM):
    _training_loss_flux2 = flow_loss


def config(data, assets, output):
    data, assets = Path(data).resolve(), Path(assets).resolve()
    cfg = configuration(data, assets, output)
    action_config = OmegaConf.to_container(cfg.model.action_dit_config, resolve=True)
    action_config["action_dim"] = 47
    cfg.model = {
        "_target_": "training.robocasa.RoboCasaModel.from_flux2_klein_pretrained",
        "flux2_model_path": str(assets / "FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors"),
        "ae_model_path": str(assets / "FLUX.2-dev/ae.safetensors"),
        "flux2_src_path": str(assets / "sources/flux2"),
        "qwen3_model_spec": str(assets / "Qwen3-4B"),
        "action_dit_pretrained_path": str(assets / "action_init.pt"),
        "variant": "klein-base-4b", "qwen_context_len": 128,
        "load_text_encoder": False, "proprio_dim": 76,
        "action_dit_config": action_config, "keep_double": 5, "keep_single": 20,
        "dift_mode": True, "action_regression": False, "dift_tshift": .25,
        "video_train_shift": 5., "video_infer_shift": 5.,
        "action_train_shift": 5., "action_infer_shift": 5.,
        "loss_lambda_video": .5, "loss_lambda_action": 1.,
        "mot_checkpoint_mixed_attn": True, "mot_gqa_implementation": "repeat",
        "mot_force_flash_attention": False, "pack_proprio_after_text": True,
        "frozen_qwen_dtype": "bf16", "frozen_vae_dtype": "bf16",
    }
    cfg.data.train = {"_target_": "training.robocasa_data.RoboCasaDataset",
                      "data": str(data), "text_cache": str(data / "text_cache")}
    cfg.eval_num_inference_steps = 4
    cfg.save_epoch_weights = False
    return cfg


def build_model(cfg, device):
    torch.manual_seed(42)
    arguments = OmegaConf.to_container(cfg.model, resolve=True)
    arguments.pop("_target_")
    arguments.update(frozen_qwen_dtype=torch.bfloat16, frozen_vae_dtype=torch.bfloat16)
    model = RoboCasaModel.from_flux2_klein_pretrained(
        **arguments, torch_dtype=torch.float32, device=device
    )
    model.mot.checkpoint_flux2_blocks = True
    aliases = dict(model.named_parameters(remove_duplicate=False))
    with torch.no_grad():
        for name, value in task_parameters().items():
            target = aliases[name]
            if target.shape != value.shape:
                raise ValueError(f"Task initialization shape mismatch: {name}")
            target.copy_(value.to(device=target.device, dtype=target.dtype))
    return model


class Trainer(Wan22Trainer):
    _build_scheduler = LiberoTrainer._build_scheduler

    def save_checkpoint(self):
        result = super().save_checkpoint()
        if self.accelerator.is_main_process:
            model = self.accelerator.unwrap_model(self.model)
            ema = torch.load(Path(self.weights_dir) / f"ema_step_{self.global_step:06d}.pt",
                             map_location="cpu", mmap=True, weights_only=True)
            destination = Path(self.output_dir) / "exports" / f"step_{self.global_step:06d}"
            destination.mkdir(parents=True, exist_ok=True)
            temporary = destination / "model.pth.partial"
            torch.save(ema_policy_state(model, ema), temporary)
            os.replace(temporary, destination / "model.pth")
            shutil.copyfile(Path(self.output_dir) / "dataset_stats.json",
                            destination / "dataset_stats.json")
            source = Path(__file__).with_name("robocasa_eval.json")
            shutil.copyfile(source, destination / "config.json")
            print(f"Saved EMA: {destination}", flush=True)
        self.accelerator.wait_for_everyone()
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=Path("outputs/robocasa"))
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        raise ValueError("Use four DDP workers for global batch 64")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = config(Path("data/robocasa"), Path("assets"), output)
    local_rank, rank = int(os.environ["LOCAL_RANK"]), int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    with (output / ".checkpoint.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cfg.resume = select(output / "checkpoints", contract(cfg, 4))
    setup_logging(log_level=logging.INFO, is_main_process=rank == 0)
    misc.register_work_dir(str(output))
    if rank == 0:
        OmegaConf.save(cfg, output / "config.yaml")
        shutil.copyfile(Path(cfg.data.train.data) / "global_stats.json", output / "dataset_stats.json")
    model = build_model(cfg, f"cuda:{local_rank}")
    dataset = instantiate(cfg.data.train)
    trainer = Trainer(model, dataset, val_dataset=None, cfg=cfg)
    try:
        trainer.train()
        if rank == 0:
            (output / "TRAINING_DONE").touch()
    finally:
        trainer.accelerator.end_training()


if __name__ == "__main__":
    main()
