"""Klein policy loading and action prediction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prefix_mask(text_mask, image_len, action_len):
    batch, text_len = text_mask.shape
    prefix = text_len + image_len
    total = prefix + action_len
    mask = torch.ones(batch, total, total, dtype=torch.bool, device=text_mask.device)
    mask[:, :prefix, prefix:] = False
    mask[:, :, :text_len] &= text_mask[:, None, :]
    return {"double_joint": mask, "single": mask.clone()}


class Policy:
    """Single-observation policy. Action noise is explicit, never monkeypatched."""

    def __init__(
        self,
        checkpoint,
        *,
        vae,
        text_encoder,
        device="cuda",
        text_device="cpu",
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.root = Path(checkpoint)
        self.config = json.loads((self.root / "config.json").read_text())
        self.stats = json.loads((self.root / "dataset_stats.json").read_text())
        self.device, self.dtype = torch.device(device), torch.bfloat16
        if self.device.type != "cuda":
            raise ValueError("The scored inference path requires a CUDA device")
        self.backbone = self.config["backbone"]
        self.is_libero = self.config["action_dim"] == 7
        self._text_cache = {}
        cfg = self.config
        if self.backbone == "klein":
            from flux2.autoencoder import AutoEncoder, AutoEncoderParams
            from flux2.model import Flux2, Klein4BParams
            from safetensors.torch import load_file

            from .klein import ActionDiTFlux2, Flux2VideoExpert, KleinMoT

            with torch.device("meta"):
                transformer = Flux2(
                    Klein4BParams(
                        depth=cfg["double_blocks"], depth_single_blocks=cfg["single_blocks"]
                    )
                )
                video = Flux2VideoExpert(transformer, "klein-base-4b")
                action = ActionDiTFlux2(
                    action_dim=cfg["action_dim"],
                    hidden_dim=cfg["action_hidden_dim"],
                    num_layers_double=cfg["double_blocks"],
                    num_layers_single=cfg["single_blocks"],
                )
                self.mot = KleinMoT(
                    video,
                    action,
                    fp32_residual=cfg.get("jax_fp32_residual", False),
                    explicit_attention=cfg.get("jax_naive_attention", False),
                )
                self.proprio = nn.Linear(cfg["proprio_dim"], 7680)
                self.vae = AutoEncoder(AutoEncoderParams())
            vae_state = load_file(str(vae))
            # Encoder-only archives omit the unused decoder, not encoder tensors.
            if not any(key.startswith("decoder.") for key in vae_state):
                del self.vae.decoder
            self.vae.load_state_dict(vae_state, strict=True, assign=True)
            text_class = AutoModelForCausalLM
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")
        state = torch.load(
            self.root / "model.pth", map_location="cpu", mmap=True, weights_only=True
        )
        if set(state) != {"mot", "proprio_encoder"}:
            raise ValueError("Expected only mot and proprio_encoder checkpoint groups")
        self.mot.load_state_dict(state["mot"], strict=True, assign=True)
        self.proprio.load_state_dict(state["proprio_encoder"], strict=True, assign=True)
        for module in (self.mot, self.proprio, self.vae):
            module.to(device=self.device, dtype=self.dtype).eval().requires_grad_(False)
        self.video, self.action = self.mot.mixtures["video"], self.mot.mixtures["action"]
        self.text = (
            text_class.from_pretrained(text_encoder, torch_dtype=self.dtype).to(text_device).eval()
        )
        self.text.requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder)

    @torch.inference_mode()
    def encode_text(self, instruction):
        if instruction not in self._text_cache:
            rendered = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": PROMPT.format(task=instruction)}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.config["qwen_thinking"],
            )
            encoded = self.tokenizer(
                rendered, return_tensors="pt", padding="max_length", truncation=True, max_length=128
            ).to(self.text.device)
            result = self.text(**encoded, output_hidden_states=True, use_cache=False)
            hidden = torch.cat(
                [result.hidden_states[i] for i in self.config["qwen_hidden_layers"]], dim=-1
            )
            # Cache on CPU: thousands of instructions must not accumulate on GPU.
            if len(self._text_cache) >= 32:
                self._text_cache.pop(next(iter(self._text_cache)))
            self._text_cache[instruction] = (hidden.cpu(), encoded.attention_mask.bool().cpu())
        return tuple(t.to(self.device) for t in self._text_cache[instruction])

    def append_proprio(self, context, mask, state):
        token = self.proprio(state.to(self.device, context.dtype).reshape(1, 1, -1))
        valid = mask.sum(dim=1)
        indices = torch.where(
            mask, mask.cumsum(dim=1) - 1, valid[:, None] + 1 + (~mask).cumsum(dim=1) - 1
        )
        packed = context.new_zeros(1, context.shape[1] + 1, context.shape[2])
        packed.scatter_(1, indices[:, :, None].expand_as(context), context)
        packed[torch.arange(1, device=self.device), valid] = token[:, 0]
        packed_mask = torch.arange(packed.shape[1], device=self.device)[None] <= valid[:, None]
        return packed, packed_mask

    def normalize_state(self, state):
        state = torch.as_tensor(np.asarray(state).copy(), dtype=torch.float32)
        if state.shape != (self.config["proprio_dim"],) or not torch.isfinite(state).all():
            raise ValueError("Invalid proprioception")
        if self.is_libero:
            scale, offset = self.minmax("state")
            state = state * scale + offset
        else:
            mean = torch.tensor(self.stats["proprio"]["mean"], dtype=torch.float32)
            std = torch.tensor(self.stats["proprio"]["std"], dtype=torch.float32)
            idx = [0, 1, 2, 9, 10, 11]
            state[idx] = (state[idx] - mean[idx]) / (std[idx] + 1e-8)
        return state.clamp(-5, 5)

    def minmax(self, section):
        stats = self.stats[section]["default"]
        low = torch.tensor(stats["global_min"], dtype=torch.float32)
        high = torch.tensor(stats["global_max"], dtype=torch.float32)
        extent = high - low
        ignored = extent < 1e-4
        extent[ignored] = 2.0
        scale = 2.0 / extent
        offset = -1.0 - scale * low
        offset[ignored] = -low[ignored]
        return scale, offset

    def noise(self, seed):
        shape = (1, self.config["action_horizon"], self.config["action_dim"])
        if self.config["action_noise_rng"] == "torch.Generator(cpu)":
            noise = torch.randn(
                shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float32
            )
        elif (
            self.config["action_noise_rng"]
            == "numpy.default_rng(PCG64).standard_normal(shape).astype(float32)"
        ):
            noise = torch.from_numpy(
                np.random.default_rng(seed).standard_normal(shape).astype(np.float32)
            )
        else:
            raise ValueError("Unsupported action noise generator")
        return noise.to(self.device, self.dtype)

    @torch.inference_mode()
    def predict(self, pixels, state, instruction, seed):
        pixels = np.asarray(pixels)
        expected = (224, 448 if self.is_libero else 224, 3)
        if pixels.dtype != np.uint8 or pixels.shape != expected:
            raise ValueError(f"Expected uint8 image {expected}, got {pixels.shape}/{pixels.dtype}")
        image = torch.from_numpy(pixels.copy()).permute(2, 0, 1).unsqueeze(0)
        # Preserve the two scored evaluators' different input-rounding order.
        if self.is_libero:
            image = image.to(self.device, self.dtype) * (2.0 / 255.0) - 1.0
        else:
            image = image.float().div(127.5).sub(1).to(self.device, self.dtype)
        context, mask = self.append_proprio(
            *self.encode_text(instruction), self.normalize_state(state)
        )
        noise = self.noise(seed)
        normalized = self.klein_forward(image, context, mask, noise)
        action = normalized[0].float().cpu()
        if self.is_libero:
            scale, offset = self.minmax("action")
            action = (action - offset) / scale
            action[:, -1] = torch.sign(-(action[:, -1] * 2 - 1))
        else:
            action = action * torch.tensor(
                self.stats["actions"]["std"], dtype=torch.float32
            ) + torch.tensor(self.stats["actions"]["mean"], dtype=torch.float32)
        if not torch.isfinite(action).all():
            raise ValueError("Non-finite predicted action")
        return action.numpy()

    def klein_forward(self, image, context, mask, noise):
        latent = self.vae.encode(image)
        tokens = self.video.pack_latents(latent)
        ids = self.video.build_img_ids(
            1, *latent.shape[-2:], time_value=10, device=self.device, dtype=self.dtype
        )
        vp = self.video.pre_dit(
            tokens[:, :0],
            tokens.new_zeros(1),
            context,
            mask,
            ref_image_hidden_states=tokens,
            target_img_ids=ids[:, :0],
            ref_img_ids=ids,
        )
        full_mask = prefix_mask(mask, tokens.shape[1], noise.shape[1])
        if self.config["action_regression"]:
            ap = self.action.pre_dit(noise, noise.new_ones(1))
            output = self.mot(
                embeds_all={"video": vp["tokens"], "action": ap["tokens"]},
                attention_mask=full_mask,
                freqs_all={"video": vp["freqs"]},
                context_all={"action": {"ids": ap["ids"]}},
                t_mod_all={"video": vp["t_mod"], "action": ap["t_mod"]},
            )
            return self.action.post_dit(output["action"], ap)
        cache = self.mot.prefill_flux2_video_cache(
            vp["tokens"], vp["freqs"], vp["t_mod"], prefix_mask(mask, tokens.shape[1], 0)
        )
        steps, shift = self.config["num_inference_steps"], self.config["sigma_shift"]
        u = torch.linspace(1, 0, steps + 1, device=self.device, dtype=torch.float32)
        sigma = shift * u / (1 + (shift - 1) * u)
        timesteps = (sigma[:-1] * 1000).to(self.dtype)
        deltas = (sigma[1:] - sigma[:-1]).to(self.dtype)
        for timestep, delta in zip(timesteps, deltas):
            ap = self.action.pre_dit(noise, (timestep.expand(1) / 1000).to(self.dtype))
            output = self.mot.forward_flux2_action_with_video_cache(
                ap["tokens"],
                ap["ids"],
                ap["t_mod"],
                cache,
                full_mask,
                mask.shape[1] + tokens.shape[1],
            )
            noise = noise + self.action.post_dit(output, ap) * delta
        return noise
