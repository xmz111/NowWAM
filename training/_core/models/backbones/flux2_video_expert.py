from __future__ import annotations
from contextlib import nullcontext
from typing import Any, Dict
import torch
import torch.nn as nn
from einops import rearrange, repeat
from safetensors.torch import load_file as load_sft
from .flux2_imports import ensure_flux2_importable


class Flux2VideoExpert(nn.Module):
    block_protocol = "flux2"

    def __init__(self, transformer: nn.Module, variant: str):
        super().__init__()
        self.transformer = transformer
        self.variant = str(variant)
        self.double_blocks = transformer.double_blocks
        self.single_blocks = transformer.single_blocks
        self.hidden_dim = int(transformer.hidden_size)
        self.num_heads = int(transformer.num_heads)
        self.num_kv_heads = self.num_heads
        self.attn_head_dim = self.hidden_dim // self.num_heads
        self.double_layers = len(self.double_blocks)
        self.single_layers = len(self.single_blocks)
        self.use_gradient_checkpointing = bool(
            getattr(transformer, "gradient_checkpointing", False)
        )

    @property
    def blocks(self):
        return list(self.double_blocks) + list(self.single_blocks)

    @classmethod
    def from_pretrained(
        cls,
        flux2_model_path: str,
        variant: str = "klein-base-4b",
        flux2_src_path: str | None = None,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "Flux2VideoExpert":
        ensure_flux2_importable(flux2_src_path)
        from flux2.model import Flux2, Klein4BParams, Klein9BParams

        key = str(variant).lower().replace("_", "-")
        if key in {"klein-base-4b", "flux.2-klein-base-4b", "4b", "base-4b"}:
            params = Klein4BParams()
        elif key in {"klein-base-9b", "flux.2-klein-base-9b", "9b", "base-9b"}:
            params = Klein9BParams()
        else:
            raise ValueError(f"Unsupported FLUX.2 Klein variant: {variant!r}")
        import os as _os

        if "klein" == "random_dit":
            transformer = Flux2(params).to(device=device, dtype=torch_dtype)
            print(
                "[WEIGHT_INIT=random_dit] FLUX-2 DiT RANDOM-INITIALIZED (Klein weights NOT loaded)",
                flush=True,
            )
            return cls(transformer, variant=key)
        with torch.device("meta"):
            transformer = Flux2(params).to(torch_dtype)
        state_dict = load_sft(str(flux2_model_path), device=str(device))
        transformer.load_state_dict(state_dict, strict=True, assign=True)
        return cls(transformer.to(device=device, dtype=torch_dtype), variant=key)

    @staticmethod
    def build_txt_ids(
        batch_size: int, seq_len: int, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        ids = torch.zeros(batch_size, seq_len, 4, device=device, dtype=dtype)
        ids[..., 3] = torch.arange(seq_len, device=device, dtype=dtype)[None, :]
        return ids

    @staticmethod
    def build_img_ids(
        batch_size: int,
        token_height: int,
        token_width: int,
        *,
        time_value: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        ids = torch.zeros(token_height, token_width, 4, device=device, dtype=dtype)
        ids[..., 0] = float(time_value)
        ids[..., 1] = torch.arange(token_height, device=device, dtype=dtype)[:, None]
        ids[..., 2] = torch.arange(token_width, device=device, dtype=dtype)[None, :]
        return repeat(ids, "h w c -> b (h w) c", b=batch_size)

    @staticmethod
    def pack_latents(latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 4:
            raise ValueError(f"`latents` must be [B,C,H,W], got {tuple(latents.shape)}")
        return rearrange(latents, "b c h w -> b (h w) c")

    @staticmethod
    def unpack_latents(tokens: torch.Tensor, latent_height: int, latent_width: int) -> torch.Tensor:
        return rearrange(tokens, "b (h w) c -> b c h w", h=latent_height, w=latent_width)

    def pre_dit(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        ref_image_hidden_states: torch.Tensor | None = None,
        target_img_ids: torch.Tensor | None = None,
        ref_img_ids: torch.Tensor | None = None,
        **_: Any,
    ) -> Dict[str, Any]:
        if x.ndim != 3:
            raise ValueError(f"`x` must be FLUX.2 image tokens [B,N,128], got {tuple(x.shape)}")
        if context.ndim != 3:
            raise ValueError(f"`context` must be [B,L,D], got {tuple(context.shape)}")
        batch_size, target_len = (int(x.shape[0]), int(x.shape[1]))
        cond_len = 0 if ref_image_hidden_states is None else int(ref_image_hidden_states.shape[1])
        if target_img_ids is None:
            raise ValueError("`target_img_ids` is required for Flux2VideoExpert.pre_dit().")
        if ref_image_hidden_states is not None and ref_img_ids is None:
            raise ValueError(
                "`ref_img_ids` is required when `ref_image_hidden_states` is provided."
            )
        if timestep.ndim != 1:
            raise ValueError(f"`timestep` must be [B], got {tuple(timestep.shape)}")
        if timestep.shape[0] == 1 and batch_size > 1:
            timestep = timestep.expand(batch_size)
        if timestep.shape[0] != batch_size:
            raise ValueError(
                f"`timestep` length must match batch size {batch_size}, got {timestep.shape[0]}"
            )
        transformer = self.transformer
        img_tokens = (
            x if ref_image_hidden_states is None else torch.cat([ref_image_hidden_states, x], dim=1)
        )
        img_ids = (
            target_img_ids
            if ref_img_ids is None
            else torch.cat([ref_img_ids, target_img_ids], dim=1)
        )
        from flux2.model import timestep_embedding

        vec = transformer.time_in(timestep_embedding(timestep, 256))
        txt = transformer.txt_in(context)
        img = transformer.img_in(img_tokens)
        txt_ids = self.build_txt_ids(
            batch_size=batch_size, seq_len=txt.shape[1], device=img.device, dtype=img_ids.dtype
        )
        txt_pe = transformer.pe_embedder(txt_ids)
        img_pe = transformer.pe_embedder(img_ids)
        double_mod_img = transformer.double_stream_modulation_img(vec)
        double_mod_txt = transformer.double_stream_modulation_txt(vec)
        single_mod, _ = transformer.single_stream_modulation(vec)
        if context_mask is None:
            text_mask = torch.ones(batch_size, txt.shape[1], device=img.device, dtype=torch.bool)
        else:
            text_mask = context_mask.to(device=img.device, dtype=torch.bool)
        return {
            "tokens": {"txt": txt, "img": img},
            "freqs": {"txt": txt_pe, "img": img_pe},
            "t_mod": {
                "vec": vec,
                "double_img": double_mod_img,
                "double_txt": double_mod_txt,
                "single": single_mod,
            },
            "context": None,
            "context_mask": text_mask,
            "txt_len": int(txt.shape[1]),
            "target_len": target_len,
            "cond_len": cond_len,
            "text_mask": text_mask,
            "meta": {
                "batch_size": batch_size,
                "txt_len": int(txt.shape[1]),
                "target_len": target_len,
                "cond_len": cond_len,
            },
        }

    def post_dit(self, tokens: Dict[str, torch.Tensor], pre_state: Dict[str, Any]) -> torch.Tensor:
        img = tokens["img"] if isinstance(tokens, dict) else tokens
        cond_len = int(pre_state["cond_len"])
        target_len = int(pre_state["target_len"])
        target = img[:, cond_len : cond_len + target_len]
        explicit_autocast = "1" == "1" and (not torch.is_autocast_enabled())
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if explicit_autocast
            else nullcontext()
        )
        with autocast_context:
            return self.transformer.final_layer(target, pre_state["t_mod"]["vec"])

    def post_dit_ref(
        self, tokens: Dict[str, torch.Tensor], pre_state: Dict[str, Any]
    ) -> torch.Tensor:
        img = tokens["img"] if isinstance(tokens, dict) else tokens
        cond_len = int(pre_state["cond_len"])
        target_len = int(pre_state["target_len"])
        if cond_len <= 0 or target_len != 0:
            raise RuntimeError(
                f"post_dit_ref requires ref-only tokens: cond_len={cond_len} target_len={target_len}"
            )
        ref = img[:, :cond_len]
        explicit_autocast = "1" == "1" and (not torch.is_autocast_enabled())
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if explicit_autocast
            else nullcontext()
        )
        with autocast_context:
            return self.transformer.final_layer(ref, pre_state["t_mod"]["vec"])
