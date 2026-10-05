# Derived from ImageWAM (MIT); see LICENSE and NOTICE.
from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Dict, Optional

import torch
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F


class SlimFlux2SelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, attn_head_dim: int):
        super().__init__()
        from flux2.model import QKNorm

        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.attn_head_dim = int(attn_head_dim)
        self.attn_dim = self.num_heads * self.attn_head_dim
        self.qkv = nn.Linear(self.hidden_dim, 3 * self.attn_dim, bias=False)
        self.norm = QKNorm(self.attn_head_dim)
        self.proj = nn.Linear(self.attn_dim, self.hidden_dim, bias=False)


class SlimFlux2DoubleBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, attn_head_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        from flux2.model import SiLUActivation

        self.hidden_size = int(hidden_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.attn_head_dim = int(attn_head_dim)
        self.attn_dim = self.num_heads * self.attn_head_dim
        self.mlp_hidden_dim = int(round(self.hidden_dim * float(mlp_ratio)))
        self.mlp_mult_factor = 2
        self.img_norm1 = nn.LayerNorm(self.hidden_dim, elementwise_affine=False, eps=1e-06)
        self.img_attn = SlimFlux2SelfAttention(self.hidden_dim, self.num_heads, self.attn_head_dim)
        self.img_norm2 = nn.LayerNorm(self.hidden_dim, elementwise_affine=False, eps=1e-06)
        self.img_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.mlp_hidden_dim * self.mlp_mult_factor, bias=False),
            SiLUActivation(),
            nn.Linear(self.mlp_hidden_dim, self.hidden_dim, bias=False),
        )

    def prepare_qkv(self, x: torch.Tensor, pe: torch.Tensor, mod_img):
        from einops import rearrange
        from flux2.model import apply_rope

        mod1, mod2 = mod_img
        mod1_shift, mod1_scale, mod1_gate = mod1
        mod2_shift, mod2_scale, mod2_gate = mod2
        x_mod = (1 + mod1_scale) * self.img_norm1(x) + mod1_shift
        qkv = self.img_attn.qkv(x_mod)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.img_attn.norm(q, k, v)
        q, k = apply_rope(q, k, pe)
        return {
            "q": q.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.attn_dim),
            "k": k.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.attn_dim),
            "v": v.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.attn_dim),
            "residual_x": x,
            "mod2_shift": mod2_shift,
            "mod2_scale": mod2_scale,
            "mod1_gate": mod1_gate,
            "mod2_gate": mod2_gate,
        }

    def apply_post(self, mixed_attn_out: torch.Tensor, state: dict) -> torch.Tensor:
        x = state["residual_x"] + state["mod1_gate"] * self.img_attn.proj(mixed_attn_out)
        x = x + state["mod2_gate"] * self.img_mlp(
            (1 + state["mod2_scale"]) * self.img_norm2(x) + state["mod2_shift"]
        )
        return x


class SlimFlux2SingleBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, attn_head_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        from flux2.model import QKNorm, SiLUActivation

        self.hidden_size = int(hidden_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.attn_head_dim = int(attn_head_dim)
        self.attn_dim = self.num_heads * self.attn_head_dim
        self.mlp_hidden_dim = int(round(self.hidden_dim * float(mlp_ratio)))
        self.mlp_mult_factor = 2
        self.linear1 = nn.Linear(
            self.hidden_dim,
            3 * self.attn_dim + self.mlp_hidden_dim * self.mlp_mult_factor,
            bias=False,
        )
        self.linear2 = nn.Linear(self.attn_dim + self.mlp_hidden_dim, self.hidden_dim, bias=False)
        self.norm = QKNorm(self.attn_head_dim)
        self.pre_norm = nn.LayerNorm(self.hidden_dim, elementwise_affine=False, eps=1e-06)
        self.mlp_act = SiLUActivation()

    def prepare_qkv(self, x: torch.Tensor, pe: torch.Tensor, mod):
        from einops import rearrange
        from flux2.model import apply_rope

        mod_shift, mod_scale, mod_gate = mod
        x_mod = (1 + mod_scale) * self.pre_norm(x) + mod_shift
        qkv, mlp = torch.split(
            self.linear1(x_mod),
            [3 * self.attn_dim, self.mlp_hidden_dim * self.mlp_mult_factor],
            dim=-1,
        )
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        q, k = apply_rope(q, k, pe)
        return {
            "q": q.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.attn_dim),
            "k": k.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.attn_dim),
            "v": v.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.attn_dim),
            "mlp": mlp,
            "gate": mod_gate,
            "residual_x": x,
        }

    def apply_post(self, mixed_attn_out: torch.Tensor, state: dict) -> torch.Tensor:
        output = self.linear2(torch.cat((mixed_attn_out, self.mlp_act(state["mlp"])), dim=2))
        return state["residual_x"] + state["gate"] * output


class Flux2ActionHead(nn.Module):
    def __init__(self, hidden_dim: int, action_dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-06)
        self.linear = nn.Linear(hidden_dim, action_dim, bias=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim, bias=False)
        )

    def forward(self, x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=-1)
        return self.linear((1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :])


class ActionDiTFlux2(nn.Module):
    """Slim action expert with FLUX.2-compatible attention dimensions."""

    block_protocol = "flux2"

    def __init__(
        self,
        action_dim: int,
        hidden_dim: int = 1024,
        num_heads: int = 24,
        attn_head_dim: int = 128,
        num_layers_double: int = 5,
        num_layers_single: int = 20,
        mlp_ratio: float = 4.0,
        max_action_horizon: int = 64,
    ) -> None:
        super().__init__()
        from flux2.model import MLPEmbedder, Modulation

        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.num_kv_heads = self.num_heads
        self.attn_head_dim = int(attn_head_dim)
        self.attn_dim = self.num_heads * self.attn_head_dim
        self.double_layers = int(num_layers_double)
        self.single_layers = int(num_layers_single)
        self.max_action_horizon = int(max_action_horizon)
        self.action_encoder = nn.Linear(self.action_dim, self.hidden_dim)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_dim, disable_bias=True)
        self.double_stream_modulation_img = Modulation(
            self.hidden_dim, double=True, disable_bias=True
        )
        self.single_stream_modulation = Modulation(self.hidden_dim, double=False, disable_bias=True)
        self.double_blocks = nn.ModuleList(
            [
                SlimFlux2DoubleBlock(
                    hidden_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    attn_head_dim=self.attn_head_dim,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(self.double_layers)
            ]
        )
        self.single_blocks = nn.ModuleList(
            [
                SlimFlux2SingleBlock(
                    hidden_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    attn_head_dim=self.attn_head_dim,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(self.single_layers)
            ]
        )
        self.head = Flux2ActionHead(self.hidden_dim, self.action_dim)

    @property
    def blocks(self):
        return list(self.double_blocks) + list(self.single_blocks)

    @staticmethod
    def build_action_ids(
        batch_size: int, seq_len: int, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        ids = torch.zeros(batch_size, seq_len, 4, device=device, dtype=dtype)
        ids[..., 0] = 2.0
        ids[..., 1] = torch.arange(seq_len, device=device, dtype=dtype)[None, :]
        return ids

    def pre_dit(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> Dict[str, Any]:
        del context, context_mask
        if action_tokens.ndim != 3:
            raise ValueError(f"`action_tokens` must be [B,T,D], got {tuple(action_tokens.shape)}")
        batch_size, seq_len, action_dim = action_tokens.shape
        if action_dim != self.action_dim:
            raise ValueError(f"Expected action_dim={self.action_dim}, got {action_dim}")
        if seq_len > self.max_action_horizon:
            raise ValueError(
                f"Action length {seq_len} exceeds max_action_horizon={self.max_action_horizon}"
            )
        if timestep.ndim != 1:
            raise ValueError(f"`timestep` must be [B], got {tuple(timestep.shape)}")
        if timestep.shape[0] == 1 and batch_size > 1:
            timestep = timestep.expand(batch_size)
        if timestep.shape[0] != batch_size:
            raise ValueError(
                f"`timestep` length must match batch size {batch_size}, got {timestep.shape[0]}"
            )
        from flux2.model import timestep_embedding

        tokens = self.action_encoder(action_tokens)
        vec = self.time_in(timestep_embedding(timestep, 256)).to(dtype=tokens.dtype)
        double_mod_img = self.double_stream_modulation_img(vec)
        single_mod, _ = self.single_stream_modulation(vec)
        ids = self.build_action_ids(batch_size, seq_len, device=tokens.device, dtype=tokens.dtype)
        return {
            "tokens": tokens,
            "ids": ids,
            "t_mod": {"vec": vec, "double_img": double_mod_img, "single": single_mod},
            "context": None,
            "context_mask": None,
            "meta": {"batch_size": batch_size, "seq_len": seq_len},
        }

    def post_dit(self, tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        explicit_autocast = self.fp32_residual and (not torch.is_autocast_enabled())
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if explicit_autocast
            else nullcontext()
        )
        with autocast_context:
            return self.head(tokens, pre_state["t_mod"]["vec"])


class Flux2VideoExpert(nn.Module):
    """ImageWAM video expert wrapper around the official FLUX.2 transformer."""

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

    @property
    def blocks(self):
        return list(self.double_blocks) + list(self.single_blocks)

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


class KleinMoT(nn.Module):
    def __init__(self, video, action, *, fp32_residual, explicit_attention):
        super().__init__()
        self.mixtures = nn.ModuleDict({"video": video, "action": action})
        self.fp32_residual = bool(fp32_residual)
        action.fp32_residual = self.fp32_residual
        self.explicit_attention = bool(explicit_attention)
        self.num_heads = video.num_heads
        self.attn_head_dim = video.attn_head_dim
        self.block_protocol = "flux2"
        if len(video.blocks) != len(action.blocks):
            raise ValueError("Video/action depths differ")

    def _mixed_attention(self, q_cat, k_cat, v_cat, attention_mask):
        batch, queries, _ = q_cat.shape
        keys = k_cat.shape[1]
        heads, dim = (self.num_heads, self.attn_head_dim)
        q = q_cat.view(batch, queries, heads, dim).transpose(1, 2)
        k = k_cat.view(batch, keys, heads, dim).transpose(1, 2)
        v = v_cat.view(batch, keys, heads, dim).transpose(1, 2)
        mask = attention_mask.to(device=q.device, dtype=torch.bool)
        if mask.ndim == 3:
            mask = mask[:, None]
        if self.explicit_attention:
            with torch.autocast(device_type=q.device.type, enabled=False):
                scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * dim ** (-0.5)
                scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
                probs = torch.softmax(scores, dim=-1)
            out = torch.matmul(probs.to(v.dtype), v)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return out.transpose(1, 2).reshape(batch, queries, heads * dim)

    def forward(self, **kwargs):
        return self._forward_flux2(**kwargs)

    def _flux2_flatten_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.transpose(1, 2).reshape(
            tensor.shape[0], tensor.shape[2], tensor.shape[1] * tensor.shape[3]
        )

    def _flux2_video_single_io(self, block, x: torch.Tensor, pe: torch.Tensor, mod) -> dict:
        from flux2.model import apply_rope

        q, k, v, mlp, gate = block._qkv(x, mod)
        q, k = apply_rope(q, k, pe)
        return {
            "q": self._flux2_flatten_heads(q),
            "k": self._flux2_flatten_heads(k),
            "v": self._flux2_flatten_heads(v),
            "mlp": mlp,
            "gate": gate,
            "residual_x": x,
        }

    def _forward_flux2(
        self,
        embeds_all: Dict[str, object],
        attention_mask: dict[str, torch.Tensor],
        freqs_all: Dict[str, object],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, object],
    ):
        if self.fp32_residual and (not torch.is_autocast_enabled()):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return self._forward_flux2(
                    embeds_all=embeds_all,
                    attention_mask=attention_mask,
                    freqs_all=freqs_all,
                    context_all=context_all,
                    t_mod_all=t_mod_all,
                )
        if "video" not in self.mixtures or "action" not in self.mixtures:
            raise ValueError("FLUX.2 MoT requires `video` and `action` experts.")
        if not isinstance(attention_mask, dict):
            raise ValueError("FLUX.2 MoT expects attention_mask={'double_joint', 'single'}.")
        video_expert = self.mixtures["video"]
        action_expert = self.mixtures["action"]
        video_state = embeds_all["video"]
        if not isinstance(video_state, dict):
            raise ValueError("FLUX.2 video embeds must be a dict with `txt` and `img` tensors.")
        jax_fp32_residual = self.fp32_residual
        txt = video_state["txt"].float() if jax_fp32_residual else video_state["txt"]
        img = video_state["img"].float() if jax_fp32_residual else video_state["img"]
        action = embeds_all["action"]
        if not isinstance(action, torch.Tensor):
            raise ValueError("FLUX.2 action embeds must be a tensor.")
        if jax_fp32_residual:
            action = action.float()
        video_freqs = freqs_all["video"]
        txt_pe = video_freqs["txt"]
        img_pe = video_freqs["img"]
        action_ids = context_all["action"]["ids"]
        action_pe = video_expert.transformer.pe_embedder(
            action_ids.to(device=img.device, dtype=img.dtype)
        )
        video_t_mod = t_mod_all["video"]
        action_t_mod = t_mod_all["action"]
        for layer_idx in range(int(getattr(video_expert, "double_layers"))):
            v_block = video_expert.double_blocks[layer_idx]
            a_block = action_expert.double_blocks[layer_idx]
            args = (
                img,
                txt,
                action,
                img_pe,
                txt_pe,
                action_pe,
                video_t_mod["double_img"],
                video_t_mod["double_txt"],
                action_t_mod["double_img"],
                attention_mask["double_joint"],
            )
            img, txt, action = self._flux2_double_joint(v_block, a_block, *args)
        video_stream = torch.cat([txt, img], dim=1)
        stream_pe = torch.cat([txt_pe, img_pe], dim=2)
        for layer_idx in range(int(getattr(video_expert, "single_layers"))):
            v_block = video_expert.single_blocks[layer_idx]
            a_block = action_expert.single_blocks[layer_idx]
            args = (
                video_stream,
                action,
                stream_pe,
                action_pe,
                video_t_mod["single"],
                action_t_mod["single"],
                attention_mask["single"],
            )
            video_stream, action = self._flux2_single_joint(v_block, a_block, *args)
        txt_len = int(txt.shape[1])
        txt, img = (video_stream[:, :txt_len], video_stream[:, txt_len:])
        return {"video": {"txt": txt, "img": img}, "action": action}

    def _flux2_double_joint(
        self,
        v_block,
        a_block,
        img,
        txt,
        action,
        img_pe,
        txt_pe,
        action_pe,
        img_mod,
        txt_mod,
        action_mod,
        mask,
    ):
        from flux2.model import apply_rope

        q, k, v, pe, nt, mods = v_block._prepare_qkv(img, txt, img_pe, txt_pe, img_mod, txt_mod)
        q, k = apply_rope(q, k, pe)
        state = a_block.prepare_qkv(action, action_pe, action_mod)
        mixed = self._mixed_attention(
            torch.cat([self._flux2_flatten_heads(q), state["q"]], dim=1),
            torch.cat([self._flux2_flatten_heads(k), state["k"]], dim=1),
            torch.cat([self._flux2_flatten_heads(v), state["v"]], dim=1),
            mask,
        )
        va, aa = torch.split(mixed, [txt.shape[1] + img.shape[1], action.shape[1]], dim=1)
        ta, ia = torch.split(va, [nt, img.shape[1]], dim=1)
        img, txt = v_block._apply_residuals(img, txt, ia, ta, mods)
        return (img, txt, a_block.apply_post(aa, state))

    def _flux2_single_joint(
        self,
        v_block,
        a_block,
        video,
        action,
        pe,
        action_pe,
        video_mod,
        action_mod,
        mask,
    ):
        vs = self._flux2_video_single_io(v_block, video, pe, video_mod)
        ac = a_block.prepare_qkv(action, action_pe, action_mod)
        mixed = self._mixed_attention(
            torch.cat([vs["q"], ac["q"]], dim=1),
            torch.cat([vs["k"], ac["k"]], dim=1),
            torch.cat([vs["v"], ac["v"]], dim=1),
            mask,
        )
        va, aa = torch.split(mixed, [video.shape[1], action.shape[1]], dim=1)
        return (
            v_block._out(vs["residual_x"], va, vs["mlp"], vs["gate"]),
            a_block.apply_post(aa, ac),
        )

    def prefill_flux2_video_cache(
        self,
        video_tokens: dict[str, torch.Tensor],
        video_freqs: dict[str, torch.Tensor],
        video_t_mod: dict[str, object],
        attention_mask: dict[str, torch.Tensor],
    ) -> dict[str, object]:
        if self.block_protocol != "flux2":
            raise ValueError("`prefill_flux2_video_cache` requires block_protocol='flux2'.")
        video_expert = self.mixtures["video"]
        txt = video_tokens["txt"]
        img = video_tokens["img"]
        txt_pe = video_freqs["txt"]
        img_pe = video_freqs["img"]
        from flux2.model import apply_rope

        double_cache = []
        for layer_idx in range(int(getattr(video_expert, "double_layers"))):
            block = video_expert.double_blocks[layer_idx]
            q, k, v, pe_full, num_txt_tokens, mods = block._prepare_qkv(
                img, txt, img_pe, txt_pe, video_t_mod["double_img"], video_t_mod["double_txt"]
            )
            q, k = apply_rope(q, k, pe_full)
            flat_q = self._flux2_flatten_heads(q)
            flat_k = self._flux2_flatten_heads(k)
            flat_v = self._flux2_flatten_heads(v)
            double_cache.append({"k": flat_k, "v": flat_v})
            mixed = self._mixed_attention(flat_q, flat_k, flat_v, attention_mask["double_joint"])
            txt_attn, img_attn = torch.split(mixed, [num_txt_tokens, img.shape[1]], dim=1)
            img, txt = block._apply_residuals(img, txt, img_attn, txt_attn, mods)
        video_stream = torch.cat([txt, img], dim=1)
        stream_pe = torch.cat([txt_pe, img_pe], dim=2)
        single_cache = []
        for layer_idx in range(int(getattr(video_expert, "single_layers"))):
            block = video_expert.single_blocks[layer_idx]
            state = self._flux2_video_single_io(
                block, video_stream, stream_pe, video_t_mod["single"]
            )
            single_cache.append({"k": state["k"], "v": state["v"]})
            mixed = self._mixed_attention(
                state["q"], state["k"], state["v"], attention_mask["single"]
            )
            video_stream = block._out(state["residual_x"], mixed, state["mlp"], state["gate"])
        return {
            "double": double_cache,
            "single": single_cache,
            "txt_len": int(txt.shape[1]),
            "img_len": int(img.shape[1]),
            "final_video": {
                "txt": video_stream[:, : txt.shape[1]],
                "img": video_stream[:, txt.shape[1] :],
            },
        }

    def forward_flux2_action_with_video_cache(
        self,
        action_tokens: torch.Tensor,
        action_ids: torch.Tensor,
        action_t_mod: dict[str, object],
        video_kv_cache: dict[str, object],
        attention_mask: dict[str, torch.Tensor],
        video_seq_len: int,
    ) -> torch.Tensor:
        if self.block_protocol != "flux2":
            raise ValueError(
                "`forward_flux2_action_with_video_cache` requires block_protocol='flux2'."
            )
        video_expert = self.mixtures["video"]
        action_expert = self.mixtures["action"]
        action = action_tokens
        action_pe = video_expert.transformer.pe_embedder(
            action_ids.to(device=action.device, dtype=action.dtype)
        )
        action_seq_len = int(action.shape[1])
        total_seq_len = int(video_seq_len) + action_seq_len

        def _action_mask(mask: torch.Tensor) -> torch.Tensor:
            if mask.ndim == 2:
                return mask[video_seq_len:total_seq_len, :total_seq_len]
            if mask.ndim == 3:
                return mask[:, video_seq_len:total_seq_len, :total_seq_len]
            return mask[:, :, video_seq_len:total_seq_len, :total_seq_len]

        double_mask = _action_mask(attention_mask["double_joint"])
        for layer_idx, cache in enumerate(video_kv_cache["double"]):
            block = action_expert.double_blocks[layer_idx]
            state = block.prepare_qkv(action, action_pe, action_t_mod["double_img"])
            k_cat = torch.cat([cache["k"].to(dtype=state["k"].dtype), state["k"]], dim=1)
            v_cat = torch.cat([cache["v"].to(dtype=state["v"].dtype), state["v"]], dim=1)
            mixed = self._mixed_attention(state["q"], k_cat, v_cat, double_mask)
            action = block.apply_post(mixed, state)
        single_mask = _action_mask(attention_mask["single"])
        for layer_idx, cache in enumerate(video_kv_cache["single"]):
            block = action_expert.single_blocks[layer_idx]
            state = block.prepare_qkv(action, action_pe, action_t_mod["single"])
            k_cat = torch.cat([cache["k"].to(dtype=state["k"].dtype), state["k"]], dim=1)
            v_cat = torch.cat([cache["v"].to(dtype=state["v"].dtype), state["v"]], dim=1)
            mixed = self._mixed_attention(state["q"], k_cat, v_cat, single_mask)
            action = block.apply_post(mixed, state)
        return action
