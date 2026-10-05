# Derived from ImageWAM (MIT); see LICENSE and NOTICE.
from __future__ import annotations

from functools import partial
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def _rms_norm_cls():
    from diffusers.models.normalization import RMSNorm

    return RMSNorm


def _timestep_embedder_cls():
    from diffusers.models.transformers.transformer_z_image import TimestepEmbedder

    return TimestepEmbedder


def _apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply diffusers Z-Image complex RoPE to ``[B,L,H,D]`` tensors."""
    with torch.amp.autocast("cuda", enabled=False):
        xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        out = torch.view_as_real(xc * freqs.unsqueeze(2)).flatten(3)
    return out.to(dtype=x.dtype)


class ZImageActionBlock(nn.Module):
    def __init__(self, hidden_dim: int, inner_dim: int, num_heads: int, head_dim: int, eps: float):
        super().__init__()
        RMSNorm = _rms_norm_cls()
        self.hidden_dim = hidden_dim
        self.inner_dim = inner_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.adaLN_modulation = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.to_q = nn.Linear(hidden_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(hidden_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(hidden_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, hidden_dim, bias=False)
        self.norm_q = RMSNorm(head_dim, eps=eps)
        self.norm_k = RMSNorm(head_dim, eps=eps)
        self.attention_norm1 = RMSNorm(hidden_dim, eps=eps)
        self.attention_norm2 = RMSNorm(hidden_dim, eps=eps)
        self.ffn_norm1 = RMSNorm(hidden_dim, eps=eps)
        self.ffn_norm2 = RMSNorm(hidden_dim, eps=eps)
        self.w1 = nn.Linear(hidden_dim, 4 * hidden_dim, bias=False)
        self.w2 = nn.Linear(4 * hidden_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, 4 * hidden_dim, bias=False)

    def prepare(
        self, x: torch.Tensor, vec: torch.Tensor, freqs: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(vec).chunk(4, dim=-1)
        xn = self.attention_norm1(x) * (1.0 + scale_msa[:, None])
        shape = (*xn.shape[:2], self.num_heads, self.head_dim)
        q = _apply_rope(self.norm_q(self.to_q(xn).view(shape)), freqs)
        k = _apply_rope(self.norm_k(self.to_k(xn).view(shape)), freqs)
        v = self.to_v(xn).view(shape)
        return (
            {"q": q, "k": k, "v": v},
            {
                "x": x,
                "gate_msa": gate_msa.tanh(),
                "scale_mlp": scale_mlp,
                "gate_mlp": gate_mlp.tanh(),
            },
        )

    def finish(self, attn: torch.Tensor, state: dict[str, torch.Tensor]) -> torch.Tensor:
        x = state["x"] + state["gate_msa"][:, None] * self.attention_norm2(self.to_out(attn))
        xn = self.ffn_norm1(x) * (1.0 + state["scale_mlp"][:, None])
        ff = self.w2(F.silu(self.w1(xn)) * self.w3(xn))
        return x + state["gate_mlp"][:, None] * self.ffn_norm2(ff)


class ZImageActionExpert(nn.Module):
    block_protocol = "zimage"

    def __init__(
        self,
        action_dim: int,
        num_layers: int,
        inner_dim: int = 3840,
        hidden_dim: int = 1280,
        num_heads: int = 30,
        head_dim: int = 128,
        eps: float = 1e-05,
    ):
        super().__init__()
        TimestepEmbedder = _timestep_embedder_cls()
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_heads)
        self.attn_head_dim = int(head_dim)
        self.action_encoder = nn.Linear(action_dim, hidden_dim)
        self.time_embedder = TimestepEmbedder(hidden_dim, mid_size=hidden_dim)
        self.blocks = nn.ModuleList(
            [
                ZImageActionBlock(hidden_dim, inner_dim, num_heads, head_dim, eps)
                for _ in range(num_layers)
            ]
        )
        self.head_norm = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.head_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim))
        self.head = nn.Linear(hidden_dim, action_dim)

    def pre_dit(
        self, action_tokens: torch.Tensor, timestep: torch.Tensor, rope_embedder, axes_lens
    ) -> dict[str, Any]:
        vec = self.time_embedder(timestep * 1000.0).to(dtype=action_tokens.dtype)
        tokens = self.action_encoder(action_tokens) + vec[:, None]
        horizon = int(tokens.shape[1])
        base = int(axes_lens[0]) - horizon - 1
        ids = torch.zeros((horizon, 3), dtype=torch.int32, device=tokens.device)
        ids[:, 0] = torch.arange(base, base + horizon, dtype=torch.int32, device=tokens.device)
        freqs = rope_embedder(ids).unsqueeze(0).expand(tokens.shape[0], -1, -1)
        return {"tokens": tokens, "vec": vec, "freqs": freqs}

    def post_dit(self, tokens: torch.Tensor, pre: dict[str, Any]) -> torch.Tensor:
        shift, scale = self.head_modulation(pre["vec"]).chunk(2, dim=-1)
        return self.head(self.head_norm(tokens) * (1.0 + scale[:, None]) + shift[:, None])


class ZImageVideoExpert(nn.Module):
    block_protocol = "zimage"

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer
        self.blocks = transformer.layers
        self.num_heads = int(transformer.config.n_heads)
        self.num_kv_heads = int(transformer.config.n_kv_heads)
        self.attn_head_dim = int(transformer.config.dim // transformer.config.n_heads)

    def pre_dit(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> dict[str, Any]:
        tr = self.transformer
        images = [item.unsqueeze(1) for item in latents]
        captions = [context[i] for i in range(context.shape[0])]
        x, cap, sizes, x_ids, cap_ids, x_pad, cap_pad = tr.patchify_and_embed(
            images, captions, patch_size=2, f_patch_size=1
        )
        raw_cap_len = int(context.shape[1])
        raw_x_lens = [
            int(size[0]) // 1 * (int(size[1]) // 2) * (int(size[2]) // 2) for size in sizes
        ]
        x = [item[:raw_len] for item, raw_len in zip(x, raw_x_lens)]
        x_ids = [item[:raw_len] for item, raw_len in zip(x_ids, raw_x_lens)]
        x_pad = [item[:raw_len] for item, raw_len in zip(x_pad, raw_x_lens)]
        cap = [item[:raw_cap_len] for item in cap]
        cap_ids = [item[:raw_cap_len] for item in cap_ids]
        cap_pad = [item[:raw_cap_len] for item in cap_pad]
        x_ids = [item.clone() for item in x_ids]
        for item in x_ids:
            item[:, 0] = raw_cap_len + 1
        x_lens, cap_lens = ([len(v) for v in x], [len(v) for v in cap])
        x = tr.all_x_embedder["2-1"](torch.cat(x, dim=0))
        x, x_freqs, x_mask, _, _ = tr._prepare_sequence(
            list(x.split(x_lens)), x_ids, x_pad, tr.x_pad_token, None, latents.device
        )
        vec = tr.t_embedder(timestep * tr.t_scale).to(dtype=latents.dtype)
        for layer in tr.noise_refiner:
            x = layer(x, x_mask, x_freqs, vec)
        cap = tr.cap_embedder(torch.cat(cap, dim=0))
        cap, cap_freqs, cap_mask, _, _ = tr._prepare_sequence(
            list(cap.split(cap_lens)), cap_ids, cap_pad, tr.cap_pad_token, None, latents.device
        )
        for layer in tr.context_refiner:
            cap = layer(cap, cap_mask, cap_freqs)
        unified, freqs, valid, _ = tr._build_unified_sequence(
            x,
            x_freqs,
            x_lens,
            None,
            cap,
            cap_freqs,
            cap_lens,
            None,
            None,
            None,
            None,
            None,
            False,
            latents.device,
        )
        valid = torch.cat(
            [
                torch.ones(
                    (context.shape[0], int(x.shape[1])), dtype=torch.bool, device=context.device
                ),
                context_mask.to(device=context.device, dtype=torch.bool),
            ],
            dim=1,
        )
        if tuple(valid.shape) != tuple(unified.shape[:2]):
            raise RuntimeError(
                f"JAX-padded Z-Image validity shape mismatch: valid={tuple(valid.shape)} unified={tuple(unified.shape[:2])}"
            )
        if cap.shape[1] != context.shape[1]:
            raise RuntimeError(
                f"JAX-padded Z-Image caption length mismatch: prepared={cap.shape[1]} raw={context.shape[1]}"
            )
        return {
            "tokens": unified,
            "freqs": freqs,
            "valid": valid,
            "vec": vec,
            "image_len": int(x.shape[1]),
            "sizes": sizes,
        }


class ZImageMoT(nn.Module):
    """Faithful read-only MoT loop for diffusers Z-Image blocks."""

    def __init__(
        self,
        video: ZImageVideoExpert,
        action: ZImageActionExpert,
    ):
        super().__init__()
        self.mixtures = nn.ModuleDict({"video": video, "action": action})
        self.num_heads = video.num_heads
        self.head_dim = video.attn_head_dim

    @property
    def video(self):
        return self.mixtures["video"]

    @property
    def action(self):
        return self.mixtures["action"]

    def _attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:

        def run(q_, k_, v_):
            return F.scaled_dot_product_attention(
                q_.transpose(1, 2),
                k_.transpose(1, 2),
                v_.transpose(1, 2),
                attn_mask=mask[:, None],
                dropout_p=0.0,
            ).transpose(1, 2)

        return run(q, k, v)

    @staticmethod
    def _video_prepare(block, x: torch.Tensor, vec: torch.Tensor, freqs: torch.Tensor):
        scale_msa, gate_msa, scale_mlp, gate_mlp = block.adaLN_modulation(vec).chunk(4, dim=-1)
        xn = block.attention_norm1(x) * (1.0 + scale_msa[:, None])
        attn = block.attention
        q = attn.to_q(xn).unflatten(-1, (attn.heads, -1))
        k = attn.to_k(xn).unflatten(-1, (attn.heads, -1))
        v = attn.to_v(xn).unflatten(-1, (attn.heads, -1))
        q = _apply_rope(attn.norm_q(q), freqs)
        k = _apply_rope(attn.norm_k(k), freqs)
        return (
            {"q": q, "k": k, "v": v},
            {
                "x": x,
                "gate_msa": gate_msa.tanh(),
                "scale_mlp": scale_mlp,
                "gate_mlp": gate_mlp.tanh(),
            },
        )

    @staticmethod
    def _video_finish(
        block, attn_out: torch.Tensor, state: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        proj = block.attention.to_out[0](attn_out.flatten(2, 3))
        x = state["x"] + state["gate_msa"][:, None] * block.attention_norm2(proj)
        xn = block.ffn_norm1(x) * (1.0 + state["scale_mlp"][:, None])
        return x + state["gate_mlp"][:, None] * block.ffn_norm2(block.feed_forward(xn))

    def _forward_block(self, vb, ab, vtok, atok, vvec, avec, vfreqs, afreqs, mask):
        prefix = vtok.shape[1]
        vqkv, vs = self._video_prepare(vb, vtok, vvec, vfreqs)
        aqkv, ass = ab.prepare(atok, avec, afreqs)
        q = torch.cat([vqkv["q"], aqkv["q"]], dim=1)
        k = torch.cat([vqkv["k"], aqkv["k"]], dim=1)
        v = torch.cat([vqkv["v"], aqkv["v"]], dim=1)
        mixed = self._attention(q, k, v, mask)
        return (
            self._video_finish(vb, mixed[:, :prefix], vs),
            ab.finish(mixed[:, prefix:].flatten(2, 3), ass),
        )

    def forward(self, embeds_all, attention_mask, freqs_all, context_all, t_mod_all):
        del attention_mask, freqs_all, context_all
        video, action = (embeds_all["video"], embeds_all["action"])
        vtok, atok = (video["tokens"], action["tokens"])
        prefix_valid = video["valid"]
        action_valid = torch.ones(atok.shape[:2], dtype=torch.bool, device=atok.device)
        col_valid = torch.cat([prefix_valid, action_valid], dim=1)
        prefix = vtok.shape[1]
        total = prefix + atok.shape[1]
        row_action = torch.arange(total, device=atok.device)[None, :, None] >= prefix
        col_action = torch.arange(total, device=atok.device)[None, None, :] >= prefix
        mask = col_valid[:, None, :] & (row_action | ~col_action)
        for vb, ab in zip(self.video.blocks, self.action.blocks):
            run = partial(self._forward_block, vb, ab)
            args = (vtok, atok, video["vec"], action["vec"], video["freqs"], action["freqs"], mask)
            vtok, atok = run(*args)
        return {"video": vtok, "action": atok}
