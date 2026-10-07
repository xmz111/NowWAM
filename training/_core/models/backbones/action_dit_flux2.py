from __future__ import annotations
from contextlib import nullcontext
from typing import Any, Dict
import torch
import torch.nn as nn
from training._core.utils.logging_config import get_logger
from .flux2_imports import ensure_flux2_importable

logger = get_logger(__name__)


class SlimFlux2SelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, attn_head_dim: int):
        super().__init__()
        ensure_flux2_importable()
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
        ensure_flux2_importable()
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
        ensure_flux2_importable()
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
        use_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        ensure_flux2_importable()
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
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
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

    @classmethod
    def from_pretrained(
        cls,
        action_dit_config: dict[str, Any],
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "ActionDiTFlux2":
        if action_dit_config is None:
            raise ValueError(
                "`action_dit_config` is required for ActionDiTFlux2.from_pretrained()."
            )
        model = cls(**action_dit_config).to(device=device, dtype=torch_dtype)
        if skip_dit_load_from_pretrain or not action_dit_pretrained_path:
            logger.info("Initializing ActionDiTFlux2 without pretrained action weights.")
            return model
        payload = torch.load(action_dit_pretrained_path, map_location="cpu")
        state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        if not isinstance(state_dict, dict):
            raise ValueError(f"Invalid ActionDiTFlux2 checkpoint type: {type(payload)}")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        allowed_missing_prefixes = ("action_encoder.", "head.")
        bad_missing = [key for key in missing if not key.startswith(allowed_missing_prefixes)]

        def _allowed_truncated_block(key: str) -> bool:
            for prefix, kept in (
                ("double_blocks.", model.double_layers),
                ("single_blocks.", model.single_layers),
            ):
                if key.startswith(prefix):
                    suffix = key[len(prefix) :]
                    index_text = suffix.split(".", 1)[0]
                    return index_text.isdigit() and int(index_text) >= kept
            return False

        bad_unexpected = [key for key in unexpected if not _allowed_truncated_block(key)]
        if bad_missing or bad_unexpected:
            raise RuntimeError(
                f"ActionDiTFlux2 pretrained-load contract failed: bad_missing={bad_missing[:20]} bad_unexpected={bad_unexpected[:20]} all_missing={list(missing)[:20]} all_unexpected={list(unexpected)[:20]}"
            )
        return model

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
        explicit_autocast = "1" == "1" and (not torch.is_autocast_enabled())
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if explicit_autocast
            else nullcontext()
        )
        with autocast_context:
            return self.head(tokens, pre_state["t_mod"]["vec"])
