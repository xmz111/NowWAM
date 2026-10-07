from __future__ import annotations
from contextlib import nullcontext
from functools import partial
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from training._core.utils.logging_config import get_logger

logger = get_logger(__name__)


class MoT(nn.Module):
    def __init__(
        self,
        mixtures: Dict[str, nn.Module],
        mot_checkpoint_mixed_attn: bool = True,
        gqa_implementation: str = "repeat",
        force_flash_attention: bool = False,
    ):
        super().__init__()
        if not mixtures:
            raise ValueError("`mixtures` cannot be empty.")
        self.mixtures = nn.ModuleDict(mixtures)
        self.expert_order = list(self.mixtures.keys())
        self.mot_checkpoint_mixed_attn = mot_checkpoint_mixed_attn
        self.gqa_implementation = str(gqa_implementation).strip().lower()
        if self.gqa_implementation not in {"repeat", "sdpa"}:
            raise ValueError(
                f"`gqa_implementation` must be 'repeat' or 'sdpa', got {gqa_implementation!r}."
            )
        self.force_flash_attention = bool(force_flash_attention)
        self.jax_naive_attention = "1" == "1"
        if self.jax_naive_attention and self.force_flash_attention:
            raise ValueError(
                "IMAGEWAM_JAX_NAIVE_ATTENTION=1 is incompatible with force_flash_attention=True."
            )
        if mot_checkpoint_mixed_attn:
            logger.info("Using gradient checkpointing for mixture attention.")
        first_expert = self.mixtures[self.expert_order[0]]
        self.num_layers = len(first_expert.blocks)
        self.num_heads = int(first_expert.num_heads)
        self.num_kv_heads = int(getattr(first_expert, "num_kv_heads", first_expert.num_heads))
        self.attn_head_dim = int(first_expert.attn_head_dim)
        self.block_protocol = str(getattr(first_expert, "block_protocol", "wan22"))
        for name in self.expert_order[1:]:
            expert = self.mixtures[name]
            protocol = str(getattr(expert, "block_protocol", "wan22"))
            num_kv_heads = int(getattr(expert, "num_kv_heads", expert.num_heads))
            checks = {
                "num_layers": (len(expert.blocks), self.num_layers),
                "num_heads": (int(expert.num_heads), self.num_heads),
                "num_kv_heads": (num_kv_heads, self.num_kv_heads),
                "attn_head_dim": (int(expert.attn_head_dim), self.attn_head_dim),
                "block_protocol": (protocol, self.block_protocol),
            }
            for attr, (got, expected) in checks.items():
                if got != expected:
                    raise ValueError(
                        f"All experts must share {attr}; got {got} vs {expected} for expert {name}."
                    )
        logger.info(
            "Initialized MoT with experts=%s protocol=%s layers=%d heads=%d kv_heads=%d head_dim=%d",
            self.expert_order,
            self.block_protocol,
            self.num_layers,
            self.num_heads,
            self.num_kv_heads,
            self.attn_head_dim,
        )
        logger.info(
            "MoT attention config: gqa_implementation=%s force_flash_attention=%s jax_naive_attention=%s",
            self.gqa_implementation,
            self.force_flash_attention,
            self.jax_naive_attention,
        )

    @staticmethod
    def _format_attention_mask(
        attention_mask: torch.Tensor,
        batch_size: int,
        query_len: int,
        key_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        mask = attention_mask.to(device=device, dtype=torch.bool)
        if mask.ndim == 2:
            if tuple(mask.shape) != (query_len, key_len):
                raise ValueError(
                    f"2D attention mask must be {(query_len, key_len)}, got {tuple(mask.shape)}"
                )
            return mask.view(1, 1, query_len, key_len)
        if mask.ndim == 3:
            if mask.shape[0] != batch_size or tuple(mask.shape[1:]) != (query_len, key_len):
                raise ValueError(
                    f"3D attention mask must be {(batch_size, query_len, key_len)}, got {tuple(mask.shape)}"
                )
            return mask.unsqueeze(1)
        if mask.ndim == 4:
            if mask.shape[0] not in (1, batch_size) or tuple(mask.shape[-2:]) != (
                query_len,
                key_len,
            ):
                raise ValueError(
                    f"4D attention mask must end with {(query_len, key_len)}, got {tuple(mask.shape)}"
                )
            return mask
        raise ValueError(f"attention_mask must be 2D/3D/4D, got shape {tuple(mask.shape)}")

    def _mixed_attention(
        self,
        q_cat: torch.Tensor,
        k_cat: torch.Tensor,
        v_cat: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attn_probs: bool = False,
        skip_checkpoint: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch_size, query_len, _ = q_cat.shape
        key_len = k_cat.shape[1]
        H, H_kv, D = (self.num_heads, self.num_kv_heads, self.attn_head_dim)
        attn_mask = self._format_attention_mask(
            attention_mask, batch_size, query_len, key_len, q_cat.device
        )
        if return_attn_probs:
            q = q_cat.view(batch_size, query_len, H, D).transpose(1, 2)
            k = k_cat.view(batch_size, key_len, H_kv, D).transpose(1, 2)
            v = v_cat.view(batch_size, key_len, H_kv, D).transpose(1, 2)
            if H_kv != H:
                if self.gqa_implementation != "repeat":
                    raise ValueError(
                        "Attention capture with GQA currently requires gqa_implementation='repeat'."
                    )
                repeat_factor = H // H_kv
                k = k.repeat_interleave(repeat_factor, dim=1)
                v = v.repeat_interleave(repeat_factor, dim=1)
            scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * D ** (-0.5)
            scores = scores.masked_fill(~attn_mask, torch.finfo(scores.dtype).min)
            attn_probs = torch.softmax(scores, dim=-1).to(dtype=v.dtype)
            out = torch.matmul(attn_probs, v)
            out = out.transpose(1, 2).reshape(batch_size, query_len, H * D)
            return (out, attn_probs)

        def _sdpa_context():
            if not self.force_flash_attention:
                return nullcontext()
            try:
                from torch.nn.attention import SDPBackend, sdpa_kernel
            except Exception as exc:
                raise RuntimeError(
                    "`force_flash_attention=True` requires torch.nn.attention.sdpa_kernel."
                ) from exc
            return sdpa_kernel([SDPBackend.FLASH_ATTENTION])
            force_flash_context = sdpa_kernel([SDPBackend.FLASH_ATTENTION])

        def _forward(
            q_flat: torch.Tensor, k_flat: torch.Tensor, v_flat: torch.Tensor
        ) -> torch.Tensor:
            q = q_flat.view(batch_size, query_len, H, D).transpose(1, 2)
            k = k_flat.view(batch_size, key_len, H_kv, D).transpose(1, 2)
            v = v_flat.view(batch_size, key_len, H_kv, D).transpose(1, 2)
            enable_gqa = False
            if H_kv != H and self.gqa_implementation == "repeat":
                repeat_factor = H // H_kv
                k = k.repeat_interleave(repeat_factor, dim=1)
                v = v.repeat_interleave(repeat_factor, dim=1)
            elif H_kv != H:
                enable_gqa = True
            if self.jax_naive_attention:
                if enable_gqa:
                    raise ValueError(
                        "JAX naive attention requires gqa_implementation='repeat' when num_kv_heads != num_heads."
                    )
                with torch.autocast(device_type=q.device.type, enabled=False):
                    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * D ** (-0.5)
                    scores = scores.masked_fill(~attn_mask, torch.finfo(scores.dtype).min)
                    attn_probs = torch.softmax(scores, dim=-1)
                out = torch.matmul(attn_probs.to(dtype=v.dtype), v)
            else:
                with _sdpa_context():
                    out = F.scaled_dot_product_attention(
                        q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa
                    )
            return out.transpose(1, 2).reshape(batch_size, query_len, H * D)

        if self.mot_checkpoint_mixed_attn and self.training and (not skip_checkpoint):
            return torch.utils.checkpoint.checkpoint(
                _forward, q_cat, k_cat, v_cat, use_reentrant=False
            )
        return _forward(q_cat, k_cat, v_cat)

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
        if "1" == "1" and (not torch.is_autocast_enabled()):
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
        jax_fp32_residual = "1" == "1"
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
        full_checkpoint = (
            bool(getattr(self, "checkpoint_flux2_blocks", False))
            and self.training
            and torch.is_grad_enabled()
        )
        for layer_idx in range(int(getattr(video_expert, "double_layers"))):
            v_block = video_expert.double_blocks[layer_idx]
            a_block = action_expert.double_blocks[layer_idx]
            fn = partial(self._flux2_double_joint, v_block, a_block, full_checkpoint)
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
            img, txt, action = (
                torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)
                if full_checkpoint
                else fn(*args)
            )
        video_stream = torch.cat([txt, img], dim=1)
        stream_pe = torch.cat([txt_pe, img_pe], dim=2)
        for layer_idx in range(int(getattr(video_expert, "single_layers"))):
            v_block = video_expert.single_blocks[layer_idx]
            a_block = action_expert.single_blocks[layer_idx]
            fn = partial(self._flux2_single_joint, v_block, a_block, full_checkpoint)
            args = (
                video_stream,
                action,
                stream_pe,
                action_pe,
                video_t_mod["single"],
                action_t_mod["single"],
                attention_mask["single"],
            )
            video_stream, action = (
                torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)
                if full_checkpoint
                else fn(*args)
            )
        txt_len = int(txt.shape[1])
        txt, img = (video_stream[:, :txt_len], video_stream[:, txt_len:])
        return {"video": {"txt": txt, "img": img}, "action": action}

    def _flux2_double_joint(
        self,
        v_block,
        a_block,
        full_checkpoint,
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
            skip_checkpoint=full_checkpoint,
        )
        va, aa = torch.split(mixed, [txt.shape[1] + img.shape[1], action.shape[1]], dim=1)
        ta, ia = torch.split(va, [nt, img.shape[1]], dim=1)
        img, txt = v_block._apply_residuals(img, txt, ia, ta, mods)
        return (img, txt, a_block.apply_post(aa, state))

    def _flux2_single_joint(
        self,
        v_block,
        a_block,
        full_checkpoint,
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
            skip_checkpoint=full_checkpoint,
        )
        va, aa = torch.split(mixed, [video.shape[1], action.shape[1]], dim=1)
        return (
            v_block._out(vs["residual_x"], va, vs["mlp"], vs["gate"]),
            a_block.apply_post(aa, ac),
        )

    def forward(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        freqs_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
    ):
        return self._forward_flux2(embeds_all, attention_mask, freqs_all, context_all, t_mod_all)
