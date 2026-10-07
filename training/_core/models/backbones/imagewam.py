import os
from typing import Any, Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from training._core.utils.logging_config import get_logger
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler

logger = get_logger(__name__)


class ImageWAM(torch.nn.Module):
    def __init__(
        self,
        video_expert,
        action_expert: Any,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        stack: str = "wan22",
        omnigen2_online_text_cache_compatible: bool = False,
        qwen_context_len: int = 128,
        pack_proprio_after_text: bool = False,
        dift_mode: bool = False,
        action_regression: bool = False,
        dift_tshift: float = 0.25,
    ):
        super().__init__()
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.mot = mot
        self.dit = self.mot
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError("`text_dim` is required when `text_encoder` is not loaded.")
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
        else:
            self.proprio_encoder = None
        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps, shift=video_train_shift
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps, shift=video_infer_shift
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps, shift=action_train_shift
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps, shift=action_infer_shift
        )
        self.train_scheduler = self.train_video_scheduler
        self.infer_scheduler = self.infer_video_scheduler
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)
        self.stack = str(stack)
        self.omnigen2_online_text_cache_compatible = bool(omnigen2_online_text_cache_compatible)
        self.qwen_context_len = int(qwen_context_len)
        self.pack_proprio_after_text = bool(pack_proprio_after_text)
        self.dift_mode = bool(dift_mode)
        self.action_regression = bool(action_regression)
        self.dift_tshift = float(dift_tshift)
        self.train_video_scheduler_dift = WanContinuousFlowMatchScheduler(
            num_train_timesteps=self.train_video_scheduler.num_train_timesteps,
            shift=self.dift_tshift,
        )
        expected_dift_tshift = "0.25"
        if expected_dift_tshift is not None:
            expected_dift_tshift = float(expected_dift_tshift)
            if not self.dift_mode:
                raise RuntimeError(
                    "IMAGEWAM_EXPECT_DIFT_TSHIFT was set but model.dift_mode is false"
                )
            if abs(self.dift_tshift - expected_dift_tshift) > 1e-12:
                raise RuntimeError(
                    f"Configured DIFT shift does not match launcher contract: model={self.dift_tshift} expected={expected_dift_tshift}"
                )
            if abs(self.train_video_scheduler_dift.shift - expected_dift_tshift) > 1e-12:
                raise RuntimeError(
                    f"Constructed DIFT scheduler does not match launcher contract: active={self.train_video_scheduler_dift.shift} expected={expected_dift_tshift}"
                )
        self.to(self.device)

    @classmethod
    def from_flux2_klein_pretrained(
        cls,
        flux2_model_path: str,
        ae_model_path: str,
        action_dit_config: dict[str, Any],
        action_dit_pretrained_path: str | None = None,
        flux2_src_path: str | None = None,
        variant: str = "klein-base-4b",
        proprio_dim: Optional[int] = None,
        load_text_encoder: bool = False,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        mot_gqa_implementation: str = "repeat",
        mot_force_flash_attention: bool = False,
        pack_proprio_after_text: bool = True,
        flux2_lora_config: Optional[dict[str, Any]] = None,
        qwen3_model_spec: str | None = None,
        qwen_context_len: int = 512,
        dift_mode: bool = False,
        action_regression: bool = False,
        dift_tshift: float = 0.25,
        keep_double: int | None = None,
        keep_single: int | None = None,
        frozen_qwen_dtype: torch.dtype | None = None,
        frozen_vae_dtype: torch.dtype | None = None,
    ):
        from safetensors.torch import load_file as load_sft
        from .action_dit_flux2 import ActionDiTFlux2
        from .flux2_imports import ensure_flux2_importable
        from .flux2_video_expert import Flux2VideoExpert
        from .mot import MoT

        ensure_flux2_importable(flux2_src_path)
        from flux2.autoencoder import AutoEncoder, AutoEncoderParams

        key = str(variant).lower().replace("_", "-")
        if key in {"klein-base-4b", "flux.2-klein-base-4b", "4b", "base-4b"}:
            text_dim = 7680
            default_qwen3_model_spec = "Qwen/Qwen3-4B"
        elif key in {"klein-base-9b", "flux.2-klein-base-9b", "9b", "base-9b"}:
            text_dim = 12288
            default_qwen3_model_spec = "Qwen/Qwen3-8B"
        else:
            raise ValueError(f"Unsupported FLUX.2 Klein variant: {variant!r}")
        video_expert = Flux2VideoExpert.from_pretrained(
            flux2_model_path=flux2_model_path,
            variant=key,
            flux2_src_path=flux2_src_path,
            device=device,
            torch_dtype=torch_dtype,
        )
        flux2_lora_config = dict(flux2_lora_config or {})
        if bool(flux2_lora_config.get("enabled", False)):
            from .lora import apply_lora_to_linear_suffixes

            target_suffixes = flux2_lora_config.get(
                "target_suffixes",
                [
                    "qkv",
                    "proj",
                    "linear1",
                    "linear2",
                    "img_mlp.0",
                    "img_mlp.2",
                    "txt_mlp.0",
                    "txt_mlp.2",
                ],
            )
            apply_lora_to_linear_suffixes(
                video_expert.transformer,
                target_suffixes=target_suffixes,
                rank=int(flux2_lora_config.get("rank", 16)),
                alpha=float(flux2_lora_config.get("alpha", 16.0)),
                dropout=float(flux2_lora_config.get("dropout", 0.0)),
            )
            video_expert.flux2_lora_enabled = True
            video_expert.flux2_lora_target_suffixes = tuple((str(item) for item in target_suffixes))
        else:
            video_expert.flux2_lora_enabled = False
        _t = video_expert.transformer
        if keep_double is not None and int(keep_double) < len(_t.double_blocks):
            _t.double_blocks = _t.double_blocks[: int(keep_double)]
            video_expert.double_blocks = _t.double_blocks
            video_expert.double_layers = len(_t.double_blocks)
        if keep_single is not None and int(keep_single) < len(_t.single_blocks):
            _t.single_blocks = _t.single_blocks[: int(keep_single)]
            video_expert.single_blocks = _t.single_blocks
            video_expert.single_layers = len(_t.single_blocks)
        action_cfg = dict(action_dit_config)
        expected_action_shape = {
            "num_heads": int(video_expert.num_heads),
            "attn_head_dim": int(video_expert.attn_head_dim),
            "num_layers_double": int(video_expert.double_layers),
            "num_layers_single": int(video_expert.single_layers),
        }
        action_cfg.setdefault("hidden_dim", 1024)
        for key_name, expected_value in expected_action_shape.items():
            if key_name in action_cfg and int(action_cfg[key_name]) != int(expected_value):
                logger.warning(
                    "Overriding action_dit_config.%s=%s to match FLUX.2 value %s.",
                    key_name,
                    action_cfg[key_name],
                    expected_value,
                )
            action_cfg[key_name] = expected_value
        action_expert = ActionDiTFlux2.from_pretrained(
            action_dit_config=action_cfg,
            action_dit_pretrained_path=action_dit_pretrained_path,
            device=device,
            torch_dtype=torch_dtype,
        )
        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
            gqa_implementation=mot_gqa_implementation,
            force_flash_attention=mot_force_flash_attention,
        )
        vae_encoder_only = "0" == "1"
        if vae_encoder_only:
            ae = AutoEncoder(AutoEncoderParams())
        else:
            with torch.device("meta"):
                ae = AutoEncoder(AutoEncoderParams())
        ae_state = load_sft(str(ae_model_path), device=str(device))
        ae.load_state_dict(ae_state, strict=not vae_encoder_only, assign=True)
        ae = ae.to(device=device, dtype=frozen_vae_dtype or torch_dtype).eval()
        ae.requires_grad_(False)
        if load_text_encoder:
            from types import SimpleNamespace
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_spec = qwen3_model_spec or default_qwen3_model_spec
            text_encoder_device = str(device)
            qwen3_model = (
                AutoModelForCausalLM.from_pretrained(
                    model_spec, torch_dtype=frozen_qwen_dtype or torch_dtype
                )
                .to(text_encoder_device)
                .eval()
            )
            qwen3_model.requires_grad_(False)
            text_encoder = SimpleNamespace(
                model=qwen3_model,
                tokenizer=AutoTokenizer.from_pretrained(model_spec),
                max_length=int(qwen_context_len),
            )
        else:
            text_encoder = None
        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=ae,
            text_encoder=text_encoder,
            tokenizer=None,
            text_dim=text_dim,
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            stack="flux2",
            qwen_context_len=int(qwen_context_len),
            pack_proprio_after_text=bool(pack_proprio_after_text),
            dift_mode=bool(dift_mode),
            action_regression=bool(action_regression),
            dift_tshift=float(dift_tshift),
        )
        model.model_paths = {
            "flux2": flux2_model_path,
            "flux2_src": flux2_src_path,
            "ae": ae_model_path,
            "action_dit": action_dit_pretrained_path,
            "qwen3": qwen3_model_spec or default_qwen3_model_spec,
        }
        model.flux2_qwen3_model_spec = qwen3_model_spec or default_qwen3_model_spec
        model.frozen_qwen_dtype = frozen_qwen_dtype
        model.frozen_vae_dtype = frozen_vae_dtype
        model.save_lora_merged = bool(
            flux2_lora_config.get("save_lora_merged", bool(flux2_lora_config.get("enabled", False)))
        )
        model.save_trainable_only = bool(flux2_lora_config.get("save_trainable_only", False))
        return model

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        if self.text_encoder is not None and "" != "cpu":
            if hasattr(self.text_encoder, "to"):
                self.text_encoder.to(*args, **kwargs)
            elif hasattr(self.text_encoder, "model") and hasattr(self.text_encoder.model, "to"):
                self.text_encoder.model.to(*args, **kwargs)
        if hasattr(self, "dim_projector"):
            self.dim_projector.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        if getattr(self, "frozen_qwen_dtype", None) is not None and self.text_encoder is not None:
            self.text_encoder.model.to(dtype=self.frozen_qwen_dtype)
        if getattr(self, "frozen_vae_dtype", None) is not None:
            self.vae.to(dtype=self.frozen_vae_dtype)
        return self

    @staticmethod
    def _scheduler_timestep_to_unit(timestep: torch.Tensor, scheduler) -> torch.Tensor:
        num_train_timesteps = float(getattr(scheduler, "num_train_timesteps", 1000))
        if num_train_timesteps <= 0:
            raise ValueError(f"`num_train_timesteps` must be positive, got {num_train_timesteps}.")
        return timestep / num_train_timesteps

    def _append_proprio_to_context(
        self, context: torch.Tensor, context_mask: torch.Tensor, proprio: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None or proprio is None:
            return (context, context_mask)
        if proprio.ndim != 2:
            raise ValueError(f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}")
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
            )
        proprio_token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype)
        if not getattr(self, "pack_proprio_after_text", False):
            proprio_mask = torch.ones(
                (context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device
            )
            return (
                torch.cat([context, proprio_token], dim=1),
                torch.cat([context_mask, proprio_mask], dim=1),
            )
        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        if context.shape[:2] != context_mask.shape:
            raise ValueError(
                f"`context/context_mask` leading dims must match, got {tuple(context.shape[:2])} and {tuple(context_mask.shape)}"
            )
        if context.shape[0] != proprio_token.shape[0]:
            raise ValueError(
                f"`proprio` batch size must match context batch size ({context.shape[0]}), got {proprio_token.shape[0]}"
            )
        context_mask = context_mask.to(device=context.device, dtype=torch.bool)
        new_context = context.new_zeros(context.shape[0], context.shape[1] + 1, context.shape[2])
        valid_counts = context_mask.sum(dim=1)
        valid_rank = context_mask.cumsum(dim=1) - 1
        invalid_mask = ~context_mask
        invalid_rank = invalid_mask.cumsum(dim=1) - 1
        target_indices = torch.where(
            context_mask, valid_rank, valid_counts[:, None] + 1 + invalid_rank
        )
        new_context.scatter_(
            dim=1, index=target_indices[:, :, None].expand(-1, -1, context.shape[2]), src=context
        )
        batch_indices = torch.arange(context.shape[0], device=context.device)
        new_context[batch_indices, valid_counts] = proprio_token[:, 0]
        positions = torch.arange(context.shape[1] + 1, device=context.device)
        new_context_mask = positions[None, :] <= valid_counts[:, None]
        return (new_context, new_context_mask)

    def _append_proprio_to_context_if_enabled(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
        source: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None:
            return (context, context_mask)
        if proprio is None:
            raise ValueError(f"`{source}` requires `proprio` when `proprio_dim` is enabled.")
        if proprio.ndim == 3:
            proprio = proprio[:, 0, :]
        elif proprio.ndim == 2:
            pass
        elif proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        else:
            raise ValueError(
                f"`{source}` `proprio` must be [B,T,D], [B,D], or [D], got shape {tuple(proprio.shape)}"
            )
        if proprio.shape[0] != context.shape[0]:
            raise ValueError(
                f"`{source}` `proprio` batch size must match context batch size ({context.shape[0]}), got {proprio.shape[0]}"
            )
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"`{source}` `proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
            )
        return self._append_proprio_to_context(
            context=context,
            context_mask=context_mask,
            proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
        )

    @torch.no_grad()
    def _encode_flux2_image_tokens(
        self, image: torch.Tensor, *, time_value: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from .flux2_video_expert import Flux2VideoExpert

        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"`image` must be [B,3,H,W] or [3,H,W], got {tuple(image.shape)}")
        if image.shape[-2] % 16 != 0 or image.shape[-1] % 16 != 0:
            raise ValueError(
                f"FLUX.2 image spatial dims must be multiples of 16, got {tuple(image.shape[-2:])}"
            )
        image = image.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        if getattr(self, "frozen_vae_dtype", None) is not None:
            image = image.to(dtype=self.frozen_vae_dtype)
        latents = self.vae.encode(image).to(dtype=self.torch_dtype)
        tokens = Flux2VideoExpert.pack_latents(latents)
        _, _, latent_h, latent_w = latents.shape
        ids = Flux2VideoExpert.build_img_ids(
            batch_size=int(latents.shape[0]),
            token_height=int(latent_h),
            token_width=int(latent_w),
            time_value=float(time_value),
            device=tokens.device,
            dtype=tokens.dtype,
        )
        return (tokens, ids)

    @torch.no_grad()
    def _encode_flux2_text(self, sample) -> tuple[torch.Tensor, torch.Tensor]:
        cached_text_hidden_states = sample.get("text_hidden_states")
        if cached_text_hidden_states is not None:
            text_hidden_states = cached_text_hidden_states.to(
                device=self.device, dtype=self.torch_dtype, non_blocking=True
            )
            text_attention_mask = sample.get("text_attention_mask")
            if text_attention_mask is None:
                raise ValueError(
                    "FLUX.2 cached `text_hidden_states` must be paired with `text_attention_mask`."
                )
            else:
                text_attention_mask = text_attention_mask.to(
                    device=self.device, dtype=torch.bool, non_blocking=True
                )
            return (text_hidden_states, text_attention_mask)
        prompt = sample.get("instruction", sample.get("prompt", sample.get("task")))
        if prompt is None:
            raise ValueError(
                "FLUX.2 stack requires precomputed `text_hidden_states` or an `instruction`/`prompt`/`task`."
            )
        if self.text_encoder is None:
            raise ValueError("FLUX.2 online text encoding requires `load_text_encoder=true`.")
        video = sample.get("video")
        batch_size = (
            int(video.shape[0]) if isinstance(video, torch.Tensor) and video.ndim == 5 else 1
        )
        prompts = [prompt] * batch_size if isinstance(prompt, str) else list(prompt)
        return self._encode_flux2_prompts(prompts)

    @torch.no_grad()
    def _encode_flux2_prompts(self, prompts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.text_encoder is None:
            self._load_flux2_text_encoder_for_inference()
        if not hasattr(self.text_encoder, "tokenizer") or not hasattr(self.text_encoder, "model"):
            text_hidden = self.text_encoder(list(prompts)).to(
                device=self.device, dtype=self.torch_dtype
            )
            text_mask = torch.ones(text_hidden.shape[:2], device=self.device, dtype=torch.bool)
            return (text_hidden, text_mask)
        ensure_flux2_importable_path = getattr(self, "model_paths", {}).get("flux2_src")
        from .flux2_imports import ensure_flux2_importable

        ensure_flux2_importable(ensure_flux2_importable_path)
        from flux2.text_encoder import OUTPUT_LAYERS_QWEN3

        tokenizer = self.text_encoder.tokenizer
        model = self.text_encoder.model
        max_length = int(getattr(self.text_encoder, "max_length", 512))
        all_input_ids = []
        all_attention_masks = []
        for prompt in prompts:
            messages = [{"role": "user", "content": str(prompt)}]
            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
            except TypeError:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            model_inputs = tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
            all_input_ids.append(model_inputs["input_ids"])
            all_attention_masks.append(model_inputs["attention_mask"])
        input_ids = torch.cat(all_input_ids, dim=0).to(model.device)
        attention_mask = torch.cat(all_attention_masks, dim=0).to(model.device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden = torch.stack([outputs.hidden_states[k] for k in OUTPUT_LAYERS_QWEN3], dim=1)
        hidden = rearrange(hidden, "b c l d -> b l (c d)")
        return (
            hidden.to(device=self.device, dtype=self.torch_dtype),
            attention_mask.to(device=self.device, dtype=torch.bool),
        )

    def build_inputs_flux2(self, sample, tiled: bool = False):
        del tiled
        video = sample.get("video")
        if self.dift_mode:
            if video is None or video.ndim != 5:
                raise ValueError(
                    f"dift_mode requires `sample['video']` [B,C,T,H,W], got {(None if video is None else tuple(video.shape))}"
                )
            current_frame = video[:, :, 0]
            target_tokens, target_img_ids = self._encode_flux2_image_tokens(
                current_frame, time_value=10.0
            )
            ref_tokens = None
            ref_img_ids = None
        else:
            if "target_latent" in sample and "target_img_ids" in sample:
                target_tokens = sample["target_latent"].to(
                    device=self.device, dtype=self.torch_dtype, non_blocking=True
                )
                target_img_ids = sample["target_img_ids"].to(
                    device=self.device, dtype=self.torch_dtype, non_blocking=True
                )
            else:
                next_frame = sample.get("next_frame", sample.get("target_image"))
                if next_frame is None and video is not None:
                    if video.ndim != 5:
                        raise ValueError(
                            f"`sample['video']` must be [B,C,T,H,W], got {tuple(video.shape)}"
                        )
                    next_frame = video[:, :, -1]
                if next_frame is None:
                    raise ValueError(
                        "FLUX.2 stack sample requires `next_frame`, `target_image`, or target token fields."
                    )
                target_tokens, target_img_ids = self._encode_flux2_image_tokens(
                    next_frame, time_value=0.0
                )
            if "ref_image_latents" in sample and "ref_img_ids" in sample:
                ref_tokens = sample["ref_image_latents"].to(
                    device=self.device, dtype=self.torch_dtype, non_blocking=True
                )
                ref_img_ids = sample["ref_img_ids"].to(
                    device=self.device, dtype=self.torch_dtype, non_blocking=True
                )
            else:
                current_frame = sample.get("current_frame", sample.get("input_image"))
                if current_frame is None and video is not None:
                    if video.ndim != 5:
                        raise ValueError(
                            f"`sample['video']` must be [B,C,T,H,W], got {tuple(video.shape)}"
                        )
                    current_frame = video[:, :, 0]
                if current_frame is None:
                    raise ValueError(
                        "FLUX.2 stack sample requires `current_frame`, `input_image`, or ref token fields."
                    )
                ref_tokens, ref_img_ids = self._encode_flux2_image_tokens(
                    current_frame, time_value=10.0
                )
        text_hidden_states, text_attention_mask = self._encode_flux2_text(sample)
        if self.proprio_encoder is not None:
            text_hidden_states, text_attention_mask = self._append_proprio_to_context_if_enabled(
                context=text_hidden_states,
                context_mask=text_attention_mask,
                proprio=sample.get("proprio"),
                source="FLUX.2 training sample",
            )
        action = sample["action"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        action_is_pad = sample.get("action_is_pad")
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(
                device=self.device, dtype=torch.bool, non_blocking=True
            )
        action_dim_is_pad = sample.get("action_dim_is_pad")
        if action_dim_is_pad is not None:
            action_dim_is_pad = action_dim_is_pad.to(
                device=self.device, dtype=torch.bool, non_blocking=True
            )
        return {
            "target_latent": target_tokens,
            "target_img_ids": target_img_ids,
            "ref_image_latents": ref_tokens,
            "ref_img_ids": ref_img_ids,
            "text_hidden_states": text_hidden_states,
            "text_attention_mask": text_attention_mask,
            "action": action,
            "action_is_pad": action_is_pad,
            "action_dim_is_pad": action_dim_is_pad,
        }

    @torch.no_grad()
    def _build_mot_attention_mask_flux2(
        self,
        batch_size: int,
        txt_len: int,
        target_len: int,
        cond_len: int,
        action_len: int,
        device: torch.device,
        text_attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        t0 = 0
        r0 = txt_len
        x0 = txt_len + cond_len
        a0 = txt_len + cond_len + target_len
        total = a0 + action_len
        mask = torch.zeros(batch_size, total, total, dtype=torch.bool, device=device)
        if self.dift_mode:
            mask[:, t0:r0, t0:a0] = True
        else:
            mask[:, t0:r0, t0:x0] = True
        mask[:, r0:x0, t0:x0] = True
        mask[:, x0:a0, t0:a0] = True
        if self.dift_mode:
            mask[:, a0:total, t0:a0] = True
        else:
            mask[:, a0:total, t0:x0] = True
        mask[:, a0:total, a0:total] = True
        if text_attention_mask is not None:
            if text_attention_mask.ndim != 2 or tuple(text_attention_mask.shape) != (
                batch_size,
                txt_len,
            ):
                raise ValueError(
                    f"`text_attention_mask` must be [B,txt_len], got {tuple(text_attention_mask.shape)} for B={batch_size}, txt_len={txt_len}"
                )
            text_valid = text_attention_mask.to(device=device, dtype=torch.bool)
            mask[:, :, t0:r0] &= text_valid[:, None, :]
        return {"double_joint": mask, "single": mask.clone()}

    def _compute_action_loss_per_sample(
        self,
        pred_action: torch.Tensor,
        target_action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
        action_dim_is_pad: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        action_loss_dim = F.mse_loss(pred_action.float(), target_action.float(), reduction="none")
        if action_dim_is_pad is not None:
            dim_valid = (~action_dim_is_pad).to(
                device=action_loss_dim.device, dtype=action_loss_dim.dtype
            )
            dim_valid_sum = dim_valid.sum(dim=1).clamp(min=1.0).unsqueeze(1)
            action_loss_token = (action_loss_dim * dim_valid.unsqueeze(1)).sum(
                dim=2
            ) / dim_valid_sum
        else:
            action_loss_token = action_loss_dim.mean(dim=2)
        if action_is_pad is not None:
            valid = (~action_is_pad).to(
                device=action_loss_token.device, dtype=action_loss_token.dtype
            )
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            return (action_loss_token * valid).sum(dim=1) / valid_sum
        return action_loss_token.mean(dim=1)

    def _training_loss_flux2(self, sample, tiled: bool = False):
        inputs = self.build_inputs_flux2(sample, tiled=tiled)
        target_latent = inputs["target_latent"]
        action = inputs["action"]
        batch_size = int(target_latent.shape[0])
        if self.dift_mode and "1" == "1":
            clean_video = target_latent.to(dtype=torch.bfloat16)
            noise_video = torch.randn(
                clean_video.shape, device=clean_video.device, dtype=torch.bfloat16
            )
            timestep_video = self.train_video_scheduler_dift.sample_training_t(
                batch_size=batch_size, device=self.device, dtype=torch.float32
            )
            sigma_video = timestep_video / float(self.train_video_scheduler.num_train_timesteps)
            sigma_video = sigma_video.view(-1, *[1] * (target_latent.ndim - 1))
            noisy_latent = (
                (1.0 - sigma_video) * clean_video.float() + sigma_video * noise_video.float()
            ).to(dtype=torch.bfloat16)
            target_video = noise_video - clean_video
            expected_dtype = torch.bfloat16
            observed_dtypes = {
                "latent": clean_video.dtype,
                "noise": noise_video.dtype,
                "timestep": timestep_video.dtype,
                "model_input": noisy_latent.dtype,
                "target": target_video.dtype,
            }
            expected_dtypes = {
                "latent": expected_dtype,
                "noise": expected_dtype,
                "timestep": torch.float32,
                "model_input": expected_dtype,
                "target": expected_dtype,
            }
            if observed_dtypes != expected_dtypes:
                raise RuntimeError(
                    f"strict JAX mix dtype mismatch: observed={observed_dtypes} expected={expected_dtypes}"
                )
        elif self.dift_mode and "0" == "1":
            clean_video_fp32 = target_latent.float()
            noise_video_fp32 = torch.randn(
                target_latent.shape, device=target_latent.device, dtype=torch.float32
            )
            timestep_video = self.train_video_scheduler_dift.sample_training_t(
                batch_size=batch_size, device=self.device, dtype=torch.float32
            )
            sigma_video = timestep_video / float(self.train_video_scheduler.num_train_timesteps)
            sigma_video = sigma_video.view(-1, *[1] * (target_latent.ndim - 1))
            noisy_latent = (
                (1.0 - sigma_video) * clean_video_fp32 + sigma_video * noise_video_fp32
            ).to(dtype=target_latent.dtype)
            target_video = noise_video_fp32 - clean_video_fp32
        else:
            noise_video = torch.randn_like(target_latent)
            video_sampling_scheduler = (
                self.train_video_scheduler_dift if self.dift_mode else self.train_video_scheduler
            )
            if self.dift_mode and video_sampling_scheduler.shift != self.dift_tshift:
                raise RuntimeError(
                    f"DIFT timestep scheduler mismatch: active_shift={video_sampling_scheduler.shift} configured={self.dift_tshift}"
                )
            timestep_video = video_sampling_scheduler.sample_training_t(
                batch_size=batch_size, device=self.device, dtype=target_latent.dtype
            )
            noisy_latent = self.train_video_scheduler.add_noise(
                target_latent, noise_video, timestep_video
            )
            target_video = self.train_video_scheduler.training_target(
                target_latent, noise_video, timestep_video
            )
        strict_action_precision = self.action_regression and "1" == "1"
        if strict_action_precision:
            action = action.to(dtype=torch.bfloat16)
            inputs["text_hidden_states"] = inputs["text_hidden_states"].to(dtype=torch.bfloat16)
        noise_action = torch.randn_like(action)
        if self.action_regression:
            timestep_action = torch.full(
                (batch_size,),
                float(self.train_action_scheduler.num_train_timesteps),
                device=self.device,
                dtype=action.dtype,
            )
            noisy_action = noise_action
            target_action = action
        else:
            timestep_action = self.train_action_scheduler.sample_training_t(
                batch_size=batch_size, device=self.device, dtype=action.dtype
            )
            noisy_action = self.train_action_scheduler.add_noise(
                action, noise_action, timestep_action
            )
            target_action = self.train_action_scheduler.training_target(
                action, noise_action, timestep_action
            )
        jax_n2f_exact = self.dift_mode and "1" == "1"
        if jax_n2f_exact and "1" != "1":
            raise RuntimeError(
                "IMAGEWAM_JAX_N2F_EXACT=1 requires IMAGEWAM_JAX_STRICT_MIX_PRECISION=1"
            )
        video_x = noisy_latent
        video_ref = inputs["ref_image_latents"]
        video_target_ids = inputs["target_img_ids"]
        video_ref_ids = inputs["ref_img_ids"]
        if jax_n2f_exact:
            video_x = noisy_latent[:, :0]
            video_ref = noisy_latent
            video_target_ids = inputs["target_img_ids"][:, :0]
            video_ref_ids = inputs["target_img_ids"]
        video_pre = self.video_expert.pre_dit(
            x=video_x,
            timestep=self._scheduler_timestep_to_unit(
                timestep_video, self.train_video_scheduler
            ).to(device=noisy_latent.device, dtype=noisy_latent.dtype),
            context=inputs["text_hidden_states"],
            context_mask=inputs["text_attention_mask"],
            ref_image_hidden_states=video_ref,
            target_img_ids=video_target_ids,
            ref_img_ids=video_ref_ids,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=self._scheduler_timestep_to_unit(timestep_action, self.train_action_scheduler),
        )
        attention_mask = self._build_mot_attention_mask_flux2(
            batch_size=batch_size,
            txt_len=int(video_pre["txt_len"]),
            target_len=int(video_pre["target_len"]),
            cond_len=int(video_pre["cond_len"]),
            action_len=int(action_pre["tokens"].shape[1]),
            device=noisy_latent.device,
            text_attention_mask=video_pre["text_mask"],
        )
        tokens_out = self.mot(
            embeds_all={"video": video_pre["tokens"], "action": action_pre["tokens"]},
            attention_mask=attention_mask,
            freqs_all={"video": video_pre["freqs"]},
            context_all={"video": None, "action": {"ids": action_pre["ids"]}},
            t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
        )
        pred_video = (
            self.video_expert.post_dit_ref(tokens_out["video"], video_pre)
            if jax_n2f_exact
            else self.video_expert.post_dit(tokens_out["video"], video_pre)
        )
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        video_loss_per_sample = (
            F.mse_loss(pred_video.float(), target_video.float(), reduction="none")
            .flatten(1)
            .mean(dim=1)
        )
        if self.dift_mode:
            video_weight = torch.ones_like(video_loss_per_sample)
        else:
            video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
                video_loss_per_sample.device, dtype=video_loss_per_sample.dtype
            )
        loss_video = (video_loss_per_sample * video_weight).mean()
        action_loss_per_sample = self._compute_action_loss_per_sample(
            pred_action=pred_action,
            target_action=target_action,
            action_is_pad=inputs["action_is_pad"],
            action_dim_is_pad=inputs.get("action_dim_is_pad"),
        )
        if self.action_regression:
            action_weight = torch.ones_like(action_loss_per_sample)
        else:
            action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
                action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
            )
        loss_action = (action_loss_per_sample * action_weight).mean()
        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        return (
            loss_total,
            {
                "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
                "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
            },
        )

    def training_loss(self, sample, tiled: bool = False):
        return self._training_loss_flux2(sample, tiled=tiled)

    def save_checkpoint(self, path, optimizer=None, step=None):
        if bool(getattr(self, "save_lora_merged", False)):
            from .lora import lora_merged_state_dict

            mot_state = lora_merged_state_dict(self.mot)
            checkpoint_format = "lora_merged"
        elif bool(getattr(self, "save_trainable_only", False)):
            trainable_names = {
                name for name, param in self.mot.named_parameters() if param.requires_grad
            }
            mot_state = {
                key: value for key, value in self.mot.state_dict().items() if key in trainable_names
            }
            checkpoint_format = "trainable_only"
        else:
            mot_state = self.mot.state_dict()
            checkpoint_format = "full"
        payload = {"mot": mot_state, "step": step, "torch_dtype": str(self.torch_dtype)}
        if checkpoint_format != "full":
            payload["checkpoint_format"] = checkpoint_format
            payload["save_trainable_only"] = bool(getattr(self, "save_trainable_only", False))
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu")
        logger.info(
            "Loading ImageWAM checkpoint from %s with payload keys=%s step=%s",
            path,
            sorted(payload.keys()),
            payload.get("step"),
        )
        if "mot" in payload:
            mot_state = payload["mot"]
            if self.stack == "flux2":
                from .lora import (
                    merge_lora_state_dict_to_plain,
                    remap_plain_linear_keys_to_lora_base,
                )

                mot_state = merge_lora_state_dict_to_plain(mot_state)
                mot_state = remap_plain_linear_keys_to_lora_base(self.mot, mot_state)
            load_result = self.mot.load_state_dict(mot_state, strict=False)
            missing_keys = list(load_result.missing_keys)
            unexpected_keys = list(load_result.unexpected_keys)
            logger.info(
                "Loaded MoT weights from checkpoint: missing_keys=%d unexpected_keys=%d",
                len(missing_keys),
                len(unexpected_keys),
            )
            if missing_keys:
                logger.warning("First missing MoT keys: %s", missing_keys[:20])
            if unexpected_keys:
                logger.warning("First unexpected MoT keys: %s", unexpected_keys[:20])
            if missing_keys or unexpected_keys:
                raise RuntimeError(
                    f"Exact MoT checkpoint load required for formal evaluation, but missing_keys={len(missing_keys)} unexpected_keys={len(unexpected_keys)}"
                )
        elif "dit" in payload:
            logger.warning("Loading legacy `dit` checkpoint into video expert only.")
            load_result = self.video_expert.load_state_dict(payload["dit"], strict=False)
            logger.info(
                "Loaded legacy video expert weights: missing_keys=%d unexpected_keys=%d",
                len(load_result.missing_keys),
                len(load_result.unexpected_keys),
            )
        else:
            raise ValueError(f"Checkpoint missing both `mot` and `dit` keys: {path}")
        if self.proprio_encoder is not None:
            if "proprio_encoder" in payload:
                self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
                logger.info("Loaded proprio_encoder weights from checkpoint.")
            else:
                logger.warning(
                    "Checkpoint has no `proprio_encoder` weights; keeping current `proprio_encoder` params."
                )
        elif "proprio_encoder" in payload:
            logger.warning(
                "Checkpoint contains `proprio_encoder` weights but current model has `proprio_dim=None`; ignoring."
            )
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def apply_trainable_policy(self) -> None:
        if self.stack != "flux2":
            return
        video_expert = self.mot.mixtures["video"] if "video" in self.mot.mixtures else None
        action_expert = self.mot.mixtures["action"] if "action" in self.mot.mixtures else None
        if action_expert is not None:
            action_expert.train()
            action_expert.requires_grad_(True)
        if video_expert is None or not bool(getattr(video_expert, "flux2_lora_enabled", False)):
            return
        video_expert.train()
        for param in video_expert.parameters():
            param.requires_grad = False
        for name, param in video_expert.named_parameters():
            if ".lora_A" in name or ".lora_B" in name:
                param.requires_grad = True

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
