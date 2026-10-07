import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf
from .utils.logging_config import get_logger

logger = get_logger(__name__)


def create_imagewam_flux2_klein(
    flux2_model_path: str,
    ae_model_path: str,
    flux2_src_path: str | None = None,
    variant: str = "klein-base-4b",
    qwen3_model_spec: str | None = None,
    qwen_context_len: int = 512,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    proprio_dim: int | None = None,
    load_text_encoder: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    mot_gqa_implementation: str = "repeat",
    mot_force_flash_attention: bool = False,
    pack_proprio_after_text: bool = True,
    flux2_lora_config=None,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    dift_mode: bool = False,
    action_regression: bool = False,
    dift_tshift: float = 0.25,
    keep_double: int | None = None,
    keep_single: int | None = None,
    compile_backbone: bool = False,
    frozen_qwen_dtype: str | None = None,
    frozen_vae_dtype: str | None = None,
    checkpoint_flux2_blocks: bool = False,
):
    from .models.backbones.imagewam import ImageWAM

    def encoder_dtype(value):
        if value is None:
            return None
        choices = {"bf16": torch.bfloat16, "fp32": torch.float32}
        if value not in choices:
            raise ValueError(f"Frozen encoder dtype must be bf16/fp32/null, got {value!r}")
        return choices[value]

    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if action_dit_config is None:
        action_dit_config = {}
    if not isinstance(action_dit_config, dict):
        raise ValueError(
            f"`action_dit_config` must resolve to a dict, got {type(action_dit_config)}"
        )
    if isinstance(video_scheduler, DictConfig):
        video_scheduler = OmegaConf.to_container(video_scheduler, resolve=True)
    if video_scheduler is None:
        video_scheduler = {}
    if not isinstance(video_scheduler, dict):
        raise ValueError(f"`video_scheduler` must be dict-like, got {type(video_scheduler)}")
    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
    if action_scheduler is None:
        raise ValueError("`action_scheduler` is required for ImageWAM FLUX.2 stack.")
    if not isinstance(action_scheduler, dict):
        raise ValueError(f"`action_scheduler` must be dict-like, got {type(action_scheduler)}")
    if isinstance(loss, DictConfig):
        loss = OmegaConf.to_container(loss, resolve=True)
    if loss is None:
        loss = {}
    if not isinstance(loss, dict):
        raise ValueError(f"`loss` must be dict-like, got {type(loss)}")
    if isinstance(flux2_lora_config, DictConfig):
        flux2_lora_config = OmegaConf.to_container(flux2_lora_config, resolve=True)
    if flux2_lora_config is None:
        flux2_lora_config = {}
    if not isinstance(flux2_lora_config, dict):
        raise ValueError(f"`flux2_lora_config` must be dict-like, got {type(flux2_lora_config)}")
    model = ImageWAM.from_flux2_klein_pretrained(
        flux2_model_path=flux2_model_path,
        ae_model_path=ae_model_path,
        flux2_src_path=flux2_src_path,
        variant=str(variant),
        qwen3_model_spec=qwen3_model_spec,
        qwen_context_len=int(qwen_context_len),
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        proprio_dim=None if proprio_dim is None else int(proprio_dim),
        load_text_encoder=bool(load_text_encoder),
        device=device,
        torch_dtype=model_dtype,
        frozen_qwen_dtype=encoder_dtype(frozen_qwen_dtype),
        frozen_vae_dtype=encoder_dtype(frozen_vae_dtype),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        mot_gqa_implementation=str(mot_gqa_implementation),
        mot_force_flash_attention=bool(mot_force_flash_attention),
        pack_proprio_after_text=bool(pack_proprio_after_text),
        flux2_lora_config=flux2_lora_config,
        dift_mode=bool(dift_mode),
        action_regression=bool(action_regression),
        dift_tshift=float(dift_tshift),
        keep_double=None if keep_double is None else int(keep_double),
        keep_single=None if keep_single is None else int(keep_single),
    )
    model.mot.checkpoint_flux2_blocks = bool(checkpoint_flux2_blocks)
    if bool(compile_backbone):
        model.mot.forward = torch.compile(model.mot.forward)
    return model
