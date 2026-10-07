"""Current-frame denoising and masked action regression."""

import torch
from torch.nn import functional as F


def sample_video_t(batch_size, device, shift=0.25):
    u = torch.rand((batch_size,), device=device, dtype=torch.float32)
    sigma = shift * u / (1.0 + (shift - 1.0) * u)
    return sigma * 1000.0


def video_flow_pair(clean_video, noise_video, timestep):
    clean_video = clean_video.to(torch.bfloat16)
    noise_video = noise_video.to(torch.bfloat16)
    sigma = (timestep.float() / 1000.0).view(-1, *([1] * (clean_video.ndim - 1)))
    noisy = ((1.0 - sigma) * clean_video.float() + sigma * noise_video.float()).to(torch.bfloat16)
    return noisy, noise_video - clean_video


def action_loss_per_sample(pred_action, target_action, action_is_pad, action_dim_is_pad=None):
    action_loss_dim = F.mse_loss(pred_action.float(), target_action.float(), reduction="none")
    if action_dim_is_pad is not None:
        dim_valid = (~action_dim_is_pad).to(
            device=action_loss_dim.device, dtype=action_loss_dim.dtype
        )
        dim_valid_sum = dim_valid.sum(dim=1).clamp(min=1.0).unsqueeze(1)
        action_loss_token = (action_loss_dim * dim_valid.unsqueeze(1)).sum(dim=2) / dim_valid_sum
    else:
        action_loss_token = action_loss_dim.mean(dim=2)
    if action_is_pad is not None:
        valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (action_loss_token * valid).sum(dim=1) / valid_sum
    return action_loss_token.mean(dim=1)
