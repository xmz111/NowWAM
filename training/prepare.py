"""Prepare the action expert from the public base transformer."""

import argparse
import os
from pathlib import Path

import torch
from torch.nn import functional as F
from safetensors.torch import load_file

from ._core.models.backbones.action_dit_flux2 import ActionDiTFlux2


def resize_tensor(source, shape):
    if tuple(source.shape) == tuple(shape):
        return source
    value = source.float()
    while value.ndim < len(shape):
        value = value.unsqueeze(0)
    while value.ndim > len(shape):
        if value.shape[0] != 1:
            raise ValueError("Cannot reduce a non-singleton tensor dimension")
        value = value.squeeze(0)
    for dimension, size in enumerate(shape):
        if value.shape[dimension] == size:
            continue
        permutation = [i for i in range(value.ndim) if i != dimension] + [dimension]
        inverse = [0] * value.ndim
        for i, p in enumerate(permutation):
            inverse[p] = i
        value = value.permute(*permutation).contiguous()
        leading = value.shape[:-1]
        value = F.interpolate(
            value.reshape(-1, 1, value.shape[-1]), size=size, mode="linear", align_corners=True
        ).reshape(*leading, size)
        value = value.permute(*inverse).contiguous()
    return value.to(dtype=source.dtype)


def prepare(assets):
    from ._core.models.backbones.flux2_imports import ensure_flux2_importable

    ensure_flux2_importable(str(assets / "sources/flux2"))
    from flux2.model import Flux2, Klein4BParams

    destination = assets / "action_init.pt"
    if destination.exists():
        raise FileExistsError(f"Refusing to replace {destination}")
    with torch.device("meta"):
        transformer = Flux2(Klein4BParams()).to(torch.bfloat16)
    source = load_file(str(assets / "FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors"))
    transformer.load_state_dict(source, strict=True, assign=True)
    action = ActionDiTFlux2(
        action_dim=7,
        hidden_dim=1024,
        num_heads=24,
        attn_head_dim=128,
        num_layers_double=5,
        num_layers_single=20,
        mlp_ratio=4.0,
        max_action_horizon=64,
        use_gradient_checkpointing=True,
    )
    output = {}
    for key, target in action.state_dict().items():
        if key.startswith(("action_encoder.", "head.")):
            continue
        if key not in source:
            raise KeyError(f"Base transformer has no initializer for {key}")
        original = source[key]
        value = resize_tensor(original, target.shape)
        if (
            original.shape != target.shape
            and original.ndim >= 2
            and original.shape[-1] != target.shape[-1]
        ):
            value = value.float() * (float(original.shape[-1]) / float(target.shape[-1])) ** 0.5
        output[key] = value.cpu().to(dtype=target.dtype).contiguous()
    temporary = destination.with_suffix(".pt.partial")
    torch.save({"state_dict": output}, temporary)
    os.replace(temporary, destination)
    print(f"ACTION_INITIALIZER_PREPARED tensors={len(output)}", flush=True)


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    prepare(Path("assets"))
