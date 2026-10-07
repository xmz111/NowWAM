from __future__ import annotations
import math
from dataclasses import dataclass
from collections import OrderedDict
from typing import Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F
from training._core.utils.logging_config import get_logger

logger = get_logger(__name__)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"`rank` must be positive, got {rank}")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(self.rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.base.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out.to(dtype=base_out.dtype)

    def merged_linear(self) -> nn.Linear:
        merged = nn.Linear(
            self.base.in_features,
            self.base.out_features,
            bias=self.base.bias is not None,
            device=self.base.weight.device,
            dtype=self.base.weight.dtype,
        )
        delta = self.lora_B.float() @ self.lora_A.float() * float(self.scaling)
        merged.weight.data.copy_(
            (self.base.weight.float() + delta).to(dtype=self.base.weight.dtype)
        )
        if self.base.bias is not None:
            merged.bias.data.copy_(self.base.bias.data)
        return merged


@dataclass
class LoRAApplyResult:
    replaced: int
    target_suffixes: tuple[str, ...]


def _iter_named_linears(module: nn.Module) -> Iterable[tuple[str, nn.Linear]]:
    for name, child in module.named_modules():
        if isinstance(child, nn.Linear):
            yield (name, child)


def _get_parent(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return (parent, parts[-1])


def apply_lora_to_linear_suffixes(
    module: nn.Module,
    *,
    target_suffixes: Iterable[str],
    rank: int,
    alpha: float,
    dropout: float = 0.0,
) -> LoRAApplyResult:
    suffixes = tuple((str(item) for item in target_suffixes))
    replaced = 0
    for name, linear in list(_iter_named_linears(module)):
        if isinstance(linear, LoRALinear):
            continue
        if not any((name.endswith(suffix) for suffix in suffixes)):
            continue
        parent, child_name = _get_parent(module, name)
        setattr(parent, child_name, LoRALinear(linear, rank=rank, alpha=alpha, dropout=dropout))
        replaced += 1
    logger.info("Applied LoRA to %d Linear layers; target_suffixes=%s", replaced, suffixes)
    return LoRAApplyResult(replaced=replaced, target_suffixes=suffixes)


def merge_lora_linear_layers(module: nn.Module) -> int:
    replaced = 0
    for name, child in list(module.named_modules()):
        if not isinstance(child, LoRALinear):
            continue
        parent, child_name = _get_parent(module, name)
        setattr(parent, child_name, child.merged_linear())
        replaced += 1
    logger.info("Merged %d LoRA Linear layers into base weights.", replaced)
    return replaced


def lora_merged_state_dict(module: nn.Module) -> OrderedDict[str, torch.Tensor]:
    lora_modules = {
        name: child for name, child in module.named_modules() if isinstance(child, LoRALinear)
    }
    merged = OrderedDict()
    for name, child in lora_modules.items():
        linear = child.merged_linear()
        merged[f"{name}.weight"] = linear.weight.detach().cpu()
        if linear.bias is not None:
            merged[f"{name}.bias"] = linear.bias.detach().cpu()
    for key, value in module.state_dict().items():
        if any((key == prefix or key.startswith(prefix + ".") for prefix in lora_modules)):
            continue
        merged[key] = value.detach().cpu()
    return merged


def remap_plain_linear_keys_to_lora_base(
    module: nn.Module, state_dict: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    lora_names = {name for name, child in module.named_modules() if isinstance(child, LoRALinear)}
    if not lora_names:
        return state_dict
    remapped = {}
    for key, value in state_dict.items():
        mapped_key = key
        for name in lora_names:
            if key == f"{name}.weight":
                mapped_key = f"{name}.base.weight"
                break
            if key == f"{name}.bias":
                mapped_key = f"{name}.base.bias"
                break
        remapped[mapped_key] = value
    return remapped


def merge_lora_state_dict_to_plain(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefixes = set()
    for key in state_dict:
        if key.endswith(".lora_A"):
            prefixes.add(key[: -len(".lora_A")])
    if not prefixes:
        return state_dict
    merged: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()
    for prefix in prefixes:
        base_key = f"{prefix}.base.weight"
        a_key = f"{prefix}.lora_A"
        b_key = f"{prefix}.lora_B"
        if base_key not in state_dict or a_key not in state_dict or b_key not in state_dict:
            continue
        base = state_dict[base_key]
        lora_a = state_dict[a_key]
        lora_b = state_dict[b_key]
        delta = lora_b.float() @ lora_a.float()
        merged[f"{prefix}.weight"] = (base.float() + delta).to(dtype=base.dtype)
        consumed.update({base_key, a_key, b_key})
        bias_key = f"{prefix}.base.bias"
        if bias_key in state_dict:
            merged[f"{prefix}.bias"] = state_dict[bias_key]
            consumed.add(bias_key)
    for key, value in state_dict.items():
        if key in consumed:
            continue
        if ".lora_A" in key or ".lora_B" in key or ".base." in key:
            continue
        merged[key] = value
    return merged
