from typing import Dict
import torch
from torch.nn.functional import pad


class ConcatLeftAlign:
    def __init__(self, action_target_dim: int | None = None, state_target_dim: int | None = None):
        self.action_target_dim = action_target_dim
        self.state_target_dim = state_target_dim

    def set_shape_meta(self, shape_meta):
        self.action_meta = shape_meta["action"]
        self.state_meta = shape_meta["state"]

    def forward(self, batch):
        if "action" in batch:
            existing_action_dim_is_pad = batch.get("action_dim_is_pad")
            batch["action"] = self._concat(batch["action"], self.action_meta)
            batch["action"], pad_mask = self._pad(batch["action"], self.action_target_dim)
            batch["action_dim_is_pad"] = self._merge_mask(existing_action_dim_is_pad, pad_mask)
        existing_state_dim_is_pad = batch.get("state_dim_is_pad")
        batch["state"] = self._concat(batch["state"], self.state_meta)
        batch["state"], pad_mask = self._pad(batch["state"], self.state_target_dim)
        batch["state_dim_is_pad"] = self._merge_mask(existing_state_dim_is_pad, pad_mask)
        return batch

    def backward(self, batch):
        if self.state_target_dim is not None:
            assert batch["state"].shape[-1] == self.state_target_dim
        batch["state"] = self._crop(batch["state"], self.state_meta)
        batch["state"] = self._split(batch["state"], self.state_meta)
        if self.action_target_dim is not None:
            assert batch["action"].shape[-1] == self.action_target_dim
        batch["action"] = self._crop(batch["action"], self.action_meta)
        batch["action"] = self._split(batch["action"], self.action_meta)
        return batch

    @staticmethod
    def _pad(x: torch.Tensor, dim: int):
        if dim is None:
            dim = x.shape[-1]
        assert x.ndim == 2 and x.shape[-1] <= dim
        pad_dim = dim - x.shape[-1]
        x_padded = pad(x, (0, pad_dim))
        mask = torch.zeros_like(x[0]).bool()
        mask = pad(mask, (0, pad_dim), value=True)
        return (x_padded, mask)

    @staticmethod
    def _merge_mask(existing_mask, pad_mask: torch.Tensor):
        if existing_mask is None:
            return pad_mask
        existing_mask = torch.as_tensor(existing_mask, dtype=torch.bool, device=pad_mask.device)
        if existing_mask.ndim != 1:
            raise ValueError(f"Dimension mask must be 1D, got shape {tuple(existing_mask.shape)}")
        if existing_mask.shape[0] > pad_mask.shape[0]:
            raise ValueError(
                f"Existing dimension mask length {existing_mask.shape[0]} exceeds padded dim {pad_mask.shape[0]}"
            )
        merged = pad_mask.clone()
        merged[: existing_mask.shape[0]] |= existing_mask
        return merged

    @staticmethod
    def _crop(x: torch.Tensor, meta: int):
        assert x.ndim == 3
        dim = sum([m["shape"] for m in meta])
        x = x[:, :, :dim]
        return x

    @staticmethod
    def _concat(x: Dict[str, torch.Tensor], meta: Dict[str, Dict]):
        x = torch.cat([x[m["key"]] for m in meta], dim=-1)
        assert x.ndim == 2
        return x

    @staticmethod
    def _split(x: torch.Tensor, meta: Dict[str, Dict]):
        assert x.ndim == 3
        y = {}
        idx = 0
        for m in meta:
            key, dim = (m["key"], m["shape"])
            y[key] = x[:, :, idx : idx + dim]
            idx += dim
        return y
