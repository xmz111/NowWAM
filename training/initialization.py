"""Task-specific initialization and linear warmup."""

import numpy as np
import torch


def task_parameters(seed=1234):
    rng = np.random.default_rng(seed)

    def uniform(shape, fan_in):
        bound = 1.0 / np.sqrt(fan_in)
        return rng.uniform(-bound, bound, size=shape).astype(np.float32)

    arrays = {
        "mot.mixtures.action.action_encoder.weight": uniform((7, 1024), 7).T,
        "mot.mixtures.action.action_encoder.bias": uniform((1024,), 7),
        "mot.mixtures.action.head.linear.weight": uniform((1024, 7), 1024).T,
        "mot.mixtures.action.head.adaLN_modulation.1.weight": uniform((1024, 2048), 1024).T,
        "proprio_encoder.weight": uniform((8, 7680), 8).T,
        "proprio_encoder.bias": uniform((7680,), 8),
    }
    return {name: torch.from_numpy(np.ascontiguousarray(value)) for name, value in arrays.items()}


def warmup_factor(update, warmup_steps=2110):
    return min(float(update) / float(warmup_steps), 1.0)
