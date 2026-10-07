"""Export an EMA policy using the inference checkpoint schema."""

import json
import os
from pathlib import Path

import torch


def ema_policy_state(model, ema):
    canonical = {id(p): name for name, p in model.named_parameters()}
    expected = {name for name, p in model.named_parameters() if p.requires_grad}
    if set(ema) != expected:
        raise ValueError(
            f"EMA coverage mismatch: missing={sorted(expected - set(ema))}, extra={sorted(set(ema) - expected)}"
        )
    result = {}
    for group, module in (("mot", model.mot), ("proprio_encoder", model.proprio_encoder)):
        parameters = dict(module.named_parameters(remove_duplicate=False))
        state = {}
        original = module.state_dict()
        for key, value in original.items():
            if key in parameters and parameters[key].requires_grad:
                value = ema[canonical[id(parameters[key])]]
            if value.shape != original[key].shape:
                raise ValueError(f"EMA shape mismatch: {group}.{key}")
            state[key] = value.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        result[group] = state
    return result


def export_policy(model, ema_path, stats_path, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    ema = torch.load(ema_path, map_location="cpu", mmap=True, weights_only=True)
    state = ema_policy_state(model, ema)
    temporary = destination / "model.pth.partial"
    torch.save(state, temporary)
    os.replace(temporary, destination / "model.pth")
    stats = json.loads(Path(stats_path).read_text())
    stats = {
        section: {
            "default": {key: stats[section]["default"][key] for key in ("global_min", "global_max")}
        }
        for section in ("state", "action")
    }
    (destination / "dataset_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    config = json.loads(Path(__file__).with_name("eval_config.json").read_text())
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print("POLICY_EXPORTED", destination, flush=True)
