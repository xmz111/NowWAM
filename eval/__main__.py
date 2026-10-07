"""Evaluate a released model or a LIBERO training export."""

import argparse
import json
import os
import sys
from pathlib import Path

MODELS = {
    "Klein-LIBERO": ("libero", "assets/flux2/ae.safetensors"),
    "Klein-RoboCasa-100shot": ("robocasa", "assets/flux2/ae.safetensors"),
}


def evaluation_command(model, *, gpus="0", lanes=1, limit=None, output=None, checkpoint=None):
    benchmark, vae = MODELS[model]
    if lanes < 1 or (limit is not None and limit < 1):
        raise ValueError("Lane count and episode limit must be positive")
    devices = gpus.split(",")
    if len(set(devices)) != len(devices) or any(not d.isdigit() for d in devices):
        raise ValueError("GPU indices must be distinct numbers, e.g. 0,1")
    suffix = f"-smoke-{limit}" if limit is not None else ""
    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        output = output or checkpoint.parent.parent / f"eval-{checkpoint.name}{suffix}"
    command = [
        sys.executable,
        "-m",
        "eval.run",
        "run",
        "--benchmark",
        benchmark,
        "--checkpoint",
        str(checkpoint) if checkpoint is not None else f"checkpoints/NowWAM/{model}",
        "--vae",
        vae,
        "--text-encoder",
        "assets/Qwen3-4B",
        "--sim-python",
        f".venv-{benchmark}/bin/python",
        "--gpus",
        gpus,
        "--lanes-per-gpu",
        str(lanes),
        "--output",
        str(output or f"outputs/{model}{suffix}"),
    ]
    if benchmark == "libero":
        command += [
            "--sim-source",
            "assets/LIBERO-plus",
            "--sim-library",
            "assets/osmesa/lib",
            "--libero-config",
            "assets/libero-config",
        ]
    else:
        command += [
            "--sim-source",
            "assets/DIAL",
            "--sim-source",
            "assets/robocasa-gr1-tabletop-tasks",
        ]
    if limit is not None:
        command += ["--limit", str(limit)]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Released model name or LIBERO training export directory")
    parser.add_argument("--gpus", default="0", help="Visible GPU indices, e.g. 0,1")
    parser.add_argument("--lanes", type=int, default=1, help="Parallel lanes per GPU")
    parser.add_argument("--limit", type=int, help="Run a smoke-test prefix instead of the full set")
    parser.add_argument("--output", type=Path, help="Override the output directory")
    args = parser.parse_args()
    try:
        if args.model not in MODELS:
            args.checkpoint = Path(args.model)
            config = json.loads((args.checkpoint / "config.json").read_text())
            if (config.get("backbone"), config.get("action_dim")) != ("klein", 7):
                raise ValueError("Training exports must be Klein LIBERO checkpoints")
            args.model = "Klein-LIBERO"
        command = evaluation_command(**vars(args))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    required = [
        Path(command[command.index(flag) + 1])
        for flag in ("--checkpoint", "--vae", "--text-encoder", "--sim-python")
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        parser.error("Missing prepared files: " + ", ".join(missing) + ". See README.md.")
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
