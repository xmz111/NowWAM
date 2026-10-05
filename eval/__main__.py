"""Evaluate a released checkpoint using the directory layout in SETUP.md."""

import argparse
import os
import sys
from pathlib import Path

MODELS = {
    "Klein-LIBERO": ("libero", "assets/flux2/ae.safetensors"),
    "ZImage-LIBERO": ("libero", "assets/Z-Image"),
    "Klein-RoboCasa-100shot": ("robocasa", "assets/flux2/ae.safetensors"),
}


def evaluation_command(model, *, gpus="0", lanes=1, limit=None, output=None):
    benchmark, vae = MODELS[model]
    if lanes < 1 or (limit is not None and limit < 1):
        raise ValueError("Lane count and episode limit must be positive")
    devices = gpus.split(",")
    if len(set(devices)) != len(devices) or any(not d.isdigit() for d in devices):
        raise ValueError("GPU indices must be distinct numbers, e.g. 0,1")
    suffix = f"-smoke-{limit}" if limit is not None else ""
    command = [
        sys.executable,
        "-m",
        "eval.run",
        "run",
        "--benchmark",
        benchmark,
        "--checkpoint",
        f"checkpoints/NowWAM/{model}",
        "--vae",
        vae,
        "--text-encoder",
        "assets/Qwen3-4B",
        "--sim-python",
        f".venv-{benchmark}/bin/python",
        "--assets-manifest",
        f"assets/{model}-manifest.json",
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
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--gpus", default="0", help="Visible GPU indices, e.g. 0,1")
    parser.add_argument("--lanes", type=int, default=1, help="Parallel lanes per GPU")
    parser.add_argument("--limit", type=int, help="Run a smoke-test prefix instead of the full set")
    parser.add_argument("--output", type=Path, help="Override the output directory")
    args = parser.parse_args()
    try:
        command = evaluation_command(**vars(args))
    except ValueError as error:
        parser.error(str(error))
    required = [
        Path(command[command.index(flag) + 1])
        for flag in ("--checkpoint", "--vae", "--text-encoder", "--sim-python", "--assets-manifest")
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        parser.error("Missing prepared files: " + ", ".join(missing) + ". See SETUP.md.")
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
