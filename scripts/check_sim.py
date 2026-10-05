"""Check benchmark resets and camera observations without loading policy weights."""

import argparse
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class ResetComplete(Exception):
    pass


def check_case(benchmark, index):
    import numpy as np

    from eval.run import cases_for

    case = cases_for(benchmark)[index]
    expected = (224, 448 if benchmark == "libero" else 224, 3)
    state_dim = 8 if benchmark == "libero" else 76

    def observe(image, state, instruction, step):
        if image.shape != expected or image.dtype != np.uint8:
            raise ValueError(f"Invalid camera output: {image.shape}, {image.dtype}")
        if np.asarray(state).shape != (state_dim,) or not np.isfinite(state).all():
            raise ValueError("Invalid proprioception")
        if not instruction or step < 0:
            raise ValueError("Invalid instruction/control step")
        raise ResetComplete

    if benchmark == "libero":
        from eval.libero import rollout
    else:
        from eval.robocasa import rollout
    try:
        rollout(case, observe)
    except ResetComplete:
        print(f"RESET_OK {case['case_id']}", flush=True)
        return
    raise RuntimeError("Reset did not produce a policy observation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=("libero", "robocasa"))
    parser.add_argument("--case", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.case is not None:
        return check_case(args.benchmark, args.case)
    from eval.run import sim_environment

    assets = ROOT / "assets"
    sources = (
        [assets / "LIBERO-plus"]
        if args.benchmark == "libero"
        else [assets / "DIAL", assets / "robocasa-gr1-tabletop-tasks"]
    )
    env = sim_environment(
        SimpleNamespace(
            benchmark=args.benchmark,
            sim_source=list(map(str, sources)),
            gpu="0",
            sim_library=[str(assets / "osmesa/lib")] if args.benchmark == "libero" else [],
            libero_config=str(assets / "libero-config"),
        )
    )
    if args.benchmark == "libero":
        env["LIBGL_DRIVERS_PATH"] = str(assets / "osmesa/lib/dri")
    # One case per suite for LIBERO, one seed per task for RoboCasa.
    indices = [0, 2519, 5110, 7512] if args.benchmark == "libero" else range(0, 1200, 50)
    for index in indices:
        subprocess.run(
            [
                str(ROOT / f".venv-{args.benchmark}/bin/python"),
                str(Path(__file__).resolve()),
                args.benchmark,
                "--case",
                str(index),
            ],
            env=env,
            cwd=ROOT,
            check=True,
            timeout=300,
        )
    print("Simulator reset checks passed; this is not a policy success-rate test.")


if __name__ == "__main__":
    main()
