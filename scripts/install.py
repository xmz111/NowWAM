"""Install selected NowWAM policies and their isolated simulator environments."""

import argparse
import ctypes
import os
import platform
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

from assets import (
    apply_source_patch,
    download,
    extract_zip,
    manifest,
    patch_dial,
    patch_robocasa,
    prepare_libero,
    sha256,
    zip_subtree,
)

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
MODELS = ("Klein-LIBERO", "ZImage-LIBERO", "Klein-RoboCasa-100shot")
HF_REVISION = "679cdcdd10346130dc1f22bff3455822d3e8d61d"
QWEN_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
ZIMAGE_REVISION = "04cc4abb7c5069926f75c9bfde9ef43d49423021"
LIBERO_REVISION = "4976dc30028e805ff8094b55501d532c48fec182"
LIBERO_ASSET_REVISION = "dd2bd61b7d9a6fef1abc52d606e983b41886a149"
SOURCES = {
    "flux2": ("black-forest-labs/flux2", "50fe5162777813d869182b139e83b10743caef15"),
    "LIBERO-plus": ("sylvestf/LIBERO-plus", LIBERO_REVISION),
    "DIAL": ("xpeng-robotics/DIAL", "b611fed034c62441413fea2b136ff7c841f382fe"),
    "robocasa-gr1-tabletop-tasks": (
        "robocasa/robocasa-gr1-tabletop-tasks",
        "4840e671596f93ca03651524b9f72ffb1aadfeff",
    ),
}


def run(*args, env=None):
    print("+", " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=ROOT, env=env, check=True)


def checkout(name):
    repo, revision = SOURCES[name]
    dest = ASSETS / "sources" / name if name == "flux2" else ASSETS / name
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", "--no-checkout", f"https://github.com/{repo}.git", dest)
        run("git", "-C", dest, "checkout", "--detach", revision)
    actual = subprocess.check_output(
        ["git", "-C", str(dest), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != revision:
        raise ValueError(f"Unexpected source revision in {dest}: {actual}")
    return dest


def environment(name):
    dest = ROOT / name
    python = dest / "bin/python"
    if not dest.exists():
        run(sys.executable, "-m", "venv", dest)
    if not python.is_file():
        raise ValueError(f"Not a Python environment: {dest}")
    run(python, "-c", "import sys; assert sys.version_info[:2] == (3, 11)")
    run(python, "-m", "pip", "install", "pip==26.0.1", "setuptools==80.9.0", "wheel==0.45.1")
    return python


def install_dependencies(models):
    python = environment(".venv")
    run(
        python,
        "-m",
        "pip",
        "install",
        "torch==2.7.1",
        "torchvision==0.22.1",
        "--index-url",
        "https://download.pytorch.org/whl/cu118",
    )
    flux = checkout("flux2")
    apply_source_patch(flux, "flux2-dependencies.patch")
    run(
        python,
        "-m",
        "pip",
        "install",
        "-r",
        ROOT / "requirements/model.txt",
        "-e",
        ROOT,
        flux,
        "--extra-index-url",
        "https://download.pytorch.org/whl/cu118",
    )
    run(python, "-m", "pip", "check")
    for benchmark in sorted({"robocasa" if "RoboCasa" in m else "libero" for m in models}):
        sim = environment(f".venv-{benchmark}")
        torch_index = (
            "https://download.pytorch.org/whl/cu118"
            if benchmark == "libero"
            else "https://pypi.org/simple"
        )
        run(
            sim,
            "-m",
            "pip",
            "install",
            "torch==2.7.1",
            "--index-url",
            torch_index,
        )
        requirements = ROOT / "requirements" / f"{benchmark}.txt"
        if benchmark == "libero":
            source = checkout("LIBERO-plus")
            run(
                sim,
                "-m",
                "pip",
                "install",
                "-r",
                requirements,
                "--extra-index-url",
                torch_index,
            )
            # Import the pinned simulator source via the evaluator's PYTHONPATH.
            if not (source / "libero/libero/benchmark").is_dir():
                raise ValueError("Missing LIBERO benchmark source")
        else:
            checkout("DIAL")
            source = checkout("robocasa-gr1-tabletop-tasks")
            run(
                sim,
                "-m",
                "pip",
                "install",
                "-r",
                requirements,
                "-e",
                source,
            )
        run(sim, "-m", "pip", "check")
    return python


def hf(python, repo, destination, revision, *selection):
    run(
        python.parent / "hf",
        "download",
        repo,
        "--revision",
        revision,
        "--local-dir",
        destination,
        *selection,
    )


def osmesa():
    archive = download(
        "https://api.anaconda.org/download/conda-forge/micromamba/2.3.3/"
        "linux-64/micromamba-2.3.3-0.tar.bz2",
        ASSETS / "downloads/micromamba.tar.bz2",
        "e7274528ceb9c20d048a428d6c22d7e02e268f8ffb762c4c365422347c8b8ba2",
    )
    binary = ASSETS / "bin/micromamba"
    if not binary.exists():
        binary.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as tar:
            item = tar.getmember("bin/micromamba")
            if not item.isfile():
                raise ValueError("Invalid micromamba archive")
            with tar.extractfile(item) as src, binary.open("xb") as dest:
                shutil.copyfileobj(src, dest)
        binary.chmod(0o755)
    env = dict(os.environ, MAMBA_ROOT_PREFIX=str(ASSETS / "mamba"))
    if not (ASSETS / "osmesa/conda-meta").is_dir():
        run(
            binary,
            "--no-rc",
            "create",
            "-y",
            "-p",
            ASSETS / "osmesa",
            "--file",
            ROOT / "requirements/osmesa-linux-64.txt",
            env=env,
        )
    library = ASSETS / "osmesa/lib/libOSMesa.so.8"
    if sha256(library) != "39b96746959da7f83ced4fe77f0d1eb93730471b9e74674fd9eb52e3c153b8a4":
        raise ValueError("Unexpected OSMesa library")


def prepare_simulator(python, benchmark):
    if benchmark == "libero":
        root = ASSETS / "LIBERO-plus"
        folder = ASSETS / "downloads/libero"
        hf(
            python,
            "Sylvest/LIBERO-plus",
            folder,
            LIBERO_ASSET_REVISION,
            "--repo-type",
            "dataset",
            "--include",
            "assets.zip",
        )
        archive = folder / "assets.zip"
        if sha256(archive) != "96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf":
            raise ValueError("Unexpected LIBERO asset archive")
        extract_zip(archive, root / "libero/libero/assets", zip_subtree(archive, "assets"))
        prepare_libero(root, ASSETS / "libero-config")
        osmesa()
        return [root, ASSETS / "libero-config", ASSETS / "osmesa/lib"]
    root = ASSETS / "robocasa-gr1-tabletop-tasks"
    sim = ROOT / ".venv-robocasa/bin/python"
    patch_dial(ASSETS / "DIAL")
    marker = ASSETS / "robocasa-download.complete"
    if not marker.exists():
        # The upstream all-assets command would overwrite checked-in fixture XML.
        # Download its other registries normally; fill missing fixtures separately.
        run(
            sim,
            "-c",
            "import sys; "
            f"sys.path.insert(0, {str(root / 'robocasa/scripts')!r}); "
            "import download_kitchen_assets as k; "
            "k.DOWNLOAD_ASSET_REGISTRY.pop('fixtures'); "
            "from download_tabletop_assets import download_dc_assets; download_dc_assets(True)",
        )
    archive = download(
        "https://utexas.box.com/shared/static/pobhbsjyacahg2mx8x4rm5fkz3wlmyzp.zip",
        ASSETS / "downloads/fixtures.zip",
        "27905b494e7bf12826a8d67c167c73d95eade467841f5d2ddbdd87efc15b4c5a",
    )
    # Keep the task repository's fixture XML; only fill absent public meshes/textures.
    extract_zip(archive, root / "robocasa/models/assets/fixtures", "fixtures/", missing_only=True)
    patch_robocasa(root)
    return [ASSETS / "DIAL", root]


def prepare_models(python, models):
    run(python, "-c", "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'")
    if any("LIBERO" in model for model in models):
        run(ROOT / ".venv-libero/bin/python", "-c", "import cv2; from wand.api import library")
    hf(
        python,
        "Qwen/Qwen3-4B",
        ASSETS / "Qwen3-4B",
        QWEN_REVISION,
        "--include",
        "*.json",
        "*.safetensors",
        "merges.txt",
        "vocab.json",
        "tokenizer.model",
    )
    roots = {}
    for model in models:
        hf(
            python,
            "xmz111/NowWAM",
            ROOT / "checkpoints/NowWAM",
            HF_REVISION,
            "--include",
            f"{model}/*",
        )
        if model == "ZImage-LIBERO":
            vae = ASSETS / "Z-Image"
            hf(
                python,
                "Tongyi-MAI/Z-Image",
                vae,
                ZIMAGE_REVISION,
                "--include",
                "vae/*",
                "transformer/config.json",
            )
        else:
            hf(
                python,
                "black-forest-labs/FLUX.2-dev",
                ASSETS / "flux2",
                "26afe3a78bb242c0a8bb181dcc8937bb16e5c66c",
                "--include",
                "ae.safetensors",
            )
            vae = ASSETS / "flux2/ae.safetensors"
            if sha256(vae) != "868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3":
                raise ValueError("Unexpected Klein VAE")
        benchmark = "robocasa" if "RoboCasa" in model else "libero"
        if benchmark not in roots:
            roots[benchmark] = prepare_simulator(python, benchmark)
            run(python, ROOT / "scripts/check_sim.py", benchmark)
            if benchmark == "robocasa":
                (ASSETS / "robocasa-download.complete").touch()
        manifest(
            [vae, ASSETS / "Qwen3-4B", ASSETS / "sources/flux2", *roots[benchmark]],
            ASSETS / f"{model}-manifest.json",
        )
        run(python, "-m", "eval", model, "--limit", "1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=(*MODELS, "all"))
    parser.add_argument(
        "--dependencies-only",
        action="store_true",
        help="Install packages only; do not download weights/assets or run a rollout",
    )
    args = parser.parse_args()
    if (
        sys.version_info[:2] != (3, 11)
        or platform.system() != "Linux"
        or platform.machine() != "x86_64"
    ):
        parser.error("Use Python 3.11 on x86-64 Linux")
    if not shutil.which("git"):
        parser.error("Install git first; see SETUP.md")
    if not args.dependencies_only:
        for name in ("libGL.so.1", "libEGL.so.1", "libglib-2.0.so.0"):
            try:
                ctypes.CDLL(name)
            except OSError:
                parser.error(f"Missing system library {name}; see SETUP.md")
        run("nvidia-smi")
    models = MODELS if args.model == "all" else (args.model,)
    python = install_dependencies(models)
    if args.dependencies_only:
        print("Dependencies installed. Weights, assets and CUDA rollouts have NOT been checked.")
    else:
        prepare_models(python, models)
        print("Installation and smoke evaluation completed. Activate .venv to run full evaluation.")


if __name__ == "__main__":
    main()
