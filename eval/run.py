"""Parallel, resumable evaluation with one fresh simulator process per case."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager
from multiprocessing.connection import Client, Listener
from pathlib import Path

from nowwam.protocol import (
    lane_indices,
    libero_action_seed,
    libero_cases,
    robocasa_action_seed,
    validate_results,
)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


@contextmanager
def exclusive_run(directory):
    """Prevent two launchers from writing the same output directory."""
    import fcntl

    with (directory / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another evaluator owns this output directory") from None
        yield


def cases_for(benchmark):
    if benchmark == "libero":
        return [
            dict(item, case_id=f"{item['suite']}/{item['task_idx']}", canonical_index=i)
            for i, item in enumerate(libero_cases())
        ]
    from .robocasa import TASKS

    return [
        {"task": task, "seed": seed, "case_id": f"{task}/{seed:02d}"}
        for task in TASKS
        for seed in range(50)
    ]


def completed(path, case, contract):
    if not path.exists():
        return False
    receipt = json.loads(path.read_text())
    if receipt.get("case_id") != case["case_id"] or receipt.get("contract_sha256") != contract:
        raise ValueError(f"Conflicting receipt: {path}")
    if receipt.get("status") == "error":
        return False
    validate_results([case["case_id"]], [receipt], contract)
    return True


def sim_environment(args):
    env = os.environ.copy()
    for key in (
        "PYTHONPATH",
        "MUJOCO_GL",
        "PYOPENGL_PLATFORM",
        "LD_PRELOAD",
        "MUJOCO_EGL_DEVICE_ID",
        "LIBGL_DRIVERS_PATH",
    ):
        env.pop(key, None)
    env.update(
        PYTHONPATH=os.pathsep.join([str(Path(__file__).resolve().parents[1]), *args.sim_source]),
        PYTHONHASHSEED="0",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        LP_NUM_THREADS="2",
        GALLIUM_DRIVER="llvmpipe",
    )
    renderer = "osmesa" if args.benchmark == "libero" else "egl"
    env.update(MUJOCO_GL=renderer, PYOPENGL_PLATFORM=renderer)
    if renderer == "egl":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        ordinal = int(args.gpu)
        env["MUJOCO_EGL_DEVICE_ID"] = visible[ordinal] if visible != [""] else str(ordinal)
        if not env["MUJOCO_EGL_DEVICE_ID"].isdigit():
            raise ValueError(
                "EGL GPU UUID mapping is not validated; use numeric CUDA_VISIBLE_DEVICES"
            )
    # Do not inherit an unrelated OSMesa runtime into the EGL worker.
    env["LD_LIBRARY_PATH"] = os.pathsep.join(args.sim_library)
    if args.benchmark == "libero":
        if args.sim_library:
            drivers = Path(args.sim_library[0]) / "dri"
            if drivers.is_dir():
                env["LIBGL_DRIVERS_PATH"] = str(drivers.resolve())
        if not args.libero_config:
            raise ValueError(
                "--libero-config must point to an isolated LIBERO-Plus config directory"
            )
        env["LIBERO_CONFIG_PATH"] = str(Path(args.libero_config).resolve())
    return env


def worker(socket, benchmark, case_json):
    case = json.loads(case_json)
    with Client(socket, family="AF_UNIX") as conn:

        def predict(image, state, instruction, step):
            conn.send(("observation", image, state, instruction, step))
            if not conn.poll(600):
                raise TimeoutError("Policy response timeout")
            kind, action = conn.recv()
            if kind != "action":
                raise RuntimeError("Expected policy action")
            return action

        try:
            if benchmark == "libero":
                from .libero import rollout
            else:
                from .robocasa import rollout
            conn.send(("result", rollout(case, predict)))
        except Exception:
            conn.send(("error", traceback.format_exc()))
            raise


def run_case(args, policy, case, environment, log):
    with tempfile.TemporaryDirectory(prefix="nowwam-") as temp:
        socket = str(Path(temp) / "sim.sock")
        with Listener(socket, family="AF_UNIX") as listener:
            listener._listener._socket.settimeout(args.timeout)
            process = subprocess.Popen(
                [
                    args.sim_python,
                    "-m",
                    "eval.run",
                    "worker",
                    socket,
                    args.benchmark,
                    json.dumps(case),
                ],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )
            try:
                with listener.accept() as conn:
                    while True:
                        if not conn.poll(args.timeout):
                            raise TimeoutError("Simulator response timeout; inspect case log")
                        message = conn.recv()
                        if message[0] == "result":
                            process.wait(timeout=30)
                            if process.returncode:
                                raise RuntimeError(
                                    f"Simulator exited {process.returncode} after result"
                                )
                            return message[1]
                        if message[0] == "error":
                            raise RuntimeError(message[1])
                        if message[0] != "observation":
                            raise RuntimeError("Unexpected simulator message")
                        _, image, state, instruction, step = message
                        if args.benchmark == "libero":
                            profile = (
                                "zimage_jax"
                                if policy.backbone == "zimage"
                                else "klein_torch_reference"
                            )
                            seed = libero_action_seed(profile, case["canonical_index"], int(step))
                        else:
                            seed = robocasa_action_seed(case["task"], case["seed"], int(step))
                        conn.send(("action", policy.predict(image, state, instruction, seed)))
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()


def lane(args):
    import torch

    from nowwam.policy import Policy

    torch.set_num_threads(2)
    torch.manual_seed(42)
    contract = json.loads((args.output / "contract.json").read_text())
    contract_hash = digest(contract)
    records = contract["cases"]
    assigned = list(lane_indices(len(records), args.total_lanes, args.lane))
    pending = [
        i
        for i in assigned
        if not completed(args.output / "receipts" / f"{i:05d}.json", records[i], contract_hash)
    ]
    if not pending:
        return
    policy = Policy(
        args.checkpoint,
        vae=args.vae,
        text_encoder=args.text_encoder,
        device=f"cuda:{args.gpu}",
        text_device=f"cuda:{args.gpu}" if args.text_device == "cuda" else "cpu",
    )
    if policy.is_libero != (args.benchmark == "libero"):
        raise ValueError("Checkpoint does not match the requested benchmark")
    environment = sim_environment(args)
    for index in pending:
        case = records[index]
        receipt = {"case_id": case["case_id"], "contract_sha256": contract_hash}
        started = time.monotonic()
        with (args.output / "logs" / f"{index:05d}.log").open("a") as log:
            try:
                result = run_case(args, policy, case, environment, log)
                receipt.update(result, status="completed", error=None)
            except Exception as exc:
                receipt.update(status="error", error=f"{type(exc).__name__}: {exc}")
            receipt["elapsed_seconds"] = time.monotonic() - started
        atomic_json(args.output / "receipts" / f"{index:05d}.json", receipt)
        print(
            json.dumps(
                {
                    "case": case["case_id"],
                    "status": receipt["status"],
                    "success": receipt.get("success"),
                }
            ),
            flush=True,
        )
        if receipt["status"] == "error":
            raise RuntimeError(
                "Lane stopped after an infrastructure/model error; see receipt and log"
            )


def summarize(output):
    contract = json.loads((output / "contract.json").read_text())
    receipts = [
        json.loads(path.read_text()) for path in sorted((output / "receipts").glob("*.json"))
    ]
    result = validate_results([c["case_id"] for c in contract["cases"]], receipts, digest(contract))
    result["coverage"] = (
        "full" if len(contract["cases"]) == contract["full_case_count"] else "subset"
    )
    result["contract_sha256"] = digest(contract)
    atomic_json(output / "summary.json", result)
    print(json.dumps(result, indent=2))


def source_hash():
    root = Path(__file__).resolve().parents[1]
    return digest(
        {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ("nowwam", "eval")
            for p in sorted((root / folder).glob("*.py"))
        }
    )


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        return worker(*sys.argv[2:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "lane", "summary"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--benchmark", choices=["libero", "robocasa"])
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--vae")
    parser.add_argument("--text-encoder")
    parser.add_argument("--text-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--sim-python")
    parser.add_argument("--sim-source", action="append", default=[])
    parser.add_argument("--sim-library", action="append", default=[])
    parser.add_argument("--libero-config")
    parser.add_argument(
        "--assets-manifest", type=Path, help="Reviewed JSON mapping absolute asset paths to SHA256"
    )
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--lanes-per-gpu", type=int, default=1)
    parser.add_argument(
        "--limit", type=int, help="Smoke-test prefix only; never labelled full coverage"
    )
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--gpu", default="0", help=argparse.SUPPRESS)
    parser.add_argument("--lane", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--total-lanes", type=int, default=1, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.command == "summary":
        return summarize(args.output)
    if args.command == "lane":
        return lane(args)
    for key in ("benchmark", "checkpoint", "vae", "text_encoder", "sim_python", "assets_manifest"):
        if getattr(args, key) is None:
            parser.error(f"--{key.replace('_', '-')} is required")
    if args.lanes_per_gpu < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Lane count and limit must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    with exclusive_run(args.output):
        from nowwam.policy import file_sha256

        asset_files = json.loads(args.assets_manifest.read_text())
        if not asset_files:
            raise ValueError("Asset manifest is empty")
        for path, expected in asset_files.items():
            if file_sha256(path) != expected:
                raise ValueError(f"Changed asset: {path}")
        # Sources and encoders must be covered, not just an arbitrary asset file.
        for resource in [args.vae, args.text_encoder, *args.sim_source]:
            resource = Path(resource).resolve()
            if resource.is_file():
                covered = str(resource) in asset_files
            else:
                required = [
                    p.resolve()
                    for p in resource.rglob("*")
                    if p.is_file()
                    and p.suffix in (".py", ".json", ".safetensors")
                    and not {".git", ".cache", "__pycache__"}.intersection(p.parts)
                ]
                covered = bool(required) and all(str(p) in asset_files for p in required)
            if not covered:
                raise ValueError(f"Resource is missing from the asset manifest: {resource}")
        environment = sim_environment(args)
        probe = "import importlib.metadata as m,json; print(json.dumps({n:m.version(n) for n in ['numpy','mujoco','robosuite','Pillow']}))"
        versions = json.loads(
            subprocess.check_output([args.sim_python, "-c", probe], env=environment, text=True)
        )
        expected_versions = {
            "numpy": "1.26.4",
            "mujoco": "3.3.2" if args.benchmark == "libero" else "3.2.6",
            "robosuite": "1.4.0" if args.benchmark == "libero" else "1.5.1",
            "Pillow": "12.0.0",
        }
        if versions != expected_versions:
            raise ValueError(f"Simulator versions differ: {versions}; expected {expected_versions}")
        all_cases = cases_for(args.benchmark)
        contract = {
            "benchmark": args.benchmark,
            "cases": all_cases[: args.limit],
            "full_case_count": len(all_cases),
            "source_sha256": source_hash(),
            "checkpoint_sha256": file_sha256(args.checkpoint / "model.pth"),
            "config": json.loads((args.checkpoint / "config.json").read_text()),
            "stats_sha256": file_sha256(args.checkpoint / "dataset_stats.json"),
            "asset_manifest_sha256": digest(asset_files),
            "sim_versions": versions,
            "text_device": args.text_device,
            "model_versions": {
                n: importlib.metadata.version(n)
                for n in ("torch", "transformers", "diffusers", "numpy", "Pillow")
            },
        }
        path = args.output / "contract.json"
        if path.exists() and json.loads(path.read_text()) != contract:
            raise ValueError("Output directory belongs to a different evaluation contract")
        atomic_json(path, contract)
        for directory in ("receipts", "logs"):
            (args.output / directory).mkdir(exist_ok=True)
        gpus = args.gpus.split(",")
        if len(set(gpus)) != len(gpus) or any(not gpu.isdigit() for gpu in gpus):
            parser.error("--gpus expects distinct visible GPU indices, e.g. 0,1")
        total = len(gpus) * args.lanes_per_gpu
        processes = []
        try:
            for index in range(total):
                command = [
                    sys.executable,
                    "-m",
                    "eval.run",
                    "lane",
                    *sys.argv[2:],
                    "--gpu",
                    gpus[index // args.lanes_per_gpu],
                    "--lane",
                    str(index),
                    "--total-lanes",
                    str(total),
                ]
                processes.append(subprocess.Popen(command, start_new_session=True))
            while True:
                statuses = [process.poll() for process in processes]
                if any(status not in (None, 0) for status in statuses):
                    raise RuntimeError(
                        "A lane failed; stopping this run. Completed receipts are retained"
                    )
                if all(status == 0 for status in statuses):
                    break
                time.sleep(0.2)
        finally:
            for process in processes:
                try:
                    # Simulator children inherit the lane's process group.
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    continue
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        summarize(args.output)


if __name__ == "__main__":
    main()
