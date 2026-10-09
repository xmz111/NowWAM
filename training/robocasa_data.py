"""RoboCasa episode preparation, instruction encoding and training samples."""

import hashlib
import json
import importlib.util
import os
from functools import lru_cache
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
STATE_INDICES = np.asarray([0, 1, 2, 9, 10, 11], dtype=np.int32)


def normalize(proprio, actions, stats):
    proprio = np.asarray(proprio, dtype=np.float32).copy()
    actions = np.asarray(actions, dtype=np.float32).copy()
    mean = np.asarray(stats["proprio"]["mean"], dtype=np.float32)
    std = np.asarray(stats["proprio"]["std"], dtype=np.float32)
    proprio[:, STATE_INDICES] = (proprio[:, STATE_INDICES] - mean[STATE_INDICES]) / (
        std[STATE_INDICES] + 1e-8
    )
    mean = np.asarray(stats["actions"]["mean"], dtype=np.float32)
    std = np.asarray(stats["actions"]["std"], dtype=np.float32)
    actions = (actions - mean) / (std + 1e-8)
    return np.clip(proprio, -5, 5), np.clip(actions, -5, 5)


def augment_image(frame, rng):
    height, width = int(168 * .95), int(256 * .95)
    top = int(rng.integers(0, 168 - height + 1))
    left = int(rng.integers(0, 256 - width + 1))
    image = Image.fromarray(frame[top:top + height, left:left + width]).resize(
        (224, 224), Image.Resampling.BILINEAR
    )
    operations = [
        ("brightness", float(rng.uniform(.7, 1.3))),
        ("contrast", float(rng.uniform(.6, 1.4))),
        ("saturation", float(rng.uniform(.5, 1.5))),
        ("hue", float(rng.uniform(-.08, .08))),
    ]
    rng.shuffle(operations)
    for name, value in operations:
        if name == "brightness":
            image = ImageEnhance.Brightness(image).enhance(value)
        elif name == "contrast":
            image = ImageEnhance.Contrast(image).enhance(value)
        elif name == "saturation":
            image = ImageEnhance.Color(image).enhance(value)
        else:
            hsv = np.asarray(image.convert("HSV")).copy()
            hsv[..., 0] = (hsv[..., 0].astype(np.int16) + int(value * 255)) % 256
            image = Image.fromarray(hsv.astype(np.uint8), "HSV").convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def action_chunk(actions, frame):
    length = len(actions)
    indices = np.minimum(np.arange(frame, frame + 16), length - 1)
    chunk = actions[indices].astype(np.float32).copy()
    padding = np.arange(16) >= min(16, length - frame)
    # Match the original loader, including masked tail values.
    chunk[padding, :6] = 0
    return chunk, padding


class RoboCasaDataset(Dataset):
    def __init__(self, data, text_cache):
        self.root = Path(data)
        self.text_cache = Path(text_cache)
        self.stats = json.loads((self.root / "global_stats.json").read_text())
        self.episodes, lengths = [], []
        for path in sorted(self.root.glob("*/manifest.json")):
            manifest = json.loads(path.read_text())
            for record in sorted(manifest["records"], key=lambda row: row["file"]):
                if record["frames"] > 1:
                    self.episodes.append(path.parent / record["file"])
                    lengths.append(record["frames"] - 1)
        if not self.episodes:
            raise ValueError("No episode manifests found")
        self.ends = np.cumsum(lengths)
        self._rng = None

    def __len__(self):
        return int(self.ends[-1])

    @lru_cache(maxsize=32)
    def episode(self, index):
        mapped = self.episodes[index].with_suffix(".frames.npy")
        with np.load(self.episodes[index], allow_pickle=False) as source:
            images = np.load(mapped, mmap_mode="r") if mapped.exists() else source["imgs_offset"]
            proprio, actions = normalize(source["proprio"], source["actions"], self.stats)
            instruction = str(source["lang"]).strip()
        if images.shape[1:] != (168, 256, 3):
            raise ValueError("Expected original offset-cropped training images")
        if proprio.shape != (len(images), 76) or actions.shape != (len(images), 47):
            raise ValueError("Episode action/state shape mismatch")
        return images, proprio, actions, instruction

    @lru_cache(maxsize=186)
    def text(self, instruction):
        key = hashlib.sha256(PROMPT.format(task=instruction).encode()).hexdigest()
        with np.load(self.text_cache / f"{key}.npz", allow_pickle=False) as source:
            hidden = source["hidden_states"].astype(np.float16)
            mask = source["attention_mask"].astype(bool)
        if hidden.shape != (128, 7680) or mask.shape != (128,):
            raise ValueError("Text cache shape mismatch")
        return torch.from_numpy(hidden), torch.from_numpy(mask)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self._rng is None:
            self._rng = np.random.default_rng(torch.initial_seed())
        episode = int(np.searchsorted(self.ends, index, side="right"))
        start = int(self.ends[episode - 1]) if episode else 0
        frame = index - start
        images, proprio, actions, instruction = self.episode(episode)
        image = augment_image(images[frame], self._rng)
        chunk, padding = action_chunk(actions, frame)
        hidden, mask = self.text(instruction)
        return {
            "video": torch.from_numpy(image.copy()).permute(2, 0, 1).float().div(127.5).sub(1).unsqueeze(1),
            "proprio": torch.from_numpy(proprio[frame].copy()),
            "action": torch.from_numpy(chunk),
            "action_is_pad": torch.from_numpy(padding),
            "text_hidden_states": hidden,
            "text_attention_mask": mask,
        }


def prepare_task(raw, dial, output, task, count=100):
    """Replay saved states with DIAL, augment next-frame actions, and decode RGB."""
    import h5py
    import pandas as pd

    raw, dial, output = Path(raw), Path(dial), Path(output)
    source = raw / "LeRobot" / f"gr1_unified.{task}"
    rows = [json.loads(line) for line in (source / "meta/episodes.jsonl").read_text().splitlines()]
    rows = sorted((row for row in rows if 0 <= row["episode_index"] < count),
                  key=lambda row: row["episode_index"])
    if len(rows) != count or len({row["trajectory_id"] for row in rows}) != count:
        raise ValueError("Missing or duplicate trajectory selection")
    script = dial / "preprocessing/extract_and_visualize_3d-pos_6d-rot_from_gr1.py"
    spec = importlib.util.spec_from_file_location("dial_replay", script)
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    replay_dir = output / "replay" / task
    replay_dir.mkdir(parents=True, exist_ok=True)
    hdf5 = raw / "HDF5" / f"{task}.hdf5"
    for row in rows:
        demo = int(row["trajectory_id"].rsplit("-", 1)[1])
        destination = replay_dir / f"demo_{demo}.parquet"
        if destination.exists():
            continue
        metadata = replay.get_env_metadata_from_dataset(str(hdf5))
        kwargs = dict(metadata["env_kwargs"])
        kwargs.update(env_name=metadata["env_name"], has_renderer=False,
                      has_offscreen_renderer=False, use_camera_obs=False)
        kwargs.pop("env_lang", None)
        # Replay restores recorded states; it never executes IK or control actions.
        controller = kwargs["controller_configs"]
        controller.update(type="BASIC", composite_controller_specific_configs={},
                          control_delta=False)
        env = replay.robosuite.make(**kwargs)
        try:
            with h5py.File(hdf5, "r") as handle:
                group = handle[f"data/demo_{demo}"]
                states = group["states"][()]
                model = ET.fromstring(group.attrs["model_file"])
                for site in model.findall(".//site"):
                    if "pinch_spheres" in site.get("name", ""):
                        site.set("rgba", "0 0 0 0")
                initial = {"states": states[0], "model": ET.tostring(model),
                           "ep_meta": group.attrs.get("ep_meta", None)}
            records = replay.playback_trajectory_with_env(
                args=SimpleNamespace(), ep=f"demo_{demo}", env=env,
                initial_state=initial, states=states, actions=None,
                video_writer=None, camera_names=["egoview"], verbose=False,
            )
            if len(records) != len(states):
                raise ValueError("Replay frame count changed")
            if any(not np.isfinite(value).all() for record in records for value in record.values()):
                raise ValueError(f"Non-finite replay: {task}/{demo}")
            temporary = destination.with_suffix(".partial")
            pd.DataFrame(records).to_parquet(temporary)
            os.replace(temporary, destination)
            print(f"REPLAY_DONE task={task} demo={demo} frames={len(records)}", flush=True)
        finally:
            env.close()

    augmented = output / "augmented" / source.name
    subprocess.run([sys.executable, str(dial / "preprocessing/aug_lerobot_data.py"),
                    "--lerobot_base_path", str(source), "--replay_base_path", str(replay_dir),
                    "--output_base_path", str(augmented)], check=True)
    convert_episodes(augmented, output, count)
    print(f"PREPARE_DONE task={task} episodes={count}", flush=True)


def convert_episodes(source, output, count=100):
    import cv2
    import pandas as pd

    source, output = Path(source), Path(output)
    task = source.name.removeprefix("gr1_unified.")
    destination = output / task
    destination.mkdir(parents=True, exist_ok=True)
    metadata = {row["episode_index"]: row for row in
                (json.loads(line) for line in (source / "meta/episodes.jsonl").read_text().splitlines())}

    def select(array, spans):
        return np.concatenate([array[..., a:b] for a, b in spans], axis=-1)

    wrist = [(98, 107), (44, 53)]
    joints = [(22, 29), (29, 35), (0, 7), (7, 13), (41, 44)]
    records = []
    for parquet in sorted((source / "data/chunk-000").glob("episode_*.parquet")):
        index = int(parquet.stem.rsplit("_", 1)[-1])
        frame = pd.read_parquet(parquet)
        state = np.stack(frame["observation.state"].to_numpy()).astype(np.float32)
        action = np.stack(frame["action"].to_numpy()).astype(np.float32)
        state_wrist, state_joints = select(state, wrist), select(state, joints)
        state_raw = np.concatenate([state_wrist, state_joints], axis=-1)
        state76 = np.concatenate([state_wrist, np.sin(state_joints), np.cos(state_joints)], axis=-1)
        action47 = np.concatenate([select(action, wrist), select(action, joints)], axis=-1)
        video = source / "videos/chunk-000/observation.images.ego_view" / f"episode_{index:06d}.mp4"
        capture = cv2.VideoCapture(str(video))
        images = []
        try:
            while True:
                ok, bgr = capture.read()
                if not ok:
                    break
                if bgr.shape[:2] != (256, 256):
                    raise ValueError(f"Unexpected video resolution: {video}")
                images.append(cv2.cvtColor(bgr[43:211, :256], cv2.COLOR_BGR2RGB))
        finally:
            capture.release()
        if len(images) != len(frame):
            raise ValueError(f"Video/parquet frame mismatch: {video}")
        meta = metadata[index]
        instruction = str(meta.get("remarks") or meta.get("description") or task).strip()
        path = destination / f"{task}__demo_{index:03d}.npz"
        temporary = path.with_suffix(".partial.npz")
        np.savez_compressed(temporary, imgs_offset=np.stack(images), proprio_raw=state_raw,
                            proprio=state76, actions=action47, lang=np.asarray(instruction),
                            episode_index=np.int32(index), trajectory_id=np.asarray(str(meta["trajectory_id"])))
        os.replace(temporary, path)
        records.append({"episode": index, "frames": len(frame), "file": path.name,
                        "bytes": path.stat().st_size, "lang": instruction})
    if len(records) != count:
        raise ValueError(f"Expected {count} episodes for {task}, got {len(records)}")
    manifest = {"task": task, "episodes": len(records), "frames": sum(r["frames"] for r in records),
                "image_shape": [168, 256, 3], "state_raw_dim": 47, "state_dim": 76,
                "action_dim": 47, "action_horizon": 16, "records": records}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def encode_instructions(raw, model_path, output):
    from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

    tasks = set()
    for meta in sorted(Path(raw).glob("*/meta/episodes.jsonl")):
        for line in meta.read_text().splitlines():
            row = json.loads(line)
            if 0 <= row["episode_index"] < 100:
                tasks.add(str(row.get("remarks") or row.get("description") or
                              meta.parents[1].name.removeprefix("gr1_unified.")).strip())
    if not tasks:
        raise ValueError("No training instructions found")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = Qwen2TokenizerFast.from_pretrained(model_path, local_files_only=True)
    model = Qwen3ForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, output_hidden_states=True,
        local_files_only=True,
    ).eval()
    for task in sorted(tasks):
        prompt = PROMPT.format(task=task)
        digest = hashlib.sha256(prompt.encode()).hexdigest()
        destination = output / f"{digest}.npz"
        if destination.exists():
            continue
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        inputs = tokenizer([rendered], return_tensors="pt", padding="max_length",
                           truncation=True, max_length=128)
        with torch.inference_mode():
            result = model(**inputs, output_hidden_states=True, use_cache=False)
        hidden = torch.cat([result.hidden_states[i] for i in (9, 18, 27)], dim=-1)[0]
        temporary = destination.with_suffix(".partial.npz")
        np.savez(temporary, hidden_states=hidden.float().numpy(),
                 attention_mask=inputs.attention_mask[0].bool().numpy())
        os.replace(temporary, destination)


def finalize_data(output):
    """Compute training-set statistics and memory-map the decoded images."""
    output = Path(output)
    accum = {key: {"n": 0, "sum": np.zeros(dim), "sumsq": np.zeros(dim),
                   "min": np.full(dim, np.inf), "max": np.full(dim, -np.inf)}
             for key, dim in (("proprio", 76), ("actions", 47))}
    for manifest in sorted(output.glob("*/manifest.json")):
        for record in json.loads(manifest.read_text())["records"]:
            path = manifest.parent / record["file"]
            with np.load(path, allow_pickle=False) as episode:
                for key, acc in accum.items():
                    value = episode[key].astype(np.float64)
                    acc["n"] += len(value)
                    acc["sum"] += value.sum(0)
                    acc["sumsq"] += (value * value).sum(0)
                    acc["min"] = np.minimum(acc["min"], value.min(0))
                    acc["max"] = np.maximum(acc["max"], value.max(0))
                mapped = path.with_suffix(".frames.npy")
                if not mapped.exists():
                    temporary = path.with_suffix(".frames.partial.npy")
                    np.save(temporary, episode["imgs_offset"], allow_pickle=False)
                    os.replace(temporary, mapped)
    stats = {}
    for key, acc in accum.items():
        if not acc["n"]:
            raise ValueError("No prepared training frames")
        mean = acc["sum"] / acc["n"]
        std = np.sqrt(np.maximum(acc["sumsq"] / acc["n"] - mean**2, 0))
        stats[key] = {name: value.tolist() for name, value in
                      (("mean", mean), ("std", std), ("min", acc["min"]), ("max", acc["max"]))}
    temporary = output / "global_stats.json.partial"
    temporary.write_text(json.dumps(stats, indent=2) + "\n")
    os.replace(temporary, output / "global_stats.json")


def main():
    from huggingface_hub import snapshot_download
    from eval.robocasa import TASKS

    root = Path.cwd()
    sim = root / ".venv-robocasa/bin/python"
    dial = root / "assets/DIAL"
    if not sim.is_file() or not (root / "assets/robocasa-download.complete").exists():
        raise FileNotFoundError("Run the RoboCasa installation in README.md first")
    raw, output = root / "data/robocasa_raw", root / "data/robocasa"
    patterns = []
    for task in TASKS:
        prefix = f"LeRobot/gr1_unified.{task}"
        episode = "episode_0000[0-9][0-9]"
        patterns.extend([f"HDF5/{task}.hdf5", f"{prefix}/meta/*",
                         f"{prefix}/data/chunk-000/{episode}.parquet",
                         f"{prefix}/videos/chunk-000/observation.images.ego_view/{episode}.mp4"])
    snapshot_download(
        "nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim", repo_type="dataset",
        revision="09c6de8af50168090e7e9cc01e1ec3bce788de24",
        allow_patterns=patterns, local_dir=raw,
    )
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
               PYTHONPATH=os.pathsep.join(map(str, (
                   root, dial, root / "assets/robocasa-gr1-tabletop-tasks"))))
    for task in TASKS:
        subprocess.run(
            [str(sim), "-c", "from training.robocasa_data import prepare_task; "
             f"prepare_task({str(raw)!r}, {str(dial)!r}, {str(output)!r}, {task!r})"],
            env=env, check=True,
        )
    finalize_data(output)
    encode_instructions(raw / "LeRobot", root / "assets/Qwen3-4B", output / "text_cache")
    print(f"Prepared 24 tasks x 100 demonstrations: {output}", flush=True)


if __name__ == "__main__":
    main()
