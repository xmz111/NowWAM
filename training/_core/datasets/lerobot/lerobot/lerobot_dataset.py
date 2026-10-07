#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable, List, Literal
import warnings
import datasets
import numpy as np
import packaging.version
import torch
import torch.utils
import pyarrow.parquet as pq
from datasets import concatenate_datasets, load_dataset
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.constants import REPOCARD_NAME
from huggingface_hub.errors import RevisionNotFoundError
from ..constants import HF_LEROBOT_HOME
from .datasets.compute_stats import aggregate_stats, compute_episode_stats
from .datasets.utils import (
    DEFAULT_FEATURES,
    DEFAULT_IMAGE_PATH,
    INFO_PATH,
    TASKS_PATH,
    _validate_feature_names,
    append_jsonlines,
    backward_compatible_episodes_stats,
    check_timestamps_sync,
    create_empty_dataset_info,
    create_lerobot_dataset_card,
    embed_images,
    get_delta_indices,
    get_episode_data_index,
    get_hf_features_from_features,
    hf_transform_to_torch,
    load_episodes,
    load_episodes_stats,
    load_info,
    load_stats,
    load_tasks,
    load_annotations,
    validate_episode_buffer,
    validate_frame,
    write_episode,
    write_episode_stats,
    write_info,
    write_json,
)
from .datasets.video_utils import (
    VideoFrame,
    decode_video_frames,
    encode_video_frames,
    get_safe_default_codec,
    get_video_info,
)
import traceback


CODEBASE_VERSION = "v2.1"


def _as_plain_dict(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "items"):
        return {k: _as_plain_dict(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_as_plain_dict(v) for v in value]
    return value


def _is_bridge_enabled(hetero_bridge: dict | None) -> bool:
    return isinstance(hetero_bridge, dict) and bool(hetero_bridge.get("enabled", False))


class HeteroLeRobotBridge:
    _PART_DIMS = {"left_pose": 7, "left_gripper": 1, "right_pose": 7, "right_gripper": 1}
    _PART_ORDER = ("left_pose", "left_gripper", "right_pose", "right_gripper")

    def __init__(self, config: dict, dataset_dirs: list[str]):
        self.config = _as_plain_dict(config) or {}
        self.dataset_dirs = [str(p) for p in dataset_dirs]
        canonical = self.config.get("canonical", {})
        self.action_key = canonical.get("action_key", "action")
        self.state_key = canonical.get("state_key", "observation.state")
        self.image_keys = canonical.get(
            "image_keys",
            {
                "cam_high": "observation.images.cam_high",
                "cam_left": "observation.images.cam_left",
                "cam_right": "observation.images.cam_right",
            },
        )
        self.specs = self._normalize_specs(self.config.get("embodiments", {}))
        self.dataset_specs = [self._resolve_spec(ds_dir) for ds_dir in self.dataset_dirs]

    @staticmethod
    def _normalize_specs(raw_specs: dict | list) -> dict[str, dict]:
        if isinstance(raw_specs, list):
            specs = {}
            for spec in raw_specs:
                name = spec.get("name") or spec.get("embodiment")
                if not name:
                    raise ValueError(
                        "Each hetero bridge embodiment spec must define `name` or `embodiment`."
                    )
                specs[str(name)] = dict(spec)
            return specs
        if isinstance(raw_specs, dict):
            return {str(name): dict(spec) for name, spec in raw_specs.items()}
        raise TypeError(
            f"`hetero_bridge.embodiments` must be a dict or list, got {type(raw_specs)}"
        )

    def _resolve_spec(self, dataset_dir: str) -> dict:
        candidates = []
        path_text = f"/{Path(dataset_dir).as_posix().strip('/')}/"
        for name, spec in self.specs.items():
            patterns = spec.get("path_patterns") or spec.get("path_contains") or [f"/{name}/"]
            if isinstance(patterns, str):
                patterns = [patterns]
            for pattern in patterns:
                pattern_text = str(pattern)
                if pattern_text in path_text or pattern_text.strip("/") in Path(dataset_dir).parts:
                    candidate = dict(spec)
                    candidate["name"] = name
                    candidates.append(candidate)
                    break
        if len(candidates) != 1:
            raise ValueError(
                f"Expected exactly one hetero bridge embodiment match for dataset_dir={dataset_dir!r}, got {len(candidates)}. Configure `hetero_bridge.embodiments.*.path_patterns`."
            )
        return candidates[0]

    @staticmethod
    def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            return x.unsqueeze(-1)
        return x

    @staticmethod
    def _zeros_like_time(ref: torch.Tensor, dim: int) -> torch.Tensor:
        if ref.ndim == 1:
            steps = ref.shape[0]
            device = ref.device
            dtype = torch.float32
        else:
            steps = ref.shape[0]
            device = ref.device
            dtype = ref.dtype if ref.dtype.is_floating_point else torch.float32
        return torch.zeros((steps, dim), dtype=dtype, device=device)

    def _raw_keys_for_spec(self, spec: dict, kind: str) -> list[str]:
        mapping = spec.get(kind, {})
        keys = []
        for part in self._PART_ORDER:
            raw_key = mapping.get(part)
            if raw_key is not None:
                keys.append(str(raw_key))
        return keys

    def map_delta_timestamps(
        self, canonical_delta_timestamps: dict | None, dataset_idx: int
    ) -> dict | None:
        if canonical_delta_timestamps is None:
            return None
        spec = self.dataset_specs[dataset_idx]
        mapped: dict[str, list[float]] = {}
        raw_image_by_canonical = {
            canonical_key: spec.get("images", {}).get(role)
            for role, canonical_key in self.image_keys.items()
        }
        for key, deltas in canonical_delta_timestamps.items():
            if key == self.action_key:
                for raw_key in self._raw_keys_for_spec(spec, "action"):
                    mapped[raw_key] = deltas
            elif key == self.state_key:
                for raw_key in self._raw_keys_for_spec(spec, "state"):
                    mapped[raw_key] = deltas
            elif key in raw_image_by_canonical:
                raw_key = raw_image_by_canonical[key]
                if raw_key is not None:
                    mapped[str(raw_key)] = deltas
            else:
                mapped[key] = deltas
        return mapped

    def _compose_vector(
        self, raw_item: dict, spec: dict, kind: str
    ) -> tuple[torch.Tensor, torch.BoolTensor, torch.BoolTensor]:
        mapping = spec.get(kind, {})
        ref = None
        for raw_key in mapping.values():
            if raw_key is not None and raw_key in raw_item:
                ref = self._ensure_2d(raw_item[raw_key])
                break
        if ref is None:
            raise KeyError(f"No raw {kind} keys from hetero bridge spec were found in sample.")
        parts = []
        dim_mask = []
        temporal_masks = []
        for part in self._PART_ORDER:
            dim = self._PART_DIMS[part]
            raw_key = mapping.get(part)
            if raw_key is None:
                parts.append(self._zeros_like_time(ref, dim))
                dim_mask.extend([True] * dim)
                continue
            if raw_key not in raw_item:
                raise KeyError(
                    f"Missing raw {kind} key {raw_key!r} for hetero bridge part {part!r}."
                )
            value = self._ensure_2d(raw_item[raw_key]).to(dtype=torch.float32)
            if value.shape[-1] != dim:
                raise ValueError(
                    f"Raw {kind} key {raw_key!r} has dim {value.shape[-1]}, expected {dim} for part {part!r}."
                )
            parts.append(value)
            dim_mask.extend([False] * dim)
            pad_key = f"{raw_key}_is_pad"
            if pad_key in raw_item:
                temporal_masks.append(raw_item[pad_key].to(dtype=torch.bool))
        vector = torch.cat(parts, dim=-1)
        if temporal_masks:
            temporal_mask = torch.stack(temporal_masks, dim=0).any(dim=0)
        else:
            temporal_mask = torch.zeros(vector.shape[0], dtype=torch.bool, device=vector.device)
        return (vector, torch.BoolTensor(dim_mask), temporal_mask)

    def _format_images(self, raw_item: dict, spec: dict, canonical: dict) -> None:
        images = spec.get("images", {})
        ref_image = None
        ref_mask = None
        for role in self.image_keys:
            raw_key = images.get(role)
            if raw_key is not None and raw_key in raw_item:
                ref_image = raw_item[raw_key]
                ref_mask = raw_item.get(f"{raw_key}_is_pad")
                break
        for role, canonical_key in self.image_keys.items():
            raw_key = images.get(role)
            if raw_key is None:
                if ref_image is None:
                    continue
                canonical[canonical_key] = torch.zeros_like(ref_image)
                canonical[f"{canonical_key}_is_pad"] = (
                    torch.zeros(ref_image.shape[0], dtype=torch.bool)
                    if ref_mask is None
                    else torch.ones_like(ref_mask, dtype=torch.bool)
                )
                continue
            if raw_key not in raw_item:
                raise KeyError(f"Missing raw image key {raw_key!r} for role {role!r}.")
            canonical[canonical_key] = raw_item[raw_key]
            canonical[f"{canonical_key}_is_pad"] = raw_item.get(
                f"{raw_key}_is_pad", torch.zeros(raw_item[raw_key].shape[0], dtype=torch.bool)
            )

    def format_item(self, raw_item: dict, dataset_idx: int) -> dict:
        spec = self.dataset_specs[dataset_idx]
        canonical = dict(raw_item)
        action, action_dim_mask, action_pad = self._compose_vector(raw_item, spec, "action")
        state, state_dim_mask, state_pad = self._compose_vector(raw_item, spec, "state")
        canonical[self.action_key] = action
        canonical[self.state_key] = state
        canonical[f"{self.action_key}_is_pad"] = action_pad
        canonical[f"{self.state_key}_is_pad"] = state_pad
        canonical["action_dim_is_pad"] = action_dim_mask
        canonical["state_dim_is_pad"] = state_dim_mask
        canonical["embodiment"] = spec["name"]
        self._format_images(raw_item, spec, canonical)
        return canonical


class LeRobotDatasetMetadata:
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        revision: str | None = None,
        force_cache_sync: bool = False,
        load_stats_metadata: bool = True,
    ):
        self.repo_id = repo_id
        self.revision = revision if revision else CODEBASE_VERSION
        self.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id
        self.load_stats_metadata = bool(load_stats_metadata)
        try:
            if force_cache_sync:
                raise FileNotFoundError
            self.load_metadata()
        except (FileNotFoundError, NotADirectoryError):
            (self.root / "meta").mkdir(exist_ok=True, parents=True)
            self.pull_from_repo(allow_patterns="meta/")
            self.load_metadata()

    def load_metadata(self):
        self.info = load_info(self.root)
        self.tasks, self.task_to_task_index = load_tasks(self.root)
        if (self.root / "annotations").exists():
            self.annotations = load_annotations(self.root)
        self.episodes = load_episodes(self.root)
        if self._version < packaging.version.parse("v2.1"):
            if self.load_stats_metadata:
                self.stats = load_stats(self.root)
                self.episodes_stats = backward_compatible_episodes_stats(self.stats, self.episodes)
            else:
                self.stats = {}
                self.episodes_stats = {}
        elif self.load_stats_metadata:
            self.episodes_stats = load_episodes_stats(self.root)
            self.stats = aggregate_stats(list(self.episodes_stats.values()))
        else:
            self.episodes_stats = {}
            self.stats = {}

    def pull_from_repo(
        self,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
    ) -> None:
        snapshot_download(
            self.repo_id,
            repo_type="dataset",
            revision=self.revision,
            local_dir=self.root,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )

    @property
    def _version(self) -> packaging.version.Version:
        return packaging.version.parse(self.info["codebase_version"])

    def get_data_file_path(self, ep_index: int) -> Path:
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.data_path.format(episode_chunk=ep_chunk, episode_index=ep_index)
        return Path(fpath)

    def get_video_file_path(self, ep_index: int, vid_key: str) -> Path:
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.video_path.format(
            episode_chunk=ep_chunk, video_key=vid_key, episode_index=ep_index
        )
        return Path(fpath)

    def get_episode_chunk(self, ep_index: int) -> int:
        return ep_index // self.chunks_size

    @property
    def data_path(self) -> str:
        return self.info["data_path"]

    @property
    def video_path(self) -> str | None:
        return self.info["video_path"]

    @property
    def robot_type(self) -> str | None:
        return self.info["robot_type"]

    @property
    def fps(self) -> int:
        return self.info["fps"]

    @property
    def features(self) -> dict[str, dict]:
        return self.info["features"]

    @property
    def image_keys(self) -> list[str]:
        return [key for key, ft in self.features.items() if ft["dtype"] == "image"]

    @property
    def video_keys(self) -> list[str]:
        return [key for key, ft in self.features.items() if ft["dtype"] == "video"]

    @property
    def camera_keys(self) -> list[str]:
        return [key for key, ft in self.features.items() if ft["dtype"] in ["video", "image"]]

    @property
    def names(self) -> dict[str, list | dict]:
        return {key: ft["names"] for key, ft in self.features.items()}

    @property
    def shapes(self) -> dict:
        return {key: tuple(ft["shape"]) for key, ft in self.features.items()}

    @property
    def total_episodes(self) -> int:
        return self.info["total_episodes"]

    @property
    def total_frames(self) -> int:
        return self.info["total_frames"]

    @property
    def total_tasks(self) -> int:
        return self.info["total_tasks"]

    @property
    def total_chunks(self) -> int:
        return self.info["total_chunks"]

    @property
    def chunks_size(self) -> int:
        return self.info["chunks_size"]

    def get_task_index(self, task: str) -> int | None:
        return self.task_to_task_index.get(task, None)

    def add_task(self, task: str):
        if task in self.task_to_task_index:
            raise ValueError(f"The task '{task}' already exists and can't be added twice.")
        task_index = self.info["total_tasks"]
        self.task_to_task_index[task] = task_index
        self.tasks[task_index] = task
        self.info["total_tasks"] += 1
        task_dict = {"task_index": task_index, "task": task}
        append_jsonlines(task_dict, self.root / TASKS_PATH)

    def save_episode(
        self,
        episode_index: int,
        episode_length: int,
        episode_tasks: list[str],
        episode_stats: dict[str, dict],
        raw_file_name: str | None = None,
    ) -> None:
        self.info["total_episodes"] += 1
        self.info["total_frames"] += episode_length
        chunk = self.get_episode_chunk(episode_index)
        if chunk >= self.total_chunks:
            self.info["total_chunks"] += 1
        self.info["splits"] = {"train": f"0:{self.info['total_episodes']}"}
        self.info["total_videos"] += len(self.video_keys)
        if len(self.video_keys) > 0:
            self.update_video_info()
        write_info(self.info, self.root)
        episode_dict = {
            "episode_index": episode_index,
            "tasks": episode_tasks,
            "length": episode_length,
        }
        if raw_file_name is not None:
            episode_dict["raw_file_name"] = raw_file_name
        self.episodes[episode_index] = episode_dict
        write_episode(episode_dict, self.root)
        self.episodes_stats[episode_index] = episode_stats
        self.stats = aggregate_stats([self.stats, episode_stats]) if self.stats else episode_stats
        write_episode_stats(episode_index, episode_stats, self.root)

    def update_video_info(self) -> None:
        for key in self.video_keys:
            if not self.features[key].get("info", None):
                video_path = self.root / self.get_video_file_path(ep_index=0, vid_key=key)
                self.info["features"][key]["info"] = get_video_info(video_path)

    def __repr__(self):
        feature_keys = list(self.features)
        return f"{self.__class__.__name__}({{\n    Repository ID: '{self.repo_id}',\n    Total episodes: '{self.total_episodes}',\n    Total frames: '{self.total_frames}',\n    Features: '{feature_keys}',\n}})',\n"

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        features: dict,
        robot_type: str | None = None,
        root: str | Path | None = None,
        use_videos: bool = True,
    ) -> "LeRobotDatasetMetadata":
        obj = cls.__new__(cls)
        obj.repo_id = repo_id
        obj.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id
        obj.root.mkdir(parents=True, exist_ok=False)
        features = {**features, **DEFAULT_FEATURES}
        _validate_feature_names(features)
        obj.tasks, obj.task_to_task_index = ({}, {})
        obj.episodes_stats, obj.stats, obj.episodes = ({}, {}, {})
        obj.info = create_empty_dataset_info(
            CODEBASE_VERSION, fps, features, use_videos, robot_type
        )
        if len(obj.video_keys) > 0 and (not use_videos):
            raise ValueError()
        write_json(obj.info, obj.root / INFO_PATH)
        obj.revision = None
        return obj


class CachedLeRobotDatasetMetadata(LeRobotDatasetMetadata):
    def __init__(
        self, repo_id: str, root: str | Path, payload: dict[str, Any], revision: str | None = None
    ):
        self.repo_id = repo_id
        self.revision = revision if revision else CODEBASE_VERSION
        self.root = Path(root)
        self.load_stats_metadata = False
        self.info = payload["info"]
        self.tasks = {
            int(item["task_index"]): item["task"]
            for item in sorted(payload.get("tasks", []), key=lambda x: x["task_index"])
        }
        self.task_to_task_index = {task: task_index for task_index, task in self.tasks.items()}
        self.episodes = {
            int(item["episode_index"]): item
            for item in sorted(payload.get("episodes", []), key=lambda x: x["episode_index"])
        }
        self.annotations = {}
        self.stats = {}
        self.episodes_stats = {}


class LeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 0.0001,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        video_codec: Literal["h264", "hevc", "libsvtav1", "h264_nvenc"] = "libsvtav1",
        is_compute_episode_stats_image: bool = True,
        nonidle_filter_path: str | Path | None = None,
        load_stats_metadata: bool = True,
        cached_metadata: dict[str, Any] | None = None,
        hf_dataset_cache_dir: str | Path | None = None,
    ):
        super().__init__()
        self.repo_id = repo_id
        self.root = Path(root) if root else HF_LEROBOT_HOME / repo_id

        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.episodes = episodes
        self._episodes_to_pos = (
            {int(ep): i for i, ep in enumerate(self.episodes)}
            if self.episodes is not None
            else None
        )
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else CODEBASE_VERSION
        self.video_backend = video_backend if video_backend else get_safe_default_codec()
        self.video_codec = video_codec
        self.hf_dataset_cache_dir = (
            Path(hf_dataset_cache_dir).expanduser() if hf_dataset_cache_dir else None
        )
        self.is_compute_episode_stats_image = is_compute_episode_stats_image
        self.delta_indices = None
        self.during_training = True
        self.nonidle_filter_path = (
            None if nonidle_filter_path is None else Path(nonidle_filter_path).expanduser()
        )
        self._nonidle_filtered_indices: list[int] | None = None
        self._nonidle_keep_indices_by_episode_pos: list[list[int]] | None = None
        self._nonidle_raw_index_to_keep_rank: dict[int, int] | None = None
        self.image_writer = None
        self.episode_buffer = None
        self.root.mkdir(exist_ok=True, parents=True)
        using_cached_metadata = (
            cached_metadata is not None and (not force_cache_sync) and (not load_stats_metadata)
        )
        if using_cached_metadata:
            self.meta = CachedLeRobotDatasetMetadata(
                self.repo_id, self.root, cached_metadata, revision=self.revision
            )
        else:
            self.meta = LeRobotDatasetMetadata(
                self.repo_id,
                self.root,
                self.revision,
                force_cache_sync=force_cache_sync,
                load_stats_metadata=load_stats_metadata,
            )
        if (
            load_stats_metadata
            and self.episodes is not None
            and (self.meta._version >= packaging.version.parse("v2.1"))
        ):
            episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]
            self.stats = aggregate_stats(episodes_stats)
        elif not load_stats_metadata:
            self.stats = {}
        try:
            if force_cache_sync:
                raise FileNotFoundError
            if not using_cached_metadata:
                stat_start = time.perf_counter()
                assert all(
                    ((self.root / fpath).is_file() for fpath in self.get_episodes_file_paths())
                )
            self.hf_dataset = self.load_hf_dataset()
        except (AssertionError, FileNotFoundError, NotADirectoryError):
            self.revision = CODEBASE_VERSION
            self.download_episodes(download_videos)
            self.hf_dataset = self.load_hf_dataset()
        self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)
        self._selected_episode_indices = (
            list(self.episodes) if self.episodes is not None else sorted(self.meta.episodes)
        )
        self._load_nonidle_filter()
        if self.delta_timestamps is not None:
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

    def _load_nonidle_filter(self) -> None:
        if self.nonidle_filter_path is None:
            return
        if not self.nonidle_filter_path.exists():
            raise FileNotFoundError(f"Non-idle filter JSON not found: {self.nonidle_filter_path}")
        payload = json.loads(self.nonidle_filter_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "episodes" in payload:
            episode_ranges = payload["episodes"]
        else:
            episode_ranges = payload
        if not isinstance(episode_ranges, dict):
            raise ValueError(
                f"Non-idle filter JSON must contain an episode range mapping, got {type(episode_ranges)}"
            )
        filtered_indices: list[int] = []
        keep_by_episode_pos: list[list[int]] = []
        raw_to_rank: dict[int, int] = {}
        for episode_pos, episode_idx in enumerate(self._selected_episode_indices):
            ep_start = int(self.episode_data_index["from"][episode_pos].item())
            ep_end = int(self.episode_data_index["to"][episode_pos].item())
            ranges = episode_ranges.get(
                str(episode_idx), episode_ranges.get(int(episode_idx), None)
            )
            if ranges is None:
                keep_indices = list(range(ep_start, ep_end))
            else:
                keep_indices = []
                for raw_start, raw_end in ranges:
                    start = max(0, int(raw_start))
                    end = min(ep_end - ep_start, int(raw_end))
                    if end <= start:
                        continue
                    keep_indices.extend(range(ep_start + start, ep_start + end))
            keep_indices = sorted(set(keep_indices))
            keep_by_episode_pos.append(keep_indices)
            for keep_rank, raw_idx in enumerate(keep_indices):
                raw_to_rank[int(raw_idx)] = keep_rank
            filtered_indices.extend(keep_indices)
        if len(filtered_indices) == 0:
            raise ValueError(f"Non-idle filter removed all frames: {self.nonidle_filter_path}")
        self._nonidle_filtered_indices = filtered_indices
        self._nonidle_keep_indices_by_episode_pos = keep_by_episode_pos
        self._nonidle_raw_index_to_keep_rank = raw_to_rank
        logging.info(
            "Loaded non-idle filter %s: kept %d/%d frames.",
            self.nonidle_filter_path,
            len(filtered_indices),
            len(self.hf_dataset),
        )

    def push_to_hub(
        self,
        branch: str | None = None,
        tags: list | None = None,
        license: str | None = "apache-2.0",
        tag_version: bool = True,
        push_videos: bool = True,
        private: bool = False,
        allow_patterns: list[str] | str | None = None,
        upload_large_folder: bool = False,
        **card_kwargs,
    ) -> None:
        ignore_patterns = ["images/"]
        if not push_videos:
            ignore_patterns.append("videos/")
        hub_api = HfApi()
        hub_api.create_repo(
            repo_id=self.repo_id, private=private, repo_type="dataset", exist_ok=True
        )
        if branch:
            hub_api.create_branch(
                repo_id=self.repo_id,
                branch=branch,
                revision=self.revision,
                repo_type="dataset",
                exist_ok=True,
            )
        upload_kwargs = {
            "repo_id": self.repo_id,
            "folder_path": self.root,
            "repo_type": "dataset",
            "revision": branch,
            "allow_patterns": allow_patterns,
            "ignore_patterns": ignore_patterns,
        }
        if upload_large_folder:
            hub_api.upload_large_folder(**upload_kwargs)
        else:
            hub_api.upload_folder(**upload_kwargs)
        if not hub_api.file_exists(
            self.repo_id, REPOCARD_NAME, repo_type="dataset", revision=branch
        ):
            card = create_lerobot_dataset_card(
                tags=tags, dataset_info=self.meta.info, license=license, **card_kwargs
            )
            card.push_to_hub(repo_id=self.repo_id, repo_type="dataset", revision=branch)
        if tag_version:
            with contextlib.suppress(RevisionNotFoundError):
                hub_api.delete_tag(self.repo_id, tag=CODEBASE_VERSION, repo_type="dataset")
            hub_api.create_tag(
                self.repo_id, tag=CODEBASE_VERSION, revision=branch, repo_type="dataset"
            )

    def pull_from_repo(
        self,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
    ) -> None:
        snapshot_download(
            self.repo_id,
            repo_type="dataset",
            revision=self.revision,
            local_dir=self.root,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )

    def download_episodes(self, download_videos: bool = True) -> None:
        files = None
        ignore_patterns = None if download_videos else "videos/"
        if self.episodes is not None:
            files = self.get_episodes_file_paths()
        self.pull_from_repo(allow_patterns=files, ignore_patterns=ignore_patterns)

    def get_episodes_file_paths(self) -> list[Path]:
        episodes = (
            self.episodes if self.episodes is not None else list(range(self.meta.total_episodes))
        )
        fpaths = [str(self.meta.get_data_file_path(ep_idx)) for ep_idx in episodes]
        if len(self.meta.video_keys) > 0:
            video_files = [
                str(self.meta.get_video_file_path(ep_idx, vid_key))
                for vid_key in self.meta.video_keys
                for ep_idx in episodes
            ]
            fpaths += video_files
        return fpaths

    def load_hf_dataset(self) -> datasets.Dataset:
        start = time.perf_counter()

        load_kwargs = {}
        if self.hf_dataset_cache_dir is not None:
            load_kwargs["cache_dir"] = str(self.hf_dataset_cache_dir)
        if self.episodes is None:
            episode_indices = sorted(self.meta.episodes)
            files = [
                str(self.root / self.meta.get_data_file_path(ep_idx)) for ep_idx in episode_indices
            ]
            hf_dataset = load_dataset("parquet", data_files=files, split="train", **load_kwargs)
        else:
            files = [
                str(self.root / self.meta.get_data_file_path(ep_idx)) for ep_idx in self.episodes
            ]
            hf_dataset = load_dataset("parquet", data_files=files, split="train", **load_kwargs)
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    def create_hf_dataset(self) -> datasets.Dataset:
        features = get_hf_features_from_features(self.features)
        ft_dict = {col: [] for col in features}
        hf_dataset = datasets.Dataset.from_dict(ft_dict, features=features, split="train")
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    @property
    def fps(self) -> int:
        return self.meta.fps

    @property
    def num_frames(self) -> int:
        if self._nonidle_filtered_indices is not None:
            return len(self._nonidle_filtered_indices)
        return len(self.hf_dataset) if self.hf_dataset is not None else self.meta.total_frames

    @property
    def num_episodes(self) -> int:
        return len(self.episodes) if self.episodes is not None else self.meta.total_episodes

    @property
    def features(self) -> dict[str, dict]:
        return self.meta.features

    @property
    def hf_features(self) -> datasets.Features:
        if self.hf_dataset is not None:
            return self.hf_dataset.features
        else:
            return get_hf_features_from_features(self.features)

    def _get_query_indices(self, idx: int, ep_idx: int) -> tuple[dict[str, list[int | bool]]]:
        if self._nonidle_filtered_indices is not None:
            if (
                self._nonidle_keep_indices_by_episode_pos is None
                or self._nonidle_raw_index_to_keep_rank is None
            ):
                raise RuntimeError("Non-idle filter index tables are not initialized.")
            keep_indices = self._nonidle_keep_indices_by_episode_pos[ep_idx]
            if len(keep_indices) == 0:
                raise IndexError(f"Episode position {ep_idx} has no non-idle frames.")
            if idx not in self._nonidle_raw_index_to_keep_rank:
                raise IndexError(f"Raw index {idx} is not in the non-idle filter.")
            keep_rank = self._nonidle_raw_index_to_keep_rank[idx]
            query_indices = {}
            padding = {}
            for key, delta_idx in self.delta_indices.items():
                cur_indices = []
                cur_padding = []
                for delta in delta_idx:
                    target_rank = keep_rank + int(delta)
                    is_pad = target_rank < 0 or target_rank >= len(keep_indices)
                    clamped_rank = max(0, min(len(keep_indices) - 1, target_rank))
                    cur_indices.append(int(keep_indices[clamped_rank]))
                    cur_padding.append(bool(is_pad))
                query_indices[key] = cur_indices
                padding[f"{key}_is_pad"] = torch.BoolTensor(cur_padding)
            return (query_indices, padding)
        ep_start = self.episode_data_index["from"][ep_idx]
        ep_end = self.episode_data_index["to"][ep_idx]
        query_indices = {
            key: [max(ep_start.item(), min(ep_end.item() - 1, idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {
            f"{key}_is_pad": torch.BoolTensor(
                [
                    (idx + delta < ep_start.item()) | (idx + delta >= ep_end.item())
                    for delta in delta_idx
                ]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return (query_indices, padding)

    def _get_query_timestamps(
        self,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
        idx: int | None = None,
    ) -> dict[str, list[float]]:
        query_timestamps = {}
        fps_inv = 1.0 / float(self.fps)
        for key in self.meta.video_keys:
            if query_indices is not None and key in query_indices:
                q_idx = query_indices[key]
                if idx is not None:
                    query_timestamps[key] = [
                        current_ts + (int(qi) - int(idx)) * fps_inv for qi in q_idx
                    ]
                else:
                    timestamps = self.hf_dataset.select(q_idx)["timestamp"]
                    query_timestamps[key] = torch.stack(timestamps).tolist()
            else:
                query_timestamps[key] = [current_ts]
        return query_timestamps

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        return {
            key: torch.stack(self.hf_dataset.select(q_idx)[key])
            for key, q_idx in query_indices.items()
            if key not in self.meta.video_keys
        }

    def _query_hf_dataset_fast(self, query_indices: dict[str, list[int]]) -> dict:
        result = {}
        processed_indices = set()
        index_to_selected = {}
        for key, q_idx in query_indices.items():
            if key not in self.meta.video_keys:
                if "images" in key and (not self.during_training):
                    continue
                q_idx_tuple = tuple(q_idx)
                if q_idx_tuple not in processed_indices:
                    selected_data = self.hf_dataset[q_idx]
                    index_to_selected[q_idx_tuple] = selected_data
                    processed_indices.add(q_idx_tuple)
                else:
                    selected_data = index_to_selected[q_idx_tuple]
                values = selected_data[key]
                result[key] = values if isinstance(values, torch.Tensor) else torch.stack(values)
        return result

    def get_episode_data(self, episode_id: int) -> dict:
        ep_start = self.episode_data_index["from"][episode_id].item()
        ep_end = self.episode_data_index["to"][episode_id].item()
        q_idx = list(range(ep_start, ep_end))
        selected_data = self.hf_dataset[q_idx]
        res_keys = self.meta.features.keys() - set(self.meta.video_keys)
        res = {key: torch.stack(selected_data[key]) for key in res_keys}
        return res

    def _query_videos(
        self, query_timestamps: dict[str, list[float]], ep_idx: int
    ) -> dict[str, torch.Tensor]:
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
            frames = decode_video_frames(video_path, query_ts, self.tolerance_s, self.video_backend)
            item[vid_key] = frames.squeeze(0)
        return item

    def _add_padding_keys(self, item: dict, padding: dict[str, list[bool]]) -> dict:
        for key, val in padding.items():
            item[key] = torch.BoolTensor(val)
        return item

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx) -> dict:
        raw_idx = (
            int(self._nonidle_filtered_indices[idx])
            if self._nonidle_filtered_indices is not None
            else idx
        )
        item = self.hf_dataset[raw_idx]
        ep_idx = item["episode_index"].item()
        query_indices = None
        if self.delta_indices is not None:
            current_ep_idx = (
                self._episodes_to_pos[ep_idx] if self._episodes_to_pos is not None else ep_idx
            )
            query_indices, padding = self._get_query_indices(raw_idx, current_ep_idx)
            query_result = self._query_hf_dataset_fast(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val
        if len(self.meta.video_keys) > 0 and self.during_training:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices, idx=raw_idx)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}
        if self.image_transforms is not None:
            image_keys = self.meta.camera_keys
            for cam in image_keys:
                item[cam] = self.image_transforms(item[cam])
        task_idx = item["task_index"].item()
        item["task"] = self.meta.tasks[task_idx]
        if "coarse_task_index" in item:
            coarse_task_index = item["coarse_task_index"].item()
            item["coarse_task"] = self.meta.tasks[coarse_task_index]
        if "operating_hand_index" in item:
            operating_hand_index = item["operating_hand_index"].item()
            item["operating_hand"] = self.meta.tasks[operating_hand_index]
        if "subtask_annotation" in item and hasattr(self.meta, "annotations"):
            index = item["subtask_annotation"][0].item()
            item["subtask"] = self.meta.annotations["subtask"][index]
        if "atomic_task_index" in item and item["atomic_task_index"] is not None:
            atomic_task_index = item["atomic_task_index"].item()
            item["subtask"] = self.meta.tasks[int(atomic_task_index)]
        return item

    def __repr__(self):
        feature_keys = list(self.features)
        return f"{self.__class__.__name__}({{\n    Repository ID: '{self.repo_id}',\n    Number of selected episodes: '{self.num_episodes}',\n    Number of selected samples: '{self.num_frames}',\n    Features: '{feature_keys}',\n}})',\n"

    def create_episode_buffer(self, episode_index: int | None = None) -> dict:
        current_ep_idx = self.meta.total_episodes if episode_index is None else episode_index
        ep_buffer = {}
        ep_buffer["size"] = 0
        ep_buffer["task"] = []
        for key in self.features:
            ep_buffer[key] = current_ep_idx if key == "episode_index" else []
        return ep_buffer

    def _get_image_file_path(self, episode_index: int, image_key: str, frame_index: int) -> Path:
        fpath = DEFAULT_IMAGE_PATH.format(
            image_key=image_key, episode_index=episode_index, frame_index=frame_index
        )
        return self.root / fpath

    def add_frame(self, frame: dict, task: List[str], timestamp: float | None = None) -> None:
        assert len(task) == 4, "Task frame must be of two elements"
        for name in frame:
            if isinstance(frame[name], torch.Tensor):
                frame[name] = frame[name].numpy()
        validate_frame(frame, self.features)
        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()
        frame_index = self.episode_buffer["size"]
        if timestamp is None:
            timestamp = frame_index / self.fps
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)
        self.episode_buffer["task"].append(task)
        for key in frame:
            if key not in self.features:
                raise ValueError(
                    f"An element of the frame is not in the features. '{key}' not in '{self.features.keys()}'."
                )
            if self.features[key]["dtype"] in ["image", "video"]:
                img_path = self._get_image_file_path(
                    episode_index=self.episode_buffer["episode_index"],
                    image_key=key,
                    frame_index=frame_index,
                )
                if frame_index == 0:
                    img_path.parent.mkdir(parents=True, exist_ok=True)
                self.episode_buffer[key].append(str(img_path))
            else:
                self.episode_buffer[key].append(frame[key])
        self.episode_buffer["size"] += 1

    def save_episode(
        self, episode_data: dict | None = None, raw_file_name: str | None = None
    ) -> None:
        if not episode_data:
            episode_buffer = self.episode_buffer
        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)
        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set([item for sublist in tasks for item in sublist]))
        episode_index = episode_buffer["episode_index"]
        episode_buffer["index"] = np.arange(
            self.meta.total_frames, self.meta.total_frames + episode_length
        )
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)
        for task in episode_tasks:
            task_index = self.meta.get_task_index(task)
            if task_index is None:
                self.meta.add_task(task)
        episode_buffer["coarse_task_index"] = np.array(
            [self.meta.get_task_index(task[0]) for task in tasks]
        )
        episode_buffer["task_index"] = np.array(
            [self.meta.get_task_index(task[1]) for task in tasks]
        )
        episode_buffer["coarse_quality_index"] = np.array(
            [self.meta.get_task_index(task[2]) for task in tasks]
        )
        episode_buffer["quality_index"] = np.array(
            [self.meta.get_task_index(task[3]) for task in tasks]
        )
        for key, ft in self.features.items():
            if key in [
                "index",
                "episode_index",
                "coarse_task_index",
                "task_index",
                "coarse_quality_index",
                "quality_index",
            ] or ft["dtype"] in ["image", "video"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])
        self._wait_image_writer()
        self._save_episode_table(episode_buffer, episode_index)
        ep_stats = compute_episode_stats(
            episode_buffer, self.features, self.is_compute_episode_stats_image
        )
        if len(self.meta.video_keys) > 0:
            video_paths = self.encode_episode_videos(episode_index)
            for key in self.meta.video_keys:
                episode_buffer[key] = video_paths[key]
        self.meta.save_episode(
            episode_index, episode_length, episode_tasks, ep_stats, raw_file_name
        )
        ep_data_index = get_episode_data_index(self.meta.episodes, [episode_index])
        ep_data_index_np = {k: t.numpy() for k, t in ep_data_index.items()}
        check_timestamps_sync(
            episode_buffer["timestamp"],
            episode_buffer["episode_index"],
            ep_data_index_np,
            self.fps,
            self.tolerance_s,
        )
        video_files = list(self.root.rglob("*.mp4"))
        assert len(video_files) == self.num_episodes * len(self.meta.video_keys)
        parquet_files = list(self.root.rglob("*.parquet"))
        assert len(parquet_files) == self.num_episodes
        img_dir = self.root / "images"
        if img_dir.is_dir():
            shutil.rmtree(self.root / "images")
        if not episode_data:
            self.episode_buffer = self.create_episode_buffer()

    def _save_episode_table(self, episode_buffer: dict, episode_index: int) -> None:
        episode_dict = {key: episode_buffer[key] for key in self.hf_features}
        ep_dataset = datasets.Dataset.from_dict(
            episode_dict, features=self.hf_features, split="train"
        )
        ep_dataset = embed_images(ep_dataset)
        self.hf_dataset = concatenate_datasets([self.hf_dataset, ep_dataset])
        self.hf_dataset.set_transform(hf_transform_to_torch)
        ep_data_path = self.root / self.meta.get_data_file_path(ep_index=episode_index)
        ep_data_path.parent.mkdir(parents=True, exist_ok=True)
        ep_dataset.to_parquet(ep_data_path)

    def clear_episode_buffer(self) -> None:
        episode_index = self.episode_buffer["episode_index"]
        if self.image_writer is not None:
            for cam_key in self.meta.camera_keys:
                img_dir = self._get_image_file_path(
                    episode_index=episode_index, image_key=cam_key, frame_index=0
                ).parent
                if img_dir.is_dir():
                    shutil.rmtree(img_dir)
        self.episode_buffer = self.create_episode_buffer()

    def stop_image_writer(self) -> None:
        if self.image_writer is not None:
            self.image_writer.stop()
            self.image_writer = None

    def _wait_image_writer(self) -> None:
        if self.image_writer is not None:
            self.image_writer.wait_until_done()

    def encode_videos(self) -> None:
        for ep_idx in range(self.meta.total_episodes):
            self.encode_episode_videos(ep_idx)

    def encode_episode_videos(self, episode_index: int) -> dict:
        video_paths = {}
        for key in self.meta.video_keys:
            video_path = self.root / self.meta.get_video_file_path(episode_index, key)
            video_paths[key] = str(video_path)
            if video_path.is_file():
                continue
            img_dir = self._get_image_file_path(
                episode_index=episode_index, image_key=key, frame_index=0
            ).parent
            encode_video_frames(
                img_dir, video_path, self.fps, overwrite=True, vcodec=self.video_codec
            )
        return video_paths

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        features: dict,
        root: str | Path | None = None,
        robot_type: str | None = None,
        use_videos: bool = True,
        tolerance_s: float = 0.0001,
        image_writer_processes: int = 0,
        image_writer_threads: int = 0,
        video_backend: str | None = None,
        video_codec: Literal["h264", "hevc", "libsvtav1", "h264_nvenc"] = "libsvtav1",
        is_compute_episode_stats_image=True,
    ) -> "LeRobotDataset":
        obj = cls.__new__(cls)
        obj.meta = LeRobotDatasetMetadata.create(
            repo_id=repo_id,
            fps=fps,
            robot_type=robot_type,
            features=features,
            root=root,
            use_videos=use_videos,
        )
        obj.repo_id = obj.meta.repo_id
        obj.root = obj.meta.root
        obj.revision = None
        obj.tolerance_s = tolerance_s
        obj.image_writer = None
        obj.episode_buffer = obj.create_episode_buffer()
        obj.episodes = None
        obj.hf_dataset = obj.create_hf_dataset()
        obj.image_transforms = None
        obj.delta_timestamps = None
        obj.delta_indices = None
        obj.episode_data_index = None
        obj.video_backend = video_backend if video_backend is not None else get_safe_default_codec()
        obj.video_codec = video_codec
        obj.is_compute_episode_stats_image = is_compute_episode_stats_image
        return obj


class MultiLeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dirs: list[str],
        episodes: dict | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerances_s: dict | None = None,
        download_videos: bool = True,
        video_backend: str | None = None,
        nonidle_filter_path: str | Path | None = None,
        hetero_bridge: dict | None = None,
        lerobot_meta_cache: dict[str, dict[str, Any]] | None = None,
        hf_dataset_cache_dir: str | Path | None = None,
    ):
        super().__init__()
        self.dataset_dirs = dataset_dirs
        ds_roots = [Path(ds_dir) for ds_dir in dataset_dirs]
        ds_names = [ds_dir for ds_dir in dataset_dirs]
        self.ds_names = ds_names
        self.ds_roots = ds_roots
        self.hetero_bridge = (
            HeteroLeRobotBridge(hetero_bridge, ds_names)
            if _is_bridge_enabled(hetero_bridge)
            else None
        )
        self.tolerances_s = tolerances_s if tolerances_s else dict.fromkeys(ds_names, 0.0001)
        self._datasets = []
        for dataset_idx, (ds_root, ds_name) in enumerate(zip(ds_roots, ds_names, strict=True)):
            try:
                child_delta_timestamps = (
                    self.hetero_bridge.map_delta_timestamps(delta_timestamps, dataset_idx)
                    if self.hetero_bridge is not None
                    else delta_timestamps
                )
                _dataset = LeRobotDataset(
                    ds_name,
                    root=ds_root,
                    episodes=episodes[ds_name] if episodes else None,
                    image_transforms=image_transforms,
                    delta_timestamps=child_delta_timestamps,
                    tolerance_s=self.tolerances_s[ds_name],
                    download_videos=download_videos,
                    video_backend=video_backend,
                    nonidle_filter_path=nonidle_filter_path,
                    load_stats_metadata=self.hetero_bridge is None,
                    cached_metadata=lerobot_meta_cache.get(str(ds_root.expanduser().resolve()))
                    if lerobot_meta_cache is not None
                    else None,
                    hf_dataset_cache_dir=hf_dataset_cache_dir,
                )
                self._datasets.append(_dataset)
            except Exception as e:
                logging.error(f"Exception while process ds_root: {ds_root}, ds_name: {ds_name}")
                traceback.print_exc()
                raise e
        self.disabled_features = set()
        if self.hetero_bridge is None:
            intersection_features = set(self._datasets[0].features)
            for ds in self._datasets:
                intersection_features.intersection_update(ds.features)
            if len(intersection_features) == 0:
                raise RuntimeError(
                    "Multiple datasets were provided but they had no keys common to all of them. The multi-dataset functionality currently only keeps common keys."
                )
            for ds_name, ds in zip(self.ds_names, self._datasets, strict=True):
                extra_keys = set(ds.features).difference(intersection_features)
                if extra_keys:
                    logging.warning(
                        f"keys {extra_keys} of {ds_name} were disabled as they are not contained in all the other datasets."
                    )
                self.disabled_features.update(extra_keys)
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.stats = (
            aggregate_stats([dataset.meta.stats for dataset in self._datasets])
            if self.hetero_bridge is None
            else {}
        )

    def set_during_training(self, during_training: bool):
        for dataset in self._datasets:
            dataset.during_training = during_training

    @property
    def repo_id_to_index(self):
        return {repo_id: i for i, repo_id in enumerate(self.ds_names)}

    @property
    def repo_index_to_id(self):
        return {v: k for k, v in self.repo_id_to_index}

    @property
    def fps(self) -> int:
        return self._datasets[0].meta.info["fps"]

    @property
    def video(self) -> bool:
        return self._datasets[0].meta.info.get("video", False)

    @property
    def features(self) -> datasets.Features:
        features = {}
        for dataset in self._datasets:
            features.update(
                {k: v for k, v in dataset.hf_features.items() if k not in self.disabled_features}
            )
        return features

    @property
    def camera_keys(self) -> list[str]:
        keys = []
        for key, feats in self.features.items():
            if isinstance(feats, (datasets.Image, VideoFrame)):
                keys.append(key)
        return keys

    @property
    def video_frame_keys(self) -> list[str]:
        video_frame_keys = []
        for key, feats in self.features.items():
            if isinstance(feats, VideoFrame):
                video_frame_keys.append(key)
        return video_frame_keys

    @property
    def num_frames(self) -> int:
        return sum((d.num_frames for d in self._datasets))

    @property
    def num_episodes(self) -> int:
        return sum((d.num_episodes for d in self._datasets))

    @property
    def tolerance_s(self) -> float:
        return 1 / self.fps - 0.0001

    def get_episode_data(self, episode_idx: int) -> dict:
        for dataset_idx, dataset in enumerate(self._datasets):
            if episode_idx < dataset.num_episodes:
                episode_id = (
                    episode_idx if dataset.episodes is None else dataset.episodes[episode_idx]
                )
                file = str(dataset.root / dataset.meta.get_data_file_path(episode_id))
                table = pq.read_table(str(file))
                result_dict = {}
                for col_name in table.column_names:
                    col = table[col_name]
                    try:
                        np_arr = col.to_numpy(zero_copy_only=True)
                    except Exception:
                        raw = col.to_numpy()
                        np_arr = np.stack(raw) if raw.dtype == object else raw
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message="The given NumPy array is not writable",
                            category=UserWarning,
                        )
                        if np_arr.dtype == "O":
                            result_dict[col_name] = np_arr
                        else:
                            result_dict[col_name] = torch.from_numpy(np_arr)
                if self.hetero_bridge is not None:
                    result_dict = self.hetero_bridge.format_item(result_dict, dataset_idx)
                return result_dict
            else:
                episode_idx -= dataset.num_episodes
        raise IndexError(f"Episode index {episode_idx} out of bounds.")

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")
        start_idx = 0
        dataset_idx = 0
        for dataset in self._datasets:
            if idx >= start_idx + dataset.num_frames:
                start_idx += dataset.num_frames
                dataset_idx += 1
                continue
            break
        else:
            raise AssertionError(
                "We expect the loop to break out as long as the index is within bounds."
            )
        item = self._datasets[dataset_idx][idx - start_idx]
        item["dataset_index"] = torch.tensor(dataset_idx)
        if self.hetero_bridge is not None:
            item = self.hetero_bridge.format_item(item, dataset_idx)
            item["dataset_index"] = torch.tensor(dataset_idx)
        else:
            for data_key in self.disabled_features:
                if data_key in item:
                    del item[data_key]
        return item

    def __repr__(self):
        return f"{self.__class__.__name__}(\n  Dataset Names: '{self.ds_names}',\n  Number of Samples: {self.num_frames},\n  Number of Episodes: {self.num_episodes},\n  Type: {('video (.mp4)' if self.video else 'image (.png)')},\n  Recorded Frames per Second: {self.fps},\n  Camera Keys: {self.camera_keys},\n  Video Frame Keys: {(self.video_frame_keys if self.video else 'N/A')},\n  Transformations: {self.image_transforms},\n)"
