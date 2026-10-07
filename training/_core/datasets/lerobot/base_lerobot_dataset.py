import torch
import numpy as np
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Any, DefaultDict
from tqdm import tqdm
from .lerobot.lerobot_dataset import LeRobotDatasetMetadata, MultiLeRobotDataset
from .lerobot.lerobot_dataset_v3 import MultiLeRobotDatasetV3
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
import time
from training._core.utils.logging_config import get_logger
from .processors.base_processor import BaseProcessor

logger = get_logger(__name__)
MAX_GETITEM_ATTEMPT = 5


class _CachedLeRobotMeta:
    def __init__(self, repo_id: str, root: Path, payload: dict[str, Any]):
        self.repo_id = repo_id
        self.root = root
        self.info = payload["info"]
        self.episodes = {int(ep["episode_index"]): ep for ep in payload.get("episodes", [])}

    @property
    def fps(self) -> int:
        return int(self.info["fps"])

    @property
    def total_episodes(self) -> int:
        return int(self.info["total_episodes"])


class _InfoLeRobotMeta:
    def __init__(self, repo_id: str, root: Path, info: dict[str, Any]):
        self.repo_id = repo_id
        self.root = root
        self.info = info

    @property
    def fps(self) -> int:
        return int(self.info["fps"])

    @property
    def total_episodes(self) -> int:
        return int(self.info["total_episodes"])


def _read_lerobot_meta_cache(path: str | Path) -> dict[str, dict[str, Any]]:
    cache_path = Path(path).expanduser()
    if not cache_path.exists():
        raise FileNotFoundError(f"LeRobot meta cache not found: {cache_path}")
    with cache_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    records = payload.get("datasets", payload)
    if isinstance(records, dict):
        iterable = records.values()
    else:
        iterable = records
    cache = {}
    for record in iterable:
        root = str(Path(record["root"]).expanduser().resolve())
        cache[root] = record
    return cache


def _empty_lerobot_meta_cache() -> dict[str, dict[str, Any]]:
    return {}


class BaseLerobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dirs: List[str],
        shape_meta: Dict[str, Any],
        action_size: int = 1,
        past_action_size: int = 0,
        obs_size: int = 1,
        past_obs_size: int = 0,
        val_set_proportion: float = 0.05,
        is_training_set: bool = False,
        val_split_level: str = "episode",
        seed: int = 42,
        global_sample_stride: int = 1,
        sample_index_stride: int = 1,
        image_obs_indices: Optional[List[int]] = None,
        nonidle_filter_path: Optional[str] = None,
        hetero_bridge: Optional[Dict[str, Any]] = None,
        lerobot_meta_cache: Optional[str] = None,
        arrow_cache_dir: Optional[str] = None,
        lerobot_backend: str = "v2",
        lerobot_v3_init_num_workers: int = 1,
        lerobot_v3_index_cache: Optional[str] = None,
        lerobot_v3_video_backend: Optional[str] = None,
        lerobot_tolerance_s: Optional[float] = None,
        episode_index_filter: Optional[Dict[str, Any]] = None,
    ):
        assert len(dataset_dirs) > 0, "At least one dataset directory is required"
        assert past_action_size == 0
        assert past_obs_size == 0
        assert action_size == obs_size - 1, "In this dataset, action_size should be obs_size - 1"

        self.dataset_dirs = dataset_dirs
        self.shape_meta = shape_meta
        self.action_size = action_size
        self.past_action_size = past_action_size
        self.obs_size = obs_size
        self.processor = None
        self.hetero_bridge = hetero_bridge
        self.episode_index_filter = episode_index_filter
        self.lerobot_backend = str(lerobot_backend).strip().lower()
        if self.lerobot_backend not in {"v2", "v3"}:
            raise ValueError(
                f"Unsupported lerobot_backend={lerobot_backend!r}. Expected 'v2' or 'v3'."
            )
        meta_cache_by_root = (
            self._load_lerobot_meta_cache(lerobot_meta_cache)
            if self.lerobot_backend == "v2"
            else _empty_lerobot_meta_cache()
        )
        metas = []
        for ds_dir in tqdm(
            dataset_dirs, desc=f"Loading LeRobot {self.lerobot_backend} root metadata"
        ):
            ds_root = Path(ds_dir)
            repo_id = ds_dir
            cached_meta = meta_cache_by_root.get(str(ds_root.resolve()))
            if cached_meta is not None:
                metas.append(_CachedLeRobotMeta(repo_id=repo_id, root=ds_root, payload=cached_meta))
            elif self.lerobot_backend == "v3":
                with (ds_root / "meta" / "info.json").open("r", encoding="utf-8") as f:
                    metas.append(_InfoLeRobotMeta(repo_id=repo_id, root=ds_root, info=json.load(f)))
            else:
                meta = LeRobotDatasetMetadata(repo_id=repo_id, root=ds_root)
                metas.append(meta)
        fps_list = [m.fps for m in metas]
        assert len(set(fps_list)) == 1, f"All dataset_dirs must have the same fps, got {fps_list}"
        fps = fps_list[0]
        self.global_sample_stride = global_sample_stride
        self.sample_index_stride = int(sample_index_stride)
        assert self.sample_index_stride > 0, (
            f"sample_index_stride must be positive, got {sample_index_stride}"
        )
        self.val_set_proportion = val_set_proportion
        self.is_training_set = is_training_set
        self.val_split_level = str(val_split_level).strip().lower()
        if self.val_split_level not in {"episode", "root"}:
            raise ValueError(
                f"Unsupported val_split_level={val_split_level!r}. Expected 'episode' or 'root'."
            )
        if val_set_proportion >= 1e-06 and self.val_split_level == "root":
            root_indices = list(range(len(metas)))
            rng = np.random.default_rng(seed)
            rng.shuffle(root_indices)
            split_idx = int(len(root_indices) * (1 - val_set_proportion))
            selected_indices = (
                root_indices[:split_idx] if self.is_training_set else root_indices[split_idx:]
            )
            selected_indices = sorted(selected_indices)
            self.dataset_dirs = [self.dataset_dirs[i] for i in selected_indices]
            metas = [metas[i] for i in selected_indices]
        self.image_meta = shape_meta["images"]
        self.state_meta = shape_meta["state"]
        self.action_meta = shape_meta["action"]
        delta_timestamps = {}
        for meta in self.image_meta:
            key = meta["key"]
            default_lerobot_key = (
                f"observation.images.{key}" if key != "default" else "observation.images"
            )
            meta["lerobot_key"] = meta.get("lerobot_key") or default_lerobot_key
            image_steps = (
                image_obs_indices
                if image_obs_indices is not None
                else range(-past_obs_size, -past_obs_size + obs_size)
            )
            delta_timestamps[meta["lerobot_key"]] = [
                t * global_sample_stride / fps for t in image_steps
            ]
        for meta in self.state_meta:
            key = meta["key"]
            default_lerobot_key = (
                f"observation.state.{key}" if key != "default" else "observation.state"
            )
            meta["lerobot_key"] = meta.get("lerobot_key") or default_lerobot_key
            delta_timestamps[meta["lerobot_key"]] = [
                t * global_sample_stride / fps
                for t in range(-past_obs_size, -past_obs_size + obs_size)
            ]
        for meta in self.action_meta:
            key = meta["key"]
            default_lerobot_key = f"action.{key}" if key != "default" else "action"
            meta["lerobot_key"] = meta.get("lerobot_key") or default_lerobot_key
            delta_timestamps[meta["lerobot_key"]] = [
                t * global_sample_stride / fps
                for t in range(-past_action_size, -past_action_size + action_size)
            ]
        episodes = None
        needs_episode_selection = self.episode_index_filter is not None or (
            val_set_proportion >= 1e-06 and self.val_split_level == "episode"
        )
        if needs_episode_selection:
            episodes = {}
            for meta in metas:
                episode_indices = self._filter_episode_indices(
                    list(range(meta.total_episodes)), repo_id=meta.repo_id
                )
                if val_set_proportion >= 1e-06 and self.val_split_level == "episode":
                    split_idx = int(len(episode_indices) * (1 - val_set_proportion))
                    rng = np.random.default_rng(seed)
                    rng.shuffle(episode_indices)
                    episode_indices = (
                        episode_indices[:split_idx]
                        if self.is_training_set
                        else episode_indices[split_idx:]
                    )
                episodes.update({meta.repo_id: episode_indices})
        dataset_cls = MultiLeRobotDatasetV3 if self.lerobot_backend == "v3" else MultiLeRobotDataset
        dataset_kwargs = {
            "dataset_dirs": self.dataset_dirs,
            "episodes": episodes,
            "delta_timestamps": delta_timestamps,
            "nonidle_filter_path": nonidle_filter_path,
            "hetero_bridge": hetero_bridge,
        }
        if self.lerobot_backend == "v2":
            dataset_kwargs["lerobot_meta_cache"] = (
                meta_cache_by_root if meta_cache_by_root else None
            )
            dataset_kwargs["hf_dataset_cache_dir"] = arrow_cache_dir
        else:
            dataset_kwargs["init_num_workers"] = int(lerobot_v3_init_num_workers)
            dataset_kwargs["index_cache_path"] = lerobot_v3_index_cache
            dataset_kwargs["video_backend"] = lerobot_v3_video_backend
            if lerobot_tolerance_s is not None:
                dataset_kwargs["tolerances_s"] = {
                    ds_dir: float(lerobot_tolerance_s) for ds_dir in self.dataset_dirs
                }
        self.multi_dataset = dataset_cls(**dataset_kwargs)
        if hasattr(self.multi_dataset, "episode_data_index"):
            self.episode_data_index = self.multi_dataset.episode_data_index
        else:
            episode_data_index = []
            end_index = 0
            for dataset in self.multi_dataset._datasets:
                multi_episode_data_index = {
                    "from": dataset.episode_data_index["from"] + end_index,
                    "to": dataset.episode_data_index["to"] + end_index,
                }
                episode_data_index.append(multi_episode_data_index)
                end_index = multi_episode_data_index["to"][-1]
            self.episode_data_index = {
                "from": torch.cat([dataset["from"] for dataset in episode_data_index]),
                "to": torch.cat([dataset["to"] for dataset in episode_data_index]),
            }

    @staticmethod
    def _load_lerobot_meta_cache(path: Optional[str]) -> dict[str, dict[str, Any]]:
        if path is None or str(path).strip() == "":
            return _empty_lerobot_meta_cache()
        return _read_lerobot_meta_cache(path)

    def _filter_episode_indices(self, episode_indices: list[int], repo_id: str) -> list[int]:
        cfg = self.episode_index_filter
        if cfg is None:
            return episode_indices
        mode = str(cfg.get("mode", "")).strip().lower()
        if mode in {"", "none", "all"}:
            return episode_indices
        if mode in {"periodic_prefix", "periodic_first", "first_k_per_period"}:
            period = int(cfg["period"])
            keep_first = int(cfg["keep_first"])
            offset = int(cfg.get("offset", 0))
            if period <= 0:
                raise ValueError(f"episode_index_filter.period must be positive, got {period}")
            if keep_first < 0 or keep_first > period:
                raise ValueError(
                    f"episode_index_filter.keep_first must be in [0, period], got {keep_first} for period={period}"
                )
            return [
                ep_idx for ep_idx in episode_indices if (int(ep_idx) - offset) % period < keep_first
            ]
        raise ValueError(
            f"Unsupported episode_index_filter.mode={mode!r} for repo_id={repo_id!r}. Expected one of: periodic_prefix."
        )

    def _get_action(self, meta, lerobot_sample) -> torch.Tensor:
        key, lerobot_key, raw_shape = (meta["key"], meta["lerobot_key"], meta["raw_shape"])
        action: torch.Tensor = lerobot_sample[lerobot_key]
        if action.ndim == 1:
            action = action.unsqueeze(-1)
        assert action.shape[-1] == raw_shape, (
            f"Action '{key}' shape {action.shape[-1]} mismatch with meta {raw_shape}."
        )
        return action

    def _get_state(self, meta, lerobot_sample) -> torch.Tensor:
        key, lerobot_key, raw_shape = (meta["key"], meta["lerobot_key"], meta["raw_shape"])
        state: torch.Tensor = lerobot_sample[lerobot_key]
        if state.ndim == 1:
            state = state.unsqueeze(-1)
        assert state.shape[-1] == raw_shape, (
            f"State '{key}' shape {state.shape[-1]} mismatch with meta {raw_shape}."
        )
        return state

    def _get_image(self, meta, lerobot_sample) -> torch.Tensor:
        key, lerobot_key, raw_shape = (meta["key"], meta["lerobot_key"], meta["raw_shape"])
        image: torch.Tensor = lerobot_sample[lerobot_key]
        if image.ndim == 3:
            image = image.unsqueeze(0)
        image = (image * 255).to(torch.uint8)
        return image

    def _split_lerobot_sample(self, lerobot_sample) -> Dict[str, Any]:
        return lerobot_sample

    def _get_episode_data(self, episode_idx):
        lerobot_sample = self.multi_dataset.get_episode_data(episode_idx)
        lerobot_sample = self._split_lerobot_sample(lerobot_sample)
        state, action = ({}, {})
        for meta in self.state_meta:
            s = self._get_state(meta, lerobot_sample)
            state[meta["key"]] = s.unsqueeze(1).float()
        for meta in self.action_meta:
            a = self._get_action(meta, lerobot_sample)
            a = sliding_window_with_replication(a, self.action_size)
            action[meta["key"]] = a.float()
        result = {"action": action, "state": state}
        for key in ("action_dim_is_pad", "state_dim_is_pad", "embodiment"):
            if key in lerobot_sample:
                result[key] = lerobot_sample[key]
        return result

    def _set_return_images(self, flag: bool):
        self.return_images = flag
        self.multi_dataset.set_during_training(flag)

    def __len__(self):
        return (
            self.multi_dataset.num_frames + self.sample_index_stride - 1
        ) // self.sample_index_stride

    def _get_additional_data(self, sample, lerobot_sample):
        return sample

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds {len(self)}.")

        sample_idx = idx * self.sample_index_stride
        attempt = 0
        last_exception: Optional[Exception] = None
        while attempt < MAX_GETITEM_ATTEMPT:
            try:
                lerobot_sample = self.multi_dataset[sample_idx]
                lerobot_sample = self._split_lerobot_sample(lerobot_sample)
                break
            except Exception as err:
                attempt += 1
                last_exception = err
                logger.warning(
                    f"Error loading sample {sample_idx} (attempt {attempt}). Retrying with a random index. Error: {err}"
                )
                sample_idx = np.random.randint(len(self))
                print(traceback.format_exc())
        else:
            raise RuntimeError(
                f"Failed to load a valid sample after {MAX_GETITEM_ATTEMPT} attempts for index {idx}."
            ) from last_exception
        sample = {
            "idx": sample_idx,
            "task": lerobot_sample["task"],
            "action": {},
            "state": {},
            "images": {},
        }
        for meta in self.state_meta:
            sample["state"][meta["key"]] = self._get_state(meta, lerobot_sample)
        for meta in self.action_meta:
            sample["action"][meta["key"]] = self._get_action(meta, lerobot_sample)
        for meta in self.image_meta:
            sample["images"][meta["key"]] = self._get_image(meta, lerobot_sample)
        sample["action_is_pad"] = lerobot_sample[f"{self.action_meta[0]['lerobot_key']}_is_pad"]
        sample["state_is_pad"] = lerobot_sample[f"{self.state_meta[0]['lerobot_key']}_is_pad"]
        sample["image_is_pad"] = lerobot_sample[f"{self.image_meta[0]['lerobot_key']}_is_pad"]
        for key in ("action_dim_is_pad", "state_dim_is_pad", "embodiment"):
            if key in lerobot_sample:
                sample[key] = lerobot_sample[key]
        sample = self._get_additional_data(sample, lerobot_sample)
        for key in lerobot_sample:
            if key not in sample and "observation" not in key and ("action" not in key):
                sample[key] = lerobot_sample[key]
        if self.processor is not None:
            sample = self.processor.preprocess(sample)
        return sample

    def set_processor(self, processor: BaseProcessor):
        self.processor = processor
        if self.is_training_set:
            self.processor.train()
        else:
            self.processor.eval()
        return self

    def get_dataset_stats(self, preprocessor: BaseProcessor):
        if getattr(self.multi_dataset, "hetero_bridge", None) is not None:
            return self._get_per_embodiment_dataset_stats(preprocessor)
        state_min = DefaultDict(list)
        state_max = DefaultDict(list)
        state_mean = DefaultDict(list)
        state_var = DefaultDict(list)
        state_q01 = DefaultDict(list)
        state_q99 = DefaultDict(list)
        action_min = DefaultDict(list)
        action_max = DefaultDict(list)
        action_mean = DefaultDict(list)
        action_var = DefaultDict(list)
        action_q01 = DefaultDict(list)
        action_q99 = DefaultDict(list)
        episodes_num = self.multi_dataset.num_episodes

        def process_episode(episode_idx):
            batch = self._get_episode_data(episode_idx)
            batch = preprocessor.action_state_transform(batch)
            return batch

        multi_thread = True
        if not multi_thread:
            for episode_idx in tqdm(
                range(episodes_num), desc="Iterating dataset to get normalization"
            ):
                batch = process_episode(episode_idx)
                for meta in self.state_meta:
                    key = meta["key"]
                    cur_state: torch.Tensor = batch["state"][key]
                    state_min[key].append(cur_state.amin(0))
                    state_max[key].append(cur_state.amax(0))
                    state_mean[key].append(cur_state.mean(0))
                    state_var[key].append(cur_state.var(0))
                    state_q01[key].append(torch.quantile(cur_state, 0.01, dim=0, keepdim=False))
                    state_q99[key].append(torch.quantile(cur_state, 0.99, dim=0, keepdim=False))
                for meta in self.action_meta:
                    key = meta["key"]
                    cur_action: torch.Tensor = batch["action"][key]
                    action_min[key].append(cur_action.amin(0))
                    action_max[key].append(cur_action.amax(0))
                    action_mean[key].append(cur_action.mean(0))
                    action_var[key].append(cur_action.var(0))
                    action_q01[key].append(torch.quantile(cur_action, 0.01, dim=0, keepdim=False))
                    action_q99[key].append(torch.quantile(cur_action, 0.99, dim=0, keepdim=False))
        else:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_episode, num) for num in range(episodes_num)]
                for future in tqdm(
                    as_completed(futures),
                    total=episodes_num,
                    desc="Iterating dataset to get normalization",
                ):
                    try:
                        batch = future.result()
                        for meta in self.state_meta:
                            key = meta["key"]
                            cur_state: torch.Tensor = batch["state"][key]
                            state_min[key].append(cur_state.amin(0))
                            state_max[key].append(cur_state.amax(0))
                            state_mean[key].append(cur_state.mean(0))
                            state_var[key].append(cur_state.var(0))
                            state_q01[key].append(
                                torch.quantile(cur_state, 0.01, dim=0, keepdim=False)
                            )
                            state_q99[key].append(
                                torch.quantile(cur_state, 0.99, dim=0, keepdim=False)
                            )
                        for meta in self.action_meta:
                            key = meta["key"]
                            cur_action: torch.Tensor = batch["action"][key]
                            action_min[key].append(cur_action.amin(0))
                            action_max[key].append(cur_action.amax(0))
                            action_mean[key].append(cur_action.mean(0))
                            action_var[key].append(cur_action.var(0))
                            action_q01[key].append(
                                torch.quantile(cur_action, 0.01, dim=0, keepdim=False)
                            )
                            action_q99[key].append(
                                torch.quantile(cur_action, 0.99, dim=0, keepdim=False)
                            )
                    except Exception as e:
                        logger.error(f"Error processing episode: {e}")
                        print(traceback.format_exc())
                        raise e

        def get_mean_std(means, vars):
            means = torch.stack(means)
            vars = torch.stack(vars)
            stepwise_mean = means.mean(0)
            stepwise_std = (vars + (means - stepwise_mean) ** 2).mean(0).sqrt()
            global_mean = means.mean((0, 1))
            global_std = (vars + (means - global_mean) ** 2).mean((0, 1)).sqrt()
            return (stepwise_mean, stepwise_std, global_mean, global_std)

        stats = {
            "state": DefaultDict(dict),
            "action": DefaultDict(dict),
            "num_episodes": episodes_num,
            "num_transition": self.multi_dataset.num_frames,
        }
        for meta in self.state_meta:
            key = meta["key"]
            stats["state"][key]["stepwise_min"] = torch.stack(state_min[key]).amin(0)
            stats["state"][key]["stepwise_max"] = torch.stack(state_max[key]).amax(0)
            stats["state"][key]["global_min"] = stats["state"][key]["stepwise_min"].amin(0)
            stats["state"][key]["global_max"] = stats["state"][key]["stepwise_max"].amax(0)
            stats["state"][key]["stepwise_q01"] = torch.stack(state_q01[key]).amin(0)
            stats["state"][key]["stepwise_q99"] = torch.stack(state_q99[key]).amax(0)
            stats["state"][key]["global_q01"] = stats["state"][key]["stepwise_q01"].amin(0)
            stats["state"][key]["global_q99"] = stats["state"][key]["stepwise_q99"].amax(0)
            (
                stats["state"][key]["stepwise_mean"],
                stats["state"][key]["stepwise_std"],
                stats["state"][key]["global_mean"],
                stats["state"][key]["global_std"],
            ) = get_mean_std(state_mean[key], state_var[key])
        for meta in self.action_meta:
            key = meta["key"]
            stats["action"][key]["stepwise_min"] = torch.stack(action_min[key]).amin(0)
            stats["action"][key]["stepwise_max"] = torch.stack(action_max[key]).amax(0)
            stats["action"][key]["global_min"] = stats["action"][key]["stepwise_min"].amin(0)
            stats["action"][key]["global_max"] = stats["action"][key]["stepwise_max"].amax(0)
            stats["action"][key]["stepwise_q01"] = torch.stack(action_q01[key]).amin(0)
            stats["action"][key]["stepwise_q99"] = torch.stack(action_q99[key]).amax(0)
            stats["action"][key]["global_q01"] = stats["action"][key]["stepwise_q01"].amin(0)
            stats["action"][key]["global_q99"] = stats["action"][key]["stepwise_q99"].amax(0)
            (
                stats["action"][key]["stepwise_mean"],
                stats["action"][key]["stepwise_std"],
                stats["action"][key]["global_mean"],
                stats["action"][key]["global_std"],
            ) = get_mean_std(action_mean[key], action_var[key])
        return stats

    @staticmethod
    def _as_dim_mask(mask, dim: int) -> torch.BoolTensor:
        if mask is None:
            return torch.zeros(dim, dtype=torch.bool)
        mask = torch.as_tensor(mask, dtype=torch.bool)
        if mask.ndim != 1 or mask.shape[0] != dim:
            raise ValueError(
                f"Dimension mask must be 1D with length {dim}, got {tuple(mask.shape)}"
            )
        return mask

    @staticmethod
    def _masked_episode_stats(
        x: torch.Tensor, dim_mask: torch.BoolTensor
    ) -> dict[str, torch.Tensor]:
        _, steps, dim = x.shape
        dim_mask = dim_mask.to(device=x.device)
        valid = ~dim_mask
        step_template = torch.zeros((steps, dim), dtype=x.dtype, device=x.device)
        global_template = torch.zeros((dim,), dtype=x.dtype, device=x.device)
        out = {
            "stepwise_min": step_template.clone(),
            "stepwise_max": step_template.clone(),
            "stepwise_mean": step_template.clone(),
            "stepwise_var": step_template.clone(),
            "stepwise_q01": step_template.clone(),
            "stepwise_q99": step_template.clone(),
            "global_min": global_template.clone(),
            "global_max": global_template.clone(),
            "global_mean": global_template.clone(),
            "global_var": global_template.clone(),
            "global_q01": global_template.clone(),
            "global_q99": global_template.clone(),
        }
        if not bool(valid.any().item()):
            return out
        xv = x[..., valid]
        out["stepwise_min"][:, valid] = xv.amin(0)
        out["stepwise_max"][:, valid] = xv.amax(0)
        out["stepwise_mean"][:, valid] = xv.mean(0)
        out["stepwise_var"][:, valid] = xv.var(0)
        out["stepwise_q01"][:, valid] = torch.quantile(xv, 0.01, dim=0, keepdim=False)
        out["stepwise_q99"][:, valid] = torch.quantile(xv, 0.99, dim=0, keepdim=False)
        out["global_min"][valid] = xv.amin((0, 1))
        out["global_max"][valid] = xv.amax((0, 1))
        out["global_mean"][valid] = xv.mean((0, 1))
        out["global_var"][valid] = xv.var((0, 1))
        out["global_q01"][valid] = torch.quantile(
            xv.reshape(-1, xv.shape[-1]), 0.01, dim=0, keepdim=False
        )
        out["global_q99"][valid] = torch.quantile(
            xv.reshape(-1, xv.shape[-1]), 0.99, dim=0, keepdim=False
        )
        return out

    @staticmethod
    def _merge_episode_stats(stat_lists: dict[str, list[torch.Tensor]]) -> dict[str, torch.Tensor]:
        means = torch.stack(stat_lists["global_mean"])
        vars_ = torch.stack(stat_lists["global_var"])
        global_mean = means.mean(0)
        global_std = (vars_ + (means - global_mean) ** 2).mean(0).sqrt()
        step_means = torch.stack(stat_lists["stepwise_mean"])
        step_vars = torch.stack(stat_lists["stepwise_var"])
        stepwise_mean = step_means.mean(0)
        stepwise_std = (step_vars + (step_means - stepwise_mean) ** 2).mean(0).sqrt()
        return {
            "stepwise_min": torch.stack(stat_lists["stepwise_min"]).amin(0),
            "stepwise_max": torch.stack(stat_lists["stepwise_max"]).amax(0),
            "stepwise_mean": stepwise_mean,
            "stepwise_std": stepwise_std,
            "stepwise_q01": torch.stack(stat_lists["stepwise_q01"]).amin(0),
            "stepwise_q99": torch.stack(stat_lists["stepwise_q99"]).amax(0),
            "global_min": torch.stack(stat_lists["global_min"]).amin(0),
            "global_max": torch.stack(stat_lists["global_max"]).amax(0),
            "global_mean": global_mean,
            "global_std": global_std,
            "global_q01": torch.stack(stat_lists["global_q01"]).amin(0),
            "global_q99": torch.stack(stat_lists["global_q99"]).amax(0),
        }

    def _get_per_embodiment_dataset_stats(self, preprocessor: BaseProcessor):
        by_embodiment: dict[str, dict[str, dict[str, DefaultDict[str, list]]]] = {}
        episodes_num = self.multi_dataset.num_episodes

        def process_episode(episode_idx):
            batch = self._get_episode_data(episode_idx)
            batch = preprocessor.action_state_transform(batch)
            return batch

        for episode_idx in tqdm(
            range(episodes_num), desc="Iterating dataset to get per-embodiment normalization"
        ):
            batch = process_episode(episode_idx)
            embodiment = str(batch.get("embodiment", "default"))
            emb_store = by_embodiment.setdefault(
                embodiment,
                {
                    "state": DefaultDict(lambda: DefaultDict(list)),
                    "action": DefaultDict(lambda: DefaultDict(list)),
                },
            )
            for meta in self.state_meta:
                key = meta["key"]
                cur_state: torch.Tensor = batch["state"][key]
                dim_mask = self._as_dim_mask(batch.get("state_dim_is_pad"), cur_state.shape[-1])
                cur_stats = self._masked_episode_stats(cur_state, dim_mask)
                for stat_name, stat_value in cur_stats.items():
                    emb_store["state"][key][stat_name].append(stat_value)
            for meta in self.action_meta:
                key = meta["key"]
                cur_action: torch.Tensor = batch["action"][key]
                dim_mask = self._as_dim_mask(batch.get("action_dim_is_pad"), cur_action.shape[-1])
                cur_stats = self._masked_episode_stats(cur_action, dim_mask)
                for stat_name, stat_value in cur_stats.items():
                    emb_store["action"][key][stat_name].append(stat_value)
        stats = {
            "type": "per_embodiment",
            "num_episodes": episodes_num,
            "num_transition": self.multi_dataset.num_frames,
            "embodiments": {},
        }
        for embodiment, emb_store in by_embodiment.items():
            stats["embodiments"][embodiment] = {
                "state": DefaultDict(dict),
                "action": DefaultDict(dict),
            }
            for field in ("state", "action"):
                for key, stat_lists in emb_store[field].items():
                    stats["embodiments"][embodiment][field][key] = self._merge_episode_stats(
                        stat_lists
                    )
        return stats


def sliding_window_with_replication(x: torch.Tensor, window_size: int) -> torch.Tensor:
    assert x.dim() == 2
    assert window_size > 0
    N, D = x.shape
    i_indices = torch.arange(N).unsqueeze(1)
    j_indices = torch.arange(window_size).unsqueeze(0)
    indices = i_indices + j_indices
    clamped_indices = torch.clamp(indices, min=0, max=N - 1)
    out = x[clamped_indices]
    return out
