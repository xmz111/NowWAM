"""DIAL GR1 adapter. Its image, action and success conventions are not LIBERO's."""

import hashlib
import random

import numpy as np
from PIL import Image

TASKS = (
    "PnPPotatoToMicrowaveClose",
    "PnPMilkToMicrowaveClose",
    "PnPCanToDrawerClose",
    "PnPCupToDrawerClose",
    "PnPBottleToCabinetClose",
    "PnPWineToCabinetClose",
    "PosttrainPnPNovelFromPlacematToBowlSplitA",
    "PosttrainPnPNovelFromPlateToPlateSplitA",
    "PosttrainPnPNovelFromPlacematToPlateSplitA",
    "PosttrainPnPNovelFromCuttingboardToPotSplitA",
    "PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA",
    "PosttrainPnPNovelFromCuttingboardToPanSplitA",
    "PosttrainPnPNovelFromTrayToCardboardboxSplitA",
    "PosttrainPnPNovelFromTrayToTieredshelfSplitA",
    "PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA",
    "PosttrainPnPNovelFromPlacematToTieredshelfSplitA",
    "PosttrainPnPNovelFromPlateToCardboardboxSplitA",
    "PosttrainPnPNovelFromPlacematToBasketSplitA",
    "PosttrainPnPNovelFromPlateToPanSplitA",
    "PosttrainPnPNovelFromTrayToTieredbasketSplitA",
    "PosttrainPnPNovelFromTrayToPotSplitA",
    "PosttrainPnPNovelFromPlateToBowlSplitA",
    "PosttrainPnPNovelFromCuttingboardToBasketSplitA",
    "PosttrainPnPNovelFromTrayToPlateSplitA",
)
STATE_KEYS = (
    "state.wrist_r_pos",
    "state.wrist_r_rot6d",
    "state.wrist_l_pos",
    "state.wrist_l_rot6d",
    "state.right_arm",
    "state.right_hand",
    "state.left_arm",
    "state.left_hand",
    "state.waist",
)
ACTION_KEYS = (
    "action.wrist_r_pos",
    "action.wrist_r_rot6d",
    "action.wrist_l_pos",
    "action.wrist_l_rot6d",
    "action.right_arm",
    "action.right_hand",
    "action.left_arm",
    "action.left_hand",
    "action.waist",
)
ACTION_EDGES = (0, 3, 9, 12, 18, 25, 31, 38, 44, 47)
CAMERA = "video.ego_view_bg_crop_pad_res256_freq20"


def state76(obs):
    wrist = np.concatenate([np.asarray(obs[k], np.float32).ravel() for k in STATE_KEYS[:4]])
    joints = np.concatenate([np.asarray(obs[k], np.float32).ravel() for k in STATE_KEYS[4:]])
    state = np.concatenate([wrist, np.sin(joints), np.cos(joints)]).astype(np.float32)
    if state.shape != (76,):
        raise ValueError(f"Expected state76, got {state.shape}")
    return state


def image224(obs):
    image = np.asarray(obs[CAMERA], np.uint8)
    if image.ndim == 4:
        image = image[-1]
    image = image[43:211, :256]
    height, width = int(168 * 0.95), int(256 * 0.95)
    top, left = (168 - height) // 2, (256 - width) // 2
    return np.asarray(
        Image.fromarray(image[top : top + height, left : left + width]).resize(
            (224, 224), Image.Resampling.BILINEAR
        ),
        np.uint8,
    )


def action_dict(action):
    action = np.asarray(action, np.float32)
    if action.shape != (16, 47) or not np.isfinite(action).all():
        raise ValueError("Expected finite action chunk [16,47]")
    return {
        key: action[:, start:end]
        for key, start, end in zip(ACTION_KEYS, ACTION_EDGES, ACTION_EDGES[1:])
    }


def rollout(case, predict):
    import gymnasium as gym
    import robocasa  # noqa: F401 -- registers the benchmark environments
    from gr00t.eval.wrappers.multistep_wrapper import MultiStepWrapper
    from gr00t.eval.wrappers.video_recording_wrapper import VideoRecorder, VideoRecordingWrapper
    from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401

    seed = case["seed"]
    random.seed(seed)
    np.random.seed(seed)
    env = gym.make(
        f"gr1_unified/{case['task']}_GR1ArmsAndWaistFourierHands_Env", enable_render=True, seed=seed
    )
    try:
        raw = env.unwrapped
        for target in (raw, getattr(raw, "env", None)):
            if target is not None:
                if hasattr(target, "rng"):
                    target.rng = np.random.default_rng(seed)
                if hasattr(target, "seed"):
                    target.seed = seed
        env = VideoRecordingWrapper(
            env,
            VideoRecorder.create_h264(fps=10, thread_type="FRAME", thread_count=1),
            video_dir=None,
            state_modality_keys=set(STATE_KEYS),
        )
        env = MultiStepWrapper(
            env,
            video_delta_indices=np.asarray([0]),
            state_delta_indices=np.asarray([0]),
            n_action_steps=16,
            max_episode_steps=720,
            state_modality_keys=set(STATE_KEYS),
        )
        obs, _ = env.reset(seed=seed)
        initial_hash = hashlib.sha256(
            np.asarray(obs[CAMERA], np.uint8).tobytes() + state76(obs).tobytes()
        ).hexdigest()
        success = False
        while len(env.reward) < 720:
            instruction = str(obs["annotation.human.coarse_action"])
            for prefix in ("locked_waist:", "unlocked_waist:"):
                if instruction.startswith(prefix):
                    instruction = instruction[len(prefix) :].strip()
            action = predict(image224(obs), state76(obs), instruction, len(env.reward))
            obs, _, terminated, truncated, info = env.step(action_dict(action))
            success = bool(np.asarray(info.get("success", [False])).reshape(-1)[-1])
            if success or bool(terminated) or bool(truncated):
                break
        return {"success": success, "steps": len(env.reward), "initial_hash": initial_hash}
    finally:
        env.close()
