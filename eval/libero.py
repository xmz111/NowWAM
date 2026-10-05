"""LIBERO-Plus rollout; model inference runs in a separate process."""

import hashlib
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image

from nowwam.protocol import LIBERO_SUITE_COUNTS


def observation(obs):
    views = []
    for key in ("agentview_image", "robot0_eye_in_hand_image"):
        image = Image.fromarray(np.ascontiguousarray(obs[key][::-1, ::-1]))
        scale = max(224 / image.width, 224 / image.height)
        image = image.resize(
            (round(image.width * scale), round(image.height * scale)), Image.Resampling.BILINEAR
        )
        left, top = (image.width - 224) // 2, (image.height - 224) // 2
        views.append(np.asarray(image.crop((left, top, left + 224, top + 224))))
    quat = np.asarray(obs["robot0_eef_quat"]).copy()
    quat[3] = np.clip(quat[3], -1, 1)
    den = np.sqrt(1 - quat[3] * quat[3])
    angle = np.zeros(3) if math.isclose(den, 0) else quat[:3] * 2 * math.acos(quat[3]) / den
    state = np.concatenate([obs["robot0_eef_pos"], angle, obs["robot0_gripper_qpos"]]).astype(
        np.float32
    )
    return np.concatenate(views, axis=1), state


def rollout(case, predict, *, fast_render=True):
    import torch
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    suite = benchmark.get_benchmark_dict()[case["suite"]]()
    if suite.n_tasks != LIBERO_SUITE_COUNTS[case["suite"]]:
        raise ValueError("Wrong LIBERO suite: expected the complete Plus task set")
    task = suite.get_task(case["task_idx"])
    initial_state = suite.get_task_init_states(case["task_idx"])[0]
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    try:
        env.seed(0)
        env.reset()
        obs = env.set_init_state(initial_state)
        initial_hash = hashlib.sha256(
            np.asarray(env.sim.get_state().flatten()).tobytes()
        ).hexdigest()
        # Sensor tasks corrupt RGB inside step(); their camera path must stay enabled.
        sparse = fast_render and int(getattr(env, "noise", 0)) == 0
        if sparse:
            cameras = {name for name in env.env.enabled_observables if name.endswith("_image")}
            if not {"agentview_image", "robot0_eye_in_hand_image"} <= cameras:
                raise RuntimeError("Expected camera observables are missing")
            for name in cameras:
                env.env.modify_observable(name, "enabled", False)
        for _ in range(30):
            obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
        pending = []
        limit = 700 if case["suite"] == "libero_10" else 400
        for step in range(limit):
            if not pending:
                if sparse:
                    obs = dict(obs)
                    for camera in ("agentview", "robot0_eye_in_hand"):
                        obs[camera + "_image"] = env.sim.render(
                            height=256, width=256, camera_name=camera
                        )
                pixels, state = observation(obs)
                action = np.asarray(predict(pixels, state, task.language, step), dtype=np.float32)
                if action.shape != (16, 7) or not np.isfinite(action).all():
                    raise ValueError("Expected finite action chunk [16,7]")
                pending = action[:12].tolist()
            obs, _, done, _ = env.step(pending.pop(0))
            if done:
                return {"success": True, "steps": step + 1, "initial_hash": initial_hash}
        return {"success": False, "steps": limit, "initial_hash": initial_hash}
    finally:
        env.close()
