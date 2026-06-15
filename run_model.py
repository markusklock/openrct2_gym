#!/usr/bin/env python3
"""Run a trained MaskablePPO model to build a coaster in a live OpenRCT2 instance.

Rebuilds the same wrapper chain used in training (curriculum + ActionMasker + the
custom feature extractor's observation space) and loads the matching VecNormalize
stats, with running statistics frozen for inference. Requires an OpenRCT2 API server
on the given --port.
"""
import argparse
import os

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from train import create_curriculum_masked_env, _vecnormalize_path


def main():
    parser = argparse.ArgumentParser(description="Run a trained coaster-building model")
    parser.add_argument("--model", default="logs_parallel_curriculum_masked_1envs/final_model",
                        help="Path to the saved MaskablePPO model (with or without .zip)")
    parser.add_argument("--port", type=int, default=8080,
                        help="OpenRCT2 API server port")
    parser.add_argument("--vecnormalize", default=None,
                        help="Path to VecNormalize stats (.pkl). Defaults to the model's sibling file.")
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args()

    env = DummyVecEnv([
        lambda: create_curriculum_masked_env(args.port, verbose=1)
    ])

    stats_path = args.vecnormalize or _vecnormalize_path(args.model)
    if os.path.exists(stats_path):
        print(f"Loading VecNormalize stats from {stats_path}")
        env = VecNormalize.load(stats_path, env)
    else:
        print(f"⚠️ VecNormalize stats not found at {stats_path}; running without learned "
              f"obs normalization (results may be degraded).")
        env = VecNormalize(env, norm_obs=True, norm_reward=False, norm_obs_keys=["scalars"])

    # Inference: freeze running stats and never normalize reward (set on the actual instance).
    env.training = False
    env.norm_reward = False

    model = MaskablePPO.load(args.model, env=env)

    for ep in range(args.episodes):
        obs = env.reset()
        done = np.array([False])
        total_reward = 0.0
        steps = 0
        while not done[0]:
            action_masks = get_action_masks(env)
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += float(reward[0])
            steps += 1
        print(f"Episode {ep + 1}: steps={steps}, total reward={total_reward:.2f}, "
              f"loop_completed={info[0].get('loop_completed')}")

    env.close()


if __name__ == "__main__":
    main()
