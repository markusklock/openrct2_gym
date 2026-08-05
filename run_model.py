#!/usr/bin/env python3
"""Run a trained MaskablePPO model to build coasters in a live OpenRCT2 instance.

Rebuilds the same wrapper chain used in training (curriculum + ActionMasker + the
custom feature extractor's observation space) and loads the matching VecNormalize
stats, with running statistics frozen for inference. Requires an OpenRCT2 API server
on the given --port (a GUI instance with the plugin works -- that's the showcase mode).

Showcase example (watchable speed, current-era checkpoint):
    python run_model.py --model checkpoints_showcase/<ckpt>_steps.zip \
        --port 8080 --episodes 3 --game-speed 1
"""
import argparse
import os

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from train import create_curriculum_masked_env, _vecnormalize_path


def make_inference_env(port, start_phase=6, game_speed=1, verbose=1):
    """The training-equivalent env for inference: COLD (no warm-start scaffolding --
    the env would otherwise pre-build most of the loop before the model acts) and at
    the checkpoint's phase (a mature P6 policy in Phase 1 truncates at 40 pieces)."""
    return create_curriculum_masked_env(
        port, verbose=verbose, warm_start_enabled=False,
        game_speed=game_speed, start_phase=start_phase)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run a trained coaster-building model")
    parser.add_argument("--model", default="logs_parallel_curriculum_masked_1envs/final_model",
                        help="Path to the saved MaskablePPO model (with or without .zip)")
    parser.add_argument("--port", type=int, default=8080,
                        help="OpenRCT2 API server port")
    parser.add_argument("--vecnormalize", default=None,
                        help="Path to VecNormalize stats (.pkl). Defaults to the model's sibling file.")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--start-phase", type=int, default=6,
                        help="Curriculum phase for the env (default 6 -- every current "
                             "checkpoint is a mature P6 policy; Phase 1's 40-piece "
                             "budget would truncate its builds mid-loop)")
    parser.add_argument("--game-speed", type=int, default=1,
                        help="Game speed during the demo (1 = watchable building; "
                             "8 = fast ride tests like training)")
    parser.add_argument("--sample", action="store_true",
                        help="Sample actions instead of deterministic argmax "
                             "(shows build variety across episodes)")
    return parser.parse_args(argv)


def main():
    args = parse_args()

    env = DummyVecEnv([
        lambda: make_inference_env(args.port, start_phase=args.start_phase,
                                   game_speed=args.game_speed)
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
            action, _ = model.predict(obs, action_masks=action_masks,
                                      deterministic=not args.sample)
            obs, reward, done, info = env.step(action)
            total_reward += float(reward[0])
            steps += 1
        rating = info[0].get("ride_rating") or {}
        print(f"Episode {ep + 1}: steps={steps}, total reward={total_reward:.2f}, "
              f"loop_completed={info[0].get('loop_completed')}, "
              f"E={rating.get('excitement', 0):.2f} I={rating.get('intensity', 0):.2f} "
              f"N={rating.get('nausea', 0):.2f}")

    env.close()


if __name__ == "__main__":
    main()
