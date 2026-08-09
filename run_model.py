#!/usr/bin/env python3
"""Run a trained MaskablePPO model to build coasters in a live OpenRCT2 instance.

Rebuilds the same wrapper chain used in training (curriculum + ActionMasker + the
custom feature extractor's observation space) and loads the matching VecNormalize
stats, with running statistics frozen for inference. Requires an OpenRCT2 API server
on the given --port (a GUI instance with the plugin works -- that's the showcase mode).

Showcase example (watchable speed, current-era checkpoint):
    python run_model.py --model checkpoints_showcase/<ckpt>_steps.zip \
        --port 8080 --episodes 3 --game-speed 1

Ask for a specific coaster shape (Task 10 fix 1):
    python run_model.py --model checkpoints_showcase/<ckpt>_steps.zip \
        --port 8080 --family 3 --episodes 1
"""
import argparse
import os

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from openrct2_gym.envs import footprint
from openrct2_gym.envs.improved_phased_curriculum_wrapper import (
    ImprovedPhasedCurriculumWrapper,
)
from train import create_curriculum_masked_env, _vecnormalize_path


def make_inference_env(port, start_phase=6, game_speed=1, verbose=1):
    """The training-equivalent env for inference: COLD (no warm-start scaffolding --
    the env would otherwise pre-build most of the loop before the model acts) and at
    the checkpoint's phase (a mature P6 policy in Phase 1 truncates at 40 pieces)."""
    return create_curriculum_masked_env(
        port, verbose=verbose, warm_start_enabled=False,
        game_speed=game_speed, start_phase=start_phase)


def find_curriculum_wrapper(env):
    """Walk the wrapper chain from the OUTERMOST wrapper (what make_inference_env
    returns) down to the ImprovedPhasedCurriculumWrapper instance -- the same downward
    walk ImprovedPhasedCurriculumWrapper._get_base_env uses to reach the raw OpenRCT2Env
    a few layers further down (improved_phased_curriculum_wrapper.py). None if the
    chain has no such wrapper (defensive; every env make_inference_env builds does)."""
    while env is not None and not isinstance(env, ImprovedPhasedCurriculumWrapper):
        env = getattr(env, "env", None)
    return env


def pin_family(wrapper, z):
    """Pin every future reset() on `wrapper` to footprint family `z` (Task 10 fix 1).

    ImprovedPhasedCurriculumWrapper.reset() draws base_env.target_family fresh from
    self._sample_target_family() on EVERY episode (see reset(), which comments that
    target_family is deliberately not reset by the base env itself), so writing
    target_family directly from here would be silently clobbered on the very next
    reset. Replacing the WRAPPER INSTANCE's sampler is an inference-only override: it
    never touches the class method or PHASE_FAMILIES, so no training code path is
    affected, and it holds across every subsequent reset() on this instance.

    Out-of-range z is rejected here rather than left to fail downstream: family_match()
    indexes footprint.FAMILIES directly by z and would raise an unhelpful IndexError
    deep inside the reward calculation instead of a clear CLI error.
    """
    if not 0 <= z < footprint.FAMILY_N:
        names = ", ".join(f"{i}={name}" for i, (name, *_rest) in enumerate(footprint.FAMILIES))
        raise SystemExit(
            f"--family {z} out of range; must be an int in [0, {footprint.FAMILY_N - 1}] "
            f"({names})")
    wrapper._sample_target_family = lambda: z


def track_recorder(wrapper, base_env):
    """Install an instance-level override (same technique as pin_family) on
    `wrapper.step` that snapshots the just-finished episode's action history the
    instant its terminal step() call returns.

    This is needed because SB3's DummyVecEnv auto-resets the underlying env as soon as
    step() returns to the caller (step_wait() calls .reset() on any done env before
    returning), which clears OpenRCT2Env.track_builder.history -- by the time
    run_model.py's/build_variety_gallery.py's loop observes done=True the raw history
    of the track that was just built is already gone. Capturing it here, inside the
    still-running terminal step() call (before that later, separate reset() call),
    is the only point where it is both complete and still live.

    Returns a dict the caller reads after the episode ends: {'actions': [...]}.
    """
    orig_step = wrapper.step
    state = {"actions": []}

    def step_and_record(action):
        result = orig_step(action)
        _obs, _reward, terminated, truncated, _info = result
        if terminated or truncated:
            state["actions"] = [h.get("action") for h in base_env.track_builder.history]
        return result

    wrapper.step = step_and_record
    return state


def run_episode(env, model, deterministic=True):
    """Run one episode on `env` (a VecEnv) to completion with `model`. Returns
    (steps, total_reward, info) for the episode's terminal step -- shared by the
    showcase loop below and build_variety_gallery.py's per-family capture."""
    obs = env.reset()
    done = np.array([False])
    total_reward = 0.0
    steps = 0
    info = [{}]
    while not done[0]:
        action_masks = get_action_masks(env)
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=deterministic)
        obs, reward, done, info = env.step(action)
        total_reward += float(reward[0])
        steps += 1
    return steps, total_reward, info[0]


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
    parser.add_argument("--family", type=int, default=None,
                        help="Pin every episode's footprint-family seed instead of "
                             "letting the curriculum sample it (int in "
                             f"[0, {footprint.FAMILY_N - 1}]; see footprint.FAMILIES "
                             "for the name/turn/switch bands). Default: sample as in "
                             "training.")
    return parser.parse_args(argv)


def main():
    args = parse_args()

    single_env = make_inference_env(args.port, start_phase=args.start_phase,
                                    game_speed=args.game_speed)

    recorder = None
    if args.family is not None:
        wrapper = find_curriculum_wrapper(single_env)
        if wrapper is None:
            raise SystemExit("could not find ImprovedPhasedCurriculumWrapper in the "
                              "env chain -- --family has nothing to pin")
        pin_family(wrapper, args.family)
        recorder = track_recorder(wrapper, wrapper._get_base_env())

    env = DummyVecEnv([lambda: single_env])

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
        steps, total_reward, info = run_episode(env, model, deterministic=not args.sample)
        rating = info.get("ride_rating") or {}
        banner = (f"Episode {ep + 1}: steps={steps}, total reward={total_reward:.2f}, "
                  f"loop_completed={info.get('loop_completed')}, "
                  f"E={rating.get('excitement', 0):.2f} I={rating.get('intensity', 0):.2f} "
                  f"N={rating.get('nausea', 0):.2f}")
        if args.family is not None:
            # What a human watching should compare against TensorBoard: the seed we
            # pinned versus classify_family() of the track the policy actually placed.
            built = footprint.classify_family(recorder["actions"])
            requested_name = footprint.FAMILIES[args.family][0]
            built_name = footprint.FAMILIES[built][0] if built is not None else "none"
            hit = "HIT" if built == args.family else "MISS"
            banner += f", requested={requested_name} built={built_name} ({hit})"
        print(banner)

    env.close()


if __name__ == "__main__":
    main()
