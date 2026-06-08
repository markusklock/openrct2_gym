#!/usr/bin/env python3
"""
Parallel training script with curriculum learning AND proper action masking using MaskablePPO
Trains on multiple OpenRCT2 instances simultaneously for faster learning
"""
import gymnasium as gym
import openrct2_gym
from openrct2_gym.envs.curriculum_wrapper import CurriculumWrapper, AdaptiveCurriculumWrapper
from openrct2_gym.envs.improved_phased_curriculum_wrapper import ImprovedPhasedCurriculumWrapper
from openrct2_gym.envs.wrappers import OpenRCT2Wrapper
from openrct2_gym.envs.feature_extractor import BuildHistoryExtractor
from openrct2_gym.envs.openrct2_env import RewardParams
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
import numpy as np
import os
import argparse
import time
import math
from typing import List, Callable
from contextlib import ExitStack

# Single source of truth for the discount factor. PBRS policy-invariance requires the
# shaping potential to discount with the SAME gamma as PPO, so the reward's
# RewardParams.gamma and the model's gamma are tied to this one constant.
GAMMA = RewardParams().gamma


def _vecnormalize_path(model_path: str) -> str:
    """Sibling path for a model's VecNormalize stats (``X.zip`` -> ``X_vecnormalize.pkl``)."""
    if model_path.endswith(".zip"):
        model_path = model_path[:-4]
    return model_path + "_vecnormalize.pkl"


def _unwrap_to_vecenv_with_envs(env):
    """Return the underlying vec env exposing ``.envs`` (e.g. DummyVecEnv), or None.

    VecNormalize wraps the vec env in ``.venv``; access ``.envs`` explicitly rather than
    relying on attribute forwarding. SubprocVecEnv has no ``.envs`` -> returns None.
    """
    base = env
    for _ in range(8):
        if base is None:
            break
        if hasattr(base, "envs"):
            return base
        base = getattr(base, "venv", None)
    return None


class SaveVecNormalizeCallback(BaseCallback):
    """Persist VecNormalize running stats next to each model checkpoint.

    Without matching stats a saved checkpoint cannot be correctly reloaded for eval or
    resume (the obs normalization would be wrong). Filenames mirror CheckpointCallback's.
    """

    def __init__(self, save_freq: int, save_path: str, name_prefix: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        if self.save_freq > 0 and self.n_calls % self.save_freq == 0:
            vec_env = self.model.get_vec_normalize_env()
            if vec_env is not None:
                path = os.path.join(
                    self.save_path,
                    f"{self.name_prefix}_{self.num_timesteps}_steps_vecnormalize.pkl",
                )
                vec_env.save(path)
        return True


class ParallelCurriculumMaskableCallback(BaseCallback):
    """
    Tensorboard callback that tracks both curriculum and masking metrics
    across multiple parallel environments
    """
    def __init__(self, n_envs=1, verbose=0, training_verbose=0):
        super().__init__(verbose)
        self.n_envs = n_envs
        self.training_verbose = training_verbose  # Store the training verbosity level
        self.episode_counts = [0] * n_envs
        self.loop_completed_counts = [0] * n_envs
        self.total_episode_count = 0
        self.total_loop_completed = 0
        self.invalid_action_count = 0
        self.total_actions = 0
        self.start_time = time.time()
        self.total_steps = 0
        self.last_dashboard_episode = 0  # Track last dashboard print to avoid repeats
        
    def _on_step(self) -> bool:
        # Track total steps for throughput calculation
        self.total_steps += self.n_envs
        
        # Get environment through vectorized wrapper
        env = self.model.get_env()

        # Retrieve metrics from each subprocess using VecEnv helper methods
        try:
            track_lengths = env.get_attr('track_length')
            if track_lengths and track_lengths[0] is not None:
                # Log only from first env to reduce clutter
                self.logger.record('metrics/track_length', track_lengths[0])
        except (AttributeError, NotImplementedError):
            if self.verbose:
                print("Warning: track_length attribute not available in environments")

        try:
            distances = env.env_method('_calculate_distance_to_start')
            if distances and distances[0] is not None:
                dist = distances[0]
                # Method may return tuple/list/array; extract numeric distance
                if isinstance(dist, (list, tuple, np.ndarray)):
                    dist = dist[0]
                self.logger.record('metrics/current_distance', dist)
        except (AttributeError, NotImplementedError):
            if self.verbose:
                print("Warning: _calculate_distance_to_start method not available in environments")

        try:
            collision_counts = env.get_attr('collision_count')
            if collision_counts and collision_counts[0] is not None:
                self.logger.record('metrics/collision_count', collision_counts[0])
        except (AttributeError, NotImplementedError):
            if self.verbose:
                print("Warning: collision_count attribute not available in environments")
        
        self.total_actions += self.n_envs
        
        # Check for episode ends across all environments
        for env_idx in range(self.n_envs):
            if self.locals['dones'][env_idx]:
                self.episode_counts[env_idx] += 1
                self.total_episode_count += 1

                # Check success
                loop_completed = self.locals['infos'][env_idx].get('loop_completed', False)
                if loop_completed:
                    self.loop_completed_counts[env_idx] += 1
                    self.total_loop_completed += 1
                    self.logger.record(f'success/env_{env_idx}_loop_completed', 1.0)
                else:
                    self.logger.record(f'success/env_{env_idx}_loop_completed', 0.0)
                
                # Print episode details in verbose mode
                if self.training_verbose >= 1:
                    # Get episode metrics from info dict
                    info = self.locals['infos'][env_idx]
                    episode_metrics = info.get('episode_metrics', {})
                    
                    # Determine if truncated (max steps/length) or terminated (loop completed)
                    termination_type = "completed" if loop_completed else "truncated"
                    
                    # Get final reward from the Monitor wrapper's episode info
                    final_reward = info.get('episode', {}).get('r', 0)
                    
                    # Get track length from episode metrics
                    track_length = episode_metrics.get('track_length', 0)
                    
                    print(f"Episode {self.episode_counts[env_idx]} (Env {env_idx}): "
                          f"Reward={final_reward:.1f}, "
                          f"Track={track_length} pieces, "
                          f"Loop={'✓' if loop_completed else '✗'}, "
                          f"Status={termination_type}")
                
                # Log overall success rate
                if self.total_episode_count > 0:
                    overall_success_rate = self.total_loop_completed / self.total_episode_count
                    self.logger.record('success/overall_loop_completion_rate', overall_success_rate)
                
                # Log per-environment success rate
                if self.episode_counts[env_idx] > 0:
                    env_success_rate = self.loop_completed_counts[env_idx] / self.episode_counts[env_idx]
                    self.logger.record(f'success/env_{env_idx}_completion_rate', env_success_rate)
                
                # Log curriculum info if available (from first env that completes)
                if 'curriculum_stage' in self.locals['infos'][env_idx]:
                    self.logger.record('curriculum/stage', self.locals['infos'][env_idx]['curriculum_stage'])
                if 'max_track_length' in self.locals['infos'][env_idx]:
                    self.logger.record('curriculum/max_length', self.locals['infos'][env_idx]['max_track_length'])
                if 'curriculum_success_rate' in self.locals['infos'][env_idx]:
                    self.logger.record('curriculum/stage_success_rate', 
                                     self.locals['infos'][env_idx]['curriculum_success_rate'])
                
                # Episode metrics provided via info dict before reset
                info_metrics = self.locals['infos'][env_idx].get('episode_metrics', {})
                if info_metrics:
                    if 'track_length' in info_metrics and loop_completed:
                        self.logger.record(f'success/env_{env_idx}_completed_track_length', info_metrics['track_length'])
                    if 'min_distance' in info_metrics:
                        self.logger.record(f'navigation/env_{env_idx}_min_distance', info_metrics['min_distance'])
                    if env_idx == 0:  # Log detailed metrics only from first env
                        if 'phase_rewards' in info_metrics:
                            for phase, reward in info_metrics['phase_rewards'].items():
                                self.logger.record(f'rewards/{phase}_total', reward)
                        if 'chain_lift_count' in info_metrics:
                            self.logger.record('chain_lift/count', info_metrics['chain_lift_count'])
                        if 'remove_count' in info_metrics:
                            self.logger.record('behavior/remove_count', info_metrics['remove_count'])

        # Calculate and log throughput
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            steps_per_second = self.total_steps / elapsed_time
            self.logger.record('performance/steps_per_second', steps_per_second)
            self.logger.record('performance/total_episodes', self.total_episode_count)
            
        # Print progress dashboard (only once per milestone)
        dashboard_interval = 10 * self.n_envs
        if (self.total_episode_count > 0 and 
            self.total_episode_count >= self.last_dashboard_episode + dashboard_interval):
            # Update last printed milestone
            self.last_dashboard_episode = (self.total_episode_count // dashboard_interval) * dashboard_interval
            
            overall_success_rate = self.total_loop_completed / self.total_episode_count
            episodes_per_second = self.total_episode_count / elapsed_time if elapsed_time > 0 else 0
            
            # Calculate dynamic dashboard width based on number of environments
            # Minimum 58, but expand if needed for environment status
            env_status_str_len = 16 + (self.n_envs * 2) + 2  # "Environments: [" + emojis + spaces + "]"
            dashboard_width = max(58, env_status_str_len + 2)  # +2 for padding
            
            # Create a clean dashboard display
            print("\n" + "┌" + "─" * dashboard_width + "┐")
            header = f"│ 🎮 Parallel Training Dashboard ({self.n_envs} environments)"
            print(header.ljust(dashboard_width + 1) + "│")
            print("├" + "─" * dashboard_width + "┤")
            
            # Environment status indicators
            env_status = []
            status_counts = {"🟢": 0, "🟡": 0, "🔴": 0, "⚪": 0}
            for i in range(self.n_envs):
                if self.episode_counts[i] > 0:
                    rate = self.loop_completed_counts[i] / self.episode_counts[i]
                    if rate >= 0.3:
                        env_status.append("🟢")  # Good performance
                        status_counts["🟢"] += 1
                    elif rate >= 0.1:
                        env_status.append("🟡")  # Learning
                        status_counts["🟡"] += 1
                    else:
                        env_status.append("🔴")  # Struggling
                        status_counts["🔴"] += 1
                else:
                    env_status.append("⚪")  # No episodes yet
                    status_counts["⚪"] += 1
            
            # Display environment status based on count
            if self.n_envs <= 20:
                # Show individual status for up to 20 environments
                print(f"│ Environments: [{' '.join(env_status)}]".ljust(dashboard_width + 1) + "│")
            else:
                # Show summary for many environments
                summary = f"🟢×{status_counts['🟢']} 🟡×{status_counts['🟡']} 🔴×{status_counts['🔴']} ⚪×{status_counts['⚪']}"
                print(f"│ Environments ({self.n_envs}): {summary}".ljust(dashboard_width + 1) + "│")
            print(f"│ Episodes: {self.total_episode_count:,} | Success: {overall_success_rate:.1%} ({self.total_loop_completed}/{self.total_episode_count})".ljust(dashboard_width + 1) + "│")
            print(f"│ Throughput: {steps_per_second:.1f} steps/s | {episodes_per_second:.2f} eps/s".ljust(dashboard_width + 1) + "│")
            
            # Get curriculum info from first environment if available
            if hasattr(env, 'envs') and len(env.envs) > 0:
                wrapped_env = env.envs[0]
                temp_env = wrapped_env
                while temp_env is not None:
                    if hasattr(temp_env, 'current_stage') and hasattr(temp_env, 'current_max_length'):
                        print(f"│ Curriculum: Stage {temp_env.current_stage} | Max Length: {temp_env.current_max_length}".ljust(dashboard_width + 1) + "│")
                        break
                    if hasattr(temp_env, 'env'):
                        temp_env = temp_env.env
                    else:
                        break
            
            print("└" + "─" * dashboard_width + "┘")
            
            # Show detailed per-environment stats every 50 episodes
            if self.total_episode_count % (50 * self.n_envs) == 0:
                print("\n  Per-environment performance:")
                # Show all environments, but format differently for many environments
                if self.n_envs <= 8:
                    # Show detailed stats for up to 8 environments
                    for i in range(self.n_envs):
                        if self.episode_counts[i] > 0:
                            rate = self.loop_completed_counts[i] / self.episode_counts[i]
                            print(f"    Env {i}: {rate:.1%} success ({self.loop_completed_counts[i]}/{self.episode_counts[i]} episodes)")
                else:
                    # For many environments, show in a more compact format
                    print("    ", end="")
                    for i in range(self.n_envs):
                        if self.episode_counts[i] > 0:
                            rate = self.loop_completed_counts[i] / self.episode_counts[i]
                            print(f"E{i}:{rate:.0%} ", end="")
                            if (i + 1) % 8 == 0 and i < self.n_envs - 1:
                                print("\n    ", end="")
                    print()  # Final newline
        
        return True

def mask_fn(env: gym.Env) -> np.ndarray:
    """
    Returns the action mask for the current environment state.
    Navigates through all wrappers to find the base environment.
    """
    # Navigate through wrappers to find the base OpenRCT2 environment
    current_env = env
    while current_env is not None:
        # Check if this environment has the valid_action_mask method
        if hasattr(current_env, 'valid_action_mask'):
            return current_env.valid_action_mask()
        
        # Try to go deeper through the wrapper chain
        if hasattr(current_env, 'env'):
            current_env = current_env.env
        elif hasattr(current_env, 'unwrapped'):
            current_env = current_env.unwrapped
        else:
            break
    
    # Fallback - all actions valid (shouldn't reach here)
    print("Warning: Could not find valid_action_mask method, allowing all actions")
    return np.ones(env.action_space.n, dtype=bool)

def create_curriculum_masked_env(port: int, use_adaptive: bool = False,
                                  use_improved: bool = False, verbose: int = 0) -> gym.Env:
    """Create environment with curriculum wrapper and action masking for a specific port"""
    # Base environment with specific port and verbosity
    base_env = gym.make('OpenRCT2-v0', host='localhost', port=port, verbose=verbose)

    # Add OpenRCT2Wrapper to expose valid_action_mask method
    # This is crucial for the mask_fn to work
    base_env = OpenRCT2Wrapper(base_env)

    # Apply curriculum wrapper
    if use_improved:
        # Use improved 5-phase curriculum with physics-aware rewards
        env = ImprovedPhasedCurriculumWrapper(
            base_env,
            phase1_success_threshold=0.5,   # 50% loop completion
            phase2_success_threshold=0.4,   # 40% with 3+ chain lifts
            phase3_success_threshold=0.35,  # 35% with good patterns
            phase4_success_threshold=0.30,  # 30% clean completions
            phase5_success_threshold=0.25,  # 25% with quality ratings
            window_size=50,
            phase1_max_length=25,   # Return practice
            phase2_max_length=40,   # Lift hill building
            phase3_max_length=60,   # Drop & turn
            phase4_max_length=80,   # Circuit mastery
            phase5_initial_length=80,
            phase5_target_length=120,
            phase5_increase_step=10,
            verbose=verbose
        )
    elif use_adaptive:
        env = AdaptiveCurriculumWrapper(
            base_env,
            initial_max_length=50,
            target_max_length=120,
            success_threshold=0.2,  # 20% success to advance
            window_size=50,
            increase_step=10
        )
    else:
        env = CurriculumWrapper(
            base_env,
            initial_max_length=50,
            target_max_length=120,
            success_threshold=0.2,
            window_size=50,
            increase_step=10
        )
    
    # Add Monitor for logging
    env = Monitor(env)
    
    # Add ActionMasker for MaskablePPO
    env = ActionMasker(env, mask_fn)
    
    return env

def make_env_factory(port: int, use_adaptive: bool = False,
                      use_improved: bool = False, verbose: int = 0) -> Callable[[], gym.Env]:
    """Create a factory function for an environment on a specific port"""
    def _init() -> gym.Env:
        try:
            env = create_curriculum_masked_env(port, use_adaptive, use_improved, verbose)
            print(f"✅ Successfully connected to OpenRCT2 on port {port}")
            return env
        except Exception as e:
            print(f"❌ Failed to connect to OpenRCT2 on port {port}: {e}")
            raise
    return _init

def train_parallel_curriculum_masked(
    ports: List[int],
    total_timesteps: int,
    checkpoint_freq: int,
    eval_freq: int,
    model_path: str | None = None,
    use_adaptive: bool = False,
    use_improved: bool = False,
    use_subproc: bool = False,
    verbose: int = 0,
    eval_episodes: int = 10,
    disable_eval: bool = False,
):
    """Train agent with curriculum learning AND action masking on multiple parallel environments"""

    n_envs = len(ports)

    print("="*60)
    print("🎓 PARALLEL CURRICULUM LEARNING + ACTION MASKING")
    print("="*60)
    print(f"Training on {n_envs} parallel OpenRCT2 instances")
    print(f"Ports: {', '.join(map(str, ports))}")
    if use_improved:
        print("Using IMPROVED 5-phase curriculum with physics-aware rewards:")
        print("  Phase 1: Return Practice (25 pieces) - Learn navigation")
        print("  Phase 2: Lift Hill Building (40 pieces) - Learn chain lifts & energy")
        print("  Phase 3: Drop & Turn (60 pieces) - Learn drops & turnarounds")
        print("  Phase 4: Circuit Mastery (80 pieces) - Full integration")
        print("  Phase 5: Quality Optimization (80-120 pieces) - E=7-9, I=4.5-6.5, N<4.5")
        print("  + Energy estimation, pattern detection, approach guidance")
    elif use_adaptive:
        print("Using adaptive curriculum (50-120 pieces)")
    else:
        print("Starting with short tracks (50 pieces)")
        print("Will gradually increase to 120 pieces")
    print("Using MaskablePPO to prevent invalid actions")
    print("="*60 + "\n")
    
    # Create environment factories for each port
    env_factories = [make_env_factory(port, use_adaptive, use_improved, verbose) for port in ports]
    
    # Create vectorized environments
    print(f"\n🔌 Connecting to {n_envs} OpenRCT2 instances...")
    if use_subproc and n_envs > 1:
        # SubprocVecEnv for true parallel execution
        # Now more stable with retry logic and proper resource cleanup
        env = SubprocVecEnv(env_factories)
        print(f"✅ Created {n_envs} parallel environments using SubprocVecEnv")
    else:
        # DummyVecEnv runs sequentially (stable but no parallelism)
        env = DummyVecEnv(env_factories)
        if n_envs > 1:
            print(f"✅ Created {n_envs} environments using DummyVecEnv (sequential)")
            print("   💡 Use --use-subproc for true parallel execution")
        else:
            print(f"✅ Created {n_envs} environment using DummyVecEnv")
    
    # Normalize ONLY the continuous 'scalars' key. norm_reward=False preserves the
    # curriculum's tuned absolute reward magnitudes and phase-advancement thresholds.
    # Other keys (map, tokens, mask, already-clipped goal vectors, Discretes) are left
    # alone (VecNormalize handles only Box keys, and would corrupt token ids).
    vecnorm_path = _vecnormalize_path(model_path) if model_path else None
    if vecnorm_path and os.path.exists(vecnorm_path):
        print(f"Loading VecNormalize stats from {vecnorm_path}")
        env = VecNormalize.load(vecnorm_path, env)
        env.training = True
        env.norm_reward = False
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=False, norm_obs_keys=["scalars"])

    # IMPORTANT: Do NOT create a separate eval env on the same ports.
    # We will evaluate using the training env between learn chunks to avoid
    # corrupting in-progress episodes on shared API ports.
    
    # Create or load model
    if model_path and os.path.exists(model_path):
        print(f"Loading MaskablePPO model from {model_path}")
        model = MaskablePPO.load(model_path, env=env)
    else:
        print(f"Creating new MaskablePPO model for {n_envs} parallel environments")
        policy_kwargs = dict(
            features_extractor_class=BuildHistoryExtractor,
            features_extractor_kwargs=dict(encoder="gru"),
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            normalize_images=False,
        )
        
        # Adjust n_steps and batch_size ensuring train_batch_size % batch_size == 0
        # Aim for rollout size around 2048, but keep n_steps >= 128
        target_rollout = 2048
        base = target_rollout // max(1, n_envs)
        # Align to 64 for better batch divisibility and keep >= 128
        n_steps = max(128, (base // 64) * 64 if base >= 64 else 128)
        train_batch_size = n_envs * n_steps
        # Start with a reasonable minibatch size that divides train_batch_size
        batch_size = math.gcd(train_batch_size, 64 * n_envs)
        batch_size = max(32, batch_size)
        assert train_batch_size % batch_size == 0, (
            "train_batch_size must be divisible by batch_size"
        )
        
        model = MaskablePPO(
            MaskableMultiInputActorCriticPolicy,
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log="./parallel_curriculum_masked_tensorboard/",
            learning_rate=3e-4,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=10,
            gamma=GAMMA,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,  # Exploration
        )

    # PBRS invariance precondition: the model's discount must match the reward's gamma.
    # Catches a stale loaded model trained under a different gamma.
    assert model.gamma == GAMMA, (
        f"model gamma {model.gamma} != reward gamma {GAMMA}; PBRS invariance would break"
    )

    # Create log directory
    log_dir = f"logs_parallel_curriculum_masked_{n_envs}envs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Callbacks
    _ckpt_save_freq = max(1, checkpoint_freq // max(1, n_envs))  # Guard against zero
    _ckpt_prefix = f"parallel_curriculum_masked_{n_envs}envs"
    checkpoint_callback = CheckpointCallback(
        save_freq=_ckpt_save_freq,
        save_path=log_dir,
        name_prefix=_ckpt_prefix
    )

    # Save VecNormalize stats alongside each checkpoint so checkpoints stay reloadable.
    vecnormalize_callback = SaveVecNormalizeCallback(
        save_freq=_ckpt_save_freq,
        save_path=log_dir,
        name_prefix=_ckpt_prefix,
    )

    tensorboard_callback = ParallelCurriculumMaskableCallback(n_envs=n_envs, training_verbose=verbose)

    # Container for final curriculum statistics
    stats = None

    # Train
    try:
        print(f"\n🚂 Starting parallel training on {n_envs} environments...")
        print("Features enabled:")
        print("  ✓ Curriculum learning (50 → 120 pieces)")
        print("  ✓ True action masking (invalid actions prevented)")
        print("  ✓ Stronger return rewards")
        print("  ✓ Distance checkpoints")
        print("  ✓ Chain lift incentives")
        print(f"  ✓ {n_envs}x parallel environment execution")
        print("\nMonitor progress in Tensorboard:")
        print("  tensorboard --logdir ./parallel_curriculum_masked_tensorboard/\n")
        
        # Train in chunks, evaluating with the SAME env in between chunks.
        remaining = total_timesteps
        chunk = max(1, eval_freq) if not disable_eval else remaining
        learned = 0
        while remaining > 0:
            this_chunk = remaining if disable_eval else min(chunk, remaining)
            model.learn(
                total_timesteps=this_chunk,
                callback=[checkpoint_callback, vecnormalize_callback, tensorboard_callback],
                reset_num_timesteps=False,
            )
            learned += this_chunk
            remaining -= this_chunk
            
            # Evaluate between chunks using the training env to avoid port conflicts
            if not disable_eval and eval_episodes > 0:
                print(f"\n📈 Intermediate evaluation after {learned:,} timesteps...")

                # Temporarily disable curriculum statistics during evaluation
                curriculum_wrappers = []
                base_vec = _unwrap_to_vecenv_with_envs(env)
                if base_vec is not None:
                    for wrapped_env in base_vec.envs:
                        temp_env = wrapped_env
                        while temp_env is not None:
                            if hasattr(temp_env, 'evaluation_mode'):
                                curriculum_wrappers.append(temp_env)
                                break
                            if hasattr(temp_env, 'env'):
                                temp_env = temp_env.env
                            else:
                                break

                # Freeze VecNormalize running stats during evaluation so eval rollouts
                # don't pollute the obs normalization statistics.
                vecn = model.get_vec_normalize_env()
                prev_training = vecn.training if vecn is not None else None
                if vecn is not None:
                    vecn.training = False
                try:
                    with ExitStack() as stack:
                        for cw in curriculum_wrappers:
                            stack.enter_context(cw.evaluation_mode())
                        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=eval_episodes)
                finally:
                    if vecn is not None:
                        vecn.training = prev_training
                print(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Retrieve final curriculum stats before closing the environment
        base_vec = _unwrap_to_vecenv_with_envs(env)
        if base_vec is not None and len(base_vec.envs) > 0:
            env_instance = base_vec.envs[0]
            curriculum_env = None
            temp_env = env_instance

            # Navigate through wrappers to find CurriculumWrapper
            while temp_env is not None:
                if hasattr(temp_env, 'get_curriculum_stats') or hasattr(temp_env, 'get_phase_stats'):
                    curriculum_env = temp_env
                    break
                if hasattr(temp_env, 'env'):
                    temp_env = temp_env.env
                else:
                    break

            if curriculum_env:
                # Check if it's a phased curriculum wrapper
                if hasattr(curriculum_env, 'get_phase_stats'):
                    stats = curriculum_env.get_phase_stats()
                else:
                    stats = curriculum_env.get_curriculum_stats()

        # Persist final VecNormalize stats (needed to reload the model for eval/resume)
        # before the env is closed.
        try:
            vec_env = model.get_vec_normalize_env()
            if vec_env is not None:
                vec_env.save(_vecnormalize_path(os.path.join(log_dir, "final_model")))
        except Exception as e:
            print(f"⚠️ Could not save VecNormalize stats: {e}")

        # Clean up environments after collecting stats
        env.close()

    # Save final model
    final_model_path = os.path.join(log_dir, "final_model")
    model.save(final_model_path)
    print(f"\n💾 Final model saved to {final_model_path}")
    print(f"💾 VecNormalize stats: {_vecnormalize_path(final_model_path)}")

    # Log final curriculum stats if available
    if stats:
        if 'current_phase' in stats:  # Phased curriculum stats
            print("\n📊 Final Phased Curriculum Stats:")
            print(f"  Current phase: {stats['current_phase']}")
            if stats['current_phase'] == 5 and stats.get('phase5_stage'):
                print(f"  Phase 5 stage: {stats['phase5_stage']}")
            print(f"  Max track length: {stats['current_max_length']}")
            print(f"  Total episodes: {stats['total_episodes']}")
            print(f"  Success rate: {stats['success_rate']:.1%}")
            print(f"  Total loops completed: {stats['total_loops_completed']}")
            
            if stats['phases_completed']:
                print("\n  Phase progression:")
                for phase in stats['phases_completed']:
                    if 'phase' in phase:
                        print(f"    Phase {phase['phase']}: "
                              f"{phase['success_rate']:.1%} success rate, "
                              f"{phase.get('episodes', 0)} episodes")
        else:  # Regular curriculum stats
            print("\n📊 Final Curriculum Stats:")
            print(f"  Stage reached: {stats['current_stage']}")
            print(f"  Max track length: {stats.get('max_track_length', stats.get('current_max_length', 'N/A'))}")
            print(f"  Total episodes: {stats['total_episodes']}")
            print(f"  Success rate: {stats['success_rate']:.1%}")
            print(f"  Stages completed: {len(stats.get('stages_completed', []))}")

            if stats.get('stages_completed'):
                print("\n  Stage progression:")
                for stage in stats['stages_completed']:
                    print(f"    Stage {stage['stage']}: {stage['max_length']} pieces, "
                          f"{stage['success_rate']:.1%} success rate")

    return model, env

def main():
    parser = argparse.ArgumentParser(description="Parallel training with curriculum + masking")
    parser.add_argument("--ports", type=str, default="8080",
                       help="Comma-separated list of ports for OpenRCT2 API servers (e.g., 8080,8081,8082)")
    parser.add_argument("--timesteps", type=int, default=1000000,
                       help="Total timesteps to train (default: 1M)")
    parser.add_argument("--checkpoint-freq", type=int, default=10000,
                       help="Checkpoint frequency (in timesteps)")
    parser.add_argument("--eval-freq", type=int, default=100000,
                       help="Evaluate between learn chunks every N timesteps using the training env; set 0 to disable")
    parser.add_argument("--eval-episodes", type=int, default=10,
                       help="Number of episodes per intermediate evaluation")
    parser.add_argument("--disable-eval", action="store_true",
                       help="Disable intermediate evaluation entirely (safer for maximum throughput)")
    parser.add_argument("--model-path", type=str,
                       help="Path to existing MaskablePPO model to continue training")
    parser.add_argument("--adaptive", action="store_true",
                       help="Use adaptive curriculum with dynamic reward scaling")
    parser.add_argument("--improved", action="store_true",
                       help="Use improved 5-phase curriculum with physics-aware rewards (RECOMMENDED)")
    parser.add_argument("--use-subproc", action="store_true",
                       help="Use SubprocVecEnv for true parallel execution (test with retry fixes)")
    parser.add_argument("--verbose", type=int, default=None,
                       help="Verbosity level: 0=silent, 1=important, 2=detailed (default: auto)")
    args = parser.parse_args()
    
    # Parse ports
    try:
        ports = [int(port.strip()) for port in args.ports.split(',')]
    except ValueError:
        print("❌ Error: Invalid port format. Please provide comma-separated integers (e.g., 8080,8081)")
        return
    
    print("\n" + "="*60)
    print("🎢 OpenRCT2 Parallel Training: Curriculum + Action Masking")
    print("="*60)
    print("This combines the best approaches with parallel execution:")
    print("  • Curriculum learning for gradual difficulty")
    print("  • True action masking to prevent invalid moves")
    print("  • All reward improvements for better navigation")
    print(f"  • {len(ports)}x parallel environments for faster training")
    print("="*60 + "\n")
    
    # Validate that we can connect to at least one server
    print("🔍 Checking OpenRCT2 API server availability...")
    available_ports = []
    for port in ports:
        try:
            from openrct2_gym.envs.api_controller import APIController
            controller = APIController('localhost', port, verbose=0)  # Silent for connection check
            if controller.connect():
                available_ports.append(port)
                print(f"  ✅ Port {port}: Available")
            else:
                print(f"  ⚠️ Port {port}: Cannot connect")
            # Always disconnect the probe socket to avoid leaking the connection
            controller.disconnect()
        except Exception as e:
            print(f"  ⚠️ Port {port}: Error - {e}")
    
    if not available_ports:
        print("\n❌ Error: No OpenRCT2 API servers available on specified ports")
        print("Please ensure OpenRCT2 is running with the API plugin on the specified ports")
        return
    
    if len(available_ports) < len(ports):
        print(f"\n⚠️ Warning: Only {len(available_ports)} out of {len(ports)} ports are available")
        print(f"Continuing with available ports: {', '.join(map(str, available_ports))}")
    
    # Clean up any leftover rides from previous training sessions
    print("\n🧹 Cleaning up leftover rides from previous sessions...")
    for port in available_ports:
        try:
            controller = APIController('localhost', port, verbose=0)  # Silent for connection check
            if controller.connect():
                result = controller.delete_all_rides()
                if result.get("success"):
                    print(f"  ✅ Port {port}: Cleaned up all rides")
                else:
                    print(f"  ⚠️ Port {port}: Cleanup failed - {result.get('error', 'Unknown error')}")
                controller.disconnect()
        except Exception as e:
            print(f"  ⚠️ Port {port}: Error during cleanup - {e}")
    print("Cleanup complete!\n")
    
    # Auto-determine verbosity if not specified
    if args.verbose is None:
        # Use verbose=0 for multiple environments, 1 for single
        verbose = 0 if len(available_ports) > 1 else 1
    else:
        verbose = args.verbose
    
    if verbose == 0 and len(available_ports) > 1:
        print("\n💡 Tip: Running in silent mode. Use --verbose 1 or 2 for more details")
    
    model, env = train_parallel_curriculum_masked(
        available_ports,
        args.timesteps,
        args.checkpoint_freq,
        args.eval_freq,
        args.model_path,
        args.adaptive,
        args.improved,
        args.use_subproc,
        verbose,
        args.eval_episodes,
        args.disable_eval or args.eval_freq <= 0,
    )
    # Training function already evaluates between chunks and closes env.
    # No additional evaluation here to avoid interfering with API ports.

if __name__ == "__main__":
    main()
