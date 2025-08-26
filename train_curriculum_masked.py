#!/usr/bin/env python3
"""
Training script with curriculum learning AND proper action masking using MaskablePPO
Combines the best of both approaches for optimal learning
"""
import gymnasium as gym
import openrct2_gym
from openrct2_gym.envs.curriculum_wrapper import CurriculumWrapper, AdaptiveCurriculumWrapper
from openrct2_gym.envs.wrappers import OpenRCT2Wrapper
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
import numpy as np
import os
import argparse

class CurriculumMaskableCallback(BaseCallback):
    """
    Tensorboard callback that tracks both curriculum and masking metrics
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0
        self.loop_completed_count = 0
        self.invalid_action_count = 0
        self.total_actions = 0
        
    def _on_step(self) -> bool:
        # Get environment through vectorized wrapper
        env = self.model.get_env()
        
        # Try to get base environment
        base_env = None
        if hasattr(env, 'envs') and len(env.envs) > 0:
            wrapped_env = env.envs[0]
            # Navigate through wrappers to find base environment
            temp_env = wrapped_env
            while temp_env is not None:
                if hasattr(temp_env, 'track_length'):
                    base_env = temp_env
                    break
                if hasattr(temp_env, 'env'):
                    temp_env = temp_env.env
                else:
                    break
        
        # Log metrics if we found the base environment
        if base_env:
            if hasattr(base_env, 'track_length'):
                self.logger.record('metrics/track_length', base_env.track_length)
            
            if hasattr(base_env, '_calculate_distance_to_start'):
                distance = base_env._calculate_distance_to_start()[0]
                self.logger.record('metrics/current_distance', distance)
            
            if hasattr(base_env, 'collision_count'):
                self.logger.record('metrics/collision_count', base_env.collision_count)
        
        self.total_actions += 1
        
        # Check for episode end
        if self.locals['dones'][0]:
            self.episode_count += 1

            # Check success
            terminated = self.locals['infos'][0].get('terminal_observation') is not None
            if terminated:
                self.loop_completed_count += 1
                self.logger.record('success/loop_completed', 1.0)
            else:
                self.logger.record('success/loop_completed', 0.0)
            
            # Log success rate
            loop_completion_rate = self.loop_completed_count / self.episode_count
            self.logger.record('success/loop_completion_rate', loop_completion_rate)
            
            # Log curriculum info if available
            if 'curriculum_stage' in self.locals['infos'][0]:
                self.logger.record('curriculum/stage', self.locals['infos'][0]['curriculum_stage'])
            if 'max_track_length' in self.locals['infos'][0]:
                self.logger.record('curriculum/max_length', self.locals['infos'][0]['max_track_length'])
            if 'curriculum_success_rate' in self.locals['infos'][0]:
                self.logger.record('curriculum/stage_success_rate', 
                                 self.locals['infos'][0]['curriculum_success_rate'])
            
            # Episode metrics provided via info dict before reset
            info_metrics = self.locals['infos'][0].get('episode_metrics', {})
            if info_metrics:
                if 'track_length' in info_metrics and terminated:
                    self.logger.record('success/completed_track_length', info_metrics['track_length'])
                if 'min_distance' in info_metrics:
                    self.logger.record('navigation/min_distance', info_metrics['min_distance'])
                if 'phase_rewards' in info_metrics:
                    for phase, reward in info_metrics['phase_rewards'].items():
                        self.logger.record(f'rewards/{phase}_total', reward)
                if 'chain_lift_count' in info_metrics:
                    self.logger.record('chain_lift/count', info_metrics['chain_lift_count'])
                if 'remove_count' in info_metrics:
                    self.logger.record('behavior/remove_count', info_metrics['remove_count'])

            # Print progress
            if self.episode_count % 10 == 0:
                print(f"\n📊 Episode {self.episode_count}: "
                      f"Success rate: {loop_completion_rate:.1%} "
                      f"({self.loop_completed_count}/{self.episode_count})")
                track_info = []
                if 'track_length' in info_metrics:
                    track_info.append(f"{info_metrics['track_length']} pieces")
                if 'min_distance' in info_metrics:
                    track_info.append(f"Min dist: {info_metrics['min_distance']:.1f}")
                if track_info:
                    print(f"   Last track: {', '.join(track_info)}")
        
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

def create_curriculum_masked_env(use_adaptive=False):
    """Create environment with curriculum wrapper and action masking"""
    # Base environment
    base_env = gym.make('OpenRCT2-v0')
    
    # Add OpenRCT2Wrapper to expose valid_action_mask method
    # This is crucial for the mask_fn to work
    base_env = OpenRCT2Wrapper(base_env)
    
    # Apply curriculum wrapper
    if use_adaptive:
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

def train_curriculum_masked(total_timesteps, checkpoint_freq, eval_freq, 
                           model_path=None, use_adaptive=False):
    """Train agent with curriculum learning AND action masking"""
    
    print("="*60)
    print("🎓 CURRICULUM LEARNING + ACTION MASKING ENABLED")
    print("="*60)
    print("Starting with short tracks (30 pieces)")
    print("Will gradually increase to 100 pieces")
    print("Using MaskablePPO to prevent invalid actions")
    print("="*60 + "\n")
    
    # Create environments
    env_fn = lambda: create_curriculum_masked_env(use_adaptive)
    env = DummyVecEnv([env_fn])
    
    eval_env_fn = lambda: create_curriculum_masked_env(use_adaptive)
    eval_env = DummyVecEnv([eval_env_fn])
    
    # Create or load model
    if model_path and os.path.exists(model_path):
        print(f"Loading MaskablePPO model from {model_path}")
        model = MaskablePPO.load(model_path, env=env)
    else:
        print("Creating new MaskablePPO model with curriculum learning")
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        )
        model = MaskablePPO(
            MaskableMultiInputActorCriticPolicy,
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log="./curriculum_masked_tensorboard/",
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,  # Exploration
        )
    
    # Create log directory
    log_dir = "logs_curriculum_masked"
    os.makedirs(log_dir, exist_ok=True)
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=log_dir,
        name_prefix="curriculum_masked_model"
    )
    
    # Use MaskableEvalCallback for proper evaluation with masking
    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=eval_freq,
        deterministic=True,
        render=False
    )
    
    tensorboard_callback = CurriculumMaskableCallback()
    
    # Train
    try:
        print("\n🚂 Starting training with curriculum learning AND action masking...")
        print("Features enabled:")
        print("  ✓ Curriculum learning (30 → 100 pieces)")
        print("  ✓ True action masking (invalid actions prevented)")
        print("  ✓ Stronger return rewards")
        print("  ✓ Distance checkpoints")
        print("  ✓ Chain lift incentives")
        print("\nMonitor progress in Tensorboard:")
        print("  tensorboard --logdir ./curriculum_masked_tensorboard/\n")
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback, tensorboard_callback],
            reset_num_timesteps=False
        )
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
    
    # Save final model
    final_model_path = os.path.join(log_dir, "final_model")
    model.save(final_model_path)
    print(f"\n💾 Final model saved to {final_model_path}")
    
    # Get final curriculum stats
    env_instance = env.envs[0]
    curriculum_env = None
    temp_env = env_instance
    
    # Navigate through wrappers to find CurriculumWrapper
    while temp_env is not None:
        if isinstance(temp_env, (CurriculumWrapper, AdaptiveCurriculumWrapper)):
            curriculum_env = temp_env
            break
        if hasattr(temp_env, 'env'):
            temp_env = temp_env.env
        else:
            break
    
    if curriculum_env:
        stats = curriculum_env.get_curriculum_stats()
        print("\n📊 Final Curriculum Stats:")
        print(f"  Stage reached: {stats['current_stage']}")
        print(f"  Max track length: {stats['max_track_length']}")
        print(f"  Total episodes: {stats['total_episodes']}")
        print(f"  Success rate: {stats['success_rate']:.1%}")
        print(f"  Stages completed: {len(stats['stages_completed'])}")
        
        if stats['stages_completed']:
            print("\n  Stage progression:")
            for stage in stats['stages_completed']:
                print(f"    Stage {stage['stage']}: {stage['max_length']} pieces, "
                      f"{stage['success_rate']:.1%} success rate")
    
    return model, env

def main():
    parser = argparse.ArgumentParser(description="Train with curriculum + masking")
    parser.add_argument("--timesteps", type=int, default=1000000,
                       help="Total timesteps to train (default: 1M)")
    parser.add_argument("--checkpoint-freq", type=int, default=10000,
                       help="Checkpoint frequency")
    parser.add_argument("--eval-freq", type=int, default=10000,
                       help="Evaluation frequency")
    parser.add_argument("--model-path", type=str,
                       help="Path to existing MaskablePPO model to continue training")
    parser.add_argument("--adaptive", action="store_true",
                       help="Use adaptive curriculum with dynamic reward scaling")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🎢 OpenRCT2 Training: Curriculum + Action Masking")
    print("="*60)
    print("This combines the best of both approaches:")
    print("  • Curriculum learning for gradual difficulty")
    print("  • True action masking to prevent invalid moves")
    print("  • All reward improvements for better navigation")
    print("="*60 + "\n")
    
    model, env = train_curriculum_masked(
        args.timesteps,
        args.checkpoint_freq,
        args.eval_freq,
        args.model_path,
        args.adaptive
    )
    
    # Final evaluation using maskable evaluation
    print("\n📈 Final evaluation with masking...")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    
    env.close()

if __name__ == "__main__":
    main()
