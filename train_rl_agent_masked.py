#!/usr/bin/env python3
"""
Training script for OpenRCT2 RL agent using MaskablePPO with action masking
"""
import gymnasium as gym
import openrct2_gym
from openrct2_gym import make_openrct2_env
from openrct2_gym.envs.wrappers import OpenRCT2Wrapper
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import os
import argparse
import numpy as np

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.loop_completed_count = 0
        self.episode_count = 0
        self.collision_count = 0

    def _on_step(self) -> bool:
        # Log scalar value
        track_length = self.model.get_env().get_attr('track_length')[0]
        self.logger.record('track_length', track_length)
        
        # Log collision count
        collision_count = self.model.get_env().get_attr('collision_count')[0]
        self.logger.record('collision_count', collision_count)
        
        # Check if episode is done
        if self.locals['dones'][0]:
            self.episode_count += 1
            
            # Check if the episode terminated (loop completed) or was truncated
            terminated = self.locals['infos'][0].get('terminal_observation') is not None
            
            if terminated:
                self.loop_completed_count += 1
                self.logger.record('loop_completed', 1.0)
                self.logger.record('completed_track_length', track_length)
            else:
                self.logger.record('loop_completed', 0.0)
            
            # Log the percentage of episodes where loop was completed
            loop_completion_rate = self.loop_completed_count / self.episode_count
            self.logger.record('loop_completion_rate', loop_completion_rate)

        return True

class ProgressCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(ProgressCallback, self).__init__(verbose)
        self.episode_count = 0

    def _on_step(self) -> bool:
        if self.locals['dones'][0]:
            self.episode_count += 1
            total_timesteps = self.num_timesteps
            
            # Check if the episode terminated (loop completed) or was truncated
            terminated = self.locals['infos'][0].get('terminal_observation') is not None
            
            # Get collision count
            collision_count = self.model.get_env().get_attr('collision_count')[0]
            
            print(f"Episode: {self.episode_count}")
            print(f"Total timesteps: {total_timesteps}")
            print(f"Episode reward: {self.locals['rewards'][0]:.2f}")
            print(f"Loop completed: {terminated}")
            print(f"Collisions: {collision_count}")
            print("------")
        return True

def mask_fn(env: gym.Env) -> np.ndarray:
    """
    Returns the action mask for the current environment state.
    """
    # The env here is Monitor wrapped around OpenRCT2Wrapper
    # We need to access the unwrapped environment to get to our custom method
    if hasattr(env, 'env') and hasattr(env.env, 'valid_action_mask'):
        return env.env.valid_action_mask()  # Access through Monitor -> OpenRCT2Wrapper
    elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'valid_action_mask'):
        return env.unwrapped.valid_action_mask()  # Access the base environment
    else:
        # Fallback - all actions valid
        return np.ones(env.action_space.n, dtype=bool)

def create_env():
    """
    Creates and wraps the environment with action masking.
    """
    env = gym.make('OpenRCT2-v0')  # Get base environment without our wrapper
    env = OpenRCT2Wrapper(env)  # Add our wrapper first to expose valid_action_mask
    env = Monitor(env)  # Then Monitor
    env = ActionMasker(env, mask_fn)  # Finally ActionMasker which needs valid_action_mask
    return DummyVecEnv([lambda: env])

def train_agent(total_timesteps, checkpoint_freq, eval_freq, model_path=None):
    env = create_env()
    
    # Create evaluation environment (also with masking)
    eval_env = gym.make('OpenRCT2-v0')  # Get base environment
    eval_env = OpenRCT2Wrapper(eval_env)  # Add our wrapper first to expose valid_action_mask
    eval_env = Monitor(eval_env)  # Then Monitor
    eval_env = ActionMasker(eval_env, mask_fn)  # Finally ActionMasker which needs valid_action_mask
    eval_env = DummyVecEnv([lambda: eval_env])

    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model = MaskablePPO.load(model_path, env=env)
    else:
        print("Creating new MaskablePPO model with action masking")
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
        )
        model = MaskablePPO(
            MaskableMultiInputActorCriticPolicy, 
            env, 
            policy_kwargs=policy_kwargs, 
            verbose=1, 
            tensorboard_log="./ppo_openrct2_tensorboard/",
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
        )

    # Create log directory
    log_dir = "logs_masked"
    os.makedirs(log_dir, exist_ok=True)

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=log_dir,
        name_prefix="maskable_ppo_openrct2"
    )
    
    # Use MaskableEvalCallback instead of regular EvalCallback
    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=eval_freq,
        deterministic=True,
        render=False
    )
    
    tensorboard_callback = TensorboardCallback()
    progress_callback = ProgressCallback()

    try:
        print("Starting training with action masking...")
        print("Valid actions will be dynamically determined by the API")
        print("Collisions will trigger automatic backtracking")
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback, tensorboard_callback, progress_callback],
            reset_num_timesteps=False  # Important for continuing training
        )
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()
    
    # Save the final model
    final_model_path = os.path.join(log_dir, "final_model")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")

    return model, env

def evaluate_agent(model, env):
    """
    Evaluate the trained agent using maskable evaluation.
    """
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

def main():
    parser = argparse.ArgumentParser(description="Train RL agent for OpenRCT2 with action masking")
    parser.add_argument("--timesteps", type=int, default=200000, help="Total timesteps to train")
    parser.add_argument("--checkpoint-freq", type=int, default=10000, help="Frequency of checkpoints")
    parser.add_argument("--eval-freq", type=int, default=10000, help="Frequency of evaluations")
    parser.add_argument("--model-path", type=str, help="Path to a saved model to continue training")
    args = parser.parse_args()

    print("=" * 60)
    print("OpenRCT2 RL Training with Action Masking")
    print("=" * 60)
    print("Features enabled:")
    print("  ✓ Dynamic action masking from API")
    print("  ✓ Collision detection and tracking")
    print("  ✓ Automatic backtracking on repeated failures")
    print("  ✓ API-based ride demolition on reset")
    print("=" * 60)
    
    model, env = train_agent(args.timesteps, args.checkpoint_freq, args.eval_freq, args.model_path)
    evaluate_agent(model, env)

    env.close()

if __name__ == "__main__":
    main()