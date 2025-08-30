"""
Curriculum Learning Wrapper for OpenRCT2 Environment
Gradually increases difficulty as agent improves
"""
import gymnasium as gym
import numpy as np
from collections import deque

class CurriculumWrapper(gym.Wrapper):
    """
    Wrapper that implements curriculum learning by adjusting task difficulty
    based on agent performance.
    """
    
    def __init__(self, env, 
                 initial_max_length=30,
                 target_max_length=100,
                 success_threshold=0.3,
                 window_size=100,
                 increase_step=10):
        """
        Args:
            env: Base OpenRCT2 environment
            initial_max_length: Starting maximum track length
            target_max_length: Final maximum track length
            success_threshold: Success rate needed to increase difficulty
            window_size: Number of episodes to consider for success rate
            increase_step: How much to increase max length each time
        """
        super().__init__(env)
        
        # Curriculum parameters
        self.initial_max_length = initial_max_length
        self.target_max_length = target_max_length
        self.current_max_length = initial_max_length
        self.success_threshold = success_threshold
        self.increase_step = increase_step
        
        # Performance tracking
        self.episode_results = deque(maxlen=window_size)
        self.episode_count = 0
        self.current_stage = 1
        self.stages_completed = []
        
        # Update environment's max track length
        self._update_difficulty()
        
    def _update_difficulty(self):
        """Update the environment's maximum track length"""
        if hasattr(self.env, 'env'):
            # Handle wrapped environments
            base_env = self.env
            while hasattr(base_env, 'env'):
                base_env = base_env.env
            base_env.max_track_length = self.current_max_length
        else:
            self.env.max_track_length = self.current_max_length
    
    def _get_base_env(self):
        """Get the base OpenRCT2 environment"""
        env = self.env
        while hasattr(env, 'env'):
            env = env.env
        return env
    
    def reset(self, **kwargs):
        """Reset environment and potentially adjust difficulty"""
        # Check if we should increase difficulty
        if len(self.episode_results) >= 50:  # Need at least 50 episodes
            success_rate = sum(self.episode_results) / len(self.episode_results)
            
            # Check if we should advance to next stage
            if (success_rate >= self.success_threshold and 
                self.current_max_length < self.target_max_length):
                
                # Record completion of current stage
                self.stages_completed.append({
                    'stage': self.current_stage,
                    'max_length': self.current_max_length,
                    'success_rate': success_rate,
                    'episodes': self.episode_count
                })
                
                # Increase difficulty
                self.current_max_length = min(
                    self.current_max_length + self.increase_step,
                    self.target_max_length
                )
                self.current_stage += 1
                self._update_difficulty()
                
                # Clear history for new stage
                self.episode_results.clear()
                
                print(f"\n{'='*60}")
                print(f"🎯 CURRICULUM: Advancing to Stage {self.current_stage}")
                print(f"   Max track length: {self.current_max_length}")
                print(f"   Success rate achieved: {success_rate:.1%}")
                print(f"{'='*60}\n")
        
        # Reset the environment
        obs, info = self.env.reset(**kwargs)
        
        # Add curriculum info
        info['curriculum_stage'] = self.current_stage
        info['max_track_length'] = self.current_max_length
        info['episodes_at_stage'] = len(self.episode_results)
        
        return obs, info
    
    def step(self, action):
        """Execute action and track performance"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Track episode completion
        if terminated or truncated:
            self.episode_count += 1
            
            # Check if loop was completed
            base_env = self._get_base_env()
            success = base_env.loop_completed if hasattr(base_env, 'loop_completed') else False
            self.episode_results.append(success)
            
            # Add curriculum info to episode info
            info['curriculum_success'] = success
            info['curriculum_stage'] = self.current_stage
            info['curriculum_success_rate'] = (
                sum(self.episode_results) / len(self.episode_results) 
                if self.episode_results else 0
            )
            
            # Provide curriculum feedback
            if self.episode_count % 10 == 0:
                success_rate = sum(self.episode_results) / len(self.episode_results) if self.episode_results else 0
                print(f"📊 Curriculum Stage {self.current_stage}: "
                      f"Success rate: {success_rate:.1%} "
                      f"({sum(self.episode_results)}/{len(self.episode_results)}) "
                      f"Max length: {self.current_max_length}")
        
        # Add reward shaping based on curriculum stage
        if self.current_stage == 1 and terminated and info.get('curriculum_success'):
            # Extra reward for early successes to bootstrap learning
            reward += 20
        
        return obs, reward, terminated, truncated, info
    
    def get_curriculum_stats(self):
        """Get current curriculum learning statistics"""
        success_rate = sum(self.episode_results) / len(self.episode_results) if self.episode_results else 0
        
        return {
            'current_stage': self.current_stage,
            'max_track_length': self.current_max_length,
            'success_rate': success_rate,
            'episodes_at_stage': len(self.episode_results),
            'total_episodes': self.episode_count,
            'stages_completed': self.stages_completed,
            'target_reached': self.current_max_length >= self.target_max_length
        }


class AdaptiveCurriculumWrapper(CurriculumWrapper):
    """
    Advanced version that also adjusts other parameters like
    distance penalties and reward scaling based on performance.
    """
    
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.reward_scale = 1.0
        self.distance_penalty_scale = 1.0
    
    def _update_difficulty(self):
        """Update multiple difficulty parameters"""
        super()._update_difficulty()
        
        # Adjust reward scaling based on stage
        if self.current_stage <= 2:
            # Early stages: stronger rewards for navigation
            self.reward_scale = 1.5
            self.distance_penalty_scale = 0.5
        elif self.current_stage <= 4:
            # Mid stages: balanced
            self.reward_scale = 1.0
            self.distance_penalty_scale = 1.0
        else:
            # Late stages: normal rewards, stronger penalties
            self.reward_scale = 0.8
            self.distance_penalty_scale = 1.5
    
    def step(self, action):
        """Execute action with adjusted rewards"""
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Apply reward scaling
        base_env = self._get_base_env()
        if hasattr(base_env, 'current_phase'):
            if base_env.current_phase == 'return':
                # Scale return phase rewards based on curriculum
                reward *= self.reward_scale
        
        return obs, reward, terminated, truncated, info