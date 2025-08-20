"""
Simple action masking implementation for environments that doesn't require sb3-contrib.
This provides a basic alternative that works with standard PPO.
"""
import gymnasium as gym
import numpy as np
from typing import Optional, Tuple, Dict, Any

class SimpleActionMasker(gym.Wrapper):
    """
    A simple wrapper that masks invalid actions by giving them large negative rewards.
    This is less efficient than true action masking but works with standard PPO.
    """
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.invalid_action_penalty = -10.0
        
    def step(self, action: int) -> Tuple[Any, float, bool, bool, Dict]:
        # Get valid actions before taking the step
        if hasattr(self.env, 'valid_action_mask'):
            valid_mask = self.env.valid_action_mask()
            is_valid = valid_mask[action]
        else:
            is_valid = True
        
        # Take the action
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Apply penalty for invalid actions
        if not is_valid:
            reward += self.invalid_action_penalty
            info['invalid_action'] = True
        else:
            info['invalid_action'] = False
            
        return obs, reward, terminated, truncated, info
    
    def get_valid_actions(self) -> np.ndarray:
        """Returns array of valid action indices."""
        if hasattr(self.env, 'valid_action_mask'):
            mask = self.env.valid_action_mask()
            return np.where(mask)[0]
        else:
            return np.arange(self.action_space.n)


class SmartActionSampler:
    """
    Helper class to sample actions with preference for valid ones.
    Can be used during training to bias exploration towards valid actions.
    """
    
    def __init__(self, env: gym.Env, epsilon: float = 0.1):
        """
        Args:
            env: The environment (should have valid_action_mask method)
            epsilon: Probability of choosing a random action instead of valid one
        """
        self.env = env
        self.epsilon = epsilon
        
    def sample_action(self, policy_action: Optional[int] = None) -> int:
        """
        Sample an action with preference for valid ones.
        
        Args:
            policy_action: Action suggested by the policy (optional)
            
        Returns:
            Selected action
        """
        if hasattr(self.env, 'valid_action_mask'):
            mask = self.env.valid_action_mask()
            valid_actions = np.where(mask)[0]
            
            if len(valid_actions) == 0:
                # No valid actions, return remove action if possible
                return 18 if 18 < self.env.action_space.n else 0
            
            # With probability epsilon, explore randomly among valid actions
            if np.random.random() < self.epsilon:
                return np.random.choice(valid_actions)
            
            # Otherwise, use policy action if it's valid
            if policy_action is not None and mask[policy_action]:
                return policy_action
            
            # If policy action is invalid, choose random valid action
            return np.random.choice(valid_actions)
        else:
            # No masking available, return policy action or random
            if policy_action is not None:
                return policy_action
            return self.env.action_space.sample()


def filter_invalid_actions(logits: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """
    Filters invalid actions by setting their logits to very negative values.
    
    Args:
        logits: Action logits from the policy network
        valid_mask: Boolean mask of valid actions
        
    Returns:
        Modified logits with invalid actions masked
    """
    filtered_logits = logits.copy()
    filtered_logits[~valid_mask] = -1e10
    return filtered_logits