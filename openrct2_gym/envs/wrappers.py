"""
Custom wrappers for OpenRCT2 environment that properly forward custom methods.
"""
import gymnasium as gym
import numpy as np
from typing import Any, Dict, Tuple, Optional


class OpenRCT2Wrapper(gym.Wrapper):
    """
    Wrapper that properly forwards OpenRCT2-specific methods through the wrapper chain.
    This ensures methods like valid_action_mask() are accessible even when the environment
    is wrapped by gym.make().
    """
    
    def __init__(self, env):
        super().__init__(env)
    
    def valid_action_mask(self) -> np.ndarray:
        """Forward the valid_action_mask method to the unwrapped environment."""
        return self.unwrapped.valid_action_mask()
    
    @property
    def collision_count(self) -> int:
        """Forward collision_count property to the unwrapped environment."""
        return self.unwrapped.collision_count
    
    @property
    def consecutive_failures(self) -> int:
        """Forward consecutive_failures property to the unwrapped environment."""
        return self.unwrapped.consecutive_failures
    
    @property
    def max_consecutive_failures(self) -> int:
        """Forward max_consecutive_failures property to the unwrapped environment."""
        return self.unwrapped.max_consecutive_failures
    
    @property
    def track_length(self) -> int:
        """Forward track_length property to the unwrapped environment."""
        return self.unwrapped.track_length
    
    @property
    def track_pieces(self) -> list:
        """Forward track_pieces property to the unwrapped environment."""
        return self.unwrapped.track_pieces
    
    def delete_all_rides(self):
        """Forward delete_all_rides to the API controller."""
        return self.unwrapped.api_controller.delete_all_rides()