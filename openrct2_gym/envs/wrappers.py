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
    
    @property
    def min_distance_reached(self) -> float:
        """Forward min_distance_reached property to the unwrapped environment."""
        return self.unwrapped.min_distance_reached
    
    @property
    def max_height_reached(self) -> int:
        """Forward max_height_reached property to the unwrapped environment."""
        return self.unwrapped.max_height_reached
    
    @property
    def chain_lift_count(self) -> int:
        """Forward chain_lift_count property to the unwrapped environment."""
        return self.unwrapped.chain_lift_count
    
    @property
    def remove_count(self) -> int:
        """Forward remove_count property to the unwrapped environment."""
        return self.unwrapped.remove_count
    
    @property
    def phase_rewards(self) -> dict:
        """Forward phase_rewards property to the unwrapped environment."""
        return self.unwrapped.phase_rewards
    
    @property
    def current_phase(self) -> str:
        """Forward current_phase property to the unwrapped environment."""
        return self.unwrapped.current_phase
    
    @property
    def loop_completed(self) -> bool:
        """Forward loop_completed property to the unwrapped environment."""
        return self.unwrapped.loop_completed
    
    def _calculate_distance_to_start(self) -> np.ndarray:
        """Forward distance calculation to the unwrapped environment."""
        return self.unwrapped._calculate_distance_to_start()

    def delete_all_rides(self):
        """Forward delete_all_rides to the API controller."""
        return self.unwrapped.api_controller.delete_all_rides()

    # Physics-aware reward methods
    @property
    def height_history(self) -> list:
        """Forward height_history property to the unwrapped environment."""
        return self.unwrapped.height_history

    def _calculate_estimated_energy(self) -> float:
        """Forward energy calculation to the unwrapped environment."""
        return self.unwrapped._calculate_estimated_energy()

    def _calculate_energy_margin(self) -> float:
        """Forward energy margin calculation to the unwrapped environment."""
        return self.unwrapped._calculate_energy_margin()