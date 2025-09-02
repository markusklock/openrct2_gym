"""
Phased Curriculum Learning Wrapper for OpenRCT2 Environment
Implements progressive learning phases focusing first on returning to station,
then gradually adding more complex objectives.
"""
import gymnasium as gym
import numpy as np
from collections import deque
from contextlib import contextmanager
from typing import Dict, Any, Tuple

class PhasedCurriculumWrapper(gym.Wrapper):
    """
    Wrapper that implements phased curriculum learning with different reward structures
    for each learning phase.
    
    Phase 1: "Find Home" - Focus on returning to station
    Phase 2: "Explore & Return" - Build longer tracks while maintaining return ability  
    Phase 3: "Build Quality" - Full complexity with all reward components
    """
    
    def __init__(self, env,
                 # Phase progression parameters
                 phase1_success_threshold=0.5,  # 50% success rate to advance from phase 1
                 phase2_success_threshold=0.4,  # 40% success rate to advance from phase 2
                 window_size=100,
                 # Track length curriculum
                 phase1_max_length=30,  # Short tracks for learning to return
                 phase2_max_length=60,  # Medium tracks
                 phase3_initial_length=60,
                 phase3_target_length=120,
                 phase3_increase_step=10,
                 phase3_success_threshold=0.3,
                 # Verbosity
                 verbose=1):
        """
        Args:
            env: Base OpenRCT2 environment
            phase1_success_threshold: Success rate needed to advance from phase 1
            phase2_success_threshold: Success rate needed to advance from phase 2
            window_size: Number of episodes to consider for success rate
            phase1_max_length: Maximum track length in phase 1
            phase2_max_length: Maximum track length in phase 2
            phase3_initial_length: Starting max length for phase 3
            phase3_target_length: Final maximum track length in phase 3
            phase3_increase_step: How much to increase max length in phase 3
            phase3_success_threshold: Success rate needed to increase difficulty in phase 3
            verbose: Verbosity level for logging
        """
        super().__init__(env)
        
        # Phase management
        self.current_phase = 1
        self.phases_completed = []
        
        # Phase-specific parameters
        self.phase1_success_threshold = phase1_success_threshold
        self.phase2_success_threshold = phase2_success_threshold
        self.phase3_success_threshold = phase3_success_threshold
        
        # Track length parameters per phase
        self.phase1_max_length = phase1_max_length
        self.phase2_max_length = phase2_max_length
        self.phase3_initial_length = phase3_initial_length
        self.phase3_target_length = phase3_target_length
        self.phase3_current_length = phase3_initial_length
        self.phase3_increase_step = phase3_increase_step
        
        # Performance tracking
        self.episode_results = deque(maxlen=window_size)
        self.episode_count = 0
        self.phase_episode_count = 0
        self.total_loops_completed = 0
        
        # Phase 3 sub-stages
        self.phase3_stage = 1
        
        # Statistics tracking control
        self._track_stats = True
        
        # Verbosity
        self.verbose = verbose
        
        # Store original reward calculation method
        self._get_base_env()._original_calculate_reward = self._get_base_env()._calculate_reward
        
        # Update environment settings for current phase
        self._update_phase_settings()
    
    def _get_base_env(self):
        """Get the base OpenRCT2 environment"""
        env = self.env
        while hasattr(env, 'env'):
            env = env.env
        return env
    
    def _update_phase_settings(self):
        """Update environment settings based on current phase"""
        base_env = self._get_base_env()
        
        if self.current_phase == 1:
            base_env.max_track_length = self.phase1_max_length
            # Override reward calculation for phase 1
            base_env._calculate_reward = self._phase1_reward
        elif self.current_phase == 2:
            base_env.max_track_length = self.phase2_max_length
            base_env._calculate_reward = self._phase2_reward
        else:  # Phase 3
            base_env.max_track_length = self.phase3_current_length
            base_env._calculate_reward = self._phase3_reward
        
        if self.verbose >= 1:
            print(f"📚 Phase {self.current_phase} settings applied: max_length={base_env.max_track_length}")
    
    def _phase1_reward(self, success):
        """
        Phase 1: Focus entirely on finding way back to station
        - Strong distance-based rewards
        - Large bonus for completing loop
        - No rewards for track length or other metrics
        """
        base_env = self._get_base_env()
        reward = 0
        
        # Basic reward for successful placement
        if success:
            reward += 0.5  # Small reward for valid placement
            
            # Check if this is a removal action
            if base_env.last_action == base_env.action_space.n - 1:
                # Small penalty for removal to discourage excessive backtracking
                reward -= 1
            else:
                # MAIN FOCUS: Distance to start position
                current_distance = base_env._calculate_distance_to_start()[0]
                
                if len(base_env.position_history) > 1:
                    prev_distance = np.linalg.norm(
                        np.array(base_env.position_history[-2]) - np.array(base_env.goal_position)
                    )
                    distance_delta = prev_distance - current_distance
                    
                    # Strong rewards for getting closer to start
                    if distance_delta > 0:
                        reward += distance_delta * 3.0  # Strong multiplier
                        
                        # Extra bonus for getting very close
                        if current_distance < 10:
                            reward += (10 - current_distance) * 2.0
                        elif current_distance < 20:
                            reward += (20 - current_distance) * 0.5
                    else:
                        # Penalty for moving away
                        reward += distance_delta * 0.5  # Gentle penalty
                
                # Distance checkpoint rewards
                if current_distance < base_env.min_distance_achieved:
                    base_env.min_distance_achieved = current_distance
                    if current_distance < 50 and base_env.min_distance_achieved >= 50:
                        reward += 10
                    if current_distance < 30 and base_env.min_distance_achieved >= 30:
                        reward += 15
                    if current_distance < 20 and base_env.min_distance_achieved >= 20:
                        reward += 20
                    if current_distance < 10 and base_env.min_distance_achieved >= 10:
                        reward += 30
                
                # HUGE reward for completing the loop
                if base_env.loop_completed:
                    reward += 200  # Massive reward for phase 1 success
                    
                    # Store phase completion stats
                    if hasattr(self, 'phase_rewards'):
                        self.phase_rewards['phase1_loop_bonus'] = 200
        else:
            # Penalty for invalid action
            reward -= 1
        
        return reward
    
    def _phase2_reward(self, success):
        """
        Phase 2: Explore while maintaining ability to return
        - Continue rewarding loop completion
        - Add rewards for track length
        - Keep distance-based rewards but reduced
        """
        base_env = self._get_base_env()
        reward = 0
        
        if success:
            reward += 0.5
            
            if base_env.last_action == base_env.action_space.n - 1:
                reward -= 1
            else:
                # Distance rewards (reduced from phase 1)
                current_distance = base_env._calculate_distance_to_start()[0]
                
                if len(base_env.position_history) > 1:
                    prev_distance = np.linalg.norm(
                        np.array(base_env.position_history[-2]) - np.array(base_env.goal_position)
                    )
                    distance_delta = prev_distance - current_distance
                    
                    if distance_delta > 0:
                        reward += distance_delta * 1.5  # Reduced from phase 1
                    else:
                        reward += distance_delta * 0.3
                
                # Track length rewards (new in phase 2)
                if base_env.track_length > 10:
                    reward += 0.2  # Small reward for building
                if base_env.track_length > 20:
                    reward += 0.3
                if base_env.track_length > 30:
                    reward += 0.5
                
                # Distance checkpoints
                if current_distance < base_env.min_distance_achieved:
                    base_env.min_distance_achieved = current_distance
                    if current_distance < 30:
                        reward += 10
                    if current_distance < 20:
                        reward += 15
                    if current_distance < 10:
                        reward += 20
                
                # Loop completion (still important)
                if base_env.loop_completed:
                    reward += 150  # Still high but less than phase 1
                    
                    # Bonus for longer tracks
                    if base_env.track_length > 30:
                        reward += (base_env.track_length - 30) * 1.0
        else:
            reward -= 1
        
        return reward
    
    def _phase3_reward(self, success):
        """
        Phase 3: Full complexity - use original reward structure
        with all components enabled
        """
        base_env = self._get_base_env()
        # Use the original reward calculation
        return base_env._original_calculate_reward(success)
    
    def _check_phase_advancement(self):
        """Check if we should advance to the next phase"""
        if not self._track_stats or len(self.episode_results) < 50:
            return False
        
        success_rate = sum(self.episode_results) / len(self.episode_results)
        
        if self.current_phase == 1 and success_rate >= self.phase1_success_threshold:
            # Advance to phase 2
            self._advance_to_phase(2)
            return True
        elif self.current_phase == 2 and success_rate >= self.phase2_success_threshold:
            # Advance to phase 3
            self._advance_to_phase(3)
            return True
        elif self.current_phase == 3:
            # Handle phase 3 sub-stage progression
            if (success_rate >= self.phase3_success_threshold and 
                self.phase3_current_length < self.phase3_target_length):
                
                # Record current sub-stage completion
                self.phases_completed.append({
                    'phase': f"3.{self.phase3_stage}",
                    'max_length': self.phase3_current_length,
                    'success_rate': success_rate,
                    'episodes': self.phase_episode_count
                })
                
                # Increase difficulty within phase 3
                self.phase3_current_length = min(
                    self.phase3_current_length + self.phase3_increase_step,
                    self.phase3_target_length
                )
                self.phase3_stage += 1
                self._update_phase_settings()
                
                # Clear history for new sub-stage
                self.episode_results.clear()
                self.phase_episode_count = 0
                
                if self.verbose >= 1:
                    print(f"\n{'='*60}")
                    print(f"📈 PHASE 3: Advancing to sub-stage {self.phase3_stage}")
                    print(f"   Max track length: {self.phase3_current_length}")
                    print(f"   Success rate achieved: {success_rate:.1%}")
                    print(f"{'='*60}\n")
                
                return True
        
        return False
    
    def _advance_to_phase(self, new_phase):
        """Advance to a new phase"""
        # Record completion of current phase
        success_rate = sum(self.episode_results) / len(self.episode_results) if self.episode_results else 0
        self.phases_completed.append({
            'phase': self.current_phase,
            'success_rate': success_rate,
            'episodes': self.phase_episode_count,
            'total_loops': self.total_loops_completed
        })
        
        # Update phase
        self.current_phase = new_phase
        self.phase_episode_count = 0
        self.episode_results.clear()
        
        # Update environment settings
        self._update_phase_settings()
        
        if self.verbose >= 1:
            print(f"\n{'='*70}")
            print(f"🎯 ADVANCING TO PHASE {new_phase}")
            if new_phase == 2:
                print("   Phase 2: Explore & Return")
                print("   - Now encouraging longer tracks")
                print("   - Maintaining focus on loop completion")
                print(f"   - Max track length: {self.phase2_max_length}")
            elif new_phase == 3:
                print("   Phase 3: Build Quality")
                print("   - Full reward structure enabled")
                print("   - Progressive difficulty increase")
                print(f"   - Starting max length: {self.phase3_current_length}")
            print(f"   Previous phase success rate: {success_rate:.1%}")
            print(f"{'='*70}\n")
    
    def reset(self, **kwargs):
        """Reset environment and check for phase advancement"""
        # Check phase advancement before reset
        self._check_phase_advancement()
        
        # Reset the environment
        obs, info = self.env.reset(**kwargs)
        
        # Add phase info
        info['learning_phase'] = self.current_phase
        if self.current_phase == 3:
            info['phase3_stage'] = self.phase3_stage
            info['max_track_length'] = self.phase3_current_length
        elif self.current_phase == 2:
            info['max_track_length'] = self.phase2_max_length
        else:
            info['max_track_length'] = self.phase1_max_length
        
        info['episodes_in_phase'] = self.phase_episode_count
        info['phase_success_rate'] = (
            sum(self.episode_results) / len(self.episode_results)
            if self.episode_results else 0
        )
        
        # Store rewards dict for tracking
        self.phase_rewards = {}
        
        return obs, info
    
    def step(self, action):
        """Execute action and track performance"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Track episode completion
        if (terminated or truncated) and self._track_stats:
            self.episode_count += 1
            self.phase_episode_count += 1
            
            # Check if loop was completed
            base_env = self._get_base_env()
            success = base_env.loop_completed if hasattr(base_env, 'loop_completed') else False
            self.episode_results.append(success)
            
            if success:
                self.total_loops_completed += 1
            
            # Add phase info to episode info
            info['learning_phase'] = self.current_phase
            info['phase_success'] = success
            info['phase_success_rate'] = (
                sum(self.episode_results) / len(self.episode_results)
                if self.episode_results else 0
            )
            
            # Add phase-specific rewards breakdown
            if hasattr(self, 'phase_rewards'):
                info['phase_rewards'] = self.phase_rewards
            
            # Provide feedback at regular intervals
            if self.phase_episode_count % 10 == 0 and self.verbose >= 1:
                success_rate = sum(self.episode_results) / len(self.episode_results) if self.episode_results else 0
                phase_name = {
                    1: "Find Home",
                    2: "Explore & Return", 
                    3: f"Build Quality (stage {self.phase3_stage})"
                }.get(self.current_phase, f"Phase {self.current_phase}")
                
                print(f"📊 Phase {self.current_phase} ({phase_name}): "
                      f"Success: {success_rate:.1%} "
                      f"({sum(self.episode_results)}/{len(self.episode_results)}) "
                      f"Total loops: {self.total_loops_completed}")
        
        return obs, reward, terminated, truncated, info
    
    @contextmanager
    def evaluation_mode(self):
        """Context manager to temporarily disable statistics tracking during evaluation"""
        old_track_stats = self._track_stats
        self._track_stats = False
        try:
            yield
        finally:
            self._track_stats = old_track_stats
    
    def get_phase_stats(self):
        """Get statistics about phase progression"""
        return {
            'current_phase': self.current_phase,
            'phase3_stage': self.phase3_stage if self.current_phase == 3 else None,
            'total_episodes': self.episode_count,
            'phase_episodes': self.phase_episode_count,
            'success_rate': sum(self.episode_results) / len(self.episode_results) if self.episode_results else 0,
            'total_loops_completed': self.total_loops_completed,
            'phases_completed': self.phases_completed,
            'current_max_length': (
                self.phase1_max_length if self.current_phase == 1 else
                self.phase2_max_length if self.current_phase == 2 else
                self.phase3_current_length
            )
        }