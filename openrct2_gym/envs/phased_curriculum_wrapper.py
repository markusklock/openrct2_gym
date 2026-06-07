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
                 phase2_success_threshold=0.6,  # 60% success rate to advance from phase 2
                 window_size=100,
                 # Track length curriculum
                 phase1_max_length=40,  # More room for exploration while learning to return
                 phase2_max_length=80,  # Longer tracks for exploration and chain lift building
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
            phase1_max_length: Maximum track length in phase 1 (default: 40)
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
        self.episode_qualified_results = deque(maxlen=window_size)  # Phase 2: success with chain lift requirement
        self.episode_count = 0
        self.phase_episode_count = 0
        self.total_loops_completed = 0

        # Phase 3 sub-stages
        self.phase3_stage = 1
        
        # Statistics tracking control
        self._track_stats = True
        
        # Verbosity
        self.verbose = verbose
        
        # Exploit prevention: track repetitive actions
        self.recent_action_positions = deque(maxlen=10)
        self.repetition_penalty_count = 0
        
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
            # Skip ride testing in phase 1 - just learning to build circuits
            base_env.skip_ride_testing = True
        elif self.current_phase == 2:
            base_env.max_track_length = self.phase2_max_length
            base_env._calculate_reward = self._phase2_reward
            # Skip ride testing in phase 2 - still learning basic track building
            base_env.skip_ride_testing = True
        else:  # Phase 3
            base_env.max_track_length = self.phase3_current_length
            base_env._calculate_reward = self._phase3_reward
            # Enable ride testing in phase 3 - now building proper coasters
            base_env.skip_ride_testing = False
        
        if self.verbose >= 1:
            print(f"📚 Phase {self.current_phase} settings applied: max_length={base_env.max_track_length}, skip_testing={base_env.skip_ride_testing}")
    
    def _phase1_reward(self, success, action):
        """
        Phase 1: Focus entirely on finding way back to station
        - Strong distance-based rewards
        - Large bonus for completing loop
        - No rewards for track length or other metrics
        """
        base_env = self._get_base_env()

        # Special case: For remove actions, use symmetric removal from base env
        if success and action == 31:  # Remove piece
            if base_env.piece_rewards:
                # Return the exact negative of what was earned when placing this piece
                removed_reward = base_env.piece_rewards.pop()
                if base_env.verbose >= 2:
                    print(f"Phase1 Remove: reversing reward of {removed_reward:.2f}")
                return -removed_reward
            else:
                # No pieces to remove, small penalty
                return -1

        reward = 0
        
        # Basic reward for successful placement
        if success:
            reward += 0.5  # Small reward for valid placement
            
            # Track action position for repetition detection
            current_pos_tuple = tuple(base_env.current_position)
            action_type = 'remove' if base_env.last_action == base_env.action_space.n - 1 else 'place'
            
            # Check for repetitive place/remove pattern at same position
            if len(self.recent_action_positions) >= 4:
                # Check if we're doing remove-place-remove-place at same position
                recent = list(self.recent_action_positions)[-4:]
                if (len(recent) == 4 and 
                    recent[0] == (current_pos_tuple, 'place') and
                    recent[1] == (current_pos_tuple, 'remove') and
                    recent[2] == (current_pos_tuple, 'place') and
                    action_type == 'remove'):
                    # Heavy penalty for exploitative pattern
                    self.repetition_penalty_count += 1
                    reward -= 10 * self.repetition_penalty_count  # Escalating penalty
                    if self.verbose >= 1:
                        print(f"⚠️ Repetitive action detected! Penalty: {-10 * self.repetition_penalty_count}")
            
            # Add current action to history
            self.recent_action_positions.append((current_pos_tuple, action_type))
            
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
                        
                        # Extra bonus for getting very close - but only if it's a NEW minimum
                        if not hasattr(base_env, 'min_distance_reached'):
                            base_env.min_distance_reached = float('inf')
                        
                        if current_distance < base_env.min_distance_reached:
                            if current_distance < 10:
                                reward += (10 - current_distance) * 2.0
                            elif current_distance < 20:
                                reward += (20 - current_distance) * 0.5
                    else:
                        # Penalty for moving away
                        reward += distance_delta * 0.5  # Gentle penalty
                
                # Distance checkpoint rewards
                if not hasattr(base_env, 'min_distance_reached'):
                    base_env.min_distance_reached = float('inf')
                    
                if current_distance < base_env.min_distance_reached:
                    prev_min = base_env.min_distance_reached
                    base_env.min_distance_reached = current_distance
                    
                    # Checkpoint rewards for crossing distance thresholds
                    if current_distance < 50 and prev_min >= 50:
                        reward += 10
                    if current_distance < 30 and prev_min >= 30:
                        reward += 15
                    if current_distance < 20 and prev_min >= 20:
                        reward += 20
                    if current_distance < 10 and prev_min >= 10:
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
    
    def _phase2_reward(self, success, action):
        """
        Phase 2: Build chain lifts and explore with longer tracks
        - Strong incentives for chain lifts early in the track
        - Rewards for building longer tracks (up to 80 pieces)
        - Minimal distance penalties to encourage exploration
        - Variety and height variation rewards
        """
        base_env = self._get_base_env()

        # Special case: For remove actions, use symmetric removal from base env
        if success and action == 31:  # Remove piece
            if base_env.piece_rewards:
                # Check if the piece being removed was a chain lift
                if hasattr(base_env, 'track_pieces') and base_env.track_pieces:
                    removed_piece = base_env.track_pieces[-1]  # Last piece in the track
                    if removed_piece in [9, 10]:  # Was it a chain lift?
                        if hasattr(base_env, 'phase2_chain_lift_count') and base_env.phase2_chain_lift_count > 0:
                            base_env.phase2_chain_lift_count -= 1
                            if base_env.verbose >= 2:
                                print(f"Phase2: Chain lift removed, count now: {base_env.phase2_chain_lift_count}")

                # Return the exact negative of what was earned when placing this piece
                removed_reward = base_env.piece_rewards.pop()
                if base_env.verbose >= 2:
                    print(f"Phase2 Remove: reversing reward of {removed_reward:.2f}")
                return -removed_reward
            else:
                # No pieces to remove, small penalty
                return -1

        reward = 0

        if success:
            reward += 0.5  # Base success reward

            if action != 31:  # Not a remove action

                # CHAIN LIFT INCENTIVES - Primary focus of Phase 2
                if action in [9, 10]:  # Chain lift pieces
                    # Track total chain lifts (no position tracking needed - symmetric removal prevents exploitation)
                    if not hasattr(base_env, 'phase2_chain_lift_count'):
                        base_env.phase2_chain_lift_count = 0

                    base_env.phase2_chain_lift_count += 1

                    # Strong rewards for chain lifts placed early
                    if base_env.track_length <= 15:
                        reward += 10  # Big reward for early chain lifts
                        if base_env.verbose >= 2:
                            print(f"Phase2: Early chain lift #{base_env.phase2_chain_lift_count} at piece {base_env.track_length}, +10 reward")
                    elif base_env.track_length <= 30:
                        reward += 5  # Moderate reward for mid-track chain lifts
                    else:
                        reward += 2  # Small reward for late chain lifts

                    # Milestone bonuses for total chain lifts
                    if base_env.phase2_chain_lift_count == 1:
                        reward += 5  # First chain lift bonus
                    elif base_env.phase2_chain_lift_count == 3:
                        reward += 10  # Three chain lifts bonus
                    elif base_env.phase2_chain_lift_count == 5:
                        reward += 15  # Five chain lifts bonus

                    # Height bonus for chain lifts going higher
                    if base_env.current_position[2] > 20:
                        reward += min((base_env.current_position[2] - 20) * 0.5, 5)  # Extra reward for height

                # TRACK LENGTH REWARDS - Encourage building longer tracks
                # Progressive milestones up to 80 pieces
                if base_env.track_length == 10:
                    reward += 2
                elif base_env.track_length == 20:
                    reward += 3
                elif base_env.track_length == 30:
                    reward += 4
                elif base_env.track_length == 40:
                    reward += 5
                elif base_env.track_length == 50:
                    reward += 6
                elif base_env.track_length == 60:
                    reward += 8
                elif base_env.track_length == 70:
                    reward += 10

                # Continuous building reward after 30 pieces
                if base_env.track_length > 30:
                    reward += 0.2  # Consistent reward for each new piece

                # VARIETY REWARDS - Encourage diverse track building
                # Track piece type variety
                if not hasattr(base_env, 'phase2_used_pieces'):
                    base_env.phase2_used_pieces = set()
                if action not in base_env.phase2_used_pieces:
                    base_env.phase2_used_pieces.add(action)
                    reward += 0.5  # Bonus for using new piece types

                # Height variation reward
                if action in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]:  # Slope pieces
                    if not hasattr(base_env, 'phase2_last_height'):
                        base_env.phase2_last_height = base_env.current_position[2]
                    height_change = abs(base_env.current_position[2] - base_env.phase2_last_height)
                    if height_change > 0:
                        reward += min(height_change * 0.4, 3.0)  # Reward height changes
                    base_env.phase2_last_height = base_env.current_position[2]

                # EXPLORATION BONUS - Reward reaching new positions
                current_pos_tuple = tuple(base_env.current_position)
                if not hasattr(self, 'phase2_visited_positions'):
                    self.phase2_visited_positions = set()
                if current_pos_tuple not in self.phase2_visited_positions:
                    reward += 0.3  # Bonus for exploring new areas
                    self.phase2_visited_positions.add(current_pos_tuple)

                # DISTANCE REWARDS - Greatly reduced to avoid conflicting signals
                current_distance = base_env._calculate_distance_to_start()[0]

                if len(base_env.position_history) > 1:
                    prev_distance = np.linalg.norm(
                        np.array(base_env.position_history[-2]) - np.array(base_env.goal_position)
                    )
                    distance_delta = prev_distance - current_distance

                    if distance_delta > 0:
                        reward += distance_delta * 0.3  # Small reward for getting closer
                    else:
                        reward += distance_delta * 0.05  # Very gentle penalty for moving away

                # Minimal distance checkpoints (only when very close)
                if not hasattr(base_env, 'min_distance_reached'):
                    base_env.min_distance_reached = float('inf')

                if current_distance < base_env.min_distance_reached:
                    prev_min = base_env.min_distance_reached
                    base_env.min_distance_reached = current_distance

                    # Only reward when getting very close (to help with loop completion)
                    if current_distance < 10 and prev_min >= 10:
                        reward += 5

                # LOOP COMPLETION - Still important with length bonuses
                if base_env.loop_completed:
                    reward += 100  # Reduced from 150 to balance with other rewards

                    # Bonus for chain lifts in completed track
                    if hasattr(base_env, 'phase2_chain_lift_count'):
                        reward += base_env.phase2_chain_lift_count * 5  # +5 per chain lift

                    # Strong bonus for longer completed tracks
                    if base_env.track_length > 40:
                        reward += (base_env.track_length - 40) * 2.0
                    if base_env.track_length > 60:
                        reward += (base_env.track_length - 60) * 3.0  # Extra bonus for very long tracks
        else:
            # Small penalty for invalid action
            reward -= 0.5

        return reward
    
    def _phase3_reward(self, success, action):
        """
        Phase 3: Build quality rides - optimize for ride statistics
        - Uses base reward structure but scaled down (0.5x)
        - Primary focus on ride quality metrics at episode completion
        - Ride stats bonus is added separately in wrapper's step function
        """
        base_env = self._get_base_env()

        # Special case: For remove actions, use symmetric removal from base env
        if success and action == 31:  # Remove piece
            if base_env.piece_rewards:
                # Return the exact negative of what was earned when placing this piece
                removed_reward = base_env.piece_rewards.pop()
                if base_env.verbose >= 2:
                    print(f"Phase3 Remove: reversing reward of {removed_reward:.2f}")
                return -removed_reward
            else:
                # No pieces to remove, small penalty
                return -1

        # Get the original reward and scale it down
        # We scale it because ride quality will be the primary signal in Phase 3
        original_reward = base_env._original_calculate_reward(success, action)

        # Scale down base rewards (but not too much - still need guidance)
        # Loop completion bonus is already in original_reward (+500), reduce it
        if base_env.loop_completed:
            # Original gives +500 for completion, reduce to +200
            scaled_reward = (original_reward - 500) * 0.5 + 200
        else:
            # Scale other rewards moderately
            scaled_reward = original_reward * 0.5

        return scaled_reward

    def _calculate_ride_quality_bonus(self, excitement, intensity, nausea):
        """
        Calculate bonus reward based on ride quality metrics.
        Target ranges:
        - Excitement: 7.0-9.0 (high)
        - Intensity: 4.5-6.5 (medium-high)
        - Nausea: <4.5 (low-medium)

        Returns total bonus (can be up to ~400 points)
        """
        bonus = 0

        # EXCITEMENT SCORING (target: 7.0-9.0)
        if 7.0 <= excitement <= 9.0:
            # Perfect range - scale from +100 to +200 based on position in range
            # Peak at 8.0 (middle of range)
            distance_from_optimal = abs(excitement - 8.0)
            excitement_bonus = 200 - (distance_from_optimal * 50)  # Max 200 at 8.0, min 150 at edges
            bonus += excitement_bonus
        elif 5.0 <= excitement < 7.0:
            # Below target but acceptable
            bonus += (excitement - 5.0) / 2.0 * 100  # 0 to +100 scaled
        elif 9.0 < excitement <= 11.0:
            # Above target but acceptable
            bonus += (11.0 - excitement) / 2.0 * 100  # +100 to 0 scaled
        elif excitement < 5.0:
            # Too boring
            bonus += excitement * 10  # Small partial credit (0 to +50)
        else:
            # Way too exciting (>11.0)
            bonus += max(0, 50 - (excitement - 11.0) * 10)  # Penalty for extreme

        # INTENSITY SCORING (target: 4.5-6.5)
        if 4.5 <= intensity <= 6.5:
            # Perfect range - scale from +50 to +100 based on position
            # Peak at 5.5 (middle of range)
            distance_from_optimal = abs(intensity - 5.5)
            intensity_bonus = 100 - (distance_from_optimal * 25)  # Max 100 at 5.5, min 75 at edges
            bonus += intensity_bonus
        elif 3.0 <= intensity < 4.5:
            # Below target but acceptable
            bonus += (intensity - 3.0) / 1.5 * 50  # 0 to +50 scaled
        elif 6.5 < intensity <= 8.0:
            # Above target but still rideable
            bonus += (8.0 - intensity) / 1.5 * 50  # +50 to 0 scaled
        elif intensity < 3.0:
            # Too mild
            bonus += intensity * 10  # Small partial credit (0 to +30)
        else:
            # Too intense (>8.0) - penalty
            bonus -= (intensity - 8.0) * 10  # Penalty for excessive intensity

        # NAUSEA SCORING (target: <4.5, lower is better)
        if nausea < 2.0:
            # Excellent - very comfortable ride
            bonus += 100
        elif 2.0 <= nausea <= 4.5:
            # Acceptable range - scale from +100 to 0
            bonus += (4.5 - nausea) / 2.5 * 100  # +100 at 2.0, 0 at 4.5
        elif 4.5 < nausea <= 6.0:
            # Too nauseating but not terrible
            bonus -= (nausea - 4.5) / 1.5 * 50  # 0 to -50 penalty
        else:
            # Way too nauseating (>6.0) - heavy penalty
            bonus -= 50 + (nausea - 6.0) * 25  # Increasing penalty

        return bonus

    def _check_phase_advancement(self):
        """Check if we should advance to the next phase"""
        if not self._track_stats or len(self.episode_results) < 50:
            return False

        success_rate = sum(self.episode_results) / len(self.episode_results)

        if self.current_phase == 1 and success_rate >= self.phase1_success_threshold:
            # Advance to phase 2
            self._advance_to_phase(2)
            return True
        elif self.current_phase == 2 and len(self.episode_qualified_results) >= 50:
            # Phase 2 requires BOTH loop completion AND chain lift usage
            qualified_success_rate = sum(self.episode_qualified_results) / len(self.episode_qualified_results)
            if qualified_success_rate >= self.phase2_success_threshold:
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
                self.episode_qualified_results.clear()  # Clear this too to prevent stale data
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
        self.episode_qualified_results.clear()  # Clear qualified results too

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
                print("   Phase 3: Build Quality Rides")
                print("   - Optimize for ride statistics (Excitement, Intensity, Nausea)")
                print("   - Ride quality bonus up to ~400 points")
                print("   - Target: E=7-9, I=4.5-6.5, N<4.5")
                print(f"   - Starting max length: {self.phase3_current_length}")
            print(f"   Previous phase success rate: {success_rate:.1%}")
            print(f"{'='*70}\n")
    
    def reset(self, **kwargs):
        """Reset environment and check for phase advancement"""
        # Check phase advancement before reset
        self._check_phase_advancement()

        # Reset repetition tracking
        self.recent_action_positions.clear()
        self.repetition_penalty_count = 0

        # Reset Phase 2 specific tracking
        if hasattr(self, 'phase2_visited_positions'):
            self.phase2_visited_positions.clear()

        # Reset Phase 2 chain lift counter
        base_env = self._get_base_env()
        if hasattr(base_env, 'phase2_chain_lift_count'):
            base_env.phase2_chain_lift_count = 0

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

            # Phase 2 specific: Track "qualified" success (loop completion + chain lift requirement)
            if self.current_phase == 2:
                chain_lift_count = getattr(base_env, 'phase2_chain_lift_count', 0)
                qualified_success = success and chain_lift_count >= 8
                self.episode_qualified_results.append(qualified_success)

                if self.verbose >= 2 and success:
                    if qualified_success:
                        print(f"✅ Qualified success: Loop completed with {chain_lift_count} chain lifts")
                    else:
                        print(f"⚠️ Loop completed but only {chain_lift_count} chain lifts (need 8+, not counted for phase advancement)")

            if success:
                self.total_loops_completed += 1

            # Phase 3: Add ride quality bonus if ride test was performed
            if self.current_phase == 3 and success and 'ride_rating' in info:
                ride_rating = info['ride_rating']
                excitement = ride_rating.get('excitement', 0)
                intensity = ride_rating.get('intensity', 0)
                nausea = ride_rating.get('nausea', 0)

                # Calculate and add quality bonus
                quality_bonus = self._calculate_ride_quality_bonus(excitement, intensity, nausea)
                reward += quality_bonus

                # Log the bonus for transparency
                info['ride_quality_bonus'] = quality_bonus
                if self.verbose >= 1 and quality_bonus != 0:
                    print(f"🎢 Ride Quality: E={excitement:.1f} I={intensity:.1f} N={nausea:.1f} → Bonus: {quality_bonus:+.1f}")

                # Track best ride stats
                if not hasattr(self, 'best_ride_quality'):
                    self.best_ride_quality = quality_bonus
                    self.best_ride_stats = (excitement, intensity, nausea)
                elif quality_bonus > self.best_ride_quality:
                    self.best_ride_quality = quality_bonus
                    self.best_ride_stats = (excitement, intensity, nausea)
                    if self.verbose >= 1:
                        print(f"🏆 New best ride quality: {quality_bonus:.1f} points!")

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

                # Phase 2: Show both loop completion and qualified (with chain lifts) rates
                if self.current_phase == 2 and self.episode_qualified_results:
                    qualified_rate = sum(self.episode_qualified_results) / len(self.episode_qualified_results)
                    print(f"📊 Phase {self.current_phase} ({phase_name}): "
                          f"Success: {success_rate:.1%} | "
                          f"With Chain Lifts: {qualified_rate:.1%} "
                          f"({sum(self.episode_qualified_results)}/{len(self.episode_qualified_results)}) "
                          f"Total loops: {self.total_loops_completed}")
                else:
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