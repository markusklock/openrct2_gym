"""
Improved Phased Curriculum Learning Wrapper for OpenRCT2 Environment
Implements 5-phase progressive learning with physics-aware rewards.
"""
import gymnasium as gym
import numpy as np
from collections import deque
from contextlib import contextmanager
from typing import Dict, Any, Tuple


class ImprovedPhasedCurriculumWrapper(gym.Wrapper):
    """
    Wrapper that implements 5-phase curriculum learning with physics-aware rewards.

    Phase 1: "Return Practice" (25 pieces) - Focus on navigation
    Phase 2: "Lift Hill Building" (40 pieces) - Learn chain lifts and energy
    Phase 3: "Drop & Turn" (60 pieces) - Learn drops and turnarounds
    Phase 4: "Circuit Mastery" (80 pieces) - Full integration
    Phase 5: "Quality Optimization" (120 pieces) - Optimize ride ratings
    """

    def __init__(self, env,
                 # Phase progression thresholds
                 phase1_success_threshold=0.5,   # 50% loop completion
                 phase2_success_threshold=0.4,   # 40% with 3+ chain lifts
                 phase3_success_threshold=0.35,  # 35% with good patterns
                 phase4_success_threshold=0.30,  # 30% clean completions
                 phase5_success_threshold=0.25,  # 25% with quality ratings
                 window_size=100,
                 # Track length per phase
                 phase1_max_length=25,
                 phase2_max_length=40,
                 phase3_max_length=60,
                 phase4_max_length=80,
                 phase5_initial_length=80,
                 phase5_target_length=120,
                 phase5_increase_step=10,
                 # Verbosity
                 verbose=1):
        """
        Args:
            env: Base OpenRCT2 environment
            phase*_success_threshold: Success rate needed to advance from each phase
            window_size: Number of episodes to consider for success rate
            phase*_max_length: Maximum track length for each phase
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
        self.phase4_success_threshold = phase4_success_threshold
        self.phase5_success_threshold = phase5_success_threshold

        # Track length parameters per phase
        self.phase1_max_length = phase1_max_length
        self.phase2_max_length = phase2_max_length
        self.phase3_max_length = phase3_max_length
        self.phase4_max_length = phase4_max_length
        self.phase5_initial_length = phase5_initial_length
        self.phase5_target_length = phase5_target_length
        self.phase5_current_length = phase5_initial_length
        self.phase5_increase_step = phase5_increase_step

        # Performance tracking
        self.episode_results = deque(maxlen=window_size)
        self.episode_qualified_results = deque(maxlen=window_size)
        self.episode_count = 0
        self.phase_episode_count = 0
        self.total_loops_completed = 0

        # Phase 5 sub-stages
        self.phase5_stage = 1

        # Statistics tracking control
        self._track_stats = True

        # Verbosity
        self.verbose = verbose

        # Pattern tracking (to give one-time bonuses)
        self._lift_hill_rewarded = False
        self._drop_rewarded = False
        self._turnaround_rewarded = False

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

        phase_configs = {
            1: (self.phase1_max_length, self._phase1_reward, True),
            2: (self.phase2_max_length, self._phase2_reward, True),
            3: (self.phase3_max_length, self._phase3_reward, True),
            4: (self.phase4_max_length, self._phase4_reward, True),
            5: (self.phase5_current_length, self._phase5_reward, False),
        }

        max_length, reward_fn, skip_testing = phase_configs.get(
            self.current_phase,
            (self.phase5_current_length, self._phase5_reward, False)
        )

        base_env.max_track_length = max_length
        base_env._calculate_reward = reward_fn
        base_env.skip_ride_testing = skip_testing

        if self.verbose >= 1:
            phase_names = {
                1: "Return Practice",
                2: "Lift Hill Building",
                3: "Drop & Turn",
                4: "Circuit Mastery",
                5: "Quality Optimization"
            }
            print(f"📚 Phase {self.current_phase} ({phase_names.get(self.current_phase, '')}) "
                  f"settings applied: max_length={max_length}, skip_testing={skip_testing}")

    def _phase1_reward(self, success, action):
        """
        Phase 1: Return Practice - Focus entirely on navigation back to station.
        Pure distance-based rewards with loop completion bonus.
        """
        base_env = self._get_base_env()

        # Symmetric removal
        if success and action == 31:
            if base_env.piece_rewards:
                return -base_env.piece_rewards.pop()
            return -1

        reward = 0

        if success and action != 31:
            reward += 0.5  # Base success reward

            current_distance = base_env._calculate_distance_to_start()[0]

            # Strong distance-based rewards (4x multiplier)
            if base_env.previous_distance is not None:
                distance_delta = base_env.previous_distance - current_distance
                if distance_delta > 0:
                    reward += distance_delta * 4.0  # Strong reward for getting closer
                else:
                    reward += distance_delta * 0.5  # Gentle penalty for moving away

            # Proximity bonuses
            if current_distance < 20:
                reward += (20 - current_distance) * 0.3
            if current_distance < 10:
                reward += (10 - current_distance) * 0.5

            # Loop completion - main goal
            if base_env.loop_completed:
                reward += 300
        elif not success:
            reward -= 0.5

        return reward

    def _phase2_reward(self, success, action):
        """
        Phase 2: Lift Hill Building - Learn chain lifts and energy management.
        Strong incentives for chain lifts, energy tracking, reduced distance rewards.
        """
        base_env = self._get_base_env()

        # Symmetric removal
        if success and action == 31:
            if base_env.piece_rewards:
                # Track chain lift removal
                if base_env.track_pieces and base_env.track_pieces[-1] in [9, 10]:
                    if hasattr(base_env, 'phase2_chain_lift_count') and base_env.phase2_chain_lift_count > 0:
                        base_env.phase2_chain_lift_count -= 1
                return -base_env.piece_rewards.pop()
            return -1

        reward = 0

        if success and action != 31:
            reward += 0.5  # Base success reward

            # CHAIN LIFT REWARDS - Primary focus
            if action in [9, 10]:
                if not hasattr(base_env, 'phase2_chain_lift_count'):
                    base_env.phase2_chain_lift_count = 0
                base_env.phase2_chain_lift_count += 1

                # Very strong rewards for chain lifts
                if base_env.track_length <= 15:
                    reward += 15  # Big reward for early chain lifts
                elif base_env.track_length <= 25:
                    reward += 10
                else:
                    reward += 5

                # Consecutive chain lift bonus
                if len(base_env.track_pieces) >= 2 and base_env.track_pieces[-2] in [9, 10]:
                    reward += 5  # Building a proper sequence

            # ENERGY REWARDS
            energy_reward = base_env._calculate_energy_reward()
            reward += energy_reward * 2.0  # Double weight in this phase

            # LIFT HILL PATTERN (one-time)
            if not self._lift_hill_rewarded:
                lift_hill_score = base_env._detect_lift_hill_pattern()
                if lift_hill_score >= 0.5:
                    reward += 20
                    self._lift_hill_rewarded = True

            # Reduced distance rewards (0.5x)
            current_distance = base_env._calculate_distance_to_start()[0]
            if base_env.previous_distance is not None:
                distance_delta = base_env.previous_distance - current_distance
                if distance_delta > 0:
                    reward += distance_delta * 0.5
                else:
                    reward += distance_delta * 0.1

            # Loop completion with chain lift requirement
            if base_env.loop_completed:
                chain_count = getattr(base_env, 'phase2_chain_lift_count', 0)
                reward += 150
                reward += chain_count * 10  # Bonus per chain lift
        elif not success:
            reward -= 0.5

        return reward

    def _phase3_reward(self, success, action):
        """
        Phase 3: Drop & Turn - Learn drops after lift hills and turnarounds.
        Rewards for drop patterns, turnarounds, continued energy management.
        """
        base_env = self._get_base_env()

        # Symmetric removal
        if success and action == 31:
            if base_env.piece_rewards:
                return -base_env.piece_rewards.pop()
            return -1

        reward = 0

        if success and action != 31:
            reward += 0.5

            # DROP PATTERN REWARDS
            if action in [6, 8, 12, 14]:  # Drop pieces
                if base_env.track_length > 10 and base_env.track_length < 45:
                    reward += 5  # Reward drops after expected lift hill

                    # Drop pattern bonus (one-time)
                    if not self._drop_rewarded:
                        drop_score = base_env._detect_drop_pattern()
                        if drop_score >= 0.5:
                            reward += 15
                            self._drop_rewarded = True

            # TURNAROUND REWARDS (one-time)
            if not self._turnaround_rewarded:
                turnaround_score = base_env._detect_turnaround()
                if turnaround_score >= 0.7:
                    reward += 20
                    self._turnaround_rewarded = True

            # Energy management (moderate weight)
            energy_reward = base_env._calculate_energy_reward()
            reward += energy_reward * 1.5

            # Distance rewards (moderate)
            current_distance = base_env._calculate_distance_to_start()[0]
            if base_env.previous_distance is not None:
                distance_delta = base_env.previous_distance - current_distance
                if distance_delta > 0:
                    reward += distance_delta * 1.5
                else:
                    reward += distance_delta * 0.3

            # Loop completion with pattern bonuses
            if base_env.loop_completed:
                reward += 200
                # Pattern bonuses
                if self._lift_hill_rewarded:
                    reward += 30
                if self._drop_rewarded:
                    reward += 20
        elif not success:
            reward -= 0.5

        return reward

    def _phase4_reward(self, success, action):
        """
        Phase 4: Circuit Mastery - Full integration of all skills.
        All rewards active, approach guidance emphasized.
        """
        base_env = self._get_base_env()

        # Symmetric removal
        if success and action == 31:
            if base_env.piece_rewards:
                return -base_env.piece_rewards.pop()
            return -1

        reward = 0

        if success and action != 31:
            reward += 0.5

            # Energy reward
            energy_reward = base_env._calculate_energy_reward()
            reward += energy_reward

            # Pattern rewards (all types, scaled down since probably already earned)
            if not self._lift_hill_rewarded:
                if base_env._detect_lift_hill_pattern() >= 0.5:
                    reward += 10
                    self._lift_hill_rewarded = True

            if not self._drop_rewarded:
                if base_env._detect_drop_pattern() >= 0.5:
                    reward += 8
                    self._drop_rewarded = True

            if not self._turnaround_rewarded:
                if base_env._detect_turnaround() >= 0.7:
                    reward += 10
                    self._turnaround_rewarded = True

            # APPROACH GUIDANCE (emphasized in this phase)
            approach_reward = base_env._calculate_approach_reward(action)
            reward += approach_reward * 2.0  # Double weight

            # Distance rewards (standard)
            current_distance = base_env._calculate_distance_to_start()[0]
            if base_env.previous_distance is not None:
                distance_delta = base_env.previous_distance - current_distance
                if distance_delta > 0:
                    reward += distance_delta * 2.0
                else:
                    reward += distance_delta * 0.5

            # Variety bonus
            if action not in base_env.used_piece_types:
                reward += 0.5

            # Loop completion with full bonuses
            if base_env.loop_completed:
                reward += 350
                # Track length bonus
                if base_env.track_length > 50:
                    reward += (base_env.track_length - 50) * 1.0
        elif not success:
            reward -= 0.3

        return reward

    def _phase5_reward(self, success, action):
        """
        Phase 5: Quality Optimization - Optimize for ride statistics.
        Scaled down base rewards, ride quality bonus added in step().
        """
        base_env = self._get_base_env()

        # Symmetric removal
        if success and action == 31:
            if base_env.piece_rewards:
                return -base_env.piece_rewards.pop()
            return -1

        # Use phase 4 rewards scaled to 0.6x
        # Ride quality bonus is added separately in step()
        reward = 0

        if success and action != 31:
            reward += 0.3

            # All rewards at reduced scale
            energy_reward = base_env._calculate_energy_reward()
            reward += energy_reward * 0.6

            approach_reward = base_env._calculate_approach_reward(action)
            reward += approach_reward * 0.6

            current_distance = base_env._calculate_distance_to_start()[0]
            if base_env.previous_distance is not None:
                distance_delta = base_env.previous_distance - current_distance
                if distance_delta > 0:
                    reward += distance_delta * 1.2
                else:
                    reward += distance_delta * 0.3

            if base_env.loop_completed:
                reward += 200  # Reduced completion bonus
        elif not success:
            reward -= 0.2

        return reward

    def _calculate_ride_quality_bonus(self, excitement, intensity, nausea):
        """
        Calculate bonus reward based on ride quality metrics.
        Target ranges:
        - Excitement: 7.0-9.0 (high)
        - Intensity: 4.5-6.5 (medium-high)
        - Nausea: <4.5 (low-medium)
        """
        bonus = 0

        # EXCITEMENT (target: 7.0-9.0, peak at 8.0)
        if 7.0 <= excitement <= 9.0:
            distance_from_optimal = abs(excitement - 8.0)
            bonus += 200 - (distance_from_optimal * 50)
        elif 5.0 <= excitement < 7.0:
            bonus += (excitement - 5.0) / 2.0 * 100
        elif 9.0 < excitement <= 11.0:
            bonus += (11.0 - excitement) / 2.0 * 100
        elif excitement < 5.0:
            bonus += excitement * 10
        else:
            bonus += max(0, 50 - (excitement - 11.0) * 10)

        # INTENSITY (target: 4.5-6.5, peak at 5.5)
        if 4.5 <= intensity <= 6.5:
            distance_from_optimal = abs(intensity - 5.5)
            bonus += 100 - (distance_from_optimal * 25)
        elif 3.0 <= intensity < 4.5:
            bonus += (intensity - 3.0) / 1.5 * 50
        elif 6.5 < intensity <= 8.0:
            bonus += (8.0 - intensity) / 1.5 * 50
        elif intensity < 3.0:
            bonus += intensity * 10
        else:
            bonus -= (intensity - 8.0) * 10

        # NAUSEA (target: <4.5, lower is better)
        if nausea < 2.0:
            bonus += 100
        elif 2.0 <= nausea <= 4.5:
            bonus += (4.5 - nausea) / 2.5 * 100
        elif 4.5 < nausea <= 6.0:
            bonus -= (nausea - 4.5) / 1.5 * 50
        else:
            bonus -= 50 + (nausea - 6.0) * 25

        return bonus

    def _check_phase_advancement(self):
        """Check if we should advance to the next phase"""
        if not self._track_stats or len(self.episode_results) < 50:
            return False

        success_rate = sum(self.episode_results) / len(self.episode_results)
        base_env = self._get_base_env()

        if self.current_phase == 1:
            if success_rate >= self.phase1_success_threshold:
                self._advance_to_phase(2)
                return True

        elif self.current_phase == 2:
            # Need loop completion with chain lifts
            if len(self.episode_qualified_results) >= 50:
                qualified_rate = sum(self.episode_qualified_results) / len(self.episode_qualified_results)
                if qualified_rate >= self.phase2_success_threshold:
                    self._advance_to_phase(3)
                    return True

        elif self.current_phase == 3:
            # Need loop completion with patterns
            if len(self.episode_qualified_results) >= 50:
                qualified_rate = sum(self.episode_qualified_results) / len(self.episode_qualified_results)
                if qualified_rate >= self.phase3_success_threshold:
                    self._advance_to_phase(4)
                    return True

        elif self.current_phase == 4:
            if success_rate >= self.phase4_success_threshold:
                self._advance_to_phase(5)
                return True

        elif self.current_phase == 5:
            # Handle phase 5 sub-stage progression
            if (success_rate >= self.phase5_success_threshold and
                self.phase5_current_length < self.phase5_target_length):

                self.phases_completed.append({
                    'phase': f"5.{self.phase5_stage}",
                    'max_length': self.phase5_current_length,
                    'success_rate': success_rate,
                    'episodes': self.phase_episode_count
                })

                self.phase5_current_length = min(
                    self.phase5_current_length + self.phase5_increase_step,
                    self.phase5_target_length
                )
                self.phase5_stage += 1
                self._update_phase_settings()

                self.episode_results.clear()
                self.episode_qualified_results.clear()
                self.phase_episode_count = 0

                if self.verbose >= 1:
                    print(f"\n{'='*60}")
                    print(f"📈 PHASE 5: Advancing to sub-stage {self.phase5_stage}")
                    print(f"   Max track length: {self.phase5_current_length}")
                    print(f"   Success rate achieved: {success_rate:.1%}")
                    print(f"{'='*60}\n")

                return True

        return False

    def _advance_to_phase(self, new_phase):
        """Advance to a new phase"""
        success_rate = sum(self.episode_results) / len(self.episode_results) if self.episode_results else 0
        self.phases_completed.append({
            'phase': self.current_phase,
            'success_rate': success_rate,
            'episodes': self.phase_episode_count,
            'total_loops': self.total_loops_completed
        })

        self.current_phase = new_phase
        self.phase_episode_count = 0
        self.episode_results.clear()
        self.episode_qualified_results.clear()

        self._update_phase_settings()

        if self.verbose >= 1:
            phase_names = {
                2: ("Lift Hill Building", "Learning chain lifts and energy management"),
                3: ("Drop & Turn", "Learning drops and turnarounds"),
                4: ("Circuit Mastery", "Full integration of all skills"),
                5: ("Quality Optimization", "Optimizing for ride ratings")
            }
            name, desc = phase_names.get(new_phase, (f"Phase {new_phase}", ""))

            print(f"\n{'='*70}")
            print(f"🎯 ADVANCING TO PHASE {new_phase}: {name}")
            print(f"   {desc}")
            print(f"   Previous phase success rate: {success_rate:.1%}")
            print(f"{'='*70}\n")

    def reset(self, **kwargs):
        """Reset environment and check for phase advancement"""
        self._check_phase_advancement()

        # Reset pattern tracking for new episode
        self._lift_hill_rewarded = False
        self._drop_rewarded = False
        self._turnaround_rewarded = False

        # Reset phase-specific tracking
        base_env = self._get_base_env()
        if hasattr(base_env, 'phase2_chain_lift_count'):
            base_env.phase2_chain_lift_count = 0

        obs, info = self.env.reset(**kwargs)

        # Add phase info
        info['learning_phase'] = self.current_phase
        info['phase_name'] = {
            1: "Return Practice",
            2: "Lift Hill Building",
            3: "Drop & Turn",
            4: "Circuit Mastery",
            5: "Quality Optimization"
        }.get(self.current_phase, f"Phase {self.current_phase}")

        if self.current_phase == 5:
            info['phase5_stage'] = self.phase5_stage
            info['max_track_length'] = self.phase5_current_length
        else:
            info['max_track_length'] = getattr(self, f'phase{self.current_phase}_max_length')

        info['episodes_in_phase'] = self.phase_episode_count
        info['phase_success_rate'] = (
            sum(self.episode_results) / len(self.episode_results)
            if self.episode_results else 0
        )

        return obs, info

    def step(self, action):
        """Execute action and track performance"""
        obs, reward, terminated, truncated, info = self.env.step(action)

        if (terminated or truncated) and self._track_stats:
            self.episode_count += 1
            self.phase_episode_count += 1

            base_env = self._get_base_env()
            success = getattr(base_env, 'loop_completed', False)
            self.episode_results.append(success)

            # Phase-specific qualified success tracking
            if self.current_phase == 2:
                # Need chain lifts
                chain_count = getattr(base_env, 'phase2_chain_lift_count', 0)
                qualified = success and chain_count >= 3
                self.episode_qualified_results.append(qualified)

            elif self.current_phase == 3:
                # Need patterns
                has_patterns = self._lift_hill_rewarded or self._drop_rewarded
                qualified = success and has_patterns
                self.episode_qualified_results.append(qualified)

            if success:
                self.total_loops_completed += 1

            # Phase 5: Add ride quality bonus
            if self.current_phase == 5 and success and 'ride_rating' in info:
                ride_rating = info['ride_rating']
                excitement = ride_rating.get('excitement', 0)
                intensity = ride_rating.get('intensity', 0)
                nausea = ride_rating.get('nausea', 0)

                quality_bonus = self._calculate_ride_quality_bonus(excitement, intensity, nausea)
                reward += quality_bonus

                info['ride_quality_bonus'] = quality_bonus
                if self.verbose >= 1 and quality_bonus != 0:
                    print(f"🎢 Ride Quality: E={excitement:.1f} I={intensity:.1f} N={nausea:.1f} → Bonus: {quality_bonus:+.1f}")

            # Add phase info
            info['learning_phase'] = self.current_phase
            info['phase_success'] = success
            info['phase_success_rate'] = (
                sum(self.episode_results) / len(self.episode_results)
                if self.episode_results else 0
            )

            # Periodic logging
            if self.phase_episode_count % 10 == 0 and self.verbose >= 1:
                success_rate = info['phase_success_rate']
                phase_names = {
                    1: "Return Practice",
                    2: "Lift Hill Building",
                    3: "Drop & Turn",
                    4: "Circuit Mastery",
                    5: f"Quality Opt (stage {self.phase5_stage})"
                }
                phase_name = phase_names.get(self.current_phase, f"Phase {self.current_phase}")

                qualified_str = ""
                if self.current_phase in [2, 3] and self.episode_qualified_results:
                    qualified_rate = sum(self.episode_qualified_results) / len(self.episode_qualified_results)
                    qualified_str = f" | Qualified: {qualified_rate:.1%}"

                print(f"📊 Phase {self.current_phase} ({phase_name}): "
                      f"Success: {success_rate:.1%}{qualified_str} "
                      f"Total loops: {self.total_loops_completed}")

        return obs, reward, terminated, truncated, info

    @contextmanager
    def evaluation_mode(self):
        """Context manager to disable statistics tracking during evaluation"""
        old_track_stats = self._track_stats
        self._track_stats = False
        try:
            yield
        finally:
            self._track_stats = old_track_stats

    def get_phase_stats(self):
        """Get statistics about phase progression"""
        max_lengths = {
            1: self.phase1_max_length,
            2: self.phase2_max_length,
            3: self.phase3_max_length,
            4: self.phase4_max_length,
            5: self.phase5_current_length
        }

        return {
            'current_phase': self.current_phase,
            'phase5_stage': self.phase5_stage if self.current_phase == 5 else None,
            'total_episodes': self.episode_count,
            'phase_episodes': self.phase_episode_count,
            'success_rate': sum(self.episode_results) / len(self.episode_results) if self.episode_results else 0,
            'total_loops_completed': self.total_loops_completed,
            'phases_completed': self.phases_completed,
            'current_max_length': max_lengths.get(self.current_phase, self.phase5_current_length)
        }
