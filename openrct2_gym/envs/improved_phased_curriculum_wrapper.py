"""
Improved Phased Curriculum Learning Wrapper for OpenRCT2 Environment
Implements 5-phase progressive learning with physics-aware rewards.
"""
import gymnasium as gym
import numpy as np
from collections import deque
from contextlib import contextmanager
from typing import Dict, Any, Tuple

from openrct2_gym.envs.openrct2_env import RewardParams


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

        # Update environment settings for current phase. The reward is now a single
        # parametrized function owned by the env; the curriculum only sets parameters.
        self._update_phase_settings()

    def _get_base_env(self):
        """Get the base OpenRCT2 environment"""
        env = self.env
        while hasattr(env, 'env'):
            env = env.env
        return env

    @staticmethod
    def _phase_reward_params(phase):
        """Per-phase RewardParams. The PBRS potential (Phi weights/normalizers) is FIXED
        across phases for global policy-invariance; only the sparse objectives vary:
        ride-quality scoring and a small living cost turn on in phase 5."""
        if phase >= 5:
            return RewardParams(R_quality_max=500.0, step_cost=-0.01)
        return RewardParams()

    @staticmethod
    def _history_chain_count(base_env):
        """Number of chain-lift pieces in the (removal-safe) track history."""
        history = getattr(base_env.track_builder, 'history', [])
        return sum(1 for h in history if h.get('action') in (9, 10))

    @staticmethod
    def _history_has_drop(base_env):
        """Whether the track history contains a drop/descent piece."""
        history = getattr(base_env.track_builder, 'history', [])
        return any(h.get('action') in (6, 8, 12, 14) for h in history)

    def _update_phase_settings(self):
        """Update environment settings based on current phase.

        Sets ONLY parameters (reward params, max length, ride-testing) — the reward
        function itself is the env's single _calculate_reward and is never swapped.
        """
        base_env = self._get_base_env()

        phase_configs = {
            1: (self.phase1_max_length, True),
            2: (self.phase2_max_length, True),
            3: (self.phase3_max_length, True),
            4: (self.phase4_max_length, True),
            5: (self.phase5_current_length, False),
        }

        max_length, skip_testing = phase_configs.get(
            self.current_phase,
            (self.phase5_current_length, False)
        )

        base_env.max_track_length = max_length
        base_env.reward_params = self._phase_reward_params(self.current_phase)
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

            # Phase-specific qualified success tracking, sourced from the removal-safe
            # track history (not from reward-time bookkeeping, which no longer exists).
            if self.current_phase == 2:
                qualified = success and self._history_chain_count(base_env) >= 3
                self.episode_qualified_results.append(qualified)

            elif self.current_phase == 3:
                has_patterns = self._history_chain_count(base_env) >= 2 or self._history_has_drop(base_env)
                qualified = success and has_patterns
                self.episode_qualified_results.append(qualified)

            if success:
                self.total_loops_completed += 1

            # Ride-quality scoring now lives in the env's terminal reward (single
            # authority); the wrapper no longer adds a quality bonus here.

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
