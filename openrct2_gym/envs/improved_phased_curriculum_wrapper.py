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
                 verbose=1,
                 # Phase 2 sub-stage thresholds (kept at the end for positional compatibility)
                 phase2_roundtrip_threshold=0.30,  # 30% one-chain climb-and-return
                 phase2_chain1_success_threshold=0.30):  # 30% completion with >=1 chain
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
        self.phase2_roundtrip_threshold = phase2_roundtrip_threshold
        self.phase2_chain1_success_threshold = phase2_chain1_success_threshold
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
        self.phase2_summit_results = deque(maxlen=window_size)
        self.phase2_roundtrip_results = deque(maxlen=window_size)
        self.phase2_chain1_completion_results = deque(maxlen=window_size)
        self.phase2_chain2_completion_results = deque(maxlen=window_size)
        self.phase2_chain3_completion_results = deque(maxlen=window_size)
        self.episode_count = 0
        self.phase_episode_count = 0
        self.total_loops_completed = 0

        # Phase 2 sub-stages:
        #   2.1: one-chain climb-and-return, no completion required
        #   2.2: completion with >=1 chain
        #   2.3: completion with >=3 chains
        self.phase2_stage = 1

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
    def _phase_reward_params(phase, phase2_stage=1):
        """Per-phase RewardParams. The PBRS geometry weights (w_xy/w_z/w_dir/w_e) are fixed,
        but the elevation-discovery term (w_h) is ON only in the hill-building phases 2-4 and
        OFF in phase 1 (pure completion) and phase 5 (quality): an always-on climb pull traps
        Phase-1 exploration on building hills instead of closing the loop. The structural bonus
        rewards the structure each intermediate phase gates on (staged chains in P2; chains AND
        drop in P3/P4); phase 5 turns both off and hands over to ride-quality scoring.

        Curriculum logic: master completion FIRST (phase 1, no climb distraction), THEN bridge
        into hills gradually (P2.1/P2.2/P2.3), THEN add drops and integration."""
        if phase >= 5:  # quality only; discovery off
            return RewardParams(R_quality_max=500.0, step_cost=-0.01, w_h=0.0)
        if phase == 2:
            if phase2_stage == 1:  # 2.1 climb-and-return: find the chain hill, no completion gate
                # Closure-first: a RESTORED completion floor (0.2) keeps closing a loop always
                # worth more than an unclosed climb. (The earlier 0.05 de-valuation made a flat
                # close pay ~50 < the ~100 a climb-and-stop banked, so the agent abandoned the
                # ~20% loop-closing it entered Phase 2 with and collapsed to climb-only.) The hill
                # bonus stays additive on top. Strong discovery pull (w_h=6) finds the climb; dense
                # descent shaping (w_return=6) makes the RETURN learnable; round-trip pays reaching
                # station height. No summit payout (R_summit=0) so 'climb and stop' is never a
                # satisfying terminal.
                return RewardParams(
                    R_struct_max=250.0,
                    struct_chain_target=1,
                    struct_w_chain=1.0,
                    struct_w_drop=0.0,
                    completion_hill_floor=0.2,
                    R_roundtrip=300.0,
                    R_summit=0.0,
                    w_h=6.0,
                    w_return=6.0,
                    w_close=8.0,
                )
            if phase2_stage == 2:  # 2.2 one-chain completion: completion-gated, relax w_h
                return RewardParams(
                    R_struct_max=250.0,
                    struct_chain_target=1,
                    struct_w_chain=1.0,
                    struct_w_drop=0.0,
                    completion_hill_floor=0.25,
                    R_roundtrip=300.0,
                    R_summit=0.0,
                    w_h=4.0,
                    w_return=4.0,
                    w_close=8.0,
                )
            return RewardParams(  # 2.3 tighten to >=3 chains; discovery back to the modest default
                R_struct_max=250.0,
                struct_chain_target=3,
                struct_w_chain=1.0,
                struct_w_drop=0.0,
                completion_hill_floor=0.10,
                R_roundtrip=200.0,
                w_h=3.0,
                w_return=3.0,
                w_close=8.0,
            )
        if phase == 3:  # gate: >=2 chains AND drop
            return RewardParams(
                R_struct_max=250.0,
                struct_chain_target=2,
                struct_w_chain=0.5,
                struct_w_drop=0.5,
                completion_hill_floor=0.0,
                R_roundtrip=100.0,
                w_return=3.0,
                w_close=8.0,
            )
        if phase == 4:  # integration: hill + drop
            return RewardParams(
                R_struct_max=250.0,
                struct_chain_target=3,
                struct_w_chain=0.5,
                struct_w_drop=0.5,
                completion_hill_floor=0.0,
                R_roundtrip=100.0,
                w_return=3.0,
                w_close=8.0,
            )
        return RewardParams(w_h=0.0, w_close=8.0)  # phase 1: struct+discovery off; closure densified

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

    @staticmethod
    def _history_chain_max_gain(base_env):
        """Max elevation gained via CHAIN-LIFT pieces (actions 9/10) in the removal-safe history.
        Mirrors the env's `_chain_max_gain` so the wrapper's round-trip diagnostic agrees with
        the env's chain-gated award (single definition of 'climbed with a chain lift')."""
        history = getattr(base_env.track_builder, 'history', [])
        station_z = getattr(base_env, 'STATION_HEIGHT', 14)
        gains = [
            h['next_position'][2] - station_z
            for h in history
            if h.get('action') in (9, 10)
            and h.get('next_position') is not None and len(h.get('next_position')) >= 3
        ]
        return max(gains, default=0.0)

    @staticmethod
    def _returned_near_station_height(base_env):
        """Whether the current build head is back near station height."""
        current_position = getattr(base_env, 'current_position', None)
        if current_position is None or len(current_position) < 3:
            return False
        station_z = getattr(base_env, 'STATION_HEIGHT', 14)
        return abs(current_position[2] - station_z) <= 1

    def _phase2_signals(self, base_env, success):
        """Rolling Phase-2 diagnostics and staged gate predicates."""
        chain_count = self._history_chain_count(base_env)
        params = self._phase_reward_params(2, getattr(self, 'phase2_stage', 1))
        chain_gain = self._history_chain_max_gain(base_env)
        # Summit = chain-climbed to threshold (no return needed); round-trip also requires the
        # return. Trust the env's once-per-episode latches; fall back to recomputing from history.
        summit_awarded = bool(getattr(base_env, '_summit_awarded', False)) or (
            chain_gain >= params.roundtrip_gain
        )
        roundtrip_awarded = bool(getattr(base_env, '_roundtrip_awarded', False)) or (
            chain_gain >= params.roundtrip_gain
            and self._returned_near_station_height(base_env)
        )

        return {
            'phase2_summit': bool(chain_count >= 1 and summit_awarded),
            'phase2_roundtrip': bool(chain_count >= 1 and roundtrip_awarded),
            'phase2_complete_chain1': bool(success and chain_count >= 1),
            'phase2_complete_chain2': bool(success and chain_count >= 2),
            'phase2_complete_chain3': bool(success and chain_count >= 3),
            'completed_chain_count': chain_count if success else 0,
        }

    def _is_qualified(self, base_env, success):
        """Phase-specific 'qualified completion' predicate, matching each phase's reward:
        P2.1 = one-chain climb-and-return; P2.2 = completion with >=1 chain; P2.3 = completion
        with >=3 chains; P3 = completed with >=2 chain lifts AND a drop
        (tightened from OR so the agent must keep the lift hill and add a drop). Returns
        None for phases without a structural gate (1, 4, 5)."""
        if self.current_phase == 2:
            signals = self._phase2_signals(base_env, success)
            stage = getattr(self, 'phase2_stage', 1)
            if stage <= 1:
                return signals['phase2_roundtrip']
            if stage == 2:
                return signals['phase2_complete_chain1']
            return signals['phase2_complete_chain3']
        if self.current_phase == 3:
            return bool(success and self._history_chain_count(base_env) >= 2
                        and self._history_has_drop(base_env))
        return None

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
        base_env.reward_params = self._phase_reward_params(self.current_phase, self.phase2_stage)
        base_env.skip_ride_testing = skip_testing

        if self.verbose >= 1:
            phase_names = {
                1: "Return Practice",
                2: f"Lift Hill Building {self.phase2_stage_name()}",
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
        if self.current_phase == 1:
            if success_rate >= self.phase1_success_threshold:
                self._advance_to_phase(2)
                return True

        elif self.current_phase == 2:
            # Staged bridge from flat completion to chain-lift completion.
            if len(self.episode_qualified_results) >= 50:
                qualified_rate = sum(self.episode_qualified_results) / len(
                    self.episode_qualified_results
                )
                if (
                    self.phase2_stage == 1
                    and qualified_rate >= self.phase2_roundtrip_threshold
                ):
                    self._advance_phase2_stage(2, qualified_rate)
                    return True
                if (
                    self.phase2_stage == 2
                    and qualified_rate >= self.phase2_chain1_success_threshold
                ):
                    self._advance_phase2_stage(3, qualified_rate)
                    return True
                if self.phase2_stage >= 3 and qualified_rate >= self.phase2_success_threshold:
                    self._advance_to_phase(3)
                    return True

        elif self.current_phase == 3:
            # Need loop completion with patterns
            if len(self.episode_qualified_results) >= 50:
                qualified_rate = sum(self.episode_qualified_results) / len(
                    self.episode_qualified_results
                )
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

                self._clear_phase_windows()
                self.phase_episode_count = 0

                if self.verbose >= 1:
                    print(f"\n{'='*60}")
                    print(f"📈 PHASE 5: Advancing to sub-stage {self.phase5_stage}")
                    print(f"   Max track length: {self.phase5_current_length}")
                    print(f"   Success rate achieved: {success_rate:.1%}")
                    print(f"{'='*60}\n")

                return True

        return False

    def phase2_stage_name(self):
        names = {
            1: "(stage 2.1: one-chain roundtrip)",
            2: "(stage 2.2: one-chain completion)",
            3: "(stage 2.3: three-chain completion)",
        }
        return names.get(getattr(self, 'phase2_stage', 1), f"(stage 2.{self.phase2_stage})")

    def _phase2_threshold(self):
        if self.phase2_stage == 1:
            return self.phase2_roundtrip_threshold
        if self.phase2_stage == 2:
            return self.phase2_chain1_success_threshold
        return self.phase2_success_threshold

    def _clear_phase_windows(self):
        self.episode_results.clear()
        self.episode_qualified_results.clear()
        self.phase2_summit_results.clear()
        self.phase2_roundtrip_results.clear()
        self.phase2_chain1_completion_results.clear()
        self.phase2_chain2_completion_results.clear()
        self.phase2_chain3_completion_results.clear()

    def _advance_phase2_stage(self, new_stage, qualified_rate):
        """Advance within Phase 2 without changing the public curriculum phase."""
        success_rate = (
            sum(self.episode_results) / len(self.episode_results)
            if self.episode_results
            else 0
        )
        self.phases_completed.append({
            'phase': f"2.{self.phase2_stage}",
            'success_rate': success_rate,
            'qualified_rate': qualified_rate,
            'episodes': self.phase_episode_count,
            'total_loops': self.total_loops_completed,
        })

        self.phase2_stage = new_stage
        self.phase_episode_count = 0
        self._clear_phase_windows()
        self._update_phase_settings()

        if self.verbose >= 1:
            print(f"\n{'='*70}")
            print(f"🎯 ADVANCING TO PHASE 2.{new_stage}: {self.phase2_stage_name()}")
            print(f"   Previous stage qualified rate: {qualified_rate:.1%}")
            print(f"{'='*70}\n")

    def _advance_to_phase(self, new_phase):
        """Advance to a new phase"""
        success_rate = (
            sum(self.episode_results) / len(self.episode_results)
            if self.episode_results
            else 0
        )
        self.phases_completed.append({
            'phase': f"2.{self.phase2_stage}" if self.current_phase == 2 else self.current_phase,
            'success_rate': success_rate,
            'episodes': self.phase_episode_count,
            'total_loops': self.total_loops_completed
        })

        self.current_phase = new_phase
        if new_phase == 2:
            self.phase2_stage = 1
        self.phase_episode_count = 0
        self._clear_phase_windows()

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
            2: f"Lift Hill Building {self.phase2_stage_name()}",
            3: "Drop & Turn",
            4: "Circuit Mastery",
            5: "Quality Optimization"
        }.get(self.current_phase, f"Phase {self.current_phase}")
        if self.current_phase == 2:
            info['phase2_stage'] = self.phase2_stage
            info['phase2_threshold'] = self._phase2_threshold()

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
            chain_count = self._history_chain_count(base_env)
            phase2_signals = None
            if self.current_phase == 2:
                phase2_signals = self._phase2_signals(base_env, success)
                self.phase2_summit_results.append(phase2_signals['phase2_summit'])
                self.phase2_roundtrip_results.append(phase2_signals['phase2_roundtrip'])
                self.phase2_chain1_completion_results.append(
                    phase2_signals['phase2_complete_chain1']
                )
                self.phase2_chain2_completion_results.append(
                    phase2_signals['phase2_complete_chain2']
                )
                self.phase2_chain3_completion_results.append(
                    phase2_signals['phase2_complete_chain3']
                )

            # Phase-specific qualified success, sourced from the removal-safe track history.
            qualified = self._is_qualified(base_env, success)
            if qualified is not None:
                self.episode_qualified_results.append(qualified)

            if success:
                self.total_loops_completed += 1

            # Ride-quality scoring now lives in the env's terminal reward (single
            # authority); the wrapper no longer adds a quality bonus here.

            # Add phase info + diagnostics (so the curriculum progress is visible in TB).
            info['learning_phase'] = self.current_phase
            info['curriculum_phase'] = self.current_phase
            info['chain_count'] = chain_count
            if phase2_signals is not None:
                info['phase2_stage'] = self.phase2_stage
                info['phase2_threshold'] = self._phase2_threshold()
                info.update(phase2_signals)
                info['phase2_summit_rate'] = (
                    sum(self.phase2_summit_results) / len(self.phase2_summit_results)
                    if self.phase2_summit_results else 0.0
                )
                info['phase2_roundtrip_rate'] = (
                    sum(self.phase2_roundtrip_results) / len(self.phase2_roundtrip_results)
                    if self.phase2_roundtrip_results else 0.0
                )
                info['phase2_chain1_completion_rate'] = (
                    sum(self.phase2_chain1_completion_results)
                    / len(self.phase2_chain1_completion_results)
                    if self.phase2_chain1_completion_results else 0.0
                )
                info['phase2_chain2_completion_rate'] = (
                    sum(self.phase2_chain2_completion_results)
                    / len(self.phase2_chain2_completion_results)
                    if self.phase2_chain2_completion_results else 0.0
                )
                info['phase2_chain3_completion_rate'] = (
                    sum(self.phase2_chain3_completion_results)
                    / len(self.phase2_chain3_completion_results)
                    if self.phase2_chain3_completion_results else 0.0
                )
            if qualified is not None:
                info['qualified'] = bool(qualified)
                info['qualified_rate'] = (
                    sum(self.episode_qualified_results) / len(self.episode_qualified_results)
                    if self.episode_qualified_results else 0.0
                )
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
                    2: f"Lift Hill Building {self.phase2_stage_name()}",
                    3: "Drop & Turn",
                    4: "Circuit Mastery",
                    5: f"Quality Opt (stage {self.phase5_stage})"
                }
                phase_name = phase_names.get(self.current_phase, f"Phase {self.current_phase}")

                qualified_str = ""
                if self.current_phase in [2, 3] and self.episode_qualified_results:
                    qualified_rate = sum(self.episode_qualified_results) / len(
                        self.episode_qualified_results
                    )
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
            'phase2_stage': self.phase2_stage if self.current_phase == 2 else None,
            'phase5_stage': self.phase5_stage if self.current_phase == 5 else None,
            'total_episodes': self.episode_count,
            'phase_episodes': self.phase_episode_count,
            'success_rate': (
                sum(self.episode_results) / len(self.episode_results)
                if self.episode_results
                else 0
            ),
            'total_loops_completed': self.total_loops_completed,
            'phases_completed': self.phases_completed,
            'current_max_length': max_lengths.get(self.current_phase, self.phase5_current_length)
        }
