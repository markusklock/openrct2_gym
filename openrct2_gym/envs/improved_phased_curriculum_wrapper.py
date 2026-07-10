"""
Improved Phased Curriculum Learning Wrapper for OpenRCT2 Environment
Implements 5-phase progressive learning with physics-aware rewards.
"""
import gymnasium as gym
import numpy as np
from collections import deque
from contextlib import contextmanager
from typing import Dict, Any, Tuple

from openrct2_gym.envs.openrct2_env import OpenRCT2Env, RewardParams
from openrct2_gym.envs.warm_start import LoopLibrary, WarmStartAnnealer, WarmStartPlan


class ImprovedPhasedCurriculumWrapper(gym.Wrapper):
    """
    Wrapper that implements 5-phase curriculum learning with physics-aware rewards.

    Phase 1: "Return Practice" (40 pieces) - Focus on navigation
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
                 phase1_max_length=40,
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
                 phase2_chain1_success_threshold=0.30,  # 30% completion with >=1 chain
                 # Warm-start reverse curriculum (see warm_start.py). None for the library
                 # path defers to OpenRCT2Env._LOOP_LIBRARY_PATH at construction, so test
                 # fixtures that isolate the env's harvest file isolate the wrapper too.
                 warm_start_enabled=True,
                 loop_library_path=None,
                 p_cold=0.25,
                 warm_k_init=3):
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

        # Warm-start reverse curriculum: the wrapper owns the per-worker annealer and a
        # read-view of the shared loop library (the env writes harvests to the same file).
        # Gate windows below are COLD-ONLY; scaffolded outcomes feed the annealer instead.
        self.warm_start_enabled = warm_start_enabled
        self._loop_library = LoopLibrary(loop_library_path or OpenRCT2Env._LOOP_LIBRARY_PATH)
        self._annealer = WarmStartAnnealer(k_init=warm_k_init, p_cold=p_cold)
        self._current_plan = WarmStartPlan(prefix=[], k=0, loop_len=0, cold=True)

        # Performance tracking. episode_results (and the qualified/phase-2 windows below)
        # see only COLD episodes -- a scaffolded win must never advance a phase gate.
        self.episode_results = deque(maxlen=window_size)
        self.scaffold_results = deque(maxlen=window_size)
        self._cold_flags = deque(maxlen=window_size)
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
    def _validate_completion_first(params, label):
        """Completion-first invariant: the once-per-episode climb milestones (R_roundtrip + R_summit)
        are earnable WITHOUT closing the loop, so they must never out-pay completing it -- otherwise
        the agent farms the milestone and abandons closure. (Observed at 1.1M steps: Phase-2.3
        completion collapsed 0.44 -> 0.08 while struct_bonus stayed 0, because R_roundtrip=200 beat
        the 0.10*1000=100 a flat close paid.) Two checks:
          * ALWAYS: milestones < R_complete -- a perfect (full-hill) completion must beat farming.
          * when completion_hill_floor > 0: milestones < floor*R_complete -- even a FLAT close must
            beat farming. (floor==0 phases pay nothing for a flat close by design and rely on the
            agent already producing hill-ful completions; only the first check applies there.)"""
        milestones = params.R_roundtrip + params.R_summit
        assert milestones < params.R_complete, (
            f"{label}: climb milestones {milestones} >= R_complete {params.R_complete} "
            f"-- not closing the loop can out-pay even a perfect completion")
        if params.completion_hill_floor > 0.0:
            # Worst-case completion pay compounds ALL gates (hill x length x quality floors).
            flat = (params.completion_hill_floor * params.completion_length_floor
                    * params.completion_quality_floor * params.R_complete)
            assert milestones < flat, (
                f"{label}: climb milestones {milestones} >= flat-completion floor {flat} "
                f"({params.completion_hill_floor}*{params.completion_length_floor}"
                f"*{params.R_complete}) -- not closing out-pays a flat close")

    @staticmethod
    def _phase_reward_params(phase, phase2_stage=1):
        """Per-phase RewardParams, validated for the completion-first invariant (see
        _validate_completion_first). The raw per-phase config lives in _phase_reward_params_raw."""
        params = ImprovedPhasedCurriculumWrapper._phase_reward_params_raw(phase, phase2_stage)
        ImprovedPhasedCurriculumWrapper._validate_completion_first(
            params, f"phase {phase}" + (f".{phase2_stage}" if phase == 2 else ""))
        return params

    @staticmethod
    def _phase_reward_params_raw(phase, phase2_stage=1):
        """Per-phase RewardParams. The PBRS geometry weights (w_xy/w_z/w_dir/w_e) are fixed,
        but the elevation-discovery term (w_h) is ON only in the hill-building phases 2-4 and
        OFF in phase 1 (pure completion) and phase 5 (quality): an always-on climb pull traps
        Phase-1 exploration on building hills instead of closing the loop. The structural bonus
        rewards the structure each intermediate phase gates on (staged chains in P2; chains AND
        drop in P3/P4); phase 5 turns both off and hands over to ride-quality scoring.

        Curriculum logic: master completion FIRST (phase 1, no climb distraction), THEN bridge
        into hills gradually (P2.1/P2.2/P2.3), THEN add drops and integration."""
        if phase >= 5:  # quality; discovery off (w_h=0), step_cost 0 (quality scales with
            # size). Jul-9 redesign: the old params zeroed every gate/struct term and the
            # policy promptly shrank to a 24-piece E=1.15 loop. The reward now points at
            # the game's actual rating math: the completion gate ramps with MEASURED
            # excitement, struct credit ramps the wooden-RC rating caps (single drop
            # >=12z, >=2 drops, length; banked turns for the turns sub-rating), and
            # discrete milestone bars pay every excitement increment on the way to E7-9.
            return RewardParams(
                R_quality_max=500.0,
                step_cost=0.0,
                w_h=0.0,
                completion_quality_floor=0.4,
                exc_gate_target=6.0,
                R_struct_max=250.0,
                struct_w_chain=0.0,
                struct_w_single_drop=0.30,
                struct_single_drop_target=12.0,   # the game's RequirementDropHeight cap
                struct_w_drop_runs=0.20,
                struct_drop_runs_target=2.0,      # the game's RequirementNumDrops cap
                struct_w_drop=0.15,
                struct_drop_target=16.0,
                struct_w_length=0.20,
                struct_length_target=70.0,        # ~370m cap at the MEASURED 5.5 m/piece
                                                  # (probe_measurements, Jul-10)
                struct_w_banked=0.15,
                struct_banked_target=4.0,
                R_viable=150.0,                   # keep P4's verified-run bonus
                R_caps_max=250.0,                 # measured rating-cap ramps (getRideMeasurements)
                R_exc_milestone=100.0,
                exc_milestone_bars=(2.5, 4.0, 5.5),
                w_exc_feat=6.0,                   # dense per-piece excitement gradient
            )
        if phase == 2:
            if phase2_stage == 1:  # 2.1 climb-and-return: find the chain hill, no completion gate
                # Closure-first: a RESTORED completion floor (0.2) keeps closing a loop always
                # worth more than an unclosed climb. (The earlier 0.05 de-valuation made a flat
                # close pay ~50 < the ~100 a climb-and-stop banked, so the agent abandoned the
                # ~20% loop-closing it entered Phase 2 with and collapsed to climb-only.) The hill
                # bonus stays additive on top. Strong discovery pull (w_h=6) finds the climb; dense
                # descent shaping (w_return=6) makes the RETURN learnable; round-trip pays reaching
                # station height. The annealed roundtrip_gain (1 z here and through 2.2; only 2.3
                # demands the full 4-z hill) makes that round-trip reachable from a flat-loop policy,
                # and a SMALL summit
                # breadcrumb (R_summit) pays the climb itself so it is worth starting before the
                # return is learned. CRITICAL: R_roundtrip + R_summit (80+40=120) is kept BELOW the
                # flat-completion floor (0.2*1000=200) so 'climb and stop' never out-pays closing the
                # loop (_validate_completion_first enforces this). The dense PBRS shaping (w_h=6,
                # w_return=6), not the sparse milestone, is what teaches the climb.
                return RewardParams(
                    R_struct_max=250.0,
                    struct_chain_target=1,
                    struct_w_chain=1.0,
                    struct_w_drop=0.0,
                    completion_hill_floor=0.2,
                    R_roundtrip=80.0,
                    roundtrip_gain=1.0,
                    R_summit=40.0,
                    w_h=6.0,
                    w_return=6.0,
                    w_close=8.0,
                    w_route=3.0,
                )
            if phase2_stage == 2:  # 2.2 one-chain completion: integrate the chain INTO a closed
                # loop. Keep the climb cheap (roundtrip_gain=1.0, not 2.0) so the 2.1 climb habit
                # and its breadcrumbs survive the jump, and lower the flat floor (0.25 -> 0.15) so
                # flat completion can't out-pay chain completion -- at 0.25 the agent collapsed onto
                # flat closure here. Milestones (60+30=90) stay below the flat floor (0.15*1000=150)
                # so closing always wins. (The early-Phase-2 entropy floor 0.018 is carried through
                # 2.2 in train.py.)
                return RewardParams(
                    R_struct_max=250.0,
                    struct_chain_target=1,
                    struct_w_chain=1.0,
                    struct_w_drop=0.0,
                    completion_hill_floor=0.15,
                    R_roundtrip=60.0,
                    roundtrip_gain=1.0,
                    R_summit=30.0,
                    w_h=4.0,
                    w_return=4.0,
                    w_close=8.0,
                    w_route=3.0,
                )
            # 2.3 tighten to >=3 chains; discovery back to the modest default. R_roundtrip (60) stays
            # below the flat floor (0.10*1000=100) -- this is the stage that broke: at R_roundtrip=200
            # > 100 the agent farmed the climb-and-return milestone and stopped closing the loop.
            return RewardParams(
                R_struct_max=250.0,
                struct_chain_target=3,
                struct_w_chain=1.0,
                struct_w_drop=0.0,
                completion_hill_floor=0.10,
                R_roundtrip=60.0,
                roundtrip_gain=3.0,   # == chain gain of the canonical hill (crest isn't chained)
                w_h=3.0,
                w_return=3.0,
                w_close=8.0,
                w_route=3.0,
            )
        if phase == 3:  # "Real Drops & Scale": graded height/drop/length toward targets the
            # 2.3 mini-loop does NOT meet (h>=4, drop>=4, len>=25) -- the old piece-count gate
            # was already satisfied on entry and taught nothing (cleared in ~30k steps twice).
            return RewardParams(
                R_struct_max=250.0,
                struct_chain_target=2,
                struct_w_chain=0.0,        # height carries the chain credit now
                struct_w_height=0.4,
                struct_height_target=4.0,
                struct_w_drop=0.4,
                struct_drop_target=4.0,
                struct_w_length=0.2,
                struct_length_target=25.0,
                completion_hill_floor=0.0,
                # Length-trap fix (Jul-5 overnight run: converged 18-piece mini-loop,
                # qualified_rate 0): gate the completion payout on length so each piece
                # toward 25 is worth ~+30 (vs the ~-10 discount cost), and pay the phase
                # gate itself as a discrete event.
                completion_length_floor=0.25,
                R_qualify=200.0,
                qualify_requires_energy=True,
                R_roundtrip=100.0,
                w_return=3.0,
                w_close=8.0,
                w_route=3.0,
                h_scale=8.0,               # taller climbs keep paying the discovery term
            )
        if phase == 4:  # "Big & Verified": steeper/taller/longer, and the ride test is ON --
            # R_viable pays only when the train demonstrably made it around (nonzero stats).
            return RewardParams(
                R_struct_max=250.0,
                struct_chain_target=3,
                struct_w_chain=0.0,
                # Steepness graded in (Jul-7): the qualified gate's 60-degree leg had no
                # gradient -- 9h of P4 never placed a steep piece. Reweighted so a no-steep
                # build caps at 0.8 (leaving R_complete + struct money on the table) and one
                # full 8z steep segment closes the gap; weights still sum to 1.0.
                struct_w_height=0.3,
                struct_height_target=6.0,
                struct_w_drop=0.3,
                struct_drop_target=8.0,
                struct_w_steep=0.2,
                struct_steep_target=8.0,
                struct_w_length=0.2,
                struct_length_target=40.0,
                completion_hill_floor=0.0,
                # Same length economics as P3, at the P4 bar (40); the qualification
                # bonus additionally requires the steep drop and the verified ride test,
                # mirroring _is_qualified's P4 legs.
                completion_length_floor=0.25,
                R_qualify=200.0,
                qualify_requires_steep_drop=True,
                qualify_requires_test=True,
                R_roundtrip=100.0,
                R_viable=150.0,
                w_return=3.0,
                w_close=8.0,
                w_route=3.0,
                h_scale=8.0,
            )
        # phase 1: struct+discovery off; closure densified; route term guides the detour
        return RewardParams(w_h=0.0, w_close=8.0, w_route=3.0)

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
    def _history_drop_z(base_env):
        """Total z dropped over descent pieces (mirrors env._total_drop_z)."""
        history = getattr(base_env.track_builder, 'history', [])
        return float(sum(max(0.0, h['position'][2] - h['next_position'][2])
                         for h in history
                         if h.get('position') is not None and h.get('next_position') is not None))

    @staticmethod
    def _history_has_steep_drop(base_env):
        """Whether the history contains a 60-degree descent piece (actions 8/27/28)."""
        history = getattr(base_env.track_builder, 'history', [])
        return any(h.get('action') in (8, 27, 28) for h in history)

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
        # Mirror of the env's award: near station height AND strictly below the climb bar
        # (at gain=1 the +1 tolerance otherwise contains the summit -- no return required).
        station_z = getattr(base_env, 'STATION_HEIGHT', 14)
        position = getattr(base_env, 'current_position', None)
        below_bar = (position is not None and len(position) >= 3
                     and position[2] < station_z + params.roundtrip_gain)
        roundtrip_awarded = bool(getattr(base_env, '_roundtrip_awarded', False)) or (
            chain_gain >= params.roundtrip_gain
            and self._returned_near_station_height(base_env)
            and below_bar
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
            # "Real Drops & Scale": the thresholds are the phase's own structure targets
            # (single source of truth), plus the cheap energy-viability proxy.
            P = self._phase_reward_params(3)
            return bool(success
                        and self._history_chain_max_gain(base_env) >= P.struct_height_target
                        and self._history_drop_z(base_env) >= P.struct_drop_target
                        and getattr(base_env, 'track_length', 0) >= P.struct_length_target
                        and base_env._calculate_energy_margin() >= 0.0)
        if self.current_phase == 4:
            # "Big & Verified": steeper/taller/longer AND the ride test actually returned
            # stats -- the train demonstrably made it around.
            P = self._phase_reward_params(4)
            return bool(success
                        and self._history_chain_max_gain(base_env) >= P.struct_height_target
                        and self._history_drop_z(base_env) >= P.struct_drop_target
                        and self._history_has_steep_drop(base_env)
                        and getattr(base_env, 'track_length', 0) >= P.struct_length_target
                        and getattr(base_env, '_last_test_ok', False))
        if self.current_phase >= 5:
            # Quality diagnostic (does NOT gate the P5 length ladder, which stays on raw
            # cold success): completed, TESTED, and rated at least the middle milestone
            # bar. Gives curriculum/qualified_rate meaning in P5.
            return bool(success
                        and getattr(base_env, '_last_test_ok', False)
                        and float(getattr(base_env, 'last_ride_excitement', 0.0)) >= 4.0)
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
            4: (self.phase4_max_length, False),   # ride test ON: P4 verifies the train runs
            5: (self.phase5_current_length, False),
        }

        max_length, skip_testing = phase_configs.get(
            self.current_phase,
            (self.phase5_current_length, False)
        )

        base_env.max_track_length = max_length
        base_env.reward_params = self._phase_reward_params(self.current_phase, self.phase2_stage)
        base_env.skip_ride_testing = skip_testing
        # Harvest budget follows the phase's track budget (a P4/P5 loop longer than the
        # old fixed 40 cap is exactly the material the later pools need).
        base_env.harvest_max_len = max_length

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
            # Qualified window, not raw completions: raw success was already ~100% on entry
            # (the old gate advanced without any new capability being learned).
            if len(self.episode_qualified_results) >= 50:
                qualified_rate = sum(self.episode_qualified_results) / len(
                    self.episode_qualified_results
                )
                if qualified_rate >= self.phase4_success_threshold:
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
                # Each length rung changes the scaffold pool (bigger budget admits bigger
                # exemplars) -- restart the anneal like any phase change (the
                # _advance_phase2_stage precedent). Moot before Jul-9 when P5 was cold.
                self._annealer.on_phase_change(self.current_phase)

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
        self.scaffold_results.clear()
        self._cold_flags.clear()
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
        # A sub-stage is a new gate: re-anneal the scaffold for it.
        self._annealer.on_phase_change(self.current_phase)
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
        # New phase == new target skill (P2 flips the pool to hill loops): restart the anneal.
        self._annealer.on_phase_change(new_phase)

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

    def _sample_warm_start(self):
        """This episode's warm-start plan. Cold when disabled, during evaluation (eval must
        measure the true task), or past the scaffolded phases (1-5). Each phase prefers
        loops that can actually satisfy its gate (with graceful pool fallback)."""
        if (not self.warm_start_enabled or not self._track_stats
                or self.current_phase > 5):
            return WarmStartPlan(prefix=[], k=0, loop_len=0, cold=True)
        self._loop_library.maybe_refresh()   # pick up other workers' harvested loops
        base_env = self._get_base_env()
        budget = getattr(base_env, 'max_track_length', 40)
        min_chains, min_len, min_drop_z, min_steep_z = 1, 0, 0, 0
        min_single_drop_z, min_excitement = 0, 0.0
        if self.current_phase == 2 and self.phase2_stage >= 3:
            min_chains = 3
        elif self.current_phase == 3:
            min_chains, min_len, min_drop_z = 2, 20, 4
        elif self.current_phase == 4:
            # Match the P4 gate: qualifying length AND a steep segment. min_len was 25
            # (the P3 bar) which let non-steep 26-38 piece harvests dominate the pool
            # while the 60-degree leg went unpracticed (Jul-8 run: 12h, zero own steep).
            min_chains, min_len, min_drop_z, min_steep_z = 3, 40, 8, 8
        elif self.current_phase >= 5:
            # P5 (Jul-9): scaffold from excitement exemplars. Shape criteria mirror the
            # rating caps (>=12z single drop on a >=40 piece loop); the excitement bar
            # SELF-RATCHETS at 0.8 x the best tagged loop fitting the budget -- 0 on a
            # legacy-only pool (everything qualifies), then rising behind every better
            # exemplar the harvest tags. No persistent state; recomputed per episode.
            min_chains, min_len, min_drop_z, min_single_drop_z = 1, 40, 12, 12
            min_excitement = 0.8 * self._loop_library.best_excitement(budget)
        return self._annealer.sample_plan(
            self._loop_library, self.current_phase, budget,
            min_chains=min_chains, min_len=min_len, min_drop_z=min_drop_z,
            min_steep_z=min_steep_z, min_single_drop_z=min_single_drop_z,
            min_excitement=min_excitement)

    def reset(self, **kwargs):
        """Reset environment and check for phase advancement"""
        self._check_phase_advancement()

        # Stage this episode's warm-start prefix on the base env AFTER the advancement
        # check (so phase/max_length are current); the env replays it one-shot inside
        # reset(), before Phi seeding. The suffix k sizes the tight scaffolded budget.
        self._current_plan = self._sample_warm_start()
        base_env = self._get_base_env()
        base_env.warm_start_actions = list(self._current_plan.prefix) or None
        base_env.warm_start_suffix_k = self._current_plan.k or None

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
            # Self-imitation ratchet diagnostics: the pool's excitement bar and the best
            # tagged exemplar it trails (watch the bar climb behind better harvests).
            best_exc = self._loop_library.best_excitement(self.phase5_current_length)
            info['library_best_excitement'] = best_exc
            info['p5_pool_exc_bar'] = 0.8 * best_exc
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
            cold = bool(info.get('cold_start', getattr(base_env, '_warm_cold', True)))
            self._cold_flags.append(cold)
            # Cold-only gating: every phase-gate window (success, qualified, phase-2 chain
            # diagnostics) sees only true-task episodes -- a scaffolded win must never
            # advance a gate. Scaffolded outcomes drive the annealer's frontier instead.
            if cold:
                self.episode_results.append(success)
            else:
                self.scaffold_results.append(success)
            # Aborted prefixes are infrastructure events, not agent outcomes: they must
            # not feed the frontier (a burst of aborts would demote k_max on noise).
            if not info.get('warm_aborted', False):
                self._annealer.record_outcome(self._current_plan, success)
            chain_count = self._history_chain_count(base_env)
            phase2_signals = None
            if self.current_phase == 2:
                phase2_signals = self._phase2_signals(base_env, success)
                if cold:
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
            if qualified is not None and cold:
                self.episode_qualified_results.append(qualified)

            if success:
                self.total_loops_completed += 1

            # Ride-quality scoring now lives in the env's terminal reward (single
            # authority); the wrapper no longer adds a quality bonus here.

            # Add phase info + diagnostics (so the curriculum progress is visible in TB).
            info['learning_phase'] = self.current_phase
            info['curriculum_phase'] = self.current_phase
            info['chain_count'] = chain_count
            # On the done-info too (reset-infos never reach the callback under SubprocVecEnv,
            # so curriculum/max_length was silently never logged).
            info['max_track_length'] = getattr(base_env, 'max_track_length', 0)
            if phase2_signals is not None:
                info['phase2_stage'] = self.phase2_stage
                info['phase2_threshold'] = self._phase2_threshold()
                # Surface the live annealed schedule so the discoverability bootstrap is visible
                # in TB next to summit/roundtrip rates (diagnostic-per-term).
                stage_params = self._phase_reward_params(2, self.phase2_stage)
                info['phase2_roundtrip_gain'] = stage_params.roundtrip_gain
                info['phase2_summit_reward'] = stage_params.R_summit
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
            # phase_success_rate is now the COLD-episode rate (the gate-driving number);
            # cold_success_rate is the explicit alias, scaffold_success_rate its counterpart.
            cold_rate = (
                sum(self.episode_results) / len(self.episode_results)
                if self.episode_results else 0
            )
            info['phase_success_rate'] = cold_rate
            info['cold_success_rate'] = cold_rate
            info['scaffold_success_rate'] = (
                sum(self.scaffold_results) / len(self.scaffold_results)
                if self.scaffold_results else 0.0
            )
            info['cold_fraction'] = (
                sum(self._cold_flags) / len(self._cold_flags)
                if self._cold_flags else 1.0
            )
            info['warm_k'] = self._current_plan.k
            info['warm_k_max'] = self._annealer.k_max
            info['loop_library_size'] = len(self._loop_library)
            frontier_rate = self._annealer.frontier_rate
            if frontier_rate is not None:
                info['warm_frontier_rate'] = frontier_rate

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
            # cold-episode rate: episode_results is cold-only under warm starts
            'success_rate': (
                sum(self.episode_results) / len(self.episode_results)
                if self.episode_results
                else 0
            ),
            'total_loops_completed': self.total_loops_completed,
            'phases_completed': self.phases_completed,
            'current_max_length': max_lengths.get(self.current_phase, self.phase5_current_length),
            'warm_k_max': self._annealer.k_max,
            'loop_library_size': len(self._loop_library),
        }
