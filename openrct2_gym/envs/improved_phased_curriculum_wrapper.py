"""
Improved Phased Curriculum Learning Wrapper for OpenRCT2 Environment
Implements 5-phase progressive learning with physics-aware rewards.
"""
import random
import gymnasium as gym
import numpy as np
from collections import deque
from contextlib import contextmanager
from typing import Dict, Any, Tuple

from openrct2_gym.envs.openrct2_env import OpenRCT2Env, RewardParams
from openrct2_gym.envs.warm_start import LoopLibrary, WarmStartAnnealer, WarmStartPlan
from openrct2_gym.envs.track_pieces import SBEND_ACTIONS, TURN_ACTIONS
from openrct2_gym.envs.footprint import classify_family, FAMILY_N


class ImprovedPhasedCurriculumWrapper(gym.Wrapper):
    """
    Wrapper that implements 6-phase curriculum learning with physics-aware rewards.

    Phase 1: "Return Practice" (40 pieces) - Focus on navigation
    Phase 2: "Lift Hill Building" (40 pieces) - Learn chain lifts and energy
    Phase 3: "Drop & Turn" (60 pieces) - Learn drops and turnarounds
    Phase 4: "Circuit Mastery" (80 pieces) - Full integration
    Phase 5: "Quality Optimization" (80-120 pieces) - Optimize ride ratings
    Phase 6: "Style / Variety" (120 pieces) - Winding layouts at held quality
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
                 phase5_target_length=120,     # Aug-12: a same-policy, 10-episode probe
                                                # (oval/out-and-back/winding at 120
                                                # vs. 80 pieces) read a feasibility inversion
                                                # -- shorter budgets closing the requested
                                                # shape better than the default oval -- and
                                                # briefly lowered this to 90 on that basis.
                                                # It did not replicate: measured on a policy
                                                # actually TRAINED at 90 (~600k steps, 10
                                                # unaided episodes/seed), everything closed
                                                # worse (oval 40%, out-and-back 20%, winding
                                                # 30%) and the oval was still best -- no
                                                # inversion. The original reading came from a
                                                # policy trained at 120 under a different
                                                # phase configuration, on a single n=10
                                                # sample, which is why it misled. Meanwhile
                                                # unaided completion fell 0.72->0.48 and the
                                                # cold shape distribution didn't move at all
                                                # (99.6% oval, n=768). Conclusion: track
                                                # budget is not the lever for shape variety;
                                                # reverted to 120. Do not re-run this
                                                # experiment without a different lever.
                 phase5_increase_step=10,
                 phase6_entry_threshold=0.30,   # cold tested-E>=4 rate that opens P6
                 phase6_max_length=120,         # Aug-12: matches phase5_target_length's
                                                # topped-out ceiling (non-decreasing budget
                                                # ordering across phases); see that param's
                                                # comment -- the 90-piece experiment it once
                                                # matched did not replicate and was reverted.
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
                 warm_k_init=3,
                 warm_min_prefix=None,
                 # Jul-19: a deep-P5/P6 policy cannot re-walk Phase 1 (its committed
                 # 90+ piece builds truncate inside the 40-piece budget -- live stall:
                 # cold completion 0.00 and active unlearning). Start the curriculum
                 # where the policy actually is.
                 initial_phase=1):
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
        self.phase6_entry_threshold = phase6_entry_threshold
        self.phase6_max_length = phase6_max_length

        # Warm-start reverse curriculum: the wrapper owns the per-worker annealer and a
        # read-view of the shared loop library (the env writes harvests to the same file).
        # Gate windows below are COLD-ONLY; scaffolded outcomes feed the annealer instead.
        self.warm_start_enabled = warm_start_enabled
        self._loop_library = LoopLibrary(loop_library_path or OpenRCT2Env._LOOP_LIBRARY_PATH)
        self._annealer = WarmStartAnnealer(k_init=warm_k_init, p_cold=p_cold)
        # Continuation override for the prefix descent (in-memory like k): None ->
        # the phase default (P6 arms 6/6); an explicit value seeds min_prefix at the
        # achieved rung while keeping the init=6 demote ceiling.
        self._warm_min_prefix_override = warm_min_prefix
        self._current_plan = WarmStartPlan(prefix=[], k=0, loop_len=0, cold=True)

        # Performance tracking. episode_results (and the qualified/phase-2 windows below)
        # see only COLD episodes -- a scaffolded win must never advance a phase gate.
        self.episode_results = deque(maxlen=window_size)
        self.scaffold_results = deque(maxlen=window_size)
        self._cold_flags = deque(maxlen=window_size)
        self.episode_qualified_results = deque(maxlen=window_size)
        # Per-phase family seed sampling (Aug-9) and its own cold-only measurement
        # window, one per family index (see PHASE_FAMILIES / _sample_target_family).
        self._family_rng = random.Random()
        self.episode_family_results = {z: deque(maxlen=window_size)
                                       for z in range(FAMILY_N)}
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

        # Deep start (resume a mature policy without the phase-1 re-walk): mark the
        # earlier ladders complete and put the annealer in its late-phase mode.
        if initial_phase > 1:
            self.current_phase = initial_phase
            if initial_phase >= 5:
                self.phase5_current_length = self.phase5_target_length
            self._annealer.on_phase_change(initial_phase)

        # Update environment settings for current phase. The reward is now a single
        # parametrized function owned by the env; the curriculum only sets parameters.
        self._update_phase_settings()

    # Heading turns a floor-bound (smallest-seed) build must reach before the prefix
    # descent may shrink the seed further. 6, not the pool's 8 (Aug-8): at 8 the descent
    # stalled ~570k steps because floor-bound builds -- where the agent builds everything
    # behind the seed -- sit near the cold end of the distribution. The bar only has to
    # be clearable, because the demote path makes the descent an EQUILIBRIUM SEARCH: the
    # seed widens back wherever style fails, so it settles at the width the policy can
    # actually hold, and that settling point is itself the measurement.
    #
    # DEAD ON THE CURRENT CURRICULUM (Aug-9 final review), and a trap if it is revived:
    # only P6 arms min_prefix, and P6 takes the family branch in step() instead of this
    # bar. Were any phase BELOW P6 ever to arm min_prefix, this bar would apply -- and 6
    # heading turns CONTRADICTS the oval band (0-5 turns by definition, footprint.py
    # FAMILIES), so a correctly-built oval could never be "styled" and that phase's prefix
    # descent would stall at min_prefix_init forever, exactly the failure the family
    # branch was added to fix. Arm min_prefix below P6 only together with a
    # seed-conditioned predicate.
    FLOOR_STYLE_MIN_TURNS = 6

    # Families the seed may request, per phase. Widens with the track budget: a
    # 40-piece build cannot express a serpentine. Phases 1-2 pin the seed to 0 so
    # the observation input is constant rather than noise while its reward is off.
    #
    # MEASURED, not guessed (pre-launch final re-review): the length distribution of
    # every family in the 196,058-record deployment library, checked against each
    # phase's pool budget (pool() admits length <= max_len - 2, so P3 <= 58, P4 <= 78):
    #
    #   family          records   min  median  max   fit P3 (<=58)  fit P4 (<=78)
    #   0 oval          148,511     6      70  118         25,595        129,597
    #   1 spiral            894     40     102  115              1              1
    #   2 out_and_back   29,235      8      74  117            194         18,403
    #   3 winding        13,950     15      88  116              8            592
    #   4 serpentine         63     76      98  102              0             10
    #
    # Spiral is a ~100-piece shape: 0 direction switches on a closed circuit forces a
    # turn count that is a multiple of 4, so its 6-9 turn band means exactly 8 same-
    # handed 90-degree turns -- two full revolutions that still have to return to the
    # dock. A phase's family set includes a shape only when the library shows it is
    # closable inside that phase's budget; a floor/weight retune cannot substitute for
    # a shape the budget cannot express. Spiral therefore leaves P3 and P4 (1 of 894
    # fits either budget) and first appears at P5, where its median 102 sits inside the
    # 80-120 range. Winding leaves P3 (8 of 13,950 fit) but stays at P4 (592 fit).
    # Out-and-back stays at P3: 194 verified loops fit the budget, so the shape is
    # demonstrably achievable there.
    PHASE_FAMILIES = {
        1: (), 2: (), 3: (0, 2), 4: (0, 2, 3), 5: (0, 1, 2, 3, 4),
        6: (0, 1, 2, 3, 4),
    }

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
            # Worst-case completion pay compounds ALL gates (hill x length x quality x
            # style x family floors).
            flat = (params.completion_hill_floor * params.completion_length_floor
                    * params.completion_quality_floor * params.completion_style_floor
                    * params.completion_family_floor * params.R_complete)
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
        if phase >= 6:  # P6 "Style": layout variety without losing the quality won in P5.
            # Every earlier phase converged onto ONE rectangle motif because nothing paid
            # for shape. Variety legs: graded turn count, S-bends, and handedness BALANCE
            # (a rectangle is all-one-direction turns), each mirrored in the qualified
            # gate with a tested-excitement floor slightly below the measured ceiling.
            return RewardParams(
                R_quality_max=500.0,
                step_cost=0.0,
                w_h=0.0,
                completion_quality_floor=0.4,
                completion_length_floor=0.5,     # quick-loop trap (Jul-15): the 28pc
                                                  # attractor must leave length money
                                                  # on the table from step one
                completion_style_floor=1.0,      # superseded by the family gate: the
                                                  # fixed turns+balance ramp fought the
                                                  # seed on every non-winding family
                completion_family_floor=0.4,     # ignore your seed -> forfeit 60% of
                                                  # the completion payout, from step one.
                                                  # Lowered 0.5 -> 0.4 (Aug-12): R_family
                                                  # (200) and R_qualify (200) are both
                                                  # gated on tested excitement >=
                                                  # qualify_min_excitement=4.5, so while
                                                  # unaided quality sat at ~2.6 both bonuses
                                                  # were structurally unpayable and the
                                                  # shape choice lost to the reliable oval
                                                  # even at this floor -- unaided builds
                                                  # ignored the seed on ~99% of episodes,
                                                  # which was RATIONAL: at the P6 120-piece
                                                  # budget the requested shapes close less
                                                  # often (measured, 10 unaided episodes/
                                                  # seed: oval 90%, out-and-back 40%,
                                                  # winding 60%), and at floor=0.5 an
                                                  # out-and-back seed's EV was
                                                  # 0.90*1000*0.625=562 for an oval build
                                                  # vs 0.40*(1000+200+200)=560 for the
                                                  # requested shape -- a virtual tie the
                                                  # oval edged. Unaided quality has since
                                                  # reached ~5.5 (cold-only episode window
                                                  # + recent cold harvests, two independent
                                                  # measurements), making R_family/
                                                  # R_qualify payable, so at 0.4 the same
                                                  # comparison becomes 495 vs 560 -- the
                                                  # requested shape wins by ~13%. A matched
                                                  # build is unaffected either way (its
                                                  # family_match is 1.0, so the gate is 1.0
                                                  # at any floor); only mismatched builds
                                                  # forfeit more.
                R_family=200.0,
                qualify_requires_family=True,
                exc_gate_target=6.0,
                R_struct_max=250.0,
                struct_w_chain=0.0,
                struct_w_single_drop=0.35,
                struct_single_drop_target=12.0,
                struct_w_drop_runs=0.15,
                struct_drop_runs_target=2.0,
                struct_w_length=0.30,
                struct_length_target=70.0,
                struct_w_banked=0.15,
                struct_banked_target=4.0,
                struct_w_turns=0.0,               # target comes from the seed now
                struct_turns_target=12.0,
                struct_w_sbend=0.05,
                struct_sbend_target=4.0,
                struct_w_turn_balance=0.0,        # switches measure this better
                # ...and the TARGET has to go too (Aug-9 final review), not just the
                # weight: _exc_feature_quality reads the target directly, so at 2.0 the
                # dense w_exc_feat potential kept paying a balance leg that min(left,
                # right) makes structurally unreachable for the 0-switch families. On an
                # oval seed the gain (w_exc_feat/6 = 1.0) exactly cancelled the family
                # potential's loss for the same switch, zeroing the dense gradient on the
                # very axis that separates the bands. 0.0 = the leg returns a constant.
                struct_turn_balance_target=0.0,   # remaining weights sum to 1.0
                R_viable=150.0,
                R_caps_max=250.0,
                R_exc_milestone=100.0,
                exc_milestone_bars=(2.5, 4.0, 5.5),
                R_qualify=200.0,
                qualify_min_excitement=4.5,
                qualify_min_turns=0.0,            # a fixed turns>=12 bar contradicts the
                qualify_min_turn_balance=0.0,     # oval/spiral/out-and-back seeds outright
                qualify_max_sbend=6.0,            # ...but turns>=12 also blocked the Aug-6
                                                  # S-bend farm by accident, and the
                                                  # footprint classifier is blind to
                                                  # S-bends, so the density cap has to be
                                                  # explicit. 6 is a TUNABLE, not a derived
                                                  # constant: struct_sbend_target=4 already
                                                  # saturates the S-bend struct leg, so 6
                                                  # leaves honest headroom while the
                                                  # observed farm (8) stays blocked.
                qualify_requires_test=True,
                w_exc_feat=6.0,
                w_family=6.0,  # dense per-piece pull toward the seed's requested family
                               # (Aug-9): the completion gate + R_family bonus are both
                               # terminal-only, and terminal-only shaping has repeatedly
                               # been too slow here (this style gate itself ran ~900k
                               # steps without reaching cold builds).
                w_route=3.0,   # Jul-27: wound layouts fail closure on the RETURN ROUTE
                               # (cold winding attempts truncate; rectangles never do).
                               # The P1-4 angular-progress potential taught exactly that
                               # navigation and was retired in P5 when the memorized
                               # rectangle stopped needing it -- novel shapes still do.
            )
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
                completion_length_floor=0.5,     # quick-loop trap (Jul-15): the 28pc
                                                  # attractor must leave length money
                                                  # on the table from step one
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
                # Family reward ramp, last rung before P6 (see the P3 block's comment for
                # the full rationale and the table): all 5 families are active here
                # (PHASE_FAMILIES), the floor keeps loosening, the dense potential reaches
                # its full P6 weight and R_family steps up to 125 (Aug-9 final review).
                # qualify_requires_family stays off -- P5 advances on its own
                # length-ladder-on-cold-success predicate, which a family leg would redefine.
                completion_family_floor=0.60,
                w_family=6.0,
                R_family=125.0,
                # Retired here for the same reason as in P6 (Aug-9 final review):
                # struct_w_turn_balance is already 0, but _exc_feature_quality reads the
                # TARGET, and w_exc_feat=6.0 is live in P5 -- so at the 2.0 default the
                # unreachable-for-oval/spiral balance leg exactly cancelled the family
                # potential on the switch axis. See the P6 block.
                struct_turn_balance_target=0.0,
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
                # Family reward, armed (Aug-9 gap fix): PHASE_FAMILIES already varies the
                # seed from P3 onward (oval/out-and-back), but until now nothing in
                # the reward read it, so the observation carried a one-hot that predicted
                # nothing -- the "pure noise" PHASE_FAMILIES' own docstring says P1-2 must
                # avoid. The spec's Phase-3 early read (non-zero, rising per-family hit
                # rates by ~1 day of training) needs this armed to be measurable at all.
                # Ramps with the track budget across P3/P4/P5 rather than switching on at
                # full P6 strength, because P3/P4 are solved/tuned and a large completion
                # share must not shift onto a skill they aren't teaching yet:
                #   completion_family_floor: 0.85 (P3) -> 0.75 (P4) -> 0.60 (P5) -> 0.50 (P6)
                #   w_family:                 3.0 (P3) ->  4.0 (P4) ->  6.0 (P5) ->  6.0 (P6)
                #   R_family:                 0.0 (P3) -> 75.0 (P4) -> 125.0 (P5) -> 200 (P6)
                # R_family joined the ramp at P4 (Aug-9 final review): below P6 the
                # multiplicative gate was the ONLY family incentive, and at these floors it
                # forfeits too little on the NEAR bands to pay for the shape -- an out-and-back
                # seed diverges structurally from the default oval in P3, requiring a dedicated
                # shape reward to make the cost visible; at P5/P6 where spiral and winding join,
                # R_family ramps to 125+ points to cover their documented higher piece counts.
                # P3 stays at 0: ride testing is off there, so
                # R_family (paid only inside the completion-AND-tested branch) could never
                # fire. qualify_requires_family stays False below P6 -- each of P3/4/5 has
                # its own tuned advancement predicate that a family leg would change the
                # meaning of. Per-family hit rates are still tracked every phase
                # (episode_family_results) against whatever that phase's `qualified` means,
                # which is what makes the early read readable without touching the gate.
                # See docs/superpowers/specs/2026-08-09-seed-conditioned-coaster-variety-design.md.
                completion_family_floor=0.85,
                w_family=3.0,
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
                # Family reward ramp, next rung after P3 (see the P3 block's comment for
                # the full rationale and the table): floor loosens, weight rises, and
                # R_family arms here -- P4 is the first phase whose ride test runs, so it
                # is the first phase where a post-test discrete bonus can pay at all.
                # 75 (not P6's 200) keeps it a rung on the ramp: it is ~half of the P4
                # gate's own near-band margin, enough to clear the shape's build cost
                # without re-balancing the tuned R_viable/R_qualify economics around it.
                completion_family_floor=0.75,
                w_family=4.0,
                R_family=75.0,
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
    def _history_turn_count(base_env):
        """Turn-family pieces in the history (mirrors env._turn_count)."""
        history = getattr(base_env.track_builder, 'history', [])
        return sum(1 for h in history if h.get('action') in TURN_ACTIONS)

    @staticmethod
    def _history_family_hit(base_env, skip=0):
        """Whether the build lands in the family the episode's seed asked for
        (mirrors env._family_hit).

        `skip` drops that many leading history entries, so a caller can ask about the
        AGENT-BUILT suffix instead of the whole track -- the same distinction
        LoopRecord.agent_turn_count draws between composed and inherited structure.
        Default 0 = the whole track, which is what the reward gate scores."""
        history = getattr(base_env.track_builder, 'history', [])
        return classify_family([h.get('action') for h in history[skip:]]) == int(
            getattr(base_env, 'target_family', 0))

    @staticmethod
    def _history_sbend_count(base_env):
        """S-bend pieces in the history (mirrors env._sbend_count). They weave without
        changing the heading, so the footprint classifier cannot see them -- the density
        cap is the only thing that keeps them from padding a qualified build."""
        history = getattr(base_env.track_builder, 'history', [])
        return sum(1 for h in history if h.get('action') in SBEND_ACTIONS)

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
        None only for phase 1, which has no structural gate; every other phase (2-6) has
        its own branch below."""
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
        if self.current_phase >= 6:
            # "Style": completed, tested at the E floor, shaped like the family the
            # episode's seed asked for, and not padded with S-bends. Aug-9: the fixed
            # turns>=12 / balance>=2 legs are gone -- they were only ever a stand-in for
            # "not the same rectangle again", and they contradict an oval/spiral/
            # out-and-back seed outright. The S-bend density cap keeps the one thing
            # turns>=12 blocked by accident (the Aug-6 farm), since the footprint
            # classifier is blind to S-bends. Mirrors env._qualifies'
            # qualify_requires_family + qualify_max_sbend legs.
            P = self._phase_reward_params(6)
            return bool(success
                        and getattr(base_env, '_last_test_ok', False)
                        and float(getattr(base_env, 'last_ride_excitement', 0.0))
                        >= P.qualify_min_excitement
                        and self._history_family_hit(base_env)
                        and self._history_sbend_count(base_env) <= P.qualify_max_sbend)
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
            6: (self.phase6_max_length, False),   # style phase: full budget, testing ON
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
        # P6 opening-seed mode: deep scaffold draws replay a 6-piece winding OPENING
        # instead of dissolving to cold, so the one skill the cold conversion is stuck
        # on (starting a build with a jog) finally gets perpetual practice.
        # Arm both the working floor and its demote ceiling; the prefix anneal
        # (Aug-3) walks min_prefix 6 -> 0 on floor-bound success, never above init.
        # Do not re-arm on every settings refresh: P6 refreshes (e.g. harvest-cap
        # updates) must not reset a partially-annealed floor back to 6.
        if self.current_phase >= 6:
            if getattr(self._annealer, "min_prefix_init", 0) != 6:
                ov = getattr(self, "_warm_min_prefix_override", None)
                self._annealer.min_prefix = 6 if ov is None else max(0, min(6, ov))
                self._annealer.min_prefix_init = 6
        else:
            self._annealer.min_prefix = 0
            self._annealer.min_prefix_init = 0

        if self.verbose >= 1:
            phase_names = {
                1: "Return Practice",
                2: f"Lift Hill Building {self.phase2_stage_name()}",
                3: "Drop & Turn",
                4: "Circuit Mastery",
                5: "Quality Optimization",
                6: "Style / Variety"
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
                # Jul-15 (reversing Jul-12): rung advances KEEP the anneal frontier.
                # Resetting per rung discarded k progress ~4x per run segment while the
                # pool only ever grows -- closure skill transfers across budgets.

                self._clear_phase_windows()
                self.phase_episode_count = 0

                if self.verbose >= 1:
                    print(f"\n{'='*60}")
                    print(f"📈 PHASE 5: Advancing to sub-stage {self.phase5_stage}")
                    print(f"   Max track length: {self.phase5_current_length}")
                    print(f"   Success rate achieved: {success_rate:.1%}")
                    print(f"{'='*60}\n")

                return True

            # Ladder topped out: open Phase 6 (Style) once quality HOLDS on cold
            # episodes -- the P5 qualified diagnostic (tested E >= 4) over the cold
            # window, same cold-only gating discipline as every phase.
            if (self.phase5_current_length >= self.phase5_target_length
                    and len(self.episode_qualified_results) >= 50):
                qualified_rate = sum(self.episode_qualified_results) / len(
                    self.episode_qualified_results
                )
                if qualified_rate >= self.phase6_entry_threshold:
                    self._advance_to_phase(6)
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
        for window in self.episode_family_results.values():
            window.clear()

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
                5: ("Quality Optimization", "Optimizing for ride ratings"),
            6: ("Style / Variety", "Winding layouts at held quality")
            }
            name, desc = phase_names.get(new_phase, (f"Phase {new_phase}", ""))

            print(f"\n{'='*70}")
            print(f"🎯 ADVANCING TO PHASE {new_phase}: {name}")
            print(f"   {desc}")
            print(f"   Previous phase success rate: {success_rate:.1%}")
            print(f"{'='*70}\n")

    def _sample_target_family(self):
        """The episode's seed: a family drawn from the phase's active set (0 when the
        phase has none, so the observation input stays constant rather than noisy)."""
        active = self.PHASE_FAMILIES.get(self.current_phase, ())
        return self._family_rng.choice(active) if active else 0

    def _sample_warm_start(self):
        """This episode's warm-start plan. Cold when disabled, during evaluation (eval must
        measure the true task), or past the scaffolded phases (1-5). Each phase prefers
        loops that can actually satisfy its gate (with graceful pool fallback)."""
        if (not self.warm_start_enabled or not self._track_stats
                or self.current_phase > 6):
            return WarmStartPlan(prefix=[], k=0, loop_len=0, cold=True)
        self._loop_library.maybe_refresh()   # pick up other workers' harvested loops
        base_env = self._get_base_env()
        budget = getattr(base_env, 'max_track_length', 40)
        # Family-filtered scaffold (Aug-9): gated generically on PHASE_FAMILIES rather
        # than a per-branch flag, so it activates automatically for every phase whose
        # family reward is armed and stays exactly None (unchanged) for phases 1-2,
        # where PHASE_FAMILIES is empty. target_family must already be THIS episode's
        # draw by the time we read it here -- see reset()'s ordering. Computed BEFORE
        # the excitement-ratchet branches below (fix pass, Aug-9 review) so the P5/P6
        # best_excitement() calls can be scoped to the same family the pool will draw
        # from -- otherwise the ratchet bar comes from the whole library while the pool
        # is narrowed to one family, and any family lacking the library's top exemplar
        # gets an unreachable bar by construction.
        family = int(getattr(base_env, "target_family", 0)) \
            if self.PHASE_FAMILIES.get(self.current_phase) else None
        min_chains, min_len, min_drop_z, min_steep_z = 1, 0, 0, 0
        min_single_drop_z, min_excitement, min_turns = 0, 0.0, 0
        if self.current_phase == 2 and self.phase2_stage >= 3:
            min_chains = 3
        elif self.current_phase == 3:
            min_chains, min_len, min_drop_z = 2, 20, 4
        elif self.current_phase == 4:
            # Match the P4 gate: qualifying length AND a steep segment. min_len was 25
            # (the P3 bar) which let non-steep 26-38 piece harvests dominate the pool
            # while the 60-degree leg went unpracticed (Jul-8 run: 12h, zero own steep).
            min_chains, min_len, min_drop_z, min_steep_z = 3, 40, 8, 8
        elif self.current_phase >= 6:
            # P6 (Style): exemplar-shaped. min_turns=8 (removed task 6b, Aug-9 review) was
            # a leftover proxy for "not another rectangle" from before family seeding
            # existed, and it was silently unsatisfiable for two of the five families:
            # oval is <=5 turns BY DEFINITION (footprint.py FAMILIES), so no oval record
            # could ever clear an 8-turn bar -- measured against the live 196,058-record
            # library, 0 of 148,511 oval records met it (out_and_back was only partly
            # hit: 21,470 of 29,235). That made the family-narrowing check in
            # LoopLibrary.pool() (which requires a same-family record to clear the FULL
            # structural bar before it narrows) unable to ever narrow to those two
            # families, so an oval- or out_and_back-seeded episode fell back to the
            # unnarrowed pool and got scaffolded with an off-family exemplar before the
            # agent placed a piece. Shape is now the family filter's job (`family` above);
            # left at the default 0. The pool's per-bin cap still keeps multiple styles
            # in every draw.
            min_chains, min_len, min_drop_z, min_single_drop_z = 1, 40, 12, 12
            min_excitement = 0.8 * self._loop_library.best_excitement(budget, family=family)
        elif self.current_phase >= 5:
            # P5 (Jul-9): scaffold from excitement exemplars. Shape criteria mirror the
            # rating caps (>=12z single drop on a >=40 piece loop); the excitement bar
            # SELF-RATCHETS at 0.8 x the best tagged loop fitting the budget -- 0 on a
            # legacy-only pool (everything qualifies), then rising behind every better
            # exemplar the harvest tags. No persistent state; recomputed per episode.
            min_chains, min_len, min_drop_z, min_single_drop_z = 1, 40, 12, 12
            min_excitement = 0.8 * self._loop_library.best_excitement(budget, family=family)
        return self._annealer.sample_plan(
            self._loop_library, self.current_phase, budget,
            min_chains=min_chains, min_len=min_len, min_drop_z=min_drop_z,
            min_steep_z=min_steep_z, min_single_drop_z=min_single_drop_z,
            min_excitement=min_excitement, min_turns=min_turns, family=family)

    def reset(self, **kwargs):
        """Reset environment and check for phase advancement"""
        self._check_phase_advancement()

        # The episode's seed (footprint family) is drawn FIRST and set on the base env
        # (which does NOT reset it in reset() -- see PHASE_FAMILIES / _sample_target_family)
        # so that _sample_warm_start below reads THIS episode's family, not the previous
        # one's (Aug-9 fix: the two calls used to run in the opposite order, which meant
        # every episode's scaffold pool was filtered by last episode's seed).
        base_env = self._get_base_env()
        base_env.target_family = self._sample_target_family()

        # Stage this episode's warm-start prefix on the base env AFTER the advancement
        # check (so phase/max_length are current); the env replays it one-shot inside
        # reset(), before Phi seeding. The suffix k sizes the tight scaffolded budget.
        self._current_plan = self._sample_warm_start()
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
            5: "Quality Optimization",
                6: "Style / Variety"
        }.get(self.current_phase, f"Phase {self.current_phase}")
        if self.current_phase == 2:
            info['phase2_stage'] = self.phase2_stage
            info['phase2_threshold'] = self._phase2_threshold()

        if self.current_phase == 5:
            info['phase5_stage'] = self.phase5_stage
            info['max_track_length'] = self.phase5_current_length
            # Self-imitation ratchet diagnostics: the excitement bar that actually gated
            # THIS episode's pool, and the best tagged exemplar it trails (watch the bar
            # climb behind better harvests). Family-scoped (Task 7 fix, resolution 5) the
            # same way _sample_warm_start scopes the real gate -- an unscoped query here
            # would report a different (higher) number whenever the drawn family lacks
            # the library's cross-family top exemplar. Empty for phases 1-2
            # (PHASE_FAMILIES has no entries there), leaving the query unscoped and this
            # branch dead anyway (current_phase == 5 never holds in phases 1-2).
            family = int(getattr(base_env, "target_family", 0)) \
                if self.PHASE_FAMILIES.get(self.current_phase) else None
            best_exc = self._loop_library.best_excitement(
                self.phase5_current_length, family=family)
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
                # The prefix descent may only shrink the opening seed when the build had
                # the SHAPE this phase is teaching, not merely when it closed -- see
                # WarmStartAnnealer.record_outcome. What "shape" means is phase-dependent,
                # and the branch is load-bearing: phases whose reward still pays for turns
                # keep the fixed turn-count bar bit-identical, while a seed-conditioned
                # phase (qualify_requires_family, P6 only today) pays maximally at <=5
                # turns on an oval seed -- there the turn bar would refuse to shrink the
                # seed for a correctly built oval and the descent would stall forever.
                # WHICH TRACK the family is judged on depends on where the opening came
                # from (Aug-9 final review):
                #  * pool() NARROWED to the requested family -> the replayed opening is
                #    itself from a record of that family, so the shape the seed asked for
                #    is the shape of the WHOLE track and that is what must be scored.
                #    Scoring the suffix instead drops the prefix's ~2 turns and ~1 switch
                #    -- a whole band for the high-turn families. Measured on the 196,058-
                #    record deployment library, a record's suffix still classifies as its
                #    own family for 100% of oval, 100% of spiral, 73.5% of out_and_back,
                #    92.3% of winding and 0.0% of SERPENTINE records; with uniform seeds
                #    that caps the floor frontier at (1+1+.735+.923+0)/5 = 0.732, and at a
                #    realistic 80% closure rate 0.586 -- under promote_rate (0.60) and over
                #    demote_rate (0.15), so min_prefix would never anneal to 0 and the
                #    settling point that "is itself the measurement" would be an artifact.
                #  * narrowing FELL BACK (no same-family record cleared the phase's
                #    structural bar) -> the opening is an arbitrary off-family exemplar,
                #    typically a winding jog whose direction switch an oval seed forbids.
                #    Scoring the whole track there would let the scaffold decide the
                #    predicate against the agent and stall the descent at min_prefix_init,
                #    which is the regression the suffix rule was added for. Keep it.
                if self._phase_reward_params(self.current_phase,
                                             self.phase2_stage).qualify_requires_family:
                    plan = self._current_plan
                    narrowed = bool(getattr(self._loop_library,
                                            'last_family_narrowed', False))
                    skip = 0 if narrowed else (len(plan.prefix) if plan is not None else 0)
                    styled = self._history_family_hit(base_env, skip=skip)
                else:
                    styled = self._history_turn_count(base_env) >= self.FLOOR_STYLE_MIN_TURNS
                self._annealer.record_outcome(self._current_plan, success, styled=styled)
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

            # Per-family hit tracking (Aug-9 seed conditioning): cold-only, like every
            # other gate window -- a scaffolded build inherits its shape from the
            # exemplar, so counting it would measure the scaffold rather than the policy.
            # `qualified` is None on phases without a structural gate (e.g. phase 1),
            # where the seed is pinned to 0 and family_hit is reward-inert noise; bool()
            # collapses that to False rather than letting a bare None poison the window.
            z = int(base_env.target_family)
            # NOTE: this is env._family_hit() (openrct2_env.py:1583) ANDed with
            # loop_completed -- not the suffix-only variant _is_qualified uses for P6's
            # gate -- so a truncated build that happens to classify into the seed's
            # family reads 0 here, not a false hit. On a warm episode this reports the
            # scaffold's shape, not the agent's -- read it together with `cold_start`.
            family_hit = bool(info.get('episode_metrics', {}).get('family_hit', 0.0))
            if cold:
                self.episode_family_results[z].append(bool(family_hit and qualified))
            info['target_family'] = z
            info['family_hit'] = float(family_hit)
            # Emit the rate (+ its family_n_ sample count) only for families the current
            # phase actually draws from -- matching qualified_rate's convention (:1113)
            # of gating the KEY itself on applicability, not just its value. 0.0 is kept
            # as the empty-window fallback so an active family with zero cold samples
            # still reports a number, but family_n_{z}==0 tells that number apart from a
            # real all-miss 0.0.
            for fz in self.PHASE_FAMILIES.get(self.current_phase, ()):
                window = self.episode_family_results[fz]
                info[f'family_hit_rate_{fz}'] = (
                    sum(window) / len(window) if window else 0.0)
                info[f'family_n_{fz}'] = len(window)

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
            info['warm_min_prefix'] = self._annealer.min_prefix
            info['loop_library_size'] = len(self._loop_library)
            # Fix pass (Aug-9 review): pool()'s family-narrowing decision, read straight
            # off the library the same way library_best_excitement is below -- was a
            # family requested this episode, and did the narrowing apply or fall back to
            # the phase's structural criteria (Fix 1)? None means no family was active.
            info['warm_family_requested'] = self._loop_library.last_family_requested
            info['warm_family_narrowed'] = self._loop_library.last_family_narrowed
            frontier_rate = self._annealer.frontier_rate
            if frontier_rate is not None:
                info['warm_frontier_rate'] = frontier_rate
            if self.current_phase >= 5:
                # Ratchet diagnostics must ride the STEP done-info (the TB callback never
                # sees reset infos -- the reset()-side copy exists for humans/debuggers).
                # Family-scoped (Task 7 fix, resolution 5) -- see the matching comment in
                # reset() for why an unscoped query here would misreport the bar that
                # actually gated this episode's pool.
                family = int(getattr(base_env, "target_family", 0)) \
                    if self.PHASE_FAMILIES.get(self.current_phase) else None
                best_exc = self._loop_library.best_excitement(
                    self.phase5_current_length, family=family)
                info['library_best_excitement'] = best_exc
                info['p5_pool_exc_bar'] = 0.8 * best_exc

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
