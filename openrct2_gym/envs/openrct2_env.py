import gymnasium as gym
import numpy as np
import time
import json
import os
from collections import deque
from dataclasses import dataclass
from .api_track_builder import APITrackBuilder
from .api_controller import APIController
from .obs_config import (
    make_observation_space, SEQ_LEN, HIST_FEAT_DIM, MAP_SHAPE, SCALE, H_SCALE,
)
from .warm_start import LoopLibrary


@dataclass(frozen=True)
class RewardParams:
    """All tunables for the unified, potential-based reward.

    The CORE Phi geometry weights (w_xy/w_z/w_dir/w_e) and normalizers are frozen across
    all curriculum phases; the auxiliary Phi terms (w_h, w_return, w_close, w_route) are
    deliberately phase-gated (PBRS invariance holds within a phase; phase changes shift
    the potential once, at a reset). Sparse-objective fields (R_complete gating, struct,
    milestones, R_quality_max, step_cost) vary per phase.
    """
    gamma: float = 0.99            # MUST equal the PPO discount for PBRS invariance
    # --- Phi potential weights (fixed across phases) ---
    w_xy: float = 10.0             # horizontal alignment to the closing tile
    w_z: float = 6.0               # height alignment to station height
    w_dir: float = 6.0             # heading alignment to the closing direction
    w_e: float = 2.0               # energy viability bonus
    # --- Phi normalizers (decoupled from obs SCALE/H_SCALE; see plan #6) ---
    d_xy: float = 40.0
    # Directional approach corridor: when >0 (and the entry axis is calibrated) the horizontal w_xy
    # pull is DIRECTIONAL -- it rewards closing in ALONG the entry axis from the -entry (approach)
    # side within a cone of this half-width (the cone widens 1 tile per tile away from the dock so
    # the head is caught as it rounds back onto the corridor) and gives ZERO pull from the wrong
    # side. This removes the isotropic w_xy attractor that let the head minimise distance by parking
    # behind/beside the station (approaching the dock from the wrong direction). 0 restores the
    # legacy isotropic radial w_xy term. Pure state function -> PBRS-clean.
    approach_perp_range: float = 2.0
    d_z: float = 20.0              # m_z slope = w_z/d_z = 0.3/z. Must stay steep enough that
                                   # the energy term's chain-lift bump (~+0.47) cannot make
                                   # climbing profitable where it shouldn't be (at d_z=60 the
                                   # 0.1/z slope lost to it and phase 1 climbed instead of
                                   # completing). High-altitude reach comes from m_z being
                                   # UNCLIPPED, not from a large d_z.
    e_scale: float = 50.0
    # --- Discovery potential (makes climbing findable; ON in hill phases 2-4, the curriculum
    #     sets w_h=0 for phase 1/5 since an always-on climb pull derails Phase-1 completion) ---
    w_h: float = 3.0               # banked-elevation weight in Phi (0 disables the term).
                                   # 3, not 6: still tilts climb>flat (~+0.5/z vs the 0.1/z
                                   # m_z cost) but halves the attractor a wrecked policy can
                                   # settle into instead of completing
    h_scale: float = 6.0           # elevation normalizer (z-units); saturate at ~a 3-chain hill
    # --- Route potential (fills the west-side shaping hole; ON in completion phases 1-4) ---
    # The directional approach cone deliberately gives ZERO horizontal pull for along<0, so the
    # detour AROUND the station (the first half of every loop) had no gradient -- the Jun-24 run
    # parked ~5 tiles out and never completed. w_route pays angular progress of the head's bearing
    # around the station center: 0 on the start/west bearing rising monotonically to w_route on the
    # approach/east bearing, along BOTH the north and south detours. Bounded, radius-independent,
    # pure function of position -> PBRS-clean; its maximum sits where the w_xy cone + w_close
    # funnel already dominate, so it cannot create a new parking optimum. 0 disables (P5 default).
    w_route: float = 0.0
    # --- Sparse real objectives ---
    R_complete: float = 1000.0     # fixed completion bonus across all phases
    R_quality_max: float = 0.0     # 0 disables quality (phases 1-4); 500 in phase 5
    # --- Structural objective (completion-conditioned; phase-scaled, rewards lift hills/drops) ---
    R_struct_max: float = 0.0      # 0 in P1/P5; 250 in P2-4
    struct_chain_target: int = 3   # chain-lift count for full chain credit (matches the phase gate)
    struct_w_chain: float = 1.0    # weight on the chain-lift component
    struct_w_drop: float = 0.0     # weight on the drop component (graded: total drop-z / target)
    struct_drop_target: float = 1.0   # total drop-z for full drop credit (1 == legacy any-drop)
    # Scale components (P3/P4 redesign: "bigger but viable"). Weights default 0 -> every
    # pre-redesign params object computes the exact same structural quality as before.
    struct_w_height: float = 0.0      # weight on chain-banked height / struct_height_target
    struct_height_target: float = 4.0
    struct_w_length: float = 0.0      # weight on completed track length / struct_length_target
    struct_length_target: float = 25.0
    # Steepness component (P4): z dropped via 60-degree pieces (actions 8/27/28) toward
    # struct_steep_target. P4's qualified gate requires a 60-degree segment, but that leg was
    # reward-INVISIBLE outside the full R_qualify conjunction -- 9h of fixed-P4 training never
    # placed one steep piece (entropy at the collapse line makes joint discovery ~impossible).
    # Steepness is a piece-type swap (not extra pieces), so a graded additive ramp is enough.
    struct_w_steep: float = 0.0       # weight on steep-dropped z / struct_steep_target
    struct_steep_target: float = 8.0  # one minimal 25->60->25 segment (4z + 4z)
    # Verified-viability bonus (P4+): paid on completion ONLY when the ride test returned
    # nonzero stats -- "the train physically made it around" as a paid, verified event.
    R_viable: float = 0.0
    # Fraction of R_complete a HILL-LESS completion earns. 1.0 = no gating (P1/P5); 0.25 in
    # the hill phases means a flat loop pays only 250 while a full hill pays the rest -- a
    # flat-completer can no longer ignore the lift hill (it leaves 75% of R_complete on the
    # table). Completion stays first: even the gated floor (250) >> Phi_max (~27).
    completion_hill_floor: float = 1.0
    # Length analogue of the hill gate (P3/P4). Fraction of the (hill-gated) completion payout
    # a min-length loop earns; the rest ramps in with track_length/struct_length_target. 1.0 =
    # no gating. The ADDITIVE struct length credit cannot carry this incentive: at P3 scale it
    # pays ~+2/piece while gamma-discounting the ~1200-point completion costs ~-10/piece, so
    # the Jul-5 overnight run converged on an 18-piece mini-loop with qualified_rate 0. At 0.25
    # each piece toward the target is worth ~R_complete*0.75/target (~+30/piece in P3) -- the
    # gradient finally points at the phase gate.
    completion_length_floor: float = 1.0
    # Discrete qualification bonus: paid once, on completion, when the episode meets the
    # phase's qualified-gate targets (the same struct targets the curriculum advances on; see
    # _qualifies). Makes crossing the gate itself a paid event instead of reward-invisible.
    # 0 disables (P1/P2/P5).
    R_qualify: float = 0.0
    qualify_requires_energy: bool = False      # P3: cheap viability proxy (ride testing is off)
    qualify_requires_steep_drop: bool = False  # P4: a 60-degree descent segment is required
    qualify_requires_test: bool = False        # P4: the ride test must return real stats
    # Round-trip milestone: a one-time-per-episode bonus for climbing >= roundtrip_gain above
    # the station AND returning to ~station height, INDEPENDENT of closing the loop. Teaches
    # the climb-and-return sub-skill decoupled from completion (a hill is then a reachable
    # add-on to the agent's closure skill). Kept below the gated completion reward so it never
    # lures the agent away from completing. 0 disables it (P1/P5).
    R_roundtrip: float = 0.0
    roundtrip_gain: float = 3.0    # min CHAIN-banked elevation (z above station) that counts as
                                   # "climbed". 3, not 4: the canonical hill [10,9,13] banks chain
                                   # gain 3 (the crest piece isn't chained) -- a 4.0 bar made the
                                   # milestone + w_return descent shaping silently inert for the
                                   # exact hill the curriculum teaches.
    # Summit milestone: the reachable FIRST HALF of the round-trip — a one-time-per-episode
    # bonus for chain-climbing >= roundtrip_gain above the station, INDEPENDENT of returning or
    # completing. Pays the climb before the return so the climb-and-return conjunction becomes a
    # gradient (place chain -> climb -> summit -> learn return -> round-trip). Kept strictly
    # below R_roundtrip so the return is still worth learning. 0 disables it (P1/P5/P3/P4).
    R_summit: float = 0.0
    # Descent/return shaping (PBRS Phi term): a climb-gated potential that is 0 at/above the
    # summit threshold height (STATION_HEIGHT + roundtrip_gain) -- so crossing the threshold
    # injects no Phi jump and cannot re-pay the summit -- and rises to w_return as the head
    # returns to station height. This is the dense downhill gradient that lets the agent LEARN
    # the return (ascent had w_h; descent had no per-step shaping). 0 disables it (P1/P5); >0 in
    # the hill phases 2-4. Part of the single policy-invariant Phi, so it telescopes.
    w_return: float = 0.0
    # Near-closure densification: a STEEP, local Phi bonus in the final `close_range` tiles (at
    # station height) so the agent gets a strong gradient to COMMIT to the closing tile instead of
    # drifting near it and churning. The gentle w_xy term (0.25/tile over 40) is too flat to drive
    # the last-piece closure in a cold start. Pure state function -> PBRS-clean. 0 disables it; the
    # curriculum turns it on in the completion phases 1-4 (off in phase 5, completion already mastered).
    w_close: float = 0.0
    close_range: float = 3.0       # tiles ALONG the entry axis within which the funnel ramps to w_close
    close_perp_range: float = 2.0  # corridor half-width PERPENDICULAR to the entry axis (off-axis falloff)
    # Funnel taper: the perpendicular corridor opens from a POINT at the throat (along=0) to the
    # full close_perp_range over this many tiles of along-distance, so the off-axis tiles right
    # beside the dock (the throat row) are EXCLUDED -- the head must be ON the entry axis to dock.
    # 0 = no pinch (legacy rectangular corridor).
    close_throat_pinch: float = 1.0
    close_z_range: float = 2.0     # height tolerance (z) for the bonus
    exc_target: float = 8.0
    exc_sigma: float = 1.0
    int_target: float = 5.5
    int_sigma: float = 1.0
    nausea_max: float = 4.5
    nausea_tau: float = 1.0
    q_w_exc: float = 0.5
    q_w_int: float = 0.3
    q_w_nausea: float = 0.2
    # --- costs (negative; added to reward) ---
    fail_penalty: float = -0.1
    step_cost: float = 0.0


class OpenRCT2Env(gym.Env):
    # Calibrated closing geometry, shared across in-process env instances
    # (DummyVecEnv) and backed by a JSON file for SubprocVecEnv / later runs.
    _close_cache = None
    _CLOSE_CACHE_PATH = "logs/close_geometry.json"
    # Robust calibration: accumulate completion closing-records and lock the anchor only once
    # several completions AGREE (a reproducible closure, not a one-off fluke). Replaces locking
    # the first completion, which poisoned Phi for whole runs when that closure was atypical.
    _close_records = []
    _CLOSE_MIN_CONSISTENT = 3
    # Warm-start loop library (reverse curriculum): every completion's action sequence is
    # harvested here; the curriculum wrapper replays prefixes from it at reset. Shared
    # across workers via atomic JSONL appends (see warm_start.LoopLibrary).
    _LOOP_LIBRARY_PATH = "logs/loop_library.jsonl"
    def __init__(self, render_mode=None, host="localhost", port=8080, verbose=1):
        super(OpenRCT2Env, self).__init__()
        self.render_mode = render_mode
        self.verbose = verbose  # 0=silent, 1=important only, 2=detailed
        self.api_controller = APIController(host, port, verbose)
        self.track_builder = APITrackBuilder(self.api_controller, verbose=verbose)
        self.skip_ride_testing = False  # Can be set by wrappers to skip entrance/exit and testing

        # Unified potential-based reward config (wrappers override per phase).
        self.reward_params = RewardParams()
        self._phi_prev = 0.0  # Phi(s) carried between steps for PBRS shaping

        # Warm-start reverse curriculum: the wrapper stages a loop prefix here before
        # reset(); the env replays it one-shot (see _apply_warm_start / warm_start.py).
        self.warm_start_actions = None
        self.warm_start_suffix_k = None  # planned agent-built suffix length (sizes the budget)
        self._warm_prefix_len = 0
        self._warm_cold = True
        self._warm_aborted = False       # prefix replay failed mid-way (infrastructure event)
        self._warm_track_cap = None      # scaffolded-episode piece budget (None = full budget)
        self._warm_step_cap = None       # scaffolded-episode step budget (None = full budget)
        self._loop_library = None  # lazily bound to _LOOP_LIBRARY_PATH on first harvest

        # Connect to API server
        if not self.api_controller.connect():
            raise RuntimeError(f"Failed to connect to OpenRCT2 API server at {host}:{port}")
        
        # Define action and observation space (32 actions: 0-30 track pieces, 31 remove)
        self.action_space = gym.spaces.Discrete(32)
        
        # Track collision and backtracking
        self.collision_count = 0
        self.consecutive_failures = 0
        self.steps_since_collision = 0  # Track steps since last collision for remove masking
        self.auto_backtrack_enabled = True
        self.max_consecutive_failures = 3
        # A timed-out request may have been APPLIED server-side (placements are not
        # idempotent): the client head is then possibly desynced from the real track.
        # When set, the episode is sacrificed (truncated) -- resetEpisode is retry-safe
        # and rebuilds both sides from a clean slate.
        self._connection_suspect = False

        # Initialize state variables
        self.current_position = None
        self.goal_position = None
        self.current_direction = 0
        self.track_pieces = []
        self.track_length = 0
        self.max_track_length = 250
        self.max_steps = 256
        self.station_length = self.api_controller.station_length
        self.steps = 0
        self.loop_completed = False
        self.last_piece_type = 0
        self.chain_lift_count = 0
        self.max_chain_lifts = 15
        self.last_action = None
        self.previous_distance = None
        self.position_history = deque(maxlen=self.POSITION_HISTORY_MAXLEN)
        self.chain_lift_positions = set()  # Track positions where chain lifts were placed
        self.height_history = []  # Track height at each piece for energy calculations

        # Additional metrics tracking
        self.min_distance_reached = float('inf')
        self.max_height_reached = 0
        self.unique_positions = set()
        self.phase_rewards = {'building': 0, 'transition': 0, 'return': 0}
        self.current_phase = 'building'
        self.episode_rewards = []
        self.last_height = 14  # Starting height
        self.used_piece_types = set()  # Track variety of pieces used

        # Ride stats from last completed ride (for observation space)
        self.last_ride_excitement = 0.0
        self.last_ride_intensity = 0.0
        self.last_ride_nausea = 0.0

        # Must match the OpenRCT2 API's direction encoding (probe_corridor.py: a straight from the
        # station output -- API dir 0 -- moves -X). The env stores RAW API directions, and these
        # vectors build the egocentric observation (goal_disp / angle_to_goal / forward-aligned map),
        # so a mismatched convention rotates the agent's whole spatial sense 90deg from reality.
        self.direction_vectors = [
            (-1, 0),  # 0: West   (API dir 0; the train docks heading this way)
            (0, 1),   # 1: North
            (1, 0),   # 2: East
            (0, -1),  # 3: South
        ]

        # Redesigned observation space: egocentric 2.5D map + structured build-history
        # buffer (memory) + goal-relative scalars. See obs_config.make_observation_space.
        self.observation_space = make_observation_space()

    def step(self, action):
        # Store previous position for tracking
        prev_position = self.current_position.copy()
        original_action = action
        auto_backtracked = False

        # Capture the entry a remove will pop, so chain-lift bookkeeping can be reverted.
        removed_entry = self.track_builder.history[-1] if (action == 31 and self.track_builder.history) else None
        success, new_position, new_direction = self.track_builder.take_action(action, self.current_position, self.current_direction)

        # A failure caused by a request TIMEOUT is not a normal collision: the server may
        # have applied the action anyway, desyncing the head. Sacrifice the episode.
        if not success and getattr(self.api_controller, "last_request_timed_out", False):
            self._connection_suspect = True

        # Track collisions and failures
        if not success and action != 31:  # Failed to place a piece (not a remove action)
            self.consecutive_failures += 1
            self.collision_count += 1
            self.steps_since_collision = 0  # Reset collision window

            # Auto-backtrack if enabled and too many consecutive failures
            if self.auto_backtrack_enabled and self.consecutive_failures >= self.max_consecutive_failures:
                if len(self.track_builder.history) > 0:
                    if self.verbose >= 2:
                        print(f"Auto-backtracking after {self.consecutive_failures} consecutive failures")
                    # Force a remove action instead of the failed action
                    action = 31  # Change the action to remove
                    auto_backtracked = True
                    removed_entry = self.track_builder.history[-1] if self.track_builder.history else None
                    success, new_position, new_direction = self.track_builder.take_action(action, self.current_position, self.current_direction)
                    self.consecutive_failures = 0
        elif success:
            # Only a SUCCESS is a recovery: a failed remove must not reset the counter
            # (it would delay the next auto-backtrack indefinitely).
            self.consecutive_failures = 0
            self.steps_since_collision += 1
        
        # Check if the last placement completed the circuit
        # Trust the API's isCircuitComplete flag - if the game says it's complete, it's complete
        if success and len(self.track_builder.history) > 0:
            last_entry = self.track_builder.history[-1]
            if last_entry.get("is_complete", False):
                self._last_placement_complete = True
                if self.verbose >= 1:
                    print(f"✅ Loop completed with piece {action}")
            else:
                self._last_placement_complete = False
        else:
            self._last_placement_complete = False
        
        # Update state based on the action that was actually executed
        if success:
            if action == 31:  # Remove piece
                if self.track_pieces:
                    self.track_pieces.pop()
                    self.track_length -= 1
                if self.height_history:
                    self.height_history.pop()
                self._revert_chain_lift(removed_entry)
                self.last_piece_type = action
                self.current_position = new_position
                self.current_direction = new_direction
                self.position_history.append(self.current_position.copy())
            else:
                # Single bookkeeping path shared with the warm-start prefix replay.
                self._record_placement(action, new_position, new_direction)

        self.last_action = action

        # Check for loop completion BEFORE calculating reward
        self.loop_completed = self._last_placement_complete
        if self.loop_completed:
            if self.verbose >= 1:
                print(f"Loop has been completed, great success!")
            # Learn the closing geometry from the first completion (applied next reset).
            self._maybe_capture_closing_geometry()
            # Harvest the loop into the warm-start library (dedup'd; scaffolded repeats no-op).
            self._harvest_completed_loop()
        terminated = self.loop_completed

        # Now calculate reward (with correct loop_completed status)
        observation = self._get_observation()
        reward = self._calculate_reward(success, action)
        # A forced auto-backtrack executes a remove ON BEHALF of a FAILED agent action: the
        # remove's PBRS delta must not replace the failure's penalty (the chosen action
        # would otherwise escape it exactly when it fails repeatedly).
        if auto_backtracked:
            reward += self.reward_params.fail_penalty

        # Track metrics that don't depend on the final reward value.
        current_distance = self._calculate_distance_to_start()[0]
        self.min_distance_reached = min(self.min_distance_reached, current_distance)
        self.max_height_reached = max(self.max_height_reached, self.current_position[2])
        self.unique_positions.add(tuple(self.current_position))

        # Check if episode was truncated
        truncated = self._is_trunkated()
        self.steps += 1
        
        # Create info dict with additional debugging information
        info = {
            'auto_backtracked': auto_backtracked,
            'original_action': original_action if auto_backtracked else action,
            'collision_count': self.collision_count,
            'consecutive_failures': self.consecutive_failures,
            'loop_completed': self.loop_completed,
            # Surface per-step metrics here so the training callback can log them from
            # self.locals['infos'] instead of issuing SubprocVecEnv get_attr/env_method
            # collectives (extra synchronized round-trips to all workers every step).
            'track_length': self.track_length,
            'current_distance': current_distance,
            # Warm-start provenance: gates and the entropy controller must distinguish
            # cold (true-task) episodes from scaffolded ones; aborted prefixes are
            # infrastructure events excluded from annealer accounting.
            'cold_start': self._warm_cold,
            'warm_prefix_len': self._warm_prefix_len,
            'warm_aborted': self._warm_aborted,
        }
        
        if self.verbose >= 2 and (self.steps % 10 == 0 or auto_backtracked):  # Log less frequently or when backtracking
            print("Step: %s, Track: %s, Pos: %s, Action: %s%s, Dist: %.1f, Dir: %s" % 
                  (self.steps, self.track_length, self.current_position, 
                   self.last_action, " (auto-backtrack)" if auto_backtracked else "",
                   self._calculate_distance_to_start()[0], self.current_direction))

        if terminated:
            # Additional completion bonuses are already in _calculate_reward
            # Only place entrance/exit and test ride if not in early learning phases
            if not self.skip_ride_testing:
                # Place entrance and exit before testing
                entrance_result = self.api_controller.place_entrance_exit()
                if entrance_result.get("success"):
                    if self.verbose >= 1:
                        print("✅ Entrance and exit placed successfully")
                else:
                    if self.verbose >= 1:
                        print(f"⚠️ Failed to place entrance/exit: {entrance_result.get('error', 'Unknown error')}")
                
                # Start ride test using the correct API endpoint
                test_result = self.api_controller.start_ride_test()
                if test_result.get("success"):
                    if self.verbose >= 2:
                        print(f"🎢 Ride test started: {test_result.get('payload', '')}")

                    # Smart polling for ride stats. Early-exits as soon as POSITIVE stats land,
                    # so this cap only bites when the test never rates (train stalled or still
                    # running). The train needs real time to complete a circuit before RCT2
                    # rates the ride, so the wait is a tunable (default 15s) -- a timeout
                    # honestly reports unrated (test_ok False) instead of junk stats.
                    ride_rating = self._poll_for_ride_stats(
                        max_wait=getattr(self, 'ride_test_max_wait', 15))
                else:
                    if self.verbose >= 1:
                        print(f"⚠️ Failed to start ride test: {test_result.get('error')}")
                    ride_rating = {"excitement": 0, "intensity": 0, "nausea": 0}

                info['ride_rating'] = ride_rating
                # Store ride stats for next episode's observation
                self.last_ride_excitement = float(ride_rating.get('excitement', 0))
                self.last_ride_intensity = float(ride_rating.get('intensity', 0))
                self.last_ride_nausea = float(ride_rating.get('nausea', 0))
            else:
                # Skip testing in early phases - agent is just learning to build circuits
                if self.verbose >= 1:
                    print("🎯 Loop completed! (Skipping ride testing in learning phase)")
                info['ride_rating'] = {"excitement": 0, "intensity": 0, "nausea": 0}

            # Graduated ride-quality bonus: the single authority for quality. Kept out of
            # _calculate_reward because ride stats only arrive after the post-completion
            # ride test. Gated to 0 when disabled (R_quality_max==0, phases 1-4) or when
            # the ride was skipped/untested (all stats zero), so completion is never punished.
            rr = info['ride_rating']
            # POSITIVE stats only: the unrated sentinel (-0.01/0/0) is truthy but means the
            # train never demonstrably ran.
            self._last_test_ok = bool(rr.get('excitement', 0) > 0 or rr.get('intensity', 0) > 0
                                      or rr.get('nausea', 0) > 0)
            reward += self._quality_bonus(
                rr.get('excitement', 0), rr.get('intensity', 0), rr.get('nausea', 0),
                self.reward_params)
            # Verified-viability bonus: the train demonstrably made it around (nonzero test
            # stats). Completion-conditioned by construction (this is the terminal branch).
            if self.reward_params.R_viable > 0.0 and self._last_test_ok:
                reward += self.reward_params.R_viable
            # Qualification bonus: the phase gate as a paid, discrete event. Sits with
            # R_viable (not in _calculate_reward) because P4's predicate needs the ride-test
            # verdict, which only settles here.
            if self.reward_params.R_qualify > 0.0 and self._qualifies(self.reward_params):
                reward += self.reward_params.R_qualify
                self._last_qualify_bonus = self.reward_params.R_qualify
        # Truncation gets no partial-credit bonus: PBRS already pays approach progress
        # as-you-go (a non-potential terminal bonus would be farmable and break invariance).

        # Reward accounting AFTER the terminal quality bonus is finalized, so episode_rewards
        # and phase_rewards match the reward actually returned (no phase-5 under-reporting).
        self.episode_rewards.append(reward)
        if self.track_length <= 25:
            self.current_phase = 'building'
        elif self.track_length <= 40:
            self.current_phase = 'transition'
        else:
            self.current_phase = 'return'
        self.phase_rewards[self.current_phase] += reward

        # Store episode-level metrics before the environment resets
        if terminated or truncated:
            info['episode_metrics'] = {
                'min_distance': self.min_distance_reached,
                'chain_lift_count': self.chain_lift_count,
                'chain_count': sum(1 for h in self.track_builder.history if h.get('action') in (9, 10)),
                'struct_bonus': getattr(self, '_last_struct_bonus', 0.0),
                'roundtrip': float(getattr(self, '_roundtrip_awarded', False)),
                'summit': float(getattr(self, '_summit_awarded', False)),
                'return_potential': float(self._return_potential(self.reward_params)),
                'route_potential': float(self.reward_params.w_route * self._route_progress()),
                'max_gain': max((h['next_position'][2] - self.STATION_HEIGHT
                                 for h in self.track_builder.history), default=0.0),
                'track_length': self.track_length,
                'phase_rewards': dict(self.phase_rewards),
                'collision_count': self.collision_count,
                'loop_completed': self.loop_completed,
                'warm_prefix_len': self._warm_prefix_len,
                # scale/viability diagnostics (P3-5 redesign)
                'drop_z': self._total_drop_z(),
                'steep_drop_z': self._steep_drop_z(),
                'chain_height': float(self._chain_max_gain()),
                'test_ok': bool(getattr(self, '_last_test_ok', False)),
                # length-trap fix diagnostics: the combined completion gate actually paid
                # and whether the phase-gate qualification bonus fired
                'completion_gate': float(getattr(self, '_last_completion_gate', 0.0)),
                'qualify_bonus': float(getattr(self, '_last_qualify_bonus', 0.0)),
            }

        return observation, reward, terminated, truncated, info

    def valid_action_mask(self):
        """
        Returns a boolean mask of valid actions.
        True = action is valid, False = action is invalid
        """
        mask = np.zeros(self.action_space.n, dtype=bool)
        
        # Get valid actions from the API
        valid_actions = self.track_builder.get_valid_actions()
        
        # Only allow remove action when needed (recent collision or stuck)
        # This prevents unnecessary remove-loops
        if len(self.track_builder.history) > 0:
            # Allow remove if: just had a collision OR within 3 steps of collision
            if self.consecutive_failures > 0 or self.steps_since_collision <= 3:
                valid_actions.append(31)
        
        # Set valid actions to True
        for action in valid_actions:
            if 0 <= action < self.action_space.n:
                mask[action] = True

        # If no valid actions, at least allow remove
        if not mask.any():
            if len(self.track_builder.history) > 0:
                mask[31] = True
            else:
                # Nothing valid and nothing to remove == the API failed us: an all-False
                # mask makes MaskablePPO sample uniformly over KNOWN-INVALID actions (all
                # masked logits equal). Allow everything for this step and sacrifice the
                # episode instead of grinding silent failures.
                self._connection_suspect = True
                mask[:] = True

        return mask
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all state variables FIRST
        # API hardcodes the station start at [61, 66, 14] - we must match this
        self.station_start_position = [61, 66, 14]  # Where the first station piece is (matches API)
        self.current_position = self.station_start_position.copy()
        # Goal = the STAGING tile the closing piece is placed FROM -- one tile on the approach side
        # (opposite the entry direction) of the dock. probe_corridor.py confirmed: from this tile,
        # heading the entry direction, SEVEN different pieces dock the circuit at station_start. The
        # head can never sit ON the dock tile (occupied by BeginStation), so the goal must be this
        # adjacent staging tile. (Reverts the goal-=-dock change, which sent the head onto the station
        # tile and collapsed completion to 0%.)
        _ex, _ey = self.direction_vectors[self._STATION_ENTRY_DIR]
        self.goal_position = [
            self.station_start_position[0] - _ex,
            self.station_start_position[1] - _ey,
            self.station_start_position[2],
        ]
        # Phi's geometric anchor: calibrated closing head if known, else provisional.
        self._init_closing_target()
        self.current_direction = 0
        self.track_pieces = []
        self.track_length = 0
        self.steps = 0
        self.loop_completed = False
        self.last_piece_type = 0
        self.chain_lift_count = 0
        self.last_action = None
        self.track_builder.history.clear()  # Clear the history when resetting the environment
        self.track_builder.valid_track_types = None  # Invalidate valid-piece cache
        self._last_placement_complete = False
        self.collision_count = 0
        self.consecutive_failures = 0
        self.steps_since_collision = 0
        self._connection_suspect = False
        self._last_test_ok = False
        self._last_qualify_bonus = 0.0
        self._last_completion_gate = 0.0
        self.previous_distance = None
        self.position_history = deque(maxlen=self.POSITION_HISTORY_MAXLEN)
        self.chain_lift_positions = set()
        self.height_history = []  # Reset height tracking
        self.min_distance_reached = float('inf')
        self.max_height_reached = 0
        self.unique_positions = set()
        self.phase_rewards = {'building': 0, 'transition': 0, 'return': 0}
        self.current_phase = 'building'
        self.episode_rewards = []
        self.last_height = 14  # Starting height
        self.used_piece_types = set()  # Track variety of pieces used
        # Don't reset ride stats - keep them for learning across episodes

        # Reset the episode. Prefer the batched server-side endpoint (one round-trip:
        # demolish-all + create-ride + build-station) since under heavy parallelism every
        # synchronized vector-step waits on the slowest worker's reset. Fall back to the
        # legacy multi-call path for older plugins or on failure.
        reset_episode = getattr(self.api_controller, "reset_episode", None)
        payload = reset_episode(
            station_length=self.station_length,
            start=tuple(self.station_start_position),
            start_direction=0,
        ) if callable(reset_episode) else None

        if payload is not None:
            self._apply_batched_reset(payload)
        else:
            self._legacy_reset()

        # Warm-start prefix replay (reverse curriculum): the wrapper may have staged a
        # known-completable loop prefix. Replayed BEFORE the Phi seeding below, so the
        # first agent step gets no shaping windfall from the pre-built track.
        self._apply_warm_start()

        # Once-per-episode climb milestones must pay AGENT work only: pre-latch whatever the
        # PREFIX already satisfies (a hill prefix that summited-and-returned otherwise banks
        # R_summit+R_roundtrip on the agent's first step, every scaffolded episode). A prefix
        # cut at the summit leaves the round-trip earnable -- the descent IS the agent's work.
        _prefix_summited = (self._warm_prefix_len > 0
                            and self._chain_max_gain() >= self.reward_params.roundtrip_gain)
        _head_z = self.current_position[2]
        self._summit_awarded = _prefix_summited
        self._roundtrip_awarded = (
            _prefix_summited
            and abs(_head_z - self.STATION_HEIGHT) <= 1
            and _head_z < self.STATION_HEIGHT + self.reward_params.roundtrip_gain)

        # Seed the PBRS potential at the true starting head (post-station-build + prefix).
        self._phi_prev = self._potential(self.reward_params)

        observation = self._get_observation()
        info = {}
        return observation, info

    def _record_placement(self, action, new_position, new_direction):
        """Bookkeeping for one successfully PLACED piece -- the single path shared by
        step() and the warm-start prefix replay, so prefix pieces are indistinguishable
        from agent placements to every consumer (obs history buffer, chain gates, energy
        model, track budget)."""
        self.track_length += 1
        self.track_pieces.append(action)
        self.height_history.append(new_position[2])  # Track height at this piece
        # Chain-lift bookkeeping (count distinct chain-lift endpoints; reverted on
        # remove by _revert_chain_lift). Kept out of the reward, which no longer
        # pays a chain-lift bonus.
        if action in (9, 10):
            pos_key = tuple(new_position)
            if (pos_key not in self.chain_lift_positions
                    and self.chain_lift_count < self.max_chain_lifts):
                self.chain_lift_count += 1
                self.chain_lift_positions.add(pos_key)
        self.last_piece_type = action
        self.current_position = new_position
        self.current_direction = new_direction
        # Track position history for trend analysis
        self.position_history.append(self.current_position.copy())

    def _apply_warm_start(self):
        """Replay the staged warm-start prefix (one-shot); returns pieces actually placed.

        Every placed piece goes through _record_placement (identical bookkeeping to agent
        placements). A failed placement aborts the remainder -- keep what placed, the
        episode continues from there. If a prefix piece unexpectedly closes the circuit
        (impossible for k>=1 library prefixes unless geometry drifts) it is removed and
        the prefix aborted: reset() must never yield a completed episode. Emits no reward
        and consumes no steps; prefix pieces DO consume the max_track_length budget.
        """
        actions = self.warm_start_actions
        suffix_k = self.warm_start_suffix_k
        self.warm_start_actions = None            # one-shot: auto-resets can't replay stale plans
        self.warm_start_suffix_k = None
        self._warm_cold = not actions
        self._warm_prefix_len = 0
        self._warm_aborted = False
        self._warm_track_cap = None
        self._warm_step_cap = None
        if not actions:
            return 0
        for action in actions:
            success, new_position, new_direction = self.track_builder.take_action(
                int(action), self.current_position, self.current_direction)
            if not success:
                self._warm_aborted = True
                if self._warm_prefix_len == 0:
                    # Nothing placed: the episode is bit-identical to a cold one -- classify
                    # it as cold so recurring infra hiccups don't starve the cold gate windows.
                    self._warm_cold = True
                if self.verbose >= 1:
                    print(f"⚠️ Warm-start prefix aborted at piece "
                          f"{self._warm_prefix_len + 1}/{len(actions)} (placement failed); "
                          f"episode continues from here")
                break
            self._record_placement(int(action), new_position, new_direction)
            self._warm_prefix_len += 1
            if self.track_builder.history and self.track_builder.history[-1].get("is_complete"):
                removed_entry = self.track_builder.history[-1]
                ok, pos, direction = self.track_builder.take_action(
                    31, self.current_position, self.current_direction)
                if ok:
                    if self.track_pieces:
                        self.track_pieces.pop()
                        self.track_length -= 1
                    if self.height_history:
                        self.height_history.pop()
                    self._revert_chain_lift(removed_entry)
                    self.last_piece_type = 31   # mirror step()'s remove branch for the obs
                    self.current_position = pos
                    self.current_direction = direction
                    self.position_history.append(self.current_position.copy())
                    self._warm_prefix_len -= 1
                self._warm_aborted = True
                if self.verbose >= 1:
                    print("⚠️ Warm-start prefix closed the circuit unexpectedly; "
                          "reopened and aborted the prefix")
                break
        # Tight scaffolded-episode budget: the episode exists to practice the LAST k
        # decisions, so a failed attempt must truncate fast. Without this, a wandering
        # policy burned ~100 steps/episode at k<=3 (first smoke run) and the docking
        # gradient drowned in noise. Cold episodes keep the full phase budget. An ABORTED
        # prefix leaves the head far from closure -- the tight budget would make the episode
        # geometrically impossible, so aborted episodes run with the full budget instead
        # (and the wrapper excludes them from annealer accounting via the info flag).
        if self._warm_prefix_len > 0 and suffix_k and not self._warm_aborted:
            self._warm_track_cap = self._warm_prefix_len + int(suffix_k) + self.WARM_TRACK_SLACK
            self._warm_step_cap = self.WARM_STEP_FACTOR * (int(suffix_k) + self.WARM_TRACK_SLACK)
        return self._warm_prefix_len

    def _harvest_completed_loop(self):
        """Persist this completion's action sequence into the warm-start loop library
        (dedup makes scaffolded repeats no-ops; cold discoveries are new material).
        Best-effort: library I/O must never break training."""
        try:
            if len(self.track_builder.history) > self.HARVEST_MAX_LEN:
                return
            if self._loop_library is None or self._loop_library.path != self._LOOP_LIBRARY_PATH:
                self._loop_library = LoopLibrary(self._LOOP_LIBRARY_PATH)
            self._loop_library.add(
                LoopLibrary.record_from_history(self.track_builder.history))
        except Exception:
            pass

    def _calculate_reward(self, success, action):
        """Unified potential-based reward: ``reward = F + sparse terms``.

        F = gamma*Phi(s') - Phi(s) is policy-invariant shaping (Ng/Harada/Russell 1999),
        so place-then-remove telescopes to ~(gamma-1)*Phi < 0 and cannot be farmed (the
        old symmetric-removal hack is gone). On true completion Phi(s')=0 (standard
        telescoping) and the large R_complete is added on top; the ride-quality bonus is
        added separately in step() after the ride test. A failed placement returns a flat
        penalty OUTSIDE PBRS and must NOT advance _phi_prev (the head did not move).
        """
        params = self.reward_params
        self._last_struct_bonus = 0.0          # reset each step (covers the failure early-return)
        self._last_completion_gate = 0.0
        if not success:
            return float(params.fail_penalty)

        phi_next = 0.0 if self.loop_completed else self._potential(params)
        reward = params.gamma * phi_next - self._phi_prev
        self._phi_prev = phi_next

        reward += params.step_cost
        if self.loop_completed:
            # Hill-gated completion: a flat loop earns only completion_hill_floor of
            # R_complete; building the lift hill earns the rest (plus the structural bonus).
            # In phases 1/5 the floor is 1.0 so completion is fully paid regardless.
            gate = params.completion_hill_floor + (1.0 - params.completion_hill_floor) * self._hill_quality(params)
            # Length gate (multiplicative): a short loop discounts the WHOLE completion payout,
            # so unlike the additive struct length credit it out-bids the gamma-discount cost
            # of building the extra pieces the phase gate requires.
            if params.completion_length_floor < 1.0 and params.struct_length_target > 0:
                length_frac = min(self.track_length / params.struct_length_target, 1.0)
                gate *= (params.completion_length_floor
                         + (1.0 - params.completion_length_floor) * length_frac)
            self._last_completion_gate = gate
            reward += params.R_complete * gate
            # Structural bonus, computed once and stored so episode_metrics reports exactly
            # what was added to the reward.
            self._last_struct_bonus = self._structural_bonus(params)
            reward += self._last_struct_bonus

        # Round-trip elevation milestone (completion-independent, once per episode): reward the
        # CHAIN climb-and-return motion so it can be learned as an add-on to the closure skill.
        # Keyed on _chain_max_gain (not all-history) so it requires an actual chain lift -- this
        # matches the wrapper's chain_count>=1 qualifier and stops a non-chain climb from farming
        # the bonus or burning the once-per-episode flag.
        if params.R_roundtrip > 0.0 and not getattr(self, "_roundtrip_awarded", False):
            head_z = self.current_position[2]
            # "Returned" == near station height (matching the wrapper's abs<=1 mirror; the
            # old z<=station+1 paid a climb-then-DIVE) and strictly below the required-climb
            # threshold -- at roundtrip_gain=1 the +1 tolerance CONTAINED the summit, so one
            # chain stub fired summit+roundtrip+gate with no return ever happening.
            if (self._chain_max_gain() >= params.roundtrip_gain
                    and abs(head_z - self.STATION_HEIGHT) <= 1
                    and head_z < self.STATION_HEIGHT + params.roundtrip_gain):
                reward += params.R_roundtrip
                self._roundtrip_awarded = True

        # Summit milestone (the reachable first half of the round-trip, once per episode):
        # reward chain-climbing to the threshold even before returning, so the climb is a
        # gradient the agent can follow out of the flat-loop optimum.
        if params.R_summit > 0.0 and not getattr(self, "_summit_awarded", False):
            if self._chain_max_gain() >= params.roundtrip_gain:
                reward += params.R_summit
                self._summit_awarded = True

        if getattr(self, "verbose", 0) >= 2:
            print("Reward was: %.3f (Phi'=%.3f, dist: %.1f, track: %d)" % (
                reward, phi_next, self._calculate_distance_to_start()[0], self.track_length))
        return float(reward)

    def _reward_target_position(self):
        """The geometric target Phi and the observation aim at.

        Once the closing geometry is calibrated (first completion), this is the
        measured pre-close head tile; before calibration it is the provisional
        guide tile ``goal_position``.
        """
        cp = getattr(self, "close_pos", None)
        return cp if cp is not None else self.goal_position

    def _reward_target_direction(self):
        """The calibrated closing heading (0-3), or None before calibration."""
        return getattr(self, "close_dir", None)

    # ----------------------------------------------------- closing-geometry calibration
    # goal_position is only a guide tile; real closure is the API's isCircuitComplete
    # flag on a placed piece's endpoint. So Phi's geometric anchor is learned: on the
    # first completion we capture the pre-close head state and reuse it thereafter.

    @staticmethod
    def _closing_record_from_history(history):
        """Pre-close head state + closing-piece geometry from a completed history.

        Uses the completing entry's ``position``/``direction`` (the head the closing
        piece was placed FROM), not ``next_position`` (the post-close endpoint). The
        closing ``action``/``track_type`` are kept for the sufficiency check.
        """
        if not history:
            return None
        entry = history[-1]
        return {
            "pos": list(entry["position"]),
            "dir": int(entry["direction"]),
            "action": int(entry["action"]),
            "track_type": int(entry.get("track_type", -1)),
        }

    def _persist_close_cache(self, record):
        """Set the process-wide cache and best-effort write it to disk (atomic)."""
        OpenRCT2Env._close_cache = record
        path = self._CLOSE_CACHE_PATH
        try:
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(record, f)
            os.replace(tmp, path)
        except OSError:
            pass  # in-memory cache still set; persistence is best-effort

    def _load_close_cache(self):
        """Return the calibration record from memory, else from disk, else None.

        Only a VALIDATED record is cached. A corrupted/old file is treated as no cache
        (returns None without poisoning the in-memory cache), so a later completion can
        still calibrate and overwrite it — otherwise _maybe_capture would early-exit forever.
        """
        if OpenRCT2Env._close_cache is not None:
            return OpenRCT2Env._close_cache
        try:
            with open(self._CLOSE_CACHE_PATH) as f:
                record = json.load(f)
        except (OSError, ValueError):
            return None
        if not self._valid_close_record(record):
            return None
        OpenRCT2Env._close_cache = record
        return record

    @staticmethod
    def _robust_close_anchor(records):
        """A trustworthy closing anchor from accumulated completion records, or None.

        Locks an anchor only once >= _CLOSE_MIN_CONSISTENT completions share the SAME closing
        direction -- i.e. the agent has REPRODUCED that closure, so it is a reliable target, not
        a one-off fluke. Uses the per-axis median position and modal closing piece of that
        direction group, so an occasional offset/odd closure is ignored. This replaces locking
        the first (often atypical) completion, which poisoned Phi for entire runs.
        """
        from collections import Counter
        if len(records) < OpenRCT2Env._CLOSE_MIN_CONSISTENT:
            return None
        best_dir, best_n = Counter(r["dir"] for r in records).most_common(1)[0]
        if best_n < OpenRCT2Env._CLOSE_MIN_CONSISTENT:
            return None
        group = [r for r in records if r["dir"] == best_dir]
        pos = [int(round(float(np.median([r["pos"][i] for r in group])))) for i in range(3)]
        action = Counter(r["action"] for r in group).most_common(1)[0][0]
        track_type = Counter(r["track_type"] for r in group).most_common(1)[0][0]
        return {"pos": pos, "dir": int(best_dir), "action": int(action), "track_type": int(track_type)}

    def _maybe_capture_closing_geometry(self):
        """Accumulate completion closing-geometries; lock Phi's anchor only once several
        completions AGREE (see _robust_close_anchor), so a single fluky first closure can no
        longer poison Phi for the rest of the run.

        Applied at the NEXT reset (never mid-episode) so Phi stays self-consistent within an
        episode; a no-op once the anchor is locked.
        """
        if not getattr(self, "loop_completed", False):
            return
        if not self.track_builder.history:
            return
        if OpenRCT2Env._close_cache is not None:
            return
        # Empirical closing-geometry probe: logs only UNTIL the anchor locks (its purpose --
        # confirming the deterministic closing connection -- is fulfilled then; unbounded
        # per-completion appends over a long run are pure disk growth).
        self._probe_log_closing(self.track_builder.history[-1])
        record = self._closing_record_from_history(self.track_builder.history)
        if record is None:
            return
        # rebind to a new list (don't mutate the shared class default); cap so it cannot grow
        # unbounded if completions never agree
        OpenRCT2Env._close_records = (OpenRCT2Env._close_records + [record])[-50:]
        anchor = self._robust_close_anchor(OpenRCT2Env._close_records)
        if anchor is not None:
            self._persist_close_cache(anchor)
            if getattr(self, "verbose", 0) >= 1:
                print(f"[calibrate] closing geometry locked after "
                      f"{len(OpenRCT2Env._close_records)} completions: "
                      f"head={anchor['pos']} dir={anchor['dir']} closing_action={anchor['action']}")

    def _probe_log_closing(self, entry):
        """TEMP instrumentation: append every completion's full closing geometry to a JSONL beside
        the calibration cache, so the deterministic closing connection (esp. next_direction, the
        heading INTO the station) can be confirmed empirically from real completions.

        Numpy ints must be coerced (json can't serialize int64), and this MUST NOT raise -- it is
        optional instrumentation, so any failure is swallowed rather than crashing the env worker."""
        def _i(v):
            return None if v is None else int(v)
        def _l(v):
            return [int(x) for x in (v if v is not None else [])]
        try:
            path = os.path.join(os.path.dirname(self._CLOSE_CACHE_PATH) or ".", "closing_probe.jsonl")
            with open(path, "a") as f:
                f.write(json.dumps({
                    "pos": _l(entry.get("position")),
                    "dir": _i(entry.get("direction")),
                    "action": _i(entry.get("action")),
                    "track_type": _i(entry.get("track_type")),
                    "next_position": _l(entry.get("next_position")),
                    "next_direction": _i(entry.get("next_direction")),
                }) + "\n")
        except Exception:
            pass   # instrumentation must never break training

    @staticmethod
    def _valid_close_record(cache):
        """A calibration record is usable only with a 3-D position and a cardinal dir 0-3.
        Guards against a corrupted/old logs/close_geometry.json producing an IndexError
        in _potential's heading term."""
        try:
            return (cache is not None
                    and len(cache.get("pos", [])) == 3
                    and cache.get("dir") in (0, 1, 2, 3))
        except (TypeError, AttributeError):
            return False

    def _init_closing_target(self):
        """Set close_pos/close_dir from calibration if known/valid, else the DETERMINISTIC station
        geometry. close_dir defaults to the station-entry axis (North = _STATION_ENTRY_DIR), which a
        closing-geometry probe confirmed is invariant (every completion enters BeginStation heading
        North), so the Phi heading term guides the closing approach from step 1 instead of staying
        off until a completion calibrates it (the chicken-and-egg that stalled Phase-1 bootstrap).
        Calibration can still refine close_pos/close_dir from real completions."""
        cache = self._load_close_cache()
        if self._valid_close_record(cache):
            self.close_pos = list(cache["pos"])
            self.close_dir = int(cache["dir"])
        else:
            self.close_pos = None                       # -> provisional guide tile (goal_position)
            self.close_dir = self._STATION_ENTRY_DIR    # North: confirmed deterministic entry axis

    def _potential(self, params):
        """Potential Phi(s) for PBRS shaping (F = gamma*Phi(s') - Phi(s)).

        Rewards alignment of the build head to the closing state across all three
        closure axes (horizontal position, height, heading) plus an energy-viability
        bonus. Pure function of current state + track_builder.history (no API call),
        bounded ABOVE by w_xy+w_z+w_dir+w_e+w_h and maximized at the calibrated closing
        (pos, dir) with positive energy margin. The height misalignment m_z is
        deliberately UNCLIPPED so Phi keeps falling (0.3/z) at any altitude -- a clipped
        m_z left a zero-gradient plateau above +d_z where lost climbers wandered with no
        pull home. The heading term is omitted until the closing direction is calibrated
        (avoids a wrong-heading reward wall) and is COUPLED to the dock (gated by near_xy):
        it only rewards aligning to the closing heading as the head docks, leaving the agent
        free to turn while routing the loop.
        """
        target = self._reward_target_position()
        px, py, pz = self.current_position
        tx, ty, tz = target
        close_dir = self._reward_target_direction()
        m_z = abs(pz - tz) / params.d_z
        # numerically-stable logistic of the energy margin -> viability in (0, 1)
        v = 0.5 * (1.0 + np.tanh(0.5 * self._calculate_energy_margin() / params.e_scale))
        # Entry-axis projection of the head (shared by the broad approach term below, the
        # near-closure funnel further down, and the observation's corridor scalars).
        along, perp = self._corridor_coords()
        # DIRECTIONAL horizontal approach: reward closing in ALONG the entry corridor (a cone that
        # widens away from the dock so the head is caught as it rounds back to the -entry side) and
        # give ZERO pull from the wrong side -- removing the isotropic attractor that let the head
        # park behind/beside the station. Falls back to the isotropic radial term before the axis is
        # calibrated (close_dir None) or when approach_perp_range == 0.
        if along is not None and params.approach_perp_range > 0.0:
            if along >= 0.0:
                perp_tol = params.approach_perp_range + along
                m_approach = max(0.0, 1.0 - along / params.d_xy) * max(0.0, 1.0 - perp / perp_tol)
            else:
                m_approach = 0.0
        else:
            m_approach = 1.0 - min(1.0, float(np.hypot(tx - px, ty - py)) / params.d_xy)
        phi = (params.w_xy * m_approach
               + params.w_z * (1.0 - m_z)
               + params.w_e * v)
        # Route progress: dense angular pull around the station toward the approach side, filling
        # the along<0 hole the directional cone leaves on the whole start side (see RewardParams).
        if params.w_route > 0.0:
            phi += params.w_route * self._route_progress()
        # Discovery term: highest elevation banked via CHAIN-LIFT pieces (a plain climb of the
        # same geometry earns nothing here). Keying on the chain flag makes action 9/10 strictly
        # better than the identical-geometry plain climb 5/11, so the dense gradient points at the
        # chain lift the Phase-2 gate actually counts. Pure function of the removal-safe history
        # (max recomputes lower on remove, so it telescopes; descent doesn't lower the banked peak).
        chain_gain = min(self._chain_max_gain(), params.h_scale)
        phi += params.w_h * (chain_gain / params.h_scale)
        # Descent/return shaping: 0 at/above the summit (no double-pay), rising on the way home.
        phi += self._return_potential(params)
        # Steep, local near-closure FUNNEL: ramps to w_close only in the final close_range tiles at
        # station height, so the agent gets a strong gradient to COMMIT to the closing tile. Reuses
        # the (along, perp) entry-axis projection computed above (the API closes the circuit only
        # from the -entry-side staging tile heading the entry direction; probe_corridor.py). Rewards
        # only the on-axis approach from the correct side. Pure state function -> PBRS-clean.
        if along is not None and params.close_range > 0.0:
            near_along = max(0.0, 1.0 - along / params.close_range) if along >= 0.0 else 0.0
            # Perp tolerance opens from a point at the throat (along=0) to close_perp_range over
            # close_throat_pinch tiles, so the off-axis tiles beside the dock are excluded.
            if params.close_perp_range <= 0.0:
                near_perp = 1.0
            else:
                perp_range = (params.close_perp_range * min(1.0, max(0.0, along) / params.close_throat_pinch)
                              if params.close_throat_pinch > 0.0 else params.close_perp_range)
                if perp_range > 0.0:
                    near_perp = max(0.0, 1.0 - perp / perp_range)
                else:
                    near_perp = 1.0 if perp <= 0.0 else 0.0   # throat: only the exact centerline
            near_corridor = near_along * near_perp
        else:
            near_corridor = 1.0
        if params.w_close > 0.0:
            near_z = max(0.0, 1.0 - abs(pz - tz) / params.close_z_range)
            phi += params.w_close * near_corridor * near_z
        # Heading alignment to the entry direction, gated by the corridor (rewarded only as the head
        # lines up to dock, leaving it free to turn while routing the loop).
        if close_dir is not None:
            cur = self.direction_vectors[self.current_direction]
            tgt = self.direction_vectors[close_dir]
            cos_theta = cur[0] * tgt[0] + cur[1] * tgt[1]  # unit cardinal vectors
            m_dir = (1.0 - cos_theta) / 2.0
            phi += params.w_dir * (1.0 - m_dir) * near_corridor
        return float(phi)

    def _corridor_coords(self):
        """(along, perp) of the head's offset from the closing target, projected onto the
        entry axis; (None, None) until the closing direction is known.

        along > 0 == on the -entry (approach) side of the staging tile; perp == off-axis
        distance. Single geometry authority: _potential's approach cone / funnel and the
        observation's corridor scalars must agree, or the policy sees a different corridor
        than the one the reward pays on.
        """
        close_dir = self._reward_target_direction()
        if close_dir is None:
            return None, None
        tx, ty, _ = self._reward_target_position()
        ex, ey = self.direction_vectors[close_dir]           # entry direction (head docks heading this)
        ox = self.current_position[0] - tx
        oy = self.current_position[1] - ty
        return -(ox * ex + oy * ey), abs(ox * ey - oy * ex)

    def _route_progress(self):
        """Angular progress of the build head around the station center, in [0, 1].

        0 on the start-side bearing (the entry direction, where the head exits the station)
        rising monotonically to 1 on the opposite/approach bearing -- along BOTH the north and
        south detours. Pure function of current_position (and the fixed station geometry), so
        the w_route Phi term telescopes. 0.5 at the (unreachable) center degeneracy.
        """
        ex, ey = self.direction_vectors[self._STATION_ENTRY_DIR]
        cx = self.station_start_position[0] + ex * (self.station_length - 1) / 2.0
        cy = self.station_start_position[1] + ey * (self.station_length - 1) / 2.0
        vx = self.current_position[0] - cx
        vy = self.current_position[1] - cy
        n = float(np.hypot(vx, vy))
        if n < 1e-6:
            return 0.5
        return float(np.arccos(np.clip((vx * ex + vy * ey) / n, -1.0, 1.0)) / np.pi)

    def _return_potential(self, params):
        """Climb-gated descent-shaping component of Phi (PBRS).

        Returns 0 until a CHAIN climb reaches roundtrip_gain, and 0 at/above the summit threshold
        height (STATION_HEIGHT + roundtrip_gain) -- so the gate turning on at the summit injects NO
        Phi jump (it cannot re-pay the summit milestone) -- then rises linearly to w_return as the
        head returns to station height. Pure function of (current z, removal-safe _chain_max_gain),
        so it telescopes like the rest of Phi. Exposed as a method so it doubles as a loggable
        diagnostic (rewards/return_potential)."""
        if params.w_return <= 0.0 or self._chain_max_gain() < params.roundtrip_gain:
            return 0.0
        threshold_z = self.STATION_HEIGHT + params.roundtrip_gain
        progress = (threshold_z - self.current_position[2]) / params.roundtrip_gain
        return float(params.w_return * min(max(progress, 0.0), 1.0))

    def _quality_bonus(self, excitement, intensity, nausea, params):
        """Graduated, bounded, non-negative ride-quality bonus in [0, R_quality_max].

        Smooth Gaussian bands for excitement/intensity (peaks at exc_target/int_target)
        and a logistic for nausea (lower is better). Returns 0 when quality is disabled
        (R_quality_max == 0, phases 1-4) or the ride was not tested (all stats zero), so
        a completed ride is never punished (completion-first).
        """
        if params.R_quality_max <= 0:
            return 0.0
        # Unrated gate: RCT2 reports -1 (plugin: -0.01) until the test train has actually
        # run; nausea=0 on an UNRATED ride is a sentinel, not a calm ride -- scoring it paid
        # a ~+96 'nausea freebie' for rides that never demonstrably completed a circuit.
        if excitement <= 0 and intensity <= 0 and nausea <= 0:
            return 0.0
        # Ramp + band (the Phase-5 plateau fix): the pure Gaussian band paid ZERO for any
        # excitement short of ~target-2sigma, so an agent at exc ~1.5 had no gradient and two
        # independent runs converged onto the identical minimal-loop optimum. The ramp half
        # pays every increment toward the target; the band half still peaks AT it (and decays
        # on overshoot -- an 11-intensity coaster keeps the ramp but loses the band).
        def ramp_band(stat, target, sigma):
            ramp = min(max(float(stat), 0.0) / target, 1.0)
            band = float(np.exp(-((stat - target) ** 2) / (2.0 * sigma ** 2)))
            return 0.5 * ramp + 0.5 * band
        q_e = ramp_band(excitement, params.exc_target, params.exc_sigma)
        q_i = ramp_band(intensity, params.int_target, params.int_sigma)
        q_n = 0.5 * (1.0 + np.tanh(0.5 * (params.nausea_max - nausea) / params.nausea_tau))
        return float(params.R_quality_max
                     * (params.q_w_exc * q_e + params.q_w_int * q_i + params.q_w_nausea * q_n))

    def _total_drop_z(self):
        """Total z dropped over DESCENT pieces in the removal-safe history (climbs ignored).
        The 'real drop' measure for the P3/P4 scale components and gates."""
        hist = getattr(self.track_builder, "history", None) or []
        return float(sum(max(0.0, h["position"][2] - h["next_position"][2]) for h in hist))

    def _steep_drop_z(self):
        """Total z dropped over STEEP descent pieces (the 60-degree family: 60-down and the
        25<->60 transitions, actions 8/27/28) in the removal-safe history. Feeds the graded
        steepness credit; 25-degree descents count toward _total_drop_z only."""
        hist = getattr(self.track_builder, "history", None) or []
        return float(sum(max(0.0, h["position"][2] - h["next_position"][2])
                         for h in hist if h.get("action") in (8, 27, 28)))

    def _chain_max_gain(self):
        """Highest elevation (z above station) reached via CHAIN-LIFT pieces (actions 9/10) in
        the removal-safe history; 0.0 if none. Single source of truth for every chain-elevation
        incentive — the discovery potential, the round-trip milestone, and the summit milestone
        all key on this so a chain lift is strictly rewarded over an identical-geometry plain
        climb (which shares the same track_type but lacks the chain flag the Phase-2 gate counts)."""
        hist = getattr(self.track_builder, "history", None) or []
        gains = [h["next_position"][2] - self.STATION_HEIGHT
                 for h in hist if h.get("action") in (9, 10)]
        return max(max(gains, default=0.0), 0.0)

    def _hill_quality(self, params):
        """Lift-hill / drop quality of the current track in [0, 1], from the removal-safe
        history (NOT self.chain_lift_count, which is capped/bookkept and can desync from the
        gate). Used by both the structural bonus and the completion gate."""
        hist = getattr(self.track_builder, "history", None) or []
        chain_count = sum(1 for h in hist if h.get("action") in (9, 10))
        chain_q = min(chain_count / params.struct_chain_target, 1.0)
        # Elevation factor: chain PIECES alone are farmable (three scattered 1-z stubs are
        # not a lift hill); scale by chain-banked height against the stage's climb bar.
        if params.roundtrip_gain > 0:
            chain_q *= min(self._chain_max_gain() / params.roundtrip_gain, 1.0)
        # Graded scale components (all ramps toward per-phase targets -- no cliffs):
        drop_q = (min(self._total_drop_z() / params.struct_drop_target, 1.0)
                  if params.struct_drop_target > 0 else 0.0)
        height_q = (min(self._chain_max_gain() / params.struct_height_target, 1.0)
                    if params.struct_height_target > 0 else 0.0)
        length_q = (min(self.track_length / params.struct_length_target, 1.0)
                    if params.struct_length_target > 0 else 0.0)
        steep_q = (min(self._steep_drop_z() / params.struct_steep_target, 1.0)
                   if params.struct_steep_target > 0 else 0.0)
        return min(params.struct_w_chain * chain_q
                   + params.struct_w_drop * drop_q
                   + params.struct_w_height * height_q
                   + params.struct_w_length * length_q
                   + params.struct_w_steep * steep_q, 1.0)

    def _structural_bonus(self, params):
        """Completion-conditioned bonus for building the lift-hill / drop structure each
        intermediate phase gates on, in [0, R_struct_max]. Only the caller's completion
        guard makes it a real bonus, so it cannot be place/remove-farmed. 0 when disabled."""
        if params.R_struct_max <= 0:
            return 0.0
        return float(params.R_struct_max * self._hill_quality(params))

    def _qualifies(self, params):
        """Whether the current track meets the phase's qualified-gate targets, from
        env-native state (mirrors ImprovedPhasedCurriculumWrapper._is_qualified for P3/P4:
        same struct targets, energy proxy, steep-drop and verified-test legs). Structural
        targets <= 0 are vacuous. Callers guard on completion (R_qualify is terminal-only)."""
        if params.struct_height_target > 0 and self._chain_max_gain() < params.struct_height_target:
            return False
        if params.struct_drop_target > 0 and self._total_drop_z() < params.struct_drop_target:
            return False
        if params.struct_length_target > 0 and self.track_length < params.struct_length_target:
            return False
        if params.qualify_requires_energy and self._calculate_energy_margin() < 0.0:
            return False
        if params.qualify_requires_steep_drop:
            hist = getattr(self.track_builder, "history", None) or []
            if not any(h.get("action") in (8, 27, 28) for h in hist):
                return False
        if params.qualify_requires_test and not getattr(self, "_last_test_ok", False):
            return False
        return True

    def _calculate_distance_to_start(self):
        point_a = np.array(self.current_position)
        point_b = np.array(self._reward_target_position())
        distance = float(np.linalg.norm(point_a - point_b))
        return np.array([distance], dtype=np.float32)

    def _revert_chain_lift(self, removed_entry):
        """Undo chain-lift bookkeeping when a piece is removed.

        Mirrors the increment in step()'s placement path (which records a chain lift's
        endpoint in chain_lift_positions and bumps chain_lift_count). Without this,
        counts/positions would drift after remove/auto-backtrack. Guarded so only
        previously-recorded chain cells decrement.
        """
        if not removed_entry:
            return
        if removed_entry.get("action") not in (9, 10):
            return
        pos_key = tuple(removed_entry["next_position"])
        if pos_key in self.chain_lift_positions:
            self.chain_lift_positions.discard(pos_key)
            self.chain_lift_count = max(0, self.chain_lift_count - 1)

    def _is_trunkated(self):
        # Possibly-desynced client/server track state (request timeout, all-invalid mask):
        # every further transition would be corrupt -- sacrifice the episode.
        if self._connection_suspect:
            return True

        # Check for extreme distance termination
        current_distance = self._calculate_distance_to_start()[0]
        too_far = current_distance > 100  # Terminate if more than 100 units away

        # Scaffolded-episode budget (see _apply_warm_start): a failed dock attempt must
        # truncate fast instead of wandering the full phase budget.
        if self._warm_track_cap is not None and self.track_length >= self._warm_track_cap:
            return True
        if self._warm_step_cap is not None and self.steps >= self._warm_step_cap:
            return True

        # No need to check for stuck patterns - symmetric removal prevents exploitation

        return (self.steps >= self.max_steps or
                self.track_length >= self.max_track_length or
                too_far)

    def _ego_rotate(self, world_dx, world_dy):
        """Rotate a world XY vector into the egocentric (forward-aligned) frame.

        Returns (ego_right, ego_forward): components along the agent's right and
        forward axes for the current heading. Makes spatial features rotation-
        invariant - the same local geometry maps to the same input at any heading.
        """
        fwd = self.direction_vectors[self.current_direction]
        right = self.direction_vectors[(self.current_direction + 1) % 4]
        ego_right = world_dx * right[0] + world_dy * right[1]
        ego_forward = world_dx * fwd[0] + world_dy * fwd[1]
        return ego_right, ego_forward

    def _build_history_buffer(self):
        """Structured build-history memory: the last SEQ_LEN placed pieces.

        Returns (tokens, feats, mask), oldest->newest, RIGHT-padded. Sourced from
        track_builder.history so it stays correct through remove/auto-backtrack.
          tokens: action_index + 1 (0 = PAD; +1 so action 0 collides with no padding_idx)
          feats : per-piece geometry in the egocentric frame, clipped to [-1, 1]:
                  [dx, dy, dz, cum_dz_from_station, sin_h, cos_h,
                   turn_left, turn_straight, turn_right, is_chain_lift, is_complete]
          mask  : 1 for real pieces, 0 for padding
        """
        tokens = np.zeros(SEQ_LEN, dtype=np.int32)
        feats = np.zeros((SEQ_LEN, HIST_FEAT_DIM), dtype=np.float32)
        mask = np.zeros(SEQ_LEN, dtype=np.float32)

        window = self.track_builder.history[-SEQ_LEN:]
        for i, entry in enumerate(window):
            action = entry["action"]
            pos, nxt = entry["position"], entry["next_position"]
            tokens[i] = action + 1
            mask[i] = 1.0

            er, ef = self._ego_rotate(nxt[0] - pos[0], nxt[1] - pos[1])
            dz = nxt[2] - pos[2]
            cum_dz = (nxt[2] - self.STATION_HEIGHT) / H_SCALE

            rel_dir = (entry["next_direction"] - self.current_direction) % 4
            ang = rel_dir * (np.pi / 2.0)

            turn = (entry["next_direction"] - entry["direction"]) % 4
            turn_left = 1.0 if turn == 3 else 0.0
            turn_straight = 1.0 if turn == 0 else 0.0
            turn_right = 1.0 if turn == 1 else 0.0

            row = [
                er / SCALE, ef / SCALE, dz / H_SCALE, cum_dz,
                np.sin(ang), np.cos(ang),
                turn_left, turn_straight, turn_right,
                1.0 if action in (9, 10) else 0.0,
                1.0 if entry.get("is_complete") else 0.0,
            ]
            feats[i] = np.clip(np.asarray(row, dtype=np.float32), -1.0, 1.0)

        return tokens, feats, mask

    def _build_local_map(self):
        """Egocentric, forward-aligned 2.5D map centred on the build head.

        Channels (C, H, W), all in [-1, 1]:
          0 occupancy (1 where any placed-track or station tile is)
          1 signed top-of-column height clip((z_top - STATION_HEIGHT)/H_SCALE, -1, 1)
          2 goal + station marker
          3 chain-lift / path trail (endpoints of history actions 9/10)
        Built from track_builder.history + an explicit station-tile stamp (station
        pieces are NOT in history) + the goal tile. Tiles outside the window are
        dropped; goal bearing is still carried by goal_disp/goal_direction3.
        """
        C, H, W = MAP_SHAPE
        grid = np.zeros((C, H, W), dtype=np.float32)
        top_z = np.full((H, W), -np.inf, dtype=np.float32)
        center = H // 2
        head = self.current_position

        def stamp(tile, marker=False, chain=False):
            er, ef = self._ego_rotate(tile[0] - head[0], tile[1] - head[1])
            row = center - int(round(ef))
            col = center + int(round(er))
            if not (0 <= row < H and 0 <= col < W):
                return
            grid[0, row, col] = 1.0
            if tile[2] > top_z[row, col]:
                top_z[row, col] = tile[2]
                grid[1, row, col] = np.clip(
                    (tile[2] - self.STATION_HEIGHT) / H_SCALE, -1.0, 1.0)
            if marker:
                grid[2, row, col] = 1.0
            if chain:
                grid[3, row, col] = 1.0

        # Station body (approximate: station_length tiles along direction 0 from start).
        sdir = self.direction_vectors[0]
        sx, sy, sz = self.station_start_position
        for i in range(self.station_length):
            stamp((sx + i * sdir[0], sy + i * sdir[1], sz), marker=True)

        # Placed track pieces (endpoint stamping; multi-tile turns approximate).
        for entry in self.track_builder.history:
            is_chain = entry["action"] in (9, 10)
            stamp(entry["position"], chain=is_chain)
            stamp(entry["next_position"], chain=is_chain)

        # Goal / closing-target tile.
        stamp(self._reward_target_position(), marker=True)

        return grid

    def _get_observation(self):
        current_distance = float(self._calculate_distance_to_start()[0])

        # Goal displacement / direction in the egocentric frame.
        target = self._reward_target_position()
        gdx = target[0] - self.current_position[0]
        gdy = target[1] - self.current_position[1]
        gdz = target[2] - self.current_position[2]
        ego_r, ego_f = self._ego_rotate(gdx, gdy)
        goal_disp = np.clip(
            np.array([ego_r / SCALE, ego_f / SCALE, gdz / H_SCALE], dtype=np.float32),
            -1.0, 1.0)
        norm3 = float(np.linalg.norm([ego_r, ego_f, gdz]))
        if norm3 > 0:
            goal_direction3 = np.clip(
                np.array([ego_r, ego_f, gdz], dtype=np.float32) / norm3, -1.0, 1.0)
        else:
            goal_direction3 = np.zeros(3, dtype=np.float32)

        # Angle between current heading and goal (2D), radians [0, pi].
        goal_vec2 = np.array([gdx, gdy], dtype=np.float32)
        n2 = float(np.linalg.norm(goal_vec2))
        if n2 > 0:
            cur = np.array(self.direction_vectors[self.current_direction], dtype=np.float32)
            angle_to_goal = float(np.arccos(np.clip(np.dot(cur, goal_vec2 / n2), -1.0, 1.0)))
        else:
            angle_to_goal = 0.0

        # Distance trend over the position-history window.
        if len(self.position_history) >= 2:
            old_dist = float(np.linalg.norm(
                np.array(self.position_history[0]) - np.array(target)))
            distance_trend = old_dist - current_distance
        else:
            distance_trend = 0.0

        energy_margin = self._calculate_energy_margin()
        track_len_frac = self.track_length / max(1, self.max_track_length)

        # Corridor scalars: the exact coordinates Phi's approach cone / funnel / heading /
        # route terms pay on, so the policy sees the reward manifold instead of having to
        # infer it from map+disp. Neutral zeros while the entry axis is uncalibrated (the
        # route progress is station-geometry-only and always defined).
        along, perp = self._corridor_coords()
        close_dir = self._reward_target_direction()
        if close_dir is not None:
            cur = self.direction_vectors[self.current_direction]
            tgt = self.direction_vectors[close_dir]
            heading_cos = float(cur[0] * tgt[0] + cur[1] * tgt[1])
        else:
            heading_cos = 0.0

        scalars = np.clip(np.array([
            current_distance / 100.0,
            angle_to_goal / np.pi,
            distance_trend / 100.0,
            track_len_frac,
            self.last_ride_excitement / 15.0,
            self.last_ride_intensity / 15.0,
            self.last_ride_nausea / 15.0,
            energy_margin / 100.0,
            0.0 if along is None else along / 16.0,
            0.0 if perp is None else perp / 8.0,
            heading_cos,
            self._route_progress(),
        ], dtype=np.float32), -10.0, 10.0)

        tokens, feats, mask = self._build_history_buffer()
        # +1 shift to match the token convention; 0 = no agent piece placed yet.
        last_piece = (self.last_piece_type + 1) if self.track_builder.history else 0

        return {
            'local_map': self._build_local_map(),
            'build_history_tokens': tokens,
            'build_history_feats': feats,
            'build_history_mask': mask,
            'goal_disp': goal_disp,
            'goal_direction3': goal_direction3,
            'scalars': scalars,
            'current_direction': int(self.current_direction),
            'last_piece_type': int(last_piece),
        }

    def evaluate_ride(self):
        resp = self.api_controller.get_ride_stats()  # Changed from get_ride_ratings
        if resp.get("success"):
            ratings = resp["payload"]
            return {
                'excitement': ratings.get('excitement', 0),
                'intensity': ratings.get('intensity', 0),
                'nausea': ratings.get('nausea', 0)
            }
        else:
            if self.verbose >= 1:
                print("Failed to get ride statistics")
            return {
                'excitement': 0,
                'intensity': 0,
                'nausea': 0
            }

    def _poll_for_ride_stats(self, max_wait=5, poll_interval=0.5):
        """
        Poll for ride statistics with smart detection of completion.

        Args:
            max_wait: Maximum wait time in seconds (default: 5, reduced from 10)
            poll_interval: Time between polls in seconds (default: 0.5, reduced from 1)
        """
        start_time = time.time()
        polls = 0

        while time.time() - start_time < max_wait:
            polls += 1
            result = self.api_controller.get_ride_stats()

            if result.get("success"):
                stats = result.get("payload", {})
                # Check if we have actual stats
                has_ratings = (
                    "excitement" in stats and
                    "intensity" in stats and
                    "nausea" in stats
                )
                # POSITIVE ratings only: RCT2 reports -1 (plugin: -0.01) while the ride is
                # still unrated -- '!= 0' accepted that sentinel on the very first poll and
                # returned junk stats for rides whose test train had not run yet.
                has_nonzero = (
                    stats.get("excitement", 0) > 0 or
                    stats.get("intensity", 0) > 0 or
                    stats.get("nausea", 0) > 0
                )

                if has_ratings and has_nonzero:
                    elapsed = time.time() - start_time
                    if self.verbose >= 2:
                        print(f"✅ Ride test completed after {elapsed:.1f}s ({polls} polls)")
                    return {
                        'excitement': stats.get('excitement', 0),
                        'intensity': stats.get('intensity', 0),
                        'nausea': stats.get('nausea', 0)
                    }

            time.sleep(poll_interval)

        # Timeout - return zeros
        if self.verbose >= 1:
            print(f"⚠️ Ride test timeout after {max_wait}s")
        return {'excitement': 0, 'intensity': 0, 'nausea': 0}
    
    def _apply_batched_reset(self, payload):
        """Apply a server-side resetEpisode payload (one round-trip): set the post-station head,
        preserve the station prefix in track_pieces (the legacy path appends one placeholder per
        station piece, and downstream index math such as the station-shifted
        track_pieces/height_history pairing relies on it), and seed the valid-piece cache so the
        first action needs no separate getValidNextPieces round-trip."""
        endpoint = payload["finalEndpoint"]
        self.current_position = [endpoint["x"], endpoint["y"], endpoint["z"]]
        self.current_direction = endpoint["direction"]
        self.track_pieces = [0] * self.station_length
        self.track_builder._cache_valid_from_payload(
            {"validNextPieces": payload.get("validNextPieces")}
        )

    def _legacy_reset(self):
        """Fallback reset for older plugins (no resetEpisode). Always starts from a clean slate --
        delete_all_rides() first -- so a partially-applied batched reset cannot leave the ride in a
        dirty state, then creates the ride and builds the station via individual API calls."""
        self.api_controller.delete_all_rides()
        ride_id = self.api_controller.create_ride()
        if ride_id is None:
            raise RuntimeError("Failed to create new ride")
        if not self._build_initial_station():
            raise RuntimeError("Failed to build initial station")

    def _build_initial_station(self):
        # Build station pieces
        current_x, current_y, current_z = self.station_start_position
        current_dir = 0
        
        if self.verbose >= 2:
            print(f"Building {self.station_length} station pieces starting at {self.station_start_position}")
        
        for i in range(self.station_length):
            # Station configuration:
            # Type 1 = EndStation, Type 2 = BeginStation, Type 3 = MiddleStation
            # Proper station should be: BeginStation -> MiddleStation(s) -> EndStation
            if i == 0:
                station_track_type = 2  # BeginStation
            elif i == self.station_length - 1:
                station_track_type = 1  # EndStation (corrected!)
            else:
                station_track_type = 3  # MiddleStation (corrected!)
            
            resp = self.api_controller.place_track_piece(
                current_x, current_y, current_z,
                current_dir, station_track_type
            )
            if not resp.get("success"):
                if self.verbose >= 1:
                    print(f"Failed to place station piece {i}: {resp.get('error')}")
                return False
            
            next_ep = resp["payload"]["nextEndpoint"]
            current_x = next_ep["x"]
            current_y = next_ep["y"]
            current_z = next_ep["z"]
            current_dir = next_ep["direction"]
            
            # Update track state
            self.track_pieces.append(0)  # Station pieces as action 0
            # Don't increment track_length for station pieces - they're not part of the actual track
        
        # Seed the valid-piece cache from the last station placement so the agent's
        # first action needs no separate getValidNextPieces round-trip.
        self.track_builder._cache_valid_from_payload(resp.get("payload", {}))

        # Update current position to end of station
        self.current_position = [current_x, current_y, current_z]
        self.current_direction = current_dir
        
        if self.verbose >= 2:
            print(f"Station built. Track starts at {self.station_start_position}, current position: {self.current_position}")
            print(f"Goal: Place track at {self.goal_position} to connect to BeginStation at {self.station_start_position}")
        
        return True

    # Energy estimation constants
    CHAIN_LIFT_ENERGY = 50   # Energy added per chain lift piece
    GRAVITY_GAIN = 3         # Energy gained per unit of descent
    FRICTION_BASE = 2        # Energy lost per piece
    UPHILL_COST = 5          # Extra cost climbing without chain
    TURN_FRICTION = 1        # Extra friction for turns
    STATION_HEIGHT = 14      # z-coordinate of station
    _STATION_ENTRY_DIR = 0   # cardinal dir the train enters BeginStation with (dir 0 = West/(-1,0); station built startDir=0)

    # Scaffolded-episode budget: extra pieces beyond the planned suffix k, and the step
    # multiplier that bounds no-progress churn (failures, place/remove).
    WARM_TRACK_SLACK = 4
    WARM_STEP_FACTOR = 4
    # Harvest policy: meander completions beyond this are junk scaffolds. Raised 24 -> 40
    # for the P3/P4 scale-up (their completions are prime big-loop material; the library's
    # per-class caps keep growth bounded and the 'big' class keeps minis from crowding them).
    HARVEST_MAX_LEN = 40

    # How many recent positions to retain for the observation's distance-trend signal.
    POSITION_HISTORY_MAXLEN = 8

    def _calculate_estimated_energy(self):
        """
        Approximate energy model for roller coaster physics.

        Energy sources:
        - Chain lifts add fixed energy (simulating motor work)
        - Descending converts potential energy to kinetic

        Energy costs:
        - Friction loss per piece
        - Climbing without chain lift costs energy

        Computed from track_builder.history (agent pieces only, each piece's endpoint
        height). This is the authoritative source: it excludes the flat station prefix
        and is removal-safe (history.pop keeps it in sync), unlike the old
        track_pieces/height_history pair which was index-shifted by the station and
        produced offset/noisy energy.
        """
        history = getattr(self.track_builder, "history", None)
        if not history:
            return 0.0

        energy = 0.0
        last_height = self.STATION_HEIGHT

        for entry in history:
            piece = entry["action"]
            # Chain lift adds energy
            if piece in [9, 10]:  # Chain lift pieces
                energy += self.CHAIN_LIFT_ENERGY

            current_height = entry["next_position"][2]
            height_delta = current_height - last_height

            # Height changes affect energy
            if height_delta > 0:
                # Going up costs energy (unless chain lift)
                if piece not in [9, 10]:
                    energy -= height_delta * self.UPHILL_COST
            else:
                # Going down converts potential to kinetic (adds energy)
                energy += abs(height_delta) * self.GRAVITY_GAIN

            last_height = current_height

            # Friction losses
            energy -= self.FRICTION_BASE

            # Extra friction for turns
            if piece in [1, 2, 3, 4, 21, 22, 23, 24, 29, 30]:  # Turn pieces
                energy -= self.TURN_FRICTION

        return max(0.0, energy)

    def _calculate_energy_margin(self):
        """
        Calculate if current energy is sufficient to return to station.
        Positive = have excess energy (track is viable)
        Negative = likely to stall (not enough energy to return)
        """
        current_energy = self._calculate_estimated_energy()
        current_height = self.current_position[2]

        # Distance to station
        distance = self._calculate_distance_to_start()[0]

        # Height deficit (if we're below station level, need energy to climb)
        height_deficit = max(0, self.STATION_HEIGHT - current_height)

        # Estimated energy needed to get back
        # Account for distance (friction) and height (climbing)
        energy_needed = distance * 0.5 + height_deficit * self.UPHILL_COST

        return current_energy - energy_needed

    def close(self):
        if self.api_controller:
            self.api_controller.disconnect()

    def render(self):
        if self.render_mode == "human":
            pass

