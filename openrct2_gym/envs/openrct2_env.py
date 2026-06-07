import gymnasium as gym
import numpy as np
import time
from collections import deque
from .api_track_builder import APITrackBuilder
from .api_controller import APIController
from .obs_config import (
    make_observation_space, SEQ_LEN, HIST_FEAT_DIM, MAP_SHAPE, SCALE, H_SCALE,
)

class OpenRCT2Env(gym.Env):
    def __init__(self, render_mode=None, host="localhost", port=8080, verbose=1):
        super(OpenRCT2Env, self).__init__()
        self.render_mode = render_mode
        self.verbose = verbose  # 0=silent, 1=important only, 2=detailed
        self.api_controller = APIController(host, port, verbose)
        self.track_builder = APITrackBuilder(self.api_controller)
        self.skip_ride_testing = False  # Can be set by wrappers to skip entrance/exit and testing
        
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
        self.piece_rewards = []  # Track reward earned for each placed piece (for symmetric removal)
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

        self.direction_vectors = [
            (0, 1),   # North (0)
            (1, 0),   # East (1)
            (0, -1),  # South (2)
            (-1, 0)   # West (3)
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
        else:
            self.consecutive_failures = 0
            # Increment steps since collision on successful actions
            if success:
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
            else:
                self.track_length += 1
                self.track_pieces.append(action)
                self.height_history.append(new_position[2])  # Track height at this piece

            self.last_piece_type = action
            self.current_position = new_position
            self.current_direction = new_direction
            # Track position history for trend analysis
            self.position_history.append(self.current_position.copy())

        self.last_action = action

        # Check for loop completion BEFORE calculating reward
        self.loop_completed = self._last_placement_complete
        if self.loop_completed:
            if self.verbose >= 1:
                print(f"Loop has been completed, great success!")
        terminated = self.loop_completed

        # Now calculate reward (with correct loop_completed status)
        observation = self._get_observation()
        reward = self._calculate_reward(success, action)

        # Track reward for this piece (for symmetric removal)
        # Only track for successful placements (not removes)
        if success and action != 31:
            self.piece_rewards.append(reward)
        elif success and action == 31 and self.piece_rewards:
            # For remove action, the reward was already calculated as negative of last piece
            # No need to track it
            pass
        
        # Track metrics
        current_distance = self._calculate_distance_to_start()[0]
        self.min_distance_reached = min(self.min_distance_reached, current_distance)
        self.max_height_reached = max(self.max_height_reached, self.current_position[2])
        self.unique_positions.add(tuple(self.current_position))
        self.episode_rewards.append(reward)
        
        # Track phase-based rewards
        if self.track_length <= 25:
            self.current_phase = 'building'
            self.phase_rewards['building'] += reward
        elif self.track_length <= 40:
            self.current_phase = 'transition'
            self.phase_rewards['transition'] += reward
        else:
            self.current_phase = 'return'
            self.phase_rewards['return'] += reward
        
        # Check if episode was truncated
        truncated = self._is_trunkated()
        self.steps += 1
        
        # Create info dict with additional debugging information
        info = {
            'auto_backtracked': auto_backtracked,
            'original_action': original_action if auto_backtracked else action,
            'collision_count': self.collision_count,
            'consecutive_failures': self.consecutive_failures,
            'loop_completed': self.loop_completed
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

                    # Smart polling for ride stats (max 10 seconds for training efficiency)
                    ride_rating = self._poll_for_ride_stats(max_wait=10)
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
        elif truncated:
            # Partial credit for getting close even if not completed
            final_distance = self._calculate_distance_to_start()[0]
            if final_distance < 10:
                reward += (10 - final_distance) * 2
            elif final_distance < 20:
                reward += (20 - final_distance) * 0.5

        # Store episode-level metrics before the environment resets
        if terminated or truncated:
            info['episode_metrics'] = {
                'min_distance': self.min_distance_reached,
                'chain_lift_count': self.chain_lift_count,
                'track_length': self.track_length,
                'phase_rewards': dict(self.phase_rewards),
                'collision_count': self.collision_count,
                'loop_completed': self.loop_completed
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
        if not mask.any() and len(self.track_builder.history) > 0:
            mask[31] = True
            
        return mask
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all state variables FIRST
        # API hardcodes the station start at [61, 66, 14] - we must match this
        self.station_start_position = [61, 66, 14]  # Where the first station piece is (matches API)
        self.current_position = self.station_start_position.copy()
        # Set goal one tile east of station start to guide proper connection
        # Agent needs to place a piece at this position to connect to BeginStation
        self.goal_position = [
            self.station_start_position[0] + 1,     # X + 1 (one tile east)
            self.station_start_position[1],      # Same Y 
            self.station_start_position[2]       # Same Z
        ]
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
        self.previous_distance = None
        self.position_history = deque(maxlen=self.POSITION_HISTORY_MAXLEN)
        self.chain_lift_positions = set()
        self.piece_rewards = []  # Reset reward tracking
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

        # Delete all rides to ensure clean state
        # This is more reliable than tracking individual rides across resets
        self.api_controller.delete_all_rides()

        # Create new ride
        ride_id = self.api_controller.create_ride()
        if ride_id is None:
            raise RuntimeError("Failed to create new ride")
        
        # Build initial station (this will update current_position to end of station)
        station_built = self._build_initial_station()
        if not station_built:
            raise RuntimeError("Failed to build initial station")

        observation = self._get_observation()
        info = {}
        return observation, info

    def _calculate_reward(self, success, action):
        # Special case: Remove action gets negative of last piece's reward
        if success and action == 31:  # Remove piece
            if self.piece_rewards:
                # Return the exact negative of what was earned when placing this piece
                removed_reward = self.piece_rewards.pop()
                if self.verbose >= 2:
                    print(f"Remove action: reversing reward of {removed_reward:.2f}")
                return -removed_reward
            else:
                # No pieces to remove, small penalty
                return -1

        # Normal reward calculation for all other actions
        reward = 0
        current_distance = self._calculate_distance_to_start()[0]

        if success:
            # Base reward for successful action
            reward += 1

            # Reward for placing chain lifts early (actions 9 and 10), but at a
            # reduced incentive so the agent still has pieces left to return to
            # the station. Limit the reward to the first 15 pieces to avoid
            # spending the entire budget on hills.
            if self.track_length < 15 and action in [9, 10]:
                position_key = tuple(self.current_position)
                # Reward if this is a NEW position for chain lift
                if position_key not in self.chain_lift_positions and self.chain_lift_count < self.max_chain_lifts:
                    reward += 5  # bonus for chain lift
                    self.chain_lift_count += 1
                    self.chain_lift_positions.add(position_key)
                # No penalty for rebuilding - the symmetry handles it

            # Big reward for completing the loop
            if self.loop_completed:
                reward += 500
                # Length bonus for longer completed tracks
                if self.track_length > 50:
                    reward += (self.track_length - 50) * 0.5
                # Chain lift usage bonus
                if self.chain_lift_count > 0:
                    reward += 10
            
            # Simplified distance-based rewards (no internal phases)
            if self.previous_distance is not None:
                distance_delta = self.previous_distance - current_distance

                # Consistent distance reward throughout
                if distance_delta > 0:  # Moving closer
                    reward += distance_delta * 1.0  # Moderate reward
                else:  # Moving away
                    # Small penalty that increases with track length
                    penalty_factor = min(0.5 + (self.track_length / 100), 1.5)
                    reward += distance_delta * penalty_factor * 0.2

                # Proximity bonuses when getting close
                if current_distance < 30:
                    reward += (30 - current_distance) * 0.1
                if current_distance < 15:
                    reward += (15 - current_distance) * 0.2
                if current_distance < 5:
                    reward += (5 - current_distance) * 0.5

            # Height variation reward (for interesting coasters)
            if action in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]:  # Slope pieces
                # Reward height changes for variety
                if not hasattr(self, 'last_height'):
                    self.last_height = self.current_position[2]
                height_change = abs(self.current_position[2] - self.last_height)
                if height_change > 0:
                    reward += min(height_change * 0.3, 2.0)  # Reward hills and drops
                self.last_height = self.current_position[2]

            # Track variety bonus
            if not hasattr(self, 'used_piece_types'):
                self.used_piece_types = set()
            if action not in self.used_piece_types and action != 31:
                reward += 1.0  # Bonus for using new piece types
                self.used_piece_types.add(action)
            
            # Track length milestone rewards
            if self.track_length == 30:
                reward += 5
            elif self.track_length == 50:
                reward += 10
            elif self.track_length == 75:
                reward += 15
            elif self.track_length == 100:
                reward += 20
            
            # Small continuous reward for building longer tracks
            if self.track_length > 30:
                reward += 0.1

            # Small reward for forward progress
            if action != 31:
                reward += 0.2  # Bonus for building
            
            # Reward shaping when close to goal (Solution 4)
            if current_distance < 10:
                # Strong incentive to use flat pieces when close to station
                # Use the actual action taken (not last_action which is from previous step)
                if action in [0, 13, 14]:  # Flat or transitions to flat
                    reward += 5
                    if self.verbose >= 2:
                        print(f"Good choice near goal: flat piece {action}, bonus +5")
                elif action != 31:  # Not remove action
                    reward -= 2
                    if self.verbose >= 2:
                        print(f"Poor choice near goal: non-flat piece {action}, penalty -2")

            # Height penalty only for excessive height
            if self.current_position[2] > 30:
                reward -= (self.current_position[2] - 30) * 0.1  # Progressive penalty for going too high
                
        else:
            # Reduced penalty for failed placement (exploration)
            reward -= 0.2  # Reduced from -0.5
            
        # Update previous distance for next step
        self.previous_distance = current_distance
        
        if self.verbose >= 2:
            print("Reward was: %.2f (dist: %.1f, track: %d)" % (reward, current_distance, self.track_length))
        return reward

    def _calculate_distance_to_start(self):
        point_a = np.array(self.current_position)
        point_b = np.array(self.goal_position)
        distance = float(np.linalg.norm(point_a - point_b))
        return np.array([distance], dtype=np.float32)

    def _revert_chain_lift(self, removed_entry):
        """Undo chain-lift bookkeeping when a piece is removed.

        Mirrors the increment in _calculate_reward (which records an early chain lift's
        endpoint position in chain_lift_positions and bumps chain_lift_count). The old
        code never reverted this on removal, so counts/positions drifted after
        remove/auto-backtrack. Guarded so only previously-recorded chain cells decrement.
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
        # Check for extreme distance termination
        current_distance = self._calculate_distance_to_start()[0]
        too_far = current_distance > 100  # Terminate if more than 100 units away

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

        # Goal tile.
        stamp(self.goal_position, marker=True)

        return grid

    def _get_observation(self):
        current_distance = float(self._calculate_distance_to_start()[0])

        # Goal displacement / direction in the egocentric frame.
        gdx = self.goal_position[0] - self.current_position[0]
        gdy = self.goal_position[1] - self.current_position[1]
        gdz = self.goal_position[2] - self.current_position[2]
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
                np.array(self.position_history[0]) - np.array(self.goal_position)))
            distance_trend = old_dist - current_distance
        else:
            distance_trend = 0.0

        energy_margin = self._calculate_energy_margin()
        track_len_frac = self.track_length / max(1, self.max_track_length)

        scalars = np.clip(np.array([
            current_distance / 100.0,
            angle_to_goal / np.pi,
            distance_trend / 100.0,
            track_len_frac,
            self.last_ride_excitement / 15.0,
            self.last_ride_intensity / 15.0,
            self.last_ride_nausea / 15.0,
            energy_margin / 100.0,
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
                has_nonzero = (
                    stats.get("excitement", 0) != 0 or
                    stats.get("intensity", 0) != 0 or
                    stats.get("nausea", 0) != 0
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

    # How many recent positions to retain. Must be >= 6 so _detect_turnaround()
    # (which requires 6 samples) can ever fire; with the old maxlen=5 it never did.
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

    def _calculate_energy_reward(self):
        """Reward based on energy management."""
        energy_margin = self._calculate_energy_margin()

        if energy_margin > 50:
            return 3.0   # Healthy energy reserves
        elif energy_margin > 20:
            return 1.0   # Adequate energy
        elif energy_margin > 0:
            return 0.5   # Marginal but viable
        elif energy_margin > -20:
            return -2.0  # Warning: low energy
        else:
            return -5.0  # Critical: likely to stall

    def _detect_lift_hill_pattern(self):
        """
        Detect if a proper lift hill pattern was built.
        Pattern: [chain pieces] -> [transition/drop]
        Returns a score 0-1 indicating pattern quality.
        """
        if len(self.track_pieces) < 5:
            return 0.0

        # Look for chain lift sequences
        chain_sequences = []
        current_sequence = []

        for i, piece in enumerate(self.track_pieces):
            if piece in [9, 10]:  # Chain lift pieces
                current_sequence.append(i)
            elif current_sequence:
                if len(current_sequence) >= 2:  # Minimum 2 chain pieces for a lift hill
                    chain_sequences.append(current_sequence.copy())
                current_sequence = []

        # Don't forget the last sequence if still building
        if len(current_sequence) >= 2:
            chain_sequences.append(current_sequence)

        if not chain_sequences:
            return 0.0

        # Check if chain sequence is followed by drop
        best_score = 0.0
        for seq in chain_sequences:
            end_idx = seq[-1]

            # Check next few pieces for drop pattern
            drop_score = 0.0
            for j in range(end_idx + 1, min(end_idx + 5, len(self.track_pieces))):
                piece = self.track_pieces[j]
                if piece in [13, 14]:  # Transitions to flat (could precede drop)
                    drop_score += 0.2
                elif piece in [6, 8, 12]:  # Drop pieces (down slopes)
                    drop_score += 0.4

            # Score based on chain length and drop quality
            chain_length_score = min(len(seq) / 4.0, 1.0)  # Max score at 4 pieces
            pattern_score = chain_length_score * 0.6 + min(drop_score, 0.4)
            best_score = max(best_score, pattern_score)

        return best_score

    def _detect_drop_pattern(self):
        """
        Detect well-executed drop sequences.
        Pattern: [transition down] -> [steep descent] -> [recovery/transition up]
        Returns a score 0-1.
        """
        if len(self.track_pieces) < 3:
            return 0.0

        score = 0.0
        # Look for descent patterns
        for i in range(len(self.track_pieces) - 2):
            # Start of drop: flat to down or steep transitions
            if self.track_pieces[i] in [12, 27]:  # Flat to down, down 25 to down 60
                has_descent = False
                has_recovery = False

                for j in range(i + 1, min(i + 5, len(self.track_pieces))):
                    if self.track_pieces[j] in [6, 8]:  # Down slopes
                        has_descent = True
                    if self.track_pieces[j] in [14, 28]:  # Recovery (down to flat)
                        has_recovery = True

                if has_descent and has_recovery:
                    score = max(score, 1.0)
                elif has_descent:
                    score = max(score, 0.5)

        return score

    def _detect_turnaround(self):
        """
        Detect successful turnaround patterns.
        Agent changed direction to head back toward station.
        Returns a score 0-1.
        """
        if len(self.position_history) < 6:
            return 0.0

        positions = list(self.position_history)

        # Calculate direction vectors (using 2D, ignoring height)
        if len(positions) >= 4:
            early_direction = np.array(positions[1][:2]) - np.array(positions[0][:2])
            recent_direction = np.array(positions[-1][:2]) - np.array(positions[-2][:2])

            # Normalize
            early_norm = np.linalg.norm(early_direction)
            recent_norm = np.linalg.norm(recent_direction)

            if early_norm > 0 and recent_norm > 0:
                early_dir = early_direction / early_norm
                recent_dir = recent_direction / recent_norm

                # Dot product: -1 = opposite directions (good turnaround)
                dot = np.dot(early_dir, recent_dir)

                if dot < -0.7:  # Heading back (opposite direction)
                    return 1.0
                elif dot < -0.3:
                    return 0.5
                elif dot < 0:
                    return 0.2

        return 0.0

    def _get_pattern_rewards(self):
        """Calculate combined pattern-based rewards."""
        reward = 0.0

        # Lift hill pattern reward (one-time bonus when detected)
        lift_hill_score = self._detect_lift_hill_pattern()
        if lift_hill_score >= 0.8:
            reward += 15.0  # Excellent lift hill
        elif lift_hill_score >= 0.5:
            reward += 8.0   # Good lift hill
        elif lift_hill_score >= 0.3:
            reward += 3.0   # Partial lift hill

        # Drop pattern reward
        drop_score = self._detect_drop_pattern()
        if drop_score >= 0.8:
            reward += 10.0
        elif drop_score >= 0.5:
            reward += 5.0

        # Turnaround reward
        turnaround_score = self._detect_turnaround()
        if turnaround_score >= 0.8:
            reward += 15.0
        elif turnaround_score >= 0.5:
            reward += 7.0

        return reward

    def _calculate_approach_reward(self, action):
        """
        Soft approach guidance - rewards for good approach behavior.
        Uses bonuses only, no penalties (non-restrictive guidance).
        """
        distance = self._calculate_distance_to_start()[0]

        # Only apply in approach zone (within 20 units of station)
        if distance >= 20:
            return 0.0

        reward = 0.0
        current_height = self.current_position[2]
        height_delta = abs(self.STATION_HEIGHT - current_height)

        # Height alignment bonuses (crucial for connection)
        if height_delta == 0:
            reward += 5.0   # Perfect height alignment
        elif height_delta <= 4:
            reward += 2.0   # Close to correct height
        elif height_delta <= 8:
            reward += 1.0   # Getting there

        # Soft guidance toward flat/transition pieces when very close
        if distance < 10:
            if action in [0, 13, 14]:  # Flat or transitions to flat
                reward += 3.0  # Encouraged but not required
            elif action in [11, 12]:  # Other transitions
                reward += 1.0  # Also reasonable choices

        # Bonus for facing the right direction when close
        if distance < 8:
            # Goal is at [62, 66, 14], station starts at [61, 66, 14]
            # To connect, agent typically needs to be heading East (direction 1)
            # But this depends on approach angle, so we check goal direction
            goal_vector = np.array(self.goal_position[:2]) - np.array(self.current_position[:2])
            goal_dist = np.linalg.norm(goal_vector)
            if goal_dist > 0:
                goal_dir = goal_vector / goal_dist
                current_dir_vector = np.array(self.direction_vectors[self.current_direction])
                alignment = np.dot(current_dir_vector, goal_dir)
                if alignment > 0.7:  # Facing toward goal
                    reward += 3.0

        return reward

    def close(self):
        if self.api_controller:
            self.api_controller.disconnect()

    def render(self):
        if self.render_mode == "human":
            pass

