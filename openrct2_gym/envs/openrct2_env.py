import gymnasium as gym
import numpy as np
import time
from collections import deque
from .api_track_builder import APITrackBuilder
from .api_controller import APIController

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
        self.position_history = deque(maxlen=5)
        self.chain_lift_positions = set()  # Track positions where chain lifts were placed
        self.piece_rewards = []  # Track reward earned for each placed piece (for symmetric removal)
        self._invalid_completion_attempt = False  # Track invalid loop completion attempts
        self._invalid_completion_action = None  # Track which action failed validation

        # Additional metrics tracking
        self.min_distance_reached = float('inf')
        self.max_height_reached = 0
        self.unique_positions = set()
        self.phase_rewards = {'building': 0, 'transition': 0, 'return': 0}
        self.current_phase = 'building'
        self.episode_rewards = []
        
        self.direction_vectors = [
            (0, 1),   # North (0)
            (1, 0),   # East (1)
            (0, -1),  # South (2)
            (-1, 0)   # West (3)
        ]

        # Define observation space with navigation aids
        self.observation_space = gym.spaces.Dict({
            'track_pieces': gym.spaces.Box(low=0, high=31, shape=(self.max_track_length,), dtype=np.int32),
            'current_position': gym.spaces.Box(low=-20, high=1000, shape=(3,), dtype=np.int32),
            'current_direction': gym.spaces.Discrete(4),
            'distance_to_start': gym.spaces.Box(low=0, high=np.sqrt(2000**2 + 2000**2), shape=(1,), dtype=np.float32),
            'track_length': gym.spaces.Discrete(self.max_track_length + 1),
            'last_piece_type': gym.spaces.Discrete(32),
            'goal_direction': gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            'angle_to_goal': gym.spaces.Box(low=0, high=np.pi, shape=(1,), dtype=np.float32),
            'distance_trend': gym.spaces.Box(low=-100, high=100, shape=(1,), dtype=np.float32),
        })

    def step(self, action):
        # Store previous position for tracking
        prev_position = self.current_position.copy()
        original_action = action
        auto_backtracked = False
        
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
                    success, new_position, new_direction = self.track_builder.take_action(action, self.current_position, self.current_direction)
                    self.consecutive_failures = 0
        else:
            self.consecutive_failures = 0
            # Increment steps since collision on successful actions
            if success:
                self.steps_since_collision += 1
        
        # Check if the last placement completed the circuit
        if success and len(self.track_builder.history) > 0:
            last_entry = self.track_builder.history[-1]
            if "is_complete" in last_entry:
                # Only accept loop completion if the last piece is valid for station connection
                # Valid pieces: 0 (Flat), 13 (Up to Flat), 14 (Down to Flat)
                VALID_STATION_CONNECTIONS = {0, 13, 14}

                # Use original_action if auto-backtracked, otherwise use action
                action_to_validate = original_action if auto_backtracked else action

                if last_entry["is_complete"] and action_to_validate in VALID_STATION_CONNECTIONS:
                    self._last_placement_complete = True
                    if self.verbose >= 1:
                        print(f"✅ Valid loop completion with piece {action_to_validate}")
                elif last_entry["is_complete"]:
                    # Circuit claims to be complete but last piece is invalid
                    self._last_placement_complete = False
                    self._invalid_completion_attempt = True
                    self._invalid_completion_action = action_to_validate  # Store which action failed
                    if self.verbose >= 1:
                        print(f"❌ Invalid loop completion attempt with piece {action_to_validate} - must end with flat or transition to flat")
                else:
                    self._last_placement_complete = False
                    self._invalid_completion_attempt = False
                    self._invalid_completion_action = None
            else:
                self._last_placement_complete = False
                self._invalid_completion_attempt = False
                self._invalid_completion_action = None
        
        # Update state based on the action that was actually executed
        if success:
            if action == 31:  # Remove piece
                if self.track_pieces:
                    self.track_pieces.pop()
                    self.track_length -= 1
            else:
                self.track_length += 1
                self.track_pieces.append(action)

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
                
                # Set ride to testing mode and get ratings
                self.api_controller.set_ride_status(1)  # 1 = Testing
                time.sleep(5)  # Wait for testing to complete
                ride_rating = self.evaluate_ride()
                info['ride_rating'] = ride_rating
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
        self._last_placement_complete = False
        self._invalid_completion_attempt = False  # Reset invalid completion flag
        self._invalid_completion_action = None  # Reset invalid completion action
        self.collision_count = 0
        self.consecutive_failures = 0
        self.steps_since_collision = 0
        self.previous_distance = None
        self.position_history = deque(maxlen=5)
        self.chain_lift_positions = set()
        self.piece_rewards = []  # Reset reward tracking
        self.min_distance_reached = float('inf')
        self.max_height_reached = 0
        self.unique_positions = set()
        self.phase_rewards = {'building': 0, 'transition': 0, 'return': 0}
        self.current_phase = 'building'
        self.episode_rewards = []
        
        # Demolish ALL rides to ensure clean state
        # This is more reliable than just demolishing our tracked ride
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

        # Near-miss rewards for invalid loop completion attempts (Solution 2)
        if hasattr(self, '_invalid_completion_attempt') and self._invalid_completion_attempt:
            # Use the stored action that failed validation
            failed_action = getattr(self, '_invalid_completion_action', self.last_action)

            # Vary penalty based on how close the piece is to being valid
            if failed_action in [11, 12]:  # Transitions to slopes (closer to correct)
                reward -= 15
                if self.verbose >= 2:
                    print(f"Near-miss: transition piece {failed_action}, penalty -15")
            elif failed_action in [5, 6, 9, 10]:  # Mild slopes
                reward -= 20
                if self.verbose >= 2:
                    print(f"Near-miss: mild slope {failed_action}, penalty -20")
            elif failed_action in [7, 8]:  # Steep slopes
                reward -= 30
                if self.verbose >= 2:
                    print(f"Invalid completion: steep slope {failed_action}, penalty -30")
            elif failed_action in range(1, 5):  # Turns
                reward -= 40
                if self.verbose >= 2:
                    print(f"Invalid completion: turn {failed_action}, penalty -40")
            else:  # Banking and special pieces
                reward -= 40
                if self.verbose >= 2:
                    print(f"Invalid completion: special piece {failed_action}, penalty -40")

            self._invalid_completion_attempt = False  # Reset flag after applying penalty
            self._invalid_completion_action = None  # Reset action after applying penalty

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
            
            # Progressive distance-based rewards - STRENGTHENED
            if self.previous_distance is not None:
                distance_delta = self.previous_distance - current_distance
                
                # Phase-based distance rewards with EARLIER and STRONGER return phase
                if self.track_length <= 25:
                    # Building phase - allow exploration, reward height for chain lift
                    if self.current_position[2] > self.goal_position[2]:
                        reward += 0.2
                    # Small reward for building outward initially
                    if self.last_action not in [31]:
                        reward += 0.3
                elif self.track_length <= 40:
                    # Transition phase - start gentle guidance back
                    if distance_delta > 0:  # Moving closer
                        reward += distance_delta * 0.5
                else:
                    # Return phase - MUCH stronger pull back to station
                    if distance_delta > 0:  # Moving closer
                        reward += distance_delta * 2.0  # Strong reward for progress
                    # No penalty for moving away - let the agent explore
                    
                    # Distance checkpoint rewards
                        #if current_distance < 50 and self.previous_distance >= 50:
                        #    reward += 5  # Crossed 50-unit threshold
                        #if current_distance < 30 and self.previous_distance >= 30:
                        #    reward += 10  # Crossed 30-unit threshold
                        #if current_distance < 20 and self.previous_distance >= 20:
                        #    reward += 15  # Crossed 20-unit threshold
                        #if current_distance < 10 and self.previous_distance >= 10:
                        #    reward += 20  # Crossed 10-unit threshold
                    
                    # Escalating proximity bonuses - only from valid approach directions
                    # Goal is at X+1 from station, so we need X >= goal_X to approach from the correct side
                    if self.current_position[0] >= self.goal_position[0]:
                        # Agent is approaching from east/correct side - give proximity bonuses
                        if current_distance < 40:
                            reward += (40 - current_distance) * 0.3
                        if current_distance < 20:
                            reward += (20 - current_distance) * 0.5
                        if current_distance < 10:
                            reward += (10 - current_distance) * 1.0
                        if current_distance < 5:
                            reward += (5 - current_distance) * 2.0
            
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

            # Adjusted height penalty (only after transition phase)
            if self.track_length > 40 and self.current_position[2] > 25:
                reward -= 0.2
                
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

    def _is_trunkated(self):
        # Check for extreme distance termination
        current_distance = self._calculate_distance_to_start()[0]
        too_far = current_distance > 100  # Terminate if more than 100 units away

        # No need to check for stuck patterns - symmetric removal prevents exploitation

        return (self.steps >= self.max_steps or
                self.track_length >= self.max_track_length or
                too_far)

    def _get_observation(self):
        current_distance = self._calculate_distance_to_start()[0]
        
        # Calculate direction vector to goal (normalized)
        goal_vector = np.array(self.goal_position[:2]) - np.array(self.current_position[:2])
        goal_distance_2d = np.linalg.norm(goal_vector)
        if goal_distance_2d > 0:
            goal_direction = goal_vector / goal_distance_2d
        else:
            goal_direction = np.array([0.0, 0.0])
        
        # Calculate angle to goal (how much to turn to face goal)
        current_dir_vector = np.array(self.direction_vectors[self.current_direction])
        dot_product = np.dot(current_dir_vector, goal_direction)
        angle_to_goal = np.arccos(np.clip(dot_product, -1, 1))  # in radians
        
        # Calculate distance trend (positive = getting closer)
        if len(self.position_history) >= 2:
            old_dist = np.linalg.norm(np.array(self.position_history[0]) - np.array(self.goal_position))
            distance_trend = old_dist - current_distance
        else:
            distance_trend = 0.0
        
        # Use the observation space's expected size, not the current max_track_length
        # This ensures compatibility when curriculum learning changes max_track_length
        expected_track_size = self.observation_space['track_pieces'].shape[0]
        
        observation = {
            'track_pieces': np.array(self.track_pieces + [0] * (expected_track_size - len(self.track_pieces)), dtype=np.int32),
            'current_position': np.array(self.current_position, dtype=np.int32),
            'current_direction': self.current_direction,
            'distance_to_start': np.array([current_distance], dtype=np.float32),
            'track_length': self.track_length,
            'last_piece_type': self.last_piece_type,
            'goal_direction': goal_direction.astype(np.float32),
            'angle_to_goal': np.array([angle_to_goal], dtype=np.float32),
            'distance_trend': np.array([distance_trend], dtype=np.float32),
        }
        
        # Check if any values exceed their space
        track_pieces_space = self.observation_space['track_pieces']
        if isinstance(track_pieces_space, gym.spaces.Box):
            if np.any(observation['track_pieces'] < track_pieces_space.low) or np.any(observation['track_pieces'] > track_pieces_space.high):
                raise ValueError(f"track_pieces values are outside the defined space: {observation['track_pieces']}")

        current_position_space = self.observation_space['current_position'] 
        if np.any(observation['current_position'] < current_position_space.low) or np.any(observation['current_position'] > current_position_space.high):
            raise ValueError(f"current_position values are outside the defined space: {observation['current_position']}")
        
        if observation['current_direction'] >= self.observation_space['current_direction'].n:
            raise ValueError(f"current_direction value exceeds the defined space: {observation['current_direction']}")
        
        if observation['track_length'] >= self.observation_space['track_length'].n:
            raise ValueError(f"track_length value exceeds the defined space: {observation['track_length']}")
        
        if observation['last_piece_type'] >= self.observation_space['last_piece_type'].n:
            raise ValueError(f"last_piece_type value exceeds the defined space: {observation['last_piece_type']}")
        
        return observation

    def evaluate_ride(self):
        resp = self.api_controller.get_ride_ratings()
        if resp.get("success"):
            ratings = resp["payload"]
            return {
                'excitement': ratings.get('excitement', 0),
                'intensity': ratings.get('intensity', 0),
                'nausea': ratings.get('nausea', 0)
            }
        else:
            if self.verbose >= 1:
                print("Failed to get ride ratings")
            return {
                'excitement': 0,
                'intensity': 0,
                'nausea': 0
            }
    
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
        
        # Update current position to end of station
        self.current_position = [current_x, current_y, current_z]
        self.current_direction = current_dir
        
        if self.verbose >= 2:
            print(f"Station built. Track starts at {self.station_start_position}, current position: {self.current_position}")
            print(f"Goal: Place track at {self.goal_position} to connect to BeginStation at {self.station_start_position}")
        
        return True

    def close(self):
        if self.api_controller:
            self.api_controller.disconnect()

    def render(self):
        if self.render_mode == "human":
            pass

