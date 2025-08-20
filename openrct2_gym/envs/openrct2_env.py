import gymnasium as gym
import numpy as np
import time
from collections import deque
from .api_track_builder import APITrackBuilder
from .api_controller import APIController

class OpenRCT2Env(gym.Env):
    def __init__(self, render_mode=None, host="localhost", port=8080):
        super(OpenRCT2Env, self).__init__()
        self.render_mode = render_mode
        self.api_controller = APIController(host, port)
        self.track_builder = APITrackBuilder(self.api_controller)
        
        # Connect to API server
        if not self.api_controller.connect():
            raise RuntimeError(f"Failed to connect to OpenRCT2 API server at {host}:{port}")
        
        # Define action and observation space (32 actions: 0-30 track pieces, 31 remove)
        self.action_space = gym.spaces.Discrete(32)
        
        # Track collision and backtracking
        self.collision_count = 0
        self.consecutive_failures = 0
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
        self.action_history = deque(maxlen=10)  # Track recent actions for pattern detection
        self.remove_count = 0  # Track number of remove actions
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
        if not success and action != 18:  # Failed to place a piece (not a remove action)
            self.consecutive_failures += 1
            self.collision_count += 1
            
            # Auto-backtrack if enabled and too many consecutive failures
            if self.auto_backtrack_enabled and self.consecutive_failures >= self.max_consecutive_failures:
                if len(self.track_builder.history) > 0:
                    print(f"Auto-backtracking after {self.consecutive_failures} consecutive failures")
                    # Force a remove action instead of the failed action
                    action = 31  # Change the action to remove
                    auto_backtracked = True
                    success, new_position, new_direction = self.track_builder.take_action(action, self.current_position, self.current_direction)
                    self.consecutive_failures = 0
        else:
            self.consecutive_failures = 0
        
        # Check if the last placement completed the circuit
        if success and len(self.track_builder.history) > 0:
            last_entry = self.track_builder.history[-1]
            if "is_complete" in last_entry:
                self._last_placement_complete = last_entry["is_complete"]
            else:
                self._last_placement_complete = False
        
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
        self.action_history.append(action)  # Track action history
        observation = self._get_observation()
        reward = self._calculate_reward(success)

        # Check for loop completion
        self.loop_completed = self._last_placement_complete
        if self.loop_completed:
            print(f"Loop has been completed, great success!")
        terminated = self.loop_completed
        
        # Check if episode was truncated
        truncated = self._is_trunkated()
        self.steps += 1
        
        # Create info dict with additional debugging information
        info = {
            'auto_backtracked': auto_backtracked,
            'original_action': original_action if auto_backtracked else action,
            'collision_count': self.collision_count,
            'consecutive_failures': self.consecutive_failures
        }
        
        if self.steps % 10 == 0 or auto_backtracked:  # Log less frequently or when backtracking
            print("Step: %s, Track: %s, Pos: %s, Action: %s%s, Dist: %.1f, Dir: %s" % 
                  (self.steps, self.track_length, self.current_position, 
                   self.last_action, " (auto-backtrack)" if auto_backtracked else "",
                   self._calculate_distance_to_start()[0], self.current_direction))

        if terminated:
            # Additional completion bonuses are already in _calculate_reward
            # Set ride to testing mode and get ratings
            self.api_controller.set_ride_status(1)  # 1 = Testing
            time.sleep(5)  # Wait for testing to complete
            ride_rating = self.evaluate_ride()
            info['ride_rating'] = ride_rating
        elif truncated:
            # Partial credit for getting close even if not completed
            final_distance = self._calculate_distance_to_start()[0]
            if final_distance < 10:
                reward += (10 - final_distance) * 2
            elif final_distance < 20:
                reward += (20 - final_distance) * 0.5

        return observation, reward, terminated, truncated, info

    def valid_action_mask(self):
        """
        Returns a boolean mask of valid actions.
        True = action is valid, False = action is invalid
        """
        mask = np.zeros(self.action_space.n, dtype=bool)
        
        # Get valid actions from the API
        valid_actions = self.track_builder.get_valid_actions()
        
        # Always allow remove action if there are pieces to remove
        if len(self.track_builder.history) > 0:
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
        self.station_start_position = [67, 66, 14]  # Where the first station piece is
        self.current_position = self.station_start_position.copy()
        self.goal_position = self.station_start_position.copy()  # Track must return to the FIRST station piece
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
        self.collision_count = 0
        self.consecutive_failures = 0
        self.previous_distance = None
        self.position_history = deque(maxlen=5)
        self.chain_lift_positions = set()
        self.action_history = deque(maxlen=10)
        self.remove_count = 0
        
        # Demolish ALL rides to ensure clean state
        # This is more reliable than just demolishing our tracked ride
        self.api_controller.delete_all_rides()
        time.sleep(0.5)  # Small delay to ensure cleanup completes
        
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

    def _calculate_reward(self, success):
        reward = 0
        current_distance = self._calculate_distance_to_start()[0]
        
        if success:
            # Base reward for successful action
            reward += 1

            # FIX: Reward for placing chain lifts in the beginning (actions 9 and 10)
            # But prevent exploitation by tracking positions
            if self.track_length < 30 and self.last_action in [9, 10]:
                position_key = tuple(self.current_position)
                # Only reward if this is a NEW position for chain lift
                if position_key not in self.chain_lift_positions:
                    if self.chain_lift_count < self.max_chain_lifts:
                        reward += 10  # Increased from 5
                        self.chain_lift_count += 1
                        self.chain_lift_positions.add(position_key)
                else:
                    # Penalty for rebuilding chain lift in same position
                    reward -= 3
            
            # Penalty for removing pieces with progressive penalty
            if self.last_action == 31:  # Fixed: was 18, should be 31
                self.remove_count += 1
                # Progressive penalty: increases with number of removals
                remove_penalty = min(1 + self.remove_count * 0.5, 5)
                reward -= remove_penalty
                
                # Additional penalty for remove-build-remove patterns
                if len(self.action_history) >= 4:
                    recent = list(self.action_history)[-4:]
                    # Check for alternating pattern of build(9/10) and remove(31)
                    if recent.count(31) >= 2 and any(a in [9, 10] for a in recent):
                        reward -= 5  # Heavy penalty for exploitative pattern

            # Big reward for completing the loop
            if self.loop_completed:
                reward += 100
                # Length bonus for longer completed tracks
                if self.track_length > 50:
                    reward += (self.track_length - 50) * 0.5
                # Chain lift usage bonus
                if self.chain_lift_count > 0:
                    reward += 10
            
            # Progressive distance-based rewards
            if self.previous_distance is not None:
                distance_delta = self.previous_distance - current_distance
                
                # Phase-based distance rewards
                if self.track_length <= 35:
                    # Building phase - allow exploration, reward height for chain lift
                    if self.current_position[2] > self.goal_position[2]:
                        reward += 0.2
                    # Small reward for building outward
                    if self.last_action not in [31]:
                        reward += 0.3
                elif self.track_length <= 60:
                    # Transition phase - gentle guidance back
                    if distance_delta > 0:  # Moving closer
                        reward += distance_delta * 0.3
                else:
                    # Return phase - strong pull back to station
                    if distance_delta > 0:  # Moving closer
                        reward += distance_delta * 0.5
                    else:  # Moving away
                        reward -= abs(distance_delta) * 0.3
                    
                    # Extra rewards for getting very close
                    if current_distance < 30:
                        reward += (30 - current_distance) * 0.2
                    if current_distance < 10:
                        reward += 2
            
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
            
            # Reward for continuous forward progress (no recent removes)
            if self.last_action not in [31] and 31 not in list(self.action_history)[-3:]:
                reward += 0.2  # Bonus for sustained building
            
            # Adjusted height penalty (only after building phase)
            if self.track_length > 35 and self.current_position[2] > 25:
                reward -= 0.2
                
        else:
            # Reduced penalty for failed placement (exploration)
            reward -= 0.2  # Reduced from -0.5
            
        # Update previous distance for next step
        self.previous_distance = current_distance
        
        print("Reward was: %.2f (dist: %.1f, track: %d)" % (reward, current_distance, self.track_length))
        return reward

    def _calculate_distance_to_start(self):
        point_a = np.array(self.current_position)
        point_b = np.array(self.goal_position)
        distance = float(np.linalg.norm(point_a - point_b))
        return np.array([distance], dtype=np.float32)

    def _is_trunkated(self):
        return (self.steps >= self.max_steps or 
                self.track_length >= self.max_track_length)

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
        
        observation = {
            'track_pieces': np.array(self.track_pieces + [0] * (self.max_track_length - len(self.track_pieces)), dtype=np.int32),
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
                print(f"Failed to place station piece {i}: {resp.get('error')}")
                return False
            
            next_ep = resp["payload"]["nextEndpoint"]
            current_x = next_ep["x"]
            current_y = next_ep["y"]
            current_z = next_ep["z"]
            current_dir = next_ep["direction"]
            
            # Update track state
            self.track_pieces.append(0)  # Station pieces as action 0
            self.track_length += 1
        
        # Update current position to end of station
        self.current_position = [current_x, current_y, current_z]
        self.current_direction = current_dir
        
        print(f"Station built. Track starts at {self.station_start_position}, current position: {self.current_position}")
        print(f"Goal: Return to {self.goal_position} (first station piece)")
        
        return True

    def close(self):
        if self.api_controller:
            self.api_controller.disconnect()

    def render(self):
        if self.render_mode == "human":
            pass

