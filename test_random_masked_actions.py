#!/usr/bin/env python3
"""
Test script for validating action masking in OpenRCT2 environment.
Picks random actions from the list of masked/valid actions and logs detailed information.
"""

import gymnasium as gym
import openrct2_gym
from openrct2_gym.envs.wrappers import OpenRCT2Wrapper
import numpy as np
import random
import time
from collections import defaultdict

# Action to track piece name mapping for logging
ACTION_NAMES = {
    0: "Flat/Straight",
    1: "Left Turn (5-tile)",
    2: "Right Turn (5-tile)",
    3: "Left Turn (3-tile tight)",
    4: "Right Turn (3-tile tight)",
    5: "Up 25° (no chain)",
    6: "Down 25°",
    7: "Up 60° Steep",
    8: "Down 60° Steep",
    9: "Up 25° WITH CHAIN LIFT",
    10: "Flat to Up 25° WITH CHAIN LIFT",
    11: "Flat to Up 25° (no chain)",
    12: "Flat to Down 25°",
    13: "Up 25° to Flat",
    14: "Down 25° to Flat",
    15: "Flat to Left Bank",
    16: "Flat to Right Bank",
    17: "Left Bank (continuous)",
    18: "Right Bank (continuous)",
    19: "Left Bank to Flat",
    20: "Right Bank to Flat",
    21: "Banked Left Turn (5-tile)",
    22: "Banked Right Turn (5-tile)",
    23: "Banked Left Turn (3-tile)",
    24: "Banked Right Turn (3-tile)",
    25: "Up 25° to Up 60°",
    26: "Up 60° to Up 25°",
    27: "Down 25° to Down 60°",
    28: "Down 60° to Down 25°",
    29: "S-Bend Left",
    30: "S-Bend Right",
    31: "REMOVE PIECE"
}

def test_random_actions():
    print("=" * 80)
    print("OpenRCT2 Random Action Masking Test")
    print("=" * 80)
    print("\nThis test will:")
    print("1. Build 6 station pieces")
    print("2. Take 100 random actions from valid/masked actions only")
    print("3. Log detailed information about each action")
    print("4. Automatically back off when collisions occur")
    print("=" * 80)
    
    # Create environment with wrapper to expose valid_action_mask
    env = gym.make('OpenRCT2-v0')
    env = OpenRCT2Wrapper(env)
    
    # Reset environment - this builds the initial station
    print("\n[SETUP] Resetting environment and building initial station...")
    obs, info = env.reset()
    print(f"[SETUP] Initial station built with {env.unwrapped.station_length} pieces")
    print(f"[SETUP] Starting position: {env.unwrapped.current_position}")
    print(f"[SETUP] Starting direction: {env.unwrapped.current_direction}")
    
    # Statistics tracking
    action_counts = defaultdict(int)
    successful_actions = 0
    failed_actions = 0
    collision_count = 0
    backtrack_count = 0
    
    print("\n" + "=" * 80)
    print("Starting Random Action Testing")
    print("=" * 80)
    
    for step in range(100):
        print(f"\n--- Step {step + 1}/100 ---")
        
        # Get valid action mask
        valid_mask = env.valid_action_mask()
        valid_actions = np.where(valid_mask)[0].tolist()
        
        print(f"[MASK] Number of valid actions: {len(valid_actions)}")
        print(f"[MASK] Valid action IDs: {valid_actions}")
        
        if not valid_actions:
            print("[ERROR] No valid actions available! This shouldn't happen.")
            print("[ERROR] Check the environment configuration.")
            break
        
        # Pick a random valid action
        action = random.choice(valid_actions)
        action_name = ACTION_NAMES.get(action, f"Unknown({action})")
        
        print(f"[ACTION] Selected action {action}: {action_name}")
        print(f"[STATE] Current position: {env.unwrapped.current_position}")
        print(f"[STATE] Current direction: {env.unwrapped.current_direction}")
        print(f"[STATE] Track length: {env.track_length}")
        print(f"[STATE] Collision count: {env.collision_count}")
        
        # Store previous track length to check if placement was successful
        prev_track_length = env.track_length
        
        # Take the action
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Track statistics
        action_counts[action] += 1
        
        # Check if action was successful
        if env.unwrapped.last_action == action:
            # Action was executed as intended
            if action == 31:
                # Check if track length decreased (successful removal)
                if env.track_length < prev_track_length:
                    print(f"[RESULT] ✓ Successfully removed a piece")
                    successful_actions += 1
                    backtrack_count += 1
                else:
                    print(f"[RESULT] ✗ Failed to remove piece (no pieces to remove)")
                    failed_actions += 1
            else:
                # Check if track length increased (successful placement)
                if env.track_length > prev_track_length:
                    print(f"[RESULT] ✓ Successfully placed {action_name}")
                    successful_actions += 1
                else:
                    print(f"[RESULT] ✗ Failed to place {action_name} (collision)")
                    failed_actions += 1
                    collision_count += 1
        else:
            # Auto-backtrack happened
            if env.unwrapped.last_action == 31 and action != 31:
                print(f"[RESULT] ⚠ Auto-backtracked after failure (tried {action_name})")
                failed_actions += 1
                collision_count += 1
                backtrack_count += 1
        
        print(f"[STATE] New position: {env.unwrapped.current_position}")
        print(f"[STATE] New direction: {env.unwrapped.current_direction}")
        print(f"[REWARD] {reward:.2f}")
        
        # Check if loop completed
        if terminated:
            print("\n[COMPLETE] Loop completed successfully!")
            break
        
        if truncated:
            print("\n[TRUNCATED] Episode truncated (max steps or max track length)")
            break
        
        # Small delay for readability if needed
        # time.sleep(0.1)
    
    # Print statistics
    print("\n" + "=" * 80)
    print("Test Statistics")
    print("=" * 80)
    print(f"\nAction Summary:")
    print(f"  Total actions taken: {successful_actions + failed_actions}")
    print(f"  Successful actions: {successful_actions}")
    print(f"  Failed actions: {failed_actions}")
    print(f"  Collisions: {collision_count}")
    print(f"  Backtracks: {backtrack_count}")
    print(f"  Final track length: {env.track_length}")
    
    print(f"\nAction Distribution:")
    print("  (Shows which track pieces were attempted)")
    for action, count in sorted(action_counts.items()):
        action_name = ACTION_NAMES.get(action, f"Unknown({action})")
        percentage = (count / 100.0) * 100
        print(f"  Action {action:2d} ({action_name:30s}): {count:3d} times ({percentage:5.1f}%)")
    
    print(f"\nPiece Type Coverage:")
    used_categories = set()
    if any(a in action_counts for a in [0]):
        used_categories.add("Straight")
    if any(a in action_counts for a in [1, 2, 3, 4]):
        used_categories.add("Turns")
    if any(a in action_counts for a in [5, 6, 7, 8, 11, 12, 13, 14]):
        used_categories.add("Slopes")
    if any(a in action_counts for a in [9, 10]):
        used_categories.add("Chain Lifts")
    if any(a in action_counts for a in [15, 16, 17, 18, 19, 20]):
        used_categories.add("Banking")
    if any(a in action_counts for a in [21, 22, 23, 24]):
        used_categories.add("Banked Turns")
    if any(a in action_counts for a in [25, 26, 27, 28]):
        used_categories.add("Steep Transitions")
    if any(a in action_counts for a in [29, 30]):
        used_categories.add("S-Bends")
    
    print(f"  Categories used: {', '.join(sorted(used_categories))}")
    
    print("\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)
    print("\nInspect the rollercoaster in the game to verify:")
    print("1. No invalid pieces were placed")
    print("2. The system correctly masks invalid actions")
    print("3. All piece types can be selected when valid")
    print("4. Auto-backtracking works on repeated collisions")
    
    # Close environment
    env.close()

if __name__ == "__main__":
    test_random_actions()