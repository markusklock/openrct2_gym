#!/usr/bin/env python3
"""
Verify that the env's CLOSING GOAL (position + orientation) matches the REAL loop-closure
point reported by the OpenRCT2 API.

WHY: in training the agent reaches the goal *guide* tile (min_distance -> 0) yet almost
never completes the circuit. That points at a possible mismatch between what the agent is
trained to reach and where the loop actually closes. The env hardcodes, on every reset:

    station_start_position = [61, 66, 14]                       (openrct2_env.py:418)
    goal_position          = station_start + [1, 0, 0]  -> [62, 66, 14]   (:422-426)
    close_dir              = North (_STATION_ENTRY_DIR)         (_init_closing_target)

...regardless of the actual track geometry. This script builds a known, closeable loop on
top of the auto-built 6-piece station and checks:
  1. does the API actually report a complete circuit?
  2. is the closing piece placed AT goal_position, heading close_dir?  (position+orientation)
  3. does the observation say "at goal" (goal_disp ~ 0) when the head is there?

Loop built (on top of the 6-piece BeginStation):
    2x small right curve  (action 4  / trackType 43, QuarterTurn3Tiles right)
    6x straight           (action 0  / trackType 0,  Flat)
    2x small right curve  (action 4)
Edit LOOP_ACTIONS to try other shapes.

RUN with port 8080 FREE (no training using it):
    python verify_goal_position.py
"""
import math

DIR_NAMES = {0: "N", 1: "E", 2: "S", 3: "W"}
ACTION_NAMES = {0: "straight", 4: "right-curve"}

# The loop as discrete agent actions. action 4 = small right curve (trackType 43),
# action 0 = straight (trackType 0). The 6-piece station is auto-built on reset().
LOOP_ACTIONS = [4, 4] + [0] * 6 + [4, 4]


def _d(a, b):
    return math.dist(list(a), list(b))


def main():
    from openrct2_gym.envs.openrct2_env import OpenRCT2Env

    try:
        env = OpenRCT2Env(host="localhost", port=8080, verbose=0)
    except Exception as e:
        print(f"❌ Could not connect to OpenRCT2 API on port 8080: {e}")
        print("   Make sure the plugin server is running and no training is using the port.")
        return

    obs, _ = env.reset()                 # demolish-all + create ride + build 6-piece station
    env.auto_backtrack_enabled = False   # keep the geometric build clean (no surprise removals)

    goal = list(env._reward_target_position())   # close_pos if calibrated, else goal_position
    close_dir = env._reward_target_direction()   # close_dir (station-entry heading)

    print("=" * 72)
    print("ENV's CLOSING GOAL  (what the agent is trained to reach)")
    print("=" * 72)
    print(f"  station_start_position  : {env.station_start_position}")
    print(f"  goal_position           : {env.goal_position}")
    print(f"  reward target position  : {goal}")
    print(f"  reward target heading   : {close_dir} "
          f"({DIR_NAMES.get(close_dir, '?') if close_dir is not None else 'uncalibrated'})")
    print(f"  STATION_HEIGHT          : {env.STATION_HEIGHT}")
    print(f"  build head after station: pos={list(env.current_position)} "
          f"dir={env.current_direction} ({DIR_NAMES.get(env.current_direction)})")
    print(f"  goal_disp at start      : {[round(float(x), 3) for x in obs['goal_disp']]}")
    print()

    print("=" * 72)
    print(f"BUILDING LOOP: {[ACTION_NAMES[a] for a in LOOP_ACTIONS]}")
    print("=" * 72)

    closed = False
    closing_head = closing_dir = closure_pos = closure_dir = None
    min_d, min_state = float("inf"), None

    for i, action in enumerate(LOOP_ACTIONS):
        prev_pos, prev_dir = list(env.current_position), int(env.current_direction)
        obs, reward, terminated, truncated, info = env.step(action)
        pos, cdir = list(env.current_position), int(env.current_direction)
        phi = env._potential(env.reward_params)        # Phi should rise as the head approaches/docks
        placed = pos != prev_pos
        d = _d(pos, goal)
        if d < min_d:
            min_d = d
            min_state = (pos, cdir, [round(float(x), 3) for x in obs['goal_disp']])
        flag = "CLOSED!" if env.loop_completed else ("ok" if placed else "FAILED (no move)")
        print(f"  step {i:2d}  {ACTION_NAMES[action]:11s} -> head={pos} dir={DIR_NAMES.get(cdir)}"
              f"  dist={d:5.2f}  goal_disp={[round(float(x), 3) for x in obs['goal_disp']]}"
              f"  Phi={phi:6.2f}  r={reward:7.2f}  [{flag}]")
        if env.loop_completed:
            closed = True
            closing_head, closing_dir = prev_pos, prev_dir       # the closing piece was placed here
            closure_pos, closure_dir = pos, cdir                 # API "Final position" (post-closure head)
            break

    print()
    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    print(f"  1. API circuit complete            : {'YES ✅' if closed else 'NO ❌'}")

    if closed:
        pos_match = list(closure_pos) == list(goal)
        dir_match = (close_dir is not None) and (closure_dir == close_dir)
        print(f"  2. API closure point (Final pos)   : pos={closure_pos} dir={DIR_NAMES.get(closure_dir)}")
        print(f"     env goal (reward target)        : pos={goal} dir={DIR_NAMES.get(close_dir)}")
        print(f"     --> POSITION    {'MATCH ✅' if pos_match else 'MISMATCH ❌  (goal tile is NOT the real closure tile)'}")
        print(f"     --> ORIENTATION {'MATCH ✅' if dir_match else 'MISMATCH ❌  (close_dir is NOT the real closing heading)'}")
        print(f"     (closing piece was placed at staging head {closing_head} dir {DIR_NAMES.get(closing_dir)})")
    else:
        print("     loop did NOT close with these pieces -- either the sequence doesn't form a")
        print(f"     closed circuit, or the geometry differs. final head: pos={list(env.current_position)}"
              f" dir={DIR_NAMES.get(int(env.current_direction))}")

    if min_state is not None:
        gd = min_state[2]
        near_zero = all(abs(x) < 0.05 for x in gd)
        print(f"  3. closest head got to goal        : dist={min_d:.2f} at pos={min_state[0]} dir={DIR_NAMES.get(min_state[1])}")
        print(f"     observation goal_disp there     : {gd}")
        print(f"     --> obs says 'AT GOAL' (~0)      : {'YES ✅' if near_zero else 'NO ❌  (obs never reads at-goal, even at closest)'}")

    print("=" * 72)


if __name__ == "__main__":
    main()
