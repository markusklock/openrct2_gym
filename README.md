# openrct2_gym
[Example output](https://github.com/user-attachments/assets/116cab1b-262a-4f88-a0dd-e8129195f03c)

This is a custom environment for [Gymnasium](https://gymnasium.farama.org/) that aims to train a RL agent to build great rollercoasters in OpenRCT2.

This project is heavily inspired by Dylan Eberts [neural_rct](https://dylanebert.com/neural_rct/) and Kevin Burke who held a [talk](https://www.youtube.com/watch?v=6mRFITUwCVU) which I thought was awesome.
That talk really wanted me to try it myself but I was just a mediocre python programmer with basic knowledge about neural networks. But now, thanks to LLM assisted coding I was able to create something that (kind of) works!

I started with the same method as Kevin Burke, reverse engineering the TD6-file format and create a python program that generates a rollercoaster in TD6-format. I was able to solve some of the issues Kevin had like collision detection by just creating a simple 3D-representation of the track in python.
But the real issue I then faced was trying to evaluate the ride to get its score. The game does this by just starting the ride, have the cart run through it and record all the values for speed, G-forces and such.
To do this in python I would first need to implement the whole physics engine from the game (the game even looks for scenery close to the track when evaluating the score) to get an accurate score in my python program.

Initially I tried using UI automation with pyautogui to click buttons in the game, but this was incredibly slow and fragile. The breakthrough came when I realized OpenRCT2 has an excellent [Scripting API](https://github.com/OpenRCT2/OpenRCT2/blob/develop/distribution/scripting.md) that could be used to create a game plugin exposing an HTTP API.

## Current Status

The project has evolved significantly from its UI automation origins. It now features:

- **API-based control**: Direct communication with OpenRCT2 via HTTP API (requires custom OpenRCT2 plugin)
- **Fast training**: No UI automation, direct game state manipulation
- **Sophisticated reward system**: Multi-phase rewards encouraging chain lifts, exploration, and return to station
- **Curriculum learning**: Gradually increases difficulty as the agent improves
- **Action masking**: Prevents invalid track placements
- **Comprehensive metrics**: Detailed Tensorboard tracking of all aspects of training

The agent is currently learning to:
1. Build tracks starting with chain lift hills
2. Explore outward to create interesting layouts
3. Navigate back to the station to complete the circuit
4. Gradually build longer and more complex tracks

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/openrct2_gym.git
cd openrct2_gym

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Prerequisites

1. **OpenRCT2** with the API plugin installed and running (default port 8080)
2. **Python 3.8+**
3. **Linux/Mac** (Windows may work but is untested)

## Training the Agent

### Quick Start - Train with Curriculum Learning (Recommended)
```bash
python train_with_curriculum.py --timesteps 1000000
```

This uses curriculum learning to gradually increase difficulty from 30-piece tracks to 100-piece tracks.

### Alternative Training Scripts

```bash
# Standard training with action masking
python train_rl_agent_masked.py --timesteps 500000

# Simple training without masking
python train_rl_agent_simple.py --timesteps 500000

# Enhanced training with detailed metrics
python train_rl_agent_enhanced.py --timesteps 500000
```

### Monitor Training Progress

```bash
# Start Tensorboard to view training metrics
tensorboard --logdir ./ppo_openrct2_tensorboard/

# For curriculum training logs
tensorboard --logdir ./curriculum_tensorboard/
```

## Reward System

The reward system is carefully designed to guide the agent through different phases of track construction. Here's a detailed breakdown:

### Base Rewards

- **Successful track placement**: +1.0
- **Failed placement (collision)**: -0.2
- **Track removal action**: Progressive penalty starting at -1.0, increasing by 0.5 per removal (caps at -5.0)

### Chain Lift Rewards

Chain lifts are crucial for roller coasters. The agent is encouraged to use them early:

- **Chain lift placement** (actions 9 or 10): +10.0 bonus (only for first placement at each position)
- **Rebuilding chain lift at same position**: -3.0 penalty (prevents exploitation)
- **Chain lift zone**: First 30 track pieces
- **Maximum chain lifts**: 15 per track

### Phase-Based Distance Rewards

The reward system uses three distinct phases to guide behavior:

#### Phase 1: Building Phase (0-25 pieces)
- **Height gain reward**: +0.2 for building above station height
- **Building reward**: +0.3 for placing non-removal pieces
- **Focus**: Encourage initial hill climb with chain lift

#### Phase 2: Transition Phase (25-35 pieces)
- **Distance delta reward**: +0.5 × distance_moved_closer
- **Direction reward**: +0.3 for facing toward goal
- **Focus**: Start guiding back toward station

#### Phase 3: Return Phase (35+ pieces)
- **Strong distance delta reward**: +2.0 × distance_moved_closer
- **Distance penalty**: -1.0 × distance_moved_away
- **Direction rewards**:
  - +1.0 for facing directly toward goal (< 45°)
  - +0.5 for somewhat toward goal (< 90°)
  - -0.5 for facing away from goal

### Distance Checkpoint Bonuses

Major bonuses for crossing distance thresholds when returning:
- **50 units**: +5 bonus
- **30 units**: +10 bonus
- **20 units**: +15 bonus
- **10 units**: +20 bonus

### Proximity Rewards

Escalating rewards for getting close to the station:
- **< 40 units**: +(40 - distance) × 0.3
- **< 20 units**: +(20 - distance) × 0.5
- **< 10 units**: +(10 - distance) × 1.0
- **< 5 units**: +(5 - distance) × 2.0

### Distance Penalties

Soft boundaries to prevent going too far:
- **> 60 units**: -(distance - 60) × 0.2
- **> 80 units**: -(distance - 80) × 0.5
- **> 100 units**: Episode terminates

### Track Length Milestones

Bonuses for reaching certain track lengths:
- **30 pieces**: +5
- **50 pieces**: +10
- **75 pieces**: +15
- **100 pieces**: +20
- **Continuous bonus**: +0.1 per piece after 30

### Loop Completion Rewards

Major rewards for successfully completing the circuit:
- **Base completion bonus**: +100
- **Length bonus**: +(track_length - 50) × 0.5 for tracks over 50 pieces
- **Chain lift usage bonus**: +10 if chain lifts were used

### Behavioral Rewards

- **Sustained building bonus**: +0.2 for continuous building without recent removals
- **Pattern penalty**: -5.0 for detected exploitation patterns (build-remove-build)

### Partial Success Rewards

Even if the loop isn't completed, proximity matters:
- **Final distance < 10 units**: +(10 - distance) × 2.0
- **Final distance < 20 units**: +(20 - distance) × 0.5

## Observation Space

The agent receives comprehensive information about the current state:

- **track_pieces**: Array of placed track piece IDs
- **current_position**: 3D coordinates (x, y, z)
- **current_direction**: Facing direction (0=North, 1=East, 2=South, 3=West)
- **distance_to_start**: Euclidean distance to station
- **track_length**: Number of pieces placed
- **last_piece_type**: ID of last placed piece
- **goal_direction**: Normalized 2D vector pointing to station
- **angle_to_goal**: Radians to turn to face station
- **distance_trend**: Change in distance (positive = getting closer)

## Action Space

32 discrete actions representing different track pieces and operations:
- **0**: Flat/straight track
- **1-4**: Turn pieces (various radii)
- **5-8**: Slope pieces (up/down, various angles)
- **9-10**: Chain lift pieces
- **11-30**: Various special pieces (banks, transitions, etc.)
- **31**: Remove last piece

## Metrics and Monitoring

The training scripts provide extensive metrics in Tensorboard:

- **Success metrics**: Loop completion rate, track lengths
- **Navigation metrics**: Minimum distance reached, distance improvements
- **Chain lift metrics**: Usage patterns, efficiency
- **Behavioral metrics**: Collision rates, remove action patterns
- **Phase rewards**: Rewards earned in each construction phase
- **Curriculum progress**: Stage advancement, success rates

## Current Limitations

- Requires custom OpenRCT2 plugin for API support
- Only tested with Wooden Coaster type
- Single-track construction (no multiple rides)
- No consideration of ride excitement/intensity ratings yet
- Training can take 1-2 million timesteps for consistent success

## Future Improvements

- Multi-agent training for parallel learning
- Incorporation of ride ratings into reward system
- Support for different coaster types
- Visual observation space using track renders
- Transfer learning between coaster types
