# openrct2_gym
[Example output](https://github.com/user-attachments/assets/932124f9-4e22-404b-9d74-ceb89f81cfba)

This is a custom environment for [Gymnasium](https://gymnasium.farama.org/) that aims to train a RL agent to build great rollercoasters in OpenRCT2.

This project is heavily inspired by Dylan Eberts [neural_rct](https://dylanebert.com/neural_rct/) and Kevin Burke who held a [talk](https://www.youtube.com/watch?v=6mRFITUwCVU) which I thought was awesome.
That talk really wanted me to try it myself but I was just a mediocre python programmer with basic knowledge about neural networks. But now, thanks to LLM assisted coding I was able to create something that (kind of) works!

I started with the same method as Kevin Burke, reverse engineering the TD6-file format and create a python program that generates a rollercoaster in TD6-format. I was able to solve some of the issues Kevin had like collision detection by just creating a simple 3D-representation of the track in python.
But the real issue I then faced was trying to evaluate the ride to get its score. The game does this by just starting the ride, have the cart run through it and record all the values for speed, G-forces and such.
To do this in python I would first need to implement the whole physics engine from the game (the game even looks for scenery close to the track when evaluating the score) to get an accurate score in my python program.

Initially I tried using UI automation with pyautogui to click buttons in the game, but this was incredibly slow and fragile. The breakthrough came when I realized OpenRCT2 has an excellent [Scripting API](https://github.com/OpenRCT2/OpenRCT2/blob/develop/distribution/scripting.md) that could be used to create a game plugin exposing an HTTP API.

## Current Status

The project has evolved significantly from its UI automation origins. It now features:

- **API-based control**: Direct communication with OpenRCT2 via HTTP API with retry logic
- **Physics-aware rewards**: Energy estimation helps agent understand chain lifts add energy, drops provide speed
- **Pattern detection**: Rewards for building proper lift hills, drops, and turnarounds
- **5-phase curriculum learning**: Progressive skill acquisition from navigation to ride quality optimization
- **Parallel training**: Multiple OpenRCT2 instances on different ports using DummyVecEnv
- **Action masking**: Prevents invalid track placements
- **Ride quality optimization**: Phase 5 targets Excitement 7-9, Intensity 4.5-6.5, Nausea <4.5
- **3D-aware observation + build memory**: an egocentric, forward-aligned 2.5D map (occupancy / signed height / goal+station marker / chain-lift trail) gives spatial awareness, and a structured build-history buffer (last 128 pieces: learned piece-type embedding + per-piece relative geometry) gives the agent memory of what it has built. Both are encoded by a custom feature extractor (`openrct2_gym/envs/feature_extractor.py`) with a GRU over the history and a CNN over the map, plus `VecNormalize` over the continuous scalars.

> **Note:** The observation space was redesigned (see `openrct2_gym/envs/obs_config.py`). This changes the model input shape, so **models trained before this change cannot be loaded** — retrain from scratch. Each checkpoint now has a matching `*_vecnormalize.pkl` stats file that must accompany the model when resuming or running it.

The improved curriculum teaches the agent to:
1. **Phase 1**: Navigate back to the station (basic circuit completion)
2. **Phase 2**: Build proper lift hills with chain lifts (energy accumulation)
3. **Phase 3**: Create drops and turnarounds (use stored energy)
4. **Phase 4**: Integrate all skills for consistent circuit completion
5. **Phase 5**: Optimize ride ratings for quality roller coasters

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

### Quick Start - Improved 5-Phase Curriculum (Recommended)
```bash
# Single environment
python train_parallel_curriculum_masked.py --ports 8080 --improved --timesteps 1000000

# Multiple parallel environments (faster training)
python train_parallel_curriculum_masked.py --ports 8080,8081,8082,8083 --improved --timesteps 1000000
```

The `--improved` flag enables the physics-aware 5-phase curriculum with energy estimation and pattern detection.

### Alternative Training Options

```bash
# Legacy 3-phase curriculum
python train_parallel_curriculum_masked.py --ports 8080 --phased --timesteps 500000

# Standard training with action masking (no curriculum)
python train_rl_agent_masked.py --timesteps 500000
```

### Monitor Training Progress

```bash
# Start Tensorboard to view training metrics
tensorboard --logdir ./parallel_curriculum_masked_tensorboard/
```

## Reward System (Improved 5-Phase Curriculum)

The `--improved` flag enables a physics-aware reward system that teaches roller coaster mechanics:

### Energy Model

The agent learns that roller coasters need energy:
- **Chain lifts add energy**: +50 energy per chain lift piece
- **Drops convert potential to kinetic**: +3 energy per unit of descent
- **Friction costs energy**: -2 per piece, extra for turns
- **Climbing without chain costs energy**: -5 per unit of ascent

**Energy margin reward**: Agent is rewarded for maintaining positive energy (track is viable).

### Pattern Detection

Rewards for building proper roller coaster patterns:
- **Lift hill pattern** (3+ chain pieces → drop): +15 to +20
- **Drop pattern** (transition → steep descent → recovery): +10 to +15
- **Turnaround** (reversing direction toward station): +15 to +20

### 5-Phase Curriculum

Each phase emphasizes different skills:

| Phase | Name | Max Pieces | Focus | Advancement |
|-------|------|------------|-------|-------------|
| 1 | Return Practice | 25 | Pure navigation (4x distance rewards) | 50% completion |
| 2 | Lift Hill Building | 40 | Chain lifts + energy (2x energy rewards) | 40% with 3+ chains |
| 3 | Drop & Turn | 60 | Drops + turnarounds + patterns | 35% with patterns |
| 4 | Circuit Mastery | 80 | Full integration + approach guidance | 30% completion |
| 5 | Quality Optimization | 80-120 | Ride ratings (E=7-9, I=4.5-6.5, N<4.5) | Progressive |

### Approach Guidance (Soft, Non-Restrictive)

Near the station, the agent receives bonuses (not penalties) for:
- **Height alignment**: +5 when at correct height (z=14)
- **Flat piece usage**: +3 for using flat/transition pieces when close
- **Direction alignment**: +3 when facing toward station

### Circuit Completion

Any piece that the game engine accepts as completing the circuit is valid. The artificial restriction limiting completion to only flat pieces has been removed.

## Observation Space

The agent receives comprehensive information about the current state:

- **track_pieces**: Array of placed track piece IDs (up to 250)
- **current_position**: 3D coordinates (x, y, z)
- **current_direction**: Facing direction (0=North, 1=East, 2=South, 3=West)
- **distance_to_start**: Euclidean distance to station
- **track_length**: Number of pieces placed
- **last_piece_type**: ID of last placed piece
- **goal_direction**: Normalized 2D vector pointing to station
- **angle_to_goal**: Radians to turn to face station
- **distance_trend**: Change in distance (positive = getting closer)
- **last_ride_excitement/intensity/nausea**: Ratings from previous completed ride (for learning)

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
- Training can take 1-2 million timesteps for consistent success

## Recent Improvements

- **Physics-aware rewards**: Energy model helps agent understand roller coaster mechanics
- **Pattern detection**: Rewards for lift hills, drops, and turnarounds
- **5-phase curriculum**: Progressive skill acquisition
- **Ride quality optimization**: Phase 5 targets good excitement/intensity/nausea ratings
- **Parallel training stability**: DummyVecEnv with retry logic for reliable multi-instance training
- **Removed artificial restrictions**: Any valid circuit completion is accepted

## Future Improvements

- Support for different coaster types
- Visual observation space using track renders
- Transfer learning between coaster types
- More sophisticated physics model (G-forces, speed estimation)
