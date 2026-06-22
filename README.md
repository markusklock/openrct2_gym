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
- **Potential-based reward (PBRS)**: a single, unified, policy-invariant reward that gives a dense gradient toward closing the circuit — and is provably un-farmable (no place/remove exploit)
- **Auto-calibrated closing target**: the geometry required to close the circuit is learned from the first completion, so the reward always points at the true closable state
- **5-phase curriculum learning**: Progressive skill acquisition from navigation to ride quality optimization
- **Parallel training**: Multiple OpenRCT2 instances on different ports using SubprocVecEnv
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
python train.py --ports 8080 --timesteps 1000000

# Multiple parallel environments (faster training)
python train.py --ports 8080,8081,8082,8083 --timesteps 1000000
```

`train.py` always uses the improved 5-phase curriculum with the potential-based reward (see [Reward System](#reward-system-potential-based-shaping)). When two or more available ports are provided, training automatically uses `SubprocVecEnv`; with one available port it uses `DummyVecEnv`.

### Monitor Training Progress

```bash
# Start Tensorboard to view training metrics
tensorboard --logdir ./parallel_curriculum_masked_tensorboard/
```

## Reward System (Potential-Based Shaping)

The agent is trained with a **single, unified, potential-based reward** (PBRS — Ng, Harada & Russell, 1999). The curriculum never swaps the reward function; it only sets parameters per phase. Each step's reward is:

```
reward = F + sparse terms,    where   F = γ·Φ(s′) − Φ(s)
```

### Potential Φ(s)

Φ measures how close the build head is to a state from which the circuit can **close**, combining three alignment axes plus energy viability:

- **Horizontal alignment** to the closing tile
- **Height alignment** to station height
- **Heading alignment** to the closing direction
- **Energy viability** — a sigmoid of the energy margin (does the track have enough energy to make it home?)

Φ is bounded and maximized exactly at the closable state, so `F = γ·Φ(s′) − Φ(s)` gives a **dense gradient that pulls the agent toward completing the circuit** — directly targeting the hardest part of the problem.

**Why potential-based?** PBRS is *provably policy-invariant*: shaping changes how fast the agent learns, never the optimal policy. Two consequences:
- Place-then-remove **telescopes to ≈(γ−1)·Φ < 0**, so it can never be farmed — the place/remove exploit is impossible by construction (no symmetric-removal bookkeeping needed).
- The Φ weights are **fixed across all phases**, preserving global invariance — only the sparse objectives below vary by phase.

### Sparse objectives (the real goals)

- **Circuit completion: +1000** — the dominant signal. Completion always strictly outweighs all accumulated shaping, so the agent is *completion-first*.
- **Ride quality (phase 5 only): 0–500** — a smooth, bounded, **non-negative** bonus peaking at Excitement ≈ 8, Intensity ≈ 5.5, low Nausea. Applied only on a completed, ride-tested circuit, so a finished ride is never punished.
- **Small penalties** — a tiny failed-placement penalty, plus an optional small per-step cost in phase 5 to discourage stalling.

### Closing target: deterministic heading + calibrated refinement

`goal_position` is the station's **dock endpoint** (verified by `verify_goal_position.py`: the API closes the circuit at `station_start`, so the guide points exactly at the closable tile — it was previously a tile off); the game itself decides completion via its `isCircuitComplete` flag. The **closing heading is deterministic from the station build** — the station is created with `startDir=0`, so every circuit re-enters BeginStation heading North — and Φ is handed this heading (`_STATION_ENTRY_DIR`) from step 1, the heading reward **coupled to the dock** (gated by the near-closure factor) so the agent routes freely and only aligns as it docks. This is what lets the agent learn the final connection during the **Phase-1 cold start**: previously the heading term stayed disabled until the *first* completion calibrated it, a chicken-and-egg (no heading → no completion → no heading) that stalled bootstrapping. The full closing geometry (the exact pre-close head position/direction) is still refined from real completions and cached to `logs/close_geometry.json`, but the anchor is **locked only after ≥3 completions agree** (median position, modal direction) so a single fluky closure can't poison Φ for the rest of the run.

### Energy model (feeds Φ's viability term)

- **Chain lifts add energy**: +50 per chain-lift piece
- **Drops convert height to speed**: +3 per unit of descent
- **Friction costs energy**: −2 per piece (extra for turns)
- **Climbing without a chain costs energy**: −5 per unit of ascent

The **energy margin** estimates whether the current track can still return to the station. Φ rewards keeping it positive, which naturally encourages lift-hill-then-drop structure without any explicit per-pattern bonus.

### 5-Phase Curriculum (parameters only)

All five phases share the **same reward and the same Φ**; they differ only in the track-length limit, whether ride-testing/quality is active, and the advancement criterion:

| Phase | Name | Max Pieces | What it adds | Advancement |
|-------|------|------------|--------------|-------------|
| 1 | Return Practice | 40 | completion only | 50% completion |
| 2 | Lift Hill Building | 40 | staged chain-lift bridge | 2.1 one-chain roundtrip, 2.2 one-chain completion, 2.3 three-chain completion |
| 3 | Drop & Turn | 60 | completion with lift/drop structure | 35% with chain lifts and a drop |
| 4 | Circuit Mastery | 80 | completion only | 30% completion |
| 5 | Quality Optimization | 80–120 | ride testing + quality bonus (+ tiny step cost) | progressive length |

Circuit completion is whatever the game engine accepts via `isCircuitComplete` — there are no artificial restrictions on which piece may close the loop. Height, flatness, and heading near the station are handled smoothly by Φ's alignment terms rather than by hard rules.

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

- **Potential-based reward (PBRS)**: a single unified, policy-invariant reward replacing the old per-step shaping — provably un-farmable
- **Auto-calibrated closing target**: the closing geometry is learned from the first completed circuit, so the reward targets the true closable state
- **Completion-first objective**: a large completion bonus dominates; ride quality is a smooth, bounded bonus in phase 5
- **5-phase curriculum**: phases now only set parameters — the reward and potential stay identical throughout
- **Parallel training**: `train.py` uses `SubprocVecEnv` automatically for multi-port training and `DummyVecEnv` for single-port training

## Future Improvements

- Support for different coaster types
- Visual observation space using track renders
- Transfer learning between coaster types
- More sophisticated physics model (G-forces, speed estimation)
