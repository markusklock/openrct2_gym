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
  and protocol hardening (a timed-out placement sacrifices the episode rather than risking a
  silently desynced track; a hung game instance fails loudly instead of stalling the fleet)
- **Warm-start reverse curriculum**: completion is a *discovery* problem (a minimal loop is 12
  exact pieces — eight earlier runs never got past ~0.02% completion before entropy collapsed).
  The env replays a prefix of a live-verified loop at reset and the agent builds only the last
  k pieces; k anneals upward on success until episodes are cold (unscaffolded). Phase gates
  count cold episodes only, and every completion is harvested back into the loop library.
  This took the curriculum from "stuck in Phase 1 forever" to "all five phases in <1M steps".
- **Potential-based reward (PBRS)**: a single, unified reward that gives a dense gradient
  toward closing the circuit — and is provably un-farmable (no place/remove exploit)
- **Auto-calibrated closing target**: the closing heading is deterministic from the station
  build (active from step 1); the exact closing geometry is refined from real completions
- **5-phase curriculum learning**: navigation → lift hills → real drops & scale → big,
  train-verified coasters → ride-quality optimization
- **Parallel training**: Multiple OpenRCT2 instances on different ports using SubprocVecEnv
- **Action masking**: Prevents invalid track placements
- **Ride quality optimization**: Phase 5 targets Excitement 7-9, Intensity 4.5-6.5, Nausea <4.5,
  with a ramp+band bonus so every excitement increment pays (a pure target band gave no
  gradient from low excitement — two runs plateaued identically until this was reshaped)
- **3D-aware observation + build memory**: an egocentric, forward-aligned 2.5D map (occupancy / signed height / goal+station marker / chain-lift trail) gives spatial awareness, and a structured build-history buffer (last 128 pieces: learned piece-type embedding + per-piece relative geometry) gives the agent memory of what it has built. Both are encoded by a custom feature extractor (`openrct2_gym/envs/feature_extractor.py`) with a GRU over the history and a CNN over the map, plus `VecNormalize` over the continuous scalars. The scalar block also exposes the closing-corridor coordinates the reward pays on (along/perp/heading/route progress).

> **Note:** The observation space was redesigned (see `openrct2_gym/envs/obs_config.py`). This changes the model input shape, so **models trained before this change cannot be loaded** — retrain from scratch. Each checkpoint now has a matching `*_vecnormalize.pkl` stats file that must accompany the model when resuming or running it.

The improved curriculum teaches the agent to:
1. **Phase 1**: Navigate back to the station (basic circuit completion)
2. **Phase 2**: Build proper lift hills with chain lifts (staged one-chain → three-chain bridge)
3. **Phase 3**: Scale up — taller chained climbs, real multi-z drops, longer loops that stay
   energy-viable
4. **Phase 4**: Go big and prove it — steep (60°) drop segments, 40+ piece coasters, and the
   ride test verifying the train actually makes it all the way around
5. **Phase 5**: Optimize ride ratings (excitement/intensity/nausea) on an already-big,
   already-verified coaster

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

### Seed the warm-start loop library (once per map)

```bash
python build_loop_library.py --port 8080          # flat loops (Phase 1)
python build_loop_library.py --port 8080 --hill   # chain-hill loops (Phase 2)
python build_loop_library.py --port 8080 --big    # tall/steep big loops (Phases 3-4)
```

Each candidate is replayed against the live game and only enters `logs/loop_library.jsonl`
after a clean second replay also closes the circuit. Training harvests every completion back
into the library (dedup'd, per-class capped), so agent discoveries become future scaffolds.
Warm-start flags: `--no-warm-start`, `--loop-library PATH`, `--p-cold F`.

> **Game speed matters for Phases 4-5**: ride ratings take ~35s of *simulation* time
> (measured live), and Phase 4+ blocks on the ride test at each completion. `train.py`
> requests game speed 8 at startup (`--game-speed`, needs the plugin's `setGameSpeed`
> endpoint — redeploy the plugin after updating it), which shrinks the wait to ~4-5s.
> Rides that never rate within `ride_test_max_wait` honestly count as unverified
> (RCT2's −0.01 "unrated" sentinel is rejected, never scored).

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
- The **core** Φ geometry weights are fixed across all phases; a few auxiliary Φ terms
  (climb discovery, descent shaping, closure funnel, route-around-the-station progress) are
  deliberately phase-gated — invariance holds within a phase.

### Sparse objectives (the real goals)

- **Circuit completion: +1000** — the dominant signal. Completion always strictly outweighs all accumulated shaping, so the agent is *completion-first*. In the hill phases a flat loop earns only a fraction (the hill gate), so structure is required for full pay.
- **Structural bonus (phases 2-4): 0–250, completion-conditioned** — graded credit for chain
  count/height, total drop-z, and completed length toward per-phase targets. All components
  are ramps (partial progress pays); chain credit is elevation-scaled so chain-stub
  decoration on a flat loop cannot pay as a lift hill.
- **Verified viability (phase 4+): +150, completion-conditioned** — paid only when the ride
  test returns real stats, i.e. the train demonstrably made it all the way around.
- **Ride quality (phase 5): 0–500** — ramp+band per stat: half the credit ramps monotonically
  toward the target (every increment pays), half peaks at it (Excitement ≈ 8, Intensity ≈ 5.5,
  low Nausea). Applied only on a completed, ride-tested circuit, so a finished ride is never punished.
- **Small penalties** — a tiny failed-placement penalty (kept even when auto-backtrack fires).

### Closing target: deterministic heading + calibrated refinement

`goal_position` is the **staging tile one step on the approach side of the dock** (the head can never sit on the dock tile itself — it is occupied by BeginStation; the closing piece is placed *from* the staging tile onto the dock); the game itself decides completion via its `isCircuitComplete` flag. The **closing heading is deterministic from the station build** — the station is created with `startDir=0`, so every circuit re-enters BeginStation heading the entry direction — and Φ is handed this heading (`_STATION_ENTRY_DIR`) from step 1, the heading reward **coupled to the dock** (gated by the near-closure factor) so the agent routes freely and only aligns as it docks. The full closing geometry (the exact pre-close head position/direction) is still refined from real completions and cached to `logs/close_geometry.json`, but the anchor is **locked only after ≥3 completions agree** (median position, modal direction) so a single fluky closure can't poison Φ for the rest of the run.

### Energy model (feeds Φ's viability term)

- **Chain lifts add energy**: +50 per chain-lift piece
- **Drops convert height to speed**: +3 per unit of descent
- **Friction costs energy**: −2 per piece (extra for turns)
- **Climbing without a chain costs energy**: −5 per unit of ascent

The **energy margin** estimates whether the current track can still return to the station. Φ rewards keeping it positive, which naturally encourages lift-hill-then-drop structure without any explicit per-pattern bonus.

### 5-Phase Curriculum (parameters only)

All five phases share the **same reward and the same Φ**; they differ only in the track-length limit, whether ride-testing/quality is active, and the advancement criterion:

| Phase | Name | Max Pieces | What it adds | Advancement (cold episodes only) |
|-------|------|------------|--------------|--------------|
| 1 | Return Practice | 40 | completion only | 50% completion |
| 2 | Lift Hill Building | 40 | staged chain-lift bridge | 2.1 one-chain roundtrip, 2.2 one-chain completion, 2.3 three-chain completion |
| 3 | Real Drops & Scale | 60 | chain height ≥4z, drops ≥4z, length ≥25, energy-viable | 35% qualified |
| 4 | Big & Verified | 80 | height ≥6z, drops ≥8z incl. a 60° segment, length ≥40, **ride-test verified** | 30% qualified |
| 5 | Quality Optimization | 80–120 | quality bonus only (no step cost) | progressive length |

Phases 1–4 are scaffolded by the warm-start reverse curriculum (each phase's pool prefers
loops that can satisfy its own gate); Phase 5 always builds cold. Gate advancement counts
**cold (unscaffolded) episodes only**, so a scaffolded win can never advance a phase.

Circuit completion is whatever the game engine accepts via `isCircuitComplete` — there are no artificial restrictions on which piece may close the loop. Height, flatness, and heading near the station are handled smoothly by Φ's alignment terms rather than by hard rules.

## Observation Space

A `Dict` space (see `openrct2_gym/envs/obs_config.py`), all egocentric / rotation-invariant:

- **local_map**: 4×24×24 forward-aligned 2.5D crop — occupancy, signed height, goal+station
  marker, chain-lift trail
- **build_history_tokens/feats/mask**: the last 128 placed pieces (piece-type tokens plus
  per-piece relative geometry), encoded by a GRU
- **goal_disp / goal_direction3**: displacement and unit direction to the closing target in
  the agent's frame
- **scalars (12)**: distance, angle-to-goal, distance trend, track budget used, last ride's
  excitement/intensity/nausea, energy margin, and the closing-corridor coordinates the reward
  pays on (signed along-axis distance, off-axis distance, heading match, route progress)
- **current_direction / last_piece_type**: categorical state

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

- **Warm-start reverse curriculum**: solved the Phase-1/2 discovery problem that killed eight
  consecutive runs — the full curriculum now falls in <1M steps, reproducibly, from scratch
- **Descent placement fix**: the plugin takes a piece's *base* z but validates its *train
  entry* — every descending piece had silently failed to place in every earlier run. Drops
  (including 60° segments) are now actually part of the action space
- **P3/P4 scale phases + verified viability**: graded height/drop/length targets replace
  piece-count gates that the Phase-2 mini-loop already satisfied; Phase 4 turns the ride test
  on and pays only when the train demonstrably completes the circuit
- **Quality ramp**: the Phase-5 excitement/intensity bonus pays partial progress instead of
  being a dead band (two runs plateaued at the identical nausea-only score before this)
- **Progress-conditional exploration**: entropy floors hold exploration up exactly while a
  discovery problem is unsolved (Phase-1 completions, Phase-5 quality) and hand back to the
  proven exploit config once telemetry says the skill is learned
- **Protocol hardening**: timed-out placements sacrifice the episode (they may have been
  applied server-side), poisoned sockets are dropped, hung instances fail loudly, and the
  final model is saved even when a worker died

## Future Improvements

- Support for different coaster types
- Visual observation space using track renders
- Transfer learning between coaster types
- More sophisticated physics model (G-forces, speed estimation)
