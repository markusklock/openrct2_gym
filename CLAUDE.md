# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac

# Install package in development mode with all dependencies
pip install -e ".[train,dev]"
# Or install from requirements.txt
pip install -r requirements.txt
```

### Running Tests
```bash
# Run tests with pytest
pytest openrct2_gym/tests/

# Run a specific test file
pytest openrct2_gym/tests/test_track_builder.py
```

### Training the RL Agent
```bash
# Parallel training (recommended): one OpenRCT2 instance per port.
# train.py is the single consolidated script and ALWAYS uses the 5-phase curriculum
# + potential-based reward (there is no --improved/--phased flag).
python train.py --ports 8080,8081,8082,8083 --timesteps 2000000 --disable-eval

# Single environment
python train.py --ports 8080 --timesteps 1000000 --disable-eval

# Resume from a checkpoint (keeps its VecNormalize stats + closing calibration)
python train.py --ports 8080,8081,8082,8083 --model-path logs_<run>/<ckpt>_steps.zip
```

### Running Trained Models
```bash
# Run a trained model
python run_model.py
```

## Architecture Overview

This is a Gymnasium environment for training RL agents to build roller coasters in OpenRCT2. The project has evolved from UI automation to API-based control.

### Core Components

**openrct2_gym/envs/openrct2_env.py**: Main Gymnasium environment that manages the RL training loop. Key features:
- Discrete action space (32 actions: 30 track pieces + remove action)
- Complex observation space tracking position, direction, track pieces, and distance to goal
- Auto-backtracking mechanism for handling consecutive placement failures
- Physics-aware energy estimation (chain lifts add energy, drops convert to speed)
- Pattern detection (lift hills, drops, turnarounds)
- Soft approach guidance for station connection

**openrct2_gym/envs/api_controller.py**: Handles communication with OpenRCT2 game via HTTP API
- Connects to OpenRCT2 plugin server (default port 8080)
- Manages track placement, removal, and circuit completion
- Retrieves ride statistics for reward calculation
- Retry logic with exponential backoff for reliability

**openrct2_gym/envs/improved_phased_curriculum_wrapper.py**: 5-phase curriculum learning
- Phase 1: Return Practice (40 pieces) - Learn navigation
- Phase 2: Lift Hill Building (40 pieces) - Learn chain lifts and energy (staged 2.1/2.2/2.3)
- Phase 3: Real Drops & Scale (60 pieces) - chain height >=4z, drops >=4z, length >=25,
  energy-viable at completion (graded structure credit, not piece counting)
- Phase 4: Big & Verified (80 pieces) - height >=6z, drops >=8z incl. a 60-degree segment,
  length >=40; ride testing ON, R_viable=150 paid only when the test returns real stats
- Phase 5: Quality Optimization (80-120 pieces) - ramp+band quality bonus (every increment
  toward E8/I5.5 pays), no step cost, P5 exploration floor while median excitement < 4

**openrct2_gym/envs/api_track_builder.py**: Manages track construction logic
- Translates discrete actions to API calls
- Maintains track history for backtracking
- Handles position and direction updates based on track piece geometry

**openrct2_gym/envs/action_masking.py**: Implements action filtering to prevent invalid moves
- SimpleActionMasker: Basic rule-based action filtering
- SmartActionSampler: Intelligent action sampling based on current state

### Training Scripts

- **train.py**: The single consolidated training script (the legacy `train_parallel_curriculum_masked.py`
  and `train_rl_agent_*.py` were merged into it). Always uses the 5-phase physics-aware curriculum with
  the unified potential-based reward and MaskablePPO action masking.
  - `--ports` (comma-separated): one OpenRCT2 instance per port; `SubprocVecEnv` for 2+ ports, else `DummyVecEnv`
  - other flags: `--timesteps`, `--disable-eval`, `--model-path` (resume), `--target-rollout`, `--checkpoint-freq`, `--eval-freq`
  - warm-start flags: `--no-warm-start` (disable the reverse curriculum), `--loop-library PATH`
    (default `logs/loop_library.jsonl`), `--p-cold F` (base cold-episode fraction, default 0.25)
- **build_loop_library.py**: seeds the warm-start loop library with live-verified closable loops
  (run once per map: `python build_loop_library.py --port 8080` and `--hill` for the Phase-2 pool).
- **run_model.py**: Run a trained model.

### Key Design Decisions

1. **API Integration**: The environment uses OpenRCT2's scripting API with:
   - Retry logic with exponential backoff (0.5s, 1s, 2s)
   - Proper resource cleanup (no file descriptor leaks)
   - DummyVecEnv for stable parallel training (no subprocess sync issues)

2. **Physics-Aware Reward Structure** (always on; potential-based / PBRS):
   - Energy estimation: chain lifts add energy, drops convert to speed
   - Pattern detection: rewards for lift hills, drops, turnarounds
   - Soft approach guidance: bonuses for correct height/direction near station
   - **Goal = the STAGING tile [62,66,14] (one tile east of the dock); deterministic, dock-coupled
     closing heading**: the head can never sit ON the dock tile (occupied by BeginStation), so
     `goal_position` is the staging tile the closing piece is placed FROM (openrct2_env.py reset();
     probe_corridor.py confirmed 7+ pieces dock from there; a goal-=-dock experiment collapsed
     completion to 0% and was reverted). The station is built with `startDir=0` (dir 0 = West,
     vector (-1,0)), so every circuit re-enters BeginStation heading dir 0; Φ is handed this closing
     heading (`_STATION_ENTRY_DIR`) from step 1, and the heading reward is **coupled to the dock**
     (gated by the near-closure factor) so the agent is free to turn while routing and only must align
     as it docks. The full closing geometry is still refined from real completions (anchor locked
     only after ≥3 agree).
   - **Route potential (`w_route`, on in phases 1-4)**: the directional approach cone is deliberately
     zero on the whole start/west side, which left the detour around the station unshaped (the Jun-24
     run parked ~5 tiles out forever). `_route_progress()` pays bounded angular progress of the head's
     bearing around the station center — monotone along BOTH detours, PBRS-clean, diagnosed via
     `rewards/route_potential`.
   - **Phase-2 hill-discovery bootstrap**: the climb-and-return milestone is made *discoverable* by
     annealing `roundtrip_gain` (1→1→3 z across sub-stages 2.1/2.2/2.3; the bar is CHAIN-banked gain —
     the canonical hill [10,9,13] banks 3, its crest piece isn't chained) plus a small one-time summit
     breadcrumb (`R_summit` 40→30→0), with a raised early-Phase-2 entropy floor
     (`PHASE2_EARLY_ENT_COEF=0.018` in `train.py`) so exploration survives long enough to find the climb.
     The completion hill gate scales chain credit by chain-banked ELEVATION against that bar, so
     chain-stub decoration on a flat loop cannot pay as a full hill.
   - Ride quality optimization in Phase 5 (Excitement 7-9, Intensity 4.5-6.5, Nausea <4.5)

3. **5-Phase Curriculum Learning**:
   - Phases focus on specific skills before combining them
   - Trusts API's `isCircuitComplete` flag (no artificial restrictions)
   - Progressive track length limits (40 → 40 → 60 → 80 → 120; Phase 1 raised 25→40 to give the agent
     room to route a loop back to the station before truncation)
   - **Warm-start reverse curriculum (phases 1-2)**: completion is a discovery problem (a minimal
     loop is 12 exact pieces; the Jun-24 run saw 7 completions in 31k episodes and entropy-collapsed).
     At reset the env replays a prefix of a verified loop from `logs/loop_library.jsonl`
     (`openrct2_gym/envs/warm_start.py`: `LoopLibrary` + per-worker `WarmStartAnnealer`); the agent
     builds the last k pieces, k anneals up on frontier success, and every completion is harvested
     back into the library. **Phase gates count cold (unscaffolded) episodes only** — read
     `success/cold_completion_rate` and `curriculum/warm_k_max` in TB, not the scaffold-mixed
     `overall_loop_completion_rate`. A progress-conditional entropy floor (`optim/ent_floor_mode`)
     holds ent_coef at 0.025 with a raised collapse band until cold completions flow (≥2%), then
     restores the proven 0.01 config.

4. **Action Space**: 32 discrete actions (30 track pieces + remove + flat), with action masking to prevent invalid placements
   - **Descending pieces are placed at BASE z** (`api_track_builder.py descent_entry_z_offset`,
     live-probed): the plugin validates a piece's train ENTRY against the previous end but takes the
     base z, which for descents sits below the entry by the piece's drop (25°=2, 60°=8, flat↔25=1,
     25↔60=4). Before this offset every descent placement failed silently — drops were effectively
     removed from the action space in all earlier runs.

## Important Notes

- The OpenRCT2 game must have the API plugin installed and running (default port 8080)
- Training logs and models are saved to `logs/` and `ppo_openrct2_tensorboard/`
- The environment expects the API server to be running before starting training
- Auto-backtracking helps agents recover from bad decisions during training