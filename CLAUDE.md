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
- Phase 2: Lift Hill Building (40 pieces) - Learn chain lifts and energy
- Phase 3: Drop & Turn (60 pieces) - Learn drops and turnarounds
- Phase 4: Circuit Mastery (80 pieces) - Full integration
- Phase 5: Quality Optimization (80-120 pieces) - Optimize ride ratings

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
   - **Goal = the station dock; deterministic, dock-coupled closing heading**: `goal_position` is the
     station's dock endpoint itself (verified by `verify_goal_position.py` — the API closes the circuit
     AT `station_start`, not one tile east as the old guide assumed). The station is built with
     `startDir=0`, so every circuit re-enters BeginStation heading North; Φ is handed this closing
     heading (`_STATION_ENTRY_DIR`) from step 1, and the heading reward is **coupled to the dock**
     (gated by the near-closure factor) so the agent is free to turn while routing and only must align
     as it docks. Deterministic heading is critical for Phase-1 bootstrap — otherwise the heading term
     stays off until a completion calibrates it (a chicken-and-egg that stalled the cold start). The
     full closing geometry is still refined from real completions (anchor locked only after ≥3 agree).
   - **Phase-2 hill-discovery bootstrap**: the climb-and-return milestone is made *discoverable* by
     annealing `roundtrip_gain` (1→1→4 z across sub-stages 2.1/2.2/2.3) plus a small one-time summit
     breadcrumb (`R_summit` 120→60→0), with a raised early-Phase-2 entropy floor
     (`PHASE2_EARLY_ENT_COEF=0.018` in `train.py`) so exploration survives long enough to find the climb.
   - Ride quality optimization in Phase 5 (Excitement 7-9, Intensity 4.5-6.5, Nausea <4.5)

3. **5-Phase Curriculum Learning**:
   - Phases focus on specific skills before combining them
   - Trusts API's `isCircuitComplete` flag (no artificial restrictions)
   - Progressive track length limits (40 → 40 → 60 → 80 → 120; Phase 1 raised 25→40 to give the agent
     room to route a loop back to the station before truncation)

4. **Action Space**: 32 discrete actions (30 track pieces + remove + flat), with action masking to prevent invalid placements

## Important Notes

- The OpenRCT2 game must have the API plugin installed and running (default port 8080)
- Training logs and models are saved to `logs/` and `ppo_openrct2_tensorboard/`
- The environment expects the API server to be running before starting training
- Auto-backtracking helps agents recover from bad decisions during training