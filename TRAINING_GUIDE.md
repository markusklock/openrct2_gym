# OpenRCT2 RL Training Guide

## Quick Start

Use the single maintained training entry point:

```bash
python train.py --ports 8080 --timesteps 1000000
```

For multiple OpenRCT2 API servers, pass a comma-separated port list:

```bash
python train.py --ports 8080,8081,8082,8083 --timesteps 1000000
```

`train.py` always uses the improved 5-phase curriculum, MaskablePPO action masking, the custom build-history feature extractor, and VecNormalize over scalar observations. It uses `DummyVecEnv` for one available port and automatically switches to `SubprocVecEnv` when two or more available ports are provided.

## Warm-Start Reverse Curriculum

Phase 1's completion signal is a discovery problem (a minimal loop is 12 exact pieces), so
episodes in phases 1-2 are scaffolded by default: at reset the env pre-places a prefix of a
verified completable loop and the agent builds only the last `k` pieces. `k` anneals upward
on success (backward chaining) until episodes are cold (unscaffolded). Phase gates count
**cold episodes only** — scaffolded wins can never advance a phase.

Seed the loop library once per map with a free instance:

```bash
python build_loop_library.py --port 8080          # flat loops (Phase 1)
python build_loop_library.py --port 8080 --hill   # chain-hill loops (Phase 2 pool)
```

Every training completion is also harvested back into `logs/loop_library.jsonl` (dedup'd),
so agent-discovered loops become future scaffolds. The library survives fresh runs (verified
geometry is regime-independent, unlike the Φ closing calibration, which is still cleared).

Flags: `--no-warm-start` (cold starts only), `--loop-library PATH`, `--p-cold F` (base cold
fraction, default 0.25; rises automatically as the anneal progresses).

## Common Options

```bash
# Continue from a checkpoint
python train.py --ports 8080 --model-path logs_parallel_curriculum_masked_1envs/final_model.zip

# Disable intermediate evaluation for maximum throughput
python train.py --ports 8080,8081 --disable-eval

# Change evaluation cadence
python train.py --ports 8080 --eval-freq 50000 --eval-episodes 5

# Increase environment/API logging for a single server
python train.py --ports 8080 --verbose 1
```

Checkpoints are written under `logs_parallel_curriculum_masked_<N>envs/`. Each model checkpoint has a matching `*_vecnormalize.pkl` file that must be kept with the model for resume or inference.

## Monitoring

Start TensorBoard with:

```bash
tensorboard --logdir ./parallel_curriculum_masked_tensorboard/
```

Key metrics (with warm starts active, `success/overall_loop_completion_rate` and
`ep_rew/ep_len` are scaffold-mixed — read the `cold_*` tags for true-task progress):

- `success/cold_completion_rate`: completion rate over COLD episodes (the number that matters).
- `curriculum/cold_success_rate` / `curriculum/scaffold_success_rate`: per-worker gate windows.
- `curriculum/warm_k_max`: annealer frontier (suffix length the agent must build).
- `curriculum/cold_fraction`: realized fraction of unscaffolded episodes.
- `curriculum/loop_library_size`: verified loops in the shared pool.
- `optim/ent_floor_mode`: 1 while the Phase-1 bootstrap entropy floor is held, 0 after cold
  completions flow (≥2%) and the proven exploit config is restored.
- `rewards/route_potential`: episode-end value of the west-side route-shaping term.
- `success/overall_loop_completion_rate`: completion rate across all environments (scaffold-mixed).
- `curriculum/phase`: current 5-phase curriculum phase.
- `curriculum/qualified_rate`: current phase 2/3 structural gate progress when available.
- `curriculum/phase2_stage`: Phase 2 sub-stage (`1`: one-chain roundtrip, `2`: one-chain completion, `3`: three-chain completion).
- `curriculum/phase2_roundtrip_rate`, `curriculum/phase2_chain*_completion_rate`: rolling Phase 2 bridge diagnostics.
- `navigation/env_0_min_distance`: closest approach to the station.
- `chain_lift/history_count`: chain-lift pieces in the build history.
- `optim/ent_coef`: live entropy coefficient, including collapse-guard adjustments.
- `performance/steps_per_second`: training throughput.

## Curriculum

The maintained curriculum is `ImprovedPhasedCurriculumWrapper`:

1. Phase 1, Return Practice: 40 pieces, completion-focused.
2. Phase 2, Lift Hill Building: 40 pieces, staged bridge from one-chain roundtrip to one-chain completion to three-chain completion.
3. Phase 3, Drop & Turn: 60 pieces, completion with lift/drop structure.
4. Phase 4, Circuit Mastery: 80 pieces, integrated completion.
5. Phase 5, Quality Optimization: 80-120 pieces, ride testing and quality bonus.

The reward function stays unified across phases. The curriculum changes track-length limits, ride-testing settings, and reward parameters; it does not swap in a separate legacy reward method.

## Troubleshooting

- If no ports are available, confirm the OpenRCT2 API plugin is running and reachable on the specified ports.
- If only one of several ports is available, training continues with that port and uses `DummyVecEnv`.
- If training throughput is low, run multiple OpenRCT2 instances on separate ports and pass all ports through `--ports`.
- If resume behavior looks degraded, verify the model and its sibling `*_vecnormalize.pkl` file are both present.
- If completion stalls after early progress, inspect `optim/ent_coef`, `train/entropy_loss`, and `curriculum/phase` to see whether the entropy-collapse guard or phase-2 KL guard is active.

## Running A Model

Use `run_model.py` with the saved model and matching VecNormalize stats:

```bash
python run_model.py --model logs_parallel_curriculum_masked_1envs/final_model --port 8080
```

`run_model.py` rebuilds the same improved curriculum and action-masking wrapper chain used during training.
