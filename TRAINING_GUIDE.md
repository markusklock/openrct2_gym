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

Key metrics:

- `success/overall_loop_completion_rate`: completion rate across all environments.
- `curriculum/phase`: current 5-phase curriculum phase.
- `curriculum/qualified_rate`: phase 2/3 structural gate progress when available.
- `navigation/env_0_min_distance`: closest approach to the station.
- `chain_lift/history_count`: chain-lift pieces in the build history.
- `optim/ent_coef`: live entropy coefficient, including collapse-guard adjustments.
- `performance/steps_per_second`: training throughput.

## Curriculum

The maintained curriculum is `ImprovedPhasedCurriculumWrapper`:

1. Phase 1, Return Practice: 25 pieces, completion-focused.
2. Phase 2, Lift Hill Building: 40 pieces, completion with chain-lift structure.
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
