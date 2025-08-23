# OpenRCT2 RL Training Guide

## Quick Start - Recommended Approach

For the best results, use curriculum learning with action masking:

```bash
python train_curriculum_masked.py --timesteps 1000000
```

This combines:
- **Curriculum Learning**: Gradually increases difficulty from 30 to 100 piece tracks
- **Action Masking**: Prevents invalid track placements using MaskablePPO
- **Enhanced Rewards**: Strong return incentives, distance checkpoints, chain lift bonuses

## Training Scripts Comparison

### 1. `train_curriculum_masked.py` (⭐ RECOMMENDED)
**Best for:** New training from scratch
- ✅ Curriculum learning (30→100 pieces gradually)
- ✅ True action masking (MaskablePPO)
- ✅ All reward improvements
- ✅ Prevents invalid actions completely
- 📊 Logs to: `./curriculum_masked_tensorboard/`

```bash
# Standard training
python train_curriculum_masked.py --timesteps 1000000

# With adaptive curriculum (also adjusts rewards)
python train_curriculum_masked.py --timesteps 1000000 --adaptive

# Continue from checkpoint
python train_curriculum_masked.py --model-path logs_curriculum_masked/best_model.zip
```

### 2. `train_with_curriculum.py`
**Best for:** Testing curriculum without sb3-contrib dependency
- ✅ Curriculum learning
- ✅ All reward improvements
- ⚠️ Only penalizes invalid actions (doesn't prevent them)
- Uses standard PPO
- 📊 Logs to: `./curriculum_tensorboard/`

```bash
python train_with_curriculum.py --timesteps 1000000
```

### 3. `train_rl_agent_masked.py`
**Best for:** Full difficulty training with masking
- ✅ True action masking (MaskablePPO)
- ❌ No curriculum (always 250 piece max)
- ⚠️ Harder to learn initially
- 📊 Logs to: `./ppo_openrct2_tensorboard/`

```bash
python train_rl_agent_masked.py --timesteps 500000
```

### 4. `train_rl_agent_simple.py`
**Best for:** Basic training without dependencies
- ❌ No true masking (only penalties)
- ❌ No curriculum
- Simplest implementation
- 📊 Logs to: `./ppo_openrct2_tensorboard/`

```bash
python train_rl_agent_simple.py --timesteps 500000
```

### 5. `train_rl_agent_enhanced.py`
**Best for:** Detailed metrics tracking
- ✅ Comprehensive metrics
- ⚠️ No masking or curriculum
- Best for analysis
- 📊 Logs to: `./ppo_openrct2_tensorboard/`

```bash
python train_rl_agent_enhanced.py --timesteps 500000
```

## Monitoring Training

### Tensorboard

Start Tensorboard to monitor training progress:

```bash
# For curriculum + masked training
tensorboard --logdir ./curriculum_masked_tensorboard/

# For other training scripts
tensorboard --logdir ./ppo_openrct2_tensorboard/
```

### Key Metrics to Watch

1. **Success Metrics**
   - `success/loop_completion_rate` - Should increase over time (target: >20%)
   - `success/completed_track_length` - Length of successful tracks

2. **Navigation Metrics**
   - `navigation/min_distance` - Should decrease (shows agent getting closer)
   - `metrics/current_distance` - Real-time distance to station

3. **Curriculum Progress** (if using curriculum)
   - `curriculum/stage` - Current difficulty level (1-8)
   - `curriculum/max_length` - Current maximum track length
   - `curriculum/stage_success_rate` - Success rate at current stage

4. **Behavioral Metrics**
   - `behavior/remove_count` - Should be low (not exploiting)
   - `chain_lift/count` - Should be 1-3 per episode

5. **Reward Breakdown**
   - `rewards/building_total` - Rewards during building phase
   - `rewards/return_total` - Rewards during return phase
   - `ep_rew_mean` - Average episode reward (main metric)

## Training Stages

### With Curriculum Learning

1. **Stage 1** (30 pieces max)
   - Focus: Learn basic track building and chain lifts
   - Success target: 20% completion rate
   - Typical duration: 50-100k timesteps

2. **Stage 2** (40 pieces max)
   - Focus: Slightly longer tracks
   - Success target: 20% completion rate
   - Typical duration: 50-100k timesteps

3. **Stages 3-8** (50-100 pieces)
   - Gradually increasing difficulty
   - Each stage requires 20% success to advance
   - Full progression: 500k-1M timesteps

### Expected Timeline

- **First successful loop**: 50-100k timesteps (with curriculum)
- **Consistent 30-piece loops**: 100-200k timesteps
- **50+ piece loops**: 300-500k timesteps
- **100 piece loops**: 800k-1M+ timesteps

## Troubleshooting

### Agent not returning to station
- Check `navigation/min_distance` - if not decreasing, return rewards may be too weak
- Verify `current_phase` transitions from building → return
- Consider using adaptive curriculum for dynamic reward scaling

### Too many collisions
- Check if action masking is working (`collision_count` should be low with masking)
- Ensure OpenRCT2 API is responding correctly
- Verify valid_action_mask is being called

### Slow training
- Reduce batch_size if using CPU (default: 64)
- Ensure OpenRCT2 is running in headless mode
- Check API response times

### Curriculum not advancing
- Current stage needs 20% success rate over 50 episodes
- Check `curriculum/stage_success_rate` in Tensorboard
- May need to lower success_threshold in curriculum wrapper

## Best Practices

1. **Start with curriculum + masking** for fastest learning
2. **Monitor min_distance_reached** - key indicator of navigation learning
3. **Save checkpoints frequently** (every 10k steps)
4. **Train for at least 1M timesteps** for good results
5. **Use adaptive curriculum** if standard curriculum is too easy/hard

## Advanced Options

### Custom Curriculum Settings

Edit the curriculum wrapper initialization in training script:

```python
env = CurriculumWrapper(
    base_env,
    initial_max_length=20,      # Start even easier
    target_max_length=150,      # Go even longer
    success_threshold=0.15,     # Lower threshold (easier progression)
    window_size=30,            # Fewer episodes to judge success
    increase_step=5            # Smaller difficulty jumps
)
```

### Hyperparameter Tuning

Key parameters in MaskablePPO initialization:

```python
model = MaskablePPO(
    learning_rate=3e-4,        # Lower = more stable, Higher = faster
    n_steps=2048,             # Steps before update
    batch_size=64,            # Batch size for updates
    n_epochs=10,              # Training epochs per update
    ent_coef=0.01,           # Exploration (higher = more random)
    clip_range=0.2,          # PPO clipping
)
```