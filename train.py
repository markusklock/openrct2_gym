#!/usr/bin/env python3
"""
Parallel training script with curriculum learning AND proper action masking using MaskablePPO
Trains on multiple OpenRCT2 instances simultaneously for faster learning
"""
# Thread caps MUST be set before NumPy/Torch are imported (they are pulled in transitively by
# the openrct2_gym / sb3 imports below). Without this, a 20-worker SubprocVecEnv run on one host
# spawns 20 competing BLAS/OMP thread pools and oversubscribes the cores. Forked workers inherit
# these; the main process's PPO-update thread count is set explicitly in train().
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import gymnasium as gym
import openrct2_gym
from openrct2_gym.envs.improved_phased_curriculum_wrapper import ImprovedPhasedCurriculumWrapper
from openrct2_gym.envs.wrappers import OpenRCT2Wrapper
from openrct2_gym.envs.feature_extractor import BuildHistoryExtractor
from openrct2_gym.envs.openrct2_env import RewardParams, OpenRCT2Env
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
import numpy as np
import torch
import argparse
import time
import math
from collections import deque
from typing import Callable, List, Optional
from contextlib import ExitStack

from openrct2_gym.envs.warm_start import LoopLibrary

# Single source of truth for the discount factor. PBRS policy-invariance requires the
# shaping potential to discount with the SAME gamma as PPO, so the reward's
# RewardParams.gamma and the model's gamma are tied to this one constant.
GAMMA = RewardParams().gamma

# Phase-conditional optimizer settings. Phase 1's bootstrap depends on aggressively
# exploiting rare +1000 completion spikes -- a global target_kl throttled exactly those
# updates and froze phase 1 (17 completions in 38k episodes, never amplified). Both runs
# that learned phase 1 used the unguarded config below. The guard (target_kl caps the
# per-rollout update; a live run hit approx_kl=2.49 at the phase-2 transition and the
# policy never recovered) plus extra entropy (for the multi-piece hill motif) are armed
# by ParallelCurriculumMaskableCallback when the curriculum reaches phase 2.
OPT_PHASE1 = dict(target_kl=None, ent_coef=0.01)    # proven phase-1 bootstrap config
# Phases >= 2: arm the KL guard + a modest entropy FLOOR. ent_coef=0.02 over-explored and
# exploded entropy (completion destroyed); 0.01 under-explored and imploded (entropy -> 0.02,
# policy froze on a non-completing near-miss, never sampling chain lifts). 0.015 sits between
# to keep chain lifts sampled without the explosion. (A proper adaptive entropy controller is
# the principled fix if this floor proves insufficient.)
OPT_GUARDED = dict(target_kl=0.04, ent_coef=0.015)
# Early-Phase-2 entropy floor: the chain-hill discovery (2.1) AND integration (2.2) stages need
# MORE exploration than the guarded base (0.015 under-explores -- it never samples, then abandons,
# the chain), but a permanent 0.02 floor already "over-explored and exploded entropy (completion
# destroyed)" (the OPT_GUARDED note above). 0.018 sits just under that redline; it is the resting
# floor while phase==2 and phase2_stage in (1, 2), dropping back to OPT_GUARDED at stage 2.3. Watch
# optim/ent_coef vs the completion rate and back toward 0.015 if closure degrades.
PHASE2_EARLY_ENT_COEF = 0.018

# Adaptive entropy-collapse guard (the principled fix the OPT_GUARDED note foreshadowed).
# A run silently dies when entropy bleeds to ~0: the softmax saturates, KL->0, gradients
# vanish, and the deterministic policy then collides off the station for all 256 steps,
# building nothing. A saturated softmax can't be revived by the entropy bonus alone, so the
# guard re-injects exploration BEFORE saturation -- the moment entropy crosses ENT_COLLAPSE_LO
# -- and backs off once it recovers past ENT_COLLAPSE_HI (hysteresis), restoring the phase
# base so the policy can still sharpen into completions. Tuned to this run's curves: productive
# entropy stayed >=0.17 (dormant zone); the fatal collapse fell through 0.12 -> 0.0003. The HI
# ceiling self-limits the boost so it can't "explode" entropy the way a permanent high ent_coef did.
ENT_COLLAPSE_LO = 0.12      # mean policy entropy (nats) below this => collapsing; boost
ENT_COLLAPSE_HI = 0.30      # once recovered above this => (eventually) restore the phase-base ent_coef
ENT_COLLAPSE_BOOST = 0.03   # temporary ent_coef while boosted. 0.03 not 0.05: 0.05 over-corrected
                            # -- it spiked approx_kl (~0.15) and cratered the closure skill (ep_rew
                            # 227->159) every time it fired, a destructive limit cycle.
# Stabilizers on top of the LO/HI hysteresis band, to stop the boost<->relax thrash:
ENT_BOOST_MIN_HOLD = 3      # min updates to HOLD the boost before any relax (rides out the overshoot)

# Progress-conditional Phase-1 entropy floor ("bootstrap" mode). The Jun-24 run proved the
# 0.01 floor + 0.12/0.30 band is a DISCOVERY killer: entropy sat at ~0.2 nats from 130k steps
# and the ~12-piece docking sequence was never sampled again (7 completions in 31k episodes,
# all before the collapse). That config is right for EXPLOITING completions once they flow --
# so it is conditional: while phase 1 has ~zero completions, hold a higher floor and a raised
# collapse band; hand back to the proven OPT_PHASE1 once completions flow. Keyed on the
# ANY-episode completion rate (scaffolded included): the floor is anti-collapse INSURANCE,
# and with warm starts active scaffolded +1000s flow by construction -- holding 0.025 then
# CAPS the sharpening the scaffold exists to teach (smoke run 2 plateaued at ~15%, the
# random-choice baseline, under the cold-keyed floor). Cold rate remains the PHASE-GATE
# metric; if completions of every kind die, the floor re-arms.
PHASE1_BOOTSTRAP_ENT_COEF = 0.025   # resting floor while bootstrapping (vs 0.01 proven-exploit)
COMPLETION_RATE_EXIT = 0.02         # any-episode completion rate that ends bootstrap mode
COMPLETION_RATE_ENTER = 0.005       # relapse threshold that re-enters it
COMPLETION_RATE_WINDOW = 400        # episodes kept (pooled across all envs)
COMPLETION_RATE_MIN_SAMPLES = 50    # min episodes before the rate is trusted
COLD_RATE_WINDOW = 400              # cold episodes kept for the success/cold_completion_rate tag
BOOT_ENT_LO = 0.30                  # bootstrap collapse band: the Jun-24 "dormant zone" (~0.2)
BOOT_ENT_HI = 0.55                  # is itself a freeze while discovering -> band sits above it
BOOT_ENT_BOOST = 0.04               # must exceed the 0.025 floor to be a boost at all
ENT_MODE_MIN_HOLD = 10              # min updates between mode flips (anti-thrash)

# Phase-5 quality exploration floor (the plateau fix's second half). Two from-scratch runs
# converged onto a ~0.2-nat mini-loop policy earning the identical +90/500 nausea-only
# quality bonus: even with the reward reshaped into a ramp, a policy with no entropy left
# cannot DISCOVER the bigger coasters the ramp now pays for. While the fleet is in phase 5
# and the median tested excitement sits below the floor target, the resting ent_coef is
# raised and the collapse guard uses the raised (bootstrap) band; once excitement clears
# the bar, the proven exploit config returns.
P5_QUALITY_ENT_COEF = 0.02
QUALITY_EXC_TARGET_FLOOR = 4.0      # median excitement that releases the quality floor
QUALITY_WINDOW = 200                # tested-episode window (pooled across envs)
QUALITY_MIN_SAMPLES = 30            # min tested episodes before the median is trusted

# Fixed PPO hyperparameters, module-level so tests can pin them (n_steps/batch_size are
# computed per run). Starts in the phase-1 config; the callback arms OPT_GUARDED later.
PPO_HYPERPARAMS = dict(
    learning_rate=3e-4,
    n_epochs=10,
    gamma=GAMMA,
    gae_lambda=0.95,
    clip_range=0.2,
    **OPT_PHASE1,
)


def _clear_calibration_cache() -> bool:
    """Drop any persisted closing-geometry calibration (in-memory class cache + the JSON
    file shared across SubprocVecEnv workers) so a fresh run recalibrates Phi's closing
    anchor from its OWN reproducible completions. A cache left over from an earlier reward
    regime silently misguides Phi in every later run. Returns True if a file was removed."""
    OpenRCT2Env._close_cache = None
    OpenRCT2Env._close_records = []   # drop accumulated completion records too (fresh run)
    try:
        os.remove(OpenRCT2Env._CLOSE_CACHE_PATH)
        return True
    except OSError:
        return False


def _vecnormalize_path(model_path: str) -> str:
    """Sibling path for a model's VecNormalize stats (``X.zip`` -> ``X_vecnormalize.pkl``)."""
    if model_path.endswith(".zip"):
        model_path = model_path[:-4]
    return model_path + "_vecnormalize.pkl"


def _resolve_model_path(model_path: Optional[str]) -> Optional[str]:
    """Validate --model-path BEFORE anything destructive. A typo'd path must not silently
    fall through to training-from-scratch: that wipes the closing calibration and pairs the
    OLD run's VecNormalize stats with a brand-new policy -- a whole-run waste with a single
    'Creating new model' line as the only symptom. Accepts the extension-less form SB3's
    load tolerates, by resolving to the sibling .zip."""
    if not model_path:
        return None
    if os.path.exists(model_path):
        return model_path
    if os.path.exists(model_path + ".zip"):
        return model_path + ".zip"
    raise SystemExit(f"❌ --model-path {model_path} not found (nor {model_path}.zip); "
                     f"refusing to silently train from scratch on a resume request")


def _unwrap_to_vecenv_with_envs(env):
    """Return the underlying vec env exposing ``.envs`` (e.g. DummyVecEnv), or None.

    VecNormalize wraps the vec env in ``.venv``; access ``.envs`` explicitly rather than
    relying on attribute forwarding. SubprocVecEnv has no ``.envs`` -> returns None.
    """
    base = env
    for _ in range(8):
        if base is None:
            break
        if hasattr(base, "envs"):
            return base
        base = getattr(base, "venv", None)
    return None


class SaveVecNormalizeCallback(BaseCallback):
    """Persist VecNormalize running stats next to each model checkpoint.

    Without matching stats a saved checkpoint cannot be correctly reloaded for eval or
    resume (the obs normalization would be wrong). Filenames mirror CheckpointCallback's.
    """

    def __init__(self, save_freq: int, save_path: str, name_prefix: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        if self.save_freq > 0 and self.n_calls % self.save_freq == 0:
            vec_env = self.model.get_vec_normalize_env()
            if vec_env is not None:
                path = os.path.join(
                    self.save_path,
                    f"{self.name_prefix}_{self.num_timesteps}_steps_vecnormalize.pkl",
                )
                vec_env.save(path)
        return True


class ParallelCurriculumMaskableCallback(BaseCallback):
    """
    Tensorboard callback that tracks both curriculum and masking metrics
    across multiple parallel environments
    """
    def __init__(self, n_envs=1, verbose=0, training_verbose=0):
        super().__init__(verbose)
        self.n_envs = n_envs
        self.training_verbose = training_verbose  # Store the training verbosity level
        self.episode_counts = [0] * n_envs
        self.loop_completed_counts = [0] * n_envs
        self.total_episode_count = 0
        self.total_loop_completed = 0
        self.invalid_action_count = 0
        self.total_actions = 0
        self.start_time = time.time()
        self.total_steps = 0
        self.last_dashboard_episode = 0  # Track last dashboard print to avoid repeats
        self._opt_guarded = False  # becomes True once OPT_GUARDED is armed (phase >= 2)
        self._ent_boosted = False  # True while the entropy-collapse guard holds ent_coef boosted
        self._ent_boost_calls = 0  # updates elapsed since the current boost armed (min-hold counter)
        # Per-worker curricula advance INDEPENDENTLY (each wrapper has its own gates), so the
        # fleet straddles phase boundaries. _phase/_phase2_stage are the FLEET view derived in
        # _note_env_phase (lower-median phase; min stage among phase-2 workers) -- last-writer-
        # wins made the KL guard and entropy floor functions of episode-finish ordering, and
        # armed target_kl on the FIRST worker to touch phase 2 while the rest still needed the
        # unguarded phase-1 bootstrap.
        self._env_phase = {i: 1 for i in range(n_envs)}   # every worker starts in phase 1
        self._env_stage = {}
        self._phase = 1            # fleet (lower-median) curriculum phase
        self._phase2_stage = None  # min phase-2 sub-stage among workers currently in phase 2
        # Progress-conditional Phase-1 entropy floor: starts in bootstrap mode (completion
        # rate is 0 by definition at the start of a run); exits once completions flow
        # (scaffolded count -- the floor is anti-collapse insurance, not a gate).
        self._ent_mode = "bootstrap"
        self._completion_window = deque(maxlen=COMPLETION_RATE_WINDOW)  # all episodes, all envs
        self._cold_window = deque(maxlen=COLD_RATE_WINDOW)  # cold episodes (TB tag + gates readout)
        self._ent_mode_calls = 0       # _maybe_update_ent_mode invocations (one per rollout end)
        self._ent_mode_last_flip = 0   # call index of the last mode flip (min-hold anchor)
        # Phase-4+ ride-test telemetry: tested-episode excitement (drives the P5 quality
        # floor) and test-success outcomes, pooled across envs.
        self._exc_window = deque(maxlen=QUALITY_WINDOW)
        self._test_window = deque(maxlen=QUALITY_WINDOW)

    def _note_env_phase(self, env_idx, info):
        """Record one worker's curriculum phase/stage and refresh the fleet view."""
        phase = info.get('learning_phase')
        if phase is not None:
            self._env_phase[env_idx] = phase
        stage = info.get('phase2_stage')
        if stage is not None:
            self._env_stage[env_idx] = stage
        phases = sorted(self._env_phase.values())
        self._phase = phases[(len(phases) - 1) // 2]   # lower median: majority-conservative
        stages = [s for i, s in self._env_stage.items() if self._env_phase.get(i) == 2]
        if stages:
            self._phase2_stage = min(stages)           # earliest stage keeps the raised floor

    def _maybe_arm_kl_guard(self, info):
        """Arm the guarded optimizer config (target_kl + raised ent_coef) once the FLEET
        majority reaches phase 2. One-way: phase 1 needs the unguarded bootstrap config to
        amplify rare completion spikes -- arming on the FIRST worker to touch phase 2 would
        throttle every other worker's bootstrap (the guard exists to survive the
        phase-transition reward shift, which only matters from phase 2 onward)."""
        if self._opt_guarded:
            return
        if self._phase < 2:
            return
        self.model.target_kl = OPT_GUARDED['target_kl']
        self._opt_guarded = True   # _phase_base_ent_coef() now returns the guarded floor
        # Raise the live ent_coef to the guarded base -- unless the collapse guard is actively
        # boosting, in which case the boost survives and recovery restores to this new base.
        if not self._ent_boosted:
            self.model.ent_coef = self._phase_base_ent_coef()
        print(f"🛡️ Phase 2 reached: armed guard {OPT_GUARDED}")

    def _median_excitement(self):
        """Median excitement over pooled TESTED episodes, or None until enough samples."""
        if len(self._exc_window) < QUALITY_MIN_SAMPLES:
            return None
        return float(np.median(self._exc_window))

    def _p5_quality_boost_active(self):
        """Whether the Phase-5 exploration floor is held: fleet in phase 5 and median tested
        excitement below the floor target (or unknown -- entering P5 holds exploration until
        the telemetry says quality is climbing)."""
        if self._phase < 5:
            return False
        med = self._median_excitement()
        return med is None or med < QUALITY_EXC_TARGET_FLOOR

    def _phase_base_ent_coef(self):
        """The ent_coef the entropy guard restores to: the raised bootstrap floor while phase 1
        still has ~zero cold completions, the proven phase-1 value once they flow, then the
        guarded floor after the KL guard arms -- raised through the early Phase-2 chain stages
        and again in Phase 5 while quality is still below the floor target."""
        if not self._opt_guarded:
            if self._phase == 1 and self._ent_mode == "bootstrap":
                return PHASE1_BOOTSTRAP_ENT_COEF
            return OPT_PHASE1['ent_coef']
        if self._phase >= 5 and self._p5_quality_boost_active():
            return P5_QUALITY_ENT_COEF
        if self._phase == 2 and self._phase2_stage in (1, 2):
            return PHASE2_EARLY_ENT_COEF
        return OPT_GUARDED['ent_coef']

    def _ent_band(self):
        """(lo, hi, boost) for the collapse guard: the raised band while DISCOVERING (phase-1
        bootstrap, or phase-5 quality still below its floor -- the legacy 'dormant zone'
        ~0.2 nats is itself a freeze in both), the proven band while exploiting."""
        if (not self._opt_guarded) and self._phase == 1 and self._ent_mode == "bootstrap":
            return BOOT_ENT_LO, BOOT_ENT_HI, BOOT_ENT_BOOST
        if self._phase >= 5 and self._p5_quality_boost_active():
            return BOOT_ENT_LO, BOOT_ENT_HI, BOOT_ENT_BOOST
        return ENT_COLLAPSE_LO, ENT_COLLAPSE_HI, ENT_COLLAPSE_BOOST

    def _cold_completion_rate(self):
        """Completion rate over pooled COLD episodes (the phase-gate readout), or None."""
        if len(self._cold_window) < COMPLETION_RATE_MIN_SAMPLES:
            return None
        return sum(self._cold_window) / len(self._cold_window)

    def _completion_rate(self):
        """Completion rate over ALL pooled episodes (scaffolded included) -- what the
        entropy floor keys on -- or None until enough samples exist."""
        if len(self._completion_window) < COMPLETION_RATE_MIN_SAMPLES:
            return None
        return sum(self._completion_window) / len(self._completion_window)

    def _maybe_update_ent_mode(self):
        """bootstrap <-> normal mode machine, one call per rollout end. Exit bootstrap when
        the phase advances or completions flow at all (rate >= COMPLETION_RATE_EXIT,
        scaffolded included); re-enter on a phase-1 relapse (rate < COMPLETION_RATE_ENTER).
        Flips respect ENT_MODE_MIN_HOLD. Pure decision logic (no logger access) so it is
        server-free testable."""
        self._ent_mode_calls += 1
        if self._ent_mode_calls - self._ent_mode_last_flip < ENT_MODE_MIN_HOLD:
            return
        rate = self._completion_rate()
        if self._ent_mode == "bootstrap":
            if self._phase > 1 or (rate is not None and rate >= COMPLETION_RATE_EXIT):
                self._ent_mode = "normal"
                self._ent_mode_last_flip = self._ent_mode_calls
                print(f"🧊 Phase-1 bootstrap floor released (completion rate "
                      f"{-1.0 if rate is None else rate:.3f}); restoring proven entropy config")
        else:
            if self._phase == 1 and rate is not None and rate < COMPLETION_RATE_ENTER:
                self._ent_mode = "bootstrap"
                self._ent_mode_last_flip = self._ent_mode_calls
                print(f"🧊 Completions relapsed to {rate:.3f}: re-arming the "
                      f"Phase-1 bootstrap entropy floor")

    def _rebaseline_ent_coef(self):
        """Snap the resting ent_coef to the current phase base when NOT actively boosted, so a stage
        advance (2.1 -> 2.2) lowers the floor on the next rollout. The collapse-guard owns ent_coef
        while boosted, so a live boost is left untouched."""
        if not self._ent_boosted:
            self.model.ent_coef = self._phase_base_ent_coef()

    def _maybe_guard_entropy_collapse(self, entropy, kl=None):
        """Hysteresis controller that prevents the silent entropy-collapse freeze WITHOUT the
        destructive boost<->relax limit cycle the bare LO/HI band produced.

        ``entropy`` is the mean policy entropy (nats) from the last update (None before the first
        train()); ``kl`` is the last update's approx_kl (None to skip the KL check). Boost ent_coef
        when entropy falls below ENT_COLLAPSE_LO (re-inject exploration before the softmax saturates).
        Relax back to the phase base only once ALL hold conditions are met: entropy recovered past
        ENT_COLLAPSE_HI, the boost has been held >= ENT_BOOST_MIN_HOLD updates (rides out the
        overshoot instead of relaxing on a single recovery sample), and approx_kl is below target_kl
        (don't hand control back mid-explosion). Pure decision logic (args, no logger access) so it
        is server-free testable."""
        if entropy is None:
            return
        lo, hi, boost = self._ent_band()   # mode-aware: raised band while bootstrapping phase 1
        if not self._ent_boosted:
            if entropy < lo:
                self.model.ent_coef = boost
                self._ent_boosted = True
                self._ent_boost_calls = 0
                print(f"🌀 Entropy collapse guard: entropy={entropy:.3f} < {lo} "
                      f"-> ent_coef boosted to {boost}")
            return
        # Currently boosted: count this update toward the min-hold, then test all relax conditions.
        self._ent_boost_calls += 1
        target_kl = getattr(self.model, "target_kl", None)
        kl_ok = kl is None or target_kl is None or kl < target_kl
        if (entropy > hi
                and self._ent_boost_calls >= ENT_BOOST_MIN_HOLD
                and kl_ok):
            base = self._phase_base_ent_coef()
            self.model.ent_coef = base
            self._ent_boosted = False
            print(f"🌀 Entropy recovered: entropy={entropy:.3f} > {hi} "
                  f"(held {self._ent_boost_calls} updates) -> ent_coef restored to {base}")

    def _on_rollout_end(self) -> None:
        """Drive the entropy-collapse guard once per update from the last train()'s entropy.

        SB3 records train/entropy_loss (= -mean_entropy) inside train(); at on_rollout_end it
        still holds the previous update's value (the logger dump that clears it runs after this
        hook), so the read is reliable from the second update on (None before the first train()).
        Also surface the live ent_coef so the guard's action is visible in TensorBoard."""
        # A RESUMED model carries its previous run's armed KL guard (target_kl is saved with
        # the model) while the callback restarts fresh -- recognizing it here prevents the
        # destructive bootstrap-floor(0.025)+tight-KL combination on resume.
        if not self._opt_guarded and getattr(self.model, "target_kl", None) is not None:
            self._opt_guarded = True

        name_to_value = getattr(self.model.logger, 'name_to_value', {})
        ent_loss = name_to_value.get('train/entropy_loss')
        approx_kl = name_to_value.get('train/approx_kl')
        self._maybe_update_ent_mode()   # mode first: the guard reads the mode-aware band below
        self._maybe_guard_entropy_collapse(
            None if ent_loss is None else -float(ent_loss),
            kl=None if approx_kl is None else float(approx_kl),
        )
        self._rebaseline_ent_coef()   # drop the resting floor when the stage advances (2.1 -> 2.2)
        self.logger.record('optim/ent_coef', float(self.model.ent_coef))
        self.logger.record('optim/phase2_stage',
                           float(self._phase2_stage or 0) if self._phase == 2 else 0.0)
        self.logger.record('optim/ent_floor_mode', 1.0 if self._ent_mode == "bootstrap" else 0.0)
        self.logger.record('optim/p5_quality_boost', 1.0 if self._p5_quality_boost_active() else 0.0)
        cold_rate = self._cold_completion_rate()
        if cold_rate is not None:
            self.logger.record('success/cold_completion_rate', cold_rate)
        median_exc = self._median_excitement()
        if median_exc is not None:
            self.logger.record('quality/median_excitement', median_exc)

    def _on_step(self) -> bool:
        # Track total steps for throughput calculation
        self.total_steps += self.n_envs
        
        # Per-step metrics are read from the step info dict (already transferred by the step
        # barrier), NOT via env.get_attr()/env.env_method(). Each of those is a separate
        # SubprocVecEnv collective -- a synchronized round-trip to all workers -- so issuing
        # three of them on every vector step adds three needless barriers at 20x parallelism.
        # We log only the first env's values, exactly as before.
        infos = self.locals.get('infos') or []
        first_info = infos[0] if infos else {}

        track_length = first_info.get('track_length')
        if track_length is not None:
            self.logger.record('metrics/track_length', track_length)

        current_distance = first_info.get('current_distance')
        if current_distance is not None:
            self.logger.record('metrics/current_distance', current_distance)

        collision_count = first_info.get('collision_count')
        if collision_count is not None:
            self.logger.record('metrics/collision_count', collision_count)

        self.total_actions += self.n_envs
        
        # Check for episode ends across all environments
        for env_idx in range(self.n_envs):
            if self.locals['dones'][env_idx]:
                self.episode_counts[env_idx] += 1
                self.total_episode_count += 1

                # Check success
                loop_completed = self.locals['infos'][env_idx].get('loop_completed', False)
                if loop_completed:
                    self.loop_completed_counts[env_idx] += 1
                    self.total_loop_completed += 1
                    self.logger.record(f'success/env_{env_idx}_loop_completed', 1.0)
                else:
                    self.logger.record(f'success/env_{env_idx}_loop_completed', 0.0)

                # Pool episode outcomes (no get_attr/env_method IPC): every episode feeds
                # the entropy floor's any-completion window; cold episodes additionally
                # feed the phase-gate readout tag.
                self._completion_window.append(bool(loop_completed))
                if self.locals['infos'][env_idx].get('cold_start', False):
                    self._cold_window.append(bool(loop_completed))

                # Ride-quality telemetry (phase 4+ only: earlier phases skip testing and
                # emit all-zero sentinels that must not pollute the windows). Untested
                # completions count against test success but not excitement.
                _info_q = self.locals['infos'][env_idx]
                rr = _info_q.get('ride_rating')
                phase_i = _info_q.get('learning_phase', self._env_phase.get(env_idx, 1))
                if rr is not None and phase_i >= 4:
                    # POSITIVE stats only (the -0.01 unrated sentinel is truthy)
                    tested = bool(rr.get('excitement', 0) > 0 or rr.get('intensity', 0) > 0
                                  or rr.get('nausea', 0) > 0)
                    self._test_window.append(tested)
                    if tested:
                        self._exc_window.append(float(rr.get('excitement', 0.0)))
                        self.logger.record('quality/excitement', float(rr.get('excitement', 0.0)))
                        self.logger.record('quality/intensity', float(rr.get('intensity', 0.0)))
                        self.logger.record('quality/nausea', float(rr.get('nausea', 0.0)))
                    self.logger.record('quality/test_success_rate',
                                       sum(self._test_window) / len(self._test_window))
                
                # Print episode details in verbose mode
                if self.training_verbose >= 1:
                    # Get episode metrics from info dict
                    info = self.locals['infos'][env_idx]
                    episode_metrics = info.get('episode_metrics', {})
                    
                    # Determine if truncated (max steps/length) or terminated (loop completed)
                    termination_type = "completed" if loop_completed else "truncated"
                    
                    # Get final reward from the Monitor wrapper's episode info
                    final_reward = info.get('episode', {}).get('r', 0)
                    
                    # Get track length from episode metrics
                    track_length = episode_metrics.get('track_length', 0)
                    
                    print(f"Episode {self.episode_counts[env_idx]} (Env {env_idx}): "
                          f"Reward={final_reward:.1f}, "
                          f"Track={track_length} pieces, "
                          f"Loop={'✓' if loop_completed else '✗'}, "
                          f"Status={termination_type}")
                
                # Log overall success rate
                if self.total_episode_count > 0:
                    overall_success_rate = self.total_loop_completed / self.total_episode_count
                    self.logger.record('success/overall_loop_completion_rate', overall_success_rate)
                
                # Log per-environment success rate
                if self.episode_counts[env_idx] > 0:
                    env_success_rate = self.loop_completed_counts[env_idx] / self.episode_counts[env_idx]
                    self.logger.record(f'success/env_{env_idx}_completion_rate', env_success_rate)
                
                # Log curriculum info if available (from first env that completes).
                # The improved 5-phase wrapper emits 'learning_phase'/'curriculum_phase',
                # Phase-2 sub-stage diagnostics, and (in phases 2-3) 'qualified_rate' so
                # progress past each lift-hill gate is visible in TensorBoard.
                _info = self.locals['infos'][env_idx]
                # Capture phase/stage BEFORE arming: _maybe_arm_kl_guard sets ent_coef via
                # _phase_base_ent_coef (which keys on the fleet view), and it early-returns
                # once armed -- this capture also keeps _phase2_stage current for the
                # rollout-end re-baseline after the guard is armed.
                self._note_env_phase(env_idx, _info)
                self._maybe_arm_kl_guard(_info)
                if 'curriculum_phase' in _info:
                    self.logger.record('curriculum/phase', _info['curriculum_phase'])
                elif 'learning_phase' in _info:
                    self.logger.record('curriculum/phase', _info['learning_phase'])
                if 'max_track_length' in _info:
                    self.logger.record('curriculum/max_length', _info['max_track_length'])
                if 'qualified_rate' in _info:
                    self.logger.record('curriculum/qualified_rate', _info['qualified_rate'])
                for key in (
                    'phase2_stage',
                    'phase2_threshold',
                    'phase2_roundtrip_gain',
                    'phase2_summit_reward',
                    'phase2_summit_rate',
                    'phase2_roundtrip_rate',
                    'phase2_chain1_completion_rate',
                    'phase2_chain2_completion_rate',
                    'phase2_chain3_completion_rate',
                    'completed_chain_count',
                    # warm-start reverse curriculum diagnostics (cold_* are the numbers that
                    # matter -- success/overall_* and ep_rew/ep_len are scaffold-mixed now)
                    'cold_success_rate',
                    'scaffold_success_rate',
                    'cold_fraction',
                    'warm_k',
                    'warm_k_max',
                    'warm_frontier_rate',
                    'warm_prefix_len',
                    'loop_library_size',
                ):
                    if key in _info:
                        self.logger.record(f'curriculum/{key}', _info[key])
                
                # Episode metrics provided via info dict before reset
                info_metrics = self.locals['infos'][env_idx].get('episode_metrics', {})
                if info_metrics:
                    if 'track_length' in info_metrics and loop_completed:
                        self.logger.record(f'success/env_{env_idx}_completed_track_length', info_metrics['track_length'])
                    if 'min_distance' in info_metrics:
                        self.logger.record(f'navigation/env_{env_idx}_min_distance', info_metrics['min_distance'])
                    if env_idx == 0:  # Log detailed metrics only from first env
                        if 'phase_rewards' in info_metrics:
                            for phase, reward in info_metrics['phase_rewards'].items():
                                self.logger.record(f'rewards/{phase}_total', reward)
                        if 'chain_lift_count' in info_metrics:
                            self.logger.record('chain_lift/count', info_metrics['chain_lift_count'])
                        if 'chain_count' in info_metrics:   # history-based, gate-aligned
                            self.logger.record('chain_lift/history_count', info_metrics['chain_count'])
                        if 'struct_bonus' in info_metrics:
                            self.logger.record('rewards/struct_bonus', info_metrics['struct_bonus'])
                        if 'max_gain' in info_metrics:
                            self.logger.record('height/max_gain', info_metrics['max_gain'])
                        if 'roundtrip' in info_metrics:
                            self.logger.record('height/roundtrip', info_metrics['roundtrip'])
                        if 'summit' in info_metrics:
                            self.logger.record('height/summit', info_metrics['summit'])
                        if 'return_potential' in info_metrics:
                            self.logger.record('rewards/return_potential', info_metrics['return_potential'])
                        if 'route_potential' in info_metrics:
                            self.logger.record('rewards/route_potential', info_metrics['route_potential'])
                        # scale diagnostics (P3-5 redesign): is the agent actually building bigger?
                        if 'drop_z' in info_metrics:
                            self.logger.record('structure/drop_z', info_metrics['drop_z'])
                        if 'chain_height' in info_metrics:
                            self.logger.record('structure/chain_height', info_metrics['chain_height'])
                        if 'track_length' in info_metrics:
                            self.logger.record('structure/completed_length', info_metrics['track_length'])
                        if 'remove_count' in info_metrics:
                            self.logger.record('behavior/remove_count', info_metrics['remove_count'])

        # Calculate and log throughput
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            steps_per_second = self.total_steps / elapsed_time
            self.logger.record('performance/steps_per_second', steps_per_second)
            self.logger.record('performance/total_episodes', self.total_episode_count)
            
        # Print progress dashboard (only once per milestone)
        dashboard_interval = 10 * self.n_envs
        if (self.total_episode_count > 0 and 
            self.total_episode_count >= self.last_dashboard_episode + dashboard_interval):
            # Update last printed milestone
            self.last_dashboard_episode = (self.total_episode_count // dashboard_interval) * dashboard_interval
            
            overall_success_rate = self.total_loop_completed / self.total_episode_count
            episodes_per_second = self.total_episode_count / elapsed_time if elapsed_time > 0 else 0
            
            # Calculate dynamic dashboard width based on number of environments
            # Minimum 58, but expand if needed for environment status
            env_status_str_len = 16 + (self.n_envs * 2) + 2  # "Environments: [" + emojis + spaces + "]"
            dashboard_width = max(58, env_status_str_len + 2)  # +2 for padding
            
            # Create a clean dashboard display
            print("\n" + "┌" + "─" * dashboard_width + "┐")
            header = f"│ 🎮 Parallel Training Dashboard ({self.n_envs} environments)"
            print(header.ljust(dashboard_width + 1) + "│")
            print("├" + "─" * dashboard_width + "┤")
            
            # Environment status indicators
            env_status = []
            status_counts = {"🟢": 0, "🟡": 0, "🔴": 0, "⚪": 0}
            for i in range(self.n_envs):
                if self.episode_counts[i] > 0:
                    rate = self.loop_completed_counts[i] / self.episode_counts[i]
                    if rate >= 0.3:
                        env_status.append("🟢")  # Good performance
                        status_counts["🟢"] += 1
                    elif rate >= 0.1:
                        env_status.append("🟡")  # Learning
                        status_counts["🟡"] += 1
                    else:
                        env_status.append("🔴")  # Struggling
                        status_counts["🔴"] += 1
                else:
                    env_status.append("⚪")  # No episodes yet
                    status_counts["⚪"] += 1
            
            # Display environment status based on count
            if self.n_envs <= 20:
                # Show individual status for up to 20 environments
                print(f"│ Environments: [{' '.join(env_status)}]".ljust(dashboard_width + 1) + "│")
            else:
                # Show summary for many environments
                summary = f"🟢×{status_counts['🟢']} 🟡×{status_counts['🟡']} 🔴×{status_counts['🔴']} ⚪×{status_counts['⚪']}"
                print(f"│ Environments ({self.n_envs}): {summary}".ljust(dashboard_width + 1) + "│")
            print(f"│ Episodes: {self.total_episode_count:,} | Success: {overall_success_rate:.1%} ({self.total_loop_completed}/{self.total_episode_count})".ljust(dashboard_width + 1) + "│")
            print(f"│ Throughput: {steps_per_second:.1f} steps/s | {episodes_per_second:.2f} eps/s".ljust(dashboard_width + 1) + "│")
            
            # Get curriculum info from first environment if available. get_env() just returns the
            # VecEnv reference (no worker round-trip / IPC barrier, unlike get_attr/env_method), and
            # .envs only exists for DummyVecEnv -- under SubprocVecEnv this block is simply skipped.
            env = self.model.get_env()
            if hasattr(env, 'envs') and len(env.envs) > 0:
                wrapped_env = env.envs[0]
                temp_env = wrapped_env
                while temp_env is not None:
                    if hasattr(temp_env, 'current_stage') and hasattr(temp_env, 'current_max_length'):
                        print(f"│ Curriculum: Stage {temp_env.current_stage} | Max Length: {temp_env.current_max_length}".ljust(dashboard_width + 1) + "│")
                        break
                    if hasattr(temp_env, 'env'):
                        temp_env = temp_env.env
                    else:
                        break
            
            print("└" + "─" * dashboard_width + "┘")
            
            # Show detailed per-environment stats every 50 episodes
            if self.total_episode_count % (50 * self.n_envs) == 0:
                print("\n  Per-environment performance:")
                # Show all environments, but format differently for many environments
                if self.n_envs <= 8:
                    # Show detailed stats for up to 8 environments
                    for i in range(self.n_envs):
                        if self.episode_counts[i] > 0:
                            rate = self.loop_completed_counts[i] / self.episode_counts[i]
                            print(f"    Env {i}: {rate:.1%} success ({self.loop_completed_counts[i]}/{self.episode_counts[i]} episodes)")
                else:
                    # For many environments, show in a more compact format
                    print("    ", end="")
                    for i in range(self.n_envs):
                        if self.episode_counts[i] > 0:
                            rate = self.loop_completed_counts[i] / self.episode_counts[i]
                            print(f"E{i}:{rate:.0%} ", end="")
                            if (i + 1) % 8 == 0 and i < self.n_envs - 1:
                                print("\n    ", end="")
                    print()  # Final newline
        
        return True

def mask_fn(env: gym.Env) -> np.ndarray:
    """
    Returns the action mask for the current environment state.
    Navigates through all wrappers to find the base environment.
    """
    # Navigate through wrappers to find the base OpenRCT2 environment
    current_env = env
    while current_env is not None:
        # Check if this environment has the valid_action_mask method
        if hasattr(current_env, 'valid_action_mask'):
            return current_env.valid_action_mask()
        
        # Try to go deeper through the wrapper chain
        if hasattr(current_env, 'env'):
            current_env = current_env.env
        elif hasattr(current_env, 'unwrapped'):
            current_env = current_env.unwrapped
        else:
            break
    
    # Fallback - all actions valid (shouldn't reach here)
    print("Warning: Could not find valid_action_mask method, allowing all actions")
    return np.ones(env.action_space.n, dtype=bool)


def create_curriculum_masked_env(port: int, verbose: int = 0,
                                 warm_start_enabled: bool = True,
                                 loop_library_path: Optional[str] = None,
                                 p_cold: float = 0.25,
                                 game_speed: int = 8) -> gym.Env:
    """Create an improved-curriculum environment with action masking for a port."""
    # A custom library path must redirect BOTH sides: the wrapper's read pool AND the
    # env's harvest destination (class attr; this runs inside each SubprocVecEnv worker).
    # Redirecting only the read side silently leaks every harvest into the default file.
    if loop_library_path:
        OpenRCT2Env._LOOP_LIBRARY_PATH = loop_library_path

    # Base environment with specific port and verbosity
    base_env = gym.make('OpenRCT2-v0', host='localhost', port=port, verbose=verbose)

    # Ride ratings take ~35s of SIM time; speed 8 makes P4/P5 ride tests ~4-5s. Best-effort:
    # an older plugin without setGameSpeed just declines (env still works, tests just slow).
    if game_speed and game_speed > 1:
        api = base_env.unwrapped.api_controller
        set_speed = getattr(api, "set_game_speed", None)
        resp = set_speed(game_speed) if callable(set_speed) else {"success": False}
        if not resp.get("success") and verbose >= 1:
            print(f"⚠️ setGameSpeed unsupported on port {port} (deploy the updated plugin "
                  f"for ~{game_speed}x faster ride tests)")

    # Add OpenRCT2Wrapper to expose valid_action_mask method
    # This is crucial for the mask_fn to work
    base_env = OpenRCT2Wrapper(base_env)

    env = ImprovedPhasedCurriculumWrapper(
        base_env,
        phase1_success_threshold=0.5,   # 50% loop completion
        phase2_roundtrip_threshold=0.30,  # 30% with one-chain climb-and-return
        phase2_chain1_success_threshold=0.30,  # 30% completion with >=1 chain
        phase2_success_threshold=0.4,   # 40% completion with >=3 chain lifts
        phase3_success_threshold=0.35,  # 35% with good patterns
        phase4_success_threshold=0.30,  # 30% clean completions
        phase5_success_threshold=0.25,  # 25% with quality ratings
        window_size=50,
        phase1_max_length=40,   # Return practice (raised 25->40: give the agent room to finish the loop)
        phase2_max_length=40,   # Lift hill building
        phase3_max_length=60,   # Drop & turn
        phase4_max_length=80,   # Circuit mastery
        phase5_initial_length=80,
        phase5_target_length=120,
        phase5_increase_step=10,
        verbose=verbose,
        warm_start_enabled=warm_start_enabled,
        loop_library_path=loop_library_path,
        p_cold=p_cold,
    )

    # Add Monitor for logging
    env = Monitor(env)

    # Add ActionMasker for MaskablePPO
    env = ActionMasker(env, mask_fn)

    return env


def make_env_factory(port: int, verbose: int = 0,
                     warm_start_enabled: bool = True,
                     loop_library_path: Optional[str] = None,
                     p_cold: float = 0.25,
                     game_speed: int = 8) -> Callable[[], gym.Env]:
    """Create a factory function for an environment on a specific port"""
    def _init() -> gym.Env:
        try:
            env = create_curriculum_masked_env(port, verbose, warm_start_enabled,
                                               loop_library_path, p_cold, game_speed)
            print(f"✅ Successfully connected to OpenRCT2 on port {port}")
            return env
        except Exception as e:
            print(f"❌ Failed to connect to OpenRCT2 on port {port}: {e}")
            raise
    return _init


def _create_vector_env(env_factories: List[Callable[[], gym.Env]]):
    """Use subprocess workers only when there is real multi-port parallelism."""
    n_envs = len(env_factories)
    if n_envs > 1:
        env = SubprocVecEnv(env_factories)
        print(f"✅ Created {n_envs} parallel environments using SubprocVecEnv")
        return env

    env = DummyVecEnv(env_factories)
    print("✅ Created 1 environment using DummyVecEnv")
    return env


def train(
    ports: List[int],
    total_timesteps: int,
    checkpoint_freq: int,
    eval_freq: int,
    model_path: Optional[str] = None,
    verbose: int = 0,
    eval_episodes: int = 10,
    disable_eval: bool = False,
    target_rollout: int = 2048,
    warm_start_enabled: bool = True,
    loop_library_path: Optional[str] = None,
    p_cold: float = 0.25,
    game_speed: int = 8,
):
    """Train agent with curriculum learning AND action masking on multiple parallel environments"""

    n_envs = len(ports)

    # A fresh run must NOT inherit a stale closing-geometry calibration from a previous run
    # (it would anchor Phi at an old/atypical closing state). Clear it unless we're resuming
    # an existing model, in which case the calibration should stay consistent with it.
    # NOTE: the warm-start loop library is deliberately NOT cleared -- verified loop geometry
    # is regime-independent, unlike the Phi anchor.
    model_path = _resolve_model_path(model_path)   # hard error on a typo'd resume path
    resuming = model_path is not None
    if not resuming and _clear_calibration_cache():
        print("🧭 Fresh run: cleared stale closing-geometry calibration (will recalibrate "
              "from this run's first completion)")

    from openrct2_gym.envs.openrct2_env import OpenRCT2Env as _Env
    _lib_path = loop_library_path or _Env._LOOP_LIBRARY_PATH
    _lib_size = len(LoopLibrary(_lib_path))
    if warm_start_enabled:
        print(f"📚 Warm-start loop library: {_lib_size} verified loops at {_lib_path}"
              + ("" if _lib_size else " (empty: run build_loop_library.py to seed, or the"
                 " run bootstraps cold until the first harvested completion)"))
    else:
        print("📚 Warm-start reverse curriculum DISABLED (--no-warm-start)")

    print("="*60)
    print("🎓 PARALLEL CURRICULUM LEARNING + ACTION MASKING")
    print("="*60)
    print(f"Training on {n_envs} parallel OpenRCT2 instances")
    print(f"Ports: {', '.join(map(str, ports))}")
    print("Using improved 5-phase curriculum with physics-aware rewards:")
    print("  Phase 1: Return Practice (40 pieces) - Learn navigation")
    print("  Phase 2: Lift Hill Building (40 pieces) - staged chain roundtrip/completion gates")
    print("  Phase 3: Drop & Turn (60 pieces) - Learn drops & turnarounds")
    print("  Phase 4: Circuit Mastery (80 pieces) - Full integration")
    print("  Phase 5: Quality Optimization (80-120 pieces) - E=7-9, I=4.5-6.5, N<4.5")
    print("  + Energy estimation, pattern detection, approach guidance")
    print("Using MaskablePPO to prevent invalid actions")
    print("="*60 + "\n")

    # Intermediate evaluation reaches into the env wrappers to suppress curriculum-stat
    # updates and force cold (unscaffolded) episodes, via _unwrap_to_vecenv_with_envs().
    # That returns None under SubprocVecEnv (n_envs > 1, no .envs), so eval episodes would
    # be SCAFFOLDED (measuring the loop library, not the policy), pollute curriculum gates
    # and the annealer, and desync the rollout collector. Not a warning -- hard-disable.
    if not disable_eval and n_envs > 1:
        print(f"⚠️  Intermediate eval DISABLED automatically for {n_envs} parallel envs "
              "(SubprocVecEnv): eval cannot reach the wrappers to suppress curriculum stats "
              "or force cold episodes. Evaluate checkpoints separately with run_model.py.\n")
        disable_eval = True

    # Create environment factories for each port
    env_factories = [make_env_factory(port, verbose, warm_start_enabled,
                                      loop_library_path, p_cold, game_speed) for port in ports]

    # Create vectorized environments
    print(f"\n🔌 Connecting to {n_envs} OpenRCT2 instances...")
    env = _create_vector_env(env_factories)

    # Normalize ONLY the continuous 'scalars' key. norm_reward=False preserves the
    # curriculum's tuned absolute reward magnitudes and phase-advancement thresholds.
    # Other keys (map, tokens, mask, already-clipped goal vectors, Discretes) are left
    # alone (VecNormalize handles only Box keys, and would corrupt token ids).
    vecnorm_path = _vecnormalize_path(model_path) if model_path else None
    if vecnorm_path and os.path.exists(vecnorm_path):
        print(f"Loading VecNormalize stats from {vecnorm_path}")
        env = VecNormalize.load(vecnorm_path, env)
        env.training = True
        env.norm_reward = False
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=False, norm_obs_keys=["scalars"])

    # IMPORTANT: Do NOT create a separate eval env on the same ports.
    # We will evaluate using the training env between learn chunks to avoid
    # corrupting in-progress episodes on shared API ports.

    # Limit the main process's intra-op threads for the PPO update. The SubprocVecEnv workers
    # (already spawned above) run only env code -- not Torch ops -- and inherit OMP/BLAS=1 from
    # the env vars set at import time, so this configures only the trainer and avoids thread
    # oversubscription on a many-core host running 20 game instances + 20 workers.
    torch.set_num_threads(min(8, os.cpu_count() or 8))

    # Create or load model
    if model_path and os.path.exists(model_path):
        print(f"Loading MaskablePPO model from {model_path}")
        model = MaskablePPO.load(model_path, env=env)
    else:
        print(f"Creating new MaskablePPO model for {n_envs} parallel environments")
        policy_kwargs = dict(
            features_extractor_class=BuildHistoryExtractor,
            features_extractor_kwargs=dict(encoder="gru"),
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            normalize_images=False,
        )
        
        # Adjust n_steps and batch_size ensuring train_batch_size % batch_size == 0.
        # Aim for ~target_rollout transitions per PPO update (CLI --target-rollout, default
        # 2048), keeping n_steps >= 128. At high n_envs n_steps shrinks (2048/20 -> 128); raise
        # target_rollout to hold n_steps >= 256 (e.g. 5120 with 20 envs -> base 256).
        base = target_rollout // max(1, n_envs)
        # Align to 64 for better batch divisibility and keep >= 128
        n_steps = max(128, (base // 64) * 64 if base >= 64 else 128)
        train_batch_size = n_envs * n_steps
        # Start with a reasonable minibatch size that divides train_batch_size
        batch_size = math.gcd(train_batch_size, 64 * n_envs)
        batch_size = max(32, batch_size)
        assert train_batch_size % batch_size == 0, (
            "train_batch_size must be divisible by batch_size"
        )
        
        model = MaskablePPO(
            MaskableMultiInputActorCriticPolicy,
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log="./parallel_curriculum_masked_tensorboard/",
            n_steps=n_steps,
            batch_size=batch_size,
            **PPO_HYPERPARAMS,
        )

    # PBRS invariance precondition: the model's discount must match the reward's gamma.
    # Catches a stale loaded model trained under a different gamma.
    assert model.gamma == GAMMA, (
        f"model gamma {model.gamma} != reward gamma {GAMMA}; PBRS invariance would break"
    )

    # Create log directory
    log_dir = f"logs_parallel_curriculum_masked_{n_envs}envs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Callbacks
    _ckpt_save_freq = max(1, checkpoint_freq // max(1, n_envs))  # Guard against zero
    _ckpt_prefix = f"parallel_curriculum_masked_{n_envs}envs"
    checkpoint_callback = CheckpointCallback(
        save_freq=_ckpt_save_freq,
        save_path=log_dir,
        name_prefix=_ckpt_prefix
    )

    # Save VecNormalize stats alongside each checkpoint so checkpoints stay reloadable.
    vecnormalize_callback = SaveVecNormalizeCallback(
        save_freq=_ckpt_save_freq,
        save_path=log_dir,
        name_prefix=_ckpt_prefix,
    )

    tensorboard_callback = ParallelCurriculumMaskableCallback(n_envs=n_envs, training_verbose=verbose)

    # Container for final curriculum statistics
    stats = None

    # Train
    try:
        print(f"\n🚂 Starting parallel training on {n_envs} environments...")
        print("Features enabled:")
        print("  ✓ Curriculum learning (50 → 120 pieces)")
        print("  ✓ True action masking (invalid actions prevented)")
        print("  ✓ Stronger return rewards")
        print("  ✓ Distance checkpoints")
        print("  ✓ Chain lift incentives")
        print(f"  ✓ {n_envs}x parallel environment execution")
        print("\nMonitor progress in Tensorboard:")
        print("  tensorboard --logdir ./parallel_curriculum_masked_tensorboard/\n")
        
        # Train in chunks, evaluating with the SAME env in between chunks.
        remaining = total_timesteps
        chunk = max(1, eval_freq) if not disable_eval else remaining
        learned = 0
        first_chunk = True
        while remaining > 0:
            this_chunk = remaining if disable_eval else min(chunk, remaining)
            # reset_num_timesteps=True on a fresh run's FIRST chunk only: that is when SB3
            # starts a new TB run dir (False appended every fresh run into the previous
            # run's PPO_0). Resumes and later chunks continue the same run/counter.
            model.learn(
                total_timesteps=this_chunk,
                callback=[checkpoint_callback, vecnormalize_callback, tensorboard_callback],
                reset_num_timesteps=first_chunk and not resuming,
            )
            first_chunk = False
            learned += this_chunk
            remaining -= this_chunk
            
            # Evaluate between chunks using the training env to avoid port conflicts
            if not disable_eval and eval_episodes > 0:
                print(f"\n📈 Intermediate evaluation after {learned:,} timesteps...")

                # Temporarily disable curriculum statistics during evaluation
                curriculum_wrappers = []
                base_vec = _unwrap_to_vecenv_with_envs(env)
                if base_vec is not None:
                    for wrapped_env in base_vec.envs:
                        temp_env = wrapped_env
                        while temp_env is not None:
                            if hasattr(temp_env, 'evaluation_mode'):
                                curriculum_wrappers.append(temp_env)
                                break
                            if hasattr(temp_env, 'env'):
                                temp_env = temp_env.env
                            else:
                                break

                # Freeze VecNormalize running stats during evaluation so eval rollouts
                # don't pollute the obs normalization statistics.
                vecn = model.get_vec_normalize_env()
                prev_training = vecn.training if vecn is not None else None
                if vecn is not None:
                    vecn.training = False
                try:
                    with ExitStack() as stack:
                        for cw in curriculum_wrappers:
                            stack.enter_context(cw.evaluation_mode())
                        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=eval_episodes)
                finally:
                    if vecn is not None:
                        vecn.training = prev_training
                    # evaluate_policy stepped the TRAINING env: model._last_obs is now stale
                    # against the real env state, and the next collect would pair pre-eval
                    # observations with post-eval transitions. Clearing it forces a clean
                    # env reset at the start of the next learn chunk.
                    model._last_obs = None
                print(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Retrieve final curriculum stats before closing the environment
        base_vec = _unwrap_to_vecenv_with_envs(env)
        if base_vec is not None and len(base_vec.envs) > 0:
            env_instance = base_vec.envs[0]
            curriculum_env = None
            temp_env = env_instance

            # Navigate through wrappers to find the curriculum wrapper.
            while temp_env is not None:
                if hasattr(temp_env, 'get_phase_stats'):
                    curriculum_env = temp_env
                    break
                if hasattr(temp_env, 'env'):
                    temp_env = temp_env.env
                else:
                    break

            if curriculum_env:
                stats = curriculum_env.get_phase_stats()

        # Persist final VecNormalize stats (needed to reload the model for eval/resume)
        # before the env is closed.
        try:
            vec_env = model.get_vec_normalize_env()
            if vec_env is not None:
                vec_env.save(_vecnormalize_path(os.path.join(log_dir, "final_model")))
        except Exception as e:
            print(f"⚠️ Could not save VecNormalize stats: {e}")

        # Save the final model BEFORE closing the env: a dead SubprocVecEnv worker makes
        # close() raise (EOFError from the worker pipe), which previously skipped the save
        # entirely and ended a multi-hour run without its final model.
        final_model_path = os.path.join(log_dir, "final_model")
        try:
            model.save(final_model_path)
            print(f"\n💾 Final model saved to {final_model_path}")
            print(f"💾 VecNormalize stats: {_vecnormalize_path(final_model_path)}")
        except Exception as e:
            print(f"⚠️ Could not save final model: {e}")

        # Clean up environments after collecting stats; never let a dead worker's pipe
        # error mask the outcome above.
        try:
            env.close()
        except Exception as e:
            print(f"⚠️ env.close() failed (dead worker?): {e}")

    # Log final curriculum stats if available
    if stats:
        print("\n📊 Final Phased Curriculum Stats:")
        print(f"  Current phase: {stats['current_phase']}")
        if stats['current_phase'] == 2 and stats.get('phase2_stage'):
            print(f"  Phase 2 stage: {stats['phase2_stage']}")
        if stats['current_phase'] == 5 and stats.get('phase5_stage'):
            print(f"  Phase 5 stage: {stats['phase5_stage']}")
        print(f"  Max track length: {stats['current_max_length']}")
        print(f"  Total episodes: {stats['total_episodes']}")
        print(f"  Success rate: {stats['success_rate']:.1%}")
        print(f"  Total loops completed: {stats['total_loops_completed']}")

        if stats['phases_completed']:
            print("\n  Phase progression:")
            for phase in stats['phases_completed']:
                if 'phase' in phase:
                    print(f"    Phase {phase['phase']}: "
                          f"{phase['success_rate']:.1%} success rate, "
                          f"{phase.get('episodes', 0)} episodes")

    return model, env

def main():
    parser = argparse.ArgumentParser(description="Parallel training with curriculum + masking")
    parser.add_argument("--ports", type=str, default="8080",
                       help="Comma-separated list of ports for OpenRCT2 API servers (e.g., 8080,8081,8082)")
    parser.add_argument("--timesteps", type=int, default=1000000,
                       help="Total timesteps to train (default: 1M)")
    parser.add_argument("--checkpoint-freq", type=int, default=10000,
                       help="Checkpoint frequency (in timesteps)")
    parser.add_argument("--eval-freq", type=int, default=100000,
                       help="Evaluate between learn chunks every N timesteps using the training env; set 0 to disable")
    parser.add_argument("--eval-episodes", type=int, default=10,
                       help="Number of episodes per intermediate evaluation")
    parser.add_argument("--disable-eval", action="store_true",
                       help="Disable intermediate evaluation entirely (safer for maximum throughput)")
    parser.add_argument("--model-path", type=str,
                       help="Path to existing MaskablePPO model to continue training")
    parser.add_argument("--verbose", type=int, default=None,
                       help="Verbosity level: 0=silent, 1=important, 2=detailed (default: auto)")
    parser.add_argument("--target-rollout", type=int, default=2048,
                       help="Target transitions per PPO update; n_steps ~= target_rollout/n_envs "
                            "(min 128). Raise (e.g. 5120) to keep n_steps>=256 at many envs.")
    parser.add_argument("--no-warm-start", action="store_true",
                       help="Disable the warm-start reverse curriculum (cold starts only)")
    parser.add_argument("--loop-library", type=str, default=None,
                       help="Path to the warm-start loop library JSONL "
                            "(default: logs/loop_library.jsonl)")
    parser.add_argument("--p-cold", type=float, default=0.25,
                       help="Base probability of a cold (unscaffolded) episode while warm starts "
                            "are active; rises automatically as the anneal progresses (default 0.25)")
    parser.add_argument("--game-speed", type=int, default=8,
                       help="Requested OpenRCT2 game speed (needs the plugin's setGameSpeed "
                            "endpoint). Ride ratings take ~35s of sim time; speed 8 makes "
                            "P4/P5 ride tests ~4-5s. Set 1 to leave the game untouched.")
    args = parser.parse_args()
    
    # Parse ports
    try:
        ports = [int(port.strip()) for port in args.ports.split(',')]
    except ValueError:
        print("❌ Error: Invalid port format. Please provide comma-separated integers (e.g., 8080,8081)")
        return
    
    print("\n" + "="*60)
    print("🎢 OpenRCT2 Parallel Training: Curriculum + Action Masking")
    print("="*60)
    print("This combines the best approaches with parallel execution:")
    print("  • Curriculum learning for gradual difficulty")
    print("  • True action masking to prevent invalid moves")
    print("  • All reward improvements for better navigation")
    print(f"  • {len(ports)}x parallel environments for faster training")
    print("="*60 + "\n")
    
    # Validate that we can connect to at least one server
    print("🔍 Checking OpenRCT2 API server availability...")
    available_ports = []
    for port in ports:
        try:
            from openrct2_gym.envs.api_controller import APIController
            controller = APIController('localhost', port, verbose=0)  # Silent for connection check
            if controller.connect():
                available_ports.append(port)
                print(f"  ✅ Port {port}: Available")
            else:
                print(f"  ⚠️ Port {port}: Cannot connect")
            # Always disconnect the probe socket to avoid leaking the connection
            controller.disconnect()
        except Exception as e:
            print(f"  ⚠️ Port {port}: Error - {e}")
    
    if not available_ports:
        print("\n❌ Error: No OpenRCT2 API servers available on specified ports")
        print("Please ensure OpenRCT2 is running with the API plugin on the specified ports")
        return
    
    if len(available_ports) < len(ports):
        print(f"\n⚠️ Warning: Only {len(available_ports)} out of {len(ports)} ports are available")
        print(f"Continuing with available ports: {', '.join(map(str, available_ports))}")
    
    # Clean up any leftover rides from previous training sessions
    print("\n🧹 Cleaning up leftover rides from previous sessions...")
    for port in available_ports:
        try:
            controller = APIController('localhost', port, verbose=0)  # Silent for connection check
            if controller.connect():
                result = controller.delete_all_rides()
                if result.get("success"):
                    print(f"  ✅ Port {port}: Cleaned up all rides")
                else:
                    print(f"  ⚠️ Port {port}: Cleanup failed - {result.get('error', 'Unknown error')}")
                controller.disconnect()
        except Exception as e:
            print(f"  ⚠️ Port {port}: Error during cleanup - {e}")
    print("Cleanup complete!\n")
    
    # Auto-determine verbosity if not specified
    if args.verbose is None:
        # Use verbose=0 for multiple environments, 1 for single
        verbose = 0 if len(available_ports) > 1 else 1
    else:
        verbose = args.verbose
    
    if verbose == 0 and len(available_ports) > 1:
        print("\n💡 Tip: Running in silent mode. Use --verbose 1 or 2 for more details")
    
    model, env = train(
        available_ports,
        args.timesteps,
        args.checkpoint_freq,
        args.eval_freq,
        args.model_path,
        verbose,
        args.eval_episodes,
        args.disable_eval or args.eval_freq <= 0,
        args.target_rollout,
        warm_start_enabled=not args.no_warm_start,
        loop_library_path=args.loop_library,
        p_cold=args.p_cold,
        game_speed=args.game_speed,
    )
    # Training function already evaluates between chunks and closes env.
    # No additional evaluation here to avoid interfering with API ports.

if __name__ == "__main__":
    main()
