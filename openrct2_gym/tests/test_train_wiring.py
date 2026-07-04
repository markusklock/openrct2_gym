"""Phase 2 wiring tests (server-free).

Drive the real training wrappers (create_curriculum_masked_env -> curriculum ->
Monitor -> ActionMasker -> DummyVecEnv -> VecNormalize -> MaskablePPO + custom
extractor) with a fake API. Catches the documented VecNormalize Dict footgun
(norm_obs_keys must list only Box keys) and full-chain integration before any GPU/server.
"""
import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from openrct2_gym.envs import openrct2_env as oe_mod
from openrct2_gym.envs.feature_extractor import BuildHistoryExtractor

import train as T


class FakeAPI:
    def __init__(self, host=None, port=None, verbose=0):
        self.station_length = 3
        self._dv = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        self._stack = []

    def connect(self):
        return True

    def disconnect(self):
        pass

    def delete_all_rides(self):
        return {"success": True}

    def create_ride(self):
        self._stack = []
        return 1

    def place_track_piece(self, x, y, z, direction, track_type, has_chain=False):
        dx, dy = self._dv[direction]
        ep = {"x": x + dx, "y": y + dy, "z": z, "direction": direction}
        self._stack.append(ep)
        return {"success": True, "payload": {
            "nextEndpoint": ep, "isCircuitComplete": False,
            "validNextPieces": {"validPieces": list(range(46))}}}

    def get_valid_next_pieces(self):
        return {"success": True, "payload": {"validPieces": list(range(46))}}

    def delete_last_track_piece(self):
        if self._stack:
            self._stack.pop()
        prev = self._stack[-1] if self._stack else {"x": 61, "y": 66, "z": 14, "direction": 0}
        return {"success": True, "payload": {"nextEndpoint": prev, "piecesRemaining": len(self._stack)}}


def _make_vecnorm_env():
    return VecNormalize(
        DummyVecEnv([lambda: T.create_curriculum_masked_env(8080, verbose=0)]),
        norm_obs=True, norm_reward=False, norm_obs_keys=["scalars"],
    )


def test_vecnormalize_path_convention():
    assert T._vecnormalize_path("a/b/final_model.zip") == "a/b/final_model_vecnormalize.pkl"
    assert T._vecnormalize_path("a/b/final_model") == "a/b/final_model_vecnormalize.pkl"


def test_vector_env_uses_dummy_for_single_factory(monkeypatch):
    created = []

    class FakeDummyVecEnv:
        def __init__(self, factories):
            created.append(("dummy", len(factories)))

    class FakeSubprocVecEnv:
        def __init__(self, factories):
            created.append(("subproc", len(factories)))

    monkeypatch.setattr(T, "DummyVecEnv", FakeDummyVecEnv)
    monkeypatch.setattr(T, "SubprocVecEnv", FakeSubprocVecEnv)

    env = T._create_vector_env([lambda: None])

    assert isinstance(env, FakeDummyVecEnv)
    assert created == [("dummy", 1)]


def test_vector_env_uses_subproc_for_multiple_factories(monkeypatch):
    created = []

    class FakeDummyVecEnv:
        def __init__(self, factories):
            created.append(("dummy", len(factories)))

    class FakeSubprocVecEnv:
        def __init__(self, factories):
            created.append(("subproc", len(factories)))

    monkeypatch.setattr(T, "DummyVecEnv", FakeDummyVecEnv)
    monkeypatch.setattr(T, "SubprocVecEnv", FakeSubprocVecEnv)

    env = T._create_vector_env([lambda: None, lambda: None])

    assert isinstance(env, FakeSubprocVecEnv)
    assert created == [("subproc", 2)]


def test_unwrap_finds_dummy_vecenv_under_vecnormalize(monkeypatch):
    monkeypatch.setattr(oe_mod, "APIController", FakeAPI)
    env = _make_vecnorm_env()
    base = T._unwrap_to_vecenv_with_envs(env)
    assert base is not None and hasattr(base, "envs") and len(base.envs) == 1
    env.close()


def test_full_pipeline_trains_and_stats_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setattr(oe_mod, "APIController", FakeAPI)
    from sb3_contrib import MaskablePPO

    env = _make_vecnorm_env()  # must not raise on the new Dict space (the footgun)
    model = MaskablePPO(
        "MultiInputPolicy", env,
        policy_kwargs=dict(
            features_extractor_class=BuildHistoryExtractor,
            features_extractor_kwargs=dict(encoder="gru"),
            net_arch=dict(pi=[64], vf=[64]),
            normalize_images=False,
        ),
        n_steps=16, batch_size=16, n_epochs=1, verbose=0,
    )
    assert model.policy.features_extractor.features_dim == 352
    model.learn(total_timesteps=16)

    stats_path = tmp_path / "vn.pkl"
    model.get_vec_normalize_env().save(str(stats_path))
    assert stats_path.exists()
    env.close()

    # Reload the stats onto a fresh env (resume path)
    fresh = VecNormalize.load(
        str(stats_path),
        DummyVecEnv([lambda: T.create_curriculum_masked_env(8080, verbose=0)]),
    )
    assert "scalars" in fresh.obs_rms
    fresh.close()


def test_ppo_hyperparams_start_with_phase1_bootstrap_config(monkeypatch):
    """The model must START with the proven phase-1 bootstrap config: NO target_kl and
    ent_coef=0.01 (both runs that learned phase 1 used exactly this; adding the KL guard
    + doubled entropy globally froze phase 1 at 17 completions in 38k episodes -- the
    rare +1000 completion updates got throttled and the snowball never started). The
    guarded config is armed by the callback at phase 2 (see the arming test)."""
    monkeypatch.setattr(oe_mod, "APIController", FakeAPI)
    from sb3_contrib import MaskablePPO

    assert T.PPO_HYPERPARAMS["target_kl"] is None
    assert T.PPO_HYPERPARAMS["ent_coef"] == 0.01
    assert T.PPO_HYPERPARAMS["gamma"] == T.GAMMA
    # the guard the callback arms for phases >= 2: target_kl + a modest entropy FLOOR.
    # ent_coef=0.02 exploded entropy (completion destroyed); 0.01 imploded it (policy froze
    # on a non-completing near-miss). 0.015 sits between to keep chain lifts sampled.
    assert T.OPT_GUARDED == {"target_kl": 0.04, "ent_coef": 0.015}

    env = _make_vecnorm_env()
    model = MaskablePPO(
        "MultiInputPolicy", env,
        policy_kwargs=dict(
            features_extractor_class=BuildHistoryExtractor,
            features_extractor_kwargs=dict(encoder="gru"),
            net_arch=dict(pi=[64], vf=[64]),
            normalize_images=False,
        ),
        n_steps=16, batch_size=16, verbose=0,
        **T.PPO_HYPERPARAMS,
    )
    assert model.target_kl is None
    assert model.ent_coef == 0.01
    assert model.gamma == T.GAMMA
    env.close()


def test_clear_calibration_cache_removes_stale_file(tmp_path, monkeypatch):
    """A fresh run must drop the persisted closing-geometry cache so it recalibrates from
    its own first completion (a cache from an old reward regime misguides Phi)."""
    import json
    from openrct2_gym.envs.openrct2_env import OpenRCT2Env
    cache = tmp_path / "close_geometry.json"
    cache.write_text(json.dumps({"pos": [63, 67, 14], "dir": 3}))
    monkeypatch.setattr(OpenRCT2Env, "_CLOSE_CACHE_PATH", str(cache))
    monkeypatch.setattr(OpenRCT2Env, "_close_cache", {"pos": [63, 67, 14], "dir": 3})

    assert T._clear_calibration_cache() is True
    assert not cache.exists()                    # stale file gone
    assert OpenRCT2Env._close_cache is None       # in-memory cache reset (DummyVecEnv path)
    assert T._clear_calibration_cache() is False  # idempotent: nothing to remove


def test_callback_arms_kl_guard_when_phase2_begins():
    """The phase-2 transition is where the KL=2.49 catastrophe happened, so the guard
    (target_kl + raised ent_coef) must arm when the FLEET majority reaches phase 2 --
    and stay armed (one-way switch, phases never go backward)."""
    from types import SimpleNamespace
    cb = T.ParallelCurriculumMaskableCallback(n_envs=2)
    cb.model = SimpleNamespace(target_kl=None, ent_coef=0.01)

    cb._maybe_arm_kl_guard({})                          # no phase info -> no-op
    cb._note_env_phase(0, {'learning_phase': 1})        # phase 1 -> stays in bootstrap config
    cb._maybe_arm_kl_guard({'learning_phase': 1})
    assert cb.model.target_kl is None
    assert cb.model.ent_coef == 0.01

    cb._note_env_phase(0, {'learning_phase': 2, 'phase2_stage': 3})
    cb._note_env_phase(1, {'learning_phase': 2, 'phase2_stage': 3})
    cb._maybe_arm_kl_guard({'learning_phase': 2})       # fleet at phase 2 -> guard arms
    assert cb.model.target_kl == 0.04
    assert cb.model.ent_coef == 0.015                   # modest entropy floor (between 0.01/0.02)

    cb.model.target_kl = 0.99                           # one-way: arming never re-fires
    cb._maybe_arm_kl_guard({'learning_phase': 3})
    cb._maybe_arm_kl_guard({'learning_phase': 1})
    assert cb.model.target_kl == 0.99


def test_entropy_collapse_guard_boosts_when_entropy_low():
    """A run silently freezes when phase-1 entropy bleeds to ~0: the softmax saturates,
    KL->0, gradients vanish, and the now-deterministic policy collides off the station for
    all 256 steps building NOTHING (observed at ~1.0M steps: entropy_loss -0.46 -> -0.0003,
    track_length 13 -> 0, ep_len 29 -> 256). Recovery from a saturated softmax is ~hopeless,
    so the guard must re-inject exploration BEFORE that -- when entropy drops below the floor."""
    from types import SimpleNamespace
    cb = T.ParallelCurriculumMaskableCallback(n_envs=2)
    cb.model = SimpleNamespace(target_kl=None, ent_coef=0.01)
    cb._ent_mode = "normal"                             # legacy band (bootstrap band tested below)
    cb._maybe_guard_entropy_collapse(0.02)              # well under the floor -> boost
    assert cb.model.ent_coef == T.ENT_COLLAPSE_BOOST
    assert cb._ent_boosted is True


def test_entropy_collapse_guard_dormant_in_healthy_band():
    """No interference during normal convergence: this run's productive entropy stayed
    >=0.17, so the guard must stay dormant there -- a permanent boost would stop the rare
    completion snowball from sharpening into a reliable policy."""
    from types import SimpleNamespace
    cb = T.ParallelCurriculumMaskableCallback(n_envs=2)
    cb.model = SimpleNamespace(target_kl=None, ent_coef=0.01)
    cb._ent_mode = "normal"
    cb._maybe_guard_entropy_collapse(0.20)
    assert cb.model.ent_coef == 0.01
    assert cb._ent_boosted is False


def test_entropy_collapse_guard_restores_phase_base_after_recovery():
    """Hysteresis: once entropy climbs back above the recovery threshold AND the boost has been
    held for the min-hold window, hand ent_coef back to the phase base (the boost is a temporary
    impulse, not a permanent floor)."""
    from types import SimpleNamespace
    cb = T.ParallelCurriculumMaskableCallback(n_envs=2)
    cb.model = SimpleNamespace(target_kl=None, ent_coef=T.ENT_COLLAPSE_BOOST)
    cb._ent_mode = "normal"
    cb._ent_boosted = True
    cb._ent_boost_calls = 0
    for _ in range(T.ENT_BOOST_MIN_HOLD):              # recovered; ride out the min-hold -> restore
        cb._maybe_guard_entropy_collapse(0.40)
    assert cb.model.ent_coef == T.OPT_PHASE1['ent_coef']
    assert cb._ent_boosted is False


def test_entropy_collapse_guard_hysteresis_holds_boost_in_band():
    """Between floor and recovery threshold the state must NOT flip (no per-rollout
    thrashing): a boosted guard stays boosted until entropy fully recovers."""
    from types import SimpleNamespace
    cb = T.ParallelCurriculumMaskableCallback(n_envs=2)
    cb.model = SimpleNamespace(target_kl=None, ent_coef=T.ENT_COLLAPSE_BOOST)
    cb._ent_mode = "normal"
    cb._ent_boosted = True
    cb._maybe_guard_entropy_collapse(0.20)              # in the band -> hold the boost
    assert cb.model.ent_coef == T.ENT_COLLAPSE_BOOST
    assert cb._ent_boosted is True


def test_entropy_collapse_guard_restore_tracks_phase2_base():
    """If the phase-2 KL guard armed (raising the base to the guarded floor) while a boost
    was active, the boost must survive arming and recovery must restore to the PHASE-2 base
    (0.015), not the phase-1 base."""
    from types import SimpleNamespace
    cb = T.ParallelCurriculumMaskableCallback(n_envs=2)
    cb.model = SimpleNamespace(target_kl=None, ent_coef=0.01)
    cb._ent_mode = "normal"
    cb._maybe_guard_entropy_collapse(0.02)              # boost in phase 1
    assert cb.model.ent_coef == T.ENT_COLLAPSE_BOOST
    cb._phase = 2                                       # _on_step captures this before arming
    cb._maybe_arm_kl_guard({'learning_phase': 2})       # arm while boosted
    assert cb.model.target_kl == 0.04
    assert cb.model.ent_coef == T.ENT_COLLAPSE_BOOST    # boost preserved, NOT clobbered to 0.015
    for _ in range(T.ENT_BOOST_MIN_HOLD):              # recover -> restore to phase-2 base
        cb._maybe_guard_entropy_collapse(0.40)
    assert cb.model.ent_coef == T.OPT_GUARDED['ent_coef']
    assert cb._ent_boosted is False


def test_phase_base_ent_coef_is_early_phase2_aware():
    """The entropy FLOOR the collapse-guard restores to is raised through the early Phase-2
    discovery+integration stages (2.1 AND 2.2, where the agent must keep building chains) and
    drops back to the guarded base at stage 2.3 / later phases."""
    from types import SimpleNamespace
    cb = T.ParallelCurriculumMaskableCallback(n_envs=2)
    cb.model = SimpleNamespace(target_kl=None, ent_coef=0.01)
    assert cb._phase_base_ent_coef() == T.PHASE1_BOOTSTRAP_ENT_COEF  # fresh run: bootstrap floor
    cb._ent_mode = "normal"
    assert cb._phase_base_ent_coef() == T.OPT_PHASE1['ent_coef']     # cold completions flowing
    cb._opt_guarded = True
    cb._phase, cb._phase2_stage = 2, 1
    assert cb._phase_base_ent_coef() == T.PHASE2_EARLY_ENT_COEF      # raised floor in stage 2.1
    cb._phase2_stage = 2
    assert cb._phase_base_ent_coef() == T.PHASE2_EARLY_ENT_COEF      # ...and stage 2.2 (integration)
    cb._phase2_stage = 3
    assert cb._phase_base_ent_coef() == T.OPT_GUARDED['ent_coef']    # stage 2.3: back to guarded base
    cb._phase, cb._phase2_stage = 3, None
    assert cb._phase_base_ent_coef() == T.OPT_GUARDED['ent_coef']    # later phases: guarded base


def test_arming_kl_guard_lands_on_early_phase2_floor():
    """Arming at the phase-1->2 transition must set ent_coef via the phase-aware base, so a
    stage-2.1 (or 2.2) arm lands on the raised early-Phase-2 floor (0.018), not the bare guarded
    base. (The _phase_base_ent_coef change alone is not enough -- arming sets model.ent_coef.)"""
    from types import SimpleNamespace
    for stage in (1, 2):
        cb = T.ParallelCurriculumMaskableCallback(n_envs=2)
        cb.model = SimpleNamespace(target_kl=None, ent_coef=0.01)
        cb._phase, cb._phase2_stage = 2, stage   # captured before arming, as _on_step does
        cb._maybe_arm_kl_guard({'learning_phase': 2})
        assert cb.model.target_kl == T.OPT_GUARDED['target_kl']
        assert cb.model.ent_coef == T.PHASE2_EARLY_ENT_COEF

    cb3 = T.ParallelCurriculumMaskableCallback(n_envs=2)
    cb3.model = SimpleNamespace(target_kl=None, ent_coef=0.01)
    cb3._phase, cb3._phase2_stage = 2, 3
    cb3._maybe_arm_kl_guard({'learning_phase': 2})
    assert cb3.model.ent_coef == T.OPT_GUARDED['ent_coef']    # stage 2.3 arms at the guarded base


def test_rebaseline_ent_coef_drops_floor_after_stage_advance():
    """The raised early-Phase-2 floor (0.018) is held through stages 2.1 AND 2.2, then drops to
    0.015 when stage 2.3 begins -- the re-baseline applies it on the next rollout (the collapse
    guard only restores on recovery). A live entropy BOOST is never clobbered."""
    from types import SimpleNamespace
    cb = T.ParallelCurriculumMaskableCallback(n_envs=2)
    cb.model = SimpleNamespace(target_kl=0.04, ent_coef=0.01)
    cb._opt_guarded = True
    cb._phase, cb._phase2_stage = 2, 1
    cb._rebaseline_ent_coef()
    assert cb.model.ent_coef == T.PHASE2_EARLY_ENT_COEF
    cb._phase2_stage = 2                          # still the raised floor through 2.2
    cb._rebaseline_ent_coef()
    assert cb.model.ent_coef == T.PHASE2_EARLY_ENT_COEF
    cb._phase2_stage = 3                          # drops to the guarded base at 2.3
    cb._rebaseline_ent_coef()
    assert cb.model.ent_coef == T.OPT_GUARDED['ent_coef']
    cb._phase2_stage = 1                          # a live boost must survive the re-baseline
    cb._ent_boosted = True
    cb.model.ent_coef = T.ENT_COLLAPSE_BOOST
    cb._rebaseline_ent_coef()
    assert cb.model.ent_coef == T.ENT_COLLAPSE_BOOST


def test_entropy_collapse_guard_ignores_missing_entropy():
    """Before the first train() the logged entropy is absent; the guard must no-op."""
    from types import SimpleNamespace
    cb = T.ParallelCurriculumMaskableCallback(n_envs=2)
    cb.model = SimpleNamespace(target_kl=None, ent_coef=0.01)
    cb._maybe_guard_entropy_collapse(None)
    assert cb.model.ent_coef == 0.01
    assert cb._ent_boosted is False


def test_entropy_collapse_constants_form_a_valid_hysteresis_band():
    assert 0.0 < T.ENT_COLLAPSE_LO < T.ENT_COLLAPSE_HI
    assert T.ENT_COLLAPSE_BOOST > T.OPT_PHASE1['ent_coef']     # boost is an increase ...
    assert T.ENT_COLLAPSE_BOOST > T.OPT_GUARDED['ent_coef']    # ... above either phase base


# --------------------------- progress-conditional Phase-1 entropy floor (bootstrap mode)
# The Jun-24 run collapsed to ~0.2 nats by 130k steps and never sampled a docking sequence
# again: the 0.12/0.30 band is tuned for EXPLOITING completions, not discovering them. The
# callback now starts in "bootstrap" mode (higher floor 0.025 + raised band) and hands back
# to the proven config only once COLD-episode completions actually flow.

def _fresh_cb(phase=1):
    from types import SimpleNamespace
    cb = T.ParallelCurriculumMaskableCallback(n_envs=2)
    cb.model = SimpleNamespace(target_kl=None, ent_coef=0.01)
    cb._phase = phase
    return cb


def test_bootstrap_mode_is_initial_floor_and_band():
    cb = _fresh_cb()
    assert cb._ent_mode == "bootstrap"                  # rate is 0 by definition at start
    assert cb._phase_base_ent_coef() == T.PHASE1_BOOTSTRAP_ENT_COEF
    assert cb._ent_band() == (T.BOOT_ENT_LO, T.BOOT_ENT_HI, T.BOOT_ENT_BOOST)
    cb._rebaseline_ent_coef()                           # applied at the first rollout end
    assert cb.model.ent_coef == T.PHASE1_BOOTSTRAP_ENT_COEF


def test_bootstrap_band_boosts_at_dormant_zone_entropy():
    """0.20 nats is 'healthy' for exploiting completions but is exactly the dormant-zone
    freeze the Jun-24 run died in -- in bootstrap mode the guard must fire there."""
    cb = _fresh_cb()
    cb._maybe_guard_entropy_collapse(0.20)
    assert cb._ent_boosted is True
    assert cb.model.ent_coef == T.BOOT_ENT_BOOST


def test_bootstrap_exits_on_any_completions_after_min_hold():
    """The floor is anti-collapse insurance, keyed on ANY-episode completions (scaffolded
    included): with warm starts active, scaffolded +1000s flow by construction, and holding
    0.025 entropy then CAPS the sharpening the scaffold exists to teach (smoke run 2
    plateaued at ~15% = the random baseline). Cold rate remains the PHASE-GATE metric only."""
    cb = _fresh_cb()
    for _ in range(T.COMPLETION_RATE_MIN_SAMPLES):
        cb._completion_window.append(True)              # scaffolded completions flowing
    for _ in range(T.ENT_MODE_MIN_HOLD - 1):
        cb._maybe_update_ent_mode()
    assert cb._ent_mode == "bootstrap"                  # min-hold not yet elapsed
    cb._maybe_update_ent_mode()
    assert cb._ent_mode == "normal"
    assert cb._phase_base_ent_coef() == T.OPT_PHASE1['ent_coef']   # proven config restored
    assert cb._ent_band() == (T.ENT_COLLAPSE_LO, T.ENT_COLLAPSE_HI, T.ENT_COLLAPSE_BOOST)


def test_bootstrap_holds_without_enough_completion_samples():
    cb = _fresh_cb()
    for _ in range(T.COMPLETION_RATE_MIN_SAMPLES - 1):  # not yet statistically meaningful
        cb._completion_window.append(True)
    for _ in range(3 * T.ENT_MODE_MIN_HOLD):
        cb._maybe_update_ent_mode()
    assert cb._ent_mode == "bootstrap"


def test_bootstrap_reenters_when_all_completions_die():
    cb = _fresh_cb()
    for _ in range(T.COMPLETION_RATE_MIN_SAMPLES):
        cb._completion_window.append(True)
    for _ in range(T.ENT_MODE_MIN_HOLD):
        cb._maybe_update_ent_mode()
    assert cb._ent_mode == "normal"
    cb._completion_window.clear()
    for _ in range(T.COMPLETION_RATE_MIN_SAMPLES):
        cb._completion_window.append(False)             # completions died entirely (rate 0)
    for _ in range(T.ENT_MODE_MIN_HOLD - 1):
        cb._maybe_update_ent_mode()
    assert cb._ent_mode == "normal"                     # min-hold since last flip
    cb._maybe_update_ent_mode()
    assert cb._ent_mode == "bootstrap"


def test_bootstrap_exits_when_phase_advances():
    cb = _fresh_cb(phase=2)                             # bootstrap is a phase-1 concern
    for _ in range(T.ENT_MODE_MIN_HOLD):
        cb._maybe_update_ent_mode()
    assert cb._ent_mode == "normal"


def test_callback_pools_cold_flags_from_infos_without_ipc():
    """The mode machine reads (cold_start, loop_completed) straight from episode-end infos --
    no get_attr/env_method collectives (the no-IPC invariant the metric logging already pins)."""
    from types import SimpleNamespace

    cb = T.ParallelCurriculumMaskableCallback(n_envs=2)

    class _SpyEnv:
        def __init__(self):
            self.get_attr_calls = []
            self.env_method_calls = []

        def get_attr(self, name, *a, **k):
            self.get_attr_calls.append(name)
            return [0, 0]

        def env_method(self, name, *a, **k):
            self.env_method_calls.append(name)
            return [(0.0,), (0.0,)]

    spy = _SpyEnv()
    logger = SimpleNamespace(record=lambda key, val, *a, **k: None)
    cb.model = SimpleNamespace(get_env=lambda: spy, logger=logger, ent_coef=0.01, target_kl=None)
    cb.locals = {
        'dones': [True, True],
        'infos': [
            {'loop_completed': True, 'cold_start': True,
             'track_length': 12, 'current_distance': 0.0, 'collision_count': 0},
            {'loop_completed': True, 'cold_start': False,   # scaffolded: must NOT pollute
             'track_length': 3, 'current_distance': 0.0, 'collision_count': 0},
        ],
    }
    assert cb._on_step() is True
    assert list(cb._completion_window) == [True, True]  # every episode feeds the floor's window
    assert list(cb._cold_window) == [True]              # only the cold episode feeds the gate tag
    assert spy.get_attr_calls == [] and spy.env_method_calls == []


def test_rollout_end_logs_ent_floor_mode_and_cold_rate():
    from types import SimpleNamespace
    cb = _fresh_cb()
    recorded = {}
    cb.model.logger = SimpleNamespace(
        name_to_value={},
        record=lambda key, val, *a, **k: recorded.__setitem__(key, val))
    for _ in range(T.COMPLETION_RATE_MIN_SAMPLES):
        cb._cold_window.append(False)
    cb._on_rollout_end()
    assert recorded['optim/ent_floor_mode'] == 1.0      # bootstrap
    assert recorded['success/cold_completion_rate'] == 0.0


def test_bootstrap_constants_are_sane():
    assert T.OPT_PHASE1['ent_coef'] < T.PHASE1_BOOTSTRAP_ENT_COEF < T.BOOT_ENT_BOOST
    assert 0.0 < T.BOOT_ENT_LO < T.BOOT_ENT_HI
    assert T.ENT_COLLAPSE_HI <= T.BOOT_ENT_LO or T.BOOT_ENT_LO > T.ENT_COLLAPSE_LO  # raised band
    assert 0.0 < T.COMPLETION_RATE_ENTER < T.COMPLETION_RATE_EXIT < 1.0


def test_entropy_guard_reads_live_entropy_at_rollout_end(monkeypatch):
    """Integration: the guard's one untestable-in-isolation assumption is that SB3 exposes
    train/entropy_loss at on_rollout_end. Run two real updates with a FakeAPI and confirm the
    guard is actually driven with a non-None entropy -- if the logger timing were wrong it would
    only ever see None and silently never fire (wasting a multi-hour run to discover that)."""
    monkeypatch.setattr(oe_mod, "APIController", FakeAPI)
    from sb3_contrib import MaskablePPO

    seen = []
    orig = T.ParallelCurriculumMaskableCallback._maybe_guard_entropy_collapse

    def spy(self, entropy, kl=None):
        seen.append(entropy)
        return orig(self, entropy, kl=kl)

    monkeypatch.setattr(T.ParallelCurriculumMaskableCallback, "_maybe_guard_entropy_collapse", spy)

    env = _make_vecnorm_env()
    model = MaskablePPO(
        "MultiInputPolicy", env,
        policy_kwargs=dict(
            features_extractor_class=BuildHistoryExtractor,
            features_extractor_kwargs=dict(encoder="gru"),
            net_arch=dict(pi=[64], vf=[64]),
            normalize_images=False,
        ),
        n_steps=16, batch_size=16, n_epochs=1, verbose=0,
    )
    cb = T.ParallelCurriculumMaskableCallback(n_envs=1)
    model.learn(total_timesteps=32, callback=cb)   # two rollouts: the 2nd sees the 1st's entropy
    env.close()
    assert seen, "_on_rollout_end never drove the entropy guard"
    assert any(e is not None for e in seen), f"guard only ever saw None entropy: {seen}"


def test_on_step_logs_metrics_from_infos_without_vecenv_ipc():
    """Per-step metric logging must read track_length/current_distance/collision_count from
    self.locals['infos'][0] -- already transferred by the step barrier -- NOT via SubprocVecEnv
    get_attr/env_method collectives. Each of those is an extra synchronized round-trip to ALL
    workers on EVERY vector step; with 20 envs that is three needless barriers per step."""
    from types import SimpleNamespace

    cb = T.ParallelCurriculumMaskableCallback(n_envs=2)

    class _SpyEnv:
        def __init__(self):
            self.get_attr_calls = []
            self.env_method_calls = []

        def get_attr(self, name, *a, **k):
            self.get_attr_calls.append(name)
            return [0, 0]

        def env_method(self, name, *a, **k):
            self.env_method_calls.append(name)
            return [(0.0,), (0.0,)]

    spy = _SpyEnv()
    recorded = {}
    logger = SimpleNamespace(record=lambda key, val, *a, **k: recorded.__setitem__(key, val))
    cb.model = SimpleNamespace(get_env=lambda: spy, logger=logger, ent_coef=0.01, target_kl=None)

    cb.locals = {
        'dones': [False, False],
        'infos': [
            {'track_length': 7, 'current_distance': 3.5, 'collision_count': 2},
            {'track_length': 9, 'current_distance': 1.0, 'collision_count': 0},
        ],
    }

    assert cb._on_step() is True
    assert spy.get_attr_calls == []      # no get_attr IPC barrier
    assert spy.env_method_calls == []    # no env_method IPC barrier
    assert recorded['metrics/track_length'] == 7        # first-env value, straight from infos
    assert recorded['metrics/current_distance'] == 3.5
    assert recorded['metrics/collision_count'] == 2


def test_on_step_dashboard_milestone_renders_without_ipc_or_nameerror():
    """The dashboard block (fires at episode milestones) reads env.envs for the DummyVecEnv
    curriculum line, so `env` must still be defined there after we dropped the per-step metric
    IPC. Regression guard for the NameError crash when removing env = self.model.get_env(): it
    only surfaces once enough episodes complete to trip the milestone, which the no-done metric
    test never reached. Still must issue no get_attr/env_method collectives."""
    from types import SimpleNamespace

    cb = T.ParallelCurriculumMaskableCallback(n_envs=2)
    # Arrange just below the milestone (10*n_envs = 20) so this step crosses it.
    cb.total_episode_count = 19
    cb.episode_counts = [10, 9]
    cb.loop_completed_counts = [0, 0]
    cb.total_loop_completed = 0
    cb.last_dashboard_episode = 0

    class _SpyEnv:
        def __init__(self):
            self.get_attr_calls = []
            self.env_method_calls = []

        def get_attr(self, name, *a, **k):
            self.get_attr_calls.append(name)
            return [0, 0]

        def env_method(self, name, *a, **k):
            self.env_method_calls.append(name)
            return [(0.0,), (0.0,)]

    spy = _SpyEnv()  # SubprocVecEnv-like: no .envs attribute
    logger = SimpleNamespace(record=lambda key, val, *a, **k: None)
    cb.model = SimpleNamespace(get_env=lambda: spy, logger=logger, ent_coef=0.01, target_kl=None)

    cb.locals = {
        'dones': [True, False],   # one episode completes -> total_episode_count 19 -> 20 -> milestone
        'infos': [
            {'loop_completed': False, 'track_length': 25, 'current_distance': 4.0, 'collision_count': 1},
            {'track_length': 25, 'current_distance': 4.0, 'collision_count': 0},
        ],
    }

    assert cb._on_step() is True       # must NOT raise NameError when the dashboard renders
    assert spy.get_attr_calls == []
    assert spy.env_method_calls == []


def test_save_vecnormalize_callback_writes_per_checkpoint_stats(monkeypatch, tmp_path):
    monkeypatch.setattr(oe_mod, "APIController", FakeAPI)
    from sb3_contrib import MaskablePPO

    env = _make_vecnorm_env()
    model = MaskablePPO(
        "MultiInputPolicy", env,
        policy_kwargs=dict(
            features_extractor_class=BuildHistoryExtractor,
            features_extractor_kwargs=dict(encoder="gru"),
            net_arch=dict(pi=[64], vf=[64]),
            normalize_images=False,
        ),
        n_steps=8, batch_size=8, n_epochs=1, verbose=0,
    )
    cb = T.SaveVecNormalizeCallback(save_freq=8, save_path=str(tmp_path), name_prefix="ckpt")
    model.learn(total_timesteps=16, callback=cb)
    assert list(tmp_path.glob("ckpt_*_steps_vecnormalize.pkl")), "callback wrote no stats"
    env.close()


def test_fresh_run_resets_tb_run_dir_resume_does_not(monkeypatch, tmp_path):
    """A fresh run must pass reset_num_timesteps=True on its FIRST learn chunk: SB3 only
    starts a new TB run dir then, so passing False unconditionally appended every fresh
    run's events into the previous run's PPO_0 (overlaid, unrelated curves). Later chunks
    and resumes keep False (continue the same run / step counter)."""
    from types import SimpleNamespace
    from openrct2_gym.envs.openrct2_env import OpenRCT2Env
    monkeypatch.setattr(oe_mod, "APIController", FakeAPI)
    monkeypatch.setattr(OpenRCT2Env, "_CLOSE_CACHE_PATH", str(tmp_path / "close.json"))
    monkeypatch.setattr(OpenRCT2Env, "_LOOP_LIBRARY_PATH", str(tmp_path / "lib.jsonl"))
    monkeypatch.setattr(T, "evaluate_policy", lambda *a, **k: (0.0, 0.0))

    calls = []

    class StubPPO:
        def __init__(self, *a, **k):
            self.gamma = T.GAMMA

        @classmethod
        def load(cls, path, env=None):
            return cls()

        def learn(self, total_timesteps, callback=None, reset_num_timesteps=None):
            calls.append(reset_num_timesteps)

        def save(self, path):
            pass

        def get_vec_normalize_env(self):
            return None

        def get_env(self):
            return None

    monkeypatch.setattr(T, "MaskablePPO", StubPPO)

    # Fresh run, two chunks (eval_freq 8 over 16 steps): first True, then False.
    T.train([8080], total_timesteps=16, checkpoint_freq=1000, eval_freq=8,
            verbose=0, eval_episodes=1, disable_eval=False)
    assert calls == [True, False]

    # Resume (model file exists): never reset -- continue the checkpoint's run.
    calls.clear()
    ckpt = tmp_path / "model.zip"
    ckpt.write_bytes(b"stub")
    T.train([8080], total_timesteps=16, checkpoint_freq=1000, eval_freq=8,
            model_path=str(ckpt), verbose=0, eval_episodes=1, disable_eval=False)
    assert calls == [False, False]


def test_resolve_model_path_variants(tmp_path):
    """--model-path with a typo (or the extension-less form SB3 tolerates on load) must
    NOT silently fall through to training-from-scratch: that wipes the calibration, loads
    the OLD VecNormalize stats onto a fresh policy, and wastes the whole run."""
    exact = tmp_path / "model.zip"
    exact.write_bytes(b"x")
    assert T._resolve_model_path(None) is None
    assert T._resolve_model_path(str(exact)) == str(exact)
    assert T._resolve_model_path(str(tmp_path / "model")) == str(exact)   # .zip appended
    with pytest.raises(SystemExit):
        T._resolve_model_path(str(tmp_path / "nope"))


def test_final_model_saved_even_when_env_close_raises(monkeypatch, tmp_path):
    """A dead SubprocVecEnv worker makes env.close() raise inside finally -- the final
    model save must happen BEFORE close and survive it."""
    from types import SimpleNamespace
    from openrct2_gym.envs.openrct2_env import OpenRCT2Env
    monkeypatch.setattr(oe_mod, "APIController", FakeAPI)
    monkeypatch.setattr(OpenRCT2Env, "_CLOSE_CACHE_PATH", str(tmp_path / "close.json"))
    monkeypatch.setattr(OpenRCT2Env, "_LOOP_LIBRARY_PATH", str(tmp_path / "lib.jsonl"))
    monkeypatch.setattr(DummyVecEnv, "close", lambda self: (_ for _ in ()).throw(EOFError("dead worker")))

    saved = []

    class StubPPO:
        def __init__(self, *a, **k):
            self.gamma = T.GAMMA
            self._last_obs = object()

        @classmethod
        def load(cls, path, env=None):
            return cls()

        def learn(self, total_timesteps, callback=None, reset_num_timesteps=None):
            pass

        def save(self, path):
            saved.append(path)

        def get_vec_normalize_env(self):
            return None

        def get_env(self):
            return None

    monkeypatch.setattr(T, "MaskablePPO", StubPPO)
    T.train([8080], total_timesteps=8, checkpoint_freq=1000, eval_freq=0,
            verbose=0, disable_eval=True)                 # must not raise
    assert saved and saved[-1].endswith("final_model")


def test_eval_auto_disabled_for_multi_env(monkeypatch, tmp_path):
    """SubprocVecEnv eval cannot reach the wrappers (no stat suppression, scaffolded
    episodes measured, model._last_obs desync): eval must hard-disable at n_envs>1."""
    from openrct2_gym.envs.openrct2_env import OpenRCT2Env
    monkeypatch.setattr(oe_mod, "APIController", FakeAPI)
    monkeypatch.setattr(OpenRCT2Env, "_CLOSE_CACHE_PATH", str(tmp_path / "close.json"))
    monkeypatch.setattr(OpenRCT2Env, "_LOOP_LIBRARY_PATH", str(tmp_path / "lib.jsonl"))
    monkeypatch.setattr(T, "SubprocVecEnv", DummyVecEnv)  # run "2 workers" in-process
    evals = []
    monkeypatch.setattr(T, "evaluate_policy", lambda *a, **k: evals.append(1) or (0.0, 0.0))

    class StubPPO:
        def __init__(self, *a, **k):
            self.gamma = T.GAMMA
            self._last_obs = object()

        @classmethod
        def load(cls, path, env=None):
            return cls()

        def learn(self, total_timesteps, callback=None, reset_num_timesteps=None):
            pass

        def save(self, path):
            pass

        def get_vec_normalize_env(self):
            return None

        def get_env(self):
            return None

    monkeypatch.setattr(T, "MaskablePPO", StubPPO)
    T.train([8080, 8081], total_timesteps=16, checkpoint_freq=1000, eval_freq=8,
            verbose=0, eval_episodes=1, disable_eval=False)
    assert evals == []                                    # eval never ran


def test_single_env_eval_resets_last_obs(monkeypatch, tmp_path):
    """evaluate_policy steps the TRAINING env, leaving model._last_obs stale against the
    real env state; the next collect would pair pre-eval obs with post-eval env. Clearing
    _last_obs forces a clean reset at the next learn chunk."""
    from openrct2_gym.envs.openrct2_env import OpenRCT2Env
    monkeypatch.setattr(oe_mod, "APIController", FakeAPI)
    monkeypatch.setattr(OpenRCT2Env, "_CLOSE_CACHE_PATH", str(tmp_path / "close.json"))
    monkeypatch.setattr(OpenRCT2Env, "_LOOP_LIBRARY_PATH", str(tmp_path / "lib.jsonl"))
    monkeypatch.setattr(T, "evaluate_policy", lambda *a, **k: (0.0, 0.0))
    observed = []

    class StubPPO:
        def __init__(self, *a, **k):
            self.gamma = T.GAMMA
            self._last_obs = object()

        @classmethod
        def load(cls, path, env=None):
            return cls()

        def learn(self, total_timesteps, callback=None, reset_num_timesteps=None):
            observed.append(self._last_obs)               # what the next chunk would start from
            self._last_obs = object()

        def save(self, path):
            pass

        def get_vec_normalize_env(self):
            return None

        def get_env(self):
            return None

    monkeypatch.setattr(T, "MaskablePPO", StubPPO)
    T.train([8080], total_timesteps=16, checkpoint_freq=1000, eval_freq=8,
            verbose=0, eval_episodes=1, disable_eval=False)
    assert len(observed) == 2
    assert observed[1] is None                            # post-eval chunk starts clean


def test_kl_guard_arms_on_fleet_majority_not_first_worker():
    """Per-worker curricula desync: arming target_kl the moment ONE worker touches phase 2
    throttles the +1000-spike exploitation every other worker's phase-1 bootstrap depends
    on. The guard arms only when the majority of workers are past phase 1."""
    from types import SimpleNamespace
    cb = T.ParallelCurriculumMaskableCallback(n_envs=3)
    cb.model = SimpleNamespace(target_kl=None, ent_coef=0.01)
    cb._note_env_phase(0, {'learning_phase': 2, 'phase2_stage': 1})
    cb._maybe_arm_kl_guard({'learning_phase': 2})
    assert cb.model.target_kl is None                     # 1 of 3: no arm
    cb._note_env_phase(1, {'learning_phase': 2, 'phase2_stage': 1})
    cb._maybe_arm_kl_guard({'learning_phase': 2})
    assert cb.model.target_kl == T.OPT_GUARDED['target_kl']   # 2 of 3: majority -> arm
    assert cb._phase == 2                                 # fleet phase follows the majority


def test_resumed_model_with_armed_guard_is_recognized():
    """Resume rebuilds the callback fresh (bootstrap floor 0.025) while the LOADED model
    still carries target_kl=0.04 -- the destructive high-entropy+tight-KL combination.
    The callback must recognize an already-armed guard at the first rollout end."""
    from types import SimpleNamespace
    cb = _fresh_cb()
    cb.model.target_kl = 0.04                             # loaded from checkpoint
    cb.model.logger = SimpleNamespace(name_to_value={}, record=lambda *a, **k: None)
    cb._on_rollout_end()
    assert cb._opt_guarded is True
    assert cb.model.ent_coef == T.OPT_GUARDED['ent_coef'] # guarded floor, not bootstrap 0.025
