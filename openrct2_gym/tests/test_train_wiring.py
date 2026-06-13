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

import train_parallel_curriculum_masked as T


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
        DummyVecEnv([lambda: T.create_curriculum_masked_env(8080, use_improved=True, verbose=0)]),
        norm_obs=True, norm_reward=False, norm_obs_keys=["scalars"],
    )


def test_vecnormalize_path_convention():
    assert T._vecnormalize_path("a/b/final_model.zip") == "a/b/final_model_vecnormalize.pkl"
    assert T._vecnormalize_path("a/b/final_model") == "a/b/final_model_vecnormalize.pkl"


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
        DummyVecEnv([lambda: T.create_curriculum_masked_env(8080, use_improved=True, verbose=0)]),
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
    # the guard the callback arms for phases >= 2 touches ONLY target_kl: a doubled
    # ent_coef destabilized the converged completion policy (entropy exploded 0.03->1.5,
    # completion collapsed), so entropy stays at the proven 0.01 in every phase.
    assert T.OPT_GUARDED == {"target_kl": 0.04}

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


def test_callback_arms_kl_guard_when_phase2_begins():
    """The phase-2 transition is where the KL=2.49 catastrophe happened, so the guard
    (target_kl + raised ent_coef) must arm exactly when the curriculum reaches phase 2 --
    and stay armed (one-way switch, phases never go backward)."""
    from types import SimpleNamespace
    cb = T.ParallelCurriculumMaskableCallback(n_envs=2)
    cb.model = SimpleNamespace(target_kl=None, ent_coef=0.01)

    cb._maybe_arm_kl_guard({})                          # no phase info -> no-op
    cb._maybe_arm_kl_guard({'learning_phase': 1})       # phase 1 -> stays in bootstrap config
    assert cb.model.target_kl is None
    assert cb.model.ent_coef == 0.01

    cb._maybe_arm_kl_guard({'learning_phase': 2})       # phase 2 -> guard arms
    assert cb.model.target_kl == 0.04
    assert cb.model.ent_coef == 0.01                    # entropy UNCHANGED (no longer doubled)

    cb.model.target_kl = 0.99                           # one-way: arming never re-fires
    cb._maybe_arm_kl_guard({'learning_phase': 3})
    cb._maybe_arm_kl_guard({'learning_phase': 1})
    assert cb.model.target_kl == 0.99


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
