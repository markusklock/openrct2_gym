"""Phase 0 tests: the new observation space + BuildHistoryExtractor.

All server-free. The extractor is fed tensors in SB3 *post-preprocessing* format
(Discrete keys already one-hot, every key a float tensor) the same way SB3's
``extract_features`` would, so these tests exercise the real forward path.
"""
import gymnasium as gym
import numpy as np
import torch as th
import pytest

from openrct2_gym.envs.obs_config import (
    make_observation_space,
    SEQ_LEN,
    MAP_SHAPE,
    HIST_FEAT_DIM,
    NUM_ACTIONS,
)
from openrct2_gym.envs.feature_extractor import BuildHistoryExtractor


EXPECTED_FEATURES_DIM = 352


def _make_batch(batch=2, lengths=(5, 5), corrupt_pad=False):
    """Build a post-preprocessing observation batch as torch tensors.

    ``lengths`` gives the number of real (unpadded) pieces per sample; the history is
    right-padded (real pieces at indices 0..L-1, padding after).
    """
    rng = np.random.default_rng(0)
    tokens = np.zeros((batch, SEQ_LEN), dtype=np.float32)
    feats = np.zeros((batch, SEQ_LEN, HIST_FEAT_DIM), dtype=np.float32)
    mask = np.zeros((batch, SEQ_LEN), dtype=np.float32)
    for b, L in enumerate(lengths):
        if L > 0:
            tokens[b, :L] = rng.integers(1, NUM_ACTIONS + 1, size=L)
            feats[b, :L] = rng.uniform(-1, 1, size=(L, HIST_FEAT_DIM))
            mask[b, :L] = 1.0

    obs = {
        "local_map": th.as_tensor(rng.uniform(-1, 1, size=(batch, *MAP_SHAPE)), dtype=th.float32),
        "build_history_tokens": th.as_tensor(tokens),
        "build_history_feats": th.as_tensor(feats),
        "build_history_mask": th.as_tensor(mask),
        "goal_disp": th.as_tensor(rng.uniform(-1, 1, size=(batch, 3)), dtype=th.float32),
        "goal_direction3": th.as_tensor(rng.uniform(-1, 1, size=(batch, 3)), dtype=th.float32),
        "scalars": th.as_tensor(rng.uniform(-1, 1, size=(batch, 8)), dtype=th.float32),
        # Discrete keys arrive one-hot (float) after SB3 preprocessing
        "current_direction": th.as_tensor(np.eye(4, dtype=np.float32)[rng.integers(0, 4, batch)]),
        "last_piece_type": th.as_tensor(np.eye(33, dtype=np.float32)[rng.integers(0, 33, batch)]),
    }
    if corrupt_pad:
        # Garbage in the padded region only, drawn from a SEPARATE rng so the shared
        # (real-history + map + scalar) inputs are byte-identical to the clean batch.
        # mask stays 0 there, so a correct extractor must ignore it entirely.
        pad_rng = np.random.default_rng(123)
        for b, L in enumerate(lengths):
            if L < SEQ_LEN:
                obs["build_history_tokens"][b, L:] = th.as_tensor(
                    pad_rng.integers(1, NUM_ACTIONS + 1, size=SEQ_LEN - L), dtype=th.float32)
                obs["build_history_feats"][b, L:] = th.as_tensor(
                    pad_rng.uniform(-1, 1, size=(SEQ_LEN - L, HIST_FEAT_DIM)), dtype=th.float32)
    return obs


def test_make_observation_space_shapes():
    space = make_observation_space()
    assert space["local_map"].shape == MAP_SHAPE
    assert space["build_history_tokens"].shape == (SEQ_LEN,)
    assert space["build_history_feats"].shape == (SEQ_LEN, HIST_FEAT_DIM)
    assert space["build_history_mask"].shape == (SEQ_LEN,)
    assert space["goal_disp"].shape == (3,)
    assert space["goal_direction3"].shape == (3,)
    assert space["scalars"].shape == (8,)
    assert isinstance(space["current_direction"], gym.spaces.Discrete)
    assert space["current_direction"].n == 4
    assert isinstance(space["last_piece_type"], gym.spaces.Discrete)
    assert space["last_piece_type"].n == 33


def test_observation_space_sample_is_contained():
    space = make_observation_space()
    for _ in range(5):
        assert space.contains(space.sample())


def test_extractor_features_dim():
    extractor = BuildHistoryExtractor(make_observation_space())
    assert extractor.features_dim == EXPECTED_FEATURES_DIM


def test_extractor_forward_shape():
    extractor = BuildHistoryExtractor(make_observation_space()).eval()
    out = extractor(_make_batch(batch=3, lengths=(0, 4, SEQ_LEN)))
    assert out.shape == (3, EXPECTED_FEATURES_DIM)
    assert th.isfinite(out).all()


def test_padding_embedding_is_zero_and_frozen():
    extractor = BuildHistoryExtractor(make_observation_space())
    assert th.allclose(extractor.token_embed.weight[0], th.zeros(extractor.token_embed.embedding_dim))

    out = extractor(_make_batch(batch=2, lengths=(3, 6)))
    out.sum().backward()
    # padding_idx row must remain zero (its gradient is frozen)
    assert th.allclose(extractor.token_embed.weight[0], th.zeros(extractor.token_embed.embedding_dim))


def test_history_readout_ignores_padding():
    """Corrupting the padded region must not change the encoding (pad-length invariance)."""
    extractor = BuildHistoryExtractor(make_observation_space()).eval()
    clean = _make_batch(batch=2, lengths=(5, 9), corrupt_pad=False)
    dirty = _make_batch(batch=2, lengths=(5, 9), corrupt_pad=True)
    with th.no_grad():
        out_clean = extractor(clean)
        out_dirty = extractor(dirty)
    assert th.allclose(out_clean, out_dirty, atol=1e-6)


def test_empty_build_readout_is_finite():
    extractor = BuildHistoryExtractor(make_observation_space()).eval()
    with th.no_grad():
        out = extractor(_make_batch(batch=2, lengths=(0, 0)))
    assert th.isfinite(out).all()


def test_transformer_encoder_forward_shape():
    extractor = BuildHistoryExtractor(make_observation_space(), encoder="transformer").eval()
    with th.no_grad():
        out = extractor(_make_batch(batch=2, lengths=(0, 7)))
    assert out.shape == (2, EXPECTED_FEATURES_DIM)
    assert th.isfinite(out).all()


def test_maskable_ppo_constructs_and_learns():
    """Full server-free wiring: stub env -> ActionMasker -> DummyVecEnv -> MaskablePPO."""
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.vec_env import DummyVecEnv

    space = make_observation_space()

    class _StubEnv(gym.Env):
        def __init__(self):
            self.observation_space = space
            self.action_space = gym.spaces.Discrete(NUM_ACTIONS)
            self._n = 0

        def reset(self, *, seed=None, options=None):
            self._n = 0
            return self.observation_space.sample(), {}

        def step(self, action):
            self._n += 1
            return self.observation_space.sample(), 0.0, self._n >= 4, False, {}

        def valid_action_mask(self):
            m = np.zeros(NUM_ACTIONS, dtype=bool)
            m[:5] = True
            return m

    def _mask_fn(env):
        return env.valid_action_mask()

    venv = DummyVecEnv([lambda: ActionMasker(_StubEnv(), _mask_fn)])
    model = MaskablePPO(
        "MultiInputPolicy",
        venv,
        policy_kwargs=dict(
            features_extractor_class=BuildHistoryExtractor,
            features_extractor_kwargs=dict(encoder="gru"),
            net_arch=dict(pi=[64], vf=[64]),
            normalize_images=False,
        ),
        n_steps=16,
        batch_size=16,
        n_epochs=1,
        verbose=0,
    )
    assert model.policy.features_extractor.features_dim == EXPECTED_FEATURES_DIM
    model.learn(total_timesteps=16)
