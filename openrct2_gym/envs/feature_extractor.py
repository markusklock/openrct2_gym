"""Custom multi-modal SB3 feature extractor for the OpenRCT2 coaster builder.

Fuses four branches into one feature vector, fully compatible with MaskablePPO's
``MaskableMultiInputActorCriticPolicy`` (masking acts on action logits downstream, so
the extractor never touches it):

  A) build-history buffer  -> piece-type embedding + per-piece geometry -> GRU
     (or a Transformer encoder), read out the last *real* timestep. This is the
     agent's memory.
  B) egocentric 2.5D map   -> small 2D CNN. This is the agent's 3D spatial awareness.
  C) continuous scalars    -> MLP (goal displacement/direction + scalar features).
  D) categoricals          -> MLP over the SB3-provided one-hots.

Key correctness points (verified against SB3 source during design):
  * SB3 preprocesses every observation before the extractor: Box -> float, Discrete ->
    one-hot float. So token ids arrive as floats and MUST be cast to long for
    ``nn.Embedding``; Discrete keys arrive already one-hot.
  * PAD is token id 0 with ``padding_idx=0`` (its embedding row is zero and frozen), so
    real actions are stored as ``action_index + 1``.
  * The GRU uses ``pack_padded_sequence`` with lengths from the validity mask, so padded
    rows are never consumed -> the encoding depends only on the real history, regardless
    of how much padding follows (a frozen-zero embedding alone does NOT guarantee this,
    because the per-piece projection's bias makes pad rows non-zero).
"""
from typing import Dict

import gymnasium as gym
import torch as th
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .obs_config import SEQ_LEN, TOKEN_VOCAB, HIST_FEAT_DIM, MAP_CHANNELS


class BuildHistoryExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        token_embed_dim: int = 16,
        proj_dim: int = 64,
        gru_hidden: int = 128,
        cnn_out: int = 128,
        scalar_hidden: int = 64,
        cat_hidden: int = 32,
        encoder: str = "gru",
        n_heads: int = 4,
        n_transformer_layers: int = 1,
    ):
        assert encoder in ("gru", "transformer"), f"unknown encoder {encoder!r}"
        features_dim = gru_hidden + cnn_out + scalar_hidden + cat_hidden
        super().__init__(observation_space, features_dim=features_dim)
        self.encoder = encoder
        self.gru_hidden = gru_hidden

        # --- Branch A: build-history buffer -----------------------------------------
        self.token_embed = nn.Embedding(TOKEN_VOCAB, token_embed_dim, padding_idx=0)
        self.piece_proj = nn.Sequential(
            nn.Linear(token_embed_dim + HIST_FEAT_DIM, proj_dim), nn.ReLU()
        )
        if encoder == "gru":
            self.gru = nn.GRU(proj_dim, gru_hidden, batch_first=True)
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=proj_dim, nhead=n_heads, batch_first=True, dropout=0.0
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=n_transformer_layers)
            self.seq_out = nn.Linear(proj_dim, gru_hidden)

        # --- Branch B: egocentric 2.5D map CNN --------------------------------------
        self.cnn = nn.Sequential(
            nn.Conv2d(MAP_CHANNELS, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4), nn.Flatten(),
            nn.Linear(32 * 4 * 4, cnn_out), nn.ReLU(),
        )

        # --- Branch C: continuous scalars -------------------------------------------
        self._scalar_keys = ["goal_disp", "goal_direction3", "scalars"]
        scalar_in = sum(int(observation_space[k].shape[0]) for k in self._scalar_keys)
        self.scalar_mlp = nn.Sequential(nn.Linear(scalar_in, scalar_hidden), nn.ReLU())

        # --- Branch D: categoricals (SB3 one-hots Discrete to n floats) -------------
        self._cat_keys = ["current_direction", "last_piece_type"]
        cat_in = sum(int(observation_space[k].n) for k in self._cat_keys)
        self.cat_mlp = nn.Sequential(nn.Linear(cat_in, cat_hidden), nn.ReLU())

    def _encode_history(self, obs: Dict[str, th.Tensor]) -> th.Tensor:
        tokens = obs["build_history_tokens"].long()          # (B, L) - cast: arrives as float
        feats = obs["build_history_feats"].float()           # (B, L, F)
        mask = obs["build_history_mask"].float()             # (B, L)

        emb = self.token_embed(tokens)                       # (B, L, E)
        x = self.piece_proj(th.cat([emb, feats], dim=-1))    # (B, L, proj_dim)

        lengths = mask.sum(dim=1).long()                     # (B,)
        valid = lengths > 0

        if self.encoder == "transformer":
            pad_mask = mask == 0                             # True where PAD (ignored by attn)
            enc = self.transformer(x, src_key_padding_mask=pad_mask)   # (B, L, proj_dim)
            m = mask.unsqueeze(-1)
            pooled = (enc * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)  # masked mean
            return self.seq_out(pooled) * valid.unsqueeze(-1).float()

        # GRU: pack so PAD rows are never consumed. Clamp lengths to >=1 for packing,
        # then zero the readout for genuinely empty builds.
        safe_lengths = lengths.clamp(min=1).cpu()
        packed = pack_padded_sequence(x, safe_lengths, batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)                            # h_n: (1, B, gru_hidden)
        readout = h_n.squeeze(0)                             # (B, gru_hidden)
        return readout * valid.unsqueeze(-1).float()

    @staticmethod
    def _flat(t: th.Tensor) -> th.Tensor:
        # SB3 one-hots Discrete keys to (B, 1, n) in the VecEnv path; flatten to (B, n).
        return t.float().reshape(t.shape[0], -1)

    def forward(self, obs: Dict[str, th.Tensor]) -> th.Tensor:
        hist = self._encode_history(obs)
        spatial = self.cnn(obs["local_map"].float())
        scalars = self.scalar_mlp(th.cat([self._flat(obs[k]) for k in self._scalar_keys], dim=1))
        cats = self.cat_mlp(th.cat([self._flat(obs[k]) for k in self._cat_keys], dim=1))
        return th.cat([hist, spatial, scalars, cats], dim=1)
