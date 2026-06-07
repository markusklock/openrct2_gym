"""Shared configuration for the redesigned observation space.

Kept in one place so the environment (which *builds* observations) and the feature
extractor (which *consumes* them) agree on shapes and constants, and so tests can
construct the space without a live OpenRCT2 connection.
"""
import numpy as np
import gymnasium as gym

# --- sequence (build-history) buffer ---------------------------------------------
SEQ_LEN = 128            # last N placed pieces kept in the buffer (covers phase-5 target 120)
NUM_ACTIONS = 32         # discrete actions 0..31 (31 = remove)
TOKEN_VOCAB = NUM_ACTIONS + 1   # +1 for PAD at index 0; real tokens are action_index+1 in 1..32
HIST_FEAT_DIM = 11       # per-piece geometry features (see openrct2_env._build_history_buffer)

# --- egocentric 2.5D map ----------------------------------------------------------
MAP_CHANNELS = 4         # [occupancy, signed-height, goal/station marker, chain-lift/path]
MAP_SIZE = 24            # H == W of the egocentric, forward-aligned crop
MAP_SHAPE = (MAP_CHANNELS, MAP_SIZE, MAP_SIZE)

# --- normalization scales (tunable) ----------------------------------------------
SCALE = 100.0            # horizontal/relative displacement normalizer (tiles)
H_SCALE = 30.0           # vertical relief normalizer (tiles), signed in [-1, 1]

# --- categorical sizes ------------------------------------------------------------
DIRECTION_N = 4          # current_direction (cardinal)
LAST_PIECE_N = NUM_ACTIONS + 1   # last_piece_type, +1 shifted (0 = none yet)


def make_observation_space() -> gym.spaces.Dict:
    """Construct the redesigned Dict observation space.

    Every Box value is clipped to these bounds by the env's builder helpers so the
    bounds-validation in _get_observation never raises. Categorical fields stay
    Discrete so SB3 one-hot-encodes them automatically.
    """
    return gym.spaces.Dict({
        "local_map": gym.spaces.Box(low=-1.0, high=1.0, shape=MAP_SHAPE, dtype=np.float32),
        "build_history_tokens": gym.spaces.Box(low=0, high=NUM_ACTIONS, shape=(SEQ_LEN,), dtype=np.int32),
        "build_history_feats": gym.spaces.Box(low=-1.0, high=1.0, shape=(SEQ_LEN, HIST_FEAT_DIM), dtype=np.float32),
        "build_history_mask": gym.spaces.Box(low=0.0, high=1.0, shape=(SEQ_LEN,), dtype=np.float32),
        "goal_disp": gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        "goal_direction3": gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        "scalars": gym.spaces.Box(low=-10.0, high=10.0, shape=(8,), dtype=np.float32),
        "current_direction": gym.spaces.Discrete(DIRECTION_N),
        "last_piece_type": gym.spaces.Discrete(LAST_PIECE_N),
    })
