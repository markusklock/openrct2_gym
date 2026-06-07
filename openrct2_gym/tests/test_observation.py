"""Phase 1 tests: the env's observation builders (_ego_rotate, _build_local_map,
_build_history_buffer) and the rewritten _get_observation.

Server-free: the env is built via __new__ and its internal state set by hand. The
builders are pure functions of internal state (no API calls).
"""
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from openrct2_gym.envs.openrct2_env import OpenRCT2Env
from openrct2_gym.envs.obs_config import (
    make_observation_space, SEQ_LEN, HIST_FEAT_DIM, MAP_SHAPE,
)

DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # N, E, S, W


def _entry(action, pos, nxt, direction=0, next_direction=0, is_complete=False):
    return {
        "action": action, "position": list(pos), "next_position": list(nxt),
        "direction": direction, "next_direction": next_direction,
        "track_type": 0, "is_complete": is_complete,
    }


def _bare_env(history=None, current_position=(61, 70, 14), current_direction=1,
              goal_position=(62, 66, 14)):
    env = OpenRCT2Env.__new__(OpenRCT2Env)
    env.direction_vectors = DIRS
    env.track_builder = SimpleNamespace(history=list(history or []))
    env.current_position = list(current_position)
    env.current_direction = current_direction
    env.goal_position = list(goal_position)
    env.station_start_position = [61, 66, 14]
    env.station_length = 6
    env.track_length = len(env.track_builder.history)
    env.max_track_length = 250
    env.last_piece_type = env.track_builder.history[-1]["action"] if env.track_builder.history else 0
    env.last_ride_excitement = 0.0
    env.last_ride_intensity = 0.0
    env.last_ride_nausea = 0.0
    env.position_history = deque([list(current_position)], maxlen=OpenRCT2Env.POSITION_HISTORY_MAXLEN)
    env.observation_space = make_observation_space()
    return env


# --------------------------------------------------------------------- ego_rotate

def test_ego_rotate_is_rotation_invariant():
    """A tile 2-forward + 1-right of the head maps to the same ego cell for any heading."""
    env = _bare_env()
    env.current_position = [10, 10, 14]
    for d in range(4):
        env.current_direction = d
        fwd, right = DIRS[d], DIRS[(d + 1) % 4]
        wx = 2 * fwd[0] + 1 * right[0]
        wy = 2 * fwd[1] + 1 * right[1]
        er, ef = env._ego_rotate(wx, wy)
        assert (round(er), round(ef)) == (1, 2)


# ----------------------------------------------------------------- history buffer

def test_history_buffer_shapes_and_token_shift():
    hist = [_entry(0, (61, 70, 14), (61, 71, 15)),    # action 0 (flat) -> token 1
            _entry(9, (61, 71, 15), (61, 72, 16))]    # chain lift -> token 10
    env = _bare_env(history=hist, current_position=(61, 72, 16))
    tokens, feats, mask = env._build_history_buffer()

    assert tokens.shape == (SEQ_LEN,)
    assert feats.shape == (SEQ_LEN, HIST_FEAT_DIM)
    assert mask.shape == (SEQ_LEN,)
    # oldest -> newest, right-padded
    assert tokens[0] == 1 and tokens[1] == 10
    assert np.all(tokens[2:] == 0)
    assert mask[0] == 1 and mask[1] == 1 and np.all(mask[2:] == 0)
    # pad rows are zero; real feats finite & in [-1, 1]
    assert np.all(feats[2:] == 0)
    assert np.all(np.isfinite(feats))
    assert feats.min() >= -1.0 and feats.max() <= 1.0


def test_history_buffer_marks_chain_lift_feature():
    hist = [_entry(9, (61, 70, 14), (61, 71, 15))]
    env = _bare_env(history=hist, current_position=(61, 71, 15))
    _, feats, _ = env._build_history_buffer()
    # is_chain_lift is the second-to-last feature column (see _build_history_buffer)
    assert feats[0, -2] == pytest.approx(1.0)


def test_history_buffer_truncates_to_seq_len():
    hist = [_entry(0, (61, 70 + i, 14), (61, 71 + i, 14)) for i in range(SEQ_LEN + 20)]
    env = _bare_env(history=hist, current_position=(61, 70 + SEQ_LEN + 20, 14))
    tokens, _, mask = env._build_history_buffer()
    assert mask.sum() == SEQ_LEN          # full window, no padding
    assert np.all(tokens > 0)


# -------------------------------------------------------------------- local map

def test_local_map_shape_and_bounds():
    hist = [_entry(0, (61, 70, 14), (61, 71, 14))]
    env = _bare_env(history=hist, current_position=(61, 71, 14))
    m = env._build_local_map()
    assert m.shape == MAP_SHAPE
    assert m.dtype == np.float32
    assert m.min() >= -1.0 and m.max() <= 1.0


def test_local_map_head_cell_occupied():
    # head is the endpoint of the last piece -> centre cell occupied
    hist = [_entry(0, (61, 70, 14), (61, 71, 14))]
    env = _bare_env(history=hist, current_position=(61, 71, 14), current_direction=0)
    m = env._build_local_map()
    c = MAP_SHAPE[1] // 2
    assert m[0, c, c] == 1.0          # occupancy channel at centre


def test_local_map_signed_height_below_station_is_negative():
    # a piece whose endpoint is below station height must read negative on the height channel
    hist = [_entry(6, (61, 71, 14), (61, 72, 8))]
    env = _bare_env(history=hist, current_position=(61, 72, 8), current_direction=0)
    m = env._build_local_map()
    c = MAP_SHAPE[1] // 2
    assert m[1, c, c] < 0.0


def test_local_map_rotation_invariance():
    """The track contribution is identical at any heading once expressed egocentrically.

    Station and goal are pushed out of the window (they are absolute-world structures,
    not part of the rotated track), isolating the placed-track piece, which arrives from
    directly behind the head in both scenes.
    """
    far = [9000, 9000, 14]
    # scenario A: head facing East; piece arrives from the west (behind)
    envA = _bare_env(history=[_entry(0, (19, 20, 14), (20, 20, 14))],
                     current_position=(20, 20, 14), current_direction=1, goal_position=far)
    envA.station_start_position = list(far)
    mA = envA._build_local_map()

    # scenario B: head facing North; piece arrives from the south (behind) - a 90deg rotation
    envB = _bare_env(history=[_entry(0, (20, 19, 14), (20, 20, 14))],
                     current_position=(20, 20, 14), current_direction=0, goal_position=far)
    envB.station_start_position = list(far)
    mB = envB._build_local_map()
    assert np.allclose(mA, mB)


# ----------------------------------------------------------------- get_observation

def test_get_observation_is_contained_in_space():
    hist = [_entry(9, (61, 70, 14), (61, 71, 15)),
            _entry(6, (61, 71, 15), (61, 72, 10), is_complete=False)]
    env = _bare_env(history=hist, current_position=(61, 72, 10))
    obs = env._get_observation()
    assert env.observation_space.contains(obs)


def test_get_observation_last_piece_type_shift_and_sentinel():
    # empty build -> sentinel 0
    env_empty = _bare_env(history=[], current_position=(61, 72, 14))
    assert int(env_empty._get_observation()["last_piece_type"]) == 0
    # after an action-9 piece -> token 10 (action + 1)
    hist = [_entry(9, (61, 70, 14), (61, 71, 15))]
    env = _bare_env(history=hist, current_position=(61, 71, 15))
    assert int(env._get_observation()["last_piece_type"]) == 10
