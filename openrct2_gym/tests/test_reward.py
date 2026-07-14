"""Reward-system overhaul tests (PBRS + unified parametrized reward).

Server-free: the env is built via ``__new__`` and its internal state set by hand,
or driven through a ``FakeAPI`` (mirroring test_env_smoke.py). Reward math is a pure
function of internal state (no API calls), so these run without an OpenRCT2 server.

Note: ``FakeAPI`` hardcodes ``isCircuitComplete=False``, so completion / terminal-Phi /
completion-first tests use the ``__new__`` + hand-set ``loop_completed`` path.
"""
from collections import deque
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from openrct2_gym.envs import openrct2_env as oe_mod
from openrct2_gym.envs.openrct2_env import OpenRCT2Env, RewardParams
from openrct2_gym.envs.obs_config import make_observation_space, SCALE, H_SCALE
from openrct2_gym.tests.test_env_smoke import FakeAPI

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # API encoding: 0=W, 1=N, 2=E, 3=S (matches env.direction_vectors)


@pytest.fixture(autouse=True)
def _isolate_close_cache(tmp_path):
    """Isolate the process-wide calibration cache + record buffer + its file per test,
    and the warm-start loop-library file (completions harvest into it; the curriculum
    wrapper's default library path follows the same class attr)."""
    orig_cache = OpenRCT2Env._close_cache
    orig_path = OpenRCT2Env._CLOSE_CACHE_PATH
    orig_records = OpenRCT2Env._close_records
    orig_library = OpenRCT2Env._LOOP_LIBRARY_PATH
    OpenRCT2Env._close_cache = None
    OpenRCT2Env._close_records = []
    OpenRCT2Env._CLOSE_CACHE_PATH = str(tmp_path / "close_geometry.json")
    OpenRCT2Env._LOOP_LIBRARY_PATH = str(tmp_path / "loop_library.jsonl")
    yield
    OpenRCT2Env._close_cache = orig_cache
    OpenRCT2Env._close_records = orig_records
    OpenRCT2Env._CLOSE_CACHE_PATH = orig_path
    OpenRCT2Env._LOOP_LIBRARY_PATH = orig_library


def _bare_env(current_position=(61, 70, 14), current_direction=1,
              goal_position=(62, 66, 14), history=None):
    """An OpenRCT2Env without __init__ (no API), with the state reward math needs."""
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
    env.reward_params = RewardParams()
    return env


# ----------------------------------------------------------------- RewardParams

def test_reward_params_defaults_support_completion_first():
    p = RewardParams()
    assert p.gamma == 0.99
    assert p.R_complete == 1000.0
    assert p.R_quality_max == 0.0          # quality off by default (phases 1-4)
    assert p.fail_penalty == -0.1
    assert p.step_cost == 0.0
    # Any incomplete episode's return is bounded by Phi_max (incl. the discovery term);
    # completion strictly dominates it.
    phi_max = p.w_xy + p.w_z + p.w_dir + p.w_e + p.w_h
    assert p.R_complete > phi_max
    # R_struct / R_quality are completion-ONLY bonuses (not part of the non-completion bound),
    # each bounded below R_complete -> a hill completion (R_complete + struct) beats a flat
    # completion (R_complete) without ever letting an incomplete episode win.
    assert 0.0 < 250.0 < p.R_complete          # the P2-4 R_struct_max
    assert p.R_quality_max == 0.0 and 500.0 < p.R_complete


def test_reward_params_is_frozen():
    p = RewardParams()
    with pytest.raises(Exception):
        p.w_xy = 1.0


# ------------------------------------------------------------- reward target getter

def test_reward_target_defaults_to_goal_position():
    env = _bare_env(goal_position=(62, 66, 14))
    assert list(env._reward_target_position()) == [62, 66, 14]
    assert env._reward_target_direction() is None


def test_reward_target_uses_calibration_when_set():
    env = _bare_env(goal_position=(62, 66, 14))
    env.close_pos = [10, 20, 14]
    env.close_dir = 3
    assert list(env._reward_target_position()) == [10, 20, 14]
    assert env._reward_target_direction() == 3


# --------------------------------------------- obs follows reward target (test 16)

def test_observation_goal_disp_follows_calibrated_target():
    """After calibration, goal_disp must point at close_pos, not goal_position."""
    env = _bare_env(current_position=(61, 70, 14), current_direction=1,
                    goal_position=(62, 66, 14))
    env.close_pos = [61, 60, 14]   # distinct from goal_position
    env.close_dir = 3

    gdx, gdy, gdz = (61 - 61), (60 - 70), (14 - 14)
    er, ef = env._ego_rotate(gdx, gdy)
    expected = np.clip(np.array([er / SCALE, ef / SCALE, gdz / H_SCALE], dtype=np.float32), -1.0, 1.0)

    obs = env._get_observation()
    assert np.allclose(obs["goal_disp"], expected)

    # sanity: a goal_position-based disp would differ
    ggx, ggy = (62 - 61), (66 - 70)
    ger, gef = env._ego_rotate(ggx, ggy)
    goal_based = np.clip(np.array([ger / SCALE, gef / SCALE, 0.0], dtype=np.float32), -1.0, 1.0)
    assert not np.allclose(obs["goal_disp"], goal_based)


def test_distance_and_energy_margin_follow_calibrated_target():
    env = _bare_env(current_position=(2, 0, 14), goal_position=(99, 99, 14),
                    history=[{"action": 9, "position": [0, 0, 14], "next_position": [1, 0, 15]},
                             {"action": 6, "position": [1, 0, 15], "next_position": [2, 0, 14]}])
    env.close_pos = [2, 0, 14]     # head is exactly at the calibrated target
    env.close_dir = 1
    assert float(env._calculate_distance_to_start()[0]) == pytest.approx(0.0)
    # margin uses the corrected (history) energy and the calibrated target distance (0)
    assert env._calculate_energy_margin() == pytest.approx(49.0)


# --------------------------------------------------------------------- Phi (test 1)

_DROP_HIST = [{"action": 9, "position": [0, 0, 14], "next_position": [1, 0, 15]},
              {"action": 6, "position": [1, 0, 15], "next_position": [2, 0, 14]}]


def _phi_env(pos, direction=1, close_pos=(0, 0, 14), close_dir=1):
    env = _bare_env(current_position=pos, current_direction=direction,
                    history=[dict(h) for h in _DROP_HIST])
    env.close_pos = list(close_pos)
    env.close_dir = close_dir
    return env


def test_phi_increases_as_head_approaches_target_xy():
    # Approach ALONG the entry corridor (close_dir=1 -> entry +Y, so the approach side is -Y).
    # The horizontal w_xy pull is now directional, so it rewards closing in along this axis.
    geo = RewardParams(w_e=0.0)  # isolate geometry
    phis = [_phi_env((0, -d, 14))._potential(geo) for d in (30, 20, 10, 0)]
    assert phis == sorted(phis)              # monotonically increasing
    assert phis[-1] > phis[0]


def test_approach_is_directional_no_wrong_side_pull():
    """The horizontal w_xy pull is directional: at equal distance, approaching ALONG the entry
    corridor from the -entry side earns the pull, while the wrong side (behind the dock) and the
    far off-axis earn ZERO -- so the head can't minimise distance by parking behind/beside the
    station. (close_dir=1 -> entry +Y; approach side -Y; off-axis is X.)"""
    geo = RewardParams(w_xy=10.0, w_z=0.0, w_dir=0.0, w_e=0.0, w_h=0.0, w_close=0.0)
    on_corridor = _phi_env((0, -2, 14))._potential(geo)   # along=2 on-axis  -> strong pull
    wrong_side  = _phi_env((0,  2, 14))._potential(geo)    # along=-2 behind dock -> no pull
    far_off     = _phi_env((10, 0, 14))._potential(geo)    # along=0, perp=10 (cone tol=2) -> no pull
    assert on_corridor > 0.0
    assert wrong_side == pytest.approx(0.0)
    assert far_off == pytest.approx(0.0)
    assert on_corridor > wrong_side


def test_approach_cone_widens_away_from_dock():
    """The approach corridor is a cone: an off-axis tile far from the dock is inside it (caught as
    the head rounds back), while the same perp offset close to the dock is outside it."""
    geo = RewardParams(w_xy=10.0, w_z=0.0, w_dir=0.0, w_e=0.0, w_h=0.0, w_close=0.0)
    far_offaxis  = _phi_env((3, -8, 14))._potential(geo)   # along=8, perp=3 < tol(2+8=10) -> inside
    near_offaxis = _phi_env((3, -1, 14))._potential(geo)   # along=1, perp=3 > tol(2+1=3)? equal->0
    assert far_offaxis > 0.0
    assert near_offaxis == pytest.approx(0.0)


def test_approach_directional_disabled_restores_isotropic():
    """approach_perp_range=0 restores the legacy isotropic radial w_xy pull (wrong side no longer
    zeroed -- equal distance gives equal pull regardless of side)."""
    geo = replace(RewardParams(w_z=0.0, w_dir=0.0, w_e=0.0, w_h=0.0, w_close=0.0),
                  approach_perp_range=0.0)
    approach_side = _phi_env((0, -2, 14))._potential(geo)
    wrong_side    = _phi_env((0,  2, 14))._potential(geo)
    assert approach_side == pytest.approx(wrong_side)
    assert wrong_side > 0.0


def test_phi_increases_as_head_approaches_station_height():
    geo = RewardParams(w_e=0.0)
    phis = [_phi_env((0, 0, z))._potential(geo) for z in (34, 24, 18, 14)]
    assert phis == sorted(phis)
    assert phis[-1] > phis[0]


def test_phi_increases_as_heading_aligns_with_close_dir():
    geo = RewardParams(w_e=0.0)
    reversed_ = _phi_env((0, 0, 14), direction=3)._potential(geo)   # West vs close East
    perp = _phi_env((0, 0, 14), direction=0)._potential(geo)        # North
    aligned = _phi_env((0, 0, 14), direction=1)._potential(geo)     # East == close_dir
    assert reversed_ < perp < aligned


def test_heading_term_is_coupled_to_the_dock():
    """Tier-2.1: the closing-heading reward is gated by near_xy, so it only matters AS the head
    docks (within close_range), leaving the agent free to turn while routing the loop. AT the dock
    aligned vs opposed differ by the full w_dir; FAR from it heading is irrelevant (a curved
    closing approach arrives a few tiles out heading some non-entry direction -- it must not be
    penalised there). Isolated to the heading term so the gating is unambiguous."""
    geo = replace(RewardParams(), w_xy=0.0, w_z=0.0, w_e=0.0, w_h=0.0, w_close=0.0,
                  w_return=0.0, w_dir=6.0, close_range=3.0)
    at_aligned = _phi_env((0, 0, 14), direction=1, close_dir=1)._potential(geo)   # at dock, aligned
    at_opposed = _phi_env((0, 0, 14), direction=3, close_dir=1)._potential(geo)   # at dock, opposed
    far_aligned = _phi_env((10, 0, 14), direction=1, close_dir=1)._potential(geo) # 10 tiles out
    far_opposed = _phi_env((10, 0, 14), direction=3, close_dir=1)._potential(geo)
    assert at_aligned == pytest.approx(6.0)     # near_xy=1, aligned -> full w_dir
    assert at_opposed == pytest.approx(0.0)     # near_xy=1, opposed -> 0
    assert at_aligned > at_opposed              # heading matters AT the dock
    assert far_aligned == pytest.approx(0.0) and far_opposed == pytest.approx(0.0)
    assert far_aligned == far_opposed           # heading is free while routing (gated off)


def test_phi_maximal_at_anchor():
    geo = RewardParams(w_e=0.0, w_h=0.0)   # isolate geometry (no energy/discovery contribution)
    at_anchor = _phi_env((0, 0, 14), direction=1)._potential(geo)
    assert at_anchor == pytest.approx(geo.w_xy + geo.w_z + geo.w_dir)
    for neighbor in [(1, 0, 14), (0, 0, 15), (0, 0, 14)]:
        env = _phi_env(neighbor, direction=(1 if neighbor != (0, 0, 14) else 2))
        assert env._potential(geo) <= at_anchor + 1e-9


def test_phi_increases_with_energy_margin():
    energy_only = RewardParams(w_xy=0.0, w_z=0.0, w_dir=0.0, w_e=2.0)
    low = _bare_env(current_position=(0, 0, 14), history=[])
    low.close_pos = [0, 0, 14]; low.close_dir = 1
    high = _bare_env(current_position=(0, 0, 14), history=[dict(h) for h in _DROP_HIST])
    high.close_pos = [0, 0, 14]; high.close_dir = 1
    assert high._potential(energy_only) > low._potential(energy_only)


def test_phi_drops_heading_term_before_calibration():
    """With close_dir None, Phi excludes the heading term (no wrong-heading wall)."""
    geo = RewardParams(w_e=0.0)
    env = _bare_env(current_position=(0, 0, 14), current_direction=3, goal_position=(0, 0, 14))
    # no close_pos/close_dir -> provisional, heading off
    assert env._potential(geo) == pytest.approx(geo.w_xy + geo.w_z)


# --------------------------------------------- dense constructive-move gate (test 14)

def test_constructive_xy_move_yields_net_positive_shaping():
    """A one-tile XY move toward the (aligned) anchor ALONG the entry corridor must give
    F = gamma*Phi' - Phi > 0 at the default D_xy, even at high Phi where discount leakage
    (1-gamma)*Phi bites. (close_dir=1 -> entry +Y, so the approach axis is -Y.)"""
    p = RewardParams()
    far = _phi_env((0, -11, 14), direction=1)._potential(p)
    near = _phi_env((0, -10, 14), direction=1)._potential(p)
    f = p.gamma * near - far
    assert f > 0


# ------------------------------------------------------------ quality bonus (test 7)

def test_quality_bonus_disabled_when_R_quality_max_zero():
    env = _bare_env()
    assert env._quality_bonus(8.0, 5.5, 1.0, RewardParams(R_quality_max=0.0)) == 0.0


def test_quality_bonus_gated_to_zero_for_untested_ride():
    env = _bare_env()
    assert env._quality_bonus(0.0, 0.0, 0.0, RewardParams(R_quality_max=500.0)) == 0.0


def test_quality_bonus_peaks_at_target_band_and_is_bounded():
    env = _bare_env()
    p = RewardParams(R_quality_max=500.0)
    peak = env._quality_bonus(8.0, 5.5, 1.0, p)
    assert 0.0 <= peak <= 500.0
    assert peak > 0.95 * 500.0
    # off-target is strictly worse
    assert env._quality_bonus(12.0, 9.0, 8.0, p) < peak


def test_quality_bonus_monotone_ramp_below_target():
    """Replaces the symmetric-falloff pin: quality is now ramp+band, so the below-target
    side is strictly MONOTONE (every excitement increment pays -- the Phase-5 plateau fix)
    and the above-target side keeps the ramp half (see the overshoot test)."""
    env = _bare_env()
    p = RewardParams(R_quality_max=500.0)
    vals = [env._quality_bonus(e, 5.5, 2.0, p) for e in (1.0, 3.0, 5.0, 7.0, 8.0)]
    assert all(b > a for a, b in zip(vals, vals[1:]))


def test_quality_bonus_bounded_and_finite_on_extremes():
    env = _bare_env()
    p = RewardParams(R_quality_max=500.0)
    for E in (-5.0, 0.5, 15.0, 50.0):
        for I in (-5.0, 0.5, 15.0, 50.0):
            for N in (-5.0, 0.5, 15.0, 50.0):
                q = env._quality_bonus(E, I, N, p)
                assert np.isfinite(q)
                assert 0.0 <= q <= 500.0


# ----------------------------------------------------------- auto-calibration (test 11)

def _completing_history():
    return [
        {"action": 0, "position": [61, 67, 14], "direction": 2,
         "next_position": [61, 68, 14], "next_direction": 2, "track_type": 0, "is_complete": False},
        # closing piece: pre-close head is [61, 68, 14] facing South (2)
        {"action": 13, "position": [61, 68, 14], "direction": 2,
         "next_position": [61, 66, 14], "next_direction": 2, "track_type": 9, "is_complete": True},
    ]


def test_closing_record_captures_preclose_head_not_endpoint():
    rec = OpenRCT2Env._closing_record_from_history(_completing_history())
    assert rec["pos"] == [61, 68, 14]      # pre-close head, NOT next_position [61,66,14]
    assert rec["dir"] == 2
    assert rec["action"] == 13             # closing piece geometry (sufficiency gate)
    assert rec["track_type"] == 9


def _rec(pos, d, action=13, track_type=9):
    return {"pos": list(pos), "dir": d, "action": action, "track_type": track_type}


def test_robust_close_anchor_requires_consistency():
    """A trustworthy anchor needs >= _CLOSE_MIN_CONSISTENT completions sharing a direction --
    so a single fluky closure (or a few that disagree) never locks Phi's target."""
    assert OpenRCT2Env._robust_close_anchor([]) is None
    assert OpenRCT2Env._robust_close_anchor([_rec([61, 68, 14], 2), _rec([61, 68, 14], 2)]) is None
    # 3 records but no 3 agree on direction
    assert OpenRCT2Env._robust_close_anchor(
        [_rec([61, 68, 14], 0), _rec([61, 68, 14], 1), _rec([61, 68, 14], 2)]) is None


def test_robust_close_anchor_median_pos_modal_dir_ignores_fluke():
    """With >=3 completions agreeing on direction, the anchor is that direction + the group's
    median position; an offset/other-direction outlier among them is ignored."""
    records = [_rec([61, 68, 14], 2), _rec([61, 68, 14], 2), _rec([62, 68, 14], 2),  # consistent
               _rec([99, 99, 14], 1)]                                                # fluke
    anchor = OpenRCT2Env._robust_close_anchor(records)
    assert anchor["dir"] == 2                     # modal direction (fluke outvoted)
    assert anchor["pos"] == [61, 68, 14]          # median of the dir-2 group, not the [99,99] fluke


def test_maybe_capture_locks_only_after_consistent_completions():
    """The fix: ONE completion no longer locks the anchor (locking a fluky first closure poisoned
    whole runs). It takes >= _CLOSE_MIN_CONSISTENT agreeing completions; then the robust anchor
    persists to cache + file."""
    import json, os
    env = _bare_env()
    env.loop_completed = True
    env.track_builder = SimpleNamespace(history=_completing_history())  # pre-close head [61,68,14] dir 2
    env._maybe_capture_closing_geometry()
    assert OpenRCT2Env._close_cache is None                 # one completion is not enough
    assert not os.path.exists(OpenRCT2Env._CLOSE_CACHE_PATH)
    for _ in range(OpenRCT2Env._CLOSE_MIN_CONSISTENT - 1):  # reach the consistency threshold
        env.track_builder = SimpleNamespace(history=_completing_history())
        env._maybe_capture_closing_geometry()
    assert OpenRCT2Env._close_cache["pos"] == [61, 68, 14]
    assert OpenRCT2Env._close_cache["dir"] == 2
    assert os.path.exists(OpenRCT2Env._CLOSE_CACHE_PATH)
    with open(OpenRCT2Env._CLOSE_CACHE_PATH) as f:
        assert json.load(f)["dir"] == 2


def test_maybe_capture_single_fluke_does_not_poison_then_consistent_wins():
    """A lone fluky first closure does not lock; subsequent consistent closures set the real
    anchor and outvote the fluke -- the exact failure that stuck 4 runs is prevented."""
    fluke = _completing_history()
    fluke[-1]["position"] = [63, 65, 14]
    fluke[-1]["direction"] = 1
    env = _bare_env()
    env.loop_completed = True
    env.track_builder = SimpleNamespace(history=fluke)
    env._maybe_capture_closing_geometry()                   # 1 fluke -> no lock
    assert OpenRCT2Env._close_cache is None
    for _ in range(OpenRCT2Env._CLOSE_MIN_CONSISTENT):      # consistent good closures
        env.track_builder = SimpleNamespace(history=_completing_history())
        env._maybe_capture_closing_geometry()
    assert OpenRCT2Env._close_cache["dir"] == 2             # consistent closure won
    assert OpenRCT2Env._close_cache["pos"] == [61, 68, 14]  # not the [63,65,14] fluke


def test_maybe_capture_locked_anchor_not_overwritten():
    """Once a robust anchor locks, later completions don't overwrite it (Phi stays stable)."""
    env = _bare_env()
    env.loop_completed = True
    for _ in range(OpenRCT2Env._CLOSE_MIN_CONSISTENT):
        env.track_builder = SimpleNamespace(history=_completing_history())
        env._maybe_capture_closing_geometry()
    assert OpenRCT2Env._close_cache["pos"] == [61, 68, 14]
    other = _completing_history()
    other[-1]["position"] = [99, 99, 14]
    env.track_builder = SimpleNamespace(history=other)
    env._maybe_capture_closing_geometry()
    assert OpenRCT2Env._close_cache["pos"] == [61, 68, 14]  # unchanged once locked


def test_probe_log_closing_coerces_numpy_and_never_raises():
    """The closing-geometry probe must coerce numpy ints (json can't serialize int64) and must
    never raise -- an unguarded TypeError here crashed a real training run. Writes a valid line."""
    import json, os
    env = _bare_env()
    entry = {"position": [np.int64(62), np.int64(66), np.int64(14)], "direction": np.int64(0),
             "action": np.int64(13), "track_type": np.int64(9),
             "next_position": [np.int64(61), np.int64(66), np.int64(14)], "next_direction": np.int64(0)}
    env._probe_log_closing(entry)                       # must not raise on numpy types
    path = os.path.join(os.path.dirname(OpenRCT2Env._CLOSE_CACHE_PATH), "closing_probe.jsonl")
    rec = json.loads(open(path).read().strip())
    assert rec["pos"] == [62, 66, 14] and rec["next_direction"] == 0
    env._probe_log_closing({"position": None, "direction": None})   # malformed -> still no raise


def test_maybe_capture_noop_without_completion():
    env = _bare_env()
    env.track_builder = SimpleNamespace(history=_completing_history())
    env.loop_completed = False
    env._maybe_capture_closing_geometry()
    assert OpenRCT2Env._close_cache is None


def test_load_close_cache_reads_file_when_class_cache_empty():
    import json
    rec = {"pos": [1, 2, 14], "dir": 3, "action": 13, "track_type": 9}
    with open(OpenRCT2Env._CLOSE_CACHE_PATH, "w") as f:
        json.dump(rec, f)
    OpenRCT2Env._close_cache = None
    env = _bare_env()
    assert env._load_close_cache() == rec
    assert OpenRCT2Env._close_cache == rec      # populates the in-memory cache


def test_init_closing_target_applies_calibration():
    OpenRCT2Env._close_cache = {"pos": [1, 2, 14], "dir": 3, "action": 13, "track_type": 9}
    env = _bare_env(goal_position=(62, 66, 14))
    env._init_closing_target()
    assert env.close_pos == [1, 2, 14]
    assert env.close_dir == 3
    assert list(env._reward_target_position()) == [1, 2, 14]
    assert env._reward_target_direction() == 3


def test_init_closing_target_provisional_uses_station_entry_axis():
    """No calibration yet: close_pos still falls back to the guide tile, but close_dir is now the
    DETERMINISTIC station-entry axis (North=0, confirmed by the closing-geometry probe: every
    completion enters BeginStation [61,66,14] heading North). So the heading reward term is ON from
    step 1 instead of None-until-calibration -- breaking the chicken-and-egg that stalled bootstrap."""
    OpenRCT2Env._close_cache = None
    env = _bare_env(goal_position=(62, 66, 14))
    env._init_closing_target()
    assert env.close_pos is None
    assert env.close_dir == 0                                    # North = station-entry axis
    assert list(env._reward_target_position()) == [62, 66, 14]   # provisional guide tile
    assert env._reward_target_direction() == 0


# ------------------------------------------------------- PBRS reward (tests 2,3,5,10)

def test_pbrs_constructive_move_reward_is_positive():
    p = RewardParams()
    env = _phi_env((0, -11, 14), direction=1)   # on the -Y entry corridor (close_dir=1)
    env._phi_prev = env._potential(p)
    env.current_position = [0, -10, 14]         # one tile closer to the anchor along the corridor
    env.loop_completed = False
    assert env._calculate_reward(True, 0) > 0


def test_pbrs_completion_reward_is_strongly_positive_and_zeros_phi():
    p = RewardParams()
    env = _phi_env((1, 0, 14), direction=1)
    phi_prev = env._potential(p)
    env._phi_prev = phi_prev
    env.loop_completed = True
    r = env._calculate_reward(True, 13)
    assert r == pytest.approx(p.R_complete - phi_prev)   # net = -Phi(s_prev) + R_complete
    assert r > 900
    assert env._phi_prev == 0.0                          # terminal Phi forced to 0


def test_pbrs_shaping_telescopes_to_negative_phi0():
    p = RewardParams()
    env = _phi_env((30, 0, 14), direction=1)
    phi0 = env._potential(p)
    env._phi_prev = phi0
    path = [((25, 0, 14), False), ((18, 0, 14), False),
            ((9, 0, 14), False), ((2, 0, 14), False), ((0, 0, 14), True)]
    discounted = 0.0
    for i, (pos, complete) in enumerate(path):
        env.current_position = list(pos)
        env.loop_completed = complete
        r = env._calculate_reward(True, 0)
        shaping = r - (p.R_complete if complete else 0.0)   # step_cost = 0 default
        discounted += (p.gamma ** i) * shaping
    assert discounted == pytest.approx(-phi0, abs=1e-6)


def test_pbrs_failure_returns_flat_penalty_without_phi_drift():
    p = RewardParams()
    env = _phi_env((10, 0, 14))
    env._phi_prev = 5.0
    r = env._calculate_reward(False, 0)
    assert r == pytest.approx(p.fail_penalty)
    assert env._phi_prev == 5.0                          # head did not move -> Phi unchanged


def test_place_then_remove_nets_strictly_negative(monkeypatch):
    """PBRS telescoping makes place+remove = (gamma-1)*(Phi+Phi') < 0 (old hack netted 0)."""
    monkeypatch.setattr(oe_mod, "APIController", FakeAPI)
    env = OpenRCT2Env(verbose=0)
    env.skip_ride_testing = True
    env.reset()
    _, r_place, *_ = env.step(0)
    _, r_remove, *_ = env.step(31)
    assert r_place + r_remove < 0


def test_oscillating_place_remove_stays_bounded_and_non_positive(monkeypatch):
    monkeypatch.setattr(oe_mod, "APIController", FakeAPI)
    env = OpenRCT2Env(verbose=0)
    env.skip_ride_testing = True
    env.reset()
    total = 0.0
    for _ in range(10):
        _, r1, *_ = env.step(0)
        _, r2, *_ = env.step(31)
        total += r1 + r2
    assert total < 0
    assert np.isfinite(total)


def test_truncation_adds_no_partial_credit_bonus(monkeypatch):
    monkeypatch.setattr(oe_mod, "APIController", FakeAPI)
    env = OpenRCT2Env(verbose=0)
    env.skip_ride_testing = True
    env.reset()
    p = env.reward_params
    env.max_track_length = 4
    for _ in range(3):
        env.step(0)
    # Anchor the target at the head so the OLD code would add a large proximity bonus.
    env.close_pos = list(env.current_position)
    env.close_dir = env.current_direction
    phi_prev_before = env._phi_prev
    _, reward, terminated, truncated, _ = env.step(0)     # 4th piece -> truncates
    assert truncated and not terminated
    expected = p.gamma * env._potential(p) - phi_prev_before + p.step_cost
    assert reward == pytest.approx(expected)


# ------------------------------------------------- terminal quality bonus (tests 5,7)

class CompletingAPI(FakeAPI):
    """FakeAPI that completes after a couple of agent pieces and serves immediate ride
    stats, so the env's terminal ride-test path runs server-free (poll returns at once)."""
    excitement = 8.0
    intensity = 5.5
    nausea = 1.0
    complete_after = 2  # agent (non-station) pieces

    def __init__(self, host=None, port=None, verbose=0):
        super().__init__(host, port, verbose)
        self._agent_pieces = 0

    def create_ride(self):
        self._agent_pieces = 0
        return super().create_ride()

    def place_track_piece(self, x, y, z, direction, track_type, has_chain=False):
        resp = super().place_track_piece(x, y, z, direction, track_type, has_chain)
        if track_type not in (1, 2, 3):   # agent piece, not station
            self._agent_pieces += 1
            if self._agent_pieces >= self.complete_after:
                resp["payload"]["isCircuitComplete"] = True
        return resp

    def place_entrance_exit(self):
        return {"success": True}

    def start_ride_test(self):
        return {"success": True, "payload": {}}

    def get_ride_stats(self):
        return {"success": True, "payload": {
            "excitement": self.excitement, "intensity": self.intensity, "nausea": self.nausea}}

    def get_ride_measurements(self):
        # models an OLD plugin (endpoint not deployed); MeasuredAPI overrides with data
        return {"success": False, "error": "Unknown endpoint: getRideMeasurements"}


def _drive_to_terminal(env, max_steps=12):
    phi_prev_before, reward, info = None, None, {}
    for _ in range(max_steps):
        phi_prev_before = env._phi_prev
        _, reward, terminated, truncated, info = env.step(0)
        if terminated or truncated:
            return phi_prev_before, reward, terminated, truncated, info
    raise AssertionError("episode did not terminate")


def test_completion_terminal_adds_quality_bonus(monkeypatch):
    monkeypatch.setattr(oe_mod, "APIController", CompletingAPI)
    env = OpenRCT2Env(verbose=0)
    env.skip_ride_testing = False
    env.reward_params = RewardParams(R_quality_max=500.0)
    env.reset()
    p = env.reward_params
    phi_prev_before, reward, terminated, _, info = _drive_to_terminal(env)
    assert terminated
    rr = info['ride_rating']
    quality = env._quality_bonus(rr['excitement'], rr['intensity'], rr['nausea'], p)
    assert quality > 0
    assert reward == pytest.approx(p.R_complete - phi_prev_before + quality)


def test_completion_no_quality_when_disabled(monkeypatch):
    monkeypatch.setattr(oe_mod, "APIController", CompletingAPI)
    env = OpenRCT2Env(verbose=0)
    env.skip_ride_testing = False
    env.reward_params = RewardParams(R_quality_max=0.0)   # phases 1-4
    env.reset()
    p = env.reward_params
    phi_prev_before, reward, terminated, _, _ = _drive_to_terminal(env)
    assert terminated
    assert reward == pytest.approx(p.R_complete - phi_prev_before)   # no quality term


def test_goal_position_is_the_dock_staging_tile(monkeypatch):
    """Goal = the STAGING tile one step on the approach side of the dock (dock - entry-direction
    vector), NOT the dock tile itself. probe_corridor.py: the head cannot sit on the occupied
    station tile; it docks FROM (62,66,14) heading the entry direction. (Reverts the goal-=-dock
    change that collapsed completion to 0%.)"""
    monkeypatch.setattr(oe_mod, "APIController", CompletingAPI)
    env = OpenRCT2Env(verbose=0)
    env.reset()
    ex, ey = env.direction_vectors[env._STATION_ENTRY_DIR]
    expected = [env.station_start_position[0] - ex,
                env.station_start_position[1] - ey,
                env.station_start_position[2]]
    assert list(env.goal_position) == expected == [62, 66, 14]


# ----------------------------------------------- curriculum unification (tests 12,13,15)

from openrct2_gym.envs.improved_phased_curriculum_wrapper import ImprovedPhasedCurriculumWrapper


# ============================ lift-hill incentive (structural bonus + discovery) ============

def test_phase_reward_params_structural_per_phase():
    W = ImprovedPhasedCurriculumWrapper
    p1 = W._phase_reward_params(1)
    assert p1.R_struct_max == 0.0                       # struct off in P1
    p2a = W._phase_reward_params(2, phase2_stage=1)
    p2b = W._phase_reward_params(2, phase2_stage=2)
    p2c = W._phase_reward_params(2, phase2_stage=3)
    assert (p2a.R_struct_max, p2a.struct_w_chain, p2a.struct_w_drop, p2a.struct_chain_target) \
        == (250.0, 1.0, 0.0, 1)                         # stage 2.1: one-chain bridge
    assert (p2b.R_struct_max, p2b.struct_w_chain, p2b.struct_w_drop, p2b.struct_chain_target) \
        == (250.0, 1.0, 0.0, 1)                         # stage 2.2: one-chain completion
    assert (p2c.R_struct_max, p2c.struct_w_chain, p2c.struct_w_drop, p2c.struct_chain_target) \
        == (250.0, 1.0, 0.0, 3)                         # stage 2.3: tighten to >=3 chains
    p3 = W._phase_reward_params(3)
    # P3/P4 redesign: struct credit is graded height/drop/length toward per-phase targets
    # (piece counting alone was already satisfied by the 2.3 mini-loop and taught nothing).
    assert (p3.R_struct_max, p3.struct_w_chain, p3.struct_w_height, p3.struct_w_drop,
            p3.struct_w_length) == (250.0, 0.0, 0.4, 0.4, 0.2)
    p4 = W._phase_reward_params(4)
    # Jul-7: height/drop reweighted 0.4->0.3 to grade the steep leg in (see
    # test_p4_params_grade_steepness); the four components still sum to 1.0.
    assert (p4.R_struct_max, p4.struct_w_chain, p4.struct_w_height, p4.struct_w_drop,
            p4.struct_w_length) == (250.0, 0.0, 0.3, 0.3, 0.2)
    p5 = W._phase_reward_params(5)
    # Jul-9: P5 struct credit is back ON, re-aimed at the wooden-RC rating caps (see
    # test_p5_params_pay_the_quality_gate for the full P5 economics spec).
    assert p5.R_struct_max == 250.0 and p5.R_quality_max == 500.0
    # discovery potential: ON only in the hill-building phases 2-4; OFF in the
    # pure-completion phase 1 and the quality phase 5 (an always-on climb pull derails
    # Phase-1 completion learning). w_h=3 (not 6): a 1M-step run showed the deeper
    # attractor let a wrecked policy settle into climbing instead of completing.
    assert p1.w_h == 0.0 and p5.w_h == 0.0
    # strong discovery pull to FIND the chain climb in 2.1, relaxing once learned so it cannot
    # recreate the climb-away attractor in later stages
    assert p2a.w_h == 6.0 and p2b.w_h == 4.0 and p2c.w_h == 3.0
    assert p3.w_h == 3.0 and p4.w_h == 3.0
    # completion gating (closure-first): phase 2 keeps a real flat-completion floor so closing a
    # loop always out-pays an unclosed climb (the phase-1 skill survives while hills are added);
    # phase 3/4 remove that floor once the bridge is done so structure is required.
    assert p1.completion_hill_floor == 1.0 and p5.completion_hill_floor == 1.0
    assert p2a.completion_hill_floor == pytest.approx(0.2)   # 2.1 restores the closure floor
    assert p2b.completion_hill_floor == 0.15                 # 2.2 lowered to widen the chain-vs-flat gap
    assert p2c.completion_hill_floor == 0.10
    assert p3.completion_hill_floor == 0.0 and p4.completion_hill_floor == 0.0
    # descent/return shaping (w_return): ON in the hill phases 2-4 to make the RETURN learnable,
    # OFF in phase 1 (pure completion) and phase 5 (quality), mirroring the discovery term w_h.
    assert p2a.w_return == 6.0 and p2b.w_return == 4.0 and p2c.w_return == 3.0
    assert p3.w_return == 3.0 and p4.w_return == 3.0
    assert p1.w_return == 0.0 and p5.w_return == 0.0
    # d_z=20 keeps the near-station m_z slope at 0.3/z -- steep enough that the energy
    # term's chain-lift bump (~+0.47) cannot make climbing profitable in phase 1
    # (d_z=60 weakened the slope to 0.1/z and the energy term became an accidental
    # discovery term: a 1M-step run climbed to +75z in phase 1 and never completed).
    # High-altitude reach comes from m_z being UNCLIPPED instead (see gradient test).
    for p in (p1, p2a, p2b, p2c, p3, p4, p5):
        assert p.d_z == 20.0


def test_discovery_potential_off_in_phase1_and_5_on_in_phase2():
    W = ImprovedPhasedCurriculumWrapper
    # a track that climbed (banked elevation) so the discovery term would fire if active
    env = _bare_env(current_position=(0, 0, 20),
                    history=[{"action": 9, "next_position": [0, 0, 20]}])
    env.close_pos = [0, 0, 14]
    env.close_dir = 1
    phi_p1 = env._potential(W._phase_reward_params(1))   # discovery OFF -> no elevation term
    phi_p2 = env._potential(W._phase_reward_params(2))   # discovery ON
    phi_p5 = env._potential(W._phase_reward_params(5))   # discovery OFF
    assert phi_p2 > phi_p1 + 1.0                         # P2 gains the banked-elevation term
    # P5 has no discovery pull either; it differs from P1 by the route term (on in the
    # completion phases 1-4, off in phase 5) and -- since the Jul-9 quality redesign --
    # by the excitement-feature term (P5's own dense gradient).
    p5 = W._phase_reward_params(5)
    route = W._phase_reward_params(1).w_route * env._route_progress()
    exc_feat = p5.w_exc_feat * env._exc_feature_quality(p5)
    assert phi_p5 == pytest.approx(phi_p1 - route + exc_feat)


# ---- structural bonus

def _struct_env(chains=0, drops=0):
    """A bare env whose history has `chains` chain-lift pieces (action 9) climbing +1 z each,
    then `drops` drop pieces (action 6) descending -1 z each. Entries carry full positions
    (the drop-z accounting reads entry AND exit z, like real take_action histories)."""
    hist, z, x = [], 14, 0
    for _ in range(chains):
        hist.append({"action": 9, "position": [x, 0, z], "next_position": [x + 1, 0, z + 1]})
        z += 1
        x += 1
    for _ in range(drops):
        hist.append({"action": 6, "position": [x, 0, z], "next_position": [x + 1, 0, z - 1]})
        z -= 1
        x += 1
    return _bare_env(history=hist)


def test_structural_bonus_disabled_returns_zero():
    env = _struct_env(chains=3, drops=1)
    assert env._structural_bonus(RewardParams(R_struct_max=0.0)) == 0.0      # P1/P5


def test_structural_bonus_p2_scales_with_chain_count():
    """Chain COUNT times chain ELEVATION (vs the roundtrip_gain bar): pieces alone are
    farmable -- three scattered 1-z stubs must not equal a lift hill. _struct_env climbs
    +1 z per chain, so count == gain here."""
    p = RewardParams(R_struct_max=250.0, struct_chain_target=3, struct_w_chain=1.0, struct_w_drop=0.0)
    assert _struct_env(chains=1)._structural_bonus(p) == pytest.approx(250.0 / 9)        # 1/3 x 1/3
    assert _struct_env(chains=2)._structural_bonus(p) == pytest.approx(250.0 * 4 / 9)    # 2/3 x 2/3
    assert _struct_env(chains=3)._structural_bonus(p) == pytest.approx(250.0)
    assert _struct_env(chains=4)._structural_bonus(p) == pytest.approx(250.0)   # clipped at target
    assert _struct_env(chains=0)._structural_bonus(p) == 0.0


def test_structural_bonus_p3_requires_chains_and_drop():
    p = RewardParams(R_struct_max=250.0, struct_chain_target=2, struct_w_chain=0.5, struct_w_drop=0.5)
    # chains=2 climbs +2 z against the default gain bar (3): chain credit scales by 2/3
    assert _struct_env(chains=2, drops=1)._structural_bonus(p) == pytest.approx(125.0 * 2 / 3 + 125.0)
    assert _struct_env(chains=2, drops=0)._structural_bonus(p) == pytest.approx(125.0 * 2 / 3)
    assert _struct_env(chains=0, drops=1)._structural_bonus(p) == pytest.approx(125.0)   # drop only
    assert _struct_env(chains=0, drops=0)._structural_bonus(p) == 0.0
    # a full-height hill restores the full chain credit
    assert _struct_env(chains=3, drops=1)._structural_bonus(p) == pytest.approx(250.0)


def test_structural_bonus_p4_integration():
    p = RewardParams(R_struct_max=250.0, struct_chain_target=3, struct_w_chain=0.5, struct_w_drop=0.5)
    assert _struct_env(chains=3, drops=1)._structural_bonus(p) == pytest.approx(250.0)   # hill + drop
    assert _struct_env(chains=3, drops=0)._structural_bonus(p) == pytest.approx(125.0)
    assert _struct_env(chains=0, drops=1)._structural_bonus(p) == pytest.approx(125.0)


def test_structural_bonus_uses_history_not_live_counter():
    p = RewardParams(R_struct_max=250.0, struct_chain_target=3, struct_w_chain=1.0, struct_w_drop=0.0)
    env = _struct_env(chains=3)
    env.chain_lift_count = 0                 # deliberately desynced live counter
    assert env._structural_bonus(p) == pytest.approx(250.0)   # follows history, not the counter


def test_structural_bonus_added_to_reward_only_on_completion():
    p = RewardParams(R_struct_max=250.0, struct_chain_target=3, struct_w_chain=1.0, struct_w_drop=0.0)
    env = _struct_env(chains=3)
    env.reward_params = p
    env._phi_prev = phi_prev = env._potential(p)
    # not completed: no struct term, ordinary PBRS
    env.loop_completed = False
    assert env._calculate_reward(True, 0) == pytest.approx(p.gamma * env._potential(p) - phi_prev)
    # completed: R_complete + full structural bonus (3 chains)
    env._phi_prev = phi_prev = env._potential(p)
    env.loop_completed = True
    r = env._calculate_reward(True, 0)
    assert r == pytest.approx(p.R_complete - phi_prev + 250.0)
    assert env._last_struct_bonus == pytest.approx(250.0)


def test_structural_bonus_not_added_on_truncation(monkeypatch):
    monkeypatch.setattr(oe_mod, "APIController", FakeAPI)
    env = OpenRCT2Env(verbose=0)
    env.skip_ride_testing = True
    env.reward_params = RewardParams(R_struct_max=250.0, struct_chain_target=1,
                                     struct_w_chain=1.0, struct_w_drop=0.0)
    env.reset()
    env.max_track_length = 4
    for _ in range(3):
        env.step(9)                          # place chain lifts (would qualify if completed)
    phi_prev_before = env._phi_prev
    _, reward, terminated, truncated, _ = env.step(9)   # 4th piece -> truncates (FakeAPI never completes)
    assert truncated and not terminated
    p = env.reward_params
    assert reward == pytest.approx(p.gamma * env._potential(p) - phi_prev_before)   # no struct term
    assert env._last_struct_bonus == 0.0


def test_structural_bonus_not_farmable_without_completion(monkeypatch):
    monkeypatch.setattr(oe_mod, "APIController", FakeAPI)
    env = OpenRCT2Env(verbose=0)
    env.skip_ride_testing = True
    env.reward_params = RewardParams(R_struct_max=250.0, struct_chain_target=1,
                                     struct_w_chain=1.0, struct_w_drop=0.0)
    env.reset()
    for _ in range(8):                       # FakeAPI never completes
        _, r, *_ = env.step(9)
        assert r < 50.0                      # no completion -> no +250 struct ever leaks in


def test_episode_metrics_expose_struct_and_height_diagnostics(monkeypatch):
    monkeypatch.setattr(oe_mod, "APIController", CompletingAPI)
    env = OpenRCT2Env(verbose=0)
    env.skip_ride_testing = True
    # roundtrip_gain=0 opts out of the elevation scaling: the FakeAPI geometry never gains
    # z, and this test is about the metrics contract, not the hill economy.
    env.reward_params = RewardParams(R_struct_max=250.0, struct_chain_target=1,
                                     struct_w_chain=1.0, struct_w_drop=0.0,
                                     roundtrip_gain=0.0)
    env.reset()
    info = {}
    for _ in range(6):
        _, _, terminated, truncated, info = env.step(9)   # chain lifts -> completes
        if terminated or truncated:
            break
    m = info['episode_metrics']
    assert {'chain_count', 'struct_bonus', 'max_gain'}.issubset(m)   # callback's contract
    assert m['chain_count'] >= 1 and m['struct_bonus'] > 0 and m['max_gain'] >= 0


# ---- completion gating (force hills: a flat loop is worth little in phases 2-4)

def _complete_payoff(params, chains=0, drops=0, isolate=False):
    """Total reward for a completing step under `params`, with _phi_prev=0 so the PBRS term is 0
    and only the completion payoff (gated R_complete + struct bonus) remains. With isolate=True
    the once-per-episode summit/roundtrip latches are pre-burned so the milestones don't add on
    top (a fresh chain completion at a low roundtrip_gain banks them too) -- use it to assert the
    isolated gated-completion magnitude."""
    env = _struct_env(chains=chains, drops=drops)
    env.reward_params = params
    env._phi_prev = 0.0
    env.loop_completed = True
    if isolate:
        env._summit_awarded = True
        env._roundtrip_awarded = True
    return env._calculate_reward(True, 0)


def test_completion_gate_lowers_flat_floor_in_phase2_stage2():
    # Stage 2.2 LOWERS the flat-completion floor to 0.15 (was 0.25) to widen the chain-vs-flat gap
    # after the agent collapsed onto flat completion here. Completion is isolated from the
    # summit/roundtrip milestones (which a fresh 1-chain completion also banks now that gain=1) to
    # check the gated completion magnitude itself.
    P = ImprovedPhasedCurriculumWrapper._phase_reward_params(2, phase2_stage=2)
    flat = _complete_payoff(P, chains=0, isolate=True)
    full = _complete_payoff(P, chains=1, isolate=True)
    assert flat == pytest.approx(150.0)        # floor=.15: a hill-less close pays less now
    assert full == pytest.approx(1250.0)       # R_complete * 1.0 + struct 250
    assert flat < full


def test_phase2_final_stage_tightens_to_three_chains():
    # isolate=True: a full 3-chain hill at gain bar 3 now ALSO banks the roundtrip milestone
    # on the completing step; burn the latches to assert the gated-completion magnitude itself.
    P = ImprovedPhasedCurriculumWrapper._phase_reward_params(2, phase2_stage=3)
    flat = _complete_payoff(P, chains=0, isolate=True)
    one_chain = _complete_payoff(P, chains=1, isolate=True)
    full = _complete_payoff(P, chains=3, isolate=True)
    assert flat == pytest.approx(100.0)        # mostly devalued, but not zeroed out
    assert flat < one_chain < full
    assert full == pytest.approx(1250.0)


def test_phase2_w_h_relaxes_across_stages():
    """Change D: a strong chain-discovery pull to FIND the climb in stage 2.1, relaxing once
    learned so it cannot recreate the climb-away attractor in later stages."""
    W = ImprovedPhasedCurriculumWrapper
    assert W._phase_reward_params(2, phase2_stage=1).w_h == 6.0
    assert W._phase_reward_params(2, phase2_stage=2).w_h == 4.0
    assert W._phase_reward_params(2, phase2_stage=3).w_h == 3.0
    assert W._phase_reward_params(1).w_h == 0.0 and W._phase_reward_params(5).w_h == 0.0


def test_phase2_stage1_restores_closure_floor():
    """Closure-first repair: stage 2.1's flat-completion floor is RESTORED (0.2) so a closed
    loop always out-pays an unclosed climb -- the agent keeps the loop-closing skill while it
    learns to add a hill. (Reverses the 0.05 de-valuation that drove the climb-only collapse:
    flat-close paid 50 < the ~100 a climb-and-stop banked, so the agent abandoned closure.)"""
    W = ImprovedPhasedCurriculumWrapper
    assert W._phase_reward_params(2, phase2_stage=1).completion_hill_floor == pytest.approx(0.2)
    assert W._phase_reward_params(2, phase2_stage=2).completion_hill_floor == 0.15  # lowered: widen chain gap
    assert W._phase_reward_params(2, phase2_stage=3).completion_hill_floor == 0.10  # unchanged
    flat = _complete_payoff(W._phase_reward_params(2, phase2_stage=1), chains=0)
    assert flat == pytest.approx(200.0)       # closing a flat loop is worth something again


def test_phase2_summit_breadcrumb_schedule():
    """Discoverability bootstrap: R_summit pays the chain CLIMB itself as a small breadcrumb,
    tapering 120 -> 60 -> 0 across the bridge stages (the climb is learned by 2.3). It stays
    strictly below R_roundtrip so the RETURN is still worth more than stopping at the summit
    (and below the flat-completion floor so 'climb and stop' never out-pays closing the loop)."""
    W = ImprovedPhasedCurriculumWrapper
    summit = [W._phase_reward_params(2, phase2_stage=s).R_summit for s in (1, 2, 3)]
    assert summit == [40.0, 30.0, 0.0]
    assert summit == sorted(summit, reverse=True)              # tapering
    for s in (1, 2, 3):
        P = W._phase_reward_params(2, phase2_stage=s)
        assert P.R_summit < P.R_roundtrip                      # return stays worth learning
        # the SUM must stay below the flat-completion floor, else 'climb and stop' out-pays closing
        assert P.R_summit + P.R_roundtrip < P.completion_hill_floor * P.R_complete
    assert W._phase_reward_params(1).R_summit == 0.0
    assert W._phase_reward_params(5).R_summit == 0.0


def test_phase2_roundtrip_gain_anneals_monotonically():
    """Make the existing climb-and-return milestone DISCOVERABLE: the required chain-climb stays
    a single piece's worth (1 z) through stages 2.1 AND 2.2 -- so the climb habit and its
    breadcrumbs survive the integration step -- and only 2.3 demands the full hill. The bar is
    chain gain 3, not 4: the canonical hill [10,9,13] banks 3 via CHAIN pieces (the crest piece
    isn't chained), so a 4.0 bar made the milestone + w_return silently inert for that hill."""
    W = ImprovedPhasedCurriculumWrapper
    gains = [W._phase_reward_params(2, phase2_stage=s).roundtrip_gain for s in (1, 2, 3)]
    assert gains == [1.0, 1.0, 3.0]
    assert gains == sorted(gains)                              # monotone non-decreasing
    assert gains[-1] == RewardParams().roundtrip_gain          # stage 2.3 == default full hill


def test_completion_not_gated_in_phase1():
    P = ImprovedPhasedCurriculumWrapper._phase_reward_params(1)
    assert _complete_payoff(P, chains=0) == pytest.approx(1000.0)   # flat fully paid in P1


def test_hill_completion_beats_incomplete_flat_does_not_in_phase2():
    # Completion-first is now hill-conditioned, but Phase 2 keeps a small flat-completion floor
    # to avoid erasing the Phase-1 skill while it introduces chain lifts. A real hill completion
    # still dominates the best bounded incomplete return (Phi_max via PBRS telescoping).
    P = ImprovedPhasedCurriculumWrapper._phase_reward_params(2, phase2_stage=3)
    phi_max = P.w_xy + P.w_z + P.w_dir + P.w_e + P.w_h
    assert _complete_payoff(P, chains=3) > phi_max                 # a hill completion dominates
    assert _complete_payoff(P, chains=0) > phi_max                 # completion remains alive
    assert _complete_payoff(P, chains=0) < _complete_payoff(P, chains=1)


# ---- round-trip elevation milestone (decomposition: teach climb-and-return)

def _roundtrip_env(peak_z, head_z, p):
    env = _bare_env(current_position=(0, 0, head_z),
                    history=[{"action": 9, "next_position": [0, 0, peak_z]}])
    env.reward_params = p
    env._phi_prev = 0.0          # isolate the milestone delta from PBRS shaping
    env._roundtrip_awarded = False
    env.loop_completed = False
    return env


def test_roundtrip_milestone_awarded_on_climb_and_return():
    p = RewardParams(R_roundtrip=100.0, roundtrip_gain=4.0)
    env = _roundtrip_env(peak_z=20, head_z=14, p=p)        # climbed +6, back at station height
    r = env._calculate_reward(True, 0)
    assert r == pytest.approx(p.gamma * env._potential(p) + 100.0)
    assert env._roundtrip_awarded is True


def test_roundtrip_not_awarded_while_still_elevated():
    p = RewardParams(R_roundtrip=100.0, roundtrip_gain=4.0)
    env = _roundtrip_env(peak_z=20, head_z=20, p=p)        # climbed but hasn't returned
    r = env._calculate_reward(True, 0)
    assert r == pytest.approx(p.gamma * env._potential(p))  # no milestone
    assert env._roundtrip_awarded is False


def test_roundtrip_not_awarded_without_enough_climb():
    p = RewardParams(R_roundtrip=100.0, roundtrip_gain=4.0)
    env = _roundtrip_env(peak_z=16, head_z=14, p=p)        # only +2 < gain 4
    r = env._calculate_reward(True, 0)
    assert r == pytest.approx(p.gamma * env._potential(p))
    assert env._roundtrip_awarded is False


def test_roundtrip_awarded_once_per_episode():
    p = RewardParams(R_roundtrip=100.0, roundtrip_gain=4.0)
    env = _roundtrip_env(peak_z=20, head_z=14, p=p)
    env._calculate_reward(True, 0)                          # first -> awarded
    env._phi_prev = 0.0
    r2 = env._calculate_reward(True, 0)                     # second -> no re-award
    assert r2 == pytest.approx(p.gamma * env._potential(p))


def test_roundtrip_requires_chain_lift_not_plain_climb():
    """Change B: a plain (non-chain) climb-and-return earns NO round-trip milestone and does
    NOT burn the once-per-episode flag, so a later chain climb in the same episode can still
    qualify. Aligns the env award with the wrapper's chain_count>=1 gate."""
    p = RewardParams(R_roundtrip=100.0, roundtrip_gain=4.0)
    env = _bare_env(current_position=(0, 0, 14),
                    history=[{"action": 0, "next_position": [0, 0, 20]}])  # plain climb +6, returned
    env.reward_params = p
    env._phi_prev = 0.0
    env._roundtrip_awarded = False
    env.loop_completed = False
    r = env._calculate_reward(True, 0)
    assert r == pytest.approx(p.gamma * env._potential(p))   # no milestone
    assert env._roundtrip_awarded is False                   # flag not burned


def test_phase2_roundtrip_fires_at_each_stages_annealed_gain():
    """The annealed gain is what makes the round-trip reachable: a 1-z chain climb-and-return
    qualifies the milestone in stages 2.1 AND 2.2 (gain 1), but not 2.3 (gain 4); a full 4-z
    hill qualifies in every stage."""
    W = ImprovedPhasedCurriculumWrapper
    for stage, should_fire in [(1, True), (2, True), (3, False)]:
        P = W._phase_reward_params(2, phase2_stage=stage)
        env = _roundtrip_env(peak_z=15, head_z=14, p=P)       # one chain piece (+1 z), returned
        env._summit_awarded = False
        env._calculate_reward(True, 0)
        assert env._roundtrip_awarded is should_fire
    for stage in (1, 2, 3):                                    # a 4-z hill qualifies everywhere
        P = W._phase_reward_params(2, phase2_stage=stage)
        env = _roundtrip_env(peak_z=18, head_z=14, p=P)
        env._summit_awarded = False
        env._calculate_reward(True, 0)
        assert env._roundtrip_awarded is True


def test_phase2_1_summit_breadcrumb_fires_once():
    """Stage 2.1 pays R_summit the first time the chain climb reaches the (annealed) gain,
    exactly once per episode -- the breadcrumb that makes the climb worth starting before the
    return is learned. Isolated here with the head still elevated, so only summit (not the
    round-trip) fires."""
    P = ImprovedPhasedCurriculumWrapper._phase_reward_params(2, phase2_stage=1)  # R_summit 40
    env = _roundtrip_env(peak_z=20, head_z=20, p=P)           # climbed, NOT returned -> summit only
    env._summit_awarded = False
    r1 = env._calculate_reward(True, 0)
    assert env._summit_awarded is True
    assert env._roundtrip_awarded is False                    # head still elevated
    assert r1 == pytest.approx(P.gamma * env._potential(P) + 40.0)
    env._phi_prev = 0.0
    r2 = env._calculate_reward(True, 0)                        # once-per-episode: no re-award
    assert r2 == pytest.approx(P.gamma * env._potential(P))


def test_flat_completion_below_hill_completion_all_stages():
    """Closure stays dominant: in every bridge stage a hill completion out-pays a flat one, so
    the agent keeps closing the loop -- only now the biggest reward requires the hill too.
    (Magnitudes include the freshly-latched summit/roundtrip milestones, hence an inequality.)"""
    W = ImprovedPhasedCurriculumWrapper
    for stage in (1, 2, 3):
        P = W._phase_reward_params(2, phase2_stage=stage)
        assert _complete_payoff(P, chains=0) < _complete_payoff(P, chains=3)


def test_phase2_info_exposes_schedule_diagnostics(monkeypatch):
    """Diagnostic-per-term: the live annealed schedule (roundtrip_gain, R_summit) is surfaced on
    the Phase-2 terminal info so the bootstrap is visible in TensorBoard next to the summit/
    roundtrip rates."""
    monkeypatch.setattr(oe_mod, "APIController", CompletingAPI)
    base = OpenRCT2Env(verbose=0)
    wrapper = ImprovedPhasedCurriculumWrapper(base, verbose=0)
    wrapper.current_phase = 2
    wrapper.phase2_stage = 1
    wrapper._update_phase_settings()
    wrapper.reset()
    info = {}
    for _ in range(12):
        _, _, terminated, truncated, info = wrapper.step(9)   # chain lifts -> completes
        if terminated or truncated:
            break
    assert info.get('phase2_roundtrip_gain') == 1.0
    assert info.get('phase2_summit_reward') == 40.0


def test_roundtrip_disabled_and_below_completion_per_phase():
    assert RewardParams().R_roundtrip == 0.0               # off by default
    W = ImprovedPhasedCurriculumWrapper
    assert W._phase_reward_params(1).R_roundtrip == 0.0     # off in phase 1
    assert W._phase_reward_params(5).R_roundtrip == 0.0     # off in phase 5
    assert W._phase_reward_params(2, phase2_stage=1).R_roundtrip == 80.0
    assert W._phase_reward_params(2, phase2_stage=2).R_roundtrip == 60.0
    assert W._phase_reward_params(2, phase2_stage=3).R_roundtrip == 60.0
    for P in (
        W._phase_reward_params(2, phase2_stage=1),
        W._phase_reward_params(2, phase2_stage=2),
        W._phase_reward_params(2, phase2_stage=3),
        W._phase_reward_params(3),
        W._phase_reward_params(4),
    ):
        # must stay below a real hill completion (R_complete) so climb-and-return is a stepping
        # stone, never a substitute for closing the hill loop.
        assert P.R_roundtrip < P.R_complete


def test_completion_first_invariant_holds_for_every_phase():
    """Regression guard for the Phase-2.3 milestone-farming collapse: the once-per-episode climb
    milestones (R_roundtrip + R_summit, earnable WITHOUT closing) must never out-pay completion.
    _phase_reward_params is now validated, so a violating config would raise on construction."""
    W = ImprovedPhasedCurriculumWrapper
    for phase, stage in [(1, 1), (2, 1), (2, 2), (2, 3), (3, 1), (4, 1), (5, 1)]:
        P = W._phase_reward_params(phase, phase2_stage=stage)   # raises if the invariant is violated
        milestones = P.R_roundtrip + P.R_summit
        assert milestones < P.R_complete                        # a perfect completion always wins
        if P.completion_hill_floor > 0.0:                       # ...and so does a flat completion
            assert milestones < P.completion_hill_floor * P.R_complete


def test_validate_completion_first_rejects_milestone_farming():
    """The guard fails fast on the exact pre-fix Phase-2.3 config (R_roundtrip 200 > 0.10*1000)."""
    W = ImprovedPhasedCurriculumWrapper
    W._validate_completion_first(RewardParams(completion_hill_floor=0.10, R_roundtrip=60.0), "ok")
    bad = RewardParams(completion_hill_floor=0.10, R_roundtrip=200.0)   # the bug that collapsed P2.3
    with pytest.raises(AssertionError):
        W._validate_completion_first(bad, "bad")


# ---- summit milestone (reachable first half of the round-trip bridge)

def test_summit_milestone_awarded_on_chain_climb_without_return():
    """Change C: a chain climb to >= roundtrip_gain earns R_summit ONCE, independent of
    returning or completing -- the reachable stepping stone before the full round-trip."""
    p = RewardParams(R_summit=80.0, R_roundtrip=300.0, roundtrip_gain=4.0)
    assert p.R_summit < p.R_roundtrip
    env = _bare_env(current_position=(0, 0, 20),        # still elevated: no return
                    history=[{"action": 9, "next_position": [0, 0, 20]}])
    env.reward_params = p
    env._phi_prev = 0.0
    env._summit_awarded = False
    env._roundtrip_awarded = False
    env.loop_completed = False
    r = env._calculate_reward(True, 9)
    assert r == pytest.approx(p.gamma * env._potential(p) + 80.0)
    assert env._summit_awarded is True
    assert env._roundtrip_awarded is False             # not returned -> no round-trip


def test_summit_not_awarded_for_plain_or_small_climb():
    p = RewardParams(R_summit=80.0, roundtrip_gain=4.0)
    plain = _bare_env(current_position=(0, 0, 20),
                      history=[{"action": 0, "next_position": [0, 0, 20]}])   # plain, no chain
    plain.reward_params = p
    plain._phi_prev = 0.0
    plain._summit_awarded = False
    plain.loop_completed = False
    assert plain._calculate_reward(True, 0) == pytest.approx(p.gamma * plain._potential(p))
    assert plain._summit_awarded is False
    small = _bare_env(current_position=(0, 0, 16),
                      history=[{"action": 9, "next_position": [0, 0, 16]}])   # chain but only +2
    small.reward_params = p
    small._phi_prev = 0.0
    small._summit_awarded = False
    small.loop_completed = False
    assert small._calculate_reward(True, 9) == pytest.approx(p.gamma * small._potential(p))


def test_summit_awarded_once_per_episode():
    p = RewardParams(R_summit=80.0, roundtrip_gain=4.0)
    env = _bare_env(current_position=(0, 0, 20),
                    history=[{"action": 9, "next_position": [0, 0, 20]}])
    env.reward_params = p
    env._phi_prev = 0.0
    env._summit_awarded = False
    env.loop_completed = False
    env._calculate_reward(True, 9)                       # first -> awarded
    env._phi_prev = 0.0
    r2 = env._calculate_reward(True, 9)                  # second -> no re-award
    assert r2 == pytest.approx(p.gamma * env._potential(p))


# ---- discovery potential (elevation term in Phi)

_DISC = RewardParams(w_xy=0.0, w_z=0.0, w_dir=0.0, w_e=0.0, w_h=6.0, h_scale=6.0)  # isolate discovery


def _peak_env(peak_z, head_z=None):
    """Bare env whose history reaches `peak_z` via a CHAIN-LIFT piece (discovery is
    chain-specific); head height defaults to peak."""
    head_z = peak_z if head_z is None else head_z
    return _bare_env(current_position=(0, 0, head_z),
                     history=[{"action": 9, "next_position": [0, 0, peak_z]}])


def test_discovery_potential_increases_and_saturates():
    phis = [_peak_env(z)._potential(_DISC) for z in (14, 16, 20, 30)]
    assert phis[0] == pytest.approx(0.0)        # no gain
    assert phis[1] == pytest.approx(2.0)        # gain 2 -> 6*2/6
    assert phis[2] == pytest.approx(6.0)        # gain 6 -> saturates
    assert phis[3] == pytest.approx(6.0)        # gain 16 -> clipped at h_scale
    assert phis == sorted(phis)


def test_discovery_potential_banks_peak_after_descent():
    env = _bare_env(current_position=(0, 0, 14),
                    history=[{"action": 9, "next_position": [0, 0, 30]},   # chain-climbed
                             {"action": 6, "next_position": [0, 0, 14]}])  # back to station height
    assert env._potential(_DISC) == pytest.approx(6.0)   # banked chain peak, NOT 0 despite head at z=14


def test_discovery_potential_empty_history_no_raise():
    env = _bare_env(history=[])              # max() of empty would raise without the guard
    assert env._potential(_DISC) == pytest.approx(0.0)


def test_discovery_is_chain_specific():
    """Change A: only chain-lift pieces (actions 9/10) earn the discovery term. An
    identical-geometry plain climb (action 5/0, same track_type) earns ZERO discovery, so
    the agent has a real gradient toward the chain flag the Phase-2 gate counts."""
    plain = _bare_env(current_position=(0, 0, 20),
                      history=[{"action": 0, "next_position": [0, 0, 20]}])
    chain = _bare_env(current_position=(0, 0, 20),
                      history=[{"action": 9, "next_position": [0, 0, 20]}])
    assert plain._potential(_DISC) == pytest.approx(0.0)   # plain climb: no discovery
    assert chain._potential(_DISC) == pytest.approx(6.0)   # chain climb: full discovery (gain 6)


def test_chain_max_gain_helper_filters_non_chain_pieces():
    """_chain_max_gain banks the highest elevation reached via chain pieces only."""
    env = _bare_env(history=[{"action": 0, "next_position": [0, 0, 30]},   # plain climb to +16
                             {"action": 9, "next_position": [0, 0, 18]}])  # chain climb to +4
    assert env._chain_max_gain() == pytest.approx(4.0)     # only the chain piece counts
    assert _bare_env(history=[])._chain_max_gain() == pytest.approx(0.0)


class ClimbingAPI(FakeAPI):
    """FakeAPI whose agent pieces climb one z per placement (station pieces stay flat)."""
    def place_track_piece(self, x, y, z, direction, track_type, has_chain=False):
        dx, dy = self._dv[direction]
        nz = z + (1 if track_type not in (1, 2, 3) else 0)
        ep = {"x": x + dx, "y": y + dy, "z": nz, "direction": direction}
        self._stack.append(ep)
        return {"success": True, "payload": {
            "nextEndpoint": ep, "isCircuitComplete": False,
            "validNextPieces": {"validPieces": list(range(46))}}}

    def delete_last_track_piece(self):
        if self._stack:
            self._stack.pop()
        prev = self._stack[-1] if self._stack else {"x": 61, "y": 66, "z": 14, "direction": 0}
        return {"success": True, "payload": {"nextEndpoint": prev, "piecesRemaining": len(self._stack)}}


def test_discovery_term_telescopes_on_place_then_remove(monkeypatch):
    monkeypatch.setattr(oe_mod, "APIController", ClimbingAPI)
    env = OpenRCT2Env(verbose=0)
    env.skip_ride_testing = True
    env.reset()
    _, r_place, *_ = env.step(9)     # climb a chain lift (raises max_gain)
    _, r_remove, *_ = env.step(31)   # remove it (max_gain recomputes lower)
    assert r_place + r_remove < 0    # discovery term doesn't break place/remove<0


def _climb_vs_flat_gap(w_h):
    """reward(climb step) - reward(flat step) from the same prior state + _phi_prev,
    at the given discovery weight. Both next-states advance xy equally; climb also goes up."""
    p = RewardParams(w_h=w_h)
    prior_hist = [{"action": 0, "next_position": [12, 0, 14]}]

    def make(history, pos):
        env = _bare_env(current_position=pos, current_direction=1, history=history)
        env.close_pos = [0, 0, 14]
        env.close_dir = 1
        env.reward_params = p
        return env

    phi_prior = make(prior_hist, [12, 0, 14])._potential(p)
    flat = make(prior_hist + [{"action": 0, "next_position": [11, 0, 14]}], [11, 0, 14])
    flat._phi_prev = phi_prior; flat.loop_completed = False
    r_flat = flat._calculate_reward(True, 0)
    climb = make(prior_hist + [{"action": 9, "next_position": [11, 0, 15]}], [11, 0, 15])
    climb._phi_prev = phi_prior; climb.loop_completed = False
    r_climb = climb._calculate_reward(True, 9)
    return r_climb - r_flat


def test_climb_step_beats_flat_step_and_discovery_does_the_work():
    # Load-bearing (feedback #6): compare actual transition REWARDS at the DEFAULT w_h,
    # and pin that the discovery term materially widens the climb>flat margin (it must,
    # or w_h is dead).
    w_h_default = RewardParams().w_h
    assert _climb_vs_flat_gap(w_h_default) > 0                    # climb strictly preferred
    assert _climb_vs_flat_gap(w_h_default) > _climb_vs_flat_gap(0.0) + 0.4  # discovery does real work


def test_phase1_chain_climb_does_not_beat_flat_progress():
    """Regression for the d_z=60 failure: in phase 1 (w_h=0) a chain-lift climb step must
    LOSE to a flat step toward the goal, or the energy term turns into an accidental
    discovery term and the agent climbs instead of completing. Uses the real game
    geometry (~2z gained per chain piece) and phase-1 params from the curriculum."""
    p = ImprovedPhasedCurriculumWrapper._phase_reward_params(1)
    assert p.w_h == 0.0
    prior = [{"action": 0, "position": [70, 66, 14], "next_position": [69 - i, 66, 14]}
             for i in range(3)]
    head0 = [67, 66, 14]

    def make(history, pos):
        env = _bare_env(current_position=pos, current_direction=3, history=history)
        env.close_pos = None
        env.close_dir = None
        env.reward_params = p
        return env

    phi_prior = make(prior, head0)._potential(p)
    flat = make(prior + [{"action": 0, "position": head0, "next_position": [66, 66, 14]}],
                [66, 66, 14])
    flat._phi_prev = phi_prior; flat.loop_completed = False
    r_flat = flat._calculate_reward(True, 0)
    climb = make(prior + [{"action": 9, "position": head0, "next_position": [66, 66, 16]}],
                 [66, 66, 16])
    climb._phi_prev = phi_prior; climb.loop_completed = False
    r_climb = climb._calculate_reward(True, 9)
    assert r_flat > r_climb            # flat looping must stay optimal in phase 1


def test_height_gradient_reaches_high_altitude():
    """The m_z pull home must NOT clip flat at moderate altitude: a lost climber at
    z=+20..+60 above station needs a strictly decreasing Phi as it climbs further
    (at the old d_z=20 everything above +20z was a flat plateau with no gradient home)."""
    geo = RewardParams(w_e=0.0, w_h=0.0)   # isolate the height-alignment term
    phi_34 = _phi_env((0, 0, 34))._potential(geo)   # +20 above station
    phi_54 = _phi_env((0, 0, 54))._potential(geo)   # +40
    phi_74 = _phi_env((0, 0, 74))._potential(geo)   # +60
    assert phi_34 > phi_54 > phi_74                 # gradient still pulls home up high


def test_build_tall_and_stall_is_dominated_by_completion():
    p = RewardParams()                       # defaults, w_h=6
    env = _bare_env(current_position=(5, 0, 14), current_direction=1,
                    history=[{"action": 9, "next_position": [5, 0, 14]}])
    env.close_pos = [0, 0, 14]; env.close_dir = 1
    env.reward_params = p
    phi0 = env._potential(p)
    env._phi_prev = phi0
    discounted = 0.0
    for i, z in enumerate(range(15, 23)):    # climb 8 tiles, never complete
        env.current_position = [5, 0, z]
        env.track_builder.history.append({"action": 9, "next_position": [5, 0, z]})
        env.loop_completed = False
        discounted += (p.gamma ** i) * env._calculate_reward(True, 9)
    assert discounted < p.R_complete - phi0  # far below a flat completion
    assert discounted < 30.0                 # bounded by ~Phi_max, nowhere near +1000


# ============================================================================
# Closure-first redesign: descent shaping (w_return) + the reachability ladder.
# The descent term is PBRS-clean and gated to be 0 at/above the summit threshold
# height (STATION_HEIGHT + roundtrip_gain), rising to w_return only on the return.
# ============================================================================

def _return_only(w_return=5.0, roundtrip_gain=4.0):
    """Params isolating the descent-shaping term: all other Phi weights zeroed."""
    return RewardParams(w_xy=0.0, w_z=0.0, w_dir=0.0, w_e=0.0, w_h=0.0,
                        w_return=w_return, roundtrip_gain=roundtrip_gain)


def _climbed_env(head_z, peak_z=20):
    """Bare env that chain-climbed to peak_z (gain = peak_z - STATION_HEIGHT), head at head_z."""
    return _bare_env(current_position=(0, 0, head_z),
                     history=[{"action": 9, "next_position": [0, 0, peak_z]}])


def test_return_potential_zero_above_threshold_rises_on_descent():
    """The descent-shaping term is 0 at/above the summit threshold height (14 + 4 = 18) and
    rises monotonically to w_return as the head returns to station height (14) -- the
    continuous downhill gradient that was missing (descent had no per-step shaping)."""
    p = _return_only(w_return=5.0)
    above = _climbed_env(head_z=22)._potential(p)
    at_thresh = _climbed_env(head_z=18)._potential(p)
    mid = _climbed_env(head_z=16)._potential(p)
    home = _climbed_env(head_z=14)._potential(p)
    assert above == pytest.approx(0.0)
    assert at_thresh == pytest.approx(0.0)
    assert 0.0 < mid < home
    assert home == pytest.approx(5.0)


def test_return_potential_gated_on_chain_climb():
    """The term stays 0 until a CHAIN climb reaches roundtrip_gain: a head at station with no
    prior chain hill (or only a plain climb) earns no return shaping -- so it never rewards
    digging below the station without having built a hill first."""
    p = _return_only(w_return=5.0)
    no_climb = _bare_env(current_position=(0, 0, 14), history=[])._potential(p)
    plain = _bare_env(current_position=(0, 0, 14),
                      history=[{"action": 0, "next_position": [0, 0, 20]}])._potential(p)
    chained = _climbed_env(head_z=14)._potential(p)
    assert no_climb == pytest.approx(0.0)
    assert plain == pytest.approx(0.0)
    assert chained == pytest.approx(5.0)


def test_crossing_roundtrip_threshold_creates_no_return_reward():
    """High-priority review point: the term is 0 both just-below the gate (chain gain < 4) and
    exactly AT the summit threshold height, so the gate turning on injects NO positive Phi jump
    (F = gamma*Phi' - Phi) -- it cannot re-pay the summit. Only the descent below the threshold
    earns shaping."""
    p = _return_only(w_return=5.0)
    below_gate = _bare_env(current_position=(0, 0, 16),
                           history=[{"action": 9, "next_position": [0, 0, 16]}])._potential(p)
    at_gate = _climbed_env(head_z=18, peak_z=18)._potential(p)   # gate flips on here
    above = _climbed_env(head_z=22, peak_z=22)._potential(p)
    descending = _climbed_env(head_z=17, peak_z=20)._potential(p)
    assert below_gate == pytest.approx(0.0)   # gain 2 < 4 -> gate off
    assert at_gate == pytest.approx(0.0)      # gate on but term 0 -> no jump, no summit re-pay
    assert above == pytest.approx(0.0)
    assert descending > 0.0                   # only the descent earns the shaping


def test_return_shaping_telescopes_and_is_not_farmable():
    """The descent term is part of Phi, so a descend-then-ascend round trip telescopes to
    (gamma-1)*Phi < 0 -- the agent cannot farm reward by bobbing up and down. Descending pays
    as-you-go (that gradient is the point); only the closed cycle must be non-positive."""
    p = _return_only(w_return=5.0)
    env = _climbed_env(head_z=18)             # chain peak 20 (gate on); head at summit (term 0)
    env.reward_params = p
    env.loop_completed = False
    env._phi_prev = env._potential(p)         # Phi(z=18) == 0
    env.current_position = [0, 0, 14]         # descend to station: term rises
    r_down = env._calculate_reward(True, 0)
    env.current_position = [0, 0, 18]         # ascend back to summit: term falls
    r_up = env._calculate_reward(True, 0)
    assert r_down > 0.0 and r_up < 0.0
    assert r_down + r_up == pytest.approx((p.gamma - 1.0) * 5.0)   # telescopes, net < 0


def test_return_shaping_weight_enabled_in_hill_phases_off_elsewhere():
    """w_return gates the descent shaping: >0 in the hill phases 2-4, 0 in phase 1 (pure
    completion) and phase 5 (quality), mirroring the discovery term w_h."""
    W = ImprovedPhasedCurriculumWrapper
    assert RewardParams().w_return == 0.0                        # off by default
    for stage in (1, 2, 3):
        assert W._phase_reward_params(2, phase2_stage=stage).w_return > 0.0
    assert W._phase_reward_params(3).w_return > 0.0
    assert W._phase_reward_params(4).w_return > 0.0
    assert W._phase_reward_params(1).w_return == 0.0
    assert W._phase_reward_params(5).w_return == 0.0


def test_return_shaping_cannot_affect_phase1_or_phase5():
    """Regression guard / evidence: the descent term is inert outside the hill phases. Even for a
    chain-climbed env sitting at station height -- where the term is MAXIMAL when enabled -- it
    contributes exactly 0 under phase-1 and phase-5 params, while it IS positive under phase-2.1.
    So a Phase-1 training collapse can never be attributable to this change (the Phase-1 reward is
    byte-for-byte unchanged)."""
    W = ImprovedPhasedCurriculumWrapper
    env = _climbed_env(head_z=14)             # climbed (gain 6) then returned: max shaping if on
    assert env._return_potential(W._phase_reward_params(1)) == 0.0
    assert env._return_potential(W._phase_reward_params(5)) == 0.0
    assert env._return_potential(W._phase_reward_params(2, phase2_stage=1)) > 0.0


def _no_geo(P):
    """Stage params with the dense Phi geometry weights zeroed, so _calculate_reward returns
    essentially only the sparse ladder rewards (completion floor, struct, roundtrip, summit)."""
    return replace(P, w_xy=0.0, w_z=0.0, w_dir=0.0, w_e=0.0, w_h=0.0)


def _ladder_rung(P, *, chains, head_z, completed):
    hist = [{"action": 9, "position": [i, 0, 14], "next_position": [i + 1, 0, 20]}
            for i in range(chains)]                                          # gain 6 >= 4
    env = _bare_env(current_position=(0, 0, head_z), history=hist)
    env.reward_params = _no_geo(P)
    env._phi_prev = 0.0
    env._summit_awarded = False
    env._roundtrip_awarded = False
    env.loop_completed = completed
    return env._calculate_reward(True, 9 if chains else 0)


def test_phase2_stage1_reward_ladder_is_monotone():
    """Closure-first ladder: stage-2.1 sparse rewards form a monotone, reachable ladder
    climb-only < climb-and-descend < flat-close < hill-close. CLOSING THE LOOP ALWAYS OUT-PAYS
    NOT CLOSING IT (flat_close > climb_descend) -- the milestones are a stepping stone, never a
    substitute -- and the big hill bonus (hill_close >> flat_close), not an inflated round-trip,
    is what pulls past flat looping toward the hill. (The pre-fix params made climb_descend
    out-pay flat_close, which let the agent farm the round-trip and abandon closure: at 1.1M
    steps Phase-2.3 completion collapsed 0.44 -> 0.08 with zero hill completions.)"""
    P = ImprovedPhasedCurriculumWrapper._phase_reward_params(2, phase2_stage=1)
    climb_only = _ladder_rung(P, chains=1, head_z=20, completed=False)     # elevated, no close
    climb_descend = _ladder_rung(P, chains=1, head_z=14, completed=False)  # returned, not closed
    flat_close = _ladder_rung(P, chains=0, head_z=14, completed=True)      # flat loop closed
    hill_close = _ladder_rung(P, chains=1, head_z=14, completed=True)      # hill loop closed
    assert climb_only < climb_descend < flat_close < hill_close


def test_episode_metrics_expose_return_potential(monkeypatch):
    """Diagnostic-per-term: episode_metrics carries return_potential so training can watch the
    return gradient fire (gate flag w_return + a logged diagnostic, per the reward-design prefs)."""
    monkeypatch.setattr(oe_mod, "APIController", CompletingAPI)
    env = OpenRCT2Env(verbose=0)
    env.skip_ride_testing = True
    env.reward_params = RewardParams(w_return=5.0, roundtrip_gain=4.0)
    env.reset()
    info = {}
    for _ in range(6):
        _, _, terminated, truncated, info = env.step(9)   # chain lifts -> completes
        if terminated or truncated:
            break
    assert 'return_potential' in info['episode_metrics']


# ============================================================================
# Near-closure densification: a steep local Phi bonus driving last-piece closure
# in the cold-start (the gentle w_xy approach term is too flat to bootstrap it).
# ============================================================================

def _close_only(w_close=8.0, close_range=3.0, close_z_range=2.0):
    """Params isolating the near-closure bonus: all other Phi weights zeroed."""
    return RewardParams(w_xy=0.0, w_z=0.0, w_dir=0.0, w_e=0.0, w_h=0.0, w_return=0.0,
                        w_close=w_close, close_range=close_range, close_z_range=close_z_range)


def test_close_bonus_off_by_default():
    """Off by default so the frozen-Phi tests are unaffected; only the curriculum turns it on."""
    assert RewardParams().w_close == 0.0
    assert _phi_env((0, 0, 14))._potential(_close_only(w_close=0.0)) == pytest.approx(0.0)


def test_close_bonus_zero_beyond_range():
    """Strictly local: zero beyond close_range tiles (XY) or close_z_range (height)."""
    p = _close_only(w_close=8.0, close_range=3.0, close_z_range=2.0)   # target = close_pos (0,0,14)
    assert _phi_env((10, 0, 14))._potential(p) == pytest.approx(0.0)   # 10 tiles away (>3)
    assert _phi_env((0, 0, 20))._potential(p) == pytest.approx(0.0)    # +6 above station (>2)


def test_close_bonus_ramps_steeply_to_target():
    """Along the entry-axis approach corridor it ramps monotonically (and steeply) to w_close at the
    staging tile. (_phi_env uses close_dir=1, so the entry axis is +Y and the approach comes from
    the south; moving along it -- not perpendicular -- is what the funnel rewards.)"""
    p = _close_only(w_close=8.0, close_range=3.0)
    phis = [_phi_env((0, -d, 14))._potential(p) for d in (3, 2, 1, 0)]   # approach along the corridor
    assert phis[0] == pytest.approx(0.0)             # at the range edge
    assert phis[0] < phis[1] < phis[2] < phis[3]     # steep monotonic climb in the final tiles
    assert phis[3] == pytest.approx(8.0)             # full bonus at the staging tile
    assert (phis[3] - phis[2]) > 1.0                 # steeper than the 0.25/tile w_xy approach


def test_close_bonus_is_a_directional_corridor():
    """The near-closure funnel is DIRECTIONAL: full only when the head approaches the staging tile
    along the entry axis from the correct (-entry) side. Off-axis or past the tile gives less/nothing,
    so the agent can't farm it by parking beside the station from the wrong direction. (_phi_env:
    close_dir=1 -> entry dir +Y; approach side is -Y (south); off-axis is the X direction.)"""
    p = _close_only(w_close=8.0, close_range=3.0)
    on_axis    = _phi_env((0, -1, 14))._potential(p)   # 1 tile south, on the corridor
    off_axis   = _phi_env((1,  0, 14))._potential(p)   # 1 tile to the side (perpendicular)
    wrong_side = _phi_env((0,  1, 14))._potential(p)   # 1 tile PAST the staging tile (along < 0)
    assert on_axis > 0.0
    assert off_axis < on_axis                          # off-axis funnels down
    assert wrong_side == pytest.approx(0.0)            # past the staging tile -> no funnel


def test_close_funnel_pinches_to_centerline_at_throat():
    """The funnel pinches to a POINT at the throat: the off-axis tiles immediately beside the
    staging tile (along=0) are excluded, but off-axis tiles one tile back (along>=1) are retained.
    (_phi_env: close_dir=1 -> entry +Y, throat staging (0,0); off-axis is the X direction.)"""
    p = _close_only(w_close=8.0, close_range=3.0)
    throat_off = _phi_env((1,  0, 14))._potential(p)   # off-axis AT the throat -> excluded
    throat_off2 = _phi_env((-1, 0, 14))._potential(p)  # other side, AT the throat -> excluded
    back_off   = _phi_env((1, -1, 14))._potential(p)   # off-axis ONE tile back -> retained
    on_throat  = _phi_env((0,  0, 14))._potential(p)   # centerline throat -> full bonus
    assert throat_off == pytest.approx(0.0)
    assert throat_off2 == pytest.approx(0.0)
    assert back_off > 0.0
    assert on_throat == pytest.approx(8.0)


def test_close_funnel_pinch_disabled_restores_rectangular_corridor():
    """close_throat_pinch=0 falls back to the legacy un-pinched corridor (throat off-axis tiles
    still earn the perpendicular falloff)."""
    p = replace(_close_only(w_close=8.0, close_range=3.0), close_throat_pinch=0.0)
    assert _phi_env((1, 0, 14))._potential(p) > 0.0     # throat off-axis no longer excluded


def test_close_bonus_enabled_in_completion_phases_off_in_phase5():
    """The curriculum turns the densified closure signal ON in the completion phases 1-4 (to drive
    the cold-start bootstrap) and OFF in phase 5 (completion already mastered there)."""
    W = ImprovedPhasedCurriculumWrapper
    assert W._phase_reward_params(1).w_close > 0.0
    assert W._phase_reward_params(2, phase2_stage=1).w_close > 0.0
    assert W._phase_reward_params(2, phase2_stage=2).w_close > 0.0
    assert W._phase_reward_params(2, phase2_stage=3).w_close > 0.0
    assert W._phase_reward_params(3).w_close > 0.0
    assert W._phase_reward_params(4).w_close > 0.0
    assert W._phase_reward_params(5).w_close == 0.0


def test_phase_switch_keeps_single_reward_method_and_only_changes_params(monkeypatch):
    monkeypatch.setattr(oe_mod, "APIController", FakeAPI)
    base = OpenRCT2Env(verbose=0)
    wrapper = ImprovedPhasedCurriculumWrapper(base, verbose=0)
    base_env = wrapper._get_base_env()
    reward_fn = base_env._calculate_reward            # the env's own method, never swapped
    assert base_env.reward_params.R_quality_max == 0.0          # phase 1
    assert base_env.skip_ride_testing is True

    wrapper.current_phase = 5
    wrapper._update_phase_settings()
    assert base_env._calculate_reward == reward_fn             # SAME method (no per-phase swap)
    assert base_env.reward_params.R_quality_max == 500.0       # only params changed
    assert base_env.reward_params.step_cost == 0.0             # P5 no longer punishes length
    assert base_env.skip_ride_testing is False


def test_p2_qualified_stage_predicates():
    """P2 stage predicates (P3/P4 now have their own scale gates, covered by the
    test_p3/p4_qualified_* tests)."""
    W = ImprovedPhasedCurriculumWrapper
    w = W.__new__(W)                          # no __init__/env needed for the predicate

    def base(actions, roundtrip=False, current_z=14):
        return SimpleNamespace(track_builder=SimpleNamespace(
            history=[{"action": a, "next_position": [0, 0, 20 if a in (9, 10) else current_z]}
                     for a in actions]),
            current_position=[0, 0, current_z],
            _roundtrip_awarded=roundtrip)

    w.current_phase = 2
    w.phase2_stage = 1
    assert w._is_qualified(base([9], roundtrip=True), False) is True    # P2.1: no completion needed
    assert w._is_qualified(base([0], roundtrip=True), False) is False   # must include a chain
    w.phase2_stage = 2
    assert w._is_qualified(base([9]), True) is True            # P2.2: >=1 chain completion
    assert w._is_qualified(base([9]), False) is False
    w.phase2_stage = 3
    assert w._is_qualified(base([9, 9, 9]), True) is True      # P2.3: >=3 chains
    assert w._is_qualified(base([9, 9]), True) is False
    w.current_phase = 5
    # P5 (Jul-9): now a quality diagnostic -- this fake base env is untested, so False
    # (the full P5 truth table lives in test_p5_qualified_is_tested_excitement_diagnostic)
    assert w._is_qualified(base([9, 9, 6]), True) is False


def test_phase2_summit_signal_tracks_chain_climb():
    W = ImprovedPhasedCurriculumWrapper
    w = W.__new__(W)
    w.current_phase = 2
    w.phase2_stage = 1
    chain = SimpleNamespace(
        track_builder=SimpleNamespace(history=[{"action": 9, "next_position": [0, 0, 20]}]),
        current_position=[0, 0, 20],            # chain-climbed +6, still elevated (no return)
        _summit_awarded=False, _roundtrip_awarded=False, STATION_HEIGHT=14)
    sig = w._phase2_signals(chain, success=False)
    assert sig['phase2_summit'] is True         # summit = chain climb past threshold, no return needed
    assert sig['phase2_roundtrip'] is False      # did not return -> no round-trip
    plain = SimpleNamespace(
        track_builder=SimpleNamespace(history=[{"action": 0, "next_position": [0, 0, 20]}]),
        current_position=[0, 0, 20],
        _summit_awarded=False, _roundtrip_awarded=False, STATION_HEIGHT=14)
    assert w._phase2_signals(plain, success=False)['phase2_summit'] is False   # plain climb earns no summit


def test_phase2_substage_advancement_sequence():
    W = ImprovedPhasedCurriculumWrapper
    w = W.__new__(W)
    w.current_phase = 2
    w.phase2_stage = 1
    w.phase2_roundtrip_threshold = 0.30
    w.phase2_chain1_success_threshold = 0.30
    w.phase2_success_threshold = 0.40
    w._track_stats = True
    w.verbose = 0
    w.phases_completed = []
    w.phase_episode_count = 50
    w.total_loops_completed = 0
    w.phase2_summit_results = deque(maxlen=50)
    w.phase2_roundtrip_results = deque(maxlen=50)
    w.phase2_chain1_completion_results = deque(maxlen=50)
    w.phase2_chain2_completion_results = deque(maxlen=50)
    w.phase2_chain3_completion_results = deque(maxlen=50)
    # warm-start state touched by _clear_phase_windows / _advance_* hooks
    w.scaffold_results = deque(maxlen=50)
    w._cold_flags = deque(maxlen=50)
    from openrct2_gym.envs.warm_start import WarmStartAnnealer
    w._annealer = WarmStartAnnealer()
    updates = []
    w._update_phase_settings = lambda: updates.append(w.phase2_stage)

    def fill_window(qualified_count):
        w.episode_results = deque([False] * 50, maxlen=50)
        w.episode_qualified_results = deque(
            [True] * qualified_count + [False] * (50 - qualified_count),
            maxlen=50,
        )

    fill_window(15)                            # 30% -> leave stage 2.1
    assert w._check_phase_advancement() is True
    assert w.current_phase == 2 and w.phase2_stage == 2
    assert w.phases_completed[-1]['phase'] == "2.1"

    w.phase_episode_count = 50
    fill_window(15)                            # 30% -> leave stage 2.2
    assert w._check_phase_advancement() is True
    assert w.current_phase == 2 and w.phase2_stage == 3
    assert w.phases_completed[-1]['phase'] == "2.2"

    w.phase_episode_count = 50
    fill_window(20)                            # 40% -> leave phase 2
    assert w._check_phase_advancement() is True
    assert w.current_phase == 3
    assert w.phases_completed[-1]['phase'] == "2.3"
    assert updates == [2, 3, 3]


def test_history_based_qualified_predicates():
    W = ImprovedPhasedCurriculumWrapper
    base = SimpleNamespace(track_builder=SimpleNamespace(
        history=[{"action": 9}, {"action": 0}, {"action": 10}, {"action": 6}]))
    assert W._history_chain_count(base) == 2
    assert W._history_has_drop(base) is True

    base2 = SimpleNamespace(track_builder=SimpleNamespace(
        history=[{"action": 0}, {"action": 13}]))
    assert W._history_chain_count(base2) == 0
    assert W._history_has_drop(base2) is False


def test_no_terminal_double_count_through_wrapper(monkeypatch):
    monkeypatch.setattr(oe_mod, "APIController", CompletingAPI)
    base = OpenRCT2Env(verbose=0)
    wrapper = ImprovedPhasedCurriculumWrapper(base, verbose=0)
    wrapper.current_phase = 5
    wrapper._update_phase_settings()       # phase 5: quality on, ride testing on
    wrapper.reset()
    base_env = wrapper._get_base_env()
    p = base_env.reward_params
    phi_prev_before, reward, info = None, None, {}
    for _ in range(12):
        phi_prev_before = base_env._phi_prev
        _, reward, terminated, truncated, info = wrapper.step(0)
        if terminated or truncated:
            break
    assert terminated
    rr = info['ride_rating']
    quality = base_env._quality_bonus(rr['excitement'], rr['intensity'], rr['nausea'], p)
    assert quality > 0
    # completion + quality counted EXACTLY once (env owns both; wrapper adds nothing).
    # Jul-9 P5 economics: E=8 fully releases the quality gate (full R_complete), and the
    # terminal step also pays R_viable, all three milestone bars, and the struct credit.
    assert reward == pytest.approx(
        p.R_complete - phi_prev_before + quality + p.step_cost
        + p.R_viable + 3 * p.R_exc_milestone + base_env._last_struct_bonus)


# ----------------------------------------------------- gamma single source (test 9)

def test_gamma_single_sourced_to_reward_params(monkeypatch):
    import train as T
    # The model discount is sourced from RewardParams (the same class the env uses).
    assert T.GAMMA == RewardParams().gamma

    monkeypatch.setattr(oe_mod, "APIController", FakeAPI)
    env = T.create_curriculum_masked_env(8080, verbose=0)
    base = env
    while hasattr(base, "env"):
        base = base.env
    # The env the PPO model trains against discounts its PBRS potential with the same gamma.
    assert base.reward_params.gamma == T.GAMMA


# ----------------------------------------------- entropy-collapse guard (Change E)

def _make_guard_cb(ent_coef=0.015, target_kl=0.04):
    import train as T
    cb = T.ParallelCurriculumMaskableCallback.__new__(T.ParallelCurriculumMaskableCallback)
    cb._opt_guarded = True            # phase >= 2: base restores to the guarded floor
    cb._ent_boosted = False
    cb._ent_boost_calls = 0
    cb._phase = 1                     # __init__ defaults (skipped by __new__); non-2.1 -> guarded base
    cb._phase2_stage = None
    cb.model = SimpleNamespace(ent_coef=ent_coef, target_kl=target_kl)
    return cb


def test_entropy_guard_boost_is_gentler():
    import train as T
    assert T.ENT_COLLAPSE_BOOST == 0.03      # gentler than the old 0.05 that cratered closure
    cb = _make_guard_cb()
    cb._maybe_guard_entropy_collapse(0.05)   # below LO -> boost
    assert cb._ent_boosted and cb.model.ent_coef == T.ENT_COLLAPSE_BOOST


def test_entropy_guard_holds_boost_through_cooldown():
    import train as T
    cb = _make_guard_cb()
    cb._maybe_guard_entropy_collapse(0.05)               # boost
    cb._maybe_guard_entropy_collapse(0.40, kl=0.0)       # recovered immediately -> must NOT relax yet
    assert cb._ent_boosted and cb.model.ent_coef == T.ENT_COLLAPSE_BOOST
    for _ in range(T.ENT_BOOST_MIN_HOLD):                # ride out the min-hold
        cb._maybe_guard_entropy_collapse(0.40, kl=0.0)
    assert not cb._ent_boosted                           # now relaxes
    assert cb.model.ent_coef == T.OPT_GUARDED['ent_coef']


def test_entropy_guard_relax_is_kl_aware():
    import train as T
    cb = _make_guard_cb(target_kl=0.04)
    cb._maybe_guard_entropy_collapse(0.05)               # boost
    for _ in range(T.ENT_BOOST_MIN_HOLD + 1):
        cb._maybe_guard_entropy_collapse(0.40, kl=0.20)  # recovered + hold elapsed BUT KL too high
    assert cb._ent_boosted                               # do not hand back control mid-explosion
    cb._maybe_guard_entropy_collapse(0.40, kl=0.0)       # KL safe now -> relax
    assert not cb._ent_boosted


# ----------------------------------------------- review-driven coverage (edge paths)

class FlakyAPI(FakeAPI):
    """FakeAPI whose agent placements can be made to fail on demand, to drive the
    auto-backtrack path (3 consecutive place failures -> forced remove)."""
    def __init__(self, host=None, port=None, verbose=0):
        super().__init__(host, port, verbose)
        self.fail_places = False

    def place_track_piece(self, x, y, z, direction, track_type, has_chain=False):
        if self.fail_places and track_type not in (1, 2, 3):   # fail agent pieces only
            return {"success": False, "error": "blocked"}
        return super().place_track_piece(x, y, z, direction, track_type, has_chain)


def test_autobacktrack_forced_remove_nets_non_positive(monkeypatch):
    """deliberate-fail -> auto-backtrack remove must not be farmable (plan test 10b)."""
    monkeypatch.setattr(oe_mod, "APIController", FlakyAPI)
    env = OpenRCT2Env(verbose=0)
    env.skip_ride_testing = True
    env.reset()
    _, r_place, *_ = env.step(0)        # one real placement to give the remove something to undo
    env.api_controller.fail_places = True
    rewards, info = [], {}
    for _ in range(3):                   # 3 consecutive failures -> forced auto-backtrack remove
        _, r, term, trunc, info = env.step(0)
        rewards.append(r)
        if term or trunc:
            break
    assert info.get('auto_backtracked') is True
    assert r_place + sum(rewards) < 0    # place + failures + forced remove never nets positive


def test_calibration_seeds_phi_prev_with_calibrated_target_next_reset(monkeypatch):
    monkeypatch.setattr(oe_mod, "APIController", CompletingAPI)
    env = OpenRCT2Env(verbose=0)
    env.skip_ride_testing = True         # calibration still captured on completion
    # Robust calibration: one completion no longer locks the anchor -- it takes several
    # consistent (reproducible) closures, so a fluky first closure can't poison Phi.
    for _ in range(OpenRCT2Env._CLOSE_MIN_CONSISTENT):
        env.reset()
        assert env.close_pos is None     # provisional until the anchor locks
        _drive_to_terminal(env)          # completes -> a closing record is recorded
    assert OpenRCT2Env._close_cache is not None   # enough agreeing closures -> locked
    captured_pos = list(OpenRCT2Env._close_cache["pos"])
    captured_dir = OpenRCT2Env._close_cache["dir"]

    env.reset()                          # next episode applies the calibration
    assert env.close_pos == captured_pos
    assert env.close_dir == captured_dir
    # _phi_prev is seeded from the calibrated target (post-station-build head)
    assert env._phi_prev == pytest.approx(env._potential(env.reward_params))


def test_remove_on_empty_history_returns_fail_penalty_without_drift(monkeypatch):
    monkeypatch.setattr(oe_mod, "APIController", FakeAPI)
    env = OpenRCT2Env(verbose=0)
    env.skip_ride_testing = True
    env.reset()                          # station pieces are not in track history
    phi_prev_before = env._phi_prev
    _, reward, *_ = env.step(31)         # remove with empty agent history -> fails
    assert reward == pytest.approx(env.reward_params.fail_penalty)
    assert env._phi_prev == phi_prev_before


def test_step_cost_applied_on_normal_placement(monkeypatch):
    monkeypatch.setattr(oe_mod, "APIController", FakeAPI)
    env = OpenRCT2Env(verbose=0)
    env.skip_ride_testing = True
    env.reward_params = RewardParams(step_cost=-0.01)
    env.reset()
    p = env.reward_params
    phi_prev_before = env._phi_prev
    _, reward, *_ = env.step(0)
    assert reward == pytest.approx(p.gamma * env._potential(p) - phi_prev_before + p.step_cost)


def test_corrupted_calibration_record_is_ignored():
    import json
    bad = {"pos": [1, 2, 14], "dir": 7, "action": 13, "track_type": 9}   # dir out of range
    with open(OpenRCT2Env._CLOSE_CACHE_PATH, "w") as f:
        json.dump(bad, f)
    OpenRCT2Env._close_cache = None
    env = _bare_env(goal_position=(62, 66, 14))
    env._init_closing_target()
    assert env.close_pos is None         # corrupted -> ignored, falls back to provisional guide tile
    assert env.close_dir == 0            # provisional dir is the deterministic station-entry axis (North)


def test_quality_gate_only_fires_on_all_zero():
    env = _bare_env()
    p = RewardParams(R_quality_max=500.0)
    assert env._quality_bonus(0.0, 0.0, 0.0, p) == 0.0   # untested-ride sentinel -> gated
    assert env._quality_bonus(0.0, 5.5, 1.0, p) > 0.0    # partial-zero is a real ride -> scored
    # quality is always non-negative, so a completed ride is never punished
    assert env._quality_bonus(15.0, 12.0, 9.0, p) >= 0.0


def test_corrupted_calibration_does_not_block_recalibration():
    """A bad logs/close_geometry.json must NOT poison the in-memory cache, else
    _maybe_capture_closing_geometry would early-exit forever and never self-repair."""
    import json
    with open(OpenRCT2Env._CLOSE_CACHE_PATH, "w") as f:
        json.dump({"pos": [1, 2, 14], "dir": 7}, f)   # corrupted: dir out of range
    OpenRCT2Env._close_cache = None
    env = _bare_env()
    env._init_closing_target()
    assert env.close_pos is None                       # ignored -> provisional
    assert OpenRCT2Env._close_cache is None            # bad record not cached -> capture unblocked

    # subsequent reproducible completions now calibrate (self-repair)
    env.loop_completed = True
    for _ in range(OpenRCT2Env._CLOSE_MIN_CONSISTENT):
        env.track_builder = SimpleNamespace(history=_completing_history())
        env._maybe_capture_closing_geometry()
    assert OpenRCT2Env._close_cache is not None
    assert OpenRCT2Env._close_cache["pos"] == [61, 68, 14]


def test_phase5_episode_metrics_include_quality_bonus(monkeypatch):
    """episode_rewards / phase_rewards must match the reward actually returned to PPO,
    which includes the terminal quality bonus (added after _calculate_reward)."""
    monkeypatch.setattr(oe_mod, "APIController", CompletingAPI)
    env = OpenRCT2Env(verbose=0)
    env.skip_ride_testing = False
    env.reward_params = RewardParams(R_quality_max=500.0)
    env.reset()
    _, reward, terminated, _, _ = _drive_to_terminal(env)
    assert terminated
    assert env.episode_rewards[-1] == pytest.approx(reward)                    # not under-reported
    assert sum(env.phase_rewards.values()) == pytest.approx(sum(env.episode_rewards))


# --------------------------------------------- route potential (west-side detour shaping)
# The approach cone gives ZERO horizontal pull for along<0 (the whole start side), so the
# detour AROUND the station was unshaped: the 1.3M-step Jun-24 run parked at ~5 tiles and
# never completed. Phi gains a bounded angular-progress term (w_route * bearing progress
# around the station center, 0 on the start/west bearing -> 1 on the approach/east bearing)
# that is monotone along BOTH detours. Pure function of current_position -> PBRS-clean.

def _route_env(pos, direction=0):
    """Bare env at the REAL station geometry ([61,66,14], length 6) with the calibrated
    closing target at the staging tile, matching a live phase-1 episode."""
    env = _bare_env(current_position=pos, current_direction=direction,
                    goal_position=(62, 66, 14), history=[])
    env.close_pos = [62, 66, 14]
    env.close_dir = 0
    return env


def test_route_progress_zero_on_start_side_full_on_approach_side():
    assert _route_env((55, 66, 14))._route_progress() == pytest.approx(0.0)   # post-station head
    assert _route_env((62, 66, 14))._route_progress() == pytest.approx(1.0)   # staging tile
    assert _route_env((70, 66, 14))._route_progress() == pytest.approx(1.0)   # radius-independent


def test_route_progress_monotone_on_both_detours():
    """Strictly increasing along a real racetrack path (live-verified waypoints) around the
    NORTH side, and its mirror around the SOUTH side -- both detours get a gradient."""
    north = [(55, 66, 14), (54, 68, 14), (56, 69, 14), (59, 69, 14),
             (63, 69, 14), (64, 67, 14), (62, 66, 14)]
    south = [(x, 66 - (y - 66), z) for (x, y, z) in north]
    for path in (north, south):
        vals = [_route_env(p)._route_progress() for p in path]
        assert all(b > a for a, b in zip(vals, vals[1:]))
        assert vals[0] == pytest.approx(0.0) and vals[-1] == pytest.approx(1.0)


def test_route_potential_off_by_default_and_bounded():
    assert RewardParams().w_route == 0.0          # kill-switch default: all frozen-Phi tests hold
    iso = RewardParams(w_xy=0.0, w_z=0.0, w_dir=0.0, w_e=0.0, w_h=0.0,
                       w_close=0.0, w_route=3.0)  # isolate the route term
    for pos in ((55, 66, 14), (56, 69, 14), (62, 66, 14), (80, 90, 14)):
        phi = _route_env(pos)._potential(iso)
        assert 0.0 <= phi <= 3.0 + 1e-9           # bounded by w_route
    assert _route_env((62, 66, 14))._potential(iso) == pytest.approx(3.0)
    assert _route_env((55, 66, 14))._potential(iso) == pytest.approx(0.0)


def test_route_potential_telescopes_on_place_remove(monkeypatch):
    """Route term is part of the single Phi -> place+remove telescopes to (gamma-1)*(Phi+Phi') < 0
    (not farmable), exactly like the rest of the potential."""
    monkeypatch.setattr(oe_mod, "APIController", FakeAPI)
    env = OpenRCT2Env(verbose=0)
    env.skip_ride_testing = True
    env.reward_params = RewardParams(w_xy=0.0, w_z=0.0, w_dir=0.0, w_e=0.0, w_h=0.0,
                                     w_close=0.0, w_route=3.0)
    env.reset()
    _, r_place, *_ = env.step(0)
    _, r_remove, *_ = env.step(31)
    assert r_place + r_remove < 0


def test_route_potential_no_new_parking_optimum():
    """Under full phase-1 params the route term must not create a resting place: Phi at the
    docked (staging, aligned) state strictly dominates every detour waypoint."""
    p1 = ImprovedPhasedCurriculumWrapper._phase_reward_params(1)
    assert p1.w_route > 0.0
    docked = _route_env((62, 66, 14), direction=0)._potential(p1)
    for probe in ((55, 66, 14), (54, 68, 14), (56, 69, 14), (59, 69, 14),
                  (63, 69, 14), (56, 63, 14), (59, 63, 14), (70, 66, 14)):
        assert _route_env(probe)._potential(p1) < docked


def test_phase_params_enable_route_in_completion_phases():
    """w_route follows the w_close pattern: ON in the completion-learning phases 1-4, OFF in
    phase 5 (completion mastered; quality phase keeps Phi lean)."""
    W = ImprovedPhasedCurriculumWrapper
    assert W._phase_reward_params(1).w_route == 3.0
    for s in (1, 2, 3):
        assert W._phase_reward_params(2, phase2_stage=s).w_route == 3.0
    assert W._phase_reward_params(3).w_route == 3.0
    assert W._phase_reward_params(4).w_route == 3.0
    assert W._phase_reward_params(5).w_route == 0.0


def test_episode_metrics_expose_route_potential(monkeypatch):
    """Diagnostic-per-term: the route term's episode-end value is surfaced in episode_metrics
    so the training callback can log rewards/route_potential."""
    monkeypatch.setattr(oe_mod, "APIController", CompletingAPI)
    env = OpenRCT2Env(verbose=0)
    env.skip_ride_testing = True
    env.reward_params = ImprovedPhasedCurriculumWrapper._phase_reward_params(1)
    env.reset()
    _, _, terminated, truncated, info = _drive_to_terminal(env)
    assert terminated or truncated
    m = info['episode_metrics']
    assert 'route_potential' in m
    assert 0.0 <= m['route_potential'] <= env.reward_params.w_route + 1e-9


# --------------------------------------------- roundtrip degeneracy + hill elevation

def test_roundtrip_gain1_requires_actual_descent():
    """At roundtrip_gain=1 the +1 return tolerance CONTAINED the 1-z summit: a single
    action-10 stub fired summit AND roundtrip (and the 2.1 gate) with no return ever.
    The return must now be strictly below the required-climb threshold."""
    p = RewardParams(R_roundtrip=100.0, roundtrip_gain=1.0)
    at_summit = _roundtrip_env(peak_z=15, head_z=15, p=p)     # placed one stub, still on it
    at_summit._calculate_reward(True, 0)
    assert at_summit._roundtrip_awarded is False              # no descent -> no round-trip
    returned = _roundtrip_env(peak_z=15, head_z=14, p=p)      # actually came back down
    returned._calculate_reward(True, 0)
    assert returned._roundtrip_awarded is True


def test_roundtrip_not_awarded_on_dive_below_station():
    """The env paid a chain-climb-then-dive (z<=station+1 had no lower bound) while the
    wrapper's mirror (abs<=1) did not count it -- silent gate/reward disagreement."""
    p = RewardParams(R_roundtrip=100.0, roundtrip_gain=4.0)
    dived = _roundtrip_env(peak_z=20, head_z=11, p=p)         # 3 below station
    dived._calculate_reward(True, 0)
    assert dived._roundtrip_awarded is False


def test_hill_quality_scales_with_chain_elevation():
    """Chain PIECES alone are farmable: three scattered 1-z stubs must not equal a full
    lift hill for the completion gate / structural bonus. Quality is elevation-scaled
    against the stage's climb bar (roundtrip_gain)."""
    p = RewardParams(R_struct_max=250.0, struct_chain_target=3, struct_w_chain=1.0,
                     struct_w_drop=0.0, roundtrip_gain=3.0)
    stubs = _bare_env(history=[
        {"action": 9, "position": [i, 0, 14], "next_position": [i, 0, 15]} for i in range(3)
    ])                                                        # 3 chain pieces, peak +1
    full = _struct_env(chains=3)                              # 3 chain pieces climbing to +3
    assert full._structural_bonus(p) == pytest.approx(250.0)
    assert stubs._structural_bonus(p) == pytest.approx(250.0 / 3)   # count 1.0 x gain 1/3
    assert stubs._hill_quality(p) < full._hill_quality(p)


def test_stage23_climb_bar_matches_canonical_hill():
    """The canonical hill [10,9,13] banks chain-gain 3 (the crest piece isn't chained);
    a 4.0 bar made the 2.3/P3/P4 roundtrip milestone and w_return descent shaping
    silently inert for the exact hill the curriculum teaches."""
    W = ImprovedPhasedCurriculumWrapper
    assert RewardParams().roundtrip_gain == 3.0
    assert W._phase_reward_params(2, phase2_stage=3).roundtrip_gain == 3.0
    for phase in (3, 4):
        assert W._phase_reward_params(phase).roundtrip_gain == 3.0


def test_autobacktrack_still_pays_the_failure_penalty(monkeypatch):
    """The forced remove used to REPLACE the fail penalty with the remove's PBRS delta --
    the agent's chosen (failed) action escaped its penalty whenever auto-backtrack fired."""
    monkeypatch.setattr(oe_mod, "APIController", FlakyAPI)
    env = OpenRCT2Env(verbose=0)
    env.skip_ride_testing = True
    env.reset()
    env.step(0)                                # one real piece so the remove has a target
    env.api_controller.fail_places = True
    rewards = []
    for _ in range(3):                         # 3rd failure triggers the forced remove
        _, r, _, _, info = env.step(0)
        rewards.append(r)
    assert info['auto_backtracked'] is True
    p = env.reward_params
    phi_term = rewards[-1] - p.fail_penalty    # forced-remove step = PBRS delta + fail penalty
    assert rewards[-1] < phi_term              # the penalty is actually included


def test_failed_remove_does_not_reset_failure_counter(monkeypatch):
    monkeypatch.setattr(oe_mod, "APIController", FakeAPI)
    env = OpenRCT2Env(verbose=0)
    env.skip_ride_testing = True
    env.reset()
    env.consecutive_failures = 2
    env.step(31)                               # empty history -> remove fails
    assert env.consecutive_failures == 2       # a failed remove is not a recovery


def test_probe_log_stops_after_anchor_locks():
    """closing_probe.jsonl exists to confirm the closing geometry empirically; once the
    anchor is locked its purpose is fulfilled -- unbounded per-completion appends over a
    multi-million-step run are pure disk growth."""
    import os
    OpenRCT2Env._close_cache = {"pos": [62, 66, 14], "dir": 0, "action": 0, "track_type": 0}
    env = _bare_env()
    env.loop_completed = True
    env.track_builder = SimpleNamespace(history=_completing_history())
    env._maybe_capture_closing_geometry()
    path = os.path.join(os.path.dirname(OpenRCT2Env._CLOSE_CACHE_PATH), "closing_probe.jsonl")
    assert not os.path.exists(path)            # locked anchor -> no more probe lines


# --------------------------------------------- P3-5 redesign: structure quality + quality ramp

def _tall_hill_env(track_len_pad=0):
    """History with a +5z chain climb and a full 5z descent (drop_z 5), optionally padded
    with flat pieces to stretch track_length."""
    hist = [
        {"action": 10, "position": [0, 0, 14], "next_position": [1, 0, 15]},
        {"action": 9, "position": [1, 0, 15], "next_position": [2, 0, 17]},
        {"action": 9, "position": [2, 0, 17], "next_position": [3, 0, 19]},
        {"action": 12, "position": [3, 0, 19], "next_position": [4, 0, 18]},
        {"action": 6, "position": [4, 0, 18], "next_position": [5, 0, 16]},
        {"action": 6, "position": [5, 0, 16], "next_position": [6, 0, 14]},
    ] + [{"action": 0, "position": [6 + i, 0, 14], "next_position": [7 + i, 0, 14]}
         for i in range(track_len_pad)]
    return _bare_env(history=hist)


def test_total_drop_z_sums_descents_only():
    env = _tall_hill_env()
    assert env._total_drop_z() == pytest.approx(5.0)      # 1 + 2 + 2, climbs ignored


def test_structure_quality_grades_height_drop_length():
    """P3/P4 structural bonus components: chain height, total drop-z, completed length,
    each ramping toward per-phase targets (partial progress pays -- no cliffs)."""
    p = RewardParams(R_struct_max=250.0, struct_w_chain=0.0, struct_w_height=0.4,
                     struct_height_target=4.0, struct_w_drop=0.4, struct_drop_target=4.0,
                     struct_w_length=0.2, struct_length_target=25.0)
    env = _tall_hill_env()                                # height 5, drop 5, length 6
    expected = 0.4 * 1.0 + 0.4 * 1.0 + 0.2 * (6 / 25)
    assert env._hill_quality(p) == pytest.approx(expected)
    long_env = _tall_hill_env(track_len_pad=19)           # length 25 -> full length credit
    assert long_env._hill_quality(p) == pytest.approx(1.0)
    small = _struct_env(chains=1)                         # 1z stub, no real drop, tiny
    assert small._hill_quality(p) < 0.3


def test_structure_quality_defaults_preserve_legacy_behavior():
    """New component weights default to 0: every pre-redesign params object computes the
    exact same hill quality as before."""
    p = RewardParams(R_struct_max=250.0, struct_chain_target=3, struct_w_chain=1.0,
                     struct_w_drop=0.0)
    assert _struct_env(chains=3)._structural_bonus(p) == pytest.approx(250.0)


def test_quality_ramp_pays_partial_progress():
    """The Phase-5 plateau: exc ~1.5 vs a band at 8 paid ZERO for any improvement short of
    ~6 (two runs converged to the identical +90 nausea-only bonus). The ramp half pays
    every increment from 0 up; the band half still peaks at the target."""
    env = _bare_env()
    p = RewardParams(R_quality_max=500.0)
    q = lambda e: env._quality_bonus(e, 5.5, 1.0, p)      # intensity/nausea held at target
    assert q(2.0) > q(1.5) > q(1.0)                       # gradient exists at the plateau
    assert q(2.0) - q(1.0) > 10.0                         # ...and is material, not epsilon
    assert q(8.0) == max(q(e) for e in (1, 2, 4, 6, 7, 8))  # still peaks at the target
    assert q(8.0) > 0.95 * (p.q_w_exc + p.q_w_int + p.q_w_nausea * 0.97) * 500 * 0.95


def test_quality_overshoot_halves_not_zeroes():
    """Above-target stats keep the ramp half (an 8.5-excitement coaster is not worthless)
    while the band half decays -- replaces the old symmetric falloff."""
    env = _bare_env()
    p = RewardParams(R_quality_max=500.0)
    at = env._quality_bonus(8.0, 5.5, 1.0, p)
    over = env._quality_bonus(11.0, 5.5, 1.0, p)
    under = env._quality_bonus(5.0, 5.5, 1.0, p)
    assert under < over < at                              # overshoot beats equal undershoot
    assert over > 0.55 * at                               # but keeps at least the ramp half


def test_r_viable_paid_only_when_ride_test_returns_stats(monkeypatch):
    """P4's 'the train physically made it around' bonus: paid on completion ONLY when the
    ride test came back with nonzero stats."""
    monkeypatch.setattr(oe_mod, "APIController", CompletingAPI)
    env = OpenRCT2Env(verbose=0)
    env.skip_ride_testing = False                         # CompletingAPI serves stats instantly
    env.reward_params = RewardParams(R_viable=150.0, roundtrip_gain=0.0)
    env.reset()
    phi_prev_before, reward, terminated, _, info = _drive_to_terminal(env)
    assert terminated
    p = env.reward_params
    rr = info['ride_rating']
    quality = env._quality_bonus(rr['excitement'], rr['intensity'], rr['nausea'], p)
    assert reward == pytest.approx(p.R_complete - phi_prev_before + quality + 150.0)
    assert env._last_test_ok is True
    assert info['episode_metrics']['test_ok'] is True

    env2 = OpenRCT2Env(verbose=0)
    env2.skip_ride_testing = True                         # untested -> all-zero stats
    env2.reward_params = RewardParams(R_viable=150.0, roundtrip_gain=0.0)
    env2.reset()
    phi_prev_before, reward, terminated, _, info = _drive_to_terminal(env2)
    assert terminated
    assert reward == pytest.approx(env2.reward_params.R_complete - phi_prev_before)  # no bonus
    assert env2._last_test_ok is False


def test_episode_metrics_expose_structure_diagnostics(monkeypatch):
    monkeypatch.setattr(oe_mod, "APIController", CompletingAPI)
    env = OpenRCT2Env(verbose=0)
    env.skip_ride_testing = True
    env.reset()
    info = {}
    for _ in range(6):
        _, _, terminated, truncated, info = env.step(9)
        if terminated or truncated:
            break
    m = info['episode_metrics']
    assert {'drop_z', 'chain_height', 'test_ok'}.issubset(m)


# --------------------------------------------- P3/P4 scale phases + P5 quality phase

def test_phase34_scale_params_and_p5_step_cost():
    """P3 'Real Drops & Scale' and P4 'Big & Verified': structure credit moves from piece
    counting to graded height/drop/length toward per-phase targets; P4 pays the verified-
    viability bonus and turns ride testing on; P5 stops punishing length (step_cost 0)."""
    W = ImprovedPhasedCurriculumWrapper
    p3 = W._phase_reward_params(3)
    assert (p3.struct_w_height, p3.struct_height_target) == (0.4, 4.0)
    assert (p3.struct_w_drop, p3.struct_drop_target) == (0.4, 4.0)
    assert (p3.struct_w_length, p3.struct_length_target) == (0.2, 25.0)
    assert p3.struct_w_chain == 0.0 and p3.R_viable == 0.0
    assert p3.h_scale == 8.0                      # taller climbs keep paying discovery
    p4 = W._phase_reward_params(4)
    assert (p4.struct_w_height, p4.struct_height_target) == (0.3, 6.0)
    assert (p4.struct_w_drop, p4.struct_drop_target) == (0.3, 8.0)
    assert (p4.struct_w_length, p4.struct_length_target) == (0.2, 40.0)
    assert p4.R_viable == 150.0 and p4.h_scale == 8.0
    # completion-conditioned bonuses stay strictly below R_complete
    assert p4.R_struct_max + p4.R_viable < p4.R_complete
    p5 = W._phase_reward_params(5)
    assert p5.step_cost == 0.0                    # the one phase that WANTS length


def _scale_base(actions_z, head_z=14, track_length=None, energy=10.0, test_ok=True):
    """SimpleNamespace base env for the qualified predicates: actions_z is a list of
    (action, entry_z, exit_z)."""
    hist = [{"action": a, "position": [i, 0, z0], "next_position": [i + 1, 0, z1]}
            for i, (a, z0, z1) in enumerate(actions_z)]
    return SimpleNamespace(
        track_builder=SimpleNamespace(history=hist),
        current_position=[0, 0, head_z],
        track_length=len(hist) if track_length is None else track_length,
        STATION_HEIGHT=14,
        _calculate_energy_margin=lambda: energy,
        _last_test_ok=test_ok,
        _summit_awarded=False, _roundtrip_awarded=False,
    )


def _big_hill(chain_peak=19, steep=False, length=30):
    """(action, z_in, z_out) rows: chain climb to chain_peak, descent back to 14, flat pad."""
    rows, z, a_idx = [], 14, 0
    rows.append((10, z, z + 1)); z += 1
    while z < chain_peak:
        rows.append((9, z, min(z + 2, chain_peak))); z = min(z + 2, chain_peak)
    if steep:
        rows += [(12, z, z - 1), (27, z - 1, z - 5), (28, z - 5, z - 9), (14, z - 9, z - 10)]
        z -= 10
    while z > 14:
        rows.append((6, z, max(z - 2, 14))); z = max(z - 2, 14)
    while len(rows) < length:
        rows.append((0, 14, 14))
    return rows


def test_p3_qualified_requires_height_drop_length_energy():
    W = ImprovedPhasedCurriculumWrapper
    w = W.__new__(W)
    w.current_phase = 3
    good = _scale_base(_big_hill(chain_peak=19, length=26))     # h5, drop 5, len 26
    assert w._is_qualified(good, True) is True
    assert w._is_qualified(good, False) is False                # must complete
    small = _scale_base(_big_hill(chain_peak=17, length=26))    # h3 < 4
    assert w._is_qualified(small, True) is False
    short = _scale_base(_big_hill(chain_peak=19, length=10), track_length=10)
    assert w._is_qualified(short, True) is False                # len < 25
    stalled = _scale_base(_big_hill(chain_peak=19, length=26), energy=-5.0)
    assert w._is_qualified(stalled, True) is False              # energy proxy says the train dies


def test_p4_qualified_requires_steep_and_verified():
    W = ImprovedPhasedCurriculumWrapper
    w = W.__new__(W)
    w.current_phase = 4
    good = _scale_base(_big_hill(chain_peak=25, steep=True, length=41))   # h11, drop>=10+steep
    assert w._is_qualified(good, True) is True
    unverified = _scale_base(_big_hill(chain_peak=25, steep=True, length=41), test_ok=False)
    assert w._is_qualified(unverified, True) is False           # train never demonstrably ran
    no_steep = _scale_base(_big_hill(chain_peak=25, steep=False, length=41))
    assert w._is_qualified(no_steep, True) is False             # 60-degree segment required


def test_p4_advancement_uses_qualified_window():
    from collections import deque
    W = ImprovedPhasedCurriculumWrapper
    w = W.__new__(W)
    w.current_phase = 4
    w.phase4_success_threshold = 0.30
    w._track_stats = True
    w.verbose = 0
    w.phases_completed = []
    w.phase_episode_count = 50
    w.total_loops_completed = 0
    w.episode_results = deque([True] * 50, maxlen=50)           # raw completions maxed...
    w.episode_qualified_results = deque([False] * 50, maxlen=50)  # ...but nothing qualifies
    advanced = []
    w._advance_to_phase = lambda p: advanced.append(p)
    assert w._check_phase_advancement() is False                # raw success must NOT advance P4
    w.episode_qualified_results = deque([True] * 20 + [False] * 30, maxlen=50)   # 40% qualified
    w._check_phase_advancement()
    assert advanced == [5]


# ------------------------------- P3/P4 length-trap fix (length gate + qualify bonus)
# The Jul-5 overnight run converged onto an 18-piece mini-loop in Phase 3: the additive
# length credit (+2/piece) lost to gamma-discounting the ~1200-point completion payout
# (-10/piece), so qualified_rate decayed 0.14 -> 0 while reward plateaued at its max.
# Fix: (a) a multiplicative completion length gate (mirrors completion_hill_floor), and
# (b) a discrete R_qualify bonus paid when the episode meets the phase's qualified
# predicate -- the gate the curriculum advances on finally shows up in the reward.

def _env_hist(rows):
    """(action, z_in, z_out) rows -> env-format history dicts (see _big_hill)."""
    return [{"action": a, "position": [i, 0, z0], "next_position": [i + 1, 0, z1]}
            for i, (a, z0, z1) in enumerate(rows)]


def test_qualify_fields_default_inert():
    """New fields must default OFF so every pre-fix params object (P1/P2/P5) is unchanged."""
    p = RewardParams()
    assert p.completion_length_floor == 1.0
    assert p.R_qualify == 0.0
    assert p.qualify_requires_energy is False
    assert p.qualify_requires_steep_drop is False
    assert p.qualify_requires_test is False


def test_completion_length_gate_scales_r_complete():
    """A completed loop below struct_length_target earns only the length-gated fraction of
    R_complete; at the target the gate releases fully."""
    params = replace(RewardParams(), completion_length_floor=0.25, struct_length_target=25.0)

    def payout(length):
        env = _bare_env(history=_env_hist(_big_hill(chain_peak=19, length=length)))
        env.loop_completed = True
        env._phi_prev = 0.0
        env.reward_params = params
        return env._calculate_reward(True, 0)

    gate18 = 0.25 + 0.75 * (18 / 25)
    assert payout(18) == pytest.approx(params.R_complete * gate18)
    assert payout(25) == pytest.approx(params.R_complete)


def test_length_gate_composes_with_hill_gate():
    """P3-style gating multiplies: hill quality gates R_complete AND the short-loop
    discount applies on top (a hill-ful mini-loop still leaves length money on the table)."""
    params = replace(
        RewardParams(),
        completion_hill_floor=0.0, completion_length_floor=0.25,
        struct_w_chain=0.0, struct_w_height=0.4, struct_height_target=4.0,
        struct_w_drop=0.4, struct_drop_target=4.0,
        struct_w_length=0.2, struct_length_target=25.0,
    )
    env = _bare_env(history=_env_hist(_big_hill(chain_peak=19, length=18)))  # h5, drop 5, len 18
    env.loop_completed = True
    env._phi_prev = 0.0
    env.reward_params = params
    hill_q = 0.4 + 0.4 + 0.2 * (18 / 25)
    length_gate = 0.25 + 0.75 * (18 / 25)
    assert env._calculate_reward(True, 0) == pytest.approx(
        params.R_complete * hill_q * length_gate)


def test_qualify_predicate_checks_structure_energy_steep_test():
    params = replace(RewardParams(), R_qualify=200.0, struct_height_target=4.0,
                     struct_drop_target=4.0, struct_length_target=25.0,
                     qualify_requires_energy=True)

    def env_for(rows, energy=10.0, test_ok=True):
        env = _bare_env(history=_env_hist(rows))
        env._calculate_energy_margin = lambda: energy
        env._last_test_ok = test_ok
        return env

    assert env_for(_big_hill(chain_peak=19, length=26))._qualifies(params) is True
    assert env_for(_big_hill(chain_peak=17, length=26))._qualifies(params) is False  # h3 < 4
    assert env_for(_big_hill(chain_peak=19, length=18))._qualifies(params) is False  # len < 25
    assert env_for(_big_hill(chain_peak=19, length=26), energy=-5.0)._qualifies(params) is False
    p4ish = replace(params, qualify_requires_energy=False,
                    qualify_requires_steep_drop=True, qualify_requires_test=True)
    assert env_for(_big_hill(chain_peak=25, steep=True, length=41))._qualifies(p4ish) is True
    assert env_for(_big_hill(chain_peak=25, steep=False, length=41))._qualifies(p4ish) is False
    assert env_for(_big_hill(chain_peak=25, steep=True, length=41),
                   test_ok=False)._qualifies(p4ish) is False


def test_r_qualify_paid_in_terminal_step(monkeypatch):
    """R_qualify is paid in the terminal step (next to R_viable) so P4 can require the
    ride test; a completion short of the length target earns nothing."""
    monkeypatch.setattr(oe_mod, "APIController", CompletingAPI)
    env = OpenRCT2Env(verbose=0)
    env.skip_ride_testing = True
    env.reward_params = replace(RewardParams(), R_qualify=200.0, struct_height_target=0.0,
                                struct_drop_target=0.0, struct_length_target=2.0)
    env.reset()
    phi_prev_before, reward, terminated, _, info = _drive_to_terminal(env)
    assert terminated
    assert reward == pytest.approx(env.reward_params.R_complete - phi_prev_before + 200.0)
    assert info['episode_metrics']['qualify_bonus'] == 200.0

    env2 = OpenRCT2Env(verbose=0)
    env2.skip_ride_testing = True
    env2.reward_params = replace(env.reward_params, struct_length_target=99.0)
    env2.reset()
    phi_prev_before, reward, terminated, _, info = _drive_to_terminal(env2)
    assert terminated
    assert reward == pytest.approx(env2.reward_params.R_complete - phi_prev_before)
    assert info['episode_metrics']['qualify_bonus'] == 0.0


def test_episode_metrics_expose_completion_gate(monkeypatch):
    monkeypatch.setattr(oe_mod, "APIController", CompletingAPI)
    env = OpenRCT2Env(verbose=0)
    env.skip_ride_testing = True
    env.reset()
    _, _, terminated, _, info = _drive_to_terminal(env)
    assert terminated
    m = info['episode_metrics']
    assert {'qualify_bonus', 'completion_gate'}.issubset(m)
    assert m['completion_gate'] == pytest.approx(1.0)     # default floors -> ungated


def test_p3_p4_params_pay_the_length_gate():
    W = ImprovedPhasedCurriculumWrapper
    p3 = W._phase_reward_params(3)
    assert p3.completion_length_floor == 0.25
    assert p3.R_qualify == 200.0
    assert p3.qualify_requires_energy is True
    assert (p3.qualify_requires_steep_drop, p3.qualify_requires_test) == (False, False)
    p4 = W._phase_reward_params(4)
    assert p4.completion_length_floor == 0.25
    assert p4.R_qualify == 200.0
    assert p4.qualify_requires_energy is False
    assert (p4.qualify_requires_steep_drop, p4.qualify_requires_test) == (True, True)
    # completion-conditioned extras stay strictly below R_complete (completion-first)
    assert p4.R_struct_max + p4.R_viable + p4.R_qualify < p4.R_complete


def _completion_payout(params, length, chain_peak=19, steep=False, test_ok=False):
    """Terminal payout of a completed chain-hill loop under `params`, including the
    step()-level bonuses (R_viable/R_qualify) the terminal branch adds."""
    env = _bare_env(history=_env_hist(_big_hill(chain_peak=chain_peak, steep=steep,
                                                length=length)))
    env.loop_completed = True
    env._phi_prev = 0.0
    env.reward_params = params
    env._calculate_energy_margin = lambda: 10.0     # viable; isolates the length economics
    env._last_test_ok = test_ok
    r = env._calculate_reward(True, 0)
    if test_ok and params.R_viable > 0.0:
        r += params.R_viable
    if params.R_qualify > 0.0 and env._qualifies(params):
        r += params.R_qualify
    return r


def test_p3_extending_to_target_beats_min_loop():
    """THE regression test for the Jul-5 trap: from the 18-piece mini-loop state, the
    discounted value of building the 7 more pieces Phase 3's gate requires must
    decisively beat banking the mini-loop now."""
    P3 = ImprovedPhasedCurriculumWrapper._phase_reward_params(3)
    stay = _completion_payout(P3, length=18)
    extend = (P3.gamma ** 7) * _completion_payout(P3, length=25)
    assert extend > stay + 100.0


def test_p4_extending_to_target_beats_min_loop():
    """Same economics at the Phase-4 bar (length 40): completing a verified 30-piece
    hill now must not beat extending to the qualifying length."""
    P4 = ImprovedPhasedCurriculumWrapper._phase_reward_params(4)
    stay = _completion_payout(P4, length=30, chain_peak=25, steep=True, test_ok=True)
    extend = (P4.gamma ** 10) * _completion_payout(P4, length=40, chain_peak=25,
                                                   steep=True, test_ok=True)
    assert extend > stay + 100.0


# --------------------------- P5 economics: quality-gated completion + milestones
# P5 converged onto a 24-piece E=1.15 loop: completion paid 1000 ungated while the five
# wooden-RC rating caps (single drop >=12z, >=2 drops, speed, airtime, ~370m) were
# reward-invisible. The gate splits completion pay: a floor at close, the remainder
# ramping with MEASURED excitement (paid post-test in the same terminal step); struct
# credit gains cap-aligned components; discrete milestone bars pay E crossings.

class ExcitedAPI(CompletingAPI):
    excitement = 3.0
    intensity = 2.0
    nausea = 1.0


class MildlyExcitedAPI(CompletingAPI):
    excitement = 4.5
    intensity = 3.0
    nausea = 1.0


def test_p5_economics_fields_default_inert():
    p = RewardParams()
    assert p.completion_quality_floor == 1.0
    assert p.exc_gate_target == 6.0
    assert (p.struct_w_single_drop, p.struct_single_drop_target) == (0.0, 12.0)
    assert (p.struct_w_drop_runs, p.struct_drop_runs_target) == (0.0, 2.0)
    assert (p.struct_w_banked, p.struct_banked_target) == (0.0, 4.0)
    assert p.R_exc_milestone == 0.0 and p.exc_milestone_bars == ()


def test_quality_gated_completion_splits_floor_and_ramp(monkeypatch):
    """floor*R_complete at close; the remainder releases with measured excitement:
    full at E>=exc_gate_target, proportional below, nothing when untested."""
    params = replace(RewardParams(), completion_quality_floor=0.4, exc_gate_target=6.0,
                     roundtrip_gain=0.0)

    def run(api, skip):
        monkeypatch.setattr(oe_mod, "APIController", api)
        env = OpenRCT2Env(verbose=0)
        env.skip_ride_testing = skip
        env.reward_params = params
        env.reset()
        return env, _drive_to_terminal(env)

    env, (phi0, reward, terminated, _, info) = run(CompletingAPI, skip=False)   # E=8 -> full
    assert terminated
    quality = env._quality_bonus(8.0, 5.5, 1.0, params)
    assert reward == pytest.approx(params.R_complete - phi0 + quality)
    assert info['episode_metrics']['completion_gate'] == pytest.approx(1.0)

    env, (phi0, reward, terminated, _, info) = run(ExcitedAPI, skip=False)      # E=3 -> half ramp
    assert terminated
    quality = env._quality_bonus(3.0, 2.0, 1.0, params)
    gate = 0.4 + 0.6 * (3.0 / 6.0)
    assert reward == pytest.approx(params.R_complete * gate - phi0 + quality)
    assert info['episode_metrics']['completion_gate'] == pytest.approx(gate)

    env, (phi0, reward, terminated, _, info) = run(CompletingAPI, skip=True)    # untested -> floor
    assert terminated
    assert reward == pytest.approx(0.4 * params.R_complete - phi0)
    assert info['episode_metrics']['completion_gate'] == pytest.approx(0.4)


def test_exc_milestones_pay_staged_bars(monkeypatch):
    params = replace(RewardParams(), R_exc_milestone=100.0,
                     exc_milestone_bars=(2.5, 4.0, 5.5), roundtrip_gain=0.0)

    def run(api, skip=False):
        monkeypatch.setattr(oe_mod, "APIController", api)
        env = OpenRCT2Env(verbose=0)
        env.skip_ride_testing = skip
        env.reward_params = params
        env.reset()
        phi0, reward, terminated, _, info = _drive_to_terminal(env)
        assert terminated
        return env, phi0, reward, info

    env, phi0, reward, info = run(MildlyExcitedAPI)                 # E=4.5 -> bars 2.5, 4.0
    quality = env._quality_bonus(4.5, 3.0, 1.0, params)
    assert reward == pytest.approx(params.R_complete - phi0 + quality + 200.0)
    assert info['episode_metrics']['exc_milestone_bonus'] == pytest.approx(200.0)
    env, phi0, reward, info = run(CompletingAPI)                    # E=8 -> all three
    quality = env._quality_bonus(8.0, 5.5, 1.0, params)
    assert reward == pytest.approx(params.R_complete - phi0 + quality + 300.0)
    _, phi0, reward, info = run(CompletingAPI, skip=True)           # untested -> none
    assert reward == pytest.approx(params.R_complete - phi0)
    assert info['episode_metrics']['exc_milestone_bonus'] == 0.0


def test_hill_quality_pays_cap_aligned_components():
    params = replace(RewardParams(), R_struct_max=250.0, struct_w_chain=0.0,
                     struct_w_drop=0.0, struct_height_target=0.0,
                     struct_w_single_drop=0.5, struct_single_drop_target=12.0,
                     struct_w_drop_runs=0.25, struct_drop_runs_target=2.0,
                     struct_w_banked=0.25, struct_banked_target=4.0)
    rows = [(10, 14, 15), (9, 15, 17), (9, 17, 19), (13, 19, 20),   # chain to +6
            (12, 20, 19), (6, 19, 17), (6, 17, 15), (14, 15, 14),   # one 6z run
            (21, 14, 14), (24, 14, 14)]                             # two banked turns
    env = _bare_env(history=_env_hist(rows))
    # single drop 6/12 = .5, runs 1/2 = .5, banked 2/4 = .5 -> weighted sum = 0.5
    assert env._hill_quality(params) == pytest.approx(0.5)


def test_validate_completion_first_folds_quality_floor():
    W = ImprovedPhasedCurriculumWrapper
    bad = replace(RewardParams(), completion_hill_floor=0.5,
                  completion_quality_floor=0.2, R_roundtrip=150.0)
    with pytest.raises(AssertionError):
        W._validate_completion_first(bad, "test")                   # 150 >= .5*.2*1000
    ok = replace(bad, R_roundtrip=50.0)
    W._validate_completion_first(ok, "test")


def test_p5_params_pay_the_quality_gate():
    p5 = ImprovedPhasedCurriculumWrapper._phase_reward_params(5)
    assert (p5.completion_quality_floor, p5.exc_gate_target) == (0.4, 6.0)
    assert p5.R_struct_max == 250.0 and p5.struct_w_chain == 0.0
    assert (p5.struct_w_single_drop, p5.struct_single_drop_target) == (0.30, 12.0)
    assert (p5.struct_w_drop_runs, p5.struct_drop_runs_target) == (0.20, 2.0)
    assert (p5.struct_w_drop, p5.struct_drop_target) == (0.15, 16.0)
    # length target 70, not 60: probe_measurements measured 5.5 m/piece live (Jul-10),
    # so the game's ~370m length cap sits near ~67 pieces -- the static ramp must not
    # saturate a hundred metres short of the cap it proxies.
    assert (p5.struct_w_length, p5.struct_length_target) == (0.20, 70.0)
    assert (p5.struct_w_banked, p5.struct_banked_target) == (0.15, 4.0)
    assert p5.struct_w_single_drop + p5.struct_w_drop_runs + p5.struct_w_drop \
        + p5.struct_w_length + p5.struct_w_banked == pytest.approx(1.0)
    assert p5.R_viable == 150.0
    assert (p5.R_exc_milestone, p5.exc_milestone_bars) == (100.0, (2.5, 4.0, 5.5))
    assert p5.R_caps_max == 250.0
    assert p5.R_quality_max == 500.0 and p5.step_cost == 0.0 and p5.w_h == 0.0
    # untested/flat completion still dominates Phi_max (completion-first)
    assert p5.completion_quality_floor * p5.R_complete > 100.0


# -------------------- measured-caps bonus (getRideMeasurements; graceful degradation)
# The five wooden-RC rating caps are MEASURED quantities (test-run stats). With the new
# plugin endpoint the env pays a graded ramp on the real measurements; an old plugin
# (unknown endpoint) degrades to 0 bonus with everything else intact.

MEASUREMENTS_FIXTURE = {
    "excitement": 8.0, "intensity": 5.5, "nausea": 1.0,
    "maxSpeed": 30.0, "averageSpeed": 12.0, "rideTime": 40, "rideLength": 400.0,
    "maxPositiveVerticalGs": 2.5, "maxNegativeVerticalGs": -0.2, "maxLateralGs": 1.1,
    "totalAirTime": 1.2, "numDrops": 3, "highestDropHeight": 14.0,
}


class MeasuredAPI(CompletingAPI):
    measurement_calls = 0

    def get_ride_measurements(self):
        type(self).measurement_calls += 1
        return {"success": True, "payload": dict(MEASUREMENTS_FIXTURE)}


def test_caps_quality_ramp_math():
    env = _bare_env()
    full = {"highestDropHeight": 12, "numDrops": 2, "maxSpeed": 23,
            "maxNegativeVerticalGs": 0.10, "rideLength": 370}
    assert env._caps_quality(full) == pytest.approx(1.0)
    zero = {"highestDropHeight": 0, "numDrops": 0, "maxSpeed": 0,
            "maxNegativeVerticalGs": 0.5, "rideLength": 0}
    assert env._caps_quality(zero) == 0.0
    half = {"highestDropHeight": 6, "numDrops": 1, "maxSpeed": 11.5,
            "maxNegativeVerticalGs": 0.3, "rideLength": 185}
    assert env._caps_quality(half) == pytest.approx(0.5)
    assert env._caps_quality(None) == 0.0


def test_caps_bonus_paid_from_measurements(monkeypatch):
    monkeypatch.setattr(oe_mod, "APIController", MeasuredAPI)
    MeasuredAPI.measurement_calls = 0
    env = OpenRCT2Env(verbose=0)
    env.skip_ride_testing = False
    env.reward_params = replace(RewardParams(), R_caps_max=250.0, roundtrip_gain=0.0)
    env.reset()
    phi0, reward, terminated, _, info = _drive_to_terminal(env)
    assert terminated
    quality = env._quality_bonus(8.0, 5.5, 1.0, env.reward_params)
    # the fixture clears every cap ramp -> full 250
    assert reward == pytest.approx(env.reward_params.R_complete - phi0 + quality + 250.0)
    m = info['episode_metrics']
    assert m['caps_bonus'] == pytest.approx(250.0)
    assert m['meas_available'] == 1.0
    assert {'meas_num_drops', 'meas_highest_drop', 'meas_max_speed',
            'meas_ride_length', 'meas_air_time', 'meas_neg_g'}.issubset(m)
    assert info['ride_measurements'] == MEASUREMENTS_FIXTURE
    assert MeasuredAPI.measurement_calls == 1        # fetched once, on the tested terminal


def test_caps_bonus_degrades_without_endpoint(monkeypatch):
    """CompletingAPI models an old plugin (unknown endpoint): no bonus, no exception,
    everything else pays normally."""
    monkeypatch.setattr(oe_mod, "APIController", CompletingAPI)
    env = OpenRCT2Env(verbose=0)
    env.skip_ride_testing = False
    env.reward_params = replace(RewardParams(), R_caps_max=250.0, roundtrip_gain=0.0)
    env.reset()
    phi0, reward, terminated, _, info = _drive_to_terminal(env)
    assert terminated
    quality = env._quality_bonus(8.0, 5.5, 1.0, env.reward_params)
    assert reward == pytest.approx(env.reward_params.R_complete - phi0 + quality)
    assert info['episode_metrics']['caps_bonus'] == 0.0
    assert info['episode_metrics']['meas_available'] == 0.0


def test_measurements_not_fetched_when_untested(monkeypatch):
    monkeypatch.setattr(oe_mod, "APIController", MeasuredAPI)
    MeasuredAPI.measurement_calls = 0
    env = OpenRCT2Env(verbose=0)
    env.skip_ride_testing = True                     # untested -> no stats, no fetch
    env.reward_params = replace(RewardParams(), R_caps_max=250.0, roundtrip_gain=0.0)
    env.reset()
    phi0, reward, terminated, _, info = _drive_to_terminal(env)
    assert terminated
    assert MeasuredAPI.measurement_calls == 0
    assert reward == pytest.approx(env.reward_params.R_complete - phi0)


def test_exc_feature_potential_monotone_and_phase_gated():
    """The dense per-piece quality gradient: the feature quality rises with pieces the
    rating pays (banked turns, deeper single drops) and telescopes on removal; Phi's
    exc component is exactly w_exc_feat * quality (so weight-0 phases pay nothing).
    Compared as a Phi DIFFERENCE because Phi's energy term also reads the history."""
    rows = [(10, 14, 15), (9, 15, 17), (13, 17, 18),
            (12, 18, 17), (6, 17, 15), (14, 15, 14)]
    p5 = ImprovedPhasedCurriculumWrapper._phase_reward_params(5)
    assert p5.w_exc_feat == 6.0
    env = _bare_env(history=_env_hist(rows))
    q0 = env._exc_feature_quality(p5)
    env.track_builder.history.extend(_env_hist([(21, 14, 14)]))     # banked turn
    q_banked = env._exc_feature_quality(p5)
    assert q_banked > q0
    env.track_builder.history.pop()                                 # telescopes back
    assert env._exc_feature_quality(p5) == pytest.approx(q0)
    env.track_builder.history.extend(_env_hist([(6, 14, 12), (6, 12, 10)]))  # deeper drop
    assert env._exc_feature_quality(p5) > q0
    # Phi wiring: the exc component is exactly w * quality (0 when the weight is 0)
    delta = env._potential(p5) - env._potential(replace(p5, w_exc_feat=0.0))
    assert delta == pytest.approx(p5.w_exc_feat * env._exc_feature_quality(p5))


def test_p5_qualified_is_tested_excitement_diagnostic():
    """P5 'qualified' (diagnostics only; the length ladder still gates on raw cold
    success): a completed, TESTED ride rating E >= 4."""
    W = ImprovedPhasedCurriculumWrapper
    w = W.__new__(W)
    w.current_phase = 5
    good = SimpleNamespace(_last_test_ok=True, last_ride_excitement=4.5)
    assert w._is_qualified(good, True) is True
    assert w._is_qualified(good, False) is False
    low = SimpleNamespace(_last_test_ok=True, last_ride_excitement=3.0)
    assert w._is_qualified(low, True) is False
    untested = SimpleNamespace(_last_test_ok=False, last_ride_excitement=4.5)
    assert w._is_qualified(untested, True) is False


def _p5_payout(params, length, chain_peak, steep, excitement, intensity, nausea):
    """Terminal payout mirroring step()'s terminal branch under a quality-gated params:
    _calculate_reward + exc-gated remainder + viable + milestones + quality bonus."""
    env = _bare_env(history=_env_hist(_big_hill(chain_peak=chain_peak, steep=steep,
                                                length=length)))
    env.loop_completed = True
    env._phi_prev = 0.0
    env.reward_params = params
    env._calculate_energy_margin = lambda: 10.0
    env._last_test_ok = excitement > 0
    r = env._calculate_reward(True, 0)
    E = excitement if env._last_test_ok else 0.0
    if params.completion_quality_floor < 1.0 and params.exc_gate_target > 0:
        ramp = min(max(E, 0.0) / params.exc_gate_target, 1.0)
        r += (params.R_complete * env._last_gate_prequality
              * (1.0 - params.completion_quality_floor) * ramp)
    if env._last_test_ok and params.R_viable > 0.0:
        r += params.R_viable
    r += params.R_exc_milestone * sum(1 for b in params.exc_milestone_bars if E >= b)
    if env._last_test_ok:
        r += env._quality_bonus(E, intensity, nausea, params)
    return r


def test_p5_extending_to_caps_beats_mini_loop():
    """THE P5 regression: banking the 24-piece E=1.15 mini-loop now must lose decisively
    to building out a 40-piece caps-shaped loop that rates E~4."""
    P5 = ImprovedPhasedCurriculumWrapper._phase_reward_params(5)
    stay = _p5_payout(P5, length=24, chain_peak=19, steep=False,
                      excitement=1.15, intensity=1.33, nausea=0.79)
    extend = (P5.gamma ** 16) * _p5_payout(P5, length=40, chain_peak=25, steep=True,
                                           excitement=4.0, intensity=5.0, nausea=2.5)
    assert extend > stay + 200.0


# ------------------------------ static excitement-feature helpers (P5 substrate)
# The game's wooden-RC rating caps key on the HIGHEST SINGLE drop (>=12z), the number
# of drops (>=2), and turn variety. These helpers make those legs visible to struct
# credit and the excitement PBRS term, statically from the removal-safe history.

def test_max_single_drop_and_run_count():
    rows = [(10, 14, 15), (9, 15, 17), (9, 17, 19), (9, 19, 21), (9, 21, 23),
            (9, 23, 25), (13, 25, 26),                       # chain climb to +12
            (12, 26, 25), (27, 25, 21), (28, 21, 17),
            (6, 17, 15), (6, 15, 13), (14, 13, 12),          # one continuous 14z drop run
            (5, 12, 14),                                     # climb breaks the run
            (6, 14, 12),                                     # second drop run: 2z
            (0, 12, 12)]
    env = _bare_env(history=_env_hist(rows))
    assert env._max_single_drop_z() == pytest.approx(14.0)
    assert env._drop_run_count() == 2
    env.track_builder.history.pop()                          # flat tail: unchanged
    env.track_builder.history.pop()                          # second run gone
    assert env._drop_run_count() == 1                        # removal-safe recompute


def test_turn_and_banked_counters():
    rows = [(3, 14, 14), (21, 14, 14), (24, 14, 14), (29, 14, 14), (6, 14, 12)]
    env = _bare_env(history=_env_hist(rows))
    assert env._banked_turn_count() == 2                     # 21, 24
    assert env._turn_count() == 4                            # 3, 21, 24, 29; drop is not a turn


# ------------------------------------- P6 variety legs (the monoculture problem)
# Every build is the same rectangle because nothing ever paid for shape: exemplars,
# ratchet, and reward all select one motif. P6 grades turn count, S-bends, and
# HANDEDNESS BALANCE (rectangles are all-one-direction; balance forces winding).

def test_turn_balance_and_sbend_counters():
    # left family: 1,3,21,23,29 -- right family: 2,4,22,24,30
    rows = [(4, 14, 14), (4, 14, 14), (4, 14, 14), (4, 14, 14),   # 4 right (a rectangle)
            (3, 14, 14), (29, 14, 14), (30, 14, 14)]              # 1 left + S-pair (1L/1R)
    env = _bare_env(history=_env_hist(rows))
    assert env._sbend_count() == 2                           # 29, 30
    assert env._turn_balance_count() == 2                    # min(left=2, right=5)
    rect = _bare_env(history=_env_hist([(4, 14, 14)] * 4))
    assert rect._turn_balance_count() == 0                   # single-handed: no balance


def test_p6_variety_fields_default_inert():
    p = RewardParams()
    assert (p.struct_w_turns, p.struct_turns_target) == (0.0, 12.0)
    assert (p.struct_w_sbend, p.struct_sbend_target) == (0.0, 4.0)
    assert (p.struct_w_turn_balance, p.struct_turn_balance_target) == (0.0, 2.0)
    assert p.qualify_min_excitement == 0.0
    assert p.qualify_min_turns == 0.0
    assert p.qualify_min_turn_balance == 0.0


def test_hill_quality_pays_variety_components():
    params = replace(RewardParams(), R_struct_max=250.0, struct_w_chain=0.0,
                     struct_w_drop=0.0, struct_height_target=0.0,
                     struct_w_turns=0.5, struct_turns_target=12.0,
                     struct_w_sbend=0.25, struct_sbend_target=4.0,
                     struct_w_turn_balance=0.25, struct_turn_balance_target=2.0)
    rows = ([(4, 14, 14)] * 4 + [(3, 14, 14)] * 2                 # 6 turns, balance 2
            + [(29, 14, 14), (30, 14, 14)])                       # +2 S (also turns)
    env = _bare_env(history=_env_hist(rows))
    # turns 8/12, sbend 2/4, balance min(3,5)=3 -> capped 2/2
    expected = 0.5 * (8 / 12) + 0.25 * (2 / 4) + 0.25 * 1.0
    assert env._hill_quality(params) == pytest.approx(expected)


def test_qualify_variety_legs():
    params = replace(RewardParams(), R_qualify=200.0, struct_height_target=0.0,
                     struct_drop_target=0.0, struct_length_target=0.0,
                     qualify_min_excitement=4.5, qualify_min_turns=6.0,
                     qualify_min_turn_balance=2.0, qualify_requires_test=True)
    rows = [(4, 14, 14)] * 4 + [(3, 14, 14)] * 2
    env = _bare_env(history=_env_hist(rows))
    env._last_test_ok = True
    env.last_ride_excitement = 5.0
    assert env._qualifies(params) is True
    env.last_ride_excitement = 4.0                           # below the E floor
    assert env._qualifies(params) is False
    env.last_ride_excitement = 5.0
    env._last_test_ok = False                                # untested -> E not trusted
    assert env._qualifies(params) is False
    env._last_test_ok = True
    rect = _bare_env(history=_env_hist([(4, 14, 14)] * 8))   # turns ok, balance 0
    rect._last_test_ok = True
    rect.last_ride_excitement = 5.0
    assert rect._qualifies(params) is False


def test_p6_params_grade_variety():
    p6 = ImprovedPhasedCurriculumWrapper._phase_reward_params(6)
    assert (p6.struct_w_turns, p6.struct_turns_target) == (0.25, 12.0)
    assert (p6.struct_w_sbend, p6.struct_sbend_target) == (0.05, 4.0)
    assert (p6.struct_w_turn_balance, p6.struct_turn_balance_target) == (0.10, 2.0)
    assert (p6.struct_w_single_drop, p6.struct_w_drop_runs) == (0.20, 0.15)
    assert (p6.struct_w_length, p6.struct_w_banked) == (0.15, 0.10)
    total = (p6.struct_w_single_drop + p6.struct_w_drop_runs + p6.struct_w_length
             + p6.struct_w_banked + p6.struct_w_turns + p6.struct_w_sbend
             + p6.struct_w_turn_balance)
    assert total == pytest.approx(1.0)
    assert p6.qualify_min_excitement == 4.5
    assert (p6.qualify_min_turns, p6.qualify_min_turn_balance) == (12.0, 2.0)
    assert p6.qualify_requires_test is True and p6.R_qualify == 200.0
    # quality economics carried over from P5 unchanged
    assert (p6.completion_quality_floor, p6.exc_gate_target) == (0.4, 6.0)
    assert p6.R_caps_max == 250.0 and p6.R_viable == 150.0
    # P5 params are untouched by the P6 branch
    p5 = ImprovedPhasedCurriculumWrapper._phase_reward_params(5)
    assert p5.struct_w_turns == 0.0 and p5.qualify_min_excitement == 0.0


def test_p6_qualified_requires_variety_and_tested_excitement():
    W = ImprovedPhasedCurriculumWrapper
    w = W.__new__(W)
    w.current_phase = 6

    def base(actions, test_ok=True, exc=5.0):
        hist = [{"action": a, "position": [i, 0, 14], "next_position": [i + 1, 0, 14]}
                for i, a in enumerate(actions)]
        return SimpleNamespace(track_builder=SimpleNamespace(history=hist),
                               _last_test_ok=test_ok, last_ride_excitement=exc)

    winding = [4] * 6 + [3] * 4 + [29, 30]                   # 12 turns, balance 5
    assert w._is_qualified(base(winding), True) is True
    assert w._is_qualified(base(winding), False) is False    # must complete
    assert w._is_qualified(base(winding, exc=4.0), True) is False      # E floor
    assert w._is_qualified(base(winding, test_ok=False), True) is False
    rectangle = [4] * 12                                     # turns ok, single-handed
    assert w._is_qualified(base(rectangle), True) is False
    few_turns = [4] * 4 + [3] * 4                            # balanced but only 8 turns
    assert w._is_qualified(base(few_turns), True) is False


def test_p5_advances_to_p6_when_ladder_done_and_quality_holds(monkeypatch):
    monkeypatch.setattr(oe_mod, "APIController", FakeAPI)
    base = OpenRCT2Env(verbose=0)
    w = ImprovedPhasedCurriculumWrapper(base, verbose=0)
    w.current_phase = 5
    w._update_phase_settings()
    w._track_stats = True
    w.phase_episode_count = 60
    w.phase5_current_length = w.phase5_target_length          # ladder topped out
    w.episode_results.extend([True] * 50)
    w.episode_qualified_results.extend([True] * 20 + [False] * 30)   # 40% >= E4 cold
    assert w._check_phase_advancement() is True
    assert w.current_phase == 6
    assert base.max_track_length == w.phase6_max_length == 120
    assert base.skip_ride_testing is False
    # below the entry bar: no advancement
    w2 = ImprovedPhasedCurriculumWrapper(OpenRCT2Env(verbose=0), verbose=0)
    w2.current_phase = 5
    w2._track_stats = True
    w2.phase_episode_count = 60
    w2.phase5_current_length = w2.phase5_target_length
    w2.episode_results.extend([True] * 50)
    w2.episode_qualified_results.extend([True] * 10 + [False] * 40)  # 20%
    assert w2._check_phase_advancement() is False
    assert w2.current_phase == 5


# --------------------------------- P4 steep-drop credit (the last reward-invisible leg)
# 9h into the fixed P4 run: tests verified (0.74), height/drop/length legs green or
# ramping, but qualified_rate pinned at 0 -- the 60-degree steep-drop leg had NO gradient
# (zero steep pieces in the last 80 harvested loops; entropy at the collapse line).
# Fix: grade steepness into the structure credit like every other leg. Steepness is a
# piece-type swap, not extra pieces, so an additive ramp suffices (no discounting fight).

def test_steep_fields_default_inert():
    p = RewardParams()
    assert p.struct_w_steep == 0.0
    assert p.struct_steep_target == 8.0


def test_steep_drop_z_sums_steep_descents_only():
    """Only the 60-degree family (8/27/28) counts; 25-degree descents are excluded."""
    env = _bare_env(history=_env_hist([
        (10, 14, 15), (9, 15, 17),            # chain climb
        (27, 17, 13), (28, 13, 9),            # steep segment: drops 4 + 4
        (6, 9, 7), (0, 7, 7),                 # 25-deg drop (2) + flat
    ]))
    assert env._steep_drop_z() == pytest.approx(8.0)
    assert _bare_env(history=_env_hist([(6, 14, 12), (6, 12, 10)]))._steep_drop_z() == 0.0


def test_hill_quality_pays_graded_steep_credit():
    """The steep component ramps with steep-dropped z toward the target -- half a
    segment pays half the credit (no cliff), a full segment pays it all."""
    params = replace(RewardParams(), R_struct_max=250.0, struct_w_chain=0.0,
                     struct_w_drop=0.0, struct_height_target=0.0,
                     struct_w_steep=1.0, struct_steep_target=8.0)
    full = _bare_env(history=_env_hist([(27, 22, 18), (28, 18, 14)]))   # 8z steep
    half = _bare_env(history=_env_hist([(27, 18, 14)]))                 # 4z steep
    none = _bare_env(history=_env_hist([(6, 16, 14)]))                  # 25-deg only
    assert full._hill_quality(params) == pytest.approx(1.0)
    assert half._hill_quality(params) == pytest.approx(0.5)
    assert none._hill_quality(params) == pytest.approx(0.0)


def test_p4_params_grade_steepness():
    """P4 structure credit reweighted to carry the steep leg (weights still sum to 1.0);
    P3 keeps steepness out (its gate has no steep requirement)."""
    W = ImprovedPhasedCurriculumWrapper
    p4 = W._phase_reward_params(4)
    assert (p4.struct_w_steep, p4.struct_steep_target) == (0.2, 8.0)
    assert (p4.struct_w_height, p4.struct_w_drop, p4.struct_w_length) == (0.3, 0.3, 0.2)
    assert p4.struct_w_steep + p4.struct_w_height + p4.struct_w_drop + p4.struct_w_length \
        == pytest.approx(1.0)
    assert W._phase_reward_params(3).struct_w_steep == 0.0


def test_p4_steep_segment_is_reward_visible_before_qualifying():
    """Swapping a steep segment into an otherwise-identical verified P4 completion must
    raise the payout decisively (gate release + struct credit + qualify), NOT only via
    the R_qualify conjunction -- that was reward-invisible at low entropy."""
    P4 = ImprovedPhasedCurriculumWrapper._phase_reward_params(4)
    no_steep = _completion_payout(P4, length=40, chain_peak=25, steep=False, test_ok=True)
    with_steep = _completion_payout(P4, length=40, chain_peak=25, steep=True, test_ok=True)
    assert with_steep > no_steep + 300.0


def test_episode_metrics_expose_steep_drop_z(monkeypatch):
    monkeypatch.setattr(oe_mod, "APIController", CompletingAPI)
    env = OpenRCT2Env(verbose=0)
    env.skip_ride_testing = True
    env.reset()
    _, _, terminated, _, info = _drive_to_terminal(env)
    assert terminated
    assert 'steep_drop_z' in info['episode_metrics']


def test_ride_testing_enabled_from_phase4(monkeypatch):
    monkeypatch.setattr(oe_mod, "APIController", FakeAPI)
    base = OpenRCT2Env(verbose=0)
    wrapper = ImprovedPhasedCurriculumWrapper(base, verbose=0)
    for phase, expect_skip in ((1, True), (2, True), (3, True), (4, False), (5, False)):
        wrapper.current_phase = phase
        wrapper._update_phase_settings()
        assert base.skip_ride_testing is expect_skip, f"phase {phase}"


# --------------------------------------------- unrated-ride sentinel (live-probe finding)

class SentinelThenRatedAPI(CompletingAPI):
    """The live plugin returns excitement=-0.01 (RCT2's 'not yet rated' -1/100) until the
    test train has actually run: the poll must reject non-positive ratings and keep waiting."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.stat_calls = 0

    def get_ride_stats(self):
        self.stat_calls += 1
        if self.stat_calls < 3:
            return {"success": True, "payload": {"excitement": -0.01, "intensity": 0, "nausea": 0}}
        return super().get_ride_stats()


class NeverRatedAPI(CompletingAPI):
    def get_ride_stats(self):
        return {"success": True, "payload": {"excitement": -0.01, "intensity": 0, "nausea": 0}}


def test_poll_rejects_unrated_sentinel_then_accepts_real_stats(monkeypatch):
    monkeypatch.setattr(oe_mod, "APIController", SentinelThenRatedAPI)
    env = OpenRCT2Env(verbose=0)
    stats = env._poll_for_ride_stats(max_wait=2, poll_interval=0.01)
    assert stats['excitement'] == pytest.approx(8.0)      # waited past the sentinel
    assert env.api_controller.stat_calls >= 3


def test_unrated_ride_is_not_test_ok_and_earns_nothing(monkeypatch):
    """A test that never rates (train still running / stalled) must NOT count as verified,
    must NOT pay R_viable, and must NOT collect the nausea-term freebie through the quality
    bonus (nausea=0 on an UNRATED ride is not a calm ride)."""
    monkeypatch.setattr(oe_mod, "APIController", NeverRatedAPI)
    env = OpenRCT2Env(verbose=0)
    env.skip_ride_testing = False
    env.reward_params = RewardParams(R_viable=150.0, R_quality_max=500.0, roundtrip_gain=0.0)
    env.ride_test_max_wait = 0.05                          # keep the test fast
    env.reset()
    phi_prev_before, reward, terminated, _, info = _drive_to_terminal(env)
    assert terminated
    assert env._last_test_ok is False
    assert reward == pytest.approx(env.reward_params.R_complete - phi_prev_before)


def test_quality_bonus_gates_out_sentinel_stats():
    env = _bare_env()
    p = RewardParams(R_quality_max=500.0)
    assert env._quality_bonus(-0.01, 0.0, 0.0, p) == 0.0   # unrated sentinel -> no freebie
    assert env._quality_bonus(1.5, 1.0, 0.5, p) > 0.0      # real (low) ratings still score
