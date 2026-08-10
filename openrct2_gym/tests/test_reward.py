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
from openrct2_gym.envs.footprint import classify_family, switch_count, FAMILY_N
from openrct2_gym.envs.track_pieces import RIGHT_TURN_ACTIONS
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
    env.target_family = 0
    return env


# ----------------------------------------------------------------- RewardParams

def test_reward_params_defaults_support_completion_first():
    p = RewardParams()
    # 0.995, not 0.99 (Jul-15): the cold-opening trap, quantified from step ONE -- at
    # gamma 0.99 a 90-piece E5.4 build discounts to ~890 vs ~1130 for the 28-piece E2.3
    # quick loop, so the SHORT loop was genuinely optimal and the policy was right to
    # refuse the long opening. 0.995 flips it (~1395 vs ~909). Uniform across phases to
    # keep the PBRS gamma-tie exact (model gamma == reward gamma == this constant).
    assert p.gamma == 0.995
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
    # completion phases 1-4, off in phase 5), by the excitement-feature term (P5's own
    # dense gradient, since the Jul-9 quality redesign), and -- since task 5b armed the
    # family ramp in P3-5 (Aug-9) -- by the dense family-match potential (off in P1).
    p5 = W._phase_reward_params(5)
    route = W._phase_reward_params(1).w_route * env._route_progress()
    exc_feat = p5.w_exc_feat * env._exc_feature_quality(p5)
    # _potential's w_family term reads _family_phi_match (its own wide falloffs), not
    # _family_match -- see the phi-falloff-fix.
    family = p5.w_family * env._family_phi_match(p5)
    assert phi_p5 == pytest.approx(phi_p1 - route + exc_feat + family)


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
    w.episode_family_results = {z: deque(maxlen=50) for z in range(FAMILY_N)}
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
    # P5 draws the episode's seed from PHASE_FAMILIES at random and (Aug-9) pays R_family
    # on a hit, so pin it -- an unpinned draw makes this exact-arithmetic test flaky.
    wrapper._sample_target_family = lambda: 0
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
    # Jul-15: the P5 length gate is armed, so this tiny CompletingAPI loop pays only the
    # composed effective gate (hill x length x quality release) -- read it back from the
    # env's own accounting rather than assuming full R_complete.
    # Aug-9: P5 now also pays R_family, and this all-flat loop (0 turns, 0 switches) is a
    # hit on the phase-pinned oval seed -- so it belongs in the once-only accounting too.
    assert base_env.target_family == 0 and base_env._family_hit()
    assert base_env._last_family_bonus == p.R_family > 0.0
    assert reward == pytest.approx(
        p.R_complete * base_env._last_completion_gate - phi_prev_before + quality
        + p.step_cost + p.R_viable + 3 * p.R_exc_milestone + base_env._last_struct_bonus
        + p.R_family)
    assert base_env._last_completion_gate < 1.0          # the length gate actually bit


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
    assert p.qualify_max_sbend == float('inf')   # no density cap unless a phase sets one


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


def test_exc_feature_quality_pays_turn_balance_densely_when_the_target_is_armed():
    """Jul-22 deadlock fix: cold builds never sample the winding opening because the jog
    only pays after a risky completion, so turn BALANCE joined the dense Phi features.
    The leg is still there for any params that arm struct_turn_balance_target -- but no
    live phase does any more (Aug-9 final review: on a 0-switch seed it is unreachable AND
    it cancelled the family potential; see test_p5_p6_turn_balance_target_is_retired)."""
    armed = replace(ImprovedPhasedCurriculumWrapper._phase_reward_params(6),
                    struct_turn_balance_target=2.0)
    # SAME turn count (6 pieces), different handedness mix: only balance distinguishes
    rect6 = _bare_env(history=_env_hist([(4, 14, 14)] * 6))
    jogged = _bare_env(history=_env_hist([(4, 14, 14)] * 4 + [(3, 14, 14)] * 2))
    assert (jogged._exc_feature_quality(armed)
            - rect6._exc_feature_quality(armed)) == pytest.approx(1 / 6, abs=0.01)


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
    # Aug-6: S-bends are NOT turns (no heading change) -- 3, 21, 24 only.
    assert env._turn_count() == 3
    assert env._sbend_count() == 1                           # 29 counted on its own leg


# ------------------------------------- P6 variety legs (the monoculture problem)
# Every build is the same rectangle because nothing ever paid for shape: exemplars,
# ratchet, and reward all select one motif. P6 grades turn count, S-bends, and
# HANDEDNESS BALANCE (rectangles are all-one-direction; balance forces winding).

def test_turn_balance_and_sbend_counters():
    # heading turns only: left 1,3,21,23 -- right 2,4,22,24 (Aug-6: S-bends excluded,
    # an alternating S-stack used to manufacture balance out of a net-zero weave)
    rows = [(4, 14, 14), (4, 14, 14), (4, 14, 14), (4, 14, 14),   # 4 right (a rectangle)
            (3, 14, 14), (29, 14, 14), (30, 14, 14)]              # 1 left + S-pair
    env = _bare_env(history=_env_hist(rows))
    assert env._sbend_count() == 2                           # 29, 30
    assert env._turn_balance_count() == 1                    # min(left=1, right=4)
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
            + [(29, 14, 14), (30, 14, 14)])                       # +2 S (own leg only)
    env = _bare_env(history=_env_hist(rows))
    # turns 6/12 (S-bends excluded), sbend 2/4, balance min(2,4)=2 -> capped 2/2
    expected = 0.5 * (6 / 12) + 0.25 * (2 / 4) + 0.25 * 1.0
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


def test_p5_long_build_beats_quick_loop_from_step_one():
    """THE missing economics test: earlier stay-vs-extend regressions measured MID-build
    states; the cold policy chooses at step ONE, where 60+ extra steps of discounting
    crushed the long build at gamma 0.99 (live-probed Jul-15: deterministic cold builds
    turn at piece 1 into a 28-piece banked loop). From a bare station, the discounted
    value of the long exemplar build must now decisively beat the quick loop."""
    P5 = ImprovedPhasedCurriculumWrapper._phase_reward_params(5)
    quick = (P5.gamma ** 27) * _p5_payout(P5, length=28, chain_peak=27, steep=True,
                                          excitement=2.34, intensity=2.6, nausea=1.4)
    long_build = (P5.gamma ** 89) * _p5_payout(P5, length=90, chain_peak=27, steep=True,
                                               excitement=5.4, intensity=6.2, nausea=2.2)
    assert long_build > quick * 1.15


def test_p6_params_grade_variety():
    """Aug-9: shape is no longer a struct leg -- the SEED names the target footprint, so
    the fixed turns/balance ramps are off and their weight went to the cap-aligned legs.
    Variety now lives in the multiplicative family gate (see the seed-conditioned block)."""
    p6 = ImprovedPhasedCurriculumWrapper._phase_reward_params(6)
    assert (p6.struct_w_turns, p6.struct_turns_target) == (0.0, 12.0)
    assert (p6.struct_w_sbend, p6.struct_sbend_target) == (0.05, 4.0)
    # target retired alongside the weight (Aug-9 final review): _exc_feature_quality reads
    # the TARGET directly, so a nonzero one kept the leg live inside the dense potential
    # -- see test_p5_p6_turn_balance_target_is_retired.
    assert (p6.struct_w_turn_balance, p6.struct_turn_balance_target) == (0.0, 0.0)
    assert (p6.struct_w_single_drop, p6.struct_w_drop_runs) == (0.35, 0.15)
    assert (p6.struct_w_length, p6.struct_w_banked) == (0.30, 0.15)
    total = (p6.struct_w_single_drop + p6.struct_w_drop_runs + p6.struct_w_length
             + p6.struct_w_banked + p6.struct_w_turns + p6.struct_w_sbend
             + p6.struct_w_turn_balance)
    assert total == pytest.approx(1.0)
    assert p6.qualify_min_excitement == 4.5
    assert (p6.qualify_min_turns, p6.qualify_min_turn_balance) == (0.0, 0.0)
    assert p6.qualify_requires_family is True
    # the shape guard the retired turns>=12 leg used to provide by accident: the struct
    # S-bend leg saturates at 4, so 6 leaves honest headroom and blocks the observed farm
    assert p6.qualify_max_sbend == 6.0
    assert p6.qualify_requires_test is True and p6.R_qualify == 200.0
    # quality economics carried over from P5 unchanged
    assert (p6.completion_quality_floor, p6.exc_gate_target) == (0.4, 6.0)
    assert p6.R_caps_max == 250.0 and p6.R_viable == 150.0
    # P5 params are untouched by the P6 branch
    p5 = ImprovedPhasedCurriculumWrapper._phase_reward_params(5)
    assert p5.struct_w_turns == 0.0 and p5.qualify_min_excitement == 0.0


def test_p6_qualified_requires_seed_family_and_tested_excitement():
    """Aug-9: the curriculum's P6 predicate mirrors env._qualifies -- completed, tested
    at the E floor, and shaped like the family the SEED asked for. The old fixed
    turns>=12 / balance>=2 legs are gone: they contradict an oval/spiral/out-and-back
    seed outright (3 of the 5 families could never have qualified)."""
    W = ImprovedPhasedCurriculumWrapper
    w = W.__new__(W)
    w.current_phase = 6

    def base(actions, test_ok=True, exc=5.0, family=0):
        hist = [{"action": a, "position": [i, 0, 14], "next_position": [i + 1, 0, 14]}
                for i, a in enumerate(actions)]
        return SimpleNamespace(track_builder=SimpleNamespace(history=hist),
                               _last_test_ok=test_ok, last_ride_excitement=exc,
                               target_family=family)

    winding = [4, 4, 3, 3] * 3                  # 12 heading turns, 5 switches -> winding
    assert w._is_qualified(base(winding, family=3), True) is True
    assert w._is_qualified(base(winding, family=3), False) is False    # must complete
    assert w._is_qualified(base(winding, exc=4.0, family=3), True) is False        # E floor
    assert w._is_qualified(base(winding, test_ok=False, family=3), True) is False
    assert w._is_qualified(base(winding, family=0), True) is False     # oval was asked for
    rectangle = [4] * 4                         # 4 turns, no alternation -> oval
    assert w._is_qualified(base(rectangle, family=0), True) is True    # oval seed: correct
    assert w._is_qualified(base(rectangle, family=3), True) is False   # winding seed: wrong
    # 12 SAME-handed turns: the count lands in the winding band but the alternation count
    # (0) lands in no band at all, so classify_family returns None. A build that matches
    # nothing is a hit for no seed -- not a near-miss that qualifies for the closest one.
    same_handed = [4] * 12
    assert classify_family(same_handed) is None
    assert w._is_qualified(base(same_handed, family=3), True) is False
    assert w._is_qualified(base(same_handed, family=0), True) is False
    # Aug-6 exploit: two 180s + a stack of S-bends farmed turns AND balance. S-bends hand
    # back the heading, so they can no more manufacture a winding FOOTPRINT than they
    # could manufacture the old turn/balance legs -- and under the OVAL seed they
    # geometrically match, so the density leg is what blocks them there (mirrors
    # env._qualifies' qualify_max_sbend leg).
    sbend_farm = [4] * 4 + [29, 30] * 5                      # 4 turns + 10 S-bends
    assert w._is_qualified(base(sbend_farm, family=3), True) is False
    assert w._is_qualified(base(sbend_farm, family=0), True) is False


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


# ----------------------------------- P6 style gate (the winding-frequency war, Jul-24)
# With every style term ADDITIVE (R_qualify 200 + variety struct legs ~100), the live run
# kept completing 4-turn drop-rectangles: k_max annealed 59->83 while balance density
# thinned 11/20 -> 5/20 -- the annealer promotes on COMPLETION, so only the reward can
# make the cold policy CHOOSE winding, and a ~+16% payout edge demonstrably lost the
# frequency war to the plain shape's reliability. Fix: the thrice-proven multiplicative
# floor, now on shape -- a P6 completion is style-gated by a composite turns+balance
# ramp mirroring the qualified gate's two variety legs.

def test_style_floor_defaults_inert():
    p = RewardParams()
    assert p.completion_style_floor == 1.0
    for phase in (1, 2, 3, 4, 5):
        assert ImprovedPhasedCurriculumWrapper._phase_reward_params(
            phase).completion_style_floor == 1.0


def _styled_loop(n_right=0, n_left=0, pad_to=12):
    """Flat closed-loop rows with the given turn mix (right=4, left=3), padded straight."""
    rows = [(4, 14, 14)] * n_right + [(3, 14, 14)] * n_left
    rows += [(0, 14, 14)] * max(pad_to - len(rows), 0)
    return rows


def test_completion_style_gate_scales_r_complete():
    """A completed loop below the variety targets earns only the style-gated fraction of
    R_complete; the composite ramp averages the turns and balance legs so a single-handed
    zigzag (turns maxed, balance zero) still leaves half the remainder on the table."""
    params = replace(RewardParams(), completion_style_floor=0.6,
                     struct_turns_target=12.0, struct_turn_balance_target=2.0)

    def payout(rows):
        env = _bare_env(history=_env_hist(rows))
        env.loop_completed = True
        env._phi_prev = 0.0
        env.reward_params = params
        return env._calculate_reward(True, 0)

    assert payout(_styled_loop()) == pytest.approx(0.6 * params.R_complete)
    gate_rect = 0.6 + 0.4 * (0.5 * (4 / 12) + 0.5 * 0.0)          # 4 same-hand turns
    assert payout(_styled_loop(n_right=4)) == pytest.approx(params.R_complete * gate_rect)
    gate_zigzag = 0.6 + 0.4 * (0.5 * 1.0 + 0.5 * 0.0)             # 12 turns, one-handed
    assert payout(_styled_loop(n_right=12)) == pytest.approx(params.R_complete * gate_zigzag)
    assert payout(_styled_loop(n_right=10, n_left=2)) == pytest.approx(params.R_complete)


def test_style_gate_scales_quality_remainder():
    """The style factor must fold into _last_gate_prequality so the post-test excitement
    remainder scales with it too (style and quality form ONE multiplicative gate)."""
    params = replace(RewardParams(), completion_style_floor=0.6,
                     completion_quality_floor=0.4,
                     struct_turns_target=12.0, struct_turn_balance_target=2.0)
    env = _bare_env(history=_env_hist(_styled_loop(n_right=4)))
    env.loop_completed = True
    env._phi_prev = 0.0
    env.reward_params = params
    style = 0.6 + 0.4 * (0.5 * (4 / 12))
    assert env._calculate_reward(True, 0) == pytest.approx(
        params.R_complete * style * 0.4)
    assert env._last_gate_prequality == pytest.approx(style)


def test_validate_completion_first_folds_style_floor():
    W = ImprovedPhasedCurriculumWrapper
    bad = replace(RewardParams(), completion_hill_floor=0.5,
                  completion_quality_floor=0.4, completion_style_floor=0.4,
                  R_roundtrip=100.0)
    with pytest.raises(AssertionError):
        W._validate_completion_first(bad, "test")      # 100 >= .5*.4*.4*1000 = 80
    ok = replace(bad, completion_style_floor=1.0)      # 100 < .5*.4*1000 = 200
    W._validate_completion_first(ok, "test")


def test_p6_retires_the_style_gate_for_the_family_gate():
    """Aug-9: the style gate's fixed turns+balance ramp is superseded by the SEED-
    conditioned family gate -- same multiplicative shape, but the target now varies per
    episode instead of always meaning 'wind more'. The style machinery stays in the env
    (default-inert) so earlier phases and its own tests are untouched."""
    p6 = ImprovedPhasedCurriculumWrapper._phase_reward_params(6)
    assert p6.completion_style_floor == 1.0
    assert p6.completion_family_floor == 0.5


def _p6_payout(params, rows, excitement, intensity=6.0, nausea=2.5, family=0):
    """Terminal payout mirroring step()'s P6 terminal branch: _calculate_reward +
    exc-gated remainder + viable + milestones + quality bonus + qualify + family."""
    env = _bare_env(history=_env_hist(rows))
    env.loop_completed = True
    env._phi_prev = 0.0
    env.reward_params = params
    env.target_family = family
    env._calculate_energy_margin = lambda: 10.0
    env._last_test_ok = excitement > 0
    env.last_ride_excitement = excitement
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
    if params.R_qualify > 0.0 and env._qualifies(params):
        r += params.R_qualify
    if (params.R_family > 0.0 and env._family_hit() and env._last_test_ok
            and E >= params.qualify_min_excitement):
        r += params.R_family
    return r


def test_p6_winding_beats_rectangle_from_step_one_on_a_winding_seed():
    """THE Jul-24 regression, now seed-conditioned (Aug-9). WHEN THE SEED ASKS FOR
    WINDING, the discounted value of the winding build must still beat the 4-turn
    drop-rectangle DECISIVELY (>= 1.25x) at step one -- the extra 10 pieces of
    gamma-discount and the shape-blind quality payments (milestones/viable/quality) both
    dilute the gate, and pre-fix the additive carrots left winding only a ~1.03x edge.
    The oval seed's mirror image lives in test_oval_seed_beats_winding_build_from_step_one:
    the SAME machinery must reverse the ordering when an oval is asked for."""
    P6 = ImprovedPhasedCurriculumWrapper._phase_reward_params(6)
    rect = (P6.gamma ** 39) * _p6_payout(
        P6, _p6_mix(n_right=4, n_left=0, chain_peak=27, length=40),
        excitement=5.6, family=3)
    winder = (P6.gamma ** 49) * _p6_payout(
        P6, _p6_mix(fill=[4, 4, 3, 3] * 3, chain_peak=27, length=50),
        excitement=5.6, family=3)
    assert winder > rect * 1.25


def test_style_gate_reported_in_episode_metrics(monkeypatch):
    """House rule: every new reward gate streams its own diagnostic. The style factor
    actually applied at close must appear in episode_metrics regardless of shape."""
    monkeypatch.setattr(oe_mod, "APIController", CompletingAPI)
    env = OpenRCT2Env(verbose=0)
    env.reward_params = replace(RewardParams(), completion_style_floor=0.6,
                                struct_turns_target=12.0,
                                struct_turn_balance_target=2.0, roundtrip_gain=0.0)
    env.reset()
    _, _, terminated, _, info = _drive_to_terminal(env)
    assert terminated
    frac = 0.5 * (min(env._turn_count() / 12.0, 1.0)
                  + min(env._turn_balance_count() / 2.0, 1.0))
    assert info['episode_metrics']['style_gate'] == pytest.approx(0.6 + 0.4 * frac)


def test_p6_arms_route_potential():
    """Jul-27 bundle: wound layouts fail closure precisely where rectangles never do --
    on the RETURN ROUTE. w_route (angular progress around the station, monotone along
    both detours, PBRS-clean) taught that navigation in P1-4 and was retired in P5 when
    the memorized rectangle stopped needing it; novel winding shapes still do."""
    assert ImprovedPhasedCurriculumWrapper._phase_reward_params(6).w_route == 3.0
    assert ImprovedPhasedCurriculumWrapper._phase_reward_params(5).w_route == 0.0


def test_ride_stats_poll_interval_matches_fast_ratings():
    """Aug-1 straggler fix: at 5,000 ticks/s ratings land in <1s, so the 0.5s poll
    granularity (tuned for the 4-5s laptop era) had become a large share of the
    terminal step -- and under SubprocVecEnv every other worker idles at the barrier
    for exactly that overhang. Poll at 0.1s; the max_wait timeout guard is unchanged."""
    import inspect
    sig = inspect.signature(OpenRCT2Env._poll_for_ride_stats)
    assert sig.parameters["poll_interval"].default == 0.1


# ------------------------------- S-bend turn-farming (Aug-6, live-observed exploit)
# Markus watched the checkpoint build: 180-turn, hill, 180-turn back, then SIX-TO-EIGHT
# stacked S-bends drifting diagonally home. Library audit confirmed the farm: in the 40
# newest COLD harvests, 61% of counted "turns" were S-bends (10.4 counted -> 4.1 real
# heading changes; longest run 8). S-bends leaked into BOTH variety legs -- _turn_count
# counted them as "direction changes" while _sbend_count's own docstring calls them
# "lateral weave WITHOUT a heading change", and in the balance leg 29 scored left / 30
# scored right, so an alternating stack manufactured handedness too. Fix: the style legs
# count HEADING changes only; S-bends keep their own leg (weight .05, capped at 4), so
# stacking past 4 earns exactly nothing.

def _p6_mix(n_right=0, n_left=0, n_sbend_pairs=0, chain_peak=27, length=80, fill=None):
    """A P6-scale build (chain hill + steep drop + length) whose trailing flat pads are
    converted into the requested turn / S-bend mix, so tests vary ONLY the style legs.
    `fill` overrides the mix with an explicit action list (for alternating footprints,
    where the ORDER of the turns -- not just the counts -- is what the family reads)."""
    rows = _big_hill(chain_peak=chain_peak, steep=True, length=length)
    if fill is not None:
        fill = list(fill)
    else:
        fill = [4] * n_right + [3] * n_left
        for _ in range(n_sbend_pairs):
            fill += [29, 30]
    for i in range(len(rows) - 1, -1, -1):
        if not fill:
            break
        if rows[i] == (0, 14, 14):
            rows[i] = (fill.pop(), 14, 14)
    assert not fill, "not enough flat pads to convert"
    return rows


def test_turn_count_excludes_sbends():
    env = _bare_env(history=_env_hist(
        [(4, 14, 14), (3, 14, 14), (21, 14, 14), (24, 14, 14)]        # 4 heading turns
        + [(29, 14, 14), (30, 14, 14)] * 4))                          # 8 S-bends
    assert env._turn_count() == 4
    assert env._sbend_count() == 8                                    # own leg unchanged


def test_turn_balance_excludes_sbends():
    """An alternating S-bend stack is net-zero heading change: it must score NO
    handedness, or the balance leg (the leg that exists to force genuine winding)
    is farmable by the flattest possible motif."""
    stack = _bare_env(history=_env_hist([(29, 14, 14), (30, 14, 14)] * 4))
    assert stack._turn_balance_count() == 0
    genuine = _bare_env(history=_env_hist([(4, 14, 14)] * 3 + [(3, 14, 14)] * 3))
    assert genuine._turn_balance_count() == 3


def test_sbend_stack_cannot_farm_p6_shape_credit():
    """THE regression, re-aimed at where shape credit now lives (Aug-9: the family gate,
    not a struct leg). Asked for a winding footprint, the observed exploit build must
    score strictly below a genuinely winding one -- S-bends hand back the heading, so
    they move neither the turn count nor the alternation count the family reads. Piling
    on more S-bends must still add nothing to struct once its own leg caps."""
    P6 = ImprovedPhasedCurriculumWrapper._phase_reward_params(6)

    def shape_credit(rows, family=3):
        env = _bare_env(history=_env_hist(rows))
        env.target_family = family
        return env._family_match(P6)

    exploit = [(4, 14, 14)] * 4 + [(29, 14, 14), (30, 14, 14)] * 4     # 4 turns + 8 S-bends
    genuine = [(4, 14, 14), (4, 14, 14), (3, 14, 14), (3, 14, 14)] * 3  # 12 turns, 5 switches
    assert shape_credit(exploit) < shape_credit(genuine)
    assert shape_credit(genuine) == pytest.approx(1.0)   # exactly the requested footprint
    # ... and the S-bends buy nothing back by being stacked higher. Measured against a
    # base whose match is strictly BETWEEN 0 and 1 (10 turns / 2 switches: the turn band
    # is met, the alternation band is one short), because the exploit itself scores a flat
    # 0.0 here and "0.0 == 0.0" cannot fail for the reason this assertion claims.
    partial = [(4, 14, 14)] * 5 + [(3, 14, 14)] * 3 + [(4, 14, 14)] * 2
    assert 0.0 < shape_credit(partial) < 1.0
    assert shape_credit(partial + [(29, 14, 14), (30, 14, 14)] * 3) == pytest.approx(
        shape_credit(partial))
    assert shape_credit(exploit + [(29, 14, 14), (30, 14, 14)] * 3) == pytest.approx(
        shape_credit(exploit))

    def struct_credit(rows):
        return _bare_env(history=_env_hist(rows))._hill_quality(P6)

    # Marginal S-bends past the capped target earn nothing the same number of PLAIN
    # pieces would not (length credit is the only thing extra track buys).
    pad_sbends = exploit + [(29, 14, 14), (30, 14, 14)] * 3
    pad_straights = exploit + [(0, 14, 14)] * 6
    assert struct_credit(pad_sbends) == pytest.approx(struct_credit(pad_straights))


def test_sbend_stack_does_not_meet_p6_qualified_gate():
    """THE Aug-6 exploit, guarded on BOTH legs -- an oval is a fine answer to an oval
    seed, an oval stuffed with S-bend padding is not.

    footprint.py deliberately ignores S-bends (they hand back the heading), so
    "oval + 8 S-bends" classifies as a pristine oval. That is right for CLASSIFICATION
    and precisely why the family leg alone cannot police padding: under an oval seed the
    padded build would clear the gate and collect R_qualify + R_family on top of the
    length credit its 8 padding pieces buy. The density leg (qualify_max_sbend) is what
    the retired turns>=12 leg used to do by accident."""
    P6 = ImprovedPhasedCurriculumWrapper._phase_reward_params(6)

    def env_for(rows, family=3):
        env = _bare_env(history=_env_hist(rows))
        env.target_family = family
        env._last_test_ok = True
        env.last_ride_excitement = 6.0
        env._calculate_energy_margin = lambda: 10.0
        return env

    # full-size builds so only the FOOTPRINT differs (P6 also gates height/drop/length)
    exploit = _p6_mix(n_right=4, n_left=0, n_sbend_pairs=4)   # 4 turns + 8 S-bends
    clean = _p6_mix(n_right=4, n_left=0)                      # the same oval, unpadded
    genuine = _p6_mix(fill=[4, 4, 3, 3] * 3)
    # winding seed: no amount of S-bends makes the rectangle a winder
    assert env_for(exploit)._qualifies(P6) is False
    assert env_for(genuine)._qualifies(P6) is True
    # oval seed: the honest oval qualifies, the padded one is still blocked
    assert env_for(clean, family=0)._qualifies(P6) is True
    assert env_for(exploit, family=0)._qualifies(P6) is False


def test_energy_model_still_charges_sbend_friction():
    """S-bends are curved track: the PHYSICS estimate must keep charging them turn
    friction even though the STYLE legs no longer count them as heading changes."""
    climb = [(10, 14, 15)] + [(9, 15 + 2 * i, 17 + 2 * i) for i in range(8)]  # banked energy
    straight = _bare_env(
        history=_env_hist(climb + [(0, 31, 31)] * 4))._calculate_estimated_energy()
    sbends = _bare_env(history=_env_hist(
        climb + [(29, 31, 31), (30, 31, 31)] * 2))._calculate_estimated_energy()
    assert straight > 0 and sbends < straight


# ------------------------------- seed-conditioned family reward (Aug-9 redesign)
# The target now comes from the seed, so "more turns is always better" is WRONG for
# oval seeds. The both-directions economics test below is the guard whose absence
# allowed the S-bend exploit to survive for a week.

def _family_env(rows, family, params, excitement=6.0):
    env = _bare_env(history=_env_hist(rows))
    env.loop_completed = True
    env._phi_prev = 0.0
    env.reward_params = params
    env.target_family = family
    env._last_test_ok = True
    env.last_ride_excitement = excitement
    env._calculate_energy_margin = lambda: 10.0
    return env


def test_family_reward_fields_default_inert():
    p = RewardParams()
    assert p.completion_family_floor == 1.0
    assert p.R_family == 0.0
    assert p.qualify_requires_family is False


def test_family_gate_scales_completion_by_match():
    params = replace(RewardParams(), completion_family_floor=0.5, roundtrip_gain=0.0)
    oval_rows = [(4, 14, 14)] * 4 + [(0, 14, 14)] * 8
    hit = _family_env(oval_rows, 0, params)._calculate_reward(True, 0)
    miss = _family_env(oval_rows, 3, params)._calculate_reward(True, 0)   # asked winding
    assert hit > miss
    assert hit == pytest.approx(params.R_complete)          # perfect match, full pay


def test_family_bonus_requires_hit_and_quality():
    params = replace(RewardParams(), R_family=200.0, qualify_requires_family=True,
                     qualify_min_excitement=4.5, qualify_requires_test=True,
                     struct_height_target=0.0, struct_drop_target=0.0,
                     struct_length_target=0.0)
    oval = [(4, 14, 14)] * 4 + [(0, 14, 14)] * 8
    assert _family_env(oval, 0, params)._qualifies(params) is True
    assert _family_env(oval, 3, params)._qualifies(params) is False          # wrong family
    assert _family_env(oval, 0, params, excitement=3.0)._qualifies(params) is False


def test_oval_seed_beats_winding_build_from_step_one():
    """THE inversion test. With an oval requested, building winding must LOSE.
    Without this, 'always add more turns' creeps back in.

    The winder is `fill=[4,4,3,3]*3` (12 turns, 5 switches -> genuinely `winding`), not
    six rights followed by six lefts: the latter alternates ONCE, classifies to no family
    at all, and scored only a partial 0.667 even under the seed it was named for."""
    P6 = ImprovedPhasedCurriculumWrapper._phase_reward_params(6)
    oval_rows = _p6_mix(n_right=4, n_left=0)
    wind_rows = _p6_mix(fill=[4, 4, 3, 3] * 3)
    oval_payout = _family_env(oval_rows, 0, P6)._calculate_reward(True, 0)
    wind_payout = _family_env(wind_rows, 0, P6)._calculate_reward(True, 0)
    assert oval_payout > wind_payout * 1.2


def test_winding_seed_beats_oval_build_from_step_one():
    """The mirror of the inversion test: same pair of builds, opposite seed."""
    P6 = ImprovedPhasedCurriculumWrapper._phase_reward_params(6)
    oval_rows = _p6_mix(n_right=4, n_left=0)
    wind_rows = _p6_mix(fill=[4, 4, 3, 3] * 3)
    assert (_family_env(wind_rows, 3, P6)._calculate_reward(True, 0)
            > _family_env(oval_rows, 3, P6)._calculate_reward(True, 0) * 1.2)


def test_p6_params_switch_off_the_fixed_turn_target_and_balance_leg():
    """The target comes from the seed now; the old always-more-turns legs must be off
    or they fight it on oval seeds."""
    p6 = ImprovedPhasedCurriculumWrapper._phase_reward_params(6)
    assert p6.struct_w_turns == 0.0
    assert p6.struct_w_turn_balance == 0.0
    assert p6.completion_style_floor == 1.0
    assert p6.completion_family_floor == 0.5
    assert p6.R_family == 200.0
    assert p6.qualify_requires_family is True
    # the fixed variety legs are gone from the qualified predicate too -- with them on,
    # an oval/spiral/out-and-back seed could never qualify (turns>=12 contradicts the
    # seed), so R_qualify would be dead for 3 of the 5 families.
    assert (p6.qualify_min_turns, p6.qualify_min_turn_balance) == (0.0, 0.0)
    # the freed struct weight is redistributed; the remaining legs still sum to 1.0
    total = (p6.struct_w_single_drop + p6.struct_w_drop_runs + p6.struct_w_length
             + p6.struct_w_banked + p6.struct_w_turns + p6.struct_w_sbend
             + p6.struct_w_turn_balance)
    assert total == pytest.approx(1.0)


def test_validate_completion_first_folds_the_family_floor():
    W = ImprovedPhasedCurriculumWrapper
    bad = replace(RewardParams(), completion_hill_floor=0.5,
                  completion_quality_floor=0.4, completion_family_floor=0.4,
                  R_roundtrip=100.0)
    with pytest.raises(AssertionError):
        W._validate_completion_first(bad, "test")      # 100 >= .5*.4*.4*1000 = 80
    W._validate_completion_first(replace(bad, completion_family_floor=1.0), "test")


def test_family_diagnostics_in_episode_metrics(monkeypatch):
    """House rule: every new reward gate streams its own diagnostic."""
    monkeypatch.setattr(oe_mod, "APIController", CompletingAPI)
    env = OpenRCT2Env(verbose=0)
    env.reward_params = replace(RewardParams(), completion_family_floor=0.5,
                                roundtrip_gain=0.0)
    env.reset()
    env.target_family = 0
    _, _, terminated, _, info = _drive_to_terminal(env)
    assert terminated
    m = info['episode_metrics']
    assert m['family_match'] == pytest.approx(env._family_match(env.reward_params))
    assert m['family_gate'] == pytest.approx(0.5 + 0.5 * m['family_match'])
    assert m['family_hit'] == float(env._family_hit())
    assert m['target_family'] == 0.0
    assert 'switch_count' in m


def test_family_gate_resets_between_episodes(monkeypatch):
    """Gate diagnostics must reset with the other gate state, or a truncated episode
    reports the previous episode's family gate."""
    monkeypatch.setattr(oe_mod, "APIController", CompletingAPI)
    env = OpenRCT2Env(verbose=0)
    env.reward_params = replace(RewardParams(), completion_family_floor=0.5,
                                roundtrip_gain=0.0)
    env.reset()
    _drive_to_terminal(env)
    assert env._last_family_gate > 0.0
    env.reset()
    assert env._last_family_gate == 0.0 and env._last_family_match == 0.0


# ---------------------------------------------------------------- dense family potential
# Terminal-only shaping (the family completion gate + qualify bonus above) has repeatedly
# been too slow to discover in this project (the style gate ran ~900k steps without
# reaching cold builds). w_family adds a THIRD consumer of _family_match: a dense
# per-piece PBRS potential, so family progress pays every step instead of only at close.

def test_family_potential_defaults_off_and_leaves_phase1_2_identical():
    """Phases 1-2 pin the seed to family 0 (PHASE_FAMILIES) with no reward reading it, so
    the potential must stay off there. Phases 3-5 arm it on a ramp -- see
    test_family_ramp_phases_3_4_5_match_the_table (task 5b, Aug-9)."""
    assert RewardParams().w_family == 0.0
    for phase in (1, 2):
        assert ImprovedPhasedCurriculumWrapper._phase_reward_params(phase).w_family == 0.0


def test_family_potential_rises_toward_the_requested_family():
    """Terminal-only shaping has been consistently too slow here (the style gate ran
    ~900k steps without reaching cold builds), so family progress pays per piece."""
    params = replace(RewardParams(), w_family=6.0)
    near = _bare_env(history=_env_hist([(4, 14, 14)] * 4))       # 4 turns: oval band
    far = _bare_env(history=_env_hist([(4, 14, 14)] * 12))       # 12 turns: family_match=0.5 (turn band 0, switch band 1)
    near.target_family = 0
    far.target_family = 0
    assert near._potential(params) > far._potential(params)


def test_family_potential_telescopes_as_a_state_function():
    """PBRS requires Phi to depend only on state, so place-then-remove must not pay.

    The appended piece must actually move the match, or this test would pass for an
    implementation that ignores the family entirely. The history is 3 RIGHT turns
    (action 4) -- already a full oval match (3 turns in [0,5], 0 switches in [0,0], so
    family_match == 1.0). Appending another RIGHT turn keeps turns in-band and switches
    at 0, so Phi would not move. Appending a LEFT turn (action 1) instead introduces one
    direction switch (R,R,R,L), pushing switches out of the oval's [0,0] band and
    dropping family_match to 0.5*(1.0 + (1 - 1/3)) = 0.8333 -- a real, verifiable change.
    """
    params = replace(RewardParams(), w_family=6.0)
    env = _bare_env(history=_env_hist([(4, 14, 14)] * 3))
    env.target_family = 0
    before = env._potential(params)
    env.track_builder.history.append(
        {"action": 1, "position": [9, 0, 14], "next_position": [10, 0, 14]})
    assert env._potential(params) != pytest.approx(before)
    env.track_builder.history.pop()
    assert env._potential(params) == pytest.approx(before)


def test_family_potential_streams_its_own_diagnostic(monkeypatch):
    """House rule: every new reward gate/potential streams its own diagnostic."""
    monkeypatch.setattr(oe_mod, "APIController", CompletingAPI)
    env = OpenRCT2Env(verbose=0)
    env.reward_params = replace(RewardParams(), w_family=6.0, roundtrip_gain=0.0)
    env.reset()
    env.target_family = 0
    _, _, terminated, _, info = _drive_to_terminal(env)
    assert terminated
    m = info['episode_metrics']
    # The diagnostic mirrors what _potential actually adds: _family_phi_match, not the
    # gate's _family_match -- see the phi-falloff-fix.
    assert m['family_potential'] == pytest.approx(
        env.reward_params.w_family * env._family_phi_match(env.reward_params))


def test_p6_reward_params_enable_family_potential():
    """P6 consumes the dense family potential at its full ramped-to weight (task 5b arms
    the same potential, at a lower weight, in P3-5 -- see the ramp tests below)."""
    params = ImprovedPhasedCurriculumWrapper._phase_reward_params(6)
    assert params.w_family == 6.0


# ------------------------------- family ramp, phases 3-5 (task 5b, Aug-9)
# PHASE_FAMILIES already varies the episode's seed from P3 onward, but until this task
# the entire family reward config (completion_family_floor / R_family /
# qualify_requires_family / w_family) lived inside `if phase >= 6:` -- so P3-5 handed the
# agent a one-hot that predicted nothing about its reward, exactly the "pure noise in the
# observation" PHASE_FAMILIES' own docstring cites as the reason P1-2 pin the seed. This
# also silently dropped the approved spec's Phase-3 early read (non-zero, rising per-
# family hit rates by ~1 day of training -- the go/no-go gate before the full 3-5 day
# retrain). See docs/superpowers/specs/2026-08-09-seed-conditioned-coaster-variety-design.md
# and the task-5b brief's ramp table for the exact numbers asserted below.

def test_family_reward_inert_in_phases_1_and_2():
    """P1-2 stay exactly at the RewardParams() defaults (inert) for every phase-2
    sub-stage -- PHASE_FAMILIES pins the seed to family 0 there precisely because
    nothing may read it yet."""
    W = ImprovedPhasedCurriculumWrapper
    for phase, stage in [(1, 1), (2, 1), (2, 2), (2, 3)]:
        p = W._phase_reward_params(phase, phase2_stage=stage)
        assert p.completion_family_floor == 1.0
        assert p.w_family == 0.0
        assert p.R_family == 0.0
        assert p.qualify_requires_family is False


def test_family_ramp_phases_3_4_5_match_the_table():
    """The ramp: floor loosens 0.85 -> 0.75 -> 0.60 as the budget widens, w_family rises
    3.0 -> 4.0 -> 6.0, and R_family arms at 0 -> 75 -> 125 (Aug-9 final review: the gate
    alone forfeits too little on the near bands to pay for the shape; P3 stays at 0
    because its ride test is off and R_family pays only post-test).
    qualify_requires_family stays off below P6 -- each phase keeps its own tuned
    advancement predicate."""
    W = ImprovedPhasedCurriculumWrapper
    table = {3: (0.85, 3.0, 0.0), 4: (0.75, 4.0, 75.0), 5: (0.60, 6.0, 125.0)}
    for phase, (floor, weight, bonus) in table.items():
        p = W._phase_reward_params(phase)
        assert p.completion_family_floor == floor
        assert p.w_family == weight
        assert p.R_family == bonus
        assert p.qualify_requires_family is False


def test_family_ramp_p6_unchanged():
    """Explicit values, not a delta -- so this task cannot silently retune the phase the
    earlier family tasks already tuned."""
    p6 = ImprovedPhasedCurriculumWrapper._phase_reward_params(6)
    assert p6.completion_family_floor == 0.5
    assert p6.w_family == 6.0
    assert p6.R_family == 200.0
    assert p6.qualify_requires_family is True


def test_family_ramp_is_monotone_across_phases():
    """The floor may only loosen and the potential weight may only rise as the phase
    number rises (computed by iterating the phases, not by restating the table) --
    catches a future edit to one phase that breaks the curriculum's shape."""
    W = ImprovedPhasedCurriculumWrapper
    phases = (1, 2, 3, 4, 5, 6)
    floors = [W._phase_reward_params(p).completion_family_floor for p in phases]
    weights = [W._phase_reward_params(p).w_family for p in phases]
    assert all(a >= b for a, b in zip(floors, floors[1:]))
    assert all(a <= b for a, b in zip(weights, weights[1:]))


def test_qualify_requires_family_true_only_in_p6():
    W = ImprovedPhasedCurriculumWrapper
    for phase in (1, 2, 3, 4, 5):
        assert W._phase_reward_params(phase).qualify_requires_family is False
    assert W._phase_reward_params(6).qualify_requires_family is True


def test_p3_family_gate_economics_seed_match_out_pays_mismatch():
    """Phase-3 analogue of the P6 inversion tests above (test_oval_seed_beats_winding_
    build_from_step_one and its mirror): even at P3's mild 0.85 floor -- only 15% of the
    completion payout at stake -- matching the episode's seed must out-pay ignoring it."""
    P3 = ImprovedPhasedCurriculumWrapper._phase_reward_params(3)
    oval_rows = _p6_mix(n_right=4, n_left=0)                            # oval footprint,
                                                                         # well past P3's
                                                                         # height/drop/length bars
    hit = _family_env(oval_rows, 0, P3)._calculate_reward(True, 0)      # seed asks oval
    miss = _family_env(oval_rows, 3, P3)._calculate_reward(True, 0)     # seed asks winding
    assert hit > miss


def test_family_margin_ramps_stronger_at_p6_than_p3():
    """A ramp, not a switch: the RELATIVE advantage of matching the seed must be strictly
    larger at P6 (floor 0.5) than at P3 (floor 0.85). Without this assertion a later edit
    that flattened the curriculum's shape (e.g. equalizing the floors) would go unnoticed
    by the pairwise hit/miss tests above, which only check direction, not magnitude."""
    P3 = ImprovedPhasedCurriculumWrapper._phase_reward_params(3)
    P6 = ImprovedPhasedCurriculumWrapper._phase_reward_params(6)
    oval_rows = _p6_mix(n_right=4, n_left=0)

    def margin(params):
        hit = _family_env(oval_rows, 0, params)._calculate_reward(True, 0)
        miss = _family_env(oval_rows, 3, params)._calculate_reward(True, 0)
        return hit / miss

    assert margin(P6) > margin(P3)


# ---------------------------------------------------------------- Fix 1 (Aug-9 final
# review): the family margin must beat the BUILD COST of the shape, not merely point the
# right way.
#
# Deferred finding #8 is what let this defect through: the pre-existing P3 test compares
# ONE build under two seeds, which measures gate SENSITIVITY. The decision the agent
# actually faces is the opposite -- one seed, two builds -- and it has an entry fee. With
# family_turn_falloff=5.0 / family_switch_falloff=3.0 (wider than the 4-6 turn and 1-3
# switch gaps between footprint.FAMILIES' bands) the 4-turn oval the policy already makes
# scored 0.800 against a spiral seed and 0.633 against an out_and_back seed, so the
# forfeited share of the completion payout (30 / 55 at P3) was smaller than the cost of
# building the shape and the optimal policy was "build the oval anyway".
#
# THE COST MODEL, stated:
#   * extra pieces -- FAMILY_EXTRA_PIECES below. The near-band footprints need 4-10 more
#     pieces than the 4-turn oval (final-fix-brief.md, Fix 1).
#   * cost per piece -- the gamma discount of one extra build step, (1 - gamma) * payout.
#     Derived from the phase's own payout rather than hardcoded; it lands at 7.8-10.2
#     points here, matching CLAUDE.md's recorded "~-10/piece gamma-discount of the
#     completion payout" from the Jul-5 length-trap diagnosis.
#
# Pre-launch final re-review: the prior value (9) was chosen as "the largest value P3's
# repaired margin still covers" -- a constant fitted to the result it certifies proves
# nothing, and at 10 (still inside the 4-10 range above) the old P3 spiral assertion
# failed outright (spiral was never closable in P3's budget at all -- see PHASE_FAMILIES,
# now fixed by moving spiral to P5/P6 and re-aiming P3 at out-and-back, which it actually
# offers). This bar covers ONLY the gamma discount of the extra pieces -- it deliberately
# EXCLUDES closure risk, which the re-review measured as the dominant term the agent
# actually weighs. 12 is comfortably above the 4-10 range the design discussed; every
# phase below still clears it with room (P3/family 2: margin 112.5 vs bar 93.0, headroom
# 19.5; P4/family 3: margin 325.0 vs bar 106.5; P5/family 3-4: margin 509.0 vs bar 122.4;
# P6/family 3-4: margin 880.0 vs bar 138.9 -- narrowest at P3, as expected from its mild
# 0.85 floor, but still clearing).
FAMILY_EXTRA_PIECES = 12


def _family_fixture(z):
    """A P6-scale build (chain hill + steep drop + length, so every phase's structural
    bar is saturated and only the SHAPE differs) whose footprint is VERIFIED to classify
    into family `z`. This plan has shipped a mis-specified footprint fixture four times;
    the assert is the guard."""
    rows = {
        0: _p6_mix(n_right=4, n_left=0),                        # 4 turns, 0 switches
        1: _p6_mix(n_right=8, n_left=0),                        # 8 turns, 0 switches
        2: _p6_mix(fill=[4, 4, 4, 3, 3, 3, 4, 4, 4]),           # 9 turns, 2 switches
        3: _p6_mix(fill=[4, 4, 3, 3] * 3),                      # 12 turns, 5 switches
        4: _p6_mix(fill=[4, 4, 3, 3] * 4),                      # 16 turns, 7 switches
    }[z]
    assert classify_family([r[0] for r in rows]) == z, \
        "fixture must BE the family it is named for"
    return rows


def _family_inversion_payouts(params, z, excitement=5.6):
    """The four terminal payouts of the inversion pair: the default oval build and the
    family-`z` build, each scored under an oval seed and under a family-`z` seed."""
    oval, shaped = _family_fixture(0), _family_fixture(z)
    return {
        ("oval", 0): _p6_payout(params, oval, excitement=excitement, family=0),
        ("shaped", 0): _p6_payout(params, shaped, excitement=excitement, family=0),
        ("oval", z): _p6_payout(params, oval, excitement=excitement, family=z),
        ("shaped", z): _p6_payout(params, shaped, excitement=excitement, family=z),
    }


def _assert_family_economics(phase):
    """Inversion + magnitude for every non-oval family the phase actually seeds."""
    W = ImprovedPhasedCurriculumWrapper
    params = W._phase_reward_params(phase)
    for z in W.PHASE_FAMILIES[phase]:
        if z == 0:
            continue
        pay = _family_inversion_payouts(params, z)
        # Inversion: the SAME two builds, each seed preferring its own.
        assert pay[("oval", 0)] > pay[("shaped", 0)], f"phase {phase}, family {z}"
        assert pay[("shaped", z)] > pay[("oval", z)], f"phase {phase}, family {z}"
        # Magnitude: the win must pay for the shape, not just exist.
        per_piece = (1.0 - params.gamma) * pay[("shaped", z)]
        cost = FAMILY_EXTRA_PIECES * per_piece
        margin = pay[("shaped", z)] - pay[("oval", z)]
        assert margin > cost, (
            f"phase {phase}, family {z}: margin {margin:.1f} does not cover "
            f"{FAMILY_EXTRA_PIECES} extra pieces at {per_piece:.2f}/piece ({cost:.1f})")


def test_p3_seeded_family_out_pays_the_default_oval_by_the_build_cost():
    _assert_family_economics(3)


def test_p4_seeded_family_out_pays_the_default_oval_by_the_build_cost():
    _assert_family_economics(4)


def test_p5_seeded_family_out_pays_the_default_oval_by_the_build_cost():
    _assert_family_economics(5)


def test_p6_seeded_family_out_pays_the_default_oval_by_the_build_cost():
    """P6 was already correct by construction (floor 0.5 + R_family=200); asserted here
    so the whole ramp is covered by one criterion."""
    _assert_family_economics(6)


def test_family_falloffs_make_the_bands_distinguishable():
    """Fix 1A. The falloffs must be NARROWER than the gaps between footprint.FAMILIES'
    bands (4-6 turns, 1-3 switches) or a build scores high credit against seeds it is
    nothing like. 2.0 is the floor, not a target: one turn short of the band still scores
    0.5 on that leg and one switch short still scores 0.5, so the near-miss ramp this
    codebase's house rule requires survives ('every leg ... needs its own ramp')."""
    p = RewardParams()
    assert (p.family_turn_falloff, p.family_switch_falloff) == (2.0, 2.0)
    oval = [r[0] for r in _family_fixture(0)]
    env = _bare_env(history=_env_hist(_family_fixture(0)))

    def match(seed):
        env.target_family = seed
        return env._family_match(p)

    assert match(0) == pytest.approx(1.0)
    assert match(1) == pytest.approx(0.50)     # spiral:       6-9 turns, 0 switches
    assert match(2) == pytest.approx(0.25)     # out_and_back: 6-9 turns, 1-2 switches
    assert match(3) == pytest.approx(0.0)      # winding
    assert match(4) == pytest.approx(0.0)      # serpentine
    # ...and the near-miss ramp is still graded, not a cliff: one turn short of the
    # spiral band scores exactly 0.5 on the turn leg.
    near = _bare_env(history=_env_hist([(4, 14, 14)] * 5))
    near.target_family = 1
    assert near._family_match(p) == pytest.approx(0.5 * (0.5 + 1.0))
    assert len(oval) == len(_family_fixture(1))   # the pair differs only in SHAPE


def test_family_ramp_arms_r_family_at_p4_and_p5():
    """Fix 1B. Below P6 the multiplicative gate was the ONLY family incentive, and at
    P4/P5 floors (0.75/0.60) it is too small on the near bands. R_family gives them the
    same discrete step P6 gives. P3 stays at 0: ride testing is off there, so R_family --
    which pays only inside the completion-AND-tested branch -- could never fire."""
    W = ImprovedPhasedCurriculumWrapper
    assert W._phase_reward_params(3).R_family == 0.0
    assert W._phase_reward_params(4).R_family == 75.0
    assert W._phase_reward_params(5).R_family == 125.0
    assert W._phase_reward_params(6).R_family == 200.0
    # ...and it stays a RAMP: monotone with the phase number, like the floor and weight.
    bonuses = [W._phase_reward_params(p).R_family for p in (1, 2, 3, 4, 5, 6)]
    assert all(a <= b for a, b in zip(bonuses, bonuses[1:]))


def test_r_family_cannot_break_completion_first_in_p4_p5(monkeypatch):
    """R_family is paid inside step()'s completion-and-tested branch, so it can only ever
    ADD to the completion side of the completion-first inequality (climb milestones must
    stay below what closing the loop pays). Checked here with the numbers, not assumed."""
    W = ImprovedPhasedCurriculumWrapper
    for phase in (4, 5):
        p = W._phase_reward_params(phase)          # raises if the invariant is violated
        milestones = p.R_roundtrip + p.R_summit
        assert milestones < p.R_complete
        flat = (p.completion_hill_floor * p.completion_length_floor
                * p.completion_quality_floor * p.completion_style_floor
                * p.completion_family_floor * p.R_complete)
        # P4 sets completion_hill_floor=0.0, so only the first check binds there (see
        # _validate_completion_first); P5's floors are all >= its own milestone total.
        assert p.completion_hill_floor == 0.0 or milestones < flat
    # Fix 3 (Aug-9 final review): the OLD version of this section called
    # env._calculate_reward(True, 0) on two COMPLETED episodes -- a branch that never
    # reads R_family at all (R_family is paid at openrct2_env.py:570, inside `if
    # terminated:`, which sits in step(), one level above _calculate_reward) -- so the
    # comparison held for EVERY value of R_family and could never fail. This instead
    # drives a real step()-level TRUNCATED episode whose partial build already matches
    # the seed (a guaranteed _family_hit()), and checks the actual payment site: the
    # reward must come out identical whether R_family is armed or zeroed, and the
    # _last_family_bonus / episode_metrics 'family_bonus' tag must read exactly 0.0. It
    # fails immediately if R_family's payment is ever moved outside the completion
    # branch (verified below: monkeypatching it onto the truncation path flips this
    # test red while leaving the old completed-vs-completed comparison unmoved).
    monkeypatch.setattr(oe_mod, "APIController", FakeAPI)   # never completes the circuit

    def _run_truncated(params):
        env = OpenRCT2Env(verbose=0)
        env.skip_ride_testing = True
        env.reward_params = params
        env.reset()
        env.target_family = 0                            # oval: <= 5 turns, 0 switches
        env.max_track_length = 4
        for _ in range(3):
            env.step(4)                                  # right turns, one handedness
        _, reward, terminated, truncated, info = env.step(4)
        assert truncated and not terminated
        assert env.loop_completed is False
        assert env._family_hit() is True                 # would be a hit, WERE it read
        return reward, env, info

    for phase in (4, 5):
        p = W._phase_reward_params(phase)
        assert p.R_family > 0.0                          # the branch is actually armed here
        armed_reward, armed_env, armed_info = _run_truncated(p)
        zeroed_reward, _, _ = _run_truncated(replace(p, R_family=0.0))
        assert armed_env._last_family_bonus == 0.0
        assert armed_info['episode_metrics']['family_bonus'] == 0.0
        assert armed_reward == pytest.approx(zeroed_reward), (
            f"phase {phase}: truncation reward changed with R_family armed -- it must "
            "only be reachable through the completion (terminated) branch")


# --------------------------------------------------- Fix 3 (Aug-9 final review): the
# family_hit diagnostic must be completion-conditioned.
#
# family_hit was computed LIVE at metrics time while family_gate/family_match are only
# set on completion. A truncated 20-piece build with 2 turns and 0 switches classifies as
# `oval`, so every oval-seeded TRUNCATION reported family_hit=1.0 next to family_gate=0.0.
# Oval is 20% of the seed draws in P5/P6, so structure/family_hit_cold would sit near 0.2
# for a policy that never completes -- a false floor on the tag a human watches hourly
# through an unattended multi-day run.

def test_family_hit_is_zero_on_a_truncated_episode(monkeypatch):
    """The partial track DOES classify into the requested family; the episode still did
    not close, so the hit tag must read 0."""
    monkeypatch.setattr(oe_mod, "APIController", FakeAPI)   # never completes the circuit
    env = OpenRCT2Env(verbose=0)
    env.skip_ride_testing = True
    env.reset()
    env.target_family = 0                                   # oval: <= 5 turns, 0 switches
    env.max_track_length = 4
    for _ in range(3):
        env.step(4)                                         # right turns, one handedness
    _, _, terminated, truncated, info = env.step(4)
    assert truncated and not terminated
    assert env.loop_completed is False
    # the build really is an oval -- this is the false-positive the fix removes
    assert classify_family(env._history_actions()) == 0
    assert env._family_hit() is True
    m = info['episode_metrics']
    assert m['family_hit'] == 0.0
    assert m['family_gate'] == 0.0                          # ...consistent with its sibling


def test_family_hit_still_reports_on_a_completed_episode(monkeypatch):
    """Guard against the null fix (zeroing the tag outright)."""
    monkeypatch.setattr(oe_mod, "APIController", CompletingAPI)
    env = OpenRCT2Env(verbose=0)
    env.reward_params = replace(RewardParams(), completion_family_floor=0.5,
                                roundtrip_gain=0.0)
    env.reset()
    env.target_family = 0
    _, _, terminated, _, info = _drive_to_terminal(env)
    assert terminated and env.loop_completed is True
    assert info['episode_metrics']['family_hit'] == float(env._family_hit()) == 1.0


# ------------------------------------------------- Fix 5 (Aug-9 final review): the dense
# potential's balance leg exactly cancelled the dense family potential on the switch axis.
#
# _exc_feature_quality's balance leg is min(turn_balance / struct_turn_balance_target, 1),
# and the target was still 2.0 in P5/P6 even though struct_w_turn_balance was correctly
# zeroed. turn_balance is min(left, right), so the leg is STRUCTURALLY unreachable for the
# two 0-switch families (oval and spiral) -- a permanent forfeit on 2 of 5 seeds -- and on
# an oval seed the arithmetic cancelled: adding one direction switch (2L+2R => balance 2)
# gained w_exc_feat * (1/6) * 1 = 1.0 of Phi and lost w_family * 0.5 * (1/3) = 1.0 (at the
# pre-Fix-1 switch falloff of 3.0). Zero net dense gradient on exactly the axis that
# separates oval from out_and_back and spiral from out_and_back -- and the dense potential
# exists because terminal-only shaping is too slow here.

def _switch_axis_pair():
    """Two builds identical in EVERY other reward-visible feature -- same length, same
    piece count, same hill/drop/banked profile, same TURN count (4) -- differing only in
    handedness: 4 right turns (0 switches, balance 0) vs R,R,L,L (1 switch, balance 2)."""
    one_handed = _p6_mix(n_right=4, n_left=0)
    switched = _p6_mix(fill=[4, 4, 3, 3])
    a, b = _bare_env(history=_env_hist(one_handed)), _bare_env(history=_env_hist(switched))
    assert a._turn_count() == b._turn_count() == 4
    assert (a._turn_balance_count(), b._turn_balance_count()) == (0, 2)
    assert (switch_count(a._history_actions()), switch_count(b._history_actions())) == (0, 1)
    assert a.track_length == b.track_length
    return a, b


def test_p5_p6_turn_balance_target_is_retired():
    """Fix 5: with the weight already 0, a nonzero target only kept the unreachable leg
    alive inside the DENSE potential. Retiring the target makes that leg a constant 0."""
    W = ImprovedPhasedCurriculumWrapper
    for phase in (5, 6):
        p = W._phase_reward_params(phase)
        assert p.struct_w_turn_balance == 0.0
        assert p.struct_turn_balance_target == 0.0
        # no other live consumer: the style gate is retired and the struct leg is unweighted
        assert p.completion_style_floor == 1.0
        a, b = _switch_axis_pair()
        assert a._exc_feature_quality(p) == pytest.approx(b._exc_feature_quality(p))


def test_adding_a_switch_on_an_oval_seed_is_net_negative_in_phi():
    """THE cancellation, asserted with its arithmetic so it cannot silently return.

    On an oval seed (0 switches by band) the switch must cost, and cost the FULL family
    loss. Aug-10: the dense w_family term in _potential now reads its OWN wide falloff
    (family_phi_switch_falloff=6.0), not the gate's family_switch_falloff=2.0 -- see the
    phi-falloff-fix. So the loss is w_family * 0.5 * (1 / family_phi_switch_falloff) =
    6 * 0.5 * (1/6) = 0.5, with the exc-feature balance leg contributing exactly nothing
    (Fix 5 retired its target)."""
    W = ImprovedPhasedCurriculumWrapper
    for phase in (5, 6):
        p = W._phase_reward_params(phase)
        a, b = _switch_axis_pair()
        a.target_family = b.target_family = 0
        expected_family_loss = p.w_family * 0.5 * (1.0 / p.family_phi_switch_falloff)
        assert expected_family_loss == pytest.approx(0.5)
        d_phi = b._potential(p) - a._potential(p)
        assert d_phi < 0.0
        assert d_phi == pytest.approx(-expected_family_loss), f"phase {phase}"
        # ...and the whole delta is the family term: the exc-feature term is flat here.
        d_exc = p.w_exc_feat * (b._exc_feature_quality(p) - a._exc_feature_quality(p))
        assert d_exc == pytest.approx(0.0)


# ----------------------------------------------- Fix 6.1 (Aug-9 final review): a retired
# gate must not stream as a total forfeit.
#
# completion_style_floor=1.0 in P6 means the style block in _calculate_reward never runs,
# so _last_style_gate stayed at its 0.0 reset and rewards/style_gate streamed a constant
# 0.0 -- which reads as "every completion forfeited all of its style money", the opposite
# of "this gate is retired". Chosen fix: stop EMITTING the key while the gate is inert
# (rather than emitting 1.0). It matches the convention the curriculum wrapper already
# uses for family_hit_rate_* -- gate the KEY itself on applicability, not just its value
# -- train.py already writes the scalar only `if 'style_gate' in info_metrics`, and an
# absent series is unambiguous in TB in a way that a flat line is not.

def test_style_gate_key_absent_while_the_gate_is_retired(monkeypatch):
    monkeypatch.setattr(oe_mod, "APIController", CompletingAPI)
    env = OpenRCT2Env(verbose=0)
    P6 = ImprovedPhasedCurriculumWrapper._phase_reward_params(6)
    assert P6.completion_style_floor == 1.0            # retired by the family gate
    env.reward_params = P6
    env.reset()
    _, _, terminated, _, info = _drive_to_terminal(env)
    assert terminated
    m = info['episode_metrics']
    assert 'style_gate' not in m
    assert 'family_gate' in m                          # its live successor still streams


def test_style_gate_key_present_while_the_gate_is_armed(monkeypatch):
    """Guard against the null fix (dropping the diagnostic outright)."""
    monkeypatch.setattr(oe_mod, "APIController", CompletingAPI)
    env = OpenRCT2Env(verbose=0)
    env.reward_params = replace(RewardParams(), completion_style_floor=0.6,
                                struct_turns_target=12.0, roundtrip_gain=0.0)
    env.reset()
    _, _, terminated, _, info = _drive_to_terminal(env)
    assert terminated
    assert info['episode_metrics']['style_gate'] > 0.0


def test_phases_1_and_2_reward_is_bit_identical_under_every_seed():
    """The Aug-9 final-fix pass changed RewardParams' family falloff DEFAULTS, which P1/P2
    inherit. Nothing there may read them: the gate floor is 1.0, w_family and R_family are
    0 and qualify_requires_family is False, so the reward must be numerically identical
    for every possible seed. Verified, not assumed (the brief's standing requirement that
    phases 1-2 stay bit-identical)."""
    W = ImprovedPhasedCurriculumWrapper
    rows = _p6_mix(fill=[4, 4, 3, 3] * 3)                  # a shape that is NOT an oval
    for phase, stage in [(1, 1), (2, 1), (2, 2), (2, 3)]:
        p = W._phase_reward_params(phase, phase2_stage=stage)
        assert (p.completion_family_floor, p.w_family, p.R_family) == (1.0, 0.0, 0.0)
        assert p.qualify_requires_family is False
        rewards, phis = set(), set()
        for seed in range(FAMILY_N):
            env = _family_env(rows, seed, p)
            phis.add(round(env._potential(p), 10))
            env._phi_prev = 0.0
            rewards.add(round(env._calculate_reward(True, 0), 10))
        assert len(rewards) == 1, f"phase {phase}.{stage} reward varies with the seed"
        assert len(phis) == 1, f"phase {phase}.{stage} potential varies with the seed"


# ------------------------------------------------- phi-falloff fix (Aug-10): the dense
# family potential (w_family term) shared its falloffs with the completion gate. Fix 1A
# correctly narrowed the gate's falloffs (2.0/2.0) so the bands became distinguishable,
# but the SAME constants also drove the dense potential, which then went flat at exactly
# 0.000 over most of the discovery range (measured: 0-6 turns toward winding, 0-13 toward
# serpentine) -- no per-step gradient at all from a standing start. _family_phi_match now
# grades the potential with its own, wider falloffs (family_phi_turn_falloff=12.0,
# family_phi_switch_falloff=6.0); _family_match (the gate) is untouched.

def test_family_phi_potential_rises_from_a_standing_start_toward_winding():
    """THE regression. At the gate falloffs (2.0/2.0), Phi_family for a winding-target
    build stayed at exactly 0.000 from 0 to 6 turns -- no gradient at all until the agent
    stumbled into the band by chance. Under the repaired falloffs it must be strictly
    positive at 0 turns and rise monotonically (non-decreasing) through 0->10 turns."""
    W = ImprovedPhasedCurriculumWrapper
    right = RIGHT_TURN_ACTIONS[0]
    for phase in (4, 5):
        p = W._phase_reward_params(phase)
        assert (p.family_phi_turn_falloff, p.family_phi_switch_falloff) == (12.0, 6.0)
        phis = []
        for n in range(11):
            env = _bare_env(history=_env_hist([(right, 14, 14)] * n))
            env.target_family = 3   # winding: 10-13 turns, 3-5 switches
            phis.append(env._family_phi_match(p))
        assert phis[0] > 0.0, (
            f"phase {phase}: Phi_family must not be flat at 0 from a standing start, got {phis}")
        assert all(a <= b + 1e-9 for a, b in zip(phis, phis[1:])), (
            f"phase {phase}: Phi_family must rise monotonically 0->10 turns, got {phis}")


def test_family_phi_potential_rises_well_before_the_serpentine_band():
    """Same defect, the serpentine band (14+ turns, unbounded above): at the gate
    falloffs Phi_family stayed at 0.000 until turn 13. Under the repaired falloffs it
    must already be positive and rising well before the band (at turn 6, 8 turns short)."""
    W = ImprovedPhasedCurriculumWrapper
    right = RIGHT_TURN_ACTIONS[0]
    for phase in (4, 5):
        p = W._phase_reward_params(phase)

        def phi_at(n, p=p):
            env = _bare_env(history=_env_hist([(right, 14, 14)] * n))
            env.target_family = 4   # serpentine: 14+ turns, 6+ switches
            return env._family_phi_match(p)

        phi6, phi10, phi13 = phi_at(6), phi_at(10), phi_at(13)
        assert phi6 > 0.0, (
            f"phase {phase}: Phi_family must be positive well before the 14-turn band, got {phi6}")
        assert phi10 > phi6, f"phase {phase}"
        assert phi13 > phi10, f"phase {phase}"


def test_gate_family_match_unchanged_by_the_phi_falloff_fix():
    """The completion gate (_family_match) must be untouched: same narrow falloffs, same
    values. These are the exact numbers that motivated narrowing the falloffs in the
    first place (Fix 1A) -- an oval build against a spiral seed and against an
    out_and_back seed."""
    p = RewardParams()
    assert (p.family_turn_falloff, p.family_switch_falloff) == (2.0, 2.0)
    oval_env = _bare_env(history=_env_hist(_family_fixture(0)))
    oval_env.target_family = 1   # spiral: 6-9 turns, 0 switches
    assert oval_env._family_match(p) == pytest.approx(0.500)
    oval_env.target_family = 2   # out_and_back: 6-9 turns, 1-2 switches
    assert oval_env._family_match(p) == pytest.approx(0.250)


def test_family_phi_potential_telescopes_toward_winding():
    """PBRS hygiene under the REPAIRED (wide) falloffs and a non-oval target: a
    place-then-remove cycle must return Phi to its prior value exactly, and the placed
    piece must genuinely move Phi in between (else this would pass for an implementation
    that ignores the family term entirely)."""
    W = ImprovedPhasedCurriculumWrapper
    p = W._phase_reward_params(5)
    right = RIGHT_TURN_ACTIONS[0]
    env = _bare_env(history=_env_hist([(right, 14, 14)] * 3))
    env.target_family = 3   # winding
    before = env._potential(p)
    env.track_builder.history.append(
        {"action": right, "position": [9, 0, 14], "next_position": [10, 0, 14]})
    after = env._potential(p)
    assert after != pytest.approx(before)
    env.track_builder.history.pop()
    assert env._potential(p) == pytest.approx(before)


def test_family_phi_falloff_fields_default_wide_and_gate_falloffs_stay_narrow():
    """The two pairs are independent RewardParams fields: the gate's stay narrow
    (discrimination), the potential's default wide (guidance range)."""
    p = RewardParams()
    assert p.family_phi_turn_falloff == 12.0
    assert p.family_phi_switch_falloff == 6.0
    assert (p.family_turn_falloff, p.family_switch_falloff) == (2.0, 2.0)


def test_family_phi_falloffs_inert_in_phases_1_and_2():
    """w_family is 0 in P1/P2 (PHASE_FAMILIES pins the seed to family 0 there with
    nothing reading it), so _family_phi_match is never called from _potential and the
    new falloffs cannot affect those phases -- inherited defaults only, no per-phase
    override exists for them."""
    W = ImprovedPhasedCurriculumWrapper
    for phase, stage in [(1, 1), (2, 1), (2, 2), (2, 3)]:
        p = W._phase_reward_params(phase, phase2_stage=stage)
        assert p.w_family == 0.0
        assert (p.family_phi_turn_falloff, p.family_phi_switch_falloff) == (12.0, 6.0)
