"""Reward-system overhaul tests (PBRS + unified parametrized reward).

Server-free: the env is built via ``__new__`` and its internal state set by hand,
or driven through a ``FakeAPI`` (mirroring test_env_smoke.py). Reward math is a pure
function of internal state (no API calls), so these run without an OpenRCT2 server.

Note: ``FakeAPI`` hardcodes ``isCircuitComplete=False``, so completion / terminal-Phi /
completion-first tests use the ``__new__`` + hand-set ``loop_completed`` path.
"""
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from openrct2_gym.envs import openrct2_env as oe_mod
from openrct2_gym.envs.openrct2_env import OpenRCT2Env, RewardParams
from openrct2_gym.envs.obs_config import make_observation_space, SCALE, H_SCALE
from openrct2_gym.tests.test_env_smoke import FakeAPI

DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # N, E, S, W


@pytest.fixture(autouse=True)
def _isolate_close_cache(tmp_path):
    """Isolate the process-wide calibration cache + its file per test."""
    orig_cache = OpenRCT2Env._close_cache
    orig_path = OpenRCT2Env._CLOSE_CACHE_PATH
    OpenRCT2Env._close_cache = None
    OpenRCT2Env._CLOSE_CACHE_PATH = str(tmp_path / "close_geometry.json")
    yield
    OpenRCT2Env._close_cache = orig_cache
    OpenRCT2Env._CLOSE_CACHE_PATH = orig_path


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
    geo = RewardParams(w_e=0.0)  # isolate geometry
    phis = [_phi_env((d, 0, 14))._potential(geo) for d in (30, 20, 10, 0)]
    assert phis == sorted(phis)              # monotonically increasing
    assert phis[-1] > phis[0]


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
    """A one-tile XY move toward the (aligned) anchor must give F = gamma*Phi' - Phi > 0
    at the default D_xy, even at high Phi where discount leakage (1-gamma)*Phi bites."""
    p = RewardParams()
    far = _phi_env((11, 0, 14), direction=1)._potential(p)
    near = _phi_env((10, 0, 14), direction=1)._potential(p)
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


def test_quality_bonus_symmetric_excitement_falloff():
    env = _bare_env()
    p = RewardParams(R_quality_max=500.0)
    assert env._quality_bonus(7.0, 5.5, 2.0, p) == pytest.approx(
        env._quality_bonus(9.0, 5.5, 2.0, p), rel=1e-6)


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


def test_maybe_capture_persists_once_to_cache_and_file():
    import json, os
    env = _bare_env()
    env.track_builder = SimpleNamespace(history=_completing_history())
    env.loop_completed = True

    env._maybe_capture_closing_geometry()
    assert OpenRCT2Env._close_cache["pos"] == [61, 68, 14]
    assert os.path.exists(OpenRCT2Env._CLOSE_CACHE_PATH)
    with open(OpenRCT2Env._CLOSE_CACHE_PATH) as f:
        assert json.load(f)["dir"] == 2

    # second completion must NOT overwrite the first calibration
    other = _completing_history()
    other[-1]["position"] = [99, 99, 14]
    env.track_builder = SimpleNamespace(history=other)
    env._maybe_capture_closing_geometry()
    assert OpenRCT2Env._close_cache["pos"] == [61, 68, 14]


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


def test_init_closing_target_provisional_without_cache():
    OpenRCT2Env._close_cache = None
    env = _bare_env(goal_position=(62, 66, 14))
    env._init_closing_target()
    assert env.close_pos is None
    assert env.close_dir is None
    assert list(env._reward_target_position()) == [62, 66, 14]   # provisional
    assert env._reward_target_direction() is None


# ------------------------------------------------------- PBRS reward (tests 2,3,5,10)

def test_pbrs_constructive_move_reward_is_positive():
    p = RewardParams()
    env = _phi_env((11, 0, 14), direction=1)
    env._phi_prev = env._potential(p)
    env.current_position = [10, 0, 14]          # one tile closer to the anchor
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


# ----------------------------------------------- curriculum unification (tests 12,13,15)

from openrct2_gym.envs.improved_phased_curriculum_wrapper import ImprovedPhasedCurriculumWrapper


# ============================ lift-hill incentive (structural bonus + discovery) ============

def test_phase_reward_params_structural_per_phase():
    W = ImprovedPhasedCurriculumWrapper
    p1 = W._phase_reward_params(1)
    assert p1.R_struct_max == 0.0                       # struct off in P1
    p2 = W._phase_reward_params(2)
    assert (p2.R_struct_max, p2.struct_w_chain, p2.struct_w_drop, p2.struct_chain_target) \
        == (250.0, 1.0, 0.0, 3)                         # chains only, target 3 (matches >=3 gate)
    p3 = W._phase_reward_params(3)
    assert (p3.R_struct_max, p3.struct_w_chain, p3.struct_w_drop, p3.struct_chain_target) \
        == (250.0, 0.5, 0.5, 2)                         # chains AND drop, target 2 (matches >=2 gate)
    p4 = W._phase_reward_params(4)
    assert (p4.R_struct_max, p4.struct_w_chain, p4.struct_w_drop, p4.struct_chain_target) \
        == (250.0, 0.5, 0.5, 3)                         # integration, target 3
    p5 = W._phase_reward_params(5)
    assert p5.R_struct_max == 0.0 and p5.R_quality_max == 500.0   # struct off, quality on
    # discovery potential weights are FIXED across all phases
    for p in (p1, p2, p3, p4, p5):
        assert p.w_h == 6.0 and p.h_scale == 6.0


# ---- structural bonus

def _struct_env(chains=0, drops=0):
    """A bare env whose history has `chains` chain-lift pieces (action 9) climbing, then
    `drops` drop pieces (action 6) descending."""
    hist, z = [], 14
    for _ in range(chains):
        z += 1
        hist.append({"action": 9, "next_position": [0, 0, z]})
    for _ in range(drops):
        z -= 1
        hist.append({"action": 6, "next_position": [0, 0, z]})
    return _bare_env(history=hist)


def test_structural_bonus_disabled_returns_zero():
    env = _struct_env(chains=3, drops=1)
    assert env._structural_bonus(RewardParams(R_struct_max=0.0)) == 0.0      # P1/P5


def test_structural_bonus_p2_scales_with_chain_count():
    p = RewardParams(R_struct_max=250.0, struct_chain_target=3, struct_w_chain=1.0, struct_w_drop=0.0)
    assert _struct_env(chains=1)._structural_bonus(p) == pytest.approx(250.0 / 3)
    assert _struct_env(chains=2)._structural_bonus(p) == pytest.approx(250.0 * 2 / 3)
    assert _struct_env(chains=3)._structural_bonus(p) == pytest.approx(250.0)
    assert _struct_env(chains=4)._structural_bonus(p) == pytest.approx(250.0)   # clipped at target
    assert _struct_env(chains=0)._structural_bonus(p) == 0.0


def test_structural_bonus_p3_requires_chains_and_drop():
    p = RewardParams(R_struct_max=250.0, struct_chain_target=2, struct_w_chain=0.5, struct_w_drop=0.5)
    assert _struct_env(chains=2, drops=1)._structural_bonus(p) == pytest.approx(250.0)   # both -> full
    assert _struct_env(chains=2, drops=0)._structural_bonus(p) == pytest.approx(125.0)   # chains only
    assert _struct_env(chains=0, drops=1)._structural_bonus(p) == pytest.approx(125.0)   # drop only
    assert _struct_env(chains=0, drops=0)._structural_bonus(p) == 0.0


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
    env.reward_params = RewardParams(R_struct_max=250.0, struct_chain_target=1,
                                     struct_w_chain=1.0, struct_w_drop=0.0)
    env.reset()
    info = {}
    for _ in range(6):
        _, _, terminated, truncated, info = env.step(9)   # chain lifts -> completes
        if terminated or truncated:
            break
    m = info['episode_metrics']
    assert {'chain_count', 'struct_bonus', 'max_gain'}.issubset(m)   # callback's contract
    assert m['chain_count'] >= 1 and m['struct_bonus'] > 0 and m['max_gain'] >= 0


# ---- discovery potential (elevation term in Phi)

_DISC = RewardParams(w_xy=0.0, w_z=0.0, w_dir=0.0, w_e=0.0, w_h=6.0, h_scale=6.0)  # isolate discovery


def _peak_env(peak_z, head_z=None):
    """Bare env whose history reaches `peak_z`; head height defaults to peak."""
    head_z = peak_z if head_z is None else head_z
    return _bare_env(current_position=(0, 0, head_z),
                     history=[{"action": 0, "next_position": [0, 0, peak_z]}])


def test_discovery_potential_increases_and_saturates():
    phis = [_peak_env(z)._potential(_DISC) for z in (14, 16, 20, 30)]
    assert phis[0] == pytest.approx(0.0)        # no gain
    assert phis[1] == pytest.approx(2.0)        # gain 2 -> 6*2/6
    assert phis[2] == pytest.approx(6.0)        # gain 6 -> saturates
    assert phis[3] == pytest.approx(6.0)        # gain 16 -> clipped at h_scale
    assert phis == sorted(phis)


def test_discovery_potential_banks_peak_after_descent():
    env = _bare_env(current_position=(0, 0, 14),
                    history=[{"action": 0, "next_position": [0, 0, 30]},   # climbed
                             {"action": 6, "next_position": [0, 0, 14]}])  # back to station height
    assert env._potential(_DISC) == pytest.approx(6.0)   # banked peak, NOT 0 despite head at z=14


def test_discovery_potential_empty_history_no_raise():
    env = _bare_env(history=[])              # max() of empty would raise without the guard
    assert env._potential(_DISC) == pytest.approx(0.0)


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
    # Load-bearing (feedback #6): compare actual transition REWARDS, and pin that the
    # discovery term materially widens the climb>flat margin (it must, or w_h is dead).
    assert _climb_vs_flat_gap(6.0) > 0                            # climb strictly preferred
    assert _climb_vs_flat_gap(6.0) > _climb_vs_flat_gap(0.0) + 0.5  # discovery does real work


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
    assert base_env.reward_params.step_cost < 0
    assert base_env.skip_ride_testing is False


def test_p3_qualified_requires_chains_and_drop():
    W = ImprovedPhasedCurriculumWrapper
    w = W.__new__(W)                          # no __init__/env needed for the predicate

    def base(actions):
        return SimpleNamespace(track_builder=SimpleNamespace(
            history=[{"action": a} for a in actions]))

    w.current_phase = 3
    assert w._is_qualified(base([9, 9, 6]), True) is True     # 2 chains AND a drop
    assert w._is_qualified(base([9, 9]), True) is False       # chains, no drop
    assert w._is_qualified(base([6]), True) is False          # drop only (was True under old OR gate)
    assert w._is_qualified(base([9, 9, 6]), False) is False   # not completed
    w.current_phase = 2
    assert w._is_qualified(base([9, 9, 9]), True) is True     # P2: >=3 chains
    assert w._is_qualified(base([9, 9]), True) is False
    w.current_phase = 4
    assert w._is_qualified(base([9, 9, 6]), True) is None     # no structural gate


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
    # phase 5 also applies the small step_cost on the completing step.
    assert reward == pytest.approx(p.R_complete - phi_prev_before + quality + p.step_cost)


# ----------------------------------------------------- gamma single source (test 9)

def test_gamma_single_sourced_to_reward_params(monkeypatch):
    import train_parallel_curriculum_masked as T
    # The model discount is sourced from RewardParams (the same class the env uses).
    assert T.GAMMA == RewardParams().gamma

    monkeypatch.setattr(oe_mod, "APIController", FakeAPI)
    env = T.create_curriculum_masked_env(8080, use_improved=True, verbose=0)
    base = env
    while hasattr(base, "env"):
        base = base.env
    # The env the PPO model trains against discounts its PBRS potential with the same gamma.
    assert base.reward_params.gamma == T.GAMMA


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
    env.reset()
    assert env.close_pos is None         # provisional in the first episode
    _drive_to_terminal(env)              # completes -> calibration captured
    assert OpenRCT2Env._close_cache is not None
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
    assert env.close_pos is None         # corrupted -> ignored, falls back to provisional
    assert env.close_dir is None


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

    # a subsequent real completion now calibrates (self-repair)
    env.track_builder = SimpleNamespace(history=_completing_history())
    env.loop_completed = True
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
