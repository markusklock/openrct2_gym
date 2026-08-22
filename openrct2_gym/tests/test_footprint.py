"""Footprint families: the shape a coaster traces on the ground.

Turn count and direction-switch count are what the eye registers from above -- an
oval never alternates direction, a serpentine alternates constantly. Both come from
the action sequence alone, so this is pure and server-free.
"""
import pytest

from openrct2_gym.envs.footprint import (
    FAMILIES, FAMILY_N, classify_family, family_match, switch_count)

OVAL = [4, 0, 4, 0, 4, 0, 4]                        # 4 right turns, never alternates
SPIRAL = [4] * 8                                     # 8 right turns, never alternates
OUT_AND_BACK = [4, 4, 4, 0, 3, 3, 3, 0]              # 6 turns, one L/R switch
WINDING = [4, 4, 4, 0, 3, 3, 3, 0, 4, 4, 4, 0, 3, 3, 3, 0]       # 12 turns, 3 switches
SERPENTINE = [4, 3] * 8                              # 16 turns, alternating every piece


def test_switch_count_counts_direction_alternations():
    assert switch_count(OVAL) == 0
    assert switch_count(OUT_AND_BACK) == 1
    assert switch_count(WINDING) == 3
    assert switch_count(SERPENTINE) == 15


def test_switch_count_ignores_sbends_and_straights():
    """S-bends hand back the original heading, so they are not turns (Aug-6 exploit)."""
    assert switch_count([4, 29, 30, 4]) == 0
    assert switch_count([0, 0, 0]) == 0


def test_each_family_classifies_to_its_own_index():
    names = [f[0] for f in FAMILIES]
    assert names == ["oval", "spiral", "out_and_back", "winding", "serpentine"]
    assert FAMILY_N == 5
    assert classify_family(OVAL) == 0
    assert classify_family(SPIRAL) == 1
    assert classify_family(OUT_AND_BACK) == 2
    assert classify_family(WINDING) == 3
    assert classify_family(SERPENTINE) == 4


def test_classify_returns_none_when_no_family_fits():
    """11 turns with no alternation belongs to nothing: too many for oval, too few
    switches for winding. An honest None beats forcing it into a bin."""
    assert classify_family([4] * 11) is None


def test_family_match_is_full_inside_the_band():
    assert family_match(WINDING, 3, 2.0, 2.0) == pytest.approx(1.0)
    assert family_match(OVAL, 0, 2.0, 2.0) == pytest.approx(1.0)


def test_family_match_falls_off_gradually_outside_the_band():
    """Graded, not pass/fail: partial credit for getting nearer, or the target is
    never discovered (the campaign's every-leg-needs-a-ramp rule)."""
    near = family_match([4, 4, 4, 0, 3, 3, 0, 4, 4, 4, 0, 3, 3], 3, 2.0, 2.0)      # 10 turns, 3 switches
    far = family_match([4, 4, 0, 3, 3, 0, 4, 4], 3, 2.0, 2.0)                       # 6 turns, 2 switches
    assert 0.0 < far < near <= 1.0


def test_family_match_penalises_the_wrong_family():
    """The core inversion: with an oval requested, a winding build must score worse."""
    assert family_match(OVAL, 0, 2.0, 2.0) > family_match(WINDING, 0, 2.0, 2.0)
    assert family_match(WINDING, 3, 2.0, 2.0) > family_match(OVAL, 3, 2.0, 2.0)


# --------------------------------------------- the seed as an observation input
# The agent must SEE which family it was asked for, or it cannot condition on it.
# Discrete is used because SB3 one-hot-encodes it automatically, exactly like the
# existing current_direction / last_piece_type fields.

def test_observation_space_exposes_the_target_family():
    from openrct2_gym.envs.obs_config import TARGET_FAMILY_N, make_observation_space
    space = make_observation_space()
    assert TARGET_FAMILY_N == FAMILY_N
    assert "target_family" in space.spaces
    assert space["target_family"].n == FAMILY_N


def test_env_reports_the_target_family_in_its_observation(monkeypatch):
    import openrct2_gym.envs.openrct2_env as oe_mod
    from openrct2_gym.envs.openrct2_env import OpenRCT2Env
    from openrct2_gym.tests.test_env_smoke import FakeAPI

    monkeypatch.setattr(oe_mod, "APIController", FakeAPI)
    env = OpenRCT2Env(verbose=0)
    assert env.target_family == 0                      # inert default
    env.target_family = 3                              # wrapper sets this before reset
    obs, _ = env.reset()
    assert obs["target_family"] == 3
    assert env.observation_space.contains(obs)


def test_feature_extractor_consumes_the_target_family():
    """A Discrete field the extractor does not read is dead weight -- the policy
    would never see the seed."""
    from openrct2_gym.envs.feature_extractor import BuildHistoryExtractor
    from openrct2_gym.envs.obs_config import make_observation_space
    ex = BuildHistoryExtractor(make_observation_space())
    assert "target_family" in ex._cat_keys


# --- classify_counts: the same bands, from counts the env already reports ------------
# The Phase-6 variety exploration floor decides from per-episode telemetry
# (turn_count / switch_count), not from the action list, so the band logic has to be
# reachable without re-deriving turn directions. It must agree with classify_family
# exactly -- two copies of the bands is precisely how this project has shipped a
# mis-specified footprint four times.

def test_classify_counts_matches_classify_family_on_every_family():
    import itertools
    from openrct2_gym.envs.footprint import (
        FAMILIES, classify_counts, classify_family, turn_directions,
    )
    # exhaustive over the interesting range, via real action sequences
    for turns in range(0, 20):
        for switches in range(0, turns if turns else 1):
            actions = _actions_with(turns, switches)
            dirs = turn_directions(actions)
            sw = sum(1 for x, y in zip(dirs, dirs[1:]) if x != y)
            assert classify_counts(len(dirs), sw) == classify_family(actions), (
                "disagreement at turns=%d switches=%d" % (len(dirs), sw))


def test_classify_counts_returns_none_for_a_gap_in_the_bands():
    from openrct2_gym.envs.footprint import classify_counts
    # 11 turns with no alternation belongs to no family (the docstring's own example)
    assert classify_counts(11, 0) is None


def test_classify_counts_oval_and_open_ended_serpentine():
    from openrct2_gym.envs.footprint import classify_counts
    assert classify_counts(0, 0) == 0      # nothing built yet is trivially oval-shaped
    assert classify_counts(5, 0) == 0
    assert classify_counts(8, 2) == 2      # out_and_back
    assert classify_counts(12, 4) == 3     # winding
    assert classify_counts(40, 30) == 4    # serpentine band is open above


def _actions_with(turns, switches):
    """Action list with exactly `turns` turn pieces and `switches` direction changes."""
    from openrct2_gym.envs.track_pieces import (
        LEFT_TURN_ACTIONS as L, RIGHT_TURN_ACTIONS as R,
    )
    left, right = sorted(L)[0], sorted(R)[0]
    out, cur, remaining = [], left, switches
    for i in range(turns):
        out.append(cur)
        if remaining > 0 and i < turns - 1:
            cur = right if cur == left else left
            remaining -= 1
    return out


# --- behaviour descriptor cells for the diversity reward ------------------------------
# Entropy-based exploration was measured to DELAY the collapse, not prevent it: a fixed
# ent_coef has an equilibrium that falls as the policy converges, so the floor held
# ~0.95 nats for two days and then leaked back to 0.74 with the guard pinned at max
# boost. A diversity term instead makes difference part of the OBJECTIVE, which does not
# decay with convergence. It needs a behaviour descriptor: reuse the family bands'
# own boundaries so the reward and every family metric stay one source of truth, and so
# a build that gains its first direction switch moves cell even if no family matches.

def test_descriptor_cell_uses_the_family_band_boundaries():
    from openrct2_gym.envs.footprint import descriptor_cell
    assert descriptor_cell(0, 0) == (0, 0)     # oval territory
    assert descriptor_cell(5, 0) == (0, 0)     # last turn count still in the low band
    assert descriptor_cell(6, 0) == (1, 0)     # spiral territory
    assert descriptor_cell(8, 2) == (1, 1)     # out_and_back
    assert descriptor_cell(12, 4) == (2, 2)    # winding
    assert descriptor_cell(20, 9) == (3, 3)    # serpentine, both bands open-ended


def test_descriptor_cell_separates_the_first_direction_switch():
    """The measured gap: 6,191 of 6,193 unaided builds had ZERO switches. Gaining one
    switch must change cell even when turn count does not, so the diversity reward pays
    for the missing skill directly rather than only for a completed family."""
    from openrct2_gym.envs.footprint import descriptor_cell
    assert descriptor_cell(4, 0) != descriptor_cell(4, 1)


def test_descriptor_cell_is_defined_for_every_family_exemplar():
    from openrct2_gym.envs.footprint import FAMILIES, descriptor_cell
    for _, tlo, thi, slo, shi in FAMILIES:
        t = tlo if thi is None else (tlo + thi) // 2
        sw = slo if shi is None else (slo + shi) // 2
        cell = descriptor_cell(t, sw)
        assert isinstance(cell, tuple) and len(cell) == 2
        assert all(isinstance(x, int) and x >= 0 for x in cell)
