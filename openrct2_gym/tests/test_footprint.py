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
    assert family_match(WINDING, 3) == pytest.approx(1.0)
    assert family_match(OVAL, 0) == pytest.approx(1.0)


def test_family_match_falls_off_gradually_outside_the_band():
    """Graded, not pass/fail: partial credit for getting nearer, or the target is
    never discovered (the campaign's every-leg-needs-a-ramp rule)."""
    near = family_match([4, 4, 4, 0, 3, 3, 0, 4, 4, 4, 0, 3, 3], 3)      # 10 turns, 3 switches
    far = family_match([4, 4, 0, 3, 3, 0, 4, 4], 3)                       # 6 turns, 2 switches
    assert 0.0 < far < near <= 1.0


def test_family_match_penalises_the_wrong_family():
    """The core inversion: with an oval requested, a winding build must score worse."""
    assert family_match(OVAL, 0) > family_match(WINDING, 0)
    assert family_match(WINDING, 3) > family_match(OVAL, 3)


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
