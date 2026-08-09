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
