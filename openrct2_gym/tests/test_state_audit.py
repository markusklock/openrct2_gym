"""Phase 0a state-audit tests.

These cover pre-existing internal-state bugs that corrupt reward signals today and
would feed garbage into the redesigned observation:

  * energy must be computed from ``track_builder.history`` (agent pieces + their
    endpoint heights), NOT from the ``track_pieces``/``height_history`` pair which is
    index-shifted by the station prefix and is not removal-safe.
  * ``position_history`` must retain enough samples for the observation's distance trend.
  * chain-lift bookkeeping must be reverted on piece removal.

All tests build the env via ``__new__`` so no live OpenRCT2 API connection is needed.
"""
from types import SimpleNamespace

import pytest

from openrct2_gym.envs.openrct2_env import OpenRCT2Env


def _bare_env():
    """An OpenRCT2Env instance without __init__ (no API connection)."""
    return OpenRCT2Env.__new__(OpenRCT2Env)


# --------------------------------------------------------------------------- energy


def test_energy_computed_from_history_chain_then_drop():
    env = _bare_env()
    env.track_builder = SimpleNamespace(history=[
        {"action": 9, "position": [0, 0, 14], "next_position": [1, 0, 15]},  # chain lift up
        {"action": 6, "position": [1, 0, 15], "next_position": [2, 0, 14]},  # down 25
    ])
    # last_height starts at STATION_HEIGHT (14):
    #  piece 9 (chain): +50 energy; delta +1 but chain => no uphill cost; -2 friction => 48
    #  piece 6 (drop):  delta -1 => +3 gravity; -2 friction => 49
    assert env._calculate_estimated_energy() == pytest.approx(49.0)


def test_energy_ignores_station_prefix_in_track_pieces():
    """The station/height_history misalignment bug: energy must depend ONLY on history."""
    env = _bare_env()
    env.track_builder = SimpleNamespace(history=[
        {"action": 9, "position": [0, 0, 14], "next_position": [1, 0, 15]},
        {"action": 6, "position": [1, 0, 15], "next_position": [2, 0, 14]},
    ])
    # Deliberately set the legacy fields to garbage. The corrected impl must ignore them.
    env.track_pieces = [0, 0, 0, 0, 0, 0, 9, 6]   # 6 station pieces + 2 agent pieces
    env.height_history = [15, 14]
    assert env._calculate_estimated_energy() == pytest.approx(49.0)


def test_energy_is_removal_safe_pure_function_of_history():
    env = _bare_env()
    p1 = {"action": 9, "position": [0, 0, 14], "next_position": [1, 0, 15]}
    p2 = {"action": 6, "position": [1, 0, 15], "next_position": [2, 0, 14]}

    env.track_builder = SimpleNamespace(history=[p1, p2])
    full = env._calculate_estimated_energy()

    env.track_builder.history = [p1]            # simulate a remove
    after_remove = env._calculate_estimated_energy()

    env.track_builder.history = [p1, p2]        # re-place the same piece
    after_replace = env._calculate_estimated_energy()

    assert full == pytest.approx(49.0)
    assert after_remove == pytest.approx(48.0)
    assert after_replace == pytest.approx(full)


def test_energy_margin_uses_corrected_energy():
    env = _bare_env()
    env.track_builder = SimpleNamespace(history=[
        {"action": 9, "position": [0, 0, 14], "next_position": [1, 0, 15]},
        {"action": 6, "position": [1, 0, 15], "next_position": [2, 0, 14]},
    ])
    env.current_position = [2, 0, 14]
    env.goal_position = [2, 0, 14]              # distance 0, no height deficit
    # margin = energy - (distance*0.5 + height_deficit*UPHILL_COST) = 49 - 0
    assert env._calculate_energy_margin() == pytest.approx(49.0)


# ----------------------------------------------------- position-history window (obs)


def test_position_history_window_supports_distance_trend():
    # The observation's distance-trend signal compares position_history[0] to the
    # current head, so the window must retain at least two samples.
    assert OpenRCT2Env.POSITION_HISTORY_MAXLEN >= 2


# ----------------------------------------------------------------- chain-lift revert


def test_revert_chain_lift_decrements_count_and_discards_position():
    env = _bare_env()
    env.chain_lift_count = 2
    env.chain_lift_positions = {(1, 0, 15), (5, 0, 18)}
    env.max_chain_lifts = 15

    env._revert_chain_lift({"action": 9, "next_position": [1, 0, 15]})

    assert env.chain_lift_count == 1
    assert (1, 0, 15) not in env.chain_lift_positions


def test_revert_chain_lift_ignores_non_chain_and_unrecorded():
    env = _bare_env()
    env.chain_lift_count = 1
    env.chain_lift_positions = {(5, 0, 18)}
    env.max_chain_lifts = 15

    env._revert_chain_lift({"action": 0, "next_position": [9, 0, 14]})   # not a chain piece
    env._revert_chain_lift({"action": 9, "next_position": [1, 0, 15]})   # chain, but never recorded

    assert env.chain_lift_count == 1
    assert env.chain_lift_positions == {(5, 0, 18)}


def test_revert_chain_lift_does_not_go_negative():
    env = _bare_env()
    env.chain_lift_count = 0
    env.chain_lift_positions = set()
    env.max_chain_lifts = 15

    env._revert_chain_lift({"action": 10, "next_position": [1, 0, 15]})

    assert env.chain_lift_count == 0
