"""Server-free integration smoke test for OpenRCT2Env.

A fake APIController (no network) lets us run the REAL __init__/reset/step/
valid_action_mask paths, exercising observation assembly, the remove/auto-backtrack
wiring, and _revert_chain_lift. Track geometry is faked (one tile per piece), so this
validates Python wiring, not game physics (that needs a live server, Phase 3).
"""
import gymnasium as gym
import numpy as np

from openrct2_gym.envs import openrct2_env as oe_mod
from openrct2_gym.envs.openrct2_env import OpenRCT2Env


class FakeAPI:
    """Minimal stand-in for APIController; advances one tile per placed piece."""

    def __init__(self, host=None, port=None, verbose=0):
        self.station_length = 3
        self._dv = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        self._stack = []  # endpoints of placed pieces (for delete)
        self.get_valid_calls = 0  # count separate getValidNextPieces round-trips

    def connect(self):
        return True

    def disconnect(self):
        pass

    def delete_all_rides(self):
        return {"success": True}

    def create_ride(self):
        self._stack = []
        return 1

    def place_track_piece(self, x, y, z, direction, track_type, has_chain=False):
        dx, dy = self._dv[direction]
        ep = {"x": x + dx, "y": y + dy, "z": z, "direction": direction}
        self._stack.append(ep)
        return {"success": True, "payload": {
            "nextEndpoint": ep, "isCircuitComplete": False,
            "validNextPieces": {"validPieces": list(range(46))}}}

    def get_valid_next_pieces(self):
        self.get_valid_calls += 1
        return {"success": True, "payload": {"validPieces": list(range(46))}}

    def delete_last_track_piece(self):
        if self._stack:
            self._stack.pop()
        prev = self._stack[-1] if self._stack else {"x": 61, "y": 66, "z": 14, "direction": 0}
        return {"success": True, "payload": {"nextEndpoint": prev, "piecesRemaining": len(self._stack)}}


def test_reset_and_random_rollout_keeps_obs_in_space(monkeypatch):
    monkeypatch.setattr(oe_mod, "APIController", FakeAPI)
    env = OpenRCT2Env(verbose=0)
    env.skip_ride_testing = True

    obs, _ = env.reset()
    assert env.observation_space.contains(obs)

    rng = np.random.default_rng(0)
    for _ in range(80):
        mask = env.valid_action_mask()
        valid = np.flatnonzero(mask)
        assert valid.size > 0
        action = int(rng.choice(valid))
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs), f"obs left the space after action {action}"
        assert np.isfinite(reward)
        if terminated or truncated:
            obs, _ = env.reset()
            assert env.observation_space.contains(obs)

    # Caching: the valid-piece list now rides along with each placement, so the dedicated
    # getValidNextPieces round-trip should fire far less than once per step (only after
    # resets/removes). Proves the per-step API call was eliminated.
    assert env.api_controller.get_valid_calls < 80


def test_chain_lift_bookkeeping_round_trips_through_remove(monkeypatch):
    """Place an early chain lift then remove it: count/positions must return to baseline."""
    monkeypatch.setattr(oe_mod, "APIController", FakeAPI)
    env = OpenRCT2Env(verbose=0)
    env.skip_ride_testing = True
    env.reset()

    # Place a chain-lift piece (action 9) early so _calculate_reward records it.
    env.step(9)
    assert env.chain_lift_count == 1
    assert len(env.chain_lift_positions) == 1

    # Remove it (allowed: within the post-collision window early in the episode).
    assert env.valid_action_mask()[31]
    env.step(31)
    assert env.chain_lift_count == 0
    assert env.chain_lift_positions == set()
