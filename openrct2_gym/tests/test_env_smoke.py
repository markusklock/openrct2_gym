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


class BatchedResetAPI(FakeAPI):
    """FakeAPI that supports the one-round-trip resetEpisode endpoint."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.reset_episode_calls = 0
        self.place_calls = 0
        self.ride_id = None

    def place_track_piece(self, *a, **k):
        self.place_calls += 1
        return super().place_track_piece(*a, **k)

    def reset_episode(self, station_length=6, start=(61, 66, 14), start_direction=0, ride_type=52):
        self.reset_episode_calls += 1
        self.ride_id = 7
        return {
            "rideId": 7,
            "finalEndpoint": {"x": 61, "y": 70, "z": 14, "direction": 0},
            "validNextPieces": {"validPieces": [0, 16, 17]},
        }


def test_reset_uses_batched_reset_episode_when_available(monkeypatch):
    """When the plugin offers resetEpisode, reset() takes the one-round-trip path: no per-piece
    station placement, head + valid-piece cache taken from the payload, station prefix preserved."""
    monkeypatch.setattr(oe_mod, "APIController", BatchedResetAPI)
    env = OpenRCT2Env(verbose=0)
    env.skip_ride_testing = True

    api = env.api_controller
    api.reset_episode_calls = 0
    api.place_calls = 0

    env.reset()

    assert api.reset_episode_calls == 1                       # exactly one batched round-trip
    assert api.place_calls == 0                               # NO per-piece station build
    assert env.current_position == [61, 70, 14]               # head from finalEndpoint
    assert env.current_direction == 0
    assert env.track_pieces == [0] * env.station_length       # station prefix preserved (index parity)
    assert env.track_builder.valid_track_types == [0, 16, 17]  # cache seeded from payload
    assert env.api_controller.ride_id == 7


def test_reset_falls_back_to_legacy_build_without_reset_episode(monkeypatch):
    """Old plugin (no resetEpisode): reset() falls back to delete+create+station build and still
    produces a valid post-station head and station prefix."""
    monkeypatch.setattr(oe_mod, "APIController", FakeAPI)  # stock FakeAPI has no reset_episode
    env = OpenRCT2Env(verbose=0)
    env.skip_ride_testing = True
    assert not hasattr(env.api_controller, "reset_episode")

    env.reset()

    assert len(env.api_controller._stack) == env.station_length      # built piece-by-piece
    assert env.track_pieces == [0] * env.station_length
    assert env.current_position == [61, 66 + env.station_length, 14]  # FakeAPI advances +1 in y/piece
    assert env.current_direction == 0


class FailingResetAPI(FakeAPI):
    """resetEpisode present but failing (e.g. partially applied) -> must fall back cleanly."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.calls = []

    def reset_episode(self, **k):
        self.calls.append("reset_episode")
        return None

    def delete_all_rides(self):
        self.calls.append("delete_all_rides")
        return super().delete_all_rides()

    def create_ride(self):
        self.calls.append("create_ride")
        return super().create_ride()

    def place_track_piece(self, *a, **k):
        self.calls.append("place")
        return super().place_track_piece(*a, **k)


def test_reset_falls_back_cleanly_when_reset_episode_fails(monkeypatch):
    """A failed batched reset (possibly partially applied) must restart from a clean slate:
    delete_all_rides BEFORE create_ride, then the station build."""
    monkeypatch.setattr(oe_mod, "APIController", FailingResetAPI)
    env = OpenRCT2Env(verbose=0)
    env.skip_ride_testing = True

    env.api_controller.calls.clear()
    env.reset()
    calls = env.api_controller.calls

    assert calls[0] == "reset_episode"                                    # tried batched first
    assert "delete_all_rides" in calls and "create_ride" in calls
    assert calls.index("delete_all_rides") < calls.index("create_ride")   # clean slate
    assert calls.count("place") == env.station_length


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


class TimeoutFlagAPI(FakeAPI):
    """FakeAPI that can fail one placement with the timed-out flag set (the server may
    have APPLIED the piece even though the client saw a timeout)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.last_request_timed_out = False
        self.fail_next_with_timeout = False

    def place_track_piece(self, x, y, z, direction, track_type, has_chain=False):
        if self.fail_next_with_timeout and track_type not in (1, 2, 3):
            self.fail_next_with_timeout = False
            self.last_request_timed_out = True
            return {"success": False, "error": "Request timeout after 3 attempts"}
        return super().place_track_piece(x, y, z, direction, track_type, has_chain)


def test_timed_out_placement_sacrifices_the_episode(monkeypatch):
    """A timed-out placement may have been APPLIED server-side (placements are not
    idempotent): the client head is possibly desynced from the real track, so every
    further transition would be corrupt. The episode must truncate; the next
    resetEpisode (retry-safe) rebuilds both sides from a clean slate."""
    monkeypatch.setattr(oe_mod, "APIController", TimeoutFlagAPI)
    env = OpenRCT2Env(verbose=0)
    env.skip_ride_testing = True
    env.reset()
    _, _, _, truncated, _ = env.step(0)
    assert not truncated
    env.api_controller.fail_next_with_timeout = True
    _, reward, terminated, truncated, _ = env.step(0)
    assert truncated and not terminated                  # sacrificed, not rewarded/punished extra
    assert env._connection_suspect is True

    env.api_controller.last_request_timed_out = False
    env.reset()                                          # clean slate
    assert env._connection_suspect is False
    _, _, _, truncated, _ = env.step(0)
    assert not truncated


def test_all_invalid_mask_falls_back_and_sacrifices_episode(monkeypatch):
    """An all-False mask makes MaskablePPO sample uniformly over KNOWN-INVALID actions
    (masked logits all equal) -- a silent failure grind. With no valid pieces and no
    history to remove, allow everything for the step and sacrifice the episode."""
    monkeypatch.setattr(oe_mod, "APIController", FakeAPI)
    env = OpenRCT2Env(verbose=0)
    env.skip_ride_testing = True
    env.reset()
    env.track_builder.valid_track_types = []             # API failure state: nothing valid
    mask = env.valid_action_mask()
    assert mask.all()                                    # degenerate-uniform avoided
    assert env._connection_suspect is True               # ...and the episode is sacrificed
