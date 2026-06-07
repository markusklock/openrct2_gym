"""Tests for APITrackBuilder's static action mapping.

The original suite tested a long-gone client-side ``TrackBuilder`` whose geometry is
now computed server-side by the OpenRCT2 API, so those tests are obsolete. Track
geometry now requires a live API and is covered by integration runs; here we cover the
parts that are pure and server-free: the action->track-type catalog and chain-lift set.
``APITrackBuilder.__init__`` only stores its api_controller (no connection), so ``None``
is a valid stand-in.
"""
from openrct2_gym.envs.api_track_builder import APITrackBuilder


def _builder():
    return APITrackBuilder(None)


def test_action_space_is_fully_mapped():
    builder = _builder()
    assert set(builder.action_to_track_type.keys()) == set(range(32))


def test_remove_action_maps_to_sentinel():
    assert _builder().action_to_track_type[31] == -1


def test_chain_lift_actions():
    assert _builder().chain_lift_actions == {9, 10}


def test_history_starts_empty():
    assert _builder().history == []


class _CountingAPI:
    """Fake API that returns validNextPieces in the place response (like the plugin now does)
    and counts separate get_valid_next_pieces calls."""

    def __init__(self, valid_track_types):
        self.valid = valid_track_types
        self.get_valid_calls = 0

    def place_track_piece(self, x, y, z, d, track_type, has_chain=False):
        return {"success": True, "payload": {
            "nextEndpoint": {"x": x + 1, "y": y, "z": z, "direction": d},
            "isCircuitComplete": False,
            "validNextPieces": {"validPieces": self.valid},
        }}

    def get_valid_next_pieces(self):
        self.get_valid_calls += 1
        return {"success": True, "payload": {"validPieces": self.valid}}

    def delete_last_track_piece(self):
        return {"success": True, "payload": {
            "nextEndpoint": {"x": 60, "y": 66, "z": 14, "direction": 0}, "piecesRemaining": 0}}


def test_get_valid_actions_uses_place_response_cache():
    api = _CountingAPI([0, 16, 17])          # track types -> actions 0, 1, 2
    tb = APITrackBuilder(api)
    tb.take_action(0, [61, 66, 14], 1)       # successful place caches validNextPieces
    actions = tb.get_valid_actions()
    assert api.get_valid_calls == 0          # no separate API call: used the cache
    assert {0, 1, 2}.issubset(set(actions))


def test_get_valid_actions_falls_back_to_api_without_cache():
    api = _CountingAPI([0])
    tb = APITrackBuilder(api)
    tb.get_valid_actions()                    # no placement yet -> must query the API
    assert api.get_valid_calls == 1


def test_remove_invalidates_cache():
    api = _CountingAPI([0, 16])
    tb = APITrackBuilder(api)
    tb.take_action(0, [61, 66, 14], 1)        # caches
    tb.take_action(31, [62, 66, 14], 1)       # remove -> endpoint changed -> cache invalid
    tb.get_valid_actions()
    assert api.get_valid_calls == 1           # had to re-query after the remove
