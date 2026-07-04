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
    and counts separate get_valid_next_pieces calls. ``delete_valid`` controls whether the
    delete response carries validNextPieces (new plugin) or omits them (old plugin)."""

    def __init__(self, valid_track_types, delete_valid=None):
        self.valid = valid_track_types
        self.delete_valid = delete_valid
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
        payload = {"nextEndpoint": {"x": 60, "y": 66, "z": 14, "direction": 0}, "piecesRemaining": 1}
        if self.delete_valid is not None:
            payload["validNextPieces"] = {"validPieces": self.delete_valid}
        return {"success": True, "payload": payload}


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


def test_remove_without_delete_validpieces_refetches():
    """Old plugin: deleteLastTrackPiece omits validNextPieces, so the cache is invalidated and
    the next query must re-fetch via getValidNextPieces (the documented fallback)."""
    api = _CountingAPI([0, 16])               # delete_valid=None -> delete omits validNextPieces
    tb = APITrackBuilder(api)
    tb.take_action(0, [61, 66, 14], 1)        # place caches
    tb.take_action(31, [62, 66, 14], 1)       # remove -> no validNextPieces -> cache invalid
    tb.get_valid_actions()
    assert api.get_valid_calls == 1           # had to re-query after the remove


def test_remove_with_delete_validpieces_uses_cache():
    """New plugin: deleteLastTrackPiece returns validNextPieces, so the client serves the next
    query straight from that cache -- no separate getValidNextPieces round-trip after a remove."""
    api = _CountingAPI([0, 16], delete_valid=[0, 17])
    tb = APITrackBuilder(api)
    tb.take_action(0, [61, 66, 14], 1)        # place caches [0, 16]
    tb.take_action(31, [62, 66, 14], 1)       # remove returns [0, 17] -> cache refreshed, not cleared
    actions = tb.get_valid_actions()
    assert api.get_valid_calls == 0           # served from the delete-response cache
    assert {0, 2}.issubset(set(actions))      # reflects delete payload (type 0->action 0, 17->action 2)


class _ZRecordingAPI:
    """Fake API recording the z each placement was issued at."""

    def __init__(self):
        self.placed = []   # (track_type, z)

    def place_track_piece(self, x, y, z, d, track_type, has_chain=False):
        self.placed.append((track_type, z))
        return {"success": True, "payload": {
            "nextEndpoint": {"x": x + 1, "y": y, "z": z, "direction": d},
            "isCircuitComplete": False,
            "validNextPieces": {"validPieces": []},
        }}


def test_descending_pieces_place_at_base_z_below_entry():
    """Live-probed plugin contract (port 8080, Jul 2026): placeTrackPiece takes a piece's
    BASE z, but validation requires its TRAIN ENTRY -- which for DESCENDING pieces sits
    ABOVE the base by the piece's drop -- to continue from the previous piece's end.
    Passing the head z unadjusted made EVERY descent fail ("train entry would be ...,
    previous piece ends at ..."), silently walling off drops for entire training runs."""
    api = _ZRecordingAPI()
    builder = APITrackBuilder(api)
    head_z = 20
    expected_drop = {
        6: 2,    # Down25          (type 10, spans 2 z)
        8: 8,    # Down60          (type 11, spans 8 z)
        12: 1,   # FlatToDown25    (type 12, spans 1 z)
        14: 1,   # Down25ToFlat    (type 15, spans 1 z)
        27: 4,   # Down25ToDown60  (type 13, spans 4 z)
        28: 4,   # Down60ToDown25  (type 14, spans 4 z)
    }
    for action, dz in expected_drop.items():
        ok, _, _ = builder.take_action(action, [10, 10, head_z], 0)
        assert ok
        ttype, z = api.placed[-1]
        assert z == head_z - dz, f"action {action} (type {ttype}) placed at z={z}, want {head_z - dz}"


def test_non_descending_pieces_place_at_entry_z():
    api = _ZRecordingAPI()
    builder = APITrackBuilder(api)
    for action in (0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 13, 15, 17, 21, 25, 26, 29, 30):
        builder.take_action(action, [10, 10, 20], 0)
        _, z = api.placed[-1]
        assert z == 20, f"action {action} shifted z but is not a descent"
