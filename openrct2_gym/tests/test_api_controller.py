"""Protocol-hygiene tests for APIController (server-free, stub sockets).

The request/response stream is line-oriented over one socket: an abandoned in-flight
request MUST poison the socket permanently (every later response would be off-by-one --
the next endpoint reads the previous endpoint's reply, crashing on a missing key or,
worse, silently teleporting the build head). These tests pin the containment contract:
any final-attempt failure drops the socket; timeouts are counted and escalate.
"""
import json

import pytest

from openrct2_gym.envs.api_controller import APIController


class StubSock:
    """Minimal socket stand-in: scripted readline() lines ('' = timeout/empty)."""

    def __init__(self, lines):
        self.lines = list(lines)
        self.closed = False

    def settimeout(self, t):
        pass

    def sendall(self, data):
        pass

    def makefile(self, mode):
        outer = self

        class _F:
            def readline(self):
                return outer.lines.pop(0) if outer.lines else ""

            def close(self):
                pass

        return _F()

    def close(self):
        self.closed = True


def _controller(lines_per_socket):
    """APIController whose connect() installs the next scripted StubSock."""
    ctrl = APIController("localhost", 0, verbose=0)
    sockets = [StubSock(lines) for lines in lines_per_socket]

    def fake_connect():
        if sockets:
            ctrl.sock = sockets.pop(0)
            return True
        ctrl.sock = None
        return False

    ctrl.connect = fake_connect
    fake_connect()
    return ctrl


def test_final_timeout_drops_poisoned_socket():
    ctrl = _controller([[], [], []])                     # every attempt: empty read = timeout
    resp = ctrl.send_request({"endpoint": "placeTrackPiece"})
    assert resp["success"] is False
    assert ctrl.sock is None                             # in-flight request never reusable
    assert ctrl.last_request_timed_out is True
    assert ctrl.consecutive_timeouts == 1


def test_success_resets_timeout_state():
    ok = json.dumps({"success": True, "payload": {}}) + "\n"
    ctrl = _controller([[ok]])
    ctrl.consecutive_timeouts = 5
    ctrl.last_request_timed_out = True
    resp = ctrl.send_request({"endpoint": "getRideStats"})
    assert resp["success"] is True
    assert ctrl.consecutive_timeouts == 0
    assert ctrl.last_request_timed_out is False


def test_repeated_timeouts_escalate_to_hard_error():
    """A hung game instance must fail LOUDLY (worker death -> run ends, checkpoints intact),
    not drag the synchronized 20-worker fleet at 1 step / 7s forever."""
    ctrl = _controller([[], [], []])
    ctrl.consecutive_timeouts = APIController.MAX_CONSECUTIVE_TIMEOUTS - 1
    with pytest.raises(RuntimeError):
        ctrl.send_request({"endpoint": "placeTrackPiece"})
    assert ctrl.sock is None


def test_garbage_response_drops_socket():
    """A non-JSON line means the stream framing is unknown -- the socket cannot be trusted."""
    ctrl = _controller([["this is not json\n"]])
    resp = ctrl.send_request({"endpoint": "getRideStats"})
    assert resp["success"] is False
    assert ctrl.sock is None


def test_set_game_speed_request_shape():
    import json as _json
    ok = _json.dumps({"success": True, "payload": {"speed": 8}}) + "\n"
    ctrl = _controller([[ok]])
    sent = []
    orig_sock = ctrl.sock
    orig_sendall = orig_sock.sendall
    orig_sock.sendall = lambda data: sent.append(_json.loads(data.decode()))
    resp = ctrl.set_game_speed(8)
    assert resp["success"] is True
    assert sent[0] == {"endpoint": "setGameSpeed", "params": {"speed": 8}}


def test_get_ride_measurements_request_shape():
    import json as _json
    ok = _json.dumps({"success": True, "payload": {"numDrops": 3}}) + "\n"
    ctrl = _controller([[ok]])
    ctrl.ride_id = 7
    sent = []
    ctrl.sock.sendall = lambda data: sent.append(_json.loads(data.decode()))
    resp = ctrl.get_ride_measurements()
    assert resp["success"] is True
    assert sent[0] == {"endpoint": "getRideMeasurements", "params": {"rideId": 7}}
    # no ride yet -> honest failure, no request
    ctrl2 = _controller([[ok]])
    ctrl2.ride_id = None
    assert ctrl2.get_ride_measurements()["success"] is False


# ----------------------- ride-object resolution (the htpc balloon-stall bug, Jul-31)
# create_ride/resetEpisode hardcoded rideObject=0 == "first loaded ride object". On the
# laptop that happened to be the wooden coaster trains; on the htpc scenario the object
# scan order put a Balloon Stall at index 0 -- every ride was track-with-no-train, so
# tests never dispatched and NOTHING ever rated (test_ok pinned at 0.00 while
# placements worked perfectly). The object index must be resolved from the live
# instance, never assumed.

def _obj_list_payload():
    return json.dumps({"success": True, "payload": [
        {"index": 0, "identifier": "rct2.ride.balln", "name": "Balloon Stall",
         "rideType": [32, 255, 255]},
        {"index": 4, "identifier": "rct2.ride.arrt1", "name": "Corkscrew Trains",
         "rideType": [19, 255, 255]},
        {"index": 15, "identifier": "rct2.ride.ptct1", "name": "Wooden RC Trains",
         "rideType": [52, 255, 255]},
    ]}) + "\n"


def test_create_ride_resolves_vehicle_object_by_identifier():
    created = json.dumps({"success": True, "payload": {"rideId": 3}}) + "\n"
    ctrl = _controller([[_obj_list_payload(), created]])
    sent = []
    ctrl.sock.sendall = lambda data: sent.append(json.loads(data.decode()))
    assert ctrl.create_ride() == 3
    assert sent[0]["endpoint"] == "listLoadedRideObjects"
    assert sent[1]["endpoint"] == "createRide"
    assert sent[1]["params"]["rideObject"] == 15          # ptct1, not blind index 0
    # resolution is cached: a second create must not re-query the object list
    created2 = json.dumps({"success": True, "payload": {"rideId": 4}}) + "\n"
    ctrl.sock.lines.append(created2)
    sent.clear()
    assert ctrl.create_ride() == 4
    assert sent[0]["endpoint"] == "createRide"
    assert sent[0]["params"]["rideObject"] == 15


def test_create_ride_falls_back_to_ride_type_match():
    objs = json.dumps({"success": True, "payload": [
        {"index": 0, "identifier": "rct2.ride.balln", "name": "Balloon Stall",
         "rideType": [32, 255, 255]},
        {"index": 7, "identifier": "custom.wood.trains", "name": "Some Wooden Trains",
         "rideType": [52, 255, 255]},
    ]}) + "\n"
    created = json.dumps({"success": True, "payload": {"rideId": 1}}) + "\n"
    ctrl = _controller([[objs, created]])
    sent = []
    ctrl.sock.sendall = lambda data: sent.append(json.loads(data.decode()))
    assert ctrl.create_ride() == 1
    assert sent[1]["params"]["rideObject"] == 7           # first rideType-52 entry


def test_create_ride_uses_legacy_default_when_resolution_unavailable():
    """Old plugins without listLoadedRideObjects (or a failing call) must keep the
    pre-fix behavior: rideObject 0, no crash -- and no poisoned-cache (retry next time)."""
    err = json.dumps({"success": False, "error": "Unknown endpoint"}) + "\n"
    created = json.dumps({"success": True, "payload": {"rideId": 9}}) + "\n"
    ctrl = _controller([[err, created]])
    sent = []
    ctrl.sock.sendall = lambda data: sent.append(json.loads(data.decode()))
    assert ctrl.create_ride() == 9
    assert sent[1]["params"]["rideObject"] == 0


def test_reset_episode_carries_resolved_ride_object():
    reset_ok = json.dumps({"success": True, "payload": {
        "rideId": 2, "finalEndpoint": {"x": 55, "y": 66, "z": 14, "direction": 0},
        "validNextPieces": []}}) + "\n"
    ctrl = _controller([[_obj_list_payload(), reset_ok]])
    sent = []
    ctrl.sock.sendall = lambda data: sent.append(json.loads(data.decode()))
    payload = ctrl.reset_episode()
    assert payload["rideId"] == 2
    assert sent[0]["endpoint"] == "listLoadedRideObjects"
    assert sent[1]["endpoint"] == "resetEpisode"
    assert sent[1]["params"]["rideObject"] == 15
