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
