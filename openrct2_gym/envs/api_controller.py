import socket
import json
import random
import time

class APIController:
    # Consecutive fully-timed-out requests before raising: a HUNG game instance otherwise
    # drags the whole synchronized SubprocVecEnv fleet to ~1 step / 7s forever with no error
    # surfaced. Raising kills this worker loudly; the run ends with checkpoints intact.
    MAX_CONSECUTIVE_TIMEOUTS = 10

    def __init__(self, host="localhost", port=8080, verbose=1):
        self.host = host
        self.port = port
        self.verbose = verbose  # 0=silent, 1=important only, 2=detailed
        self.sock = None
        self.ride_id = None
        self.station_length = 6
        # Protocol-hygiene state: a request that timed out may have been APPLIED server-side
        # (placements are not idempotent), so callers must be able to see it and sacrifice
        # the episode rather than continue from a possibly-desynced head.
        self.last_request_timed_out = False
        self.consecutive_timeouts = 0
        
    def connect(self):
        try:
            self.sock = socket.create_connection((self.host, self.port), timeout=5.0)
            self.sock.settimeout(2.0)  # Set timeout for all socket operations
            if self.verbose >= 1:  # Print for important or detailed mode
                print(f"Connected to OpenRCT2 API server at {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"Failed to connect to API server at {self.host}:{self.port}: {e}")
            print("Make sure the OpenRCT2 API server is running!")
            return False
    
    def disconnect(self):
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

    def _reconnect(self):
        """Close and reopen socket connection with brief delay."""
        self.disconnect()
        time.sleep(0.1)  # Brief delay before reconnect
        return self.connect()

    def send_request(self, request, max_retries=3, base_timeout=1.0):
        """
        Send request with retry logic and proper resource cleanup.

        Args:
            request: The request dictionary to send
            max_retries: Maximum number of retry attempts (default: 3)
            base_timeout: Base timeout in seconds, doubles each retry (default: 1.0). Raised
                from 0.5 so a transient slow response under heavy (~20x) parallelism doesn't
                trip a spurious timeout -> reconnect, which would re-establish the socket
                mid-episode and turn that worker into a straggler under the synchronized barrier.

        Returns:
            Response dictionary with 'success' key
        """
        endpoint = request.get('endpoint', 'unknown')
        self.last_request_timed_out = False

        for attempt in range(max_retries):
            # Ensure we have a connection
            if not self.sock:
                if not self.connect():
                    if attempt < max_retries - 1:
                        time.sleep(0.1 * (attempt + 1))
                        continue
                    return {"success": False, "error": "Not connected to API server"}

            try:
                # Set timeout with exponential backoff (default base 1.0s): 1s, 2s, 4s
                timeout = base_timeout * (2 ** attempt)
                self.sock.settimeout(timeout)

                # Send request
                message = json.dumps(request) + "\n"
                self.sock.sendall(message.encode("utf-8"))

                # Receive response with proper resource cleanup
                file_obj = self.sock.makefile("r")
                try:
                    line = file_obj.readline()
                    if not line:
                        raise socket.timeout("Empty response from server")
                    resp = json.loads(line)
                    self.consecutive_timeouts = 0
                    return resp
                finally:
                    file_obj.close()  # Fix: Always close the file object

            except socket.timeout:
                if self.verbose >= 2:
                    print(f"Timeout (attempt {attempt + 1}/{max_retries}) for endpoint: {endpoint}")
                if attempt < max_retries - 1:
                    self._reconnect()
                    continue
                # Final attempt: the socket has an ABANDONED in-flight request -- keeping it
                # would make every later response off-by-one (the next endpoint would read
                # THIS request's late reply). Drop it; the next request reconnects clean.
                self.disconnect()
                self.last_request_timed_out = True
                self.consecutive_timeouts += 1
                if self.consecutive_timeouts >= self.MAX_CONSECUTIVE_TIMEOUTS:
                    raise RuntimeError(
                        f"OpenRCT2 instance on port {self.port} unresponsive: "
                        f"{self.consecutive_timeouts} consecutive request timeouts "
                        f"(endpoint {endpoint})")
                return {"success": False, "error": f"Request timeout after {max_retries} attempts"}

            except (ConnectionError, BrokenPipeError, ConnectionResetError) as e:
                if self.verbose >= 2:
                    print(f"Connection error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    self._reconnect()
                    continue
                self.disconnect()   # stream state unknown -- never reuse
                return {"success": False, "error": f"Connection error: {e}"}

            except json.JSONDecodeError as e:
                # Non-JSON line: the stream framing is unknown (torn/foreign reply); the
                # socket cannot be trusted for even one more request.
                self.disconnect()
                return {"success": False, "error": f"Invalid JSON response: {e}"}

            except Exception as e:
                if self.verbose >= 2:
                    print(f"Unexpected error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    self._reconnect()
                    continue
                self.disconnect()
                return {"success": False, "error": f"Request failed: {e}"}

        return {"success": False, "error": "Max retries exceeded"}
    
    def create_ride(self, ride_type=52, ride_object=0):
        req = {
            "endpoint": "createRide",
            "params": {
                "rideType": ride_type,
                "rideObject": ride_object,
                "entranceObject": 0,
                "colour1": 0,
                "colour2": 1
            }
        }
        resp = self.send_request(req)
        if resp.get("success"):
            self.ride_id = resp["payload"]["rideId"]
            if self.verbose >= 2:  # Only print in detailed mode
                print(f"Created ride with rideId: {self.ride_id}")
            return self.ride_id
        else:
            print(f"createRide failed: {resp}")
            return None

    def reset_episode(self, station_length=6, start=(61, 66, 14), start_direction=0, ride_type=52):
        """One-round-trip episode reset: server-side demolish-all + create-ride + build-station.

        Replaces the legacy delete_all_rides() + create_ride() + N*place_track_piece() sequence
        (8 round-trips) with a single request. Under heavy parallelism every synchronized
        vector-step waits on the slowest worker's reset, so collapsing it is a large win.

        Returns the payload dict {rideId, finalEndpoint, validNextPieces} on success (and sets
        self.ride_id), or None on failure / when the plugin lacks the endpoint, so the caller
        can fall back to the multi-call path.
        """
        req = {
            "endpoint": "resetEpisode",
            "params": {
                "stationLength": station_length,
                "startX": start[0],
                "startY": start[1],
                "startZ": start[2],
                "startDir": start_direction,
                "rideType": ride_type,
            },
        }
        # resetEpisode runs ~8 game actions server-side (demolish + create + N station
        # placements), so it gets a more generous base timeout than a per-step request. A
        # timeout/retry is safe: the handler demolishes all rides first, so a re-run just
        # rebuilds from a clean slate.
        resp = self.send_request(req, base_timeout=2.0)
        if resp.get("success"):
            payload = resp.get("payload", {})
            ride_id = payload.get("rideId")
            if ride_id is not None:
                self.ride_id = ride_id
                return payload
        if self.verbose >= 1:
            print(f"resetEpisode failed or unsupported: {resp.get('error', resp)}")
        return None

    def place_track_piece(self, tile_x, tile_y, tile_z, direction, track_type, has_chain_lift=False):
        if self.ride_id is None:
            return {"success": False, "error": "No ride created"}
            
        req = {
            "endpoint": "placeTrackPiece",
            "params": {
                "tileCoordinateX": tile_x,
                "tileCoordinateY": tile_y,
                "tileCoordinateZ": tile_z,
                "direction": direction,
                "ride": self.ride_id,
                "trackType": track_type,
                "rideType": 52,
                "brakeSpeed": 0,
                "colour": 0,
                "seatRotation": 0,
                "trackPlaceFlags": 0,
                "isFromTrackDesign": True,
                "hasChainLift": has_chain_lift
            }
        }
        return self.send_request(req)
    
    def get_valid_next_pieces(self):
        if self.ride_id is None:
            return {"success": False, "error": "No ride created"}
            
        req = {
            "endpoint": "getValidNextPieces",
            "params": {
                "rideId": self.ride_id
            }
        }
        return self.send_request(req)
    
    def delete_last_track_piece(self):
        """
        Deletes the most recently placed track piece using the new API endpoint.
        Returns the response including the new endpoint position after deletion.
        """
        if self.ride_id is None:
            return {"success": False, "error": "No ride created"}
            
        req = {
            "endpoint": "deleteLastTrackPiece",
            "params": {
                "rideId": self.ride_id
            }
        }
        return self.send_request(req)
    
    def demolish_ride(self):
        if self.ride_id is None:
            return {"success": False, "error": "No ride to demolish"}
            
        req = {
            "endpoint": "demolishRide",
            "params": {
                "rideId": self.ride_id
            }
        }
        resp = self.send_request(req)
        if resp.get("success"):
            self.ride_id = None
        return resp
    
    def delete_all_rides(self):
        """
        Deletes all rides from the park using the deleteAllRides endpoint.
        """
        req = {
            "endpoint": "deleteAllRides"
        }
        resp = self.send_request(req)
        if resp.get("success"):
            self.ride_id = None  # Clear our ride_id since all rides are deleted
        return resp
    
    def place_entrance_exit(self):
        """
        Places entrance and exit for the ride's station.
        Must be called after placing station pieces.
        Returns the positions of the placed entrance and exit.
        """
        if self.ride_id is None:
            return {"success": False, "error": "No ride created"}
            
        req = {
            "endpoint": "placeEntranceExit",
            "params": {
                "rideId": self.ride_id
            }
        }
        return self.send_request(req)
    
    def get_ride_status(self):
        if self.ride_id is None:
            return {"success": False, "error": "No ride created"}
            
        req = {
            "endpoint": "getRideStatus",
            "params": {
                "rideId": self.ride_id
            }
        }
        return self.send_request(req)
    
    def set_ride_status(self, status):
        if self.ride_id is None:
            return {"success": False, "error": "No ride created"}
            
        req = {
            "endpoint": "setRideStatus",
            "params": {
                "rideId": self.ride_id,
                "status": status
            }
        }
        return self.send_request(req)
    
    def get_ride_ratings(self):
        if self.ride_id is None:
            return {"success": False, "error": "No ride created"}

        req = {
            "endpoint": "getRideRatings",
            "params": {
                "rideId": self.ride_id
            }
        }
        return self.send_request(req)

    def start_ride_test(self):
        """Start ride test mode using the correct API endpoint."""
        if self.ride_id is None:
            return {"success": False, "error": "No ride created"}

        req = {
            "endpoint": "startRideTest",
            "params": {
                "rideId": self.ride_id
            }
        }
        return self.send_request(req)

    def get_ride_stats(self):
        """Get ride statistics using the correct API endpoint."""
        if self.ride_id is None:
            return {"success": False, "error": "No ride created"}

        req = {
            "endpoint": "getRideStats",
            "params": {
                "rideId": self.ride_id
            }
        }
        return self.send_request(req)