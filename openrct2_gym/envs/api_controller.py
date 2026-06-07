import socket
import json
import random
import time

class APIController:
    def __init__(self, host="localhost", port=8080, verbose=1):
        self.host = host
        self.port = port
        self.verbose = verbose  # 0=silent, 1=important only, 2=detailed
        self.sock = None
        self.ride_id = None
        self.station_length = 6
        
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

    def send_request(self, request, max_retries=3, base_timeout=0.5):
        """
        Send request with retry logic and proper resource cleanup.

        Args:
            request: The request dictionary to send
            max_retries: Maximum number of retry attempts (default: 3)
            base_timeout: Base timeout in seconds, doubles each retry (default: 0.5)

        Returns:
            Response dictionary with 'success' key
        """
        endpoint = request.get('endpoint', 'unknown')

        for attempt in range(max_retries):
            # Ensure we have a connection
            if not self.sock:
                if not self.connect():
                    if attempt < max_retries - 1:
                        time.sleep(0.1 * (attempt + 1))
                        continue
                    return {"success": False, "error": "Not connected to API server"}

            try:
                # Set timeout with exponential backoff: 0.5s, 1s, 2s
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
                    return json.loads(line)
                finally:
                    file_obj.close()  # Fix: Always close the file object

            except socket.timeout:
                if self.verbose >= 2:
                    print(f"Timeout (attempt {attempt + 1}/{max_retries}) for endpoint: {endpoint}")
                if attempt < max_retries - 1:
                    self._reconnect()
                    continue
                return {"success": False, "error": f"Request timeout after {max_retries} attempts"}

            except (ConnectionError, BrokenPipeError, ConnectionResetError) as e:
                if self.verbose >= 2:
                    print(f"Connection error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    self._reconnect()
                    continue
                return {"success": False, "error": f"Connection error: {e}"}

            except json.JSONDecodeError as e:
                return {"success": False, "error": f"Invalid JSON response: {e}"}

            except Exception as e:
                if self.verbose >= 2:
                    print(f"Unexpected error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    self._reconnect()
                    continue
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
    
    def place_entrance_exit(self):
        if self.ride_id is None:
            return {"success": False, "error": "No ride created"}
            
        req = {
            "endpoint": "placeEntranceExit",
            "params": {
                "rideId": self.ride_id
            }
        }
        return self.send_request(req)