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
            self.sock.close()
            self.sock = None
            
    def send_request(self, request):
        if not self.sock:
            if not self.connect():
                return {"success": False, "error": "Not connected to API server"}
        
        try:
            message = json.dumps(request) + "\n"
            self.sock.sendall(message.encode("utf-8"))
            
            file_obj = self.sock.makefile("r")
            line = file_obj.readline()
            if not line:
                return {"success": False, "error": "No response from server"}
            response = json.loads(line)
        except socket.timeout:
            print(f"Request timeout for endpoint: {request.get('endpoint', 'unknown')}")
            response = {"success": False, "error": "Request timeout"}
        except Exception as e:
            response = {"success": False, "error": "Failed to decode response: " + str(e)}
        return response
    
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