class APITrackBuilder:
    def __init__(self, api_controller):
        self.api = api_controller
        self.direction_vectors = [
            (0, 1),   # North (0)
            (1, 0),   # East (1)
            (0, -1),  # South (2)
            (-1, 0)   # West (3)
        ]
        self.history = []
        
        # Comprehensive track type mapping with all available pieces
        self.action_to_track_type = {
            # Basic
            0: 0,   # Flat/straight
            
            # Turns (5-tile radius)
            1: 16,  # Left turn 5-tile
            2: 17,  # Right turn 5-tile
            
            # Turns (3-tile radius - tighter)
            3: 42,  # Left turn 3-tile
            4: 43,  # Right turn 3-tile
            
            # Basic slopes WITHOUT chain
            5: 4,   # Up 25°
            6: 10,  # Down 25°
            7: 5,   # Up 60° (steep, no chain possible)
            8: 11,  # Down 60° (steep)
            
            # Slopes WITH CHAIN LIFT
            9: 4,   # Up 25° WITH CHAIN LIFT
            10: 6,  # Flat to Up 25° WITH CHAIN LIFT
            
            # Slope transitions
            11: 6,  # Flat to Up 25° (no chain)
            12: 12, # Flat to Down 25°
            13: 9,  # Up 25° to Flat
            14: 15, # Down 25° to Flat
            
            # Banking
            15: 18, # Flat to Left Bank
            16: 19, # Flat to Right Bank
            17: 32, # Left Bank (continuous)
            18: 33, # Right Bank (continuous)
            19: 20, # Left Bank to Flat
            20: 21, # Right Bank to Flat
            
            # Banked turns (5-tile)
            21: 22, # Banked left turn 5-tile
            22: 23, # Banked right turn 5-tile
            
            # Banked turns (3-tile)
            23: 44, # Banked left turn 3-tile
            24: 45, # Banked right turn 3-tile
            
            # Steep transitions
            25: 8,  # Up 25° to Up 60°
            26: 7,  # Up 60° to Up 25°
            27: 13, # Down 25° to Down 60°
            28: 14, # Down 60° to Down 25°
            
            # S-Bends
            29: 38, # S-Bend left
            30: 39, # S-Bend right
            
            # Special action
            31: -1  # Remove piece
        }
        
        # Actions that should add chain lift
        self.chain_lift_actions = {9, 10}  # Up 25° with chain, Flat to Up 25° with chain
        
    def take_action(self, action, current_position, current_direction):
        success = False
        new_position = current_position.copy()
        new_direction = current_direction
        
        if action == 31:  # Remove piece
            if not self.history:
                return False, new_position, new_direction
                
            # Use the new deleteLastTrackPiece API endpoint
            resp = self.api.delete_last_track_piece()
            
            if resp.get("success"):
                success = True
                # Pop from history to keep it in sync
                if self.history:
                    self.history.pop()
                
                # Get the new position from the API response
                payload = resp.get("payload", {})
                next_endpoint = payload.get("nextEndpoint")
                
                if next_endpoint:
                    # There are still pieces remaining, update position
                    new_position = [next_endpoint["x"], next_endpoint["y"], next_endpoint["z"]]
                    new_direction = next_endpoint["direction"]
                elif self.history:
                    # No pieces remaining according to API, but we have history
                    # Use the last entry in our history
                    last_entry = self.history[-1] if self.history else None
                    if last_entry:
                        new_position = last_entry["next_position"].copy()
                        new_direction = last_entry["next_direction"]
                
                pieces_remaining = payload.get("piecesRemaining", 0)
                if pieces_remaining == 0:
                    print("All track pieces removed, only station remains")
            else:
                success = False
                error_msg = resp.get("error", "Unknown error")
                print(f"Failed to delete track piece: {error_msg}")
                
            return success, new_position, new_direction
        
        # Get track type for this action
        track_type = self.action_to_track_type.get(action, 0)
        
        # Check if we should add chain lift (only for specific actions)
        has_chain = action in self.chain_lift_actions
        
        # Try to place the track piece
        resp = self.api.place_track_piece(
            current_position[0],
            current_position[1],
            current_position[2],
            current_direction,
            track_type,
            has_chain
        )
        
        if resp.get("success"):
            success = True
            next_ep = resp["payload"]["nextEndpoint"]
            new_position = [next_ep["x"], next_ep["y"], next_ep["z"]]
            new_direction = next_ep["direction"]
            
            # Check if circuit is complete
            is_complete = resp["payload"].get("isCircuitComplete", False)
            
            # Store in history for potential removal
            self.history.append({
                "action": action,
                "position": current_position.copy(),
                "direction": current_direction,
                "next_position": new_position.copy(),
                "next_direction": new_direction,
                "track_type": track_type,
                "is_complete": is_complete
            })
            
            if is_complete:
                print(f"[API] Circuit complete detected! Track returned to station after {len(self.history)} pieces.")
                print(f"[API] Final position: {new_position}, Direction: {new_direction}")
        
        return success, new_position, new_direction
    
    def get_valid_actions(self):
        resp = self.api.get_valid_next_pieces()
        if resp.get("success"):
            valid_pieces = resp["payload"]["validPieces"]
            # Convert track types back to actions
            valid_actions = []
            for action, track_type in self.action_to_track_type.items():
                if track_type in valid_pieces and track_type != -1:  # Skip remove action
                    valid_actions.append(action)
                    # Special case: if track type 4 is valid, both with/without chain are valid
                    if track_type == 4 and action == 5:  # Up 25° without chain
                        valid_actions.append(9)  # Also add Up 25° with chain
                    elif track_type == 6 and action == 11:  # Flat to Up 25° without chain
                        valid_actions.append(10)  # Also add with chain
            return valid_actions
        return []