class TrackBuilder:
    def __init__(self, ui_controller):
        self.ui_controller = ui_controller
        self.direction_vectors = [
            (0, 1),   # North (0)
            (1, 0),   # East (1)
            (0, -1),  # South (2)
            (-1, 0)   # West (3)
        ]
        self.history = []  # List to store (action, position, direction) tuples

    
    def take_action(self, action, current_position, current_direction):
        success = False
        new_position = current_position.copy()
        new_direction = current_direction
   
        # Map action to track piece and call appropriate method
        if action == 0:
            success = self.ui_controller.add_track_piece("straight_level_noroll")
            if success:
                # Move 1 step in the current direction (x and y only)
                dx, dy = self.direction_vectors[current_direction]
                new_position[0] += dx
                new_position[1] += dy
                # Z (height) doesn't change for level pieces
        elif action == 1:
            success = self.ui_controller.add_track_piece("left_level_noroll")
            if success:
                # Move 3 steps forward and 2 steps left (x and y only)
                dx, dy = self.direction_vectors[current_direction]
                new_position[0] += 3 * dx
                new_position[1] += 3 * dy

                left_direction = (current_direction - 1) % 4
                dx, dy = self.direction_vectors[left_direction]
                new_position[0] += 2 * dx
                new_position[1] += 2 * dy

                # Z (height) doesn't change for level pieces

                # Update direction (turn counter-clockwise)
                new_direction = (current_direction - 1) % 4
        elif action == 2:
            success = self.ui_controller.add_track_piece("small_left_level_noroll")
            if success:
                # Move 2 steps forward and 1 steps left
                dx, dy = self.direction_vectors[current_direction]
                new_position[0] += 2 * dx
                new_position[1] += 2 * dy

                left_direction = (current_direction - 1) % 4
                dx, dy = self.direction_vectors[left_direction]
                new_position[0] += 1 * dx
                new_position[1] += 1 * dy

                # Update direction (turn counter-clockwise)
                new_direction = (current_direction - 1) % 4
        elif action == 3:
            success = self.ui_controller.add_track_piece("right_level_noroll")
            if success:
                # Move 3 steps forward and 2 steps right (x and y only)
                dx, dy = self.direction_vectors[current_direction]
                new_position[0] += 3 * dx
                new_position[1] += 3 * dy

                right_direction = (current_direction + 1) % 4
                dx, dy = self.direction_vectors[right_direction]
                new_position[0] += 2 * dx
                new_position[1] += 2 * dy

                # Z (height) doesn't change for level pieces

                # Update direction (turn clockwise)
                new_direction = (current_direction + 1) % 4
        elif action == 4:
            success = self.ui_controller.add_track_piece("small_right_level_noroll")
            if success:
                # Move 2 steps forward and 1 steps right (x and y only)
                dx, dy = self.direction_vectors[current_direction]
                new_position[0] += 2 * dx
                new_position[1] += 2 * dy

                right_direction = (current_direction + 1) % 4
                dx, dy = self.direction_vectors[right_direction]
                new_position[0] += 1 * dx
                new_position[1] += 1 * dy

                # Z (height) doesn't change for level pieces

                # Update direction (turn clockwise)
                new_direction = (current_direction + 1) % 4
        elif action == 5:
            success = self.ui_controller.add_track_piece("straight_down_noroll")
            if success:
                # Move 1 step in the current direction (x and y only)
                dx, dy = self.direction_vectors[current_direction]
                new_position[0] += dx
                new_position[1] += dy
                # Z (height) decrease by 1
                new_position[2] -= 1
        elif action == 6:
            success = self.ui_controller.add_track_piece("straight_up_noroll")
            if success:
                # Move 1 step in the current direction (x and y only)
                dx, dy = self.direction_vectors[current_direction]
                new_position[0] += dx
                new_position[1] += dy
                # Z (height) increase by 1
                new_position[2] += 1
        elif action == 7:
            success = self.ui_controller.add_track_piece("left_up_noroll")
            if success:
                # Move 3 steps forward, 2 steps left, and 3 steps up
                dx, dy = self.direction_vectors[current_direction]
                new_position[0] += 3 * dx
                new_position[1] += 3 * dy

                left_direction = (current_direction - 1) % 4
                dx, dy = self.direction_vectors[left_direction]
                new_position[0] += 2 * dx
                new_position[1] += 2 * dy

                # Move 3 steps up
                new_position[2] += 3

                # Update direction (turn counter-clockwise)
                new_direction = (current_direction - 1) % 4
        elif action == 8:
            success = self.ui_controller.add_track_piece("right_up_noroll")
            if success:
                # Move 3 steps forward, 2 steps right, and 3 steps up
                dx, dy = self.direction_vectors[current_direction]
                new_position[0] += 3 * dx
                new_position[1] += 3 * dy

                right_direction = (current_direction + 1) % 4
                dx, dy = self.direction_vectors[right_direction]
                new_position[0] += 2 * dx
                new_position[1] += 2 * dy

                # Move 3 steps up
                new_position[2] += 3

                # Update direction (turn clockwise)
                new_direction = (current_direction + 1) % 4
        elif action == 9:
            success = self.ui_controller.add_track_piece("left_down_noroll")
            if success:
                # Move 3 steps forward, 2 steps left
                dx, dy = self.direction_vectors[current_direction]
                new_position[0] += 3 * dx
                new_position[1] += 3 * dy

                left_direction = (current_direction - 1) % 4
                dx, dy = self.direction_vectors[left_direction]
                new_position[0] += 2 * dx
                new_position[1] += 2 * dy

                # Move 3 steps down
                new_position[2] -= 3

                # Update direction (turn counter-clockwise)
                new_direction = (current_direction - 1) % 4
        elif action == 10:
            success = self.ui_controller.add_track_piece("right_down_noroll")
            if success:
                # Move 3 steps forward, 2 steps right
                dx, dy = self.direction_vectors[current_direction]
                new_position[0] += 3 * dx
                new_position[1] += 3 * dy

                right_direction = (current_direction + 1) % 4
                dx, dy = self.direction_vectors[right_direction]
                new_position[0] += 2 * dx
                new_position[1] += 2 * dy

                # Move 3 steps down
                new_position[2] -= 3

                # Update direction (turn clockwise)
                new_direction = (current_direction + 1) % 4
        elif action == 11:
            success = self.ui_controller.add_track_piece("left_level_leftroll")
            if success:
                # Move 3 steps forward and 2 steps left (x and y only)
                dx, dy = self.direction_vectors[current_direction]
                new_position[0] += 3 * dx
                new_position[1] += 3 * dy

                left_direction = (current_direction - 1) % 4
                dx, dy = self.direction_vectors[left_direction]
                new_position[0] += 2 * dx
                new_position[1] += 2 * dy

                # Z (height) doesn't change for level pieces

                # Update direction (turn counter-clockwise)
                new_direction = (current_direction - 1) % 4
        elif action == 12:
            success = self.ui_controller.add_track_piece("right_level_rightroll")
            if success:
                # Move 3 steps forward and 2 steps right (x and y only)
                dx, dy = self.direction_vectors[current_direction]
                new_position[0] += 3 * dx
                new_position[1] += 3 * dy

                right_direction = (current_direction + 1) % 4
                dx, dy = self.direction_vectors[right_direction]
                new_position[0] += 2 * dx
                new_position[1] += 2 * dy

                # Z (height) doesn't change for level pieces

                # Update direction (turn clockwise)
                new_direction = (current_direction + 1) % 4
        elif action == 13:
            success = self.ui_controller.add_track_piece("straight_level_leftroll")
            if success:
                # Move 1 step in the current direction (x and y only)
                dx, dy = self.direction_vectors[current_direction]
                new_position[0] += dx
                new_position[1] += dy
                # Z (height) doesn't change for level pieces
        elif action == 14:
            success = self.ui_controller.add_track_piece("straight_level_rightroll")
            if success:
                # Move 1 step in the current direction (x and y only)
                dx, dy = self.direction_vectors[current_direction]
                new_position[0] += dx
                new_position[1] += dy
                # Z (height) doesn't change for level pieces
        elif action == 15:
            success = self.ui_controller.add_track_piece("straight_up_noroll_chain")
            if success:
                # Move 1 step in the current direction (x and y only)
                dx, dy = self.direction_vectors[current_direction]
                new_position[0] += dx
                new_position[1] += dy
                # Z (height) increase by 1
                new_position[2] += 1
        elif action == 16:
            success = self.ui_controller.add_track_piece("straight_steep_down_noroll")
            if success:
                # Move 1 step in the current direction (x and y only)
                dx, dy = self.direction_vectors[current_direction]
                new_position[0] += dx
                new_position[1] += dy
                # Z (height) decrease by 2
                new_position[2] -= 2
        elif action == 17:
            success = self.ui_controller.add_track_piece("straight_steep_up_noroll")
            if success:
                # Move 1 step in the current direction (x and y only)
                dx, dy = self.direction_vectors[current_direction]
                new_position[0] += dx
                new_position[1] += dy
                # Z (height) increase by 2
                new_position[2] += 2
        elif action == 18:
            if not self.history:
                # The first action can not be to remove a piece
                return False, new_position, new_direction
            success = self.ui_controller.remove_piece()
            if success and self.history:
                # Revert to the previous state
                previous_action, new_position, new_direction = self.history.pop()
            return success, new_position, new_direction
        if success and action != 18:
            # Add the current state to history before updating
            self.history.append((action, current_position.copy(), current_direction))

        return success, new_position, new_direction

