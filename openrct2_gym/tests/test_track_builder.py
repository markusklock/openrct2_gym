import unittest
from unittest.mock import Mock
from track_builder import TrackBuilder  # Adjust the import as needed


class TestTrackBuilder(unittest.TestCase):
    def setUp(self):
        self.mock_ui = Mock()
        self.builder = TrackBuilder(self.mock_ui)
        self.initial_position = [0, 0, 0]
        self.initial_direction = 0  # North

    def test_action_0_success(self):
        """
        Action 0: Add 'straight_level_noroll'
        - If successful, we move one step forward in the current direction (north).
        """
        self.mock_ui.add_track_piece.return_value = True

        success, new_position, new_direction = self.builder.take_action(
            0, self.initial_position, self.initial_direction
        )

        # Assertions
        self.assertTrue(success)
        self.mock_ui.add_track_piece.assert_called_once_with("straight_level_noroll")
        # Since direction = 0 (north => (0, 1)), we expect (0,1,0)
        self.assertEqual(new_position, [0, 1, 0])
        # Direction should remain the same
        self.assertEqual(new_direction, 0)
        # Check history
        self.assertEqual(len(self.builder.history), 1)
        last_history_entry = self.builder.history[-1]
        self.assertEqual(last_history_entry[0], 0)              # action
        self.assertEqual(last_history_entry[1], [0, 0, 0])      # old position
        self.assertEqual(last_history_entry[2], 0)              # old direction

    def test_action_0_failure(self):
        """
        Action 0: Add 'straight_level_noroll'
        - If NOT successful, do not update position or direction, do not update history.
        """
        self.mock_ui.add_track_piece.return_value = False

        success, new_position, new_direction = self.builder.take_action(
            0, self.initial_position, self.initial_direction
        )

        # Assertions
        self.assertFalse(success)
        self.mock_ui.add_track_piece.assert_called_once_with("straight_level_noroll")
        # Position/direction should not change
        self.assertEqual(new_position, self.initial_position)
        self.assertEqual(new_direction, self.initial_direction)
        # History should remain empty
        self.assertEqual(len(self.builder.history), 0)

    def test_action_1_success(self):
        """
        Action 1: Add 'left_level_noroll'
        - If successful, move 3 steps forward + 2 steps left, update direction counter-clockwise.
        - For direction = 0 (north), forward = (0,1), left = (-1,0).
          3*(0,1) = (0,3) and 2*(-1,0) = (-2,0). Summed: (0+(-2), 3+0) => (-2, 3).
          New direction = (0 - 1) % 4 = 3 (west).
        """
        self.mock_ui.add_track_piece.return_value = True

        success, new_position, new_direction = self.builder.take_action(
            1, [0, 0, 0], 0
        )

        # Assertions
        self.assertTrue(success)
        self.mock_ui.add_track_piece.assert_called_once_with("left_level_noroll")
        # Expect position = (-2, 3, 0)
        self.assertEqual(new_position, [-2, 3, 0])
        self.assertEqual(new_direction, 3)
        # Check history
        self.assertEqual(len(self.builder.history), 1)
        last_history_entry = self.builder.history[-1]
        self.assertEqual(last_history_entry[0], 1)
        self.assertEqual(last_history_entry[1], [0, 0, 0])
        self.assertEqual(last_history_entry[2], 0)

    def test_action_5_straight_down_noroll_success(self):
        """
        Action 5: 'straight_down_noroll'.
        - If successful, move 1 step forward in the current direction, reduce Z by 1.
        - direction=0 => forward is (0,1).
        """
        self.mock_ui.add_track_piece.return_value = True

        success, new_position, new_direction = self.builder.take_action(
            5, [0, 0, 5], 0
        )

        self.assertTrue(success)
        self.mock_ui.add_track_piece.assert_called_once_with("straight_down_noroll")
        # Moved from (0,0,5) to (0,1,4)
        self.assertEqual(new_position, [0, 1, 4])
        # Direction remains unchanged
        self.assertEqual(new_direction, 0)
        # Check history
        self.assertEqual(len(self.builder.history), 1)

    def test_action_6_straight_up_noroll_success(self):
        """
        Action 6: 'straight_up_noroll'.
        - If successful, move 1 step forward, Z + 1.
        """
        self.mock_ui.add_track_piece.return_value = True

        success, new_position, new_direction = self.builder.take_action(
            6, [2, 3, 0], 0
        )

        self.assertTrue(success)
        self.mock_ui.add_track_piece.assert_called_once_with("straight_up_noroll")
        # Moved from (2,3,0) to (2,4,1)
        self.assertEqual(new_position, [2, 4, 1])
        self.assertEqual(new_direction, 0)
        self.assertEqual(len(self.builder.history), 1)

    def test_action_18_remove_no_history(self):
        """
        Action 18: remove the latest piece.
        - If history is empty, cannot remove. Should return False and no changes.
        """
        success, new_position, new_direction = self.builder.take_action(
            18, self.initial_position, self.initial_direction
        )

        self.assertFalse(success)
        # remove_piece should not be called
        self.mock_ui.remove_piece.assert_not_called()
        # No position/direction changes
        self.assertEqual(new_position, self.initial_position)
        self.assertEqual(new_direction, self.initial_direction)
        self.assertEqual(len(self.builder.history), 0)

    def test_action_18_remove_with_history(self):
        """
        Action 18: remove the latest piece.
        - If history is not empty and remove is successful, we revert to previous state.
        """
        # First, simulate a successful action 0 to populate history
        self.mock_ui.add_track_piece.return_value = True
        self.builder.take_action(0, self.initial_position, self.initial_direction)

        # Now remove
        self.mock_ui.remove_piece.return_value = True
        success, new_position, new_direction = self.builder.take_action(
            18, [0, 1, 0], 0
        )

        self.assertTrue(success)
        self.mock_ui.remove_piece.assert_called_once()
        # We revert to the old state
        self.assertEqual(new_position, [0, 0, 0])
        self.assertEqual(new_direction, 0)
        # History should now be empty again
        self.assertEqual(len(self.builder.history), 0)

    def test_action_18_remove_with_history_failure(self):
        """
        Action 18: remove the latest piece.
        - If history is not empty but remove fails, no revert happens.
        """
        # First, simulate a successful action 0 to populate history
        self.mock_ui.add_track_piece.return_value = True
        self.builder.take_action(0, self.initial_position, self.initial_direction)

        # Attempt to remove, but removal fails
        self.mock_ui.remove_piece.return_value = False
        success, new_position, new_direction = self.builder.take_action(
            18, [0, 1, 0], 0
        )

        self.assertFalse(success)
        self.mock_ui.remove_piece.assert_called_once()
        # We do not revert to the old state
        self.assertEqual(new_position, [0, 1, 0])
        self.assertEqual(new_direction, 0)
        # History should remain the same
        self.assertEqual(len(self.builder.history), 1)

    def test_action_2_small_left_level_noroll_success(self):
        """
        Action 2: "small_left_level_noroll"
         - If successful:
             1. Move 2 steps forward in the current direction.
             2. Move 1 step left.
             3. Turn direction counter-clockwise.
         - For current_direction = 0 (North):
             Forward = (0, 1)
             Left = (current_direction - 1) % 4 => 3 (West), which is (-1, 0).
             So total change in x,y = 2*(0,1) + 1*(-1,0) = (0,2) + (-1,0) = (-1,2)
        """
        self.mock_ui.add_track_piece.return_value = True
        success, new_position, new_direction = self.builder.take_action(
            2, [1, 1, 0], 0
        )

        self.assertTrue(success)
        self.mock_ui.add_track_piece.assert_called_once_with("small_left_level_noroll")
        # Start (1,1,0), move (-1,2) => (0,3,0)
        self.assertEqual(new_position, [0, 3, 0])
        # Direction is now west = 3
        self.assertEqual(new_direction, 3)
        # History updated
        self.assertEqual(len(self.builder.history), 1)
        self.assertEqual(self.builder.history[-1][0], 2)

    def test_action_2_small_left_level_noroll_failure(self):
        """
        Action 2: "small_left_level_noroll"
         - If NOT successful, no movement or direction change, no history update.
        """
        self.mock_ui.add_track_piece.return_value = False
        success, new_position, new_direction = self.builder.take_action(
            2, self.initial_position, self.initial_direction
        )

        self.assertFalse(success)
        self.mock_ui.add_track_piece.assert_called_once_with("small_left_level_noroll")
        self.assertEqual(new_position, self.initial_position)
        self.assertEqual(new_direction, self.initial_direction)
        self.assertEqual(len(self.builder.history), 0)

    def test_action_3_right_level_noroll_success(self):
        """
        Action 3: "right_level_noroll"
         - If successful, move 3 steps forward, 2 steps right, turn clockwise.
         - For current_direction=0 (North):
             forward = (0,1) => 3*(0,1) = (0,3)
             right_dir = (0 + 1) % 4 = 1 (East => (1,0))
             2*(1,0) = (2,0)
             total (x,y) change: (0+2, 3+0) = (2,3)
             new direction = East (1)
        """
        self.mock_ui.add_track_piece.return_value = True
        success, new_position, new_direction = self.builder.take_action(
            3, [5, 5, 0], 0
        )

        self.assertTrue(success)
        self.mock_ui.add_track_piece.assert_called_once_with("right_level_noroll")
        # (5,5,0) + (2,3,0) => (7,8,0)
        self.assertEqual(new_position, [7, 8, 0])
        self.assertEqual(new_direction, 1)  # East
        self.assertEqual(len(self.builder.history), 1)

    def test_action_4_small_right_level_noroll_success(self):
        """
        Action 4: "small_right_level_noroll"
         - If successful, move 2 steps forward, 1 step right, turn clockwise.
         - For current_direction=2 (South):
             forward = (0, -1)
             right = (current_direction+1) % 4 => 3 (West => (-1,0)) 
             But be careful: if direction=2 is south => forward=(0,-1).
                 2*(0,-1) = (0,-2)
                 right=(3 => West=(-1,0))
                 1*West = (-1,0)
             total => (0+(-1), -2+0) = (-1, -2)
         - new direction = 3 (West)
        """
        self.mock_ui.add_track_piece.return_value = True
        success, new_position, new_direction = self.builder.take_action(
            4, [10, 10, 0], 2  # Start facing South
        )

        self.assertTrue(success)
        self.mock_ui.add_track_piece.assert_called_once_with("small_right_level_noroll")
        # (10,10,0) + (-1,-2) => (9,8,0)
        self.assertEqual(new_position, [9, 8, 0])
        self.assertEqual(new_direction, 3)  # West
        self.assertEqual(len(self.builder.history), 1)

    def test_action_15_straight_up_noroll_chain_success(self):
        """
        Action 15: "straight_up_noroll_chain"
         - If successful, move forward 1 in current direction, increase Z by 1.
           This also implies a chain-lift track piece is placed.
         - For direction=1 (East => (1,0)):
             position += (1,0), z += 1
        """
        self.mock_ui.add_track_piece.return_value = True
        success, new_position, new_direction = self.builder.take_action(
            15, [0, 0, 5], 1  # facing East
        )

        self.assertTrue(success)
        self.mock_ui.add_track_piece.assert_called_once_with("straight_up_noroll_chain")
        # Move from (0,0,5) to (1,0,6)
        self.assertEqual(new_position, [1, 0, 6])
        self.assertEqual(new_direction, 1)
        self.assertEqual(len(self.builder.history), 1)

    def test_action_16_straight_steep_down_noroll_success(self):
        """
        Action 16: "straight_steep_down_noroll"
         - If successful, move forward 1 in current direction, decrease Z by 2.
         - For direction=0 (North => (0,1)):
             position += (0,1), z -= 2
        """
        self.mock_ui.add_track_piece.return_value = True
        success, new_position, new_direction = self.builder.take_action(
            16, [3, 3, 10], 0
        )

        self.assertTrue(success)
        self.mock_ui.add_track_piece.assert_called_once_with("straight_steep_down_noroll")
        # (3,3,10) => (3,4,8)
        self.assertEqual(new_position, [3, 4, 8])
        self.assertEqual(new_direction, 0)
        self.assertEqual(len(self.builder.history), 1)

    def test_action_17_straight_steep_up_noroll_success(self):
        """
        Action 17: "straight_steep_up_noroll"
         - If successful, move forward 1 in current direction, increase Z by 2.
         - For direction=2 (South => (0,-1)):
             position += (0,-1), z += 2
        """
        self.mock_ui.add_track_piece.return_value = True
        success, new_position, new_direction = self.builder.take_action(
            17, [5, 5, 0], 2
        )

        self.assertTrue(success)
        self.mock_ui.add_track_piece.assert_called_once_with("straight_steep_up_noroll")
        # (5,5,0) => (5,4,2)
        self.assertEqual(new_position, [5, 4, 2])
        self.assertEqual(new_direction, 2)
        self.assertEqual(len(self.builder.history), 1)

if __name__ == '__main__':
    unittest.main()

