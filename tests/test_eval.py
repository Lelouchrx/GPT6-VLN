import unittest

import numpy as np

from eval import History, Settings, parse_model_output


class EvalTest(unittest.TestCase):
    def test_frontier_coordinate_parsing(self):
        parsed = parse_model_output("<frontiers_coord>(320, 240)", Settings())
        self.assertEqual(parsed["task_type"], "frontier")
        self.assertEqual(parsed["coordinate"], [320.0, 240.0])

    def test_action_sequence_parsing(self):
        parsed = parse_model_output("<action>TURN_LEFT FORWARD STOP</action>", Settings())
        self.assertEqual(parsed["action_sequence"], [2, 1, 0])

    def test_uniform_history_spans_episode(self):
        history = History("uniform", 4)
        for index in range(10):
            history.add(np.full((1, 1, 3), index, np.uint8))
        selected = [int(frame[0, 0, 0]) for frame in history.select()]
        self.assertEqual((selected[0], selected[-1]), (0, 9))

    def test_recent_history_is_sliding_window(self):
        history = History("recent", 3)
        for index in range(6):
            history.add(np.full((1, 1, 3), index, np.uint8))
        self.assertEqual([int(frame[0, 0, 0]) for frame in history.select()], [3, 4, 5])


if __name__ == "__main__":
    unittest.main()
