import unittest
from unittest import mock

import numpy as np

from gpt_vln.depth_preflight import forward_clearance
from gpt_vln.pose_map import OwnPoseMap


class DepthPreflightTest(unittest.TestCase):
    def test_near_surface_warns_before_two_forward_steps(self):
        depth = np.full((400, 400, 1), 0.55, dtype=np.float32)
        result = forward_clearance({'rgb_map_depth': depth})
        self.assertEqual(result['status'], 'possible_obstacle')
        self.assertEqual(result['lookahead_m'], 0.5)

    def test_far_or_missing_depth_does_not_certify_a_route(self):
        far = forward_clearance({'rgb_map_depth': np.full((400, 400), 2.0)})
        self.assertEqual(far['status'], 'no_near_obstacle_seen')
        self.assertIn('floor remains unverified', far['reason'])
        self.assertEqual(forward_clearance({})['status'], 'unknown')
        self.assertEqual(forward_clearance({'rgb_map_depth': np.zeros((400, 400))})['status'], 'unknown')

    def test_pose_history_is_short_and_chronological(self):
        state = mock.Mock()
        state.position = [0.0, 0.0, 0.0]
        state.rotation = mock.Mock(x=0.0, y=0.0, z=0.0, w=1.0)
        sim = mock.Mock(get_agent_state=mock.Mock(return_value=state))
        history = OwnPoseMap(sim)
        for step in range(1, 20):
            state.position = [step * .25, 0.0, 0.0]
            history.record(step)
        result = history.history_summary(list(range(1, 19)))
        frames = [row['frame'] for row in result['points']]
        self.assertEqual(frames, [0, 13, 14, 15, 16, 17, 18, 19])
        self.assertNotIn('occupancy', str(result).lower())


if __name__ == '__main__':
    unittest.main()
