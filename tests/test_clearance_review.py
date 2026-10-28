import unittest

import numpy as np

from gpt_vln.clearance import forward_clearance
from gpt_vln.review_resolution import pending_route_review, validate_review_resolution


class ClearanceReviewTest(unittest.TestCase):
    def test_dense_near_surface_interrupts_before_motion(self):
        depth = np.full((400, 400), 2.0)
        depth[176:240, 160:240] = .32
        result = forward_clearance(depth)
        self.assertTrue(result['interrupt'])
        self.assertEqual(result['verdict'], 'near_surface')

    def test_sparse_or_invalid_depth_does_not_block(self):
        depth = np.full((400, 400), 2.0)
        depth[176:240, 160:180] = .2
        self.assertFalse(forward_clearance(depth)['interrupt'])
        depth[:] = np.nan
        self.assertEqual(forward_clearance(depth)['verdict'], 'unknown')

    def test_contradicted_review_requires_inspection_or_return(self):
        latest = {'route': 'contradicted',
                  'route_error_since_previous_review': 'possible_wrong_branch',
                  'direction_assessment': 'possibly_wrong',
                  'prediction_errors': [], 'checked_before_decision': 10,
                  'suggested_return_anchor_frame': 4}
        pending = pending_route_review(latest)
        self.assertEqual(pending['route'], 'contradicted')
        output = {'skill': 'stop', 'map_operation': 'none', 'map_node_id': None,
                  'actions': []}
        response = {'resolution': 'continue_with_counterevidence',
                    'evidence_frames': [14], 'map_node_id': None,
                    'current_test': 'Inspect the candidate entrance'}
        with self.assertRaisesRegex(ValueError, 'requires a branch inspection'):
            validate_review_resolution(pending, response, output, {14}, 14)
        output['skill'] = 'move'
        output['actions'] = [{'type': 'forward', 'centimeters': 25}]
        response['resolution'] = 'inspect_suspected_branch'
        validate_review_resolution(pending, response, output, {14}, 14)


if __name__ == '__main__':
    unittest.main()
