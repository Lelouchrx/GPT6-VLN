import unittest

import numpy as np

from gpt_vln.clearance import forward_clearance
from gpt_vln.review_resolution import (normalize_review_metadata,
                                       pending_route_review, validate_review_resolution)


class ClearanceReviewTest(unittest.TestCase):
    def test_redundant_review_and_retrieved_frame_are_normalized(self):
        output = {'skill': 'move', 'map_operation': 'none', 'map_node_id': None,
                  'actions': [{'type': 'forward', 'centimeters': 25}],
                  'history_panorama_ids': [40],
                  'review_response': {'resolution': 'continue_with_counterevidence',
                                      'map_node_id': None, 'evidence_frames': [40, 44],
                                      'current_test': 'Compare the two doors'}}
        normalized, changes = normalize_review_metadata(output, None, {40})
        self.assertEqual(normalized['history_panorama_ids'], [])
        self.assertEqual(normalized['review_response']['resolution'], 'not_applicable')
        self.assertEqual(normalized['actions'], output['actions'])
        self.assertEqual(len(changes), 2)
        self.assertEqual(output['history_panorama_ids'], [40])
        validate_review_resolution(None, normalized['review_response'],
                                   normalized, {40, 44}, 44)

    def test_unretrieved_frame_and_real_review_are_not_normalized(self):
        output = {'skill': 'move', 'map_operation': 'none', 'map_node_id': None,
                  'history_panorama_ids': [40],
                  'review_response': {'resolution': 'continue_with_counterevidence',
                                      'map_node_id': None}}
        pending = {'route_error_since_previous_review': 'possible_wrong_branch'}
        normalized, changes = normalize_review_metadata(output, pending, set())
        self.assertEqual(normalized, output)
        self.assertEqual(changes, [])

    def test_dense_near_surface_interrupts_before_motion(self):
        depth = np.full((400, 400), 2.0)
        depth[176:240, 160:240] = .48
        result = forward_clearance(depth)
        self.assertTrue(result['interrupt'])
        self.assertEqual(result['verdict'], 'near_surface')
        self.assertEqual(result['threshold_m'], .5)

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
