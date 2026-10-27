import copy
import unittest
from unittest import mock

from gpt_vln.action_policy import run
from gpt_vln.model import navigation_prompt, without_semantic_map
from gpt_vln.model_contract import PROMPT
from gpt_vln.route_review import RouteReviewMemory


class NavigationContractTest(unittest.TestCase):
    def test_geometry_input_keeps_return_nodes_without_labels(self):
        context = {'observed_semantic_map': {
            'current_pose': {'xyz_m': [0, 0, 0]},
            'nodes': [{'id': 1, 'links': [0], 'name': 'private room',
                       'caption': 'private label', 'annotations': []}],
            'edges': [{'a': 0, 'b': 1}],
            'room_stage_hypotheses': [{'room': 'private room'}]}}
        public = without_semantic_map(context)['observed_semantic_map']
        self.assertEqual(public['nodes'], [{'id': 1, 'links': [0]}])
        self.assertEqual(public['edges'], [{'a': 0, 'b': 1}])
        self.assertNotIn('private room', str(public))
        self.assertNotIn('ROOM STAGE MEMORY', navigation_prompt(PROMPT, True))

    def test_review_is_due_before_next_decision(self):
        review = RouteReviewMemory(10)
        self.assertFalse(review.due(9))
        self.assertTrue(review.due(10))

    def test_predicted_plan_is_cropped_before_execution(self):
        planned = {'skill': 'move', 'maneuver': 'travel',
                   'actions': [{'type': 'forward', 'centimeters': 25}] * 3}
        with mock.patch('gpt_vln.action_policy.navigate_cli',
                        return_value={'output': planned, 'session_id': 'thread'}):
            response = run({'horizon_mode': '3_to_2', 'context': {}})
        self.assertEqual(len(response['output']['actions']), 2)
        self.assertEqual(len(response['horizon']['planned']), 3)
        self.assertEqual(len(planned['actions']), 3)


if __name__ == '__main__':
    unittest.main()
