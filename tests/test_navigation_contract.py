import copy
import unittest
from unittest import mock

from gpt_vln.action_policy import run
from gpt_vln.model import navigation_prompt, pose_only_prompt, without_semantic_map
from gpt_vln.model_contract import PROMPT
from gpt_vln.route_review import RouteReviewMemory
from gpt_vln.execution_contract import ExecutionContractCritic
from gpt_vln.stall import PoseStallCritic
from gpt_vln.actions import crop_horizon


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

    def test_pose_only_prompt_does_not_request_a_map(self):
        prompt = pose_only_prompt(navigation_prompt(PROMPT, True))
        self.assertNotIn('MAP SENSOR OVERRIDE', prompt)
        self.assertNotIn('OBSERVED RGB-D MAP', prompt)
        self.assertIn('measured own_pose', prompt)
        self.assertIn('MANDATORY REVIEW RESPONSE', prompt)

    def test_pose_prompt_retains_action_horizon(self):
        prompt = pose_only_prompt(
            'MEMORY CONTRACT: old map text\n'
            'STABLE TASK-ENDPOINT ANCHOR: Keep the endpoint identity.\n'
            'MANDATORY REVIEW RESPONSE: Handle pending review.\n'
            'HORIZON OVERRIDE: MOVE predicts exactly 4 commands')
        self.assertIn('MOVE predicts exactly 4', prompt)
        self.assertIn('MANDATORY REVIEW RESPONSE', prompt)
        self.assertNotIn('old map text', prompt)

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

    def test_free_mode_preserves_model_actions(self):
        planned = {'skill': 'move', 'maneuver': 'climb_up',
                   'actions': [{'type': 'forward', 'centimeters': 75}]}
        with mock.patch('gpt_vln.action_policy.navigate_cli',
                        return_value={'output': planned, 'session_id': 'thread'}) as call:
            response = run({'horizon_mode': 'free', 'context': {}})
        self.assertEqual(response['output']['actions'], planned['actions'])
        self.assertTrue(call.call_args.args[0]['merge_forward_commands'])

    def test_long_stair_command_allowed_and_checked_per_primitive(self):
        planned = {'skill': 'move', 'maneuver': 'climb_up',
                   'actions': [{'type': 'forward', 'centimeters': 75}] * 3}
        executed, _ = crop_horizon(planned, '3_to_2')
        self.assertEqual([x['centimeters'] for x in executed['actions']], [75, 75])

        critic = ExecutionContractCritic()
        first = critic.evaluate({'maneuver': 'climb_up', 'travel_heading_deg': 0},
                                {'step': 1, 'action': 1, 'source': 'measured_pose',
                                 'height_delta_m': 0.0, 'translation_xz_m': [0, 0]})
        second = critic.evaluate({'maneuver': 'climb_up', 'travel_heading_deg': 0},
                                 {'step': 2, 'action': 1, 'source': 'measured_pose',
                                  'height_delta_m': 0.0, 'translation_xz_m': [0, 0]})
        self.assertFalse(first['interrupt'])
        self.assertTrue(second['interrupt'])

    def test_pose_stall_interrupts_repeated_blocked_forward(self):
        critic = PoseStallCritic(min_forward_m=.075, consecutive=2)
        first = critic.check(1, [0, 0], 0, 1)
        second = critic.check(1, [0, 0], 0, 2)
        self.assertFalse(first['interrupt'])
        self.assertTrue(second['interrupt'])


if __name__ == '__main__':
    unittest.main()
