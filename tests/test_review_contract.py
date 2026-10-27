import json
import unittest
from pathlib import Path
from unittest import mock

from gpt_vln.model_reviewer import review


class ReviewContractTest(unittest.TestCase):
    def test_unsupplied_frame_is_rejected(self):
        verdict = {'route': 'unknown', 'evidence': 'uncertain', 'frames': [9],
                   'suggested_return_anchor_frame': None, 'next_check': 'inspect',
                   'task_summary': 'unknown', 'progress_hypothesis': 'unknown',
                   'direction_assessment': 'unknown', 'prediction_errors': [],
                   'stage_progress': 'unknown',
                   'route_error_since_previous_review': 'unknown'}

        def fake_run(command, **kwargs):
            Path(command[command.index('--output-last-message') + 1]).write_text(json.dumps(verdict))
            return mock.Mock(returncode=0, stderr='')

        request = {'role': 'periodic_route_review', 'content': [
            {'type': 'text', 'text': json.dumps({'available_frame_ids': [1, 2]})}]}
        with mock.patch('gpt_vln.model_reviewer.subprocess.run', side_effect=fake_run):
            result = review(request)
        self.assertIn('outside supplied images', result['error'])


if __name__ == '__main__':
    unittest.main()
