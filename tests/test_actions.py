import unittest
from gpt_vln.actions import crop_horizon

class ActionTest(unittest.TestCase):
    def test_move_prefix(self):
        output = {'skill': 'move', 'maneuver': 'travel',
                  'actions': [{'type': 'forward', 'centimeters': 25}] * 3}
        executed, record = crop_horizon(output, '3_to_2')
        self.assertEqual(len(executed['actions']), 2)
        self.assertEqual(len(record['planned']), 3)

    def test_stop(self):
        executed, _ = crop_horizon({'skill': 'stop', 'actions': []}, '3_to_2')
        self.assertEqual(executed['actions'], [])
