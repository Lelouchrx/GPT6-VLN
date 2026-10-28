import base64
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from gpt_vln.model_runtime import RGBBackend
from gpt_vln.model import navigation_prompt, pose_only_prompt, stateless_prompt
from gpt_vln.model_contract import PROMPT


class StatelessMemoryTest(unittest.TestCase):
    def test_prompt_matches_visible_history(self):
        prompt = stateless_prompt(pose_only_prompt(navigation_prompt(PROMPT, True)))
        self.assertNotIn('saved KEY', prompt)
        self.assertNotIn('retrieve the anchor image', prompt)
        self.assertIn('Only the sampled FRONT images are visible history', prompt)

    def test_only_selected_front_frames_are_attached(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = {}
            for frame in range(9):
                path = root / f'{frame}.jpg'
                path.write_bytes(f'front {frame}'.encode())
                archive[str(frame)] = str(path)
            selected = [0, 2, 3, 4, 5, 6, 7, 8]
            backend = RGBBackend(root, {'provider': 'claude',
                                        'pose_without_map': True,
                                        'no_semantic_map': True,
                                        'memory_policy': 'uniform8'}, 0)
            payload = {'role': 'navigate', 'context': {
                'observed_semantic_map': {'current_pose': {'xyz_from_start_m': [0, 0, 0]},
                                          'nodes': []}},
                'frames': [{'label': 'CURRENT four views',
                            'data': base64.b64encode(b'current').decode()}],
                'memory_catalog': {'root': str(root), 'front_archive': archive,
                                   'uniform_frame_ids': selected}}
            with mock.patch.object(backend, '_codex_request',
                                   side_effect=lambda request, directory, timeout, adapter: request):
                result = backend.request('gpt_vln/model.py', payload, root)
            self.assertEqual(len(result['frames']), 9)
            self.assertEqual(result['supplied_history_frame_ids'], selected)
            self.assertTrue(all('HISTORICAL FRONT' in frame['label']
                                for frame in result['frames'][1:]))
            self.assertNotIn('front 1', str(result['frames']))
            self.assertNotIn('observed_semantic_map', result['context'])
            self.assertTrue(result['stateless_navigation'])


if __name__ == '__main__':
    unittest.main()
