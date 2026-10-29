import base64
from pathlib import Path
import tempfile
import unittest

from gpt_vln.history import uniform_history_indices
from gpt_vln.model import history_images


class UniformHistoryTest(unittest.TestCase):
    def test_matches_original_uniform_strategy_across_history_sizes(self):
        for total in (0, 1, 7, 8, 9, 20, 101):
            for count in (0, 1, 4, 8):
                with self.subTest(total=total, count=count):
                    selected = uniform_history_indices(total, count)
                    self.assertEqual(len(selected), min(total, count))
                    self.assertEqual(selected, sorted(set(selected)))
                    if selected:
                        self.assertEqual(selected[0], 0)
                    if len(selected) > 1:
                        self.assertEqual(selected[-1], total - 1)
        self.assertEqual(uniform_history_indices(20), [0, 2, 5, 8, 10, 13, 16, 19])

    def test_negative_budget_or_history_is_rejected(self):
        for total, count in ((-1, 8), (20, -1)):
            with self.assertRaises(ValueError):
                uniform_history_indices(total, count)

    def test_actual_images_use_full_chronology_despite_keyframe_bias(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = {}
            # Sparse frame IDs represent primitive-step IDs, not history positions.
            for decision in reversed(range(20)):
                frame = decision * 7
                path = root / f'{frame}.jpg'
                path.write_bytes(f'frame {frame}'.encode())
                archive[str(frame)] = {'decision': decision, 'path': str(path)}
            payload = {'memory_catalog': {
                'root': directory, 'archive': archive,
                'keyframe_ids': [119, 126], 'uniform_frame_ids': [0],
            }}
            selected = history_images(payload)
            self.assertEqual([frame for frame, _, _ in selected],
                             [0, 14, 35, 56, 70, 91, 112, 133])
            self.assertEqual([base64.b64decode(data).decode() for _, _, data in selected],
                             [f'frame {frame}' for frame, _, _ in selected])
            payload['memory_catalog']['archive'] = {
                str(frame): archive[str(frame)] for frame in (14, 0, 7)}
            self.assertEqual([frame for frame, _, _ in history_images(payload)], [0, 7, 14])

    def test_selected_image_must_remain_inside_registered_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registered = root / 'archive'
            registered.mkdir()
            outside = root / 'outside.jpg'
            outside.write_bytes(b'outside')
            payload = {'memory_catalog': {
                'root': str(registered),
                'archive': {'0': {'decision': 0, 'path': str(outside)}},
            }}
            with self.assertRaisesRegex(ValueError, 'outside the registered archive'):
                history_images(payload)


    def test_navigation_attaches_uniform_images_and_preserves_astra(self):
        import json
        import os
        from unittest import mock
        from gpt_vln.model import navigate_cli
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = {}
            for decision in range(20):
                path = root / f'{decision}.jpg'
                path.write_bytes(f'image {decision}'.encode())
                archive[str(decision)] = {'decision': decision, 'path': str(path)}
            payload = {'provider': 'codex', 'context': {},
                       'frames': [{'label': 'CURRENT', 'data': base64.b64encode(b'current').decode()}],
                       'memory_catalog': {'root': str(root), 'archive': archive}}
            def fake_run(command, **kwargs):
                self.assertEqual(command[command.index('--model') + 1], 'gpt-6-astra')
                paths = [Path(command[i + 1]) for i, arg in enumerate(command) if arg == '--image']
                self.assertEqual([path.read_bytes() for path in paths],
                                 [b'current'] + [f'image {i}'.encode() for i in [0, 2, 5, 8, 10, 13, 16, 19]])
                self.assertIn('uniformly sampled', kwargs['input'])
                self.assertIn('No memory tools are available', kwargs['input'])
                Path(command[command.index('--output-last-message') + 1]).write_text(json.dumps({'skill': 'stop'}))
                return mock.Mock(returncode=0, stdout=json.dumps({'type': 'thread.started', 'thread_id': 'test-session'}), stderr='')
            with mock.patch.dict(os.environ, {'NAV_CODEX_BIN': 'codex'}), \
                 mock.patch('gpt_vln.model.subprocess.run', side_effect=fake_run):
                response = navigate_cli(payload)
            self.assertNotIn('error', response)
            self.assertEqual(response['retrieved_frame_ids'], [0, 2, 5, 8, 10, 13, 16, 19])
            self.assertEqual(response['history_strategy'], 'uniform')
            self.assertEqual(response['history_limit'], 8)
            self.assertEqual(response['model'], 'gpt-6-astra')
