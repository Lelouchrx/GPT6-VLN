
import json
import os
from pathlib import Path
import subprocess
import sys
from PIL import Image

from gpt_vln.storage import CampaignHold, immutable, read
from gpt_vln.runtime import RGBBackend as BaseBackend


class RGBBackend(BaseBackend):
    def role_request(self, role, body, directory, images=None):
        if role == 'periodic_route_review' and images and len(images) == 2:
            recent = body.get('actual_recent_transitions') or []
            if len(recent) >= 4:
                indices = [round(i * (len(recent) - 1) / 3) for i in range(4)]
                source = Image.open(images[0]['source']).convert('RGB')
                sheet = Image.new('RGB', (768, 636), (23, 28, 34))
                for slot, index in enumerate(indices):
                    x, y = (index % 4) * 384, (index // 4) * 318
                    sheet.paste(source.crop((x, y, x + 384, y + 318)),
                                ((slot % 2) * 384, (slot // 2) * 318))
                directory = Path(directory)
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / 'four_sampled_views.jpg'
                sheet.save(path, 'JPEG', quality=85)
                images = [dict(images[0], source=str(path),
                               label='Four uniformly sampled chronological four-view panoramas'),
                          images[1]]
                body = dict(body, available_frame_ids=sorted({
                    recent[index]['frame'] for index in indices}
                    | {images[1]['frame']}))
        return super().role_request(role, body, directory, images)

    def request(self, module, payload, directory, timeout=600):
        if payload.get('role') == 'periodic_route_review':
            return self._codex_request(payload, directory, timeout,
                                       'gpt_vln/model_reviewer.py')
        if module != 'gpt_vln/model.py':
            return super().request(module, payload, directory, timeout)
        if self.config.get('no_semantic_map'):
            payload = dict(payload, no_semantic_map=True)
        if self.config.get('execution_feedback_candidate'):
            payload = dict(payload, execution_feedback_candidate=True)
        if self.config.get('adaptive_batching'):
            payload = dict(payload, adaptive_batching=True)
        return self._codex_request(payload, directory, timeout,
                                   'gpt_vln/model.py')

    def _codex_request(self, payload, directory, timeout, adapter):
        directory = Path(directory)
        immutable(directory / 'input.json', payload)
        response_path = directory / 'response_0.json'
        if response_path.exists():
            response = read(response_path)
        else:
            root = Path(__file__).resolve().parents[1]
            sdk = os.environ.get('NAV_MODEL_PYTHON') or str(
                root.parent / 'miniconda3/envs/claude-agent-sdk/bin/python')
            if not Path(sdk).is_file():
                sdk = sys.executable
            env = os.environ.copy()
            env['PYTHONPATH'] = str(root)
            env.setdefault('CODEX_HOME', str(Path.home() / '.codex'))
            process = subprocess.run(
                [sdk, str(root / adapter)],
                input=json.dumps(payload), text=True, capture_output=True,
                timeout=timeout, env=env)
            try:
                response = json.loads(process.stdout.splitlines()[-1])
            except (IndexError, ValueError):
                response = {'error': f'Codex adapter exit {process.returncode}: '
                            f'{process.stderr[-500:]}', 'usage_unknown': True}
            from gpt_vln.files import write
            write(response_path, response)
        if response.get('error') or not response.get('output'):
            raise CampaignHold(str(response.get('error') or 'Missing Codex navigation output'))
        return response
