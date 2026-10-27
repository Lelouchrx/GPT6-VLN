import base64
import hashlib
import json
from pathlib import Path
from gpt_vln.storage import digest

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class LiveBackend:
    def __init__(self, root, config, gpu):
        self.root = Path(root)
        self.config = config
        self.gpu = gpu
        self.env = None
        self.episodes = {}

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    def run_identity(self, version, task):
        return digest({'task': task, 'source': version['source_sha256'],
                       'code': self.config['runtime_hashes'],
                       'budget': self.config['episode_budget'],
                       'mode': self.config['mode']})

    def code_text(self, version):
        source = version.get('source')
        if source is None:
            return None
        text = Path(source).read_text()
        if hashlib.sha256(text.encode()).hexdigest() != version['source_sha256']:
            raise ValueError('Navigation program changed')
        return text

    def runtime_unchanged(self):
        return all(sha(ROOT / name) == expected
                   for name, expected in self.config['runtime_hashes'].items())

    def role_request(self, role, body, directory, images=None):
        content = [{'type': 'text', 'text': json.dumps(body)}]
        for item in images or []:
            content.extend([{'type': 'text', 'text': item['label']},
                            {'type': 'image', 'source': {
                                'type': 'base64', 'media_type': 'image/jpeg',
                                'data': base64.b64encode(Path(item['source']).read_bytes()).decode()}}])
        return self.request('gpt_vln/model_reviewer.py',
                            {'role': role, 'content': content}, directory, 600)['output']
