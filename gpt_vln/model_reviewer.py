
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from gpt_vln.review_model import PROMPT, SCHEMA
from gpt_vln.model import cli_command


def review(payload, timeout=600):
    try:
        if payload.get('role') != 'periodic_route_review':
            raise ValueError('Unexpected review role')
        with tempfile.TemporaryDirectory(prefix='vln_review_') as directory:
            directory = Path(directory)
            content = payload['content']
            prompt = PROMPT + '\nReturn one JSON object, no Markdown, matching this schema:\n'
            prompt += json.dumps(SCHEMA, separators=(',', ':'))
            images = []
            for item in content:
                if item['type'] == 'text':
                    prompt += '\n' + item['text']
                elif item['type'] == 'image':
                    path = directory / f'image_{len(images):02d}.jpg'
                    path.write_bytes(base64.b64decode(item['source']['data'], validate=True))
                    images.append(path)
            output = directory / 'output.json'
            command = cli_command({}, images, output)
            env = os.environ.copy()
            env.setdefault('CODEX_HOME', str(Path.home() / '.codex'))
            process = subprocess.run(command, input=prompt, text=True, capture_output=True,
                                     timeout=timeout, cwd=directory, env=env)
            if process.returncode or not output.exists():
                return {'error': f'Codex reviewer exit {process.returncode}: '
                        f'{process.stderr[-500:]}', 'usage_unknown': True}
            verdict = json.loads(output.read_text())
            if set(verdict) != set(SCHEMA['required']):
                raise ValueError('Reviewer output does not match required keys')
            body = next((json.loads(item['text']) for item in content
                         if item['type'] == 'text' and item['text'].startswith('{')), None)
            supplied = set((body or {}).get('available_frame_ids') or [])
            if not set(verdict['frames']) <= supplied:
                raise ValueError('Reviewer cited a frame outside supplied images')
            anchor = verdict['suggested_return_anchor_frame']
            if anchor is not None and anchor not in supplied:
                raise ValueError('Reviewer return anchor was not supplied')
            return {'output': verdict, 'usage': {}, 'cost_usd': None}
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
        return {'error': f'{type(exc).__name__}: {exc}', 'usage_unknown': True}


if __name__ == '__main__':
    print(json.dumps(review(json.load(sys.stdin))))
