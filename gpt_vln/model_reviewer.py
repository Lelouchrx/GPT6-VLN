
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
            command = cli_command({}, images, output) if payload.get('provider') != 'claude' else None
            env = os.environ.copy()
            env.setdefault('CODEX_HOME', str(Path.home() / '.codex'))
            body = next((json.loads(item['text']) for item in content
                         if item['type'] == 'text' and item['text'].startswith('{')), None)
            supplied = set((body or {}).get('available_frame_ids') or [])
            attempts = []
            for attempt in range(2):
                metadata = {}
                if payload.get('provider') == 'claude':
                    from gpt_vln.claude import complete
                    metadata = complete(prompt, [('review image', base64.b64encode(path.read_bytes()).decode()) for path in images], SCHEMA)
                    if metadata.get('error'):
                        return metadata
                    raw = json.dumps(metadata['output'])
                    process = subprocess.CompletedProcess([], 0, stderr='')
                else:
                    process = subprocess.run(command, input=prompt, text=True, capture_output=True,
                                             timeout=timeout, cwd=directory, env=env)
                    raw = output.read_text() if output.exists() else ''
                try:
                    if process.returncode or not raw:
                        raise ValueError(f'Codex reviewer exit {process.returncode}: {process.stderr[-300:]}')
                    verdict = json.loads(raw)
                    if not isinstance(verdict, dict):
                        raise ValueError('Reviewer output must be an object')
                    missing = set(SCHEMA['required']) - set(verdict)
                    if missing:
                        raise ValueError('Missing reviewer fields: ' + ', '.join(sorted(missing)))
                    verdict = {key: verdict[key] for key in SCHEMA['required']}
                    if not set(verdict['frames']) <= supplied:
                        raise ValueError('Reviewer cited a frame outside supplied images')
                    anchor = verdict['suggested_return_anchor_frame']
                    if anchor is not None and (anchor not in supplied or anchor not in {row['frame'] for row in (body or {}).get('actual_recent_transitions', [])}):
                        raise ValueError('Reviewer return anchor was not supplied')
                    return {**metadata, 'output': verdict, 'usage': metadata.get('usage', {}), 'cost_usd': metadata.get('cost_usd'),
                            'format_repairs': len(attempts)}
                except (TypeError, ValueError, KeyError) as exc:
                    attempts.append({'error': str(exc), 'raw_output': raw[:12000]})
                    prompt += ('\nThe previous response did not match the requested JSON contract: '
                               + str(exc) + '. Return the complete object with exactly the required '
                               'fields, using only supplied frame IDs. No Markdown.')
            return {'error': 'Reviewer failed the output contract after one repair',
                    'attempts': attempts, 'usage_unknown': True}
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
        return {'error': f'{type(exc).__name__}: {exc}', 'usage_unknown': True}


if __name__ == '__main__':
    print(json.dumps(review(json.load(sys.stdin))))
