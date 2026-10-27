
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def history_images(payload, limit=4):
    catalog = payload.get('memory_catalog') or {}
    archive = catalog.get('archive') or {}
    root = Path(catalog.get('root', '/nonexistent')).resolve()
    ids = []
    for value in reversed(catalog.get('keyframe_ids') or []):
        if value not in ids:
            ids.append(value)
        if len(ids) >= 2:
            break
    uniform = catalog.get('uniform_frame_ids') or []
    if uniform:
        for value in (uniform[0], uniform[len(uniform) // 2]):
            if value not in ids:
                ids.append(value)
    selected = []
    for frame in ids[:limit]:
        item = archive.get(str(frame))
        if item is None:
            continue
        path = Path(item['path']).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Historical image outside the registered archive')
        selected.append((frame, item, base64.b64encode(path.read_bytes()).decode()))
    return selected


def without_semantic_map(context):
    context = json.loads(json.dumps(context))
    observed = context.get('observed_semantic_map')
    if isinstance(observed, dict):
        observed.pop('room_stage_hypotheses', None)
        observed.pop('unnamed_node_notice', None)
        observed.pop('unnamed_places', None)
        for node in observed.get('nodes', []):
            for key in ('name', 'caption', 'annotation_anchor', 'annotations'):
                node.pop(key, None)
    for key in ('keyframe_index', 'uniform_history_index'):
        for entry in context.get(key, []):
            if isinstance(entry, dict):
                entry.pop('description_hypothesis', None)
                entry.pop('description', None)
    return context


def recent_execution_evidence(payload):
    root = Path((payload.get('memory_catalog') or {}).get('root', '/nonexistent'))
    path = root / 'trajectory.partial.json'
    if not path.is_file():
        return None
    rows = json.loads(path.read_text())[-6:]
    attempts = []
    for row in rows:
        output = row.get('output') or {}
        actual = (row.get('execution') or {}).get('actual') or []
        forward = [item for item in actual if item.get('action') == 1]
        if not forward:
            continue
        attempts.append({'decision': row['decision'], 'maneuver': output.get('maneuver'),
                         'actual_forward_m': round(sum(float(item.get('measured_translation_m') or 0)
                                                       for item in forward), 3),
                         'height_change_m': round(sum(float(item.get('gt_height_delta_m') or 0)
                                                      for item in forward), 3)})
    return attempts


def action_pacing(context):
    if ((context.get('last_execution') or {}).get('blocked')
            or context.get('candidate_conflicts')
            or context.get('pending_node_return')):
        return ('LOCAL CHECK: The last motion was blocked or a branch/return is unresolved. '
                'Choose a short 25 cm observation-producing forward action if movement is needed.')
    return ('ADAPTIVE FLAT-GROUND BATCHING: On a visually clear straight corridor with no '
            'nearby doorway, branch, stair, target or obstacle, prefer one 75 or 100 cm '
            'forward command so several 25 cm primitives execute before the next model '
            'decision. If the visible corridor is shorter, choose 50 cm. Near a doorway, '
            'branch, target, obstacle or any stair, choose 25 cm and reobserve. '
            'For climb_up/down keep the existing mandatory single 25 cm step. '
            'Use only visible evidence; unknown space is not clear. You choose the action.')


def navigation_prompt(base_prompt, no_semantic_map):
    if not no_semantic_map:
        return base_prompt
    return re.sub(r'ROOM STAGE MEMORY:.*?(?=STABLE TASK-ENDPOINT ANCHOR:)',
                  '', base_prompt, flags=re.DOTALL)


def cli_command(payload, image_paths, output_path, model='gpt-6-sol'):
    codex = os.environ.get('NAV_CODEX_BIN') or shutil.which('codex')
    if not codex:
        raise FileNotFoundError('Codex CLI not found; set NAV_CODEX_BIN')
    session = payload.get('session_id')
    if session:
        command = [codex, 'exec', 'resume', '--model', model,
                   '--skip-git-repo-check', '--ignore-rules',
                   '--json', '--output-last-message', str(output_path)]
        for path in image_paths:
            command.extend(['--image', str(path)])
        command.extend([session, '-'])
    else:
        command = [codex, 'exec', '--model', model, '--sandbox', 'read-only',
                   '--ignore-rules', '--skip-git-repo-check',
                   '--json', '--output-last-message', str(output_path)]
        for path in image_paths:
            command.extend(['--image', str(path)])
        command.append('-')
    return command


def navigate_cli(payload, model='gpt-6-sol', timeout=600):
    try:
        from gpt_vln.model_contract import PROMPT
        from gpt_vln.memory_model import runner

        selected = history_images(payload)
        no_semantic_map = payload.get('no_semantic_map', False)
        context = without_semantic_map(payload['context']) if no_semantic_map else payload['context']
        prompt = (navigation_prompt(PROMPT, no_semantic_map)
                  + '\nOnly use attached images and public navigation context. '
                  'Historical images retain their original heading. No memory tools are available. '
                  'Set history_panorama_ids=[]. Return exactly one JSON object, with no Markdown. '
                  'review_response.current_test must be a nonempty sentence even when '
                  'review_response.resolution is not_applicable; state the current visual check. '
                  'If context.action_contract_error exists, correct that rejected output. '
                  'It must satisfy this JSON schema; the runtime rejects invalid output:\n'
                  + json.dumps(runner.nav.NAV, separators=(',', ':')) + '\n\n'
                  + json.dumps(context, separators=(',', ':')))
        if no_semantic_map:
            prompt += ('\nThe supplied map contains geometry and visited-node connectivity only. '
                       'Do not infer room identity from node IDs. Return room_stage_hypothesis '
                       'with relation=none, node_id=null, room="", instruction_stage=null, '
                       'evidence_frame=null. Do not annotate nodes: map_operation must be '
                       'none or navigate_to_node. Visited-node return remains available.')
        if payload.get('execution_feedback_candidate'):
            attempts = recent_execution_evidence(payload)
            if attempts:
                prompt += ('\nEXECUTION EVIDENCE (measured, not a route verdict): '
                           + json.dumps(attempts, separators=(',', ':')))
                prompt += ('\nIf repeated stair attempts near the same place made no net height '
                           'gain, change the approach position before another climb. '
                           'A brief rise followed by equal descent is not stair progress. '
                           'Near a suspected final doorway, use one short observation-producing '
                           'move and reassess the doorway relationship. Do not claim arrival '
                           'from a blocked action. You choose the action and STOP.')
        pacing = action_pacing(context) if payload.get('adaptive_batching') else None
        if pacing:
            prompt += '\n' + pacing
        images = [(frame['label'], frame['data']) for frame in payload.get('frames', [])]
        for frame_id, item, data in selected:
            images.append((f'HISTORICAL frame {frame_id}: '
                           + ('' if no_semantic_map else item.get('description', 'unverified place hypothesis')), data))
        with tempfile.TemporaryDirectory(prefix='vln_codex_') as directory:
            directory = Path(directory)
            output = directory / 'output.json'
            paths = []
            for index, (label, data) in enumerate(images):
                path = directory / f'image_{index:02d}.jpg'
                path.write_bytes(base64.b64decode(data, validate=True))
                paths.append(path)
                prompt += f'\nAttached image {index}: {label}'
            command = cli_command(payload, paths, output, model)
            env = os.environ.copy()
            env.setdefault('CODEX_HOME', str(Path.home() / '.codex'))
            process = subprocess.run(command, input=prompt, text=True,
                                     capture_output=True, timeout=timeout,
                                     cwd=directory, env=env)
            events = []
            for line in process.stdout.splitlines():
                try:
                    events.append(json.loads(line))
                except ValueError:
                    continue
            if process.returncode != 0 or not output.exists():
                detail = next((event.get('message', '') for event in reversed(events)
                               if event.get('type') == 'error'), '')
                return {'error': f'Codex CLI exit {process.returncode}: '
                        f'{detail or process.stderr[-500:]}', 'usage_unknown': True}
            answer = json.loads(output.read_text())
            if not isinstance(answer, dict):
                raise ValueError('Codex CLI did not return a navigation object')
            thread = next((event.get('thread_id') for event in events
                           if event.get('type') == 'thread.started'), None)
            session_id = thread or payload.get('session_id')
            if not session_id:
                raise ValueError('Codex CLI did not return a resumable thread ID')
            usage = next((event.get('usage') for event in reversed(events)
                          if event.get('type') == 'turn.completed'), None) or {}
            frames = [frame_id for frame_id, _, _ in selected]
            return {'output': answer, 'session_id': session_id, 'model': model,
                    'usage': usage, 'cost_usd': None, 'retrieved_frame_ids': frames,
                    'action_pacing_prompt': pacing,
                    'memory_calls': [{'tool': 'automatic_history', 'frames': frames}] if frames else []}
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
        return {'error': f'{type(exc).__name__}: {exc}', 'usage_unknown': True}


if __name__ == '__main__':
    print(json.dumps(navigate_cli(json.load(sys.stdin))))
