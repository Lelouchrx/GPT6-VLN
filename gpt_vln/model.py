
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from gpt_vln.history import uniform_history_indices


def history_images(payload, limit=8):
    catalog = payload.get('memory_catalog') or {}
    archive = catalog.get('archive') or {}
    root = Path(catalog.get('root', '/nonexistent')).resolve()
    ordered = sorted(((int(frame), item) for frame, item in archive.items()),
                     key=lambda row: (row[1]['decision'], row[0]))
    selected = []
    for index in uniform_history_indices(len(ordered), limit):
        frame, item = ordered[index]
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
    return ('Choose each forward distance freely from 25/50/75/100 cm and each '
            'turn from 15 to 90 degrees in 15-degree increments. Combine '
            'continuous same-heading motion into the fewest valid commands. '
            'Measured progress is checked after each execution primitive.')


def navigation_prompt(base_prompt, no_semantic_map):
    if not no_semantic_map:
        return base_prompt
    return re.sub(r'ROOM STAGE MEMORY:.*?(?=STABLE TASK-ENDPOINT ANCHOR:)',
                  '', base_prompt, flags=re.DOTALL)


def pose_only_prompt(base_prompt):
    prompt = re.sub(r'MEMORY CONTRACT:.*?(?=HORIZON OVERRIDE:|\Z)', '', base_prompt, flags=re.DOTALL)
    prompt = re.sub(r'MAP SENSOR OVERRIDE:.*?(?=GT LOCALIZATION OVERRIDE:)',
                    '', prompt, flags=re.DOTALL)
    prompt = prompt.replace('Own pose and RGB-D observed mapping are available;',
                            'Measured own pose is available;')
    return prompt + ('\nPOSE AND VISITED NODES: The current input gives your measured '
                     'own_pose relative to start and a list of visited node IDs with '
                     'recorded connections and relative return bearings. There is no '
                     'map image or map geometry in your input. Use the current four-view '
                     'RGB and historical photos to judge route identity. You may request '
                     'navigate_to_node for an existing visited node ID; the executor '
                     'follows only recorded movement. Set room_stage_hypothesis to '
                     'relation=none, node_id=null, room="", instruction_stage=null, '
                     'evidence_frame=null; do not annotate nodes.\n'
                     'POSE HISTORY is a short chronological list of visited own poses, '
                     'not a floor plan or a route to the hidden goal. Compare its '
                     'heading and height with the instruction and the actual images. '
                     'DEPTH PREFLIGHT samples a narrow forward corridor for roughly '
                     'two 25 cm steps. possible_obstacle asks for a new visual check '
                     'or shorter approach; unknown is not clear. The horizontal '
                     'depth view cannot certify low steps, glass, or room identity. '
                     'A warning around two steps ahead is advisory; only an '
                     'imminent obstacle may interrupt a forward primitive. '
                     'you still choose every subsequent movement and STOP.\n')


def cli_command(payload, image_paths, output_path, model='gpt-6-astra'):
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


def navigate_cli(payload, model='gpt-6-astra', timeout=600):
    try:
        from gpt_vln.model_contract import PROMPT
        from gpt_vln.memory_model import runner

        selected = [] if payload.get('provider') == 'claude' else history_images(payload)
        no_semantic_map = payload.get('no_semantic_map', False)
        context = without_semantic_map(payload['context']) if no_semantic_map else payload['context']
        base_prompt = navigation_prompt(PROMPT, no_semantic_map)
        if payload.get('pose_without_map'):
            base_prompt = pose_only_prompt(base_prompt)
        base_prompt += (
            '\nLANDMARK IDENTITY CHECK: Before committing to an opening or landmark '
            'that advances an instruction clause, compare the selected candidate '
            'with the strongest visible alternative in the current or attached '
            'historical panoramas. In review_response.current_test, name both by '
            'visible appearance and frame/view, or state that no alternative is '
            'visible. State the distinguishing instruction relation: approach '
            'direction, order, landmark association, or turn at the junction. '
            'A matching object category or an enlarging target supports recognition '
            'or approach, but does not distinguish route identity. A small steering '
            'correction does not by itself establish an instructed junction turn. '
            'If both candidates remain plausible, keep identity unresolved and '
            'choose a movement whose next view can distinguish them, using the '
            'existing action horizon. Do not invent an alternative or reject a '
            'candidate solely because another looks more typical. Once evidence '
            'distinguishes the candidate, proceed without redundant comparison. '
            'Record this check in existing fields; do not add output fields.\n')
        memory_prompt = (
            'Historical images retain their original heading. If a route or '
            'branch comparison needs an old image, Claude may call retrieve_history '
            'with one or two frame IDs from keyframe_index or uniform_history_index. '
            'Do not retrieve history routinely. '
            if payload.get('provider') == 'claude' else
            'Up to eight historical four-view panoramas are attached automatically, '
            'uniformly sampled from the full chronological decision history. '
            'Historical images retain their original heading. No memory tools are available. ')
        prompt = (base_prompt
                  + '\nOnly use attached images and public navigation context. '
                  + memory_prompt
                  + 'Set history_panorama_ids=[]. Return exactly one JSON object, with no Markdown. '
                  'review_response.current_test must be a nonempty sentence even when '
                  'review_response.resolution is not_applicable; state the current visual check. '
                  'If context.action_contract_error exists, correct that rejected output. '
                  'It must satisfy this JSON schema; the runtime rejects invalid output:\n'
                  + json.dumps(runner.nav.NAV, separators=(',', ':')) + '\n\n'
                  + json.dumps(context, separators=(',', ':')))
        if no_semantic_map and not payload.get('pose_without_map'):
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
        if payload.get('merge_forward_commands'):
            prompt += ('\nACTION FORMAT: Choose freely from one to three movement commands. '
                       'Each forward command is 25, 50, 75, or 100 cm; each turn is '
                       '15 to 90 degrees in 15 degree increments. Consecutive '
                       'same-heading forward commands whose total is at most 100 cm '
                       'represent one continuous movement: express them as one '
                       'forward command with that total distance. If the total '
                       'exceeds 100 cm, use the fewest valid forward commands. '
                       'Choose the distance yourself from the available observation; '
                       'there is no minimum distance or scene-specific distance rule. '
                       'STOP is separate and has actions=[].')
        pacing = action_pacing(context) if payload.get('adaptive_batching') else None
        if pacing:
            prompt += '\n' + pacing
        images = [(frame['label'], frame['data']) for frame in payload.get('frames', [])]
        for frame_id, item, data in selected:
            images.append((f'HISTORICAL frame {frame_id}: '
                           + ('' if no_semantic_map else item.get('description', 'unverified place hypothesis')), data))
        if payload.get('provider') == 'claude':
            from gpt_vln.claude import complete
            result = complete(prompt, images, runner.nav.NAV, payload.get('session_id'),
                              payload.get('memory_catalog'))
            return result
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
                    'history_strategy': 'uniform', 'history_limit': 8,
                    'memory_calls': [{'tool': 'automatic_history', 'frames': frames}] if frames else []}
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
        return {'error': f'{type(exc).__name__}: {exc}', 'usage_unknown': True}


if __name__ == '__main__':
    print(json.dumps(navigate_cli(json.load(sys.stdin))))
