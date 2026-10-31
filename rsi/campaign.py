"""Failure-driven code RSI for the GPT6-VLN navigation runtime."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).resolve()
SCHEMA = {"type": "object", "additionalProperties": False,
    "properties": {"diagnosis": {"type": "string"}, "candidates": {"type": "array",
        "minItems": 3, "maxItems": 3, "items": {"type": "object",
        "additionalProperties": False, "properties": {
            key: {"type": "string"} for key in ("category", "mechanism", "evidence",
            "competing_explanation", "skill_instruction", "prediction",
            "control_risk", "falsifier", "patch")},
        "required": ["category", "mechanism", "evidence",
            "competing_explanation", "skill_instruction", "prediction",
            "control_risk", "falsifier", "patch"]}},
    "required": ["diagnosis", "candidates"]}}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def register(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.dataset, 'rt') as handle:
        episodes = json.load(handle)['episodes']
    by_scene = {}
    for row in episodes:
        by_scene.setdefault(row['scene_id'], []).append(row)
    import random
    rng = random.Random(args.seed)
    scenes = sorted(by_scene)
    rng.shuffle(scenes)
    for rows in by_scene.values():
        rng.shuffle(rows)
    tasks = []
    while len(tasks) < args.tasks:
        for scene in scenes:
            if by_scene[scene] and len(tasks) < args.tasks:
                row = by_scene[scene].pop()
                tasks.append({'id': str(row['episode_id']), 'scene': scene,
                              'instruction': row['instruction']['instruction_text']})
        if not any(by_scene.values()) and len(tasks) < args.tasks:
            raise ValueError('Insufficient TRAIN episodes')
    registration = {'split': 'train', 'seed': args.seed, 'tasks': tasks,
        'rounds': args.rounds, 'candidates_per_round': 3,
        'base_source_sha256': {p: digest(ROOT / p) for p in ('eval.py', 'gpt_vln/model.py',
            'gpt_vln/runtime.py', 'gpt_vln/depth_preflight.py')},
        'campaign_source_sha256': digest(SOURCE),
        'dataset_sha256': digest(args.dataset),
        'rule': 'same TRAIN matrix for parent and all candidates; no combination'}
    target = root / 'registration.json'
    if target.exists() and read(target) != registration:
        raise ValueError('Campaign registration changed')
    write(target, registration)
    write(root / 'tasks.json', {'split': 'train', 'tasks': tasks})
    return root, registration


def source_copy(root, name, parent=None):
    target = root / 'versions' / name
    if target.exists():
        return target
    origin = root / 'versions' / parent if parent else ROOT
    shutil.copytree(origin, target, ignore=shutil.ignore_patterns('.git', 'outputs', '__pycache__',
        '.venv', '.runtime', 'rsi', 'experiments'))
    return target


def apply_candidate(root, name, parent, proposal):
    project = source_copy(root, name, parent)
    patch = root / 'proposals' / f'{name}.patch'
    if patch.exists() and patch.read_text() != proposal['patch']:
        raise ValueError(f'{name} proposal changed')
    if not patch.exists():
        patch.parent.mkdir(parents=True, exist_ok=True)
        patch.write_text(proposal['patch'])
    paths = set()
    for line in proposal['patch'].splitlines():
        if line.startswith(('--- a/', '+++ b/')):
            paths.add(line[6:])
    if not paths or any(not path.startswith('gpt_vln/') and path != 'eval.py'
                        for path in paths):
        raise ValueError('Candidate patch may change only the navigation runtime')
    applied = root / 'versions' / name / 'patch_applied.json'
    if applied.exists():
        if read(applied) != {'patch_sha256': digest(patch), 'parent': parent}:
            raise ValueError(f'{name} applied patch identity changed')
        return project
    check = subprocess.run(['git', 'apply', '--check', str(patch)], cwd=project,
        capture_output=True, text=True)
    if check.returncode:
        raise RuntimeError(f'{name} patch invalid: {check.stderr}')
    subprocess.run(['git', 'apply', str(patch)], cwd=project, check=True)
    subprocess.run([sys.executable, '-m', 'compileall', '-q', str(project/'gpt_vln')], check=True)
    write(applied, {'patch_sha256': digest(patch), 'parent': parent})
    return project


def run_episode(root, version, task, gpu):
    project = root / 'versions' / version
    output = root / 'runs' / version
    target = output / f"codex_pose_only_climb_free_{task['id']}_d0" / 'run/result.json'
    if target.exists():
        return read(target)
    env = os.environ.copy()
    env['VLN_WORKSPACE'] = str(Path(env.get('VLN_WORKSPACE', ROOT.parent)).resolve())
    env['CODEX_HOME'] = str(Path(env.get('CODEX_HOME', Path.home() / '.codex')).resolve())
    cmd = [sys.executable, str(project / 'eval.py'), '--registration', str(root / 'tasks.json'),
        '--episode', task['id'], '--provider', 'codex', '--decisions', '0', '--chunk-end', '400',
        '--action-horizon', '4_to_3', '--gpu', str(gpu), '--output-root', str(output)]
    log = root / 'logs' / version / (task['id'] + '.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a') as handle:
        result = subprocess.run(cmd, cwd=project, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode or not target.exists():
        raise RuntimeError(f"Episode {task['id']} did not finish; inspect {log}")
    return read(target)


def sweep(root, version, tasks, gpu):
    outcomes = {}
    for task in tasks:
        row = run_episode(root, version, task, gpu)
        outcomes[task['id']] = {'success': float(row['success']), 'spl': float(row['spl']),
            'decisions': row['decisions'], 'steps': row['steps'],
            'processed_tokens': row.get('processed_tokens'),
            'run_dir': str(root / 'runs' / version / f"codex_pose_only_climb_free_{task['id']}_d0" / 'run')}
        write(root / 'results' / (version + '.json'), outcomes)
    return outcomes


def author_context(root, version, tasks, rows, round_number):
    prior = root / 'ledger.json'
    previous = read(prior) if prior.exists() else []
    cases = []
    for task in tasks:
        result = rows[task['id']]
        run = Path(result['run_dir'])
        trace = read(run / 'trajectory.partial.json')
        timeline = [{'decision': r['decision'], 'frame': r['step'],
            'stage': r['output'].get('stage'), 'skill': r['output'].get('skill'),
            'note': r['output'].get('note', '')[:150],
            'forward_attempts': sum(a['action'] == 1 for a in r['execution']['actual']),
            'measured_translation_m': round(sum(a.get('measured_translation_m', 0)
                for a in r['execution']['actual'] if a['action'] == 1), 3)} for r in trace]
        cases.append({'id': task['id'], 'instruction': task['instruction'],
            'outcome': result, 'timeline': timeline,
            'trace': str(run / 'trajectory.partial.json'),
            'decisions_dir': str(run / 'decisions'), 'images_dir': str(run / 'images')})
    parent = root / 'versions' / version
    return ('You are the offline RSI author. Group ALL current-parent TRAIN failures by '
        'shared observable mechanism; include successful controls and competing causes. '
        'Inspect raw decision inputs, executed actions, and images on demand using the indexed '
        'paths. Produce EXACTLY THREE distinct independent code patches in ONE JSON response. '
        'Each patch starts from the SAME parent and changes ONE general mechanism. Do not use '
        'hidden goal, reference route, navmesh, GT room labels, or held-out feedback online. '
        'Keep GPT-6 Astra, four current RGB views, automatic eight historical panoramas, '
        'simulator own pose, visited-node table, action horizon and scorer fixed. Preserve '
        'node table; do not add map image or map rendering. Give evidence, competing cause, '
        'prediction, control risk and falsifier. No episode-specific rules. Do not claim gain. '
        'Return only JSON in the supplied schema.\n' +
        'ROUND=' + str(round_number) + '\nPRIOR=' + json.dumps(previous, ensure_ascii=False) +
        '\nPARENT_MODEL=' + (parent / 'gpt_vln/model.py').read_text() +
        '\nPARENT_RUNTIME=' + (parent / 'gpt_vln/runtime.py').read_text() +
        '\nALL_TRAIN_CASES=' + json.dumps(cases, ensure_ascii=False))


def propose(root, version, tasks, rows, round_number):
    path = root / 'proposals' / f'round_{round_number:02d}.json'
    if path.exists():
        return read(path)
    schema = root / 'author_schema.json'
    write(schema, SCHEMA)
    prompt = author_context(root, version, tasks, rows, round_number)
    raw = root / 'proposals' / f'round_{round_number:02d}.codex.json'
    cmd = ['codex', 'exec', '--sandbox', 'read-only', '--skip-git-repo-check',
        '--output-schema', str(schema), '--output-last-message', str(raw), '-']
    result = subprocess.run(cmd, cwd=ROOT, input=prompt, text=True, capture_output=True, timeout=1800)
    if result.returncode or not raw.exists():
        raise RuntimeError(f'RSI author failed: {result.stderr[-500:]}')
    batch = read(raw)
    if len(batch.get('candidates', [])) != 3 or len({c['category'] for c in batch['candidates']}) != 3:
        raise ValueError('Author must return three distinct candidates')
    write(path, batch)
    return batch


def compare(parent, candidate):
    ids = list(parent)
    repaired = [i for i in ids if not parent[i]['success'] and candidate[i]['success']]
    regressed = [i for i in ids if parent[i]['success'] and not candidate[i]['success']]
    return {'parent_successes': sum(bool(parent[i]['success']) for i in ids),
        'candidate_successes': sum(bool(candidate[i]['success']) for i in ids),
        'repairs': repaired, 'regressions': regressed, 'net': len(repaired)-len(regressed)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True, help='R2R TRAIN json.gz')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tasks', type=int, default=50)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20261031)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.rounds <= 5 or args.tasks < 3:
        parser.error('Use 1–5 rounds and at least three tasks')
    root, registration = register(args)
    if args.prepare_only:
        print(json.dumps({'state': 'prepared', 'tasks': len(registration['tasks']),
            'rounds': registration['rounds'], 'candidate_count': 3}));return
    tasks = registration['tasks']
    source_copy(root, 'v0')
    parent = 'v0'
    ledger = read(root / 'ledger.json') if (root / 'ledger.json').exists() else []
    for round_number in range(1, args.rounds + 1):
        parent_rows = sweep(root, parent, tasks, args.gpu)
        if len(ledger) >= round_number:
            row = ledger[round_number-1]
            if row['parent'] != parent:raise ValueError('Version lineage changed')
            parent = row['selected'];continue
        batch = propose(root, parent, tasks, parent_rows, round_number)
        candidates = []
        # Every proposal is already saved before evaluating the first candidate.
        for index, proposal in enumerate(batch['candidates'], 1):
            name = f'r{round_number:02d}_c{index:02d}'
            apply_candidate(root, name, parent, proposal)
            rows = sweep(root, name, tasks, args.gpu)
            verdict = compare(parent_rows, rows)
            write(root / 'results' / f'{name}_paired.json', verdict)
            candidates.append({'id': name, 'category': proposal['category'], 'verdict': verdict})
        viable = [c for c in candidates if c['verdict']['net'] > 0]
        best = max(viable, key=lambda c: (c['verdict']['net'],c['verdict']['candidate_successes']),
                   default=None)
        selected = best['id'] if best else parent
        row = {'round': round_number, 'parent': parent, 'candidates': candidates,
               'selected': selected, 'note': 'development selection only; verifier required before library promotion'}
        ledger.append(row);write(root / 'ledger.json', ledger)
        parent = selected
    print(json.dumps({'selected_development_version': parent, 'rounds': len(ledger),
        'admitted': False, 'reason': 'independent verifier and unseen-check not run'}))


if __name__ == '__main__':
    main()
