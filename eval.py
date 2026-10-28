#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
WORKSPACE = Path(os.environ.get('VLN_WORKSPACE', ROOT.parent))
HABITAT = Path(os.environ.get('HABITAT_LAB_PATH', WORKSPACE / 'StreamVLN/deps/habitat-lab'))
for path in (HABITAT / 'habitat-lab', HABITAT / 'habitat-baselines'):
    if path.exists():
        sys.path.insert(0, str(path))

def require_habitat():
    if Env is None:
        raise RuntimeError('Habitat is unavailable') from _HABITAT_ERROR

try:
    import habitat
    from habitat import Env
    from habitat_baselines.config.default import get_config
    from gpt_vln.habitat_extensions import measures as _measures
except (ImportError, ValueError) as exc:
    habitat = None
    Env = None
    get_config = None
    _HABITAT_ERROR = exc

def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value

def main():
    parser = argparse.ArgumentParser(description='Run observed RGB-D navigation')
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--episode', required=True)
    parser.add_argument('--provider', choices=('claude', 'codex'), default='claude')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--decisions', type=int, default=20)
    parser.add_argument('--chunk-end', type=int, default=None)
    parser.add_argument('--action-horizon', choices=('free', '3_to_2', '4_to_3'), default='4_to_3')
    parser.add_argument('--memory-policy', choices=('uniform8', 'session'), default='uniform8')
    parser.add_argument('--max-usd', type=float, default=None)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.decisions < 0:
        parser.error('decisions must be nonnegative')
    if args.max_usd is not None and args.max_usd <= 0:
        parser.error('max-usd must be positive')
    boundary = args.chunk_end if args.chunk_end is not None else (args.decisions or 20)
    if boundary < 1 or (args.decisions and boundary > args.decisions):
        parser.error('chunk-end must be within the decision limit')
    registration = json.loads(args.registration.read_text())
    task = next((row for row in registration['tasks'] if str(row['id']) == args.episode), None)
    if task is None:
        parser.error('episode is absent from registration')
    from gpt_vln.actions import HORIZONS
    if args.preflight:
        print(json.dumps({'episode': args.episode, 'model': 'claude-opus-5' if args.provider == 'claude' else 'gpt-6-sol',
                          'semantic_map_input': False, 'map_input': False,
                          'own_pose_input': True, 'observed_geometry': False,
                          'visited_node_return': True,
                          'action_horizon': None if args.action_horizon == 'free'
                          else HORIZONS[args.action_horizon],
                          'max_commands': 3 if args.action_horizon == 'free' else None,
                          'critic_every_decisions': 10,
                          'memory_policy': args.memory_policy,
                          'max_usd': args.max_usd,
                          'historical_front_views': 8 if args.memory_policy == 'uniform8' else 0,
                          'scheduled_critic_calls': max(0, (boundary - 1) // 10)}))
        return
    from gpt_vln.backend import sha
    from gpt_vln.files import write
    from gpt_vln.storage import CampaignHold, BudgetHold, immutable
    from gpt_vln.navigation import RGBBackend
    config = dict(registration.get('config') or {})
    config.update(mode='navigation_' + args.action_horizon + '_' + args.memory_policy,
                  split=config.get('split', registration.get('split', 'train')),
                  route_review_interval=10, fork_decision=0, fork_frame=0,
                  no_semantic_map=True, pose_without_map=True, map_enabled=False,
                  provider=args.provider, memory_policy=args.memory_policy,
                  max_usd=args.max_usd,
                  execution_feedback_candidate=False,
                  adaptive_batching=False, horizon_mode=args.action_horizon)
    config['episode_budget'] = {**config.get('episode_budget', {}),
                                'decisions': args.decisions or 2**31 - 1,
                                'primitives': config.get('episode_budget', {}).get('primitives', 400)}
    config['runtime_hashes'] = {str(path.relative_to(ROOT)): sha(path)
        for path in (ROOT / 'gpt_vln').rglob('*.py')}
    config['runtime_hashes']['eval.py'] = sha(ROOT / 'eval.py')
    version = {'id': 'navigation', 'source': None, 'source_sha256': 'empty'}
    label = 'pose_only_climb_free' if args.action_horizon == '4_to_3' else args.action_horizon
    output = args.output_root / f'{args.provider}_{label}_{args.memory_policy}_{args.episode}_d{args.decisions}'
    output.mkdir(parents=True, exist_ok=True)
    immutable(output / 'registration.json', {'config': config, 'task': task, 'version': version})
    run = output / 'run'
    run.mkdir(exist_ok=True)
    backend = RGBBackend(output, config, args.gpu)
    backend.chunk_end = boundary
    try:
        write(output / 'state.json', {'status': 'running', 'episode': args.episode})
        result = backend.evaluate(version, task, run)
        immutable(run / 'result.json', result)
        write(output / 'state.json', {'status': 'complete', 'episode': args.episode,
                                        'decisions': result['decisions'], 'success': result['success']})
    except BudgetHold as exc:
        write(output / 'state.json', {'status': 'budget_hold', 'episode': args.episode,
                                        'reason': str(exc)})
    except CampaignHold as exc:
        write(output / 'state.json', {'status': 'resumable_hold', 'episode': args.episode,
                                        'reason': str(exc)})
    finally:
        backend.close()

if __name__ == '__main__':
    main()
