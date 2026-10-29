"""Sequential, resumable evaluation of a pinned random-episode navigation cohort."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path, fallback=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return fallback


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, path)


def summary(rows, planned):
    completed = [row for row in rows if row['status'] == 'completed']
    errors = [row for row in rows if row['status'] != 'completed']
    successes = sum(float(row['success']) for row in completed)
    def average(key):
        values = [float(row[key]) for row in completed if row.get(key) is not None]
        return sum(values) / len(values) if values else None
    return {
        'planned_episodes': planned, 'attempted_episodes': len(rows),
        'completed_episodes': len(completed), 'operational_errors': len(errors),
        'remaining_episodes': planned - len(rows), 'successes': int(successes),
        'navigation_failures': len(completed) - int(successes),
        'success_rate_completed': successes / len(completed) if completed else None,
        'spl_mean_completed': average('spl'),
        'navigation_error_mean_m_completed': average('distance_to_goal_m'),
        'oracle_success_rate_completed': average('oracle_success'),
        'success_fraction_of_planned': successes / planned,
        'total_steps_completed': sum(row['steps'] for row in completed),
        'total_decisions_completed': sum(row['decisions'] for row in completed),
        'updated_at': now(),
    }


def error_kind(reason):
    value = reason.lower()
    if any(term in value for term in ('unauthorized', 'not logged in', 'invalid api key',
                                      'authentication', '401', 'insufficient_quota',
                                      'usage limit', 'quota exceeded', 'model_not_found', 'model not found',
                                      'not supported when using codex', 'do not have access')):
        return 'service_hold'
    if any(term in value for term in ('timeout', 'timed out', '429', 'rate limit',
                                      'rate_limit', 'too many requests', '502', '503',
                                      '504', 'connection reset', 'temporary',
                                      'temporarily', 'stream disconnected',
                                      'error sending request', 'connection closed')):
        return 'transient'
    return 'episode_error'


def progress(batch, manifest, rows, state, active=None, **extra):
    value = {'batch_id': manifest['batch_id'], 'state': state,
             'strategy': 'uniform', 'history_frames': 8,
             'split': manifest['split'], 'model': manifest['model'],
             'action_horizon': manifest['action_horizon'], 'seed': manifest['seed'],
             'summary': summary(rows, len(manifest['tasks'])),
             'active_episode': active, **extra}
    write(batch / 'progress.json', value)
    return value


def save_results(batch, manifest, rows):
    write(batch / 'results.json', rows)
    write(batch / 'summary.json', summary(rows, len(manifest['tasks'])))
    fields = ['episode_id', 'scene_id', 'status', 'success', 'spl',
              'distance_to_goal_m', 'oracle_success', 'steps', 'decisions',
              'elapsed_s', 'attempts', 'error', 'output_directory']
    temporary = batch / 'results.csv.tmp'
    with temporary.open('w', newline='') as output:
        writer = csv.DictWriter(output, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, batch / 'results.csv')


def check_source(project, manifest):
    for name, expected in manifest['source_sha256'].items():
        if hashlib.sha256((project / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Pinned batch source changed: ' + name)
    if hashlib.sha256((project / 'config/tasks.batch.json').read_bytes()).hexdigest() != manifest['registration_sha256']:
        raise RuntimeError('Pinned task registration changed')


def quarantine_transient_response(run, attempt):
    """Keep every failed response; retry only errors, never a valid model action."""
    for path in run.rglob('response_0.json'):
        response = read(path, {})
        if response.get('error') and error_kind(str(response['error'])) == 'transient':
            destination = path.with_name(f'response_transient_attempt_{attempt}.json')
            if destination.exists():
                raise RuntimeError('Transient response archive already exists: ' + str(destination))
            path.rename(destination)


def main():
    import fcntl
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=Path, required=True)
    args = parser.parse_args()
    batch = args.batch.resolve()
    manifest = read(batch / 'manifest.json')
    assert manifest and len(manifest['tasks']) == manifest['count']
    project = batch / 'project'
    lock = (batch / 'runner.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(os.getpid()))
    lock.flush()
    write(batch / 'process.json', {'pid': os.getpid(), 'started_at': now()})
    rows = read(batch / 'results.json', [])
    ids = [row['episode_id'] for row in rows]
    assert len(ids) == len(set(ids))
    assert set(ids) <= {task['id'] for task in manifest['tasks']}
    child = None

    def interrupted(signum, frame):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        progress(batch, manifest, rows, 'interrupted', signal=signum)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    progress(batch, manifest, rows, 'running')
    for number, task in enumerate(manifest['tasks'], start=1):
        if task['id'] in ids:
            continue
        check_source(project, manifest)
        output = project / 'outputs' / f"codex_pose_only_climb_free_{task['id']}_d0"
        run = output / 'run'
        reason = ''
        started = time.monotonic()
        for attempt in range(1, 5):
            active = {'index': number, 'episode_id': task['id'],
                      'scene_id': task['scene_id'], 'attempt': attempt,
                      'started_at': now(), 'output_directory': str(output)}
            print(f"[{now()}] START {number}/{manifest['count']} episode={task['id']} attempt={attempt}", flush=True)
            log_path = batch / 'logs' / f"episode_{task['id']}_attempt_{attempt}.log"
            with log_path.open('ab') as log:
                child = subprocess.Popen([
                    sys.executable, 'eval.py', '--registration', 'config/tasks.batch.json',
                    '--episode', task['id'], '--decisions', '0', '--chunk-end', '400',
                    '--provider', manifest['provider'],
                    '--action-horizon', manifest['action_horizon'], '--gpu', str(manifest['gpu']),
                ], cwd=project, stdout=log, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, start_new_session=True)
                while child.poll() is None:
                    trace = read(run / 'trajectory.partial.json', [])
                    active['completed_decisions'] = len(trace)
                    active['executed_steps'] = sum(
                        (row.get('execution') or {}).get('executed_primitives', 0) for row in trace)
                    active['elapsed_s'] = round(time.monotonic() - started, 1)
                    active['child_pid'] = child.pid
                    progress(batch, manifest, rows, 'running', active)
                    time.sleep(10)
            result = read(run / 'result.json')
            state = read(output / 'state.json', {})
            if child.returncode == 0 and result is not None and state.get('status') == 'complete':
                if not result.get('implementation_unchanged'):
                    raise RuntimeError('Episode reported changed runtime source')
                metrics = result['metrics']
                record = {'episode_id': task['id'], 'scene_id': task['scene_id'],
                    'status': 'completed', 'terminal_status': result['status'],
                    'success': result['success'], 'spl': result['spl'],
                    'distance_to_goal_m': metrics.get('distance_to_goal'),
                    'oracle_success': metrics.get('oracle_success'),
                    'steps': result['steps'], 'decisions': result['decisions'],
                    'elapsed_s': round(time.monotonic() - started, 3),
                    'attempts': attempt, 'error': '', 'output_directory': str(output),
                    'finished_at': now()}
                break
            log_tail = '\n'.join(log_path.read_text(errors='replace').splitlines()[-12:])
            reason = str(state.get('reason') or log_tail or f'Episode exit {child.returncode}')
            kind = error_kind(reason)
            if kind == 'service_hold':
                progress(batch, manifest, rows, 'service_hold', active, reason=reason)
                print('SERVICE HOLD: ' + reason, flush=True)
                return 2
            if kind == 'transient' and attempt < 4:
                quarantine_transient_response(run, attempt)
                progress(batch, manifest, rows, 'retry_wait', active, reason=reason)
                print(f'Transient service error; retrying in {30 * attempt}s: {reason}', flush=True)
                time.sleep(30 * attempt)
                continue
            record = {'episode_id': task['id'], 'scene_id': task['scene_id'],
                'status': 'operational_error', 'success': None, 'spl': None,
                'distance_to_goal_m': None, 'oracle_success': None, 'steps': 0,
                'decisions': 0, 'elapsed_s': round(time.monotonic() - started, 3),
                'attempts': attempt, 'error': reason, 'output_directory': str(output),
                'finished_at': now()}
            break
        rows.append(record)
        ids.append(task['id'])
        save_results(batch, manifest, rows)
        with (batch / 'results.jsonl').open('a') as audit:
            audit.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"[{now()}] END {number}/{manifest['count']} episode={task['id']} status={record['status']} success={record['success']}", flush=True)
        # Repeated operational failures usually indicate a shared service/runtime issue.
        if len(rows) >= 3 and all(row['status'] == 'operational_error' for row in rows[-3:]):
            progress(batch, manifest, rows, 'service_hold', reason='Three consecutive operational errors; inspect logs before resuming')
            return 2
    state = 'complete' if all(row['status'] == 'completed' for row in rows) else 'finished_with_errors'
    progress(batch, manifest, rows, state, finished_at=now())
    print(json.dumps(summary(rows, len(manifest['tasks'])), indent=2), flush=True)
    return 0 if state == 'complete' else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        parser = argparse.ArgumentParser()
        parser.add_argument('--batch', type=Path, required=True)
        args = parser.parse_args()
        batch = args.batch.resolve()
        manifest = read(batch / 'manifest.json', {})
        rows = read(batch / 'results.json', [])
        if manifest:
            progress(batch, manifest, rows, 'controller_error', reason=f'{type(error).__name__}: {error}')
        raise
