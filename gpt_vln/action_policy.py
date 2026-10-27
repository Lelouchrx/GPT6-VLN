
import copy
import json
import sys

from gpt_vln.actions import HORIZONS, crop_horizon
from gpt_vln.model import navigate_cli
import gpt_vln.model_contract as contract


def run(payload):
    mode = payload['horizon_mode']
    predicted, executed = HORIZONS[mode]
    original_prompt = contract.PROMPT
    original_schema = contract.runner.nav.NAV
    try:
        contract.runner.nav.NAV = copy.deepcopy(original_schema)
        contract.runner.nav.NAV['properties']['actions']['maxItems'] = predicted
        contract.PROMPT = original_prompt.replace(
            'Output skill=move or stop; actions contains 1 to 3 ordered commands for move, empty for STOP.',
            'Output skill=move or stop; MOVE predicts an ordered command horizon, STOP has actions=[].'
        ).replace(
            'Maximum THREE commands per decision, not three primitives.',
            'The horizon override below sets the predicted and executed command counts.'
        ).replace(
            'issue exactly ONE forward command of 25cm, then await returned RGB before selecting',
            'predict short 25cm forward commands; the selected prefix executes with measured checks before selecting'
        ) + (
            f'\nHORIZON OVERRIDE: MOVE predicts exactly {predicted} ordered high-level '
            f'commands; only the first {executed} will execute, then the robot '
            'reobserves. Put at least one forward command in that executed prefix. '
            'STOP is separate: skill=stop and actions=[]. On stairs every forward '
            'command is 25 cm, each turn at most 30 degrees. The executed prefix '
            'may request at most 150 cm total forward travel. The execution engine '
            'checks measured progress after every primitive and may interrupt early. '
            'This override supersedes the older one-step climb and three-command '
            'limits in the preceding prompt.')
        attempts = []
        current = payload
        for _ in range(2):
            response = navigate_cli(current)
            if response.get('error'):
                return dict(response, horizon_attempts=attempts)
            raw = response['output']
            try:
                cropped, audit = crop_horizon(raw, mode)
            except ValueError as exc:
                attempts.append({'error': str(exc), 'proposed_output': raw})
                current = dict(payload, session_id=response['session_id'],
                               context=dict(payload['context'],
                                            action_contract_error={'error': str(exc),
                                                                   'rejected_output': raw}))
                continue
            response['output'] = cropped
            response['horizon'] = audit
            response['horizon_attempts'] = attempts
            return response
        return {'error': 'Invalid predicted horizon after one repair',
                'horizon_attempts': attempts, 'usage_unknown': True}
    finally:
        contract.PROMPT = original_prompt
        contract.runner.nav.NAV = original_schema


if __name__ == '__main__':
    print(json.dumps(run(json.load(sys.stdin))))
