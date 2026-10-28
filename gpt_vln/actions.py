
import copy


HORIZONS = {'4_to_3': (4, 3), '3_to_2': (3, 2)}


def crop_horizon(output, mode):
    predicted_count, executed_count = HORIZONS[mode]
    planned = output.get('actions')
    if output.get('skill') == 'stop':
        if planned != []:
            raise ValueError('STOP must have no predicted or executed movement')
        return copy.deepcopy(output), {'mode': mode, 'planned': [], 'executed': []}
    if output.get('skill') != 'move' or not isinstance(planned, list):
        raise ValueError('MOVE needs a predicted action list')
    if len(planned) != predicted_count:
        raise ValueError(f'{mode} requires exactly {predicted_count} predicted commands')
    for action in planned:
        if action.get('type') == 'forward':
            if action.get('centimeters') not in (25, 50, 75, 100):
                raise ValueError('Forward distance must be 25, 50, 75 or 100 cm')
        elif action.get('type') in ('turn_left', 'turn_right'):
            if action.get('degrees') not in (15, 30, 45, 60, 75, 90):
                raise ValueError('Turn must be in 15 degree increments up to 90')
        else:
            raise ValueError('Unknown predicted command')
    executed = planned[:executed_count]
    if not any(action['type'] == 'forward' for action in executed):
        raise ValueError('Executed prefix must contain a forward command')
    if sum(action['centimeters'] for action in executed
           if action['type'] == 'forward') > 150:
        raise ValueError('Executed prefix cannot request more than 150 cm forward')
    cropped = copy.deepcopy(output)
    cropped['actions'] = copy.deepcopy(executed)
    return cropped, {'mode': mode, 'planned': copy.deepcopy(planned),
                     'executed': copy.deepcopy(executed)}
