


def validate_stop_clauses(output):
    if output.get('skill') != 'stop':
        return
    events = output.get('events') or []
    if not events:
        raise ValueError('STOP requires the ordered instruction clauses')
    for event in events[:-1]:
        if event.get('state') not in ('passed', 'reached'):
            raise ValueError('STOP blocked: earlier instruction clause remains unresolved: '
                             + str(event.get('clause', 'unknown')))
