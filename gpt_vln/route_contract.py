


def validate_compound_plan(output, context):
    compound = context.get('compound_route')
    if compound is None:
        return
    assessment = output['compound_route_assessment']
    choice = output['route_choice']
    if assessment['intended_final_turn'] != compound['turn']:
        raise ValueError('The final turn must retain the original instruction direction')
    visible = set(context['available_frame_ids'])
    if not set(choice.get('evidence_frames', [])) <= visible:
        raise ValueError('Route choice cites a frame not supplied to this decision')
    if choice['role'] in ('switch_branch', 'finish') and not choice.get('evidence_frames'):
        raise ValueError('Route switch or finish needs a supplied panorama')
    if choice['role'] == 'switch_branch' and choice['route_support'] != 'observed':
        raise ValueError('A branch switch needs positive route evidence, not candidate uncertainty')
    if choice['role'] == 'finish' and assessment['completion'] != 'satisfied':
        raise ValueError('Cannot finish while the linked landmark-door relation is unresolved')
    if output['skill'] == 'stop' and assessment['completion'] != 'satisfied':
        raise ValueError('STOP needs an explicit judgment that the linked relation holds')
