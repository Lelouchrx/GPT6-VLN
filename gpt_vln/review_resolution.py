


def pending_route_review(latest, addressed=False):
    if latest is None or addressed:
        return None
    findings = latest.get('prediction_errors') or []
    route_error = latest.get('route_error_since_previous_review', 'unknown')
    direction = latest.get('direction_assessment', 'unknown')
    if not (route_error in ('possible_wrong_branch', 'confirmed_wrong_branch')
            or findings or direction == 'possibly_wrong'):
        return None
    effective = route_error
    if effective == 'no_error_found' and (findings or direction == 'possibly_wrong'):
        effective = 'possible_wrong_branch'
    pending = {key: latest.get(key) for key in (
        'route_error_since_previous_review', 'stage_progress', 'progress_hypothesis',
        'prediction_errors', 'frames', 'suggested_return_anchor_frame', 'next_check',
        'route',
        'checked_before_decision')}
    pending['route_error_since_previous_review'] = effective
    pending['original_review_verdict'] = route_error
    pending['pending_reason'] = 'Prediction errors or a possible-wrong route assessment require explicit reconciliation.'
    pending['repeated_prediction_errors'] = repeated_prediction_errors(findings)
    return pending


def repeated_prediction_errors(findings):

    cues = ('repeated', 'consecutive', 'three times', 'four times',
            'five times', 'six times', 'three decisions', 'four decisions')
    return any(any(cue in str(item).lower() for cue in cues) for item in findings)


def validate_review_resolution(review, response, output, allowed_frames, current_frame):
    resolution = response['resolution']
    cited = set(response['evidence_frames'])
    if not cited <= set(allowed_frames):
        raise ValueError('Review response cites a frame not currently supplied or retrieved')
    if not response['current_test'].strip():
        raise ValueError('Review response must state the current visual test')
    if review is None:
        if resolution != 'not_applicable':
            raise ValueError('No route review is pending; use not_applicable')
        return
    severity = review['route_error_since_previous_review']
    if resolution == 'not_applicable':
        raise ValueError('A route-review warning is pending and must be explicitly resolved')
    unfinished_return = review.get('node_return_in_progress')
    if unfinished_return is not None:
        if (resolution != 'backtrack_to_node'
                or response['map_node_id'] != unfinished_return['node_id']):
            raise ValueError('Unfinished map-node return must continue to the same visited node')
    if output['skill'] == 'stop' and resolution != 'continue_with_counterevidence':
        raise ValueError('Cannot STOP before resolving the pending route warning')
    if severity == 'confirmed_wrong_branch' and resolution not in (
            'backtrack_to_node', 'inspect_suspected_branch'):
        raise ValueError('Confirmed wrong branch requires backtracking or inspecting the disputed branch')
    if review.get('repeated_prediction_errors') and resolution == 'continue_with_counterevidence':
        raise ValueError('Repeated failed predictions require a short inspection or visited-node return')
    operation = output['map_operation']
    if resolution == 'backtrack_to_node':
        if (operation != 'navigate_to_node'
                or output['map_node_id'] != response['map_node_id']):
            raise ValueError('Backtrack response must navigate to its selected visited map node')
    elif operation == 'navigate_to_node':
        raise ValueError('Map-node navigation must be declared as backtrack_to_node')
    if resolution == 'inspect_suspected_branch' and (
            output['skill'] != 'move' or not output['actions']):
        raise ValueError('Branch inspection requires an observation-producing movement action')
    if resolution == 'inspect_suspected_branch':
        forwards = [action['centimeters'] for action in output['actions']
                    if action['type'] == 'forward']
        if not forwards or sum(forwards) > 50:
            raise ValueError('Review branch inspection must use at most 50 cm forward')
        if review.get('inspection_count', 0) >= 2:
            raise ValueError('Two inspections failed to resolve this review; return to a visited node')
    if resolution == 'continue_with_counterevidence':
        if review.get('route') == 'contradicted' and not review.get('followup_after_inspection'):
            raise ValueError('Contradicted route requires a branch inspection or visited-node return first')
        if severity == 'confirmed_wrong_branch' or not cited:
            raise ValueError('Continuing requires a non-confirmed warning and supplied counterevidence frames')
        if current_frame not in cited:
            raise ValueError('Continuing after a review requires current-frame counterevidence')


def review_remains_open(review, response):

    return review is not None and response['resolution'] == 'inspect_suspected_branch'
