


def route_progress_feedback(previous, execution, current_graph):
    if not previous:
        return None
    output = previous['output']
    choice = output.get('route_choice') or {}
    candidate_id = choice.get('target_candidate_id')
    old_graph = (previous.get('context') or {}).get('predicted_semantic_graph') or {}

    def candidate(graph):
        return next((row for row in graph.get('predicted_entrances', [])
                     if row['candidate_id'] == candidate_id), None)

    old = candidate(old_graph) if candidate_id else None
    current = candidate(current_graph) if candidate_id else None
    old_geometry = (old or {}).get('geometry') or {}
    new_geometry = (current or {}).get('geometry') or {}
    distance_change = None
    if (old_geometry.get('status') == 'tracked' and new_geometry.get('status') == 'tracked'
            and old_geometry.get('source_depth_m', 99) <= 4.5
            and new_geometry.get('source_depth_m', 99) <= 4.5):
        distance_change = round(new_geometry['distance_xz_m'] - old_geometry['distance_xz_m'], 2)
    trend = ('unknown' if distance_change is None else 'toward' if distance_change < -.2
             else 'away' if distance_change > .2 else 'stable')
    actions = execution.get('actual', [])
    measured_translation = round(sum(row.get('measured_translation_m', 0) or 0 for row in actions), 2)
    expected = (output.get('expected_effect') or {}).get('expected_distance_change', 'unknown')
    comparison = ('unknown' if trend == 'unknown' or expected == 'unknown'
                  else 'supported' if expected == 'decrease' and trend == 'toward'
                  else 'contradicted' if expected == 'decrease' and trend == 'away'
                  else 'unknown')
    return {'previous_choice': choice.get('role'), 'candidate_id': candidate_id,
            'expected_distance_change': expected, 'measured_distance_change_m': distance_change,
            'measured_trend': trend, 'expected_effect_comparison': comparison,
            'measured_translation_m': measured_translation,
            'candidate_status': (current or {}).get('status', 'unknown'),
            'execution_interrupted': bool(execution.get('critic_interrupted')),
            'limitation': 'Geometry tests movement relative to a provisional anchor; it cannot identify the instructed door or prove route completion.'}
