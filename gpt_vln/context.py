
from collections import Counter
import copy

def execution(value):
    rows=value.get('actual',[]);counts=Counter(r['action'] for r in rows)
    poses=[r.get('relative_pose') for r in rows if r.get('relative_pose')]
    return {'first_step':rows[0]['step'] if rows else None,'last_step':rows[-1]['step'] if rows else None,
        'executed_forward_primitives':counts[1], 'executed_turn_left_primitives':counts[2],
        'executed_turn_right_primitives':counts[3],'stop_executed':bool(counts[0]),
        'executed_turn_command_deg':sum(r.get('executed_turn_deg',0) for r in rows),
        'metric_translation_m':None,'metric_scale':'unknown',
        'pose_quality_counts':dict(Counter(p.get('quality','unknown') for p in poses)),
        'requested_primitives':value.get('requested_primitives'),
        'executed_primitives':value.get('executed_primitives',len(rows))}

def pose(value):
    p=value.get('last_relative_pose') or {}
    return {'source':'RGB relative-direction estimate; no true pose',
        'commanded_heading_deg':value.get('commanded_heading_deg'),
        'last_relative_yaw_deg':p.get('visual_relative_yaw_deg'),
        'translation_direction_camera':p.get('translation_direction_camera'),
        'quality':p.get('quality','unknown'),'inlier_ratio':p.get('inlier_ratio'),
        'metric_translation_m':None,'metric_scale':'unknown'}

def context(value):
    c=copy.deepcopy(value)
    c.pop('panorama_action_contract',None)
    c.pop('observation_memory',None)
    c.pop('execution_summary',None)
    c['last_execution']=execution(c.get('last_execution') or {})
    c['pose_evidence']=pose(c.get('pose_evidence') or {})
    for item in c.get('uniform8_action_history',[]):
        item['executed_after_frame']=execution(item.get('executed_after_frame') or {})
    return c
