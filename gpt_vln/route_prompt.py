
import asyncio
import copy
import json
import os
import sys

import gpt_vln.entrance_prompt as navigation

runner = navigation.runner
runner.nav.NAV = copy.deepcopy(runner.nav.NAV)
runner.nav.NAV['properties']['route_choice'] = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'role': {'type': 'string', 'enum': ['follow_current_route', 'inspect_candidate', 'switch_branch', 'finish']},
        'target_candidate_id': {'type': ['string', 'null']},
        'route_support': {'type': 'string', 'enum': ['observed', 'uncertain', 'contradicted']},
        'evidence_frames': {'type': 'array', 'maxItems': 3, 'items': {'type': 'integer'}},
    },
    'required': ['role', 'target_candidate_id', 'route_support'],
}
runner.nav.NAV['properties']['compound_route_assessment'] = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'landmark_progress': {'type': 'string', 'enum': ['not_seen', 'approaching', 'alongside', 'passed', 'unknown']},
        'associated_door_id': {'type': ['string', 'null']},
        'door_relation': {'type': 'string', 'enum': ['unseen', 'before_landmark', 'alongside_landmark', 'after_landmark', 'unknown']},
        'intended_final_turn': {'type': 'string', 'enum': ['left', 'right', 'unknown']},
        'completion': {'type': 'string', 'enum': ['unresolved', 'satisfied', 'unknown']},
        'evidence_frames': {'type': 'array', 'maxItems': 3, 'items': {'type': 'integer'}},
    },
    'required': ['landmark_progress', 'associated_door_id', 'door_relation',
                 'intended_final_turn', 'completion'],
}
runner.nav.NAV['required'] += ['compound_route_assessment', 'route_choice']
runner.nav.PROMPT += '''
COMPOUND ROUTE CONTRACT: Keep the original instruction sequence and its
landmark-door-turn relationship across decisions. The explicit final turn stays
fixed; its timing and the door identity depend on visual evidence. A landmark
need not disappear or move completely behind you before a nearby door is
considered. Do not reject an entrance only because rails are alongside, the
room looks like a side room, or a farther opening is brighter. Retain nearby
unresolved entrances for comparison with the instructed spatial relation.
landmark_progress and door_relation record an assessment, never an automatic
prerequisite for accepting or rejecting a door. An uncertain door remains
available. Before switching routes consider current and historical geometry;
do not promote a far door just because it is at the corridor end. A temporary
turn may align the robot without changing the instructed final turn.
Use the compact enum and candidate-ID fields. Omit evidence_frames unless a
route switch, candidate exclusion, or STOP requires a visual citation. Do not
write free-text reasons for each candidate or for each decision.
At every decision compare route_progress_feedback from the last action with
the prior expected_effect and the intact instruction relation. Consider
continuing, inspecting a still-unresolved entrance, and changing branch
using the current and supplied historical views. If movement contradicts
the expected local effect, update the route hypothesis before issuing another
batch. Geometry can show that an anchor moved behind or farther away; it
cannot establish which entrance is the instructed one. A right turn used to
align with a hall must still be checked after execution against the full
landmark-door-turn relation. Return only the compact chosen route status.
'''

if __name__ == '__main__':
    os.environ.pop('CLAUDECODE', None)
    try:
        print(json.dumps(asyncio.run(asyncio.wait_for(runner.nav.run(json.load(sys.stdin)), 240))))
    except asyncio.TimeoutError:
        print(json.dumps({'error': 'Claude navigation call exceeded 240 seconds', 'usage_unknown': True}))
    except Exception as exc:
        print(json.dumps({'error': str(exc) or type(exc).__name__, 'usage_unknown': True}))
