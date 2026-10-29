
import asyncio
import json
import os
import sys

import gpt_vln.memory_model as base_nav
from gpt_vln.memory_model import runner

TASK_GOAL = {'type': 'object', 'additionalProperties': False, 'properties': {
    'description': {'type': 'string'},
    'view': {'type': 'string', 'enum': ['front', 'right', 'rear', 'left']},
    'point_norm': {'type': ['array', 'null'], 'minItems': 2, 'maxItems': 2,
                   'items': {'type': 'number'}},
    'status': {'type': 'string',
               'enum': ['unknown', 'seen', 'approaching', 'possibly_passed', 'reached']},
}, 'required': ['description', 'view', 'point_norm', 'status']}

runner.nav.NAV = dict(runner.nav.NAV)
runner.nav.NAV['properties'] = dict(runner.nav.NAV['properties'])
runner.nav.NAV['properties']['task_goal_anchor'] = TASK_GOAL
if 'task_goal_anchor' not in runner.nav.NAV['required']:
    runner.nav.NAV['required'] = list(runner.nav.NAV['required']) + ['task_goal_anchor']
runner.nav.NAV['properties']['review_response'] = {
    'type': 'object', 'additionalProperties': False, 'properties': {
        'resolution': {'type': 'string', 'enum': [
            'not_applicable', 'backtrack_to_node', 'inspect_suspected_branch',
            'continue_with_counterevidence']},
        'evidence_frames': {'type': 'array', 'maxItems': 3,
                            'items': {'type': 'integer'}},
        'map_node_id': {'type': ['integer', 'null']},
        'current_test': {'type': 'string'},
    },
    'required': ['resolution', 'evidence_frames', 'map_node_id', 'current_test'],
}
if 'review_response' not in runner.nav.NAV['required']:
    runner.nav.NAV['required'] = list(runner.nav.NAV['required']) + ['review_response']
runner.nav.NAV['properties']['room_stage_hypothesis']={
    'type':'object','additionalProperties':False,'properties':{
        'node_id':{'type':['integer','null']},
        'room':{'type':'string'},
        'relation':{'type':'string','enum':['none','possible','confirmed','rejected']},
        'instruction_stage':{'type':['integer','null']},
        'evidence_frame':{'type':['integer','null']},
    },'required':['node_id','room','relation','instruction_stage','evidence_frame']}
runner.nav.NAV['required']=list(runner.nav.NAV['required'])+['room_stage_hypothesis']

PROMPT = base_nav.PROMPT + '''
ROOM STAGE MEMORY: The map lists unnamed visited nodes and robot-relative bearings.
When a room or junction is actually experienced, use room_stage_hypothesis to
associate a visited node with a visible room description and a possible instruction
stage. A family/sofa room may correspond to an instruction's den; keep it possible
until the route and visual evidence support confirmation. Never use hidden room
labels. Use observe_node(node_id) when an unnamed node's panorama would resolve
which place or branch it was. A node coordinate is a past pose, not a room centroid.
Use the recorded return_walked_m as a historical-path estimate; never shortcut
through unknown map cells. The map image includes F/L/R/B beside the robot arrow.
STABLE TASK-ENDPOINT ANCHOR: In task_goal_anchor, describe the FINAL instructed
endpoint, not the local waypoint or the next doorway. If that exact endpoint is
not visually identified in a current view, use status=unknown and point_norm=null.
When the endpoint itself is visible, give its point in the named current camera
tile. Keep the same endpoint identity over time. Do not replace it with an
intermediate landmark just to keep a distance measurement available. The Python
critic stores the first reliable distance to this one depth-backed visual
hypothesis as the baseline, then compares against that same initial distance at
decisions 10, 20, 30, and onward. Regression is only fallible evidence for your
next route decision; Python does not end the episode from this hypothesis. It does
not prove route correctness or task completion. Unknown endpoint distance cannot
be treated as progress or regression. If the endpoint was not visible at task
start, the baseline is the first later reliable visual anchor, and the checkpoint
record states which decision supplied it.
NO-PROGRESS RECOVERY: At each ten-decision checkpoint, compare the ordered
instruction subgoal with the last ten executed decisions, not only the actor's
stage number. If the same required transition remains unresolved and neither
visual evidence nor the measured endpoint distance shows progress, compare
continuing with revisiting the nearest already-visited junction before the
unresolved branch. If revisiting is the better evidence-gathering move, choose
map_operation=navigate_to_node with that visited node ID; Python will follow
only recorded walked edges and return a fresh observation. Never invent a
connection or return just because one action was blocked. The critic may report
stall or regression, but the navigator chooses whether and where to backtrack.
MANDATORY REVIEW RESPONSE: If context contains pending_route_review, you must
fill review_response and resolve that specific finding before acting. For a
confirmed_wrong_branch finding, only backtrack_to_node or inspect_suspected_branch
is permitted; do not continue the old branch or STOP. For possible_wrong_branch,
choose backtrack_to_node, inspect_suspected_branch, or continue_with_counterevidence
with actual current/retrieved frame IDs. backtrack_to_node requires map_node_id
and map_operation=navigate_to_node to the same previously visited node.
inspect_suspected_branch requires a short action directed at obtaining the stated
visual check, no more than 50 cm forward, not another long batch on the disputed
route. Inspection does NOT clear the warning: on the next decision, use the new
current image to determine whether to backtrack or continue with counterevidence.
For the same pending review, at most TWO short inspections are allowed. If the
warning remains unresolved after two checks, select backtrack_to_node to a
visited junction and reobserve there; do not inch along the same corridor.
Backtracking only resolves the review after measured pose confirms arrival at
the selected node. If pending_node_return is present, continue returning to
that same node; an interrupted route segment is not a completed return.
When pending_route_review.repeated_prediction_errors is true, do not select
continue_with_counterevidence on that review turn. Choose a short branch
inspection or return to a visited node; another long forward prediction on
the same corridor is not a valid response to repeated failed predictions.
If no review is
pending, use resolution=not_applicable and null map_node_id.
'''


async def run(request):
    original = base_nav.PROMPT
    base_nav.PROMPT = PROMPT
    try:
        return await base_nav.run(request)
    finally:
        base_nav.PROMPT = original


if __name__ == '__main__':
    os.environ.pop('CLAUDECODE', None)
    try:
        print(json.dumps(asyncio.run(run(json.load(sys.stdin)))))
    except Exception as exc:
        print(json.dumps({'error': str(exc) or type(exc).__name__, 'usage_unknown': True}))
