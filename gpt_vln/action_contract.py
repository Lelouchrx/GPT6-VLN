
import copy

CONTRACT='''BATCH ACTION CONTRACT (supersedes legacy turn_deg/distance_cm and zero-move interfaces):
Output skill=move or stop; actions contains 1 to 3 ordered commands for move, empty for STOP.
Each command is exactly one of {"type":"turn_left","degrees":15..90 in 15deg increments},
{"type":"turn_right","degrees":15..90 in 15deg increments}, or
{"type":"forward","centimeters":25,50,75,100}. No zero-valued movement, empty move,
negative magnitude, implicit observation command or silent action rewrite is allowed.
STOP is {skill:stop, actions:[]}; do not add zero movement fields to it.
Maximum THREE commands per decision, not three primitives. Execution is exact in order.
Four simultaneous horizontal cameras already cover horizontal directions; FLOOR4 does too.
Never turn solely to scan, refill black reconstruction pixels, test sensors or re-see a tile.
Travel turns must include a forward command in the same batch. Pure turns only satisfy an
explicit orientation goal in original instruction, quoted exactly in instruction_clause.
Choose an RGB-supported destination and expected NEW evidence from changed position.
No room layout, metric position, passability or floor number is established by a model label.

CLIMB SKILL: When ascending/descending, use maneuver=climb_up/climb_down only when supplied
RGB shows relevant steps and an approach supported by the instruction. Identify visible tread,
landing edge and stair direction in note; cite evidence_frames. Select any supported forward
command from 25/50/75/100 cm and turn from 15..90 degrees. The executor expands forward travel
into 25 cm primitives and checks measured motion and height after each primitive; a stall,
collision, or reversed height progress interrupts the remaining batch and returns a fresh view.
One forward primitive is not one stair. Never infer number of risers or reaching a floor from
requested travel, nominal motion or RGB-only pose scale. At apparent
landing, inspect all existing FLOOR4 tiles for continuation, then decide normal travel or
another flight. For second-stair instructions require visual before/after tread evidence;
unknown does not prove arrival. Repeated low progress enters bounded recovery; no repeated
blind long hops on stairs. Stairs use ordinary forward primitives, no teleport/jump.

MULTIPLE DOORS / RETURN ANCHOR SKILL: Before entering an ambiguous door, choose a previously
observed entrance/junction frame from return_anchor_candidates; set return_anchor_frame.
Save target doorway description in branch_target and reasons/evidence in note. A candidate
is a chronological observation, NOT a confirmed door connection or reversible path. Do not
assume all openings lead to distinct rooms or that seeing the first similar object is arrival.
If new views contradict the route, maneuver=return_to_anchor, retrieve the anchor image and
intermediate history, and choose one short ordinary action batch toward a visibly supported
segment. No straight-line coordinate return, invented topology, inverse-action replay or
shortcut through unobserved areas. Recheck at each segment. Returning is unverified until
current RGB supports the old anchor AND original ordered route. A historical frame ID alone
is not proof of return. Carry the chosen anchor while a branch is unresolved.

maneuver is travel, climb_up, climb_down, return_to_anchor, or instruction_orientation.
evidence_frames must refer to attached or already retained observations. Return anchor must
be an earlier real frame, never a proposed future waypoint. Frame IDs may be 0; action
magnitudes may never be 0. All actions and STOP are selected by navigation Claude.
'''


def schema(original):
    value=copy.deepcopy(original)
    for key in ('turn_deg','distance_cm','node_id','action_purpose','instruction_clause'):
        value['properties'].pop(key,None)
        if key in value['required']:value['required'].remove(key)
    value['properties']['skill']={'type':'string','enum':['move','stop']}
    cmd={'anyOf':[
        {'type':'object','additionalProperties':False,'properties':{'type':{'const':'turn_left'},'degrees':{'type':'integer','enum':[15,30,45,60,75,90]}},'required':['type','degrees']},
        {'type':'object','additionalProperties':False,'properties':{'type':{'const':'turn_right'},'degrees':{'type':'integer','enum':[15,30,45,60,75,90]}},'required':['type','degrees']},
        {'type':'object','additionalProperties':False,'properties':{'type':{'const':'forward'},'centimeters':{'type':'integer','enum':[25,50,75,100]}},'required':['type','centimeters']}]}
    fields={'actions':{'type':'array','maxItems':3,'items':cmd},
        'maneuver':{'type':'string','enum':['travel','climb_up','climb_down','return_to_anchor','instruction_orientation']},
        'return_anchor_frame':{'type':['integer','null']},
        'branch_target':{'type':'string'},'instruction_clause':{'type':'string'},
        'evidence_frames':{'type':'array','maxItems':3,'items':{'type':'integer'}}}
    value['properties'].update(fields);value['required']+=list(fields)
    value.pop('anyOf',None)


    return value


def validate_action(out,instruction,available_frames=None,history_frames=None):
    actions=out['actions'];mode=out['maneuver']
    if not isinstance(actions,list) or len(actions)>3:raise ValueError('Maximum three commands')
    if out['skill']=='stop':
        if actions:raise ValueError('STOP cannot contain movement')
    elif out['skill']=='move':
        if not actions:raise ValueError('No empty/zero observation action')
    else:raise ValueError('Unknown skill')
    for a in actions:
        kind=a.get('type')
        if kind in ('turn_left','turn_right'):
            if set(a)!={'type','degrees'} or type(a['degrees']) is not int or a['degrees'] not in (15,30,45,60,75,90):raise ValueError('Turn must be 15..90deg in 15deg increments')
        elif kind=='forward':
            if set(a)!={'type','centimeters'} or type(a['centimeters']) is not int or a['centimeters'] not in (25,50,75,100):raise ValueError('Forward must be 25..100cm in 25cm increments')
        else:raise ValueError('Unknown command')
    forward=[a for a in actions if a['type']=='forward']
    if actions and not forward:
        clause=out.get('instruction_clause','')
        if mode!='instruction_orientation' or not clause.strip() or clause not in instruction:raise ValueError('No panoramic scan: pure turn requires original orientation goal')
    if mode in ('climb_up','climb_down') and actions and not out.get('evidence_frames'):
        raise ValueError('Climb needs RGB evidence frames')
    anchor=out.get('return_anchor_frame')
    if history_frames is not None and anchor is not None and (type(anchor) is not int or anchor not in history_frames):raise ValueError('Return anchor must be an actual earlier frame')
    if mode=='return_to_anchor' and anchor is None:raise ValueError('Return needs an observed anchor')
    if available_frames is not None and not set(out.get('evidence_frames',[]))<=set(available_frames):raise ValueError('Unknown evidence frame')
    return out


def primitives(out):
    if out['skill']=='stop':return [0]
    seq=[]
    for a in out['actions']:
        if a['type']=='forward':seq.extend([1]*(a['centimeters']//25))
        else:seq.extend([2 if a['type']=='turn_left' else 3]*(a['degrees']//15))
    return seq
