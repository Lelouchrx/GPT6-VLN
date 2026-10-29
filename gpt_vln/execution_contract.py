





import math


def signed_angle(degrees):return (degrees+180)%360-180

def preflight_action(plan,features):

    results=[]
    prohibited={'goal_distance','reference_path','navmesh','goal_position'} & set(features)
    if prohibited:raise ValueError('Privileged task information not allowed: '+str(sorted(prohibited)))
    observed=set(features.get('verified_predicates',[]))

    for requirement in plan.get('requires',[]):
        known=requirement in observed
        results.append({'kind':'ordered_precondition','predicate':requirement,
                        'verdict':'supported' if known else 'unknown',
                        'authority':'Missing evidence does not prove wrong branch'})
    for forbidden in plan.get('must_not_precede',[]):
        if forbidden in observed:
            results.append({'kind':'ordered_conflict','predicate':forbidden,'verdict':'contradicted',
                            'authority':'A verified event violates explicit ordering'})
    turn=plan.get('required_turn')
    actual=sum((1 if a['type']=='turn_left' else -1)*a['degrees'] for a in plan.get('actions',[]) if a['type'] in ('turn_left','turn_right'))
    if turn:
        anchor=features.get('anchor_heading_deg');current=features.get('heading_deg')
        if anchor is None or current is None:
            verdict='unknown';relative=None
        else:
            relative=signed_angle(current+actual-anchor)
            verdict='supported' if (turn=='left' and 15<=relative<=165) or (turn=='right' and -165<=relative<=-15) else 'contradicted'
        results.append({'kind':'anchored_turn','verdict':verdict,'relative_intended_heading_deg':relative,
                        'limitation':'Checks heading relative to the specified anchor, not doorway identity or walkability'})
    if plan.get('target_frame') is not None:
        valid=plan['target_frame'] in features.get('visible_frames',[])
        results.append({'kind':'target_evidence','verdict':'supported' if valid else 'unknown',
                        'limitation':'Frame visibility does not independently ground the target object'})
    return results


class ExecutionContractCritic:

    def __init__(self):self.flat_climb_steps=0;self.last_step=None
    def evaluate(self,intent,measurement):
        if self.last_step==measurement['step']:return {'verdict':'already_checked','interrupt':False}
        if self.last_step is not None and measurement['step']!=self.last_step+1:
            self.flat_climb_steps=0
        self.last_step=measurement['step']
        out={'step':measurement['step'],'source':measurement.get('source'),'checks':[],
             'interrupt':False,'global_route':'unknown','completion':'unknown','model_calls':0}
        if not measurement.get('source'):
            self.flat_climb_steps=0
            out['checks'].append({'kind':'own_motion','verdict':'unknown'});return out
        mode=intent.get('maneuver');dy=measurement.get('height_delta_m')
        if mode not in ('climb_up','climb_down') or measurement.get('action')!=1:
            self.flat_climb_steps=0
        if mode in ('climb_up','climb_down') and measurement.get('action')==1:
            if dy is None:
                self.flat_climb_steps=0
                out['checks'].append({'kind':'climb_direction','verdict':'unknown'})
            else:
                sign=1 if mode=='climb_up' else -1;signed=sign*dy
                if signed<-.03:
                    out['checks'].append({'kind':'climb_direction','verdict':'contradicted',
                      'evidence':'Measured height changes opposite to intended stair direction',
                      'height_delta_m':dy});out['interrupt']=True;self.flat_climb_steps=0
                elif signed>.03:
                    out['checks'].append({'kind':'climb_direction','verdict':'supported','height_delta_m':dy,
                      'limitation':'Correct vertical direction, not one-tread or ordinal completion proof'})
                    self.flat_climb_steps=0
                else:
                    self.flat_climb_steps+=1
                    out['checks'].append({'kind':'climb_direction','verdict':'unknown','height_delta_m':dy,
                      'evidence':'No significant vertical progress; may still be approach/landing'})
                    if self.flat_climb_steps>=2:
                        out['checks'].append({'kind':'climb_phase_review','verdict':'review',
                          'evidence':'Two measured advances without vertical progress: inspect approach/landing before extending climb'})
                        out['interrupt']=True
        delta=measurement.get('translation_xz_m');heading=intent.get('travel_heading_deg')
        if delta is not None and heading is not None and measurement.get('action')==1:
            angle=math.radians(heading);forward=delta[0]*math.cos(angle)+delta[1]*math.sin(angle)
            lateral=-delta[0]*math.sin(angle)+delta[1]*math.cos(angle)
            length=math.hypot(*delta)
            verdict='unknown' if length<.02 else 'contradicted' if forward<-.02 or abs(lateral)>.15 else 'supported'
            out['checks'].append({'kind':'local_motion_direction','verdict':verdict,
              'forward_projection_m':round(forward,4),'lateral_m':round(lateral,4),
              'limitation':'Local planned segment consistency only; not distance to hidden reference route'})
            if verdict=='contradicted':out['interrupt']=True
        if measurement.get('collision') is True:
            out['checks'].append({'kind':'collision','verdict':'observed'});out['interrupt']=True
        return out
