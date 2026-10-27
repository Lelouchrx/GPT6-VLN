
import copy

VERDICTS=['confirmed','contradicted','unknown']
FRAME={'type':'integer','minimum':0}
JUDGMENT={'type':'object','additionalProperties':False,'properties':{
 'description':{'type':'string'},'verdict':{'type':'string','enum':VERDICTS},
 'frames':{'type':'array','maxItems':3,'items':FRAME}},
 'required':['description','verdict','frames']}
SCHEMA={'type':'object','additionalProperties':False,'properties':{
 'current_clause':{'type':'string'},
 'landmarks':{'type':'array','maxItems':4,'items':JUDGMENT},
 'prerequisites':{'type':'array','maxItems':4,'items':JUDGMENT},
 'door_crossing':{'type':'object','additionalProperties':False,'properties':{
  'verdict':{'type':'string','enum':VERDICTS},'description':{'type':'string'},
  'before_frame':{'type':['integer','null']},'after_frame':{'type':['integer','null']}},
  'required':['verdict','description','before_frame','after_frame']},
 'route_verdict':{'type':'string','enum':VERDICTS},
 'route_frames':{'type':'array','maxItems':3,'items':FRAME},
 'expected_next':{'type':'string'}},
 'required':['current_clause','landmarks','prerequisites','door_crossing','route_verdict','route_frames','expected_next']}

PROMPT='''
STRUCTURED VISUAL EVIDENCE: in this SAME navigation call output route_evidence.
stage is your estimated current route stage, never independently verified truth.
current_clause must be an exact contiguous quote from the ORIGINAL instruction.
Describe currently observed landmarks, prerequisite relations for the current stage,
and cite ONLY image frames attached to THIS call. confirmed means YOUR visual judgment,
not code-certified truth. Use unknown when unsupported; never invent confirming frames.
For a claimed door crossing cite an earlier before_frame and a later after_frame,
describe why the intended doorway was crossed, and judge the COMPLETED interval only.
Do not claim crossing from a planned forward command, disappearance alone or rotation.
If no door crossing is being assessed use unknown and null frames. Every confirmed
landmark/prerequisite and non-unknown route judgment needs actual image citations.
If proposing an advance beyond the tracked hypothesis stage, supply the observable
prerequisites that support the new stage. No stage skipping without evidence.
expected_next describes the observable change expected AFTER the actions now proposed.
You may disagree with a prior Python audit but must provide new evidence. Python checks
provenance, missing conditions and actual executed actions, not doorway identity.
'''


class SemanticEvidenceAudit:
    def __init__(self,instruction):
        self.instruction=instruction;self.stage=None;self.latest=None;self.records=[];self.last_visual=None

    def context(self):
        return {'tracked_hypothesis_stage':self.stage,'last_audit':copy.deepcopy(self.latest),
                'last_visual_hypotheses':copy.deepcopy(self.last_visual),
                'authority':'Claude visual hypotheses; Python structural/execution checks only. No independent semantic confirmation.'}

    def inspect(self,out,current_frame,visible_frames,history):
        e=out['route_evidence'];issues=[];visible=set(visible_frames)
        def issue(kind,evidence,verdict='unknown'):
            issues.append({'kind':kind,'evidence':evidence,'verdict':verdict})
        def cite(frames,required,label):
            if any(type(f) is not int or f not in visible for f in frames):
                issue('invalid_frame_reference',label,'contradicted')
            if required and not frames:issue('missing_visual_evidence',label)
        clause=e['current_clause']
        if not clause.strip() or clause not in self.instruction:
            issue('instruction_clause_mismatch',clause,'contradicted')
        for label in ('landmarks','prerequisites'):
            for item in e[label]:cite(item['frames'],item['verdict']=='confirmed',label+': '+item['description'])
        cite(e['route_frames'],e['route_verdict']!='unknown','route judgment')
        stage=out['stage'];advance=self.stage is not None and stage>self.stage

        if advance or (self.stage is None and stage>0):
            if not e['prerequisites'] or any(p['verdict']!='confirmed' or not p['frames'] for p in e['prerequisites']):
                issue('stage_preconditions_missing','Cannot trace the claimed new stage to positive prerequisite hypotheses')
        cross=e['door_crossing'];cross_check={'semantic_crossing':'unknown','executed_forward_count':None}
        if cross['verdict']!='unknown':
            a,b=cross['before_frame'],cross['after_frame']
            cite([f for f in (a,b) if f is not None],True,'door crossing')
            if type(a) is not int or type(b) is not int or not a<b<=current_frame:
                issue('crossing_time_order','Crossing must refer to an earlier completed interval','contradicted')
            else:
                rows={r['step']:r for h in history for r in h['last_execution'].get('actual',[]) if a<r['step']<=b}
                if len(rows)!=b-a:
                    issue('crossing_execution_missing','Incomplete executed interval; cannot test motion')
                else:
                    n=sum(r['action']==1 for r in rows.values());cross_check['executed_forward_count']=n
                    if cross['verdict']=='confirmed' and n==0:
                        issue('crossing_without_translation_action','Completed interval contains only turn/STOP actions','contradicted')
                    cross_check['motion_authority']='Executed forward is an attempt, not measured displacement or semantic crossing'
        if e['route_verdict']=='contradicted':
            issue('actor_reported_route_conflict','Main navigator reports contradictory visual route evidence; not independently verified')
        if not e['expected_next'].strip():issue('missing_expected_effect','No observable expected post-action effect')
        if type(stage) is not int or stage<0:issue('invalid_stage','Stage must be a nonnegative integer','contradicted')
        if not issues:self.stage=stage
        audit={'frame':current_frame,'claimed_stage':stage,'tracked_hypothesis_stage':self.stage,
               'contract_verdict':'contradicted' if any(x['verdict']=='contradicted' for x in issues) else 'unknown' if issues else 'traceable_hypothesis',
               'issues':issues,'crossing_execution_check':cross_check,'semantic_route':'unknown',
               'model_calls':0,'action_authority':'advisory; no forced motion or STOP veto',
               'issue_classes':sorted({x['kind'] for x in issues}),
               'false_positive':None,'false_negative':None,'scoring_note':'Requires independent annotations; unknown is not an error label'}
        self.last_visual=copy.deepcopy(e)
        self.latest=copy.deepcopy(audit);self.records.append(copy.deepcopy(audit));return audit
