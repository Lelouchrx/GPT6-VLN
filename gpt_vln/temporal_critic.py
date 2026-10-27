





import copy
import math
from collections import deque
import cv2
import numpy as np

RULES=(
 {'id':'unknown_frame_citation','feature':'unknown_frame_citation','equals':True,'dwell':1,'cooldown':4,'interrupt':False},
 {'id':'completion_after_only_turns','feature':'completion_after_only_turns','equals':True,'dwell':1,'cooldown':4,'interrupt':False},
 {'id':'unscaled_ordinal_certainty','feature':'unscaled_ordinal_certainty','equals':True,'dwell':1,'cooldown':4,'interrupt':False},
 {'id':'repeated_branch_without_progress','feature':'repeated_branch_without_progress','equals':True,'dwell':2,'cooldown':12,'interrupt':False},
 {'id':'verified_collision_sensor','feature':'collision_observed','equals':True,'dwell':2,'cooldown':4,'interrupt':True},
 {'id':'verified_metric_stall','feature':'metric_stall','equals':True,'dwell':3,'cooldown':8,'interrupt':True},
 {'id':'return_visual_correspondence','feature':'anchor_visual_match','equals':True,'dwell':2,'cooldown':8,'interrupt':False},
)
EXPLANATIONS={
 'unknown_frame_citation':'Actor cited an unavailable frame; citation provenance is invalid, not proof of task failure.',
 'completion_after_only_turns':'A spatial crossing was claimed with only turn/STOP commands between cited observations. Review the missing crossing evidence; no metric motion is inferred.',
 'unscaled_ordinal_certainty':'Absolute riser-count or zero-climb claim lacks scaled base-height/tread measurements. It is a hypothesis; code cannot prove second stair or overshoot.',
 'repeated_branch_without_progress':'Geometric image correspondence recurred with unchanged unresolved clause states after intervening actions. Possible loop; intentional return is also possible.',
 'verified_collision_sensor':'A declared collision sensor reported consecutive contact. Stop remaining batch and let navigator choose the next action.',
 'verified_metric_stall':'A declared metric odometry source measured repeated forward displacement below its declared stall threshold. Unknown displacement never counts as zero.',
 'return_visual_correspondence':'Current image geometrically matches the stored return-anchor image. This supports visual reidentification only; metric return and correct route are still unverified.',
}

class TemporalRules:
 def __init__(self,rules=RULES):
  self.rules=copy.deepcopy(rules);self.states={r['id']:{'count':0,'cooldown':0} for r in rules};self.last={}
 def evaluate(self,features,step,phase):
  proposals=[];outcomes=[]
  for r in self.rules:
   expected='proposal' if r['id'] in ('unknown_frame_citation','completion_after_only_turns','unscaled_ordinal_certainty') else 'decision' if r['id']=='return_visual_correspondence' else 'primitive'
   if phase!=expected:continue
   key=(r['id'],phase)
   if self.last.get(key)==step:continue
   self.last[key]=step;s=self.states[r['id']];v=features.get(r['feature'])

   if v is None:
    s['count']=0;outcomes.append({'rule':r['id'],'result':'unknown'});continue
   match=v==r['equals']
   if s['cooldown']:
    s['cooldown']-=1;s['count']=0
   else:
    s['count']=s['count']+1 if match else 0
    if s['count']>=r['dwell']:
     proposals.append({'rule_id':r['id'],'step':step,'phase':phase,'feature':r['feature'],
       'observed_value':v,'evidence':EXPLANATIONS.get(r['id'],r['feature']),
       'interrupt':r['interrupt'],'environment_write':False})
     s['count']=0;s['cooldown']=r['cooldown']
   outcomes.append({'rule':r['id'],'result':'condition_observed' if match else 'not_observed'})
  return proposals,outcomes

def visual_correspondence(a,b):

 gray=lambda im:cv2.cvtColor(cv2.resize(np.asarray(im)[...,:3],(320,240)),cv2.COLOR_RGB2GRAY)
 a,b=gray(a),gray(b);orb=cv2.ORB_create(nfeatures=700)
 ka,da=orb.detectAndCompute(a,None);kb,db=orb.detectAndCompute(b,None)
 if da is None or db is None or min(len(ka),len(kb))<30:return {'status':'unknown','reason':'insufficient_features'}
 pairs=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da,db,k=2)
 good=[p[0] for p in pairs if len(p)==2 and p[0].distance<.72*p[1].distance]
 if len(good)<30:return {'status':'not_supported','matches':len(good)}
 p=np.float32([ka[m.queryIdx].pt for m in good]);q=np.float32([kb[m.trainIdx].pt for m in good])
 H,mask=cv2.findHomography(p,q,cv2.RANSAC,2.5)
 if H is None or mask is None:return {'status':'unknown','reason':'geometric_fit_failed'}
 keep=mask.ravel().astype(bool);n=int(keep.sum());ratio=n/len(good)
 hull=cv2.convexHull(p[keep]);coverage=float(cv2.contourArea(hull)/(320*240)) if n>=3 else 0.
 error=float(np.median(np.linalg.norm(cv2.perspectiveTransform(p[None],H)[0][keep]-q[keep],axis=1))) if n else None
 supported=n>=25 and ratio>=.55 and coverage>=.10 and error is not None and error<=2.5
 return {'status':'supported' if supported else 'not_supported','inliers':n,'ratio':round(ratio,3),
  'coverage':round(coverage,3),'median_reprojection_px':round(error,3) if error is not None else None,
  'limitation':'Visual correspondence only; repeated geometry, viewpoint and planar scenes can be ambiguous.'}

class NavigationCodeCritic:
 def __init__(self,instruction):
  self.instruction=instruction;self.rules=TemporalRules();self.actions=[];self.anchors={}
  self.latest=[];self.audit=[];self.branch_visits={};self.episode_events=[];self.active=None
  self.recovered_visual=False
 def packet(self):
  return {'implementation':'Python temporal feature rules + ORB/RANSAC correspondence',
   'extra_model_calls':0,'proposals':copy.deepcopy(self.latest[-4:]),
   'completion':'unknown; no environment goal predicates or scaled stair geometry available',
   'route':'unknown unless backed by independent declared measurements; actor labels are hypotheses',
   'capabilities_missing':['metric_odometry','collision_sensor','grounded_goal_predicates'],
   'recovery_event':{'id':self.active['id'],'state':self.active['state']} if self.active else None}
 def _record(self,features,step,phase,evidence=None):
  proposals,outcomes=self.rules.evaluate(features,step,phase)
  self.latest=[p for p in self.latest if step-p['step']<=40]
  self.latest=(self.latest+proposals)[-8:]
  row={'step':step,'phase':phase,'features':features,'rule_outcomes':outcomes,
       'proposals':proposals,'visual_evidence':evidence,'model_calls':0}
  self.audit.append(row)
  errors=[p for p in proposals if p['rule_id'] in ('verified_collision_sensor','verified_metric_stall','repeated_branch_without_progress')]
  if errors and self.active is None:
   self.active={'id':len(self.episode_events),'state':'active','start_step':step,'actions':[],
     'trigger_rules':[p['rule_id'] for p in errors],'movement_restored':False,'route_reentry':'unknown'}
   self.episode_events.append(self.active)
  if self.active and step-self.active['start_step']>=40:
   self.active.update(state='budget_exit_unverified',end_step=step);self.active=None
  return row
 def before_decision(self,frame,history,obs,return_anchor=None):
  evidence=None;features={}

  if return_anchor in self.anchors:
   old=self.anchors[return_anchor]['rgb']
   matches=[visual_correspondence(old,obs[k]) for k in ('rgb','right_rgb','rear_rgb','left_rgb')]
   evidence={'anchor':return_anchor,'views':matches}
   features['anchor_visual_match']=any(m['status']=='supported' for m in matches)
  return self._record(features,frame,'decision',evidence)
 def proposal(self,out,frame,history,allowed_frames):
  events=out.get('events',[]);known={r['id'] for r in history}|{frame}
  bad=any(not set(e.get('frames',[]))<=known for e in events)
  crossing_words=('pass','cross','enter','exit','through')
  unsupported_cross=False
  for e in events:
   ids=e.get('frames',[])
   if e.get('state') in ('passed','reached') and len(set(ids))>=2 and any(w in e.get('clause','').lower() for w in crossing_words):
    interval=[a for a in self.actions if min(ids)<a['step']<=max(ids)]
    if interval and not any(a['action']==1 for a in interval):unsupported_cross=True
  txt=' '.join(e.get('observation','') for e in events)+' '+out.get('note','')
  absolute=any(x in txt.lower() for x in ('zero riser','zero stair','no riser climbed','no tread climbed','full flight','entire flight','second tread','second stair'))
  feats={'unknown_frame_citation':bad,'completion_after_only_turns':unsupported_cross,
     'unscaled_ordinal_certainty':absolute if any(w in self.instruction.lower() for w in ('stair','step')) else False}
  return self._record(feats,frame,'proposal')
 def after_primitive(self,row,obs,history,branch_anchor=None,route_events=None):
  self.actions.append(copy.deepcopy(row));features={};metric=row.get('measured_translation_m')


  if row['action']==1 and row.get('metric_odometry_source') and metric is not None:
   features['metric_stall']=metric<.02
   if self.active and metric>.12:self.active['movement_restored']=True
  if row.get('collision_sensor_source') and type(row.get('collision')) is bool:features['collision_observed']=row['collision']
  evidence=None
  if row['action']==1:
   candidates=[fid for fid in self.anchors if row['step']-fid>=12]
   if candidates:
    fid=branch_anchor if branch_anchor in candidates else candidates[-1]
    old=self.anchors[fid]
    match=visual_correspondence(old['rgb'],obs['rgb'])
    same_progress=old['progress']==tuple((e['clause'],e['state']) for e in (route_events or []))
    features['repeated_branch_without_progress']=match['status']=='supported' and same_progress
    evidence={'past_frame':fid,'correspondence':match,'actor_progress_unchanged':same_progress,
              'limitation':'Revisit with unchanged actor hypothesis is a possible loop, not proof of deviation.'}
  if self.active:self.active['actions'].append({'step':row['step'],'action':row['action']})
  return self._record(features,row['step'],'primitive',evidence)
 def save_observation(self,frame,obs,out):
  self.anchors[frame]={'rgb':cv2.resize(np.asarray(obs['rgb'])[...,:3],(320,240)),
    'progress':tuple((e['clause'],e['state']) for e in out.get('events',[])),
    'actions':copy.deepcopy(out.get('actions',[]))}
 def end(self,step):
  if self.active:self.active.update(state='episode_end_unverified',end_step=step);self.active=None
 def summary(self):
  return {'triggers':len(self.episode_events),'executed_interventions':sum(bool(e['actions']) for e in self.episode_events),
    'motion_restored':sum(e['movement_restored'] for e in self.episode_events),'route_verified':0,'route_reentry_claims':0,
    'verified_events':0,'unverified_events':len(self.episode_events),'extra_critic_model_calls':0,
    'interpretation':'Code-only sensor/rule evidence. Unsupported goal and route semantics remain unknown.'}
