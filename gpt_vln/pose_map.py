
import math,numpy as np
class OwnPoseMap:
 def __init__(self,sim):
  self.sim=sim;self.origin=np.array(sim.get_agent_state().position);self.rows=[]
  self.record(0)
 def record(self,step):
  s=self.sim.get_agent_state();p=np.array(s.position);q=s.rotation
  u=np.array([q.x,q.y,q.z]);v=np.array([0.,0.,-1.]);t=2*np.cross(u,v);d=v+q.w*t+np.cross(u,t)
  row={'step':step,'xyz_m':(p-self.origin).round(4).tolist(),
       'heading_xz_deg':round(math.degrees(math.atan2(d[2],d[0])),3)}
  if not self.rows or self.rows[-1]['step']!=step:self.rows.append(row)
  return row
 def context(self,anchor=None):
  current=self.rows[-1];a=next((r for r in self.rows if r['step']==anchor),None)
  selected=self.rows if len(self.rows)<=80 else [self.rows[round(i*(len(self.rows)-1)/79)] for i in range(80)]
  return {'source':'simulator GT own pose, origin at episode start; x/z ground plane,y height',
   'current':current,'cumulative_map':{'type':'executed trajectory and visited points, NOT occupancy/free-space map',
    'points':selected,'total_recorded_points':len(self.rows),'unvisited_space':'unknown',
    'connections':'chronological executed trail only; no guaranteed reverse passability'},
   'return_anchor':a,'distance_to_anchor_m':round(float(np.linalg.norm(np.array(current['xyz_m'])-a['xyz_m'])),4) if a else None,
   'limitations':'Own localization does not prove correct room, target completion or stair ordinal. No goal/reference/navmesh access.'}

 def history_summary(self, decision_frames, limit=8):
  by_step={row['step']:row for row in self.rows}
  frames=[step for step in decision_frames if step in by_step]
  selected=[0]+frames[-(limit-2):]+[self.rows[-1]['step']]
  selected=list(dict.fromkeys(selected))
  rows=[by_step[step] for step in selected]
  return {'source':'simulator own pose; episode-relative meters, no target or map',
          'points':[{'frame':row['step'],'xyz_m':row['xyz_m'],
                     'heading_xz_deg':row['heading_xz_deg']} for row in rows],
          'meaning':'chronological visited poses; straight lines do not imply traversable shortcuts'}
