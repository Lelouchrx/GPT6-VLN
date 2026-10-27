




import math
import numpy as np
from PIL import Image, ImageDraw


def _derive_places(path, spacing_m=2.5, snap_frac=0.6, jump_m=0.5, max_nodes=64):

































    if not path:
        return {"nodes": [], "edges": [], "current": None}

    def xz(a, b):
        if abs(float(a[1]) - float(b[1])) > 0.6:
            return float('inf')
        return math.hypot(float(a[0]) - float(b[0]), float(a[2]) - float(b[2]))

    spacing = float(spacing_m)
    snap = float(snap_frac) * spacing
    jump = float(jump_m)

    nodes = [{"id": 0, "position": list(map(float, path[0])), "visits": 1,
              "path_index": 0}]
    edges = {}
    current = 0
    left_index = 0
    arc = 0.0
    clean = True

    def link(a, b, length, span):
        key = (a, b) if a < b else (b, a)
        prev = edges.get(key)


        if prev is None or length < prev["arc_m"]:
            edges[key] = {"arc_m": round(float(length), 3), "span": list(span)}

    for i in range(1, len(path)):
        seg = xz(path[i - 1], path[i])
        arc += seg
        if seg > jump:
            clean = False

        best, best_d = None, None
        for n in nodes:
            d = xz(path[i], n["position"])
            if best_d is None or d < best_d:
                best, best_d = n, d

        if best_d <= snap and best["id"] != current:
            if clean:
                link(current, best["id"], arc, (left_index, i))
            best["visits"] += 1
            current, left_index, arc, clean = best["id"], i, 0.0, True
        elif best_d > spacing and len(nodes) < int(max_nodes):
            new = {"id": len(nodes), "position": list(map(float, path[i])),
                   "visits": 1, "path_index": i}
            nodes.append(new)
            if clean:
                link(current, new["id"], arc, (left_index, i))
            current, left_index, arc, clean = new["id"], i, 0.0, True

    return {
        "nodes": nodes,
        "edges": [{"a": a, "b": b, **v} for (a, b), v in sorted(edges.items())],
        "current": current,
    }

class ObservedMap:
    def __init__(self, sim, mpp=0.02, height_below_m=0.3):
        self.sim, self.mpp = sim, mpp
        self.height_below_m = float(height_below_m)
        start = np.asarray(sim.get_agent_state().position, dtype=float)[[0, 2]]
        self.low = start - 20.0
        size = np.ceil(np.array([40.0, 40.0]) / mpp).astype(int) + 1
        self.shape = (int(size[1]), int(size[0]))
        self.sums = np.zeros((*self.shape, 3), np.float32)
        self.hits = np.zeros(self.shape, np.uint16)
        self.path = []
        self.nodes = []
        self.edges = []

    def record_pose(self):
        pos = np.asarray(self.sim.get_agent_state().position, dtype=float)
        if not self.path or np.linalg.norm(pos - self.path[-1]) > 0.01:
            self.path.append(pos.copy())
            annotations={n['id']:(n['name'],n['caption'],n['annotations'],n.get('anchor')) for n in self.nodes}
            graph=_derive_places(self.path)
            self.nodes=graph['nodes'];self.edges=graph['edges']
            for node in self.nodes:
                node['name'],node['caption'],node['annotations'],node['anchor']=annotations.get(node['id'],('','',[],None))

    def observe(self, obs):
        self.record_pose()
        state = self.sim.get_agent_state()
        q = state.rotation
        import quaternion
        rotation = quaternion.as_rotation_matrix(q)
        camera = np.asarray(state.position, dtype=float) + [0., 1.25, 0.]
        floor = float(state.position[1])
        for offset, prefix in enumerate(('', 'right_', 'rear_', 'left_')):
            rgb = np.asarray(obs[prefix+'rgb_map_color'])[::4, ::4, :3]
            depth = np.asarray(obs[prefix+'rgb_map_depth']).squeeze()[::4, ::4]
            h,w = depth.shape


            fx = (w/2) / math.tan(math.radians(120)/2)
            u,v = np.meshgrid(np.arange(w),np.arange(h))
            good = np.isfinite(depth) & (depth > .001) & (depth < 10.)
            if not good.any():continue
            xyz = np.stack(((u-(w-1)/2)/fx*depth,
                            -((v-(h-1)/2)/fx)*depth, -depth),axis=-1)[good]
            angle = (0,-math.pi/2,math.pi,math.pi/2)[offset]
            c,s = math.cos(angle),math.sin(angle)
            turn = np.array([[c,0,s],[0,1,0],[-s,0,c]])
            world = xyz @ (rotation @ turn).T + camera
            band = (world[:,1] > floor-self.height_below_m) & (world[:,1] < floor+1.6)
            world = world[band]
            xy = np.rint((world[:,[0,2]]-self.low)/self.mpp).astype(int)
            colors = rgb[good][band]
            inbounds = (xy[:,0]>=0)&(xy[:,0]<self.shape[1])&(xy[:,1]>=0)&(xy[:,1]<self.shape[0])
            xy,colors,world = xy[inbounds],colors[inbounds],world[inbounds]


            radius = np.clip(np.rint(np.linalg.norm(world-camera,axis=1)
                * (math.radians(120)/max(1,w)) / self.mpp / 2),0,5).astype(int)
            for dy in range(-5,6):
                for dx in range(-5,6):
                    chosen = radius >= max(abs(dx),abs(dy))
                    if not chosen.any():continue
                    q=xy[chosen]+(dx,dy);col=colors[chosen]
                    valid=(q[:,0]>=0)&(q[:,0]<self.shape[1])&(q[:,1]>=0)&(q[:,1]<self.shape[0])
                    q,col=q[valid],col[valid]
                    np.add.at(self.sums,(q[:,1],q[:,0]),col)
                    np.add.at(self.hits,(q[:,1],q[:,0]),1)

    def annotate_node(self, node_id, name, caption='', clause=None, frame=None, image=None):
        if type(node_id) is not int or not 0 <= node_id < len(self.nodes):
            raise ValueError('node must already exist')
        if not name.strip():raise ValueError('name required')
        node=self.nodes[node_id];node['name']=name[:80];node['caption']=caption[:250]
        pose=self.sim.get_agent_state()
        import quaternion
        forward=quaternion.as_rotation_matrix(pose.rotation) @ np.array([0.,0.,-1.])
        anchor={'position':np.asarray(pose.position,dtype=float).round(4).tolist(),
                'heading_xz_deg':round(math.degrees(math.atan2(forward[2],forward[0])),3),
                'path_index':len(self.path)-1,'frame_id':frame,'image':image,
                'instruction_stage_hypothesis':clause}
        node['anchor']=anchor
        node['annotations'].append({'name':node['name'],'caption':node['caption'],
                                    'path_index':len(self.path)-1,'anchor':anchor})

    def table(self):
        here=np.asarray(self.sim.get_agent_state().position,dtype=float)
        import quaternion
        forward=quaternion.as_rotation_matrix(self.sim.get_agent_state().rotation) @ np.array([0.,0.,-1.])
        facing=math.atan2(float(forward[2]),float(forward[0]))
        out=[]
        for n in self.nodes:
            anchor=n.get('anchor')
            target=np.asarray(anchor['position'] if anchor else n['position'],dtype=float)
            delta=target[[0,2]]-here[[0,2]]
            bearing=math.atan2(float(delta[1]),float(delta[0]))
            turn=(math.degrees(bearing-facing)+180)%360-180
            links=sorted(e['b'] if e['a']==n['id'] else e['a'] for e in self.edges if n['id'] in (e['a'],e['b']))
            path_index=anchor['path_index'] if anchor else n['path_index']
            walked=sum(float(np.linalg.norm(a[[0,2]]-b[[0,2]])) for a,b in zip(self.path[path_index:],self.path[path_index+1:]))
            out.append({'id':n['id'],'xy_m':[round(float(v),2) for v in np.asarray(n['position'])[[0,2]]],
                        'xy_from_start_m':[round(float(v),2) for v in (np.asarray(n['position'])-self.path[0])[[0,2]]],
                        'name':n['name'],'caption':n['caption'],'visits':n['visits'],'links':links,
                        'straight_m':round(float(np.linalg.norm(delta)),2),
                        'return_walked_m':round(walked,2),'turn_deg_left':round(-turn,1),
                        'path_index':n['path_index'],
                        'annotation_anchor':anchor,'annotations':n['annotations']})
        return out

    def context(self):
        pose=self.sim.get_agent_state()
        import quaternion
        direction=quaternion.as_rotation_matrix(pose.rotation) @ np.array([0.,0.,-1.])
        return {'current_pose':{'xyz_from_start_m':(np.asarray(pose.position)-self.path[0]).round(3).tolist(),
                                'heading_xz_deg':round(math.degrees(math.atan2(direction[2],direction[0])),1)},
                'nodes':self.table(),'edges':self.edges,
                'nearest_place':min(self.table(),key=lambda n:n['straight_m'])['id'],
                'unnamed_places':[n['id'] for n in self.table() if not n['name']],
                'topology':'NavGPT-3 derived place graph; edges are observed traversals',
                'return_walked_m':'length of recorded reverse path to the annotation anchor, or node if unnamed; not a shortest path',
                'turn_deg_left':'relative turn to face the anchor, positive left',
                'unknown_space':'never assumed free'}

    def render(self, path):
        mask=self.hits>0
        points=np.asarray([p[[0,2]] for p in self.path])
        coords=np.rint((points-self.low)/self.mpp).astype(int)
        seen=np.argwhere(mask)
        all_x=np.r_[coords[:,0],seen[:,1]] if len(seen) else coords[:,0]
        all_y=np.r_[coords[:,1],seen[:,0]] if len(seen) else coords[:,1]
        margin=int(1.5/self.mpp)
        x0=max(0,int(all_x.min())-margin);x1=min(self.shape[1],int(all_x.max())+margin+1)
        y0=max(0,int(all_y.min())-margin);y1=min(self.shape[0],int(all_y.max())+margin+1)
        if x1-x0<300:x1=min(self.shape[1],x0+300)
        if y1-y0<300:y1=min(self.shape[0],y0+300)
        hits=self.hits[y0:y1,x0:x1];sums=self.sums[y0:y1,x0:x1]
        yy,xx=np.indices(hits.shape)
        canvas=np.where(((xx//5+yy//5)%2)[...,None]==0,36,45).astype(np.uint8)
        canvas=np.repeat(canvas,3,axis=2)
        measured=hits>0
        canvas[measured]=(sums[measured]/hits[measured,None]).clip(0,255).astype(np.uint8)
        im=Image.fromarray(canvas).resize((min(1000,canvas.shape[1]*2),min(1000,canvas.shape[0]*2)))
        draw=ImageDraw.Draw(im)
        sx,sy=im.width/canvas.shape[1],im.height/canvas.shape[0]
        def xy(pos):
            q=np.rint((np.asarray(pos)[[0,2]]-self.low)/self.mpp).astype(int)
            return ((q[0]-x0)*sx,(q[1]-y0)*sy)
        if len(self.path)>1:draw.line([xy(p) for p in self.path],fill=(70,120,240),width=3)
        for n in self.nodes:
            x,y=xy(n['position']);draw.ellipse((x-10,y-10,x+10,y+10),fill=(255,190,50),outline='black',width=2)
            draw.text((x-4,y-7),str(n['id']),fill='black')
        x,y=xy(self.path[-1])
        import quaternion
        rot=quaternion.as_rotation_matrix(self.sim.get_agent_state().rotation)
        forward=rot @ np.array([0.,0.,-1.])
        norm=max(1e-6,math.hypot(forward[0],forward[2]))
        vx,vy=forward[0]/norm,forward[2]/norm
        arrow=[(x+vx*17,y+vy*17),(x-vx*9-vy*7,y-vy*9+vx*7),
               (x-vx*5,y-vy*5),(x-vx*9+vy*7,y-vy*9-vx*7)]
        draw.polygon(arrow,fill=(250,70,65),outline='black')

        for letter,dx,dz in (('F',vx,vy),('L',vy,-vx),('R',-vy,vx),('B',-vx,-vy)):
            draw.text((x+dx*35-5,y+dz*35-6),letter,fill='white',
                      stroke_width=2,stroke_fill='black')
        draw.text((8,8),'OBSERVED RGB-D  |  dark checker = unknown',fill='white',stroke_width=2,stroke_fill='black')
        draw.text((8,26),f'{len(self.nodes)} nodes  |  {round(float(mask.sum())*self.mpp*self.mpp,1)} m2 measured',fill='white',stroke_width=2,stroke_fill='black')
        bar=min(2/self.mpp*sx,im.width*.35)
        draw.line((im.width-20-bar,im.height-25,im.width-20,im.height-25),fill='white',width=4)
        draw.text((im.width-50,im.height-43),'2 m',fill='white',stroke_width=2,stroke_fill='black')
        im.save(path)
        return {'source':'four RGB-D cameras, measured cells only',
                'seen_area_m2':round(float(mask.sum())*self.mpp*self.mpp,2),
                'unknown_fraction_of_frame':round(1-float(measured.mean()),3),
                'meters_per_pixel':round(self.mpp/sx,4), 'nodes':self.table(),
                'image':str(path)}

    def return_target(self,node_id):
        if type(node_id) is not int or not 0 <= node_id < len(self.nodes):
            raise ValueError('node must already exist')
        node=self.nodes[node_id]
        return np.asarray(node['anchor']['position'] if node.get('anchor') else node['position'],dtype=float).copy()

    def actions_to(self,node_id,cap=80):
        if type(node_id) is not int or not 0 <= node_id < len(self.nodes):
            raise ValueError('node must already exist')
        node=self.nodes[node_id]
        stop=node['anchor']['path_index'] if node.get('anchor') else node['path_index']
        indices=list(range(len(self.path)-1,stop,-1))
        points=[self.path[index] for index in indices]
        points.append(self.path[stop])
        count=0
        for point in points:
            while count<cap:
                pose=self.sim.get_agent_state()
                here=np.asarray(pose.position,dtype=float)
                delta=np.asarray(point,dtype=float)-here
                planar=float(np.linalg.norm(delta[[0,2]]))
                if planar<.20:break
                import quaternion
                forward=quaternion.as_rotation_matrix(pose.rotation) @ np.array([0.,0.,-1.])
                facing=math.atan2(float(forward[2]),float(forward[0]))
                bearing=math.atan2(float(delta[2]),float(delta[0]))
                turn=(math.degrees(bearing-facing)+180)%360-180
                action=2 if turn< -7.5 else 3 if turn>7.5 else 1
                count+=1
                yield action
