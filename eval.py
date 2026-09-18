#!/usr/bin/env python3
"""Complete GPT-VLN evaluation: input, history, map, API, and Habitat loop."""
from __future__ import annotations

import argparse, base64, json, os, re, sys, time, urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
HABITAT = WORKSPACE / "StreamVLN/deps/habitat-lab"
AGENTVLN = WORKSPACE / "AgentVLN"
for path in (HABITAT / "habitat-lab", HABITAT / "habitat-baselines", AGENTVLN):
    sys.path.insert(0, str(path))

import habitat
from habitat import Env
from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat.utils.visualizations import fog_of_war, maps
from habitat_baselines.config.default import get_config
from agentvln.eval.habitat_utils.candidate_utils import build_visible_exploration_targets
from agentvln.eval.habitat_utils.coordinate_transformer import CoordinateTransformer
from agentvln.eval.habitat_utils.exploration_target_generator import ExplorationTargetGenerator
from agentvln.eval.habitat_utils.topdown_map_builder import (
    TopDownMapBuilder, convert_square_meters_to_pixel_area,
)
from gpt_vln.habitat_extensions import measures as _measures  # noqa: F401
from gpt_vln.visualization import (annotate_rgb, color_global_map, color_local_map,
    save_dashboard, save_depth, save_inference_input, save_route_topmap)

STOP, FORWARD, LEFT, RIGHT = 0, 1, 2, 3
UNKNOWN, FREE, OCCUPIED = np.uint8(127), np.uint8(255), np.uint8(0)
ACTION_RE = re.compile(r"\b(stop)|\b(forward)\s*(\d+(?:\.\d+)?)?|\bturn\s+(left|right)\s*(\d+(?:\.\d+)?)?", re.I)


@dataclass
class Settings:
    width: int = 640
    height: int = 480
    hfov: float = 90.0
    depth_max: float = 10.0
    mpp: float = 0.05
    local_m: float = 8.0
    robot_radius: float = 0.18
    forward_step: float = 0.25
    turn_step: float = 15.0
    min_unexplored_area_m2: float = 2.0
    obstacle_distance_m: float = 0.2
    min_target_spacing_m: float = 0.5
    max_frontier_targets: int = 5
    occlusion_tolerance_m: float = 0.2


@dataclass
class Waypoint:
    waypoint_id: int
    world_xyz: Tuple[float, float, float]
    pixel_xy: Tuple[int, int]
    bearing_deg: float
    distance_m: float
    frontier_length: float = 0.0

    def as_dict(self):
        return {"waypoint_id": self.waypoint_id, "world_xyz": list(self.world_xyz),
                "pixel_xy": list(self.pixel_xy), "bearing_deg": self.bearing_deg,
                "distance_m": self.distance_m, "frontier_length": self.frontier_length}


class History:
    def __init__(self, mode: str, count: int):
        self.mode, self.count, self.frames = mode, count, []

    def add(self, frame):
        self.frames.append(np.asarray(frame, dtype=np.uint8)[..., :3].copy())

    def select(self):
        if self.count <= 0 or not self.frames:
            return []
        if len(self.frames) <= self.count:
            return list(self.frames)
        if self.mode == "recent":
            return self.frames[-self.count:]
        if self.mode == "hybrid":
            tail_n = max(1, self.count // 2)
            head = self.frames[:-tail_n]
            ids = np.linspace(0, len(head) - 1, self.count - tail_n, dtype=int)
            return [head[i] for i in ids] + self.frames[-tail_n:]
        ids = np.linspace(0, len(self.frames) - 1, self.count, dtype=int)
        return [self.frames[i] for i in ids]


class MapProcessor:
    def __init__(self, sim, settings, normalized):
        self.sim, self.s, self.normalized = sim, settings, normalized
        state = sim.get_agent_state()
        self.builder = TopDownMapBuilder(sim, {
            "resolution": 512, "visible_radius": 8.0,
            "floor_management": {"floor_match_tolerance": 0.5},
        }, float(state.position[1]))
        self.generator = ExplorationTargetGenerator({
            "obstacle_distance_threshold": settings.obstacle_distance_m,
            "min_target_spacing": settings.min_target_spacing_m,
            "max_targets": settings.max_frontier_targets,
        })
        self.mpp = maps.calculate_meters_per_pixel(512, sim=sim)
        self.generator.set_pixel_scale(self.mpp)
        self.transformer = CoordinateTransformer({
            "width": settings.width, "height": settings.height,
            "hfov": settings.hfov, "camera_height": 1.25,
        }, sim)
        self.area_thresh = convert_square_meters_to_pixel_area(
            settings.min_unexplored_area_m2, 512, sim,
        )
        self.full = self.builder.full_map
        self.fog = self.builder.fog_of_war_mask
        self.level = float(state.position[1])
        self.frontier_stats = {}

    def depth_m(self, depth):
        value = np.asarray(depth, dtype=np.float32)
        if value.ndim == 3: value = value[..., 0]
        return value * self.s.depth_max if self.normalized else value

    def update(self):
        state = self.sim.get_agent_state()
        self.builder.update_visibility(state, fov=self.s.hfov)
        self.full, self.fog = self.builder.full_map, self.builder.fog_of_war_mask
        row, col = self.builder.get_agent_map_position(state.position)
        observed = self.fog
        return self._crop(state, observed), color_global_map(self.full, observed, (row, col))

    def _crop(self, state, observed):
        n = int(self.s.local_m / self.s.mpp); center = (n - 1) / 2
        rows, cols = np.indices((n, n), dtype=float)
        right, forward = (cols-center)*self.s.mpp, (center-rows)*self.s.mpp
        rv = quaternion_rotate_vector(state.rotation, np.array([1.,0.,0.]))
        fv = quaternion_rotate_vector(state.rotation, np.array([0.,0.,-1.]))
        wx = state.position[0] + right*rv[0] + forward*fv[0]
        wz = state.position[2] + right*rv[2] + forward*fv[2]
        lower, upper = self.sim.pathfinder.get_bounds()
        rr = np.floor((wz-lower[2])/((upper[2]-lower[2])/self.full.shape[0])).astype(int)
        cc = np.floor((wx-lower[0])/((upper[0]-lower[0])/self.full.shape[1])).astype(int)
        valid = (rr>=0)&(rr<self.full.shape[0])&(cc>=0)&(cc<self.full.shape[1])
        local = np.full((n,n), UNKNOWN, np.uint8); seen = np.zeros_like(valid)
        seen[valid] = observed[rr[valid],cc[valid]] > 0
        nav = np.zeros_like(valid); nav[seen] = self.full[rr[seen],cc[seen]] > 0
        local[seen&nav], local[seen&~nav] = FREE, OCCUPIED
        return local

    def waypoints(self, local, depth):
        """Direct AgentVLN ExplorationTargetGenerator candidate pipeline."""
        state = self.sim.get_agent_state()
        targets, _, self.frontier_stats = build_visible_exploration_targets(
            self.generator, self.transformer, self.builder, state, 0, depth,
            self.area_thresh, float(state.position[1]), 0.5,
            self.s.depth_max, self.s.occlusion_tolerance_m,
        )
        result = []
        forward = quaternion_rotate_vector(state.rotation, np.array([0., 0., -1.]))
        for index, target in enumerate(targets):
            world = np.asarray(target["world_coords"], dtype=float)
            delta = world - np.asarray(state.position)
            cross = forward[2]*delta[0] - forward[0]*delta[2]
            dot = forward[0]*delta[0] + forward[2]*delta[2]
            result.append(Waypoint(
                index+1, tuple(world.tolist()), tuple(target["pixel_coords"]),
                float(np.degrees(np.arctan2(cross, dot))),
                float(np.linalg.norm(delta[[0, 2]])), 0.0,
            ))
        return result


def parse_model_output(text, settings):
    """AgentVLN protocol: one frontier/target coordinate, actions, or STOP."""
    value = text.strip()
    result = {"task_type": "unknown", "coordinate": None, "action_sequence": None,
              "raw_text": value, "parse_success": False}
    if value.upper() == "STOP" or "<stop>" in value.lower():
        result.update(task_type="stop", parse_success=True)
        return result
    coordinate = re.search(r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)", value)
    if coordinate:
        task_type = "target" if "<target>" in value.lower() else "frontier"
        result.update(task_type=task_type,
                      coordinate=[float(coordinate.group(1)), float(coordinate.group(2))],
                      parse_success=True)
        return result
    action_map = {"FORWARD": FORWARD, "TURN_LEFT": LEFT, "TURN_RIGHT": RIGHT, "STOP": STOP}
    actions = [action_map[token] for token in re.findall(r"FORWARD|TURN_LEFT|TURN_RIGHT|STOP", value.upper())]
    if actions:
        result.update(task_type="action", action_sequence=actions, parse_success=True)
    return result


class Policy:
    def __init__(self,args,settings):
        self.a,self.s=args,settings; self.key=os.environ.get(args.api_key_env,"")
        if args.use_api and not self.key: raise RuntimeError(f"${args.api_key_env} is empty")

    def call(self,instruction,history,current,waypoints,outcomes):
        coords = [[w.pixel_xy[0], w.pixel_xy[1]] for w in waypoints]
        prompt=("You are an autonomous navigation assistant. "
          f"Your task is to {instruction}. Where should you go next to stay on track? "
          f"Select one suitable waypoint coordinate from <frontiers_coord>{coords}. "
          "Output exactly <frontiers_coord>(u, v). If the destination itself is visible, "
          "output <target>(u, v). If no suitable coordinate exists, output "
          "<action>FORWARD TURN_LEFT TURN_RIGHT</action> using only needed actions. "
          "Output STOP only when the task is complete. "
          f"The first {len(history)} images are historical observations and the last image is current. "
          f"Previous execution outcomes: {json.dumps(outcomes[-12:])}")
        model_input={"model":self.a.model,"prompt":prompt,"history_strategy":self.a.history_strategy,
                     "history_frames":len(history),"waypoints":[w.as_dict() for w in waypoints]}
        start=time.perf_counter()
        if not self.a.use_api: raw="STOP"
        else:
            def url(frame):
                ok,data=cv2.imencode(".jpg",cv2.cvtColor(frame,cv2.COLOR_RGB2BGR));
                if not ok: raise RuntimeError("image encoding failed")
                return "data:image/jpeg;base64,"+base64.b64encode(data).decode()
            content=[{"type":"input_text","text":prompt}]+[{"type":"input_image","image_url":url(f)} for f in history]+[{"type":"input_image","image_url":url(current)}]
            body={"model":self.a.model,"input":[{"role":"user","content":content}],"max_output_tokens":self.a.max_output_tokens}
            request=urllib.request.Request(self.a.base_url.rstrip("/")+"/responses",data=json.dumps(body).encode(),headers={"Authorization":"Bearer "+self.key,"Content-Type":"application/json","Accept":"application/json","User-Agent":"OpenAI/Python GPT-VLN","X-Stainless-Lang":"python"},method="POST")
            with urllib.request.urlopen(request,timeout=self.a.api_timeout) as response: payload=json.loads(response.read().decode())
            raw=payload.get("output_text","") or "".join(p.get("text","") for i in payload.get("output",[]) for p in i.get("content",[]) if p.get("type")=="output_text")
        parsed=parse_model_output(raw,self.s)
        return {"model_input":model_input,"raw_output":raw,"parsed":parsed,"latency_s":time.perf_counter()-start}


def clearance(depth,s):
    h,w=depth.shape; strip=depth[int(.3*h):int(.8*h)]; focal=w/(2*np.tan(np.deg2rad(s.hfov)/2)); offsets=np.abs(np.arange(w)[None,:]-(w-1)/2)*strip/focal
    values=strip[np.isfinite(strip)&(strip>0)&(offsets<=s.robot_radius)]
    return float(np.percentile(values,5)) if values.size>=30 else float("inf")


def execute(env,actions,processor,obs,remaining,s):
    done=[]; frames=[]; trajectory=[]; start=np.asarray(env.sim.get_agent_state().position); status="chunk_complete"; depth=processor.depth_m(obs["depth"])
    for action in list(actions)[:remaining]:
        if action==FORWARD and clearance(depth,s)<.32: status="blocked"; break
        obs=env.step(int(action)); done.append(int(action)); frames.append(np.asarray(obs["rgb"])[...,:3].copy()); trajectory.append(np.asarray(env.sim.get_agent_state().position).tolist()); depth=processor.depth_m(obs["depth"])
        if action==STOP or env.episode_over: status="stopped" if action==STOP else "episode_over"; break
    final=np.asarray(env.sim.get_agent_state().position)
    return obs,{"status":status,"actions":done,"trajectory":trajectory,"final_position":final.tolist(),"travelled_m":float(np.linalg.norm(final[[0,2]]-start[[0,2]]))},frames


def pixel_to_world(pixel, waypoints, observation, sim, processor, settings):
    """Match an AgentVLN candidate first, then back-project through depth."""
    point = np.asarray(pixel, dtype=float)
    if waypoints:
        nearest = min(waypoints, key=lambda item: np.linalg.norm(point - np.asarray(item.pixel_xy)))
        if np.linalg.norm(point - np.asarray(nearest.pixel_xy)) < 15.0:
            return np.asarray(nearest.world_xyz, dtype=float)
    u, v = int(round(point[0])), int(round(point[1]))
    depth = processor.depth_m(observation["depth"])
    if not (0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]):
        return None
    z = float(depth[v, u])
    if not np.isfinite(z) or z <= 0:
        return None
    focal = settings.width / (2 * np.tan(np.deg2rad(settings.hfov) / 2))
    camera_point = np.array([(u-(settings.width-1)/2)*z/focal,
                             -((v-(settings.height-1)/2)*z/focal), -z])
    state = sim.get_agent_state()
    sensor = state.sensor_states.get("rgb") or state.sensor_states.get("rgb_sensor")
    world = np.asarray(sensor.position) + quaternion_rotate_vector(sensor.rotation, camera_point)
    snapped = np.asarray(sim.pathfinder.snap_point(world))
    return None if np.isnan(snapped).any() else snapped


def execute_navigation_skill(env, target, observation, remaining, max_skill_steps=50):
    """AgentVLN navigation skill: ShortestPathFollower owns primitive actions."""
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
    requested_target = np.asarray(target, dtype=float)
    target = np.asarray(env.sim.pathfinder.snap_point(requested_target))
    if np.isnan(target).any():
        return observation, {"status": "target_not_navigable", "skill": "ShortestPathFollower",
            "actions": [], "trajectory": []}, []
    follower = ShortestPathFollower(env.sim, goal_radius=0.5, return_one_hot=False)
    actions, frames, trajectory = [], [], []
    start = np.asarray(env.sim.get_agent_state().position).copy()
    status = "skill_complete"
    for _ in range(min(max_skill_steps, remaining)):
        action = follower.get_next_action(target)
        if action is None or int(action) == STOP:
            status = "waypoint_reached"
            break
        observation = env.step(int(action)); actions.append(int(action))
        frames.append(np.asarray(observation["rgb"])[..., :3].copy())
        trajectory.append(np.asarray(env.sim.get_agent_state().position).tolist())
        if env.episode_over:
            status = "episode_over"
            break
    final = np.asarray(env.sim.get_agent_state().position).copy()
    return observation, {"status": status, "skill": "ShortestPathFollower",
        "requested_target_world": requested_target.tolist(),
        "target_world": np.asarray(target).tolist(),
        "target_height_correction_m": float(target[1] - requested_target[1]),
        "actions": actions,
        "trajectory": trajectory,
        "final_position": final.tolist(),
        "travelled_m": float(np.linalg.norm(final[[0,2]]-start[[0,2]]))}, frames


def create_env(args,s):
    cfg=get_config(str(ROOT/"config/vln_r2r.yaml"))
    with habitat.config.read_write(cfg):
        cfg.habitat.dataset.split=args.split; cfg.habitat.dataset.data_path=str(WORKSPACE/"StreamVLN/data/datasets/r2r/{split}/{split}.json.gz"); cfg.habitat.dataset.scenes_dir=str(WORKSPACE/"StreamVLN/data/scene_datasets")
        cfg.habitat.environment.max_episode_steps=args.max_actions; sim=cfg.habitat.simulator; sim.forward_step_size=s.forward_step; sim.turn_angle=int(s.turn_step)
        sensors=sim.agents.main_agent.sim_sensors
        for sensor in (sensors.rgb_sensor,sensors.depth_sensor): sensor.width,sensor.height,sensor.hfov=s.width,s.height,int(s.hfov)
        normalized=bool(getattr(sensors.depth_sensor,"normalize_depth",True))
    return Env(cfg),normalized


def jsonable(v):
    if isinstance(v,dict): return {str(k):jsonable(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [jsonable(x) for x in v]
    if isinstance(v,np.ndarray): return v.tolist()
    if isinstance(v,np.generic): return v.item()
    return v


def align_start_heading(env, episode):
    """Rotate in place toward the first non-trivial GT reference-path point."""
    import quaternion
    state = env.sim.get_agent_state()
    position = np.asarray(state.position, dtype=float)
    target = next(
        (np.asarray(point, dtype=float) for point in getattr(episode, "reference_path", [])
         if np.linalg.norm(np.asarray(point, dtype=float)[[0, 2]] - position[[0, 2]]) > 0.25),
        None,
    )
    if target is None:
        return None
    direction = target - position
    yaw = float(np.arctan2(-direction[0], -direction[2]))
    rotation = quaternion.from_rotation_vector(np.array([0.0, yaw, 0.0]))
    env.sim.set_agent_state(position, rotation, reset_sensors=True)
    return {"target": target.tolist(), "yaw_deg": float(np.degrees(yaw))}


def run_episode(env,policy,args,s,normalized):
    obs=env.reset(); episode=env.current_episode; out=args.output/f"episode_{episode.episode_id}"; out.mkdir(parents=True,exist_ok=True)
    alignment = align_start_heading(env, episode) if args.align_start_heading else None
    if alignment is not None:
        obs = env.sim.get_sensor_observations()
    processor=MapProcessor(env.sim,s,normalized); history=History(args.history_strategy,args.history_frames)
    raw_dir=out/"raw_action_frames"; raw_dir.mkdir(parents=True,exist_ok=True)
    trace=[]; outcomes=[]; action_frames=[]; positions=[np.asarray(env.sim.get_agent_state().position).copy()]; inference_points=[]; used=0
    for turn in range(args.max_turns):
        local,global_map=processor.update(); waypoints=processor.waypoints(local,obs["depth"]); rgb=np.asarray(obs["rgb"])[...,:3]; current=annotate_rgb(rgb,waypoints); selected=history.select(); local_color=color_local_map(local,waypoints,s.mpp)
        if args.visualize:
            save_inference_input(out/f"turn_{turn:02d}_input.png",selected,current); cv2.imwrite(str(out/f"turn_{turn:02d}_rgb.png"),cv2.cvtColor(current,cv2.COLOR_RGB2BGR)); cv2.imwrite(str(out/f"turn_{turn:02d}_local_map.png"),local_color); cv2.imwrite(str(out/f"turn_{turn:02d}_global_map.png"),global_map); save_depth(out/f"turn_{turn:02d}_depth.png",processor.depth_m(obs["depth"]),s.depth_max); save_dashboard(out/f"turn_{turn:02d}_dashboard.png",current,local_color,episode.instruction.instruction_text)
        decision=policy.call(episode.instruction.instruction_text,selected,current,waypoints,outcomes)
        decision["model_input"]["frontier_stats"] = processor.frontier_stats
        parsed=decision["parsed"]; frames=[]
        if args.dry_run: execution={"status":"dry_run","actions":[]}
        elif parsed["task_type"]=="stop":
            obs=env.step(STOP); stop_frame=np.asarray(obs["rgb"])[...,:3].copy(); frames=[stop_frame]
            stop_pos=np.asarray(env.sim.get_agent_state().position).tolist()
            execution={"status":"stopped","actions":[STOP],"trajectory":[stop_pos],"final_position":stop_pos,"travelled_m":0.0}
        elif parsed.get("coordinate") is not None:
            target=pixel_to_world(parsed["coordinate"],waypoints,obs,env.sim,processor,s)
            if target is None: execution={"status":"coordinate_unresolved","actions":[]}
            else: obs,execution,frames=execute_navigation_skill(env,target,obs,args.max_actions-used)
        elif parsed.get("action_sequence"):
            obs,execution,frames=execute(env,parsed["action_sequence"],processor,obs,args.max_actions-used,s)
        else:
            obs,execution,frames=execute(env,[FORWARD],processor,obs,args.max_actions-used,s)
            execution["status"]="fallback_forward" if execution["status"]=="chunk_complete" else execution["status"]
        used+=len(execution["actions"])
        first_action_index=len(action_frames)
        for offset,(action,frame) in enumerate(zip(execution.get("actions",[]),frames)):
            frame_index=first_action_index+offset
            frame_path=raw_dir/f"action_{frame_index:04d}.png"
            cv2.imwrite(str(frame_path),cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
            history.add(frame)
            action_frames.append({"frame_index":frame_index,"turn":turn,"primitive_index":offset,
                                  "action":int(action),"path":str(frame_path.relative_to(out))})
        (raw_dir/"index.json").write_text(json.dumps(action_frames,indent=2))
        positions.extend(np.asarray(point) for point in execution.get("trajectory", []))
        if not execution.get("trajectory") and execution.get("final_position") is not None:
            positions.append(np.asarray(execution["final_position"]))
        inference_points.append((turn,np.asarray(env.sim.get_agent_state().position).copy()))
        item={"turn":turn,**decision,"execution":execution}; trace.append(item); outcomes.append({"prediction":decision["raw_output"],"status":execution["status"],"travelled_m":execution.get("travelled_m",0)})
        (out/f"turn_{turn:02d}_inference.json").write_text(json.dumps(jsonable(item),indent=2)); (out/"trace.json").write_text(json.dumps(jsonable(trace),indent=2))
        if args.dry_run or env.episode_over or execution["status"]=="stopped": break
    if not args.dry_run and not env.episode_over: env.step(STOP)
    if args.visualize: save_route_topmap(out/"top_map_gt_pred.png",env.sim,episode,positions,inference_points)
    result={"episode_id":str(episode.episode_id),"scene_id":episode.scene_id,"instruction":episode.instruction.instruction_text,"start_alignment":alignment,"turns":len(trace),"total_actions":used,"history":{"strategy":args.history_strategy,"stored":len(history.frames)},"metrics":jsonable(env.get_metrics())}
    (out/"result.json").write_text(json.dumps(result,indent=2)); return result


def parser():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--episode-id",default="412"); p.add_argument("--num-episodes",type=int,default=0); p.add_argument("--split",default="val_unseen"); p.add_argument("--output",type=Path,default=Path("outputs/eval")); p.add_argument("--model",default="gpt-6"); p.add_argument("--base-url",default="https://api.openai.com/v1"); p.add_argument("--api-key-env",default="OPENAI_API_KEY"); p.add_argument("--api-timeout",type=float,default=180); p.add_argument("--max-output-tokens",type=int,default=2048); p.add_argument("--use-api",action=argparse.BooleanOptionalAction,default=True); p.add_argument("--visualize",action=argparse.BooleanOptionalAction,default=True); p.add_argument("--align-start-heading",action=argparse.BooleanOptionalAction,default=False); p.add_argument("--hfov",type=float,default=90.0); p.add_argument("--history-strategy",choices=("uniform","recent","hybrid"),default="uniform"); p.add_argument("--history-frames",type=int,default=8); p.add_argument("--max-turns",type=int,default=10); p.add_argument("--max-actions",type=int,default=500); p.add_argument("--dry-run",action="store_true"); return p


def main():
    args=parser().parse_args(); args.output=args.output.resolve(); args.output.mkdir(parents=True,exist_ok=True); s=Settings(hfov=args.hfov); policy=Policy(args,s); env,normalized=create_env(args,s)
    try:
        episodes=list(env.episodes)
        if args.episode_id.lower()!="all": episodes=[e for e in episodes if str(e.episode_id)==args.episode_id]
        elif args.num_episodes>0: episodes=episodes[:args.num_episodes]
        if not episodes: raise ValueError(f"no episode matched {args.episode_id}")
        env.episodes=episodes; results=[]
        for i in range(len(episodes)):
            result=run_episode(env,policy,args,s,normalized); results.append(result)
            with (args.output/"results.jsonl").open("a") as f: f.write(json.dumps(result)+"\n")
            print(f"[{i+1}/{len(episodes)}] episode={result['episode_id']} metrics={result['metrics']}")
        keys=sorted({k for r in results for k,v in r["metrics"].items() if isinstance(v,(int,float))}); summary={"episodes":len(results),"averages":{k:float(np.mean([r["metrics"][k] for r in results])) for k in keys}}; (args.output/"summary.json").write_text(json.dumps(summary,indent=2))
    finally: env.close()

if __name__=="__main__": main()
