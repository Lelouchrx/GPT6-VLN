#!/usr/bin/env python3
"""Complete GPT-VLN evaluation: input, history, map, API, and Habitat loop."""
from __future__ import annotations

import argparse, base64, json, multiprocessing as mp, os, re, subprocess, sys, tempfile, time, urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

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
from habitat.utils.visualizations import maps
from habitat_baselines.config.default import get_config
from agentvln.eval.habitat_utils.candidate_utils import build_visible_exploration_targets
from agentvln.eval.habitat_utils.coordinate_transformer import CoordinateTransformer
from agentvln.eval.habitat_utils.exploration_target_generator import ExplorationTargetGenerator
from agentvln.eval.habitat_utils.floor_map_manager import FloorMapManager
from agentvln.eval.habitat_utils.topdown_map_builder import convert_square_meters_to_pixel_area
from gpt_vln.habitat_extensions import measures as _measures  # noqa: F401
from gpt_vln.visualization import (color_global_map, color_local_map, overlay_navigation,
    overlay_selected_waypoint,
    save_dashboard, save_depth, save_inference_input, save_route_topmap)

STOP, FORWARD, LEFT, RIGHT = 0, 1, 2, 3
UNKNOWN, FREE, OCCUPIED = np.uint8(127), np.uint8(255), np.uint8(0)

# --- safety valve: hard-stop after too many consecutive blocked FORWARD steps --------
# execute() skips (not aborts) a FORWARD whose clearance()<.32 so queued TURN actions in
# the same batch still run (staircases/railings: see outputs/claude_opus_val_unseen_10
# episodes 4/5/6, where frontier_stats.raw==0 for the whole climb and the model was stuck
# on <action> fallback). But if the model keeps re-issuing FORWARD into the same wall
# turn after turn, that skip can repeat indefinitely with zero progress. This constant
# caps it regardless of any later prompt/tuning change -- do not remove without also
# fixing whatever caused the regression that made this trip.
MAX_CONSECUTIVE_BLOCKED_STEPS = 5
MAX_ATOMIC_ACTIONS_PER_TURN = 20

# --- action protocol ------------------------------------------------------------------
# The only <action> grammar: up to MACRO_MAX_ACTIONS magnitude-bearing moves
# (FORWARD(cm), TURN_LEFT/RIGHT(deg), STOP) per call. The magnitude is mandatory --
# bare tokens used to be accepted as one atomic 15deg/25cm step, which burned an API
# call per quarter-metre; parse_model_output() now rejects them. Magnitudes expand into
# the same flat FORWARD/LEFT/RIGHT/STOP action_sequence execute() already consumes, so
# the rest of the pipeline (execute(), run_episode() dispatch, env stepping) is untouched.
MACRO_TOKEN_RE = re.compile(
    r"\b(FORWARD|TURN_LEFT|TURN_RIGHT|STOP)\s*(?:\(\s*(\d+(?:\.\d+)?)\s*\)|(\d+(?:\.\d+)?))?"
)
MACRO_MAX_ACTIONS = 3
MACRO_FORWARD_CM_RANGE = (25.0, 75.0)
MACRO_TURN_DEG_RANGE = (15.0, 45.0)

# Reasoning models narrate their plan before emitting the tag (e.g. "...both
# turn+forward pairs, ... left+three-forwards. <action>TURN_RIGHT FORWARD
# FORWARD FORWARD</action>"). Scanning the *whole* raw_output for action/coord
# tokens sweeps up every stray mention of "forward" in that prose, so
# token extraction MUST be scoped to the actual tagged block, never the full text.
ACTION_BLOCK_RE = re.compile(r"<action>(.*?)</action>", re.I | re.S)
PROPOSAL_BLOCK_RE = re.compile(r"<proposal>(.*?)</proposal>", re.I | re.S)
FINAL_STOP_RE = re.compile(r"(?:^|\n)\s*(?:<stop>|STOP)\s*$", re.I)
COORD_TAG_RE = re.compile(
    r"<(frontiers_coord|target|waypoint)>\s*\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)", re.I)
VIEW_COORD_RE = re.compile(
    r"\b(LEFT|FRONT|RIGHT)\s*\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)", re.I)


@dataclass
class Settings:
    width: int = 640
    height: int = 480
    hfov: float = 120.0
    depth_max: float = 10.0
    mpp: float = 0.05
    local_history_steps: int = 8
    robot_radius: float = 0.18
    forward_step: float = 0.25
    turn_step: float = 15.0
    min_unexplored_area_m2: float = 2.0
    obstacle_distance_m: float = 0.2
    min_target_spacing_m: float = 0.5
    max_frontier_targets: int = 5
    occlusion_tolerance_m: float = 0.2
    target_max_snap_drift_m: float = 0.75

    # Recent RGB-D local geometry. Poses come from Habitat here; a real robot supplies
    # the same transform from visual-inertial odometry or SLAM.
    local_mpp: float = 0.05
    local_depth_max_m: float = 5.0
    local_surface_height_span_m: float = 0.35
    local_depth_stride: int = 8


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


def build_rgb_panorama(sim, observation):
    """Return three separate 120-degree views covering 360 degrees."""
    import quaternion
    state = sim.get_agent_state()
    panels = []
    for label, degrees in (("LEFT", 120), ("FRONT", 0), ("RIGHT", -120)):
        rotation = quaternion.from_rotation_vector(
            np.array([0.0, np.deg2rad(degrees), 0.0])) * state.rotation
        view = (observation if degrees == 0 else
                sim.get_observations_at(state.position, rotation, keep_agent_at_new_pose=False))
        panels.append({"label": label, "rotation": rotation, "depth": view["depth"],
                       "rgb": np.asarray(view["rgb"])[..., :3]})
    return panels


class MapProcessor:
    def __init__(self, sim, settings, normalized):
        self.sim, self.s, self.normalized = sim, settings, normalized
        state = sim.get_agent_state()
        map_config = {
            "resolution": 512, "visible_radius": 8.0,
            "floor_management": {
                "enabled": True,
                "floor_match_tolerance": 0.5,
                "floor_departure_threshold": 0.75,
                "min_floor_separation": 1.5,
                "stable_window_steps": 3,
                "stable_height_tolerance": 0.12,
                "min_navigable_component_area_m2": 2.0,
            },
        }
        self.floor_manager = FloorMapManager(
            sim, map_config, initial_height=float(state.position[1]),
        )
        self.builder = self.floor_manager.active_builder
        self.generator = ExplorationTargetGenerator({
            "obstacle_distance_threshold": settings.obstacle_distance_m,
            "min_target_spacing": settings.min_target_spacing_m,
            "max_targets": settings.max_frontier_targets,
        })
        self.mpp = maps.calculate_meters_per_pixel(512, sim=sim)
        self.local_mpp = float(settings.local_mpp)
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
        self.floor_update = self.floor_manager.update(state)
        self.frontier_stats = {}
        self.recent_depth = deque(maxlen=settings.local_history_steps)
        self._last_visibility_pose = None
        self._last_depth_pose = None

    def depth_m(self, depth):
        value = np.asarray(depth, dtype=np.float32)
        if value.ndim == 3: value = value[..., 0]
        return value * self.s.depth_max if self.normalized else value

    def update(self, depth):
        state = self.sim.get_agent_state()
        self.record_visibility(state)
        self.record_depth(depth, state)
        row, col = self.builder.get_agent_map_position(state.position)
        return (self.local_from_recent_depth(state),
                color_global_map(self.full, self.fog, (row, col)))

    def record_visibility(self, state=None):
        state = state or self.sim.get_agent_state()
        self.floor_update = self.floor_manager.update(state)
        self.builder = self.floor_manager.active_builder
        self.full = self.builder.full_map
        forward = quaternion_rotate_vector(state.rotation, np.array([0., 0., -1.]))
        pose = tuple(np.round(np.r_[state.position, forward], 4))
        if pose == self._last_visibility_pose:
            self.fog = self.builder.fog_of_war_mask
            return
        cumulative = self.builder.fog_of_war_mask.copy()
        self.builder.fog_of_war_mask = np.zeros_like(cumulative)
        current = self.builder.update_visibility(state, fov=self.s.hfov).copy()
        self.builder.fog_of_war_mask = cv2.bitwise_or(cumulative, current)
        self.fog = self.builder.fog_of_war_mask
        self._last_visibility_pose = pose

    def record_depth(self, depth, state=None):
        """Back-project one metric depth frame and retain the latest action steps."""
        import quaternion
        state = state or self.sim.get_agent_state()
        forward = quaternion_rotate_vector(state.rotation, np.array([0., 0., -1.]))
        pose_key = tuple(np.round(np.r_[state.position, forward], 4))
        if pose_key == self._last_depth_pose:
            return
        sensor = (state.sensor_states.get("depth") or state.sensor_states.get("depth_sensor")
                  or state.sensor_states.get("rgb") or state.sensor_states.get("rgb_sensor"))
        values = self.depth_m(depth)
        stride = max(1, int(self.s.local_depth_stride))
        vs, us = np.indices(values.shape, dtype=np.float32)
        mask = np.zeros_like(values, dtype=bool)
        mask[::stride, ::stride] = True
        mask &= (np.isfinite(values) & (values > .05)
                 & (values <= self.s.local_depth_max_m))
        z = values[mask].astype(np.float64)
        if not z.size:
            return
        focal = values.shape[1]/(2*np.tan(np.deg2rad(self.s.hfov)/2))
        camera = np.stack(((us[mask]-(values.shape[1]-1)/2)*z/focal,
                           -(vs[mask]-(values.shape[0]-1)/2)*z/focal, -z), axis=1)
        rotation = np.asarray(quaternion.as_rotation_matrix(sensor.rotation))
        world = np.asarray(sensor.position, dtype=float) + camera @ rotation.T
        self.recent_depth.append({"origin": np.asarray(sensor.position, dtype=float),
                                  "points": world})
        self._last_depth_pose = pose_key

    def local_from_recent_depth(self, state=None):
        """Fuse recent depth rays into local geometry without planning on the map."""
        state = state or self.sim.get_agent_state()
        mpp = self.local_mpp
        origin = np.asarray(state.position, dtype=float)
        right = quaternion_rotate_vector(state.rotation, np.array([1., 0., 0.]))[[0,2]]
        forward = quaternion_rotate_vector(state.rotation, np.array([0., 0., -1.]))[[0,2]]
        right /= max(float(np.linalg.norm(right)), 1e-9)
        forward /= max(float(np.linalg.norm(forward)), 1e-9)
        if not self.recent_depth:
            return np.full((41, 41), UNKNOWN, np.uint8)

        projected, ray_origins = [], []
        radius = 1.0
        for frame in self.recent_depth:
            delta = frame["points"]-origin
            xy = np.column_stack((delta[:,[0,2]] @ right,
                                  delta[:,[0,2]] @ forward))
            camera_delta = frame["origin"]-origin
            camera_xy = np.array([camera_delta[[0,2]] @ right,
                                  camera_delta[[0,2]] @ forward])
            projected.append((xy, delta[:,1]))
            ray_origins.append(camera_xy)
            radius = max(radius, float(np.max(np.abs(xy))))
        radius = min(self.s.local_depth_max_m, radius+mpp)
        n = max(41, int(np.ceil(2*radius/mpp)) | 1)
        center = (n-1)/2
        free = np.zeros((n,n), np.uint8)
        min_height = np.full(n*n, np.inf)
        max_height = np.full(n*n, -np.inf)
        counts = np.zeros(n*n, np.int32)

        for (xy, heights), camera_xy in zip(projected, ray_origins):
            cols = np.round(center+xy[:,0]/mpp).astype(int)
            rows = np.round(center-xy[:,1]/mpp).astype(int)
            camera_col = int(np.clip(round(center+camera_xy[0]/mpp), 0, n-1))
            camera_row = int(np.clip(round(center-camera_xy[1]/mpp), 0, n-1))
            valid = (rows>=0)&(rows<n)&(cols>=0)&(cols<n)
            for row, col in zip(rows[valid], cols[valid]):
                cv2.line(free, (camera_col,camera_row), (int(col),int(row)), 1, 1)
            flat = rows[valid]*n+cols[valid]
            np.minimum.at(min_height, flat, heights[valid])
            np.maximum.at(max_height, flat, heights[valid])
            np.add.at(counts, flat, 1)

        local = np.full((n,n), UNKNOWN, np.uint8)
        local[free>0] = FREE
        vertical_span = (max_height-min_height).reshape(n,n)
        surface = ((counts.reshape(n,n)>=2)
                   & (vertical_span>=self.s.local_surface_height_span_m))
        local[surface] = OCCUPIED
        return local

    def waypoints(self, local, depth):
        """Direct AgentVLN ExplorationTargetGenerator candidate pipeline."""
        state = self.sim.get_agent_state()
        if self.floor_update.in_transition:
            self.frontier_stats = {
                "raw": 0, "wrong_floor": 0, "out_of_view": 0,
                "occluded_or_invalid_depth": 0, "kept": 0,
                "floor_transition": True,
            }
            return []
        targets, _, self.frontier_stats = build_visible_exploration_targets(
            self.generator, self.transformer, self.builder, state, 0, depth,
            self.area_thresh, self.floor_update.floor_height,
            self.floor_manager.floor_match_tolerance,
            self.s.depth_max, self.s.occlusion_tolerance_m,
        )
        self.frontier_stats.update({
            "floor_id": self.floor_update.floor_id,
            "floor_height": self.floor_update.floor_height,
            "floor_transition": False,
            "floor_switched": self.floor_update.switched,
            "floor_created": self.floor_update.created,
            "known_floors": self.floor_manager.get_floor_metadata(),
        })
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
    """One annotated frontier or visible-floor target, macro actions, or STOP.

    Up to MACRO_MAX_ACTIONS moves - FORWARD(cm) in MACRO_FORWARD_CM_RANGE,
    TURN_LEFT/RIGHT(deg) in MACRO_TURN_DEG_RANGE, or STOP - expanded into atomic
    FORWARD/TURN_LEFT/TURN_RIGHT/STOP steps sized by settings.forward_step/turn_step.
    A magnitude is mandatory: a bare token fails the parse instead of silently
    becoming a minimum-size step, so no turn can spend an API call on 25cm/15deg the
    model never asked for. Coordinate and STOP outputs are unaffected.
    """
    value = text.strip()
    result = {"task_type": "unknown", "coordinate": None, "coordinate_view": None,
              "action_sequence": None,
              "raw_text": value, "parse_success": False, "macro_tokens": []}
    if value.upper() == "STOP" or FINAL_STOP_RE.search(value):
        result.update(task_type="stop", parse_success=True)
        return result
    coordinate = COORD_TAG_RE.search(value)
    if coordinate:
        tag = coordinate.group(1).lower()
        task_type = "frontier" if tag == "frontiers_coord" else "waypoint"
        result.update(task_type=task_type,
                      coordinate=[float(coordinate.group(2)), float(coordinate.group(3))],
                      parse_success=True)
    view_coordinate = VIEW_COORD_RE.search(value)
    if view_coordinate:
        result.update(task_type="waypoint", coordinate_view=view_coordinate.group(1).upper(),
                      coordinate=[float(view_coordinate.group(2)),
                                  float(view_coordinate.group(3))], parse_success=True)
    action_block = ACTION_BLOCK_RE.search(value)
    action_text = action_block.group(1) if action_block else value
    if action_block is None:
        # No <action> tag: a reasoning preamble (Analysis:/Observation:/Goal-progress:/
        # Done? or free prose) may precede the real answer, blank-line separated from it
        # -- the same convention already relied on for a bare STOP after reasoning (see
        # FINAL_STOP_RE). Scope the untagged-action residue check to that last paragraph
        # only, not the whole preamble, or reasoning text left over in `command_only`
        # would fail every untagged action turn once a reasoning_clause is prepended.
        paragraphs = re.split(r"\n\s*\n", value)
        if len(paragraphs) > 1:
            action_text = paragraphs[-1]
        command_only = VIEW_COORD_RE.sub("", action_text)
        command_only = re.sub(r"<progress>\s*\d+\s*</progress>", "", command_only,
                              flags=re.I)
        residue = MACRO_TOKEN_RE.sub("", command_only)
        if residue.strip(" \t\r\n,"):
            return result
    action_map = {"FORWARD": FORWARD, "TURN_LEFT": LEFT, "TURN_RIGHT": RIGHT, "STOP": STOP}
    atomic, macro_tokens = [], []
    for name, paren_mag, bare_mag in MACRO_TOKEN_RE.findall(action_text.upper()):
        if len(macro_tokens) >= MACRO_MAX_ACTIONS:
            break
        if name == "STOP":
            macro_tokens.append({"type": "STOP"})
            atomic.append(STOP)
            break  # STOP ends the sequence; nothing after it executes
        magnitude = paren_mag or bare_mag
        if not magnitude:
            result.update(macro_tokens=[{"type": name, "error": "missing magnitude"}])
            return result
        requested = float(magnitude)
        if name == "FORWARD":
            lo, hi = MACRO_FORWARD_CM_RANGE
            cm = min(max(requested, lo), hi)
            steps = max(1, round(cm / 100.0 / settings.forward_step))
            macro_tokens.append({"type": name, "requested_cm": requested, "clamped_cm": cm, "atomic_steps": steps})
        else:  # TURN_LEFT / TURN_RIGHT
            lo, hi = MACRO_TURN_DEG_RANGE
            deg = min(max(requested, lo), hi)
            steps = max(1, round(deg / settings.turn_step))
            macro_tokens.append({"type": name, "requested_deg": requested, "clamped_deg": deg, "atomic_steps": steps})
        atomic.extend([action_map[name]] * steps)
    if atomic:
        if result["task_type"] == "unknown":
            result["task_type"] = "action"
        result.update(action_sequence=atomic, macro_tokens=macro_tokens, parse_success=True)
    return result


def _proposal_command_key(parsed):
    coordinate = parsed.get("coordinate")
    return (
        parsed.get("task_type"),
        tuple(coordinate) if isinstance(coordinate, (list, tuple)) else None,
        parsed.get("coordinate_view"),
        tuple(parsed.get("action_sequence") or ()),
    )


def parse_proposal_blocks(text, settings, max_proposals=5):
    """Extract up to max_proposals diverse <proposal> blocks from one model reply."""
    blocks = [block.strip() for block in PROPOSAL_BLOCK_RE.findall(text) if block.strip()]
    proposals, seen = [], set()
    for block in blocks[:max(1, int(max_proposals))]:
        # Keep full block (with Analysis) for Jev; parse only the command section.
        paragraphs = re.split(r"\n\s*\n", block)
        command_text = paragraphs[-1] if len(paragraphs) > 1 else block
        if len(paragraphs) == 1:
            command_text = "\n".join(
                line for line in block.splitlines()
                if not re.match(r"^\s*(Analysis|Observation|Goal-progress)\s*:", line, re.I)
            ).strip() or block
        parsed = parse_model_output(command_text, settings)
        if not parsed["parse_success"]:
            continue
        key = _proposal_command_key(parsed)
        if key in seen:
            continue
        seen.add(key)
        proposals.append({"raw_output": block, "parsed": parsed})
    if proposals:
        return proposals
    parsed = parse_model_output(text, settings)
    if parsed["parse_success"]:
        return [{"raw_output": text.strip(), "parsed": parsed}]
    return []


class Policy:
    def __init__(self,args,settings):
        self.a,self.s=args,settings; self.key=os.environ.get(args.api_key_env,"")
        if args.use_api and args.provider=="openai" and not self.key:
            raise RuntimeError(f"${args.api_key_env} is empty")
        jev_api_key_env = getattr(args, "jev_api_key_env", "JEV_AGENT_KEY")
        self.jev_key=os.environ.get(jev_api_key_env,"")
        if args.use_api and getattr(args, "jev_enabled", False) and not self.jev_key:
            raise RuntimeError(f"${jev_api_key_env} is empty")
        if args.use_api and args.provider=="claude_sdk":
            probe=subprocess.run([args.claude_command,"auth","status"],capture_output=True,text=True)
            if probe.returncode!=0:
                raise RuntimeError("Claude Code is not logged in; run `claude` interactively first")

    def _call_jev(self, instruction, proposals):
        """Let text-only Jev rank valid vision-model proposals given the task."""
        state=json.dumps({
            "task":instruction,
            "proposals":[
                {"id":f"candidate_{i}",
                 "analysis_and_command":p["raw_output"],
                 "parsed_command":{
                     "task_type":p["parsed"].get("task_type"),
                     "coordinate":p["parsed"].get("coordinate"),
                     "coordinate_view":p["parsed"].get("coordinate_view"),
                     "macro_tokens":p["parsed"].get("macro_tokens"),
                 }}
                for i,p in enumerate(proposals)
            ],
        },ensure_ascii=False)
        criteria={
            f"candidate_{i}":
                ("Pick this option only when it respects the instruction's ordered steps, "
                 "identifies the earliest uncompleted step using explicit observation/history "
                 "evidence, and its command advances that step safely. Full proposal "
                 "(progress evidence + analysis + command): "+p["raw_output"][-1800:])
            for i,p in enumerate(proposals)
        }
        criteria["reject_all"] = (
            "Choose this when every proposal skips an uncompleted instruction step, claims a "
            "landmark was passed without explicit evidence, mistakes a side opening for a later "
            "destination, stops before all ordered steps are complete, or is otherwise not "
            "grounded enough to execute safely."
        )
        body={"model":self.a.jev_model,"state":state,"questions":{"next_command":{
            "type":"choice",
            "instructions":
                f"Navigation task: {instruction}\n"
                "Audit the task strictly in its stated order. A later landmark or room must not "
                "be pursued until the proposal gives concrete evidence that all earlier steps "
                "were completed. Choose the best grounded proposal, or reject_all if none meets "
                "that standard. Use STOP only when every ordered step is evidenced complete and "
                "the agent is just inside the final room.",
            "criteria":criteria,
        }}}
        request=urllib.request.Request(
            self.a.jev_base_url,data=json.dumps(body).encode(),
            headers={"Authorization":"Bearer "+self.jev_key,
                     "Content-Type":"application/json","Accept":"application/json"},
            method="POST")
        with urllib.request.urlopen(request,timeout=self.a.api_timeout) as response:
            payload=json.loads(response.read().decode())
        answer=payload.get("answers",{}).get("next_command",{})
        choice=answer.get("choice","")
        match=re.fullmatch(r"candidate_(\d+)",choice)
        index=int(match.group(1)) if match else None
        if index is not None and not (0 <= index < len(proposals)):
            index=None
        confidence=float(answer.get("confidence",0.0))
        return index,{
            "accepted":index is not None,"selected_index":index,
            "jev_choice":choice,"confidence":confidence,"response":payload,
            "task":instruction,
        }

    def _call_claude(self,prompt,history,current,local_map):
        with tempfile.TemporaryDirectory(prefix="gpt-vln-claude-") as temp:
            root=Path(temp); paths=[]
            current_frames = list(current) if isinstance(current, (list, tuple)) else [current]
            frames=list(history)+current_frames
            if self.a.local_map_input:
                frames.append(local_map)
            for index,frame in enumerate(frames):
                if index < len(history): name=f"history_{index:02d}.png"
                elif len(current_frames) == 3 and index < len(history) + 3:
                    name=f"current_{('LEFT','FRONT','RIGHT')[index-len(history)]}.png"
                elif index == len(history): name="current.png"
                else: name="local_map.png"
                path=root/name
                cv2.imwrite(str(path),cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)); paths.append(path)
            suffix = ("The penultimate image is current RGB; the final image is the local map.\n"
                      if self.a.local_map_input else "The final image is current RGB.\n")
            if len(current_frames) == 3:
                suffix = ("The final three images are separate current observations named "
                          "current_LEFT, current_FRONT, and current_RIGHT.\n")
            visual=("Use the Read tool to inspect every image in chronological order:\n"+
                    "\n".join(str(path) for path in paths)+"\n"+suffix)
            command=[self.a.claude_command,"-p","--output-format","json","--model",self.a.model,
                     "--tools","Read","--allowedTools","Read","--no-session-persistence",
                     "--permission-mode","dontAsk",visual+prompt]
            env=os.environ.copy()
            for var in ("ANTHROPIC_API_KEY","ANTHROPIC_AUTH_TOKEN","ANTHROPIC_BASE_URL"): env.pop(var,None)
            last_error = None
            for attempt in range(4):
                try:
                    completed=subprocess.run(command,capture_output=True,text=True,timeout=self.a.api_timeout,env=env)
                    if completed.returncode!=0:
                        raise RuntimeError(
                            f"exit={completed.returncode} stderr={completed.stderr[-1000:]!r} "
                            f"stdout={completed.stdout[-1000:]!r}")
                    payload=json.loads(completed.stdout)
                    if payload.get("subtype")!="success":
                        raise RuntimeError(
                            f"result={payload.get('subtype')}: {payload.get('result')}")
                    break
                except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError, RuntimeError) as exc:
                    last_error = exc
                    if attempt == 3:
                        raise RuntimeError(f"Claude Code failed after 4 attempts: {last_error}") from exc
                    time.sleep(2 ** attempt)
            return str(payload.get("result","")).strip(),{
                "subtype":payload.get("subtype"),"num_turns":payload.get("num_turns"),
                "duration_ms":payload.get("duration_ms"),"total_cost_usd":payload.get("total_cost_usd"),
                "usage":payload.get("usage"),
            }

    def call(self,instruction,history,current,local_map,waypoints,outcomes):
        progress_report = "<progress>" in instruction
        coords = [[w.pixel_xy[0], w.pixel_xy[1]] for w in waypoints]
        rgb_choices = f"RGB frontier waypoints: {coords}. " if coords else ""
        outputs = ["<waypoint>(u,v)</waypoint>",
                   "<action> up to 3 of FORWARD(25-75), TURN_LEFT(15-45), "
                   "TURN_RIGHT(15-45) </action>", "STOP only when done"]
        if coords: outputs.append("<frontiers_coord>(u,v)")
        outputs.extend(("<action> up to 3 of FORWARD(25-75), TURN_LEFT(15-45), "
                        "TURN_RIGHT(15-45) </action>", "STOP only when done"))
        visual_clause = ("The last two images are current RGB and up-to-8-step RGB-D local "
                         "geometry (agent centered/facing up). " if self.a.local_map_input
                         else "The last image is current RGB. ")
        if getattr(self.a, "panorama_input", False):
            visual_clause = ""
        # NB: phrase this as "describe under Analysis:" rather than "think in
        # <reasoning>...</reasoning>" -- the latter reliably trips Opus's
        # reasoning_extraction safeguard and the whole call comes back refused
        # (stop_reason=refusal, empty output). See outputs/rgb10_adaptive_reasoning
        # first-attempt logs.
        reasoning_mode = getattr(self.a, "reasoning_mode", "baseline")
        reasoning_clause = (
            "Before answering, briefly describe under 'Analysis:' what you see, how "
            "far along the instruction you are, and whether the visible options are "
            "ambiguous -- as short or long as the situation actually needs, a clear "
            "one-step case may need one sentence, an ambiguous junction may need more. "
            "Then answer. "
            if reasoning_mode == "adaptive" else
            "Before answering, output exactly these two lines:\n"
            "Observation: <what you currently see>\n"
            "Goal-progress: <how far along the instruction you are>\n"
            "Then answer on a new line. "
            if reasoning_mode == "structured" else "")
        recent_feedback = ""
        if outcomes:
            feedback = []
            for item in outcomes[-6:]:
                parsed = item.get("parsed")
                if isinstance(parsed, list):
                    command = " ".join(
                        token.get("type", "?") for token in parsed if isinstance(token, dict))
                elif isinstance(parsed, dict):
                    coordinate = parsed.get("coordinate")
                    command = (f"{parsed.get('task_type')}@{coordinate}" if coordinate is not None
                               else str(parsed.get("task_type") or parsed.get("action_sequence")))
                else:
                    command = str(parsed)
                feedback.append(
                    f"{command}: {item.get('status', 'unknown')}, "
                    f"moved {float(item.get('travelled_m', 0.0)):.2f}m")
            recent_feedback = "Recent executed decisions: " + "; ".join(feedback) + ". "
        response_rule = ("Reply with exactly one command followed by <progress>k</progress>, "
                         "and no explanation: " if progress_report else
                         "Reply with exactly one command and no explanation: ")
        waypoint_rule = ("Select a waypoint only on visible traversable ground or a visible "
                         "object surface; never select through an occlusion or infer a hidden "
                         "location. ")
        max_proposals = max(1, int(getattr(self.a, "jev_candidates", 5)))
        if self.a.jev_enabled:
            proposal_rule = (
                f"Propose between 1 and {max_proposals} alternative next steps for this "
                "same observation. Use fewer proposals when the next step is clear; use "
                "more only when the scene is genuinely ambiguous. Wrap each alternative in "
                "<proposal>...</proposal>. Treat the navigation instruction as an ORDERED "
                "checklist: do not pursue a later landmark, doorway, or destination until the "
                "images provide concrete evidence that every earlier step was completed. "
                "Inside every proposal include these fields:\n"
                "Ordered progress: <each instruction step as COMPLETE or NOT COMPLETE, with "
                "specific evidence from current or historical images; uncertainty is NOT "
                "COMPLETE>\n"
                "Earliest unmet step: <the first NOT COMPLETE step>\n"
                "Goal complete: YES or NO\n"
                "Analysis: <why the command advances the earliest unmet step without skipping>\n"
                "Then add a blank line and exactly one executable command. A side opening that "
                "looks like the final room is not enough if the stairs-top and mirror steps "
                "have not first been evidenced complete. STOP is legal only when Goal complete "
                "is YES and all ordered steps have evidence. "
                "Commands must differ across proposals: different pixel targets and/or "
                "different actions/STOP — do not rephrase the same command. "
            )
            response_rule = proposal_rule
            reasoning_clause = ""
        if getattr(self.a, "panorama_input", False):
            prompt=("Imagine you are a robot performing a navigation task. You are given "
                    "historical observations and the current observations. Your task is: "
                    f"'{instruction}'.\n\nThe current observation consists of three separate RGB "
                    "images: LEFT, FRONT, and RIGHT. Waypoint coordinates are given directly "
                    "in the pixel coordinates of the corresponding image. Prefix a waypoint "
                    "with its image name so it is executable: LEFT(u,v), FRONT(u,v), or "
                    f"RIGHT(u,v).\n\n{waypoint_rule}Analyze the observations and decide the next step. Depending "
                    "on the current situation, you may output only a waypoint; output only "
                    "discrete actions for local adjustment; or output both a waypoint and "
                    "actions, where the actions describe the adjustment needed after reaching "
                    "the waypoint.\n\nActions are: FORWARD(25-75), TURN_LEFT(15-45), "
                    "TURN_RIGHT(15-45). Choose the output form that best fits the current step. "
                    "After execution, use the new observation to decide the next step.\n\n"
                    f"{recent_feedback}{reasoning_clause}{response_rule}LEFT(u,v), FRONT(u,v), or RIGHT(u,v), "
                    "optionally followed by up to 3 actions; or up to 3 actions without a "
                    "waypoint. Output STOP only when done.")
        else:
            prompt=(f"Navigate: {instruction}\n{rgb_choices}{visual_clause}{recent_feedback}"
                    f"{waypoint_rule}{reasoning_clause}{response_rule}{', '.join(outputs)}.")
        if getattr(self.a, "expect_verify", False):
            from gpt_vln.e2_residual import PRED_EQA_PROMPT
            prompt += "\n" + PRED_EQA_PROMPT
        model_input={"provider":self.a.provider,"model":self.a.model,"prompt":prompt,"history_strategy":self.a.history_strategy,
                     "history_frames":len(history),
                     "image_inputs":("history_rgb+current_LEFT+current_FRONT+current_RIGHT"
                                     if getattr(self.a,"panorama_input",False) else
                                     "history_rgb+current_rgb") + ("+local_map" if self.a.local_map_input else ""),
                     "waypoints":[w.as_dict() for w in waypoints],
                     "jev_enabled":bool(self.a.jev_enabled),
                     "jev_max_proposals":max_proposals if self.a.jev_enabled else 1}
        start=time.perf_counter()
        def call_primary(active_prompt=prompt):
            if not self.a.use_api:
                return "STOP",{}
            if self.a.provider=="claude_sdk":
                return self._call_claude(active_prompt,history,current,local_map)
            def url(frame):
                ok,data=cv2.imencode(".jpg",cv2.cvtColor(frame,cv2.COLOR_RGB2BGR));
                if not ok: raise RuntimeError("image encoding failed")
                return "data:image/jpeg;base64,"+base64.b64encode(data).decode()
            current_frames = list(current) if isinstance(current, (list, tuple)) else [current]
            content=([{"type":"input_text","text":active_prompt}]
                     +[{"type":"input_image","image_url":url(f)} for f in history]
                     +[{"type":"input_image","image_url":url(f)} for f in current_frames]
                     +([{"type":"input_image","image_url":url(local_map)}]
                       if self.a.local_map_input else []))
            body={"model":self.a.model,"input":[{"role":"user","content":content}],"max_output_tokens":self.a.max_output_tokens}
            request=urllib.request.Request(self.a.base_url.rstrip("/")+"/responses",data=json.dumps(body).encode(),headers={"Authorization":"Bearer "+self.key,"Content-Type":"application/json","Accept":"application/json","User-Agent":"OpenAI/Python GPT-VLN","X-Stainless-Lang":"python"},method="POST")
            with urllib.request.urlopen(request,timeout=self.a.api_timeout) as response: payload=json.loads(response.read().decode())
            raw=payload.get("output_text","") or "".join(p.get("text","") for i in payload.get("output",[]) for p in i.get("content",[]) if p.get("type")=="output_text")
            return raw,{"response":payload}

        raw_bundle,provider_primary=call_primary()
        if self.a.jev_enabled and self.a.use_api:
            proposals=parse_proposal_blocks(raw_bundle,self.s,max_proposals)
            for proposal in proposals:
                proposal["provider_response"]=provider_primary
        else:
            parsed_single=parse_model_output(raw_bundle,self.s)
            proposals=[{"raw_output":raw_bundle,"parsed":parsed_single,
                        "provider_response":provider_primary}] if parsed_single["parse_success"] else []
        if not proposals:
            parsed=parse_model_output(raw_bundle,self.s)
            raw,provider_response=raw_bundle,provider_primary
            jev_response={"skipped":"no valid proposal"}
        elif self.a.jev_enabled and len(proposals)>1:
            selected,jev_response=self._call_jev(instruction,proposals)
            if selected is None:
                rejected_proposals=proposals
                repair_prompt=(
                    prompt+"\n\nJev rejected every proposal because none was sufficiently "
                    "grounded in the ordered task. Reassess the chronological images. Do not "
                    "assume that seeing a later landmark proves earlier steps were completed. "
                    "Generate a new set that advances the earliest evidence-backed unmet step."
                )
                repair_raw,repair_provider=call_primary(repair_prompt)
                repaired=parse_proposal_blocks(repair_raw,self.s,max_proposals)
                for candidate in repaired:
                    candidate["provider_response"]=repair_provider
                if repaired:
                    repair_selected,repair_jev=self._call_jev(instruction,repaired)
                    jev_response["replan"]={
                        "bundle_raw_output":repair_raw,
                        "proposals":[{"raw_output":p["raw_output"],
                                      "parsed":p["parsed"]} for p in repaired],
                        "jev":repair_jev,
                    }
                    if repair_selected is not None:
                        proposals=repaired
                        selected=repair_selected
                    else:
                        # Do not execute a rejected semantic waypoint. A short in-place turn
                        # obtains a different observation without claiming task progress.
                        fallback_raw="<action>TURN_LEFT(15)</action>"
                        fallback_parsed=parse_model_output(fallback_raw,self.s)
                        proposals=rejected_proposals
                        raw,parsed,provider_response=(
                            fallback_raw,fallback_parsed,repair_provider)
                        jev_response["safe_fallback"]="TURN_LEFT(15)"
                        selected=None
                else:
                    fallback_raw="<action>TURN_LEFT(15)</action>"
                    raw,parsed,provider_response=(
                        fallback_raw,parse_model_output(fallback_raw,self.s),repair_provider)
                    jev_response["safe_fallback"]="TURN_LEFT(15)"
                    selected=None
            if selected is not None:
                proposal=proposals[selected]
                raw,parsed,provider_response=(proposal["raw_output"],proposal["parsed"],
                                              proposal["provider_response"])
        else:
            proposal=proposals[0]
            raw,parsed,provider_response=(proposal["raw_output"],proposal["parsed"],
                                          proposal["provider_response"])
            jev_response=({"skipped":"only one valid proposal","task":instruction}
                          if self.a.jev_enabled else {})
        provider_response={"primary":provider_response,
                           "bundle_raw_output":raw_bundle if self.a.jev_enabled else None,
                           "proposals":[{"raw_output":p["raw_output"],
                                         "parsed":p["parsed"]} for p in proposals],
                           "jev":jev_response}
        return {"model_input":model_input,"raw_output":raw,"parsed":parsed,
                "provider_response":provider_response,"latency_s":time.perf_counter()-start}


def clearance(depth,s):
    h,w=depth.shape; strip=depth[int(.3*h):int(.8*h)]; focal=w/(2*np.tan(np.deg2rad(s.hfov)/2)); offsets=np.abs(np.arange(w)[None,:]-(w-1)/2)*strip/focal
    values=strip[np.isfinite(strip)&(strip>0)&(offsets<=s.robot_radius)]
    return float(np.percentile(values,5)) if values.size>=30 else float("inf")


def execute(env,actions,processor,obs,remaining,s):
    done=[]; frames=[]; trajectory=[]; step_progress=[]; start=np.asarray(env.sim.get_agent_state().position).copy(); status="chunk_complete"; blocked_actions=0
    for action in list(actions)[:remaining]:
        before=np.asarray(env.sim.get_agent_state().position).copy()
        obs=env.step(int(action)); after=np.asarray(env.sim.get_agent_state().position).copy()
        processor.record_visibility(); processor.record_depth(obs["depth"])
        progress=float(np.linalg.norm(after[[0,2]]-before[[0,2]]))
        done.append(int(action)); frames.append(np.asarray(obs["rgb"])[...,:3].copy()); trajectory.append(after.tolist()); step_progress.append(progress)
        if action==FORWARD and progress<0.02: blocked_actions+=1; status="blocked"
        if action==STOP or env.episode_over: status="stopped" if action==STOP else "episode_over"; break
    final=np.asarray(env.sim.get_agent_state().position)
    return obs,{"status":status,"requested_actions":list(actions)[:remaining],"actions":done,"blocked_actions":blocked_actions,"blocked_skips":0,"step_progress_m":step_progress,"trajectory":trajectory,"final_position":final.tolist(),"travelled_m":float(np.linalg.norm(final[[0,2]]-start[[0,2]]))},frames


def update_blocked_streak(previous, execution):
    """Count no-motion forward attempts; a moving/turning chunk breaks the streak."""
    blocked = int(execution.get("blocked_actions", execution.get("blocked_skips", 0)))
    if execution.get("actions") and blocked < len(execution.get("actions", [])):
        return 0
    return previous + blocked


def pixel_to_world(pixel, waypoints, observation, sim, processor, settings,
                   match_waypoints=True, panorama_views=None, return_pixel=False):
    """Back-project a pixel, falling back to the nearest navigable image pixel."""
    point = np.asarray(pixel, dtype=float)
    if match_waypoints and waypoints:
        nearest = min(waypoints, key=lambda item: np.linalg.norm(point - np.asarray(item.pixel_xy)))
        if np.linalg.norm(point - np.asarray(nearest.pixel_xy)) < 15.0:
            target = np.asarray(nearest.world_xyz, dtype=float)
            return (target, list(nearest.pixel_xy)) if return_pixel else target
    panorama_u, requested_v = int(round(point[0])), int(round(point[1]))
    panel_offset = 0
    rotation = None
    if panorama_views is not None:
        panel_index = panorama_u // settings.width
        if not (0 <= panel_index < len(panorama_views)):
            return (None, None) if return_pixel else None
        panel_offset = panel_index * settings.width
        requested_u = panorama_u - panel_offset
        view = panorama_views[panel_index]
        depth = processor.depth_m(view["depth"])
        rotation = view["rotation"]
    else:
        requested_u = panorama_u
        depth = processor.depth_m(observation["depth"])
    if not (0 <= requested_u < depth.shape[1] and 0 <= requested_v < depth.shape[0]):
        return (None, None) if return_pixel else None
    focal = settings.width / (2 * np.tan(np.deg2rad(settings.hfov) / 2))
    state = sim.get_agent_state()
    sensor = state.sensor_states.get("rgb") or state.sensor_states.get("rgb_sensor")
    candidates = [(requested_u, requested_v)]
    for radius in range(8, 81, 8):
        ring = []
        for du in range(-radius, radius + 1, 8):
            ring.extend(((requested_u + du, requested_v - radius),
                         (requested_u + du, requested_v + radius)))
        for dv in range(-radius + 8, radius, 8):
            ring.extend(((requested_u - radius, requested_v + dv),
                         (requested_u + radius, requested_v + dv)))
        ring.sort(key=lambda p: ((p[0]-requested_u)**2 + (p[1]-requested_v)**2,
                                 p[1] < requested_v))
        candidates.extend(ring)
    for u, v in candidates:
        if not (0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]):
            continue
        z = float(depth[v, u])
        if not np.isfinite(z) or z <= 0:
            continue
        camera_point = np.array([(u-(settings.width-1)/2)*z/focal,
                                 -((v-(settings.height-1)/2)*z/focal), -z])
        world = np.asarray(sensor.position) + quaternion_rotate_vector(
            rotation if rotation is not None else sensor.rotation, camera_point)
        snapped = np.asarray(sim.pathfinder.snap_point(world))
        if (np.isnan(snapped).any() or not sim.pathfinder.is_navigable(snapped)
                or np.linalg.norm(snapped[[0,2]]-world[[0,2]]) > settings.target_max_snap_drift_m):
            continue
        resolved = [u + panel_offset, v]
        return (snapped, resolved) if return_pixel else snapped
    return (None, None) if return_pixel else None


def execute_navigation_skill(env, target, observation, remaining, processor, max_skill_steps=50):
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
    step_limit = min(max_skill_steps, remaining)
    status = "waypoint_timeout"
    for _ in range(step_limit):
        action = follower.get_next_action(target)
        if action is None or int(action) == STOP:
            status = "waypoint_reached"
            break
        observation = env.step(int(action)); actions.append(int(action))
        processor.record_visibility(); processor.record_depth(observation["depth"])
        frames.append(np.asarray(observation["rgb"])[..., :3].copy())
        trajectory.append(np.asarray(env.sim.get_agent_state().position).tolist())
        if env.episode_over:
            status = "episode_over"
            break
    else:
        # The last permitted primitive may itself have reached the target.
        final_action = follower.get_next_action(target)
        if final_action is None or int(final_action) == STOP:
            status = "waypoint_reached"
        elif remaining <= max_skill_steps:
            status = "action_budget_exhausted"
    final = np.asarray(env.sim.get_agent_state().position).copy()
    return observation, {"status": status, "skill": "ShortestPathFollower",
        "requested_target_world": requested_target.tolist(),
        "target_world": np.asarray(target).tolist(),
        "target_height_correction_m": float(target[1] - requested_target[1]),
        "actions": actions,
        "trajectory": trajectory,
        "final_position": final.tolist(),
        "travelled_m": float(np.linalg.norm(final[[0,2]]-start[[0,2]]))}, frames


def execute_waypoint_command(env, target, trailing_actions, observation, remaining,
                             processor, settings):
    """Navigate to a waypoint, then execute optional actions from the arrival pose."""
    remaining = min(remaining, MAX_ATOMIC_ACTIONS_PER_TURN)
    observation, navigation, frames = execute_navigation_skill(
        env, target, observation, remaining, processor,
        max_skill_steps=MAX_ATOMIC_ACTIONS_PER_TURN)
    navigation["waypoint_status"] = navigation["status"]
    if (not trailing_actions or env.episode_over
            or navigation["status"] != "waypoint_reached"):
        return observation, navigation, frames
    left = max(0, remaining - len(navigation["actions"]))
    if not left:
        navigation["trailing_status"] = "action_budget_exhausted"
        return observation, navigation, frames
    observation, trailing, trailing_frames = execute(
        env, trailing_actions, processor, observation, left, settings)
    navigation["status"] = trailing["status"]
    navigation["trailing_status"] = trailing["status"]
    navigation["trailing_actions"] = list(trailing_actions)
    navigation["actions"].extend(trailing["actions"])
    navigation["trajectory"].extend(trailing["trajectory"])
    navigation["blocked_actions"] = trailing.get("blocked_actions", 0)
    navigation["blocked_skips"] = trailing.get("blocked_skips", 0)
    navigation["step_progress_m"] = trailing.get("step_progress_m", [])
    navigation["final_position"] = trailing["final_position"]
    navigation["travelled_m"] += trailing["travelled_m"]
    frames.extend(trailing_frames)
    return observation, navigation, frames


def create_env(args,s,gpu_id=None):
    cfg=get_config(str(ROOT/"config/vln_r2r.yaml"))
    with habitat.config.read_write(cfg):
        cfg.habitat.dataset.split=args.split; cfg.habitat.dataset.data_path=str(WORKSPACE/"StreamVLN/data/datasets/r2r/{split}/{split}.json.gz"); cfg.habitat.dataset.scenes_dir=str(WORKSPACE/"StreamVLN/data/scene_datasets")
        cfg.habitat.environment.max_episode_steps=args.max_actions; sim=cfg.habitat.simulator; sim.forward_step_size=s.forward_step; sim.turn_angle=int(s.turn_step)
        if gpu_id is not None: sim.habitat_sim_v0.gpu_device_id=gpu_id
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
        # get_sensor_observations() returns raw simulator output, skipping the habitat-lab
        # sensor suite -- and with it normalize_depth, so depth_m() would scale an already
        # metric frame by depth_max and put every point outside the local window. Going
        # through get_observations_at() keeps turn 0 consistent with every later step.
        state = env.sim.get_agent_state()
        obs = env.sim.get_observations_at(state.position, state.rotation,
                                          keep_agent_at_new_pose=True)
    processor=MapProcessor(env.sim,s,normalized); history=History(args.history_strategy,args.history_frames)
    raw_dir=out/"raw_action_frames"; raw_dir.mkdir(parents=True,exist_ok=True)
    trace=[]; outcomes=[]; action_frames=[]; positions=[np.asarray(env.sim.get_agent_state().position).copy()]; inference_points=[]; used=0; blocked_streak=0
    turn_limit = args.max_turns if args.max_turns > 0 else 10
    for turn in range(turn_limit):
        local,global_map=processor.update(obs["depth"])
        waypoints=processor.waypoints(local,obs["depth"]) if args.frontier_input else []
        rgb=np.asarray(obs["rgb"])[...,:3]
        panorama_views = None
        if args.panorama_input:
            panorama_views = build_rgb_panorama(env.sim, obs)
            current = [view["rgb"] for view in panorama_views]
        else:
            current=overlay_navigation(rgb,waypoints)
        selected=history.select(); local_color=color_local_map(local,waypoints,processor.local_mpp)
        if args.visualize:
            if args.panorama_input:
                if selected:
                    save_inference_input(out/f"turn_{turn:02d}_history_input.png",
                                         selected[:-1],selected[-1])
                for view in panorama_views:
                    cv2.imwrite(str(out/f"turn_{turn:02d}_current_{view['label']}.png"),
                                cv2.cvtColor(view["rgb"],cv2.COLOR_RGB2BGR))
            else:
                save_inference_input(out/f"turn_{turn:02d}_input.png",selected,current,
                                     local_color if args.local_map_input else None)
                cv2.imwrite(str(out/f"turn_{turn:02d}_rgb.png"),
                            cv2.cvtColor(current,cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out/f"turn_{turn:02d}_local_map.png"),cv2.cvtColor(local_color,cv2.COLOR_RGB2BGR)); cv2.imwrite(str(out/f"turn_{turn:02d}_global_map.png"),cv2.cvtColor(global_map,cv2.COLOR_RGB2BGR)); save_depth(out/f"turn_{turn:02d}_depth.png",processor.depth_m(obs["depth"]),s.depth_max)
        decision=policy.call(episode.instruction.instruction_text,selected,current,local_color,waypoints,outcomes)
        decision["model_input"]["frontier_stats"] = processor.frontier_stats
        decision["model_input"]["local_depth_frames"] = len(processor.recent_depth)
        parsed=decision["parsed"]; frames=[]
        if args.visualize and not args.panorama_input:
            selected_rgb = overlay_selected_waypoint(current, parsed.get("coordinate"))
            cv2.imwrite(str(out/f"turn_{turn:02d}_waypoint.png"),
                        cv2.cvtColor(selected_rgb,cv2.COLOR_RGB2BGR))
        if args.dry_run: execution={"status":"dry_run","actions":[]}
        elif parsed["task_type"]=="stop":
            obs=env.step(STOP); stop_frame=np.asarray(obs["rgb"])[...,:3].copy(); frames=[stop_frame]
            stop_pos=np.asarray(env.sim.get_agent_state().position).tolist()
            execution={"status":"stopped","actions":[STOP],"trajectory":[stop_pos],"final_position":stop_pos,"travelled_m":0.0}
        elif parsed.get("coordinate") is not None:
            coordinate = parsed["coordinate"]
            if panorama_views is not None:
                view_names = [view["label"] for view in panorama_views]
                if parsed.get("coordinate_view") not in view_names:
                    execution={"status":"coordinate_view_missing","actions":[]}
                    target=None; resolved_pixel=None
                else:
                    view_index=view_names.index(parsed["coordinate_view"])
                    coordinate=[view_index*s.width+coordinate[0],coordinate[1]]
            else:
                view_index=None
            if panorama_views is None or parsed.get("coordinate_view") in view_names:
                target,resolved_pixel=pixel_to_world(
                    coordinate,waypoints,obs,env.sim,processor,s,
                    # AgentVLN snaps any prediction near a visible candidate, not
                    # only outputs using the <frontiers_coord> tag.
                    match_waypoints=bool(waypoints),
                    panorama_views=panorama_views,return_pixel=True)
            if target is None: execution={"status":"coordinate_unresolved","actions":[]}
            else:
                obs,execution,frames=execute_waypoint_command(
                    env,target,parsed.get("action_sequence"),obs,
                    args.max_actions-used,processor,s)
                execution["requested_coordinate"] = parsed["coordinate"]
                execution["coordinate_view"] = parsed.get("coordinate_view")
                execution["resolved_coordinate"] = (
                    [resolved_pixel[0]-view_index*s.width,resolved_pixel[1]]
                    if panorama_views is not None else resolved_pixel)
                if args.visualize:
                    display_requested=parsed["coordinate"]
                    display_resolved=resolved_pixel
                    display_image=current
                    if panorama_views is not None:
                        display_image=current[view_index]
                        display_resolved=[resolved_pixel[0]-view_index*s.width,
                                          resolved_pixel[1]]
                    projected=overlay_selected_waypoint(
                        display_image,display_requested,display_resolved)
                    suffix=(f"_{parsed['coordinate_view']}" if panorama_views is not None else "")
                    cv2.imwrite(str(out/f"turn_{turn:02d}_waypoint{suffix}.png"),
                                cv2.cvtColor(projected,cv2.COLOR_RGB2BGR))
        elif parsed.get("action_sequence"):
            obs,execution,frames=execute(
                env,parsed["action_sequence"],processor,obs,
                min(args.max_actions-used,MAX_ATOMIC_ACTIONS_PER_TURN),s)
        else:
            # A malformed model answer contains no grounded translation target.
            # Rotate to acquire a different view; blindly moving forward can
            # collide, skip an opening, and trigger the blocked safety stop.
            obs,execution,frames=execute(env,[LEFT],processor,obs,args.max_actions-used,s)
            execution["status"]=(
                "parse_failure_recovery"
                if execution["status"]=="chunk_complete" else execution["status"])
        used+=len(execution["actions"])
        blocked_streak=update_blocked_streak(blocked_streak,execution)
        if blocked_streak>=MAX_CONSECUTIVE_BLOCKED_STEPS and execution["status"] not in ("stopped","episode_over"):
            if not env.episode_over: obs=env.step(STOP)
            execution["status"]="safety_stop_blocked"; execution["blocked_streak"]=blocked_streak
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
        item={"turn":turn,**decision,"execution":execution}; trace.append(item)
        outcomes.append({"parsed": decision["parsed"].get("macro_tokens") or {
            "task_type": decision["parsed"].get("task_type"),
            "coordinate": decision["parsed"].get("coordinate"),
            "action_sequence": decision["parsed"].get("action_sequence")},
            "status": execution["status"], "travelled_m": execution.get("travelled_m", 0)})
        (out/f"turn_{turn:02d}_inference.json").write_text(json.dumps(jsonable(item),indent=2)); (out/"trace.json").write_text(json.dumps(jsonable(trace),indent=2))
        if (args.dry_run or env.episode_over or used >= args.max_actions
                or execution["status"] in ("stopped","safety_stop_blocked")): break
    if not args.dry_run and not env.episode_over: env.step(STOP)
    if args.visualize: save_route_topmap(out/"top_map_gt_pred.png",env.sim,episode,positions,inference_points)
    result={"episode_id":str(episode.episode_id),"scene_id":episode.scene_id,"instruction":episode.instruction.instruction_text,"start_alignment":alignment,"turns":len(trace),"total_actions":used,"history":{"strategy":args.history_strategy,"stored":len(history.frames)},"metrics":jsonable(env.get_metrics())}
    (out/"result.json").write_text(json.dumps(result,indent=2)); return result


def parser():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--episode-id",default="412",help='"412" single id, "4,5,6" comma list, "all" every episode, or "random" (use with --num-episodes/--seed)'); p.add_argument("--num-episodes",type=int,default=0,help="with --episode-id all: cap to first N; with random: sample size"); p.add_argument("--seed",type=int,default=0,help="RNG seed for --episode-id random"); p.add_argument("--split",default="val_unseen"); p.add_argument("--output",type=Path,default=Path("outputs/eval")); p.add_argument("--provider",choices=("openai","claude_sdk"),default="openai"); p.add_argument("--model",default="gpt-6"); p.add_argument("--claude-command",default="claude"); p.add_argument("--base-url",default="https://api.openai.com/v1"); p.add_argument("--api-key-env",default="OPENAI_API_KEY"); p.add_argument("--api-timeout",type=float,default=180); p.add_argument("--max-output-tokens",type=int,default=2048); p.add_argument("--use-api",action=argparse.BooleanOptionalAction,default=True); p.add_argument("--jev-enabled",action=argparse.BooleanOptionalAction,default=False,help="sample multiple vision proposals and use text-only Jev to select one"); p.add_argument("--jev-candidates",type=int,default=5,help="max diverse Claude proposals per turn when --jev-enabled (model may return fewer)"); p.add_argument("--jev-base-url",default="https://api.typesafe.ai/v1/systemone"); p.add_argument("--jev-api-key-env",default="JEV_AGENT_KEY"); p.add_argument("--jev-model",default="jev-latest"); p.add_argument("--jev-min-confidence",type=float,default=0.85); p.add_argument("--visualize",action=argparse.BooleanOptionalAction,default=True); p.add_argument("--align-start-heading",action=argparse.BooleanOptionalAction,default=False); p.add_argument("--hfov",type=float,default=120.0); p.add_argument("--history-strategy",choices=("uniform","recent","hybrid"),default="uniform"); p.add_argument("--history-frames",type=int,default=8); p.add_argument("--panorama-input",action=argparse.BooleanOptionalAction,default=False,help="use a LEFT|FRONT|RIGHT|BACK current RGB panorama"); p.add_argument("--max-turns",type=int,default=0,help="inference turn cap; 0 means no turn cap (bounded by --max-actions)"); p.add_argument("--reasoning-mode",choices=("baseline","adaptive","structured"),default="baseline",help="adaptive: prompt the model to self-pace an explicit <reasoning> block before answering; structured: require fixed Observation:/Goal-progress: fields before the command; baseline: unchanged prompt"); p.add_argument("--max-actions",type=int,default=400); p.add_argument("--local-map-input",action=argparse.BooleanOptionalAction,default=False,help="append RGB-D local geometry to model input (off for the RGB baseline)"); p.add_argument("--frontier-input",action=argparse.BooleanOptionalAction,default=False,help="annotate RGB with map-derived frontier candidates (off for the RGB baseline)"); p.add_argument("--dry-run",action="store_true"); p.add_argument("--skip-existing",action=argparse.BooleanOptionalAction,default=True,help="skip episodes whose output/episode_<id>/result.json already exists (resume)"); p.add_argument("--force",action="store_true",help="alias for --no-skip-existing"); p.add_argument("--gpu-ids",default="0,1",help="comma-separated CUDA device ids; episodes are chunked across worker processes, one per chunk, round-robin over these devices"); p.add_argument("--episodes-per-worker",type=int,default=8,help="episodes handled by each worker process/GPU chunk"); return p


def select_episodes(all_episodes,args):
    eid=args.episode_id.strip().lower()
    if eid=="all":
        episodes=list(all_episodes)
        if args.num_episodes>0: episodes=episodes[:args.num_episodes]
    elif eid=="random":
        if args.num_episodes<=0: raise ValueError("--episode-id random requires --num-episodes N")
        import random; rng=random.Random(args.seed); episodes=rng.sample(list(all_episodes),min(args.num_episodes,len(all_episodes)))
    else:
        wanted=[t.strip() for t in args.episode_id.split(",") if t.strip()]
        by_id={str(e.episode_id):e for e in all_episodes}
        missing=[t for t in wanted if t not in by_id]
        if missing: raise ValueError(f"no episode matched {missing}")
        episodes=[by_id[t] for t in wanted]
    return sorted(episodes,key=lambda e: e.scene_id)  # group by scene to cut sim scene-reload cost


def run_worker(gpu_id,episodes_chunk,args,s,lock):
    policy=Policy(args,s); env,normalized=create_env(args,s,gpu_id)
    try:
        env.episodes=episodes_chunk
        for i in range(len(episodes_chunk)):
            result=run_episode(env,policy,args,s,normalized)
            with lock, (args.output/"results.jsonl").open("a") as f: f.write(json.dumps(result)+"\n")
            print(f"[gpu{gpu_id} {i+1}/{len(episodes_chunk)}] episode={result['episode_id']} metrics={result['metrics']}")
    finally: env.close()


def main():
    args=parser().parse_args(); args.output=args.output.resolve(); args.output.mkdir(parents=True,exist_ok=True)
    if args.force: args.skip_existing=False
    s=Settings(hfov=args.hfov); gpu_ids=[int(g) for g in args.gpu_ids.split(",") if g.strip()!=""]
    Policy(args,s)  # fail fast on missing/invalid API credentials before spawning workers
    if not gpu_ids: raise ValueError("--gpu-ids must contain at least one device id")
    env,normalized=create_env(args,s,gpu_ids[0])
    try:
        episodes=select_episodes(list(env.episodes),args)
        if args.skip_existing:
            episodes=[e for e in episodes if not (args.output/f"episode_{e.episode_id}"/"result.json").exists()]
    finally: env.close()
    if not episodes: raise ValueError(f"no episode matched {args.episode_id} (or all already done, see --force)")
    chunks=[episodes[i:i+args.episodes_per_worker] for i in range(0,len(episodes),args.episodes_per_worker)]
    # Habitat-Sim/EGL is not fork-safe after the discovery environment has created a
    # CUDA context. Spawn clean workers instead of inheriting that native state.
    mp_ctx=mp.get_context("spawn"); lock=mp_ctx.Lock(); processes=[]
    for i,chunk in enumerate(chunks):
        gpu_id=gpu_ids[i%len(gpu_ids)]
        p=mp_ctx.Process(target=run_worker,args=(gpu_id,chunk,args,s,lock),daemon=True); p.start(); processes.append(p)
    for p in processes: p.join()
    failed=[p.pid for p in processes if p.exitcode != 0]
    if failed:
        raise RuntimeError(f"evaluation workers failed: {failed}")
    prior=[]
    if (args.output/"results.jsonl").exists():
        prior=[json.loads(line) for line in (args.output/"results.jsonl").read_text().splitlines() if line.strip()]
    by_id={r["episode_id"]:r for r in prior}; merged=list(by_id.values())
    keys=sorted({k for r in merged for k,v in r["metrics"].items() if isinstance(v,(int,float))}); summary={"episodes":len(merged),"averages":{k:float(np.mean([r["metrics"][k] for r in merged])) for k in keys}}; (args.output/"summary.json").write_text(json.dumps(summary,indent=2))

if __name__=="__main__": main()
