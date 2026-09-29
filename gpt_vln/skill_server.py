"""Navigation skills the model invokes itself, served to a separate process.

Runs inside the streamvln (py3.9) process that owns the Habitat Env, because
that is the only place `env.step` may be called. gpt_vln/skill_bridge.py runs
under the claude-agent-sdk env (py3.10+, required by claude_agent_sdk) and
reaches these handlers over an AF_UNIX socket: one connect per call, one
newline-terminated JSON object each way.

Everything here is deterministic bookkeeping over eval.py's existing execution
primitives -- no skill calls an LLM. Progress and evidence are authored by the
model in its own session (where it is already reasoning, so they cost no extra
API call) and this module only stores, validates and replays them.
"""
from __future__ import annotations

import json
import os
import socket
import time

import cv2
import numpy as np

from eval import (
    MAX_ATOMIC_ACTIONS_PER_TURN, LEFT, RIGHT, STOP, UNKNOWN, FREE, OCCUPIED,
    build_rgb_panorama, execute, execute_navigation_skill,
    execute_waypoint_command, jsonable, parse_model_output, pixel_to_world,
)
from habitat.utils.geometry_utils import quaternion_rotate_vector
from gpt_vln.visualization import color_local_map

FAILURE_STATUSES = {"coordinate_unresolved", "target_not_navigable", "blocked",
                    "safety_stop_blocked", "coordinate_view_missing",
                    "waypoint_timeout", "action_budget_exhausted"}
MAX_AVOID_ENTRIES = 8
MAX_RECALL_DECISIONS = 6
VIEW_LABELS = ("LEFT", "FRONT", "RIGHT")
MAX_RPC_REQUEST_BYTES = 1024 * 1024


class SkillError(Exception):
    """A skill was called with arguments the server refuses; reported to the model."""


class EpisodeState:
    """Budgets, memory, checkpoints and the avoid-list for one episode."""

    def __init__(self, instruction, max_moves, max_actions):
        self.instruction = instruction
        self.max_moves, self.max_actions = max_moves, max_actions
        self.moves_used = self.actions_used = 0
        self.subtasks = []
        self.refinement_reason = None
        self.stage = 0  # monotonic high-water mark, 0-indexed
        self.progress_notes = []
        self.avoid = []
        self.checkpoints = []
        self.decisions = []
        self.transcript = []
        self.stopped = False
        self.stop_evidence = None
        self.timed_out = False

    @property
    def moves_remaining(self):
        return max(0, self.max_moves - self.moves_used)

    @property
    def actions_remaining(self):
        return max(0, self.max_actions - self.actions_used)

    def spend_move(self, skill):
        if self.moves_remaining <= 0:
            raise SkillError(
                f"move budget exhausted ({self.moves_used}/{self.max_moves} movement "
                f"skills used); call stop(evidence) to end the episode")
        if self.actions_remaining <= 0:
            raise SkillError(
                f"atomic action budget exhausted ({self.actions_used}/{self.max_actions}); "
                "call stop(evidence) to end the episode")
        self.moves_used += 1

    def note_stage(self, stage):
        """1-based from the model, stored 0-indexed and never allowed to regress."""
        if not self.subtasks:
            return self.stage
        k = min(max(int(stage), 1), len(self.subtasks))
        self.stage = max(self.stage, k - 1)
        return self.stage

    def record_failure(self, description, reason):
        self.avoid.append({"target": description, "reason": reason,
                           "move": self.moves_used})
        del self.avoid[:-MAX_AVOID_ENTRIES]

    def record_checkpoint(self, position, rotation, note):
        entry = {"id": len(self.checkpoints), "move": self.moves_used,
                 "position": list(position), "rotation": list(rotation), "note": note}
        self.checkpoints.append(entry)
        return entry

    def record_decision(self, skill, detail, execution):
        self.decisions.append({
            "skill": skill, "detail": detail,
            "status": execution.get("status", "unknown"),
            "travelled_m": round(float(execution.get("travelled_m", 0.0)), 2)})

    def recall(self):
        """Everything the model may have forgotten, computed from records only."""
        report = {
            "instruction": self.instruction,
            "moves_used": self.moves_used, "moves_remaining": self.moves_remaining,
            "actions_used": self.actions_used, "actions_remaining": self.actions_remaining,
            "recent_decisions": self.decisions[-MAX_RECALL_DECISIONS:],
            "avoid": list(self.avoid),
            "checkpoints": [{"id": c["id"], "move": c["move"], "note": c["note"]}
                            for c in self.checkpoints],
        }
        if self.subtasks:
            report["subtasks"] = [
                {"index": i + 1, "text": text,
                 "state": "done" if i < self.stage else
                          "current" if i == self.stage else "pending"}
                for i, text in enumerate(self.subtasks)]
        if self.refinement_reason:
            report["instruction_refinement_reason"] = self.refinement_reason
        if self.progress_notes:
            report["progress_notes"] = self.progress_notes[-MAX_RECALL_DECISIONS:]
        return report


class SkillServer:
    """Dispatches skill calls onto the live Habitat env. Main thread only."""

    def __init__(self, env, processor, history, settings, state, out_dir,
                 observation, frontier_input=False):
        self.env, self.processor, self.history = env, processor, history
        self.s, self.state, self.out = settings, state, out_dir
        self.obs = observation
        self.frontier_input = frontier_input
        self.panorama_views = None
        self.waypoints = []
        self.frames = []  # (action, rgb) appended for every executed primitive
        self.image_dir = out_dir / "skill_views"
        self.image_dir.mkdir(parents=True, exist_ok=True)

    # --- helpers ---------------------------------------------------------------
    def _write_view(self, rgb, name):
        path = self.image_dir / name
        cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return str(path)

    def _status_block(self):
        agent = self.env.sim.get_agent_state()
        return {"position": [round(float(v), 3) for v in agent.position],
                "moves_used": self.state.moves_used,
                "moves_remaining": self.state.moves_remaining,
                "actions_remaining": self.state.actions_remaining}

    def _absorb(self, execution, frames):
        """Fold one execution's cost, frames and failure status into episode state."""
        self.state.actions_used += len(execution.get("actions", []))
        for action, frame in zip(execution.get("actions", []), frames):
            self.history.add(frame)
            self.frames.append((int(action), frame))
        self.processor.update(self.obs["depth"])

    def _post_move_report(self, execution):
        report = {"status": execution.get("status"),
                  "travelled_m": round(float(execution.get("travelled_m", 0.0)), 2),
                  **self._status_block()}
        if execution.get("status") in FAILURE_STATUSES:
            report["warning"] = ("this movement did not go as asked; it is now in the "
                                 "avoid-list, consider observe_panorama() or "
                                 "go_back(checkpoint_id)")
        report["views"] = self._panorama_paths(f"after_move_{self.state.moves_used:02d}")
        return report

    def _panorama_paths(self, prefix):
        self.panorama_views = build_rgb_panorama(self.env.sim, self.obs)
        return {view["label"]: self._write_view(view["rgb"], f"{prefix}_{view['label']}.png")
                for view in self.panorama_views}

    # --- skills ----------------------------------------------------------------
    def observe(self, _params):
        rgb = np.asarray(self.obs["rgb"])[..., :3]
        return {"front_view": self._write_view(
                    rgb, f"observe_{len(self.state.transcript):02d}_FRONT.png"),
                "note": "use the Read tool on this path to look at it",
                **self._status_block()}

    def observe_panorama(self, _params):
        views = self._panorama_paths(f"panorama_{len(self.state.transcript):02d}")
        return {"views": views,
                "note": ("use the Read tool on each path; pixel coordinates for "
                         "navigate_to are relative to the image you name"),
                "image_size": {"width": self.s.width, "height": self.s.height},
                **self._status_block()}

    def observe_local_map(self, _params):
        """On-demand local occupancy from recent RGB-D. Agent-centered, facing up."""
        local, _ = self.processor.update(self.obs["depth"])
        try:
            raw = self.processor.waypoints(local, self.obs["depth"]) or []
            waypoints = [item for item in raw if hasattr(item, "as_dict")]
        except Exception:
            waypoints = []
        self.waypoints = waypoints
        image = color_local_map(local, self.waypoints, self.processor.local_mpp)
        counts = {
            "free": int(np.sum(local == FREE)),
            "occupied": int(np.sum(local == OCCUPIED)),
            "unknown": int(np.sum(local == UNKNOWN)),
        }
        return {
            "local_map": self._write_view(
                image, f"local_map_{len(self.state.transcript):02d}.png"),
            "note": ("agent-centered, facing up. light=free/traversable floor, "
                     "dark=occupied, gray=unknown, grid=1m. Use the Read tool "
                     "on this path. Green dots are visible walkable candidates."),
            "meters_per_pixel": float(self.processor.local_mpp),
            "cell_counts": counts,
            "waypoints": [item.as_dict() for item in self.waypoints],
            **self._status_block(),
        }

    def recall(self, _params):
        return self.state.recall()

    def note_progress(self, params):
        stage = params.get("stage")
        evidence = str(params.get("evidence", "")).strip()
        if not evidence:
            raise SkillError("evidence is required: state what you actually see that "
                             "justifies this progress claim")
        if stage is not None:
            self.state.note_stage(stage)
        entry = {"move": self.state.moves_used, "stage": self.state.stage + 1,
                 "evidence": evidence}
        self.state.progress_notes.append(entry)
        return {"recorded": entry, "note": "recall() will replay this later"}

    def refine_instruction(self, params):
        subtasks = params.get("subtasks") or []
        if (not isinstance(subtasks, list) or not subtasks
                or not all(str(x).strip() for x in subtasks)):
            raise SkillError("subtasks must be a non-empty list of non-empty strings")
        self.state.subtasks = [str(x).strip() for x in subtasks]
        self.state.refinement_reason = str(params.get("reason", "")).strip() or None
        self.state.stage = min(self.state.stage, len(self.state.subtasks) - 1)
        return {"subtasks": self.state.subtasks, "current_stage": self.state.stage + 1}

    def navigate_to(self, params):
        view = str(params.get("view", "")).upper()
        if view not in VIEW_LABELS:
            raise SkillError(f"view must be one of {VIEW_LABELS}, got {view!r}")
        if self.panorama_views is None:
            raise SkillError("call observe_panorama() first so pixel coordinates have "
                             "a known frame of reference")
        try:
            u, v = float(params["u"]), float(params["v"])
        except (KeyError, TypeError, ValueError):
            raise SkillError("u and v must be numbers (pixel coordinates in `view`)")
        stitched = [VIEW_LABELS.index(view) * self.s.width + u, v]
        target, resolved = pixel_to_world(
            stitched, self.waypoints, self.obs, self.env.sim, self.processor, self.s,
            match_waypoints=False, panorama_views=self.panorama_views, return_pixel=True)
        if target is None:
            execution = {"status": "coordinate_unresolved", "actions": []}
            self.state.record_failure(f"{view}({u:.0f},{v:.0f})", "coordinate_unresolved")
            self.state.record_decision("navigate_to", f"{view}({u:.0f},{v:.0f})", execution)
            return {"status": "coordinate_unresolved",
                    "reason": ("that pixel does not back-project onto reachable floor -- "
                               "pick a point on visible traversable ground"),
                    **self._status_block()}

        # Validation failures are not movements and must not consume the scarce move
        # budget. Commit the budget/checkpoint only once there is an executable target.
        self.state.spend_move("navigate_to")
        agent = self.env.sim.get_agent_state()
        self.state.record_checkpoint(
            list(agent.position), list(quat_to_list(agent.rotation)),
            f"before navigate_to {view}({u:.0f},{v:.0f})")
        trailing = macro_to_actions(params.get("then_actions", ""), self.s)
        self.obs, execution, frames = execute_waypoint_command(
            self.env, target, trailing, self.obs, self.state.actions_remaining,
            self.processor, self.s)
        resolved_local = [resolved[0] - VIEW_LABELS.index(view) * self.s.width, resolved[1]]
        execution["requested_pixel"] = [round(u, 1), round(v, 1)]
        execution["resolved_pixel"] = resolved_local
        execution["pixel_correction"] = round(
            float(np.linalg.norm(np.asarray(resolved_local) - np.asarray([u, v]))), 1)
        self._absorb(execution, frames)
        if execution.get("status") in FAILURE_STATUSES:
            self.state.record_failure(f"{view}({u:.0f},{v:.0f})", execution["status"])
        self.state.record_decision("navigate_to", f"{view}({u:.0f},{v:.0f})", execution)
        return self._post_move_report(execution)

    def move(self, params):
        actions = macro_to_actions(params.get("actions", ""), self.s)
        if not actions:
            raise SkillError(
                "actions must be up to 3 of FORWARD(25-75), TURN_LEFT(15-45), "
                "TURN_RIGHT(15-45) with explicit magnitudes, e.g. 'TURN_LEFT(30) FORWARD(50)'")
        self.state.spend_move("move")
        agent = self.env.sim.get_agent_state()
        self.state.record_checkpoint(
            list(agent.position), list(quat_to_list(agent.rotation)),
            f"before move {params.get('actions', '')}")
        self.obs, execution, frames = execute(
            self.env, actions, self.processor, self.obs,
            min(self.state.actions_remaining, MAX_ATOMIC_ACTIONS_PER_TURN), self.s)
        self._absorb(execution, frames)
        if execution.get("status") in FAILURE_STATUSES:
            self.state.record_failure(str(params.get("actions")), execution["status"])
        self.state.record_decision("move", str(params.get("actions")), execution)
        return self._post_move_report(execution)

    def go_back(self, params):
        if not self.state.checkpoints:
            raise SkillError("no checkpoints recorded yet; they are created by "
                             "navigate_to and move")
        try:
            checkpoint_id = int(params["checkpoint_id"])
        except (KeyError, TypeError, ValueError):
            raise SkillError("checkpoint_id must be an integer from recall()")
        match = next((c for c in self.state.checkpoints if c["id"] == checkpoint_id), None)
        if match is None:
            raise SkillError(f"no checkpoint with id {checkpoint_id}; see recall()")
        self.state.spend_move("go_back")
        budget = min(self.state.actions_remaining, MAX_ATOMIC_ACTIONS_PER_TURN)
        self.obs, execution, frames = execute_navigation_skill(
            self.env, np.asarray(match["position"], dtype=float), self.obs,
            budget, self.processor, max_skill_steps=budget)
        # ShortestPathFollower only returns to the saved XY. A wrong TURN leaves
        # the agent on the same cell, so heading has to be restored separately.
        used = len(execution.get("actions") or [])
        turns = heading_restore_actions(
            self.env.sim.get_agent_state().rotation, match["rotation"],
            self.s, budget - used)
        if turns and not self.env.episode_over:
            self.obs, heading_exec, heading_frames = execute(
                self.env, turns, self.processor, self.obs, len(turns), self.s)
            execution["heading_actions"] = list(heading_exec.get("actions") or [])
            execution["actions"] = list(execution.get("actions") or []) + execution["heading_actions"]
            execution.setdefault("trajectory", []).extend(heading_exec.get("trajectory") or [])
            frames.extend(heading_frames)
        self._absorb(execution, frames)
        self.state.record_decision("go_back", f"checkpoint {checkpoint_id}", execution)
        report = self._post_move_report(execution)
        report["returned_to"] = match["note"]
        return report

    def stop(self, params):
        evidence = str(params.get("evidence", "")).strip()
        if not evidence:
            raise SkillError("evidence is required: state what you see that shows the "
                             "instruction is complete")
        self.state.stopped = True
        self.state.stop_evidence = evidence
        return {"stopped": True, "evidence": evidence}

    # --- dispatch --------------------------------------------------------------
    HANDLERS = ("observe", "observe_panorama", "observe_local_map", "recall",
                "note_progress", "refine_instruction", "navigate_to", "move",
                "go_back", "stop")

    def dispatch(self, method, params):
        started = time.perf_counter()
        if method not in self.HANDLERS:
            result = error = f"unknown skill {method!r}"
            result = {"error": error}
        else:
            try:
                result = getattr(self, method)(params or {})
                error = None
            except SkillError as exc:
                result, error = {"error": str(exc)}, str(exc)
            except Exception as exc:
                # A bad sensor frame or planner edge case should become a recoverable
                # tool error, not strand the SDK client waiting on a dead socket.
                error = f"{type(exc).__name__}: {exc}"
                result = {"error": f"internal skill failure: {error}"}
        self.state.transcript.append({
            "skill": method, "params": jsonable(params or {}),
            "result": jsonable(result), "error": error,
            "elapsed_s": round(time.perf_counter() - started, 3)})
        return result


def quat_to_list(rotation):
    return [float(rotation.x), float(rotation.y), float(rotation.z), float(rotation.w)]


def agent_yaw_deg(rotation):
    """Habitat agent yaw in degrees; identity looks along -Z, LEFT increases yaw."""
    if not hasattr(rotation, "w"):
        import quaternion as nq
        x, y, z, w = (float(v) for v in rotation)
        rotation = nq.quaternion(w, x, y, z)
    forward = quaternion_rotate_vector(rotation, np.array([0.0, 0.0, -1.0]))
    return float(np.degrees(np.arctan2(-forward[0], -forward[2])))


def heading_restore_actions(current_rotation, target_rotation, settings, max_steps):
    """Discrete TURN_LEFT/RIGHT sequence that undoes a heading error."""
    delta = ((agent_yaw_deg(target_rotation) - agent_yaw_deg(current_rotation) + 180.0)
             % 360.0) - 180.0
    step = float(getattr(settings, "turn_step", 15.0) or 15.0)
    n = int(round(abs(delta) / step))
    if n <= 0 or max_steps <= 0:
        return []
    return ([LEFT] if delta > 0 else [RIGHT]) * min(n, int(max_steps))


def macro_to_actions(text, settings):
    """'TURN_LEFT(30) FORWARD(50)' -> atomic action ids, via eval.py's own grammar.

    Routing through parse_model_output keeps one implementation of magnitude
    clamping and macro->atomic expansion, so a skill can never execute a step
    size the text protocol would have rejected.
    """
    text = str(text or "").strip()
    if not text:
        return []
    parsed = parse_model_output(f"<action>{text}</action>", settings)
    return parsed.get("action_sequence") or []


def serve(socket_path, server, process, poll_interval=0.5, deadline_s=None):
    """Serve skill calls until the bridge process exits or stop() is called.

    Habitat is driven from this loop's thread and no other. The listening socket
    carries a timeout so a bridge that dies mid-call cannot wedge the episode,
    and `deadline_s` bounds a bridge that hangs without exiting -- otherwise one
    stuck episode would stall an unattended multi-episode run indefinitely.
    """
    # Fail loudly and early: the kernel's AF_UNIX limit is ~108 bytes, and a path
    # a couple of bytes over only surfaces as a bind() error mid-episode.
    if len(socket_path.encode()) >= 108:
        raise ValueError(
            f"socket path is {len(socket_path.encode())} bytes, over the ~108-byte "
            f"AF_UNIX limit; use a shorter directory: {socket_path}")
    if os.path.exists(socket_path):
        os.unlink(socket_path)
    started = time.perf_counter()
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        listener.bind(socket_path)
        listener.listen(4)
        listener.settimeout(poll_interval)
        while True:
            if process.poll() is not None:
                break
            if deadline_s is not None and time.perf_counter() - started > deadline_s:
                server.state.timed_out = True
                break
            try:
                connection, _ = listener.accept()
            except socket.timeout:
                continue
            with connection:
                try:
                    payload = _read_line(connection)
                except (SkillError, UnicodeDecodeError) as exc:
                    _write_line(connection, {"error": f"invalid request: {exc}"})
                    continue
                if not payload:
                    continue
                try:
                    request = json.loads(payload)
                except json.JSONDecodeError as exc:
                    _write_line(connection, {"error": f"malformed request: {exc}"})
                    continue
                response = server.dispatch(request.get("method"), request.get("params"))
                _write_line(connection, response)
            if server.state.stopped or server.env.episode_over:
                break
    finally:
        listener.close()
        if os.path.exists(socket_path):
            os.unlink(socket_path)


def _read_line(connection):
    chunks = []
    size = 0
    while True:
        chunk = connection.recv(65536)
        if not chunk:
            break
        chunks.append(chunk)
        size += len(chunk)
        if size > MAX_RPC_REQUEST_BYTES:
            raise SkillError("skill request exceeded 1 MiB")
        if b"\n" in chunk:
            break
    return b"".join(chunks).split(b"\n", 1)[0].decode()


def _write_line(connection, payload):
    connection.sendall((json.dumps(jsonable(payload)) + "\n").encode())
