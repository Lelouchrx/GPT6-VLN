from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .config import AgentConfig
from .controller import ChunkExecutor
from .habitat_env import ensure_output_dir
from .history import HistoryManager
from .mapping import LocalMapper, depth_to_meters
from .selector import Decision
from .types import Observation
from .visualization import annotate_rgb, color_global_map, color_local_map, dashboard


class NavigationAgent:
    """One inference per turn: history + annotated view -> action chunk -> execute."""

    def __init__(self, env, selector, config: AgentConfig):
        self.env = env
        self.selector = selector
        self.config = config
        self.mapper = LocalMapper(env.sim, config, depth_normalized=config.depth_normalized)
        self.controller = ChunkExecutor(env, config)
        self.history = HistoryManager(config.history)

    def reset(self) -> Dict[str, Any]:
        self.mapper.reset()
        self.history.reset()
        return self.env.reset()

    def observe(self, raw_observation: Optional[Dict[str, Any]] = None) -> Observation:
        raw = raw_observation if raw_observation is not None else self.env.sim.get_sensor_observations()
        rgb = np.asarray(raw["rgb"], dtype=np.uint8)[..., :3]
        depth_m = depth_to_meters(raw["depth"], self.config.depth_max_m, self.config.depth_normalized)
        map_state = self.mapper.update()
        waypoints, diagnostics = self.mapper.sample_waypoints(map_state.local_map, raw["depth"])
        return Observation(
            rgb=rgb,
            depth_m=depth_m,
            annotated_rgb=annotate_rgb(rgb, waypoints),
            local_map=map_state.local_map,
            global_map=color_global_map(map_state.full_map, map_state.observed_mask, map_state.agent_grid_rc),
            waypoints=waypoints,
            instruction=self.env.current_episode.instruction.instruction_text,
            episode_id=str(self.env.current_episode.episode_id),
            diagnostics=diagnostics,
            local_map_raw=map_state.local_map,
        )

    def run_episode(self, execute: bool = True) -> Dict[str, Any]:
        budget = self.config.budget
        raw = self.reset()
        episode_dir = ensure_output_dir(
            self.config.output_dir / f"episode_{self.env.current_episode.episode_id}"
        )
        trace: List[Dict[str, Any]] = []
        actions_used = 0
        executed: List[str] = []

        for turn in range(budget.max_agent_turns):
            remaining = budget.max_env_actions - actions_used
            if remaining <= 0:
                break
            observation = self.observe(raw)
            self._save_observation(episode_dir, turn, observation)
            # Every atomic frame is stored; 8 are sampled uniformly at read time.
            self.history.add(observation.rgb, kind="atomic")

            decision = self.selector.select(
                observation.instruction,
                self.history.select(),
                observation.annotated_rgb,
                observation.waypoints,
                executed if self.config.history.include_text_history else None,
            )
            print(f"  turn {turn}: {decision.raw[:120]!r} -> {decision.text} ({len(decision.primitives)} primitives)")

            if not execute:
                trace.append(self._trace_item(turn, observation, decision, {"status": "dry_run"}))
                break

            if decision.is_stop_only:
                action_result = self.controller.stop()
            elif decision.primitives:
                action_result = self.controller.execute(decision.primitives, observation.depth_m, remaining)
            else:
                # Unparseable answer: do nothing this turn rather than guess.
                action_result = {"status": "no_action", "actions": [], "observation": None}

            executed.append(_outcome(decision.text, action_result))
            raw = action_result.pop("observation", None)
            actions_used += len(action_result.get("actions", []))
            trace.append(self._trace_item(turn, observation, decision, action_result))
            self._write_json(episode_dir / "trace.json", trace)
            if self.env.episode_over or action_result["status"] == "stopped":
                break

        if execute and not self.env.episode_over:
            final = self.controller.stop()
            final.pop("observation", None)
            trace.append({"turn": len(trace), "chunk": "stop (budget exhausted)", "execution": final})
            self._write_json(episode_dir / "trace.json", trace)

        result = {
            "episode_id": str(self.env.current_episode.episode_id),
            "scene_id": self.env.current_episode.scene_id,
            "instruction": self.env.current_episode.instruction.instruction_text,
            "turns": len(trace),
            "model_calls": getattr(self.selector, "calls", None),
            "total_actions": sum(len(i.get("execution", {}).get("actions", [])) for i in trace),
            "history": self.history.describe(),
            "chunks": executed,
            "trace": trace,
            "metrics": _jsonable(self.env.get_metrics()),
        }
        self._write_json(episode_dir / "result.json", result)
        return result

    def _save_observation(self, output: Path, turn: int, observation: Observation) -> None:
        local_color = color_local_map(
            observation.local_map, observation.waypoints, self.config.map.meters_per_pixel
        )
        cv2.imwrite(str(output / f"turn_{turn:02d}_raw_rgb.png"), cv2.cvtColor(observation.rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(output / f"turn_{turn:02d}_rgb.png"), cv2.cvtColor(observation.annotated_rgb, cv2.COLOR_RGB2BGR))
        depth = np.clip(observation.depth_m / self.config.depth_max_m * 255, 0, 255).astype(np.uint8)
        cv2.imwrite(str(output / f"turn_{turn:02d}_depth.png"), cv2.applyColorMap(255 - depth, cv2.COLORMAP_TURBO))
        cv2.imwrite(str(output / f"turn_{turn:02d}_local_map.png"), local_color)
        cv2.imwrite(str(output / f"turn_{turn:02d}_global_fog.png"), observation.global_map)
        cv2.imwrite(str(output / f"turn_{turn:02d}_dashboard.png"),
                    dashboard(observation.annotated_rgb, local_color, observation.instruction))
        self._write_json(output / f"turn_{turn:02d}_waypoints.json", {
            "agent_position": _jsonable(self.env.sim.get_agent_state().position),
            "diagnostics": observation.diagnostics,
            "waypoints": [item.to_dict() for item in observation.waypoints],
        })

    @staticmethod
    def _trace_item(turn: int, observation: Observation, decision: Decision, execution) -> Dict[str, Any]:
        return {
            "turn": turn,
            "raw_output": decision.raw,
            "chunk": decision.text,
            "primitives": decision.primitives,
            "candidate_count": len(observation.waypoints),
            "diagnostics": observation.diagnostics,
            "execution": _jsonable(execution),
        }

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.write_text(json.dumps(_jsonable(value), indent=2), encoding="utf-8")


_OUTCOME_NOTE = {
    "collision": "BLOCKED - the robot did not move, something is in the way",
    "blocked": "BLOCKED - the robot did not move, something is in the way",
    "no_action": "NOT UNDERSTOOD - no motion command was parsed",
}


def _outcome(chunk_text: str, execution: Dict[str, Any]) -> Dict[str, Any]:
    """What the model ordered *and* what actually happened.

    Without the outcome the model cannot tell a refused command from a
    completed one, and repeats a failing command until the turn budget runs out.
    """
    status = execution.get("status", "")
    travelled = float(execution.get("travelled_m", 0.0) or 0.0)
    entry = {
        "commanded": chunk_text,
        "moved_m": round(travelled, 2),
        "result": _OUTCOME_NOTE.get(status, status),
    }
    return entry


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value
