from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .actions import ActionId
from .config import AgentConfig
from .types import CameraIntrinsics

STOP, FORWARD, TURN_LEFT, TURN_RIGHT = 0, 1, 2, 3


def forward_clearance_m(
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    band: Sequence[float],
    robot_radius_m: float,
    min_valid_px: int,
) -> float:
    """Free distance straight ahead inside the robot's width, from depth only.

    Returns +inf when the depth image carries too little evidence to judge, so
    a blind patch never fabricates an obstacle.
    """
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    if depth.ndim != 2 or depth.size == 0:
        return float("inf")
    height, width = depth.shape
    row0 = max(0, int(band[0] * height))
    row1 = min(height, max(row0 + 1, int(band[1] * height)))
    strip = depth[row0:row1]
    columns = np.arange(width, dtype=np.float32)[None, :]
    valid = np.isfinite(strip) & (strip > 1e-3)
    # Horizontal offset in metres of each pixel at its own measured depth.
    offset = np.abs(columns - intrinsics.cx) * strip / intrinsics.fx
    inside = valid & (offset <= robot_radius_m)
    samples = strip[inside]
    if samples.size < min_valid_px:
        return float("inf")
    return float(np.percentile(samples, 5.0))


class ChunkExecutor:
    """Executes the model's primitives, guarding FORWARD with depth.

    Uses no navmesh: no snap_point, no is_navigable, no find_path, no
    ShortestPathFollower. A turn always runs; a forward step is skipped when the
    depth image shows the corridor blocked, and the chunk is abandoned so the
    model sees the obstacle on its next turn rather than grinding into it.
    """

    def __init__(self, env: Any, config: AgentConfig):
        self.env = env
        self.sim = env.sim
        self.config = config
        self.executor = config.executor
        self.intrinsics = CameraIntrinsics.from_hfov(config.width, config.height, config.hfov_deg)

    def _position(self) -> np.ndarray:
        return np.asarray(self.sim.get_agent_state().position, dtype=np.float64)

    def _depth_of(self, observation) -> Optional[np.ndarray]:
        from .mapping import depth_to_meters

        if observation is None or "depth" not in observation:
            return None
        return depth_to_meters(observation["depth"], self.config.depth_max_m, self.config.depth_normalized)

    def _blocked(self, depth_m: Optional[np.ndarray]) -> bool:
        if depth_m is None:
            return False
        clearance = forward_clearance_m(
            depth_m,
            self.intrinsics,
            self.executor.collision_band,
            self.config.map.robot_radius_m,
            self.executor.collision_min_valid_px,
        )
        return clearance < self.executor.collision_clearance_m

    def execute(
        self,
        primitives: Sequence[int],
        depth_m: Optional[np.ndarray] = None,
        max_actions: Optional[int] = None,
    ) -> Dict[str, Any]:
        budget = len(primitives)
        if max_actions is not None:
            budget = min(budget, max(0, int(max_actions)))
        actions: List[int] = []
        observation = None
        current_depth = depth_m
        status = "chunk_complete"
        stalls = 0
        start = self._position()

        for action in list(primitives)[:budget]:
            action = int(action)
            if action == STOP:
                observation = self.env.step(STOP)
                actions.append(STOP)
                status = "stopped"
                break
            if action == FORWARD and self._blocked(current_depth):
                status = "blocked"
                break
            before = self._position()
            observation = self.env.step(action)
            actions.append(action)
            fresh = self._depth_of(observation)
            if fresh is not None:
                current_depth = fresh
            if action == FORWARD and float(np.linalg.norm(self._position() - before)) < self.executor.stall_epsilon_m:
                stalls += 1
                if stalls >= self.executor.max_stalls:
                    status = "collision"
                    break
            if self.env.episode_over:
                status = "episode_over"
                break

        position = self._position()
        return {
            "status": status,
            "actions": actions,
            "observation": observation,
            "final_position": position.tolist(),
            "travelled_m": float(np.linalg.norm(position[[0, 2]] - start[[0, 2]])),
            "stalls": stalls,
        }

    def stop(self) -> Dict[str, Any]:
        observation = self.env.step(STOP)
        return {"status": "stopped", "actions": [STOP], "observation": observation}
