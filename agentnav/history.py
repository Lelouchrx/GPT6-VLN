from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

from .config import HistoryConfig


STRATEGIES = ("all_uniform", "keyframe_uniform", "recent", "hybrid")


def uniform_indices(count: int, keep: int) -> List[int]:
    """Evenly spaced indices over [0, count-1], endpoints included."""
    if count <= 0 or keep <= 0:
        return []
    if count <= keep:
        return list(range(count))
    return [int(i) for i in np.linspace(0, count - 1, keep)]


class HistoryManager:
    """Visual memory for the selector.

    `all_uniform` keeps every atomic frame and subsamples at read time, so the
    8 frames always span the whole episode. `keyframe_uniform` only records
    frames the agent actually reasoned over, which is what AgentVLN does.
    """

    def __init__(self, config: HistoryConfig):
        if config.strategy not in STRATEGIES:
            raise ValueError(f"unknown history strategy {config.strategy!r}, expected one of {STRATEGIES}")
        self.config = config
        self.frames: List[np.ndarray] = []
        self.kinds: List[str] = []

    def reset(self) -> None:
        self.frames = []
        self.kinds = []

    def add(self, rgb: np.ndarray, kind: str = "atomic") -> None:
        if rgb is None:
            return
        if self.config.strategy == "keyframe_uniform" and kind != "decision":
            return
        self.frames.append(np.asarray(rgb, dtype=np.uint8)[..., :3].copy())
        self.kinds.append(kind)

    def select(self) -> List[np.ndarray]:
        keep = self.config.max_frames
        total = len(self.frames)
        if keep <= 0 or total == 0:
            return []
        strategy = self.config.strategy
        if strategy == "recent":
            return list(self.frames[-keep:])
        if strategy == "hybrid":
            recent = max(1, keep // 2)
            tail = self.frames[-recent:]
            head_pool = self.frames[: max(0, total - recent)]
            head = [head_pool[i] for i in uniform_indices(len(head_pool), keep - len(tail))]
            return head + list(tail)
        return [self.frames[i] for i in uniform_indices(total, keep)]

    def describe(self) -> Dict[str, Any]:
        return {
            "strategy": self.config.strategy,
            "stored": len(self.frames),
            "emitted": min(self.config.max_frames, len(self.frames)),
            "decision_frames": sum(1 for k in self.kinds if k == "decision"),
        }


def format_text_history(trace: Sequence[Dict[str, Any]], limit: int = 12) -> str:
    """Compact textual trail of what the agent already did."""
    items = []
    for entry in list(trace)[-limit:]:
        execution = entry.get("execution", {})
        items.append({
            "turn": entry.get("turn"),
            "chose": entry.get("selected_waypoint_id"),
            "actions": execution.get("actions", []),
            "status": execution.get("status"),
        })
    return items
