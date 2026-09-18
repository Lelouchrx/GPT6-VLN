"""Continuous action chunks and their expansion into Habitat primitives.

The text format is the one already used by vln_real/actions.py -- sub-actions
separated by ", ", each carrying a magnitude -- so the simulator agent and the
real-robot agent consume model output identically.

Qwen-RobotNav is NOT the model to copy here: it emits a numeric waypoint
trajectory (8 x (x, y, theta), 24 dims) from a trained MLP regression head, so
its interface is unavailable to a zero-shot API model. Its *inputs* are the
"move forward 2.0 meters" style strings, which is the shape adopted below.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Tuple

STOP, FORWARD, TURN_LEFT, TURN_RIGHT = 0, 1, 2, 3


class ActionId(IntEnum):
    STOP = 0
    FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3


@dataclass(frozen=True)
class SubAction:
    """One continuous command, before quantisation to primitives."""

    action: ActionId
    forward_cm: float = 0.0
    turn_deg: float = 0.0

    def __str__(self) -> str:
        if self.action == ActionId.FORWARD:
            return f"forward {self.forward_cm:.0f}"
        if self.action == ActionId.TURN_LEFT:
            return f"turn left {self.turn_deg:.0f}"
        if self.action == ActionId.TURN_RIGHT:
            return f"turn right {self.turn_deg:.0f}"
        return "stop"


def parse_sub_action(text: str, forward_cm_range: Tuple[float, float],
                     turn_deg_range: Tuple[float, float]) -> Optional[SubAction]:
    """Parse one sub-action, clamping its magnitude into the allowed range."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    body = (match.group(1) if match else text).strip().lower()
    if not body:
        return None
    if "stop" in body:
        return SubAction(ActionId.STOP)
    number = re.search(r"-?\d+(?:\.\d+)?", body)
    value = abs(float(number.group())) if number else None
    if "forward" in body:
        lo, hi = forward_cm_range
        return SubAction(ActionId.FORWARD, forward_cm=float(np_clip(value if value is not None else lo, lo, hi)))
    if "left" in body:
        lo, hi = turn_deg_range
        return SubAction(ActionId.TURN_LEFT, turn_deg=float(np_clip(value if value is not None else lo, lo, hi)))
    if "right" in body:
        lo, hi = turn_deg_range
        return SubAction(ActionId.TURN_RIGHT, turn_deg=float(np_clip(value if value is not None else lo, lo, hi)))
    return None


def np_clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def parse_chunk(text: str, forward_cm_range: Tuple[float, float], turn_deg_range: Tuple[float, float],
                max_sub_actions: Optional[int] = None) -> List[SubAction]:
    """Split on ", " and parse each piece, stopping at the first STOP.

    The chunk length is unbounded by default: the model decides how far to
    commit, and only each magnitude is clamped to its allowed range.
    """
    chunk: List[SubAction] = []
    for piece in str(text).split(","):
        parsed = parse_sub_action(piece, forward_cm_range, turn_deg_range)
        if parsed is None:
            continue
        chunk.append(parsed)
        if parsed.action == ActionId.STOP:
            break
        if max_sub_actions is not None and len(chunk) >= max_sub_actions:
            break
    return chunk


def to_primitives(chunk: List[SubAction], forward_step_m: float, turn_angle_deg: float) -> List[int]:
    """Quantise a chunk into Habitat's discrete actions.

    A 'forward 75' with a 0.25 m step becomes three FORWARD actions; a
    'turn left 45' with a 15 deg step becomes three TURN_LEFT actions. Each
    magnitude yields at least one primitive so a command is never dropped.
    """
    step_cm = forward_step_m * 100.0
    primitives: List[int] = []
    for sub in chunk:
        if sub.action == ActionId.STOP:
            primitives.append(int(ActionId.STOP))
            break
        if sub.action == ActionId.FORWARD:
            count = max(1, int(round(sub.forward_cm / step_cm)))
            primitives.extend([int(ActionId.FORWARD)] * count)
        else:
            count = max(1, int(round(sub.turn_deg / turn_angle_deg)))
            primitives.extend([int(sub.action)] * count)
    return primitives


def describe(chunk: List[SubAction]) -> str:
    return ", ".join(str(sub) for sub in chunk) if chunk else "(empty)"
