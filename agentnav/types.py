from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_hfov(cls, width: int, height: int, hfov_deg: float) -> "CameraIntrinsics":
        hfov = np.deg2rad(hfov_deg)
        fx = width / (2.0 * np.tan(hfov / 2.0))
        vfov = 2.0 * np.arctan(np.tan(hfov / 2.0) * height / width)
        fy = height / (2.0 * np.tan(vfov / 2.0))
        return cls(width, height, float(fx), float(fy), (width - 1) / 2, (height - 1) / 2)


@dataclass(frozen=True)
class Waypoint:
    waypoint_id: int
    world_xyz: Tuple[float, float, float]
    pixel_xy: Tuple[int, int]
    local_right_m: float
    local_forward_m: float
    distance_m: float
    bearing_deg: float
    geodesic_m: float
    depth_m: Optional[float]
    source: str = "bearing"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Observation:
    rgb: np.ndarray
    depth_m: np.ndarray
    annotated_rgb: np.ndarray
    local_map: np.ndarray
    global_map: np.ndarray
    waypoints: list[Waypoint]
    instruction: str
    episode_id: str
    diagnostics: Dict[str, Any]
    local_map_raw: Optional[np.ndarray] = None
