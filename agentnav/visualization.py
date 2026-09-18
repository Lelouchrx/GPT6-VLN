from __future__ import annotations

import cv2
import numpy as np

from .mapping import FREE, OCCUPIED, UNKNOWN
from .types import Waypoint


def annotate_rgb(rgb: np.ndarray, waypoints: list[Waypoint]) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.uint8)[..., :3].copy()
    for waypoint in waypoints:
        x, y = waypoint.pixel_xy
        cv2.circle(image, (x, y), 14, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(image, (x, y), 11, (30, 225, 70) if waypoint.source != "frontier" else (40, 170, 255), -1, cv2.LINE_AA)
        label = str(waypoint.waypoint_id)
        size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
        cv2.putText(image, label, (x-size[0]//2, y+size[1]//2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
    return image


def color_local_map(local_map: np.ndarray, waypoints: list[Waypoint], meters_per_pixel: float) -> np.ndarray:
    image = np.empty((*local_map.shape, 3), dtype=np.uint8)
    image[local_map == UNKNOWN] = (65, 65, 65)
    image[local_map == FREE] = (238, 238, 238)
    image[local_map == OCCUPIED] = (18, 18, 18)
    center = (local_map.shape[0] - 1) / 2
    for waypoint in waypoints:
        row = int(round(center - waypoint.local_forward_m / meters_per_pixel))
        col = int(round(center + waypoint.local_right_m / meters_per_pixel))
        cv2.circle(image, (col, row), 5, (40, 210, 60), -1)
        cv2.putText(image, str(waypoint.waypoint_id), (col+5, row-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30, 180, 50), 1)
    c = int(round(center))
    cv2.drawMarker(image, (c, c), (40, 60, 240), cv2.MARKER_TRIANGLE_UP, 12, 2)
    return image


def color_global_map(full_map: np.ndarray, observed_mask: np.ndarray, agent_rc) -> np.ndarray:
    image = np.full((*full_map.shape, 3), 55, dtype=np.uint8)
    visible = observed_mask > 0
    image[visible & (full_map > 0)] = (235, 235, 235)
    image[visible & (full_map == 0)] = (15, 15, 15)
    row, col = agent_rc
    cv2.circle(image, (col, row), 5, (40, 60, 240), -1)
    return image


def dashboard(rgb: np.ndarray, local_map: np.ndarray, instruction: str) -> np.ndarray:
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    side = cv2.resize(local_map, (rgb_bgr.shape[0], rgb_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    canvas = np.hstack([rgb_bgr, side])
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(canvas, instruction[:110], (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas
