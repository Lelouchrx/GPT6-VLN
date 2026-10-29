"""Top-map drawing helpers for reference and predicted trajectories."""

from typing import Iterable, Sequence

import cv2
import numpy as np
from habitat.utils.visualizations import maps as habitat_maps


def draw_routes(
    sim,
    reference_path: Iterable[Sequence[float]],
    predicted_path: Iterable[Sequence[float]],
    height: float,
    inference_points=(),
    resolution: int = 1024,
) -> np.ndarray:
    raw = habitat_maps.get_topdown_map(sim.pathfinder, height, map_resolution=resolution)
    image = habitat_maps.colorize_topdown_map(raw)

    def project(points):
        result = []
        for point in points:
            row, col = habitat_maps.to_grid(point[2], point[0], raw.shape, sim=sim)
            result.append((int(col), int(row)))
        return result

    gt = project(reference_path)
    pred = project(predicted_path)
    if len(gt) > 1:
        cv2.polylines(image, [np.asarray(gt)], False, (35, 190, 75), 5, cv2.LINE_AA)
    if len(pred) > 1:
        cv2.polylines(image, [np.asarray(pred)], False, (235, 70, 45), 5, cv2.LINE_AA)
    if pred:
        cv2.circle(image, pred[0], 8, (40, 100, 240), -1)
        cv2.circle(image, pred[-1], 8, (235, 70, 45), -1)
    if gt:
        cv2.circle(image, gt[-1], 8, (35, 190, 75), -1)
    occupied = []
    for turn, point in inference_points:
        marker = project([point])[0]
        cv2.circle(image, marker, 7, (255, 210, 30), -1, cv2.LINE_AA)
        cv2.circle(image, marker, 8, (25, 25, 25), 1, cv2.LINE_AA)
        label = f"T{turn}"
        (width, height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 2)
        label_pos = None
        for radius in (14, 24, 36, 50, 66, 84):
            for angle in np.linspace(0, 2*np.pi, 16, endpoint=False):
                x = int(marker[0] + radius*np.cos(angle))
                y = int(marker[1] + radius*np.sin(angle))
                rect = (x-2, y-height-2, x+width+2, y+baseline+2)
                in_bounds = rect[0] >= 0 and rect[1] >= 0 and rect[2] < image.shape[1] and rect[3] < image.shape[0]
                overlaps = any(not (rect[2] < old[0] or rect[0] > old[2] or rect[3] < old[1] or rect[1] > old[3]) for old in occupied)
                if in_bounds and not overlaps:
                    label_pos = (x, y); occupied.append(rect); break
            if label_pos is not None:
                break
        if label_pos is None:
            label_pos = (marker[0] + 10, marker[1] - 9)
        cv2.line(image, marker, label_pos, (80, 80, 80), 1, cv2.LINE_AA)
        cv2.putText(image, label, label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 2, cv2.LINE_AA)
    return image
