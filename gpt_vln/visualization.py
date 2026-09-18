from __future__ import annotations

import cv2
import numpy as np

UNKNOWN, FREE, OCCUPIED = np.uint8(127), np.uint8(255), np.uint8(0)


def save_route_topmap(path, sim, episode, predicted_path, inference_points=()) -> None:
    from .habitat_extensions.maps import draw_routes

    image = draw_routes(
        sim,
        getattr(episode, "reference_path", []),
        predicted_path,
        float(episode.start_position[1]),
        inference_points,
    )
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def save_inference_input(path, history_frames, current_frame) -> None:
    frames = list(history_frames) + [current_frame]
    tile_w, tile_h = 240, 180
    canvas = np.full((tile_h + 30, tile_w * len(frames), 3), 245, dtype=np.uint8)
    for index, frame in enumerate(frames):
        image = cv2.cvtColor(np.asarray(frame, dtype=np.uint8)[..., :3], cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        canvas[:tile_h, index * tile_w:(index + 1) * tile_w] = image
        label = "current" if index == len(frames) - 1 else f"history {index + 1}"
        cv2.putText(canvas, label, (index * tile_w + 8, tile_h + 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (25, 25, 25), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def annotate_rgb(rgb: np.ndarray, waypoints: list) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.uint8)[..., :3].copy()
    for waypoint in waypoints:
        x, y = waypoint.pixel_xy
        cv2.circle(image, (x, y), 14, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(image, (x, y), 11, (30, 225, 70), -1, cv2.LINE_AA)
        label = str(waypoint.waypoint_id)
        size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
        cv2.putText(image, label, (x-size[0]//2, y+size[1]//2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
    return image


def color_local_map(local_map: np.ndarray, waypoints: list, meters_per_pixel: float) -> np.ndarray:
    image = np.empty((*local_map.shape, 3), dtype=np.uint8)
    image[local_map == UNKNOWN] = (65, 65, 65)
    image[local_map == FREE] = (238, 238, 238)
    image[local_map == OCCUPIED] = (18, 18, 18)
    center = (local_map.shape[0] - 1) / 2
    for waypoint in waypoints:
        angle = np.deg2rad(waypoint.bearing_deg)
        forward = waypoint.distance_m * np.cos(angle)
        right = waypoint.distance_m * np.sin(angle)
        row = int(round(center - forward / meters_per_pixel))
        col = int(round(center + right / meters_per_pixel))
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


def save_depth(path, depth_m: np.ndarray, max_depth_m: float) -> None:
    depth = np.clip(depth_m / max_depth_m * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.applyColorMap(255 - depth, cv2.COLORMAP_TURBO))


def save_dashboard(path, rgb: np.ndarray, local_map: np.ndarray, instruction: str) -> None:
    cv2.imwrite(str(path), dashboard(rgb, local_map, instruction))
