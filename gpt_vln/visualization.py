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


def save_inference_input(path, history_frames, current_frame, local_map=None) -> None:
    frames = list(history_frames) + [current_frame]
    if local_map is not None:
        frames.append(local_map)
    tile_h = 180
    widths = [max(240, round(tile_h * np.asarray(frame).shape[1] /
                             np.asarray(frame).shape[0])) for frame in frames]
    offsets = np.cumsum([0] + widths[:-1]).tolist()
    canvas = np.full((tile_h + 30, sum(widths), 3), 245, dtype=np.uint8)
    for index, frame in enumerate(frames):
        image = cv2.cvtColor(np.asarray(frame, dtype=np.uint8)[..., :3], cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (widths[index], tile_h), interpolation=cv2.INTER_AREA)
        start = offsets[index]
        canvas[:tile_h, start:start + widths[index]] = image
        if local_map is not None and index == len(frames) - 1:
            label = "local map"
        elif index == len(history_frames):
            label = "current"
        else:
            label = f"history {index + 1}"
        cv2.putText(canvas, label, (start + 8, tile_h + 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (25, 25, 25), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def color_local_map(local_map: np.ndarray, waypoints: list, meters_per_pixel: float,
                    scale: int = 3) -> np.ndarray:
    """Render recent RGB-D geometry with a metric grid."""
    image = np.empty((*local_map.shape, 3), dtype=np.uint8)
    image[local_map == UNKNOWN] = (65, 65, 65)
    image[local_map == FREE] = (238, 238, 238)
    image[local_map == OCCUPIED] = (18, 18, 18)
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    size = image.shape[0]
    center = (local_map.shape[0] - 1) / 2

    step = int(round(scale / meters_per_pixel))  # one metre
    for offset in range(step, size, step):
        for position in (int(center*scale) + offset, int(center*scale) - offset):
            if 0 <= position < size:
                cv2.line(image, (position, 0), (position, size), (110, 110, 110), 1)
                cv2.line(image, (0, position), (size, position), (110, 110, 110), 1)
    cv2.putText(image, "grid = 1m", (6, size-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)

    for waypoint in waypoints:
        angle = np.deg2rad(waypoint.bearing_deg)
        row = int(round(center - waypoint.distance_m*np.cos(angle)/meters_per_pixel))
        col = int(round(center + waypoint.distance_m*np.sin(angle)/meters_per_pixel))
        cv2.circle(image, (col*scale, row*scale), 5, (40, 210, 60), -1)
        cv2.putText(image, str(waypoint.waypoint_id), (col*scale+6, row*scale-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 180, 50), 1, cv2.LINE_AA)
    marker = int(center*scale)
    cv2.drawMarker(image, (marker, marker), (40, 60, 240), cv2.MARKER_TRIANGLE_UP, 14, 2)
    return image


def overlay_navigation(rgb: np.ndarray, waypoints: list) -> np.ndarray:
    """Annotate only AgentVLN's current visible frontier candidates."""
    image = np.asarray(rgb, dtype=np.uint8)[..., :3].copy()
    for waypoint in waypoints:
        x, y = waypoint.pixel_xy
        cv2.circle(image, (x, y), 13, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(image, (x, y), 10, (30, 225, 70), -1, cv2.LINE_AA)
        cv2.putText(image, str(waypoint.waypoint_id), (x-5, y+6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 2, cv2.LINE_AA)
    return image


def overlay_selected_waypoint(rgb: np.ndarray, coordinate, resolved_coordinate=None) -> np.ndarray:
    """Draw the model-selected image coordinate on the current RGB frame."""
    image = np.asarray(rgb, dtype=np.uint8)[..., :3].copy()
    if coordinate is None:
        return image
    x, y = (int(round(float(value))) for value in coordinate)
    height, width = image.shape[:2]
    if not (0 <= x < width and 0 <= y < height):
        return image
    cv2.circle(image, (x, y), 16, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.circle(image, (x, y), 11, (230, 45, 35), 3, cv2.LINE_AA)
    cv2.drawMarker(image, (x, y), (230, 45, 35), cv2.MARKER_CROSS, 30, 3,
                   cv2.LINE_AA)
    label = f"waypoint ({x}, {y})"
    (text_width, text_height), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
    label_x = min(max(4, x + 18), max(4, width - text_width - 8))
    label_y = min(max(text_height + 8, y - 18), height - 8)
    cv2.rectangle(image, (label_x - 4, label_y - text_height - 6),
                  (label_x + text_width + 4, label_y + 5), (0, 0, 0), -1)
    cv2.putText(image, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, (255, 255, 255), 2, cv2.LINE_AA)
    if resolved_coordinate is not None:
        rx, ry = (int(round(float(value))) for value in resolved_coordinate)
        if 0 <= rx < width and 0 <= ry < height:
            cv2.line(image, (x, y), (rx, ry), (40, 220, 70), 2, cv2.LINE_AA)
            cv2.drawMarker(image, (rx, ry), (40, 220, 70), cv2.MARKER_CROSS,
                           30, 3, cv2.LINE_AA)
            cv2.circle(image, (rx, ry), 12, (40, 220, 70), 3, cv2.LINE_AA)
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
    side = cv2.cvtColor(local_map, cv2.COLOR_RGB2BGR)
    side = cv2.resize(side, (rgb_bgr.shape[0], rgb_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    canvas = np.hstack([rgb_bgr, side])
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(canvas, instruction[:110], (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def save_depth(path, depth_m: np.ndarray, max_depth_m: float) -> None:
    depth = np.clip(depth_m / max_depth_m * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.applyColorMap(255 - depth, cv2.COLORMAP_TURBO))


def save_dashboard(path, rgb: np.ndarray, local_map: np.ndarray, instruction: str) -> None:
    cv2.imwrite(str(path), dashboard(rgb, local_map, instruction))
