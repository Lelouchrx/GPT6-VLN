from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np


def generate_visualizations(episode_dir: Path) -> Path:
    output = episode_dir / "visualizations"
    output.mkdir(parents=True, exist_ok=True)
    result = _read_json(episode_dir / "result.json")
    trace = result.get("trace", [])

    groups = [
        ("01_raw_rgb_sequence.png", "Raw RGB sequence", "turn_*_raw_rgb.png"),
        ("02_waypoint_rgb_sequence.png", "RGB + projected waypoints", "turn_*_rgb.png"),
        ("03_depth_sequence.png", "RGB-D depth sequence", "turn_*_depth.png"),
        ("04_local_map_sequence.png", "Robot-centric 8m local maps", "turn_*_local_map.png"),
        ("05_global_fog_sequence.png", "PathFinder fog-of-war", "turn_*_global_fog.png"),
        ("06_dashboard_sequence.png", "RGB / local-map decision views", "turn_*_dashboard.png"),
    ]
    for filename, title, pattern in groups:
        images = [cv2.imread(str(path)) for path in sorted(episode_dir.glob(pattern))]
        images = [image for image in images if image is not None]
        cv2.imwrite(str(output / filename), _contact_sheet(images, title))

    cv2.imwrite(str(output / "07_world_trajectory.png"), _world_trajectory(episode_dir, trace))
    cv2.imwrite(str(output / "08_action_timeline.png"), _action_timeline(trace))
    cv2.imwrite(str(output / "09_waypoint_geometry.png"), _waypoint_geometry(episode_dir))
    cv2.imwrite(str(output / "10_result_summary.png"), _result_summary(result))
    return output


def _contact_sheet(images: List[np.ndarray], title: str) -> np.ndarray:
    if not images:
        return _text_canvas(title, ["No frames"])
    tile_w, tile_h = 480, 360
    tiles = []
    for index, image in enumerate(images):
        scale = min(tile_w / image.shape[1], tile_h / image.shape[0])
        resized = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
        tile = np.full((tile_h + 34, tile_w, 3), 28, dtype=np.uint8)
        y = 34 + (tile_h - resized.shape[0]) // 2
        x = (tile_w - resized.shape[1]) // 2
        tile[y:y+resized.shape[0], x:x+resized.shape[1]] = resized
        cv2.putText(tile, f"Turn {index + 1}", (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 235, 235), 1, cv2.LINE_AA)
        tiles.append(tile)
    columns = 2
    rows = int(np.ceil(len(tiles) / columns))
    sheet = np.full((50 + rows * tiles[0].shape[0], columns * tile_w, 3), 20, dtype=np.uint8)
    cv2.putText(sheet, title, (16, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (245, 245, 245), 2, cv2.LINE_AA)
    for index, tile in enumerate(tiles):
        row, col = divmod(index, columns)
        y, x = 50 + row * tile.shape[0], col * tile_w
        sheet[y:y+tile.shape[0], x:x+tile.shape[1]] = tile
    return sheet


def _world_trajectory(episode_dir: Path, trace) -> np.ndarray:
    positions = []
    targets = []
    for path in sorted(episode_dir.glob("turn_*_waypoints.json")):
        data = _read_json(path)
        if data.get("agent_position"):
            positions.append(data["agent_position"])
        targets.extend(item["world_xyz"] for item in data.get("waypoints", []))
    for item in trace:
        final = item.get("execution", {}).get("final_position")
        if final:
            positions.append(final)
    canvas = np.full((700, 900, 3), 245, dtype=np.uint8)
    points = positions + targets
    if not points:
        return canvas
    xs = np.array([point[0] for point in points]); zs = np.array([point[2] for point in points])
    pad = 0.8
    xmin, xmax = xs.min()-pad, xs.max()+pad; zmin, zmax = zs.min()-pad, zs.max()+pad
    def project(point):
        x = int(70 + (point[0]-xmin) / max(xmax-xmin, 1e-6) * 760)
        y = int(620 - (point[2]-zmin) / max(zmax-zmin, 1e-6) * 540)
        return x, y
    for a, b in zip(positions, positions[1:]):
        cv2.line(canvas, project(a), project(b), (225, 105, 30), 4, cv2.LINE_AA)
    for index, point in enumerate(positions):
        cv2.circle(canvas, project(point), 9, (220, 70, 35), -1)
        cv2.putText(canvas, str(index), (project(point)[0]+10, project(point)[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1)
    for point in targets:
        cv2.circle(canvas, project(point), 7, (40, 180, 60), 2)
    cv2.putText(canvas, "World trajectory (X/Z)", (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20, 20, 20), 2)
    return canvas


def _action_timeline(trace) -> np.ndarray:
    labels = {0: "STOP", 1: "FORWARD", 2: "LEFT", 3: "RIGHT"}
    lines = []
    for item in trace:
        actions = item.get("execution", {}).get("actions", [])
        action_text = " ".join(labels.get(value, str(value)) for value in actions) or "none"
        lines.append(f"Step {item['turn'] + 1} ({item.get('kind', 'final')}): waypoint={item.get('selected_waypoint_id')} | {action_text}")
    return _text_canvas("Discrete action timeline", lines)


def _waypoint_geometry(episode_dir: Path) -> np.ndarray:
    lines = []
    for turn, path in enumerate(sorted(episode_dir.glob("turn_*_waypoints.json")), 1):
        data = _read_json(path)
        if not data.get("waypoints"):
            lines.append(f"Turn {turn}: no waypoint, search rotation")
        for item in data.get("waypoints", []):
            lines.append(
                f"Turn {turn} / id {item['waypoint_id']}: pixel={item['pixel_xy']} "
                f"bearing={item['bearing_deg']:.1f} deg distance={item['distance_m']:.2f} m "
                f"geodesic={item['geodesic_m']:.2f} m"
            )
    return _text_canvas("Waypoint geometry", lines)


def _result_summary(result) -> np.ndarray:
    metrics = result.get("metrics", {})
    lines = [
        f"Episode: {result.get('episode_id')}",
        f"Instruction: {result.get('instruction')}",
        f"Reasoning turns: {result.get('reasoning_turns')}",
        f"Total discrete actions: {result.get('total_actions')}",
        f"Success: {metrics.get('success')}",
        f"SPL: {metrics.get('spl')}",
        f"Distance to goal: {metrics.get('distance_to_goal')}",
        f"Oracle success: {metrics.get('oracle_success')}",
    ]
    return _text_canvas("AgentNav result", lines)


def _text_canvas(title: str, lines: Iterable[str]) -> np.ndarray:
    canvas = np.full((620, 1200, 3), 246, dtype=np.uint8)
    cv2.putText(canvas, title, (35, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (22, 22, 22), 2, cv2.LINE_AA)
    y = 105
    for line in lines:
        words = str(line).split()
        current = ""
        for word in words:
            proposal = (current + " " + word).strip()
            if cv2.getTextSize(proposal, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1)[0][0] > 1120:
                cv2.putText(canvas, current, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (45, 45, 45), 1, cv2.LINE_AA)
                y += 34; current = word
            else:
                current = proposal
        cv2.putText(canvas, current, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (45, 45, 45), 1, cv2.LINE_AA)
        y += 42
    return canvas


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))
