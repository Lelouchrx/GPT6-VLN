from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import AgentConfig
from .types import CameraIntrinsics, Waypoint


UNKNOWN = np.uint8(127)
FREE = np.uint8(255)
OCCUPIED = np.uint8(0)


@dataclass
class MapState:
    full_map: np.ndarray
    fog_mask: np.ndarray
    observed_mask: np.ndarray
    local_map: np.ndarray
    agent_grid_rc: Tuple[int, int]


def depth_to_meters(depth: np.ndarray, max_depth_m: float, normalized: bool = True) -> np.ndarray:
    """Convert a Habitat depth frame to metres.

    `normalized` mirrors habitat's depth_sensor.normalize_depth (default True),
    read from the config rather than guessed from the pixel range: in a narrow
    aisle every pixel can legitimately be under 1 m.
    """
    value = np.asarray(depth, dtype=np.float32)
    if value.ndim == 3 and value.shape[-1] == 1:
        value = value[..., 0]
    if normalized:
        value = value * float(max_depth_m)
    return value


# Back-compat for callers/tests that predate the explicit flag.
def _depth_to_meters(depth: np.ndarray, max_depth_m: float) -> np.ndarray:
    value = np.asarray(depth, dtype=np.float32)
    if value.ndim == 3 and value.shape[-1] == 1:
        value = value[..., 0]
    finite = value[np.isfinite(value)]
    if finite.size and float(finite.max()) <= 1.0 + 1e-6:
        value = value * max_depth_m
    return value


def _query_depth(depth: np.ndarray, pixel: Tuple[int, int], radius: int = 3) -> Optional[float]:
    x, y = pixel
    patch = depth[max(0, y - radius):y + radius + 1, max(0, x - radius):x + radius + 1]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    return float(np.median(valid)) if valid.size else None


def occlusion_depth(
    depth_m: np.ndarray, pixel: Tuple[int, int], radius: int, percentile: float
) -> Optional[float]:
    """Depth of the nearest surface actually covering the marker.

    Two statistics, whichever is nearer:
      * the median of the 3x3 right under the marker - noise-robust, and it
        fires when an occluder sits exactly on the marker even though that is
        a small share of the wider patch;
      * a low percentile over the full patch - absorbs a few pixels of
        projection error without letting background pixels drag the estimate up
        the way a plain patch median does.
    """
    x, y = pixel
    patch = depth_m[max(0, y - radius):y + radius + 1, max(0, x - radius):x + radius + 1]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size == 0:
        return None
    wide = float(np.percentile(valid, percentile))
    core = depth_m[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
    core_valid = core[np.isfinite(core) & (core > 0)]
    if core_valid.size == 0:
        return wide
    return min(wide, float(np.median(core_valid)))


# Kept under the old name for callers that only want the percentile branch.
def depth_percentile(
    depth_m: np.ndarray, pixel: Tuple[int, int], radius: int, percentile: float
) -> Optional[float]:
    return occlusion_depth(depth_m, pixel, radius, percentile)


class LocalMapper:
    def __init__(self, sim, config: AgentConfig, depth_normalized: bool = True):
        self.sim = sim
        self.config = config
        self.depth_normalized = depth_normalized
        self.intrinsics = CameraIntrinsics.from_hfov(config.width, config.height, config.hfov_deg)
        self.full_map: Optional[np.ndarray] = None
        self.fog_mask: Optional[np.ndarray] = None
        self.map_height: Optional[float] = None

    def reset(self) -> None:
        self.full_map = None
        self.fog_mask = None
        self.map_height = None

    # -- map -------------------------------------------------------------
    def update(self) -> MapState:
        from habitat.utils.geometry_utils import quaternion_rotate_vector
        from habitat.utils.visualizations import fog_of_war, maps

        state = self.sim.get_agent_state()
        height = float(state.position[1])
        if self.full_map is None or self.map_height is None or abs(height - self.map_height) > 0.5:
            raw = maps.get_topdown_map(
                self.sim.pathfinder,
                height,
                draw_border=False,
                meters_per_pixel=self.config.map.meters_per_pixel,
            )
            self.full_map = (raw > 0).astype(np.uint8)
            self.fog_mask = np.zeros_like(self.full_map, dtype=np.uint8)
            self.map_height = height

        row, col = maps.to_grid(state.position[2], state.position[0], self.full_map.shape, sim=self.sim)
        forward_world = quaternion_rotate_vector(state.rotation, np.array([0.0, 0.0, -1.0]))
        # fog_of_war walks [row, col] along [cos(angle), sin(angle)], and rows
        # index world z while cols index world x.
        angle = float(np.arctan2(forward_world[0], forward_world[2]))
        max_line = self.config.map.fog_visibility_m / self.config.map.meters_per_pixel
        self.fog_mask = fog_of_war.reveal_fog_of_war(
            self.full_map,
            self.fog_mask,
            np.asarray([row, col], dtype=np.int64),
            angle,
            fov=float(self.config.hfov_deg),
            max_line_len=float(max_line),
        )
        observed = self._observed_mask(self.fog_mask)
        local = self._robot_centric_crop(state, observed)
        return MapState(self.full_map.copy(), self.fog_mask.copy(), observed, local, (row, col))

    def _observed_mask(self, fog_mask: np.ndarray) -> np.ndarray:
        """Grow the revealed set so the blocking cells count as observed.

        reveal_fog_of_war breaks *before* writing the cell that stopped a ray,
        so `revealed & ~navigable` is empty by construction and OCCUPIED could
        never be produced. One dilation recovers the wall surface.
        """
        band = max(1, int(self.config.map.obstacle_band_px))
        kernel = np.ones((3, 3), np.uint8)
        return cv2.dilate((fog_mask > 0).astype(np.uint8), kernel, iterations=band)

    def _robot_centric_crop(self, state, observed: np.ndarray) -> np.ndarray:
        from habitat.utils.geometry_utils import quaternion_rotate_vector

        assert self.full_map is not None
        cfg = self.config.map
        cells = int(round(cfg.local_size_m / cfg.meters_per_pixel))
        center = (cells - 1) / 2.0
        rows, cols = np.indices((cells, cells), dtype=np.float64)
        local_right = (cols - center) * cfg.meters_per_pixel
        local_forward = (center - rows) * cfg.meters_per_pixel

        right_vector = quaternion_rotate_vector(state.rotation, np.array([1.0, 0.0, 0.0]))
        forward_vector = quaternion_rotate_vector(state.rotation, np.array([0.0, 0.0, -1.0]))
        world_x = state.position[0] + local_right * right_vector[0] + local_forward * forward_vector[0]
        world_z = state.position[2] + local_right * right_vector[2] + local_forward * forward_vector[2]

        lower, upper = self.sim.pathfinder.get_bounds()
        row_scale = (upper[2] - lower[2]) / self.full_map.shape[0]
        col_scale = (upper[0] - lower[0]) / self.full_map.shape[1]
        map_rows = np.floor((world_z - lower[2]) / row_scale).astype(np.int32)
        map_cols = np.floor((world_x - lower[0]) / col_scale).astype(np.int32)
        valid = (
            (map_rows >= 0) & (map_rows < self.full_map.shape[0])
            & (map_cols >= 0) & (map_cols < self.full_map.shape[1])
        )
        local = np.full((cells, cells), UNKNOWN, dtype=np.uint8)
        seen = np.zeros_like(valid)
        seen[valid] = observed[map_rows[valid], map_cols[valid]] > 0
        navigable = np.zeros_like(valid)
        navigable[seen] = self.full_map[map_rows[seen], map_cols[seen]] > 0
        local[seen & navigable] = FREE
        local[seen & ~navigable] = OCCUPIED
        return local

    # -- candidates ------------------------------------------------------
    def sample_waypoints(
        self, local_map: np.ndarray, depth: np.ndarray
    ) -> Tuple[List[Waypoint], Dict[str, object]]:
        state = self.sim.get_agent_state()
        sensor_state = state.sensor_states.get("rgb") or state.sensor_states.get("rgb_sensor")
        if sensor_state is None:
            raise RuntimeError(f"RGB sensor state missing: {list(state.sensor_states)}")
        depth_m = depth_to_meters(depth, self.config.depth_max_m, self.depth_normalized)
        cfg = self.config.sampling

        clearance = self._clearance(local_map)
        samples = self._ray_march_samples(local_map, clearance)
        frontier_samples = self._frontier_samples(local_map, clearance) if cfg.use_frontiers else []
        diagnostics: Dict[str, object] = {
            "bearings": len(cfg.bearings_deg),
            "ray_samples": len(samples),
            "frontier_samples": len(frontier_samples),
            "merged": 0,
            "occluded": 0,
            "off_screen": 0,
            "off_screen_bearings": [],
            "kept": 0,
        }

        merged = self._merge(samples + frontier_samples, cfg.min_separation_m)
        diagnostics["merged"] = len(merged)

        candidates: List[Waypoint] = []
        for right, forward, bearing_deg, distance, source in merged:
            world = self._local_to_world(state, right, forward)
            ground = world.copy()
            pixel, camera_z = self._project(world, sensor_state, raise_marker=True)
            if pixel is None:
                # Beyond the camera FOV. Worth remembering: the only navigable
                # direction out of an aisle is often at +-90 deg, which a 90 deg
                # HFOV can never show, so the agent must turn toward it.
                diagnostics["off_screen"] = int(diagnostics["off_screen"]) + 1
                diagnostics["off_screen_bearings"].append(round(float(bearing_deg), 1))
                continue
            # Occlusion test uses the ground point's depth, not the raised
            # marker's: a marker lifted 0.40 m lands on a seat back and would
            # otherwise be discarded as hidden.
            # Compare like with like: measure depth at the pixel whose expected
            # depth we use. Prefer the ground point (AgentVLN does the same),
            # and fall back to the marker when the ground point is out of frame.
            ground_pixel, ground_z = self._project(ground, sensor_state, raise_marker=False)
            if cfg.ground_projection and ground_pixel is not None:
                probe_pixel, expected = ground_pixel, ground_z
            else:
                probe_pixel, expected = pixel, camera_z
            measured = occlusion_depth(depth_m, probe_pixel, cfg.depth_patch_radius, cfg.depth_percentile)
            tolerance = cfg.depth_base_tol_m + cfg.depth_rel_tol * max(0.0, expected)
            if measured is not None and measured + tolerance < expected:
                diagnostics["occluded"] = int(diagnostics["occluded"]) + 1
                continue
            candidates.append(Waypoint(
                waypoint_id=len(candidates) + 1,
                world_xyz=tuple(float(v) for v in world),
                pixel_xy=pixel,
                local_right_m=float(right),
                local_forward_m=float(forward),
                distance_m=float(distance),
                bearing_deg=float(bearing_deg),
                geodesic_m=float(distance),
                depth_m=measured,
                source=source,
            ))
            if len(candidates) >= cfg.max_candidates:
                break
        diagnostics["kept"] = len(candidates)
        return candidates, diagnostics

    def _clearance(self, local_map: np.ndarray) -> np.ndarray:
        """Distance to anything not known-free. Conservative: used for driving."""
        return cv2.distanceTransform((local_map == FREE).astype(np.uint8), cv2.DIST_L2, 5)

    def _obstacle_clearance(self, local_map: np.ndarray) -> np.ndarray:
        """Distance to a known wall. Unexplored space is not an obstacle.

        A frontier sits against unexplored space by definition, so measuring its
        clearance on the FREE mask always returns ~1 px and the gate wipes out
        every frontier. Only OCCUPIED should count here.
        """
        return cv2.distanceTransform((local_map != OCCUPIED).astype(np.uint8), cv2.DIST_L2, 5)

    def _required_clearance_px(self, clearance: np.ndarray) -> float:
        """Clearance gate that can never reject the cell the agent stands on.

        Measured on R2R-CE episode 412 the agent's own cell has 0.20 m of
        clearance while robot_radius_m was 0.22, so every candidate died.
        """
        cells = clearance.shape[0]
        center = int(round((cells - 1) / 2.0))
        here = float(clearance[center, center])
        wanted = self.config.map.robot_radius_m / self.config.map.meters_per_pixel
        return min(wanted, max(1.0, here))

    def _ray_march_samples(self, local_map: np.ndarray, clearance: np.ndarray):
        """Walk outward along each bearing, keep the farthest cell still free."""
        cfg = self.config.sampling
        mpp = self.config.map.meters_per_pixel
        cells = local_map.shape[0]
        center = (cells - 1) / 2.0
        needed = self._required_clearance_px(clearance)
        samples = []
        for bearing_deg in cfg.bearings_deg:
            angle = np.deg2rad(bearing_deg)
            best = None
            steps = int(cfg.max_range_m / mpp)
            for index in range(1, steps + 1):
                distance = index * mpp
                right = distance * np.sin(angle)
                forward = distance * np.cos(angle)
                row = int(round(center - forward / mpp))
                col = int(round(center + right / mpp))
                if not (0 <= row < cells and 0 <= col < cells):
                    break
                if local_map[row, col] != FREE or clearance[row, col] < needed:
                    break
                if distance >= cfg.min_advance_m:
                    best = (float(right), float(forward), float(bearing_deg), float(distance))
            if best is not None:
                samples.append(best + ("bearing",))
        return samples

    def _frontier_samples(self, local_map: np.ndarray, clearance: np.ndarray):
        """AgentVLN-style frontiers: free cells that border unexplored space."""
        cfg = self.config.sampling
        mpp = self.config.map.meters_per_pixel
        cells = local_map.shape[0]
        center = (cells - 1) / 2.0
        kernel = np.ones((3, 3), np.uint8)
        free = (local_map == FREE).astype(np.uint8)
        unknown = (local_map == UNKNOWN).astype(np.uint8)
        # No wall-adjacency filter here: once OCCUPIED is populated correctly,
        # almost every free cell in a 0.55 m aisle is within one dilation of a
        # wall, which zeroed the frontier set entirely. The clearance gate below
        # already keeps frontiers off the walls.
        border = free & cv2.dilate(unknown, kernel, iterations=1)
        wall_clearance = self._obstacle_clearance(local_map)
        needed = self._required_clearance_px(clearance)
        border = (border > 0) & (wall_clearance >= needed)
        if not border.any():
            return []
        count, labels = cv2.connectedComponents(border.astype(np.uint8))
        min_cells = max(2, int(round(cfg.frontier_min_length_m / mpp)))
        samples = []
        for index in range(1, count):
            mask = labels == index
            if mask.sum() < min_cells:
                continue
            rows, cols = np.where(mask)
            row, col = float(rows.mean()), float(cols.mean())
            right = (col - center) * mpp
            forward = (center - row) * mpp
            distance = float(np.hypot(right, forward))
            if distance < cfg.min_advance_m or distance > cfg.max_range_m:
                continue
            bearing = float(np.degrees(np.arctan2(right, forward)))
            samples.append((float(right), float(forward), bearing, distance, "frontier"))
        samples.sort(key=lambda item: -item[3])
        return samples

    @staticmethod
    def _merge(samples, min_separation_m: float):
        """Keep frontier points first, then fill in with ray-march points."""
        ordered = [s for s in samples if s[4] == "frontier"] + [s for s in samples if s[4] != "frontier"]
        kept = []
        for sample in ordered:
            if any(np.hypot(sample[0] - old[0], sample[1] - old[1]) < min_separation_m for old in kept):
                continue
            kept.append(sample)
        kept.sort(key=lambda item: item[2])
        return kept

    def _local_to_world(self, state, right: float, forward: float) -> np.ndarray:
        from habitat.utils.geometry_utils import quaternion_rotate_vector

        local = np.array([right, 0.0, -forward])
        return np.asarray(state.position) + quaternion_rotate_vector(state.rotation, local)

    def _camera_local(self, world_point: np.ndarray, sensor_state):
        from habitat.utils.geometry_utils import quaternion_rotate_vector

        local = quaternion_rotate_vector(
            sensor_state.rotation.inverse(), np.asarray(world_point, dtype=np.float64) - sensor_state.position
        )
        return float(local[0]), float(-local[1]), float(-local[2])

    def _project(self, world_point: np.ndarray, sensor_state, raise_marker: bool):
        """Project a point, lifting the marker just enough to stay in frame.

        The camera sits 1.25 m up and the marker was drawn 0.40 m above the
        floor, i.e. 0.85 m below the lens. With a 73.7 deg VFOV that falls off
        the bottom edge for anything nearer than ~1.13 m, so every close
        candidate was silently discarded as off-screen even when it was dead
        ahead. Lifting the marker up the vertical through the same ground point
        keeps it both visible and geometrically honest.
        """
        intr = self.intrinsics
        point = np.asarray(world_point, dtype=np.float64).copy()
        x, y, z = self._camera_local(point, sensor_state)
        if z <= 1e-5:
            return None, z
        u = intr.fx * x / z + intr.cx
        # Horizontal FOV is a real limit: no amount of lifting brings it back.
        if not (0 <= u < intr.width):
            return None, z
        if not raise_marker:
            v = intr.fy * y / z + intr.cy
            if not (0 <= v < intr.height):
                return None, z
            return (int(round(u)), int(round(v))), z

        cfg = self.config.sampling
        margin = 8.0
        v_max = intr.height - 1.0 - margin
        lift = cfg.marker_height_m
        # y is measured downward from the lens, so lifting the marker lowers y.
        needed = y - (v_max - intr.cy) * z / intr.fy
        if needed > lift:
            lift = min(cfg.max_marker_height_m, needed)
        point[1] += lift
        x, y, z = self._camera_local(point, sensor_state)
        if z <= 1e-5:
            return None, z
        u = intr.fx * x / z + intr.cx
        v = intr.fy * y / z + intr.cy
        if not (0 <= u < intr.width and 0 <= v < intr.height):
            return None, z
        return (int(round(u)), int(round(v))), z


# Name kept so existing imports/tests keep working.
PathfinderLocalMapper = LocalMapper
