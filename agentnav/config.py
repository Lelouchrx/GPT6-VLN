from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


WORKSPACE = Path("/media/mldadmin/home/s125mdg38_06")


@dataclass(frozen=True)
class MapConfig:
    meters_per_pixel: float = 0.05
    local_size_m: float = 8.0
    fog_visibility_m: float = 6.0
    # R2R-CE aisles measure ~0.55 m across; 0.22 rejected the agent's own cell.
    robot_radius_m: float = 0.18
    # fog_of_war stops *before* the blocking cell, so wall surfaces are never
    # revealed. Dilating the revealed mask recovers them as OCCUPIED.
    obstacle_band_px: int = 3


@dataclass(frozen=True)
class SamplingConfig:
    # Wider fan: the only navigable direction is often beyond +-60 deg.
    bearings_deg: Tuple[float, ...] = (-90, -70, -50, -30, -15, 0, 15, 30, 50, 70, 90)
    # Ray-march bounds instead of a hard "forward >= 1.5 m" gate.
    min_advance_m: float = 0.5
    max_range_m: float = 3.5
    max_candidates: int = 8
    min_separation_m: float = 0.5
    # Frontier candidates, AgentVLN style.
    use_frontiers: bool = True
    # A frontier is a boundary line, not a region: measure it in metres of
    # passable width. Anything narrower than this is contour noise.
    frontier_min_length_m: float = 0.20
    # Marker is drawn raised, but the occlusion test uses the ground point.
    marker_height_m: float = 0.40
    # Ceiling for the adaptive lift that keeps near markers inside the frame.
    max_marker_height_m: float = 1.10
    ground_projection: bool = True
    # Occlusion test: a low percentile is sensitive to a near occluder, where a
    # median gets dragged up by background pixels in the same patch.
    depth_patch_radius: int = 4
    depth_percentile: float = 25.0
    depth_base_tol_m: float = 0.25
    depth_rel_tol: float = 0.10


@dataclass(frozen=True)
class ExecutorConfig:
    """Runs the model's primitives. No navmesh, no shortest-path follower."""

    forward_step_m: float = 0.25
    turn_angle_deg: float = 15.0
    # Depth-based collision guard.
    collision_clearance_m: float = 0.32
    collision_band: Tuple[float, float] = (0.30, 0.80)
    collision_min_valid_px: int = 30
    stall_epsilon_m: float = 0.02
    max_stalls: int = 2


@dataclass(frozen=True)
class HistoryConfig:
    """Every atomic frame is kept; 8 are uniformly sampled at read time."""

    max_frames: int = 8
    strategy: str = "all_uniform"
    # History frames stay unannotated: an id means a different point per frame.
    annotate: bool = False
    include_text_history: bool = True


@dataclass(frozen=True)
class BudgetConfig:
    max_env_actions: int = 500
    # One inference per turn; the model commits an unbounded chunk each time.
    max_agent_turns: int = 10


@dataclass(frozen=True)
class SelectorConfig:
    """Continuous action-chunk contract. Chunk length is unbounded."""

    forward_cm_range: Tuple[float, float] = (25.0, 100.0)
    turn_deg_range: Tuple[float, float] = (15.0, 90.0)
    # Habitat's primitive granularity, used to quantise each magnitude.
    forward_step_m: float = 0.25
    turn_angle_deg: float = 15.0
    max_output_tokens: int = 2048


@dataclass(frozen=True)
class AgentConfig:
    width: int = 640
    height: int = 480
    hfov_deg: int = 90
    depth_max_m: float = 10.0
    camera_height_m: float = 1.25
    # Read from habitat's depth_sensor.normalize_depth, never guessed.
    depth_normalized: bool = True
    map: MapConfig = field(default_factory=MapConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)
    history: HistoryConfig = field(default_factory=HistoryConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    selector: SelectorConfig = field(default_factory=SelectorConfig)
    habitat_config: Path = WORKSPACE / "StreamVLN/config/vln_r2r.yaml"
    dataset_path: Path = WORKSPACE / "StreamVLN/data/datasets/r2r/{split}/{split}.json.gz"
    scenes_dir: Path = WORKSPACE / "StreamVLN/data/scene_datasets"
    codex_config: Path = WORKSPACE / ".codex/config.toml"
    codex_auth: Path = WORKSPACE / ".codex/auth.json"
    output_dir: Path = WORKSPACE / "AgentNav/outputs"
