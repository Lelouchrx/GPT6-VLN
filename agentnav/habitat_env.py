from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

from .config import AgentConfig, WORKSPACE


def add_vendored_habitat_to_path() -> None:
    root = WORKSPACE / "StreamVLN/deps/habitat-lab"
    for path in (root / "habitat-lab", root / "habitat-baselines", WORKSPACE / "StreamVLN"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def create_env(config: AgentConfig, split: str = "val_unseen") -> Tuple[object, bool]:
    """Returns (env, depth_is_normalized).

    The caller needs the depth convention explicitly: guessing it from the pixel
    range breaks in a narrow corridor where every measurement is under 1 m.
    """
    add_vendored_habitat_to_path()
    import habitat
    from habitat import Env
    from habitat_baselines.config.default import get_config

    from streamvln.habitat_extensions import measures as _measures  # noqa: F401

    habitat_config = get_config(str(config.habitat_config))
    with habitat.config.read_write(habitat_config):
        habitat_config.habitat.dataset.split = split
        habitat_config.habitat.dataset.data_path = str(config.dataset_path)
        habitat_config.habitat.dataset.scenes_dir = str(config.scenes_dir)
        habitat_config.habitat.environment.max_episode_steps = config.budget.max_env_actions
        simulator = habitat_config.habitat.simulator
        simulator.forward_step_size = config.executor.forward_step_m
        simulator.turn_angle = int(config.executor.turn_angle_deg)
        sensors = simulator.agents.main_agent.sim_sensors
        for sensor in (sensors.rgb_sensor, sensors.depth_sensor):
            sensor.width = config.width
            sensor.height = config.height
            sensor.hfov = int(config.hfov_deg)
        sensors.depth_sensor.max_depth = config.depth_max_m
        depth_normalized = bool(getattr(sensors.depth_sensor, "normalize_depth", True))
    return Env(habitat_config), depth_normalized


def select_episode(env, episode_id: str) -> None:
    matches = [episode for episode in env.episodes if str(episode.episode_id) == str(episode_id)]
    if len(matches) != 1:
        raise ValueError(f"episode {episode_id} not found exactly once")
    env.episodes = matches


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
