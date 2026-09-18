from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from agentnav.agent import NavigationAgent
from agentnav.config import AgentConfig
from agentnav.habitat_env import create_env, select_episode
from agentnav.selector import VLNSelector


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AgentNav: action-chunk VLN in Habitat")
    p.add_argument("--episode-id", default="412")
    p.add_argument("--split", default="val_unseen")
    p.add_argument("--model", default=None, help="Defaults to the model in the Codex config")
    p.add_argument("--max-turns", type=int, default=None, help="Reasoning turns (default 10)")
    p.add_argument("--max-actions", type=int, default=None)
    p.add_argument("--history-frames", type=int, default=None)
    p.add_argument("--no-text-history", action="store_true")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true", help="One inference, no movement")
    return p


def build_config(args) -> AgentConfig:
    config = AgentConfig()
    budget = config.budget
    if args.max_turns is not None:
        budget = replace(budget, max_agent_turns=args.max_turns)
    if args.max_actions is not None:
        budget = replace(budget, max_env_actions=args.max_actions)
    history = config.history
    if args.history_frames is not None:
        history = replace(history, max_frames=args.history_frames)
    if args.no_text_history:
        history = replace(history, include_text_history=False)
    return replace(config, budget=budget, history=history,
                   output_dir=args.output or config.output_dir)


def main() -> None:
    args = parser().parse_args()
    config = build_config(args)
    selector = VLNSelector(config.codex_config, config.codex_auth,
                           model=args.model, config=config.selector)
    env, depth_normalized = create_env(config, args.split)
    config = replace(config, depth_normalized=depth_normalized)
    try:
        select_episode(env, args.episode_id)
        agent = NavigationAgent(env, selector, config)
        print(f"model={selector.model}  episode={args.episode_id}  "
              f"turns<={config.budget.max_agent_turns}  hfov={config.hfov_deg}  "
              f"history={config.history.max_frames} of all atomic frames")
        result = agent.run_episode(execute=not args.dry_run)
        print(json.dumps({
            "episode_id": result["episode_id"],
            "instruction": result["instruction"],
            "turns": result["turns"],
            "model_calls": result["model_calls"],
            "total_actions": result["total_actions"],
            "chunks": result["chunks"],
            "metrics": result["metrics"],
            "output": str((config.output_dir / f"episode_{result['episode_id']}").resolve()),
        }, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
