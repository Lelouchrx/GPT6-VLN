"""Agentic GPT-VLN evaluation: the model drives navigation through skills.

Unlike eval.py (one parsed text command per turn) and eval_sdk.py (tool-use for
bookkeeping only, movement still parsed from text), here the model itself calls
skills, and the episode ends when it calls stop or runs out of budget.

Three-column ablation (--skill-mode):
  minimal   observe + move + stop; extra skills are not in the interface
  optional  all skills visible; the model chooses if and when to use them
  forced    all skills visible; prompt requires refine / note / recall+go_back

This process owns the Habitat env and every skill implementation
(gpt_vln/skill_server.py); the model runs in gpt_vln/skill_bridge.py under the
claude-agent-sdk env (py3.10+) and reaches the skills over a Unix socket,
because claude_agent_sdk cannot be installed into streamvln's py3.9.

Usage:
  conda activate streamvln
  python eval_skills.py --episode-id 5 --max-turns 10 --skill-mode optional
  python eval_skills.py --episode-id 4,5,22,23,32,43,44,60,84,97 \\
      --skill-mode all --output outputs/skills_ablation --gpu-ids 3
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from eval import (
    STOP, History, MapProcessor, Settings, align_start_heading, create_env,
    jsonable, select_episodes,
)
from eval import parser as eval_parser
from gpt_vln.skill_modes import (
    MODES, comparison_row, count_skills, mode_config, resolve_mode,
    summarize_records as summarize_skill_records, usage_flags,
)
from gpt_vln.skill_server import EpisodeState, SkillServer, serve
from gpt_vln.visualization import save_route_topmap

DEFAULT_MAX_MOVES = 10


def resolve_sdk_python(args):
    if args.sdk_python:
        return args.sdk_python
    return str(Path(sys.prefix).parent / args.sdk_env / "bin" / "python")


def run_episode_skills(env, args, settings, normalized, sdk_python):
    obs = env.reset()
    episode = env.current_episode
    out = args.output / f"episode_{episode.episode_id}"
    out.mkdir(parents=True, exist_ok=True)

    alignment = align_start_heading(env, episode) if args.align_start_heading else None
    if alignment is not None:
        state = env.sim.get_agent_state()
        obs = env.sim.get_observations_at(state.position, state.rotation,
                                           keep_agent_at_new_pose=True)

    processor = MapProcessor(env.sim, settings, normalized)
    processor.update(obs["depth"])
    history = History(args.history_strategy, args.history_frames)
    max_moves = args.max_turns if args.max_turns > 0 else DEFAULT_MAX_MOVES
    episode_state = EpisodeState(episode.instruction.instruction_text,
                                 max_moves, args.max_actions)
    server = SkillServer(env, processor, history, settings, episode_state, out, obs,
                         frontier_input=args.frontier_input)

    # Not under `out`: AF_UNIX paths are capped at ~108 bytes and a deep output
    # directory silently blows past it (OSError: AF_UNIX path too long).
    socket_dir = tempfile.mkdtemp(prefix="vln-skill-")
    socket_path = os.path.join(socket_dir, "s.sock")
    request = {
        "socket_path": socket_path,
        "instruction": episode.instruction.instruction_text,
        "max_moves": max_moves,
        "model": args.model,
        "claude_command": args.claude_command,
        "max_agent_turns": args.max_agent_turns,
        "skill_mode": args.skill_mode,
    }

    bridge = Path(__file__).resolve().parent / "gpt_vln" / "skill_bridge.py"
    env_vars = os.environ.copy()
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"):
        env_vars.pop(var, None)

    command = ([sdk_python, str(bridge)] if args.use_api
               else [sys.executable, str(args.stub_bridge)])
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, env=env_vars)
    process.stdin.write(json.dumps(request))
    process.stdin.close()
    process.stdin = None  # else communicate() below re-flushes a closed pipe

    serve(socket_path, server, process, deadline_s=args.episode_timeout)
    if episode_state.timed_out:
        print(f"[skills] episode {episode.episode_id} hit the "
              f"{args.episode_timeout}s deadline; killing the bridge")
        process.kill()
    try:
        stdout, stderr = process.communicate(timeout=args.api_timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
    shutil.rmtree(socket_dir, ignore_errors=True)
    if process.returncode != 0:
        print(f"[skills] bridge exited {process.returncode}: {stderr[-2000:]}")

    try:
        bridge_result = json.loads(stdout) if stdout.strip() else {}
    except json.JSONDecodeError:
        bridge_result = {"final_text": "", "parse_error": stdout[-2000:]}

    obs = server.obs
    if not env.episode_over:
        env.step(STOP)

    raw_dir = out / "raw_action_frames"
    raw_dir.mkdir(parents=True, exist_ok=True)
    action_frames = []
    for index, (action, frame) in enumerate(server.frames):
        path = raw_dir / f"action_{index:04d}.png"
        cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        action_frames.append({"frame_index": index, "action": int(action),
                              "path": str(path.relative_to(out))})
    (raw_dir / "index.json").write_text(json.dumps(action_frames, indent=2))

    if args.visualize:
        positions = [np.asarray(p["position"]) for p in episode_state.checkpoints]
        positions.append(np.asarray(env.sim.get_agent_state().position))
        try:
            save_route_topmap(out / "top_map_gt_pred.png", env.sim, episode, positions, [])
        except Exception as exc:
            print(f"[skills] top map skipped: {exc}")

    config = mode_config(args.skill_mode)
    skill_counts = count_skills(episode_state.transcript)
    result = {
        "episode_id": str(episode.episode_id),
        "scene_id": episode.scene_id,
        "instruction": episode.instruction.instruction_text,
        "start_alignment": alignment,
        "protocol": {
            "split": args.split,
            "reference_path_start_alignment": alignment is not None,
            "rgb_only_policy": True,
            "skill_mode": config["mode"],
            "available_skills": list(config["tools"]),
            "movement_budget": max_moves,
            "atomic_action_budget": args.max_actions,
        },
        "moves_used": episode_state.moves_used,
        "total_actions": episode_state.actions_used,
        "stopped_by_model": episode_state.stopped,
        "timed_out": episode_state.timed_out,
        "stop_evidence": episode_state.stop_evidence,
        "subtasks": episode_state.subtasks,
        "final_stage": episode_state.stage + 1 if episode_state.subtasks else None,
        "avoid_list": episode_state.avoid,
        "skill_call_count": len(episode_state.transcript),
        "skill_counts": skill_counts,
        "used_extra_skill": usage_flags(skill_counts),
        "final_text": bridge_result.get("final_text", ""),
        "usage": bridge_result.get("usage", {}),
        "bridge_error": bridge_result.get("bridge_error"),
        "history": {"strategy": args.history_strategy, "stored": len(history.frames)},
        "metrics": jsonable(env.get_metrics()),
    }
    (out / "skill_transcript.json").write_text(
        json.dumps(jsonable(episode_state.transcript), indent=2))
    (out / "progress_notes.json").write_text(
        json.dumps(jsonable(episode_state.progress_notes), indent=2))
    (out / "result.json").write_text(json.dumps(jsonable(result), indent=2))
    return result


def parser():
    p = eval_parser()
    p.add_argument("--sdk-env", default="claude-agent-sdk",
                    help="conda env with claude_agent_sdk installed (Python >=3.10)")
    p.add_argument("--sdk-python", default=None,
                    help="explicit interpreter path for that env")
    p.add_argument("--max-agent-turns", type=int, default=80,
                    help="hard cap on SDK agent turns per episode (runaway guard)")
    p.add_argument("--episode-timeout", type=float, default=1200,
                    help="wall-clock seconds before a hung bridge is killed")
    p.add_argument("--stub-bridge", default="tests/stub_skill_bridge.py",
                    help="with --no-use-api: scripted bridge used to exercise the "
                         "skill server without spending API credit")
    p.add_argument("--skill-mode", default="optional", choices=(*MODES, "all"),
                    help="minimal / optional / forced ablation column, or all three")
    p.set_defaults(provider="claude_sdk", model="opus", api_timeout=1800)
    return p


def load_episode_results(output):
    records = []
    for path in sorted(Path(output).glob("episode_*/result.json")):
        try:
            records.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[skills] ignoring unreadable result {path}: {exc}")
    return list({r["episode_id"]: r for r in records}.values())


def summarize_records(records, args):
    summary = summarize_skill_records(records)
    summary["protocol"] = {
        "split": args.split,
        "reference_path_start_alignment": bool(args.align_start_heading),
        "skill_mode": args.skill_mode,
    }
    return summary


def write_comparison(root):
    rows = []
    for mode in MODES:
        path = Path(root) / mode / "summary.json"
        if not path.is_file():
            continue
        try:
            rows.append(comparison_row(mode, json.loads(path.read_text())))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[skills] ignoring unreadable summary {path}: {exc}")
    payload = {"columns": rows}
    (Path(root) / "comparison.json").write_text(json.dumps(payload, indent=2))
    return payload


def print_comparison(payload):
    rows = payload.get("columns") or []
    if not rows:
        return
    header = f"{'mode':<10} {'n':>3} {'SR':>6} {'SPL':>6} {'extra':>6} {'refine':>7} {'go_back':>8} {'nav_to':>6}"
    print("[skills] three-column ablation")
    print(header)
    for row in rows:
        print(f"{row['mode']:<10} {row.get('episodes') or 0:3d} "
              f"{_fmt(row.get('success')):>6} {_fmt(row.get('spl')):>6} "
              f"{_fmt(row.get('used_any_extra')):>6} {_fmt(row.get('used_refine')):>7} "
              f"{_fmt(row.get('used_go_back')):>8} {_fmt(row.get('used_navigate_to')):>6}")


def _fmt(value):
    return "   —" if value is None else f"{float(value):.2f}"


def run_mode(env, args, settings, normalized, sdk_python, episodes):
    args.output.mkdir(parents=True, exist_ok=True)
    pending = list(episodes)
    if args.skip_existing:
        pending = [episode for episode in pending
                   if not (args.output / f"episode_{episode.episode_id}" / "result.json").exists()]
    if not pending:
        print(f"[skills] {args.skill_mode}: all {len(episodes)} episodes already done")
    else:
        env.episodes = pending
        for i, _ in enumerate(pending):
            result = run_episode_skills(env, args, settings, normalized, sdk_python)
            with (args.output / "results.jsonl").open("a") as f:
                f.write(json.dumps(result) + "\n")
            extra = ",".join(skill for skill, used in (result.get("used_extra_skill") or {}).items()
                             if used) or "-"
            print(f"[{args.skill_mode} {i + 1}/{len(pending)}] episode={result['episode_id']} "
                  f"moves={result['moves_used']} extra={extra} "
                  f"stopped={result['stopped_by_model']} metrics={result['metrics']}")
    records = load_episode_results(args.output)
    summary = summarize_records(records, args)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    args = parser().parse_args()
    args.output = args.output.resolve()
    if args.force:
        args.skip_existing = False
    settings = Settings(hfov=args.hfov)
    sdk_python = resolve_sdk_python(args)
    if args.use_api and not Path(sdk_python).exists():
        raise RuntimeError(f"{sdk_python} not found; create the env or pass --sdk-python")

    gpu_ids = [int(g) for g in args.gpu_ids.split(",") if g.strip() != ""]
    if not gpu_ids:
        raise ValueError("--gpu-ids must contain at least one device id")
    modes = list(MODES) if args.skill_mode == "all" else [resolve_mode(args.skill_mode)]
    base_output = args.output
    env, normalized = create_env(args, settings, gpu_ids[0])
    try:
        episodes = select_episodes(list(env.episodes), args)
        if not episodes:
            raise ValueError(f"no episode matched {args.episode_id}")
        for mode in modes:
            args.skill_mode = mode
            args.output = base_output / mode if len(modes) > 1 else base_output
            run_mode(env, args, settings, normalized, sdk_python, episodes)
    finally:
        env.close()

    if len(modes) > 1:
        print_comparison(write_comparison(base_output))


if __name__ == "__main__":
    main()
