# AgentNav

Action-chunk VLN in Habitat. One inference per turn: uniformly sampled history
frames plus the annotated current view go in, a continuous motion chunk comes
out, and a depth-guarded executor runs it.

## Task contract

| | |
|---|---|
| Model | whatever `~/.codex/config.toml` selects (currently `gpt-5.6-sol`) |
| History | **every** atomic frame stored, **8** uniformly sampled per call, plain RGB |
| Current view | RGB with numbered waypoint circles drawn on it |
| HFOV | 90 |
| Reasoning turns | 10 max |
| Output | `forward <25-100>` cm, `turn left/right <15-90>` deg, `stop` |
| Chunk length | unbounded; the model decides how far to commit |

## Loop

```
observe()
  ├─ RGB / depth
  ├─ LocalMapper.update()          -> 160x160 robot-centric map {FREE, OCCUPIED, UNKNOWN}
  ├─ sample_waypoints()            -> ray-march over 11 bearings + frontier points
  │                                   occlusion-tested against depth
  └─ annotate_rgb()                -> numbered circles (green = reachable, orange = frontier)

VLNSelector.select()
  ├─ prompt: vln_real PROMPT_TEMPLATE + the motion vocabulary
  ├─ 8 history frames (plain RGB, oldest first)
  └─ current annotated view        -> "<answer>turn left 30, forward 75</answer>"

actions.parse_chunk()              -> [turn left 30, forward 75]  (magnitudes clamped)
actions.to_primitives()            -> [2, 2, 1, 1, 1]             (0.25 m / 15 deg steps)

ChunkExecutor.execute()
  └─ runs each primitive; FORWARD is skipped and the chunk abandoned when depth
     shows the corridor blocked, so the model sees the obstacle next turn
```

The circles are **reference only** — the model answers with motion, not with an
id. History frames stay unannotated: an id means a different 3D point in every
frame, so annotating history invites the model to conflate ids across time.

## Why not Qwen-RobotNav's interface

Qwen-RobotNav emits a numeric waypoint trajectory — 8 x (x, y, theta), 24 dims —
from a trained 4-layer MLP regression head on the LLM's final hidden state. That
needs a fine-tuned head and is unavailable to a zero-shot API model. Its
`"Move forward 2.0 meters"` strings are *inputs*, not outputs, and that is the
shape adopted here. The text chunk format itself matches `vln_real/actions.py`,
so the simulator agent and the real-robot agent consume model output identically.

## No navmesh in the executor

`ChunkExecutor` uses no `snap_point`, no `is_navigable`, no `find_path`, no
`ShortestPathFollower`. It runs the model's primitives and guards FORWARD with a
depth-derived corridor clearance. The topdown map fed to the candidate sampler is
still a navmesh slice, matching AgentVLN; that is the remaining privileged
component and is worth stating in any write-up.

## Usage

```bash
cd /media/mldadmin/home/s125mdg38_06/AgentNav
conda run -n streamvln python run.py --episode-id 412
conda run -n streamvln python run.py --episode-id 1378 --max-turns 10
conda run -n streamvln python run.py --episode-id 412 --dry-run      # one call, no movement
```

Per-turn artefacts land in `outputs/episode_<id>/`: `turn_NN_rgb.png` (what the
model saw), `turn_NN_local_map.png`, `turn_NN_depth.png`, `turn_NN_dashboard.png`,
plus `trace.json` with every raw model answer and the primitives it expanded to.

## Results

| episode | instruction | turns | calls | actions | SR | SPL | NE |
|---|---|---|---|---|---|---|---|
| 412 | Walk past altar book stands. Wait under wooden rafter. | 1 | 1 | 10 | **1.0** | **1.0** | 2.52 |

Verified against habitat's own metrics (`success_distance = 3.0 m`); success
requires the agent to call STOP itself.
