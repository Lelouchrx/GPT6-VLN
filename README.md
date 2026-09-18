# GPT-VLN

AgentVLN-style waypoint navigation in Habitat, with the local VLM replaced by a
GPT-6 Responses API call. The VLM selects one candidate coordinate; a Habitat
navigation skill converts that target into primitive actions.

## Task contract

| | |
|---|---|
| Model | GPT-6 through an OpenAI-compatible Responses API |
| History | every post-action raw RGB, **8** uniformly sampled per call |
| Current view | RGB annotated with numbered frontier candidates |
| HFOV | configurable; use `HFOV=90` or `HFOV=120` for comparison |
| Reasoning turns | 10 max |
| Output | `<frontiers_coord>(u,v)`, `<target>(u,v)`, `<action>...</action>`, or `STOP` |
| Skill | `ShortestPathFollower` executes a selected world-space waypoint |

## Loop

```
observe()
  ├─ RGB / depth
  ├─ LocalMapper.update()          -> 160x160 robot-centric map {FREE, OCCUPIED, UNKNOWN}
  ├─ ExplorationTargetGenerator    -> explored/unexplored frontier contours
  │                                   with floor, spacing, FOV and depth filters
  └─ annotate_rgb()                -> numbered circles (green = reachable, orange = frontier)

GPT-6 Responses API
  ├─ prompt: instruction + all visible candidate pixel coordinates
  ├─ 8 sampled raw action frames + current annotated RGB
  └─ selects one coordinate, a fallback action sequence, or STOP

pixel_to_world()
  └─ candidate match first; depth back-projection fallback

ShortestPathFollower skill
  └─ repeatedly plans and executes Habitat primitives until the waypoint is reached
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

## Navigation skill

Like AgentVLN, coordinate predictions are matched to a candidate world point or
back-projected with depth, then executed with Habitat's `ShortestPathFollower`.
The fallback `<action>` output executes discrete Habitat actions directly.

## Layout

- `config/vln_r2r.yaml`: the only Habitat task configuration.
- `eval.py`: complete model/history/map/API/Habitat evaluation loop in one file.
- `gpt_vln/habitat_extensions/`: local measures and GT/pred top-map drawing.
- `gpt_vln/visualization.py`: per-turn views and episode route visualization.
- `scripts/run_eval.sh`: evaluation launcher and common experiment parameters.

## Usage

```bash
cd /media/mldadmin/home/s125mdg38_06/GPT-VLN
bash scripts/run_eval.sh
EPISODE_ID=1378 MAX_TURNS=10 bash scripts/run_eval.sh
EPISODE_ID=all NUM_EPISODES=20 bash scripts/run_eval.sh
HISTORY_STRATEGY=recent HISTORY_FRAMES=6 bash scripts/run_eval.sh
HFOV=120 HISTORY_STRATEGY=uniform HISTORY_FRAMES=8 bash scripts/run_eval.sh
USE_API=0 VISUALIZE=1 bash scripts/run_eval.sh  # no-network pipeline smoke test
```

Per-turn artefacts land in `outputs/eval/episode_<id>/`: `turn_NN_input.png` (a
diagnostic contact sheet), `turn_NN_rgb.png` (the annotated visualization),
`turn_NN_local_map.png`, `turn_NN_depth.png`, `turn_NN_dashboard.png`,
`turn_NN_inference.json` with the complete model input/output, and `trace.json`
with the primitives it expanded to. `top_map_gt_pred.png` overlays the green GT
route and red predicted route. Disable image output with `VISUALIZE=0`; JSON
results are still produced.

Every post-action raw RGB is stored in `raw_action_frames/action_NNNN.png`, with
turn/action provenance in `raw_action_frames/index.json`. The top-map draws the
full primitive trajectory and marks every inference endpoint as `T0`, `T1`, etc.

## Results

| episode | instruction | turns | calls | actions | SR | SPL | NE |
|---|---|---|---|---|---|---|---|
| 412 | Walk past altar book stands. Wait under wooden rafter. | 1 | 1 | 10 | **1.0** | **1.0** | 2.52 |

Verified against habitat's own metrics (`success_distance = 3.0 m`); success
requires the agent to call STOP itself.
