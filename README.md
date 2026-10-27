# GPT-VLN

GPT-VLN navigates indoor Habitat scenes from language instructions and four
simultaneous RGB views. The runtime builds an observed RGB-D geometry map,
records measured robot motion, and keeps images of visited nodes for return.
Node names and room labels are omitted from the navigation model input.

## Requirements

Use a Python environment with Habitat-Sim, Habitat-Lab, NumPy, OpenCV, Pillow,
and the dependencies in `requirements.txt`. Supply R2R scene assets and a task
registration JSON separately. Install and configure the Codex CLI for model
requests. Set `VLN_WORKSPACE` if the scene assets are outside the parent of this
repository. Set `CODEX_HOME`, `NAV_CODEX_BIN`, and `NAV_MODEL_PYTHON` when their
default discovery paths do not apply.

The registration format is shown in `config/tasks.example.json`. Replace its
placeholder episode ID and instruction with a task from the configured dataset.

## Run

Check the selected task and action contract without starting Habitat:

```bash
python eval.py --registration /path/to/tasks.json --episode EPISODE_ID --preflight
```

Run navigation:

```bash
python eval.py --registration /path/to/tasks.json --episode EPISODE_ID \
  --decisions 20 --action-horizon 3_to_2 --gpu 0
```

`3_to_2` predicts three ordered commands and executes the first two.
`4_to_3` predicts four and executes the first three. STOP uses an empty action
list. The executed prefix is checked after every primitive and can be
interrupted when actual motion conflicts with the command. To continue a saved
run, use the same arguments with a larger cumulative `--chunk-end`; source and
registration hashes must remain unchanged.

## Flow

1. Habitat supplies four RGB views and paired depth. The map contains observed
   geometry only; unknown space is not treated as free.
2. The model receives the instruction, current views, measured own pose,
   previous execution, and recorded visited-node connections.
3. The model chooses movement or STOP. The runtime validates the response,
   executes a bounded prefix, and saves requested and actual actions.
4. After each ten completed decisions, an independent visual reviewer checks
   route evidence before the next decision. It gives advice; the navigator
   retains movement and STOP authority.
5. Requests, images, map snapshots, action logs, review records, and the final
   result are saved under `outputs/`.

## Tests

```bash
python -m unittest discover -s tests
```
