# GPT-VLN

GPT-VLN navigates indoor Habitat scenes from language instructions and four
simultaneous RGB views. The runtime records measured robot motion and keeps
images and IDs of visited nodes for return. The navigation model receives its
own measured pose, not a map image or room labels.

## Requirements

Use a Python environment with Habitat-Sim, Habitat-Lab, NumPy, OpenCV, Pillow,
and the dependencies in `requirements.txt`. Supply R2R scene assets and a task
registration JSON separately. Configure Claude Agent SDK authentication for model requests. The Codex CLI
can also be selected with `--provider codex`. Set `VLN_WORKSPACE` if the scene assets are outside the parent of this
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
  --provider claude --decisions 20 --action-horizon 4_to_3 --gpu 0
```

The default asks the model to predict four ordered commands and executes the
first three before observing again. Forward commands use 25, 50, 75, or 100 cm;
turns use 15-degree increments up to 90 degrees. The same range is available
when ascending or descending stairs. STOP uses an empty action list. The chosen
actions are checked after every 25 cm primitive and can be interrupted when
actual motion conflicts with the command. To continue a saved
run, use the same arguments with a larger cumulative `--chunk-end`; source and
registration hashes must remain unchanged.

## Flow

1. Habitat supplies four RGB views and paired depth for internal local checks.
   No map image or map geometry is sent to the navigation model.
2. The model receives the instruction, current views, measured own pose,
   previous execution, and IDs and connections of visited nodes. Claude can
   retrieve a saved historical panorama when a route comparison needs it.
3. The model chooses movement or STOP. The runtime validates the response,
   executes a bounded prefix, and saves requested and actual actions.
4. After each ten completed decisions, an independent visual reviewer checks
   route evidence before the next decision. It gives advice; the navigator
   retains movement and STOP authority.
5. Requests, images, action logs, review records, and the final
   result are saved under `outputs/`.

The map module retains its interface; RGB-D map accumulation and rendering are
disabled. Measured pose and recorded visited-node connections remain active.

## Tests

```bash
python -m unittest discover -s tests
```
