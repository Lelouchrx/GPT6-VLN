# Failure-driven navigation RSI

This directory is independent of the navigation runtime. The starting agent is
the repository version: GPT-6 Astra, four current views, up to eight uniformly
sampled historical panoramas, simulator own pose, a visited-node table, and no
map image. The shared runtime also supplies a short pose history and an
advisory forward-depth warning. Node records are not a rendered map.

## Loop

1. Freeze 50 R2R **TRAIN** tasks and the v0 code. Run the whole set and retain
   the original requests, responses, panoramas, primitive actions, measured
   motion and final outcomes.
2. Give the RSI author every result and a chronological decision index, with
   paths to the complete raw artifacts. The author can inspect images and
   failure points on demand; the index is not a truncated replacement.
3. In **one** author response, generate exactly three independent mechanisms
   and three code patches against the same parent. Proposals cannot use one
   another's results.
4. Independently compile and run each patch from the original start on the
   same 50 tasks. Count repaired failures and regressed successes, and report
   decisions, SPL and provider usage. Do not combine candidates.
5. Promote a development winner only when repairs exceed regressions. Repeat
   on that winner's new errors, up to five rounds. A development winner is **not
   an admitted Skill**: a separate verifier and scene-separated evaluation
   must pass before adding it to a Skill Library or claiming generalization.

Run a dry registration without model or simulator calls:

```bash
python rsi/campaign.py \
  --dataset /path/to/StreamVLN/data/datasets/r2r/train/train.json.gz \
  --output /path/to/rsi-run --prepare-only
```

Run the development loop with the Habitat Python environment and a configured
Codex CLI. `CODEX_HOME` and `VLN_WORKSPACE` may be set explicitly. The output
directory is resumable and contains the registration, exact code copies,
whole-task results, all three proposals per round and paired comparisons.

```bash
python rsi/campaign.py \
  --dataset /path/to/StreamVLN/data/datasets/r2r/train/train.json.gz \
  --output /path/to/rsi-run --gpu 0
```

The earlier random100 archives are `val_unseen` development diagnostics once
inspected. Their ten selected failures are not an independent test set. The
archived no-map Astra baseline and one RSI-authored landmark-discrimination
candidate were evaluated separately; those runs are not part of the five-round
TRAIN campaign.
