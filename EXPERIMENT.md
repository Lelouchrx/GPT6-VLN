# GPT6-VLN R2R evaluation snapshot

This repository contains the source used for the 100-episode R2R `val_unseen`
evaluation started on 2026-10-28. The pinned source files in the original batch
match this checkout. Scene assets, model credentials, generated images, and run
logs are not included.

The batch runner is in `experiments/random20_astra_uniform_20261028_161233/`;
the separate retry runner for episodes 180 and 1737 is in its
`retry_180_1737/` subdirectory. Episode registration files and batch manifests
are intentionally excluded. Batch runners expect a private batch directory
containing a `project/` copy of this source tree and a task manifest.

Configuration: Codex provider with `gpt-6-astra`, R2R `val_unseen`, GPU 0,
100 unique episodes (20 original random samples plus 80 additional samples
without replacement), seed `20261028`, up to eight uniformly sampled historical
four-view panoramas, action horizon `4_to_3`, 400-primitive budget, and a route
review every 10 completed decisions.

The original batch completed 98 episodes: 85 successes, 13 navigation failures,
and two operational errors. Both error episodes were rerun successfully, giving
87 successes out of 100 completed episodes (87% success rate). The mean SPL
across the 100 completed episodes is 0.7648684108. These combined metrics are
derived from the original and retry outputs; the original batch result files
were not rewritten.

To run one episode after installing the required Habitat and R2R dependencies,
provide your own task registration and use `./run.sh --registration
path/to/tasks.json --episode EPISODE_ID --provider codex --decisions 0
--chunk-end 400 --action-horizon 4_to_3 --gpu 0`. The environment paths in
`env.sh` may need adjustment for a different server.
