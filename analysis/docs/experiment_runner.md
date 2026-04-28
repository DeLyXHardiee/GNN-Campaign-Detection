# Experiment Runner (Two-Phase)

Use the **only** supported entrypoint for the full flow (anchor → seed → candidate → seed–candidate pairgraph, then scoring + community per target):

`python analysis/pipelines/run_experiment.py --config analysis/configs/experiments/<name>.json`

For a full build and evaluation in one go, set `experiment.mode` to `setup_and_score` (see e.g. [`analysis/configs/experiments/exp03.seedcand.setupscore.pu.json`](../configs/experiments/exp03.seedcand.setupscore.pu.json)) and tune `setup.enable` / `selection.score_targets` as needed.

## Pipeline model

The system runs in two explicit phases:

1. **Graph setup phase** (repo-local graph artifacts)
2. **Scoring/community phase** (separate scoring artifact root)

Execution routing is registry-driven:

- target registry: target -> edge resolver + community executor
- scorer registry metadata: score_mode -> param-key resolution + compatibility
- runner composition modules: `runner_config`, `runner_targets`, `runner_manifest`

Run mode is controlled by `experiment.mode`:

- `setup_only`
- `score_only`
- `setup_and_score`

## Required config shape

Top-level keys:

- `experiment` (must include `graph_id` and `scoring_run_id`)
- `artifacts`
- `setup`
- `selection`
- `scoring`
- `community`

Stage JSON under `analysis/configs/` may include `run.graph_id` (e.g. `local_default`) for **direct** Python calls to stage helpers with those files. **`run_experiment`** always injects `run.graph_id` from **`experiment.graph_id`** via `run_graph_setup` before any stage runs.

## Artifact boundaries

Graph bundle root:

- `artifacts.graph_bundle_root/<graph_id>/`
- contains only needed component dirs (`anchor/`, `seed/`, `candidate/`, `seed_candidate/`, `semantic_shard/`, `pair_training/`) for that run

Scoring bundle root:

- `artifacts.scoring_output_root/<scoring_run_id>/`
- contains per target output (`<target>/community/...`) plus `run_manifest.json`

Community output filenames are unchanged:

- `anchor_community_multi_gt_summary.json`
- `anchor_community_sweep__{gt_slug}.csv`
- `anchor_community_best__{gt_slug}.json`

Run manifest v2 target entries now include normalized sections:

- `inputs` (resolved edge/scoring inputs)
- `artifacts` (output dir + summary)
- `metrics` (selection metric metadata)
- `community_result` (target-specific details)

## Setup behavior

`setup.enable` toggles each component:

- `anchor`
- `seed`
- `candidate`
- `seed_candidate`
- `semantic_shard`
- `pair_training`

`setup.policy` controls reuse behavior:

- `on_missing`: `build` or `fail`
- `on_present`: `reuse` or `rebuild`

`pair_training` is auto-created in setup when enabled and prerequisites are present.

## Dry run

With `--dry-run`, the runner does not execute graph setup or community sweeps. It also does **not** require the graph bundle directory to exist, including for `experiment.mode=score_only` (planned paths are used for manifests and injected community config).

## Scoring targets

Select graph surfaces via `selection.score_targets`, for example:

- `anchor`
- `seed`
- `candidate`
- `seed_candidate`

Supported scorer target mappings:

- `seed_candidate_handcrafted_v1` -> `seed_candidate`
- `seed_candidate_pu_v1` -> `seed_candidate`
- `semantic_shard_handcrafted_v1` -> `semantic_shard`
- `semantic_shard_affine_v1` -> `semantic_shard`

Semantic shard mode semantics:

- `score_mode: none` => explicit unweighted topology evaluation (`edge_weight=1.0` in sweep input)
- weighted shard score modes (`semantic_shard_*`) => preserve weighted-edge semantics from setup/scorer output
- semantic shard best-setting selection is ranked by `v_measure`

## Quickstart

- PU end-to-end:
  - `python analysis/pipelines/run_experiment.py --config analysis/configs/experiments/exp03.seedcand.setupscore.pu.json`
- Unweighted setup+score:
  - `python analysis/pipelines/run_experiment.py --config analysis/configs/experiments/exp01.seedcand.setupscore.unweighted.json`
- Handcrafted score-only on existing bundle:
  - `python analysis/pipelines/run_experiment.py --config analysis/configs/experiments/exp02.seedcand.scoreonly.handcrafted_on_base.json`
- Semantic shard unweighted setup+score:
  - `python analysis/pipelines/run_experiment.py --config analysis/configs/experiments/exp07.shard.setupscore.unweighted.json`
- Semantic shard notebook-parity weighted setup+score:
  - `python analysis/pipelines/run_experiment.py --config analysis/configs/experiments/exp08.shard.setupscore.weighted_notebook.json`

## Useful CLI overrides

- `--scoring-run-id`
- `--run-mode` (`setup_only|score_only|setup_and_score`)
- `--mode-override` (scoring mode)
- `--graph-id` (override `experiment.graph_id`)
- `--gt-set`

## Breaking changes and migration notes

- `run_manifest.json` target rows changed shape (now normalized with `inputs/artifacts/metrics`).
- `run_experiment(...)` also returns `community_results_legacy` as a compatibility view.
- Dry-run payload for semantic shard no longer embeds anchor-style `community_config`.
- Config validation is stricter for semantic shard nested setup fields (`step1`, `step2`, channel scoring/switch objects).
